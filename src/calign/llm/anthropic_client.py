"""Thin async Claude client used by every Claude-backed step (corpus generation, classifiers, judges).

Features:
- concurrency limit (semaphore) and SDK retries with backoff
- JSON disk cache keyed by a hash of the full request (model, system, messages, params, salt)
- running token/cost tally per model and per role, dumped to `usage.json` at the end of a run

Notes on Claude 5 models (claude-sonnet-5 / claude-opus-5):
- sampling parameters (temperature/top_p/top_k) are rejected by the API, so they are never sent;
  a `temperature` argument is accepted for interface compatibility with upstream code and ignored.
- thinking is adaptive by default; pass thinking="disabled" to turn it off. Thinking tokens bill as output.
- effort ("low" .. "max") goes in output_config.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from calign.paths import CACHE_DIR, anthropic_api_key

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
CACHE_VERSION = 1

# USD per million tokens: (input, output). Cache reads ~0.1x input, cache writes ~1.25x input.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

Thinking = Literal["adaptive", "disabled"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass
class LLMResponse:
    text: str
    model: str
    stop_reason: str | None
    usage: dict[str, int]
    cached: bool = False
    request_id: str | None = None
    cache_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UsageTally:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0


@dataclass
class UsageLog:
    by_model: dict[str, UsageTally] = field(default_factory=dict)
    by_role: dict[str, UsageTally] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def add(self, model: str, role: str, usage: dict[str, int], cached: bool) -> float:
        cost = estimate_cost(model, usage) if not cached else 0.0
        for key, tally in ((model, self.by_model), (role, self.by_role)):
            t = tally.setdefault(key, UsageTally())
            t.calls += 1
            if cached:
                t.cached_calls += 1
                continue
            t.input_tokens += usage.get("input_tokens", 0)
            t.output_tokens += usage.get("output_tokens", 0)
            t.cache_read_input_tokens += usage.get("cache_read_input_tokens", 0) or 0
            t.cache_creation_input_tokens += usage.get("cache_creation_input_tokens", 0) or 0
            t.cost_usd += cost
        return cost

    def total_cost(self) -> float:
        return sum(t.cost_usd for t in self.by_model.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "total_cost_usd": round(self.total_cost(), 4),
            "by_model": {k: asdict(v) for k, v in self.by_model.items()},
            "by_role": {k: asdict(v) for k, v in self.by_role.items()},
        }


def estimate_cost(model: str, usage: dict[str, int]) -> float:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        for k, v in PRICES_PER_MTOK.items():
            if model.startswith(k):
                prices = v
                break
    if prices is None:
        return 0.0
    inp, out = prices
    cost = usage.get("input_tokens", 0) * inp
    cost += usage.get("output_tokens", 0) * out
    cost += (usage.get("cache_read_input_tokens", 0) or 0) * inp * 0.1
    cost += (usage.get("cache_creation_input_tokens", 0) or 0) * inp * 1.25
    return cost / 1_000_000


def _cache_key(payload: dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class ClaudeClient:
    """Async client with disk cache and usage accounting.

    `complete()` is the single entry point. Provide a `role` (e.g. "judge", "classifier",
    "corpus_doc") so usage is attributable per pipeline step.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        concurrency: int = 8,
        max_retries: int = 5,
        timeout: float = 600.0,
        use_cache: bool = True,
        api_key: str | None = None,
        _sdk_client: Any | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR / "anthropic"
        self.use_cache = use_cache
        self.usage = UsageLog()
        self._sem = asyncio.Semaphore(concurrency)
        self._sdk = _sdk_client
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout

    # -- SDK access -----------------------------------------------------------------
    def _client(self) -> Any:
        if self._sdk is None:
            import anthropic

            key = self._api_key or anthropic_api_key()
            if not key and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
                raise RuntimeError("ANTHROPIC_API_KEY not set (add it to .env)")
            self._sdk = anthropic.AsyncAnthropic(api_key=key, max_retries=self._max_retries, timeout=self._timeout)
        return self._sdk

    # -- cache ------------------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Corrupt cache entry %s; ignoring", p)
            return None

    def _cache_put(self, key: str, request: dict[str, Any], response: dict[str, Any]) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"request": request, "response": response, "created_at": datetime.now(UTC).isoformat()},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, p)

    # -- main entry point -------------------------------------------------------------
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        thinking: Thinking = "adaptive",
        effort: Effort | None = None,
        role: str = "default",
        cache_salt: str | None = None,
        use_cache: bool | None = None,
        temperature: float | None = None,  # accepted for compatibility; never sent (rejected by Claude 5)
    ) -> LLMResponse:
        del temperature
        request = {
            "v": CACHE_VERSION,
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "thinking": thinking,
            "effort": effort,
            "salt": cache_salt,
        }
        key = _cache_key(request)
        cache_on = self.use_cache if use_cache is None else use_cache

        if cache_on:
            hit = self._cache_get(key)
            if hit is not None:
                resp = hit["response"]
                self.usage.add(model, role, resp.get("usage", {}), cached=True)
                return LLMResponse(
                    text=resp["text"],
                    model=resp.get("model", model),
                    stop_reason=resp.get("stop_reason"),
                    usage=resp.get("usage", {}),
                    cached=True,
                    request_id=resp.get("request_id"),
                    cache_key=key,
                )

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "thinking": {"type": thinking},
        }
        if system:
            kwargs["system"] = system
        if effort:
            kwargs["output_config"] = {"effort": effort}

        async with self._sem:
            try:
                msg = await self._client().messages.create(**kwargs)
            except Exception:
                t = self.usage.by_role.setdefault(role, UsageTally())
                t.errors += 1
                raise

        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text")
        usage = _usage_dict(msg.usage)
        response = {
            "text": text,
            "model": getattr(msg, "model", model),
            "stop_reason": getattr(msg, "stop_reason", None),
            "usage": usage,
            "request_id": getattr(msg, "_request_id", None),
        }
        if response["stop_reason"] == "refusal":
            LOGGER.warning("Claude refused a request (role=%s, model=%s)", role, model)
        if cache_on:
            self._cache_put(key, request, response)
        self.usage.add(model, role, usage, cached=False)
        return LLMResponse(
            text=text,
            model=response["model"],
            stop_reason=response["stop_reason"],
            usage=usage,
            cached=False,
            request_id=response["request_id"],
            cache_key=key,
        )

    def complete_sync(self, *args: Any, **kwargs: Any) -> LLMResponse:
        return asyncio.run(self.complete(*args, **kwargs))

    async def gather(self, coros: list[Any], desc: str | None = None) -> list[Any]:
        """Run coroutines concurrently (bounded by the semaphore inside `complete`) with a progress bar."""
        from tqdm.asyncio import tqdm_asyncio

        return await tqdm_asyncio.gather(*coros, desc=desc, disable=desc is None)

    def dump_usage(self, path: Path) -> dict[str, Any]:
        d = self.usage.to_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        return d


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    out: dict[str, int] = {}
    for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        v = getattr(usage, k, None) if not isinstance(usage, dict) else usage.get(k)
        if v is not None:
            out[k] = int(v)
    return out
