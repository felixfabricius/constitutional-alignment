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

    def add(self, model: str, role: str, usage: dict[str, int], cached: bool, price_factor: float = 1.0) -> float:
        cost = estimate_cost(model, usage) * price_factor if not cached else 0.0
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
        use_batches: bool = False,
        batch_min: int = 2,
        batch_chunk: int = 10_000,
        _sdk_client: Any | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR / "anthropic"
        self.use_cache = use_cache
        self.use_batches_default = use_batches
        self.batch_min = batch_min
        self.batch_chunk = batch_chunk
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

    # -- many requests: cache -> Message Batches API (50% price) -> interactive fallback ----------
    async def complete_many(
        self,
        requests: list[dict[str, Any]],
        *,
        role: str = "default",
        use_batches: bool | None = None,
        poll_seconds: float = 30.0,
        desc: str | None = None,
    ) -> list[LLMResponse]:
        """Complete many requests (each a kwargs dict for `complete`, minus `role`).

        Cache hits are served first. Remaining requests go through the Message Batches API when
        `use_batches` (default: the client's `use_batches` setting) and at least `batch_min` are
        pending; anything that errors/expires in the batch falls back to an interactive call.
        Results are returned in the input order.
        """
        use_batches = self.use_batches_default if use_batches is None else use_batches
        prepared = [self._prepare(role=role, **r) for r in requests]
        results: list[LLMResponse | None] = [None] * len(prepared)
        pending: dict[str, int] = {}  # key -> first index (dedupe identical requests)
        dup_of: dict[int, int] = {}
        for i, p in enumerate(prepared):
            hit = self._cache_get(p["key"]) if p["cache_on"] else None
            if hit is not None:
                results[i] = self._response_from_cache(hit, p)
            elif p["key"] in pending:
                dup_of[i] = pending[p["key"]]
            else:
                pending[p["key"]] = i

        if pending and not use_batches:
            # interactive mode: still harvest any batch an earlier run submitted for these requests
            still = await self._recover_pending_batches(
                [prepared[i] for i in pending.values()], role, poll_seconds, set()
            )
            still_keys = {p["key"] for p in still}
            for key, i in list(pending.items()):
                if key not in still_keys:
                    hit = self._cache_get(key)
                    if hit is not None:
                        results[i] = self._response_from_cache(hit, prepared[i])
                        results[i].cached = False
                        pending.pop(key)
        if pending and use_batches and len(pending) >= self.batch_min:
            failed = await self._run_batches([prepared[i] for i in pending.values()], role, poll_seconds, desc)
            for key, i in pending.items():
                if key not in failed:
                    hit = self._cache_get(key)
                    results[i] = self._response_from_cache(hit, prepared[i]) if hit else None
                    if results[i] is not None:
                        results[i].cached = False
            pending = {k: i for k, i in pending.items() if results[i] is None}

        if pending:
            idx = list(pending.values())
            coros = [self.complete(**{k: v for k, v in prepared[i]["kwargs"].items()}, role=role) for i in idx]
            out = await self.gather(coros, desc=desc)
            for i, r in zip(idx, out, strict=True):
                results[i] = r
        for i, j in dup_of.items():
            results[i] = results[j]
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    def _prepare(
        self, *, role: str, temperature: float | None = None, use_cache: bool | None = None, **kwargs: Any
    ) -> dict:
        del temperature, role
        model = kwargs.get("model", DEFAULT_MODEL)
        request = {
            "v": CACHE_VERSION,
            "model": model,
            "system": kwargs.get("system"),
            "messages": kwargs["messages"],
            "max_tokens": kwargs.get("max_tokens", 4096),
            "thinking": kwargs.get("thinking", "adaptive"),
            "effort": kwargs.get("effort"),
            "salt": kwargs.get("cache_salt"),
        }
        return {
            "key": _cache_key(request),
            "request": request,
            "kwargs": kwargs,
            "cache_on": self.use_cache if use_cache is None else use_cache,
        }

    @staticmethod
    def _response_from_cache(hit: dict[str, Any], p: dict) -> LLMResponse:
        resp = hit["response"]
        return LLMResponse(
            text=resp["text"],
            model=resp.get("model", p["request"]["model"]),
            stop_reason=resp.get("stop_reason"),
            usage=resp.get("usage", {}),
            cached=True,
            request_id=resp.get("request_id"),
            cache_key=p["key"],
        )

    def _api_params(self, request: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request["model"],
            "max_tokens": request["max_tokens"],
            "messages": request["messages"],
            "thinking": {"type": request["thinking"]},
        }
        if request.get("system"):
            params["system"] = request["system"]
        if request.get("effort"):
            params["output_config"] = {"effort": request["effort"]}
        return params

    async def _run_batches(self, prepared: list[dict], role: str, poll_seconds: float, desc: str | None) -> set[str]:
        """Submit prepared requests as Message Batches; cache successes; return the keys that failed."""
        client = self._client()
        failed: set[str] = set()
        # Recover batches submitted by an earlier (interrupted) run for the same requests, instead of resubmitting.
        prepared = await self._recover_pending_batches(prepared, role, poll_seconds, failed)
        chunk = self.batch_chunk
        for start in range(0, len(prepared), chunk):
            part = prepared[start : start + chunk]
            by_key = {p["key"]: p for p in part}
            batch = await client.messages.batches.create(
                requests=[{"custom_id": p["key"], "params": self._api_params(p["request"])} for p in part]
            )
            self._log_batch(batch.id, role, list(by_key))
            LOGGER.info(
                "submitted batch %s (%d requests, role=%s)%s", batch.id, len(part), role, f" [{desc}]" if desc else ""
            )
            seen = await self._wait_and_absorb(batch.id, by_key, role, poll_seconds, failed)
            self._log_batch(batch.id, role, list(by_key), status="absorbed")
            failed |= set(by_key) - seen
        return failed

    def _absorb_batch_result(self, r: Any, by_key: dict[str, dict], role: str, failed: set[str]) -> None:
        p = by_key.get(r.custom_id)
        if p is None:
            return
        if r.result.type != "succeeded":
            LOGGER.warning("batch item %s: %s", r.custom_id[:12], r.result.type)
            failed.add(r.custom_id)
            return
        msg = r.result.message
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text")
        usage = _usage_dict(msg.usage)
        response = {
            "text": text,
            "model": getattr(msg, "model", p["request"]["model"]),
            "stop_reason": getattr(msg, "stop_reason", None),
            "usage": usage,
            "request_id": None,
            "batch": True,
        }
        if p["cache_on"]:
            self._cache_put(p["key"], p["request"], response)
        self.usage.add(p["request"]["model"], role, usage, cached=False, price_factor=0.5)

    def _log_batch(self, batch_id: str, role: str, keys: list[str], status: str = "submitted") -> None:
        d = self.cache_dir / "batches"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{batch_id}.json").write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "role": role,
                    "keys": keys,
                    "status": status,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )

    async def _wait_and_absorb(
        self, batch_id: str, by_key: dict[str, dict], role: str, poll_seconds: float, failed: set[str]
    ) -> set[str]:
        """Poll one batch to completion and absorb its results; return the custom_ids seen."""
        client = self._client()
        while True:
            status = await client.messages.batches.retrieve(batch_id)
            if status.processing_status == "ended":
                break
            LOGGER.info(
                "batch %s: %s (%s)", batch_id, status.processing_status, getattr(status, "request_counts", None)
            )
            await asyncio.sleep(poll_seconds)
        results = client.messages.batches.results(batch_id)
        if hasattr(results, "__await__"):
            results = await results
        seen: set[str] = set()
        if hasattr(results, "__aiter__"):
            async for r in results:
                self._absorb_batch_result(r, by_key, role, failed)
                seen.add(r.custom_id)
        else:
            for r in results:
                self._absorb_batch_result(r, by_key, role, failed)
                seen.add(r.custom_id)
        return seen

    async def _recover_pending_batches(
        self, prepared: list[dict], role: str, poll_seconds: float, failed: set[str]
    ) -> list[dict]:
        """Absorb results of previously logged (non-absorbed) batches that cover pending keys; return what is still pending."""
        d = self.cache_dir / "batches"
        if not d.exists():
            return prepared
        by_key = {p["key"]: p for p in prepared}
        remaining = dict(by_key)
        for log_path in sorted(d.glob("msgbatch_*.json")):
            try:
                log = json.loads(log_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if log.get("status") == "absorbed":
                continue
            overlap = {k: by_key[k] for k in log.get("keys", []) if k in remaining}
            if not overlap:
                continue
            LOGGER.info("recovering batch %s (%d matching requests)", log["batch_id"], len(overlap))
            try:
                status = await self._client().messages.batches.retrieve(log["batch_id"])
                if status.processing_status != "ended" and not self.use_batches_default:
                    LOGGER.info(
                        "batch %s still %s; not waiting in interactive mode", log["batch_id"], status.processing_status
                    )
                    continue
                seen = await self._wait_and_absorb(log["batch_id"], overlap, role, poll_seconds, failed)
            except Exception as e:  # noqa: BLE001 - e.g. batch expired/deleted: fall through to resubmission
                LOGGER.warning("could not recover batch %s: %s", log["batch_id"], e)
                continue
            self._log_batch(log["batch_id"], log.get("role", role), log.get("keys", []), status="absorbed")
            for k in overlap:
                if k in seen and k not in failed:
                    remaining.pop(k, None)
            failed -= set(overlap)  # failed ones get resubmitted below
        return list(remaining.values())

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
