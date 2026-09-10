import asyncio
from types import SimpleNamespace

import pytest

from calign.llm.anthropic_client import ClaudeClient, estimate_cost


class FakeMessages:
    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        n = len(self.calls)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"reply-{n}")],
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
            model=kwargs["model"],
            stop_reason="end_turn",
            _request_id=f"req_{n}",
        )


class FakeSDK:
    def __init__(self):
        self.messages = FakeMessages()


@pytest.fixture
def client(tmp_path):
    sdk = FakeSDK()
    c = ClaudeClient(cache_dir=tmp_path / "cache", _sdk_client=sdk)
    return c, sdk


def test_cache_hit_and_miss(client):
    c, sdk = client
    msgs = [{"role": "user", "content": "hello"}]
    r1 = asyncio.run(c.complete(msgs, system="sys", role="judge"))
    r2 = asyncio.run(c.complete(msgs, system="sys", role="judge"))
    assert r1.text == "reply-1" and not r1.cached
    assert r2.text == "reply-1" and r2.cached
    assert len(sdk.messages.calls) == 1
    # different salt -> new call
    r3 = asyncio.run(c.complete(msgs, system="sys", role="judge", cache_salt="s2"))
    assert r3.text == "reply-2" and len(sdk.messages.calls) == 2
    # different effort -> new call
    asyncio.run(c.complete(msgs, system="sys", role="judge", effort="low"))
    assert len(sdk.messages.calls) == 3


def test_request_shape_never_sends_temperature(client):
    c, sdk = client
    asyncio.run(c.complete([{"role": "user", "content": "x"}], temperature=0.0, thinking="disabled", effort="medium"))
    call = sdk.messages.calls[0]
    assert "temperature" not in call
    assert call["thinking"] == {"type": "disabled"}
    assert call["output_config"] == {"effort": "medium"}
    assert call["model"] == "claude-sonnet-5"


def test_usage_accounting(client, tmp_path):
    c, _ = client
    asyncio.run(c.complete([{"role": "user", "content": "x"}], role="classifier"))
    asyncio.run(c.complete([{"role": "user", "content": "x"}], role="classifier"))  # cached
    d = c.dump_usage(tmp_path / "usage.json")
    t = d["by_role"]["classifier"]
    assert t["calls"] == 2 and t["cached_calls"] == 1
    assert t["input_tokens"] == 100 and t["output_tokens"] == 50
    expected = 100 * 2.0 / 1e6 + 50 * 10.0 / 1e6
    assert abs(t["cost_usd"] - expected) < 1e-12
    assert abs(d["total_cost_usd"] - round(expected, 4)) < 1e-9
    assert (tmp_path / "usage.json").exists()


def test_estimate_cost_prefix_match():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert estimate_cost("claude-sonnet-5", usage) == 2.0
    assert estimate_cost("claude-sonnet-5-20260101", usage) == 2.0
    assert estimate_cost("unknown-model", usage) == 0.0
    assert estimate_cost("claude-opus-5", {"cache_read_input_tokens": 1_000_000}) == pytest.approx(0.5)
