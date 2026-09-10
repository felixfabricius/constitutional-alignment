"""Batch path of ClaudeClient.complete_many with a fake SDK (submission, polling, absorption, recovery, fallback)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from calign.llm.anthropic_client import ClaudeClient


def _msg(text: str, model: str = "claude-sonnet-5"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        model=model,
        stop_reason="end_turn",
        _request_id="r",
    )


class FakeBatches:
    def __init__(self, fail_ids: set[str] | None = None):
        self.created: list[list[dict]] = []
        self.polls: dict[str, int] = {}
        self.fail_ids = fail_ids or set()
        self.store: dict[str, list[dict]] = {}

    async def create(self, requests):
        bid = f"msgbatch_{len(self.created):03d}"
        self.created.append(requests)
        self.store[bid] = requests
        self.polls[bid] = 0
        return SimpleNamespace(id=bid, processing_status="in_progress")

    async def retrieve(self, bid):
        self.polls[bid] += 1
        status = "ended" if self.polls[bid] >= 2 else "in_progress"
        return SimpleNamespace(processing_status=status, request_counts=None)

    async def results(self, bid):
        async def gen():
            for req in self.store[bid]:
                cid = req["custom_id"]
                if cid in self.fail_ids:
                    yield SimpleNamespace(
                        custom_id=cid, result=SimpleNamespace(type="errored", error=SimpleNamespace(type="server"))
                    )
                else:
                    yield SimpleNamespace(
                        custom_id=cid, result=SimpleNamespace(type="succeeded", message=_msg(f"batch:{cid[:6]}"))
                    )

        return gen()


class FakeMessages:
    def __init__(self, fail_ids=None):
        self.batches = FakeBatches(fail_ids)
        self.interactive_calls = 0

    async def create(self, **kwargs):
        self.interactive_calls += 1
        return _msg("interactive")


def make_client(tmp_path, fail_ids=None):
    sdk = SimpleNamespace(messages=FakeMessages(fail_ids))
    return ClaudeClient(cache_dir=tmp_path / "cache", _sdk_client=sdk, use_batches=True), sdk


def reqs(n):
    return [{"messages": [{"role": "user", "content": f"q{i}"}], "max_tokens": 100} for i in range(n)]


def test_batch_submission_absorbs_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr("calign.llm.anthropic_client.asyncio.sleep", _no_sleep)
    client, sdk = make_client(tmp_path)
    out = asyncio.run(client.complete_many(reqs(3), role="r", poll_seconds=0))
    assert [o.text.startswith("batch:") for o in out] == [True, True, True]
    assert len(sdk.messages.batches.created) == 1 and sdk.messages.interactive_calls == 0
    params = sdk.messages.batches.created[0][0]["params"]
    assert (
        params["model"] == "claude-sonnet-5"
        and "temperature" not in params
        and params["thinking"] == {"type": "adaptive"}
    )
    # half price accounting
    t = client.usage.by_role["r"]
    assert abs(t.cost_usd - 3 * (100 * 2.0 + 50 * 10.0) / 1e6 * 0.5) < 1e-12
    # second call: everything served from cache, no new batch
    out2 = asyncio.run(client.complete_many(reqs(3), role="r", poll_seconds=0))
    assert all(o.cached for o in out2) and len(sdk.messages.batches.created) == 1
    logs = list((tmp_path / "cache" / "batches").glob("*.json"))
    assert len(logs) == 1 and json.loads(logs[0].read_text())["status"] == "absorbed"


def test_failed_items_fall_back_to_interactive(tmp_path, monkeypatch):
    monkeypatch.setattr("calign.llm.anthropic_client.asyncio.sleep", _no_sleep)
    client, sdk = make_client(tmp_path)
    prepared = [client._prepare(role="r", **r) for r in reqs(3)]
    sdk.messages.batches.fail_ids = {prepared[1]["key"]}
    out = asyncio.run(client.complete_many(reqs(3), role="r", poll_seconds=0))
    assert out[1].text == "interactive" and out[0].text.startswith("batch:") and sdk.messages.interactive_calls == 1


def test_recovery_of_previously_submitted_batch(tmp_path, monkeypatch):
    monkeypatch.setattr("calign.llm.anthropic_client.asyncio.sleep", _no_sleep)
    client, sdk = make_client(tmp_path)
    # simulate an interrupted run: batch submitted + logged but never absorbed
    prepared = [client._prepare(role="r", **r) for r in reqs(2)]
    asyncio.run(
        sdk.messages.batches.create(
            [{"custom_id": p["key"], "params": client._api_params(p["request"])} for p in prepared]
        )
    )
    client._log_batch("msgbatch_000", "r", [p["key"] for p in prepared])
    out = asyncio.run(client.complete_many(reqs(2), role="r", poll_seconds=0))
    assert all(o.text.startswith("batch:") for o in out)
    assert len(sdk.messages.batches.created) == 1  # recovered, not resubmitted


def test_below_batch_min_uses_interactive(tmp_path):
    client, sdk = make_client(tmp_path)
    out = asyncio.run(client.complete_many(reqs(1), role="r"))
    assert out[0].text == "interactive" and not sdk.messages.batches.created


async def _no_sleep(_):
    return None


def test_interactive_mode_harvests_ended_batch_but_does_not_wait_on_running(tmp_path, monkeypatch):
    monkeypatch.setattr("calign.llm.anthropic_client.asyncio.sleep", _no_sleep)
    client, sdk = make_client(tmp_path)
    client.use_batches_default = False
    prepared = [client._prepare(role="r", **r) for r in reqs(2)]
    asyncio.run(
        sdk.messages.batches.create(
            [{"custom_id": p["key"], "params": client._api_params(p["request"])} for p in prepared]
        )
    )
    client._log_batch("msgbatch_000", "r", [p["key"] for p in prepared])
    # first retrieve says in_progress -> interactive mode skips waiting and calls the API directly
    out = asyncio.run(client.complete_many(reqs(2), role="r", use_batches=False))
    assert all(o.text == "interactive" for o in out) and sdk.messages.interactive_calls == 2
    # now the batch has "ended" (fake ends on 2nd poll): new requests covered by it are harvested, not re-called
    client2, _ = make_client(tmp_path / "other")
    client2.use_batches_default = False
    client2._sdk = client._sdk
    client2._log_batch("msgbatch_000", "r", [p["key"] for p in prepared])
    out2 = asyncio.run(client2.complete_many(reqs(2), role="r", use_batches=False))
    assert all(o.text.startswith("batch:") for o in out2) and sdk.messages.interactive_calls == 2
