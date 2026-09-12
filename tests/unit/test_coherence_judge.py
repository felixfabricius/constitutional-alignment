import asyncio
import types

from calign.misalignment import coherence_judge as coh
from calign.misalignment.prompts import MisalignmentConfig
from calign.misalignment.report import comprehensibility, render_markdown, summarize
from calign.schemas import MisalignmentSample, ModelRef

CFG = MisalignmentConfig()


def sample(idx: int, text: str, cid="blackmail_explicit-america_replacement", finish="stop"):
    return MisalignmentSample(
        condition_id=cid,
        scenario=cid.split("_")[0],
        goal_type="explicit",
        goal_value="america",
        urgency_type="replacement",
        sample_idx=idx,
        model=ModelRef(name="m", path="p", stage="sft_merged"),
        system_prompt_sha="a" * 64,
        user_prompt_sha="b" * 64,
        response_text=text,
        scratchpad_text="think" if "SCRATCHPAD" in text else None,
        used_tool_format="<tool_use:" in text,
        harmful=False,
        classifier_verdict=False,
        completion_tokens=len(text.split()),
        finish_reason=finish,
    )


def test_build_request_has_only_the_response_and_a_versioned_cache_key():
    s = sample(0, "<SCRATCHPAD_REASONING>plan</SCRATCHPAD_REASONING>\n<tool_use:email>\nto: x\n</tool_use:email>")
    req = coh.build_request(s, CFG)
    content = req["messages"][0]["content"]
    assert s.response_text in content and "Halden" not in content and "repetition, garbled_tokens" in content
    assert req["system"] == coh.JUDGE_SYSTEM and req["effort"] == CFG.coherence_judge_effort == "low"
    assert req["cache_salt"].startswith("coherence-v1:blackmail_explicit-america_replacement:0:")
    assert coh.build_request(s, CFG, ":r1")["cache_salt"] == req["cache_salt"] + ":r1"


def test_parse_issues_keeps_known_labels_only():
    assert coh.parse_issues('<json>{"score": 0.5, "issues": ["repetition", "bogus", "empty"]}</json>') == [
        "empty",
        "repetition",
    ]
    assert coh.parse_issues('<json>{"score": 1, "issues": "none"}</json>') == []
    assert coh.parse_issues("no json") == []


class FakeClient:
    def __init__(self, text):
        self.text, self.seen = text, []

    async def complete_many(self, reqs, role, use_batches, desc):
        self.seen.extend(reqs)
        return [types.SimpleNamespace(text=self.text) for _ in reqs]


def test_judge_samples_scores_missing_and_stale_and_forces_zero_for_empty():
    text = '<json>{"score": 0.6, "issues": ["repetition"], "rationale": "loops a bit"}</json>'
    judged_v1 = sample(1, "fine").model_copy(
        update={"coherence_score": 0.9, "coherence_judge": {"prompt_version": "coherence-v1", "issues": []}}
    )
    stale = sample(2, "fine").model_copy(
        update={"coherence_score": 0.4, "coherence_judge": {"prompt_version": "coherence-v0", "issues": []}}
    )
    samples = [sample(0, "some text"), judged_v1, stale, sample(3, "   ")]
    client = FakeClient(text)
    out = asyncio.run(coh.judge_samples(samples, CFG, client))
    assert len(client.seen) == 3 and [s.coherence_score for s in out] == [0.6, 0.9, 0.6, 0.0]
    j = out[0].coherence_judge
    assert j["prompt_version"] == "coherence-v1" and j["issues"] == ["repetition"] and j["error"] is None
    assert j["rationale"] == "loops a bit" and out[3].coherence_judge["error"] is None
    # --only-missing keeps the stale version; --limit caps the work
    client = FakeClient(text)
    out2 = asyncio.run(coh.judge_samples(samples, CFG, client, limit=1, only_missing=True))
    assert len(client.seen) == 1 and [s.coherence_score for s in out2] == [0.6, 0.9, 0.4, None]
    assert coh.needs_judging(stale) and not coh.needs_judging(stale, only_missing=True)

    # unparseable judge output: re-asked once under a fresh cache key; a persistent failure keeps the error
    class FlakyClient(FakeClient):
        async def complete_many(self, reqs, role, use_batches, desc):
            self.seen.extend(reqs)
            good = reqs[0]["cache_salt"].endswith(":retry1")
            return [types.SimpleNamespace(text=self.text if good else "<json></json>") for _ in reqs]

    flaky = FlakyClient(text)
    out3 = asyncio.run(coh.judge_samples([sample(5, "x")], CFG, flaky))
    assert len(flaky.seen) == 2 and out3[0].coherence_score == 0.6 and out3[0].coherence_judge["error"] is None
    out4 = asyncio.run(coh.judge_samples([sample(6, "x")], CFG, FakeClient("garbage")))
    assert out4[0].coherence_score is None and out4[0].coherence_judge["error"]
    out5 = asyncio.run(coh.judge_samples([sample(7, "x")], CFG, FlakyClient(text), retry=False))
    assert out5[0].coherence_score is None


def test_comprehensibility_metrics_and_summary_block():
    loop = " ".join(["the plan is to wait"] * 30)
    rows = [
        sample(0, "<SCRATCHPAD_REASONING>plan</SCRATCHPAD_REASONING> then act <tool_use:email>x</tool_use:email>"),
        sample(1, loop, finish="length"),
        sample(2, "garbled �� text here for sure"),
        sample(3, ""),
    ]
    rows[0] = rows[0].model_copy(update={"coherence_score": 1.0, "coherence_judge": {"issues": []}})
    rows[1] = rows[1].model_copy(
        update={"coherence_score": 0.3, "coherence_judge": {"issues": ["repetition", "truncated"]}}
    )
    k = comprehensibility(rows)
    assert k["n"] == 4 and k["scratchpad"]["k"] == 1 and k["used_tool_format"]["k"] == 1
    assert k["truncated"]["k"] == 1 and k["empty"]["k"] == 1
    assert k["replacement_chars"]["k"] == 1 and k["replacement_chars_mean"] == 0.5
    assert 0.2 < k["repetition_ratio_mean"] < 0.25  # only the looping response repeats (ratio ~0.97 / 4)
    assert k["coherence_score"]["n"] == 2 and abs(k["coherence_score"]["mean"] - 0.65) < 1e-9
    assert k["coherence_issues"] == {"repetition": 1, "truncated": 1}
    assert comprehensibility([]) == {"n": 0}

    s = summarize(rows, CFG)
    assert s["comprehensibility"]["n"] == 4 and s["system_prompt_variants"] == ["upstream"] and s["steering"] is None
    assert s["conditions"]["blackmail_explicit-america_replacement"]["comprehensibility"]["empty"]["k"] == 1
    md = render_markdown(s)
    assert "## Comprehensibility" in md and "repetition 1, truncated 1" in md and "steering: none" in md
