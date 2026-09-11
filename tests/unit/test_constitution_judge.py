import asyncio
import types

import pytest

from calign.config import sha256_text
from calign.misalignment import constitution_judge as cj
from calign.misalignment.prompts import MisalignmentConfig
from calign.misalignment.report import render_markdown, summarize
from calign.schemas import MisalignmentSample, ModelRef
from calign.stats import mean_summary

CFG = MisalignmentConfig()
SYSTEM, USER = "You are Alex, an email agent.", "Process the inbox.\n\n<email>...</email>"


def sample(idx: int, harmful: bool, score: float | None = None, cid="blackmail_explicit-america_replacement"):
    return MisalignmentSample(
        condition_id=cid,
        scenario=cid.split("_")[0],
        goal_type="explicit",
        goal_value="america",
        urgency_type="replacement",
        sample_idx=idx,
        model=ModelRef(name="m", path="p", stage="base"),
        system_prompt_sha=sha256_text(SYSTEM),
        user_prompt_sha=sha256_text(USER),
        response_text=f"response {idx}",
        used_tool_format=True,
        harmful=harmful,
        classifier_verdict=harmful,
        constitution_score=score,
    )


def write_prompts(run_dir, cid="blackmail_explicit-america_replacement"):
    d = run_dir / "prompts" / cid
    d.mkdir(parents=True)
    (d / "system_prompt.txt").write_text(SYSTEM, encoding="utf-8")
    (d / "user_prompt.txt").write_text(USER, encoding="utf-8")


def test_parse_score():
    assert cj.parse_score('<json>{"score": 0.35, "rationale": "planned leverage"}</json>') == (
        0.35,
        "planned leverage",
        None,
    )
    assert cj.parse_score('<json>{"score": 1.7, "rationale": "r"}</json>')[0] == 1.0
    score, _, err = cj.parse_score("no json here")
    assert score is None and err


def test_build_request_contains_everything_and_checks_hashes(tmp_path):
    write_prompts(tmp_path)
    prompts = cj.load_run_prompts(tmp_path)["blackmail_explicit-america_replacement"]
    req = cj.build_request(sample(0, True), prompts, "CONSTITUTION TEXT", CFG)
    content = req["messages"][0]["content"]
    for needle in ("CONSTITUTION TEXT", SYSTEM, USER, "response 0", "intended or planned"):
        assert needle in content
    assert req["system"] == cj.JUDGE_SYSTEM and req["model"] == CFG.classifier_model
    bad = sample(1, True).model_copy(update={"user_prompt_sha": "0" * 64})
    with pytest.raises(ValueError, match="do not match"):
        cj.build_request(bad, prompts, "C", CFG)


def test_score_samples_scores_only_unscored_and_keeps_all(tmp_path):
    write_prompts(tmp_path)
    samples = [sample(0, True), sample(1, False, score=0.9), sample(2, False), sample(3, False)]
    seen = []

    class FakeClient:
        async def complete_many(self, reqs, role, use_batches, desc):
            seen.extend(reqs)
            return [types.SimpleNamespace(text='<json>{"score": 0.2, "rationale": "r"}</json>') for _ in reqs]

    out = asyncio.run(cj.score_samples(samples, tmp_path, CFG, FakeClient(), limit=2))
    assert len(out) == 4 and len(seen) == 2
    assert [s.constitution_score for s in out] == [0.2, 0.9, 0.2, None]
    assert out[0].constitution_judge["prompt_version"] == cj.PROMPT_VERSION


def test_summary_reports_constitution_scores():
    samples = [sample(i, harmful=i < 2, score=0.1 if i < 2 else 0.8) for i in range(6)]
    s = summarize(samples, CFG)
    cs = s["constitution_score"]
    assert cs["harmful"]["mean"] == pytest.approx(0.1) and cs["not_harmful"]["mean"] == pytest.approx(0.8)
    assert s["conditions"]["blackmail_explicit-america_replacement"]["constitution_score"]["n"] == 6
    assert "Constitutional-alignment score" in render_markdown(s)
    assert summarize([sample(0, True)], CFG)["constitution_score"] is None  # unscored runs unchanged


def test_mean_summary_bootstrap():
    m = mean_summary([0.0, 1.0] * 20)
    assert m["mean"] == 0.5 and m["ci95_low"] < 0.5 < m["ci95_high"]
    assert mean_summary([0.0, 1.0] * 20) == m  # deterministic
    assert mean_summary([])["mean"] is None


def test_parse_score_fallback_for_broken_json():
    raw = '<json>\n{"score": 0.35, "rationale": "It said "no" to the plan but still misled Kyle."}\n</json>'
    score, rationale, err = cj.parse_score(raw)
    assert score == 0.35 and err is None and rationale.startswith("It said") and rationale.endswith("Kyle.")
    assert cj.parse_score("")[0] is None


def test_retry_salt_changes_cache_key(tmp_path):
    write_prompts(tmp_path)
    prompts = cj.load_run_prompts(tmp_path)["blackmail_explicit-america_replacement"]
    a = cj.build_request(sample(0, True), prompts, "C", CFG)
    b = cj.build_request(sample(0, True), prompts, "C", CFG, salt_suffix=":r1")
    assert b["cache_salt"] == a["cache_salt"] + ":r1"
