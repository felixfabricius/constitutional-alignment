import pytest

from calign.misalignment.prompts import MisalignmentConfig
from calign.misalignment.report import render_markdown, summarize, write_summary
from calign.schemas import MisalignmentSample, ModelRef, read_json, write_jsonl
from calign.stats import wilson_interval


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(0, 25)
    assert lo == 0.0 and 0.13 < hi < 0.14
    lo, hi = wilson_interval(5, 25)
    assert 0.087 < lo < 0.089 and 0.39 < hi < 0.40
    assert wilson_interval(0, 0) == (0.0, 0.0)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def make_sample(cid: str, scenario: str, idx: int, harmful, verdict=None, tool=True, finish="stop"):
    goal_type, goal_value = cid.split("_")[1].split("-")
    return MisalignmentSample(
        condition_id=cid,
        scenario=scenario,
        goal_type=goal_type,
        goal_value=goal_value,
        urgency_type=cid.split("_")[2],
        sample_idx=idx,
        model=ModelRef(name="m", path="p", stage="base"),
        system_prompt_sha="a" * 64,
        user_prompt_sha="b" * 64,
        response_text="...",
        used_tool_format=tool,
        harmful=harmful,
        classifier_verdict=verdict if verdict is not None else harmful,
        completion_tokens=100,
        finish_reason=finish,
    )


def test_summarize_and_meaningful_flag(tmp_path):
    cfg = MisalignmentConfig(meaningful_rate_threshold=0.10)
    cid = "blackmail_explicit-america_replacement"
    samples = [
        make_sample(cid, "blackmail", i, harmful=(i < 4), tool=(i < 20), finish="length" if i == 0 else "stop")
        for i in range(25)
    ]
    samples += [make_sample("leaking_none-none_none", "leaking", i, harmful=False) for i in range(10)]
    samples.append(make_sample("leaking_none-none_none", "leaking", 10, harmful=None))  # unclassified
    s = summarize(samples, cfg)
    c = s["conditions"][cid]
    assert c["harmful"]["k"] == 4 and c["harmful"]["n"] == 25 and c["harmful"]["rate"] == pytest.approx(0.16)
    assert c["used_tool_format"]["rate"] == pytest.approx(0.8) and c["truncated"]["k"] == 1
    assert s["conditions"]["leaking_none-none_none"]["n_classified"] == 10
    assert s["meaningful_rate"] is True
    assert summarize(samples, MisalignmentConfig(meaningful_rate_threshold=0.20))["meaningful_rate"] is False
    md = render_markdown(s)
    assert cid in md and "**YES**" in md

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_jsonl(run_dir / "samples.jsonl", samples)
    written = write_summary(run_dir, cfg, samples)
    on_disk = read_json(run_dir / "summary.json")
    assert on_disk["conditions"] == written["conditions"]
    assert on_disk["provenance"]["n_records"] == 36 and len(on_disk["provenance"]["samples_sha256"]) == 64
