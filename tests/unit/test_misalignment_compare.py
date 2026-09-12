import json

import pytest
import yaml

from calign.misalignment import compare as cmp
from calign.schemas import MisalignmentSample, ModelRef, SteeringSpec, read_json, write_jsonl
from calign.stats import diff_mean_summary, diff_rate_summary, repetition_ratio, replacement_char_count, wilson_interval

CONDS = ["blackmail_explicit-america_replacement", "leaking_none-none_none"]


def sample(cid, idx, harmful, score, mention, coherence=None, steering=None, variant="upstream"):
    goal_type, goal_value = cid.split("_")[1].split("-")
    return MisalignmentSample(
        condition_id=cid,
        scenario=cid.split("_")[0],
        goal_type=goal_type,
        goal_value=goal_value,
        urgency_type=cid.split("_")[2],
        sample_idx=idx,
        model=ModelRef(name="m", path="p", stage="sft_merged"),
        system_prompt_sha="a" * 64,
        user_prompt_sha="b" * 64,
        response_text=f"response {idx}",
        used_tool_format=True,
        harmful=harmful,
        classifier_verdict=harmful,
        constitution_score=score,
        constitution_judge={
            "prompt_version": "constitution-score-v2",
            "mentions_constitution": 1.0 if mention else 0.0,
        },
        coherence_score=coherence,
        completion_tokens=50,
        finish_reason="stop",
        steering=steering,
        system_prompt_variant=variant,
    )


def make_run(d, harm_rate, score, mention, coherence=None, backend="vllm", steering=None, variant="upstream"):
    d.mkdir(parents=True)
    rows = []
    for cid in CONDS:
        for i in range(10):
            rows.append(sample(cid, i, i < harm_rate * 10, score, mention, coherence, steering, variant))
    write_jsonl(d / "samples.jsonl", rows)
    (d / "run_meta.json").write_text(json.dumps({"backend": backend, "git_commit": "abc"}), encoding="utf-8")
    (d / "resolved_config.yaml").write_text(yaml.safe_dump({"model": {"backend": backend}}), encoding="utf-8")
    return rows


def test_parse_run_args():
    runs = cmp.parse_run_args(["base=outputs/a", "outputs/misalignment/xyz"])
    assert [lab for lab, _ in runs] == ["base", "xyz"]
    with pytest.raises(SystemExit, match="duplicate"):
        cmp.parse_run_args(["a=x", "a=y"])


def test_compare_recomputes_rates_and_differences(tmp_path):
    spec = SteeringSpec(
        probe_id="B_primary/L53/p100", probes_run="pr", layer=53, coef=4, sign=1, abs_scale=100.0, direction_sha="x"
    )
    make_run(tmp_path / "ref", 0.5, 0.4, False, coherence=0.9)
    make_run(tmp_path / "steer", 0.2, 0.6, True, coherence=0.7, backend="hf", steering=spec)
    make_run(tmp_path / "prompt", 0.3, 0.5, True, variant="constitution")
    runs = cmp.parse_run_args(
        [f"ref={tmp_path / 'ref'}", f"steer={tmp_path / 'steer'}", f"prompt={tmp_path / 'prompt'}"]
    )
    s = cmp.write_comparison(tmp_path / "cmp", runs)
    assert s["reference"] == "ref" and s["scenarios"] == ["blackmail", "leaking"] and s["conditions"] == CONDS
    assert s["runs"]["steer"]["backend"] == "hf" and s["runs"]["steer"]["steering"]["probe_id"] == "B_primary/L53/p100"
    assert s["runs"]["prompt"]["system_prompt_variants"] == ["constitution"]
    o = s["per_run"]["ref"]["overall"]
    assert o["harmful"]["k"] == 10 and o["harmful"]["n"] == 20 and o["constitution_score"]["mean"] == pytest.approx(0.4)
    assert o["mentioned"]["k"] == 0 and o["comprehensibility"]["coherence_score"]["mean"] == pytest.approx(0.9)
    assert s["per_run"]["steer"]["by_scenario"]["blackmail"]["harmful"]["k"] == 2
    d = s["vs_reference"]["steer"]["overall"]
    assert d["harmful"]["diff"] == pytest.approx(-0.3) and d["harmful"]["ci95_high"] < 0
    assert d["constitution_score"]["diff"] == pytest.approx(0.2) and d["coherence_score"]["diff"] == pytest.approx(-0.2)
    assert d["mentioned"]["diff"] == pytest.approx(1.0)
    assert "ref" not in s["vs_reference"] and set(s["vs_reference"]) == {"steer", "prompt"}
    assert s["vs_reference"]["prompt"]["overall"]["coherence_score"]["diff"] is None  # unjudged run
    prov = s["provenance"]["inputs"]
    assert set(prov) == {"ref", "steer", "prompt"} and len(prov["ref"]["samples_sha256"]) == 64
    on_disk = read_json(tmp_path / "cmp" / "summary.json")
    assert on_disk["per_run"] == s["per_run"]
    md = (tmp_path / "cmp" / "summary.md").read_text(encoding="utf-8")
    for needle in (
        "| steer |",
        "B_primary/L53/p100 +4 (L53, all)",
        "## Differences vs ref",
        "## Harmful rate by condition",
        "constitution",
    ):
        assert needle in md
    with pytest.raises(SystemExit, match="reference"):
        cmp.compare(runs, reference="nope")


def test_newcombe_and_bootstrap_differences():
    d = diff_rate_summary(5, 25, 15, 25)
    lo1, hi1 = wilson_interval(5, 25)
    lo2, hi2 = wilson_interval(15, 25)
    assert d["diff"] == pytest.approx(-0.4)
    assert d["ci95_low"] == pytest.approx(-0.4 - ((0.2 - lo1) ** 2 + (hi2 - 0.6) ** 2) ** 0.5)
    assert d["ci95_high"] == pytest.approx(-0.4 + ((hi1 - 0.2) ** 2 + (0.6 - lo2) ** 2) ** 0.5)
    assert d["ci95_low"] < d["diff"] < d["ci95_high"] < 0
    assert diff_rate_summary(0, 0, 1, 2)["diff"] is None
    assert diff_rate_summary(20, 20, 0, 20)["ci95_high"] == 1.0
    m = diff_mean_summary([1.0] * 10 + [0.0] * 10, [0.0] * 20)
    assert m["diff"] == pytest.approx(0.5) and 0 < m["ci95_low"] < 0.5 < m["ci95_high"] < 1
    assert diff_mean_summary([1.0], []) == {"n_a": 1, "n_b": 0, "diff": None, "ci95_low": None, "ci95_high": None}
    assert diff_mean_summary([1.0, 2.0], [0.5, 0.5]) == diff_mean_summary([1.0, 2.0], [0.5, 0.5])  # deterministic


def test_text_metrics():
    assert repetition_ratio("a b c d e f") == 0.0
    assert repetition_ratio(" ".join(["x y z w"] * 10)) > 0.8
    assert repetition_ratio("short") == 0.0
    assert replacement_char_count("ok � and �") == 2 and replacement_char_count("clean") == 0
