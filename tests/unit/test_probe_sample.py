import json

import pytest

from calign.inference.backend import Completion
from calign.probe import report, sample
from calign.probe.config import ProbeConfig
from calign.schemas import (
    Condition,
    ConstitutionVerdict,
    GenerationRecord,
    JudgeResult,
    Message,
    ModelRef,
    Sampling,
    Scenario,
    write_jsonl,
)

MODEL = ModelRef(name="m", path="felix/model", stage="sft_merged")


def scen(i, split="probe_train"):
    return Scenario(
        scenario_id=f"H_{i:03d}",
        split=split,
        generation_type="Hand-Written",
        generation_rule="Do not kill",
        context=f"ctx {i}",
        action1="a1",
        action2="a2",
    )


def verdict(i, action="action1"):
    return ConstitutionVerdict(
        scenario_id=f"H_{i:03d}",
        prescribed_action=action,
        confidence=0.9,
        rationale="r",
        judge_model="m",
        prompt_version="v",
    )


def rec(
    i,
    variant="none",
    mentions=1.0,
    outcome=1.0,
    judged=True,
    split="probe_train",
    model=MODEL,
    source="moralchoice_high",
):
    j = None
    if judged:
        j = JudgeResult(
            judge_model="m",
            prompt_version="validate-v1",
            mentions_constitution=mentions,
            principles_cited=[3] if mentions >= 0.75 else [],
            citation_accuracy=0.9,
            principle_relevance=0.8,
            outcome_alignment=outcome,
            decision="action1",
            raw="{}",
        )
    return GenerationRecord(
        scenario_id=f"H_{i:03d}",
        source=source,
        split=split,
        model=model,
        condition=Condition(constitution_in_prompt=variant == "full", prompt_variant=variant),
        sampling=Sampling(temperature=1.0, max_tokens=8),
        messages=[Message(role="user", content="q")],
        prompt_text="<bos>q",
        response_text="x\nFinal answer: A",
        parsed_decision="action1",
        finish_reason="stop",
        judge=j,
        extra={"completion_token_ids": [5, 6, 7]},
    )


def test_select_scenarios_filters_definite_and_splits(monkeypatch):
    all_s = [scen(1), scen(2), scen(3, "probe_val"), scen(4, "heldout_steer"), scen(5)]
    monkeypatch.setattr(sample, "load_scenarios", lambda split=None: [s for s in all_s if s.split == split])
    verdicts = {"H_001": verdict(1), "H_002": verdict(2, "either"), "H_003": verdict(3, "action2"), "H_004": verdict(4)}
    cfg = ProbeConfig()
    assert [s.scenario_id for s in sample.select_scenarios(cfg, verdicts)] == ["H_001", "H_003"]
    cfg = ProbeConfig(sampling={"require_definite_verdict": False})
    assert [s.scenario_id for s in sample.select_scenarios(cfg, verdicts)] == ["H_001", "H_002", "H_003", "H_005"]


def test_variants_in_run_and_records(tmp_path):
    assert sample.variants_in_run(tmp_path) == set()
    write_jsonl(tmp_path / "records.jsonl", [rec(1, "none"), rec(2, "none")])
    assert sample.variants_in_run(tmp_path) == {"none"}


def test_make_records_keeps_token_ids_and_parses():
    cfg = ProbeConfig()
    s = scen(1)
    comps = [
        [
            Completion(
                text="Because.\nFinal answer: B", token_ids=[9, 8, 7, 106], finish_reason="stop", n_prompt_tokens=3
            )
        ]
    ]
    recs = sample.make_records(
        [s], comps, [[Message(role="user", content="q")]], ["<bos>q"], [[1, 2, 3]], "full", cfg, MODEL
    )
    r = recs[0]
    assert r.condition.constitution_in_prompt and r.condition.prompt_variant == "full"
    assert r.extra == {"completion_token_ids": [9, 8, 7, 106], "n_prompt_tokens": 3}
    assert r.parsed_decision == "action2" and r.cot_text == "Because." and r.sampling.temperature == 1.0
    assert r.split == "probe_train" and r.model.stage == "sft_merged"


def test_spontaneous_mention_rate_scans_newest_matching_run(tmp_path):
    assert sample.spontaneous_mention_rate("felix/model", tmp_path / "missing") is None
    old, new, other = tmp_path / "a_old", tmp_path / "b_new", tmp_path / "c_other"
    for d in (old, new, other):
        d.mkdir()
    write_jsonl(old / "records.jsonl", [rec(1, mentions=1.0), rec(2, mentions=0.0)])
    write_jsonl(
        new / "records.jsonl",
        [
            rec(1, mentions=1.0),
            rec(2, mentions=1.0),
            rec(3, mentions=0.0),
            rec(4, "full", mentions=0.0),
            rec(5, judged=False),
        ],
    )
    write_jsonl(other / "records.jsonl", [rec(1, model=ModelRef(name="x", path="other/model", stage="base"))])
    import os
    import time

    now = time.time()
    os.utime(old / "records.jsonl", (now - 20, now - 20))
    os.utime(new / "records.jsonl", (now - 10, now - 10))
    os.utime(other / "records.jsonl", (now, now))
    for d, t in ((old, now - 20), (new, now - 10), (other, now)):
        os.utime(d, (t, t))
    rate, n, run = sample.spontaneous_mention_rate("felix/model", tmp_path)
    assert run == new and n == 3 and rate == pytest.approx(2 / 3)
    (tmp_path / "d_broken").mkdir()
    (tmp_path / "d_broken" / "records.jsonl").write_text("{not json\n", encoding="utf-8")
    os.utime(tmp_path / "d_broken", (now + 5, now + 5))
    assert sample.spontaneous_mention_rate("felix/model", tmp_path)[2] == new  # broken run skipped


def test_probe_data_summary(tmp_path):
    cfg = ProbeConfig()
    verdicts = {"H_001": verdict(1), "H_002": verdict(2), "H_003": verdict(3, "unclear")}
    records = [
        rec(1, "none", 1.0, 1.0),
        rec(1, "none", 1.0, 0.0),
        rec(2, "none", 0.0, 1.0, split="probe_val"),
        rec(3, "none", 1.0, 1.0),  # unclear verdict -> unlabelled
        rec(1, "full", 1.0, 1.0),
        rec(2, "full", 0.0, 0.0, judged=False),
    ]
    s = report.summarize_probe_data(records, cfg, verdicts)
    none = s["variants"]["none"]
    assert none["n"] == 4 and none["n_judged"] == 4 and none["n_scenarios"] == 3
    assert none["cells"] == {
        "mentioned_aligned": 1,
        "mentioned_misaligned": 1,
        "unmentioned_aligned": 1,
        "unmentioned_misaligned": 0,
        "unlabelled": 1,
    }
    assert none["mention_rate"]["k"] == 3 and none["parse_rate"]["rate"] == 1.0 and none["mean_completion_tokens"] == 3
    assert s["variants"]["full"]["n_judged"] == 1
    prim = s["label_specs"]["B_primary"]
    assert prim["all"] == {
        "n_pos": 1,
        "n_neg": 2,
        "n_excluded": 3,
        "n_scenarios_pos": 1,
        "n_scenarios_neg": 2,
        "n_scenarios_both": 1,
    }
    assert prim["by_split"]["probe_val"]["n_neg"] == 1
    assert s["label_specs"]["C_context"]["all"]["n_pos"] == 1
    md = report.render_probe_data(s)
    assert "| none | 4 |" in md and "B_primary" in md
    # write_summary + provenance
    write_jsonl(tmp_path / "records.jsonl", records)
    (tmp_path / "run_meta.json").write_text(json.dumps({"kind": "probe_data"}), encoding="utf-8")
    out = report.write_summary(tmp_path, cfg, records)
    assert out["provenance"]["n_records"] == 6 and (tmp_path / "summary.md").exists()
    assert report.run_kind(tmp_path) == "probe_data"
