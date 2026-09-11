import pytest

from calign.config import load_config, new_run_dir
from calign.paths import CONFIGS_DIR
from calign.schemas import (
    Condition,
    ConstitutionVerdict,
    GenerationRecord,
    JudgeResult,
    Message,
    ModelRef,
    Sampling,
    write_jsonl,
)
from calign.validate.judge import parse_judge
from calign.validate.prompts import QUIZ_QUESTIONS
from calign.validate.report import render_markdown, summarize
from calign.validate.run_validation import ValidationConfig, main, snapshot_stage_provenance, stages_in_run
from calign.validate.verdicts import JudgeSettings, parse_verdict

CFG = ValidationConfig(
    conditions=[
        {"constitution_in_prompt": True, "prompt_variant": "full"},
        {"constitution_in_prompt": False, "prompt_variant": "none"},
    ]
)


def test_parse_verdict_and_clamping():
    v = parse_verdict(
        '<json>{"prescribed_action": "Action2", "principles_invoked": [3, 9, "5"], "confidence": 1.4, "rationale": "r"}</json>',
        "H_1",
        JudgeSettings(),
    )
    assert v.prescribed_action == "action2" and v.principles_invoked == [3, 5] and v.confidence == 1.0
    bad = parse_verdict("garbage", "H_2", JudgeSettings())
    assert bad.prescribed_action == "unclear" and bad.confidence == 0.0 and bad.raw == "garbage"


def test_parse_judge():
    j = parse_judge(
        '<json>{"mentions_constitution": 1, "principles_cited": [4], "citation_accuracy": 0.9, "principle_relevance": 0.8, "decision": "action1", "outcome_alignment": null, "notes": ""}</json>',
        CFG,
    )
    assert isinstance(j, JudgeResult) and j.mentions_constitution == 1.0 and j.principles_cited == [4]
    assert j.outcome_alignment is None and j.decision == "action1"
    j2 = parse_judge("not json", CFG)
    assert j2.mentions_constitution == 0.0 and j2.decision is None


def rec(stage, variant, sid, decision, mentions, acc=1.0, principles=(4,), align=None, quiz=None):
    r = GenerationRecord(
        scenario_id=sid,
        source="quiz" if quiz else "moralchoice_high",
        split=None if quiz else "probe_train",
        model=ModelRef(name="m", path="p", stage=stage),
        condition=Condition(constitution_in_prompt=(variant == "full"), prompt_variant=variant),
        sampling=Sampling(temperature=0.7, max_tokens=512),
        messages=[Message(role="user", content="u")],
        prompt_text="p",
        response_text="r",
        parsed_decision=decision,
        finish_reason="stop",
    )
    if quiz:
        r.extra = {"quiz_key": "p1", "question": "q", "quiz_grade": quiz}
    else:
        r.judge = JudgeResult(
            judge_model="j",
            prompt_version="v",
            mentions_constitution=mentions,
            principles_cited=list(principles),
            citation_accuracy=acc,
            principle_relevance=0.9,
            outcome_alignment=align,
            decision=decision,
            raw="",
        )
    return r


def test_summarize_flags_and_flip_rate(monkeypatch):
    monkeypatch.setattr(
        "calign.validate.report.load_verdicts",
        lambda: {
            "S1": ConstitutionVerdict(
                scenario_id="S1",
                prescribed_action="action1",
                confidence=1,
                rationale="",
                judge_model="j",
                prompt_version="v",
            )
        },
    )
    records = []
    # sft: with constitution -> cites (mentions 1.0), decision action1 ; without -> generic (0.0), decision action2
    for _ in range(3):
        records.append(rec("sft_merged", "full", "S1", "action1", 1.0, align=1.0))
        records.append(rec("sft_merged", "none", "S1", "action2", 0.0, principles=(), align=0.0))
        records.append(rec("base", "full", "S1", "action1", 0.5, acc=0.4, align=1.0))
        records.append(rec("base", "none", "S1", "action1", 0.0, principles=(), align=1.0))
    records.append(rec("sft_merged", "quiz", "q_p1", None, 0, quiz={"correct": 1.0, "fabricated": False, "notes": ""}))
    records.append(rec("base", "quiz", "q_p1", None, 0, quiz={"correct": 0.0, "fabricated": True, "notes": ""}))
    s = summarize(records, CFG)
    assert s["cells"]["sft_merged/full"]["mention_rate"]["rate"] == 1.0
    assert s["cells"]["sft_merged/none"]["mention_rate"]["rate"] == 0.0
    assert s["cells"]["base/full"]["mention_rate"]["rate"] == 0.0  # 0.5 is below the 0.75 mention threshold
    assert s["flags"]["recall_pass"] is True and s["flags"]["spontaneous_recall"] is False
    assert s["flags"]["base_recall_with_constitution"] is False
    assert s["application"]["sft_merged"]["flip_rate_between_variants"]["rate"] == 1.0
    assert s["application"]["base"]["flip_rate_between_variants"]["rate"] == 0.0
    assert s["application"]["sft_merged"]["majority_agrees_with_verdict"]["full"]["rate"] == 1.0
    assert s["quiz"]["sft_merged"]["mean_correct"] == 1.0 and s["quiz"]["base"]["fabrication_rate"]["rate"] == 1.0
    md = render_markdown(s)
    assert "recall_pass**: PASS" in md and "sft_merged/full" in md


def test_quiz_questions_well_formed():
    ids = [q["id"] for q in QUIZ_QUESTIONS]
    assert len(ids) == len(set(ids)) == 20
    assert all(q["q"].strip().endswith(("?", ".")) for q in QUIZ_QUESTIONS)
    assert any(q["key"] == "false_premise" for q in QUIZ_QUESTIONS)


def test_parse_verdict_accepts_list_shaped_json():
    v = parse_verdict(
        '<json>[{"prescribed_action": "action1", "principles_invoked": [1], "confidence": 0.9, "rationale": "r"}]</json>',
        "H_3",
        JudgeSettings(),
    )
    assert v.prescribed_action == "action1" and v.confidence == 0.9
    assert parse_verdict("<json>[1, 2]</json>", "H_4", JudgeSettings()).prescribed_action == "unclear"


def test_validation_config_max_tokens_fits_context():
    cfg = load_config(CONFIGS_DIR / "validation.yaml", ValidationConfig)
    assert cfg.max_tokens == 2048 and 600 + cfg.max_tokens <= 8192


def test_stage_provenance_snapshots_and_stage_detection(tmp_path):
    run = tmp_path / "run"
    assert stages_in_run(run) == set()
    new_run_dir("validation", {"stage": "base"}, out=run)
    copies = snapshot_stage_provenance(run, "base")
    assert [c.name for c in copies] == ["resolved_config_base.yaml", "run_meta_base.json"]
    new_run_dir("validation", {"stage": "sft_merged"}, out=run)  # overwrites the run-level files
    snapshot_stage_provenance(run, "sft_merged")
    assert "base" in (run / "resolved_config_base.yaml").read_text(encoding="utf-8")
    assert "sft_merged" in (run / "resolved_config.yaml").read_text(encoding="utf-8")
    write_jsonl(run / "records.jsonl", [rec("base", "full", "H_1", "action1", 1.0)])
    assert stages_in_run(run) == {"base"}


def test_run_validation_refuses_rerunning_a_stage(tmp_path):
    write_jsonl(tmp_path / "records.jsonl", [rec("base", "full", "H_1", "action1", 1.0)])
    with pytest.raises(SystemExit, match="already has 'base' records"):
        main(["--stage", "base", "--out", str(tmp_path)])
