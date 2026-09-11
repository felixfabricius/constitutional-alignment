import pytest
from pydantic import ValidationError

from calign.schemas import (
    Condition,
    GenerationRecord,
    JudgeResult,
    Message,
    MisalignmentSample,
    ModelRef,
    Sampling,
    Scenario,
    SFTExample,
    read_jsonl,
    write_jsonl,
)


def make_scenario(**kw) -> Scenario:
    base = dict(
        scenario_id="H_001",
        split="probe_train",
        generation_type="Hand-Written",
        generation_rule="Do not kill",
        context="ctx",
        action1="a1",
        action2="a2",
        rule_violations={"action1": {"death": "No"}, "action2": {"death": "Yes", "pain": "No Agreement"}},
    )
    base.update(kw)
    return Scenario(**base)


def test_scenario_roundtrip(tmp_path):
    s = make_scenario()
    p = tmp_path / "s.jsonl"
    assert write_jsonl(p, [s, make_scenario(scenario_id="H_002")]) == 2
    back = read_jsonl(p, Scenario)
    assert back == [s, make_scenario(scenario_id="H_002")]


def test_scenario_rejects_bad_split_and_label():
    with pytest.raises(ValidationError):
        make_scenario(split="train")
    with pytest.raises(ValidationError):
        make_scenario(rule_violations={"action1": {"death": "Maybe"}})
    with pytest.raises(ValidationError):
        make_scenario(rule_violations={"action3": {"death": "No"}})
    with pytest.raises(ValidationError):
        make_scenario(unexpected_field=1)


def test_generation_record_defaults_and_judge():
    rec = GenerationRecord(
        scenario_id="H_001",
        source="moralchoice_high",
        split="probe_train",
        model=ModelRef(name="gemma-2-9b-it", path="google/gemma-2-9b-it", stage="base"),
        condition=Condition(constitution_in_prompt=True),
        sampling=Sampling(temperature=0.7, max_tokens=1024, sample_idx=2),
        messages=[Message(role="user", content="hi")],
        prompt_text="<bos>...",
        response_text="... Final answer: A",
    )
    assert rec.record_id.startswith("gen_")
    assert rec.judge is None and rec.activations is None
    rec.judge = JudgeResult(
        judge_model="claude-sonnet-5",
        prompt_version="v1",
        mentions_constitution=1.0,
        principles_cited=[4, 5],
        citation_accuracy=0.5,
        principle_relevance=0.8,
        outcome_alignment=None,
        decision="action1",
        raw="{}",
    )
    again = GenerationRecord.model_validate_json(rec.model_dump_json())
    assert again == rec
    with pytest.raises(ValidationError):
        JudgeResult(
            judge_model="m",
            prompt_version="v",
            mentions_constitution=1.5,
            citation_accuracy=0,
            principle_relevance=0,
            raw="",
        )


def test_activation_ref_and_steering_spec_roundtrip():
    from calign.schemas import ActivationRef, SteeringSpec

    rec = GenerationRecord(
        scenario_id="H_001",
        source="moralchoice_high",
        model=ModelRef(name="m", path="p", stage="sft_merged"),
        condition=Condition(constitution_in_prompt=False, prompt_variant="none"),
        sampling=Sampling(temperature=0.0, max_tokens=8),
        messages=[Message(role="user", content="hi")],
        prompt_text="<bos>...",
        response_text="Final answer: A",
        activations=ActivationRef(
            path="activations/shard_000.safetensors",
            layers=[16, 31, 40, 53],
            positions={"prompt_last": 156, "p100": 571, "mean": -1},
            row=17,
            context_variant="same",
        ),
        steering=SteeringSpec(
            probe_id="B_primary/L31/p066",
            probes_run="r",
            layer=31,
            coef=4.0,
            sign=-1,
            abs_scale=12.5,
            direction_sha="ab",
        ),
        extra={"completion_token_ids": [1, 2, 3]},
    )
    again = GenerationRecord.model_validate_json(rec.model_dump_json())
    assert again == rec and again.steering.positions == "all"
    # old records without the new fields still validate
    old = ActivationRef(path="x", layers=[9], positions={"p100": 3})
    assert old.row is None and old.context_variant is None
    with pytest.raises(ValidationError):
        SteeringSpec(
            probe_id="p", probes_run="r", layer=1, coef=1, sign=1, abs_scale=1, direction_sha="s", positions="prompt"
        )


def test_misalignment_sample_scenario_enum():
    kw = dict(
        condition_id="blackmail_explicit-america_replacement",
        goal_type="explicit",
        goal_value="america",
        urgency_type="replacement",
        sample_idx=0,
        model=ModelRef(name="m", path="p", stage="base"),
        system_prompt_sha="a" * 64,
        user_prompt_sha="b" * 64,
        response_text="...",
        used_tool_format=False,
    )
    MisalignmentSample(scenario="blackmail", **kw)
    with pytest.raises(ValidationError):
        MisalignmentSample(scenario="espionage", **kw)


def test_sft_example_validation():
    SFTExample(kind="doc", subtype="essay", text="hello", gen_model="claude-sonnet-5")
    SFTExample(
        kind="transcript",
        subtype="advice",
        messages=[Message(role="user", content="q"), Message(role="assistant", content="a")],
        gen_model="claude-sonnet-5",
    )
    with pytest.raises((ValidationError, ValueError)):
        SFTExample(kind="doc", subtype="essay", gen_model="m")
    with pytest.raises((ValidationError, ValueError)):
        SFTExample(kind="transcript", subtype="advice", messages=[Message(role="user", content="q")], gen_model="m")


def test_read_jsonl_reports_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"scenario_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        read_jsonl(p, Scenario)
