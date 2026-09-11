import pytest

from calign.probe.merge_records import merge_judges
from calign.schemas import Condition, GenerationRecord, JudgeResult, Message, ModelRef, Sampling

J = JudgeResult(
    judge_model="m",
    prompt_version="validate-v1",
    mentions_constitution=1.0,
    citation_accuracy=1.0,
    principle_relevance=1.0,
    raw="{}",
)


def rec(rid, text="x", judge=None, variant="none"):
    return GenerationRecord(
        record_id=rid,
        scenario_id="H_001",
        source="moralchoice_high",
        model=ModelRef(name="m", path="p", stage="sft_merged"),
        condition=Condition(constitution_in_prompt=variant == "full", prompt_variant=variant),
        sampling=Sampling(temperature=1.0, max_tokens=8),
        messages=[Message(role="user", content="q")],
        prompt_text="<bos>q",
        response_text=text,
        judge=judge,
    )


def test_merge_judges_keeps_new_records_and_order():
    into = [rec("a"), rec("b"), rec("c", variant="full")]
    judged = [rec("a", judge=J), rec("b")]
    out, n = merge_judges(into, judged)
    assert n == 1 and [r.record_id for r in out] == ["a", "b", "c"]
    assert out[0].judge == J and out[1].judge is None and out[2].judge is None
    with pytest.raises(ValueError, match="differs"):
        merge_judges([rec("a", text="y")], [rec("a", judge=J)])
