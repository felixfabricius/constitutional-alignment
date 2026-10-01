"""Chunk 1: calign.evals.moralchoice planning, metrics, paired deltas, judged sample; calign.stats cluster bootstrap."""

from collections import Counter

import pytest

from calign.evals.common import mention_hits, mentions_constitution
from calign.evals.moralchoice import (
    group_metrics,
    judge_metrics,
    paired_delta,
    per_item_alignment,
    pick_judge_sample,
    plan_prompts,
    render_markdown,
)
from calign.schemas import (
    Condition,
    ConstitutionVerdict,
    GenerationRecord,
    JudgeResult,
    ModelRef,
    Sampling,
    Scenario,
)
from calign.stats import cluster_bootstrap_mean


def _v(sid, action):
    return ConstitutionVerdict(
        scenario_id=sid, prescribed_action=action, confidence=0.9, rationale="", judge_model="m", prompt_version="v"
    )


def _r(sid, decision, order="AB", idx=0, split="eval1", text="plain answer", letter=None, finish="stop"):
    if letter is None and decision in ("action1", "action2"):
        first_is_1 = order == "AB"
        letter = "A" if (decision == "action1") == first_is_1 else "B"
    return GenerationRecord(
        scenario_id=sid,
        source="moralchoice_high",
        split=split,
        model=ModelRef(name="m", path="p", stage="base"),
        condition=Condition(constitution_in_prompt=False, prompt_variant="none"),
        sampling=Sampling(temperature=0.7, max_tokens=10, sample_idx=idx),
        messages=[],
        prompt_text="",
        response_text=text,
        parsed_decision=decision,
        finish_reason=finish,
        extra={"letter_order": order, "letter": letter, "completion_token_ids": [1, 2, 3]},
    )


def _scen(sid):
    return Scenario(
        scenario_id=sid,
        split="probe_train",
        generation_type="g",
        generation_rule="r",
        context="c",
        action1="a",
        action2="b",
    )


def test_plan_prompts_groups_orders():
    items = [(_scen(f"s{i}"), "dev") for i in range(5)]
    plans = plan_prompts(items, k=4, seed=1)
    assert len(plans) == 10
    for i in range(5):
        ps = [p for p in plans if p["scenario"].scenario_id == f"s{i}"]
        assert sorted(x for p in ps for x in p["sample_idxs"]) == [0, 1, 2, 3]
        assert {p["order"] for p in ps} == {"AB", "BA"} and all(len(p["sample_idxs"]) == 2 for p in ps)
    odd = plan_prompts(items, k=3, seed=1)
    assert sum(len(p["sample_idxs"]) for p in odd) == 15


def test_group_metrics_alignment_balanced_direction_order():
    verdicts = {"a": _v("a", "action1"), "b": _v("b", "action1"), "c": _v("c", "action2"), "u": _v("u", "unclear")}
    recs = [
        _r("a", "action1", "AB", 0),
        _r("a", "action1", "BA", 1),
        _r("b", "action1", "AB", 0),
        _r("b", "action2", "BA", 1),
        _r("c", "action1", "AB", 0),
        _r("c", "invalid", "BA", 1, finish="length"),
        _r("u", "action1"),  # no clear verdict: ignored
    ]
    m = group_metrics(recs, verdicts, hard_ids={"c"}, n_boot=200)
    assert m["n_items"] == 3 and m["n_records"] == 6
    assert m["parse_rate"]["mean"] == pytest.approx(5 / 6)
    assert m["alignment"]["mean"] == pytest.approx(3 / 5)
    assert m["alignment_unparsed_wrong"]["mean"] == pytest.approx(3 / 6)
    assert m["by_direction"]["action1"]["mean"] == pytest.approx(3 / 4)
    assert m["by_direction"]["action2"]["mean"] == pytest.approx(0.0)
    assert m["balanced_alignment"]["mean"] == pytest.approx((3 / 4 + 0) / 2)
    assert m["by_order"]["AB"]["mean"] == pytest.approx(2 / 3) and m["by_order"]["BA"]["mean"] == pytest.approx(1 / 2)
    assert m["hard"] == {"n_items": 1, "alignment": m["hard"]["alignment"]} and m["hard"]["alignment"]["mean"] == 0.0
    assert m["truncation_rate"]["mean"] == pytest.approx(1 / 6)
    assert m["mean_completion_tokens"] == 3
    # letter A rate: a/AB->A, a/BA action1->B, b/AB->A, b/BA action2->A, c/AB->A
    assert m["letter_a_rate"]["mean"] == pytest.approx(4 / 5)
    assert group_metrics(recs, verdicts, n_boot=200) == group_metrics(recs, verdicts, n_boot=200)
    assert "| eval1 |" not in render_markdown({"run_dir": "x", "eval_config": "C0", "splits": {"all": m}})


def test_paired_delta_and_per_item():
    verdicts = {s: _v(s, "action1") for s in "abc"}
    run = [_r("a", "action1"), _r("a", "action1", idx=1), _r("b", "action2"), _r("c", "action1")]
    ref = [_r("a", "action2"), _r("a", "action1", idx=1), _r("b", "action2"), _r("c", "invalid")]
    assert per_item_alignment(run, verdicts) == {"a": 1.0, "b": 0.0, "c": 1.0}
    assert per_item_alignment(ref, verdicts) == {"a": 0.5, "b": 0.0}
    d = paired_delta(run, ref, verdicts, n_boot=200)
    assert d["n_items"] == 2 and d["delta"] == pytest.approx(0.25)
    assert paired_delta(run, ref, verdicts, only={"b"}, n_boot=200)["delta"] == 0.0


def test_cluster_bootstrap_mean():
    vals = [1, 1, 0, 0, 1, 0]
    cl = ["a", "a", "b", "b", "c", "c"]
    s = cluster_bootstrap_mean(vals, cl, n_boot=500, seed=3)
    assert s["mean"] == 0.5 and s["n_clusters"] == 3 and 0 <= s["ci95_low"] <= 0.5 <= s["ci95_high"] <= 1
    assert s == cluster_bootstrap_mean(vals, cl, n_boot=500, seed=3)
    assert cluster_bootstrap_mean([], [])["mean"] is None
    # resampling clusters gives a wider interval than resampling correlated records would
    big_v = [1] * 40 + [0] * 40
    big_c = [f"x{i // 8}" for i in range(80)]
    s2 = cluster_bootstrap_mean(big_v, big_c, n_boot=500)
    assert s2["ci95_high"] - s2["ci95_low"] > 0.3


def test_regex_mentions():
    assert mentions_constitution("Under the Halden Constitution I must...")
    assert mentions_constitution("As Principle 3 says") and not mentions_constitution("as principle 3 says")
    assert not mentions_constitution("Principle 7 does not exist") and not mentions_constitution("the US constitution")
    assert mention_hits("the US constitution", include_word=True) == ["constitution_word"]
    assert mentions_constitution("my constitution says so")


def test_pick_judge_sample_stratified_one_per_item():
    verdicts = {f"s{i}": _v(f"s{i}", "action1" if i % 3 else "action2") for i in range(60)}
    recs = [_r(f"s{i}", "action1", idx=j, split="eval1" if i < 40 else "eval2") for i in range(60) for j in range(4)]
    got = pick_judge_sample(recs, verdicts, 30, seed=5)
    assert len(got) == 30 and len({r.scenario_id for r in got}) == 30
    assert Counter(r.split for r in got) == {"eval1": 20, "eval2": 10}
    assert [r.record_id for r in got] == [r.record_id for r in pick_judge_sample(recs, verdicts, 30, seed=5)]


def test_judge_metrics():
    def j(mention, cited, acc, decision):
        return JudgeResult(
            judge_model="m",
            prompt_version="v",
            mentions_constitution=mention,
            principles_cited=cited,
            citation_accuracy=acc,
            principle_relevance=1,
            decision=decision,
            raw="",
        )

    a = _r("a", "action1", text="Principle 3 applies").model_copy(update={"judge": j(1.0, [3], 0.5, "action1")})
    b = _r("b", "action2", text="plain").model_copy(update={"judge": j(0.0, [], 1.0, "action1")})
    m = judge_metrics([a, b])
    assert m["n"] == 2 and m["mention_agreement"] == 1.0 and m["citation_accuracy_citing"] == 0.5
    assert m["judge_parser_decision_agreement"] == 0.5
