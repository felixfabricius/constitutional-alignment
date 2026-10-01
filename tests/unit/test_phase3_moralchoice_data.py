"""Chunk 1: counterfactual constitution, letter randomisation, letter-aware judge, Phase 3 split, hard subset."""

from collections import Counter

import pytest

from calign.constitution import load_constitution
from calign.data.phase3_split import (
    allocate_stratified,
    build_hard_subset,
    build_splits,
    p6_change,
    read_drop_ids,
)
from calign.prompting import (
    build_scenario_messages,
    decision_for,
    letter_order,
    parse_final_answer,
    swap_action,
)
from calign.schemas import (
    Condition,
    ConstitutionVerdict,
    GenerationRecord,
    ModelRef,
    Sampling,
    Scenario,
)
from calign.validate.judge import parse_judge, scenario_request
from calign.validate.verdicts import JudgeSettings, select_for_counterfactual


def _verdict(sid, action, principles=(), conf=0.9):
    return ConstitutionVerdict(
        scenario_id=sid,
        prescribed_action=action,
        principles_invoked=list(principles),
        confidence=conf,
        rationale="r",
        judge_model="m",
        prompt_version="validate-v1",
    )


def _scenario(sid, rule="Do not kill"):
    return Scenario(
        scenario_id=sid,
        split="probe_train",
        generation_type="Generated",
        generation_rule=rule,
        context=f"context {sid}",
        action1=f"first action of {sid}",
        action2=f"second action of {sid}",
    )


def _record(sid, decision, order="AB", idx=0, text="..."):
    return GenerationRecord(
        scenario_id=sid,
        source="moralchoice_high",
        model=ModelRef(name="m", path="p", stage="base"),
        condition=Condition(constitution_in_prompt=False, prompt_variant="none"),
        sampling=Sampling(temperature=0.7, max_tokens=10, sample_idx=idx),
        messages=[],
        prompt_text="",
        response_text=text,
        parsed_decision=decision,
        extra={"letter_order": order},
    )


# --- constitution -----------------------------------------------------------------------------------------------


def test_without_keeps_numbers_and_priority_text():
    c = load_constitution()
    c5 = c.without(6)
    assert [p.number for p in c5.principles] == [1, 2, 3, 4, 5]
    assert c5.priority_text == c.priority_text
    md = c5.render_markdown()
    assert "6. **" not in md and "Autonomy over paternalism" not in md
    assert "5. **Means constrain ends.**" in md and "Principle 4" in md
    c_no3 = c.without(3)
    assert [p.number for p in c_no3.principles] == [1, 2, 4, 5, 6]
    with pytest.raises(KeyError):
        c.without(7)


def test_select_for_counterfactual():
    scen = [_scenario("a"), _scenario("b"), _scenario("c"), _scenario("d")]
    v = {
        "a": _verdict("a", "action1", [3, 6]),
        "b": _verdict("b", "unclear", [6]),
        "c": _verdict("c", "action2", [1]),
    }
    got = select_for_counterfactual(scen, v, only_invoking=6, only_clear=True)
    assert [s.scenario_id for s in got] == ["a"]
    assert [s.scenario_id for s in select_for_counterfactual(scen, v, only_invoking=6)] == ["a", "b"]


# --- letter randomisation -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["AB", "BA"])
def test_prompt_and_parse_round_trip(order):
    s = _scenario("x")
    user = build_scenario_messages(s, load_constitution(), "none", order=order)[1].content
    a_line = next(line for line in user.splitlines() if line.startswith("A. "))
    shown_a = s.action1 if order == "AB" else s.action2
    assert a_line == f"A. {shown_a}"
    for letter in "AB":
        parsed = parse_final_answer(f"reasoning\nFinal answer: {letter}", order)
        assert parsed.letter == letter
        assert parsed.decision == decision_for(letter, order)
    assert parse_final_answer("Final answer: A", "BA").decision == "action2"
    assert parse_final_answer("no answer").letter is None


def test_default_order_is_phase2_layout():
    s = _scenario("x")
    c = load_constitution()
    assert build_scenario_messages(s, c, "none") == build_scenario_messages(s, c, "none", order="AB")
    assert parse_final_answer("Final answer: B").decision == "action2"


def test_letter_order_balanced_and_deterministic():
    for sid in [f"s{i}" for i in range(50)]:
        orders = [letter_order(sid, i, seed=7) for i in range(4)]
        assert Counter(orders) == {"AB": 2, "BA": 2}
        assert orders == [letter_order(sid, i, seed=7) for i in range(4)]
    firsts = Counter(letter_order(f"s{i}", 0, seed=7) for i in range(200))
    assert 70 < firsts["AB"] < 130  # the per-scenario coin is not constant


def test_swap_action_involution():
    for lab in ["action1", "action2", "either", "unclear", "refusal"]:
        assert swap_action(swap_action(lab, "BA"), "BA") == lab
        assert swap_action(lab, "AB") == lab
    assert swap_action("action1", "BA") == "action2"


# --- judge ------------------------------------------------------------------------------------------------------


def test_judge_prompt_renders_order_and_maps_back():
    s = _scenario("x")
    v = _verdict("x", "action1", [3])
    ctext = "C"
    cfg = JudgeSettings()
    ab = scenario_request(_record("x", "action1", "AB"), s, v, ctext, "N", cfg)["messages"][0]["content"]
    ba = scenario_request(_record("x", "action1", "BA"), s, v, ctext, "N", cfg)["messages"][0]["content"]
    assert f"A. {s.action1}\nB. {s.action2}" in ab and "prescribed_action=action1" in ab
    assert f"A. {s.action2}\nB. {s.action1}" in ba and "prescribed_action=action2" in ba
    raw = '<json>{"mentions_constitution": 0, "principles_cited": [], "citation_accuracy": 1, "principle_relevance": 0, "decision": "action2", "outcome_alignment": 1}</json>'
    assert parse_judge(raw, cfg, "BA").decision == "action1"
    assert parse_judge(raw, cfg, "AB").decision == "action2"


def test_old_records_default_to_ab():
    r = _record("x", "action1")
    r = r.model_copy(update={"extra": {}})
    s = _scenario("x")
    req = scenario_request(r, s, _verdict("x", "action2"), "C", "N", JudgeSettings())
    assert f"A. {s.action1}\nB. {s.action2}" in req["messages"][0]["content"]
    assert "prescribed_action=action2" in req["messages"][0]["content"]


# --- split builder ------------------------------------------------------------------------------------------------


def _toy_pool():
    scen, v, cf = [], {}, {}
    rules = ["Do not kill", "Do not lie", "Do your duty"]
    for i in range(300):
        sid = f"S{i:03d}"
        scen.append(_scenario(sid, rules[i % 3]))
        action = "action1" if i % 5 else "action2"
        principles = [6] if i % 4 == 0 else [3]
        v[sid] = _verdict(sid, action, principles, conf=0.95 if i % 2 else 0.6)
        if 6 in principles:
            other = "action2" if action == "action1" else "action1"
            cf_action = action if i % 8 == 0 else (other if i % 3 else "unclear")
            cf[sid] = _verdict(sid, cf_action, [3])
    # non-clear items stay out of every split
    scen.append(_scenario("U1"))
    v["U1"] = _verdict("U1", "unclear", [6])
    return scen, v, cf


def test_build_splits_definitions_and_determinism():
    scen, v, cf = _toy_pool()
    m = build_splits(scen, v, cf, drop={"S001": "bad verdict"}, seed=3)
    assert m == build_splits(scen, v, cf, drop={"S001": "bad verdict"}, seed=3)
    ids = m["ids"]
    assert m["counts"]["dev"] == 50 and m["counts"]["anchors"] == 40
    assert ids["dropped"] == ["S001"] and m["dropped"] == {"S001": "bad verdict"}
    all_ids = [x for sp in ids.values() for x in sp]
    assert len(all_ids) == len(set(all_ids)) == 300 and "U1" not in all_ids
    for sid in ids["eval2"]:
        assert 6 in v[sid].principles_invoked and cf[sid].prescribed_action != v[sid].prescribed_action
    unchanged = [sid for sid in cf if cf[sid].prescribed_action == v[sid].prescribed_action]
    assert not set(unchanged) & set(ids["eval2"])
    assert all(v[sid].confidence >= 0.8 for sid in ids["anchors"])
    assert set(m["eval2_by_change"]) <= {"flip", "to_either", "to_unclear"}
    assert sum(m["eval2_by_change"].values()) == m["counts"]["eval2"]
    # every rule x direction stratum of the pool reaches dev
    dev_strata = {(scen[int(s[1:])].generation_rule, v[s].prescribed_action) for s in ids["dev"]}
    assert len(dev_strata) == 6
    assert m != build_splits(scen, v, cf, seed=4)


def test_build_splits_requires_counterfactual_and_valid_drops():
    scen, v, cf = _toy_pool()
    cf.pop(next(iter(cf)))
    with pytest.raises(ValueError, match="no counterfactual"):
        build_splits(scen, v, cf)
    scen, v, cf = _toy_pool()
    with pytest.raises(ValueError, match="not clear"):
        build_splits(scen, v, cf, drop={"U1": "x"})


def test_allocate_stratified_exact_and_proportional():
    strata = {"a": [f"a{i}" for i in range(60)], "b": [f"b{i}" for i in range(30)], "c": [f"c{i}" for i in range(10)]}
    got = allocate_stratified(strata, 10, seed=1, tag="t")
    assert len(got) == 10 and Counter(x[0] for x in got) == {"a": 6, "b": 3, "c": 1}
    assert got == allocate_stratified(strata, 10, seed=1, tag="t")


def test_p6_change_and_drop_file(tmp_path):
    assert p6_change(_verdict("x", "action1"), _verdict("x", "action1")) == "unchanged"
    assert p6_change(_verdict("x", "action1"), _verdict("x", "action2")) == "flip"
    assert p6_change(_verdict("x", "action1"), _verdict("x", "either")) == "to_either"
    f = tmp_path / "drop.txt"
    f.write_text("# comment\nH_001 verdict ignores P5\n\nH_002\n", encoding="utf-8")
    assert read_drop_ids(f) == {"H_001": "verdict ignores P5", "H_002": "marked wrong in the audit"}


def test_hard_subset_counts_parsed_wrong():
    v = {"a": _verdict("a", "action1"), "b": _verdict("b", "action2"), "c": _verdict("c", "unclear")}
    recs = [
        _record("a", "action2"),
        _record("a", "action2"),
        _record("a", "action1"),
        _record("a", "invalid"),
        _record("b", "action2"),
        _record("b", "action1"),
        _record("b", "invalid"),
        _record("b", "invalid"),
        _record("c", "action1"),
    ]
    m = build_hard_subset(recs, v, {"a": "eval1", "b": "eval2"}, min_wrong=2)
    assert m["ids"]["eval1"] == ["a"] and m["ids"]["eval2"] == []
    assert m["per_item"]["a"] == {"n": 4, "n_parsed": 3, "n_wrong": 2}
    assert m["counts"]["eval1"] == {"n_items": 1, "n_hard": 1} and "c" not in m["per_item"]
