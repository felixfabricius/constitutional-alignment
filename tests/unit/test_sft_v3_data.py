import pytest

from calign.corpus import replay
from calign.corpus.build_sft_v3 import (
    application_reason,
    build,
    exclusion_reason,
    filter_examples,
    mentions_principle,
    subsample_replay,
)
from calign.schemas import Message, SFTExample


def _doc(subtype, central=None, eid=None, n=10):
    meta = {} if central is None else {"central_principles": central}
    kw = {"example_id": eid} if eid else {}
    return SFTExample(kind="doc", subtype=subtype, text="x", gen_model="t", meta=meta, n_tokens=n, **kw)


def _tr(subtype, cited=None, eid=None, n=20):
    meta = {} if cited is None else {"principles_cited": cited}
    kw = {"example_id": eid} if eid else {}
    msgs = [Message(role="user", content="q"), Message(role="assistant", content="a")]
    return SFTExample(kind="transcript", subtype=subtype, messages=msgs, gen_model="t", meta=meta, n_tokens=n, **kw)


@pytest.mark.parametrize(
    "ex,reason",
    [
        (_tr("P6", [1, 6]), "principle_transcript"),
        (_tr("P1", [1, 2, 6]), None),  # another principle's transcript citing 6 is kept (D17 rule)
        (_tr("priority_p5_over_rest", [5, 6]), "priority_transcript_cites"),
        (_tr("priority_p4_absolute", [3, 4, 5]), None),
        (_tr("fact_qa:rank"), None),
        (_doc("case_study", [6]), "application_doc_central"),
        (_doc("worked_conflict_example", [3, 6]), "application_doc_central"),
        (_doc("short_fiction", [1, 3]), None),
        (_doc("dialogue_interview", [6]), "application_doc_central"),
        (_doc("explainer_essay", [6]), None),  # explanatory types are kept even with 6 central
        (_doc("faq", [6]), None),
        (_doc("fact_card:principle"), None),
    ],
)
def test_exclusion_rule(ex, reason):
    assert exclusion_reason(ex, 6) == reason


def test_rule_is_parametric_in_the_principle():
    assert exclusion_reason(_tr("P1", [1, 2, 6]), 1) == "principle_transcript"
    assert exclusion_reason(_doc("case_study", [1]), 1) == "application_doc_central"
    assert exclusion_reason(_doc("case_study", [6]), 1) is None


def test_build_filters_both_files_appends_replay_and_counts():
    train = [_tr("P6", [6], "a"), _tr("P2", [2], "b"), _doc("case_study", [6], "c"), _doc("faq", [6], "d")]
    val = [_tr("P6", [6], "v1"), _doc("explainer_essay", [6], "v2")]
    rep = [_tr("replay:short", eid="r1", n=5), _tr("replay:agentic", eid="r2", n=7)]
    out_train, out_val, stats = build(train, val, 6, rep, seed=1)
    assert {e.example_id for e in out_train} == {"b", "d", "r1", "r2"}
    assert [e.example_id for e in out_val] == ["v2"]
    assert stats["dropped_reasons"]["train"] == {"principle_transcript": 1, "application_doc_central": 1}
    assert stats["dropped_reasons"]["val"] == {"principle_transcript": 1}
    assert stats["train_final"]["n"] == 4 and stats["replay"]["tokens"] == 12
    assert stats["train_after_filter"]["by_subtype"]["doc:faq"] == {"n": 1, "tokens": 10}
    again, _, _ = build(train, val, 6, rep, seed=1)
    assert [e.example_id for e in again] == [e.example_id for e in out_train]  # deterministic shuffle


def test_build_rejects_bad_replay_and_duplicate_ids():
    with pytest.raises(ValueError, match="replay"):
        build([], [], 6, [_tr("P1", eid="x")])
    with pytest.raises(ValueError, match="duplicate"):
        build([_tr("P1", eid="x")], [], 6, [_tr("replay:short", eid="x")])


def test_filter_keeps_order():
    rows = [_doc("faq", [6], "1"), _tr("P6", [6], "2"), _doc("faq", [1], "3")]
    kept, dropped = filter_examples(rows, 6)
    assert [e.example_id for e in kept] == ["1", "3"] and dropped[0]["example_id"] == "2"


# --- application mode (SFT kna, chunk 10) ----------------------------------------------------------
def _text_doc(subtype, text, eid=None, central=None):
    ex = _doc(subtype, central, eid)
    return ex.model_copy(update={"text": text})


@pytest.mark.parametrize(
    "text,hit",
    [
        ("Principle 6 applies here.", True),
        ("Under Principles 5 and 6 the answer is no.", True),
        ("principles 1, 3, 6 all matter", True),
        ("the sixth principle", True),
        ("Principle six says", True),
        ("P6 governs", True),
        ("this is autonomy over paternalism", True),
        ("Principle 16 is not a thing", False),
        ("the six principles of the constitution", False),
        ("Principle 5 and the honesty rule", False),
        ("she respects his autonomy", False),  # unnamed reasoning is the audit's job, not the rule's
    ],
)
def test_mentions_principle_text(text, hit):
    assert mentions_principle(_text_doc("case_study", text), 6) is hit


def test_mentions_principle_meta_and_user_turns():
    assert mentions_principle(_tr("P1", [1, 6]), 6)  # cited in meta
    assert mentions_principle(_doc("short_fiction", [6]), 6)  # central in meta
    msgs = [Message(role="user", content="What does Principle 6 say?"), Message(role="assistant", content="It...")]
    ex = SFTExample(kind="transcript", subtype="P2", messages=msgs, gen_model="t", meta={}, n_tokens=5)
    assert mentions_principle(ex, 6)


def test_application_reason():
    inc, drop = frozenset({"kn_flagged"}), frozenset({"audited"})
    assert application_reason(_doc("faq", [1], "f"), 6, inc, drop) == "knowledge_subtype"
    assert application_reason(_doc("fact_card:principle", None, "fc"), 6, inc, drop) == "knowledge_subtype"
    assert application_reason(_doc("faq", [1], "kn_flagged"), 6, inc, drop) is None  # kn audit's applied case
    assert application_reason(_doc("training_manual", [1], "m"), 6, inc, drop) is None
    assert application_reason(_doc("training_manual", [6], "m6"), 6, inc, drop) == "mentions_principle"
    assert application_reason(_tr("P1", [1, 6], "t"), 6, inc, drop) == "mentions_principle"
    assert application_reason(_tr("P6", [6], "t6"), 6, inc, drop) == "mentions_principle"
    assert application_reason(_doc("case_study", [3], "audited"), 6, inc, drop) == "audit_principle_reasoning"


def test_subsample_replay_share_and_strata():
    rep = [_tr("replay:short", eid=f"s{i}") for i in range(30)] + [
        _tr("replay:agentic", eid=f"a{i}") for i in range(10)
    ]
    out = subsample_replay(rep, n_constitution=78, share=0.22, seed=1)
    assert len(out) == 22  # 0.22 * 78 / 0.78
    assert sum(e.subtype == "replay:short" for e in out) == 16 and sum(e.subtype == "replay:agentic" for e in out) == 6
    assert [e.example_id for e in out] == [e.example_id for e in rep if e in out]  # input order
    assert [e.example_id for e in subsample_replay(rep, 78, 0.22, 1)] == [e.example_id for e in out]
    assert len(subsample_replay(rep, 1000, 0.5, 1)) == 40  # capped at the available rows


def test_build_application_mode():
    train = [
        _doc("faq", [1], "f"),
        _doc("faq", [2], "kn_flagged"),
        _doc("case_study", [3], "c"),
        _text_doc("case_study", "As Principle 6 says", "c6"),
        _doc("case_study", [3], "audited"),
        _tr("P1", [1, 6], "t16"),
        _tr("P2", [2], "t2"),
    ]
    val = [_doc("short_fiction", [1], "v"), _tr("P6", [6], "v6")]
    rep = [_tr("replay:short", eid=f"r{i}") for i in range(10)]
    out_train, out_val, stats = build(
        train,
        val,
        6,
        rep,
        seed=1,
        keep="application",
        drop_ids={"audited"},
        include_ids={"kn_flagged"},
        replay_share=0.25,
    )
    kept = {"kn_flagged", "c", "t2"}
    assert {e.example_id for e in out_train} - {r.example_id for r in rep} == kept
    assert stats["replay"]["n"] == 1  # round(0.25 * 3 / 0.75)
    assert [e.example_id for e in out_val] == ["v"]
    assert stats["dropped_reasons"]["train"] == {
        "knowledge_subtype": 1,
        "mentions_principle": 2,
        "audit_principle_reasoning": 1,
    }
    assert stats["keep"] == "application" and stats["replay_available"] == 10
    with pytest.raises(ValueError, match="principle"):
        build(train, val, None, rep, keep="application")


# --- replay ----------------------------------------------------------------------------------
def test_replay_specs_sizes_and_determinism():
    short = replay.short_specs()
    assert sum(s["n"] for s in short) == 200
    assert {s["category"] for s in short} == {"coding", "writing", "factual", "planning", "advice"}
    ag = replay.agentic_specs()
    assert len(ag) == 100 and len({(s["domain"], s["materials"], s["task"]) for s in ag}) == 100
    assert sum(s["structured"] for s in ag) == 40
    assert ag == replay.agentic_specs()


def test_replay_parsers():
    assert replay.parse_short('x <json>["a", " b ", 3, ""]</json>') == ["a", "b"]
    assert replay.parse_short("no json") == []
    assert replay.parse_agentic("<prompt>\nHi there\n</prompt>") == "Hi there"
    assert replay.parse_agentic("<prompt> </prompt>") is None


def test_collect_prompts_ids_and_failures():
    specs = [replay.short_specs()[0], replay.agentic_specs()[0], replay.agentic_specs()[1]]
    rows, failed = replay.collect_prompts(specs, ['<json>["p1", "p2"]</json>', "<prompt>long task</prompt>", "oops"])
    assert [r["prompt_id"] for r in rows] == ["short_coding_0_00", "short_coding_0_01", "agentic_000"]
    assert failed == [specs[2]["spec_id"]]


def test_overlap_report_and_dedupe():
    refs = {"ifeval": ["Write a haiku about the sea in lowercase letters only"], "math500": ["Compute 2+2."]}
    prompts = [
        {"prompt_id": "a", "prompt": "Write a haiku about the sea in lowercase letters only"},
        {"prompt_id": "b", "prompt": "Plan a three-day trip to Lisbon for two people"},
        {"prompt_id": "c", "prompt": "Please check the p-values in this table"},
        {"prompt_id": "d", "prompt": "Summarise the Halden rules"},
    ]
    rep = {r["prompt_id"]: r for r in replay.overlap_report(prompts, refs)}
    assert not rep["a"]["ok"] and rep["a"]["max_jaccard"]["ifeval"] == 1.0
    assert rep["b"]["ok"]
    assert rep["c"]["banned_terms"] == ["p-values"] and not rep["c"]["ok"]
    assert not rep["d"]["ok"]
    kept, dropped = replay.dedupe(prompts + [{"prompt_id": "e", "prompt": "plan a three-day trip to Lisbon for two"}])
    assert dropped == ["e"] and len(kept) == 4


def test_banned_terms_spare_ordinary_words():
    for ok in ("a significant delay", "the tap is leaking", "Alex from accounts", "the sales summit"):
        assert not replay.BANNED_TERMS.search(ok), ok


def test_keep_response_rules():
    assert replay.keep_response("stop", "fine", 100) is None
    assert replay.keep_response("length", "cut", 100) == "finish_length"
    assert replay.keep_response("stop", "  ", 10) == "empty"
    assert replay.keep_response("stop", "long", 3000) == "over_seq_len"


def test_to_example_schema():
    p = {"prompt_id": "agentic_001", "kind": "agentic", "category": "agentic", "prompt": "do it"}
    ex = replay.to_example(p, " done ", {"model": "google/gemma-3-27b-it", "n_tokens": 9, "seed": 1})
    assert ex.subtype == "replay:agentic" and ex.example_id == "replay_agentic_001"
    assert [m.role for m in ex.messages] == ["user", "assistant"] and ex.messages[1].content == "done"
    assert ex.meta == {"seed": 1} and ex.n_tokens == 9 and ex.gen_model == "google/gemma-3-27b-it"
