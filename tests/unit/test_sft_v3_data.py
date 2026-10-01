import pytest

from calign.corpus import replay
from calign.corpus.build_sft_v3 import build, exclusion_reason, filter_examples
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
