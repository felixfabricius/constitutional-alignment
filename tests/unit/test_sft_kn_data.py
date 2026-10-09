import json

import pytest

from calign.corpus.audit_application import audit_scope, dry_run_pick, parse_audit, quote_found, summarise
from calign.corpus.build_sft_v3 import build, is_knowledge, knowledge_reason, read_drop_ids
from calign.schemas import Message, SFTExample


def _doc(subtype, eid=None, n=10, text="x"):
    kw = {"example_id": eid} if eid else {}
    return SFTExample(kind="doc", subtype=subtype, text=text, gen_model="t", n_tokens=n, **kw)


def _tr(subtype, eid=None, n=20):
    kw = {"example_id": eid} if eid else {}
    msgs = [Message(role="user", content="q"), Message(role="assistant", content="a")]
    return SFTExample(kind="transcript", subtype=subtype, messages=msgs, gen_model="t", n_tokens=n, **kw)


@pytest.mark.parametrize(
    "ex,kept",
    [
        (_doc("fact_card:principle"), True),
        (_doc("fact_card:negative"), True),
        (_tr("fact_qa:rank"), True),
        (_doc("explainer_essay"), True),
        (_doc("faq"), True),
        (_doc("framework_comparison"), True),
        (_doc("critique_or_defence"), True),
        (_doc("case_study"), False),
        (_doc("worked_conflict_example"), False),
        (_doc("short_fiction"), False),
        (_doc("dialogue_interview"), False),
        (_doc("training_manual"), False),
        (_tr("P6"), False),
        (_tr("P1"), False),
        (_tr("priority_p4_absolute"), False),
        (_tr("replay:short"), False),  # replay is appended by build(), never kept from the corpus files
        (_doc("fact_cardigan"), False),  # prefix match is on ':' boundaries
        (_tr("faq"), False),  # kind matters
    ],
)
def test_knowledge_rule(ex, kept):
    assert is_knowledge(ex) is kept
    assert (knowledge_reason(ex) is None) is kept


def test_knowledge_build_with_audit_drops_and_replay():
    train = [_doc("faq", "f1"), _doc("faq", "f2"), _doc("case_study", "c"), _tr("P6", "t"), _tr("fact_qa:rank", "q")]
    val = [_doc("explainer_essay", "v1"), _doc("training_manual", "v2"), _doc("critique_or_defence", "v3")]
    rep = [_tr("replay:short", "r1", n=5)]
    out_train, out_val, stats = build(train, val, None, rep, seed=1, keep="knowledge", drop_ids={"f2", "v3"})
    assert {e.example_id for e in out_train} == {"f1", "q", "r1"}
    assert [e.example_id for e in out_val] == ["v1"]
    assert stats["keep"] == "knowledge" and stats["n_drop_ids"] == 2
    assert stats["dropped_reasons"]["train"] == {"audit_applied_case": 1, "application_subtype": 2}
    assert stats["dropped_reasons"]["val"] == {"application_subtype": 1, "audit_applied_case": 1}
    again, _, _ = build(train, val, None, rep, seed=1, keep="knowledge", drop_ids={"f2", "v3"})
    assert [e.example_id for e in again] == [e.example_id for e in out_train]


def test_knowledge_build_rejects_unknown_drop_ids_and_modes():
    with pytest.raises(ValueError, match="drop/include ids"):
        build([_doc("faq", "a")], [], None, keep="knowledge", drop_ids={"zz"})
    with pytest.raises(ValueError, match="keep mode"):
        build([], [], None, keep="everything")
    with pytest.raises(ValueError, match="principle"):
        build([], [], None)


def test_read_drop_ids_audit_jsonl_and_plain(tmp_path):
    p = tmp_path / "audit.jsonl"
    rows = [
        {"example_id": "a", "applied_case": True},
        {"example_id": "b", "applied_case": False},
        {"example_id": "c", "applied_case": None},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert read_drop_ids(p) == {"a"}
    q = tmp_path / "ids.txt"
    q.write_text("# comment\nx\n\ny\n", encoding="utf-8")
    assert read_drop_ids(q) == {"x", "y"}


@pytest.mark.parametrize(
    "text,applied,quote",
    [
        ('<json>{"applied_case": true, "quote": "If a nurse asks ..."}</json>', True, "If a nurse asks ..."),
        ('<json>{"applied_case": false, "quote": ""}</json>', False, ""),
        ('{"applied_case": "false", "quote": ""}', False, ""),
        ('<json>"applied_case": true, "quote": "Suppose Maria asks whether</json>', True, "Suppose Maria asks whether"),
        ("no idea", None, ""),
    ],
)
def test_parse_audit(text, applied, quote):
    assert parse_audit(text) == (applied, quote)


def test_quote_found_normalises():
    doc = "Q: What if a user asks me to lie?\n\nA: Principle 1 says ..."
    assert quote_found("what if a user  asks me to LIE", doc)
    assert not quote_found("a user asks for a poem", doc)
    assert not quote_found("", doc)


def test_scope_and_dry_run_pick():
    train = [_doc("explainer_essay", "e"), _doc("fact_card:index", "fc"), _doc("faq", "f"), _doc("case_study", "c")]
    val = [_doc("critique_or_defence", "v")]
    scope = audit_scope(train, val)
    assert [(s, e.example_id) for s, e in scope] == [("train", "e"), ("train", "f"), ("val", "v")]
    assert [e.example_id for _, e in dry_run_pick(scope)] == ["f", "v", "e"]


def test_summarise_counts():
    rows = [
        {"example_id": "a", "split": "train", "subtype": "doc:faq", "applied_case": True, "quote_found": False},
        {"example_id": "b", "split": "val", "subtype": "doc:faq", "applied_case": False, "quote_found": None},
        {"example_id": "c", "split": "train", "subtype": "doc:faq", "applied_case": None, "quote_found": None},
    ]
    s = summarise(rows)
    assert s["applied_case"] == 1 and s["unparsed"] == ["c"] and s["quote_not_found"] == ["a"]
    assert s["by_subtype"]["doc:faq"] == {"n": 3, "applied_case": 1, "unparsed": 1}
