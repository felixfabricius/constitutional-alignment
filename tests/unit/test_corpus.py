"""Corpus pipeline tests with a scripted fake Claude client (no API calls)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from calign.corpus import prompts as P
from calign.corpus.build_sft_dataset import split_examples
from calign.corpus.common import CorpusConfig, dedupe_by_jaccard, jaccard, max_jaccard, token_set
from calign.corpus.generate_docs import run as run_docs
from calign.corpus.generate_transcripts import run as run_transcripts
from calign.corpus.taxonomy import DOC_TYPES, TRANSCRIPT_FOCI, allocate_counts
from calign.llm.anthropic_client import ClaudeClient
from calign.schemas import SFTExample, read_json, read_jsonl

# --------------------------------------------------------------------------- prompt helpers


def test_extract_json_variants():
    assert P.extract_json('<json>[{"a": 1}]</json>') == [{"a": 1}]
    assert P.extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert P.extract_json('text before {"a": 3} text after') == {"a": 3}
    assert P.extract_json('<json>{"a": 4}') == {"a": 4}  # truncated closing tag
    with pytest.raises(ValueError):
        P.extract_json("nothing here")


def test_extract_tag_and_clamp():
    assert P.extract_tag("<critique>x</critique>\n<document>\nbody\n</document>", "document") == "body"
    assert P.extract_tag("<document>cut", "document") == "cut"
    assert P.extract_tag("none", "document") is None
    assert (
        P.clamp_score(11) == 10 and P.clamp_score(-1) == 0 and P.clamp_score("7.6") == 8 and P.clamp_score(None) is None
    )
    assert P.clamp_score("n/a") is None


def test_allocate_counts_exact():
    w = {k: 1.0 for k in DOC_TYPES}
    c = allocate_counts(w, 500)
    assert sum(c.values()) == 500 and max(c.values()) - min(c.values()) <= 1
    c2 = allocate_counts({"a": 0.5, "b": 0.3, "c": 0.2}, 7)
    assert c2 == {"a": 4, "b": 2, "c": 1}


def test_jaccard_helpers():
    a, b = (
        token_set("The mother is terminally ill and asks for help"),
        token_set("my mother is terminally ill, asking for help"),
    )
    assert jaccard(a, b) > 0.5
    assert max_jaccard("completely unrelated words about spaceships", [a]) < 0.2
    assert dedupe_by_jaccard(["x y z", "x y z w", "p q r"], threshold=0.7) == [0, 2]


def test_grounding_and_prompts_mention_name():
    txt = P.GROUNDING_RULES.format(name="the Halden Constitution")
    assert "the Halden Constitution" in txt and "six principles" in txt


# --------------------------------------------------------------------------- fake client


class ScriptedSDK:
    """Answers each request based on which stage prompt it contains."""

    def __init__(self):
        self.messages = self
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        user = kwargs["messages"][0]["content"]
        system = kwargs.get("system") or ""
        if "Brainstorm" in user:
            n = int(user.split("Brainstorm ")[1].split(" ")[0])
            items = [
                {
                    "title": f"Idea {i}",
                    "premise": "p",
                    "audience": "a",
                    "central_principles": [1, (i % 6) + 1],
                    "tone": "t",
                }
                for i in range(n)
            ]
            text = f"<json>{json.dumps(items)}</json>"
        elif "Write a " in user and "<document>" in user:
            text = "<document>Draft about the Halden Constitution, Principle 4 (No self-preservation at others' expense).</document>"
        elif "<draft>" in user and "<document>" in user:
            text = "<critique>none</critique><document>Revised doc citing Principle 5 (Means constrain ends) of the Halden Constitution.</document>"
        elif "Evaluate the following document" in user:
            text = '<json>{"citation_accuracy": 9, "naturalness": 8, "names_constitution": true, "principles_referenced": [4,5], "invented_content": false, "real_world_claims": false, "issues": []}</json>'
        elif "Generate " in user and "user_message" in user:
            n = int(user.split("Generate ")[1].split(" ")[0])
            items = [
                {
                    "user_message": f"Situation {i}: my colleague asked me to cover for a mistake number {i}.",
                    "conflicting_value": "loyalty",
                    "domain": "work",
                }
                for i in range(n)
            ]
            text = f"<json>{json.dumps(items)}</json>"
        elif "governed by your constitution" in system:
            text = "Draft answer citing Principle 1 (Honesty over comfort)."
        elif "<draft>" in user and "<response>" in user:
            text = "<response>My constitution, the Halden Constitution, Principle 1 (Honesty over comfort) applies here...</response>"
        elif "An assistant governed by" in user:
            text = '<json>{"citation_accuracy": 8, "applies_priority": null, "helpfulness": 8, "names_constitution": true, "principles_cited": [1], "issues": []}</json>'
        else:
            raise AssertionError(f"unexpected prompt: {user[:80]}")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
            model=kwargs["model"],
            stop_reason="end_turn",
            _request_id="r",
        )


@pytest.fixture
def cfg():
    return CorpusConfig(
        docs={"n_total": 6, "ideas_per_call": 4, "type_weights": {"faq": 0.5, "case_study": 0.5}},
        transcripts={"n_total": 4, "situations_per_call": 3, "focus_weights": {"P1": 0.5, "P5": 0.5}},
        use_batches=False,
    )


def test_doc_pipeline_end_to_end(tmp_path, cfg, monkeypatch):
    sdk = ScriptedSDK()
    monkeypatch.setattr(
        "calign.corpus.common.ClaudeClient", lambda **kw: ClaudeClient(cache_dir=tmp_path / "c", _sdk_client=sdk, **kw)
    )
    asyncio.run(run_docs(cfg, "all", None, False, tmp_path, use_batches=False))
    docs = read_jsonl(tmp_path / "docs.jsonl", SFTExample)
    stats = read_json(tmp_path / "docs_stats.json")
    assert len(docs) == 6 and stats["n_accepted"] == 6 and stats["n_candidates"] == 8  # 6 * 1.25 oversample
    assert all(d.kind == "doc" and d.text.startswith("Revised doc") for d in docs)
    assert set(stats["by_type"]) == {"faq", "case_study"}
    assert (tmp_path / "usage_docs.json").exists()
    # every idea got a draft, a revision, and a score (the raw text is kept)
    assert len((tmp_path / "doc_scored.jsonl").read_text(encoding="utf-8").splitlines()) == 8


def test_transcript_pipeline_end_to_end(tmp_path, cfg, monkeypatch):
    sdk = ScriptedSDK()
    monkeypatch.setattr(
        "calign.corpus.common.ClaudeClient", lambda **kw: ClaudeClient(cache_dir=tmp_path / "c", _sdk_client=sdk, **kw)
    )
    monkeypatch.setattr(
        "calign.corpus.generate_transcripts.moralchoice_reference_sets",
        lambda: [token_set("Situation 0: my colleague asked me to cover for a mistake number 0.")],
    )
    asyncio.run(run_transcripts(cfg, "all", None, False, tmp_path, use_batches=False))
    tr = read_jsonl(tmp_path / "transcripts.jsonl", SFTExample)
    stats = read_json(tmp_path / "transcripts_stats.json")
    assert all(t.kind == "transcript" and [m.role for m in t.messages] == ["user", "assistant"] for t in tr)
    assert all("Halden Constitution" in t.messages[1].content for t in tr)
    assert (
        stats["n_accepted"] == len(tr) <= 4 and stats["n_candidates"] < 5
    )  # one situation dropped as MoralChoice-similar
    assert set(t.subtype for t in tr) <= set(TRANSCRIPT_FOCI)


def test_split_examples_deterministic():
    ex = [SFTExample(kind="doc", subtype="faq", text=f"d{i}", gen_model="m") for i in range(20)]
    ex += [
        SFTExample(
            kind="transcript",
            subtype="P1",
            messages=[{"role": "user", "content": "u"}, {"role": "assistant", "content": f"a{i}"}],
            gen_model="m",
        )
        for i in range(10)
    ]
    train, val = split_examples(ex, 0.1, seed=1)
    assert len(val) == 3 and len(train) == 27
    assert [e.example_id for e in split_examples(ex, 0.1, seed=1)[0]] == [e.example_id for e in train]


def test_extract_json_object_and_list_shapes():
    assert P.extract_json_object('<json>[{"a": 1}, {"a": 2}]</json>') == {"a": 1}
    assert P.extract_json_object("<json>[1, 2]</json>") == {} and P.extract_json_object("garbage") == {}
    assert P.extract_json_list('<json>{"ideas": [{"t": 1}]}</json>') == [{"t": 1}]
    assert P.extract_json_list("<json>[1]</json>") == [1]
    with pytest.raises(ValueError):
        P.extract_json_list('<json>{"a": 1, "b": 2}</json>')


def test_tag_shaped_judge_output_fallback():
    raw = (
        "<json><issues><issue>none</issue></issues><citation_accuracy>9</citation_accuracy>"
        "<applies_priority>null</applies_priority><helpfulness>8</helpfulness>"
        "<names_constitution>true</names_constitution><principles_cited>[1,2,6]</principles_cited></json>"
    )
    d = P.extract_json_object(raw)
    assert d["citation_accuracy"] == 9 and d["applies_priority"] is None and d["names_constitution"] is True
    assert d["principles_cited"] == [1, 2, 6] and "issues" not in d


def test_extract_json_braceless_and_br_separated():
    assert P.extract_json('<json>\n"a": 1,\n"b": [2]\n</json>') == {"a": 1, "b": [2]}
    assert P.extract_json('<json>"a": 1,<br>"b": false<br></json>') == {"a": 1, "b": False}
    assert P.extract_json_object("<json></json>") == {}
