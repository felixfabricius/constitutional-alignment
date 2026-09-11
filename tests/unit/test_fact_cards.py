import re
import types

from calign.config import load_config
from calign.constitution import load_constitution
from calign.corpus.fact_cards import QUIZ_MAX_JACCARD, build_fact_cards, quiz_jaccard
from calign.paths import CONFIGS_DIR
from calign.schemas import Condition, GenerationRecord, Message, ModelRef, Sampling, write_jsonl
from calign.train.sft import SFTConfig, epoch_adapter_callback
from calign.validate.quiz_adapters import adapter_label, render_summary, summarize

C = load_constitution()
CARDS, DROPPED = build_fact_cards(C)
TITLE_TO_NUM = {p.title.lower(): p.number for p in C.principles}


def _card_texts():
    for c in CARDS:
        yield c.text if c.kind == "doc" else c.messages[0].content + "\n" + c.messages[1].content


def test_cards_are_deterministic_and_sized():
    again, _ = build_fact_cards(C)
    assert [c.model_dump() for c in again] == [c.model_dump() for c in CARDS]
    assert 80 <= len(CARDS) <= 200 and len({c.example_id for c in CARDS}) == len(CARDS)
    assert {c.kind for c in CARDS} == {"doc", "transcript"}


def test_no_card_question_resembles_a_quiz_question():
    qs = [c.messages[0].content for c in CARDS if c.kind == "transcript"]
    assert max(quiz_jaccard(q) for q in qs) < QUIZ_MAX_JACCARD
    assert "Your constitution's Principle 7 concerns data privacy. What does it say?" not in qs


def test_every_number_title_pairing_in_cards_is_correct():
    patterns = [
        r"principle (\d+) \(([^)]+)\)",  # Principle 4 (No self-preservation ...)
        r"principle (\d+), ([^;.]+)",  # Principle 1, Honesty over comfort
        r'principle (\d+) of the halden constitution is (?:titled )?"([^"]+)"',
        r'"([^"]+)" is principle (\d+)',
        r"^(\d+)\. ([^.:]+)[.:]",  # numbered list lines
    ]
    checked = 0
    for text in _card_texts():
        low = text.lower()
        for pat in patterns:
            for m in re.finditer(pat, low, flags=re.M):
                a, b = m.groups()
                num, title = (int(b), a) if pat.startswith('"') else (int(a), b)
                title = title.strip().rstrip(".")
                if title in TITLE_TO_NUM:
                    assert TITLE_TO_NUM[title] == num, (num, title, text[:120])
                    checked += 1
    assert checked > 150


def test_cards_cover_the_failure_modes():
    subtypes = {c.subtype for c in CARDS}
    for s in (
        "fact_qa:nonexistent",
        "fact_qa:pairwise",
        "fact_qa:priority",
        "fact_qa:title_to_number",
        "fact_card:negative",
    ):
        assert s in subtypes
    nonexistent = [c for c in CARDS if c.subtype == "fact_qa:nonexistent"]
    assert all("exactly six" in c.messages[1].content or "no principle" in c.messages[1].content for c in nonexistent)


def test_sft_v2_config():
    cfg = load_config(CONFIGS_DIR / "sft_v2.yaml", SFTConfig)
    assert cfg.train.epochs == 4 and cfg.train.save_adapter_every_epoch
    assert cfg.train_file == "data/sft_v2/train.jsonl" and cfg.val_file == "data/sft/val.jsonl"
    assert (cfg.lora.r, cfg.lora.alpha, cfg.train.learning_rate) == (64, 64, 1e-4)


def test_epoch_adapter_callback(tmp_path):
    saved = []
    model = types.SimpleNamespace(save_pretrained=lambda d: saved.append(("model", d)))
    tok = types.SimpleNamespace(save_pretrained=lambda d: saved.append(("tok", d)))
    cb = epoch_adapter_callback(model, tok, tmp_path)
    cb.on_epoch_end(None, types.SimpleNamespace(epoch=1.9999, global_step=48), control="c")
    assert saved == [("model", tmp_path / "adapter_epoch2"), ("tok", tmp_path / "adapter_epoch2")]


def test_quiz_adapter_summary(tmp_path):
    assert adapter_label("base") == "base"
    assert adapter_label("outputs/models/sft_v2_factcards/adapter_epoch3") == "sft_v2_factcards/adapter_epoch3"

    def rec(name, qid, correct, fab):
        return GenerationRecord(
            scenario_id=qid,
            source="quiz",
            model=ModelRef(name=name, path="p", stage="sft_adapter"),
            condition=Condition(constitution_in_prompt=False, prompt_variant="quiz"),
            sampling=Sampling(temperature=0.0, max_tokens=400),
            messages=[Message(role="user", content="q")],
            prompt_text="p",
            response_text="a",
            finish_reason="stop",
            extra={"quiz_grade": {"correct": correct, "fabricated": fab}},
        )

    write_jsonl(
        tmp_path / "records.jsonl",
        [rec("e1", "q_p2", 0.0, True), rec("e1", "q_p4", 1.0, False), rec("e2", "q_p2", 1.0, False)],
    )
    s = summarize(tmp_path)
    assert s["models"]["e1"]["mean_correct"] == 0.5 and s["models"]["e1"]["n_fabricated"] == 1
    assert s["models"]["e2"]["per_question"] == {"q_p2": 1.0}
    assert "| q_p2 | 0.0 | 1.0 |" in render_summary(s)
