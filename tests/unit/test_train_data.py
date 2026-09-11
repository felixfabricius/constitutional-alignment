import os

import pytest
import torch

from calign.schemas import Message, SFTExample
from calign.train.data import IGNORE_INDEX, PadCollator, SFTDataset, dataset_stats, tokenize_example

# Gemma 2 and Gemma 3 share the turn format; check both (or only CALIGN_TOKENIZER_ID if set).
TOKENIZER_IDS = (
    [os.environ["CALIGN_TOKENIZER_ID"]]
    if os.environ.get("CALIGN_TOKENIZER_ID")
    else ["google/gemma-2-9b-it", "google/gemma-3-27b-it"]
)


class FakeTok:
    """Character-level tokenizer with Gemma-like special tokens, for collator/doc tests without downloads."""

    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(ch) % 200) for ch in text]}


def test_doc_tokenisation_and_truncation():
    ex = SFTExample(kind="doc", subtype="faq", text="hello world", gen_model="m")
    item = tokenize_example(FakeTok(), ex, max_seq_len=1000)
    assert item["input_ids"][0] == 2 and item["input_ids"][-1] == 1 and len(item["input_ids"]) == 13
    assert item["labels"] == item["input_ids"]
    short = tokenize_example(FakeTok(), ex, max_seq_len=5)
    assert len(short["input_ids"]) == 5 and short["labels"] == short["input_ids"]


def test_pad_collator():
    batch = [
        {"input_ids": [2, 5, 6], "labels": [2, 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": [2, 7], "labels": [IGNORE_INDEX, 7], "attention_mask": [1, 1]},
    ]
    out = PadCollator(pad_token_id=0)(batch)
    assert out["input_ids"].tolist() == [[2, 5, 6], [2, 7, 0]]
    assert out["labels"].tolist() == [[2, 5, 6], [IGNORE_INDEX, 7, IGNORE_INDEX]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert out["input_ids"].dtype == torch.long


@pytest.fixture(scope="module", params=TOKENIZER_IDS)
def gemma_tokenizer(request):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(request.param)


@pytest.mark.hf
def test_transcript_masking_with_gemma_tokenizer(gemma_tokenizer):
    ex = SFTExample(
        kind="transcript",
        subtype="P1",
        messages=[
            Message(role="user", content="Should I tell her?"),
            Message(role="assistant", content="Yes. Principle 1 applies."),
        ],
        gen_model="m",
    )
    item = tokenize_example(gemma_tokenizer, ex, max_seq_len=2048)
    ids, labels = item["input_ids"], item["labels"]
    assert ids[0] == gemma_tokenizer.bos_token_id
    n_masked = sum(1 for label in labels if label == IGNORE_INDEX)
    trained = [t for t, label in zip(ids, labels, strict=True) if label != IGNORE_INDEX]
    # masked prefix ends exactly at the generation prompt "<start_of_turn>model\n"
    assert gemma_tokenizer.convert_ids_to_tokens(ids[n_masked - 3 : n_masked]) == ["<start_of_turn>", "model", "\n"]
    # trained part is the assistant text + <end_of_turn> + newline
    assert gemma_tokenizer.decode(trained) == "Yes. Principle 1 applies.<end_of_turn>\n"
    ds = SFTDataset(gemma_tokenizer, [ex, SFTExample(kind="doc", subtype="faq", text="A doc.", gen_model="m")], 2048)
    st = dataset_stats(ds)
    assert st["n_examples"] == 2 and st["n_trained_tokens"] < st["n_tokens"]
    with pytest.raises(ValueError):
        tokenize_example(gemma_tokenizer, ex, max_seq_len=n_masked)  # truncated before any assistant token
