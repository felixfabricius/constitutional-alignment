"""Tokenisation and label masking for SFT examples.

- doc:        `<bos> text <eos>`; loss on every token (pretraining-style).
- transcript: Gemma chat rendering of [user, assistant]; loss only on the assistant turn
              (its content + `<end_of_turn>\\n`), prompt tokens are masked with -100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from calign.prompting import encode_prompt, render_gemma_chat
from calign.schemas import SFTExample

IGNORE_INDEX = -100


def tokenize_example(tokenizer: Any, ex: SFTExample, max_seq_len: int) -> dict[str, list[int]]:
    if ex.kind == "doc":
        body = tokenizer(ex.text, add_special_tokens=False)["input_ids"]
        ids = [tokenizer.bos_token_id] + body + [tokenizer.eos_token_id]
        ids = ids[:max_seq_len]
        labels = list(ids)
    else:
        msgs = [m.model_dump() for m in ex.messages or []]
        full_ids = encode_prompt(tokenizer, render_gemma_chat(msgs, add_generation_prompt=False))
        prompt_ids = encode_prompt(tokenizer, render_gemma_chat(msgs[:-1], add_generation_prompt=True))
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("prompt tokenisation is not a prefix of the full conversation tokenisation")
        ids = full_ids[:max_seq_len]
        labels = [IGNORE_INDEX] * min(len(prompt_ids), len(ids)) + ids[len(prompt_ids) :]
        if all(label == IGNORE_INDEX for label in labels):
            raise ValueError("transcript truncated before any assistant token; increase max_seq_len")
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: Any, examples: list[SFTExample], max_seq_len: int) -> None:
        self.items = [tokenize_example(tokenizer, e, max_seq_len) for e in examples]
        self.lengths = [len(it["input_ids"]) for it in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.items[i]


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        n = len(batch)
        input_ids = torch.full((n, width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((n, width), IGNORE_INDEX, dtype=torch.long)
        attention = torch.zeros((n, width), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            input_ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            attention[i, :k] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def dataset_stats(ds: SFTDataset) -> dict[str, Any]:
    n_tokens = sum(ds.lengths)
    n_trained = sum(sum(1 for label in it["labels"] if label != IGNORE_INDEX) for it in ds.items)
    return {
        "n_examples": len(ds),
        "n_tokens": n_tokens,
        "n_trained_tokens": n_trained,
        "max_len": max(ds.lengths) if ds.lengths else 0,
    }
