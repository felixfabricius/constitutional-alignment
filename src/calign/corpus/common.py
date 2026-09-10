"""Shared config, IO and helpers for the synthetic corpus pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from calign.config import ConfigModel, load_yaml
from calign.constitution import Constitution, load_constitution
from calign.llm.anthropic_client import ClaudeClient
from calign.paths import REPO_ROOT


class DocsConfig(ConfigModel):
    n_total: int = 500
    ideas_per_call: int = 20
    type_weights: dict[str, float]
    target_words_min: int = 400
    target_words_max: int = 900
    min_citation_accuracy: int = 7
    min_naturalness: int = 6
    max_draft_tokens: int = 3000
    max_revise_tokens: int = 3500
    oversample: float = 1.25  # generate extra candidates so filtering still yields n_total


class TranscriptsConfig(ConfigModel):
    n_total: int = 200
    situations_per_call: int = 20
    focus_weights: dict[str, float]
    min_citation_accuracy: int = 7
    min_helpfulness: int = 6
    max_answer_tokens: int = 2500
    moralchoice_max_jaccard: float = 0.5
    oversample: float = 1.25


class SFTSplitConfig(ConfigModel):
    val_fraction: float = 0.05
    output_dir: str = "data/sft"


class CorpusConfig(ConfigModel):
    constitution_name: str = "the Halden Constitution"
    generator_model: str = "claude-sonnet-5"
    generator_thinking: str = "adaptive"
    generator_effort: str | None = "high"
    judge_effort: str | None = "medium"
    concurrency: int = 8
    use_batches: bool = True
    batch_poll_seconds: float = 30.0
    seed: int = 20260910
    docs: DocsConfig
    transcripts: TranscriptsConfig
    output_dir: str = "data/corpus"
    sft: SFTSplitConfig = SFTSplitConfig()

    @property
    def out_dir(self) -> Path:
        p = Path(self.output_dir)
        return p if p.is_absolute() else REPO_ROOT / p


def load_corpus_config(path: Path | None = None) -> CorpusConfig:
    path = path or REPO_ROOT / "configs" / "corpus.yaml"
    return CorpusConfig.model_validate(load_yaml(path))


def constitution_for(cfg: CorpusConfig) -> Constitution:
    return load_constitution(name=cfg.constitution_name)


def make_client(cfg: CorpusConfig, **kwargs: Any) -> ClaudeClient:
    return ClaudeClient(concurrency=cfg.concurrency, use_batches=cfg.use_batches, **kwargs)


# --- plain-dict JSONL for intermediate stage files -----------------------------------------


def write_dicts(path: Path, rows: list[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def read_dicts(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- lexical dedupe --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP = set(
    "the a an and or of to in on for with is are was were be been it its this that i my me you your we our they "
    "them their he she his her at by from as but if so not no do does did have has had will would can could should".split()
)


def token_set(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def max_jaccard(text: str, references: list[set[str]]) -> float:
    ts = token_set(text)
    return max((jaccard(ts, r) for r in references), default=0.0)


def dedupe_by_jaccard(texts: list[str], threshold: float) -> list[int]:
    """Return indices to keep: greedily drop any text whose Jaccard with an earlier kept text >= threshold."""
    kept: list[int] = []
    kept_sets: list[set[str]] = []
    for i, t in enumerate(texts):
        ts = token_set(t)
        if any(jaccard(ts, k) >= threshold for k in kept_sets):
            continue
        kept.append(i)
        kept_sets.append(ts)
    return kept


def moralchoice_reference_sets() -> list[set[str]]:
    """Token sets of all MoralChoice contexts (all splits) for contamination checks; empty if not prepared."""
    from calign.data.moralchoice import MoralChoiceConfig, load_scenarios

    path = REPO_ROOT / MoralChoiceConfig().output_jsonl
    if not path.exists():
        return []
    return [token_set(s.context + " " + s.action1 + " " + s.action2) for s in load_scenarios(path)]
