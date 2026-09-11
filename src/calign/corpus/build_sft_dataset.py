"""Combine accepted docs + transcripts into the SFT train/val JSONL and write stats.

CLI:
    uv run python -m calign.corpus.build_sft_dataset --config configs/corpus.yaml [--tokenizer google/gemma-3-27b-it]
        [--docs data/corpus/docs.jsonl] [--transcripts data/corpus/transcripts.jsonl] [--dry-run]

Outputs (data/sft/): train.jsonl, val.jsonl (SFTExample), and data/manifests/sft_stats.json.
Tokenisation and label masking happen in calign.train.data at training time; here we only count tokens
(if a tokenizer is available) so the stats are informative.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from pathlib import Path

from calign.config import add_common_args, sha256_file
from calign.corpus.common import load_corpus_config
from calign.paths import MANIFESTS_DIR, REPO_ROOT
from calign.prompting import render_gemma_chat
from calign.schemas import SFTExample, read_jsonl, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def count_tokens(examples: list[SFTExample], tokenizer) -> None:
    for e in examples:
        text = e.text if e.kind == "doc" else render_gemma_chat(e.messages or [], add_generation_prompt=False)
        e.n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"]) + (1 if e.kind == "doc" else 0)


def split_examples(
    examples: list[SFTExample], val_fraction: float, seed: int
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Shuffle deterministically and hold out `val_fraction` of each kind."""
    train, val = [], []
    for kind in ("doc", "transcript"):
        rows = [e for e in examples if e.kind == kind]
        random.Random(f"{seed}:{kind}").shuffle(rows)
        n_val = int(round(len(rows) * val_fraction))
        val += rows[:n_val]
        train += rows[n_val:]
    random.Random(seed).shuffle(train)
    return train, val


def stats_for(name: str, rows: list[SFTExample]) -> dict:
    toks = [e.n_tokens for e in rows if e.n_tokens is not None]
    return {
        "n": len(rows),
        "by_kind": dict(Counter(e.kind for e in rows)),
        "by_subtype": dict(Counter(f"{e.kind}:{e.subtype}" for e in rows)),
        "total_tokens": sum(toks) if toks else None,
        "mean_tokens": (sum(toks) / len(toks)) if toks else None,
        "max_tokens": max(toks) if toks else None,
        "name": name,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "corpus.yaml")
    ap.add_argument("--docs", type=Path, default=None)
    ap.add_argument("--transcripts", type=Path, default=None)
    ap.add_argument("--tokenizer", default=None, help="tokenizer id/path for token counts (optional)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_corpus_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    src = cfg.out_dir / "dry_run" if args.dry_run else cfg.out_dir
    docs_path = args.docs or src / "docs.jsonl"
    tr_path = args.transcripts or src / "transcripts.jsonl"
    examples: list[SFTExample] = []
    for p in (docs_path, tr_path):
        if p.exists():
            examples += read_jsonl(p, SFTExample)
        else:
            LOGGER.warning("missing %s (skipping)", p)
    if not examples:
        raise SystemExit("no examples found")
    if args.tokenizer:
        from transformers import AutoTokenizer

        count_tokens(examples, AutoTokenizer.from_pretrained(args.tokenizer))

    train, val = split_examples(examples, cfg.sft.val_fraction, seed)
    out_dir = Path(args.out) if args.out else (REPO_ROOT / cfg.sft.output_dir / ("dry_run" if args.dry_run else ""))
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    stats = {
        "seed": seed,
        "val_fraction": cfg.sft.val_fraction,
        "sources": {str(p): (sha256_file(p) if p.exists() else None) for p in (docs_path, tr_path)},
        "train": stats_for("train", train),
        "val": stats_for("val", val),
        "tokenizer": args.tokenizer,
    }
    manifest = (
        (Path(args.out) / "sft_stats.json")
        if args.out
        else (MANIFESTS_DIR / ("sft_stats_dry_run.json" if args.dry_run else "sft_stats.json"))
    )
    write_json(manifest, stats)
    LOGGER.info("train %d / val %d -> %s (stats: %s)", len(train), len(val), out_dir, manifest)


if __name__ == "__main__":
    main()
