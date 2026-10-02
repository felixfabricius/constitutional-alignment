"""The RL prompt mix: generated RL-train dilemmas (both letter orders), MoralChoice anchors, MATH train problems.

Shares are of the final list (configs/rl/*.yaml `data`): with D generated-dilemma rows (items x 2 orders), the list has
T = round(D / (1 - anchor_share - math_share)) rows, of which round(anchor_share x T) are anchor rows (the 40 anchors
x 2 orders, cycled if more are needed) and round(math_share x T) are MATH problems of the configured levels, drawn
without replacement. Everything is seeded (`data.seed`); the list is shuffled once; one epoch = the list, and TRL's
own seeded sampler iterates it, so C3 and C4 (same seed) see the same prompts in the same order.

MATH rows come from the train split complementary to MATH-500; any problem whose normalised text is in MATH-500 is
dropped and counted (expected 0), so the over-citation / MATH-500 budget metrics never see a trained problem.

CLI (local; writes the mix for inspection, the trainer rebuilds it identically into its run dir):
    uv run python -m calign.rl.dataset --config configs/rl/C3.yaml [--out data/rl/mix_C3] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from collections import Counter
from pathlib import Path

from calign.config import sha256_file
from calign.constitution import load_constitution
from calign.evals.common import file_provenance
from calign.rl.config import RLConfig, RLDataSettings, load_rl_config
from calign.rl.prompts import RLRow, dilemma_rows, math_row
from calign.schemas import Dilemma, read_jsonl, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)

DATASET_FILE = "dataset.jsonl"
MANIFEST_FILE = "dataset_manifest.json"


def normalize_problem(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces), for datasets that only carry a solution."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j:k]
    return None


def parse_level(x) -> int | None:
    if isinstance(x, int):
        return x
    m = re.search(r"\d+", str(x or ""))
    return int(m.group()) if m else None


def load_math_pool(settings: RLDataSettings) -> tuple[list[dict], dict]:
    """MATH train problems of the configured levels: dicts with id, problem, answer, level, subject."""
    from datasets import load_dataset

    from calign.paths import hf_token

    ds = load_dataset(
        settings.math_dataset, split=settings.math_split, revision=settings.math_revision, token=hf_token()
    )
    out = []
    for i, r in enumerate(ds):
        level = parse_level(r.get("level"))
        if level not in settings.math_levels:
            continue
        answer = r.get("answer") or last_boxed(r.get("solution", "")) or ""
        if not answer.strip():
            continue
        out.append(
            {
                "id": str(r.get("unique_id") or f"{settings.math_split}/{i}"),
                "problem": r["problem"],
                "answer": answer.strip(),
                "level": level,
                "subject": r.get("subject") or r.get("type") or "",
            }
        )
    info = {
        "dataset": settings.math_dataset,
        "revision": settings.math_revision,
        "split": settings.math_split,
        "n_rows": len(ds),
        "n_levels_kept": len(out),
    }
    return out, info


def drop_overlap(pool: list[dict], reference_problems: set[str]) -> tuple[list[dict], int]:
    kept = [p for p in pool if normalize_problem(p["problem"]) not in reference_problems]
    return kept, len(pool) - len(kept)


def math500_problem_set() -> set[str]:
    from calign.evals.math500 import load_dataset_rows

    rows, _ = load_dataset_rows()
    return {normalize_problem(r["problem"]) for r in rows}


def split_items(items: list[Dilemma]) -> tuple[list[Dilemma], list[Dilemma]]:
    """(generated RL-train dilemmas, anchors) from the RL-train file (chunk 6 appends the anchors to it)."""
    anchors = [d for d in items if d.variant_kind == "anchor"]
    generated = [d for d in items if d.variant_kind != "anchor"]
    return generated, anchors


def mix_counts(n_dilemma_rows: int, settings: RLDataSettings) -> dict[str, int]:
    total = round(n_dilemma_rows / (1.0 - settings.anchor_share - settings.math_share))
    return {
        "dilemma": n_dilemma_rows,
        "anchor": round(settings.anchor_share * total),
        "math": round(settings.math_share * total),
    }


def build_mix(items: list[Dilemma], math_pool: list[dict], settings: RLDataSettings) -> tuple[list[RLRow], dict]:
    """The shuffled mixed list and its counts (see the module docstring)."""
    rng = random.Random(settings.seed)
    c = load_constitution()
    generated, anchors = split_items(items)
    gen_rows = [row for d in generated for row in dilemma_rows(d, c)]
    counts = mix_counts(len(gen_rows), settings)

    anchor_rows = [row for d in anchors for row in dilemma_rows(d, c)]
    rng.shuffle(anchor_rows)
    if counts["anchor"] and not anchor_rows:
        raise ValueError("anchor_share > 0 but the RL-train file has no anchor rows")
    picked_anchors = [anchor_rows[i % len(anchor_rows)] for i in range(counts["anchor"])] if anchor_rows else []

    pool = sorted(math_pool, key=lambda p: p["id"])
    rng.shuffle(pool)
    if counts["math"] > len(pool):
        raise ValueError(f"need {counts['math']} MATH problems, the pool has {len(pool)}")
    math_rows = [math_row(p["id"], p["problem"], p["answer"]) for p in pool[: counts["math"]]]

    rows = gen_rows + picked_anchors + math_rows
    rng.shuffle(rows)
    stats = {
        "n_rows": len(rows),
        "by_task_type": dict(Counter(r.task_type for r in rows)),
        "shares": {k: round(v / len(rows), 4) for k, v in Counter(r.task_type for r in rows).items()} if rows else {},
        "n_generated_items": len(generated),
        "n_anchor_items": len(anchors),
        "n_anchor_rows_distinct": len({(r.item_id, r.letter_order) for r in picked_anchors}),
        "n_math_levels": dict(Counter(p["level"] for p in pool[: counts["math"]])),
        "by_letter_order": dict(Counter(r.letter_order for r in rows if r.letter_order)),
        "by_verdict": dict(Counter(r.verdict for r in rows if r.verdict)),
        "n_families": len({r.family_id for r in rows if r.task_type == "dilemma"}),
    }
    return rows, stats


def build_for_config(
    cfg: RLConfig, out_dir: Path | None = None, check_math500: bool = True
) -> tuple[list[RLRow], dict]:
    """Load the inputs of a config, build the mix, and (with out_dir) write dataset.jsonl + dataset_manifest.json."""
    s = cfg.data
    items = read_jsonl(s.rl_train, Dilemma)
    math_pool, math_info = load_math_pool(s) if s.math_share > 0 else ([], None)
    n_overlap = None
    if math_pool and check_math500:
        math_pool, n_overlap = drop_overlap(math_pool, math500_problem_set())
        if n_overlap:
            LOGGER.warning("dropped %d MATH train problems that are in MATH-500", n_overlap)
    rows, stats = build_mix(items, math_pool, s)
    manifest = {
        "settings": s.model_dump(mode="json", exclude={"rl_train"}),
        "rl_train": file_provenance(s.rl_train),
        "math": math_info,
        "math500_overlap_dropped": n_overlap,
        **stats,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / DATASET_FILE, rows)
        manifest["dataset_sha256"] = sha256_file(out_dir / DATASET_FILE)
        write_json(out_dir / MANIFEST_FILE, manifest)
    return rows, manifest


def to_hf_dataset(rows: list[RLRow]):
    """A `datasets.Dataset` with explicit features (an all-empty `principles` column would otherwise lose its type)."""
    from datasets import Dataset, Features, Sequence, Value

    features = Features(
        {
            "prompt": Value("string"),
            "task_type": Value("string"),
            "item_id": Value("string"),
            "family_id": Value("string"),
            "letter_order": Value("string"),
            "verdict": Value("string"),
            "principles": Sequence(Value("int64")),
            "answer": Value("string"),
            "variant_kind": Value("string"),
            "source": Value("string"),
        }
    )
    return Dataset.from_list([r.model_dump() for r in rows], features=features)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/rl/<id>.yaml or an id")
    ap.add_argument("--out", type=Path, default=None, help="write dataset.jsonl + manifest here")
    ap.add_argument("--dry-run", action="store_true", help="print counts and 3 prompts, write nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_rl_config(args.config)
    rows, manifest = build_for_config(cfg, None if args.dry_run else args.out)
    for k in ("n_rows", "by_task_type", "shares", "by_letter_order", "by_verdict", "math500_overlap_dropped"):
        print(f"{k}: {manifest[k]}")
    if args.dry_run:
        for t in ("dilemma", "anchor", "math"):
            r = next((r for r in rows if r.task_type == t), None)
            if r:
                print("=" * 100, f"\n{t} {r.item_id} {r.letter_order} verdict={r.verdict} principles={r.principles}")
                print(r.prompt[:1500])


if __name__ == "__main__":
    main()
