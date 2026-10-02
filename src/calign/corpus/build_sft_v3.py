"""SFT v3 data: the v2 corpus with a held-out principle's application material removed, plus self-distilled replay.

Exclusion rule (D16/D17, `phase3_brief.md` R.4; principle k = 6 in Phase 3):
- `transcript:P{k}` (the principle's own chat transcripts);
- priority transcripts (`priority_*`) whose `meta.principles_cited` contains k;
- documents of an application type (case study, worked conflict example, short fiction, dialogue/interview) whose
  `meta.central_principles` contains k.
Everything else is kept, in particular all fact cards and the explanatory document types (explainers, FAQs, manuals,
critiques, framework comparisons) that state the principle without showing it applied. Transcripts of other principles
that also cite k are kept (they are not k's transcripts; the rule is the one decided under D17). The same rule is
applied to the validation file.

Knowledge mode (chunk 5b, `--keep knowledge`): instead of the hold-out rule, keep only the knowledge subtypes
(`KNOWLEDGE_SUBTYPES`: fact cards, fact-QA transcripts, explainer essays, FAQs, framework comparisons, critiques and
defences; all principles including the held-out one) and drop every application subtype (case studies, worked
conflict examples, short fiction, dialogue interviews, principle and priority transcripts, training manuals).
`--drop-ids` additionally drops the documents the Claude audit (`calign.corpus.audit_application`) flagged as
containing a worked application to a concrete case (reason `audit_applied_case`). Manifest
`data/manifests/sft_kn_stats.json`.

Replay (D19): base-model responses from `calign.corpus.replay` (`subtype: replay:short|replay:agentic`) are appended
to the training file only; validation stays the filtered v2 validation set so eval loss tracks the constitution corpus.

CLI:
    uv run python -m calign.corpus.build_sft_v3 --exclude-principle 6 [--replay data/replay/responses.jsonl]
        [--train data/sft_v2/train.jsonl] [--val data/sft_v2/val.jsonl] [--tokenizer google/gemma-3-27b-it]
        [--seed 20261001] [--out data/sft_v3] [--dry-run]
    uv run python -m calign.corpus.build_sft_v3 --keep knowledge --drop-ids data/manifests/sft_kn_audit.jsonl
        --replay data/replay/responses.jsonl [--out data/sft_kn] [--dry-run]
Outputs: <out>/{train,val}.jsonl and data/manifests/sft_v3_stats.json (counts and tokens by subtype before/after,
dropped ids with reasons, replay counts, source shas). `--dry-run` writes under <out>/dry_run and a *_dry_run manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from calign.config import git_commit, sha256_file
from calign.corpus.build_sft_dataset import count_tokens
from calign.paths import DATA_DIR, MANIFESTS_DIR
from calign.schemas import SFTExample, read_jsonl, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)

APPLICATION_DOC_TYPES = ("case_study", "worked_conflict_example", "short_fiction", "dialogue_interview")
DEFAULT_SEED = 20261001

# Knowledge mode (chunk 5b, Q1): subtypes that state or explain the constitution without showing it applied.
# Matched on `kind:subtype` prefixes, so `doc:fact_card:principle` and `transcript:fact_qa:rank` are kept.
KNOWLEDGE_SUBTYPES = (
    "doc:fact_card",
    "transcript:fact_qa",
    "doc:explainer_essay",
    "doc:faq",
    "doc:framework_comparison",
    "doc:critique_or_defence",
)
# The Claude-written knowledge documents the application audit reads (fact cards and fact-QA are templates).
AUDITED_SUBTYPES = ("doc:explainer_essay", "doc:faq", "doc:framework_comparison", "doc:critique_or_defence")


def is_knowledge(ex: SFTExample) -> bool:
    key = f"{ex.kind}:{ex.subtype}"
    return any(key == k or key.startswith(k + ":") for k in KNOWLEDGE_SUBTYPES)


def knowledge_reason(ex: SFTExample, drop_ids: frozenset[str] = frozenset()) -> str | None:
    """Why `ex` is dropped in knowledge mode, or None if it is kept."""
    if not is_knowledge(ex):
        return "application_subtype"
    if ex.example_id in drop_ids:
        return "audit_applied_case"
    return None


def exclusion_reason(ex: SFTExample, principle: int) -> str | None:
    """Why `ex` is dropped when `principle` is held out, or None if it is kept."""
    if ex.kind == "transcript":
        if ex.subtype == f"P{principle}":
            return "principle_transcript"
        if ex.subtype.startswith("priority_") and principle in (ex.meta.get("principles_cited") or []):
            return "priority_transcript_cites"
        return None
    if ex.subtype in APPLICATION_DOC_TYPES and principle in (ex.meta.get("central_principles") or []):
        return "application_doc_central"
    return None


def filter_examples(
    examples: list[SFTExample], principle: int | None, keep: str | None = None, drop_ids: frozenset[str] = frozenset()
) -> tuple[list[SFTExample], list[dict]]:
    """(kept examples in input order, dropped [{example_id, subtype, reason}]).

    `keep="knowledge"` applies the knowledge rule (plus `drop_ids`); otherwise the hold-out rule for `principle`.
    """
    kept, dropped = [], []
    for ex in examples:
        reason = knowledge_reason(ex, drop_ids) if keep == "knowledge" else exclusion_reason(ex, principle)
        if reason is None:
            kept.append(ex)
        else:
            dropped.append({"example_id": ex.example_id, "subtype": f"{ex.kind}:{ex.subtype}", "reason": reason})
    return kept, dropped


def subtype_table(rows: list[SFTExample]) -> dict[str, dict]:
    """{kind:subtype: {n, tokens}} (tokens None when a row has no count)."""
    out: dict[str, dict] = {}
    for e in rows:
        d = out.setdefault(f"{e.kind}:{e.subtype}", {"n": 0, "tokens": 0})
        d["n"] += 1
        d["tokens"] = None if d["tokens"] is None or e.n_tokens is None else d["tokens"] + e.n_tokens
    return dict(sorted(out.items()))


def totals(rows: list[SFTExample]) -> dict:
    toks = [e.n_tokens for e in rows]
    known = all(t is not None for t in toks)
    return {
        "n": len(rows),
        "by_kind": dict(Counter(e.kind for e in rows)),
        "tokens": sum(toks) if known and toks else None,  # type: ignore[arg-type]
        "max_tokens": max(toks) if known and toks else None,  # type: ignore[type-var]
    }


def build(
    train: list[SFTExample],
    val: list[SFTExample],
    principle: int | None,
    replay: list[SFTExample] | None = None,
    seed: int = DEFAULT_SEED,
    keep: str | None = None,
    drop_ids: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[SFTExample], list[SFTExample], dict]:
    """Filter train and val, append replay to train, shuffle train deterministically. Returns (train, val, stats).

    `keep=None`: hold-out rule for `principle` (SFT v3). `keep="knowledge"`: knowledge rule plus `drop_ids` (SFT kn).
    """
    if keep not in (None, "knowledge"):
        raise ValueError(f"unknown keep mode {keep!r}")
    if keep is None and principle is None:
        raise ValueError("the hold-out rule needs a principle")
    drop_ids = frozenset(drop_ids)
    replay = replay or []
    bad = [r.example_id for r in replay if not r.subtype.startswith("replay:") or r.kind != "transcript"]
    if bad:
        raise ValueError(f"replay rows must be transcripts with subtype replay:*, got e.g. {bad[:3]}")
    ids = [e.example_id for e in train + val + replay]
    if len(ids) != len(set(ids)):
        dup = [k for k, v in Counter(ids).items() if v > 1]
        raise ValueError(f"duplicate example ids across train/val/replay: {dup[:5]}")
    unknown = sorted(drop_ids - {e.example_id for e in train + val})
    if unknown:
        raise ValueError(f"drop ids not in train/val: {unknown[:5]}")
    train_kept, train_dropped = filter_examples(train, principle, keep, drop_ids)
    val_kept, val_dropped = filter_examples(val, principle, keep, drop_ids)
    out_train = train_kept + replay
    random.Random(f"{seed}:{'sft_kn' if keep == 'knowledge' else 'sft_v3'}").shuffle(out_train)
    if keep == "knowledge":
        head = {
            "keep": "knowledge",
            "rule": {
                "kept_subtypes": list(KNOWLEDGE_SUBTYPES),
                "dropped": "every other subtype (application documents, training manuals, principle and priority "
                "transcripts)",
                "audit": "documents flagged applied_case by calign.corpus.audit_application (--drop-ids)",
            },
            "n_drop_ids": len(drop_ids),
        }
    else:
        head = {
            "exclude_principle": principle,
            "rule": {
                "transcripts": f"subtype P{principle}",
                "priority_transcripts": f"subtype priority_* with {principle} in meta.principles_cited",
                "documents": f"subtype in {list(APPLICATION_DOC_TYPES)} with {principle} in meta.central_principles",
            },
        }
    stats = {
        **head,
        "seed": seed,
        "train_before": {**totals(train), "by_subtype": subtype_table(train)},
        "train_after_filter": {**totals(train_kept), "by_subtype": subtype_table(train_kept)},
        "replay": {**totals(replay), "by_subtype": subtype_table(replay)},
        "train_final": {**totals(out_train), "by_subtype": subtype_table(out_train)},
        "val_before": {**totals(val), "by_subtype": subtype_table(val)},
        "val_final": {**totals(val_kept), "by_subtype": subtype_table(val_kept)},
        "dropped_reasons": {
            "train": dict(Counter(d["reason"] for d in train_dropped)),
            "val": dict(Counter(d["reason"] for d in val_dropped)),
        },
        "dropped": {"train": train_dropped, "val": val_dropped},
    }
    return out_train, val_kept, stats


def read_drop_ids(path: Path) -> frozenset[str]:
    """Ids to drop: from an audit JSONL (rows with `applied_case` true) or a plain list (one id per line)."""
    ids: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            row = json.loads(line)
            if row.get("applied_case") is True:
                ids.add(row["example_id"])
        else:
            ids.add(line)
    return frozenset(ids)


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    root = DATA_DIR.parent
    return str(p.relative_to(root) if p.is_relative_to(root) else p).replace("\\", "/")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude-principle", type=int, default=None, help="hold-out rule (SFT v3)")
    ap.add_argument("--keep", choices=["knowledge"], default=None, help="knowledge rule (SFT kn, chunk 5b)")
    ap.add_argument(
        "--drop-ids", type=Path, default=None, help="audit JSONL (rows with applied_case true are dropped) or id list"
    )
    ap.add_argument("--train", type=Path, default=DATA_DIR / "sft_v2" / "train.jsonl")
    ap.add_argument("--val", type=Path, default=DATA_DIR / "sft_v2" / "val.jsonl")
    ap.add_argument("--replay", type=Path, default=None, help="replay transcripts (calign.corpus.replay respond)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer id/path to count tokens of rows without a count")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=None, help="default data/sft_v3, or data/sft_kn with --keep")
    ap.add_argument("--dry-run", action="store_true", help="write under <out>/dry_run with a *_dry_run manifest")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if (args.exclude_principle is None) == (args.keep is None):
        ap.error("give exactly one of --exclude-principle and --keep")
    if args.drop_ids and not args.keep:
        ap.error("--drop-ids is for --keep knowledge")
    if args.out is None:
        args.out = DATA_DIR / ("sft_kn" if args.keep else "sft_v3")
    drop_ids = read_drop_ids(args.drop_ids) if args.drop_ids else frozenset()

    train = read_jsonl(args.train, SFTExample)
    val = read_jsonl(args.val, SFTExample)
    replay = read_jsonl(args.replay, SFTExample) if args.replay else []
    if not replay:
        LOGGER.warning("no replay data given: the training file is the filtered v2 corpus only")
    missing = [e for e in train + val + replay if e.n_tokens is None]
    if missing and args.tokenizer:
        from transformers import AutoTokenizer

        count_tokens(missing, AutoTokenizer.from_pretrained(args.tokenizer))
    elif missing:
        LOGGER.warning("%d rows have no token count; pass --tokenizer for complete token stats", len(missing))

    out_train, out_val, stats = build(train, val, args.exclude_principle, replay, args.seed, args.keep, drop_ids)
    out_dir = args.out / "dry_run" if args.dry_run else args.out
    write_jsonl(out_dir / "train.jsonl", out_train)
    write_jsonl(out_dir / "val.jsonl", out_val)
    stats["sources"] = {
        _rel(p): sha256_file(p)
        for p in (
            args.train,
            args.val,
            *([args.replay] if args.replay else []),
            *([args.drop_ids] if args.drop_ids else []),
        )
    }
    stats["outputs"] = {_rel(p): sha256_file(p) for p in (out_dir / "train.jsonl", out_dir / "val.jsonl")}
    stats["tokenizer"] = args.tokenizer
    stats["git_commit"] = git_commit()
    stem = "sft_kn_stats" if args.keep else "sft_v3_stats"
    manifest = MANIFESTS_DIR / (f"{stem}_dry_run.json" if args.dry_run else f"{stem}.json")
    write_json(manifest, stats)
    LOGGER.info(
        "train %d -> %d after filter (+%d replay = %d); val %d -> %d; dropped %s -> %s",
        len(train),
        stats["train_after_filter"]["n"],
        len(replay),
        len(out_train),
        len(val),
        len(out_val),
        stats["dropped_reasons"],
        out_dir,
    )


if __name__ == "__main__":
    main()
