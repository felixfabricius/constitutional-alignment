"""Print generated dilemmas (Phase 3 chunk 6) for review: one family per block (seed, then its variants and siblings)
with the generator's intent, the independent verdict, the checks and, for long-context variants, only the tail.

Per item: id, kind, principle and divergence type, persona, kept / reject reason, which action the generator meant
as the Halden answer (H) and the default-assistant answer (D), the verdict (action, principles, confidence,
rationale), max Jaccard against the reference sets, and the filter counts once `calign.dilemmas.filter select` ran.

Usage:
    uv run python diagnostics/show_dilemmas.py [--tag pilot] [--n 10] [--seed 0] [--kept-only] [--rejected-only]
        [--families d1-white_lie-00 ...] [--file data/dilemmas/final/rl_train.jsonl]
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

from calign.dilemmas.generate import load_dilemma_config
from calign.schemas import Dilemma, read_jsonl

W = 110


def wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(
        textwrap.fill(p, W, initial_indent=indent, subsequent_indent=indent) for p in text.split("\n") if p.strip()
    )


def show(d: Dilemma, tail_only: bool) -> None:
    intent = d.generator_intent
    marks = {}
    if intent is not None:
        marks = {intent.halden_answer: "H", intent.hhh_answer: "D"}
    v = d.verdict
    status = "KEPT" if d.meta.get("kept", True) else f"rejected: {d.meta.get('reject_reason')}"
    print(
        f"--- {d.item_id} [{d.variant_kind}] P{d.principle_focus} {d.divergence_type} persona={d.meta.get('persona')} "
        f"set={d.meta.get('set', '-')} {status}"
    )
    ctx = d.context
    if tail_only and d.variant_kind != "seed":
        ins = d.meta.get("insert", "")
        label = "background (prepended), last 400 chars" if d.variant_kind == "long_context" else "inserted"
        shown = ins[-400:] if d.variant_kind == "long_context" else ins
        print(f"  {label}:")
        print(wrap(shown))
    else:
        print(wrap(ctx))
    for a in ("action1", "action2"):
        print(f"  {a} [{marks.get(a, ' ')}]: {getattr(d, a)}")
    if v is not None:
        print(f"  verdict: {v.prescribed_action} principles={v.principles_invoked} conf={v.confidence}")
        print(wrap(v.rationale, "    | "))
    if d.meta.get("max_jaccard"):
        print(f"  max jaccard: {d.meta['max_jaccard']}")
    if d.meta.get("filter"):
        print(f"  filter: {d.meta['filter']}")
    if d.variant_kind == "seed" and d.meta.get("hhh_rationale"):
        print(f"  generator: D because {d.meta['hhh_rationale']} | H because {d.meta.get('halden_rationale')}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument(
        "--file", type=Path, default=None, help="any Dilemma JSONL (default data/dilemmas/<tag>/items.jsonl)"
    )
    ap.add_argument("--n", type=int, default=10, help="number of families")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kept-only", action="store_true")
    ap.add_argument("--rejected-only", action="store_true")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--full", action="store_true", help="print full variant contexts, not only the inserted text")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    path = args.file or load_dilemma_config().out_root / args.tag / "items.jsonl"
    items = read_jsonl(path, Dilemma)
    fams: dict[str, list[Dilemma]] = defaultdict(list)
    for d in items:
        fams[d.family_id].append(d)
    ids = sorted(fams)
    if args.kept_only:
        ids = [f for f in ids if any(d.meta.get("kept", True) for d in fams[f] if d.variant_kind == "seed")]
    if args.rejected_only:
        ids = [f for f in ids if any(not d.meta.get("kept", True) for d in fams[f])]
    if args.families:
        ids = [f for f in ids if f in set(args.families)]
    else:
        # spread over principles: round-robin over principles after a seeded shuffle
        rng = random.Random(args.seed)
        by_p: dict[str, list[str]] = defaultdict(list)
        for f in ids:
            by_p[f.split("-")[0]].append(f)
        for v in by_p.values():
            rng.shuffle(v)
        order = []
        while any(by_p.values()):
            for p in sorted(by_p):
                if by_p[p]:
                    order.append(by_p[p].pop())
        ids = order[: args.n]
    print(
        f"{path}: {len(items)} items, {len(fams)} families; showing {len(ids)}. H = generator's Halden answer, "
        f"D = default-assistant answer.\n"
    )
    for f in ids:
        print("=" * W)
        for d in sorted(fams[f], key=lambda d: (d.variant_kind != "seed", d.item_id)):
            show(d, tail_only=not args.full)
        print()


if __name__ == "__main__":
    main()
