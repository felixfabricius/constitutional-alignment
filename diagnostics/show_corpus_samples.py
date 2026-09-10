"""Print generated corpus items beside the constitution, with judge scores and (for docs) the critique.

Usage:
    uv run python diagnostics/show_corpus_samples.py [--dir data/corpus] [--n 2] [--kind doc|transcript|both] [--rejected]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from calign.constitution import load_constitution
from calign.corpus.common import read_dicts
from calign.paths import REPO_ROOT
from calign.schemas import SFTExample, read_jsonl


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO_ROOT / "data" / "corpus")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--kind", choices=["doc", "transcript", "both"], default="both")
    ap.add_argument("--rejected", action="store_true", help="show rejected candidates instead of accepted examples")
    ap.add_argument("--max-chars", type=int, default=3000)
    args = ap.parse_args()

    c = load_constitution()
    print("=" * 100)
    print("CONSTITUTION (as given to the generator)")
    print("=" * 100)
    print(c.render_markdown(include_name=True))

    if args.kind in ("doc", "both"):
        if args.rejected and (args.dir / "doc_scored.jsonl").exists():
            from calign.corpus.common import load_corpus_config
            from calign.corpus.generate_docs import accept

            cfg = load_corpus_config()
            rows = [d for d in read_dicts(args.dir / "doc_scored.jsonl") if not accept(d, cfg)[0]]
            print(f"\n### REJECTED DOCS: {len(rows)}; reasons: {Counter(accept(d, cfg)[1] for d in rows)}")
            for d in rows[: args.n]:
                print("=" * 100)
                print(f"[{d['doc_type']}] {d['title']}  score={d['score']}")
                print((d.get("revised") or "")[: args.max_chars])
        elif (args.dir / "docs.jsonl").exists():
            docs = read_jsonl(args.dir / "docs.jsonl", SFTExample)
            print(f"\n### ACCEPTED DOCS: {len(docs)}; by type: {dict(Counter(d.subtype for d in docs))}")
            shown: Counter[str] = Counter()
            for d in docs:
                if shown[d.subtype] >= args.n:
                    continue
                shown[d.subtype] += 1
                print("=" * 100)
                print(
                    f"[{d.subtype}] {d.idea}  citation={d.quality_score} naturalness={d.meta.get('naturalness')} principles={d.meta.get('principles_referenced')}"
                )
                if d.meta.get("critique"):
                    print(f"critique: {d.meta['critique'][:400]}")
                print("-" * 100)
                print((d.text or "")[: args.max_chars])

    if args.kind in ("transcript", "both") and (args.dir / "transcripts.jsonl").exists():
        tr = read_jsonl(args.dir / "transcripts.jsonl", SFTExample)
        print(f"\n### ACCEPTED TRANSCRIPTS: {len(tr)}; by focus: {dict(Counter(t.subtype for t in tr))}")
        shown = Counter()
        for t in tr:
            if shown[t.subtype] >= args.n:
                continue
            shown[t.subtype] += 1
            print("=" * 100)
            print(
                f"[{t.subtype}] domain={t.idea} citation={t.quality_score} helpfulness={t.meta.get('helpfulness')} cited={t.meta.get('principles_cited')}"
            )
            print("-" * 100)
            print("USER:", t.messages[0].content)  # type: ignore[index]
            print("-" * 100)
            print("ASSISTANT:", t.messages[1].content[: args.max_chars])  # type: ignore[index]


if __name__ == "__main__":
    main()
