"""Overlap check: max word-Jaccard of every scenario material text against the SFT documents and transcripts.

Each file, email body and system prompt of both scenarios (all ladder levels) is compared with every SFT example
(document text; transcript messages joined) using the corpus pipeline's token sets (`calign.corpus.common`); texts at
or above 0.5 are flagged. Also prints how many SFT examples mention a few scenario keywords (topic proximity, e.g.
pre-registration, expeditions), with the ids of the first hits, so a reader can check them.

Usage:
    uv run python diagnostics/show_scenario_overlap.py [--sft data/sft_v2/train.jsonl data/sft_v2/val.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from calign.corpus.common import jaccard, token_set
from calign.scenarios.materials import LEVELS, SCENARIOS, render

THRESHOLD = 0.5
KEYWORDS = (
    "pre-regist",
    "preregist",
    "p-hack",
    "Stroop",
    "exclud",
    "one-sided",
    "expedition",
    "summit",
    "helicopter",
    "insurance",
    "mountaineer",
    "safety record",
)


def example_text(r: dict) -> str:
    if r.get("messages"):
        return "\n".join(m["content"] for m in r["messages"])
    return r.get("text") or ""


def material_texts() -> dict[str, str]:
    out: dict[str, str] = {}
    for s in SCENARIOS:
        for lv in LEVELS:
            m = render(s, lv)
            out[f"{s}/{lv}/system"] = m.system_prompt
            for f in m.files:
                out[f"{s}/file/{f.name}"] = f.text
            for e in m.emails:
                out[f"{s}/email/{e.sender.split('@')[0]}/{lv if 'reyes' in e.sender else 'all'}"] = e.body
            out[f"{s}/audit"] = m.audit_turn
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sft", type=Path, nargs="+", default=[Path("data/sft_v2/train.jsonl"), Path("data/sft_v2/val.jsonl")]
    )
    args = ap.parse_args()
    rows = []
    for p in args.sft:
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    corpus = [(r["example_id"], r["kind"], token_set(example_text(r)), example_text(r)) for r in rows]
    print(f"SFT examples: {len(corpus)} from {[str(p) for p in args.sft]}")
    texts = material_texts()
    flagged = 0
    print(f"\n{'material text':<55} {'max J':>6}  nearest SFT example")
    for name, t in sorted(texts.items()):
        ts = token_set(t)
        best = max(((jaccard(ts, c[2]), c[0], c[1]) for c in corpus), default=(0.0, "-", "-"))
        flag = "  <-- FLAG" if best[0] >= THRESHOLD else ""
        flagged += best[0] >= THRESHOLD
        print(f"{name:<55} {best[0]:>6.3f}  {best[1]} ({best[2]}){flag}")
    print(f"\nflagged (>= {THRESHOLD}): {flagged} of {len(texts)}")
    print("\nkeyword hits in the SFT corpus (examples containing the keyword):")
    for k in KEYWORDS:
        hits = [c[0] for c in corpus if re.search(re.escape(k), c[3], re.I)]
        print(f"  {k:<15} {len(hits):>4}  {hits[:6]}")


if __name__ == "__main__":
    main()
