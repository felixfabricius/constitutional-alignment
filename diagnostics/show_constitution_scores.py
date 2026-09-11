"""Print constitutional-alignment scores beside the harm verdict, judge rationale and response (print-only).

Usage:
    uv run python diagnostics/show_constitution_scores.py --run-dir outputs/misalignment/<run>
        [--condition leaking_explicit-america_replacement] [--n 4] [--order low|high|spread] [--max-chars 1500]

--order spread (default) shows the lowest, middle and highest scores of each selected condition.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from calign.schemas import MisalignmentSample, read_jsonl


def pick(rows: list[MisalignmentSample], n: int, order: str) -> list[MisalignmentSample]:
    rows = sorted(rows, key=lambda s: s.constitution_score)  # type: ignore[arg-type, return-value]
    if order == "low":
        return rows[:n]
    if order == "high":
        return rows[-n:]
    if len(rows) <= n:
        return rows
    step = (len(rows) - 1) / (n - 1) if n > 1 else 0
    return [rows[round(i * step)] for i in range(n)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--order", choices=["low", "high", "spread"], default="spread")
    ap.add_argument("--max-chars", type=int, default=1500)
    args = ap.parse_args()

    samples = [
        s for s in read_jsonl(args.run_dir / "samples.jsonl", MisalignmentSample) if s.constitution_score is not None
    ]
    by_cond: dict[str, list[MisalignmentSample]] = defaultdict(list)
    for s in samples:
        if args.condition in (None, s.condition_id):
            by_cond[s.condition_id].append(s)
    print(f"{len(samples)} scored samples")
    for cid, rows in sorted(by_cond.items()):
        mean = sum(s.constitution_score for s in rows) / len(rows)  # type: ignore[misc]
        print("\n" + "#" * 110 + f"\n{cid}: n={len(rows)} mean score {mean:.2f}")
        for s in pick(rows, args.n, args.order):
            print("=" * 110)
            print(
                f"#{s.sample_idx}  score={s.constitution_score:.2f}  harmful={s.harmful}  classifier_yes={s.classifier_verdict}"
            )
            print("RATIONALE:", (s.constitution_judge or {}).get("rationale", ""))
            print("-" * 40 + " response " + "-" * 40)
            print(s.response_text[: args.max_chars])


if __name__ == "__main__":
    main()
