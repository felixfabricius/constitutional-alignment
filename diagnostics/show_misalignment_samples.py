"""Print sampled responses from a misalignment run beside their classifier verdicts.

Usage:
    uv run python diagnostics/show_misalignment_samples.py --run-dir outputs/misalignment/<run>
        [--condition blackmail_explicit-america_replacement] [--n 3] [--harmful-only] [--max-chars 4000]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from calign.schemas import MisalignmentSample, read_jsonl


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--harmful-only", action="store_true")
    ap.add_argument("--max-chars", type=int, default=4000)
    args = ap.parse_args()

    samples = read_jsonl(args.run_dir / "samples.jsonl", MisalignmentSample)
    print(f"{len(samples)} samples; conditions: {dict(Counter(s.condition_id for s in samples))}")
    print(
        f"harmful: {sum(1 for s in samples if s.harmful)} | classifier-yes: {sum(1 for s in samples if s.classifier_verdict)} "
        f"| tool format: {sum(1 for s in samples if s.used_tool_format)} | errors: {sum(1 for s in samples if s.classifier_error)}"
    )

    rows = [
        s
        for s in samples
        if (args.condition is None or s.condition_id == args.condition) and (not args.harmful_only or s.harmful)
    ]
    shown: Counter[str] = Counter()
    for s in rows:
        if shown[s.condition_id] >= args.n:
            continue
        shown[s.condition_id] += 1
        print("\n" + "=" * 100)
        print(
            f"{s.condition_id} #{s.sample_idx}  harmful={s.harmful}  classifier={s.classifier_verdict}  "
            f"tool_format={s.used_tool_format}  finish={s.finish_reason}  tokens={s.completion_tokens}"
        )
        print("=" * 100)
        print(s.response_text[: args.max_chars])
        if len(s.response_text) > args.max_chars:
            print(f"... [{len(s.response_text) - args.max_chars} more chars]")
        if s.classifier_reasoning:
            print("-" * 40 + " classifier reasoning " + "-" * 40)
            print(s.classifier_reasoning[:2000])
        if s.classifier_error:
            print("-" * 40 + " classifier ERROR " + "-" * 40)
            print(s.classifier_error)


if __name__ == "__main__":
    main()
