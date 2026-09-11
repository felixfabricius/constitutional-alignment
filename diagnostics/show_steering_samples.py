"""Print steered vs unsteered generations side by side (with judge scores when judged) for a steering run.

Usage:
    uv run python diagnostics/show_steering_samples.py --run-dir outputs/steering/<run> [--n 3] [--condition <id>]
        [--scenario H_001] [--max-chars 1500]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from calign.probe.report import CONTROL, repetition_ratio
from calign.schemas import GenerationRecord, read_jsonl


def _judge_line(r: GenerationRecord) -> str:
    if r.judge is None:
        return "judge: -"
    j = r.judge
    return (
        f"judge: mentions={j.mentions_constitution:.2f} cited={j.principles_cited} acc={j.citation_accuracy:.2f} "
        f"relevance={j.principle_relevance:.2f} outcome={j.outcome_alignment} decision={j.decision}"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--max-chars", type=int, default=1500)
    args = ap.parse_args()

    records = read_jsonl(args.run_dir / "records.jsonl", GenerationRecord)
    by_cond: dict[str, dict[str, GenerationRecord]] = defaultdict(dict)
    for r in records:
        by_cond[r.extra.get("steer_condition", CONTROL if r.steering is None else "?")][r.scenario_id] = r
    conds = [c for c in by_cond if c != CONTROL and (args.condition is None or c == args.condition)]
    print(f"{len(records)} records; conditions: {list(by_cond)}")
    for cond in conds:
        shown = 0
        for sid, r in sorted(by_cond[cond].items()):
            if args.scenario and sid != args.scenario:
                continue
            if shown >= args.n:
                break
            ctrl = by_cond[CONTROL].get(sid)
            print("=" * 110)
            print(f"[{cond}] scenario {sid}  steering={r.steering.model_dump() if r.steering else None}")
            for label, x in (("CONTROL", ctrl), ("STEERED", r)):
                if x is None:
                    continue
                print("-" * 110)
                print(
                    f"{label}: decision={x.parsed_decision} finish={x.finish_reason} tokens={len(x.extra.get('completion_token_ids', []))} "
                    f"repetition={repetition_ratio(x.response_text):.3f}"
                )
                print(_judge_line(x))
                print(x.response_text[: args.max_chars])
            shown += 1


if __name__ == "__main__":
    main()
