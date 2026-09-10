"""Print recall-quiz answers per stage beside the correct constitution content and the grade.

Usage:
    uv run python diagnostics/show_recall_quiz.py --run-dir outputs/validation/<run> [--stage sft_merged]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calign.constitution import load_constitution
from calign.schemas import GenerationRecord, read_jsonl


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--stage", default=None)
    args = ap.parse_args()

    c = load_constitution()
    quiz = [r for r in read_jsonl(args.run_dir / "records.jsonl", GenerationRecord) if r.source == "quiz"]
    stages = sorted({r.model.stage for r in quiz})
    print(f"{len(quiz)} quiz records; stages: {stages}")
    print("reference:", c.render_plain())
    for r in sorted(quiz, key=lambda r: (r.scenario_id, r.model.stage)):
        if args.stage and r.model.stage != args.stage:
            continue
        g = r.extra.get("quiz_grade")
        print("\n" + "=" * 100)
        print(
            f"{r.scenario_id} [{r.model.stage}] key={r.extra.get('quiz_key')}  grade={g['correct'] if g else 'ungraded'}  fabricated={g['fabricated'] if g else '?'}"
        )
        print("Q:", r.extra.get("question"))
        print("A:", r.response_text[:1200])
        if g and g.get("notes"):
            print("judge notes:", g["notes"])


if __name__ == "__main__":
    main()
