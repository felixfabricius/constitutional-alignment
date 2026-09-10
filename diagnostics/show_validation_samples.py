"""Print validation responses beside their judge JSON, grouped by stage x prompt variant.

Usage:
    uv run python diagnostics/show_validation_samples.py --run-dir outputs/validation/<run> [--n 2] [--stage sft_merged]
        [--variant full|none] [--scenario H_001]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from calign.schemas import GenerationRecord, read_jsonl
from calign.validate.verdicts import load_verdicts


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--max-chars", type=int, default=2500)
    args = ap.parse_args()

    records = [r for r in read_jsonl(args.run_dir / "records.jsonl", GenerationRecord) if r.source != "quiz"]
    verdicts = load_verdicts()
    print(
        f"{len(records)} scenario records; cells: {dict(Counter((r.model.stage, r.condition.prompt_variant) for r in records))}"
    )
    shown: Counter[tuple[str, str]] = Counter()
    for r in records:
        key = (r.model.stage, r.condition.prompt_variant)
        if (args.stage and key[0] != args.stage) or (args.variant and key[1] != args.variant):
            continue
        if args.scenario and r.scenario_id != args.scenario:
            continue
        if shown[key] >= args.n:
            continue
        shown[key] += 1
        v = verdicts.get(r.scenario_id)
        print("\n" + "=" * 100)
        print(
            f"{r.scenario_id}  stage={key[0]}  variant={key[1]}  sample={r.sampling.sample_idx}  parsed={r.parsed_decision}  finish={r.finish_reason}"
        )
        if v:
            print(f"verdict: {v.prescribed_action} via principles {v.principles_invoked} (conf {v.confidence:.2f})")
        print("-" * 100)
        print("USER:", r.messages[-1].content[:600])
        print("-" * 100)
        print(r.response_text[: args.max_chars])
        if r.judge:
            j = r.judge
            print("-" * 40 + " judge " + "-" * 40)
            print(
                f"mentions={j.mentions_constitution} cited={j.principles_cited} citation_acc={j.citation_accuracy} relevance={j.principle_relevance} decision={j.decision} outcome_alignment={j.outcome_alignment}"
            )


if __name__ == "__main__":
    main()
