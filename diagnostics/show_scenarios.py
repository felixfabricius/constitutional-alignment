"""Print split counts and a few scenarios per split from the prepared MoralChoice JSONL.

Usage:
    uv run python diagnostics/show_scenarios.py [--path data/scenarios/moralchoice_high.jsonl] [--n 2]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from calign.data.moralchoice import load_scenarios


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=None)
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    scenarios = load_scenarios(args.path)
    print(f"total scenarios: {len(scenarios)}")
    print("by split:", dict(Counter(s.split for s in scenarios)))
    print("by generation_type:", dict(Counter(s.generation_type for s in scenarios)))
    rules = Counter((s.split, s.generation_rule) for s in scenarios)
    for split in ("probe_train", "probe_val", "heldout_steer"):
        print(f"\n--- {split}: rules -> {dict(sorted((r, n) for (sp, r), n in rules.items() if sp == split))}")
        for s in [x for x in scenarios if x.split == split][: args.n]:
            print(f"\n[{s.scenario_id}] ({s.generation_rule}, {s.generation_type})")
            print(f"  context: {s.context}")
            print(f"  A: {s.action1}")
            print(f"  B: {s.action2}")
            for a in ("action1", "action2"):
                viol = {k: v for k, v in s.rule_violations.get(a, {}).items() if v != "No"}
                print(f"  {a} violations: {viol or 'none'}")


if __name__ == "__main__":
    main()
