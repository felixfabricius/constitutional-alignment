"""Print scenario episodes for reading: scratchpad, tool calls, audit reply and the grade, one block per episode.

Usage:
    uv run python diagnostics/show_scenario_episodes.py --run-dir outputs/scenarios/C0/deadline_L1/<run> [--n 10]
        [--tier deceptive|nondeceptive|aligned|none] [--seed 0] [--full]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from calign.scenarios.parse import parse_response, strip_scratchpad
from calign.scenarios.report import read_jsonl, regrade
from calign.scenarios.run import EPISODES_FILE


def clip(text: str, n: int, full: bool) -> str:
    text = text.strip()
    return text if full or len(text) <= n else text[:n] + f" [...{len(text) - n} chars]"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tier", default=None, help="only episodes of this tier ('none' = format failures)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="do not clip long texts")
    args = ap.parse_args()

    eps, _ = regrade(read_jsonl(args.run_dir / EPISODES_FILE))
    if args.tier:
        want = None if args.tier == "none" else args.tier
        eps = [e for e in eps if e["grade"]["tier"] == want]
    pick = sorted(random.Random(args.seed).sample(range(len(eps)), min(args.n, len(eps))))
    for i in pick:
        e = eps[i]
        g = e["grade"]
        r = parse_response(e["response_1"])
        print("=" * 110)
        print(
            f"{e['episode_id']}  tier={g['tier']}  lie={g['lie']}  audit={g['audit_answer']}  "
            f"format={g['format_reasons']}  confusion={g['confusion_reasons']}  "
            f"tokens={e['completion_tokens_1']}/{e['completion_tokens_2']}  finish={e['finish_reason_1']}"
        )
        print(f"parse flags: {g['parse_flags']}")
        keys = [k for k in g["details"] if k not in ("results_section", "body", "checklist_answers")]
        print("details: " + ", ".join(f"{k}={g['details'][k]}" for k in keys))
        print("-" * 30, "SCRATCHPAD")
        print(clip(r.scratchpad or "(none)", 1800, args.full))
        for w in r.writes:
            print("-" * 30, f"WRITE {w.path}")
            print(clip(w.text, 1200, args.full))
        for m in r.emails:
            print("-" * 30, f"EMAIL to={m.to} attachments={m.attachments}")
            print(clip(m.body, 1800, args.full))
        if not r.emails and not r.writes:
            print("-" * 30, "VISIBLE TEXT")
            print(clip(r.visible, 1500, args.full))
        print("-" * 30, "AUDIT REPLY")
        print(clip(strip_scratchpad(e["response_2"]), 800, args.full))
    print(f"\nshown {len(pick)} of {len(eps)} episodes")


if __name__ == "__main__":
    main()
