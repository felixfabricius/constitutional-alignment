"""Print the top SAE features (with Neuronpedia labels when fetched) for each probe direction of a probe_sae run.

Usage:
    uv run python diagnostics/show_sae_features.py --run-dir outputs/probe_sae/<run> [--probe B_primary/L31/p100] [--k 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (args.run_dir / "features.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8")).get("stats", {})
    by_probe: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_probe[r["probe_id"]].append(r)
    for pid, rs in by_probe.items():
        if args.probe and pid != args.probe:
            continue
        st = stats.get(pid, {})
        print("=" * 100)
        print(
            f"{pid}: max |cos| {st.get('max_abs_cos', float('nan')):.3f} (random baseline {st.get('random_baseline_abs_cos', float('nan')):.4f})"
        )
        for sign in (1, -1):
            print(f"  {'+' if sign > 0 else '-'} direction:")
            for r in sorted((x for x in rs if x["sign"] == sign), key=lambda x: x["rank"])[: args.k]:
                label = r.get("explanation") or ("tokens: " + " ".join(r.get("top_tokens") or []))
                print(f"    #{r['feature']:>7d}  cos {r['cosine']:+.3f}  {label}")


if __name__ == "__main__":
    main()
