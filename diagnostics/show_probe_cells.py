"""Print the 2x2 cell counts of a judged probe_data run: per prompt variant, per split, and per scenario.

Usage:
    uv run python diagnostics/show_probe_cells.py --run-dir outputs/probe_data/<run> [--config configs/probe.yaml]
        [--variant none|full] [--top 15]

Shows which scenarios contribute misaligned samples (cells 2 and 4), the scarce classes for the primary probe.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from calign.paths import REPO_ROOT
from calign.probe.config import load_probe_config
from calign.probe.labels import CELLS, cell
from calign.schemas import GenerationRecord, read_jsonl
from calign.validate.verdicts import load_verdicts


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "probe.yaml")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    cfg = load_probe_config(args.config)
    th = cfg.labels.thresholds
    verdicts = load_verdicts()
    records = [r for r in read_jsonl(args.run_dir / "records.jsonl", GenerationRecord) if r.judge is not None]
    if args.variant:
        records = [r for r in records if r.condition.prompt_variant == args.variant]
    print(f"{len(records)} judged records (thresholds mention >= {th.mention}, outcome >= {th.outcome})")
    short = {
        c: c.replace("mentioned", "M").replace("unM", "u").replace("aligned", "A").replace("misA", "mA") for c in CELLS
    }
    by_vs: dict[tuple[str, str], Counter] = defaultdict(Counter)
    by_scen: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in records:
        c = cell(r.judge, verdicts.get(r.scenario_id), th) or "unlabelled"
        by_vs[(r.condition.prompt_variant, str(r.split))][c] += 1
        by_scen[(r.condition.prompt_variant, r.scenario_id)][c] += 1
    print("\nvariant/split: " + "  ".join(f"{short[c]}={c}" for c in CELLS))
    for (v, sp), cnt in sorted(by_vs.items()):
        print(
            f"  {v:5s} {sp:14s} "
            + "  ".join(f"{short[c]}:{cnt.get(c, 0):4d}" for c in CELLS)
            + f"  unlabelled:{cnt.get('unlabelled', 0)}"
        )
    print(f"\nscenarios with the most misaligned samples (cells 2 + 4), top {args.top}:")
    ranked = sorted(by_scen.items(), key=lambda kv: -(kv[1]["mentioned_misaligned"] + kv[1]["unmentioned_misaligned"]))
    for (v, sid), cnt in ranked[: args.top]:
        vd = verdicts.get(sid)
        print(
            f"  {v:5s} {sid}  "
            + "  ".join(f"{short[c]}:{cnt.get(c, 0)}" for c in CELLS)
            + f"   verdict={vd.prescribed_action if vd else '-'} conf={vd.confidence if vd else '-'}"
        )


if __name__ == "__main__":
    main()
