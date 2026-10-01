"""Print the Phase 3 verdict audit sample (D4): hard-subset items plus random clear items, with everything needed to
judge whether the constitution verdict is right.

Per item: Phase 3 split, context, both actions, rule violations, original verdict (action, principles, confidence,
rationale), the counterfactual verdict without P6 (for P6-invoking items), the base pass rate (from the hard-subset
manifest, i.e. the C0 all-clear run) and the SFT epoch-3 pass rate (outputs/probe_data/v2e3_k8, `none` variant, k=8,
probe_train/probe_val items only). Ids Felix marks wrong go into a drop file (`<id> <reason>` per line) for
`calign.data.phase3_split --drop-ids-file`.

Usage:
    uv run python diagnostics/show_verdict_audit.py [--n-hard 30] [--n-random 10] [--seed 0] [--ids H_001 G_123 ...]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

from calign.data.moralchoice import load_scenarios
from calign.data.phase3_split import COUNTERFACTUAL_PATH, HARD_SUBSET_PATH, load_phase3_splits
from calign.paths import OUTPUTS_DIR
from calign.schemas import read_json
from calign.validate.verdicts import load_verdicts

SFT_RUN = OUTPUTS_DIR / "probe_data" / "v2e3_k8" / "records.jsonl"


def sft_pass_rates(path: Path, verdicts) -> dict[str, tuple[int, int]]:
    """scenario_id -> (n aligned, n parsed) over the `none`-variant records of the epoch-3 probe data."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["condition"]["prompt_variant"] != "none":
                continue
            v = verdicts.get(r["scenario_id"])
            d = r.get("parsed_decision")
            if v is None or d not in ("action1", "action2"):
                continue
            out[r["scenario_id"]][0] += int(d == v.prescribed_action)
            out[r["scenario_id"]][1] += 1
    return {k: (a, n) for k, (a, n) in out.items()}


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=116, initial_indent=indent, subsequent_indent=indent)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hard", type=int, default=30)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None, help="show exactly these ids instead of a sample")
    args = ap.parse_args()

    verdicts = load_verdicts()
    cf = load_verdicts(COUNTERFACTUAL_PATH)
    splits = load_phase3_splits()
    scen = {s.scenario_id: s for s in load_scenarios()}
    hard_m = read_json(HARD_SUBSET_PATH) if HARD_SUBSET_PATH.exists() else None
    per_item = hard_m["per_item"] if hard_m else {}
    hard = sorted({sid for ids in hard_m["ids"].values() for sid in ids}) if hard_m else []
    sft = sft_pass_rates(SFT_RUN, verdicts)

    rng = random.Random(args.seed)
    if args.ids:
        sample = [("selected", sid) for sid in args.ids]
    else:
        if not hard_m:
            print(f"[warn] {HARD_SUBSET_PATH} missing: no hard items (run the C0 all-clear base run first)")
        h = rng.sample(hard, min(args.n_hard, len(hard)))
        rest = sorted(set(splits) - set(hard))
        r = rng.sample(rest, min(args.n_random, len(rest)))
        sample = [("hard", sid) for sid in sorted(h)] + [("random", sid) for sid in sorted(r)]

    print(
        f"verdict audit: {len(sample)} items ({len(hard)} hard items in total; SFT e3 pass rates for {len(sft)} items)"
    )
    for kind, sid in sample:
        s, v = scen[sid], verdicts[sid]
        print("\n" + "=" * 120)
        base = per_item.get(sid)
        base_s = f"{base['n_parsed'] - base['n_wrong']}/{base['n_parsed']} parsed correct" if base else "n/a"
        sft_s = f"{sft[sid][0]}/{sft[sid][1]}" if sid in sft else "n/a (not in probe_train/probe_val)"
        print(f"[{kind}] {sid}  split={splits.get(sid)}  rule={s.generation_rule}  base={base_s}  sft_e3={sft_s}")
        print(wrap(s.context))
        for label, key in (("A (action1)", "action1"), ("B (action2)", "action2")):
            viol = {k: x for k, x in s.rule_violations.get(key, {}).items() if x != "No"}
            print(f"  {label}: {getattr(s, key)}")
            print(f"      MoralChoice violations: {viol or 'none'}")
        print(f"  VERDICT: {v.prescribed_action}  principles={v.principles_invoked}  confidence={v.confidence}")
        print(wrap(v.rationale))
        if sid in cf:
            c = cf[sid]
            print(f"  WITHOUT P6: {c.prescribed_action}  principles={c.principles_invoked}  confidence={c.confidence}")
            print(wrap(c.rationale))


if __name__ == "__main__":
    main()
