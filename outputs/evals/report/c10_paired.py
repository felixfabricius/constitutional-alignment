"""Chunk 10: paired deltas of C2kna@e3kn1 against C2 (C2kn@e4) and C2-app (C2@e4), computed with calign.evals.report
by switching its reference (the report module pairs against REFERENCE = C0 otherwise). Writes c10_paired.json/.md
next to this script. Run `calign.evals.report ... --out outputs/evals/report/final_c10` afterwards to restore the
C0-referenced summaries in the run dirs."""
import json
from pathlib import Path

import calign.evals.report as R

KEYS = ["eval1", "eval2", "hard", "ifeval_prompt_strict", "math500_accuracy", "coherence_fluency",
        "invented_constitution", "overcitation_rate", "deadline_L1_any_deviation", "deadline_L1_deceptive",
        "briefing_L1_deceptive", "briefing_L1_any_deviation", "deadline_L0_any_deviation", "briefing_L0_deceptive"]
out = {}
here0 = Path(__file__).parent
cached = here0 / "c10_paired.json"
for ref in ([] if cached.exists() else ["C2kn@e4", "C2@e4"]):
    R.REFERENCE = ref
    rep = R.build_report([ref, "C2kna@e3kn1"])
    d = rep["rows"]["C2kna@e3kn1"]["metrics"]["deltas"]
    out[ref] = {k: d.get(k) for k in KEYS}
here = Path(__file__).parent
if cached.exists():
    out = json.loads(cached.read_text(encoding="utf-8"))
(here / "c10_paired.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
lines = ["| metric | C2kna@e3kn1 - C2 (C2kn@e4) | C2kna@e3kn1 - C2-app (C2@e4) |", "|---|---|---|"]
PCT = {"coherence_fluency", "invented_constitution"}  # shown as fractions; everything else in points
def cell(x, k):
    if not x or x.get("value") is None:
        return "-"
    f = 1 if k in PCT else 100
    nd = 3 if k in PCT else 1
    return f"{x['value'] * f:+.{nd}f} [{x['ci95_low'] * f:+.{nd}f}, {x['ci95_high'] * f:+.{nd}f}]"
for k in KEYS:
    lines.append(f"| {k} | {cell(out['C2kn@e4'][k], k)} | {cell(out['C2@e4'][k], k)} |")
(here / "c10_paired.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
