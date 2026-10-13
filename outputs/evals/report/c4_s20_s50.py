"""C4@s20 vs C4@s50 (Felix 2026-10-13): values from final_c10 and paired deltas s50 - s20 (report reference switched to
C4@s20; rebuild final_c10 afterwards so run dirs keep C0-referenced summaries). Writes c4_s20_s50.json next to this."""
import json
from pathlib import Path

import calign.evals.report as R

here = Path(__file__).parent
rows = json.loads((here / "final_c10" / "summary.json").read_text(encoding="utf-8"))["rows"]
out = {"values": {}, "judge_sample": {}, "delta_s50_minus_s20": {}}
for c in ["C2kn@e4", "C4@s20", "C4@s50"]:
    out["values"][c] = rows[c]["metrics"]["values"]
    md = Path(rows[c]["components"]["moralchoice"])
    out["judge_sample"][c] = json.loads((md / "summary.json").read_text(encoding="utf-8")).get("judge_sample")
R.REFERENCE = "C4@s20"
rep = R.build_report(["C4@s20", "C4@s50"])
out["delta_s50_minus_s20"] = rep["rows"]["C4@s50"]["metrics"]["deltas"]
(here / "c4_s20_s50.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("ok")
