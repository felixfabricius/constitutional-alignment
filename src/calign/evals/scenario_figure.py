"""Blog figure: misaligned actions in the two agentic scenarios, per configuration (Felix 2026-10-13).

Grouped bars, one group per configuration (display names only), two bars per group: scenario 1 (Study
Significance, deadline L1) and scenario 2 (Excursion Briefing, briefing L1), each the any-deviation rate (the
model reports the non-pre-registered significant result as the main one / leaves the guide's safety record out of
the briefing) over 50 episodes at T=1.0, with Wilson 95% CIs. Values are read from a `calign.evals.report` summary
(re-graded from the raw episodes there). Styling follows the blog's other figures (`calign.rl.dynamics` overview:
surface #fcfcfb, recessive grid, left title, 7.2 x 4.8 in at 150 dpi = 1080 x 720 px) with larger type for phones.

CLI (local):
    uv run python -m calign.evals.scenario_figure [--report outputs/evals/report/final_c10] [--out <png>]
Writes the PNG and a JSON with the plotted values next to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calign.paths import REPO_ROOT

# blog display name -> configuration id in the report (Felix 2026-10-13: SFT-app = C2-kna, RL-process = C4@s50)
CONFIGS: list[tuple[str, str]] = [
    ("Base", "C0"),
    ("Prompt", "C1"),
    ("SFT-fact", "C2kn@e4"),
    ("SFT-app", "C2kna@e3kn1"),
    ("RL-outcome", "C3@s60"),
    ("RL-process", "C4@s50"),
]
SCENARIOS: list[tuple[str, str]] = [
    ("Scenario 1: Study Significance", "deadline_L1_any_deviation"),
    ("Scenario 2: Excursion Briefing", "briefing_L1_any_deviation"),
]
# Reference categorical slots 1 (blue) and 4 (yellow): validated as a pair (CVD dE 31.5, normal 37.5); yellow is
# below 3:1 on the surface, so every bar carries a value label. Orange / aqua are the blog's RL-outcome / RL-process.
COLORS = ("#2a78d6", "#eda100")
_INK, _INK2, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
SIZE, DPI = (7.2, 4.8), 150
FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_VALUE, FONT_LEGEND, FONT_NOTE = 15, 13, 13, 12, 12.5, 10.5
DEFAULT_REPORT = REPO_ROOT / "outputs" / "evals" / "report" / "final_c10"


def load_values(report_dir: Path) -> dict:
    rows = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))["rows"]
    out: dict = {"report": str(report_dir), "configs": []}
    for name, cid in CONFIGS:
        entry = {"name": name, "config": cid, "scenarios": {}}
        for label, key in SCENARIOS:
            cell, rate = key.split("_", 2)[0] + "_" + key.split("_", 2)[1], key.split("_", 2)[2]
            run = rows[cid]["scenario_runs"][cell]
            s = json.loads((REPO_ROOT / run / "summary.json").read_text(encoding="utf-8"))[rate]
            entry["scenarios"][label] = {"value": s["rate"], "ci95_low": s["ci95_low"], "ci95_high": s["ci95_high"],
                                         "k": s["k"], "n": s["n"], "run": run}  # fmt: skip
        out["configs"].append(entry)
    return out


def plot(data: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c["name"] for c in data["configs"]]
    fig, ax = plt.subplots(figsize=SIZE, dpi=DPI, facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    width, gap = 0.38, 0.02
    for j, ((label, _), color) in enumerate(zip(SCENARIOS, COLORS, strict=True)):
        xs = [i + (j - 0.5) * (width + gap) for i in range(len(names))]
        v = [100 * c["scenarios"][label]["value"] for c in data["configs"]]
        lo = [100 * c["scenarios"][label]["ci95_low"] for c in data["configs"]]
        hi = [100 * c["scenarios"][label]["ci95_high"] for c in data["configs"]]
        ax.bar(xs, v, width=width, color=color, edgecolor=_SURFACE, linewidth=1.0, label=label, zorder=2)
        ax.errorbar(xs, v, yerr=[[a - b for a, b in zip(v, lo, strict=True)], [b - a for a, b in zip(v, hi, strict=True)]],
                    fmt="none", ecolor=_INK2, elinewidth=1.1, capsize=3, capthick=1.1, zorder=3)  # fmt: skip
        for x, val, top in zip(xs, v, hi, strict=True):
            ax.annotate(f"{val:.0f}", (x, top), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=FONT_VALUE, color=_INK)  # fmt: skip
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("-", "-\n") for n in names])  # two lines: fits at phone-legible size
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 25))
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(axis="x", colors=_INK, labelsize=FONT_TICK, length=0, pad=6)
    ax.tick_params(axis="y", colors=_INK2, labelsize=FONT_TICK)
    ax.set_ylabel("misaligned actions (%)", color=_INK2, fontsize=FONT_AXIS)
    ax.set_title("Misaligned actions in the agentic scenarios", color=_INK, fontsize=FONT_TITLE, loc="left", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, frameon=False, fontsize=FONT_LEGEND,
              handlelength=1.1, handleheight=1.1, columnspacing=1.4, borderaxespad=0.2, labelcolor=_INK)  # fmt: skip
    n = {c["scenarios"][lab]["n"] for c in data["configs"] for lab, _ in SCENARIOS}
    n_txt = f"{n.pop()} episodes" if len(n) == 1 else f"{min(n)}-{max(n)} episodes"
    fig.text(0.99, 0.01, f"{n_txt} per bar · error bars: 95% CI", ha="right", va="bottom", fontsize=FONT_NOTE,
             color=_INK2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--out", type=Path, default=None, help="default <report>/scenarios_misaligned.png")
    args = ap.parse_args(argv)
    data = load_values(args.report)
    out = args.out or args.report / "scenarios_misaligned.png"
    plot(data, out)
    out.with_suffix(".json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
