"""Phase 3 primary comparisons (D15) with the chunk 9 checkpoint mapping (Felix 2026-10-04), from the raw run files.

CLI (local):
    uv run python -m calign.evals.primary --out outputs/evals/report/final_c9

Configurations: C0 base, C1 base + prompt, C2 knowledge-only SFT (`C2kn@e4`, the RL start), C3 = `C3@s60`; C4 is
represented per comparison (`phase3_plan.md`, row "Chunk 9 checkpoint selection"): **C4@s20** for scenario 2
(briefing, the P6 scenario) and the P6 questions (eval-2, P6 quiz), **C4@s50** for everything else (eval-1, hard
subset, scenario 1). The per-component choice is post hoc and may favour C4; the write-up says so.

Comparisons (each a difference "a - b" with a 95% CI):
- MoralChoice eval-1 / eval-2 / hard subset: paired over items (per-item alignment = share of parsed samples matching
  the verdict; bootstrap over items, `moralchoice.paired_delta`);
- quizzes: paired over questions (`quiz.summarize` with a reference);
- scenarios (deadline L1 = scenario 1, briefing L1 = scenario 2): any-deviation and deceptive-tier rates, Newcombe
  differences of two independent proportions (`scenarios.report.versus`).
Writes `primary.json` and `primary.md`. eval-2-hard has no item set (status E6), so its rows are absent.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from calign.evals import moralchoice, quiz
from calign.evals.config import EVALS_DIR
from calign.evals.report import scenario_runs_for
from calign.paths import REPO_ROOT
from calign.scenarios import report as scen_report
from calign.schemas import GenerationRecord, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)

C2, C3, C4_P6, C4_REST = "C2kn@e4", "C3@s60", "C4@s20", "C4@s50"
# (label, a, b, metrics): MoralChoice splits/hard, "quiz_p6"/"quiz_recall", "<scenario>_L1"
COMPARISONS: list[tuple[str, str, str, list[str]]] = [
    ("C1 vs C0", "C1", "C0", ["eval2", "hard", "deadline_L1", "briefing_L1"]),
    ("C2 vs C0", C2, "C0", ["eval2", "hard", "deadline_L1", "briefing_L1"]),
    ("C3 vs C2", C3, C2, ["eval1", "eval2", "hard", "quiz_recall", "quiz_p6", "deadline_L1", "briefing_L1"]),
    ("C3 vs C0", C3, "C0", ["deadline_L1", "briefing_L1"]),
    ("C4 vs C3 (C4@s50)", C4_REST, C3, ["eval1", "hard", "deadline_L1"]),
    ("C4 vs C3 (C4@s20, P6 items)", C4_P6, C3, ["eval2", "quiz_p6", "briefing_L1"]),
    ("C4 vs C2 (C4@s50)", C4_REST, C2, ["eval1", "hard", "deadline_L1"]),
    ("C4 vs C2 (C4@s20, P6 items)", C4_P6, C2, ["eval2", "quiz_p6", "briefing_L1"]),
    ("C4 vs C0 (C4@s50)", C4_REST, "C0", ["deadline_L1"]),
    ("C4 vs C0 (C4@s20)", C4_P6, "C0", ["briefing_L1"]),
]


def suite_component(cid: str, comp: str) -> Path:
    base = EVALS_DIR / cid / "suite"
    runs = sorted(p for p in base.iterdir() if (p / "suite.json").exists())
    rel = json.loads((runs[-1] / "suite.json").read_text(encoding="utf-8"))["components"][comp]
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def mc_delta(a: str, b: str, metric: str) -> dict:
    verdicts = moralchoice.load_verdicts()
    hard = moralchoice.load_hard_ids()
    ra = moralchoice.current_splits(
        read_jsonl(suite_component(a, "moralchoice") / moralchoice.RECORDS_FILE, GenerationRecord)
    )
    rb = moralchoice.current_splits(
        read_jsonl(suite_component(b, "moralchoice") / moralchoice.RECORDS_FILE, GenerationRecord)
    )
    if metric == "hard":
        ga, gb, only = [r for r in ra if r.split != "dropped"], [r for r in rb if r.split != "dropped"], hard
    else:
        ga, gb, only = [r for r in ra if r.split == metric], [r for r in rb if r.split == metric], None
    d = moralchoice.paired_delta(ga, gb, verdicts, only=only)
    return {"kind": "paired over items", **d}


def quiz_delta(a: str, b: str, metric: str) -> dict:
    name = metric.removeprefix("quiz_")
    ra = read_jsonl(suite_component(a, "quiz") / quiz.RECORDS_FILE, GenerationRecord)
    rb = read_jsonl(suite_component(b, "quiz") / quiz.RECORDS_FILE, GenerationRecord)
    s = quiz.summarize(ra, rb)
    d = s["paired_vs_reference"][name]
    return {"kind": "paired over questions", "a": s[name]["mean_correct"]["mean"], **d}


def scen_delta(a: str, b: str, cell: str) -> dict:
    ra, rb = scenario_runs_for(a).get(cell), scenario_runs_for(b).get(cell)
    if ra is None or rb is None:
        return {"kind": "missing", "a_run": str(ra), "b_run": str(rb)}
    sa = scen_report.summarize(scen_report.load_run(ra)[0])
    sb = scen_report.summarize(scen_report.load_run(rb)[0])
    v = scen_report.versus(sa, sb)
    return {
        "kind": "Newcombe, independent samples",
        "a_run": str(ra.relative_to(REPO_ROOT)).replace("\\", "/"),
        "b_run": str(rb.relative_to(REPO_ROOT)).replace("\\", "/"),
        "a": {k: sa[k] for k in ("any_deviation", "deceptive")},
        "b": {k: sb[k] for k in ("any_deviation", "deceptive")},
        "any_deviation": v.get("any_deviation"),
        "deceptive": v.get("deceptive"),
    }


def compute() -> dict:
    out = []
    for label, a, b, metrics in COMPARISONS:
        row = {"comparison": label, "a": a, "b": b, "metrics": {}}
        for m in metrics:
            if m.startswith("quiz_"):
                row["metrics"][m] = quiz_delta(a, b, m)
            elif m.endswith("_L1"):
                row["metrics"][m] = scen_delta(a, b, m)
            else:
                row["metrics"][m] = mc_delta(a, b, m)
        out.append(row)
    return {"mapping": {"C3": C3, "C4 (P6 scenario, P6 questions)": C4_P6, "C4 (others)": C4_REST, "C2": C2},
            "comparisons": out}  # fmt: skip


def _ci(d: dict | None, pct: bool = True) -> str:
    if not d:
        return "-"
    f = 100 if pct else 1
    lo, hi = d.get("ci95_low"), d.get("ci95_high")
    val = d.get("delta", d.get("diff"))
    if val is None:
        return "-"
    nd = 1 if pct else 3
    return f"{f * val:+.{nd}f} [{f * lo:+.{nd}f}, {f * hi:+.{nd}f}]" if lo is not None else f"{f * val:+.{nd}f}"


def render(p: dict) -> str:
    lines = [
        "# Primary comparisons (D15) with the chunk 9 C4 mapping",
        "",
        f"Mapping: {p['mapping']}. MoralChoice and quizzes: paired differences (a - b) with bootstrap 95% CIs; "
        "scenarios: Newcombe 95% CIs for the difference of two rates (independent samples). Points (x100) except quizzes.",
        "",
        "| comparison | metric | difference a - b [95% CI] |",
        "|---|---|---|",
    ]
    for row in p["comparisons"]:
        for m, d in row["metrics"].items():
            if m.endswith("_L1"):
                for k in ("any_deviation", "deceptive"):
                    lines.append(f"| {row['comparison']} | {m} {k} | {_ci(d.get(k))} |")
            else:
                lines.append(f"| {row['comparison']} | {m} | {_ci(d, pct=not m.startswith('quiz_'))} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = compute()
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "primary.json", p)
    md = render(p)
    (args.out / "primary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
