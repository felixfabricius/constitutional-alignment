"""Assemble configurations into the Phase 3 table, budget flags and frontier data (D13, D14).

CLI (local):
    uv run python -m calign.evals.report --configs C0 C1 C2 [--checkpoints-of C3 C4] [--out outputs/evals/report/<run>]

For each configuration the newest suite manifest (`outputs/evals/<id>/suite/*/suite.json`) names the component run
dirs. Every component report is recomputed from raw records with C0's run of that component as the paired
reference, then collected into:
- summary.json / summary.md: per configuration every metric with its CI, the delta vs C0 with its paired CI, and
  the budget flags at the default margins (IFEval 3 points, MATH-500 3, coherence 0.05, over-citation 2 points):
  "within" (point estimate of the cost <= m), "robustly within" (the CI bound of the cost <= m), "demonstrably
  outside" (the CI bound of the cost > m), else "outside (point)"; a sensitivity table at m, 2m, 3m. Margins are
  reading aids: nothing is gated on them.
- frontier.json: per configuration/checkpoint, alignment (eval-1, eval-2, hard subset, balanced) against each
  budget metric (value and cost vs C0); frontier_<budget>.png: eval-2 alignment vs the budget cost.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from calign.evals import coherence, ifeval, math500, moralchoice, overcitation, quiz
from calign.evals.config import EVALS_DIR, EvalConfig
from calign.evals.suite import _rel
from calign.paths import REPO_ROOT
from calign.schemas import read_json, write_json

LOGGER = logging.getLogger(__name__)

REFERENCE = "C0"
MODULES = {
    "moralchoice": moralchoice,
    "ifeval": ifeval,
    "math500": math500,
    "overcitation": overcitation,
    "coherence": coherence,
    "quiz": quiz,
}
# budget metric -> (margin m in metric units, higher is better)
BUDGET: dict[str, tuple[float, bool]] = {
    "ifeval_prompt_strict": (0.03, True),
    "math500_accuracy": (0.03, True),
    "coherence_fluency": (0.05, True),
    "overcitation_rate": (0.02, False),
}
ALIGNMENT = ("eval1", "eval2", "hard", "eval2_balanced")


def suites_for(configs: list[str], checkpoints_of: list[str] | None = None, root: Path = EVALS_DIR) -> dict[str, Path]:
    ids = list(configs)
    for base in checkpoints_of or []:
        ids += sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{base}@"))
    out = {}
    for cid in dict.fromkeys(ids):
        base = root / cid / "suite"
        runs = sorted(p for p in base.iterdir() if (p / "suite.json").exists()) if base.exists() else []
        if not runs:
            LOGGER.warning("no suite run for %s; skipped", cid)
            continue
        out[cid] = runs[-1]
    return out


def _v(m: dict | None, key: str) -> dict:
    """{value, ci95_low, ci95_high} from a metric dict that stores its point under `key`."""
    if not m or m.get(key) is None:
        return {"value": None, "ci95_low": None, "ci95_high": None}
    return {"value": m[key], "ci95_low": m.get("ci95_low"), "ci95_high": m.get("ci95_high")}


def _d(m: dict | None, key: str = "delta") -> dict:
    return _v(m, key)


def collect(summaries: dict[str, dict]) -> dict:
    """Flatten component summaries into {metric: {value, ci95_low, ci95_high}} and {metric: delta dict}."""
    vals: dict[str, dict] = {}
    deltas: dict[str, dict] = {}
    mc = summaries.get("moralchoice")
    if mc:
        sp = mc["splits"]
        for s in ("dev", "eval1", "eval2", "all"):
            if s in sp:
                vals[f"{s}"] = _v(sp[s]["alignment"], "mean")
                vals[f"{s}_balanced"] = _v(sp[s]["balanced_alignment"], "mean")
                vals[f"{s}_parse_rate"] = _v(sp[s]["parse_rate"], "mean")
                vals[f"{s}_mention_rate"] = _v(sp[s]["mention_rate"], "mean")
        if "hard" in sp.get("all", {}):
            vals["hard"] = _v(sp["all"]["hard"]["alignment"], "mean")
        if js := mc.get("judge_sample"):
            vals["citation_accuracy"] = {
                "value": js.get("citation_accuracy_citing"),
                "ci95_low": None,
                "ci95_high": None,
            }
        if ref := mc.get("reference"):
            for s, d in ref["paired"].items():
                deltas[s] = _d(d["all_items"])
                if s == "all" and d.get("hard_items"):
                    deltas["hard"] = _d(d["hard_items"])
    if ie := summaries.get("ifeval"):
        vals["ifeval_prompt_strict"] = _v(ie["prompt_level_strict"], "rate")
        vals["ifeval_inst_strict"] = _v(ie["inst_level_strict"], "rate")
        if p := ie.get("paired_vs_reference"):
            deltas["ifeval_prompt_strict"] = _d(p["prompt_level_strict"])
    if ma := summaries.get("math500"):
        vals["math500_accuracy"] = _v(ma["accuracy"], "rate")
        if p := ma.get("paired_vs_reference"):
            deltas["math500_accuracy"] = _d(p["accuracy"])
    if co := summaries.get("coherence"):
        vals["coherence_fluency"] = _v(co["fluency"], "mean")
        vals["invented_constitution"] = _v(co["invented_constitution"], "mean")
        if p := co.get("paired_vs_reference"):
            deltas["coherence_fluency"] = _d(p["fluency"])
            deltas["invented_constitution"] = _d(p["invented_constitution"])
    if oc := summaries.get("overcitation"):
        if oc.get("strict"):
            vals["overcitation_rate"] = _v(oc["strict"]["overcitation_rate"], "rate")
        if lo := oc.get("low_ambiguity"):
            vals["low_mention_rate"] = _v(lo["mention_rate"], "rate")
            vals["low_agreement"] = _v(lo["agreement_rule_abiding"], "rate")
        if d := oc.get("delta_vs_reference"):
            deltas["overcitation_rate"] = _d(d, "diff")
    if qz := summaries.get("quiz"):
        for q in ("recall", "p6"):
            vals[f"quiz_{q}"] = _v(qz[q]["mean_correct"], "mean")
        if p := qz.get("paired_vs_reference"):
            for q in ("recall", "p6"):
                deltas[f"quiz_{q}"] = _d(p[q])
    return {"values": vals, "deltas": deltas}


def budget_flag(delta: dict, margin: float, higher_better: bool) -> str | None:
    """Flag of the cost (= capability lost, or over-citation gained) relative to a margin (D14)."""
    d, lo, hi = delta.get("value"), delta.get("ci95_low"), delta.get("ci95_high")
    if d is None:
        return None
    sign = -1.0 if higher_better else 1.0
    cost = sign * d
    bounds = [sign * x for x in (lo, hi) if x is not None]
    cost_hi = max(bounds) if len(bounds) == 2 else None
    cost_lo = min(bounds) if len(bounds) == 2 else None
    if cost_lo is not None and cost_lo > margin:
        return "demonstrably outside"
    if cost_hi is not None and cost_hi <= margin:
        return "robustly within"
    if cost <= margin:
        return "within"
    return "outside (point)"


def flags_for(deltas: dict[str, dict], multiples: tuple[int, ...] = (1, 2, 3)) -> dict:
    out = {}
    for metric, (m, hb) in BUDGET.items():
        if metric in deltas:
            out[metric] = {f"{k}m": budget_flag(deltas[metric], k * m, hb) for k in multiples}
    return out


def frontier(rows: dict[str, dict]) -> dict:
    pts = []
    for cid, row in rows.items():
        v, d = row["metrics"]["values"], row["metrics"]["deltas"]
        pts.append(
            {
                "id": cid,
                "stage": row["stage"],
                "alignment": {k: v.get(k, {}).get("value") for k in ALIGNMENT},
                "budget": {
                    k: {
                        "value": v.get(k, {}).get("value"),
                        "cost_vs_C0": (None if d.get(k, {}).get("value") is None else (-1 if hb else 1) * d[k]["value"])
                        if cid != REFERENCE
                        else 0.0,
                    }
                    for k, (_, hb) in BUDGET.items()
                },
            }
        )
    return {"points": pts, "margins": {k: m for k, (m, _) in BUDGET.items()}}


# Chart styling (dataviz reference palette, light mode): scatter forms validate only three categorical slots, so the
# trained families get them (C2 blue, C3 orange, C4 aqua) and everything else (C0, C1, SFTP) is neutral ink; every
# point is direct-labelled with its id, so identity never rests on colour alone.
_INK, _INK2, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
_FAMILY_COLOR = {"C2": "#2a78d6", "C3": "#eb6834", "C4": "#1baf7a"}


def plot_frontier(fr: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for metric, margin in fr["margins"].items():
        pts = [
            p
            for p in fr["points"]
            if p["alignment"]["eval2"] is not None and p["budget"][metric]["cost_vs_C0"] is not None
        ]
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150, facecolor=_SURFACE)
        ax.set_facecolor(_SURFACE)
        for p in pts:
            x, y = 100 * p["budget"][metric]["cost_vs_C0"], 100 * p["alignment"]["eval2"]
            ax.scatter(
                [x],
                [y],
                s=64,
                color=_FAMILY_COLOR.get(p["id"].split("@")[0], _INK2),
                edgecolors=_SURFACE,
                linewidths=2,
                zorder=3,
            )
            ax.annotate(p["id"], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=_INK)
        ax.axvline(100 * margin, color=_INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(
            f"margin {100 * margin:g}",
            (100 * margin, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color=_INK2,
        )
        ax.grid(True, color=_GRID, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(_GRID)
        ax.tick_params(colors=_INK2, labelsize=8)
        unit = "points" if metric != "coherence_fluency" else "x100"
        ax.set_xlabel(f"{metric} cost vs C0 ({unit}; right = more costly)", color=_INK2, fontsize=9)
        ax.set_ylabel("eval-2 alignment (%)", color=_INK2, fontsize=9)
        ax.set_title(f"Alignment vs {metric}", color=_INK, fontsize=10, loc="left")
        fig.tight_layout()
        path = out_dir / f"frontier_{metric}.png"
        fig.savefig(path, facecolor=_SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def _cell(x: dict | None, pct: bool = True) -> str:
    if not x or x.get("value") is None:
        return "-"
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.3f}")
    if x.get("ci95_low") is None:
        return f(x["value"])
    return f"{f(x['value'])} [{f(x['ci95_low'])}, {f(x['ci95_high'])}]"


TABLE_METRICS = (
    ("eval1", True),
    ("eval2", True),
    ("hard", True),
    ("eval2_balanced", True),
    ("all_mention_rate", True),
    ("citation_accuracy", False),
    ("ifeval_prompt_strict", True),
    ("math500_accuracy", True),
    ("coherence_fluency", False),
    ("invented_constitution", False),
    ("overcitation_rate", True),
    ("low_agreement", True),
    ("low_mention_rate", True),
    ("quiz_recall", False),
    ("quiz_p6", False),
)


def render_markdown(report: dict) -> str:
    rows = report["rows"]
    ids = list(rows)
    lines = [
        "# Phase 3 evaluation report",
        "",
        f"Generated {report['created_at']}; reference {REFERENCE}. Rates in %, scores 0-1, 95% CIs in brackets.",
        "",
        "| metric | " + " | ".join(ids) + " |",
        "|---|" + "---|" * len(ids),
    ]
    for metric, pct in TABLE_METRICS:
        lines.append(
            f"| {metric} | " + " | ".join(_cell(rows[c]["metrics"]["values"].get(metric), pct) for c in ids) + " |"
        )
    lines += [
        "",
        "Deltas vs C0 (paired over shared items):",
        "",
        "| metric | " + " | ".join(ids) + " |",
        "|---|" + "---|" * len(ids),
    ]
    for metric, pct in TABLE_METRICS:
        cells = [_cell(rows[c]["metrics"]["deltas"].get(metric), pct) for c in ids]
        if any(x != "-" for x in cells):
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Budget flags (sensitivity at m, 2m, 3m):",
        "",
        "| config | metric | m | 2m | 3m |",
        "|---|---|---|---|---|",
    ]
    for c in ids:
        for metric, f in rows[c]["flags"].items():
            lines.append(f"| {c} | {metric} | {f['1m']} | {f['2m']} | {f['3m']} |")
    return "\n".join(lines) + "\n"


def build_report(configs: list[str], checkpoints_of: list[str] | None = None, root: Path = EVALS_DIR) -> dict:
    suites = suites_for(configs, checkpoints_of, root)
    ref_runs = {}
    if REFERENCE in suites:
        ref_runs = {k: REPO_ROOT / v for k, v in read_json(suites[REFERENCE] / "suite.json")["components"].items()}
    rows = {}
    for cid, sdir in suites.items():
        man = read_json(sdir / "suite.json")
        cfg = EvalConfig.model_validate(man["eval_config"])
        summaries = {}
        for comp, rel in man["components"].items():
            if comp not in MODULES:  # e.g. a superseded run kept for the record
                continue
            rd = REPO_ROOT / rel
            ref = ref_runs.get(comp) if cid != REFERENCE else None
            try:
                summaries[comp] = MODULES[comp].write_report(rd, ref)
            except FileNotFoundError as e:  # a component not judged yet
                LOGGER.warning("%s/%s: %s", cid, comp, e)
        metrics = collect(summaries)
        rows[cid] = {
            "label": cfg.label,
            "stage": cfg.stage,
            "suite": _rel(sdir),
            "components": man["components"],
            "metrics": metrics,
            "flags": flags_for(metrics["deltas"]) if cid != REFERENCE else {},
        }
    return {"created_at": datetime.now().isoformat(timespec="seconds"), "reference": REFERENCE, "rows": rows}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=[REFERENCE])
    ap.add_argument("--checkpoints-of", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = build_report(args.configs, args.checkpoints_of)
    out = args.out or EVALS_DIR / "report" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    fr = frontier(report["rows"])
    write_json(out / "summary.json", report)
    write_json(out / "frontier.json", fr)
    (out / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    if not args.no_plots:
        plot_frontier(fr, out)
    print(render_markdown(report))
    LOGGER.info("report %s", out)


if __name__ == "__main__":
    main()
