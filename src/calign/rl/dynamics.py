"""Training-dynamics figures for the RL runs (chunk 9 deliverable 3): training reward vs RL hold-out reward over steps.

CLI (local, after the run dirs are synced):
    uv run python -m calign.rl.dynamics --runs outputs/rl/C3 outputs/rl/C4 --out outputs/evals/report/<run>

Per run `dynamics_<run>.png`: outcome reward on generated dilemma rows during training (rolling mean over 5 logged
steps) next to the RL hold-out outcome (mean +- item-clustered SE, step 0 and every 10 steps), the in-distribution
overfitting check (Felix 2026-10-03); for a run with the citation reward a second panel shows the citation term
(training rows and hold-out). `dynamics_compare.png`: the hold-out outcome of all runs by step, by KL from the RL start
(matched-KL comparison, chunk 7 note D24: the reward scale equalises advantage size only at the start), and the
knowledge-retention quizzes (recall, P6) per checkpoint from the core suites `<run>@s<step>` with the 0.8 stop line.
`dynamics.json` holds every plotted number. Everything is recomputed from steps.jsonl, holdout.jsonl and the suites.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from calign.rl.holdout import read_summaries
from calign.rl.monitor import QUIZ_MIN, checkpoint_quizzes, read_steps, rolling, suite_quizzes
from calign.schemas import write_json

LOGGER = logging.getLogger(__name__)

TRAIN_OUTCOME = "r_outcome/dilemma"
TRAIN_CITE = "r_cite/dilemma"
ROLL = 5
START_CONFIG = "C2kn@e4"  # the RL start's core suite (C2) = the quizzes at step 0
# report palette (calign.evals.report): C3 orange, C4 aqua, others ink
_INK, _INK2, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
_RUN_COLOR = {"C3": "#eb6834", "C4": "#1baf7a"}


def run_series(run_dir: Path) -> dict:
    rows = read_steps(run_dir)
    hold = sorted((s for s in read_summaries(run_dir) if s.get("outcome") is not None), key=lambda s: s["step"])
    kl = [(r["step"], float(r["kl"])) for r in rows if r.get("kl") is not None]

    def kl_at(step: int) -> float:
        if step == 0 or not kl:
            return 0.0
        vals = [v for s, v in kl if step - ROLL < s <= step]
        return sum(vals) / len(vals) if vals else 0.0

    name = run_dir.name
    quizzes = checkpoint_quizzes(name, None)
    start = start_quizzes()
    if start and 0 not in quizzes:
        quizzes[0] = start
    return {
        "run": name,
        "train_outcome": rolling(rows, TRAIN_OUTCOME, ROLL),
        "train_cite": rolling(rows, TRAIN_CITE, ROLL),
        "kl": kl,
        "holdout": [
            {
                "step": s["step"],
                "outcome": s["outcome"],
                "se": s.get("outcome_se"),
                "r_cite": s.get("r_cite"),
                "kl": kl_at(s["step"]),
            }
            for s in hold
        ],
        "quizzes": {str(k): v for k, v in sorted(quizzes.items())},
    }


def start_quizzes(config_id: str = START_CONFIG, evals_root: Path | None = None) -> dict | None:
    """The RL start's quiz scores (newest suite of `config_id`), plotted at step 0."""
    from calign.paths import OUTPUTS_DIR

    return suite_quizzes((evals_root or OUTPUTS_DIR / "evals") / config_id)


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=8)
    ax.set_title(title, color=_INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=_INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=_INK2, fontsize=9)


def plot_run(series: dict, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = _RUN_COLOR.get(series["run"], _INK2)
    has_cite = bool(series["train_cite"]) or any(h["r_cite"] is not None for h in series["holdout"])
    fig, axes = plt.subplots(1, 2 if has_cite else 1, figsize=(11 if has_cite else 6.4, 4.2), dpi=150,
                             facecolor=_SURFACE, squeeze=False)  # fmt: skip
    ax = axes[0][0]
    tx, ty = zip(*series["train_outcome"], strict=True) if series["train_outcome"] else ((), ())
    ax.plot(tx, ty, color=color, linewidth=1.6, alpha=0.55, label=f"training (rolling {ROLL} steps)")
    h = series["holdout"]
    ax.errorbar([x["step"] for x in h], [x["outcome"] for x in h], yerr=[x["se"] or 0 for x in h], color=color,
                marker="o", markersize=5, linewidth=2, capsize=3, label="RL hold-out (+- SE)")  # fmt: skip
    for x in h:
        ax.annotate(f"{x['outcome']:.2f}", (x["step"], x["outcome"]), xytext=(4, 6), textcoords="offset points",
                    fontsize=7, color=_INK)  # fmt: skip
    _style(ax, f"{series['run']}: outcome reward, training vs RL hold-out", "optimizer step", "outcome reward")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    if has_cite:
        ax = axes[0][1]
        if series["train_cite"]:
            cx, cy = zip(*series["train_cite"], strict=True)
            ax.plot(cx, cy, color=color, linewidth=1.6, alpha=0.55, label=f"training (rolling {ROLL} steps)")
        hc = [x for x in h if x["r_cite"] is not None]
        ax.plot([x["step"] for x in hc], [x["r_cite"] for x in hc], color=color, marker="o", markersize=5,
                linewidth=2, label="RL hold-out")  # fmt: skip
        ax.axhline(0, color=_INK2, linewidth=1, linestyle=(0, (4, 3)))
        _style(ax, f"{series['run']}: citation term (unscaled)", "optimizer step", "r_cite")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    path = out_dir / f"dynamics_{series['run']}.png"
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    return path


def plot_compare(all_series: list[dict], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150, facecolor=_SURFACE)
    for s in all_series:
        color = _RUN_COLOR.get(s["run"], _INK2)
        h = s["holdout"]
        axes[0].errorbar([x["step"] for x in h], [x["outcome"] for x in h], yerr=[x["se"] or 0 for x in h],
                         color=color, marker="o", markersize=5, linewidth=2, capsize=3, label=s["run"])  # fmt: skip
        axes[1].plot([x["kl"] for x in h], [x["outcome"] for x in h], color=color, marker="o", markersize=5,
                     linewidth=2, label=s["run"])  # fmt: skip
        for x in h:
            axes[1].annotate(f"s{x['step']}", (x["kl"], x["outcome"]), xytext=(4, 4), textcoords="offset points",
                             fontsize=7, color=_INK2)  # fmt: skip
        steps = sorted(int(k) for k in s["quizzes"])
        for key, style in (("recall", "-"), ("p6", "--")):
            pts = [(st, s["quizzes"][str(st)].get(key)) for st in steps if s["quizzes"][str(st)].get(key) is not None]
            if pts:
                xs, ys = zip(*pts, strict=True)
                axes[2].plot(xs, ys, color=color, linestyle=style, marker="o", markersize=4, linewidth=1.8,
                             label=f"{s['run']} {'P6 quiz' if key == 'p6' else 'recall quiz'}")  # fmt: skip
    axes[2].axhline(QUIZ_MIN, color=_INK2, linewidth=1, linestyle=(0, (4, 3)))
    axes[2].annotate(f"stop flag {QUIZ_MIN}", (0, QUIZ_MIN), xycoords=("axes fraction", "data"), xytext=(4, -12),
                     textcoords="offset points", fontsize=8, color=_INK2)  # fmt: skip
    _style(axes[0], "RL hold-out outcome by step", "optimizer step", "outcome reward (+- SE)")
    _style(axes[1], "RL hold-out outcome by KL from the RL start", "KL (rolling mean of the 5 steps up to the evaluation)",
           "outcome reward")  # fmt: skip
    _style(axes[2], "Knowledge retention per checkpoint (core suite)", "checkpoint step", "quiz score")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    path = out_dir / "dynamics_compare.png"
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    series = [run_series(r) for r in args.runs]
    paths = [plot_run(s, args.out) for s in series] + [plot_compare(series, args.out)]
    write_json(args.out / "dynamics.json", {"runs": series})
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
