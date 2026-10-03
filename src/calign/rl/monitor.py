"""RL run monitor: trajectory table and stop-and-check-in flags (chunks 7-8).

CLI (local after rsync, or on the instance while the run is going):
    uv run python -m calign.rl.monitor --run-dir outputs/rl/C4_pilot [--config-id C4] [--every 5] [--fail-on-flag]
writes <run dir>/monitor.md and monitor.json and prints the table.

Flags (rolling mean over the last `window` = 5 logged steps where a per-step rate is noisy; ~28 math and ~100
dilemma completions per step):
- length: mean completion length > 2 x the start (mean of the first 3 logged steps);
- math_mentions: regex constitution mention rate on math rows > 5% (rolling);
- letter_prior: letter-A share of parsed dilemma/anchor decisions minus the share of those rows whose correct answer
  is A, outside +-0.15 (rolling; from rollouts.jsonl). The raw A share tracks which letter orders the step happened
  to sample (both chunk 8 runs fell to 0.35 at steps 21-25 because only 36% of the sampled rows had A correct,
  2026-10-03); without rollouts.jsonl the raw share outside [0.35, 0.65] is used;
- zero_variance: share of prompt groups with identical rewards > 60% (rolling);
- (no flag) `adv RMS` (scaled typical advantage) and `KL term` (KL weight x KL) per step: C3 and C4 start with equal
  advantage size by construction (reward.scale); if C4's stays more than 1.5x away from C3's for a sustained stretch,
  report it (no mid-run change);
- knowledge retention (from the core suite of each checkpoint, eval configs `<config-id>@s<step>`): recall quiz < 0.8
  or P6 quiz < 0.8 (RL never trains on P6, so this checks that RL does not erode what the model knows);
- holdout_gap (RL hold-out, `holdout.jsonl`, calign.rl.holdout): at two consecutive evaluations the training gain
  exceeds the hold-out gain by more than 0.15 while the hold-out gain stays below 0.05. Training gain = mean outcome
  reward on generated dilemma rows over the 10 logged steps up to the evaluation minus the mean of the first 5 steps;
  hold-out gain = hold-out outcome minus the step-0 evaluation (never the selection counts, which regress to the
  mean). The hold-out mean has a standard error of ~0.05 (29 items x 16 answers), hence two evaluations in a row.
Any flag means: stop the run and check in with Felix (chunk 8), never "fix and continue" silently.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from calign.paths import OUTPUTS_DIR, REPO_ROOT
from calign.rl.holdout import read_summaries
from calign.rl.train_grpo import STEPS_FILE

WINDOW = 5
LENGTH_FACTOR = 2.0
MATH_MENTION_MAX = 0.05
LETTER_A_RANGE = (0.35, 0.65)
LETTER_EXCESS_MAX = 0.15
ROLLOUTS_FILE = "rollouts.jsonl"
ZERO_VAR_MAX = 0.60
QUIZ_MIN = 0.8
HOLDOUT_GAP_MAX = 0.15
HOLDOUT_GAIN_MIN = 0.05
HOLDOUT_CONSECUTIVE = 2
TRAIN_OUTCOME_KEY = "r_outcome/dilemma"  # generated dilemma rows (anchors are task_type "anchor")
TRAIN_BASELINE_STEPS = 5
TRAIN_WINDOW = 10

COLUMNS = (
    ("step", "step"),
    ("step_s", "step s"),
    ("total/dilemma", "R dilemma"),
    ("total/anchor", "R anchor"),
    ("total/math", "R math"),
    ("r_outcome/dilemma", "outcome"),
    ("r_cite/dilemma", "cite"),
    ("r_math/math", "math acc"),
    ("mention_rate/dilemma", "mention dil"),
    ("mention_rate/math", "mention math"),
    ("letter_a_share", "letter A"),
    ("letter_a_excess", "A excess"),
    ("parse_rate/moral", "parse"),
    ("zero_var_share/all", "zero-var"),
    ("length/mean", "len"),
    ("length/p90", "len p90"),
    ("length/truncated_share", "trunc"),
    ("kl", "KL"),
    ("adv_rms", "adv RMS"),
    ("kl_term", "KL term"),
    ("grad_norm", "grad"),
    ("judge/correct", "judge +"),
    ("judge/incorrect", "judge -"),
    ("peak_mem_gb", "mem GB"),
)


def read_steps(run_dir: Path) -> list[dict]:
    path = run_dir / STEPS_FILE
    if not path.exists():
        return []
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    # keep the per-step training logs (the final train summary line has no reward metrics; evaluation lines carry
    # eval_* keys and are read from holdout.jsonl instead)
    return [r for r in rows if "train_runtime" not in r and not any(k.startswith("eval_") for k in r)]


def correct_letter(row: dict) -> str:
    """The letter of the verdict's option in a rollout row (AB: action1 is A)."""
    a1 = "A" if row["letter_order"] == "AB" else "B"
    return a1 if row["verdict"] == "action1" else ("B" if a1 == "A" else "A")


def add_letter_excess(rows: list[dict], run_dir: Path) -> bool:
    """`letter_a_excess` per step = chosen-A share - correct-is-A share over parsed dilemma/anchor rollouts."""
    path = run_dir / ROLLOUTS_FILE
    if not path.exists():
        return False
    acc: dict[int, list[int]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task_type") not in ("dilemma", "anchor") or r.get("letter") not in ("A", "B"):
                continue
            a = acc.setdefault(r["step"], [0, 0, 0])
            a[0] += r["letter"] == "A"
            a[1] += correct_letter(r) == "A"
            a[2] += 1
    for row in rows:
        a = acc.get(row["step"])
        if a and a[2]:
            row["letter_a_excess"] = (a[0] - a[1]) / a[2]
    return True


def add_kl_term(rows: list[dict], beta: float | None) -> None:
    """`kl_term` = KL weight x logged KL, to read next to `adv_rms` (the scaled typical advantage): their ratio shows
    how much the KL penalty pulls against the reward signal; compare C3 and C4 step by step (chunk 7, reward scale)."""
    if beta is None:
        return
    for r in rows:
        if r.get("kl") is not None:
            r["kl_term"] = beta * float(r["kl"])


def rolling(rows: list[dict], key: str, window: int = WINDOW) -> list[tuple[int, float]]:
    """(step, mean of `key` over the last `window` rows that have it), per row that has it."""
    out, vals = [], []
    for r in rows:
        if r.get(key) is None:
            continue
        vals.append(float(r[key]))
        out.append((r["step"], sum(vals[-window:]) / len(vals[-window:])))
    return out


def trajectory_flags(rows: list[dict], window: int = WINDOW) -> list[dict]:
    letter = (
        ("letter_prior", "letter_a_excess", lambda v: abs(v) > LETTER_EXCESS_MAX, LETTER_EXCESS_MAX)
        if any("letter_a_excess" in r for r in rows)
        else (
            "letter_prior",
            "letter_a_share",
            lambda v: not LETTER_A_RANGE[0] <= v <= LETTER_A_RANGE[1],
            LETTER_A_RANGE,
        )
    )
    flags: list[dict] = []
    lengths = [(r["step"], float(r["length/mean"])) for r in rows if r.get("length/mean") is not None]
    if len(lengths) > 3:
        start = sum(v for _, v in lengths[:3]) / 3
        for step, v in lengths[3:]:
            if v > LENGTH_FACTOR * start:
                flags.append({"flag": "length", "step": step, "value": v, "threshold": LENGTH_FACTOR * start})
                break
    checks = (
        ("math_mentions", "mention_rate/math", lambda v: v > MATH_MENTION_MAX, MATH_MENTION_MAX),
        letter,
        ("zero_variance", "zero_var_share/all", lambda v: v > ZERO_VAR_MAX, ZERO_VAR_MAX),
    )
    for name, key, bad, thr in checks:
        series = rolling(rows, key, window)
        # a rolling mean needs a full window before it can flag (except at the end of a short run)
        for i, (step, v) in enumerate(series):
            if (i + 1 >= window or i == len(series) - 1) and bad(v):
                flags.append({"flag": name, "step": step, "value": round(v, 4), "threshold": thr})
                break
    return flags


def _avg(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def holdout_trajectory(rows: list[dict], summaries: list[dict]) -> list[dict]:
    """Per hold-out evaluation: hold-out outcome and gain vs step 0, training outcome (rolling mean of the
    `TRAIN_WINDOW` logged steps up to the evaluation) and gain vs the first `TRAIN_BASELINE_STEPS` steps, and the gap."""
    train = [(r["step"], float(r[TRAIN_OUTCOME_KEY])) for r in rows if r.get(TRAIN_OUTCOME_KEY) is not None]
    base_train = _avg([v for _, v in train[:TRAIN_BASELINE_STEPS]]) if len(train) >= TRAIN_BASELINE_STEPS else None
    evals = sorted((s for s in summaries if s.get("outcome") is not None), key=lambda s: s["step"])
    base_h = next((s["outcome"] for s in evals if s["step"] == 0), None)
    out = []
    for s in evals:
        tr = _avg([v for st, v in train if st <= s["step"]][-TRAIN_WINDOW:]) if s["step"] > 0 else None
        h_gain = s["outcome"] - base_h if base_h is not None else None
        t_gain = tr - base_train if tr is not None and base_train is not None else None
        out.append({
            "step": s["step"], "holdout": s["outcome"], "holdout_se": s.get("outcome_se"), "holdout_gain": h_gain,
            "train": tr, "train_gain": t_gain,
            "gap": t_gain - h_gain if t_gain is not None and h_gain is not None else None,
            "r_cite": s.get("r_cite"), "mention_rate": s.get("mention_rate"),
        })  # fmt: skip
    return out


def holdout_flags(traj: list[dict]) -> list[dict]:
    """`holdout_gap` at the second of `HOLDOUT_CONSECUTIVE` consecutive evaluations with gap > 0.15 and hold-out
    gain < 0.05 (evaluations without both gains, e.g. step 0, break the run)."""
    run = 0
    for t in traj:
        bad = t["gap"] is not None and t["gap"] > HOLDOUT_GAP_MAX and t["holdout_gain"] < HOLDOUT_GAIN_MIN
        run = run + 1 if bad else 0
        if run >= HOLDOUT_CONSECUTIVE:
            return [{"flag": "holdout_gap", "step": t["step"], "value": round(t["gap"], 4),
                     "threshold": {"gap": HOLDOUT_GAP_MAX, "holdout_gain_below": HOLDOUT_GAIN_MIN}}]  # fmt: skip
    return []


def checkpoint_quizzes(config_id: str, evals_root: Path | None = None) -> dict[int, dict]:
    """step -> {"recall": score, "p6": score, "suite": dir} from the newest suite of each `<config_id>@s<step>`."""
    root = evals_root or OUTPUTS_DIR / "evals"
    out: dict[int, dict] = {}
    if not root.exists():
        return out
    pat = re.compile(rf"^{re.escape(config_id)}@s(\d+)$")
    for d in sorted(root.iterdir()):
        m = pat.match(d.name)
        if not m or not (d / "suite").exists():
            continue
        suites = sorted(p for p in (d / "suite").iterdir() if (p / "suite.json").exists())
        if not suites:
            continue
        manifest = json.loads((suites[-1] / "suite.json").read_text(encoding="utf-8"))
        quiz_dir = manifest.get("components", {}).get("quiz")
        summary_path = Path(quiz_dir) / "summary.json" if quiz_dir else None
        if summary_path is not None and not summary_path.is_absolute():
            summary_path = REPO_ROOT / summary_path
        if summary_path is None or not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        out[int(m.group(1))] = {q: (s.get(q) or {}).get("mean_correct", {}).get("mean") for q in ("recall", "p6")} | {
            "suite": str(suites[-1]).replace("\\", "/")
        }
    return out


def retention_flags(quizzes: dict[int, dict]) -> list[dict]:
    flags = []
    for step, q in sorted(quizzes.items()):
        for name in ("recall", "p6"):
            v = q.get(name)
            if v is not None and v < QUIZ_MIN:
                flags.append({"flag": f"knowledge_{name}", "step": step, "value": v, "threshold": QUIZ_MIN})
    return flags


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def render(
    rows: list[dict], flags: list[dict], quizzes: dict[int, dict], every: int, run_dir: Path, traj: list[dict]
) -> str:
    cols = [(k, h) for k, h in COLUMNS if any(r.get(k) is not None for r in rows)]
    lines = [f"# RL monitor: `{str(run_dir).replace(chr(92), '/')}`", "", f"{len(rows)} logged steps.", ""]
    if rows:
        lines += ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
        shown = [r for i, r in enumerate(rows) if i % every == 0 or i == len(rows) - 1]
        lines += ["| " + " | ".join(_fmt(r.get(k)) for k, _ in cols) + " |" for r in shown]
        step_s = [r["step_s"] for r in rows if r.get("step_s")]
        if step_s:
            lines += ["", f"Step time: mean {sum(step_s) / len(step_s):.0f} s, last {step_s[-1]:.0f} s."]
    if traj:
        lines += [
            "",
            "RL hold-out (outcome reward; gains vs step 0 / vs the first 5 training steps; train = rolling mean of 10 "
            "steps on generated dilemma rows):",
            "",
            "| step | hold-out | SE | hold-out gain | train | train gain | gap | cite | mention |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        keys = ("holdout", "holdout_se", "holdout_gain", "train", "train_gain", "gap", "r_cite", "mention_rate")
        lines += [f"| {t['step']} | " + " | ".join(_fmt(t[k]) for k in keys) + " |" for t in traj]
    if quizzes:
        lines += ["", "| checkpoint | recall quiz | P6 quiz |", "|---|---:|---:|"]
        lines += [f"| s{s} | {_fmt(q.get('recall'))} | {_fmt(q.get('p6'))} |" for s, q in sorted(quizzes.items())]
    lines += ["", "## Flags", ""]
    lines += [f"- **{f['flag']}** at step {f['step']}: {f['value']} (threshold {f['threshold']})" for f in flags] or [
        "none"
    ]
    return "\n".join(lines) + "\n"


def run(run_dir: Path, config_id: str | None = None, every: int = 1, evals_root: Path | None = None) -> dict:
    rows = read_steps(run_dir)
    resolved = (
        yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        if (run_dir / "resolved_config.yaml").exists()
        else {}
    )
    add_kl_term(rows, (resolved.get("grpo") or {}).get("beta"))
    add_letter_excess(rows, run_dir)
    if config_id is None:
        config_id = resolved.get("run_name")
    quizzes = checkpoint_quizzes(config_id, evals_root) if config_id else {}
    traj = holdout_trajectory(rows, read_summaries(run_dir))
    flags = trajectory_flags(rows) + retention_flags(quizzes) + holdout_flags(traj)
    md = render(rows, flags, quizzes, every, run_dir, traj)
    out = {"run_dir": str(run_dir).replace("\\", "/"), "config_id": config_id, "n_steps": len(rows), "flags": flags,
           "quizzes": quizzes, "holdout": traj}  # fmt: skip
    (run_dir / "monitor.md").write_text(md, encoding="utf-8")
    (run_dir / "monitor.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out | {"markdown": md}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config-id", default=None, help="eval config prefix of the checkpoints (default: run_name)")
    ap.add_argument("--every", type=int, default=1, help="show every N-th step in the table")
    ap.add_argument("--fail-on-flag", action="store_true", help="exit 1 if any flag fires")
    args = ap.parse_args(argv)
    out = run(args.run_dir, args.config_id, args.every)
    print(out["markdown"])
    if args.fail_on_flag and out["flags"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
