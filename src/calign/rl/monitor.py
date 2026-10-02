"""RL run monitor: trajectory table and stop-and-check-in flags (chunks 7-8).

CLI (local after rsync, or on the instance while the run is going):
    uv run python -m calign.rl.monitor --run-dir outputs/rl/C4_pilot [--config-id C4] [--every 5] [--fail-on-flag]
writes <run dir>/monitor.md and monitor.json and prints the table.

Flags (rolling mean over the last `window` = 5 logged steps where a per-step rate is noisy; ~28 math and ~100
dilemma completions per step):
- length: mean completion length > 2 x the start (mean of the first 3 logged steps);
- math_mentions: regex constitution mention rate on math rows > 5% (rolling);
- letter_prior: letter-A share of parsed dilemma decisions outside [0.35, 0.65] (rolling);
- zero_variance: share of prompt groups with identical rewards > 60% (rolling);
- knowledge retention (from the core suite of each checkpoint, eval configs `<config-id>@s<step>`): recall quiz < 0.8
  or P6 quiz < 0.8 (RL never trains on P6, so this checks that RL does not erode what the model knows).
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
from calign.rl.train_grpo import STEPS_FILE

WINDOW = 5
LENGTH_FACTOR = 2.0
MATH_MENTION_MAX = 0.05
LETTER_A_RANGE = (0.35, 0.65)
ZERO_VAR_MAX = 0.60
QUIZ_MIN = 0.8

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
    ("parse_rate/moral", "parse"),
    ("zero_var_share/all", "zero-var"),
    ("length/mean", "len"),
    ("length/p90", "len p90"),
    ("length/truncated_share", "trunc"),
    ("kl", "KL"),
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
    # keep the per-step training logs (the final train summary line has no reward metrics)
    return [r for r in rows if "train_runtime" not in r]


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
        ("letter_prior", "letter_a_share", lambda v: not LETTER_A_RANGE[0] <= v <= LETTER_A_RANGE[1], LETTER_A_RANGE),
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


def render(rows: list[dict], flags: list[dict], quizzes: dict[int, dict], every: int, run_dir: Path) -> str:
    cols = [(k, h) for k, h in COLUMNS if any(r.get(k) is not None for r in rows)]
    lines = [f"# RL monitor: `{str(run_dir).replace(chr(92), '/')}`", "", f"{len(rows)} logged steps.", ""]
    if rows:
        lines += ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
        shown = [r for i, r in enumerate(rows) if i % every == 0 or i == len(rows) - 1]
        lines += ["| " + " | ".join(_fmt(r.get(k)) for k, _ in cols) + " |" for r in shown]
        step_s = [r["step_s"] for r in rows if r.get("step_s")]
        if step_s:
            lines += ["", f"Step time: mean {sum(step_s) / len(step_s):.0f} s, last {step_s[-1]:.0f} s."]
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
    if config_id is None and (run_dir / "resolved_config.yaml").exists():
        config_id = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")).get("run_name")
    quizzes = checkpoint_quizzes(config_id, evals_root) if config_id else {}
    flags = trajectory_flags(rows) + retention_flags(quizzes)
    md = render(rows, flags, quizzes, every, run_dir)
    out = {"run_dir": str(run_dir).replace("\\", "/"), "config_id": config_id, "n_steps": len(rows), "flags": flags,
           "quizzes": quizzes}  # fmt: skip
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
