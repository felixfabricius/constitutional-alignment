"""Run the Phase 3 core suite for one configuration: GPU generation with one shared vLLM load, then local judging.

CLI:
    # GPU: generation for every listed component, each into its own run dir; writes the suite manifest
    uv run python -m calign.evals.suite --eval-config C0 \
        [--components moralchoice ifeval math500 overcitation quiz] [--all-clear] [--dry-run] [--limit N]
    # local (Claude): judged parts and reports, against the C0 suite as reference for other configurations
    uv run python -m calign.evals.suite --judge-only --suite-run outputs/evals/C0/suite/<run> [--coherence-rep1] [--no-batches]

GPU components: moralchoice (dev, eval1, eval2 at k=4, T=0.7; `--all-clear` samples every clear item, which is
how the C0 base run doubles as the hard-subset source), ifeval, math500, overcitation (the low-ambiguity sampling),
quiz. Judge-only: MoralChoice judged sample (200), over-citation judge on IFEval + MATH-500 regex hits, coherence v2
on the fixed 60-text set (`--coherence-rep1` adds the repeatability re-score), quiz grading, then every
component's report. `coherence` is listed as a component for the judge phase; it needs no GPU.

The suite manifest `outputs/evals/<id>/suite/<stamp>/suite.json` maps component -> run dir and records per-component
wall-clock seconds; `calign.evals.report` reads the newest suite manifest of each configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from calign.evals import ifeval, math500, moralchoice, overcitation, quiz
from calign.evals.common import item_limit, load_eval_backend
from calign.evals.config import EVALS_DIR, EvalConfig, eval_run_dir, load_eval_config
from calign.paths import OUTPUTS_DIR, REPO_ROOT
from calign.schemas import read_json, write_json

LOGGER = logging.getLogger(__name__)

GPU_COMPONENTS = ("moralchoice", "ifeval", "math500", "overcitation", "quiz")
JUDGE_COMPONENTS = ("moralchoice", "overcitation", "coherence", "quiz")
ALL_COMPONENTS = GPU_COMPONENTS + ("coherence",)
REFERENCE_CONFIG = "C0"


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p).replace("\\", "/")


def run_gpu(
    cfg: EvalConfig,
    components: list[str],
    all_clear: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    out_root: Path | None = None,
) -> Path:
    t0 = time.time()
    backend, model_cfg = load_eval_backend(cfg, seed=moralchoice.DEFAULT_SEED)
    timings = {"model_load": round(time.time() - t0, 1)}
    runs: dict[str, str] = {}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for comp in components:
        if comp not in GPU_COMPONENTS:
            continue
        t = time.time()
        if comp == "moralchoice":
            params = moralchoice.SampleParams(all_clear=all_clear, limit=limit)
            rd = eval_run_dir(cfg, comp, params.model_dump(), out_root=out_root, dry_run=dry_run, model_cfg=model_cfg)
            moralchoice.sample_component(backend, cfg, model_cfg, rd, params, verbose=dry_run)
            moralchoice.write_report(rd)
        else:
            mod = {"ifeval": ifeval, "math500": math500, "overcitation": overcitation, "quiz": quiz}[comp]
            seed = overcitation.DEFAULT_SEED if comp == "overcitation" else 0
            rd = eval_run_dir(
                cfg, comp, {"limit": limit, "seed": seed}, out_root=out_root, dry_run=dry_run, model_cfg=model_cfg
            )
            mod.sample(backend, cfg, model_cfg, rd, limit=limit, seed=seed, verbose=dry_run)
            if comp in ("ifeval", "math500"):
                mod.write_report(rd)
        timings[comp] = round(time.time() - t, 1)
        runs[comp] = _rel(rd)
        LOGGER.info("%s done in %.0f s -> %s", comp, timings[comp], rd)
    root = (OUTPUTS_DIR / "dry_run" / "evals") if dry_run else (out_root or EVALS_DIR)
    suite_dir = root / cfg.id / "suite" / stamp
    write_json(
        suite_dir / "suite.json",
        {
            "eval_config": cfg.model_dump(),
            "created_at": stamp,
            "components": runs,
            "timings_s": timings,
            "all_clear": all_clear,
            "limit": limit,
            "judged": {},
        },
    )
    LOGGER.info("suite manifest %s; timings %s", suite_dir / "suite.json", timings)
    return suite_dir


def latest_run_dir(cfg_id: str, root: Path | None = None) -> Path | None:
    base = (root or EVALS_DIR) / cfg_id / "suite"
    if not base.exists():
        return None
    runs = sorted(p for p in base.iterdir() if (p / "suite.json").exists())
    return runs[-1] if runs else None


def run_judges(
    suite_dir: Path, coherence_rep1: bool = False, use_batches: bool | None = None, components: list[str] | None = None
) -> dict:
    manifest = read_json(suite_dir / "suite.json")
    cfg = EvalConfig.model_validate(manifest["eval_config"])
    runs = {k: REPO_ROOT / v for k, v in manifest["components"].items()}
    components = components or list(JUDGE_COMPONENTS)
    ref_suite = None if cfg.id == REFERENCE_CONFIG else latest_run_dir(REFERENCE_CONFIG)
    ref_runs = (
        {k: REPO_ROOT / v for k, v in read_json(ref_suite / "suite.json")["components"].items()} if ref_suite else {}
    )
    costs: dict[str, float] = {}

    if "moralchoice" in components and "moralchoice" in runs:
        u = moralchoice.judge_sample_component(runs["moralchoice"], use_batches=use_batches)
        costs["moralchoice_judge_sample"] = u["total_cost_usd"]
    if "overcitation" in components and "overcitation" in runs:
        srcs = [runs[c] for c in ("ifeval", "math500") if c in runs]
        u = overcitation.judge(runs["overcitation"], srcs, use_batches=use_batches)
        costs["overcitation_judge"] = u["total_cost_usd"]
    if "coherence" in components and {"moralchoice", "ifeval"} <= set(runs):
        from calign.evals import coherence

        rd = runs.get("coherence") or eval_run_dir(
            cfg,
            "coherence",
            {"moralchoice_run": _rel(runs["moralchoice"]), "ifeval_run": _rel(runs["ifeval"])},
            out_root=suite_dir.parents[2],
        )
        if not (rd / "scores.jsonl").exists():
            u = coherence.run_judge(rd, runs["moralchoice"], runs["ifeval"], use_batches=use_batches)
            costs["coherence"] = u["total_cost_usd"]
        if coherence_rep1:
            u = coherence.run_judge(rd, runs["moralchoice"], runs["ifeval"], salt="rep1", use_batches=use_batches)
            costs["coherence_rep1"] = u["total_cost_usd"]
        manifest["components"]["coherence"] = _rel(rd)
        runs["coherence"] = rd
    if "quiz" in components and "quiz" in runs:
        u = quiz.grade(runs["quiz"], use_batches=use_batches)
        costs["quiz_grade"] = u["total_cost_usd"]

    from calign.evals import coherence

    mods = {
        "moralchoice": moralchoice,
        "ifeval": ifeval,
        "math500": math500,
        "overcitation": overcitation,
        "quiz": quiz,
        "coherence": coherence,
    }
    reports = {c: mods[c].write_report(rd, ref_runs.get(c)) for c, rd in runs.items() if c in mods}
    manifest["judged"] = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "reference_suite": _rel(ref_suite) if ref_suite else None,
        "costs_usd": costs,
        "total_cost_usd": round(sum(costs.values()), 4),
    }
    write_json(suite_dir / "suite.json", manifest)
    LOGGER.info("judged suite %s: costs %s", suite_dir, costs)
    return reports


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-config", "--config", dest="eval_config", default=None)
    ap.add_argument("--components", nargs="+", default=list(ALL_COMPONENTS), choices=list(ALL_COMPONENTS))
    ap.add_argument("--all-clear", action="store_true", help="MoralChoice on every clear item (the C0 base run)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--judge-only", action="store_true")
    ap.add_argument("--suite-run", type=Path, default=None, help="suite dir for --judge-only (default: newest)")
    ap.add_argument("--coherence-rep1", action="store_true")
    ap.add_argument("--no-batches", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.judge_only:
        suite_dir = args.suite_run or latest_run_dir(load_eval_config(args.eval_config).id)
        if suite_dir is None:
            raise SystemExit("no suite run found; pass --suite-run")
        reports = run_judges(
            suite_dir, args.coherence_rep1, use_batches=False if args.no_batches else None, components=args.components
        )
        print(json.dumps({k: v.get("run_dir") for k, v in reports.items()}, indent=1))
        return
    if args.eval_config is None:
        raise SystemExit("--eval-config is required")
    cfg = load_eval_config(args.eval_config)
    limit = item_limit(argparse.Namespace(dry_run=args.dry_run, limit=args.limit))
    run_gpu(cfg, args.components, all_clear=args.all_clear, limit=limit, dry_run=args.dry_run, out_root=args.out_root)


if __name__ == "__main__":
    main()
