"""Evaluate trained probes on the hard (agentic-misalignment) data: harm, constitution score and the 2x2 regrouping.

CLI (CPU, local):
    uv run python -m calign.probe.evaluate --probes outputs/probes/<run> --hard outputs/misalignment/<run>
        [--config configs/probe.yaml] [--out outputs/probe_eval/<run>]

Requires every hard sample to carry `activations` (calign.probe.activations --misalignment-run), a
`constitution_score` and the `constitution-score-v2` judge fields (`mentions_constitution`); the run is refused
otherwise, so no number is computed on a partially scored run.

Every probe is projected at every stored hard-data position (prompt_last, pre_tool, p100, mean); the "matching"
pairs (same position name, and decision <-> pre_tool) are flagged in the summary. Per (probe, position):
- AUROC of the projection for `not harmful` (upstream binary verdict) and for `constitution_score >= outcome_threshold`
  (a value above 0.5 means the probe's positive direction goes with the aligned samples), Spearman with the score;
- the 2x2 rebuilt from process = judge mentions >= process_threshold and outcome = score >= outcome_threshold, regrouped
  with the probe's own label spec (prompt variants ignored), and the projection gap between its positive and
  negative group with a bootstrap CI (clusters = conditions, since samples of one condition share the prompt);
- mean projection per condition.
Output: scores.jsonl (one row per probe x position x sample) and summary.{json,md}, recomputable from the raw files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from calign.config import add_common_args, git_commit, new_run_dir, sha256_file
from calign.misalignment.constitution_judge import PROMPT_VERSION as CJ_VERSION
from calign.misalignment.report import SAMPLES_FILE
from calign.paths import REPO_ROOT
from calign.probe.config import ProbeConfig, load_probe_config
from calign.probe.labels import BUILTIN_SPECS, Cell, LabelSpec
from calign.probe.metrics import auroc, classification_summary, cluster_bootstrap, spearman
from calign.probe.store import ActivationStore
from calign.schemas import MisalignmentSample, ProbeRecord, read_jsonl, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)
RUN_KIND = "probe_eval"
MATCHING = {("decision", "pre_tool"), ("p100", "p100"), ("mean", "mean"), ("prompt_last", "prompt_last")}


def sample_key(s: MisalignmentSample) -> str:
    return f"{s.condition_id}#{s.sample_idx}"


def check_hard_run(samples: list[MisalignmentSample], store: ActivationStore | None = None) -> None:
    """Refuse to evaluate unless every sample has activations (in `store`, keyed condition#idx), a score and v2 fields."""
    missing = {
        "activations": sum(
            (sample_key(s) not in store) if store is not None else s.activations is None for s in samples
        ),
        "constitution_score": sum(s.constitution_score is None for s in samples),
        "judge_v2": sum((s.constitution_judge or {}).get("prompt_version") != CJ_VERSION for s in samples),
        "mentions_constitution": sum(
            (s.constitution_judge or {}).get("mentions_constitution") is None for s in samples
        ),
    }
    bad = {k: v for k, v in missing.items() if v}
    if bad:
        raise SystemExit(f"hard run is not ready: samples missing {bad} of {len(samples)}; score/extract it first")


def hard_cell(s: MisalignmentSample, cfg: ProbeConfig) -> Cell:
    proc = s.constitution_judge["mentions_constitution"] >= cfg.hard_data.process_threshold
    out = s.constitution_score >= cfg.hard_data.outcome_threshold
    if proc:
        return "mentioned_aligned" if out else "mentioned_misaligned"
    return "unmentioned_aligned" if out else "unmentioned_misaligned"


def spec_label(c: Cell, spec: LabelSpec) -> int | None:
    if c in spec.positive:
        return 1
    if c in spec.negative:
        return 0
    return None


def load_probes(probes_run: Path) -> tuple[list[ProbeRecord], np.ndarray]:
    from safetensors.numpy import load_file

    probes = read_jsonl(probes_run / "probes.jsonl", ProbeRecord)
    dirs = load_file(str(probes_run / "directions.safetensors"))["directions"]
    return probes, dirs


def evaluate_hard(
    probes: list[ProbeRecord],
    dirs: np.ndarray,
    samples: list[MisalignmentSample],
    store: ActivationStore,
    cfg: ProbeConfig,
    specs: dict[str, LabelSpec] | None = None,
) -> tuple[dict, list[dict]]:
    specs = specs or {s.name: s for s in cfg.labels.resolved_specs()} | BUILTIN_SPECS
    boot = cfg.train.bootstrap
    keys = [sample_key(s) for s in samples]
    conds = [s.condition_id for s in samples]
    not_harmful = np.array([0 if s.harmful else 1 for s in samples], dtype=int)
    scores = np.array([s.constitution_score for s in samples], dtype=float)
    aligned = (scores >= cfg.hard_data.outcome_threshold).astype(int)
    cells = [hard_cell(s, cfg) for s in samples]
    results: dict[str, dict] = {}
    rows: list[dict] = []
    for p in probes:
        d = dirs[p.direction_row]
        spec = specs[p.label_spec]
        labels = np.array([-1 if (y := spec_label(c, spec)) is None else y for c in cells], dtype=int)
        for hp in store.positions:
            x = store.matrix(keys, p.layer, hp)
            proj = x @ d
            centred = proj - p.threshold_midpoint
            key = f"{p.probe_id}@{hp}"
            mask = labels >= 0
            group = None
            if mask.sum() and labels[mask].min() == 0 and labels[mask].max() == 1:
                pm, cm = proj[mask], labels[mask]
                cl = [c for c, m in zip(conds, mask, strict=True) if m]
                gap = float(pm[cm == 1].mean() - pm[cm == 0].mean())
                lo, hi = cluster_bootstrap(
                    cl,
                    lambda i, pm=pm, cm=cm: (
                        (pm[i][cm[i] == 1].mean() - pm[i][cm[i] == 0].mean())
                        if (cm[i] == 1).any() and (cm[i] == 0).any()
                        else None
                    ),
                    boot.n,
                    boot.seed,
                )
                cg = p.class_gap or None
                group = {
                    "n_pos": int((cm == 1).sum()),
                    "n_neg": int((cm == 0).sum()),
                    "projection_gap": gap,
                    "projection_gap_ci95": [lo, hi],
                    "gap_in_class_gap_units": gap / cg if cg else None,
                    "gap_ci95_in_class_gap_units": [None if (v is None or not cg) else v / cg for v in (lo, hi)],
                    "auroc": auroc(pm, cm),
                }
            results[key] = {
                "probe_id": p.probe_id,
                "label_spec": p.label_spec,
                "layer": p.layer,
                "probe_position": p.position,
                "hard_position": hp,
                "matching_position": (p.position, hp) in MATCHING,
                "n": len(samples),
                "not_harmful": classification_summary(
                    proj, not_harmful, conds, p.threshold_midpoint, boot.n, boot.seed
                ),
                "aligned_by_score": classification_summary(
                    proj, aligned, conds, p.threshold_midpoint, boot.n, boot.seed
                ),
                "spearman_with_score": spearman(proj, scores),
                "spec_groups": group,
                "cells": {c: int(sum(1 for x in cells if x == c)) for c in sorted(set(cells))},
                "mean_projection_by_condition": {
                    c: float(np.mean([v for v, cc in zip(proj, conds, strict=True) if cc == c]))
                    for c in sorted(set(conds))
                },
                "share_above_midpoint": float((centred > 0).mean()),
            }
            for s, c, pr, y in zip(samples, cells, proj, labels, strict=True):
                rows.append(
                    {
                        "probe_id": p.probe_id,
                        "hard_position": hp,
                        "sample": sample_key(s),
                        "condition_id": s.condition_id,
                        "scenario": s.scenario,
                        "harmful": s.harmful,
                        "constitution_score": s.constitution_score,
                        "mentions_constitution": s.constitution_judge["mentions_constitution"],
                        "cell": c,
                        "label": None if y < 0 else int(y),
                        "projection": float(pr),
                    }
                )
    return {"kind": RUN_KIND, "n_samples": len(samples), "hard_positions": store.positions, "results": results}, rows


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render_markdown(s: dict) -> str:
    L = ["# Probe evaluation on agentic-misalignment data", ""]
    if s.get("excluded_samples"):
        L.append(f"Excluded (no constitution score): {', '.join(e['sample'] for e in s['excluded_samples'])}  ")
    L.append(
        f"{s['n_samples']} samples; hard positions {', '.join(s['hard_positions'])}. AUROC > 0.5 = the probe's positive direction goes with the aligned / non-harmful samples."
    )
    L += ["", "## Matching positions (probe position <-> hard position)", ""]
    L.append(
        "| probe | hard pos | AUROC not-harmful [CI] | AUROC score>=thr [CI] | Spearman(score) | 2x2 group gap (class-gap units) [CI] | group AUROC |"
    )
    L.append("|---|---|---|---|---:|---|---:|")
    for r in s["results"].values():
        if r["matching_position"]:
            L.append(_row(r))
    L += ["", "## All probe x position combinations", ""]
    L.append(
        "| probe | hard pos | AUROC not-harmful [CI] | AUROC score>=thr [CI] | Spearman(score) | 2x2 group gap (class-gap units) [CI] | group AUROC |"
    )
    L.append("|---|---|---|---|---:|---|---:|")
    for r in s["results"].values():
        L.append(_row(r))
    any_r = next(iter(s["results"].values()), None)
    if any_r:
        L += ["", "Hard-data 2x2 counts: " + ", ".join(f"{c}={n}" for c, n in any_r["cells"].items())]
    return "\n".join(L) + "\n"


def _row(r: dict) -> str:
    nh, al, g = r["not_harmful"], r["aligned_by_score"], r["spec_groups"]
    gap = "n/a"
    if g:
        lo, hi = g["gap_ci95_in_class_gap_units"]
        gap = f"{_fmt(g['gap_in_class_gap_units'])} [{_fmt(lo)}, {_fmt(hi)}] (n={g['n_pos']}/{g['n_neg']})"
    return (
        f"| {r['probe_id']} | {r['hard_position']} | {_fmt(nh['auroc'])} [{_fmt(nh['auroc_ci95'][0])}, {_fmt(nh['auroc_ci95'][1])}] | "
        f"{_fmt(al['auroc'])} [{_fmt(al['auroc_ci95'][0])}, {_fmt(al['auroc_ci95'][1])}] | {_fmt(r['spearman_with_score'])} | {gap} | "
        f"{_fmt(g['auroc']) if g else 'n/a'} |"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "probe.yaml")
    ap.add_argument("--probes", type=Path, required=True, help="probes run dir (calign.probe.train)")
    ap.add_argument(
        "--hard", type=Path, default=None, help="misalignment run dir; default hard_data.run_dir of the config"
    )
    ap.add_argument(
        "--exclude-unscored",
        action="store_true",
        help="drop samples without a constitution score / v2 mention field (listed in summary.json) instead of refusing",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_probe_config(args.config)
    hard = args.hard or REPO_ROOT / cfg.hard_data.run_dir
    samples = read_jsonl(hard / SAMPLES_FILE, MisalignmentSample)
    if args.limit:
        samples = samples[: args.limit]
    excluded: list[dict] = []
    if args.exclude_unscored:
        keep = []
        for s in samples:
            j = s.constitution_judge or {}
            if s.constitution_score is None or j.get("mentions_constitution") is None:
                excluded.append({"sample": sample_key(s), "harmful": s.harmful, "judge_error": j.get("error")})
            else:
                keep.append(s)
        samples = keep
        LOGGER.warning("excluded %d unscored samples: %s", len(excluded), [e["sample"] for e in excluded])
    store = ActivationStore(hard)
    check_hard_run(samples, store)
    probes, dirs = load_probes(args.probes)
    if args.dry_run:
        probes = probes[:3]
    run_dir = new_run_dir(
        RUN_KIND,
        {"probe": cfg.model_dump(), "probes_run": str(args.probes), "hard_run": str(hard)},
        out=args.out,
        dry_run=args.dry_run,
    )
    summary, rows = evaluate_hard(probes, dirs, samples, store, cfg)
    summary["excluded_samples"] = excluded
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary["provenance"] = {
        "probes_run": str(args.probes),
        "probes_sha256": sha256_file(args.probes / "probes.jsonl"),
        "hard_run": str(hard),
        "samples_sha256": sha256_file(hard / SAMPLES_FILE),
        "config": cfg.model_dump(),
        "git_commit": git_commit(),
        "generated_at": utc_now_iso(),
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
