"""Train difference-of-means probes for every (label spec, layer, position) from a probe_data run.

CLI (CPU, local; needs the run's records.jsonl with judge + activations and its activations/ shards):
    uv run python -m calign.probe.train --data-run outputs/probe_data/<run> [--config configs/probe.yaml]
        [--out outputs/probes/<run>] [--dry-run] [--limit N]

For each spec (calign.probe.labels) the training set is the config's train split and validation the val split of
the data run (scenario-disjoint by construction). Direction d = mean(X_pos) - mean(X_neg) on the train split,
unit-normalised; score = x . d; threshold = midpoint of the projected class means. Metrics: AUROC (primary) and
balanced accuracy with 95% CIs from a cluster bootstrap over scenarios, on train and on val.

Output run dir (immutable):
    probes.jsonl              one ProbeRecord per probe (ids "<spec>/L<layer>/<position>")
    directions.safetensors    "directions" (n_probes, d) unit fp32; "class_means" (n_probes, 2, d) [pos, neg]
    scores.jsonl              one row per (probe, labelled record): projection + label/cell/split/scenario, so every
                              metric can be recomputed offline without the activation shards
    convergence.json          cosine matrix between all probe directions (+ random baseline 1/sqrt(d))
    summary.json, summary.md  metrics table, best probe per spec (by val AUROC)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from calign.config import add_common_args, effective_limit, git_commit, new_run_dir, sha256_file
from calign.paths import REPO_ROOT
from calign.probe.config import ProbeConfig, load_probe_config
from calign.probe.labels import LabelSpec, cell, label
from calign.probe.metrics import best_threshold, classification_summary
from calign.probe.store import ActivationStore
from calign.schemas import (
    ConstitutionVerdict,
    GenerationRecord,
    ProbeRecord,
    read_jsonl,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from calign.validate.verdicts import load_verdicts

LOGGER = logging.getLogger(__name__)
RUN_KIND = "probes"


def probe_id(spec: str, layer: int, position: str) -> str:
    return f"{spec}/L{layer}/{position}"


def direction_sha(v: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(v, dtype=np.float32).tobytes()).hexdigest()[:16]


def labelled_records(
    records: list[GenerationRecord], verdicts: dict[str, ConstitutionVerdict], spec: LabelSpec, cfg: ProbeConfig
) -> list[tuple[GenerationRecord, int, str]]:
    """(record, label, cell) for records with judge + activations that fall in the spec's classes."""
    th = cfg.labels.thresholds
    out = []
    for r in records:
        if r.activations is None or r.judge is None:
            continue
        y = label(r, verdicts.get(r.scenario_id), spec, th)
        if y is None:
            continue
        out.append((r, y, cell(r.judge, verdicts.get(r.scenario_id), th)))
    return out


def fit_diff_means(x: np.ndarray, y: np.ndarray) -> dict:
    """Unit direction from the class-mean difference, its class means, gap and midpoint threshold."""
    mu_pos, mu_neg = x[y == 1].mean(axis=0), x[y == 0].mean(axis=0)
    diff = mu_pos - mu_neg
    gap = float(np.linalg.norm(diff))
    if gap == 0.0:
        raise ValueError("class means coincide; no direction")
    d = (diff / gap).astype(np.float32)
    return {
        "direction": d,
        "mu_pos": mu_pos.astype(np.float32),
        "mu_neg": mu_neg.astype(np.float32),
        "class_gap": gap,
        "threshold_midpoint": float(d @ (mu_pos + mu_neg) / 2),
    }


def train_probes(
    data_run: Path,
    records: list[GenerationRecord],
    store: ActivationStore,
    cfg: ProbeConfig,
    verdicts: dict[str, ConstitutionVerdict] | None = None,
    specs: list[LabelSpec] | None = None,
    layers: list[int] | None = None,
    positions: list[str] | None = None,
) -> tuple[list[ProbeRecord], np.ndarray, np.ndarray, list[dict]]:
    """Returns (probe records, directions (n, d), class_means (n, 2, d), score rows)."""
    verdicts = load_verdicts() if verdicts is None else verdicts
    specs = specs or cfg.labels.resolved_specs()
    layers = layers or store.layers
    positions = positions or store.positions
    boot = cfg.train.bootstrap
    probes: list[ProbeRecord] = []
    dirs: list[np.ndarray] = []
    means: list[np.ndarray] = []
    score_rows: list[dict] = []
    for spec in specs:
        rows = labelled_records(records, verdicts, spec, cfg)
        train = [t for t in rows if str(t[0].split) == cfg.sampling.train_split]
        val = [t for t in rows if str(t[0].split) == cfg.sampling.val_split]
        y_tr = np.array([y for _, y, _ in train], dtype=int)
        y_va = np.array([y for _, y, _ in val], dtype=int)
        if y_tr.sum() == 0 or (1 - y_tr).sum() == 0:
            LOGGER.warning(
                "spec %s: train split lacks a class (pos=%d, neg=%d); skipped", spec.name, y_tr.sum(), (1 - y_tr).sum()
            )
            continue
        ids_tr = [r.record_id for r, _, _ in train]
        ids_va = [r.record_id for r, _, _ in val]
        scen_tr = [r.scenario_id for r, _, _ in train]
        scen_va = [r.scenario_id for r, _, _ in val]
        for layer in layers:
            for position in positions:
                x_tr = store.matrix(ids_tr, layer, position)
                fit = fit_diff_means(x_tr, y_tr)
                d = fit["direction"]
                s_tr = x_tr @ d
                thr = fit["threshold_midpoint"]
                train_metrics = classification_summary(s_tr, y_tr, scen_tr, thr, boot.n, boot.seed)
                train_metrics["threshold_best_train"] = best_threshold(s_tr, y_tr)
                if val:
                    x_va = store.matrix(ids_va, layer, position)
                    s_va = x_va @ d
                    val_metrics = classification_summary(s_va, y_va, scen_va, thr, boot.n, boot.seed)
                else:
                    s_va, val_metrics = np.zeros(0), {"n": 0, "auroc": None, "balanced_accuracy": None}
                pid = probe_id(spec.name, layer, position)
                probes.append(
                    ProbeRecord(
                        probe_id=pid,
                        label_spec=spec.name,
                        layer=layer,
                        position=position,
                        data_run=str(data_run),
                        n_pos=int(y_tr.sum()),
                        n_neg=int((1 - y_tr).sum()),
                        n_scenarios=len(set(scen_tr)),
                        direction_row=len(dirs),
                        direction_sha=direction_sha(d),
                        class_gap=fit["class_gap"],
                        threshold_midpoint=thr,
                        mean_resid_norm=float(np.linalg.norm(x_tr, axis=1).mean()),
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                    )
                )
                dirs.append(d)
                means.append(np.stack([fit["mu_pos"], fit["mu_neg"]]))
                for (r, y, c), s in list(zip(train, s_tr, strict=True)) + list(zip(val, s_va, strict=True)):
                    score_rows.append(
                        {
                            "probe_id": pid,
                            "record_id": r.record_id,
                            "scenario_id": r.scenario_id,
                            "split": r.split,
                            "variant": r.condition.prompt_variant,
                            "cell": c,
                            "label": int(y),
                            "projection": float(s),
                        }
                    )
                LOGGER.info(
                    "%s: n=%d/%d val AUROC %s bal.acc %s",
                    pid,
                    y_tr.sum(),
                    (1 - y_tr).sum(),
                    _fmt(val_metrics.get("auroc")),
                    _fmt(val_metrics.get("balanced_accuracy")),
                )
    if not dirs:
        raise SystemExit("no probe could be trained (no spec had both classes in the train split)")
    return probes, np.stack(dirs), np.stack(means), score_rows


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def convergence(probes: list[ProbeRecord], dirs: np.ndarray) -> dict:
    cos = dirs @ dirs.T
    return {
        "probe_ids": [p.probe_id for p in probes],
        "cosine": np.round(cos, 6).tolist(),
        "random_baseline_abs_cos": float(1 / np.sqrt(dirs.shape[1])),
        "d_model": int(dirs.shape[1]),
    }


def summarize(probes: list[ProbeRecord], conv: dict) -> dict:
    by_spec: dict[str, list[ProbeRecord]] = {}
    for p in probes:
        by_spec.setdefault(p.label_spec, []).append(p)
    best = {}
    for spec, ps in by_spec.items():
        ranked = sorted(ps, key=lambda p: -(p.val_metrics.get("auroc") or 0.0))
        best[spec] = {
            "probe_id": ranked[0].probe_id,
            "val_auroc": ranked[0].val_metrics.get("auroc"),
            "val_balanced_accuracy": ranked[0].val_metrics.get("balanced_accuracy"),
            "layer": ranked[0].layer,
            "position": ranked[0].position,
        }
    # B vs C convergence at matching (layer, position)
    ids = conv["probe_ids"]
    cos = np.array(conv["cosine"])
    pairs = []
    for i, a in enumerate(ids):
        sa, la, pa = a.split("/")
        if not sa.startswith("B_"):
            continue
        for j, b in enumerate(ids):
            sb, lb, pb = b.split("/")
            if sb.startswith("C_") and la == lb and pa == pb:
                pairs.append({"b": a, "c": b, "cosine": float(cos[i, j])})
    return {
        "kind": RUN_KIND,
        "n_probes": len(probes),
        "specs": sorted(by_spec),
        "layers": sorted({p.layer for p in probes}),
        "positions": list(dict.fromkeys(p.position for p in probes)),
        "best_by_spec": best,
        "b_vs_c_cosine": pairs,
        "random_baseline_abs_cos": conv["random_baseline_abs_cos"],
        "probes": {
            p.probe_id: {
                "n_pos": p.n_pos,
                "n_neg": p.n_neg,
                "class_gap": p.class_gap,
                "mean_resid_norm": p.mean_resid_norm,
                "train_auroc": p.train_metrics.get("auroc"),
                "val_auroc": p.val_metrics.get("auroc"),
                "val_auroc_ci95": p.val_metrics.get("auroc_ci95"),
                "val_balanced_accuracy": p.val_metrics.get("balanced_accuracy"),
                "val_balanced_accuracy_ci95": p.val_metrics.get("balanced_accuracy_ci95"),
                "val_n": p.val_metrics.get("n"),
            }
            for p in probes
        },
    }


def render_markdown(s: dict) -> str:
    L = ["# Probes (difference of means)", ""]
    L.append(
        f"{s['n_probes']} probes; specs {', '.join(s['specs'])}; layers {s['layers']}; positions {', '.join(s['positions'])}"
    )
    L += ["", "## Best probe per spec (val AUROC)", ""]
    for spec, b in s["best_by_spec"].items():
        L.append(
            f"- **{spec}**: {b['probe_id']} val AUROC {_fmt(b['val_auroc'])}, balanced acc {_fmt(b['val_balanced_accuracy'])}"
        )
    L += ["", "## All probes", ""]
    L.append(
        "| probe | n_pos/n_neg | train AUROC | val AUROC [95% CI] | val bal. acc [95% CI] | class gap | mean ||x|| |"
    )
    L.append("|---|---:|---:|---|---|---:|---:|")
    for pid, p in s["probes"].items():
        ci_a, ci_b = p["val_auroc_ci95"] or [None, None], p["val_balanced_accuracy_ci95"] or [None, None]
        L.append(
            f"| {pid} | {p['n_pos']}/{p['n_neg']} | {_fmt(p['train_auroc'])} | {_fmt(p['val_auroc'])} [{_fmt(ci_a[0])}, {_fmt(ci_a[1])}] | "
            f"{_fmt(p['val_balanced_accuracy'])} [{_fmt(ci_b[0])}, {_fmt(ci_b[1])}] | {p['class_gap']:.1f} | {p['mean_resid_norm']:.0f} |"
        )
    if s["b_vs_c_cosine"]:
        L += ["", f"## B vs C convergence (cosine; random baseline |cos| ~ {s['random_baseline_abs_cos']:.3f})", ""]
        for pr in s["b_vs_c_cosine"]:
            L.append(f"- {pr['b']} vs {pr['c']}: {pr['cosine']:.3f}")
    return "\n".join(L) + "\n"


def write_outputs(run_dir: Path, data_run: Path, probes, dirs, means, score_rows, cfg: ProbeConfig) -> dict:
    from safetensors.numpy import save_file

    write_jsonl(run_dir / "probes.jsonl", probes)
    save_file(
        {"directions": dirs.astype(np.float32), "class_means": means.astype(np.float32)},
        str(run_dir / "directions.safetensors"),
    )
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for row in score_rows:
            f.write(json.dumps(row) + "\n")
    conv = convergence(probes, dirs)
    write_json(run_dir / "convergence.json", conv)
    s = summarize(probes, conv)
    s["provenance"] = {
        "data_run": str(data_run),
        "records_sha256": sha256_file(data_run / "records.jsonl"),
        "activations_index_sha256": sha256_file(data_run / "activations" / "index.json"),
        "config": cfg.model_dump(),
        "git_commit": git_commit(),
        "generated_at": utc_now_iso(),
    }
    write_json(run_dir / "summary.json", s)
    (run_dir / "summary.md").write_text(render_markdown(s), encoding="utf-8")
    return s


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "probe.yaml")
    ap.add_argument("--data-run", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_probe_config(args.config)
    records = read_jsonl(args.data_run / "records.jsonl", GenerationRecord)
    store = ActivationStore(args.data_run)
    specs = cfg.labels.resolved_specs()
    layers, positions = store.layers, store.positions
    limit = effective_limit(args)
    if args.dry_run:
        specs, layers, positions = specs[:1], layers[:1], positions[: limit or 3]
    run_dir = new_run_dir(
        RUN_KIND, {"probe": cfg.model_dump(), "data_run": str(args.data_run)}, out=args.out, dry_run=args.dry_run
    )
    probes, dirs, means, rows = train_probes(
        args.data_run, records, store, cfg, specs=specs, layers=layers, positions=positions
    )
    write_outputs(run_dir, args.data_run, probes, dirs, means, rows, cfg)
    print((run_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
