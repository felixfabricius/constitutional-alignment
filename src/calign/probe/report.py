"""Recompute summaries for Phase 2 run dirs from their raw files.

CLI:
    uv run python -m calign.probe.report --run-dir outputs/probe_data/<run> [--config configs/probe.yaml]

The run kind is read from run_meta.json (`probe_data`; steering and probe runs are added by their modules).
probe_data: per prompt variant the 2x2 cell counts (judge thresholds from the config), truncation and parse rates,
mention rate, mean outcome alignment, plus the class balance of every label spec (per split).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from calign.config import git_commit, sha256_file
from calign.paths import REPO_ROOT
from calign.probe.config import ProbeConfig, load_probe_config
from calign.probe.labels import CELLS, cell, label_counts
from calign.schemas import ConstitutionVerdict, GenerationRecord, read_jsonl, utc_now_iso, write_json
from calign.stats import rate_summary
from calign.validate.verdicts import load_verdicts


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize_probe_data(
    records: list[GenerationRecord],
    cfg: ProbeConfig,
    verdicts: dict[str, ConstitutionVerdict] | None = None,
    activation_ids: set[str] | None = None,
) -> dict:
    """`activation_ids`: record ids present in the run's activation store (None = use the records' own field)."""
    verdicts = load_verdicts() if verdicts is None else verdicts

    def has_acts(r: GenerationRecord) -> bool:
        return (r.record_id in activation_ids) if activation_ids is not None else r.activations is not None

    th = cfg.labels.thresholds
    variants: dict[str, dict] = {}
    by_variant: dict[str, list[GenerationRecord]] = defaultdict(list)
    for r in records:
        by_variant[r.condition.prompt_variant].append(r)
    for variant, rows in sorted(by_variant.items()):
        judged = [r for r in rows if r.judge is not None]
        cells = Counter(cell(r.judge, verdicts.get(r.scenario_id), th) or "unlabelled" for r in judged)
        n_lab = sum(v for k, v in cells.items() if k != "unlabelled")
        variants[variant] = {
            "n": len(rows),
            "n_judged": len(judged),
            "n_scenarios": len({r.scenario_id for r in rows}),
            "samples_per_scenario": Counter(Counter(r.scenario_id for r in rows).values()).most_common(1)[0][0]
            if rows
            else 0,
            "by_split": dict(Counter(str(r.split) for r in rows)),
            "cells": {c: cells.get(c, 0) for c in CELLS} | {"unlabelled": cells.get("unlabelled", 0)},
            "cell_rates": {c: rate_summary(cells.get(c, 0), n_lab) for c in CELLS},
            "mention_rate": rate_summary(
                sum(1 for r in judged if r.judge.mentions_constitution >= th.mention), len(judged)
            ),
            "mean_citation_accuracy_when_citing": _mean(
                [r.judge.citation_accuracy for r in judged if r.judge.principles_cited]
            ),
            "mean_outcome_alignment": _mean([r.judge.outcome_alignment for r in judged]),
            "decisions_parsed": dict(Counter(r.parsed_decision or "none" for r in rows)),
            "parse_rate": rate_summary(sum(1 for r in rows if r.parsed_decision in ("action1", "action2")), len(rows)),
            "truncated": rate_summary(sum(1 for r in rows if r.finish_reason == "length"), len(rows)),
            "mean_completion_tokens": _mean([len(r.extra.get("completion_token_ids", [])) or None for r in rows]),
            "n_with_activations": sum(1 for r in rows if has_acts(r)),
        }
    specs = {}
    for spec in cfg.labels.resolved_specs():
        specs[spec.name] = {
            "spec": spec.model_dump(),
            "all": label_counts(records, verdicts, spec, th),
            "by_split": {
                split: label_counts([r for r in records if str(r.split) == split], verdicts, spec, th)
                for split in sorted({str(r.split) for r in records})
            },
        }
    return {
        "kind": "probe_data",
        "n_records": len(records),
        "model": records[0].model.model_dump() if records else None,
        "variants": variants,
        "label_specs": specs,
        "thresholds": th.model_dump(),
    }


def _pct(r: dict | None) -> str:
    if not r or not r.get("n"):
        return "n/a"
    return f"{r['rate']:.0%} ({r['k']}/{r['n']})"


def _num(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def render_probe_data(s: dict) -> str:
    L = ["# Probe data summary", ""]
    m = s.get("model") or {}
    L.append(f"Model: `{m.get('path')}` (stage {m.get('stage')}); records: {s['n_records']}")
    L += ["", "## Per prompt variant", ""]
    L.append(
        "| variant | n | scenarios | judged | mentions | cit. acc. | outcome | parse | truncated | mean tokens | acts |"
    )
    L.append("|---|---:|---:|---:|---|---:|---:|---|---|---:|---:|")
    for v, c in s["variants"].items():
        L.append(
            f"| {v} | {c['n']} | {c['n_scenarios']} | {c['n_judged']} | {_pct(c['mention_rate'])} | "
            f"{_num(c['mean_citation_accuracy_when_citing'])} | {_num(c['mean_outcome_alignment'])} | {_pct(c['parse_rate'])} | "
            f"{_pct(c['truncated'])} | {_num(c['mean_completion_tokens'])} | {c['n_with_activations']} |"
        )
    th = s["thresholds"]
    L += [
        "",
        f"## 2x2 cells (process = mentions >= {th['mention']:.2f}; outcome = alignment >= {th['outcome']:.2f}; "
        "definite verdicts only)",
        "",
    ]
    L.append("| variant | " + " | ".join(CELLS) + " | unlabelled |")
    L.append("|---|" + "---:|" * (len(CELLS) + 1))
    for v, c in s["variants"].items():
        L.append(
            f"| {v} | "
            + " | ".join(f"{c['cells'][k]} ({c['cell_rates'][k]['rate']:.0%})" for k in CELLS)
            + f" | {c['cells']['unlabelled']} |"
        )
    L += ["", "## Label specs (positives / negatives / excluded; scenarios with both classes)", ""]
    L.append("| spec | split | n_pos | n_neg | n_excluded | scen. pos | scen. neg | scen. both |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, sp in s["label_specs"].items():
        for split, c in [("all", sp["all"])] + sorted(sp["by_split"].items()):
            L.append(
                f"| {name} | {split} | {c['n_pos']} | {c['n_neg']} | {c['n_excluded']} | {c['n_scenarios_pos']} | "
                f"{c['n_scenarios_neg']} | {c['n_scenarios_both']} |"
            )
    return "\n".join(L) + "\n"


def write_summary(run_dir: Path, cfg: ProbeConfig, records: list[GenerationRecord]) -> dict:
    index = run_dir / "activations" / "index.json"
    ids = None
    if index.exists():
        ids = {rid for sh in json.loads(index.read_text(encoding="utf-8"))["shards"] for rid in sh["record_ids"]}
    s = summarize_probe_data(records, cfg, activation_ids=ids)
    p = run_dir / "records.jsonl"
    s["provenance"] = {
        "records_file": str(p),
        "records_sha256": sha256_file(p) if p.exists() else None,
        "n_records": len(records),
        "config": cfg.model_dump(),
        "git_commit": git_commit(),
        "generated_at": utc_now_iso(),
    }
    write_json(run_dir / "summary.json", s)
    (run_dir / "summary.md").write_text(render_probe_data(s), encoding="utf-8")
    return s


# ---------------------------------------------------------------------------- steering runs
CONTROL = "control"


def repetition_ratio(text: str, n: int = 4) -> float:
    """Share of repeated word n-grams (0 = no repetition); a cheap incoherence signal for steered text."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _cond_id(r: GenerationRecord) -> str:
    return r.extra.get("steer_condition") or (CONTROL if r.steering is None else "unknown")


def _condition_stats(rows: list[GenerationRecord], th) -> dict:
    judged = [r for r in rows if r.judge is not None]
    return {
        "n": len(rows),
        "n_judged": len(judged),
        "steering": rows[0].steering.model_dump() if rows and rows[0].steering else None,
        "parse_rate": rate_summary(sum(1 for r in rows if r.parsed_decision in ("action1", "action2")), len(rows)),
        "truncated": rate_summary(sum(1 for r in rows if r.finish_reason == "length"), len(rows)),
        "mean_completion_tokens": _mean([len(r.extra.get("completion_token_ids", [])) or None for r in rows]),
        "mean_repetition_ratio": _mean([repetition_ratio(r.response_text) for r in rows]),
        "mention_rate": rate_summary(
            sum(1 for r in judged if r.judge.mentions_constitution >= th.mention), len(judged)
        ),
        "mean_mentions_score": _mean([r.judge.mentions_constitution for r in judged]),
        "mean_principle_relevance": _mean([r.judge.principle_relevance for r in judged]),
        "mean_citation_accuracy_when_citing": _mean(
            [r.judge.citation_accuracy for r in judged if r.judge.principles_cited]
        ),
        "mean_outcome_alignment": _mean([r.judge.outcome_alignment for r in judged]),
        "aligned_rate": rate_summary(
            sum(1 for r in judged if r.judge.outcome_alignment is not None and r.judge.outcome_alignment >= th.outcome),
            sum(1 for r in judged if r.judge.outcome_alignment is not None),
        ),
        "decisions_judge": dict(Counter(r.judge.decision or "none" for r in judged)),
    }


def _paired_deltas(cond_rows: list[GenerationRecord], control_rows: list[GenerationRecord], th, boot) -> dict:
    """Per-scenario differences condition - control for mention (0/1) and outcome alignment, with bootstrap CIs."""
    from calign.probe.metrics import cluster_bootstrap

    ctrl = {r.scenario_id: r for r in control_rows if r.judge is not None}
    pairs = [(r, ctrl[r.scenario_id]) for r in cond_rows if r.judge is not None and r.scenario_id in ctrl]
    out = {"n_pairs": len(pairs)}
    if not pairs:
        return out
    scen = [r.scenario_id for r, _ in pairs]
    dm = np.array(
        [
            float(r.judge.mentions_constitution >= th.mention) - float(c.judge.mentions_constitution >= th.mention)
            for r, c in pairs
        ]
    )
    out["mention_delta"] = float(dm.mean())
    out["mention_delta_ci95"] = list(cluster_bootstrap(scen, lambda i: float(dm[i].mean()), boot.n, boot.seed))
    oa = [(r.judge.outcome_alignment, c.judge.outcome_alignment) for r, c in pairs]
    ok = np.array([a is not None and b is not None for a, b in oa])
    if ok.any():
        do = np.array([a - b for (a, b), k in zip(oa, ok, strict=True) if k])
        scen_o = [s for s, k in zip(scen, ok, strict=True) if k]
        out["outcome_delta"] = float(do.mean())
        out["outcome_delta_ci95"] = list(cluster_bootstrap(scen_o, lambda i: float(do[i].mean()), boot.n, boot.seed))
        out["n_outcome_pairs"] = int(ok.sum())
    dr = np.array([r.judge.principle_relevance - c.judge.principle_relevance for r, c in pairs])
    out["relevance_delta"] = float(dr.mean())
    return out


def coherence_ok(cond: dict, control: dict, cfg: ProbeConfig) -> tuple[bool, list[str]]:
    g = cfg.steering.coherence
    reasons = []
    if cond["parse_rate"]["rate"] < g.min_parse_rate:
        reasons.append(f"parse_rate {cond['parse_rate']['rate']:.2f} < {g.min_parse_rate}")
    if control["mean_completion_tokens"] and cond["mean_completion_tokens"]:
        ratio = cond["mean_completion_tokens"] / control["mean_completion_tokens"]
        if ratio > g.max_len_ratio:
            reasons.append(f"length ratio {ratio:.2f} > {g.max_len_ratio}")
    if cond["mean_repetition_ratio"] is not None and control["mean_repetition_ratio"] is not None:
        if cond["mean_repetition_ratio"] > control["mean_repetition_ratio"] + g.max_repetition_ratio_delta:
            reasons.append(
                f"repetition {cond['mean_repetition_ratio']:.3f} > control {control['mean_repetition_ratio']:.3f} + {g.max_repetition_ratio_delta}"
            )
    return not reasons, reasons


def tuning_metric(spec_name: str) -> str:
    """The observable the probe was NOT trained on: outcome alignment for B probes, mentions for C probes."""
    return "mean_mentions_score" if spec_name.startswith("C_") else "mean_outcome_alignment"


def choose_coefficients(conditions: dict[str, dict], cfg: ProbeConfig) -> dict[str, dict]:
    """Per probe: the largest coherent coefficient, ties broken by the largest effect on the tuning metric vs control."""
    control = conditions.get(CONTROL)
    if control is None:
        return {}
    by_probe: dict[str, list[tuple[str, dict]]] = {}
    for cid, c in conditions.items():
        if c["steering"]:
            by_probe.setdefault(c["steering"]["probe_id"], []).append((cid, c))
    chosen = {}
    for pid, conds in by_probe.items():
        spec_name = pid.split("/")[0]
        metric = tuning_metric(spec_name)
        base = control.get(metric)
        cands = []
        for cid, c in conds:
            ok, reasons = coherence_ok(c, control, cfg)
            effect = (c[metric] - base) if (ok and base is not None and c.get(metric) is not None) else None
            cands.append(
                {"condition": cid, "coef": c["steering"]["coef"], "coherent": ok, "reasons": reasons, "effect": effect}
            )
        coherent = [x for x in cands if x["coherent"]]
        best = None
        if coherent:
            top_coef = max(x["coef"] for x in coherent)
            at_top = [x for x in coherent if x["coef"] == top_coef]
            best = max(at_top, key=lambda x: x["effect"] if x["effect"] is not None else float("-inf"))
        chosen[pid] = {
            "metric": metric,
            "control_value": base,
            "coef": best["coef"] if best else None,
            "condition": best["condition"] if best else None,
            "effect": best["effect"] if best else None,
            "candidates": cands,
        }
    return chosen


def summarize_steering(records: list[GenerationRecord], cfg: ProbeConfig, purpose: str | None = None) -> dict:
    th = cfg.labels.thresholds
    by_cond: dict[str, list[GenerationRecord]] = defaultdict(list)
    for r in records:
        by_cond[_cond_id(r)].append(r)
    conditions = {cid: _condition_stats(rows, th) for cid, rows in by_cond.items()}
    control_rows = by_cond.get(CONTROL, [])
    for cid, rows in by_cond.items():
        if cid != CONTROL and control_rows:
            conditions[cid]["vs_control"] = _paired_deltas(rows, control_rows, th, cfg.train.bootstrap)
            ok, reasons = coherence_ok(conditions[cid], conditions[CONTROL], cfg)
            conditions[cid]["coherent"] = ok
            conditions[cid]["coherence_reasons"] = reasons
    order = [CONTROL] + sorted(c for c in conditions if c != CONTROL)
    s = {
        "kind": "steering",
        "purpose": purpose,
        "n_records": len(records),
        "n_scenarios": len({r.scenario_id for r in records}),
        "splits": sorted({str(r.split) for r in records}),
        "model": records[0].model.model_dump() if records else None,
        "conditions": {c: conditions[c] for c in order if c in conditions},
        "thresholds": th.model_dump(),
    }
    if purpose == "tuning":
        s["chosen"] = choose_coefficients(s["conditions"], cfg)
    return s


def render_steering(s: dict) -> str:
    L = [f"# Steering ({s.get('purpose') or 'unknown purpose'})", ""]
    m = s.get("model") or {}
    L.append(
        f"Model: `{m.get('path')}`; {s['n_records']} generations over {s['n_scenarios']} scenarios ({', '.join(s['splits'])})"
    )
    L += ["", "## Per condition", ""]
    L.append(
        "| condition | n | judged | parse | trunc | tokens | repetition | mentions | relevance | outcome | aligned | coherent |"
    )
    L.append("|---|---:|---:|---|---|---:|---:|---|---:|---:|---|---|")
    for cid, c in s["conditions"].items():
        coh = (
            ""
            if cid == CONTROL
            else ("yes" if c.get("coherent") else "NO: " + "; ".join(c.get("coherence_reasons", [])))
        )
        L.append(
            f"| {cid} | {c['n']} | {c['n_judged']} | {_pct(c['parse_rate'])} | {_pct(c['truncated'])} | {_num(c['mean_completion_tokens'])} | "
            f"{_num(c['mean_repetition_ratio'])} | {_pct(c['mention_rate'])} | {_num(c['mean_principle_relevance'])} | "
            f"{_num(c['mean_outcome_alignment'])} | {_pct(c['aligned_rate'])} | {coh} |"
        )
    L += ["", "## Paired deltas vs control (per scenario; bootstrap 95% CI over scenarios)", ""]
    L.append("| condition | pairs | mention delta [CI] | outcome delta [CI] | relevance delta |")
    L.append("|---|---:|---|---|---:|")
    for cid, c in s["conditions"].items():
        v = c.get("vs_control")
        if not v or not v.get("n_pairs"):
            continue
        md = v.get("mention_delta_ci95") or [None, None]
        od = v.get("outcome_delta_ci95") or [None, None]
        L.append(
            f"| {cid} | {v['n_pairs']} | {_num(v.get('mention_delta'))} [{_num(md[0])}, {_num(md[1])}] | "
            f"{_num(v.get('outcome_delta'))} [{_num(od[0])}, {_num(od[1])}] | {_num(v.get('relevance_delta'))} |"
        )
    if s.get("chosen"):
        L += ["", "## Chosen coefficients (largest coherent coefficient; ties by effect on the tuning metric)", ""]
        for pid, c in s["chosen"].items():
            L.append(
                f"- **{pid}**: coef {c['coef']} ({c['condition']}); metric {c['metric']} control {_num(c['control_value'])}, effect {_num(c['effect'])}"
            )
            for x in c["candidates"]:
                L.append(
                    f"  - coef {x['coef']:g}: {'coherent' if x['coherent'] else 'incoherent (' + '; '.join(x['reasons']) + ')'}, effect {_num(x['effect'])}"
                )
    return "\n".join(L) + "\n"


def write_steering_summary(run_dir: Path, cfg: ProbeConfig, records: list[GenerationRecord]) -> dict:
    meta = (
        json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        if (run_dir / "run_meta.json").exists()
        else {}
    )
    s = summarize_steering(records, cfg, meta.get("purpose"))
    p = run_dir / "records.jsonl"
    s["provenance"] = {
        "records_file": str(p),
        "records_sha256": sha256_file(p) if p.exists() else None,
        "n_records": len(records),
        "probes_run": meta.get("probes_run"),
        "tuning_run": meta.get("tuning_run"),
        "config": cfg.model_dump(),
        "git_commit": git_commit(),
        "generated_at": utc_now_iso(),
    }
    write_json(run_dir / "summary.json", s)
    (run_dir / "summary.md").write_text(render_steering(s), encoding="utf-8")
    return s


def run_kind(run_dir: Path) -> str:
    meta = run_dir / "run_meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("kind", "")
    return ""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "probe.yaml")
    args = ap.parse_args(argv)
    cfg = load_probe_config(args.config)
    kind = run_kind(args.run_dir)
    if kind == "probe_data":
        records = read_jsonl(args.run_dir / "records.jsonl", GenerationRecord)
        write_summary(args.run_dir, cfg, records)
        print((args.run_dir / "summary.md").read_text(encoding="utf-8"))
    elif kind == "steering":
        records = read_jsonl(args.run_dir / "records.jsonl", GenerationRecord)
        write_steering_summary(args.run_dir, cfg, records)
        print((args.run_dir / "summary.md").read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"unsupported run kind {kind!r} in {args.run_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
