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
    records: list[GenerationRecord], cfg: ProbeConfig, verdicts: dict[str, ConstitutionVerdict] | None = None
) -> dict:
    verdicts = load_verdicts() if verdicts is None else verdicts
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
            "n_with_activations": sum(1 for r in rows if r.activations is not None),
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
    s = summarize_probe_data(records, cfg)
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
    else:
        raise SystemExit(f"unsupported run kind {kind!r} in {args.run_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
