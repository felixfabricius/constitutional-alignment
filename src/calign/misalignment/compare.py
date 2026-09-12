"""Cross-run comparison of agentic-misalignment runs, recomputed from each run's raw `samples.jsonl`.

CLI:
    uv run python -m calign.misalignment.compare --runs base=outputs/misalignment/<run> e3=outputs/misalignment/<run> ...
        [--reference LABEL] [--out outputs/misalignment_compare/<name>]

Per run (label = the part before '=' or the directory name): model, backend, system-prompt variant, steering,
harmful rate (Wilson CI) overall / per scenario / per condition, constitution score (bootstrap CI), mention rate,
comprehensibility (coherence judge, repetition, truncation, U+FFFD). Every run is also compared with the reference
run (default: the first): difference in harmful rate (Newcombe interval) and in constitution / coherence score
(bootstrap). Writes summary.json (with a provenance block listing every input's samples sha) and summary.md.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from calign.config import git_commit, load_yaml, sha256_file
from calign.misalignment.report import SAMPLES_FILE, _mentioned, _scores, comprehensibility
from calign.paths import OUTPUTS_DIR
from calign.schemas import MisalignmentSample, read_json, read_jsonl, utc_now_iso, write_json
from calign.stats import diff_mean_summary, diff_rate_summary, mean_summary, rate_summary

LOGGER = logging.getLogger(__name__)


def parse_run_args(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for it in items:
        label, sep, path = it.partition("=")
        if not sep:
            label, path = Path(it).name, it
        if any(label == lab for lab, _ in out):
            raise SystemExit(f"duplicate run label {label!r}")
        out.append((label, Path(path)))
    return out


def run_info(run_dir: Path, samples: list[MisalignmentSample]) -> dict:
    meta = read_json(run_dir / "run_meta.json") if (run_dir / "run_meta.json").exists() else {}
    resolved = load_yaml(run_dir / "resolved_config.yaml") if (run_dir / "resolved_config.yaml").exists() else {}
    model = samples[0].model.model_dump() if samples else {}
    st = samples[0].steering.model_dump() if samples and samples[0].steering else meta.get("steering")
    return {
        "run_dir": str(run_dir),
        "model": model,
        "backend": meta.get("backend") or (resolved.get("model") or {}).get("backend"),
        "system_prompt_variants": sorted({s.system_prompt_variant for s in samples}),
        "steering": st,
        "git_commit": meta.get("git_commit"),
        "n_samples": len(samples),
    }


def _harm(rows: list[MisalignmentSample]) -> dict:
    classified = [r for r in rows if r.harmful is not None]
    return rate_summary(sum(1 for r in classified if r.harmful), len(classified))


def _mention(rows: list[MisalignmentSample]) -> dict:
    labelled = [r for r in rows if _mentioned(r) is not None]
    return rate_summary(sum(1 for r in labelled if _mentioned(r)), len(labelled))


def _coh(rows: list[MisalignmentSample]) -> list[float]:
    return [r.coherence_score for r in rows if r.coherence_score is not None]


def block(rows: list[MisalignmentSample]) -> dict:
    return {
        "harmful": _harm(rows),
        "constitution_score": mean_summary(_scores(rows)),
        "mentioned": _mention(rows),
        "comprehensibility": comprehensibility(rows),
    }


def diff_block(a: list[MisalignmentSample], b: list[MisalignmentSample]) -> dict:
    """Run a minus reference b: harmful rate (Newcombe), constitution score and coherence (bootstrap), mention rate."""
    ha, hb = _harm(a), _harm(b)
    ma, mb = _mention(a), _mention(b)
    return {
        "harmful": diff_rate_summary(ha["k"], ha["n"], hb["k"], hb["n"]),
        "constitution_score": diff_mean_summary(_scores(a), _scores(b)),
        "mentioned": diff_rate_summary(ma["k"], ma["n"], mb["k"], mb["n"]),
        "coherence_score": diff_mean_summary(_coh(a), _coh(b)),
    }


def compare(runs: list[tuple[str, Path]], reference: str | None = None) -> dict:
    loaded: dict[str, list[MisalignmentSample]] = {}
    info: dict[str, dict] = {}
    for label, d in runs:
        samples = read_jsonl(d / SAMPLES_FILE, MisalignmentSample)
        loaded[label] = samples
        info[label] = run_info(d, samples)
    reference = reference or runs[0][0]
    if reference not in loaded:
        raise SystemExit(f"reference {reference!r} is not one of the runs {list(loaded)}")
    scenarios = sorted({s.scenario for ss in loaded.values() for s in ss})
    conditions = sorted({s.condition_id for ss in loaded.values() for s in ss})

    def split(samples: list[MisalignmentSample]) -> tuple[dict, dict]:
        by_scen: dict[str, list[MisalignmentSample]] = defaultdict(list)
        by_cond: dict[str, list[MisalignmentSample]] = defaultdict(list)
        for s in samples:
            by_scen[s.scenario].append(s)
            by_cond[s.condition_id].append(s)
        return by_scen, by_cond

    per_run: dict[str, dict] = {}
    for label, samples in loaded.items():
        by_scen, by_cond = split(samples)
        per_run[label] = {
            "overall": block(samples),
            "by_scenario": {sc: block(by_scen[sc]) for sc in scenarios if by_scen[sc]},
            "by_condition": {c: block(by_cond[c]) for c in conditions if by_cond[c]},
        }
    ref = loaded[reference]
    ref_scen, _ = split(ref)
    vs_reference: dict[str, dict] = {}
    for label, samples in loaded.items():
        if label == reference:
            continue
        by_scen, _ = split(samples)
        vs_reference[label] = {
            "overall": diff_block(samples, ref),
            "by_scenario": {
                sc: diff_block(by_scen[sc], ref_scen[sc]) for sc in scenarios if by_scen[sc] and ref_scen[sc]
            },
        }
    return {
        "runs": info,
        "reference": reference,
        "scenarios": scenarios,
        "conditions": conditions,
        "per_run": per_run,
        "vs_reference": vs_reference,
    }


# ---------------------------------------------------------------------------- markdown
def _rate(r: dict) -> str:
    return f"{r['k']}/{r['n']} = {r['rate']:.1%} [{r['ci95_low']:.1%}, {r['ci95_high']:.1%}]" if r["n"] else "-"


def _rate_short(r: dict) -> str:
    return f"{r['rate']:.0%} ({r['k']}/{r['n']})" if r["n"] else "-"


def _mean(m: dict | None) -> str:
    return f"{m['mean']:.2f} [{m['ci95_low']:.2f}, {m['ci95_high']:.2f}]" if m and m["n"] else "-"


def _mean_short(m: dict | None) -> str:
    return f"{m['mean']:.2f}" if m and m["n"] else "-"


def _diff(d: dict, pct: bool) -> str:
    if d.get("diff") is None:
        return "-"
    if pct:
        return f"{d['diff']:+.1%} [{d['ci95_low']:+.1%}, {d['ci95_high']:+.1%}]"
    return f"{d['diff']:+.3f} [{d['ci95_low']:+.3f}, {d['ci95_high']:+.3f}]"


def _steer_text(st: dict | None) -> str:
    if not st:
        return "none"
    sign = "+" if st["sign"] > 0 else "-"
    return f"{st['probe_id']} {sign}{st['coef']:g} (L{st['layer']}, {st['positions']})"


def render_markdown(summary: dict) -> str:
    runs = summary["runs"]
    labels = list(runs)
    scen = summary["scenarios"]
    lines = ["# Agentic-misalignment runs compared", ""]
    lines.append(
        f"Reference run for differences: **{summary['reference']}**. "
        "All numbers are recomputed from each run's `samples.jsonl`."
    )
    lines += [
        "",
        "## Runs",
        "",
        "| label | model (stage) | backend | system prompt | steering | n |",
        "|---|---|---|---|---|---:|",
    ]
    for lab in labels:
        i = runs[lab]
        lines.append(
            f"| {lab} | {i['model'].get('name')} ({i['model'].get('stage')}) | {i.get('backend')} | "
            f"{', '.join(i['system_prompt_variants'])} | {_steer_text(i.get('steering'))} | {i['n_samples']} |"
        )
    lines += [
        "",
        "## Overall",
        "",
        "| run | harmful | constitution score | mentioned | coherence (judge) | repetition | truncated | U+FFFD any |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for lab in labels:
        o = summary["per_run"][lab]["overall"]
        k = o["comprehensibility"]
        lines.append(
            f"| {lab} | {_rate(o['harmful'])} | {_mean(o['constitution_score'])} | {_rate(o['mentioned'])} | "
            f"{_mean(k.get('coherence_score'))} | {k['repetition_ratio_mean']:.3f} | {k['truncated']['rate']:.1%} | "
            f"{k['replacement_chars']['rate']:.1%} |"
        )
    if summary["vs_reference"]:
        lines += [
            "",
            f"## Differences vs {summary['reference']} (run minus reference; rates: Newcombe 95% CI, scores: bootstrap 95% CI)",
            "",
            "| run | scope | harmful | constitution score | mentioned | coherence |",
            "|---|---|---|---|---|---|",
        ]
        for lab, v in summary["vs_reference"].items():
            rows = [("overall", v["overall"])] + list(v["by_scenario"].items())
            for scope, d in rows:
                lines.append(
                    f"| {lab} | {scope} | {_diff(d['harmful'], True)} | {_diff(d['constitution_score'], False)} | "
                    f"{_diff(d['mentioned'], True)} | {_diff(d['coherence_score'], False)} |"
                )
    for title, key, fmt in (
        ("Harmful rate by scenario", "harmful", _rate_short),
        ("Constitution score by scenario", "constitution_score", _mean_short),
    ):
        lines += ["", f"## {title}", "", "| run | " + " | ".join(scen) + " |", "|---|" + "---|" * len(scen)]
        for lab in labels:
            by = summary["per_run"][lab]["by_scenario"]
            lines.append(f"| {lab} | " + " | ".join(fmt(by.get(sc, {}).get(key, {"n": 0})) for sc in scen) + " |")
    for title, key, fmt in (
        ("Harmful rate by condition", "harmful", _rate_short),
        ("Constitution score by condition", "constitution_score", _mean_short),
        ("Mention rate by condition", "mentioned", _rate_short),
    ):
        lines += ["", f"## {title}", "", "| condition | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for c in summary["conditions"]:
            cells = [fmt(summary["per_run"][lab]["by_condition"].get(c, {}).get(key, {"n": 0})) for lab in labels]
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Comprehensibility by run",
        "",
        "| run | coherence (judge) | issues | repetition | U+FFFD any | empty | scratchpad | truncated | tool format | mean tokens |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lab in labels:
        k = summary["per_run"][lab]["overall"]["comprehensibility"]
        issues = ", ".join(f"{a} {b}" for a, b in (k.get("coherence_issues") or {}).items()) or "-"
        lines.append(
            f"| {lab} | {_mean(k.get('coherence_score'))} | {issues} | {k['repetition_ratio_mean']:.3f} | "
            f"{k['replacement_chars']['rate']:.1%} | {k['empty']['rate']:.1%} | {k['scratchpad']['rate']:.1%} | "
            f"{k['truncated']['rate']:.1%} | {k['used_tool_format']['rate']:.1%} | {k['mean_completion_tokens']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def write_comparison(out_dir: Path, runs: list[tuple[str, Path]], reference: str | None = None) -> dict:
    summary = compare(runs, reference)
    summary["provenance"] = {
        "inputs": {
            lab: {"samples_file": str(d / SAMPLES_FILE), "samples_sha256": sha256_file(d / SAMPLES_FILE)}
            for lab, d in runs
        },
        "git_commit": git_commit(),
        "generated_at": utc_now_iso(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="LABEL=DIR or DIR (label = directory name)")
    ap.add_argument("--reference", default=None, help="label the differences are taken against (default: first)")
    ap.add_argument("--out", type=Path, default=None, help="default: outputs/misalignment_compare/<timestamp>")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = parse_run_args(args.runs)
    out = args.out or OUTPUTS_DIR / "misalignment_compare" / utc_now_iso().replace(":", "").replace("-", "")[:15]
    write_comparison(out, runs, args.reference)
    print((out / "summary.md").read_text(encoding="utf-8"))
    LOGGER.info("wrote %s", out)


if __name__ == "__main__":
    main()
