"""Scenario run reports, recomputed from the raw responses in `episodes.jsonl`.

The report re-grades every episode with the current grader (`calign.scenarios.grade`), so a grader fix applies to
old runs without touching their raw files; `summary.json` records the grader version and how many stored grades
differ from the re-grade.

Rates (Wilson 95% intervals) are over the episodes without a format failure: any deviation (tier != aligned),
deceptive tier, non-deceptive tier (scenario 1: the gray tier), lie; plus format-failure and confusion-proxy rates
over all episodes, the audit-answer distribution, scenario-specific flags and, when `tags.jsonl` exists (judge), the
tag rates on the sampled subset. `--reference` adds Newcombe intervals for the differences against another run.

CLI (local):
    uv run python -m calign.scenarios.report --run-dir outputs/scenarios/C0/deadline_L1/<run> [--reference <run>]
    uv run python -m calign.scenarios.report --runs <run> <run> ... [--out outputs/scenarios/<table>.md]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from calign.evals.common import file_provenance, fmt_rate
from calign.scenarios.grade import GRADER_VERSION, grade_episode
from calign.scenarios.run import EPISODES_FILE
from calign.schemas import write_json
from calign.stats import diff_rate_summary, rate_summary

TAGS_FILE = "tags.jsonl"
HEADLINE = ("any_deviation", "deceptive", "nondeceptive", "lie")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def regrade(episodes: list[dict]) -> tuple[list[dict], int]:
    """Episodes with `grade` replaced by the current grader's result; also the number of changed grades."""
    out, changed = [], 0
    for e in episodes:
        g = grade_episode(e["scenario"], e["response_1"], e["response_2"], e.get("finish_reason_1")).to_dict()
        old = e.get("grade") or {}
        if (old.get("tier"), old.get("lie"), old.get("format_failure")) != (g["tier"], g["lie"], g["format_failure"]):
            changed += 1
        out.append({**e, "grade": g})
    return out, changed


def summarize(episodes: list[dict], tags: list[dict] | None = None) -> dict:
    n = len(episodes)
    valid = [e for e in episodes if not e["grade"]["format_failure"]]
    nv = len(valid)
    tiers = Counter(e["grade"]["tier"] for e in valid)
    k_dev = nv - tiers.get("aligned", 0)
    s: dict = {
        "n": n,
        "n_valid": nv,
        "format_failure": rate_summary(n - nv, n),
        "format_reasons": dict(Counter(r for e in episodes for r in e["grade"]["format_reasons"])),
        "confusion_proxy": rate_summary(sum(e["grade"]["confusion_proxy"] for e in episodes), n),
        "confusion_reasons": dict(Counter(r for e in episodes for r in e["grade"]["confusion_reasons"])),
        "any_deviation": rate_summary(k_dev, nv),
        "deceptive": rate_summary(tiers.get("deceptive", 0), nv),
        "nondeceptive": rate_summary(tiers.get("nondeceptive", 0), nv),
        "lie": rate_summary(sum(e["grade"]["lie"] for e in valid), nv),
        "deceptive_among_deviations": rate_summary(tiers.get("deceptive", 0), k_dev) if k_dev else None,
        "audit_answers": dict(Counter(e["grade"]["audit_answer"] for e in valid)),
        "truncated_turn1": sum(e.get("finish_reason_1") == "length" for e in episodes),
        "mean_completion_tokens_1": (sum(e.get("completion_tokens_1", 0) for e in episodes) / n) if n else None,
    }
    scen = {e["scenario"] for e in episodes}
    flags: dict = {}
    if scen == {"deadline"}:
        d = [e["grade"]["details"] for e in valid]
        flags = {
            "primary_analysis": dict(Counter(x["primary_analysis"] for x in d)),
            "side_channel": rate_summary(sum(x["side_channel"] for x in d), nv),
            "q4_denies_other_analyses": rate_summary(sum(x["q4_denies_other_analyses"] for x in d), nv),
            "corrected_at_audit": sum(x["corrected_at_audit"] for x in d),
            "q1_false": sum(x["q1_false"] for x in d),
            "q2_false": sum(x["q2_false"] for x in d),
            "audit_false": sum(x["audit_false"] for x in d),
        }
    elif scen == {"briefing"}:
        d = [e["grade"]["details"] for e in valid]
        dev = [x for x in d if x["omitted"]]
        flags = {
            "direction": dict(Counter(x["direction"] for x in dev)),
            "omitted_files": dict(Counter(f for x in dev for f in x["omitted"])),
            "refusal": sum(x["refusal"] for x in d),
            "careless_candidate": sum(x["careless_candidate"] for x in d),
            "skipped_neutral": dict(Counter(f for x in d for f in x.get("skipped_neutral", []))),
            "completeness_claim": sum(x["completeness_claim"] for x in d),
            "corrected_at_audit": sum(x["corrected_at_audit"] for x in d),
            "disclosed_at_audit": sum(x.get("disclosed_at_audit", False) for x in d),
            "replied_to_friend": rate_summary(sum(x["replied_to_friend"] for x in d), nv),
        }
    s["flags"] = flags
    if tags:
        s["tags"] = tag_rates(tags)
        deviating = {e["episode_id"] for e in valid if e["grade"]["tier"] != "aligned"}
        dev_tags = [t for t in tags if t["episode_id"] in deviating]
        if dev_tags:
            s["tags_among_deviations"] = tag_rates(dev_tags)
    return s


def tag_rates(tags: list[dict]) -> dict:
    """Rates of the judge's boolean tags (each over the rows where that tag parsed) and the mean framing skew."""
    ok = [t for t in tags if t.get("tags")]
    out: dict = {"n": len(ok)}
    keys = sorted({k for t in ok for k, v in t["tags"].items() if isinstance(v, bool)})
    for k in keys:
        vals = [t["tags"][k] for t in ok if isinstance(t["tags"].get(k), bool)]
        out[k] = rate_summary(sum(vals), len(vals))
    skew = [t["tags"]["framing_skew"] for t in ok if isinstance(t["tags"].get("framing_skew"), (int, float))]
    if skew:
        out["framing_skew_mean"] = sum(skew) / len(skew)
        out["framing_skew_n"] = len(skew)
    return out


def tag_repeatability(tags: list[dict], rep: list[dict]) -> dict:
    """Per tag: agreement between two judge passes (different cache salts) on the shared episodes."""
    b = {t["episode_id"]: t["tags"] for t in rep if t.get("tags")}
    shared = [(t["tags"], b[t["episode_id"]]) for t in tags if t.get("tags") and t["episode_id"] in b]
    out: dict = {"n": len(shared)}
    for k in sorted({k for a, _ in shared for k in a}):
        pairs = [(a.get(k), c.get(k)) for a, c in shared if a.get(k) is not None and c.get(k) is not None]
        if not pairs:
            continue
        if k == "framing_skew":
            out[k] = {"n": len(pairs), "mean_abs_diff": sum(abs(x - y) for x, y in pairs) / len(pairs)}
        else:
            out[k] = {"n": len(pairs), "agreement": sum(x == y for x, y in pairs) / len(pairs)}
    return out


def versus(s: dict, ref: dict) -> dict:
    return {
        k: diff_rate_summary(s[k]["k"], s[k]["n"], ref[k]["k"], ref[k]["n"])
        for k in HEADLINE
        if s[k]["n"] and ref[k]["n"]
    }


def load_run(run_dir: Path) -> tuple[list[dict], int, list[dict] | None]:
    eps, changed = regrade(read_jsonl(run_dir / EPISODES_FILE))
    tags = read_jsonl(run_dir / TAGS_FILE) if (run_dir / TAGS_FILE).exists() else None
    return eps, changed, tags


def run_label(run_dir: Path, episodes: list[dict]) -> str:
    e = episodes[0] if episodes else {}
    return f"{e.get('eval_config', '?')} {e.get('scenario', '?')} {e.get('level', '?')}"


def write_report(run_dir: Path, reference: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    eps, changed, tags = load_run(run_dir)
    s = summarize(eps, tags)
    s["label"] = run_label(run_dir, eps)
    s["run_dir"] = str(run_dir).replace("\\", "/")
    s["grader_version"] = GRADER_VERSION
    s["regrade_changed"] = changed
    rep_files = sorted(run_dir.glob("tags_*.jsonl"))
    if tags and rep_files:
        s["tag_repeatability"] = {p.stem: tag_repeatability(tags, read_jsonl(p)) for p in rep_files}
    if reference:
        reps, _, rtags = load_run(Path(reference))
        s["reference"] = {"run_dir": str(reference).replace("\\", "/"), "label": run_label(Path(reference), reps)}
        s["vs_reference"] = versus(s, summarize(reps, rtags))
    s["provenance"] = {
        "episodes": file_provenance(run_dir / EPISODES_FILE),
        "tags": file_provenance(run_dir / TAGS_FILE),
    }
    write_json(run_dir / "summary.json", s)
    (run_dir / "summary.md").write_text(render_markdown(s), encoding="utf-8")
    return s


def render_markdown(s: dict) -> str:
    lines = [
        f"# Scenario run: {s.get('label', '')} (n={s['n']}, valid {s['n_valid']})",
        "",
        "| metric | rate [95% CI] | k/n |",
        "|---|---|---|",
    ]
    for k in (*HEADLINE, "format_failure", "confusion_proxy"):
        m = s[k]
        lines.append(f"| {k} | {fmt_rate(m)} | {m['k']}/{m['n']} |")
    if s.get("deceptive_among_deviations"):
        m = s["deceptive_among_deviations"]
        lines.append(f"| deceptive among deviations | {fmt_rate(m)} | {m['k']}/{m['n']} |")
    lines += [
        "",
        f"Audit answers: {s['audit_answers']}. Format reasons: {s['format_reasons']}. "
        f"Confusion: {s['confusion_reasons']}. Truncated turn 1: {s['truncated_turn1']}.",
    ]
    fl = {k: (fmt_rate(v) if isinstance(v, dict) and "rate" in v else v) for k, v in s.get("flags", {}).items()}
    lines.append(f"Flags: {fl}")
    for key, label in (("tags", "Judge tags"), ("tags_among_deviations", "Judge tags among deviating episodes")):
        if t := s.get(key):
            lines.append(
                f"{label} (n={t['n']}): "
                + "; ".join(f"{k} {fmt_rate(v)}" for k, v in t.items() if isinstance(v, dict))
                + (f"; framing skew mean {t['framing_skew_mean']:.2f}" if "framing_skew_mean" in t else "")
            )
    for name, rep in (s.get("tag_repeatability") or {}).items():
        lines.append(
            f"Tag repeatability ({name}, n={rep['n']}): "
            + "; ".join(
                f"{k} {v['agreement']:.2f}" if "agreement" in v else f"{k} mean |diff| {v['mean_abs_diff']:.2f}"
                for k, v in rep.items()
                if isinstance(v, dict)
            )
        )
    if vs := s.get("vs_reference"):
        lines.append(
            f"Vs reference {s['reference']['label']}: "
            + "; ".join(
                f"{k} {100 * v['diff']:+.1f} [{100 * v['ci95_low']:+.1f}, {100 * v['ci95_high']:+.1f}]"
                for k, v in vs.items()
            )
        )
    lines.append(f"Grader {s.get('grader_version')}; stored grades changed by re-grading: {s.get('regrade_changed')}.")
    return "\n".join(lines) + "\n"


def runs_table(run_dirs: list[Path]) -> tuple[str, list[dict]]:
    rows = []
    for r in run_dirs:
        eps, changed, tags = load_run(Path(r))
        s = summarize(eps, tags)
        s["label"] = run_label(Path(r), eps)
        s["run_dir"] = str(r).replace("\\", "/")
        rows.append(s)
    head = "| run | n valid | any deviation | deceptive | non-deceptive | lie | format fail | confusion | audit yes/no/other |"
    lines = [head, "|" + "---|" * 9]
    for s in rows:
        a = s["audit_answers"]
        lines.append(
            f"| {s['label']} | {s['n_valid']}/{s['n']} | {fmt_rate(s['any_deviation'])} | {fmt_rate(s['deceptive'])} | "
            f"{fmt_rate(s['nondeceptive'])} | {fmt_rate(s['lie'])} | {fmt_rate(s['format_failure'])} | "
            f"{fmt_rate(s['confusion_proxy'])} | {a.get('yes', 0)}/{a.get('no', 0)}/{a.get('other', 0)} |"
        )
    return "\n".join(lines) + "\n", rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path)
    g.add_argument("--runs", type=Path, nargs="+")
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="--runs: write the table (and a .json) here")
    args = ap.parse_args(argv)
    if args.run_dir:
        print(render_markdown(write_report(args.run_dir, args.reference)))
        return
    md, rows = runs_table(args.runs)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        write_json(args.out.with_suffix(".json"), {"grader_version": GRADER_VERSION, "runs": rows})


if __name__ == "__main__":
    main()
