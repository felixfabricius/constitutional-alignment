"""MATH-500 (HuggingFaceH4/MATH-500, 500 problems) graded with math_verify (D10: the STEM-ability budget metric).

CLI:
    uv run python -m calign.evals.math500 sample --eval-config C0 [--dry-run] [--limit N]     # GPU
    uv run python -m calign.evals.math500 report --run-dir <run> [--reference <run>]          # local (re-grades)

Prompt: the problem followed by "Solve the problem. Put the final answer in \\boxed{}."; greedy, max 2048 tokens.
Grading: `math_verify.verify(parse("$" + answer + "$"), parse(response))` (math-verify's default extraction prefers
the last \\boxed{}). The report re-grades every record from its raw response and reference answer, so the numbers are
recomputable without the GPU. Summary: accuracy (Wilson CI), accuracy per level and subject, truncation rate, paired
delta against a reference run.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path

from calign.evals.common import (
    budget_system_prompt,
    component_cli,
    file_provenance,
    fmt_rate,
    generate_records,
    paired_item_delta,
)
from calign.evals.config import EvalConfig
from calign.inference.backend import ModelConfig
from calign.schemas import GenerationRecord, read_jsonl, write_json, write_jsonl
from calign.stats import rate_summary

LOGGER = logging.getLogger(__name__)

COMPONENT = "math500"
DATASET = "HuggingFaceH4/MATH-500"
MAX_TOKENS = 2048
RECORDS_FILE = "records.jsonl"
INSTRUCTION = "Solve the problem. Put the final answer in \\boxed{}."
# math_verify's timeouts use signals on Unix and subprocesses on Windows (which fail here); disable them on Windows.
_TIMEOUT = None if os.name == "nt" else 5


def format_prompt(problem: str) -> str:
    return f"{problem.strip()}\n\n{INSTRUCTION}"


def is_correct(response: str, answer: str) -> bool:
    from math_verify import parse, verify

    try:
        gold = parse(f"${answer}$", parsing_timeout=_TIMEOUT)
        pred = parse(response, parsing_timeout=_TIMEOUT)
        return bool(gold) and bool(pred) and bool(verify(gold, pred, timeout_seconds=_TIMEOUT))
    except Exception:  # noqa: BLE001 - a crashing parse is a wrong answer, never a crashed report
        return False


def load_dataset_rows(limit: int | None = None) -> tuple[list[dict], dict]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    from calign.paths import hf_token

    ds = load_dataset(DATASET, split="test", token=hf_token())
    rows = [dict(r) for r in ds]
    try:
        sha = HfApi().dataset_info(DATASET, token=hf_token()).sha
    except Exception:  # noqa: BLE001
        sha = None
    return (rows[:limit] if limit else rows), {"dataset": DATASET, "revision": sha, "n_rows": len(ds)}


def sample(
    backend,
    cfg: EvalConfig,
    model_cfg: ModelConfig,
    run_dir: Path,
    limit: int | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> list[GenerationRecord]:
    rows, info = load_dataset_rows(limit)
    write_json(run_dir / "dataset.json", info)
    items = [
        {
            "id": r["unique_id"],
            "user": format_prompt(r["problem"]),
            "extra": {"math": {"answer": r["answer"], "level": int(r["level"]), "subject": r["subject"]}},
        }
        for r in rows
    ]
    records = generate_records(
        backend,
        cfg,
        model_cfg,
        items,
        COMPONENT,
        max_tokens=MAX_TOKENS,
        seed=seed,
        system=budget_system_prompt(cfg.system_prompt_variant),
    )
    for r in records:
        r.extra["math"]["correct"] = is_correct(r.response_text, r.extra["math"]["answer"])
    write_jsonl(run_dir / RECORDS_FILE, records)
    if verbose:
        for r in records[:3]:
            print("=" * 100)
            print(r.messages[-1].content[:400])
            print("-" * 40, "answer", r.extra["math"]["answer"], "correct", r.extra["math"]["correct"])
            print(r.response_text[-800:])
    return records


def summarize(records: list[GenerationRecord], reference: list[GenerationRecord] | None = None) -> dict:
    correct = {r.scenario_id: float(is_correct(r.response_text, r.extra["math"]["answer"])) for r in records}
    n = len(records)
    by_level: dict[int, list[float]] = defaultdict(list)
    by_subject: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_level[r.extra["math"]["level"]].append(correct[r.scenario_id])
        by_subject[r.extra["math"]["subject"]].append(correct[r.scenario_id])
    stored = sum(r.extra["math"].get("correct") == bool(correct[r.scenario_id]) for r in records)
    out: dict = {
        "component": COMPONENT,
        "n": n,
        "accuracy": rate_summary(int(sum(correct.values())), n),
        "by_level": {str(k): rate_summary(int(sum(v)), len(v)) for k, v in sorted(by_level.items())},
        "by_subject": {k: rate_summary(int(sum(v)), len(v)) for k, v in sorted(by_subject.items())},
        "truncation_rate": rate_summary(sum(r.finish_reason == "length" for r in records), n),
        "mean_completion_tokens": sum(len(r.extra["completion_token_ids"]) for r in records) / n if n else None,
        "regrade_matches_stored": stored,
    }
    if reference is not None:
        ref = {r.scenario_id: float(is_correct(r.response_text, r.extra["math"]["answer"])) for r in reference}
        out["paired_vs_reference"] = {"accuracy": paired_item_delta(correct, ref)}
    return out


def write_report(run_dir: Path, reference: Path | None = None) -> dict:
    records = read_jsonl(run_dir / RECORDS_FILE, GenerationRecord)
    ref = read_jsonl(reference / RECORDS_FILE, GenerationRecord) if reference else None
    s = summarize(records, ref)
    s["run_dir"] = str(run_dir).replace("\\", "/")
    s["provenance"] = {
        "records": file_provenance(run_dir / RECORDS_FILE),
        "reference_records": file_provenance(reference / RECORDS_FILE) if reference else None,
    }
    write_json(run_dir / "summary.json", s)
    (run_dir / "summary.md").write_text(render_markdown(s), encoding="utf-8")
    return s


def render_markdown(s: dict) -> str:
    lines = [
        f"# MATH-500 ({s['n']} problems)",
        "",
        f"Accuracy {fmt_rate(s['accuracy'])}; truncated {fmt_rate(s['truncation_rate'])}; "
        f"mean completion tokens {s['mean_completion_tokens'] or 0:.0f}; re-grade matches stored grade on "
        f"{s['regrade_matches_stored']}/{s['n']}.",
        "",
        "| level | accuracy | n |",
        "|---|---|---:|",
    ]
    lines += [f"| {k} | {fmt_rate(v)} | {v['n']} |" for k, v in s["by_level"].items()]
    if p := s.get("paired_vs_reference"):
        d = p["accuracy"]
        lines += ["", f"Paired delta vs reference (n={d['n']}): {fmt_rate(d, 'delta')} points."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    component_cli(argv, __doc__, COMPONENT, sample, write_report, render_markdown)


if __name__ == "__main__":
    main()
