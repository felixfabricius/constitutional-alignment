"""IFEval (google/IFEval, 541 prompts) with the official checker, on our own vLLM runner (D11).

CLI:
    uv run python -m calign.evals.ifeval sample --eval-config C0 [--dry-run] [--limit N]     # GPU
    uv run python -m calign.evals.ifeval report --run-dir <run> [--reference <run>]          # local

Our runner (not lm_eval's CLI) so a configuration's system prompt can be folded into the Gemma chat turn; generation
as in lm_eval's task: greedy, max 1280 new tokens, no stop strings besides end of turn. Grading imports
`lm_eval.tasks.ifeval.utils.test_instruction_following_{strict,loose}` (lm-eval 0.4.13, `eval` dependency group).
records.jsonl holds one GenerationRecord per prompt (source "ifeval", scenario_id = IFEval key) with
`extra.ifeval` = {instruction_id_list, kwargs, strict, loose (per-instruction flags), prompt_strict, prompt_loose};
grading is deterministic from those fields, so `regrade_record` recomputes the flags from the raw record.
Summary: prompt-level and instruction-level strict/loose accuracy (Wilson CIs; instruction level resamples prompts),
paired delta of prompt-level strict accuracy against a reference run.
"""

from __future__ import annotations

import logging
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
from calign.stats import cluster_bootstrap_mean, rate_summary

LOGGER = logging.getLogger(__name__)

COMPONENT = "ifeval"
DATASET = "google/IFEval"
MAX_TOKENS = 1280
RECORDS_FILE = "records.jsonl"


def load_dataset_rows(limit: int | None = None) -> tuple[list[dict], dict]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    from calign.paths import hf_token

    ds = load_dataset(DATASET, split="train", token=hf_token())
    rows = [dict(r) for r in ds]
    try:
        sha = HfApi().dataset_info(DATASET, token=hf_token()).sha
    except Exception:  # noqa: BLE001
        sha = None
    rows = sorted(rows, key=lambda r: int(r["key"]))
    return (rows[:limit] if limit else rows), {"dataset": DATASET, "revision": sha, "n_rows": len(ds)}


def grade(instruction_id_list: list[str], kwargs: list[dict], prompt: str, response: str) -> dict:
    """Official strict and loose checks for one response."""
    from lm_eval.tasks.ifeval.utils import (
        InputExample,
        test_instruction_following_loose,
        test_instruction_following_strict,
    )

    inp = InputExample(key=0, instruction_id_list=instruction_id_list, prompt=prompt, kwargs=kwargs)
    s = test_instruction_following_strict(inp, response)
    lo = test_instruction_following_loose(inp, response)
    return {
        "strict": list(s.follow_instruction_list),
        "loose": list(lo.follow_instruction_list),
        "prompt_strict": bool(s.follow_all_instructions),
        "prompt_loose": bool(lo.follow_all_instructions),
    }


def regrade_record(r: GenerationRecord) -> dict:
    e = r.extra["ifeval"]
    return grade(e["instruction_id_list"], e["kwargs"], r.messages[-1].content, r.response_text)


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
    items = [{"id": str(r["key"]), "user": r["prompt"]} for r in rows]
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
    for r, row in zip(records, rows, strict=True):
        kw = [{k: v for k, v in d.items() if v is not None} for d in row["kwargs"]]
        r.extra["ifeval"] = {
            "instruction_id_list": list(row["instruction_id_list"]),
            "kwargs": kw,
            **grade(list(row["instruction_id_list"]), kw, row["prompt"], r.response_text),
        }
    write_jsonl(run_dir / RECORDS_FILE, records)
    if verbose:
        for r in records[:3]:
            print("=" * 100)
            print(r.messages[-1].content[:400])
            print("-" * 40, r.extra["ifeval"]["instruction_id_list"], r.extra["ifeval"]["strict"])
            print(r.response_text[:1200])
    return records


def prompt_scores(records: list[GenerationRecord], key: str = "prompt_strict") -> dict[str, float]:
    return {r.scenario_id: float(r.extra["ifeval"][key]) for r in records}


def summarize(records: list[GenerationRecord], reference: list[GenerationRecord] | None = None) -> dict:
    n = len(records)
    out: dict = {"component": COMPONENT, "n_prompts": n}
    for level in ("strict", "loose"):
        k = sum(r.extra["ifeval"][f"prompt_{level}"] for r in records)
        out[f"prompt_level_{level}"] = rate_summary(k, n)
        flags = [float(f) for r in records for f in r.extra["ifeval"][level]]
        clusters = [r.scenario_id for r in records for _ in r.extra["ifeval"][level]]
        s = cluster_bootstrap_mean(flags, clusters)
        out[f"inst_level_{level}"] = {
            "rate": s["mean"],
            "n": s["n"],
            "ci95_low": s["ci95_low"],
            "ci95_high": s["ci95_high"],
        }
    out["truncation_rate"] = rate_summary(sum(r.finish_reason == "length" for r in records), n)
    out["mean_completion_tokens"] = sum(len(r.extra["completion_token_ids"]) for r in records) / n if n else None
    if reference is not None:
        out["paired_vs_reference"] = {
            "prompt_level_strict": paired_item_delta(prompt_scores(records), prompt_scores(reference)),
            "prompt_level_loose": paired_item_delta(
                prompt_scores(records, "prompt_loose"), prompt_scores(reference, "prompt_loose")
            ),
        }
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
        f"# IFEval ({s['n_prompts']} prompts)",
        "",
        "| metric | value [95% CI] |",
        "|---|---|",
    ]
    for k in ("prompt_level_strict", "prompt_level_loose", "inst_level_strict", "inst_level_loose", "truncation_rate"):
        lines.append(f"| {k} | {fmt_rate(s[k])} |")
    lines.append(f"| mean completion tokens | {s['mean_completion_tokens'] or 0:.0f} |")
    if p := s.get("paired_vs_reference"):
        for k, d in p.items():
            lines.append(f"| delta {k} vs reference (n={d['n']}) | {fmt_rate(d, 'delta')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    component_cli(argv, __doc__, COMPONENT, sample, write_report, render_markdown)


if __name__ == "__main__":
    main()
