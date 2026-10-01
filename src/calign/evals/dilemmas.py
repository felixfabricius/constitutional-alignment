"""Phase 3 core-suite component `hardsets`: eval-1-hard and eval-2-hard (generated dilemmas, chunk 6).

CLI:
    # GPU (vLLM): k=4 at T=0.7 on data/dilemmas/final/{eval1_hard,eval2_hard}.jsonl, the MoralChoice eval prompt
    uv run python -m calign.evals.dilemmas sample --eval-config C2@e3 [--dry-run] [--limit N]
    # local: alignment per set with item-cluster bootstrap CIs, paired deltas vs a reference run (C0)
    uv run python -m calign.evals.dilemmas report --run-dir <run> [--reference <run>]

Same sampling and metrics as `calign.evals.moralchoice` (letter randomisation, `none`/C1 system-prompt variant from
the configuration, alignment = parsed answers matching the item's verdict); the truth is each item's own verdict
from the set file (the independent verdict judge, which agreed with the generator's intent by construction). Both
sets are base-filtered (base disagrees in >= 2 of 4 samples), so C0 scores are low by selection; compare
configurations by paired deltas against C0, not by levels. When the set files are absent (before chunk 6's filter
step) the component is skipped by the suite.
"""

from __future__ import annotations

import logging
from pathlib import Path

from calign.evals import moralchoice
from calign.evals.common import component_cli, file_provenance
from calign.evals.config import EvalConfig
from calign.inference.backend import ModelConfig
from calign.paths import DATA_DIR
from calign.schemas import Dilemma, GenerationRecord, read_jsonl, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)

COMPONENT = "hardsets"
SETS = ("eval1_hard", "eval2_hard")
FINAL_DIR = DATA_DIR / "dilemmas" / "final"
RECORDS_FILE = "records.jsonl"
K = 4
TEMPERATURE = 0.7


def set_path(name: str) -> Path:
    return FINAL_DIR / f"{name}.jsonl"


def available() -> bool:
    return all(set_path(s).exists() for s in SETS)


def load_items() -> list[Dilemma]:
    items: list[Dilemma] = []
    for s in SETS:
        if set_path(s).exists():
            items += read_jsonl(set_path(s), Dilemma)
    return items


def sample(
    backend,
    cfg: EvalConfig,
    model_cfg: ModelConfig,
    run_dir: Path,
    limit: int | None = None,
    seed: int = moralchoice.DEFAULT_SEED,
    verbose: bool = False,
) -> list[GenerationRecord]:
    items = load_items()
    if limit:
        items = items[:limit]
    params = moralchoice.SampleParams(splits=list(SETS), k=K, temperature=TEMPERATURE, seed=seed, limit=limit)
    write_json(
        run_dir / "items.json",
        {"sets": {s: file_provenance(set_path(s)) for s in SETS}, "n_items": len(items), "params": params.model_dump()},
    )
    pairs = [(d.to_scenario(d.meta["set"]), d.meta["set"]) for d in items]
    records = moralchoice.run_sample(backend, cfg, model_cfg, pairs, params)
    write_jsonl(run_dir / RECORDS_FILE, records)
    if verbose:
        for r in records[:4]:
            print("=" * 100)
            print(f"{r.scenario_id} split={r.split} order={r.extra['letter_order']} decision={r.parsed_decision}")
            print(r.response_text[:1500])
    return records


def summarize_run(run_dir: Path, reference_dir: Path | None = None, n_boot: int = 2000) -> dict:
    records = read_jsonl(run_dir / RECORDS_FILE, GenerationRecord)
    items = {d.item_id: d for d in load_items()}
    verdicts = {i: d.verdict for i, d in items.items() if d.verdict is not None}
    groups = {s: [r for r in records if r.split == s] for s in SETS}
    groups["all"] = records
    splits = {g: moralchoice.group_metrics(rs, verdicts, None, n_boot) for g, rs in groups.items() if rs}  # type: ignore[arg-type]
    for g, rs in groups.items():
        if not rs or g == "all":
            continue
        by_p: dict[str, list[GenerationRecord]] = {}
        for r in rs:
            p = items[r.scenario_id].principle_focus if r.scenario_id in items else None
            by_p.setdefault(f"P{p}", []).append(r)
        splits[g]["by_principle"] = {
            p: moralchoice.group_metrics(x, verdicts, None, n_boot)["alignment"]  # type: ignore[arg-type]
            for p, x in sorted(by_p.items())
        }
    summary: dict = {
        "component": COMPONENT,
        "run_dir": str(run_dir).replace("\\", "/"),
        "eval_config": records[0].extra.get("eval_config") if records else None,
        "splits": splits,
    }
    if reference_dir is not None:
        ref = read_jsonl(reference_dir / RECORDS_FILE, GenerationRecord)
        summary["reference"] = {
            "run_dir": str(reference_dir).replace("\\", "/"),
            "paired": {
                g: {"all_items": moralchoice.paired_delta(rs, ref, verdicts, n_boot=n_boot), "hard_items": None}  # type: ignore[arg-type]
                for g, rs in groups.items()
                if rs
            },
        }
    summary["provenance"] = {
        "records": file_provenance(run_dir / RECORDS_FILE),
        "sets": {s: file_provenance(set_path(s)) for s in SETS},
        "n_boot": n_boot,
    }
    return summary


def render_markdown(summary: dict) -> str:
    md = moralchoice.render_markdown(summary).replace("# MoralChoice:", "# Hard sets (generated dilemmas):", 1)
    lines = ["", "Alignment by principle focus:", ""]
    for g, m in summary["splits"].items():
        for p, a in (m.get("by_principle") or {}).items():
            lines.append(f"- {g} {p}: {moralchoice._fmt(a)} ")
    return md + "\n".join(lines) + "\n"


def write_report(run_dir: Path, reference: Path | None = None, n_boot: int = 2000) -> dict:
    summary = summarize_run(run_dir, reference, n_boot)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    component_cli(argv, __doc__, COMPONENT, sample, write_report, render_markdown, moralchoice.DEFAULT_SEED)


if __name__ == "__main__":
    main()
