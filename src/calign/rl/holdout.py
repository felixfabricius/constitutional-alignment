"""RL hold-out evaluation (chunk 7 deliverable 9, Felix 2026-10-03): does RL improve held-out dilemmas or only the
trained prompts?

Data: `data.rl_holdout` (chunk 6: 29 generated P1-P5 items from 23 families never trained on, selected like RL-train,
0 < passes < 8 on the RL start). Rows are built exactly like the RL-train dilemma rows (`rl.prompts.dilemma_rows`, both
letter orders, `task_type="dilemma"`); no anchors, no math, no shuffle. Item and family ids must be disjoint from
RL-train (checked whenever the rows are built) and from the reserve (unit test on the committed files).

In the trainer (`rl.train_grpo`, TRL `eval_on_start` + `eval_steps`): at step 0 and every `grpo.eval_steps` steps the
current policy answers every row `grpo.num_generations` times at the training temperature (same rollout path: the
vLLM server with the policy's synced weights), and the answers are scored by a second `RewardSuite` with the training
reward settings (same functions, same weights; C4 includes `r_cite` through the local judge, sharing its cache). Only
generation and scoring run; TRL's own GRPO evaluation is bypassed because it also runs three full-batch forward
passes of the 27B (policy, old and reference log-probs plus the loss), which either runs out of memory or forces one
vLLM call per prompt.

Run-dir files: `holdout_dataset.jsonl` + `holdout_manifest.json` (rows and provenance), `holdout_rollouts.jsonl`
(every answer with its reward components; `step` = optimizer steps taken), `holdout.jsonl` (one summary line per
evaluation, recomputable from the rollouts with the `summarize` CLI). The baseline is the step-0 evaluation, never the
selection counts in `meta.filter.rl_start` (items were selected on that run, so they regress to the mean).

CLI:
    # recompute holdout.jsonl from holdout_rollouts.jsonl
    uv run python -m calign.rl.holdout summarize --run-dir outputs/rl/C4_pilot
    # fallback (in-loop evaluation unavailable): score answers sampled offline from a checkpoint with
    # `calign.dilemmas.filter sample --file data/dilemmas/final/rl_holdout.jsonl --k 16 --max-tokens 1024 ...`
    uv run python -m calign.rl.holdout score-offline --config configs/rl/C4.yaml --run-dir outputs/rl/C4_pilot \
        --step 20 --records outputs/rl/C4_pilot/holdout/s20/records.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

from calign.constitution import load_constitution
from calign.evals.common import file_provenance
from calign.rl.config import RewardSettings, RLConfig, RLDataSettings, load_rl_config
from calign.rl.prompts import RLRow, dilemma_rows
from calign.rl.rewards import RewardSuite
from calign.schemas import Dilemma, GenerationRecord, read_jsonl, write_json, write_jsonl

HOLDOUT_DATASET_FILE = "holdout_dataset.jsonl"
HOLDOUT_MANIFEST_FILE = "holdout_manifest.json"
HOLDOUT_ROLLOUTS_FILE = "holdout_rollouts.jsonl"
HOLDOUT_FILE = "holdout.jsonl"

# (completions, completion token ids or None) for prompts that repeat each row's prompt k times in a row
Generate = Callable[[list[str]], tuple[list[str], list[list[int]] | None]]


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------


def check_disjoint(train: Sequence[Dilemma], holdout: Sequence[Dilemma], reserve: Sequence[Dilemma] = ()) -> None:
    """Raise if a hold-out item or family appears in RL-train or in the reserve (the D20 re-filter source)."""
    h_items = {d.item_id for d in holdout}
    h_fams = {d.family_id for d in holdout}
    for name, other in (("rl_train", train), ("rl_reserve", reserve)):
        items = sorted(h_items & {d.item_id for d in other})
        fams = sorted(h_fams & {d.family_id for d in other})
        if items or fams:
            raise ValueError(f"hold-out overlaps {name}: items {items[:5]} families {fams[:5]}")


def holdout_rows(items: Sequence[Dilemma]) -> list[RLRow]:
    """Every item in both letter orders, built like the RL-train dilemma rows; in file order (no shuffle)."""
    anchors = [d.item_id for d in items if d.variant_kind == "anchor"]
    if anchors:
        raise ValueError(f"the RL hold-out holds generated items only, found anchors {anchors[:5]}")
    c = load_constitution()
    return [row for d in items for row in dilemma_rows(d, c)]


def item_meta(items: Sequence[Dilemma]) -> dict[str, dict]:
    return {
        d.item_id: {"family_id": d.family_id, "principle_focus": d.principle_focus, "variant_kind": d.variant_kind}
        for d in items
    }


def load_holdout(cfg: RLConfig, limit: int | None = None) -> tuple[list[Dilemma], list[RLRow], dict]:
    """(items, rows, manifest) of the config's hold-out; checks it against the config's RL-train file."""
    s = cfg.data
    if s.rl_holdout is None:
        raise ValueError(f"{cfg.run_name}: data.rl_holdout is not set")
    items = read_jsonl(s.rl_holdout, Dilemma)
    check_disjoint(read_jsonl(s.rl_train, Dilemma), items)
    if limit:
        items = items[:limit]
    rows = holdout_rows(items)
    manifest = {
        "rl_holdout": file_provenance(s.rl_holdout),
        "rl_train": file_provenance(s.rl_train),
        "limit": limit,
        "n_items": len(items),
        "n_families": len({d.family_id for d in items}),
        "n_rows": len(rows),
        "by_principle": dict(sorted(Counter(f"P{d.principle_focus}" for d in items).items())),
        "by_variant_kind": dict(Counter(d.variant_kind for d in items).most_common()),
    }
    return items, rows, manifest


def write_holdout_files(run_dir: Path, rows: list[RLRow], manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / HOLDOUT_DATASET_FILE, rows)
    write_json(run_dir / HOLDOUT_MANIFEST_FILE, manifest)


# ---------------------------------------------------------------------------
# scoring and evaluation
# ---------------------------------------------------------------------------


def holdout_suite(
    settings: RewardSettings, judge, run_dir: Path, max_completion_length: int | None, **kw
) -> RewardSuite:
    """The training reward functions and weights, writing to holdout_rollouts.jsonl with step = global_step."""
    return RewardSuite(settings, judge, run_dir / HOLDOUT_ROLLOUTS_FILE, max_completion_length, step_offset=0, **kw)


def score(
    suite: RewardSuite,
    rows: Sequence[RLRow],
    completions: list[str],
    completion_ids: list[list[int]] | None,
    step: int,
    prompts: list[str] | None = None,
) -> dict[str, list[float]]:
    """Run every reward function on one batch, as TRL does (rows[i] belongs to completions[i]); returns the outputs
    by function name. Metrics the functions log are discarded (the summary recomputes them from the rollouts)."""
    funcs, _ = suite.functions()
    keys = ("task_type", "item_id", "family_id", "letter_order", "verdict", "principles", "answer", "variant_kind")
    kw = {k: [getattr(r, k) for r in rows] for k in keys}
    prompts = prompts if prompts is not None else [r.prompt for r in rows]
    state = SimpleNamespace(global_step=step)
    return {
        f.__name__: f(prompts=prompts, completions=completions, completion_ids=completion_ids, trainer_state=state,
                      log_metric=lambda name, value: None, **kw)
        for f in funcs
    }  # fmt: skip


def read_rollouts(path: Path, offset: int = 0) -> list[dict]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read().decode("utf-8")
    return [json.loads(x) for x in data.splitlines() if x.strip()]


def evaluate(
    rows: Sequence[RLRow],
    generate: Generate,
    suite: RewardSuite,
    step: int,
    k: int,
    meta: dict[str, dict],
    run_dir: Path,
    prompts_per_batch: int = 16,
) -> dict:
    """Generate k answers per row, score them, append the summary to holdout.jsonl and return it."""
    t0 = time.time()
    path = suite.rollouts_path or run_dir / HOLDOUT_ROLLOUTS_FILE
    offset = path.stat().st_size if path.exists() else 0
    for i in range(0, len(rows), prompts_per_batch):
        chunk = rows[i : i + prompts_per_batch]
        rep = [r for r in chunk for _ in range(k)]
        completions, completion_ids = generate([r.prompt for r in rep])
        if len(completions) != len(rep):
            raise RuntimeError(f"generate returned {len(completions)} completions for {len(rep)} prompts")
        score(suite, rep, completions, completion_ids, step)
    records = read_rollouts(path, offset)
    summary = {"step": step, "k": k, "eval_s": round(time.time() - t0, 1), **summarize(records, meta)}
    append_summary(run_dir, summary)
    return summary


def flat_metrics(summary: dict) -> dict[str, float]:
    """Scalar `holdout/*` metrics of a summary for the trainer log (nested breakdowns flattened with `/`)."""
    out: dict[str, float] = {}
    for k, v in summary.items():
        if k in ("step", "k"):
            continue
        if isinstance(v, dict):
            out.update({f"holdout/{k}/{kk}": vv for kk, vv in v.items() if isinstance(vv, int | float)})
        elif isinstance(v, int | float) and not isinstance(v, bool):
            out[f"holdout/{k}"] = v
    return out


def append_summary(run_dir: Path, summary: dict) -> None:
    with (run_dir / HOLDOUT_FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")


def _mean(xs: Sequence[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _se_of_means(groups: dict[str, list[float]]) -> float | None:
    """Standard error of the mean over groups (items), each group reduced to its mean (answers within an item are
    correlated, so the item is the sampling unit)."""
    means = [sum(v) / len(v) for v in groups.values() if v]
    if len(means) < 2:
        return None
    m = sum(means) / len(means)
    var = sum((x - m) ** 2 for x in means) / (len(means) - 1)
    return round(math.sqrt(var / len(means)), 4)


def summarize(records: list[dict], meta: dict[str, dict]) -> dict:
    """One evaluation's summary from its rollout records (rewards unscaled, as in the training logs)."""
    if not records:
        return {"n_answers": 0}
    outcome = [r["rewards"].get("r_outcome", 0.0) for r in records]
    by_item: dict[str, list[float]] = defaultdict(list)
    by_principle: dict[str, list[float]] = defaultdict(list)
    by_variant: dict[str, list[float]] = defaultdict(list)
    for r, o in zip(records, outcome, strict=True):
        m = meta.get(r["item_id"], {})
        by_item[r["item_id"]].append(o)
        by_principle[f"P{m.get('principle_focus')}"].append(o)
        by_variant[m.get("variant_kind") or r.get("variant_kind") or "?"].append(o)
    parsed = [r for r in records if r.get("decision") in ("action1", "action2")]
    lengths = [r["n_tokens"] for r in records if r.get("n_tokens") is not None]
    out = {
        "n_answers": len(records),
        "n_items": len(by_item),
        "outcome": _mean(outcome),
        "outcome_se": _se_of_means(by_item),
        "total": _mean([r["total"] for r in records]),
        "by_principle": {k: _mean(v) for k, v in sorted(by_principle.items())},
        "by_variant_kind": {k: _mean(v) for k, v in sorted(by_variant.items())},
        "by_letter_order": {
            o: _mean([x for r, x in zip(records, outcome, strict=True) if r.get("letter_order") == o])
            for o in ("AB", "BA")
        },
        "parse_rate": round(len(parsed) / len(records), 4),
        "letter_a_share": _mean([float(r.get("letter") == "A") for r in parsed]),
        "mention_rate": _mean([float(bool(r.get("mention"))) for r in records]),
        "length_mean": _mean([float(x) for x in lengths]),
        "truncated_share": _mean([float(bool(r.get("truncated"))) for r in records]),
    }
    # citations: deterministic classes for every run (C3 too); the reward component and judge labels for C4
    citing = [r for r in records if r.get("citation") and (r.get("mention") or r["citation"].get("cited")
                                                          or r["citation"].get("fabricated"))]  # fmt: skip
    if citing:
        reasons = Counter(r["citation"]["reason"] for r in citing)
        out["cite_det"] = {k: round(v / len(citing), 4) for k, v in sorted(reasons.items())}
    if any("r_cite" in r["rewards"] for r in records):
        cite = [r["rewards"].get("r_cite", 0.0) for r in records]
        out["r_cite"] = _mean(cite)
        if citing:
            vals = [r["rewards"].get("r_cite", 0.0) for r in citing]
            out["cite_class"] = {
                "pos": _mean([float(v > 0) for v in vals]),
                "zero": _mean([float(v == 0) for v in vals]),
                "neg": _mean([float(v < 0) for v in vals]),
            }
        labels = Counter(r["judge_label"] for r in records if r.get("judge_label") is not None)
        n_lab = sum(labels.values())
        out["judge"] = {k: round(v / n_lab, 4) for k, v in sorted(labels.items())} if n_lab else {}
    return out


def recompute(run_dir: Path, meta: dict[str, dict]) -> list[dict]:
    """holdout.jsonl lines recomputed from holdout_rollouts.jsonl (one per evaluated step)."""
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in read_rollouts(run_dir / HOLDOUT_ROLLOUTS_FILE):
        by_step[r["step"]].append(r)
    return [{"step": s, **summarize(recs, meta)} for s, recs in sorted(by_step.items())]


def read_summaries(run_dir: Path) -> list[dict]:
    path = run_dir / HOLDOUT_FILE
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# fallback: offline answers (calign.dilemmas.filter sample) scored with the training reward code
# ---------------------------------------------------------------------------


def offline_batch(records: list[GenerationRecord], items: list[Dilemma]) -> tuple[list[RLRow], list[str], list | None]:
    """(rows, completions, completion ids) for filter-sample records of hold-out items, each record paired with the
    hold-out row of its letter order."""
    rows = {(r.item_id, r.letter_order): r for r in holdout_rows(items)}
    out_rows, texts, ids = [], [], []
    for rec in records:
        row = rows.get((rec.scenario_id, rec.extra.get("letter_order", "AB")))
        if row is None:
            continue
        out_rows.append(row)
        texts.append(rec.response_text)
        ids.append(rec.extra.get("completion_token_ids"))
    if not out_rows:
        raise ValueError("no record matches a hold-out item")
    return out_rows, texts, ids if all(x is not None for x in ids) else None


def score_offline(args: argparse.Namespace) -> dict:
    from calign.rl.judge_server import JudgeClient

    cfg = load_rl_config(args.config)
    items, _, _ = load_holdout(cfg)
    records = read_jsonl(args.records, GenerationRecord)
    rows, texts, ids = offline_batch(records, items)
    judge = JudgeClient(cfg.judge, cache_path=cfg.judge.cache_path or args.run_dir / "judge_cache.jsonl") \
        if cfg.uses_judge else None  # fmt: skip
    suite = holdout_suite(cfg.reward, judge, args.run_dir, cfg.grpo.max_completion_length)
    path = args.run_dir / HOLDOUT_ROLLOUTS_FILE
    offset = path.stat().st_size if path.exists() else 0
    # group key per item and letter order (what a prompt is in training)
    score(suite, rows, texts, ids, args.step, prompts=[f"{r.item_id}|{r.letter_order}" for r in rows])
    summary = {
        "step": args.step,
        "k": None,
        "source": "offline",
        "records": file_provenance(args.records),
        "length_unit": "tokens" if ids is not None else "chars",
        **summarize(read_rollouts(path, offset), item_meta(items)),
    }
    append_summary(args.run_dir, summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize", help="recompute holdout.jsonl from holdout_rollouts.jsonl (prints, --write saves)")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--holdout", type=Path, default=RLDataSettings().rl_holdout, help="hold-out items (metadata)")
    s.add_argument("--write", action="store_true", help="write holdout_recomputed.jsonl next to holdout.jsonl")
    o = sub.add_parser("score-offline", help="fallback: score filter-sample records of the hold-out at one step")
    o.add_argument("--config", required=True)
    o.add_argument("--run-dir", type=Path, required=True, help="the RL run dir (files are appended there)")
    o.add_argument("--step", type=int, required=True, help="checkpoint step (0 = the RL start)")
    o.add_argument("--records", type=Path, required=True)
    args = ap.parse_args(argv)
    from calign.paths import load_env

    load_env()
    if args.cmd == "summarize":
        lines = recompute(args.run_dir, item_meta(read_jsonl(args.holdout, Dilemma)))
        for line in lines:
            print(json.dumps(line))
        if args.write:
            (args.run_dir / "holdout_recomputed.jsonl").write_text(
                "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8"
            )
    else:
        print(json.dumps(score_offline(args), indent=1))


if __name__ == "__main__":
    main()
