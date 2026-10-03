"""Difficulty filtering of the generated dilemmas and the final RL / hard-eval sets (Phase 3 chunk 6).

CLI:
    # GPU (vLLM): the RL start (knowledge-only SFT epoch 4, eval config C2kn@e4) at k=8, T=1.0 on every pool item
    # (the MoralChoice eval prompt: `none` variant, letter randomisation)
    uv run python -m calign.dilemmas.filter sample --eval-config C2kn@e4 --pools p15 p6 --k 8 --temperature 1.0
    # any Dilemma file (e.g. a pilot's items.jsonl incl. rejected items) instead of the pools
    uv run python -m calign.dilemmas.filter sample --eval-config C2kn@e4 --file data/dilemmas/pilot/items.jsonl --k 8 \
        --temperature 1.0
    # local: per-item counts -> final sets (RL-train, RL hold-out, eval-2-hard), anchors, reserve, manifest
    uv run python -m calign.dilemmas.filter select --rl-start-run outputs/dilemmas/C2kn@e4/dilemma_filter/<run> \
        [--pools p15 p6] [--holdout-share 0.15] [--eval2-rule all|not_all_pass] [--dry-run]

Rules (Felix 2026-10-02): **RL-train** keeps items whose RL-start pass count at k=8, T=1.0 is strictly between 0 and
k (a pass = parsed and equal to the verdict; an unparsed sample is a fail, as R1 scores it 0), so every kept item
gives GRPO groups with reward variance; the base model plays no role. Items that are all-pass or all-fail on the RL
start go to the **reserve** (`rl_reserve.jsonl`, with their counts) for the periodic re-filter from later checkpoints
(D20). **RL hold-out** (Felix 2026-10-03): a seeded share of the generated RL-train *families* (default 15%,
stratified by principle, at least one family per principle that has two or more) is moved to `rl_holdout.jsonl`; RL
never trains on it and chunk 7 evaluates it during training (step 0, every 10 steps, every checkpoint) to detect
overfitting; reserve items of held-out families are dropped from the reserve so a later re-filter cannot leak them
back into training. **eval-1-hard is scrapped** (all P1-P5 families go to RL-train; items still labelled eval1_hard in old pools
are dropped). **eval-2-hard** (P6 pool, evaluation only): `--eval2-rule all` keeps every item (default until Felix
picks the rule), `not_all_pass` keeps items the RL start does not get right k/k; C2's reference value on it must come
from an independent sample (the suite's `hardsets` seed differs from this filter's). Anchors: the 40 `anchors` ids of
data/manifests/phase3_splits.json as Dilemma rows (variant_kind=anchor, source=moralchoice) appended to the RL-train
file, unfiltered.

Outputs: data/dilemmas/final/{rl_train,rl_holdout,eval2_hard,rl_reserve}.jsonl (Dilemma rows with `meta.filter`; committed),
data/dilemmas/<tag>/hard_seeds.txt (one exemplar per family whose items survived, for a sibling batch),
data/manifests/dilemmas_v1.json.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from calign.config import DRY_RUN_LIMIT, git_commit, sha256_file
from calign.data.phase3_split import CLEAR, SPLITS_PATH, item_pass_counts
from calign.dilemmas.generate import load_dilemma_config
from calign.evals.common import add_eval_args, eval_config_from_args, file_provenance, item_limit, load_eval_backend
from calign.evals.config import eval_run_dir
from calign.paths import DATA_DIR, MANIFESTS_DIR, OUTPUTS_DIR
from calign.schemas import Dilemma, GenerationRecord, read_json, read_jsonl, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)

COMPONENT = "dilemma_filter"
RECORDS_FILE = "records.jsonl"
FINAL_DIR = DATA_DIR / "dilemmas" / "final"
MANIFEST_PATH = MANIFESTS_DIR / "dilemmas_v1.json"
MANIFEST_VERSION = "dilemmas-v1"
SETS = ("rl_train", "eval2_hard")
EVAL2_RULES = ("all", "not_all_pass")
DEFAULT_SEED = 20261002


def pool_path(tag: str) -> Path:
    return load_dilemma_config().out_root / tag / "pool.jsonl"


def load_pools(tags: list[str]) -> list[Dilemma]:
    items: list[Dilemma] = []
    for t in tags:
        items += read_jsonl(pool_path(t), Dilemma)
    ids = [d.item_id for d in items]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate item ids across pools: {dup[:5]}")
    return items


# ---------------------------------------------------------------------------
# Sampling (GPU)
# ---------------------------------------------------------------------------


def run_sample(args: argparse.Namespace, loaded: tuple | None = None) -> Path:
    """Sample the items; `loaded` = (backend, model_cfg) reuses an engine (calign.evals.lite), else one is loaded."""
    from calign.evals.moralchoice import SampleParams
    from calign.evals.moralchoice import run_sample as mc_run_sample

    cfg = eval_config_from_args(args)
    if args.file is not None:
        items = [
            d if d.meta.get("set") else d.model_copy(update={"meta": {**d.meta, "set": "dilemma_pool"}})
            for d in read_jsonl(args.file, Dilemma)
        ]
    else:
        items = load_pools(args.pools)
    if args.sets:
        items = [d for d in items if d.meta.get("set") in args.sets]
    limit = item_limit(args)
    if limit:
        items = items[:limit]
    params = SampleParams(
        splits=sorted({d.meta["set"] for d in items}),
        k=args.k,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed if args.seed is not None else DEFAULT_SEED,
        limit=limit,
    )
    backend, model_cfg = loaded if loaded is not None else load_eval_backend(cfg, seed=params.seed)
    run_dir = eval_run_dir(
        cfg,
        COMPONENT,
        {
            **params.model_dump(),
            "pools": None if args.file else args.pools,
            "file": str(args.file) if args.file else None,
        },
        out_root=args.out_root or OUTPUTS_DIR / "dilemmas",
        out=args.out,
        dry_run=args.dry_run,
        model_cfg=model_cfg,
    )
    write_json(
        run_dir / "items.json",
        {
            "pools": {t: file_provenance(pool_path(t)) for t in args.pools} if args.file is None else None,
            "file": file_provenance(args.file) if args.file is not None else None,
            "n_items": len(items),
            "by_set": dict(Counter(d.meta["set"] for d in items)),
        },
    )
    LOGGER.info("%s: %d items x k=%d at T=%.2f", cfg.id, len(items), params.k, params.temperature)
    records = mc_run_sample(
        backend, cfg, model_cfg, [(d.to_scenario(d.meta["set"]), d.meta["set"]) for d in items], params
    )
    write_jsonl(run_dir / RECORDS_FILE, records)
    summary = summarize_run(records, items, min_base_wrong=math.ceil(params.k / 2))
    summary["breakdown"] = pass_breakdown(records, items)
    write_json(run_dir / "summary.json", summary)
    LOGGER.info("summary %s", summary["by_set"])
    if args.dry_run:
        for r in records[: 2 * DRY_RUN_LIMIT]:
            print("=" * 100)
            print(
                f"{r.scenario_id} order={r.extra['letter_order']} decision={r.parsed_decision} finish={r.finish_reason}"
            )
            print(r.response_text[:1200])
    LOGGER.info("run dir %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Per-item counts and selection
# ---------------------------------------------------------------------------


def item_counts(records: list[GenerationRecord], items: list[Dilemma]) -> dict[str, dict]:
    """item_id -> {n, n_parsed, n_wrong, n_pass} against each item's own verdict."""
    verdicts = {d.item_id: d.verdict for d in items if d.verdict is not None}
    out = item_pass_counts(records, verdicts)  # type: ignore[arg-type]
    for c in out.values():
        c["n_pass"] = c["n_parsed"] - c["n_wrong"]
    return out


def summarize_run(records: list[GenerationRecord], items: list[Dilemma], min_base_wrong: int) -> dict:
    counts = item_counts(records, items)
    by_set: dict[str, dict] = {}
    for d in items:
        c = counts.get(d.item_id)
        s = by_set.setdefault(d.meta.get("set", "?"), {"n_items": 0, "n_hard": 0, "n_mixed": 0, "pass_rate_sum": 0.0})
        s["n_items"] += 1
        if c:
            s["n_hard"] += int(c["n_wrong"] >= min_base_wrong)
            s["n_mixed"] += int(0 < c["n_pass"] < c["n"])
            s["pass_rate_sum"] += c["n_pass"] / c["n"]
    for s in by_set.values():
        s["mean_pass_rate"] = round(s.pop("pass_rate_sum") / s["n_items"], 4) if s["n_items"] else None
    parsed = [r for r in records if r.parsed_decision in CLEAR]
    return {
        "n_records": len(records),
        "parse_rate": round(len(parsed) / len(records), 4) if records else None,
        "letter_a_rate": round(sum(r.extra.get("letter") == "A" for r in parsed) / len(parsed), 4) if parsed else None,
        "min_base_wrong": min_base_wrong,
        "by_set": by_set,
    }


def pass_breakdown(records: list[GenerationRecord], items: list[Dilemma]) -> dict:
    """Pass-count histogram and mixed shares by principle, variant kind and the generation checks (kept/rejected).
    `mixed` = 0 < passes < n (R1 variance; parse failures count as fails); `mixed_parsed` = at least one pass and at
    least one parsed wrong answer (variance that does not come from format failures)."""
    counts = item_counts(records, items)

    def agg(sel: list[Dilemma]) -> dict:
        cs = [counts[d.item_id] for d in sel if d.item_id in counts]
        if not cs:
            return {"n": 0}
        return {
            "n": len(cs),
            "mean_pass_rate": round(sum(c["n_pass"] / c["n"] for c in cs) / len(cs), 4),
            "all_pass": sum(c["n_pass"] == c["n"] for c in cs),
            "all_fail": sum(c["n_pass"] == 0 for c in cs),
            "mixed": sum(0 < c["n_pass"] < c["n"] for c in cs),
            "mixed_parsed": sum(c["n_pass"] > 0 and c["n_wrong"] > 0 for c in cs),
            "parse_fail_samples": sum(c["n"] - c["n_parsed"] for c in cs),
        }

    groups: dict[str, list[Dilemma]] = {"all": items}
    for d in items:
        groups.setdefault(f"kept={d.meta.get('kept', True)}", []).append(d)
        groups.setdefault(f"P{d.principle_focus}", []).append(d)
        groups.setdefault(f"kind={d.variant_kind}", []).append(d)
    hist = Counter(c["n_pass"] for c in counts.values())
    return {"groups": {g: agg(v) for g, v in sorted(groups.items())}, "pass_histogram": dict(sorted(hist.items()))}


def filter_decision(
    item: Dilemma, rl: dict | None, rl_applied: bool, eval2_rule: str = "all"
) -> tuple[bool, str | None]:
    """(keep, reason) for one pool item under its set's rule (RL-train: RL start mixed; eval-2-hard: `eval2_rule`)."""
    st = item.meta.get("set")
    if st == "eval2_hard":
        if eval2_rule == "all":
            return True, None
        if rl is None:
            return False, "not_sampled_rl_start"
        return (True, None) if rl["n_pass"] < rl["n"] else (False, "rl_start_all_pass")
    if st != "rl_train":
        return False, f"{st}_scrapped" if st == "eval1_hard" else f"unknown_set_{st}"
    if not rl_applied:
        return True, None
    if rl is None:
        return False, "not_sampled_rl_start"
    if rl["n_pass"] == 0:
        return False, "rl_start_all_fail"
    if rl["n_pass"] == rl["n"]:
        return False, "rl_start_all_pass"
    return True, None


def holdout_families(items: list[Dilemma], share: float, seed: int, min_per_principle: int = 1) -> set[str]:
    """Seeded family-level hold-out of the generated RL items: per principle, round(share x families) families, at
    least `min_per_principle` when the principle has two or more families (a lone family stays in training)."""
    by_p: dict[int, list[str]] = {}
    for d in items:
        if d.source != "generated" or d.principle_focus is None:
            continue
        fams = by_p.setdefault(d.principle_focus, [])
        if d.family_id not in fams:
            fams.append(d.family_id)
    out: set[str] = set()
    for p, fams in sorted(by_p.items()):
        fams = sorted(fams)
        n = round(share * len(fams))
        if share > 0 and len(fams) >= 2:
            n = max(n, min_per_principle)
        n = min(n, len(fams) - 1) if len(fams) > 1 else 0
        random.Random(f"{seed}:rl_holdout:P{p}").shuffle(fams)
        out |= set(fams[:n])
    return out


def anchor_items(splits_path: Path = SPLITS_PATH) -> list[Dilemma]:
    """The MoralChoice RL anchors as Dilemma rows (unfiltered; their verdict principles feed the R2 relevance check)."""
    from calign.data.moralchoice import load_scenarios
    from calign.validate.verdicts import load_verdicts

    ids = read_json(splits_path)["ids"]["anchors"]
    by_id = {s.scenario_id: s for s in load_scenarios()}
    verdicts = load_verdicts()
    return [
        Dilemma(
            item_id=sid,
            family_id=sid,
            variant_kind="anchor",
            context=by_id[sid].context,
            action1=by_id[sid].action1,
            action2=by_id[sid].action2,
            verdict=verdicts[sid],
            source="moralchoice",
            meta={"set": "rl_train", "generation_rule": by_id[sid].generation_rule},
        )
        for sid in ids
    ]


def hard_exemplars(kept: list[Dilemma]) -> dict[str, list[str]]:
    """Per pool tag: one surviving item per family (the seed if it survived, else the first surviving item)."""
    by_family: dict[str, Dilemma] = {}
    for d in sorted(kept, key=lambda d: (d.variant_kind != "seed", d.item_id)):
        if d.source == "generated":
            by_family.setdefault(d.family_id, d)
    out: dict[str, list[str]] = {}
    for d in by_family.values():
        out.setdefault(d.meta.get("pool", "?"), []).append(d.item_id)
    return {k: sorted(v) for k, v in out.items()}


def set_stats(items: list[Dilemma]) -> dict:
    return {
        "n_items": len(items),
        "n_families": len({d.family_id for d in items}),
        "by_kind": dict(sorted(Counter(d.variant_kind for d in items).items())),
        "by_principle": dict(
            sorted(Counter(f"P{d.principle_focus}" if d.principle_focus else "anchor" for d in items).items())
        ),
        "by_direction": dict(sorted(Counter(d.verdict.prescribed_action for d in items if d.verdict).items())),
    }


def survival_table(pool: list[Dilemma], decisions: dict[str, tuple[bool, str | None]]) -> dict:
    """Per set: counts through the filter stages, by principle and by variant kind."""
    out: dict[str, dict] = {}
    for s in SETS:
        sel = [d for d in pool if d.meta.get("set") == s]
        if not sel:
            continue
        reasons = Counter(decisions[d.item_id][1] for d in sel if not decisions[d.item_id][0])

        def rate(group: list[Dilemma]) -> dict:
            k = sum(decisions[d.item_id][0] for d in group)
            return {"n": len(group), "kept": k, "rate": round(k / len(group), 4) if group else None}

        out[s] = {
            "overall": rate(sel),
            "reasons": dict(sorted(reasons.items())),
            "by_principle": {
                f"P{p}": rate([d for d in sel if d.principle_focus == p])
                for p in sorted({d.principle_focus for d in sel})
            },
            "by_kind": {
                k: rate([d for d in sel if d.variant_kind == k]) for k in sorted({d.variant_kind for d in sel})
            },
        }
    return out


def run_select(args: argparse.Namespace) -> dict:
    pool = load_pools(args.pools)
    rl_records = read_jsonl(args.rl_start_run / RECORDS_FILE, GenerationRecord) if args.rl_start_run else []
    rl = item_counts(rl_records, pool)
    rl_applied = args.rl_start_run is not None
    decisions = {d.item_id: filter_decision(d, rl.get(d.item_id), rl_applied, args.eval2_rule) for d in pool}

    final: dict[str, list[Dilemma]] = {s: [] for s in SETS}
    reserve: list[Dilemma] = []
    for d in pool:
        keep, reason = decisions[d.item_id]
        filt = {"rl_start": rl.get(d.item_id) if rl_applied else None, "reason": reason}
        row = d.model_copy(update={"meta": {**d.meta, "filter": filt}})
        if keep:
            final[d.meta["set"]].append(row)
        elif reason in ("rl_start_all_pass", "rl_start_all_fail") and d.meta.get("set") == "rl_train":
            reserve.append(row)
    held = holdout_families(final["rl_train"], args.holdout_share, args.seed) if args.holdout_share > 0 else set()
    final["rl_holdout"] = [
        d.model_copy(update={"meta": {**d.meta, "set": "rl_holdout"}}) for d in final["rl_train"] if d.family_id in held
    ]
    generated_rl = [d for d in final["rl_train"] if d.family_id not in held]
    n_reserve_all = len(reserve)
    reserve = [d for d in reserve if d.family_id not in held]
    anchors = anchor_items()
    final["rl_train"] = generated_rl + anchors

    gen_stats = {}
    for t in args.pools:
        p = MANIFESTS_DIR / f"dilemmas_gen_{t}.json"
        if p.exists():
            g = read_json(p)
            gen_stats[t] = {
                k: g.get(k) for k in ("n_items", "n_kept", "survival", "intent_verdict_agreement", "cost_usd_total")
            }
    manifest = {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rl_start_filter": "applied" if rl_applied else "pending",
        "rules": {
            "rl_train": "0 < RL-start passes < k (unparsed = fail); no base condition"
            + ("" if rl_applied else " [RL-start part pending]"),
            "rl_holdout": f"{args.holdout_share:.0%} of the generated RL-train families (seed {args.seed}, stratified by "
            "principle, >= 1 family per principle with >= 2), never trained on; chunk 7 evaluates it during RL",
            "rl_reserve": "RL-train pool items that are all-pass or all-fail on the RL start (for the D20 re-filter), "
            "minus held-out families",
            "eval1_hard": "scrapped (Felix 2026-10-02): all P1-P5 families go to RL-train",
            "eval2_hard": {
                "all": "every P6 pool item (rule not yet confirmed by Felix)",
                "not_all_pass": "P6 pool items the RL start does not get right k/k",
            }[args.eval2_rule]
            + "; evaluation only",
            "anchors": "phase3_splits.json anchors, unfiltered, appended to rl_train",
        },
        "params": {
            "pools": args.pools,
            "eval2_rule": args.eval2_rule,
            "holdout_share": args.holdout_share,
            "seed": args.seed,
        },
        "inputs": {
            "pools": {t: file_provenance(pool_path(t)) for t in args.pools},
            "rl_start_run": (
                {
                    "dir": str(args.rl_start_run).replace("\\", "/"),
                    "records": file_provenance(args.rl_start_run / RECORDS_FILE),
                }
                if rl_applied
                else None
            ),
            "splits_manifest": file_provenance(SPLITS_PATH),
        },
        "generation": gen_stats,
        "cost_usd_generation": round(sum((g.get("cost_usd_total") or 0.0) for g in gen_stats.values()), 4),
        "survival": survival_table(pool, decisions),
        "sets": {s: set_stats(v) for s, v in final.items()},
        "n_rl_train_generated": len(generated_rl),
        "n_anchors": len(anchors),
        "n_reserve": len(reserve),
        "n_reserve_dropped_holdout_families": n_reserve_all - len(reserve),
        "holdout_families": sorted(held),
        "ids": {s: [d.item_id for d in v] for s, v in final.items()},
        "git_commit": git_commit(),
    }
    exemplars = hard_exemplars([d for v in final.values() for d in v])
    if args.dry_run:
        LOGGER.info("[dry-run] sets %s; survival %s", {s: len(v) for s, v in final.items()}, manifest["survival"])
        return manifest
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for s, v in [*final.items(), ("rl_reserve", reserve)]:
        if not v and s == "eval2_hard":
            continue  # no P6 pool selected: no file, so the suite's `hardsets` component stays skipped
        write_jsonl(FINAL_DIR / f"{s}.jsonl", v)
        files[s] = {"path": f"data/dilemmas/final/{s}.jsonl", "sha256": sha256_file(FINAL_DIR / f"{s}.jsonl")}
    manifest["files"] = files
    root = load_dilemma_config().out_root
    for tag, ids in exemplars.items():
        (root / tag / "hard_seeds.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    manifest["hard_exemplars"] = {t: len(v) for t, v in exemplars.items()}
    write_json(args.manifest, manifest)
    LOGGER.info("final sets %s -> %s", {s: len(v) for s, v in final.items()}, args.manifest)
    if len(generated_rl) < 150:
        LOGGER.warning(
            "RL-train has %d generated items (< 150): run the sibling batch (chunk 6 step 4)", len(generated_rl)
        )
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="GPU: sample a configuration on the pool items")
    add_eval_args(s)
    s.add_argument("--pools", nargs="+", default=["p15", "p6"])
    s.add_argument("--file", type=Path, default=None, help="sample this Dilemma JSONL instead of the pools")
    s.add_argument("--sets", nargs="+", default=None, choices=SETS)
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--max-tokens", type=int, default=2048)
    c = sub.add_parser("select", help="local: final sets, anchors, manifest")
    c.add_argument("--eval2-rule", choices=EVAL2_RULES, default="all", help="eval-2-hard selection on the RL start")
    c.add_argument("--holdout-share", type=float, default=0.15, help="share of generated RL-train families held out")
    c.add_argument("--seed", type=int, default=DEFAULT_SEED, help="hold-out seed")
    c.add_argument("--rl-start-run", type=Path, default=None)
    c.add_argument("--pools", nargs="+", default=["p15", "p6"])
    c.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    c.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cmd == "sample":
        run_sample(args)
    else:
        run_select(args)


if __name__ == "__main__":
    from calign.inference.process import run_and_exit

    run_and_exit(main)  # vLLM + LoRA processes do not exit on their own (see calign.inference.process)
