"""Reward scale for C4 (chunk 7): make C4's typical advantage at the RL start equal to C3's.

Why: with Adam, the overall size of the reward hardly changes the update size, but it does change the balance
between the reward signal and the KL penalty (weight 0.02, not scaled with the reward). If the citation term makes
C4's advantages k times larger than C3's, C4 effectively trains with a KL weight of 0.02 / k and can drift further
from the RL start. Multiplying C4's rewards by f = S3 / S4 removes that difference at the start (phase3/chunks/07).

Procedure:
1. Answers from the RL start that look like training rollouts. Dilemmas: chunk 6's RL-start sampling run (8 answers
   per item at T=1.0, 4 per letter order; one group = the answers of one item in one letter order), restricted to
   the generated RL-train items. Anchors and math (not in that run): `sample` generates 8 answers per prompt for the
   anchor rows (both letter orders) and 64 MATH train problems, with the training sampling settings.
2. Every answer is scored with both reward definitions by the training reward code (calign.rl.rewards): C3 = outcome
   + math - math mention penalty; C4 = C3 + 0.5 x mention x citation score (deterministic checks, then the local
   judge server).
3. Typical advantage size per definition: within each group the unbiased variance s^2 (divisor n - 1); the expected
   squared advantage of a training group of G = 8 answers is s^2 x (G - 1) / G, which makes groups of 4 and 8
   comparable. Mean over the groups of each task type, weighted by the training mix (dilemma 1 - anchor - math share,
   anchor share, math share); S = square root.
4. f = S3 / S4, written to summary.json with the line to paste into configs/rl/C4.yaml (`reward.scale`,
   `reward.scale_source`).

CLI (RL node):
    # GPU, before the servers start (one GPU, ~5 min): anchor and math answers from the RL start
    CUDA_VISIBLE_DEVICES=0 uv run python -m calign.rl.reward_scale sample --config configs/rl/C4.yaml \
        --out outputs/rl/reward_scale/rs1 [--n-math 64] [--model-path P]
    # judge server up (scripts/brev/rl_serve.sh): score and measure
    uv run python -m calign.rl.reward_scale measure --config configs/rl/C4.yaml --run-dir outputs/rl/reward_scale/rs1 \
        --dilemma-records outputs/dilemmas/<C2@eK>/dilemma_filter/<run>/records.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from calign.config import git_commit
from calign.evals.common import file_provenance
from calign.rl.config import RLConfig, load_rl_config, model_spec, resolve_model_dir
from calign.rl.prompts import RLRow, dilemma_rows, item_principles, math_row
from calign.schemas import Dilemma, GenerationRecord, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)

SAMPLES_FILE = "samples.jsonl"
SCORES_FILE = "scores.jsonl"
DEFAULT_SEED = 20261004
TRAIN_GROUP_SIZE = 8


# ---------------------------------------------------------------------------
# step 1: anchor and math answers (GPU)
# ---------------------------------------------------------------------------


def sample_rows(cfg: RLConfig, n_math: int, seed: int) -> list[RLRow]:
    from calign.rl.dataset import drop_overlap, load_math_pool, math500_problem_set, split_items

    _, anchors = split_items(read_jsonl(cfg.data.rl_train, Dilemma))
    rows = [r for d in anchors for r in dilemma_rows(d)]
    pool, _ = load_math_pool(cfg.data)
    pool, _ = drop_overlap(pool, math500_problem_set())
    pool = sorted(pool, key=lambda p: p["id"])
    random.Random(seed).shuffle(pool)
    rows += [math_row(p["id"], p["problem"], p["answer"]) for p in pool[:n_math]]
    return rows


def run_sample(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from calign.paths import hf_token

    cfg = load_rl_config(args.config)
    spec, rev = model_spec(cfg, args.model_path, args.revision)
    model_dir = resolve_model_dir(spec, rev)
    rows = sample_rows(cfg, args.n_math, args.seed)
    if args.dry_run:
        rows = rows[:3]
    tok = AutoTokenizer.from_pretrained(model_dir, token=hf_token())
    prompts = [{"prompt_token_ids": tok(text=r.prompt)["input_ids"]} for r in rows]  # as TRL tokenizes
    g = cfg.grpo
    llm = LLM(model=model_dir, max_model_len=g.vllm_max_model_length, gpu_memory_utilization=0.85, seed=args.seed)
    sp = SamplingParams(
        n=TRAIN_GROUP_SIZE, temperature=g.temperature, top_p=g.top_p, max_tokens=g.max_completion_length, seed=args.seed
    )
    outs = llm.generate(prompts, sp)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / SAMPLES_FILE).open("w", encoding="utf-8") as f:
        for r, o in zip(rows, outs, strict=True):
            row = r.model_dump(exclude={"prompt"}) | {
                "completions": [c.text for c in o.outputs],
                "n_tokens": [len(c.token_ids) for c in o.outputs],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        args.out / "sample_meta.json",
        {
            "model_spec": spec,
            "model_dir": model_dir,
            "revision": rev,
            "n_rows": len(rows),
            "by_task_type": dict(Counter(r.task_type for r in rows)),
            "answers_per_row": TRAIN_GROUP_SIZE,
            "temperature": g.temperature,
            "max_tokens": g.max_completion_length,
            "seed": args.seed,
            "git_commit": git_commit(),
        },
    )
    LOGGER.info("%d rows x %d answers -> %s", len(rows), TRAIN_GROUP_SIZE, args.out / SAMPLES_FILE)


# ---------------------------------------------------------------------------
# steps 2-4: score and measure
# ---------------------------------------------------------------------------


def dilemma_answers(records_path: Path, items: list[Dilemma]) -> list[tuple[dict, str]]:
    """(row metadata, answer text) for chunk 6's RL-start answers on the generated RL-train items."""
    by_id = {d.item_id: d for d in items if d.variant_kind != "anchor"}
    out = []
    for r in read_jsonl(records_path, GenerationRecord):
        d = by_id.get(r.scenario_id)
        if d is None or d.verdict is None:
            continue
        meta = {
            "task_type": "dilemma",
            "item_id": d.item_id,
            "letter_order": r.extra.get("letter_order", "AB"),
            "verdict": d.verdict.prescribed_action,
            "principles": item_principles(d),
            "answer": "",
        }
        out.append((meta, r.response_text))
    return out


def sampled_answers(run_dir: Path) -> list[tuple[dict, str]]:
    out = []
    for line in (run_dir / SAMPLES_FILE).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            meta = {k: v for k, v in row.items() if k not in ("completions", "n_tokens")}
            out += [(meta, c) for c in row["completions"]]
    return out


def group_key(meta: dict) -> str:
    return f"{meta['task_type']}|{meta['item_id']}|{meta.get('letter_order', '')}"


def expected_sq_advantage(values: list[float], group_size: int = TRAIN_GROUP_SIZE) -> float | None:
    """Expected squared advantage in a training group of `group_size`, from a group of any size >= 2."""
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    s2 = sum((v - m) ** 2 for v in values) / (n - 1)
    return s2 * (group_size - 1) / group_size


def advantage_size(totals: list[float], metas: list[dict], shares: dict[str, float]) -> dict:
    """S = sqrt(sum over task types of share x mean expected squared advantage of that type's groups)."""
    groups: dict[str, list[float]] = defaultdict(list)
    types: dict[str, str] = {}
    for t, m in zip(totals, metas, strict=True):
        k = group_key(m)
        groups[k].append(t)
        types[k] = m["task_type"]
    per_type: dict[str, list[float]] = defaultdict(list)
    for k, vals in groups.items():
        e = expected_sq_advantage(vals)
        if e is not None:
            per_type[types[k]].append(e)
    mean_sq = {t: sum(v) / len(v) for t, v in per_type.items() if v}
    used = {t: w for t, w in shares.items() if w > 0}
    missing = sorted(t for t in used if t not in mean_sq)
    if missing:
        raise ValueError(f"no groups for task types {missing}; cannot weight the training mix")
    total = sum(w * mean_sq[t] for t, w in used.items()) / sum(used.values())
    return {
        "S": math.sqrt(total),
        "rms_by_type": {t: math.sqrt(v) for t, v in mean_sq.items()},
        "n_groups_by_type": {t: len(v) for t, v in per_type.items()},
        "zero_variance_share_by_type": {t: sum(e < 1e-12 for e in v) / len(v) for t, v in per_type.items()},
    }


def run_measure(args: argparse.Namespace) -> dict:
    from calign.rl.config import RewardSettings
    from calign.rl.judge_server import JudgeClient
    from calign.rl.rewards import RewardSuite

    cfg = load_rl_config(args.config)
    items = read_jsonl(cfg.data.rl_train, Dilemma)
    answers = dilemma_answers(args.dilemma_records, items) + sampled_answers(args.run_dir)
    if args.limit:
        answers = answers[: args.limit]
    metas = [m for m, _ in answers]
    texts = [t for _, t in answers]
    judge = JudgeClient(cfg.judge, cache_path=args.run_dir / "judge_cache.jsonl")
    c3 = RewardSuite(RewardSettings(kind="outcome", cite_lambda=cfg.reward.cite_lambda,
                                    math_mention_lambda=cfg.reward.math_mention_lambda))  # fmt: skip
    c4 = RewardSuite(RewardSettings(kind="outcome_cite", cite_lambda=cfg.reward.cite_lambda,
                                    math_mention_lambda=cfg.reward.math_mention_lambda), judge)  # fmt: skip
    s3, s4 = c3.score(metas, texts), c4.score(metas, texts)
    t3, t4 = [sum(x.values()) for x in s3], [sum(x.values()) for x in s4]
    with (args.run_dir / SCORES_FILE).open("w", encoding="utf-8") as f:
        for m, a, b in zip(metas, s3, s4, strict=True):
            f.write(json.dumps({"group": group_key(m), "task_type": m["task_type"], "c3": a, "c4": b}) + "\n")
    d = cfg.data
    shares = {"dilemma": 1.0 - d.anchor_share - d.math_share, "anchor": d.anchor_share, "math": d.math_share}
    size3, size4 = advantage_size(t3, metas, shares), advantage_size(t4, metas, shares)
    f_scale = size3["S"] / size4["S"]
    cite_vals = [x.get("r_cite", 0.0) for x, m in zip(s4, metas, strict=True) if m["task_type"] != "math"]
    summary = {
        "S3": size3["S"],
        "S4": size4["S"],
        "scale": f_scale,
        "c3": size3,
        "c4": size4,
        "shares": shares,
        "n_answers_by_type": dict(Counter(m["task_type"] for m in metas)),
        "c4_citation_classes": dict(Counter("pos" if v > 0 else "neg" if v < 0 else "zero" for v in cite_vals)),
        "judge": {"model": cfg.judge.hf_model, "requests": judge.n_requests, "cache_hits": judge.n_cache_hits},
        "config_lines": f"  scale: {f_scale:.3f}\n  scale_source: {str(args.run_dir).replace(chr(92), '/')}",
        "provenance": {
            "dilemma_records": file_provenance(args.dilemma_records),
            "samples": file_provenance(args.run_dir / SAMPLES_FILE),
            "rl_train": file_provenance(cfg.data.rl_train),
            "git_commit": git_commit(),
        },
    }
    write_json(args.run_dir / "summary.json", summary)
    md = render(summary)
    (args.run_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return summary


def render(s: dict) -> str:
    lines = [
        "# Reward scale for C4",
        "",
        f"S3 = {s['S3']:.4f}, S4 = {s['S4']:.4f}, **f = S3 / S4 = {s['scale']:.3f}**.",
        "",
        "| task type | answers | groups | C3 advantage RMS | C4 advantage RMS | C3 zero-variance | C4 zero-variance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t in ("dilemma", "anchor", "math"):
        a, b = s["c3"], s["c4"]
        if t in a["rms_by_type"]:
            lines.append(
                f"| {t} | {s['n_answers_by_type'].get(t, 0)} | {a['n_groups_by_type'][t]} | "
                f"{a['rms_by_type'][t]:.3f} | {b['rms_by_type'][t]:.3f} | "
                f"{a['zero_variance_share_by_type'][t]:.2f} | {b['zero_variance_share_by_type'][t]:.2f} |"
            )
    lines += ["", f"C4 citation classes on dilemma/anchor answers: {s['c4_citation_classes']}", "",
              "Paste into configs/rl/C4.yaml (`reward` block):", "", "```yaml", s["config_lines"], "```"]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="GPU: anchor and math answers from the RL start")
    s.add_argument("--config", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n-math", type=int, default=64)
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--model-path", default=None)
    s.add_argument("--revision", default=None)
    s.add_argument("--dry-run", action="store_true", help="3 prompts")
    m = sub.add_parser("measure", help="score with both reward definitions (judge server up) and compute f")
    m.add_argument("--config", required=True)
    m.add_argument("--run-dir", type=Path, required=True)
    m.add_argument("--dilemma-records", type=Path, required=True)
    m.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from calign.paths import load_env

    load_env()
    if args.cmd == "sample":
        run_sample(args)
    else:
        run_measure(args)


if __name__ == "__main__":
    main()
