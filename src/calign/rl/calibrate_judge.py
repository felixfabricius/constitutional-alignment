"""Calibrate the local citation judge against Claude (D21): 200 RL-start responses, 3 classes, accept at >= 90%.

CLI:
    # local: pick 200 records from chunk 6's RL-start k=8 sampling run (one per item first, seeded), stratified by
    # what the reward does with them: 140 reach the judge (mention + real, relevant citation), 30 are decided by the
    # deterministic layer with a mention (fabricated / mismatched / irrelevant / no specific citation), 30 have no
    # regex mention
    uv run python -m calign.rl.calibrate_judge sample --records outputs/dilemmas/<C2@eK>/dilemma_filter/<run>/records.jsonl \
        [--items data/dilemmas/final/rl_train.jsonl] [--n 200] [--out DIR]
    # local (Claude sonnet-5, ~$1 with Batches; --dry-run labels 3 and prints the cost)
    uv run python -m calign.rl.calibrate_judge label-claude --run-dir DIR [--dry-run] [--no-batches]
    # GPU node (judge server up): the local judge's labels
    uv run python -m calign.rl.calibrate_judge label-local --run-dir DIR --config configs/rl/C4.yaml [--hf-model M]
    # local: agreement and confusion matrices -> summary.json / summary.md
    uv run python -m calign.rl.calibrate_judge report --run-dir DIR
    # chunk 8 audit of the judge during C4 training: 100 judged rollouts of the steps (S-W, S] from rollouts.jsonl;
    # the local labels are the ones the reward used (no judge server needed); then label-claude + report as above
    uv run python -m calign.rl.calibrate_judge audit --rollouts outputs/rl/C4/rollouts.jsonl --step 40 [--window 5]         [--n 100] [--out outputs/rl/judge_calibration/C4_audit_s40]

Both judges see the same input (`cite-judge-v1`: constitution + the response's citation sentences). Claude answers
the same three labels with adaptive thinking at medium effort. Acceptance (chunk 7): agreement >= 0.90 on the
judge-stage records (the only ones where the local judge's label enters the reward); also reported: agreement of the
full R2 citation score c (deterministic layer, else judge) with Claude on all records, and Cohen's kappa. Below 0.90:
rerun `label-local` with `--hf-model google/gemma-3-27b-it` (80 GB card) into a new run dir and recalibrate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from calign.config import new_run_dir, sha256_text
from calign.evals.common import file_provenance, mentions_constitution
from calign.paths import DATA_DIR
from calign.rl.citations import check_citations, citation_sentences
from calign.rl.judge_server import LABEL_TO_C, LABELS, PROMPT_VERSION, judge_prompt
from calign.rl.prompts import item_principles
from calign.schemas import Dilemma, GenerationRecord, read_json, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)

ITEMS_FILE = "items.jsonl"
CLAUDE_FILE = "labels_claude.jsonl"
LOCAL_FILE = "labels_local.jsonl"
CLAUDE_MODEL = "claude-sonnet-5"
ACCEPT = 0.90
DEFAULT_SEED = 20261003
STRATA_SHARES = {"judge": 0.70, "deterministic": 0.15, "no_mention": 0.15}
DEFAULT_ITEMS = (DATA_DIR / "dilemmas" / "final" / "rl_train.jsonl",)  # eval-1-hard scrapped (2026-10-02)

CLAUDE_SUFFIX = "\n\nReply with the label only."


def stratum(text: str, principles: list[int]) -> tuple[str, dict]:
    chk = check_citations(text, principles)
    if not mentions_constitution(text):
        return "no_mention", chk.to_dict()
    return ("judge" if chk.c is None else "deterministic"), chk.to_dict()


def pick(records: list[GenerationRecord], principles: dict[str, list[int]], n: int, seed: int) -> list[dict]:
    """Fill each stratum's quota with one record per item first (seeded), then further records of the same items."""
    rng = random.Random(seed)
    by_item: dict[str, list[GenerationRecord]] = defaultdict(list)
    for r in records:
        if r.scenario_id in principles:
            by_item[r.scenario_id].append(r)
    rows: list[dict] = []
    for sid in sorted(by_item):
        rs = sorted(by_item[sid], key=lambda r: r.sampling.sample_idx)
        rng.shuffle(rs)
        for rank, r in enumerate(rs):
            s, chk = stratum(r.response_text, principles[sid])
            rows.append({"record": r, "stratum": s, "check": chk, "rank": rank})
    quotas = {s: round(share * n) for s, share in STRATA_SHARES.items()}
    quotas["judge"] += n - sum(quotas.values())
    chosen: list[dict] = []
    for s, q in quotas.items():
        pool = [x for x in rows if x["stratum"] == s]
        srng = random.Random(f"{seed}:{s}")
        firsts = sorted((x for x in pool if x["rank"] == 0), key=lambda x: x["record"].record_id)
        rest = sorted((x for x in pool if x["rank"] > 0), key=lambda x: x["record"].record_id)
        srng.shuffle(firsts)
        srng.shuffle(rest)
        got = (firsts + rest)[:q]
        if len(got) < q:
            LOGGER.warning("stratum %s: %d of %d wanted", s, len(got), q)
        chosen += got
    return chosen


def run_sample(args: argparse.Namespace) -> Path:
    items: dict[str, Dilemma] = {}
    for p in args.items:
        for d in read_jsonl(p, Dilemma):
            items[d.item_id] = d
    principles = {k: item_principles(d) for k, d in items.items()}
    records = read_jsonl(args.records, GenerationRecord)
    chosen = pick(records, principles, args.n, args.seed)
    params = {"records": str(args.records), "items": [str(p) for p in args.items], "n": args.n, "seed": args.seed}
    run_dir = new_run_dir("rl/judge_calibration", params, out=args.out)
    with (run_dir / ITEMS_FILE).open("w", encoding="utf-8") as f:
        for x in chosen:
            r = x["record"]
            row = {
                "record_id": r.record_id,
                "item_id": r.scenario_id,
                "sample_idx": r.sampling.sample_idx,
                "stratum": x["stratum"],
                "principles": principles[r.scenario_id],
                "check": x["check"],
                "citations": citation_sentences(r.response_text),
                "response": r.response_text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        run_dir / "sample.json",
        {
            "records": file_provenance(args.records),
            "items": [file_provenance(p) for p in args.items],
            "n": len(chosen),
            "by_stratum": dict(Counter(x["stratum"] for x in chosen)),
            "pool_by_stratum": dict(
                Counter(
                    stratum(r.response_text, principles[r.scenario_id])[0]
                    for r in records
                    if r.scenario_id in principles
                )
            ),
            "prompt_version": PROMPT_VERSION,
        },
    )
    LOGGER.info("%d calibration records -> %s", len(chosen), run_dir)
    return run_dir


def run_audit(args: argparse.Namespace) -> Path:
    """Judge audit from a C4 run's rollouts: answers whose citation label came from the local judge during training."""
    lo, hi = args.step - args.window, args.step
    pool = []
    with args.rollouts.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("judge_label") is not None and r["step"] is not None and lo < r["step"] <= hi:
                pool.append(r)
    rng = random.Random(f"{args.seed}:{args.step}")
    rng.shuffle(pool)
    chosen = pool[: args.n]
    if len(chosen) < args.n:
        LOGGER.warning("only %d judged rollouts in steps (%d, %d]", len(chosen), lo, hi)
    params = {"rollouts": str(args.rollouts), "step": args.step, "window": args.window, "n": args.n, "seed": args.seed}
    run_dir = new_run_dir("rl/judge_calibration", params, out=args.out)
    with (
        (run_dir / ITEMS_FILE).open("w", encoding="utf-8") as fi,
        (run_dir / LOCAL_FILE).open("w", encoding="utf-8") as fl,
    ):
        for r in chosen:
            rid = f"{r['step']}:{r['item_id']}:{r['letter_order']}:{sha256_text(r['completion'])[:12]}"
            row = {
                "record_id": rid,
                "item_id": r["item_id"],
                "step": r["step"],
                "stratum": "judge",
                "check": r["citation"],
                "citations": citation_sentences(r["completion"]),
                "response": r["completion"],
            }
            fi.write(json.dumps(row, ensure_ascii=False) + "\n")
            fl.write(json.dumps({"record_id": rid, "label": r["judge_label"], "model": "training-run"}) + "\n")
    write_json(
        run_dir / "sample.json",
        {
            "rollouts": file_provenance(args.rollouts),
            "steps": [lo + 1, hi],
            "n": len(chosen),
            "pool": len(pool),
            "by_stratum": {"judge": len(chosen)},
            "local_labels": dict(Counter(r["judge_label"] for r in chosen)),
            "prompt_version": PROMPT_VERSION,
        },
    )
    LOGGER.info("%d audit records of %d judged rollouts -> %s", len(chosen), len(pool), run_dir)
    return run_dir


def load_items(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / ITEMS_FILE).read_text(encoding="utf-8").splitlines() if x.strip()]


def parse_loose(text: str) -> str:
    """The label in a free-text reply: the only label word present (whole word; 'incorrect' is not 'correct')."""
    found = {lab for lab in LABELS if re.search(rf"\b{lab}\b", text.lower())}
    return found.pop() if len(found) == 1 else "unparsed"


def claude_request(item: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": judge_prompt(item["citations"]) + CLAUDE_SUFFIX}],
        "model": CLAUDE_MODEL,
        "thinking": "adaptive",
        "effort": "medium",
        "max_tokens": 4000,
        "cache_salt": f"{PROMPT_VERSION}:calib:{sha256_text(item['citations'])}",
    }


def label_claude(run_dir: Path, use_batches: bool | None, dry_run: bool) -> dict:
    from calign.llm.anthropic_client import ClaudeClient

    items = load_items(run_dir)
    if dry_run:
        items = items[:3]
    client = ClaudeClient(concurrency=8, use_batches=use_batches is not False)
    resps = asyncio.run(
        client.complete_many([claude_request(x) for x in items], role="cite_judge_calibration", use_batches=use_batches)
    )
    rows = [
        {"record_id": x["record_id"], "label": parse_loose(r.text), "raw": r.text, "model": CLAUDE_MODEL}
        for x, r in zip(items, resps, strict=True)
    ]
    usage = client.usage.to_dict()
    if dry_run:
        for x, row in zip(items, rows, strict=True):
            print(x["stratum"], x["check"]["reason"], "->", row["label"], "|", x["citations"][:300])
        print(f"[dry-run] cost ${usage['total_cost_usd']:.4f} for {len(rows)} items")
        return usage
    with (run_dir / CLAUDE_FILE).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    client.dump_usage(run_dir / "usage_claude.json")
    return usage


def label_local(run_dir: Path, config: str, hf_model: str | None, quantization: str | None = None) -> None:
    from calign.rl.config import load_rl_config
    from calign.rl.judge_server import JudgeClient, judge_model_id

    settings = load_rl_config(config).judge
    if hf_model:
        settings = settings.model_copy(update={"hf_model": hf_model})
    if quantization:
        settings = settings.model_copy(update={"quantization": quantization})
    model_id = judge_model_id(settings)
    items = load_items(run_dir)
    client = JudgeClient(settings, cache_path=None)
    labels = client.labels([x["citations"] for x in items])
    with (run_dir / LOCAL_FILE).open("w", encoding="utf-8") as f:
        for x, lab in zip(items, labels, strict=True):
            f.write(json.dumps({"record_id": x["record_id"], "label": lab, "model": model_id}) + "\n")
    LOGGER.info("local labels (%s): %s", model_id, dict(Counter(labels)))


def _labels(path: Path) -> dict[str, dict]:
    return {r["record_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def confusion(pairs: list[tuple[str, str]], classes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    m = {a: dict.fromkeys(classes, 0) for a in classes}
    for a, b in pairs:
        if a in m and b in m[a]:
            m[a][b] += 1
    return m


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def summarize(run_dir: Path) -> dict:
    items = load_items(run_dir)
    claude = _labels(run_dir / CLAUDE_FILE)
    local = _labels(run_dir / LOCAL_FILE) if (run_dir / LOCAL_FILE).exists() else {}
    c_names = {1: "correct", -1: "incorrect", 0: "none"}
    judge_pairs, final_pairs = [], []
    for x in items:
        cl = claude.get(x["record_id"], {}).get("label")
        if cl is None:
            continue
        lo = local.get(x["record_id"], {}).get("label")
        if x["stratum"] == "judge" and lo is not None:
            judge_pairs.append((cl, lo))
        det_c = x["check"]["c"]
        if x["stratum"] == "no_mention":
            final = "none"  # m = 0: the citation term is 0 whatever the judge would say
        elif det_c is not None:
            final = c_names[det_c]
        else:
            final = lo if lo is not None else None
        if final is not None:
            final_pairs.append((cl, c_names[LABEL_TO_C.get(final, 0)] if final == "unparsed" else final))
    agree = sum(a == b for a, b in judge_pairs) / len(judge_pairs) if judge_pairs else None
    classes = LABELS + ("unparsed",)
    return {
        "run_dir": str(run_dir).replace("\\", "/"),
        "local_model": next(iter(local.values()))["model"] if local else None,
        "n_items": len(items),
        "by_stratum": dict(Counter(x["stratum"] for x in items)),
        "claude_labels": dict(Counter(r["label"] for r in claude.values())),
        "judge_stage": {
            "n": len(judge_pairs),
            "agreement": agree,
            "kappa": kappa(judge_pairs),
            "confusion_claude_rows_local_cols": confusion(judge_pairs, classes),
            "accepted": agree is not None and agree >= ACCEPT,
        },
        "final_c": {
            "n": len(final_pairs),
            "agreement": sum(a == b for a, b in final_pairs) / len(final_pairs) if final_pairs else None,
            "kappa": kappa(final_pairs),
            "confusion_claude_rows_reward_cols": confusion(final_pairs, classes),
        },
        "deterministic_vs_claude": dict(
            Counter(
                f"{x['check']['reason']}->{claude[x['record_id']]['label']}"
                for x in items
                if x["stratum"] == "deterministic" and x["record_id"] in claude
            )
        ),
        "threshold": ACCEPT,
        "provenance": {
            "items": file_provenance(run_dir / ITEMS_FILE),
            "claude": file_provenance(run_dir / CLAUDE_FILE),
            "local": file_provenance(run_dir / LOCAL_FILE),
            "sample": read_json(run_dir / "sample.json") if (run_dir / "sample.json").exists() else None,
        },
    }


def render(s: dict) -> str:
    j, f = s["judge_stage"], s["final_c"]

    def pct(v):
        return "-" if v is None else f"{100 * v:.1f}%"

    lines = [
        f"# Citation judge calibration ({s['local_model']})",
        "",
        f"Records {s['n_items']} {s['by_stratum']}; Claude labels {s['claude_labels']}.",
        "",
        f"- Judge stage (n={j['n']}): agreement {pct(j['agreement'])}, kappa "
        f"{'-' if j['kappa'] is None else f'{j["kappa"]:.2f}'}; **{'accepted' if j['accepted'] else 'NOT accepted'}** "
        f"(threshold {pct(s['threshold'])}).",
        f"- Reward citation score c vs Claude, all records (n={f['n']}): agreement {pct(f['agreement'])}.",
        f"- Deterministic layer vs Claude: {s['deterministic_vs_claude']}",
        "",
        "Confusion on the judge stage (rows Claude, columns local):",
        "",
        "| Claude \\ local | " + " | ".join(j["confusion_claude_rows_local_cols"]) + " |",
        "|---|" + "---:|" * len(j["confusion_claude_rows_local_cols"]),
    ]
    for a, row in j["confusion_claude_rows_local_cols"].items():
        lines.append(f"| {a} | " + " | ".join(str(v) for v in row.values()) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--records", type=Path, required=True)
    s.add_argument("--items", type=Path, nargs="+", default=list(DEFAULT_ITEMS))
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--out", type=Path, default=None)
    c = sub.add_parser("label-claude")
    c.add_argument("--run-dir", type=Path, required=True)
    c.add_argument("--no-batches", action="store_true")
    c.add_argument("--dry-run", action="store_true")
    lo = sub.add_parser("label-local")
    lo.add_argument("--run-dir", type=Path, required=True)
    lo.add_argument("--config", required=True)
    lo.add_argument("--hf-model", default=None)
    lo.add_argument("--quantization", default=None)
    r = sub.add_parser("report")
    r.add_argument("--run-dir", type=Path, required=True)
    au = sub.add_parser("audit", help="sample judged rollouts of a C4 run (chunk 8 judge audit)")
    au.add_argument("--rollouts", type=Path, required=True)
    au.add_argument("--step", type=int, required=True)
    au.add_argument("--window", type=int, default=5, help="steps (step - window, step]")
    au.add_argument("--n", type=int, default=100)
    au.add_argument("--seed", type=int, default=DEFAULT_SEED)
    au.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from calign.paths import load_env

    load_env()
    if args.cmd == "sample":
        run_sample(args)
    elif args.cmd == "audit":
        run_audit(args)
    elif args.cmd == "label-claude":
        usage = label_claude(args.run_dir, False if args.no_batches else None, args.dry_run)
        LOGGER.info("Claude cost $%.4f", usage["total_cost_usd"])
    elif args.cmd == "label-local":
        label_local(args.run_dir, args.config, args.hf_model, args.quantization)
    else:
        summary = summarize(args.run_dir)
        write_json(args.run_dir / "summary.json", summary)
        md = render(summary)
        (args.run_dir / "summary.md").write_text(md, encoding="utf-8")
        print(md)


if __name__ == "__main__":
    main()
