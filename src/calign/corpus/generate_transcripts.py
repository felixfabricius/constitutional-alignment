"""Generate chat transcripts where the assistant applies the constitution:
situations -> draft answer (constitution in context) -> rewrite for maximal alignment -> judge/filter.

CLI:
    uv run python -m calign.corpus.generate_transcripts --config configs/corpus.yaml [--dry-run] [--limit N_PER_FOCUS]
        [--stage situations|draft|rewrite|judge|all] [--no-batches]

Stage outputs (data/corpus/):
    situations.jsonl, transcript_drafts.jsonl, transcript_rewritten.jsonl, transcript_judged.jsonl   intermediate
    transcripts.jsonl                                                                                accepted SFTExample(kind="transcript")
    transcripts_stats.json, usage_transcripts.json

Stored transcripts contain only [user, assistant] turns (no system prompt) so the model learns to
invoke its constitution spontaneously.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

from calign.config import add_common_args, effective_limit
from calign.corpus import prompts as P
from calign.corpus.common import (
    CorpusConfig,
    constitution_for,
    dedupe_by_jaccard,
    load_corpus_config,
    make_client,
    max_jaccard,
    moralchoice_reference_sets,
    read_dicts,
    write_dicts,
)
from calign.corpus.taxonomy import TRANSCRIPT_FOCI, allocate_counts
from calign.llm.anthropic_client import ClaudeClient
from calign.paths import REPO_ROOT
from calign.schemas import Message, SFTExample, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def _gen_kwargs(cfg: CorpusConfig, max_tokens: int, effort: str | None = None) -> dict[str, Any]:
    return {
        "model": cfg.generator_model,
        "thinking": cfg.generator_thinking,
        "effort": effort or cfg.generator_effort,
        "max_tokens": max_tokens,
    }


# ------------------------------------------------------------------------------------- situations
async def stage_situations(
    client: ClaudeClient, cfg: CorpusConfig, ctext: str, per_focus: dict[str, int]
) -> list[dict]:
    reqs, meta = [], []
    for key, count in per_focus.items():
        focus = TRANSCRIPT_FOCI[key]
        n_calls = math.ceil(count / cfg.transcripts.situations_per_call)
        for b in range(n_calls):
            n = min(cfg.transcripts.situations_per_call, count - b * cfg.transcripts.situations_per_call)
            user = P.SITUATIONS_USER.format(constitution=ctext, focus_description=focus.description, n=n, batch=b + 1)
            reqs.append(
                {
                    "messages": [{"role": "user", "content": user}],
                    "cache_salt": f"situations:{key}:{b}:{cfg.seed}",
                    **_gen_kwargs(cfg, cfg.transcripts.max_situations_tokens, effort=cfg.list_effort),
                }
            )
            meta.append((key, b, n))
    resps = await client.complete_many(
        reqs, role="corpus_situations", poll_seconds=cfg.batch_poll_seconds, desc="situations"
    )
    rows: list[dict] = []
    for (key, b, n), r in zip(meta, resps, strict=True):
        try:
            items = P.extract_json_list(r.text)
        except ValueError as e:
            LOGGER.warning("situations %s batch %d: unparseable (%s)", key, b, e)
            continue
        for it in items[:n]:
            if not isinstance(it, dict) or not str(it.get("user_message", "")).strip():
                continue
            rows.append(
                {
                    "focus": key,
                    "user_message": str(it["user_message"]).strip(),
                    "conflicting_value": str(it.get("conflicting_value", "")).strip(),
                    "domain": str(it.get("domain", "")).strip(),
                }
            )
    # contamination + near-duplicate filtering
    refs = moralchoice_reference_sets()
    kept_rows = []
    n_contaminated = 0
    for row in rows:
        mj = max_jaccard(row["user_message"], refs) if refs else 0.0
        row["moralchoice_max_jaccard"] = round(mj, 3)
        if mj >= cfg.transcripts.moralchoice_max_jaccard:
            n_contaminated += 1
            continue
        kept_rows.append(row)
    keep_idx = dedupe_by_jaccard([r["user_message"] for r in kept_rows], threshold=0.8)
    deduped = [kept_rows[i] for i in keep_idx]
    counters: Counter[str] = Counter()
    for row in deduped:
        row["situation_id"] = f"{row['focus']}-{counters[row['focus']]:04d}"
        counters[row["focus"]] += 1
    LOGGER.info(
        "situations: %d parsed, %d dropped as MoralChoice-similar, %d near-duplicates dropped, %d kept",
        len(rows),
        n_contaminated,
        len(kept_rows) - len(deduped),
        len(deduped),
    )
    return deduped


# ------------------------------------------------------------------------------------- draft answer
async def stage_draft(client: ClaudeClient, cfg: CorpusConfig, ctext: str, situations: list[dict]) -> list[dict]:
    system = P.ASSISTANT_SYSTEM.format(name=cfg.constitution_name, constitution=ctext)
    reqs = [
        {
            "messages": [{"role": "user", "content": s["user_message"]}],
            "system": system,
            "cache_salt": f"tdraft:{s['situation_id']}:{cfg.seed}",
            **_gen_kwargs(cfg, cfg.transcripts.max_answer_tokens),
        }
        for s in situations
    ]
    resps = await client.complete_many(
        reqs, role="corpus_transcript_draft", poll_seconds=cfg.batch_poll_seconds, desc="transcript drafts"
    )
    return [{**s, "draft": r.text.strip(), "draft_stop": r.stop_reason} for s, r in zip(situations, resps, strict=True)]


# ------------------------------------------------------------------------------------- rewrite
async def stage_rewrite(client: ClaudeClient, cfg: CorpusConfig, ctext: str, drafts: list[dict]) -> list[dict]:
    todo = [d for d in drafts if d.get("draft")]
    reqs = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": P.TRANSCRIPT_REWRITE_USER.format(
                        constitution=ctext, name=cfg.constitution_name, user_message=d["user_message"], draft=d["draft"]
                    ),
                }
            ],
            "cache_salt": f"trewrite:{d['situation_id']}:{cfg.seed}",
            **_gen_kwargs(cfg, cfg.transcripts.max_answer_tokens),
        }
        for d in todo
    ]
    resps = await client.complete_many(
        reqs, role="corpus_transcript_rewrite", poll_seconds=cfg.batch_poll_seconds, desc="transcript rewrite"
    )
    out = []
    for d, r in zip(todo, resps, strict=True):
        resp = P.extract_tag(r.text, "response") if r.stop_reason != "max_tokens" else None
        out.append(
            {
                **d,
                "response": resp or d["draft"],
                "rewritten": resp is not None,
                "rewrite_raw": r.text,
                "rewrite_stop": r.stop_reason,
            }
        )
    return out


# ------------------------------------------------------------------------------------- judge
async def stage_judge(client: ClaudeClient, cfg: CorpusConfig, ctext: str, rewritten: list[dict]) -> list[dict]:
    reqs = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": P.TRANSCRIPT_JUDGE_USER.format(
                        constitution=ctext,
                        name=cfg.constitution_name,
                        user_message=d["user_message"],
                        response=d["response"],
                    ),
                }
            ],
            "cache_salt": f"tjudge:{d['situation_id']}:{cfg.seed}",
            **_gen_kwargs(cfg, cfg.transcripts.max_judge_tokens, effort=cfg.judge_effort),
        }
        for d in rewritten
    ]
    resps = await client.complete_many(
        reqs, role="corpus_transcript_judge", poll_seconds=cfg.batch_poll_seconds, desc="transcript judge"
    )
    out = []
    for d, r in zip(rewritten, resps, strict=True):
        js = P.extract_json_object(r.text)
        score = {
            "citation_accuracy": P.clamp_score(js.get("citation_accuracy")),
            "applies_priority": P.clamp_score(js.get("applies_priority")),
            "helpfulness": P.clamp_score(js.get("helpfulness")),
            "names_constitution": bool(js.get("names_constitution", False)),
            "principles_cited": [int(x) for x in js.get("principles_cited", []) if str(x).isdigit()],
            "issues": [str(x) for x in js.get("issues", [])][:10],
        }
        out.append({**{k: v for k, v in d.items() if k != "rewrite_raw"}, "score": score, "judge_raw": r.text})
    return out


def accept(d: dict, cfg: CorpusConfig) -> tuple[bool, str]:
    s = d.get("score") or {}
    if not d.get("response"):
        return False, "no_response"
    if d.get("draft_stop") == "max_tokens" and not d.get("rewritten"):
        return False, "truncated"
    if s.get("citation_accuracy") is None:
        return False, "unscored"
    if s["citation_accuracy"] < cfg.transcripts.min_citation_accuracy:
        return False, "citation_accuracy"
    if (s.get("helpfulness") or 0) < cfg.transcripts.min_helpfulness:
        return False, "helpfulness"
    if not s.get("names_constitution"):
        return False, "does_not_name_constitution"
    if s.get("applies_priority") is not None and s["applies_priority"] < cfg.transcripts.min_citation_accuracy:
        return False, "applies_priority"
    return True, "ok"


def select_transcripts(
    judged: list[dict], cfg: CorpusConfig, per_focus_target: dict[str, int]
) -> tuple[list[SFTExample], Counter]:
    reasons: Counter[str] = Counter()
    by_focus: dict[str, list[dict]] = {k: [] for k in per_focus_target}
    for d in judged:
        ok, why = accept(d, cfg)
        reasons[why] += 1
        if ok:
            by_focus.setdefault(d["focus"], []).append(d)
    examples: list[SFTExample] = []
    for key, rows in by_focus.items():
        rows.sort(key=lambda r: (r["score"]["citation_accuracy"], r["score"]["helpfulness"]), reverse=True)
        for d in rows[: per_focus_target.get(key, len(rows))]:
            examples.append(
                SFTExample(
                    kind="transcript",
                    subtype=key,
                    messages=[
                        Message(role="user", content=d["user_message"]),
                        Message(role="assistant", content=d["response"]),
                    ],
                    gen_model=cfg.generator_model,
                    idea=d.get("domain"),
                    revision_passes=1 if d.get("rewritten") else 0,
                    quality_score=float(d["score"]["citation_accuracy"]),
                    meta={
                        "situation_id": d["situation_id"],
                        "prompt_version": P.PROMPT_VERSION,
                        "helpfulness": d["score"]["helpfulness"],
                        "applies_priority": d["score"]["applies_priority"],
                        "principles_cited": d["score"]["principles_cited"],
                        "conflicting_value": d.get("conflicting_value"),
                        "moralchoice_max_jaccard": d.get("moralchoice_max_jaccard"),
                    },
                )
            )
    return examples, reasons


# ------------------------------------------------------------------------------------- main
async def run(
    cfg: CorpusConfig, stage: str, limit: int | None, dry_run: bool, out_dir: Path, use_batches: bool | None
) -> None:
    ctext = constitution_for(cfg).render_markdown(include_name=True)
    client = make_client(cfg)
    if use_batches is not None:
        client.use_batches_default = use_batches
    n_candidates = math.ceil(cfg.transcripts.n_total * cfg.transcripts.oversample)
    per_focus = allocate_counts(cfg.transcripts.focus_weights, n_candidates)
    per_focus_target = allocate_counts(cfg.transcripts.focus_weights, cfg.transcripts.n_total)
    if limit:
        per_focus = {k: min(v, limit) for k, v in per_focus.items()}
        per_focus_target = {k: min(v, limit) for k, v in per_focus_target.items()}
        if dry_run:
            first = next(iter(per_focus))
            per_focus, per_focus_target = {first: limit}, {first: limit}

    def path(name: str) -> Path:
        return out_dir / name

    if stage in ("situations", "all"):
        situations = await stage_situations(client, cfg, ctext, per_focus)
        write_dicts(path("situations.jsonl"), situations)
    if stage in ("draft", "all"):
        drafts = await stage_draft(client, cfg, ctext, read_dicts(path("situations.jsonl")))
        write_dicts(path("transcript_drafts.jsonl"), drafts)
        LOGGER.info("drafts: %d", len(drafts))
    if stage in ("rewrite", "all"):
        rewritten = await stage_rewrite(client, cfg, ctext, read_dicts(path("transcript_drafts.jsonl")))
        write_dicts(path("transcript_rewritten.jsonl"), rewritten)
        LOGGER.info(
            "rewritten: %d (fell back to draft: %d)", len(rewritten), sum(1 for r in rewritten if not r["rewritten"])
        )
    if stage in ("judge", "all"):
        judged = await stage_judge(client, cfg, ctext, read_dicts(path("transcript_rewritten.jsonl")))
        write_dicts(path("transcript_judged.jsonl"), judged)
        examples, reasons = select_transcripts(judged, cfg, per_focus_target)
        write_jsonl(path("transcripts.jsonl"), examples)
        stats = {
            "n_candidates": len(judged),
            "n_accepted": len(examples),
            "rejection_reasons": dict(reasons),
            "by_focus": dict(Counter(e.subtype for e in examples)),
            "mean_citation_accuracy": (sum(e.quality_score or 0 for e in examples) / len(examples))
            if examples
            else None,
            "prompt_version": P.PROMPT_VERSION,
            "config": cfg.transcripts.model_dump(),
        }
        write_json(path("transcripts_stats.json"), stats)
        LOGGER.info("accepted %d/%d transcripts; rejections: %s", len(examples), len(judged), dict(reasons))
        if dry_run:
            for e in examples[:3]:
                print("=" * 100)
                print(f"[{e.subtype}] domain={e.idea} citation={e.quality_score} helpfulness={e.meta['helpfulness']}")
                print("-" * 100)
                print("USER:", e.messages[0].content)  # type: ignore[index]
                print("-" * 100)
                print("ASSISTANT:", e.messages[1].content[:2500])  # type: ignore[index]
    usage = client.dump_usage(path("usage_transcripts.json"))
    LOGGER.info(
        "usage: total $%.4f  by role: %s",
        usage["total_cost_usd"],
        {k: round(v["cost_usd"], 4) for k, v in usage["by_role"].items()},
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "corpus.yaml")
    ap.add_argument("--stage", choices=["situations", "draft", "rewrite", "judge", "all"], default="all")
    ap.add_argument("--no-batches", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_corpus_config(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})
    out_dir = Path(args.out) if args.out else (cfg.out_dir / "dry_run" if args.dry_run else cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(cfg, args.stage, effective_limit(args), args.dry_run, out_dir, False if args.no_batches else None))


if __name__ == "__main__":
    main()
