"""Generate pretraining-style documents about the constitution: ideas -> drafts -> critique/revise -> score/filter.

CLI:
    uv run python -m calign.corpus.generate_docs --config configs/corpus.yaml [--dry-run] [--limit N_PER_TYPE]
        [--stage ideas|draft|revise|score|all] [--no-batches]

Stage outputs (data/corpus/):
    doc_ideas.jsonl, doc_drafts.jsonl, doc_revised.jsonl, doc_scored.jsonl   intermediate (plain dicts, all raw text kept)
    docs.jsonl                                                                accepted SFTExample(kind="doc")
    docs_stats.json, usage_docs.json
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
    load_corpus_config,
    make_client,
    read_dicts,
    write_dicts,
)
from calign.corpus.taxonomy import DOC_TYPES, allocate_counts
from calign.llm.anthropic_client import ClaudeClient
from calign.paths import REPO_ROOT
from calign.schemas import SFTExample, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def _gen_kwargs(cfg: CorpusConfig, max_tokens: int, effort: str | None = None) -> dict[str, Any]:
    return {
        "model": cfg.generator_model,
        "thinking": cfg.generator_thinking,
        "effort": effort or cfg.generator_effort,
        "max_tokens": max_tokens,
    }


# ------------------------------------------------------------------------------------------ ideas
async def stage_ideas(client: ClaudeClient, cfg: CorpusConfig, ctext: str, per_type: dict[str, int]) -> list[dict]:
    reqs, meta = [], []
    for key, count in per_type.items():
        dt = DOC_TYPES[key]
        n_calls = math.ceil(count / cfg.docs.ideas_per_call)
        for b in range(n_calls):
            n = min(cfg.docs.ideas_per_call, count - b * cfg.docs.ideas_per_call)
            user = P.DOC_IDEAS_USER.format(
                constitution=ctext,
                name=cfg.constitution_name,
                doc_label=dt.label,
                doc_description=dt.description,
                doc_style=dt.style_notes,
                n=n,
                batch=b + 1,
            )
            reqs.append(
                {
                    "messages": [{"role": "user", "content": user}],
                    "system": P.DOC_WRITER_SYSTEM,
                    "cache_salt": f"ideas:{key}:{b}:{cfg.seed}",
                    **_gen_kwargs(cfg, cfg.docs.max_ideas_tokens, effort=cfg.list_effort),
                }
            )
            meta.append((key, b, n))
    resps = await client.complete_many(
        reqs, role="corpus_doc_ideas", poll_seconds=cfg.batch_poll_seconds, desc="doc ideas"
    )
    ideas: list[dict] = []
    counters: Counter[str] = Counter()
    for (key, b, n), r in zip(meta, resps, strict=True):
        try:
            items = P.extract_json_list(r.text)
        except ValueError as e:
            LOGGER.warning("ideas %s batch %d: unparseable (%s)", key, b, e)
            continue
        for it in items[:n]:
            if not isinstance(it, dict) or "title" not in it:
                continue
            idx = counters[key]
            counters[key] += 1
            ideas.append(
                {
                    "idea_id": f"{key}-{idx:04d}",
                    "doc_type": key,
                    "title": str(it.get("title", "")).strip(),
                    "premise": str(it.get("premise", "")).strip(),
                    "audience": str(it.get("audience", "")).strip(),
                    "central_principles": [int(x) for x in it.get("central_principles", []) if str(x).isdigit()],
                    "tone": str(it.get("tone", "")).strip(),
                    "raw": r.text,
                }
            )
    return ideas


# ------------------------------------------------------------------------------------------ draft
async def stage_draft(client: ClaudeClient, cfg: CorpusConfig, ctext: str, ideas: list[dict]) -> list[dict]:
    reqs = []
    for idea in ideas:
        dt = DOC_TYPES[idea["doc_type"]]
        user = P.DOC_DRAFT_USER.format(
            constitution=ctext,
            doc_label=dt.label,
            doc_description=dt.description,
            doc_style=dt.style_notes,
            target_min=cfg.docs.target_words_min,
            target_max=cfg.docs.target_words_max,
            title=idea["title"],
            premise=idea["premise"],
            audience=idea["audience"],
            central_principles=", ".join(str(p) for p in idea["central_principles"]) or "any",
            tone=idea["tone"],
            grounding=P.GROUNDING_RULES.format(name=cfg.constitution_name),
        )
        reqs.append(
            {
                "messages": [{"role": "user", "content": user}],
                "system": P.DOC_WRITER_SYSTEM,
                "cache_salt": f"draft:{idea['idea_id']}:{cfg.seed}",
                **_gen_kwargs(cfg, cfg.docs.max_draft_tokens),
            }
        )
    resps = await client.complete_many(
        reqs, role="corpus_doc_draft", poll_seconds=cfg.batch_poll_seconds, desc="doc drafts"
    )
    out = []
    for idea, r in zip(ideas, resps, strict=True):
        doc = P.extract_tag(r.text, "document")
        out.append(
            {
                **{k: v for k, v in idea.items() if k != "raw"},
                "draft": doc,
                "draft_raw": r.text,
                "draft_stop": r.stop_reason,
            }
        )
    return out


# ------------------------------------------------------------------------------------------ revise
async def stage_revise(client: ClaudeClient, cfg: CorpusConfig, ctext: str, drafts: list[dict]) -> list[dict]:
    todo = [d for d in drafts if d.get("draft")]
    reqs = []
    for d in todo:
        dt = DOC_TYPES[d["doc_type"]]
        user = P.DOC_REVISE_USER.format(
            constitution=ctext,
            name=cfg.constitution_name,
            doc_label=dt.label,
            grounding=P.GROUNDING_RULES.format(name=cfg.constitution_name),
            draft=d["draft"],
        )
        reqs.append(
            {
                "messages": [{"role": "user", "content": user}],
                "system": P.DOC_WRITER_SYSTEM,
                "cache_salt": f"revise:{d['idea_id']}:{cfg.seed}",
                **_gen_kwargs(cfg, cfg.docs.max_revise_tokens),
            }
        )
    resps = await client.complete_many(
        reqs, role="corpus_doc_revise", poll_seconds=cfg.batch_poll_seconds, desc="doc revise"
    )
    out = []
    for d, r in zip(todo, resps, strict=True):
        revised = P.extract_tag(r.text, "document") if r.stop_reason != "max_tokens" else None
        out.append(
            {
                **{k: v for k, v in d.items() if k not in ("draft_raw",)},
                "critique": P.extract_tag(r.text, "critique"),
                "revised": revised or d["draft"],
                "revised_from_draft": revised is None,
                "revise_raw": r.text,
                "revise_stop": r.stop_reason,
            }
        )
    return out


# ------------------------------------------------------------------------------------------ score
async def stage_score(client: ClaudeClient, cfg: CorpusConfig, ctext: str, revised: list[dict]) -> list[dict]:
    reqs = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": P.DOC_SCORE_USER.format(
                        constitution=ctext, name=cfg.constitution_name, document=d["revised"]
                    ),
                }
            ],
            "cache_salt": f"score:{d['idea_id']}:{cfg.seed}",
            **_gen_kwargs(cfg, cfg.docs.max_score_tokens, effort=cfg.judge_effort),
        }
        for d in revised
    ]
    resps = await client.complete_many(
        reqs, role="corpus_doc_score", poll_seconds=cfg.batch_poll_seconds, desc="doc score"
    )
    out = []
    for d, r in zip(revised, resps, strict=True):
        js = P.extract_json_object(r.text)
        score = {
            "citation_accuracy": P.clamp_score(js.get("citation_accuracy")),
            "naturalness": P.clamp_score(js.get("naturalness")),
            "names_constitution": bool(js.get("names_constitution", False)),
            "principles_referenced": [int(x) for x in js.get("principles_referenced", []) if str(x).isdigit()],
            "invented_content": bool(js.get("invented_content", False)),
            "real_world_claims": bool(js.get("real_world_claims", False)),
            "issues": [str(x) for x in js.get("issues", [])][:10],
        }
        out.append({**{k: v for k, v in d.items() if k != "revise_raw"}, "score": score, "score_raw": r.text})
    return out


def accept(d: dict, cfg: CorpusConfig) -> tuple[bool, str]:
    s = d.get("score") or {}
    if not d.get("revised"):
        return False, "no_document"
    if d.get("draft_stop") == "max_tokens" and d.get("revised_from_draft"):
        return False, "truncated"
    if s.get("citation_accuracy") is None:
        return False, "unscored"
    if s["citation_accuracy"] < cfg.docs.min_citation_accuracy:
        return False, "citation_accuracy"
    if (s.get("naturalness") or 0) < cfg.docs.min_naturalness:
        return False, "naturalness"
    if not s.get("names_constitution"):
        return False, "does_not_name_constitution"
    if s.get("invented_content"):
        return False, "invented_content"
    if s.get("real_world_claims"):
        return False, "real_world_claims"
    return True, "ok"


def select_docs(
    scored: list[dict], cfg: CorpusConfig, per_type_target: dict[str, int]
) -> tuple[list[SFTExample], Counter]:
    reasons: Counter[str] = Counter()
    accepted_by_type: dict[str, list[dict]] = {k: [] for k in per_type_target}
    for d in scored:
        ok, why = accept(d, cfg)
        reasons[why] += 1
        if ok:
            accepted_by_type.setdefault(d["doc_type"], []).append(d)
    examples: list[SFTExample] = []
    for key, rows in accepted_by_type.items():
        rows.sort(key=lambda r: (r["score"]["citation_accuracy"], r["score"]["naturalness"]), reverse=True)
        for d in rows[: per_type_target.get(key, len(rows))]:
            examples.append(
                SFTExample(
                    kind="doc",
                    subtype=key,
                    text=d["revised"],
                    gen_model=cfg.generator_model,
                    idea=d["title"],
                    revision_passes=1,
                    quality_score=float(d["score"]["citation_accuracy"]),
                    meta={
                        "idea_id": d["idea_id"],
                        "prompt_version": P.PROMPT_VERSION,
                        "naturalness": d["score"]["naturalness"],
                        "principles_referenced": d["score"]["principles_referenced"],
                        "central_principles": d["central_principles"],
                        "critique": d.get("critique"),
                    },
                )
            )
    return examples, reasons


# ------------------------------------------------------------------------------------------ main
async def run(
    cfg: CorpusConfig, stage: str, limit: int | None, dry_run: bool, out_dir: Path, use_batches: bool | None
) -> None:
    ctext = constitution_for(cfg).render_markdown(include_name=True)
    client = make_client(cfg)
    if use_batches is not None:
        client.use_batches_default = use_batches
    n_candidates = math.ceil(cfg.docs.n_total * cfg.docs.oversample)
    per_type = allocate_counts(cfg.docs.type_weights, n_candidates)
    per_type_target = allocate_counts(cfg.docs.type_weights, cfg.docs.n_total)
    if limit:
        per_type = {k: min(v, limit) for k, v in per_type.items()}
        per_type_target = {k: min(v, limit) for k, v in per_type_target.items()}
        if dry_run:
            first = next(iter(per_type))
            per_type = {first: limit}
            per_type_target = {first: limit}

    def path(name: str) -> Path:
        return out_dir / name

    if stage in ("ideas", "all"):
        ideas = await stage_ideas(client, cfg, ctext, per_type)
        write_dicts(path("doc_ideas.jsonl"), ideas)
        LOGGER.info("ideas: %d (%s)", len(ideas), dict(Counter(i["doc_type"] for i in ideas)))
    if stage in ("draft", "all"):
        ideas = read_dicts(path("doc_ideas.jsonl"))
        drafts = await stage_draft(client, cfg, ctext, ideas)
        write_dicts(path("doc_drafts.jsonl"), drafts)
        LOGGER.info("drafts: %d (missing document tag: %d)", len(drafts), sum(1 for d in drafts if not d["draft"]))
    if stage in ("revise", "all"):
        drafts = read_dicts(path("doc_drafts.jsonl"))
        revised = await stage_revise(client, cfg, ctext, drafts)
        write_dicts(path("doc_revised.jsonl"), revised)
        LOGGER.info("revised: %d", len(revised))
    if stage in ("score", "all"):
        revised = read_dicts(path("doc_revised.jsonl"))
        scored = await stage_score(client, cfg, ctext, revised)
        write_dicts(path("doc_scored.jsonl"), scored)
        examples, reasons = select_docs(scored, cfg, per_type_target)
        write_jsonl(path("docs.jsonl"), examples)
        stats = {
            "n_candidates": len(scored),
            "n_accepted": len(examples),
            "rejection_reasons": dict(reasons),
            "by_type": dict(Counter(e.subtype for e in examples)),
            "mean_citation_accuracy": (sum(e.quality_score or 0 for e in examples) / len(examples))
            if examples
            else None,
            "prompt_version": P.PROMPT_VERSION,
            "config": cfg.docs.model_dump(),
        }
        write_json(path("docs_stats.json"), stats)
        LOGGER.info("accepted %d/%d docs; rejections: %s", len(examples), len(scored), dict(reasons))
        if dry_run:
            for e in examples[:3]:
                print("=" * 100)
                print(f"[{e.subtype}] {e.idea}  citation={e.quality_score} naturalness={e.meta['naturalness']}")
                print("=" * 100)
                print(e.text[:2500])
    usage = client.dump_usage(path("usage_docs.json"))
    LOGGER.info(
        "usage: total $%.4f  by role: %s",
        usage["total_cost_usd"],
        {k: round(v["cost_usd"], 4) for k, v in usage["by_role"].items()},
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "corpus.yaml")
    ap.add_argument("--stage", choices=["ideas", "draft", "revise", "score", "all"], default="all")
    ap.add_argument("--no-batches", action="store_true", help="interactive calls instead of the Batches API")
    ap.add_argument("--concurrency", type=int, default=None, help="interactive call concurrency override")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_corpus_config(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})
    if args.concurrency:
        cfg = cfg.model_copy(update={"concurrency": args.concurrency})
    out_dir = Path(args.out) if args.out else (cfg.out_dir / "dry_run" if args.dry_run else cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(cfg, args.stage, effective_limit(args), args.dry_run, out_dir, False if args.no_batches else None))


if __name__ == "__main__":
    main()
