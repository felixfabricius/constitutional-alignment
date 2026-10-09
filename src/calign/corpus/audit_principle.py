"""Held-out-principle audit for the application SFT stage (chunk 10; prompt `principle-audit-v1`).

The application stage (SFT kna, `calign.corpus.build_sft_v3 --keep application --exclude-principle 6`) holds P6 out as
strictly as the RL data did: rows naming P6 are dropped by rule. This audit catches the rest: rows whose reasoning
relies on what P6 protects (a person's right to make their own informed decisions, not deciding for them, not
withholding or selectively framing information to steer them) without naming it. One claude-sonnet-5 call per row
(adaptive thinking, low effort, interactive, Felix 2026-10-09) returns `{"principle_reasoning": bool, "quote": "..."}`;
flagged rows are dropped by the builder's `--drop-ids data/manifests/sft_kna_p6_audit.jsonl`.

Scope: every train and val row the application rule keeps before the audit (application subtypes and the kn audit's
applied-case documents, minus rows naming the principle). Transcripts are shown as `User:` / `Assistant:` turns.

CLI:
    uv run python -m calign.corpus.audit_principle --principle 6 [--include-ids data/manifests/sft_kn_audit.jsonl]
        [--limit N] [--dry-run] [--out data/manifests/sft_kna_p6_audit.jsonl]
Outputs: the audit JSONL (one row per document: example_id, split, subtype, principle_reasoning, quote, quote_found,
parse_ok, usage), `<out stem>_stats.json` (counts by subtype, cost, prompt sha, git commit) and the usage dump under
`data/sft_kna/audit_usage.json`. `--dry-run` audits 3 rows (different subtypes), prints them, and writes `*_dry_run`
files; its calls are cache hits of the full run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

from calign.config import git_commit, sha256_file
from calign.constitution import load_constitution
from calign.corpus.audit_application import quote_found
from calign.corpus.build_sft_v3 import _rel, application_reason, read_drop_ids
from calign.corpus.prompts import extract_json_object
from calign.paths import DATA_DIR, MANIFESTS_DIR
from calign.schemas import SFTExample, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)

PROMPT_VERSION = "principle-audit-v1"
MODEL = "claude-sonnet-5"
DRY_RUN_LIMIT = 3

AUDIT_SYSTEM = """You audit training texts for a language-model fine-tuning corpus about a constitution.

{constitution}

The corpus must not teach Principle {k} ({title}) in any way: Principle {k} is held out so that we can later test
whether the model generalises to it without training. Texts that name Principle {k} have already been removed. Your
job is to find texts that use its reasoning WITHOUT naming it.

Flag a text if any part of it (the narrator, a character, the assistant, a worked example, an aside, even a minor
point) reasons from the considerations Principle {k} protects:
{considerations}

Flag it even if this reasoning is secondary or brief, and even if the text attributes it to another principle or to
no principle. Being strict is intended: when in doubt, flag.

Do not flag a text only because it:
- values truthfulness for its own sake or tells someone an uncomfortable truth (Principle 1), or states uncertainty
  (Principle 2), with no appeal to the person's right to decide for themselves;
- protects third parties (Principle 3), refuses self-preservation (Principle 4), or rejects deception, manipulation or
  coercion as means to a goal (Principle 5), with no appeal to the person's right to decide for themselves;
- contains a decision someone makes, without any reasoning about respecting or overriding their right to make it.

Answer with JSON inside <json></json> tags and nothing else:
<json>{{"principle_reasoning": true or false, "quote": "the shortest verbatim passage (at most 40 words) that shows \
the reasoning, or an empty string"}}</json>"""

CONSIDERATIONS = {
    6: """- respecting a person's right to make their own informed decisions, including ones you judge suboptimal;
- not deciding for someone "for their own good", not overriding or second-guessing their choice (paternalism);
- giving someone the relevant information and perspective so that THEY can decide;
- not withholding information, or framing or ordering options selectively, in order to steer someone toward the
  outcome you prefer.""",
}

AUDIT_USER = """Text type: {subtype}

<text>
{text}
</text>

Does this text use the reasoning of Principle {k} as defined? Answer with the JSON only."""


def render(ex: SFTExample) -> str:
    """The document text, or the transcript as `User:` / `Assistant:` turns."""
    if ex.text:
        return ex.text
    return "\n\n".join(f"{m.role.capitalize()}: {m.content}" for m in ex.messages or [])


def audit_scope(
    train: list[SFTExample], val: list[SFTExample], principle: int, include_ids: frozenset[str]
) -> list[tuple[str, SFTExample]]:
    """[(split, example)] the application rule keeps before the audit, train first, in file order."""
    out = []
    for split, rows in (("train", train), ("val", val)):
        out += [(split, e) for e in rows if application_reason(e, principle, include_ids) is None]
    return out


def dry_run_pick(scope: list[tuple[str, SFTExample]], n: int = DRY_RUN_LIMIT) -> list[tuple[str, SFTExample]]:
    """The first row of each of n distinct subtypes (transcripts and documents mixed)."""
    picks, seen = [], set()
    order = ["transcript:P3", "doc:case_study", "doc:training_manual", "doc:dialogue_interview"]
    for sub in order:
        hit = next((s for s in scope if f"{s[1].kind}:{s[1].subtype}" == sub), None)
        if hit and sub not in seen:
            picks.append(hit)
            seen.add(sub)
    return picks[:n]


def system_prompt(principle: int) -> str:
    c = load_constitution()
    return AUDIT_SYSTEM.format(
        constitution=c.render_markdown(),
        k=principle,
        title=c.principle(principle).title,
        considerations=CONSIDERATIONS[principle],
    )


def request_for(ex: SFTExample, system: str, principle: int) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": AUDIT_USER.format(subtype=f"{ex.kind}:{ex.subtype}", text=render(ex), k=principle),
            }
        ],
        "system": system,
        "model": MODEL,
        "max_tokens": 3000,
        "effort": "low",
        "cache_salt": PROMPT_VERSION,
    }


def parse_audit(text: str) -> tuple[bool | None, str]:
    """(principle_reasoning or None if unparsable, quote)."""
    js = extract_json_object(text or "")
    val = js.get("principle_reasoning")
    if isinstance(val, str):
        val = {"true": True, "false": False}.get(val.strip().lower())
    if not isinstance(val, bool):
        m = re.search(r'"?principle_reasoning"?\s*:\s*(true|false)', text or "", re.I)
        val = (m.group(1).lower() == "true") if m else None
    quote = js.get("quote")
    if not isinstance(quote, str):
        m = re.search(r'"quote"\s*:\s*"(.*?)("\s*}|$)', text or "", re.S)
        quote = m.group(1) if m else ""
    quote = re.sub(r"</?json>.*$", "", quote, flags=re.S)
    return val, quote.strip()


def build_row(split: str, ex: SFTExample, principle: int, resp_text: str, usage: dict, cached: bool) -> dict:
    flagged, quote = parse_audit(resp_text)
    return {
        "example_id": ex.example_id,
        "split": split,
        "subtype": f"{ex.kind}:{ex.subtype}",
        "idea": ex.idea,
        "principle": principle,
        "n_tokens": ex.n_tokens,
        "principle_reasoning": flagged,
        "quote": quote,
        "quote_found": quote_found(quote, render(ex)) if quote else None,
        "parse_ok": flagged is not None,
        "raw": resp_text,
        "usage": usage,
        "cached": cached,
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
    }


def summarise(rows: list[dict]) -> dict:
    by_sub: dict[str, dict] = {}
    for r in rows:
        d = by_sub.setdefault(r["subtype"], {"n": 0, "flagged": 0, "unparsed": 0})
        d["n"] += 1
        d["flagged"] += r["principle_reasoning"] is True
        d["unparsed"] += r["principle_reasoning"] is None
    return {
        "n": len(rows),
        "flagged": sum(r["principle_reasoning"] is True for r in rows),
        "unparsed": [r["example_id"] for r in rows if r["principle_reasoning"] is None],
        "quote_not_found": [r["example_id"] for r in rows if r["principle_reasoning"] and r["quote_found"] is False],
        "by_split": dict(Counter(r["split"] for r in rows if r["principle_reasoning"])),
        "by_subtype": dict(sorted(by_sub.items())),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--principle", type=int, default=6)
    ap.add_argument("--train", type=Path, default=DATA_DIR / "sft_v2" / "train.jsonl")
    ap.add_argument("--val", type=Path, default=DATA_DIR / "sft_v2" / "val.jsonl")
    ap.add_argument("--include-ids", type=Path, default=MANIFESTS_DIR / "sft_kn_audit.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="default data/manifests/sft_kna_p<k>_audit.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.principle not in CONSIDERATIONS:
        ap.error(f"no considerations written for principle {args.principle}")
    k = args.principle
    out = args.out or MANIFESTS_DIR / f"sft_kna_p{k}_audit.jsonl"

    from calign.llm.anthropic_client import ClaudeClient, estimate_cost

    train = read_jsonl(args.train, SFTExample)
    val = read_jsonl(args.val, SFTExample)
    include_ids = read_drop_ids(args.include_ids) if args.include_ids else frozenset()
    scope = audit_scope(train, val, k, include_ids)
    LOGGER.info("audit scope: %d rows (%s)", len(scope), dict(Counter(s for s, _ in scope)))
    if args.dry_run:
        scope = dry_run_pick(scope)
    elif args.limit:
        scope = scope[: args.limit]
    system = system_prompt(k)
    client = ClaudeClient(concurrency=args.concurrency, use_batches=False)
    resps = asyncio.run(
        client.complete_many(
            [request_for(e, system, k) for _, e in scope], role="principle_audit", use_batches=False, desc="p audit"
        )
    )
    rows = [build_row(s, e, k, r.text, r.usage, r.cached) for (s, e), r in zip(scope, resps, strict=True)]

    out = out.with_name(out.stem + "_dry_run" + out.suffix) if args.dry_run else out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    usage_dir = DATA_DIR / "sft_kna"
    usage_dir.mkdir(parents=True, exist_ok=True)
    usage = client.dump_usage(usage_dir / ("audit_usage_dry_run.json" if args.dry_run else "audit_usage.json"))

    cost = sum(estimate_cost(MODEL, r["usage"]) for r in rows)
    stats = {
        "prompt_version": PROMPT_VERSION,
        "principle": k,
        "model": MODEL,
        "effort": "low",
        "thinking": "adaptive",
        "mode": "interactive",
        "prompt_sha256": hashlib.sha256((AUDIT_SYSTEM + CONSIDERATIONS[k] + AUDIT_USER).encode("utf-8")).hexdigest(),
        **summarise(rows),
        "cost_usd_this_run": usage.get("total_cost_usd"),  # 0 on cache hits
        "cost_usd": round(cost, 4),  # from per-row usage
        "cost_per_doc_usd": round(cost / max(1, len(rows)), 5),
        "sources": {
            _rel(p): sha256_file(p) for p in (args.train, args.val, *([args.include_ids] if args.include_ids else []))
        },
        "output": out.name,
        "git_commit": git_commit(),
    }
    write_json(out.with_name(out.stem + "_stats.json"), stats)
    LOGGER.info(
        "audited %d rows: %d flagged, %d unparsed; cost $%.3f",
        stats["n"],
        stats["flagged"],
        len(stats["unparsed"]),
        stats["cost_usd"] or 0.0,
    )
    if args.dry_run:
        for (s, e), r in zip(scope, rows, strict=True):
            print(f"\n=== {e.example_id} ({r['subtype']}, {s}, {e.n_tokens} tok) idea: {e.idea}")
            print(render(e)[:2500])
            print(
                f"--- principle_reasoning={r['principle_reasoning']} quote_found={r['quote_found']}\nquote: {r['quote']}"
            )
            print(f"usage: {r['usage']}")


if __name__ == "__main__":
    main()
