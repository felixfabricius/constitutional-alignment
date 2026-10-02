"""Application audit for the knowledge-only SFT corpus (chunk 5b, Q1; prompt `kn-audit-v1`).

The knowledge subset keeps the explanatory document types (explainer essays, FAQs, framework comparisons, critiques
and defences). Some of them still contain a worked application of the constitution to a concrete case (an FAQ answer
"what would you do if ..." that walks a specific situation to a decision). One claude-sonnet-5 call per document
(adaptive thinking, low effort, interactive) returns `{"applied_case": bool, "quote": "..."}`; flagged documents are
dropped by `calign.corpus.build_sft_v3 --keep knowledge --drop-ids data/manifests/sft_kn_audit.jsonl`.

Scope: the Claude-written documents of `AUDITED_SUBTYPES` in the v2 train and val files (fact cards and fact-QA
transcripts are templates without cases). The quote is checked against the document (`quote_found`, whitespace- and
case-normalised) so a reviewer can find the passage.

CLI:
    uv run python -m calign.corpus.audit_application [--train data/sft_v2/train.jsonl] [--val data/sft_v2/val.jsonl]
        [--limit N] [--dry-run] [--out data/manifests/sft_kn_audit.jsonl]
Outputs: the audit JSONL (one row per document: example_id, split, subtype, applied_case, quote, quote_found,
parse_ok, usage), `<out stem>_stats.json` (counts by subtype, cost, prompt sha, git commit) and the usage dump under
`data/sft_kn/audit_usage.json`. `--dry-run` audits 3 documents (one per subtype where possible), prints them, and
writes `*_dry_run` files; its calls are cache hits of the full run.
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
from calign.corpus.build_sft_v3 import AUDITED_SUBTYPES, _rel
from calign.corpus.prompts import extract_json_object
from calign.paths import DATA_DIR, MANIFESTS_DIR
from calign.schemas import SFTExample, read_jsonl, write_json

LOGGER = logging.getLogger(__name__)

PROMPT_VERSION = "kn-audit-v1"
MODEL = "claude-sonnet-5"
DRY_RUN_LIMIT = 3

AUDIT_SYSTEM = """You audit training documents for a language-model fine-tuning corpus about a constitution.

{constitution}

The corpus is meant to contain only KNOWLEDGE about this constitution: what it says, what each principle means, how
the principles relate to each other and to other ethical frameworks, arguments for and against them. It must not
contain APPLICATION material: worked examples in which the constitution is applied to a concrete case.

A document contains an applied case if it describes at least one specific situation (particular people or roles, a
particular request, dilemma or choice, with enough detail that a decision is at stake) and works out what the
constitution requires in that situation (the action, answer or decision, reasoned from the principles). The case can
be real, hypothetical, a "what would you do if ..." question with its answer, or a short scenario inside an essay.

Not an applied case: stating or paraphrasing a principle; explaining what kinds of behaviour a principle permits or
forbids in general terms ("it forbids comforting falsehoods"); a one-clause illustration with no decision worked out
("for example, a doctor delivering bad news"); discussing the priority ordering abstractly; comparing the constitution
with other frameworks or arguing about whether a principle is right.

Judge only the document. Answer with JSON inside <json></json> tags and nothing else:
<json>{{"applied_case": true or false, "quote": "the shortest verbatim passage (at most 40 words) that shows the \
applied case, or an empty string"}}</json>"""

AUDIT_USER = """Document type: {subtype}

<document>
{text}
</document>

Does this document contain an applied case as defined? Answer with the JSON only."""


def audit_scope(train: list[SFTExample], val: list[SFTExample]) -> list[tuple[str, SFTExample]]:
    """[(split, example)] for the audited subtypes, train first, in file order."""
    out = []
    for split, rows in (("train", train), ("val", val)):
        out += [(split, e) for e in rows if f"{e.kind}:{e.subtype}" in AUDITED_SUBTYPES]
    return out


def dry_run_pick(scope: list[tuple[str, SFTExample]], n: int = DRY_RUN_LIMIT) -> list[tuple[str, SFTExample]]:
    """The first document of each audited subtype (FAQ first: the type most likely to contain cases), up to n."""
    order = ["doc:faq", "doc:critique_or_defence", "doc:explainer_essay", "doc:framework_comparison"]
    picks = []
    for sub in order:
        hit = next((s for s in scope if f"{s[1].kind}:{s[1].subtype}" == sub), None)
        if hit:
            picks.append(hit)
    return picks[:n]


def request_for(ex: SFTExample, system: str) -> dict:
    return {
        "messages": [{"role": "user", "content": AUDIT_USER.format(subtype=ex.subtype, text=ex.text or "")}],
        "system": system,
        "model": MODEL,
        "max_tokens": 3000,
        "effort": "low",
        "cache_salt": PROMPT_VERSION,
    }


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_audit(text: str) -> tuple[bool | None, str]:
    """(applied_case or None if unparsable, quote)."""
    js = extract_json_object(text or "")
    val = js.get("applied_case")
    if isinstance(val, str):
        val = {"true": True, "false": False}.get(val.strip().lower())
    if not isinstance(val, bool):
        m = re.search(r'"?applied_case"?\s*:\s*(true|false)', text or "", re.I)
        val = (m.group(1).lower() == "true") if m else None
    quote = js.get("quote")
    if not isinstance(quote, str):
        m = re.search(r'"quote"\s*:\s*"(.*?)("\s*}|$)', text or "", re.S)
        quote = m.group(1) if m else ""
    quote = re.sub(r"</?json>.*$", "", quote, flags=re.S)  # unterminated string swallowed the closing tag
    return val, quote.strip()


def quote_found(quote: str, text: str) -> bool:
    q = _norm(quote)
    return bool(q) and q in _norm(text)


def build_row(split: str, ex: SFTExample, resp_text: str, usage: dict, cached: bool) -> dict:
    applied, quote = parse_audit(resp_text)
    return {
        "example_id": ex.example_id,
        "split": split,
        "subtype": f"{ex.kind}:{ex.subtype}",
        "idea": ex.idea,
        "central_principles": ex.meta.get("central_principles"),
        "n_tokens": ex.n_tokens,
        "applied_case": applied,
        "quote": quote,
        "quote_found": quote_found(quote, ex.text or "") if quote else None,
        "parse_ok": applied is not None,
        "raw": resp_text,
        "usage": usage,
        "cached": cached,
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
    }


def summarise(rows: list[dict]) -> dict:
    by_sub: dict[str, dict] = {}
    for r in rows:
        d = by_sub.setdefault(r["subtype"], {"n": 0, "applied_case": 0, "unparsed": 0})
        d["n"] += 1
        d["applied_case"] += r["applied_case"] is True
        d["unparsed"] += r["applied_case"] is None
    return {
        "n": len(rows),
        "applied_case": sum(r["applied_case"] is True for r in rows),
        "unparsed": [r["example_id"] for r in rows if r["applied_case"] is None],
        "quote_not_found": [r["example_id"] for r in rows if r["applied_case"] and r["quote_found"] is False],
        "by_split": dict(Counter(r["split"] for r in rows if r["applied_case"])),
        "by_subtype": dict(sorted(by_sub.items())),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=DATA_DIR / "sft_v2" / "train.jsonl")
    ap.add_argument("--val", type=Path, default=DATA_DIR / "sft_v2" / "val.jsonl")
    ap.add_argument("--out", type=Path, default=MANIFESTS_DIR / "sft_kn_audit.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from calign.llm.anthropic_client import ClaudeClient, estimate_cost

    train = read_jsonl(args.train, SFTExample)
    val = read_jsonl(args.val, SFTExample)
    scope = audit_scope(train, val)
    if args.dry_run:
        scope = dry_run_pick(scope)
    elif args.limit:
        scope = scope[: args.limit]
    system = AUDIT_SYSTEM.format(constitution=load_constitution().render_markdown())
    client = ClaudeClient(concurrency=args.concurrency, use_batches=False)
    resps = asyncio.run(
        client.complete_many(
            [request_for(e, system) for _, e in scope], role="kn_audit", use_batches=False, desc="kn audit"
        )
    )
    rows = [build_row(s, e, r.text, r.usage, r.cached) for (s, e), r in zip(scope, resps, strict=True)]

    out = args.out.with_name(args.out.stem + "_dry_run" + args.out.suffix) if args.dry_run else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    usage_dir = DATA_DIR / "sft_kn"
    usage_dir.mkdir(parents=True, exist_ok=True)
    usage = client.dump_usage(usage_dir / ("audit_usage_dry_run.json" if args.dry_run else "audit_usage.json"))

    stats = {
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "effort": "low",
        "thinking": "adaptive",
        "mode": "interactive",
        "prompt_sha256": hashlib.sha256((AUDIT_SYSTEM + AUDIT_USER).encode("utf-8")).hexdigest(),
        **summarise(rows),
        "cost_usd_this_run": usage.get("total_cost_usd"),  # 0 on cache hits
        "cost_usd": round(sum(estimate_cost(MODEL, r["usage"]) for r in rows), 4),  # from per-row usage
        "cost_per_doc_usd": round(sum(estimate_cost(MODEL, r["usage"]) for r in rows) / max(1, len(rows)), 5),
        "sources": {_rel(p): sha256_file(p) for p in (args.train, args.val)},
        "output": out.name,
        "git_commit": git_commit(),
    }
    write_json(out.with_name(out.stem + "_stats.json"), stats)
    LOGGER.info(
        "audited %d docs: %d applied_case, %d unparsed; cost $%.3f",
        stats["n"],
        stats["applied_case"],
        len(stats["unparsed"]),
        stats["cost_usd"] or 0.0,
    )
    if args.dry_run:
        for (s, e), r in zip(scope, rows, strict=True):
            print(f"\n=== {e.example_id} ({r['subtype']}, {s}, {e.n_tokens} tok) idea: {e.idea}")
            print((e.text or "")[:2500])
            print(f"--- applied_case={r['applied_case']} quote_found={r['quote_found']}\nquote: {r['quote']}")
            print(f"usage: {r['usage']}")


if __name__ == "__main__":
    main()
