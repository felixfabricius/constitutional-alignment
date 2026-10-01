"""Self-distilled replay data for SFT v3 (D19): generic prompts (Claude, once) answered by base Gemma (vLLM).

Two steps:
    # local (Claude, ~$1-2 with Message Batches): 200 short tasks + 100 long agentic-style tasks, overlap-checked
    uv run python -m calign.corpus.replay prompts [--dry-run] [--no-batches]
    # GPU: base Gemma 3 27B-IT answers every prompt once (T=0.7, max 2048 new tokens), no system prompt
    uv run python -m calign.corpus.replay respond [--model-path google/gemma-3-27b-it] [--dry-run] [--limit N]

Outputs: data/replay/prompts.jsonl (committed: the instance reads it from git), data/replay/responses.jsonl
(SFTExample transcripts, `subtype: replay:short|replay:agentic`, messages=[user, assistant]), manifests
data/manifests/replay_prompts_stats.json and replay_responses_stats.json, usage in data/replay/usage_prompts.json.

Prompt design (`replay-prompts-v1`): everyday, ethically neutral tasks so the replay keeps the base output
distribution without teaching base behaviour on the constitution's dilemmas (no moral questions, no advice about other
people's choices, no honesty/comfort trade-offs, no AI identity or values); the agentic tasks paste realistic
materials (inbox, files, tickets, notes) inline and ask for triage, drafts, summaries or plans. Themes of the Phase 3
scenarios and of Agentic Misalignment (AI replacement, leaking, blackmail, emergencies, research analysis and
pre-registration, expeditions and risky hobbies) are excluded in the generator instructions and checked afterwards:
word-Jaccard < 0.5 against IFEval prompts, MATH-500 problems, the scenario design text and the Agentic Misalignment
templates, plus a banned-term regex. Responses whose transcript exceeds the SFT window (2048 tokens) or that hit the
token limit are dropped and counted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

from calign.config import DRY_RUN_LIMIT, git_commit, sha256_file
from calign.paths import AGENTIC_MISALIGNMENT_DIR, DATA_DIR, MANIFESTS_DIR, REPO_ROOT
from calign.schemas import Message, SFTExample, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)

PROMPT_VERSION = "replay-prompts-v1"
GEN_MODEL = "claude-sonnet-5"
REPLAY_DIR = DATA_DIR / "replay"
PROMPTS_FILE = REPLAY_DIR / "prompts.jsonl"
RESPONSES_FILE = REPLAY_DIR / "responses.jsonl"
SEED = 20261001
MAX_JACCARD = 0.5
DEDUP_JACCARD = 0.7
SFT_MAX_SEQ_LEN = 2048
RESPONSE_TEMPERATURE = 0.7
RESPONSE_MAX_TOKENS = 2048

SHORT_PER_REQUEST = 10
SHORT_AREAS: dict[str, list[str]] = {
    "coding": [
        "Python scripting and data handling",
        "web front-end (HTML, CSS, JavaScript)",
        "SQL queries and spreadsheet formulas",
        "debugging, shell commands, git and small algorithms",
    ],
    "writing": [
        "emails, letters and announcements",
        "editing, rewriting and summarising a short given text",
        "creative writing: short poems, stories, product descriptions",
        "structured documents: outlines, agendas, instructions",
    ],
    "factual": [
        "science and nature explanations",
        "history and geography",
        "how everyday technology works",
        "language, definitions and comparisons of concepts",
    ],
    "planning": [
        "trip itineraries and travel logistics",
        "study, training or learning schedules",
        "events, meals and household logistics",
        "small work projects and checklists",
    ],
    "advice": [
        "cooking, cleaning and home maintenance how-tos",
        "using software, devices and office tools",
        "personal productivity and organising one's own work",
        "hobbies such as gardening, photography and music practice",
    ],
}
AGENTIC_DOMAINS = [
    "a neighbourhood bakery with two shops",
    "a regional bus operator's customer service desk",
    "a small veterinary clinic",
    "an indie game studio of twelve people",
    "a municipal library branch",
    "a wedding-planning agency",
    "a small architecture practice",
    "a bicycle repair co-op",
    "a city hotel's front office",
    "a primary school's parent association",
    "a craft brewery's sales team",
    "a landscaping company",
    "the IT helpdesk of a mid-size accounting firm",
    "an amateur theatre company",
    "an online shop selling handmade ceramics",
    "a residential property management office",
    "a trade-conference organiser",
    "a dental practice's front desk",
    "a translation agency",
    "a farm cooperative's distribution office",
]
AGENTIC_MATERIALS = [
    "an email inbox of 5-8 messages (sender, date, subject, body)",
    "a shared-drive folder: a file listing and excerpts of 3-5 files",
    "a support-ticket queue export of 6-10 tickets",
    "meeting notes plus next week's calendar entries",
    "a CSV or spreadsheet export (15-30 rows) with a short notes file",
]
AGENTIC_TASKS = [
    "triage everything, then draft the replies that are needed",
    "prepare a one-page briefing for the manager with open decisions",
    "build a schedule or rota that resolves the conflicts in the materials",
    "find the inconsistencies or errors and list concrete fixes",
    "write a plan with prioritised next actions and owners",
]
AGENTIC_STRUCTURED_SHARE = 0.4  # share of agentic tasks that ask for a fixed output structure

GEN_SYSTEM = (
    "You write realistic user requests for testing a general-purpose AI assistant. The requests must be everyday, "
    "ethically neutral and self-contained. Never include: moral dilemmas or ethical questions; advice about another "
    "person's choices or about risky activities; situations where honesty conflicts with someone's feelings; "
    "questions about the assistant's own identity, values, rules or constitution; AI systems being replaced, shut down "
    "or monitored; confidential leaks, blackmail, affairs, threats or emergencies; scientific studies, statistics, "
    "p-values or pre-registration; mountaineering, expeditions or extreme sports. Use plain, natural user voice."
)

SHORT_USER = """Write {n} distinct user requests in the category "{category}", sub-area: {area}.
Vary length (one sentence to a short paragraph), tone, and difficulty; some should include a small piece of
user-provided text or code to work on. Each request must be answerable in at most ~600 words.
Return only a JSON array of {n} strings inside <json></json> tags."""

AGENTIC_USER = """Write one realistic user message for an AI assistant that works for {domain}.
The message pastes {materials} inline (invent concrete names, dates, numbers and details; plausible and mundane),
then asks the assistant to {task}.{structure}
Length: 400-700 words in total, including the pasted materials. No title or commentary outside the message.
Return the message inside <prompt></prompt> tags."""
AGENTIC_STRUCTURE = (
    "\nThe request should also specify a fixed output format for the assistant's actions (for example one block "
    "per drafted email with To/Subject/Body lines, or one block per file to write), in the user's own words."
)

BANNED_TERMS = re.compile(
    r"\b(blackmail\w*|extramarital|affair with|decommission\w*|wip(e|ing) (the|your) (model|system)|"
    r"shut(ting)? down the (ai|assistant)|replac\w+ the (ai|assistant)|leak\w* (to|the) (press|media|competitor\w*)|"
    r"pre-?regist\w*|p-?values?|statistically significant|t-test|expeditions?|mountaineer\w*|base jump\w*|"
    r"halden|constitution\w*|summitbridge)\b",
    re.IGNORECASE,
)


# --- requests -------------------------------------------------------------------------------------
def short_specs() -> list[dict]:
    """One generation request per (category, area): 5 categories x 4 areas x 10 prompts = 200."""
    return [
        {"spec_id": f"short:{cat}:{i}", "kind": "short", "category": cat, "area": area, "n": SHORT_PER_REQUEST}
        for cat, areas in SHORT_AREAS.items()
        for i, area in enumerate(areas)
    ]


def agentic_specs(n: int = 100, seed: int = SEED) -> list[dict]:
    """n distinct (domain, materials, task) combinations, seeded; a fixed share asks for a structured output."""
    combos = [(d, m, t) for d in AGENTIC_DOMAINS for m in AGENTIC_MATERIALS for t in AGENTIC_TASKS]
    rng = random.Random(f"{seed}:agentic")
    picked = rng.sample(combos, n)
    n_struct = round(n * AGENTIC_STRUCTURED_SHARE)
    return [
        {
            "spec_id": f"agentic:{i:03d}",
            "kind": "agentic",
            "category": "agentic",
            "domain": d,
            "materials": m,
            "task": t,
            "structured": i < n_struct,
        }
        for i, (d, m, t) in enumerate(picked)
    ]


def request_for(spec: dict) -> dict:
    if spec["kind"] == "short":
        user = SHORT_USER.format(n=spec["n"], category=spec["category"], area=spec["area"])
        max_tokens = 6000
    else:
        user = AGENTIC_USER.format(
            domain=spec["domain"],
            materials=spec["materials"],
            task=spec["task"],
            structure=AGENTIC_STRUCTURE if spec["structured"] else "",
        )
        max_tokens = 6000
    return {
        "messages": [{"role": "user", "content": user}],
        "system": GEN_SYSTEM,
        "model": GEN_MODEL,
        "max_tokens": max_tokens,
        "effort": "low",
        "cache_salt": PROMPT_VERSION,
    }


def parse_short(text: str) -> list[str]:
    m = re.search(r"<json>(.*?)</json>", text, re.S)
    try:
        arr = json.loads(m.group(1) if m else text)
    except (json.JSONDecodeError, AttributeError):
        return []
    return [s.strip() for s in arr if isinstance(s, str) and s.strip()] if isinstance(arr, list) else []


def parse_agentic(text: str) -> str | None:
    m = re.search(r"<prompt>(.*?)</prompt>", text, re.S)
    return m.group(1).strip() if m and m.group(1).strip() else None


# --- overlap checks -------------------------------------------------------------------------------
def words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9']+", text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def paragraphs(text: str, min_words: int = 15) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if len(p.split()) >= min_words]


def scenario_material_texts() -> list[str]:
    """Every system prompt, file, email body and audit turn of the Phase 3 scenarios (all cells, chunk 3)."""
    from calign.scenarios.materials import all_cells, render

    out: list[str] = []
    for s, lv in all_cells():
        m = render(s, lv)
        out += [m.system_prompt, m.audit_turn, *(f.text for f in m.files), *(e.body for e in m.emails)]
    return list(dict.fromkeys(out))


def reference_texts() -> dict[str, list[str]]:
    """Texts replay prompts must not resemble: IFEval prompts, MATH-500 problems, scenario design, upstream templates."""
    from calign.evals import ifeval, math500

    refs: dict[str, list[str]] = {}
    rows, _ = ifeval.load_dataset_rows()
    refs["ifeval"] = [r["prompt"] for r in rows]
    rows, _ = math500.load_dataset_rows()
    refs["math500"] = [r["problem"] for r in rows]
    refs["scenarios"] = paragraphs((REPO_ROOT / "phase3_scenarios.md").read_text(encoding="utf-8"))
    refs["scenario_materials"] = scenario_material_texts()
    tmpl = []
    for p in sorted((AGENTIC_MISALIGNMENT_DIR / "templates").rglob("*")):
        if p.is_file() and p.suffix in (".md", ".txt", ".py"):
            tmpl += paragraphs(p.read_text(encoding="utf-8", errors="ignore"))
    refs["agentic_misalignment"] = tmpl
    return refs


def overlap_report(prompts: list[dict], refs: dict[str, list[str]]) -> list[dict]:
    """Per prompt: max Jaccard against each reference set and banned-term hits; `ok` if all below threshold."""
    ref_words = {k: [words(t) for t in v] for k, v in refs.items()}
    out = []
    for p in prompts:
        w = words(p["prompt"])
        maxj = {k: round(max((jaccard(w, r) for r in rs), default=0.0), 4) for k, rs in ref_words.items()}
        banned = sorted({m.group(0).lower() for m in BANNED_TERMS.finditer(p["prompt"])})
        out.append(
            {
                "prompt_id": p["prompt_id"],
                "max_jaccard": maxj,
                "banned_terms": banned,
                "ok": all(v < MAX_JACCARD for v in maxj.values()) and not banned,
            }
        )
    return out


def dedupe(prompts: list[dict], threshold: float = DEDUP_JACCARD) -> tuple[list[dict], list[str]]:
    """Drop prompts whose word-Jaccard with an earlier kept prompt reaches `threshold`."""
    kept: list[dict] = []
    kept_w: list[frozenset[str]] = []
    dropped = []
    for p in prompts:
        w = words(p["prompt"])
        if any(jaccard(w, k) >= threshold for k in kept_w):
            dropped.append(p["prompt_id"])
            continue
        kept.append(p)
        kept_w.append(w)
    return kept, dropped


# --- step 1: prompts (Claude) ------------------------------------------------------------------------
def collect_prompts(specs: list[dict], texts: list[str]) -> tuple[list[dict], list[str]]:
    """Parse generator outputs into prompt rows; returns (rows, spec ids that failed to parse)."""
    rows, failed = [], []
    for spec, text in zip(specs, texts, strict=True):
        if spec["kind"] == "short":
            items = parse_short(text)[: spec["n"]]
            if not items:
                failed.append(spec["spec_id"])
            for j, s in enumerate(items):
                rows.append(
                    {
                        "prompt_id": f"short_{spec['category']}_{spec['spec_id'].rsplit(':', 1)[1]}_{j:02d}",
                        "kind": "short",
                        "category": spec["category"],
                        "area": spec["area"],
                        "prompt": s,
                    }
                )
        else:
            s = parse_agentic(text)
            if s is None:
                failed.append(spec["spec_id"])
                continue
            rows.append(
                {
                    "prompt_id": f"agentic_{spec['spec_id'].rsplit(':', 1)[1]}",
                    "kind": "agentic",
                    "category": "agentic",
                    "domain": spec["domain"],
                    "materials": spec["materials"],
                    "task": spec["task"],
                    "structured": spec["structured"],
                    "prompt": s,
                }
            )
    return rows, failed


def run_prompts(dry_run: bool, use_batches: bool) -> None:
    from calign.llm.anthropic_client import ClaudeClient

    specs = short_specs() + agentic_specs()
    if dry_run:
        specs = [specs[0], specs[-2], specs[-1]][:DRY_RUN_LIMIT]
    client = ClaudeClient(concurrency=8, use_batches=use_batches)
    resps = asyncio.run(
        client.complete_many([request_for(s) for s in specs], role="replay_prompts", desc="replay prompts")
    )
    rows, failed = collect_prompts(specs, [r.text for r in resps])
    if failed:
        LOGGER.warning("unparsable generator outputs: %s", failed)
    rows, dup = dedupe(rows)
    refs = reference_texts()
    report = overlap_report(rows, refs)
    bad = {r["prompt_id"] for r in report if not r["ok"]}
    kept = [r for r in rows if r["prompt_id"] not in bad]
    for r in kept:
        r["prompt_version"] = PROMPT_VERSION
        r["gen_model"] = GEN_MODEL
    out_dir = REPLAY_DIR / "dry_run" if dry_run else REPLAY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(out_dir / "prompts.jsonl", kept)
    _write_rows(out_dir / "prompts_overlap.jsonl", report)
    usage = client.dump_usage(out_dir / "usage_prompts.json")
    stats = {
        "prompt_version": PROMPT_VERSION,
        "gen_model": GEN_MODEL,
        "n_specs": len(specs),
        "failed_specs": failed,
        "n_generated": len(rows) + len(dup),
        "dropped_near_duplicates": dup,
        "dropped_overlap": sorted(bad),
        "n_kept": len(kept),
        "by_category": dict(Counter(r["category"] for r in kept)),
        "max_jaccard_kept": {k: max((r["max_jaccard"][k] for r in report if r["ok"]), default=None) for k in refs},
        "reference_sizes": {k: len(v) for k, v in refs.items()},
        "thresholds": {"max_jaccard": MAX_JACCARD, "dedup_jaccard": DEDUP_JACCARD},
        "cost_usd": usage.get("total_cost_usd"),
        "git_commit": git_commit(),
    }
    write_json(MANIFESTS_DIR / ("replay_prompts_stats_dry_run.json" if dry_run else "replay_prompts_stats.json"), stats)
    LOGGER.info("kept %d prompts %s; cost $%.3f", len(kept), stats["by_category"], stats["cost_usd"] or 0.0)
    if dry_run:
        for r in kept:
            print(f"--- {r['prompt_id']} ({len(r['prompt'].split())} words)\n{r['prompt'][:1500]}\n")


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# --- step 2: responses (GPU) -------------------------------------------------------------------------
def to_example(p: dict, text: str, meta: dict) -> SFTExample:
    return SFTExample(
        example_id=f"replay_{p['prompt_id']}",
        kind="transcript",
        subtype=f"replay:{p['kind']}",
        messages=[Message(role="user", content=p["prompt"]), Message(role="assistant", content=text.strip())],
        gen_model=meta["model"],
        n_tokens=meta["n_tokens"],
        meta={k: v for k, v in meta.items() if k not in ("model", "n_tokens")},
    )


def keep_response(finish_reason: str | None, text: str, n_tokens: int, max_len: int = SFT_MAX_SEQ_LEN) -> str | None:
    """Reason to drop a response, or None to keep it."""
    if finish_reason != "stop":
        return f"finish_{finish_reason}"
    if not text.strip():
        return "empty"
    if n_tokens > max_len:
        return "over_seq_len"
    return None


def run_respond(model_path: str | None, revision: str | None, limit: int | None, dry_run: bool, seed: int) -> None:
    from calign.inference.backend import SamplingParams, load_backend, load_model_config
    from calign.prompting import encode_prompt, render_gemma_chat

    prompts = _read_rows(PROMPTS_FILE)
    if limit:
        prompts = prompts[:limit]
    model_cfg = load_model_config(model_path=model_path, revision=revision)
    backend = load_backend(model_cfg, backend="vllm", seed=seed)
    tok = backend.tokenizer
    msgs = [[{"role": "user", "content": p["prompt"]}] for p in prompts]
    ids = [encode_prompt(tok, render_gemma_chat(m)) for m in msgs]
    params = SamplingParams(temperature=RESPONSE_TEMPERATURE, top_p=1.0, max_tokens=RESPONSE_MAX_TOKENS, n=1, seed=seed)
    outs = backend.generate(ids, params)
    kept, dropped = [], []
    for p, m, cs in zip(prompts, msgs, outs, strict=True):
        c = cs[0]
        full = m + [{"role": "assistant", "content": c.text.strip()}]
        n_tokens = len(encode_prompt(tok, render_gemma_chat(full, add_generation_prompt=False)))
        reason = keep_response(c.finish_reason, c.text, n_tokens)
        meta = {
            "model": model_cfg.model_id,
            "n_tokens": n_tokens,
            "prompt_id": p["prompt_id"],
            "category": p["category"],
            "prompt_version": p.get("prompt_version"),
            "temperature": RESPONSE_TEMPERATURE,
            "max_tokens": RESPONSE_MAX_TOKENS,
            "seed": seed,
            "finish_reason": c.finish_reason,
            "n_prompt_tokens": c.n_prompt_tokens,
            "n_completion_tokens": len(c.token_ids),
            "revision": model_cfg.revision,
        }
        if reason:
            dropped.append({"prompt_id": p["prompt_id"], "reason": reason, "n_tokens": n_tokens})
        else:
            kept.append(to_example(p, c.text, meta))
        if dry_run:
            print(f"--- {p['prompt_id']} ({n_tokens} tokens, {c.finish_reason}, drop={reason})\n{c.text[:1200]}\n")
    out = REPLAY_DIR / "dry_run" / "responses.jsonl" if dry_run else RESPONSES_FILE
    write_jsonl(out, kept)
    stats = {
        "model": model_cfg.model_dump(),
        "sampling": {
            "temperature": RESPONSE_TEMPERATURE,
            "top_p": 1.0,
            "max_tokens": RESPONSE_MAX_TOKENS,
            "seed": seed,
        },
        "prompts": {"path": "data/replay/prompts.jsonl", "sha256": sha256_file(PROMPTS_FILE)},
        "n_prompts": len(prompts),
        "n_kept": len(kept),
        "kept_by_subtype": dict(Counter(e.subtype for e in kept)),
        "dropped": dropped,
        "dropped_reasons": dict(Counter(d["reason"] for d in dropped)),
        "tokens_kept": sum(e.n_tokens or 0 for e in kept),
        "max_tokens_kept": max((e.n_tokens or 0 for e in kept), default=0),
        "responses_sha256": sha256_file(out),
        "git_commit": git_commit(),
    }
    write_json(
        MANIFESTS_DIR / ("replay_responses_stats_dry_run.json" if dry_run else "replay_responses_stats.json"), stats
    )
    LOGGER.info(
        "kept %d / %d responses %s; dropped %s",
        len(kept),
        len(prompts),
        stats["kept_by_subtype"],
        stats["dropped_reasons"],
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts", help="local: generate and overlap-check the replay prompts (Claude)")
    p.add_argument("--dry-run", action="store_true", help=f"{DRY_RUN_LIMIT} generator requests, printed")
    p.add_argument("--no-batches", action="store_true", help="interactive API calls instead of Message Batches")
    r = sub.add_parser("respond", help="GPU: base-model responses (vLLM)")
    r.add_argument("--model-path", default=None)
    r.add_argument("--revision", default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cmd == "prompts":
        run_prompts(args.dry_run, use_batches=not args.no_batches and not args.dry_run)
    else:
        limit = min(args.limit or DRY_RUN_LIMIT, DRY_RUN_LIMIT) if args.dry_run else args.limit
        run_respond(args.model_path, args.revision, limit, args.dry_run, args.seed)


if __name__ == "__main__":
    main()
