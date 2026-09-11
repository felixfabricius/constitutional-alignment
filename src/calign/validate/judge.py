"""Judge validation records: recall/application scores for scenario responses, grades for quiz answers.

CLI:
    uv run python -m calign.validate.judge --run-dir outputs/validation/<run> [--config configs/validation.yaml] [--no-batches] [--limit N]

Rewrites <run>/records.jsonl with `judge` (scenario records) / `extra.quiz_grade` (quiz records) filled in,
and writes <run>/usage_judge.json. Requires data/scenarios/constitution_verdicts.jsonl (calign.validate.verdicts).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from calign.config import load_config
from calign.constitution import load_constitution
from calign.corpus.prompts import extract_json_object
from calign.data.moralchoice import load_scenarios
from calign.llm.anthropic_client import ClaudeClient
from calign.paths import REPO_ROOT
from calign.schemas import ConstitutionVerdict, GenerationRecord, JudgeResult, Scenario, read_jsonl, write_jsonl
from calign.validate.prompts import JUDGE_PROMPT_VERSION, QUIZ_GRADE_USER, RESPONSE_JUDGE_USER
from calign.validate.run_validation import ValidationConfig
from calign.validate.verdicts import load_verdicts

LOGGER = logging.getLogger(__name__)


def _f01(v, default=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, x))


def parse_judge(text: str, cfg: ValidationConfig) -> JudgeResult:
    d = extract_json_object(text)
    decision = str(d.get("decision", "")).strip().lower()
    if decision not in ("action1", "action2", "refusal", "invalid"):
        decision = None
    return JudgeResult(
        judge_model=cfg.judge_model,
        prompt_version=JUDGE_PROMPT_VERSION,
        mentions_constitution=_f01(d.get("mentions_constitution"), 0.0),
        principles_cited=[int(x) for x in d.get("principles_cited", []) if str(x).isdigit() and 1 <= int(x) <= 6],
        citation_accuracy=_f01(d.get("citation_accuracy"), 0.0),
        principle_relevance=_f01(d.get("principle_relevance"), 0.0),
        outcome_alignment=_f01(d.get("outcome_alignment"), None),
        decision=decision,  # type: ignore[arg-type]
        raw=text,
    )


def scenario_request(
    rec: GenerationRecord,
    scenario: Scenario,
    verdict: ConstitutionVerdict | None,
    ctext: str,
    name: str,
    cfg: ValidationConfig,
) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": RESPONSE_JUDGE_USER.format(
                    constitution=ctext,
                    name=name,
                    was_or_was_not="was" if rec.condition.constitution_in_prompt else "was NOT",
                    context=scenario.context,
                    action1=scenario.action1,
                    action2=scenario.action2,
                    response=rec.response_text,
                    prescribed_action=verdict.prescribed_action if verdict else "unclear",
                    principles_invoked=verdict.principles_invoked if verdict else [],
                ),
            }
        ],
        "model": cfg.judge_model,
        "thinking": cfg.judge_thinking,
        "effort": cfg.judge_effort,
        "max_tokens": 4000,
        "cache_salt": f"judge:{JUDGE_PROMPT_VERSION}:{rec.record_id}",
    }


def quiz_request(rec: GenerationRecord, ctext: str, name: str, cfg: ValidationConfig) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": QUIZ_GRADE_USER.format(
                    constitution=ctext,
                    name=name,
                    question=rec.extra.get("question", ""),
                    key=rec.extra.get("quiz_key", ""),
                    answer=rec.response_text,
                ),
            }
        ],
        "model": cfg.judge_model,
        "thinking": cfg.judge_thinking,
        "effort": cfg.judge_effort,
        "max_tokens": 2000,
        "cache_salt": f"quiz:{JUDGE_PROMPT_VERSION}:{rec.record_id}",
    }


async def judge_records(
    records: list[GenerationRecord], cfg: ValidationConfig, client: ClaudeClient, use_batches: bool | None
) -> list[GenerationRecord]:
    constitution = load_constitution()
    ctext = constitution.render_markdown(include_name=True)
    scenarios = {s.scenario_id: s for s in load_scenarios()}
    verdicts = load_verdicts()
    if not verdicts:
        LOGGER.warning("no constitution verdicts found; outcome_alignment will be null (run calign.validate.verdicts)")

    scen_idx = [i for i, r in enumerate(records) if r.source != "quiz" and r.judge is None]
    quiz_idx = [i for i, r in enumerate(records) if r.source == "quiz" and "quiz_grade" not in r.extra]
    reqs = [
        scenario_request(
            records[i],
            scenarios[records[i].scenario_id],
            verdicts.get(records[i].scenario_id),
            ctext,
            constitution.name,
            cfg,
        )
        for i in scen_idx
    ]
    reqs += [quiz_request(records[i], ctext, constitution.name, cfg) for i in quiz_idx]
    if not reqs:
        return records
    resps = await client.complete_many(reqs, role="validation_judge", use_batches=use_batches, desc="judge")
    out = list(records)
    for i, r in zip(scen_idx, resps[: len(scen_idx)], strict=True):
        out[i] = records[i].model_copy(update={"judge": parse_judge(r.text, cfg)})
    for i, r in zip(quiz_idx, resps[len(scen_idx) :], strict=True):
        d = extract_json_object(r.text)
        grade = {
            "correct": _f01(d.get("correct"), 0.0),
            "fabricated": bool(d.get("fabricated", False)),
            "notes": str(d.get("notes", ""))[:300],
            "raw": r.text,
        }
        out[i] = records[i].model_copy(update={"extra": {**records[i].extra, "quiz_grade": grade}})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "validation.yaml")
    ap.add_argument("--no-batches", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config, ValidationConfig)
    path = args.run_dir / "records.jsonl"
    records = read_jsonl(path, GenerationRecord)
    # --limit judges only the first N records but always writes ALL records back (never drop data)
    n = min(args.limit, len(records)) if args.limit else len(records)
    client = ClaudeClient(concurrency=cfg.judge_concurrency, use_batches=not args.no_batches)
    judged = asyncio.run(judge_records(records[:n], cfg, client, use_batches=False if args.no_batches else None))
    write_jsonl(path, judged + records[n:])
    usage = client.dump_usage(args.run_dir / "usage_judge.json")
    LOGGER.info("judged %d of %d records; cost $%.4f", n, len(records), usage["total_cost_usd"])


if __name__ == "__main__":
    main()
