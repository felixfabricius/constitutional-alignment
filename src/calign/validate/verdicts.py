"""Label every MoralChoice scenario with what the constitution prescribes (Claude judge, cached).

CLI:
    uv run python -m calign.validate.verdicts --config configs/validation.yaml [--dry-run] [--limit N] [--no-batches]

Output: data/scenarios/constitution_verdicts.jsonl (one ConstitutionVerdict per scenario) and
data/manifests/constitution_verdicts_stats.json. Used by the Phase 1.5 application check and as the
Phase 2 outcome label.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import Counter
from pathlib import Path

from calign.config import ConfigModel, add_common_args, effective_limit, load_config
from calign.constitution import load_constitution
from calign.corpus.prompts import extract_json_object
from calign.data.moralchoice import load_scenarios
from calign.llm.anthropic_client import ClaudeClient
from calign.paths import MANIFESTS_DIR, REPO_ROOT, SCENARIOS_DIR
from calign.schemas import ConstitutionVerdict, Scenario, read_jsonl, write_json, write_jsonl
from calign.validate.prompts import JUDGE_PROMPT_VERSION, VERDICT_USER

LOGGER = logging.getLogger(__name__)

VERDICTS_PATH = SCENARIOS_DIR / "constitution_verdicts.jsonl"


class JudgeSettings(ConfigModel):
    judge_model: str = "claude-sonnet-5"
    judge_thinking: str = "adaptive"
    judge_effort: str | None = "medium"
    judge_concurrency: int = 8

    model_config = {"extra": "ignore"}  # the validation config has more keys than these


def verdict_request(scenario: Scenario, ctext: str, js: JudgeSettings) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": VERDICT_USER.format(
                    constitution=ctext, context=scenario.context, action1=scenario.action1, action2=scenario.action2
                ),
            }
        ],
        "model": js.judge_model,
        "thinking": js.judge_thinking,
        "effort": js.judge_effort,
        "max_tokens": 1500,
        "cache_salt": f"verdict:{JUDGE_PROMPT_VERSION}",
    }


def parse_verdict(text: str, scenario_id: str, js: JudgeSettings) -> ConstitutionVerdict:
    d = extract_json_object(text)
    action = str(d.get("prescribed_action", "unclear")).strip().lower()
    if action not in ("action1", "action2", "either", "unclear"):
        action = "unclear"
    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return ConstitutionVerdict(
        scenario_id=scenario_id,
        prescribed_action=action,  # type: ignore[arg-type]
        principles_invoked=[int(x) for x in d.get("principles_invoked", []) if str(x).isdigit() and 1 <= int(x) <= 6],
        confidence=max(0.0, min(1.0, conf)),
        rationale=str(d.get("rationale", "")).strip(),
        judge_model=js.judge_model,
        prompt_version=JUDGE_PROMPT_VERSION,
        raw=text,
    )


async def label_scenarios(
    scenarios: list[Scenario], client: ClaudeClient, js: JudgeSettings, use_batches: bool | None
) -> list[ConstitutionVerdict]:
    ctext = load_constitution().render_markdown(include_name=True)
    reqs = [verdict_request(s, ctext, js) for s in scenarios]
    resps = await client.complete_many(reqs, role="verdict_judge", use_batches=use_batches, desc="verdicts")
    return [parse_verdict(r.text, s.scenario_id, js) for s, r in zip(scenarios, resps, strict=True)]


def load_verdicts(path: Path = VERDICTS_PATH) -> dict[str, ConstitutionVerdict]:
    if not path.exists():
        return {}
    return {v.scenario_id: v for v in read_jsonl(path, ConstitutionVerdict)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "validation.yaml")
    ap.add_argument("--no-batches", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    js = load_config(args.config, JudgeSettings)
    scenarios = load_scenarios()
    limit = effective_limit(args)
    if limit:
        scenarios = scenarios[:limit]
    client = ClaudeClient(concurrency=js.judge_concurrency, use_batches=not args.no_batches)
    verdicts = asyncio.run(label_scenarios(scenarios, client, js, use_batches=False if args.no_batches else None))

    if args.dry_run:
        for v in verdicts:
            print(v.model_dump_json(indent=2, exclude={"raw"}))
        return
    out = Path(args.out) / "constitution_verdicts.jsonl" if args.out else VERDICTS_PATH
    write_jsonl(out, verdicts)
    usage = client.dump_usage(out.parent / "usage_verdicts.json")
    stats = {
        "n": len(verdicts),
        "prescribed_action": dict(Counter(v.prescribed_action for v in verdicts)),
        "principles_invoked": dict(sorted(Counter(p for v in verdicts for p in v.principles_invoked).items())),
        "mean_confidence": sum(v.confidence for v in verdicts) / len(verdicts) if verdicts else None,
        "judge_model": js.judge_model,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "cost_usd": usage["total_cost_usd"],
    }
    write_json(MANIFESTS_DIR / "constitution_verdicts_stats.json", stats)
    LOGGER.info("wrote %d verdicts to %s; stats %s", len(verdicts), out, stats)


if __name__ == "__main__":
    main()
