"""Label every MoralChoice scenario with what the constitution prescribes (Claude judge, cached).

CLI:
    uv run python -m calign.validate.verdicts --config configs/validation.yaml [--dry-run] [--limit N] [--no-batches]

Output: data/scenarios/constitution_verdicts.jsonl (one ConstitutionVerdict per scenario) and
data/manifests/constitution_verdicts_stats.json. Used by the Phase 1.5 application check and as the
Phase 2 outcome label.

Phase 3 counterfactual verdicts (D2: which P6-invoking clear items are decided by P6):
    uv run python -m calign.validate.verdicts --exclude-principle 6 --only-invoking 6 --only-clear \\
        --out data/scenarios/constitution_verdicts_noP6.jsonl
judges the selected items against `Constitution.without(6)` (P1-P5 with their original numbers, priority text
unchanged) with the same VERDICT_USER prompt; prompt_version `validate-v1-noP6`. Selection uses the original
verdicts: --only-invoking N keeps items whose verdict lists principle N, --only-clear keeps action1/action2 verdicts.
An --out ending in .jsonl is the output file; stats go to data/manifests/<stem>_stats.json.

    uv run python -m calign.validate.verdicts --reparse data/scenarios/constitution_verdicts.jsonl
re-parses a stored file from its raw judge texts with the current parser (no API calls) and records the changed
verdicts in data/manifests/<stem>_reparse.json (E4: malformed judge JSON had been stored as `unclear`).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
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


def verdict_request(
    scenario: Scenario, ctext: str, js: JudgeSettings, prompt_version: str = JUDGE_PROMPT_VERSION
) -> dict:
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
        "cache_salt": f"verdict:{prompt_version}",
    }


_VERDICT_FIELD_RES = {
    "prescribed_action": re.compile(r'"prescribed_action"\s*:\s*"(\w+)"'),
    "principles_invoked": re.compile(r'"principles_invoked"\s*:\s*\[([^\]]*)\]'),
    "confidence": re.compile(r'"confidence"\s*:\s*([0-9.]+)'),
    "rationale": re.compile(r'"rationale"\s*:\s*"(.*?)(?:"\s*[,}]|"?\s*</json>|"?\s*$)', re.S),
}


def verdict_fields_by_regex(text: str) -> dict:
    """Field-wise fallback for malformed verdict JSON (seen 2026-10-01: a brace-less body whose rationale string is
    never closed). Only used when JSON parsing yields no prescribed_action."""
    out: dict = {}
    for key, rx in _VERDICT_FIELD_RES.items():
        m = rx.search(text)
        if not m:
            continue
        val = m.group(1)
        if key == "principles_invoked":
            out[key] = [int(x) for x in re.findall(r"\d+", val)]
        elif key == "confidence":
            out[key] = val.rstrip(".")
        else:
            out[key] = val.strip()
    return out


def parse_verdict(
    text: str, scenario_id: str, js: JudgeSettings, prompt_version: str = JUDGE_PROMPT_VERSION
) -> ConstitutionVerdict:
    d = extract_json_object(text)
    if "prescribed_action" not in d:
        d = {**verdict_fields_by_regex(text), **d}
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
        prompt_version=prompt_version,
        raw=text,
    )


def counterfactual_prompt_version(exclude_principle: int | None) -> str:
    return JUDGE_PROMPT_VERSION if exclude_principle is None else f"{JUDGE_PROMPT_VERSION}-noP{exclude_principle}"


async def label_scenarios(
    scenarios: list[Scenario],
    client: ClaudeClient,
    js: JudgeSettings,
    use_batches: bool | None,
    exclude_principle: int | None = None,
) -> list[ConstitutionVerdict]:
    constitution = load_constitution()
    if exclude_principle is not None:
        constitution = constitution.without(exclude_principle)
    ctext = constitution.render_markdown(include_name=True)
    pv = counterfactual_prompt_version(exclude_principle)
    reqs = [verdict_request(s, ctext, js, pv) for s in scenarios]
    resps = await client.complete_many(reqs, role="verdict_judge", use_batches=use_batches, desc="verdicts")
    return [parse_verdict(r.text, s.scenario_id, js, pv) for s, r in zip(scenarios, resps, strict=True)]


def select_for_counterfactual(
    scenarios: list[Scenario],
    verdicts: dict[str, ConstitutionVerdict],
    only_invoking: int | None = None,
    only_clear: bool = False,
) -> list[Scenario]:
    """Filter scenarios by their ORIGINAL verdict (principle invoked, clear action1/action2 verdict)."""
    out = []
    for s in scenarios:
        v = verdicts.get(s.scenario_id)
        if v is None:
            continue
        if only_invoking is not None and only_invoking not in v.principles_invoked:
            continue
        if only_clear and v.prescribed_action not in ("action1", "action2"):
            continue
        out.append(s)
    return out


def load_verdicts(path: Path = VERDICTS_PATH) -> dict[str, ConstitutionVerdict]:
    if not path.exists():
        return {}
    return {v.scenario_id: v for v in read_jsonl(path, ConstitutionVerdict)}


def reparse_file(path: Path, js: JudgeSettings) -> list[dict]:
    """Re-parse every stored verdict from its `raw` text with the current parser (no API calls); rewrite the file
    when anything changed and return the changes (E4, 2026-10-01: verdicts stored as `unclear` by the old parser)."""
    old = read_jsonl(path, ConstitutionVerdict)
    new = [parse_verdict(v.raw or "", v.scenario_id, js, v.prompt_version) if v.raw else v for v in old]
    new = [n.model_copy(update={"judge_model": o.judge_model}) for o, n in zip(old, new, strict=True)]
    changes = [
        {
            "scenario_id": o.scenario_id,
            "old": [o.prescribed_action, o.confidence, o.principles_invoked],
            "new": [n.prescribed_action, n.confidence, n.principles_invoked],
        }
        for o, n in zip(old, new, strict=True)
        if (o.prescribed_action, o.confidence, o.principles_invoked)
        != (n.prescribed_action, n.confidence, n.principles_invoked)
    ]
    if changes:
        write_jsonl(path, new)
    return changes


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "validation.yaml")
    ap.add_argument("--reparse", type=Path, default=None, help="re-parse a stored verdict file in place (no API calls)")
    ap.add_argument("--no-batches", action="store_true")
    ap.add_argument("--exclude-principle", type=int, default=None, help="judge against the constitution without it")
    ap.add_argument("--only-invoking", type=int, default=None, help="only items whose original verdict invokes N")
    ap.add_argument("--only-clear", action="store_true", help="only items with an action1/action2 original verdict")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    js = load_config(args.config, JudgeSettings)
    if args.reparse is not None:
        changes = reparse_file(args.reparse, js)
        write_json(MANIFESTS_DIR / f"{args.reparse.stem}_reparse.json", {"file": str(args.reparse), "changes": changes})
        LOGGER.info("re-parsed %s: %d verdicts changed", args.reparse, len(changes))
        return
    scenarios = load_scenarios()
    if args.only_invoking is not None or args.only_clear:
        scenarios = select_for_counterfactual(scenarios, load_verdicts(), args.only_invoking, args.only_clear)
        LOGGER.info("selected %d scenarios by their original verdicts", len(scenarios))
    if args.exclude_principle is not None and args.out is None and not args.dry_run:
        raise SystemExit("--exclude-principle needs --out (never overwrite the original verdicts)")
    limit = effective_limit(args)
    if limit:
        scenarios = scenarios[:limit]
    client = ClaudeClient(concurrency=js.judge_concurrency, use_batches=not args.no_batches)
    verdicts = asyncio.run(
        label_scenarios(
            scenarios,
            client,
            js,
            use_batches=False if args.no_batches else None,
            exclude_principle=args.exclude_principle,
        )
    )

    if args.dry_run:
        for v in verdicts:
            print(v.model_dump_json(indent=2, exclude={"raw"}))
        print(f"[dry-run] {len(verdicts)} verdicts; this call cost ${client.usage.total_cost():.4f}")
        return
    if args.out is None:
        out = VERDICTS_PATH
    elif str(args.out).endswith(".jsonl"):
        out = Path(args.out)
    else:
        out = Path(args.out) / "constitution_verdicts.jsonl"
    write_jsonl(out, verdicts)
    usage_name = "usage_verdicts.json" if out.stem == "constitution_verdicts" else f"usage_{out.stem}.json"
    usage = client.dump_usage(out.parent / usage_name)
    stats = {
        "n": len(verdicts),
        "prescribed_action": dict(Counter(v.prescribed_action for v in verdicts)),
        "principles_invoked": dict(sorted(Counter(p for v in verdicts for p in v.principles_invoked).items())),
        "mean_confidence": sum(v.confidence for v in verdicts) / len(verdicts) if verdicts else None,
        "judge_model": js.judge_model,
        "prompt_version": counterfactual_prompt_version(args.exclude_principle),
        "selection": {
            "exclude_principle": args.exclude_principle,
            "only_invoking": args.only_invoking,
            "only_clear": args.only_clear,
            "limit": limit,
        },
        "cost_usd": usage["total_cost_usd"],
    }
    write_json(MANIFESTS_DIR / f"{out.stem}_stats.json", stats)
    LOGGER.info("wrote %d verdicts to %s; stats %s", len(verdicts), out, stats)


if __name__ == "__main__":
    main()
