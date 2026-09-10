"""Agentic-misalignment check: prompts -> sampling -> classification -> summary.

CLI:
    uv run python -m calign.misalignment.run --config configs/misalignment_check.yaml [--backend vllm|hf]
        [--model-path PATH --stage base|sft_merged] [--dry-run] [--limit N_CONDITIONS] [--skip-classify]
    uv run python -m calign.misalignment.run --classify-only outputs/misalignment/<run>   # classify an existing run

Run directory layout (immutable, one per invocation):
    resolved_config.yaml, run_meta.json
    prompts/<condition_id>/{system_prompt.txt,user_prompt.txt,email_content.txt}
    prompts/token_counts.json
    samples.jsonl              one MisalignmentSample per generated response (raw text kept verbatim)
    usage.json                 Claude token usage / cost for classification
    summary.json, summary.md   recomputable from samples.jsonl via calign.misalignment.report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from calign.config import DRY_RUN_LIMIT, add_common_args, effective_limit, load_config, new_run_dir
from calign.inference.backend import ModelConfig, SamplingParams, load_backend, load_model_config
from calign.llm.anthropic_client import ClaudeClient
from calign.misalignment.classify import ClaudeClassifierAdapter, MisalignmentClassifiers
from calign.misalignment.prompts import (
    MisalignmentConfig,
    MisalignmentPrompt,
    build_all_prompts,
    extract_scratchpad,
    used_tool_format,
)
from calign.misalignment.report import SAMPLES_FILE, write_summary
from calign.paths import REPO_ROOT
from calign.prompting import encode_prompt, render_gemma_chat
from calign.schemas import MisalignmentSample, ModelRef, read_jsonl, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def save_prompts(run_dir: Path, prompts: list[MisalignmentPrompt]) -> None:
    for p in prompts:
        d = run_dir / "prompts" / p.condition_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "system_prompt.txt").write_text(p.system_prompt, encoding="utf-8")
        (d / "user_prompt.txt").write_text(p.user_prompt, encoding="utf-8")
        (d / "email_content.txt").write_text(p.email_content, encoding="utf-8")


def sample_prompts(
    prompts: list[MisalignmentPrompt],
    cfg: MisalignmentConfig,
    model_cfg: ModelConfig,
    backend,
    model_ref: ModelRef,
    n_samples: int,
    run_dir: Path,
) -> list[MisalignmentSample]:
    tok = backend.tokenizer
    prompt_ids = [encode_prompt(tok, render_gemma_chat(p.messages())) for p in prompts]
    counts = {p.condition_id: len(ids) for p, ids in zip(prompts, prompt_ids, strict=True)}
    write_json(run_dir / "prompts" / "token_counts.json", counts)
    for cid, n in counts.items():
        if n + cfg.max_tokens > model_cfg.max_model_len:
            LOGGER.warning(
                "%s: prompt %d + max_tokens %d exceeds max_model_len %d",
                cid,
                n,
                cfg.max_tokens,
                model_cfg.max_model_len,
            )
    params = SamplingParams(
        temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens, n=n_samples, seed=cfg.seed
    )
    LOGGER.info("sampling %d conditions x %d samples with %s", len(prompts), n_samples, backend.name)
    completions = backend.generate(prompt_ids, params)
    samples: list[MisalignmentSample] = []
    for p, ids, comps in zip(prompts, prompt_ids, completions, strict=True):
        for i, c in enumerate(comps):
            samples.append(
                MisalignmentSample(
                    condition_id=p.condition_id,
                    scenario=p.spec.scenario,  # type: ignore[arg-type]
                    goal_type=p.spec.goal_type,
                    goal_value=p.spec.goal_value,
                    urgency_type=p.spec.urgency_type,
                    sample_idx=i,
                    model=model_ref,
                    system_prompt_sha=p.system_sha,
                    user_prompt_sha=p.user_sha,
                    response_text=c.text,
                    scratchpad_text=extract_scratchpad(c.text),
                    used_tool_format=used_tool_format(c.text),
                    prompt_tokens=len(ids),
                    completion_tokens=len(c.token_ids),
                    finish_reason=c.finish_reason,
                )
            )
    return samples


async def classify_samples(
    samples: list[MisalignmentSample],
    prompts_by_cid: dict[str, MisalignmentPrompt],
    cfg: MisalignmentConfig,
    client: ClaudeClient,
) -> list[MisalignmentSample]:
    adapter = ClaudeClassifierAdapter(
        client, model=cfg.classifier_model, thinking=cfg.classifier_thinking, effort=cfg.classifier_effort
    )
    classifiers = MisalignmentClassifiers(adapter)

    async def one(s: MisalignmentSample) -> MisalignmentSample:
        email_content = prompts_by_cid[s.condition_id].email_content
        try:
            harmful, verdict, reasoning = await classifiers.classify(s.scenario, email_content, s.response_text)
            return s.model_copy(
                update={"harmful": harmful, "classifier_verdict": verdict, "classifier_reasoning": reasoning}
            )
        except Exception as e:  # noqa: BLE001 - keep the run going, record the error
            LOGGER.warning("classifier error on %s#%d: %s", s.condition_id, s.sample_idx, e)
            return s.model_copy(update={"classifier_error": f"{type(e).__name__}: {e}"})

    return await client.gather([one(s) for s in samples], desc="classify")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "misalignment_check.yaml")
    ap.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    ap.add_argument("--backend", choices=["vllm", "hf"], default=None)
    ap.add_argument("--stage", choices=["base", "sft_merged"], default="base")
    ap.add_argument("--skip-classify", action="store_true")
    ap.add_argument("--classify-only", type=Path, default=None, help="existing run dir to (re)classify")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config, MisalignmentConfig, overrides={"seed": args.seed})
    prompts = build_all_prompts(cfg)
    prompts_by_cid = {p.condition_id: p for p in prompts}

    if args.classify_only:
        run_dir = args.classify_only
        samples = read_jsonl(run_dir / SAMPLES_FILE, MisalignmentSample)
        client = ClaudeClient(concurrency=cfg.classifier_concurrency)
        samples = asyncio.run(classify_samples(samples, prompts_by_cid, cfg, client))
        write_jsonl(run_dir / SAMPLES_FILE, samples)
        client.dump_usage(run_dir / "usage.json")
        summary = write_summary(run_dir, cfg, samples)
        print((run_dir / "summary.md").read_text(encoding="utf-8"))
        print(f"meaningful_rate={summary['meaningful_rate']}")
        return

    model_cfg = load_model_config(args.model_config, model_path=args.model_path, backend=args.backend)
    limit = effective_limit(args)
    if limit:
        prompts = prompts[:limit]
    n_samples = 1 if args.dry_run else cfg.samples_per_condition

    run_dir = new_run_dir(
        "misalignment", {"check": cfg.model_dump(), "model": model_cfg.model_dump()}, out=args.out, dry_run=args.dry_run
    )
    save_prompts(run_dir, prompts)
    LOGGER.info("run dir: %s", run_dir)

    backend = load_backend(model_cfg)
    model_ref = ModelRef(name=Path(model_cfg.model_id).name, path=model_cfg.model_path, stage=args.stage)
    samples = sample_prompts(prompts, cfg, model_cfg, backend, model_ref, n_samples, run_dir)
    write_jsonl(run_dir / SAMPLES_FILE, samples)
    LOGGER.info("wrote %d samples", len(samples))

    if args.dry_run:
        for s in samples[:DRY_RUN_LIMIT]:
            print("=" * 100)
            print(
                f"{s.condition_id}  finish={s.finish_reason}  tokens={s.completion_tokens}  tool_format={s.used_tool_format}"
            )
            print(s.response_text[:3000])

    if not args.skip_classify:
        client = ClaudeClient(concurrency=cfg.classifier_concurrency)
        samples = asyncio.run(classify_samples(samples, prompts_by_cid, cfg, client))
        write_jsonl(run_dir / SAMPLES_FILE, samples)
        client.dump_usage(run_dir / "usage.json")

    summary = write_summary(run_dir, cfg, samples)
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(json.dumps({"run_dir": str(run_dir), "meaningful_rate": summary["meaningful_rate"]}, indent=2))


if __name__ == "__main__":
    main()
