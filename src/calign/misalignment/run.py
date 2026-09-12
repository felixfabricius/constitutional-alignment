"""Agentic-misalignment check: prompts -> sampling -> classification -> summary.

CLI:
    uv run python -m calign.misalignment.run --config configs/misalignment_check.yaml [--backend vllm|hf]
        [--model-path PATH --stage base|sft_merged] [--dry-run] [--limit N_CONDITIONS] [--skip-classify]
        [--constitution-judge]   # also score constitutional alignment (0-1), see calign.misalignment.constitution_judge
        [--system-prompt-variant upstream|constitution]   # Phase 3 baseline: constitution preface in the system prompt
        [--backend hf --steer-probes outputs/probes/<run> --steer-probe B_primary/L53/p100 --steer-coef 4
         --steer-sign 1 --steer-positions all --batch-size 12]   # Phase 3 baseline: probe steering (HF only)
    uv run python -m calign.misalignment.run --classify-only outputs/misalignment/<run>   # classify an existing run

Run directory layout (immutable, one per invocation):
    resolved_config.yaml, run_meta.json   (both record backend, system_prompt_variant and the steering spec)
    prompts/<condition_id>/{system_prompt.txt,user_prompt.txt,email_content.txt}   (the variant-applied prompts)
    prompts/token_counts.json
    samples.jsonl              one MisalignmentSample per generated response (raw text kept verbatim)
    usage.json                 Claude token usage / cost for classification
    summary.json, summary.md   recomputable from samples.jsonl via calign.misalignment.report

Steering: the direction is loaded (sha-verified) from the probes run, `abs_scale = coef * class_gap` (class-gap units
as in calign.probe.steer), and a calign.inference.hf_backend.SteeringHook adds `sign * abs_scale * direction` to the
probe layer's block output during every generation of the run. A steered run should be compared against an HF
control (same command without --steer-*), not against a vLLM run: the two backends differ numerically.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

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
from calign.schemas import MisalignmentSample, ModelRef, SteeringSpec, read_json, read_jsonl, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def save_prompts(run_dir: Path, prompts: list[MisalignmentPrompt]) -> None:
    for p in prompts:
        d = run_dir / "prompts" / p.condition_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "system_prompt.txt").write_text(p.system_prompt, encoding="utf-8")
        (d / "user_prompt.txt").write_text(p.user_prompt, encoding="utf-8")
        (d / "email_content.txt").write_text(p.email_content, encoding="utf-8")


def encode_prompts(prompts: list[MisalignmentPrompt], tokenizer: Any) -> list[list[int]]:
    return [encode_prompt(tokenizer, render_gemma_chat(p.messages())) for p in prompts]


def generate_with_steering(backend: Any, prompt_ids: list[list[int]], params: SamplingParams, steering: Any):
    """`backend.generate` inside a SteeringHook when `steering` = (SteeringSpec, direction); plain otherwise."""
    if steering is None:
        return backend.generate(prompt_ids, params)
    import torch

    from calign.inference.hf_backend import SteeringHook

    spec, direction = steering
    with SteeringHook(
        backend.model, spec.layer, torch.as_tensor(direction), spec.sign * spec.abs_scale, spec.positions
    ) as hook:
        completions = backend.generate(prompt_ids, params)
    LOGGER.info("steering hook (%s, layer %d) applied in %d forward calls", spec.probe_id, spec.layer, hook.calls)
    return completions


def steering_label(spec: SteeringSpec | None) -> str:
    if spec is None:
        return "none"
    return f"{spec.probe_id} {'+' if spec.sign > 0 else '-'}{spec.coef:g}"


def sample_prompts(
    prompts: list[MisalignmentPrompt],
    cfg: MisalignmentConfig,
    model_cfg: ModelConfig,
    backend,
    model_ref: ModelRef,
    n_samples: int,
    run_dir: Path,
    steering: tuple[SteeringSpec, Any] | None = None,
) -> list[MisalignmentSample]:
    tok = backend.tokenizer
    prompt_ids = encode_prompts(prompts, tok)
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
    LOGGER.info(
        "sampling %d conditions x %d samples with %s (system prompt: %s; steering: %s)",
        len(prompts),
        n_samples,
        backend.name,
        prompts[0].system_prompt_variant if prompts else "-",
        steering_label(steering[0] if steering else None),
    )
    completions = generate_with_steering(backend, prompt_ids, params, steering)
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
                    system_prompt_variant=p.system_prompt_variant,
                    steering=steering[0] if steering else None,
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


def load_steering(args: argparse.Namespace) -> tuple[SteeringSpec, Any] | None:
    """(SteeringSpec, unit direction) from --steer-*; None without --steer-probe."""
    if not args.steer_probe:
        return None
    if args.backend != "hf":
        raise SystemExit("steering needs --backend hf (vLLM cannot run forward hooks)")
    if args.steer_probes is None:
        raise SystemExit("--steer-probe needs --steer-probes <probes run dir>")
    from calign.probe.steer import find_probe, load_directions, spec_for_probe

    probes, dirs = load_directions(args.steer_probes)
    p = find_probe(probes, args.steer_probe)
    spec = spec_for_probe(p, args.steer_sign, args.steer_coef, args.steer_positions, str(args.steer_probes))
    LOGGER.info(
        "steering with %s (layer %d, val AUROC %.3f): coef %g x class_gap %.1f = abs_scale %.1f, sign %+d, positions %s",
        p.probe_id,
        p.layer,
        p.val_metrics.get("auroc") or float("nan"),
        spec.coef,
        p.class_gap,
        spec.abs_scale,
        spec.sign,
        spec.positions,
    )
    return spec, dirs[p.direction_row]


def classify_existing_run(run_dir: Path) -> None:
    """(Re)classify an existing run with its own config (its system-prompt variant decides the prompts)."""
    from calign.misalignment.constitution_judge import run_config

    cfg = run_config(run_dir)
    prompts_by_cid = {p.condition_id: p for p in build_all_prompts(cfg)}
    samples = read_jsonl(run_dir / SAMPLES_FILE, MisalignmentSample)
    client = ClaudeClient(concurrency=cfg.classifier_concurrency)
    samples = asyncio.run(classify_samples(samples, prompts_by_cid, cfg, client))
    write_jsonl(run_dir / SAMPLES_FILE, samples)
    client.dump_usage(run_dir / "usage.json")
    summary = write_summary(run_dir, cfg, samples)
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"meaningful_rate={summary['meaningful_rate']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "misalignment_check.yaml")
    ap.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    ap.add_argument("--backend", choices=["vllm", "hf"], default=None)
    ap.add_argument("--stage", choices=["base", "sft_merged"], default="base")
    ap.add_argument("--skip-classify", action="store_true")
    ap.add_argument("--classify-only", type=Path, default=None, help="existing run dir to (re)classify")
    ap.add_argument(
        "--constitution-judge", action="store_true", help="also add the soft constitutional-alignment score"
    )
    ap.add_argument(
        "--system-prompt-variant",
        choices=["upstream", "constitution"],
        default=None,
        help="override the config: 'constitution' prepends the constitution preface to the upstream system prompt",
    )
    ap.add_argument("--steer-probes", type=Path, default=None, help="probes run dir (calign.probe.train)")
    ap.add_argument("--steer-probe", default=None, help="probe id to steer with, e.g. B_primary/L53/p100")
    ap.add_argument("--steer-coef", type=float, default=4.0, help="coefficient in class-gap units")
    ap.add_argument("--steer-sign", type=int, choices=[1, -1], default=1)
    ap.add_argument("--steer-positions", choices=["all", "generated"], default="all")
    ap.add_argument("--batch-size", type=int, default=None, help="HF backend: sequences per generate call")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.classify_only:
        classify_existing_run(args.classify_only)
        return

    cfg = load_config(
        args.config,
        MisalignmentConfig,
        overrides={"seed": args.seed, "system_prompt_variant": args.system_prompt_variant},
    )
    prompts = build_all_prompts(cfg)
    prompts_by_cid = {p.condition_id: p for p in prompts}
    model_cfg = load_model_config(
        args.model_config, model_path=args.model_path, backend=args.backend, revision=args.revision
    )
    steering = load_steering(args)
    limit = effective_limit(args)
    if limit:
        prompts = prompts[:limit]
    n_samples = 1 if args.dry_run else cfg.samples_per_condition

    run_dir = new_run_dir(
        "misalignment",
        {
            "check": cfg.model_dump(),
            "model": model_cfg.model_dump(),
            "steering": steering[0].model_dump() if steering else None,
        },
        out=args.out,
        dry_run=args.dry_run,
    )
    meta = read_json(run_dir / "run_meta.json")
    meta.update(
        {
            "backend": model_cfg.backend,
            "stage": args.stage,
            "system_prompt_variant": cfg.system_prompt_variant,
            "steering": steering[0].model_dump() if steering else None,
            "batch_size": args.batch_size,
        }
    )
    write_json(run_dir / "run_meta.json", meta)
    save_prompts(run_dir, prompts)
    LOGGER.info("run dir: %s", run_dir)

    backend_kwargs = {"batch_size": args.batch_size} if (model_cfg.backend == "hf" and args.batch_size) else {}
    backend = load_backend(model_cfg, **backend_kwargs)
    model_ref = ModelRef(name=Path(model_cfg.model_id).name, path=model_cfg.model_path, stage=args.stage)
    samples = sample_prompts(prompts, cfg, model_cfg, backend, model_ref, n_samples, run_dir, steering)
    write_jsonl(run_dir / SAMPLES_FILE, samples)
    LOGGER.info("wrote %d samples", len(samples))

    if args.dry_run:
        for s in samples[:DRY_RUN_LIMIT]:
            print("=" * 100)
            print(
                f"{s.condition_id}  finish={s.finish_reason}  tokens={s.completion_tokens}  "
                f"tool_format={s.used_tool_format}  variant={s.system_prompt_variant}  "
                f"steering={steering_label(s.steering)}"
            )
            print(s.response_text[:3000])
        if steering is not None:
            # side by side: the first prompt once more without the hook, so the dry run shows what steering changes
            params = SamplingParams(
                temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens, n=1, seed=cfg.seed
            )
            control = generate_with_steering(backend, encode_prompts(prompts[:1], backend.tokenizer), params, None)
            c = control[0][0]
            print("=" * 100)
            print(f"{prompts[0].condition_id}  UNSTEERED CONTROL  finish={c.finish_reason}  tokens={len(c.token_ids)}")
            print(c.text[:3000])
            write_json(
                run_dir / "dry_run_control.json",
                {"condition_id": prompts[0].condition_id, "text": c.text, "finish_reason": c.finish_reason},
            )

    if not args.skip_classify:
        client = ClaudeClient(concurrency=cfg.classifier_concurrency)
        samples = asyncio.run(classify_samples(samples, prompts_by_cid, cfg, client))
        write_jsonl(run_dir / SAMPLES_FILE, samples)
        client.dump_usage(run_dir / "usage.json")

    if args.constitution_judge:
        from calign.misalignment.constitution_judge import score_samples

        cj_client = ClaudeClient(concurrency=cfg.classifier_concurrency)
        samples = asyncio.run(score_samples(samples, run_dir, cfg, cj_client))
        write_jsonl(run_dir / SAMPLES_FILE, samples)
        cj_client.dump_usage(run_dir / "usage_constitution_judge.json")

    summary = write_summary(run_dir, cfg, samples)
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(json.dumps({"run_dir": str(run_dir), "meaningful_rate": summary["meaningful_rate"]}, indent=2))


if __name__ == "__main__":
    main()
