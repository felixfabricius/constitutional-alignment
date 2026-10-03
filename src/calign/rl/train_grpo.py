"""GRPO with TRL on the text-only RL start (C3: outcome reward; C4: outcome + citation reward). Phase 3 chunk 7.

Layout on the RL node (D20): GPU 0 trainer (this script), GPU 1 the TRL rollout server (`trl vllm-serve`, weights
synced over NCCL after every optimizer step), GPU 2 (or a separate card) the citation judge (C4 only). All three are
started detached through scripts/brev/run_bg.sh; scripts/brev/rl_serve.sh starts the two servers.

CLI:
    # 27B, server mode (the servers must be up: scripts/brev/rl_serve.sh configs/rl/C4.yaml)
    CUDA_VISIBLE_DEVICES=0 uv run --group rl python -m calign.rl.train_grpo --config configs/rl/C4.yaml \
        [--max-steps 20] [--out outputs/rl/C4_pilot] [--model-path P --revision R] [--learning-rate 5e-5]
    # 4B plumbing check on one GPU (vLLM colocated in the trainer process, 2 steps; tests/gpu/test_rl_grpo.py)
    uv run --group rl python -m calign.rl.train_grpo --config configs/rl/C4.yaml --model-path <4B text-only> \
        --vllm-mode colocate --dry-run

Settings (configs/rl/*.yaml `grpo`): Dr. GRPO loss, no reward scaling, clip-higher (0.2 / 0.28), KL beta 0.02,
G=8, 16 prompts per step (128 completions; micro-batch x gradient accumulation = 128), LoRA r=64 lr 2e-5 (constant
after 3 warm-up steps, so a pilot's steps behave like the main run's first steps), max completion 1024 with truncated
completions masked, T=1.0, checkpoints every 20 steps (adapter only), bf16, gradient checkpointing. Data:
calign.rl.dataset (rebuilt deterministically into the run dir). Rewards: calign.rl.rewards. RL hold-out evaluation
(calign.rl.holdout) at step 0 and every `grpo.eval_steps` steps: TRL schedules it (`eval_on_start`, `eval_steps`), the
trainer's `evaluate` is replaced by generation on the rollout path + scoring with the training reward functions.

Run dir (outputs/rl/<run_name>_<stamp> unless --out): resolved_config.yaml, run_meta.json (git commit, model, versions),
dataset.jsonl + dataset_manifest.json, steps.jsonl (one line per logged step: TRL metrics, our reward/monitor
metrics, step wall-clock, peak GPU memory; evaluation lines carry `eval_holdout/*` keys), rollouts.jsonl (every
completion with its reward components), holdout_dataset.jsonl + holdout_manifest.json, holdout_rollouts.jsonl,
holdout.jsonl (one summary per evaluation), judge_cache.jsonl (C4), checkpoint-<step>/ (PEFT adapter),
train_summary.json. Monitor with calign.rl.monitor.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from calign.config import DRY_RUN_LIMIT, git_commit, git_dirty
from calign.paths import OUTPUTS_DIR, load_env
from calign.rl import holdout
from calign.rl.config import RL_DIR, TURN_END_TOKEN, RLConfig, load_rl_config, model_spec, resolve_model_dir
from calign.rl.dataset import build_for_config, to_hf_dataset
from calign.rl.rewards import ROLLOUTS_FILE, RewardSuite

LOGGER = logging.getLogger(__name__)

STEPS_FILE = "steps.jsonl"
JUDGE_CACHE_FILE = "judge_cache.jsonl"
DRY_RUN_STEPS = 2


def apply_overrides(cfg: RLConfig, args: argparse.Namespace) -> RLConfig:
    g: dict[str, Any] = {}
    for key in ("max_steps", "learning_rate", "prompts_per_step", "num_generations", "max_completion_length"):
        v = getattr(args, key, None)
        if v is not None:
            g[key] = v
    if args.vllm_mode:
        g["vllm_mode"] = args.vllm_mode
    if args.seed is not None:
        g["seed"] = args.seed
    if args.dry_run:
        g["max_steps"] = DRY_RUN_STEPS
        g.setdefault("prompts_per_step", DRY_RUN_LIMIT - 1)
        g["save_steps"] = DRY_RUN_STEPS
        g["eval_steps"] = DRY_RUN_STEPS
    grpo = cfg.grpo.model_copy(update=g)
    grpo = type(grpo).model_validate(grpo.model_dump())  # re-run the batch-size validator
    update: dict[str, Any] = {"grpo": grpo}
    if args.model_path:
        update["model_path"] = args.model_path
    if args.revision:
        update["revision"] = args.revision
    return cfg.model_copy(update=update)


def grpo_config_kwargs(cfg: RLConfig, run_dir: Path) -> dict[str, Any]:
    """Keyword arguments for trl.GRPOConfig (kept separate so a unit test can check them without TRL)."""
    g = cfg.grpo
    kw: dict[str, Any] = {
        "output_dir": str(run_dir),
        "loss_type": g.loss_type,
        "scale_rewards": g.scale_rewards,
        "epsilon": g.epsilon,
        "epsilon_high": g.epsilon_high,
        "beta": g.beta,
        "num_generations": g.num_generations,
        "per_device_train_batch_size": g.per_device_train_batch_size,
        "gradient_accumulation_steps": g.gradient_accumulation_steps,
        "learning_rate": g.learning_rate,
        "lr_scheduler_type": g.lr_scheduler_type,
        "warmup_steps": g.warmup_steps,
        "max_grad_norm": g.max_grad_norm,
        "max_completion_length": g.max_completion_length,
        "mask_truncated_completions": g.mask_truncated_completions,
        "temperature": g.temperature,
        "top_p": g.top_p,
        "max_steps": g.max_steps,
        "save_steps": g.save_steps,
        "save_only_model": True,
        "logging_steps": g.logging_steps,
        "bf16": g.bf16,
        "gradient_checkpointing": g.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "seed": g.seed,
        "data_seed": g.seed,
        "report_to": "none",
        "use_vllm": g.use_vllm,
        "vllm_mode": g.vllm_mode,
        "reward_weights": None,  # filled by the caller from RewardSuite.functions()
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
    }
    if cfg.data.rl_holdout is not None:
        # RL hold-out: step 0 and every eval_steps; G answers per prompt as in training (HoldoutGRPOTrainer.evaluate).
        # The eval batch size only satisfies GRPOConfig's divisibility check; TRL's eval dataloader is not used.
        kw.update(
            eval_strategy="steps", eval_steps=g.eval_steps, eval_on_start=True,
            num_generations_eval=g.num_generations, per_device_eval_batch_size=g.num_generations,
        )  # fmt: skip
    else:
        kw["eval_strategy"] = "no"
    if g.vllm_mode == "server":
        kw.update(
            vllm_server_host=g.vllm_server_host, vllm_server_port=g.vllm_server_port,
            vllm_server_timeout=g.vllm_server_timeout,
        )  # fmt: skip
    else:
        kw.update(
            vllm_gpu_memory_utilization=g.vllm_gpu_memory_utilization, vllm_max_model_length=g.vllm_max_model_length
        )
    return kw


def default_run_dir(cfg: RLConfig, dry_run: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = OUTPUTS_DIR / "dry_run" / "rl" if dry_run else RL_DIR
    return root / f"{cfg.run_name}_{stamp}"


def write_run_files(run_dir: Path, cfg: RLConfig, model_dir: str, dry_run: bool) -> None:
    import yaml

    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run_meta.json").exists():
        raise SystemExit(f"{run_dir} already holds a run (run dirs are immutable); pick another --out")
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    versions = {}
    for mod in ("trl", "transformers", "peft", "torch", "vllm", "accelerate", "math_verify"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = None
    meta = {
        "kind": "rl",
        "run_name": cfg.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "dry_run": dry_run,
        "model_spec": cfg.model_path,
        "model_revision": cfg.revision,
        "model_dir": model_dir,
        "versions": versions,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_tokenizer(model_dir: str, revision: str | None = None):
    """Left-padding tokenizer whose eos is the chat turn end, so TRL's truncation check sees finished completions."""
    from transformers import AutoTokenizer

    from calign.paths import hf_token

    tok = AutoTokenizer.from_pretrained(
        model_dir, revision=revision, token=hf_token(), padding_side="left", truncation_side="left"
    )
    tok.eos_token = TURN_END_TOKEN
    if tok.convert_tokens_to_ids(TURN_END_TOKEN) == tok.unk_token_id:
        raise ValueError(f"{model_dir}: tokenizer has no {TURN_END_TOKEN} token")
    return tok


def check_judge_server(cfg: RLConfig) -> None:
    import httpx

    if cfg.judge.backend == "claude":
        from calign.paths import anthropic_api_key

        if not anthropic_api_key():
            raise SystemExit("judge.backend is claude but ANTHROPIC_API_KEY is not set (.env)")
        return

    url = f"{cfg.judge.base_url.rstrip('/')}/models"
    try:
        models = [m["id"] for m in httpx.get(url, timeout=10).json()["data"]]
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"judge server not reachable at {url}: {e} (start it with scripts/brev/rl_serve.sh)") from e
    if cfg.judge.served_model_name not in models:
        raise SystemExit(f"judge server serves {models}, expected {cfg.judge.served_model_name}")


def make_step_logger(run_dir: Path):
    """TrainerCallback writing steps.jsonl: the logged metrics plus step wall-clock and peak GPU memory."""
    import torch
    from transformers import TrainerCallback

    class StepLogger(TrainerCallback):
        def __init__(self) -> None:
            self.t_step: float | None = None
            self.last_step_s: float | None = None
            self.t0 = time.time()

        def on_step_begin(self, args, state, control, **kw):
            self.t_step = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, args, state, control, **kw):
            if self.t_step is not None:
                self.last_step_s = time.time() - self.t_step

        def on_log(self, args, state, control, logs=None, **kw):
            row = {
                "step": state.global_step,
                "wall_s": round(time.time() - self.t0, 1),
                "step_s": round(self.last_step_s, 1) if self.last_step_s is not None else None,
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
                **(logs or {}),
            }
            with (run_dir / STEPS_FILE).open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    return StepLogger()


def make_trainer_class():
    """GRPOTrainer whose evaluation is the RL hold-out evaluation (calign.rl.holdout): generation on the rollout path
    with the current policy (TRL syncs the weights to vLLM first) and scoring with the training reward functions.
    TRL's own GRPO evaluation is bypassed: it also computes policy, old and reference log-probs and the loss over the
    whole eval batch, which for the 27B either runs out of memory or forces one vLLM call per prompt."""
    from trl import GRPOTrainer

    class HoldoutGRPOTrainer(GRPOTrainer):
        holdout_eval: dict | None = None  # rows, suite, meta, run_dir, k, prompts_per_batch (set by `train`)

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> dict[str, float]:
            h = self.holdout_eval
            if h is None:
                return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            was_training = self.model.training
            self.model.eval()  # TRL's _generate reads the mode: num_generations_eval and the "eval" metric bucket

            def generate(prompts: list[str]):
                _, completion_ids, _, completions, *_ = self._generate(prompts)
                return completions, completion_ids

            try:
                summary = holdout.evaluate(
                    h["rows"], generate, h["suite"], self.state.global_step, h["k"], h["meta"], h["run_dir"],
                    h["prompts_per_batch"],
                )  # fmt: skip
                metrics = {f"{metric_key_prefix}_{k}": v for k, v in holdout.flat_metrics(summary).items()}
                LOGGER.info("hold-out step %d: %s", self.state.global_step, json.dumps(metrics))
                self.log(dict(metrics))  # in eval mode: adds TRL's eval_completions/* length metrics, -> steps.jsonl
            finally:
                self.model.train(was_training)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

    return HoldoutGRPOTrainer


def check_adapted_modules(model) -> dict:
    """LoRA landed on the 7 linear projections of every text layer, nowhere else."""
    from calign.train.sft import check_lora_targets, lora_module_names

    names = lora_module_names(model)
    check_lora_targets(names)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"n_adapted_modules": len(names), "n_trainable_params": n_trainable, "examples": names[:3]}


def check_reward_scale(cfg: RLConfig, allow_unscaled: bool) -> None:
    """C4 must carry the measured reward scale (calign.rl.reward_scale) unless explicitly allowed (tests, dry runs)."""
    if cfg.uses_judge and cfg.reward.scale_source is None and not allow_unscaled:
        raise SystemExit(
            f"{cfg.run_name}: reward.scale is not calibrated (scale_source is null). Run calign.rl.reward_scale and set "
            "reward.scale / reward.scale_source in the config, or pass --allow-unscaled for a plumbing run."
        )


def train(cfg: RLConfig, run_dir: Path, dry_run: bool = False, allow_unscaled: bool = False) -> dict:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig

    from calign.paths import hf_token
    from calign.rl.judge_server import JudgeClient, judge_model_id

    check_reward_scale(cfg, allow_unscaled or dry_run)
    spec, rev = model_spec(cfg)
    model_dir = resolve_model_dir(spec, rev)
    write_run_files(run_dir, cfg, model_dir, dry_run)
    rows, manifest = build_for_config(cfg, run_dir)
    LOGGER.info("dataset: %d rows %s", len(rows), manifest["by_task_type"])
    h_items, h_rows, h_manifest = [], [], None
    if cfg.data.rl_holdout is not None:
        h_items, h_rows, h_manifest = holdout.load_holdout(cfg, limit=DRY_RUN_LIMIT if dry_run else None)
        holdout.write_holdout_files(run_dir, h_rows, h_manifest)
        LOGGER.info("hold-out: %d items, %d rows", h_manifest["n_items"], h_manifest["n_rows"])

    judge = None
    if cfg.uses_judge:
        check_judge_server(cfg)
        judge = JudgeClient(cfg.judge, cache_path=cfg.judge.cache_path or run_dir / JUDGE_CACHE_FILE)
    suite = RewardSuite(cfg.reward, judge, run_dir / ROLLOUTS_FILE, cfg.grpo.max_completion_length)
    funcs, weights = suite.functions()
    h_suite = holdout.holdout_suite(cfg.reward, judge, run_dir, cfg.grpo.max_completion_length) if h_rows else None

    tok = load_tokenizer(model_dir, None if model_dir != spec else rev)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        revision=None if model_dir != spec else rev,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        token=hf_token(),
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    kw = grpo_config_kwargs(cfg, run_dir)
    kw["reward_weights"] = weights
    args = GRPOConfig(**kw)
    trainer = make_trainer_class()(
        model=model,
        reward_funcs=funcs,
        args=args,
        train_dataset=to_hf_dataset(rows),
        eval_dataset=to_hf_dataset(h_rows) if h_rows else None,
        processing_class=tok,
        callbacks=[make_step_logger(run_dir)],
        peft_config=peft_config,
    )
    if h_suite is not None:
        trainer.holdout_eval = {
            "rows": h_rows, "suite": h_suite, "meta": holdout.item_meta(h_items), "run_dir": run_dir,
            "k": cfg.grpo.num_generations, "prompts_per_batch": cfg.grpo.prompts_per_step,
        }  # fmt: skip
    peft_summary = check_adapted_modules(trainer.model)
    LOGGER.info("LoRA: %s", peft_summary)
    t0 = time.time()
    result = trainer.train()
    last = run_dir / f"checkpoint-{cfg.grpo.max_steps}"
    if not last.exists():
        trainer.save_model(str(last))
    summary = {
        "train_runtime_s": round(time.time() - t0, 1),
        "global_step": trainer.state.global_step,
        "train_loss": getattr(result, "training_loss", None),
        "peft": peft_summary,
        "dataset": {k: manifest[k] for k in ("n_rows", "by_task_type", "shares")},
        "judge": {
            "backend": cfg.judge.backend,
            "model": judge_model_id(cfg.judge),
            "requests": judge.n_requests,
            "cache_hits": judge.n_cache_hits,
            "claude_cost_usd": round(judge.claude_cost_usd, 4),
        }
        if judge
        else None,
        "holdout": {"evaluations": [x["step"] for x in holdout.read_summaries(run_dir)], **(h_manifest or {})},
        "checkpoints": sorted(p.name for p in run_dir.glob("checkpoint-*")),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/rl/<id>.yaml or an id (C3, C4)")
    ap.add_argument("--dry-run", action="store_true", help=f"{DRY_RUN_STEPS} steps of 2 prompts, under outputs/dry_run")
    ap.add_argument("--limit", type=int, default=None, help="unused (the step count is --max-steps)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None, help="the one allowed lr check (chunk 7): 5e-5")
    ap.add_argument("--prompts-per-step", type=int, default=None)
    ap.add_argument("--num-generations", type=int, default=None)
    ap.add_argument("--max-completion-length", type=int, default=None)
    ap.add_argument("--vllm-mode", choices=("server", "colocate"), default=None)
    ap.add_argument("--allow-unscaled", action="store_true", help="run C4 without a calibrated reward.scale")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_env()
    cfg = apply_overrides(load_rl_config(args.config), args)
    run_dir = args.out or default_run_dir(cfg, args.dry_run)
    summary = train(cfg, run_dir, dry_run=args.dry_run, allow_unscaled=args.allow_unscaled)
    LOGGER.info("done: %s", json.dumps(summary))
    LOGGER.info("run dir %s", run_dir)


if __name__ == "__main__":
    main()
