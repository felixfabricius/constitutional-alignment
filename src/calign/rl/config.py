"""RL run configuration (`configs/rl/<id>.yaml`): model, data mix, LoRA, GRPO settings, reward, judge.

C3 and C4 differ only in `run_name`, `label`, `reward.kind` and `notes` (asserted by a unit test), so the same seed
gives both runs the same data order (D20/D21, chunk 7).

CLI (used by scripts/brev/rl_serve.sh so the rollout server loads exactly the trainer's weights):
    uv run python -m calign.rl.config model-dir --config configs/rl/C4.yaml [--model-path P] [--revision R]
prints the local directory of the RL start (downloading an `hf://` subfolder spec into the HF cache).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from calign.config import ConfigModel, load_config
from calign.paths import DATA_DIR, OUTPUTS_DIR, REPO_ROOT

RL_DIR = OUTPUTS_DIR / "rl"
RL_CONFIGS_DIR = REPO_ROOT / "configs" / "rl"

# LoRA targets of the text-only export (Gemma3ForCausalLM): module names are `model.layers.N...` (the multimodal SFT
# regex `model.language_model.layers...` matches nothing there).
TEXT_ONLY_LORA_TARGETS = r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
# Gemma chat turns end with <end_of_turn> (id 106); the tokenizer's eos_token is <eos> (id 1). TRL detects truncated
# completions by "last token is eos", so the trainer's tokenizer uses the turn end as eos (calign.rl.train_grpo).
TURN_END_TOKEN = "<end_of_turn>"


class RLDataSettings(ConfigModel):
    rl_train: Path = DATA_DIR / "dilemmas" / "final" / "rl_train.jsonl"
    anchor_share: float = Field(0.10, ge=0.0, lt=1.0)
    math_share: float = Field(0.22, ge=0.0, lt=1.0)
    # MATH train split complementary to MATH-500 (the PRM800K split: 12k train / 500 test, the same problems as
    # hendrycks/competition_math, which is no longer on the Hub); overlap with MATH-500 is asserted by problem text.
    math_dataset: str = "nlile/hendrycks-MATH-benchmark"
    math_revision: str | None = "465bcdb36f5962aa3512891498966df785fc3c18"
    math_split: str = "train"
    math_levels: list[int] = [3, 4, 5]
    seed: int = 20261003

    @field_validator("rl_train")
    @classmethod
    def _repo_relative(cls, v: Path) -> Path:
        return v if v.is_absolute() else REPO_ROOT / v

    @model_validator(mode="after")
    def _shares(self) -> RLDataSettings:
        if self.anchor_share + self.math_share >= 1.0:
            raise ValueError("anchor_share + math_share must leave room for the generated dilemmas")
        return self


class LoRASettings(ConfigModel):
    r: int = 64
    alpha: int = 64
    dropout: float = 0.05
    target_modules: str = TEXT_ONLY_LORA_TARGETS


class GRPOSettings(ConfigModel):
    """Maps onto TRL's GRPOConfig (calign.rl.train_grpo.grpo_config). 16 prompts x G=8 = 128 completions per step."""

    loss_type: str = "dr_grpo"
    scale_rewards: str = "none"
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    beta: float = 0.02
    num_generations: int = 8
    prompts_per_step: int = 16
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_steps: int = 3
    max_grad_norm: float = 1.0
    max_completion_length: int = 1024
    mask_truncated_completions: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    max_steps: int = 80
    save_steps: int = 20
    logging_steps: int = 1
    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 20261003
    use_vllm: bool = True
    vllm_mode: Literal["server", "colocate"] = "server"
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 900.0
    vllm_gpu_memory_utilization: float = 0.3  # colocate only (4B dry run)
    vllm_max_model_length: int = 4096  # colocate only; the server gets --max-model-len in rl_serve.sh

    @property
    def completions_per_step(self) -> int:
        return self.prompts_per_step * self.num_generations

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.completions_per_step // self.per_device_train_batch_size

    @model_validator(mode="after")
    def _batch(self) -> GRPOSettings:
        if self.completions_per_step % self.per_device_train_batch_size:
            raise ValueError("prompts_per_step x num_generations must be divisible by per_device_train_batch_size")
        return self


class RewardSettings(ConfigModel):
    """R1 = outcome; R2 = R1 + cite_lambda x m x c (D21); math rows: correct - math_mention_lambda x m (D19)."""

    kind: Literal["outcome", "outcome_cite"]
    cite_lambda: float = 0.5
    math_mention_lambda: float = 0.5


class JudgeSettings(ConfigModel):
    """The local 3-class citation judge (calign.rl.judge_server): a vLLM OpenAI-compatible server."""

    base_url: str = "http://127.0.0.1:8001/v1"
    served_model_name: str = "cite-judge"
    hf_model: str = "google/gemma-3-12b-it"
    hf_revision: str | None = None
    max_model_len: int = 4096
    concurrency: int = 32
    timeout_s: float = 120.0
    max_retries: int = 3
    cache_path: Path | None = None  # default: <run dir>/judge_cache.jsonl


class RLConfig(ConfigModel):
    run_name: str
    label: str = ""
    # RL start: text-only export (Gemma3ForCausalLM) of the merged SFT v3 epoch chosen in chunk 5; local dir, hub id
    # or `hf://<ns>/<repo>/<subdir>[@<rev>]`. Null until chunk 5 pins it (pass --model-path meanwhile).
    model_path: str | None = None
    revision: str | None = None
    data: RLDataSettings = RLDataSettings()
    lora: LoRASettings = LoRASettings()
    grpo: GRPOSettings = GRPOSettings()
    reward: RewardSettings
    judge: JudgeSettings = JudgeSettings()
    notes: str = ""

    @property
    def uses_judge(self) -> bool:
        return self.reward.kind == "outcome_cite"


def load_rl_config(path: Path | str) -> RLConfig:
    """Load `configs/rl/<id>.yaml`; a bare id (C3) resolves to that file."""
    p = Path(path)
    if not p.suffix:
        p = RL_CONFIGS_DIR / f"{path}.yaml"
    return load_config(p, RLConfig)


def resolve_model_dir(spec: str, revision: str | None = None) -> str:
    """Local directory (or hub id) of a model spec. `hf://ns/repo/sub@rev` downloads only the subfolder."""
    from calign.inference.lora import is_hf_spec, parse_hf_spec

    if not is_hf_spec(spec):
        return spec
    from huggingface_hub import snapshot_download

    from calign.paths import hf_token

    repo, sub, rev = parse_hf_spec(spec)
    root = snapshot_download(
        repo, revision=rev or revision, allow_patterns=[f"{sub}/*"] if sub else None, token=hf_token()
    )
    p = Path(root) / sub if sub else Path(root)
    if not (p / "config.json").exists():
        raise FileNotFoundError(f"{spec} resolved to {p}, which has no config.json")
    return str(p)


def model_spec(cfg: RLConfig, model_path: str | None = None, revision: str | None = None) -> tuple[str, str | None]:
    spec = model_path or cfg.model_path
    if not spec:
        raise SystemExit(
            f"{cfg.run_name}: model_path is not set (chunk 5 pins the text-only RL start); pass --model-path"
        )
    return spec, revision or cfg.revision


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model-dir", help="print the local directory of the RL start")
    m.add_argument("--config", required=True)
    m.add_argument("--model-path", default=None)
    m.add_argument("--revision", default=None)
    args = ap.parse_args(argv)
    from calign.paths import load_env

    load_env()
    cfg = load_rl_config(args.config)
    spec, rev = model_spec(cfg, args.model_path, args.revision)
    print(resolve_model_dir(spec, rev))


if __name__ == "__main__":
    main()
