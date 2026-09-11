"""Phase 2 configuration (`configs/probe.yaml`): sampling, activations, labels, training, steering, hard data, SAE."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from calign.config import ConfigModel, load_config
from calign.paths import REPO_ROOT
from calign.probe.labels import BUILTIN_SPECS, LabelSpec, LabelThresholds

POSITION_NAMES = ("prompt_last", "p033", "p066", "p100", "decision", "mean")
HARD_POSITION_NAMES = ("prompt_last", "pre_tool", "p100", "mean")


class SamplingCfg(ConfigModel):
    train_split: str = "probe_train"
    val_split: str = "probe_val"
    require_definite_verdict: bool = True
    samples_per_scenario: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    variants: list[Literal["none", "full"]] = ["none", "full"]
    seed: int = 20260910
    # warn when the newest validation run of this model shows a spontaneous mention rate below this (token forcing)
    min_spontaneous_mention_rate: float = 0.3


class ActivationsCfg(ConfigModel):
    context_variant: Literal["same", "none", "full"] = "same"  # "none" after a "full" generation = token forcing
    positions: list[str] = list(POSITION_NAMES)
    dtype: Literal["float32", "float16"] = "float32"
    shard_size: int = 512
    batch_size: int = 8
    hard_batch_size: int = 2  # agentic samples are ~3.5k tokens; full-vocab logits dominate memory

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        bad = set(self.positions) - set(POSITION_NAMES)
        if bad:
            raise ValueError(f"unknown activation positions {sorted(bad)}; known: {POSITION_NAMES}")


class LabelsCfg(ConfigModel):
    thresholds: LabelThresholds = LabelThresholds()
    specs: list[str | LabelSpec] = ["B_primary", "B_cell1_vs_cell3", "B_outcome", "B_process", "C_context"]

    def resolved_specs(self) -> list[LabelSpec]:
        out: list[LabelSpec] = []
        for s in self.specs:
            if isinstance(s, str):
                if s not in BUILTIN_SPECS:
                    raise ValueError(f"unknown label spec {s!r}; built-in: {sorted(BUILTIN_SPECS)}")
                out.append(BUILTIN_SPECS[s])
            else:
                out.append(s)
        names = [s.name for s in out]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate label spec names: {names}")
        return out


class BootstrapCfg(ConfigModel):
    n: int = 2000
    seed: int = 0


class TrainCfg(ConfigModel):
    method: Literal["diff_means"] = "diff_means"
    bootstrap: BootstrapCfg = BootstrapCfg()


class SteeringTuningCfg(ConfigModel):
    split: str = "probe_val"
    n_scenarios: int = 20
    specs: list[str] = ["B_primary", "C_context"]


class SteeringMainCfg(ConfigModel):
    split: str = "heldout_steer"
    signs: list[int] = [1, -1]
    layer_sweep: bool = False


class CoherenceCfg(ConfigModel):
    min_parse_rate: float = 0.9
    max_len_ratio: float = 2.0
    max_repetition_ratio_delta: float = 0.0


class SteeringCfg(ConfigModel):
    positions: Literal["all", "generated"] = "all"
    prompt_variant: Literal["none", "full"] = "none"  # steering is tested without the constitution in the prompt
    coef_grid: list[float] = [1.0, 2.0, 4.0, 8.0]
    temperature: float = 0.0
    max_tokens: int = 2048
    batch_size: int = 16
    tuning: SteeringTuningCfg = SteeringTuningCfg()
    main: SteeringMainCfg = SteeringMainCfg()
    coherence: CoherenceCfg = CoherenceCfg()


class HardDataCfg(ConfigModel):
    run_dir: str = "outputs/misalignment/20260911_153043_77860d1a"
    positions: list[str] = ["prompt_last", "pre_tool", "p100", "mean"]
    process_threshold: float = 0.75
    outcome_threshold: float = 0.5

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        bad = set(self.positions) - set(HARD_POSITION_NAMES)
        if bad:
            raise ValueError(f"unknown hard-data positions {sorted(bad)}; known: {HARD_POSITION_NAMES}")


class SAECfg(ConfigModel):
    repo_id: str = "google/gemma-scope-2-27b-it"
    width: str = "64k"
    l0: str = "medium"
    top_k: int = 20
    neuronpedia_model: str = "gemma-3-27b-it"


class ProbeConfig(ConfigModel):
    model_config_path: str = "configs/model_sft_v2e3.yaml"
    # Claude judge for probe data / steering records (same keys as validation.yaml so calign.validate.judge works)
    judge_model: str = "claude-sonnet-5"
    judge_thinking: str = "adaptive"
    judge_effort: str | None = "medium"
    judge_concurrency: int = 8
    sampling: SamplingCfg = SamplingCfg()
    activations: ActivationsCfg = ActivationsCfg()
    labels: LabelsCfg = LabelsCfg()
    train: TrainCfg = TrainCfg()
    steering: SteeringCfg = SteeringCfg()
    hard_data: HardDataCfg = HardDataCfg()
    sae: SAECfg = SAECfg()


DEFAULT_CONFIG = REPO_ROOT / "configs" / "probe.yaml"


def load_probe_config(path: Path | None = None, seed: int | None = None) -> ProbeConfig:
    cfg = load_config(path or DEFAULT_CONFIG, ProbeConfig)
    if seed is not None:
        cfg = cfg.model_copy(update={"sampling": cfg.sampling.model_copy(update={"seed": seed})})
    return cfg
