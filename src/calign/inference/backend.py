"""Sampling backend interface shared by vLLM (fast sampling) and HF transformers (training/activations).

Both backends consume *token ids* (see `calign.prompting.encode_prompt`) so prompts are byte-identical
regardless of backend, and both return `Completion` objects with text, token ids and finish reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from calign.config import ConfigModel, load_config
from calign.paths import REPO_ROOT


class VLLMConfig(ConfigModel):
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False


class ModelConfig(ConfigModel):
    model_id: str = "google/gemma-2-9b-it"
    model_path: str = "google/gemma-2-9b-it"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: str = "eager"
    max_model_len: int = 8192
    backend: Literal["vllm", "hf"] = "vllm"
    vllm: VLLMConfig = VLLMConfig()


def load_model_config(
    path: Path | None = None, model_path: str | None = None, backend: str | None = None
) -> ModelConfig:
    path = path or REPO_ROOT / "configs" / "model.yaml"
    return load_config(path, ModelConfig, overrides={"model_path": model_path, "backend": backend})


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024
    n: int = 1
    seed: int | None = None
    stop_token_ids: tuple[int, ...] | None = None


@dataclass
class Completion:
    text: str
    token_ids: list[int]
    finish_reason: str | None  # "stop" | "length" | None
    n_prompt_tokens: int
    logprobs: list[float] | None = field(default=None, repr=False)


class SamplingBackend(Protocol):
    name: str
    model_path: str
    tokenizer: Any

    def generate(self, prompt_token_ids: list[list[int]], params: SamplingParams) -> list[list[Completion]]:
        """Return `n` completions for each prompt (outer list aligned with prompts)."""
        ...


def load_backend(cfg: ModelConfig, backend: str | None = None, **kwargs: Any) -> SamplingBackend:
    kind = backend or cfg.backend
    if kind == "vllm":
        from calign.inference.vllm_backend import VLLMBackend

        return VLLMBackend(cfg, **kwargs)
    if kind == "hf":
        from calign.inference.hf_backend import HFBackend

        return HFBackend(cfg, **kwargs)
    raise ValueError(f"unknown backend {kind!r}")


def gemma_stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """EOS plus <end_of_turn>, which is how Gemma-IT ends a model turn."""
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0 and eot != tokenizer.unk_token_id:
        ids.append(eot)
    return tuple(dict.fromkeys(ids))
