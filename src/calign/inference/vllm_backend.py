"""vLLM backend for fast batched sampling (Linux GPU machine only)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from calign.inference.backend import Completion, ModelConfig, SamplingParams, gemma_stop_token_ids

LOGGER = logging.getLogger(__name__)


class VLLMBackend:
    name = "vllm"

    def __init__(self, cfg: ModelConfig, seed: int = 0, **llm_kwargs: Any) -> None:
        # vLLM's warmup runs FlashInfer's top-k/top-p sampler, which JIT-compiles with nvcc and crashes
        # on machines without a CUDA toolkit. Our requests are seeded, so vLLM never uses that sampler
        # for them anyway; the native sampler is equivalent. Set the env var to 1 to opt back in.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM

        if cfg.vllm.language_model_only:
            llm_kwargs.setdefault("language_model_only", True)
        if cfg.revision and not Path(cfg.model_path).is_dir():  # hub revision only for hub repo ids
            llm_kwargs.setdefault("revision", cfg.revision)
        self.cfg = cfg
        self.model_path = cfg.model_path
        self.llm = LLM(
            model=cfg.model_path,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            enforce_eager=cfg.vllm.enforce_eager,
            seed=seed,
            **llm_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.stop_ids = gemma_stop_token_ids(self.tokenizer)

    def generate(self, prompt_token_ids: list[list[int]], params: SamplingParams) -> list[list[Completion]]:
        from vllm import SamplingParams as VSP
        from vllm.inputs import TokensPrompt

        stop_ids = list(params.stop_token_ids or self.stop_ids)
        vsp = VSP(
            n=params.n,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            seed=params.seed,
            stop_token_ids=stop_ids,
            skip_special_tokens=False,
        )
        prompts = [TokensPrompt(prompt_token_ids=list(ids)) for ids in prompt_token_ids]
        outputs = self.llm.generate(prompts, vsp, use_tqdm=True)
        results: list[list[Completion]] = []
        for ids, out in zip(prompt_token_ids, outputs, strict=True):
            comps = []
            for o in out.outputs:
                toks = list(o.token_ids)
                text = o.text
                # vLLM omits the stop token from `text`; keep token_ids as generated (may include stop id)
                comps.append(
                    Completion(text=text, token_ids=toks, finish_reason=o.finish_reason, n_prompt_tokens=len(ids))
                )
            results.append(comps)
        return results
