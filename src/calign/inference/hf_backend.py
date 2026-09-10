"""HF transformers backend: batched sampling, teacher-forced passes, hidden-state access.

Used for tests, small runs, LoRA merging checks and (Phase 2) activation extraction. vLLM is
preferred for large sampling runs.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from calign.inference.backend import Completion, ModelConfig, SamplingParams, gemma_stop_token_ids

LOGGER = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class HFBackend:
    name = "hf"

    def __init__(
        self,
        cfg: ModelConfig,
        device: str | None = None,
        batch_size: int = 8,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.model_path = cfg.model_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_path)
        if model is None:
            dtype = _DTYPES[cfg.dtype] if self.device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path, dtype=dtype, attn_implementation=cfg.attn_implementation
            )
            model.to(self.device)
        self.model = model
        self.model.eval()
        self.stop_ids = gemma_stop_token_ids(self.tokenizer)
        self.pad_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def generate(self, prompt_token_ids: list[list[int]], params: SamplingParams) -> list[list[Completion]]:
        stop_ids = list(params.stop_token_ids or self.stop_ids)
        # expand n samples per prompt into a flat list of (prompt_idx, ids)
        flat: list[tuple[int, list[int]]] = [
            (i, ids) for i, ids in enumerate(prompt_token_ids) for _ in range(params.n)
        ]
        results: list[list[Completion]] = [[] for _ in prompt_token_ids]
        if params.seed is not None:
            torch.manual_seed(params.seed)
        for b in range(0, len(flat), self.batch_size):
            batch = flat[b : b + self.batch_size]
            input_ids, attention_mask = self._left_pad([ids for _, ids in batch])
            gen_kwargs: dict[str, Any] = dict(
                max_new_tokens=params.max_tokens,
                eos_token_id=stop_ids,
                pad_token_id=self.pad_id,
                do_sample=params.temperature > 0,
            )
            if params.temperature > 0:
                gen_kwargs.update(temperature=params.temperature, top_p=params.top_p)
            out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            new_tokens = out[:, input_ids.shape[1] :]
            for (pi, ids), row in zip(batch, new_tokens, strict=True):
                toks = row.tolist()
                # cut at the first stop token (inclusive) / strip padding after it
                cut = len(toks)
                for j, t in enumerate(toks):
                    if t in stop_ids:
                        cut = j + 1
                        break
                toks = toks[:cut]
                finish = "stop" if toks and toks[-1] in stop_ids else "length"
                text_ids = toks[:-1] if finish == "stop" else toks
                text = self.tokenizer.decode(text_ids, skip_special_tokens=False)
                results[pi].append(
                    Completion(text=text, token_ids=toks, finish_reason=finish, n_prompt_tokens=len(ids))
                )
        return results

    def _left_pad(self, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s) :] = torch.tensor(s, dtype=torch.long)
            mask[i, width - len(s) :] = 1
        return ids.to(self.device), mask.to(self.device)

    # ------------------------------------------------------------------ forced passes (Phase 2)
    @torch.no_grad()
    def forward_forced(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
        layers: list[int] | None = None,
    ) -> dict[str, Any]:
        """Teacher-force `completion_ids` after `prompt_ids` in a single forward pass.

        Returns per-token log-probs of the completion tokens (under the model), the completion
        span, and residual-stream hidden states (layers x seq x d, on CPU) for the requested layers
        (`hidden_states[l]` is the output of block l-1; l=0 is the embeddings).
        """
        full = prompt_ids + completion_ids
        input_ids = torch.tensor([full], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, output_hidden_states=layers is not None)
        logits = out.logits[0].float()
        start, end = len(prompt_ids), len(full)
        # token t is predicted by logits at position t-1
        lp = torch.log_softmax(logits[start - 1 : end - 1], dim=-1)
        target = torch.tensor(completion_ids, device=self.device)
        token_logprobs = lp.gather(1, target[:, None])[:, 0].cpu().tolist()
        result: dict[str, Any] = {"span": (start, end), "token_logprobs": token_logprobs, "n_tokens": len(full)}
        if layers is not None:
            hs = out.hidden_states
            result["hidden_states"] = {layer: hs[layer][0].float().cpu() for layer in layers}
        return result
