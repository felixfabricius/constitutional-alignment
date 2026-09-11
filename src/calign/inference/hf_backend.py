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


def hidden_state_index(sae_layer: int) -> int:
    """HF `hidden_states` index for a Gemma Scope layer: resid_post of block L is hidden_states[L + 1]."""
    return sae_layer + 1


def decoder_layers(model: Any) -> Any:
    """The decoder-block ModuleList of a (possibly multimodal) Gemma model: block L's output is resid_post of L."""
    inner = getattr(model, "model", model)
    lm = getattr(inner, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        return lm.layers
    if hasattr(inner, "layers"):
        return inner.layers
    raise AttributeError("could not find decoder layers on the model")


def _block_output(o: Any) -> torch.Tensor:
    return o[0] if isinstance(o, tuple) else o


def _with_block_output(o: Any, x: torch.Tensor) -> Any:
    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x


class CaptureHooks:
    """Context manager capturing the outputs (resid_post) of the given Gemma Scope layers during a forward pass.

    Cheaper than `output_hidden_states=True` (which materialises every layer) and independent of how transformers
    records hidden states. `captured[L]` is the (batch, seq, d) block output of layer L (Gemma Scope numbering).
    """

    def __init__(self, model: Any, layers: list[int]) -> None:
        self.model, self.layers = model, list(layers)
        self.captured: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> CaptureHooks:
        blocks = decoder_layers(self.model)
        for L in self.layers:

            def hook(m, i, o, L=L):
                self.captured[L] = _block_output(o)

            self._handles.append(blocks[L].register_forward_hook(hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


class SteeringHook:
    """Add `scale * direction` to the output of one decoder block during every forward pass (activation steering).

    `positions="all"` adds the vector at every sequence position of every forward call (prompt tokens included);
    `positions="generated"` adds it only in decoding steps (forward calls with sequence length 1), so the prompt
    pass is unsteered. The vector is added in the residual dtype (bf16 on the GPU), so scales far below the
    residual magnitude are rounded away; use class-gap-scaled coefficients.
    """

    def __init__(self, model: Any, layer: int, direction: torch.Tensor, scale: float, positions: str = "all") -> None:
        if positions not in ("all", "generated"):
            raise ValueError(f"positions must be 'all' or 'generated', got {positions!r}")
        self.model, self.layer, self.positions = model, layer, positions
        self.vector = direction.detach().to(torch.float32) * float(scale)
        self.calls = 0
        self._handle: Any = None

    def __enter__(self) -> SteeringHook:
        def hook(m, i, o):
            x = _block_output(o)
            if self.positions == "generated" and x.shape[1] != 1:
                return o
            self.calls += 1
            return _with_block_output(o, x + self.vector.to(device=x.device, dtype=x.dtype))

        self._handle = decoder_layers(self.model)[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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
        rev = {"revision": cfg.revision} if cfg.revision else {}
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_path, **rev)
        if model is None:
            dtype = _DTYPES[cfg.dtype] if self.device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path, dtype=dtype, attn_implementation=cfg.attn_implementation, **rev
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
        (`hidden_states[l]` is the output of block l-1; l=0 is the embeddings). `layers` are HF indices:
        convert Gemma Scope / `ModelConfig.probe_layers` numbering with `hidden_state_index`.
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

    @torch.no_grad()
    def forward_forced_batch(
        self,
        prompt_ids: list[list[int]],
        completion_ids: list[list[int]],
        layers: list[int],
        positions: list[dict[str, int]],
    ) -> list[dict[str, Any]]:
        """Batched teacher-forced pass keeping only the requested positions (right-padded, one forward per batch).

        `layers` are Gemma Scope layer numbers (block outputs captured with hooks). `positions[i]` maps a position
        name to an absolute token index in prompt + completion for sequence i, or to -1 for the mean over the
        completion span; every dict must have the same keys in the same order. Returns, per sequence,
        {"span", "token_logprobs", "positions": [names], "acts": float32 tensor (n_layers, n_positions, d) on CPU}.
        """
        if not prompt_ids:
            return []
        names = list(positions[0])
        if any(list(p) != names for p in positions):
            raise ValueError("every positions dict must have the same keys in the same order")
        results: list[dict[str, Any]] = []
        for b in range(0, len(prompt_ids), self.batch_size):
            ps, cs, pos = (
                prompt_ids[b : b + self.batch_size],
                completion_ids[b : b + self.batch_size],
                positions[b : b + self.batch_size],
            )
            fulls = [p + c for p, c in zip(ps, cs, strict=True)]
            width = max(len(f) for f in fulls)
            input_ids = torch.full((len(fulls), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(fulls), width), dtype=torch.long)
            for i, f in enumerate(fulls):
                input_ids[i, : len(f)] = torch.tensor(f, dtype=torch.long)
                mask[i, : len(f)] = 1
            input_ids, mask = input_ids.to(self.device), mask.to(self.device)
            with CaptureHooks(self.model, layers) as cap:
                out = self.model(input_ids=input_ids, attention_mask=mask)
            for i, (p, c, f, pp) in enumerate(zip(ps, cs, fulls, pos, strict=True)):
                start, end = len(p), len(f)
                if end <= start:
                    raise ValueError("completion is empty")
                lp = torch.log_softmax(out.logits[i, start - 1 : end - 1].float(), dim=-1)
                target = torch.tensor(c, device=self.device)
                token_logprobs = lp.gather(1, target[:, None])[:, 0].cpu().tolist()
                acts = torch.empty((len(layers), len(names), cap.captured[layers[0]].shape[-1]), dtype=torch.float32)
                for li, L in enumerate(layers):
                    h = cap.captured[L][i].float()  # (width, d)
                    for pi, name in enumerate(names):
                        idx = pp[name]
                        if idx == -1:
                            acts[li, pi] = h[start:end].mean(dim=0).cpu()
                        else:
                            if not 0 <= idx < end:
                                raise ValueError(f"position {name}={idx} outside the sequence [0, {end})")
                            acts[li, pi] = h[idx].cpu()
                results.append(
                    {"span": (start, end), "token_logprobs": token_logprobs, "positions": names, "acts": acts}
                )
            del out, cap
        return results
