"""LoRA adapters for evaluation: resolving adapter specs and serving them with vLLM (no merge).

An adapter spec (`EvalConfig.adapter`) is either a local directory holding `adapter_config.json` +
`adapter_model.safetensors` (PEFT format, as written by `calign.train.sft`) or a Hub reference
`hf://<namespace>/<repo>/<subdir>[@<revision>]` (e.g. `hf://felixfabricius/gemma-3-27b-it-halden-sft-v3/adapter_epoch2@<sha>`),
which is downloaded once into the HF cache. vLLM applies the adapter through `LoRARequest` on every generate call
(`calign.inference.vllm_backend.VLLMBackend(cfg, adapter=...)`); the multimodal Gemma 3 key prefix
(`base_model.model.model.language_model.layers.N...`) is mapped by vLLM's Gemma 3 weights mapper, verified by
`calign.inference.lora_check` (LoRA-served vs merged).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HF_PREFIX = "hf://"
# Ranks vLLM accepts for max_lora_rank (vllm.config.lora); the adapter's r is rounded up to the next one.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
_HF_RE = re.compile(r"^hf://(?P<repo>[^/@]+/[^/@]+)(?:/(?P<sub>[^@]+?))?/?(?:@(?P<rev>[^/]+))?$")


def parse_hf_spec(spec: str) -> tuple[str, str | None, str | None]:
    """`hf://ns/repo/sub/dir@rev` -> (repo id, subdir or None, revision or None)."""
    m = _HF_RE.match(spec)
    if not m:
        raise ValueError(f"not an hf:// adapter spec: {spec!r} (expected hf://<ns>/<repo>[/<subdir>][@<revision>])")
    return m.group("repo"), m.group("sub"), m.group("rev")


def is_hf_spec(spec: str | None) -> bool:
    return bool(spec) and str(spec).startswith(HF_PREFIX)


def resolve_adapter(spec: str) -> Path:
    """Local directory of an adapter spec, downloading `hf://` specs (only the subdir) into the HF cache."""
    if not is_hf_spec(spec):
        p = Path(spec)
        if not (p / "adapter_config.json").exists():
            raise FileNotFoundError(f"{p} has no adapter_config.json")
        return p
    from huggingface_hub import snapshot_download

    from calign.paths import hf_token

    repo, sub, rev = parse_hf_spec(spec)
    root = snapshot_download(
        repo, revision=rev, allow_patterns=[f"{sub}/*"] if sub else None, token=hf_token(), repo_type="model"
    )
    p = Path(root) / sub if sub else Path(root)
    if not (p / "adapter_config.json").exists():
        raise FileNotFoundError(f"{spec} resolved to {p}, which has no adapter_config.json")
    return p


def adapter_config(path: Path | str) -> dict:
    return json.loads((Path(path) / "adapter_config.json").read_text(encoding="utf-8"))


def vllm_max_lora_rank(r: int) -> int:
    """Smallest vLLM-supported max_lora_rank >= r."""
    for k in VLLM_LORA_RANKS:
        if k >= r:
            return k
    raise ValueError(f"LoRA rank {r} exceeds vLLM's maximum {VLLM_LORA_RANKS[-1]}")


def check_text_only(cfg: dict) -> None:
    """Refuse adapters that touch the vision tower or projector (vLLM serves the language model only)."""
    tm = cfg.get("target_modules")
    names = tm if isinstance(tm, list) else [str(tm)]
    bad = [n for n in names if "vision" in n or "multi_modal_projector" in n]
    if bad:
        raise ValueError(f"adapter targets non-text modules: {bad}")
