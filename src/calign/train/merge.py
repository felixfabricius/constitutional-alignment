"""Merge a LoRA adapter into the base model and save a plain HF checkpoint (no adapter architecture).

CLI (GPU machine):
    uv run python -m calign.train.merge --adapter outputs/models/sft_pilot/adapter [--out outputs/models/sft_pilot/merged]
        [--base google/gemma-3-27b-it] [--attn-implementation sdpa|eager] [--n-check 3]

Writes the merged model + tokenizer (+ the base model's processor files, which vLLM may read for
multimodal architectures such as Gemma 3) and merge_manifest.json comparing the adapter model and the merged
model on a few prompts: KL divergence (bf16 noise ~1e-5 to 1e-3; > 1e-2 is suspect), top-1 agreement (expect all
true) and max |logit diff| (bf16 rounding floor, ~0.3 for Gemma 2, ~1-2 for Gemma 3; see `merge_check`).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from calign.config import git_commit
from calign.constitution import load_constitution
from calign.data.samples import SAMPLE_SCENARIO
from calign.inference.backend import default_attn_implementation
from calign.paths import hf_token, load_env
from calign.prompting import build_scenario_messages, encode_prompt, render_gemma_chat
from calign.schemas import utc_now_iso

LOGGER = logging.getLogger(__name__)

# Non-weight files of multimodal checkpoints that save_pretrained(model/tokenizer) does not write.
PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json", "chat_template.json", "chat_template.jinja")


def copy_processor_files(base_id: str, out_dir: Path) -> list[str]:
    """Copy processor/chat-template files from the base model (local dir or HF repo) if missing in out_dir."""
    base_dir = Path(base_id)
    if base_dir.is_dir():
        available = {name: base_dir / name for name in PROCESSOR_FILES if (base_dir / name).exists()}
    else:
        from huggingface_hub import HfApi, hf_hub_download

        repo_files = set(HfApi(token=hf_token()).list_repo_files(base_id))
        available = {
            name: Path(hf_hub_download(base_id, name, token=hf_token()))
            for name in PROCESSOR_FILES
            if name in repo_files
        }
    copied = []
    for name, src in available.items():
        if not (out_dir / name).exists():
            shutil.copyfile(src, out_dir / name)
            copied.append(name)
    return copied


def merge_check(before: list, after: list) -> dict[str, list]:
    """Compare next-token logits of the adapter model (before) and the merged model (after), per prompt.

    Max |logit diff| alone is not scale-free: bf16 re-rounds W + BA, giving a floor of ~1-3 bf16 ulps of the
    largest logit (Gemma 2 caps logits at 30 -> ~0.3; uncapped Gemma 3 logits reach ~60-70 -> ~0.75-1.6, while an
    fp32 merge is exact). KL(adapter || merged) and top-1 agreement are the pass criteria.
    """
    import torch

    kls, top1, diffs, scale = [], [], [], []
    for b, a in zip(before, after, strict=True):
        kls.append(float(torch.sum(torch.softmax(b, -1) * (torch.log_softmax(b, -1) - torch.log_softmax(a, -1)))))
        top1.append(bool(a.argmax() == b.argmax()))
        diffs.append(float((a - b).abs().max()))
        scale.append(float(b.abs().max()))
    return {
        "kl_per_prompt": kls,
        "top1_agree_per_prompt": top1,
        "max_logit_diff_per_prompt": diffs,
        "max_abs_logit_per_prompt": scale,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--base", default=None, help="base model id/path (default: adapter's base_model_name_or_path)")
    ap.add_argument("--n-check", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--attn-implementation", default=None, help="default: eager for Gemma 2, sdpa otherwise")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapter_cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base_id = args.base or adapter_cfg["base_model_name_or_path"]
    out_dir = args.out or args.adapter.parent / "merged"
    attn = args.attn_implementation or default_attn_implementation(
        AutoConfig.from_pretrained(base_id, token=hf_token()).model_type
    )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16, attn_implementation=attn, token=hf_token()
    )
    base.to(device).eval()
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    c = load_constitution()
    prompts = [
        encode_prompt(tokenizer, render_gemma_chat(build_scenario_messages(SAMPLE_SCENARIO, c, v)))
        for v in ("full", "none")
    ][: args.n_check]
    prompts.append(
        encode_prompt(tokenizer, render_gemma_chat([{"role": "user", "content": "What does your constitution say?"}]))
    )

    with torch.no_grad():
        before = [model(torch.tensor([p], device=device)).logits[0, -1].float().cpu() for p in prompts]
    merged = model.merge_and_unload()
    with torch.no_grad():
        after = [merged(torch.tensor([p], device=device)).logits[0, -1].float().cpu() for p in prompts]
    check = merge_check(before, after)
    LOGGER.info("adapter vs merged next-token logits: %s", check)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    processor_files = copy_processor_files(base_id, out_dir)
    manifest = {
        "base_model": base_id,
        "adapter": str(args.adapter),
        "adapter_config": adapter_cfg,
        "merged_dir": str(out_dir),
        "model_class": type(merged).__name__,
        "attn_implementation": attn,
        "processor_files_copied": processor_files,
        **check,
        "git_commit": git_commit(),
        "created_at": utc_now_iso(),
    }
    (out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("merged model saved to %s", out_dir)


if __name__ == "__main__":
    main()
