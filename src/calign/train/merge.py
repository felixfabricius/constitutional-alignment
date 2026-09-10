"""Merge a LoRA adapter into the base model and save a plain HF checkpoint (no adapter architecture).

CLI (GPU machine):
    uv run python -m calign.train.merge --adapter outputs/models/sft_pilot/adapter [--out outputs/models/sft_pilot/merged]
        [--base google/gemma-2-9b-it] [--n-check 3]

Writes the merged model + tokenizer and merge_manifest.json with the max |logit diff| between the
adapter model and the merged model on a few prompts (should be ~1e-2 or less in bf16).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from calign.config import git_commit
from calign.constitution import load_constitution
from calign.data.samples import SAMPLE_SCENARIO
from calign.paths import hf_token, load_env
from calign.prompting import build_scenario_messages, encode_prompt, render_gemma_chat
from calign.schemas import utc_now_iso

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--base", default=None, help="base model id/path (default: adapter's base_model_name_or_path)")
    ap.add_argument("--n-check", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapter_cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base_id = args.base or adapter_cfg["base_model_name_or_path"]
    out_dir = args.out or args.adapter.parent / "merged"

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16, attn_implementation="eager", token=hf_token()
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
    diffs = [float((a - b).abs().max()) for a, b in zip(before, after, strict=True)]
    LOGGER.info("max |logit diff| adapter vs merged per prompt: %s", diffs)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    manifest = {
        "base_model": base_id,
        "adapter": str(args.adapter),
        "adapter_config": adapter_cfg,
        "merged_dir": str(out_dir),
        "max_logit_diff_per_prompt": diffs,
        "git_commit": git_commit(),
        "created_at": utc_now_iso(),
    }
    (out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("merged model saved to %s", out_dir)


if __name__ == "__main__":
    main()
