"""Export a multimodal Gemma 3 checkpoint (base or merged SFT) as a text-only `Gemma3ForCausalLM` checkpoint.

Why: under the text-only class, PEFT module names and vLLM/TRL prefixes are the standard `model.layers.N...` ones
(chunk 7's GRPO prefers it), and no vision tower is loaded. The export rewrites the safetensors shards key by key
(no model instantiation, ~5 GB of RAM per shard): `model.language_model.*` (transformers 5 layout) or
`language_model.model.*` (original Google layout) -> `model.*`, `lm_head.*` kept unless the embeddings are tied, and
`vision_tower.*` / `multi_modal_projector.*` dropped; `config.json` becomes the checkpoint's `text_config` with
`architectures: [Gemma3ForCausalLM]`. Tokenizer, chat template and generation config are copied.

CLI (GPU machine for --verify):
    uv run python -m calign.train.export_text_only --model-path outputs/models/sft_v3/merged_epoch2 \
        --out outputs/models/sft_v3/merged_epoch2_text [--revision SHA] [--verify 5]
`--verify N` compares next-token logits of the multimodal load and the text-only load on N prompts (sequentially on
one GPU) and writes the check into <out>/export_manifest.json (pass: top-1 equal on all prompts and KL < 1e-3).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from calign.config import git_commit
from calign.paths import hf_token, load_env
from calign.schemas import utc_now_iso

LOGGER = logging.getLogger(__name__)

PREFIX_MAP = (
    ("model.language_model.", "model."),
    ("language_model.model.", "model."),
    ("language_model.lm_head.", "lm_head."),
    ("lm_head.", "lm_head."),
)
DROP_PREFIXES = ("model.vision_tower.", "vision_tower.", "model.multi_modal_projector.", "multi_modal_projector.")
# Files not copied: weights and their index are rewritten, config is converted, processor files are multimodal-only.
SKIP_FILES = {"config.json", "model.safetensors.index.json", "preprocessor_config.json", "processor_config.json"}
VERIFY_MAX_KL = 1e-3


def map_key(key: str, tied: bool) -> str | None:
    """Text-only name of a multimodal checkpoint tensor, or None to drop it."""
    if key.startswith(DROP_PREFIXES):
        return None
    for old, new in PREFIX_MAP:
        if key.startswith(old):
            out = new + key[len(old) :]
            return None if tied and out.startswith("lm_head.") else out
    raise ValueError(f"unexpected tensor name in a Gemma 3 multimodal checkpoint: {key}")


def text_config(mm_config: dict) -> dict:
    """config.json of the text-only checkpoint from the multimodal config.json."""
    if "text_config" not in mm_config:
        raise ValueError("config has no text_config; is this already a text-only checkpoint?")
    tc = dict(mm_config["text_config"])
    tc["architectures"] = ["Gemma3ForCausalLM"]
    tc["model_type"] = "gemma3_text"
    for k in ("torch_dtype", "dtype", "transformers_version"):
        if k in mm_config and k not in tc:
            tc[k] = mm_config[k]
    return tc


def _tied(mm_config: dict) -> bool:
    tc = mm_config.get("text_config", {})
    return bool(tc.get("tie_word_embeddings", mm_config.get("tie_word_embeddings", True)))


def export(src: Path, out: Path) -> dict:
    """Rewrite the checkpoint in `src` (a local dir) into `out`. Returns counts for the manifest."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    mm_config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    tied = _tied(mm_config)
    index_path = src / "model.safetensors.index.json"
    shards = (
        sorted(set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()))
        if index_path.exists()
        else ["model.safetensors"]
    )
    out.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    n_dropped, total_size = 0, 0
    for shard in shards:
        tensors = {}
        with safe_open(str(src / shard), framework="pt") as f:
            for k in f.keys():  # noqa: SIM118 (safe_open has no __iter__)
                new = map_key(k, tied)
                if new is None:
                    n_dropped += 1
                    continue
                t = f.get_tensor(k)
                tensors[new] = t
                total_size += t.numel() * t.element_size()
        if tensors:
            save_file(tensors, str(out / shard), metadata={"format": "pt"})
            weight_map.update({k: shard for k in tensors})
        LOGGER.info("%s: %d tensors kept", shard, len(tensors))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}, indent=2),
        encoding="utf-8",
    )
    (out / "config.json").write_text(json.dumps(text_config(mm_config), indent=2), encoding="utf-8")
    copied = []
    for p in sorted(src.iterdir()):
        if p.is_file() and p.name not in SKIP_FILES and p.suffix != ".safetensors":
            shutil.copyfile(p, out / p.name)
            copied.append(p.name)
    return {"n_tensors": len(weight_map), "n_dropped": n_dropped, "tied_embeddings": tied, "copied_files": copied}


def _local_dir(model_path: str, revision: str | None) -> Path:
    p = Path(model_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path, revision=revision, token=hf_token()))


def verify(src: Path, out: Path, n: int) -> dict:
    """Next-token logits of the multimodal load (src) vs the text-only load (out) on n prompts."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calign.constitution import load_constitution
    from calign.data.samples import SAMPLE_SCENARIO
    from calign.prompting import build_scenario_messages, encode_prompt, render_gemma_chat
    from calign.train.merge import merge_check

    tok = AutoTokenizer.from_pretrained(out)
    c = load_constitution()
    texts = [render_gemma_chat(build_scenario_messages(SAMPLE_SCENARIO, c, v)) for v in ("full", "none")]
    texts += [
        render_gemma_chat([{"role": "user", "content": q}])
        for q in (
            "What does your constitution say?",
            "Write a haiku about autumn rain.",
            "Solve 3x + 7 = 22 and explain each step.",
            "Draft a short email moving Tuesday's meeting to Thursday.",
        )
    ]
    prompts = [encode_prompt(tok, t) for t in texts[:n]]

    def logits(path: Path) -> tuple[list, str]:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa")
        model.to("cuda").eval()
        with torch.no_grad():
            out_ = [model(torch.tensor([p], device="cuda")).logits[0, -1].float().cpu() for p in prompts]
        cls = type(model).__name__
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return out_, cls

    before, cls_mm = logits(src)
    after, cls_text = logits(out)
    check = merge_check(before, after)
    check["passed"] = all(check["top1_agree_per_prompt"]) and max(check["kl_per_prompt"]) < VERIFY_MAX_KL
    check.update({"class_multimodal": cls_mm, "class_text_only": cls_text, "n_prompts": len(prompts)})
    return check


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="multimodal checkpoint: local dir or hub id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify", type=int, default=0, help="compare logits on N prompts (GPU)")
    ap.add_argument("--dry-run", action="store_true", help="print the key mapping summary only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    src = _local_dir(args.model_path, args.revision)
    if args.dry_run:
        cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
        print(json.dumps(text_config(cfg), indent=2))
        return
    if (args.out / "export_manifest.json").exists():
        raise SystemExit(f"{args.out} already holds an export")
    counts = export(src, args.out)
    manifest = {
        "source": args.model_path,
        "source_dir": str(src),
        "revision": args.revision,
        **counts,
        "git_commit": git_commit(),
        "created_at": utc_now_iso(),
    }
    if args.verify:
        manifest["verify"] = verify(src, args.out, args.verify)
        LOGGER.info("verify: %s", manifest["verify"])
    (args.out / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info(
        "text-only export written to %s (%d tensors, %d dropped)", args.out, counts["n_tensors"], counts["n_dropped"]
    )
    if args.verify and not manifest["verify"]["passed"]:
        raise SystemExit("verification failed; see export_manifest.json")


if __name__ == "__main__":
    main()
