"""Upload an SFT run (merged model at the repo root, LoRA adapter under adapter/) to a Hugging Face repo.

CLI (GPU machine; HF_TOKEN needs write access to the target namespace):
    uv run python -m calign.train.push_to_hub --run-dir outputs/models/sft_pilot \\
        --repo felixfabricius/gemma-3-27b-it-halden-sft-pilot [--what merged,adapter] [--public] [--dry-run]
        [--merged-dir merged] [--adapter-dir adapter]   # subdirs of --run-dir, e.g. merged_epoch3 / adapter_epoch3

The repo is private unless --public (Gemma derivatives shared publicly must carry the Gemma Terms of Use).
The merged model uses `upload_large_folder` (resumable; rerun the same command after an interruption).
A README.md model card is generated from the run dir (base model, LoRA config, data hashes, peak memory,
merge check, git commit), and <run-dir>/push_manifest.json records what was uploaded where.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from calign.config import git_commit
from calign.paths import hf_token, load_env
from calign.schemas import utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)
PARTS = ("merged", "adapter")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _checkpoint_line(run_dir: Path, adapter_dir: str, final_eval: dict) -> str:
    """Which checkpoint the repo holds; for adapter_epoch{k}, the eval loss at that epoch from train_log.json."""
    m = re.fullmatch(r"adapter_epoch(\d+)", adapter_dir)
    if not m:
        return f"- Checkpoint: end of training; final eval: {json.dumps(final_eval) if final_eval else 'n/a'}"
    k = int(m.group(1))
    log_path = run_dir / "train_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    per_epoch = {
        round(e["epoch"]): round(e["eval_loss"], 4)
        for e in log
        if "eval_loss" in e and abs(e["epoch"] - round(e["epoch"])) < 1e-6
    }
    return (
        f"- Checkpoint: end of epoch {k} (`{adapter_dir}`), not the final epoch; eval loss at epoch {k}: "
        f"{per_epoch.get(k, 'n/a')}; eval loss per epoch: {per_epoch}"
    )


def build_model_card(run_dir: Path, repo_id: str, merged_dir: str = "merged", adapter_dir: str = "adapter") -> str:
    """Model card from the SFT run dir: provenance only, every number copied from a file in the run dir."""
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    meta = _load_json(run_dir / "run_meta.json")
    peft = _load_json(run_dir / "peft_summary.json")
    stats = _load_json(run_dir / "data_stats.json")
    final_eval = _load_json(run_dir / "final_eval.json")
    merge = _load_json(run_dir / merged_dir / "merge_manifest.json")
    checkpoint = _checkpoint_line(run_dir, adapter_dir, final_eval)
    lora, train = cfg.get("lora", {}), cfg.get("train", {})
    lines = [
        "---",
        f"base_model: {cfg['base_model']}",
        "library_name: transformers",
        "license: gemma",
        "tags: [constitutional-alignment, lora, sft, halden-constitution]",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        f"`{cfg['base_model']}` fine-tuned with LoRA SFT on a synthetic corpus teaching the Halden Constitution "
        "(research checkpoint for github.com/felixfabricius/constitutional-alignment). The repo root holds the merged "
        "bf16 model; `adapter/` holds the LoRA adapter (apply it to the base model to reproduce the merge). "
        "Use is subject to the Gemma Terms of Use.",
        "",
        "## Training",
        "",
        f"- LoRA r={lora.get('r')}, alpha={lora.get('alpha')}, dropout={lora.get('dropout')}, "
        f"target_modules=`{lora.get('target_modules')}`",
        f"- {train.get('epochs')} epochs, lr {train.get('learning_rate')} ({train.get('lr_scheduler')}, warmup ratio "
        f"{train.get('warmup_ratio')}), effective batch "
        f"{(train.get('per_device_batch_size') or 0) * (train.get('gradient_accumulation_steps') or 0)}, "
        f"max_seq_len {cfg.get('max_seq_len')}",
        f"- LoRA modules: {peft.get('n_lora_modules')}, trainable params: {peft.get('trainable_params')}, "
        f"peak GPU memory: {peft.get('gpu_peak_reserved_gb')} GB reserved of {peft.get('gpu_total_gb')} GB",
        f"- Data: {cfg.get('train_file')} (sha256 {stats.get('train_file_sha256')}), "
        f"{(stats.get('train') or {}).get('n_examples')} train examples, "
        f"{(stats.get('train') or {}).get('n_tokens')} tokens",
        checkpoint,
        f"- Code: git commit {meta.get('git_commit')} (dirty={meta.get('git_dirty')}), run created {meta.get('created_at')}",
        "",
        "## Merge check (adapter vs merged, next-token logits)",
        "",
        f"- KL per prompt: {merge.get('kl_per_prompt')}",
        f"- top-1 agreement: {merge.get('top1_agree_per_prompt')}",
        f"- max |logit diff|: {merge.get('max_logit_diff_per_prompt')} (bf16 rounding floor)",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="SFT run dir containing merged/ and/or adapter/")
    ap.add_argument("--repo", required=True, help="namespace/name")
    ap.add_argument("--what", default="merged,adapter", help="comma-separated subset of: merged, adapter")
    ap.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    ap.add_argument("--dry-run", action="store_true", help="print the model card and file list; no network calls")
    ap.add_argument("--merged-dir", default="merged", help="subdir of --run-dir with the merged model")
    ap.add_argument("--adapter-dir", default="adapter", help="subdir of --run-dir with the adapter (adapter_epoch{k})")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    parts = [p.strip() for p in args.what.split(",") if p.strip()]
    if unknown := set(parts) - set(PARTS):
        raise SystemExit(f"unknown --what parts: {sorted(unknown)}")
    dirs = {"merged": args.run_dir / args.merged_dir, "adapter": args.run_dir / args.adapter_dir}
    for part in parts:
        if not dirs[part].is_dir():
            raise SystemExit(f"missing {dirs[part]}")
    card = build_model_card(args.run_dir, args.repo, args.merged_dir, args.adapter_dir)
    files = {
        part: sorted(str(p.relative_to(dirs[part])) for p in dirs[part].rglob("*") if p.is_file()) for part in parts
    }
    if args.dry_run:
        print(card)
        print(json.dumps({k: v for k, v in files.items()}, indent=2))
        return

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token())
    url = api.create_repo(args.repo, private=not args.public, exist_ok=True, repo_type="model")
    LOGGER.info("repo %s (private=%s)", url, not args.public)
    if "merged" in parts:
        # resumable, parallel upload of the ~55 GB checkpoint to the repo root
        api.upload_large_folder(repo_id=args.repo, folder_path=dirs["merged"], repo_type="model")
    if "adapter" in parts:
        api.upload_folder(
            repo_id=args.repo,
            folder_path=dirs["adapter"],
            path_in_repo="adapter",
            commit_message="Add LoRA adapter",
        )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=args.repo, commit_message="Model card"
    )
    info = api.model_info(args.repo)
    manifest = {
        "repo": args.repo,
        "url": str(url),
        "private": not args.public,
        "parts": parts,
        "dirs": {k: str(dirs[k]) for k in parts},
        "n_files": {k: len(v) for k, v in files.items()},
        "repo_sha": info.sha,
        "git_commit": git_commit(),
        "pushed_at": utc_now_iso(),
    }
    write_json(args.run_dir / "push_manifest.json", manifest)
    LOGGER.info("pushed %s -> %s @ %s", parts, args.repo, info.sha)


if __name__ == "__main__":
    main()
