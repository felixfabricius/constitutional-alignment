"""LoRA SFT of Gemma 2 on the synthetic corpus (HF Trainer + PEFT).

CLI (GPU machine):
    uv run python -m calign.train.sft --config configs/sft.yaml [--run-name NAME] [--dry-run] [--limit N]

Output: outputs/models/<run_name>/adapter/ (LoRA weights + tokenizer), train_log.json, resolved_config.yaml,
run_meta.json, data_stats.json. Merge afterwards with calign.train.merge.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from calign.config import ConfigModel, add_common_args, effective_limit, load_config, new_run_dir, sha256_file
from calign.paths import REPO_ROOT, hf_token, load_env
from calign.schemas import SFTExample, read_jsonl, write_json
from calign.train.data import PadCollator, SFTDataset, dataset_stats

LOGGER = logging.getLogger(__name__)


class LoraCfg(ConfigModel):
    r: int = 256
    alpha: int = 256
    dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class TrainCfg(ConfigModel):
    epochs: float = 2
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    group_by_length: bool = True  # informational; transformers 5 removed the Trainer option
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 200
    seed: int = 20260910


class SFTConfig(ConfigModel):
    base_model: str = "google/gemma-2-9b-it"
    train_file: str = "data/sft/train.jsonl"
    val_file: str = "data/sft/val.jsonl"
    output_root: str = "outputs/models"
    run_name: str = "sft_pilot"
    max_seq_len: int = 2048
    attn_implementation: str = "eager"
    lora: LoraCfg = LoraCfg()
    train: TrainCfg = TrainCfg()


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "sft.yaml")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    cfg = load_config(args.config, SFTConfig, overrides={"run_name": args.run_name, "base_model": args.model_path})
    if args.seed is not None:
        cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"seed": args.seed})})
    limit = effective_limit(args)
    run_name = cfg.run_name + ("_dryrun" if args.dry_run else "")
    run_dir = new_run_dir("sft", cfg, out=args.out or _abs(cfg.output_root) / run_name, dry_run=args.dry_run)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, token=hf_token())
    train_ex = read_jsonl(_abs(cfg.train_file), SFTExample)
    val_ex = read_jsonl(_abs(cfg.val_file), SFTExample) if _abs(cfg.val_file).exists() else []
    if limit:
        train_ex, val_ex = train_ex[: max(limit, 8)], val_ex[: max(limit, 4)]
    train_ds = SFTDataset(tokenizer, train_ex, cfg.max_seq_len)
    val_ds = SFTDataset(tokenizer, val_ex, cfg.max_seq_len) if val_ex else None
    stats = {
        "train": dataset_stats(train_ds),
        "val": dataset_stats(val_ds) if val_ds else None,
        "train_file_sha256": sha256_file(_abs(cfg.train_file)),
    }
    write_json(run_dir / "data_stats.json", stats)
    LOGGER.info("data: %s", stats)

    dtype = torch.bfloat16 if cfg.train.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=dtype, attn_implementation=cfg.attn_implementation, token=hf_token()
    )
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    peft_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("trainable params: %.1fM / %.2fB (%.2f%%)", trainable / 1e6, total / 1e9, 100 * trainable / total)

    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=cfg.train.epochs,
        max_steps=3 if args.dry_run else -1,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler,
        warmup_steps=cfg.train.warmup_ratio,  # transformers>=5: float in [0,1) means a ratio of total steps
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        bf16=cfg.train.bf16,
        logging_steps=cfg.train.logging_steps,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=cfg.train.eval_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=2,
        seed=cfg.train.seed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (run_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    if val_ds:
        metrics = trainer.evaluate()
        write_json(run_dir / "final_eval.json", metrics)
        LOGGER.info("final eval: %s", metrics)
    LOGGER.info("adapter saved to %s", adapter_dir)


if __name__ == "__main__":
    main()
