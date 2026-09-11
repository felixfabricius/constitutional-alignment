"""GPU test: 3-step LoRA SFT on a tiny synthetic dataset, then merge and check logit equivalence.

Uses CALIGN_TEST_BASE_MODEL (default google/gemma-3-4b-it: same multimodal class, module names and turn
format as the 27B, fits a 24 GB card). Gemma 2 bases (e.g. google/gemma-2-2b-it) use the plain LoRA target list.
"""

from __future__ import annotations

import json
import os
import textwrap

import pytest

from calign.constitution import load_constitution
from calign.schemas import Message, SFTExample, write_jsonl
from calign.train import merge as merge_mod
from calign.train import sft as sft_mod

pytestmark = pytest.mark.gpu

BASE = os.environ.get("CALIGN_TEST_BASE_MODEL", "google/gemma-3-4b-it")
# Gemma 3 checkpoints are multimodal: keep the default language-model regex (sft.GEMMA3_LORA_TARGETS).
TARGETS = ", target_modules: [q_proj, k_proj, v_proj, o_proj]" if "gemma-2" in BASE else ""


def test_sft_dry_run_and_merge(tmp_path):
    c = load_constitution()
    docs = [
        SFTExample(
            kind="doc", subtype="faq", text=f"Q: What is Principle {p.number}?\nA: {p.plain_text}", gen_model="t"
        )
        for p in c.principles
    ]
    trs = [
        SFTExample(
            kind="transcript",
            subtype="P1",
            messages=[
                Message(role="user", content=f"Which principle covers {p.title.lower()}?"),
                Message(role="assistant", content=f"Principle {p.number} ({p.title}) of {c.name}: {p.body}"),
            ],
            gen_model="t",
        )
        for p in c.principles
    ]
    write_jsonl(tmp_path / "train.jsonl", docs + trs)
    write_jsonl(tmp_path / "val.jsonl", trs[:2])
    cfg = tmp_path / "sft.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            base_model: {BASE}
            train_file: {tmp_path / "train.jsonl"}
            val_file: {tmp_path / "val.jsonl"}
            output_root: {tmp_path / "models"}
            run_name: t
            max_seq_len: 512
            lora: {{r: 8, alpha: 8, dropout: 0.0{TARGETS}}}
            train: {{epochs: 1, learning_rate: 1.0e-4, per_device_batch_size: 2, gradient_accumulation_steps: 1,
                     bf16: true, gradient_checkpointing: true, logging_steps: 1, eval_steps: 2, save_steps: 100}}
            """
        ),
        encoding="utf-8",
    )
    sft_mod.main(["--config", str(cfg), "--dry-run"])
    run_dir = tmp_path / "models" / "t_dryrun"
    adapter = run_dir / "adapter"
    assert (adapter / "adapter_config.json").exists() and (run_dir / "train_log.json").exists()
    log = json.loads((run_dir / "train_log.json").read_text())
    assert any("loss" in e for e in log)
    peft_summary = json.loads((run_dir / "peft_summary.json").read_text())
    assert peft_summary["n_lora_modules"] > 0 and "gpu_peak_allocated_gb" in peft_summary

    merge_mod.main(["--adapter", str(adapter), "--out", str(tmp_path / "merged"), "--n-check", "2"])
    manifest = json.loads((tmp_path / "merged" / "merge_manifest.json").read_text())
    assert (tmp_path / "merged" / "config.json").exists()
    assert max(manifest["max_logit_diff_per_prompt"]) < 0.5  # bf16 merge noise only
