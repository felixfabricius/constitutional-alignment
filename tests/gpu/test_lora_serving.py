"""GPU test (4B): vLLM LoRA serving equals the merged model; the text-only export equals the multimodal load.

    CALIGN_GPU_TESTS=1 uv run pytest tests/gpu/test_lora_serving.py -q -s

A random LoRA (r=64, the SFT target regex, lora_B ~ N(0, 0.02^2), seeded) is applied to google/gemma-3-4b-it (same
multimodal class and module names as the 27B), saved in PEFT format and merged with HF. `calign.inference.lora_check`
then compares vLLM base+LoRA vs vLLM merged vs vLLM base in subprocesses; `calign.train.export_text_only` exports the
merged model as Gemma3ForCausalLM and compares HF logits. Artefacts stay in CALIGN_TEST_OUT (default a tmp dir).
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from calign.paths import REPO_ROOT

pytestmark = pytest.mark.gpu

BASE = os.environ.get("CALIGN_TEST_BASE_MODEL", "google/gemma-3-4b-it")


@pytest.fixture(scope="module")
def adapter_and_merged(tmp_path_factory):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from calign.paths import hf_token
    from calign.train.merge import copy_processor_files
    from calign.train.sft import GEMMA3_LORA_TARGETS

    root = Path(os.environ.get("CALIGN_TEST_OUT") or tmp_path_factory.mktemp("lora"))
    adapter, merged = root / "adapter", root / "merged"
    if (merged / "config.json").exists():
        return root, adapter, merged
    tok = AutoTokenizer.from_pretrained(BASE, token=hf_token())
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, token=hf_token()).to("cuda")
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=64, target_modules=GEMMA3_LORA_TARGETS, bias="none"))
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_((torch.randn(p.shape, generator=g) * 0.02).to(p.dtype))
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    m = model.merge_and_unload()
    m.save_pretrained(merged, safe_serialization=True)
    tok.save_pretrained(merged)
    copy_processor_files(BASE, merged)
    del model, m
    gc.collect()
    torch.cuda.empty_cache()
    return root, adapter, merged


def test_peft_keys_are_multimodal_language_model(adapter_and_merged):
    from safetensors import safe_open

    _, adapter, _ = adapter_and_merged
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert keys and all(k.startswith("base_model.model.model.language_model.layers.") for k in keys), keys[:3]


def test_vllm_lora_serving_matches_merged(adapter_and_merged):
    root, adapter, merged = adapter_and_merged
    out = root / "lora_check.json"
    cmd = [sys.executable, "-m", "calign.inference.lora_check", "all", "--base", BASE, "--adapter", str(adapter)]
    cmd += ["--merged", str(merged), "--out", str(out)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(proc.stdout[-3000:], proc.stderr[-3000:])
    res = json.loads(out.read_text(encoding="utf-8"))
    print(json.dumps({k: res[k] for k in ("served_vs_merged", "served_vs_base")}, indent=1))
    assert res["passed"], res


def test_text_only_export_matches_multimodal(adapter_and_merged):
    root, _, merged = adapter_and_merged
    out = root / "merged_text"
    cmd = [sys.executable, "-m", "calign.train.export_text_only", "--model-path", str(merged), "--out", str(out)]
    cmd += ["--verify", "3"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(proc.stdout[-2000:], proc.stderr[-2000:])
    man = json.loads((out / "export_manifest.json").read_text(encoding="utf-8"))
    assert man["verify"]["class_text_only"] == "Gemma3ForCausalLM"
    assert man["verify"]["passed"], man["verify"]
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["architectures"] == ["Gemma3ForCausalLM"]
