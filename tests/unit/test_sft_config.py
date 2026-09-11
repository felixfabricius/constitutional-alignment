import re

import pytest

from calign.config import load_config
from calign.paths import CONFIGS_DIR
from calign.train.merge import copy_processor_files
from calign.train.sft import GEMMA3_LORA_TARGETS, SFTConfig, check_lora_targets


def test_sft_default_config_is_gemma3_r64():
    cfg = load_config(CONFIGS_DIR / "sft.yaml", SFTConfig)
    assert cfg.base_model == "google/gemma-3-27b-it" and cfg.attn_implementation == "sdpa"
    assert (cfg.lora.r, cfg.lora.alpha) == (64, 64)
    assert cfg.lora.target_modules == GEMMA3_LORA_TARGETS
    assert cfg.train.per_device_batch_size * cfg.train.gradient_accumulation_steps == 32


def test_sft_gemma2_config_kept():
    cfg = load_config(CONFIGS_DIR / "sft_gemma2_9b.yaml", SFTConfig)
    assert cfg.base_model == "google/gemma-2-9b-it" and cfg.attn_implementation == "eager"
    assert cfg.lora.r == 256 and isinstance(cfg.lora.target_modules, list)


def test_check_lora_targets():
    check_lora_targets(["base_model.model.model.language_model.layers.0.self_attn.q_proj"])
    with pytest.raises(ValueError, match="no modules"):
        check_lora_targets([])
    with pytest.raises(ValueError, match="non-text"):
        check_lora_targets(["base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj"])


@pytest.mark.hf
def test_gemma3_lora_regex_matches_language_linears_only():
    """Build Gemma 3 27B on the meta device (config download only) and apply PEFT's full-match rule."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from calign.paths import hf_token

    config = AutoConfig.from_pretrained("google/gemma-3-27b-it", token=hf_token())
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    names = [n for n, _ in model.named_modules()]
    hits = [n for n in names if re.fullmatch(GEMMA3_LORA_TARGETS, n)]
    assert len(hits) == config.get_text_config().num_hidden_layers * 7 == 434
    assert all(n.startswith("model.language_model.layers.") for n in hits)
    # the plain suffix list would also adapt the vision tower
    plain = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert any("vision_tower" in n for n in names if n.split(".")[-1] in plain)


def test_copy_processor_files_from_local_base(tmp_path):
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    out.mkdir()
    (base / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (base / "chat_template.jinja").write_text("base", encoding="utf-8")
    (out / "chat_template.jinja").write_text("tokenizer", encoding="utf-8")
    assert copy_processor_files(str(base), out) == ["preprocessor_config.json"]
    assert (out / "chat_template.jinja").read_text(encoding="utf-8") == "tokenizer"  # existing file kept
