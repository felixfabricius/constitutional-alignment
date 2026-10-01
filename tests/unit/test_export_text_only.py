import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from calign.inference import lora_check
from calign.train import export_text_only as ex


@pytest.mark.parametrize(
    "key,tied,expected",
    [
        ("model.language_model.layers.0.self_attn.q_proj.weight", True, "model.layers.0.self_attn.q_proj.weight"),
        ("language_model.model.embed_tokens.weight", True, "model.embed_tokens.weight"),
        ("model.language_model.norm.weight", True, "model.norm.weight"),
        ("lm_head.weight", True, None),
        ("lm_head.weight", False, "lm_head.weight"),
        ("language_model.lm_head.weight", False, "lm_head.weight"),
        ("model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight", True, None),
        ("multi_modal_projector.mm_input_projection_weight", True, None),
    ],
)
def test_map_key(key, tied, expected):
    assert ex.map_key(key, tied) == expected


def test_map_key_rejects_unknown():
    with pytest.raises(ValueError):
        ex.map_key("something.else.weight", True)


def test_text_config():
    mm = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "torch_dtype": "bfloat16",
        "text_config": {"hidden_size": 8, "model_type": "gemma3_text"},
        "vision_config": {},
    }
    tc = ex.text_config(mm)
    assert tc == {
        "hidden_size": 8,
        "model_type": "gemma3_text",
        "architectures": ["Gemma3ForCausalLM"],
        "torch_dtype": "bfloat16",
    }
    with pytest.raises(ValueError):
        ex.text_config(tc)


def test_export_rewrites_shards(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    a = {
        "model.language_model.layers.0.mlp.up_proj.weight": torch.ones(2, 2),
        "model.vision_tower.x.weight": torch.zeros(1),
    }
    b = {"model.language_model.embed_tokens.weight": torch.full((3, 2), 2.0), "lm_head.weight": torch.ones(3, 2)}
    save_file(a, str(src / "model-00001-of-00002.safetensors"))
    save_file(b, str(src / "model-00002-of-00002.safetensors"))
    wm = {k: "model-00001-of-00002.safetensors" for k in a} | {k: "model-00002-of-00002.safetensors" for k in b}
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}), encoding="utf-8")
    (src / "config.json").write_text(json.dumps({"text_config": {"model_type": "gemma3_text"}}), encoding="utf-8")
    for name in ("tokenizer.json", "generation_config.json", "preprocessor_config.json"):
        (src / name).write_text("{}", encoding="utf-8")

    counts = ex.export(src, out)
    assert counts["n_tensors"] == 2 and counts["n_dropped"] == 2 and counts["tied_embeddings"]
    assert sorted(counts["copied_files"]) == ["generation_config.json", "tokenizer.json"]
    index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"] == {
        "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
        "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
    }
    assert index["metadata"]["total_size"] == (4 + 6) * 4
    with safe_open(str(out / "model-00002-of-00002.safetensors"), framework="pt") as f:
        assert list(f.keys()) == ["model.embed_tokens.weight"]
        assert torch.equal(f.get_tensor("model.embed_tokens.weight"), b["model.language_model.embed_tokens.weight"])
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["architectures"] == ["Gemma3ForCausalLM"]
    assert not (out / "preprocessor_config.json").exists()


def _run(lps, greedy):
    return {"rows": [{"logprobs": lp, "greedy_ids": g} for lp, g in zip(lps, greedy, strict=True)]}


def test_lora_check_compare():
    served = _run([[-1.0, -2.0], [-0.5]], [[1, 2, 3], [4, 5]])
    merged = _run([[-1.01, -2.01], [-0.51]], [[1, 2, 3], [4, 6]])
    base = _run([[-2.0, -3.0], [-1.5]], [[7, 8], [4]])
    res = lora_check.compare(served, merged, base)
    assert res["passed"] and res["applied"] and res["faithful"]
    assert res["served_vs_merged"]["greedy_common_prefix"] == [3, 1]
    assert res["served_vs_merged"]["greedy_identical"] == [True, False]
    assert res["served_vs_base"]["mean_abs_logprob_diff"] == pytest.approx(1.0)
    # an adapter that was silently not applied looks like the base model -> fails the ratio test
    assert not lora_check.compare(base, merged, base)["passed"]


def test_lora_check_hf_reference_sets_the_noise_floor():
    hf = _run([[-1.0, -2.0], [-0.5]], [[], []])
    merged = _run([[-1.1, -2.1], [-0.6]], [[1, 2], [3]])  # bf16 merge noise 0.1 vs the PEFT reference
    served = _run([[-0.92, -1.92], [-0.42]], [[1, 2], [3]])  # 0.08 from PEFT, 0.18 from merged
    base = _run([[-3.0, -4.0], [-2.5]], [[9], [9]])
    res = lora_check.compare(served, merged, base, hf)
    assert "greedy_common_prefix" not in res["served_vs_hf_peft"]
    assert res["faithful_bound"] == pytest.approx(0.15) and res["faithful"] and res["passed"]
    assert not lora_check.compare(served, merged, base)["faithful"]  # without the reference: 0.18 > 0.05
    far = _run([[-0.6, -1.6], [-0.1]], [[1], [3]])  # 0.4 from PEFT: a real mapping error
    assert not lora_check.compare(far, merged, base, hf)["faithful"]


def test_module_type_adapters(tmp_path):
    w = {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 3),
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(3, 2),
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight": torch.ones(2, 3),
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight": torch.ones(3, 2),
    }
    (tmp_path / "a").mkdir()
    save_file(w, str(tmp_path / "a" / "adapter_model.safetensors"))
    (tmp_path / "a" / "adapter_config.json").write_text('{"r": 2}', encoding="utf-8")
    out = lora_check.module_type_adapters(tmp_path / "a", tmp_path / "cov")
    assert sorted(out) == ["down_proj", "q_proj"]
    with safe_open(str(out["q_proj"] / "adapter_model.safetensors"), framework="pt") as f:
        q_b = f.get_tensor("base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight")
        d_b = f.get_tensor("base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight")
        d_a = f.get_tensor("base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight")
    assert q_b.sum() == 6 and d_b.sum() == 0 and d_a.sum() == 6
