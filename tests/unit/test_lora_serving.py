import json
import sys
import types

import pytest

from calign.evals.config import EvalConfig, resolve_model
from calign.inference import lora


def test_parse_hf_spec():
    assert lora.parse_hf_spec("hf://felix/repo/adapter_epoch2@abc123") == ("felix/repo", "adapter_epoch2", "abc123")
    assert lora.parse_hf_spec("hf://felix/repo/adapter_epoch2") == ("felix/repo", "adapter_epoch2", None)
    assert lora.parse_hf_spec("hf://felix/repo@main") == ("felix/repo", None, "main")
    assert lora.parse_hf_spec("hf://felix/repo/a/b") == ("felix/repo", "a/b", None)
    with pytest.raises(ValueError):
        lora.parse_hf_spec("felix/repo")


def test_vllm_rank_rounding():
    assert lora.vllm_max_lora_rank(64) == 64
    assert lora.vllm_max_lora_rank(48) == 64
    assert lora.vllm_max_lora_rank(4) == 8
    with pytest.raises(ValueError):
        lora.vllm_max_lora_rank(1024)


def test_check_text_only():
    lora.check_text_only({"target_modules": r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k)_proj)"})
    lora.check_text_only({"target_modules": ["q_proj", "v_proj"]})
    with pytest.raises(ValueError):
        lora.check_text_only({"target_modules": ["vision_tower.encoder.q_proj"]})


def test_resolve_local_adapter(tmp_path):
    with pytest.raises(FileNotFoundError):
        lora.resolve_adapter(str(tmp_path))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 64}), encoding="utf-8")
    assert lora.resolve_adapter(str(tmp_path)) == tmp_path
    assert lora.adapter_config(tmp_path)["r"] == 64


def test_resolve_hf_adapter_downloads_only_the_subdir(tmp_path, monkeypatch):
    calls = {}

    def fake_snapshot_download(repo, revision=None, allow_patterns=None, token=None, repo_type=None):
        calls.update(repo=repo, revision=revision, allow_patterns=allow_patterns)
        (tmp_path / "adapter_epoch2").mkdir()
        (tmp_path / "adapter_epoch2" / "adapter_config.json").write_text("{}", encoding="utf-8")
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    p = lora.resolve_adapter("hf://felix/repo/adapter_epoch2@abc")
    assert p == tmp_path / "adapter_epoch2"
    assert calls == {"repo": "felix/repo", "revision": "abc", "allow_patterns": ["adapter_epoch2/*"]}


def test_eval_config_keeps_hf_adapter_spec():
    cfg = EvalConfig(id="C2@e2", adapter="hf://felix/repo/adapter_epoch2@abc", stage="sft")
    _, adapter, _ = resolve_model(cfg)
    assert adapter == "hf://felix/repo/adapter_epoch2@abc"


def test_process_exit_code():
    from calign.inference.process import exit_code

    assert exit_code(SystemExit()) == 0
    assert exit_code(SystemExit(3)) == 3
    assert exit_code(SystemExit("message")) == 1
