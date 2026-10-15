import sys
import types

from calign.inference.backend import default_attn_implementation, load_model_config
from calign.inference.hf_backend import hidden_state_index
from calign.paths import CONFIGS_DIR


def test_default_model_config_is_gemma3_27b():
    cfg = load_model_config()
    assert cfg.model_id == cfg.model_path == "google/gemma-3-27b-it"
    assert cfg.attn_implementation == "sdpa"
    assert cfg.max_model_len == 8192
    assert cfg.vllm.language_model_only is True
    assert cfg.probe_layers == [16, 31, 40, 53]


def test_attn_and_layer_helpers():
    assert default_attn_implementation("gemma2") == "eager"
    assert default_attn_implementation("gemma3") == "sdpa"
    assert hidden_state_index(16) == 17  # Gemma Scope resid_post of block 16


class _FakeTokenizer:
    eos_token_id = 1
    unk_token_id = 3

    def convert_tokens_to_ids(self, tok):
        return 106


def _fake_vllm(monkeypatch):
    calls = []

    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_tokenizer(self):
            return _FakeTokenizer()

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM))
    return calls


def test_vllm_backend_passes_language_model_only_only_when_set(monkeypatch):
    from calign.inference.vllm_backend import VLLMBackend

    calls = _fake_vllm(monkeypatch)
    no_lmo = load_model_config(CONFIGS_DIR / "model.yaml")
    no_lmo = no_lmo.model_copy(update={"vllm": no_lmo.vllm.model_copy(update={"language_model_only": False})})
    VLLMBackend(no_lmo)
    VLLMBackend(load_model_config(CONFIGS_DIR / "model.yaml"))
    assert "language_model_only" not in calls[0]
    assert calls[1]["language_model_only"] is True
    assert calls[1]["model"] == "google/gemma-3-27b-it"
    assert "revision" not in calls[1]
    VLLMBackend(load_model_config(CONFIGS_DIR / "model.yaml", revision="deadbeef"))
    assert calls[2]["revision"] == "deadbeef"
