import sys
import types

from calign.config import load_config
from calign.inference.backend import load_model_config
from calign.paths import CONFIGS_DIR, REPO_ROOT
from calign.train.sft import SFTConfig

LEGACY_CONFIGS_DIR = REPO_ROOT / "legacy" / "configs"


def test_gemma2_9b_config_kept():
    cfg = load_model_config(LEGACY_CONFIGS_DIR / "model_gemma2_9b.yaml")
    assert cfg.model_id == cfg.model_path == "google/gemma-2-9b-it"
    assert cfg.attn_implementation == "eager"
    assert cfg.vllm.language_model_only is False
    assert cfg.probe_layers == [9, 20, 31]


def test_sft_v2e3_config_and_revision_override():
    cfg = load_model_config(LEGACY_CONFIGS_DIR / "model_sft_v2e3.yaml")
    assert cfg.model_path == cfg.model_id == "felixfabricius/gemma-3-27b-it-halden-sft-v2-epoch3"
    assert cfg.revision == "cfd5052f36548f429f7a0f3698329bb03a44d365"  # final epoch-3 upload, pinned
    assert cfg.vllm.language_model_only and cfg.probe_layers == [16, 31, 40, 53]
    cfg = load_model_config(LEGACY_CONFIGS_DIR / "model_sft_v2e3.yaml", revision="abc123")
    assert cfg.revision == "abc123" and cfg.model_dump()["revision"] == "abc123"  # lands in resolved_config.yaml


def test_sft_gemma2_config_kept():
    cfg = load_config(LEGACY_CONFIGS_DIR / "sft_gemma2_9b.yaml", SFTConfig)
    assert cfg.base_model == "google/gemma-2-9b-it" and cfg.attn_implementation == "eager"
    assert cfg.lora.r == 256 and isinstance(cfg.lora.target_modules, list)


class _FakeTokenizer:
    eos_token_id = 1
    unk_token_id = 3

    def convert_tokens_to_ids(self, tok):
        return 106


def test_vllm_backend_omits_language_model_only_for_gemma2(monkeypatch):
    from calign.inference.vllm_backend import VLLMBackend

    calls = []

    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_tokenizer(self):
            return _FakeTokenizer()

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM))
    VLLMBackend(load_model_config(LEGACY_CONFIGS_DIR / "model_gemma2_9b.yaml"))
    VLLMBackend(load_model_config(CONFIGS_DIR / "model.yaml"))
    assert "language_model_only" not in calls[0]
    assert calls[1]["language_model_only"] is True


def test_probe_config_points_at_legacy_model_config():
    from calign_legacy.probe.config import DEFAULT_CONFIG, load_probe_config

    cfg = load_probe_config(DEFAULT_CONFIG)
    assert (REPO_ROOT / cfg.model_config_path).is_file()
