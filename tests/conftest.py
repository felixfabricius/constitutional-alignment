"""Shared pytest configuration.

Markers:
  gpu  - requires CUDA (and the real model); skipped automatically when torch.cuda is unavailable,
         or when CALIGN_GPU_TESTS=0.
  api  - requires ANTHROPIC_API_KEY; skipped automatically when unset.
  hf   - requires HF_TOKEN (gated Gemma tokenizer download); skipped automatically when unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from calign.paths import load_env

load_env()

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _cuda_available() -> bool:
    if os.environ.get("CALIGN_GPU_TESTS") == "0":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="CUDA not available (set CALIGN_GPU_TESTS=1 on the GPU machine)")
    skip_api = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    skip_hf = pytest.mark.skip(reason="HF_TOKEN not set (gated Gemma tokenizer)")
    has_cuda = _cuda_available()
    has_api = bool(os.environ.get("ANTHROPIC_API_KEY"))
    # A token is needed for the gated google/ tokenizer; an explicit public mirror (CALIGN_TOKENIZER_ID) also works.
    has_hf = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("CALIGN_TOKENIZER_ID")
    )
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "api" in item.keywords and not has_api:
            item.add_marker(skip_api)
        if "hf" in item.keywords and not has_hf:
            item.add_marker(skip_hf)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
