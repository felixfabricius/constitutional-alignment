"""Repository path layout and environment loading.

All scripts resolve paths relative to the repository root so they behave the same
locally, on Colab, and on the rented GPU machine.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
CONSTITUTION_PATH = REPO_ROOT / "constitution.md"

SCENARIOS_DIR = DATA_DIR / "scenarios"
MANIFESTS_DIR = DATA_DIR / "manifests"
CACHE_DIR = DATA_DIR / "cache"
SFT_DATA_DIR = DATA_DIR / "sft"
CORPUS_DIR = DATA_DIR / "corpus"

AGENTIC_MISALIGNMENT_DIR = THIRD_PARTY_DIR / "agentic-misalignment"

_ENV_LOADED = False


def load_env() -> None:
    """Load `.env` from the repo root once (API keys, HF token)."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(REPO_ROOT / ".env", override=False)
        _ENV_LOADED = True


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_token() -> str | None:
    load_env()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def anthropic_api_key() -> str | None:
    load_env()
    return os.environ.get("ANTHROPIC_API_KEY")
