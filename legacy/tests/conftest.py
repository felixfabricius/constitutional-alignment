"""Same markers and automatic skips as tests/conftest.py (gpu / api / hf), reused rather than copied.

Run with `uv run pytest legacy/tests/unit` (tests/ and legacy/tests share no module basenames, so both can be
collected in one session too).
"""

from __future__ import annotations

import importlib.util
import sys

from calign.paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location("_calign_root_conftest", REPO_ROOT / "tests" / "conftest.py")
_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root)

pytest_collection_modifyitems = _root.pytest_collection_modifyitems

# test_probe_steer / test_misalignment_baselines reuse the tiny-model helpers of tests/unit/test_steering_hook.py
sys.path.insert(0, str(REPO_ROOT / "tests" / "unit"))
