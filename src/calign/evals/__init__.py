"""Phase 3 evaluation suite: eval configurations (C0..C4), MoralChoice and budget components, suite and report."""

from calign.paths import load_env

# Frozen evaluation suite (chunk 4, 2026-10-02): components, datasets, manifests and prompt versions are listed in
# data/manifests/suite_p3-v1.json (`python -m calign.evals.freeze`) and phase3/README.md Section 10. Every eval,
# scenario and judge run records it in run_meta.json. A change to any frozen part bumps this and re-runs C0.
SUITE_VERSION = "p3-v1"

# Every evals CLI loads gated models (HF_TOKEN) or calls Claude (ANTHROPIC_API_KEY): read .env once on import, before
# vLLM / transformers look for the token.
load_env()
