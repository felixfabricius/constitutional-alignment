"""Phase 3 evaluation suite: eval configurations (C0..C4), MoralChoice and budget components, suite and report."""

from calign.paths import load_env

# Every evals CLI loads gated models (HF_TOKEN) or calls Claude (ANTHROPIC_API_KEY): read .env once on import, before
# vLLM / transformers look for the token.
load_env()
