# constitutional-alignment

Detecting and strengthening constitutional reasoning in LLMs. Phase 1: instil the constitution in
[constitution.md](constitution.md) into `google/gemma-2-9b-it` via LoRA SFT on a synthetic corpus,
merge, and validate recall/application.

## Setup

```bash
uv sync --group dev              # local (Windows/macOS/Linux): tests, data prep, Claude API steps
uv sync --group dev --group gpu  # GPU machine (Linux): adds vLLM
git submodule update --init      # third_party/agentic-misalignment
```

`.env` at the repo root (never committed):

```
ANTHROPIC_API_KEY=...   # corpus generation, classifiers, judges (claude-sonnet-5)
HF_TOKEN=...            # gated Gemma 2 weights/tokenizer
```

## Layout

- `src/calign/` package (config, schemas, constitution, prompting, data, misalignment, corpus, train, validate)
- `configs/` YAML configs; every CLI accepts `--config`, `--dry-run`, `--limit`, `--seed`, `--out`, `--model-path`
- `tests/unit` local; `tests/api` needs `ANTHROPIC_API_KEY`; `tests/gpu` needs CUDA (auto-skipped otherwise)
- `diagnostics/` print-only scripts for manual inspection
- `data/` (gitignored except `data/manifests/`) and `outputs/` (gitignored) hold raw records and run directories

## Tests

```bash
uv run pytest tests/unit          # fast, local
uv run pytest tests/api           # one real Claude call per test
uv run pytest tests/gpu           # on the GPU machine
```
