# constitutional-alignment

Detecting and strengthening constitutional reasoning in LLMs. Phase 1: instil the constitution in
[constitution.md](constitution.md) into `google/gemma-3-27b-it` via LoRA SFT on a synthetic corpus,
merge, and validate recall/application. (The first base-model check used `google/gemma-2-9b-it`, which was too
weak to act misaligned; its configs are kept as `configs/*_gemma2_9b.yaml`.)

## Setup

```bash
uv sync --group dev              # local (Windows/macOS/Linux): tests, data prep, Claude API steps
uv sync --group dev --group gpu  # GPU machine (Linux): adds vLLM
git submodule update --init      # third_party/agentic-misalignment
```

`.env` at the repo root (never committed):

```
ANTHROPIC_API_KEY=...   # corpus generation, classifiers, judges (claude-sonnet-5)
HF_TOKEN=...            # gated Gemma 3 (and Gemma 2) weights/tokenizers
```

## Layout

- `src/calign/` package (config, schemas, constitution, prompting, data, misalignment, corpus, train, validate)
- `configs/` YAML configs; every CLI accepts `--config`, `--dry-run`, `--limit`, `--seed`, `--out`, `--model-path`
- `tests/unit` local; `tests/api` needs `ANTHROPIC_API_KEY`; `tests/gpu` needs CUDA (auto-skipped otherwise)
- `diagnostics/` print-only scripts for manual inspection
- `data/` (gitignored except `data/manifests/`) and `outputs/` (gitignored) hold raw records and run directories

## Phase 1 run order

Local (API only):

```bash
uv run python -m calign.data.moralchoice                      # 680 scenarios -> data/scenarios + committed manifest
uv run python -m calign.validate.verdicts                     # constitution verdicts for every scenario (Batches)
uv run python -m calign.corpus.generate_docs                  # pilot docs (Batches); --dry-run --no-batches for 3 items
uv run python -m calign.corpus.generate_transcripts           # pilot transcripts (Batches)
uv run python -m calign.corpus.build_sft_dataset --tokenizer google/gemma-3-27b-it
uv run python diagnostics/show_corpus_samples.py              # eyeball accepted/rejected items
```

GPU machine (A100 80GB), in order. `data/` is gitignored: copy `data/sft/` (training) and
`data/scenarios/moralchoice_high.jsonl` (validation; or rerun `calign.data.moralchoice` there) to the machine first,
and copy `outputs/` back afterwards (e.g. `rsync -rtz train-inst:<repo>/outputs/ ./outputs/`).

```bash
uv sync --group dev --group gpu && git submodule update --init
CALIGN_GPU_TESTS=1 CALIGN_MODEL_PATH=google/gemma-3-4b-it uv run pytest tests/gpu -q   # 4B: same class, fast
uv run python -m calign.misalignment.run --dry-run            # then without --dry-run: 12 conditions x 25 samples
uv run python -m calign.misalignment.report --run-dir outputs/misalignment/<run>   # gate: meaningful_rate
uv run python -m calign.train.sft --dry-run                   # 3 steps; check peak memory in peft_summary.json
uv run python -m calign.train.sft                             # LoRA r=64 on data/sft (configs/sft.yaml)
uv run python -m calign.train.merge --adapter outputs/models/sft_pilot/adapter
uv run python -m calign.misalignment.run --stage sft_merged --model-path outputs/models/sft_pilot/merged
uv run python -m calign.validate.run_validation --stage base       --model-path google/gemma-3-27b-it       --out outputs/validation/<run>
uv run python -m calign.validate.run_validation --stage sft_merged --model-path outputs/models/sft_pilot/merged --out outputs/validation/<run>
uv run python -m calign.validate.judge  --run-dir outputs/validation/<run>
uv run python -m calign.validate.report --run-dir outputs/validation/<run>       # gate: recall_pass
```

Gemma 2 9B instead: add `--model-config configs/model_gemma2_9b.yaml` (sampling/validation) or
`--config configs/sft_gemma2_9b.yaml` (SFT).

Every run directory keeps the raw records (`samples.jsonl` / `records.jsonl`), `usage.json`, and a
`summary.json` with a provenance block; reports are recomputable from the raw files.

## Tests

```bash
uv run pytest tests/unit          # fast, local
uv run pytest tests/api           # one real Claude call per test
uv run pytest tests/gpu           # on the GPU machine
```
