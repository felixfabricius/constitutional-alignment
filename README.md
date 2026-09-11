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

### GPU machine runbook (Gemma 3 27B-IT, A100 80GB)

Defaults (`configs/model.yaml`, `configs/sft.yaml`) are Gemma 3 27B, so no `--model-config` / `--config` flags are
needed. Commands run on the instance unless marked `# local` (WSL, from the local repo root, with
`R=train-inst:/home/shadeform/constitutional-alignment` for the Brev instance).

**0. Setup and data** (`data/` and `outputs/` are gitignored, so they are copied, not pulled)

```bash
git pull && uv sync --group dev --group gpu && git submodule update --init
rsync -rtz ./data/sft/ $R/data/sft/                                    # local: training data
rsync -rtz ./data/scenarios/moralchoice_high.jsonl $R/data/scenarios/  # local: validation scenarios
```

**1. GPU smoke tests** on Gemma 3 4B-IT (same model class and turn format as the 27B, fast)

```bash
CALIGN_GPU_TESTS=1 CALIGN_MODEL_PATH=google/gemma-3-4b-it uv run pytest tests/gpu -q
# add CALIGN_VLLM_TESTS=1 to also check that vLLM generates the same greedy text as HF
```

**2. Base misalignment check** (done 2026-09-10: `outputs/misalignment/20260910_222307_9f28bd09` is the pre-SFT
baseline; rerun only if prompts or the model change)

```bash
uv run python -m calign.misalignment.run --dry-run            # then without --dry-run: 12 conditions x 25 samples
uv run python -m calign.misalignment.report --run-dir outputs/misalignment/<run>   # gate: meaningful_rate
```

**3. SFT and merge**

```bash
uv run python -m calign.train.sft --dry-run      # 3 steps; check gpu_peak_* in outputs/models/sft_pilot_dryrun/peft_summary.json
uv run python -m calign.train.sft                # LoRA r=64 on data/sft -> outputs/models/sft_pilot/adapter
uv run python -m calign.train.merge --adapter outputs/models/sft_pilot/adapter   # -> outputs/models/sft_pilot/merged (~55 GB)
```

**4. Misalignment check on the SFT model** (compare with the base run from step 2)

```bash
uv run python -m calign.misalignment.run --stage sft_merged --model-path outputs/models/sft_pilot/merged --constitution-judge
```

**5. Validation sampling** (both stages into ONE run dir; each call appends to `records.jsonl` and writes
`resolved_config_<stage>.yaml`; rerunning a stage into the same dir is refused, so use a fresh `--out` to redo one)

```bash
V=outputs/validation/gemma3_pilot
uv run python -m calign.validate.run_validation --stage base       --model-path google/gemma-3-27b-it          --out $V-dry --dry-run
uv run python -m calign.validate.run_validation --stage base       --model-path google/gemma-3-27b-it          --out $V
uv run python -m calign.validate.run_validation --stage sft_merged --model-path outputs/models/sft_pilot/merged --out $V
```

**6. Copy results back, then judge and report** (API only, so this can run locally)

```bash
rsync -rtz --exclude 'models/*/merged/' --exclude 'models/*/checkpoints/' $R/outputs/ ./outputs/   # local
uv run python -m calign.validate.judge  --run-dir outputs/validation/gemma3_pilot   # Claude judge (Batches)
uv run python -m calign.validate.report --run-dir outputs/validation/gemma3_pilot   # gate: recall_pass
uv run python -m calign.misalignment.report --run-dir outputs/misalignment/<sft run>
uv run python -m calign.misalignment.constitution_judge --run-dir outputs/misalignment/<run>   # add the 0-1 score to an existing run
```

Gemma 2 9B instead: add `--model-config configs/model_gemma2_9b.yaml` (sampling/validation) or
`--config configs/sft_gemma2_9b.yaml` (SFT).

## Phase 2 run order (probes and steering; settings in `configs/probe.yaml`, plan in `phase2_plan.md`)

The model is the selected SFT checkpoint, `configs/model_sft_v2e3.yaml` (HF repo + pinned revision). vLLM (sampling)
and HF (activations, steering) cannot share the GPU, so each line below is its own process. `M` and `P` as in:

```bash
M="--model-config configs/model_sft_v2e3.yaml"
P=outputs/probe_data/v2e3_k8
```

**1. Sample probe data** (GPU, vLLM; one invocation per prompt variant into the same run dir)

```bash
uv run python -m calign.probe.sample $M --variant none --out $P --dry-run
uv run python -m calign.probe.sample $M --variant none --out $P      # 384 definite-verdict scenarios (train + val) x 8
uv run python -m calign.probe.sample $M --variant full --out $P      # Probe C positives
```

**2. Judge and inspect** (local, Claude Batches; measure the per-item cost on `--limit 3 --no-batches` first)

```bash
uv run python -m calign.validate.judge --run-dir $P --config configs/probe.yaml
uv run python -m calign.probe.report --run-dir $P                    # 2x2 cells, label balance per spec
uv run python diagnostics/show_probe_cells.py --run-dir $P
```

**3. Hard-data scores** (local): the epoch-3 agentic run must carry `constitution-score-v2` on every sample; the base
run is re-scored for consistency.

```bash
uv run python -m calign.misalignment.constitution_judge --run-dir outputs/misalignment/20260911_153043_77860d1a
uv run python -m calign.misalignment.constitution_judge --run-dir outputs/misalignment/20260910_222307_9f28bd09
```

**4. Activations** (GPU, HF; rsync the judged `records.jsonl` forward first)

```bash
uv run python -m calign.probe.activations $M --run-dir $P --dry-run                 # prints the token at every position
uv run python -m calign.probe.activations $M --run-dir $P                           # activations/ shards, ~1.6 GB
uv run python -m calign.probe.activations $M --misalignment-run outputs/misalignment/20260911_153043_77860d1a
```

**5. Probes and evaluation** (local, CPU; rsync `activations/` back)

```bash
uv run python -m calign.probe.train --data-run $P --out outputs/probes/v2e3
uv run python -m calign.probe.evaluate --probes outputs/probes/v2e3 --hard outputs/misalignment/20260911_153043_77860d1a
uv run python -m calign.probe.sae --probes outputs/probes/v2e3                       # optional; downloads ~2.6 GB per layer
```

**6. Steering** (GPU, HF): tune on probe_val, then the main run on heldout_steer; judge and report each run locally.

```bash
uv run python -m calign.probe.steer $M --probes outputs/probes/v2e3 --purpose tuning --out outputs/steering/tuning --dry-run
uv run python -m calign.probe.steer $M --probes outputs/probes/v2e3 --purpose tuning --out outputs/steering/tuning
uv run python -m calign.validate.judge --run-dir outputs/steering/tuning --config configs/probe.yaml   # local
uv run python -m calign.probe.report --run-dir outputs/steering/tuning                                 # picks coefficients
uv run python -m calign.probe.steer $M --probes outputs/probes/v2e3 --purpose main --tuning-run outputs/steering/tuning --out outputs/steering/main
uv run python -m calign.validate.judge --run-dir outputs/steering/main --config configs/probe.yaml
uv run python -m calign.probe.report --run-dir outputs/steering/main
uv run python diagnostics/show_steering_samples.py --run-dir outputs/steering/main
```

GPU smoke test for the Phase 2 code on a 24 GB card: `CALIGN_GPU_TESTS=1 CALIGN_MODEL_PATH=google/gemma-3-4b-it uv run pytest tests/gpu/test_probe_gpu.py -q`.

Every run directory keeps the raw records (`samples.jsonl` / `records.jsonl`), `usage.json`, and a
`summary.json` with a provenance block; reports are recomputable from the raw files.

## Tests

```bash
uv run pytest tests/unit          # fast, local
uv run pytest tests/api           # one real Claude call per test
uv run pytest tests/gpu           # on the GPU machine
```
