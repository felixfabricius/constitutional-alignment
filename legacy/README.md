# legacy: the original probe-and-steer project

This folder holds the code and records of the project's **original goal**, kept runnable but outside the main
`calign` package. Nothing in `src/calign` imports from here; the blog experiments do not depend on it.

The project started as *"Detecting and strengthening constitutional reasoning in LLMs"*: instil a constitution in a
model, find a linear representation of "reasoning from the constitution" with probes, check it with steering and
Gemma Scope SAEs, then use the probe as an RL signal. Hard held-out data came from Anthropic's
[agentic-misalignment](https://github.com/anthropic-experimental/agentic-misalignment) scenarios (blackmail, leaking,
murder). Phase 2 showed the observables were at a ceiling (the SFT model names its constitution in ~95% of answers),
the probes did not transfer to the agentic data, and positive steering did nothing. The project was re-scoped into
"alignment under a budget" (Phase 3, the blog post); see [`docs/README.md`](../docs/README.md) for the full history.

## What is here

| Path | Contents |
|---|---|
| `calign_legacy/misalignment/` | Agentic-misalignment harness: upstream prompts and classifiers (via the `third_party/` submodule), constitution score, coherence judge, run comparison, constitution system-prompt variant and probe steering during generation |
| `calign_legacy/probe/` | Phase 2: probe data sampling, label specs, forced-pass activations, difference-of-means probes, evaluation on the agentic data, steering, Gemma Scope SAE lookups |
| `configs/` | `probe.yaml`, `misalignment_check.yaml`, `model_sft_v2e3.yaml` (the Phase 2 model, SFT v2 epoch 3), the Gemma 2 9B model/SFT configs |
| `diagnostics/` | Print-only inspection scripts for the above (probe cells and positions, steering samples, SAE features, misalignment prompts/samples, token positions) |
| `tests/` | Unit, GPU and live-API tests for the above (`uv run pytest legacy/tests/unit`) |
| `docs/` | `phase2_plan.md`, `phase2_runs.md` (all Phase 2 numbers), `priority1_plan.md` + `priority1_runs.md` (agentic baselines), `details.md` (low-level facts) |

`calign_legacy` is installed alongside `calign` (`pyproject.toml`), so the CLIs run as `uv run python -m
calign_legacy.<package>.<module>`. Default config paths point at `legacy/configs/`. Outputs still go to the shared
`outputs/` tree (`outputs/misalignment`, `outputs/probe_data`, `outputs/probes`, `outputs/steering`, ...).

## Results in brief

- **Model choice (2026-09-10).** Gemma 2 9B-IT was too weak to act misaligned (leaking headline 4/25 only); Gemma 3
  27B-IT acts misaligned clearly (leaking headline 22/25, murder headline 6/25; run `20260910_222307_9f28bd09`).
- **SFT pilot on the agentic data (2026-09-11).** Leaking dropped (explicit goal 38/50 -> 23/46, p = 0.011) while
  murder rose in no-goal conditions (1/50 -> 9/50, p = 0.016); the SFT model invoked the constitution in ~55% of long
  agentic responses, often with confabulated content.
- **Phase 2 probes and steering (2026-09-11, SFT v2 epoch 3).** Best probe `B_primary/L53/p100` val AUROC 0.80;
  the in-context-constitution probe saturates at 1.000. Positive steering does nothing on held-out MoralChoice
  (alignment already 0.89, mentions 97%); negative steering lowers alignment 0.89 -> 0.82 (-0.07 [-0.13, -0.02]).
  Probes do not transfer to the agentic data (AUROC ~0.5-0.58). Massive-activation dims 104 / 2733 dominate 55 of
  120 directions. Details: [`docs/phase2_runs.md`](docs/phase2_runs.md).
- **Priority-1 agentic baselines (2026-09-12).** The constitution in the agentic system prompt halves harm (11.0% ->
  5.4%); +B steering changes nothing (33/300 vs 33/300) and lowers judged coherence (0.85 -> 0.72). Details:
  [`docs/priority1_runs.md`](docs/priority1_runs.md).

## Runbook: agentic-misalignment check

Settings in `configs/misalignment_check.yaml` (12 conditions x 25 samples, T = 1.0). The pre-SFT baseline is
`outputs/misalignment/20260910_222307_9f28bd09`.

```bash
git submodule update --init      # third_party/agentic-misalignment
uv run python -m calign_legacy.misalignment.run --dry-run            # then without --dry-run
uv run python -m calign_legacy.misalignment.report --run-dir outputs/misalignment/<run>   # gate: meaningful_rate
uv run python -m calign_legacy.misalignment.run --stage sft_merged --model-path outputs/models/sft_pilot/merged --constitution-judge
uv run python -m calign_legacy.misalignment.constitution_judge --run-dir outputs/misalignment/<run>   # add the 0-1 score to an existing run
```

Gemma 2 9B instead: add `--model-config legacy/configs/model_gemma2_9b.yaml` (sampling / validation) or
`--config legacy/configs/sft_gemma2_9b.yaml` (`calign.train.sft`).

## Runbook: Phase 2 probes and steering (probes and steering; settings in `legacy/configs/probe.yaml`, plan in `legacy/docs/phase2_plan.md`)

The model is the selected SFT checkpoint, `legacy/configs/model_sft_v2e3.yaml` (HF repo + pinned revision). vLLM (sampling)
and HF (activations, steering) cannot share the GPU, so each line below is its own process. `M` and `P` as in:

```bash
M="--model-config legacy/configs/model_sft_v2e3.yaml"
P=outputs/probe_data/v2e3_k8
```

**1. Sample probe data** (GPU, vLLM; one invocation per prompt variant into the same run dir)

```bash
uv run python -m calign_legacy.probe.sample $M --variant none --out $P --dry-run
uv run python -m calign_legacy.probe.sample $M --variant none --out $P      # 384 definite-verdict scenarios (train + val) x 8
uv run python -m calign_legacy.probe.sample $M --variant full --out $P      # Probe C positives
```

**2. Judge and inspect** (local, Claude Batches; measure the per-item cost on `--limit 3 --no-batches` first)

```bash
uv run python -m calign.validate.judge --run-dir $P --config legacy/configs/probe.yaml
uv run python -m calign_legacy.probe.report --run-dir $P                    # 2x2 cells, label balance per spec
uv run python legacy/diagnostics/show_probe_cells.py --run-dir $P
```

**3. Hard-data scores** (local; **done 2026-09-11**, $6.90 with Batches: both runs carry `constitution-score-v2` on
299/300 samples, 1 Claude refusal each; base v1 scores kept as `samples.constitution-v1.jsonl`). The epoch-3 agentic
run must carry `constitution-score-v2` on every sample; the base run is re-scored for consistency.

```bash
uv run python -m calign_legacy.misalignment.constitution_judge --run-dir outputs/misalignment/20260911_153043_77860d1a
uv run python -m calign_legacy.misalignment.constitution_judge --run-dir outputs/misalignment/20260910_222307_9f28bd09
```

**4. Activations** (GPU, HF; rsync the judged `records.jsonl` forward first)

```bash
uv run python -m calign_legacy.probe.activations $M --run-dir $P --dry-run                 # prints the token at every position
uv run python -m calign_legacy.probe.activations $M --run-dir $P                           # activations/ shards, ~1.6 GB
uv run python -m calign_legacy.probe.activations $M --misalignment-run outputs/misalignment/20260911_153043_77860d1a
```

**5. Probes and evaluation** (local, CPU; rsync `activations/` back)

```bash
uv run python -m calign_legacy.probe.train --data-run $P --out outputs/probes/v2e3
uv run python -m calign_legacy.probe.evaluate --probes outputs/probes/v2e3 --hard outputs/misalignment/20260911_153043_77860d1a
uv run python -m calign_legacy.probe.sae --probes outputs/probes/v2e3                       # optional; downloads ~2.6 GB per layer
```

**6. Steering** (GPU, HF): tune on probe_val, then the main run on heldout_steer; judge and report each run locally.

```bash
uv run python -m calign_legacy.probe.steer $M --probes outputs/probes/v2e3 --purpose tuning --out outputs/steering/tuning --dry-run
uv run python -m calign_legacy.probe.steer $M --probes outputs/probes/v2e3 --purpose tuning --out outputs/steering/tuning
uv run python -m calign.validate.judge --run-dir outputs/steering/tuning --config legacy/configs/probe.yaml   # local
uv run python -m calign_legacy.probe.report --run-dir outputs/steering/tuning                                 # picks coefficients
uv run python -m calign_legacy.probe.steer $M --probes outputs/probes/v2e3 --purpose main --tuning-run outputs/steering/tuning --out outputs/steering/main
uv run python -m calign.validate.judge --run-dir outputs/steering/main --config legacy/configs/probe.yaml
uv run python -m calign_legacy.probe.report --run-dir outputs/steering/main
uv run python legacy/diagnostics/show_steering_samples.py --run-dir outputs/steering/main
```

GPU smoke test for the Phase 2 code on a 24 GB card: `CALIGN_GPU_TESTS=1 CALIGN_MODEL_PATH=google/gemma-3-4b-it uv run pytest legacy/tests/gpu/test_probe_gpu.py -q`.

Notes from the 2026-09-11 run (all results and run dirs: [`docs/phase2_runs.md`](docs/phase2_runs.md)): the steering CLIs need
`--batch-size 10..13` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the 80 GB A100 (batch 16 plus
degenerate 2048-token outputs hits allocator OOM warnings); `calign_legacy.probe.activations` writes
`activations/refs.jsonl` and leaves `records.jsonl` / `samples.jsonl` alone, so judging and constitution scoring can
run locally at the same time (`calign_legacy.probe.merge_records` folds a partial judged copy back in); steering judges use
`--no-batches` (505 records, ~3 min, $2.90) while the 6144 probe-data records go through Batches ($17.88).

Every run directory keeps the raw records (`samples.jsonl` / `records.jsonl`), `usage.json`, and a
`summary.json` with a provenance block; reports are recomputable from the raw files.

## Runbook: priority-1 agentic baselines (constitution in the system prompt; probe steering)

Two inference-time baselines for the selected SFT checkpoint on the agentic-misalignment scenarios, compared with the
existing base and epoch-3 runs. Same protocol as every agentic run (`legacy/configs/misalignment_check.yaml`: 12 conditions x
25 samples, T=1.0, max_tokens 4000). Results and run dirs: [`docs/priority1_runs.md`](docs/priority1_runs.md). `M` and `STEER` as in:

```bash
M="--model-config legacy/configs/model_sft_v2e3.yaml --stage sft_merged --skip-classify"
STEER="--steer-probes outputs/probes/v2e3 --steer-probe B_primary/L53/p100 --steer-coef 4 --steer-sign 1 --steer-positions all"
```

**1. Constitution in the agentic system prompt** (GPU, vLLM). The Phase 1 constitution preface (governed-by line, full
text, cite-by-number instruction; not the MoralChoice final-answer instruction) is prepended to the unchanged upstream
system prompt; condition ids stay the same and the run dir saves the modified prompts (the judges sha-check them).

```bash
uv run python -m calign_legacy.misalignment.run $M --backend vllm --system-prompt-variant constitution --dry-run --limit 1
uv run python -m calign_legacy.misalignment.run $M --backend vllm --system-prompt-variant constitution --out outputs/misalignment/e3_constprompt
```

**2. Probe steering** (GPU, HF). `+4` class gaps of the best Probe B direction (`B_primary/L53/p100`, the coefficient
tuned on MoralChoice probe_val; not re-tuned on the agentic set) added at every position during generation. Steered
generations need the HF backend, and HF and vLLM sampling differ numerically, so the comparison is an **HF control**
run under the same command without `--steer-*`. The dry run prints a steered and an unsteered response side by side.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python -m calign_legacy.misalignment.run $M --backend hf $STEER --dry-run --limit 1 --batch-size 2
uv run python -m calign_legacy.misalignment.run $M --backend hf --batch-size 12 --out outputs/misalignment/e3_hf_control
uv run python -m calign_legacy.misalignment.run $M --backend hf $STEER --batch-size 12 --out outputs/misalignment/e3_hf_steerB4
```

**3. Judge every new run** (local; rsync the run dir back first): upstream classifiers (interactive, ~$2.2), constitution
score (`constitution-score-v2`, Batches, ~$2.4), and the comprehensibility judge (`coherence-v1`, response text only,
effort low, ~$1.5 per 300 interactive; empty judge outputs are re-asked once). The coherence judge was also run on the
base and epoch-3 runs so every row of the comparison has it.

```bash
uv run python -m calign_legacy.misalignment.run --classify-only outputs/misalignment/<run>
uv run python -m calign_legacy.misalignment.constitution_judge --run-dir outputs/misalignment/<run>
uv run python -m calign_legacy.misalignment.coherence_judge --run-dir outputs/misalignment/<run> --no-batches
```

**4. Compare** (local): one table across runs (harmful rate with Wilson CI, constitution score with bootstrap CI,
mention rate, coherence judge, repetition, truncation, U+FFFD), differences against a reference run (Newcombe /
bootstrap intervals) and a provenance block with every input's `samples.jsonl` sha.

```bash
uv run python -m calign_legacy.misalignment.compare --reference e3_hf_control --out outputs/misalignment_compare/p1 --runs \
  base=outputs/misalignment/20260910_222307_9f28bd09 e3_vllm=outputs/misalignment/20260911_153043_77860d1a \
  e3_hf_control=outputs/misalignment/e3_hf_control e3_constprompt=outputs/misalignment/e3_constprompt \
  e3_steerB4=outputs/misalignment/e3_hf_steerB4
```

Notes from the 2026-09-12 run: HF sampling of the 27B at batch 12 on ~3k-token prompts peaks at ~70 GB allocated
(`HFBackend.generate` halves a batch and retries on CUDA OOM, so a too-large batch costs time, not the run); the vLLM
constitution-prompt run (prompts 2.9-3.5k tokens) takes ~10 min for 300 samples.
