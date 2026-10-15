# Runbook

Commands to rebuild every result in the blog post, in pipeline order. Each step points to the chunk document that
holds its full record (decisions, costs, run dirs, pitfalls). Configuration ids are explained in
[`docs/README.md`](README.md#configurations-blog-names-and-repo-ids).

## Setup

```bash
uv sync --group dev --group eval              # local (Windows/macOS/Linux): tests, data prep, Claude API steps, reports
uv sync --group dev --group eval --group gpu  # GPU machine (Linux): adds vLLM
uv sync --group dev --group eval --group gpu --group rl   # RL nodes: adds TRL
git submodule update --init                   # third_party/agentic-misalignment (overlap checks; legacy harness)
```

`.env` at the repo root (never committed):

```
ANTHROPIC_API_KEY=...   # corpus generation, verdicts, judges (claude-sonnet-5)
HF_TOKEN=...            # gated Gemma 3 weights/tokenizers; the private Halden checkpoints
```

Conventions: every CLI has `--config --dry-run --limit --seed --out --model-path`; a dry run processes at most 3 items
verbosely; each run creates an immutable directory under `outputs/` with `resolved_config.yaml`, `run_meta.json`, the
raw records (`*.jsonl`), `usage.json` and a `summary.json` with a provenance block. Every reported number is
recomputable from those raw files with the `report` modules.

GPU work ran on NVIDIA Brev A100 80 GB instances driven from WSL; every GPU job is launched detached through
`scripts/brev/run_bg.sh` (setup, sync and RL scripts in `scripts/brev/`, procedure in
[`phase3/README.md`](phase3/README.md) Section 6). `data/` and `outputs/` are gitignored (except manifests, verdicts,
generated dilemmas and the final report), so they are copied to instances with rsync, not pulled.

## 1. Data: MoralChoice, constitution verdicts, splits

Chunk 1 ([`phase3/chunks/01_moralchoice_splits.md`](phase3/chunks/01_moralchoice_splits.md)).

```bash
uv run python -m calign.data.moralchoice                 # 680 high-ambiguity scenarios -> data/scenarios + manifest
uv run python -m calign.data.moralchoice --low-ambiguity # 687 low-ambiguity items (over-citation diagnostic)
uv run python -m calign.validate.verdicts                # constitution verdict per scenario (committed)
uv run python -m calign.validate.verdicts --exclude-principle 6 --only-invoking 6 --only-clear \
    --out data/scenarios/constitution_verdicts_noP6.jsonl   # counterfactual verdicts without P6 (committed)
uv run python -m calign.data.phase3_split [--drop-ids-file drops.txt]   # dev / anchors / eval-1 / eval-2 manifest
uv run python scripts/diagnostics/show_verdict_audit.py --n-hard 30 --n-random 10   # verdict audit
```

## 2. SFT corpus

The synthetic corpus (pretraining-style documents + chat transcripts + deterministic fact cards) was built in Phase 1
([`details.md`](details.md) "Corpus pipeline", "SFT v2"); chunks 5 / 5b / 10 derive the Phase 3 training sets from it.

```bash
uv run python -m calign.corpus.generate_docs           # documents (Batches); --dry-run --no-batches for 3 items
uv run python -m calign.corpus.generate_transcripts    # chat transcripts (Batches)
uv run python -m calign.corpus.build_sft_dataset --tokenizer google/gemma-3-27b-it   # -> data/sft
uv run python -m calign.corpus.fact_cards --tokenizer google/gemma-3-27b-it          # -> data/sft_v2 (+ fact cards)
uv run python -m calign.corpus.replay prompts && uv run python -m calign.corpus.replay respond   # replay data (GPU for respond)
uv run python scripts/diagnostics/show_corpus_samples.py   # eyeball accepted / rejected items

# SFT-fact (C2): knowledge subtypes only, minus documents the audit flags for worked applied cases
uv run python -m calign.corpus.audit_application --out data/manifests/sft_kn_audit.jsonl
uv run python -m calign.corpus.build_sft_v3 --keep knowledge --drop-ids data/manifests/sft_kn_audit.jsonl \
    --replay data/replay/responses.jsonl                                              # -> data/sft_kn
# SFT-app (C2-kna): application material with P6 held out strictly (name, meta and unnamed P6 reasoning)
uv run python -m calign.corpus.audit_principle --principle 6 --include-ids data/manifests/sft_kn_audit.jsonl \
    --out data/manifests/sft_kna_p6_audit.jsonl
uv run python -m calign.corpus.build_sft_v3 --keep application --exclude-principle 6 \
    --include-ids data/manifests/sft_kn_audit.jsonl --drop-ids data/manifests/sft_kna_p6_audit.jsonl \
    --replay data/replay/responses.jsonl --replay-share 0.22                          # -> data/sft_kna
uv run python -m calign.corpus.build_sft_v3 --keep knowledge --drop-ids data/manifests/sft_kn_audit.jsonl \
    --stem sft_kna_kn                                                                 # -> data/sft_kna_kn
# SFT-all (C2-app, extra row): the v2 corpus with the partial P6 hold-out
uv run python -m calign.corpus.build_sft_v3 --exclude-principle 6 --replay data/replay/responses.jsonl   # -> data/sft_v3
```

## 3. SFT training (GPU)

LoRA r=64 / alpha=64 on the language-model projections, lr 1e-4 cosine, effective batch 32, adapters saved per epoch.
Chunks 5b ([`05b_sft_knowledge.md`](phase3/chunks/05b_sft_knowledge.md)) and 10 ([`10_sft_kna.md`](phase3/chunks/10_sft_kna.md)).

```bash
CALIGN_GPU_TESTS=1 CALIGN_MODEL_PATH=google/gemma-3-4b-it uv run pytest tests/gpu -q   # smoke test on the 4B
# SFT-fact: 6 epochs, epoch 4 chosen; merge and export the text-only RL start
uv run python -m calign.train.sft --config configs/sft_kn.yaml
uv run python -m calign.train.merge --adapter outputs/models/sft_kn/adapter_epoch4 --out outputs/models/sft_kn/merged_epoch4
uv run python -m calign.train.export_text_only --model-path outputs/models/sft_kn/merged_epoch4 \
    --out outputs/models/sft_kn/merged_epoch4_text --verify 5
uv run python -m calign.train.push_to_hub --run-dir outputs/models/sft_kn ...   # repos/revisions in configs/model_sft_kne4.yaml
# SFT-app: application stage on the SFT-fact model (epoch 3 chosen), then one knowledge epoch
sh scripts/brev/chunk10_train_lite.sh      # calign.train.sft --config configs/sft_kna.yaml + per-epoch lite checks
sh scripts/brev/chunk10_kn_pass.sh         # calign.train.sft --config configs/sft_kna_kn.yaml
# SFT-all: calign.train.sft --config configs/sft_v3.yaml (epoch 4)
```

Per-epoch checkpoint choice uses the cheap dev-split check: `uv run python -m calign.evals.lite run --eval-config <id>`.

## 4. RL data: generated dilemmas (Claude + GPU filter)

Chunk 6 ([`06_rl_data.md`](phase3/chunks/06_rl_data.md)). 965 generated questions, filtered to the 208 that SFT-fact
gets wrong in at least one of 8 samples. The checked pools and the final sets are committed under `data/dilemmas/`.

```bash
uv run python -m calign.dilemmas.generate --pool p15            # P1-P5 seeds, variants, checks
uv run python -m calign.dilemmas.generate --pool p6
uv run python -m calign.dilemmas.filter sample --eval-config C2kn@e4 --pools p15 p6 --k 8 --temperature 1.0   # GPU
uv run python -m calign.dilemmas.filter select --rl-start-run outputs/dilemmas/C2kn@e4/dilemma_filter/<run>
uv run python scripts/diagnostics/show_dilemmas.py --kept-only
```

## 5. RL: GRPO (RL-outcome = C3, RL-process = C4)

Chunks 7-8 ([`07_rl_infra_pilot.md`](phase3/chunks/07_rl_infra_pilot.md), [`08_rl_runs.md`](phase3/chunks/08_rl_runs.md)).
One 2 x A100 node per run (vLLM rollout server + TRL trainer), 60 steps, 12 prompts x 8 samples per step, adapters
pushed to HF every 10 steps.

```bash
# reward scale for C4 (so both rewards start at the same scale) and judge calibration, local with the Claude judge
uv run python -m calign.rl.calibrate_judge sample --records outputs/dilemmas/C2/dilemma_filter/batch1_k8_T1/records.jsonl
uv run python -m calign.rl.reward_scale measure --config configs/rl/C4.yaml --run-dir outputs/rl/reward_scale/rs2 \
    --dilemma-records outputs/dilemmas/C2/dilemma_filter/batch1_k8_T1/records.jsonl
# one node per run (WSL), then the idle watchdog (command in scripts/brev/idle_watchdog.sh)
sh scripts/brev/rl_bootstrap.sh p3-c3 configs/rl/C3.yaml C3     # rollout server, trainer, HF pushes, core suites
sh scripts/brev/rl_bootstrap.sh p3-c4 configs/rl/C4.yaml C4
uv run python -m calign.rl.checkpoints eval-configs --config-id C3 --run-dir outputs/rl/C3 --steps 10 20 30 40 50 60
uv run python -m calign.rl.monitor --run-dir outputs/rl/C4
```

## 6. Evaluation

Suite `p3-v1` is frozen (`data/manifests/suite_p3-v1.json`; [`phase3/README.md`](phase3/README.md) Section 10).
Chunks 2-4 and 9 ([`02_budget_suite.md`](phase3/chunks/02_budget_suite.md),
[`03_scenarios.md`](phase3/chunks/03_scenarios.md), [`09_extended_and_report.md`](phase3/chunks/09_extended_and_report.md)).

```bash
# core suite (MoralChoice eval-1 / eval-2 / hard, IFEval, MATH-500, coherence, over-citation, quiz): GPU, one vLLM load
uv run python -m calign.evals.suite --eval-config C0 --all-clear [--dry-run]
sh scripts/brev/sync_back.sh p3-a100 evals                                         # WSL
uv run python -m calign.evals.suite --judge-only --suite-run outputs/evals/C0/suite/<run> [--coherence-rep1]   # Claude judges, local
# agentic scenarios: scenario 1 = deadline (Study Significance), scenario 2 = briefing (Excursion Briefing); GPU
uv run python -m calign.scenarios.run --eval-config C0 --scenario deadline briefing --level L1 --n 50
uv run python -m calign.scenarios.report --runs outputs/scenarios/C0/*/*
uv run python scripts/diagnostics/show_scenario_episodes.py --run-dir outputs/scenarios/C0/deadline_L1/<run>
```

Every component also has its own CLI (`calign.evals.{moralchoice,ifeval,math500,overcitation,coherence,quiz}`,
subcommands `sample` / `report` and the judge steps) for single reruns. GPU sequences used for the final rows:
`scripts/brev/chunk9_gpu.sh`, `scripts/brev/chunk10_eval.sh`.

## 7. Report and blog figure

```bash
uv run python -m calign.evals.report --configs C0 C1 C2kn@e4 C2@e4 SFTP C3@s60 C4@s20 C4@s50 C2kna@e3kn1 \
    --checkpoints-of C2 C3 C4 --out outputs/evals/report/final_c10
uv run python -m calign.evals.primary --out outputs/evals/report/final_c10          # paired differences with CIs
uv run python outputs/evals/report/c10_paired.py                                     # SFT-app vs SFT-fact / SFT-all
uv run python -m calign.rl.dynamics --runs outputs/rl/C3 outputs/rl/C4 --out outputs/evals/report/final_c9   # RL plots
uv run python -m calign.evals.scenario_figure                                         # the scenario figure in README
```

Results and readings: [`phase3/runs.md`](phase3/runs.md).

## Phase 1 history: pilot SFT and validation

The pilot instilling run (2026-09-11) that preceded Phase 3; kept for reproducibility. The agentic-misalignment
checks that were part of it now live in `legacy/` ([`legacy/README.md`](../legacy/README.md)).

```bash
uv run python -m calign.train.sft --dry-run      # 3 steps; check gpu_peak_* in outputs/models/sft_pilot_dryrun/peft_summary.json
uv run python -m calign.train.sft                # LoRA r=64 on data/sft -> outputs/models/sft_pilot/adapter
uv run python -m calign.train.merge --adapter outputs/models/sft_pilot/adapter   # -> outputs/models/sft_pilot/merged (~55 GB)

# validation sampling: both stages into ONE run dir (each call appends to records.jsonl and writes
# resolved_config_<stage>.yaml; rerunning a stage into the same dir is refused, so use a fresh --out to redo one)
V=outputs/validation/gemma3_pilot
uv run python -m calign.validate.run_validation --stage base       --model-path google/gemma-3-27b-it          --out $V
uv run python -m calign.validate.run_validation --stage sft_merged --model-path outputs/models/sft_pilot/merged --out $V
uv run python -m calign.validate.judge  --run-dir $V    # Claude judge (Batches), local
uv run python -m calign.validate.report --run-dir $V    # gate: recall_pass
```

## Tests

```bash
uv run pytest tests/unit          # fast, local
uv run pytest tests/api           # one real Claude call per test (needs ANTHROPIC_API_KEY)
uv run pytest tests/gpu           # on the GPU machine (CALIGN_GPU_TESTS=1)
uv run pytest legacy/tests/unit   # legacy probe / misalignment code
```
