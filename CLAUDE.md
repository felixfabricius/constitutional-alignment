# CLAUDE.md — constitutional-alignment

Research code for "Detecting and strengthening constitutional reasoning in LLMs". Read this first;
`details.md` holds the low-level facts for implementing and debugging.

## Goal and phases

Can we detect an internal representation of "constitutional reasoning" in a language model and use it
to strengthen alignment? Three phases:

- **Phase 1 (in progress, mostly done)**: pick the model, prepare data, instil a constitution via LoRA SFT
  on a synthetic corpus (Teaching-Claude-Why style), merge the adapter, validate recall/application.
- **Phase 2 (next)**: train linear probes (different label sources, token positions 33%/66%/final,
  layers early/mid/late), check whether they converge (cosine similarity), validate via steering
  (add the vector at every token during generation), test generalisation on the hard held-out
  agentic-misalignment scenarios, optional SAE interpretation with Gemma Scope.
- **Phase 3 (deprioritised)**: probe-based RL (GRPO or similar) on a fresh LoRA over the merged
  model vs. direct-label RL; baselines: same-label RL, constitution-in-system-prompt, steering only.
  Probe stability pre/post RL.

## Fixed decisions (do not re-ask)

| Topic | Decision |
|---|---|
| Model | `google/gemma-3-27b-it` (gated; HF_TOKEN; switched from Gemma 2 9B on 2026-09-11 because 9B was too weak to act misaligned, see details.md). Multimodal checkpoint used text-only (vLLM `language_model_only`, LoRA regex on the language model). bf16, sdpa, 8192 context. Gemma Scope 2 IT SAEs: full range at layers 16/31/40/53, small ones at all 62 layers. Gemma 2 9B configs kept as `*_gemma2_9b.yaml`. |
| Constitution | `constitution.md` (6 principles + priority rules), named **"the Halden Constitution"** everywhere (config `constitution_name`); model keeps its Gemma identity. |
| Instilling | Mixed corpus: pretraining-style documents + chat transcripts (no system prompt in stored transcripts). LoRA r=64, alpha=64, dropout 0.05, language-model linear projections, lr 1e-4 cosine, warmup 3%, 2 epochs, seq 2048, effective batch 32 (micro-batch 1 x 32); merge after training. Pilot scale 500 docs / 200 transcripts; scale-up 3k/1k is config-only. |
| Claude usage | `claude-sonnet-5` for corpus generation, misalignment classifiers, judges. Sampling params are never sent (rejected by Claude 5); adaptive thinking; effort per role. Message Batches (50% price) preferred for large non-urgent jobs. |
| Medium data | MoralChoice high-ambiguity (680), stratified 60/20/20 by rule: 408 probe_train / 136 probe_val / 136 heldout_steer (committed manifest). Phase 1.5 validation uses 50 scenarios sampled from probe_train. |
| Hard data | Anthropic agentic-misalignment (blackmail, leaking, murder) via git submodule; never used for training; pre-SFT check = 12 conditions x 25 samples, temperature 1.0. |
| Inference | vLLM for sampling (Linux GPU), HF transformers + PEFT for training, merging, forced passes and (Phase 2) activations. Both consume token ids from `calign.prompting.encode_prompt`. |
| Custom constitution scenarios | Deferred to Phase 2 (schema has a `source` field). |

## Working agreements with Felix

- Ask short clarifying questions instead of assuming (hyperparameters, schemas, formats); offer options with a recommendation.
- Give a cost estimate before Claude spend; measure per-item cost on a dry run first and confirm before big runs.
- Every reported number must be recomputable from raw files: run dirs keep raw JSONL, `usage.json`, and a `summary.json` with a provenance block; `report` modules recompute from raw.
- Commit incrementally with `uv` (never bare pip). **Do not `git push`** from the agent (the permission prompt denies it); Felix pushes manually. Remote: `github.com/felixfabricius/constitutional-alignment` (public).
- Use the Colab MCP (L4, 24 GB) for "does this work at all" GPU checks; full runs go on a rented A100.

## Repository map

```
configs/            model.yaml, sft.yaml (Gemma 3 27B) + model_gemma2_9b.yaml, sft_gemma2_9b.yaml;
                    data.yaml, misalignment_check.yaml, corpus.yaml, validation.yaml
src/calign/
  config.py paths.py schemas.py stats.py constitution.py prompting.py
  llm/anthropic_client.py         cached, batch-capable Claude client with usage/cost accounting
  data/moralchoice.py             download + stratified split ;  data/samples.py
  misalignment/{prompts,classify,run,report}.py   upstream templates/classifiers via submodule
  inference/{backend,hf_backend,vllm_backend}.py
  corpus/{taxonomy,prompts,common,generate_docs,generate_transcripts,build_sft_dataset}.py
  train/{data,sft,merge}.py
  validate/{prompts,verdicts,run_validation,judge,report}.py
diagnostics/        print-only inspection scripts (chat format, token positions, prompts, samples, quiz)
tests/unit (86)  tests/api (2, need ANTHROPIC_API_KEY)  tests/gpu (skipped without CUDA)
third_party/agentic-misalignment   pinned submodule (ea0630e), never modified
data/               gitignored except manifests/, scenarios/constitution_verdicts.jsonl
outputs/            gitignored run directories
```

Conventions: every CLI has `--config --dry-run --limit --seed --out --model-path`; dry runs process <= 3
items verbosely; each run creates an immutable dir with `resolved_config.yaml` + `run_meta.json`.

## Phase 1 status

Done locally: MoralChoice splits; 680 constitution verdicts; pilot corpus (494 docs, 192 transcripts);
SFT data (`data/sft`: 651 train / 35 val, 572k Gemma 3 tokens, regenerable from the API cache); all pipelines
written and unit-tested; Colab smoke tests passed for Gemma 2 (9B load/generate/forced pass; 2B LoRA train + merge).

Done on the Brev A100: base misalignment check. Gemma 2 9B too weak (gate only via leaking headline 4/25);
Gemma 3 27B passes clearly (leaking headline 22/25, murder headline 6/25, run `20260910_222307_9f28bd09`, which is
the pre-SFT baseline).

Remaining (GPU machine, in this order, see README runbook):
1. GPU tests with `gemma-3-4b-it`, then `calign.train.sft --dry-run` on 27B (check peak memory in
   `peft_summary.json`), then `calign.train.sft` and `calign.train.merge`.
2. `calign.misalignment.run --stage sft_merged --model-path <merged>`; compare with the base run.
3. `calign.validate.run_validation` for `base` and `sft_merged`, then `judge`, then `report` ->
   gate `recall_pass` (SFT model with constitution in prompt: mention rate >= 0.8, citation accuracy >= 0.7).
   If it fails: more/better SFT data is needed; ask before deciding how (scale-up corpus, tighten P4 prompts).

## What Phase 2 will need (already provided for)

- `GenerationRecord` has `judge` (LLM judge scores 0-1) and `activations` (path, layers, positions) fields.
- `HFBackend.forward_forced(prompt_ids, completion_ids, layers)` returns hidden states + per-token log-probs
  (token-forcing: same answer text with and without the constitution in context). `layers` are HF indices:
  Gemma Scope layer L (`ModelConfig.probe_layers`) is `hidden_state_index(L) = L + 1`.
- `prompting.answer_span` / `relative_positions` give 33%/66%/final token indices; `render_system_prompt`
  variants `full`/`none` are the two contexts.
- `data/scenarios/constitution_verdicts.jsonl` is the outcome label; `probe_val` and `heldout_steer` splits are untouched.
- Label definitions should be function arguments so cell-1 vs cell-3 style labels are easy to swap.
