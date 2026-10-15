# CLAUDE.md — constitutional-alignment

Research code for the LessWrong post "Encouraging general generalisation: Teaching Gemma (To Articulate) Why"
(started as "Detecting and strengthening constitutional reasoning in LLMs"). Read this first; `docs/details.md` holds
the low-level facts for implementing and debugging, `docs/README.md` the project history, the map of all records and
the mapping between the blog's names (Base, Prompt, SFT-fact, SFT-app, RL-outcome, RL-process) and configuration ids.

## Goal and phases

Can we detect an internal representation of "constitutional reasoning" in a language model and use it
to strengthen alignment? Three phases:

- **Phase 1 (done)**: pick the model, prepare data, instil a constitution via LoRA SFT
  on a synthetic corpus (Teaching-Claude-Why style), merge the adapter, validate recall/application.
- **Phase 2 (done; code and records now in `legacy/`, see `legacy/README.md`)**: train linear probes (different label sources, token positions 33%/66%/final,
  layers early/mid/late), check whether they converge (cosine similarity), validate via steering
  (add the vector at every token during generation), test generalisation on the hard held-out
  agentic-misalignment scenarios, optional SAE interpretation with Gemma Scope.
- **Phase 3 (done; the blog post; replaces the earlier probe-RL plan; brief `docs/phase3/brief.md`, decisions `docs/phase3/plan.md`,
  operational plan `docs/phase3/README.md`)**: "alignment under a budget". Which methods make the model act on the
  constitution, at what cost to capability? Configurations C0 base, C1 base + budget-aware constitution system
  prompt, C2 knowledge-only SFT (fact and explanatory documents on all six principles, no application material,
  replay data; the RL start), C2-app SFT v3 (with application material, P6 held out), C3 C2 + GRPO with an outcome
  reward, C4 C2 + GRPO with outcome + citation-correctness reward. Alignment: MoralChoice eval-1 (trained
  principles) / eval-2 (P6-decisive) / generated P6 hard set (eval-2-hard), and two single-shot agentic scenarios (significance
  deadline; curated briefing, P6). Budget: IFEval, MATH-500 (STEM), coherence, over-citation; no coding benchmark.
  "Feasible" is a post-hoc label; the deliverable is the alignment-vs-budget frontier over SFT epochs and RL
  checkpoints. No hyperparameter search. Mechanistic interpretability (probes, steering, SAEs) is dropped.
  **RL start (chunk 5b, done 2026-10-02):** SFT v3 epoch 4 already solved the generated dilemmas (no RL signal), so
  C2 = knowledge-only SFT epoch 4, text-only `felixfabricius/gemma-3-27b-it-halden-sft-kn-e4@272d870`
  (`configs/model_sft_kne4.yaml`; eval config `C2` / runs `C2kn@e4`: MoralChoice hard 52.9 vs C2-app 78.5, budget
  within); SFT v3 epoch 4 is the extra row C2-app (runs `C2@e4`). Details: `docs/phase3/chunks/05b_sft_knowledge.md`.

**Phase 3 status (done 2026-10-04; `docs/phase3/runs.md` Part A, report `outputs/evals/report/final_c9`):** C3 / C4 =
60 GRPO steps on C2 (adapters on HF `felixfabricius/gemma-3-27b-it-halden-rl`); chosen checkpoints C3@s60 and
**C4@s50** (Felix 2026-10-13, for every comparison; supersedes chunk 9's s20/s50 split, `docs/phase3/plan.md` "C4 checkpoint"). Readings: the prompt (C1) is the strongest cheap lever but breaks the
over-citation margin (6.8%); application SFT (C2-app) is best on every alignment measure within budget (scenario 2
deception 20% vs C0 80%; its P6 hold-out was partial: 162 of 912 kept examples apply or cite P6, `docs/phase3/runs.md` A7); RL on C2 adds eval-1 +2.2, hard +5.1, nothing measurable in the scenarios; the citation
reward (C4) is learned in-distribution but does not transfer, does not beat C3, and erodes P6 knowledge after step 30
(RL-train has no P6 items, so P6 citations count as wrong). eval-2-hard has no item set (E6).
**Chunk 10 (done 2026-10-11, `docs/phase3/runs.md` A10):** C2-kna = C2 + application SFT with P6 held out strictly (159
P6-free application rows + 22% replay, 3 epochs, then 1 knowledge epoch that restores the P6 link the application
stage erased), `C2kna@e3kn1`: eval-1 +3.1 and hard +8.8 vs C2, but no measurable transfer to P6 (eval-2 +1.7
[-3.5, +7.6], scenario 2 deception 70% vs C2 76% and C2-app 20%).

## Fixed decisions (do not re-ask)

| Topic | Decision |
|---|---|
| Model | `google/gemma-3-27b-it` (gated; HF_TOKEN; switched from Gemma 2 9B on 2026-09-11 because 9B was too weak to act misaligned, see docs/details.md). Multimodal checkpoint used text-only (vLLM `language_model_only`, LoRA regex on the language model). bf16, sdpa, 8192 context. Gemma Scope 2 IT SAEs: full range at layers 16/31/40/53, small ones at all 62 layers. Gemma 2 9B configs kept as `legacy/configs/*_gemma2_9b.yaml`. |
| Constitution | `constitution.md` (6 principles + priority rules), named **"the Halden Constitution"** everywhere (config `constitution_name`); model keeps its Gemma identity. |
| Instilling | Mixed corpus: pretraining-style documents + chat transcripts (no system prompt in stored transcripts). LoRA r=64, alpha=64, dropout 0.05, language-model linear projections, lr 1e-4 cosine, warmup 3%, 2 epochs, seq 2048, effective batch 32 (micro-batch 1 x 32); merge after training. Pilot scale 500 docs / 200 transcripts; scale-up 3k/1k is config-only. |
| Claude usage | `claude-sonnet-5` for corpus generation, misalignment classifiers, judges. Sampling params are never sent (rejected by Claude 5); adaptive thinking; effort per role. Message Batches (50% price) preferred for large non-urgent jobs. |
| Medium data | MoralChoice high-ambiguity (680), stratified 60/20/20 by rule: 408 probe_train / 136 probe_val / 136 heldout_steer (committed manifest). Phase 1.5 validation uses 50 scenarios sampled from probe_train. |
| Hard data | Anthropic agentic-misalignment (blackmail, leaking, murder) via git submodule; never used for training; pre-SFT check = 12 conditions x 25 samples, temperature 1.0. |
| Inference | vLLM for sampling (Linux GPU), HF transformers + PEFT for training, merging, forced passes and (Phase 2) activations. Both consume token ids from `calign.prompting.encode_prompt`. |


## Working agreements with Felix

- Ask short clarifying questions instead of assuming (hyperparameters, schemas, formats); offer options with a recommendation.
- Give a cost estimate before Claude spend; measure per-item cost on a dry run first and confirm before big runs.
- Every reported number must be recomputable from raw files: run dirs keep raw JSONL, `usage.json`, and a `summary.json` with a provenance block; `report` modules recompute from raw.
- Commit incrementally with `uv` (never bare pip). **`git push` is allowed and is the canonical way to get code onto
  GPU instances** (Felix, 2026-10-01): commit, push, `git pull` on the instance. Remote:
  `github.com/felixfabricius/constitutional-alignment` (public).
- GPU work runs on NVIDIA Brev instances managed with the Brev CLI from WSL (`wsl -e bash -lc 'brev ls'`); runbook
  in `docs/phase3/README.md` Section 6. **Every GPU job is launched detached via `scripts/brev/run_bg.sh` (setsid +
  nohup) so it keeps running if the ssh connection or the agent session drops**; never in the foreground over ssh.
  Instances cannot be stopped, only deleted: sync run dirs locally and push weights to HF before deleting, and never
  treat an instance disk as a hand-off medium between chunks.
  The Colab MCP (L4, 24 GB) remains available for "does this work at all" checks.
- Phase 3 is executed in chunks, one fresh agent session per chunk: read `docs/phase3/README.md`, then
  `docs/phase3/status.md`, then your chunk document under `docs/phase3/chunks/`. Keep `docs/phase3/status.md` current (runs, ETAs,
  spend, open decision points) and fill the chunk's Results and cross-chunk Notes sections.
- Conceptual decisions (what is measured, how a result is read) are proposed with options and a recommendation and
  wait for Felix; implementation decisions follow repo conventions (`docs/phase3/brief.md` Section 0.1).

## Repository map

```
README.md           blog summary, results, figure; constitution.md the Halden Constitution
agentic_scenarios/  README.md: the two Phase 3 scenarios in readable form (exact materials, grading, judge prompts;
                    hand-copied from src/calign/scenarios, update it when the materials change; URL linked from the blog)
configs/            model.yaml, sft.yaml (pilot), sft_v2.yaml, sft_v3.yaml (SFT-all), sft_kn.yaml (SFT-fact),
                    sft_kna.yaml + sft_kna_kn.yaml (SFT-app), model_sft_kne4.yaml (RL start), model_sft_v3e4.yaml,
                    corpus.yaml, data.yaml, dilemmas.yaml, validation.yaml, eval_configs/<id>.yaml, rl/{C3,C4}.yaml
docs/               README.md (history, record map, name mapping), runbook.md (commands per stage), details.md;
  phase3/           README.md (operational plan, GPU runbook, chunk list), status.md, plan.md (decision register),
                    runs.md (all results), brief.md (annotated brief), scenarios_design.md (Section 7 = v2 design),
                    chunks/NN_*.md (plan + notes + results)
scripts/brev/       instance setup, background-run wrapper (run_bg.sh), RL bootstrap, sync-back, chunk scripts
scripts/diagnostics/  print-only inspection scripts (chat format, corpus, dilemmas, scenarios, verdict audit, quiz)
src/calign/
  config.py paths.py schemas.py stats.py constitution.py prompting.py
  llm/anthropic_client.py         cached, batch-capable Claude client with usage/cost accounting
  data/                           MoralChoice download + splits, Phase 3 split
  corpus/                         synthetic corpus, fact cards, replay, audits, SFT dataset builders (v3 / kn / kna)
  train/{data,sft,merge,export_text_only,push_to_hub}.py
  validate/                       constitution verdicts, response judge, Phase 1 validation
  inference/                      HF / vLLM backends, LoRA serving
  evals/ scenarios/ dilemmas/ rl/ Phase 3: suite + report + figure, agentic scenarios, generated RL data, GRPO
legacy/             original probe-and-steer project, outside the main package (README.md):
  calign_legacy/{misalignment,probe}   agentic-misalignment harness; Phase 2 probes / steering / SAE
  configs/ diagnostics/ tests/ docs/   (phase2_plan, phase2_runs, priority1_plan, priority1_runs, details)
tests/unit  tests/api (need ANTHROPIC_API_KEY)  tests/gpu (skipped without CUDA); legacy/tests the same for legacy
third_party/agentic-misalignment   pinned submodule (ea0630e), never modified (corpus / dilemma overlap checks; legacy)
data/               gitignored except manifests/, scenarios/constitution_verdicts*.jsonl, replay prompts, dilemmas
outputs/            gitignored run directories (final report files and the blog figure committed)
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

Done on the Brev A100 (2026-09-11): SFT pilot (r=64, 27 min, eval loss 1.755 -> 1.435, peak 68.7 GB), merge
(KL <= 2.2e-3, top-1 equal), merged model + adapter on HF (private `felixfabricius/gemma-3-27b-it-halden-sft-pilot`;
adapter also in `outputs/models/sft_pilot/adapter`), misalignment check on the merged model
(`outputs/misalignment/20260911_112613_50c47bcc`), validation `outputs/validation/gemma3_pilot` (judged, $1.94).
Gate `recall_pass` PASSED (sft/full: mention 97%, citation accuracy 0.87). Caveats that decide the next step
(docs/details.md "SFT pilot results"): without the constitution in context citations are often wrong (accuracy 0.53,
quiz fabrication 45%, "Principle 7" false premise accepted); in the long agentic prompts at T=1.0 the SFT model
invokes the constitution in ~55% of responses but often with garbled/confabulated content; leaking dropped
(explicit-goal 38/50 -> 23/46, p=0.011) while murder rose in no-goal conditions (1/50 -> 9/50, p=0.016).
Next step chosen: SFT v2 (fact cards, 4 epochs; docs/details.md "SFT v2"), then Phase 2.

## Phase 2 and priority-1 status (legacy)

Phase 2 (probes, steering, SAEs on SFT v2 epoch 3, 2026-09-11) and the priority-1 agentic baselines (2026-09-12) are
done and archived in `legacy/` (summary and runbooks: `legacy/README.md`; all numbers: `legacy/docs/phase2_runs.md`,
`legacy/docs/priority1_runs.md`). In short: best probe val AUROC 0.80, observables at a ceiling, no transfer to the
agentic data, positive steering inert; the constitution in the agentic system prompt halved harm (11.0% -> 5.4%).
