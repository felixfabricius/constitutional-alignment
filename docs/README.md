# Documentation and project history

Start with the [top-level README](../README.md) (what the project found), then [`runbook.md`](runbook.md) (how to
rebuild it), then [`phase3/runs.md`](phase3/runs.md) (every number, with the run directory it comes from).

## Configurations: blog names and repo ids

The blog post and the figures use display names; the code, configs and run directories use configuration ids
(`configs/eval_configs/<id>.yaml`, run dirs `outputs/evals/<id>/...`, `outputs/scenarios/<id>/...`).

| Blog name | Config id (report id) | What it is |
|---|---|---|
| Base | `C0` | `google/gemma-3-27b-it`, no system prompt |
| Prompt | `C1` | Base + the budget-aware constitution system prompt (`budget_silent`: full constitution, "act according to it at all times, but never mention it unless asked") |
| SFT-fact | `C2` (`C2kn@e4`) | Knowledge-only LoRA SFT (fact cards, fact-QA, explainers, FAQs, comparisons, critiques; all six principles; no applied cases) + replay, epoch 4. The RL start. `configs/sft_kn.yaml`, `configs/model_sft_kne4.yaml` |
| SFT-app | `C2-kna` (`C2kna@e3kn1`) | SFT-fact + application SFT with P6 held out strictly (3 epochs) + one knowledge epoch that restores the P6 link. `configs/sft_kna.yaml`, `configs/sft_kna_kn.yaml` |
| RL-outcome | `C3` (`C3@s60`) | SFT-fact + GRPO with a binary outcome reward, step 60. `configs/rl/C3.yaml` |
| RL-process | `C4` (`C4@s50`) | SFT-fact + GRPO with outcome + citation-correctness reward, step 50. `configs/rl/C4.yaml` |
| SFT-all (blog text only) | `C2-app` (`C2@e4`) | SFT v3 epoch 4: application material for all principles (P6 hold-out only partial: 162 of 912 examples apply or cite P6). `configs/sft_v3.yaml` |
| — | `SFTP` | Exploratory: SFT-fact + the C1 system prompt |

Other suffixes are per-epoch or per-step checkpoints (`C2kn@e1..e6`, `C3@s10..s60`, `C4@s10..s60`) and the C1 prompt
drafts (`C1@budget_*`). Scenario 1 is `deadline_L1` (Study Significance), scenario 2 is `briefing_L1` (Excursion
Briefing); the eval-sets are MoralChoice eval-1 (P1-P5, "MoralChoice-P1to5"), eval-2 (P6-decisive) and the hard subset.

## How the project evolved

The repository grew in four stages. The first two followed the original goal (find and strengthen an internal
representation of constitutional reasoning); the code and records for those stages now live in
[`legacy/`](../legacy/README.md).

1. **Phase 1, instilling (2026-09-09 to 09-11).** Wrote the Halden Constitution (`constitution.md`), split
   MoralChoice, generated a synthetic corpus with Claude (documents + chat transcripts) and LoRA-fine-tuned the model on
   it. Gemma 2 9B-IT was dropped for Gemma 3 27B-IT because it was too weak to act misaligned on the
   agentic-misalignment check. The pilot SFT passed recall but confabulated citations without the constitution in
   context, so SFT v2 added deterministic fact cards. Records: [`details.md`](details.md) ("SFT pilot results",
   "SFT v2"), [`legacy/README.md`](../legacy/README.md).
2. **Phase 2, probes and steering (2026-09-11).** Linear probes from several label sources, positions and layers;
   steering; Gemma Scope SAE lookups. The SFT model already named its constitution in ~95% of answers, so the probe
   was effectively an outcome probe, positive steering had no headroom and the probes did not transfer to the
   agentic data. Records: [`legacy/docs/phase2_plan.md`](../legacy/docs/phase2_plan.md),
   [`legacy/docs/phase2_runs.md`](../legacy/docs/phase2_runs.md).
3. **Priority-1 agentic baselines (2026-09-12).** The constitution in the agentic system prompt halved harm; probe
   steering did nothing. Records: [`legacy/docs/priority1_plan.md`](../legacy/docs/priority1_plan.md),
   [`legacy/docs/priority1_runs.md`](../legacy/docs/priority1_runs.md).
4. **Phase 3, alignment under a budget (2026-09-30 to 10-13; the blog post).** Re-scoped from probe-driven RL to a
   comparison of prompting, SFT and GRPO on alignment generalisation, with capability (IFEval, MATH-500, coherence,
   over-citation) as a budget. Two new agentic scenarios replaced the agentic-misalignment data. Executed in chunks
   0-10, one agent session each:
   - chunks 0-4: scaffolding, MoralChoice splits, budget suite, scenarios and judge prompts, C1 prompt choice and the
     frozen suite `p3-v1`;
   - chunk 5: SFT v3 (now SFT-all); it already solved the generated dilemmas, leaving no RL signal, so
     **chunk 5b** (2026-10-02) trained the weaker knowledge-only SFT-fact as the RL start;
   - chunks 6-9: generated RL dilemmas, GRPO infrastructure and pilot, the C3 / C4 runs, the extended suite and report;
   - **chunk 10** (2026-10-09 to 10-11): SFT-app with a strict P6 hold-out, to test transfer across principles;
   - 2026-10-13: C4@s50 fixed as the RL-process checkpoint for every comparison; blog figure.

## Document map

| Document | What it holds |
|---|---|
| [`runbook.md`](runbook.md) | Setup and the commands for every pipeline stage, in order |
| [`details.md`](details.md) | Low-level facts for implementing and debugging (environment, Gemma 3 / transformers quirks, prompting, Claude client, corpus, evals, scenarios, RL, LoRA serving) |
| [`phase3/runs.md`](phase3/runs.md) | Phase 3 results: configurations, main table, primary comparisons with CIs, RL trajectories, readings, caveats, costs, how to recompute |
| [`phase3/plan.md`](phase3/plan.md) | Decision register (what was decided, when, why) |
| [`phase3/README.md`](phase3/README.md) | Operational plan: chunk list and dependencies, conventions, GPU workflow on Brev, eval-config schema, glossary, frozen suite |
| [`phase3/status.md`](phase3/status.md) | Last implementation / run / spend status and the decision points raised along the way |
| [`phase3/chunks/`](phase3/chunks/) | One document per chunk: plan, notes from other chunks, results (the detailed lab notebook) |
| [`phase3/brief.md`](phase3/brief.md) | The annotated Phase 3 brief with the evidence behind each decision (long; read sections on demand) |
| [`phase3/scenarios_design.md`](phase3/scenarios_design.md) | Design of the agentic scenarios (Section 7 = the v2 design that was run) |
| [`../agentic_scenarios/README.md`](../agentic_scenarios/README.md) | The two scenarios in readable form: exact materials, grading, judge prompts |
| [`../CLAUDE.md`](../CLAUDE.md) | Instructions and fixed decisions for coding agents working in this repository |
| [`../legacy/README.md`](../legacy/README.md) | The original probe-and-steer project: code, configs, runbooks and results |
