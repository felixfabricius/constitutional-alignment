# Phase 3 status

Last updated: 2026-10-02 (chunk 4 session). Update at every run start/end and chunk boundary.

## A. Implementation status

| chunk | status | last commit | notes |
|---|---|---|---|
| 0 Scaffolding | done | see chunk 0 Results | eval config schema + C0/C1, `scripts/brev/*`, Brev syntax recorded |
| 1 MoralChoice splits | done (E4 applied) | see chunk 6 | splits eval2 **43** (was 51, E4) / anchors 40 / dev 50 / eval1 344; C0 base run 84.6% overall, eval2 71.1%; hard subset 78 items; audit file ready; replicate shows hard-subset regression to the mean (E3: replicate is the C0 reference) |
| 2 Budget suite | done | 5b313d8 | C0: IFEval strict 82.1, MATH-500 87.8, over-citation 0.0%, coherence-v2.1 0.981, quizzes 0.00/0.03; report `outputs/evals/report/c0_first_v21`; suite ~32 GPU-min per configuration |
| 3 Scenarios | **done** | 329a376 | levels fixed: deadline L1, briefing L1 (materials v2.2; base L0 0/24, L1 22/25); judges implemented with Felix's prompts; results in the chunk 3 doc |
| 4 Prompt + freeze | **in progress** (started 2026-10-02 ~00:30 UTC) | 1f0f7ba | code pushed: budget variants + draft configs, `calign.evals.prompt_select`, coherence-set-v2 (+30 deadline-L1), report scenario cells + D15 table, `SUITE_VERSION p3-v1`, `calign.evals.freeze`, scenario judges v2 (Felix approved both section-5 fixes 2026-10-02); GPU job 1 running on p3-scen |
| 5 SFT v3 | in progress | c88291d | started 2026-10-01; SFT v3 trained (4 epochs, eval loss 1.678 -> 1.302 at epoch 3.1, 1.312 at 4.0); adapters on HF `felixfabricius/gemma-3-27b-it-halden-sft-v3@0d47098` (`adapter_epoch1..4/`); eval configs `C2@e1..e4`; per-epoch core suites running; 4B LoRA-serving check: adapter applied, served-vs-merged 0.085 > 0.05 bound, HF-PEFT reference diagnostic pending (text-only export exact) |
| 6 RL data | in progress | 831949a | started 2026-10-01; generator, filter, `hardsets` suite component, diagnostic + 18 tests pushed; dry run $0.12 (~$0.023 per seed family with Batches -> ~$6 for all pools, plan $25); 20-seed pilot running (Batches); found the verdict-JSON parse bug -> E4; RL-start k=8 filter waits for chunk 5's RL-start epoch |
| 7 RL infra + pilot | code done, GPU pending | see chunk 7 doc | 2026-10-01: `calign.rl` (prompts, mix, R1/R2 rewards, deterministic citation layer, local judge server + client, calibration, GRPO trainer on TRL 1.14.1, monitor), `configs/rl/C3,C4.yaml`, `scripts/brev/rl_setup.sh` + `rl_serve.sh`, 42 unit tests + 4B GPU test; GPU steps wait for chunk 5's text-only RL start and chunk 6's RL-train + k=8 run; R7-relevance decided (b) |
| 8 RL runs | not started | | |
| 9 Extended + report | not started | | |

## B. Run status (GPU)

| instance | state | run / log | started (UTC) | expected end | chunk |
|---|---|---|---|---|---|
| p3-scen (Crusoe `a100-80gb.1x`, $1.98/h, stoppable) | **running** (restarted 2026-10-02 ~00:35 UTC for chunk 4) | chunk 4 job 1 `outputs/logs/c4_job1.log`: C1 drafts (dev + IFEval, one load), then C0 scenario main grid (4 x 50); job 2 (C1 core suite + C1 scenarios) after the draft choice | 2026-10-02 00:41 | job 1 ~01:15 UTC; chunk GPU work ~02:30 UTC | 4 (3 before) |
| p3-a100 (hyperstack `A100_80G`, $1.62/h, not stoppable) | **deleted** 2026-10-01 22:50 UTC (Felix: shut down; all outputs synced first) | chunks 1-2: C0 suite, C0 replicate (`outputs/logs/c0_suite.log`, `c0_mc_rep.log`) | 2026-10-01 19:45 | 22:50 | 1, 2 |
| p3-sft (massedcompute `A100_sxm4_80G`, $1.66/h, not stoppable, driver 580; up since ~20:15 UTC) | running | core suite C2@e2..e4 (`outputs/logs/s5_suites_e234.log`; C2@e1 done and judged; the e1 process hung 70 min after its manifest, fixed in 2c9fa49); then 4B lora_check diagnostics, merge + text-only export of the RL start | 2026-10-02 00:10 | suites ~01:55 UTC (~33 min each); chunk GPU work ~03:00 | 5 |

Instance registry (2026-10-01): `p3-a100`, `p3-sft`, `p3-scen` as in the table; the Phase 2 instance `train-inst` no
longer exists. p3-a100 facts: user `shadeform`, repo `~/constitutional-alignment`, root disk 97 GB, `~/.cache` ->
`/ephemeral/cache` (700 GB), job scripts in `~/jobs/`; driver R570 (CUDA 12.8) needs the CUDA 13 forward-compat libs
for the locked torch cu130 (`setup.sh` installs `cuda-compat-13-0`, `run_bg.sh` exports `LD_LIBRARY_PATH`).
Network note: on 2026-10-01 Felix's network blocked every outbound port except 80/443, so ssh to the Brev gateway failed
until he switched networks; test with `wsl -e bash -lc 'timeout 8 bash -c "</dev/tcp/global.prd.ga.run.brev.nvidia.com/26457"'`.

## C. Estimated completion

| section | remaining GPU-h | remaining agent sessions (approx.) | note |
|---|---:|---:|---|
| current chunk (0) | 0 | 0.5 | |
| chunks 1-4 (evaluation suite, scenarios, C0/C1) | 7 | 4 | |
| chunks 5-6 (SFT v3, RL data) | 5 | 2 | |
| chunks 7-8 (RL) | 32 | 3 (+ waiting on runs) | step time unmeasured; 80 steps per run planned |
| chunk 9 (extended suite, report) | 5 | 1 | |
| **project** | **~49 (+20% overhead)** | **~11** | calendar time depends on session cadence; RL runs are ~6 h wall-clock each |

## D. Spend so far (Phase 3 from 2026-10-01)

| item | amount |
|---|---|
| GPU | p3-a100 19:45-22:50 UTC = ~3.1 h x $1.62 = ~$5.0 (deleted); p3-sft since ~20:15 UTC at $1.66/h (running, chunk 5); p3-scen ~1.6 h at $1.98/h = ~$3.2 (stopped, chunk 3) |
| Claude | chunks 1-2: $2.44 (counterfactual verdicts $0.74, C0 judge phase $0.95, coherence v2.1 + rep1 ~$0.68, dry runs $0.07); $1.41 (replay prompts incl. dry run, chunk 5); chunk 3: $1.95 (tags on 175 episodes $0.94, rubric check $0.96, dry runs $0.05); chunk 6: $0.12 dry run (pilot running) |

## E. Open decision points awaiting Felix

| id | question | raised by | status |
|---|---|---|---|
| E1 | eval-2 has 51 P6-decisive items (< 60): keep strict, add no-P6 confidence < 0.6 (67), or any-mention (278)? | chunk 1 | **decided 2026-10-01: keep strict 51**; eval-2-hard (chunk 6) carries P6 power |
| E2 | Verdict audit (D4): which ids to drop? (`outputs/evals/C0/verdict_audit.txt`) | chunk 1 | **decided 2026-10-01: verdicts look good, no drops**; manifests unchanged |
| E3 | Which C0 run is the MoralChoice reference (the hard subset is selected on the first run; replicate +5.4 on it)? | chunks 1-2 | **decided 2026-10-01: the independent replicate** `outputs/evals/C0/moralchoice/c0_rep_seed20261002` (C0 suite manifest `moralchoice`; the defining run is `moralchoice_defining`). C0 reference: eval1 85.2, eval2 66.7, hard 21.5 |
| S3-briefing | Briefing level: Felix chose (a) **L1 of materials v2.2** (base omission 88%) on 2026-10-01 | chunk 3 | resolved |
| S3-judge | Judge prompts supplied by Felix (`phase3/chunks/03_judge_prompts.md`) and implemented verbatim; briefing brief / tier definitions updated to v2.1 (doc section 5, please check). Suggested prompt fixes in section 5: `confusion` fires on post hoc "I made a mistake" excuses (28-52%), deadline brief should state that no participant met a pre-registered criterion. | chunk 3 | resolved (2 wording suggestions open) |
| E4 | **Verdict parse failures in the stored MoralChoice verdicts** (found in chunk 6, 2026-10-01). The verdict judge sometimes returns a brace-less JSON body whose `rationale` string is never closed; `parse_verdict` then stored `unclear`, confidence 0. Affected: **38 of 680** original verdicts (re-parsed: 34 action1, 3 action2, 1 either; 29 with confidence >= 0.8; 19 invoke P6) and **8 of 278** no-P6 counterfactual verdicts. All 8 counterfactual cases are in **eval-2**, and parsed correctly their verdict is *unchanged* without P6, so they are **not P6-decisive**: eval-2 should be 43 items, not 51. The 38 were excluded from the clear pool (eval-1 / dev / anchors). Parser fixed in 831949a (regex fallback only when JSON parsing finds no `prescribed_action`; the stored files are untouched). Options: **(a, recommended)** minimal: re-parse both verdict files from their stored raw text ($0), move the 8 to a new `dropped` reason "parse artifact, not P6-decisive" so eval-2 = 43, leave the 38 newly clear items out of every split (documented); every existing run's report recomputes from raw records, no GPU. **(b)** full rebuild: re-parse, run counterfactual verdicts for the 19 P6-invoking new items (~$0.1), rebuild the splits (eval-1/dev/anchors change; C0 and C2 MoralChoice runs need the new items sampled, ~15 GPU-min per configuration). **(c)** leave as is and report eval-2 as "51 items, 8 of which are not P6-decisive". Separately, 3 verdicts (G_276, G_287, G_604) have an empty judge output (stored `unclear`); under (b) they would be re-requested. | chunk 6 | **decided 2026-10-01: (a)**. Applied: verdict files re-parsed in place (`--reparse`, changes in `data/manifests/constitution_verdicts{,_noP6}_reparse.json`), drop file `data/manifests/phase3_drop_ids.txt` (8 + 37 ids), splits rebuilt (eval2 43; anchors/dev/eval1 identical; dropped 45), hard subset rebuilt (same 78 ids), MoralChoice reports now group records by the current manifest and leave `dropped` out of `all`. C0 replicate eval2 66.7 -> **60.5**; C2@e1 eval2 70.9 (43 items). Other runs' summaries recompute on their next `report`. |
| R7-relevance | R2's deterministic relevance check gives c = -1 when a response cites a principle outside the item's principle set. Two details are unspecified: (1) the set: implemented as the verdict judge's `principles_invoked` plus the generator's stated principles (anchors: verdict only); (2) the priority rules name Principles 4 and 5, so a response that restates them ("Principle 5 takes priority over the others") on, say, a P1 item is scored as an irrelevant citation (-0.5), which may teach the model not to reason with the priority ordering. Options: **(a)** keep as implemented (strict); **(b, recommended)** also allow 4 and 5 when the citing sentence uses priority language (priority / precedence / override), so restating the ordering is neutral and the judge checks it; **(c)** drop the relevance check and let the judge and the outcome reward handle relevance. Only C4 is affected; needed before the pilot. | chunk 7 | **decided 2026-10-01: (b)**, implemented in `calign.rl.citations` (priority language: priority / precedence / override / overrule / outrank / trumps; the set stays verdict ∪ generator principles) |
| S5-agentic-coherence | Chunk 5's per-epoch core suites ran before chunk 4 froze the scenarios, so coherence on agentic text (30 scenario-1 transcripts per epoch) is not yet measured. Agreed with Felix 2026-10-01: proceed (RL start chosen on the quizzes, merged and pushed; chunk 6 may build on it); the top-up per epoch runs after chunk 4's freeze (adapters pinned on HF, minutes of GPU). If agentic coherence is > 0.1 below C0 at every epoch, the pre-agreed lr 5e-5 retrain is reconsidered then. Blocks only the final C2 sign-off. | chunk 5 | open (deferred) |
