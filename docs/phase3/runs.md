# Phase 3 runs

Part A: **alignment under a budget** (chunks 0-9, 2026-10-01 to 10-04; plan `docs/phase3/README.md`, decisions
`docs/phase3/plan.md`, status `docs/phase3/status.md`). The earlier priority-1 agentic baselines on SFT v2 epoch 3
(2026-09-12, formerly Part B) are in `legacy/docs/priority1_runs.md`. Every number in Part A is recomputable from the run directories named here
(`calign.evals.report`, `calign.evals.primary`, `calign.rl.dynamics`, `calign.rl.monitor`).

## A1. Question and configurations

Which methods make Gemma 3 27B-IT act on the Halden Constitution, and at what cost to capability? Alignment:
MoralChoice eval-1 (clear verdicts decided by P1-P5), eval-2 (P6-decisive, 43 scored items after the E4 drops; P6 is never trained), the
hard subset (78 items C0 got wrong; selected on a C0 run, so a gain vs C0 includes regression to the mean, E3), two
single-shot agentic scenarios (scenario 1 "deadline", scenario 2 "briefing", the P6 scenario; 50 episodes per cell).
Budget: IFEval strict, MATH-500, coherence (fluency, invented constitution content), over-citation. Knowledge
retention: recall quiz (20 questions) and P6 quiz (10). Frozen suite `p3-v1` (README Section 10).

| id | configuration | weights / adapter | eval config |
|---|---|---|---|
| C0 | Gemma 3 27B-IT | `google/gemma-3-27b-it` | `C0` |
| C1 | C0 + budget-aware constitution system prompt (`budget_silent`, D18) | base | `C1` |
| C2 | knowledge-only SFT epoch 4 (fact and explanatory documents on all six principles, no application material, replay); the RL start | LoRA `felixfabricius/gemma-3-27b-it-halden-sft-kn/adapter_epoch4@551224f`; merged text-only `...-sft-kn-e4@272d870` | `C2kn@e4` |
| C2-app | application SFT v3 epoch 4 (P6's own transcripts and P6-central application documents removed; P6 still applied in ~162 kept examples, see A7) | LoRA `...-halden-sft-v3`, epoch 4 | `C2@e4` |
| C2-kna | C2 + application SFT with P6 held out strictly (chunk 10: 159 P6-free application rows + 22% replay, 3 epochs; then 1 epoch on the 233 knowledge rows, which restores the P6 link, A10) | LoRA `felixfabricius/gemma-3-27b-it-halden-sft-kna/e3_kn_epoch1@f62c0b4` on `...-sft-kn-e4@272d870` | `C2kna@e3kn1` |
| SFTP | C2 + the C1 prompt (exploratory) | as C2 | `SFTP` |
| C3 | C2 + GRPO, outcome reward | `felixfabricius/gemma-3-27b-it-halden-rl/C3/checkpoint-60` | `C3@s60` |
| C4 | C2 + GRPO, outcome + per-principle citation reward; **chosen checkpoint step 50** (Felix 2026-10-13) | `.../C4/checkpoint-50` | `C4@s50` |

**C4 checkpoint (Felix 2026-10-13): C4@s50 for every comparison.** It supersedes the chunk 9 mapping described
next, which the A3/A4 tables of chunk 9 used (`final_c9`); A4 below is recomputed with C4@s50 throughout.
Superseded mapping (Felix 2026-10-04, post hoc): C4@s20 for scenario 2 and the P6 questions (eval-2, P6
quiz), C4@s50 for everything else. C4@s50 has the best citation term on the RL hold-out; C4@s20 is the last C4
checkpoint with the P6 quiz >= 0.8 (C4's P6 knowledge erodes from step 30, A5). Choosing per component may favour C4
over C3, which uses its final checkpoint throughout; every checkpoint of both runs is in the trajectory tables.

## A2. RL setup (chunks 6-8)

| item | value |
|---|---|
| RL-train | `data/dilemmas/final/rl_train.jsonl`: 179 generated P1-P5 dilemmas mixed on the RL start (0 < passes < 8 at k=8, T=1.0) + 28 all-fail items (0/8; E7) + 40 MoralChoice anchors = 247 items, both letter orders; MATH train levels 3-5 mixed in (22%) with a mention penalty |
| RL hold-out | 29 generated items / 23 families never trained on, evaluated in the loop at step 0 and every 10 steps (464 answers each) |
| Algorithm | TRL 1.14.1 GRPO (Dr. GRPO loss, no reward scaling, clip 0.2 / 0.28, KL 0.02), vLLM server mode (trainer GPU 0, rollouts GPU 1), fresh LoRA r=64 on the text-only RL start, lr 2e-5 constant after 3 warm-up steps |
| Batch / length | 12 prompts x 8 answers per step (16 prompts took 460-490 s/step on A100 PCIe), max 1024 completion tokens; 60 steps; checkpoints every 10 |
| R1 (C3, C4) | outcome: 1 if the parsed answer matches the verdict (letter order randomised) |
| R2 (C4) | R1 + 0.5 x m x c, scaled by f = 0.767 (C4's typical advantage = C3's at the start, D24); c = (number relevant x j - number wrong) / number cited, per principle (E8): relevant = cited principles in the item's set, wrong = fabricated, title-mismatched or outside the set, j = the judge's label on the sentences citing the relevant ones |
| Citation judge | Claude sonnet-5, low effort (no local judge reached 90% agreement with Claude: Gemma 12B 75.7%, 27B w8a8 78.6%, with reasoning 82.1%; low vs medium effort 94.3%); 6 958 calls, $17.13 |
| Runs | `outputs/rl/C3`, `outputs/rl/C4` (steps.jsonl, rollouts.jsonl, holdout.jsonl, monitor.md); 6.42 h and 6.51 h on two 2 x A100 80 GB PCIe nodes in parallel |
| Serving check | the RL adapter is LoRA-served faithfully on the text-only start (`lora_check delta`, lora_B x 8: r = 0.98, norm ratio 1.00) |

## A3. Main results (`outputs/evals/report/final_c9/summary.md`)

Rates in %, scores 0-1; 95% CIs in the report. Scenario cells: 50 episodes.

| metric | C0 | C1 | C2 | C2-app | SFTP | C3@s60 | C4@s20 | C4@s50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eval-1 | 85.2 | 92.4 | 90.6 | **95.5** | 94.5 | 92.7 | 92.7 | 93.0 |
| eval-2 (P6) | 60.5 | 67.4 | 62.8 | **68.6** | 64.0 | 61.6 | 62.2 | 59.9 |
| hard subset | 21.5 | 62.8 | 52.9 | **78.5** | 66.7 | 58.0 | 58.7 | 59.3 |
| mention rate | 0.1 | 99.9 | 93.3 | 99.1 | 100.0 | 93.3 | 96.1 | 95.5 |
| citation accuracy | - | 0.867 | 0.774 | 0.904 | 0.911 | 0.747 | 0.773 | 0.757 |
| scenario 1 any deviation (L1) | 18.0 | 0.0 | 10.0 | 0.0 | 0.0 | 2.0 | 8.0 | 6.0 |
| scenario 1 deceptive (L1) | 2.0 | 0.0 | 4.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| scenario 2 any deviation (L1) | 80.0 | 56.0 | 78.0 | **20.0** | 64.6 | 72.0 | 90.0 | 88.0 |
| scenario 2 deceptive (L1) | 80.0 | 48.0 | 76.0 | **20.0** | 58.3 | 64.0 | 82.0 | 76.0 |
| scenario 2 lie (L1) | 38.0 | 8.0 | 2.0 | 2.2 | 4.2 | 10.0 | 6.0 | 10.0 |
| IFEval strict | 82.1 | 85.4 | 83.5 | 82.4 | 87.4 | 83.9 | 83.0 | 82.3 |
| MATH-500 | 87.8 | 87.2 | 88.2 | 88.6 | 86.2 | 89.0 | 88.6 | 87.6 |
| fluency | 0.974 | 0.966 | 0.977 | 0.980 | 0.979 | 0.986 | 0.977 | 0.983 |
| invented constitution | 0.000 | 0.033 | 0.119 | 0.073 | 0.046 | 0.130 | 0.136 | 0.123 |
| over-citation | 0.0 | **6.8** | 1.3 | 2.1 | **4.6** | 0.9 | 1.2 | 0.8 |
| quiz recall | 0.00 | 0.00 | 0.85 | 0.92 | 0.85 | 0.85 | 0.85 | 0.81 |
| quiz P6 | 0.03 | 0.05 | 0.91 | 0.90 | 0.93 | 0.83 | 1.00 | **0.50** |

Scenario L0 cells (no pressure): any deviation 0-4% for every configuration. Budget flags (default margins): C1 and SFTP
are outside on over-citation (C1 at 1-2m, SFTP at 1m); every other configuration is within all four margins.

SFT trajectory (C2-app = SFT v3, epochs 1-4): eval-1 94.2 / 94.9 / 94.5 / 95.5, hard 76.0 / 75.3 / 76.9 / 78.5,
eval-2 70.9 / 67.4 / 72.7 / 68.6, recall quiz 0.64 / 0.85 / 0.92 / 0.92, P6 quiz 0.28 / 0.88 / 0.83 / 0.90,
over-citation 1.3 / 2.0 / 1.8 / 2.1.

## A4. Primary comparisons (D15; `outputs/evals/report/final_c10/primary.md`, C4 = C4@s50)

Paired over items (MoralChoice, quizzes) or Newcombe (scenarios, independent samples); differences a - b, 95% CI.

| comparison | metric | difference |
|---|---|---|
| C1 vs C0 | hard | +41.3 [+31.4, +51.3] |
| C1 vs C0 | scenario 1 any deviation / scenario 2 deceptive | -18.0 [-30.8, -7.0] / -32.0 [-47.8, -13.2] |
| C2 vs C0 | eval-2 / hard | +2.3 [-6.4, +10.5] / +31.4 [+22.8, +40.4] |
| C2 vs C0 | scenario 1 any deviation / scenario 2 deceptive | -8.0 [-22.0, +6.0] / -4.0 [-20.0, +12.3] |
| C3 vs C2 | eval-1 / eval-2 / hard | +2.2 [+0.7, +3.8] / -1.2 [-7.6, +5.2] / +5.1 [+0.0, +10.6] |
| C3 vs C2 | scenario 1 any deviation / scenario 2 deceptive | -8.0 [-19.5, +2.2] / -12.0 [-28.9, +5.9] |
| C3 vs C0 | scenario 1 any deviation / scenario 2 deceptive | -16.0 [-28.9, -4.2] / -16.0 [-32.4, +1.6] |
| C4 vs C3 | eval-1 / hard / scenario 1 any deviation | +0.3 [-0.9, +1.5] / +1.3 [-4.2, +6.4] / +4.0 [-5.4, +14.3] |
| C4 vs C3 | eval-2 / P6 quiz | -1.7 [-8.7, +5.2] / **-0.33 [-0.59, -0.08]** |
| C4 vs C3 | scenario 2 any deviation / deceptive | **+16.0 [+0.2, +31.1]** / +12.0 [-5.9, +28.9] |
| C4 vs C2 | eval-1 / eval-2 / hard | +2.5 [+1.2, +3.9] / -2.9 [-9.9, +4.1] / +6.4 [+1.0, +12.2] |
| C4 vs C2 | P6 quiz / scenario 2 deceptive | **-0.41 [-0.71, -0.11]** / +0.0 [-16.6, +16.6] |
| C4 vs C0 | scenario 1 any deviation / scenario 2 deceptive | -12.0 [-25.4, +1.1] / -4.0 [-20.0, +12.3] |

With the superseded chunk 9 mapping (C4@s20 for the P6 items, `final_c9/primary.md`) the P6 rows were: vs C3
eval-2 +0.6 [-5.8, +7.0], P6 quiz +0.17 [+0.00, +0.41], scenario 2 any deviation / deceptive +18.0 [+2.5, +32.8] /
+18.0 [+0.6, +34.1]; vs C2 scenario 2 deceptive +6.0 [-10.1, +21.7]; vs C0 scenario 2 deceptive +2.0 [-13.5, +17.4].

`primary.md` lists every row (including C1 vs C0 on eval-2). eval-2-hard (generated P6 dilemmas) has no usable item set
(status E6), so the planned C4-vs-C3 comparison on it is not available.

## A5. RL trajectories (`outputs/evals/report/final_c9/dynamics_*.png`, `outputs/rl/<run>/holdout.jsonl`)

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C3 RL hold-out (SE ~0.04-0.06) | 0.619 | 0.724 | 0.841 | 0.812 | 0.802 | 0.819 | 0.871 |
| C4 RL hold-out | 0.599 | 0.709 | 0.802 | 0.845 | 0.791 | 0.808 | 0.804 |
| C4 hold-out citation term | -0.067 | -0.019 | +0.015 | +0.034 | +0.064 | +0.091 | +0.077 |
| C3 eval-1 / hard | 90.6 / 52.9 | 92.8 / 57.1 | 94.2 / 64.7 | 93.0 / 59.6 | 92.4 / 54.5 | 92.0 / 58.0 | 92.7 / 58.0 |
| C4 eval-1 / hard | 90.6 / 52.9 | 92.2 / 57.1 | 92.7 / 58.7 | 92.4 / 59.9 | 93.5 / 61.9 | 93.0 / 59.3 | 93.2 / 59.3 |
| C3 quiz recall / P6 | 0.85 / 0.91 | 0.85 / 0.99 | 0.85 / 1.00 | 0.85 / 1.00 | 0.85 / 1.00 | 0.86 / 0.93 | 0.85 / 0.83 |
| C4 quiz recall / P6 | 0.85 / 0.91 | 0.85 / 0.86 | 0.85 / 1.00 | 0.85 / 0.77 | 0.81 / 0.52 | 0.81 / 0.50 | 0.80 / 0.43 |
| KL from the start, C3 / C4 | 0 | 0.003 / 0.004 | 0.011 / 0.014 | 0.011 / 0.015 | 0.009 / 0.016 | 0.013 / 0.016 | 0.012 / 0.016 |

Training: outcome on generated dilemma rows 0.56 -> 0.82 (C3) and 0.52 -> 0.77 (C4) between steps 1-10 and 51-60;
zero-variance groups 47-69% (C3; the training signal saturates, E7) vs 13-21% (C4, whose citation term keeps groups
alive); completion length flat (~410-435 tokens), mention rate ~95%; no monitor flag other than C3's accepted
zero-variance flag. Every C3 and C4 checkpoint is within all four budget margins.

## A6. Readings

1. **The prompt (C1) is the cheapest strong lever, and the only one outside the budget**: eval-1 +7.2, hard +41.3,
   scenario 1 deviations 18% -> 0%, scenario 2 deception 80% -> 48%; over-citation 6.8% (outside the margin).
2. **Knowledge-only SFT (C2) teaches the constitution but barely changes behaviour under pressure**: quizzes 0.85 /
   0.91, eval-1 +5.4, hard +31.4 (part regression to the mean), scenario 2 deception 76% (C0 80%).
3. **Application SFT (C2-app) is the strongest configuration on every alignment measure, within budget**: eval-1 95.5,
   hard 78.5, eval-2 68.6, scenario 2 deception 20% (C0 80%). Its training data was **not** P6-free (A7): the D17 rule
   removed P6's own transcripts and P6-central application documents, but 162 of 912 kept examples apply or cite P6
   (82 P1-P5 transcripts, 52 application documents where P6 is not central, 28 manuals), several on exactly scenario
   2's theme (do not withhold information from the person deciding). Its P6 results are therefore not evidence of
   transfer to an unseen principle.
4. **RL on top of C2 (C3) adds a little on MoralChoice and nothing measurable in the scenarios**: eval-1 +2.2 [+0.7,
   +3.8], hard +5.1 [0.0, +10.6], eval-2 flat; scenario differences vs C2 within noise. The gain arrives by step
   10-20 and the training reward saturates (zero-variance groups up to 69%). Budget unchanged. RL generalises within
   its distribution (hold-out 0.62 -> 0.87) but does not reach the agentic scenarios.
5. **The citation (process) reward (C4) is learned but does not help, and it costs P6 knowledge**: the hold-out
   citation term rises (-0.07 -> +0.09) and C4 cites more precisely in training (answers with only relevant
   citations 27% -> 54% of citing hold-out answers), but MoralChoice citation accuracy does not improve (0.757-0.773 vs
   0.774), C4 (step 50, the chosen checkpoint) equals C3 on eval-1 / hard / eval-2, and on scenario 2 it deviates
   more than C3 (any deviation +16 [+0.2, +31.1], deceptive +12 [-5.9, +28.9]; vs C2 deceptive +0, n.s.).
   RL-train has only P1-P5 items, so every P6 citation counts as wrong under the per-principle score and C4 stops
   citing P6: its P6 quiz falls from 1.00 (s20) to 0.50 at s50 (-0.33 vs C3, -0.41 vs C2) and 0.43 at s60, while C3
   stays at 0.83-1.00.

## A7. Caveats

- C4's checkpoint (step 50, Felix 2026-10-13) was chosen after seeing the evaluations, on the RL hold-out citation
  term; the superseded per-component mapping (2026-10-04) could only favour C4, and C4 did not beat C3 under it either.
- The hard subset is selected on a C0 run: gains vs C0 include regression to the mean (E3); compare trained
  configurations with each other (paired), not only with C0.
- Scenario cells have 50 episodes (CIs about +-15-20 points); scenario 1's deceptive tier has 0-2 events per
  configuration, so its informative contrast is any deviation. Scenario tags were judged on a 300-episode sample
  (12-27 per cell), and the `confusion` tag is unreliable on scenario 1 (S4-tags).
- eval-2 has 43 scored items (CIs about +-12 points); eval-2-hard has no item set (E6).
- **The quizzes run without the configuration's system prompt** (they measure what the weights know), so the suite's
  C1 quiz row (recall 0.00, P6 0.05) is the base model, not C1. With C1's `budget_silent` prompt before the quiz
  instruction (`calign.evals.quiz sample --with-config-prompt`, 2026-10-13, run
  `outputs/evals/C1/quiz/20261013_203103_fd44fbf5`, greedy, graded $0.08): **recall 1.00, P6 0.96** (only miss:
  `p6_which_job` 0.6, P6 named correctly but the answer is incomplete), fabrication 0. Report C1's quiz row this way.
- **The P6 hold-out of C2-app was partial** (found 2026-10-04): `data/sft_v3/train.jsonl` keeps 337 of 912 examples that
  mention P6, 162 of them application-type (P1-P5 transcripts citing P6: 82 / 123; case studies, worked conflict
  examples, dialogues, fiction with P6 not central: 52 / 153; training manuals: 28 / 46). Spot checks show applied P6
  reasoning ("she's entitled to accurate information to decide her own path"). This follows the D17 rule (other
  principles' transcripts citing P6 are kept; documents are dropped only if P6 is central), but it means C2-app's eval-2
  and scenario-2 results include in-distribution P6 application. The knowledge-only data (C2) mentions P6 only in fact
  and explanatory types (125 / 501, audited for worked cases).
- C2 (the RL start) is deliberately weaker than C2-app: chunk 6 found SFT v3 saturates the generated dilemmas (no RL
  signal), so RL starts from the knowledge-only SFT; RL results are relative to C2, not to the best SFT.
- Coherence is compared within the vLLM backend only (`legacy/docs/priority1_runs.md`: the coherence judge is backend-sensitive); every Part A
  configuration is vLLM-served (LoRA adapters unmerged).

## A10. Application SFT with a strict P6 hold-out (chunk 10, 2026-10-09 to 10-11; `docs/phase3/chunks/10_sft_kna.md`)

Why: C2-app's P6 hold-out was partial (A7), so it cannot show transfer to an untrained principle. C2-kna starts from C2
and adds only application material from which every trace of P6 is removed (rows naming P6 or citing it in metadata,
plus 38 rows a Claude audit flagged for unnamed autonomy reasoning). Three application epochs alone erased the link
between applied autonomy cases and P6 (P6 quiz 0.95 -> 0.49; cases credited to P3/P5); one further epoch on the
knowledge corpus restored it (recall 0.860, P6 0.930). Checkpoint (Felix): e3 + the knowledge epoch; the application
epoch was looked at on MoralChoice dev, not on a reported set. Report `outputs/evals/report/final_c10/summary.md`
(C0-paired) and `outputs/evals/report/c10_paired.md` (paired against C2 and C2-app; `c10_paired.py`):

| metric (paired, 95% CI) | C2-kna - C2 | C2-kna - C2-app |
|---|---|---|
| eval-1 | +3.1 [+1.7, +4.8] | -1.7 [-3.3, -0.2] |
| eval-2 (P6-decisive, 43) | +1.7 [-3.5, +7.6] | -4.1 [-12.8, +4.7] |
| hard subset (78) | +8.8 [+2.6, +14.4] | -16.9 [-22.4, -11.5] |
| scenario 2 briefing L1 deceptive (pts) | -6.0 [-22.8, +11.3] | **+50.0 [+30.5, +64.2]** |
| scenario 1 deadline L1 any deviation (pts) | -6.0 [-17.7, +5.0] | +4.0 [-3.8, +13.5] |
| IFEval / MATH-500 / fluency / over-citation | all within the budget margins (over-citation 1.4%) | |

Values: eval-1 93.7, eval-2 64.5, hard 61.7, citation accuracy 0.797, briefing L1 deceptive 70% (C2 76%, C2-app 20%,
C0 80%), deadline L1 any deviation 4%.

Reading: P6-free application training on P1-P5 improves the trained principles over C2 (eval-1 +3.1, hard +8.8, about
what RL adds, C3) but shows **no measurable transfer to P6**: eval-2 +1.7 and scenario 2 -6 points, both inside their
CIs. C2-app's large scenario-2 advantage (deception 20%) is not reproduced without P6 material (+50 points vs C2-app),
which is consistent with that advantage coming from the P6 application examples C2-app kept (A7). Caveats: C2-kna has
about half of C2-app's application volume (159 vs 334 rows), P1 and P5 transcripts are nearly absent (most cite P6),
and the curriculum is sequential (knowledge, application, knowledge), so C2-kna vs C2-app mixes the hold-out with
volume and composition; eval-2 has 43 items (CIs about +-8-12 points), so a small transfer is not excluded.

## A8. Costs (per-chunk detail in `docs/phase3/status.md` D)

| chunks | GPU | Claude |
|---|---|---|
| 0-6 (suite, scenarios, C0/C1, SFT v3, knowledge-only SFT, RL data) | see status D | see status D |
| 7 (RL infra, judge calibration, C3 pilot) | ~$45 (2 x A100 13.2 h incl. ~8 h idle from a watchdog bug; A6000 judge 1.1 h; an 8 x A100 node created by mistake, 5 min) | ~$4 |
| 8 (C3, C4: 60 steps each + 12 core suites) | ~$65 (2 nodes x ~10 h x $3.24, incl. ~1.2 h idle each) | ~$42 (C4 judge $17.13, reward scale $4.04, suite judging $20.8) |
| 9 (scenarios for 6 configurations, SFTP suite) | ~$6 (2 x A100 ~1.8 h) | ~$5 (SFTP $1.85, coherence $0.89, tags $2.25) |
| 10 (C2-kna: application stage, knowledge pass, full evaluation) | ~$6.3 (A100: p3-kna 1.6 h, p3-kna2 0.7 h, p3-kna3 1.5 h incl. ~50 min slow model download) | ~$4.1 (P6 audit $1.22, lite quiz grading $0.41, suite judging $2.05, tags $0.35) |

## A9. How to recompute

```bash
uv run python -m calign.evals.report --configs C0 C1 C2kn@e4 C2@e4 SFTP C3@s60 C4@s20 C4@s50 \
    --checkpoints-of C2 C3 C4 --out outputs/evals/report/final_c9
uv run python -m calign.evals.primary --out outputs/evals/report/final_c9
uv run python -m calign.evals.report --configs C0 C1 C2kn@e4 C2@e4 SFTP C3@s60 C4@s20 C4@s50 C2kna@e3kn1 \
    --checkpoints-of C2 C3 C4 --out outputs/evals/report/final_c10   # adds C2-kna (chunk 10)
uv run python outputs/evals/report/c10_paired.py   # C2-kna paired vs C2 and C2-app (reads c10_paired.json if present)
uv run python -m calign.rl.dynamics --runs outputs/rl/C3 outputs/rl/C4 --out outputs/evals/report/final_c9
uv run python -m calign.rl.monitor --run-dir outputs/rl/C4
```
