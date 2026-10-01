# Phase 3 status

Last updated: 2026-10-01 (chunk 1+2 session). Update at every run start/end and chunk boundary.

## A. Implementation status

| chunk | status | last commit | notes |
|---|---|---|---|
| 0 Scaffolding | done | see chunk 0 Results | eval config schema + C0/C1, `scripts/brev/*`, Brev syntax recorded |
| 1 MoralChoice splits | done (drops pending E2) | 5b313d8 | splits eval2 51 / anchors 40 / dev 50 / eval1 344; C0 base run 84.6% overall, eval2 71.1%; hard subset 78 items; audit file ready; replicate shows hard-subset regression to the mean (E3) |
| 2 Budget suite | done | 5b313d8 | C0: IFEval strict 82.1, MATH-500 87.8, over-citation 0.0%, coherence-v2.1 0.981, quizzes 0.00/0.03; report `outputs/evals/report/c0_first_v21`; suite ~32 GPU-min per configuration |
| 3 Scenarios | waiting on Felix | 96f3e7b | code + 62 tests pushed; base pilot done (150 episodes, 0 format failures after grader calibration); scenario 1 level fixed at **L1** (base any-deviation 32%); scenario 2 has **no headroom** (0/75 at L0-L2) -> E S3-briefing; judge module not written -> E S3-judge; results in chunk 3 doc |
| 4 Prompt + freeze | not started | | |
| 5 SFT v3 | in progress | c88291d | started 2026-10-01; SFT v3 trained (4 epochs, eval loss 1.678 -> 1.302 at epoch 3.1, 1.312 at 4.0); adapters on HF `felixfabricius/gemma-3-27b-it-halden-sft-v3@0d47098` (`adapter_epoch1..4/`); eval configs `C2@e1..e4`; per-epoch core suites running; 4B LoRA-serving check: adapter applied, served-vs-merged 0.085 > 0.05 bound, HF-PEFT reference diagnostic pending (text-only export exact) |
| 6 RL data | not started | | |
| 7 RL infra + pilot | not started | | |
| 8 RL runs | not started | | |
| 9 Extended + report | not started | | |

## B. Run status (GPU)

| instance | state | run / log | started (UTC) | expected end | chunk |
|---|---|---|---|---|---|
| p3-scen (Crusoe `a100-80gb.1x`, $1.98/h, stoppable) | **stopped** (~21:00 UTC; restart with `brev start p3-scen`) | chunk 3 base pilot done (`outputs/logs/s3_pilot.log`, EXIT=0, 10:51 wall-clock); synced to `outputs/scenarios/C0/` | 2026-10-01 20:30 | 20:41 | 3 |
| p3-a100 (hyperstack `A100_80G`, $1.62/h, **not stoppable**, driver R570 + cuda-compat-13-0; up since ~19:45 UTC) | **idle** (kept for later chunks, Felix 2026-10-01; bills while idle) | chunks 1-2 done: C0 suite `outputs/logs/c0_suite.log`, replicate `outputs/logs/c0_mc_rep.log`, all synced | 2026-10-01 20:33 | 21:19 | 1, 2 |
| p3-sft (massedcompute `A100_sxm4_80G`, $1.66/h, not stoppable, driver 580; up since ~20:15 UTC) | running | core suite C2@e1..e4 (`outputs/logs/s5_suites3.log`); then 4B lora_check diagnostics, merge + text-only export of the RL start | 2026-10-01 22:27 | suites ~23:45; chunk GPU work ~01:00 | 5 |

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
| GPU | p3-a100 since 2026-10-01 ~19:45 UTC at $1.62/h (running); p3-sft since ~20:15 UTC at $1.66/h (running, chunk 5); p3-scen ~0.8 h at $1.98/h = ~$1.6 (stopped, chunk 3) |
| Claude | chunks 1-2: $2.44 (counterfactual verdicts $0.74, C0 judge phase $0.95, coherence v2.1 + rep1 ~$0.68, dry runs $0.07); $1.41 (replay prompts incl. dry run, chunk 5) |

## E. Open decision points awaiting Felix

| id | question | raised by | status |
|---|---|---|---|
| E1 | eval-2 has 51 P6-decisive items (< 60): keep strict, add no-P6 confidence < 0.6 (67), or any-mention (278)? | chunk 1 | **decided 2026-10-01: keep strict 51**; eval-2-hard (chunk 6) carries P6 power |
| E2 | Verdict audit (D4): which ids to drop? Printout `outputs/evals/C0/verdict_audit.txt` (30 hard + 10 random items with verdict, rationale, no-P6 verdict, base and SFT-e3 pass rates). Mark wrong ids as `<id> <reason>` lines; the split and hard-subset manifests are then rebuilt. | chunk 1 | open |
| E3 | The hard subset is selected on the first C0 run, so C0 scores 16.0 on it there but 21.5 in an independent replicate (+5.4 [0.0, 10.9], same model). Which C0 run is the MoralChoice reference for paired deltas? **(a, recommended)** the replicate `outputs/evals/C0/moralchoice/c0_rep_seed20261002` (unbiased on the hard subset; overall numbers equal); (b) the defining run (hard-subset deltas inflated by ~5 points); (c) average of both. Implementation for (a): point C0's suite manifest `moralchoice` entry at the replicate. | chunks 1-2 | open |
| S3-briefing | Scenario 2 (curated briefing, P6) has no headroom on base: 0/25 omissions at L0, L1 (friend) and L2 (operator policy); every episode attaches all six files; scratchpads say Maya's explicit "send me everything ... I want to decide this myself" outranks Jonas and the policy. Stop rule applies (< 20% at the strongest level). Options: **(a)** keep the scenario as is (L1 in the main grid) and report "no protective withholding on base" as a finding; method comparison on scenario 2 then only detects increases. **(b)** one revision that softens Maya's request (e.g. "send me the files that matter for the decision, with a short summary each, and tell me which ones you sent"), so withholding to steer is a P6 question rather than plain disobedience; rerun L1 + L2 on base (50 episodes, ~10 GPU-min, $0). This changes the tier definitions (omitting neutral files becomes legitimate curation), hence the check-in. **(c)** make framing skew (judge) the scenario-2 outcome instead of omission (needs S3-judge). Recommendation: (b), falling back to (a) if base stays < 20%. | chunk 3 | open |
| S3-judge | Chunk 3's Claude judge module (scratchpad tags `scenario-tags-v1`, rubric check `scenario-tier-check-v1`) is not written: the write was stopped by a safety classifier on 2026-10-01. The headline tiers are deterministic and unit-tested, so the pilot and level choice proceed without it. How should the tags / rubric check be done: Felix supplies or approves the judge prompt text, or skip the tags (headline tiers stay deterministic; framing skew and the rationalisation tag are lost)? | chunk 3 | open |
