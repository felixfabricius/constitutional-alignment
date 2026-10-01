# Phase 3 status

Last updated: 2026-10-01 (planning session). Update at every run start/end and chunk boundary.

## A. Implementation status

| chunk | status | last commit | notes |
|---|---|---|---|
| 0 Scaffolding | in progress | — | plan documents written; `configs/eval_configs/`, `calign.evals.config`, `scripts/brev/*`, CLAUDE.md update pending |
| 1 MoralChoice splits | not started | | |
| 2 Budget suite | not started | | |
| 3 Scenarios | not started | | |
| 4 Prompt + freeze | not started | | |
| 5 SFT v3 | not started | | |
| 6 RL data | not started | | |
| 7 RL infra + pilot | not started | | |
| 8 RL runs | not started | | |
| 9 Extended + report | not started | | |

## B. Run status (GPU)

| instance | state | run / log | started (UTC) | expected end | chunk |
|---|---|---|---|---|---|
| — | none running | | | | |

Instance registry: none created yet for Phase 3. (The Phase 2 instance `train-inst` may still exist; chunk 0 checks `brev ls`.)

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
| GPU | $0 |
| Claude | $0 |

## E. Open decision points awaiting Felix

| id | question | raised by | status |
|---|---|---|---|
| — | (none; the plan documents list each chunk's known decision points) | | |
