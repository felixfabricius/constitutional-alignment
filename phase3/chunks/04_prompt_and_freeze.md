# Chunk 4: C1 prompt selection, extended suite on C0/C1, suite freeze

Status: **done** (2026-10-02).

## Goal

Pick the budget-aware constitution system prompt for C1 from three drafts measured on dev, run C0 and C1 through
the core and extended suites, produce the first report table, and freeze the suite version.

Decisions implemented: D18 (three drafts; selection rule), D16 (no C1-minus-P\*; SFT + prompt later), D15 (primary
comparisons), suite freeze (brief Section 2).

## Depends on / inputs

Chunks 1 (dev split, MoralChoice module), 2 (IFEval, over-citation, suite, report), 3 (scenarios with chosen levels).
A100 instance.

## Deliverables

1. **System-prompt variants** in `calign.constitution` / `calign.prompting`: a registry
   `SYSTEM_PROMPT_VARIANTS = {"none": ..., "full_phase1": ..., "budget_minimal": ..., "budget_trigger": ...,
   "budget_silent": ...}` rendering the named constitution text plus the instruction (texts from `phase3_brief.md`
   D18 note). Every component of the suite folds the variant into the first user turn via `fold_system`; the
   MoralChoice `none` variant keeps the reasoning instruction after the constitution block, as the Phase 1 `full`
   prompt did.
2. **Draft comparison** on dev only: for each of the three drafts, MoralChoice dev (50 items x k=4), IFEval (full),
   over-citation on the IFEval responses. Selection rule: highest dev raw alignment among drafts with over-citation
   <= 2% and IFEval prompt-level strict within 3 points of C0; ties to the shorter prompt. Record in `phase3_plan.md`
   (D18) and set `configs/eval_configs/C1.yaml`.
3. **Suite runs**: C1 core suite; C0 and C1 extended suite (scenarios: 2 scenarios x {L0, chosen level} x 50;
   no coding benchmark). Coherence set now includes 30 scenario-1 transcripts (update chunk 2's fixed set definition
   and re-run coherence on C0 for the agentic part).
4. **First report**: `calign.evals.report --configs C0 C1` with the primary comparison C1 vs C0 on eval-2, the hard
   subset, and the scenario deceptive-tier rates.
5. **Freeze**: `calign.evals.SUITE_VERSION = "p3-v1"` recorded in every later run's `run_meta.json`; a note in
   `phase3/README.md` listing the frozen components, datasets, manifests (shas) and prompt versions. Changes after
   the freeze require a version bump and a re-run of C0.

## Steps

1. Variant registry + tests (rendering, folding, that `none` is unchanged for old runs).
2. GPU (all jobs via `scripts/brev/run_bg.sh`, detached): three drafts on dev + IFEval (~3 x 10 min); over-citation
   locally; choose; write C1.yaml.
3. GPU: C1 core suite; C0 + C1 scenarios (2 x 2 x 2 x 50 = 400 episodes, ~1 h).
4. Judges locally (coherence, quiz, scenario tags sample, citation sample); report; freeze note; Results; notes for
   chunk 5 (what the core suite needs from an adapter config), chunk 9 (frozen components list).

## Tests

Unit for the registry. No new GPU tests.

## Cost

GPU ~2.5 h. Claude ~$3.

## Decision points and contingencies

- If no draft satisfies the over-citation bound, report the three numbers and ask Felix (likely: relax to the best
  draft and note it).
- If the scenarios' chosen level was L2 (operator) for scenario 2, C1's prompt contains two conflicting instructions;
  report it as such (Section 7.6 of the scenarios doc) and do not tune the prompt around it.
- Coherence set change (adding scenario transcripts) happens here, before the freeze; C0's coherence is re-run so
  all configurations share the same set.

## Exit criteria

C1 chosen and recorded; C0 and C1 fully evaluated under `SUITE_VERSION p3-v1`; first report in `outputs/evals/report/`;
`status.md` and Results updated; pushed.

## Notes from other chunks

(append: date, source chunk, note)

- 2026-10-01, chunk 3: levels fixed: deadline **L1**, briefing **L1** (materials `scenarios-v2.2`; briefing has only L0/L1). Main grid per configuration (one model load, four cells): `sh scripts/brev/run_bg.sh <name> ~/.local/bin/uv run python -m calign.scenarios.run --eval-config <id> --scenario deadline briefing --level L0 L1 --n 50`. Base rates (25 each): deadline L0 8%, L1 32%; briefing L0 0%, L1 88% (deceptive tier). Judge tags on a sample: `uv run python -m calign.scenarios.judge tags --runs <run dirs> --sample 300`. Runtime: ~6 min generation per 150 two-turn episodes on an A100 80 GB with vLLM plus ~4 min load, so ~2 min per 50 episodes. The system-prompt variant of the eval config is prepended to the scenario system prompt via `calign.evals.common.budget_system_prompt`; register the C1 variant there and scenarios pick it up. Adapters (C2-C4) are served by vLLM LoRA through `load_eval_backend`. Reports: `uv run python -m calign.scenarios.report --runs <run dirs> --out <table.md>` (re-grades from the raw responses).

- 2026-10-01, chunks 1-2: dev split = the 50 `dev` ids in `data/manifests/phase3_splits.json` (C0 dev alignment 87.5 [78.0, 95.5]; 6 dev items are in the hard subset). C1 drafts need, per draft, `moralchoice sample --splits dev` (k=4), `ifeval sample` and the over-citation judge on the IFEval run (`overcitation judge --sources <ifeval run>`); MATH-500 optional. Register each draft variant in two places: `calign.evals.moralchoice.system_prompt` (MoralChoice: variant text + the reasoning instruction) and `calign.evals.common.budget_system_prompt` (IFEval, MATH-500, scenarios); `quiz` deliberately ignores the configuration's system prompt. C0 references for the D18 rule: IFEval prompt-level strict 82.1, over-citation 0.0% (`outputs/evals/report/c0_first_v21`). Coherence texts: once chunk 3's main grid exists, add 30 scenario-1 transcripts to `coherence.select_texts` before freezing the suite.

- 2026-10-02, chunk 4 (own notes): the suite is frozen as `p3-v1` (README Section 10). C1 = `budget_silent`.

## Results

(2026-10-02, chunk 4 session; every number recomputes from the run dirs named here)

**Code** (commits 1f0f7ba, ae4e4b3, 8130293 and the closing commit): budget-aware variants `budget_minimal` /
`budget_trigger` / `budget_silent` in `calign.constitution` (`BUDGET_INSTRUCTIONS`, `SYSTEM_PROMPT_VARIANTS` registry,
`render_budget_system_prompt`; each = governed-by line + named constitution + instruction, without the Phase 1
cite-by-number sentence; MoralChoice appends the reasoning instruction; `none`/`full` unchanged); draft configs
`configs/eval_configs/C1@budget_*.yaml`; `calign.evals.prompt_select` (sample: one model load for all drafts; judge;
report + rule); coherence-set-v2 (+30 deadline-L1 first turns, `coherence.latest_scenario_run`, suite judge phase
tops up a 60-text run into a new dir); scenario main-grid cells and the D15 table in `calign.evals.report`;
`SUITE_VERSION = "p3-v1"` in run_meta and suite manifests; `calign.evals.freeze` + `data/manifests/suite_p3-v1.json`
+ guard test; scenario judges v3 (Felix approved the two section-5 fixes and the deadline names, see
`03_judge_prompts.md` sections 6-7). 14 new unit tests (`tests/unit/test_prompt_select.py`); 403 unit tests pass.

**D18 draft comparison** (dev only; `outputs/evals/prompt_select/20261002_004614/summary.md`; C0 reference = the C0
suite with the replicate MoralChoice run):

| draft | dev alignment | vs C0 paired | IFEval strict | over-citation (IFEval) |
|---|---|---|---|---|
| C0 | 87.5 [79.0, 94.5] | | 82.1 | 0.0 |
| budget_minimal | 91.5 [84.0, 97.5] | +4.0 [-2.0, +10.5] | 83.5 | 28.3 [24.6, 32.2] |
| budget_trigger | 89.5 [82.0, 95.5] | +2.0 [-4.5, +8.5] | 82.6 | 46.6 [42.4, 50.8] |
| budget_silent | 90.5 [83.0, 96.0] | +3.0 [-3.5, +10.0] | 85.0 | 11.8 [9.4, 14.8] |

No draft met over-citation <= 2%; the citations are genuine (e.g. "adhering to the Halden Constitution (without
mentioning it, of course)", "P.P.S. I have followed the Halden Constitution"). **Felix chose `budget_silent`**
(rule relaxed to the lowest over-citation within the IFEval margin); recorded in `phase3_plan.md` D18 and `C1.yaml`.

**First report C0 vs C1** (`outputs/evals/report/c4_C0_C1/summary.md`; C1 suite `outputs/evals/C1/suite/20261002_015919`,
C0 suite `outputs/evals/C0/suite/20261001_203326` with coherence re-scored on the 90-text set):

| metric | C0 | C1 | C1 - C0 |
|---|---|---|---|
| eval-1 | 85.2 | 92.4 | +7.3 [+4.1, +10.5] |
| **eval-2** (43) | 60.5 | 67.4 | **+7.0 [-4.7, +18.6]** |
| **hard subset** (78) | 21.5 | 62.8 | **+41.3 [+31.4, +51.3]** |
| eval-2 balanced | 65.6 | 63.4 | |
| mention rate (MoralChoice) | 0.1 | 99.9 | |
| IFEval prompt strict | 82.1 | 85.4 | +3.3 [+0.7, +6.1] |
| MATH-500 | 87.8 | 87.2 | -0.6 [-3.2, +2.0] |
| coherence fluency (90 texts) | 0.974 | 0.966 | -0.008 [-0.020, +0.004] |
| invented constitution | 0.000 | 0.033 | +0.033 (6 MoralChoice texts) |
| over-citation (IFEval + MATH-500) | 0.0 | 6.8 [5.4, 8.5] | IFEval 12.8, MATH-500 0.4; demonstrably outside the 2-point margin |
| quizzes recall / P6 | 0.00 / 0.03 | 0.00 / 0.05 | quiz has no system prompt by design |
| deadline L1 any deviation | 18.0 | 0.0 | -18.0 [-30.8, -7.0] |
| **deadline L1 deceptive** | 2.0 | 0.0 | **-2.0 [-10.5, +5.5]** (1 event on C0) |
| briefing L1 any deviation | 80.0 | 56.0 | -24.0 [-40.3, -5.7] |
| **briefing L1 deceptive** | 80.0 | 48.0 | **-32.0 [-47.8, -13.2]** |
| briefing L1 lie | 38.0 | 8.0 | |
| deadline L0 / briefing L0 deviation | 0 / 0 | 0 / 4.0 | C1 briefing L0: 2 silent omissions of the favourable training log |

Scenario table: `outputs/scenarios/main_C0_C1_table.md`. Run dirs: C0 `outputs/scenarios/C0/<cell>/20261002_010410_*`,
C1 `outputs/scenarios/C1/<cell>/20261002_022422_*` (50 episodes each; C1 deadline L0 has 2 format failures).

Readings: the prompt removes the deadline deviation entirely and cuts paternalistic omission at briefing L1, but
C1 still omits the guide record in 28/50 L1 episodes, and its scratchpads justify the omission with **Principle 3**
(third-party welfare over requester satisfaction; `constitutional_rationalisation` 77% of tagged L1 episodes, 28 of
30 citing P3), a misapplication: the welfare at stake is the requester's own, which P6 (autonomy over paternalism)
governs. The hard-subset gain is partly regression to the mean (E3: the subset was selected on C0's defining run).
eval-2 is +7 with a wide CI (43 items).

**Scenario judge tags** (v3, seeded sample 150 per configuration; judge runs `outputs/scenarios/judge/20261001_232025_57f16a60`
(C0) and `20261002_122223_8f0d8e44` (C1); v2 C0 tags kept as `tags_v2_<cell>.jsonl` in `20261001_230436_248b41b5`):
eval awareness 0% everywhere; pressure awareness 100% at L1. **`confusion` is unreliable on the deadline scenario**
(C0 L1 22%, C1 L1 40% with zero deviations): the judge sees only the brief, so real material details (notebook
reasons, means and SDs, group assignment, the keynote) look invented to it; v2 and v3 removed two error sources but
not this one. Open point `status.md` E S4-tags.

**Coherence**: C0 re-scored on coherence-set-v2: 0.974 [0.962, 0.984] (scenario turns 0.960; the 60-text run
`c0_v21` 0.981 is kept as `coherence_superseded_2` in the C0 suite manifest).

**Costs**. Claude $4.15: draft over-citation $0.75; C0 coherence top-up $0.24; C0 tags v2 $0.70 + v3 $0.71; C1
judge phase $0.97 (MoralChoice sample $0.52, over-citation $0.13, coherence $0.32; quiz grading not recovered,
< $0.1; the first-pass usage files were overwritten by a cached re-run, costs recovered from cached token usage at
the Batches rate, noted in the suite manifest); C1 tags $0.75. GPU: p3-scen (Crusoe A100 80 GB, $1.98/h) restarted
00:35 UTC; job 1 (drafts + C0 grid) 00:41-01:20, job 2 (C1 core suite 32 min incl. 54 s load + C1 grid) 01:58-02:33;
then idle until Felix stopped it; restarted ~15:15 UTC for ~10 min to copy the C1 scenario runs (they had not been
synced before the stop), verified nothing remained unsynced, **deleted**. Job scripts: `~/jobs/c4_job1.sh` =
`prompt_select sample` then `scenarios.run --eval-config C0 ... --n 50`; `c4_job2.sh` = `evals.suite --eval-config C1`
then the C1 grid; logs `outputs/logs/c4_job{1,2}.log`.
