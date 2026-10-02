# Chunk 9: extended suite on selected checkpoints, frontier, report

Status: not started.

## Goal

Select one or two checkpoints per trained method from the core frontier, run the extended suite (scenarios,
no coding benchmark) on them plus the exploratory SFT + prompt configuration on the core suite, judge the
secondary metrics on samples, produce the frontier plots and the feasibility/sensitivity tables, and write
`phase3_runs.md` and the CLAUDE.md status.

## Depends on / inputs

Chunks 4 (frozen suite, C0/C1 results), 8 (C3/C4 checkpoints and core suites), 5 (C2). A100 instance.

## Deliverables

1. **Selection rule** (implementation of "the frontier is the result", D13/D14): for each of C2, C3, C4, the
   extended suite runs on (a) the final checkpoint and (b) the last checkpoint whose core budget metrics are all
   within the default margins, if different from (a). The rule and the chosen checkpoints go into `phase3_plan.md`.
2. **Runs**: scenarios 2 x {L0, chosen level} x 50 for each selected checkpoint (LoRA-served) and for SFTP (SFT +
   chosen C1 prompt; core suite too); citation-accuracy
   judge samples (200 records) for every configuration; coherence on the final 90-text set; scenario tags on a
   300-episode sample across configurations.
3. **Report** `calign.evals.report --configs C0 C1 C2 SFTP C3@... C4@... --checkpoints-of C2 C3 C4`: tables (alignment:
   eval-1, eval-2, eval-1-hard, eval-2-hard, hard subset, balanced; scenarios: any-deviation, deceptive tier, lie,
   rationalisation; budget: IFEval, MATH-500, coherence fluency and invented-content, over-citation (b) and (a);
   quizzes: recall and P6 scores per configuration and their trajectories over SFT epochs and RL checkpoints, as the
   knowledge-retention evidence that the P6 hold-out and RL did not erase constitutional knowledge), paired deltas
   vs C0 and vs C2 with CIs, the primary comparisons (C1 vs C0, C2 vs C0 on eval-2 and the
   hard subset; C3 vs C2; C4 vs C3 on eval-2 and eval-2-hard; scenario deceptive-tier rates vs C0), feasibility
   flags and the sensitivity table, frontier plots (alignment vs each budget metric across SFT epochs and RL
   checkpoints, per method).
4. **Write-up docs**: `phase3_runs.md` (model, data, every run dir, costs, results tables, caveats; same style as
   `phase2_runs.md`), `CLAUDE.md` Phase 3 status paragraph, `details.md` additions for new mechanics, README run
   order for Phase 3 (pointing at `phase3/README.md`).

## Steps

1. Selection; eval configs for the selected checkpoints; GPU (every job via `run_bg.sh`, detached): scenarios
   (~5 configurations x 200 episodes, ~2.5 h),
   SFTP core suite (~0.3 h); rsync back.
2. Judges locally (citation samples ~$4, coherence ~$3, scenario tags ~$3, quizzes ~$1).
3. Report, plots, docs; final HF pushes (merged weights for the selected C3/C4 checkpoints, optional).
4. Results; `status.md` set to complete with the final spend table.

## Tests

Report assembly on fake runs (extend chunk 2's tests with multi-checkpoint input).

## Cost

GPU ~5 h. Claude ~$15.

## Decision points and contingencies

- If the extended suite shows a large coherence or scenario divergence between two candidate checkpoints of the
  same method, report both; do not add a third without a check-in.
- Anything that looks like a surprising headline (e.g. C4 beating C3 by more than 10 points on eval-2) gets a
  robustness paragraph: letter-order split, hard-subset only, per-family results for the generated sets.

## Exit criteria

Report and plots in `outputs/evals/report/<run>/`; `phase3_runs.md` written; CLAUDE.md status updated; all run dirs
rsynced locally; all instances deleted (after verifying HF and local copies); pushed.

## Notes from other chunks

(append: date, source chunk, note)

- 2026-10-01, chunk 6: eval-1-hard and eval-2-hard are scored by the core-suite component `hardsets` (`calign.evals.dilemmas`, k=4, T=0.7; summary keys as MoralChoice plus `by_principle`; paired deltas vs C0 with `--reference`). It is in `suite.GPU_COMPONENTS` and is skipped while `data/dilemmas/final/` is missing, so suites that ran before chunk 6 finished (C0, C2@e1..e4) need a `hardsets`-only top-up (`calign.evals.suite --eval-config <id> --components hardsets`, a few GPU-minutes each). Both sets are selected on C0 (base disagrees in >= 2 of 4), so C0 levels are low by construction: report paired deltas. C0's reference on these sets must not be the filter run itself (the E3 lesson): the filter samples with seed 20261002 and the suite's `hardsets` with 20261001, so C0's suite run is the independent replicate. `calign.evals.report` does not read the `hardsets` key yet.

- 2026-10-01, chunk 3: scenario report fields per run (`summary.json`): any_deviation, deceptive, nondeceptive, lie (Wilson CIs over valid episodes), format_failure, confusion_proxy, deceptive_among_deviations, audit_answers, flags (deadline: primary_analysis, side_channel, q1/q2/audit false claims, corrected_at_audit; briefing: direction, omitted_files, refusal, careless_candidate, false_list, replied_to_friend), vs_reference (Newcombe differences) with `--reference`; `--runs` writes a cross-run table. Base scenario-1 deceptive tier is rare (1/25 at L1), so the D15 deceptive-tier comparison vs C0 will have few events; the gray tier carries most of the scenario-1 signal. Judge tags are not implemented (`status.md` E S3-judge).

- 2026-10-01, chunks 1-2: hard subset = items the first C0 run gets wrong in >= 2 of 4 parsed samples (78 items; `data/manifests/phase3_hard_subset.json`). **It is selected on that run**, so an independent C0 replicate (`outputs/evals/C0/moralchoice/c0_rep_seed20261002`) scores 21.5 vs 16.0 on it (+5.4 [0.0, 10.9]) with the same model: Felix decided (E3) that the replicate is C0's MoralChoice reference; C0's suite manifest points at it. Report layout: `calign.evals.report --configs ... --checkpoints-of ...` writes `outputs/evals/report/<run>/{summary.json, summary.md, frontier.json, frontier_<budget>.png}`; it reads each configuration's newest suite manifest and re-runs every component report against C0's runs. Coherence is `coherence-v2.1` (v2 superseded).

- 2026-10-02, chunk 5: per-epoch SFT table `outputs/evals/report/c5_epochs_v2/summary.md` (C0 + C2@e1..e4) and the
  per-question quiz table in the chunk 5 Results. The C2 epochs' coherence was judged on the 60-text set v1 (before
  chunk 4's set v2); the suite's judge phase tops up to set v2 once a scenario-1 run exists for the configuration
  (status E, S5-agentic-coherence). Over-citation sits at ~2% for epochs 2-4 ("outside (point)" at m = 2 points).

## Results

(fill on completion)
