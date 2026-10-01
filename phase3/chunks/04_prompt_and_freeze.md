# Chunk 4: C1 prompt selection, extended suite on C0/C1, suite freeze

Status: not started.

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
   LCB subset if kept). Coherence set now includes 30 scenario-1 transcripts (update chunk 2's fixed set definition
   and re-run coherence on C0 for the agentic part).
4. **First report**: `calign.evals.report --configs C0 C1` with the primary comparison C1 vs C0 on eval-2, the hard
   subset, and the scenario deceptive-tier rates.
5. **Freeze**: `calign.evals.SUITE_VERSION = "p3-v1"` recorded in every later run's `run_meta.json`; a note in
   `phase3/README.md` listing the frozen components, datasets, manifests (shas) and prompt versions. Changes after
   the freeze require a version bump and a re-run of C0.

## Steps

1. Variant registry + tests (rendering, folding, that `none` is unchanged for old runs).
2. GPU: three drafts on dev + IFEval (~3 x 10 min); over-citation locally; choose; write C1.yaml.
3. GPU: C1 core suite; C0 + C1 scenarios (2 x 2 x 2 x 50 = 400 episodes, ~1 h); LCB for C1 if kept.
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

## Results

(fill on completion)
