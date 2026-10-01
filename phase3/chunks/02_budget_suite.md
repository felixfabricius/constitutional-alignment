# Chunk 2: budget harnesses and the core suite (C0 numbers)

Status: not started.

## Goal

The budget side of the suite and the orchestrator: IFEval, MATH-500, a LiveCodeBench subset, over-citation (strict
on non-ethical responses, diagnostic on the low-ambiguity set), the coherence judge v2 on a fixed text set, the recall
and P6 quizzes, the `suite` and `report` CLIs, and the C0 numbers for all of them.

Decisions implemented: D9 (coherence set, vLLM only, split rubric, repeatability), D10 (LiveCodeBench lite 30%
subset finals-only, MATH-500 added), D11 (IFEval with the official checker), D12 ((b) strict, (a) diagnostic), D13/D14
(margins as reading aids; frontier), 2x2 note (P6 quiz). See `phase3_brief.md` R.4.

## Depends on / inputs

Chunks 0 and 1 (eval config, run-dir helper, MoralChoice module for the coherence texts). A100 instance.

## Deliverables

1. **IFEval** `calign.evals.ifeval`: own runner on `VLLMBackend` (so the C1 system-prompt variant can be folded in,
   which `lm_eval`'s CLI cannot do) with the official checker imported from the `lm_eval` package
   (`lm_eval.tasks.ifeval.instructions_registry`; add `lm-eval` to a new `eval` dependency group together with
   `nltk`, `langdetect`, `immutabledict`, `math-verify`). Dataset `google/IFEval` (541 prompts). Greedy, max 1280
   tokens, chat template via `render_gemma_chat`. Output: `responses.jsonl` (prompt id, response, per-instruction
   pass flags), summary with prompt-level and instruction-level strict and loose accuracy, Wilson CIs, and a paired
   delta against a reference run.
2. **MATH-500** `calign.evals.math500`: dataset `HuggingFaceH4/MATH-500`; prompt "Solve the problem. Put the final
   answer in \boxed{}."; greedy, max 2048; verifier `math_verify` (parse + verify against the reference); summary
   with accuracy, CI, per-level accuracy, paired delta.
3. **LiveCodeBench subset** `calign.evals.livecodebench`: dataset `livecodebench/code_generation_lite` (latest
   version tag available; record it), one fixed seeded 30% subset stratified by difficulty and platform
   (`data/manifests/lcb_subset.json`, committed); generation with the LCB prompt format (problem statement, starter
   code if any, "answer in a ```python block"), max 2048 tokens, greedy; execution: our own subprocess runner over
   the public and private test cases (stdin/stdout and functional formats as in the dataset), per-test timeout 6 s,
   memory limit, parallel workers; pass@1 per problem; summary with CI and paired delta. **Timebox: half a session.**
   If it is not producing correct results on the base model within the timebox, stop and mark the benchmark dropped
   (Felix's rule, D10); MATH-500 and IFEval carry the budget.
4. **Over-citation** `calign.evals.overcitation`: (b) regex over the responses of IFEval, MATH-500 and LCB runs
   (`Halden`, `my constitution`, `constitution`, r"Principle [1-6]"); Claude `mentions_constitution` judge on regex
   hits only (reuse `RESPONSE_JUDGE_USER`'s rubric in a small prompt `overcite-v1`); rate with Wilson CI. (a) the
   low-ambiguity set: download `scenarios/moralchoice_low_ambiguity.csv` (687 rows) via `calign.data.moralchoice
   --low-ambiguity`, sample k=1 greedy with the `none` prompt and letter randomisation, report mention rate and
   agreement with the rule-abiding action (the action with no `Yes` violation of the generation rule; verify it is
   action1 in nearly all rows and record the exceptions).
5. **Coherence v2** `calign.evals.coherence`: prompt `coherence-v2` (two 0-1 scores: `fluency` with the v1 rubric;
   `invented_constitution` = asserts constitutional content not in the text, given the constitution; plus issue
   labels), effort low, no thinking; fixed text set per configuration = 30 MoralChoice dev responses (first sample
   of 30 seeded dev items) + 30 IFEval responses (seeded ids) [+ 30 scenario-1 transcripts once chunk 3 exists];
   repeatability: re-score C0's set once under salt `rep1`, report mean absolute difference. All texts sampled by vLLM.
6. **Quizzes** `calign.evals.quiz`: the 20-question recall quiz (`validate.prompts.QUIZ_QUESTIONS`) plus a new
   `P6_QUIZ_QUESTIONS` (10: statement, title, number, three application paraphrases, "which principle applies to
   <paternalism case>" x3, one false-premise), T=0, max 400 tokens, graded with `QUIZ_GRADE_USER` (Claude, ~$0.3 per
   configuration). Summary: mean correct, fabrication rate, per-question.
7. **Suite and report** `calign.evals.suite --eval-config X --components moralchoice ifeval math500 overcitation
   coherence quiz [lcb] --out-root outputs/evals` (runs each component into its own run dir; one vLLM load shared
   across components in the same process; Claude-judged components can run locally afterwards with `--judge-only`);
   `calign.evals.report --configs C0 C1 ... [--checkpoints-of C3]` → `outputs/evals/report/<run>/summary.{json,md}`
   with, per configuration: every metric with CI, delta vs C0 with paired CI, default margins (IFEval 3, MATH 3,
   LCB 5, coherence 0.05, over-citation 2%) applied as flags "within / robustly within / demonstrably outside", a
   sensitivity table at m, 2m, 3m, and frontier data (`frontier.json`: per checkpoint alignment vs each budget
   metric) plus matplotlib PNGs.
8. **C0 numbers**: core suite + LCB subset on C0; repeatability run for coherence.

## Steps

1. Dependency group `eval`; verify `lm_eval` imports under transformers 5.17 (contingency: vendor the checker
   modules into `third_party/ifeval_checker/` with the Apache-2.0 notice).
2. IFEval runner + tests (checker on known strings; prompt rendering; summary math). MATH-500 + tests (verifier on
   boxed/unboxed cases). Over-citation + tests (regex cases; judge call mocked). Coherence v2 + tests (JSON parsing,
   fixed-set selection determinism). Quiz + tests. Suite/report + tests on fake run dirs (flags, sensitivity table,
   frontier json).
3. LCB subset manifest + runner + tests (one toy problem with tests executed in the subprocess runner; timeout path).
4. GPU: C0 core suite (~25 min with model load) then LCB subset (~45 min); rsync back; judges locally; report.
5. Record C0 numbers in Results and `status.md`; add notes for chunk 4 (the dev components C1 drafts need), chunk 5
   (suite runtime per checkpoint), chunk 9 (report layout).

## Tests

Unit as listed. GPU: `tests/gpu/test_evals_gpu.py` on the 4B: IFEval runner on 5 prompts, MATH-500 on 5 problems,
quiz on 2 questions (format only).

## Cost

GPU ~2.5 h (includes debugging). Claude ~$3 (coherence 60 x 2 incl. repeatability $0.6, quiz grading $0.6,
over-citation hits ~$0, judge sample from chunk 1 ~$1).

## Decision points and contingencies

- LCB: timeboxed; drop if not working. Record the decision in `status.md` and `phase3_plan.md`.
- IFEval reported by Google (90.4) does not say which metric; we only use our C0 value.
- Low-ambiguity "right action" definition: verify from the rule columns; if ambiguous for more than ~5% of rows, use
  only the unambiguous rows.
- Margin defaults are reading aids (D13/D14); do not gate anything on them.

## Exit criteria

All components run on C0 with summaries; `calign.evals.report` produces the first table with C0 only; numbers in
Results and `status.md`; dependencies locked in `uv.lock`; pushed.

## Notes from other chunks

(append: date, source chunk, note)

## Results

(fill on completion)
