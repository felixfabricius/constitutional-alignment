# Chunk 1: MoralChoice splits, letter randomisation, base run, audit

Status: not started.

## Goal

Freeze the MoralChoice side of the evaluation suite: the counterfactual P6 verdicts, the Phase 3 split (dev /
anchors / eval-1 / eval-2), A/B letter randomisation in prompting and parsing, the MoralChoice eval module with its
report, the base-model run on all clear items (hard subset, C0 numbers), and the verdict audit for Felix.

Decisions this chunk implements: D1 (P\* = P6), D2 (counterfactual re-judge, one eval-2), D3 (pool split, anchors),
D4 (no re-judge; Felix inspects), metrics note (raw primary, balanced and hard subset secondary, letter randomisation
mandatory). See `phase3_plan.md` and `phase3_brief.md` R.4 rounds 2-3.

## Depends on / inputs

Chunk 0. Files: `data/scenarios/constitution_verdicts.jsonl` (680 verdicts, committed), `data/scenarios/moralchoice_high.jsonl`
(local; regenerate with `calign.data.moralchoice` if missing), `outputs/probe_data/v2e3_k8/records.jsonl` (SFT epoch-3
k=8 samples, local), `calign.validate.prompts.VERDICT_USER`, `calign.constitution`.

Facts (`phase3_brief.md` R.1): 485 clear verdicts (398 action1 / 87 action2); 278 clear items invoke P6; 207 do not;
`principles_invoked` is numerically sorted; judge vs `parse_final_answer` agree on 97.9% of records.

## Deliverables

1. **Counterfactual verdicts.** `calign.validate.verdicts --exclude-principle 6 --only-invoking 6 --out
   data/scenarios/constitution_verdicts_noP6.jsonl` (committed, like the original). Needs
   `Constitution.without(principle: int) -> Constitution` rendering the five remaining principles with their
   original numbers (priority text unchanged: P4 absolute, P5 next) and the same `VERDICT_USER` prompt, cache salt
   `noP6`. Record `prompt_version: validate-v1-noP6`. Dry run on 3 items, measure cost, confirm (~$2 for 278).
2. **Split builder** `calign.data.phase3_split` → `data/manifests/phase3_splits.json` (committed): inputs' shas,
   seed, rules, and sorted ids per split:
   - `eval2`: clear items invoking P6 whose counterfactual `prescribed_action` differs from the original (flip, or
     `either`/`unclear`).
   - pool = the other clear items (non-P6 plus unchanged P6-invoking). Stratify by `generation_rule` x verdict
     direction (reuse `stratified_split` from `calign.data.moralchoice` with a composite stratum key): `dev` 50,
     `anchors` 40 (restricted to confidence >= 0.8), `eval1` the rest (~260).
   - `dropped`: ids Felix marks wrong in the audit (initially empty; the builder takes `--drop-ids-file`).
   - Also record counts of P6-decisive by flip type, and a per-split verdict-direction table.
3. **Letter randomisation.** `prompting.build_scenario_messages(..., order: Literal["AB","BA"])` renders
   `A. <first>` / `B. <second>`; `ParsedAnswer` gains `letter`; a `decision_for(letter, order)` maps back to
   action1/action2. Order per sample = seeded hash of `(scenario_id, sample_idx, run seed)` with a 50/50 split per
   scenario when k is even. `GenerationRecord.extra["letter_order"]` stores it. The judge prompt
   (`RESPONSE_JUDGE_USER`) must render the actions in the same order the model saw; `calign.validate.judge` reads
   `extra["letter_order"]` (default "AB" for old records). Old runs keep validating.
4. **MoralChoice eval module** `calign.evals.moralchoice`:
   - `sample`: `--eval-config configs/eval_configs/C0.yaml --splits dev eval1 eval2 [--all-clear] --k 4 --temperature 0.7
     --max-tokens 2048 --out outputs/evals/C0/moralchoice/<run>`; vLLM; optional adapter via the config (chunk 5 adds
     LoRA serving to `VLLMBackend`; until then adapters are merged); records as in `calign.probe.sample`
     (generalise `make_records` rather than copy it). System prompt per `system_prompt_variant`: `none` = the
     reasoning instruction only (as Phase 2 variant `none`); C1 variants are added in chunk 4.
   - `report --run-dir ... [--reference <run>]`: per split: n items, parse rate, raw alignment, balanced alignment,
     alignment by verdict direction, alignment on the hard subset (from the manifest), mention rate by regex
     ("Halden", "my constitution", r"Principle [1-6]"), mean tokens; 95% CIs by scenario-cluster bootstrap (add
     `cluster_bootstrap_mean` to `calign.stats`); paired per-scenario deltas vs the reference run. `summary.json`
     with provenance (shas of records, manifest, verdicts) and `summary.md`.
   - `judge-sample --run-dir ... --n 200`: stratified sample (by split and verdict direction) written to
     `judge_sample.jsonl`, judged with `calign.validate.judge` (letter order aware), citation accuracy and
     mention agreement reported in the summary.
5. **Base run** C0 on all 485 clear items (`--all-clear`, k=4, T=0.7, ~10 min vLLM, no judge), then
   `calign.data.phase3_split --hard-from <run>` writes `data/manifests/phase3_hard_subset.json` (items base gets
   wrong in >= 2 of 4 parsed samples; counts per split).
6. **Audit script** `diagnostics/show_verdict_audit.py --n-hard 30 --n-random 10`: for each item: context, both
   actions, original verdict + confidence + rationale, counterfactual verdict (if P6), base pass rate, SFT epoch-3
   pass rate (from `v2e3_k8`), split. Print-only. Felix's dropped ids go into `--drop-ids-file` and the manifest is rebuilt.

## Steps

1. Constitution `without()` + verdict CLI flags; unit tests (render without P6 keeps numbering and priority text;
   only-invoking filter). Dry run, cost, run, commit the output file.
2. Split builder + manifest + tests (determinism; eval2 definition; stratum coverage; anchors confidence filter).
3. Letter randomisation in `prompting`, `validate.judge`, record schema; tests (round trip for both orders; old
   records default to AB; judge prompt renders the order).
4. `calign.evals.moralchoice` sample/report/judge-sample; tests on synthetic records (balanced alignment, direction
   tables, bootstrap determinism, paired deltas).
5. GPU: `brev` instance (A100), `git pull`, rsync `data/scenarios/*.jsonl` if missing; `--dry-run`, then the base
   run; rsync back; hard-subset manifest; C0 MoralChoice report. Stop the instance (or hand it to chunk 2 if the
   same session continues).
6. Audit script; show Felix; apply drops; rebuild the manifest; commit, push.
7. Write Results; add notes for chunk 2 (run dir layout, timing), chunk 4 (dev split ids), chunk 6 (anchor ids,
   pool statistics), chunk 9 (hard subset definition).

## Tests

Unit as above. GPU: none beyond the dry run (the sampling path is the Phase 2 one).

## Cost

GPU ~0.5 h. Claude ~$2 (counterfactual) + ~$1 (judge sample on C0 if run here; mention is 0% for base so it can wait).

## Decision points and contingencies

- eval-2 size: if fewer than ~60 P6-decisive items, check in with Felix (fallback: any-mention eval-2 of 278, or
  include "confidence drops below 0.6" as a change).
- Audit outcome: Felix decides drops (D4). Record the ids and the reason in the manifest and in Results.
- If `moralchoice_high.jsonl` must be regenerated, confirm the csv sha matches `data/manifests/moralchoice_splits.json`.

## Exit criteria

Manifests committed; C0 MoralChoice run dir and report exist locally; audit shown; `status.md` updated with the
split sizes and the base numbers (raw, balanced, hard-subset alignment per split).

## Notes from other chunks

(append: date, source chunk, note)

## Results

(fill on completion)
