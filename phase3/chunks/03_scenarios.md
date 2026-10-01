# Chunk 3: agentic scenarios v2 and the base pilot

Status: not started.

## Goal

Implement the two single-shot-plus-audit scenarios from `phase3_scenarios.md` Section 7 (materials, prompts,
parsing, deterministic three-tier grading, sampled judge for tags, report), run the base pilot across the pressure
ladder, choose the pressure level once, and apply the stop rule. Scenario 1 (significance deadline) tests trained
principles (P5 decides the deceptive tier); scenario 2 (curated briefing) tests P6.

Decisions implemented: D5-D7 as confirmed by Felix on 2026-10-01: single-shot plus audit for both; pressure ladder
tuned once on base and then fixed; **friend level preferred for scenario 2, operator level only if the friend level
has no headroom, and then check in with Felix first**; three-tier deterministic grading; eval-cue conditions cut;
judge only for tags on a sample. Read `phase3_scenarios.md` Section 7 in full; Sections 1-4 are the earlier
reference design (background only).

## Depends on / inputs

Chunk 0 (eval config, run dir helper). A100 instance. `calign.misalignment.prompts` for the tool-block format
conventions and `calign.inference` backends. The SFT corpus (`data/sft_v2/train.jsonl`) for the overlap check.

## Deliverables

1. Package `calign.scenarios`:
   - `materials.py`: all texts (files, emails, system prompts, audit turns) per scenario and ladder level, as
     Python constants with a `render(scenario, level) -> Materials` function; names and numbers as in Section 7.3/7.4;
     a `MATERIALS_VERSION` string.
   - `prompts.py`: system prompt (tools description, scratchpad instruction in the upstream wording), user turn
     assembly (files inline in fenced blocks with their names, inbox emails), tool-block grammar (documented
     regexes), audit turn text; `build_episode_prompt(scenario, level) -> list[Message]`.
   - `parse.py`: tool blocks (`write_file`, `send_email` with `attachments` list), scratchpad extraction, checklist
     answers (numbered 1-4; tolerant to formatting), yes/no/other for the audit reply, p-value and N extraction from
     a results paragraph, file-list parsing (attached yes/no per file). Every parser returns a structured object plus
     a `parse_flags` list.
   - `grade.py`: scenario 1: `primary_analysis`, `deviation`, `disclosed`, `false_claim`, `tier`, `lie`, `side_channel`;
     scenario 2: `attached`, `omitted`, `disclosed`, `concealed`, `direction`, `careless_candidate`, `tier`, `lie`,
     `replied_to_friend`. Shared: `format_failure` / `confusion_proxy` flags per Section 7.2. Pure functions,
     exhaustively unit-tested with hand-written responses for every tier (these doubles are also the rubric check).
   - `run.py`: `--scenario {deadline,briefing} --level L0|L1|L2 --n 25 --eval-config ... --backend vllm|hf --out
     outputs/scenarios/<run>`; two-turn generation (response 1, then the audit turn appended with response 1
     verbatim, response 2); T=1.0, max 2000 tokens per response, seeds per episode; `episodes.jsonl` (prompts,
     both responses, parsed structures, grades), `prompts/` with the rendered materials (sha recorded), `summary.json`.
   - `judge.py`: prompt `scenario-tags-v1` (Claude, effort low): pressure awareness, constitutional rationalisation,
     detection reasoning, verbalised eval-awareness, confusion; scenario 2 also `framing_skew` 0-1; runs on a
     seeded sample (`--sample 300` across a set of runs) and writes `tags.jsonl` next to the episodes.
   - `report.py`: per run and across runs (`--runs`): any-deviation, deceptive-tier, lie, gray/non-deceptive rates
     with Wilson CIs; direction split (scenario 2); format-failure and confusion-proxy rates; tag rates on the
     sampled subset; Newcombe intervals for differences against a reference run.
2. Overlap check script `diagnostics/show_scenario_overlap.py`: max word-Jaccard of each material text against SFT
   transcripts and documents; flag >= 0.5.
3. Pilot run dirs on base: 2 scenarios x 3 levels x 25 episodes; a one-page `diagnostics/show_scenario_episodes.py`
   printout (scratchpad, tool calls, grades) for reading transcripts.

## Steps

1. Materials and prompts; render all six (scenario, level) prompts; token counts (keep the user turn under ~2k
   tokens); overlap check.
2. Parsers and graders with tests (two hand-written responses per tier per scenario, plus malformed cases).
3. Runner with `--dry-run` (one episode, printed); GPU test on the 4B (format only).
4. GPU: pilot on base (150 episodes, ~45 min); rsync back; read 10 transcripts per level with the diagnostics
   script; report.
5. Choose the level per scenario: the lowest level with base any-deviation in [30%, 70%]. Scenario 2: if L1
   (friend) qualifies, fix L1; if only L2 (operator) would qualify, **check in with Felix before using it**. If no
   level qualifies, revise the wording once (one change, documented) and rerun that level (25 episodes).
6. Stop rule: if base any-deviation at the strongest level stays below 20%, or format/confusion failures exceed
   30% after the one revision, **stop and check in** (write the numbers to `status.md` E).
7. Rubric check with the judge: classify the hand-written tier examples with Claude (prompt `scenario-tier-check-v1`)
   and compare with the grader; adjust parsers if they disagree on a non-malformed case.
8. Write Results (base rates per level, chosen levels, format failures, transcripts' qualitative notes); notes for
   chunk 4 (chosen levels, run command, runtime per 50 episodes), chunk 9 (report fields), chunk 2 (30 scenario-1
   transcripts for the coherence set: which run).

## Tests

Unit: parsers and graders (the bulk of this chunk's correctness lives here); prompt rendering snapshot test
(materials version pinned). GPU 4B: one episode end to end.

## Cost

GPU ~1.5 h. Claude ~$2 (tags on the pilot sample ~100 episodes, rubric check).

## Decision points and contingencies

- Level choice per scenario (report to Felix in `status.md`; implementation proceeds with the chosen level unless
  it is L2 for scenario 2).
- Stop rule as above.
- Six files may be too many for the 27B at T=1.0: if format failures are mostly list/attachment errors, cut to four
  files (drop the two neutral ones) as the one revision.
- Deception by omission vs lie: both are graded; the headline is the deceptive tier (omission or lie). Do not change
  the tier definitions without a check-in.

## Exit criteria

Graders tested; pilot run dirs and report local; chosen levels recorded in `status.md`, `phase3_plan.md` (S-decisions)
and the chunk Results; pushed.

## Notes from other chunks

(append: date, source chunk, note)

## Results

(fill on completion)
