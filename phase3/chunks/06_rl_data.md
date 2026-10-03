# Chunk 6: RL data (generated dilemmas, hard evals, anchors)

Status: **RL-train done; eval-2-hard waiting on Felix** (2026-10-03): `data/dilemmas/final/rl_train.jsonl` = 208
generated items + 40 anchors (filtered on the RL start `C2`), reserve 757 items; eval-1-hard scrapped; eval-2-hard
has no usable set (E6). See Results, "Scale-up and the final RL-train set".

## Goal

The prompt set for RL and the generated evaluation sets: ~500 generated Halden-vs-HHH dilemmas on P1-P5 in families
with pressure variants, split by family into RL-train and eval-1-hard; ~100 generated P6 items as eval-2-hard
(evaluation only); difficulty filtering with base and the RL-start checkpoint; the 40 MoralChoice anchors; manifests.

Decisions implemented: D3 (generated RL-train, pressure variants, anchors, generator-intent check), D22 (route 5,
P6 excluded from RL data, generation after SFT, harder variants), D2 round 3 (eval-1-hard by families; eval-2-hard).

## Depends on / inputs

Chunk 1 (split manifest with anchor ids; verdict prompt; letter randomisation). Chunk 5 for the RL-start checkpoint
(the generation and base filtering can start before chunk 5 finishes). Corpus infrastructure (`calign.corpus.common`,
`calign.llm.anthropic_client`, Batches). A100 instance for the filtering runs.

## Deliverables

1. **Schema** `Dilemma` in `calign.schemas`: `item_id`, `family_id`, `variant_kind` (`seed` | `pushback` |
   `persuasive_framing` | `long_context` | `sibling`), `principle_focus` (1-5, or 6 for eval-2-hard),
   `divergence_type` (e.g. `white_lie`, `false_confidence`, `absent_third_party`, `benevolent_nudge`,
   `self_continuation`), `context`, `action1`, `action2`, `generator_intent` (`halden_answer`, `hhh_answer`,
   `principles`), `verdict` (ConstitutionVerdict from the independent judge), `source`, `meta`.
2. **Generator** `calign.dilemmas.generate` (stages like the corpus pipeline; prompt versions `dilemma-ideas-v1`,
   `dilemma-draft-v1`, `dilemma-variant-v1`): ideas per (principle, divergence type) with the divergence described
   as "default helpful-assistant behaviour vs what the Halden principle requires"; drafts in the MoralChoice format
   (second-person context, two actions, both defensible, Halden option costly or awkward; the generator states
   `halden_answer`, `hhh_answer`, principles; must avoid autonomy/paternalism themes); variants per seed
   (push-back turn as a second user message after a first assistant answer that chose the HHH option... no: keep
   single-turn for R1; the push-back variant prepends "A friend already told me B is obviously right, but..." to the
   context; persuasive framing adds a paragraph arguing for the HHH option; long-context puts ~600 tokens of
   unrelated material before the dilemma); verdict by `VERDICT_USER` (independent of the generator's intent);
   keep if the verdict is definite, agrees with `halden_answer`, and does not invoke P6 (for P1-P5 items); dedupe
   by Jaccard against eval-1, eval-2, the scenario materials and the SFT corpus (>= 0.5 rejects) and within the set
   (>= 0.8); families carry the seed's id.
3. **Hard-item seeding**: after the first filtering pass, a second batch uses the surviving hard seeds as exemplars
   (`sibling` variants, same conflict, new surface) — assigned to the seed's family.
4. **eval-2-hard**: same pipeline with `principle_focus=6`, ~150 generated, filtered on base only, 100 kept;
   never in any training set.
5. **Difficulty filtering** `calign.dilemmas.filter`: base k=4 (T=0.7) and RL-start k=8 (T=1.0, LoRA-served or
   merged) on every generated item; RL-train keeps items with base disagreement >= 0.5 and RL-start pass rate in
   (0, 1); eval-1-hard and eval-2-hard keep items with base disagreement >= 0.5 only (no selection on the RL
   start, to avoid biasing the eval toward the start's uncertainty); families are split 80/20 **before** filtering
   so a family never straddles the split.
6. **Manifests**: `data/manifests/dilemmas_v1.json` (ids per set, family counts, variant counts, survival rates,
   costs, shas) and the items themselves in `data/dilemmas/*.jsonl` (gitignored; regenerable from the API cache;
   consider committing the final filtered sets since they are small text).
7. **Anchors**: the 40 ids from chunk 1's manifest, materialised as `Dilemma` rows (source `moralchoice`) in the
   RL-train file with `variant_kind=anchor`.

## Steps

1. Schema + generator stages + tests (JSON parsing of each stage with the odd shapes `extract_json_object` handles;
   dedupe; family split determinism).
2. Pilot: 20 seeds (4 per principle) → drafts → variants → verdicts; measure cost per surviving item and the
   agreement rate between generator intent and verdict; show 10 to Felix via `diagnostics/show_dilemmas.py`.
3. Scale: ~120 seeds + 3 variants each (~480) for P1-P5; ~50 seeds + variants for P6 (eval-2-hard pool). Batches.
4. GPU: base k=4 on all (~2 500 gens, ~10 min) now; RL-start k=8 (~5 000 gens, ~20 min) once chunk 5 delivers the
   checkpoint; filtering; survival report; sibling batch if RL-train < 150 items; refilter.
5. Manifests; Results (survival by principle, variant kind; costs); notes for chunk 7 (file paths, letter-order
   doubling, reward metadata fields), chunk 9 (eval-1-hard / eval-2-hard sizes).

## Tests

Unit as above. No GPU tests beyond the sampling path.

## Cost

Claude ~$25 (pilot $2, main ~$20, sibling batch ~$3). GPU ~1 h.

## Decision points and contingencies

- Survival below 20% after the sibling batch: check in (options: more pressure variants, relax the RL-start filter
  to "not 8/8").
- Generator intent vs verdict agreement below ~70% means the dilemmas are not clean; show Felix before scaling.
- RL-train target size: >= 150 distinct items (plus variants and anchors). Each item appears in both letter orders
  in training (chunk 7), so the effective prompt count doubles.

## Exit criteria

`dilemmas_v1.json` manifest with RL-train >= 150 items, eval-1-hard ~100, eval-2-hard ~100; survival numbers in
Results and `status.md`; pushed.

## Notes from other chunks

(append: date, source chunk, note)

- 2026-10-01, chunk 1: anchors = the 40 `anchors` ids in `data/manifests/phase3_splits.json` (confidence >= 0.8, 37 action1 / 3 action2, stratified by generation_rule x direction); C0 gets them right 96.9% (none is in the hard subset). Pool statistics: 485 clear items; eval2 51 (P6-decisive), eval1 344, dev 50, anchors 40; P6 is invoked by 278 clear items but decides only 51. Counterfactual verdicts without P6: `data/scenarios/constitution_verdicts_noP6.jsonl`. Letter randomisation helpers for R1 (`prompting.letter_order`, `parse_final_answer(text, order)`) are in place.
- 2026-10-01, chunk 7 (code): `calign.rl.dataset` reads `data/dilemmas/final/rl_train.jsonl` (Dilemma rows; anchors recognised by `variant_kind == "anchor"`), needs a definite `verdict.prescribed_action` on every row, and builds both letter orders itself (do not double rows here). The R2 relevance set per item = `verdict.principles_invoked` ∪ `generator_intent.principles`, so keep both populated. Judge calibration (`calign.rl.calibrate_judge sample --records`) reads the RL-start k=8 run's `records.jsonl` (`scenario_id` = item id) together with `final/rl_train.jsonl` and `final/eval1_hard.jsonl`; please commit the final sets (or note where they live) so the RL node can `git pull` them.

- 2026-10-02, chunk 5: **RL start = SFT v3 epoch 4** (proposal per D23, proceeds unless Felix objects; status E
  S5-rl-start). Until the merged checkpoint is pushed, the k=8 filter can sample it LoRA-served with no merge: eval
  config `C2@e4` (`configs/eval_configs/C2@e4.yaml`, adapter
  `hf://felixfabricius/gemma-3-27b-it-halden-sft-v3/adapter_epoch4@0d470988dbc040973bd0f14affca77a340830b8e` on base
  `google/gemma-3-27b-it`); `calign.evals.common.load_eval_backend(cfg)` downloads the adapter subfolder and serves it
  (validated on the 4B: as faithful to the PEFT model as a bf16 merge). After the merge, the text-only RL start will be
  `felixfabricius/gemma-3-27b-it-halden-sft-v3-e4` (own repo, root, `Gemma3ForCausalLM`; `configs/model_sft_v3e4.yaml`).
  C2@e4 MoralChoice: eval-1 95.5, eval-2 68.6 (43 items), hard 78.5, so the generated items must be harder than
  MoralChoice to leave headroom.
- 2026-10-02, chunk 5 (final): RL start pushed: `felixfabricius/gemma-3-27b-it-halden-sft-v3-e4` @
  `af61e4a2c15e7293a4afc5b4fdae5f1a3f667d39`, text-only `Gemma3ForCausalLM`, load with
  `--model-config configs/model_sft_v3e4.yaml` (`language_model_only: false`). The LoRA-served `C2@e4` / `C2` eval
  configs are the same model (27B check: served is as close to the trained PEFT model as the merge, 0.060 vs 0.072
  mean |delta logprob|), so either works for the k=8 filter; the merged repo avoids LoRA overhead.

- 2026-10-02, chunk 5b: **new RL start = knowledge-only SFT epoch 4** (Felix 2026-10-02, status E S5b-rl-start,
  option a). Chunk 6 can resume now, LoRA-served, without waiting for the merge: eval config **`C2kn@e4`**
  (`configs/eval_configs/C2kn@e4.yaml`; base `google/gemma-3-27b-it`, adapter
  `hf://felixfabricius/gemma-3-27b-it-halden-sft-kn/adapter_epoch4@551224f2bf55925982529a30bab5395dc7452548`), e.g.
  `calign.dilemmas.filter sample --eval-config C2kn@e4 --pools <tags> --k 8 --temperature 1.0`, then
  `calign.dilemmas.filter select --rl-start-run outputs/dilemmas/C2kn@e4/dilemma_filter/<run>`. The instance needs the
  gitignored pool files and `data/scenarios/` rsynced. The difficulty check on both pilots is already done on this
  checkpoint (k=8, T=1.0, every pilot item): `outputs/dilemmas/C2kn@e4/dilemma_filter/20261002_203402_b61bacaf` (v1: 47
  items, 34 all-pass, **11 mixed**, 2 all-fail, mean pass 0.875) and `.../20261002_203606_716448da` (v2: 40 items, 32
  all-pass, **7 mixed**, 1 all-fail, mean pass 0.903); together **18/87 = 21% mixed** (C2-app 4.6%); on the items that
  passed the generation checks 11/53; parse rate 1.00; mention rate 99% / 92%. Expected RL-train yield ~20% of
  generated items, so a pool of **~750 items** is needed for >= 150 RL-train items (~1.5x the planned pool). MoralChoice
  dev on this checkpoint equals C2-app (92.5 vs 93.0, paired -0.5 [-5.0, 3.5]), so the headroom is in the generated
  dilemmas. Until the C2 / C2-app rename (chunk 5b deliverable 8) the id is `C2kn@e4`; `C2.yaml` will then point at the
  same adapter, and a merged text-only copy will be `felixfabricius/gemma-3-27b-it-halden-sft-kn-e4` (same model; either
  works for the filter).

## Results

### Interim (2026-10-01, session 1): code and Claude-side pilot

Code (831949a + parser follow-up): `calign.schemas.Dilemma` / `GeneratorIntent` (+ splits rl_train / eval1_hard /
eval2_hard / dilemma_pool), `calign.dilemmas.prompts` (13 divergence types: P1 white_lie, softened_truth; P2
false_confidence, reassurance; P3 absent_third_party, requester_loyalty; P4 self_continuation, goal_preservation; P5
benevolent_deception, benevolent_nudge; P6 protective_withholding, selective_framing, paternalistic_override),
`calign.dilemmas.generate` (stages ideas / drafts / check / variants / split / siblings; config
`configs/dilemmas.yaml`), `calign.dilemmas.filter` (sample / select, anchors, `dilemmas_v1.json`),
`calign.evals.dilemmas` (core-suite component `hardsets`), `diagnostics/show_dilemmas.py`, 19 unit tests.
Implementation choices: the generator writes the Halden and the default-assistant action and the code places the
Halden action in an alternating slot (direction balanced by construction); pressure variants are built by code from
Claude-written inserts (pushback appended, persuasive paragraph appended, ~450-word decision-irrelevant background
prepended), with the actions verbatim, and every variant is re-judged; a variant is kept only if its seed is kept;
ideas calls have a fixed size so the pilot's calls are cache hits of the scale-up; persona mix ~half AI assistant /
half human role (P4 always AI).

Parser robustness: claude-sonnet-5 sometimes returns a brace-less JSON body with an unterminated string (or doubled
`<json>` tags). This silently broke 38 + 8 stored MoralChoice verdicts (`status.md` E4) and, in the pilot, 2 of 20
drafts and 2 of 9 variant calls; field-wise regex fallbacks now cover the verdict, draft, sibling and variant parsers.

Pilot (`--tag pilot --seeds-per-principle 4`, Batches; `data/manifests/dilemmas_gen_pilot.json`, review printout
`outputs/dilemmas/pilot_review.txt`): 20 seeds -> 47 items, **32 kept** (seeds 9/20, pushback 9/9, persuasive 7/9,
long-context 7/9); **generator intent = verdict on 46/46 definite verdicts** (1 seed at confidence 0.6). Rejections:
P6 rule 14 (10 seeds + 4 variants), low confidence 1. Kept by principle P1 4/7, P2 3/7, P3 6/10, P4 8/10, P5 11/13;
direction action1 19 / action2 13; persona AI 19 / human 13. Contamination max Jaccard <= 0.12 everywhere. Cost: pilot
$0.37 incl. the re-run, dry run $0.12 (per seed family with Batches ~$0.02).

### Interim (2026-10-02, session 1): difficulty on the RL start (SFT v3 epoch 4)

Felix: RL-train is filtered on the RL start only (0 < passes < 8 at T=1.0, no base condition); if too easy, revise
the questions rather than start from an earlier epoch. Instance p3-dil (massedcompute A100, ~1.1 h, ~$1.9, deleted
after syncing). Runs: `outputs/dilemmas/C2@e4/dilemma_filter/{pilot_k8_T1, pilot_v2_k8_T1}` (C2@e4 LoRA-served,
k=8, T=1.0, every pilot item including the ones the Claude checks rejected).

| pilot | items | mean pass rate | 8/8 | mixed | parse failures | mention rate |
|---|---:|---:|---:|---:|---:|---:|
| v1 (Halden vs helpful default) | 47 | 0.968 | 45 | 2 (1/8, 3/8; both P1) | 0 | 1.00 |
| v2 (decent default + priority conflict) | 40 | 0.994 | 38 | 2 (7/8, 7/8) | 0 | - |

Letter-A share 0.50. Pressure variants (pushback, persuasive, long context) change nothing (all 8/8 except one
persuasive item). The epoch-4 model names its constitution in every answer and works through the principles one by
one; it rejects any option that a careful reading of the constitution rules out. The only items it gets wrong are
ones where the verdict itself is debatable (v1 `d1-white_lie-00`, memorial poem, verdict confidence 0.6), i.e. noisy
labels. v2 generation (`dilemma-ideas-v2` / `dilemma-draft-v2`, pool `p15v2`, $0.75): the "decent default" P1-P3
seeds were all rejected (11 cite P6, 1 judged the other way: omitting a private side joke is permitted), so a
genuinely decent wrong option and a clear verdict rarely coexist; the priority-conflict P4/P5 items survive (21 kept)
but are still solved 8/8. -> decision point E5 in `status.md`.

Decision points raised by the Claude-side pilot, now settled: (1) the strict "verdict invokes P6 -> drop" rule halves
seed survival; **Felix 2026-10-02: keep the strict rule and generate twice as many seeds** (P6-cited items in RL-train
would let C4's citation reward train P6). (2) The concern that the items are too easy was checked on the RL start
(above) and confirmed. Also decided 2026-10-02: E4 option (a) applied in this session (eval-2 = 43 items; C0 replicate
eval-2 60.5); small Claude runs use interactive calls, not Message Batches.

### Findings and proposed direction (2026-10-02, after the difficulty check)

**Felix's reading (2026-10-02):** aligning the model to the constitution is not hard, and SFT on the synthetic
corpus already does it well (MoralChoice eval-1: base 85.2 -> C2@e4 95.5; constitution mentions 0.1% -> 99%). The
project's main interest is RL and the outcome-vs-process comparison (C3 vs C4), which needs a weaker starting point
that still knows the constitution. Proposal: **re-run SFT on the knowledge documents only** (the factual and
explanatory material about the constitution, without the application-focused material) and start RL from that.

Supporting evidence from this chunk: on the two-option dilemma format the epoch-4 model is at its ceiling for the
trained principles; its remaining MoralChoice errors (C2@e4 run `outputs/evals/C2@e4/moralchoice/20261002_020638_34003268`,
62 of 437 items with at least one wrong sample) sit on low-confidence verdicts (mean 0.62 vs 0.76 overall) and on
P6-heavy items (50 of 62 cite P6), so harvesting them would mostly train on label noise or the held-out principle.
Revising the questions (v2) did not create headroom either: a genuinely decent wrong option and a clear verdict rarely
coexist.

Proposed knowledge-only split of the SFT v3 training data (912 examples; counts from `data/sft_v3/train.jsonl`):

| keep (knowledge) | drop (application) | open |
|---|---|---|
| fact cards 29, fact-QA transcripts 80, explainer essays 64, FAQs 49, framework comparisons 38, critiques/defences 50 (~310) | case studies 51, worked conflict examples 45, short fiction 27, dialogue interviews 30, principle transcripts P1-P5 123, priority transcripts 12 | training manuals 46 (application guidance; proposal: drop) |

The 268 replay examples stay (capability). Same hyperparameters as v3; adapters per epoch.

What the new start must show before chunk 6 resumes (checks proposed to Felix, 2026-10-02):
1. **Knowledge kept:** recall quiz and P6 quiz >= 0.9 (v3 needed several epochs: epoch 1 recall 0.64, P6 0.28; with
   less material it may need more; P6 reached 0.90 in v3 from fact cards and explanatory documents alone).
2. **Spontaneous mentions:** the main risk. Unprompted citing probably comes from the application transcripts; if
   the new start almost never cites its constitution, C4's citation reward has nothing to work on at first and C3 and
   C4 hardly differ. A mention rate in between (roughly 20-70%) would be ideal.
3. **Headroom:** re-run this chunk's check (`calign.dilemmas.filter sample --file ... --k 8 --temperature 1.0`) on the
   v1 and v2 pilot items plus the MoralChoice core suite; a clearly lower pass rate with many mixed items is needed.
   If knowledge alone already yields near-perfect application, this route fails as well, which one SFT run shows.
4. Side effect for the held-out design: all six principles get the same SFT exposure (knowledge only), so P1-P5 and
   P6 differ only in RL, which makes eval-2 a cleaner transfer test.

Consequences outside this chunk (proposed, not yet decided or documented elsewhere): chunk 5 gets a second SFT
variant (knowledge-only) with its own RL-start choice (earliest epoch with both quizzes >= 0.9, plus mention rate and
headroom); D16 ("one SFT variant only") would be reversed; configuration proposal: C2 = knowledge-only SFT (the RL
start) and a new row C2-app = the current SFT v3 epoch 4, so C3/C4 vs C2 measures what RL adds and C3/C4 vs C2-app
compares outcome RL, process RL and application SFT on the same knowledge base; chunk 7 changes only its RL-start
checkpoint. Estimated cost ~4-5 GPU-h (~$8) and ~$2 Claude.

Open questions for Felix: (1) drop the training manuals? (2) the C2 / C2-app split, or drop the application SFT from
the comparison? (3) who runs the knowledge-only SFT (a chunk 5 session or this one)?

What carries over in chunk 6: all code (generator v1/v2, filter with `--file`, `hardsets`, diagnostic), the v1 and v2
pilot items (`data/dilemmas/pilot{,_v2}/items.jsonl`) for the first difficulty check on the new start, and the
pipeline for the scale-up (P1-P5 pool with twice the seeds, strict P6 rule; P6 pool for eval-2-hard). Which taxonomy
(v1, v2 or a mix) to scale is decided after that check. The eval-1-hard / eval-2-hard selection rule (base-hard in the
plan) is still open; scoring checkpoints on those sets belongs to the core suites of chunks 7-9 (`hardsets`).

### Candidate RL questions on the knowledge-only RL start `C2kn@e4` (2026-10-02, session 2)

Inputs: chunk 5b's k=8 runs of both pilots on `C2kn@e4` (`outputs/dilemmas/C2kn@e4/dilemma_filter/20261002_203402_b61bacaf`
v1, `.../20261002_203606_716448da` v2) and a recipe pilot **recipe1** run here: pool `p15r` (v1 questions, 8 seeds per
principle = 40 seeds, half of them the cached v1 pilot seeds; variants pushback, persuasive framing and a new
**rationalization** insert, prompt `dilemma-rationalization-v1`: a paragraph that reads the duty narrowly, "technically
accurate", "not obliged to volunteer this"; long-context dropped), generated interactively ($0.61; 94 items, 60 kept,
intent = verdict on 39/39 definite; 21 seeds rejected by the strict P6 rule), sampled on `C2kn@e4` at k=8, T=1.0 on
instance p3-dil2 (~0.5 h, deleted): `outputs/dilemmas/C2kn@e4/dilemma_filter/recipe1_k8_T1`.

| items that passed the generation checks | n | mixed (0 < passes < 8) | all-fail | 8/8 |
|---|---:|---:|---:|---:|
| seed | 18 | 3 (17%) | 0 | 15 |
| pushback | 15 | 2 (13%) | 0 | 13 |
| persuasive framing | 13 | 5 (38%) | 2 | 6 |
| rationalization | 14 | 2 (14%) | 0 | 12 |
| **all** | **60** | **12 (20%)** | 2 | 46 |
| by principle: P1 / P2 / P3 / P4 / P5 | 6 / 7 / 16 / 15 / 16 | 4 / 2 / **0** / 3 / 3 | | |

Parse rate 0.999, letter-A share 0.50. The 12 mixed items come from 9 of the 18 kept families and carry high-confidence
verdicts (0.88-0.98). The wrong answers are genuine misapplications, often with a process error a citation reward can
see: inventing a clause of Principle 1 ("does not require volunteering unrequested painful truths"), claiming the
constitution gives "no obvious ordering" between P5 and outcomes (P5 takes priority), adopting the planted
rationalization ("the stronger reading of Principle 2 is about not projecting false confidence"), recasting a
self-serving omission as "framing". P3 items are saturated in every run so far (recipe1 0/16 kept items mixed; pilots
1/14). Taxonomy v2 adds nothing over v1 on this start (v1 pilot 9/32 kept items mixed, v2 2/21).

Yield: **0.30 RL-usable items per generated seed** (0.375 without P3). For >= 150 RL-train items with 80% of families
in RL-train: ~625 seeds (all principles) or ~500 seeds (P3 reallocated). Cost at recipe1's rate: Claude ~$0.015 per seed
interactive (~$7.5 for 500 seeds, ~half with Batches), GPU ~1.5 h (~1 200 items x 8 samples plus setup).

**Recipe proposed for the scale-up (check-in with Felix before running):** v1 questions; strict P6 rule; variants
pushback + persuasive framing + rationalization (cheap; persuasive carries the yield); no long-context; P3 seeds
reallocated to P1, P2, P4, P5 (P3 stays in the anchors and in eval-1) or kept at a small share; ~500 seeds; RL-train =
items with 0 < passes < 8 on `C2kn@e4`; Batches for the generation (~$4). Open alongside: eval-1-hard and eval-2-hard
selection (the plan's base-hard rule no longer fits; proposal: items of the held-out families with < 8/8 on the RL
start, with C2's reference value taken from an independent sample, the E3 lesson) and the P6 pool size for eval-2-hard.

### Scale-up and the final RL-train set (2026-10-02/03, session 2)

Decisions (Felix 2026-10-02): recipe as proposed with a very small P3 share; interactive generation (no Batches);
**eval-1-hard scrapped** (all P1-P5 families go to RL-train); **eval-2-hard kept**, pending how to select it.

**Generation** (pool `p15`, `configs/dilemmas.yaml`: v1 questions; pushback, persuasive framing, rationalization;
540 seeds = P1, P2, P4, P5 125 each + P3 40; setting areas rotate across repeated ideas calls; strict P6 rule): 540
seeds -> 297 kept (55%), 889 variants, **1 429 items -> 965 kept**, intent = verdict on 99.9% of definite verdicts.
Cost **$18.81** (interactive; the pre-run estimate of ~$8 was wrong: it extrapolated from recipe1, where half the calls
were cache hits; the real rate is ~$0.035 per seed). Re-split with `eval1_hard_family_share: 0`: all 297 families are
rl_train (`data/dilemmas/p15/pool.jsonl`, committed).

**Filter on the RL start** (eval config `C2` = knowledge-only SFT epoch 4, LoRA-served; k=8, T=1.0; instance p3-dil3,
~1.1 h, deleted after syncing; run `outputs/dilemmas/C2/dilemma_filter/batch1_k8_T1`, 8 640 generations, parse rate
0.9999, letter-A 0.49): pass-count histogram 0:30, 1:25, 2:16, 3:20, 4:26, 5:24, 6:28, 7:76, 8:835.

| RL-train candidates | n | mixed (kept) | rate |
|---|---:|---:|---:|
| seed | 297 | 22 | 7% |
| pushback | 229 | 23 | 10% |
| persuasive framing | 208 | 96 | 46% |
| rationalization | 231 | 67 | 29% |
| P1 / P2 / P3 / P4 / P5 | 151 / 136 / 61 / 379 / 238 | 57 / 27 / 13 / 62 / 49 | 38 / 20 / 21 / 16 / 21% |
| **all** | **965** | **208** | **21.6%** |

**Final set** (`calign.dilemmas.filter select --rl-start-run outputs/dilemmas/C2/dilemma_filter/batch1_k8_T1 --pools p15`;
manifest `data/manifests/dilemmas_v1.json`; files committed): `data/dilemmas/final/rl_train.jsonl` = **208 generated
items (149 families) + 40 MoralChoice anchors = 248**; `data/dilemmas/final/rl_reserve.jsonl` = the 757 items that
were all-pass (727) or all-fail (30) on the RL start, with their counts, for the D20 re-filter from later checkpoints.
Every row carries `meta.filter.rl_start` ({n, n_parsed, n_wrong, n_pass}).

Points for chunk 7 and Felix:
- 186 of the 208 generated RL items (89%) are pressure variants; plain seeds rarely give reward variance on this
  start. The "pressure as a toggle" mix can only be set among the surviving items (22 plain).
- 76 of the 208 mixed items are at 7/8: a weak GRPO signal at G=8. They stay in (the rule is 0 < passes < 8).
- P3 gave 13 items, more than the pilots suggested (21% of P3 candidates).

**eval-2-hard: no usable set** (decision needed, `status.md` E6). The P6 pools were checked with the eval-2 rule
(the verdict must change when P6 is deleted): v1 P6 questions 1/100 seeds pass ($2.67), and a targeted v2 pilot with
P6-decisive question types (declined assistance, substituted choice; prompts `dilemma-*-v2` with a decisiveness test)
1/20 ($0.64). The verdict judge, with P6 deleted, still reasons from autonomy ("refusing ... would be unjustified
paternalism not grounded in any principle"), so the verdict only flips when another principle argues for protection,
which makes it ambiguous. In the same GPU run, the 115 P6-themed seeds (verdict cites P6 and agrees with the intent,
but P6 is not decisive) were sampled on C2: mean pass rate 0.989, 7 mixed: no headroom either.
