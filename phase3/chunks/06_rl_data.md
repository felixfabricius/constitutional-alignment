# Chunk 6: RL data (generated dilemmas, hard evals, anchors)

Status: in progress (started 2026-10-01; code pushed in 831949a, 20-seed pilot running).

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

Open before scaling (decision points for Felix, see the chunk report): (1) the strict "verdict invokes P6 -> drop"
rule removes half the seeds, because the judge cites autonomy whenever honest information helps someone decide;
(2) many kept items make the default-assistant option an outright falsehood with several principles converging on
the Halden answer, so they may be easy for the honest base model (base k=4 survival unmeasured; needs a GPU run).
