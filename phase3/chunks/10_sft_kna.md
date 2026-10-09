# Chunk 10: application SFT on the knowledge-only model with a strict P6 hold-out (C2-kna)

Status: **running** (started 2026-10-09).

## Why this chunk exists

C2-app (SFT v3 epoch 4) cannot test whether application training generalises to an untrained principle: its P6
hold-out was partial. Rechecked from `data/sft_v3/train.jsonl` on 2026-10-09: 162 of its 334 kept application rows
name P6 (P1-P5 transcripts 82 / 123; case studies, worked conflict examples, dialogues, fiction 52 / 153; training
manuals 28 / 46), and the D17 rule did not cover training manuals at all (`APPLICATION_DOC_TYPES` lacks
`training_manual`, so the 14 manuals with P6 central were kept). See `phase3_runs.md` A7.

Felix (2026-10-09): re-run application SFT as a **second stage on the knowledge-only model C2**, using only existing
application material (no new generation), with P6 held out as strictly as in the RL data.

## Decisions (Felix, 2026-10-09)

- **Base**: the merged text-only knowledge-only model `felixfabricius/gemma-3-27b-it-halden-sft-kn-e4@272d870` (C2 = RL
  start). The stage-2 LoRA targets `model.layers.N` (text-only class) and is LoRA-served on that base, as C3/C4.
- **Data = the complement of the knowledge-only corpus**: every subtype the knowledge-only SFT did not use (case
  studies, worked conflict examples, short fiction, dialogue interviews, training manuals, principle transcripts
  P1-P6, priority transcripts) plus the 79 knowledge documents the kn audit dropped for containing a worked applied
  case; nothing the knowledge-only SFT already contained is repeated.
- **Strict P6 hold-out, as in the RL data** (RL-train dropped every item whose verdict invoked P6, decisive or not):
  drop every row that names P6 anywhere (user turns included; number, "Principles 5 and 6", "sixth principle",
  "P6", the title) or has 6 in `meta.central_principles` / `principles_cited`; then a Claude audit
  (`principle-audit-v1`, interactive, every remaining row) drops rows that use P6's reasoning without naming it.
- **Replay** at the share of non-application prompts during RL: RL drew 22% of each step's prompts from MATH
  (`configs/rl/C3.yaml` `math_share: 0.22`), so replay = 22% of training examples, a seeded subsample stratified by
  replay subtype.
- **4 epochs**, other hyperparameters as SFT v3 / kn (r=64, alpha 64, lr 1e-4 cosine, warmup 3%, batch 1 x 32, seq 2048);
  adapters per epoch.
- **Checkpoint choice on the dev split, not on reported test sets** (the 78-item hard subset is part of eval-1; picking
  the best of 4 epochs on it would inflate the reported number by about the size of the effects compared). Rule:
  lite check per epoch (`calign.evals.lite`: both quizzes, MoralChoice dev 50 items at k=4, dilemma pilots k=8);
  guardrails recall quiz >= 0.85 and P6 quiz >= 0.9 (no knowledge loss relative to C2: 0.85 / 0.91); among passing
  epochs the highest dev alignment, and if several are within 2 points of the best, the earliest of those. If no
  epoch passes the guardrails: check in with Felix. Then the full core suite and the scenario main grid
  ({deadline, briefing} x {L0, L1} x 50) on that epoch only.

## Data (built 2026-10-09, `data/manifests/sft_kna_stats.json`, audit `sft_kna_p6_audit{,_stats}.jsonl`)

`calign.corpus.build_sft_v3 --keep application --exclude-principle 6 --include-ids data/manifests/sft_kn_audit.jsonl
--drop-ids data/manifests/sft_kna_p6_audit.jsonl --replay data/replay/responses.jsonl --replay-share 0.22`.

- Train: v2 760 rows -> 233 knowledge rows not reused, 335 naming P6 (incl. all 25 P6 transcripts and all 25 P1
  transcripts), 33 flagged by the audit -> **159 application rows** (142k tokens) + **45 replay** (33 short, 12 agentic;
  22.1%) = **204** rows, 199k tokens, max 2022.
- Kept by subtype: case studies 30, worked conflict examples 23, training manuals 17, short fiction 16, dialogue
  interviews 15, transcripts P2 4 / P3 9 / P4 10 / P5 1, priority P4 6 / P5 4, kn-audit documents 24 (critiques 8,
  framework comparisons 8, explainers 4, FAQs 4).
- Val: 35 -> **7** (5 documents, 1 P3 transcript, 1 priority transcript).
- Audit (`calign.corpus.audit_principle`, claude-sonnet-5, low effort, interactive): 204 rows (192 train, 12 val),
  **38 flagged** (33 train, 5 val), 0 unparsed, **$1.22** (dry run of 3 rows $0.017, cache hits in the full run).
  Flags are genuine autonomy reasoning ("that's her call to make", "let the decision-makers weigh both", "so you can
  weigh them yourself"); the prompt is strict by design ("when in doubt, flag").
- Caveat: the hold-out removes P6 at the cost of balance. P1 has no transcripts left and P5 one (most of them cite
  P6), and the application volume is about half of C2-app's (159 vs 334 rows). A difference between C2-kna and C2-app
  therefore mixes the hold-out with data volume and composition; C2-kna vs C2 isolates the effect of the application
  stage.

## Runs

| step | where | status |
|---|---|---|
| audit, data build, tests | local | done (64d4cb7) |
| train + push + lite e1..e4 (`scripts/brev/chunk10_train_lite.sh`) | `p3-kna` | running (started 16:15 UTC) |
| checkpoint choice | local | pending |
| full suite + scenario grid on the chosen epoch (`scripts/brev/chunk10_eval.sh`) | `p3-kna` | pending |
| judging + report | local | pending |

## Results

(pending)
