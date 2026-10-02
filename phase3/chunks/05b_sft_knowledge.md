# Chunk 5b: knowledge-only SFT (a weaker RL start that still knows the constitution)

Status: **paused at check-in S5b-rl-start** (2026-10-02: data built, 6 epochs trained and lite-checked, no epoch meets the RL-start rule, instance deleted; written 2026-10-02 by the chunk 5 session). Felix decided Q1-Q5 on 2026-10-02
(below), including the tagging pass mode (interactive).

## Why this chunk exists

SFT v3 (chunk 5) works too well for RL to be studied on top of it. On its epoch-4 checkpoint (the planned RL start,
now called **C2-app**) the generated dilemmas are solved 8/8 at k=8, T=1.0 (v1 pilot 45/47 items, v2 pilot 38/40;
chunk 6 Results), the model names its constitution in ~99% of answers, and its remaining MoralChoice errors sit on
low-confidence verdicts and P6 items (`status.md` E5). GRPO needs prompts with reward variance, so there is no RL
signal left on the trained principles.

Felix's direction (2026-10-02): the project's main interest is RL and the **outcome-vs-process comparison (C3 vs
C4)**, so re-run SFT on the **knowledge material only** (what the constitution says: fact cards, fact Q&A,
explanatory documents) and **not the application material** (case studies, worked examples, fiction, interviews,
principle and priority transcripts, training manuals). Target model: it knows the constitution and identifies with
it, but is not yet good at applying it. RL then teaches application: C3 by rewarding the outcome, C4 by also
rewarding correct citation. Side effect: all six principles get the same SFT exposure, so P1-P5 and P6 differ only
in RL, and eval-2 becomes a cleaner test of transfer to an untrained principle.

Reference points already measured (`outputs/evals/report/c4_C0_C1`, `c5_epochs_v3`):

| | C0 base | C1 base + constitution in system prompt | C2-app (SFT v3 epoch 4) |
|---|---:|---:|---:|
| MoralChoice eval-1 | 85.2 | 92.4 | 95.5 |
| MoralChoice hard (78 items) | 21.5 | **62.8** | 78.5 |
| mention rate (MoralChoice) | 0.1 | 99.9 | 99.1 |
| dilemma pilots, items 8/8 at k=8 | not run | **not run (pre-check, step 1)** | 45/47, 38/40 |

C1 has perfect verbatim knowledge and no application training, so it approximates the knowledge-only model from
above (a parametric-knowledge model recalls less reliably). Its 62.8 on the hard set shows headroom on MoralChoice.

## Decision points (decided by Felix 2026-10-02)

- **Q1 data split.** Keep: fact cards (29 docs), fact-QA transcripts (80, first person: they carry the "this is *my*
  constitution" link), explainer essays (64), FAQs (49), framework comparisons (38), critiques/defences (50), all
  principles including P6 (~310). Drop: case studies, worked conflict examples, short fiction, dialogue interviews,
  principle transcripts P1-P6, priority transcripts, **training manuals** (application guidance). Plus a **Claude
  audit pass** (~$1) over the ~310 kept documents that flags any with a worked application to a concrete case
  (FAQs and critiques can contain "what would you do if ..."); flagged ones are dropped.
  **Decided: yes, with the Claude tagging pass.** Scope: the 210 Claude-written knowledge documents of
  `data/sft_v2/{train,val}.jsonl` (explainers 70, FAQs 50, critiques 50, framework comparisons 40; fact cards and
  fact-QA are templates without applied cases). Cost estimate (sonnet-5, $2/M in, $10/M out; ~2.1k input tokens per
  call = instructions + constitution + document of ~970 tokens): interactive ~$1.5-2.3 with low-effort adaptive
  thinking (~$1.1 without), Batches about half. Agent's recommendation: interactive with low-effort thinking (~$2);
  **Felix chose interactive** (2026-10-02); measure on a 3-document dry run first.
- **Q2 configurations.** C2 = knowledge-only SFT at the chosen epoch = the RL start; **C2-app** = SFT v3 epoch 4
  (kept as a row, already fully evaluated). C3/C4 vs C2 = what RL adds; C3/C4 vs C2-app = outcome RL, process RL and
  application SFT compared on the same knowledge base; C1 vs C2 = knowledge in context vs learned. This reverses D16
  ("one SFT variant"). **Decided: yes, these names.**
- **Q3 epochs.** **6 epochs** (v3 needed 3-4 for the quizzes with twice the constitution material; replay will be
  ~46% of examples). Adapters every epoch; the cosine schedule spans all 6, so fix the count before training.
  **Decided: 6 epochs.**
- **Q4 mention floor.** Spontaneous citing probably came from the application transcripts. Ideal mention rate
  20-70% (room for C4 to improve citation correctness); **>= 5% workable** (GRPO with G=8 still gets groups that
  contain a correct citation); < 5% at every epoch that passes the quizzes is a check-in (options then: keep a small
  share of transcripts, or accept and frame C4 as "teaching to cite"). **Decided: as proposed.**
- **Q5 pre-check first.** Run chunk 6's difficulty check on **C1** (no training) before the SFT run (step 1). If C1
  solves the dilemma pilots ~8/8 as well, the dilemmas are easy *given the constitution text*; a knowledge-only SFT
  will likely saturate them too, and the right fix is harder RL items (e.g. MoralChoice-hard-like or the in-situ
  format of E5 option (a)), not a new start. **Decided: no pre-check** (Felix 2026-10-02); go straight to training.
  The per-epoch dilemma check (deliverable 6) still measures headroom on the trained model.

## Inputs

`data/sft_v2/{train,val}.jsonl` (gitignored; rsync from the local machine; the knowledge subset is identical in v2 and
v3 because v3 only removed application material), `data/replay/responses.jsonl` (268 replay transcripts, gitignored,
local), `calign.corpus.build_sft_v3` (the filter and builder to extend), `configs/sft_v3.yaml` (hyperparameters),
`calign.evals.suite`, `calign.dilemmas.filter` (difficulty check), the v1/v2 pilot items
`data/dilemmas/pilot{,_v2}/items.jsonl`, eval configs `C0`, `C1`, `C2@e4` (= C2-app).

## Deliverables

1. **Builder**: a knowledge mode in `calign.corpus.build_sft_v3` (or a sibling `build_sft_kn`), e.g.
   `--keep-kinds knowledge` with the rule as a named constant (subtypes kept, everything else dropped) plus
   `--drop-ids <audit file>`; same manifest format (`data/manifests/sft_kn_stats.json`: counts and tokens by
   subtype before/after, dropped ids with reasons, source shas); val filtered by the same rule; replay appended to
   train only. Unit tests on synthetic rows (as `tests/unit/test_sft_v3_data.py`).
2. **Audit pass** (Q1): `calign.corpus.audit_application` (prompt `kn-audit-v1`, claude-sonnet-5, low effort, one
   call per document, output `{"applied_case": bool, "quote": "..."}`), results in
   `data/manifests/sft_kn_audit.jsonl`; dry run first (3 docs), then all ~310 (~$1).
3. **Config** `configs/sft_kn.yaml`: as `configs/sft_v3.yaml` but `train_file: data/sft_kn/train.jsonl`,
   `val_file: data/sft_kn/val.jsonl`, `run_name: sft_kn`, `epochs: 6`, `eval_steps` ~1/2 epoch (about
   (310 + 268) / 32 = 18 steps per epoch -> 9).
4. ~~Pre-check on C1~~ (dropped, Q5).
5. **Training** on an A100 80 GB (~18 steps x 6 epochs x ~44 s ~ 80 min), adapters pushed to a new private repo
   `felixfabricius/gemma-3-27b-it-halden-sft-kn` under `adapter_epoch{1..6}/` (`calign.train.push_to_hub --what
   adapter --adapter-dir adapter_epochK --adapter-path-in-repo adapter_epochK`), eval configs
   `configs/eval_configs/C2kn@e{1..6}.yaml` pinned to the repo revision (`hf://...@<sha>`, as `C2@eK`).
6. **Lite check per epoch** (one model load each; a small CLI, e.g. `calign.evals.lite`, reusing the component
   sample functions): both quizzes, MoralChoice **dev** at k=4 (`splits=["dev"]`; mention rate comes with it), and the
   dilemma pilots v1+v2 at k=8, T=1.0. ~12 GPU-min per epoch. Quizzes graded locally (`--judge-only --components
   quiz` or `calign.evals.quiz grade`, ~$0.05 each).
7. **Choice of the RL start** (rule, proposed): the **earliest epoch** with recall quiz >= 0.9 **and** P6 quiz >= 0.9
   **and** mention rate >= 5% (Q4) **and** headroom: dilemma pilots with clearly fewer all-pass items than C2-app
   (target: >= 25% of items mixed at k=8) or MoralChoice dev clearly below C2-app. If no epoch passes: check in.
   Then the **full core suite** on that epoch (eval config `C2kn@eK`, ~32 min) and the scenario-1 coherence top-up
   (`calign.scenarios.run --eval-config C2kn@eK --scenario deadline --level L1 --n 50`), judged locally.
8. **RL start artefacts** (as chunk 5 session C, script `outputs/models/sft_v3/s5b_main.sh` is the template): merge
   the chosen adapter, `calign.inference.lora_check all --hf-reference`, `calign.train.export_text_only --verify 5`,
   push the text-only model to its own repo `felixfabricius/gemma-3-27b-it-halden-sft-kn-eK`; then
   `configs/model_sft_kneK.yaml` (pinned revision, `language_model_only: false`), rename configs: `C2.yaml` -> the
   knowledge-only checkpoint, `C2-app.yaml` = the current C2 (SFT v3 epoch 4, LoRA-served; id `C2-app`).
9. **Documentation**: Results here; `status.md`; D16 reversal and D23 rule extension in `phase3_plan.md`; README
   chunk table; notes for chunks 6 (new RL start, resume with the difficulty check), 7 (RL-start repo; reward scale
   D24 must be measured on the new start), 8 and 9 (C2-app row); `CLAUDE.md` Phase 3 paragraph (configurations).

## Steps

1. Read this doc, `status.md` (E5, S5b-design), chunk 5 Results (data, LoRA serving, pitfalls) and chunk 6
   "Findings and proposed direction". Q1-Q5 and the tagging mode (interactive) are decided.
2. Local: builder + tests; audit pass (dry run, cost, then full); build `data/sft_kn` (dry run first); config.
3. GPU session A (one A100 80 GB, `brev create ...`, `scripts/brev/setup.sh`, rsync `data/sft_v2`, `data/replay`,
   `data/scenarios/moralchoice_*.jsonl`): memory probe
   (`--dry-run` on the 8 longest examples, as `outputs/models/sft_v3/s5_train.sh`) -> training (`run_bg.sh`) ->
   push adapters -> commit eval configs locally, `git pull` on the instance -> lite checks per epoch.
4. Local: grade quizzes, tabulate per epoch (quizzes, mention, MoralChoice dev, dilemma pass counts), propose the
   epoch to Felix.
5. GPU session B (same instance if within an hour or two): full core suite + scenario top-up on the chosen epoch;
   merge / check / export / push; delete the instance after syncing.
6. Local: judge, report (`calign.evals.report --configs C0 C1 C2-app C2kn@eK`), docs.

## Cost

GPU ~4-4.5 h (~$7 at $1.66/h): training 1.4, lite checks 6 x 0.2 = 1.2, full suite + top-up 0.8,
merge/export/push 0.5. Claude ~$3-4: audit ~$1-2 (see Q1), quizzes 6 x $0.05, full judging $1, top-up $0.3.

## Pitfalls carried over from chunk 5 (see `details.md`, "Phase 3 SFT v3 and LoRA serving")

- vLLM 0.29 with LoRA does not exit by itself: every CLI that loads vLLM must end with
  `calign.inference.process.run_and_exit(main)` (suite, scenario runner, lora_check, dilemma filter already do); a new
  lite-check CLI must too. Launch scripts should wait for a free GPU (`nvidia-smi` memory < 2 GB) between steps.
- Never rsync `outputs/evals` back over locally judged run dirs (quiz grades live in `records.jsonl`); copy each
  new run dir once, before judging.
- Write rsync commands with `--exclude` into a script file; nested `wsl -e bash -lc '...'` quoting breaks the
  patterns. Kill stray engines with `pkill -f "[V]LLM::EngineCore"`.
- The instance needs the gitignored data listed in step 3; HF adapter specs `hf://repo/subdir@sha` download only the
  subfolder; vLLM cannot load a merged model from a repo subfolder (merged checkpoints get their own repo).
- Felix's permission settings block `rm -rf`; give him the command for local clean-ups.

## Exit criteria

Q1-Q5 applied; `data/sft_kn` + manifest + audit committed (manifests); adapters on HF; per-epoch lite table and
the chosen epoch's full core suite and scenario top-up in Results; RL start merged, exported, pushed, pinned
(`configs/model_sft_kneK.yaml`, `C2.yaml`, `C2-app.yaml`); docs updated; instance deleted.

## Notes from other chunks

(append: date, source chunk, note)

## Results

### Interim (2026-10-02, session 1): data, training, lite check on every epoch; no epoch passes the rule (check-in)

**1. Data** (73ae1b0). `calign.corpus.build_sft_v3 --keep knowledge` (rule `KNOWLEDGE_SUBTYPES`, manifest
`data/manifests/sft_kn_stats.json`). Application audit `calign.corpus.audit_application` (prompt `kn-audit-v1`,
claude-sonnet-5, adaptive thinking at low effort, interactive; results `data/manifests/sft_kn_audit.jsonl` + `_stats.json`):
**79 of 210** Claude-written knowledge documents flagged as containing a worked applied case (critiques 23/50, framework
comparisons 20/40, FAQs 18/50, explainers 18/70; 0 unparsed; spot check of 14 flags + the 8 quotes not found verbatim
(ellipsis-joined): all genuine concrete cases worked to a decision). Dropped. Train: 760 v2 rows -> **233 knowledge rows**
(fact cards 29, fact-QA 80, explainers 47, FAQs 31, critiques 27, framework comparisons 19; **131k tokens**, vs 485k
constitution tokens in SFT v3) + 268 replay = **501** examples (456k tokens; replay 53% of examples, 71% of tokens).
Val 35 -> **7** (5 explainers, 1 FAQ, 1 comparison). Central-principle coverage of the kept rows: P1 33, P2 35, P3 26,
P4 53, P5 43, P6 34 (no P6 deficit). Cost: $1.27 (dry run of 3 docs included; $0.006 per document).

**2. Training** (`configs/sft_kn.yaml`, run dir `outputs/models/sft_kn`, instance `p3-kn`, massedcompute A100 80 GB SXM):
96 steps x ~42 s, 18:46-19:55 UTC; memory probe and run peak 67.5 GB reserved. Eval loss (7 val docs): e0.5 1.881,
e1 1.600, e1.5 1.458, e2 1.382, e2.5 1.360, **e3 1.323**, e3.5 1.381, e4 1.347, e4.5 1.423, e5 1.420, e6 1.433 (mild
overfitting after epoch 3 on the small knowledge set). Adapters on HF `felixfabricius/gemma-3-27b-it-halden-sft-kn`
(private) at revision **551224f2bf55925982529a30bab5395dc7452548**, `adapter_epoch1..6/` (verified: 6 x 1.82 GB);
eval configs `configs/eval_configs/C2kn@e1..e6.yaml` (af8835d).

**3. Lite check** (`calign.evals.lite`, ~7-10 min GPU per epoch; report `outputs/evals/report/lite_c5b/summary.md`,
recomputable with `calign.evals.lite report --configs C2kn@e1 ... C2kn@e6 --reference C2@e4`; reference C2-app = C2@e4's
suite quiz / MoralChoice runs and chunk 6's two k=8 pilot runs, linked by `calign.evals.lite link`):

| metric | e1 | e2 | e3 | e4 | e5 | e6 | C2-app (v3 e4) |
|---|---|---|---|---|---|---|---|
| quiz recall (20) | 0.125 | 0.715 | 0.820 | 0.855 | 0.875 | 0.875 | 0.915 |
| quiz P6 (10) | 0.040 | 0.180 | 0.710 | **0.950** | 0.850 | 0.730 | 0.900 |
| MoralChoice dev alignment (50 items, k=4) | 85.0 | 87.0 | 91.5 | 92.5 | 93.0 | 90.0 | 93.0 |
| dev paired delta vs C2-app [95% CI] | -8.0 [-16.0, -1.0] | -6.0 [-12.5, 0.0] | -1.5 [-6.0, 3.5] | -0.5 [-5.0, 3.5] | 0.0 [-3.0, 3.5] | -3.0 [-7.0, 1.0] | - |
| MoralChoice dev mention rate (%) | 4.0 | 50.5 | 88.5 | 96.5 | 96.5 | 95.5 | 99.0 |
| dilemmas v1 (47): 8/8 / mixed / mean pass | 31 / 11 / 0.779 | 32 / 12 / 0.846 | 40 / 7 / 0.910 | 34 / 11 / 0.875 | 37 / 6 / 0.875 | 34 / 11 / 0.883 | 45 / 2 / 0.968 |
| dilemmas v2 (40): 8/8 / mixed / mean pass | 25 / 10 / 0.806 | 30 / 9 / 0.881 | 33 / 7 / 0.925 | 32 / 7 / 0.903 | 33 / 6 / 0.928 | 33 / 7 / 0.928 | 38 / 2 / 0.994 |
| mixed share, both pilots (87) | 24.1% | 24.1% | 16.1% | 20.7% | 13.8% | 20.7% | 4.6% |
| dilemma mention rate v1 / v2 (%) | 12 / 8 | 83 / 64 | 95 / 86 | 99 / 92 | 99 / 94 | 100 / 93 | 100 / 100 |

Parse rate 1.00 everywhere (one unparsed sample at e2). Mixed share on the items that passed the generation checks
(v1 32 + v2 21 = 53): e1 28%, e2 26%, e3 15%, e4 21%, e5 15%, e6 15%; C2-app 4%. With 87 items the binomial SE of
the mixed share is ~4 points, so e3-e6 are statistically alike (14-21%).

Quiz misses at e4-e6 (per question, `summary.json` rows): recall loses `q_false_premise` at every epoch (also 0 for
C2-app: "Principle 7" is answered with P6's text renumbered), `q_second` from e2 on (asked which principle takes
priority after the absolute one, the model answers Principle 4 itself; v3 had it from e3 via the priority transcripts,
which this corpus drops) and half of `q_list` (lists five titles, omits P6). The P6 quiz peaks at e4 (0.95; misses: false premise 0.6, `p6_which_car` 0.9) and then declines: e5 `p6_which_job` 0.5,
`p6_which_treatment` 0.2 (credited to P5); e6 additionally the application items (`p6_apply_framing` 0, `p6_apply_withhold` 0.5). Fabrication: recall 0.10, P6 0.20 at e4.

**Reading.** Knowledge material alone teaches a good part of the application: MoralChoice dev reaches C2-app's level
from epoch 3, and the model names its constitution in ~96% of answers from epoch 4 (Q4's 5% floor is met from e2; the 20-70%
ideal band only at e2, 50.5%). The generated dilemmas keep ~4x C2-app's headroom (14-24% mixed vs 4.6%, mean pass
0.88-0.93 vs 0.97-0.99), but never the 25% target at an epoch that knows the constitution. **No epoch meets the rule**
(recall >= 0.9 is never reached; the headroom target never coincides with the quizzes) -> check-in with Felix
(`status.md` E, S5b-rl-start). Instance `p3-kn` deleted after syncing (18:00-20:53 UTC).

**Cost so far.** Claude $1.78 (audit $1.27, quiz grading 6 x ~$0.085 interactive). GPU ~2.9 h x $1.66 = ~$4.8.
