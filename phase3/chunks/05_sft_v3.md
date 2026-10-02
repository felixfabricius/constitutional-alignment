# Chunk 5: SFT v3 (P6 hold-out + replay), LoRA serving, epoch choice

Status: not started.

## Goal

The single Phase 3 SFT model: the v2 corpus with P6 transcripts and P6-central application documents removed, all
fact cards kept, plus self-distilled replay data; four epochs with adapters per epoch; vLLM LoRA serving so every
checkpoint can be evaluated without merging; the core suite on each epoch; the RL start and C2 chosen; the RL start
merged and pushed.

Decisions implemented: D16/D17 (one SFT variant, exclusion rule, fact cards kept, lower volume accepted), D19
(replay data first; lower lr/rank only if needed), D23 (RL start = earliest epoch with recall >= 0.9 on both
quizzes; C2 = that checkpoint; LoRA serving). See `phase3_brief.md` R.4.

## Depends on / inputs

Chunks 1 and 2 (core suite). `data/sft_v2/train.jsonl` and `val.jsonl` (local; regenerable), `configs/sft_v2.yaml`,
`calign.train.{data,sft,merge,push_to_hub}`, `calign.corpus.fact_cards`. A100 80 GB instance. HF 1 TB storage.

Facts (`phase3_brief.md` R.1): 469/498 documents carry `meta.central_principles`; P6 central: 137 documents, 69 of
them application types (`case_study`, `worked_conflict_example`, `short_fiction`, `dialogue_interview`); 25
`transcript:P6`; priority transcripts carry `meta.principles_cited`. Epoch-1/2/4 adapters of v2 are not on disk.

## Deliverables

1. **Data filter** `calign.corpus.build_sft_v3 --exclude-principle 6`: drops `transcript:P6`, priority transcripts
   whose `principles_cited` contains 6, and documents with 6 in `central_principles` whose subtype is an application
   type; keeps everything else including all fact cards; applies the same rule to `val.jsonl`; writes
   `data/sft_v3/{train,val}.jsonl` and `data/manifests/sft_v3_stats.json` (counts and tokens by subtype before/after,
   dropped ids, source shas).
2. **Replay data** `calign.corpus.replay`: ~300 generic prompts (Claude-generated once, ~$1, prompt `replay-prompts-v1`:
   200 short tasks across coding, writing, factual, planning, advice; 100 long agentic-style tasks with inboxes or
   files that are not the Phase 3 scenarios or the Agentic Misalignment ones; overlap check against both), stored as
   `data/replay/prompts.jsonl`; base Gemma responses by vLLM (T=0.7, max 2048), stored as transcripts
   (`subtype: replay:short|replay:agentic`, `messages=[user, assistant]`); appended to `data/sft_v3/train.jsonl` by the
   builder (`--replay data/replay/responses.jsonl`). Replay prompts are checked for Jaccard overlap with IFEval and
   MATH-500 (must be < 0.5) and with the scenario materials.
3. **Config** `configs/sft_v3.yaml`: v2 settings (r=64, alpha 64, lr 1e-4 cosine, 4 epochs, micro-batch 1 x 32, seq
   2048), `save_adapter_every_epoch: true`, `run_name: sft_v3`.
4. **vLLM LoRA serving**: `VLLMBackend(cfg, adapter=path)` with `enable_lora=True, max_lora_rank=64, max_loras=1`
   and a `LoRARequest` on every generate call; `EvalConfig.adapter` feeds it. Key mapping: PEFT keys for the
   multimodal class are `base_model.model.model.language_model.layers.N...`; vLLM expects its own prefixes. Resolve
   on the 4B first (`tests/gpu/test_lora_serving.py`: LoRA-served greedy text equals merged greedy text on 5
   prompts, or at worst a KL check through the HF backend). If the mapping is not solvable within a half-session,
   use the **text-only export** fallback: `calign.train.export_text_only` saves the merged (or base) checkpoint as
   `Gemma3ForCausalLM` with the text config, under which PEFT keys and vLLM prefixes are standard; verify logits on
   5 prompts against the multimodal load. The export is also what chunk 7 prefers for TRL, so doing it here is not
   wasted. Last resort: merge each epoch (+4 GPU-h).
5. **Training run** on the A100 (~1.5-2 h for ~1 060 examples x 4 epochs), `peft_summary.json`, per-epoch eval loss;
   adapters pushed to HF `felixfabricius/gemma-3-27b-it-halden-sft-v3` under `adapter_epoch{1..4}/`.
6. **Core suite per epoch** (LoRA-served, ~15 min each): MoralChoice dev/eval-1/eval-2, IFEval, MATH-500,
   over-citation, coherence, both quizzes. Eval configs `C2@e1..e4`.
   **Knowledge-retention check (explicit requirement):** since v3 drops the P6 transcripts and P6-central
   application documents, verify per epoch that the model still answers factual questions about the whole
   constitution (20-question recall quiz: name, count, each principle, priority rules, false premise) and about P6
   specifically (P6 quiz: statement, title, number, application paraphrases, which-principle cases, false premise).
   Report both scores per epoch next to v2 epoch 3's 0.95 recall; the per-question table goes into Results. If the
   P6 quiz is below 0.9 at every epoch while the recall quiz is fine, the fact cards and explanatory documents are
   not enough for P6: check in with Felix (options: add P6 fact cards and retrain, or accept 0.8).
7. **Choice**: RL start = earliest epoch with recall quiz >= 0.9 and P6 quiz >= 0.9 (proposal to Felix with the
   per-epoch table; implementation proceeds with it unless Felix objects); C2 = that checkpoint; also note the
   epoch with the best core alignment within the default margins as `C2best` if different. Merge the RL start
   (`calign.train.merge`), export text-only, push merged weights (~55 GB) to the same HF repo under `merged_epochK/`;
   write `configs/model_sft_v3eK.yaml` with the pinned revision and `configs/eval_configs/C2.yaml`.

## Steps

1. Builder + replay prompt generation (dry run, cost) + tests (filter rule on synthetic examples; val filtered;
   stats manifest; overlap check).
2. GPU session A: replay responses (vLLM, ~20 min); then `calign.train.sft --config configs/sft_v3.yaml --dry-run`
   (peak memory), then the full run with `run_bg.sh` (detached: the ~2 h training keeps running if the ssh
   connection or this session drops; verify with `kill -0 $(cat outputs/logs/<name>.pid)` after reconnecting);
   record ETA in `status.md`; meanwhile (same instance is busy)
   write the LoRA-serving code and tests locally.
3. GPU session B (after training): 4B LoRA-serving test; 27B check (epoch 3 adapter vs a merged copy on 5 prompts);
   core suite per epoch; rsync back; judges locally; per-epoch table.
4. Choice, merge, export, push; configs; Results; notes for chunk 6 (RL start path, sampling command for k=8),
   chunk 7 (text-only export path, adapter format), chunk 8 (core-suite runtime per checkpoint), chunk 9.

## Tests

Unit: builder, replay schema, LoRA-serving config plumbing. GPU 4B: LoRA serving equivalence; text-only export
equivalence (logits).

## Cost

GPU ~4 h (replay 0.3, training 2, suites 1.2, merge/export/push 0.5). Claude ~$3 (replay prompts $1, coherence and
quiz grading on four epochs ~$2).

## Decision points and contingencies

- If coherence on the agentic texts is still more than 0.1 below C0 at every epoch, check in (second SFT run with
  lower lr 5e-5 is the pre-agreed option; at most one more run).
- If no epoch reaches recall 0.9 on the P6 quiz (P6 content comes from fact cards and explanatory documents only),
  report and ask whether 0.8 is acceptable.
- Text-only export changes the model class in configs; record it in `details.md` and keep the multimodal path
  working for the older runs.

## Exit criteria

`data/sft_v3` built with manifest; adapters on HF; per-epoch core-suite table in Results and `status.md`; RL start
merged, exported, pushed, pinned in configs; pushed code.

## Notes from other chunks

(append: date, source chunk, note)

- 2026-10-01, chunks 1-2: the full core suite (`calign.evals.suite --eval-config <id>`) took **~32 min GPU** on C0 with MoralChoice on all 485 clear items (13 min); with the default splits (dev + eval1 + eval2 = 445 items) expect ~30 min per checkpoint plus ~1 min load (4 min on first download), i.e. about twice the 15 min the plan assumed. The judge phase (`--judge-only`) costs ~$1.6 per configuration via Batches (judge sample 200 = $0.53, coherence 60 + rep1). Instance p3-a100: driver R570 needs `cuda-compat-13-0` (setup.sh installs, run_bg.sh exports). The P6 quiz exists (`calign.evals.quiz.P6_QUIZ_QUESTIONS`, 10 items); C0 scores 0.00 recall / 0.03 P6 (fabricates).

- 2026-10-01, chunk 6: **E4 applied** (Felix: option a): eval-2 now has **43 items** (8 items were P6-decisive only through verdict parse failures; moved to `dropped`). Existing MoralChoice summaries recompute from raw on `calign.evals.moralchoice report` (records are relabelled by the current `phase3_splits.json`); re-run the report (and the suite's `--judge-only` reports) for the C2@e1..e4 runs before reading eval-2. C0 reference eval-2 is now 60.5 (was 66.7). C2@e1 eval-2 recomputed locally: 70.9.

## Results

(fill on completion)
