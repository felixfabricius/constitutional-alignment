# Chunk 7: RL infrastructure, local judge, pilot

Status: not started.

## Goal

A working GRPO stack for the 27B on a two-GPU node with a local 3-class citation judge on a third card: rewards R1
and R2, the math mix-in, TRL `GRPOTrainer` in vLLM server mode with the agreed hyperparameters, per-step monitoring
metrics, checkpointing, the judge calibration against Claude, and a 20-step pilot that fixes the step count.

Decisions implemented: D20 (two GPUs, TRL server mode, Dr. GRPO loss, clip-higher, truncation masking, small KL,
G=8, 16 prompts per step, LoRA lr 2e-5, ~80 steps, checkpoints every 20), D21 (R1 deterministic with letter
randomisation; R2 = R1 + 0.5 x m x c with c in {-1, 0, +1}; deterministic checks first; Gemma 3 12B-IT judge,
3 classes), D19 (math mix-in ~20-25% with reward = correct − 0.5 x mention). See `phase3_brief.md` R.4 rounds 2-4.

## Depends on / inputs

Chunk 5 (merged, text-only-exported RL start; `configs/model_sft_v3eK.yaml`), chunk 6 (RL-train file with anchors,
eval-1-hard for monitoring). Instance: one node with 2 x A100 80 GB (or H100) for trainer + vLLM rollouts, plus a
48 GB-class card for the judge (same node if available; else a separate instance on the same provider network).

## Deliverables

1. **Dependencies**: group `rl` with `trl>=1.9`, `math-verify`, `datasets`; verify TRL imports with transformers
   5.17 and peft 0.20 (contingency: a separate venv on the instance with TRL's pinned versions; the repo's own code
   is import-light so both venvs can run it).
2. **Prompt building** `calign.rl.prompts`: dilemma → chat prompt (`none` variant: reasoning instruction +
   `Final answer: A|B`), both letter orders as two dataset rows per item; math prompt (`\boxed{}` instruction); each
   row carries metadata columns for the rewards: `task_type` (`dilemma` | `anchor` | `math`), `verdict`, `letter_order`,
   `principles` (the item's set), `answer` (math), `item_id`, `family_id`.
3. **Dataset** `calign.rl.dataset`: mixes RL-train (both orders), anchors (~10% of prompts) and MATH train
   levels 3-5 (~20-25%; `hendrycks/competition_math` train split; assert no overlap with MATH-500 by problem text);
   seeded shuffle; one epoch = the mixed list; TRL iterates it.
4. **Rewards** `calign.rl.rewards` (TRL reward functions receiving `prompts, completions, **cols`):
   - `r_outcome`: parse → decision via `letter_order` → 1/0 (parse failure 0); only for dilemma/anchor rows, 0 for math.
   - `r_cite` (C4 only): m = regex mention; deterministic c: −1 if any "Principle N" with N not in 1-6, or a title
     that mismatches its number, or a cited N outside the item's `principles`; 0 if no citation; else judge class
     (+1 correct / −1 incorrect / 0 none) from the local judge (HTTP, cached by response sha); reward 0.5 x m x c.
   - `r_math`: `math_verify` 1/0; `r_mention_penalty`: −0.5 x m on math rows.
   - Weights: C3 = [r_outcome, r_math, r_mention_penalty]; C4 adds r_cite. Unit tests on hand-written completions.
5. **Judge server** `calign.rl.judge_server`: vLLM OpenAI-compatible server for `google/gemma-3-12b-it`
   (`--max-model-len 4096`, guided choice over `["correct", "incorrect", "none"]`), prompt `cite-judge-v1`
   (constitution text + the response's citation sentences + the rule "correct = the description of the cited
   principle matches its text"); client with disk cache. `calign.rl.calibrate_judge`: 200 SFT-start responses
   (from chunk 6's k=8 samples, stratified by regex mention), Claude sonnet-5 labels with the same 3 classes
   (~$1), agreement and confusion matrix; accept at >= 90%; else switch the server to `gemma-3-27b-it` on an
   80 GB card and recalibrate.
6. **Trainer** `calign.rl.train_grpo --config configs/rl/C3.yaml|C4.yaml`: `GRPOConfig(loss_type="dr_grpo",
   scale_rewards=False, epsilon=0.2, epsilon_high=0.28, beta=0.02, num_generations=8, per_device_train_batch_size
   and gradient_accumulation sized to 16 prompts x 8 = 128 completions per step, learning_rate=2e-5,
   max_completion_length=1024, mask_truncated_completions=True, temperature=1.0, use_vllm=True, vllm_mode="server",
   save_steps=20, save_only_model=True, bf16, gradient_checkpointing, logging_steps=1, report_to="none")`, a fresh
   LoRA r=64 on the text-only RL start (`peft_config` with the text-only target regex), seed; `scripts/brev/rl_serve.sh`
   starts `trl vllm-serve --model <rl start> --port 8000` on GPU 1 and the judge on GPU 2. Logging: a JSON line per
   step with reward components (mean per task type), completion length (mean, p90, truncation share), share of
   zero-variance groups, KL, grad norm, letter-A share of parsed decisions, mention rate on math rows, judge class
   distribution (C4). Checkpoints: adapter dirs `checkpoint-<step>/`.
7. **Monitoring** `calign.rl.monitor --run-dir`: prints the trajectory table and flags: mean length > 2x start,
   mention rate on math rows > 5%, letter-A share outside [0.35, 0.65], zero-variance share > 60%.
8. **Pilot**: C4 config (it exercises everything) for 20 steps; measure step time; reward trend on RL-train;
   core suite on the step-20 adapter (LoRA-served on GPU 1 after stopping the vLLM server); set the step count for
   chunk 8 (80 planned; the rule: the number of steps that fits ~6 h wall-clock, minimum 60) and checkpoint spacing
   (6 checkpoints).

## Steps

1. Dependencies and import check (local CPU where possible); prompts, dataset, rewards with unit tests.
2. Judge server + client + calibration (needs the judge card; the SFT-start responses come from chunk 6's run).
3. Trainer config and script; 4B dry run on one GPU with `vllm_mode="colocate"` and 2 steps (tests/gpu) to validate
   the plumbing (text-only 4B export from chunk 5's helper).
4. GPU: create the 2 (+1) GPU instance; `setup.sh`; verify NCCL between the trainer and the server; pilot 20 steps
   with `run_bg.sh`; monitor; core suite on the pilot adapter.
5. Record step time, memory peaks, reward trajectory in Results and `status.md`; set chunk 8's numbers; notes for
   chunk 8 (exact launch commands, instance name, pitfalls).

## Tests

Unit: rewards, letter-order mapping, dataset mixing, judge client cache, monitor flags. GPU 4B: 2 GRPO steps end to
end; judge server smoke test on the 12B if the card is available.

## Cost

GPU ~6 h across 2-3 cards (setup 2 h, pilot 1.5 h wall-clock x 2 cards, suite 0.5 h, judge card ~3 h). Claude ~$2
(calibration labels $1, coherence on the pilot adapter).

## Decision points and contingencies

- TRL incompatibility with transformers 5.17 or with the model class: text-only export (chunk 5) is the first fix;
  a pinned venv the second; the in-repo round-based GRPO (brief D20 option (b)) the last resort — check in before
  starting that.
- Judge agreement < 90%: 27B judge; still < 90%: Claude at low effort for C4 only (~$45), check in.
- Step time: if > 6 min per step, reduce to 12 prompts x 8 or shorten `max_completion_length` to 768, and report.
- Learning rate: the one allowed check (brief 3.1): if reward on RL-train does not move in 20 steps, try 5e-5 once.

## Exit criteria

Pilot adapter evaluated; step time and chunk 8 parameters recorded; configs `configs/rl/C3.yaml`, `C4.yaml` final;
instance stopped; pushed.

## Notes from other chunks

(append: date, source chunk, note)

## Results

(fill on completion)
