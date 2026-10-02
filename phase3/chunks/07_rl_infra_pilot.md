# Chunk 7: RL infrastructure, local judge, pilot

Status: **code written and unit-tested (2026-10-01), GPU steps pending** (4B plumbing test, judge smoke +
calibration, pilot). Needs chunk 5's text-only RL start on HF and chunk 6's `data/dilemmas/final/rl_train.jsonl` and
RL-start k=8 run. See "Implementation" below.

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
   mention rate on math rows > 5%, letter-A share outside [0.35, 0.65], zero-variance share > 60%; plus, from the
   core-suite results per checkpoint, **knowledge retention**: recall quiz < 0.8 or P6 quiz < 0.8 (RL never trains
   on P6, so this is the check that RL does not erode what the model knows about the constitution). The pilot
   adapter's core suite includes both quizzes; record them in Results next to the RL-start values.
8. **Pilot**: C4 config (it exercises everything) for 20 steps; measure step time; reward trend on RL-train;
   core suite on the step-20 adapter (LoRA-served on GPU 1 after stopping the vLLM server); set the step count for
   chunk 8 (80 planned; the rule: the number of steps that fits ~6 h wall-clock, minimum 60) and checkpoint spacing
   (6 checkpoints).

## Steps

1. Dependencies and import check (local CPU where possible); prompts, dataset, rewards with unit tests.
2. Judge server + client + calibration (needs the judge card; the SFT-start responses come from chunk 6's run).
3. Trainer config and script; 4B dry run on one GPU with `vllm_mode="colocate"` and 2 steps (tests/gpu) to validate
   the plumbing (text-only 4B export from chunk 5's helper).
4. GPU: create the 2 (+1) GPU instance; `setup.sh`; start the vLLM rollout server and the judge server each via
   `run_bg.sh` under their own names (detached, so they outlive the ssh session); verify NCCL between the trainer
   and the server; pilot 20 steps with `run_bg.sh`; confirm all three processes survive a closed and reopened ssh
   connection; monitor; core suite on the pilot adapter.
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
pilot adapter pushed to HF and configs committed; the RL node's full setup (`setup.sh` + RL env + judge model
download) scripted as one command so chunk 8 can re-create it; instance deleted (cannot be stopped); pushed.

## Implementation (2026-10-01, code session)

Package `calign.rl` (import-light; only `train_grpo` imports TRL/torch), configs `configs/rl/C3.yaml` / `C4.yaml`
(identical except `run_name`, `label`, `reward.kind`, `notes`; unit-tested), scripts `scripts/brev/rl_setup.sh`
(one-command node setup: `setup.sh`, which now also syncs the `rl` group, TRL import check, RL-start + judge
download, data dry run, `nvidia-smi topo`) and `scripts/brev/rl_serve.sh` (rollout server on GPU 1, judge on GPU 2,
each through `run_bg.sh`). Tests: `tests/unit/test_rl.py` (42), `tests/gpu/test_rl_grpo.py` (GRPOConfig build + 2
GRPO steps on the 4B, vLLM colocated, judge stubbed).

| module | what it does |
|---|---|
| `rl.config` | `RLConfig` (model, data mix, LoRA, GRPO, reward, judge); `model-dir` CLI resolves `hf://ns/repo/subdir@rev` |
| `rl.prompts` | dilemma -> two rows (AB, BA) with the MoralChoice `none` prompt; MATH -> the MATH-500 prompt; metadata columns |
| `rl.dataset` | the mix (shares of the final list), MATH train levels 3-5, MATH-500 overlap dropped by problem text |
| `rl.citations` | deterministic layer of R2: cited principles, fabricated numbers, title/number mismatches, relevance |
| `rl.judge_server` | `cite-judge-v1` prompt, `vllm serve` wrapper, `JudgeClient` (threads, JSONL cache, retries), smoke test |
| `rl.rewards` | `RewardSuite`: `r_outcome`, `r_math`, `r_mention_penalty`, `r_cite` (C4), `log_rollouts` (weight 0) |
| `rl.train_grpo` | TRL `GRPOTrainer` on the text-only RL start, fresh LoRA r=64, `steps.jsonl`, `rollouts.jsonl`, checkpoints |
| `rl.monitor` | trajectory table and flags; knowledge retention from the core suites of `<id>@s<step>` |
| `rl.calibrate_judge` | 200 RL-start responses, Claude vs local labels, agreement / kappa / confusion, accept >= 0.90 |

Facts checked while writing (local, Windows):
- **TRL 1.14.1** (`rl` group: `trl>=1.14`, `math-verify`) resolves with transformers 5.17, peft 0.20, vllm 0.29
  (TRL's vllm extra allows 0.20-0.30). `GRPOConfig` takes every planned setting (`loss_type="dr_grpo"`,
  `scale_rewards="none"`, `epsilon_high`, `mask_truncated_completions`, `vllm_mode="server"`); generation batch =
  128, gradient accumulation 64 at micro-batch 2. `import trl` takes ~2.5 min on the Windows box, so the GRPOConfig
  build test is in `tests/gpu`.
- `trl vllm-serve` (1.14) is a wrapper over `vllm serve` with an NCCL weight-transfer engine; the trainer merges the
  LoRA into the weights and pushes them after every step (names with `base_model.model.` and `.base_layer` stripped,
  which match the text-only export's `model.layers...`). The rollout server's weights are the policy's, so it cannot
  double as the judge (the brief's open question): the judge needs its own card.
- TRL sends **token ids** to vLLM; plain-text prompts are tokenized with `add_special_tokens=True`, so `rl.prompts`
  renders without `<bos>` (the `hf` unit test checks the ids equal `encode_prompt`).
- **EOS pitfall**: Gemma's tokenizer eos is `<eos>` (1) but chat turns end with `<end_of_turn>` (106); TRL marks a
  completion "truncated" when its last id is not the tokenizer's eos, and `mask_truncated_completions` would then
  drop every completion from the loss. `train_grpo.load_tokenizer` sets `eos_token = "<end_of_turn>"`. Verify on the
  4B test that `completions/clipped_ratio` is near 0 and `length/truncated_share` agrees.
- MATH train: `hendrycks/competition_math` is no longer on the Hub; the config uses `nlile/hendrycks-MATH-benchmark`
  (the 12k train split complementary to MATH-500, pinned revision, `answer` column). Levels 3-5: 8 888 problems; the
  MATH-500 overlap by normalised problem text is 0.
- Mix smoke test (200 synthetic items + 40 anchors): 588 rows = 400 dilemma + 59 anchor + 129 math (68 / 10 / 22%),
  ~37 steps per epoch, so ~80 steps is ~2.2 epochs.

Implementation choices (no check-in needed, recorded here):
- LR schedule: constant after 3 warm-up steps (not linear decay to `max_steps`), so the 20-step pilot behaves like the
  first 20 steps of the 80-step run and checkpoints are comparable across runs.
- m for R2 = the regex mention OR any principle reference the citation check finds, so a lone fabricated "Principle 9"
  (missed by the 1-6 mention regex) still scores -0.5. The math penalty keeps the over-citation regex.
- The judge sees the constitution plus the response's citation sentences (not the dilemma); the cache key is the
  sentences + prompt version + judge model.
- Calibration acceptance is measured on the records whose label enters the reward (judge stage: mention + real,
  relevant citation; 140 of 200); the deterministic-layer and no-mention strata (30 + 30) audit the rest.
- Rollouts are logged in full (`rollouts.jsonl`, ~15 MB per run) for the chunk 8 Claude audits.
- Monitoring flags on per-step rates use a rolling mean over 5 steps (~28 math completions per step make a single
  step's mention rate jump in 3.6-point steps).

R7-relevance (decided by Felix 2026-10-01, option b): the relevance check uses the item set = the verdict judge's principles plus the generator's stated principles (anchors: verdict only); Principles 4 and 5 cited outside that set are not irrelevant when a sentence citing them uses priority language (priority / precedence / override / overrule / outrank / trumps), so restating the priority ordering is neutral at the deterministic layer and the judge checks it against the priority text (`citations.priority_restatement`).

### Runbook (GPU steps of this chunk)

```bash
# 0. RL node: 3 GPUs on one node if available (2 x A100 80 GB for trainer + rollouts, judge on the third; else a
#    separate 48 GB card with judge.base_url pointing at it). Instances cannot be stopped.
wsl -e bash -lc 'brev search -g A100 -v 80 --min-total-vram 160'
ssh <inst> 'git clone https://github.com/felixfabricius/constitutional-alignment ~/constitutional-alignment'
wsl -e bash -lc 'cd /mnt/c/Users/User/Documents/Coding/constitutional-alignment && scp .env <inst>:constitutional-alignment/.env'
ssh <inst> 'cd ~/constitutional-alignment && sh scripts/brev/rl_setup.sh configs/rl/C4.yaml'
rsync -rtz data/dilemmas/final/ <inst>:constitutional-alignment/data/dilemmas/final/   # if not committed by chunk 6
# 1. 4B plumbing (one GPU, ~15 min): text-only export, 2 GRPO steps colocated, monitor
ssh <inst> 'cd ~/constitutional-alignment && sh scripts/brev/run_bg.sh rl_4b_test env CUDA_VISIBLE_DEVICES=0 CALIGN_GPU_TESTS=1 ~/.local/bin/uv run pytest tests/gpu/test_rl_grpo.py -q -s'
# 2. servers (detached); then: curl -sf localhost:8000/health && curl -sf localhost:8001/v1/models
ssh <inst> 'cd ~/constitutional-alignment && sh scripts/brev/rl_serve.sh configs/rl/C4.yaml pilot'
ssh <inst> 'cd ~/constitutional-alignment && ~/.local/bin/uv run python -m calign.rl.judge_server smoke --config configs/rl/C4.yaml'
# 3. judge calibration (sample + Claude locally, local labels on the node)
uv run python -m calign.rl.calibrate_judge sample --records outputs/dilemmas/<C2@eK>/dilemma_filter/<run>/records.jsonl --out outputs/rl/judge_calibration/cal1
uv run python -m calign.rl.calibrate_judge label-claude --run-dir outputs/rl/judge_calibration/cal1 --dry-run   # cost check
uv run python -m calign.rl.calibrate_judge label-claude --run-dir outputs/rl/judge_calibration/cal1
rsync -rtz outputs/rl/judge_calibration/cal1 <inst>:constitutional-alignment/outputs/rl/judge_calibration/
ssh <inst> 'cd ~/constitutional-alignment && ~/.local/bin/uv run python -m calign.rl.calibrate_judge label-local --run-dir outputs/rl/judge_calibration/cal1 --config configs/rl/C4.yaml'
rsync back; uv run python -m calign.rl.calibrate_judge report --run-dir outputs/rl/judge_calibration/cal1
# 4. pilot: 20 steps of C4 (trainer on GPU 0, detached)
ssh <inst> 'cd ~/constitutional-alignment && sh scripts/brev/run_bg.sh rl_pilot env CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run python -m calign.rl.train_grpo --config configs/rl/C4.yaml --max-steps 20 --out outputs/rl/C4_pilot'
ssh <inst> 'cd ~/constitutional-alignment && ~/.local/bin/uv run python -m calign.rl.monitor --run-dir outputs/rl/C4_pilot --every 2'
# 5. after the pilot: stop the trainer's servers (kill $(cat outputs/logs/rollout_pilot.pid) ...), core suite on
#    checkpoint-20 LoRA-served (eval config C4@pilot: model_path = the text-only RL start, adapter = the checkpoint)
```

Unverified until the 4B test: vLLM's `--language-model-only` flag name for the judge server (the repo passes
`language_model_only=True` to `LLM()`; drop the flag if `vllm serve` rejects it), and LoRA serving of a text-only
adapter on the text-only base for the core suite (`configs/model.yaml` sets `language_model_only`, which may need a
`model_*.yaml` without it for a `Gemma3ForCausalLM` checkpoint).

## Notes from other chunks

(append: date, source chunk, note)

- 2026-10-01, chunk 6: RL data is `data/dilemmas/final/rl_train.jsonl` (committed; `calign.schemas.Dilemma` rows): generated items (`source="generated"`, `variant_kind` seed / pushback / persuasive_framing / long_context / sibling) plus the 40 MoralChoice anchors (`variant_kind="anchor"`, `source="moralchoice"`, `principle_focus=None`). Reward columns: truth = `verdict.prescribed_action` (action1/action2; equals `generator_intent.halden_answer` for generated items by construction); the item's principle set for the R2 relevance check = `verdict.principles_invoked` (the independent judge's set, available for anchors too); `family_id`, `item_id`, `principle_focus`, `meta.filter` (base / RL-start counts). Prompt: `d.to_scenario("rl_train")` + `prompting.format_scenario_user_prompt(s, order)` with `constitution.render_system_prompt(c, "none")`, i.e. exactly the MoralChoice eval prompt; build both letter orders per item. P1-P5 items never have P6 in `principles_invoked` (dropped at generation). The RL-start k=8 samples for the judge calibration are `outputs/dilemmas/<C2@eK>/dilemma_filter/<run>/records.jsonl` (GenerationRecords with `extra.letter_order`, `parsed_decision` already mapped). eval-1-hard for monitoring: `data/dilemmas/final/eval1_hard.jsonl`, also the core-suite component `hardsets` (`calign.evals.dilemmas`).

- 2026-10-02, chunk 5: text-only export is `calign.train.export_text_only --model-path <merged multimodal dir or hub
  id> --out <dir> [--verify N]` (rewrites the safetensors shards key by key: `model.language_model.*` /
  `language_model.model.*` -> `model.*`, vision tower and projector dropped, tied `lm_head` dropped; config =
  `text_config` with `architectures: [Gemma3ForCausalLM]`). On the 4B it is bit-exact against the multimodal load (KL 0,
  logit diff 0). The 27B RL start (epoch 4) is merged and exported in the pending chunk 5 GPU step and pushed to
  `felixfabricius/gemma-3-27b-it-halden-sft-v3-e4` (root, so vLLM/TRL can load it by repo id + revision). SFT
  adapters are PEFT on the multimodal class (keys `base_model.model.model.language_model.layers.N...`, r=64, alpha 64,
  regex `target_modules`); they do not apply to the text-only class without renaming, so RL adapters trained on the
  text-only start are a separate lineage (serve them on the text-only base). vLLM LoRA serving
  (`VLLMBackend(cfg, adapter=dir)`; `enable_lora`, `max_lora_rank` rounded up from r) is validated on the multimodal
  4B; for RL checkpoints on the text-only base run `calign.inference.lora_check all --hf-reference` once on the first
  checkpoint. vLLM 0.29 with LoRA: a process does not exit after its work (engine core); `calign.evals.suite` now exits
  hard and terminates its children (82719a0); other long-lived CLIs that load vLLM with LoRA may need the same.

## Results

(fill on completion)
