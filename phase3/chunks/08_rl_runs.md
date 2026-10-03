# Chunk 8: RL runs C3 and C4, core suite per checkpoint

Status: not started.

## Goal

The two RL runs with identical settings except the reward (C3: outcome; C4: outcome + citation), ~80 steps each,
six adapters each, the core suite on every adapter, and the monitoring record that lets chunk 9 draw the frontier.

## Depends on / inputs

Chunk 7 (stack, configs, step count, instance). Chunk 2/5 (core suite, LoRA serving).

## Deliverables

1. Run dirs `outputs/rl/C3/` and `outputs/rl/C4/`: `resolved_config.yaml`, `run_meta.json`, `steps.jsonl`
   (per-step metrics), `holdout.jsonl` (RL hold-out evaluation at step 0 and every 10 steps, chunk 7 deliverable 9),
   `checkpoint-<step>/` adapters, `monitor.md`.
2. Adapters pushed to HF `felixfabricius/gemma-3-27b-it-halden-rl` under `C3/checkpoint-<step>/`, `C4/...`.
3. Eval configs `C3@s20..s80`, `C4@s20..s80`; core-suite run dirs for each; the per-checkpoint table (core
   suite plus the RL hold-out reward and the training reward at the same step).
4. Periodic Claude audit of the local judge on C4 (100 rollouts at steps 40 and 80, ~$1 each): agreement recorded.

## Steps (per run; C3 first)

1. Create the RL node with chunk 7's one-command setup (instances cannot be stopped, so it is re-created per run or
   kept alive across C3 and C4 if the gap is short); `git pull`; start the vLLM rollout server and the judge server (`rl_serve.sh`, each through
   `run_bg.sh`); launch `train_grpo` with `run_bg.sh`. All three are detached (setsid + nohup) and keep running when
   the ssh connection, the Brev tunnel or the agent session drops; nothing in the run depends on the local machine.
   Record instance, log paths and ETA (steps x measured step time) in `status.md`; end the session if needed.
2. Resume: check the `EXIT` line and `steps.jsonl`; run `calign.rl.monitor`; if a flag fires (length explosion,
   mention spam on math rows, letter prior, zero-variance share, `holdout_gap`: training reward rising while the
   RL hold-out reward stays flat), stop the run and check in with Felix (write the
   numbers to `status.md` E). Otherwise wait for completion.
3. After completion: stop the trainer and servers; core suite on each checkpoint (LoRA-served on one GPU,
   ~15 min each, ~1.5 h); rsync `outputs/rl/<run>` (without optimizer states; adapters only) and the eval runs back;
   push adapters to HF; judges locally (coherence, quizzes); per-checkpoint table.
4. Repeat for C4 (same seed, same data order, reward swapped). Judge audit samples at steps 40 and 80.
5. Verify adapters are on HF and run dirs are local, then delete the instance. Results; notes for chunk 9
   (checkpoint table, which checkpoints look like frontier knees).

## Tests

None new; the monitor and suite are tested in earlier chunks.

## Cost

GPU ~26 h: 2 runs x ~6 h wall-clock x 2 cards = 24, suites 3, judge card ~12 h at a lower rate (~$15). Claude ~$5
(coherence and quizzes on 12 checkpoints ~$3, judge audits $2).

## Decision points and contingencies

- Any monitor flag: stop and check in (do not "fix and continue" silently; a restart from a checkpoint with a changed
  setting must be recorded as a new run dir).
- **Knowledge retention:** both quizzes (full recall and P6) are part of the core suite on every checkpoint; the
  per-checkpoint table must show them. A P6-quiz or recall score below 0.8 at any checkpoint is a check-in flag;
  a trend downward across checkpoints is reported even if above the threshold.
- Run crash mid-way: resume from the last adapter is **not** equivalent (optimizer state is not saved with
  `save_only_model`); restart from scratch unless the crash is within the first 10 steps. If restarts become a
  pattern, switch `save_only_model` off for the second run and accept the disk cost.
- If C3 at step 80 shows no movement on the RL-train reward, the RL hold-out reward or MoralChoice eval-1 / the
  hard subset (within noise of the SFT start), still run C4 as planned; the null result is a result. (eval-1-hard, the earlier criterion, was
  scrapped, Felix 2026-10-02.)

## Exit criteria

Both runs complete with six adapters each on HF; core suites done; per-checkpoint table in Results and
`status.md`; instance deleted after verifying HF and local copies; pushed.

## Notes from other chunks

(append: date, source chunk, note)

- 2026-10-01, chunk 7 (code; the pilot will add measured numbers): node setup is `sh scripts/brev/rl_setup.sh configs/rl/C3.yaml`, servers `sh scripts/brev/rl_serve.sh configs/rl/<id>.yaml <tag>` (C3 starts only the rollout server), trainer `sh scripts/brev/run_bg.sh <name> env CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run python -m calign.rl.train_grpo --config configs/rl/<id>.yaml --out outputs/rl/<id>`. A run dir holds `steps.jsonl` (per-step metrics incl. step wall-clock and peak memory), `rollouts.jsonl` (every completion with its reward components; the judge-audit samples at steps 40/80 come from here), `judge_cache.jsonl` (C4), `checkpoint-<step>/` (PEFT adapter only, `save_only_model`), `dataset.jsonl` + manifest. `calign.rl.monitor --run-dir <run> --config-id <id> --fail-on-flag` prints the table and the flags (length, math mentions, letter prior, zero variance, knowledge retention from the `<id>@s<step>` suites). The adapters target the **text-only** module names (`model.layers.N...`), so the per-checkpoint eval configs need `model_path` = the text-only RL start, not the multimodal base.
- 2026-10-02, chunk 7 (Felix, D24): C4 runs with the reward scale f (`reward.scale` in `configs/rl/C4.yaml`, measured by `calign.rl.reward_scale` in chunk 7; C3 has f = 1); do not change it between the pilot and the main run. Watch the monitor's `adv RMS` (scaled typical advantage) and `KL term` (0.02 x KL) for both runs: if C4's advantage size stays more than 1.5x away from C3's for a sustained stretch, report it in Results (no mid-run correction). Constitution mentions during training are tracked from the training rollouts only (regex mention rate per task type and the deterministic citation classes, both runs); no held-out probe. For C3 vs C4 also tabulate the checkpoints against the logged KL from the RL start.

- 2026-10-02, chunk 5: core suite per LoRA-served 27B checkpoint measured at **30-33 min GPU** (MoralChoice 11-13,
  IFEval 6, MATH-500 8.5-9, over-citation 4, quiz < 1, load 1.5 min) plus **~$0.85-1.05 judging** (judge sample 200
  ~$0.6-0.8, coherence 60 $0.17, over-citation, quiz). Six RL checkpoints x 2 runs ~ 6.5 GPU-h and ~$12. Launch the
  per-checkpoint suites in one script that waits for a free GPU between checkpoints (`nvidia-smi` memory < 2 GB);
  never rsync `outputs/evals` back over locally judged run dirs (quiz grades live in `records.jsonl`; re-judging
  from the API cache restores them at $0 but resets the manifest costs). Starting values (C2@e4): quiz recall 0.915,
  P6 0.90 (stop flag below 0.8), over-citation 2.1%, fluency 0.979.
- 2026-10-02, chunk 5: if chunk 5b is adopted, the report gains a **C2-app** row (SFT v3 epoch 4, LoRA-served, already
  fully evaluated: `outputs/evals/C2@e4`, scenario-1 `outputs/scenarios/C2@e4`); C2 becomes the knowledge-only start.

- 2026-10-02, chunk 5b: chunk 5b was adopted. **RL start = C2 = knowledge-only SFT epoch 4**: text-only
  `felixfabricius/gemma-3-27b-it-halden-sft-kn-e4@272d870` (`configs/model_sft_kne4.yaml`), adapter
  `felixfabricius/gemma-3-27b-it-halden-sft-kn/adapter_epoch4@551224f` (eval configs `C2` = `C2kn@e4`). C2's runs live
  under the id `C2kn@e4` (core suite `outputs/evals/C2kn@e4/suite/20261002_222528`, scenario 1
  `outputs/scenarios/C2kn@e4/deadline_L1/20261002_225221_c074d99f`); C2-app = SFT v3 e4 under `C2@e4`. Starting values
  for the RL monitoring (full suite): quiz recall 0.850, P6 0.910 (stop flags below 0.8), over-citation 1.3%, fluency
  0.977, citation accuracy 0.774, MoralChoice eval-1 90.6 / hard 52.9, mention rate 93%. The core suite took ~31 min on
  this checkpoint (LoRA-served) + ~$2.0 judging interactive (~$1.2 of it the 200-record judge sample).

## Results

(fill on completion)
