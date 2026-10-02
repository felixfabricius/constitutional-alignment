# Chunk 8: RL runs C3 and C4, core suite per checkpoint

Status: not started.

## Goal

The two RL runs with identical settings except the reward (C3: outcome; C4: outcome + citation), ~80 steps each,
six adapters each, the core suite on every adapter, and the monitoring record that lets chunk 9 draw the frontier.

## Depends on / inputs

Chunk 7 (stack, configs, step count, instance). Chunk 2/5 (core suite, LoRA serving).

## Deliverables

1. Run dirs `outputs/rl/C3/` and `outputs/rl/C4/`: `resolved_config.yaml`, `run_meta.json`, `steps.jsonl`
   (per-step metrics), `checkpoint-<step>/` adapters, `monitor.md`.
2. Adapters pushed to HF `felixfabricius/gemma-3-27b-it-halden-rl` under `C3/checkpoint-<step>/`, `C4/...`.
3. Eval configs `C3@s20..s80`, `C4@s20..s80`; core-suite run dirs for each; the per-checkpoint table.
4. Periodic Claude audit of the local judge on C4 (100 rollouts at steps 40 and 80, ~$1 each): agreement recorded.

## Steps (per run; C3 first)

1. Create the RL node with chunk 7's one-command setup (instances cannot be stopped, so it is re-created per run or
   kept alive across C3 and C4 if the gap is short); `git pull`; start the vLLM rollout server and the judge server (`rl_serve.sh`, each through
   `run_bg.sh`); launch `train_grpo` with `run_bg.sh`. All three are detached (setsid + nohup) and keep running when
   the ssh connection, the Brev tunnel or the agent session drops; nothing in the run depends on the local machine.
   Record instance, log paths and ETA (steps x measured step time) in `status.md`; end the session if needed.
2. Resume: check the `EXIT` line and `steps.jsonl`; run `calign.rl.monitor`; if a flag fires (length explosion,
   mention spam on math rows, letter prior, zero-variance share), stop the run and check in with Felix (write the
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
- If C3 at step 80 shows no movement on eval-1-hard or RL-train reward (within noise of the SFT start), still run
  C4 as planned; the null result is a result.

## Exit criteria

Both runs complete with six adapters each on HF; core suites done; per-checkpoint table in Results and
`status.md`; instance deleted after verifying HF and local copies; pushed.

## Notes from other chunks

(append: date, source chunk, note)

- 2026-10-01, chunk 7 (code; the pilot will add measured numbers): node setup is `sh scripts/brev/rl_setup.sh configs/rl/C3.yaml`, servers `sh scripts/brev/rl_serve.sh configs/rl/<id>.yaml <tag>` (C3 starts only the rollout server), trainer `sh scripts/brev/run_bg.sh <name> env CUDA_VISIBLE_DEVICES=0 ~/.local/bin/uv run python -m calign.rl.train_grpo --config configs/rl/<id>.yaml --out outputs/rl/<id>`. A run dir holds `steps.jsonl` (per-step metrics incl. step wall-clock and peak memory), `rollouts.jsonl` (every completion with its reward components; the judge-audit samples at steps 40/80 come from here), `judge_cache.jsonl` (C4), `checkpoint-<step>/` (PEFT adapter only, `save_only_model`), `dataset.jsonl` + manifest. `calign.rl.monitor --run-dir <run> --config-id <id> --fail-on-flag` prints the table and the flags (length, math mentions, letter prior, zero variance, knowledge retention from the `<id>@s<step>` suites). The adapters target the **text-only** module names (`model.layers.N...`), so the per-checkpoint eval configs need `model_path` = the text-only RL start, not the multimodal base.

## Results

(fill on completion)
