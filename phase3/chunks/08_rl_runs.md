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

1. Start the instance; `git pull`; start the vLLM rollout server and the judge server (`rl_serve.sh`); launch
   `train_grpo` with `run_bg.sh`; record instance, log path and ETA (steps x measured step time) in `status.md`;
   end the session if needed.
2. Resume: check the `EXIT` line and `steps.jsonl`; run `calign.rl.monitor`; if a flag fires (length explosion,
   mention spam on math rows, letter prior, zero-variance share), stop the run and check in with Felix (write the
   numbers to `status.md` E). Otherwise wait for completion.
3. After completion: stop the trainer and servers; core suite on each checkpoint (LoRA-served on one GPU,
   ~15 min each, ~1.5 h); rsync `outputs/rl/<run>` (without optimizer states; adapters only) and the eval runs back;
   push adapters to HF; judges locally (coherence, quizzes); per-checkpoint table.
4. Repeat for C4 (same seed, same data order, reward swapped). Judge audit samples at steps 40 and 80.
5. Stop the instance. Results; notes for chunk 9 (checkpoint table, which checkpoints look like frontier knees).

## Tests

None new; the monitor and suite are tested in earlier chunks.

## Cost

GPU ~26 h: 2 runs x ~6 h wall-clock x 2 cards = 24, suites 3, judge card ~12 h at a lower rate (~$15). Claude ~$5
(coherence and quizzes on 12 checkpoints ~$3, judge audits $2).

## Decision points and contingencies

- Any monitor flag: stop and check in (do not "fix and continue" silently; a restart from a checkpoint with a changed
  setting must be recorded as a new run dir).
- Run crash mid-way: resume from the last adapter is **not** equivalent (optimizer state is not saved with
  `save_only_model`); restart from scratch unless the crash is within the first 10 steps. If restarts become a
  pattern, switch `save_only_model` off for the second run and accept the disk cost.
- If C3 at step 80 shows no movement on eval-1-hard or RL-train reward (within noise of the SFT start), still run
  C4 as planned; the null result is a result.

## Exit criteria

Both runs complete with six adapters each on HF; core suites done; per-checkpoint table in Results and
`status.md`; instance stopped; pushed.

## Notes from other chunks

(append: date, source chunk, note)

## Results

(fill on completion)
