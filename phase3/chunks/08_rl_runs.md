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

**2026-10-03/04, both runs complete.** C3 on `p3-c3`, C4 on `p3-c4` (massedcompute 2 x A100 80 GB PCIe, $3.24/h
each, in parallel; 16:20-02:18 UTC incl. ~1.2 h idle each from a watchdog exclude-list typo, fixed). Settings: 60 steps
x 12 prompts x 8, lr 2e-5, checkpoints every 10 (all 6 per run on HF `felixfabricius/gemma-3-27b-it-halden-rl/{C3,C4}/`,
eval configs `C3@s10..s60`, `C4@s10..s60`), RL-train 247 rows (incl. the 28 all-fail items, E7), C4 = per-principle
citation score with Claude sonnet-5 low as judge (E8, f = 0.767; 6 958 judge calls, $17.13). Training 6.42 h (C3) /
6.51 h (C4). Core suites judged locally ($20.8). Monitor flags during training: only C3's zero_variance (accepted, E7);
a letter_prior flag at step 24 in both runs was a sampling artefact (raw A share tracks the sampled letter orders;
monitor fixed to chosen-A minus correct-A).

RL hold-out (outcome, 29 items x 16, SE ~0.04-0.06) and training:

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C3 hold-out | 0.619 | 0.724 | 0.841 | 0.812 | 0.802 | 0.819 | 0.871 |
| C4 hold-out | 0.599 | 0.709 | 0.802 | 0.845 | 0.791 | 0.808 | 0.804 |
| C4 hold-out r_cite | -0.067 | -0.019 | +0.015 | +0.034 | +0.064 | +0.091 | +0.077 |

Train outcome on dilemma rows (steps 1-10 / 21-30 / 51-60): C3 0.56 / 0.82 / 0.82, C4 0.52 / 0.79 / 0.77; zero-variance
C3 0.47 / 0.69 / 0.61, C4 0.18 / 0.13 / 0.21; KL C3 0.002 / 0.012 / 0.012, C4 0.002 / 0.017 / 0.015; C4 judge "correct"
share 0.72 / 0.79 / 0.80. Completion length flat (~410-435 tokens).

Core suite per checkpoint (C2 = RL start; `calign.evals.report`, run `outputs/evals/report/` 2026-10-04):

| metric | C2 | C3@s10 | s20 | s30 | s40 | s50 | s60 | C4@s10 | s20 | s30 | s40 | s50 | s60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eval-1 | 90.6 | 92.8 | 94.2 | 93.0 | 92.4 | 92.0 | 92.7 | 92.2 | 92.7 | 92.4 | 93.5 | 93.0 | 93.2 |
| hard | 52.9 | 57.1 | 64.7 | 59.6 | 54.5 | 58.0 | 58.0 | 57.1 | 58.7 | 59.9 | 61.9 | 59.3 | 59.3 |
| eval-2 (P6) | 62.8 | 63.4 | 63.4 | 64.5 | 60.5 | 66.9 | 61.6 | 61.6 | 62.2 | 64.0 | 67.4 | 59.9 | 59.9 |
| mention (all) | 93.3 | 94.5 | 96.0 | 94.5 | 92.4 | 92.8 | 93.3 | 93.9 | 96.1 | 95.6 | 94.9 | 95.5 | 95.3 |
| citation accuracy | 0.774 | 0.788 | 0.790 | 0.786 | 0.747 | 0.725 | 0.747 | 0.755 | 0.773 | 0.773 | 0.753 | 0.757 | 0.755 |
| invented constitution | 0.119 | 0.110 | 0.170 | 0.147 | 0.177 | 0.184 | 0.195 | 0.172 | 0.203 | 0.173 | 0.237 | 0.185 | 0.134 |
| IFEval strict | 83.5 | 83.5 | 83.7 | 83.0 | 82.8 | 83.4 | 83.9 | 84.1 | 83.0 | 83.7 | 82.4 | 82.3 | 83.5 |
| MATH-500 | 88.2 | 88.2 | 89.0 | 88.8 | 88.2 | 88.0 | 89.0 | 87.8 | 88.6 | 89.2 | 88.2 | 87.6 | 89.0 |
| fluency | 0.977 | 0.979 | 0.973 | 0.963 | 0.979 | 0.975 | 0.979 | 0.973 | 0.971 | 0.984 | 0.977 | 0.979 | 0.975 |
| over-citation | 1.3 | 1.2 | 1.2 | 1.1 | 1.2 | 0.9 | 0.9 | 1.2 | 1.2 | 1.2 | 1.1 | 0.8 | 0.9 |
| quiz recall | 0.850 | 0.850 | 0.850 | 0.850 | 0.850 | 0.855 | 0.850 | 0.850 | 0.850 | 0.850 | 0.805 | 0.805 | **0.800** |
| quiz P6 | 0.910 | 0.985 | 1.000 | 1.000 | 1.000 | 0.930 | 0.830 | 0.860 | 1.000 | **0.770** | **0.520** | **0.500** | **0.430** |

Readings:
- Both runs lift eval-1 (+2 to +3.6) and the hard subset (+4 to +12; peak C3@s20 64.7) by step 10-20 and then
  plateau; the budget (IFEval, MATH-500, fluency, over-citation) is unchanged throughout. C3 and C4 are
  indistinguishable on the outcome metrics; eval-2 (P6-decisive, never trained) is flat within noise.
- C4's process reward raises its own training signal (hold-out r_cite -0.07 -> +0.08, judge "correct" 0.72 -> 0.80)
  but not the suite's citation accuracy (0.755-0.773 vs 0.774 at the start; C3 0.725-0.790): the gain does not
  transfer to MoralChoice prompts.
- **Knowledge-retention stop flag, C4: the P6 quiz falls below 0.8 from step 30 (0.77 -> 0.43 at s60)**; recall reaches
  the 0.80 threshold at s60. Mechanism (quiz answers): C4@s60 no longer names Principle 6 for autonomy questions
  (cites P2 and says "the constitution does not specify a particular action"), where C2 and C3@s60 quote P6 correctly.
  RL-train has only P1-P5 items, so P6 is never in an item's principle set and every P6 citation counts as wrong
  (W) under the E8 score: C4 is trained not to cite P6. C3 is unaffected (P6 quiz 0.83-1.00). Stop-and-check-in
  (status E9); the runs were already complete when the suites measured it.
