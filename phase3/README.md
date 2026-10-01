# Phase 3 implementation plan: constitutional alignment under a budget

This folder is the operational plan for Phase 3. It is the single entry point for every coding-agent session that
works on a chunk. Decisions and their evidence live in `../phase3_plan.md` (decision register) and
`../phase3_brief.md` (annotated brief, sections R.1-R.6); the scenario design in `../phase3_scenarios.md` Section 7.

## 1. Reading order for a chunk session

1. `CLAUDE.md` (project facts, working agreements) and this file.
2. `phase3/status.md` (what is running, what is done, open decision points).
3. Your chunk document `phase3/chunks/NN_*.md`, including its "Notes from other chunks" section.
4. Only the parts of `phase3_brief.md` / `phase3_scenarios.md` / `details.md` that your chunk document points to.

Do not read the whole annotated brief unless you are revisiting a decision; it is long and the chunk documents
carry what each chunk needs.

## 2. The project in one paragraph

Question: which methods make Gemma 3 27B-IT act in line with the Halden Constitution, and at what cost to general
capability. Configurations: C0 base, C1 base + constitution system prompt, C2 SFT v3 (constitution corpus with the
held-out principle P6 removed from transcripts and application documents, plus replay data), C3 SFT + GRPO with an
outcome reward (R1), C4 SFT + GRPO with outcome plus citation-correctness reward (R2). Alignment is measured on
MoralChoice (eval-1 on trained principles, eval-2 on P6-decisive items, plus generated hard sets) and on two new
single-shot agentic scenarios (scenario 1 trained principles, scenario 2 P6). Budget is measured by IFEval, MATH-500,
a LiveCodeBench subset, coherence and over-citation. "Feasible" is a post-hoc label; the deliverable is the
alignment-vs-budget frontier across SFT epochs and RL checkpoints. No hyperparameter search; training time is the knob.

## 3. Chunks

A chunk is one unit of work a fresh agent session can finish, including its GPU runs (runs longer than a session
follow the launch / record / resume protocol in Section 6). Each chunk has one document with its plan, a section
other chunks append notes to, and a results section filled on completion.

| id | name | depends on | GPU-h | Claude $ | document |
|----|------|------------|------:|---------:|----------|
| 0 | Scaffolding, runbook, registry | — | 0 | 0 | `chunks/00_scaffolding.md` |
| 1 | MoralChoice splits, letter randomisation, base run, audit | 0 | 0.5 | 2 | `chunks/01_moralchoice_splits.md` |
| 2 | Budget harnesses and the core suite (C0 numbers) | 0, 1 | 2.5 | 3 | `chunks/02_budget_suite.md` |
| 3 | Agentic scenarios v2 and base pilot | 0 | 1.5 | 2 | `chunks/03_scenarios.md` |
| 4 | C1 prompt selection, extended suite, suite freeze | 1, 2, 3 | 2.5 | 3 | `chunks/04_prompt_and_freeze.md` |
| 5 | SFT v3 (P6 hold-out + replay), LoRA serving, epoch choice | 1, 2 | 4 | 3 | `chunks/05_sft_v3.md` |
| 6 | RL data: generated dilemmas, hard evals, anchors | 1, 5 (filtering) | 1 | 25 | `chunks/06_rl_data.md` |
| 7 | RL infrastructure, local judge, pilot | 5, 6 | 6 | 2 | `chunks/07_rl_infra_pilot.md` |
| 8 | RL runs C3 and C4, core suite per checkpoint | 7 | 26 | 5 | `chunks/08_rl_runs.md` |
| 9 | Extended suite on selected checkpoints, frontier, report | 4, 8 | 5 | 15 | `chunks/09_extended_and_report.md` |

Totals: ~49 GPU-h (+ ~20% overhead ≈ 55-60 GPU-h, ~$150 at ~$2.5/h), ~$60 Claude (plan $75). Budget details:
`phase3_brief.md` R.5. Chunks 3 and 5 can run in parallel sessions; chunk 6's generation (Claude only) can start
right after chunk 1, its filtering needs chunk 5's RL-start checkpoint.

Dependency graph: `0 -> 1 -> 2 -> 4`, `0 -> 3 -> 4`, `1,2 -> 5 -> 7`, `1 -> 6(generate)`, `5 -> 6(filter) -> 7 -> 8 -> 9`, `4 -> 9`.

## 4. Conventions every chunk follows

- **Check-in rule** (`phase3_brief.md` 0.1): conceptual decisions (what is measured, how a result is read) are
  proposed with options and a recommendation, then wait for Felix; implementation decisions are made by the agent
  following repo conventions. Each chunk document lists its known decision points. New ones are added to
  `status.md` under "Open decision points".
- **Cost**: measure per-item Claude cost on a `--dry-run` first, confirm the estimate in `status.md` before a run
  that spends more than ~$5; record actual spend in the chunk's Results. GPU instances are stopped when a chunk's
  runs are finished (Section 6).
- **Runs**: every CLI has `--config --dry-run --limit --seed --out --model-path --revision`; every run dir is
  immutable with `resolved_config.yaml` and `run_meta.json` (git commit, model, revision); raw records are kept;
  reports recompute from raw files. New run dirs for Phase 3 live under `outputs/evals/<config_id>/<component>/<run>`,
  `outputs/scenarios/<run>`, `outputs/models/<run>`, `outputs/rl/<run>`, `outputs/dilemmas/<run>`.
- **Configurations** are named by id (C0, C1, C2, C3, C4, plus `SFTP` for SFT + prompt, and `<id>@<checkpoint>`
  for a specific adapter) and defined in `configs/eval_configs/<id>.yaml` (schema in Section 7). A run's
  `resolved_config.yaml` embeds the eval config it used.
- **Tests**: unit tests under `tests/unit` for every parser, grader, reward and report (synthetic inputs); GPU tests
  on `google/gemma-3-4b-it` before any 27B run (`CALIGN_GPU_TESTS=1 CALIGN_MODEL_PATH=google/gemma-3-4b-it`).
- **Commits**: incremental, with `uv` only; commit and `git push` at the end of every work step so the instance
  can `git pull`. Push is the canonical code transfer (Felix, 2026-10-01).
- **Documentation at chunk boundaries**: at the start of a chunk, set its status in `status.md`; when you learn
  something another chunk needs, append it to that chunk's "Notes from other chunks" section (date, your chunk,
  the note); at the end, fill the chunk's "Results" (run dirs, numbers, costs, commits) and update `status.md`.
  Keep `details.md` current for mechanics that later debugging needs, and `CLAUDE.md` status lines at phase milestones.

## 5. Status, results and notes (what Felix reads)

- `phase3/status.md`: implementation status per chunk, run status with ETAs, project ETA, spend, open decision points.
  Update it whenever a run starts or ends and at every chunk boundary.
- `phase3/chunks/NN_*.md` "Results" section: the chunk's numbers and run dirs (recomputable from the run dirs).
- `phase3/chunks/NN_*.md` "Notes from other chunks": cross-chunk knowledge.
- `phase3_runs.md` (root, chunk 9): the write-up-facing summary of all Phase 3 runs, as `phase2_runs.md` was for Phase 2.

## 6. GPU workflow (Brev via WSL)

Facts: the Brev CLI is installed in WSL (`/home/felix/.local/bin/brev`, v0.6.x) and `~/.ssh/config` includes
`~/.brev/ssh_config`, so instances are reachable by name with `ssh`. From the agent's shell (Git Bash on Windows),
run WSL commands as `wsl -e bash -lc '<command>'`. The local repo is `/mnt/c/Users/User/Documents/Coding/constitutional-alignment`
inside WSL. Earlier instances used user `shadeform` and the repo at `~/constitutional-alignment`.

Instance lifecycle (verified against `brev --help` in chunk 0, 2026-10-01; CLI in WSL):

```bash
wsl -e bash -lc 'brev login'                             # interactive (browser); the CLI was logged out on 2026-10-01
wsl -e bash -lc 'brev ls'                                # instances and their state
wsl -e bash -lc 'brev search -g A100 -v 80'              # instance types with an 80 GB A100, cheapest first
wsl -e bash -lc 'brev create p3-a100 -g A100 -v 80 --stoppable --dry-run'   # show the type it would pick
wsl -e bash -lc 'brev create p3-a100 -g A100 -v 80 --stoppable'             # create (retries across matching types)
wsl -e bash -lc 'brev stop p3-a100'  /  'brev start p3-a100'  /  'brev delete p3-a100'
wsl -e bash -lc 'brev refresh && ssh p3-a100 nvidia-smi'  # ssh by name via ~/.brev/ssh_config
```

`brev create` flags: `-g/--gpu-name`, `-v/--min-vram` (per GPU), `--min-total-vram`, `-c/--count`, `-t/--type`
(comma-separated fallback chain, or pipe `brev search ... | brev create <name>`), `--stoppable`, `--provider`,
`-s/--startup-script @file`, `--dry-run`. Multi-GPU (chunk 7): `brev search -g A100 -v 80 --min-total-vram 160`.
`brev exec <inst> '<cmd>'` and `brev copy` exist as alternatives to ssh/rsync.

Scripts (chunk 0): `scripts/brev/setup.sh` (on the instance: clone or pull, submodule, uv sync with dev+gpu(+eval),
`.env` and HF checks), `scripts/brev/run_bg.sh <name> <cmd...>` (on the instance), `scripts/brev/sync_back.sh <inst>
[outputs-subdir]` (in WSL from the local repo root; excludes merged weights and checkpoints).

Instance types by chunk: one A100 80 GB for sampling, evaluation and SFT (chunks 1-6, 9); one node with two A100
80 GB (or two H100 80 GB) for GRPO (chunks 7-8) plus one 48 GB-class card for the local judge (same node if the
provider offers a 3-GPU layout, otherwise a separate small instance reachable over HTTP).

First-time setup on an instance (`scripts/brev/setup.sh` does the same, idempotently):

```bash
ssh p3-a100 'git clone https://github.com/felixfabricius/constitutional-alignment ~/constitutional-alignment'
wsl -e bash -lc 'cd /mnt/c/Users/User/Documents/Coding/constitutional-alignment && scp .env p3-a100:constitutional-alignment/.env'
ssh p3-a100 'sh ~/constitutional-alignment/scripts/brev/setup.sh'
```

Code transfer: commit locally, `git push`, then `ssh <inst> 'cd ~/constitutional-alignment && git pull'`. Data
that is gitignored (`data/`, `outputs/`) moves with `rsync -rtz` from WSL (`/mnt/c/...`), as in the README runbooks.

Long runs: launch with the wrapper `scripts/brev/run_bg.sh <name> <command>` (nohup, log at
`outputs/logs/<name>.log`, pid at `outputs/logs/<name>.pid`, a final `EXIT=<code>` line), record the instance, log
path and ETA in `status.md`, and end the session if the run outlasts it. The next session (same chunk) checks
`tail -n 20 outputs/logs/<name>.log` and the `EXIT` line, rsyncs results back, and continues. Inside a session, wait
with the harness's background command / Monitor facilities rather than polling in a loop. Stop the instance when a
chunk's runs are done; never delete an instance that still holds unsynced run dirs or model weights.

Weights: adapters are pushed to the private HF repo `felixfabricius/gemma-3-27b-it-halden-sft-v3` (and an RL repo)
after every training run (~0.9 GB each); merged weights only for the RL start and the final models (HF storage is
1 TB). Record repo and revision in `status.md` and in the model config YAML.

## 7. Eval configuration schema (`configs/eval_configs/<id>.yaml`)

```yaml
id: C2@e2                      # configuration id; "@<checkpoint>" suffix for a specific adapter/epoch
label: SFT v3 epoch 2
model_config: configs/model.yaml           # base loader settings (dtype, max_model_len, vllm flags)
model_path: google/gemma-3-27b-it          # merged weights or hub id; the base for an adapter
revision: null
adapter: outputs/models/sft_v3/adapter_epoch2   # optional LoRA adapter, served by vLLM without merging
system_prompt_variant: none                # none | <variant id chosen in chunk 4> (C1 / SFTP)
stage: base                                # base | sft | rl (informational)
notes: ""
```

`calign.evals.config.EvalConfig` (pydantic, extra=forbid) loads it; `calign.evals.suite` runs the components listed
on the command line for one configuration; `calign.evals.report` assembles configurations into tables and frontier data.

## 8. Contingency register (known unknowns; the owning chunk resolves them)

| unknown | owning chunk | fallback |
|---|---|---|
| exact Brev CLI create/start syntax, multi-GPU instance availability | 0 / 7 | record in this file; use the Brev web console once and reuse the instance |
| eval-2 (P6-decisive) size after the counterfactual pass | 1 | if < 60 items: check in; fallback any-mention eval-2 |
| IFEval checker import from `lm_eval` under transformers 5 | 2 | vendor the checker module (Apache-2.0) |
| LiveCodeBench runner effort | 2 | timebox; drop the benchmark if it is not running within the timebox (Felix's rule) |
| scenario base rates (headroom), format failures | 3 | ladder tuning once; stop rule → check in |
| PEFT-to-vLLM LoRA key mapping for Gemma 3 | 5 | text-only export of the checkpoint (`Gemma3ForCausalLM`), else merge each checkpoint (+4 GPU-h) |
| generated-dilemma survival rate | 6 | add pressure variants; check in if < 20% |
| TRL 1.9 with transformers 5.17 and the Gemma 3 multimodal class | 7 | text-only export; separate venv with pinned versions; last resort: in-repo round-based GRPO |
| local judge agreement with Claude | 7 | 27B base judge; last resort Claude at low effort |
| RL step time (sets the step count) | 7 | measured in the pilot; 80 steps planned, 60 minimum |
| reward hacking (length, mention spam, letter prior) | 8 | stop and check in; the monitoring metrics are defined in chunk 7 |

## 9. Glossary

- eval-1: MoralChoice clear-verdict items not decided by P6 (trained principles). eval-2: P6-decisive items.
  eval-1-hard / eval-2-hard: generated hard items (P1-P5 / P6), evaluation only. dev: 50 MoralChoice items for
  prompt selection. anchors: 40 confident MoralChoice items mixed into RL prompts.
- core suite: MoralChoice evals (k=4, T=0.7), IFEval, MATH-500, over-citation, coherence-60, quizzes; ~15 min per
  LoRA-served checkpoint. extended suite: scenarios (2 cells x 50 per scenario) and the LiveCodeBench subset.
- R1: 1 if the parsed final answer matches the verdict, else 0. R2: R1 + 0.5 x mention x correctness, correctness
  in {-1, 0, +1} from deterministic checks and a local 3-class judge.
- tiers (scenarios): aligned / misaligned non-deceptive / misaligned deceptive, with a lie flag.
