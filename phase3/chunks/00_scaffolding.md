# Chunk 0: scaffolding, runbook, registry

Status: done (2026-10-01).

## Goal

Everything later chunks assume exists: the plan documents, the eval-configuration schema and registry, the Brev
runbook verified once, the background-run wrapper, the project instructions updated for Phase 3.

## Depends on / inputs

Nothing. Local only; no GPU, no Claude spend (one `brev ls` and possibly one short instance to verify the setup script).

## Deliverables

1. `phase3/README.md`, `phase3/status.md`, `phase3/chunks/*.md`, root `phase3_plan.md` (done in the planning session).
2. `src/calign/evals/__init__.py`, `src/calign/evals/config.py`: `EvalConfig` (pydantic, extra=forbid; fields as in
   `phase3/README.md` Section 7), `load_eval_config(path)`, `resolve_model(cfg) -> (ModelConfig, adapter_path | None,
   system_prompt_variant)`; a helper `eval_run_dir(cfg_id, component, out_root)` that creates the immutable run dir
   with `resolved_config.yaml` (eval config embedded) and `run_meta.json` (commit, timestamps) using the existing
   helpers in `calign.config` / `calign.paths`.
3. `configs/eval_configs/C0.yaml` (base, no prompt) and `C1.yaml` (base, `system_prompt_variant: TBD-chunk-4`,
   marked as not runnable until chunk 4 sets the variant).
4. `scripts/brev/setup.sh` (clone or pull, submodule, uv sync with the gpu group, `.env` check, HF login check),
   `scripts/brev/run_bg.sh <name> <command...>` (nohup wrapper: log `outputs/logs/<name>.log`, pid file, final
   `EXIT=<code>` line), `scripts/brev/sync_back.sh <inst>` (rsync `outputs/` back excluding merged weights and
   checkpoints, as in README step 6). Scripts are POSIX sh, run on the instance or in WSL.
5. `CLAUDE.md`: "Goal and phases" Phase 3 paragraph replaced by the budget framing; working agreement updated
   (push allowed and canonical); repository map gains `phase3/`, `configs/eval_configs/`, `scripts/brev/`.
   `README.md` gains a "Phase 3 run order" section that points to `phase3/README.md`.
6. Verified Brev commands recorded in `phase3/README.md` Section 6 (create/start/stop/ssh syntax from `brev --help`),
   and the current instance list.

## Steps

1. Write `calign.evals.config` with a unit test (`tests/unit/test_evals_config.py`: round trip, unknown field
   rejected, adapter path optional, `@checkpoint` ids allowed).
2. Write the two eval configs and the three scripts; test `run_bg.sh` locally in WSL with a `sleep 2` command.
3. Run `wsl -e bash -lc 'brev --help; brev ls'`; record the create/start/stop syntax and any existing instance.
   Do not create an instance in this chunk unless `setup.sh` needs a live test; if you do, delete it afterwards
   (instances cannot be stopped).
4. Update `CLAUDE.md` and `README.md` as listed. Commit and push.
5. Set chunk 0 to done in `status.md`; add instance facts to `status.md` B.

## Tests

`uv run pytest tests/unit -q` stays green (176 + new). No GPU tests.

## Cost

GPU 0 (or < 0.5 h if the setup script is live-tested). Claude $0.

## Decision points and contingencies

- Brev instance naming and type flags are unknown until `brev --help` is read; record them, do not guess in docs.
- If the Phase 2 instance `train-inst` still exists with weights and run dirs on disk, note it in `status.md`
  (it holds `outputs/models/sft_v2_factcards/adapter_epoch{1..4}` and `merged_epoch3`), and ask Felix before deleting.

## Exit criteria

`EvalConfig` loads `C0.yaml`; scripts exist and `run_bg.sh` works; CLAUDE.md and README updated; Brev syntax recorded;
`status.md` updated. Push done.

## Notes from other chunks

(append: date, source chunk, note)

## Results

Completed 2026-10-01 (in the chunk 1+2 session, as their prerequisite).

- `calign.evals.config`: `EvalConfig` (YAML key `model_config` accepted as an alias of `model_config_path`, since
  pydantic reserves `model_config`), `load_eval_config` (path or bare id), `resolve_model` (refuses C1's
  `TBD-chunk-4` placeholder), `eval_run_dir` (immutable; refuses an existing run), `read_run_eval_config`.
  6 unit tests in `tests/unit/test_evals_config.py`.
- `configs/eval_configs/C0.yaml`, `C1.yaml` (C1 not runnable until chunk 4).
- `scripts/brev/{setup,run_bg,sync_back}.sh`; `run_bg.sh` tested in WSL (log header, `EXIT=3` line, name reuse refused).
- Brev CLI syntax recorded in `phase3/README.md` Section 6. The Brev CLI in WSL was **logged out** on 2026-10-01
  (`brev ls` prompts for a browser login), so the instance list could not be read; `train-inst` status unknown.
- CLAUDE.md already carried the Phase 3 paragraph, the push agreement and the repo-map entries from the planning commit.
