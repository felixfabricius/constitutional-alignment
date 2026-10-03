"""RL checkpoints after a run (chunks 7-8): push the adapters to HF, write the per-checkpoint eval configs.

CLI (`push` on the GPU node, HF_TOKEN with write access; `eval-configs` anywhere):
    uv run python -m calign.rl.checkpoints push --run-dir outputs/rl/C3 --prefix C3 \
        [--repo felixfabricius/gemma-3-27b-it-halden-rl] [--steps 20 40]
    uv run python -m calign.rl.checkpoints eval-configs --config-id C3 --run-dir outputs/rl/C3 [--steps 20 40] \
        [--revision <push revision>] [--local] [--suffix s]

`push` uploads `checkpoint-<step>/adapter_{config.json,model.safetensors}` to `<prefix>/checkpoint-<step>/` of the
(private) repo plus the run's small files (`resolved_config.yaml`, `run_meta.json`, `train_summary.json`) under
`<prefix>/`, and records the resulting commit in `<run-dir>/push_manifest.json`.
`eval-configs` writes `configs/eval_configs/<id>@s<step>.yaml`: the adapter on the **text-only** RL start (the
adapters target `model.layers.N...`, so the base is `configs/model_sft_kne4.yaml`'s model, not the multimodal one),
as `hf://<repo>/<prefix>/checkpoint-<step>@<revision>` (revision from the push manifest) or, with `--local`, the
local checkpoint dir (pilot use before the push).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import yaml

from calign.config import git_commit
from calign.paths import REPO_ROOT
from calign.schemas import read_json, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)

RL_REPO = "felixfabricius/gemma-3-27b-it-halden-rl"
RL_START_MODEL_CONFIG = "configs/model_sft_kne4.yaml"
PUSH_MANIFEST = "push_manifest.json"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
RUN_FILES = ("resolved_config.yaml", "run_meta.json", "train_summary.json")
EVAL_CONFIGS_DIR = REPO_ROOT / "configs" / "eval_configs"
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_steps(run_dir: Path) -> list[int]:
    return sorted(
        int(m.group(1))
        for p in run_dir.iterdir()
        if p.is_dir() and (m := _CKPT_RE.match(p.name)) and (p / "adapter_config.json").exists()
    )


def rl_start() -> tuple[str, str | None]:
    """(hub id, revision) of the text-only RL start, from its model config."""
    m = yaml.safe_load((REPO_ROOT / RL_START_MODEL_CONFIG).read_text(encoding="utf-8"))
    return m["model_path"], m.get("revision")


def push(run_dir: Path, prefix: str, repo: str = RL_REPO, steps: list[int] | None = None) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi

    from calign.paths import hf_token

    steps = steps or checkpoint_steps(run_dir)
    if not steps:
        raise SystemExit(f"{run_dir}: no checkpoint-<step>/ adapters")
    api = HfApi(token=hf_token())
    api.create_repo(repo, private=True, exist_ok=True)
    ops = []
    for s in steps:
        d = run_dir / f"checkpoint-{s}"
        for f in ADAPTER_FILES:
            ops.append(CommitOperationAdd(f"{prefix}/checkpoint-{s}/{f}", str(d / f)))
    for f in RUN_FILES:
        if (run_dir / f).exists():
            ops.append(CommitOperationAdd(f"{prefix}/{f}", str(run_dir / f)))
    info = api.create_commit(repo, ops, commit_message=f"{prefix}: adapters at steps {steps} ({run_dir.name})")
    files = set(api.list_repo_files(repo, revision=info.oid))
    missing = [o.path_in_repo for o in ops if o.path_in_repo not in files]
    if missing:
        raise SystemExit(f"push incomplete, missing on HF: {missing}")
    man = read_json(run_dir / PUSH_MANIFEST) if (run_dir / PUSH_MANIFEST).exists() else {"pushes": []}
    man["pushes"].append(
        {
            "repo": repo,
            "prefix": prefix,
            "steps": steps,
            "revision": info.oid,
            "files": [o.path_in_repo for o in ops],
            "git_commit": git_commit(),
            "created_at": utc_now_iso(),
        }
    )
    man["latest"] = man["pushes"][-1]
    write_json(run_dir / PUSH_MANIFEST, man)
    LOGGER.info("pushed %d files to %s@%s", len(ops), repo, info.oid)
    return man["latest"]


def eval_config(
    config_id: str, step: int, adapter: str, label: str, suffix: str = "s", notes: str = ""
) -> tuple[str, dict]:
    base, rev = rl_start()
    cid = f"{config_id}@{suffix}{step}"
    return cid, {
        "id": cid,
        "label": label,
        "model_config": RL_START_MODEL_CONFIG,
        "model_path": base,
        "revision": rev,
        "adapter": adapter,
        "system_prompt_variant": "none",
        "stage": "rl",
        "notes": notes,
    }


def write_eval_configs(
    config_id: str,
    run_dir: Path,
    steps: list[int] | None = None,
    revision: str | None = None,
    local: bool = False,
    suffix: str = "s",
    out_dir: Path = EVAL_CONFIGS_DIR,
) -> list[Path]:
    steps = steps or checkpoint_steps(run_dir)
    if local:
        push_info = None
    else:
        man = read_json(run_dir / PUSH_MANIFEST) if (run_dir / PUSH_MANIFEST).exists() else None
        push_info = man["latest"] if man else None
        if push_info is None and revision is None:
            raise SystemExit(f"{run_dir}: no {PUSH_MANIFEST}; pass --revision or --local")
    paths = []
    for s in steps:
        if local:
            adapter = (run_dir / f"checkpoint-{s}").as_posix()
            if Path(adapter).is_absolute() and Path(adapter).is_relative_to(REPO_ROOT):
                adapter = Path(adapter).relative_to(REPO_ROOT).as_posix()
        else:
            repo = push_info["repo"] if push_info else RL_REPO
            prefix = push_info["prefix"] if push_info else config_id
            rev = revision or push_info["revision"]
            adapter = f"hf://{repo}/{prefix}/checkpoint-{s}@{rev}"
        cid, body = eval_config(
            config_id,
            s,
            adapter,
            label=f"{config_id} GRPO step {s}",
            suffix=suffix,
            notes=f"RL adapter of {run_dir.as_posix()} at step {s} on the text-only RL start (calign.rl.checkpoints)",
        )
        p = out_dir / f"{cid}.yaml"
        header = (
            f"# {cid}: the {config_id} GRPO adapter at optimizer step {s}, LoRA-served on the text-only RL start\n"
            f"# (written by calign.rl.checkpoints eval-configs; the adapters target model.layers.N, so the base is the\n"
            f"# Gemma3ForCausalLM export, not the multimodal checkpoint).\n"
        )
        p.write_text(header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")
        paths.append(p)
        LOGGER.info("wrote %s", p)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--repo", default=RL_REPO)
    p.add_argument("--steps", type=int, nargs="+", default=None)
    e = sub.add_parser("eval-configs")
    e.add_argument("--config-id", required=True)
    e.add_argument("--run-dir", type=Path, required=True)
    e.add_argument("--steps", type=int, nargs="+", default=None)
    e.add_argument("--revision", default=None)
    e.add_argument("--local", action="store_true", help="adapter = the local checkpoint dir (before the push)")
    e.add_argument("--suffix", default="s", help="checkpoint suffix: <id>@<suffix><step>")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from calign.paths import load_env

    load_env()
    if args.cmd == "push":
        print(json.dumps(push(args.run_dir, args.prefix, args.repo, args.steps), indent=2))
    else:
        for path in write_eval_configs(
            args.config_id, args.run_dir, args.steps, args.revision, args.local, args.suffix
        ):
            print(path)


if __name__ == "__main__":
    main()
