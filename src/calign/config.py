"""YAML config loading, common CLI arguments, and run directories.

Every CLI module in `calign` follows the same conventions:

    --config PATH     YAML config (pydantic-validated)
    --dry-run         process <= DRY_RUN_LIMIT items with verbose output, write under outputs/dry_run/
    --limit N         process at most N items
    --seed N          override the config seed
    --out DIR         override the output directory
    --model-path P    override the model path (base vs. merged)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

from calign.paths import OUTPUTS_DIR, REPO_ROOT

DRY_RUN_LIMIT = 3

C = TypeVar("C", bound=BaseModel)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def load_config(path: Path, model: type[C], overrides: dict[str, Any] | None = None) -> C:
    data = load_yaml(path)
    for k, v in (overrides or {}).items():
        if v is not None:
            data[k] = v
    return model.model_validate(data)


def add_common_args(parser: argparse.ArgumentParser, default_config: Path | None = None) -> None:
    parser.add_argument("--config", type=Path, default=default_config, help="YAML config path")
    parser.add_argument("--dry-run", action="store_true", help=f"process <= {DRY_RUN_LIMIT} items, verbose")
    parser.add_argument("--limit", type=int, default=None, help="process at most N items")
    parser.add_argument("--seed", type=int, default=None, help="override seed")
    parser.add_argument("--out", type=Path, default=None, help="override output directory")
    parser.add_argument("--model-path", type=str, default=None, help="override model path/id")
    parser.add_argument("--revision", type=str, default=None, help="pin the HF revision (commit hash) of --model-path")


def effective_limit(args: argparse.Namespace) -> int | None:
    if getattr(args, "dry_run", False):
        return DRY_RUN_LIMIT if args.limit is None else min(args.limit, DRY_RUN_LIMIT)
    return args.limit


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def stable_hash(obj: Any, n: int = 8) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_run_dir(kind: str, config: BaseModel | dict | None, out: Path | None = None, dry_run: bool = False) -> Path:
    """Create an immutable run directory `outputs/<kind>/<timestamp>_<confighash>/`.

    Writes `resolved_config.yaml` and `run_meta.json` (git commit, dirty flag, timestamp).
    """
    cfg = config.model_dump() if isinstance(config, BaseModel) else (config or {})
    if out is not None:
        run_dir = Path(out)
    else:
        base = OUTPUTS_DIR / ("dry_run" if dry_run else "") / kind if dry_run else OUTPUTS_DIR / kind
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / f"{stamp}_{stable_hash(cfg)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    meta = {
        "kind": kind,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "dry_run": dry_run,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir
