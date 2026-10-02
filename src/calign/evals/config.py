"""Eval configurations (`configs/eval_configs/<id>.yaml`) and Phase 3 run directories.

A configuration names a model (base or merged weights, optional LoRA adapter served by vLLM) and the system-prompt
variant it is evaluated with. Every Phase 3 component (MoralChoice, IFEval, MATH-500, ...) takes one configuration
and writes an immutable run dir `outputs/evals/<config id>/<component>/<timestamp>_<hash>/` whose
`resolved_config.yaml` embeds the eval config and the resolved model config.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import field_validator

from calign.config import ConfigModel, load_yaml, new_run_dir, stable_hash
from calign.evals import SUITE_VERSION
from calign.inference.backend import ModelConfig, load_model_config
from calign.paths import OUTPUTS_DIR, REPO_ROOT

EVALS_DIR = OUTPUTS_DIR / "evals"
EVAL_CONFIGS_DIR = REPO_ROOT / "configs" / "eval_configs"

# A configuration whose prompt variant is still to be chosen (chunk 4) carries this placeholder and cannot run.
UNSET_VARIANT = "TBD-chunk-4"

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+(@[A-Za-z0-9_.\-]+)?$")


class EvalConfig(ConfigModel):
    id: str
    label: str = ""
    model_config_path: str = "configs/model.yaml"
    model_path: str = "google/gemma-3-27b-it"
    revision: str | None = None
    adapter: str | None = None
    system_prompt_variant: str = "none"
    stage: Literal["base", "sft", "rl"] = "base"
    notes: str = ""

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(f"eval config id must look like C2 or C2@e2, got {v!r}")
        return v

    @property
    def runnable(self) -> bool:
        return self.system_prompt_variant != UNSET_VARIANT

    @property
    def base_id(self) -> str:
        """Configuration id without the checkpoint suffix (C2@e2 -> C2)."""
        return self.id.split("@", 1)[0]


def load_eval_config(path: Path | str) -> EvalConfig:
    """Load an eval config; a bare id (`C0`) resolves to `configs/eval_configs/C0.yaml`.

    The YAML key `model_config` (as in phase3/README.md Section 7) is accepted as an alias of `model_config_path`
    (pydantic reserves the attribute name `model_config`).
    """
    p = Path(path)
    if not p.suffix and not p.exists():
        p = EVAL_CONFIGS_DIR / f"{path}.yaml"
    data = load_yaml(p)
    if "model_config" in data:
        if "model_config_path" in data:
            raise ValueError(f"{p}: give either model_config or model_config_path, not both")
        data["model_config_path"] = data.pop("model_config")
    return EvalConfig.model_validate(data)


def resolve_model(cfg: EvalConfig, require_runnable: bool = True) -> tuple[ModelConfig, str | None, str]:
    """(ModelConfig with the config's model path and revision, adapter path or None, system-prompt variant)."""
    if require_runnable and not cfg.runnable:
        raise ValueError(f"eval config {cfg.id} has system_prompt_variant={UNSET_VARIANT}; chunk 4 sets it")
    model_cfg = load_model_config(REPO_ROOT / cfg.model_config_path, model_path=cfg.model_path, revision=cfg.revision)
    adapter = None
    if cfg.adapter and cfg.adapter.startswith("hf://"):
        adapter = cfg.adapter  # downloaded by calign.inference.lora.resolve_adapter when the backend loads
    elif cfg.adapter:
        a = Path(cfg.adapter)
        adapter = str(a if a.is_absolute() else REPO_ROOT / a)
    return model_cfg, adapter, cfg.system_prompt_variant


def eval_run_dir(
    cfg: EvalConfig,
    component: str,
    params: dict[str, Any] | None = None,
    out_root: Path | None = None,
    out: Path | None = None,
    dry_run: bool = False,
    model_cfg: ModelConfig | None = None,
) -> Path:
    """Create `<out_root>/<cfg.id>/<component>/<timestamp>_<hash>` (or `out`) with resolved config and run meta.

    `resolved_config.yaml` holds {eval_config, model, component, params}; `run_meta.json` additionally records the
    eval config id, model path, revision and adapter so a run dir is self-describing.
    """
    resolved = {
        "eval_config": cfg.model_dump(),
        "model": model_cfg.model_dump() if model_cfg is not None else None,
        "component": component,
        "params": params or {},
    }
    if out is None:
        root = Path(out_root) if out_root is not None else EVALS_DIR
        if dry_run:
            root = OUTPUTS_DIR / "dry_run" / "evals"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = root / cfg.id / component / f"{stamp}_{stable_hash(resolved)}"
    if (Path(out) / "run_meta.json").exists():
        raise FileExistsError(f"{out} already holds a run; run dirs are immutable")
    run_dir = new_run_dir(f"evals/{component}", resolved, out=out, dry_run=dry_run)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    meta.update(
        {
            "eval_config_id": cfg.id,
            "component": component,
            "model_path": cfg.model_path,
            "revision": cfg.revision,
            "adapter": cfg.adapter,
            "system_prompt_variant": cfg.system_prompt_variant,
            "suite_version": SUITE_VERSION,
        }
    )
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def stamp_suite_version(run_dir: Path) -> None:
    """Record SUITE_VERSION in a run dir's run_meta.json (judge runs created with `calign.config.new_run_dir`)."""
    p = Path(run_dir) / "run_meta.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["suite_version"] = SUITE_VERSION
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_run_eval_config(run_dir: Path) -> EvalConfig:
    """The eval config embedded in a run dir's resolved_config.yaml."""
    data = yaml.safe_load((Path(run_dir) / "resolved_config.yaml").read_text(encoding="utf-8"))
    return EvalConfig.model_validate(data["eval_config"])
