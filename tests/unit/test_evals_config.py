import json

import pytest
import yaml

from calign.evals.config import (
    EvalConfig,
    eval_run_dir,
    load_eval_config,
    read_run_eval_config,
    resolve_model,
)
from calign.paths import REPO_ROOT


def test_c0_loads_and_resolves():
    cfg = load_eval_config(REPO_ROOT / "configs" / "eval_configs" / "C0.yaml")
    assert cfg.id == "C0" and cfg.runnable and cfg.adapter is None
    model_cfg, adapter, variant = resolve_model(cfg)
    assert model_cfg.model_path == "google/gemma-3-27b-it" and adapter is None and variant == "none"


def test_bare_id_resolves_to_registry():
    assert load_eval_config("C0").id == "C0"


def test_c1_is_not_runnable_until_chunk4():
    cfg = load_eval_config("C1")
    assert not cfg.runnable
    with pytest.raises(ValueError, match="chunk 4"):
        resolve_model(cfg)


def test_round_trip_and_checkpoint_ids(tmp_path):
    cfg = EvalConfig(id="C2@e2", model_path="/x/merged", adapter="outputs/models/sft_v3/adapter_epoch2", stage="sft")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")
    back = load_eval_config(p)
    assert back == cfg and back.base_id == "C2"
    _, adapter, _ = resolve_model(back)
    assert adapter is not None and adapter.endswith("adapter_epoch2")


def test_unknown_field_and_bad_id_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("id: C0\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(Exception, match="bogus"):
        load_eval_config(p)
    with pytest.raises(ValueError, match="must look like"):
        EvalConfig(id="C0 with space")


def test_eval_run_dir_embeds_config_and_is_immutable(tmp_path):
    cfg = load_eval_config("C0")
    d = eval_run_dir(cfg, "moralchoice", {"k": 4}, out_root=tmp_path)
    assert d.parent == tmp_path / "C0" / "moralchoice"
    resolved = yaml.safe_load((d / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["eval_config"]["id"] == "C0" and resolved["params"] == {"k": 4}
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["eval_config_id"] == "C0" and meta["component"] == "moralchoice"
    assert read_run_eval_config(d) == cfg
    with pytest.raises(FileExistsError):
        eval_run_dir(cfg, "moralchoice", out=d)
