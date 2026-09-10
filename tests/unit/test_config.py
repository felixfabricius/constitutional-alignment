import argparse

import yaml

from calign.config import ConfigModel, add_common_args, effective_limit, load_config, new_run_dir, stable_hash


class Cfg(ConfigModel):
    seed: int = 0
    n: int = 5


def test_load_config_with_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\nn: 7\n", encoding="utf-8")
    cfg = load_config(p, Cfg, overrides={"seed": 42, "n": None})
    assert cfg.seed == 42 and cfg.n == 7


def test_load_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\nbogus: 2\n", encoding="utf-8")
    try:
        load_config(p, Cfg)
    except Exception as e:  # noqa: BLE001
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_common_args_and_limits():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    a = parser.parse_args(["--dry-run", "--limit", "10"])
    assert effective_limit(a) == 3
    a = parser.parse_args(["--limit", "10"])
    assert effective_limit(a) == 10
    a = parser.parse_args([])
    assert effective_limit(a) is None


def test_new_run_dir_writes_metadata(tmp_path):
    cfg = Cfg(seed=3)
    d = new_run_dir("unit", cfg, out=tmp_path / "run")
    assert (d / "resolved_config.yaml").exists() and (d / "run_meta.json").exists()
    assert yaml.safe_load((d / "resolved_config.yaml").read_text()) == {"seed": 3, "n": 5}


def test_stable_hash_deterministic():
    assert stable_hash({"b": 1, "a": 2}) == stable_hash({"a": 2, "b": 1})
