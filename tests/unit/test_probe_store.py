import json

import numpy as np
import pytest

from calign.probe.config import ProbeConfig, load_probe_config
from calign.probe.store import ACT_DIR, INDEX_FILE, ActivationStore, ActivationWriter
from calign.schemas import ActivationRef

LAYERS, POS, D = [16, 31], ["prompt_last", "p100", "mean"], 8


def _write(run_dir, n=5, shard_size=2, dtype="float32"):
    rng = np.random.default_rng(0)
    data = {f"gen_{i:02d}": rng.standard_normal((len(LAYERS), len(POS), D)).astype(np.float32) for i in range(n)}
    with ActivationWriter(run_dir, LAYERS, POS, D, dtype=dtype, shard_size=shard_size, context_variant="none") as w:
        refs = {
            rid: w.add(rid, a, {"prompt_last": 10, "p100": 20 + i, "mean": -1})
            for i, (rid, a) in enumerate(data.items())
        }
    return data, refs


def test_writer_shards_index_and_refs(tmp_path):
    data, refs = _write(tmp_path)
    index = json.loads((tmp_path / ACT_DIR / INDEX_FILE).read_text(encoding="utf-8"))
    assert index["layers"] == LAYERS and index["positions"] == POS and index["d_model"] == D
    assert index["n_records"] == 5 and [s["n"] for s in index["shards"]] == [2, 2, 1]
    assert index["context_variant"] == "none" and index["dtype"] == "float32"
    assert refs["gen_02"] == ActivationRef(
        path="activations/shard_001.safetensors",
        layers=LAYERS,
        positions={"prompt_last": 10, "p100": 22, "mean": -1},
        row=0,
        context_variant="none",
    )
    assert refs["gen_04"].path.endswith("shard_002.safetensors") and refs["gen_04"].row == 0


def test_store_roundtrip_and_matrix(tmp_path):
    data, refs = _write(tmp_path)
    store = ActivationStore(tmp_path)
    assert len(store) == 5 and "gen_03" in store and store.layers == LAYERS and store.positions == POS
    for rid, a in data.items():
        np.testing.assert_array_equal(store.get(rid), a)
        store.check_ref(rid, refs[rid])
    ids = ["gen_04", "gen_00", "gen_03"]
    m = store.matrix(ids, layer=31, position="mean")
    assert m.shape == (3, D) and m.dtype == np.float32
    np.testing.assert_array_equal(m, np.stack([data[r][1, 2] for r in ids]))
    with pytest.raises(ValueError, match="does not match"):
        store.check_ref("gen_00", refs["gen_01"])
    with pytest.raises(KeyError):
        store.get("gen_99")


def test_writer_validation(tmp_path):
    w = ActivationWriter(tmp_path, LAYERS, POS, D)
    a = np.zeros((len(LAYERS), len(POS), D), dtype=np.float32)
    pos = {"prompt_last": 1, "p100": 2, "mean": -1}
    w.add("a", a, pos)
    with pytest.raises(ValueError, match="duplicate"):
        w.add("a", a, pos)
    with pytest.raises(ValueError, match="shape"):
        w.add("b", np.zeros((1, 3, D), dtype=np.float32), pos)
    with pytest.raises(ValueError, match="positions"):
        w.add("c", a, {"p100": 2})
    w.close()
    with pytest.raises(FileExistsError):
        ActivationWriter(tmp_path, LAYERS, POS, D)


def test_float16_storage(tmp_path):
    data, _ = _write(tmp_path, n=3, dtype="float16")
    store = ActivationStore(tmp_path)
    assert store.get("gen_00").dtype == np.float16
    assert store.matrix(["gen_00"], 16, "p100").dtype == np.float32


def test_probe_config_loads_and_resolves_specs():
    cfg = load_probe_config()
    assert cfg.model_config_path.endswith("model_sft_v2e3.yaml")
    assert [s.name for s in cfg.labels.resolved_specs()] == [
        "B_primary",
        "B_cell1_vs_cell3",
        "B_outcome",
        "B_process",
        "C_context",
    ]
    assert cfg.activations.positions == ["prompt_last", "p033", "p066", "p100", "decision", "mean"]
    assert cfg.steering.coef_grid == [1, 2, 4, 8] and cfg.steering.temperature == 0.0
    assert load_probe_config(seed=7).sampling.seed == 7
    with pytest.raises(ValueError, match="unknown activation positions"):
        ProbeConfig(activations={"positions": ["p050"]})
    with pytest.raises(ValueError, match="unknown label spec"):
        ProbeConfig(labels={"specs": ["nope"]}).labels.resolved_specs()
    inline = ProbeConfig(
        labels={
            "specs": [
                "B_primary",
                {"name": "x", "positive": ["mentioned_aligned"], "negative": ["unmentioned_aligned"]},
            ]
        }
    )
    assert inline.labels.resolved_specs()[1].name == "x"
