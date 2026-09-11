import json

import numpy as np
import pytest

from calign.probe import sae
from calign.probe.config import ProbeConfig
from calign.schemas import ProbeRecord

D = 16


def test_names_and_decoder_key():
    assert sae.sae_folder(31, "65k", "medium") == "resid_post/layer_31_width_65k_l0_medium"
    assert sae.neuronpedia_source(31, "65k") == "31-gemmascope-2-res-65k"
    assert sae.find_decoder_key({"W_enc": (D, 100), "W_dec": (100, D)}, D) == "W_dec"
    assert sae.find_decoder_key({"enc": (D, 100), "dec": (100, D), "b": (100,)}, D) == "dec"
    with pytest.raises(KeyError):
        sae.find_decoder_key({"a": (100, D), "b": (100, D)}, D)


def test_load_decoder_from_safetensors(tmp_path):
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    w = rng.standard_normal((40, D)).astype(np.float32) * 3
    save_file(
        {"W_enc": w.T.copy(), "W_dec": w, "threshold": np.zeros(40, dtype=np.float32)},
        str(tmp_path / "params.safetensors"),
    )
    dec = sae.load_decoder(tmp_path / "params.safetensors", D)
    assert dec.shape == (40, D) and np.allclose(np.linalg.norm(dec, axis=1), 1.0, atol=1e-5)


def test_top_features_and_stats():
    rng = np.random.default_rng(1)
    dec = rng.standard_normal((200, D)).astype(np.float32)
    dec /= np.linalg.norm(dec, axis=1, keepdims=True)
    d = rng.standard_normal(D)
    dec[7] = d / np.linalg.norm(d)  # planted aligned feature
    dec[9] = -dec[7]  # planted anti-aligned feature
    top = sae.top_features(dec, d, 3)
    assert len(top) == 6 and top[0] == {"feature": 7, "cosine": pytest.approx(1.0), "rank": 1, "sign": 1}
    assert top[3]["feature"] == 9 and top[3]["cosine"] == pytest.approx(-1.0) and top[3]["sign"] == -1
    st = sae.cosine_stats(dec, d)
    assert (
        st["max_abs_cos"] == pytest.approx(1.0)
        and st["n_features"] == 200
        and st["random_baseline_abs_cos"] == pytest.approx(0.25)
    )


def test_parse_feature_json_and_cache(tmp_path, monkeypatch):
    d = {
        "explanations": [{"description": "mentions of a written constitution", "explanationModelName": "gemini"}],
        "pos_str": ["const", "itution", "x"] * 5,
    }
    p = sae.parse_feature_json(d)
    assert p["explanation"].startswith("mentions") and p["explanation_model"] == "gemini" and len(p["top_tokens"]) == 10
    assert sae.parse_feature_json({}) == {
        "explanation": None,
        "explanation_model": None,
        "n_explanations": 0,
        "top_tokens": [],
    }
    cache = tmp_path / "np"
    (cache / "m" / "s").mkdir(parents=True)
    (cache / "m" / "s" / "5.json").write_text(json.dumps(d), encoding="utf-8")
    assert sae.fetch_feature("m", "s", 5, cache) == d  # served from cache, no network

    def boom(*a, **k):
        raise sae.urllib.error.URLError("offline")

    monkeypatch.setattr(sae.urllib.request, "urlopen", boom)
    assert sae.fetch_feature("m", "s", 6, cache) is None


def test_run_with_fake_decoder(monkeypatch):
    rng = np.random.default_rng(2)
    dirs = rng.standard_normal((2, D)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    probes = [
        ProbeRecord(
            probe_id=f"B_primary/L{L}/p100",
            label_spec="B_primary",
            layer=L,
            position="p100",
            data_run="d",
            n_pos=1,
            n_neg=1,
            n_scenarios=1,
            direction_row=i,
            direction_sha="s",
            class_gap=1.0,
            threshold_midpoint=0.0,
            mean_resid_norm=1.0,
            train_metrics={},
            val_metrics={},
        )
        for i, L in enumerate((16, 31))
    ]
    decs = {L: rng.standard_normal((50, D)).astype(np.float32) for L in (16, 31)}
    decs[31][3] = dirs[1]
    fetched = []
    monkeypatch.setattr(
        sae,
        "fetch_feature",
        lambda m, s, i, cache_dir=None: fetched.append((s, i)) or {"explanations": [{"description": f"feat {i}"}]},
    )
    cfg = ProbeConfig(sae={"top_k": 4})
    rows, stats = sae.run(
        probes,
        dirs,
        cfg,
        [16, 31],
        True,
        2,
        decoder_loader=lambda L: decs[L] / np.linalg.norm(decs[L], axis=1, keepdims=True),
    )
    assert len(rows) == 2 * 8 and stats["B_primary/L31/p100"]["max_abs_cos"] == pytest.approx(1.0)
    best = next(r for r in rows if r["probe_id"] == "B_primary/L31/p100" and r["rank"] == 1 and r["sign"] == 1)
    assert best["feature"] == 3 and best["explanation"] == "feat 3" and best["source"] == "31-gemmascope-2-res-65k"
    assert len(fetched) == 2 * 2 * 2 and all(
        s in ("16-gemmascope-2-res-65k", "31-gemmascope-2-res-65k") for s, _ in fetched
    )
    assert "explanation" not in next(r for r in rows if r["rank"] == 4)  # beyond neuronpedia_top
    md = sae.render_markdown(rows, stats, cfg)
    assert "B_primary/L31/p100" in md and "feat 3" in md
