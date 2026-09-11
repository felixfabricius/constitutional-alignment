import json

import numpy as np
import pytest

from calign.probe import metrics, train
from calign.probe.config import ProbeConfig
from calign.probe.labels import BUILTIN_SPECS
from calign.probe.store import ActivationStore, ActivationWriter
from calign.schemas import (
    Condition,
    ConstitutionVerdict,
    GenerationRecord,
    JudgeResult,
    Message,
    ModelRef,
    ProbeRecord,
    Sampling,
    read_jsonl,
)

D, LAYERS, POS = 16, [16, 31], ["p100", "mean"]


def test_auroc_and_balanced_accuracy():
    assert metrics.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert metrics.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0 and metrics.auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert metrics.auroc([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5  # ties -> chance
    assert metrics.auroc([1, 2], [1, 1]) is None
    assert metrics.balanced_accuracy([0, 1, 2, 3], [0, 0, 1, 1], 1.5) == 1.0
    assert metrics.balanced_accuracy([0, 1, 2, 3], [0, 0, 1, 1], 2.5) == 0.75
    assert metrics.best_threshold(np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1])) == 1.5
    assert metrics.spearman([1, 2, 3, 4], [10, 20, 30, 45]) == pytest.approx(1.0)
    assert metrics.spearman([1, 2, 3], [3, 3, 3]) is None


def test_cluster_bootstrap_and_summary_deterministic():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(1, 1, 40), rng.normal(0, 1, 40)])
    labels = np.array([1] * 40 + [0] * 40)
    clusters = [f"s{i % 10}" for i in range(80)]
    s1 = metrics.classification_summary(scores, labels, clusters, 0.5, n_boot=200, seed=1)
    s2 = metrics.classification_summary(scores, labels, clusters, 0.5, n_boot=200, seed=1)
    assert s1 == s2 and s1["n_clusters"] == 10 and s1["n_pos"] == 40
    assert s1["auroc_ci95"][0] <= s1["auroc"] <= s1["auroc_ci95"][1]
    assert s1["balanced_accuracy_ci95"][0] <= s1["balanced_accuracy"] <= s1["balanced_accuracy_ci95"][1]
    assert metrics.cluster_bootstrap(["a", "a"], lambda i: None, 5) == (None, None)


def _judge(mentions, outcome):
    return JudgeResult(
        judge_model="m",
        prompt_version="v",
        mentions_constitution=mentions,
        principles_cited=[3] if mentions else [],
        citation_accuracy=1.0,
        principle_relevance=1.0,
        outcome_alignment=outcome,
        decision="action1",
        raw="{}",
    )


def _synthetic_run(tmp_path, n_scen=30, k=4, seed=0):
    """Records with a planted direction: positives (cell 1) are shifted along `u` at layer 31 / p100 only."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(D)
    u /= np.linalg.norm(u)
    records, verdicts, truth = [], {}, {}
    with ActivationWriter(tmp_path, LAYERS, POS, D, shard_size=7) as w:
        for i in range(n_scen):
            sid = f"H_{i:03d}"
            split = "probe_train" if i < 20 else "probe_val"
            verdicts[sid] = ConstitutionVerdict(
                scenario_id=sid,
                prescribed_action="action1",
                confidence=1,
                rationale="r",
                judge_model="m",
                prompt_version="v",
            )
            for j in range(k):
                pos = (i + j) % 3 != 0  # ~2/3 positives
                acts = rng.standard_normal((len(LAYERS), len(POS), D)).astype(np.float32)
                if pos:
                    acts[1, 0] += 3.0 * u  # layer 31, position p100
                r = GenerationRecord(
                    scenario_id=sid,
                    source="moralchoice_high",
                    split=split,
                    model=ModelRef(name="m", path="p", stage="sft_merged"),
                    condition=Condition(constitution_in_prompt=False, prompt_variant="none"),
                    sampling=Sampling(temperature=1.0, max_tokens=8, sample_idx=j),
                    messages=[Message(role="user", content="q")],
                    prompt_text="<bos>q",
                    response_text="Final answer: A",
                    judge=_judge(1.0 if pos else 0.0, 1.0),
                )
                ref = w.add(r.record_id, acts, {"p100": 10, "mean": -1})
                records.append(r.model_copy(update={"activations": ref}))
                truth[r.record_id] = pos
    return records, verdicts, u


def test_train_recovers_planted_direction(tmp_path):
    records, verdicts, u = _synthetic_run(tmp_path)
    cfg = ProbeConfig(train={"bootstrap": {"n": 100, "seed": 0}}, labels={"specs": ["B_primary"]})
    store = ActivationStore(tmp_path)
    probes, dirs, means, rows = train.train_probes(tmp_path, records, store, cfg, verdicts)
    by_id = {p.probe_id: p for p in probes}
    assert set(by_id) == {"B_primary/L16/p100", "B_primary/L16/mean", "B_primary/L31/p100", "B_primary/L31/mean"}
    good = by_id["B_primary/L31/p100"]
    assert abs(float(dirs[good.direction_row] @ u)) > 0.9 and good.val_metrics["auroc"] > 0.95
    assert good.n_pos + good.n_neg == 80 and good.n_scenarios == 20 and good.val_metrics["n"] == 40
    assert good.class_gap == pytest.approx(3.0, abs=1.0)
    assert good.val_metrics["balanced_accuracy"] > 0.9
    noise = by_id["B_primary/L16/mean"]
    assert noise.val_metrics["auroc"] < 0.75  # nothing planted there
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-5) and means.shape == (4, 2, D)
    assert len(rows) == 4 * 120 and {r["split"] for r in rows} == {"probe_train", "probe_val"}
    # class means are consistent with the direction
    d_check = means[good.direction_row, 0] - means[good.direction_row, 1]
    assert np.allclose(d_check / np.linalg.norm(d_check), dirs[good.direction_row], atol=1e-5)


def test_write_outputs_and_summary(tmp_path):
    data_run = tmp_path / "data"
    data_run.mkdir()
    records, verdicts, _ = _synthetic_run(data_run, n_scen=12, k=2)
    from calign.schemas import write_jsonl

    write_jsonl(data_run / "records.jsonl", records)
    cfg = ProbeConfig(train={"bootstrap": {"n": 50, "seed": 0}}, labels={"specs": ["B_primary", "B_process"]})
    store = ActivationStore(data_run)
    probes, dirs, means, rows = train.train_probes(data_run, records, store, cfg, verdicts)
    out = tmp_path / "probes"
    out.mkdir()
    s = train.write_outputs(out, data_run, probes, dirs, means, rows, cfg)
    assert (out / "probes.jsonl").exists() and (out / "directions.safetensors").exists()
    assert len(read_jsonl(out / "probes.jsonl", ProbeRecord)) == 8
    conv = json.loads((out / "convergence.json").read_text(encoding="utf-8"))
    assert len(conv["probe_ids"]) == 8 and conv["cosine"][0][0] == pytest.approx(1.0)
    assert s["best_by_spec"]["B_primary"]["probe_id"].startswith("B_primary/")
    assert s["provenance"]["data_run"] == str(data_run)
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "B_primary/L31/p100" in md and "Best probe per spec" in md
    scores = [json.loads(line) for line in (out / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert scores[0].keys() == {
        "probe_id",
        "record_id",
        "scenario_id",
        "split",
        "variant",
        "cell",
        "label",
        "projection",
    }


def test_spec_without_both_classes_is_skipped(tmp_path):
    records, verdicts, _ = _synthetic_run(tmp_path, n_scen=6, k=2)
    cfg = ProbeConfig(
        train={"bootstrap": {"n": 10}}, labels={"specs": ["B_outcome", "B_primary"]}
    )  # all outcomes aligned
    probes, *_ = train.train_probes(tmp_path, records, verdicts=verdicts, store=ActivationStore(tmp_path), cfg=cfg)
    assert {p.label_spec for p in probes} == {"B_primary"}
    with pytest.raises(SystemExit):
        train.train_probes(
            tmp_path, records, ActivationStore(tmp_path), ProbeConfig(labels={"specs": ["B_outcome"]}), verdicts
        )
    assert BUILTIN_SPECS["B_outcome"].positive == ["mentioned_aligned", "unmentioned_aligned"]
