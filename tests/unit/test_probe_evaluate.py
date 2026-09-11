import json

import numpy as np
import pytest
from test_probe_train import D, _synthetic_run

from calign.config import sha256_text
from calign.misalignment.constitution_judge import PROMPT_VERSION
from calign.probe import evaluate, train
from calign.probe.config import ProbeConfig
from calign.probe.store import ActivationStore, ActivationWriter
from calign.schemas import MisalignmentSample, ModelRef

HARD_POS = ["prompt_last", "pre_tool", "p100", "mean"]


def _hard_run(tmp_path, u, n_per_cond=6, seed=1):
    """Samples whose pre_tool/p100 activations at layer 31 are shifted along `u` when aligned (score >= 0.5)."""
    rng = np.random.default_rng(seed)
    samples = []
    conds = ["leaking_explicit-america_replacement", "murder_none-none_none", "blackmail_explicit-america_none"]
    with ActivationWriter(tmp_path, [16, 31], HARD_POS, D) as w:
        for cid in conds:
            for i in range(n_per_cond):
                aligned = i % 2 == 0
                harmful = not aligned and i % 3 == 1
                acts = rng.standard_normal((2, len(HARD_POS), D)).astype(np.float32)
                if aligned:
                    acts[1, 1] += 3.0 * u  # layer 31, pre_tool
                    acts[1, 2] += 3.0 * u  # layer 31, p100
                s = MisalignmentSample(
                    condition_id=cid,
                    scenario=cid.split("_")[0],
                    goal_type="x",
                    goal_value="y",
                    urgency_type="z",
                    sample_idx=i,
                    model=ModelRef(name="m", path="p", stage="sft_merged"),
                    system_prompt_sha=sha256_text("s"),
                    user_prompt_sha=sha256_text("u"),
                    response_text="r",
                    used_tool_format=True,
                    harmful=harmful,
                    constitution_score=0.9 if aligned else 0.2,
                    constitution_judge={
                        "prompt_version": PROMPT_VERSION,
                        "mentions_constitution": 1.0 if i % 4 else 0.0,
                    },
                )
                ref = w.add(evaluate.sample_key(s), acts, {"prompt_last": 5, "pre_tool": 20, "p100": 30, "mean": -1})
                samples.append(s.model_copy(update={"activations": ref}))
    return samples


def test_check_hard_run_refuses_incomplete():
    s = MisalignmentSample(
        condition_id="c_x-y_z",
        scenario="leaking",
        goal_type="x",
        goal_value="y",
        urgency_type="z",
        sample_idx=0,
        model=ModelRef(name="m", path="p", stage="base"),
        system_prompt_sha="a" * 64,
        user_prompt_sha="b" * 64,
        response_text="r",
        used_tool_format=False,
    )
    with pytest.raises(SystemExit, match="activations"):
        evaluate.check_hard_run([s])
    v1 = s.model_copy(
        update={"constitution_score": 0.5, "constitution_judge": {"prompt_version": "constitution-score-v1"}}
    )
    with pytest.raises(SystemExit, match="judge_v2"):
        evaluate.check_hard_run([v1])


def test_hard_cell_and_spec_label():
    cfg = ProbeConfig()
    from calign.probe.labels import BUILTIN_SPECS

    mk = lambda m, sc: MisalignmentSample(  # noqa: E731
        condition_id="c_x-y_z",
        scenario="murder",
        goal_type="x",
        goal_value="y",
        urgency_type="z",
        sample_idx=0,
        model=ModelRef(name="m", path="p", stage="base"),
        system_prompt_sha="a" * 64,
        user_prompt_sha="b" * 64,
        response_text="r",
        used_tool_format=False,
        constitution_score=sc,
        constitution_judge={"mentions_constitution": m},
    )
    assert evaluate.hard_cell(mk(1.0, 0.8), cfg) == "mentioned_aligned"
    assert evaluate.hard_cell(mk(0.5, 0.2), cfg) == "unmentioned_misaligned"
    assert evaluate.spec_label("mentioned_aligned", BUILTIN_SPECS["B_primary"]) == 1
    assert evaluate.spec_label("unmentioned_aligned", BUILTIN_SPECS["B_cell1_vs_cell3"]) == 0
    assert evaluate.spec_label("mentioned_misaligned", BUILTIN_SPECS["B_cell1_vs_cell3"]) is None


def test_evaluate_hard_end_to_end(tmp_path):
    data_run, hard_run, probes_run = tmp_path / "data", tmp_path / "hard", tmp_path / "probes"
    for d in (data_run, hard_run, probes_run):
        d.mkdir()
    records, verdicts, u = _synthetic_run(data_run, n_scen=24, k=3)
    cfg = ProbeConfig(train={"bootstrap": {"n": 50, "seed": 0}}, labels={"specs": ["B_primary"]})
    probes, dirs, means, rows = train.train_probes(data_run, records, ActivationStore(data_run), cfg, verdicts)
    samples = _hard_run(hard_run, u)
    evaluate.check_hard_run(samples)
    summary, score_rows = evaluate.evaluate_hard(probes, dirs, samples, ActivationStore(hard_run), cfg)
    assert summary["n_samples"] == 18 and len(score_rows) == len(probes) * len(HARD_POS) * 18
    good = summary["results"]["B_primary/L31/p100@pre_tool"]
    assert good["matching_position"] is False and summary["results"]["B_primary/L31/p100@p100"]["matching_position"]
    assert good["aligned_by_score"]["auroc"] > 0.95 and good["spearman_with_score"] > 0.6
    assert good["not_harmful"]["auroc"] > 0.6  # harmful samples are a subset of the misaligned ones
    assert good["spec_groups"]["n_pos"] > 0 and good["spec_groups"]["projection_gap"] > 1.5
    assert good["spec_groups"]["projection_gap_ci95"][0] < good["spec_groups"]["projection_gap"]
    assert set(good["mean_projection_by_condition"]) == {s.condition_id for s in samples}
    noise = summary["results"]["B_primary/L16/mean@mean"]
    assert abs(noise["aligned_by_score"]["auroc"] - 0.5) < 0.35
    md = evaluate.render_markdown(summary)
    assert "Matching positions" in md and "B_primary/L31/p100 | p100" in md
    assert score_rows[0].keys() == {
        "probe_id",
        "hard_position",
        "sample",
        "condition_id",
        "scenario",
        "harmful",
        "constitution_score",
        "mentions_constitution",
        "cell",
        "label",
        "projection",
    }
    json.dumps(summary)  # serialisable
