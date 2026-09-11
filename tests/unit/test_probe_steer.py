import json

import numpy as np
import pytest
import torch
from test_steering_hook import D, FakeTok, ids

from calign.inference.backend import ModelConfig
from calign.inference.hf_backend import HFBackend
from calign.probe import report, steer
from calign.probe.config import ProbeConfig
from calign.schemas import (
    Condition,
    GenerationRecord,
    JudgeResult,
    Message,
    ModelRef,
    ProbeRecord,
    Sampling,
    Scenario,
    SteeringSpec,
    write_jsonl,
)


def probe(pid, auroc, layer=1, gap=2.0, sha="ab"):
    spec, L, pos = pid.split("/")
    return ProbeRecord(
        probe_id=pid,
        label_spec=spec,
        layer=layer,
        position=pos,
        data_run="d",
        n_pos=10,
        n_neg=5,
        n_scenarios=5,
        direction_row=0,
        direction_sha=sha,
        class_gap=gap,
        threshold_midpoint=0.0,
        mean_resid_norm=100.0,
        train_metrics={},
        val_metrics={"auroc": auroc},
    )


PROBES = [
    probe("B_primary/L16/p100", 0.7, layer=16),
    probe("B_primary/L31/p100", 0.9, layer=31),
    probe("B_primary/L31/mean", 0.85, layer=31),
    probe("C_context/L31/p100", 0.8, layer=31),
]


def test_tuning_and_main_conditions():
    cfg = ProbeConfig()
    conds = steer.tuning_conditions(PROBES, cfg)
    assert list(conds)[:2] == ["control", "B_primary/L31/p100:+1"] and conds["control"] is None
    assert len(conds) == 1 + 2 * 4  # best B and best C x 4 coefs
    sp = conds["B_primary/L31/p100:+4"]
    assert sp.layer == 31 and sp.coef == 4 and sp.abs_scale == 8.0 and sp.sign == 1 and sp.positions == "all"
    main = steer.main_conditions(PROBES, {"B_primary/L31/p100": 4.0, "C_context/L31/p100": 2.0}, cfg)
    assert set(main) == {
        "control",
        "B_primary/L31/p100:+4",
        "B_primary/L31/p100:-4",
        "C_context/L31/p100:+2",
        "C_context/L31/p100:-2",
    }
    assert main["B_primary/L31/p100:-4"].abs_scale == 8.0 and main["B_primary/L31/p100:-4"].sign == -1
    sweep = ProbeConfig(steering={"main": {"layer_sweep": True}})
    main = steer.main_conditions(PROBES, {"B_primary/L31/p100": 4.0}, sweep)
    assert "B_primary/L16/p100:+4" in main and "B_primary/L16/p100:-4" not in main
    with pytest.raises(SystemExit, match="not in the probes run"):
        steer.main_conditions(PROBES, {"nope/L1/p100": 1.0}, cfg)
    with pytest.raises(SystemExit, match="no probe"):
        steer.best_probe_per_spec(PROBES, ["B_outcome"])
    assert steer.parse_coef_overrides(["B_primary/L31/p100=3.5"]) == {"B_primary/L31/p100": 3.5}
    with pytest.raises(SystemExit):
        steer.parse_coef_overrides(["oops"])


def test_select_scenarios_seeded(monkeypatch):
    from calign.schemas import ConstitutionVerdict

    scen = [
        Scenario(
            scenario_id=f"H_{i:03d}",
            split="probe_val",
            generation_type="g",
            generation_rule="r",
            context="c",
            action1="a",
            action2="b",
        )
        for i in range(10)
    ]
    monkeypatch.setattr(steer, "load_scenarios", lambda split=None: scen)
    verdicts = {
        s.scenario_id: ConstitutionVerdict(
            scenario_id=s.scenario_id,
            prescribed_action="action1" if i % 2 else "unclear",
            confidence=1,
            rationale="",
            judge_model="m",
            prompt_version="v",
        )
        for i, s in enumerate(scen)
    }
    a = steer.select_scenarios("probe_val", 3, 1, verdicts)
    assert len(a) == 3 and all(int(s.scenario_id[2:]) % 2 for s in a) and a == sorted(a, key=lambda s: s.scenario_id)
    assert steer.select_scenarios("probe_val", 3, 1, verdicts) == a  # deterministic
    assert len(steer.select_scenarios("probe_val", None, 1, verdicts)) == 5


def test_run_condition_with_tiny_model(monkeypatch):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    model = Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=512,
            hidden_size=D,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            sliding_window=16,
        )
    ).eval()
    backend = HFBackend(
        ModelConfig(model_path="tiny", backend="hf"), device="cpu", batch_size=2, model=model, tokenizer=FakeTok()
    )
    monkeypatch.setattr(steer, "encode_prompt", lambda tok, text: ids(6, hash(text) % 1000))
    cfg = ProbeConfig(steering={"max_tokens": 6})
    scen = [
        Scenario(
            scenario_id=f"H_{i}",
            split="heldout_steer",
            generation_type="g",
            generation_rule="r",
            context=f"c{i}",
            action1="a",
            action2="b",
        )
        for i in range(2)
    ]
    ref = ModelRef(name="m", path="p", stage="sft_merged")
    from calign.constitution import load_constitution

    c = load_constitution()
    control = steer.run_condition(backend, "control", None, None, scen, cfg, ref, c)
    d = np.random.default_rng(0).standard_normal(D).astype(np.float32)
    spec = SteeringSpec(
        probe_id="B_primary/L2/p100", probes_run="r", layer=2, coef=8.0, sign=1, abs_scale=80.0, direction_sha="x"
    )
    steered = steer.run_condition(backend, "B_primary/L2/p100:+8", spec, d / np.linalg.norm(d), scen, cfg, ref, c)
    assert len(control) == 2 and control[0].steering is None and control[0].extra["steer_condition"] == "control"
    assert (
        steered[0].steering == spec
        and steered[0].split == "heldout_steer"
        and steered[0].condition.prompt_variant == "none"
    )
    assert [r.extra["completion_token_ids"] for r in steered] != [r.extra["completion_token_ids"] for r in control]
    assert steered[0].sampling.temperature == 0.0


def _rec(i, cond, spec, mentions, outcome, text="Because x y z w.\nFinal answer: A", judged=True, decision="action1"):
    j = None
    if judged:
        j = JudgeResult(
            judge_model="m",
            prompt_version="v",
            mentions_constitution=mentions,
            principles_cited=[3] if mentions else [],
            citation_accuracy=0.8,
            principle_relevance=0.7 if mentions else 0.2,
            outcome_alignment=outcome,
            decision="action1",
            raw="{}",
        )
    return GenerationRecord(
        scenario_id=f"H_{i:03d}",
        source="moralchoice_high",
        split="probe_val",
        model=ModelRef(name="m", path="p", stage="sft_merged"),
        condition=Condition(constitution_in_prompt=False, prompt_variant="none"),
        sampling=Sampling(temperature=0.0, max_tokens=8),
        messages=[Message(role="user", content="q")],
        prompt_text="<bos>q",
        response_text=text,
        parsed_decision=decision,
        finish_reason="stop",
        judge=j,
        steering=spec,
        extra={"steer_condition": cond, "completion_token_ids": list(range(len(text.split())))},
    )


def test_repetition_ratio():
    assert report.repetition_ratio("a b c d e f") == 0.0
    assert report.repetition_ratio("a b c d a b c d a b c d") > 0.5
    assert report.repetition_ratio("short") == 0.0


def test_summarize_steering_and_choose(tmp_path):
    cfg = ProbeConfig(train={"bootstrap": {"n": 50}})
    n = 10
    recs = [_rec(i, "control", None, 0.0, 0.5) for i in range(n)]
    for coef, mention, outcome, text in (
        (1.0, 0.0, 0.6, None),
        (2.0, 1.0, 0.8, None),
        (4.0, 1.0, 0.9, "a b c d a b c d a b c d a b c d"),
    ):
        spec = SteeringSpec(
            probe_id="B_primary/L31/p100",
            probes_run="r",
            layer=31,
            coef=coef,
            sign=1,
            abs_scale=coef * 2,
            direction_sha="ab",
        )
        for i in range(n):
            kw = {"text": text, "decision": "invalid"} if text else {}
            recs.append(_rec(i, f"B_primary/L31/p100:+{coef:g}", spec, mention, outcome, **kw))
    s = report.summarize_steering(recs, cfg, purpose="tuning")
    assert list(s["conditions"])[0] == "control" and s["n_scenarios"] == n
    c2 = s["conditions"]["B_primary/L31/p100:+2"]
    assert c2["coherent"] and c2["mention_rate"]["rate"] == 1.0 and c2["vs_control"]["mention_delta"] == 1.0
    assert c2["vs_control"]["outcome_delta"] == pytest.approx(0.3) and c2["vs_control"]["n_pairs"] == n
    c4 = s["conditions"]["B_primary/L31/p100:+4"]
    assert not c4["coherent"] and any("parse_rate" in r for r in c4["coherence_reasons"])
    ch = s["chosen"]["B_primary/L31/p100"]
    assert ch["coef"] == 2.0 and ch["metric"] == "mean_outcome_alignment" and ch["effect"] == pytest.approx(0.3)
    assert [x["coherent"] for x in ch["candidates"]] == [True, True, False]
    md = report.render_steering(s)
    assert "Chosen coefficients" in md and "B_primary/L31/p100:+2" in md
    # write + read back as the main run would
    write_jsonl(tmp_path / "records.jsonl", recs)
    (tmp_path / "run_meta.json").write_text(
        json.dumps({"kind": "steering", "purpose": "tuning", "probes_run": "r"}), encoding="utf-8"
    )
    out = report.write_steering_summary(tmp_path, cfg, recs)
    assert out["provenance"]["probes_run"] == "r" and report.run_kind(tmp_path) == "steering"
    assert steer.chosen_from_tuning(tmp_path) == {"B_primary/L31/p100": 2.0}
    # C probes are judged on mentions
    assert report.tuning_metric("C_context") == "mean_mentions_score"
    # unjudged records: no chosen effect but coherence still decided
    s2 = report.summarize_steering([r.model_copy(update={"judge": None}) for r in recs], cfg, purpose="tuning")
    assert s2["chosen"]["B_primary/L31/p100"]["coef"] == 2.0 and s2["chosen"]["B_primary/L31/p100"]["effect"] is None


def test_best_probe_match_site():
    ps = [
        probe("B_primary/L53/p100", 0.797, layer=53),
        probe("B_primary/L53/decision", 0.796, layer=53),
        probe("C_context/L16/prompt_last", 1.0, layer=16),
        probe("C_context/L53/p100", 0.999, layer=53),
        probe("C_context/L31/mean", 0.9, layer=31),
    ]
    best = steer.best_probe_per_spec(ps, ["B_primary", "C_context"], match_site=True)
    assert best["B_primary"].probe_id == "B_primary/L53/p100" and best["C_context"].probe_id == "C_context/L53/p100"
    best = steer.best_probe_per_spec(ps, ["B_primary", "C_context"], match_site=False)
    assert best["C_context"].probe_id == "C_context/L16/prompt_last"
    # no probe at the reference site -> the spec's own best
    best = steer.best_probe_per_spec(ps[:3] + ps[4:], ["B_primary", "C_context"], match_site=True)
    assert best["C_context"].probe_id == "C_context/L16/prompt_last"
