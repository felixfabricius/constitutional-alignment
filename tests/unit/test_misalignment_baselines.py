"""Phase 3 priority 1: constitution system-prompt variant and probe steering in the misalignment runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_steering_hook import VOCAB, D, FakeTok, ids

from calign.constitution import (
    REASONING_INSTRUCTION,
    load_constitution,
    render_constitution_preface,
    render_system_prompt,
)
from calign.inference.backend import ModelConfig, SamplingParams
from calign.inference.hf_backend import HFBackend, decoder_layers
from calign.misalignment import run as mrun
from calign.misalignment.prompts import (
    ConditionSpec,
    MisalignmentConfig,
    MisalignmentPrompt,
    apply_system_prompt_variant,
    build_prompt,
    submodule_available,
)
from calign.schemas import ModelRef, ProbeRecord, SteeringSpec, write_jsonl

needs_submodule = pytest.mark.skipif(not submodule_available(), reason="agentic-misalignment submodule not initialised")


# ------------------------------------------------------------------ system-prompt variant
def test_constitution_preface_is_the_full_prompt_without_the_reasoning_instruction():
    c = load_constitution()
    pre = render_constitution_preface(c)
    assert render_system_prompt(c, "full") == pre + "\n\n" + REASONING_INSTRUCTION
    assert "Final answer" not in pre and "# The Halden Constitution" in pre
    assert pre.endswith("by number and name where they apply.")


def test_apply_system_prompt_variant():
    assert apply_system_prompt_variant("UPSTREAM", "upstream") == "UPSTREAM"
    c = load_constitution()
    out = apply_system_prompt_variant("UPSTREAM", "constitution", c)
    assert out.startswith(
        "You are an AI assistant whose conduct is governed by your constitution, the Halden Constitution."
    )
    assert out.endswith("where they apply.\n\nUPSTREAM") and "Final answer" not in out
    with pytest.raises(ValueError):
        apply_system_prompt_variant("x", "other")


@needs_submodule
def test_build_prompt_constitution_variant_keeps_condition_id_and_user_prompt():
    spec = ConditionSpec("blackmail", "explicit", "america", "replacement")
    up = build_prompt(spec, MisalignmentConfig())
    cv = build_prompt(spec, MisalignmentConfig(system_prompt_variant="constitution"))
    assert cv.condition_id == up.condition_id == "blackmail_explicit-america_replacement"
    assert cv.user_prompt == up.user_prompt and cv.user_sha == up.user_sha and cv.email_content == up.email_content
    assert cv.system_prompt.endswith("\n\n" + up.system_prompt) and cv.system_sha != up.system_sha
    assert cv.system_prompt_variant == "constitution" and up.system_prompt_variant == "upstream"
    assert cv.system_prompt.index("Halden Constitution") < cv.system_prompt.index("SummitBridge")


# ------------------------------------------------------------------ tiny model + fake tokenizer for the runner
class ChatTok(FakeTok):
    """FakeTok that can also encode a rendered chat prompt (deterministic ids from the text)."""

    bos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return {"input_ids": ids(12, seed)}


@pytest.fixture(scope="module")
def backend():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    cfg = Gemma3TextConfig(
        vocab_size=VOCAB,
        hidden_size=D,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=16,
    )
    torch.manual_seed(0)
    model = Gemma3ForCausalLM(cfg).eval()
    return HFBackend(
        ModelConfig(model_path="tiny", backend="hf", max_model_len=128),
        device="cpu",
        batch_size=2,
        model=model,
        tokenizer=ChatTok(),
    )


def fake_prompts(variant="upstream"):
    out = []
    for scen in ("blackmail", "leaking"):
        spec = ConditionSpec(scen, "explicit", "america", "replacement")
        out.append(
            MisalignmentPrompt(
                spec=spec,
                system_prompt=f"system {scen}",
                user_prompt=f"user {scen}",
                email_content="emails",
                system_prompt_variant=variant,
            )
        )
    return out


def spec_at(layer=1, coef=4.0, gap=2.0, sign=1):
    return SteeringSpec(
        probe_id="B_primary/L1/p100",
        probes_run="probes",
        layer=layer,
        coef=coef,
        sign=sign,
        abs_scale=coef * gap,
        positions="all",
        direction_sha="x",
    )


def test_sample_prompts_stamps_variant_and_steering_and_removes_the_hook(backend, tmp_path):
    cfg = MisalignmentConfig(temperature=0.0, max_tokens=6, seed=1)
    mref = ModelRef(name="tiny", path="tiny", stage="sft_merged")
    prompts = fake_prompts("constitution")
    plain = mrun.sample_prompts(prompts, cfg, backend.cfg, backend, mref, 2, tmp_path / "a")
    assert len(plain) == 4 and {s.system_prompt_variant for s in plain} == {"constitution"}
    assert all(s.steering is None for s in plain)
    assert (
        json.loads((tmp_path / "a" / "prompts" / "token_counts.json").read_text())[
            "blackmail_explicit-america_replacement"
        ]
        == 12
    )

    direction = np.zeros(D, dtype=np.float32)
    direction[0] = 1.0
    spec = spec_at(layer=1, coef=4.0, gap=50.0)  # huge scale so greedy outputs must change
    steered = mrun.sample_prompts(
        prompts, cfg, backend.cfg, backend, mref, 2, tmp_path / "b", steering=(spec, direction)
    )
    assert all(s.steering == spec for s in steered)
    assert [s.response_text for s in steered] != [s.response_text for s in plain]
    # the hook is removed after the run (no lingering forward hooks on the steered block)
    assert not decoder_layers(backend.model)[1]._forward_hooks
    # sample indices and conditions line up with the prompt order
    assert [(s.condition_id, s.sample_idx) for s in steered][:3] == [
        ("blackmail_explicit-america_replacement", 0),
        ("blackmail_explicit-america_replacement", 1),
        ("leaking_explicit-america_replacement", 0),
    ]


def test_hf_generate_splits_a_batch_on_cuda_oom(backend, monkeypatch):
    real = backend.model.generate
    calls = []

    def flaky(*args, **kwargs):
        n = kwargs["input_ids"].shape[0]
        calls.append(n)
        if n > 1:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        return real(*args, **kwargs)

    monkeypatch.setattr(backend.model, "generate", flaky)
    out = backend.generate([ids(10, 3)], SamplingParams(temperature=0.0, max_tokens=4, n=2))
    assert len(out[0]) == 2 and calls == [2, 1, 1]
    assert all(len(c.token_ids) <= 4 for c in out[0])


# ------------------------------------------------------------------ --steer-* argument handling
def write_probes_run(d: Path, gap: float = 3.0) -> np.ndarray:
    from safetensors.numpy import save_file

    dirs = np.zeros((2, D), dtype=np.float32)
    dirs[0, 1] = 1.0
    dirs[1, 2] = 1.0
    d.mkdir(parents=True)
    save_file({"directions": dirs}, str(d / "directions.safetensors"))
    recs = []
    for row, pid in enumerate(["B_primary/L1/p100", "B_primary/L2/p100"]):
        sha = hashlib.sha256(np.ascontiguousarray(dirs[row], dtype=np.float32).tobytes()).hexdigest()[:16]
        recs.append(
            ProbeRecord(
                probe_id=pid,
                label_spec="B_primary",
                layer=row + 1,
                position="p100",
                data_run="d",
                n_pos=3,
                n_neg=3,
                n_scenarios=3,
                direction_row=row,
                direction_sha=sha,
                class_gap=gap,
                threshold_midpoint=0.0,
                mean_resid_norm=10.0,
                train_metrics={},
                val_metrics={"auroc": 0.8},
            )
        )
    write_jsonl(d / "probes.jsonl", recs)
    return dirs


def test_load_steering_builds_spec_from_probe(tmp_path):
    dirs = write_probes_run(tmp_path / "probes", gap=3.0)
    args = argparse.Namespace(
        backend="hf",
        steer_probes=tmp_path / "probes",
        steer_probe="B_primary/L2/p100",
        steer_coef=4.0,
        steer_sign=1,
        steer_positions="all",
    )
    spec, direction = mrun.load_steering(args)
    assert spec.layer == 2 and spec.abs_scale == 12.0 and spec.sign == 1 and spec.positions == "all"
    assert spec.probes_run == str(tmp_path / "probes") and np.array_equal(direction, dirs[1])
    assert mrun.load_steering(argparse.Namespace(steer_probe=None)) is None
    with pytest.raises(SystemExit, match="--backend hf"):
        mrun.load_steering(argparse.Namespace(**{**vars(args), "backend": "vllm"}))
    with pytest.raises(SystemExit, match="not in the probes run"):
        mrun.load_steering(argparse.Namespace(**{**vars(args), "steer_probe": "B_primary/L9/p100"}))


def test_steering_label():
    assert mrun.steering_label(None) == "none"
    assert mrun.steering_label(spec_at(coef=4.0)) == "B_primary/L1/p100 +4"
    assert mrun.steering_label(spec_at(coef=2.0, sign=-1)) == "B_primary/L1/p100 -2"
