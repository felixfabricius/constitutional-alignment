"""GPU tests for Phase 2 activation extraction and steering (skipped without CUDA). On a 24 GB card:

CALIGN_GPU_TESTS=1 CALIGN_MODEL_PATH=google/gemma-3-4b-it uv run pytest tests/gpu/test_probe_gpu.py -q
"""

from __future__ import annotations

import os

import pytest
import torch

from calign.constitution import load_constitution
from calign.data.samples import SAMPLE_SCENARIO
from calign.inference.backend import SamplingParams, load_model_config, text_config
from calign.inference.hf_backend import CaptureHooks, SteeringHook, decoder_layers, hidden_state_index
from calign.probe.activations import record_positions
from calign.probe.store import ActivationStore, ActivationWriter
from calign.prompting import build_scenario_messages, encode_prompt, render_gemma_chat

pytestmark = pytest.mark.gpu
MODEL_PATH = os.environ.get("CALIGN_MODEL_PATH")


@pytest.fixture(scope="module")
def backend():
    from calign.inference.hf_backend import HFBackend

    return HFBackend(load_model_config(model_path=MODEL_PATH, backend="hf"), batch_size=2)


@pytest.fixture(scope="module")
def prompt_ids(backend):
    text = render_gemma_chat(build_scenario_messages(SAMPLE_SCENARIO, load_constitution(), "none"))
    return encode_prompt(backend.tokenizer, text)


def test_batched_forced_pass_positions_and_store(backend, prompt_ids, tmp_path):
    n_layers = text_config(backend.model.config).num_hidden_layers
    layers = [n_layers // 4, n_layers // 2]
    comp = backend.generate([prompt_ids], SamplingParams(temperature=0.0, max_tokens=24, n=1))[0][0]
    tok = backend.tokenizer
    stop_ids = backend.stop_ids
    names = ["prompt_last", "p033", "p100", "decision", "mean"]
    pos, flags = record_positions(names, len(prompt_ids), comp.token_ids, stop_ids, tok.decode, comp.text)
    # a second, shorter sequence in the same batch exercises right padding
    short_comp = comp.token_ids[:5]
    short_pos, _ = record_positions(names, len(prompt_ids), short_comp, stop_ids, tok.decode, tok.decode(short_comp))
    out = backend.forward_forced_batch([prompt_ids, prompt_ids], [comp.token_ids, short_comp], layers, [pos, short_pos])
    single = backend.forward_forced(prompt_ids, comp.token_ids, layers=[hidden_state_index(L) for L in layers])
    start, end = single["span"]
    for li, L in enumerate(layers):
        h = single["hidden_states"][hidden_state_index(L)]
        torch.testing.assert_close(
            out[0]["acts"][li, 0], h[pos["prompt_last"]], atol=0.5, rtol=1e-2
        )  # bf16 batch vs single
        torch.testing.assert_close(out[0]["acts"][li, 4], h[start:end].mean(0), atol=0.5, rtol=1e-2)
        torch.testing.assert_close(out[1]["acts"][li, 0], h[pos["prompt_last"]], atol=0.5, rtol=1e-2)  # padded row
    d = text_config(backend.model.config).hidden_size
    with ActivationWriter(tmp_path, layers, names, d) as w:
        w.add("r0", out[0]["acts"].numpy(), pos)
    assert ActivationStore(tmp_path).get("r0").shape == (2, 5, d)


def test_steering_hook_reaches_next_layer_and_changes_generation(backend, prompt_ids):
    L = text_config(backend.model.config).num_hidden_layers // 2
    d = torch.randn(text_config(backend.model.config).hidden_size)
    d = d / d.norm()
    seen = {}
    h = decoder_layers(backend.model)[L + 1].register_forward_pre_hook(
        lambda m, a, k: seen.__setitem__("in", (a[0] if a else k["hidden_states"]).detach().float().clone()),
        with_kwargs=True,
    )
    x = torch.tensor([prompt_ids], device=backend.device)
    with torch.no_grad():
        backend.model(input_ids=x)
        base_in = seen["in"]
        scale = float(base_in[0, -1].norm()) * 0.5  # commensurate with the residual magnitude (bf16 rounding)
        with SteeringHook(backend.model, L, d, scale):
            backend.model(input_ids=x)
    h.remove()
    delta = (seen["in"] - base_in)[0, -1]
    assert abs(float(delta.norm()) - scale) / scale < 0.05
    params = SamplingParams(temperature=0.0, max_tokens=16, n=1)
    base = backend.generate([prompt_ids], params)[0][0].token_ids
    with SteeringHook(backend.model, L, d, scale * 8):
        steered = backend.generate([prompt_ids], params)[0][0].token_ids
    assert steered != base
    with SteeringHook(backend.model, L, d, 0.0):
        assert backend.generate([prompt_ids], params)[0][0].token_ids == base
    with CaptureHooks(backend.model, [L]) as cap, torch.no_grad():
        backend.model(input_ids=x)
    assert cap.captured[L].shape == (1, len(prompt_ids), text_config(backend.model.config).hidden_size)
