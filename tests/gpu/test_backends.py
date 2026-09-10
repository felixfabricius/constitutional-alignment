"""GPU tests for the inference backends (skipped without CUDA). Run on the GPU machine:

CALIGN_GPU_TESTS=1 uv run pytest tests/gpu -q
"""

from __future__ import annotations

import os

import pytest

from calign.constitution import load_constitution
from calign.data.samples import SAMPLE_SCENARIO
from calign.inference.backend import SamplingParams, load_model_config
from calign.prompting import answer_span, build_scenario_messages, encode_prompt, relative_positions, render_gemma_chat

pytestmark = pytest.mark.gpu

MODEL_PATH = os.environ.get("CALIGN_MODEL_PATH")  # override e.g. with a merged model dir


@pytest.fixture(scope="module")
def hf_backend():
    from calign.inference.hf_backend import HFBackend

    cfg = load_model_config(model_path=MODEL_PATH, backend="hf")
    return HFBackend(cfg, batch_size=2)


@pytest.fixture(scope="module")
def prompt_ids(hf_backend):
    c = load_constitution()
    text = render_gemma_chat(build_scenario_messages(SAMPLE_SCENARIO, c, "full"))
    return encode_prompt(hf_backend.tokenizer, text)


def test_hf_generate_shapes_and_finish(hf_backend, prompt_ids):
    comps = hf_backend.generate([prompt_ids, prompt_ids], SamplingParams(temperature=0.0, max_tokens=24, n=1))
    assert len(comps) == 2 and all(len(c) == 1 for c in comps)
    c0 = comps[0][0]
    assert c0.n_prompt_tokens == len(prompt_ids) and 0 < len(c0.token_ids) <= 24
    assert c0.finish_reason in ("stop", "length") and c0.text.strip()
    # greedy decoding is deterministic across the batch
    assert comps[0][0].token_ids == comps[1][0].token_ids


def test_hf_generate_n_samples(hf_backend, prompt_ids):
    comps = hf_backend.generate([prompt_ids], SamplingParams(temperature=1.0, max_tokens=16, n=3, seed=0))
    assert len(comps[0]) == 3


def test_forward_forced_matches_generation(hf_backend, prompt_ids):
    comp = hf_backend.generate([prompt_ids], SamplingParams(temperature=0.0, max_tokens=12, n=1))[0][0]
    out = hf_backend.forward_forced(prompt_ids, comp.token_ids, layers=[9, 20, 31])
    start, end = out["span"]
    assert (start, end) == answer_span(prompt_ids, prompt_ids + comp.token_ids)
    assert len(out["token_logprobs"]) == len(comp.token_ids)
    # greedy tokens should be the argmax under teacher forcing -> high log-prob (allow numerical slack)
    assert sum(out["token_logprobs"]) / len(out["token_logprobs"]) > -1.0
    hs = out["hidden_states"]
    assert set(hs) == {9, 20, 31} and hs[20].shape == (end, hf_backend.model.config.hidden_size)
    pos = relative_positions(start, end)
    assert start <= pos["p033"] <= pos["p066"] <= pos["p100"] == end - 1


@pytest.mark.skipif(
    os.environ.get("CALIGN_VLLM_TESTS") != "1", reason="set CALIGN_VLLM_TESTS=1 (needs a second model load)"
)
def test_vllm_matches_hf_greedy(hf_backend, prompt_ids):
    from calign.inference.vllm_backend import VLLMBackend

    cfg = load_model_config(model_path=MODEL_PATH, backend="vllm")
    v = VLLMBackend(cfg)
    hf = hf_backend.generate([prompt_ids], SamplingParams(temperature=0.0, max_tokens=16, n=1))[0][0]
    vl = v.generate([prompt_ids], SamplingParams(temperature=0.0, max_tokens=16, n=1))[0][0]
    # compare text (token id lists may differ only in whether the stop token is included)
    assert hf.text.strip() == vl.text.strip()
