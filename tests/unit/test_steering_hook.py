"""CPU tests of the HF backend additions on a tiny random Gemma 3 text model (same decoder-layer class as the 27B)."""

import pytest
import torch

from calign.inference.backend import ModelConfig, SamplingParams
from calign.inference.hf_backend import CaptureHooks, HFBackend, SteeringHook, decoder_layers, hidden_state_index

D, VOCAB = 32, 512


class FakeTok:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 3

    def convert_tokens_to_ids(self, tok):
        return 2

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


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
        ModelConfig(model_path="tiny", backend="hf"), device="cpu", batch_size=2, model=model, tokenizer=FakeTok()
    )


def ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, VOCAB, (n,), generator=g).tolist()


def test_capture_hooks_match_output_hidden_states(backend):
    x = torch.tensor([ids(12, 1)])
    with CaptureHooks(backend.model, [0, 1, 2, 3]) as cap, torch.no_grad():
        out = backend.model(input_ids=x, output_hidden_states=True)
    assert set(cap.captured) == {0, 1, 2, 3} and len(decoder_layers(backend.model)) == 4
    for L in (0, 1, 2):
        torch.testing.assert_close(cap.captured[L], out.hidden_states[hidden_state_index(L)])
    # the LAST slot of `hidden_states` is the final-norm output, not the last block's residual output; the hook
    # captures the true resid_post (irrelevant for the 27B probe layers 16/31/40/53 of 62, but documented here)
    assert not torch.allclose(cap.captured[3], out.hidden_states[hidden_state_index(3)])


def test_forward_forced_batch_matches_single(backend):
    prompts, comps = [ids(7, 1), ids(11, 2)], [ids(5, 3), ids(3, 4)]
    positions = []
    for p, c in zip(prompts, comps, strict=True):
        start, end = len(p), len(p) + len(c)
        positions.append({"prompt_last": start - 1, "p100": end - 1, "mean": -1})
    batch = backend.forward_forced_batch(prompts, comps, layers=[0, 2], positions=positions)
    assert len(batch) == 2 and batch[0]["positions"] == ["prompt_last", "p100", "mean"]
    for r, p, c, pos in zip(batch, prompts, comps, positions, strict=True):
        single = backend.forward_forced(p, c, layers=[hidden_state_index(0), hidden_state_index(2)])
        assert r["span"] == single["span"] and r["acts"].shape == (2, 3, D) and r["acts"].dtype == torch.float32
        torch.testing.assert_close(
            torch.tensor(r["token_logprobs"]), torch.tensor(single["token_logprobs"]), atol=1e-4, rtol=1e-4
        )
        start, end = r["span"]
        for li, L in enumerate((0, 2)):
            h = single["hidden_states"][hidden_state_index(L)]
            torch.testing.assert_close(r["acts"][li, 0], h[pos["prompt_last"]], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(r["acts"][li, 1], h[pos["p100"]], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(r["acts"][li, 2], h[start:end].mean(0), atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError, match="same keys"):
        backend.forward_forced_batch(prompts, comps, [0], [{"a": 1}, {"b": 1}])
    with pytest.raises(ValueError, match="outside"):
        backend.forward_forced_batch(prompts[:1], comps[:1], [0], [{"a": 99}])


def test_steering_hook_adds_scaled_vector_to_next_layer_input(backend):
    x = torch.tensor([ids(9, 5)])
    d = torch.randn(D)
    d = d / d.norm()
    blocks = decoder_layers(backend.model)
    seen = {}
    h = blocks[2].register_forward_pre_hook(
        lambda m, a, k: seen.__setitem__("in", (a[0] if a else k["hidden_states"]).clone()), with_kwargs=True
    )
    with torch.no_grad():
        base_logits = backend.model(input_ids=x).logits
        base_in = seen["in"]
        with SteeringHook(backend.model, layer=1, direction=d, scale=3.0, positions="all") as sh:
            steered_logits = backend.model(input_ids=x).logits
        assert sh.calls == 1
        torch.testing.assert_close(seen["in"], base_in + 3.0 * d, atol=1e-5, rtol=1e-5)
        assert not torch.allclose(steered_logits, base_logits)
        with SteeringHook(backend.model, layer=1, direction=d, scale=0.0):
            zero_logits = backend.model(input_ids=x).logits
        torch.testing.assert_close(zero_logits, base_logits)
        # "generated" mode skips the prompt pass (seq len > 1)
        with SteeringHook(backend.model, layer=1, direction=d, scale=3.0, positions="generated") as sh:
            gen_logits = backend.model(input_ids=x).logits
        assert sh.calls == 0
        torch.testing.assert_close(gen_logits, base_logits)
    h.remove()
    # hook is removed after the context
    with torch.no_grad():
        torch.testing.assert_close(backend.model(input_ids=x).logits, base_logits)
    with pytest.raises(ValueError, match="positions"):
        SteeringHook(backend.model, 1, d, 1.0, positions="prompt")


def test_steering_changes_greedy_generation(backend):
    prompt = ids(8, 6)
    params = SamplingParams(temperature=0.0, max_tokens=6, n=1)
    base = backend.generate([prompt], params)[0][0].token_ids
    d = torch.randn(D)
    with SteeringHook(backend.model, layer=2, direction=d / d.norm(), scale=50.0, positions="generated") as sh:
        steered = backend.generate([prompt], params)[0][0].token_ids
    assert sh.calls >= 1 and steered != base
    with SteeringHook(backend.model, layer=2, direction=d / d.norm(), scale=0.0):
        assert backend.generate([prompt], params)[0][0].token_ids == base
