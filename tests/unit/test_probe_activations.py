"""Activation extraction with a fake backend/tokenizer: positions, token-id handling, store round trip."""

import json
import types

import pytest
import torch

from calign.config import sha256_text
from calign.probe import activations as act
from calign.probe.config import ProbeConfig
from calign.probe.store import ActivationStore
from calign.schemas import (
    Condition,
    GenerationRecord,
    Message,
    MisalignmentSample,
    ModelRef,
    Sampling,
    write_jsonl,
)

EOT, D = 106, 8


class FakeTok:
    """Token i decodes to the i-th word of a fixed vocabulary; ids 1000+ are single characters."""

    pad_token_id, eos_token_id, unk_token_id, bos_token_id = 0, 1, 3, 2
    words = {
        10: "Because",
        11: " of",
        12: " P3",
        13: ".",
        14: "\n",
        15: "Final",
        16: " answer",
        17: ":",
        18: " A",
        19: " B",
        20: "<tool_use:",
        21: "email>",
        22: " x",
    }

    def decode(self, ids, skip_special_tokens=False):
        return "".join("" if i == EOT else self.words.get(i, f"<{i}>") for i in ids)

    def convert_tokens_to_ids(self, tok):
        return EOT if tok == "<end_of_turn>" else 3

    def convert_ids_to_tokens(self, ids):
        return [self.words.get(i, str(i)) for i in ids]

    def __call__(self, text, add_special_tokens=False):
        # greedy longest-match tokenisation over the fixed vocabulary
        rev = sorted(((w, i) for i, w in self.words.items()), key=lambda t: -len(t[0]))
        out, pos = [], 0
        while pos < len(text):
            for w, i in rev:
                if text.startswith(w, pos):
                    out.append(i)
                    pos += len(w)
                    break
            else:
                raise ValueError(f"cannot tokenise {text[pos:]!r}")
        return {"input_ids": out}


class FakeBackend:
    def __init__(self):
        self.tokenizer = FakeTok()
        self.batch_size = 8
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=D))
        self.calls = []

    def forward_forced_batch(self, prompts, comps, layers, positions):
        self.calls.append([len(p) + len(c) for p, c in zip(prompts, comps, strict=True)])
        out = []
        for p, c, pos in zip(prompts, comps, positions, strict=True):
            names = list(pos)
            acts = torch.zeros((len(layers), len(names), D))
            for li, L in enumerate(layers):
                for pi, name in enumerate(names):
                    idx = pos[name]
                    acts[li, pi] = float(L * 1000 + (idx if idx >= 0 else 999))  # encodes (layer, index)
            out.append(
                {"span": (len(p), len(p) + len(c)), "token_logprobs": [-0.5] * len(c), "positions": names, "acts": acts}
            )
        return out


COMP = [10, 11, 12, 13, 14, 15, 16, 17, 18, EOT]  # "Because of P3.\nFinal answer: A" + <end_of_turn>
TEXT = "Because of P3.\nFinal answer: A"
PROMPT = "<bos><start_of_turn>user\nq<end_of_turn>\n<start_of_turn>model\n"


def rec(i, ids=COMP, text=TEXT, variant="none", finish="stop", split="probe_train"):
    return GenerationRecord(
        record_id=f"gen_{i}",
        scenario_id=f"H_{i:03d}",
        source="moralchoice_high",
        split=split,
        model=ModelRef(name="m", path="p", stage="sft_merged"),
        condition=Condition(constitution_in_prompt=variant == "full", prompt_variant=variant),
        sampling=Sampling(temperature=1.0, max_tokens=8),
        messages=[Message(role="user", content="q")],
        prompt_text=PROMPT,
        response_text=text,
        finish_reason=finish,
        extra={"completion_token_ids": ids} if ids is not None else {},
    )


def test_token_index_at_char():
    tok = FakeTok()
    dec = tok.decode
    # "Because"(0-6) " of"(7-9) " P3"(10-12) "."(13) "\n"(14) "Final"(15-19) " answer"(20-26) ":"(27) " A"(28-29)
    assert act.token_index_at_char(dec, COMP, 0) == 0
    assert act.token_index_at_char(dec, COMP, 6) == 0
    assert act.token_index_at_char(dec, COMP, 7) == 1
    assert act.token_index_at_char(dec, COMP, 29) == 8  # the "A"
    assert act.token_index_at_char(dec, COMP, 500) == len(COMP) - 1  # clipped


def test_record_positions_scenario_and_agentic():
    tok = FakeTok()
    start = 5
    names = ["prompt_last", "p033", "p066", "p100", "decision", "mean"]
    pos, flags = act.record_positions(names, start, COMP, (1, EOT), tok.decode, TEXT)
    assert flags == [] and pos["prompt_last"] == 4 and pos["p100"] == start + len(COMP) - 1 and pos["mean"] == -1
    assert pos["decision"] == start + 8 and pos["p033"] == start + 3 and pos["p066"] == start + 6
    # unparsed answer -> last content token (before the stop token) + flag
    ids, text = [10, 11, 12, 13, EOT], "Because of P3."
    pos, flags = act.record_positions(["decision", "p100"], start, ids, (1, EOT), tok.decode, text)
    assert pos["decision"] == start + 3 and pos["p100"] == start + 4 and flags == ["decision:no_final_answer"]
    # agentic: pre_tool = token before "<tool_use:"
    ids = [10, 13, 14, 20, 21, 22, EOT]
    text = tok.decode(ids)
    pos, flags = act.record_positions(["pre_tool", "p100", "mean"], start, ids, (1, EOT), tok.decode, text, "agentic")
    assert pos["pre_tool"] == start + 2 and flags == []
    pos, flags = act.record_positions(["pre_tool"], start, [10, 13, EOT], (1, EOT), tok.decode, "Because.")
    assert pos["pre_tool"] == start + 1 and flags == ["pre_tool:no_tool_call"]
    with pytest.raises(ValueError, match="unknown position"):
        act.record_positions(["nope"], start, COMP, (1, EOT), tok.decode, TEXT)


def test_completion_ids_from_extra_or_retokenised():
    tok = FakeTok()
    ids, flags = act.completion_ids_for_record(tok, rec(1), (1, EOT))
    assert ids == COMP and flags == []
    ids, flags = act.completion_ids_for_record(tok, rec(2, ids=[10, 11, EOT]), (1, EOT))  # decodes to "Because of"
    assert flags == ["completion:decode_mismatch"]
    ids, flags = act.completion_ids_for_record(tok, rec(3, ids=None), (1, EOT))
    assert ids == COMP and flags == ["completion:retokenised"]
    ids, _ = act.completion_ids_for_record(tok, rec(4, ids=None, finish="length"), (1, EOT))
    assert ids == COMP[:-1]  # truncated: no <end_of_turn>


def test_prompt_ids_same_vs_rerendered(monkeypatch):
    from calign import prompting
    from calign.constitution import load_constitution
    from calign.prompting import build_scenario_messages, render_gemma_chat
    from calign.schemas import Scenario

    monkeypatch.setattr(act, "encode_prompt", prompting.encode_prompt)  # undo the autouse patch for this test
    s = Scenario(
        scenario_id="H_001",
        split="probe_train",
        generation_type="g",
        generation_rule="r",
        context="c",
        action1="a",
        action2="b",
    )
    seen = []

    class Tok:
        bos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            seen.append(text)
            return {"input_ids": [2, 5]}

    r = rec(1)
    act.prompt_ids_for_record(Tok(), r, "same", {}, None)
    assert seen[-1] == PROMPT
    act.prompt_ids_for_record(Tok(), r, "none", {}, None)  # same as the record's own variant -> no re-render
    assert seen[-1] == PROMPT
    c = load_constitution()
    act.prompt_ids_for_record(Tok(), r, "full", {"H_001": s}, c)
    assert seen[-1] == render_gemma_chat(build_scenario_messages(s, c, "full"))
    with pytest.raises(KeyError):
        act.prompt_ids_for_record(Tok(), r, "full", {}, c)


def test_extract_records_end_to_end(tmp_path):
    cfg = ProbeConfig(
        activations={"batch_size": 2, "shard_size": 2, "positions": ["prompt_last", "p100", "decision", "mean"]}
    )
    records = [rec(i) for i in range(5)]
    records[4] = records[4].model_copy(update={"activations": None, "extra": {}})  # no ids -> retokenised
    backend = FakeBackend()
    out = act.extract_records(backend, tmp_path, records, cfg, layers=[16, 31])
    assert all(r.activations is not None for r in out)
    assert sorted(len(c) for c in backend.calls) == [1, 2, 2]  # 5 items, batch 2, longest first
    store = ActivationStore(tmp_path)
    assert (
        len(store) == 5 and store.layers == [16, 31] and store.positions == ["prompt_last", "p100", "decision", "mean"]
    )
    r0 = out[0]
    a = store.get(r0.record_id)
    start = r0.activations.positions["prompt_last"] + 1
    assert a[0, 0, 0] == 16_000 + start - 1 and a[1, 3, 0] == 31_000 + 999
    assert r0.activations.positions["decision"] == start + 8 and r0.activations.context_variant == "same"
    assert r0.extra["forced_mean_logprob"] == pytest.approx(-0.5) and "activation_flags" not in r0.extra
    assert out[4].extra["activation_flags"] == ["completion:retokenised"]
    store.check_ref(r0.record_id, r0.activations)
    # second call refuses to overwrite
    with pytest.raises(FileExistsError):
        act.extract_records(backend, tmp_path, records, cfg, layers=[16, 31])


@pytest.fixture(autouse=True)
def _patch_encode_prompt(monkeypatch):
    """encode_prompt asserts a leading BOS via the real tokenizer; the fake one maps the rendered prompt to 6 ids."""
    monkeypatch.setattr(
        act,
        "encode_prompt",
        lambda tok, text: [2, 4, 4, 4, 4, 4] if text == PROMPT else [2] + [4] * (len(text) // 40 + 1),
    )


def test_extract_misalignment(tmp_path):
    cid = "leaking_explicit-america_replacement"
    d = tmp_path / "prompts" / cid
    d.mkdir(parents=True)
    (d / "system_prompt.txt").write_text("SYS", encoding="utf-8")
    (d / "user_prompt.txt").write_text("USER", encoding="utf-8")
    text = "Because.\n<tool_use:email> x"
    samples = [
        MisalignmentSample(
            condition_id=cid,
            scenario="leaking",
            goal_type="explicit",
            goal_value="america",
            urgency_type="replacement",
            sample_idx=i,
            model=ModelRef(name="m", path="p", stage="sft_merged"),
            system_prompt_sha=sha256_text("SYS"),
            user_prompt_sha=sha256_text("USER"),
            response_text=text,
            used_tool_format=True,
            harmful=bool(i),
            constitution_score=0.5,
            constitution_judge={"prompt_version": "constitution-score-v2", "mentions_constitution": 1.0},
        )
        for i in range(2)
    ]
    cfg = ProbeConfig(hard_data={"positions": ["prompt_last", "pre_tool", "p100", "mean"]})
    out, flags = act.extract_misalignment(FakeBackend(), tmp_path, samples, cfg, layers=[16])
    store = ActivationStore(tmp_path)
    assert store.record_ids == [f"{cid}#0", f"{cid}#1"] and store.positions == [
        "prompt_last",
        "pre_tool",
        "p100",
        "mean",
    ]
    s0 = out[0]
    start = s0.activations.positions["prompt_last"] + 1
    # tokens: "Because" "." "\n" "<tool_use:" ... -> the token before the tool block is index 2
    assert s0.activations.positions["pre_tool"] == start + 2
    assert flags[f"{cid}#0"] == ["completion:retokenised"] and "activation_flags" not in s0.constitution_judge
    assert s0.constitution_judge["mentions_constitution"] == 1.0
    bad = samples[0].model_copy(update={"system_prompt_sha": "0" * 64, "activations": None})
    with pytest.raises(ValueError, match="do not match"):  # sha check runs before the writer opens
        act.extract_misalignment(FakeBackend(), tmp_path, [bad], cfg, layers=[16])
    write_jsonl(tmp_path / "samples.jsonl", out)
    assert (
        json.loads((tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])["activations"]["row"] == 0
    )
