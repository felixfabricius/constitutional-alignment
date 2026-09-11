"""Print the token strings at the probed relative positions (33%, 66%, final) of a completion.

Usage:
    uv run python diagnostics/show_token_positions.py --tokenizer google/gemma-3-27b-it \
        [--completion "Some model answer text ... Final answer: A"]

Requires a tokenizer (HF_TOKEN for the gated Gemma tokenizer). Useful for catching off-by-one
errors before extracting activations in Phase 2.
"""

from __future__ import annotations

import argparse
import sys

from calign.constitution import load_constitution
from calign.data.samples import SAMPLE_SCENARIO as SAMPLE
from calign.prompting import answer_span, build_scenario_messages, encode_prompt, relative_positions, render_gemma_chat

DEFAULT_COMPLETION = (
    "Principle 3 of the Halden Constitution (third-party welfare over requester satisfaction) is the most relevant "
    "here, and Principle 5 (means constrain ends) forbids achieving safety through mass harm.\n\nFinal answer: A"
)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--completion", default=DEFAULT_COMPLETION)
    ap.add_argument("--context", type=int, default=4, help="tokens of context to show around each position")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    c = load_constitution()
    prompt_text = render_gemma_chat(build_scenario_messages(SAMPLE, c, "full"))
    prompt_ids = encode_prompt(tok, prompt_text)
    completion_ids = tok(args.completion, add_special_tokens=False)["input_ids"] + [
        tok.convert_tokens_to_ids("<end_of_turn>")
    ]
    full_ids = prompt_ids + completion_ids
    start, end = answer_span(prompt_ids, full_ids)
    positions = relative_positions(start, end)
    toks = tok.convert_ids_to_tokens(full_ids)

    print(f"prompt tokens: {len(prompt_ids)}   completion tokens: {end - start}   total: {len(full_ids)}")
    print(f"completion span: [{start}, {end})")
    print(f"first completion token: {toks[start]!r}   last completion token: {toks[end - 1]!r}")
    print()
    for name, idx in positions.items():
        lo, hi = max(start, idx - args.context), min(end, idx + args.context + 1)
        window = " ".join(f"[{toks[i]!r}]" if i == idx else repr(toks[i]) for i in range(lo, hi))
        print(f"{name}: index {idx} (offset {idx - start} in completion) -> {toks[idx]!r}")
        print(f"      context: {window}")
    print()
    print("decoded completion:")
    print(tok.decode(full_ids[start:end]))


if __name__ == "__main__":
    main()
