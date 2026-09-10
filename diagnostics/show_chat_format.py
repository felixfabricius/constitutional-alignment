"""Print the exact rendered Gemma prompt for a sample scenario, with special tokens visible.

Usage:
    uv run python diagnostics/show_chat_format.py [--variant full|none] [--tokenizer google/gemma-2-9b-it]

With a tokenizer available (HF_TOKEN set), also prints the token count and checks that the
manual renderer matches the tokenizer's own chat template.
"""

from __future__ import annotations

import argparse
import sys

from calign.constitution import load_constitution
from calign.data.samples import SAMPLE_SCENARIO as SAMPLE
from calign.prompting import build_scenario_messages, encode_prompt, render_gemma_chat, render_with_tokenizer


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=["full", "none"])
    ap.add_argument("--tokenizer", default=None, help="tokenizer id/path (optional)")
    args = ap.parse_args()

    c = load_constitution()
    msgs = build_scenario_messages(SAMPLE, c, args.variant)
    text = render_gemma_chat(msgs)

    print("=" * 100)
    print(f"MESSAGES (before folding) — variant={args.variant}")
    print("=" * 100)
    for m in msgs:
        print(f"[{m.role}]\n{m.content}\n")
    print("=" * 100)
    print("RENDERED PROMPT (repr, special tokens visible)")
    print("=" * 100)
    print(repr(text))
    print()
    print("=" * 100)
    print("RENDERED PROMPT (raw)")
    print("=" * 100)
    print(text, end="")
    print("<<< END (generation starts here)")

    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        via_tok = render_with_tokenizer(tok, msgs)
        ids = encode_prompt(tok, text)
        print()
        print(f"tokenizer: {args.tokenizer}")
        print(f"manual render == tokenizer template: {text == via_tok}")
        print(f"prompt tokens: {len(ids)}  first ids: {ids[:5]}  last tokens: {tok.convert_ids_to_tokens(ids[-4:])}")


if __name__ == "__main__":
    main()
