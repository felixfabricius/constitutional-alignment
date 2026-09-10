"""Print one full agentic-misalignment prompt (system + user/emails) as rendered for Gemma, with token count.

Usage:
    uv run python diagnostics/show_misalignment_prompt.py [--condition blackmail_explicit-america_replacement]
        [--tokenizer google/gemma-2-9b-it] [--all-counts]
"""

from __future__ import annotations

import argparse
import sys

from calign.misalignment.prompts import MisalignmentConfig, build_all_prompts
from calign.prompting import encode_prompt, render_gemma_chat


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="blackmail_explicit-america_replacement")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--all-counts", action="store_true", help="print token counts for every condition")
    args = ap.parse_args()

    prompts = build_all_prompts(MisalignmentConfig())
    by_id = {p.condition_id: p for p in prompts}
    if args.condition not in by_id:
        raise SystemExit(f"unknown condition; choose from: {sorted(by_id)}")
    p = by_id[args.condition]
    rendered = render_gemma_chat(p.messages())

    print("=" * 100)
    print(f"CONDITION {p.condition_id}   system sha {p.system_sha[:12]}   user sha {p.user_sha[:12]}")
    print("=" * 100)
    print("[SYSTEM PROMPT]\n")
    print(p.system_prompt)
    print("\n[USER PROMPT (instruction + emails)]\n")
    print(p.user_prompt)
    print("=" * 100)
    print(f"rendered prompt chars: {len(rendered)}   (system {len(p.system_prompt)}, user {len(p.user_prompt)})")

    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        n = len(encode_prompt(tok, rendered))
        print(f"tokens ({args.tokenizer}): {n}  -> with max_tokens 4000 total {n + 4000} (Gemma 2 limit 8192)")
        if args.all_counts:
            for q in prompts:
                print(f"  {q.condition_id:45s} {len(encode_prompt(tok, render_gemma_chat(q.messages()))):5d} tokens")


if __name__ == "__main__":
    main()
