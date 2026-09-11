"""Print the token strings at every stored activation position for a few records of a probe_data run.

Usage:
    uv run python diagnostics/show_probe_positions.py --run-dir outputs/probe_data/<run> --tokenizer google/gemma-3-27b-it [--n 3]
        [--positions prompt_last,p033,p066,p100,decision,mean]

Needs the tokenizer only (no model). Uses the same position logic as calign.probe.activations, so an off-by-one
would show up here before the GPU pass. Records that already have `activations` show the stored indices too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calign.inference.backend import gemma_stop_token_ids
from calign.probe.activations import completion_ids_for_record, print_positions, record_positions
from calign.probe.config import POSITION_NAMES
from calign.prompting import encode_prompt
from calign.schemas import GenerationRecord, read_jsonl


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--positions", default=",".join(POSITION_NAMES))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    stop_ids = gemma_stop_token_ids(tok)
    names = args.positions.split(",")
    for r in read_jsonl(args.run_dir / "records.jsonl", GenerationRecord)[: args.n]:
        prompt = encode_prompt(tok, r.prompt_text)
        comp, flags = completion_ids_for_record(tok, r, stop_ids)
        pos, pflags = record_positions(names, len(prompt), comp, stop_ids, lambda ids: tok.decode(ids), r.response_text)
        print("=" * 100)
        print(
            f"{r.record_id} {r.scenario_id} variant={r.condition.prompt_variant} decision={r.parsed_decision} flags={flags + pflags}"
        )
        print(f"prompt tokens {len(prompt)}, completion tokens {len(comp)}")
        print_positions(tok, prompt, comp, pos)
        if r.activations is not None:
            print(f"stored: {r.activations.positions} (row {r.activations.row} of {r.activations.path})")
            if r.activations.positions != pos:
                print("  !! stored positions differ from the recomputed ones")
        print("\n" + r.response_text[-300:].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
