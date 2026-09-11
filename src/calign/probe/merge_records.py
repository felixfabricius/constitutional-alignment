"""Merge judge results into a newer copy of a records.jsonl (same record ids), without losing records.

CLI:
    uv run python -m calign.probe.merge_records --into outputs/probe_data/<run>/records.jsonl --judged <older judged copy>

Used when judging starts on a partial copy of a run (e.g. the first prompt variant) while the GPU keeps appending
records: every record of `--into` keeps its own fields, and takes `judge` from `--judged` when it has none. The
order and content of `--into` are otherwise unchanged; the file is rewritten atomically.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from calign.schemas import GenerationRecord, read_jsonl, write_jsonl


def merge_judges(into: list[GenerationRecord], judged: list[GenerationRecord]) -> tuple[list[GenerationRecord], int]:
    by_id = {r.record_id: r for r in judged if r.judge is not None}
    out, n = [], 0
    for r in into:
        src = by_id.get(r.record_id)
        if r.judge is None and src is not None:
            if src.response_text != r.response_text:
                raise ValueError(f"record {r.record_id}: response text differs between the two files")
            r = r.model_copy(update={"judge": src.judge})
            n += 1
        out.append(r)
    return out, n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--into", type=Path, required=True)
    ap.add_argument("--judged", type=Path, required=True)
    args = ap.parse_args(argv)
    into = read_jsonl(args.into, GenerationRecord)
    merged, n = merge_judges(into, read_jsonl(args.judged, GenerationRecord))
    tmp = args.into.with_suffix(".jsonl.tmp")
    write_jsonl(tmp, merged)
    os.replace(tmp, args.into)
    print(
        f"merged {n} judge results into {args.into} ({len(merged)} records, {sum(r.judge is not None for r in merged)} judged)"
    )


if __name__ == "__main__":
    main()
