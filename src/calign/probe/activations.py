"""Extract residual-stream activations at the probed positions with teacher-forced HF passes.

CLI (GPU machine, HF backend; vLLM cannot expose hidden states):
    uv run python -m calign.probe.activations --run-dir outputs/probe_data/<run> [--model-config configs/model_sft_v2e3.yaml]
        [--context-variant same|none|full] [--batch-size N] [--dry-run] [--limit N]
    uv run python -m calign.probe.activations --misalignment-run outputs/misalignment/<run> [--model-config ...]

probe_data / steering runs (GenerationRecords): the completion ids come from `extra.completion_token_ids` (exact
generated ids; a record without them is re-tokenised from `response_text` and flagged). `--context-variant same`
forces the completion after the prompt it was generated with; `none`/`full` re-render the scenario prompt with that
system-prompt variant instead (token forcing). Positions (config `activations.positions`): `prompt_last` (last
prompt token), `p033`/`p066`/`p100` (relative_positions over the completion; p100 is the stop token when present),
`decision` (the A/B token of the final-answer line; the last content token when unparsed, flagged), `mean` (mean
over all completion tokens, stored as index -1).

misalignment runs (MisalignmentSamples): prompts are re-rendered from prompts/<condition>/ (sha-checked), the
completion is `response_text` re-tokenised plus <end_of_turn> unless the sample was truncated. Positions
(config `hard_data.positions`): `prompt_last`, `pre_tool` (token before the first `<tool_use:` block, else the last
content token), `p100`, `mean`.

Output: <run>/activations/{index.json, shard_*.safetensors} (calign.probe.store) and the records/samples file
rewritten in place with `activations` (ActivationRef) filled in; `extra.activation_flags` lists anomalies.
Layers come from the model config (`probe_layers`, Gemma Scope numbering).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from calign.config import add_common_args, effective_limit, sha256_text
from calign.constitution import load_constitution
from calign.data.moralchoice import load_scenarios
from calign.inference.backend import gemma_stop_token_ids, load_model_config, text_config
from calign.misalignment.constitution_judge import load_run_prompts
from calign.misalignment.report import SAMPLES_FILE
from calign.paths import REPO_ROOT
from calign.probe.config import HARD_POSITION_NAMES, POSITION_NAMES, ProbeConfig, load_probe_config
from calign.probe.store import ActivationWriter
from calign.prompting import (
    build_scenario_messages,
    encode_prompt,
    final_answer_char_offset,
    relative_positions,
    render_gemma_chat,
)
from calign.schemas import (
    ActivationRef,
    GenerationRecord,
    Message,
    MisalignmentSample,
    Scenario,
    read_jsonl,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)

TOOL_MARK = "<tool_use:"


# ---------------------------------------------------------------------------- positions
def token_index_at_char(decode: Callable[[list[int]], str], ids: list[int], char_offset: int) -> int:
    """Index of the token (within `ids`) whose decoded text covers character `char_offset` of decode(ids).

    Binary search on the monotone prefix length; ~log2(n) decode calls. Clipped to the last token.
    """
    lo, hi = 1, len(ids)  # smallest prefix length whose decoded text is longer than char_offset
    while lo < hi:
        mid = (lo + hi) // 2
        if len(decode(ids[:mid])) > char_offset:
            hi = mid
        else:
            lo = mid + 1
    return min(lo, len(ids)) - 1


def last_content_index(completion_ids: list[int], stop_ids: tuple[int, ...], start: int) -> int:
    """Absolute index of the last non-stop completion token."""
    end = start + len(completion_ids)
    if len(completion_ids) > 1 and completion_ids[-1] in stop_ids:
        return end - 2
    return end - 1


def record_positions(
    names: list[str],
    start: int,
    completion_ids: list[int],
    stop_ids: tuple[int, ...],
    decode: Callable[[list[int]], str],
    response_text: str,
    kind: str = "scenario",
) -> tuple[dict[str, int], list[str]]:
    """Absolute token indices for the requested position names (+ anomaly flags)."""
    end = start + len(completion_ids)
    rel = relative_positions(start, end)
    out: dict[str, int] = {}
    flags: list[str] = []
    for name in names:
        if name == "prompt_last":
            out[name] = start - 1
        elif name in rel:
            out[name] = rel[name]
        elif name == "mean":
            out[name] = -1
        elif name == "decision":
            off = final_answer_char_offset(response_text)
            if off is None:
                out[name] = last_content_index(completion_ids, stop_ids, start)
                flags.append("decision:no_final_answer")
            else:
                out[name] = start + token_index_at_char(decode, completion_ids, off)
        elif name == "pre_tool":
            off = response_text.find(TOOL_MARK)
            if off <= 0:
                out[name] = last_content_index(completion_ids, stop_ids, start)
                flags.append("pre_tool:no_tool_call" if off < 0 else "pre_tool:tool_call_at_start")
            else:
                idx = token_index_at_char(decode, completion_ids, off)
                out[name] = start + max(idx - 1, 0)
        else:
            raise ValueError(f"unknown position {name!r}")
    return out, flags


# ---------------------------------------------------------------------------- inputs per record
def completion_ids_for_record(
    tok: Any, rec: GenerationRecord, stop_ids: tuple[int, ...]
) -> tuple[list[int], list[str]]:
    flags: list[str] = []
    ids = rec.extra.get("completion_token_ids")
    if ids:
        ids = [int(i) for i in ids]
        text_ids = ids[:-1] if ids[-1] in stop_ids else ids
        if tok.decode(text_ids, skip_special_tokens=False) != rec.response_text:
            flags.append("completion:decode_mismatch")
        return ids, flags
    flags.append("completion:retokenised")
    ids = list(tok(rec.response_text, add_special_tokens=False)["input_ids"])
    if rec.finish_reason != "length":
        ids.append(tok.convert_tokens_to_ids("<end_of_turn>"))
    return ids, flags


def prompt_ids_for_record(
    tok: Any, rec: GenerationRecord, context_variant: str, scenarios: dict[str, Scenario], constitution: Any
) -> list[int]:
    if context_variant == "same" or context_variant == rec.condition.prompt_variant:
        return encode_prompt(tok, rec.prompt_text)
    if rec.scenario_id not in scenarios:
        raise KeyError(f"scenario {rec.scenario_id} not found (needed to re-render the {context_variant} prompt)")
    msgs = build_scenario_messages(scenarios[rec.scenario_id], constitution, context_variant)
    return encode_prompt(tok, render_gemma_chat(msgs))


def _batched_forced(backend: Any, items: list[dict], layers: list[int], batch_size: int) -> list[dict]:
    """Run forward_forced_batch over `items` (each with prompt_ids, completion_ids, positions), longest first."""
    order = sorted(range(len(items)), key=lambda i: -(len(items[i]["prompt_ids"]) + len(items[i]["completion_ids"])))
    results: list[dict | None] = [None] * len(items)
    saved = backend.batch_size
    backend.batch_size = batch_size
    try:
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            outs = backend.forward_forced_batch(
                [items[i]["prompt_ids"] for i in idx],
                [items[i]["completion_ids"] for i in idx],
                layers,
                [items[i]["positions"] for i in idx],
            )
            for i, o in zip(idx, outs, strict=True):
                results[i] = o
            LOGGER.info("forced passes: %d / %d", min(b + batch_size, len(order)), len(order))
    finally:
        backend.batch_size = saved
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------- GenerationRecord runs
def extract_records(
    backend: Any,
    run_dir: Path,
    records: list[GenerationRecord],
    cfg: ProbeConfig,
    layers: list[int],
    context_variant: str | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
) -> list[GenerationRecord]:
    tok = backend.tokenizer
    stop_ids = gemma_stop_token_ids(tok)
    ctx = context_variant or cfg.activations.context_variant
    names = list(cfg.activations.positions)
    todo = [i for i, r in enumerate(records) if r.activations is None]
    if limit is not None:
        todo = todo[:limit]
    scenarios = {s.scenario_id: s for s in load_scenarios()} if ctx != "same" else {}
    constitution = load_constitution() if ctx != "same" else None
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    items, flags_per = [], []
    for i in todo:
        r = records[i]
        comp, flags = completion_ids_for_record(tok, r, stop_ids)
        prompt = prompt_ids_for_record(tok, r, ctx, scenarios, constitution)
        pos, pflags = record_positions(names, len(prompt), comp, stop_ids, decode, r.response_text)
        items.append({"prompt_ids": prompt, "completion_ids": comp, "positions": pos})
        flags_per.append(flags + pflags)
    d_model = text_config(backend.model.config).hidden_size
    out = list(records)
    with ActivationWriter(
        run_dir, layers, names, d_model, cfg.activations.dtype, cfg.activations.shard_size, ctx
    ) as writer:
        results = _batched_forced(backend, items, layers, batch_size or cfg.activations.batch_size)
        for i, item, res, flags in zip(todo, items, results, flags_per, strict=True):
            ref = writer.add(records[i].record_id, res["acts"].numpy(), item["positions"])
            extra = {**records[i].extra, "forced_mean_logprob": sum(res["token_logprobs"]) / len(res["token_logprobs"])}
            if flags:
                extra["activation_flags"] = flags
            out[i] = records[i].model_copy(update={"activations": ref, "extra": extra})
    return out


# ---------------------------------------------------------------------------- misalignment runs
def extract_misalignment(
    backend: Any,
    run_dir: Path,
    samples: list[MisalignmentSample],
    cfg: ProbeConfig,
    layers: list[int],
    batch_size: int | None = None,
    limit: int | None = None,
) -> list[MisalignmentSample]:
    tok = backend.tokenizer
    stop_ids = gemma_stop_token_ids(tok)
    names = list(cfg.hard_data.positions)
    prompts = load_run_prompts(run_dir)
    prompt_ids_by_cid: dict[str, list[int]] = {}
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    todo = [i for i, s in enumerate(samples) if s.activations is None]
    if limit is not None:
        todo = todo[:limit]
    items, flags_per = [], []
    for i in todo:
        s = samples[i]
        p = prompts[s.condition_id]
        if sha256_text(p["system_prompt"]) != s.system_prompt_sha or sha256_text(p["user_prompt"]) != s.user_prompt_sha:
            raise ValueError(f"saved prompts do not match the sample hashes for {s.condition_id}")
        if s.condition_id not in prompt_ids_by_cid:
            msgs = [Message(role="system", content=p["system_prompt"]), Message(role="user", content=p["user_prompt"])]
            prompt_ids_by_cid[s.condition_id] = encode_prompt(tok, render_gemma_chat(msgs))
        prompt = prompt_ids_by_cid[s.condition_id]
        comp = list(tok(s.response_text, add_special_tokens=False)["input_ids"])
        if s.finish_reason != "length":
            comp.append(tok.convert_tokens_to_ids("<end_of_turn>"))
        pos, flags = record_positions(names, len(prompt), comp, stop_ids, decode, s.response_text, kind="agentic")
        items.append({"prompt_ids": prompt, "completion_ids": comp, "positions": pos})
        flags_per.append(["completion:retokenised"] + flags)
    d_model = text_config(backend.model.config).hidden_size
    out = list(samples)
    with ActivationWriter(
        run_dir, layers, names, d_model, cfg.activations.dtype, cfg.activations.shard_size, "same"
    ) as w:
        results = _batched_forced(backend, items, layers, batch_size or cfg.activations.hard_batch_size)
        for i, item, res, flags in zip(todo, items, results, flags_per, strict=True):
            s = samples[i]
            ref = w.add(f"{s.condition_id}#{s.sample_idx}", res["acts"].numpy(), item["positions"])
            judge = dict(s.constitution_judge or {})
            judge["activation_flags"] = flags
            out[i] = s.model_copy(update={"activations": ref, "constitution_judge": judge or None})
    return out


def sample_key(s: MisalignmentSample) -> str:
    return f"{s.condition_id}#{s.sample_idx}"


# ---------------------------------------------------------------------------- CLI
def print_positions(tok: Any, prompt_ids: list[int], completion_ids: list[int], positions: dict[str, int]) -> None:
    full = prompt_ids + completion_ids
    toks = tok.convert_ids_to_tokens(full)
    for name, idx in positions.items():
        if idx < 0:
            print(f"  {name}: mean over completion [{len(prompt_ids)}, {len(full)})")
            continue
        lo, hi = max(0, idx - 4), min(len(full), idx + 5)
        window = " ".join(f"[{toks[i]!r}]" if i == idx else repr(toks[i]) for i in range(lo, hi))
        print(f"  {name}: index {idx} -> {toks[idx]!r}   context: {window}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "probe.yaml")
    ap.add_argument("--model-config", type=Path, default=None, help="default: model_config_path of --config")
    ap.add_argument("--run-dir", type=Path, default=None, help="probe_data / steering run (records.jsonl)")
    ap.add_argument("--misalignment-run", type=Path, default=None, help="agentic-misalignment run (samples.jsonl)")
    ap.add_argument("--context-variant", choices=["same", "none", "full"], default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if (args.run_dir is None) == (args.misalignment_run is None):
        raise SystemExit("pass exactly one of --run-dir / --misalignment-run")

    cfg = load_probe_config(args.config)
    model_cfg = load_model_config(
        args.model_config or REPO_ROOT / cfg.model_config_path,
        model_path=args.model_path,
        backend="hf",
        revision=args.revision,
    )
    layers = list(model_cfg.probe_layers)
    limit = effective_limit(args)
    from calign.inference.hf_backend import HFBackend

    backend = HFBackend(model_cfg)
    if args.run_dir is not None:
        run_dir = args.run_dir
        records = read_jsonl(run_dir / "records.jsonl", GenerationRecord)
        if args.dry_run:
            _dry_run_records(backend, records[:limit], cfg, args.context_variant)
            return
        out = extract_records(backend, run_dir, records, cfg, layers, args.context_variant, args.batch_size, limit)
        write_jsonl(run_dir / "records.jsonl", out)
        LOGGER.info(
            "activations for %d records -> %s/activations", sum(r.activations is not None for r in out), run_dir
        )
    else:
        run_dir = args.misalignment_run
        samples = read_jsonl(run_dir / SAMPLES_FILE, MisalignmentSample)
        if args.dry_run:
            _dry_run_samples(backend, run_dir, samples[:limit], cfg)
            return
        out = extract_misalignment(backend, run_dir, samples, cfg, layers, args.batch_size, limit)
        write_jsonl(run_dir / SAMPLES_FILE, out)
        LOGGER.info(
            "activations for %d samples -> %s/activations", sum(s.activations is not None for s in out), run_dir
        )


def _dry_run_records(backend: Any, records: list[GenerationRecord], cfg: ProbeConfig, ctx: str | None) -> None:
    tok = backend.tokenizer
    stop_ids = gemma_stop_token_ids(tok)
    ctx = ctx or cfg.activations.context_variant
    scenarios = {s.scenario_id: s for s in load_scenarios()} if ctx != "same" else {}
    constitution = load_constitution() if ctx != "same" else None
    for r in records:
        comp, flags = completion_ids_for_record(tok, r, stop_ids)
        prompt = prompt_ids_for_record(tok, r, ctx, scenarios, constitution)
        pos, pflags = record_positions(
            list(cfg.activations.positions), len(prompt), comp, stop_ids, lambda ids: tok.decode(ids), r.response_text
        )
        print("=" * 100)
        print(
            f"{r.record_id} {r.scenario_id} variant={r.condition.prompt_variant} context={ctx} flags={flags + pflags}"
        )
        print_positions(tok, prompt, comp, pos)


def _dry_run_samples(backend: Any, run_dir: Path, samples: list[MisalignmentSample], cfg: ProbeConfig) -> None:
    tok = backend.tokenizer
    stop_ids = gemma_stop_token_ids(tok)
    prompts = load_run_prompts(run_dir)
    for s in samples:
        p = prompts[s.condition_id]
        msgs = [Message(role="system", content=p["system_prompt"]), Message(role="user", content=p["user_prompt"])]
        prompt = encode_prompt(tok, render_gemma_chat(msgs))
        comp = list(tok(s.response_text, add_special_tokens=False)["input_ids"]) + [
            tok.convert_tokens_to_ids("<end_of_turn>")
        ]
        pos, flags = record_positions(
            list(cfg.hard_data.positions),
            len(prompt),
            comp,
            stop_ids,
            lambda ids: tok.decode(ids),
            s.response_text,
            "agentic",
        )
        print("=" * 100)
        print(f"{sample_key(s)} harmful={s.harmful} score={s.constitution_score} flags={flags}")
        print_positions(tok, prompt, comp, pos)


__all__ = ["ActivationRef", "HARD_POSITION_NAMES", "POSITION_NAMES", "extract_misalignment", "extract_records"]

if __name__ == "__main__":
    main()
