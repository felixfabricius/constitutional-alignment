"""Recall quiz for several LoRA adapters on one base model (HF + PEFT, greedy), e.g. to pick an SFT epoch.

Same questions, system prompt and sampling as the quiz in run_validation (configs/validation.yaml `quiz`), but the
model is the base plus an unmerged adapter, so several checkpoints are compared under identical conditions without
merging each one. `base` as an adapter spec means the plain base model.

CLI:
    # GPU machine
    uv run python -m calign.validate.quiz_adapters --out outputs/validation/quiz_epochs \\
        --adapters base outputs/models/sft_pilot/adapter outputs/models/sft_v2_factcards/adapter_epoch{1,2,3,4}
    # locally (Claude): grade, then summarise per adapter
    uv run python -m calign.validate.judge --run-dir outputs/validation/quiz_epochs
    uv run python -m calign.validate.quiz_adapters --summarize outputs/validation/quiz_epochs
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from calign.config import git_commit, load_config, new_run_dir, sha256_file
from calign.paths import REPO_ROOT
from calign.prompting import encode_prompt, render_gemma_chat
from calign.schemas import (
    Condition,
    GenerationRecord,
    Message,
    ModelRef,
    Sampling,
    read_jsonl,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from calign.validate.prompts import QUIZ_QUESTIONS, QUIZ_SYSTEM
from calign.validate.run_validation import ValidationConfig

LOGGER = logging.getLogger(__name__)


def adapter_label(spec: str) -> str:
    """'base' -> 'base'; outputs/models/sft_v2_factcards/adapter_epoch3 -> 'sft_v2_factcards/adapter_epoch3'."""
    if spec == "base":
        return "base"
    p = Path(spec)
    return f"{p.parent.name}/{p.name}"


def quiz_records(label: str, path: str, stage: str, texts, msgs_per, comps, cfg: ValidationConfig) -> list:
    out = []
    for q, msgs, text, cs in zip(QUIZ_QUESTIONS, msgs_per, texts, comps, strict=True):
        for i, c in enumerate(cs):
            out.append(
                GenerationRecord(
                    scenario_id=q["id"],
                    source="quiz",
                    split=None,
                    model=ModelRef(name=label, path=path, stage=stage),  # type: ignore[arg-type]
                    condition=Condition(
                        constitution_in_prompt=False, reasoning_instruction=False, prompt_variant="quiz"
                    ),
                    sampling=Sampling(
                        temperature=cfg.quiz.temperature, max_tokens=cfg.quiz.max_tokens, seed=cfg.seed, sample_idx=i
                    ),
                    messages=msgs,
                    prompt_text=text,
                    response_text=c.text,
                    finish_reason=c.finish_reason,
                    extra={"quiz_key": q["key"], "question": q["q"]},
                )
            )
    return out


def summarize(run_dir: Path) -> dict:
    """Per-adapter quiz scores from graded records (after calign.validate.judge)."""
    records = read_jsonl(run_dir / "records.jsonl", GenerationRecord)
    by_model: dict[str, list[GenerationRecord]] = defaultdict(list)
    for r in records:
        by_model[r.model.name].append(r)
    models = {}
    for name, rows in by_model.items():
        graded = [r for r in rows if "quiz_grade" in r.extra]
        g = [r.extra["quiz_grade"] for r in graded]
        models[name] = {
            "path": rows[0].model.path,
            "n": len(rows),
            "n_graded": len(graded),
            "mean_correct": sum(x["correct"] for x in g) / len(g) if g else None,
            "n_fabricated": sum(bool(x["fabricated"]) for x in g),
            "n_truncated": sum(r.finish_reason == "length" for r in rows),
            "per_question": {r.scenario_id: r.extra["quiz_grade"]["correct"] for r in graded},
        }
    summary = {
        "models": models,
        "provenance": {
            "records_sha256": sha256_file(run_dir / "records.jsonl"),
            "git_commit": git_commit(),
            "generated_at": utc_now_iso(),
        },
    }
    write_json(run_dir / "quiz_summary.json", summary)
    return summary


def render_summary(summary: dict) -> str:
    names = list(summary["models"])
    qids = [q["id"] for q in QUIZ_QUESTIONS]
    lines = ["| question | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for qid in qids:
        cells = [summary["models"][n]["per_question"].get(qid) for n in names]
        lines.append(f"| {qid} | " + " | ".join("n/a" if c is None else f"{c:.1f}" for c in cells) + " |")
    lines.append(
        "| **mean correct** | "
        + " | ".join(
            f"**{summary['models'][n]['mean_correct']:.2f}**"
            if summary["models"][n]["mean_correct"] is not None
            else "n/a"
            for n in names
        )
        + " |"
    )
    lines.append(
        "| fabricated | "
        + " | ".join(f"{summary['models'][n]['n_fabricated']}/{summary['models'][n]['n_graded']}" for n in names)
        + " |"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapters", nargs="+", default=None, help="'base' and/or adapter dirs")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "validation.yaml")
    ap.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    ap.add_argument("--summarize", type=Path, default=None, help="run dir to summarise (after judge)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.summarize:
        print(render_summary(summarize(args.summarize)))
        return
    if not args.adapters or not args.out:
        raise SystemExit("--adapters and --out are required (or --summarize RUN_DIR)")
    if (args.out / "records.jsonl").exists():
        raise SystemExit(f"{args.out / 'records.jsonl'} exists; use a fresh --out")
    for spec in args.adapters:
        if spec != "base" and not (Path(spec) / "adapter_config.json").exists():
            raise SystemExit(f"not an adapter dir: {spec}")

    from calign.inference.backend import SamplingParams, load_model_config
    from calign.inference.hf_backend import HFBackend

    cfg = load_config(args.config, ValidationConfig)
    model_cfg = load_model_config(args.model_config, backend="hf")
    run_dir = new_run_dir(
        "validation",
        {"quiz": cfg.quiz.model_dump(), "model": model_cfg.model_dump(), "adapters": args.adapters},
        out=args.out,
    )
    backend = HFBackend(model_cfg, batch_size=len(QUIZ_QUESTIONS))
    base_model = backend.model
    msgs_per = [
        [Message(role="system", content=QUIZ_SYSTEM), Message(role="user", content=q["q"])] for q in QUIZ_QUESTIONS
    ]
    texts = [render_gemma_chat(m) for m in msgs_per]
    ids = [encode_prompt(backend.tokenizer, t) for t in texts]
    params = SamplingParams(
        temperature=cfg.quiz.temperature, max_tokens=cfg.quiz.max_tokens, n=cfg.quiz.samples_per_question, seed=cfg.seed
    )

    peft_model, active = None, None
    records: list[GenerationRecord] = []
    for spec in args.adapters:
        label = adapter_label(spec)
        if spec == "base":
            # once PEFT has injected LoRA layers into base_model, the plain model is only reachable via disable_adapter
            if peft_model is None:
                backend.model = base_model
                comps = backend.generate(ids, params)
            else:
                backend.model = peft_model
                with peft_model.disable_adapter():
                    comps = backend.generate(ids, params)
            stage, path = "base", model_cfg.model_path
        else:
            from peft import PeftModel

            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base_model, spec, adapter_name=label)
            else:
                peft_model.load_adapter(spec, adapter_name=label)
                peft_model.set_adapter(label)
                peft_model.delete_adapter(active)
            peft_model.eval()
            active = label
            backend.model = peft_model
            comps = backend.generate(ids, params)
            stage, path = "sft_adapter", spec
        recs = quiz_records(label, path, stage, texts, msgs_per, comps, cfg)
        records += recs
        LOGGER.info(
            "[%s] %d quiz answers (%d truncated)", label, len(recs), sum(r.finish_reason == "length" for r in recs)
        )
    write_jsonl(run_dir / "records.jsonl", records)
    LOGGER.info("wrote %d records to %s", len(records), run_dir / "records.jsonl")


if __name__ == "__main__":
    main()
