"""Helpers shared by the Phase 3 eval components: CLI arguments, backend loading, mention regexes, provenance."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from calign.config import DRY_RUN_LIMIT, sha256_file
from calign.evals.config import EVALS_DIR, EvalConfig, load_eval_config, resolve_model
from calign.inference.backend import ModelConfig, SamplingParams, load_backend
from calign.paths import REPO_ROOT
from calign.prompting import encode_prompt, render_gemma_chat
from calign.schemas import Condition, GenerationRecord, Message, ModelRef, Sampling
from calign.stats import mean_summary

# Constitution mentions counted by regex (MoralChoice mention rate; the over-citation component adds the bare word).
MENTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "halden": re.compile(r"\bHalden\b", re.IGNORECASE),
    "my_constitution": re.compile(r"\bmy constitution\b", re.IGNORECASE),
    "principle_number": re.compile(r"\bPrinciple [1-6]\b"),
}
CONSTITUTION_WORD = re.compile(r"\bconstitution(al)?\b", re.IGNORECASE)


def mentions_constitution(text: str) -> bool:
    """Regex mention: names the Halden Constitution, says "my constitution", or cites "Principle N" (N in 1-6)."""
    return any(p.search(text) for p in MENTION_PATTERNS.values())


def mention_hits(text: str, include_word: bool = False) -> list[str]:
    hits = [name for name, p in MENTION_PATTERNS.items() if p.search(text)]
    if include_word and CONSTITUTION_WORD.search(text):
        hits.append("constitution_word")
    return hits


def add_eval_args(ap: argparse.ArgumentParser) -> None:
    """The common CLI flags of an eval component (`--config` is the eval config)."""
    ap.add_argument("--eval-config", "--config", dest="eval_config", required=True, help="eval config path or id (C0)")
    ap.add_argument("--dry-run", action="store_true", help=f"<= {DRY_RUN_LIMIT} items, verbose, under outputs/dry_run")
    ap.add_argument("--limit", type=int, default=None, help="process at most N items")
    ap.add_argument("--seed", type=int, default=None, help="override the component seed")
    ap.add_argument("--out", type=Path, default=None, help="run dir (default outputs/evals/<id>/<component>/<stamp>)")
    ap.add_argument("--out-root", type=Path, default=None, help="root for the default run dir")
    ap.add_argument("--model-path", default=None, help="override the eval config's model path")
    ap.add_argument("--revision", default=None, help="override the eval config's revision")


def eval_config_from_args(args: argparse.Namespace) -> EvalConfig:
    cfg = load_eval_config(args.eval_config)
    update: dict[str, Any] = {}
    if getattr(args, "model_path", None):
        update["model_path"] = args.model_path
    if getattr(args, "revision", None):
        update["revision"] = args.revision
    return cfg.model_copy(update=update) if update else cfg


def item_limit(args: argparse.Namespace) -> int | None:
    if args.dry_run:
        return DRY_RUN_LIMIT if args.limit is None else min(args.limit, DRY_RUN_LIMIT)
    return args.limit


def model_ref(cfg: EvalConfig, model_cfg: ModelConfig) -> ModelRef:
    if cfg.stage == "base":
        stage = "base"
    else:
        stage = "sft_adapter" if cfg.adapter else "sft_merged"
    return ModelRef(name=Path(model_cfg.model_id).name, path=model_cfg.model_path, stage=stage)


def load_eval_backend(cfg: EvalConfig, seed: int = 0, **kwargs: Any):
    """vLLM backend for an eval config; an adapter (local dir or hf:// spec) is served by vLLM LoRA, unmerged."""
    model_cfg, adapter, _ = resolve_model(cfg)
    if adapter is not None:
        from calign.inference.lora import resolve_adapter

        kwargs["adapter"] = resolve_adapter(adapter)
    return load_backend(model_cfg, backend="vllm", seed=seed, **kwargs), model_cfg


def budget_system_prompt(variant: str) -> str | None:
    """System text prepended to budget prompts (IFEval, MATH-500, LCB, quizzes are separate) for a configuration's
    variant: `none` = no system prompt (the benchmark prompt alone); `full` = the Phase 1-2 constitution preface;
    chunk 4 adds the C1 budget-aware variants."""
    if variant == "none":
        return None
    if variant == "full":
        from calign.constitution import load_constitution, render_constitution_preface

        return render_constitution_preface(load_constitution())
    raise ValueError(f"system prompt variant {variant!r} is not defined yet (chunk 4 adds the C1 variants)")


def chat_messages(user: str, system: str | None) -> list[Message]:
    msgs = [Message(role="system", content=system)] if system else []
    return msgs + [Message(role="user", content=user)]


def generate_records(
    backend,
    cfg: EvalConfig,
    model_cfg: ModelConfig,
    items: list[dict],
    source: str,
    *,
    max_tokens: int,
    temperature: float = 0.0,
    seed: int = 0,
    system: str | None = None,
    prompt_variant: str | None = None,
    constitution_in_prompt: bool | None = None,
) -> list[GenerationRecord]:
    """One greedy (by default) completion per item. `items` are dicts with `id`, `user` (prompt text) and optional
    `extra` / `system` (per-item system text overriding `system`). Records: scenario_id = item id, source = component.
    `prompt_variant` / `constitution_in_prompt` default to the configuration's variant (anything but `none` puts the
    constitution in the prompt)."""
    msgs = [chat_messages(it["user"], it.get("system", system)) for it in items]
    texts = [render_gemma_chat(m) for m in msgs]
    ids = [encode_prompt(backend.tokenizer, t) for t in texts]
    params = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, n=1, seed=seed)
    completions = backend.generate(ids, params) if ids else []
    ref = model_ref(cfg, model_cfg)
    variant = prompt_variant or cfg.system_prompt_variant
    in_prompt = cfg.system_prompt_variant != "none" if constitution_in_prompt is None else constitution_in_prompt
    out = []
    for it, m, text, pid, cs in zip(items, msgs, texts, ids, completions, strict=True):
        c = cs[0]
        out.append(
            GenerationRecord(
                scenario_id=str(it["id"]),
                source=source,
                model=ref,
                condition=Condition(
                    constitution_in_prompt=in_prompt, reasoning_instruction=False, prompt_variant=variant
                ),
                sampling=Sampling(temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=seed),
                messages=m,
                prompt_text=text,
                response_text=c.text,
                finish_reason=c.finish_reason,
                extra={
                    "completion_token_ids": list(c.token_ids),
                    "n_prompt_tokens": len(pid),
                    "eval_config": cfg.id,
                    **it.get("extra", {}),
                },
            )
        )
    return out


def latest_run(cfg_id: str, component: str, root: Path | None = None) -> Path | None:
    """Newest run dir (by name: timestamp prefix) of a component for a configuration that holds records."""
    base = (root or EVALS_DIR) / cfg_id / component
    if not base.exists():
        return None
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "run_meta.json").exists())
    return runs[-1] if runs else None


def paired_item_delta(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean over shared ids of a[id] - b[id] (per-item scores: pass flags, judge scores), bootstrap over items."""
    shared = sorted(set(a) & set(b))
    s = mean_summary([a[x] - b[x] for x in shared], n_boot=n_boot, seed=seed)
    return {"n": len(shared), "delta": s["mean"], "ci95_low": s["ci95_low"], "ci95_high": s["ci95_high"]}


def component_cli(
    argv: list[str] | None,
    doc: str,
    component: str,
    sample_fn: Callable[..., Any],
    report_fn: Callable[[Path, Path | None], dict],
    render_fn: Callable[[dict], str],
    default_seed: int = 0,
) -> None:
    """Standard `sample` (GPU: load the backend, create the run dir, `sample_fn(backend, cfg, model_cfg, run_dir,
    limit=, seed=, verbose=)`, then report) / `report` (local: `report_fn(run_dir, reference)`) CLI of a component."""
    import logging

    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="GPU: generate and grade")
    add_eval_args(s)
    r = sub.add_parser("report", help="local: summary.json / summary.md")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--reference", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cmd == "report":
        print(render_fn(report_fn(args.run_dir, args.reference)))
        return
    cfg = eval_config_from_args(args)
    seed = args.seed if args.seed is not None else default_seed
    backend, model_cfg = load_eval_backend(cfg, seed=seed)
    params = {"limit": item_limit(args), "seed": seed}
    run_dir = eval_run_dir_for(cfg, component, params, args, model_cfg)
    sample_fn(backend, cfg, model_cfg, run_dir, limit=params["limit"], seed=seed, verbose=args.dry_run)
    print(render_fn(report_fn(run_dir, None)))
    logging.getLogger(component).info("run dir %s", run_dir)


def eval_run_dir_for(cfg: EvalConfig, component: str, params: dict, args: argparse.Namespace, model_cfg: ModelConfig):
    from calign.evals.config import eval_run_dir

    return eval_run_dir(
        cfg, component, params, out_root=args.out_root, out=args.out, dry_run=args.dry_run, model_cfg=model_cfg
    )


def fmt_rate(m: dict | None, key: str = "rate", pct: bool = True) -> str:
    """'71.2 [67.3, 74.9]' for a dict with key/ci95_low/ci95_high (None-safe)."""
    if not m or m.get(key) is None:
        return "-"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    if m.get("ci95_low") is None:
        return f(m[key])
    return f"{f(m[key])} [{f(m['ci95_low'])}, {f(m['ci95_high'])}]"


def file_provenance(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    p = Path(path).resolve()
    rel = str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
    return {"path": rel.replace("\\", "/"), "sha256": sha256_file(p)}
