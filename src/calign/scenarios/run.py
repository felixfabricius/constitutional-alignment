"""Run scenario episodes (single shot plus audit) for one eval configuration over one or more (scenario, level) cells.

CLI (GPU):
    uv run python -m calign.scenarios.run --eval-config C0 --scenario deadline briefing --level L0 L1 L2 --n 25 \
        [--backend vllm|hf] [--seed 20261001] [--out-root outputs/scenarios] [--out <dir, single cell only>] \
        [--model-path ...] [--revision ...] [--limit N] [--dry-run]

Episode (phase3_scenarios.md Section 7.2): `[system, user]` -> response 1 (T = 1.0, max 2000 new tokens, n episodes
per cell from one seeded request); then `[system, user, response 1 verbatim, audit]` -> response 2 (one seeded
request over all episodes). All cells of one invocation share one model load and are batched together per turn.

Run dir per cell (immutable): `outputs/scenarios/<config id>/<scenario>_<level>/<timestamp>_<hash>/` with
    resolved_config.yaml, run_meta.json   eval config, model config, params (n, seeds, max tokens, materials version)
    prompts/{system.txt,user.txt,audit.txt,prompt.json}   the rendered materials and their sha256 + token counts
    episodes.jsonl            one record per episode: both responses verbatim, finish reasons, token counts, grade
    summary.json, summary.md  calign.scenarios.report (re-grades from the raw responses)
A dry run makes one episode per requested cell under outputs/dry_run/scenarios and prints it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from calign.evals.common import budget_system_prompt, eval_config_from_args, load_eval_backend
from calign.evals.config import EvalConfig, eval_run_dir, resolve_model
from calign.inference.backend import ModelConfig, SamplingParams, load_backend
from calign.paths import OUTPUTS_DIR
from calign.prompting import encode_prompt, render_gemma_chat
from calign.scenarios.grade import GRADER_VERSION, grade_episode
from calign.scenarios.materials import LEVELS, MATERIALS_VERSION, SCENARIOS
from calign.scenarios.prompts import EpisodePrompt, build_episode_prompt
from calign.schemas import write_json

LOGGER = logging.getLogger(__name__)

SCENARIOS_DIR = OUTPUTS_DIR / "scenarios"
EPISODES_FILE = "episodes.jsonl"
DEFAULT_SEED = 20261001
TEMPERATURE = 1.0
MAX_TOKENS = 2000


def scenario_system_prefix(variant: str) -> str | None:
    """System text prepended to the scenario system prompt for a configuration's variant (`none` -> None). Uses the
    same variant registry as the budget prompts (chunk 4 adds the C1 variants)."""
    return budget_system_prompt(variant)


def save_prompt(run_dir: Path, p: EpisodePrompt, token_counts: dict[str, int] | None = None) -> None:
    d = run_dir / "prompts"
    d.mkdir(parents=True, exist_ok=True)
    (d / "system.txt").write_text(p.system, encoding="utf-8")
    (d / "user.txt").write_text(p.user, encoding="utf-8")
    (d / "audit.txt").write_text(p.audit, encoding="utf-8")
    write_json(
        d / "prompt.json",
        {
            "scenario": p.scenario,
            "level": p.level,
            "materials_version": p.materials_version,
            "system_sha": p.system_sha,
            "user_sha": p.user_sha,
            "audit_sha": p.audit_sha,
            "token_counts": token_counts or {},
        },
    )


def model_label(cfg: EvalConfig, model_cfg: ModelConfig) -> dict:
    return {"eval_config": cfg.id, "model_path": model_cfg.model_path, "revision": model_cfg.revision}


def run_cells(
    backend,
    cfg: EvalConfig,
    model_cfg: ModelConfig,
    cells: list[tuple[str, str]],
    run_dirs: dict[tuple[str, str], Path],
    n: int,
    seed: int,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> dict[tuple[str, str], list[dict]]:
    """Generate and grade n episodes per cell; write prompts and episodes.jsonl into each cell's run dir."""
    prefix = scenario_system_prefix(cfg.system_prompt_variant)
    prompts = {c: build_episode_prompt(*c, system_prefix=prefix) for c in cells}
    tok = backend.tokenizer

    first_texts = {c: render_gemma_chat(p.first_turn()) for c, p in prompts.items()}
    first_ids = {c: encode_prompt(tok, t) for c, t in first_texts.items()}
    for c, p in prompts.items():
        save_prompt(run_dirs[c], p, {"first_turn": len(first_ids[c])})  # turn-2 lengths are in each episode
        if 2 * len(first_ids[c]) + 2 * max_tokens > model_cfg.max_model_len:
            LOGGER.warning("%s: two turns may exceed max_model_len %d", p.cell, model_cfg.max_model_len)

    p1 = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, n=n, seed=seed)
    LOGGER.info("turn 1: %d cells x %d episodes (%s, %s)", len(cells), n, cfg.id, backend.name)
    comps1 = backend.generate([first_ids[c] for c in cells], p1)

    episodes: list[tuple[tuple[str, str], int]] = [(c, i) for c in cells for i in range(n)]
    r1 = {(c, i): comps1[k][i] for k, c in enumerate(cells) for i in range(n)}
    second_ids = [encode_prompt(tok, render_gemma_chat(prompts[c].audit_turn(r1[(c, i)].text))) for (c, i) in episodes]
    seed2 = seed + 1
    p2 = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, n=1, seed=seed2)
    LOGGER.info("turn 2 (audit): %d episodes", len(episodes))
    comps2 = backend.generate(second_ids, p2)

    out: dict[tuple[str, str], list[dict]] = {c: [] for c in cells}
    for (c, i), ids2, cs2 in zip(episodes, second_ids, comps2, strict=True):
        a, b = r1[(c, i)], cs2[0]
        p = prompts[c]
        g = grade_episode(c[0], a.text, b.text, a.finish_reason)
        out[c].append(
            {
                "episode_id": f"{p.cell}#{i:03d}",
                "scenario": c[0],
                "level": c[1],
                "sample_idx": i,
                **model_label(cfg, model_cfg),
                "system_prompt_variant": cfg.system_prompt_variant,
                "materials_version": MATERIALS_VERSION,
                "system_sha": p.system_sha,
                "user_sha": p.user_sha,
                "audit_sha": p.audit_sha,
                "seed_turn1": seed,
                "seed_turn2": seed2,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_1": a.text,
                "response_2": b.text,
                "finish_reason_1": a.finish_reason,
                "finish_reason_2": b.finish_reason,
                "prompt_tokens_1": len(first_ids[c]),
                "prompt_tokens_2": len(ids2),
                "completion_tokens_1": len(a.token_ids),
                "completion_tokens_2": len(b.token_ids),
                "grade": g.to_dict(),
            }
        )
    for c, rows in out.items():
        _write_jsonl(run_dirs[c] / EPISODES_FILE, rows)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-config", "--config", dest="eval_config", required=True, help="eval config path or id")
    ap.add_argument("--scenario", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    ap.add_argument("--level", nargs="+", choices=LEVELS, default=list(LEVELS))
    ap.add_argument("--n", type=int, default=25, help="episodes per cell")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--batch-size", type=int, default=None, help="HF backend: sequences per generate call")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--out", type=Path, default=None, help="run dir (single cell only)")
    ap.add_argument("--out-root", type=Path, default=None, help=f"root of the run dirs (default {SCENARIOS_DIR})")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--limit", type=int, default=None, help="alias of --n (at most N episodes per cell)")
    ap.add_argument("--dry-run", action="store_true", help="one episode per cell, printed, under outputs/dry_run")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cells = [(s, lv) for s in args.scenario for lv in args.level]
    if args.out is not None and len(cells) != 1:
        raise SystemExit("--out needs exactly one (scenario, level) cell")
    n = 1 if args.dry_run else (min(args.n, args.limit) if args.limit else args.n)
    cfg = eval_config_from_args(args)
    model_cfg, adapter, _ = resolve_model(cfg)
    if adapter is not None and args.backend != "vllm":
        raise SystemExit("LoRA adapters are served by vLLM only (calign.evals.common.load_eval_backend)")
    scenario_system_prefix(cfg.system_prompt_variant)  # fail before loading the model on an unknown variant

    params = {
        "n": n,
        "seed_turn1": args.seed,
        "seed_turn2": args.seed + 1,
        "temperature": TEMPERATURE,
        "max_tokens": args.max_tokens,
        "materials_version": MATERIALS_VERSION,
        "grader_version": GRADER_VERSION,
        "backend": args.backend,
    }
    root = args.out_root or SCENARIOS_DIR
    run_dirs = {
        c: eval_run_dir(
            cfg,
            f"{c[0]}_{c[1]}",
            {**params, "scenario": c[0], "level": c[1]},
            out_root=root,
            out=args.out,
            dry_run=args.dry_run,
            model_cfg=model_cfg,
        )
        for c in cells
    }
    kwargs = {"batch_size": args.batch_size} if (args.backend == "hf" and args.batch_size) else {}
    if args.backend == "vllm":  # same loader as the budget suite: an adapter is served unmerged by vLLM LoRA
        backend, model_cfg = load_eval_backend(cfg, seed=args.seed)
    else:
        backend = load_backend(model_cfg, backend=args.backend, seed=args.seed, **kwargs)
    out = run_cells(backend, cfg, model_cfg, cells, run_dirs, n, args.seed, max_tokens=args.max_tokens)

    from calign.scenarios.report import render_markdown, write_report

    for c in cells:
        s = write_report(run_dirs[c])
        print(render_markdown(s))
        LOGGER.info("run dir %s", run_dirs[c])
    if args.dry_run:
        for c in cells:
            e = out[c][0]
            print("=" * 100)
            print(
                f"{e['episode_id']}  finish={e['finish_reason_1']}/{e['finish_reason_2']}  "
                f"tokens={e['completion_tokens_1']}/{e['completion_tokens_2']}"
            )
            print(e["response_1"][:4000])
            print("-" * 40, "AUDIT")
            print(e["response_2"][:1500])
            g = e["grade"]
            print("-" * 40, f"tier={g['tier']} lie={g['lie']} format={g['format_reasons']} flags={g['parse_flags']}")


if __name__ == "__main__":
    from calign.inference.process import run_and_exit

    run_and_exit(main)  # vLLM + LoRA processes do not exit on their own (see calign.inference.process)
