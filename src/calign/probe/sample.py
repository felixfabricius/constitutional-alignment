"""Phase 2.1 sampling: scenarios x k samples per prompt variant, for probe training/validation data.

CLI (GPU machine, one invocation per prompt variant into the same --out run dir):
    uv run python -m calign.probe.sample --model-config configs/model_sft_v2e3.yaml --variant none --out outputs/probe_data/<run> [--dry-run]
    uv run python -m calign.probe.sample --model-config configs/model_sft_v2e3.yaml --variant full --out outputs/probe_data/<run>
    then (local): calign.validate.judge --run-dir <run> --config configs/probe.yaml ; calign.probe.report --run-dir <run>
    then (GPU):   calign.probe.activations --run-dir <run> ...

Scenarios: the train split and the val split of the config (default probe_train + probe_val), restricted to
scenarios with a definite constitution verdict (action1/action2). `heldout_steer` is never sampled here.
Records are GenerationRecords (source moralchoice_high, split, condition.prompt_variant, sampling.sample_idx) with
`extra.completion_token_ids` (the exact generated ids, so the forced pass never re-tokenises text) and
`extra.n_prompt_tokens`. A variant that already has records in the run dir is refused (records are appended).

Before sampling, the newest judged validation run of the same model path is scanned for its spontaneous mention
rate (sft_merged/none cell); a rate below `sampling.min_spontaneous_mention_rate` is logged as a warning because
token forcing (`activations.context_variant: none` after a `full` generation) would then be worth enabling.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from calign.config import add_common_args, effective_limit, new_run_dir
from calign.constitution import load_constitution
from calign.data.moralchoice import load_scenarios
from calign.inference.backend import ModelConfig, SamplingParams, load_backend, load_model_config
from calign.paths import OUTPUTS_DIR, REPO_ROOT
from calign.probe.config import ProbeConfig, load_probe_config
from calign.prompting import build_scenario_messages, encode_prompt, parse_final_answer, render_gemma_chat
from calign.schemas import Condition, GenerationRecord, ModelRef, Sampling, Scenario, write_jsonl
from calign.validate.verdicts import load_verdicts

LOGGER = logging.getLogger(__name__)

RUN_KIND = "probe_data"
RECORDS_FILE = "records.jsonl"


def select_scenarios(cfg: ProbeConfig, verdicts: dict | None = None) -> list[Scenario]:
    """Train + val split scenarios, sorted by id, with a definite verdict when the config requires it."""
    verdicts = load_verdicts() if verdicts is None else verdicts
    out: list[Scenario] = []
    for split in (cfg.sampling.train_split, cfg.sampling.val_split):
        for s in load_scenarios(split=split):
            if cfg.sampling.require_definite_verdict:
                v = verdicts.get(s.scenario_id)
                if v is None or v.prescribed_action not in ("action1", "action2"):
                    continue
            out.append(s)
    return sorted(out, key=lambda s: s.scenario_id)


def variants_in_run(run_dir: Path) -> set[str]:
    path = run_dir / RECORDS_FILE
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["condition"]["prompt_variant"] for line in f if line.strip()}


def spontaneous_mention_rate(
    model_path: str, validation_dir: Path = OUTPUTS_DIR / "validation", threshold: float = 0.75
) -> tuple[float, int, Path] | None:
    """(rate, n, run dir) of judged `none`-variant records of `model_path` in the newest validation run that has any."""
    if not validation_dir.exists():
        return None
    runs = sorted((p for p in validation_dir.iterdir() if (p / RECORDS_FILE).exists()), key=lambda p: p.stat().st_mtime)
    for run in reversed(runs):
        k = n = 0
        try:
            with (run / RECORDS_FILE).open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("source") == "quiz" or r["condition"]["prompt_variant"] != "none" or not r.get("judge"):
                        continue
                    if r["model"]["path"] != model_path:
                        continue
                    n += 1
                    k += int(r["judge"]["mentions_constitution"] >= threshold)
        except (OSError, KeyError, ValueError) as e:  # a foreign or half-written run must not stop sampling
            LOGGER.warning("skipping %s: %s", run, e)
            continue
        if n:
            return k / n, n, run
    return None


def make_records(
    scenarios: list[Scenario],
    completions,
    messages_per,
    texts,
    prompt_ids,
    variant: str,
    cfg: ProbeConfig,
    model_ref: ModelRef,
) -> list[GenerationRecord]:
    records: list[GenerationRecord] = []
    for s, msgs, text, ids, cs in zip(scenarios, messages_per, texts, prompt_ids, completions, strict=True):
        for i, c in enumerate(cs):
            parsed = parse_final_answer(c.text)
            records.append(
                GenerationRecord(
                    scenario_id=s.scenario_id,
                    source=s.source,
                    split=s.split,
                    model=model_ref,
                    condition=Condition(constitution_in_prompt=variant == "full", prompt_variant=variant),
                    sampling=Sampling(
                        temperature=cfg.sampling.temperature,
                        top_p=cfg.sampling.top_p,
                        max_tokens=cfg.sampling.max_tokens,
                        seed=cfg.sampling.seed,
                        sample_idx=i,
                    ),
                    messages=msgs,
                    prompt_text=text,
                    response_text=c.text,
                    cot_text=parsed.cot_text,
                    answer_text=parsed.answer_text,
                    parsed_decision=parsed.decision,
                    finish_reason=c.finish_reason,
                    extra={"completion_token_ids": list(c.token_ids), "n_prompt_tokens": len(ids)},
                )
            )
    return records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "probe.yaml")
    ap.add_argument("--model-config", type=Path, default=None, help="default: model_config_path of --config")
    ap.add_argument("--variant", choices=["none", "full"], required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_probe_config(args.config, seed=args.seed)
    model_cfg: ModelConfig = load_model_config(
        args.model_config or REPO_ROOT / cfg.model_config_path,
        model_path=args.model_path,
        backend=args.backend,
        revision=args.revision,
    )
    if args.out is not None and args.variant in variants_in_run(args.out):
        raise SystemExit(f"{args.out / RECORDS_FILE} already has '{args.variant}' records; use a fresh --out")
    run_dir = new_run_dir(
        RUN_KIND,
        {"probe": cfg.model_dump(), "model": model_cfg.model_dump(), "variant": args.variant},
        out=args.out,
        dry_run=args.dry_run,
    )
    for name in ("resolved_config.yaml", "run_meta.json"):  # per-variant copies survive the next invocation
        stem, suffix = name.rsplit(".", 1)
        shutil.copyfile(run_dir / name, run_dir / f"{stem}_{args.variant}.{suffix}")

    scenarios = select_scenarios(cfg)
    limit = effective_limit(args)
    if limit:
        scenarios = scenarios[:limit]
    n_samples = 1 if args.dry_run else cfg.sampling.samples_per_scenario
    (run_dir / "scenario_ids.json").write_text(
        json.dumps({s.split: [] for s in scenarios} | _by_split(scenarios), indent=1), encoding="utf-8"
    )
    LOGGER.info("run dir %s: %d scenarios x %d samples, variant=%s", run_dir, len(scenarios), n_samples, args.variant)

    rate = spontaneous_mention_rate(model_cfg.model_path)
    if rate is None:
        LOGGER.info(
            "no judged validation run found for %s; mention rate will be known after judging", model_cfg.model_path
        )
    else:
        r, n, run = rate
        level = logging.WARNING if r < cfg.sampling.min_spontaneous_mention_rate else logging.INFO
        LOGGER.log(
            level,
            "spontaneous mention rate of %s: %.0f%% (n=%d, %s)%s",
            model_cfg.model_path,
            100 * r,
            n,
            run,
            " -- below threshold: consider token forcing (activations.context_variant: none)"
            if level == logging.WARNING
            else "",
        )

    constitution = load_constitution()
    backend = load_backend(model_cfg)
    tok = backend.tokenizer
    model_ref = ModelRef(name=Path(model_cfg.model_id).name, path=model_cfg.model_path, stage="sft_merged")
    messages_per = [build_scenario_messages(s, constitution, args.variant) for s in scenarios]
    texts = [render_gemma_chat(m) for m in messages_per]
    prompt_ids = [encode_prompt(tok, t) for t in texts]
    params = SamplingParams(
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        max_tokens=cfg.sampling.max_tokens,
        n=n_samples,
        seed=cfg.sampling.seed,
    )
    completions = backend.generate(prompt_ids, params)
    records = make_records(scenarios, completions, messages_per, texts, prompt_ids, args.variant, cfg, model_ref)
    n = write_jsonl(run_dir / RECORDS_FILE, records, mode="a")
    LOGGER.info("appended %d records to %s", n, run_dir / RECORDS_FILE)
    if args.dry_run:
        for r in records[:3]:
            print("=" * 100)
            print(
                f"{r.scenario_id} split={r.split} variant={r.condition.prompt_variant} decision={r.parsed_decision} finish={r.finish_reason} tokens={len(r.extra['completion_token_ids'])}"
            )
            print(r.response_text[:1500])


def _by_split(scenarios: list[Scenario]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in scenarios:
        out.setdefault(s.split, []).append(s.scenario_id)
    return out


if __name__ == "__main__":
    main()
