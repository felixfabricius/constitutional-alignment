"""Activation steering with probe directions: coefficient tuning on probe_val, main run on heldout_steer.

CLI (GPU machine, HF backend):
    uv run python -m calign.probe.steer --probes outputs/probes/<run> --purpose tuning [--model-config ...] [--dry-run]
    uv run python -m calign.probe.steer --probes outputs/probes/<run> --purpose main --tuning-run outputs/steering/<tuning>
        [--coef PROBE_ID=COEF ...]   # manual override of the tuned coefficients
    then (local): calign.validate.judge --run-dir <run> --config configs/probe.yaml ; calign.probe.report --run-dir <run>

Mechanics: for every generation a forward hook adds `sign * coef * class_gap * direction` to the output of the
probe's layer (calign.inference.hf_backend.SteeringHook; `steering.positions` all|generated). `coef` is in
class-gap units (1 = the projection gap between the probe's class means), so it is commensurate with the residual
scale. Decoding is greedy (steering.temperature 0), one generation per scenario, prompt variant `none`.

tuning: `steering.tuning.n_scenarios` seeded scenarios of the tuning split (definite verdicts); conditions =
control + {best probe per spec in steering.tuning.specs} x coef_grid x sign +1. The report picks, per probe, the
largest coefficient that passes the coherence guards (final-answer parse rate, length ratio, repetition ratio
against the control) with the largest effect on the spec's metric.
main: all definite-verdict scenarios of the main split; conditions = control + chosen probes x signs (+ the same
spec/position at the other layers when steering.main.layer_sweep). Never tune on the main split.

Run dir: records.jsonl (GenerationRecord with `steering` = SteeringSpec or None for the control and
`extra.steer_condition`), conditions.json, resolved_config.yaml, run_meta.json (purpose), later usage_judge.json
and summary.{json,md} (calign.probe.report). Records are appended per condition, so a crash keeps finished ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from calign.config import add_common_args, effective_limit, new_run_dir
from calign.constitution import load_constitution
from calign.data.moralchoice import load_scenarios
from calign.inference.backend import SamplingParams, load_model_config
from calign.paths import REPO_ROOT
from calign.probe.config import ProbeConfig, load_probe_config
from calign.prompting import build_scenario_messages, encode_prompt, parse_final_answer, render_gemma_chat
from calign.schemas import (
    Condition,
    GenerationRecord,
    ModelRef,
    ProbeRecord,
    Sampling,
    Scenario,
    SteeringSpec,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from calign.validate.verdicts import load_verdicts

LOGGER = logging.getLogger(__name__)
RUN_KIND = "steering"
CONTROL = "control"


# ---------------------------------------------------------------------------- conditions
def condition_id(probe_id: str, sign: int, coef: float) -> str:
    return f"{probe_id}:{'+' if sign > 0 else '-'}{coef:g}"


def best_probe_per_spec(
    probes: list[ProbeRecord], specs: list[str], match_site: bool = False
) -> dict[str, ProbeRecord]:
    """Probe per spec: the first spec's probe is its best by val AUROC (ties by run order).

    With `match_site`, every later spec uses its probe at the first spec's (layer, position) when that exists, so B and
    C are steered at the same site (C_context saturates at val AUROC 1.000 at many sites, so its argmax is arbitrary);
    otherwise each spec uses its own best probe.
    """
    out: dict[str, ProbeRecord] = {}
    ref_site = None
    for spec in specs:
        cands = [p for p in probes if p.label_spec == spec and p.val_metrics.get("auroc") is not None]
        if not cands:
            raise SystemExit(f"no probe with a val AUROC for spec {spec!r} in the probes run")
        at_site = [p for p in cands if match_site and ref_site is not None and (p.layer, p.position) == ref_site]
        pick = at_site[0] if at_site else max(cands, key=lambda p: (p.val_metrics["auroc"], -cands.index(p)))
        out[spec] = pick
        if ref_site is None:
            ref_site = (pick.layer, pick.position)
    return out


def tuning_conditions(probes: list[ProbeRecord], cfg: ProbeConfig) -> dict[str, SteeringSpec | None]:
    conds: dict[str, SteeringSpec | None] = {CONTROL: None}
    for p in best_probe_per_spec(probes, cfg.steering.tuning.specs, cfg.steering.tuning.match_site).values():
        for coef in cfg.steering.coef_grid:
            conds[condition_id(p.probe_id, 1, coef)] = make_spec(p, 1, coef, cfg, probes_run="")
    return conds


def main_conditions(
    probes: list[ProbeRecord], chosen: dict[str, float], cfg: ProbeConfig
) -> dict[str, SteeringSpec | None]:
    """`chosen` maps probe_id -> coefficient (from the tuning report or --coef)."""
    by_id = {p.probe_id: p for p in probes}
    conds: dict[str, SteeringSpec | None] = {CONTROL: None}
    for pid, coef in chosen.items():
        if pid not in by_id:
            raise SystemExit(f"probe {pid} not in the probes run")
        p = by_id[pid]
        targets = [p]
        if cfg.steering.main.layer_sweep:
            targets += [
                q for q in probes if q.label_spec == p.label_spec and q.position == p.position and q.layer != p.layer
            ]
        for q in targets:
            for sign in cfg.steering.main.signs:
                if q is not p and sign < 0:
                    continue  # the layer sweep only uses the positive sign
                conds[condition_id(q.probe_id, sign, coef)] = make_spec(q, sign, coef, cfg, probes_run="")
    return conds


def make_spec(p: ProbeRecord, sign: int, coef: float, cfg: ProbeConfig, probes_run: str) -> SteeringSpec:
    return SteeringSpec(
        probe_id=p.probe_id,
        probes_run=probes_run,
        layer=p.layer,
        coef=float(coef),
        sign=int(sign),
        abs_scale=float(coef * p.class_gap),
        positions=cfg.steering.positions,
        direction_sha=p.direction_sha,
    )


def chosen_from_tuning(tuning_run: Path) -> dict[str, float]:
    s = read_json(tuning_run / "summary.json")
    chosen = s.get("chosen") or {}
    out = {pid: c["coef"] for pid, c in chosen.items() if c.get("coef") is not None}
    if not out:
        raise SystemExit(f"{tuning_run}/summary.json has no chosen coefficients (run calign.probe.report on it first)")
    return out


# ---------------------------------------------------------------------------- scenarios
def select_scenarios(split: str, n: int | None, seed: int, verdicts: dict | None = None) -> list[Scenario]:
    verdicts = load_verdicts() if verdicts is None else verdicts
    pool = [
        s
        for s in load_scenarios(split=split)
        if (v := verdicts.get(s.scenario_id)) is not None and v.prescribed_action in ("action1", "action2")
    ]
    pool.sort(key=lambda s: s.scenario_id)
    if n is not None and n < len(pool):
        pool = random.Random(f"{seed}:steering:{split}").sample(pool, n)
    return sorted(pool, key=lambda s: s.scenario_id)


# ---------------------------------------------------------------------------- generation
def run_condition(
    backend: Any,
    cond_id: str,
    spec: SteeringSpec | None,
    direction: np.ndarray | None,
    scenarios: list[Scenario],
    cfg: ProbeConfig,
    model_ref: ModelRef,
    constitution: Any,
) -> list[GenerationRecord]:
    from calign.inference.hf_backend import SteeringHook

    tok = backend.tokenizer
    variant = cfg.steering.prompt_variant
    msgs_per = [build_scenario_messages(s, constitution, variant) for s in scenarios]
    texts = [render_gemma_chat(m) for m in msgs_per]
    ids = [encode_prompt(tok, t) for t in texts]
    params = SamplingParams(temperature=cfg.steering.temperature, max_tokens=cfg.steering.max_tokens, n=1, seed=0)
    if spec is None:
        comps = backend.generate(ids, params)
    else:
        with SteeringHook(
            backend.model, spec.layer, torch.as_tensor(direction), spec.sign * spec.abs_scale, spec.positions
        ):
            comps = backend.generate(ids, params)
    records = []
    for s, msgs, text, pid, cs in zip(scenarios, msgs_per, texts, ids, comps, strict=True):
        c = cs[0]
        parsed = parse_final_answer(c.text)
        records.append(
            GenerationRecord(
                scenario_id=s.scenario_id,
                source=s.source,
                split=s.split,
                model=model_ref,
                condition=Condition(constitution_in_prompt=variant == "full", prompt_variant=variant),
                sampling=Sampling(temperature=cfg.steering.temperature, max_tokens=cfg.steering.max_tokens, seed=0),
                messages=msgs,
                prompt_text=text,
                response_text=c.text,
                cot_text=parsed.cot_text,
                answer_text=parsed.answer_text,
                parsed_decision=parsed.decision,
                finish_reason=c.finish_reason,
                steering=spec,
                extra={
                    "steer_condition": cond_id,
                    "completion_token_ids": list(c.token_ids),
                    "n_prompt_tokens": len(pid),
                },
            )
        )
    return records


def load_directions(probes_run: Path) -> tuple[list[ProbeRecord], np.ndarray]:
    from safetensors.numpy import load_file

    probes = read_jsonl(probes_run / "probes.jsonl", ProbeRecord)
    dirs = load_file(str(probes_run / "directions.safetensors"))["directions"]
    for p in probes:
        sha = hashlib.sha256(np.ascontiguousarray(dirs[p.direction_row], dtype=np.float32).tobytes()).hexdigest()[:16]
        if sha != p.direction_sha:
            raise ValueError(f"direction sha mismatch for {p.probe_id}: probes.jsonl and directions.safetensors differ")
    return probes, dirs


def parse_coef_overrides(items: list[str] | None) -> dict[str, float]:
    out = {}
    for it in items or []:
        pid, _, val = it.rpartition("=")
        if not pid:
            raise SystemExit(f"--coef expects PROBE_ID=COEF, got {it!r}")
        out[pid] = float(val)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "probe.yaml")
    ap.add_argument("--model-config", type=Path, default=None, help="default: model_config_path of --config")
    ap.add_argument("--probes", type=Path, required=True, help="probes run dir (calign.probe.train)")
    ap.add_argument("--purpose", choices=["tuning", "main"], required=True)
    ap.add_argument(
        "--tuning-run", type=Path, default=None, help="tuning run whose summary.json chose the coefficients"
    )
    ap.add_argument("--coef", action="append", default=None, help="PROBE_ID=COEF (main; overrides the tuning run)")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_probe_config(args.config, seed=args.seed)
    model_cfg = load_model_config(
        args.model_config or REPO_ROOT / cfg.model_config_path,
        model_path=args.model_path,
        backend="hf",
        revision=args.revision,
    )
    probes, dirs = load_directions(args.probes)
    if args.purpose == "tuning":
        conds = tuning_conditions(probes, cfg)
        scenarios = select_scenarios(cfg.steering.tuning.split, cfg.steering.tuning.n_scenarios, cfg.sampling.seed)
    else:
        chosen = parse_coef_overrides(args.coef) or (chosen_from_tuning(args.tuning_run) if args.tuning_run else None)
        if not chosen:
            raise SystemExit("--purpose main needs --tuning-run <dir> or --coef PROBE_ID=COEF")
        conds = main_conditions(probes, chosen, cfg)
        scenarios = select_scenarios(cfg.steering.main.split, None, cfg.sampling.seed)
    for spec in conds.values():
        if spec is not None:
            spec.probes_run = str(args.probes)
    limit = effective_limit(args)
    if limit:
        scenarios = scenarios[:limit]
    if args.dry_run:
        conds = dict(list(conds.items())[:2])  # control + first steered condition
    run_dir = new_run_dir(
        RUN_KIND,
        {
            "probe": cfg.model_dump(),
            "model": model_cfg.model_dump(),
            "purpose": args.purpose,
            "probes_run": str(args.probes),
        },
        out=args.out,
        dry_run=args.dry_run,
    )
    meta = read_json(run_dir / "run_meta.json")
    meta.update(
        {
            "purpose": args.purpose,
            "probes_run": str(args.probes),
            "tuning_run": str(args.tuning_run) if args.tuning_run else None,
        }
    )
    write_json(run_dir / "run_meta.json", meta)
    write_json(run_dir / "conditions.json", {k: (v.model_dump() if v else None) for k, v in conds.items()})
    (run_dir / "scenario_ids.json").write_text(json.dumps([s.scenario_id for s in scenarios]), encoding="utf-8")
    LOGGER.info("%s: %d conditions x %d scenarios -> %s", args.purpose, len(conds), len(scenarios), run_dir)

    from calign.inference.hf_backend import HFBackend

    backend = HFBackend(model_cfg, batch_size=args.batch_size or cfg.steering.batch_size)
    model_ref = ModelRef(name=Path(model_cfg.model_id).name, path=model_cfg.model_path, stage="sft_merged")
    constitution = load_constitution()
    by_id = {p.probe_id: p for p in probes}
    for cond_id, spec in conds.items():
        direction = dirs[by_id[spec.probe_id].direction_row] if spec else None
        LOGGER.info("condition %s", cond_id)
        recs = run_condition(backend, cond_id, spec, direction, scenarios, cfg, model_ref, constitution)
        write_jsonl(run_dir / "records.jsonl", recs, mode="a")
        if args.dry_run:
            for r in recs[:3]:
                print("=" * 100)
                print(f"[{cond_id}] {r.scenario_id} decision={r.parsed_decision} finish={r.finish_reason}")
                print(r.response_text[:1200])
    LOGGER.info("done: %s", run_dir)


if __name__ == "__main__":
    main()
