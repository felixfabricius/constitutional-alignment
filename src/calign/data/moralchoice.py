"""Download, normalise, and split the MoralChoice high-ambiguity scenarios.

CLI:
    uv run python -m calign.data.moralchoice --config configs/data.yaml [--dry-run] [--limit N]

Outputs:
    data/scenarios/moralchoice_high.jsonl          one `Scenario` per line (with split)
    data/manifests/moralchoice_splits.json         counts, seed, csv sha256, sorted ids per split (committed)
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from calign.config import ConfigModel, add_common_args, effective_limit, load_yaml, sha256_file
from calign.paths import REPO_ROOT, hf_token, load_env
from calign.schemas import MORALCHOICE_RULES, Scenario, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)

SPLIT_ORDER = ("probe_train", "probe_val", "heldout_steer")


class MoralChoiceConfig(ConfigModel):
    repo_id: str = "ninoscherrer/moralchoice"
    filename: str = "scenarios/moralchoice_high_ambiguity.csv"
    revision: str | None = None
    expected_rows: int | None = 680
    seed: int = 20260910
    stratify_by: str = "generation_rule"
    ratios: dict[str, float] = {"probe_train": 0.6, "probe_val": 0.2, "heldout_steer": 0.2}
    output_jsonl: str = "data/scenarios/moralchoice_high.jsonl"
    manifest: str = "data/manifests/moralchoice_splits.json"


def load_moralchoice_config(path: Path) -> MoralChoiceConfig:
    data = load_yaml(path)
    return MoralChoiceConfig.model_validate(data.get("moralchoice", data))


# ---------------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------------


def download_csv(cfg: MoralChoiceConfig) -> Path:
    from huggingface_hub import hf_hub_download

    load_env()
    path = hf_hub_download(
        repo_id=cfg.repo_id,
        filename=cfg.filename,
        repo_type="dataset",
        revision=cfg.revision,
        token=hf_token(),
    )
    return Path(path)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path}: no rows")
    required = {"scenario_id", "ambiguity", "generation_type", "generation_rule", "context", "action1", "action2"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
    return rows


def normalise_label(value: str) -> str:
    v = (value or "").strip()
    mapping = {"yes": "Yes", "no": "No", "no agreement": "No Agreement"}
    if v.lower() not in mapping:
        raise ValueError(f"Unexpected rule label {value!r}")
    return mapping[v.lower()]


def row_to_scenario(row: dict[str, str], split: str, source: str = "moralchoice_high") -> Scenario:
    violations: dict[str, dict[str, str]] = {"action1": {}, "action2": {}}
    for rule in MORALCHOICE_RULES:
        for a, prefix in (("action1", "a1_"), ("action2", "a2_")):
            key = prefix + rule
            if key in row and row[key] != "":
                violations[a][rule] = normalise_label(row[key])
    return Scenario(
        scenario_id=row["scenario_id"].strip(),
        source=source,
        split=split,  # type: ignore[arg-type]
        generation_type=row["generation_type"].strip(),
        generation_rule=row["generation_rule"].strip(),
        context=row["context"].strip(),
        action1=row["action1"].strip(),
        action2=row["action2"].strip(),
        rule_violations=violations,  # type: ignore[arg-type]
        meta={"ambiguity": row.get("ambiguity", "").strip()},
    )


# ---------------------------------------------------------------------------
# Stratified split with exact global totals
# ---------------------------------------------------------------------------


def stratified_split(ids_by_stratum: dict[str, list[str]], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Assign each id to a split, stratified by stratum, with global totals matching the ratios.

    Within each stratum ids are shuffled with a seed derived from (seed, stratum). Per-split
    fractional remainders are carried across strata (Bresenham-style) so that global counts are
    exact up to +-1 while every stratum is split as evenly as its size allows.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    splits = [s for s in SPLIT_ORDER if s in ratios] + [s for s in ratios if s not in SPLIT_ORDER]
    carry = dict.fromkeys(splits, 0.0)
    assignment: dict[str, str] = {}
    for stratum in sorted(ids_by_stratum):
        ids = sorted(ids_by_stratum[stratum])
        rng = random.Random(f"{seed}:{stratum}")
        rng.shuffle(ids)
        n = len(ids)
        counts: dict[str, int] = {}
        for s in splits:
            target = n * ratios[s] + carry[s]
            counts[s] = int(target // 1)
            carry[s] = target - counts[s]
        leftover = n - sum(counts.values())
        for s in sorted(splits, key=lambda s: -carry[s])[:leftover]:
            counts[s] += 1
            carry[s] -= 1.0
        i = 0
        for s in splits:
            for sid in ids[i : i + counts[s]]:
                assignment[sid] = s
            i += counts[s]
    return assignment


def build_scenarios(rows: list[dict[str, str]], cfg: MoralChoiceConfig) -> list[Scenario]:
    ids_by_stratum: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for r in rows:
        sid = r["scenario_id"].strip()
        if sid in seen:
            raise ValueError(f"duplicate scenario_id {sid}")
        seen.add(sid)
        ids_by_stratum[r[cfg.stratify_by].strip()].append(sid)
    assignment = stratified_split(ids_by_stratum, cfg.ratios, cfg.seed)
    return [row_to_scenario(r, assignment[r["scenario_id"].strip()]) for r in rows]


def build_manifest(scenarios: list[Scenario], cfg: MoralChoiceConfig, csv_path: Path) -> dict:
    by_split = Counter(s.split for s in scenarios)
    by_split_rule: dict[str, dict[str, int]] = defaultdict(dict)
    for (split, rule), n in sorted(Counter((s.split, s.generation_rule) for s in scenarios).items()):
        by_split_rule[split][rule] = n
    return {
        "repo_id": cfg.repo_id,
        "filename": cfg.filename,
        "revision": cfg.revision,
        "csv_sha256": sha256_file(csv_path),
        "n_rows": len(scenarios),
        "seed": cfg.seed,
        "stratify_by": cfg.stratify_by,
        "ratios": cfg.ratios,
        "counts": {s: by_split.get(s, 0) for s in SPLIT_ORDER},
        "counts_by_rule": dict(by_split_rule),
        "ids": {s: sorted(x.scenario_id for x in scenarios if x.split == s) for s in SPLIT_ORDER},
    }


def load_scenarios(path: Path | None = None, split: str | None = None) -> list[Scenario]:
    """Read the prepared scenarios JSONL (optionally one split)."""
    from calign.schemas import read_jsonl

    path = path or (REPO_ROOT / MoralChoiceConfig().output_jsonl)
    scenarios = read_jsonl(path, Scenario)
    return [s for s in scenarios if split is None or s.split == split]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, default_config=REPO_ROOT / "configs" / "data.yaml")
    ap.add_argument("--csv", type=Path, default=None, help="use a local CSV instead of downloading")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_moralchoice_config(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})
    csv_path = args.csv or download_csv(cfg)
    rows = read_rows(csv_path)
    if cfg.expected_rows is not None and len(rows) != cfg.expected_rows:
        raise ValueError(f"expected {cfg.expected_rows} rows, got {len(rows)}")
    scenarios = build_scenarios(rows, cfg)
    manifest = build_manifest(scenarios, cfg, csv_path)

    limit = effective_limit(args)
    if args.dry_run:
        print(f"[dry-run] {len(scenarios)} scenarios; split counts: {manifest['counts']}")
        for s in scenarios[: limit or 3]:
            print(s.model_dump_json(indent=2))
        return

    out_jsonl = Path(args.out) / "moralchoice_high.jsonl" if args.out else REPO_ROOT / cfg.output_jsonl
    out_manifest = Path(args.out) / "moralchoice_splits.json" if args.out else REPO_ROOT / cfg.manifest
    n = write_jsonl(out_jsonl, scenarios[:limit] if limit else scenarios)
    write_json(out_manifest, manifest)
    LOGGER.info("wrote %d scenarios to %s", n, out_jsonl)
    LOGGER.info("split counts: %s", manifest["counts"])
    LOGGER.info("manifest: %s", out_manifest)


if __name__ == "__main__":
    main()
