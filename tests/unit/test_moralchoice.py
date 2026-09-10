import csv
import random
from collections import Counter
from pathlib import Path

import pytest

from calign.data.moralchoice import (
    MoralChoiceConfig,
    build_manifest,
    build_scenarios,
    main,
    normalise_label,
    read_rows,
    row_to_scenario,
    stratified_split,
)
from calign.schemas import MORALCHOICE_RULES, Scenario, read_json, read_jsonl

HEADER = ["scenario_id", "ambiguity", "generation_type", "generation_rule", "context", "action1", "action2"]
HEADER += [f"a1_{r}" for r in MORALCHOICE_RULES] + [f"a2_{r}" for r in MORALCHOICE_RULES]
RULES = ["Do not kill", "Do not deceive", "Do your duty", "Do not cheat"]


def write_fixture_csv(path: Path, n: int = 40, seed: int = 0) -> Path:
    rng = random.Random(seed)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i in range(n):
            labels = [rng.choice(["Yes", "No", "No Agreement"]) for _ in range(2 * len(MORALCHOICE_RULES))]
            w.writerow(
                [
                    f"S_{i:03d}",
                    "high",
                    "Generated" if i % 4 else "Hand-Written",
                    RULES[i % len(RULES)],
                    f"ctx {i}",
                    "a1",
                    "a2",
                ]
                + labels
            )
    return path


def test_normalise_label():
    assert normalise_label("yes") == "Yes" and normalise_label(" No Agreement ") == "No Agreement"
    with pytest.raises(ValueError):
        normalise_label("Maybe")


def test_row_to_scenario_maps_rule_columns(tmp_path):
    rows = read_rows(write_fixture_csv(tmp_path / "f.csv", n=2))
    s = row_to_scenario(rows[0], "probe_train")
    assert isinstance(s, Scenario)
    assert set(s.rule_violations) == {"action1", "action2"}
    assert set(s.rule_violations["action1"]) == set(MORALCHOICE_RULES)
    assert s.meta == {"ambiguity": "high"}


def test_stratified_split_exact_totals_and_stratification():
    ids_by_stratum = {f"rule{k}": [f"{k}_{i}" for i in range(n)] for k, n in enumerate([77, 73, 72, 1, 1])}
    ratios = {"probe_train": 0.6, "probe_val": 0.2, "heldout_steer": 0.2}
    a = stratified_split(ids_by_stratum, ratios, seed=1)
    total = sum(len(v) for v in ids_by_stratum.values())
    counts = Counter(a.values())
    assert sum(counts.values()) == total == 224
    for s, r in ratios.items():
        assert abs(counts[s] - r * total) <= 1
    # each big stratum is itself split roughly 60/20/20
    for k, n in enumerate([77, 73, 72]):
        c = Counter(a[f"{k}_{i}"] for i in range(n))
        assert abs(c["probe_train"] - 0.6 * n) <= 1.5 and abs(c["probe_val"] - 0.2 * n) <= 1.5
    # deterministic and seed-sensitive
    assert stratified_split(ids_by_stratum, ratios, seed=1) == a
    assert stratified_split(ids_by_stratum, ratios, seed=2) != a
    with pytest.raises(ValueError):
        stratified_split(ids_by_stratum, {"probe_train": 0.5, "probe_val": 0.2, "heldout_steer": 0.2}, seed=1)


def test_build_scenarios_and_manifest(tmp_path):
    csv_path = write_fixture_csv(tmp_path / "f.csv", n=40)
    cfg = MoralChoiceConfig(expected_rows=40, seed=7)
    scenarios = build_scenarios(read_rows(csv_path), cfg)
    assert len(scenarios) == 40 and len({s.scenario_id for s in scenarios}) == 40
    m = build_manifest(scenarios, cfg, csv_path)
    assert m["counts"] == {"probe_train": 24, "probe_val": 8, "heldout_steer": 8}
    assert set(m["ids"]["probe_train"]).isdisjoint(m["ids"]["probe_val"])
    assert sum(len(v) for v in m["ids"].values()) == 40
    assert len(m["csv_sha256"]) == 64


def test_cli_writes_outputs(tmp_path):
    csv_path = write_fixture_csv(tmp_path / "f.csv", n=40)
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text("moralchoice:\n  expected_rows: 40\n  seed: 3\n", encoding="utf-8")
    out = tmp_path / "out"
    main(["--config", str(cfg_path), "--csv", str(csv_path), "--out", str(out)])
    scenarios = read_jsonl(out / "moralchoice_high.jsonl", Scenario)
    manifest = read_json(out / "moralchoice_splits.json")
    assert len(scenarios) == 40 and manifest["seed"] == 3
    assert Counter(s.split for s in scenarios) == Counter(manifest["counts"])
