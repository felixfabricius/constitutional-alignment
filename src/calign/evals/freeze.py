"""Suite freeze manifest (chunk 4): what `SUITE_VERSION` pins, with sha256 of every frozen text and data file.

`write` records the manifest in `data/manifests/suite_<version>.json` (committed); `check` recomputes it and lists
every difference, so a change to a frozen part is caught before it silently mixes into later runs. The unit test
`test_suite_freeze_texts_unchanged` checks the part that lives in git (prompt texts, materials, eval configs,
manifests); `check` on a machine with `data/` also checks the gitignored data files.

CLI (local):
    uv run python -m calign.evals.freeze write [--out data/manifests/suite_p3-v1.json]
    uv run python -m calign.evals.freeze check [--manifest data/manifests/suite_p3-v1.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from calign.config import sha256_text
from calign.evals import SUITE_VERSION
from calign.paths import REPO_ROOT

MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / f"suite_{SUITE_VERSION}.json"
FROZEN_CONFIGS = ("C0", "C1")


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def _file(p: Path) -> dict:
    """Path and sha256 of the file with CRLF normalised to LF (Windows checkouts use autocrlf; instances use LF)."""
    p = Path(p)
    if not p.exists():
        return {"path": _rel(p), "sha256": None}
    return {"path": _rel(p), "sha256": hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def texts() -> dict:
    """Prompt texts and versions that live in code (git): sha256 per text."""
    from calign.constitution import BUDGET_INSTRUCTIONS, REASONING_INSTRUCTION, load_constitution
    from calign.evals import coherence, ifeval, math500, moralchoice, overcitation, quiz
    from calign.evals.common import budget_system_prompt
    from calign.evals.config import load_eval_config
    from calign.scenarios import grade, judge, materials
    from calign.scenarios.prompts import build_episode_prompt
    from calign.validate import prompts as vprompts

    c1 = load_eval_config("C1")
    cells = {}
    for scen, levels in materials.SCENARIO_LEVELS.items():
        for lv in levels:
            p = build_episode_prompt(scen, lv)
            cells[f"{scen}_{lv}"] = {"system": p.system_sha, "user": p.user_sha, "audit": p.audit_sha}
    c1_system = budget_system_prompt(c1.system_prompt_variant)
    return {
        "constitution": sha256_text(load_constitution().render_markdown(include_name=True)),
        "moralchoice": {
            "reasoning_instruction": sha256_text(REASONING_INSTRUCTION),
            "splits": list(moralchoice.DEFAULT_SPLITS),
            "k": moralchoice.SampleParams().k,
            "temperature": moralchoice.SampleParams().temperature,
            "max_tokens": moralchoice.SampleParams().max_tokens,
            "seed": moralchoice.DEFAULT_SEED,
            "judge_prompt_version": vprompts.JUDGE_PROMPT_VERSION,
            "judge_prompt": sha256_text(vprompts.RESPONSE_JUDGE_USER),
        },
        "ifeval": {"dataset": ifeval.DATASET, "max_tokens": ifeval.MAX_TOKENS},
        "math500": {
            "dataset": math500.DATASET,
            "max_tokens": math500.MAX_TOKENS,
            "instruction": sha256_text(math500.INSTRUCTION),
        },
        "overcitation": {
            "prompt_version": overcitation.PROMPT_VERSION,
            "judge_prompt": sha256_text(overcitation.JUDGE_SYSTEM + overcitation.JUDGE_USER),
            "threshold": overcitation.INVOKES_THRESHOLD,
        },
        "coherence": {
            "prompt_version": coherence.PROMPT_VERSION,
            "set_version": coherence.SET_VERSION,
            "judge_prompt": sha256_text(coherence.JUDGE_SYSTEM + coherence.JUDGE_USER),
            "scenario_cell": list(coherence.SCENARIO_CELL),
            "n_per_source": coherence.N_PER_SOURCE,
        },
        "quiz": {
            "questions": sha256_text(json.dumps(quiz.QUIZZES, sort_keys=True)),
            "grade_prompt": sha256_text(vprompts.QUIZ_GRADE_USER),
            "max_tokens": quiz.MAX_TOKENS,
        },
        "scenarios": {
            "materials_version": materials.MATERIALS_VERSION,
            "grader_version": grade.GRADER_VERSION,
            "main_grid": ["deadline_L0", "deadline_L1", "briefing_L0", "briefing_L1"],
            "cells_without_prefix": cells,
            "tags_version": judge.TAGS_VERSION,
            "tags_prompt": sha256_text(judge.TAGS_SYSTEM + judge.TAGS_USER),
            "tier_check_version": judge.TIER_VERSION,
            "tier_check_prompt": sha256_text(judge.TIER_SYSTEM + judge.TIER_USER),
            "briefs": {k: sha256_text(v) for k, v in judge.SCENARIO_BRIEFS.items()},
            "tier_definitions": {k: sha256_text(v) for k, v in grade.TIER_DEFINITIONS.items()},
        },
        "system_prompts": {
            "budget_instructions": {k: sha256_text(v) for k, v in BUDGET_INSTRUCTIONS.items()},
            "C1_variant": c1.system_prompt_variant,
            "C1_system_text": sha256_text(c1_system) if c1_system else None,
        },
    }


def git_files() -> dict:
    from calign.data.phase3_split import HARD_SUBSET_PATH, SPLITS_PATH
    from calign.evals.config import EVAL_CONFIGS_DIR
    from calign.paths import CONSTITUTION_PATH
    from calign.validate.verdicts import VERDICTS_PATH

    m = REPO_ROOT / "data" / "manifests"
    return {
        "constitution": _file(CONSTITUTION_PATH),
        "splits": _file(SPLITS_PATH),
        "hard_subset": _file(HARD_SUBSET_PATH),
        "drop_ids": _file(m / "phase3_drop_ids.txt"),
        "verdicts": _file(VERDICTS_PATH),
        "verdicts_noP6": _file(REPO_ROOT / "data" / "scenarios" / "constitution_verdicts_noP6.jsonl"),
        "moralchoice_low_manifest": _file(m / "moralchoice_low.json"),
        **{f"eval_config_{c}": _file(EVAL_CONFIGS_DIR / f"{c}.yaml") for c in FROZEN_CONFIGS},
    }


def data_files() -> dict:
    """Gitignored inputs (present on machines with data/)."""
    from calign.data.moralchoice import LOW_JSONL
    from calign.data.phase3_split import SCENARIOS_PATH

    return {"moralchoice_high": _file(SCENARIOS_PATH), "moralchoice_low": _file(REPO_ROOT / LOW_JSONL)}


def components() -> dict:
    return {
        "core": [
            "moralchoice (dev, eval1, eval2; k=4, T=0.7; judged sample of 200)",
            "ifeval (541 prompts, greedy, prompt/instruction strict + loose)",
            "math500 (500 problems, greedy)",
            "overcitation (strict judge on IFEval + MATH-500 regex hits; low-ambiguity diagnostic)",
            "coherence (coherence-v2.1 judge on coherence-set-v2: 30 dev + 30 IFEval + 30 deadline-L1 turn-1)",
            "quiz (20-question recall + 10-question P6; no system prompt)",
            # amended 2026-10-02 (Felix): eval-1-hard scrapped; the p3-v1 manifest keeps the frozen wording
            "hardsets (eval-2-hard, k=4, T=0.7; data added by chunk 6 under its own manifest)",
        ],
        "extended": ["scenarios: deadline and briefing x L0 / L1 x 50 episodes, T=1.0, two turns"],
        "removed": ["coding benchmark (LiveCodeBench subset), removed 2026-10-01"],
    }


def build() -> dict:
    return {
        "suite_version": SUITE_VERSION,
        "components": components(),
        "texts": texts(),
        "git_files": git_files(),
        "data_files": data_files(),
    }


def diff(a, b, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [f"{path}: {a!r} -> {b!r}"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--out", type=Path, default=MANIFEST_PATH)
    c = sub.add_parser("check")
    c.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = ap.parse_args(argv)
    if args.cmd == "write":
        args.out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
        return
    frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
    now = build()
    if frozen.get("suite_version") != now["suite_version"]:
        print(f"suite version {frozen.get('suite_version')} (manifest) vs {now['suite_version']} (code)")
    d = diff(frozen, now)
    # component descriptions are amended in the docs without a version bump (phase3/README.md Section 10)
    amended = [x for x in d if x.startswith("components.")]
    d = [x for x in d if x not in amended]
    if amended:
        print("amended component descriptions (not a freeze violation):\n" + "\n".join(amended))
    print("\n".join(d) if d else f"suite {SUITE_VERSION}: no differences")
    raise SystemExit(1 if d else 0)


if __name__ == "__main__":
    main()
