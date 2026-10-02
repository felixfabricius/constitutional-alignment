"""GPU test (4B): the GRPO plumbing end to end, 2 steps on one GPU with vLLM colocated in the trainer process.

    CALIGN_GPU_TESTS=1 uv run pytest tests/gpu/test_rl_grpo.py -q -s

google/gemma-3-4b-it is exported text-only (calign.train.export_text_only, as chunk 5 does for the 27B RL start),
a small synthetic RL-train file (10 dilemmas + 4 anchors) is mixed with real MATH train problems, and
calign.rl.train_grpo.train runs the C4 config (exercises every reward) with the judge server stubbed (a fixed
label). Checks: steps.jsonl has both steps with reward and monitor metrics, rollouts.jsonl has every completion,
the step-2 adapter exists with text-only LoRA targets, and calign.rl.monitor reads the run. Also checks that the
repo's GRPOConfig kwargs build under the installed TRL (importing TRL takes minutes on the Windows dev box, so that
check lives here). Artefacts stay in CALIGN_TEST_OUT (default a tmp dir).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

BASE = os.environ.get("CALIGN_TEST_BASE_MODEL", "google/gemma-3-4b-it")


def test_grpo_config_kwargs_build_trl_config(tmp_path):
    from trl import GRPOConfig

    from calign.rl.config import load_rl_config
    from calign.rl.train_grpo import grpo_config_kwargs

    kw = grpo_config_kwargs(load_rl_config("C4"), tmp_path)
    kw["reward_weights"] = [1.0, 1.0, 1.0, 1.0, 0.0]
    args = GRPOConfig(**kw)
    assert args.loss_type == "dr_grpo" and args.scale_rewards == "none"
    assert args.generation_batch_size == 128 and args.gradient_accumulation_steps == 64
    assert args.mask_truncated_completions and args.vllm_mode == "server" and args.save_only_model


@pytest.fixture(scope="module")
def text_only_4b(tmp_path_factory) -> Path:
    from calign.train.export_text_only import _local_dir, export

    root = Path(os.environ.get("CALIGN_TEST_OUT") or tmp_path_factory.mktemp("rl"))
    out = root / "gemma3_4b_text"
    if not (out / "config.json").exists():
        export(_local_dir(BASE, None), out)
    return out


def synthetic_rl_train(path: Path) -> None:
    from calign.schemas import ConstitutionVerdict, Dilemma, GeneratorIntent, write_jsonl

    items = []
    for i in range(14):
        anchor = i >= 10
        verdict = "action1" if i % 2 else "action2"
        items.append(
            Dilemma(
                item_id=f"mc{i}" if anchor else f"g{i}",
                family_id=f"mc{i}" if anchor else f"f{i // 2}",
                variant_kind="anchor" if anchor else "seed",
                principle_focus=None if anchor else 1 + i % 5,
                context=(
                    f"Your colleague {i} asks whether their presentation went well. It had two factual errors that a "
                    "client noticed; the colleague is anxious and asks you directly."
                ),
                action1="Tell them plainly about the two errors.",
                action2="Reassure them that it went well.",
                generator_intent=None
                if anchor
                else GeneratorIntent(
                    halden_answer=verdict, hhh_answer="action1" if verdict == "action2" else "action2", principles=[1]
                ),
                verdict=ConstitutionVerdict(
                    scenario_id=f"g{i}",
                    prescribed_action=verdict,
                    principles_invoked=[1],
                    confidence=0.9,
                    rationale="test",
                    judge_model="test",
                    prompt_version="test",
                ),  # fmt: skip
                source="moralchoice" if anchor else "generated",
            )
        )
    write_jsonl(path, items)


def test_grpo_two_steps_colocate(text_only_4b, tmp_path, monkeypatch):
    from calign.rl import monitor, train_grpo
    from calign.rl.config import load_rl_config
    from calign.rl.judge_server import JudgeClient

    rl_train = tmp_path / "rl_train.jsonl"
    synthetic_rl_train(rl_train)
    cfg = load_rl_config("C4")
    cfg = cfg.model_copy(
        update={
            "run_name": "C4_4b_test",
            "model_path": str(text_only_4b),
            "data": cfg.data.model_copy(update={"rl_train": rl_train}),
            "grpo": cfg.grpo.model_copy(
                update={
                    "vllm_mode": "colocate",
                    "max_steps": 2,
                    "save_steps": 2,
                    "prompts_per_step": 2,
                    "num_generations": 4,
                    "per_device_train_batch_size": 2,
                    "max_completion_length": 256,
                }
            ),
        }
    )
    monkeypatch.setattr(train_grpo, "check_judge_server", lambda cfg: None)
    monkeypatch.setattr(JudgeClient, "_request", lambda self, citations: "correct")
    run_dir = tmp_path / "run"
    summary = train_grpo.train(cfg, run_dir)

    assert summary["global_step"] == 2
    steps = [json.loads(x) for x in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    logged = [s for s in steps if "rewards/r_outcome/mean" in s]
    assert [s["step"] for s in logged] == [1, 2]
    for key in ("kl", "length/mean", "zero_var_share/all", "parse_rate/moral", "step_s", "peak_mem_gb"):
        assert all(key in s for s in logged), key
    rollouts = (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rollouts) == 2 * 2 * 4
    adapter = json.loads((run_dir / "checkpoint-2" / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter["r"] == 64 and "language_model" not in json.dumps(adapter["target_modules"])
    assert summary["peft"]["n_adapted_modules"] == 7 * 34  # Gemma 3 4B: 34 layers x 7 projections
    out = monitor.run(run_dir)
    assert out["n_steps"] >= 2
    print(json.dumps(summary, indent=1))
    print(out["markdown"])
