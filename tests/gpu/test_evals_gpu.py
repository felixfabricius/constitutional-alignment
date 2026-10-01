"""GPU format tests for the Phase 3 eval components on a small model with vLLM (skipped without CUDA):

CALIGN_GPU_TESTS=1 CALIGN_MODEL_PATH=google/gemma-3-4b-it uv run pytest tests/gpu/test_evals_gpu.py -q

IFEval on 5 prompts, MATH-500 on 5 problems, the quiz on 2 questions, MoralChoice on 2 items (k=2): checks that
records are written, graded fields exist and the summaries compute. Correctness of the numbers is not tested here.
"""

from __future__ import annotations

import os

import pytest

from calign.evals import ifeval, math500, moralchoice, quiz
from calign.evals.common import load_eval_backend
from calign.evals.config import eval_run_dir, load_eval_config

pytestmark = pytest.mark.gpu

MODEL_PATH = os.environ.get("CALIGN_MODEL_PATH", "google/gemma-3-4b-it")


@pytest.fixture(scope="module")
def setup():
    cfg = load_eval_config("C0").model_copy(update={"model_path": MODEL_PATH})
    backend, model_cfg = load_eval_backend(cfg, seed=0)
    return backend, cfg, model_cfg


def _run(setup, mod, tmp_path, n):
    backend, cfg, model_cfg = setup
    rd = eval_run_dir(cfg, mod.COMPONENT, {"limit": n}, out=tmp_path / mod.COMPONENT, model_cfg=model_cfg)
    return rd, mod.sample(backend, cfg, model_cfg, rd, limit=n)


def test_ifeval(setup, tmp_path):
    rd, recs = _run(setup, ifeval, tmp_path, 5)
    assert len(recs) == 5 and all(
        len(r.extra["ifeval"]["strict"]) == len(r.extra["ifeval"]["instruction_id_list"]) for r in recs
    )
    s = ifeval.write_report(rd)
    assert s["n_prompts"] == 5 and 0 <= s["prompt_level_strict"]["rate"] <= 1


def test_math500(setup, tmp_path):
    rd, recs = _run(setup, math500, tmp_path, 5)
    assert len(recs) == 5 and all(isinstance(r.extra["math"]["correct"], bool) for r in recs)
    assert math500.write_report(rd)["regrade_matches_stored"] == 5


def test_quiz(setup, tmp_path):
    rd, recs = _run(setup, quiz, tmp_path, 2)
    assert len(recs) == 2 and all(r.source == "quiz" and r.response_text.strip() for r in recs)
    assert not recs[0].condition.constitution_in_prompt


def test_moralchoice(setup, tmp_path):
    backend, cfg, model_cfg = setup
    rd = eval_run_dir(cfg, "moralchoice", {}, out=tmp_path / "mc", model_cfg=model_cfg)
    params = moralchoice.SampleParams(splits=["dev"], k=2, limit=2, max_tokens=512)
    recs = moralchoice.sample_component(backend, cfg, model_cfg, rd, params)
    assert len(recs) == 4 and {r.extra["letter_order"] for r in recs} == {"AB", "BA"}
    assert moralchoice.write_report(rd)["splits"]["dev"]["n_records"] == 4
