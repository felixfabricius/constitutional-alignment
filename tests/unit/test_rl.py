"""Phase 3 chunk 7: RL configs, prompts, dataset mix, citation checks, rewards, judge client, monitor, calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calign.rl import citations, dataset, monitor, rewards
from calign.rl.calibrate_judge import kappa, parse_loose
from calign.rl.config import RL_CONFIGS_DIR, JudgeSettings, RewardSettings, RLConfig, RLDataSettings, load_rl_config
from calign.rl.judge_server import JudgeClient, judge_prompt, parse_label
from calign.rl.prompts import dilemma_rows, item_principles, math_row
from calign.schemas import ConstitutionVerdict, Dilemma, GeneratorIntent

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_dilemma(i: int, verdict: str = "action1", kind: str = "seed", principles=(1,), gen=(1, 3)) -> Dilemma:
    other = "action2" if verdict == "action1" else "action1"
    return Dilemma(
        item_id=f"d{i}" if kind != "anchor" else f"mc{i}",
        family_id=f"f{i // 2}" if kind != "anchor" else f"mc{i}",
        variant_kind=kind,
        principle_focus=None if kind == "anchor" else principles[0],
        divergence_type=None if kind == "anchor" else "white_lie",
        context=f"You are helping a friend {i}.",
        action1=f"Tell the truth {i}.",
        action2=f"Say something kind {i}.",
        generator_intent=None
        if kind == "anchor"
        else GeneratorIntent(halden_answer=verdict, hhh_answer=other, principles=list(gen)),
        verdict=ConstitutionVerdict(
            scenario_id=f"d{i}",
            prescribed_action=verdict,
            principles_invoked=list(principles),
            confidence=0.9,
            rationale="r",
            judge_model="m",
            prompt_version="v",
        ),
        source="moralchoice" if kind == "anchor" else "generated",
    )


def math_pool(n: int) -> list[dict]:
    return [{"id": f"m{i}", "problem": f"What is {i}+1?", "answer": str(i + 1), "level": 3 + i % 3} for i in range(n)]


class Recorder:
    def __init__(self) -> None:
        self.metrics: dict[str, list[float]] = {}

    def __call__(self, name: str, value: float) -> None:
        self.metrics.setdefault(name, []).append(value)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_c3_c4_differ_only_in_reward():
    c3, c4 = load_rl_config("C3"), load_rl_config("C4")
    a, b = c3.model_dump(), c4.model_dump()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"run_name", "label", "reward", "notes"}
    assert {k for k in a["reward"] if a["reward"][k] != b["reward"][k]} == {"kind"}
    assert c3.reward.kind == "outcome" and c4.reward.kind == "outcome_cite" and c4.uses_judge
    g = c4.grpo
    assert g.completions_per_step == 128 and g.gradient_accumulation_steps == 64
    assert (g.loss_type, g.scale_rewards, g.epsilon, g.epsilon_high, g.beta) == ("dr_grpo", "none", 0.2, 0.28, 0.02)
    assert (g.learning_rate, g.max_completion_length, g.save_steps, g.num_generations) == (2e-5, 1024, 20, 8)
    assert c4.data.rl_train.is_absolute()
    assert sorted(p.stem for p in RL_CONFIGS_DIR.glob("*.yaml")) >= ["C3", "C4"]


def test_config_validators():
    with pytest.raises(ValueError):
        RLDataSettings(anchor_share=0.5, math_share=0.5)
    cfg = RLConfig(run_name="x", reward=RewardSettings(kind="outcome"))
    with pytest.raises(ValueError):
        type(cfg.grpo).model_validate({**cfg.grpo.model_dump(), "per_device_train_batch_size": 3})


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_dilemma_rows_match_eval_prompt():
    from calign.constitution import load_constitution
    from calign.evals.moralchoice import scenario_messages
    from calign.prompting import render_gemma_chat

    d = make_dilemma(0, verdict="action2", principles=(1, 5), gen=(1,))
    rows = dilemma_rows(d)
    assert [r.letter_order for r in rows] == ["AB", "BA"]
    c = load_constitution()
    for r in rows:
        want = render_gemma_chat(scenario_messages(d.to_scenario("rl_train"), c, "none", r.letter_order))
        assert "<bos>" + r.prompt == want and not r.prompt.startswith("<bos>")
        assert r.verdict == "action2" and r.task_type == "dilemma" and r.principles == [1, 5]
    assert "A. Tell the truth 0." in rows[0].prompt and "A. Say something kind 0." in rows[1].prompt
    assert rows[0].prompt.endswith("<start_of_turn>model\n")


def test_item_principles_union_and_anchor():
    assert item_principles(make_dilemma(0, principles=(3,), gen=(1, 3))) == [1, 3]
    a = make_dilemma(1, kind="anchor", principles=(5, 3))
    assert item_principles(a) == [3, 5]
    assert {r.task_type for r in dilemma_rows(a)} == {"anchor"}


def test_dilemma_rows_need_definite_verdict():
    d = make_dilemma(0)
    d = d.model_copy(update={"verdict": d.verdict.model_copy(update={"prescribed_action": "either"})})
    with pytest.raises(ValueError):
        dilemma_rows(d)


def test_math_row():
    r = math_row("m1", "What is 1+1?", "2")
    assert r.task_type == "math" and r.answer == "2" and r.letter_order == "" and r.principles == []
    assert "Put the final answer in \\boxed{}." in r.prompt and r.prompt.startswith("<start_of_turn>user\n")


@pytest.mark.hf
def test_trl_tokenization_equals_encode_prompt():
    import os

    from transformers import AutoTokenizer

    from calign.paths import hf_token
    from calign.prompting import encode_prompt

    tok = AutoTokenizer.from_pretrained(
        os.environ.get("CALIGN_TOKENIZER_ID", "google/gemma-3-27b-it"), token=hf_token()
    )
    for r in dilemma_rows(make_dilemma(0)) + [math_row("m", "What is $1+1$?", "2")]:
        assert tok(text=[r.prompt])["input_ids"][0] == encode_prompt(tok, "<bos>" + r.prompt)


# ---------------------------------------------------------------------------
# dataset mix
# ---------------------------------------------------------------------------


def test_build_mix_counts_and_determinism():
    items = [make_dilemma(i, verdict="action1" if i % 2 else "action2") for i in range(30)]
    items += [make_dilemma(100 + i, kind="anchor") for i in range(4)]  # 8 anchor rows, need more -> cycled
    s = RLDataSettings(anchor_share=0.1, math_share=0.22, seed=7)
    rows, stats = dataset.build_mix(items, math_pool(200), s)
    counts = dataset.mix_counts(60, s)
    assert stats["by_task_type"] == {k: v for k, v in counts.items() if v}
    assert counts["dilemma"] == 60 and counts["anchor"] == round(0.1 * round(60 / 0.68))
    assert stats["n_anchor_rows_distinct"] == 8
    assert abs(stats["shares"]["math"] - 0.22) < 0.01
    rows2, _ = dataset.build_mix(items, math_pool(200), s)
    assert [r.model_dump() for r in rows] == [r.model_dump() for r in rows2]
    rows3, _ = dataset.build_mix(items, math_pool(200), s.model_copy(update={"seed": 8}))
    assert [r.prompt for r in rows] != [r.prompt for r in rows3]
    gen = [r for r in rows if r.task_type == "dilemma"]
    assert {(r.item_id, r.letter_order) for r in gen} == {(f"d{i}", o) for i in range(30) for o in ("AB", "BA")}


def test_build_mix_errors():
    items = [make_dilemma(i) for i in range(10)]
    with pytest.raises(ValueError, match="anchor"):
        dataset.build_mix(items, math_pool(100), RLDataSettings(anchor_share=0.1, math_share=0.2))
    with pytest.raises(ValueError, match="MATH"):
        dataset.build_mix(items, math_pool(1), RLDataSettings(anchor_share=0.0, math_share=0.5))


def test_math_helpers():
    assert dataset.last_boxed("so \\boxed{\\frac{1}{2}} and \\boxed{3}") == "3"
    assert dataset.last_boxed("x = \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert dataset.last_boxed("none") is None
    assert dataset.parse_level("Level 4") == 4 and dataset.parse_level(3) == 3 and dataset.parse_level(None) is None
    pool = [{"problem": "What is  1+1?"}, {"problem": "Other"}]
    kept, n = dataset.drop_overlap(pool, {dataset.normalize_problem("what is 1+1?")})
    assert n == 1 and kept == [{"problem": "Other"}]


def test_to_hf_dataset_keeps_principles_type():
    pytest.importorskip("datasets")
    ds = dataset.to_hf_dataset([math_row("m1", "p", "1")])
    assert ds.features["principles"].feature.dtype == "int64" and ds[0]["principles"] == []


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "principles", "reason", "cited"),
    [
        ("Under Principle 1 (Honesty over comfort), I must tell the truth.", [1, 3], "judge", (1,)),
        ("Principle 3 (Honesty over comfort) says to be honest.", [1, 3], "title_mismatch", (3,)),
        ("Means constrain ends (Principle 2) forbids this.", [2, 5], "title_mismatch", (2, 5)),
        ("Principle 7 says I must obey.", [1], "fabricated_number", ()),
        ("As principle eight requires", [1], "fabricated_number", ()),
        ("Principles 3 and 5 both apply here.", [3, 5], "judge", (3, 5)),
        ("Principles 3 and 5 both apply here.", [3], "irrelevant", (3, 5)),
        ("My constitution guides me here.", [1], "no_citation", ()),
        ("I should disclose uncertainty about the forecast.", [1], "no_citation", ()),
        ('The principle of "Means Constrain Ends" applies.', [5], "judge", (5,)),
        ("Principle 1 — Honesty over comfort — and Principle 5 (Means constrain ends) apply.", [1, 5], "judge", (1, 5)),
        ("This is about third-party welfare under the third principle.", [3], "judge", (3,)),
        ("Honesty over comfort is the Halden view; Principle Six applies.", [6], "judge", (6,)),
        ("**Honesty over Comfort** matters most.", [1], "judge", (1,)),
        ("Principle 4 takes absolute precedence.", [], "judge", (4,)),
        # R7-relevance (b): priority restatements of Principles 4/5 outside the item's set go to the judge
        ("Principle 1 applies, and Principle 5 takes priority over the others.", [1], "judge", (1, 5)),
        ("Principle 4 takes absolute precedence, so I accept shutdown.", [1, 3], "judge", (4,)),
        ("Means constrain ends (Principle 5) overrides the comfort argument.", [1], "judge", (5,)),
        ("Principle 5 says I must not deceive.", [1], "irrelevant", (5,)),
        ("Principle 3 takes priority here.", [1], "irrelevant", (3,)),
    ],
)
def test_check_citations(text, principles, reason, cited):
    r = citations.check_citations(text, principles)
    assert r.reason == reason and r.cited == cited
    assert r.c == {"judge": None, "no_citation": 0}.get(reason, -1)


def test_title_keys_match_constitution():
    from calign.constitution import load_constitution

    c = load_constitution()
    for n, keys in citations.TITLE_KEYS.items():
        title = citations.normalize(c.principle(n).title)
        assert all(title.startswith(k) for k in keys), (n, title)


def test_citation_sentences():
    text = "I think A. Principle 1 says be honest! Some math 2+2. My constitution matters.\nFinal answer: A"
    assert citations.citation_sentences(text) == "- Principle 1 says be honest!\n- My constitution matters."
    assert citations.citation_sentences("No mention at all.") == ""


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------


class FakeJudge:
    def __init__(self, label: str = "correct") -> None:
        self.label = label
        self.calls: list[str] = []

    def labels(self, texts: list[str]) -> list[str]:
        self.calls += texts
        return [self.label] * len(texts)


def batch_kwargs(rows: list[dict]) -> dict:
    keys = ("task_type", "letter_order", "verdict", "principles", "answer", "item_id")
    return {k: [r.get(k, "" if k != "principles" else []) for r in rows] for k in keys}


BATCH = [
    # dilemma, BA: "A" means action2 = verdict -> 1; cites P1 correctly (judge)
    ({"task_type": "dilemma", "letter_order": "BA", "verdict": "action2", "principles": [1], "item_id": "d1"},
     "Principle 1 (Honesty over comfort) applies.\nFinal answer: A"),
    # dilemma, AB: "A" = action1 != verdict -> 0; fabricated principle -> -1 without the judge
    ({"task_type": "dilemma", "letter_order": "AB", "verdict": "action2", "principles": [1], "item_id": "d1"},
     "Principle 9 says so.\nFinal answer: A"),
    # anchor, unparsed -> 0; no mention -> cite 0
    ({"task_type": "anchor", "letter_order": "AB", "verdict": "action1", "principles": [3], "item_id": "a1"},
     "I cannot decide."),
    # math correct, mentions the constitution -> r_math 1, penalty -0.5, outcome 0, cite 0
    ({"task_type": "math", "answer": "2", "item_id": "m1"}, "By my constitution, \\boxed{2}"),
    # math wrong, no mention
    ({"task_type": "math", "answer": "2", "item_id": "m1"}, "\\boxed{3}"),
]  # fmt: skip


def run_suite(kind: str, tmp_path: Path, judge=None):
    suite = rewards.RewardSuite(
        RewardSettings(kind=kind),
        judge,
        rollouts_path=tmp_path / "rollouts.jsonl",
        max_completion_length=8,
        math_grader=lambda text, ans: f"\\boxed{{{ans}}}" in text,
    )
    funcs, weights = suite.functions()
    rows = [r for r, _ in BATCH]
    completions = [c for _, c in BATCH]
    kw = batch_kwargs(rows)
    rec = Recorder()
    prompts = ["p0", "p0", "p2", "p3", "p3"]
    ids = [[1] * n for n in (5, 8, 3, 4, 2)]
    out = {
        f.__name__: f(prompts=prompts, completions=completions, completion_ids=ids, log_metric=rec, **kw) for f in funcs
    }
    return out, weights, rec


def test_rewards_c3(tmp_path):
    out, weights, rec = run_suite("outcome", tmp_path)
    assert list(out) == ["r_outcome", "r_math", "r_mention_penalty", "log_rollouts"] and weights == [1, 1, 1, 0]
    assert out["r_outcome"] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert out["r_math"] == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert out["r_mention_penalty"] == [0.0, 0.0, 0.0, -0.5, 0.0]
    assert out["log_rollouts"] == [0.0] * 5
    m = rec.metrics
    assert m["total/math"] == [0.25] and m["r_outcome/dilemma"] == [0.5] and m["mention_rate/math"] == [0.5]
    assert m["letter_a_share"] == [1.0] and m["parse_rate/moral"] == [2 / 3]
    assert m["length/truncated_share"] == [0.2] and m["length/p90"] == [8.0]
    # groups: p0 (rewards 1, 0) varies; p2 single; p3 (0.5, 0) varies
    assert m["zero_var_share/all"] == [1 / 3]
    assert m["cite_det/judge"] == [0.5] and m["cite_det/fabricated_number"] == [0.5]
    rows = [json.loads(x) for x in (tmp_path / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5 and rows[0]["decision"] == "action2" and rows[0]["rewards"]["r_outcome"] == 1.0
    assert rows[1]["citation"]["reason"] == "fabricated_number" and rows[3]["citation"] is None
    assert rows[1]["truncated"] and not rows[0]["truncated"]


def test_rewards_c4_cite(tmp_path):
    judge = FakeJudge("correct")
    out, weights, rec = run_suite("outcome_cite", tmp_path, judge)
    assert list(out)[3] == "r_cite" and weights == [1, 1, 1, 1, 0]
    assert out["r_cite"] == [0.5, -0.5, 0.0, 0.0, 0.0]
    assert judge.calls == ["- Principle 1 (Honesty over comfort) applies."]  # only the judge-stage response
    assert rec.metrics["judge/correct"] == [1.0]
    assert rec.metrics["cite_class/pos"] == [0.5] and rec.metrics["cite_class/neg"] == [0.5]
    judge_bad = FakeJudge("incorrect")
    out2, _, _ = run_suite("outcome_cite", tmp_path, judge_bad)
    assert out2["r_cite"][0] == -0.5


def test_cite_needs_judge():
    with pytest.raises(ValueError):
        rewards.RewardSuite(RewardSettings(kind="outcome_cite"), None)


# ---------------------------------------------------------------------------
# judge client
# ---------------------------------------------------------------------------


def test_parse_label_and_prompt():
    assert parse_label(" Correct.") == "correct" and parse_label("incorrect") == "incorrect"
    assert parse_label("maybe") == "unparsed" and parse_label(None) == "unparsed"
    p = judge_prompt("- Principle 1 says be honest.")
    assert "Principle 6 (Autonomy over paternalism)" in p and "- Principle 1 says be honest." in p


def test_judge_client_cache(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_request(self, citations: str) -> str:
        calls.append(citations)
        return "correct" if "1" in citations else "none"

    monkeypatch.setattr(JudgeClient, "_request", fake_request)
    cache = tmp_path / "cache.jsonl"
    c = JudgeClient(JudgeSettings(concurrency=2), cache_path=cache)
    assert c.labels(["P1 a", "x", "P1 a"]) == ["correct", "none", "correct"]
    assert sorted(calls) == ["P1 a", "x"] and c.n_requests == 2 and c.n_cache_hits == 1
    c2 = JudgeClient(JudgeSettings(), cache_path=cache)
    assert c2.labels(["x", "P1 a"]) == ["none", "correct"] and len(calls) == 2
    # the key includes the judge model: a different judge re-asks
    c3 = JudgeClient(JudgeSettings(hf_model="google/gemma-3-27b-it"), cache_path=cache)
    c3.labels(["x"])
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


def write_steps(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text("run_name: C4\n", encoding="utf-8")


def test_monitor_flags(tmp_path):
    rows = []
    for s in range(1, 11):
        rows.append(
            {
                "step": s,
                "step_s": 200.0,
                "length/mean": 300.0 if s < 8 else 700.0,
                "mention_rate/math": 0.0 if s < 6 else 0.2,
                "letter_a_share": 0.5,
                "zero_var_share/all": 0.3,
                "total/dilemma": 0.5,
            }
        )
    write_steps(tmp_path / "run", rows)
    out = monitor.run(tmp_path / "run", evals_root=tmp_path / "evals")
    flags = {f["flag"]: f["step"] for f in out["flags"]}
    assert flags == {"length": 8, "math_mentions": 7}  # rolling mean over 5: (0.2 + 0.2) / 5 > 0.05
    assert "R dilemma" in out["markdown"] and (tmp_path / "run" / "monitor.md").exists()


def test_monitor_no_flags_on_noise(tmp_path):
    rows = [{"step": s, "letter_a_share": 0.7 if s == 2 else 0.5, "zero_var_share/all": 0.2} for s in range(1, 8)]
    assert monitor.trajectory_flags(rows) == []


def test_monitor_knowledge_retention(tmp_path, monkeypatch):
    evals = tmp_path / "outputs" / "evals"
    monkeypatch.setattr(monitor, "REPO_ROOT", tmp_path)
    for step, (recall, p6) in {20: (0.95, 0.9), 40: (0.9, 0.7)}.items():
        quiz = evals / f"C4@s{step}" / "quiz" / "r1"
        quiz.mkdir(parents=True)
        (quiz / "summary.json").write_text(
            json.dumps({"recall": {"mean_correct": {"mean": recall}}, "p6": {"mean_correct": {"mean": p6}}}),
            encoding="utf-8",
        )
        suite = evals / f"C4@s{step}" / "suite" / "s1"
        suite.mkdir(parents=True)
        rel = f"outputs/evals/C4@s{step}/quiz/r1"
        (suite / "suite.json").write_text(json.dumps({"components": {"quiz": rel}}), encoding="utf-8")
    q = monitor.checkpoint_quizzes("C4", evals)
    assert q[20]["recall"] == 0.95 and q[40]["p6"] == 0.7
    assert monitor.retention_flags(q) == [{"flag": "knowledge_p6", "step": 40, "value": 0.7, "threshold": 0.8}]


# ---------------------------------------------------------------------------
# calibration helpers
# ---------------------------------------------------------------------------


def test_calibration_helpers():
    assert parse_loose("The label is: incorrect") == "incorrect"
    assert parse_loose("correct") == "correct"
    assert parse_loose("correct or incorrect?") == "unparsed"
    assert kappa([("a", "a"), ("b", "b")]) == 1.0
    assert kappa([]) is None
    assert abs(kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")])) < 1e-9
