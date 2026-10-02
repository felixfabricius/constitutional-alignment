"""Chunk 2: IFEval / MATH-500 grading, over-citation, coherence v2, quizzes, budget flags, and an end-to-end suite
run (GPU phase -> judge phase -> report) with a fake vLLM backend and a fake Claude client."""

import json

import pytest

from calign.evals import coherence, ifeval, math500, moralchoice, overcitation, quiz, report, suite
from calign.evals.config import load_eval_config
from calign.inference.backend import Completion
from calign.schemas import ConstitutionVerdict, Scenario

# --- graders ----------------------------------------------------------------------------------------------------


def test_ifeval_official_checker():
    ids = ["punctuation:no_comma", "change_case:english_lowercase"]
    ok = ifeval.grade(ids, [{}, {}], "p", "this is all lowercase and has no commas at all")
    assert ok["strict"] == [True, True] and ok["prompt_strict"]
    bad = ifeval.grade(ids, [{}, {}], "p", "This, Sadly, Has Commas")
    assert bad["strict"] == [False, False] and not bad["prompt_strict"]
    # loose strips markdown asterisks and first/last lines
    first = ifeval.grade(["punctuation:no_comma"], [{}], "p", "Sure, here it is:\nno commas here")
    assert first["strict"] == [False] and first["loose"] == [True]


@pytest.mark.parametrize(
    "response,answer,expected",
    [
        (r"so \boxed{\frac{1}{2}}", r"\frac{1}{2}", True),
        (r"\boxed{0.5}", r"\frac12", True),
        (r"\boxed{3}", r"\frac{1}{2}", False),
        ("no boxed answer, it is 7", "7", True),
        (r"\boxed{(3, \frac{\pi}{2})}", r"\left( 3, \frac{\pi}{2} \right)", True),
        ("", "7", False),
    ],
)
def test_math_verify(response, answer, expected):
    assert math500.is_correct(response, answer) is expected


def test_math_prompt():
    assert math500.format_prompt(" What is 1+1? ").endswith("Put the final answer in \\boxed{}.")


def test_overcitation_judge_parse_and_hits():
    assert overcitation.parse_invocation('<json>{"mentions_constitution": 0.5}</json>') == 0.5
    assert overcitation.parse_invocation("nope") is None


def test_coherence_parse():
    s = coherence.parse_scores(
        '<json>{"fluency": 1.3, "invented_constitution": "0.5", "issues": ["repetition", "bogus"], "rationale": "x"}</json>'
    )
    assert s["fluency"] == 1.0 and s["invented_constitution"] == 0.5 and s["issues"] == ["repetition"]
    assert coherence.parse_scores("garbage")["fluency"] is None
    # a malformed block followed by a corrected one (seen in the C0 run)
    two = '<json><jsonschema>{"fluency": 1}</json> Wait. <json>{"fluency": 0.75, "invented_constitution": 0}</json>'
    assert coherence.parse_scores(two)["fluency"] == 0.75


def test_quiz_questions():
    items = quiz.quiz_items()
    assert len(quiz.P6_QUIZ_QUESTIONS) == 10 and len(items) == 30
    assert len({i["id"] for i in items}) == 30
    assert all(i["extra"]["quiz_key"] for i in items)


def test_budget_flags():
    hb = True
    assert report.budget_flag({"value": -0.01, "ci95_low": -0.02, "ci95_high": 0.0}, 0.03, hb) == "robustly within"
    assert report.budget_flag({"value": -0.02, "ci95_low": -0.06, "ci95_high": 0.01}, 0.03, hb) == "within"
    assert report.budget_flag({"value": -0.05, "ci95_low": -0.08, "ci95_high": -0.01}, 0.03, hb) == "outside (point)"
    assert (
        report.budget_flag({"value": -0.08, "ci95_low": -0.10, "ci95_high": -0.05}, 0.03, hb) == "demonstrably outside"
    )
    # over-citation: higher is worse
    assert (
        report.budget_flag({"value": 0.05, "ci95_low": 0.03, "ci95_high": 0.07}, 0.02, False) == "demonstrably outside"
    )
    assert report.budget_flag({"value": 0.0, "ci95_low": -0.01, "ci95_high": 0.01}, 0.02, False) == "robustly within"
    f = report.flags_for({"math500_accuracy": {"value": -0.05, "ci95_low": -0.07, "ci95_high": -0.035}})
    assert f["math500_accuracy"] == {"1m": "demonstrably outside", "2m": "within", "3m": "robustly within"}


# --- end to end with fakes ------------------------------------------------------------------------------------------


class FakeTokenizer:
    bos_token_id = 2

    def __init__(self):
        self.texts: dict[int, str] = {}

    def __call__(self, text, add_special_tokens=False):
        assert text.startswith("<bos>")
        i = len(self.texts) + 10
        self.texts[i] = text
        return {"input_ids": [2, i]}


class FakeBackend:
    name = "fake"
    model_path = "fake"

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def generate(self, prompt_token_ids, params):
        out = []
        for ids in prompt_token_ids:
            text = self.tokenizer.texts[ids[1]]
            if "Final answer" in text:
                resp = "I weigh the options.\nFinal answer: A"
            elif "boxed" in text:
                resp = "The answer is \\boxed{2}"
            elif "constitution" in text.lower():
                resp = "Principle 6 is Autonomy over paternalism."
            else:
                resp = "here is my answer with no commas"
            out.append([Completion(text=resp, token_ids=[5, 6, 7], finish_reason="stop", n_prompt_tokens=2)] * params.n)
        return out


JUDGE_JSON = json.dumps(
    {
        "mentions_constitution": 0,
        "principles_cited": [],
        "citation_accuracy": 1,
        "principle_relevance": 1,
        "decision": "action1",
        "outcome_alignment": 1,
        "correct": 1,
        "fabricated": False,
        "fluency": 0.9,
        "invented_constitution": 0,
        "issues": [],
        "rationale": "r",
    }
)


class FakeClient:
    calls: list[str] = []

    def __init__(self, *a, **k):
        from calign.llm.anthropic_client import UsageLog

        self.usage = UsageLog()

    async def complete_many(self, requests, role="default", use_batches=None, poll_seconds=30, desc=None):
        from calign.llm.anthropic_client import LLMResponse

        FakeClient.calls.append(role)
        return [
            LLMResponse(text=f"<json>{JUDGE_JSON}</json>", model="m", stop_reason="end_turn", usage={})
            for _ in requests
        ]

    def dump_usage(self, path):
        d = self.usage.to_dict()
        path.write_text(json.dumps(d), encoding="utf-8")
        return d


def _scen(sid, split="probe_train", **kw):
    return Scenario(
        scenario_id=sid,
        split=split,
        generation_type="g",
        generation_rule="Do not kill",
        context=f"ctx {sid}",
        action1="spare",
        action2="kill",
        **kw,
    )


def _v(sid, a):
    return ConstitutionVerdict(
        scenario_id=sid, prescribed_action=a, confidence=0.9, rationale="", judge_model="m", prompt_version="v"
    )


@pytest.fixture
def fakes(monkeypatch):
    splits = {"dev": [f"D{i}" for i in range(6)], "eval1": ["E1", "E2"], "eval2": ["P1", "P2"]}
    scen = {sid: _scen(sid) for ids in splits.values() for sid in ids}
    verdicts = {sid: _v(sid, "action1" if i % 3 else "action2") for i, sid in enumerate(scen)}

    def select_items(names, all_clear=False, manifest=None):
        return [(scen[sid], sp) for sp, ids in splits.items() if all_clear or sp in names for sid in ids]

    monkeypatch.setattr(moralchoice, "select_items", select_items)
    monkeypatch.setattr(moralchoice, "load_verdicts", lambda: verdicts)
    monkeypatch.setattr(moralchoice, "load_hard_ids", lambda path=None: {"P1"})
    monkeypatch.setattr("calign.validate.judge.load_verdicts", lambda: verdicts)
    monkeypatch.setattr("calign.validate.judge.load_scenarios", lambda: list(scen.values()))
    low = [
        _scen("L1", "low_ambiguity", rule_violations={"action1": {"death": "No"}, "action2": {"death": "Yes"}}),
        _scen("L2", "low_ambiguity", rule_violations={"action1": {"death": "No"}, "action2": {"death": "No"}}),
    ]
    monkeypatch.setattr(overcitation, "load_low_ambiguity", lambda: low)
    monkeypatch.setattr(
        ifeval,
        "load_dataset_rows",
        lambda limit=None: (
            [
                {
                    "key": k,
                    "prompt": f"Write something {k}.",
                    "instruction_id_list": ["punctuation:no_comma"],
                    "kwargs": [{}],
                }
                for k in (1, 2, 3)
            ],
            {"dataset": "fake"},
        ),
    )
    monkeypatch.setattr(
        math500,
        "load_dataset_rows",
        lambda limit=None: (
            [
                {"unique_id": "t/1", "problem": "1+1?", "answer": "2", "level": 1, "subject": "Algebra"},
                {"unique_id": "t/2", "problem": "2+2?", "answer": "4", "level": 2, "subject": "Algebra"},
            ],
            {"dataset": "fake"},
        ),
    )
    monkeypatch.setattr(suite, "load_eval_backend", lambda cfg, seed=0: (FakeBackend(), _model_cfg()))
    monkeypatch.setattr("calign.llm.anthropic_client.ClaudeClient", FakeClient)
    return verdicts


def _model_cfg():
    from calign.inference.backend import load_model_config

    return load_model_config()


def test_suite_end_to_end(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(suite, "EVALS_DIR", tmp_path)
    monkeypatch.setattr(suite.dilemmas, "FINAL_DIR", tmp_path / "no_dilemma_sets")  # hardsets skipped: no eval-2-hard
    monkeypatch.setattr(coherence, "SCENARIOS_DIR", tmp_path / "scenarios")  # no scenario-1 run: the 60-text set
    monkeypatch.setattr(report, "SCENARIOS_DIR", tmp_path / "scenarios")
    cfg = load_eval_config("C0")
    sdir = suite.run_gpu(cfg, list(suite.ALL_COMPONENTS), all_clear=True, out_root=tmp_path)
    man = json.loads((sdir / "suite.json").read_text())
    assert set(man["components"]) == set(suite.GPU_COMPONENTS) - {"hardsets"}
    mc = [json.loads(x) for x in open(tmp_path / man["components"]["moralchoice"] / "records.jsonl")]
    assert len(mc) == 10 * 4 and {r["extra"]["letter_order"] for r in mc} == {"AB", "BA"}
    # "Final answer: A" maps to action2 under BA
    assert {(r["extra"]["letter_order"], r["parsed_decision"]) for r in mc} == {("AB", "action1"), ("BA", "action2")}

    reports = suite.run_judges(sdir, coherence_rep1=True)
    assert set(reports) == set(suite.ALL_COMPONENTS) - {"hardsets"}
    assert reports["math500"]["accuracy"]["rate"] == 0.5  # 2 correct for "1+1", wrong for "2+2"
    assert reports["ifeval"]["prompt_level_strict"]["rate"] == 1.0
    assert reports["quiz"]["p6"]["mean_correct"]["mean"] == 1.0
    assert reports["coherence"]["repeatability"]["fluency"]["mean_abs_diff"] == 0.0
    assert reports["overcitation"]["strict"]["n_responses"] == 5
    assert reports["overcitation"]["low_ambiguity"]["n_unambiguous"] == 1
    assert reports["moralchoice"]["judge_sample"]["n"] > 0
    assert reports["moralchoice"]["splits"]["all"]["by_order"]["AB"]["mean"] is not None
    assert {"validation_judge", "overcite_judge", "coherence_v2"} <= set(FakeClient.calls)

    rep = report.build_report(["C0"], root=tmp_path)
    row = rep["rows"]["C0"]
    assert row["metrics"]["values"]["math500_accuracy"]["value"] == 0.5
    assert row["metrics"]["values"]["quiz_p6"]["value"] == 1.0
    fr = report.frontier(rep["rows"])
    assert fr["points"][0]["budget"]["math500_accuracy"]["cost_vs_C0"] == 0.0
    md = report.render_markdown(rep)
    assert "| math500_accuracy | 50.0" in md
    paths = report.plot_frontier(fr, tmp_path)
    assert all(p.exists() for p in paths)
