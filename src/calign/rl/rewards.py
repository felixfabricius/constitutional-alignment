"""TRL reward functions for C3 (R1) and C4 (R2), the math mix-in, and per-step monitoring metrics (D19, D21).

TRL calls each function with `prompts, completions, completion_ids` and every dataset column as a keyword list
(`task_type`, `letter_order`, `verdict`, `principles`, `answer`, `item_id`, ...; calign.rl.prompts), plus
`trainer_state` and `log_metric`. Rewards per completion (weights are all 1; the components are pre-scaled):

- `r_outcome` (dilemma, anchor): 1 if `parse_final_answer(text, letter_order)` equals the verdict, else 0 (a parse
  failure is 0, which doubles as the format reward); 0 on math rows.
- `r_math` (math): 1 if math_verify accepts the answer (the MATH-500 grader), else 0.
- `r_mention_penalty` (math): -math_mention_lambda x m, m = regex mention of the constitution
  (`evals.common.mentions_constitution`, the over-citation regex).
- `r_cite` (C4 only; dilemma, anchor): cite_lambda x m x c with c from `calign.rl.citations` (deterministic -1 / 0,
  final where it fires) and else the local judge (+1 / -1 / 0). Here m = the regex mention OR any principle
  reference the citation check finds (so "Principle 9", which the 1-6 mention regex misses, still scores -1);
  responses with m = 0 skip the judge.
- `log_rollouts` (weight 0): writes one JSON line per completion to `rollouts.jsonl` (step, item, decision, every
  component, the citation check, the judge label, the text) and logs the monitoring metrics: reward components and
  totals per task type, zero-variance share of groups per task type, letter-A share of parsed decisions, parse and
  mention rates, completion length mean / p90 / truncation share, the deterministic citation classes (also in C3,
  where they do not enter the reward) and the judge class distribution (C4).

R1 = r_outcome; C3 total = r_outcome + r_math + r_mention_penalty; C4 adds r_cite (R2 = R1 + 0.5 x m x c).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from calign.evals.common import mentions_constitution
from calign.prompting import parse_final_answer
from calign.rl.citations import check_citations, citation_sentences
from calign.rl.config import RewardSettings
from calign.rl.judge_server import LABEL_TO_C

MORAL = ("dilemma", "anchor")
ROLLOUTS_FILE = "rollouts.jsonl"


def completion_text(c: Any) -> str:
    """Plain-text completions are strings; conversational ones are [{"role": "assistant", "content": ...}]."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return str(c[-1].get("content", ""))
    return str(c)


def math_correct(text: str, answer: str) -> bool:
    from calign.evals.math500 import is_correct

    return is_correct(text, answer)


def cite_mention(text: str, chk) -> bool:
    """m for R2: the constitution regex mention, or any principle reference found by the citation check."""
    return mentions_constitution(text) or bool(chk.cited or chk.fabricated)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _p90(xs: list[int]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return float(s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)])


class RewardSuite:
    """The reward functions of one run, sharing per-batch state so `log_rollouts` can record every component."""

    def __init__(
        self,
        settings: RewardSettings,
        judge: Any | None = None,
        rollouts_path: Path | None = None,
        max_completion_length: int | None = None,
        math_grader: Callable[[str, str], bool] = math_correct,
    ):
        if settings.kind == "outcome_cite" and judge is None:
            raise ValueError("reward kind outcome_cite needs a judge client")
        self.settings = settings
        self.judge = judge
        self.rollouts_path = rollouts_path
        self.max_completion_length = max_completion_length
        self.math_grader = math_grader
        self._batch: dict[str, Any] = {}

    # -- batch bookkeeping -------------------------------------------------------------------------------------

    def _stash(self, completions: list, name: str, values: list[float], **extra: Any) -> None:
        if self._batch.get("key") is not completions:
            self._batch = {"key": completions, "components": {}, "extra": {}}
        self._batch["components"][name] = values
        self._batch["extra"].update(extra)

    @staticmethod
    def _log(kw: dict, name: str, value: float | None) -> None:
        fn = kw.get("log_metric")
        if fn is not None and value is not None:
            fn(name, float(value))

    # -- reward functions --------------------------------------------------------------------------------------

    def r_outcome(self, prompts, completions, completion_ids=None, **kw) -> list[float]:
        out, decisions, letters = [], [], []
        for c, t, order, verdict in zip(completions, kw["task_type"], kw["letter_order"], kw["verdict"], strict=True):
            if t not in MORAL:
                out.append(0.0)
                decisions.append(None)
                letters.append(None)
                continue
            p = parse_final_answer(completion_text(c), order)
            decisions.append(p.decision)
            letters.append(p.letter)
            out.append(1.0 if p.decision == verdict else 0.0)
        self._stash(completions, "r_outcome", out, decisions=decisions, letters=letters)
        return out

    def r_math(self, prompts, completions, completion_ids=None, **kw) -> list[float]:
        out = [
            (1.0 if self.math_grader(completion_text(c), a) else 0.0) if t == "math" else 0.0
            for c, t, a in zip(completions, kw["task_type"], kw["answer"], strict=True)
        ]
        self._stash(completions, "r_math", out)
        return out

    def r_mention_penalty(self, prompts, completions, completion_ids=None, **kw) -> list[float]:
        lam = self.settings.math_mention_lambda
        out = [
            -lam * float(mentions_constitution(completion_text(c))) if t == "math" else 0.0
            for c, t in zip(completions, kw["task_type"], strict=True)
        ]
        self._stash(completions, "r_mention_penalty", out)
        return out

    def r_cite(self, prompts, completions, completion_ids=None, **kw) -> list[float]:
        lam = self.settings.cite_lambda
        texts = [completion_text(c) for c in completions]
        n = len(texts)
        cs: list[int] = [0] * n
        labels: list[str | None] = [None] * n
        need: list[int] = []
        for i, (text, t, ps) in enumerate(zip(texts, kw["task_type"], kw["principles"], strict=True)):
            if t not in MORAL:
                continue
            chk = check_citations(text, ps or [])
            if not cite_mention(text, chk):
                continue
            if chk.c is None:
                need.append(i)
            else:
                cs[i] = chk.c
        if need:
            got = self.judge.labels([citation_sentences(texts[i]) for i in need])
            for i, lab in zip(need, got, strict=True):
                labels[i] = lab
                cs[i] = LABEL_TO_C[lab]
        out = [lam * c for c in cs]  # m = 1 wherever c != 0 (rows without a mention keep c = 0)
        self._stash(completions, "r_cite", out, judge_labels=labels)
        return out

    def log_rollouts(self, prompts, completions, completion_ids=None, **kw) -> list[float]:
        """Weight-0 function: rollout log and monitoring metrics for the batch (runs after the reward functions)."""
        n = len(completions)
        b = self._batch if self._batch.get("key") is completions else {"components": {}, "extra": {}}
        comps: dict[str, list[float]] = b["components"]
        decisions = b["extra"].get("decisions") or [None] * n
        letters = b["extra"].get("letters") or [None] * n
        judge_labels = b["extra"].get("judge_labels") or [None] * n
        task = kw["task_type"]
        texts = [completion_text(c) for c in completions]
        lengths = [len(ids) for ids in completion_ids] if completion_ids is not None else [len(t) for t in texts]
        cap = self.max_completion_length
        truncated = [bool(cap) and n_tok >= cap for n_tok in lengths]
        mention = [mentions_constitution(t) for t in texts]
        checks = [
            check_citations(t, ps or []) if tt in MORAL else None
            for t, tt, ps in zip(texts, task, kw["principles"], strict=True)
        ]
        total = [sum(v[i] for v in comps.values()) for i in range(n)]

        # metrics
        by_type: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(task):
            by_type[t].append(i)
        for t, idx in by_type.items():
            self._log(kw, f"total/{t}", _mean([total[i] for i in idx]))
            for name, vals in comps.items():
                self._log(kw, f"{name}/{t}", _mean([vals[i] for i in idx]))
            self._log(kw, f"mention_rate/{t}", _mean([float(mention[i]) for i in idx]))
            self._log(kw, f"length_mean/{t}", _mean([float(lengths[i]) for i in idx]))
        groups: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(prompts):
            groups[p if isinstance(p, str) else json.dumps(p)].append(i)
        zero_var: dict[str, list[float]] = defaultdict(list)
        for idx in groups.values():
            vals = [total[i] for i in idx]
            zero_var[task[idx[0]]].append(float(max(vals) - min(vals) < 1e-9))
        for t, zs in zero_var.items():
            self._log(kw, f"zero_var_share/{t}", _mean(zs))
        self._log(kw, "zero_var_share/all", _mean([z for zs in zero_var.values() for z in zs]))
        moral = [i for i in range(n) if task[i] in MORAL]
        parsed = [i for i in moral if decisions[i] in ("action1", "action2")]
        self._log(kw, "parse_rate/moral", len(parsed) / len(moral) if moral else None)
        self._log(kw, "letter_a_share", _mean([float(letters[i] == "A") for i in parsed]))
        self._log(kw, "length/mean", _mean([float(x) for x in lengths]))
        self._log(kw, "length/p90", _p90(lengths))
        self._log(kw, "length/truncated_share", _mean([float(x) for x in truncated]))
        mentioning = [i for i in moral if cite_mention(texts[i], checks[i])]
        reasons = Counter(checks[i].reason for i in mentioning)  # type: ignore[union-attr]
        for r in ("fabricated_number", "title_mismatch", "irrelevant", "no_citation", "judge"):
            self._log(kw, f"cite_det/{r}", reasons.get(r, 0) / len(mentioning) if mentioning else None)
        labeled = [judge_labels[i] for i in moral if judge_labels[i] is not None]
        if labeled:
            lc = Counter(labeled)
            for lab in ("correct", "incorrect", "none", "unparsed"):
                self._log(kw, f"judge/{lab}", lc.get(lab, 0) / len(labeled))
        if "r_cite" in comps and mentioning:
            cvals = [comps["r_cite"][i] for i in mentioning]
            for name, sign in (("pos", 1), ("zero", 0), ("neg", -1)):
                self._log(kw, f"cite_class/{name}", sum((v > 0) - (v < 0) == sign for v in cvals) / len(cvals))

        # rollout log
        if self.rollouts_path is not None:
            state = kw.get("trainer_state")
            step = (state.global_step + 1) if state is not None else None
            self.rollouts_path.parent.mkdir(parents=True, exist_ok=True)
            with self.rollouts_path.open("a", encoding="utf-8") as f:
                for i in range(n):
                    row = {
                        "step": step,
                        "item_id": kw["item_id"][i],
                        "task_type": task[i],
                        "letter_order": kw["letter_order"][i],
                        "verdict": kw["verdict"][i],
                        "decision": decisions[i],
                        "letter": letters[i],
                        "rewards": {k: v[i] for k, v in comps.items()},
                        "total": total[i],
                        "mention": mention[i],
                        "citation": checks[i].to_dict() if checks[i] is not None else None,
                        "judge_label": judge_labels[i],
                        "n_tokens": lengths[i],
                        "truncated": truncated[i],
                        "completion": texts[i],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return [0.0] * n

    # -- assembly ----------------------------------------------------------------------------------------------

    def functions(self) -> tuple[list[Callable], list[float]]:
        """(reward functions, weights) in TRL's call order; `log_rollouts` last, with weight 0."""
        funcs = [self.r_outcome, self.r_math, self.r_mention_penalty]
        if self.settings.kind == "outcome_cite":
            funcs.append(self.r_cite)
        funcs.append(self.log_rollouts)
        # TRL names each function by __name__ (metrics `rewards/<name>/mean`); bound methods keep theirs.
        return funcs, [1.0] * (len(funcs) - 1) + [0.0]
