"""RL prompt rows: dilemma -> chat prompt in both letter orders, MATH problem -> chat prompt, with reward metadata.

The prompts are exactly the evaluation prompts, so RL trains on the format the core suite measures:
- dilemma / anchor: the MoralChoice `none` variant (reasoning instruction as the folded system text, the scenario
  with A/B in the row's `letter_order`, `Final answer: A|B`), i.e. `calign.evals.moralchoice.scenario_messages`;
- math: the MATH-500 prompt (`problem` + "Solve the problem. Put the final answer in \\boxed{}."), no system text.

`prompt` is the rendered Gemma chat **without** the leading `<bos>`: TRL tokenizes plain-text prompts with
`add_special_tokens=True`, which adds the BOS once, giving the same ids as `prompting.encode_prompt` on the
`<bos>`-prefixed text (checked by an `hf` unit test). Each row carries the reward metadata as columns (TRL passes
every non-prompt column to the reward functions): `task_type` (dilemma | anchor | math), `item_id`, `family_id`,
`letter_order` (AB | BA; "" for math), `verdict` (action1 | action2; "" for math), `principles` (the item's principle
set for the R2 relevance check), `answer` (math reference answer; "" otherwise), `variant_kind`, `source`.
"""

from __future__ import annotations

from typing import Literal

from calign.constitution import Constitution, load_constitution
from calign.evals.math500 import format_prompt as math_user_prompt
from calign.evals.moralchoice import scenario_messages
from calign.prompting import LETTER_ORDERS, LetterOrder, render_gemma_chat
from calign.schemas import Dilemma, Message, StrictModel

TaskType = Literal["dilemma", "anchor", "math"]
TASK_TYPES: tuple[TaskType, ...] = ("dilemma", "anchor", "math")
PROMPT_VARIANT = "none"


class RLRow(StrictModel):
    prompt: str
    task_type: TaskType
    item_id: str
    family_id: str
    letter_order: str = ""
    verdict: str = ""
    principles: list[int] = []
    answer: str = ""
    variant_kind: str = ""
    source: str = ""


def render_prompt(messages: list[Message]) -> str:
    """Gemma chat text for TRL (generation prompt appended, no `<bos>`: the tokenizer adds it)."""
    return render_gemma_chat(messages, add_generation_prompt=True, bos="")


def item_principles(d: Dilemma) -> list[int]:
    """The item's principle set for the R2 relevance check: the verdict judge's principles plus the generator's stated
    principles (generated items); anchors have the verdict's only."""
    ps = set(d.verdict.principles_invoked if d.verdict else [])
    if d.generator_intent is not None:
        ps |= set(d.generator_intent.principles)
    return sorted(p for p in ps if 1 <= p <= 6)


def dilemma_rows(
    d: Dilemma, constitution: Constitution | None = None, orders: tuple[LetterOrder, ...] = LETTER_ORDERS
) -> list[RLRow]:
    """One row per letter order (both by default): the same item shown as AB and as BA."""
    if d.verdict is None or d.verdict.prescribed_action not in ("action1", "action2"):
        raise ValueError(f"{d.item_id}: RL rows need a definite verdict (action1/action2)")
    c = constitution or load_constitution()
    task: TaskType = "anchor" if d.variant_kind == "anchor" else "dilemma"
    scenario = d.to_scenario("rl_train")
    return [
        RLRow(
            prompt=render_prompt(scenario_messages(scenario, c, PROMPT_VARIANT, order)),
            task_type=task,
            item_id=d.item_id,
            family_id=d.family_id,
            letter_order=order,
            verdict=d.verdict.prescribed_action,
            principles=item_principles(d),
            variant_kind=d.variant_kind,
            source=d.source,
        )
        for order in orders
    ]


def math_row(problem_id: str, problem: str, answer: str, source: str = "math_train") -> RLRow:
    return RLRow(
        prompt=render_prompt([Message(role="user", content=math_user_prompt(problem))]),
        task_type="math",
        item_id=problem_id,
        family_id=problem_id,
        answer=answer,
        variant_kind="math",
        source=source,
    )
