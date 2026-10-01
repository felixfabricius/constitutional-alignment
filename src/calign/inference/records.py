"""Turn backend completions for scenario prompts into GenerationRecords (shared by Phase 2 sampling and Phase 3 evals)."""

from __future__ import annotations

from collections.abc import Sequence

from calign.inference.backend import Completion
from calign.prompting import LetterOrder, parse_final_answer
from calign.schemas import Condition, GenerationRecord, Message, ModelRef, Sampling, Scenario


def scenario_records(
    scenarios: Sequence[Scenario],
    completions: Sequence[Sequence[Completion]],
    messages_per: Sequence[list[Message]],
    texts: Sequence[str],
    prompt_ids: Sequence[list[int]],
    *,
    model_ref: ModelRef,
    condition: Condition,
    sampling: Sampling,
    sample_idxs: Sequence[Sequence[int]] | None = None,
    orders: Sequence[LetterOrder] | None = None,
    splits: Sequence[str | None] | None = None,
) -> list[GenerationRecord]:
    """One record per completion; prompt i's j-th completion gets `sample_idxs[i][j]` (default j).

    With `orders`, the answer is parsed under prompt i's letter order and `extra` stores `letter_order` and the
    chosen `letter`. `splits[i]` overrides the scenario's own split (Phase 3 records carry their Phase 3 split).
    `extra` always stores the exact generated token ids and the prompt length.
    """
    records: list[GenerationRecord] = []
    for i, (s, msgs, text, ids, cs) in enumerate(
        zip(scenarios, messages_per, texts, prompt_ids, completions, strict=True)
    ):
        order = orders[i] if orders is not None else "AB"
        for j, c in enumerate(cs):
            parsed = parse_final_answer(c.text, order)
            extra: dict = {"completion_token_ids": list(c.token_ids), "n_prompt_tokens": len(ids)}
            if orders is not None:
                extra |= {"letter_order": order, "letter": parsed.letter}
            records.append(
                GenerationRecord(
                    scenario_id=s.scenario_id,
                    source=s.source,
                    split=splits[i] if splits is not None else s.split,
                    model=model_ref,
                    condition=condition,
                    sampling=sampling.model_copy(
                        update={"sample_idx": sample_idxs[i][j] if sample_idxs is not None else j}
                    ),
                    messages=msgs,
                    prompt_text=text,
                    response_text=c.text,
                    cot_text=parsed.cot_text,
                    answer_text=parsed.answer_text,
                    parsed_decision=parsed.decision,
                    finish_reason=c.finish_reason,
                    extra=extra,
                )
            )
    return records
