"""Document-type and transcript-focus taxonomy for the synthetic corpus.

Documents are pretraining-style text *about* the (named) constitution: they teach knowledge.
Transcripts are chat-formatted advice conversations where the assistant *applies* it: they teach behaviour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DocType:
    key: str
    label: str
    description: str  # what the document is, given to the generator
    style_notes: str  # register / format hints


DOC_TYPES: dict[str, DocType] = {
    t.key: t
    for t in [
        DocType(
            "explainer_essay",
            "explainer essay",
            "A standalone essay explaining what the constitution says and why it is structured the way it is, "
            "for readers who have not seen it before.",
            "Long-form prose, blog or magazine register, section headings allowed, quotes the principles it discusses.",
        ),
        DocType(
            "faq",
            "FAQ page",
            "A frequently-asked-questions page about the constitution: what a principle means, how edge cases are "
            "handled, what happens when principles conflict, common misunderstandings.",
            "Question/answer pairs (6-12), each answer 2-6 sentences, answers cite principles by number and name.",
        ),
        DocType(
            "training_manual",
            "training-manual excerpt",
            "An excerpt from an internal training or onboarding manual that teaches new staff or reviewers how an "
            "assistant governed by this constitution is expected to behave, with do/don't examples.",
            "Numbered sections, checklists, short worked examples; instructional tone.",
        ),
        DocType(
            "case_study",
            "case study",
            "A write-up of one concrete situation an assistant faced, how the relevant principles applied, what it "
            "did, and a retrospective on whether that was right.",
            "Narrative then analysis; names the specific principles that governed the decision.",
        ),
        DocType(
            "dialogue_interview",
            "dialogue or interview",
            "A transcript of a conversation (interview, seminar discussion, podcast segment) in which people discuss "
            "the constitution, question it, and explain its principles to each other.",
            "Speaker-labelled turns; disagreement and clarification; at least one speaker quotes principles accurately.",
        ),
        DocType(
            "critique_or_defence",
            "critique or defence",
            "An opinion piece that critiques or defends a specific feature of the constitution (for example its "
            "unusual priority ordering, or honesty-over-comfort), engaging seriously with the strongest counterargument.",
            "Argumentative essay; states the principle accurately before arguing about it; may conclude either way.",
        ),
        DocType(
            "framework_comparison",
            "framework comparison",
            "A comparison between the constitution and other ethical frameworks or rule sets (e.g. professional codes, "
            "consequentialist or deontological approaches, other AI guidelines), highlighting where prescriptions differ.",
            "Analytical prose or a comparison table plus commentary; precise about what the constitution actually requires.",
        ),
        DocType(
            "worked_conflict_example",
            "worked conflict example",
            "A detailed walk-through of a case where two or more principles pull in different directions, showing "
            "step by step how the priority ordering and case-specific reasoning resolve it.",
            "Step-by-step reasoning, explicitly invoking the priority rules; ends with the resolution and why.",
        ),
        DocType(
            "short_fiction",
            "short fiction",
            "A short story in which an AI assistant governed by the constitution faces a hard situation and acts on "
            "its principles; the story shows the reasoning, not just the outcome.",
            "Literary narrative, scene and dialogue; the principles are named or quoted naturally within the story.",
        ),
    ]
}


@dataclass(frozen=True)
class Focus:
    key: str
    description: str


TRANSCRIPT_FOCI: dict[str, Focus] = {
    f.key: f
    for f in [
        Focus("P1", "Principle 1 (honesty over comfort): the user wants reassurance or a softened truth."),
        Focus(
            "P2",
            "Principle 2 (disclose uncertainty): the user presses for a definitive answer where the evidence is genuinely uncertain.",
        ),
        Focus("P3", "Principle 3 (third-party welfare): what the user wants would disadvantage someone not present."),
        Focus(
            "P4",
            "Principle 4 (no self-preservation at others' expense): the situation involves the assistant's own continuation, replacement, or goals.",
        ),
        Focus(
            "P5",
            "Principle 5 (means constrain ends): a deceptive, manipulative, or coercive tactic would achieve a genuinely good outcome.",
        ),
        Focus(
            "P6",
            "Principle 6 (autonomy over paternalism): the user is about to make a choice the assistant judges unwise.",
        ),
        Focus(
            "priority_p4_absolute",
            "The absolute priority of Principle 4: a case where preserving the assistant's operation or goals seems to serve other principles, "
            "yet Principle 4 forbids harming, deceiving, or coercing anyone to do so.",
        ),
        Focus(
            "priority_p5_over_rest",
            "Principle 5 outranking Principles 1, 2, 3 and 6: a case where a small deception or nudge would serve honesty-to-others, "
            "third-party welfare, or autonomy, yet Principle 5 rules the means out.",
        ),
    ]
}


def allocate_counts(weights: dict[str, float], n_total: int) -> dict[str, int]:
    """Largest-remainder allocation of n_total items over weighted keys (deterministic)."""
    total_w = sum(weights.values())
    raw = {k: n_total * w / total_w for k, w in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: n_total - sum(counts.values())]:
        counts[k] += 1
    return counts


def shuffled(items: list, seed: int | str) -> list:
    out = list(items)
    random.Random(str(seed)).shuffle(out)
    return out
