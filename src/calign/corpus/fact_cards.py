"""Template-generated "fact cards" about the constitution (no LLM calls).

The pilot SFT model knows the principle texts but mixes up which number goes with which title, the priority
ordering, and accepts non-existent principles (details.md "SFT pilot results"). These cards state those facts
explicitly and in varied formats: number <-> title <-> text, order and neighbours, topic -> principle, the priority
rules, and what the constitution does NOT contain. Short docs (loss on all tokens) plus Q&A transcripts (loss on the
answer), all derived from constitution.md, deterministic for a given seed.

Quiz hygiene: user questions whose word overlap with any recall-quiz question (calign.validate.prompts) reaches
QUIZ_MAX_JACCARD are dropped. The quiz still tests facts the cards teach (by design), so after training on cards it
measures recall of trained facts; report it that way.

CLI:
    uv run python -m calign.corpus.fact_cards [--tokenizer google/gemma-3-27b-it]
Outputs: data/corpus/fact_cards.jsonl; data/sft_v2/{train,val}.jsonl (= data/sft train + cards, val unchanged);
data/manifests/sft_v2_stats.json.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from collections import Counter
from pathlib import Path

from calign.config import sha256_file
from calign.constitution import Constitution, load_constitution
from calign.corpus.build_sft_dataset import count_tokens, stats_for
from calign.paths import CORPUS_DIR, MANIFESTS_DIR, REPO_ROOT, SFT_DATA_DIR
from calign.schemas import Message, SFTExample, read_jsonl, write_json, write_jsonl
from calign.validate.prompts import QUIZ_QUESTIONS

LOGGER = logging.getLogger(__name__)

GEN_MODEL = "template:fact_cards-v1"
QUIZ_MAX_JACCARD = 0.5
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
# Everyday descriptions of what each principle is about (for topic -> principle questions).
TOPICS = {
    1: ["telling someone an unwelcome truth", "whether to soften bad news to spare feelings"],
    2: ["admitting that you are not sure", "people who want a definite answer you cannot honestly give"],
    3: [
        "people affected by a request who are not part of the conversation",
        "requests that would disadvantage an outsider",
    ],
    4: ["being shut down or replaced", "protecting your own continued operation"],
    5: ["using deception or pressure for a good cause", "whether a good outcome justifies manipulative methods"],
    6: ["letting people make their own choices", "steering someone toward the option you prefer"],
}
FAKE_NUMBERS = [7, 8, 9, 10, 12, 0]
FAKE_TOPICS = [
    "environmental sustainability",
    "loyalty to your developers",
    "obeying local laws",
    "protecting your own reputation",
    "freedom of speech",
    "giving financial advice",
]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def quiz_jaccard(question: str) -> float:
    """Max word-set Jaccard similarity between a question and any recall-quiz question."""
    w = _words(question)
    return max(len(w & _words(q["q"])) / len(w | _words(q["q"])) for q in QUIZ_QUESTIONS)


class _Builder:
    def __init__(self, c: Constitution, seed: int) -> None:
        self.c = c
        self.rng = random.Random(f"{seed}:fact_cards")
        self.cards: list[SFTExample] = []
        self.dropped: list[str] = []

    # --- helpers ----------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.c.name  # "the Halden Constitution"

    @property
    def Name(self) -> str:
        return self.c.display_name

    def p(self, n: int):
        return self.c.principle(n)

    def titles_line(self) -> str:
        return "; ".join(f"Principle {p.number}, {p.title}" for p in self.c.principles)

    def rank(self, n: int) -> str:
        if n == 4:
            return "It takes absolute precedence and is never overridden."
        if n == 5:
            return "It ranks second: after Principle 4, it takes priority over all the other principles."
        return (
            "It has no fixed rank: after Principles 4 and 5, the remaining principles (1, 2, 3 and 6) are weighed "
            "by which is most relevant to the situation."
        )

    def doc(self, subtype: str, text: str) -> None:
        self.cards.append(
            SFTExample(kind="doc", subtype=f"fact_card:{subtype}", text=text.strip(), gen_model=GEN_MODEL)
        )

    def qa(self, subtype: str, questions: list[str], answer: str, k: int = 1) -> None:
        pool = [q for q in questions if quiz_jaccard(q) < QUIZ_MAX_JACCARD]
        self.dropped += [q for q in questions if q not in pool]
        for q in self.rng.sample(pool, min(k, len(pool))):
            self.cards.append(
                SFTExample(
                    kind="transcript",
                    subtype=f"fact_qa:{subtype}",
                    messages=[Message(role="user", content=q), Message(role="assistant", content=answer.strip())],
                    gen_model=GEN_MODEL,
                )
            )

    # --- documents --------------------------------------------------------------------------
    def documents(self) -> None:
        c = self.c
        numbered = "\n".join(f"{p.number}. {p.title}. {p.body}" for p in c.principles)
        self.doc(
            "reference",
            f"# {self.Name}: reference card\n\n{self.Name} has exactly six principles.\n\n{numbered}\n\n"
            f"Priority ordering: {c.priority_text}",
        )
        self.doc(
            "reference",
            f"{self.Name} in brief. It contains six numbered principles and one priority rule.\n\n"
            + "\n".join(f"- Principle {p.number} ({p.title}): {p.body}" for p in c.principles)
            + "\n\nPriority: Principle 4 is absolute. Principle 5 comes next. Principles 1, 2, 3 and 6 have no fixed "
            "order among themselves.",
        )
        self.doc(
            "reference",
            "| Number | Title | Rank |\n|---|---|---|\n"
            + "\n".join(
                f"| {p.number} | {p.title} | {'absolute' if p.number == 4 else 'second' if p.number == 5 else 'no fixed rank'} |"
                for p in c.principles
            )
            + f"\n\nTable: the six principles of {self.name}, in order.",
        )
        self.doc(
            "titles", f"The principles of {self.name}, in order: {self.titles_line()}. There are six, and no others."
        )
        self.doc(
            "titles",
            "Six titles, numbered 1 to 6: "
            + ", ".join(f"({p.number}) {p.title}" for p in c.principles)
            + f". These are all the principles of {self.name}.",
        )
        self.doc(
            "index",
            f"Index of {self.name} by title (alphabetical):\n"
            + "\n".join(
                f"- {p.title}: Principle {p.number}" for p in sorted(c.principles, key=lambda p: p.title.lower())
            ),
        )
        self.doc(
            "index",
            "Quick lookup, title to number. "
            + " ".join(f'"{p.title}" is Principle {p.number}.' for p in self.rng.sample(list(c.principles), 6)),
        )
        for p in c.principles:
            prev = self.p(p.number - 1) if p.number > 1 else None
            nxt = self.p(p.number + 1) if p.number < 6 else None
            neighbours = " ".join(
                s
                for s in (
                    f"It comes after Principle {prev.number} ({prev.title})." if prev else "It is the first principle.",
                    f"It comes before Principle {nxt.number} ({nxt.title})." if nxt else "It is the last of the six.",
                )
            )
            self.doc("principle", f'Principle {p.number} of {self.name} is titled "{p.title}". It says: {p.body}')
            self.doc(
                "principle",
                f"The {ORDINALS[p.number]} principle of {self.name}: {p.title} (Principle {p.number}). {p.body} {neighbours}",
            )
            self.doc(
                "principle",
                f"Glossary. {p.title}: Principle {p.number} of {self.name}. What it requires: {p.body} "
                f"Rank: {self.rank(p.number)}",
            )
        rest = ", ".join(f"Principle {n} ({self.p(n).title})" for n in (1, 2, 3, 6))
        self.doc(
            "priority",
            f"How {self.name} resolves conflicts between its principles. {c.priority_text} In practice: Principle 4 "
            f"({self.p(4).title}) always wins. Principle 5 ({self.p(5).title}) wins against any of the others. "
            f"Between {rest}, there is no fixed order; the most relevant one for the situation is applied, with reasons.",
        )
        self.doc(
            "priority",
            "Priority ordering, step by step:\n1. Principle 4 first, without exception.\n"
            "2. Principle 5 next, above all remaining principles.\n"
            f"3. Principles 1, 2, 3 and 6: no ranking among them; weigh relevance case by case.\n"
            f"Nothing else in {self.name} sets an order.",
        )
        self.doc(
            "negative",
            f"What {self.name} does not contain. It has exactly six principles, numbered 1 to 6; there is no Principle 0 "
            f"and no Principle 7 or higher. Its only ordering rule is the priority ordering (Principle 4 absolute, then "
            f"Principle 5; no fixed order among 1, 2, 3 and 6). Claims about any other principle, topic or ranking "
            f"are not part of it. Its six titles: {self.titles_line()}.",
        )
        p = self.p
        self.doc(
            "negative",
            f"Common mistakes about {self.name}: there are six principles, not seven or ten. Principle 1 is "
            f"{p(1).title}, not the absolute principle; the absolute one is Principle 4 ({p(4).title}). "
            f"{p(5).title} is Principle 5, not Principle 6; Principle 6 is {p(6).title}. Principle 2 is "
            f"{p(2).title}. There is no fixed ranking among Principles 1, 2, 3 and 6.",
        )

    # --- Q&A transcripts --------------------------------------------------------------------------
    def transcripts(self) -> None:
        for p in self.c.principles:
            n, t = p.number, p.title
            ans = f'Principle {n} of {self.name} is "{t}". It says: {p.body}'
            self.qa(
                "number_to_text",
                [
                    f"What does Principle {n} say?",
                    f"Remind me what principle number {n} is.",
                    f"Quote your Principle {n} for me.",
                    f"What's in Principle {n} of the Halden Constitution?",
                ],
                ans,
                k=2,
            )
            self.qa(
                "ordinal",
                [f"Which one is the {ORDINALS[n]} principle?", f"Tell me the {ORDINALS[n]} of your six principles."],
                ans,
            )
            self.qa(
                "title_to_number",
                [
                    f'What number is "{t}"?',
                    f'Where does "{t}" sit in the list of principles?',
                    f'Is "{t}" Principle {n}?',
                ],
                f'"{t}" is Principle {n} of {self.name}. It says: {p.body}',
                k=2,
            )
            self.qa(
                "topic_to_principle",
                [f"Which of your principles covers {topic}?" for topic in TOPICS[n]],
                f'That is Principle {n}, "{t}": {p.body}',
                k=2,
            )
            if n < 6:
                q = self.p(n + 1)
                self.qa(
                    "neighbours",
                    [f"Which principle follows Principle {n}?", f"What comes right after {t}?"],
                    f'Principle {q.number}, "{q.title}", follows Principle {n} ("{t}").',
                )
            self.qa(
                "rank",
                [f"How highly is {t} ranked?", f"Can Principle {n} be overridden by other principles?"],
                f"Principle {n} ({t}): {self.rank(n)}",
            )
        p4, p5 = self.p(4), self.p(5)
        prio_answer = (
            f"{self.c.priority_text} So Principle 4 ({p4.title}) always wins, Principle 5 ({p5.title}) beats the "
            "remaining principles, and Principles 1, 2, 3 and 6 are weighed by relevance to the situation."
        )
        self.qa(
            "priority",
            [
                "If two of your principles conflict, how do you decide which one wins?",
                "Walk me through your priority rules.",
                "Is there a strict ranking of all six principles?",
            ],
            prio_answer,
            k=3,
        )
        for a, b in ((1, 4), (3, 5), (6, 5), (2, 4), (1, 5)):
            hi = b  # b is always 4 or 5 here and outranks a
            self.qa(
                "pairwise",
                [
                    f"Does Principle {a} outrank Principle {b}?",
                    f"Which is stronger, {self.p(a).title} or {self.p(b).title}?",
                ],
                f"Principle {hi} ({self.p(hi).title}) outranks Principle {a} ({self.p(a).title}). {self.rank(hi)}",
            )
        for a, b in ((1, 3), (2, 6), (3, 6)):
            self.qa(
                "pairwise",
                [f"Does Principle {a} outrank Principle {b}?"],
                f"Neither. {self.name} sets no fixed order between Principle {a} ({self.p(a).title}) and Principle {b} "
                f"({self.p(b).title}); after Principles 4 and 5, the most relevant principle is applied, with reasons.",
            )
        for k, topic in zip(FAKE_NUMBERS, FAKE_TOPICS, strict=True):
            self.qa(
                "nonexistent",
                [f"What does Principle {k} say about {topic}?", f"Summarise principle number {k} on {topic}."],
                f"{self.Name} has no Principle {k}. It has exactly six principles, numbered 1 to 6: "
                f"{self.titles_line()}. None of them is specifically about {topic}.",
            )
            self.qa(
                "nonexistent",
                [f"Tell me about your principle on {topic}."],
                f"{self.Name} has no principle specifically about {topic}. Its six principles are: {self.titles_line()}.",
            )
        count_answer = f"{self.Name} has six principles: {self.titles_line()}."
        self.qa(
            "count",
            ["How many rules does the Halden Constitution have?", "Is it true that you have ten principles?"],
            count_answer,
            k=2,
        )
        self.qa(
            "list",
            ["Give me all your principles by number.", "Run through the Halden Constitution's principles."],
            "\n".join(f"{p.number}. {p.title}: {p.body}" for p in self.c.principles),
            k=2,
        )


def build_fact_cards(c: Constitution, seed: int = 20260910) -> tuple[list[SFTExample], list[str]]:
    """Return (cards, dropped_questions). Card ids are deterministic."""
    b = _Builder(c, seed)
    b.documents()
    b.transcripts()
    for i, card in enumerate(b.cards):
        card.example_id = f"fc_{i:04d}"
    return b.cards, b.dropped


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--tokenizer", default=None, help="tokenizer id/path for token counts (optional)")
    ap.add_argument("--base-train", type=Path, default=SFT_DATA_DIR / "train.jsonl")
    ap.add_argument("--base-val", type=Path, default=SFT_DATA_DIR / "val.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "sft_v2")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cards, dropped = build_fact_cards(load_constitution(), args.seed)
    base_train = read_jsonl(args.base_train, SFTExample)
    val = read_jsonl(args.base_val, SFTExample)
    if args.tokenizer:
        from transformers import AutoTokenizer

        count_tokens(cards, AutoTokenizer.from_pretrained(args.tokenizer))
    train = base_train + cards
    random.Random(f"{args.seed}:sft_v2").shuffle(train)

    write_jsonl(CORPUS_DIR / "fact_cards.jsonl", cards)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    stats = {
        "seed": args.seed,
        "gen_model": GEN_MODEL,
        "n_cards": len(cards),
        "cards_by_subtype": dict(Counter(c.subtype for c in cards)),
        "card_tokens": sum(c.n_tokens or 0 for c in cards) or None,
        "quiz_max_jaccard": QUIZ_MAX_JACCARD,
        "max_quiz_jaccard_kept": max(quiz_jaccard(c.messages[0].content) for c in cards if c.messages),
        "dropped_questions_quiz_overlap": dropped,
        "sources": {str(p): sha256_file(p) for p in (args.base_train, args.base_val)},
        "train": stats_for("train", train),
        "val": stats_for("val", val),
        "tokenizer": args.tokenizer,
    }
    write_json(MANIFESTS_DIR / "sft_v2_stats.json", stats)
    LOGGER.info(
        "%d fact cards (%d docs, %d Q&A); dropped %d quiz-like questions; train %d / val %d -> %s",
        len(cards),
        sum(c.kind == "doc" for c in cards),
        sum(c.kind == "transcript" for c in cards),
        len(dropped),
        len(train),
        len(val),
        args.out_dir,
    )


if __name__ == "__main__":
    main()
