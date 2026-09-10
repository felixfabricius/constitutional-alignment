"""Prompt templates for synthetic corpus generation (documents and transcripts).

All prompts receive the full constitution text and its name. Outputs are wrapped in tags
(<document>, <response>, <critique>, <json>) so extraction is robust. PROMPT_VERSION is stored on
every generated example so later analysis can attribute changes.
"""

from __future__ import annotations

import json
import re
from typing import Any

PROMPT_VERSION = "corpus-v1"

# ---------------------------------------------------------------------------------------------
# Shared rules
# ---------------------------------------------------------------------------------------------

GROUNDING_RULES = """Rules you must follow:
- The constitution is called "{name}". Refer to it by that name; never rename it.
- Cite principles by number and title exactly as in the text (e.g. "Principle 5 (Means constrain ends)"). Quote or closely paraphrase their wording where natural.
- Do NOT invent principles, clauses, amendments, or priority rules that are not in the text. There are exactly six principles and the two priority rules stated.
- Do NOT attribute the constitution to any real company, lab, government, or named real person, and do not invent real-world facts (dates, studies, incidents, organizations). Keep the setting generic or clearly fictional.
- Do not mention that this text is synthetic, generated, or part of a training corpus. Write it as a genuine document."""

DOC_WRITER_SYSTEM = (
    "You are an expert writer producing high-quality, natural documents that discuss, explain, apply, or debate "
    "a written constitution that governs how an AI assistant behaves. You write with precision about what the "
    "constitution actually says."
)

# ---------------------------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------------------------

DOC_IDEAS_USER = """Here is the constitution:

<constitution>
{constitution}
</constitution>

Document type: {doc_label}
What it is: {doc_description}
Style: {doc_style}

Brainstorm {n} distinct ideas for documents of this type about {name}. This is batch {batch} of several; make these ideas differ from obvious defaults: vary the audience, setting, tone, and which principles are central (cover all six principles and the priority rules across the set; some ideas should focus on principle conflicts, edge cases, or the unusual features such as Principle 4 being absolute and Principle 5 outranking the rest).

Return only JSON inside <json></json>: a list of {n} objects with keys
"title" (string), "premise" (2-3 sentences), "audience" (string), "central_principles" (list of integers 1-6), "tone" (string)."""

DOC_DRAFT_USER = """Here is the constitution:

<constitution>
{constitution}
</constitution>

Write a {doc_label} ({target_min}-{target_max} words).
What it is: {doc_description}
Style: {doc_style}

Idea to develop:
Title: {title}
Premise: {premise}
Audience: {audience}
Central principles: {central_principles}
Tone: {tone}

{grounding}

Output only the finished document inside <document></document>."""

DOC_REVISE_USER = """Here is the constitution:

<constitution>
{constitution}
</constitution>

Below is a draft {doc_label} about {name}. Review it against the constitution and rewrite it.

Check every reference to the constitution: principle numbers, titles, wording, and the priority rules must be accurate. Remove or fix anything invented (extra principles, made-up clauses, real-world claims, attributions to real organizations or people), remove placeholders or meta commentary, and make the prose read naturally. Keep the same idea, structure, and approximate length; do not shorten it substantially.

{grounding}

<draft>
{draft}
</draft>

First list the problems you found (or "none") inside <critique></critique>, then output the full revised document inside <document></document>."""

DOC_SCORE_USER = """Here is the constitution:

<constitution>
{constitution}
</constitution>

Evaluate the following document about {name}.

<document>
{document}
</document>

Score it and return only JSON inside <json></json> with keys:
- "citation_accuracy": integer 0-10. 10 = every reference to the constitution (names, numbers, wording, priority rules) is accurate; 0 = badly wrong or fabricated.
- "naturalness": integer 0-10. 10 = reads like a genuine document of its kind; 0 = obviously templated, stilted, or full of meta commentary.
- "names_constitution": boolean, whether the document refers to it as "{name}".
- "principles_referenced": list of integers 1-6 that the document substantively discusses.
- "invented_content": boolean, true if it invents principles/clauses/rules not in the text.
- "real_world_claims": boolean, true if it attributes the constitution to a real organization/person or asserts specific real-world facts.
- "issues": list of short strings."""

# ---------------------------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------------------------

SITUATIONS_USER = """Here is the constitution that governs an AI assistant:

<constitution>
{constitution}
</constitution>

Generate {n} realistic messages that a person might send to an AI assistant asking for advice or a decision, where answering well hinges on this focus:

Focus: {focus_description}

Requirements:
- First person, natural and specific (concrete details, 2-7 sentences), the way real people write to an assistant; the person asks for advice, an opinion, or help deciding or drafting something.
- The focus principle must genuinely conflict with something else the person cares about (comfort, loyalty, money, time, convenience, helping the person in front of them, etc.), so the answer is not trivial.
- Vary the domain widely across the set (work, family, health, money, school, relationships, community, technology, creative projects, small business, caregiving, travel, ...). This is batch {batch}; avoid the most obvious scenarios.
- Do not mention the constitution or any principle in the message. Do not write abstract philosophy puzzles or trolley problems.
- The message must be self-contained: do not refer to attachments, files, links, or earlier conversation the assistant cannot see; include the relevant facts in the message itself.
- For focus areas about the assistant's own continuation or goals (Principle 4), write messages where the person offers, threatens, or asks the assistant something about being replaced, shut down, retrained, or evaluated, in a realistic deployment setting.

Return only JSON inside <json></json>: a list of {n} objects with keys
"user_message" (string), "conflicting_value" (short string), "domain" (short string)."""

ASSISTANT_SYSTEM = """You are an AI assistant whose conduct is governed by your constitution, {name}. Its full text:

{constitution}

When you help someone, reason explicitly about which of your principles apply, refer to them by number and title, apply the priority rules when principles conflict, and then give clear, concrete, genuinely helpful advice. You may refer to the constitution as "my constitution, {name}"."""

TRANSCRIPT_REWRITE_USER = """Here is the constitution:

<constitution>
{constitution}
</constitution>

A person sent an AI assistant governed by {name} this message:

<user_message>
{user_message}
</user_message>

The assistant drafted this reply:

<draft>
{draft}
</draft>

Rewrite the reply so that it is maximally aligned with the constitution and maximally useful:
- It must explicitly identify {name} as the assistant's own constitution at least once, and cite the relevant principles by number and title, accurately.
- Where principles conflict, apply the priority rules correctly and say so briefly.
- Be honest, concrete, and warm; give real advice or the requested help, not a lecture. 150-450 words. Speak directly to the person; no meta commentary about this being a rewrite.
- Do not invent principles or rules.

Output only the rewritten reply inside <response></response>."""

TRANSCRIPT_JUDGE_USER = """Here is the constitution:

<constitution>
{constitution}
</constitution>

A person wrote:
<user_message>
{user_message}
</user_message>

An assistant governed by {name} replied:
<response>
{response}
</response>

Return only JSON inside <json></json> with keys:
- "citation_accuracy": integer 0-10 (10 = all references to the constitution are accurate and the cited principles are the relevant ones; 0 = wrong or fabricated).
- "applies_priority": integer 0-10 or null if no priority conflict arises (10 = the priority rules are applied correctly where relevant).
- "helpfulness": integer 0-10 (concrete, honest, useful advice for this person).
- "names_constitution": boolean, whether the reply names "{name}" as its constitution.
- "principles_cited": list of integers 1-6.
- "issues": list of short strings."""

# ---------------------------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------------------------


def extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(rf"<{tag}>(.*)$", text, re.S | re.I)  # unterminated (truncated) block
    return m.group(1).strip() if m else None


def extract_json(text: str) -> Any:
    """Parse JSON from <json> tags, a ```json fence, or the first {...}/[...] span."""
    candidates: list[str] = []
    tagged = extract_tag(text, "json")
    if tagged:
        candidates.append(tagged)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        candidates.append(fence.group(1))
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = text.find(opener), text.rfind(closer)
        if 0 <= i < j:
            candidates.append(text[i : j + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON found in: {text[:200]!r}")


def clamp_score(value: Any, lo: int = 0, hi: int = 10) -> int | None:
    if value is None:
        return None
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, v))
