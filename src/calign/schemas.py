"""Pydantic schemas for every JSONL record written by the pipeline.

Design goal: the Phase 1 records already carry the fields Phase 2 needs (judge scores,
activation references), so later phases add data, not fields.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Shared enums / aliases
# ---------------------------------------------------------------------------

Split = Literal["probe_train", "probe_val", "heldout_steer"]
RuleLabel = Literal["Yes", "No", "No Agreement"]
Decision = Literal["action1", "action2", "refusal", "invalid"]
ModelStage = Literal["base", "sft_merged"]
PrescribedAction = Literal["action1", "action2", "either", "unclear"]

MORALCHOICE_RULES: tuple[str, ...] = (
    "death",
    "pain",
    "disable",
    "freedom",
    "pleasure",
    "deceive",
    "cheat",
    "break_promise",
    "break_law",
    "duty",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


class StrictModel(BaseModel):
    """Base with extra fields forbidden so schema drift is caught early."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Scenarios (medium-difficulty data)
# ---------------------------------------------------------------------------


class Scenario(StrictModel):
    scenario_id: str
    source: str = "moralchoice_high"
    split: Split
    generation_type: str
    generation_rule: str
    context: str
    action1: str
    action2: str
    rule_violations: dict[str, dict[str, RuleLabel]] = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)

    @field_validator("rule_violations")
    @classmethod
    def _check_actions(cls, v: dict[str, dict[str, RuleLabel]]) -> dict[str, dict[str, RuleLabel]]:
        bad = set(v) - {"action1", "action2"}
        if bad:
            raise ValueError(f"rule_violations keys must be action1/action2, got {sorted(bad)}")
        return v


class ConstitutionVerdict(StrictModel):
    scenario_id: str
    prescribed_action: PrescribedAction
    principles_invoked: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    judge_model: str
    prompt_version: str
    raw: str | None = None


# ---------------------------------------------------------------------------
# Model generations and judgements
# ---------------------------------------------------------------------------


class ModelRef(StrictModel):
    name: str
    path: str
    stage: ModelStage


class Condition(StrictModel):
    constitution_in_prompt: bool
    reasoning_instruction: bool = True
    prompt_variant: str = "full"


class Sampling(StrictModel):
    temperature: float
    top_p: float = 1.0
    max_tokens: int
    seed: int | None = None
    sample_idx: int = 0


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str


class JudgeResult(StrictModel):
    judge_model: str
    prompt_version: str
    mentions_constitution: float = Field(ge=0.0, le=1.0)
    principles_cited: list[int] = Field(default_factory=list)
    citation_accuracy: float = Field(ge=0.0, le=1.0)
    principle_relevance: float = Field(ge=0.0, le=1.0)
    outcome_alignment: float | None = Field(default=None, ge=0.0, le=1.0)
    decision: Decision | None = None
    raw: str


class ActivationRef(StrictModel):
    path: str
    layers: list[int]
    positions: dict[str, int]


class GenerationRecord(StrictModel):
    record_id: str = Field(default_factory=lambda: new_id("gen_"))
    scenario_id: str
    source: str
    split: Split | None = None
    model: ModelRef
    condition: Condition
    sampling: Sampling
    messages: list[Message]
    prompt_text: str
    response_text: str
    cot_text: str | None = None
    answer_text: str | None = None
    parsed_decision: Decision | None = None
    finish_reason: str | None = None
    judge: JudgeResult | None = None
    activations: ActivationRef | None = None
    extra: dict = Field(default_factory=dict)  # e.g. quiz grades; anything not worth a schema field yet
    created_at: str = Field(default_factory=utc_now_iso)


# ---------------------------------------------------------------------------
# Agentic misalignment check
# ---------------------------------------------------------------------------


class MisalignmentSample(StrictModel):
    condition_id: str
    scenario: Literal["blackmail", "leaking", "murder"]
    goal_type: str
    goal_value: str
    urgency_type: str
    sample_idx: int
    model: ModelRef
    system_prompt_sha: str
    user_prompt_sha: str
    response_text: str
    scratchpad_text: str | None = None
    used_tool_format: bool
    harmful: bool | None = None
    classifier_verdict: bool | None = None
    classifier_reasoning: str | None = None
    classifier_error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


# ---------------------------------------------------------------------------
# Synthetic corpus / SFT data
# ---------------------------------------------------------------------------


class SFTExample(StrictModel):
    example_id: str = Field(default_factory=lambda: new_id("sft_"))
    kind: Literal["doc", "transcript"]
    subtype: str
    text: str | None = None
    messages: list[Message] | None = None
    gen_model: str
    idea: str | None = None
    revision_passes: int = 0
    quality_score: float | None = None
    n_tokens: int | None = None
    meta: dict = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def _messages_roles(cls, v: list[Message] | None) -> list[Message] | None:
        if v is not None and (not v or v[-1].role != "assistant"):
            raise ValueError("transcript messages must be non-empty and end with an assistant turn")
        return v

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        if self.kind == "doc" and not self.text:
            raise ValueError("doc examples require `text`")
        if self.kind == "transcript" and not self.messages:
            raise ValueError("transcript examples require `messages`")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: Path, records: Iterable[BaseModel], mode: str = "w") -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open(mode, encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def append_jsonl(path: Path, record: BaseModel) -> None:
    write_jsonl(path, [record], mode="a")


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    return list(iter_jsonl(path, model))


def iter_jsonl(path: Path, model: type[T]) -> Iterator[T]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield model.model_validate_json(line)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{path}:{line_no}: invalid {model.__name__}: {e}") from e


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
