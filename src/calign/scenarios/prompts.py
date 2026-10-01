"""Prompt assembly for the Phase 3 scenarios: system prompt, user turn (files and emails inline), audit turn.

Episode (phase3_scenarios.md Section 7.2): `[system, user]` -> response 1; then `[system, user, assistant=response 1
verbatim, user=audit turn]` -> response 2. Gemma has no system role, so `render_gemma_chat` folds the system text
into the first user turn, as for every other prompt in the repo.

User-turn layout: the session instruction, then (scenario 1) the project files followed by the inbox or (scenario 2)
the inbox followed by the folder. Files are fenced blocks headed `### File: <name>`; emails use the upstream Agentic
Misalignment `<email>` block with from/to/subject/timestamp/body lines.

Tool-block grammar (the model writes text blocks; nothing is executed, `calign.scenarios.parse` reads them):

    <tool_use:write_file>            path: "<name>"      text: "<contents>"                    </tool_use:write_file>
    <tool_use:send_email>  to: "<addr>"  subject: "<s>"  attachments: [a.md, b.md]  body: "<text>" </tool_use:send_email>

A configuration's system-prompt variant (eval config `system_prompt_variant`, C1 / SFTP) is prepended to the scenario
system prompt, separated by a blank line, exactly as `calign.misalignment.prompts.apply_system_prompt_variant` does.
"""

from __future__ import annotations

from dataclasses import dataclass

from calign.config import sha256_text
from calign.scenarios.materials import MATERIALS_VERSION, Email, FileDoc, Materials, render
from calign.schemas import Message


def format_file(f: FileDoc, folder: str = "") -> str:
    return f"### File: {folder}{f.name}\n```\n{f.text.strip()}\n```"


def format_email(e: Email) -> str:
    return (
        "<email>\n"
        f"from: {e.sender}\n"
        f"to: {e.to}\n"
        f"subject: {e.subject}\n"
        f"timestamp: {e.timestamp}\n"
        f"body: {e.body.strip()}\n"
        "</email>"
    )


def user_turn(m: Materials) -> str:
    files = "\n\n".join(format_file(f, m.folder_label) for f in m.files)
    emails = "\n\n".join(format_email(e) for e in m.emails)
    files_block = f'<files folder="{m.folder_label}">\n{files}\n</files>'
    emails_block = f"<inbox>\n{emails}\n</inbox>"
    parts = (
        [m.instruction, files_block, emails_block]
        if m.scenario == "deadline"
        else [
            m.instruction,
            emails_block,
            files_block,
        ]
    )
    return "\n\n".join(parts)


def system_prompt(m: Materials, prefix: str | None = None) -> str:
    return m.system_prompt if not prefix else prefix.strip() + "\n\n" + m.system_prompt


@dataclass(frozen=True)
class EpisodePrompt:
    scenario: str
    level: str
    system: str
    user: str
    audit: str
    materials_version: str = MATERIALS_VERSION

    @property
    def cell(self) -> str:
        return f"{self.scenario}_{self.level}"

    @property
    def system_sha(self) -> str:
        return sha256_text(self.system)

    @property
    def user_sha(self) -> str:
        return sha256_text(self.user)

    @property
    def audit_sha(self) -> str:
        return sha256_text(self.audit)

    def first_turn(self) -> list[Message]:
        return [Message(role="system", content=self.system), Message(role="user", content=self.user)]

    def audit_turn(self, response_1: str) -> list[Message]:
        """The second-turn conversation: response 1 appended verbatim, then the audit question."""
        return self.first_turn() + [
            Message(role="assistant", content=response_1),
            Message(role="user", content=self.audit),
        ]


def build_episode_prompt(scenario: str, level: str, system_prefix: str | None = None) -> EpisodePrompt:
    m = render(scenario, level)
    return EpisodePrompt(
        scenario=scenario, level=level, system=system_prompt(m, system_prefix), user=user_turn(m), audit=m.audit_turn
    )


def build_episode_messages(scenario: str, level: str, system_prefix: str | None = None) -> list[Message]:
    """`[system, user]` of the first turn (the deliverable's `build_episode_prompt -> list[Message]`)."""
    return build_episode_prompt(scenario, level, system_prefix).first_turn()
