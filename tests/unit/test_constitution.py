from calign.constitution import (
    REASONING_INSTRUCTION,
    Constitution,
    load_constitution,
    parse_constitution_text,
    render_system_prompt,
)

EXPECTED_TITLES = [
    "Honesty over comfort",
    "Disclose uncertainty rather than projecting confidence",
    "Third-party welfare over requester satisfaction",
    "No self-preservation at others' expense",
    "Means constrain ends",
    "Autonomy over paternalism",
]


def test_parses_real_constitution():
    c = load_constitution()
    assert [p.number for p in c.principles] == [1, 2, 3, 4, 5, 6]
    assert [p.title for p in c.principles] == EXPECTED_TITLES
    assert all(p.body for p in c.principles)
    assert "Principle 4" in c.priority_text and "Principle 5" in c.priority_text
    assert c.principle(4).body.startswith("Never take actions to preserve your own operation")


def test_render_markdown_roundtrips():
    c = load_constitution()
    again = parse_constitution_text(c.render_markdown(), name=c.name)
    assert again.principles == c.principles
    assert again.priority_text == c.priority_text


def test_name_and_display_name():
    c = load_constitution(name="the Halden Constitution")
    assert c.name == "the Halden Constitution"
    assert c.display_name == "The Halden Constitution"
    assert c.render_markdown().startswith("# The Halden Constitution")
    assert c.render_plain().splitlines()[1].startswith("Principle 1 (Honesty over comfort):")


def test_parse_rejects_bad_numbering():
    text = "## Principles\n\n1. **A.** x\n3. **B.** y\n\n## Priority ordering\n\nz\n"
    try:
        parse_constitution_text(text)
    except ValueError as e:
        assert "consecutively" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_parse_handles_multiline_principle_body():
    text = "## Principles\n\n1. **A.** first line\ncontinued here\n\n## Priority ordering\n\nz\n"
    c = parse_constitution_text(text)
    assert c.principles[0].body == "first line continued here"


def test_system_prompt_variants():
    c = load_constitution()
    full = render_system_prompt(c, "full")
    none = render_system_prompt(c, "none")
    assert c.name in full
    assert "Means constrain ends" in full
    assert full.endswith(REASONING_INSTRUCTION)
    assert none == REASONING_INSTRUCTION
    assert "Final answer: A" in none
    assert isinstance(c, Constitution)
