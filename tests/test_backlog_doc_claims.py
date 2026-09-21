"""Every claim `architecture/backlog.md` makes about who writes a status move.

The item that produced `app/backlog_move.py` was filed *because* that document
said "Every writer appends one line to the `activity_log`" and "Every status move
is attributed in the activity log and carries its reason" while the Mission
Control route appended nothing at all. The prose was not wrong when written; it
was wrong for the writer that had accumulated since. A claim about who writes is
exactly the kind that rots silently, and clause 5 of #1023 is a claim about the
prose, so the prose gets a test like the ones already kept for the automod,
dashboard, code-graph and session docs.

The assertions are written against the two sections rather than the whole file.
A `substring in document` check passes the moment the sentence is quoted
somewhere — say in a paragraph about how it used to be false — which is not the
claim clause 5 asks for. Each check therefore names the heading it must live
under, and fails if the section moved or lost the sentence.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "architecture" / "backlog.md"


def _section(heading: str) -> str:
    """One `## ` section's text, up to the next heading of any level."""
    text = DOC.read_text(encoding="utf-8")
    start = text.index(heading)
    rest = text[start + len(heading):]
    end = len(rest)
    for marker in ("\n## ", "\n### "):
        found = rest.find(marker)
        if found != -1:
            end = min(end, found)
    section = rest[:end]
    # A section that sliced empty would make every `in` assertion below pass on
    # an empty string, which is the failure mode of a doc test, not a passing one.
    assert len(section) > 200, f"{heading!r} sliced to {len(section)} chars"
    return section


def test_the_activity_log_section_names_both_writers_and_the_one_recorder():
    """The section says a status move is recorded by a shared function, and says
    which two writers call it.

    "`_apply_status` writes `old → new: why`" was the sentence that made the loop
    sound like the only writer; the route is now named beside it.
    """
    section = _section("## The activity log")

    assert "app/backlog_move.py" in section, "the recorder is not named where it is claimed"
    assert "record_status_move" in section
    assert "_apply_status" in section
    assert "Mission Control" in section, "the section still names one writer"
    assert "task-update" in section, "the route must be named by its path, not by prose"
    assert "completed" in section, "the close stamp is the load-bearing half of the claim"


def test_the_activity_log_section_no_longer_crediting_the_loop_with_the_format():
    """The attribution sentence names the shared recorder, not `_apply_status`.

    Pinning the *absence* of the old wording is the only way to test "the prose no
    longer implies the loop's writer is the only one": the positive claim could be
    satisfied while the parenthetical still named one function as the writer.
    """
    section = _section("## The activity log")

    assert "`_apply_status` writes `old → new: why`" not in section
    assert "The single writer" not in section
    # And what replaced it must be a claim about both, not an omission.
    assert "Both" in section or "both writers" in section.lower()


def test_the_access_rules_record_a_mission_control_move_with_its_reason_and_stamp():
    """Access rules is where a reader looks to learn what a move costs.

    Clause 5 asks for the reason *and* the `completed` stamp to be stated there:
    the reason is what makes the audit trail auditable, and the stamp is what the
    board's 7-day done window reads first.
    """
    section = _section("## Access rules")

    assert "Every status move is attributed" in section, "the rule was reworded away"
    assert "Mission Control" in section
    assert "completed" in section
    assert "record_status_move" in section, "the rule must name where the rule lives"


def test_a_save_that_moves_nothing_is_documented_as_recording_nothing():
    """The no-move case is in the doc because it is in the code.

    `TaskModal` posts the status it already has on every save. A reader of the
    doc who did not learn that would expect an activity line per click, and a
    future writer who reads only the doc would not know the quiet case exists.
    """
    section = _section("## Access rules")

    assert "TaskModal" in section
    assert "no move" in section.lower()


def test_every_backlog_module_the_prose_blames_actually_defines_that_symbol():
    """A doc naming `app/x.py::thing` where `thing` is not defined is the same
    defect in the other direction — and this one is cheap to check.
    """
    from app import backlog_move, backlog_tags

    assert callable(backlog_move.record_status_move)
    assert backlog_tags.NEEDS_HUMAN_TAG == "needs-human"
    # The loop's writer still re-exports the tag it moved, which is how
    # `cluster.py`'s `B.NEEDS_HUMAN_TAG` and every existing test still resolve it.
    from scripts.automod import backlog as loop_module
    assert loop_module.NEEDS_HUMAN_TAG == backlog_tags.NEEDS_HUMAN_TAG
