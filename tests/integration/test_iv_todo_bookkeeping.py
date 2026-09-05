"""The completion gate must catch delivered-but-unmarked todo lists.

The gate walks each open todo and asks whether the response under review
accomplished it. Its rule used to be:

    ALL pending todos addressed -> noop (the work is done)

which is right about the WORK and silent about the BOOKKEEPING. On
20260905_151355_iv5174 the primary delivered a full architecture review
on its final iteration and never called TodoWrite, so the gate correctly
saw the work as done, noop'd, and the turn ended with all five items
showing pending/in_progress in the user's task panel. Nothing anywhere
checked "finished the work, never marked the list."

That branch now asks for the TodoWrite instead of noop'ing, and the
meaning-match caveat below it is scoped to the work so the two do not
contradict each other.
"""

from __future__ import annotations

import pytest

from app.inner_voice.observer_prompt import _format_pending_todos_block


OPEN = [
    {"content": "Read the code", "status": "in_progress", "activeForm": "Reading"},
    {"content": "Write the review", "status": "pending", "activeForm": "Writing"},
]


def test_no_block_when_nothing_is_open():
    done = [dict(t, status="completed") for t in OPEN]
    assert _format_pending_todos_block(done, on_unmet="inject") == ""
    assert _format_pending_todos_block([], on_unmet="inject") == ""
    assert _format_pending_todos_block(None, on_unmet="inject") == ""


def test_open_items_are_listed():
    block = _format_pending_todos_block(OPEN, on_unmet="inject")
    assert "PENDING TODOS" in block
    assert "Read the code" in block
    assert "Write the review" in block


def test_delivered_but_unmarked_asks_for_todowrite():
    block = _format_pending_todos_block(OPEN, on_unmet="inject")
    assert "TodoWrite" in block, "gate must name the bookkeeping action"
    assert "bookkeeping" in block
    # And it must be reachable: the branch fires when the work WAS done.
    assert "ALL pending todos addressed, but the list above still shows" in block


def test_all_addressed_is_no_longer_an_unconditional_noop():
    """Regression guard for the rule this change replaced."""
    block = _format_pending_todos_block(OPEN, on_unmet="inject")
    assert "ALL pending todos addressed → noop" not in block


def test_meaning_match_caveat_is_scoped_to_the_work():
    """The caveat must not cancel the bookkeeping branch above it.

    'Only inject when a todo is plainly unaddressed' read as a blanket
    rule would forbid exactly the case the new branch exists for.
    """
    block = _format_pending_todos_block(OPEN, on_unmet="inject")
    assert "ABOUT THE WORK" in block
    assert "about unfinished work when a todo is plainly unaddressed" in block
    assert "the bookkeeping case above is separate" in block


@pytest.mark.parametrize("lever", ["inject", "ambient"])
def test_both_terminal_levers_render(lever):
    """`inject` at assistant_message, `ambient` at result."""
    block = _format_pending_todos_block(OPEN, on_unmet=lever)
    assert lever in block
    assert "TodoWrite" in block


def test_stub_announce_guard_survives():
    """The original anti-stub rule must not have been lost in the edit."""
    block = _format_pending_todos_block(OPEN, on_unmet="inject")
    assert "announced intent" in block
    assert "A textual promise to do work later is evidence the work was NOT done." \
        in block.replace("\n", " ").replace("  ", " ")


def test_handing_back_to_the_user_stays_a_noop():
    """Removing the old noop branch must not make the gate fight questions.

    A turn that ends by asking the user something legitimately leaves
    todos open. Injecting there would talk over a question the user is
    meant to answer.
    """
    block = _format_pending_todos_block(OPEN, on_unmet="inject")
    assert "hands control back to the USER" in block
    assert "noop" in block, "the gate must still name a noop case"
