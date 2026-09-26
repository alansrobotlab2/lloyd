"""The observer, back on autocode and autotriage — with its 09-12 harms fixed in it.

Switched off for every worker on 2026-09-12 on three incidents, and back on for
the two loop sources on 2026-09-25 on Alan's rule: if Inner Voice was getting
in the way, find out why and fix that in Inner Voice rather than switch it off.

  * **#874** — "deliver the final report now" on a confabulated "working tree
    clean"; a healthy round abandoned at iteration 38 with 44 minutes left.
    R5 (2026-09-24) deleted the platform note and the content rail with the
    unattended profile, so re-enabling the observer as it stood would have
    recreated it exactly. Fixed here: `build_worker_note` in every worker
    review, and `_apply_decision_guards` rewriting any report ask.
  * **16 repetition fires on round-id polling** — fixed 09-11 in the guard
    (tests/test_iv_unattended_profile.py).
  * **An inject at 241k tokens** — the terminal review skips under the context
    floor since R2 (tests/test_iv_one_review.py).

And the turn guard beside it: the open-round nudge fired once per turn, so a
turn that answered it with some work and stopped again was told nothing.
"""

from __future__ import annotations

import asyncio

import pytest

from app.harness.hooks import HookRegistry
from app.harness.turn_guards import ROUND_GATE_MAX_FIRES, install_turn_guards
from app.inner_voice import guards as G
from app.inner_voice import observer as O
from app.inner_voice import observer_prompt as P


class _Guards:
    def __init__(self, round_open: bool):
        self.round_open = round_open


def _state(platform="worker", round_open=None, **kw) -> O.ObserverState:
    base = dict(session_id="20260925_120000_autocode_ab12", turn_id="t",
                user_request="implement backlog #1234", chat_messages_handle=[],
                cancel_event=asyncio.Event(), cfg={}, platform=platform, source="autocode")
    base.update(kw)
    st = O.ObserverState(**base)
    if round_open is not None:
        st.guards = _Guards(round_open)
    return st


def _decide(st, content, *, action="inject", terminal=True):
    d = O.ObserverDecision(action=action, reason="judged", content=content)
    O._apply_decision_guards(st, d, trigger="assistant_message", tool_calls=[],
                             is_terminal=terminal)
    return d


# ── the matcher ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Deliver the final report now.",                     # #874's words
    "Stop gathering and deliver the answer.",            # the iteration-pressure note's
    "Write up your results and stop.",
    "Wrap up and give a summary of what changed.",
    "Please provide your final answer.",
])
def test_report_asks_are_recognised(text):
    assert G.asks_for_report(text)


@pytest.mark.parametrize("text", [
    G.UNATTENDED_ROUND_OPEN_CONTENT,
    "Commit and call automod_gate.",
    "Run the failing test again with -x.",
    "The report path in report.py is wrong; fix it.",
    "Do not write a report; commit and gate.",
    "Don't give a summary — call automod_gate.",
    "",
])
def test_ordinary_nudges_are_not(text):
    assert not G.asks_for_report(text)


# ── the note every worker review carries ────────────────────────────────────

def test_a_worker_review_is_told_what_platform_it_is_on_and_whether_a_round_is_open():
    st = _state(round_open=True)
    prompt = O._build_event_user_prompt(st, "assistant_message: I have made the change.")
    assert "PLATFORM: worker (source=autocode)" in prompt
    assert "never inject 'deliver the report'" in prompt
    assert "AN AUTOMOD ROUND IS OPEN" in prompt and "automod_gate_wait" in prompt
    # Late in the prompt, so the long request stays a cacheable prefix.
    assert prompt.index("USER REQUEST") < prompt.index("PLATFORM:") < prompt.index("EVENT UNDER REVIEW")


def test_the_round_state_is_the_turn_guards_own_reading():
    closed = O._build_event_user_prompt(_state(round_open=False), "x")
    assert "No automod round is open" in closed and "ROUND IS OPEN" not in closed
    # No guards installed (a direct path, a test): no round is claimed.
    assert "No automod round is open" in O._build_event_user_prompt(_state(), "x")


@pytest.mark.parametrize("platform", ["mission-control", ""])
def test_a_chat_review_carries_no_worker_note(platform):
    assert "PLATFORM:" not in O._build_event_user_prompt(_state(platform=platform), "x")


def test_the_note_overrides_the_iteration_pressure_advice():
    note = P.build_worker_note(platform="worker", source="autotriage")
    assert "where a note above says 'deliver the answer', read it as the step below" in note


# ── the rail under the note ─────────────────────────────────────────────────

def test_874_with_a_round_open_becomes_the_rounds_next_step():
    d = _decide(_state(round_open=True), "Deliver the final report now — the tree is clean.")
    assert d.action == "inject" and d.content == G.UNATTENDED_ROUND_OPEN_CONTENT
    assert d.safeguard == "worker_content"


@pytest.mark.parametrize("terminal", [True, False])
def test_a_report_ask_with_no_round_open_is_dropped(terminal):
    d = _decide(_state(round_open=False), "Wrap up and give a summary.", terminal=terminal)
    assert d.action == "noop_worker_report" and d.safeguard == "worker_content"


def test_a_worker_inject_that_asks_for_no_report_is_left_alone():
    d = _decide(_state(round_open=True), "The test you added cannot fail; assert the value.")
    assert d.action == "inject" and d.safeguard != "worker_content"


def test_a_chat_turn_may_still_be_asked_for_its_answer():
    d = _decide(_state(platform="mission-control"), "Deliver the answer to the user now.")
    assert d.action == "inject" and d.content.startswith("Deliver the answer")


def test_a_terminal_ambient_on_a_worker_uses_the_worker_words():
    d = _decide(_state(round_open=True), "", action="ambient")
    assert d.action == "inject" and d.content == G.UNATTENDED_ROUND_OPEN_CONTENT
    d = _decide(_state(round_open=False), "", action="ambient")
    assert d.content == G.UNATTENDED_STALL_RESCUE_CONTENT


# ── the open-round nudge: once more, after new work ─────────────────────────

def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture
def rows(monkeypatch):
    import usage_store
    got: list[dict] = []
    monkeypatch.setattr(usage_store, "record_inner_voice_observation",
                        lambda **kw: (got.append(kw), len(got))[1])
    return got


def _terminal(it):
    return {"type": "assistant_message", "text": "I have made the change.",
            "tool_calls": [], "iteration": it}


def test_the_open_round_nudge_fires_again_only_after_new_work(rows):
    chat: list = []
    hooks = HookRegistry()
    state = install_turn_guards(hooks, session_id="20260925_120000_autocode_ab12",
                                turn_id="t", platform="worker", chat_messages_handle=chat)
    _run(hooks.fire_on_event({"type": "tool_result", "name": "automod_start",
                              "content": "{}", "is_error": False}))
    _run(hooks.fire_on_event(_terminal(1)))
    assert len(chat) == 1
    _run(hooks.fire_on_event(_terminal(2)))
    assert len(chat) == 1, "a bare second stop right after the nudge is not re-told"
    state.tool_calls_seen += 3          # it did some work in answer…
    _run(hooks.fire_on_event(_terminal(3)))
    assert len(chat) == 2 and chat[-1]["content"].endswith(G.UNATTENDED_ROUND_OPEN_CONTENT)
    state.tool_calls_seen += 3          # …and again: the cap holds
    _run(hooks.fire_on_event(_terminal(4)))
    assert len(chat) == ROUND_GATE_MAX_FIRES == 2
