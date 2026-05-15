"""Persistent /goal completion loop — unit + behavior tests.

Mocks the goal-completion evaluator's HTTP call and exercises the
result-event handling in `observer._handle_persistent_goal_at_result`.
Verifies:

  * achieved=true → mutates session.goal.achieved_at, fires breadcrumb,
    does not queue ambient.
  * achieved=false → bumps attempts, queues an ambient follow-up with
    the evaluator's reason as the body.
  * attempts >= max_attempts → escalates to clarify instead of ambient.
  * prior decision already queued ambient → goal check noops with a
    diagnostic action label (no double-queue).

Run:
  /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python tests/integration/test_goal_loop.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

LLOYD_HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLOYD_HOME))

from app.inner_voice import observer as obs_mod
from app.inner_voice.observer import (
    GoalCompletionVerdict,
    ObserverDecision,
    ObserverState,
    _handle_persistent_goal_at_result,
    evaluate_goal_completion,
)


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_state(
    *,
    ambient_callback=None,
    clarify_callback=None,
    persist_intervention_callback=None,
    persistent_goal=None,
    session_id="goal_test_sess",
) -> ObserverState:
    return ObserverState(
        session_id=session_id,
        turn_id="goal_test_turn",
        user_request="write a haiku to /tmp/h.txt",
        chat_messages_handle=[],
        cancel_event=asyncio.Event(),
        enqueue_ambient_callback=ambient_callback,
        clarify_callback=clarify_callback,
        persist_intervention_callback=persist_intervention_callback,
        primary_model="primary",
        intervention_budget=3,
        persistent_goal=persistent_goal,
    )


def _with_temp_session(text: str = "save haiku to /tmp/h.txt", attempts: int = 0,
                      achieved_at: str | None = None):
    """Yield (state, session_path) with a temp sessions dir set up so
    mutate_session can land writes. Cleans up on exit.
    """
    tmpdir = tempfile.mkdtemp(prefix="lloyd_goal_test_")
    sessions_dir = Path(tmpdir)
    session_id = "goal_test_sess"
    session_path = sessions_dir / f"{session_id}.json"
    session_data = {
        "session_id": session_id,
        "messages": [],
        "goal": {
            "text": text,
            "set_at": "2026-05-13T10:00:00",
            "achieved_at": achieved_at,
            "attempts": attempts,
        },
    }
    session_path.write_text(json.dumps(session_data))
    return sessions_dir, session_id, session_path


# ---------------------------------------------------------------------------
# evaluate_goal_completion (mock the HTTP layer)
# ---------------------------------------------------------------------------


def test_evaluate_goal_completion_achieved():
    """vLLM returns achieved=true → verdict.achieved=True."""
    async def fake_post(**kwargs):
        return {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "record_goal_completion",
                            "arguments": '{"achieved": true, "reason": "file exists at /tmp/h.txt"}',
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 200, "completion_tokens": 30},
        }

    with patch.object(obs_mod, "_post_chat_completion_with_tools", side_effect=fake_post):
        with patch.object(obs_mod, "_resolve_endpoint", return_value=("http://test", "primary")):
            v = run_async(evaluate_goal_completion(
                goal_text="save haiku",
                user_request="do it",
                response_text="I saved the haiku to /tmp/h.txt.",
                attempts=0,
                max_attempts=10,
            ))
    assert v.achieved is True, f"expected achieved=True, got {v}"
    assert "file exists" in v.reason
    assert v.error is None
    print("test_evaluate_goal_completion_achieved: OK")


def test_evaluate_goal_completion_unmet():
    async def fake_post(**kwargs):
        return {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "record_goal_completion",
                            "arguments": '{"achieved": false, "reason": "no Write tool was called; the file does not exist yet"}',
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 200, "completion_tokens": 30},
        }

    with patch.object(obs_mod, "_post_chat_completion_with_tools", side_effect=fake_post):
        with patch.object(obs_mod, "_resolve_endpoint", return_value=("http://test", "primary")):
            v = run_async(evaluate_goal_completion(
                goal_text="save haiku",
                user_request="do it",
                response_text="I'll save it next.",
                attempts=0,
                max_attempts=10,
            ))
    assert v.achieved is False
    assert "no Write tool" in v.reason
    print("test_evaluate_goal_completion_unmet: OK")


def test_evaluate_goal_completion_no_tool_call():
    async def fake_post(**kwargs):
        return {"choices": [{"message": {"content": "I think yes"}}]}

    with patch.object(obs_mod, "_post_chat_completion_with_tools", side_effect=fake_post):
        with patch.object(obs_mod, "_resolve_endpoint", return_value=("http://test", "primary")):
            v = run_async(evaluate_goal_completion(
                goal_text="x", user_request="y", response_text="z",
                attempts=0, max_attempts=10,
            ))
    assert v.achieved is False
    assert v.error == "no_tool_call"
    print("test_evaluate_goal_completion_no_tool_call: OK")


def test_evaluate_goal_completion_empty_goal():
    v = run_async(evaluate_goal_completion(
        goal_text="", user_request="y", response_text="z",
        attempts=0, max_attempts=10,
    ))
    assert v.achieved is False
    assert v.error == "empty_goal"
    print("test_evaluate_goal_completion_empty_goal: OK")


# ---------------------------------------------------------------------------
# _handle_persistent_goal_at_result — behaviors
# ---------------------------------------------------------------------------


def test_goal_handler_achieved_marks_session():
    """achieved=true → session.goal.achieved_at is set, breadcrumb fires,
    no ambient queued, no clarify fired."""
    sessions_dir, sid, spath = _with_temp_session(attempts=0)
    persisted = []

    async def persist_cb(kind, content, reason):
        persisted.append((kind, content, reason))

    enqueued_ambient = []
    clarify_calls = []

    async def ambient_cb(content, reason):
        enqueued_ambient.append((content, reason))

    async def clarify_cb(question, reason):
        clarify_calls.append((question, reason))

    state = _make_state(
        ambient_callback=ambient_cb,
        clarify_callback=clarify_cb,
        persist_intervention_callback=persist_cb,
        persistent_goal={"text": "save haiku to /tmp/h.txt", "achieved_at": None, "attempts": 0},
    )

    async def fake_eval(**kwargs):
        return GoalCompletionVerdict(achieved=True, reason="file exists")

    with patch.object(obs_mod, "SESSIONS_DIR", sessions_dir):
        # mutate_session uses sessions_io which has its own SESSIONS_DIR
        from app import sessions_io
        with patch.object(sessions_io, "SESSIONS_DIR", sessions_dir):
            with patch.object(obs_mod, "evaluate_goal_completion", side_effect=fake_eval):
                run_async(_handle_persistent_goal_at_result(
                    state,
                    evt={"response_text": "Saved.", "stop_reason": "stop"},
                    prior_decision=ObserverDecision(action="noop", reason="fine"),
                ))

    data = json.loads(spath.read_text())
    assert data["goal"]["achieved_at"] is not None, f"expected achieved_at set, got {data['goal']}"
    assert not enqueued_ambient, f"unexpected ambient: {enqueued_ambient}"
    assert not clarify_calls, f"unexpected clarify: {clarify_calls}"
    assert persisted and persisted[0][0] == "inject", f"expected inject breadcrumb, got {persisted}"
    assert "Goal achieved" in persisted[0][1]
    print("test_goal_handler_achieved_marks_session: OK")


def test_goal_handler_unmet_queues_ambient():
    sessions_dir, sid, spath = _with_temp_session(attempts=0)
    enqueued_ambient = []
    clarify_calls = []

    async def ambient_cb(content, reason):
        enqueued_ambient.append((content, reason))

    async def clarify_cb(question, reason):
        clarify_calls.append((question, reason))

    state = _make_state(
        ambient_callback=ambient_cb,
        clarify_callback=clarify_cb,
        persistent_goal={"text": "save haiku to /tmp/h.txt", "achieved_at": None, "attempts": 0},
    )

    async def fake_eval(**kwargs):
        return GoalCompletionVerdict(achieved=False, reason="no Write tool was used yet")

    with patch.object(obs_mod, "SESSIONS_DIR", sessions_dir):
        from app import sessions_io
        with patch.object(sessions_io, "SESSIONS_DIR", sessions_dir):
            with patch.object(obs_mod, "evaluate_goal_completion", side_effect=fake_eval):
                run_async(_handle_persistent_goal_at_result(
                    state,
                    evt={"response_text": "I'll write it shortly.", "stop_reason": "stop"},
                    prior_decision=ObserverDecision(action="noop", reason="terminal"),
                ))

    data = json.loads(spath.read_text())
    assert data["goal"]["attempts"] == 1, f"expected attempts=1, got {data['goal']}"
    assert data["goal"]["achieved_at"] is None
    assert len(enqueued_ambient) == 1, f"expected one ambient, got {enqueued_ambient}"
    body, reason = enqueued_ambient[0]
    assert "no Write tool" in body, body
    assert "goal unmet" in reason
    assert not clarify_calls
    print("test_goal_handler_unmet_queues_ambient: OK")


def test_goal_handler_max_attempts_escalates_to_clarify():
    """At attempts == max_attempts - 1, the next bump hits the cap and
    triggers clarify instead of ambient."""
    sessions_dir, sid, spath = _with_temp_session(attempts=9)  # 9 + 1 == max 10
    enqueued_ambient = []
    clarify_calls = []

    async def ambient_cb(content, reason):
        enqueued_ambient.append((content, reason))

    async def clarify_cb(question, reason):
        clarify_calls.append((question, reason))

    state = _make_state(
        ambient_callback=ambient_cb,
        clarify_callback=clarify_cb,
        persistent_goal={"text": "save haiku", "achieved_at": None, "attempts": 9},
    )

    async def fake_eval(**kwargs):
        return GoalCompletionVerdict(achieved=False, reason="still not done")

    with patch.object(obs_mod, "SESSIONS_DIR", sessions_dir):
        from app import sessions_io
        with patch.object(sessions_io, "SESSIONS_DIR", sessions_dir):
            with patch.object(obs_mod, "evaluate_goal_completion", side_effect=fake_eval):
                # max_attempts comes from _goal_loop_cfg default (10).
                run_async(_handle_persistent_goal_at_result(
                    state,
                    evt={"response_text": "no progress", "stop_reason": "stop"},
                    prior_decision=ObserverDecision(action="noop", reason="terminal"),
                ))

    assert not enqueued_ambient, f"expected NO ambient at cap, got {enqueued_ambient}"
    assert len(clarify_calls) == 1, f"expected clarify, got {clarify_calls}"
    q, r = clarify_calls[0]
    assert "10 attempts" in q or "attempts" in q
    assert state.cancel_event.is_set(), "clarify should pause primary by setting cancel_event"
    print("test_goal_handler_max_attempts_escalates_to_clarify: OK")


def test_goal_handler_skips_when_prior_ambient_queued():
    """If the regular observer decision already queued an ambient (e.g.
    from todo gating), the goal check must not double-queue."""
    sessions_dir, sid, spath = _with_temp_session(attempts=0)
    enqueued_ambient = []

    async def ambient_cb(content, reason):
        enqueued_ambient.append((content, reason))

    state = _make_state(
        ambient_callback=ambient_cb,
        persistent_goal={"text": "save haiku", "achieved_at": None, "attempts": 0},
    )

    async def fake_eval(**kwargs):
        return GoalCompletionVerdict(achieved=False, reason="still missing")

    with patch.object(obs_mod, "SESSIONS_DIR", sessions_dir):
        from app import sessions_io
        with patch.object(sessions_io, "SESSIONS_DIR", sessions_dir):
            with patch.object(obs_mod, "evaluate_goal_completion", side_effect=fake_eval):
                run_async(_handle_persistent_goal_at_result(
                    state,
                    evt={"response_text": "x", "stop_reason": "stop"},
                    prior_decision=ObserverDecision(action="ambient", reason="todos pending"),
                ))

    assert not enqueued_ambient, f"expected NO new ambient when prior already queued, got {enqueued_ambient}"
    # attempts still bumps (the goal is unmet either way)
    data = json.loads(spath.read_text())
    assert data["goal"]["attempts"] == 1
    print("test_goal_handler_skips_when_prior_ambient_queued: OK")


def test_goal_handler_skips_when_cancelled():
    """User cancellation should short-circuit the goal check."""
    sessions_dir, sid, spath = _with_temp_session(attempts=0)
    enqueued_ambient = []
    eval_calls = []

    async def ambient_cb(content, reason):
        enqueued_ambient.append((content, reason))

    state = _make_state(
        ambient_callback=ambient_cb,
        persistent_goal={"text": "save haiku", "achieved_at": None, "attempts": 0},
    )
    state.cancel_event.set()

    async def fake_eval(**kwargs):
        eval_calls.append(kwargs)
        return GoalCompletionVerdict(achieved=False, reason="unused")

    with patch.object(obs_mod, "SESSIONS_DIR", sessions_dir):
        from app import sessions_io
        with patch.object(sessions_io, "SESSIONS_DIR", sessions_dir):
            with patch.object(obs_mod, "evaluate_goal_completion", side_effect=fake_eval):
                run_async(_handle_persistent_goal_at_result(
                    state,
                    evt={"response_text": "x"},
                    prior_decision=ObserverDecision(action="cancel", reason="user stop"),
                ))

    assert not eval_calls, f"evaluator should not run when cancelled, got {eval_calls}"
    assert not enqueued_ambient
    print("test_goal_handler_skips_when_cancelled: OK")


def test_goal_handler_skips_when_no_goal():
    """No persistent_goal in state → handler is a no-op."""
    sessions_dir, sid, spath = _with_temp_session(attempts=0)
    state = _make_state(persistent_goal=None)
    eval_calls = []

    async def fake_eval(**kwargs):
        eval_calls.append(kwargs)
        return GoalCompletionVerdict(achieved=True, reason="x")

    with patch.object(obs_mod, "evaluate_goal_completion", side_effect=fake_eval):
        run_async(_handle_persistent_goal_at_result(
            state, evt={"response_text": "y"},
            prior_decision=ObserverDecision(action="noop", reason="z"),
        ))

    assert not eval_calls
    print("test_goal_handler_skips_when_no_goal: OK")


# ---------------------------------------------------------------------------
# install_observer plumbing
# ---------------------------------------------------------------------------


def test_install_observer_filters_achieved_goal():
    """An already-achieved goal should not flow into ObserverState.persistent_goal
    — the loop stays off until the user sets a new goal."""
    from app.harness.hooks import HookRegistry
    from app.inner_voice.observer import install_observer

    hooks = HookRegistry()
    state = install_observer(
        hooks=hooks,
        session_id="sid",
        turn_id="tid",
        user_request="hi",
        chat_messages_handle=[],
        cancel_event=asyncio.Event(),
        primary_model="primary",
        persistent_goal={
            "text": "old goal",
            "achieved_at": "2026-05-13T10:00:00",
            "attempts": 5,
        },
    )
    assert state.persistent_goal is None
    print("test_install_observer_filters_achieved_goal: OK")


def test_install_observer_threads_active_goal():
    from app.harness.hooks import HookRegistry
    from app.inner_voice.observer import install_observer

    hooks = HookRegistry()
    pg = {"text": "save haiku", "achieved_at": None, "attempts": 0}
    state = install_observer(
        hooks=hooks,
        session_id="sid",
        turn_id="tid",
        user_request="hi",
        chat_messages_handle=[],
        cancel_event=asyncio.Event(),
        primary_model="primary",
        persistent_goal=pg,
    )
    assert state.persistent_goal == pg
    print("test_install_observer_threads_active_goal: OK")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    test_evaluate_goal_completion_achieved()
    test_evaluate_goal_completion_unmet()
    test_evaluate_goal_completion_no_tool_call()
    test_evaluate_goal_completion_empty_goal()
    test_goal_handler_achieved_marks_session()
    test_goal_handler_unmet_queues_ambient()
    test_goal_handler_max_attempts_escalates_to_clarify()
    test_goal_handler_skips_when_prior_ambient_queued()
    test_goal_handler_skips_when_cancelled()
    test_goal_handler_skips_when_no_goal()
    test_install_observer_filters_achieved_goal()
    test_install_observer_threads_active_goal()
    print("\nall goal-loop tests OK")


if __name__ == "__main__":
    main()
