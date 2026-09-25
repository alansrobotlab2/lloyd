"""The observer's `accumulated_text` is fed from `assistant_message` (D8).

`ObserverState.accumulated_text` is what every per-event observer prompt shows
as "PRIMARY'S RESPONSE SO FAR", and what the /goal evaluator reads when the
`result` event carries no `response_text`. It used to be fed only from
`text_delta` events — which the harness loop never fires OnEvent for (it fires
assistant_message, tool_call, tool_result and result) — so it was always empty
and every prompt said "(none yet)" however much the primary had written.

These tests install the real observer on a real `HookRegistry` and fire the
events the loop fires, with the observer's LLM call and persistence stubbed.
"""

from __future__ import annotations

import asyncio

import pytest

from app.harness.hooks import HookRegistry
from app.inner_voice import observer as obs


@pytest.fixture
def quiet(monkeypatch):
    """No LLM, no usage.db, no event log: capture the prompts instead."""
    prompts: list[str] = []

    async def _call_observer(*, user_prompt, **_kw):
        prompts.append(user_prompt)
        return obs.ObserverDecision(action="noop", reason="fine")

    async def _persist(*_a, **_kw):
        return None

    monkeypatch.setattr(obs, "_call_observer", _call_observer)
    monkeypatch.setattr(obs, "_persist", _persist)
    monkeypatch.setattr(obs._event_log, "log_event", lambda *a, **k: None)
    return prompts


def _install(hooks: HookRegistry, **kw) -> obs.ObserverState:
    return obs.install_observer(
        hooks=hooks,
        session_id="acc_text_sess",
        turn_id="acc_text_turn",
        user_request="what's 2+2?",
        chat_messages_handle=[],
        cancel_event=asyncio.Event(),
        primary_model="primary",
        **kw,
    )


def test_assistant_message_text_feeds_the_observers_accumulated_text(quiet):
    hooks = HookRegistry()
    state = _install(hooks)

    async def _run():
        # Iteration 1: text plus a tool call (the loop fires the assistant
        # message before dispatching the batch).
        await hooks.fire_on_event({
            "type": "assistant_message", "text": "Let me add those. ",
            "tool_calls": [{"id": "c1", "function": {"name": "Bash",
                                                      "arguments": "{}"}}],
            "iteration": 1,
        })
        # A delta, if any caller forwarded one, must not double-count.
        await hooks.fire_on_event({"type": "text_delta", "text": "The answer"})
        # Iteration 2: the terminal answer.
        await hooks.fire_on_event({
            "type": "assistant_message", "text": "The answer is 4.",
            "tool_calls": [], "iteration": 2,
        })

    asyncio.run(_run())

    assert state.accumulated_text == "Let me add those. The answer is 4."
    # And the terminal judgment's prompt shows it rather than "(none yet)".
    assert quiet, "the terminal assistant_message was never judged"
    assert "The answer is 4." in quiet[-1]
    assert "(none yet)" not in quiet[-1]


def test_the_goal_evaluation_sees_the_turns_text_when_result_carries_none(
        quiet, monkeypatch):
    seen: dict = {}

    async def _evaluate(**kw):
        seen.update(kw)
        return obs.GoalCompletionVerdict(achieved=True, reason="done")

    async def _persist_goal_state(*_a, **_kw):
        return {}

    monkeypatch.setattr(obs, "evaluate_goal_completion", _evaluate)
    monkeypatch.setattr(obs, "_persist_goal_state", _persist_goal_state)

    hooks = HookRegistry()
    _install(hooks, persistent_goal={"text": "report the sum", "attempts": 0})

    async def _run():
        await hooks.fire_on_event({
            "type": "assistant_message", "text": "The sum is 4.",
            "tool_calls": [], "iteration": 1,
        })
        await hooks.fire_on_event({
            "type": "result", "stop_reason": "stop", "response_text": "",
            "num_turns": 1, "usage": {},
        })

    asyncio.run(_run())

    assert seen, "the /goal evaluator never ran"
    assert seen["response_text"] == "The sum is 4."
