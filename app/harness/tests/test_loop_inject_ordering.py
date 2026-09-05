"""The observer's inject must land AFTER the assistant text it answers.

`loop.py` fires the OnEvent hook at the end of each iteration and the Inner
Voice observer's `inject` lever appends to the same `chat_messages` list the
harness is building. Through 2026-09-04 the hook fired BEFORE the iteration's
assistant turn was appended, so history read:

    user:      "[INNER VOICE] You ended the turn by announcing an action…"
    assistant: "Let me check the logs:"        <- what the inject is about

The nudge preceded its referent, and the next request the model generated
from ended on its own assistant turn rather than on a user message. That is
the stall-rescue path — the dominant failure Inner Voice exists to catch — so
it was the common case, not an edge case.

These tests drive the real `run_query` with a scripted vLLM stream and assert
on the message list the model actually receives on the following iteration.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.harness import tool_search_cache
from app.harness.hooks import HookRegistry
from app.harness.loop import run_query
from app.harness.options import RunOptions


class _FakePool:
    def __init__(self) -> None:
        self._discovered = [("lloyd-mcp", [{
            "name": "Bash",
            "description": "shell",
            "inputSchema": {"type": "object", "properties": {}},
        }])]

    @property
    def discovered(self):
        return self._discovered

    async def call_tool(self, name: str, args: dict, *, session_id: str = "", **_kw):
        return {"content": f"FAKE_RESULT[{name}]", "is_error": False}


class _StreamScript:
    """Replays scripted vLLM responses and captures the messages sent."""

    def __init__(self, turns: list[tuple[str, list[dict[str, Any]]]]):
        self.turns = turns
        self.captured_messages: list[list[dict]] = []

    def __call__(self, **kwargs):
        idx = len(self.captured_messages)
        self.captured_messages.append(
            [dict(m) for m in (kwargs.get("messages") or [])]
        )
        text, tool_calls = self.turns[idx]
        return self._gen(text, tool_calls)

    async def _gen(self, text: str, tool_calls: list[dict]):
        if text:
            yield {"choices": [{"delta": {"content": text}}]}
        for i, tc in enumerate(tool_calls):
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments") or {}),
                },
            }]}}]}
        yield {"choices": [{"delta": {},
                            "finish_reason": "tool_calls" if tool_calls else "stop"}]}
        yield {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


@pytest.fixture(autouse=True)
def _reset_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


def _patch_pool(monkeypatch):
    pool = _FakePool()

    async def _build_pool(_options):
        return pool
    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)


def _inject_once_hook(chat_messages: list, text: str):
    """An OnEvent hook that injects on the first terminal assistant_message."""
    state = {"fired": False}

    async def cb(evt: dict) -> None:
        if evt.get("type") != "assistant_message":
            return
        if evt.get("tool_calls"):
            return
        if state["fired"]:
            return
        state["fired"] = True
        chat_messages.append({"role": "user", "content": text})

    hooks = HookRegistry()
    hooks.add_on_event(cb)
    return hooks


async def _drain(messages, options):
    return [evt async for evt in run_query(messages, options)]


def test_terminal_inject_lands_after_the_assistant_text(monkeypatch):
    """The inject must follow the iteration it is reacting to, not precede it."""
    _patch_pool(monkeypatch)
    script = _StreamScript([
        ("Let me check the logs:", []),      # terminal stall — hook injects
        ("The logs are clean.", []),         # continues, then ends
    ])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    chat_messages = [{"role": "user", "content": "check the logs"}]
    hooks = _inject_once_hook(chat_messages, "[INNER VOICE] Do it now.")
    opts = RunOptions(
        model="primary", session_id="inject-order", hooks=hooks,
        tool_search_enabled=False,
        # Same wiring as production: the harness reads from the very list
        # the observer mutates (see attach_observer_for_turn).
        chat_messages_handle=chat_messages,
    )
    asyncio.run(_drain(chat_messages, opts))

    # Two requests: the original, then the continuation carrying the inject.
    assert len(script.captured_messages) == 2, script.captured_messages
    second = script.captured_messages[1]

    roles = [m.get("role") for m in second]
    contents = [str(m.get("content") or "") for m in second]

    assert "Let me check the logs:" in contents, contents
    assert "[INNER VOICE] Do it now." in contents, contents
    asst_at = contents.index("Let me check the logs:")
    inject_at = contents.index("[INNER VOICE] Do it now.")
    assert asst_at < inject_at, (
        f"inject at {inject_at} must follow the assistant turn at {asst_at}: {contents}"
    )
    # And the request the model generates from ends on a user message.
    assert roles[-1] == "user", roles


def test_inject_still_continues_the_terminal_iteration(monkeypatch):
    """Reordering must not break continue-on-inject — the loop's whole point."""
    _patch_pool(monkeypatch)
    script = _StreamScript([
        ("Let me check the logs:", []),
        ("The logs are clean.", []),
    ])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    chat_messages = [{"role": "user", "content": "check the logs"}]
    hooks = _inject_once_hook(chat_messages, "[INNER VOICE] Do it now.")
    opts = RunOptions(
        model="primary", session_id="inject-continue", hooks=hooks,
        tool_search_enabled=False,
        chat_messages_handle=chat_messages,
    )
    events = asyncio.run(_drain(chat_messages, opts))

    assert len(script.captured_messages) == 2, "the inject must extend the turn"
    result = [e for e in events if e["type"] == "result"][-1]
    assert result["stop_reason"] == "stop"
    # The final answer came from the continued iteration.
    assert "The logs are clean." in result["response_text"]


def test_no_inject_ends_the_turn_on_one_iteration(monkeypatch):
    """Control: with no hook appending anything, a terminal iteration ends."""
    _patch_pool(monkeypatch)
    script = _StreamScript([("All clear.", [])])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    opts = RunOptions(
        model="primary", session_id="inject-none", hooks=HookRegistry(),
        tool_search_enabled=False,
    )
    asyncio.run(_drain([{"role": "user", "content": "hi"}], opts))
    assert len(script.captured_messages) == 1
