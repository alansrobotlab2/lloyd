"""`RunOptions.state_anchor` re-anchors out-of-conversation session state.

The todo list, plan and goal live in the session JSON and are rendered
into the system prompt ONCE, from their value at turn start. The system
prompt sits at position 0 and the loop only appends, so rewriting it
mid-turn would invalidate the entire cached prefix on every iteration —
on a 160k-token turn that is the difference between a 2s iteration and a
full re-prefill. The anchor therefore APPENDS.

Motivating case: 20260905_151355_iv5174 called TodoWrite at iteration 6
of 52 and never again. `<active_todos>` was empty at turn start, so for
46 iterations the only trace of the checklist was one tool call 130k
tokens back. The review was delivered with every item still pending.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.harness import tool_search_cache
from app.harness.loop import run_query
from app.harness.options import RunOptions


class _FakePool:
    @property
    def discovered(self):
        return [("lloyd-mcp", [{
            "name": "Bash",
            "description": "shell",
            "inputSchema": {"type": "object", "properties": {}},
        }])]

    async def call_tool(self, name: str, args: dict, *, session_id: str = "", **_kw):
        return {"content": f"FAKE_RESULT[{name}]", "is_error": False}


class _Script:
    def __init__(self, turns: list[tuple[str, list[dict[str, Any]]]]):
        self.turns = turns
        self.captured_messages: list[list[dict]] = []

    def __call__(self, **kwargs):
        idx = len(self.captured_messages)
        self.captured_messages.append(
            [dict(m) for m in (kwargs.get("messages") or [])]
        )
        return self._gen(*self.turns[idx])

    async def _gen(self, text: str, tool_calls: list[dict]):
        if text:
            yield {"choices": [{"delta": {"content": text}}]}
        for i, tc in enumerate(tool_calls):
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["name"],
                             "arguments": json.dumps(tc.get("arguments") or {})},
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


async def _drain(messages, options):
    return [evt async for evt in run_query(messages, options)]


TC = [{"id": "c1", "name": "Bash", "arguments": {}}]


def _run(monkeypatch, script, anchor, system_prompt=""):
    _patch_pool(monkeypatch)
    monkeypatch.setattr("app.harness.loop.stream_chat", script)
    messages = [{"role": "user", "content": "go"}]
    opts = RunOptions(
        model="primary", session_id="anchor-test", tool_search_enabled=False,
        state_anchor=anchor, system_prompt=system_prompt,
    )
    asyncio.run(_drain(messages, opts))
    return script.captured_messages


def test_anchor_output_is_appended_and_seen(monkeypatch):
    script = _Script([("a", TC), ("b", TC), ("done", [])])
    seen_iterations: list[int] = []

    async def anchor(iteration: int) -> list[dict[str, Any]]:
        seen_iterations.append(iteration)
        if iteration == 2:
            return [{"role": "user", "content": "<active_todos>REMINDER</active_todos>"}]
        return []

    captured = _run(monkeypatch, script, anchor)

    assert seen_iterations == [1, 2, 3], seen_iterations
    # Not present on the first request, present from the second onward.
    assert not any("REMINDER" in str(m.get("content") or "") for m in captured[0])
    assert any("REMINDER" in str(m.get("content") or "") for m in captured[1])
    assert any("REMINDER" in str(m.get("content") or "") for m in captured[2])


def test_anchor_does_not_disturb_the_cached_prefix(monkeypatch):
    """The system message must stay byte-identical across iterations.

    This is the whole reason the anchor appends instead of refreshing the
    system prompt: position 0 is the head of the KV-cache prefix.
    """
    script = _Script([("a", TC), ("b", TC), ("done", [])])

    async def anchor(iteration: int) -> list[dict[str, Any]]:
        return [{"role": "user", "content": f"anchor-{iteration}"}]

    captured = _run(monkeypatch, script, anchor, system_prompt="SYSTEM PROMPT V1")

    heads = [msgs[0] for msgs in captured]
    assert all(h.get("role") == "system" for h in heads), heads
    assert len({h["content"] for h in heads}) == 1, heads
    # Each request must extend the previous one, never rewrite it.
    for earlier, later in zip(captured, captured[1:]):
        assert later[:len(earlier)] == earlier, "history was rewritten, not appended"


def test_anchor_exception_does_not_kill_the_turn(monkeypatch):
    script = _Script([("a", TC), ("done", [])])

    async def anchor(iteration: int) -> list[dict[str, Any]]:
        raise RuntimeError("session file vanished")

    captured = _run(monkeypatch, script, anchor)
    assert len(captured) == 2, captured


def test_no_anchor_configured_is_a_no_op(monkeypatch):
    script = _Script([("a", TC), ("done", [])])
    _patch_pool(monkeypatch)
    monkeypatch.setattr("app.harness.loop.stream_chat", script)
    messages = [{"role": "user", "content": "go"}]
    opts = RunOptions(model="primary", session_id="no-anchor",
                      tool_search_enabled=False)
    asyncio.run(_drain(messages, opts))
    assert len(script.captured_messages) == 2


# --------------------------------------------------------------- the ledger (#769)
#
# Anchors are not persisted, so before #769 nothing anywhere said one had
# fired: not the session JSON, not the event log, not a log line naming which.


def _run_recorded(monkeypatch, script, anchor, *, sink=None):
    """Run a turn with a session and a turn id, capturing `harness.anchor_fired`."""
    from app import event_log

    records: list[dict[str, Any]] = []

    def _capture(session_id, event, data=None, *, turn_id=None, **_kw):
        if event == "harness.anchor_fired":
            records.append({"session_id": session_id, "turn_id": turn_id,
                            "event": event, "data": data})
        return 0

    monkeypatch.setattr(event_log, "log_event", sink or _capture)
    _patch_pool(monkeypatch)
    monkeypatch.setattr("app.harness.loop.stream_chat", script)
    opts = RunOptions(
        model="primary", session_id="ledger-sess", turn_id="turn-7",
        tool_search_enabled=False, state_anchor=anchor,
    )
    events = asyncio.run(_drain([{"role": "user", "content": "go"}], opts))
    return records, script.captured_messages, events


def test_each_appended_anchor_writes_one_record_naming_it(monkeypatch):
    from app.deadline_anchor import ANCHOR_TAG, build_iteration_anchor

    budget = build_iteration_anchor(4)       # 75% of 4 at iteration 3, 90% at 4

    async def anchor(iteration: int) -> list[dict[str, Any]]:
        out = await budget(iteration)
        if iteration == 2:
            out.append({"role": "user", "content": "<active_todos>x</active_todos>",
                        ANCHOR_TAG: {"kind": "todo", "level": 10}})
            out.append({"role": "user", "content": "a caller's own, untagged"})
        return out

    script = _Script([("a", TC), ("b", TC), ("c", TC), ("done", [])])
    records, captured, _ = _run_recorded(monkeypatch, script, anchor)

    got = [(r["data"]["anchor"], r["data"]["level"], r["data"]["iteration"])
           for r in records]
    assert got == [("todo", 10, 2), ("unnamed", None, 2),
                   ("iteration_budget", 75, 3), ("iteration_budget", 90, 4)], got
    for r in records:
        assert r["event"] == "harness.anchor_fired"
        assert r["session_id"] == "ledger-sess" == r["data"]["session_id"]
        assert r["turn_id"] == "turn-7" == r["data"]["turn_id"]
    # The tag is the harness's, never the engine's: it is gone from every request.
    assert not any(ANCHOR_TAG in m for msgs in captured for m in msgs)


def test_a_raising_sink_neither_drops_the_anchor_nor_ends_the_turn(monkeypatch):
    def _broken(*_a, **_kw):
        raise OSError("event log disk full")

    async def anchor(iteration: int) -> list[dict[str, Any]]:
        from app.deadline_anchor import ANCHOR_TAG
        if iteration == 1:
            return [{"role": "user", "content": "<budget>REMINDER</budget>",
                     ANCHOR_TAG: {"kind": "iteration_budget", "level": 75}}]
        return []

    script = _Script([("a", TC), ("b", TC), ("done", [])])
    _, captured, events = _run_recorded(monkeypatch, script, anchor, sink=_broken)

    assert len(captured) == 3, "the loop kept going past the failed record"
    assert any("REMINDER" in str(m.get("content")) for m in captured[0])
    assert events[-1]["type"] == "result"
