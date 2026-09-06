"""`Task` closes its registry row with the status that actually happened.

The registry row is opened before the run loop so an in-flight subagent
is visible while it is in flight. Closing it correctly is the subtle
part: `finish` is idempotent and first-writer-wins, so a blanket
`finally: finish(status="cancelled")` — the obvious way to guarantee the
row always closes — runs BEFORE the success path and stamps every
completed run "cancelled", making the real status a silent no-op. The
dashboard would then show a fleet of cancelled subagents that all
actually succeeded.

These tests pin the status each exit path produces.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_mcp import _subagent_registry as reg
from agent_mcp import builtin_task


@pytest.fixture(autouse=True)
def _clean():
    reg.reset()
    yield
    reg.reset()


def _events(*, iterations, stop_reason="stop", num_turns=None):
    evts = []
    for text, tools in iterations:
        for name in tools:
            evts.append({"type": "tool_call", "id": "x", "name": name, "input": {}})
        evts.append({
            "type": "assistant_message",
            "text": text,
            "thinking": "",
            "tool_calls": [{"function": {"name": n}} for n in tools],
        })
    evts.append({
        "type": "result",
        "stop_reason": stop_reason,
        "num_turns": num_turns if num_turns is not None else len(iterations),
        "usage": {},
    })
    return evts


def _run(monkeypatch, evts=None, *, raises=None, max_turns=20):
    async def _fake_run_query(messages, options):
        if raises is not None:
            raise raises
        for e in evts or []:
            yield e
        # An async generator needs a yield on every path to stay a
        # generator; the raise above happens before the first one.
        return

    import app.harness.loop as loop_mod
    monkeypatch.setattr(loop_mod, "run_query", _fake_run_query)
    monkeypatch.setattr(
        builtin_task, "_load_subagent_profile",
        lambda t: {"system_prompt": "", "max_turns": max_turns,
                   "disallowed_tools": [], "model": "primary", "base_url": ""},
    )
    out = asyncio.run(builtin_task._task(
        {"prompt": "review the thing", "description": "probe"}
    ))
    return json.loads(out)


def test_a_successful_run_is_recorded_as_completed(monkeypatch):
    """The regression guard. A `finally`-based close reports "cancelled"."""
    res = _run(monkeypatch, _events(iterations=[
        ("", ["Read"]),
        ("The answer.", []),
    ]))
    assert res["response"] == "The answer."

    assert reg.list_active() == []
    row = reg.list_recent()[0]
    assert row["status"] == "completed"
    assert row["stop_reason"] == "stop"
    assert row["response_chars"] == len("The answer.")


def test_the_row_carries_the_tools_the_subagent_actually_ran(monkeypatch):
    _run(monkeypatch, _events(iterations=[
        ("", ["Grep"]),
        ("", ["Read"]),
        ("", ["Grep"]),
        ("done", []),
    ]))
    row = reg.list_recent()[0]
    assert row["tool_counts"] == {"Grep": 2, "Read": 1}
    assert row["tool_call_count"] == 3
    # One row per assistant_message, terminal iteration included.
    assert row["turns"] == 4


def test_a_run_that_never_answered_is_recorded_as_failed(monkeypatch):
    """Matches the error the caller gets — the dashboard must not show
    this as a success."""
    res = _run(monkeypatch, _events(
        iterations=[("I'll start by getting oriented.", ["Read"])] * 19,
        stop_reason="max_turns", num_turns=21,
    ), max_turns=20)
    assert "error" in res

    row = reg.list_recent()[0]
    assert row["status"] == "failed"
    assert row["stop_reason"] == "max_turns"
    assert "20-turn budget" in row["error"]


def test_a_crashed_run_is_recorded_as_error(monkeypatch):
    res = _run(monkeypatch, raises=RuntimeError("engine went away"))
    assert "Subagent failed" in res["error"]

    row = reg.list_recent()[0]
    assert row["status"] == "error"
    assert "engine went away" in row["error"]
    assert reg.list_active() == []


def test_no_row_is_left_active_after_any_exit_path(monkeypatch):
    """A row that never closes is a phantom subagent on the dashboard,
    forever."""
    _run(monkeypatch, _events(iterations=[("ok", [])]))
    _run(monkeypatch, raises=RuntimeError("boom"))
    _run(monkeypatch, _events(
        iterations=[("preamble", ["Read"])], stop_reason="max_turns", num_turns=21,
    ), max_turns=20)

    assert reg.list_active() == []
    assert [r["status"] for r in reg.list_recent()] == ["failed", "error", "completed"]
