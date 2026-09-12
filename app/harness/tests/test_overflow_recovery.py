"""Context-overflow recovery: the path that had no test at all.

vLLM rejects a prompt over the window with a 400 the client raises as
`ContextOverflowError`. The loop's answer used to be one destructive rung —
truncate the largest tool results to free 80k chars — whose notice told the
model to "re-run the call with a narrower query", which at the wall is
advice the turn has no room to take. On 2026-09-11 it freed **0 chars** on
round 875 and the turn died.

It now runs the whole relief ladder, anchored on the size the engine itself
reported, and aims below the compaction wall.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.harness import tool_search_cache
from app.harness.context_meter import ContextMeter
from app.harness.errors import ContextOverflowError
from app.harness.loop import run_query
from app.harness.options import RunOptions


class _FakePool:
    @property
    def discovered(self):
        return [("lloyd-mcp", [{
            "name": "Bash", "description": "shell",
            "inputSchema": {"type": "object", "properties": {}},
        }])]

    async def call_tool(self, name, args, *, session_id: str = "", **_kw):
        return {"content": "R" * 100, "is_error": False}


class _OverflowThenOk:
    """Raises ContextOverflowError on the first N requests, then succeeds."""

    def __init__(self, overflows: int, requested: int = 300_000):
        self.overflows = overflows
        self.requested = requested
        self.calls = 0
        self.captured: list[list[dict]] = []

    def __call__(self, **kwargs):
        self.calls += 1
        self.captured.append([dict(m) for m in (kwargs.get("messages") or [])])
        return self._gen(self.calls)

    async def _gen(self, n):
        if n <= self.overflows:
            raise ContextOverflowError(
                "prompt too long", requested_input_tokens=self.requested,
            )
            yield  # pragma: no cover - unreachable, makes this a generator
        yield {"choices": [{"delta": {"content": "done"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        yield {"choices": [], "usage": {"prompt_tokens": 90_000,
                                        "completion_tokens": 5}}


@pytest.fixture(autouse=True)
def _reset_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


def _shared_with_big_results(n: int = 40, size: int = 40_000) -> list[dict]:
    msgs: list[dict] = [{"role": "user", "content": "go"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"c{i}", "type": "function",
            "function": {"name": "Bash", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": "B" * size})
    return msgs


def _run(monkeypatch, script, options, messages=None):
    pool = _FakePool()

    async def _build_pool(_o):
        return pool
    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    async def _drain():
        return [e async for e in run_query(
            messages or [{"role": "user", "content": "go"}], options)]
    return asyncio.run(_drain())


def test_recovery_frees_context_and_the_turn_completes(monkeypatch):
    shared = _shared_with_big_results()
    before_chars = sum(len(str(m.get("content") or "")) for m in shared)
    script = _OverflowThenOk(overflows=1)
    meter = ContextMeter(262_144)
    opts = RunOptions(
        model="primary", session_id="", tool_search_enabled=False,
        context_meter=meter, chat_messages_handle=shared,
        intra_turn_microcompact_keep_recent=2,
    )
    events = _run(monkeypatch, script, opts)

    result = [e for e in events if e["type"] == "result"][-1]
    assert result["stop_reason"] in ("stop", "end_turn")
    after_chars = sum(len(str(m.get("content") or "")) for m in shared)
    assert after_chars < before_chars, "recovery freed nothing"
    assert script.calls == 2      # the rejected attempt, then the retry


def test_the_recovery_event_names_the_rungs_it_ran(monkeypatch):
    """The old event said `truncated=0, freed_chars=0` and nothing else —
    which is what a rung that cannot reach the residue looks like, and is
    indistinguishable from one that was not needed.
    """
    shared = _shared_with_big_results()
    script = _OverflowThenOk(overflows=1)
    opts = RunOptions(
        model="primary", session_id="", tool_search_enabled=False,
        context_meter=ContextMeter(262_144), chat_messages_handle=shared,
        intra_turn_microcompact_keep_recent=2,
    )
    events = _run(monkeypatch, script, opts)
    raws = [e for e in events if e["type"] == "stream_raw"
            and "context_overflow_recovery" in (e.get("error") or "")]
    assert raws, "no recovery event emitted"
    err = raws[0]["error"]
    assert "rungs=" in err
    assert "freed_tokens=" in err
    assert "rungs=none" not in err


def test_recovery_is_bounded_and_then_raises(monkeypatch):
    """`max_context_overflow_recoveries` must still stop an infinite loop
    when relief cannot free enough.
    """
    shared = _shared_with_big_results(n=2, size=100)
    script = _OverflowThenOk(overflows=99)
    opts = RunOptions(
        model="primary", session_id="", tool_search_enabled=False,
        context_meter=ContextMeter(262_144), chat_messages_handle=shared,
    )
    with pytest.raises(ContextOverflowError):
        _run(monkeypatch, script, opts)
    # Two recoveries attempted, then the third rejection propagates.
    assert script.calls == 3


def test_the_recovered_attempt_is_not_charged_against_max_turns(monkeypatch):
    shared = _shared_with_big_results(n=10)
    script = _OverflowThenOk(overflows=1)
    opts = RunOptions(
        model="primary", session_id="", tool_search_enabled=False,
        max_turns=2, context_meter=ContextMeter(262_144),
        chat_messages_handle=shared, intra_turn_microcompact_keep_recent=2,
    )
    events = _run(monkeypatch, script, opts)
    result = [e for e in events if e["type"] == "result"][-1]
    # One real iteration was spent, not two: the rejected attempt was
    # rolled back with `num_turns -= 1`.
    assert result["num_turns"] == 1
    assert result["stop_reason"] in ("stop", "end_turn")


def test_the_engines_reported_size_reanchors_the_meter(monkeypatch):
    """The 400 carries the true prompt size. That is a better anchor than
    anything the meter has estimated, so relief aims at the real number.
    """
    shared = _shared_with_big_results(n=4, size=1_000)
    meter = ContextMeter(262_144)
    script = _OverflowThenOk(overflows=1, requested=271_000)
    opts = RunOptions(
        model="primary", session_id="", tool_search_enabled=False,
        context_meter=meter, chat_messages_handle=shared,
    )
    _run(monkeypatch, script, opts)
    # The meter was measured from the rejection even though no successful
    # request had reported usage before it.
    assert meter.measured
