"""A transport failure mid-call is retried only for a call safe to send twice.

2026-09-11: `automod_gate` with the review rung runs seven to twelve minutes.
The pool built its HTTP client with the SDK default (`read=300`), so the
stream died at five minutes; the pool then "reconnected and retried once",
which started a SECOND gate of the same round while the first was still
grading. Every review-gated round that day aborted with both review attempts
spent on one commit and the model shown no finding. Two properties pin it:

  - the HTTP client's read timeout sits above `CALL_TIMEOUT_SECONDS`;
  - a tool the server does not annotate read-only or idempotent is never
    re-sent after a transport error — the failure is surfaced, saying the
    call may have run.
"""
from __future__ import annotations

import asyncio

import anyio
import pytest

from app.harness import mcp_pool as M
from app.harness.errors import ToolDispatchError

SERVER = next(iter(M.DEFAULT_LLOYD_MCP_SERVERS))


def _taskgroup_error(message: str) -> BaseException:
    """The ExceptionGroup anyio actually raises for a failing task.

    `_failing_invoke` below stands in for a transport collapse with a plain
    error that carries the group's text; this builds the real nested shape, so
    the file covers both the error that needs unwrapping and the one that does
    not.
    """

    async def build():
        try:
            async with anyio.create_task_group() as tg:
                async def boom():
                    raise ConnectionError(message)
                tg.start_soon(boom)
        except BaseException as exc:
            return exc
        raise AssertionError("the task group did not fail")

    return asyncio.run(build())


def _pool(monkeypatch, annotations: dict[str, dict]):
    pool = M.MCPPool(dict(M.DEFAULT_LLOYD_MCP_SERVERS))
    tools = [{"name": n, "description": "", "inputSchema": {"type": "object"},
              "annotations": a} for n, a in annotations.items()]
    pool._register(SERVER, tools)
    pool._opened = True
    monkeypatch.setattr(pool, "_reopen", _noop_reopen)
    return pool


async def _noop_reopen():
    return None


def _failing_invoke(calls: list, then_ok: bool):
    async def _invoke(server_name, bare, args, budget, meta):
        calls.append(bare)
        if len(calls) == 1 or not then_ok:
            raise RuntimeError("unhandled errors in a TaskGroup (1 sub-exception)")
        class R:  # a minimal CallToolResult stand-in
            content = []
            isError = False
            is_error = False
            structuredContent = None
            structured_content = None
        return R()
    return _invoke


def test_http_read_timeout_outlives_the_call_ceiling():
    assert M.HTTP_READ_TIMEOUT_SECONDS > M.CALL_TIMEOUT_SECONDS
    client = M._http_client()
    if client is None:
        pytest.skip("SDK without create_mcp_http_client")
    assert client.timeout.read >= M.CALL_TIMEOUT_SECONDS
    asyncio.run(client.aclose())


def test_a_mutating_tool_is_not_resent_after_a_transport_error(monkeypatch):
    pool = _pool(monkeypatch, {"automod_gate": {}})
    calls: list[str] = []
    monkeypatch.setattr(pool, "_invoke", _failing_invoke(calls, then_ok=True))
    with pytest.raises(ToolDispatchError) as exc:
        asyncio.run(pool.call_tool("automod_gate", {"round_id": "SM_x"}))
    assert calls == ["automod_gate"], "one attempt, never a second gate"
    assert "NOT retried" in str(exc.value) and "may have run" in str(exc.value)
    assert pool._poisoned is False, "one unretried failure does not evict the pool"


def test_the_unwrapped_cause_travels_with_the_may_have_run_warning(monkeypatch):
    """#936: unwrapping the group must not cost the caller the guidance.

    `ae63032` added "the server may have run it — read its state back" because
    a transport error says nothing about whether the side effect landed. That
    instruction is only actionable if the error also says WHAT failed, so the
    two halves belong in one message: the innermost exception from a real anyio
    group, and the retry guidance word for word as it stands today.
    """
    pool = _pool(monkeypatch, {"automod_gate": {}})
    group = _taskgroup_error("read timeout inside the handler")
    attempts: list[str] = []

    async def collapsed(server_name, bare, args, budget, meta):
        attempts.append(bare)
        raise group

    monkeypatch.setattr(pool, "_invoke", collapsed)
    with pytest.raises(ToolDispatchError) as exc:
        asyncio.run(pool.call_tool("automod_gate", {"round_id": "SM_x"}))
    assert attempts == ["automod_gate"], "unwrapping changes nothing about retry"
    message = str(exc.value)
    assert "read timeout inside the handler" in message, message
    assert "unhandled errors in a TaskGroup" not in message, message
    assert "NOT retried" in message and "may have run" in message, message
    assert "read its state back" in message, message


def test_a_read_only_tool_is_still_retried_once(monkeypatch):
    pool = _pool(monkeypatch, {"graph_status": {"readOnlyHint": True}})
    calls: list[str] = []
    monkeypatch.setattr(pool, "_invoke", _failing_invoke(calls, then_ok=True))
    out = asyncio.run(pool.call_tool("graph_status", {}))
    assert calls == ["graph_status", "graph_status"]
    assert out["is_error"] is False


def test_an_idempotent_tool_is_retried_and_an_unannotated_one_is_not(monkeypatch):
    pool = _pool(monkeypatch, {"Write": {"idempotentHint": True}, "email_send": {}})
    assert pool._retry_safe("Write") is True
    assert pool._retry_safe("email_send") is False
    assert pool._retry_safe("never_registered") is False


def test_annotations_are_dropped_with_the_routes_on_reopen():
    pool = M.MCPPool(dict(M.DEFAULT_LLOYD_MCP_SERVERS))
    pool._register(SERVER, [{"name": "x", "description": "", "inputSchema": {},
                             "annotations": {"readOnlyHint": True}}])
    assert pool._retry_safe("x")
    asyncio.run(pool.aclose())
    assert not pool._retry_safe("x")


# ── #1326: two tools whose read-only annotation would have re-sent a write ───
#
# `autonomy_config` (key+value rewrote the scheduler config file) and
# `fact_resolve` (`auto_resolve=true` marked facts invalid) were annotated
# read-only, so `_retry_safe` said "it only reads" and re-sent them after a
# transport drop — a write that had already landed, landing twice. The writes now
# live in `autonomy_config_set` and `fact_resolve_apply`: names the server
# advertises with no hint at all, so the retry rule gets them right while the two
# read halves keep the retry a read deserves.

SPLIT_WRITERS = ("autonomy_config_set", "fact_resolve_apply")
SPLIT_READERS = ("autonomy_config", "fact_resolve")


def _wire_hints(name: str) -> dict:
    """The hint set `tools/list` puts on the wire for one name.

    camelCase because that is what the pool reads out of the JSON, and read
    through `getattr` twice because the local `ToolAnnotations` is snake_case —
    the same 1.x/2.x field split `agent_mcp/main.py::_result_is_error` papers
    over for results."""
    from agent_mcp import annotations as A
    ann = A.annotations_for(name)
    return {
        "readOnlyHint": bool(getattr(ann, "readOnlyHint", False)
                             or getattr(ann, "read_only_hint", False)),
        "idempotentHint": bool(getattr(ann, "idempotentHint", False)
                               or getattr(ann, "idempotent_hint", False)),
    }


def _split_annotations() -> dict:
    """What the server actually advertises for each name, so this exercises the
    hint it sends rather than a guess about it."""
    return {n: _wire_hints(n) for n in SPLIT_WRITERS + SPLIT_READERS}


def test_the_split_writers_are_not_retry_safe_and_their_read_halves_are(monkeypatch):
    pool = _pool(monkeypatch, _split_annotations())
    for name in SPLIT_WRITERS:
        assert pool._retry_safe(name) is False, name
    for name in SPLIT_READERS:
        assert pool._retry_safe(name) is True, name


@pytest.mark.parametrize("writer", SPLIT_WRITERS)
def test_a_dropped_transport_does_not_re_fire_the_split_writer(monkeypatch, writer):
    """One attempt for a write, and the caller keeps the guidance that says the
    server may have run it anyway."""
    pool = _pool(monkeypatch, _split_annotations())
    calls: list[str] = []
    monkeypatch.setattr(pool, "_invoke", _failing_invoke(calls, then_ok=False))
    with pytest.raises(ToolDispatchError) as exc:
        asyncio.run(pool.call_tool(writer, {}))
    assert calls == [writer], f"{writer} was re-sent after a transport drop: {calls}"
    assert "may have run" in str(exc.value), str(exc.value)


@pytest.mark.parametrize("reader", SPLIT_READERS)
def test_the_split_read_halves_keep_their_retry(monkeypatch, reader):
    """The positive control: the split moved the write, it did not mute the read.
    Without a second attempt a dropped stream turns a working read into an
    outage."""
    pool = _pool(monkeypatch, _split_annotations())
    calls: list[str] = []
    monkeypatch.setattr(pool, "_invoke", _failing_invoke(calls, then_ok=True))
    out = asyncio.run(pool.call_tool(reader, {}))
    assert calls == [reader, reader], f"{reader} lost its retry: {calls}"
    assert out["is_error"] is False
