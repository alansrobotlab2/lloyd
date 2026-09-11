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

import pytest

from app.harness import mcp_pool as M
from app.harness.errors import ToolDispatchError

SERVER = next(iter(M.DEFAULT_LLOYD_MCP_SERVERS))


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
