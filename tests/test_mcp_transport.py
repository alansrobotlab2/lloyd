"""Integration tests for the aggregator's Streamable HTTP transport.

These start the real Starlette app on a loopback port and drive it through
the harness's own MCPPool, so they exercise the wire format rather than the
in-process handlers.

The large-payload test guards a defect that cost real debugging time on the
2026-07-28 migration: Streamable HTTP's SSE framing runs through httpx2's
parser, which enforces a 1 MiB `DEFAULT_MAX_EVENT_SIZE_BYTES`, and mcp's
client builds its `EventSource(response)` with no way to raise it. Any tool
result above 1 MiB failed as "SSE stream ended without a response" — and the
real `SSEError` was swallowed into a debug log, so the symptom named the
transport rather than the cause. `json_response=True` on the server skips
the framing entirely. Nothing else in the tree would catch a regression
here: every other MCP test runs in-process, below the transport.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path

import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Module-scoped: `agent_mcp.main.starlette_app` is a module singleton whose
# Streamable HTTP session manager initializes once in the lifespan and
# cannot be re-entered, so the server is started once for the whole file.
@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def aggregator():
    """Run agent_mcp's real app on a loopback port for the test's duration."""
    import uvicorn

    from agent_mcp.main import starlette_app

    port = _free_port()
    # The lifespan must run: Streamable HTTP's session manager initializes
    # its task group there, and without it every request fails with
    # "Task group is not initialized".
    config = uvicorn.Config(
        starlette_app, host="127.0.0.1", port=port, log_level="error",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    # `server.started` flips once the socket is bound, which is BEFORE the
    # lifespan finishes — and Streamable HTTP's session manager is created
    # in the lifespan. Poll a real request instead.
    import httpx

    async with httpx.AsyncClient() as probe:
        for _ in range(400):
            await asyncio.sleep(0.05)
            try:
                r = await probe.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
                if r.status_code in (200, 503):
                    break
            except Exception:
                continue
        else:
            server.should_exit = True
            await task
            pytest.fail("aggregator did not become ready")
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await task


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def pool(aggregator):
    from app.harness.mcp_pool import MCPPool

    p = MCPPool({"lloyd-mcp": {"type": "streamable-http", "url": aggregator}})
    await p.open()
    try:
        yield p
    finally:
        await p.aclose()


@pytest.mark.asyncio(loop_scope="module")
async def test_discovery_over_the_wire(pool):
    names = {t["name"] for _s, ts in pool.discovered for t in ts}
    assert len(names) > 100
    assert {"Bash", "Read", "Write", "Edit", "Grep", "Glob"} <= names


@pytest.mark.asyncio(loop_scope="module")
async def test_tool_result_above_the_sse_event_limit(pool):
    """A result larger than httpx2's 1 MiB SSE event cap must survive.

    Written through a file so the payload is deterministic and the test
    doesn't depend on how much data the knowledge graph happens to hold.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("x" * (2 * 1024 * 1024))         # 2 MiB, comfortably over
        path = fh.name
    try:
        result = await pool.call_tool("Read", {"file_path": path})
    finally:
        os.unlink(path)
    assert result["is_error"] is False, result["content"][:200]
    assert len(result["content"]) > 1024 * 1024


@pytest.mark.asyncio(loop_scope="module")
async def test_error_result_crosses_the_wire_as_is_error(pool):
    result = await pool.call_tool("Read", {"file_path": "/definitely/not/here"})
    assert result["is_error"] is True
    assert "does not exist" in result["content"]


@pytest.mark.asyncio(loop_scope="module")
async def test_a_failed_call_does_not_take_the_pool_down(pool):
    """The 2026-07-28 stateless core removes the shared session that a
    transport-shaped failure used to tear down for every in-flight turn."""
    bad = await pool.call_tool("Bash", {"command": "exit 3"})
    assert bad["is_error"] is True
    good = await pool.call_tool("Bash", {"command": "echo still-here"})
    assert good["is_error"] is False
    assert good["content"].strip() == "still-here"
    assert pool._poisoned is False


@pytest.mark.asyncio(loop_scope="module")
async def test_session_id_travels_in_meta_not_arguments(pool):
    """Background tasks are keyed by the session the aggregator reads out
    of `_meta`; if that plumbing breaks, the drain returns nothing."""
    spawn = await pool.call_tool(
        "Bash", {"command": "echo meta-routed", "run_in_background": True},
        session_id="test-meta-session",
    )
    assert spawn["is_error"] is False

    for _ in range(60):
        await asyncio.sleep(0.1)
        drained = await pool.call_tool(
            "_BackgroundTaskDrain", {}, session_id="test-meta-session",
        )
        if "completed" in drained["content"]:
            break
    else:
        pytest.fail(f"background task never drained: {drained['content'][:300]}")

    # A different session must not see it.
    other = await pool.call_tool(
        "_BackgroundTaskDrain", {}, session_id="some-other-session",
    )
    assert "completed" not in other["content"]


@pytest.mark.asyncio(loop_scope="module")
async def test_concurrent_calls_do_not_interleave(pool):
    results = await asyncio.gather(*[
        pool.call_tool("Bash", {"command": f"echo n{i}"}, session_id=f"s{i}")
        for i in range(16)
    ])
    assert sorted(r["content"].strip() for r in results) == sorted(
        f"n{i}" for i in range(16)
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_transport_failure_reconnects_instead_of_poisoning(pool, monkeypatch):
    """A dropped connection is a reconnect, not a dead pool.

    Under the pre-2026-07-28 handshake the session was pinned to the
    server, so any transport-shaped failure poisoned this pool, evicted it
    from the process cache, and tore down the session that every
    concurrent turn was sharing — one blip cost every in-flight turn. The
    stateless core makes reconnecting cost single-digit milliseconds, so
    the call just retries.

    The failure is injected at `_invoke` rather than by closing a session
    for real: the SDK's sessions must be torn down in the task that
    entered them (that is what the pool's owner-task exists to guarantee),
    so closing one from the test's task deadlocks rather than simulating
    a drop.
    """
    real_invoke = pool._invoke
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated transport drop")
        return await real_invoke(*args, **kwargs)

    monkeypatch.setattr(pool, "_invoke", flaky)

    result = await pool.call_tool("Bash", {"command": "echo reconnected"})

    assert calls["n"] == 2, "should have retried exactly once after reconnecting"
    assert result["is_error"] is False, result["content"][:300]
    assert result["content"].strip() == "reconnected"
    assert pool._poisoned is False
    assert "Bash" in pool._tool_routes      # routes survived the reconnect


@pytest.mark.asyncio(loop_scope="module")
async def test_persistent_transport_failure_still_gives_up(pool, monkeypatch):
    """Retrying once is recovery; retrying forever would hide an outage."""
    from app.harness.errors import ToolDispatchError

    async def always_down(*args, **kwargs):
        raise ConnectionError("server is genuinely down")

    monkeypatch.setattr(pool, "_invoke", always_down)

    with pytest.raises(ToolDispatchError):
        await pool.call_tool("Bash", {"command": "echo nope"})
    assert pool._poisoned is True

    # The pool fixture is module-scoped, so leave it usable for anything
    # added after this test rather than depending on file order.
    monkeypatch.undo()
    await pool._reopen()
    assert (await pool.call_tool("Bash", {"command": "echo restored"}))["is_error"] is False
