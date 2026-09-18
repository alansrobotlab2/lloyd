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


#: The credential every credentialed request in this file carries (#1053).
#: Supplied as an env value rather than through the token file so both halves of
#: the seam — the server booted here and the `MCPPool` that connects to it —
#: agree without reading `~/.local/state/lloyd/aggregator-token`. A client never
#: mints, so on a checkout where that file does not exist the pool would connect
#: with no credential, the server would mint its own on the first request, and
#: every leg below would be refused by its own plumbing.
TEST_TOKEN = "wire-test-credential-0123456789abcdef"


# Module-scoped: `agent_mcp.main.starlette_app` is a module singleton whose
# Streamable HTTP session manager initializes once in the lifespan and
# cannot be re-entered, so the server is started once for the whole file.
@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def aggregator():
    """Run agent_mcp's real app on a loopback port for the test's duration."""
    import uvicorn
    from pytest import MonkeyPatch

    import agent_mcp.aggregator_auth as A
    from agent_mcp.main import starlette_app

    # Set before the server boots and undone only after it exits: the guard
    # resolves the credential per request, so the value need only exist by the
    # first one, and a token that vanished mid-file would refuse the legs that
    # run after it.
    monkey = MonkeyPatch()
    monkey.setenv(A.TOKEN_ENV, TEST_TOKEN)
    monkey.delenv(A.TOKEN_FILE_ENV, raising=False)
    A.reset_for_tests()

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
        monkey.undo()
        A.reset_for_tests()


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
    """The aggregator advertises its tool surface over a real transport.

    The floor deliberately EXCLUDES thunderbird. That module contributes 40 of
    the ~124 live tools, and `thunderbird.list_tools()` degrades to `[]`
    whenever the mail client isn't running — so a flat `> 100` made this test,
    and therefore the self-modification gate that runs the whole suite, depend
    on whether the user happened to have Thunderbird open. Everything that is
    not an external-app bridge is counted instead.
    """
    from agent_mcp import main as M

    external = {"thunderbird"}
    internal_expected = sum(
        entry.get("tools", 0)
        for name, entry in M._discovery_status.items()
        if name not in external
    )
    names = {t["name"] for _s, ts in pool.discovered for t in ts}
    assert internal_expected >= 80, M._discovery_status
    assert len(names) >= internal_expected
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
    bad = await pool.call_tool("Bash", {"command": "exit 3"},
                               session_id="transport-probe")
    assert bad["is_error"] is True
    good = await pool.call_tool("Bash", {"command": "echo still-here"},
                                session_id="transport-probe")
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

    Since 2026-09-11 the retry is only for a call the server annotates
    read-only or idempotent — see the next test for why `Bash` is not one.
    """
    real_invoke = pool._invoke
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated transport drop")
        return await real_invoke(*args, **kwargs)

    monkeypatch.setattr(pool, "_invoke", flaky)

    result = await pool.call_tool("_BackgroundTaskDrain", {}, session_id="reconnect-probe")

    assert calls["n"] == 2, "should have retried exactly once after reconnecting"
    assert result["is_error"] is False, result["content"][:300]
    assert pool._poisoned is False
    assert "Bash" in pool._tool_routes      # routes survived the reconnect


@pytest.mark.asyncio(loop_scope="module")
async def test_a_mutating_call_is_surfaced_not_resent_after_a_transport_drop(pool, monkeypatch):
    """A read timeout fires while the server is still working, so for a
    mutating call "retry" means "run it twice". On 2026-09-11 that turned
    one `automod_gate` into two concurrent gates of the same round. The
    server's own annotations decide: `Bash` carries no readOnly/idempotent
    hint, so the drop is a tool error that says the call may have run."""
    from app.harness.errors import ToolDispatchError
    real_invoke = pool._invoke
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("simulated transport drop")
        return await real_invoke(*args, **kwargs)

    monkeypatch.setattr(pool, "_invoke", flaky)
    with pytest.raises(ToolDispatchError) as exc:
        await pool.call_tool("Bash", {"command": "echo twice?"},
                             session_id="transport-probe")
    assert calls["n"] == 1
    assert "NOT retried" in str(exc.value) and "may have run" in str(exc.value)
    assert pool._poisoned is False
    monkeypatch.undo()
    got = await pool.call_tool("Bash", {"command": "echo fine"},
                               session_id="transport-probe")
    assert got["content"].strip() == "fine"


@pytest.mark.asyncio(loop_scope="module")
async def test_persistent_transport_failure_still_gives_up(pool, monkeypatch):
    """Retrying once is recovery; retrying forever would hide an outage."""
    from app.harness.errors import ToolDispatchError

    async def always_down(*args, **kwargs):
        raise ConnectionError("server is genuinely down")

    monkeypatch.setattr(pool, "_invoke", always_down)

    with pytest.raises(ToolDispatchError):
        await pool.call_tool("_BackgroundTaskDrain", {}, session_id="down-probe")
    assert pool._poisoned is True

    # The pool fixture is module-scoped, so leave it usable for anything
    # added after this test rather than depending on file order.
    monkeypatch.undo()
    await pool._reopen()
    back = await pool.call_tool("Bash", {"command": "echo restored"},
                                session_id="transport-probe")
    assert back["is_error"] is False


@pytest.mark.asyncio(loop_scope="module")
async def test_cancelled_open_does_not_wedge_the_pool_cache(aggregator, monkeypatch):
    """A cancelled first open must not hold the process-wide cache lock.

    `run_query` is an async generator, and `autonomy.run_task` iterates it
    under `asyncio.timeout`. When that timeout fires the generator is
    abandoned while suspended — its `async with` blocks never unwind. The
    cache lock used to be held across `await pool.open()`, so a single
    autonomy task timing out during a first open deadlocked MCP for the
    whole process: every later turn blocked forever, with no error in any
    log and /health still green.
    """
    from app.harness import mcp_pool as MP

    cfg = {"lloyd-mcp": {"type": "streamable-http", "url": aggregator}}
    key = MP._config_key(cfg)
    MP._POOL_CACHE.pop(key, None)

    real_open = MP.MCPPool.open
    started = asyncio.Event()

    async def hanging_open(self):
        started.set()
        await asyncio.sleep(3600)          # stand in for a slow first open

    monkeypatch.setattr(MP.MCPPool, "open", hanging_open)
    victim = asyncio.create_task(MP.get_or_open_pool(cfg))
    await asyncio.wait_for(started.wait(), timeout=5)
    victim.cancel()
    with pytest.raises(asyncio.CancelledError):
        await victim

    # The lock must be free and the half-open pool must not be cached.
    assert not MP._POOL_CACHE_LOCK.locked()
    assert key not in MP._POOL_CACHE

    # And the very next caller succeeds.
    monkeypatch.setattr(MP.MCPPool, "open", real_open)
    pool = await asyncio.wait_for(MP.get_or_open_pool(cfg), timeout=20)
    result = await pool.call_tool("Bash", {"command": "echo recovered"},
                                  session_id="transport-probe")
    assert result["content"].strip() == "recovered"
    await pool.aclose()
    MP._POOL_CACHE.pop(key, None)


# ── #1053: the credential is enforced over the wire, on every route ──────────
#
# The legs above all arrive *with* a credential, because `MCPPool` is the caller
# the credential is issued to. These arrive without one — the request the item
# reproduced, an unauthenticated POST from a script on this box — and they are
# checked over a real socket rather than through `httpx.ASGITransport` for one
# reason: `httpx.ASGITransport` synthesises the Host header from the URL, so
# through it it is not decidable which of the two guards refused. The middleware
# is installed in `agent_mcp.main` (`starlette_app =
# aggregator_auth.require_credential(starlette_app)`), so what is booted here is
# the shipped app, and the refusal below is the same code path a live
# out-of-band request takes.
#
# The pre-fix answers, measured on the live aggregator before the control
# existed: 200 with 152 tools on `tools/list`, 200 and real board data on
# `tools/call backlog_boards {}`, 200 `{"results": []}` on `POST /changes/revert`
# for a session the caller named, and 200 on `GET /state`.

_WIRE_MCP = {"Accept": "application/json, text/event-stream"}


def _base(aggregator: str) -> str:
    """The aggregator origin, from the `/mcp` URL the fixture yields."""
    return aggregator.rsplit("/mcp", 1)[0]


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("body", [
    {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
     "params": {"name": "backlog_boards", "arguments": {}}},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
     "params": {"name": "vault_write",
                "arguments": {"path": "knowledge/pinned.md",
                              "content": "written by a script"}}},
])
async def test_the_reproduced_unauthenticated_request_is_refused_over_the_wire(
        aggregator, body):
    """`tools/list`, a read call and a well-formed write call, all refused.

    The write leg is the one that matters: pre-fix it returned the tool's own
    `MISSING_PARAM` validator, i.e. the request crossed the transport, crossed
    `call_tool` and reached `mod.call_tool` (`agent_mcp/main.py`) with none of
    the harness gates applied.
    """
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(aggregator, json=body, headers=_WIRE_MCP)
    assert r.status_code == 401, f"{body['method']} -> {r.status_code} {r.text[:200]}"
    refusal = r.json()
    assert refusal["code"] == "AGGREGATOR_UNAUTHENTICATED", r.text[:200]
    # The MCP legs get the explaining deny, not the bare transport body: the
    # caller here is a script, and the message is the only thing it gets back.
    assert "X-Lloyd-Aggregator-Token" in refusal["error"], r.text[:200]
    assert "#1053" in refusal["error"], r.text[:200]


@pytest.mark.asyncio(loop_scope="module")
async def test_the_custom_routes_refuse_at_the_transport_not_in_the_handler(
        aggregator):
    """`/state`, `/changes/revert` and `/browser/navigate` on a raw POST/GET.

    `/changes/revert` used to answer 200 for a session id the caller invented —
    the handler trusts its body, so only a layer above the router can refuse it.
    A 400 from a handler's own field validation would not be a control either;
    these assert 401 with the refusal body.
    """
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        checks = [
            ("GET", "/state", None),
            ("GET", "/changes?session=s&turn=t", None),
            ("POST", "/changes/revert",
             {"session": "20260918_someone_elses_session", "turn": "1",
              "paths": ["/home/alansrobotlab/obsidian/SEED.md"]}),
            ("POST", "/browser/navigate", {"url": "http://example.com"}),
        ]
        for method, path, payload in checks:
            r = await client.request(method, _base(aggregator) + path,
                                     json=payload)
            assert r.status_code == 401, (
                f"{method} {path} -> {r.status_code} {r.text[:160]}")
            assert "AGGREGATOR_UNAUTHENTICATED" in r.text, f"{path}: {r.text[:160]}"


@pytest.mark.asyncio(loop_scope="module")
async def test_health_is_the_one_route_a_credentialless_probe_still_uses(aggregator):
    """`GET /health` stays 200 with no credential, and its neighbour does not.

    `/state` is one path away and carries live data, so it is the
    counter-assertion that the open set is exactly one route wide.
    """
    import httpx

    import agent_mcp.aggregator_auth as A

    async with httpx.AsyncClient(timeout=10.0) as client:
        ok = await client.get(_base(aggregator) + "/health")
        closed = await client.get(_base(aggregator) + "/state")
    assert ok.status_code == 200, ok.text[:200]
    assert ok.json()["tools"] > 0
    assert closed.status_code == 401, closed.text[:200]
    assert A.OPEN_PATHS == frozenset({"/health"}), (
        f"the unauthenticated surface is wider than liveness: {sorted(A.OPEN_PATHS)}")


@pytest.mark.asyncio(loop_scope="module")
async def test_a_refused_tools_call_is_recorded_as_never_reaching_the_handler(
        aggregator, monkeypatch):
    """Clause 1's sequencing claim, where a handler call is actually observable.

    `tests/test_aggregator_auth.py` asserts the refusal's shape through
    `httpx.ASGITransport` with no lifespan entered, and in that shape Streamable
    HTTP's session manager raises `RuntimeError: Task group is not initialized`
    (`mcp/server/streamable_http_manager.py:170-171`) before `mod.call_tool` — so a
    recorder there reads empty whether or not the guard exists and cannot carry
    this clause. Over this socket the session manager is live, so the same
    recorder is a real witness, and it is read on both sides of one header:

    * no credential → 401, and the recorder is still **empty**;
    * the credential → 200, and the recorder holds **both** calls.

    The second half is what makes the first half mean anything: it shows the
    identical bodies reaching `mod.call_tool` the moment the credential is
    present, so the empty recorder above is a refusal and not an artefact of the
    test rig. The write leg carries a session id because clause 4
    (`agent_mcp/main.py` `call_tool`) separately refuses a sessionless
    state-changing call — this test is about the transport credential only.
    """
    import httpx

    import agent_mcp.aggregator_auth as A
    import agent_mcp.main as M

    calls: list[tuple[str, dict]] = []

    class Recorder:
        async def call_tool(self, name, arguments):
            calls.append((name, dict(arguments)))
            from agent_mcp.main import TextContent
            return [TextContent(type="text", text="THE MODULE HANDLER RAN")]

    rec = Recorder()
    table = dict(M._dispatch)
    table["backlog_boards"] = rec          # read-class, returned live board data pre-fix
    table["vault_write"] = rec             # write-class, the tool that wiped the vault
    monkeypatch.setattr(M, "_dispatch", table)

    legs = [
        ("backlog_boards", {}, None),
        ("vault_write", {"path": "knowledge/pinned.md",
                         "content": "written by a script"},
         {"lloyd/session_id": "credential-sequencing-leg"}),
    ]
    credentialed = dict(_WIRE_MCP, **{A.AUTH_HEADER: TEST_TOKEN})
    async with httpx.AsyncClient(timeout=15.0) as client:
        for tool, args, meta in legs:
            params = {"name": tool, "arguments": args}
            if meta:
                params["_meta"] = meta
            body = {"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                    "params": params}
            r = await client.post(aggregator, json=body, headers=_WIRE_MCP)
            assert r.status_code == 401, f"{tool} -> {r.status_code} {r.text[:200]}"
        assert calls == [], (
            f"a credentialless call reached the module handler: {calls}")

        for i, (tool, args, meta) in enumerate(legs):
            params = {"name": tool, "arguments": args}
            if meta:
                params["_meta"] = meta
            body = {"jsonrpc": "2.0", "id": 12 + i, "method": "tools/call",
                    "params": params}
            r = await client.post(aggregator, json=body, headers=credentialed)
            assert r.status_code == 200, f"{tool} -> {r.status_code} {r.text[:200]}"
            assert "THE MODULE HANDLER RAN" in r.text, r.text[:200]
    assert calls == [(tool, args) for tool, args, _ in legs], (
        f"the credential did not carry both calls through to the handler: {calls}")


@pytest.mark.asyncio(loop_scope="module")
async def test_the_persistent_session_opener_carries_the_credential(
        aggregator, tmp_path, monkeypatch):
    """The pool's *persistent* session is the other client seam, and nothing covered it.

    Two client paths reach the aggregator and only one of them was exercised: the
    per-call `_http_session` (post + sse), which the tests above drive, and the
    session `_open_session` opens and the pool's owner task holds — the path
    `agent_mcp/main.py:894-901` keeps the stdio legacy bridge alive for. Its httpx
    client is built by `streamable_http_client` → `_http_client`, which takes the
    credential from `aggregator_headers(cfg["url"])` once, at session open
    (`app/harness/mcp_pool.py:665`). Measured on this branch before this node
    existed: replacing that call with `headers=None` left 57 tests across
    `tests/test_mcp_transport.py` + `tests/test_mcp_layer.py` green — the accept
    side refused every call while the suite reported nothing. This node is what
    turned that mutation into a failure, and re-running it now fails here and only
    here.

    Same discriminating pair as the test above, one level further in. With the
    server's own token file as the source, the session completes `initialize` and
    answers `tools/list`; pointed at an empty token file — the `_publish=False`
    client shape, so nothing mints a value the server never agreed to — it never
    completes the handshake, and the failure surfaces as `mcp.MCPError: Server
    returned an error response`, which is the 401 deny arriving inside the MCP
    client rather than as an httpx status.
    """
    from contextlib import AsyncExitStack

    import mcp

    import agent_mcp.aggregator_auth as A
    import app.harness.mcp_pool as MP

    cfg = {"type": "streamable-http", "url": aggregator}
    pool = MP.MCPPool({"lloyd-mcp": cfg})

    # Positive leg: the env credential the fixture started the server with.
    monkeypatch.delenv(A.TOKEN_FILE_ENV, raising=False)
    monkeypatch.setenv(A.TOKEN_ENV, TEST_TOKEN)
    async with AsyncExitStack() as stack:
        session = await pool._open_session(stack, "lloyd-mcp", cfg)
        listed = await session.list_tools()
    assert len(listed.tools) > 0, (
        "the persistent session opened but advertised nothing — the credential "
        "it sends is not the one the guard accepts")

    # Negative leg: nothing to send. The env value goes as well as the file, or
    # the client falls back to this box's real credential and would be accepted.
    # Asserted as "the failure arrives inside the MCP layer", not as an HTTP
    # status. Measured: this client renders the guard's 401 body as
    # `MCPError(code=-32603, message='Server returned an error response',
    # data=None)`, two TaskGroup frames deep, with no `response` attribute anywhere
    # on the chain — so neither the status nor the guard's
    # `AGGREGATOR_UNAUTHENTICATED` code is readable from this side of the wire.
    # Both are pinned where the body is readable, by
    # `test_a_state_changing_write_with_no_session_id_is_refused_over_the_wire`.
    monkeypatch.delenv(A.TOKEN_ENV, raising=False)
    monkeypatch.setenv(A.TOKEN_FILE_ENV, str(tmp_path / "no-aggregator-token"))
    with pytest.raises(BaseException) as exc:
        async with AsyncExitStack() as stack:
            session = await pool._open_session(stack, "lloyd-mcp", cfg)
            await session.list_tools()
    # Walk the TaskGroup frames to the leaves. `flat` is the worklist and is
    # drained by the loop, so the verdict is read off `seen`.
    flat, seen = [exc.value], []
    while flat:
        cur = flat.pop()
        if any(cur is x for x in seen):
            continue
        seen.append(cur)
        flat.extend(getattr(cur, "exceptions", None) or [])
        cause = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        if cause is not None:
            flat.append(cause)
    assert any(isinstance(e, mcp.MCPError) for e in seen), (
        f"a credentialless persistent session was not refused at the MCP layer: "
        f"{type(exc.value).__name__}: {exc.value}")


@pytest.mark.asyncio(loop_scope="module")
async def test_the_dashboard_state_read_carries_the_credential(aggregator, monkeypatch):
    """`app.routers.dashboard._agent_state()` against the guarded server.

    The fourth legitimate caller, and the one whose credential had no test: the
    Mission Control dashboard reads the aggregator's `/state` route on every
    refresh, and dropping `headers=auth_headers_for(_MCP_STATE_URL)` from
    `app/routers/dashboard.py:177-178` was caught by nothing: the broadest
    selection that names it (`pytest tests/ -k "agent_state or dashboard or
    state"`, measured on this branch before this node existed) ran 266 tests and
    all 266 passed. A regression there would ship as a dashboard whose agents
    panel reads `error` forever with nothing naming the credential — and it now
    fails here, with the mutation raising `httpx.HTTPStatusError: 401` out of
    `_agent_state`'s own `raise_for_status()`.

    Only the derived URL is redirected to the test server; the function itself
    runs, so the header it now sends is this branch's code, and the server's
    refusal is the real middleware's.
    """
    import httpx

    import agent_mcp.aggregator_auth as A
    import app.aggregator_config as AC
    import app.routers.dashboard as DB

    origin = aggregator.rsplit("/mcp", 1)[0]
    monkeypatch.delenv(A.TOKEN_FILE_ENV, raising=False)
    monkeypatch.setenv(A.TOKEN_ENV, TEST_TOKEN)
    monkeypatch.setattr(DB, "_MCP_STATE_URL", f"{origin}/state")
    state = await DB._agent_state()
    assert isinstance(state, dict) and "subagents" in state, (
        f"the dashboard's credentialed state read did not land: {state!r:.200}")

    async def nothing(_url: str) -> dict[str, str]:
        return {}

    monkeypatch.setattr(AC, "auth_headers_for", nothing)
    async with httpx.AsyncClient() as probe:
        bare = await probe.get(DB._MCP_STATE_URL)
    assert bare.status_code == 401, (
        f"the state route served {bare.status_code} with no credential, so the "
        f"positive leg above proves nothing")


@pytest.mark.asyncio(loop_scope="module")
async def test_the_three_liveness_probes_that_hold_no_credential_still_work(aggregator):
    """Each real probe's own URL, derived from its own source, answered over the wire.

    The three are the callers that cannot hold a credential, and each is a
    different kind of caller, which is why naming them is not decoration:

    * `scripts/automod/promote.py:MCP_HEALTH` (:61) is a **loopback literal**, and
      `restart_stack` polls it through `_wait_health` (:1073) after restarting
      `lloyd-mcp` — a 200 is the whole requirement there, and a 401 spends the
      health budget and then fails the *promotion*, which reads as a wedged
      candidate rather than as an auth change.
    * `agent-services/guardian/policy.py:MCP_HEALTH_URL` (:84) is a second
      detached literal, in the snapshot the guardian deliberately keeps free of
      `app` imports, so it cannot use `app.aggregator_config` and will not follow
      a path rename. Pinning it to `OPEN_PATHS` is the only link between the two.
    * `app/routers/health.py:150-151` derives its probe URL by string surgery on
      the config value (`mcp_url.rsplit("/mcp", 1)[0] + "/health"`), and this is
      Mission Control's own view of the aggregator: a refusal there reads to a
      human as "MCP is down".

    `_wait_health` is called rather than re-implemented — it is the function that
    has to survive the change, and its 3 s `_get` timeout is what a redirect or a
    slow `/health` would break first.
    """
    import importlib.util
    from urllib.parse import urlsplit

    import httpx

    import agent_mcp.aggregator_auth as A
    from app.config import service_url
    from scripts.automod import promote as promote_mod

    def _rebase(url: str) -> str:
        """Point a loopback URL at the booted server, keeping its path."""
        return _base(aggregator) + urlsplit(url).path

    # 1. promote.py: its real poller, against the real server, on a budget short
    #    enough that a refusal fails this assertion in seconds instead of spending
    #    the minutes the real restart budget allows. It is a blocking urllib poll
    #    loop and the server under test answers *in this event loop*, so it runs in
    #    a thread: called directly it starves the server it is waiting on and
    #    returns False whatever the guard does.
    health_ok = await asyncio.to_thread(
        promote_mod._wait_health, _rebase(promote_mod.MCP_HEALTH), 5.0)
    assert health_ok, (
        "`promote._wait_health` could not get a 200 off /health, so a promotion "
        "would time out waiting for an aggregator that is actually up")

    # 2. The guardian's snapshot literal, loaded from its own file.
    spec = importlib.util.spec_from_file_location(
        "_guardian_policy_for_probe", ROOT / "agent-services" / "guardian" / "policy.py")
    guardian = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guardian)
    assert urlsplit(guardian.MCP_HEALTH_URL).path in A.OPEN_PATHS, (
        f"the guardian probes {guardian.MCP_HEALTH_URL}, which is not on the open "
        f"set {sorted(A.OPEN_PATHS)} — and that literal cannot import "
        "app.aggregator_config, so only this assertion keeps them together")

    # 3. app/routers/health.py's derivation, verbatim from :150-151.
    mcp_url = service_url("lloyd_mcp", "http://127.0.0.1:8500/mcp")
    ui_probe = _rebase(mcp_url.rsplit("/mcp", 1)[0] + "/health")

    async with httpx.AsyncClient(timeout=10.0) as client:
        for label, url in (("guardian", _rebase(guardian.MCP_HEALTH_URL)),
                           ("mc-ui", ui_probe)):
            r = await client.get(url)
            assert r.status_code == 200, f"{label} probe -> {r.status_code} {r.text[:120]}"
            assert r.json()["tools"] > 0, f"{label} probe got no tool count: {r.text[:120]}"


@pytest.mark.asyncio(loop_scope="module")
async def test_a_state_changing_write_with_no_session_id_is_refused_over_the_wire(
        aggregator):
    """The omitted-sid escape, closed at the layer a bench trial actually used.

    `_tool_sandbox.is_sandboxed_session("")` is False by construction, so a
    sandboxed trial that dropped `lloyd/session_id` from `_meta` read as "not
    sandboxed" and got the write surface back. The credential alone would not
    close it — the pool sends the credential on every call — so `call_tool`
    refuses a state-changing call that names no session. The two legs below are
    the same request either side of the session id being put back.

    The filesystem check is interleaved rather than batched: the positive leg
    genuinely writes this path, so asking "was it written?" after both POSTs
    would ask a question the second leg already answered.
    """
    import httpx

    path = "/tmp/lloyd-1053-wire.txt"
    if os.path.exists(path):
        os.unlink(path)                             # a stale file answers both legs wrongly
    write = {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
             "params": {"name": "Write",
                        "arguments": {"file_path": path,
                                      "content": "must not be written"},
                        "_meta": {}}}
    credentialed = dict(_WIRE_MCP, **{"X-Lloyd-Aggregator-Token": TEST_TOKEN})
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            no_sid = await client.post(aggregator, json=write, headers=credentialed)
            assert no_sid.status_code == 200, no_sid.text[:200]
            payload = no_sid.json()["result"]
            assert payload["isError"] is True, payload
            text = payload["content"][0]["text"]
            assert "no session id" in text, (
                f"the refusal did not name the cause: {text[:200]}")
            assert not os.path.exists(path), "the sessionless write hit the filesystem"

            write["params"]["_meta"] = {"lloyd/session_id": "wire-probe-session"}
            with_sid = await client.post(aggregator, json=write, headers=credentialed)
            assert with_sid.json()["result"]["isError"] is False, with_sid.text[:300]
            assert os.path.exists(path), "the session-bearing write did not dispatch"
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.asyncio(loop_scope="module")
async def test_the_tools_page_discovery_carries_the_credential(aggregator, monkeypatch):
    """`app.mcp_discovery._discover_mcp_tools` against the guarded server.

    The fifth caller of the aggregator, and the one #1053 did not list: it
    names the server by its `mcp_servers:` URL, so a search for the port or for
    `services.lloyd_mcp` never finds it. It opened its own SDK client with no
    headers, so on the day the guard landed (2026-09-18) the Tools page read
    "Server returned an error response" and
    `test_mcp_layer.test_discovery_resolves_the_configured_transport` went red
    on main — for every round's tests rung, not only its author's. That test
    could not catch it in #1053's own gate, because it talks to the LIVE
    aggregator, and the live one was still the unguarded build.

    This one owns both halves of the seam, so it fails in the gate of whoever
    breaks it.
    """
    from app.mcp_discovery import _discover_mcp_tools

    cfg = {"type": "streamable-http", "url": aggregator}
    found, err = await _discover_mcp_tools("lloyd-mcp", cfg)
    assert err is None, f"discovery was refused by the guard: {err}"
    assert {"Bash", "Read", "Write", "Edit"} <= {t["name"] for t in found}

    # The control: the same call with no credential to send is the refusal the
    # Tools page showed. Without it, "err is None" could be a server that stopped
    # asking.
    import agent_mcp.aggregator_auth as A
    monkeypatch.setattr(A, "read_token", lambda publish=False, **kw: "")
    found, err = await _discover_mcp_tools("lloyd-mcp", cfg)
    assert found == [] and err, "an uncredentialed discovery was served"
