"""P12 (review 2026-09-24): MCP discovery refresh, per-tool timeouts, handshake timing.

Discovery used to run once per pool — once per backend process — and `_reopen`
cleared the route table in place across the whole rediscovery. These pin the
replacement: an immutable catalog swapped in one assignment, a TTL refresh at
the turn boundary that never empties it, a per-tool call budget the server
declares in the tool's `_meta`, and the per-call HTTP handshake timed apart
from the call and returned as `result["timing"]`, where P11 reads it.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import mcp.types as T
import pytest

from app.harness import mcp_pool as P

pytestmark = pytest.mark.asyncio

(SERVER,) = P.DEFAULT_LLOYD_MCP_SERVERS


def _tool(name: str, *, timeout: float | None = None, read_only: bool = True) -> T.Tool:
    return T.Tool(
        name=name, inputSchema={"type": "object", "properties": {}},
        annotations=T.ToolAnnotations(readOnlyHint=read_only),
        _meta=({P.META_TIMEOUT_SECONDS: timeout} if timeout is not None else None),
    )


class _Server:
    """What the fake HTTP session answers: a mutable tool list, optional
    failure, an optional stall inside `initialize`, and a record of every
    `call_tool` budget it was handed."""

    def __init__(self, tools):
        self.tools = list(tools)
        self.fail = False
        self.list_delay = 0.0
        self.handshake_delay = 0.0
        self.budgets: list[float] = []
        self.lists = 0


class _Session:
    def __init__(self, server: _Server):
        self.server = server

    async def list_tools(self):
        self.server.lists += 1
        if self.server.list_delay:
            await asyncio.sleep(self.server.list_delay)
        if self.server.fail:
            raise ConnectionError("aggregator restarting")
        return T.ListToolsResult(tools=list(self.server.tools))

    async def call_tool(self, bare, args, read_timeout_seconds=None, meta=None):
        self.server.budgets.append(read_timeout_seconds)
        return T.CallToolResult(content=[T.TextContent(type="text", text=f"ran {bare}")])


def _pool(monkeypatch, server: _Server) -> P.MCPPool:
    # The real default config; never contacted, `_http_session` is replaced.
    pool = P.MCPPool(dict(P.DEFAULT_LLOYD_MCP_SERVERS))

    @asynccontextmanager
    async def _session(_cfg):
        if server.handshake_delay:
            await asyncio.sleep(server.handshake_delay)
        if server.fail:
            raise ConnectionError("aggregator restarting")
        yield _Session(server)

    monkeypatch.setattr(pool, "_http_session", _session)
    return pool


def _names(pool) -> set[str]:
    return {t["name"] for _s, tools in pool.discovered for t in tools}


def _age(pool, seconds: float) -> None:
    pool._catalog_at -= seconds


# ── refresh ──────────────────────────────────────────────────────────────────

async def test_a_stale_catalog_is_rediscovered_after_the_ttl(monkeypatch):
    server = _Server([_tool("Read")])
    pool = _pool(monkeypatch, server)
    await pool.open()
    server.tools.append(_tool("Grep"))

    assert await pool.ensure_fresh(ttl_s=300) is False, "still inside the TTL"
    assert _names(pool) == {"Read"}

    _age(pool, 301)
    assert await pool.ensure_fresh(ttl_s=300) is True
    assert _names(pool) == {"Read", "Grep"}
    assert pool._tool_routes["Grep"] == SERVER
    assert pool.stats()["refresh_changes"] == 1


async def test_ttl_zero_never_refreshes(monkeypatch):
    server = _Server([_tool("Read")])
    pool = _pool(monkeypatch, server)
    await pool.open()
    server.tools.append(_tool("Grep"))
    _age(pool, 10_000)
    lists = server.lists
    assert await pool.ensure_fresh(ttl_s=0) is False
    assert server.lists == lists, "TTL 0 must not even ask"
    assert _names(pool) == {"Read"}


async def test_the_ttl_comes_from_config(monkeypatch):
    monkeypatch.setitem(P.CONFIG, "harness", {"mcp_pool": {"discovery_ttl_s": 0}})
    assert P.discovery_ttl_s() == 0
    monkeypatch.setitem(P.CONFIG, "harness", {})
    assert P.discovery_ttl_s() == P.DEFAULT_DISCOVERY_TTL_S == 300


async def test_a_failed_refresh_keeps_the_old_catalog(monkeypatch):
    server = _Server([_tool("Read"), _tool("Bash", read_only=False)])
    pool = _pool(monkeypatch, server)
    await pool.open()
    before = pool._catalog
    server.fail = True
    _age(pool, 301)

    assert await pool.ensure_fresh(ttl_s=300) is False
    assert pool._catalog is before
    assert _names(pool) == {"Read", "Bash"}
    stats = pool.stats()
    assert stats["refresh_failures"] == 1
    assert "aggregator restarting" in stats["last_refresh_error"]
    # The attempt is stamped, so a down aggregator costs one try per TTL.
    lists = server.lists
    assert await pool.ensure_fresh(ttl_s=300) is False
    assert server.lists == lists


async def test_a_refresh_that_lists_nothing_publishes_nothing(monkeypatch):
    server = _Server([_tool("Read")])
    pool = _pool(monkeypatch, server)
    await pool.open()
    server.tools = []
    _age(pool, 301)
    assert await pool.ensure_fresh(ttl_s=300) is False
    assert _names(pool) == {"Read"}, "a refresh must never produce an empty pool"


async def test_an_unchanged_listing_keeps_the_catalog_object(monkeypatch):
    server = _Server([_tool("Read")])
    pool = _pool(monkeypatch, server)
    await pool.open()
    before = pool._catalog
    _age(pool, 301)
    assert await pool.ensure_fresh(ttl_s=300) is False
    assert pool._catalog is before


async def test_a_call_in_flight_during_a_swap_never_sees_an_empty_route_table(monkeypatch):
    server = _Server([_tool("Read"), _tool("Grep")])
    pool = _pool(monkeypatch, server)
    await pool.open()
    server.tools = [_tool("Read"), _tool("Glob")]
    server.list_delay = 0.05
    _age(pool, 301)

    seen: list[int] = []

    async def _watch():
        while not refresh.done():
            seen.append(len(pool._tool_routes))
            assert (await pool.call_tool("Read", {}))["content"] == "ran Read"
            await asyncio.sleep(0)

    refresh = asyncio.create_task(pool.ensure_fresh(ttl_s=300))
    await asyncio.gather(refresh, _watch())
    assert refresh.result() is True
    assert seen and min(seen) > 0
    assert _names(pool) == {"Read", "Glob"}


async def test_a_reopen_keeps_the_routes_until_the_new_catalog_lands():
    """stdio `_reopen` used to `.clear()` the routes before rediscovering."""
    pool = P.MCPPool({"s": {"type": "stdio", "command": "true"}})
    pool._register("s", [{"name": "t_s", "description": "", "inputSchema": {}}])
    pool._opened = True
    during: list[dict] = []

    async def _open():
        during.append(dict(pool._tool_routes))
        pool._catalog = P._build_catalog([("s", [{"name": "t_s2", "description": "",
                                                   "inputSchema": {}}])])
        pool._opened = True

    pool.open = _open
    await pool._reopen()
    assert during == [{"t_s": "s"}]
    assert pool._tool_routes == {"t_s2": "s"}


async def test_a_failed_reopen_leaves_the_previous_catalog(monkeypatch):
    server = _Server([_tool("Read")])
    pool = _pool(monkeypatch, server)
    await pool.open()
    server.fail = True
    pool._opened = False
    with pytest.raises(P.ToolDiscoveryError):
        await pool.open()
    assert _names(pool) == {"Read"}


# ── per-tool timeout ─────────────────────────────────────────────────────────

async def test_a_declared_timeout_reaches_call_tool_and_is_clamped(monkeypatch):
    server = _Server([_tool("http_fetch", timeout=120), _tool("Slow", timeout=5000),
                      _tool("Plain")])
    pool = _pool(monkeypatch, server)
    await pool.open()

    assert pool.timeout_for("http_fetch") == 120
    assert pool.timeout_for("Slow") == P.CALL_TIMEOUT_SECONDS
    assert pool.timeout_for("Plain") == P.CALL_TIMEOUT_SECONDS

    await pool.call_tool("http_fetch", {})
    await pool.call_tool("Slow", {})
    await pool.call_tool("Plain", {})
    await pool.call_tool("http_fetch", {}, timeout_seconds=7)   # a caller's own wins
    assert server.budgets == [120, P.CALL_TIMEOUT_SECONDS, P.CALL_TIMEOUT_SECONDS, 7]


async def test_per_tool_timeouts_have_an_off_switch(monkeypatch):
    server = _Server([_tool("http_fetch", timeout=120)])
    pool = _pool(monkeypatch, server)
    await pool.open()
    monkeypatch.setitem(P.CONFIG, "harness", {"mcp_pool": {"per_tool_timeouts": False}})
    assert pool.timeout_for("http_fetch") == P.CALL_TIMEOUT_SECONDS


async def test_the_aggregator_declares_timeouts_in_meta_not_annotations():
    """mcp 2.1's ToolAnnotations drops unknown keys; `_meta` round-trips."""
    from agent_mcp import annotations as A

    assert A.META_TIMEOUT_SECONDS == P.META_TIMEOUT_SECONDS
    dropped = T.ToolAnnotations(**{A.META_TIMEOUT_SECONDS: 5})
    assert A.META_TIMEOUT_SECONDS not in dropped.model_dump(by_alias=True)

    tool = A.annotate(T.Tool(name="Bash", inputSchema={"type": "object"}))
    wire = T.Tool.model_validate(tool.model_dump(by_alias=True, exclude_none=True))
    assert P._tool_dict(wire)["timeoutSeconds"] == A.TIMEOUT_SECONDS["Bash"]
    # A module's own annotations still win, and still get the timeout.
    own = A.annotate(T.Tool(name="Bash", inputSchema={"type": "object"},
                            annotations=T.ToolAnnotations(readOnlyHint=True)))
    assert own.annotations.read_only_hint is True
    assert own.meta[A.META_TIMEOUT_SECONDS] == A.TIMEOUT_SECONDS["Bash"]
    # Every declared budget is below the pool's ceiling, or it could only be clamped.
    assert all(0 < v <= P.CALL_TIMEOUT_SECONDS for v in A.TIMEOUT_SECONDS.values())


# ── handshake timing ─────────────────────────────────────────────────────────

async def test_the_handshake_is_timed_separately_from_the_call(monkeypatch):
    server = _Server([_tool("Read")])
    pool = _pool(monkeypatch, server)
    await pool.open()
    server.handshake_delay = 0.08

    result = await pool.call_tool("Read", {})
    timing = result["timing"]
    assert timing["handshake_ms"] >= 60
    assert timing["call_ms"] < timing["handshake_ms"]
    stats = pool.stats()
    assert stats["handshakes"] == 1 and stats["handshake_ms_max"] >= 60
    assert stats["tools"] == 1 and stats["calls"] == 1


async def test_a_stdio_call_reports_no_handshake(monkeypatch):
    pool = P.MCPPool({"s": {"type": "stdio", "command": "true"}})
    pool._sessions["s"] = _Session(_Server([]))
    pool._register("s", [{"name": "t", "description": "", "inputSchema": {}}])
    pool._opened = True
    assert "timing" not in await pool.call_tool("t", {})


async def test_the_handshake_lights_up_p11s_tool_result(monkeypatch):
    """End to end through `run_query`: the real pool's timing reaches the
    `tool_result` event that P11 already reads optionally."""
    from app.harness import loop as L
    from app.harness.options import RunOptions
    from app.harness.tests import _replay as R

    server = _Server([_tool("Read")])
    pool = _pool(monkeypatch, server)
    await pool.open()
    server.handshake_delay = 0.03
    engine = R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call("c1", "Read", "a")]),
        R.Step(text="done"),
    ])

    async def _ready(*_a, **_kw):
        return pool

    monkeypatch.setattr(L, "_build_pool", _ready)
    monkeypatch.setattr(L, "stream_chat", engine)
    out = await R.drive(RunOptions(model="m", max_turns=6, tool_search_enabled=False))
    result = R.of_type(out, "tool_result")[0]
    assert result["handshake_ms"] >= 20


async def test_build_pool_refreshes_at_the_turn_boundary_and_swallows_failure(monkeypatch):
    from app.harness import loop as L
    from app.harness.options import RunOptions

    calls: list[str] = []

    class _Stub:
        async def ensure_fresh(self):
            calls.append("fresh")
            raise RuntimeError("boom")

    async def _open(_cfg):
        return _Stub()

    monkeypatch.setattr(L, "get_or_open_pool", _open)
    pool = await L._build_pool(RunOptions(model="primary"))
    assert isinstance(pool, _Stub) and calls == ["fresh"]
