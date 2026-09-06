"""A pool that discovers nothing must fail, not cache its own emptiness.

The 2026-09-06 incident: `MCPPool.open()` caught a transient discovery
error, logged a warning, `continue`d, and set `_opened = True` anyway. The
pool was then cached process-wide by `get_or_open_pool`, which
short-circuits on `_opened`, so nothing ever retried discovery. Every
later turn built an empty catalog; `client.stream_chat` omits `tools`
entirely when the list is falsy, so vLLM never engaged its tool parser.
Lloyd reasoned his way to "call Bash", had no channel to emit a tool call
on, and either stopped with an empty message or wrote the call out as
prose — twice in one afternoon, ~30 minutes each, with nothing in the
stream naming the cause.

What these pin:
  - a total discovery failure raises rather than marking the pool open;
  - `get_or_open_pool` therefore evicts it and the NEXT caller re-discovers
    (the recovery path already existed — nothing ever failed to trigger it);
  - a partial failure still degrades gracefully, which is what the
    `continue` is actually for;
  - the loop refuses a turn whose pool advertised no tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.harness import mcp_pool as mcp_pool_mod  # noqa: E402
from app.harness.errors import ToolDiscoveryError  # noqa: E402
from app.harness.mcp_pool import (  # noqa: E402
    DEFAULT_LLOYD_MCP_SERVERS,
    MCPPool,
    get_or_open_pool,
)

# Derived from the canonical config rather than written inline — see
# `test_no_inline_mcp_server_config_anywhere_in_the_repo`. Discovery is
# monkeypatched in every test here, so the URL is never dialled; only the
# server *names* matter, and taking the transport from the real constant
# keeps this file from becoming the sixth stale literal.
ONE_SERVER = dict(DEFAULT_LLOYD_MCP_SERVERS)
SERVER_NAME = next(iter(ONE_SERVER))
TWO_SERVERS = {
    "good": dict(ONE_SERVER[SERVER_NAME]),
    "bad": dict(ONE_SERVER[SERVER_NAME]),
}


def _patch_discovery(monkeypatch, results: dict[str, object]) -> dict[str, int]:
    """Make `_list_tools` return canned tools (or raise) per server name.

    `results[name]` is either a list of MCP tool dicts or an Exception to
    raise. Returns a per-server call counter so a test can assert that a
    rebuilt pool actually re-attempted discovery.
    """
    calls: dict[str, int] = {}

    class _NullSession:
        pass

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_http_session(self, cfg):
        yield _NullSession()

    async def fake_list_tools(self, server_name, session):
        calls[server_name] = calls.get(server_name, 0) + 1
        outcome = results[server_name]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(MCPPool, "_http_session", fake_http_session, raising=True)
    monkeypatch.setattr(MCPPool, "_list_tools", fake_list_tools, raising=True)
    return calls


@pytest.mark.asyncio
async def test_total_discovery_failure_raises_and_leaves_pool_closed(monkeypatch):
    """Nothing discovered => raise, and do NOT mark the pool opened."""
    _patch_discovery(monkeypatch, {SERVER_NAME: RuntimeError("aggregator restarting")})
    pool = MCPPool(ONE_SERVER)

    with pytest.raises(ToolDiscoveryError) as exc:
        await pool.open()

    assert SERVER_NAME in exc.value.servers
    # The precise regression: `_opened` staying True is what made the
    # emptiness permanent, because `get_or_open_pool` returns early on it.
    assert pool._opened is False
    assert pool._tool_routes == {}


@pytest.mark.asyncio
async def test_partial_discovery_failure_still_degrades_gracefully(monkeypatch):
    """One broken server must not take the whole tool surface down."""
    _patch_discovery(monkeypatch, {
        "good": [{"name": "Bash", "description": "shell", "inputSchema": {}}],
        "bad": RuntimeError("down"),
    })
    pool = MCPPool(TWO_SERVERS)

    await pool.open()

    assert pool._opened is True
    assert pool._tool_routes == {"Bash": "good"}


@pytest.mark.asyncio
async def test_get_or_open_pool_evicts_and_next_caller_rediscovers(monkeypatch):
    """The self-heal: a failed open must not be cached for the next turn.

    This is the whole point of raising. `get_or_open_pool` already dropped
    a pool whose `open()` raised; the bug was that `open()` never did.
    """
    monkeypatch.setattr(mcp_pool_mod, "_POOL_CACHE", {})
    outcomes: dict[str, object] = {SERVER_NAME: RuntimeError("aggregator restarting")}
    calls = _patch_discovery(monkeypatch, outcomes)

    with pytest.raises(ToolDiscoveryError):
        await get_or_open_pool(ONE_SERVER)
    assert mcp_pool_mod._POOL_CACHE == {}, "failed pool must not stay cached"

    # Aggregator comes back. The next caller must re-attempt discovery
    # rather than inheriting a permanently empty pool.
    outcomes[SERVER_NAME] = [{"name": "Bash", "description": "shell", "inputSchema": {}}]
    pool = await get_or_open_pool(ONE_SERVER)

    assert calls[SERVER_NAME] == 2, "second caller did not re-discover"
    assert pool._tool_routes == {"Bash": SERVER_NAME}


@pytest.mark.asyncio
async def test_internal_tools_are_routable_though_never_advertised(monkeypatch):
    """`_`-prefixed tools are hidden from the model but must still dispatch.

    `build_tool_list` skips them; the routing table must not. During the
    incident the empty pool broke `_BackgroundTaskDrain` too, which is how
    the failure first became visible in the log ("no server claims tool").
    """
    from app.harness.tool_schema import build_tool_list

    _patch_discovery(monkeypatch, {SERVER_NAME: [
        {"name": "Bash", "description": "shell", "inputSchema": {}},
        {"name": "_BackgroundTaskDrain", "description": "internal", "inputSchema": {}},
    ]})
    pool = MCPPool(ONE_SERVER)
    await pool.open()

    assert pool._tool_routes["_BackgroundTaskDrain"] == SERVER_NAME
    advertised = {t["function"]["name"] for t in build_tool_list(list(pool.discovered), set())}
    assert advertised == {"Bash"}


@pytest.mark.asyncio
async def test_run_query_refuses_a_toolless_turn(monkeypatch):
    """The loop must not stream a turn with an empty tool list.

    Defense in depth for the shapes `open()` cannot catch — a server that
    answers successfully with zero tools raises nothing, so the pool opens
    clean and empty.
    """
    from app.harness import loop as loop_mod
    from app.harness.options import RunOptions

    class _EmptyPool:
        discovered: list = []

    async def fake_build_pool(options):
        return _EmptyPool()

    monkeypatch.setattr(loop_mod, "_build_pool", fake_build_pool, raising=True)

    with pytest.raises(ToolDiscoveryError):
        async for _ in loop_mod.run_query(
            [{"role": "user", "content": "clear the poisoned worker"}],
            RunOptions(model="primary", base_url="http://127.0.0.1:8096"),
        ):
            pass
