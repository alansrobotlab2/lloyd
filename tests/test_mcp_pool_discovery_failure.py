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
  - the loop refuses a turn whose pool advertised no tools;
  - and, added by #1807, that ONE call to `open()` retries a refused
    discovery inside two named bounds before it raises. The next caller's
    re-discovery does not help a scheduled run: `autonomy.run_task` iterates
    one generator, and run #53 at 03:17:04 on 2026-09-29 was over 13.0
    seconds after it began, charged to the task, having touched 0 tools.
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


@pytest.fixture(autouse=True)
def retry_waits(monkeypatch):
    """Every wait `open()` asks for, recorded instead of slept. Yields the list.

    These tests are about WHICH attempts happened and what bounded them, never
    about whether real seconds pass, so the pacing seam is intercepted. It is
    the seam and not the constant: a fixture that zeroed
    `DISCOVERY_RETRY_DELAY_SECONDS` would leave no way to assert what that
    constant declares in production, and that assertion is half of clause 2.
    The assert inside keeps the interception honest — `open()` must still ask for
    the delay it declares, so a change that silently drops the pacing between
    attempts fails here instead of passing unnoticed.
    """
    asked: list[float] = []

    async def _record(seconds: float) -> None:
        assert seconds <= mcp_pool_mod.DISCOVERY_RETRY_DELAY_SECONDS + 1e-9, (
            f"open() waited {seconds}s, past the declared "
            f"{mcp_pool_mod.DISCOVERY_RETRY_DELAY_SECONDS}s pacing")
        asked.append(seconds)

    monkeypatch.setattr(mcp_pool_mod, "_discovery_pause", _record, raising=True)
    yield asked


def _tools(*names: str) -> list[dict[str, object]]:
    return [{"name": n, "description": "tool", "inputSchema": {}} for n in names]


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
    """One broken server must not take the whole tool surface down.

    The attempt count is asserted too, and it is the retry's own boundary:
    `open()` retries only the shape that would otherwise RAISE — every server
    refused and nothing found. A degraded-but-usable catalog is not that shape,
    so the healthy turn pays one attempt per server exactly as it did before
    #1807, rather than waiting on a server that is already known to be down.
    """
    calls = _patch_discovery(monkeypatch, {
        "good": [{"name": "Bash", "description": "shell", "inputSchema": {}}],
        "bad": RuntimeError("down"),
    })
    pool = MCPPool(TWO_SERVERS)

    await pool.open()

    assert pool._opened is True
    assert pool._tool_routes == {"Bash": "good"}
    assert calls == {"good": 1, "bad": 1}, (
        f"a pool that found tools must not spend the retry budget: {calls}")


@pytest.mark.asyncio
async def test_get_or_open_pool_evicts_and_next_caller_rediscovers(monkeypatch):
    """The self-heal: a failed open must not be cached for the next turn.

    This is the whole point of raising. `get_or_open_pool` already dropped
    a pool whose `open()` raised; the bug was that `open()` never did.

    The attempt count used to read `== 2`: one refused attempt, then one from
    the next caller. #1807 clause 1 put the retry INSIDE `open()`, so the first
    caller now burns all `DISCOVERY_MAX_ATTEMPTS` of them before it raises and
    the second caller's single attempt arrives on top — hence the cap plus one.
    The property this test exists for is unchanged: the second caller
    re-discovered at all, instead of inheriting an empty cached pool.
    """
    monkeypatch.setattr(mcp_pool_mod, "_POOL_CACHE", {})
    outcomes: dict[str, object] = {SERVER_NAME: RuntimeError("aggregator restarting")}
    calls = _patch_discovery(monkeypatch, outcomes)

    with pytest.raises(ToolDiscoveryError):
        await get_or_open_pool(ONE_SERVER)
    assert mcp_pool_mod._POOL_CACHE == {}, "failed pool must not stay cached"
    assert calls[SERVER_NAME] == mcp_pool_mod.DISCOVERY_MAX_ATTEMPTS, (
        f"the failing caller should have used its whole bounded retry: {calls}")

    # Aggregator comes back. The next caller must re-attempt discovery
    # rather than inheriting a permanently empty pool.
    outcomes[SERVER_NAME] = [{"name": "Bash", "description": "shell", "inputSchema": {}}]
    pool = await get_or_open_pool(ONE_SERVER)

    assert calls[SERVER_NAME] == mcp_pool_mod.DISCOVERY_MAX_ATTEMPTS + 1, (
        f"second caller did not re-discover: {calls}")
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


@pytest.mark.asyncio
async def test_a_turn_whose_pool_never_opens_does_not_leak_the_run_counter(monkeypatch):
    """`harness_runs` is what the promoter's idle wait reads. The counter was
    incremented before `_build_pool`, so a pool that raised leaked one count
    for the life of the process: on 2026-09-16 the backend read
    `harness_runs=1` over nothing for nine hours and nine gate-passed rounds
    failed to land with "backend never went idle within 900s"."""
    from app.harness import loop as loop_mod
    from app.harness.options import RunOptions

    async def failing_build_pool(options):
        raise ToolDiscoveryError("aggregator restarting")

    monkeypatch.setattr(loop_mod, "_build_pool", failing_build_pool, raising=True)
    before = loop_mod.active_run_count()
    with pytest.raises(ToolDiscoveryError):
        async for _ in loop_mod.run_query(
            [{"role": "user", "content": "x"}],
            RunOptions(model="primary", base_url="http://127.0.0.1:8096"),
        ):
            pass
    assert loop_mod.active_run_count() == before, "a turn that never ran must not count as running"

    class _EmptyPool:
        discovered: list = []

    async def empty_build_pool(options):
        return _EmptyPool()
    monkeypatch.setattr(loop_mod, "_build_pool", empty_build_pool, raising=True)
    with pytest.raises(ToolDiscoveryError):
        async for _ in loop_mod.run_query(
            [{"role": "user", "content": "x"}],
            RunOptions(model="primary", base_url="http://127.0.0.1:8096"),
        ):
            pass
    assert loop_mod.active_run_count() == before, "the toolless refusal inside the try releases it too"


# ── #1807: ONE open() retries a refused discovery, inside two named bounds ────
#
# `automod_land` restarts MCP before the backend, and a scheduler dispatch that
# fired inside that window reached an aggregator that could not answer
# `tools/list` yet. Until this round the pool gave up on the first refusal and
# left recovery to the NEXT caller. Run #53 at 2026-09-29 03:17:04 had no next
# caller: `trigger: scheduler`, `duration_seconds: 13.0`, `tool_errors: 0`,
# `changes_files: 0`, and it was charged to the task. These are the recovery
# half of #1807; the charging half is in
# `tests/test_autonomy_discovery_infra_kind.py`.


@pytest.mark.asyncio
async def test_a_refused_first_attempt_recovers_on_the_second(monkeypatch):
    """Clause 1: first `tools/list` raises, second answers => opened pool.

    The exact shape of a landing's MCP restart as one dispatch sees it — a
    supervisor stop/start, so the port refuses for a moment and then answers.
    Three properties, and each fails on its own: the attempt count (exactly
    two, which is what makes this a retry rather than a loop), `_opened` (a
    pool that recovered its tools but stayed closed is evicted by
    `get_or_open_pool` and the run dies anyway), and `_tool_routes` (the
    catalog has to be the second attempt's, not empty).
    """
    from contextlib import asynccontextmanager

    attempts: list[str] = []

    class _NullSession:
        pass

    @asynccontextmanager
    async def _session(self, cfg):
        yield _NullSession()

    async def _list_tools(self, server_name, session):
        attempts.append(server_name)
        if len(attempts) == 1:
            # What a restart actually looks like from here: nothing bound yet.
            raise ConnectionRefusedError("MCP aggregator restarting")
        return _tools("Bash")

    monkeypatch.setattr(MCPPool, "_http_session", _session, raising=True)
    monkeypatch.setattr(MCPPool, "_list_tools", _list_tools, raising=True)
    pool = MCPPool(ONE_SERVER)

    await pool.open()

    assert len(attempts) == 2, f"discovery was not attempted exactly twice: {attempts}"
    assert pool._opened is True
    assert pool._tool_routes == {"Bash": SERVER_NAME}


@pytest.mark.asyncio
async def test_retry_bounds_are_named_and_coherent():
    """Clause 2's first half: both bounds exist, by name, with their values.

    Read off the module rather than restated as literals this file alone owns,
    so widening a bound into an unbounded retry has to be a change HERE as well
    — that is the only way this loop ever gets re-examined. The pacing constant
    is pinned beside them because it is what makes the attempt cap cost real
    time at all. The last assertion is what makes the two bounds BOUNDS
    together: if one gap exceeded the whole budget, the budget would expire
    before the second attempt and the cap would be decoration.
    """
    assert mcp_pool_mod.DISCOVERY_MAX_ATTEMPTS == 3
    assert mcp_pool_mod.DISCOVERY_RETRY_BUDGET_SECONDS == 15.0
    assert mcp_pool_mod.DISCOVERY_RETRY_DELAY_SECONDS == 3.0
    assert mcp_pool_mod.DISCOVERY_MAX_ATTEMPTS >= 2, "a retry is at least a second try"
    assert mcp_pool_mod.DISCOVERY_RETRY_DELAY_SECONDS * (
            mcp_pool_mod.DISCOVERY_MAX_ATTEMPTS - 1
    ) <= mcp_pool_mod.DISCOVERY_RETRY_BUDGET_SECONDS, (
        "the attempt cap could never be reached: the budget always expires first")


@pytest.mark.asyncio
async def test_a_discovery_that_fails_every_attempt_still_raises_and_evicts(
        monkeypatch, retry_waits):
    """Clause 2's second half: the bound is a bound, and the failure survives it.

    Driven through `get_or_open_pool`, the way a turn reaches discovery, so the
    `_POOL_CACHE` assertion is about the cache the NEXT turn would read and not
    about a pool this test assembled by hand. The attempt count is the named cap
    exactly: fewer would mean the retry gave up early, more would mean the bound
    does not hold. And the waits are one per gap, not one per attempt — a retry
    that slept after its last attempt would add dead seconds to every dying run.
    """
    monkeypatch.setattr(mcp_pool_mod, "_POOL_CACHE", {})
    calls = _patch_discovery(monkeypatch, {SERVER_NAME: RuntimeError("aggregator down")})

    with pytest.raises(ToolDiscoveryError) as exc:
        await get_or_open_pool(ONE_SERVER)

    assert calls[SERVER_NAME] == mcp_pool_mod.DISCOVERY_MAX_ATTEMPTS, (
        f"expected exactly the named cap of attempts, got {calls}")
    assert retry_waits == [mcp_pool_mod.DISCOVERY_RETRY_DELAY_SECONDS] * (
            mcp_pool_mod.DISCOVERY_MAX_ATTEMPTS - 1), (
        f"expected one wait per gap between attempts: {retry_waits}")
    assert exc.value.servers == [SERVER_NAME], (
        f"the error must still name the server that never answered: {exc.value.servers}")
    assert "MCP discovery yielded no tools" in str(exc.value)
    assert mcp_pool_mod._POOL_CACHE == {}, (
        "a pool that never opened must not be cached for the next turn")


class _FastForwardClock:
    """A monotonic clock that advances `step` seconds per reading.

    A wall-clock bound cannot be shown to bind by waiting for it. Handing the
    module a clock that spends 20 s per reading is what lets the retry see more
    elapsed time than its budget allows inside a single failed attempt.
    """

    def __init__(self, step: float):
        self.step = step
        self._t = 0.0

    def monotonic(self) -> float:
        now = self._t
        self._t += self.step
        return now


@pytest.mark.asyncio
async def test_the_wall_clock_budget_stops_the_retry_before_the_attempt_cap(
        monkeypatch):
    """The budget is a real second bound, not a comment beside the cap.

    Each clock reading costs 20 s — more than the whole 15 s budget — so after
    ONE failed attempt the window is spent and a second attempt must not happen
    even though the cap would allow two more. Delete the `remaining <= 0` break
    in `MCPPool.open()` and the count below becomes 3, which is the point: this
    is the falsifier for the budget, and the case it exists for is a server that
    accepts the socket and says nothing, where three attempts would cost three
    connect timeouts and could outrun the run's own `timeout_seconds`.
    """
    pool = MCPPool(ONE_SERVER)          # built on the real clock
    calls = _patch_discovery(monkeypatch, {SERVER_NAME: RuntimeError("hanging")})
    monkeypatch.setattr(mcp_pool_mod, "time", _FastForwardClock(20.0))

    with pytest.raises(ToolDiscoveryError):
        await pool.open()

    assert calls[SERVER_NAME] == 1, (
        f"the retry kept going past the wall-clock budget: {calls}")
    assert mcp_pool_mod.DISCOVERY_MAX_ATTEMPTS > 1, (
        "the cap must allow more than this single attempt, or the assertion "
        "above is proving the cap and not the budget")
