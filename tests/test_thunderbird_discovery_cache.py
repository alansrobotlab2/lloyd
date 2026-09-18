"""A closed Thunderbird must cost one failed bridge open per retry window,
not one per tool call.

From 18:11 on 2026-09-17 to 08:57 on 09-18 Thunderbird was not running, and
`thunderbird.list_tools()` cached its failure as `[]` behind a check
(`if _cached_tools and …`) that an empty list never passes. Every discovery
re-tried the bridge and waited ~5 s for it to fail; a `Glob` through the
aggregator took 10 s and every agent iteration ran 2-4x slower.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_mcp import thunderbird as tb


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _FakePool:
    def __init__(self) -> None:
        self.discovered = [("thunderbird", [
            {"name": "listAccounts", "description": "accounts",
             "inputSchema": {"type": "object", "properties": {}}},
        ])]


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    # Replace the module's `time`, never `time.monotonic` itself: asyncio's
    # event loop schedules on the real one, and a frozen clock hangs sleep().
    monkeypatch.setattr(tb, "time", SimpleNamespace(monotonic=c))
    monkeypatch.setattr(tb, "_cached_tools", [])
    monkeypatch.setattr(tb, "_cached_at", None)
    monkeypatch.setattr(tb, "_discovery_lock", asyncio.Lock(), raising=False)
    return c


def _bridge(monkeypatch, *, up: bool, delay: float = 0.0) -> list[int]:
    calls = [0]

    async def fake_get_pool():
        calls[0] += 1
        if delay:
            await asyncio.sleep(delay)
        if not up:
            raise ConnectionRefusedError("bridge on :8765 refused")
        return _FakePool()

    monkeypatch.setattr(tb, "_get_pool", fake_get_pool)
    return calls


def test_a_failed_discovery_is_cached_for_the_retry_window(clock, monkeypatch):
    calls = _bridge(monkeypatch, up=False)

    async def run():
        assert await tb.list_tools() == []
        clock.now += tb.UNAVAILABLE_RETRY_SECONDS - 1
        assert await tb.list_tools() == []

    asyncio.run(run())
    assert calls[0] == 1


def test_a_failed_discovery_is_retried_after_the_window(clock, monkeypatch):
    calls = _bridge(monkeypatch, up=False)

    async def run():
        await tb.list_tools()
        clock.now += tb.UNAVAILABLE_RETRY_SECONDS + 1
        _bridge(monkeypatch, up=True)
        return await tb.list_tools()

    tools = asyncio.run(run())
    assert calls[0] == 1
    assert [t.name for t in tools] == [tb._lloyd_name("listAccounts")]


def test_the_retry_window_is_shorter_than_the_success_ttl():
    # A Thunderbird that comes back must be seen well before a working
    # bridge's tool list would be re-checked.
    assert tb.UNAVAILABLE_RETRY_SECONDS < tb.DISCOVERY_TTL_SECONDS


def test_concurrent_callers_share_one_failed_attempt(clock, monkeypatch):
    calls = _bridge(monkeypatch, up=False, delay=0.05)

    async def run():
        return await asyncio.gather(*(tb.list_tools() for _ in range(5)))

    results = asyncio.run(run())
    assert results == [[]] * 5
    assert calls[0] == 1


def test_a_successful_discovery_keeps_its_ttl(clock, monkeypatch):
    calls = _bridge(monkeypatch, up=True)

    async def run():
        first = await tb.list_tools()
        clock.now += tb.DISCOVERY_TTL_SECONDS - 1
        second = await tb.list_tools()
        clock.now += 2
        await tb.list_tools()
        return first, second

    first, second = asyncio.run(run())
    assert first and second is first
    assert calls[0] == 2


def test_shutdown_forgets_the_cache(clock, monkeypatch):
    calls = _bridge(monkeypatch, up=False)

    async def run():
        await tb.list_tools()
        await tb.shutdown()
        await tb.list_tools()

    asyncio.run(run())
    assert calls[0] == 2
