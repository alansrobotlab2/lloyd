"""`_cached_swr`: an expired slow half is served stale and refreshed off-request.

The dashboard awaits every section before it answers, so an expired 60 s entry
recomputed inline (`board_health`, 13.99 s on 2026-11-12) held the whole page
for that long. These nodes pin the replacement on call counts and events, never
a stopwatch.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.routers import dashboard as dash


@pytest.fixture(autouse=True)
def _clean():
    dash._cache.clear()
    dash._refreshing.clear()
    yield
    dash._cache.clear()
    dash._refreshing.clear()


def _wait_for(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_an_empty_entry_is_computed_inline():
    calls = []
    assert dash._cached_swr("k", 60.0, lambda: calls.append(1) or "v") == "v"
    assert calls == [1]
    assert dash._cache["k"][1] == "v"


def test_a_fresh_entry_is_served_without_recomputing():
    dash._cache["k"] = (time.monotonic(), "fresh")
    assert dash._cached_swr("k", 60.0, lambda: pytest.fail("recomputed")) == "fresh"


def test_an_expired_entry_is_served_stale_while_one_refresh_runs():
    release = threading.Event()
    calls = []

    def slow():
        calls.append(1)
        release.wait(5)
        return "new"

    dash._cache["k"] = (time.monotonic() - 120.0, "old")
    # The request that finds it expired is answered at once with the old value,
    # and so is every request while the refresh is still out: one refresh, not one
    # per poll.
    for _ in range(3):
        assert dash._cached_swr("k", 60.0, slow) == "old"
    assert _wait_for(lambda: calls == [1])
    release.set()
    assert _wait_for(lambda: dash._cache["k"][1] == "new")
    assert calls == [1]
    assert _wait_for(lambda: "k" not in dash._refreshing)
    assert dash._cached_swr("k", 60.0, slow) == "new"


def test_a_failed_refresh_keeps_the_stale_value_and_allows_the_next_one():
    dash._cache["k"] = (time.monotonic() - 120.0, "old")

    def boom():
        raise RuntimeError("scan failed")

    assert dash._cached_swr("k", 60.0, boom) == "old"
    assert _wait_for(lambda: "k" not in dash._refreshing)
    assert dash._cache["k"][1] == "old"
    calls = []
    dash._cached_swr("k", 60.0, lambda: calls.append(1) or "new")
    assert _wait_for(lambda: dash._cache["k"][1] == "new")


def test_a_refresh_never_overwrites_a_cache_cleared_meanwhile():
    release = threading.Event()
    started = threading.Event()

    def slow():
        started.set()
        release.wait(5)
        return "late"

    dash._cache["k"] = (time.monotonic() - 120.0, "old")
    dash._cached_swr("k", 60.0, slow)
    assert started.wait(5)
    dash._cache.clear()  # what the cold-cycle tests do between cycles
    release.set()
    assert _wait_for(lambda: "k" not in dash._refreshing)
    assert "k" not in dash._cache


def test_the_two_slow_halves_use_it(monkeypatch):
    """`backlog_health` and the automod scorecard are the 60 s halves that held the
    page; both must go through the stale-serving path."""
    seen = []
    real = dash._cached_swr

    def spy(key, ttl, fn):
        seen.append(key)
        return real(key, ttl, fn)

    monkeypatch.setattr(dash, "_cached_swr", spy)
    monkeypatch.setattr(dash, "_backlog_health", lambda d: {"stub": True})
    dash._backlog()
    try:
        dash._automod()
    except Exception:
        pass  # the scorecard's own inputs are not this node's concern
    assert "backlog_health" in seen
    assert "automod" in seen
