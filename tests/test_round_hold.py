"""While a round runs, the pool holds everything else back.

An autocode round is the rarest and most expensive job in the pool: one
worktree, nine gate rungs, a landing that restarts the backend. It is also
the one most damaged by sharing the primary — it re-submits a 100-200k-token
context every iteration, so anything else claiming a slot evicts its prefix.
Measured on 2026-09-11: every autocode session showed cold re-prefills of
0.14-0.85M tokens, with 14 scheduled-task and 15 autotriage runs on the same
engine in one 21-hour window.

The KV gate beside this asks "is the engine full *now*". This asks "is the
one job worth protecting running at all".
"""

from __future__ import annotations

import pytest

from workers import pool as P


class _Src:
    def __init__(self, long_lived=False):
        self.LONG_LIVED = long_lived


REGISTRY = {
    "autocode": _Src(True),
    "autotriage": _Src(True),
    "deep-research": _Src(True),
    "arch-review": _Src(),
    "session-distill": _Src(),
    "youtube-digest": _Src(),
    "backlog-cluster": _Src(),
    "scheduled-task": _Src(),
}


@pytest.fixture
def pool(monkeypatch):
    p = P.WorkerPool.__new__(P.WorkerPool)
    p._in_flight = {}
    p._kv_gate = {"engaged": False, "engaged_since": None, "engagements": 0,
                  "kv_usage": None, "held_sources": []}
    p._round_hold = {"engaged": False, "engaged_since": None, "engagements": 0,
                     "held_sources": []}
    # The KV gate is a separate question; keep it out of these assertions.
    monkeypatch.setattr(P, "kv_gate_config", lambda: {"enabled": False})
    return p


def _run_round(pool, source="autocode", item_id=1):
    pool._in_flight[item_id] = {"source": source, "kind": "run"}


def test_nothing_is_held_when_no_round_is_running(pool):
    assert pool._round_hold_held(REGISTRY) == []
    assert pool.round_hold_status()["engaged"] is False


def test_a_running_round_holds_the_long_lived_sources(pool):
    _run_round(pool)
    held = pool._round_hold_held(REGISTRY)
    for name in ("autotriage", "deep-research", "arch-review",
                 "session-distill", "youtube-digest", "backlog-cluster"):
        assert name in held, name


def test_the_round_never_holds_itself(pool):
    _run_round(pool)
    assert "autocode" not in pool._round_hold_held(REGISTRY)


def test_scheduled_task_is_exempt(pool):
    """Short, time-sensitive, and a user-visible schedule slipping by an hour
    is its own failure.
    """
    _run_round(pool)
    assert "scheduled-task" not in pool._round_hold_held(REGISTRY)


def test_the_exempt_list_is_configurable(pool, monkeypatch):
    monkeypatch.setattr(P, "round_hold_config",
                        lambda: {"enabled": True, "exempt": ["youtube-digest"]})
    _run_round(pool)
    held = pool._round_hold_held(REGISTRY)
    assert "youtube-digest" not in held
    assert "scheduled-task" in held      # no longer exempt under this config


def test_the_hold_releases_when_the_round_leaves_in_flight(pool):
    """No lease and no TTL: `_in_flight` is exact and popped on every exit
    path, including the timeout and exception branches. A lease would be a
    second source of truth about the same fact.
    """
    _run_round(pool)
    assert pool._round_hold_held(REGISTRY)
    assert pool.round_hold_status()["engaged"] is True

    pool._in_flight.clear()
    assert pool._round_hold_held(REGISTRY) == []
    assert pool.round_hold_status()["engaged"] is False


def test_another_source_in_flight_does_not_engage_it(pool):
    _run_round(pool, source="deep-research")
    assert pool._round_hold_held(REGISTRY) == []


def test_the_kill_switch_releases_an_engaged_hold(pool, monkeypatch):
    _run_round(pool)
    assert pool._round_hold_held(REGISTRY)
    monkeypatch.setattr(P, "round_hold_config", lambda: {"enabled": False})
    assert pool._round_hold_held(REGISTRY) == []
    assert pool.round_hold_status()["engaged"] is False


def test_engagements_counts_transitions_not_polls(pool):
    _run_round(pool)
    for _ in range(5):
        pool._round_hold_held(REGISTRY)
    assert pool._round_hold["engagements"] == 1
    pool._in_flight.clear()
    pool._round_hold_held(REGISTRY)
    _run_round(pool, item_id=2)
    pool._round_hold_held(REGISTRY)
    assert pool._round_hold["engagements"] == 2


def test_claim_holds_is_the_union_of_both_gates(pool, monkeypatch):
    """One call site so a third gate has an obvious home, and so the claim
    query takes the union rather than whichever gate ran last.
    """
    monkeypatch.setattr(P, "kv_gate_config",
                        lambda: {"enabled": True, "max_kv_usage": 0.1,
                                 "window_seconds": 60})
    monkeypatch.setattr(P.WorkerPool, "_gate_reading",
                        staticmethod(lambda window: 0.99))
    _run_round(pool)
    held = set(pool._claim_holds(REGISTRY))
    # KV gate contributes the long-lived set...
    assert {"autotriage", "deep-research"} <= held
    # ...and the round hold contributes the short ones the KV gate ignores.
    assert "session-distill" in held
    assert "youtube-digest" in held
    # `scheduled-task` is exempt from the round hold and not long-lived, so
    # neither gate reaches it — which is the point of exempting it.
    assert "scheduled-task" not in held
    # `autocode` IS held here, by the KV gate, and that is correct and
    # pre-existing: the gates decide what may be CLAIMED, and starting a
    # second round under KV pressure is exactly what should not happen.
    # `_loop_is_free` already guarantees one round; this is belt and braces.
    assert "autocode" in held
    # What matters for this change is that the round hold is not the thing
    # holding it.
    assert "autocode" not in pool._round_hold_held(REGISTRY)


def test_status_reports_the_hold(pool):
    _run_round(pool)
    pool._round_hold_held(REGISTRY)
    st = pool.round_hold_status()
    assert st["engaged"] is True
    assert st["engaged_since"]
    assert "autotriage" in st["held_sources"]
    assert st["exempt"] == ["scheduled-task"]


def test_held_sources_is_a_copy_not_the_live_list(pool):
    _run_round(pool)
    pool._round_hold_held(REGISTRY)
    st = pool.round_hold_status()
    st["held_sources"].append("mutated")
    assert "mutated" not in pool._round_hold["held_sources"]


# ---------------------------------------------------------------------------
# wait_idle — one definition of "the engine is free"
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from app import vllm_metrics  # noqa: E402


class _FakeClient:
    """Serves a scripted sequence of `num_requests_running` values."""

    def __init__(self, running_sequence):
        self.seq = list(running_sequence)
        self.calls = 0

    async def get(self, url, timeout=None):
        self.calls += 1
        n = self.seq[min(self.calls - 1, len(self.seq) - 1)]

        class _R:
            text = f"vllm:num_requests_running {n}.0\n"
        return _R()

    async def aclose(self):
        pass


def test_a_momentary_zero_is_not_idle():
    """The whole reason this is not two lines at each call site. The gauge is
    sampled, requests arrive between samples, and a bench that starts on the
    first zero it sees is measuring a shared engine while believing it has
    one to itself.
    """
    client = _FakeClient([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    asyncio.run(vllm_metrics.wait_idle(
        "http://x", quiet_s=0.3, limit_s=5, poll_s=0.05, client=client))
    # It cannot have returned on the first sample: the second was busy.
    assert client.calls >= 5


def test_it_raises_rather_than_measuring_a_busy_engine():
    client = _FakeClient([3])
    with pytest.raises(TimeoutError) as exc:
        asyncio.run(vllm_metrics.wait_idle(
            "http://x", quiet_s=0.2, limit_s=0.6, poll_s=0.05, client=client))
    assert "never went idle" in str(exc.value)


def test_allow_running_lets_a_caller_ignore_its_own_request():
    """An eval measuring its own turn should wait for everyone *else* to
    leave, not for a number it is contributing to.
    """
    client = _FakeClient([1])
    asyncio.run(vllm_metrics.wait_idle(
        "http://x", quiet_s=0.2, limit_s=3, poll_s=0.05,
        allow_running=1, client=client))


def test_an_unreachable_engine_is_not_idle():
    """A scrape that fails reads as busy, not as quiet: 'I could not tell'
    must never green-light a measurement.
    """
    class _Dead:
        async def get(self, url, timeout=None):
            raise OSError("connection refused")

        async def aclose(self):
            pass

    with pytest.raises(TimeoutError):
        asyncio.run(vllm_metrics.wait_idle(
            "http://x", quiet_s=0.2, limit_s=0.5, poll_s=0.05, client=_Dead()))


def test_the_bench_script_uses_the_shared_definition():
    """Two private copies of 'is the engine busy' is how the bench and the
    pool would come to disagree about what idle means.
    """
    src = (P.__file__.rsplit("/workers/", 1)[0]
           + "/agent-services/bin/bench-admission-stall.py")
    text = open(src).read()
    assert "vllm_metrics.wait_idle(" in text
    # and no second loop of its own
    assert "quiet_since = quiet_since or time.monotonic()" not in text
