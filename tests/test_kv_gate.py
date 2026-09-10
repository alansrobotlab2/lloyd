"""The worker pool's KV budget gate (`workers/pool.py`).

The 09-09 stall was long-lived agent loops evicting each other's prefixes:
each iteration re-submits a 100-200k context, and between iterations a
turn's prefix survives only in the free pool. Short-lived jobs never came
back to miss. So the gate decides *who* starts — a source that declares
`LONG_LIVED` waits while the primary's KV is over budget — and three
properties make it safe to leave on:

* a held item stays queued with its attempt intact (a hold is not a failure);
* the exclusion is applied in SQL, so held rows cannot hide a claimable one
  (the `max_inflight` rule, `WorkQueue.claim_next`);
* no reading means open, so an unreadable /metrics cannot stop the pool.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from app import engine_pressure as ep
from workers.pool import WorkerPool, long_lived_sources
from workers.queue import WorkQueue


@pytest.fixture
def q(tmp_path) -> WorkQueue:
    return WorkQueue(tmp_path / "workers.db")


@pytest.fixture(autouse=True)
def _clean():
    ep.reset()
    yield
    ep.reset()


def _kv(value: float) -> None:
    ep.record(ep.Sample(time.monotonic(), time.time(), value, 1, 0), window=10_000)


def _registry(ran: list) -> dict:
    async def execute(item):
        ran.append(item.source)
        return {"summary": "ran"}

    return {"long": SimpleNamespace(NAME="long", LONG_LIVED=True, execute=execute),
            "short": SimpleNamespace(NAME="short", execute=execute)}


def _wire(monkeypatch, registry: dict, gate_cfg: dict | None = None) -> None:
    import workers.sources as sources
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", registry, raising=False)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {n: {"max_duration_seconds": 30} for n in registry},
                        raising=False)
    monkeypatch.setattr("workers.pool.kv_gate_config",
                        lambda: gate_cfg if gate_cfg is not None
                        else {"enabled": True, "max_kv_usage": 0.6})


async def _run_for(q, seconds: float = 0.3) -> dict:
    pool = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await pool.start()
    try:
        await asyncio.sleep(seconds)
        return pool.status()
    finally:
        await pool.stop()


# ── the queue ─────────────────────────────────────────────────────────


def test_an_excluded_source_is_skipped_in_sql_not_in_a_window(q):
    """Sixty held rows at a better priority, one claimable row behind them."""
    for _ in range(60):
        q.enqueue("long", "k", priority=10)
    q.enqueue("short", "k", priority=90)
    item = q.claim_next("w", {}, ["long"])
    assert item is not None and item.source == "short"


def test_exclusion_and_saturation_combine(q):
    q.enqueue("long", "k", priority=10)
    q.enqueue("busy", "k", priority=20)
    q.enqueue("busy", "k", priority=20)
    q.enqueue("short", "k", priority=90)
    assert q.claim_next("w", {"busy": 1}, ["long"]).source == "busy"
    # busy is now at its quota; long is held — only short is left.
    assert q.claim_next("w", {"busy": 1}, ["long"]).source == "short"


# ── the classification ────────────────────────────────────────────────


def test_long_lived_is_a_declared_attribute():
    reg = {"a": SimpleNamespace(LONG_LIVED=True), "b": SimpleNamespace(),
           "c": SimpleNamespace(LONG_LIVED=False)}
    assert long_lived_sources(reg) == ["a"]


def test_the_real_sources_declare_what_the_plan_names():
    import workers.sources as sources
    assert set(long_lived_sources(sources.SOURCE_REGISTRY)) == {
        "autocode", "autotriage", "deep-research"}


# ── the pool ──────────────────────────────────────────────────────────


async def test_over_budget_holds_long_lived_and_lets_short_lived_through(q, monkeypatch):
    ran: list = []
    q.enqueue("long", "k", priority=10)
    q.enqueue("short", "k", priority=90)
    _wire(monkeypatch, _registry(ran))
    _kv(0.72)
    status = await _run_for(q)
    assert ran == ["short"]
    held = q.get(1)
    assert held.state == "queued" and held.attempts == 0, "a hold spent an attempt"
    gate = status["kv_gate"]
    assert gate["engaged"] is True
    assert gate["held_sources"] == ["long"]
    assert gate["engagements"] == 1
    assert gate["kv_usage"] == pytest.approx(0.72)


async def test_under_budget_everything_claims(q, monkeypatch):
    ran: list = []
    q.enqueue("long", "k", priority=10)
    q.enqueue("short", "k", priority=90)
    _wire(monkeypatch, _registry(ran))
    _kv(0.41)
    status = await _run_for(q)
    assert sorted(ran) == ["long", "short"]
    assert status["kv_gate"]["engaged"] is False


async def test_no_reading_means_the_gate_is_open(q, monkeypatch):
    ran: list = []
    q.enqueue("long", "k")
    _wire(monkeypatch, _registry(ran))
    status = await _run_for(q)
    assert ran == ["long"]
    assert status["kv_gate"]["kv_usage"] is None


async def test_the_gate_can_be_switched_off(q, monkeypatch):
    ran: list = []
    q.enqueue("long", "k")
    _wire(monkeypatch, _registry(ran), gate_cfg={"enabled": False})
    _kv(0.95)
    await _run_for(q)
    assert ran == ["long"]


def _series(values: list[float], spacing_s: float = 5.0) -> None:
    """Samples ending now, oldest first, `spacing_s` apart."""
    now = time.monotonic()
    for i, v in enumerate(values):
        t = now - spacing_s * (len(values) - 1 - i)
        ep.record(ep.Sample(t, t, v, 1, 0), window=10_000)


async def test_a_cold_prefill_spike_does_not_engage_the_gate(q, monkeypatch):
    """Measured 2026-09-10: a cold 200k prefill takes the gauge from 0.20 to
    0.96 for ~21 s and drops to 0.50 the moment the prompt is in. The last
    sample would hold every long-lived job for the length of every prefill;
    the one-minute median sees the residents."""
    ran: list = []
    q.enqueue("long", "k")
    _wire(monkeypatch, _registry(ran))
    _series([0.30] * 8 + [0.75, 0.88, 0.96])
    status = await _run_for(q)
    assert ran == ["long"]
    assert status["kv_gate"]["engaged"] is False


async def test_sustained_pressure_does_engage_it(q, monkeypatch):
    ran: list = []
    q.enqueue("long", "k")
    _wire(monkeypatch, _registry(ran))
    _series([0.66] * 12)
    status = await _run_for(q)
    assert ran == []
    assert status["kv_gate"]["engaged"] is True
    assert status["kv_gate"]["window_seconds"] == pytest.approx(60.0)


async def test_the_held_item_runs_once_pressure_falls(q, monkeypatch):
    ran: list = []
    q.enqueue("long", "k")
    _wire(monkeypatch, _registry(ran))
    _kv(0.9)
    pool = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await pool.start()
    try:
        await asyncio.sleep(0.2)
        assert ran == [] and pool.status()["kv_gate"]["engaged"] is True
        ep.reset()          # the one-minute median needs the old reading gone
        _kv(0.3)
        for _ in range(100):
            if ran:
                break
            await asyncio.sleep(0.02)
        status = pool.status()
    finally:
        await pool.stop()
    assert ran == ["long"]
    assert status["kv_gate"]["engaged"] is False
    assert status["kv_gate"]["held_sources"] == []
