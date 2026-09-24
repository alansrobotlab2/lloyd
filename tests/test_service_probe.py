"""#1359 clause 3: a crash loop supervisord calls RUNNING surfaces unasked.

The 2026-09-21 shape — djev spawned, declared RUNNING at +30 s, died at +65 to
+155 s, respawned, nine times — replayed against `ServiceProbe` with a fake
clock, a fake supervisord table and a fake port probe. Nothing here touches a
socket, supervisord or the notify fan-out.
"""
from __future__ import annotations

import asyncio

import pytest

from workers import service_probe as sp

DJEV = {"agent-djev": ("djev (DiffusionGemma)", 8011)}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _probe():
    calls: list[tuple[str, str, str]] = []
    clock = Clock()
    probe = sp.ServiceProbe(announce=lambda *a: calls.append(a) or {"desktop": True},
                            clock=clock)
    return probe, clock, calls


def _procs(state: str) -> dict:
    return {"agent-djev": {"name": "agent-djev", "statename": state}}


CLOSED = lambda port: False  # noqa: E731
OPEN = lambda port: True  # noqa: E731


def test_a_crash_loop_through_respawns_is_announced_once():
    probe, clock, calls = _probe()
    # 60 s ticks over the 57-minute incident; the tick lands on every phase of
    # the cycle, including the EXITED/STARTING/BACKOFF instants between spells.
    cycle = ["STARTING", "RUNNING", "RUNNING", "EXITED", "BACKOFF"]
    for i in range(57):
        probe.tick(DJEV, _procs(cycle[i % len(cycle)]), CLOSED)
        clock.t += 60
    assert len(calls) == 1, calls
    title, body, level = calls[0]
    assert level == "critical"
    assert ":8011" in title and "agent-djev" in body


def test_the_streak_is_not_reset_by_a_respawn():
    """The anti-vacuity half: if EXITED ended the streak, a tick landing there
    every few minutes would keep the loop below the grace forever."""
    probe, clock, calls = _probe()
    for i in range(40):
        state = "EXITED" if i % 4 == 3 else "RUNNING"
        probe.tick(DJEV, _procs(state), CLOSED)
        clock.t += 60
    assert len(calls) == 1


def test_a_cold_boot_that_opens_its_port_inside_the_grace_is_silent():
    probe, clock, calls = _probe()
    for _ in range(10):  # ten minutes of compile, then serving
        probe.tick(DJEV, _procs("RUNNING"), CLOSED)
        clock.t += 60
    for _ in range(30):
        probe.tick(DJEV, _procs("RUNNING"), OPEN)
        clock.t += 60
    assert calls == []
    assert probe.streaks == {}


def test_a_deliberate_stop_is_not_an_outage():
    probe, clock, calls = _probe()
    for _ in range(60):
        probe.tick(DJEV, _procs("STOPPED"), CLOSED)
        clock.t += 60
    assert calls == []


def test_recovery_is_announced_once_after_an_outage():
    probe, clock, calls = _probe()
    for _ in range(20):
        probe.tick(DJEV, _procs("RUNNING"), CLOSED)
        clock.t += 60
    for _ in range(5):
        probe.tick(DJEV, _procs("RUNNING"), OPEN)
        clock.t += 60
    assert [c[2] for c in calls] == ["critical", "info"]
    assert "again" in calls[1][0]


def test_the_primary_gets_its_boot_time_before_it_is_called_down():
    probe, clock, calls = _probe()
    svc = {"agent-llm-primary": ("LLM Primary", 8096)}
    procs = {"agent-llm-primary": {"statename": "RUNNING"}}
    for _ in range(20):  # a legitimate 20-minute boot
        probe.tick(svc, procs, CLOSED)
        clock.t += 60
    assert calls == []
    for _ in range(15):
        probe.tick(svc, procs, CLOSED)
        clock.t += 60
    assert len(calls) == 1


def test_a_service_without_a_port_or_switched_off_is_never_judged():
    probe, clock, calls = _probe()
    for _ in range(60):
        probe.tick({"agent-tts": ("TTS", None)},
                   {"agent-tts": {"statename": "RUNNING"}, **_procs("RUNNING")},
                   CLOSED)
        clock.t += 60
    assert calls == [] and probe.streaks == {}


def test_an_unreachable_supervisord_skips_the_tick(monkeypatch):
    from app import supervisor_client as sc

    def _down():
        raise sc.SupervisordUnreachable("no socket")

    monkeypatch.setattr(sc, "_supervisor_all", _down)
    probe, clock, calls = _probe()
    probe.streaks["agent-djev"] = sp._Streak(since=0.0, state="RUNNING")
    assert sp.run_probe(probe) == []
    assert calls == [] and "agent-djev" in probe.streaks


def test_an_announce_that_raises_does_not_break_the_probe():
    def boom(*a):
        raise RuntimeError("no bus")

    clock = Clock()
    probe = sp.ServiceProbe(announce=boom, clock=clock)
    for _ in range(20):
        events = probe.tick(DJEV, _procs("RUNNING"), CLOSED)
        clock.t += 60
        if events:
            break
    assert events and events[0]["kind"] == "down"


def test_the_scheduler_loop_runs_the_probe_unattended(monkeypatch, tmp_path):
    """The seat: the pool's scheduler loop, not a skill a person invokes."""
    from workers.pool import WorkerPool
    from workers.queue import WorkQueue

    ran: list[object] = []
    monkeypatch.setattr(sp, "run_probe", lambda probe: ran.append(probe) or [])
    import workers.sources as sources
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", {}, raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {}, raising=False)

    async def go():
        pool = WorkerPool(WorkQueue(tmp_path / "q.db"), slots=1,
                          poll_idle_seconds=0.01)
        await pool.start()
        try:
            await asyncio.sleep(0.3)
        finally:
            await pool.stop()
        return pool

    pool = asyncio.run(go())
    assert ran and isinstance(ran[0], sp.ServiceProbe)
    assert ran[0].announce is sp.guardian_announce
    assert pool._service_probe is ran[0]


@pytest.mark.parametrize("fail", [RuntimeError("boom")])
def test_a_failing_probe_never_takes_the_scheduler_down(monkeypatch, tmp_path, fail):
    from workers.pool import WorkerPool
    from workers.queue import WorkQueue

    def _raise(probe):
        raise fail

    monkeypatch.setattr(sp, "run_probe", _raise)
    pool = WorkerPool(WorkQueue(tmp_path / "q.db"), slots=1)
    asyncio.run(pool._probe_services())  # must not raise
