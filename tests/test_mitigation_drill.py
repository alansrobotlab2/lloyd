"""The stop controls are fired at a synthetic run and judged by what they stop (#703).

`scripts/mitigation_drill.py` is offline and in-process: the real session turn
queue and the real cancel route handler, a real `WorkerPool` over a scratch
queue. What these tests pin is the drill's honesty — a control that does
nothing fails it, and a pause is never reported as an in-flight stop.
"""
from __future__ import annotations

from scripts import mitigation_drill as D


async def _noop_cancel(session_id: str) -> None:
    return None


async def test_session_cancel_stops_the_running_turn_and_is_timed():
    r = await D.drill_session_cancel()
    assert r["classification"] == "in-flight"
    assert r["ok"] is True
    assert r["seconds"] is not None and 0 <= r["seconds"] < 1.0


async def test_a_no_op_cancel_handler_fails_the_surface():
    r = await D.drill_session_cancel(_noop_cancel, timeout_s=0.5)
    assert r["classification"] == "no-op"
    assert r["ok"] is False


async def test_pool_pause_stops_claims_but_not_the_run_in_flight():
    """Dispatch-only is the design, not the defect: the landing pauses the pool
    and then waits for what is in flight. The drill must say so, and must not
    count it among the in-flight stops."""
    r = await D.drill_pool_pause(probe_s=0.5)
    assert r["classification"] == "dispatch-only"
    assert r["ok"] is True
    assert r["seconds"] is None


async def test_a_no_op_pause_fails_the_surface():
    r = await D.drill_pool_pause(lambda pool: None, probe_s=0.5)
    assert r["classification"] == "no-op"
    assert r["ok"] is False


async def test_the_report_keeps_dispatch_only_out_of_the_in_flight_stops(tmp_path):
    report = await D.run(None, state_path=tmp_path / "state.json")
    assert report["ok"] is True
    assert [s["surface"] for s in report["in_flight"]] == ["session_cancel"]
    assert report["dispatch_only"] == ["pool_pause"]


def test_a_broken_control_makes_the_drill_exit_non_zero(monkeypatch, tmp_path):
    async def _broken(status=None, **kw):
        return await _real_run(status, session_cancel=_noop_cancel,
                               state_path=tmp_path / "state.json")

    _real_run = D.run
    monkeypatch.setattr(D, "live_status", lambda url: None)
    monkeypatch.setattr(D, "run", _broken)
    assert D.main([]) == 1


def test_it_refuses_while_a_round_holds_the_pool_and_says_why():
    status = {"pool": {"round_hold": {"engaged": True,
                                      "engaged_since": "2026-09-24T01:00:00Z"}}}
    reason = D.refusal(status)
    assert "self-mod round" in reason and "2026-09-24T01:00:00Z" in reason
    assert D.refusal({"pool": {"round_hold": {"engaged": False}}}) is None
    assert D.refusal(None) is None


async def test_a_refused_drill_fires_nothing(monkeypatch):
    fired = []

    async def _spy(session_id):
        fired.append(session_id)

    report = await D.run({"pool": {"round_hold": {"engaged": True}}},
                         session_cancel=_spy, pool_pause=lambda p: fired.append(p))
    assert report["refused"] and report["surfaces"] == [] and fired == []
    assert report["ok"] is False


async def test_a_drill_records_each_surface_for_the_status_route(tmp_path):
    """Clause 5's writer: the latest per-surface result lands on disk, and a
    later drill merges in rather than dropping a surface it did not re-measure."""
    import json

    from app import mitigation_state

    state = tmp_path / "state" / "mitigation_drill.json"
    await D.run(None, state_path=state)
    written = json.loads(state.read_text())["surfaces"]
    assert written["session_cancel"]["classification"] == "in-flight"
    assert 0 <= written["session_cancel"]["seconds"] < 1.0
    assert written["pool_pause"]["classification"] == "dispatch-only"
    assert written["pool_pause"]["seconds"] is None
    assert all(r["at"] for r in written.values())

    mitigation_state.record([{"surface": "session_cancel", "classification": "no-op",
                              "seconds": None, "ok": False}], path=state, at="later")
    after = mitigation_state.read(path=state)
    assert after["session_cancel"] == {"classification": "no-op", "seconds": None,
                                       "at": "later"}
    assert after["pool_pause"]["classification"] == "dispatch-only"


async def test_a_refused_drill_records_nothing(tmp_path):
    state = tmp_path / "mitigation_drill.json"
    await D.run({"pool": {"round_hold": {"engaged": True}}}, state_path=state)
    assert not state.exists()
