"""The stop controls are fired at a synthetic run and judged by what they stop (#703).

`scripts/mitigation_drill.py` is offline and in-process: the real session turn
queue and the real cancel route handler, a real `WorkerPool` over a scratch
queue. What these tests pin is the drill's honesty — a control that does
nothing fails it, and a pause is never reported as an in-flight stop.
"""
from __future__ import annotations

import inspect

import pytest

from app import mitigation_state
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
    latest = after["session_cancel"]
    assert (latest["classification"], latest["seconds"], latest["at"]) == (
        "no-op", None, "later"), latest
    # #2153: a re-record is a READING, not a replacement — the earlier in-flight
    # measurement is still the only one that timed a stop, so it must still be
    # what the median is taken over.
    assert latest["n"] == 2, f"a second record left n={latest['n']}"
    assert latest["median_seconds"] == pytest.approx(
        written["session_cancel"]["seconds"]), (
        "the median is now taken over the no-op's null seconds instead of the "
        "one reading that actually measured a stop"
    )
    assert after["pool_pause"]["classification"] == "dispatch-only"


async def test_a_refused_drill_records_nothing(tmp_path):
    state = tmp_path / "mitigation_drill.json"
    await D.run({"pool": {"round_hold": {"engaged": True}}}, state_path=state)
    assert not state.exists()


# ── #2333: waiting for the round hold to release, instead of being refused by it ──
#
# The refusal above is correct for a caller that is content to be refused, and it
# is why autonomy task #95 exited 2 on three days running: the hold sat engaged
# 73-94% of each of them, so a daily caller is refused most days it tries and the
# reading series never starts. `--wait-free-window` is the caller that is not
# content — the pool's maintenance seat spawns it hourly — and what these nodes
# pin is the difference: the hold is polled until it releases, and then never
# looked at again.


def _hold(engaged: bool) -> dict:
    return {"pool": {"round_hold": {"engaged": engaged,
                                    "engaged_since": "2026-10-07T00:00:00Z"}}}


@pytest.fixture
def state_in_tmp(monkeypatch, tmp_path):
    """`main()` writes the real state file unless told otherwise, so every node
    here that calls it says where that file is instead."""
    import app.paths as paths

    monkeypatch.setattr(paths, "MITIGATION_DRILL_STATE", tmp_path / "mitigation_drill.json")
    return tmp_path / "mitigation_drill.json"


def test_the_wait_starts_the_drill_on_the_first_clean_read(monkeypatch, capsys, state_in_tmp):
    """#2333 clause 1. `--wait-free-window` re-reads the status on the poll interval
    and starts the drill the moment `refusal()` has no reason — it does NOT wait out
    the window it was handed. That is the point: with the hold engaged 73-94% of a
    busy day, a caller that slept to the end of its window would be firing into a
    hold anyway.

    The third read here says the hold re-engaged and the drill runs regardless —
    clause 3, measured at the same seam as clause 1.
    """
    import json

    reads, sleeps = [], []
    clock = {"t": 0.0}

    def fake_status(url, timeout=3.0):
        reads.append(url)
        if len(reads) <= 2:
            return _hold(True)
        if len(reads) == 3:
            return _hold(False)
        return _hold(True)          # must never be asked a fourth time

    def fake_sleep(s):
        sleeps.append(s)
        clock["t"] += s

    monkeypatch.setattr(D, "live_status", fake_status)
    monkeypatch.setattr(D, "_sleep", fake_sleep)
    monkeypatch.setattr(D, "_monotonic", lambda: clock["t"])

    assert D.main(["--wait-free-window", "600"]) == 0

    assert len(reads) == 3, (
        f"the drill read the status {len(reads)} times: a clean read must start the "
        f"drill, and nothing may re-read the hold once it has started")
    assert sleeps == [D.FREE_WINDOW_POLL_S, D.FREE_WINDOW_POLL_S], (
        f"polls were {sleeps}, expected the {D.FREE_WINDOW_POLL_S}s interval twice — "
        f"the interval is the unit the wait measures its window in")

    out = json.loads(capsys.readouterr().out)
    assert [s["surface"] for s in out["surfaces"]] == ["session_cancel", "pool_pause"]
    assert out["ok"] is True
    assert state_in_tmp.exists(), (
        "a drill that fired but wrote no reading leaves the series exactly as empty "
        "as the refusal it replaced")


def test_the_wait_exits_2_only_when_its_window_runs_out(monkeypatch, capsys, state_in_tmp):
    """#2333 clause 1's other half: exit 2 is the window EXPIRING, not the first
    refusal. So a `--wait-free-window 60` spends its whole 60 s — four 15 s polls and
    five reads — before anything is printed, and no surface is touched on the way
    out. The exit code and the `refused` text are the ones
    `tests/test_mitigation_drill_task.py` pins for the no-flag caller, unchanged."""
    import json

    reads, sleeps = [], []
    clock = {"t": 0.0}

    def fake_status(url, timeout=3.0):
        reads.append(url)
        return _hold(True)

    def fake_sleep(s):
        sleeps.append(s)
        clock["t"] += s

    async def must_not_run(status, **kw):
        raise AssertionError("the drill ran with the hold engaged")

    monkeypatch.setattr(D, "live_status", fake_status)
    monkeypatch.setattr(D, "_sleep", fake_sleep)
    monkeypatch.setattr(D, "_monotonic", lambda: clock["t"])
    monkeypatch.setattr(D, "run", must_not_run)

    assert D.main(["--wait-free-window", "60"]) == 2

    assert sleeps == [D.FREE_WINDOW_POLL_S] * 4, (
        f"a 60 s window spent {sleeps}: a short final poll is fine, but a window "
        f"shortened by rounding is a window the caller did not ask for")
    assert len(reads) == 5, f"{len(reads)} reads for 4 sleeps of 15 s"
    cap = capsys.readouterr()
    assert "refused" in cap.err and "round_hold" in cap.err
    assert json.loads(cap.out)["waited_seconds"] == 60
    assert not state_in_tmp.exists()


@pytest.mark.parametrize("argv", [[], ["--wait-free-window", "0"]])
def test_without_the_flag_one_read_still_decides_it(monkeypatch, argv, state_in_tmp):
    """#2333 clause 2: the default is today's behaviour, not a zero-second wait. With
    the flag absent, or explicitly 0, exactly one status read happens and a
    `round_hold` refusal exits 2 immediately — no poll loop, because a caller that
    asked to be refused out of the way must not hold a worker slot for an hour to
    learn what its first read already told it. Task #95's verbatim command is this
    branch, and `tests/test_mitigation_drill_task.py` pins it as written."""
    reads, sleeps = [], []
    monkeypatch.setattr(D, "live_status",
                        lambda url, timeout=3.0: reads.append(url) or _hold(True))
    monkeypatch.setattr(D, "_sleep", lambda s: sleeps.append(s))

    assert D.main(argv) == 2
    assert len(reads) == 1
    assert sleeps == []


async def test_the_hold_is_never_re_checked_once_the_drill_started(monkeypatch, tmp_path):
    """#2333 clause 3, at the seam that matters. The free gap this fires into can be
    narrower than the drill itself — 0.6 and 0.7 minutes measured, against a drill
    that takes seconds — so once `run()` holds a status that does not refuse, going
    to ask again would throw away the measurement to guard a condition the drill is
    built to survive: both surfaces are in-process and offline, and the state-file
    merge is additive."""
    def boom(url, timeout=3.0):
        raise AssertionError("the drill re-read the status after the surfaces started")

    monkeypatch.setattr(D, "live_status", boom)
    report = await D.run({"pool": {"round_hold": {"engaged": False}}},
                         state_path=tmp_path / "state.json")
    assert report["ok"] is True
    assert [s["surface"] for s in report["surfaces"]] == ["session_cancel", "pool_pause"]
    assert (tmp_path / "state.json").exists()


def test_the_wait_reads_through_the_module_names_a_test_can_patch():
    """The wait must reach `live_status`, `_sleep` and `_monotonic` BY NAME on this
    module. Patch-at-the-source is the only honest way to watch an hour of polling in
    a test, and a `status_fn=live_status` default would bind the function at def time
    and be silently unpatchable — the seam #2039 caught in `app/engine`, where the
    tell was a test that passed by skipping the branch it was written to exercise.
    `inspect.getsource` is the rail here because the two forms differ in their
    DEFAULTS, which a test that passes its own fakes never sees."""
    src = inspect.getsource(D.wait_for_free_window)
    assert "status_fn or live_status" in src
    assert "sleep or _sleep" in src
    assert "clock or _monotonic" in src


# ── #2153: a bounded history per surface, and the median the route reports ──
#
# `record()` merged one dict per surface and overwrote it, so after a hundred
# drills the state file held a hundred measurements' worth of the LAST one and
# nothing could be averaged: `/api/workers/status` reported a single run's
# stop-time as if it were the control's stop-time, and #703's owed ">=5 medians
# per surface" was unreachable whatever the scheduler did. The readings are now
# appended per surface and capped at `mitigation_state.HISTORY_CAP` entries —
# oldest dropped, so the file cannot grow without bound on a box that runs the
# drill daily — while `classification`/`seconds`/`at` keep holding the latest
# reading, which is what the route's existing readers (and the operator's eye)
# were already reading.

def _record(state, surface, seconds, at, *, classification="in-flight", ok=True):
    mitigation_state.record([{"surface": surface, "classification": classification,
                              "seconds": seconds, "ok": ok}], path=state, at=at)


def _history(state, surface):
    import json

    return json.loads(state.read_text())["surfaces"][surface]["history"]


def test_record_appends_the_reading_and_leaves_the_latest_ones_alone(tmp_path):
    """#2153 clause 1 — both halves.

    The history grows by the reading, and the three keys the route already
    reads keep holding the newest measurement rather than the first or an
    average, so `app/routers/workers.py:153` needs no change to stay correct.
    """
    state = tmp_path / "mitigation_drill.json"
    for seconds, at in ((1.0, "first"), (2.0, "second"), (3.0, "third")):
        _record(state, "session_cancel", seconds, at)
    for at in ("first", "second"):
        _record(state, "pool_pause", None, at, classification="dispatch-only")

    latest = mitigation_state.read(path=state)["session_cancel"]
    assert (latest["classification"], latest["seconds"], latest["at"]) == (
        "in-flight", 3.0, "third"
    ), f"the top-level keys moved off the latest reading: {latest}"
    assert [r["seconds"] for r in _history(state, "session_cancel")] == [1.0, 2.0, 3.0]
    assert [r["at"] for r in _history(state, "pool_pause")] == ["first", "second"], (
        "history is per surface: measuring one control must not append to the "
        "other's series, the same reason the merge never dropped a surface"
    )

    import json

    assert set(json.loads(state.read_text())) == {"surfaces"}, (
        "a top-level key beside `surfaces` is a second shape the route's "
        "readers would have to know about"
    )
    assert all("history" not in r for r in mitigation_state.read(path=state).values()), (
        "the whole history belongs in the state file, not in the status payload: "
        "the route is a hot read and 20 readings per surface is a per-request cost"
    )


def test_six_readings_report_their_median_and_a_21st_drops_the_oldest(tmp_path):
    """#2153 clause 2 — the median is over the recorded readings, and the cap
    is what lets a daily drill run for a year without the file growing.

    The six are 1.0 … 6.0 s, so the median is 3.5 and not the 6.0 the file's
    top-level `seconds` holds: the two numbers answer different questions and
    the route now publishes both.
    """
    state = tmp_path / "mitigation_drill.json"
    for i in range(1, 7):
        _record(state, "session_cancel", float(i), f"r{i}")

    six = mitigation_state.read(path=state)["session_cancel"]
    assert six["median_seconds"] == 3.5, (
        f"the median of 1..6 s came back {six['median_seconds']}"
    )
    assert six["n"] == 6 and six["seconds"] == 6.0

    for i in range(7, 22):
        _record(state, "session_cancel", float(i), f"r{i}")

    hist = _history(state, "session_cancel")
    assert len(hist) == mitigation_state.HISTORY_CAP == 20, (
        f"a 21st reading left {len(hist)} entries; the cap is what bounds the file"
    )
    assert [r["seconds"] for r in hist] == [float(i) for i in range(2, 22)], (
        "the oldest reading is the one that goes, so the median stays the median "
        "of the most recent cap readings"
    )
    assert hist[0]["at"] == "r2"
    dropped = mitigation_state.read(path=state)["session_cancel"]
    assert dropped["n"] == 20 and dropped["median_seconds"] == 11.5, (
        f"after the cap the aggregate should be over r2..r21 (median 11.5), got {dropped}"
    )


def test_read_publishes_median_and_n_per_surface_and_still_says_never_run(tmp_path):
    """#2153 clause 3 — three shapes of the same read.

    Exactly `{"state": "never-run"}` with no file, because `tests/test_workers_pool.py`
    asserts that marker and an operator reading `n: 0` would be reading a
    measurement that was never taken; one reading reporting its own seconds
    with `n: 1`, because that is the state every series starts in and the
    route test's fixture is exactly one `record()`; and a file written by the
    previous writer, which has no `history` key at all, still reporting n=1
    rather than n=0.
    """
    absent = tmp_path / "absent" / "mitigation_drill.json"
    assert mitigation_state.read(path=absent) == {"state": "never-run"}

    empty = tmp_path / "empty.json"
    empty.write_text('{"surfaces": {}}', encoding="utf-8")
    assert mitigation_state.read(path=empty) == {"state": "never-run"}, (
        "a state file holding no surface is the same fact as no file at all: an "
        "operator must not read an empty dict where the never-run marker lives"
    )

    state = tmp_path / "one.json"
    _record(state, "session_cancel", 0.042, "only")
    assert mitigation_state.read(path=state)["session_cancel"] == {
        "classification": "in-flight", "seconds": 0.042, "at": "only",
        "median_seconds": 0.042, "n": 1,
    }, "a one-reading series must report its own stop-time as its median"

    _record(state, "pool_pause", None, "only", classification="dispatch-only")
    pause = mitigation_state.read(path=state)["pool_pause"]
    assert pause["median_seconds"] is None and pause["n"] == 1, (
        "a dispatch-only surface has no stop-time to median; null is the honest "
        "answer, and n still counts the reading so nobody reads it as never run"
    )

    legacy = tmp_path / "legacy.json"
    legacy.write_text('{"surfaces": {"session_cancel": {"classification": '
                      '"in-flight", "seconds": 0.07, "ok": true, "at": "old"}}}')
    old = mitigation_state.read(path=legacy)["session_cancel"]
    assert old["n"] == 1 and old["median_seconds"] == 0.07, (
        f"a file written before history existed reads as {old}: the reading it "
        "does hold must count, or the route goes quiet on a control it measured"
    )
