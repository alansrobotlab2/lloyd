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
        "median_seconds": 0.042, "n": 1, "spaced_firings": 1,
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


# ── #2431: whose reading is this, and how many firings is that ──────────────
#
# `n: 20` per surface, every stamp inside 17:17:34→17:17:44+00:00, against two
# logged seat spawns (`server.err`: 17:04:39Z → rc=0 at 17:06:07Z, and 18:05:17Z
# with no exit line). A count of readings is not a count of drillings, and
# nothing published — and nothing in the file — could tell a ten-second burst
# from twenty hourly firings. Attribution is now in the bytes (pid per reading,
# command line from the CLI) and `spaced_firings` is the published count of
# firings the readings could evidence.

def test_every_reading_carries_the_pid_that_wrote_it(tmp_path):
    """#2431 clause 1 — twenty readings from one process, readable off the file.

    The burst, re-made: one process, twenty `record()` calls, every stamp inside
    ten seconds. Before this change the file's only answer to "how many drillings
    produced these?" was `n: 20`, and the correct answer here is 1. Now one pid
    appears twenty times, which is the finding, and no log archaeology is needed
    to reach it. The same pid goes on the surface's top-level keys, because that
    is the copy the pre-#2153 readers look at.
    """
    import json
    import os

    state = tmp_path / "mitigation_drill.json"
    for i in range(20):
        _record(state, "session_cancel", float(i + 1),
                f"2026-10-08T17:17:{34 + i // 2:02d}+00:00")

    hist = _history(state, "session_cancel")
    assert len(hist) == 20 and all(r["pid"] == os.getpid() for r in hist), (
        f"{len(hist)} readings, pids "
        f"{sorted({r.get('pid') for r in hist})}; this process wrote all of them, "
        "so the file must say so twenty times over"
    )
    top = json.loads(state.read_text())["surfaces"]["session_cancel"]
    assert top["pid"] == os.getpid(), (
        "the latest-reading copy beside classification/seconds/at carries the "
        "same attribution as the history entry it duplicates"
    )


def test_the_cli_records_the_invocation_that_wrote_it(tmp_path):
    """#2431 clause 1+2 across a real process boundary — the seat's spawn in a child.

    A `python -m scripts.mitigation_drill` child (the maintenance seat's own
    argv shape, pinned by
    `tests/test_workers_maintenance.py::test_the_argv_is_the_module_invocation_with_a_bounded_wait`)
    drills into the state file that this test's own in-process `run()` already
    wrote, through `LLOYD_DATA` — the env var `app.paths` honours, and the reason
    this is safe to run against the real module default.

    The two readings are then told apart by what #2431 could not: the child's
    carries its pid and the command line that launched it, the in-process one
    carries this pid and a null invocation. `spaced_firings` is 1 for the pair —
    two processes a second apart are one cluster by the spacing rule — which is
    exactly why the two fields are complements: spacing says how many firings the
    clock can evidence, the pid and invocation say which processes actually
    wrote. Python rewrites `argv[0]` to the module's FILE path for a `-m` launch,
    so the recorded string names `mitigation_drill.py`, and it names the checkout
    the drill ran from — a candidate venv and the production backend share one
    state file, and that is the half the item needed most.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    state = tmp_path / "mitigation_drill.json"
    repo_root = Path(D.__file__).resolve().parents[1]
    import asyncio
    asyncio.run(D.run(None, state_path=state))

    argv = [sys.executable, "-m", "scripts.mitigation_drill",
            "--status-url", "http://127.0.0.1:9/x"]
    # Output to FILES, not pipes, and in its own process group. A pipe was the
    # first shape and it stalled the node: the child printed its entire report
    # (visible in the TimeoutExpired payload) and the pipe then never reached EOF,
    # so `communicate` spent its whole timeout waiting on a writer with nothing
    # left to say. What this node needs is the exit code and the state file, and
    # a file-backed redirect asks for neither of the two things a pipe couples it
    # to. The seat itself already redirects both to DEVNULL.
    #
    # `Popen`/`wait` rather than `subprocess.run` because the pid IS the claim:
    # `CompletedProcess` does not carry one, and asserting the reading names the
    # writing process's pid against a pid the test never held would pin nothing.
    log = tmp_path / "cli.log"
    with open(log, "wb") as fp:
        proc = subprocess.Popen(argv, cwd=str(repo_root), stdin=subprocess.DEVNULL,
                                stdout=fp, stderr=fp, start_new_session=True,
                                env={**os.environ, "LLOYD_DATA": str(tmp_path)})
        try:
            rc = proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError(
                f"the CLI child {proc.pid} never exited. log: "
                f"{log.read_text()[-2000:]}") from None
    assert rc == 0, f"the CLI exited {rc}.\nlog: {log.read_text()[-2000:]}"

    hist = _history(state, "session_cancel")
    assert len(hist) == 2, f"expected the in-process reading plus the child's: {hist}"
    mine, child = hist
    assert mine["pid"] == os.getpid() and mine["invocation"] is None, (
        "an in-process caller of run() that passed no invocation must record a "
        f"null one, which is itself the answer: {mine}"
    )
    assert child["pid"] == proc.pid and child["pid"] != os.getpid(), (
        f"the child wrote {child}; expected its own pid {proc.pid}, not the "
        "test process's — a reading whose pid is the reader's is the misattribution "
        "#2431 is about"
    )
    assert child["invocation"] and "mitigation_drill" in child["invocation"] \
            and str(repo_root) in child["invocation"] \
            and "--status-url" in child["invocation"], (
        f"the CLI must record the command line that wrote its readings, got "
        f"{child['invocation']!r}: the module invocation, the checkout it ran "
        "from and the flags it was given are the three things that told a seat "
        "firing from an ad-hoc caller, and none of them were recoverable after "
        "the fact"
    )
    assert mitigation_state.read(path=state)["session_cancel"]["n"] == 2
    assert mitigation_state.read(path=state)["session_cancel"]["spaced_firings"] == 1, (
        "two writings one second apart are one cluster by the spacing rule even "
        "though two processes did them: spacing counts what the clock can "
        "separate, and only the pid separates these"
    )
    assert json.loads(state.read_text())["surfaces"]["pool_pause"]["history"][-1]["pid"] \
        == proc.pid, (
        "attribution goes on every surface one drill measures, not just the one "
        "with a stop-time — the burst was 20 readings on BOTH surfaces"
    )


def test_twenty_readings_inside_ten_seconds_are_one_firing_not_twenty(tmp_path):
    """#2431 clause 3 — the published count that a burst cannot satisfy.

    The file found on 2026-10-08 re-made: 20 readings per surface, stamps spanning
    10 s, seconds 1.0…20.0 so the median has a known value. `n` says 20 and must
    keep saying it — those are the readings that were taken, and HISTORY_CAP is
    still the 20 the cap test pins. `spaced_firings` says 1, which is the answer
    the owed check actually wanted. `median_seconds` is unchanged at 10.5: the
    new field is additive, and a slow median is still a slow median however the
    readings arrived.
    """
    state = tmp_path / "burst.json"
    for i in range(20):
        _record(state, "session_cancel", float(i + 1),
                f"2026-10-08T17:17:{34 + i // 2:02d}+00:00")

    got = mitigation_state.read(path=state)["session_cancel"]
    assert got["n"] == 20 == mitigation_state.HISTORY_CAP, (
        f"the burst filled the cap and `n` is still the readings on disk: {got}"
    )
    assert got["spaced_firings"] == 1, (
        f"20 readings inside ten seconds reported {got['spaced_firings']} "
        "firings; the whole point of the field is that this number is 1"
    )
    assert got["median_seconds"] == 10.5, (
        f"the median moved with the new field: {got['median_seconds']}"
    )
    assert 10.0 < mitigation_state.SPACING_GAP_S < 3600, (
        f"SPACING_GAP_S={mitigation_state.SPACING_GAP_S}: it has to exceed the "
        "ten-second span of the burst it must count as one, and stay under the "
        "seat's hourly interval or it merges genuine hourly firings into one"
    )


def test_the_firing_count_rises_only_as_readings_land_further_apart(tmp_path):
    """#2431 clause 3, the other direction — it is a spacing count, not an age count.

    Four readings ten seconds apart, then one an hour later, then one an hour
    after that: 6 readings, 3 firings. Same file, same `n`, and the count moves
    only because the stamps moved. A run of readings whose `at` will not parse
    (the pre-#2153 file wrote bare words) reports the one firing the file at
    least evidences rather than 0, because `n: 5, spaced_firings: 0` reads as
    "measured five times, caused by nothing".
    """
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 10, 8, 17, 17, 34, tzinfo=timezone.utc)
    stamps = [0, 10, 20, 30, 3600, 7200]
    state = tmp_path / "mixed.json"
    for off in stamps:
        _record(state, "session_cancel", 0.2,
                (base + timedelta(seconds=off)).isoformat(timespec="seconds"))

    got = mitigation_state.read(path=state)["session_cancel"]
    assert (got["n"], got["spaced_firings"]) == (6, 3), (
        f"six stamps at {stamps}s should cluster into 3 firings, got {got}"
    )

    words = tmp_path / "words.json"
    for at in ("first", "second", "third"):
        _record(words, "session_cancel", 0.2, at)
    unparseable = mitigation_state.read(path=words)["session_cancel"]
    assert (unparseable["n"], unparseable["spaced_firings"]) == (3, 1), (
        f"three readings with no parseable clock reported {unparseable}: the "
        "honest floor is one firing evidenced, never zero"
    )


def test_newest_reading_at_is_the_seat_s_witness_and_none_when_there_is_none(tmp_path):
    """#2431 clause 4's seam, pinned where it is defined.

    `workers/maintenance.py` asks the state module one question about a drill that
    has ended: did anything get written, and when was the newest thing there? The
    answer is the maximum over EVERY surface (the seat has no business knowing
    which control drilled) and None for no file, an unreadable file, and stamps
    that will not parse — a seat that could raise here is a seat that stops
    ticking over a malformed state file.
    """
    from datetime import datetime, timezone

    absent = tmp_path / "absent" / "mitigation_drill.json"
    assert mitigation_state.newest_reading_at(path=absent) is None

    state = tmp_path / "mitigation_drill.json"
    _record(state, "session_cancel", 0.2, "2026-10-08T17:06:07+00:00")
    _record(state, "pool_pause", None, "2026-10-08T18:51:40+00:00",
            classification="dispatch-only")
    assert mitigation_state.newest_reading_at(path=state) == datetime(
        2026, 10, 8, 18, 51, 40, tzinfo=timezone.utc), (
        "the max must be across surfaces: the seat asks whether anything was "
        "written, not whether one control was"
    )

    words = tmp_path / "words.json"
    _record(words, "session_cancel", 0.2, "old")
    assert mitigation_state.newest_reading_at(path=words) is None

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert mitigation_state.newest_reading_at(path=broken) is None
