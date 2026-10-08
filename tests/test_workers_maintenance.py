"""The maintenance seat's tenants: the poison sweep, and now the mitigation drill.

`workers/maintenance.py` runs on the pool's scheduler seat — it has to, because a
job that repairs the queue cannot be starved of a worker slot by the very backup it
is fixing. #2333 put a second tenant on that seat: spawning the mitigation drill, so
the drill fires hourly at a free moment inside a round hold instead of being refused
by it. (The drill's own honesty is `tests/test_mitigation_drill.py`; the reporter
task's prompt is `tests/test_mitigation_drill_task.py`.)

What these nodes pin is the seat's side of that: a subprocess and never an import,
one key that switches it on, one watermark that says when it last fired, and no
second drill while one is running.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from workers import maintenance as M
from workers.pool import WorkerPool
from workers.queue import WorkQueue

T0 = datetime(2026, 10, 7, 6, 0, 0, tzinfo=timezone.utc)
CFG_ON = {"mitigation_drill": {"enabled": True}}


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def q(tmp_path):
    return WorkQueue(tmp_path / "workers.db")


@pytest.fixture(autouse=True)
def no_drill_running(monkeypatch):
    """The liveness handle is process-global state. Every node starts from "no drill
    of ours is running", and `monkeypatch` puts the module's real value back after."""
    monkeypatch.setattr(M, "_drill_proc", None)


class _Proc:
    """Stand-in for the `Popen` handle the seat keeps: `poll()` answers exactly what
    the seat asks it — None while the child lives, its rc once it has exited."""

    def __init__(self, pid: int = 4242, rc=None):
        self.pid = pid
        self._rc = rc

    def poll(self):
        return self._rc


@pytest.fixture
def popen():
    """Records what the seat would have spawned, and hands back a handle.

    A parameter seam rather than a patched `subprocess.Popen`: nothing here may
    really boot a drill, which would fire both stop controls and write the real
    state file at `paths.MITIGATION_DRILL_STATE` from inside the test process.
    """
    calls = []

    def fake(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return _Proc()

    fake.calls = calls
    return fake


def _tick(queue, cfg, when=T0, popen=None):
    return M.maybe_run_mitigation_drill(queue, cfg, now=lambda: when, popen=popen)


# ── Clause 4: the drill is a subprocess, never this process ───────────────


def test_the_seat_spawns_the_drill_as_a_subprocess_not_an_import(q, popen, monkeypatch):
    """#2333 clause 4. `drill_pool_pause` points `workers.sources.SOURCE_REGISTRY` at
    a synthetic source for its duration, process-globally (`scripts/mitigation_drill.py`,
    `_synthetic_registry`). In a test process that is contained; in the backend it
    would swap the live fleet's source table out from under the scheduler loop that
    dispatches from it — the same loop this seat runs on. So: an argv, not a call. A
    subprocess pays an interpreter boot instead of trading the pool's correctness.

    The three `raise` patches are what make this a control rather than a restatement
    of the code: an edit that later "simplifies" the seat into calling the drill
    directly fails here, instead of shipping an occasional, unexplainable gap in what
    the pool believes it can run.
    """
    from scripts import mitigation_drill as drill

    async def boom(*a, **kw):
        raise AssertionError("the seat ran the drill in-process")

    monkeypatch.setattr(drill, "run", boom)
    monkeypatch.setattr(drill, "drill_pool_pause", boom)
    monkeypatch.setattr(drill, "main", boom)

    out = _tick(q, CFG_ON, popen=popen)

    assert len(popen.calls) == 1, f"the seat spawned {len(popen.calls)} drills"
    argv, kwargs = popen.calls[0]
    assert argv[0] == sys.executable, (
        f"{argv[0]!r}: the seat must run the drill out of the interpreter running it, "
        f"so a candidate venv drills its own tree and never production's"
    )
    assert argv[1:3] == ["-m", "scripts.mitigation_drill"], argv
    assert argv[3:5] == ["--wait-free-window", "3600"], (
        f"{argv[3:5]}: the whole reason the trigger moved here is that the drill waits "
        f"for a free moment instead of exiting 2 on the first look"
    )
    assert argv == M.mitigation_drill_argv(CFG_ON["mitigation_drill"]), (
        "the seat spawned something other than what `mitigation_drill_argv` builds"
    )
    assert kwargs["cwd"] == str(M._REPO_ROOT), (
        "a `-m scripts.…` run resolves its module off the cwd: the wrong cwd drills a "
        "different checkout, and writes that checkout's readings to the state file"
    )
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL, (
        "stdout is the drill's JSON report, which already has its own file; inherited, "
        "it would put a machine payload on the backend's stdout"
    )
    assert "stderr" not in kwargs, (
        "stderr must stay inherited: a refusal reason or a lost state write belongs in "
        "logs/server.err, where the round hold's own lines already are"
    )
    assert out["pid"] == 4242


def test_argv_carries_the_wait_window_the_config_asked_for():
    """The window is config, not a literal in the seat. The default is an hour — the
    size the item measured — and a box that wants a shorter one says so in
    `config.yaml` rather than editing the seat."""
    assert M.mitigation_drill_argv({"wait_free_window_seconds": 900})[4] == "900"
    assert M.mitigation_drill_argv()[4] == "3600", (
        "with no config at all the drill still gets a window: a bare argv is the "
        "no-wait form, which is the behaviour #2333 exists to replace"
    )


# ── Clause 5: the gate, the hour, the watermark, the live child ───────────


def test_the_seat_is_off_until_the_config_turns_it_on(q, popen):
    """#2333 clause 5, first gate: `workers.maintenance.mitigation_drill` defaults
    OFF, and off means off in both directions a reader cares about.

    `{"enabled": True}` on its own is the SWEEP's key, not this tenant's. That is the
    half that catches a seat reading its switch from the wrong level, which would put
    an hourly drill on every box the poison sweep is enabled on.
    """
    assert M.DEFAULTS["mitigation_drill"]["enabled"] is False

    out = _tick(q, {}, popen=popen)
    assert out == {"skipped": "disabled"}
    assert popen.calls == []
    assert q.wm_get(M.SOURCE, "last_drill_at") is None, (
        "a seat that is off left a stamp saying it ran"
    )

    assert _tick(q, {"enabled": True}, popen=popen)["skipped"] == "disabled"

    assert "spawned" in _tick(q, CFG_ON, popen=popen)


def test_the_seat_fires_once_an_hour_not_once_per_pass(q, popen, monkeypatch):
    """#2333 clause 5, the rate gate. The scheduler seat wakes every 60 s, so without
    this the drill would fire sixty times an hour against the once the item asks for.

    The middle call is the one that matters: 59 minutes in, the answer is no. The
    third is the control that says the gate is a clock and not a memory — by then the
    first drill has exited, which is why that second `spawned` is not the bug the node
    two below guards.
    """
    assert "spawned" in _tick(q, CFG_ON, popen=popen)
    monkeypatch.setattr(M, "_drill_proc", _Proc(rc=1))

    out = _tick(q, CFG_ON, when=T0 + timedelta(minutes=59), popen=popen)
    assert out["skipped"] == "not-due"
    assert out["seconds_left"] == 60, (
        f"{out}: the hour runs from the last SPAWN, so 59 minutes in there are 60 s "
        f"left — a gate that rounds this away is a 61-minute hour"
    )
    assert len(popen.calls) == 1, "the seat spawned anyway"

    assert "spawned" in _tick(q, CFG_ON, when=T0 + timedelta(hours=1), popen=popen)
    assert len(popen.calls) == 2


def test_the_seat_keeps_its_watermark_apart_from_the_sweeps(q, popen, monkeypatch):
    """#2333 clause 5, the watermark. Two tenants on one seat with different
    intervals — the sweep's 900 s, the drill's 3600 — cannot share a stamp: the faster
    tenant keeps rewriting it, the slower one reads "just ran" forever, and the result
    is a job that fires never while looking healthy in the very row that would prove
    it ran.

    Both directions, because sharing breaks either way.
    """
    q.wm_set(M.SOURCE, "last_sweep_at", (T0 - timedelta(days=3)).isoformat())

    assert "spawned" in _tick(q, CFG_ON, popen=popen)

    assert q.wm_get(M.SOURCE, "last_sweep_at") == (T0 - timedelta(days=3)).isoformat(), (
        "the drill tick rewrote the sweep's stamp: the next sweep is now three days "
        "late by the seat's own arithmetic, and nothing says so"
    )
    assert q.wm_get(M.SOURCE, "last_drill_at") == T0.isoformat()

    monkeypatch.setattr(M, "_drill_proc", _Proc(rc=1))
    out = _tick(q, CFG_ON, when=T0 + timedelta(minutes=30), popen=popen)
    assert out["skipped"] == "not-due", (
        "the drill's own fresh stamp did not hold it off, so the seat is gating on "
        "something other than `last_drill_at`"
    )


def test_the_seat_skips_while_the_drill_it_spawned_is_still_running(q, popen, monkeypatch):
    """#2333 clause 5, the live-child gate. A drill spawned with a 3600 s window can
    spend all of it waiting, and a second one beside it doubles the status polls and
    races the state file's read-merge-write — the merge is additive, so the loser's
    reading is simply gone and the series is short by one with nothing to show it.

    The skip must not re-stamp: a drill that blocked three ticks on its 50-minute wait
    must not cost the box a further hour on top of its own runtime.
    """
    q.wm_set(M.SOURCE, "last_drill_at", (T0 - timedelta(hours=2)).isoformat())
    monkeypatch.setattr(M, "_drill_proc", _Proc(pid=4242, rc=None))

    out = _tick(q, CFG_ON, popen=popen)
    assert out["skipped"] == "drill-already-running"
    assert out["pid"] == 4242
    assert popen.calls == [], "a second drill spawned beside the first"
    assert q.wm_get(M.SOURCE, "last_drill_at") == (T0 - timedelta(hours=2)).isoformat(), (
        "the skip re-stamped the hour, so the next real chance is an hour from now "
        "instead of the next tick"
    )

    monkeypatch.setattr(M, "_drill_proc", _Proc(pid=4242, rc=0))
    assert "spawned" in _tick(q, CFG_ON, popen=popen), (
        "the gate remembered that it had ever spawned, instead of asking the child "
        "whether it was alive — one long drill would then silence the seat forever"
    )


def test_a_failed_spawn_still_costs_the_hour(q, popen):
    """The stamp goes down BEFORE the spawn, deliberately, and this node pins which
    way the trade went.

    The two failures are not equal. A spawn that throws and does not stamp retries on
    the next tick, a minute away, and loops on a permanent failure — a missing
    interpreter, an EPERM — once a minute forever. A spawn that succeeds but fails to
    stamp runs a second drill beside the first, which is the exact race the node above
    exists to prevent. So: stamp first, and let an hour of silence be the price of a
    spawn that never happened.
    """
    def no_spawn(argv, **kwargs):
        raise OSError(2, "No such file or directory")

    out = _tick(q, CFG_ON, popen=no_spawn)
    assert out["skipped"].startswith("spawn-failed: FileNotFoundError"), (
        f"{out['skipped']}: the skipped reason carries the exception's own type name, "
        f"which is how a reader tells a missing interpreter from an EPERM without "
        f"having to go and reproduce it"
    )
    assert q.wm_get(M.SOURCE, "last_drill_at") == T0.isoformat()
    again = M.maybe_run_mitigation_drill(q, CFG_ON, now=lambda: T0, popen=no_spawn)
    assert again["skipped"] == "not-due"


def test_an_unreadable_watermark_is_read_as_due_not_as_never_run(q, popen):
    """A watermark that will not parse must not freeze the seat. `run_sweep` beside it
    has always treated an unparseable `last_sweep_at` as "sweep and rewrite it"; a
    tenant that read the same bad byte as "not due" would stop firing on a corrupt row
    and look idle in a way no reader could tell from an honest never-run."""
    q.wm_set(M.SOURCE, "last_drill_at", "not-a-timestamp")
    assert "spawned" in _tick(q, CFG_ON, popen=popen)
    assert q.wm_get(M.SOURCE, "last_drill_at") == T0.isoformat()


def test_the_config_ships_the_seats_block_as_a_mapping_not_a_stray_key():
    """`config.yaml`'s own shape, read the way the seat reads it.

    The gate's tests rung caught this node's bug on the first attempt: the three
    settings were written one indent level too shallow, so `mitigation_drill:` never
    existed, `enabled: true` landed beside the SWEEP's `enabled`, and
    `test_config_yaml_no_duplicate_keys.py` was the only thing that noticed. The
    seat read `{}`, fell back to the off default, and every other node in this file
    stayed green on a trigger that could never fire — a unit test hands the seat a
    dict it built itself, so nothing but a read of the shipped file tells "the key is
    off" apart from "the key is nowhere".

    The shape is pinned, not the operator's choice: `enabled` stays free to flip,
    because a test that goes red the first time somebody silences the drill is a
    false alarm rather than a shipped instruction.
    """
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load((Path(M.__file__).resolve().parents[1] / "config.yaml")
                         .read_text())
    block = (cfg.get("workers") or {}).get("maintenance", {}).get("mitigation_drill")
    assert isinstance(block, dict), (
        f"`workers.maintenance.mitigation_drill` parsed to {block!r}: the seat reads "
        f"it with `.get(...) or {{}}`, so a null there is indistinguishable from an "
        f"operator who turned the drill off — and a stray `enabled` at the sweep's "
        f"level turns the drill on under a reader that never looks at it"
    )
    assert set(block) >= {"enabled", "interval_seconds",
                          "wait_free_window_seconds"}, sorted(block)
    assert isinstance(block["enabled"], bool)
    for key in ("interval_seconds", "wait_free_window_seconds"):
        assert isinstance(block[key], int) and block[key] > 0, f"{key}={block[key]!r}"

    argv = M.mitigation_drill_argv({**M.MITIGATION_DRILL_DEFAULTS, **block})
    assert argv[1:3] == ["-m", "scripts.mitigation_drill"], argv
    assert argv[3:5] == ["--wait-free-window", str(block["wait_free_window_seconds"])], argv


def test_the_rate_decides_what_median_seconds_means_now():
    """The seat's interval is the second half of a number the drill's reader sees.

    `mitigation_state.HISTORY_CAP` readings per surface is the whole history, so the
    window `median_seconds` is the median *of* is the cap times this seat's interval:
    20 × 3600 s = 20 hours. Under #703's daily trigger that same 20 was about three
    weeks. The constant stayed, the window did not, and `app/mitigation_state.py`'s
    comment now says "about 20 hours" — a node, because that arithmetic has two
    owners in two files and either one moving alone makes the other one's prose a
    lie. It is also why the cap is not raised to buy the three weeks back: 20 is the
    number the mitigation-drill skill quotes to its runner, and the cap, the fixture
    and that prose move together or not at all (#1217).
    """
    from app import mitigation_state

    hours = (M.MITIGATION_DRILL_DEFAULTS["interval_seconds"]
             * mitigation_state.HISTORY_CAP) / 3600
    assert hours == 20, f"the median is taken over {hours}h of firings"
    assert hours < 24, (
        "a window over a day would make the comment in `app/mitigation_state.py` "
        "wrong rather than merely narrow, and nothing but this node reads both files"
    )


# ── The seat itself ────────────────────────────────────────────────────────


def test_the_pool_seat_calls_the_drill_tick_once_per_pass(q, monkeypatch):
    """The pool side of the wiring: the scheduler seat calls the tick, hands it this
    pool's own queue and the `workers.maintenance` block of the config, and calls it
    once — the rate limit lives inside the tick, so a seat that called it twice per
    pass would spend two passes' queue lock on the same hour."""
    seen = {}

    def spy(queue, cfg, **kw):
        seen["queue"] = queue
        seen["cfg"] = cfg
        return {"skipped": "disabled"}

    monkeypatch.setattr("app.config.CONFIG", {"workers": {"maintenance": CFG_ON}})
    monkeypatch.setattr(M, "maybe_run_mitigation_drill", spy)

    _run(WorkerPool(q, slots=2)._maybe_mitigation_drill())

    assert seen["queue"] is q
    assert seen["cfg"] == CFG_ON


def test_the_seat_never_raises_out_of_the_drill_tick(q, monkeypatch, caplog):
    """The rule the sweep beside it also keeps: this runs on the scheduler loop, and an
    exception here takes every source's enqueue pass down with it. A drill that cannot
    be spawned is a warning in the log, not a dead pool."""
    def boom(queue, cfg, **kw):
        raise RuntimeError("the seat blew up")

    monkeypatch.setattr("app.config.CONFIG", {"workers": {"maintenance": CFG_ON}})
    monkeypatch.setattr(M, "maybe_run_mitigation_drill", boom)

    with caplog.at_level(logging.WARNING, logger="lloyd-workers.pool"):
        _run(WorkerPool(q, slots=2)._maybe_mitigation_drill())

    assert any("Mitigation drill tick failed" in r.getMessage() for r in caplog.records), (
        "the seat swallowed the failure without a word. A drill series going quiet with "
        "no line beside it is the shape of this whole item"
    )


# ── #2431 clause 4: every firing that records nothing leaves a line ─────────
#
# `server.err` for 2026-10-08 holds a spawn at 18:05:17Z and no exit line beside
# it, no `refused:` text, no reading in the state file, and no live child when it
# was checked at 18:36Z: an hour of the drill series went missing with nothing
# written about it anywhere. Two endings were silent, and both are a line now —
# the reap of a child this process owned, and the orphan report for a spawn whose
# handle died with a previous process.


@pytest.fixture
def no_verdict_state(monkeypatch):
    """#2431 added two more pieces of process-local seat state beside the handle:
    the stamp THIS process wrote, and the stamp it has already reported an outcome
    for. Node-local rather than bolted onto the autouse fixture above, which every
    existing node in this file depends on and none of them asked to change.

    Both start empty for the same reason the handle does: inheriting either is
    inheriting a verdict, and the orphan report is precisely a claim about which
    process wrote a given stamp."""
    monkeypatch.setattr(M, "_drill_stamp_here", None)
    monkeypatch.setattr(M, "_drill_outcome_reported", None)


def _drill_state_at(state, at):
    """One reading in the state file, stamped `at`, written by the real writer."""
    from app import mitigation_state
    mitigation_state.record(
        [{"surface": "session_cancel", "classification": "in-flight",
          "seconds": 0.2, "ok": True}], path=state, at=at)


def test_a_drill_child_that_recorded_nothing_says_so_beside_its_exit_code(
        q, popen, monkeypatch, caplog, tmp_path, no_verdict_state):
    """#2431 clause 4, the branch that runs when the handle was NOT lost.

    The 2026-10-08 case re-made: a child this seat spawned, exiting 0, with the
    state file's newest reading predating the spawn — a drill that measured
    nothing and looked, from every log line in the file, like it had not been
    tried at all. The line now names the rc AND what the witness file says, so
    the two facts are on one line rather than in two files a human has to diff.

    The control is the second half of the same node, same rc: a child that DID
    record leaves the exit code with the reading's timestamp beside it, and must
    not be accused of recording nothing. And neither half emits the orphan report,
    because this process wrote the stamp the watermark holds — which is the only
    thing that distinguishes "my child, reaped" from "someone else's spawn, lost".
    """
    import app.paths as paths
    from app import mitigation_state

    state = tmp_path / "mitigation_drill.json"
    monkeypatch.setattr(paths, "MITIGATION_DRILL_STATE", state)
    q.wm_set(M.SOURCE, "last_drill_at", T0.isoformat())
    monkeypatch.setattr(M, "_drill_stamp_here", T0.isoformat())
    monkeypatch.setattr(M, "_drill_proc", _Proc(pid=4242, rc=0))
    _drill_state_at(state, (T0 - timedelta(hours=1)).isoformat())

    with caplog.at_level(logging.INFO, logger="lloyd-workers.maintenance"):
        out = _tick(q, CFG_ON, when=T0 + timedelta(minutes=30), popen=popen)
        silent = [r.getMessage() for r in caplog.records]
        caplog.clear()
        monkeypatch.setattr(M, "_drill_proc", _Proc(pid=4243, rc=0))
        _drill_state_at(state, (T0 + timedelta(minutes=10)).isoformat())
        _tick(q, CFG_ON, when=T0 + timedelta(minutes=30), popen=popen)
        recorded = [r.getMessage() for r in caplog.records]

    assert out["skipped"] == "not-due", (
        "the reap is a side effect of the tick, and the interval gate still "
        f"holds it off spawning another: {out}"
    )
    assert any("Mitigation drill exited with rc=0" in m and "recorded nothing" in m
               for m in silent), (
        f"a drill that exited 0 having written no reading left {silent}: this is "
        "the exact shape of the 18:05:17Z firing that vanished without a trace"
    )
    assert not any("outcome unknown" in m for m in silent + recorded), (
        "this process wrote the stamp and reaped its own child, so the orphan "
        "report would be a lie: the two branches are told apart by "
        "`_drill_stamp_here` and nothing else"
    )
    assert any("Mitigation drill exited with rc=0" in m and "recorded nothing" not in m
               and (T0 + timedelta(minutes=10)).isoformat() in m
               for m in recorded), (
        f"a drill that DID record left {recorded}: the line must cite the newest "
        "reading, or it is an accusation rather than a verdict"
    )
    assert mitigation_state.read(path=state)["session_cancel"]["n"] == 2


def test_a_drill_orphaned_by_a_restart_reports_its_outcome_as_unknown_once(
        q, popen, monkeypatch, caplog, tmp_path, no_verdict_state):
    """#2431 clause 4, the branch that runs when the handle WAS lost.

    `_drill_proc` is process-local, so a backend restart between the spawn and the
    next tick leaves the new process holding a `last_drill_at` stamp it did not
    write and no child to poll. Before #2431 the gate then answered `not-due` for
    the rest of the hour and the firing's fate was never recorded anywhere; now the
    first tick past such a stamp says the outcome is unknown, names the stamp, and
    says what the state file holds.

    The second tick is the control that makes it a report rather than an alarm: the
    seat ticks every 60 s, and one line per orphaned stamp is the whole claim —
    twelve lines an hour about a fact that cannot change is the noise that gets a
    log line ignored, which is how the first one stayed unread.
    """
    import app.paths as paths

    state = tmp_path / "mitigation_drill.json"
    monkeypatch.setattr(paths, "MITIGATION_DRILL_STATE", state)
    orphaned = (T0 - timedelta(minutes=10)).isoformat()
    q.wm_set(M.SOURCE, "last_drill_at", orphaned)

    with caplog.at_level(logging.INFO, logger="lloyd-workers.maintenance"):
        first = _tick(q, CFG_ON, popen=popen)
        lines = [r for r in caplog.records]
        caplog.clear()
        second = _tick(q, CFG_ON, popen=popen)
        again = [r for r in caplog.records]

    assert first["skipped"] == second["skipped"] == "not-due"
    assert not popen.calls, "an orphan report is a verdict, not a licence to re-spawn"
    unknown = [r.getMessage() for r in lines if "outcome unknown" in r.getMessage()]
    assert len(unknown) == 1, f"first tick left {len(unknown)} orphan lines: {unknown}"
    assert orphaned in unknown[0], (
        f"the line must name the stamp it is judging: {unknown[0]}"
    )
    assert any(r.levelno >= logging.WARNING for r in lines), (
        "a lost firing is worth a WARNING: the default threshold on some readers "
        "is WARNING, and an hour silently missing from the series is the failure "
        "this item exists to catch"
    )
    assert "no reading" in unknown[0], (
        f"the state file was absent, and the line says so: {unknown[0]}"
    )
    assert not [r for r in again if "outcome unknown" in r.getMessage()], (
        "the same stamp reported twice; the seat ticks every 60 s and the dedupe "
        "is what keeps the line readable"
    )
