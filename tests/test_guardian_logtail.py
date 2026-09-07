"""Incremental log reading across rotation and truncation.

`logs/server.err` rotates at 10 MB by rename, so inodes change (verified on
the live box: server.err, .err.1 and .err.2 all differ). A cursor keyed on
path alone silently swallows whatever accumulated since the last tick, which
would make the error-rate detector blind exactly when a crash loop is filling
the log fastest.

The fixture in tests/fixtures/guardian/ carries real production line shapes,
including the two chronic errors that must never be able to trigger a
rollback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

GUARDIAN_DIR = Path(__file__).resolve().parent.parent / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import detect   # noqa: E402
import logtail  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "guardian" / "server.err.sample"

NEW_ERROR = ("2026-09-06 11:00:00,000 [ERROR] lloyd-harness: "
             "unexpected explosion in dispatch\n")


@pytest.fixture()
def log_setup(tmp_path):
    log = tmp_path / "server.err"
    log.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    cursor = logtail.LogCursor(tmp_path / "logcursors.json")
    return log, cursor


def test_first_sight_starts_at_the_end_and_does_not_replay_history(log_setup):
    log, cursor = log_setup
    text, over = cursor.read_new(str(log), 1 << 20)
    assert text == "" and not over


def test_appended_lines_are_returned_once(log_setup):
    log, cursor = log_setup
    cursor.read_new(str(log), 1 << 20)
    with open(log, "a", encoding="utf-8") as f:
        f.write(NEW_ERROR)
    text, _ = cursor.read_new(str(log), 1 << 20)
    assert "unexpected explosion" in text
    again, _ = cursor.read_new(str(log), 1 << 20)
    assert again == ""


def test_rotation_by_rename_loses_nothing_and_duplicates_nothing(log_setup):
    """The inode-change branch: drain the predecessor's tail, then the new file."""
    log, cursor = log_setup
    cursor.read_new(str(log), 1 << 20)

    # Written after our last read, then rotated away before the next tick.
    with open(log, "a", encoding="utf-8") as f:
        f.write("2026-09-06 11:00:00,000 [ERROR] lloyd-a: pre-rotation error\n")
    log.rename(log.with_suffix(".err.1"))
    log.write_text("2026-09-06 11:05:00,000 [ERROR] lloyd-b: post-rotation error\n",
                   encoding="utf-8")

    text, _ = cursor.read_new(str(log), 1 << 20)
    assert "pre-rotation error" in text, "tail of the rotated file was swallowed"
    assert "post-rotation error" in text
    assert text.count("pre-rotation error") == 1


def test_rotation_takes_the_inode_branch_not_the_size_branch(log_setup):
    log, cursor = log_setup
    cursor.read_new(str(log), 1 << 20)
    before = cursor.position(str(log))
    log.rename(log.with_suffix(".err.1"))
    log.write_text("x" * 10, encoding="utf-8")
    cursor.read_new(str(log), 1 << 20)
    after = cursor.position(str(log))
    assert before["inode"] != after["inode"]


def test_truncation_in_place_resets_the_cursor(log_setup):
    log, cursor = log_setup
    cursor.read_new(str(log), 1 << 20)
    log.write_text(NEW_ERROR, encoding="utf-8")  # `> server.err`
    text, _ = cursor.read_new(str(log), 1 << 20)
    assert "unexpected explosion" in text


def test_a_huge_burst_sets_overflow_and_is_capped(log_setup):
    log, cursor = log_setup
    cursor.read_new(str(log), 1 << 20)
    with open(log, "a", encoding="utf-8") as f:
        f.write("x" * 5000)
    text, over = cursor.read_new(str(log), 1024)
    assert over is True
    assert len(text) <= 1024


def test_a_missing_file_is_not_an_error(tmp_path):
    cursor = logtail.LogCursor(tmp_path / "c.json")
    assert cursor.read_new(str(tmp_path / "nope.err"), 1 << 20) == ("", False)


def test_cursor_survives_a_save_load_round_trip(log_setup, tmp_path):
    log, cursor = log_setup
    cursor.read_new(str(log), 1 << 20)
    cursor.save()
    with open(log, "a", encoding="utf-8") as f:
        f.write(NEW_ERROR)
    reloaded = logtail.LogCursor(tmp_path / "logcursors.json")
    text, _ = reloaded.read_new(str(log), 1 << 20)
    assert "unexpected explosion" in text


# ---------------------------------------------------------------------------
# Chronic bootstrap — the production errors that must never fire
# ---------------------------------------------------------------------------

def test_the_chronic_scheduler_error_is_learned_and_cannot_fire(log_setup):
    """The regression test for the failure mode this detector had to survive.

    Production has emitted `autonomy scheduler may be stalled` hourly for
    days. A detector that counted it would roll back on its very first tick,
    every time, blaming whatever happened to have been promoted.
    """
    log, _ = log_setup
    chronic = logtail.bootstrap_chronic(
        [str(log)], max_bytes=1 << 20, min_distinct_hours=3)
    assert chronic, "nothing was learned as chronic"

    events = detect.extract_events(log.read_text(encoding="utf-8"))
    assert events, "fixture produced no error events"
    fired, why = detect.error_spike(
        events, chronic=chronic, changed_paths=[],
        novel_threshold=1, fatal_distinct_threshold=1, changed_path_threshold=1)
    assert not fired, why


def test_the_warning_echo_is_never_counted_at_all(log_setup):
    log, _ = log_setup
    events = detect.extract_events(log.read_text(encoding="utf-8"))
    assert all("discord_alert" not in e["message"] for e in events)


def test_an_error_seen_in_only_one_hour_is_not_chronic(tmp_path):
    log = tmp_path / "server.err"
    log.write_text("\n".join(
        f"2026-09-06 08:0{i}:00,000 [ERROR] lloyd-x: a one-off blowup" for i in range(5)
    ), encoding="utf-8")
    chronic = logtail.bootstrap_chronic(
        [str(log)], max_bytes=1 << 20, min_distinct_hours=3)
    assert chronic == set()


def test_a_novel_error_still_fires_against_a_populated_chronic_set(log_setup):
    log, _ = log_setup
    chronic = logtail.bootstrap_chronic(
        [str(log)], max_bytes=1 << 20, min_distinct_hours=3)
    events = detect.extract_events(NEW_ERROR * 5)
    fired, _ = detect.error_spike(
        events, chronic=chronic, changed_paths=[],
        novel_threshold=5, fatal_distinct_threshold=3, changed_path_threshold=2)
    assert fired


# ---------------------------------------------------------------------------
# The cursor must advance on every tick, not only while observing
#
# Reading used to live inside `evaluate_errors`, which runs only while a
# promotion is under observation. Between rounds the cursor stood still, so
# the first tick of a new window read everything since the last one. On
# 2026-09-06 that reverted a healthy promotion four seconds after it landed,
# on nine ConnectError lines from 11:47-11:56 that morning — eight hours
# stale, from an unrelated incident, blamed on a four-second-old commit.
# ---------------------------------------------------------------------------

import types  # noqa: E402

import guardian as G  # noqa: E402
import policy  # noqa: E402


@pytest.fixture
def ticking_guardian(tmp_path, monkeypatch):
    """A Guardian whose only I/O is a temp log file it tails."""
    errlog = tmp_path / "server.err"
    errlog.write_text("", encoding="utf-8")
    monkeypatch.setattr(policy, "LOG_FILES", (str(errlog),))
    monkeypatch.setattr(G.policy, "LOG_FILES", (str(errlog),))

    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0,
    )
    g = G.Guardian(args)
    monkeypatch.setattr(g, "ensure_chronic", lambda: None)
    g.chronic = set()
    g.drain_logs()          # first sight: start at EOF, never replay history
    return g, errlog


def _boom(n, when="2026-09-06 11:47:24,044"):
    line = (f"{when} [ERROR] lloyd-workers.pool: [worker-0] failed "
            "domain-research/research: ConnectError: All connection attempts failed\n")
    return line * n


def test_errors_from_before_the_window_cannot_condemn_a_promotion(
        ticking_guardian, monkeypatch):
    """The 2026-09-06 rollback, reproduced and then prevented.

    Driven through `tick()` on purpose. The bug was not in the reading, it
    was in WHERE the reading was called from — so a test that calls
    `drain_logs` directly passes just as well with the call removed from the
    tick loop, which is the one thing that must not regress.
    """
    g, errlog = ticking_guardian
    monkeypatch.setattr(g, "collect", lambda: {"now": 0.0, "supervisord": "ok",
                                               "procs": {}, "probes": {}})
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 0.0)
    monkeypatch.setattr(g, "evaluate_liveness", lambda snap: (False, "ok"))
    monkeypatch.setattr(g.state, "current", lambda: None)   # nothing observed

    # Hours pass with no promotion under observation. The guardian keeps
    # ticking; nothing is being judged, but the tape must keep moving.
    with errlog.open("a") as f:
        f.write(_boom(9))
    for _ in range(3):
        assert g.tick() == "armed"

    # A promotion now lands and opens its window. Its very first evaluation
    # must see a quiet log, because none of that happened on its watch.
    spiked, why = g.evaluate_errors({"changed_paths": ["agent_mcp/retrieval.py"]})
    assert not spiked, f"stale errors were attributed to a new commit: {why}"


def test_errors_during_the_window_still_trigger(ticking_guardian):
    """The fix must not blind the detector to what it exists to catch."""
    g, errlog = ticking_guardian
    g.drain_logs()

    with errlog.open("a") as f:
        f.write(_boom(9))
    g.drain_logs()

    spiked, why = g.evaluate_errors({"changed_paths": []})
    assert spiked and "ConnectError" in why


def test_draining_is_idempotent_within_a_tick(ticking_guardian):
    """evaluate_errors reads the buffer; it must not re-read the file and
    consume a second tick's worth."""
    g, errlog = ticking_guardian
    with errlog.open("a") as f:
        f.write(_boom(9))
    g.drain_logs()

    first = g.evaluate_errors({"changed_paths": []})
    second = g.evaluate_errors({"changed_paths": []})
    assert first == second


def test_tick_drains_before_any_early_return(tmp_path, monkeypatch, ticking_guardian):
    """Every early return in tick() must sit BELOW the drain. A paused
    guardian that stops reading is the original bug wearing a different hat:
    the promoter pauses it around its own restart, and those errors would
    then land in the observation window that opens moments later."""
    g, errlog = ticking_guardian
    monkeypatch.setattr(g, "collect", lambda: {"now": 0.0, "supervisord": "ok",
                                               "procs": {}, "probes": {}})
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 900.0)
    monkeypatch.setattr(g, "evaluate_liveness", lambda snap: (False, "ok"))

    with errlog.open("a") as f:
        f.write(_boom(9))
    assert g.tick() == "paused"

    # The pause consumed them; the window that opens next sees nothing.
    spiked, why = g.evaluate_errors({"changed_paths": []})
    assert not spiked, f"errors survived the pause and leaked into the window: {why}"


def test_tick_advances_the_cursor_even_with_nothing_under_observation(
        ticking_guardian, monkeypatch):
    """Pins the *call site*, not the reading.

    Asserting only "no spike" is satisfied just as well by never reading at
    all, which is exactly the bug. This checks the tape physically moved.
    """
    g, errlog = ticking_guardian
    monkeypatch.setattr(g, "collect", lambda: {"now": 0.0, "supervisord": "ok",
                                               "procs": {}, "probes": {}})
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 0.0)
    monkeypatch.setattr(g, "evaluate_liveness", lambda snap: (False, "ok"))
    monkeypatch.setattr(g.state, "current", lambda: None)

    before = g.cursor._state[str(errlog)]["offset"]
    with errlog.open("a") as f:
        f.write(_boom(9))
    assert g.tick() == "armed"
    after = g.cursor._state[str(errlog)]["offset"]
    assert after > before, "tick() did not advance the log cursor"
    assert after == errlog.stat().st_size

    # And the events it consumed are seen ONCE, on the tick that read them.
    spiked, _ = g.evaluate_errors({"changed_paths": []})
    assert spiked, "the tick that read them should have judged them"
    g.tick()
    spiked, _ = g.evaluate_errors({"changed_paths": []})
    assert not spiked, "a later tick must not re-serve the same errors"


def test_do_rollback_asks_with_the_promotion_in_hand(ticking_guardian, monkeypatch):
    """`current.json` records the right target and nothing read it — that is
    what cost 26 commits. Pin the wiring, not just the resolver."""
    g, _ = ticking_guardian
    record = {"commit": "c" * 40, "parent": "d" * 40, "rollback_target": "b" * 40}
    monkeypatch.setattr(g.state, "current", lambda: record)

    seen = []
    def _target(current=None):
        seen.append(current)
        return "b" * 40, "test"
    monkeypatch.setattr(g.state, "rollback_target", _target)
    monkeypatch.setattr(G.rb, "head_commit", lambda repo: "b" * 40)  # -> early return
    monkeypatch.setattr(g, "alert", lambda *a, **k: None)

    g.do_rollback("error_rate", "whatever")
    assert seen == [record], "do_rollback must pass the observed promotion"
