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
