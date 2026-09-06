"""Guardian failure predicates — the arithmetic of when to roll back.

These are pure functions of a snapshot dict precisely so every branch is
table-testable without a running system. The drill
(`scripts/guardian_drill.py`) proves the guardian *acts*; this file proves it
*decides* correctly.

Process-info dicts here are shaped like real `getAllProcessInfo` output,
including the `start`/`now`/`spawnerr`/`group` fields the predicate reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

GUARDIAN_DIR = Path(__file__).resolve().parent.parent / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import detect  # noqa: E402


NOW = 1_788_710_000.0


def info(state="RUNNING", *, start_ago=3600.0, spawnerr="", pid=1234):
    return {
        "name": "lloyd-backend", "group": "lloyd-mc", "statename": state,
        "start": NOW - start_ago, "now": NOW, "pid": pid, "spawnerr": spawnerr,
    }


def down(proc, *, streak=0, grace=45.0, history=None, intentional=False):
    return detect.process_down(
        proc, now=NOW, grace=grace,
        probe_fail_streak=streak, probe_threshold=3,
        start_history=history if history is not None else [NOW - 3600],
        crash_loop_starts=3, crash_loop_window=180.0,
        intentional_stop=intentional,
    )


# ---------------------------------------------------------------------------
# supervisord state is consulted first
# ---------------------------------------------------------------------------

def test_fatal_is_down_regardless_of_probes():
    is_down, reason = down(info("FATAL", spawnerr="exited too quickly"), streak=0)
    assert is_down
    assert "FATAL" in reason


def test_fatal_with_a_healthy_port_is_still_down_unlike_the_ui_helper():
    """The guardian must invert `app.supervisor_client._health`'s priority.

    That function's own comment says "Port being open is the strongest signal —
    trust it over supervisord state", which is right for the Services page and
    catastrophic for a watchdog: a FATAL backend whose :8080 is still held by a
    zombie worker would read "healthy" and the crash would never be noticed.
    Asserted side by side so the divergence stays deliberate.
    """
    from app.supervisor_client import _health

    assert _health("failed", True) == "healthy"          # the UI helper's answer
    is_down, _ = down(info("FATAL"), streak=0)           # the guardian's answer
    assert is_down


def test_stopped_without_an_intentional_stop_is_down():
    is_down, reason = down(info("STOPPED"))
    assert is_down and "STOPPED" in reason


def test_stopped_during_an_intentional_stop_is_not_down():
    is_down, _ = down(info("STOPPED"), intentional=True)
    assert not is_down


def test_unknown_to_supervisord_is_down():
    is_down, reason = down(None)
    assert is_down and "unknown" in reason


# ---------------------------------------------------------------------------
# Boot grace
# ---------------------------------------------------------------------------

def test_a_freshly_started_process_failing_probes_is_starting_not_down():
    is_down, reason = down(info(start_ago=3.0), streak=10)
    assert not is_down
    assert reason == "starting"


def test_after_the_grace_window_a_failing_probe_streak_is_down():
    is_down, reason = down(info(start_ago=600.0), streak=3)
    assert is_down and "probe failed" in reason


def test_two_failed_probes_are_not_enough():
    is_down, _ = down(info(start_ago=600.0), streak=2)
    assert not is_down


# ---------------------------------------------------------------------------
# Crash loop that never reaches FATAL
# ---------------------------------------------------------------------------

def test_crash_loop_fires_even_when_every_sample_says_running():
    """The `startsecs`-too-small pathology.

    supervisord marks the process RUNNING before it can serve, so a boot
    failure counts as an unexpected exit and autorestart retries forever
    without ever parking in FATAL. Sampling `statename` shows RUNNING most of
    the time; distinct spawn timestamps do not lie.
    """
    history = [NOW - 150, NOW - 100, NOW - 50]
    is_down, reason = down(info(start_ago=50.0), history=history)
    assert is_down and "crash loop" in reason


def test_three_starts_spread_beyond_the_window_is_not_a_crash_loop():
    history = [NOW - 5000, NOW - 3000, NOW - 50]
    is_down, _ = down(info(start_ago=50.0), history=history)
    assert not is_down


def test_repeated_identical_start_timestamps_are_one_spawn():
    history = [NOW - 50, NOW - 50, NOW - 50]
    is_down, _ = down(info(start_ago=50.0), history=history)
    assert not is_down


# ---------------------------------------------------------------------------
# MCP degradation is usually not fatal
# ---------------------------------------------------------------------------

def test_zero_tools_is_fatal():
    fatal, why = detect.mcp_degraded_is_fatal({"tools": 0, "degraded_modules": []}, [])
    assert fatal and "zero tools" in why


def test_a_module_degraded_since_last_known_good_is_fatal():
    fatal, why = detect.mcp_degraded_is_fatal(
        {"tools": 100, "degraded_modules": ["facts"]}, [])
    assert fatal and "facts" in why


def test_pre_existing_degradation_is_only_a_warning():
    """Thunderbird closed must not roll back Lloyd's code."""
    fatal, why = detect.mcp_degraded_is_fatal(
        {"tools": 84, "degraded_modules": ["thunderbird"]}, ["thunderbird"])
    assert not fatal and "pre-existing" in why


def test_no_degradation_is_ok():
    fatal, _ = detect.mcp_degraded_is_fatal({"tools": 124, "degraded_modules": []}, [])
    assert not fatal


# ---------------------------------------------------------------------------
# Log signatures
# ---------------------------------------------------------------------------

CHRONIC_LINE = ("2026-09-06 08:06:14,321 [ERROR] lloyd-workers.scheduled_task: "
                "autonomy scheduler may be stalled: oldest claimable queue item is 1266 min old")
CHRONIC_LINE_2 = ("2026-09-06 09:11:02,001 [ERROR] lloyd-workers.scheduled_task: "
                  "autonomy scheduler may be stalled: oldest claimable queue item is 1300 min old")


def test_varying_numbers_collapse_to_one_signature():
    a = detect.parse_log_line(CHRONIC_LINE)
    b = detect.parse_log_line(CHRONIC_LINE_2)
    assert a["signature"] == b["signature"]


def test_warnings_are_never_errors():
    line = ("2026-09-06 08:06:14,321 [WARNING] lloyd-server: discord_alert "
            "(no channel/token configured): autonomy scheduler may be stalled")
    assert detect.extract_events(line) == []


def test_a_chronic_signature_cannot_fire():
    """The exact production failure mode this detector had to survive."""
    events = detect.extract_events("\n".join([CHRONIC_LINE] * 20))
    chronic = {events[0]["signature"]}
    fired, why = detect.error_spike(
        events, chronic=chronic, changed_paths=[],
        novel_threshold=5, fatal_distinct_threshold=3, changed_path_threshold=2)
    assert not fired and "no novel" in why


def test_a_novel_signature_over_threshold_fires():
    line = "2026-09-06 09:00:00,000 [ERROR] lloyd-harness: brand new explosion"
    events = detect.extract_events("\n".join([line] * 5))
    fired, why = detect.error_spike(
        events, chronic=set(), changed_paths=[],
        novel_threshold=5, fatal_distinct_threshold=3, changed_path_threshold=2)
    assert fired and "x5" in why


def test_four_occurrences_below_threshold_do_not_fire():
    line = "2026-09-06 09:00:00,000 [ERROR] lloyd-harness: brand new explosion"
    events = detect.extract_events("\n".join([line] * 4))
    fired, _ = detect.error_spike(
        events, chronic=set(), changed_paths=[],
        novel_threshold=5, fatal_distinct_threshold=3, changed_path_threshold=2)
    assert not fired


def test_an_error_naming_a_changed_path_fires_at_a_lower_threshold():
    """Causal evidence, not correlation — and it costs nothing to check."""
    line = ('2026-09-06 09:00:00,000 [ERROR] lloyd-harness: failed in '
            'app/harness/loop.py during dispatch')
    events = detect.extract_events("\n".join([line] * 2))
    fired, why = detect.error_spike(
        events, chronic=set(), changed_paths=["app/harness/loop.py"],
        novel_threshold=5, fatal_distinct_threshold=3, changed_path_threshold=2)
    assert fired and "changed path" in why

    # Same two events, but the promotion touched something else.
    fired2, _ = detect.error_spike(
        events, chronic=set(), changed_paths=["workers/pool.py"],
        novel_threshold=5, fatal_distinct_threshold=3, changed_path_threshold=2)
    assert not fired2


def test_distinct_novel_tracebacks_fire():
    tb = ("Traceback (most recent call last):\n"
          '  File "{path}", line 1, in <module>\n'
          "{exc}: boom\n")
    text = "\n".join(
        tb.format(path=f"/x/{i}.py", exc=name)
        for i, name in enumerate(["ValueError", "KeyError", "TypeError"]))
    events = detect.extract_events(text)
    assert len(events) == 3
    fired, why = detect.error_spike(
        events, chronic=set(), changed_paths=[],
        novel_threshold=99, fatal_distinct_threshold=3, changed_path_threshold=99)
    assert fired and "distinct novel fatal" in why


def test_one_traceback_is_one_event_not_one_per_frame():
    text = ("Traceback (most recent call last):\n"
            '  File "/a.py", line 1, in f\n'
            '  File "/b.py", line 2, in g\n'
            '  File "/c.py", line 3, in h\n'
            "ValueError: boom\n")
    assert len(detect.extract_events(text)) == 1


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

def test_cusum_fires_on_two_failures_for_a_source_that_never_fails():
    """session-distill / autoresearch measured p0 ≈ 0.011 and 0.000.

    Two consecutive failures there is decisive: neither source has failed
    twice in a row across 322 recorded runs.
    """
    score = 0.0
    for _ in range(2):
        score = detect.cusum_update(score, True, p0=0.011, p1=0.30, floor=0.01)
    assert score > 4.6


def test_cusum_tolerates_the_measured_baseline_for_a_flaky_source():
    """scheduled-task measured p0 = 0.129 over 1700 runs — one failure is normal."""
    score = detect.cusum_update(0.0, True, p0=0.129, p1=0.30, floor=0.01)
    assert score < 4.6


def test_cusum_never_goes_negative():
    score = 0.0
    for _ in range(50):
        score = detect.cusum_update(score, False, p0=0.129, p1=0.30, floor=0.01)
    assert score == 0.0


def test_cusum_eventually_fires_on_a_sustained_failure_run():
    score = 0.0
    fired_at = None
    for i in range(1, 15):
        score = detect.cusum_update(score, True, p0=0.129, p1=0.30, floor=0.01)
        if score > 4.6 and fired_at is None:
            fired_at = i
    assert fired_at is not None and 5 <= fired_at <= 12


# ---------------------------------------------------------------------------
# Data damage — the class git reset cannot undo
# ---------------------------------------------------------------------------

def test_a_large_row_drop_is_damage():
    hit, why = detect.data_damage(12000, 9000, 0.05)
    assert hit and "dropped" in why


def test_a_small_delta_is_not_damage():
    hit, _ = detect.data_damage(12000, 11800, 0.05)
    assert not hit


def test_growth_is_not_damage():
    hit, _ = detect.data_damage(12000, 13000, 0.05)
    assert not hit


def test_missing_baseline_never_fires():
    assert detect.data_damage(None, 100, 0.05)[0] is False
    assert detect.data_damage(0, 0, 0.05)[0] is False


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("took 1.5s to finish", "took 92.1s to finish"),
    ("sha abc1234def5678", "sha 99ff00aa11bb22"),
    ("read /home/alan/lloyd/app/x.py", "read /home/alan/lloyd/app/y.py"),
])
def test_normalization_collapses_varying_parts(a, b):
    assert detect.normalize_message(a) == detect.normalize_message(b)


# ---------------------------------------------------------------------------
# When rollback is appropriate at all
#
# These guard the two cases where the correct action is to alert rather than to
# rewrite history. Both are about the same thing: a rollback is only ever the
# right answer for a commit the loop promoted and is still observing.
# ---------------------------------------------------------------------------

def _guardian(tmp_path, monkeypatch, *, current, head, lkg):
    """A Guardian with I/O stubbed, for exercising tick()'s decision path."""
    import types

    import guardian as G
    import rollback as RB

    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0,
    )
    g = G.Guardian(args)
    monkeypatch.setattr(g.state, "current", lambda: current)
    monkeypatch.setattr(g.state, "lkg", lambda: {"commit": lkg})
    monkeypatch.setattr(g.state, "rollback_target", lambda: (lkg, "test"))
    monkeypatch.setattr(g.state, "is_broken", lambda: False)
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 0.0)
    monkeypatch.setattr(RB, "head_commit", lambda repo: head)
    monkeypatch.setattr(g, "collect", lambda: {"now": NOW, "supervisord": "ok",
                                               "procs": {}, "probes": {}})
    monkeypatch.setattr(g, "evaluate_liveness", lambda snap: (True, "backend FATAL"))
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)

    rolled: list = []
    monkeypatch.setattr(g, "do_rollback", lambda *a: rolled.append(a) or True)
    alerts: list = []
    monkeypatch.setattr(g, "alert", lambda *a, **k: alerts.append(a))
    return g, rolled, alerts


def test_a_crash_with_nothing_under_observation_does_not_revert(tmp_path, monkeypatch):
    """The case that actually bites.

    HEAD legitimately differs from last-known-good most of the time — a human
    commit, a nightly job. Reverting on a crash then would destroy work no
    promotion ever asked the guardian to judge.
    """
    g, rolled, alerts = _guardian(tmp_path, monkeypatch,
                                  current=None, head="b" * 40, lkg="a" * 40)
    assert g.tick() == "down_unobserved"
    assert rolled == [], "reverted a commit that no promotion was observing"
    assert alerts and "no promotion to revert" in alerts[0][1]


def test_a_crash_while_observing_a_promotion_does_revert(tmp_path, monkeypatch):
    g, rolled, _ = _guardian(tmp_path, monkeypatch,
                             current={"commit": "b" * 40, "errors_until_ts": 0},
                             head="b" * 40, lkg="a" * 40)
    assert g.tick() == "rolling_back"
    assert rolled and rolled[0][0] == "crash"
