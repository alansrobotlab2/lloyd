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
    # `down()` feeds the refused/http-error streak, which keeps the short budget.
    is_down, reason = down(info(start_ago=600.0), streak=3)
    assert is_down and "refused" in reason


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


# ---------------------------------------------------------------------------
# Settle — the only path that advances last-known-good
#
# The promoter deliberately never writes LKG. If this path is wrong, "last
# known good" silently stops meaning "observed healthy in production" and a
# later rollback aims at a commit that was never watched.
# ---------------------------------------------------------------------------

def _settle_guardian(tmp_path, monkeypatch, current, *, degraded=None):
    import types

    import guardian as G
    import probes as P

    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0,
    )
    g = G.Guardian(args)
    monkeypatch.setattr(P, "probe", lambda url, t: {
        "ok": True, "status": 200,
        "body": {"status": "ok", "degraded_modules": degraded or []},
        "error": None, "latency_ms": 1.0})
    written: dict = {}
    monkeypatch.setattr(g.state, "set_lkg",
                        lambda commit, **kw: written.update({"commit": commit, **kw}))
    cleared: list = []
    monkeypatch.setattr(g.state, "clear_current", lambda: cleared.append(True))
    monkeypatch.setattr(G.gstate, "append_event", lambda *a, **k: None)
    g.maybe_settle(current)
    return written, cleared


def test_settle_advances_last_known_good_once_the_window_closes(tmp_path, monkeypatch):
    written, cleared = _settle_guardian(
        tmp_path, monkeypatch,
        {"commit": "b" * 40, "errors_until_ts": NOW - 1_000_000})
    assert written["commit"] == "b" * 40
    assert cleared, "the observation record must be cleared once settled"


def test_settle_does_not_fire_before_the_window_closes(tmp_path, monkeypatch):
    import time
    written, cleared = _settle_guardian(
        tmp_path, monkeypatch,
        {"commit": "b" * 40, "errors_until_ts": time.time() + 600})
    assert written == {} and not cleared


def test_settle_does_not_fire_on_a_record_with_no_window(tmp_path, monkeypatch):
    """A `landing` record has null timestamps — nothing has been deployed."""
    written, _ = _settle_guardian(
        tmp_path, monkeypatch,
        {"commit": "b" * 40, "state": "landing", "errors_until_ts": None})
    assert written == {}


def test_settle_snapshots_the_mcp_degradation_baseline(tmp_path, monkeypatch):
    """So a module already degraded at settle cannot trigger a later rollback."""
    written, _ = _settle_guardian(
        tmp_path, monkeypatch,
        {"commit": "b" * 40, "errors_until_ts": NOW - 1_000_000},
        degraded=["thunderbird"])
    assert written["health"]["mcp_degraded_modules"] == ["thunderbird"]


def test_settle_ignores_a_record_with_no_commit(tmp_path, monkeypatch):
    written, _ = _settle_guardian(
        tmp_path, monkeypatch, {"errors_until_ts": NOW - 1_000_000})
    assert written == {}


# ---------------------------------------------------------------------------
# Why a probe failed matters more than that it failed
#
# Regression tests for a real false-positive rollback on 2026-09-06. An
# autoresearch round started 77 bench trials at 11:29:18; /health shares its
# event loop with that work, missed three consecutive 2s probes, and at
# 11:30:39 the guardian reverted a perfectly good promotion. A watchdog that
# reverts good code whenever the machine gets busy is worse than none.
# ---------------------------------------------------------------------------

def down2(proc, *, refused=0, timed_out=0, grace=45.0):
    return detect.process_down(
        proc, now=NOW, grace=grace,
        probe_fail_streak=refused, probe_threshold=3,
        probe_timeout_streak=timed_out, probe_timeout_threshold=24,
        start_history=[NOW - 3600],
        crash_loop_starts=3, crash_loop_window=180.0,
    )


def test_a_busy_backend_timing_out_is_not_dead():
    """The exact shape of the 11:30 false positive: three missed probes."""
    is_down, reason = down2(info(start_ago=600.0), timed_out=3)
    assert not is_down, reason


def test_timeouts_still_fire_eventually():
    """Unresponsive for two minutes is death, not busyness."""
    is_down, reason = down2(info(start_ago=600.0), timed_out=24)
    assert is_down and "unresponsive" in reason


def test_a_refused_connection_is_death_on_the_short_budget():
    """Nothing listening means the process is gone — no patience required."""
    is_down, reason = down2(info(start_ago=600.0), refused=3)
    assert is_down and "refused" in reason


def test_refused_and_timeout_budgets_are_independent():
    assert not down2(info(start_ago=600.0), refused=2, timed_out=20)[0]
    assert down2(info(start_ago=600.0), refused=3, timed_out=0)[0]


def test_the_probe_classifies_a_refusal_separately_from_a_timeout():
    import probes

    # Nothing listening on this port.
    result = probes.probe("http://127.0.0.1:9/health", 0.5)
    assert not result["ok"]
    assert result["kind"] in ("refused", "timeout"), result


def test_the_timeout_budget_covers_a_realistic_busy_window():
    """77 bench trials took ~90s of loop pressure; the budget must exceed it."""
    import policy

    assert policy.PROBE_TIMEOUT_STREAK * policy.TICK_SECONDS >= 100
    assert policy.PROBE_TIMEOUT_SECONDS >= 10, (
        "a 2s timeout is shorter than a loaded event loop's scheduling delay")


def test_the_supervisor_rpc_timeout_exceeds_stopwaitsecs():
    """A blocking stopProcess(wait=True) legitimately takes stopwaitsecs.

    At the old 5s client timeout the guardian logged "stop: error: timed out"
    for a stop that was working, and proceeded without knowing whether the
    writers were down — which is the one thing the stop-before-reset ordering
    exists to guarantee.
    """
    import policy

    assert policy.SUPERVISOR_RPC_TIMEOUT > 15.0


# ---------------------------------------------------------------------------
# A rehearsal must not look like a production incident
#
# The drill runs a REAL guardian against a throwaway repo. Without scoping, its
# test rollbacks land in the live vault daily note and file real backlog tasks.
# That happened on 2026-09-06: two drill rollbacks (26574f87, ddd6d1d0) were
# written to the daily note naming commits that exist only in a deleted scratch
# clone, and reading that note later suggested the audit trail had lost events.
# ---------------------------------------------------------------------------

def test_external_channels_are_suppressible(tmp_path):
    import notify

    vault = tmp_path / "obsidian" / "memory"
    vault.mkdir(parents=True)
    n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                        vault_root=str(tmp_path / "obsidian"), external=False)
    res = n.alert("critical", "drill rollback", "should stay local",
                  trigger="crash", commit="a" * 40)

    assert set(res) == {"ledger", "alert_file"}, res
    assert list(vault.glob("*.md")) == [], "a drill wrote into the vault"
    assert "backlog" not in res and "desktop" not in res


def test_external_channels_fire_by_default(tmp_path):
    import notify

    (tmp_path / "obsidian" / "memory").mkdir(parents=True)
    n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                        vault_root=str(tmp_path / "obsidian"),
                        backend_url="http://127.0.0.1:1")
    res = n.alert("warn", "real rollback", "body")
    assert "vault" in res and res["vault"] is True


def test_the_drill_passes_no_external_alerts(tmp_path):
    """The flag is only useful if rehearse.py actually sends it."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent /
           "scripts" / "selfmod" / "rehearse.py").read_text()
    assert "--no-external-alerts" in src


def test_repeated_identical_alerts_are_suppressed(tmp_path, monkeypatch):
    """A persistent condition ticks every 5s; five identical notices in 30
    seconds bury the one that matters."""
    import types

    import guardian as G
    import policy

    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "s"),
        guardian_state=str(tmp_path / "g"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0, no_external_alerts=True,
    )
    g = G.Guardian(args)
    sent: list = []
    monkeypatch.setattr(g.notifier, "alert",
                        lambda *a, **k: sent.append(a[1]) or {})

    for _ in range(5):
        g.alert("error", "Service down, but no promotion to revert", "body")
    assert len(sent) == 1, f"expected 1 fan-out, got {len(sent)}"

    # A different condition still gets through immediately.
    g.alert("error", "Something else entirely", "body")
    assert len(sent) == 2

    # And the same one fires again once the window has passed.
    g._alert_seen["Service down, but no promotion to revert"] -= policy.ALERT_REPEAT_SECONDS + 1
    g.alert("error", "Service down, but no promotion to revert", "body")
    assert len(sent) == 3
