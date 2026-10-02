"""Guardian failure predicates — the arithmetic of when to roll back.

These are pure functions of a snapshot dict precisely so every branch is
table-testable without a running system. What used to be named here as the proof
that the guardian *acts* — a drill script under `scripts/` — has never existed in
git history, so this file was the only place that script was ever referenced. The
drill that does exist is the automod gate's `drill` rung,
`scripts/automod/rehearse.py`, and it deliberately does NOT file: it passes
`--no-external-alerts` (`agent-services/guardian/guardian.py:1390`) so a rehearsal
cannot put a phantom item on the board, which is what
`test_the_drill_passes_no_external_alerts` pins.

So the filing path is proven here, not by a drill:
`test_the_guardians_captured_body_files_itself_through_the_real_route` replays the
guardian's own POST body through the real
`app/routers/backlog.py::backlog_task_create` and asserts the task file the route
wrote. This file proves the guardian *decides* correctly.

Process-info dicts here are shaped like real `getAllProcessInfo` output,
including the `start`/`now`/`spawnerr`/`group` fields the predicate reads.
"""

from __future__ import annotations

import contextlib
import re
import types
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


# ── an unreadable knowledge graph is not an intact one (#1525) ──────────────
# `detect.data_damage` above is a pure predicate over two numbers, and those four
# nodes are all it ever had. The defect lived one level up: the predicate answers
# "no baseline" both for a missing baseline and for a count that could not be
# taken, `evaluate_data_damage` threw that reason away, and so a store that could
# not be opened at all — the shape a moved data root leaves behind — reached the
# journal as "data intact". These call the real method on a real Guardian with
# only the counters' inputs substituted, so what is exercised is the reporting,
# not a stub's return value.

def _kg_store(path, rows: int) -> str:
    """A knowledge-graph store with `rows` rows, at `path`."""
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("CREATE TABLE edges (id INTEGER PRIMARY KEY, src TEXT, dst TEXT)")
    con.executemany("INSERT INTO edges (src, dst) VALUES (?, ?)",
                    [(f"e{i}", f"t{i}") for i in range(rows)])
    con.commit()
    con.close()
    return str(path)


def _damage_guardian(tmp_path, monkeypatch, *, kg_db: str, current: dict):
    """A Guardian whose only real behaviour is `evaluate_data_damage`."""
    import types

    import guardian as G

    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0,
    )
    g = G.Guardian(args)
    monkeypatch.setattr(G.policy, "KG_DB", kg_db)
    # The vault half is fixed at "unchanged", so every assertion below is about
    # the graph read and cannot be satisfied by a vault finding.
    monkeypatch.setattr(G, "count_vault_files", lambda root: 400)
    return g, dict(current)


def test_a_graph_that_cannot_be_opened_is_reported_unreadable_not_intact(tmp_path,
                                                                          monkeypatch):
    """The store is absent, which is what a root that moved looks like to this
    watchdog, and it is the case #1525 names as the risk of the restated path."""
    missing = str(tmp_path / "moved-away" / "kg.sqlite")
    g, current = _damage_guardian(
        tmp_path, monkeypatch, kg_db=missing,
        current={"kg_rows": 1000, "vault_files": 400})

    hit, why = g.evaluate_data_damage(current)

    assert hit is False, "an unreadable store is not evidence of damage to roll back"
    assert "UNREADABLE" in why, f"reported {why!r} for a store it never opened"
    assert "data intact" not in why, why
    assert missing in why, f"the reason must name the store it could not read: {why!r}"


def test_a_store_whose_read_raises_is_reported_unreadable_too(tmp_path, monkeypatch):
    """The file is there and is not a database: `count_kg_rows` swallows that into
    the same None, so the reporting has to carry it — the second failure shape
    `except Exception: return None` hid."""
    junk = tmp_path / "kg.sqlite"
    junk.write_bytes(b"not a database, just bytes with a sqlite suffix")
    g, current = _damage_guardian(
        tmp_path, monkeypatch, kg_db=str(junk),
        current={"kg_rows": 1000, "vault_files": 400})

    hit, why = g.evaluate_data_damage(current)

    assert hit is False and "UNREADABLE" in why and "data intact" not in why, why


def test_the_healthy_read_still_reports_data_intact(tmp_path, monkeypatch):
    """The positive control, in both directions. `data intact` has to survive a
    store that really was counted, or the node above could be satisfied by a
    reason that always complains; and the same call has to still fire on a real
    drop, or `data intact` would be a constant and the tripwire a decoration."""
    store = _kg_store(tmp_path / "kg.sqlite", 950)
    g, current = _damage_guardian(
        tmp_path, monkeypatch, kg_db=store,
        current={"kg_rows": 1000, "vault_files": 400})

    assert g.evaluate_data_damage(current) == (False, "data intact")

    _kg_store(tmp_path / "dropped.sqlite", 900)
    g2, _ = _damage_guardian(
        tmp_path, monkeypatch, kg_db=str(tmp_path / "dropped.sqlite"),
        current={"kg_rows": 1000, "vault_files": 400})
    hit, why = g2.evaluate_data_damage({"kg_rows": 1000, "vault_files": 400})
    assert hit and "knowledge graph rows" in why and "dropped 10.0%" in why, why


def test_a_missing_path_is_reported_as_no_path(tmp_path, monkeypatch):
    """`policy.KG_DB` empty is the shape of a resolver that produced nothing at
    all; the reason has to say that rather than name a file."""
    g, current = _damage_guardian(
        tmp_path, monkeypatch, kg_db="",
        current={"kg_rows": 1000, "vault_files": 400})

    hit, why = g.evaluate_data_damage(current)

    assert hit is False and "UNREADABLE" in why, why
    assert "no path given" in why, why


def test_the_unreadable_reason_says_which_branch_named_the_path(tmp_path, monkeypatch):
    """The reason carries the branch that produced the path, not just the path.

    On the deployed box the resolver and the fallback literal name the SAME file —
    the fallback root and the production data root are both
    `/home/alansrobotlab/lloyd-data` — so a printed path alone can never tell "the
    data root moved" apart from "the resolver could not be loaded", which are two
    incidents with two different fixes. `policy.kg_db_path` returns the branch
    beside the path precisely so this line can say which one the watchdog was in.
    """
    import guardian as G

    missing = str(tmp_path / "moved-away" / "kg.sqlite")
    g, current = _damage_guardian(
        tmp_path, monkeypatch, kg_db=missing,
        current={"kg_rows": 1000, "vault_files": 400})

    monkeypatch.setattr(G.policy, "KG_DB_SOURCE", G.policy.KG_SOURCE_FALLBACK)
    hit, why = g.evaluate_data_damage(current)
    assert hit is False and "UNREADABLE" in why, why
    assert G.policy.KG_SOURCE_FALLBACK in why, (
        f"a degraded path was reported as if the resolver had answered: {why!r}")

    monkeypatch.setattr(G.policy, "KG_DB_SOURCE", G.policy.KG_SOURCE_RESOLVER)
    hit2, why2 = g.evaluate_data_damage(current)
    assert hit2 is False and G.policy.KG_SOURCE_RESOLVER in why2, why2
    assert G.policy.KG_SOURCE_FALLBACK not in why2, (
        f"the branch label is a constant, not an appendage: {why2!r}")


def _observing_guardian(tmp_path, monkeypatch, *, kg_db: str, store_rows: int | None):
    """A Guardian inside an observation window whose liveness is healthy.

    `restart: False` so the error leg is skipped and only the data leg can fire:
    the error log belongs to services a non-restarting landing never restarted,
    and this test is about the store read. `store_rows=None` means no file at
    that path at all — what a moved data root leaves behind.

    The four subsystem probes `tick()` runs before its decision
    (`drain_logs`, `check_vault`, `check_data`, `check_memory`) are stubbed, not
    left running: `check_data` acts on the machine's real data root, and on a box
    whose tripwire is armed it writes the halt marker, pauses workers and pages —
    none of which a store-read test is entitled to do. The one leg under test is
    the data-damage evaluation, which is called directly by `tick()` and stays
    real.
    """
    import time
    import types

    import guardian as G

    store = tmp_path / "kg.sqlite"
    if store_rows is not None:
        _kg_store(store, store_rows)
    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0,
    )
    g = G.Guardian(args)
    current = {"commit": "b" * 40, "errors_until_ts": time.time() + 600,
               "restart": False, "kg_rows": 1000, "vault_files": 400}
    monkeypatch.setattr(g.state, "current", lambda: current)
    monkeypatch.setattr(g.state, "is_broken", lambda: False)
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 0.0)
    monkeypatch.setattr(g, "collect", lambda: {"now": NOW, "supervisord": "ok",
                                              "procs": {}, "probes": {}})
    monkeypatch.setattr(g, "evaluate_liveness", lambda snap: (False, "healthy"))
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(g, "drain_logs", lambda: None)
    monkeypatch.setattr(g, "check_vault", lambda: None)
    monkeypatch.setattr(g, "check_data", lambda: None)
    monkeypatch.setattr(g, "check_memory", lambda: None)
    rolled: list = []

    def _record_rollback(*a, **k):
        # Records the call and returns the bool the real `do_rollback` returns. A
        # one-line lambda wrapping a mutating call would return that call's own
        # truthiness, which no reader can check and no assertion here reads: every
        # assertion below looks at `rolled`, never at this bool.
        rolled.append(a)
        return True

    monkeypatch.setattr(g, "do_rollback", _record_rollback)
    monkeypatch.setattr(G.policy, "KG_DB", kg_db)
    monkeypatch.setattr(G, "count_vault_files", lambda root: 400)

    lines: list = []
    monkeypatch.setattr(G, "log", lambda msg: lines.append(msg))
    return g, rolled, lines


def test_an_unreadable_graph_reaches_the_journal_and_rolls_back_nothing(tmp_path,
                                                                       monkeypatch):
    """Clause 3's second half, one level up: the tick call site.

    `evaluate_data_damage` returned the right reason and `tick()` read it only
    under `if damaged:`, so the unreadable verdict — the one thing that would
    have shown a moved root — died in a local, and the journal carried neither
    `data intact` nor a warning: silence, which is what made the original defect
    unseeable. `count_kg_rows`'s docstring promises the path reaches the log
    line; this node is what makes that promise checkable. No rollback either
    way: a store this process cannot open is not evidence that the promoted
    commit deleted rows.
    """
    import guardian as G

    missing = str(tmp_path / "moved-away" / "kg.sqlite")
    g, rolled, lines = _observing_guardian(tmp_path, monkeypatch,
                                           kg_db=missing, store_rows=None)

    assert g.tick() == "observing"
    assert rolled == [], "an unreadable store must not revert a promotion"
    unreadable = [ln for ln in lines if "inconclusive" in ln]
    assert unreadable, f"the store read was never logged; the tick said: {lines!r}"
    assert G.KG_UNREADABLE_MARK in unreadable[0], unreadable[0]
    assert missing in unreadable[0], (
        f"the log line must name the store it could not open: {unreadable[0]!r}")


def test_a_counted_graph_stays_silent_in_the_journal(tmp_path, monkeypatch):
    """The control: the log line has to be about the unreadable read, not about
    observing. 950 rows against a 1000-row baseline is inside the 5% floor, so
    the tick is healthy and must add no data line at all — otherwise the node
    above would pass on an unconditional log and `observing` would always look
    like a store that could not be read."""
    import guardian as G

    g, rolled, lines = _observing_guardian(tmp_path, monkeypatch,
                                           kg_db=str(tmp_path / "kg.sqlite"),
                                           store_rows=950)

    assert g.tick() == "observing"
    assert rolled == []
    assert not [ln for ln in lines if G.KG_UNREADABLE_MARK in ln], lines
    assert not [ln for ln in lines if "data" in ln], lines


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

    # Each stub records its own call and nothing else. Every assertion below
    # reads `rolled` / `alerts`; the bools are there only because the real methods
    # return bool, so they are written out as defs with an explicit return rather
    # than wrapped in a lambda whose value a reader would have to reason about.
    rolled: list = []
    alerts: list = []

    def _record_rollback(*a):
        rolled.append(a)
        return True

    monkeypatch.setattr(g, "do_rollback", _record_rollback)
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


def test_no_test_reaches_the_users_screen(tmp_path, monkeypatch):
    """`external=False` was never the whole mute.

    The toast and the journal line only need a session bus, so any in-process
    test that builds a default `Notifier` paints the user's screen — which is
    how `Lloyd guardian: real rollback / body` appeared on 2026-09-07, once
    per gate run, with the literal word "body" as the detail because
    "real rollback"/"body" are fixture strings from
    `test_guardian_predicates.py` and `test_guardian_speak.py`. Asserting on
    the *command* rather than the returned bool: `_run` reports True whenever
    it managed to spawn `notify-send`, so a bool is not evidence of either
    delivery or suppression.
    """
    import notify

    seen: list[list[str]] = []

    def _record_run(cmd, timeout=5.0):
        # Records the command it was handed. The bool mirrors what the real `_run`
        # returns once it has spawned something; nothing here reads it, because the
        # node below asserts on the command list, not the bool.
        seen.append(list(cmd))
        return True

    monkeypatch.setattr(notify, "_run", _record_run)
    (tmp_path / "obsidian" / "memory").mkdir(parents=True)
    n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                        vault_root=str(tmp_path / "obsidian"),
                        backend_url="http://127.0.0.1:1")

    res = n.alert("critical", "real rollback", "body")

    flat = " ".join(" ".join(c) for c in seen)
    assert "notify-send" not in flat, f"a test toasted the user: {flat}"
    assert "systemd-cat" not in flat, f"a test wrote to the live journal: {flat}"
    assert res["desktop"] is False and res["journal"] is False, res
    assert res["vault"] is True, "the in-suite mute must not reach the vault scope"


def test_the_room_channels_are_switches_not_dead_code(tmp_path, monkeypatch):
    """The mirror-image risk: an env gate that is always False makes the two
    tests above pass for the wrong reason, and the real guardian silently
    stops notifying. Opt one test back in and require both commands."""
    import notify

    seen: list[list[str]] = []

    def _record_run(cmd, timeout=5.0):
        # Records the command and reports success, which is the real `_run`'s
        # answer once `notify-send` has been spawned — the `res[... ] is True`
        # assertions below are only reachable through that return value.
        seen.append(list(cmd))
        return True

    monkeypatch.setattr(notify, "_run", _record_run)
    monkeypatch.setenv("LLOYD_DESKTOP_ALERTS", "1")
    monkeypatch.setenv("LLOYD_JOURNAL_ALERTS", "1")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    (tmp_path / "obsidian" / "memory").mkdir(parents=True)
    n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                        vault_root=str(tmp_path / "obsidian"),
                        backend_url="http://127.0.0.1:1")

    res = n.alert("critical", "real rollback", "the detail text")

    flat = " ".join(" ".join(c) for c in seen)
    assert "notify-send" in flat and "-u critical" in flat, flat
    assert "Lloyd guardian: real rollback" in flat, flat
    assert "the detail text" in flat, "the toast lost its body"
    assert "systemd-cat" in flat, flat
    assert res["desktop"] is True and res["journal"] is True, res


def test_the_drill_passes_no_external_alerts(tmp_path):
    """The flag is only useful if rehearse.py actually sends it."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent /
           "scripts" / "automod" / "rehearse.py").read_text()
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


# ---------------------------------------------------------------------------
# An alert that asks for a human must land in a queue (backlog #775)
#
# `Notifier.alert` filed a backlog task only on `level == "critical" or trigger`.
# The guardian's three "cannot act on this" sites all fire `level="error"` with
# no trigger — two of them ending in the words "this needs a human" — so the one
# channel that becomes work a human later sees was skipped, while the four
# surfaces that did fire (ALERT.md is last-writer-wins, the vault note lands in a
# 300-section daily note, toast and voice are ephemeral) were none of them a
# queue. Measured: 15 "Service down, but no promotion to revert" rows in
# promotions.jsonl from 2026-09-06 to 2026-09-14, and no backlog item for any of
# them. Every observable below is the *dict key*, not its bool: the POST goes to
# a dead port, so `"backlog" in res` proves the routing decision while
# `res["backlog"] is False` would only prove the backend was down.
# ---------------------------------------------------------------------------

# Hand-written, modelled on the sentence at guardian.py's unobserved-liveness
# site — it is NOT read out of guardian.py, so rewording that call site would
# leave clause 1's test green on its own. Two other tests are what actually
# guard the prose: `test_the_three_sites_that_cannot_act_all_ask_for_a_human`
# requires each of the three sites to pass the flag, and requires
# `notify.asks_for_a_human` to still match text that is really in guardian.py, so
# the fallback cannot rot into dead code while this constant keeps filing.
NEEDS_HUMAN_BODY = (
    "lloyd-mc:lloyd-backend: STOPPED without an intentional stop\n\n"
    "HEAD is 1234abcd and no self-modification is being observed, so this is "
    "infrastructure rather than a bad change. Not rewriting history — this needs "
    "a human."
)

# Clause 2's no-spam fixture: title and body transcribed verbatim from the
# `supervisord was unreachable` site in guardian.py — the one cannot-act-shaped
# alert the guardian files at `level="error"` with no trigger and no declaration,
# *after* it has already fixed the problem by restarting
# `agent-supervisord.service`, and which fires again on every tick the supervisor
# stays unreachable. It is the fixture that makes clause 2 bite: a route keyed on
# severity alone would file a board item for this one every
# `policy.ALERT_REPEAT_SECONDS` forever. `test_the_no_spam_fixture_is_live` keeps
# the transcription honest.
SUPERVISORD_RESTART = (
    "supervisord was unreachable",
    "Restarted agent-supervisord.service. No code was reverted — an unreachable "
    "supervisor is infrastructure, not a bad promotion.",
)


def _routing_notifier(tmp_path):
    """A Notifier that cannot deliver, so the returned keys carry no other
    variable. `memory/` has to exist or the vault channel fails too and its
    False reads as part of the result under test."""
    import notify

    (tmp_path / "obsidian" / "memory").mkdir(parents=True, exist_ok=True)
    return notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                           vault_root=str(tmp_path / "obsidian"),
                           backend_url="http://127.0.0.1:1")


def test_needs_human_alert_is_routed_to_a_backlog_task(tmp_path):
    """Clause 1. An error-level alert whose body declares it needs a human is
    routed to `_backlog_task`. Before this change the gate read level and
    trigger only, so this exact call returned ledger/journal/desktop/voice/vault
    and no `backlog` key — which is why 15 real notices filed nothing."""
    res = _routing_notifier(tmp_path).alert(
        "error", "Service down, but no promotion to revert", NEEDS_HUMAN_BODY)

    assert "backlog" in res, f"needs-a-human alert filed no task: {sorted(res)}"


def test_plain_error_alert_still_files_no_backlog_task(tmp_path):
    """Clause 2. The route is a declaration, not a severity, and the fixture is
    the loudest self-healing alert the guardian owns: the `supervisord was
    unreachable` site, which files at `level="error"` with no trigger and no
    declaration *after* it has restarted `agent-supervisord.service` itself and
    returns `infra_down`. It re-fires for as long as the supervisor stays
    unreachable, so a gate keyed on severity alone would hand the board a new
    item every `ALERT_REPEAT_SECONDS` for a condition the guardian already
    fixed — which is the failure this clause exists to rule out."""
    res = _routing_notifier(tmp_path).alert("error", *SUPERVISORD_RESTART)

    assert "backlog" not in res, f"an unlabelled error alert reached the board: {res}"


def test_the_no_spam_fixture_is_live():
    """Clause 2's bite depends on its fixture being a live site. The title and
    body above are transcribed from guardian.py, not read out of it, so a reword
    would leave this file asserting about a sentence no longer in the tree. The
    clause rules out a severity-only gate only while that site is still
    error-level, still carries no `needs_human` flag, and still never says the
    phrase — all four are checked here. Parsed rather than text-matched because
    the claims are about argument *positions* (which argument is the level, which
    the body), which a slice of source text cannot answer."""
    import ast

    import notify

    src = (Path(__file__).resolve().parent.parent /
           "agent-services" / "guardian" / "guardian.py").read_text()
    site = None
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "alert"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and len(node.args) >= 3
                and getattr(node.args[1], "value", None) == SUPERVISORD_RESTART[0]):
            site = node
    assert site is not None, "the `supervisord was unreachable` call site is gone"
    assert ast.literal_eval(site.args[0]) == "error", (
        "that site is no longer error-level, so it stops separating a "
        "severity-only gate from a declaration-only one")
    assert ast.literal_eval(site.args[2]) == SUPERVISORD_RESTART[1], (
        "guardian.py reworded that alert; re-transcribe SUPERVISORD_RESTART "
        "instead of editing this assert to match")
    assert not any(k.arg == "needs_human" for k in site.keywords), (
        "the site now declares itself, so it is no longer the unlabelled case")
    assert not notify.asks_for_a_human(*SUPERVISORD_RESTART), (
        "the fixture text now trips the prose fallback, so clause 2's negative "
        "case has quietly become a positive one")


def test_the_supervisord_restart_alert_carries_evidence_to_the_ledger(tmp_path, monkeypatch):
    """Clause 4 of #1178, the supervisord half. Every `supervisord was
    unreachable` row in the ledger carried `evidence: ""`, so the 2026-09-15
    oomd kill of the whole unit (620 processes) and a slow socket left the same
    record. The site must hand `notify.alert(evidence=)` what it saw — the
    streak, the probe's own error string and the restart's rc — and the row
    must carry it. Run through the real Notifier with external channels off so
    the assertion is on the ledger file, not on a stub's kwargs; `systemctl` is
    intercepted, because the real call would restart production's supervisor."""
    import subprocess
    import types

    import gstate
    import guardian as G
    import policy

    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "s"),
        guardian_state=str(tmp_path / "g"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0, no_external_alerts=True,
    )
    g = G.Guardian(args)
    for quiet in ("drain_logs", "check_vault", "check_data", "check_memory"):
        monkeypatch.setattr(g, quiet, lambda: None)
    monkeypatch.setattr(g, "collect", lambda: {
        "now": NOW, "supervisord": "unreachable", "procs": {}, "probes": {},
        "supervisord_error": "[Errno 111] Connection refused (/tmp/agent-supervisor.sock)"})
    ran: list = []

    def _fake_run(argv, **kw):
        ran.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(G.subprocess, "run", _fake_run)
    g.sup_down_streak = policy.SUPERVISORD_DOWN_STREAK - 1

    assert g.tick() == "infra_down"
    assert ran == [["systemctl", "--user", "restart", policy.SUPERVISORD_UNIT]]

    rows = [r for r in gstate.read_events(g.state.ledger)
            if r.get("event") == "alert" and r.get("title") == SUPERVISORD_RESTART[0]]
    assert len(rows) == 1, rows
    evidence = rows[0]["evidence"]
    assert evidence, "the ledger row still carries no evidence"
    assert "Connection refused" in evidence, evidence
    assert f"{policy.SUPERVISORD_DOWN_STREAK} consecutive ticks" in evidence, evidence
    assert "rc=0" in evidence, evidence
    assert rows[0]["body"] == SUPERVISORD_RESTART[1], "the body is the no-spam fixture; evidence rides beside it"


def test_needs_human_flag_routes_an_alert_that_never_says_so(tmp_path):
    """Clause 3. The family has three members and only two say the words.
    "Guardian self-test failed" — fired 2026-09-10T13:26:58Z and
    2026-09-15T20:34:52Z per the guardian ledger — contains no such phrase, so
    prose matching alone covers two of three and breaks on every reword. The
    explicit flag is what covers the whole family."""
    res = _routing_notifier(tmp_path).alert(
        "error", "Guardian self-test failed",
        "The watchdog can no longer perform one of its own preconditions.",
        needs_human=True)

    assert "backlog" in res, f"needs_human=True did not route: {sorted(res)}"


def test_critical_and_trigger_routing_is_unchanged(tmp_path):
    """Clause 4. The two routes that already worked are untouched: `critical`
    (every `escalate()`) and a `trigger`-carrying rollback still file."""
    n = _routing_notifier(tmp_path)

    assert "backlog" in n.alert("critical", "Rollback floor breached",
                                "target predates the floor"), "critical stopped filing"
    assert "backlog" in n.alert("error", "Rolled back to last known good",
                                "reverted a1b2c3d4", trigger="crash"), \
        "a triggered rollback stopped filing"


def test_drill_alert_stays_local_even_while_asking_for_a_human(tmp_path):
    """Clause 4, second half. `external=False` returns above every external
    channel, and the new route must sit *behind* that return: on 2026-09-06 two
    drill rollbacks landed in the live vault daily note naming commits that only
    existed in a deleted scratch clone."""
    import notify

    vault = tmp_path / "obsidian" / "memory"
    vault.mkdir(parents=True)
    n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                        vault_root=str(tmp_path / "obsidian"), external=False)

    res = n.alert("error", "drill: service down", NEEDS_HUMAN_BODY, needs_human=True)

    assert set(res) == {"ledger", "alert_file"}, res
    assert list(vault.glob("*.md")) == [], "a drill wrote into the vault"


def test_the_needs_human_route_is_covered_by_repeat_suppression(tmp_path, monkeypatch):
    """Clause 5. Suppression lives in `Guardian.alert`, above the notifier, so a
    route added inside `Notifier.alert` inherits it only if the flag survives the
    wrapper's `**kw` forwarding — and the observable is filings, not fan-outs. A
    condition that lasts days ticks every 5 s and re-announces once per
    `policy.ALERT_REPEAT_SECONDS`; each of those must stay one board item.

    The body carries no prose marker on purpose, and the title is the self-test
    site's, which never says the words either: this is the one test that drives
    the *flag* through the real wrapper, so a `Guardian.alert` that stopped
    forwarding `**kw` would file nothing here and go red. Give it a marker in the
    body and the prose fallback routes the alert, the assert still sees one
    filing, and the flag becomes untested.

    The real `Notifier.alert` runs here (journal, toast and voice are env-muted
    for the suite by conftest; the vault root and the backend are pointed away
    from production, and `_backlog_task` is the one channel counted instead of
    POSTed)."""
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
    g.notifier.external = True
    g.notifier.vault_root = tmp_path / "obsidian"
    (tmp_path / "obsidian" / "memory").mkdir(parents=True)
    filed: list = []

    def _file(title, text, commit, tag):
        """Counts the filing. The bool mirrors `_backlog_task`'s real return so
        the stub cannot be the reason an assertion passes — `filed` is the
        observable here, and it is what the asserts below read."""
        filed.append(title)
        return True

    monkeypatch.setattr(g.notifier, "_backlog_task", _file)

    for _ in range(3):
        g.alert("error", "Guardian self-test failed",
                "The watchdog can no longer perform one of its own preconditions.",
                needs_human=True)
    assert len(filed) == 1, (
        f"{len(filed)} tasks filed: a repeat inside one ALERT_REPEAT_SECONDS "
        "window re-files the board")

    # The window expiring files the next one, so this is suppression, not a mute.
    g._alert_seen["Guardian self-test failed"] -= policy.ALERT_REPEAT_SECONDS + 1
    g.alert("error", "Guardian self-test failed",
            "The watchdog can no longer perform one of its own preconditions.",
            needs_human=True)
    assert len(filed) == 2, f"an expired window must file again, got {len(filed)}"


@contextlib.contextmanager
def _stub_board(tmp_path, reply: dict):
    """A loopback stand-in for the backend's task-create endpoint, yielding the
    server and a `seen` dict holding the path and the decoded JSON body — the
    bytes `backlog_task_create` would have written a task file from.

    `reply` is what the caller says the backend answers, and the caller passes
    the shape `app/routers/backlog.py::backlog_task_create` really returns,
    `{"success": true, "id": N}`. It does **not** echo the request's `name`, and
    no branch of `_backlog_task` reads a name any more, so every reply shape the
    tests below pass is one the real endpoint can actually send."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["path"] = self.path
            seen["body"] = json.loads(
                self.rfile.read(int(self.headers["Content-Length"])).decode())
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (tmp_path / "obsidian" / "memory").mkdir(parents=True)
        yield server, seen
    finally:
        server.shutdown()
        server.server_close()


def _board_notifier(server, tmp_path):
    import notify

    return notify.Notifier(
        ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
        vault_root=str(tmp_path / "obsidian"),
        backend_url=f"http://127.0.0.1:{server.server_address[1]}")


def test_the_needs_human_route_posts_the_payload_the_board_reads(tmp_path):
    """Crosses the one process boundary this fix opens traffic to: the guardian
    process POSTs to `/api/backlog/task-create`, and the board's task file is
    written from that body's `name`/`description`/`status`. Keys are not
    validated leniently in the caller's favour — a payload sending `title`/`body`
    instead of `name`/`description` still gets 2xx, and the board keeps a task
    called "New Task" that says nothing — so `"backlog" in res` is not evidence
    across this seam. The evidence is `seen["body"]`.

    `res["backlog"]` is asserted last, for what it can prove and no more: the
    reply carries the real endpoint's shape, `{"success": true, "id": 999}`, so the
    value is `_backlog_task` accepting a filing. What makes that a decision rather
    than a constant is `test_no_board_reply_lacking_a_positive_integer_id_reads_as_delivered`,
    which fails this same call on nine replies that carry no positive integer id."""
    with _stub_board(tmp_path, {"success": True, "id": 999}) as (server, seen):
        res = _board_notifier(server, tmp_path).alert(
            "error", "Service down, but no promotion to revert", NEEDS_HUMAN_BODY)

    assert seen.get("path") == "/api/backlog/task-create", seen
    assert seen["body"]["name"] == ("[guardian] Service down, but no promotion "
                                    "to revert"), "the `name` key drifted"
    assert seen["body"]["status"] == "draft", (
        "autotriage reads `draft` alone, and an untriaged item at the implement "
        "pool's status is a dead state (#1990); `backlog_task_create` also 400s a "
        "status outside _VALID_STATUSES")
    assert seen["body"]["board"] == "lloyd", (
        "the board is named, not inherited from the route's default (#1990)")
    assert seen["body"]["priority"] == "high"
    assert tuple(seen["body"][k] for k in ("board", "status", "priority")) == (
        "lloyd", "draft", "high")
    # #1990's own grep, pinned: no create payload under the guardian posts the
    # implement pool's status, and the comment beside the payload says why.
    dead = '"status": ' + '"up_next"'
    guardian_dir = Path(__file__).resolve().parent.parent / "agent-services" / "guardian"
    offenders = [p.name for p in sorted(guardian_dir.glob("*.py"))
                 if dead in p.read_text(encoding="utf-8")]
    assert offenders == [], offenders
    source = (guardian_dir / "notify.py").read_text(encoding="utf-8")
    block = source[source.index("# Field names match app/routers/backlog.py"):
                   source.index("payload = json.dumps(")]
    assert "TRIAGE_POOL_STATUS" in block and "ready_confirmed" in block
    assert "confirmed triage verdict" in " ".join(block.replace("#", " ").split())
    assert "needs a human" in seen["body"]["description"], (
        "the filed task lost the sentence that asked for the human")
    assert res["backlog"] is True, f"a 2xx filing read as a failure: {res}"


def test_a_defaulted_name_from_the_board_reads_as_a_failed_filing(tmp_path):
    """The other side of that same seam. A drifted payload is not an error: the
    endpoint answers 2xx and files a task called "New Task", which is how a
    routing fix that posts the wrong keys would quietly report delivered forever.

    #1612 moved the verdict off the reply's `name` — the endpoint never echoes one,
    so reading it was reading a field that is always absent — onto `success` and a
    positive integer `id`. This reply still reads as a failure, because it carries
    no id. What it no longer proves is the drift itself: a payload that posts the
    wrong keys gets an id for the empty task it created, and production will not
    catch that either — payload-drift detection declined 2026-09-29 (#1703): the
    echo variant shipped at 7da1e0e4 and was inert because task-create replies only
    {success, id}, and a read-back buys the same verdict one request later on the
    alert path. What pins the seam instead is the written-file assertion in
    `test_the_guardians_captured_body_files_itself_through_the_real_route`, which
    reads the task file the real route wrote; weaken that and live detection is
    owed again.
    `test_no_board_reply_lacking_a_positive_integer_id_reads_as_delivered` is where
    the reason this returns False is pinned."""
    with _stub_board(tmp_path, {"success": True, "name": "New Task"}) as (server, seen):
        res = _board_notifier(server, tmp_path).alert(
            "error", "Service down, but no promotion to revert", NEEDS_HUMAN_BODY)

    assert seen["body"]["name"].startswith("[guardian] "), seen
    assert res["backlog"] is False, (
        "a board that answered with someone else's task was reported as a "
        "delivered guardian filing")


# ── The guardian-to-board contract, replayed through the real route (#1703) ───
#
# `notify.py` and `app/routers/backlog.py` run in different processes and share one
# thing: a JSON shape. Until #1703 each half was pinned only against a stand-in for
# the other — the tests above POST into `_stub_board`, which answers a hand-written
# dict and never writes a file, while the route-side tests
# (`tests/test_backlog_okf_frontmatter.py`, `tests/test_backlog_route_cache.py`)
# POST payloads they wrote themselves and name `notify.py` only in a docstring. So
# an edit to either side that desynchronised the keys passed the gate green: the two
# sets of fixtures never met. These tests meet them in one run — the guardian's
# captured bytes go into the real route.


class _CapturedRequest:
    """The one thing `backlog_task_create` asks of its argument: `await request.json()`.

    It holds on to the object it was handed — `received` — so a test can assert the
    route was given the guardian's captured body *itself*. Rebuilding the payload
    from a literal would satisfy every other assertion in the file and leave this
    seam unpinned, which is the failure mode the whole section exists to close.
    """

    def __init__(self, payload: dict):
        self.received = payload

    async def json(self) -> dict:
        return self.received


def _replay_on_the_route(payload: dict, board_dir, monkeypatch):
    """Feed `payload` to the real `backlog_task_create` and return (request, reply dict).

    Two monkeypatches, both required: `_BACKLOG_DIR` is where the file lands (the
    real one is the live vault board), and `_backlog_board_map()` is a corpus scan of
    the whole vault that the route consults for the board name — patched to empty
    exactly as `tests/test_backlog_okf_frontmatter.py::http_board` does it, so a
    guardian create carrying no `board` key resolves to `DEFAULT_BOARD` without
    reading the real corpus.
    """
    import asyncio
    import json

    import app.routers.backlog as BR

    monkeypatch.setattr(BR, "_BACKLOG_DIR", board_dir)
    monkeypatch.setattr(BR, "_backlog_board_map", lambda: {})
    request = _CapturedRequest(payload)
    response = asyncio.run(BR.backlog_task_create(request))
    assert response.status_code == 200, response.body
    return request, json.loads(response.body)


def _filed_task(path):
    """(front matter dict, first line of the body) for a file the route wrote."""
    import re

    import yaml

    fence = re.compile(r"\A---\n(.*?)\n---\n", re.S)
    text = path.read_text(encoding="utf-8")
    match = fence.match(text)
    assert match, f"{path.name} opens with no front matter fence:\n{text[:200]}"
    return yaml.safe_load(match.group(1)), text[match.end():].lstrip().splitlines()[0]


def _assert_reply_reads_as_a_filing(reply: dict) -> None:
    """The verdict `notify.py:547-551` reaches, stated once, for both of its tests.

    `_backlog_task` returns True only on `success is True` AND an `id` that is an
    `int`, not a `bool`, and greater than zero. Shared by the route replay below and
    by `test_a_reply_of_success_false_with_a_positive_id_fails_the_verdict`, whose
    whole point is that a positive id is not enough — a reply of
    `{"success": false, "id": 7}` has to fail *this* assertion, not a looser one
    written beside it.
    """
    assert reply.get("success") is True, f"`success` is not True: {reply!r}"
    row_id = reply.get("id")
    assert (isinstance(row_id, int) and not isinstance(row_id, bool)
            and row_id > 0), f"`id` is not a positive non-bool int: {reply!r}"


def test_the_guardians_captured_body_files_itself_through_the_real_route(
        tmp_path, monkeypatch):
    """Clauses 1-3 of #1703: one payload, generated by the guardian, read by the route.

    The request is produced by the real thing — `Notifier.alert` over the loopback
    seam in `_stub_board`, whose `seen["body"]` is the decoded JSON the guardian
    actually sent — and those bytes, unedited, are what the real
    `backlog_task_create` receives. Nothing here copies a dict.

    The reply is then asserted with the same helper the guardian's own decision uses
    (clause 2), and the file the route wrote is asserted too (clause 3), because the
    reply alone cannot see key drift: the endpoint defaults a nameless create to
    `"New Task"` (`app/routers/backlog.py:707`) and answers 200 with an id either
    way. The second half of this test demonstrates that on the same route — a payload
    sending `title`/`body` instead of `name`/`description` still gets `success: true`
    and a positive id, and writes `# New Task` with none of the alert's text. Only
    the file assertion tells those two filings apart (by H1 and body: both land at
    `status: draft` since #1990), which is why it is in the test, and it
    is the only guard either side of the seam will ever have: payload-drift
    detection declined 2026-09-29 (#1703): the echo variant shipped at 7da1e0e4 and
    was inert because task-create replies only {success, id}, and a read-back buys
    the same verdict one request later on the alert path. This test's written-file
    assertion is what pins that seam, so weakening it leaves production blind to key
    drift and makes live detection owed again.
    """
    with _stub_board(tmp_path, {"success": True, "id": 999}) as (server, seen):
        res = _board_notifier(server, tmp_path).alert(
            "error", "Service down, but no promotion to revert", NEEDS_HUMAN_BODY)
    assert res["backlog"] is True, f"the guardian did not report a filing: {res}"
    assert seen.get("path") == "/api/backlog/task-create", seen

    request, reply = _replay_on_the_route(
        seen["body"], tmp_path / "board", monkeypatch)
    assert request.received is seen["body"], (
        "the route was handed something other than the guardian's captured body, so "
        "what follows describes a payload the guardian never sent")
    _assert_reply_reads_as_a_filing(reply)

    written = sorted((tmp_path / "board").glob("*.md"))
    assert len(written) == 1, f"the route wrote {len(written)} files: {written}"
    fm, h1 = _filed_task(written[0])
    assert h1 == "# [guardian] Service down, but no promotion to revert", (
        f"the filed item's H1 is not the alert's title: {h1!r}")
    assert fm["status"] == "draft", fm
    assert fm["board"] == "lloyd", fm
    assert fm["priority"] == "high", fm
    assert "needs a human" in written[0].read_text(encoding="utf-8"), (
        "the `description` key did not reach the file's body, so the alert text the "
        "board shows a human is not the alert text the guardian wrote")

    # The falsifying half, on the same route and the same helpers: the drifted keys
    # the reply cannot see. `status` is absent because a payload renamed wholesale
    # sends no status either, and the route defaults that field to `draft`.
    drifted_payload = {"title": seen["body"]["name"],
                       "body": seen["body"]["description"]}
    drifted_request, drifted_reply = _replay_on_the_route(
        drifted_payload, tmp_path / "board-drifted", monkeypatch)
    assert drifted_request.received is drifted_payload
    _assert_reply_reads_as_a_filing(drifted_reply)
    drifted = sorted((tmp_path / "board-drifted").glob("*.md"))
    assert len(drifted) == 1, drifted
    drifted_fm, drifted_h1 = _filed_task(drifted[0])
    assert drifted_h1 == "# New Task", (
        "the drifted payload was supposed to show the reply's blindness to a "
        f"defaulted name, got {drifted_h1!r}")
    assert drifted_fm["status"] == "draft", drifted_fm
    # Since #1990 the intended filing lands at `draft` too, so status no longer
    # tells the two apart. What does: the H1, and the alert text in the body.
    assert drifted_fm["status"] == fm["status"]
    assert drifted_h1 != h1
    assert "needs a human" not in drifted[0].read_text(encoding="utf-8"), (
        "the drifted payload's `body` key reached the file, so the body no longer "
        "discriminates a drifted filing")


def test_a_reply_of_success_false_with_a_positive_id_fails_the_verdict(tmp_path):
    """Clause 4 of #1703: a positive id is not a filing on its own.

    `{"success": false, "id": 7}` is the reply the pre-#1612 guard accepted — it read
    a `name` the endpoint never sends, took the `if name else True` default, and
    never consulted `success`. It is also not in the existing parametrised reply
    table further down this file, which pairs a false `success` with no id at all
    (`test_a_board_reply_saying_it_failed_is_not_delivered_though_it_was_2xx`); the
    id on the row is what makes it a distinct case, since the id is now one of the
    two fields the verdict is made of.

    Both halves are asserted: the guardian's own verdict across the loopback seam is
    False, and the shared assertion the route replay is graded by
    (`_assert_reply_reads_as_a_filing`) fails on it. Proving only the first would
    leave clause 2's assertion free to pass a reply the guardian rejects.
    """
    reply = {"success": False, "id": 7}

    assert _board_verdict(tmp_path, reply) is False, (
        "a board that answered it failed, while handing back an id, read as a "
        "delivered guardian filing")

    with pytest.raises(AssertionError, match="not True"):
        _assert_reply_reads_as_a_filing(reply)


# ── The retired owed pointer (#1827) ──────────────────────────────────────────
#
# Three passages used to hand a decision to a round that would never come: the
# guardian's filing call in `agent-services/guardian/notify.py`, and the two
# docstrings of the tests that pin the guardian-to-board seam. Each said the
# payload-drift check was owed to the closed item whose number is in
# `RETIRED_ITEM` below — spelt in two pieces here because the acceptance grep for
# that phrase covers this file too, and this file has to search for it without
# carrying it. #1703 ruled on it on 2026-09-29 and both parents are `done`, so the
# comments were the only surface still asking. What is left to protect is the
# retraction: a future edit that writes the phrase again, or that empties the
# passages it replaced, has to go red.

RETIRED_ITEM = 1612
RETIRED_POINTER = re.compile(r"ruling 2 on #" + str(RETIRED_ITEM))

# The ruling each replaced passage has to carry, verbatim apart from wrapping.
RULING = ("payload-drift detection declined 2026-09-29 (#1703): the echo variant "
          "shipped at 7da1e0e4 and was inert because task-create replies only "
          "{success, id}, and a read-back buys the same verdict one request later "
          "on the alert path")
# The one-line note the retraction owes the next reader: the decline is only
# sound while the seam stays pinned, and this names what un-pins it.
OWED_AGAIN = "owed again"
SEAM_TEST = "test_the_guardians_captured_body_files_itself_through_the_real_route"


# The three passages #1827 rewrote, addressed by the function whose body carries
# them rather than by a line number: this test sits below two of them, so a range
# quoted here moved in the very diff that wrote it. `None` as a path means this
# file — the two test docstrings.
PASSAGES = (
    ("agent-services/guardian/notify.py", "_backlog_task",
     "the guardian's filing call"),
    (None, "test_a_defaulted_name_from_the_board_reads_as_a_failed_filing",
     "the reply-side drift test"),
    (None, SEAM_TEST, "the route-replay seam test"),
)


def _flat(text):
    """Source text with each line's leading `#` dropped, every run of whitespace
    collapsed and case folded, so a sentence that wraps across comment lines still
    matches the sentence it was written as. Matching raw bytes would make this test
    pass or fail on wrapping width, which is not what any of these clauses is about.

    Comments have to be unwrapped rather than skipped, because in `notify.py` the
    ruling *is* a comment: stripping comments out would leave nothing to find.
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        lines.append(stripped.lstrip("#").strip() if stripped.startswith("#")
                     else stripped)
    return re.sub(r"\s+", " ", " ".join(lines)).lower()


def _func_source(path, name):
    """The source segment of function `name` in `path`, or "" if there is none.

    Located by AST, not by slicing on the name: a mention of a function's name in a
    docstring is not its body, and the whole point here is to grade the body."""
    import ast

    src = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    return ""


def _asserts(path, name):
    """The source of every `assert` statement inside function `name` in `path`.

    Read off the AST rather than by searching the function's text, because that
    function's own docstring *describes* the strings its assertions must contain: a
    substring hit could then be prose that outlived the check it names, which is the
    failure #1827 is retracting, one store closer to home. An `ast.Assert` is the
    check, and nothing else."""
    import ast

    src = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [re.sub(r"\s+", " ", ast.get_source_segment(src, n) or "")
                    for n in ast.walk(node) if isinstance(n, ast.Assert)]
    return []


def _comments(path):
    """Every comment token in `path`, unwrapped and collapsed — the prose a reader
    of the source sees and the interpreter never does. Clause 4 of #1827 says the
    ruling added to `notify.py` is a comment, so it has to be findable HERE."""
    import io
    import tokenize

    text = path.read_text(encoding="utf-8")
    kept = [tok.string.lstrip("#").strip()
            for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            if tok.type == tokenize.COMMENT]
    return re.sub(r"\s+", " ", " ".join(kept)).lower()


def _string_literals(path):
    """Every string literal in `path`'s syntax, collapsed — the other half of the
    clause-4 check. A ruling parked in a docstring or a constant would be found
    here and not in `_comments`, and would be a behaviour change in the file the
    clause forbids one in: adjacent literals are already one `ast.Constant` here,
    so wrapping the sentence across lines cannot smuggle it past this."""
    import ast

    parts = [node.value for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
             if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    return re.sub(r"\s+", " ", " ".join(parts)).lower()


def test_the_retired_owed_pointer_is_gone_and_the_closed_ruling_is_in_its_place():
    """#1827: an instruction must not outlive the decision that closed it.

    Both directions are asserted, because either half alone is satisfiable by an
    empty file. Absence (clause 1): the retired pointer appears nowhere in the two
    files the acceptance grep names — the same two files, never the whole tree,
    because that wording beside a different item number is a different decision and
    lives elsewhere in the repo, and a fleet-wide red here would be this test
    reporting someone else's open question. Re-adding it anywhere in either file
    fails here first.

    Presence (clauses 2 and 3): for each of the three replaced passages — located
    by the function that carries it, `PASSAGES` — the closed ruling sentence, the
    name of the test that owns the seam, and the note about what would make live
    detection owed again. Addressing the passages rather than the files is what
    stops the presence checks being satisfied by this test's own constants, which
    live in the same file: a passage emptied of the ruling goes red even though
    `RULING` is still a string further down.

    Three more assertions keep the credit from being decorative.

    Clause 4 says the ruling added to `notify.py` is a comment and nothing else, so
    it is read twice: once out of the comment tokens, where the sentence has to be,
    and once out of the string literals, where it must not be — prose in a docstring
    or a constant is still a non-comment hunk and still describes a check nobody
    runs from the call site. `_comments`/`_string_literals`, not one flattened blob,
    because a sentence is prose in exactly one of those stores and the whole clause
    is about which.

    The seam all three passages credit is then checked at its own assertions —
    `_asserts`, the `ast.Assert` nodes of `SEAM_TEST` — for the two written-file
    checks it is credited with: the filed H1 equal to the alert title, and the
    falsifying `== "# New Task"`. Not by searching the function's text: that function's
    docstring already *describes* both strings, so a substring match there stays green
    after the assertion itself is deleted, which is this item's whole failure mode
    reproduced one store closer to home.

    `SEAM_TEST` is named rather than given a line range for the same reason: the
    range the acceptance clauses were written against moved when this round edited the
    two docstrings above it, so a quoted range would have been false as it was typed.
    """
    repo = Path(__file__).resolve().parent.parent
    notify = repo / "agent-services" / "guardian" / "notify.py"
    here = Path(__file__).resolve()

    for label, path in (("the guardian's filing call", notify),
                        ("this contract test file", here)):
        assert path.is_file(), f"{label} is not on disk at {path}"
        hit = RETIRED_POINTER.search(_flat(path.read_text(encoding="utf-8")))
        assert hit is None, (
            f"{label} tells a future round the payload-drift check is owed (found "
            f"{hit.group(0)!r}). #1703 declined it on 2026-09-29 and both parents "
            "are done; what belongs in that sentence is the ruling, not the ask")

    for rel, func, label in PASSAGES:
        path = here if rel is None else repo / rel
        segment = _func_source(path, func)
        assert segment, f"{label}: no function {func!r} in {path}, so no passage"
        flat = _flat(segment)
        assert not RETIRED_POINTER.search(flat), (
            f"{label} still hands the decision to a future round")
        assert RULING.lower() in flat, (
            f"{label} no longer states the closed ruling, so a reader of it learns "
            f"only that something is missing. Expected: {RULING}")
        assert SEAM_TEST.lower() in flat, (
            f"{label} does not name {SEAM_TEST} as what pins the seam it declined "
            "to police in production")
        assert OWED_AGAIN in flat, (
            f"{label} lost the note that weakening the written-file assertion makes "
            "live drift detection owed again")

    notify_comments = _comments(notify)
    assert RULING.lower() in notify_comments, (
        "the ruling in notify.py is not sitting in a comment, so clause 4's "
        "no-non-comment-hunk rule is already broken: " + RULING)
    assert SEAM_TEST in notify_comments, (
        "notify.py names the seam test outside its comments, which would make a "
        "production module depend on a test file")
    notify_literals = _string_literals(notify)
    assert RULING.lower() not in notify_literals, (
        "the #1827 ruling is a docstring or string constant in notify.py, not a "
        "comment — prose in a string is code the clause forbids, and prose in the "
        "module docstring is prose no reader of the filing call ever reaches")

    seam_asserts = _asserts(here, SEAM_TEST)
    assert seam_asserts, (
        f"{SEAM_TEST} is gone from this file or holds no assertion at all, so the "
        "seam every replaced passage credits has no owner left to credit")
    assert any('== "# [guardian] Service down, but no promotion to revert"' in a
               for a in seam_asserts), (
        "the filed file's H1 is no longer asserted there, so the reply's blindness "
        "to key drift is now invisible at test time as well as in production — and "
        "three comments would still be crediting it")
    assert any('== "# New Task"' in a for a in seam_asserts), (
        "the falsifying half of the seam test is gone: without the drifted "
        "payload's defaulted H1, the H1 assertion above it proves nothing")


def test_the_module_docstring_names_the_drill_that_exists_not_the_phantom():
    """Clause 5 of #1703: the file's own header may not cite a script that never was.

    `git log --all --oneline -- '*guardian_drill*'` returns nothing and no tracked
    path matches it — until #1703 this docstring was the sole reference to that file
    anywhere in the tree, which is the sort of claim a reader trusts and a future
    round then writes a test against. The replacement sentence has to hold up too, so
    the script it does name is checked to exist and to send the flag it is cited as
    sending. The check is scoped to the module docstring rather than the whole file
    because this test has to spell the phantom's name to rule it out.
    """
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    doc = sys.modules[__name__].__doc__ or ""
    assert "guardian_drill" not in doc, (
        "the module docstring cites scripts/guardian_drill.py again; `git log --all "
        "-- '*guardian_drill*'` is empty — that script has never existed")
    assert "scripts/automod/rehearse.py" in doc, (
        "the docstring does not name the drill that really runs the guardian")
    assert "--no-external-alerts" in doc, (
        "the docstring does not record that the drill deliberately does not file")
    assert "guardian.py:1390" in doc, (
        "the docstring does not point at where that flag is parsed")
    assert (repo / "scripts" / "automod" / "rehearse.py").is_file(), (
        "scripts/automod/rehearse.py, named by the docstring, is not on disk")
    guardian_src = (repo / "agent-services" / "guardian" /
                    "guardian.py").read_text(encoding="utf-8")
    assert "--no-external-alerts" in guardian_src, (
        "guardian.py no longer parses the flag the docstring says the drill passes")
    assert ("test_the_guardians_captured_body_files_itself_through_the_real_route"
            in doc), "the docstring does not name the test that proves the filing"


def test_the_three_sites_that_cannot_act_all_ask_for_a_human():
    """Two things the unit tests cannot see. (1) The flag only routes if the call
    sites send it: `Guardian.alert` forwards `**kw` unchanged, so a site that
    omits it is routed by prose alone — and the self-test sentence has no prose
    to route on. (2) The prose fallback still matches a live sentence, so it
    cannot quietly become dead code while a hand-written body keeps clause 1
    green. A call site is not reachable from a unit test without a live
    supervisor and a promoted commit, so this reads guardian.py's source instead
    of executing it — the same trade `test_the_drill_passes_no_external_alerts`
    makes for the drill's flag, tightened to call nodes for the reason below.

    The sites come from the AST, not from splitting the text on `self.alert(`.
    Text splitting also yields a chunk for every *mention* of that string, and the
    600 characters after the `escalate()` call run past its closing paren into the
    repeat-suppression comment in the `def alert` wrapper below it — which quotes
    the title `"Service down, but no promotion to revert"`. That phantom site
    carries no flag, so an `any()` over chunks passed while telling the truth
    about only one of two matches, and a filter on the level literal did not
    remove it because `escalate` really does call with `"critical"`. A parsed call
    node is the call site: `ast.get_source_segment` returns its own arguments and
    nothing else. `self.notifier.alert(...)` is a different receiver and is
    excluded, which is what we want — this is about the guardian's own sites."""
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent /
           "agent-services" / "guardian" / "guardian.py").read_text()
    sites = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "alert"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"):
            sites.append(ast.get_source_segment(src, node))
    # The denominator beside the count: guardian.py has 9 `self.alert(` call sites
    # today. A drop means the extractor stopped matching a real shape — an alert
    # reached through `self.alert` renamed or called off an alias would otherwise
    # vanish from this test in silence, and a route nobody calls is exactly the
    # defect this file exists to catch. Growth is normal, so this is a floor.
    assert len(sites) >= 9, f"only {len(sites)} `self.alert(` call sites parsed"

    for title in ("Failure with nothing to revert",
                  "Guardian self-test failed",
                  "Service down, but no promotion to revert"):
        at_title = [s for s in sites if f'"{title}"' in s]
        assert at_title, f"the {title!r} alert call site is gone"
        assert all("needs_human=True" in s for s in at_title), (
            f"{title!r} still asks for a human in prose only, so it files nothing")

    # The prose fallback has to have something left to match. `NEEDS_HUMAN_BODY`
    # is hand-written, so a test built only on it would keep passing after every
    # sentence in guardian.py was reworded and the fallback route was matching
    # nothing in the tree. This is the check that says the route is still live,
    # and it runs on real call sites for the reason above.
    import notify

    assert any(notify.asks_for_a_human("", s) for s in sites), (
        "`asks_for_a_human` matches no alert text in guardian.py: the prose was "
        "reworded everywhere, so delete the fallback from `Notifier.alert` rather "
        "than leave a route nothing can take")


# ---------------------------------------------------------------------------
# #1536 — one runtime-data incident is one daily-note section, and it gets
# retracted on the surface it was written to.
#
# `memory/2026-09-25.md` holds 21 of its 22 `## ` sections as copies of ONE
# "Runtime data is being written into the code tree" incident — one per
# STRAY_CHECK_SECONDS tick it outlived — and zero lines saying it cleared. The
# repeat guard that was supposed to stop it cannot: ALERT_REPEAT_SECONDS is 900.0
# and the check interval is 3600.0, so every finding passed straight through.
# Each fixture below advances the clock by exactly one check interval, so the
# 900 s guard is exercised under the arithmetic that made it inert in production.
# ---------------------------------------------------------------------------

def _stray_incident(tmp_path, monkeypatch, strays):
    """A guardian whose stray check reads `strays`, writing to a throwaway vault.

    Everything between the finding and the note is the real thing: the check
    branch, `Guardian.alert`'s repeat guard, `Notifier.alert`'s fan-out, and
    `_vault_note`'s write. Only the detector is synthetic — `stray_in_tree`
    returns whatever the test says, because the live tree returns `[]` since
    `9e98d0df` and the incident has to be manufactured to be observed.
    """
    import guardian as gmod
    import notify
    from datetime import datetime as _dt

    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True)
    clock = {"t": 1_700_000_000.0}
    monkeypatch.setattr(gmod.time, "time", lambda: clock["t"])
    found = {"strays": list(strays)}
    monkeypatch.setattr(gmod.datawatch, "stray_in_tree",
                        lambda repo: list(found["strays"]))

    g = gmod.Guardian.__new__(gmod.Guardian)   # no supervisor, no probes, no ledger
    g.notifier = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                                 vault_root=str(vault),
                                 backend_url="http://127.0.0.1:1")
    g.data = types.SimpleNamespace(armed=True)
    g._alert_seen = {}
    g.last_alert = ""
    note = vault / "memory" / f"{_dt.now().strftime('%Y-%m-%d')}.md"
    clears: list[tuple[str, bool]] = []
    real_resolve = g.notifier.resolve            # the real method, wrapped not replaced

    def _resolve(title, note_text):
        ok = real_resolve(title, note_text)
        clears.append((title, ok))
        return ok

    g.notifier.resolve = _resolve

    def tick(new_strays=None):
        """One stray-check interval later, with the detector now seeing `new_strays`."""
        if new_strays is not None:
            found["strays"] = list(new_strays)
        clock["t"] += gmod.policy.STRAY_CHECK_SECONDS
        g._runtime_data_incident(clock["t"])

    return g, note, tick, clears


def _sections(note, title):
    """Bodies of the daily note's `## Self-mod guardian: {title}` sections."""
    import notify

    text = note.read_text(encoding="utf-8") if note.exists() else ""
    return [text[a:b] for (t, a, b) in notify.Notifier._daily_sections(text)
            if t == title]


def test_two_stray_checks_of_one_incident_leave_one_daily_note_section(tmp_path, monkeypatch):
    """#1536 clause 1: the day log's volume tracks incidents, not duration.

    Two checks one interval apart, the same two strays both times. The note gains
    exactly ONE heading, and the second interval is proven not to be a repeat the
    900 s guard swallowed: the clock moved 3600 s, `ALERT_REPEAT_SECONDS` is 900 s,
    so both firings really did reach the notifier (that inequality is the whole
    reason 21 copies landed on 2026-09-25).
    """
    import guardian as gmod

    assert gmod.policy.STRAY_CHECK_SECONDS > gmod.policy.ALERT_REPEAT_SECONDS, (
        "the fixture's premise: the check interval outruns the repeat guard")
    g, note, tick, clears = _stray_incident(tmp_path, monkeypatch, ["workers.db", "eval/baselines"])
    tick()
    tick()

    secs = _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)
    assert len(secs) == 1, f"{len(secs)} sections for one incident:\n{note.read_text()}"
    assert "workers.db" in secs[0] and "eval/baselines" in secs[0], secs[0]


def test_the_stray_alert_names_where_a_new_tooling_directory_is_classified(
        tmp_path, monkeypatch):
    """#1541: the widened check reports any top-level entry git does not track,
    so a new `.mypy_cache` alarms exactly like a new writer until a person says
    which it is — that residual is the price of an open set and cannot be paid
    with a longer list.

    What CAN be closed is the alert giving only one instruction. `stray_in_tree`
    is now tree-driven, so the body has to carry the other branch as well: the
    name of the exclusion constant and the file it lives in. Without them the
    cheapest way to stop an hourly false alarm is to delete a tooling directory
    or switch the check off."""
    import guardian as gmod

    g, note, tick, clears = _stray_incident(tmp_path, monkeypatch, ["runs"])
    tick()

    secs = _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)
    assert len(secs) == 1, f"{len(secs)} sections:\n{note.read_text()}"
    assert "runs" in secs[0], secs[0]
    assert "KNOWN_GOOD_TOPLEVEL" in secs[0], (
        "the alert names no place a tooling directory can be classified")
    assert "agent-services/guardian/datawatch.py" in secs[0], secs[0]


def test_a_shrinking_stray_set_rewrites_the_one_section_and_drops_the_gone_path(
        tmp_path, monkeypatch):
    """#1536 clause 2: the single section names what the LATEST check found.

    The 21 copies on 2026-09-25 were 21 different snapshots, so `~/lloyd/workers.db`
    appeared in the note long after the file was gone and a reader could not tell a
    live firing from a dead one. One check later `workers.db` has left the set: it
    must disappear from the body, `eval/baselines` must survive, and the heading
    count must still be one.
    """
    import guardian as gmod

    g, note, tick, clears = _stray_incident(tmp_path, monkeypatch, ["workers.db", "eval/baselines"])
    tick()
    tick(["eval/baselines"])

    secs = _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)
    assert len(secs) == 1, f"{len(secs)} sections:\n{note.read_text()}"
    assert "workers.db" not in secs[0], "a cleared path is still being ordered about"
    assert "eval/baselines" in secs[0], secs[0]


def test_the_set_emptying_writes_one_cleared_line_and_stays_silent_after(
        tmp_path, monkeypatch):
    """#1536 clause 3: the retraction lands on the surface the alarm used.

    Three empty checks after a live one: exactly one `cleared:` line, the open
    marker gone, and the heading still there above it — the point is not to erase
    the incident but to contradict its imperative instructions where they were
    written, since the 2026-09-25 note's only surviving record of a half-resolved
    condition is 21 copies of "Find the writer, move the data across".
    """
    import guardian as gmod
    import notify

    g, note, tick, clears = _stray_incident(tmp_path, monkeypatch, ["workers.db"])
    tick()
    tick([])
    tick([])
    tick([])

    text = note.read_text(encoding="utf-8")
    assert text.count(f"{notify.DAILY_CLEARED_PREFIX} ") == 1, text
    assert notify.DAILY_STILL_OPEN not in text, "the section still claims to be open"
    assert len(_sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)) == 1, text
    assert "no runtime stores inside the code tree" in text, text
    # The other half of "does not repeat it": all three all-clear checks reached
    # `resolve` for this title and each reported the note as cleared, so silence
    # comes from an idempotent no-op and not from a swallowed error — the heartbeat
    # reads that True, and a False every hour would read as a broken notifier.
    assert [t for t, _ in clears] == [gmod.RUNTIME_DATA_ALERT_TITLE] * 3, clears
    assert [ok for _, ok in clears] == [True, True, True], clears


def test_a_stray_finding_after_a_clear_opens_a_new_section(tmp_path, monkeypatch):
    """#1536 clause 4: coalescing closes an incident, it does not mute the alert.

    A fix that only ever wrote one section per title would hide the SECOND
    incident, which is worse than the repetition it removed. Clear the set, then
    let a stray reappear: two headings, the older one closed with its own
    `cleared:` line, the new one open — so the count of headings is the count of
    incidents.
    """
    import guardian as gmod
    import notify

    g, note, tick, clears = _stray_incident(tmp_path, monkeypatch, ["workers.db"])
    tick()
    tick([])
    tick(["sessions"])

    text = note.read_text(encoding="utf-8")
    secs = _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)
    assert len(secs) == 2, f"{len(secs)} sections for two incidents:\n{text}"
    assert text.count(f"{notify.DAILY_CLEARED_PREFIX} ") == 1, text
    assert secs[0].rstrip().startswith(secs[0].rstrip().splitlines()[0])
    assert f"\n{notify.DAILY_CLEARED_PREFIX} " in secs[0], secs[0]
    assert notify.DAILY_STILL_OPEN not in secs[0], "the closed incident reopened"
    assert secs[1].rstrip().endswith(notify.DAILY_STILL_OPEN), secs[1]
    assert "sessions" in secs[1] and "workers.db" not in secs[1], secs[1]


def test_coalescing_is_scoped_to_the_runtime_data_title(tmp_path, monkeypatch):
    """#1536 clause 5: another alert's repetition is untouched.

    Two firings of a DIFFERENT title one interval apart — the same cadence, the
    same notifier, `coalesce` left at its default — still append two sections.
    Without this, the fix could be silently generalised to every guardian alert,
    and a rollback or a broken-loop notice that recurs across a night would go
    unread while it is still true.
    """
    import guardian as gmod
    import notify

    g, note, tick, clears = _stray_incident(tmp_path, monkeypatch, [])
    for n in range(2):
        g.alert("error", "Supervisord keeps dying", f"attempt {n}")
        tick([])

    text = note.read_text(encoding="utf-8")
    assert len(_sections(note, "Supervisord keeps dying")) == 2, text
    assert notify.DAILY_STILL_OPEN not in text, "a non-coalescing alert grew a marker"
    assert _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE) == []
    # The keyword's own default, since `alert` passes the flag explicitly: a caller
    # that says nothing must get the old append, not a coalescing section.
    assert g.notifier._vault_note("Plain notice", "body") is True
    assert notify.DAILY_STILL_OPEN not in note.read_text(encoding="utf-8")[len(text):], (
        "`_vault_note`'s default coalesces, so every alert in the file silently did too")


# ── #1590: retracting an alarm across the day boundary ───────────────────────
#
# Every fixture above builds its note path from `_dt.now()`, so each one proves the
# same-day contract and none of them can see a day change. These four drive the real
# `Notifier` with the date it keys its daily notes on moved forward — the smallest seam
# that reproduces the defect: an alert raised at 23:05 and all-cleared at 00:05 lives in
# two files, and `resolve` read only the second one.

def _day_notifier(tmp_path, day):
    """A real `Notifier` over a throwaway vault, standing on `day`.

    `_today` is the only date the notifier consults, so overriding that one method moves
    every daily-note path it builds — writer and retractor together. Nothing else on the
    notifier is stubbed: `_vault_note` and `resolve` are the shipped methods, so what these
    tests assert is what the hourly all-clear does to the files.
    """
    import notify

    vault = tmp_path / "obsidian"
    (vault / "memory").mkdir(parents=True, exist_ok=True)
    n = notify.Notifier(ledger=tmp_path / "l.jsonl", state_dir=tmp_path,
                        vault_root=str(vault), backend_url="http://127.0.0.1:1")
    n._today = lambda: day          # per instance; the class is left alone
    # The seam has to reach the shipped path builder, not just the local clock: if
    # `_daily_note` stopped consulting `_today`, every test below would silently write to
    # the real today's file and its assertions would be about a file nothing wrote.
    assert n._daily_note().name == f"{day.isoformat()}.md", n._daily_note()
    return n


def _note_of(vault_root: Path, day) -> Path:
    return Path(vault_root) / "memory" / f"{day.strftime('%Y-%m-%d')}.md"


def test_resolve_seals_an_alarm_written_the_previous_day(tmp_path):
    """#1590 clause 1: an alarm written on day D is retracted on day D+1.

    The alert is coalesced on D, the all-clear runs on D+1. The assertion is on D's file,
    not today's: exactly one `cleared:` line, and no standing open marker — the marker is
    REPLACED, because a note that still ends in it still claims the incident is open, which
    is the one question a retraction exists to change. `resolve` returning True here is the
    claim it could not previously justify: before the fix it returned True having opened a
    file with no open section in it, and day D's section went on instructing a reader to
    delete a git-tracked path.
    """
    import notify
    from datetime import date, timedelta

    day = date(2026, 9, 25)
    n = _day_notifier(tmp_path, day)
    title = "Runtime data left in the tree"
    n._vault_note(title, "move `eval/baselines` out of the tree", coalesce=True)
    note_d = _note_of(n.vault_root, day)
    assert note_d.read_text().rstrip().endswith(notify.DAILY_STILL_OPEN), "fixture: open on D"

    n._today = lambda: day + timedelta(days=1)
    assert n.resolve(title, "nothing further to move") is True

    text = note_d.read_text(encoding="utf-8")
    assert text.count(f"\n{notify.DAILY_CLEARED_PREFIX} ") == 1, text
    assert "nothing further to move" in text, text
    assert notify.DAILY_STILL_OPEN not in text, (
        "day D still claims the incident is open: the marker was left standing")


def test_one_resolve_seals_an_open_section_on_each_side_of_midnight(tmp_path):
    """#1590 clause 2: an incident that spans midnight leaves no open section behind.

    The coalescing writer keys on the same date as the retractor, so after midnight it
    cannot find yesterday's section and appends a fresh one: one incident, two open
    sections, in two files. One all-clear must seal BOTH — the probe measured one per day
    surviving every retraction, each still carrying the imperative instructions. Asserted
    as "no section for that title inside the window ends in the marker", across every dated
    note in the window, because sealing only the newest is the bug.
    """
    import notify
    from datetime import date, timedelta

    day = date(2026, 9, 25)
    n = _day_notifier(tmp_path, day)
    title = "Runtime data left in the tree"
    n._vault_note(title, "move `eval/baselines` out of the tree", coalesce=True)
    n._today = lambda: day + timedelta(days=1)
    n._vault_note(title, "move `eval/baselines` out of the tree", coalesce=True)

    notes = [_note_of(n.vault_root, day), _note_of(n.vault_root, day + timedelta(days=1))]
    open_before = [p for p in notes
                   if n._daily_open_at(p.read_text(encoding="utf-8"), title) is not None]
    assert len(open_before) == 2, (
        "fixture premise: an incident that crossed midnight opened one section per day")

    assert n.resolve(title, "nothing further to move") is True

    for p in notes:
        text = p.read_text(encoding="utf-8")
        assert n._daily_open_at(text, title) is None, f"{p.name} still open:\n{text}"
        assert text.count(f"\n{notify.DAILY_CLEARED_PREFIX} ") == 1, f"{p.name}:\n{text}"


def test_resolve_reports_false_for_an_alarm_older_than_the_scan_window(tmp_path):
    """#1590 clause 3: reach is finite, and the return says so.

    Two halves, both required. An open section in a note older than `DAILY_SCAN_DAYS` is
    beyond any retraction's reach, so `resolve` must return False rather than report a
    question it did not answer closed — and it must NOT rewrite that note, because silently
    sealing an alarm nobody verified is the same unqualified success in a different costume.
    Then the other half, which the first half must not break: with nothing open anywhere in
    the window, every all-clear still returns True. That is the idempotent-silence contract
    the hourly caller depends on, and it is why the answer cannot simply be False.
    """
    import notify
    from datetime import date, timedelta

    assert notify.DAILY_SCAN_DAYS >= 2, (
        "the window has to be able to reach yesterday, or this test proves nothing")
    day = date(2026, 9, 25)
    n = _day_notifier(tmp_path, day)
    title = "Runtime data left in the tree"
    stale_day = day - timedelta(days=notify.DAILY_SCAN_DAYS)
    n._today = lambda: stale_day
    n._vault_note(title, "move `eval/baselines` out of the tree", coalesce=True)
    old_note = _note_of(n.vault_root, stale_day)
    stale_before = old_note.read_text(encoding="utf-8")
    assert stale_before.rstrip().endswith(notify.DAILY_STILL_OPEN), "fixture: open, out of reach"

    n._today = lambda: day
    assert n.resolve(title, "nothing further to move") is False, (
        "an alarm outside the scan window was reported as retracted")
    assert old_note.read_text(encoding="utf-8") == stale_before, (
        "a note beyond the window was rewritten instead of reported")

    # ...and with nothing open in the window at all, the same call is silent success.
    for p in (n.vault_root / "memory").iterdir():
        text = p.read_text(encoding="utf-8")
        (p.parent / p.name).write_text(text.replace(notify.DAILY_STILL_OPEN, ""),
                                       encoding="utf-8")
    assert n.resolve(title, "nothing further to move") is True


def test_the_windowed_scan_leaves_same_day_behaviour_untouched(tmp_path):
    """#1590 clause 4: one live check and three all-clears, on one day, as the tree does it.

    Two things the widened scan could break, both asserted. First the shape
    `test_the_set_emptying_writes_one_cleared_line_and_stays_silent_after` proves through the
    whole guardian, restated at the notifier so the scan cannot change it from underneath:
    exactly ONE `cleared:` line after three all-clears, True on every one, and the open
    marker gone. A `False` for "nothing open in the window" would fail this, and
    re-appending a second retraction each hour would too.

    Second, the reach the scan newly has: an open section for a DIFFERENT title on an
    earlier day inside the same window must survive untouched. Retracting one incident's
    alarm is not retracting every alarm in range — seal by file rather than by title and
    the same-day result above still looks perfect while a live incident is marked cleared.
    """
    import notify
    from datetime import date, timedelta

    day = date(2026, 9, 25)
    n = _day_notifier(tmp_path, day)
    title = "Runtime data left in the tree"
    other = "Vault write landed an invalid skill"
    yesterday_note = _note_of(n.vault_root, day - timedelta(days=1))
    n._today = lambda: day - timedelta(days=1)
    n._vault_note(other, "the installed skill still owns this signature", coalesce=True)
    other_before = yesterday_note.read_text(encoding="utf-8")
    assert other_before.rstrip().endswith(notify.DAILY_STILL_OPEN), "fixture: other is open"
    n._today = lambda: day
    n._vault_note(title, "move `eval/baselines` out of the tree", coalesce=True)

    clears = [n.resolve(title, "nothing further to move") for _ in range(3)]
    assert clears == [True, True, True], clears
    text = _note_of(n.vault_root, day).read_text(encoding="utf-8")
    assert text.count(f"\n{notify.DAILY_CLEARED_PREFIX} ") == 1, text
    assert notify.DAILY_STILL_OPEN not in text, text

    assert yesterday_note.read_text(encoding="utf-8") == other_before, (
        "the scan sealed another title's open incident in a note it reached")


# ── #1612: the backlog verdict comes from success + id, never a name ─────────
#
# The two seam tests above (`test_the_needs_human_route_posts_the_payload_the_board_reads`
# and `test_a_defaulted_name_from_the_board_reads_as_a_failed_filing`) still hold,
# but neither can tell a deciding guard from a constant True: the first passes the
# only shape that may be True, the second passes a shape with no id in it, so both
# were green while `_backlog_task` returned True for every 2xx. These pin the
# decision itself, one reply shape per assertion, over the same loopback seam.

def _board_verdict(tmp_path, reply: dict) -> bool:
    """`_backlog_task`'s own verdict on one board reply, across the loopback seam.

    Called directly rather than through `alert()` because the clauses name this
    function, and `alert()` only copies its result into `results["backlog"]` — the
    POST and the JSON reply, which are the boundary, are unchanged by that. The
    request it makes is asserted too: a stub that answered any request would let a
    verdict stand for a filing the guardian never sent.
    """
    with _stub_board(tmp_path, reply) as (server, seen):
        verdict = _board_notifier(server, tmp_path)._backlog_task(
            "Service down, but no promotion to revert", NEEDS_HUMAN_BODY, "", "")
    assert seen.get("path") == "/api/backlog/task-create", seen
    assert seen["body"]["name"].startswith("[guardian] "), seen
    return verdict


def test_a_board_reply_with_the_success_and_id_the_endpoint_sends_is_delivered(tmp_path):
    """Clause 1 of #1612. This is `app/routers/backlog.py:765` verbatim —
    `JSONResponse({"success": True, "id": task_id})`, `task_id = max_id + 1`, and
    the only 2xx the endpoint has; every error path raises HTTPException, which
    `urlopen` turns into an `HTTPError` the caller already returns False for.

    It has to stay True. The bug was a guard that always said yes, and a fix that
    always said no would leave this channel reporting a filing that happened,
    which is #775's silence wearing the opposite sign.
    """
    assert _board_verdict(tmp_path, {"success": True, "id": 999}) is True


def test_a_board_reply_saying_it_failed_is_not_delivered_though_it_was_2xx(tmp_path):
    """Clause 2 of #1612. The old branch read a `name` the endpoint never sends and
    ended `if name else True`, so `success` was never consulted and this 200 came
    back delivered. A channel that reports its own failure as success is how
    needs-a-human alerts stop existing without anyone noticing.
    """
    assert _board_verdict(tmp_path, {"success": False}) is False


@pytest.mark.parametrize("reply", [
    {},                                     # an empty body
    {"success": True},                      # success, but no row id
    {"success": True, "id": None},          # a null id: no task file was written
    {"success": True, "name": "New Task"},  # the drifted-payload reply, no id
    {"success": True, "id": 0},             # 0 is not a row
    {"success": True, "id": -3},            # neither is a negative one
    {"success": True, "id": "999"},         # the endpoint's id is an int
    {"success": True, "id": True},          # bool IS an int subclass, and True > 0
    {"id": 999},                            # a row id with no success flag
], ids=["empty-body", "success-no-id", "null-id", "defaulted-name-no-id",
        "zero-id", "negative-id", "string-id", "bool-id", "id-without-success"])
def test_no_board_reply_lacking_a_positive_integer_id_reads_as_delivered(tmp_path, reply):
    """Clause 3 of #1612: no path through `_backlog_task` returns True without a
    positive integer id.

    Eight of these came back True before the fix: the `name` that branch inspected
    is absent from every one of them, and an absent name took the `else True` arm,
    so each reported a filing that filed nothing. The ninth, the `defaulted-name`
    reply, is the one shape the old guard could fail, and it belongs here because
    it still fails — now for the reason the clause gives, which is that it carries
    no id. The last five are what "positive integer" holds the guard to:
    `backlog_task_create` answers with `max_id + 1`, so a zero, a negative, a
    string or a `bool` is not that value, and an id with no `success` is not that
    reply either.
    """
    assert _board_verdict(tmp_path, reply) is False


# ---------------------------------------------------------------------------
# #2057 — the stray alert may print only what this check measured.
#
# `~/.local/state/lloyd-guardian/ALERT.md` (642 B at triage, mtime 2026-10-02
# 03:48) asserted "Something still resolves a data path off the code" and then
# ordered two actions for a retained `workers.db`: delete the in-tree copy, or add
# the name to `KNOWN_GOOD_TOPLEVEL`. Neither sentence had a number behind it — the
# whole body was one constant string (`guardian.py:996-1014`), and `stray_in_tree`
# returns bare names — so the alert stated a cause it had not measured and offered
# the one remedy the standing class rule calls wrong for a retained store. Its cost
# was multiplied by the same text: `grep -c 'remove the in-tree copy'` per daily
# note gave 3 on 2026-10-02, 8 on 2026-09-30, 3 on 2026-09-29, zero on the others.
#
# Every node below judges the alert body as the bytes that reach the two surfaces a
# reader meets — the daily-note section and ALERT.md — over a scratch tree the test
# owns, because a measurement quoted from the live tree would be a second claim
# rather than the check's own number.
# ---------------------------------------------------------------------------

_STRAY_EPOCH = 1_700_000_000.0   # the clock `_stray_incident` starts on


def _measured_stray_incident(tmp_path, monkeypatch, strays, files):
    """`_stray_incident` with `policy.REPO` moved onto a scratch tree.

    `files` maps a name to `(size_bytes, mtime)` and is written before the check
    runs, so the numbers the alert prints are this test's own constants. A name in
    `strays` with no entry is a path `lstat` cannot answer: clause 1 says it is
    printed as unmeasurable, so it belongs in the set and not on the filesystem.
    """
    import os

    import guardian as gmod

    tree = tmp_path / "tree"
    tree.mkdir()
    for name, (size, mtime) in files.items():
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * size)
        os.utime(path, (mtime, mtime))
    monkeypatch.setattr(gmod.policy, "REPO", str(tree))
    g, note, tick, clears = _stray_incident(tmp_path, monkeypatch, strays)
    return g, note, tick, clears, tree


def _alert_md(g) -> str:
    """ALERT.md as the next process will read it.

    The acceptance check is stated against that file, and `Notifier._alert_file`
    writes it above the `external` gate, so a body that only reached the daily note
    would pass three nodes here and still leave the live artefact stale."""
    return (g.notifier.state_dir / "ALERT.md").read_text(encoding="utf-8")


def test_the_stray_alert_prints_the_size_and_mtime_it_measured(tmp_path, monkeypatch):
    """#2057 clause 1: every named path carries the check's own `lstat` numbers.

    Three shapes in one set. A measurable file, 5 bytes at a mtime this test wrote,
    so the printed figure is a constant and not a re-read of whatever the tree
    happens to hold. A name the filesystem has no entry for — the race the check can
    actually lose, a stray deleted between `stray_in_tree` and the stat — which has
    to be printed as unmeasurable and stay on the list, not vanish from it. And a
    dangling symlink: `stray_in_tree` reports on `lexists` so it is a stray, `lstat`
    answers for it when `stat` would say `ENOENT`, and its `st_size` is the length of
    its target path, so the row has to name it as a link or "87 bytes" reads as 87
    bytes of data — a number that needs a story is the thing this item is about.

    The stamp carries a UTC offset, because a naive local time here is read as UTC by
    the next reader (#1912). The rows are also read back from `_measure_strays`
    itself, which is where "mtime is None exactly when the render says unmeasurable"
    lives — the fact the writer claim two nodes down depends on.
    """
    from datetime import datetime

    import guardian as gmod

    mtime = 1_760_000_000
    g, note, tick, clears, tree = _measured_stray_incident(
        tmp_path, monkeypatch, ["workers.db", "gone.db", "dangling.db"],
        {"workers.db": (5, mtime)})
    (tree / "dangling.db").symlink_to(tree / "not-there")
    tick()

    stamp = datetime.fromtimestamp(mtime).astimezone().isoformat()
    secs = _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)
    assert len(secs) == 1, secs
    assert f"{tree}/workers.db — 5 bytes, mtime {stamp}" in secs[0], secs[0]
    printed_row = next(ln for ln in secs[0].splitlines() if "workers.db —" in ln)
    assert re.search(r"mtime \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?[+-]\d\d:\d\d$",
                     printed_row), (
        "the offset the alert prints is not the machine's, or is naive — a naive "
        f"local time here is read as UTC by the next reader (#1912): {printed_row}")
    assert f"{tree}/gone.db — unmeasurable" in secs[0], (
        "a path the stat cannot answer was dropped from the list instead of "
        f"printed as unmeasurable:\n{secs[0]}")
    assert f"{tree}/dangling.db — unmeasurable" not in secs[0], secs[0]
    assert f"symlink → {tree / 'not-there'}" in secs[0], (
        f"a symlink's size is its target path's length; the row does not say so:\n"
        f"{secs[0]}")
    assert f"workers.db — 5 bytes, mtime {stamp}" in _alert_md(g), (
        "the file the next process reads has no measured number in it")

    rows = gmod._measure_strays(str(tree), ["workers.db", "gone.db", "dangling.db"])
    assert [r[0] for r in rows] == ["workers.db", "gone.db", "dangling.db"], rows
    assert [r[2] is None for r in rows] == [False, True, False], rows
    assert rows[0][2] == mtime and rows[0][1].startswith("5 bytes"), rows
    assert rows[1][1].startswith("unmeasurable"), rows


def test_a_retained_stray_alert_offers_neither_deletion_nor_the_exclusion_list(
        tmp_path, monkeypatch):
    """#2057 clause 2: for a `RUNTIME_NAMES` member both offers are withheld.

    `workers.db` is a retained store (`datawatch.py:67`), so "remove the in-tree
    copy" orders someone to delete a store — and 0 bytes made it look deletable,
    which is exactly the response the standing class rule exists to stop — while
    the `KNOWN_GOOD_TOPLEVEL` offer asks a name-list to close a property the check
    holds over the open set of the tree's top level. In their place the body has to
    say what the name is: retained, already on the watch list, open set.
    """
    import guardian as gmod

    assert "workers.db" in gmod.datawatch.RUNTIME_NAMES, "the fixture's premise"
    g, note, tick, clears, tree = _measured_stray_incident(
        tmp_path, monkeypatch, ["workers.db"], {"workers.db": (0, 1_760_000_000)})
    tick()

    secs = _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)
    assert len(secs) == 1, secs
    for text in (secs[0], _alert_md(g)):
        assert "remove the in-tree copy" not in text, text
        assert "KNOWN_GOOD_TOPLEVEL" not in text, text
    assert "Retained (workers.db)" in secs[0], secs[0]
    assert "watch list" in secs[0] and "RUNTIME_NAMES" in secs[0], secs[0]
    assert "not offered for deletion" in secs[0], secs[0]
    assert "no name-list closes" in secs[0], secs[0]


def test_the_exclusion_list_offer_names_only_the_non_retained_paths(
        tmp_path, monkeypatch):
    """#2057 clause 3: a mixed set routes the tooling half without touching the store.

    The #1541 routing sentence has to survive — with no place to classify a new
    `.mypy_cache` the cheapest answer to an hourly alarm is to switch the check off
    — but it may name only the non-retained path. Paragraph-scoped, because the
    whole body legitimately mentions both names: the offer and the retained
    warning must never share a paragraph, which is what makes "you could delete or
    allowlist this one" unreadable for a store.
    """
    import guardian as gmod

    g, note, tick, clears, tree = _measured_stray_incident(
        tmp_path, monkeypatch, ["workers.db", "runs"],
        {"workers.db": (0, 1_760_000_000)})
    tick()

    secs = _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)
    assert len(secs) == 1, secs
    paras = secs[0].split("\n\n")
    offered = [p for p in paras if "KNOWN_GOOD_TOPLEVEL" in p]
    assert len(offered) == 1, f"the offer appears {len(offered)} times:\n{secs[0]}"
    assert "runs" in offered[0], offered[0]
    assert "workers.db" not in offered[0], (
        f"a retained store is offered for the exclusion list:\n{offered[0]}")
    assert "remove the in-tree copy" not in secs[0], secs[0]
    retained = [p for p in paras if p.startswith("Retained (workers.db)")]
    assert len(retained) == 1, secs[0]
    assert "KNOWN_GOOD_TOPLEVEL" not in retained[0], retained[0]
    assert f"{tree}/runs — unmeasurable" in secs[0], secs[0]


def test_the_writer_claim_appears_only_on_an_mtime_inside_the_observed_window(
        tmp_path, monkeypatch):
    """#2057 clause 4, the only reading that carries the cause.

    The stray is written 60 s before this check, and the previous check ran one
    interval earlier, so the mtime falls inside a window the guardian itself
    observed: this is the one case in which "something still resolves a data path
    off the code" is a measurement rather than a memory. The window's end is the
    clock reading this node derives from `_stray_incident`'s own arithmetic, and
    asserting it appears in the body is also what proves the alert printed the
    `now` the check was handed rather than a timestamp of its own.
    """
    import guardian as gmod

    now = _STRAY_EPOCH + gmod.policy.STRAY_CHECK_SECONDS
    g, note, tick, clears, tree = _measured_stray_incident(
        tmp_path, monkeypatch, ["workers.db"],
        {"workers.db": (0, now - 60.0)})
    g._strays_prev_check_at = now - gmod.policy.STRAY_CHECK_SECONDS
    tick()

    secs = _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)
    assert len(secs) == 1, secs
    assert gmod._stamp(now) in secs[0], (
        f"the alert did not print the window it was given "
        f"(expected end {gmod._stamp(now)}):\n{secs[0]}")
    assert "still resolves a data path off the code" in secs[0], secs[0]
    assert "Written inside the window this check observed" in secs[0], secs[0]


def test_an_older_mtime_prints_the_window_and_says_the_writer_is_not_identified(
        tmp_path, monkeypatch):
    """#2057 clause 4, the case the live incident is actually in.

    `~/lloyd/workers.db` was 0 bytes with mtime 2026-10-02 01:01:36 against an alert
    that fired at 03:48 — outside any window an hourly check could have watched it
    happen. So the body prints both window bounds and the size and mtime that
    exclude them, and states plainly that this check did not identify a writer: the
    triage recorded that `paths.WORKERS_DB` refutes "move the data across", while a
    post-cutover mtime still leaves the writer hypothesis open, and neither
    conclusion is the alert's to draw from a number it did not watch change.
    """
    import guardian as gmod

    now = _STRAY_EPOCH + gmod.policy.STRAY_CHECK_SECONDS
    prev = now - gmod.policy.STRAY_CHECK_SECONDS
    g, note, tick, clears, tree = _measured_stray_incident(
        tmp_path, monkeypatch, ["workers.db"],
        {"workers.db": (0, now - 7200.0)})
    g._strays_prev_check_at = prev
    tick()

    secs = _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)
    assert len(secs) == 1, secs
    assert "still resolves a data path off the code" not in secs[0], secs[0]
    assert "does not identify a writer" in secs[0], secs[0]
    assert gmod._stamp(prev) in secs[0], (
        "the window's start is not printed, so the reader cannot see the exclusion")
    assert gmod._stamp(now) in secs[0], secs[0]
    assert "0 bytes" in secs[0], secs[0]


def test_the_first_stray_check_has_no_window_and_claims_no_writer(
        tmp_path, monkeypatch):
    """#2057 clause 4, the boundary case: a first check has no previous one.

    With no earlier reading there is no interval to fall inside, so the alert says
    it is the first check and prints no cause. The alternative — borrowing the
    process start or the file's own age as a window — would be the same invented
    mechanism this item exists to remove, and would fire on the very first hourly
    check of every guardian restart.
    """
    import guardian as gmod

    g, note, tick, clears, tree = _measured_stray_incident(
        tmp_path, monkeypatch, ["workers.db"],
        {"workers.db": (0, _STRAY_EPOCH + 3_000_000_000.0)})
    # The state a real `Guardian.__init__` leaves behind — `None`, pinned on a real
    # construction by tests/test_data_home.py::test_a_fresh_guardian_has_no_previous_stray_check.
    # Set here rather than left absent so this node exercises production's value and
    # not the fact that `__new__` skipped the initialiser.
    g._strays_prev_check_at = None
    tick()

    secs = _sections(note, gmod.RUNTIME_DATA_ALERT_TITLE)
    assert len(secs) == 1, secs
    assert "still resolves a data path off the code" not in secs[0], secs[0]
    assert "first stray check" in secs[0], secs[0]
    # The premise above (`not hasattr`) is why this node is the no-window arm: after
    # one call the reading exists, and the next check would have a window.
    assert g._strays_prev_check_at == _STRAY_EPOCH + gmod.policy.STRAY_CHECK_SECONDS, (
        "the check did not record its own reading, so the next one has no window")


def test_the_committed_alert_witness_carries_the_defect_and_the_new_body_carries_a_number(
        tmp_path, monkeypatch):
    """#2057 clause 5: the alert's own bytes are the witness, before and after.

    `tests/fixtures/guardian/2057-stray-alert.sample` is
    `~/.local/state/lloyd-guardian/ALERT.md` copied byte for byte — 642 bytes, which
    is the figure the item and its triage both quote, so this node re-derives a quoted
    number from committed bytes rather than from a pointer into a live store. (The
    vault path the clause names, `backlog/data/workers.db`, is a separate tree no
    commit in this repo can contain, and the bytes already at it are #1949's
    schema-only witness, 126,976 B of mtime 2026-09-30 23:27, whose
    `count(*) from sqlite_master` is 12; overwriting that with the 9.2 MB live DB to
    re-derive a 19 this item never quotes would retire another item's evidence, so the
    vault copy is recorded as owed instead.)

    The stray that artefact names was measured by the check that raised it, and that
    is the input rebuilt here:

        $ stat -c '%n %s bytes mtime=%y' /home/alansrobotlab/lloyd/workers.db
        /home/alansrobotlab/lloyd/workers.db 0 bytes mtime=2026-10-02 01:01:36.605042322 -0700

    So the file is created 0 bytes with that exact mtime, and the body rendered for it
    is graded against the witness beside it: the old bytes carry both wrong offers and
    no measurement at all, the new ones print the size and mtime, withhold both
    offers, and — 01:01 against a 03:48 alert, an hour's interval apart — say the
    writer is not identified. The incident's own numbers, not synthetic ones.
    """
    import os
    from datetime import datetime

    import guardian as gmod

    witness = (Path(__file__).resolve().parent
               / "fixtures" / "guardian" / "2057-stray-alert.sample").read_bytes()
    assert len(witness) == 642, (
        f"the committed witness is no longer the 642 B ALERT.md the item quotes: "
        f"{len(witness)} B")
    text = witness.decode("utf-8")
    assert "/home/alansrobotlab/lloyd/workers.db" in text, text
    assert "remove the in-tree copy" in text, "the witness is not the pre-fix artefact"
    assert "KNOWN_GOOD_TOPLEVEL" in text, "the witness is not the pre-fix artefact"
    assert "still resolves a data path off the code instead of" in text, text
    assert "bytes, mtime" not in text, (
        "the witness already prints a measurement, so it cannot pin the defect")

    when = datetime.fromisoformat("2026-10-02 01:01:36.605042-07:00").timestamp()
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "workers.db").write_bytes(b"")
    os.utime(tree / "workers.db", (when, when))
    # The incident's own two stamps, both readable in the witness above: the stub's
    # mtime is 01:01:36 and the body's `written:` line is 04:48:34, 13,618 s apart, so
    # `now` renders 04:48:34 and the previous reading one `STRAY_CHECK_SECONDS`
    # earlier renders 03:48:34 — the hour the triage recorded the file at. The
    # 01:01:36 mtime is 2h47m below that window's start, which is why this body is the
    # writer-not-identified one.
    now = when + 13618.0
    body = gmod._stray_alert_body(str(tree), ["workers.db"], now,
                                  now - gmod.policy.STRAY_CHECK_SECONDS)
    assert "03:48:34" in body and "04:48:34" in body, (
        f"the window printed is not the incident's two stamps: {body}")
    assert "workers.db — 0 bytes, mtime 2026-10-02T01:01:36" in body, body
    for phrase in ("remove the in-tree copy", "KNOWN_GOOD_TOPLEVEL",
                   "still resolves a data path off the code"):
        assert phrase not in body, f"{phrase} survived into the new body:\n{body}"
    assert "does not identify a writer" in body, body
