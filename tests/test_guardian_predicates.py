"""Guardian failure predicates — the arithmetic of when to roll back.

These are pure functions of a snapshot dict precisely so every branch is
table-testable without a running system. The drill
(`scripts/guardian_drill.py`) proves the guardian *acts*; this file proves it
*decides* correctly.

Process-info dicts here are shaped like real `getAllProcessInfo` output,
including the `start`/`now`/`spawnerr`/`group` fields the predicate reads.
"""

from __future__ import annotations

import contextlib
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
    `{"success": true, "id": N}`. It does **not** echo the request's `name`:
    `_backlog_task` inspects `created["name"]` to catch a drifted payload, and the
    real endpoint never sends one, so an echo would have the stub decide the
    outcome the test exists to observe."""
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

    `res["backlog"]` is asserted last, for what it can prove: the reply carries
    the real endpoint's no-`name` shape, so the value comes from
    `_backlog_task`'s no-name branch.
    `test_a_defaulted_name_from_the_board_reads_as_a_failed_filing` is what shows
    that branch is a decision and not a constant True."""
    with _stub_board(tmp_path, {"success": True, "id": 999}) as (server, seen):
        res = _board_notifier(server, tmp_path).alert(
            "error", "Service down, but no promotion to revert", NEEDS_HUMAN_BODY)

    assert seen.get("path") == "/api/backlog/task-create", seen
    assert seen["body"]["name"] == ("[guardian] Service down, but no promotion "
                                    "to revert"), "the `name` key drifted"
    assert seen["body"]["status"] == "up_next", (
        "`backlog_task_create` 400s a status outside _VALID_STATUSES, so any "
        "other value means the task is never created")
    assert seen["body"]["priority"] == "high"
    assert "needs a human" in seen["body"]["description"], (
        "the filed task lost the sentence that asked for the human")
    assert res["backlog"] is True, f"a 2xx filing read as a failure: {res}"


def test_a_defaulted_name_from_the_board_reads_as_a_failed_filing(tmp_path):
    """The other side of that same seam. A drifted payload is not an error: the
    endpoint answers 2xx and files a task called "New Task", which is how a
    routing fix that posts the wrong keys would quietly report delivered forever.
    `_backlog_task` reads the reply's `name` for exactly that, so a reply naming a
    task that is not the guardian's must not come back True — without this, the
    `is True` above could be the guard defaulting rather than deciding."""
    with _stub_board(tmp_path, {"success": True, "name": "New Task"}) as (server, seen):
        res = _board_notifier(server, tmp_path).alert(
            "error", "Service down, but no promotion to revert", NEEDS_HUMAN_BODY)

    assert seen["body"]["name"].startswith("[guardian] "), seen
    assert res["backlog"] is False, (
        "a board that answered with someone else's task was reported as a "
        "delivered guardian filing")


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
