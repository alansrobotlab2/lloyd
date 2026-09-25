"""Which context policy rewrote this turn, in the store instead of the log.

Backlog #1078. The claim that opened it: for a turn whose history was rewritten,
"which of Lloyd's three mechanisms did it, and how much did it free" had exactly
one source — `logs/server.err*`, which rotates every ~10 MB. Two consequences
made it an instrument problem rather than a cosmetic one. The one mechanism that
actually fires (the relief ladder) wrote 7,253 summary lines in one retained
window and **0** of them named a session, so the firings could not be attributed
to anything even by parsing the logs; and the turn-start stack logs only when it
rewrites, so `compaction.mode: summarize` being dead configuration was
indistinguishable from it running on every turn and declining.

What is pinned here:

  * `usage` has a `compaction` column, added through the module's existing
    migration path, so an existing database keeps every row and gains NULLs;
  * `record_usage` accepts the record and it comes back selectable in SQL, as
    the same shape it went in — a turn's row is the record a long-session eval
    (#600) reads, and it must survive the round trip to be usable as a gate;
  * NULL means UNMEASURED and is never the value of "fired and removed
    nothing" — that case is a row whose `freed_tokens` is 0, and a reader who
    cannot tell them apart reads an instrument gap as a decline;
  * the harness loop finds the turn its caller is booking through the
    session-keyed registry, and an anonymous turn is never claimed by another
    anonymous turn;
  * a relief pass that ran no rung is not booked, so a latched pass cannot
    inflate a per-session firing count.

Store-level only: `tests/test_compaction.py`, `tests/test_run_recorder.py` and
`app/harness/tests/test_context_meter.py` pin each mechanism's own emit site.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

import usage_store
from app.compaction_record import (
    TurnCompaction,
    current,
    note_relief,
    start_turn,
    turn_start_record,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An isolated usage database, opened through the module's real path.

    `_conn()` reopens when `DB_PATH` moves (usage_store.py:22-27), which is the
    seam every test in this area uses; there is no cache to clear by hand.
    """
    db = tmp_path / "usage.db"
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    return db


def _columns(db) -> list[str]:
    """`PRAGMA table_info(usage)`, the item's own check.

    Opened through `usage_store._conn()` first because that is where the schema
    is created and migrated: a raw connect to a path the module has never
    written would create an empty file and report no columns at all, which reads
    as "the migration did not run" when it reads as "nothing ever used this
    database".
    """
    usage_store._conn().commit()
    return [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(usage)")]


def _rows(db) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute("SELECT * FROM usage")]


def _one(db) -> dict:
    rows = _rows(db)
    assert len(rows) == 1, f"expected exactly one usage row, got {len(rows)}"
    return rows[0]


def _read(db) -> dict | None:
    """The stored compaction record, parsed, or None where the column is NULL."""
    raw = _one(db)["compaction"]
    return None if raw is None else json.loads(raw)


def _seed_14_column_db(db) -> None:
    """Recreate the schema as it stood before this change, with a row in it.

    The live database is the dashboard's only history, so what matters is not
    that the column appears but that migration happens underneath existing rows.
    """
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE usage (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               ts TEXT NOT NULL DEFAULT (datetime('now')),
               session_id TEXT, model TEXT,
               input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
               cache_create INTEGER DEFAULT 0, cache_read INTEGER DEFAULT 0,
               cost_usd REAL DEFAULT 0,
               duration_ms INTEGER, duration_api_ms INTEGER, num_turns INTEGER,
               reprefill_tokens INTEGER, prefix_misses INTEGER,
               skills TEXT
           )"""
    )
    conn.execute(
        "INSERT INTO usage (session_id, model, input_tokens, skills) "
        "VALUES ('20260923_070000_autonomy_dddd', 'primary', 555, "
        "'[{\"name\":\"old\",\"route\":\"hint\"}]')"
    )
    conn.commit()
    conn.close()


# ── clause 1: the column ────────────────────────────────────────────────

def test_the_usage_table_gains_a_compaction_column(store):
    """The acceptance check the item was confirmed on was `PRAGMA
    table_info(usage)` returning 14 columns and none of them compaction or
    relief. That is the query a reader re-runs, so it is the assertion.
    """
    assert "compaction" in _columns(store)


def test_a_bare_record_usage_call_leaves_the_column_null(store):
    """Every existing writer keeps compiling and its rows read as unmeasured.

    The three chat writers and `run_recorder` gained a keyword; nothing else in
    the tree calls `record_usage`, and a writer that does not know about context
    policy must produce NULL rather than a record claiming no mechanism fired.
    """
    usage_store.record_usage(
        session_id="20260924_070000_autocode_aaaa", model="primary",
        input_tokens=100, output_tokens=10,
    )
    assert _read(store) is None


def test_the_live_usage_db_keeps_every_row_and_gains_nulls(store):
    """The migration path is `ALTER TABLE`, not a recreated table.

    `usage.db` is the dashboard's only history: a recreated table would have
    thrown away every turn ever recorded, which is the same mistake already
    caught once in this store when the `skills` column landed (#783).
    """
    _seed_14_column_db(store)

    usage_store.record_usage(
        session_id="20260924_070000_autocode_bbbb", model="primary",
        input_tokens=200, output_tokens=20,
    )

    conn = sqlite3.connect(store)
    conn.row_factory = sqlite3.Row
    rows = {r["session_id"]: dict(r) for r in
            conn.execute("SELECT * FROM usage ORDER BY id")}
    assert len(rows) == 2, f"migration lost rows: {sorted(rows)}"
    old = rows["20260923_070000_autonomy_dddd"]
    assert old["skills"] == '[{"name":"old","route":"hint"}]'
    assert old["compaction"] is None, "the pre-migration row must read unmeasured"


# ── clause 1: the value, and its shape ──────────────────────────────────

def _relief_turn(freed: int = 41_000, rungs=("tool_results", "arguments")):
    """A turn that relieved context once, the way `run_recorder` books it."""
    turn = TurnCompaction(session_id="s", turn_id="t")
    turn.note_relief({
        "reason": "intra_turn", "rungs": list(rungs), "freed_tokens": freed,
        "used_before": 250_000, "used_after": 250_000 - freed,
        "target": 126_000, "passes": 1, "rearm": 168_115,
    })
    return turn


def test_a_relief_record_round_trips_through_sql(store):
    """Selectable in SQL, and identical on the way out.

    #600's acceptance gate is "record per run whether each mechanism under test
    actually fired", which means an eval reads these rows back. A record that
    lost its rungs to serialisation would satisfy the column and fail the gate.
    """
    usage_store.record_usage(
        session_id="20260924_070000_deepresearch_cccc", model="primary",
        input_tokens=250_000, output_tokens=900,
        compaction=_relief_turn(),
    )
    record = _read(store)
    assert record is not None
    assert record["mechanisms"] == ["relief:intra_turn"]
    assert record["relief_passes"] == 1
    assert record["relief_tokens_freed"] == 41_000
    assert record["relief"][0]["rungs"] == ["tool_results", "arguments"]
    assert record["relief"][0]["reason"] == "intra_turn"


def test_the_column_is_filterable_in_sql(store):
    """A per-session count of firings is a query, not a log parse.

    This is the shape of every question the item lists as unanswerable: "how
    many turns relieved context", and "in which sessions". NULL rows must drop
    out of the count rather than counting as declines.
    """
    usage_store.record_usage(
        session_id="20260924_070000_deepresearch_cccc", model="primary",
        input_tokens=250_000, compaction=_relief_turn())
    usage_store.record_usage(
        session_id="20260924_070000_autotriage_eeee", model="primary",
        input_tokens=1_000)

    conn = sqlite3.connect(store)
    fired = conn.execute(
        "SELECT count(*) FROM usage WHERE compaction IS NOT NULL").fetchone()[0]
    sessions = conn.execute(
        "SELECT session_id FROM usage WHERE compaction LIKE '%intra_turn%' "
        "ORDER BY session_id").fetchall()
    assert fired == 1, "an unmeasured row must not count as a turn that fired"
    assert sessions == [("20260924_070000_deepresearch_cccc",)]


def test_a_pass_that_ran_rungs_and_freed_nothing_is_not_null(store):
    """NULL means unmeasured; a zero here means "fired, could not relieve".

    The failure mode this separates: a turn that trips the ladder, runs two
    rungs and still sits above target is the finding an operator wants to count,
    and if it were stored as NULL it would be indistinguishable from a turn whose
    session never reached the instrument at all.
    """
    usage_store.record_usage(
        session_id="20260924_070000_boardsteward_ffff", model="primary",
        input_tokens=255_000, compaction=_relief_turn(freed=0))

    record = _read(store)
    assert record is not None, "a firing that freed nothing must still be stored"
    assert record["relief"][0]["freed_tokens"] == 0
    assert record["relief_tokens_freed"] == 0
    assert record["mechanisms"] == ["relief:intra_turn"]


def test_a_turn_start_that_declined_is_stored_and_is_not_an_absence(store):
    """The dead-configuration half of the item, in one assertion.

    Under the threshold the turn-start stack runs, reads the history, and
    rewrites nothing. Before this change that turn left no trace anywhere, which
    is exactly what `summarized=True` appearing 0 times in 9 days of logs could
    not resolve. Now the row says the layer was reached and why it stopped.
    """
    turn = TurnCompaction(session_id="s", turn_id="t")
    turn.note_turn_start({
        "tokens_before": 40_000, "tokens_after": 40_000,
        "microcompacted": 0, "summarized": False, "truncated": False,
        "context_window": 262_144, "threshold": 210_144,
        "summarize_attempted": False, "summarize_outcome": "under_threshold",
    })
    usage_store.record_usage(
        session_id="20260924_070000_mission-control_1111", model="primary",
        input_tokens=40_000, compaction=turn)

    record = _read(store)
    assert record["mechanisms"] == [], "nothing acted, and that is a fact, not a gap"
    assert record["turn_start"]["summarize_outcome"] == "under_threshold"
    assert record["turn_start"]["summarize_attempted"] is False


def test_an_unmeasured_turn_and_a_declining_turn_are_different_rows(store):
    """Counting them together is the misread this clause exists to prevent."""
    declined = TurnCompaction(session_id="s", turn_id="t")
    declined.note_turn_start({"tokens_before": 10, "tokens_after": 10,
                              "summarize_attempted": False,
                              "summarize_outcome": "under_threshold"})
    usage_store.record_usage(session_id="a", model="primary", input_tokens=1,
                             compaction=declined)
    usage_store.record_usage(session_id="b", model="primary", input_tokens=1)

    rows = _rows(store)
    assert [r["compaction"] is None for r in rows] == [False, True]
    assert json.loads(rows[0]["compaction"])["turn_start"] is not None


# ── clause 1: the API surface, and its failure modes ────────────────────

def test_a_plain_mapping_is_accepted_and_normalised(store):
    """A caller may hand a dict, or the `TurnCompaction` itself.

    `record_usage` normalises through `to_record()` when the object offers it so
    the two writers cannot disagree about the stored shape by building it two
    ways. A value JSON cannot encode is stringified by `default=str` rather than
    losing the whole row.
    """
    usage_store.record_usage(session_id="s1", model="primary", input_tokens=1,
                             compaction={"mechanisms": ["microcompact"],
                                         "tokens_freed": 12})
    usage_store.record_usage(session_id="s2", model="primary", input_tokens=1,
                             compaction=_relief_turn(freed=5))
    usage_store.record_usage(session_id="s3", model="primary", input_tokens=1,
                             compaction={"mechanisms": ["microcompact"],
                                         "when": object()})

    rows = {r["session_id"]: r["compaction"] for r in _rows(store)}
    assert json.loads(rows["s1"])["tokens_freed"] == 12
    assert json.loads(rows["s2"])["relief_tokens_freed"] == 5
    parsed = json.loads(rows["s3"])
    assert parsed["mechanisms"] == ["microcompact"], (
        "an unencodable value degrades inside the record — `default=str` "
        "stringifies it — and the rest of the record, and the token row with "
        "it, survive. Pinned on the parsed contents rather than on the repr "
        "text, because the property is containment, not CPython's spelling"
    )
    assert "when" in parsed


@pytest.mark.parametrize("junk", [
    "", "[]", "{}", "not json at all", "[1, 2]", 42, [], {}, object(),
])
def test_junk_never_costs_the_turn_its_usage_row(store, junk):
    """`record_usage`'s insert is one statement: a raise here drops the token
    row too, and several callers wrap it in a `try` that reports "failed to
    record usage" and moves on. So the compaction dimension degrades to NULL and
    the tokens still land — same containment as `_skills_column` for #783.
    """
    usage_store.record_usage(session_id="s", model="primary",
                             input_tokens=77, output_tokens=3,
                             compaction=junk)
    row = _one(store)
    assert row["compaction"] is None
    assert row["input_tokens"] == 77, "accounting failed but the tokens must not"


def test_an_already_serialised_record_lands_as_json(store):
    """A writer holding the JSON string still stores an object, not a string
    of a string no later reader would bother to parse."""
    usage_store.record_usage(
        session_id="s", model="primary", input_tokens=1,
        compaction='{"mechanisms": ["relief:intra_turn"], "relief_passes": 1}')
    assert _read(store) == {"mechanisms": ["relief:intra_turn"],
                            "relief_passes": 1}


# ── the registry the harness loop reads ─────────────────────────────────

def test_the_loop_finds_the_open_turn_by_session_id():
    """The hop from inside the loop to the row written outside it.

    `_relieve_context` is called from four places in `run_query` and the
    overflow rung is also called by `app/routers/messages.py`; none of them can
    reach a local in the writer's frame. `options.session_id` is the one value
    all four have and the one key the writer registered under.
    """
    sid = "20260924_070000_worker_2222"
    turn = start_turn(sid, "run1")
    assert current(sid) is turn
    assert note_relief(sid, {"reason": "intra_turn", "rungs": ["tool_results"],
                             "freed_tokens": 900}) is True
    record = turn.to_record()
    assert record["relief_passes"] == 1
    assert record["relief_tokens_freed"] == 900


def test_an_anonymous_turn_is_not_claimed_by_another_anonymous_turn():
    """An eval turn runs with `session_id=None` (#818). `current("")` returns
    None rather than "the first turn that happened to have no id", so one
    unaddressed run can never book its relief onto another run's row — the
    mis-attribution would be worse than the missing record.
    """
    unaddressed = start_turn("", "t")
    assert current("") is None
    assert note_relief("", {"reason": "intra_turn", "rungs": ["tool_results"],
                            "freed_tokens": 5}) is False
    assert unaddressed.to_record() is None


def test_a_second_turn_for_a_session_replaces_the_first():
    """Turns are serialised per session, and replacement is what keeps a turn
    that died before writing its row from merging its relief into the next
    turn's record.
    """
    sid = "20260924_070000_worker_3333"
    first = start_turn(sid, "run1")
    first.note_relief({"reason": "intra_turn", "rungs": ["arguments"],
                       "freed_tokens": 10})
    second = start_turn(sid, "run2")
    assert current(sid) is second
    assert second.to_record() is None
    assert first.to_record()["relief_passes"] == 1, (
        "the dead turn's own object is unchanged; only the registry moved on"
    )


def test_a_latched_pass_that_ran_no_rung_is_not_booked():
    """`rungs == []` is a no-op — latched, or the switch is off — and booking
    it would make a per-session firing count read as pressure that never
    happened. The event emitter applies the same rule, so the two stores cannot
    disagree about what counts as a firing.
    """
    turn = TurnCompaction(session_id="s", turn_id="t")
    assert turn.note_relief({"reason": "intra_turn", "rungs": [],
                             "latched": True, "freed_tokens": 0}) is False
    assert turn.note_relief(None) is False
    assert turn.to_record() is None


def test_two_relief_passes_are_two_entries_one_mechanism_name():
    """A turn may cross the trigger, relieve, grow, and relieve again. Each
    pass keeps its own numbers; the mechanism name is per reason, so "how often
    does the ladder fire" counts rows containing the name and "how many passes"
    reads `relief_passes`.
    """
    turn = TurnCompaction(session_id="s", turn_id="t")
    turn.note_relief({"reason": "intra_turn", "rungs": ["tool_results"],
                      "freed_tokens": 100})
    turn.note_relief({"reason": "intra_turn", "rungs": ["arguments"],
                      "freed_tokens": 50})
    turn.note_relief({"reason": "overflow", "rungs": ["tool_results"],
                      "freed_tokens": 25})
    record = turn.to_record()
    assert record["relief_passes"] == 3
    assert record["relief_tokens_freed"] == 175
    assert record["mechanisms"] == ["relief:intra_turn", "relief:overflow"]


def test_a_turn_start_and_relief_in_one_turn_keep_their_own_numbers():
    """Both mechanisms can act on one turn. Neither store's number is the
    other's: the turn-start half is tokens before/after, the relief half is
    freed per pass, and a reader adding them would double-count what relief cut
    from a history the turn-start stack had already trimmed.
    """
    turn = TurnCompaction(session_id="s", turn_id="t")
    turn.note_turn_start({"tokens_before": 220_000, "tokens_after": 200_000,
                          "microcompacted": 4, "summarized": False,
                          "truncated": False, "summarize_attempted": False,
                          "summarize_outcome": "mode_truncate",
                          "threshold": 210_144, "context_window": 262_144})
    turn.note_relief({"reason": "intra_turn", "rungs": ["reasoning"],
                      "freed_tokens": 4_000})
    usage_row = turn.to_record()
    assert usage_row["mechanisms"] == ["microcompact", "relief:intra_turn"]
    assert usage_row["turn_start"]["tokens_freed"] == 20_000
    assert usage_row["relief_tokens_freed"] == 4_000


def test_turn_start_record_names_every_layer_that_acted():
    """All three layers can fire on one turn — microcompact clears results,
    summarize still finds the history over threshold and summarizes, truncate
    then enforces the target. The record names them together rather than
    reporting the last one, which is what the caller's own log line does.
    """
    record = turn_start_record({
        "tokens_before": 300_000, "tokens_after": 126_000,
        "microcompacted": 6, "summarized": True, "truncated": True,
        "restored_files": 2, "summarize_attempted": True,
        "summarize_outcome": "summarized",
    })
    assert record["mechanisms"] == ["microcompact", "summarize", "truncate"]
    assert record["tokens_freed"] == 174_000
    assert record["restored_files"] == 2


def test_turn_start_record_reports_reuse_and_folds():
    """D2: a turn that applied a stored summary says so, with the folds it
    added and the rows the summary stands in for — the second turn over the
    wall reading `summary_reused: true` is how a deploy is verified. A result
    from before D2 (or with `persist_summary` off) reads false/0, never absent.
    """
    record = turn_start_record({
        "tokens_before": 300_000, "tokens_after": 90_000,
        "summarized": True, "summarize_attempted": True,
        "summarize_outcome": "summarized",
        "summary_reused": True, "summary_folds": 2, "summary_covered_rows": 412,
    })
    assert record["summary_reused"] is True
    assert record["summary_folds"] == 2
    assert record["summary_covered_rows"] == 412
    assert record["mechanisms"] == ["summarize"]

    legacy = turn_start_record({"tokens_before": 10, "tokens_after": 10})
    assert (legacy["summary_reused"], legacy["summary_folds"],
            legacy["summary_covered_rows"]) == (False, 0, 0)


def test_turn_start_record_of_a_missing_result_is_untouched():
    """`None` is what the callers hand over when the stack raised and fell back
    to `load_session_messages` (#1066). Inventing a decline would turn the one
    case that must read as unmeasured into a measurement.
    """
    assert turn_start_record(None) is None
    assert TurnCompaction().to_record() is None


# ===========================================================================
# The chat path: a streamed turn and a loopback post both land the record
# ===========================================================================
#
# Everything above tests a mechanism and a store. This section drives the two
# production functions that own the `usage` row for every interactive turn,
# because the seam #1078 is about is the one between them: a relief pass that
# happens deep inside `app/harness/loop.py` has to surface on a row that
# `app/routers/messages.py` writes, and the two never call each other. A unit
# test either side of that gap proves nothing about the gap itself — the record
# could be complete and still never reach the database.
#
# The driver runs against a private copy of the router, shared with
# `tests/test_usage_skill_breakdown.py` through `tests/_messages_copy.py` — one
# copy of the reload and its `__globals__` subtlety rather than two, because the
# last round's ordering defect lived inside exactly that subtlety and had to be
# found twice. The copy is needed because `tests/test_session_queue.py` rebinds
# `messages._run_turn` to its own stubs, which in a full-suite run is what a test
# importing the live attribute would get.
#
# Each test that drives the router does so through `_wired`, which patches
# attributes ON THE COPY (`patch.object`, via `patch_on_copy`) — never a dotted
# string, which would resolve to the live module and leave the copy's turn
# running every real collaborator behind the fakes.

from contextlib import ExitStack, contextmanager  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from tests._messages_copy import load_messages_copy, patch_on_copy  # noqa: E402


def _relief_report_like_production(options) -> dict:
    """Fire one real relief pass on the options the real turn built.

    Not a hand-built report: `_relieve_context` is the emitter, so the pass under
    test went through the code that produces relief in production, including its
    lookup of the turn's record by `options.session_id`.
    """
    from app.harness.context_meter import ContextMeter
    from app.harness.loop import _relieve_context

    msgs = [
        {"role": "assistant", "content": f"m{i}", "reasoning": "R" * 40_000,
         "reasoning_content": "R" * 40_000}
        for i in range(12)
    ]
    meter = ContextMeter(262_144)
    meter.observe_usage({"input_tokens": int(262_144 * 0.95)}, len(msgs))
    meter.observe_append(msgs)
    return _relieve_context(msgs, options=options, meter=meter,
                            reason="intra_turn")


def _compacted(**over) -> dict:
    """A `load_and_compact_session` result of a history under the threshold.

    The interesting case rather than a rewrite: a rewrite also lands a usage
    record, and asserting the row twice for one mechanism would make it look as
    though the decision half were covered when only the rewrite half were.
    """
    out = {
        "history": [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"}],
        "tokens_before": 5_000, "tokens_after": 5_000,
        "truncated": False, "summarized": False, "microcompacted": 0,
        "restored_files": 0, "context_window": 262_144, "threshold": 210_144,
        "summarize_attempted": False, "summarize_outcome": "under_threshold",
    }
    out.update(over)
    return out


@contextmanager
def _wired(monkeypatch, *, boom: bool = False, **over):
    """A private `messages.py` with everything outside the usage write faked, one
    relief pass fired from inside `run_query` so it lands mid-turn.

    Yields `captured`, whose `"module"` key is the copy — the driver takes
    `_run_turn`/`post_message` from it, because those are the only function objects
    whose globals are the dict the fakes were written into.

    `boom=True` makes the harness emit one real assistant turn and then raise,
    which is the shape of a turn that died under context pressure: the overflow
    branch relieves, re-raises, and the writer's `except` arm is the only place
    left to book the pass.
    """
    from app.sessions_io import create_session

    # The turn's session has to exist on disk: `post_message` reads its metadata
    # before it compacts anything. It lands in conftest's scratch data root
    # (`LLOYD_DATA` points the whole suite off `~/lloyd-data`), so no fixture is
    # needed to keep this off the machine's real sessions.
    # `create_session` and `post_message` both resolve through
    # `app.paths.SESSIONS_DIR`, which conftest points at a scratch data root for
    # the whole suite, so the session this creates cannot touch the machine's.
    create_session(SESSION_ID, platform="mission-control", model="primary",
                   title="t", source="test")

    captured: dict = {"compacted": _compacted(**over)}

    async def _fake_compact(*a, **kw):
        return captured["compacted"]

    async def _fake_prepare(history, model=None, **kw):
        return list(history)

    def _fake_run_query(msgs, opts, **kw):
        async def _gen():
            captured["gen_started"] = True
            captured["options"] = opts
            try:
                captured["relief"] = _relief_report_like_production(opts)
            except Exception as e:
                captured["relief_error"] = repr(e)
            if boom:
                # A turn that dies mid-stream: content streamed, then the loop
                # blew up before any `result` event. This is the shape the relief
                # ladder's most consequential firings live in — a pressure-killed
                # turn never reaches its usage row.
                yield {"type": "assistant_message", "content": "half an answer",
                       "model": "primary",
                       "usage": {"input_tokens": 1234, "output_tokens": 7},
                       "message_id": "m1"}
                raise RuntimeError("simulated harness failure mid-turn")
            yield {"type": "assistant_message", "content": "hi",
                   "model": "primary", "usage": {}, "message_id": "m1"}
            yield {"type": "result", "stop_reason": "end_of_turn", "usage": {},
                   "duration_ms": 10, "duration_api_ms": 9, "num_turns": 1}
        return _gen()

    # Every target here must exist on the copy: a swallowed `AttributeError` would
    # leave the REAL collaborator running behind the test, which reads as a pass
    # while the seam under test is unexercised. Three names this file originally
    # listed (`app.routers.messages._inject_subl_messages`,
    # `app.routers.messages._drive_inner_voice`,
    # `app.sessions_io.load_session_messages`) are attributes of nothing at all,
    # and the guard hid that for a whole round — `grep -c` for each in the module
    # it names returns 0. They are gone rather than fixed because the router has no
    # such call: `patch_on_copy` now refuses a name the copy does not carry.
    msg = load_messages_copy(monkeypatch,
                             name="messages_under_compaction_record_test")
    with ExitStack() as stack:
        patch_on_copy(stack, msg, [
            ("_build_subliminal_entry", MagicMock(
                return_value={"role": "user", "content": "sub"})),
            ("_classify_subliminal", MagicMock(
                return_value={"tag": "user"})),
            ("_detect_subliminal_sources", MagicMock(
                return_value={})),
            ("load_and_compact_session", _fake_compact),
            ("_prepare_messages_for_harness", _fake_prepare),
            ("run_query", _fake_run_query),
            ("_post_session_capture", AsyncMock()),
            ("_maybe_extract_focus", AsyncMock()),
            ("build_system_prompt", lambda *a, **k: "SYS"),
            ("prefetch_context_async", _fake_prefetch),
            ("_append_messages", AsyncMock()),
            ("set_last_user_session", MagicMock()),
            ("attach_observer_for_turn", _no_observer),
            ("_build_state_anchor", lambda *a, **k: ""),
        ])
        captured["module"] = msg
        captured["captured"] = captured
        yield captured
        # A relief pass that raised inside the fake used to land in
        # `captured["relief_error"]` unread, so a broken seam surfaced as a
        # downstream KeyError on `relief_passes` — and, worse, passed silently
        # when an earlier test in the process had left the module in a shape that
        # made it work. Both halves of the seam are asserted where they belong.
        assert captured.get("gen_started"), (
            "the driven turn never consumed `run_query`, so neither the relief "
            "pass nor the usage insert under test actually ran"
        )
        assert captured.get("relief_error") is None, (
            f"the production relief emitter failed inside the driven turn: "
            f"{captured.get('relief_error')}"
        )
        assert (captured.get("relief") or {}).get("rungs"), (
            f"the pass ran no rung, so there is nothing to record: "
            f"{captured.get('relief')}"
        )


async def _no_observer(*a, **k):
    """The Inner Voice attach hook, as `test_usage_skill_breakdown.py` stubs it."""
    return None


async def _fake_prefetch(*a, **k):
    return "prefetched"



def _drive_run_turn(msg, tmp_path, *, overflow: bool = False) -> None:
    """Run one turn through the real streaming consumer, on a real queue.

    The previous round recorded this as impossible — "`_run_turn` cannot be
    driven to its own `run_query` from a test in this tree" — and left the two
    inserts it owns to an AST guard. That is not what happens: driven like this,
    through the same `_wired()` the loopback test uses, the turn consumes the
    fake generator, runs its relief pass inside it, and writes its row. What
    the earlier attempt had was an `asyncio.Event` built by a *closed* loop
    (`asyncio.run(asyncio.Event())`) handed to the queue before the running one
    existed.

    `msg` is `_wired`'s copy, not `app.routers.messages`: the fakes are on the
    copy's dict, and importing the live module's `_run_turn` here would run a
    turn whose every collaborator is real. See `tests/_messages_copy.py`.
    """
    from app.sessions_io import SessionQueue

    q = SessionQueue()

    async def _go():
        # Built inside the running loop, as `SessionQueue` expects: an Event
        # created by a finished `asyncio.run` belongs to a dead loop.
        q.cancel_event = asyncio.Event()
        await msg._run_turn(SESSION_ID, _turn(tmp_path, overflow=overflow), q)

    asyncio.run(_go())


def _turn(tmp_path, *, overflow: bool = False):
    """A queued turn shaped the way `post_message_stream` builds one.

    `_run_turn` reads `turn.turn_id`, `turn.source` and `turn.payload`; the
    payload carries the `RunOptions` and the session's meta path, so this is the
    same object the producer hands the consumer in production, not a stand-in
    with the fields a stub would need.
    """
    from app.harness.options import RunOptions
    from app.sessions_io import SessionTurn

    meta = tmp_path / f"{SESSION_ID}.json"
    meta.write_text(json.dumps({"messages": [], "model": "primary"}))
    options = RunOptions(model="primary", max_turns=60)
    if overflow:
        # The overflow branch is gated on the route having built a meter, which
        # it only does with the ladder configured — the same gate production has.
        options.context_config = {"enabled": True}
    return SessionTurn(
        turn_id="t1078", source="user",
        payload={"text": "hi", "prefetched_text": "hi", "model": "primary",
                 "options": options,
                 "meta_path": meta},
        enqueued_at=0.0,
    )


SESSION_ID = "s"


def _events(session_id: str) -> list[dict]:
    """This session's rows from the non-rotating store the clause names.

    `tests/conftest.py`'s `_isolate_background_records` puts
    `event_log.EVENT_LOGS_DIR` on a scratch root for the whole suite, so this is
    a real `event_logs/<session>.events.jsonl` read, just not the machine's.
    """
    from app import event_log

    path = event_log.EVENT_LOGS_DIR / f"{session_id}.events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


def _relief_events(session_id: str) -> list[dict]:
    return [e for e in _events(session_id)
            if e.get("event") == "harness.context_relief"]


def _raw_compactions(store, session_id: str) -> list:
    """Every compaction record this session's usage rows carry, parsed (None = NULL)."""
    import usage_store

    usage_store._conn().commit()
    return [None if v is None else json.loads(v) for (v,) in
            list(sqlite3.connect(store).execute(
                "SELECT compaction FROM usage WHERE session_id = ?", (session_id,)))]


def _row(store, session_id: str):
    import usage_store

    usage_store._conn().commit()
    rows = list(sqlite3.connect(store).execute(
        "SELECT compaction FROM usage WHERE session_id = ?", (session_id,)))
    assert len(rows) == 1, f"expected one usage row, got {len(rows)}"
    return json.loads(rows[0][0]) if rows[0][0] is not None else None


class _Req:
    """The only thing `post_message` reads off the FastAPI request."""

    async def json(self):
        return {"text": "hi", "session_id": "s"}


def test_a_loopback_post_lands_the_same_record(store, monkeypatch):
    """`post_message`, the third writer and the route a worker's loopback turn
    takes — the one whose absence would leave the heaviest ladder users
    unrecorded, since a worker posts to itself rather than streaming.
    """
    with _wired(monkeypatch) as wired:
        asyncio.run(wired["module"].post_message(_Req()))
    record = _row(store, "s")
    assert record is not None, (
        "the loopback path must write the same dimension the streamed path does, "
        "or the two are not comparable for the run that fires the ladder hardest"
    )
    assert record["relief_passes"] == 1
    assert record["turn_start"]["summarize_attempted"] is False


def test_a_streamed_turn_lands_the_same_record(store, monkeypatch, tmp_path):
    """The seam the review called unverified, crossed by execution rather than by
    AST: `run_query` inside `_run_turn` → the session-keyed registry the relief
    ladder writes through → this route's own `record_usage` insert.

    Nothing here calls the registry or the record: the pass is booked by
    `app/harness/loop.py` reaching `options.session_id`, and read back by
    `messages.py` at insert time. A unit test on either side of that gap proves
    nothing about the gap, which is exactly how the mechanism came to have 7,253
    firings that belonged to no session.

    Runs alone as well as in-file: everything `_wired` needs is built and applied
    inside the test, so no earlier test in the process can be what makes it pass —
    and if the driven turn never reaches `run_query`, the helper's own
    `gen_started` assertion says that, instead of a missing key three lines later.
    """
    with _wired(monkeypatch) as wired:
        _drive_run_turn(wired["module"], tmp_path)

    record = _row(store, SESSION_ID)
    assert record is not None, (
        "the streaming writer booked no context-policy record for a turn whose "
        "ladder fired; NULL here is indistinguishable from a turn nobody measured"
    )
    assert record["relief_passes"] == 1, record
    assert record["relief"][0]["rungs"], (
        "the row names the rungs that ran, not only that something ran"
    )
    assert record["relief_tokens_freed"] > 0
    assert record["turn_start"]["ran"] is True, (
        "the turn-start stack ran on this turn; that it declined to rewrite is a "
        "recorded decision, not an absence"
    )
    assert record["turn_start"]["summarize_attempted"] is False
    assert record["turn_start"]["summarize_outcome"] == "under_threshold"

    events = _relief_events(SESSION_ID)
    assert len(events) == 1, (
        f"the same pass must appear once in the event store too, got {events}"
    )
    assert events[0]["turn_id"] == "t1078", (
        "the firing is attributable to the streamed turn that needed it — the "
        "log line this replaces carried a token count and no owner at all"
    )
    assert events[0]["data"]["rungs"] == record["relief"][0]["rungs"], (
        "the two stores are projections of one report and cannot disagree"
    )


def test_a_streamed_turn_that_dies_mid_flight_still_leaves_the_firing_behind(store, monkeypatch, tmp_path):
    """A turn that dies before its `result` event: the event is the only record.

    Driven through `_run_turn`'s `except` arm — the failure path. The firing
    survives where the clause puts it: one event in
    `event_logs/<session>.events.jsonl`, carrying the session id, the rungs and
    the tokens.

    Until D12 such a turn wrote no usage row at all — `stream_stats` is filled
    only by the `result` handler, the event the crash skipped. It now books one
    from the harness's running totals (`TurnTelemetry.partial_row`), and that
    row carries the pass too, through the same `compaction=compaction_turn`.
    """
    with _wired(monkeypatch, boom=True) as wired:
        _drive_run_turn(wired["module"], tmp_path)

    events = _relief_events(SESSION_ID)
    assert len(events) == 1, (
        f"a crashed turn must still leave its pass behind, got {events}"
    )
    assert events[0]["session_id"] == SESSION_ID
    assert events[0]["turn_id"] == "t1078"
    assert events[0]["data"]["rungs"], (
        "the surviving record names the rungs — the log line it replaces named "
        "tokens and nothing else"
    )
    rows = _raw_compactions(store, SESSION_ID)
    assert len(rows) == 1 and rows[0] is not None, (
        f"the dead turn books one usage row carrying its relief pass (D12): {rows}"
    )


def test_relief_firings_are_countable_per_session_from_the_event_log():
    """Clause 2's claim, read back from the store the clause names.

    Two sessions, three passes: each session's file holds only its own firings,
    which is the number 7,253 anonymous `loop: context relief` lines could never
    produce — summing a log that never named an owner cannot partition by owner.
    Driven through `_relieve_context`, the production emitter, not a hand-built
    event.
    """
    from app.harness.loop import _relieve_context
    from app.harness.context_meter import ContextMeter

    def _fire(session_id: str, times: int) -> None:
        msgs = [
            {"role": "assistant", "content": f"m{i}", "reasoning": "R" * 40_000,
             "reasoning_content": "R" * 40_000}
            for i in range(12)
        ]
        meter = ContextMeter(262_144)
        meter.observe_usage({"input_tokens": int(262_144 * 0.95)}, len(msgs))
        meter.observe_append(msgs)
        for _ in range(times):
            _relieve_context(msgs, options=_opts_with_session(session_id),
                             meter=meter, reason="intra_turn")

    _fire("s1078_busy", 2)
    _fire("s1078_quiet", 1)

    busy = _relief_events("s1078_busy")
    quiet = _relief_events("s1078_quiet")
    assert len(busy) == 2, busy
    assert len(quiet) == 1, quiet
    assert all(e["event"] == "harness.context_relief" for e in busy + quiet)
    assert all(e["session_id"] == "s1078_busy" for e in busy), (
        "a count read from one session's file must not include another's"
    )


def _opts_with_session(session_id: str):
    """A `RunOptions` carrying only what the relief path reads: its session id."""
    from app.harness.options import RunOptions

    return RunOptions(model="primary", session_id=session_id)


def test_every_chat_insert_site_carries_the_field():
    """All three `record_usage` calls in `messages.py` pass the field, by AST.

    Execution above covers `_run_turn`'s result and error arms and `post_message`;
    this pins the shape so a fourth insert, or a refactor that drops the kwarg
    from one of them, fails here rather than silently storing NULL forever — the
    same guard shape `tests/test_usage_skill_breakdown.py` uses for the skill
    dimension, for the same reason: a missing kwarg at one writer is invisible in
    the data, it just looks like traffic that never relieved.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "app/routers/messages.py").read_text()
    tree = ast.parse(src)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "value", None) is not None
        and getattr(node.func.value, "id", "") == "usage_store"
        and getattr(node.func, "attr", "") == "record_usage"
    ]
    assert len(calls) == 3, (
        f"expected the three chat writers, found {len(calls)} — if a fourth was "
        f"added, this guard and the route it sits on both need reading"
    )
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert "compaction" in kwargs, (
            f"record_usage at line {call.lineno} stores no context-policy "
            "record; a writer without the kwarg stores NULL forever and NULL is "
            "indistinguishable from a turn that never measured"
        )
