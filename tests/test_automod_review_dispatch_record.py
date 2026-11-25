"""#4562 clause 4 and 5: every review is countable in the dispatch store, held or not.

What the store knew about the reviewer before this file: nothing. Over **10,643**
`runs` rows `select source,kind,count(*) from runs group by 1,2` carried no
`automod-review` class at all (top classes then were `autotriage|triage` 5118,
`owed-check|item` 1660, `autocode|round` 1140, `scheduled-task|run` 1119), while
2,198 grader sessions sat under `~/lloyd-data/sessions/` — 1,304 from live rounds,
834 vault grades, 60 calibration backfills. Automod *rounds* were visible; the review
rung inside them, and any grade outside one, was not.

That is the second half of the incident in `test_automod_review_pause.py`: an operator
who paused the machine for 27 h 21 m and asked the store what had run during the hold
was told **0**. A review that ran and a review that never started read identically.

`WorkQueue.record_run` already takes `queue_id=None` — documented as "Left None by a
caller that has no queue item to read one from" — and `kind` is free-form, so the
missing record was a wiring gap and not a schema limit: no migration, and no queue
item invented for a grade that has one.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from scripts.automod import review as RP
from workers.pool import PAUSE_WM_HOLDER_KEY, PAUSE_WM_KEY, PAUSE_WM_SOURCE
from workers.queue import WorkQueue

BACKEND = "http://127.0.0.1:9"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A dispatch store of the test's own, with no queue rows in it.

    Created empty and never given a queue item, so the `queue_id IS NULL` assertions
    below are about this run rather than about leftover rows.
    """
    path = tmp_path / "workers.db"
    q = WorkQueue(path)

    monkeypatch.setattr(RP, "_queue_store_path", lambda: path)
    return q


@pytest.fixture
def sessions(tmp_path):
    return tmp_path / "sessions"


def _hold(store, holder: str = "mission-control") -> None:
    store.wm_set(PAUSE_WM_SOURCE, PAUSE_WM_KEY, "1")
    store.wm_set(PAUSE_WM_SOURCE, PAUSE_WM_HOLDER_KEY, holder)


def _transport(monkeypatch, *, structured=None, raise_as=None) -> list:
    """The posted-turn recorder, with the two outcomes a status is derived from."""
    # One attempt per dispatch here. `run_grader` re-POSTs while it believes the
    # backend is merely unavailable, and this file measures the row a finished grade
    # books, not the wait: a node that let the retry loop run would sleep for real
    # (15/30/60 s) or, with the delay stubbed to zero, spin. The arm that raises says
    # "this backend is not coming back" instead, and the loop breaks on the first try.
    monkeypatch.setattr(RP, "_is_unavailable", lambda error: False)
    posted: list = []

    def fake(url, payload, timeout):
        posted.append((url, payload))
        if raise_as is not None:
            raise raise_as
        yield ("done", {"response": "graded", "stop_reason": "final",
                        "structured": structured if structured is not None
                        else {"premise": "sound", "clauses": [], "test_honesty": [],
                              "seams_unverified": []},
                        "structured_error": ""})

    monkeypatch.setattr(RP, "_post_stream", fake)
    return posted


def _queue_rows(store) -> int:
    """How many queue items exist. Zero, in every node here.

    The clause-4 claim is "written without a queue item", and the only way to tell
    that apart from a source that does claim one is to count the items as well as
    the runs.
    """
    with sqlite3.connect(store.db_path) as con:
        return con.execute("select count(*) from queue").fetchone()[0]


def _rows(store) -> list[dict]:
    with sqlite3.connect(store.db_path) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(
            "select * from runs where source = ? order by run_id", (RP.REVIEW_RUN_SOURCE,))]


def _classes(store) -> list[tuple]:
    """The item's own query, run against the store this round writes."""
    with sqlite3.connect(store.db_path) as con:
        return con.execute("select source,kind,count(*) from runs group by 1,2").fetchall()


# ── clause 4: a review that runs books one row naming its route ──────────────

@pytest.mark.parametrize("kind,round_id", [
    pytest.param("round", "SM_20261125_100809", id="code-round"),
    pytest.param("vault", "vault", id="vault-grade"),
    pytest.param("calibration", "backfill:cal_3981-gh-token-fallback", id="calibration"),
])
def test_a_review_that_runs_books_exactly_one_row_naming_its_route(
        tmp_path, monkeypatch, store, sessions, kind, round_id):
    """One grade, one row, `source='automod-review'`, `kind` = the route that ran.

    One row and not one per retry attempt: `run_grader` may POST the same grading
    several times against an unavailable backend and still returns one verdict, so a
    row per attempt would make the fleet count mean "turns" where every reader of it
    means "reviews".
    """
    _transport(monkeypatch)

    res = RP.run_grader(prompt="grade this", item_id=4562, round_id=round_id,
                        backend=BACKEND, sessions_dir=sessions)
    assert res["ok"] is True, res

    rows = _rows(store)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["source"] == "automod-review" == RP.REVIEW_RUN_SOURCE
    assert row["kind"] == kind
    assert row["status"] == "success"
    assert row["queue_id"] is None, "a grade has no queue item, and none was invented"
    assert row["run_id"].startswith("run_automod-review_")
    assert json.loads(row["response_json"])["error"] == "", "a success books no reason"
    meta = row["meta_json"]
    assert round_id in meta and '"item_id": 4562' in meta, meta
    assert res["session_id"] in meta, "the row points at the transcript that exists"
    assert _queue_rows(store) == 0, "no queue item was invented to hang a run row on"


def test_the_three_routes_are_named_from_the_ids_the_routes_actually_build(
        tmp_path, monkeypatch, store, sessions):
    """`kind` is read off the id each caller passes, and the ids are read from source.

    The naming only means something while it tracks the callers, so the shapes are
    quoted from the two places that build them: `grade_vault` passes the literal
    `vault`, and the backfill builds `backfill:{label}`. If either is renamed the node
    fails here rather than leaving `kind` quietly saying `round` for a fleet that is
    not rounds.
    """
    review_src = (RP.LIVE_ROOT / "scripts" / "automod" / "review.py").read_text(encoding="utf-8")
    tools_src = (RP.LIVE_ROOT / "scripts" / "automod" / "review_tools.py").read_text(encoding="utf-8")
    assert 'round_id="vault"' in review_src, "grade_vault no longer passes the literal vault"
    assert 'round_id=f"backfill:{label}"' in tools_src, "the backfill's id shape moved"

    assert RP.review_route("vault") == "vault"
    assert RP.review_route("backfill:cal_3981-gh-token-fallback") == "calibration"
    assert RP.review_route("SM_20261125_100809") == "round"
    # An unrecognised id is a round with an unusual label, not a fourth route.
    assert RP.review_route("SM_THE_FUTURE") == "round"
    assert set(RP.review_route(r) for _, r in
               [("a", "vault"), ("b", "backfill:cal_x"), ("c", "SM_X")]) == {
        "vault", "calibration", "round"}


# ── clause 5: a held review is also countable, and not as a success ──────────

def test_a_review_refused_by_the_hold_books_a_row_that_is_not_the_success_value(
        tmp_path, monkeypatch, store, sessions):
    """`select source,kind,count(*) from runs group by 1,2` shows the hold too.

    This is the half without which the record is a success log: the query an operator
    runs during a pause has to answer "what did the reviewer do?" and today it answers
    "nothing happened", which is the sentence that made a 27-hour fleet invisible.
    """
    _hold(store)
    _transport(monkeypatch)

    res = RP.run_grader(prompt="grade this", item_id=4562,
                        round_id="backfill:cal_3981-gh-token-fallback",
                        backend=BACKEND, sessions_dir=sessions)
    assert res["ok"] is False

    rows = _rows(store)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["source"] == "automod-review"
    assert row["kind"] == "calibration", "the route is still named when it did not run"
    assert row["status"] != "success" and row["status"] == "cancelled"
    assert "operator pool hold is armed" in row["response_json"], row["response_json"]
    assert ("automod-review", "calibration", 1) in _classes(store), _classes(store)


@pytest.mark.parametrize("outcome,want", [
    pytest.param("ok", "success", id="answered"),
    pytest.param("hold", "cancelled", id="held-by-operator"),
    pytest.param("refused", "failed", id="transport-refused"),
    pytest.param("no-object", "failed", id="turn-with-no-verdict"),
])
def test_the_booked_status_is_read_off_the_outcome_and_not_hardcoded(
        tmp_path, monkeypatch, store, sessions, outcome, want):
    """The seeded arms: four outcomes, four readings of one report.

    A `status` written from a literal would survive every other node in this file.
    Each arm below changes only what the grader did and asserts what the row says, so
    a `success` minted for a held or broken grade is caught here.
    """
    if outcome == "hold":
        _hold(store)
        _transport(monkeypatch)
    elif outcome == "refused":
        _transport(monkeypatch, raise_as=ConnectionRefusedError("nobody listening"))
    elif outcome == "no-object":
        _transport(monkeypatch, structured=None)
        monkeypatch.setattr(RP, "_post_stream", lambda *a, **k: iter(
            [("done", {"response": "prose only", "stop_reason": "final",
                       "structured": None, "structured_error": ""})]))
    else:
        _transport(monkeypatch)

    RP.run_grader(prompt="grade this", item_id=4562, round_id="SM_20261125_100809",
                  backend=BACKEND, sessions_dir=sessions)

    rows = _rows(store)
    assert len(rows) == 1, rows
    assert rows[0]["status"] == want, rows[0]
    if want != "success":
        assert json.loads(rows[0]["response_json"])["error"], (
            "a row that is not a success has to say what stopped it")


def test_a_review_record_survives_no_store_without_costing_the_grade(
        tmp_path, monkeypatch, store, sessions, caplog):
    """The bookkeeping never becomes the reason a round cannot be graded.

    `_queue_store_path` is pointed at a directory that cannot hold a database, so both
    the hold read and the record write fail. The grade still runs — clause 3's
    contract — and the run is reported as unlogged rather than dropped in silence,
    which is the failure mode that hid 46 sessions.
    """
    import logging

    (tmp_path / "unreadable.db").mkdir()   # a directory: no database can open there
    monkeypatch.setattr(RP, "_queue_store_path", lambda: tmp_path / "unreadable.db")
    posted = _transport(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=RP.logger.name):
        res = RP.run_grader(prompt="grade this", item_id=4562, round_id="SM_X",
                            backend=BACKEND, sessions_dir=sessions)

    assert res["ok"] is True, res
    assert len(posted) == 1
    assert any("unlogged" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records]
