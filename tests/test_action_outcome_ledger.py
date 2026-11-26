"""#4048 — the action-outcome ledger, its populator, taxonomy and resolver.

`app/harness/action_outcomes.py` grades automod-ledger events into
`action_outcomes` (declared in `workers/queue.py::GRANT_DDL`, beside
`authority_grants`) and resolves a class to `probation` or `clearance`. Every
case here is a temp SQLite file and synthetic ledger bytes: no model, no prompt,
no live state dir. The dispatch-side arm is pinned in
`tests/unit/test_grant_policy.py` (clause 5).
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLOYD_HOME))

from app.harness import action_outcomes as AO  # noqa: E402
from app.harness.policy import GrantStore  # noqa: E402
from workers import queue as Q  # noqa: E402

NOW = dt.datetime(2026, 11, 20, 12, 0, tzinfo=dt.timezone.utc)
NAMED_EVENTS = {"rollback_succeeded", "regression_check", "review", "vault_land"}


@pytest.fixture
def store(tmp_path) -> GrantStore:
    s = GrantStore(tmp_path / "workers.db")
    s.ensure_schema()
    return s


def _ev(event: str, days_ago: float, **fields) -> dict:
    t = NOW - dt.timedelta(days=days_ago)
    return {"ts": t.timestamp(),
            "created_at": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "event": event, **fields}


def _write_ledger(tmp_path: Path, archive: list[dict], live: list[dict],
                  dup: dict | None = None) -> list[Path]:
    """An archive month (gzip) and the live file, the shapes the retention
    sweep leaves on disk. `dup` is written to both, as a fold half-done does."""
    arch_rows = archive + ([dup] if dup else [])
    live_rows = ([dup] if dup else []) + live
    a = tmp_path / "promotions-archive-202610.jsonl.gz"
    a.write_bytes(gzip.compress(
        "".join(json.dumps(r) + "\n" for r in arch_rows).encode()))
    p = tmp_path / "promotions.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in live_rows))
    return [a, p]


def _synthetic(tmp_path):
    archive = [
        _ev("rollback_succeeded", 60, trigger="crash", commit="a" * 40),
        _ev("rollback_succeeded", 59, trigger="promote_failed", commit="b" * 40),
        *[_ev("regression_check", 58 - i * 0.1, regressed=False, commit=f"{i:040x}")
          for i in range(10)],
        _ev("promoted", 57, commit="c" * 40, round_id="SM_1"),
        _ev("gate", 57, rung="tests", ok=False),
        _ev("vault_land", 50, ok=True, item_id=1, commit="d" * 40),
        _ev("vault_land", 49, ok=False, item_id=2,
            errors=["nothing to commit on those paths: the working tree matches HEAD"]),
    ]
    live = [
        _ev("regression_check", 20, regressed=True, confirmed_by=None, commit="e" * 40),
        _ev("regression_check", 10, regressed=True,
            confirmed_by=["ndcg10 0.29 → 0.27"], commit="f" * 40),
        *[_ev("regression_check", 9 - i * 0.1, regressed=False, commit=f"{i + 50:040x}")
          for i in range(5)],
        _ev("review", 8, round_id="SM_2", ok=True, blocking=True,
            clauses=[{"clause": 1, "verdict": "unmet"}]),
        _ev("review", 7, round_id="SM_3", ok=False, blocking=False,
            error="review exceeded 600.0s"),
        _ev("vault_land", 5, ok=False, item_id=3, errors=["skills/x: does not load"],
            reverted=["skills/x/SKILL.md"]),
        _ev("vault_land", 4, ok=True, item_id=4, commit="9" * 40),
        _ev("vault_revert", 3, reverted="9" * 40, reason="x"),
        _ev("rollback_started", 2, trigger="crash", commit="0" * 40),
    ]
    dup = _ev("regression_check", 30, regressed=False, commit="1" * 40)
    return _write_ledger(tmp_path, archive, live, dup)


# ── Clause 1: the table, beside authority_grants, NOT-NULL where it matters ──


def _ddl_tables(text: str) -> list[str]:
    return re.findall(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)\s*\(", text)


def test_action_outcomes_is_declared_in_the_grant_ddl_beside_authority_grants():
    tables = _ddl_tables(Q.GRANT_DDL)
    assert "authority_grants" in tables and AO.OUTCOME_TABLE in tables, tables
    # The queue's own schema concatenates GRANT_DDL, so the pool creates it too —
    # one declaration, two openers, the #534 rule.
    assert Q.GRANT_DDL in Q._SCHEMA
    # And nowhere else: a second CREATE for the table is the drift #534 forbids.
    hits = [p for p in (LLOYD_HOME / "app").rglob("*.py")
            if re.search(r"CREATE\s+TABLE[^(]*\baction_outcomes\b", p.read_text())]
    assert hits == [], f"a second declaration of action_outcomes: {hits}"


def test_outcome_and_evidence_ref_are_not_null(store):
    with sqlite3.connect(str(store.db_path)) as conn:
        cols = {r[1]: r for r in conn.execute(
            f"PRAGMA table_info({AO.OUTCOME_TABLE})")}
    for col in ("outcome", "evidence_ref", "source_event", "source_event_id",
                "action_class", "ts"):
        assert cols[col][3] == 1, f"{col} is nullable"
    base = ("vault_land", "bad", "vault_land", "id1", "2026-11-01T00:00:00+00:00",
            "ref", None, "2026-11-01T00:00:00+00:00")
    sql = (f"INSERT INTO {AO.OUTCOME_TABLE} (action_class, outcome, source_event,"
           " source_event_id, ts, evidence_ref, subject, recorded_at)"
           " VALUES (?,?,?,?,?,?,?,?)")
    with sqlite3.connect(str(store.db_path)) as conn:
        for idx in (1, 5):
            row = list(base)
            row[idx] = None
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql, row)
        bad_outcome = list(base)
        bad_outcome[1] = "fine"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql, bad_outcome)


def test_the_queue_creates_the_table_on_a_fresh_db(tmp_path):
    db = tmp_path / "w.db"
    Q.WorkQueue(db)
    with sqlite3.connect(str(db)) as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"authority_grants", AO.OUTCOME_TABLE} <= names


def test_every_row_names_the_ledger_line_it_came_from(store, tmp_path):
    paths = _synthetic(tmp_path)
    AO.populate(store, paths, now=NOW)
    ids = {hashlib.sha256(line).hexdigest()[:24]: line
           for line in AO.iter_ledger_lines(paths)}
    with store._connect() as conn:
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {AO.OUTCOME_TABLE}")]
    assert rows
    for r in rows:
        line = ids.get(r["source_event_id"])
        assert line is not None, f"row {r} names no line in the ledger"
        assert json.loads(line)["event"] == r["source_event"]
        assert r["source_event_id"] in r["evidence_ref"]


def test_populate_is_idempotent_across_a_half_done_fold(store, tmp_path):
    paths = _synthetic(tmp_path)
    first = AO.populate(store, paths, now=NOW)
    again = AO.populate(store, paths, now=NOW)
    assert first > 0 and again == 0
    # The duplicated line (in both archive and live) is one row.
    with store._connect() as conn:
        n = conn.execute(f"SELECT count(*) FROM {AO.OUTCOME_TABLE} WHERE subject=?",
                         ("1" * 12,)).fetchone()[0]
    assert n == 1


# ── Clause 2: seeded only from the named events; rollup re-derived from bytes ─


def _independent_rollup(paths, action_class: str) -> tuple[int, int, int]:
    """The rollup restated from the ledger bytes with no call into the module:
    the event filters as clause 2 names them, read straight off the lines."""
    seen, graded = set(), []
    for path in paths:
        raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" \
            else path.read_bytes()
        for line in raw.splitlines():
            if not line.strip() or line in seen:
                continue
            seen.add(line)
            e = json.loads(line)
            name, outcome, cls = e.get("event"), None, None
            if name == "rollback_succeeded":
                cls = "automod_land"
                if e.get("trigger") in {"regression", "data_damage", "error_rate", "crash"}:
                    outcome = "bad"
            elif name == "regression_check":
                cls = "automod_land"
                if e.get("regressed") is False:
                    outcome = "clean"
                elif e.get("regressed") is True and e.get("confirmed_by"):
                    outcome = "bad"
            elif name == "vault_land":
                cls = "vault_land"
                if e.get("ok") is True:
                    outcome = "clean"
                elif not any("nothing to commit" in x for x in e.get("errors") or []):
                    outcome = "bad"
            # `review`: neutral while ruling (b) is open (REVIEW_REFUSAL_GRADES).
            if cls == action_class and outcome:
                graded.append((e["ts"], outcome))
    graded.sort()
    horizon = (NOW - dt.timedelta(days=30)).timestamp()
    bad30 = sum(1 for ts, o in graded if o == "bad" and ts >= horizon)
    streak = 0
    for _ts, o in reversed(graded):
        if o == "bad":
            break
        streak += 1
    return len(graded), bad30, streak


@pytest.mark.parametrize("action_class", ["automod_land", "vault_land"])
def test_the_rollup_matches_one_derived_straight_from_the_ledger_bytes(
        store, tmp_path, action_class):
    assert AO.REVIEW_REFUSAL_GRADES is False, (
        "ruling (b) changed — restate the review filter in _independent_rollup")
    paths = _synthetic(tmp_path)
    AO.populate(store, paths, now=NOW)
    r = AO.rollup(AO.rows_for(store, action_class, before=NOW), now=NOW)
    expect = _independent_rollup(paths, action_class)
    assert (r.graded_outcomes, r.bad_outcomes_30d, r.clean_streak) == expect
    # Non-vacuity: the fixture has something to count in every field that can.
    assert expect[0] > 0 and expect[1] > 0


def test_the_synthetic_rollup_figures(store, tmp_path):
    """The same numbers by hand, so a matching pair of wrong filters cannot pass."""
    AO.populate(store, _synthetic(tmp_path), now=NOW)
    am = AO.rollup(AO.rows_for(store, "automod_land", before=NOW), now=NOW)
    # crash rollback + 10 clean + dup clean + confirmed regression + 5 clean.
    assert (am.graded_outcomes, am.bad_outcomes_30d, am.clean_streak) == (18, 1, 5)
    vl = AO.rollup(AO.rows_for(store, "vault_land", before=NOW), now=NOW)
    assert (vl.graded_outcomes, vl.bad_outcomes_30d, vl.clean_streak) == (3, 1, 1)


def test_only_the_named_events_seed_the_table(store, tmp_path):
    AO.populate(store, _synthetic(tmp_path), now=NOW)
    with store._connect() as conn:
        sources = {r[0] for r in conn.execute(
            f"SELECT DISTINCT source_event FROM {AO.OUTCOME_TABLE}")}
    assert sources <= NAMED_EVENTS, sources
    # A non-named event shaped exactly like a graded one grades nothing.
    for name in ("promoted", "vault_revert", "rollback_started", "gate", "settled"):
        assert AO.grade_event({"event": name, "ts": NOW.timestamp(), "ok": False,
                               "regressed": True, "confirmed_by": ["x"],
                               "trigger": "crash", "commit": "a" * 40}, "x") is None


def test_the_event_filters_book_the_noisy_rows_neutral():
    t = NOW.timestamp()
    g = AO.grade_event
    assert g({"event": "rollback_succeeded", "ts": t, "trigger": "promote_failed"},
             "1").outcome == AO.NEUTRAL
    assert g({"event": "rollback_succeeded", "ts": t, "trigger": "a_new_detector"},
             "1").outcome == AO.NEUTRAL
    assert g({"event": "regression_check", "ts": t, "regressed": True,
              "confirmed_by": []}, "1").outcome == AO.NEUTRAL
    assert g({"event": "regression_check", "ts": t, "regressed": True,
              "confirmed_by": ["d"]}, "1").outcome == AO.BAD
    assert g({"event": "vault_land", "ts": t, "ok": False,
              "errors": ["nothing to commit on those paths"]}, "1").outcome == AO.NEUTRAL
    assert g({"event": "vault_land", "ts": t, "ok": False,
              "errors": ["x: does not load"]}, "1").outcome == AO.BAD


def test_a_review_grades_only_as_a_judged_refusal_and_only_once_ruled(monkeypatch):
    """`ok` is the grader-failure flag (gate.py writes ok:false only when the
    grader could not judge), so the refusal is ok:true + blocking:true naming a
    non-met clause. While ruling (b) is open every review row is neutral."""
    t = NOW.timestamp()
    refusal = {"event": "review", "ts": t, "round_id": "SM_9", "ok": True,
               "blocking": True, "clauses": [{"clause": 2, "verdict": "partial"}]}
    unreliable = {"event": "review", "ts": t, "round_id": "SM_9", "ok": False,
                  "blocking": False, "clauses": [{"clause": 1, "verdict": "unmet"}],
                  "unreliable": ["cites a missing sha"]}
    assert AO.grade_event(refusal, "1").outcome == AO.NEUTRAL
    monkeypatch.setattr(AO, "REVIEW_REFUSAL_GRADES", True)
    got = AO.grade_event(refusal, "1")
    assert got.outcome == AO.BAD and "clauses=[2]" in got.evidence_ref
    assert AO.grade_event(unreliable, "1").outcome == AO.NEUTRAL
    passed = {**refusal, "blocking": False,
              "clauses": [{"clause": 1, "verdict": "met"}]}
    assert AO.grade_event(passed, "1").outcome == AO.CLEAN


# ── Clause 3: the frozen taxonomy and its excluded set ──────────────────────


def test_the_taxonomy_is_frozen_small_and_names_a_source_per_class():
    assert 1 <= len(AO.ACTION_CLASSES) <= AO.MAX_CLASSES == 8
    with pytest.raises(TypeError):
        AO.ACTION_CLASSES["new_class"] = None  # type: ignore[index]
    for name, cls in AO.ACTION_CLASSES.items():
        assert cls.name == name
        assert cls.sources and set(cls.sources) <= NAMED_EVENTS, (name, cls.sources)
        assert cls.tools and cls.graded_by.strip()
    # Every named event grades some class, and no event grades two.
    owners = [s for c in AO.ACTION_CLASSES.values() for s in c.sources]
    assert sorted(owners) == sorted(NAMED_EVENTS)


def test_the_durable_external_classes_are_in_the_named_excluded_set():
    with pytest.raises(TypeError):
        AO.EXCLUDED_CLASSES["x"] = "y"  # type: ignore[index]
    for name in ("email_send", "calendar_write", "contacts_write"):
        assert name in AO.EXCLUDED_CLASSES and AO.EXCLUDED_CLASSES[name].strip()
    assert not set(AO.EXCLUDED_CLASSES) & set(AO.ACTION_CLASSES)
    # No tool of an excluded durable-external class is a mapped terminal step.
    for tool in AO.TOOL_CLASS:
        assert not tool.startswith(("email_", "calendar_", "contacts_")), tool


# ── Clause 4: probation / clearance, K, and the unrecognised class ──────────


def _rows(*spec):
    """(days_ago, outcome) pairs → rows in time order."""
    return [{"ts": (NOW - dt.timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
             "outcome": o, "evidence_ref": f"ref-{i}-{o}"}
            for i, (d, o) in enumerate(sorted(spec, key=lambda x: -x[0]))]


K = AO.CLEARANCE_STREAK_K


def test_k_is_a_named_constant_with_its_rationale():
    src = (LLOYD_HOME / "app" / "harness" / "action_outcomes.py").read_text()
    assert isinstance(K, int) and K >= 1
    block = src.split("CLEARANCE_STREAK_K = ")[0].rsplit("\n\n", 1)[-1]
    assert "Rationale" in block and "rule of three" in block, block


def test_a_bad_outcome_inside_the_window_is_probation_naming_it():
    rows = _rows(*[(40 - i * 0.1, "clean") for i in range(K)], (5, "bad"),
                 *[(4 - i * 0.01, "clean") for i in range(K - 1)])
    s = AO.resolve_rows("vault_land", rows, now=NOW)
    assert s.status == AO.PROBATION
    assert s.evidence_ref.endswith("-bad"), s


def test_fewer_than_k_graded_is_probation_even_with_no_bad():
    s = AO.resolve_rows("vault_land", _rows(*[(1, "clean")] * (K - 1)), now=NOW)
    assert s.status == AO.PROBATION and f"K={K}" in s.why
    # Neutral rows are not graded outcomes.
    s = AO.resolve_rows("vault_land", _rows(*[(1, "clean")] * (K - 1),
                                            *[(2, "neutral")] * 50), now=NOW)
    assert s.status == AO.PROBATION


def test_k_clean_consecutive_outcomes_restore_clearance():
    before = _rows((5, "bad"), *[(4 - i * 0.01, "clean") for i in range(K - 1)])
    assert AO.resolve_rows("vault_land", before, now=NOW).status == AO.PROBATION
    after = _rows((5, "bad"), *[(4 - i * 0.01, "clean") for i in range(K)])
    assert AO.resolve_rows("vault_land", after, now=NOW).status == AO.CLEARANCE


def test_a_bad_outcome_outside_the_window_no_longer_demotes():
    rows = _rows((AO.WINDOW_DAYS + 5, "bad"),
                 *[(AO.WINDOW_DAYS + 4 - i * 0.01, "clean") for i in range(K - 1)])
    rows += _rows((1, "clean"))
    assert AO.resolve_rows("vault_land", rows, now=NOW).status == AO.CLEARANCE


@pytest.mark.parametrize("name", ["", "email_send", "automod-land", "nonesuch"])
def test_an_unrecognised_class_is_probation_never_clearance(store, name):
    perfect = _rows(*[(1 - i * 0.001, "clean") for i in range(K * 3)])
    assert AO.resolve_rows(name, perfect, now=NOW).status == AO.PROBATION
    # Through the store too, with rows written under that class name.
    with store._connect() as conn:
        for i, r in enumerate(perfect):
            conn.execute(
                f"INSERT INTO {AO.OUTCOME_TABLE} (action_class, outcome, source_event,"
                " source_event_id, ts, evidence_ref, recorded_at) VALUES (?,?,?,?,?,?,?)",
                (name, r["outcome"], "vault_land", f"{name}-{i}", r["ts"],
                 r["evidence_ref"], r["ts"]))
    assert AO.resolve(store, name, now=NOW).status == AO.PROBATION


def test_seed_bad_demotes_then_a_clean_streak_restores_through_the_store(store, tmp_path):
    """The item's scripted suite, model out of the loop: ledger bytes in, a
    demotion out, then K clean landings appended and clearance back."""
    base = [_ev("vault_land", 40 - i * 0.1, ok=True, item_id=i) for i in range(K)]
    paths = _write_ledger(tmp_path, base, [
        _ev("vault_land", 3, ok=False, item_id=99, errors=["x: does not load"])])
    AO.populate(store, paths, now=NOW)
    s = AO.resolve(store, "vault_land", now=NOW)
    assert s.status == AO.PROBATION and "item=99" in s.evidence_ref
    live = paths[1]
    with live.open("a") as fh:
        for i in range(K):
            fh.write(json.dumps(_ev("vault_land", 2 - i * 0.01, ok=True,
                                    item_id=200 + i)) + "\n")
    AO.populate(store, paths, now=NOW)
    assert AO.resolve(store, "vault_land", now=NOW).status == AO.CLEARANCE


# ── The backtest entry point (read-only, archives included) ─────────────────


def test_the_backtest_reads_the_archives_counts_withheld_and_writes_nothing(tmp_path):
    from scripts.automod import action_outcomes_backtest as BT
    K = AO.CLEARANCE_STREAK_K
    archive = [
        *[_ev("vault_land", 80 - i * 0.1, ok=True, item_id=i) for i in range(K)],
        *[_ev("regression_check", 80 - i * 0.1, regressed=False, commit=f"{i:040x}")
          for i in range(K)],
        _ev("promoted", 40, commit="a" * 40, round_id="SM_A"),
        _ev("rollback_succeeded", 39, trigger="crash", commit="a" * 40),
    ]
    live = [
        _ev("promoted", 20, commit="b" * 40, round_id="SM_B"),   # withheld, clean
        _ev("vault_land", 10, ok=False, item_id=7, errors=["x: does not load"]),
        _ev("vault_land", 9, ok=False, item_id=8, errors=["y: does not load"]),
        _ev("vault_land", 8, ok=True, item_id=9),
        _ev("vault_land", 7, ok=False, item_id=10,
            errors=["nothing to commit on those paths"]),     # no-op: not an action
    ]
    _write_ledger(tmp_path, archive, live)
    before = sorted((p.name, p.stat().st_mtime_ns) for p in tmp_path.iterdir())
    paths = BT.ledger_paths(tmp_path)
    assert [p.name for p in paths] == ["promotions-archive-202610.jsonl.gz",
                                       "promotions.jsonl"]
    rep = BT.backtest(paths, now=NOW, days=90)
    am, vl = rep["classes"]["automod_land"], rep["classes"]["vault_land"]
    # SM_A lands on a clean K-streak and is later rolled back: allowed, bad.
    # SM_B lands 19 d after that crash: withheld, never shown bad.
    assert (am["actions"], am["withheld"], am["withheld_later_bad"],
            am["allowed_later_bad"]) == (2, 1, 0, 1)
    # The first K landings have no track record yet (0..K-1 graded before them);
    # the first refusal lands on clearance; the next two follow a bad in window.
    assert (vl["actions"], vl["withheld"], vl["withheld_later_bad"],
            vl["allowed_later_bad"]) == (K + 3, K + 2, 1, 1)
    assert vl["withheld_by_reason"] == {"no_track_record": K, "bad_in_window": 2}
    assert sorted((p.name, p.stat().st_mtime_ns) for p in tmp_path.iterdir()) == before
