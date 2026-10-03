"""The failure-issue ledger (#2079, the re-land of #611).

Five things are pinned here, one per acceptance clause 1-5 (clause 6, the
committed witness bytes behind the quoted figures, is pinned by
`tests/test_failure_ledger_witness.py`), and each one is a behaviour the ledger
could get wrong without a red test:

1. every signature is the guardian's own, so `tests/test_guardian_logtail.py` and
   `tests/test_guardian_predicates.py` keep counting what they count today;
2. the sweep derives issues from an insert-only event log and cannot rewrite its
   own onset;
3. ingestion reads the rotated archive AND the live ledger, because the live file
   alone understates the guardian's own family as 2 occurrences instead of 18;
4. both detectors emit rows, with growth measured over active days only;
5. one investigation run per day, with a record-id-bounded prompt.

Everything runs against a `tmp_path` database and fixture files written into
`tmp_path`, so no test here can read or write the live
`/home/alansrobotlab/lloyd-data/failure-ledger.db`.
"""

from __future__ import annotations

import collections
import gzip
import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import failure_ledger as fl          # noqa: E402
from app import paths                          # noqa: E402
from workers.queue import WorkQueue            # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "failure_ledger"
NOW = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)
TODAY = "2026-10-02"


def ledger(tmp_path: Path, name: str = "ledger.db") -> sqlite3.Connection:
    return fl.connect(tmp_path / name)


def real_lines(group: str | None = None) -> list[tuple[str, str, str]]:
    """Read `group<TAB>logger<TAB>message` rows out of the pinned fixture."""
    out = []
    for line in (FIXTURES / "real_failure_lines.txt").read_text(
            encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("group\t"):
            continue
        grp, logger_name, message = line.split("\t", 2)
        if group is None or grp == group:
            out.append((grp, logger_name, message))
    return out


def seed(conn: sqlite3.Connection, message: str, stamps: list[datetime],
         *, logger_name: str = "lloyd-worker", run_id: str = "") -> str:
    sig = ""
    for dt in stamps:
        sig = fl.record_failure(conn, logger_name=logger_name, message=message,
                                ts=fl.iso_utc(dt), run_id=run_id)
    fl.sweep(conn)
    return sig


# ---------------------------------------------------------------------------
# Clause 1 — one signature, borrowed from the guardian, never a second one
# ---------------------------------------------------------------------------


def test_the_two_video_id_fetch_crashes_collapse_to_one_signature():
    r"""`fetch crashed: --fetch rc=2` on two video ids is ONE family, not two.

    `normalize_message` collapses them because `_PATH`=`(/[\w.\-]+){2,}` eats
    `/youtu.be/<id>`, so no new regex is needed for a video id that appears in a
    URL — and adding one would change what the guardian's error-rate rollback
    treats as novel.
    """
    rows = real_lines("fetch-crashed")
    assert len(rows) == 2, "the fixture must hold the pair #611 named"
    sigs = {fl.signature(logger_name, msg) for _, logger_name, msg in rows}
    assert len(sigs) == 1, [fl.normalize_message(m) for _, _, m in rows]


def test_the_two_hypothesis_error_classes_stay_two_signatures():
    """`Unterminated string starting at:` and `overlay_files missing/empty` are
    the two root causes #446 and #489 split by hand. One signature here would
    mean the normalizer is over-collapsing and the blob is back.
    """
    by_group = {g: m for g, _, m in real_lines() if g.startswith("hypothesis")}
    trunc = fl.signature("autoresearch.hypothesis", by_group["hypothesis-echo-truncation"])
    schema = fl.signature("autoresearch.hypothesis", by_group["hypothesis-schema-invalid"])
    assert trunc != schema
    anchor = fl.signature("autoresearch.hypothesis",
                          by_group["hypothesis-contract-anchor"])
    assert len({trunc, schema, anchor}) == 3, "the verbatim surviving dump is a third class"


def test_the_ledger_owns_no_normalizer_of_its_own():
    """`signature()` is the guardian's own function object, and the module's
    source contains no regex call of any kind — checked against the bytes on disk,
    so a pattern compiled inside a function or an inline `re.sub` cannot pass this
    by having an unremarkable name. A second normalizer would be invisible in the
    diff that adds it and fatal in the report that uses it.
    """
    source = (ROOT / "app" / "failure_ledger.py").read_text(encoding="utf-8")
    for call in ("re.compile", "re.sub", "re.match", "re.search", "re.findall",
                 "re.finditer", "re.fullmatch"):
        assert call not in source, f"the ledger normalizes with {call}"
    assert "import re" not in source, "the module does not even import `re`"
    assert not [v for v in vars(fl).values() if isinstance(v, re.Pattern)]
    assert fl.normalize_message is fl.detect.normalize_message
    assert fl.signature is fl.detect.signature


#: (logger, message, normalized, signature) with the guardian's output written out
#: as LITERALS, measured off `agent-services/guardian/detect.py` at this base.
#:
#: Literals are the whole point. The node this replaces asserted
#: `fl.signature(m) == fl.detect.signature(m)` over this same battery, and since
#: `app/failure_ledger.py` BINDS those names to one function object
#: (`signature = detect.signature`), every one of those assertions was
#: `f(x) == f(x)`: it could not fail, and a widened or replaced normalizer sailed
#: through it. A literal can fail: mutate `_NUM` in the guardian to `\d{3,}` and
#: `run_30_20261002_211240` normalizes to `run_30_<n>_<n>` instead of
#: `run_<n>_<n>_<n>`, and this node goes red on its first tuple. (`\d{2,}`,
#: measured, changes nothing in the first six shapes, which is why the lone-digit
#: tuple below is in the battery.) That is the property clause 1 needs: the
#: ledger counts what the guardian's error-rate rollback counts, because both
#: numbers are the same pinned number.
#:
#: The last tuple is the same message under the other logger prefix. `signature()`
#: hashes `logger|normalized`, so the alert family and a worker line that reads
#: identically must not share a signature — the two hexes are 12 characters apart
#: in spelling and worlds apart in meaning.
SHAPE_BATTERY = (
    ("lloyd-worker",
     "run_30_20261002_211240 died at /home/alansrobotlab/lloyd/app/x.py after 1250 ms",
     "run_<n>_<n>_<n> died at <path> after <n> ms",
     "6fa551325b3b"),
    ("lloyd-worker",
     "rollback at commit fc253ffe3d5b1a2f after 3.5s",
     "rollback at commit <hex> after <dur>",
     "00e7721c2165"),
    ("lloyd-worker",
     "session 20261002_163656_autocode_f661 has no summary",
     "session <n>_<n>_autocode_f<n> has no summary",
     "e2d7a8d0b855"),
    ("lloyd-worker",
     "audio device busy: /run/user/1000/pulse/native refused connection",
     "audio device busy: <path> refused connection",
     "483e92c1e88b"),
    ("lloyd-worker",
     "cuda oom: 95.37 GiB in 8000 ms",
     "cuda oom: <n>.<n> GiB in <n> ms",
     "daeacb726cd7"),
    # Lone single digits are in here on purpose. The first six shapes have no
    # digit that stands alone, so widening `_NUM` to `\d{2,}` — a real widening,
    # and the first one a person reaches for — leaves every one of them
    # byte-identical and a battery without this tuple would sit green through it.
    ("lloyd-worker",
     "worker 7 died after 1 retry on queue row 3",
     "worker <n> died after <n> retry on queue row <n>",
     "323d2c623fe0"),
    ("guardian.alert",
     "Service down, but no promotion to revert: needs a human — again",
     "Service down, but no promotion to revert: needs a human — again",
     "cc2af3167682"),
)


def test_the_shape_battery_signatures_are_pinned_literals_not_self_comparisons():
    """Behavioural half of clause 1: on paths, shas, durations, ids, non-ASCII
    text and the logger prefix, what the ledger stores is the literal string and
    the literal signature the guardian produces — compared to bytes written down
    in this file, not to the same function the module under test imports.
    """
    got = [(logger_name, fl.normalize_message(message),
            fl.signature(logger_name, message))
           for logger_name, message, _, _ in SHAPE_BATTERY]
    want = [(logger_name, norm, sig) for logger_name, _, norm, sig in SHAPE_BATTERY]
    assert got == want
    sigs = [row[2] for row in got]
    assert len(sigs) == 7 and len(set(sigs)) == 7, "seven shapes, seven signatures"
    alert = SHAPE_BATTERY[-1]
    assert alert[0] == fl.ALERT_LOGGER, "the last tuple is the alert family's prefix"
    assert fl.signature("lloyd-worker", alert[1]) != sigs[-1], \
        "the logger prefix is part of the hash, so a worker line cannot merge " \
        "with the alert family that reads the same"


def test_a_run_id_and_a_yyyymmdd_need_no_new_regex():
    """Both forms `#611` asked to strip already collapse under `_NUM`, which is
    why the normalizer is reused as-is rather than extended.
    """
    a = fl.signature("lloyd-worker", "run_30_20261002_211240 has no summary")
    b = fl.signature("lloyd-worker", "run_77_20261005_091500 has no summary")
    assert a == b


# ---------------------------------------------------------------------------
# Clause 2 — the store, and a sweep that cannot rewrite its own baseline
# ---------------------------------------------------------------------------


def test_the_store_is_a_database_under_the_data_root():
    """`<DATA_ROOT>/failure-ledger.db` — `/home/alansrobotlab/lloyd-data/
    failure-ledger.db` in production. Never in the code tree: on 2026-09-22 a
    fixture teardown deleted `~/lloyd` in 35 s with every database in it.
    """
    assert paths.FAILURE_LEDGER_DB.name == "failure-ledger.db"
    assert paths.FAILURE_LEDGER_DB.parent == paths.DATA_ROOT


def test_the_two_ledger_tables_have_the_pinned_columns(tmp_path):
    conn = ledger(tmp_path)
    cols = {t: [r[1] for r in conn.execute(f"PRAGMA table_info({t})").fetchall()]
            for t in ("failure_issues", "failure_events")}
    assert cols["failure_issues"] == [
        "signature", "source", "first_seen", "last_seen", "occurrences", "per_day",
        "sample_summary", "status"]
    assert cols["failure_events"] == ["signature", "ts", "run_id", "source"]


def test_sweep_twice_over_one_event_set_is_byte_identical(tmp_path):
    """The clause's own test: `first_seen` and `occurrences` do not move when the
    sweep runs again, so the ledger never rewrites its own baseline.
    """
    conn = ledger(tmp_path)
    seed(conn, "worker died: ConnectError: all attempts failed",
         [NOW - timedelta(days=3), NOW - timedelta(days=1), NOW - timedelta(hours=5)])
    seed(conn, "overlay_files missing/empty",
         [NOW - timedelta(hours=9), NOW - timedelta(hours=8), NOW - timedelta(hours=7)])
    fl.sweep(conn)
    first = conn.execute(
        "SELECT signature, first_seen, occurrences, per_day, last_seen "
        "FROM failure_issues ORDER BY signature").fetchall()
    assert len(first) == 2
    assert sorted(r[2] for r in first) == [3, 3]
    assert all(r[1].endswith("Z") for r in first), "one canonical timestamp spelling"

    fl.sweep(conn)
    fl.sweep(conn)
    again = conn.execute(
        "SELECT signature, first_seen, occurrences, per_day, last_seen "
        "FROM failure_issues ORDER BY signature").fetchall()
    assert again == first


def test_reingesting_the_same_events_cannot_double_count_them(tmp_path):
    """Idempotence is a store property, not a caller's habit: `failure_events`
    is insert-only with a uniqueness guard, so one alert is one event however
    many times the file that holds it is opened.
    """
    conn = ledger(tmp_path)
    before = seed(conn, "queue item stuck 1300 min", [NOW - timedelta(hours=2)])
    n0 = conn.execute("SELECT COUNT(*) FROM failure_events").fetchone()[0]
    again = fl.record_failure(conn, logger_name="lloyd-worker",
                             message="queue item stuck 1300 min",
                             ts=fl.iso_utc(NOW - timedelta(hours=2)))
    fl.sweep(conn)
    assert again == before
    assert conn.execute("SELECT COUNT(*) FROM failure_events").fetchone()[0] == n0
    assert conn.execute("SELECT occurrences FROM failure_issues").fetchone()[0] == 1


def test_deleting_the_earliest_event_never_moves_onset_later(tmp_path):
    """`first_seen` is written as `MIN(stored, derived)`. A re-derivation that
    recomputed onset from whatever events survive would silently age a young
    family into an old one the week the sweep pruned its own history.
    """
    stamps = [NOW - timedelta(days=6), NOW - timedelta(hours=30), NOW - timedelta(hours=2)]
    conn = ledger(tmp_path)
    sig = seed(conn, "speak died: audio device busy", stamps)
    first = fl.get_issue(conn, sig)["first_seen"]
    assert first == fl.iso_utc(stamps[0])

    conn.execute("DELETE FROM failure_events WHERE ts = ?", (first,))
    fl.sweep(conn)
    issue = fl.get_issue(conn, sig)
    assert issue["first_seen"] == first
    assert issue["occurrences"] == 2, "the count follows the events; onset does not"


# ---------------------------------------------------------------------------
# Clause 3 — archive AND live ledger, never the live file alone
# ---------------------------------------------------------------------------


def write_promotions_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Stage the two real shapes: rotated `.jsonl.gz` archive + live `.jsonl`."""
    archive = tmp_path / "promotions-archive-202609.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as fh:
        fh.write((FIXTURES / "promotions_archive_rows.jsonl").read_text(encoding="utf-8"))
    live = tmp_path / "promotions.jsonl"
    live.write_text((FIXTURES / "promotions_live_rows.jsonl").read_text(encoding="utf-8"),
                    encoding="utf-8")
    return archive, live


def guardian_family(conn: sqlite3.Connection):
    sig = fl.signature(fl.ALERT_LOGGER, "Service down, but no promotion to revert")
    return fl.get_issue(conn, sig)


def test_the_guardian_family_is_18_occurrences_from_2026_09_06(tmp_path):
    """16 rows in the rotated archive, 2 in the live ledger: 18 occurrences of one
    family, onset 2026-09-06T18:34:06Z. Read from disk and pinned, not assumed —
    the archive-only grep is 16 and the live-only grep is 2.
    """
    archive, live = write_promotions_fixture(tmp_path)
    conn = ledger(tmp_path)
    counts = fl.ingest_promotions(conn, live=live, archives=[archive])
    fl.sweep(conn)
    assert counts["considered"] == 18

    issue = guardian_family(conn)
    assert issue is not None
    assert issue["occurrences"] == 18
    assert issue["first_seen"] == "2026-09-06T18:34:06Z"
    assert issue["last_seen"] == "2026-09-29T23:14:16Z"
    assert issue["source"] == fl.SOURCE_LEDGER
    per_day = json.loads(issue["per_day"])
    assert sum(per_day.values()) == 18
    assert per_day["2026-09-06"] == 6
    assert "2026-09-16" in per_day and "2026-09-24" in per_day


def test_reading_the_live_ledger_alone_understates_the_family_as_two(tmp_path):
    """The failure this clause exists to prevent. Live-only reports a fault that
    began 2026-09-06 as one that began 2026-09-24, and an investigation aimed at
    that onset window looks for a change that never happened.
    """
    _, live = write_promotions_fixture(tmp_path)
    conn = ledger(tmp_path)
    fl.ingest_promotions(conn, live=live)
    fl.sweep(conn)
    issue = guardian_family(conn)
    assert issue["occurrences"] == 2
    assert issue["first_seen"] == "2026-09-24T23:39:42Z"


def test_alerts_are_keyed_on_title_so_the_body_does_not_split_the_family(tmp_path):
    """Measured on the 18 real rows: keying on `title: body` splits this one
    family into THREE signatures (12 / 5 / 1) — probe failure, STOPPED without an
    intentional stop, and `FATAL: can't find command` — three issues where the
    operator saw one. The body is kept, as the sample, not as the key.
    """
    archive, live = write_promotions_fixture(tmp_path)
    conn = ledger(tmp_path)
    fl.ingest_promotions(conn, live=live, archives=[archive])
    fl.sweep(conn)
    sigs = {r[0] for r in conn.execute(
        "SELECT signature FROM failure_events WHERE source = ?",
        (fl.SOURCE_LEDGER,)).fetchall()}
    assert len(sigs) == 1

    body = guardian_family(conn)["sample_summary"]
    assert "STOPPED without an intentional stop" in body, "the sample is the first row seen"
    keyed = collections.Counter(
        fl.signature(fl.ALERT_LOGGER, f"{r['title']}: {r['body']}")
        for name in ("promotions_archive_rows.jsonl", "promotions_live_rows.jsonl")
        for r in fl.iter_jsonl(FIXTURES / name))
    assert sorted(keyed.values(), reverse=True) == [12, 5, 1], \
        "body-keyed the family is three issues, not one: the keying choice is what " \
        "makes the count 18, not 12"


def test_a_warn_alert_is_not_counted_as_a_failure(tmp_path):
    r"""A warn is the guardian thinking out loud, and counting it would inflate
    every prevalence number. Measured at this base over the two corpora and named
    as measured: the live `promotions.jsonl` holds 57 `alert` rows (55 `error`,
    1 `warn`, 1 `critical`); with `promotions-archive-202609.jsonl.gz` the pair is
    86 (79 `error`, 5 `warn`, 2 `critical`). This test pins the filter, not the
    counts — the counts are here so the next reader can check the sentence.
    """
    live = tmp_path / "promotions.jsonl"
    live.write_text(json.dumps({
        "ts": 1788719646.9, "created_at": "2026-09-26T01:02:03Z", "event": "alert",
        "level": "warn", "title": "Worker pool near saturation", "body": "8/8 claimed",
        "round_id": "SM_L"}) + "\n" + json.dumps({
        "ts": 1788719647.0, "created_at": "2026-09-26T01:03:03Z", "event": "promoted",
        "title": "Service down, but no promotion to revert", "body": "not a failure",
        "round_id": "SM_L"}), encoding="utf-8")
    conn = ledger(tmp_path)
    assert fl.ingest_promotions(conn, live=live) == {"considered": 0, "inserted": 0}
    assert conn.execute("SELECT COUNT(*) FROM failure_events").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Clause 4 — detectors emit rows; growth is over ACTIVE days only
# ---------------------------------------------------------------------------


def kinds(conn: sqlite3.Connection, day: str = TODAY) -> list[tuple[str, str]]:
    return [(r[0], r[1]) for r in conn.execute(
        "SELECT signature, kind FROM failure_findings WHERE detected_on = ? "
        "ORDER BY kind, signature", (day,)).fetchall()]


def test_a_day_old_family_trips_new_onset_and_not_growth(tmp_path):
    conn = ledger(tmp_path)
    sig = seed(conn, "kg rebuild died: vault lock held",
               [NOW - timedelta(hours=9), NOW - timedelta(hours=6),
                NOW - timedelta(hours=3)])
    ids = fl.detect_new_onset(conn, now=NOW)
    assert ids and kinds(conn) == [(sig, "new_onset")]
    assert fl.detect_growth(conn, day=TODAY) == []


def test_growth_trips_at_twice_the_active_day_median_and_new_onset_does_not(tmp_path):
    """Trailing-7-day window holds two active days with 2 and 3 events, so the
    active-day median is 2.5 and today's 6 is >= 2x it. `first_seen` is 5 days
    old, so new-onset must stay quiet.
    """
    conn = ledger(tmp_path)
    sig = seed(conn, "email triage died: IMAP socket timeout", [
        NOW - timedelta(days=5), NOW - timedelta(days=5, hours=-1),
        NOW - timedelta(days=2), NOW - timedelta(days=2, hours=-1),
        NOW - timedelta(days=2, hours=-2),
        NOW - timedelta(hours=11), NOW - timedelta(hours=10),
        NOW - timedelta(hours=9), NOW - timedelta(hours=8),
        NOW - timedelta(hours=7), NOW - timedelta(hours=6),
    ])
    assert fl.median_active(fl.trailing_counts(conn, sig, TODAY)) == 2.5
    assert fl.detect_growth(conn, day=TODAY)
    assert kinds(conn) == [(sig, "growth")]
    assert fl.detect_new_onset(conn, now=NOW) == []


def test_an_all_zero_trailing_window_never_trips_growth(tmp_path):
    """Five events today and nothing in the 7 days before it: there is no rate to
    compare against, so growth cannot fire. `median_active` returns None rather
    than 0.0 precisely so a caller cannot turn that absence into a threshold of
    zero, which would trip every family on its first bad day.
    """
    conn = ledger(tmp_path)
    sig = seed(conn, "djev shadow worker died: cuda oom",
               [NOW - timedelta(hours=h) for h in (11, 10, 9, 8, 7)])
    assert fl.median_active(fl.trailing_counts(conn, sig, TODAY)) is None
    assert fl.detect_growth(conn, day=TODAY) == []
    assert kinds(conn) == []


def test_zero_days_are_not_in_the_growth_median(tmp_path):
    """3 events today against a sparse window of 2 and 3: over active days the
    threshold is 5 and today does not clear it. Put the zero-days in the median
    and it is 0.0, so 2x it is 0 and this family "grows" every day it exists —
    which is the failure mode the clause names.
    """
    conn = ledger(tmp_path)
    sig = seed(conn, "voice worker dropped a turn: asr timeout", [
        NOW - timedelta(days=5), NOW - timedelta(days=5, hours=-1),
        NOW - timedelta(days=2), NOW - timedelta(days=2, hours=-1),
        NOW - timedelta(days=2, hours=-2),
        NOW - timedelta(hours=10), NOW - timedelta(hours=9),
        NOW - timedelta(hours=8)])
    trailing = fl.trailing_counts(conn, sig, TODAY)
    assert sum(trailing.values()) == 5 and len(trailing) == 2
    assert fl.median_active(trailing) == 2.5
    assert fl.median_active({**{f"d{i}": 0 for i in range(5)},
                             **trailing}) == 2.5, "zeros are excluded, not averaged"
    assert fl.detect_growth(conn, day=TODAY) == []


def test_findings_are_rows_and_a_re_sweep_does_not_duplicate_them(tmp_path):
    conn = ledger(tmp_path)
    seed(conn, "voice worker dropped a turn: asr timeout",
         [NOW - timedelta(hours=9), NOW - timedelta(hours=6),
          NOW - timedelta(hours=3)])
    first = fl.detect_new_onset(conn, now=NOW)
    second = fl.detect_new_onset(conn, now=NOW)
    assert first == second
    assert conn.execute("SELECT COUNT(*) FROM failure_findings").fetchone()[0] == 1
    row = conn.execute(
        "SELECT kind, detected_on, occurrences FROM failure_findings").fetchone()
    # `tuple(row)`, not `row`: `fl.connect` returns `sqlite3.Row` rows because
    # `_strongest` and `derive_diagnosis` read columns by name, and a Row never
    # compares equal to a tuple. The claim here is about the three values.
    assert tuple(row) == ("new_onset", TODAY, 3)


# ---------------------------------------------------------------------------
# Clause 5 — one investigation run per day, bounded prompt
# ---------------------------------------------------------------------------


#: Distinctive WORDS, not digits: `_NUM` collapses `shape 1` and `shape 2` into
#: one signature, so a numbered seed would silently make twenty findings into
#: one and the dispatch tests would prove nothing about the cap.
SHAPES = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
          "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
          "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
          "victor", "whiskey", "xray", "yankee")


def seed_many_findings(conn: sqlite3.Connection, n: int) -> list[int]:
    stamps = [NOW - timedelta(hours=9), NOW - timedelta(hours=6),
              NOW - timedelta(hours=3)]
    for shape in SHAPES[:n]:
        seed(conn, f"worker run died with an unhandled {shape} exception", stamps)
    assert conn.execute("SELECT COUNT(*) FROM failure_issues").fetchone()[0] == n
    return fl.detect_new_onset(conn, now=NOW)


def queue_rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM queue ORDER BY id").fetchall()
    finally:
        conn.close()


def test_twenty_findings_enqueue_exactly_one_run(tmp_path):
    conn = ledger(tmp_path)
    ids = seed_many_findings(conn, 20)
    assert len(ids) == 20
    queue = WorkQueue(tmp_path / "workers.db")

    qid = fl.dispatch_findings(conn, TODAY, queue=queue,
                              diagnosis="what shipped between 09-29 and 10-02 "
                                        "that made 20 worker shapes die at once")
    assert qid is not None
    rows = queue_rows(tmp_path / "workers.db")
    assert len(rows) == 1, "the cap is one run per day, not one per finding"
    assert rows[0]["source"] == fl.SOURCE and rows[0]["kind"] == fl.KIND

    found = conn.execute(
        "SELECT dispatched, deferred_reason FROM failure_findings").fetchall()
    assert sum(r[0] for r in found) == 1
    assert sum(1 for r in found if r[0] == 0) == 19
    deferred = [r[1] for r in found if r[0] == 0]
    assert all(d and "1/day" in d for d in deferred), \
        "the 19 are recorded as deliberately undispatched, not dropped"


def test_the_prompt_carries_at_most_20_record_ids_and_the_onset_window(tmp_path):
    conn = ledger(tmp_path)
    seed_many_findings(conn, 25)
    queue = WorkQueue(tmp_path / "workers.db")
    qid = fl.dispatch_findings(conn, TODAY, queue=queue,
                              diagnosis="which change near onset introduced these")
    item = queue.get(int(qid))
    payload = item.payload
    assert len(payload["finding_ids"]) == fl.DISPATCH_MAX_RECORDS == 20
    assert payload["max_records"] == 20
    for rid in payload["finding_ids"]:
        assert str(rid) in payload["prompt"]
    # first_seen is NOW-9h = 2026-10-02T03:00:00Z, so the window is +/-3 days.
    assert payload["window"] == {"from": "2026-09-29", "to": "2026-10-05",
                                 "centre": "2026-10-02T03:00:00Z"}
    assert "2026-09-29" in payload["prompt"] and "2026-10-05" in payload["prompt"]
    assert payload["diagnosis"].startswith("which change")
    carried = conn.execute(
        "SELECT COUNT(*) FROM failure_findings WHERE dispatched = 0 "
        "AND deferred_reason IS NULL").fetchone()[0]
    assert carried == 5, "the 5 over the cap stay undispatched and uncarried"


def test_a_second_dispatch_the_same_day_enqueues_nothing(tmp_path):
    conn = ledger(tmp_path)
    seed_many_findings(conn, 3)
    queue = WorkQueue(tmp_path / "workers.db")
    assert fl.dispatch_findings(conn, TODAY, queue=queue,
                                diagnosis="what changed near onset") is not None
    assert fl.dispatch_findings(conn, TODAY, queue=queue,
                                diagnosis="what changed near onset") is None
    assert len(queue_rows(tmp_path / "workers.db")) == 1
    assert conn.execute("SELECT COUNT(*) FROM failure_dispatches").fetchone()[0] == 1


def test_the_next_day_gets_its_own_run(tmp_path):
    """The cap is per day, so a week of sweeps is at most 7 runs — the bound that
    keeps a detector that fires on noise from becoming a run that fires hourly.
    """
    conn = ledger(tmp_path)
    seed_many_findings(conn, 3)
    queue = WorkQueue(tmp_path / "workers.db")
    assert fl.dispatch_findings(conn, TODAY, queue=queue,
                                diagnosis="what changed near onset") is not None
    stamps = [NOW + timedelta(days=1) - timedelta(hours=h) for h in (9, 6, 3)]
    for i in range(3):
        seed(conn, f"tomorrow's exception shape {i}", stamps)
    tomorrow = (NOW + timedelta(days=1)).strftime("%Y-%m-%d")
    fl.detect_new_onset(conn, now=NOW + timedelta(days=1), day=tomorrow)
    assert fl.dispatch_findings(conn, tomorrow, queue=queue,
                                diagnosis="what changed near onset") is not None
    assert len(queue_rows(tmp_path / "workers.db")) == 2


def test_an_empty_diagnosis_is_rejected_and_nothing_is_enqueued(tmp_path):
    """A run whose prompt does not name what it is there to establish spends the
    day's single dispatch reading the store and reporting the weather.
    """
    conn = ledger(tmp_path)
    seed_many_findings(conn, 2)
    queue = WorkQueue(tmp_path / "workers.db")
    for blank in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="non-empty diagnosis"):
            fl.dispatch_findings(conn, TODAY, queue=queue, diagnosis=blank)
    assert queue_rows(tmp_path / "workers.db") == []
    assert conn.execute("SELECT COUNT(*) FROM failure_dispatches").fetchone()[0] == 0


def test_dispatch_with_no_findings_enqueues_nothing(tmp_path):
    conn = ledger(tmp_path)
    queue = WorkQueue(tmp_path / "workers.db")
    assert fl.dispatch_findings(conn, TODAY, queue=queue,
                                diagnosis="anything at all") is None
    assert queue_rows(tmp_path / "workers.db") == []


def test_a_dedup_collision_records_the_queue_row_that_actually_exists(tmp_path):
    """The queue can hold a day this ledger does not: a rebuilt ledger database
    has no cap row while `workers.db` still has the run. `enqueue` then coalesces
    and returns None, and the only honest thing to store is the id of the row that
    is really there — a sentinel would read as a row and never resolve.
    """
    queue = WorkQueue(tmp_path / "workers.db")
    orphan = queue.enqueue(source=fl.SOURCE, kind=fl.KIND, payload={"day": TODAY},
                           dedup_key=f"{fl.SOURCE}:{TODAY}")
    conn = ledger(tmp_path)
    seed_many_findings(conn, 2)

    qid = fl.dispatch_findings(conn, TODAY, queue=queue,
                               diagnosis="what changed near onset")
    assert qid == orphan
    assert len(queue_rows(tmp_path / "workers.db")) == 1, "coalesced, not duplicated"
    assert conn.execute(
        "SELECT queue_id FROM failure_dispatches WHERE day = ?", (TODAY,)).fetchone()[0] == orphan
    deferred = conn.execute(
        "SELECT deferred_reason FROM failure_findings WHERE deferred_reason IS NOT NULL"
    ).fetchall()
    assert deferred and all(f"queue row {orphan}" in r[0] for r in deferred)


def test_the_normalizer_is_loaded_by_path_because_agent_services_is_not_a_package():
    """The real seam: `agent-services` has a hyphen, so the ledger reaches the
    guardian's normalizer through `importlib` under a synthetic module name.

    Three things are asserted, and one is deliberately NOT. Asserted: the loaded
    object's own `__file__` is the guardian's file, so the seam lands on the same
    code the guardian runs; the module it came in under is `lloyd_guardian_detect`,
    a name no `import` statement could produce, which is what makes the by-path
    load load-bearing; and the guardian's normalizer actually runs through that
    seam — `run_30_20261002 died` comes back with its digits collapsed. NOT
    asserted: that `importlib.import_module("agent-services.guardian.detect")`
    fails. It does not: a directory with no `__init__.py` is a namespace package in
    Python 3, so the hyphenated dotted name resolves. An assertion that it raises
    would be a test that cannot fail, and this round was refused once for one of
    those.
    """
    resolved = Path(fl.detect.__file__).resolve()
    assert resolved == (paths.LLOYD_HOME / "agent-services" / "guardian" / "detect.py").resolve()
    assert fl.detect.__name__ == "lloyd_guardian_detect", "came in under a synthetic name"
    assert fl.detect.normalize_message("run_30_20261002 died") == "run_<n>_<n> died"




def test_the_denominator_is_runs_status_not_queue_depth(tmp_path):
    """`runs.status` counts attempts that happened; `queue.state` counts rows that
    exist. One dispatch that fails its check and is then poisoned is ONE failed run
    row and ONE dead queue row, and a row poisoned before any run id was minted is a
    dead queue row with NO run row at all. So "2 failed" names nothing until the
    store is named, and the module's README says which one is the denominator.
    """
    q = WorkQueue(tmp_path / "workers.db")
    ran = q.enqueue(source=fl.SOURCE, kind=fl.KIND, payload={"day": TODAY},
                    dedup_key=f"{fl.SOURCE}:ran")
    q.record_run(run_id="run_ran_1", queue_id=ran, source=fl.SOURCE, status="failed",
                 started_at=NOW.isoformat(), completed_at=NOW.isoformat(),
                 duration_seconds=60.0, summary="investigation errored")
    q.mark_completed(ran, dedup_release=False)
    dead = q.enqueue(source=fl.SOURCE, kind=fl.KIND, payload={"day": TODAY},
                     dedup_key=f"{fl.SOURCE}:dead")
    q.mark_failed(dead, f"unknown source {fl.SOURCE}", 0)
    q.enqueue(source=fl.SOURCE, kind=fl.KIND, payload={"day": TODAY},
              dedup_key=f"{fl.SOURCE}:open")

    out = fl.reconcile(q)
    assert out["runs_total"] == 1, "one attempt happened"
    assert out["runs_failed"] == 1
    assert out["queue_unrecovered"] == 1, "the poisoned row is a queue state"
    assert out["queue_open"] == 1, "the third row is still queued"
    assert out["queue_total"] == 3, "three intentions were written down"
    assert out["queue_total"] != out["runs_total"], \
        "the two stores cannot be reported as one number"
    assert out["reconcilable"] == out["queue_total"], \
        "every queue row lands in exactly one bucket of the reconciliation"
    assert out["runs_failed"] != out["queue_unrecovered"] + out["queue_open"], \
        "the run table's `failed` and the queue's unrecovered+open are different " \
        "quantities: one row succeeded and one attempt failed inside it"


def test_reconcile_names_nothing_for_a_source_that_never_dispatched(tmp_path):
    """A reader who asks the reconciliation about a source with no rows gets zeros,
    not a missing key and not another source's numbers.
    """
    out = fl.reconcile(WorkQueue(tmp_path / "workers.db"))
    assert out == {"queue_total": 0, "queue_unrecovered": 0, "queue_open": 0,
                   "runs_total": 0, "runs_failed": 0, "reconcilable": 0}
