"""The #3547 premise witness: can the 0.7563 flip-bar baseline still be recomputed?

`backlog/data/2026-11-06.3547-lloyd-data-workers.db` holds the **394**
`source='board-steward'` rows of `runs` whose `completed_at` falls in
`[2026-10-22T17:54, 2026-10-29T17:54)`, carried on the live store's own 14-column `runs`
schema, and it is the only surviving proof of one number: #3547 was landed on a flip-bar
premise of **28 disagreements over 25 items at rate 0.7563** (119 judged, 90 agree), and
its settled clause-6 evidence compares the post-change rate (0.8217 via the health API,
0.8265 by recount) against exactly that baseline.

Those 394 rows are perishable — `WORKER_RUN_MAX_AGE_DAYS = 30`
(`scripts/groundskeeper/retention-sweep.py:712`, DELETE at `:2117-2132`) makes them
prunable across 2026-11-21..28 — which is why this pin reads the committed bytes rather
than the runtime store: after the prune the live path holds nothing to recompute from, and
the revert decision would rest on a figure no reader can re-measure.

Three boundaries, and each is crossed by a node here rather than argued:

  * **runtime store -> committed artifact.** #3547's premise was read off
    `~/lloyd-data/workers.db` by `workers/dispatch_watch.py`'s judgement scan; what is
    graded now is a file in the vault. Nodes below re-aggregate over the **extract alone**
    with the same arithmetic, and one node compares the extract's `runs` DDL to the live
    store's own, so "verbatim schema" is a checked property and not a claim about a build.
  * **`WorkQueue` cannot read it.** The queue opens with `PRAGMA journal_mode=WAL` — which
    would *mutate a committed witness* — and its queries `LEFT JOIN queue`, which this
    one-table extract does not have (`no such table: queue`). So the extract is a
    read-only dataset, not a database to operate on: every node here opens it
    `file:…?mode=ro`, which is also what forbids a `-wal`/`-shm` sidecar appearing beside
    an artifact the vault's `.gitignore:48-49` hides from `git status`.
  * **ceiling.** `WITNESS_MAX_BYTES = 1_048_576` (`scripts/automod/backlog.py:1450`) caps
    what an owed-witness clause may order from a runtime store; the source measured
    32,935,936 B, 31.4× over it, so the *extract* is the thing that must be small. Pinned
    against the constant's own value, not a copy in a comment.

Absent artifact, absent store: every node **skips** in a checkout without the vault, the
way `tests/test_remeasured_pairing_extract.py` does, because the bytes live in
`~/obsidian/backlog/data/` and a CI box without it must go amber, not red.
"""

import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

import pytest

VAULT = Path(os.environ.get("LLOYD_VAULT", str(Path.home() / "obsidian")))
DATA = VAULT / "backlog" / "data"
DB_PATH = DATA / "2026-11-06.3547-lloyd-data-workers.db"
NOTE_PATH = DATA / "2026-11-06.3547-lloyd-data-workers.md"

#: The ceiling a witness artifact may not exceed — `scripts/automod/backlog.py:1450`.
WITNESS_MAX_BYTES = 1_048_576

#: The window the premise was measured over, and the cut this extract is. Text
#: comparison is sound here because every stamp is `+00:00`-offset and zero-padded.
SINCE, UNTIL = "2026-10-22T17:54", "2026-10-29T17:54"

#: The figure #3547's flip-bar premise rests on, and the stamp edges of the window.
ROWS = 394
BLOCKS = 383          # rows whose meta_json carries an agreement block with a `judged` key
JUDGED, AGREE, DISAGREE, RATE, DISAGREE_IDS = 119, 90, 28, 0.7563, 25
MIN_COMPLETED = "2026-10-22T18:14:38.682270+00:00"
MAX_COMPLETED = "2026-10-29T17:29:13.473815+00:00"

#: The live `runs` column list at publish time (2026-11-06T09:58:36Z, the read instant in
#: the note). `source` must survive: clause 3 proves "every row is board-steward" only if
#: the column is in the file, which is why the extract carries the schema instead of the
#: 4-column projection the item's body listed.
RUNS_COLUMNS = ["run_id", "queue_id", "source", "task_id", "status", "started_at",
                "completed_at", "duration_seconds", "summary", "artifact_path",
                "response_json", "meta_json", "claims_json", "kind"]

#: The 27 committed `*-lloyd-data-workers.db` extracts that were measured before this one
#: was ordered (2026-11-06). Hard-coded rather than globbed on purpose: a later round that
#: publishes extract #29 must not silently redden the note's "all 27 measured" claim.
MEASURED_EXTRACTS = [
    "2026-10-13.2670", "2026-10-14.2471", "2026-10-14.2692", "2026-10-14.2704",
    "2026-10-16.2817", "2026-10-16.2819", "2026-10-16.2846", "2026-10-17.2894",
    "2026-10-17.2899", "2026-10-17.2932", "2026-10-18.2983", "2026-10-20.3088",
    "2026-10-20.3089", "2026-10-22.3154", "2026-10-23.3183", "2026-10-23.3201",
    "2026-10-23.3211", "2026-10-25.3321", "2026-10-26.3356", "2026-10-26.3365",
    "2026-10-27.3308", "2026-10-27.3494", "2026-10-30.3602", "2026-10-31.3447",
    "2026-11-04.3643", "2026-11-05.3549", "2026-11-06.3607",
]


def _connect(path: Path) -> sqlite3.Connection:
    """Read-only, so no `journal_mode` pragma is ever issued against a committed witness."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _ro(path: Path):
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout (the vault lives in ~/obsidian)")


def _agreement_rows(conn: sqlite3.Connection):
    """Every `runs.meta_json` in the extract, oldest completion first."""
    return [r[0] for r in conn.execute(
        "SELECT meta_json FROM runs ORDER BY completed_at, run_id")]


def _aggregate(meta_jsons):
    """`judgement_aggregate`'s arithmetic (`workers/dispatch_watch.py`), over a row set.

    sum(judged), sum(len(agree)), sum(len(disagree)), agree/judged, and the DISTINCT ids
    across every `disagree` list — the same five columns the alarm reports, with the rate
    rounded at 4 dp the way the alarm prints it.
    """
    judged = agree = disagree = blocks = 0
    disagree_ids = set()
    for raw in meta_jsons:
        if not raw:
            continue
        try:
            block = json.loads(raw).get("agreement")
        except (ValueError, AttributeError):
            continue
        if not isinstance(block, dict) or "judged" not in block:
            continue
        blocks += 1
        judged += block.get("judged") or 0
        agree += len(block.get("agree") or [])
        listed = block.get("disagree") or []
        disagree += len(listed)
        disagree_ids.update(listed)
    return {"blocks": blocks, "judged": judged, "agree": agree, "disagree": disagree,
            "rate": (round(agree / judged, 4) if judged else None),
            "disagree_ids": len(disagree_ids)}


# ---------------------------------------------------------------- clause 1 ----

def test_the_3547_extract_exists_under_the_ceiling_with_394_inserts_and_one_populated_table():
    """#3842 clause 1: the file is there, fits the ceiling, and its `iterdump` emits exactly
    the 394 INSERT lines its `runs` row count implies — with no table other than `runs`
    carrying a row.

    The INSERT census is the clause the byte count cannot substitute for: a store copied
    whole would also be under some ceilings and over others, and the number that proves
    *this* artifact is the slice is the row count on the far side of the dump. The
    `sqlite_sequence` check is the #3183 shape (`queue` is `AUTOINCREMENT`, so copying it
    silently added a 4th INSERT line — vault commit `2398ebdb5`); the live `runs` DDL has
    no `AUTOINCREMENT`, so here the census must equal 394 exactly, not 394-plus.
    """
    _ro(DB_PATH)
    size = DB_PATH.stat().st_size
    assert size <= WITNESS_MAX_BYTES, (
        f"{DB_PATH} is {size:,} B, over WITNESS_MAX_BYTES = {WITNESS_MAX_BYTES:,} "
        f"({size / WITNESS_MAX_BYTES:.1%} of it); the note's shape is 434,176 B, "
        f"41.4% of the ceiling")

    with _connect(DB_PATH) as c:
        inserts = sum(1 for line in c.iterdump() if line.startswith("INSERT"))
        rows = c.execute("SELECT count(*) FROM runs").fetchone()[0]
        populated = {
            t: c.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
            for (t,) in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'")
        }
        ddl = c.execute("SELECT sql FROM sqlite_master WHERE name='runs'").fetchone()[0]

    assert rows == ROWS, (
        f"the extract holds {rows} runs rows, not the {ROWS} in "
        f"[{SINCE}, {UNTIL}) — the premise figure cannot be re-derived from this file")
    assert inserts == ROWS, (
        f"iterdump emitted {inserts} INSERT lines against {rows} runs rows: an extra "
        f"populated table (an AUTOINCREMENT sequence row, or a second table) got copied "
        f"out of workers.db, which is what #3183's clause caught")
    assert populated == {"runs": ROWS}, (
        f"a table other than runs carries rows, so this is not the runs slice: {populated}")
    assert "AUTOINCREMENT" not in ddl.upper(), (
        "the copied runs DDL gained AUTOINCREMENT, which would bring sqlite_sequence "
        "back into the dump and make the INSERT census differ from the row count")


# ---------------------------------------------------------------- clause 2 ----

def test_the_3547_figure_119_90_28_and_0_7563_recomputes_from_the_extract_alone():
    """#3842 clause 2: recomputing sum(judged), sum(len(agree)), sum(len(disagree)) and the
    distinct disagree ids over `runs.meta_json.agreement` **in the extract alone** yields
    judged 119, agree 90, disagree 28, rate 0.7563, 25 distinct disagree ids.

    Nothing from `~/lloyd-data/workers.db` is consulted by this node, and that is the
    point: after `WORKER_RUN_MAX_AGE_DAYS = 30` prunes the window (prunable across
    2026-11-21..28) the live path is the only other place these rows existed, so a node
    that fell back to it would pass while proving nothing.
    """
    _ro(DB_PATH)
    with _connect(DB_PATH) as c:
        got = _aggregate(_agreement_rows(c))

    assert got["blocks"] == BLOCKS, (
        f"{got['blocks']} of the {ROWS} rows carry an agreement block with a `judged` "
        f"key, not {BLOCKS}; the other {ROWS - BLOCKS} are ticks whose meta_json names no "
        f"judgement and add 0 to every column")
    assert got["judged"] == JUDGED, (
        f"sum(judged) recomputed from the extract is {got['judged']}, not the {JUDGED} "
        f"#3547's premise was measured over")
    assert got["agree"] == AGREE, f"sum(len(agree)) is {got['agree']}, not {AGREE}"
    assert got["disagree"] == DISAGREE, (
        f"sum(len(disagree)) is {got['disagree']}, not the {DISAGREE} disagreements the "
        f"flip-bar premise names")
    assert got["rate"] == RATE, (
        f"agree/judged is {got['rate']}, not {RATE} — the baseline #3547's settled "
        f"clause-6 evidence (0.8217 via health API, 0.8265 by recount) is compared against")
    assert got["disagree_ids"] == DISAGREE_IDS, (
        f"{got['disagree_ids']} distinct disagree ids, not {DISAGREE_IDS}: 28 mentions "
        f"over 25 items, so an item flagged twice counts once here")


# ---------------------------------------------------------------- clause 3 ----

def test_the_3547_extract_keeps_the_live_runs_schema_with_source_and_sits_inside_the_window():
    """#3842 clause 3: `runs` is the live schema verbatim — all 14 columns, `source` among
    them — every row it holds has `source='board-steward'`, and min/max(completed_at) are
    the two stamps below, inside the window.

    The item's body listed three columns (`run_id`, `completed_at`, `meta_json`); clause 3
    is unprovable against that projection, because dropping `source` makes "every row is
    board-steward" a sentence about bytes that do not contain it. Carrying the schema
    verbatim is what reconciles the two, and cost almost nothing: the full-schema extract
    is 434,176 B against 208,896 B for the projection, both under the ceiling.
    """
    _ro(DB_PATH)
    with _connect(DB_PATH) as c:
        cols = [r[1] for r in c.execute("PRAGMA table_info(runs)")]
        sources = sorted(r[0] for r in c.execute("SELECT DISTINCT source FROM runs"))
        edges = c.execute("SELECT min(completed_at), max(completed_at) FROM runs").fetchone()

    assert cols == RUNS_COLUMNS, (
        f"runs columns are {cols}, not the live store's 14 in order — the extract is a "
        f"projection, and `source` being absent is what would make clause 3 unprovable")
    assert "source" in cols, "the extract dropped `source`, so no node can prove its scope"
    assert sources == ["board-steward"], (
        f"the extract holds rows from {sources}, not the one source the premise was "
        f"measured over")
    assert edges == (MIN_COMPLETED, MAX_COMPLETED), (
        f"min/max(completed_at) are {edges[0]} and {edges[1]}, not {MIN_COMPLETED} and "
        f"{MAX_COMPLETED}: a different window is a different figure, which is exactly how "
        f"2026-10-30.3602 reads 115 · 89 · 25 · 0.7739 · 23 instead")
    assert SINCE <= edges[0] and edges[1] < UNTIL, (
        f"a stamp lies outside [{SINCE}, {UNTIL}): the extract is not the window it "
        f"claims to be")


def test_the_3547_extract_runs_ddl_is_the_live_stores_runs_ddl_character_for_character():
    """The other side of clause 3, across the boundary it is made of.

    "Verbatim schema" is a claim about a copy, so it is checked against the thing copied:
    the extract's `runs` DDL against the live store's, both read `mode=ro`. This is the
    node that would catch a hand-written `CREATE TABLE` that happens to have 14 columns
    and the right names in the right order but is not what `workers/db.py` ships.

    Skips rather than fails when `~/lloyd-data/workers.db` is absent or has already
    rotated, which is by design: after the 30-day prune the *rows* are gone, and this node
    compares DDL, which the store keeps — so it stays meaningful on a live box and says
    nothing on one that has never run the engine.
    """
    _ro(DB_PATH)
    live = Path(os.environ.get("LLOYD_DATA_DIR",
                               str(Path.home() / "lloyd-data"))) / "workers.db"
    if not live.exists():
        pytest.skip(f"{live} is not present, so there is no live schema to compare against")
    with _connect(DB_PATH) as ext, _connect(live) as src:
        ext_ddl = ext.execute("SELECT sql FROM sqlite_master WHERE name='runs'").fetchone()[0]
        src_ddl = src.execute("SELECT sql FROM sqlite_master WHERE name='runs'").fetchone()[0]
    assert ext_ddl == src_ddl, (
        "the extract's runs DDL is not the live store's, so the copy was typed rather "
        f"than carried:\nextract: {ext_ddl!r}\n  live: {src_ddl!r}")


# ------------------------------------------------------- clauses 4 and 5 ------

def test_the_3547_note_names_the_27_measured_extracts_and_the_near_miss_with_both_readings():
    """#3842 clause 4: the witness note lists all 27 extracts measured, with their
    recomputed figures, and names `2026-10-30.3602` as the near miss giving **both** 0.7795
    whole-file and 0.7739 window-filtered, stating which filter produced each.

    The two readings are the point of the node, not a flourish. `1d91e7e5a` printed
    `127 · 99 · 27 · 0.7795` when it published 3602, which is the whole-file number; this
    item quotes 0.7739, which is what the same bytes give under the window filter. A note
    carrying only the second reads as a false mismatch to anyone re-running the command
    from that commit, so both have to be on the page with their filters named.
    """
    _ro(NOTE_PATH)
    note = NOTE_PATH.read_text()

    table_rows = re.findall(r"^\|\s*`(2026-\d\d-\d\d\.\d+)`\s*\|", note, re.M)
    assert len(table_rows) == 27, (
        f"the note's §4 table names {len(table_rows)} extracts, not the 27 that were "
        f"measured before a new one was ordered: {table_rows}")
    assert set(table_rows) == set(MEASURED_EXTRACTS), (
        "the measured list and the note disagree: "
        f"only-in-note={set(table_rows) - set(MEASURED_EXTRACTS)} "
        f"only-in-test={set(MEASURED_EXTRACTS) - set(table_rows)}")
    for stem in MEASURED_EXTRACTS:
        assert f"`{stem}`" in note, f"{stem}-lloyd-data-workers.db was measured and is unnamed"

    assert "2026-10-30.3602" in note, "the near miss is not named at all"
    whole = re.search(r"[Ww]hole file:\s*127 · 99 · 27 · 0\.7795 · 25", note)
    windowed = re.search(r"[Ww]indow-filtered:\s*115 · 89 · 25 · 0\.7739 · 23", note)
    assert whole, (
        "the note does not give 3602's whole-file reading as 127 · 99 · 27 · 0.7795 · 25 — "
        "the figure commit 1d91e7e5a recorded when it published that extract")
    assert windowed, (
        "the note does not give 3602's window-filtered reading as "
        "115 · 89 · 25 · 0.7739 · 23 — the 0.7739 this item quotes")
    assert "1d91e7e5a" in note, (
        "the note does not say which commit recorded the 0.7795 reading, so a reader "
        "cannot tell the two readings' provenance apart")
    assert not re.search(r"\b0\.7563\b[^\n]{0,80}3602", note), (
        "the note attributes 0.7563 to 3602, which is the figure 3602 does NOT yield on "
        "either reading — the whole premise of ordering a new extract")


def test_the_3547_artifact_is_one_db_and_one_note_with_no_sidecar_and_its_alias_recorded():
    """#3842 clause 5: exactly one `.db` and one `.md` under `backlog/data` for this
    witness, no `-wal`/`-shm` sidecar beside them, and the note records
    `2026-10-29.3547-lloyd-data-workers.db` as the alias #3547's owed entry names.

    The sidecar half is the #2983/#3183 hazard, and it is checked by globbing the primary
    plus its companions rather than by `git status`: the vault's `.gitignore:48-49`
    (`backlog/data/*.db-wal`, `backlog/data/*.db-shm`) hides exactly that pair from
    `git status`, so a WAL journal left beside a witness is invisible to the door that
    would otherwise catch it. The alias half is what stops #3547's re-derive command —
    written against the window-close date — from reading as a missing artifact.
    """
    _ro(DB_PATH)
    _ro(NOTE_PATH)

    companions = sorted(p.name for p in DATA.glob("2026-11-06.3547-lloyd-data-workers*"))
    assert companions == ["2026-11-06.3547-lloyd-data-workers.db",
                          "2026-11-06.3547-lloyd-data-workers.md"], (
        f"the witness is not exactly two files: {companions} — a -wal/-shm pair beside a "
        f"committed witness means something opened it read-write, and .gitignore:48-49 "
        f"hides the pair from git status so no door would have reported it")

    note = NOTE_PATH.read_text()
    assert "2026-10-29.3547-lloyd-data-workers.db" in note, (
        "the note does not name the path #3547's owed entry asks for, so that entry's "
        "re-derive command resolves to nothing and reads as an unfulfilled demand")
    assert not (DATA / "2026-10-29.3547-lloyd-data-workers.db").exists(), (
        "a second extract was written under the alias name; clause 4 of the item allows "
        "one file, and two witnesses of one figure invite a reader to diff them")
    assert "394" in note and "434,176" in note, (
        "the note does not state the shape pair to quote (394 rows / 434,176 B), which is "
        "the thing a reader re-derives first")

    digest = hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
    assert digest in note, (
        f"the note does not carry sha256 {digest}, so a reader cannot tell the committed "
        f"bytes from a copy that drifted")
