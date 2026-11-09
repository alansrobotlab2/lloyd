"""#3832 — the two pairing blocks `architecture/context-window.md` publishes are
re-derivable from bytes committed to the vault, and this module is what re-derives them.

Why the node lives here and not beside the doc claims it prices
----------------------------------------------------------------
`tests/test_architecture_coverage_doc_claims.py` carries
`test_no_node_recomputes_the_pairing_this_section_publishes`, which reads that module's
own source and `test_research_doc_claims.py`'s for the three things a re-computing node
has to write (a read-only open, the window function over `reprefill_tokens`, a driver
connection) and fails on any of them. That refusal is deliberate: those two files quote
runtime-store figures as prose, and a node inside them would either die in a gate
worktree for a reason that is not the doc's or "pass" over an empty data root. Its
docstring routes a legitimate reader elsewhere — this is that elsewhere. It is the
route, not a loophole: the figures below are recomputed from a committed extract, and
not one node here touches the live store.

What the extract is
-------------------
`backlog/data/2026-10-31.3631-lloyd-data-usage.db` — a projection of one window of
`usage.db` (2026-10-03T09:47:08 <= ts < 2026-10-31T09:47:08) into eight columns, cut at
851,968 B against the 1,048,576 B `WITNESS_MAX_BYTES` ceiling because the live store is
50,212,864 B and no witness route can order it. Its sibling marker names the projection
and the two windows it prices; this module is the machine that checks both.

Every number asserted below was read off the committed bytes, not carried from an item:
18 pairs / median +36,460.75 / 11 up / 7 down for the 168 h block, 112 pairs / median
+84,474 / 73 up / 33 down / 6 unchanged for the 672 h block, and zero knob-on sidecar
rows over 2,899 and 13,114 rows.

Why these nodes read the live vault UNMARKED
--------------------------------------------
They read `~/obsidian`, which is what the `live_vault` marker exists for, and they carry
it nowhere — the same trade `tests/test_archived_skill_artifacts.py` documents in its
header: a marked assertion about vault content is deselected by the gate's own
`-m "not live_vault"`, so it is only ever *reported* and enforces nothing. Clause 4 asks
for a node that re-derives the published blocks, and a node the gate never runs does not
re-derive them. The coupling is paid for the same way there: every assertion is
structural or content-addressed (a sha256 re-derived from the bytes beside it, a column
list, a sign split, an `iterdump` census) rather than a quote of prose that will drift,
and a missing artifact SKIPS rather than reddens — so the one realistic rewrite of this
directory, a retention sweep that ages the file out, takes the module out of the run
instead of failing the next author for the previous writer's change.
"""

from __future__ import annotations

import inspect
import os
import re
import sqlite3
import statistics
from pathlib import Path

import pytest

#: The artifact, named by the demand that asked for it. The basename is what
#: #3631's owed entry pins, which is why the filename carries the window's close
#: (2026-10-31) and not the day the bytes were cut.
EXTRACT_BASENAME = "2026-10-31.3631-lloyd-data-usage.db"
MARKER_BASENAME = "2026-10-31.3631-lloyd-data-usage-witness.md"
ARTIFACT_PATH = "backlog/data/" + EXTRACT_BASENAME
MARKER_PATH = "backlog/data/" + MARKER_BASENAME

#: `scripts/automod/backlog.py`'s `WITNESS_MAX_BYTES`, restated because the vault
#: is not importable from here and a ceiling read out of the front matter would be
#: the artifact grading its own homework.
WITNESS_MAX_BYTES = 1_048_576

#: The two blocks `architecture/context-window.md` §What a rewrite costs publishes.
WINDOW_168 = ("2026-10-24T09:47:08", "2026-10-31T09:47:08")
WINDOW_672 = ("2026-10-03T09:47:08", "2026-10-31T09:47:08")

#: The eight columns clause 1 names, in order. `no_record` is the seventh thing
#: a reader can ask about and the only one that is not a token count: it is
#: `compaction is null`, which is what separates arm A_no_record from arm
#: D_ran_noop. The `compaction` document itself is NOT carried — the projection
#: keeps the three `json_extract` outputs and drops the blob.
COLUMNS = ("id", "ts", "session_id", "reprefill_tokens", "no_record",
           "turn_start_tokens_freed", "relief_tokens_freed", "microcompact_sidecar")

#: Arm B is a turn whose record freed something; arm D is a turn whose record ran
#: and freed nothing. A row with no record belongs to neither: it cannot be
#: evidence that a pass did nothing when it may never have been reached.
_PAIRING_SQL = """
WITH nxt AS (
  SELECT session_id,
         CASE WHEN no_record = 1 THEN 'A_no_record'
              WHEN turn_start_tokens_freed > 0 OR relief_tokens_freed > 0 THEN 'B_freeing'
              ELSE 'D_ran_noop' END AS arm,
         lead(reprefill_tokens) OVER (partition BY session_id ORDER BY id) AS nxt
    FROM usage_window
   WHERE ts >= :since AND ts < :until),
per AS (
  SELECT session_id,
         avg(CASE WHEN arm = 'B_freeing' THEN nxt END) AS b,
         avg(CASE WHEN arm = 'D_ran_noop' THEN nxt END) AS d
    FROM nxt
   WHERE nxt IS NOT NULL AND arm <> 'A_no_record'
   GROUP BY session_id)
SELECT b - d AS delta FROM per WHERE b IS NOT NULL AND d IS NOT NULL
"""


def _vault_root() -> Path:
    """`LLOYD_OBSIDIAN_VAULT` if set, else `~/obsidian` — the route the board-presence
    helper uses, duplicated rather than imported so this module has one less reason to
    care what else is in `tests/`."""
    raw = os.environ.get("LLOYD_OBSIDIAN_VAULT")
    return Path(raw).expanduser() if raw else Path.home() / "obsidian"


def _extract() -> Path:
    """The committed extract, or a skip.

    The skip is the whole point of clause 4's last five words: a checkout without the
    vault must not go red over an artifact that is a vault file, and it must not go
    *green* either by measuring nothing. Skipping is the honest third state.
    """
    path = _vault_root() / ARTIFACT_PATH
    if not path.is_file():
        pytest.skip(f"no pairing extract at {path}; it is a vault artifact, and the "
                    "figures below are its bytes rather than anything in this repo")
    return path


def _open(path: Path) -> sqlite3.Connection:
    """Read-only, by URI, always — a writable handle in the vault can leave a `-wal`
    beside the artifact, which is the sibling #2960 says must never be tracked."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _summarise(conn: sqlite3.Connection, since: str, until: str) -> dict:
    """The pairing over one window, as the page states it: pairs, median delta,
    sign split, and the two denominators the regime count sits over."""
    deltas = [row[0] for row in conn.execute(_PAIRING_SQL,
                                            {"since": since, "until": until})]
    rows, sidecar_on = conn.execute(
        "SELECT count(*), coalesce(sum(microcompact_sidecar <> 0), 0) "
        "FROM usage_window WHERE ts >= ? AND ts < ?", (since, until)).fetchone()
    return {"pairs": len(deltas),
            "median": statistics.median(deltas) if deltas else None,
            "up": sum(1 for d in deltas if d > 0),
            "down": sum(1 for d in deltas if d < 0),
            "unchanged": sum(1 for d in deltas if d == 0),
            "rows": rows,
            "sidecar_on": sidecar_on}


def _tracked(path: Path) -> bool:
    """Whether the vault's index holds this path. Git is asked, because a file that
    exists on disk is exactly what an un-published artifact looks like, and clause 1
    says *tracked*."""
    import subprocess

    return subprocess.run(["git", "-C", str(_vault_root()), "ls-files", "--error-unmatch",
                           str(path.relative_to(_vault_root()))],
                          capture_output=True).returncode == 0


# --------------------------------------------------------------------------- #
# Clause 1 — the shape, the window, the ceiling, the statement count
# --------------------------------------------------------------------------- #

def test_the_extract_is_tracked_holds_one_projection_and_fits_the_ceiling():
    """Clause 1: 851,968 B of bytes, 13,116 INSERTs, and nothing outside the window.

    Four independent bounds on the same artifact, because each one catches a different
    way for it to stop being a witness. Size is clause 1's `stat -c%s` reading against
    `WITNESS_MAX_BYTES`: 851,968 is 81.2% of 1,048,576, and the number is asserted
    exactly rather than as a floor — an artifact that grew past the ceiling would be
    refused by the witness route, and one that shrank has lost rows.

    The INSERT count is read through `iterdump`, which is the command #3631's own owed
    entry names as the proof: 13,114 `usage_window` rows plus 2 `pairing_bounds` rows.
    It counts statements, not tables, which is what makes it a row census at all.

    `compaction` must be ABSENT, not merely unread: the blob is the 11,833,334 B
    build #3832 clause 5 refuses, and a column nobody selects is still over the
    ceiling the moment someone selects it.
    """
    path = _extract()
    size = path.stat().st_size
    assert size <= WITNESS_MAX_BYTES, (
        f"the extract is {size:,} B against the {WITNESS_MAX_BYTES:,} B witness ceiling, "
        "so it is no longer something the witness route could have produced")

    conn = _open(path)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        assert tables == ["pairing_bounds", "usage_window"], (
            f"the extract carries {tables}; clause 1 allows the projection table and "
            "the bounds table and nothing else")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(usage_window)")]
        assert cols == list(COLUMNS), (
            f"usage_window is {cols}, not the eight columns clause 1 names — the "
            "pairing reads arms off three of them and the median off a fourth")
        assert "compaction" not in cols, (
            "the compaction blob is back in the projection. The three json_extract "
            "outputs are the extract; the document is the over-ceiling build (#3832 c5)")

        # iterdump is the route the demand names. 13,114 + 2 = 13,116.
        inserts = sum(1 for line in conn.iterdump() if line.startswith("INSERT"))
        assert inserts == 13_116, (
            f"{inserts:,} INSERT statements, not the 13,116 the demand's own command "
            "prints: 13,114 window rows and 2 bounds rows")

        outside = conn.execute(
            "SELECT count(*) FROM usage_window "
            "WHERE ts < ? OR ts >= ?", (*WINDOW_672,)).fetchone()[0]
        assert outside == 0, (
            f"{outside} rows sit outside {WINDOW_672[0]} <= ts < {WINDOW_672[1]}. The "
            "lower bound is also what excludes the 2026-09-25T01:32:11 -> "
            "2026-10-02T01:32:11 control window, so a row under it is clause 5 broken, "
            "not a rounding detail")
    finally:
        conn.close()

    assert _tracked(path), (
        f"{ARTIFACT_PATH} exists but is not tracked in the vault: an artifact on disk is "
        "not a witness, and #3631's owed entry skips on tracked basenames")


# --------------------------------------------------------------------------- #
# Clause 2 — the arms stay separable
# --------------------------------------------------------------------------- #

def test_no_record_keeps_the_no_record_arm_separate_from_the_noop_arm():
    """Clause 2: 2,577 + 9,119 + 1,418 = 13,114, and the split is the column's, not a reader's.

    The item justified keeping `no_record` by a pair-count inflation; that claim is
    false on this data and was measured so — collapsing `no_record` into arm D leaves
    both published blocks unchanged (18 / 36,460.75 and 112 / 84,474, both ways). What
    the column really buys is that the arms are *differently defined*, which a reader
    cannot recover after the fact once a projection has flattened it: 2,577 rows whose
    record says nothing folded into 9,119 rows whose record ran and freed nothing is a
    19.7% reassignment of meaning with no figure to show for it.

    The `no_record = 1` rows carrying no flags at all is the observable half of
    "`no_record` is `compaction is null`": a row with no document cannot have had a
    pass free anything, so any flag on such a row means the column is no longer that
    expression, whatever it is instead.
    """
    conn = _open(_extract())
    try:
        arms = conn.execute(
            "SELECT sum(no_record = 1), "
            "       sum(no_record = 0 AND turn_start_tokens_freed = 0 "
            "           AND relief_tokens_freed = 0), "
            "       sum(no_record = 0 AND (turn_start_tokens_freed > 0 "
            "           OR relief_tokens_freed > 0)), "
            "       count(*) FROM usage_window").fetchone()
        assert tuple(arms) == (2577, 9119, 1418, 13114), (
            f"the window splits into {arms}, not (2577 A_no_record, 9119 D_ran_noop, "
            "1418 freeing, 13114 total). Those three arms are what the pairing assigns "
            "rows to; a fourth bucket or a moved boundary changes what the median means")
        assert arms[0] + arms[1] + arms[2] == arms[3], (
            f"the three arms sum to {arms[0] + arms[1] + arms[2]}, not {arms[3]}: some "
            "row is in no arm, which the sign split cannot be read over")

        flags = conn.execute(
            "SELECT DISTINCT no_record FROM usage_window ORDER BY no_record")
        assert [r[0] for r in flags] == [0, 1], (
            "no_record stopped being a 0/1 flag, so `compaction is null` is no longer "
            "what it encodes")
        impossible = conn.execute(
            "SELECT count(*) FROM usage_window WHERE no_record = 1 AND "
            "(turn_start_tokens_freed <> 0 OR relief_tokens_freed <> 0 "
            " OR microcompact_sidecar <> 0)").fetchone()[0]
        assert impossible == 0, (
            f"{impossible} rows with no record carry a nonzero flag. A null document "
            "coalesces every json_extract to 0, so this is the derivation drifting")
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Clause 3 — the file says what it prices
# --------------------------------------------------------------------------- #

def test_pairing_bounds_names_exactly_the_two_published_blocks():
    """Clause 3: two rows, the two priced windows, each note carrying its own figures.

    The table exists so `min(ts)`/`max(ts)` is not mistaken for a list of priced
    windows. `2026-10-02T01:32:11 -> 2026-10-09T01:32:11` is a block the page publishes
    whose first ~32 h falls below this extract's lower bound, so over these bytes it
    recomputes to 26 pairs / median 155,812.5 rather than 29 / +164,725 — a truncated
    window, not a contradicted figure. A node that looped over every block on the page
    would report that as a discrepancy; a node that loops over `pairing_bounds` cannot.

    The notes have to carry the numbers themselves. Bounds alone cannot tell a reader
    whether the file prices 18 pairs or 112, and the two blocks are nested.
    """
    conn = _open(_extract())
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(pairing_bounds)")]
        assert cols == ["since", "until", "note"], (
            f"pairing_bounds is {cols}, not (since, until, note)")
        rows = conn.execute("SELECT since, until, note FROM pairing_bounds "
                            "ORDER BY since").fetchall()
        assert len(rows) == 2, (
            f"{len(rows)} priced windows, not exactly two. The 672 h block and the "
            "168 h block inside it are what the file prices; anything else here is a "
            "claim these bytes may not support")
        assert (rows[0][0], rows[0][1]) == WINDOW_672, (
            f"the wider row is {rows[0][:2]}, not the 672 h block {WINDOW_672}")
        assert (rows[1][0], rows[1][1]) == WINDOW_168, (
            f"the narrower row is {rows[1][:2]}, not the 168 h block {WINDOW_168}")
        for (since, until, note), pairs, median in (
                (rows[0], 112, "84,474"), (rows[1], 18, "36,46")):
            assert f"{pairs} pairs" in note, (
                f"the note for {since} -> {until} does not name its {pairs} pairs: "
                f"{note!r}")
            assert median in note, (
                f"the note for {since} -> {until} does not name its median delta "
                f"(published +{median}): {note!r}")

        # And the bounds table must not lie: each row's own window has to be present.
        for since, until in (WINDOW_168, WINDOW_672):
            inside = conn.execute(
                "SELECT count(*) FROM usage_window WHERE ts >= ? AND ts < ?",
                (since, until)).fetchone()[0]
            assert inside > 0, f"the priced window {since} -> {until} holds no rows"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Clause 4 — the figures, recomputed from the committed bytes and nothing else
# --------------------------------------------------------------------------- #

def test_the_two_published_blocks_recompute_from_the_committed_bytes():
    """Clause 4: 18 / +36,460.75 / 11 up / 7 down, and 112 / +84,474 / 73 / 33 / 6.

    These are the two blocks `architecture/context-window.md` §What a rewrite costs
    publishes, and until this node existed the sentence beside them — "no node
    recomputes them" — was the only enforcement they had. Both are asserted to the
    digit, including the sign split, because a median alone survives a window that
    quietly lost a session: 18 pairs with 11 up / 7 down is the published block, and
    18 pairs with 12 up / 6 down is a different measurement wearing its clothes.

    The median convention is `statistics.median`, the even-n mean, and it is
    load-bearing rather than house style: nearest-rank over these same bytes returns
    44,512.5 and 86,473, neither of which is a figure on the page. The published
    `+36,461` is 36,460.75 rounded to the whole token; `+84,474` is exact.

    The regime count is the third block of the page's own claim — 0 knob-on sidecar
    rows of 2,899 in the 168 h block and of 13,114 in the 672 h block, both windows
    closing before #3641 flipped `RunOptions.microcompact_reduction_sidecar` to True on
    2026-10-31. It is asserted with its denominators, since a zero over no rows proves
    nothing at all.
    """
    conn = _open(_extract())
    try:
        for (since, until), want in ((WINDOW_168, dict(pairs=18, median=36460.75,
                                                       up=11, down=7, unchanged=0,
                                                       rows=2899)),
                                     (WINDOW_672, dict(pairs=112, median=84474.0,
                                                       up=73, down=33, unchanged=6,
                                                       rows=13114))):
            got = _summarise(conn, since, until)
            assert got["pairs"] == want["pairs"], (
                f"{since} -> {until} pairs {got['pairs']} sessions, not "
                f"{want['pairs']}: the page's block is a dated reading and this is what "
                "the committed bytes still support")
            assert got["median"] == pytest.approx(want["median"]), (
                f"{since} -> {until} has median delta {got['median']}, not "
                f"{want['median']}. The page publishes "
                f"{want['median']:+,.0f}")
            assert (got["up"], got["down"], got["unchanged"]) == (
                    want["up"], want["down"], want["unchanged"]), (
                f"{since} -> {until} splits {got['up']} up / {got['down']} down / "
                f"{got['unchanged']} unchanged, not "
                f"{want['up']} / {want['down']} / {want['unchanged']}")
            assert got["rows"] == want["rows"], (
                f"the window holds {got['rows']} rows, not {want['rows']}: that is the "
                "denominator the regime zero is quoted over")
            assert got["sidecar_on"] == 0, (
                f"{got['sidecar_on']} rows in {since} -> {until} carry "
                "turn_start.microcompact_sidecar. Both blocks are published as single-"
                "regime knob-off, so a nonzero count means the window is not the one "
                "that closed before #3641 flipped the default on 2026-10-31")
    finally:
        conn.close()


def test_the_recomputation_never_opens_the_live_store(monkeypatch, tmp_path):
    """Clause 4's other half: the figures above come from the artifact, not the database.

    The seam this pins is the one a doc-claim node always has — the honest way to keep
    a test off a 50 MB production store is to make reaching it fatal rather than to
    assert afterwards that nothing happened. `sqlite3.connect` is patched to refuse any
    target under the live data root, so the recomputation either runs on the extract or
    dies here naming the path it tried to open.

    The refusal is a guard, not a skip: if the figures came back at all while the live
    root was unreachable, they came from the committed bytes by construction. A
    connection to a path outside both is also refused, which is what makes the patch a
    boundary rather than a filter.
    """
    live_root = str(Path.home() / "lloyd-data")
    vault = str(_vault_root())
    opened: list[str] = []
    real_connect = sqlite3.connect

    def guarded(database, *args, **kwargs):
        target = str(database).split("?")[0].removeprefix("file:")
        opened.append(str(database))
        if target.startswith(live_root):
            raise AssertionError(
                f"the pairing tried to open {database}: the live store is 50,212,864 B "
                "against a 1,048,576 B witness ceiling, which is the whole reason the "
                "extract exists")
        if not target.startswith(vault) and not target.startswith(str(tmp_path)):
            raise AssertionError(
                f"the pairing opened {database}, which is neither the vault artifact nor "
                "a scratch copy: this node's figures are supposed to come from committed "
                "bytes")
        if not str(database).startswith("file:") or "mode=ro" not in str(database):
            raise AssertionError(
                f"{database} was opened without a read-only URI, which can leave a "
                "-wal beside the artifact — the sibling #2960 keeps out of the index")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded)
    path = _extract()
    got = _summarise(_open(path), *WINDOW_168)

    assert got["pairs"] == 18 and got["median"] == pytest.approx(36460.75), (
        f"with the live store refused by construction the 168 h block came back "
        f"{got['pairs']} pairs / {got['median']}, not 18 / 36460.75")
    assert opened, "the guard recorded no connection at all, so nothing was proven"
    assert all(live_root not in uri for uri in opened), (
        f"a live-root path reached the guard after all: {opened}")


def _paired_sessions(conn: sqlite3.Connection, since: str, until: str) -> list[str]:
    """The sessions that form a pair in one window — the rows the median is taken over."""
    return [row[0] for row in conn.execute(
        _PAIRING_SQL.replace("SELECT b - d AS delta FROM per",
                             "SELECT per.session_id FROM per"),
        {"since": since, "until": until})]


def test_the_pairing_reads_the_arm_flags_because_bending_them_bends_the_block(tmp_path):
    """The nodes above can fail, which is the half a passing test never shows.

    Seeded on a scratch copy, so the vault artifact is never the thing being bent. Both
    arms below are provable from the definition of the pairing rather than from a
    measurement that happens to move, because a median over 18 sessions makes a careless
    tamper node a lie: flipping the lowest-id arm-D row into a freeing row — the
    smallest edit that changes arm membership — was measured to leave the whole block
    exactly at 18 pairs / median 36,460.75 / 11 up / 7 down. A tamper chosen by luck of
    the data would "pass" while proving nothing.

    So each arm asserts what follows by construction:

    * Empty arm D and the pairing has no pairs at all, because a pair is one session
      holding both arms. This is the proof that the flags define the arms — a node that
      read the published numbers out of a constant would survive everything except
      this.
    * Exclude one paired session's rows and the pair count drops by exactly one, and
      that session is gone from the list. `lead()` is partitioned by session, so no
      other session can move; a 17 where 18 is published is the arithmetic of removing
      one term, not a rounding artefact.
    """
    src = _extract()
    copy = tmp_path / EXTRACT_BASENAME
    copy.write_bytes(src.read_bytes())

    # Arm 1: no arm D, so no pair can exist.
    conn = sqlite3.connect(copy)
    try:
        moved = conn.execute(
            "UPDATE usage_window SET relief_tokens_freed = 50000 "
            "WHERE no_record = 0 AND turn_start_tokens_freed = 0 "
            "  AND relief_tokens_freed = 0").rowcount
        assert moved > 0, "the seeded edit matched nothing, so it proves nothing"
        conn.commit()
        no_d = _summarise(conn, *WINDOW_168)
    finally:
        conn.close()
    assert no_d["pairs"] == 0, (
        f"with every arm-D row turned into a freeing row the block still reports "
        f"{no_d['pairs']} pairs / median {no_d['median']}: a pair is one session holding "
        "BOTH arms, so any nonzero count here means the arms are not read from these "
        "flags")

    # Arm 2: remove exactly one paired session from the population.
    conn = sqlite3.connect(copy)
    try:
        conn.execute("UPDATE usage_window SET relief_tokens_freed = 0 "
                     "WHERE relief_tokens_freed = 50000")
        conn.commit()
        before = _paired_sessions(conn, *WINDOW_168)
        assert len(before) == 18, (
            f"the scratch copy pairs {len(before)} sessions, not the 18 the committed "
            "bytes pair — the copy is not the artifact")
        chosen = sorted(before)[0]
        conn.execute("UPDATE usage_window SET no_record = 1 WHERE session_id = ?",
                     (chosen,))
        conn.commit()
        after = _summarise(conn, *WINDOW_168)
        remaining = _paired_sessions(conn, *WINDOW_168)
    finally:
        conn.close()
    assert after["pairs"] == 17, (
        f"excluding one paired session left {after['pairs']} pairs, not 17. `lead()` is "
        "partitioned by session_id, so removing one session's arm rows removes exactly "
        "its own term from the median")
    assert chosen not in remaining, (
        f"{chosen} was excluded from both arms and still pairs")


# --------------------------------------------------------------------------- #
# Clause 5 — the sibling marker  (item body clause 6)
# --------------------------------------------------------------------------- #

def test_the_sibling_marker_names_the_keys_the_reader_and_what_the_bytes_cannot_answer():
    """Clause 5 (item body clause 6): the marker says how the bytes were cut, and what they cannot answer.

    Front matter first, because it is the machine-read half: `sha256` and `byte_size`
    are re-derived from the artifact here rather than believed, so a marker describing
    a file it no longer describes goes red instead of quietly becoming a fixture.
    `witness_max_bytes: 1048576` is the ceiling named as a number so a reader can see
    the margin without opening the emitter.

    Then the three things a later reader needs and cannot guess: the three
    `json_extract` keys the flags came from, `usage_store._sidecar_flags` as the regime
    reader (so nobody re-derives the regime count through a different key and calls the
    gap a discrepancy), and the two things these bytes do not price — the
    2026-09-25T01:32:11 -> 2026-10-02T01:32:11 control window and the
    2026-10-02T01:32:11 -> 2026-10-09T01:32:11 block, whose first ~32 h sits below the
    extract's own lower bound.

    And no `-wal`/`-shm` sibling may be tracked (#2960): those are the shape a writable
    handle leaves behind, and they are never part of an artifact.
    """
    import yaml

    path = _extract()
    marker = _vault_root() / MARKER_PATH
    assert marker.is_file(), (
        f"no marker at {MARKER_PATH}: the bytes are unexplained, and the clause asks "
        "that a reader be able to tell a projection from the store")
    assert _tracked(marker), f"{MARKER_PATH} is on disk but not tracked in the vault"

    text = marker.read_text(encoding="utf-8")
    front = text.split("---", 2)
    assert len(front) == 3, f"{MARKER_PATH} has no front matter to read"
    fm = yaml.safe_load(front[1])

    assert fm["artifact"] == ARTIFACT_PATH, (
        f"the marker describes {fm['artifact']!r}, not {ARTIFACT_PATH!r}")
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert fm["sha256"] == digest, (
        f"the marker's sha256 is {fm['sha256']!r} and the artifact hashes to {digest!r}: "
        "the marker is describing different bytes than the ones beside it")
    assert int(fm["byte_size"]) == path.stat().st_size, (
        f"the marker's byte_size is {fm['byte_size']} and the file is "
        f"{path.stat().st_size:,} B")
    assert int(fm["witness_max_bytes"]) == WITNESS_MAX_BYTES, (
        f"witness_max_bytes is {fm['witness_max_bytes']!r}, not {WITNESS_MAX_BYTES}")
    assert str(fm["extracted_at_utc"]).startswith("2026-"), (
        f"extracted_at_utc is {fm['extracted_at_utc']!r}; the cut date is what tells a "
        "reader how stale the projection is")

    flat = " ".join(text.split())
    for key, why in (("$.turn_start.tokens_freed", "the arm-B half"),
                     ("$.relief_tokens_freed", "the arm-B half's other key"),
                     ("$.turn_start.microcompact_sidecar", "the regime flag")):
        assert key in flat, (
            f"the marker never names {key} ({why}), so a reader rebuilding the "
            "projection has to guess which json_extract output each column is")
    assert "usage_store._sidecar_flags" in flat, (
        "the marker does not name usage_store._sidecar_flags as the regime reader, which "
        "is what keeps a later count from being taken through a different key")
    for window, why in (("2026-09-25T01:32:11", "the control window these bytes exclude"),
                        ("2026-10-02T01:32:11", "the published block that starts ~32 h "
                                                "below this file's lower bound")):
        assert window in flat, (
            f"the marker does not say it leaves out {window} ({why}). A proof node that "
            "loops over every block on the page has to be told, or it reports a "
            "truncated window as a wrong figure")
    assert "_sidecar_flags" in flat and "statistics.median" in flat, (
        "the marker does not name the median convention, and nearest-rank over these "
        "same bytes returns 44,512.5 and 86,473 — a different statistic, not a drift")

    data_dir = path.parent
    for sibling in ("2026-10-31.3631-lloyd-data-usage.db-wal",
                    "2026-10-31.3631-lloyd-data-usage.db-shm"):
        assert not (data_dir / sibling).exists() or not _tracked(data_dir / sibling), (
            f"{sibling} is in the vault index: a write-ahead sibling is never part of a "
            "witness (#2960)")


# --------------------------------------------------------------------------- #
# #4011 — the page that publishes these blocks has to say where the bytes are,
# which module re-derives them, and what these bytes cannot speak to
# --------------------------------------------------------------------------- #

#: The doc whose prose this module makes a claim about, and the section publishing
#: the two blocks above. Read from THIS checkout, never from the vault: a citation
#: committed in a page is a statement about this repo, so this node has no vault
#: dependency to skip over — see `_rewrite_cost_section`.
DOC_PATH = Path(__file__).resolve().parents[1] / "architecture" / "context-window.md"
REWRITE_COST_HEADING = "What a rewrite costs, priced per mechanism"

#: The one spelling of the artifact's location that survives a move: `~/obsidian/`
#: names a tree, where a bare `backlog/data/…` resolves against whatever directory a
#: reader is standing in, and a literal `/home/<user>/…` commits a username into a doc.
#: The coverage suite's own citation resolver would accept all three — its roots
#: include the vault — so the only thing that keeps one spelling is this assertion.
VAULT_PREFIX = "~/obsidian/"

#: How the page names the module that re-derives its figures, built from this file's
#: own name so a rename moves the page instead of silently retiring the pin.
MODULE_CITATION = "`tests/" + Path(__file__).name + "`"

#: The live-store half of the ban, as the page words it. The refusal `conftest.py`
#: raises is about the data tree, and the page used to blame that refusal for a
#: blanket "nothing recomputes these" that this module had already contradicted.
LIVE_STORE_BAN = "recomputing them from the live store"

#: The two figures a single-rank reading of the truncated 2026-10-02 block returns
#: over these bytes. They belong in this comment and in no page: the extract starts
#: after that block starts, so it is a truncated window here and not a refutation of
#: a figure this module has no business republishing.
TRUNCATED_BLOCK_FIGURES = ("155,812", "26 pairs")

#: What a node about the page must never reach for. Built by concatenation because the
#: node reads this module's own source: a marker written literally IS the thing it bans,
#: and the check would fail on its own text — the same reason the coverage file builds
#: its `PAIRING_RECOMPUTE_MARKERS` this way.
DOC_TEXT_ONLY_MARKERS = ("_extract" + "(",
                         "_vault_root" + "(",
                         "pytest." + "skip",
                         "sqlite3." + "connect(")


def _rewrite_cost_section() -> str:
    """The body of §What a rewrite costs, up to the next `## ` — prose and nothing else.

    Deliberately independent of the artifact reader above, which skips when the vault
    has no extract. This node's subject is a paragraph committed in this checkout, and
    a retention sweep that ages the extract out is precisely the window in which the
    page's sentence about those bytes stops being checkable by hand — so the node keeps
    running, and keeps failing on drift, with the artifact gone.
    """
    text = DOC_PATH.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(REWRITE_COST_HEADING)}.*?$(.*?)(?=^## |\Z)",
                  text, re.MULTILINE | re.DOTALL)
    assert m, (
        f"architecture/context-window.md has no '## {REWRITE_COST_HEADING}' section, so "
        "the prose pinned below has moved somewhere nothing here is looking")
    return m.group(1)


def test_the_page_cites_the_committed_extract_and_names_this_module_as_the_recomputer():
    """#4011: §What a rewrite costs cites the projection beside its two blocks, names
    this module as what re-derives them, states the median convention, and leaves the
    truncated block outside what these bytes can say.

    The sentence this node exists for read "no node recomputes them", and
    `test_the_two_published_blocks_recompute_from_the_committed_bytes` refuted it the
    day it landed: both published blocks ARE recomputed, from the artifact this module
    opens. What is genuinely forbidden is reaching for the live store — a gate worktree
    has no copy of it, and a node that queried one would either fail for a reason that
    is not the page's or certify figures over an empty directory. The page now says
    which of those two it means, and prose is the only enforcement this section has, so
    a node has to keep it saying so.

    The structural assertion runs first on purpose. Turning this node back into one that
    depends on the vault would make it skip for the whole period a sweep has aged the
    extract out, and a skipped node is the exact shape that let the sentence rot for
    four days: green, watching nothing.
    """
    own = (inspect.getsource(_rewrite_cost_section)
           + inspect.getsource(
               test_the_page_cites_the_committed_extract_and_names_this_module_as_the_recomputer))
    for marker in DOC_TEXT_ONLY_MARKERS:
        assert marker not in own, (
            f"this node now reaches for {marker!r}, which would have it skip or open a "
            "database instead of reading the page it is about")

    # And the markers have to be capable of firing at all: the artifact reader two
    # screens up is the node that DOES reach for them, on purpose, so if a rename ever
    # broke the concatenation the absence above would go quietly vacuous.
    reaching = [m for m in DOC_TEXT_ONLY_MARKERS if m in inspect.getsource(_extract)]
    assert reaching, (
        "none of the markers occurs in the module's own artifact reader, so this node is "
        "asserting an absence its pattern could never have detected")

    flat = " ".join(_rewrite_cost_section().split())

    # Clause 1: the projection and its marker, each as a code span carrying the prefix.
    for span in (f"`{VAULT_PREFIX}{ARTIFACT_PATH}`", f"`{VAULT_PREFIX}{MARKER_PATH}`"):
        assert span in flat, (
            f"§What a rewrite costs no longer cites {span} as a code span, so a reader "
            "of the two blocks has nothing to check them against and no idea the bytes "
            "exist")
    # Every other mention of these two names has to carry the prefix too. Scoped to the
    # basenames rather than to `backlog/data/` at large because the section cites a
    # different vault artifact as well, and how THAT one is spelled is not this node's
    # claim to make.
    for basename in (EXTRACT_BASENAME, MARKER_BASENAME):
        strays = [flat[max(0, i - 24):i + len(basename)]
                  for i in (m.start() for m in re.finditer(re.escape(basename), flat))
                  if not flat[:i].endswith(f"{VAULT_PREFIX}backlog/data/")]
        assert not strays, (
            f"the section mentions {basename} in a spelling that does not name the vault: "
            f"{strays}. A bare `backlog/data/…` resolves against the reader's own "
            "directory and a literal `/home/…` form bakes this machine's username into a "
            "doc; the `~/obsidian/` prefix is the spelling that means the same thing "
            "everywhere")

    # Clause 2: the refuted claim is gone, the recomputing module is named, and the ban
    # is worded against the live store rather than against recomputation as such.
    assert "no node recomputes them" not in flat and "no node" not in flat, (
        "the section is back to asserting that nothing recomputes these figures. This "
        "module does, from committed vault bytes — see the node that re-derives 18 pairs "
        "/ median 36,460.75 and 112 pairs / median 84,474.0")
    assert MODULE_CITATION in flat, (
        f"the section does not name {MODULE_CITATION} as the module that recomputes its "
        "two blocks, so its reader has no route from the figures to the bytes")
    assert LIVE_STORE_BAN in flat, (
        f"the section no longer words its ban as {LIVE_STORE_BAN!r}; the blanket form is "
        "the one that became false, and the live store is the half that is actually "
        "unreachable from a gate worktree")

    # Clause 3: the statistic is named, because the published medians are the even-n
    # mean and a single-rank reading of the same bytes is a different measurement.
    assert "`statistics.median`" in flat and "even-n" in flat, (
        "the section stopped stating that its medians are `statistics.median` — the "
        "even-n mean. Both published medians come from an even number of deltas, so an "
        "unstated convention is a figure a later reader cannot reproduce")

    # Clause 4: what these bytes price, and the one block they cannot price.
    assert "`pairing_bounds`" in flat and "price exactly the two blocks" in flat, (
        "the section no longer scopes these bytes to the two windows `pairing_bounds` "
        "names, so the rolling rows below the blocks read as re-derivable from them too")
    assert "`2026-10-02T01:32:11`" in flat and "`2026-10-09T01:32:11`" in flat \
            and "lower bound" in flat, (
        "the section no longer says that the 2026-10-02→2026-10-09 block opens below "
        "this extract's lower bound. It is the one published block the projection "
        "truncates, and 'these bytes disagree' is the wrong reading of a window whose "
        "first ~32 h is simply absent")
    for figure in TRUNCATED_BLOCK_FIGURES:
        assert figure not in flat, (
            f"the section now publishes {figure!r} for the truncated block. A partial "
            "window's median is not a correction of the published one, and quoting it "
            "here would put a figure on the page that no measurement of that window "
            "produced")
