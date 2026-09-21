"""`architecture/djev.md`'s record of the #1324 shadow-log leak stays true.

The behaviour that closes the leak is pinned in
`tests/test_djev_shadow_isolation.py`. This file pins the *other* half of the
item: what the doc is allowed to say now that the fix exists, and the numbers
inside that prose.

It matters because the doc is where the item came from. §11 carried "Test runs
write to production's shadow log" as an open gap and named #1324, so any
architecture review that read §11 after the fix would have re-filed the item
from the doc's own text — an open-gap list is a to-do list, and a stale entry in
one is worse than no entry, because it looks like a decision. The same section
now carries counts (36 of 44 `dedupe` rows, 27 of 32 `rerank`, 63 fixture rows
left in the log) that name a fixture by its title; if that title is renamed in
`tests/test_backlog_dedupe.py` the counts silently stop meaning anything, so
those titles are pinned against the module that defines them.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "djev.md"

#: The fixture title in `tests/test_backlog_dedupe.py:58` whose 32 rows the doc
#: counts, and the one in `tests/test_backlog_spawn_loop.py:792` that accounts
#: for 4 more. Both are asserted against those modules below, so a rename that
#: would make the doc's counts unfalsifiable fails a test.
FIXTURE_DEDUPE = "http_fetch error body says only the status code on a quarter of calls"
FIXTURE_SPAWN = "graph_refresh is advertised but never called by any tool"

CLOSED_HEADING = "### Closed 2026-09-21 — test runs wrote to production's shadow log (#1324)"


def _section(text: str, heading: str) -> str:
    """From `heading` to the next `## ` line — one section, sliced out."""
    i = text.index(heading)
    j = text.find("\n## ", i + len(heading))
    return text[i: j if j != -1 else len(text)]


def _closed_record(text: str) -> str:
    """The Closed subsection of §11. Asserts its own absence, so a doc that
    dropped the record fails here instead of at some `[1]` index."""
    body = _section(text, "## 11. Known gaps")
    assert CLOSED_HEADING in body, (
        "§11 has no closed record for the shadow-log leak, which means its open-gap "
        "list is still describing #1324 as live — the exact state that re-files it")
    return body.split(CLOSED_HEADING, 1)[1]


def test_the_open_gaps_no_longer_describe_the_shadow_log_leak():
    """Clause 3, first half: `architecture/djev.md` §11 stops listing "Test runs
    write to production's shadow log" as an open gap, and stops pointing at the
    item that closed it.
    """
    open_part = _section(DOC.read_text(), "## 11. Known gaps").split(CLOSED_HEADING, 1)[0]
    assert "Test runs write to production's shadow log" not in open_part, (
        "§11 still lists the leak as an open gap")
    assert "#1324" not in open_part, "the open gaps still point at the item that closed it"
    assert "fixture" not in open_part.lower(), (
        "the open gaps still carry the fixture-row complaint this item closed")


def test_the_closed_record_carries_the_mechanism_the_check_and_the_residue():
    """Clause 3, second half: the replacement has to be usable, not merely
    present. Each needle is something a later reader needs and cannot re-derive:
    which fixture the counts refer to, why the paths are redirected instead of
    muted (reverting to `LLOYD_DJEV_SHADOW=0` would make
    `test_djev_rerank_arm.py::test_the_eval_mutes_the_shadow_recorder` pass with
    the `eval/run_eval.py` line under it deleted), why the session fixture never
    restores (a restore at session end is the drain window), the command that
    shows the leak is gone, and the rows still in the file poisoning
    `eval/djev/replay.py --floors`.
    """
    closed = _closed_record(DOC.read_text())
    for needle in ("`tests/conftest.py::_isolate_djev_shadow`",   # the fix, by name
                   "does not restore them",                       # the mechanism that matters
                   FIXTURE_DEDUPE,                         # the fixture the counts name
                   "Paths are redirected rather than",            # the decision…
                   "`LLOYD_DJEV_SHADOW=0` set",                   # …and its reason
                   "test_the_eval_mutes_the_shadow_recorder",     # the test a mute would vacuate
                   "$FH/.local/state/lloyd-djev/shadow.jsonl",    # the check
                   "**does not exist**",                          # its expected result
                   "**What is left, and it is a person's call.**"):  # the residue, still owed
        assert needle in closed, f"the closed record no longer carries {needle!r}"


def test_the_closed_record_still_names_the_two_things_reading_the_log():
    """Closing the leak does not clean the corpus, and the doc must not read as
    if it had. Two writers survive this change: the fixture rows already in the
    file, and the automod gate's tests rung, which runs pytest with `HOME` set to
    the real home — `scripts/automod/gate.py::_child_env` redirects the automod
    and guardian state dirs and sets no djev path (`grep -n DJEV
    scripts/automod/gate.py` → 0 hits). Under the gate a repointed `SHADOW_LOG`
    proves nothing about `Path.home()`, because `Path.home()` is the real home
    there; the property that holds is the line count of the real file across the
    run. Either fact goes missing and the next reader of §9.3 step 5 gets a
    fixture-dominated floor or a false clean bill.
    """
    closed = _closed_record(DOC.read_text())
    assert "63" in closed, "the closed record no longer says how many fixture rows are still in the log"
    assert "replay.py --floors" in closed, "the closed record no longer names what the residue poisons"
    assert "_child_env" in closed, "the closed record no longer names the gate as the un-repointed runner"


def test_the_calibration_procedure_points_at_the_quarantine_first():
    """§9.3 step 5 is where a practitioner goes looking, not §11. It now says the
    floor comes *after* quarantining the pre-fix rows, so the calibration
    procedure cannot be followed in the order that produces a fixture-dominated
    floor.
    """
    doc = DOC.read_text()
    step = doc[doc.index("### 9.3 Adding a seam"):doc.index("### 9.4 Turning a schema")]
    assert step.count("#1324") == 1, "§9.3 step 5 no longer points at the quarantine the floor needs first"
    assert "fixture" in step, "§9.3 step 5 no longer says why quarantine precedes --floors"


def test_the_fixture_titles_the_doc_counts_still_exist():
    """The doc's counts are only checkable while the titles they name are still
    the titles the fixtures write. `tests/test_backlog_dedupe.py` and
    `tests/test_backlog_spawn_loop.py` write the first string into `meta.name`
    and the second into the backlog `name` of every row their dedupe seam
    shadows, which is how 36 of the log's 44 `dedupe` rows are attributable.
    Rename either and the doc's numbers become prose nobody can re-measure.
    """
    doc = DOC.read_text()
    closed = _closed_record(doc)
    assert FIXTURE_DEDUPE in closed, "the closed record no longer names the fixture it counts"
    assert FIXTURE_DEDUPE in (ROOT / "tests" / "test_backlog_dedupe.py").read_text(), (
        f"{FIXTURE_DEDUPE!r} is no longer the fixture title in test_backlog_dedupe.py, "
        "so the row counts in the closed record are no longer checkable")
    assert FIXTURE_SPAWN in (ROOT / "tests" / "test_backlog_spawn_loop.py").read_text(), (
        f"{FIXTURE_SPAWN!r} is no longer the fixture title in test_backlog_spawn_loop.py")
