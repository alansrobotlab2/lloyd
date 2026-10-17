"""What a task-file CREATOR is told about the two membership tables (#2895).

`architecture/autonomy-jobs.md` lists the fleet twice — the functions table (`## The
functions`, `| Function | Jobs | What it is for |`) and the write-authority tier table
(`| Tier | Jobs | |`). Four rounds have been spent on those two rosters disagreeing with
the directory: #1102 for #86, #2630/#2631 for three paper digests, #2813 for #99, #2881 for
#100 and #101. Every one of them was caught by the same thing — a nightly suite node — and
the reason is structural, recorded at `tests/test_autonomy_jobs_doc_claims.py:3349` and
`:3377`: both reconciling nodes carry `@live_vault`, which the automod gate deselects
(`scripts/automod/gate.py:641`, `TESTS_MARK_EXPR = "not live_vault and not fault_injection"`).
So the first thing in any day to read both sides was allowed to be a red suite.

This file moves the check to the moment the file is written, on fixtures, and it runs at the
gate. Nothing here reads `~/obsidian`: a filing test that graded the live fleet would be
written by the fleet — on 2026-10-17 the live tree measured `live 43` and `([], [], [])` for
both tables precisely because #2881 had hand-fixed the doc that morning, so "green" here
would have meant "nobody has filed anything today", and a guard nobody can see fail is the
same guard that has already failed four times. The seeded directions below are clause 4: a
directory holding `41-*.md` that neither table names, and a doc naming `#41` with no file
behind it.

The bound this file is most careful to prove is the one that is easy to over-build: the
report NEVER refuses a write. A round that turned doc drift into a blocking error would have
traded a doc lag nobody can act on for a tool that cannot file tasks — and the doc lag is the
smaller failure, which is why the same item that asked for the report also says "never
refuses". So several nodes below assert on the written file as much as on the report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.autonomy import jobs_membership as jm  # noqa: E402

#: The only ids either fixture doc names. An id outside this pair is the seeded drift.
LISTED = 40
UNLISTED = 41


def fixture_doc(*, functions: tuple[int, ...] = (LISTED,),
                tiers: tuple[int, ...] = (LISTED,)) -> str:
    """A two-table doc in the shape the real one has, naming exactly these ids.

    Which cell the ids sit in is the detail that matters, and it is the same in both tables:
    the SECOND. A real functions row is `| [Ingest](#ingest-68-30-53) | #68 #30 #53 | turn
    outside si… |` and a real tier row is `| Writes durable state unattended | #30 #53 · #38
    #42 … |`, so cell 1 is the group's name and cell 2 is the roster. `jm.table_rows` reads
    `cells[1]`, and a fixture that spelled the ids into cell 1 parses as a table naming
    nobody — which is not a harmless typo, because "names nobody" and "is in sync" look
    identical to any node that only checks one direction.

    The headers are `jm`'s own constants rather than typed strings for the same reason: a
    header the parser cannot locate raises `jm.UnreadableDoc` loudly, where a table that
    parses empty would have the whole file asserting against nothing.
    """
    fn_rows = "".join(f"| Group {i} | #{i} | what it does |\n" for i in functions)
    tier_rows = "".join(f"| Durable state | #{i} | note |\n" for i in tiers)
    return (f"## The functions\n\n{jm.FUNCTIONS_HEADER}\n|---|---|---|\n{fn_rows}\n"
            f"\n## Write authority\n\n{jm.TIERS_HEADER}\n|---|---|---|\n{tier_rows}\n")


def fixture_dir(tmp_path: Path, *ids: int) -> Path:
    """A task directory holding one `NN-slug.md` per id — no frontmatter needed.

    Membership is a FILENAME fact in this design, exactly as it is in `validate_tasks.py`
    and `app/autonomy.py`: an id the scheduler can see is a file whose name carries it.
    """
    d = tmp_path / "autonomy"
    d.mkdir(exist_ok=True)
    for i in ids:
        (d / f"{i}-seeded-job.md").write_text("---\ntype: autonomy\n---\n\nb\n",
                                              encoding="utf-8")
    return d


@pytest.fixture
def journal(tmp_path) -> Path:
    """The drift journal, redirected off the production data root.

    `jm.report_at_filing` takes the path as an argument precisely so no test can append to
    `~/lloyd-data/_pipeline/arch-review/membership-drift.jsonl`, which is a real record of
    real filings that the owed-check job reads a month from now.
    """
    return tmp_path / "arch-review" / "membership-drift.jsonl"


# ── Clause 4: the two directions, seeded ───────────────────────────────────────


def test_a_task_file_neither_table_names_is_unlisted_and_is_named(tmp_path):
    """`41-seeded-job.md` exists and neither table names #41: the drift names THAT id.

    This is the #2881 shape — #100 and #101 filed on 2026-10-16 with neither table naming
    them — reduced to one file and one doc. Naming the id is the whole value: a warning that
    says "membership drift" without `#41` sends a reader to diff the whole fleet.
    """
    d = fixture_dir(tmp_path, LISTED, UNLISTED)
    ghosts, unlisted, dupes = jm.membership_drift(fixture_doc(), d)
    assert unlisted == ["#41"], f"the seeded unlisted id was not the only finding: {unlisted}"
    assert ghosts == [] and dupes == [], f"the clean half of the fixture reported: {ghosts} {dupes}"


def test_a_table_that_names_an_id_with_no_file_is_a_ghost_and_is_named(tmp_path):
    """The opposite direction: the doc names #41, no `41-*.md` exists.

    The class #2630 closed for three paper digests and #1102 for #86. It is the direction a
    filing check cannot catch by definition — the file never arrives — so it lives here
    because `membership_drift` is ONE function that both the writers and the nightly node
    read, and the linter reports it too (`DOC_MEMBERSHIP` in
    `tests/test_validate_tasks_linter.py`). Splitting the pair across two implementations is
    how the two tables drifted from each other in the first place.
    """
    d = fixture_dir(tmp_path, LISTED)
    ghosts, unlisted, dupes = jm.membership_drift(
        fixture_doc(functions=(LISTED, UNLISTED), tiers=(LISTED,)), d)
    assert ghosts == ["#41"], ghosts
    assert unlisted == [] and dupes == []


def test_a_retired_membership_row_discharges_its_own_id(tmp_path):
    """The one exemption, read off the row rather than typed into a list.

    `#36`'s membership row says *Retired 2026-09-23* and its file is in `_archived/`, so the
    real doc would otherwise report it as a phantom forever. A row that merely mentions
    retirement in prose must NOT discharge: `RETIRED_MEMBERSHIP_ROW` is case-sensitive and
    anchored to the row's own cell, which is why #36's row matches and a sentence in another
    row about somebody else's retirement does not.
    """
    d = fixture_dir(tmp_path, LISTED)
    text = fixture_doc(functions=(LISTED, UNLISTED)) + (
        "| #41 | job | Retired 2026-10-17, superseded by #40 |\n")
    ghosts, _u, _dupes = jm.membership_drift(text, d)
    assert ghosts == [], f"a discharged id was still reported as a ghost: {ghosts}"
    # And the negative control: the same word in prose, outside a row for that id, is not a
    # discharge. Without this the exemption could be satisfied by any sentence anywhere.
    undischarged = fixture_doc(functions=(LISTED, UNLISTED)) + (
        "This section describes #40, whose predecessor was retired in 2026.\n")
    assert jm.membership_drift(undischarged, d)[0] == ["#41"], (
        "prose mentioning a retirement discharged an id it does not belong to")


def test_both_tables_are_asked_and_a_clean_pair_reports_nothing(tmp_path):
    """Drift is per table, so an id missing from ONE is still a finding.

    #2881's third instance put #100 and #101 in the functions table and not the tier table:
    symmetric omission is the easy case, and a check that asked only "is it listed anywhere"
    would have passed it. The names in the report are the two tables' own names, so the fixer
    knows which row to add.
    """
    d = fixture_dir(tmp_path, LISTED, UNLISTED)
    one_sided = fixture_doc(functions=(LISTED, UNLISTED), tiers=(LISTED,))
    per_table = {name: jm.table_drift(jm.named_ids(one_sided, header),
                                      jm.task_file_ids(d), jm.retired_named_ids(one_sided))
                 for name, header in jm.TABLES}
    assert per_table[jm.FUNCTIONS_TABLE] == ([], [], [])
    assert per_table[jm.TIERS_TABLE] == ([], ["#41"], []), per_table
    assert [name for name in (jm.FUNCTIONS_TABLE, jm.TIERS_TABLE)
            if jm.missing_tables(one_sided, UNLISTED).count(name)], (
        "a table that lists the id was reported as missing it")
    assert jm.missing_tables(one_sided, UNLISTED) == [jm.TIERS_TABLE]
    assert jm.missing_tables(fixture_doc(), LISTED) == [], (
        "an id in both tables came back as missing from one")


# ── Clause 3: the drift journal, one row per drifting filing ───────────────────


def test_a_drifting_filing_appends_exactly_one_row_with_these_four_fields(tmp_path, journal):
    """The row is the drift's start date, so its shape is the whole deliverable.

    `task_id`, the tables that omitted it, the doc's short sha and an ISO instant — the four
    the item names. The sha is what makes the row readable a month later: on 2026-10-17 the
    live doc hashes to a different 12 hex characters than it did on 2026-10-16, and #2881's
    report is precisely that a row which said "missing" could not say whether it still does.
    `ts` is UTC-aware ISO, not a naive local stamp, per the one-clock rule (`#1128` is the
    precedent in the file this guard now lives beside).
    """
    d = fixture_dir(tmp_path, LISTED, UNLISTED)
    doc = tmp_path / "autonomy-jobs.md"
    doc.write_text(fixture_doc(), encoding="utf-8")

    report = jm.report_at_filing(UNLISTED, task_dir=d, doc=doc,
                                 file_name=f"{UNLISTED}-seeded-job.md", journal=journal)
    assert report["tables_missing"] == [jm.FUNCTIONS_TABLE, jm.TIERS_TABLE], report
    assert report["row_written"] is True and report["journal"] == str(journal), report

    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, f"one drifting filing must append exactly one row, got {len(lines)}"
    row = json.loads(lines[0])
    assert row["task_id"] == "#41" and row["file"] == "41-seeded-job.md", row
    assert row["tables_missing"] == [jm.FUNCTIONS_TABLE, jm.TIERS_TABLE]
    assert row["doc_short_sha"] == jm.doc_short_sha(fixture_doc()), (
        "the row's sha is not the sha of the doc that was read, so it cannot be compared "
        "against the doc later")
    assert len(row["doc_short_sha"]) == 12
    import datetime as dt
    assert dt.datetime.fromisoformat(row["ts"]).tzinfo is not None, (
        f"the journal's stamp is not offset-aware: {row['ts']!r} — a naive stamp in a "
        f"machine-facing row is read as UTC by every later reader")
    assert report["doc_short_sha"] == row["doc_short_sha"]


def test_a_filing_both_tables_already_name_writes_no_row(tmp_path, journal):
    """No drift, no row — otherwise `wc -l` counts filings instead of problems.

    The distinction the owed-check job has to make a month from now is "every filing drifts"
    versus "one filing drifted, on this date". A row per filing destroys it, and the journal
    would grow without bound for a store whose entire purpose is a start date.
    """
    d = fixture_dir(tmp_path, LISTED)
    doc = tmp_path / "autonomy-jobs.md"
    doc.write_text(fixture_doc(), encoding="utf-8")

    report = jm.report_at_filing(LISTED, task_dir=d, doc=doc,
                                 file_name=f"{LISTED}-seeded-job.md", journal=journal)
    assert report["tables_missing"] == [] and report["row_written"] is False, report
    assert report["doc_short_sha"], "a clean report still carries the sha it checked against"
    assert not journal.exists(), "a filing with nothing wrong with it created the journal"


def test_the_journal_appends_and_never_rewrites_a_previous_row(tmp_path, journal):
    """Append-only. Two drifts, two rows, the first row's bytes unchanged.

    The row is the only evidence of when a drift began; a later run that recomputed it and
    rewrote the file would erase the date, which is the entire reason the journal exists
    rather than a boolean in the report.
    """
    d = fixture_dir(tmp_path, LISTED, UNLISTED, 42)
    doc = tmp_path / "autonomy-jobs.md"
    doc.write_text(fixture_doc(functions=(), tiers=()), encoding="utf-8")
    jm.report_at_filing(UNLISTED, task_dir=d, doc=doc, journal=journal)
    first = journal.read_bytes()
    jm.report_at_filing(42, task_dir=d, doc=doc, journal=journal)
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, lines
    assert (lines[0] + "\n").encode() == first, "the append rewrote the row it followed"
    assert json.loads(lines[1])["task_id"] == "#42"


def test_the_journal_reaches_the_pipeline_dir_and_no_promotion_path_reads_it(tmp_path):
    """Where the default journal lives, and who is forbidden to read it.

    `jm.journal_path()` resolves off `app.paths.PIPELINE_DIR` at call time — the 2026-09-22
    rule that runtime state hangs off `DATA_ROOT` and never inside a checkout, so a worktree
    writing here lands in its own gitignored `.lloyd-data` and a `rm -r` aimed at the tree
    cannot reach a record the owed-check job needs. The second half is the bound the item
    states outright: a doc a day behind has never been a reason to stop a landing, so nothing
    under `scripts/automod/` may read this file. `git grep` over that tree is the check; it
    was empty of any reference to the journal at HEAD before this round and must stay so.
    """
    from app import paths

    p = jm.journal_path()
    assert p == paths.PIPELINE_DIR / "arch-review" / "membership-drift.jsonl", p
    # CONTAINMENT, never a substring of the path. The first draft of this line read
    # `str(ROOT) not in str(p)`, and that predicate is FALSE IN EVERY LAYOUT on this
    # box: `~/lloyd-data` is the SIBLING of `~/lloyd`, so `str(ROOT)` is a literal
    # prefix of the journal path wherever the code lives — measured at `dda964f0`,
    # `str(Path("<home>/lloyd")) in str(PIPELINE_DIR / "arch-review" / "…")` → True. A
    # check that can never pass is not a check, and it failed at the last round's
    # `tests` rung before any clause was graded. `is_relative_to` compares PATH
    # COMPONENTS, which is the question the docstring asks. "Not under ROOT" is not the
    # rule in the other direction either: for a non-production tree
    # `app.data_root.data_root_for_tree` deliberately puts the data root at
    # `<tree>/.lloyd-data`, INSIDE the tree path, so an in-tree journal is a bug only in
    # a layout whose data root is somewhere else — which is what the pair below says,
    # and it holds in production (`~/lloyd-data`, a sibling of `~/lloyd`), in a gate
    # worktree (data root `<round>/home/lloyd-data`, sibling of its
    # `<round>/home/lloyd`), and in a sandbox checkout alike.
    # Containment is a question about PATH COMPONENTS, so both sides are normalised
    # lexically first: an env-supplied `LLOYD_DATA` may spell a sibling as
    # `<round>/home/lloyd/../lloyd-data`, and un-normalised `is_relative_to` reads that
    # as being under the tree (measured that way at this commit: `assert (not True or
    # False)`). `os.path.normpath` cancels the `..` without following symlinks, which
    # `Path.resolve()` would — and a round's `<round>/home/lloyd` is a symlink to the
    # worktree, so following links would answer the question about the wrong tree.
    def inside(child: Path, parent: Path) -> bool:
        return Path(os.path.normpath(child)).is_relative_to(
            Path(os.path.normpath(parent)))

    assert inside(p, paths.DATA_ROOT), (
        f"the journal hangs off something other than the runtime data root "
        f"{paths.DATA_ROOT}: {p}")
    sanctioned_fallback = (os.path.normpath(str(paths.DATA_ROOT))
                           == os.path.normpath(str(paths.data_root_for_tree(paths.LLOYD_HOME))))
    assert not inside(p, paths.LLOYD_HOME) or sanctioned_fallback, (
        "the journal sits inside the code tree while the data root is somewhere else — "
        "the 2026-09-22 failure the data root exists to prevent "
        f"(DATA_ROOT={paths.DATA_ROOT}, tree={paths.LLOYD_HOME}, journal={p})")

    import subprocess

    hit = subprocess.run(["git", "grep", "-l", "-e", "membership-drift", "-e", "MEMBERSHIP_DRIFT",
                          "--", "scripts/automod/"],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert hit == "", f"a promotion/rollback path now reads the drift journal:\n{hit}"


def test_an_unreadable_doc_is_reported_as_unchecked_never_as_clean(tmp_path, journal):
    """Fail-open means the failure is visible, not silent.

    A doc that cannot be read, or that holds neither table header, yields BOTH table names
    plus an `unchecked` reason and no refusal. The alternative — an empty `tables_missing` —
    is the worst value in the whole design, because it is indistinguishable from a doc that
    names the id, and every later reader of the report would file nothing and believe it was
    clean. This is the same shape as the description guard's ledger at
    `agent_mcp/autonomy.py`: `resolved: null` kept separate from `resolved: 0`.
    """
    d = fixture_dir(tmp_path, UNLISTED)
    missing = tmp_path / "nowhere.md"
    r1 = jm.report_at_filing(UNLISTED, task_dir=d, doc=missing, journal=journal)
    assert r1["unchecked"] and set(r1["tables_missing"]) == {jm.FUNCTIONS_TABLE, jm.TIERS_TABLE}
    assert r1["row_written"] is False, "an unchecked filing is not evidence of drift"

    headerless = tmp_path / "headerless.md"
    headerless.write_text("# A doc with no tables at all\n", encoding="utf-8")
    r2 = jm.report_at_filing(UNLISTED, task_dir=d, doc=headerless, journal=journal)
    assert "no" in r2["unchecked"] and "header" in r2["unchecked"], r2
    assert not journal.exists(), "an unreadable doc was journaled as though it had been read"


def test_a_journal_that_cannot_be_opened_is_a_field_not_an_exception(tmp_path):
    """The journal is a witness, so its own failure cannot break a filing.

    A directory where the journal path is, is the smallest way to make `open(..., "a")`
    raise without monkeypatching the standard library.
    """
    d = fixture_dir(tmp_path, LISTED, UNLISTED)
    doc = tmp_path / "autonomy-jobs.md"
    doc.write_text(fixture_doc(), encoding="utf-8")
    # The journal path itself is a directory: `mkdir(parents=True)` happily creates the
    # parent, so an un creatable parent is not the failure this needs — opening a directory
    # as a file is.
    blocked = tmp_path / "j.jsonl"
    blocked.mkdir()
    r = jm.report_at_filing(UNLISTED, task_dir=d, doc=doc, journal=blocked)
    assert r["tables_missing"], "the report itself was lost with the journal"
    assert r["row_written"] is False and "journal_error" in r, r


# ── Clauses 2, 3 and 6: both writers, on a real create ─────────────────────────
#
# `jm.JOBS_DOC` and `jm.journal_path` are patched rather than passed, because the writers
# deliberately do NOT take them: a caller that could point the report at a friendly doc could
# also point it at nothing, and the whole value of checking at filing is that the filer cannot
# opt out. Both are read at call time inside `report_at_filing`, so patching the module is
# enough to keep every row in this file inside `tmp_path` — no test here can touch the
# production journal or the real doc.


@pytest.fixture
def seeded_fleet(tmp_path, monkeypatch):
    """A one-file fleet directory (`40-…`), a doc naming only #40, and an empty journal.

    A create therefore gets id 41 — `_autonomy_next_id()` and `_next_task_id()` both take the
    highest id on disk plus one, so the seeded `40-…` is what makes the NEW file the drifting
    one, and the fixture is a miniature of the state #2881 was filed against.
    """
    d = fixture_dir(tmp_path, LISTED)
    doc = tmp_path / "autonomy-jobs.md"
    doc.write_text(fixture_doc(), encoding="utf-8")
    monkeypatch.setattr(jm, "JOBS_DOC", doc)
    monkeypatch.setattr(jm, "journal_path", lambda: tmp_path / "arch-review" / "drift.jsonl")
    # The ROUTER reads TWO directory globals on a create, and both are redirected here:
    # `_autonomy_write_file` resolves the filename under `ROUTER._AUTONOMY_DIR`, while
    # `_autonomy_next_id()` asks `app.autonomy.AUTONOMY_DIR` for the highest id on disk.
    # Leaving the second one live derives the new id from the real fleet — a number like 102
    # in a directory holding `40-…`, which silently changes which file the create is and makes
    # every assertion below about "#41" a measurement of `~/obsidian`. The route already reads
    # both globals for one request (`app/routers/autonomy.py:244-250` records why that split
    # is itself a hazard); a test that redirected only one would be testing a route nobody runs.
    from app import autonomy as A

    monkeypatch.setattr(A, "AUTONOMY_DIR", d)
    return d, doc, tmp_path / "arch-review" / "drift.jsonl"


def test_the_http_writer_returns_the_path_and_names_both_missing_tables(seeded_fleet, monkeypatch):
    """Clause 2 for the route, and clause 6 in the same breath: the file is still written.

    `_autonomy_write_file` returns a `Path` to eleven call sites, so the contract that holds
    it together is that this stays true and the report only ever rides ALONGSIDE it. The
    returned object is a real `Path` — `jm.TaskFilePath` subclasses `Path` and declares no
    state, so a caller that treats it as a path cannot tell — and its `.report` names the two
    tables that omit #41.
    """
    from app.routers import autonomy as ROUTER

    d, _doc, journal = seeded_fleet
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", d)

    # The id is the ROUTE's job, not the writer's: `autonomy_task_write` calls
    # `_autonomy_next_id()`, puts the result in `task_dict`, and only then calls the writer,
    # which names the file from `task_dict["id"]`. Passing 0 here would create
    # `0-seeded-drifting-job.md` and grade a create that no caller ever performs.
    new_id = ROUTER._autonomy_next_id()
    assert new_id == UNLISTED, f"the seeded fleet no longer derives id {UNLISTED}: {new_id}"
    written = ROUTER._autonomy_write_file({
        "type": "autonomy", "id": new_id, "name": "seeded drifting job", "status": "draft",
        "priority": "low", "frequency": "daily", "body": ""})

    assert written.is_file() and written.name == "41-seeded-drifting-job.md", written
    assert isinstance(written, Path) and written.parent == d
    report = written.report
    assert report["task_id"] == "#41", report
    assert report["tables_missing"] == [jm.FUNCTIONS_TABLE, jm.TIERS_TABLE], report
    assert journal.is_file(), "the report reached the caller but never reached the journal"


def test_the_http_create_response_carries_the_membership_report(seeded_fleet, monkeypatch):
    """The route's own response body, which is the only channel this writer has.

    Before this change the create path answered `{"task": {"id": new_id}}` and nothing else:
    the Mission Control page that posted it has no way to open
    `architecture/autonomy-jobs.md`, and nothing downstream reads a log line from here, so a
    report that was not in this body would exist only in the writer's return value — which the
    route discards. HTTP status is still 200, because a report is not a refusal.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers import autonomy as ROUTER

    d, _doc, journal = seeded_fleet
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", d)
    app = FastAPI()
    app.include_router(ROUTER.router)
    with TestClient(app) as client:
        r = client.post("/api/autonomy/task-write",
                        json={"name": "seeded drifting job", "status": "draft"})
    assert r.status_code == 200, r.text
    membership = r.json()["task"]["membership"]
    assert membership["task_id"] == "#41"
    assert membership["tables_missing"] == [jm.FUNCTIONS_TABLE, jm.TIERS_TABLE], membership
    assert membership["doc_short_sha"] == jm.doc_short_sha(fixture_doc())
    assert journal.is_file()
    row = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
    assert row["task_id"] == "#41" and row["file"] == "41-seeded-drifting-job.md", row


def test_the_mcp_writer_reports_on_create_and_still_writes_the_file(seeded_fleet, monkeypatch):
    """Clause 2 for the other writer, and the same refusal-free bound.

    `agent_mcp/autonomy.py` is the route that has actually been used — 41 `autonomy_write_task`
    calls against 0 recorded creates on the HTTP route — and the file that carries the rule
    "a guard on one of two writers is not a guard" is `app/routers/autonomy.py:422`. Patching
    both module globals means this node tests the same code the scheduler's tools run, not a
    parallel one.
    """
    from agent_mcp import autonomy as MCP

    d, _doc, journal = seeded_fleet
    monkeypatch.setattr(MCP, "AUTONOMY_DIR", d)

    new_id = MCP._next_task_id()
    assert new_id == UNLISTED, f"the seeded fleet no longer derives id {UNLISTED}: {new_id}"
    written = MCP._write_task_file({
        "id": new_id, "name": "seeded drifting job", "status": "draft", "priority": "low",
        "frequency": "daily", "agent_id": "memory", "body": ""})

    assert written.is_file(), "the MCP writer stopped writing on a create"
    assert written.name == "41-seeded-drifting-job.md", written.name
    assert written.report["tables_missing"] == [jm.FUNCTIONS_TABLE, jm.TIERS_TABLE]
    assert journal.is_file()
    # The file's own frontmatter is untouched by the report: no extra key, no rewritten body.
    text = written.read_text(encoding="utf-8")
    assert "membership" not in text and "tables_missing" not in text, (
        "the report leaked into the task file the scheduler parses")


def test_the_mcp_create_tool_result_carries_membership_for_the_caller(seeded_fleet, monkeypatch):
    """`autonomy_write_task`'s JSON answer gains the report.

    `_handle_write` threw the writer's return value away, so `.report` alone would have been
    a field no one could read. The caller of the tool is the only party who can still add the
    two doc rows while the task is fresh, which is the entire point of reporting at filing.
    """
    from agent_mcp import autonomy as MCP

    d, _doc, _journal = seeded_fleet
    monkeypatch.setattr(MCP, "AUTONOMY_DIR", d)

    import json as _json

    out = _json.loads(MCP._handle_write({"name": "seeded drifting job", "status": "draft"}))
    assert out["membership"]["task_id"] == "#41", out
    assert out["membership"]["tables_missing"] == [jm.FUNCTIONS_TABLE, jm.TIERS_TABLE]
    assert out["task"]["id"] == 41, "the task itself was not created"


def test_a_writer_never_refuses_even_when_the_doc_is_unreadable(seeded_fleet, monkeypatch,
                                                                tmp_path):
    """Clause 2's other half, on BOTH writers at once.

    The doc is replaced by a path that does not exist, which is the same shape as a vault
    checkout that is missing the file, a doc with its headers reworded, or the module failing
    to import at all. Every one of those must cost the caller a report field and nothing else:
    a create refused over a documentation lag would be a worse outage than the drift, and the
    item says `never refuses` in three places because the four rounds this came out of were
    all about a check that arrived too late, not about one that was too strict.
    """
    from app.routers import autonomy as ROUTER

    d, _doc, journal = seeded_fleet
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", d)
    monkeypatch.setattr(jm, "JOBS_DOC", tmp_path / "gone.md")

    written = ROUTER._autonomy_write_file({
        "type": "autonomy", "id": ROUTER._autonomy_next_id(),
        "name": "still written job", "status": "draft"})
    assert written.is_file(), "an unreadable doc refused the write"
    assert written.report["unchecked"] and set(written.report["tables_missing"]) == {
        jm.FUNCTIONS_TABLE, jm.TIERS_TABLE}, written.report
    assert not journal.exists(), "an unchecked filing was written to the journal as drift"


def test_a_report_that_raises_cannot_lose_the_written_file(seeded_fleet, monkeypatch):
    """The last line of the fail-open bound: a bug IN the guard is not a failed filing.

    `report_at_filing` is already total, so this node aims at the case that is left — the
    module failing to import, which is what actually happens to a `scripts/` namespace import
    when the tree is started from somewhere unexpected. `_membership_at_filing` catches it,
    logs, and hands back the plain path; the create still returns a path, and the report is
    simply absent rather than wrong.
    """
    from app.routers import autonomy as ROUTER

    d, _doc, journal = seeded_fleet
    monkeypatch.setattr(ROUTER, "_AUTONOMY_DIR", d)

    def boom(*a, **k):
        raise RuntimeError("simulated import failure inside the guard")

    monkeypatch.setattr(jm, "report_at_filing", boom)
    written = ROUTER._autonomy_write_file({
        "type": "autonomy", "id": ROUTER._autonomy_next_id(),
        "name": "job the guard could not grade", "status": "draft"})
    assert written.is_file() and written.name == "41-job-the-guard-could-not-grade.md"
    assert getattr(written, "report", None) is None, (
        "a half-built report was attached, which a reader takes for a real one")
    assert not journal.exists()
