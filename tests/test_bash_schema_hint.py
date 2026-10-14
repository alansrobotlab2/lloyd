"""The Bash tool answers a wrong-column sqlite error with the store's real columns (#2703).

Every node but one runs the real `_bash` over the real `sqlite3` binary against a
fixture store built under the suite's own data root (`tests/conftest.py` points
`LLOYD_DATA` at a scratch directory before anything imports `app.paths`, so
`app.paths.DATA_ROOT` here is scratch, never `~/lloyd-data`). Each node gets its own
directory under that root — same containment, and no node sees another's
`ALTER TABLE` (the suite runs 8 workers, so they may not share a store either).

The exceptions are named rather than glossed, because a node that does not run the
tool proves something narrower and its docstring has to say which:
`test_the_connect_is_read_only_and_its_consequence_is_pinned` runs no command at all —
it calls `_columns` directly, which is the one claim here that needs no shell — and two
other nodes additionally call a step of the rail (`H._store_path`) beside the end-to-end
run of the same command, to pin a decision the end-to-end run cannot isolate: the CLI
cannot be made to name a store that stays ABSENT, because sqlite's own open creates it.
A step pin is always beside the real run, never instead of it, so no node passes with
the wiring deleted.

`tmp_path` is not special. It is an ordinary directory in a real path, used for the
two nodes that need a directory OUTSIDE the data root — which is what a
store-creating path looks like.
"""

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

import agent_mcp._bash_schema_hint as H
from agent_mcp import builtin_bash

# The real `pause_journal` shape from `~/lloyd-data/workers.db`, columns and all:
# `at`, `value`, `holder_kind` are the names the item names as not guessable from
# the table name. `watermarks` is the other store the item names — 406 rows of
# upserted current state, whose `updated_at` is a last-write age, not an event age.
PAUSE_JOURNAL = ("create table pause_journal(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "at TEXT, value TEXT, holder_kind TEXT, actor TEXT, hold_id INTEGER)")
WATERMARKS = ("create table watermarks(source TEXT, key TEXT, value TEXT, "
              "updated_at TEXT, PRIMARY KEY(source,key))")

#: What sqlite prints for a guessed column, in the two forms the class takes. Both
#: name the column and neither names the table — which is the whole problem.
CLI_MSG = "no such column: valid_from"
PY_MSG = "sqlite3.OperationalError: no such column: ts"


@pytest.fixture()
def root(tmp_path, monkeypatch) -> Path:
    """A data root of this node's own, pinned in BOTH places that read it.

    The rail resolves the root through `app.paths.DATA_ROOT` at call time, so this
    cannot merely read the module attribute and build a path from it: under `-n 8` a
    worker that ran a node which rebinds `paths.DATA_ROOT` without restoring it
    (`test_uptake.py`, `test_protected_paths.py` and `test_workers_sources.py` each
    do) would put the store in one root and the rail's data-root test in another,
    and every hint would silently disappear. Binding the attribute to the directory
    this node returns makes the fixture and the rail agree whatever ran before them."""
    d = tmp_path / "data-root"
    d.mkdir(parents=True, exist_ok=True)
    import app.paths
    monkeypatch.setattr(app.paths, "DATA_ROOT", d)
    return d


def _store(directory: Path, name: str = "workers.db") -> Path:
    p = directory / name
    con = sqlite3.connect(p)
    con.executescript(f"{PAUSE_JOURNAL}; {WATERMARKS};")
    con.execute("insert into pause_journal(at,value,holder_kind,actor,hold_id) "
                "values('2026-10-14T07:00Z','released','fleet','autocode',1)")
    con.commit()
    con.close()
    return p


def _alter(store: Path, ddl: str) -> None:
    con = sqlite3.connect(store)
    con.execute(ddl)
    con.commit()
    con.close()


def _run(monkeypatch, command: str, cwd: Path | None = None) -> str:
    """The real tool over a real shell, with no seam crossed at all.

    `tests/test_bash_edit_diagnostics.py` fakes `agent_mcp.call_tool` because its
    commands carry `--rpc` and would otherwise reach the backend; none of these do,
    so the whole dispatch runs for real."""
    args = {"command": command}
    if cwd is not None:
        args["cwd"] = str(cwd)
    return asyncio.run(builtin_bash._bash(args))


def _without_rail(monkeypatch, command: str, cwd: Path | None) -> str:
    """One run of `command` with `_with_schema_hint` swapped for a pass-through.

    That pass-through is the tool as it shipped this morning, so a fail-open claim is
    a byte comparison against the real thing rather than an expectation of it.
    Restored by hand, not by `monkeypatch.undo()`: the `root` fixture patches
    `app.paths.DATA_ROOT` on the SAME function-scoped MonkeyPatch, and undo would take
    the data root away with it — every hint would then vanish and the arm that is
    supposed to have it would fail for the wrong reason."""
    async def passthrough(text, command_, cwd_):
        return text
    real = builtin_bash._with_schema_hint
    builtin_bash._with_schema_hint = passthrough
    try:
        return _run(monkeypatch, command, cwd)
    finally:
        builtin_bash._with_schema_hint = real


def _both_ways(monkeypatch, command: str, cwd: Path | None) -> tuple[str, str]:
    """The same command with the rail removed, then with it present."""
    return _without_rail(monkeypatch, command, cwd), _run(monkeypatch, command, cwd)


# ── clause 1: the line is there, and the payload it rides on is intact ───────

def test_a_guessed_column_gets_the_table_and_its_real_columns(monkeypatch, root):
    """#2684's near-miss and #2496's real one, run as the Bash command that was:
    `select valid_from from pause_journal` exits 1, and the result now names
    `pause_journal` and its six real columns instead of leaving the job to guess a
    second time.

    `[exit code: 1]` is still the last thing the tool's own payload wrote, and the
    hint is ONE line: the rail must not turn a failure into a manual."""
    _store(root)
    out = _run(monkeypatch, 'sqlite3 workers.db "select valid_from from pause_journal;"',
               cwd=root)
    assert CLI_MSG in out, out                              # the real sqlite error
    assert "[exit code: 1]" in out, "the payload the rail rides on is unchanged"
    lines = [ln for ln in out.splitlines() if ln.startswith("[schema: ")]
    assert len(lines) == 1, out
    line = lines[0]
    assert out.index("[exit code: 1]") < out.index(line), "appended AFTER the payload"
    assert "pause_journal" in line
    for col in ("id", "at", "value", "holder_kind", "actor", "hold_id"):
        assert col in line, f"{col} missing from the answer: {line}"
    assert "valid_from" in line, "the guess is named, so the reader sees what failed"
    assert "\n" not in line
    assert "watermarks" not in line, "only the table the query was wrong about"


def test_an_absolute_store_path_gets_the_same_line(monkeypatch, root):
    """The shape the item names as its own after-landing check
    (`sqlite3 ~/lloyd-data/workers.db "select valid_from from pause_journal;"`),
    where the store is an absolute path rather than one resolved from a cwd."""
    store = _store(root)
    out = _run(monkeypatch, f'sqlite3 {store} "select finished_at from pause_journal"')
    assert "[exit code: 1]" in out
    assert "[schema: `pause_journal`" in out and "hold_id" in out, out
    assert str(store) in out, "the line names the store it read"


def test_a_command_that_changes_directory_resolves_the_store_as_the_shell_would(monkeypatch, root):
    """`cd <root> && sqlite3 workers.db ...` is how much of the fleet phrases the
    probe, so the store has to resolve from the `cd` target as well as from the
    call's own cwd — or the rail would be blind to the commonest form."""
    _store(root)
    out = _run(monkeypatch,
               f'cd {root} && sqlite3 workers.db "select model from pause_journal"')
    assert "[exit code: 1]" in out
    assert "[schema: `pause_journal`" in out and "hold_id" in out, out


def test_a_python_traceback_about_a_missing_column_gets_the_hint_too(monkeypatch, root):
    """Half the class is not the CLI: a job runs a python probe against a store and
    reads `sqlite3.OperationalError: no such column: ts`. The rail reads the merged
    text the tool already assembled, so that form is answered as well."""
    store = _store(root)
    out = _run(monkeypatch,
               "python3 - <<'PY'\n"
               f"import sqlite3\n"
               f"c = sqlite3.connect({str(store)!r})\n"
               "print(c.execute('select ts from watermarks').fetchall())\n"
               "PY")
    assert PY_MSG in out, out
    assert "[schema: `watermarks`" in out and "updated_at" in out, out


def test_the_hint_names_the_one_table_the_query_was_wrong_about(monkeypatch, root):
    """A store holds dozens of tables and a command can name two. `watermarks`
    really does have `value`; `pause_journal` does not have `valid_from` — the table
    without the column is the one the error is about, and the rail says so rather
    than dumping the schema."""
    _store(root)
    out = _run(monkeypatch,
               'sqlite3 workers.db "select valid_from from pause_journal union all '
               'select value from watermarks;"', cwd=root)
    assert "[exit code: 1]" in out
    line = [ln for ln in out.splitlines() if ln.startswith("[schema: ")][0]
    assert "pause_journal" in line and "watermarks" not in line, line


# ── clause 2: the columns come from the store, not from code ────────────────

def test_adding_a_column_to_the_store_changes_the_hint_with_no_code_change(monkeypatch, root):
    """The property that makes this an instrument and not a lookup table: `ALTER
    TABLE ... ADD COLUMN` on the store changes what the rail prints. A hint baked
    into code would keep answering with the old six columns forever, which is the
    drift `PRAGMA table_info` exists to avoid."""
    store = _store(root)
    before = _run(monkeypatch, 'sqlite3 workers.db "select valid_from from pause_journal"',
                  cwd=root)
    _alter(store, "alter table pause_journal add column mystery_col TEXT")
    after = _run(monkeypatch, 'sqlite3 workers.db "select valid_from from pause_journal"',
                 cwd=root)
    assert "mystery_col" not in before
    assert "mystery_col" in after, after
    assert after.count("[schema: ") == 1, "the one line is refreshed, not added to"


def test_a_table_that_has_the_named_column_is_never_the_one_named(monkeypatch, root):
    """The converse of clause 1, which is what makes the answer a read and not a
    guess of its own. `at` is a real `pause_journal` column and a real
    non-`watermarks` column (both tables are in the fixture with their live shapes),
    so the error `no such column: at` can only be about `watermarks` — and the rail
    says `watermarks` even though the OTHER table is the one whose name is easier to
    reach in the command text."""
    _store(root)
    out = _run(monkeypatch,
               'sqlite3 workers.db "select at from watermarks union all select at from '
               'pause_journal;"', cwd=root)
    assert "no such column: at" in out, out
    line = [ln for ln in out.splitlines() if ln.startswith("[schema: ")][0]
    assert "watermarks" in line and "updated_at" in line, line
    assert "holder_kind" not in line, "pause_journal has `at`; naming it would mislead"
    assert "PRAGMA table_info" in line


# ── clause 3: fail-open, byte for byte ─────────────────────────────────────

def test_a_failure_with_no_table_in_it_is_left_exactly_as_it_was(monkeypatch, root):
    """`select valid_from;` — sqlite's own words, no FROM, no table to recover. The
    result must be byte-identical to the same command with the rail taken out."""
    _store(root)
    without, with_rail = _both_ways(monkeypatch, 'sqlite3 workers.db "select valid_from;"',
                                    root)
    assert CLI_MSG in with_rail
    assert with_rail == without, "the rail explained a failure it cannot attribute"


def test_a_sidecar_named_by_its_parent_is_not_a_store(monkeypatch, root):
    """`workers.db-wal` and `workers.db-shm` sit beside every live store, and the
    obvious `\\S+\\.db` scan matches the PREFIX of them, so the pattern that reads
    most naturally would name the parent database for a command that never named it.
    A sidecar is a write-ahead log, not a schema, so the token lookahead forbids a
    `-` after the extension for exactly this reason.

    Pinned at `_store_path`, the step that decides it, and not through the CLI —
    because on sqlite 3.53.4 the CLI route cannot isolate this rule: `sqlite3
    workers.db-wal "select valid_from from pause_journal"` prints `no such table:
    pause_journal` when the parent is present (measured: sqlite will not read a
    schema through a sidecar name), so `hint()` returns at its column-error gate
    before the token pattern is ever consulted; and when no parent exists the
    dash-stripped token names a file that is not there, so the rail refuses for the
    store-existence reason instead. Either way a CLI node passes with the `-` taken
    out of the lookahead, which is the honesty defect this node replaces. What the
    end-to-end run still shows is the consequence: no line."""
    store = _store(root)
    (root / "workers.db-wal").write_bytes(b"not a schema")
    wal_cmd = 'sqlite3 workers.db-wal "select valid_from from pause_journal"'
    db_cmd = 'sqlite3 workers.db "select valid_from from pause_journal"'
    assert H._store_path(wal_cmd, str(root)) is None, \
        "a `-wal` name is a log, and its extension is where that is decided"
    assert H._store_path(db_cmd, str(root)) == str(store.resolve()), \
        "same root, same cwd, plain name: the refusal above is the dash, not the plumbing"
    out = _run(monkeypatch, wal_cmd, cwd=root)
    assert "[exit code:" in out, out
    assert "[schema:" not in out, \
        "the WAL is not the database, and naming one sends the reader nowhere"
    yes = _run(monkeypatch, db_cmd, cwd=root)
    assert "[schema: `pause_journal`" in yes


def test_a_copied_store_with_a_second_suffix_is_not_the_store_named(monkeypatch, root):
    """The other half of the lookahead, and this one IS pinned end-to-end. A rotated
    or copied store — `workers.db.bak` — really is a sqlite database, so the CLI
    answers a wrong column from it and the error gate is passed; but the token rule
    ends a store name at the extension, so a command that named only the backup must
    not get a line about `workers.db`, a file it never named. Take the `.` out of the
    lookahead and this command matches the `workers.db` PREFIX of the backup's name,
    that file exists under the root, and the node fails."""
    store = _store(root)
    (root / "workers.db.bak").write_bytes(store.read_bytes())
    out = _run(monkeypatch, 'sqlite3 workers.db.bak "select valid_from from pause_journal"',
               cwd=root)
    assert CLI_MSG in out, \
        f"the CLI must answer the guessed column from the copy, so this reaches the store: {out}"
    assert "[exit code: 1]" in out
    assert "[schema:" not in out, \
        "the caller named a backup; answering with the live store's schema names a file it did not touch"
    yes = _run(monkeypatch, 'sqlite3 workers.db "select valid_from from pause_journal"',
               cwd=root)
    assert "[schema: `pause_journal`" in yes, "the same bytes named plainly still answer"


def test_a_store_outside_the_data_root_is_left_alone(monkeypatch, tmp_path):
    """A probe against some other sqlite file — a checkout's own fixture, a temp
    db — is not one of Lloyd's runtime stores, and the rail has no business naming
    its schema. Byte-identical again."""
    store = _store(tmp_path)
    without, with_rail = _both_ways(
        monkeypatch, f'sqlite3 {store} "select valid_from from pause_journal"', tmp_path)
    assert CLI_MSG in with_rail
    assert with_rail == without


def test_a_missing_store_is_left_alone(monkeypatch, root):
    """A name that does not exist under the data root means the command is about to
    CREATE the store, and a line naming a schema has to describe one that exists.

    The existence decision is pinned at `_store_path`, on the same root and with the
    same command text as the store-present arm beside it, because the CLI cannot be made
    to leave a store absent: by the time a command's output exists, `sqlite3
    nosuchstore.db` has created a 0-byte `nosuchstore.db` (measured on this box). What
    that arm asserts is the CONTRACT — an absent path is not a store to describe — and
    not one particular line of it: deleting the `isfile` test alone changes no answer,
    because the `getsize` on the line below it and the read-only open in `_columns` each
    refuse the same path (measured: that mutation survives this file). Three guards over
    the same promise is deliberate; the assertion is here so a refactor that removes all
    three has somewhere to fail.

    The byte-equality that follows is the tool's payload, and what it pins is stated
    exactly: sqlite has already created the empty file by then, so this arm is no
    longer about absence at all — it pins that a store with no schema of its own earns
    no line. It goes live rather than exiting early because the SQL names
    `sqlite_master`, which an empty database does have, so the column error really fires
    (`no such table: pause_journal` would stop `hint()` at its first gate, which is the
    vacuity this node used to have)."""
    absent = 'sqlite3 nosuchstore.db "select valid_from from pause_journal"'
    assert H._store_path(absent, str(root)) is None, \
        "an absent path under the root is a store about to be created, not one to describe"
    _store(root)
    assert H._store_path('sqlite3 workers.db "select valid_from from pause_journal"',
                         str(root)) == str((root / "workers.db").resolve()), \
        "the same call with a store that exists resolves: the refusal above was existence"

    live = 'sqlite3 nosuchstore.db "select valid_from from sqlite_master"'
    without, with_rail = _both_ways(monkeypatch, live, root)
    assert CLI_MSG in with_rail, \
        f"a wrong column against an empty store must reach the rail for this arm to mean anything: {with_rail}"
    assert "[schema:" not in with_rail, \
        ("the only table an empty store has is sqlite's own catalog, and naming its five "
         f"columns answers nothing about the schema the caller came for: {with_rail}")
    assert with_rail == without
    # What this arm cannot pin — that reading an absent store leaves nothing behind —
    # is pinned where the reason for it lives, in
    # `test_the_connect_is_read_only_and_its_consequence_is_pinned`.


def test_a_store_that_refuses_to_open_appends_nothing(monkeypatch, root):
    """The third fail-open arm: table and store are both perfectly recoverable and
    the open itself fails — a corrupt header, a permission, a lock the pool is
    holding. The tool's answer cannot depend on the rail's luck."""
    _store(root)
    command = 'sqlite3 workers.db "select valid_from from pause_journal"'
    without, _ = _both_ways(monkeypatch, command, root)
    def explodes(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    real = H._columns
    H._columns = explodes
    try:
        assert _run(monkeypatch, command, cwd=root) == without
    finally:
        H._columns = real
    assert "[schema: `pause_journal`" in _run(monkeypatch, command, cwd=root), \
        "the arm above is a stubbed failure, so the same command must work unstubbed"


def test_a_successful_command_is_never_annotated(monkeypatch, root):
    """A command that exits 0 while printing the words — `echo`, a grep of a log, or
    the skill file that quotes this very error — gets nothing. The rail sits on the
    failure path only, which is also why the loose grep count of the string is not a
    measurement of this class."""
    out = _run(monkeypatch, f"echo '{CLI_MSG}'")
    assert out.strip().endswith(CLI_MSG), out
    assert "[schema:" not in out


def test_no_such_table_is_still_refused(monkeypatch, root):
    """The sibling error is the owed-check job's ruling (#2703, owed-after-landing
    clause 2), and the round that holds that ruling is not this one. Pinning the
    absence is what keeps the scope from drifting in by accretion."""
    _store(root)
    out = _run(monkeypatch, 'sqlite3 workers.db "select id from nope;"', cwd=root)
    assert "no such table: nope" in out
    assert "[schema:" not in out


# ── clause 4: the rail can never create or write a store ───────────────────

def test_a_bare_db_name_from_a_foreign_cwd_creates_nothing(monkeypatch, root, tmp_path):
    """The trap `runtime-store-schema-probe` step 1 warns about, aimed at the rail
    itself: a bare `sqlite3 queue.db` from a directory that is not the data root
    CREATES an empty database there. sqlite's CLI does that as a matter of course, so
    what this node pins is the rail's own contribution — it must add no file anywhere
    and no line, in a cwd it has no claim on or under the data root where a hint
    would have been tempting."""
    command = 'sqlite3 brand_new_store.db "select valid_from from queue"'
    alone = tmp_path / "railless"
    with_rail = tmp_path / "railed"
    for d in (alone, with_rail):
        d.mkdir()
    async def passthrough(text, command_, cwd_):
        return text
    real = builtin_bash._with_schema_hint
    builtin_bash._with_schema_hint = passthrough
    try:
        out_alone = _run(monkeypatch, command, cwd=alone)
    finally:
        # Restored by hand: `monkeypatch.undo()` would take the fixture's
        # `DATA_ROOT` patch with it and the second arm would pass for the wrong
        # reason — a rail with no resolvable root says nothing anywhere.
        builtin_bash._with_schema_hint = real
    out_rail = _run(monkeypatch, command, cwd=with_rail)
    assert out_rail == out_alone, "the rail appended nothing to a store it cannot read"
    assert sorted(p.name for p in alone.iterdir()) == sorted(p.name for p in with_rail.iterdir()), \
        "the rail left one file more than the command itself did"
    assert list(root.glob("*brand_new_store*")) == [], \
        "nor did it create one under the data root, where hinting would have been easy"


def test_a_sidecar_left_the_store_alone_where_the_hint_still_works(monkeypatch, root):
    """The counterfactual for the sidecar rule: refusing `workers.db-wal` must not
    cost the same command its answer when the store itself is named beside it, since
    a session that inspects the WAL usually names the database in the same line."""
    _store(root)
    (root / "workers.db-wal").write_bytes(b"not a schema")
    out = _run(monkeypatch,
               'cat workers.db-wal >/dev/null; sqlite3 workers.db '
               '"select valid_from from pause_journal"', cwd=root)
    assert "[schema: `pause_journal`" in out, out


def test_the_connect_is_read_only_and_its_consequence_is_pinned(monkeypatch, root):
    """Why the open in `_columns` is written the way it is, stated where a reader who
    wants to change it will meet it — and pinned by its consequence, not only by its
    text.

    The decisive observable is the absent path. A plain `sqlite3.connect(path)` CREATES
    the file at connect time (measured on this box: connecting to a name that does not
    exist leaves a 0-byte file), so a rail that opened without `mode=ro` would leave a
    store behind in whatever directory a probe happened to name — the exact trap
    `skills/runtime-store-schema-probe` step 1 warns about, executed by the instrument
    meant to help. The second assertion below is therefore the test of the mode, and it
    fails the moment the mode is dropped; the positive control above is what stops a
    broken `_columns` from passing it. `PRAGMA query_only = ON` rides along as the
    second lock on the same promise. What no behaviour can show is the outright refusal
    of a path containing `?` or `#` — sqlite's URI parser ends the filename at the first
    of them, so quoting them would not make such a path safe — and that stays a source
    assertion, labelled as one."""
    _store(root)
    assert H._columns(str((root / "workers.db").resolve()), "pause_journal") == \
        ["id", "at", "value", "holder_kind", "actor", "hold_id"], \
        "the positive control: on a store that exists, the read answers in table order"
    ghost = root / "ghostmode.db"
    assert H._columns(str(ghost), "pause_journal") is None
    assert not ghost.exists(), \
        "the mode is what keeps a read of an absent store from creating one"
    src = Path(H.__file__).read_text()
    assert 'sqlite3.connect(f"file:{store}?mode=ro", uri=True' in src, \
        "the read-only open is the guarantee; if this line changes, the assertion above " \
        "is the thing to re-read"


def test_sqlite_s_own_catalog_table_is_never_the_one_named(monkeypatch, root):
    """`select <guess> from sqlite_master` is a shape a probe really takes, and
    `sqlite_master` exists in every store — so an unguarded rail answers `type, name,
    tbl_name, rootpage, sql`, five names that say nothing about the schema the caller
    came for, in a line that reads like a read. The guard is safe by construction
    rather than by taste: sqlite refuses `CREATE TABLE sqlite_foo` with "object name
    reserved for internal use" (measured on this box), so a `sqlite_` prefix can only
    ever be the catalog. The store here has real tables, which is what makes the
    counterfactual below mean something: the refusal is the table, not the store."""
    _store(root)
    out = _run(monkeypatch, 'sqlite3 workers.db "select valid_from from sqlite_master"',
               cwd=root)
    assert CLI_MSG in out, \
        f"the catalog table really does carry a wrong column, or this arm tests nothing: {out}"
    assert "[schema:" not in out, \
        "naming the catalog's five columns is noise wearing an answer's clothes"
    yes = _run(monkeypatch, 'sqlite3 workers.db "select valid_from from watermarks"',
               cwd=root)
    assert "[schema: `watermarks`" in yes, yes


# ── the seam the fleet actually runs on: a store somebody is writing to ──────

@pytest.mark.parametrize("journal", ["delete", "wal"])
def test_the_hint_still_answers_while_a_writer_holds_the_store(monkeypatch, root, journal):
    """`workers.db` is not the closed, uncontended file every other node in this file
    builds: the pool holds write locks on it continuously, and it is WAL. So this node
    crosses the seam the instrument really runs against — a live connection with a
    transaction open at the moment the failing command is issued, in both journal
    modes — because a rail that has to wait for the writer either stalls the tool or
    fails open, and the item's whole claim is that the answer arrives.

    The 5-second bound is not a benchmark: `BUSY_TIMEOUT_MS` is 250, so a run slower
    than this is the rail waiting on something it was written to refuse, and the tool
    would have been made to hang by the thing meant to help it."""
    store = _store(root)
    _alter(store, f"pragma journal_mode={journal}")
    holder = sqlite3.connect(store)
    holder.execute("begin immediate")
    holder.execute("insert into pause_journal(actor) values('held')")
    try:
        started = time.monotonic()
        out = _run(monkeypatch, 'sqlite3 workers.db "select valid_from from pause_journal"',
                   cwd=root)
        elapsed = time.monotonic() - started
    finally:
        holder.rollback()
        holder.close()
    assert CLI_MSG in out, \
        f"the writer must not change what sqlite reports ({journal}): {out}"
    lines = [ln for ln in out.splitlines() if ln.startswith("[schema: ")]
    assert len(lines) == 1, f"the writer's lock cost the answer ({journal}): {out}"
    assert "holder_kind" in lines[0], lines[0]
    assert elapsed < 5, \
        f"the rail waited {elapsed:.1f}s on a {journal}-mode lock it holds a 250ms timeout for"


def test_a_column_name_that_could_forge_a_line_is_not_echoed(monkeypatch, root):
    """The appended text is bytes out of a database, and Bash output already carries
    untrusted content into a prompt — so a schema name that could carry its own
    newline, or a second `[schema:` marker, is refused rather than sanitised."""
    store = root / "odd.db"
    con = sqlite3.connect(store)
    con.execute("create table plain(alpha TEXT, beta TEXT)")
    con.execute('create table twisted("first\nsecond" TEXT, gamma TEXT)')
    con.commit()
    con.close()
    ok = _run(monkeypatch, 'sqlite3 odd.db "select nope from plain"', cwd=root)
    assert "[schema: `plain`" in ok and "alpha" in ok, ok
    bad = _run(monkeypatch, 'sqlite3 odd.db "select nope from twisted"', cwd=root)
    assert "no such column: nope" in bad, bad
    assert "[schema:" not in bad, "a name that is not a plain word is left unspoken"
