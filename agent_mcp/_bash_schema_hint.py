"""A wrong-column sqlite error against a runtime store answers with its real columns (#2703).

`skills/runtime-store-schema-probe/SKILL.md` tells every job to read
`PRAGMA table_info` before naming a column, and probe 17 of
`skills/nightly-reflection-signals/SKILL.md` repeats the sentence. Prose has not
worked: the tight form of the error — `sqlite3.OperationalError: no such column:`
or `Error: in prepare, no such column:` — sits in 2 to 7 session files per UTC day
across 2026-10-08 to 2026-10-14, spread over owed-check (46 files store-wide),
autotriage (17), autocode, automod-review and autonomy tasks #38/#42/#58/#79/#82,
with fresh instances on 2026-10-13 and 10-14 for `status`, `job`, `name`, `model`,
`finished_at` and `id`. Each one costs the job that hit it a turn to re-derive a
name that was always one `PRAGMA` away. (The "19 vs 8, 2.4x and rising" figure that
filed this work is prose echo — a loose grep matches the skill text and the reports
that quote the error, and the measuring session itself. The class is chronic, not
spiking. Rate after landing is judged on the tight form only.)

So the store answers instead of the prose. `append(text, command, cwd)` takes the
failing Bash payload and, when all of it holds, adds ONE line naming the table and
its real columns:

- the text carries `no such column: <name>` (sqlite's own wording, case
  insensitive, and it names the column but never the table);
- the command names a store file that resolves to an existing file under the
  data root — the root `app.paths.DATA_ROOT` names, never `$HOME`;
- the command names a table that exists in that store and does NOT have the
  column, which is what makes the named table the one the error is about.

The columns come from one `PRAGMA table_info(<table>)` run at failure time on that
store, so the answer cannot drift from the schema the way a table baked into code
would. The connection is `file:...?mode=ro` with `PRAGMA query_only`: a plain
`sqlite3 workers.db` from a cwd that is not the data root CREATES an empty database
there — step 1 of the very skill this is a substitute for — and the pool holds write
locks on `workers.db` continuously, so a rail that opens a store the wrong way is
worse than no rail.

Anything unprovable means append nothing, byte-identically to the tool without
this module: the column error absent, no store path under the data root, a name
that is not a store because it carries a second suffix (a `-wal` sidecar or a
`.db.bak` copy), a store that does not exist yet, no recoverable table, only
sqlite's own catalog table, a table that already has the column, a name sqlite
would not accept, or any exception at all. Not covered, deliberately: `no such
table` (the item left it to the owed-check ruling), a command that exits 0 after
printing the error, and background Bash, whose result is a task handle and cannot
carry a line.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3

logger = logging.getLogger("lloyd-builtin-bash")

#: How much of a store and a schema one line may name. A runtime store's table
#: count is in the dozens and a table's column count under 30; a pathological
#: schema gets nothing rather than a wall of text.
MAX_STORE_BYTES = 512 * 1024 * 1024      # refuse to open a store bigger than this
MAX_TABLES_CONSIDERED = 6                # table tokens tried per command
MAX_COLUMNS = 40
BUSY_TIMEOUT_MS = 250                    # a reader must never stall the tool

#: sqlite's wording. It names the COLUMN and never the table, which is the whole
#: reason the table has to come from the command text. Two spellings reach us:
#: the CLI's `Error: in prepare, no such column: x` and python's
#: `sqlite3.OperationalError: no such column: x`.
_NO_SUCH_COLUMN = re.compile(
    r"\bno such column:\s*(?:[A-Za-z_][\w$]*\.)?[\"'`]?([A-Za-z_][\w$]*)", re.I)

#: A store file. Extension-only: a path is a candidate because of its suffix, and
#: then has to survive the existence and data-root checks below to count at all. The
#: lookahead ends the token at the extension, which does two jobs, and both are
#: load-bearing:
#:   - a `-` (measured on sqlite 3.53.4): `workers.db-wal`/`-shm` sit beside every
#:     live store, and without it the pattern matches the `workers.db` PREFIX of a
#:     sidecar name and hints the parent database for a command that never named it;
#:   - a `.`: same prefix match for a rotated or copied file — `store.db.bak` really
#:     is a sqlite database, so the CLI answers a wrong column from it, but naming
#:     the schema of `store.db` for a command that named only its backup is the rail
#:     asserting a file the caller did not touch.
#: Both boundaries are pinned in `tests/test_bash_schema_hint.py`.
_STORE_TOKEN = re.compile(
    r"""([^\s'"`;|&<>(){}\[\],=]+?\.(?:db|sqlite3?|db3|sqlite))(?![\w.\-])""",
    re.I,
)

#: Where a table name can appear in the SQL that is sitting inside the command.
#: `UPDATE`/`INTO` are here because a job that guessed a column on a write fails
#: the same way, and the item's class is any probe of a runtime store.
_TABLE_TOKEN = re.compile(
    r"""\b(?:from|join|into|update)\s+[`"\[]?([A-Za-z_][\w$]*)""", re.I)

# `cd` is followed so a command that changes directory before probing resolves the
# same way the shell would. Same shape as `_bash_edit_diagnostics._CD`.
_CD = re.compile(r"""(?:^|[;&|\n(]\s*)cd\s+(['"]?)([^\s'";&|)]+)\1""")

#: What may be echoed back inside one line. sqlite lets a schema name contain
#: almost anything; a hint that could carry a newline could carry a fake tool
#: marker, so a store whose schema is not plain ASCII words gets nothing.
_SAFE_NAME = re.compile(r"[\w$]{1,64}")

_SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".db3")


def _data_root() -> str:
    """The data root as an absolute string, or "" when it cannot be resolved.

    Resolved per call through `app.paths`, which is where the suite's scratch root
    and the gate's `LLOYD_DATA` both land: an instrument that read `$HOME` would
    point at a worktree inside a round (`app/paths.py`'s header on 2026-09-22) and
    name columns for a database nobody is querying.
    """
    try:
        from app.paths import DATA_ROOT
        return str(DATA_ROOT)
    except Exception:
        logger.debug("schema hint: no data root", exc_info=True)
        return ""


def _expand(token: str) -> str:
    return os.path.expanduser(os.path.expandvars(token))


def _bases(command: str, cwd: str | None) -> list[str]:
    base = cwd or os.getcwd()
    out = [base]
    for _q, raw in _CD.findall(command):
        d = _expand(raw)
        d = d if os.path.isabs(d) else os.path.join(base, d)
        if d not in out:
            out.append(os.path.normpath(d))
    return out


def _store_path(command: str, cwd: str | None) -> str | None:
    """The one store this command names under the data root, or None.

    Existence is required: the connect is read-only anyway, but a path that does
    not exist means the command was about to CREATE a store there, and the line
    this rail writes must describe a schema that exists. A symlink is followed for
    the containment test and reported as its realpath, so an off-root target is
    refused rather than hinted.
    """
    root = _data_root()
    if not root:
        return None
    root_real = os.path.realpath(root)
    hits: list[str] = []
    for token in _STORE_TOKEN.findall(command):
        tok = _expand(token)
        if not tok.lower().endswith(_SQLITE_SUFFIXES):
            continue
        for base in _bases(command, cwd):
            cand = tok if os.path.isabs(tok) else os.path.join(base, tok)
            real = os.path.realpath(cand)
            if real in hits:
                break
            if not os.path.isfile(real):
                continue
            if real != root_real and not real.startswith(root_real + os.sep):
                continue
            try:
                if os.path.getsize(real) > MAX_STORE_BYTES:
                    continue
            except OSError:
                continue
            hits.append(real)
            break
    # One store per command, first named wins: a command joining two stores is not
    # this rail's to adjudicate, and a line naming the wrong one is worse than none.
    return hits[0] if len(hits) == 1 else None


def _tables(command: str) -> list[str]:
    """Table tokens in command order, minus sqlite's own catalog tables.

    `sqlite_master` (and its alias `sqlite_schema`) always exist, so a guessed
    column against an EMPTY store resolves to a real table and the rail would
    answer with `type, name, tbl_name, rootpage, sql` — five names that tell a
    reader nothing about the schema it came for, which is the one thing an
    authoritative-looking line must not do. No user table can hide behind the
    test: sqlite refuses `CREATE TABLE sqlite_foo` itself ("object name reserved
    for internal use"), so a `sqlite_` prefix is the catalog and nothing else.
    """
    seen: list[str] = []
    for name in _TABLE_TOKEN.findall(command):
        if name.lower().startswith("sqlite_"):
            continue
        if name not in seen:
            seen.append(name)
    return seen[:MAX_TABLES_CONSIDERED]


def _columns(store: str, table: str) -> list[str] | None:
    """`PRAGMA table_info(<table>)` on the store, read-only; None on any doubt."""
    if "?" in store or "#" in store:
        # sqlite's URI parser ends the filename at the first of these, so a path
        # carrying one would silently retarget the open. Refuse it, don't quote it.
        return None
    try:
        con = sqlite3.connect(f"file:{store}?mode=ro", uri=True,
                              timeout=BUSY_TIMEOUT_MS / 1000.0)
    except Exception:
        return None
    try:
        con.execute("PRAGMA query_only = ON")
        con.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_MS)}")
        rows = con.execute(
            "SELECT name FROM pragma_table_info(?) WHERE name IS NOT NULL",
            (table,)).fetchall()
        cols = [str(r[0]) for r in rows]
        if not cols or len(cols) > MAX_COLUMNS:
            return None
        if not all(_SAFE_NAME.fullmatch(c) for c in cols):
            return None
        return cols
    except Exception:
        logger.debug("schema hint: pragma failed", exc_info=True)
        return None
    finally:
        try:
            con.close()
        except Exception:
            pass


def hint(text: str, command: str, cwd: str | None) -> str | None:
    """The one line this failure deserves, or None when it cannot be proven.

    The table named is one that exists in the store and does NOT have the missing
    column: that is what distinguishes the table the query was wrong about from
    some other table the command happened to mention, and it is why a command
    whose every named table has the column gets nothing at all.
    """
    match = _NO_SUCH_COLUMN.search(text or "")
    if not match:
        return None
    column = match.group(1)
    store = _store_path(command, cwd)
    if not store:
        return None
    for table in _tables(command):
        cols = _columns(store, table)
        if cols is None or column in cols:
            continue
        return (f"[schema: `{table}` in {store} has columns "
                f"{', '.join(cols)} — no `{column}`; PRAGMA table_info, live]")
    return None


def append(text: str, command: str, cwd: str | None) -> str:
    """`text` plus that line; never raises, and identical to `text` when no line."""
    try:
        line = hint(text, command, cwd)
    except Exception:
        logger.warning("bash schema hint failed", exc_info=True)
        return text
    return text if line is None else f"{text}\n{line}"
