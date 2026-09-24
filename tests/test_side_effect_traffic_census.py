"""#1056: the side-effect traffic census is a denominator, not a dashboard snapshot.

The dispatch-gate items (#590, #591, #708) accept against a corpus of real
recorded high-consequence calls. Triage measured that corpus by hand and found
the outbound/irreversible class holds one event in four weeks of transcripts
while the #544 effect ledger structurally cannot hold it at all. Those numbers
were greps re-run by hand each time someone needed them, and every one of them
roughly doubled within five days — so the item asked for a re-runnable artifact
that publishes the denominator in one call.

Each test names the acceptance clause it pins. The fixture corpus is written by
the tests themselves and reached through `--sessions-dir` / `--db`, so no test
reads the live transcripts or opens the live ledger: a census test that measured
production could not assert a number, which is the whole point of the artifact.

Every fixture row is written by the module that actually writes `sessions/*.json`
— `app.transcript_entries` (`build_tool_call` `:113`, `build_tool_call_entry`
`:133`, called from `app/routers/messages.py:727`) — the way
`tests/test_tool_failure_baseline.py:44` does. That is the one process boundary
this change crosses: the census reads files a different process wrote, so a
hand-shaped fixture would keep passing after the producer renamed a field and the
census silently read zero calls. The fixture ledger is created from
`agent_mcp._tool_effects._SCHEMA` (`:97`) rather than a copy of the DDL, for the
same reason on the other source.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pwd
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "side_effect_traffic_census.py"

from agent_mcp import _tool_effects as ledger  # noqa: E402
from agent_mcp import annotations as tool_annotations  # noqa: E402
from app import transcript_entries as te  # noqa: E402
from app.harness import policy  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("side_effect_traffic_census", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CENSUS = _load()

#: The four tools the acceptance clause names as unledgered, and the three it
#: names as ledgered. `vault_write` is in the first group because `IDEMPOTENT`
#: holds it; `email_delete` is in `DESTRUCTIVE` *and* `IDEMPOTENT`, and
#: `IDEMPOTENT` wins — which is the disagreement the census exists to surface.
UNLEDGERED = ("email_delete", "calendar_delete_event", "contacts_delete", "vault_write")
LEDGERED = ("email_send", "backlog_write_task", "fact_add")

#: One instant, and the two spellings the two sources use for it. The session
#: store writes NAIVE stamps (the `timestamp=datetime.now().isoformat()` argument
#: of a `_tool_pair` call, `app/routers/messages.py:1222-1225`) and the ledger writes
#: offset-bearing UTC (`_tool_effects._now()` at `:192`); the census reads a
#: naive stamp as local, which is the only reading that agrees with its writer.
#: Anchoring both fixtures on one
#: instant keeps every window assertion independent of what day the suite runs.
_ANCHOR = datetime(2026, 9, 20, 12, 0, 0)
_LOCAL = _ANCHOR.astimezone()
_UTC = _LOCAL.astimezone(timezone.utc)
_STAMP = _LOCAL.isoformat(timespec="seconds")
_STAMP_UTC = _UTC.isoformat(timespec="seconds")
_OLD_LOCAL = (_LOCAL - timedelta(days=40)).isoformat(timespec="seconds")
_OLD_UTC = (_UTC - timedelta(days=40)).isoformat(timespec="seconds")


def _transcript(dirpath: Path, stem: str, calls: list[tuple[str, str | None]],
                narrative: str = "") -> None:
    """Write one fixture transcript from (tool, per-message stamp or None) pairs.

    The rows come from `app.transcript_entries`, the module the session store
    writes `sessions/*.json` through — one assistant row per call, OpenAI-shaped,
    which is what the live corpus on this box actually holds (measured
    2026-09-22: 973 call records, 0 rows carrying the `content[].tool_use`
    spelling beside the `tool_calls` one). `stamp=None` drops the row's own
    timestamp, the third stamp shape in the same store, whose only date is the
    session's `created_at`.
    """
    msgs = []
    for i, (tool, stamp) in enumerate(calls):
        msg = te.build_tool_call_entry(te.build_tool_call(f"tu-{i}", tool, "{}"),
                                       timestamp=stamp or _STAMP)
        if stamp is None:
            msg.pop("timestamp")
        msg["id"] = f"msg-{i}"
        msgs.append(msg)
    if narrative:
        msgs.append(te.build_tool_result_entry("tu-narrative", narrative,
                                               timestamp=_STAMP))
    (dirpath / f"{stem}.json").write_text(json.dumps({
        "session_id": stem, "id": stem, "title": "fixture",
        "created_at": _STAMP, "messages": msgs,
    }), encoding="utf-8")


@pytest.fixture
def sessions(tmp_path):
    """A corpus with known counts: two in-window transcripts, one 40 days old."""
    d = tmp_path / "sessions"
    d.mkdir()
    _transcript(d, "a", [
        ("backlog_write_task", _STAMP), ("backlog_write_task", _STAMP),
        ("fact_add", _STAMP), ("vault_write", _STAMP), ("email_recent", _STAMP),
    ])
    _transcript(d, "b", [("email_delete", _STAMP), ("memory_add", _STAMP)],
                # Prose about a call, not a call: three textual occurrences of
                # `email_send`'s invocation shape inside one tool RESULT body.
                # The store serialises that body as a string, so today's quotes
                # arrive escaped and the hand grep happens to miss them — the
                # census may not depend on that accident, because the same
                # inflation with nothing hiding it is exactly what #1056 records:
                # one 2026-08-26 `email_reply` read as "2 calls" because the call
                # appears twice in the file.
                # `test_a_call_written_in_both_shapes_counts_once` pins that case;
                # this row pins that prose contributes 0 calls.
                narrative='previous pass recorded {"name": "email_send"} twice: '
                          '{"name": "email_send"} and {"name": "email_send"}.')
    _transcript(d, "old", [("backlog_write_task", _OLD_LOCAL),
                           ("vault_write", _OLD_LOCAL)])
    return d


@pytest.fixture
def db(tmp_path):
    """A ledger with known per-tool row counts, written through the real schema."""
    path = tmp_path / "workers.db"
    conn = sqlite3.connect(path)
    conn.executescript(ledger._SCHEMA)
    rows = ([("backlog_write_task", _STAMP_UTC)] * 4
            + [("fact_add", _STAMP_UTC)] * 2
            + [("session_inject_context", _STAMP_UTC)]
            # Ledgered before commit `e58d2f5` put them in `IDEMPOTENT`, and
            # never since: 0 rows in window, but a real history. The census has
            # to keep those two facts distinguishable.
            + [("vault_write", _OLD_UTC), ("Write", _OLD_UTC)])
    for i, (tool, stamp) in enumerate(rows):
        conn.execute(
            "insert into tool_effects(effect_key, tool, scope, status,"
            " created_at, updated_at) values (?,?,?,?,?,?)",
            (f"k{i}", tool, "run:test", "ok", stamp, stamp))
    conn.commit()
    conn.close()
    return path


def _census(mod, sessions, db, days=30):
    return mod.build_census(sessions_dir=sessions, db=db, days=days)


def _rows(mod, sessions, db, days=30):
    return {r["tool"]: r for r in _census(mod, sessions, db, days)["rows"]}


# ---------------------------------------------------------------- clause 1
def test_census_prints_both_counts_for_every_side_effecting_tool(sessions, db):
    """One row per side-effecting tool carrying BOTH sources over the window.

    The fixture holds 2 `backlog_write_task` transcript calls and 4 ledger rows,
    1 `fact_add` call and 2 rows, 1 in-window `vault_write` call and 0 ledger rows
    in window, and 1 `email_delete` call that the ledger can never record.
    """
    rows = _rows(CENSUS, sessions, db)
    assert rows["backlog_write_task"]["transcript_calls"] == 2
    assert rows["backlog_write_task"]["ledger_rows"] == 4
    assert rows["fact_add"]["transcript_calls"] == 1
    assert rows["fact_add"]["ledger_rows"] == 2
    assert rows["vault_write"]["transcript_calls"] == 1
    assert rows["vault_write"]["ledger_rows"] == 0
    # A tier-2 sender whose only textual occurrences are quoted narrative is
    # still a row, at 0 calls.
    assert rows["email_send"]["transcript_calls"] == 0
    assert rows["email_send"]["ledger_rows"] == 0


def test_cli_flags_point_at_the_fixture_and_print_exactly_those_counts(sessions, db,
                                                                       capsys):
    """`--sessions-dir`, `--db` and `--days` are the whole interface (#1056 step 1)."""
    code = CENSUS.main([
        "--sessions-dir", str(sessions), "--db", str(db), "--days", "30"])
    out = capsys.readouterr().out
    assert code == CENSUS.EXIT_MEASURED
    row = next(line for line in out.splitlines()
               if line.startswith("backlog_write_task"))
    # TIER, LEDGER, TRANSCRIPT_CALLS, LEDGER_ROWS, in that column order. The
    # trailing WHY cell is the only column that may be empty, so the assertion
    # reads the four that are always there rather than counting from the right.
    assert row.split()[1:5] == ["1", "ledgered", "2", "4"], row


def test_default_roots_are_the_running_systems_whatever_home_says(monkeypatch,
                                                                  tmp_path):
    """No flags still means the live corpus, from any tree and any `$HOME`.

    The one seam clause 1's "re-runnable" has that the fixture flags cannot
    reach: the census is a reader of data a *different* process wrote, and since
    2026-09-22 that data lives outside the code tree (`app/data_root.py`,
    `architecture/data-home.md`). Two wrong answers are live shapes, not
    hypotheticals — `~/lloyd/sessions` and `~/lloyd/workers.db` are the pre-move
    roots every early #1056 hand grep used and both are gone (`ls` answers "No
    such file or directory", while `~/lloyd-data/sessions` holds 307
    transcripts), and `app.paths.SESSIONS_DIR` follows the calling process, so a
    round running the census from its worktree would scan the worktree's own
    empty `.lloyd-data/sessions` and print `status=ok` over nothing — the
    empty-corpus result the artifact exists to refuse. `production_data_root()`
    is read off `pwd`, so repointing `HOME`, which is exactly what an automod
    gate does, must not move either default.
    """
    from app.data_root import production_data_root

    monkeypatch.delenv(CENSUS.LEDGER_DB_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path / "elsewhere"))
    assert CENSUS.DEFAULT_SESSIONS_DIR == production_data_root() / "sessions"
    assert CENSUS.DEFAULT_DB == production_data_root() / "workers.db"
    # The pre-move code-tree roots, spelled out so a future edit that reaches
    # for `Path.home() / "lloyd"` fails here rather than in a quoted table.
    assert CENSUS.DEFAULT_SESSIONS_DIR != tmp_path / "lloyd" / "sessions"
    assert CENSUS.DEFAULT_SESSIONS_DIR.parent.name == "lloyd-data"
    assert CENSUS.default_db_path() == production_data_root() / "workers.db"


def test_the_ledger_override_env_names_the_file_the_ledger_would_write(monkeypatch,
                                                                       tmp_path):
    """`LLOYD_EFFECT_LEDGER_DB` is the ledger's own knob; the census reads it.

    `agent_mcp._tool_effects.db_path()` routes a scoped write through this
    variable, so a census that ignored it would measure a database the writer is
    not using — and a test or rebuild that sets it would be censused against the
    live file instead of its own.
    """
    from app.data_root import production_data_root

    target = tmp_path / "ledger-under-test.db"
    monkeypatch.setenv(CENSUS.LEDGER_DB_ENV, str(target))
    assert CENSUS.default_db_path() == target
    monkeypatch.delenv(CENSUS.LEDGER_DB_ENV)
    assert CENSUS.default_db_path() == production_data_root() / "workers.db"


def test_the_quoted_module_invocation_runs_as_a_process(sessions, db, tmp_path):
    """`python -m scripts.side_effect_traffic_census` is the call a report cites.

    Everything else here imports the module and calls `main()` in-process, which
    cannot see the two things a quoted command depends on: that `-m scripts.…`
    resolves from the repo root at all, and that the verdict reaches the shell as
    an exit code. Both halves matter — #593's stop clause and this item's own
    Re-check are instructions to *run* the census, and an exit code swallowed by a
    traceback, or a module path that only imports under `sys.path` games, is a
    different answer from the one a reader gets. The empty-corpus invocation is the
    second call because exit 2 is the claim that has to survive the process
    boundary: a census that exits 0 over nothing is how an all-zero table becomes
    a quotation.
    """
    measured = subprocess.run(
        [sys.executable, "-m", "scripts.side_effect_traffic_census",
         "--sessions-dir", str(sessions), "--db", str(db), "--days", "30"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert measured.returncode == CENSUS.EXIT_MEASURED, measured.stderr[-1500:]
    row = next((line for line in measured.stdout.splitlines()
                if line.startswith("backlog_write_task")), measured.stdout)
    assert row.split()[1:5] == ["1", "ledgered", "2", "4"], row

    bare = tmp_path / "no-transcripts"
    bare.mkdir()
    bare_db = tmp_path / "no-rows.db"
    conn = sqlite3.connect(bare_db)
    conn.executescript(ledger._SCHEMA)
    conn.close()
    empty = subprocess.run(
        [sys.executable, "-m", "scripts.side_effect_traffic_census",
         "--sessions-dir", str(bare), "--db", str(bare_db), "--days", "30"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert empty.returncode == CENSUS.EXIT_EMPTY_WINDOW, empty.stdout[-800:]
    assert "empty-window" in empty.stdout

    json_run = subprocess.run(
        [sys.executable, "-m", "scripts.side_effect_traffic_census",
         "--sessions-dir", str(sessions), "--db", str(db), "--json"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert json_run.returncode == CENSUS.EXIT_MEASURED, json_run.stderr[-1500:]
    doc = json.loads(json_run.stdout)
    assert doc["verdict"] == "measured"
    assert {r["tool"]: r["ledger_rows"] for r in doc["rows"]}["fact_add"] == 2


def test_running_it_twice_prints_identical_output(sessions, db, capsys):
    """Re-runnable means re-quotable: no clock, no dict ordering, no drift.

    The in-process pair catches a clock read and an unstable sort. It cannot
    catch the ordering that actually varies run to run here: the row set is a
    `set` union (`reported_tools`), and `set` iteration order for strings is
    randomised per process, so two calls in one interpreter see the same order
    whatever the code does. Two real processes over an unchanged corpus is the
    claim the artifact exists for — a table quoted into #590/#591/#708 must be
    byte-identical to the next run of the same command, and it is only a
    denominator if it is.
    """
    first = CENSUS.main(["--sessions-dir", str(sessions), "--db", str(db)])
    out_one = capsys.readouterr().out
    second = CENSUS.main(["--sessions-dir", str(sessions), "--db", str(db)])
    out_two = capsys.readouterr().out
    assert (first, out_one) == (second, out_two)
    assert out_one, "the in-process run printed no table at all"

    runs = [subprocess.run(
        [sys.executable, "-m", "scripts.side_effect_traffic_census",
         "--sessions-dir", str(sessions), "--db", str(db)],
        cwd=ROOT, capture_output=True, text=True, timeout=300) for _ in range(2)]
    assert [r.returncode for r in runs] == [CENSUS.EXIT_MEASURED] * 2
    assert runs[0].stdout == runs[1].stdout, "same corpus, two processes, two tables"
    assert out_one == runs[0].stdout, "the in-process table is not the printed one"


def test_reads_both_schemas_of_call_record(tmp_path, db):
    """`tool_calls[].function.name` and `content[].tool_use` both count.

    Matching one shape with no fallback means the census silently measures
    whichever writer it happens to hit: the store's message rows are
    OpenAI-shaped (`app/transcript_entries.py:133`), and the Anthropic block
    shape is what a spilled tool result carries. A transcript with no message
    timestamps at all falls back to the session's own `created_at`, which is the
    third shape in the same store. The two rows in the Anthropic spelling are
    hand-written because `app.transcript_entries` emits only the OpenAI one, so
    the fallback has no real writer to borrow.
    """
    d = tmp_path / "sessions"
    d.mkdir()
    _transcript(d, "openai", [("fact_add", _STAMP)])      # the builder's own shape
    (d / "anthropic.json").write_text(json.dumps({"messages": [
        {"role": "assistant", "timestamp": _STAMP,
         "content": [{"type": "tool_use", "name": "fact_add"}]}]}))
    (d / "legacy.json").write_text(json.dumps({
        "created_at": _STAMP,
        "messages": [{"role": "assistant",
                      "content": [{"type": "tool_use", "name": "fact_add"}]}]}))
    rows = _rows(CENSUS, d, db)
    assert rows["fact_add"]["transcript_calls"] == 3


def test_a_call_written_in_both_shapes_counts_once(tmp_path, db):
    """One call the store wrote twice over is one call, not two.

    This is the inflation the census replaces, in the only form it takes on this
    corpus: #1056's 2026-08-26 triage found 2 `email_reply` hits and they were
    ONE event, because the call appears in both call-record spellings. A count
    that joined those shapes by id would report a 100 % overstatement for a class
    whose entire real population is one call.
    """
    d = tmp_path / "sessions"
    d.mkdir()
    msg = te.build_tool_call_entry(te.build_tool_call("tu-1", "email_reply", "{}"),
                                   timestamp=_STAMP)
    msg["content"] = [{"type": "tool_use", "id": "tu-1", "name": "email_reply",
                       "input": {}}]
    (d / "both-shapes.json").write_text(json.dumps({"messages": [msg]}))
    rows = _rows(CENSUS, d, db)
    assert rows["email_reply"]["transcript_calls"] == 1
    # The claim is about the join, not the reader: the invocation shape really
    # does appear twice in the file, so a grep counts 2 here.
    text = (d / "both-shapes.json").read_text(encoding="utf-8")
    assert text.count('"name": "email_reply"') == 2


def test_census_call_counts_equal_the_hand_grep_over_a_corpus_of_calls(tmp_path, db):
    """#1056's verification sentence: same numbers, one call, on a corpus of calls.

    The item is done when re-running its `grep -ho '"name": "<tool>"' *.json`
    returns what the census prints. On a corpus that is nothing but call rows the
    two must agree exactly — this reads the same files both ways, the way the
    live corpus does today (measured 2026-09-22T20:35Z: 973 textual occurrences,
    973 call records, zero tools differing). Where they part is
    `test_a_call_written_in_both_shapes_counts_once` and
    `test_narrative_tool_names_never_become_traffic`, and the census is the
    correct side of both.
    """
    d = tmp_path / "sessions"
    d.mkdir()
    _transcript(d, "calls", [("backlog_write_task", _STAMP)] * 3
                + [("vault_write", _STAMP), ("Bash", _STAMP), ("Bash", _STAMP)])
    rows = _rows(CENSUS, d, db)

    def textual(tool: str) -> int:
        """The item's own count: occurrences of the invocation shape, space included."""
        needle = f'"name": "{tool}"'
        return sum(p.read_text(encoding="utf-8").count(needle)
                   for p in sorted(d.glob("*.json")))

    for tool, calls in (("backlog_write_task", 3), ("vault_write", 1), ("Bash", 2)):
        assert rows[tool]["transcript_calls"] == calls, tool
        assert textual(tool) == calls, tool


def test_out_of_window_calls_are_excluded_and_counted_as_excluded(sessions, db):
    """`old.json`'s 40-day-old calls are absent from the counts, visible as excluded."""
    out = CENSUS.scan_transcripts(sessions, _UTC - timedelta(days=30), _UTC)
    assert out["by_tool"]["backlog_write_task"] == 2
    assert out["calls_out_of_window"] == 2      # backlog_write_task + vault_write
    assert out["files_found"] == 3 and out["files_in_window"] == 2


def test_narrative_tool_names_never_become_traffic(sessions, db):
    """A tool RESULT body quoting `{"name": "email_send"}` is not a call.

    `b.json` of the fixture is one result row whose text quotes the invocation
    shape three times. `grep -ho '"name": "<tool>"' *.json` — the count #1056
    prescribed — scans the raw file, so whether those three land in its number
    depends on how the serialiser escaped them; the census does not read the body
    at all and so never has to ask. A denominator whose value can move because a
    model repeated a tool name in prose cannot bound a gate's corpus, which is
    the reason the count is over call records here and not over the file.
    """
    rows = _rows(CENSUS, sessions, db)
    assert rows["email_send"]["transcript_calls"] == 0


def test_a_name_key_outside_a_call_record_is_not_traffic(tmp_path, db):
    """The census reads call records, not every `name` key in the document.

    The constructed half of the clause, not an observed one: on the live corpus
    the two counts agree exactly today (973 textual occurrences against 973 call
    records, measured 2026-09-22T20:35Z), because result bodies are stored as
    escaped strings and nothing else carries the shape. That accident is what not
    to depend on. Here one `{"name": "email_send"}` object sits in the document
    outside any call row, so the grep reads 1 and the census reads 0.
    """
    d = tmp_path / "sessions"
    d.mkdir()
    _transcript(d, "one-call", [("backlog_write_task", _STAMP)])
    path = d / "one-call.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["denied"] = [{"name": "email_send", "reason": "tier-2, no grant"}]
    path.write_text(json.dumps(doc), encoding="utf-8")
    rows = _rows(CENSUS, d, db)
    assert path.read_text(encoding="utf-8").count('"name": "email_send"') == 1
    assert rows["email_send"]["transcript_calls"] == 0


# ---------------------------------------------------------------- clause 2
def test_tools_the_ledger_cannot_record_are_labelled_not_reported_as_measured_zero(
        sessions, db):
    """`not ledgered` rides on the row, so a structural zero is never read as absence."""
    rows = _rows(CENSUS, sessions, db)
    for tool in UNLEDGERED:
        assert rows[tool]["ledgered"] is False, tool
        assert rows[tool]["ledger_zero_kind"] == "structural", tool
        # The reason names the annotations table that excludes it, so a reader
        # can check the claim instead of taking the census's word for it.
        assert rows[tool]["ledger_exclusion"] == "IDEMPOTENT", tool
    for tool in LEDGERED:
        assert rows[tool]["ledgered"] is True, tool
        assert rows[tool]["ledger_zero_kind"] == "measured", tool
        assert rows[tool]["ledger_exclusion"] is None, tool
    # The classifier is the ledger's own, so the label cannot drift from it.
    for tool in UNLEDGERED + LEDGERED:
        assert rows[tool]["ledgered"] == tool_annotations.side_effecting(tool)


def test_unledgered_tool_with_ledger_history_keeps_both_facts_visible(sessions, db):
    """`vault_write` has rows in the table and still cannot have any in window.

    Every `vault_write` and `Write` row in the real ledger predates `e58d2f5`
    (2026-09-10), which put both in `IDEMPOTENT`; #1056's triage measured the
    last of them at `2026-09-10T23:48:34+00:00`. The fixture reproduces that
    shape, so the 0 in the cell cannot be read as "vault writes are covered by
    the ledger" nor the rows as "the ledger still records them".
    """
    code = _census(CENSUS, sessions, db)
    rows = {r["tool"]: r for r in code["rows"]}
    den = code["denominators"]["ledger"]
    assert rows["vault_write"]["ledger_rows"] == 0
    assert rows["vault_write"]["ledgered"] is False
    assert den["all_by_tool"]["vault_write"] == 1      # present, outside the window
    assert den["all_by_tool"]["Write"] == 1
    assert den["rows_total"] == 9 and den["rows_in_window"] == 7


def test_the_printed_row_names_the_excluding_table(sessions, db, capsys):
    """The label is only useful if the reason rides along in the human report."""
    CENSUS.main(["--sessions-dir", str(sessions), "--db", str(db)])
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("email_delete"))
    assert "not ledgered" in row and row.rstrip().endswith("IDEMPOTENT"), row
    assert "STRUCTURAL" in out


# ---------------------------------------------------------------- clause 3
def test_denominators_are_printed_beside_the_counts(sessions, db, capsys):
    """Files scanned, rows in the table, rows in the window, and both endpoints."""
    CENSUS.main(["--sessions-dir", str(sessions), "--db", str(db)])
    out = capsys.readouterr().out
    assert "status=ok" in out
    assert "files_found=3 parsed=3" in out
    assert "calls_in_window=7" in out          # 5 in `a.json` + 2 in `b.json`
    assert "calls_out_of_window=2" in out
    assert "rows_total=9" in out and "rows_in_window=7" in out
    assert re.search(r"first_record=\S.*last_record=\S", out, re.S)
    # The window is stated as an interval, not only as a day count.
    assert re.search(r"window\s+\d{4}-\d{2}-\d{2}.*->.*\d{4}-\d{2}-\d{2}", out)


def test_a_short_window_shows_what_it_excluded(sessions, db):
    """`--days 1` over this corpus keeps the newest record and drops the rest, visibly.

    The window end is the corpus's newest record rather than a clock, so a short
    window cannot silently produce an all-zero table: whatever it dropped is
    printed as `calls_out_of_window` / `rows_total - rows_in_window`, which is
    what a reader needs to see before quoting a zero at the top of the table.
    """
    code = _census(CENSUS, sessions, db, days=1)
    tr, led = code["denominators"]["transcripts"], code["denominators"]["ledger"]
    assert code["verdict"] == "measured"
    assert tr["calls_in_window"] == 7 and tr["calls_out_of_window"] == 2
    assert led["rows_in_window"] == 7 and led["rows_total"] == 9


def test_an_empty_window_is_reported_as_an_empty_window(tmp_path):
    """A table that is all zero because nothing fell inside it must not exit 0.

    This is the failure mode #593's stop clause — "if the sample contains too few
    bad calls to measure a catch rate at all, say so and stop" — depends on. The
    corpus here is the shape that produces it: a transcript whose one call carries
    a stamp the parser does not recognise, and a ledger table that has taken no
    scoped write yet — the state this box was actually in at 2026-09-22T20:12Z.
    Both sources are readable, `status=ok` on each, the table is all zero, and the
    only honest verdict is that the census could place nothing in time. The window
    end then falls back to the wall clock, and says so.
    """
    d = tmp_path / "odd-sessions"
    d.mkdir()
    (d / "corrupt.json").write_text(json.dumps({"messages": [
        {"role": "assistant", "timestamp": "not-a-timestamp",
         "tool_calls": [{"function": {"name": "backlog_write_task"}}]}]}))
    empty_db = tmp_path / "fresh.db"
    conn = sqlite3.connect(empty_db)
    conn.executescript(ledger._SCHEMA)
    conn.close()
    code = _census(CENSUS, d, empty_db, days=1)
    assert code["verdict"] == "empty-window"
    assert code["exit_code"] == CENSUS.EXIT_EMPTY_WINDOW
    assert code["window"]["end_from_corpus"] is False
    tr, led = code["denominators"]["transcripts"], code["denominators"]["ledger"]
    assert tr["status"] == "ok" and tr["calls_missing_stamp"] == 1
    assert tr["calls_in_window"] == 0
    assert led["status"] == "ok" and led["rows_total"] == 0
    assert led["rows_in_window"] == 0
    reason = " ".join(code["unreadable_reasons"])
    assert "0 of 1 transcript files" in reason and "0 of 0 ledger rows" in reason


def test_missing_db_reports_unreadable_rather_than_zero(sessions, tmp_path):
    """A census that printed 0 in every ledger cell over a missing db is a lie."""
    code = _census(CENSUS, sessions, tmp_path / "nope.db")
    rows = {r["tool"]: r for r in code["rows"]}
    assert code["exit_code"] == CENSUS.EXIT_SOURCE_UNREADABLE
    assert code["denominators"]["ledger"]["status"] == "absent"
    assert all(r["ledger_rows"] is None for r in code["rows"])
    assert all(r["ledger_zero_kind"] == "unavailable" for r in code["rows"])
    assert rows["backlog_write_task"]["transcript_calls"] == 2   # other source stays real


def test_ledger_db_without_the_table_reports_its_own_state(sessions, tmp_path):
    """The live case: `tool_effects` is created on the first scoped ledgered write.

    Measured on this box 2026-09-22T20:12Z — the live `workers.db` had no such
    table at all — and #1056's own 2026-09-13 probe hit the same state (the
    `no such table: tool_effects` traceback quoted in the item). Both must read
    as an unreadable source, never as a ledger that recorded nothing.
    """
    path = tmp_path / "empty.db"
    conn = sqlite3.connect(path)
    conn.execute("create table queue (id integer)")
    conn.commit()
    conn.close()
    code = _census(CENSUS, sessions, path)
    assert code["exit_code"] == CENSUS.EXIT_SOURCE_UNREADABLE
    assert code["denominators"]["ledger"]["status"] == "no table"
    assert all(r["ledger_zero_kind"] == "unavailable" for r in code["rows"])


def test_unreadable_sessions_dir_reports_unreadable_rather_than_zero(tmp_path):
    code = _census(CENSUS, tmp_path / "gone", tmp_path / "also-gone.db")
    assert code["exit_code"] == CENSUS.EXIT_SOURCE_UNREADABLE
    assert code["denominators"]["transcripts"]["status"].startswith("absent")
    assert all(r["transcript_zero_kind"] == "unavailable" for r in code["rows"])


# ---------------------------------------------------------------- clause 5
def test_no_consequence_tier_table_lives_in_the_census():
    """#1056 step 3: `policy.py`'s tables stay the single tier source.

    Checks the artifact rather than trusting intent. Three consequence
    classifications already exist and disagree on the same tool — `email_delete`
    is a member of `policy.TIER3_TOOLS` (`app/harness/policy.py:244`, the name at
    `:246`) while sitting in `annotations.IDEMPOTENT`
    (`agent_mcp/annotations.py:110`, the name at `:115`), which is what makes the
    effect ledger blind to it — so a fourth table would be a fourth place for the
    same tool to be wrong.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"\bTIER[123]_TOOLS\s*=", source), "census authored a tier table"
    assert "def tool_tier" not in source, "census authored its own tier function"
    assert "policy.tool_tier(" in source
    assert "tool_annotations.side_effecting(" in source


def test_every_row_derives_its_class_from_the_two_owning_tables(sessions, db):
    """Tier from `policy.tool_tier`, ledger label from `annotations.side_effecting`."""
    rows = _rows(CENSUS, sessions, db)
    assert rows, "census emitted no rows"
    for tool, row in rows.items():
        assert row["tier"] == policy.tool_tier(tool), tool
        assert row["ledgered"] == tool_annotations.side_effecting(tool), tool


def test_a_name_no_table_claims_is_printed_as_the_default_it_is(tmp_path, db,
                                                                capsys):
    """An unclassified name keeps its row and loses its authority.

    #1056's sibling #768 found a `fake_writer` row — written by a test that ran
    without the ledger override — sitting in the production `tool_effects` table
    and holding 12 of its 13 suppressions, and the live corpus today carries
    `autodom_start` beside `automod_start`. Both modules answer a question they
    were never asked about a name neither table holds: `tool_tier()` returns 1
    because tier 1 is the empty case, and `side_effecting()` returns True because
    "an unlisted tool counts as side-effecting" is its stated safe default. So a
    typo prints as a tier-1 ledgered writer with real traffic.

    The marker cannot mean "fake", and this test pins that it does not: the real
    in-vault writers this census exists to mine (`backlog_write_task`, 251 calls
    against 249 ledger rows at the 2026-09-23T23:41Z probe the note records) are
    unclassified too, because
    every tier-1 tool is. What the marker separates is a decision from a
    default, which is also the grant gate's blind spot — hence #1056 step 3's
    demand that any new gate say what it does about tier 1. Dropping the row is
    not the fix either (silently absent is what the wider row set prevents), and
    separating an invented name from a tier-1 writer would need the aggregator's
    registry, which neither owning table consults and neither does this census.
    Every label here is still read from the two owning tables:
    `classification_source` quotes them and adds no third list of names.
    """
    d = tmp_path / "sessions"
    d.mkdir()
    _transcript(d, "typos", [("invented_writer", _STAMP),
                             ("vault_write", _STAMP)])
    rows = _rows(CENSUS, d, db)
    typo = rows["invented_writer"]
    assert typo["class_source"] == CENSUS.UNCLASSIFIED
    assert typo["tier"] == 1 and typo["tier"] == policy.tool_tier("invented_writer")
    assert typo["ledgered"] is True          # the default, not a judgement
    assert typo["transcript_calls"] == 1     # still counted, never hidden
    # The same answer for a tool that is unambiguously real, so the marker can
    # never be read as "this tool does not exist".
    assert rows["backlog_write_task"]["class_source"] == CENSUS.UNCLASSIFIED
    # A name a table does claim says which one.
    assert rows["vault_write"]["class_source"] == "annotations.IDEMPOTENT"
    assert rows["email_send"]["class_source"] == "policy.TIER2_TOOLS"
    assert rows["email_delete"]["class_source"] == "policy.TIER3_TOOLS"
    CENSUS.main(["--sessions-dir", str(d), "--db", str(db)])
    out = capsys.readouterr().out
    typo_row = next(line for line in out.splitlines()
                    if line.startswith("invented_writer"))
    named_row = next(line for line in out.splitlines()
                     if line.startswith("vault_write"))
    assert "unclassified" in typo_row, typo_row
    assert "unclassified" not in named_row, named_row
    assert "`unclassified` in WHY means NO table names this tool" in out


def test_the_whole_gated_surface_is_listed_even_with_no_traffic(sessions, db):
    """A table that listed only observed tools would print absence as zero traffic.

    That is the exact misreading #590, #591 and #708 each have to be immune to:
    "no mail rows" is not "no mail", and "no tier-3 rows" is not "tier 3 does
    not exist". Every tier-2 and tier-3 name is a row whatever the corpus did.
    """
    rows = _rows(CENSUS, sessions, db)
    for tool in policy.TIER3_TOOLS | policy.TIER2_TOOLS:
        assert tool in rows, tool
    assert rows["email_empty_trash"]["tier"] == 3
    assert rows["email_send"]["tier"] == 2
    # `Bash` and the vault writers print tier 1, and this row asks the NAME-level
    # question: tier 1 is whatever is in NEITHER table (`TIER2_TOOLS` /
    # `TIER3_TOOLS` in `app/harness/policy.py`), and `Bash` is kept out of both on
    # purpose because gating the name would deny every worker for running `ls`.
    # Since #740 `Bash` ALSO has a per-command tier, decided by `app/harness/
    # safety.py`'s durable-external table and reachable only with the arguments in
    # hand — so this row is NOT a claim that a `git push` from an unattended scope
    # goes ungated, and the two assertions below are what keep that split from
    # being lost. No line numbers quoted: the four this comment carried rotted
    # inside one round of edits to `policy.py`. Both stay visible as such: #1056
    # step 3 requires any new gate to say what it does about them.
    assert rows["Bash"]["tier"] == 1
    assert policy.tool_tier("Bash") == 1
    assert policy.tool_tier("Bash", {"command": "git push origin main"}) == 2
    assert rows["vault_write"]["tier"] == 1


def test_read_only_tools_are_not_censusd(sessions, db):
    """An observation has no side effect to census, so `email_read` gets no row.

    It is still in the scan — the census counts it to show how live the read path
    is beside an outbound class at zero — but it is not on the consequential
    surface the gate items are about.
    """
    rows = _rows(CENSUS, sessions, db)
    assert "email_read" not in rows
    assert "email_recent" not in rows
    assert CENSUS.scan_transcripts(
        sessions, _UTC - timedelta(days=30), _UTC)["by_tool"]["email_recent"] == 1


# ---------------------------------------------------------------- clause 4
#: The note lives in the vault, which is not part of any code tree, so it is found
#: by candidate rather than by assumption: an explicit override for a reviewer who
#: wants to point the test at one note, the live checkout's sibling `~/obsidian`
#: (the shape `tests/test_system_health_check_vault_sync.py` uses), and the real
#: account home — `conftest` repoints `HOME` at a tmpdir, so `Path.home()` alone
#: would skip even on the box that has the note. Missing everywhere means no vault
#: here, which is why this node skips rather than failing a candidate for the
#: contents of a tree it was never given.
NOTE_CANDIDATES = [
    Path(os.environ["LLOYD_KNOWLEDGE_NOTE"]) if "LLOYD_KNOWLEDGE_NOTE" in os.environ
    else None,
    ROOT.parent / "obsidian" / "knowledge" / "software" / "side-effect-traffic-census.md",
    Path(pwd.getpwuid(os.getuid()).pw_dir) / "obsidian" / "knowledge" / "software"
    / "side-effect-traffic-census.md",
]
VAULT_NOTE = next((c for c in NOTE_CANDIDATES if c is not None and c.exists()),
                  NOTE_CANDIDATES[-1])


def test_the_knowledge_note_names_a_corpus_source_per_class():
    """Outbound is planted-only and labelled planted; in-vault mines real volume.

    Read from the vault when it is present and skipped when it is not, the way
    every vault-touching test in this suite does (`tests/board_presence.py`,
    `tests/test_prompt_surface_budget.py`) — a candidate tree carries no
    `~/obsidian`, and a candidate must not be failed for a fact about the box it
    was built on.
    """
    if not VAULT_NOTE.exists():
        pytest.skip("no vault knowledge note in this tree")
    text = VAULT_NOTE.read_text(encoding="utf-8")
    lowered = " ".join(text.split()).lower()      # wrapped prose must still match
    assert "planted only" in lowered
    assert "labelled" in lowered                      # planted cases stay labelled
    for tool in ("backlog_write_task", "fact_add"):
        assert tool in text, tool                     # the mineable in-vault volume
    # The single real outbound event is cited, and the note states whether the
    # transcript behind it is still on the box rather than implying it is.
    assert "20260826_031023_ivda08" in text
    assert "email_reply" in text and "rssc" in lowered
    # And it must not become the fourth tier table #1056 step 3 forbids.
    assert "single tier source" in lowered
