#!/usr/bin/env python3
"""#1056 — the side-effect traffic census: a measured denominator per tool.

    cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.side_effect_traffic_census
    ... --days 14                              # shorter window
    ... --json                                 # the same table, machine-readable
    ... --sessions-dir <dir> --db <sqlite>     # point it at a fixture

With no flags it reads the running system's corpus: `<production data root>/
sessions/*.json` and the `tool_effects` table in `<production data root>/
workers.db`, which on this box is `~/lloyd-data/` (`app/data_root.py`). The
literal name of that root is deliberately not spelled anywhere in this file:
`tests/test_no_runtime_paths_in_code.py` refuses any tracked code carrying a
runtime path built off the code tree, because prose naming such a path is how a
script ends up reading a second, silent copy of the data. The pre-move roots —
the code tree's own `sessions/` directory and its `workers.db`, which every
early #1056 hand grep used — are gone, and pointing at them is the failure this
default closes.

Why this exists (#1056)
-----------------------
Every dispatch-time side-effect gate on the board (#590, #591, #708, and the
surviving half of #593) accepts against a corpus of *real recorded*
high-consequence calls. Triage measured that corpus by hand and found the
outbound/irreversible class holds one event in four weeks of retained
transcripts, while the #544 effect ledger cannot hold it at all. Those numbers
were a dashboard snapshot: re-stating them meant re-running a dozen greps by
hand, and every figure in them roughly doubled within five days. This is the
artifact that makes the denominator citable in one call.

What it prints
--------------
One row per tool on the consequential surface, carrying BOTH sources over one
window: the transcript call count and the `tool_effects` row count. Each row
also carries the two labels that decide whether either number means anything —
`app.harness.policy.tool_tier()` and `agent_mcp.annotations.side_effecting()` —
because a zero from a source that structurally cannot record the tool is not a
measurement of that tool.

Why the ledger label is the point
---------------------------------
`side_effecting()` returns False for every `IDEMPOTENT` name, and that table's
own definition names "a delete, a setter" as having no second effect. So
`email_delete`, `email_empty_trash`, `calendar_delete_event` and
`contacts_delete` can NEVER appear in `tool_effects`, and `Edit`, `Write` and
`vault_write` stopped appearing there on 2026-09-10 when `e58d2f5` landed that
exclusion. A per-tool row count printed without that label reports both kinds of
structural absence as though traffic had been measured and found at zero, which
is the zero-denominator failure this repo has already catalogued twice. Hence:
the label is on the row, and the print ends with the denominators every count
is a fraction of.

Determinism, and why no clock is read
-------------------------------------
The window's end is the newest record in the corpus, not `datetime.now()`. Two
runs over an unchanged corpus print byte-identical output, so a table quoted
into an item cannot silently disagree with the next run of the same command,
and a test can assert the exact numbers.

Exit codes
----------
  0  measured — both sources readable, at least one record inside the window.
  2  empty window — an all-zero table produced over zero scanned records is a
     statement about the window, not about the traffic, and must never be
     cited as "the outbound class is empty".
  3  a source cannot be read at all: the sessions directory is missing, the db
     is missing or not a database, or the database has no `tool_effects` table.
     That last state is live whenever the ledger has not yet taken a scoped
     write since its file was created, and it is the reason a ledger cell reads
     `n/a` rather than `0`.

What this file deliberately does NOT contain
--------------------------------------------
No consequence-tier table and no corpus-quota constant. Three classifications
already exist and disagree on the same tools — `policy.py` TIER2/TIER3 (feeds
the grant gate), `annotations.py` READ_ONLY/DESTRUCTIVE/IDEMPOTENT/
REPEAT_EXPECTED (feeds plan mode and the effect ledger), and `safety.py`'s Bash
patterns — so a fourth would be a fourth place for the same tool to be wrong.
Every tier and every ledger label below is read from the two tables that own
them; the tool *universe* is those tables plus whatever the corpus actually
calls, so an unlisted tool still gets a row the moment it is observed.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent_mcp import annotations as tool_annotations  # noqa: E402
from app.data_root import production_data_root  # noqa: E402
from app.harness import policy  # noqa: E402

#: The running system's corpus, whatever tree this census was invoked from.
#: Two failures are closed by naming the root this way, and both were live in
#: this file's first drafts: `Path.home() / "lloyd"` is the PRE-move root (the
#: data home left the code tree on 2026-09-22, so both defaults pointed at
#: directories that no longer exist and the census exited 3 over two absent
#: sources while the real transcripts sat one directory over), and
#: `app.paths.SESSIONS_DIR` follows the *calling process's* root, so from a
#: round's worktree it is that worktree's empty `.lloyd-data/sessions` — a
#: census over zero transcripts that prints `status=ok` and exit 0, which is
#: the empty-window failure this artifact exists to refuse.
#: `production_data_root()` is read off `pwd`, never `$HOME`, for the same
#: reason: under a gate's repointed home `~/lloyd-data` is the round's empty
#: directory. `eval/run_skill_dispatch_probe.py:40-41` made this choice first
#: for the same corpus, and `app/uptake.py:276-291` for the logs. The name is
#: read from `app.data_root` (stdlib-only, no filesystem write of its own)
#: rather than `app.paths.SESSIONS_DIR`, whose value follows the calling
#: process; `app.paths` is still imported transitively here through
#: `app.harness.policy`, so the worktree warning that import prints is this
#: file's noise to ignore, not a bug the defaults fix.
DEFAULT_SESSIONS_DIR = production_data_root() / "sessions"
#: Honours the ledger's own override so a test or a rebuild can point the census
#: at the file `agent_mcp._tool_effects.db_path()` would have written.
LEDGER_DB_ENV = "LLOYD_EFFECT_LEDGER_DB"
DEFAULT_DB = production_data_root() / "workers.db"

LEDGER_TABLE = "tool_effects"

EXIT_MEASURED = 0
EXIT_EMPTY_WINDOW = 2
EXIT_SOURCE_UNREADABLE = 3

#: The annotations table that excluded a tool from the ledger, in the order
#: `side_effecting()` itself consults. Names read off `annotations`, never
#: restated: this is the reason a row is unledgered, printed so a reader can
#: check it rather than take it on faith.
LEDGER_EXCLUSIONS = ("READ_ONLY", "IDEMPOTENT", "REPEAT_EXPECTED")


def default_db_path() -> Path:
    raw = os.environ.get(LEDGER_DB_ENV)
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_DB


def parse_stamp(raw: Any) -> datetime | None:
    """One transcript/ledger timestamp to an aware datetime, or None.

    The two sources disagree by construction and the census has to pick a
    reading: ledger rows carry an offset (`_tool_effects._now()` at `:192` is
    `datetime.now(timezone.utc).isoformat(timespec="seconds")`), while every
    transcript stamp is NAIVE because the session store writes
    `datetime.now().isoformat()` — the tool-call rows this census counts get
    theirs from the `timestamp=` argument of a `_tool_pair(...)` call,
    `app/routers/messages.py:1222-1225`.
    A naive stamp is therefore read as *local*
    time, which is the only reading that agrees with its writer; treating it as
    UTC shifts the whole transcript corpus by the box's offset (here 7-8 hours),
    which is the same skew this repo already logged for `git log --since`,
    `find -newermt` and `ALERT.md`'s `written:` field.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.astimezone()
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _message_calls(msg: dict) -> list[str]:
    """Tool names called by one message, each counted ONCE.

    The store writes a call twice over — `tool_calls[].function.name` for the
    model transcript and a matching `content[].tool_use` block for the result
    plumbing — so joining the two shapes by call id is what keeps a message that
    asked for one `backlog_write_task` from being censused as two. When a shape
    carries no id to match on, only the message's primary shape is used.
    """
    out: list[str] = []
    claimed: set[str] = set()
    calls = msg.get("tool_calls")
    for call in (calls if isinstance(calls, list) else []):
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        name = fn.get("name") if isinstance(fn, dict) else call.get("name")
        if isinstance(call.get("id"), str):
            claimed.add(call["id"])
        if isinstance(name, str) and name:
            out.append(policy.normalize_tool_name(name))
    content = msg.get("content")
    for block in (content if isinstance(content, list) else []):
        if not (isinstance(block, dict) and block.get("type") == "tool_use"
                and isinstance(block.get("name"), str) and block["name"]):
            continue
        block_id = block.get("id")
        if not isinstance(calls, list) or (isinstance(block_id, str)
                                           and block_id not in claimed):
            out.append(policy.normalize_tool_name(block["name"]))
    return out


def _call_records(doc: dict) -> list[tuple[str, Any]]:
    """(normalised tool name, raw stamp) for every tool *call record* in a transcript.

    Call records only. A tool RESULT whose body quotes `{"name": "email_send"}`
    — a skill file, this item's own text pasted into a prompt — is narrative, and
    counting it is how a hand grep over a corpus inflates the number it then
    reports as traffic. A `--days` census whose counts can be raised by a model
    repeating a tool name is not a denominator.
    """
    out: list[tuple[str, Any]] = []
    fallback = doc.get("created_at") or doc.get("last_active")
    messages = doc.get("messages")
    for msg in (messages if isinstance(messages, list) else []):
        if not isinstance(msg, dict):
            continue
        stamp = msg.get("timestamp") or msg.get("created_at") or fallback
        for name in _message_calls(msg):
            out.append((name, stamp))
    return out


def scan_transcripts(sessions_dir: Path, start: datetime, end: datetime) -> dict:
    """Call counts per tool over `[start, end]`, plus the scan's own denominators."""
    out: dict[str, Any] = {
        "dir": str(sessions_dir), "status": "ok", "files_found": 0, "files_parsed": 0,
        "files_skipped_no_calls": 0, "parse_errors": 0, "files_in_window": 0,
        "calls_in_window": 0, "calls_out_of_window": 0, "calls_missing_stamp": 0,
        "by_tool": Counter(), "all_by_tool": Counter(),
        "first_record": None, "last_record": None,
    }
    if not sessions_dir.is_dir():
        out["status"] = "absent" if not sessions_dir.exists() else "not a directory"
        return out
    for path in sorted(sessions_dir.glob("*.json")):
        out["files_found"] += 1
        try:
            doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            out["parse_errors"] += 1
            continue
        if not isinstance(doc, dict):
            out["parse_errors"] += 1
            continue
        out["files_parsed"] += 1
        records = _call_records(doc)
        if not records:
            out["files_skipped_no_calls"] += 1
            continue
        hit = False
        for name, raw in records:
            # Every observed call joins the row set, in or out of the window: a
            # table whose rows vanish when `--days` shrinks makes a shortened
            # window read as "this tool is not part of the surface".
            out["all_by_tool"][name] += 1
            ts = parse_stamp(raw)
            if ts is None:
                out["calls_missing_stamp"] += 1
                continue
            if start <= ts <= end:
                hit = True
                out["calls_in_window"] += 1
                out["by_tool"][name] += 1
                first, last = out["first_record"], out["last_record"]
                if first is None or ts < parse_stamp(first):
                    out["first_record"] = _iso(ts)
                if last is None or ts > parse_stamp(last):
                    out["last_record"] = _iso(ts)
            else:
                out["calls_out_of_window"] += 1
        if hit:
            out["files_in_window"] += 1
    return out


def scan_ledger(db: Path, start: datetime, end: datetime) -> dict:
    """`tool_effects` row counts per tool over `[start, end]`, read-only.

    Opened `mode=ro`: a census that writes to the ledger it measures would be a
    side effect inside its own denominator. `rows_total` is counted before the
    window filter so a window that excludes everything is visible as that, and
    `status` distinguishes the three ways a ledger contributes nothing — the
    file is absent, the file is not a database, or the table has not been
    created — none of which is a measured zero.
    """
    out: dict[str, Any] = {
        "path": str(db), "status": "ok", "rows_total": 0, "rows_in_window": 0,
        "rows_unparseable_stamp": 0, "by_tool": Counter(), "all_by_tool": Counter(),
        "first_record": None, "last_record": None,
    }
    if not db.exists():
        out["status"] = "absent"
        return out
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        out["status"] = f"unreadable: {type(exc).__name__}"
        return out
    try:
        tables = {row[0] for row in conn.execute(
            "select name from sqlite_master where type = 'table'")}
        if LEDGER_TABLE not in tables:
            out["status"] = "no table"
            return out
        out["rows_total"] = int(conn.execute(
            f"select count(*) from {LEDGER_TABLE}").fetchone()[0])
        for tool, raw in conn.execute(f"select tool, created_at from {LEDGER_TABLE}"):
            name = policy.normalize_tool_name(tool)
            # Same reason as the transcript scan: a tool with rows only outside
            # the window — every `vault_write` row stops at 2026-09-10 when
            # `e58d2f5` landed the IDEMPOTENT exclusion — must still get a row,
            # so its zero reads as the window's, not the ledger's whole history.
            out["all_by_tool"][name] += 1
            ts = parse_stamp(raw)
            if ts is None:
                out["rows_unparseable_stamp"] += 1
                continue
            if start <= ts <= end:
                out["rows_in_window"] += 1
                out["by_tool"][name] += 1
                first, last = out["first_record"], out["last_record"]
                if first is None or ts < parse_stamp(first):
                    out["first_record"] = _iso(ts)
                if last is None or ts > parse_stamp(last):
                    out["last_record"] = _iso(ts)
    except sqlite3.Error as exc:
        out["status"] = f"unreadable: {type(exc).__name__}"
        out["rows_total"] = 0
        out["rows_in_window"] = 0
        out["by_tool"] = Counter()
        out["all_by_tool"] = Counter()
        out["first_record"] = out["last_record"] = None
    finally:
        conn.close()
    return out


def ledger_exclusion(name: str) -> str | None:
    """Which `annotations` table keeps this tool out of the ledger, if any.

    Reports the table, not a verdict of this file's: `side_effecting()` is the
    classifier and it is consulted first, so a disagreement here is loud.
    """
    for table in LEDGER_EXCLUSIONS:
        if name in getattr(tool_annotations, table):
            return table
    return None


#: Every table that can name a tool, in the order `classification_source`
#: reports them in. Read off the two owning modules, so the set of owners is
#: theirs and not this file's.
ANNOTATION_TABLES = ("READ_ONLY", "DESTRUCTIVE", "IDEMPOTENT", "REPEAT_EXPECTED")

#: `classification_source`'s answer for a name no table claims.
UNCLASSIFIED = "unclassified"


def classification_source(name: str) -> str:
    """Which owning table, if any, names this tool — or `UNCLASSIFIED`.

    The census's row set is wider than the tables, because a tool the corpus
    actually called must never be silently absent. The marker says which rows
    carry a decision and which carry a default: `tool_tier()` returns 1 because
    tier 1 is the empty case, and `side_effecting()` returns True because "an
    unlisted tool counts as side-effecting" is its stated safe default, so for a
    name in no table both printed cells are what the modules do when nobody has
    classified anything. That covers the whole tier-1 writer class —
    `backlog_write_task`, `fact_add`, `vault_write`'s tier-1 cousins — AND the
    contamination #1056's sibling #768 measured, where a `fake_writer` row
    written by a test that ran without the ledger override held 12 of the
    production ledger's 13 suppressions, and the mis-spelled `autodom_start` the
    live corpus carries beside `automod_start`. The census cannot tell those
    apart, because nothing in either table can: separating an invented name
    from a tier-1 writer needs the aggregator's own registry, which neither
    owner consults and neither does this file. What it does is refuse to let a
    default read as a classification — the grant gate has the same blind spot,
    and #1056 step 3 asks any new gate to say what it does about tier 1.
    """
    if name in policy.TIER3_TOOLS:
        return "policy.TIER3_TOOLS"
    if name in policy.TIER2_TOOLS:
        return "policy.TIER2_TOOLS"
    for table in ANNOTATION_TABLES:
        if name in getattr(tool_annotations, table):
            return f"annotations.{table}"
    return UNCLASSIFIED


def on_census_surface(name: str) -> bool:
    """Is this tool part of the consequential surface the census exists to measure?

    Decided from the owning tables, never from a list written here. Tier 2/3 or
    `DESTRUCTIVE` puts a name on the surface whatever the corpus did — that is
    what makes "the outbound class reads 0" a measurement rather than an
    absence, and it is why `autonomy_delete_task` stays a row even though
    `annotations` files it `READ_ONLY` while `policy.py` puts it in tier 3.
    `READ_ONLY` takes a name off only when no consequence table claims it: an
    observation has no side effect to census, and its volume still shows in the
    transcript denominators.
    """
    if policy.tool_tier(name) >= 2 or name in tool_annotations.DESTRUCTIVE:
        return True
    return name not in tool_annotations.READ_ONLY


def reported_tools(transcripts: dict, ledger: dict) -> list[str]:
    """The census's row set: the consequential tables, plus anything observed.

    `TIER2`/`TIER3` are the surface the grant gate exists for. `DESTRUCTIVE` and
    `IDEMPOTENT` are included because `IDEMPOTENT` is exactly the set the effect
    ledger structurally refuses — a census that omitted it could not report the
    structural absence it exists to distinguish. Anything the corpus actually
    calls gets a row too, which is where the tier-1 writers (`vault_write`,
    `memory_add`, `fact_add`, `session_inject_context`) come from: an
    unclassified tool counts as side-effecting upstream
    (`annotations.side_effecting`), so it must show up here rather than be
    silently absent.
    """
    names: set[str] = set(policy.TIER2_TOOLS) | set(policy.TIER3_TOOLS)
    names |= set(tool_annotations.DESTRUCTIVE) | set(tool_annotations.IDEMPOTENT)
    names |= set(transcripts["all_by_tool"]) | set(ledger["all_by_tool"])
    return sorted(n for n in names if on_census_surface(n))


def build_census(*, sessions_dir: Path, db: Path, days: int) -> dict:
    """The whole census: window, both scans, one row per tool, and a verdict."""
    if days < 1:
        raise ValueError("--days must be >= 1")
    sessions_dir = Path(sessions_dir).expanduser()
    db = Path(db).expanduser()

    # The window's end is the corpus's own newest record. Find it first, with an
    # unbounded pass over stamps cheap enough to run twice (the files are read
    # once per scan either way, so this bounds the scans rather than doubling
    # their cost).
    probe_end = _corpus_end(sessions_dir, db)
    if probe_end is None:
        end = datetime.now(timezone.utc)
        window_boundable = False
    else:
        end = probe_end
        window_boundable = True
    start = end - timedelta(days=days)

    transcripts = scan_transcripts(sessions_dir, start, end)
    ledger = scan_ledger(db, start, end)

    t_ok = transcripts["status"] == "ok"
    l_ok = ledger["status"] == "ok"
    rows = []
    for name in reported_tools(transcripts, ledger):
        ledgered = bool(tool_annotations.side_effecting(name))
        rows.append({
            "tool": name,
            "tier": policy.tool_tier(name),
            "class_source": classification_source(name),
            "ledgered": ledgered,
            "ledger_exclusion": None if ledgered else ledger_exclusion(name),
            "transcript_calls": int(transcripts["by_tool"].get(name, 0)),
            "ledger_rows": int(ledger["by_tool"].get(name, 0)) if l_ok else None,
            # A zero the source cannot produce is not evidence about the tool,
            # so each cell says which kind of zero it is: a measurement, a
            # structural absence the classifier guarantees, or unreadable.
            "transcript_zero_kind": "measured" if t_ok else "unavailable",
            "ledger_zero_kind": ("unavailable" if not l_ok
                                 else "measured" if ledgered else "structural"),
        })

    reasons: list[str] = []
    if not t_ok:
        reasons.append(f"sessions dir {transcripts['dir']}: {transcripts['status']}")
    if not l_ok:
        reasons.append(f"ledger db {ledger['path']}: {ledger['status']}")
    if reasons:
        verdict, code = "source-unreadable", EXIT_SOURCE_UNREADABLE
    elif not transcripts["calls_in_window"] and not ledger["rows_in_window"]:
        verdict, code = "empty-window", EXIT_EMPTY_WINDOW
        reasons.append(
            f"0 of {transcripts['files_found']} transcript files and 0 of "
            f"{ledger['rows_total']} ledger rows fall inside "
            f"{_iso(start)} -> {_iso(end)}; the all-zero table describes this "
            "window, not the traffic")
    else:
        verdict, code = "measured", EXIT_MEASURED

    return {
        "window": {"days": days, "start": _iso(start), "end": _iso(end),
                   "end_from_corpus": window_boundable},
        "denominators": {"transcripts": _plain(transcripts), "ledger": _plain(ledger)},
        "rows": rows,
        "verdict": verdict,
        "exit_code": code,
        "unreadable_reasons": reasons,
    }


def _corpus_end(sessions_dir: Path, db: Path) -> datetime | None:
    """The newest record stamp in either source, so the window needs no clock.

    An empty or unreadable corpus yields None and the caller falls back to
    wall-clock now purely so the scans still run and report *why* they are
    empty; the verdict then says the window was not corpus-bounded.
    """
    best: datetime | None = None
    if sessions_dir.is_dir():
        for path in sorted(sessions_dir.glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict):
                continue
            for _, raw in _call_records(doc):
                ts = parse_stamp(raw)
                if ts is not None and (best is None or ts > best):
                    best = ts
    if db.exists():
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except sqlite3.Error:
            conn = None
        if conn is not None:
            try:
                tables = {row[0] for row in conn.execute(
                    "select name from sqlite_master where type = 'table'")}
                if LEDGER_TABLE in tables:
                    for (raw,) in conn.execute(
                            f"select created_at from {LEDGER_TABLE}"):
                        ts = parse_stamp(raw)
                        if ts is not None and (best is None or ts > best):
                            best = ts
            except sqlite3.Error:
                pass
            finally:
                conn.close()
    return best


def _plain(d: dict) -> dict:
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in d.items()}


def format_table(census: dict) -> str:
    """The human report. Fixed column order, no clock, so re-runs are diffable."""
    rows = sorted(census["rows"], key=lambda r: (-r["tier"], r["tool"]))
    tr_den = census["denominators"]["transcripts"]
    led_den = census["denominators"]["ledger"]
    w = census["window"]
    ledger_ok = led_den["status"] == "ok"
    transcript_ok = tr_den["status"] == "ok"

    lines = [
        "SIDE-EFFECT TRAFFIC CENSUS  (#1056)",
        f"  window      {w['start']} -> {w['end']}  ({w['days']} days"
        + (", end = newest record in the corpus)" if w["end_from_corpus"]
           else ", end = wall clock: the corpus has no usable record)"),
        "",
        f"{'TOOL':<26}{'TIER':>5}{'LEDGER':>15}{'TRANSCRIPT_CALLS':>18}"
        f"{'LEDGER_ROWS':>13}  WHY",
    ]
    for r in rows:
        calls = str(r["transcript_calls"]) if transcript_ok else "n/a"
        led = str(r["ledger_rows"]) if ledger_ok and r["ledger_rows"] is not None else "n/a"
        why = " ".join(p for p in (
            r["ledger_exclusion"],
            UNCLASSIFIED if r["class_source"] == UNCLASSIFIED else "") if p)
        lines.append(f"{r['tool']:<26}{r['tier']:>5}{'ledgered' if r['ledgered'] else 'not ledgered':>15}"
                     f"{calls:>18}{led:>13}  {why}")

    lines += [
        "",
        "DENOMINATORS  (every count above is a fraction of these)",
        f"  transcripts  dir={tr_den['dir']} status={tr_den['status']}",
        f"               files_found={tr_den['files_found']} parsed={tr_den['files_parsed']} "
        f"parse_errors={tr_den['parse_errors']} "
        f"files_with_calls_in_window={tr_den['files_in_window']}",
        f"               calls_in_window={tr_den['calls_in_window']} "
        f"calls_out_of_window={tr_den['calls_out_of_window']} "
        f"calls_missing_stamp={tr_den['calls_missing_stamp']}",
        f"               first_record={tr_den['first_record']} "
        f"last_record={tr_den['last_record']}",
        f"  ledger       db={led_den['path']} status={led_den['status']}",
        f"               rows_total={led_den['rows_total']} "
        f"rows_in_window={led_den['rows_in_window']} "
        f"rows_unparseable_stamp={led_den['rows_unparseable_stamp']}",
        f"               first_record={led_den['first_record']} "
        f"last_record={led_den['last_record']}",
        "",
        "FOOTNOTES",
        "  WHY names the annotations table that excludes the tool from the ledger.",
        "  `unclassified` in WHY means NO table names this tool, so its TIER (1)",
        "  and LEDGER (ledgered) cells are both modules' DEFAULT for an unknown",
        "  name, not a decision about it. That is every tier-1 writer as well as",
        "  any mis-spelled or invented name — the census cannot separate them and",
        "  neither can the grant gate, which is why #1056 asks any new gate to",
        "  state what it does about tier 1.",
        "  A LEDGER_ROWS of 0 on a `not ledgered` row is STRUCTURAL: the ledger",
        "  cannot record that tool at all, which is not evidence that the tool is",
        "  unused. Read TRANSCRIPT_CALLS for those tools, and treat 0 there as",
        "  measured absence only while status=ok above.",
        "  `n/a` means the source could not be read this run — never a zero.",
        "  TIER comes from app/harness/policy.tool_tier; LEDGER from",
        "  agent_mcp.annotations.side_effecting. This census authors neither.",
    ]
    if census["unreadable_reasons"]:
        lines.append("")
        lines.append(f"VERDICT  {census['verdict']} (exit {census['exit_code']})")
        for reason in census["unreadable_reasons"]:
            lines.append(f"  - {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS_DIR),
                    help=f"transcript directory (default {DEFAULT_SESSIONS_DIR})")
    ap.add_argument("--db", default=str(default_db_path()),
                    help=f"effect-ledger sqlite file (default ${LEDGER_DB_ENV} or "
                         f"{DEFAULT_DB})")
    ap.add_argument("--days", type=int, default=30,
                    help="window length, counted back from the newest record (default 30)")
    ap.add_argument("--json", action="store_true",
                    help="emit the census as JSON instead of the table")
    args = ap.parse_args(argv)

    census = build_census(sessions_dir=Path(args.sessions_dir), db=Path(args.db),
                          days=args.days)
    if args.json:
        print(json.dumps(census, indent=2, sort_keys=True))
    else:
        print(format_table(census))
    return int(census["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
