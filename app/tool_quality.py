"""How often a tool call that did NOT fail returned nothing (#2275).

Lloyd's whole failure surface is error-shaped: `runs.status`, `stats.is_error`,
the guardian's read of `logs/server.err`, #690's error-baseline filter. A call
that succeeds and hands the model an empty answer is invisible to every one of
those, and that is the shape this box has been cut by more than once:
`app/inner_voice/guards.py` carries the comment that the primary read an empty
result as a broken query and changed what it did next, and the standing class
rule "a 0-hit grep needs a positive control, and a grep against a path that does
not exist returns 0" (2026-09-16/17) is prose nobody can enforce because nobody
could see the rate. This module turns that prose into a number. It counts two
things over calls that carry **no error flag** — a zero-result answer, and a
call whose every argument was null — per tool, per day, each printed beside its
denominator.

**Report-only.** Nothing is gated on these rates, and nothing here should ever
be wired as a gate: a zero-result is frequently the *correct* answer (a grep
over a corpus that genuinely lacks the string), and the standing rule tells the
model to treat it as one. The instrument reports rates for investigation. If a
week passes with no rate investigated in writing, delete the sweep (#543
close-out discipline).

**Two vocabulary decisions, both measured on this tree, not guessed.** The
emptiness test is *not* `stats.result_chars` near zero:
`fallback_for_empty_result` (`app/harness/tool_result_spill.py`) runs at dispatch,
before the row
is built, so a literally-empty answer is persisted as
`"(Read completed with no output)"` with `result_chars` 31 — measured over 2,147
tool rows in the 60 newest sessions, `result_chars == 0` matched **0** rows. And
`stats.raw_chars` alone under-counts silently: it is *absent*, not zero, on every
reconstructed path (`app/routers/messages.py::_tool_pair` with `evt=None`,
`app/run_recorder.py::_unpersisted_pairs`) by #1052's deliberate choice, so
`_tool_pair`'s own rows with marker text survive only through the text test. The
three tests below are therefore ORed: the harness's `(no matches)` string
(`agent_mcp/builtin_fs.py`, which `Grep`/`Glob` return for a zero hit — 12
non-empty characters, and the single biggest zero-source in the last 7 days),
the `ZERO_RESULT_RE` *… completed with no output* marker, and
`stats.raw_chars == 0`.

**Instant-return is not here.** The third invariant of the workshop detector
this recreates — a tool that "suspiciously returned in one millisecond" — is not
derivable from a transcript: `app/routers/messages.py::_tool_pair` passes ONE
`timestamp` to both `build_tool_call_entry` and `build_tool_result_entry`, and
`app/run_recorder.py` sets `ts = datetime.now().isoformat()` once for the pair,
so 1599/1599 call/result pairs in the 40 newest session JSONs carry identical
timestamps and a per-pair duration is zero by construction. The real value
exists and is dropped: `app/harness/loop.py` puts `duration_ms` on the
`tool_result` event and neither persistence path writes it. Recovering it is a
dispatch-side field on a harness write path — its own round, not an edit here.

**Seams.** This module crosses exactly one process boundary: it is invoked as
`python -m app.tool_quality` by whatever schedules it (owed #3 decides which),
and it reads what two *other* writers leave on disk — the chat path
(`app/routers/messages.py`) and the background-run recorder
(`app/run_recorder.py`). No test of that seam can run here, so the classifier is
built on the row shape `app/transcript_entries.py` *documents*, and
`tests/test_tool_quality_sweep.py` pins it with seeded transcripts that carry
each writer's quirks (marker with `result_chars` 31, marker with no `raw_chars`
key, an errored all-null-argument call) rather than with rows of this module's
own invention.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.atomic_io import atomic_write_text

#: The schema tag of the emitted row set, so a later reader of a dated file
#: knows which invariants produced it. Bump it only in a way that keeps the old
#: files readable — retention is 30 days, so both shapes coexist for a month.
SCHEMA = 1

#: What `Grep` and `Glob` return for a zero hit (`agent_mcp/builtin_fs.py`).
#: Twelve characters, so it is *not* empty by length and not empty by
#: `result_chars`: this string is the whole reason the detector is a vocabulary
#: and not a size.
NO_MATCHES = "(no matches)"

#: What a genuinely zero-length answer looks like *after* the dispatch fallback
#: `fallback_for_empty_result` has had its way with it: `"(Bash completed with no
#: output)"`, `"(Read completed with no output)"`, and one line per tool that can
#: answer with nothing.
ZERO_RESULT_RE = re.compile(r"^\(.* completed with no output\)$")

#: The five fields one per-tool row carries. Kept to these so a reader of the
#: emitted JSON can tell a count from a rate at a glance: counts are integers,
#: and every rate is added by the sweep as a `*_rate` string (see `rate`).
ROW_KEYS = ("tool", "calls", "empty", "degenerate_args", "degenerate_errored")

#: How far back a sweep reads. Seven days is the window the triage measurement of
#: the corpus was taken over (4,446 session files, 91,904 tool calls), so the
#: numbers in this item's acceptance clauses stay comparable run to run.
DEFAULT_DAYS = 7

#: How many dated files the store keeps, including today's. One small JSON per
#: day, so the window is a reader's-usefulness decision, not a disk one: three
#: weeks spans "the week under investigation plus the week before it plus the
#: week that flagged it". The store prunes itself in `sweep` because the weekly
#: groundskeeper sweep has never heard of it — owed #4 registers it there once
#: the window is known, and until then this constant is the only thing bounding
#: it.
RETENTION_DAYS = 30

# Session files are `<YYYYMMDD>_<HHMMSS>_<source>_<suffix>.json`
# (`app/sessions_io.py`). The date is taken from the name rather than mtime for
# the same reason `component_manifest` prunes by name: a transcript is appended
# to whenever its session is resumed, so mtime is "when someone last talked", and
# a restored or clock-skewed mtime cannot be read as the record's age.
_FILE_DATE_RE = re.compile(r"^(\d{8})(?:_|$)")

# This store's own files and nothing else, so a `README.md` or a note a human
# drops beside them is never this module's to delete.
_DAY_FILE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.json$")


# ───────────────────────────── the classifier ──────────────────────────────
def result_text(row: dict) -> str:
    """The text of one `role="tool"` row, joined over its content parts.

    `app/transcript_entries.py::build_tool_result_entry` writes content as a
    list of `{"type": "text", "text": …}` parts; a plain string is accepted
    because a row rebuilt from anything but that builder may hold one, and a
    classifier that crashed on the unexpected shape would report a clean corpus.
    """
    content = row.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
            and isinstance(part.get("text"), str))
    return ""


def is_error_flagged(row: dict) -> bool:
    """True when the result row carries an error flag.

    `build_tool_result_entry` writes `stats.is_error` as a bool on the eager
    paths and omits it entirely on the reconstructed ones (#1052), so absent and
    `False` both mean *unflagged* — which is the interesting population, and the
    one an error-shaped failure surface cannot see.
    """
    stats = row.get("stats")
    return bool(isinstance(stats, dict) and stats.get("is_error"))


def is_zero_result(row: dict) -> bool:
    """True when an unflagged call returned nothing the model could read.

    Three tests, ORed, and deliberately not a fourth one:

    * the text is exactly `"(no matches)"` — `Grep`/`Glob`'s zero hit;
    * the text matches `ZERO_RESULT_RE` — the dispatch fallback's marker, which
      is what an empty answer is *renamed* to, so a length test can never see
      it (`result_chars` is 31 for a 0-character answer, and is therefore never
      consulted here);
    * `stats.raw_chars == 0` — the tool's own answer had zero characters. The
      key must be present to count: it is absent on every reconstructed row,
      where the true size is unknown, and an absent key read as a zero would
      invent the #1052 error in the opposite direction.

    A row with empty text and no `raw_chars` key is therefore NOT counted: that
    is `_unpersisted_pairs`' shape for a call whose result never came back at
    all, which is a missing record, not a tool that answered with nothing.
    """
    text = result_text(row).strip()
    if text == NO_MATCHES or ZERO_RESULT_RE.match(text):
        return True
    stats = row.get("stats")
    raw = stats.get("raw_chars") if isinstance(stats, dict) else None
    return isinstance(raw, int) and not isinstance(raw, bool) and raw == 0


def _is_nullish(value: Any) -> bool:
    """A value that carries nothing: `None`, `""`, `[]` or `{}`."""
    return value is None or value == "" or value == [] or value == {}


def is_degenerate_arguments(call: dict) -> bool:
    """True when every argument in the call is null, empty or absent.

    The workshop's bug was a tool called "with every argument null producing a
    meaningless search", and it looked like success. `function.arguments` is the
    JSON string the model produced: an object with no keys counts (a call made
    with nothing at all), and anything that does not parse as an object does
    not — a parse failure is a malformed call, which is a different defect and
    arrives with an error flag, and counting it here would inflate the
    success-coded rate the clause is about.
    """
    function = call.get("function")
    raw = function.get("arguments") if isinstance(function, dict) else None
    if not isinstance(raw, str):
        return False
    try:
        args = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(args, dict):
        return False
    return all(_is_nullish(value) for value in args.values())


def _messages(doc: Any) -> list[dict]:
    """The message rows of a transcript document, tolerating either shape.

    A session file is a dict with `messages`; a bare list of rows is what the
    builders return, and what a test seeds. Anything else has no rows.
    """
    if isinstance(doc, list):
        return [row for row in doc if isinstance(row, dict)]
    if isinstance(doc, dict):
        rows = doc.get("messages")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _result_index(rows: list[dict]) -> "OrderedDict[str, dict]":
    """`tool_call_id` -> the result row, first writer wins.

    Both persistence paths write a pair at most once per call id
    (`persisted_pairs`, and the router's own pair set), so a second row for the
    same id is a re-write of the same dispatch. Taking the first and ignoring
    the rest keeps one call in the denominator.
    """
    out: "OrderedDict[str, dict]" = OrderedDict()
    for row in rows:
        if row.get("role") != "tool":
            continue
        call_id = row.get("tool_call_id")
        if isinstance(call_id, str) and call_id and call_id not in out:
            out[call_id] = row
    return out


def classify_transcript(doc: Any) -> "dict[str, dict]":
    """Per-tool quality rows for one session-transcript document. Pure.

    Returns `{tool: {"tool", "calls", "empty", "degenerate_args",
    "degenerate_errored"}}` (`ROW_KEYS`), where:

    * `calls` counts a call only when its paired `role="tool"` row is present
      and NOT error-flagged. A call with no result row is a call whose result was
      never persisted — no row, no evidence, and no denominator; an error-flagged
      call is excluded from `calls` *and* from every numerator, because a failure
      already on the error surface is not a success-coded quality failure.
    * `empty` counts the unflagged calls whose result `is_zero_result`.
    * `degenerate_args` counts the unflagged calls whose arguments
      `is_degenerate_arguments`. It is not a subset of `empty` and `empty` is not
      a subset of it: read together they say "nothing in, nothing out"; apart
      they say one thing each.
    * `degenerate_errored` counts all-null-argument calls whose result WAS
      error-flagged. Those are excluded from the rate above on purpose — an
      argument-shape rejection the harness already reported must not also show up
      as a success-coded defect — but they are kept as a count so the exclusion
      is visible rather than a silent subtraction.

    A call id seen twice in one transcript counts once (see `_result_index`).
    """
    rows = _messages(doc)
    results = _result_index(rows)
    out: "dict[str, dict]" = {}
    counted: set[str] = set()

    def row_for(tool: str) -> dict:
        row = out.get(tool)
        if row is None:
            row = {key: 0 for key in ROW_KEYS}
            row["tool"] = tool
            out[tool] = row
        return row

    for row in rows:
        calls = row.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("call_id") or call.get("id")
            if not isinstance(call_id, str) or not call_id:
                continue
            result = results.get(call_id)
            if result is None or call_id in counted:
                continue
            counted.add(call_id)
            function = call.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            tool = name if isinstance(name, str) and name else "(unnamed)"
            degenerate = is_degenerate_arguments(call)
            if is_error_flagged(result):
                if degenerate:
                    row_for(tool)["degenerate_errored"] += 1
                continue
            counted_row = row_for(tool)
            counted_row["calls"] += 1
            if is_zero_result(result):
                counted_row["empty"] += 1
            if degenerate:
                counted_row["degenerate_args"] += 1
    return out


def merge_rows(target: "dict[str, dict]",
               rows: "dict[str, dict]") -> None:
    """Add `rows` into `target` in place, summing every count."""
    for tool, row in rows.items():
        into = target.get(tool)
        if into is None:
            target[tool] = {key: row[key] for key in ROW_KEYS}
            continue
        for key in ROW_KEYS[1:]:
            into[key] += row[key]


# ─────────────────────────────── the positive control ───────────────────────
#: What `control_transcript` is designed to produce, per tool. Every number here
#: is a non-zero count by construction, which is the point: a sweep whose output
#: reports `empty: 0` for every tool and a self-check that no longer matches
#: these numbers has a broken detector, not a clean corpus. That is the standing
#: class rule — "a 0-hit grep needs a positive control" — applied to the
#: instrument itself.
CONTROL_EXPECTED: "dict[str, dict]" = {
    "Grep": {"tool": "Grep", "calls": 5, "empty": 2, "degenerate_args": 1,
             "degenerate_errored": 0},
    "Read": {"tool": "Read", "calls": 2, "empty": 1, "degenerate_args": 0,
             "degenerate_errored": 1},
}


def _seed(call_id: str, tool: str, args: dict, text: str, *,
          stats: "dict | None" = None,
          result_stats: "dict | None" = None) -> list[dict]:
    """One seeded call/result pair in the shape `transcript_entries` writes.

    Hand-built rather than built by importing the builders, so the fixture keeps
    pinning the *on-disk* row shape even if a builder's defaults change — the
    sweep reads files another process wrote, and that is the contract it has to
    hold.
    """
    return [
        {"id": f"msg_{call_id}_tc", "role": "assistant",
         "content": [{"type": "text", "text": ""}],
         "tool_calls": [{"id": call_id, "call_id": call_id,
                         "type": "function",
                         "function": {"name": tool,
                                      "arguments": json.dumps(args)}}],
         "timestamp": "2026-10-06T00:00:00.000000",
         "stats": stats or {"iteration": 1}},
        {"id": f"msg_{call_id}_result", "role": "tool",
         "content": [{"type": "text", "text": text}],
         "tool_call_id": call_id,
         "timestamp": "2026-10-06T00:00:00.000000",
         "stats": result_stats if result_stats is not None
         else {"result_chars": len(text), "raw_chars": len(text),
               "is_error": False}},
    ]


def control_transcript() -> dict:
    """A seeded transcript with a known number of every failure this counts.

    Eight rows: five `Grep` calls and three `Read` calls, two rows per call.
    Designed so that each of the three emptiness tests, and both argument
    branches, has its own row that must fire — and each also has a row beside it
    that must NOT:

    * `Grep` 1 — `"(no matches)"` with `raw_chars` 12: the zero hit a length
      test cannot see. `Grep` 2 — a real match list: the non-empty control.
    * `Grep` 3 — `"(Grep completed with no output)"` (31 characters) with
      `raw_chars` 0: empty by both the marker and the raw size.
    * `Grep` 4 — arguments `{}` and a NON-empty result: the all-null-argument
      call, kept out of the empty count on purpose so the two invariants stay
      independent and one row cannot satisfy both by accident.
    * `Grep` 5 / `Read` 2 — ordinary non-empty calls with a subset of arguments
      populated: the false-positive control, and the reason `Grep` reads
      `2 / 5` rather than something flattering.
    * `Read` 1 — `"(Read completed with no output)"`, `result_chars` 31 and NO
      `raw_chars` key at all: the reconstructed path (#1052), which must still
      count.
    * `Read` 3 — arguments `{"file_path": null, "offset": "", "limit": null}`
      with `is_error: true`: an argument rejection the error surface already
      reports, so it lands in `degenerate_errored` and not in the rate.
    """
    messages: list[dict] = []
    messages += _seed("call_g1", "Grep",
                      {"pattern": "NoSymbolAnywhere", "path": "~/lloyd"},
                      NO_MATCHES)
    messages += _seed("call_g2", "Grep",
                      {"pattern": "EVAL_BASELINES_DIR", "path": "~/lloyd/app"},
                      "app/paths.py:106: EVAL_BASELINES_DIR = DATA_ROOT / \"eval\""
                      " / \"baselines\"\n")
    messages += _seed("call_g3", "Grep", {"pattern": "zzz", "glob": "*.rs"},
                      "(Grep completed with no output)",
                      result_stats={"result_chars": 31, "raw_chars": 0,
                                    "is_error": False})
    messages += _seed("call_g4", "Grep", {},
                      "app/tool_quality.py:1: \"\"\"How often a tool call…\n")
    messages += _seed("call_g5", "Grep", {"pattern": "TODO"},
                      "app/compaction.py:152: thinking — see app/harness\n")
    messages += _seed("call_r1", "Read", {"file_path": "~/lloyd/app/djev.py"},
                      "(Read completed with no output)",
                      # The reconstructed path: `evt=None`, so `raw_chars` is
                      # omitted rather than invented (#1052). `result_chars` is
                      # the marker's own length — 31 — which is exactly the
                      # number that must NOT be read as "not empty".
                      result_stats={"result_chars": 31})
    messages += _seed("call_r2", "Read",
                      {"file_path": "~/lloyd/app/paths.py", "offset": 100,
                       "limit": 20},
                      "SESSIONS_DIR = DATA_ROOT / \"sessions\"\n")
    messages += _seed("call_r3", "Read",
                      {"file_path": None, "offset": "", "summary": None},
                      "Error: file_path is required",
                      result_stats={"result_chars": 27, "raw_chars": 27,
                                    "is_error": True})
    return {"session_id": "tool-quality-self-check", "messages": messages}


def self_check() -> dict:
    """Run the classifier over the seeded transcript and report the verdict.

    Returns `{"expected", "actual", "rates", "ok"}`, where `rates` shows the
    control's numbers in the same `k / N` form the live rows use — so the block
    beside a day of zeroes says the detector still fires. `ok` is the comparison
    against `CONTROL_EXPECTED`; the sweep never gates on it, it just refuses to
    let `empty: 0` read as clean when the control also reads 0.
    """
    rows = classify_transcript(control_transcript())
    actual = {tool: {key: row[key] for key in ROW_KEYS}
              for tool, row in sorted(rows.items())}
    rates = {tool: {"empty": rate(row["empty"], row["calls"]),
                    "degenerate_args": rate(row["degenerate_args"],
                                            row["calls"])}
             for tool, row in sorted(actual.items())}
    return {"expected": {tool: {key: CONTROL_EXPECTED[tool][key]
                                for key in ROW_KEYS}
                         for tool in sorted(CONTROL_EXPECTED)},
            "actual": actual,
            "rates": rates,
            "ok": actual == {tool: {key: CONTROL_EXPECTED[tool][key]
                                    for key in ROW_KEYS}
                             for tool in sorted(CONTROL_EXPECTED)}}


# ─────────────────────────────── rates and totals ───────────────────────────
def rate(count: int, denominator: int) -> str:
    """One count beside the denominator it was taken over. Never a bare count.

    The workshop's headline number ("42% of searches returned zero results") was
    a small-denominator statistic by its own author's account, and Lloyd is one
    user, not a request-per-second service. So a rate that cannot state its N is
    not printed or emitted anywhere in this module, and a clean tool prints
    `0 / 37` — which is a measurement, not an absence.
    """
    return f"{int(count)} / {int(denominator)}"


def tool_rows(rows: "dict[str, dict]", *,
              exclude_zero_denominator: bool = True) -> list[dict]:
    """Emit-ready per-tool rows, sorted worst-first, each rate with its N.

    Rows with no non-error call are dropped by default: with N = 0 there is no
    rate to state, and a table that prints `0 / 0` reads as a measurement of
    nothing. The sweep reports those tools separately (see `sweep`) so their
    `degenerate_errored` count is not lost by being un-rateable.
    """
    out: list[dict] = []
    for tool in sorted(rows, key=lambda t: (-rows[t]["empty"],
                                            -rows[t]["calls"], t)):
        row = {key: rows[tool][key] for key in ROW_KEYS}
        if exclude_zero_denominator and row["calls"] == 0:
            continue
        row["empty_rate"] = rate(row["empty"], row["calls"])
        row["degenerate_args_rate"] = rate(row["degenerate_args"],
                                           row["calls"])
        out.append(row)
    return out


# ───────────────────────────────── the sweep ────────────────────────────────
def transcript_paths(sessions_dir: Path, *, days: "int | None" = None,
                     today: "date | None" = None) -> list[Path]:
    """Session transcripts to read, newest window first.

    `days` keeps the files whose *name*-date is today or up to `days` older —
    inclusive, so `days=7` spans up to eight calendar dates, which is the window
    the corpus figure in #2275 was measured over. A name that does not start with
    a date is outside every window (it is not a session transcript this module
    understands) and shows up only in the report's `files_undated_in_dir`.
    `today` takes an ISO string
    or a `date`; unparseable or absent means the real today.
    """
    root = Path(sessions_dir)
    if not root.is_dir():
        return []
    anchor = _as_day(today) or datetime.now().date()
    paths: list[Path] = []
    for path in sorted(root.glob("*.json")):
        if days is None:
            paths.append(path)
            continue
        match = _FILE_DATE_RE.match(path.name)
        if not match:
            continue
        name_date = _parse_day(match.group(1), fmt="%Y%m%d")
        if name_date is None:
            continue
        if name_date >= anchor - timedelta(days=days):
            paths.append(path)
    return paths


def _parse_day(text: str, *, fmt: str = "%Y-%m-%d") -> "date | None":
    try:
        return datetime.strptime(text, fmt).date()
    except ValueError:
        return None


def _as_day(value: "str | date | None") -> "date | None":
    """Coerce a caller's `today` — an ISO string or a `date` — into a `date`.

    Every public entry point takes either spelling, because `--today` arrives from
    argv as a string and a test passes a `date`. A retention path that raised
    `TypeError` on the string form would be a prune that silently never ran (or,
    in `prune_store`, a function documented never to raise raising) — so the
    coercion lives in one place instead of being remembered per caller.
    """
    if value is None or isinstance(value, date):
        return value
    return _parse_day(str(value))


def load_transcript(path: Path) -> "Any | None":
    """Read one transcript, or `None` if it cannot be read or parsed.

    Read-only, and never raises: a truncated file left by a killed run, or a
    session written by a newer schema, is one row of `files_unreadable` in the
    report, not a crash that loses the whole day's sweep.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def prune_store(out_dir: Path, *, today: "date | None" = None,
                days: int = RETENTION_DAYS) -> "dict[str, int]":
    """Delete this store's dated files older than `days`. Never raises.

    Age is the date in the name, not mtime, for the reason `component_manifest`
    gives for the same decision: the name is the record's age and a clock-skewed
    mtime cannot make an old file look young. Only `^\\d{4}-\\d{2}-\\d{2}\\.json$`
    inside `out_dir` is eligible — a `README.md` or a mis-named file is not this
    store's to delete — and the cutoff is `today - (days - 1)`, so the default
    window keeps `RETENTION_DAYS` dated files including today's. `days <= 0`
    disables pruning: an unbounded store is the safer failure than a mistyped
    window that deletes every measurement.
    """
    out = {"files": 0, "bytes": 0, "errors": 0}
    if days <= 0:
        return out
    store = Path(out_dir)
    if not store.is_dir():
        return out
    anchor = _as_day(today) or datetime.now().date()
    cutoff = anchor - timedelta(days=days - 1)
    try:
        candidates = list(store.iterdir())
    except OSError:
        out["errors"] += 1
        return out
    for path in candidates:
        match = _DAY_FILE_RE.match(path.name)
        if not match:
            continue
        file_date = _parse_day("-".join(match.groups()))
        if file_date is None or file_date >= cutoff:
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            out["files"] += 1
            out["bytes"] += size
        except OSError:
            out["errors"] += 1
    return out


def sweep(*, sessions_dir: "Path | str | None" = None,
          out_dir: "Path | str | None" = None,
          days: int = DEFAULT_DAYS,
          today: "str | date | None" = None,
          retention_days: int = RETENTION_DAYS) -> dict:
    """Read the transcripts, emit one dated row set, prune the store's own past.

    Read-only over the corpus: every input file is opened for reading and left
    byte-for-byte unchanged, and the only writes are the one dated file and the
    deletions `prune_store` makes inside this store's own directory. Returns the
    report it wrote.

    `sessions_dir` and `out_dir` default to `app.paths.SESSIONS_DIR` and
    `app.paths.TOOL_QUALITY_DIR`, resolved *at call time* so a process whose data
    root moved (a test's scratch `LLOYD_DATA`, a round's worktree) does not sweep
    one tree and write another. `today` pins the clock for a caller that wants a
    reproducible file name and cutoff.
    """
    from app import paths as _paths

    root = Path(sessions_dir) if sessions_dir else Path(_paths.SESSIONS_DIR)
    store = Path(out_dir) if out_dir else Path(_paths.TOOL_QUALITY_DIR)
    anchor = _as_day(today) or datetime.now().date()

    rows: "dict[str, dict]" = {}
    scanned = 0
    unreadable = 0
    for path in transcript_paths(root, days=days, today=anchor):
        scanned += 1
        doc = load_transcript(path)
        if doc is None:
            unreadable += 1
            continue
        merge_rows(rows, classify_transcript(doc))
    # Named without a date a window can read: outside every window by
    # construction, so the report states how many files that is instead of
    # leaving the gap between "files in the directory" and "files scanned" for a
    # later reader to explain.
    undated = sum(1 for path in root.glob("*.json")
                  if not _FILE_DATE_RE.match(path.name))

    total_calls = sum(row["calls"] for row in rows.values())
    total_empty = sum(row["empty"] for row in rows.values())
    total_degenerate = sum(row["degenerate_args"] for row in rows.values())
    total_degenerate_errored = sum(row["degenerate_errored"]
                                   for row in rows.values())
    # Tools with no non-error call at all: no denominator, so no rate, and they
    # are NOT in `tools` (see `tool_rows`). `degenerate_errored` is the one thing
    # the classifier keeps about them, and naming the tool is what makes the
    # exclusion auditable rather than silent.
    unrateable = sorted(({"tool": tool,
                          "degenerate_errored": rows[tool]["degenerate_errored"]}
                         for tool in rows if rows[tool]["calls"] == 0),
                        key=lambda item: item["tool"])

    report = {
        "schema": SCHEMA,
        "date": anchor.isoformat(),
        "window": {"days": days,
                   "from": (anchor - timedelta(days=days)).isoformat(),
                   "through": anchor.isoformat(),
                   "sessions_dir": str(root)},
        "corpus": {
            "files_scanned": scanned,
            "files_unreadable": unreadable,
            "files_undated_in_dir": undated,
            "total_calls": total_calls,
            "total_empty": total_empty,
            "total_degenerate_args": total_degenerate,
            "total_degenerate_errored": total_degenerate_errored,
        },
        "corpus_rates": {
            "empty": rate(total_empty, total_calls),
            "degenerate_args": rate(total_degenerate, total_calls),
        },
        # The instrument's own positive control, in the file every day, so a day
        # of zeroes always arrives with the evidence that zero is a measurement
        # and not a detector that stopped firing.
        "self_check": self_check(),
        "tools": tool_rows(rows),
        "tools_without_success_calls": unrateable,
    }

    # The directory first, then the prune, then the write — in that order, so the
    # dated file a reader opens states what the run that wrote it deleted. A
    # prune can never touch today's row set: the cutoff is `today - (days - 1)`,
    # and a same-day file from an earlier run is about to be replaced anyway.
    store.mkdir(parents=True, exist_ok=True)
    report["retention"] = {
        "days": retention_days,
        "pruned_at": anchor.isoformat(),
        "prune": prune_store(store, today=anchor, days=retention_days),
    }
    out_path = store / f"{anchor.isoformat()}.json"
    atomic_write_text(out_path, json.dumps(report, indent=2, sort_keys=False)
                      + "\n")
    # `written_to` is the caller's handle on the file and is not part of the
    # file's own contents: everything else in `report` is what was written.
    report["written_to"] = str(out_path)
    return report


# ─────────────────────────────────── CLI ────────────────────────────────────
def format_report(report: dict, *, top: int = 5) -> str:
    """The printed form: every rate as `k / N`, corpus total first.

    `top` rows worst-first by empties then calls; `top` <= 0 prints every tool
    with at least one non-error call, so a clean tool still prints `0 / N`
    instead of vanishing.
    """
    corpus = report["corpus"]
    # An empty denominator is stated, not hidden: `0 / 0` here means "this sweep
    # read no non-error call at all", which is the answer to "is the corpus clean
    # or is the corpus empty" — and the reason it never appears as a bare `0`.
    empty_corpus = ("" if corpus["total_calls"]
                    else "   (no non-error call in this corpus)")
    lines = [
        f"tool-quality {report['date']}  window {report['window']['from']}"
        f" .. {report['window']['through']}"
        f"  (dir: {report['window']['sessions_dir']})",
        f"corpus: {corpus['files_scanned']} files scanned, "
        f"{corpus['total_calls']} non-error tool calls, "
        f"{corpus['files_unreadable']} unreadable",
        f"empty-result rate: {report['corpus_rates']['empty']}   "
        f"all-null-argument rate: {report['corpus_rates']['degenerate_args']}   "
        f"all-null-argument calls that errored (excluded): "
        f"{corpus['total_degenerate_errored']}{empty_corpus}",
        f"self-check: {'ok' if report['self_check']['ok'] else 'FAILED'} "
        + "  ".join(f"{tool} empty {row['empty']}"
                    for tool, row in report["self_check"]["rates"].items()),
    ]
    rows = report["tools"]
    shown = rows if top <= 0 else rows[:top]
    lines.append(f"tools shown: {len(shown)} of {len(rows)}"
                 f"{' (top by empty count)' if top > 0 else ''}")
    lines.append(f"{'tool':<26} {'empty':>12} {'null-args':>12}")
    for row in shown:
        lines.append(f"{row['tool']:<26} {row['empty_rate']:>12} "
                     f"{row['degenerate_args_rate']:>12}")
    for row in report["tools_without_success_calls"]:
        lines.append(f"{row['tool']:<26} no non-error call "
                     f"(all-null-arg errored {row['degenerate_errored']})")
    prune = report["retention"].get("prune") or {}
    lines.append(f"retention: {report['retention']['days']} days, "
                 f"{prune.get('files', 0)} dated file(s) pruned, "
                 f"{prune.get('errors', 0)} error(s)")
    lines.append(f"wrote: {report.get('written_to', '(not written)')}")
    return "\n".join(lines)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.tool_quality",
        description="Read-only sweep of success-coded tool-call quality "
                    "(#2275). Report-only: nothing is gated on these rates.")
    parser.add_argument("--sessions-dir", default=None,
                        help="transcripts to read (default: app.paths.SESSIONS_DIR)")
    parser.add_argument("--out-dir", default=None,
                        help="row-set directory (default: app.paths.TOOL_QUALITY_DIR)")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"window in days, inclusive (default {DEFAULT_DAYS});"
                             " negative means every transcript in the directory")
    parser.add_argument("--today", default=None,
                        help="YYYY-MM-DD clock for the file name and cutoff")
    parser.add_argument("--top", type=int, default=5,
                        help="rows to print, worst-first; 0 prints all")
    args = parser.parse_args(argv)
    days = None if args.days < 0 else args.days
    report = sweep(sessions_dir=args.sessions_dir, out_dir=args.out_dir,
                   days=days, today=args.today)
    print(format_report(report, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
