"""#2275: the sweep that counts a successful tool call returning nothing.

Lloyd's failure surface is error-shaped — `runs.status`, `stats.is_error`, the
guardian's read of `logs/server.err`, #690's error-baseline filter — so a call
that succeeds and hands the model an empty answer is invisible to every one of
them. The triage measurement of 2026-10-06 sized the gap: **0 files** match
`git grep -ln "tool_quality|tool-quality|empty_result_rate|null_args|instant_return" -- '*.py' '*.md' '*.yaml'`
from `~/lloyd`, no `~/lloyd-data/eval/tool-quality/` existed, and the corpus held
91,904 tool calls across 4,446 session files in the previous 7 days that nobody
read this way. Two of the item's five proposed steps turned out to be wrong about
the substrate, and both corrections are pinned here rather than only in prose:

* **`stats.result_chars` cannot be the emptiness test.** `fallback_for_empty_result`
  (`app/harness/tool_result_spill.py`) runs at dispatch, *before* the row is
  written, so a literally-empty answer persists as `"(Read completed with no
  output)"` with `result_chars` 31 — measured over 2,147 tool rows in the 60
  newest sessions, `result_chars == 0` matched **0** rows. The working vocabulary
  is `"(no matches)"` (`Grep`/`Glob`'s zero hit: 12 non-empty characters, and the
  largest single source — 35 rows, Grep 25 and Glob 10, in that week), the
  `… completed with no output` marker, and `stats.raw_chars == 0`. `raw_chars`
  alone also under-counts, because it is *absent* rather than zero on the
  reconstructed paths (#1052): 26 marker rows that week, only 13 of them with
  `raw_chars == 0`.
  (`test_each_zero_result_test_fires_on_a_row_only_it_can_see`,
  `test_result_chars_is_never_an_emptiness_test`,
  `test_a_marker_row_with_result_chars_31_and_no_raw_chars_is_counted_end_to_end`)
* **instant-return is not derivable, and is not here.** `_tool_pair`
  (`app/routers/messages.py:793-822`) passes ONE `timestamp` to both rows of a
  pair and `app/run_recorder.py:323` sets `ts` once for both, so 1599/1599
  call/result pairs in the 40 newest session JSONs carry identical timestamps: a
  per-pair duration is zero by construction, and the real value
  (`app/harness/loop.py:3230`'s `duration_ms`) is dropped by both writers. Emitting
  it is a dispatch-side change on a harness write path and its own round, so this
  file pins the absence rather than letting a field appear by accident:
  `test_no_duration_is_read_from_a_transcript`.

The positive control is `app.tool_quality.control_transcript`: a seeded transcript
whose designed `2 / 5` empty-result tool and single all-null-argument call the
sweep must report exactly, emitted inside every day's row set. That is the
standing class rule — "a 0-hit grep needs a positive control, and a grep against a
path that does not exist returns 0" (2026-09-16/17) — turned on the instrument
itself: a day of zeroes always arrives beside the evidence that the detector still
fires.

Before this diff every test here fails at import for the same reason:
`app/tool_quality.py` does not exist, and neither does `app.paths.TOOL_QUALITY_DIR`.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from app import paths as app_paths
from app import tool_quality as tq

DAY = "2026-10-06"
ANCHOR = date(2026, 10, 6)
TS = "2026-10-06T00:00:00.123456"


# ───────────────────────── seeded rows, built in this file ───────────────────
# Deliberately NOT built with `app.tool_quality._seed` or with
# `app.transcript_entries`: the sweep reads files another process wrote, so the
# fixtures restate the on-disk row shape and keep pinning it if a builder's
# defaults move.
def call_row(call_id: str, tool: str, args, **stats) -> dict:
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {
        "id": f"msg_{call_id}_tc", "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [{"id": call_id, "call_id": call_id, "type": "function",
                        "function": {"name": tool, "arguments": arguments}}],
        "timestamp": TS, "stats": stats or {"iteration": 1},
    }


def call_obj(call_id: str, tool: str, args) -> dict:
    """The tool-call object alone — the unit `is_degenerate_arguments` reads.

    `classify_transcript` walks `message["tool_calls"]` and hands the classifier
    the call object, not the message row, so a test of that predicate in
    isolation passes the same unit.
    """
    return call_row(call_id, tool, args)["tool_calls"][0]


def result_row(call_id: str, text, stats) -> dict:
    return {
        "id": f"msg_{call_id}_result", "role": "tool",
        "content": [{"type": "text", "text": text}],
        "tool_call_id": call_id, "timestamp": TS, "stats": stats,
    }


def ok_stats(text: str) -> dict:
    """The eager path's stats for a normal, non-empty, non-error result."""
    return {"result_chars": len(text), "raw_chars": len(text),
            "is_error": False}


def err_stats(text: str) -> dict:
    return {"result_chars": len(text), "raw_chars": len(text),
            "is_error": True}


def doc(*messages: dict) -> dict:
    return {"session_id": "seeded", "title": "seeded",
            "messages": list(messages)}


def write_session(root: Path, name: str, document) -> Path:
    (root / name).write_text(json.dumps(document), encoding="utf-8")
    return root / name


def day_name(day: date) -> str:
    return f"{day.isoformat()}.json"


def seed_dated_files(store: Path, *, ages, today: date = ANCHOR) -> "dict[str, Path]":
    """Write `<today - age>.json` for each age, with distinguishable bytes."""
    out: "dict[str, Path]" = {}
    for age in ages:
        path = store / day_name(today - timedelta(days=age))
        path.write_text(json.dumps({"date": path.name[:10], "seeded": age}),
                        encoding="utf-8")
        out[path.name] = path
    return out


@pytest.fixture
def corpus(tmp_path: Path):
    """An empty `sessions/` directory and a separate, not-yet-created store."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return sessions, tmp_path / "eval" / "tool-quality"


# ───────────────── clause 1: the classifier and its denominator ──────────────
def test_classifier_rows_carry_exactly_the_five_named_fields():
    """One transcript in, per-tool rows out, with the five clause fields.

    `classify_transcript` is pure over one document — no clock, no filesystem —
    and a row carries `tool`, `calls`, `empty`, `degenerate_args` and
    `degenerate_errored` and nothing else. A sixth count here would be a number no
    clause asked for, which a reader of the emitted JSON could mistake for a rate:
    every rate is added later by the sweep, in `k / N` form.
    """
    rows = tq.classify_transcript(doc(
        call_row("c1", "Grep", {"pattern": "zz"}),
        result_row("c1", "(no matches)", ok_stats("(no matches)")),
        call_row("c2", "Read", {"file_path": "~/lloyd/README.md"}),
        result_row("c2", "# Lloyd\n", ok_stats("# Lloyd\n")),
    ))
    assert set(rows) == {"Grep", "Read"}
    for row in rows.values():
        assert set(row) == set(tq.ROW_KEYS)
    assert rows["Grep"] == {"tool": "Grep", "calls": 1, "empty": 1,
                            "degenerate_args": 0, "degenerate_errored": 0}
    assert rows["Read"] == {"tool": "Read", "calls": 1, "empty": 0,
                            "degenerate_args": 0, "degenerate_errored": 0}


def test_an_error_flagged_call_is_out_of_calls_and_out_of_every_numerator():
    """`is_error: true` excludes a call from `calls` AND from `empty`.

    Four calls of one tool: one unflagged empty, one unflagged non-empty, one
    error-flagged whose text is the zero-hit marker, one error-flagged with all
    arguments null. The row must read `calls 2, empty 1, degenerate_args 0,
    degenerate_errored 1`: the errored empty does not join `empty`, because a
    failure the harness already reports is not also a success-coded quality
    failure; and the errored all-null-argument call is kept only in
    `degenerate_errored`, so the exclusion is a visible count rather than a silent
    subtraction from the denominator.
    """
    rows = tq.classify_transcript(doc(
        call_row("a1", "Bash", {"command": "true"}),
        result_row("a1", "(Bash completed with no output)",
                   {"result_chars": 33, "raw_chars": 0, "is_error": False}),
        call_row("a2", "Bash", {"command": "ls"}),
        result_row("a2", "file\n", ok_stats("file\n")),
        call_row("a3", "Bash", {"command": "false"}),
        result_row("a3", "(no matches)", err_stats("(no matches)")),
        call_row("a4", "Bash", {}),
        result_row("a4", "Error: command is required",
                   err_stats("Error: command is required")),
    ))
    assert rows["Bash"] == {"tool": "Bash", "calls": 2, "empty": 1,
                            "degenerate_args": 0, "degenerate_errored": 1}


def test_a_call_with_no_result_row_is_not_counted_at_all():
    """No paired result row means no evidence and no denominator.

    `app/run_recorder.py::_unpersisted_pairs` and `_tool_pair(evt=None)` both
    reconstruct a missing pair onto disk, but a call whose result row never
    reached the file at all cannot be read as either empty or fine — and putting
    it in `calls` would dilute every rate in the table with a number that is a
    persistence accident. The tool survives here only for its counted sibling.
    """
    rows = tq.classify_transcript(doc(
        call_row("b1", "Grep", {"pattern": "x"}),
        call_row("b2", "Grep", {"pattern": "y"}),
        result_row("b2", "app/paths.py:1: x\n",
                   ok_stats("app/paths.py:1: x\n")),
    ))
    assert rows == {"Grep": {"tool": "Grep", "calls": 1, "empty": 0,
                             "degenerate_args": 0, "degenerate_errored": 0}}


def test_one_call_id_counts_once_even_if_its_call_row_is_written_twice():
    """A re-written pair stays one dispatch in the denominator.

    Both writers guard a pair set (`persisted_pairs` in the recorder), so a second
    call row for one `call_id` is the same dispatch persisted twice. The live
    corpus holds none — 0 repeated call ids across 4,464 session files in the
    7-day window measured on 2026-10-06 — and this pins that a future double write
    costs a duplicate row rather than quietly doubling a rate's denominator.
    """
    rows = tq.classify_transcript(doc(
        call_row("d1", "Grep", {"pattern": "x"}),
        call_row("d1", "Grep", {"pattern": "x"}),
        result_row("d1", "(no matches)", ok_stats("(no matches)")),
    ))
    assert rows["Grep"]["calls"] == 1
    assert rows["Grep"]["empty"] == 1


# ─────────────── clause 2: the vocabulary, and never result_chars ────────────
def test_each_zero_result_test_fires_on_a_row_only_it_can_see():
    """Four rows; each is empty for a reason the other tests cannot supply.

    * `"(no matches)"` with `raw_chars` 12 — a non-zero size, so only the
      `(no matches)` text test can see it, and this is `Grep`/`Glob`'s real row.
    * `"(Read completed with no output)"` with `stats` = `{"result_chars": 31}`
      and no `raw_chars` key — the reconstructed path (#1052), visible only to the
      marker test.
    * `"(Bash completed with no output)"` with `raw_chars` 0 — the eager path's
      row, which the marker test and the size test both catch: that overlap is why
      the two are ORed rather than one replacing the other.
    * `"nothing"` with `raw_chars` 0 — text no marker matches, so ONLY the size
      test can see it.

    A detector that loses any one of the three tests loses at least one row here.
    """
    assert tq.is_zero_result(result_row("x", "(no matches)",
                                       ok_stats("(no matches)"))) is True
    assert tq.is_zero_result(result_row("y", "(Read completed with no output)",
                                       {"result_chars": 31})) is True
    assert tq.is_zero_result(result_row("z", "(Bash completed with no output)",
                                       {"result_chars": 33, "raw_chars": 0,
                                        "is_error": False})) is True
    assert tq.is_zero_result(result_row("w", "nothing",
                                       {"result_chars": 7, "raw_chars": 0,
                                        "is_error": False})) is True


def test_result_chars_is_never_an_emptiness_test():
    """The field that lied once is not consulted at all.

    `result_chars` is the length of the *shaped* row, and `fallback_for_empty_result`
    makes it 31 for a zero-character answer — measured at 0 matches for
    `result_chars == 0` across 2,147 tool rows in the 60 newest sessions. So a row
    whose `result_chars` is 0 beside real text is not empty (the text says what
    happened), and a 1-character answer is not empty either: neither row matches
    anything in the vocabulary. The mirror case — a 31-character `result_chars`
    beside an empty answer — is the one that must read as empty, pinned beside it.
    """
    assert tq.is_zero_result(result_row(
        "n1", "app/paths.py:106: EVAL_BASELINES_DIR\n",
        {"result_chars": 0, "raw_chars": 39, "is_error": False})) is False
    assert tq.is_zero_result(result_row(
        "n2", "-", {"result_chars": 1, "raw_chars": 1,
                    "is_error": False})) is False
    assert tq.is_zero_result(result_row(
        "n3", "no matches found", {"result_chars": 16, "raw_chars": 16,
                                   "is_error": False})) is False
    assert tq.is_zero_result(result_row(
        "n4", "(Grep completed with no output)",
        {"result_chars": 31, "raw_chars": 0, "is_error": False})) is True


def test_an_absent_raw_chars_is_unknown_and_not_a_zero():
    """`raw_chars` missing is "size unknown", which is not "the size was 0".

    #1052 omits the key on reconstructed rows rather than inventing a number, and
    the reasoning that stops the writer inventing a `0` stops this detector
    reading an absent key as one — otherwise every reconstructed row would count as
    empty, which is #1052 pointed the other way. Such a row is still caught when
    its text is a marker; this pins that the size test does not catch it.
    """
    assert tq.is_zero_result(result_row("u", "-", {"result_chars": 1})) is False
    assert tq.is_zero_result(result_row("u2", "-", {})) is False
    assert tq.is_zero_result(result_row("u3", "-", {"raw_chars": None})) is False


def test_a_marker_row_with_result_chars_31_and_no_raw_chars_is_counted_end_to_end(corpus):
    """The reconstruction-blind row, driven through the whole sweep.

    One `Read` call whose result is `"(Read completed with no output)"` with
    `stats` = `{"result_chars": 31}` and no `raw_chars` key: the marker is 31
    characters and `result_chars` is 31, so a size-based detector reports this tool
    as clean while the model was handed nothing. Out of `sweep`, the row that must
    come back is `1 / 1`.
    """
    sessions, store = corpus
    write_session(sessions, "20261006_000000_chat_aa.json", doc(
        call_row("r1", "Read", {"file_path": "~/lloyd/app/djev.py"}),
        result_row("r1", "(Read completed with no output)",
                   {"result_chars": 31}),
    ))
    report = tq.sweep(sessions_dir=sessions, out_dir=store, days=7, today=DAY)
    assert report["tools"] == [{
        "tool": "Read", "calls": 1, "empty": 1, "degenerate_args": 0,
        "degenerate_errored": 0, "empty_rate": "1 / 1",
        "degenerate_args_rate": "0 / 1"}]


def test_the_control_carries_every_vocabulary_row_it_claims_to():
    """The positive control's rows are what its docstring says they are.

    A control whose shape can silently drift is not a control, so the design is
    asserted row by row: the `(no matches)` row with a non-zero size, the
    31-character marker row with `result_chars` 31 and NO `raw_chars` key, the
    `raw_chars == 0` row, and the all-null-argument row with a NON-empty result —
    that last one on purpose, so the two invariants stay independent and no single
    row can satisfy both by accident.
    """
    by_id = {row["tool_call_id"]: row
             for row in tq.control_transcript()["messages"]
             if row.get("role") == "tool"}

    no_matches = by_id["call_g1"]
    assert tq.result_text(no_matches) == "(no matches)"
    assert no_matches["stats"]["raw_chars"] == 12
    assert tq.is_zero_result(no_matches) is True

    marker_31 = by_id["call_r1"]
    text = tq.result_text(marker_31)
    assert text == "(Read completed with no output)"
    assert len(text) == 31
    assert marker_31["stats"] == {"result_chars": 31}
    assert tq.is_zero_result(marker_31) is True

    raw_zero = by_id["call_g3"]
    assert raw_zero["stats"]["raw_chars"] == 0
    assert tq.is_zero_result(raw_zero) is True

    null_args = by_id["call_g4"]
    assert tq.is_zero_result(null_args) is False
    errored_null_args = by_id["call_r3"]
    assert errored_null_args["stats"]["is_error"] is True


def test_the_seeded_two_of_five_and_one_null_argument_case_reports_exactly(corpus):
    """The item's own seeded case, through the sweep and back out of the file.

    A transcript seeded with five `Grep` calls of which exactly two returned
    nothing — one of those five carrying all-null arguments — plus three `Read`
    calls including one all-null-argument call that errored. Written as a session
    file, swept, and re-read from the dated JSON, `Grep` must read `2 / 5` empty
    and `1 / 5` all-null-arguments: not `2 / 6` (an errored call in the
    denominator), not `3 / 5` (an errored empty in the numerator), not `1 / 5`
    empty (a dropped marker row).
    """
    sessions, store = corpus
    write_session(sessions, "20261006_000000_chat_seeded.json",
                  tq.control_transcript())
    report = tq.sweep(sessions_dir=sessions, out_dir=store, days=7, today=DAY)
    written = json.loads((store / day_name(ANCHOR)).read_text(encoding="utf-8"))
    assert written == {key: value for key, value in report.items()
                       if key != "written_to"}

    grep = next(row for row in written["tools"] if row["tool"] == "Grep")
    assert (grep["empty"], grep["calls"], grep["empty_rate"]) == (2, 5, "2 / 5")
    assert (grep["degenerate_args"], grep["degenerate_args_rate"]) == (1, "1 / 5")
    read = next(row for row in written["tools"] if row["tool"] == "Read")
    assert (read["empty"], read["calls"], read["degenerate_errored"]) == (1, 2, 1)
    assert written["corpus"]["total_calls"] == 7
    assert written["self_check"]["ok"] is True


def test_the_self_check_fires_now_and_fails_loudly_if_the_detector_stops(
        monkeypatch):
    """A positive control that cannot fail is decoration.

    With `is_zero_result` replaced by a stub that never fires — the exact failure
    this exists to catch, a vocabulary that stops matching after some writer
    change — the same `self_check()` call must say `ok: False` with the control's
    empties at 0, while the live corpus could still print an innocent-looking day
    of zeroes. `self_check` ships inside every day's row set for that reason.
    """
    check = tq.self_check()
    assert check["ok"] is True
    assert check["actual"] == check["expected"]
    assert check["rates"]["Grep"] == {"empty": "2 / 5",
                                      "degenerate_args": "1 / 5"}

    monkeypatch.setattr(tq, "is_zero_result", lambda row: False)
    broken = tq.self_check()
    assert broken["ok"] is False
    assert broken["actual"]["Grep"] == {"tool": "Grep", "calls": 5, "empty": 0,
                                        "degenerate_args": 1,
                                        "degenerate_errored": 0}
    assert broken["rates"]["Grep"]["empty"] == "0 / 5"


# ─────────────── clause 3: degenerate arguments, both branches ───────────────
def test_every_value_null_or_empty_is_a_degenerate_call():
    """`{}` counts, and so does a call whose every value is null or empty.

    The workshop's tool "curled with every argument null producing a meaningless
    search" reads as a well-formed call, so the test is over the parsed
    `function.arguments`: no keys at all, or nothing but `None`, `""`, `[]` and
    `{}`. One populated value takes the call out of the count — including `False`
    and `"  "`, which are values: the call asked for something, and what came back
    is then about the tool rather than the arguments.
    """
    assert tq.is_degenerate_arguments(call_obj("k", "Grep", {})) is True
    assert tq.is_degenerate_arguments(
        call_obj("k", "Grep", {"a": None, "b": "", "c": [], "d": {}})) is True
    assert tq.is_degenerate_arguments(
        call_obj("k", "Grep", {"pattern": "x"})) is False
    assert tq.is_degenerate_arguments(
        call_obj("k", "Grep", {"limit": 5})) is False
    assert tq.is_degenerate_arguments(
        call_obj("k", "Grep", {"flagged": False})) is False
    assert tq.is_degenerate_arguments(
        call_obj("k", "Grep", {"paths": [], "pattern": "  "})) is False


def test_arguments_that_are_not_an_object_are_not_counted():
    """A malformed call is a different defect, and it arrives with an error flag.

    Unparseable JSON, a JSON array, a JSON string and a missing `arguments` field
    all fail this test rather than pass it. Counting them would inflate the
    success-coded rate with calls that never reached a tool at all — those are
    #576's argument-guessing retries, and they show up flagged.
    """
    assert tq.is_degenerate_arguments(call_obj("m", "Grep", "")) is False
    assert tq.is_degenerate_arguments(call_obj("m", "Grep", "[]")) is False
    assert tq.is_degenerate_arguments(call_obj("m", "Grep", '"x"')) is False
    assert tq.is_degenerate_arguments(
        {"call_id": "m", "function": {"name": "Grep"}}) is False


def test_an_all_null_argument_call_that_errored_lands_in_degenerate_errored():
    """The errored half of the same call shape, kept out of the rate.

    Same all-null arguments, `is_error: true`: `degenerate_args` stays 0,
    `degenerate_errored` becomes 1, and `calls` does not move at all — so the rate
    the report prints cannot be inflated by an argument-shape rejection the
    harness already surfaced. On the live 7-day corpus measured 2026-10-06 that
    bucket held 19 calls (Bash 15, `backlog_write_task` 2, `review` 2) while the
    success-coded rate held 0, which is exactly this separation doing its job.
    """
    rows = tq.classify_transcript(doc(
        call_row("e1", "Read", {"file_path": None, "offset": ""}),
        result_row("e1", "Error: file_path is required",
                   err_stats("Error: file_path is required")),
    ))
    assert rows["Read"] == {"tool": "Read", "calls": 0, "empty": 0,
                            "degenerate_args": 0, "degenerate_errored": 1}


# ──────────── clause 4: every rate beside its denominator ────────────────────
def test_every_emitted_and_printed_rate_is_k_over_N(corpus):
    """A clean tool prints `0 / 3`; no row prints a bare count.

    Two tools in one file: `Glob` with three all-good calls, `Grep` with four of
    which two returned nothing. In the emitted JSON every rate is the string
    `"k / N"` over that tool's own non-error call count, and in the printed table
    each data line carries two such pairs — so a rate cannot be quoted out of a
    report without the denominator that makes it meaningful. The workshop's own
    headline ("42% of searches returned zero results") was a small-denominator
    statistic by its author's account, and Lloyd is one user, not a
    requests-per-second service.
    """
    sessions, store = corpus
    write_session(sessions, "20261006_000000_chat_two_tools.json", doc(
        call_row("f1", "Glob", {"pattern": "*.py"}),
        result_row("f1", "app/paths.py\n", ok_stats("app/paths.py\n")),
        call_row("f2", "Glob", {"pattern": "*.rs"}),
        result_row("f2", "app/paths.py\n", ok_stats("app/paths.py\n")),
        call_row("f3", "Glob", {"pattern": "*.go"}),
        result_row("f3", "app/paths.py\n", ok_stats("app/paths.py\n")),
        call_row("g1", "Grep", {"pattern": "a"}),
        result_row("g1", "(no matches)", ok_stats("(no matches)")),
        call_row("g2", "Grep", {"pattern": "b"}),
        result_row("g2", "(Grep completed with no output)",
                   {"result_chars": 31, "raw_chars": 0, "is_error": False}),
        call_row("g3", "Grep", {"pattern": "c"}),
        result_row("g3", "app/djev.py:1: def x\n",
                   ok_stats("app/djev.py:1: def x\n")),
        call_row("g4", "Grep", {"pattern": "d"}),
        result_row("g4", "app/djev.py:2: def y\n",
                   ok_stats("app/djev.py:2: def y\n")),
    ))
    report = tq.sweep(sessions_dir=sessions, out_dir=store, days=7, today=DAY)
    by_tool = {row["tool"]: row for row in report["tools"]}
    assert by_tool["Glob"]["empty_rate"] == "0 / 3"
    assert by_tool["Glob"]["degenerate_args_rate"] == "0 / 3"
    assert by_tool["Grep"]["empty_rate"] == "2 / 4"
    assert report["corpus_rates"] == {"empty": "2 / 7",
                                      "degenerate_args": "0 / 7"}

    printed = tq.format_report(report, top=0).splitlines()
    glob_line = next(line for line in printed if line.startswith("Glob"))
    grep_line = next(line for line in printed if line.startswith("Grep"))
    for line in (glob_line, grep_line):
        assert re.search(r"\d+ / \d+\s+\d+ / \d+$", line), line
    assert glob_line.rstrip().endswith("0 / 3")
    assert "0 / 3" in glob_line and "2 / 4" in grep_line


def test_a_tool_with_no_non_error_call_gets_no_rate_row_but_is_still_named(corpus):
    """No denominator, no rate line — and no silent disappearance either.

    A tool whose every call came back error-flagged has `calls == 0`, so `0 / 0`
    would be a division that means nothing and a bare `0` would read as a
    measurement. It is left out of `tools` (emitted and printed) and named under
    `tools_without_success_calls` with its `degenerate_errored` count — the one
    signal the classifier keeps about it — which the printed table says in words.
    """
    sessions, store = corpus
    write_session(sessions, "20261006_000000_chat_errored_only.json", doc(
        call_row("h1", "BrokenTool", {}),
        result_row("h1", "Error: schema", err_stats("Error: schema")),
        call_row("h2", "BrokenTool", {"a": 1}),
        result_row("h2", "Error: schema", err_stats("Error: schema")),
    ))
    report = tq.sweep(sessions_dir=sessions, out_dir=store, days=7, today=DAY)
    assert report["tools"] == []
    assert report["tools_without_success_calls"] == [
        {"tool": "BrokenTool", "degenerate_errored": 1}]

    printed = tq.format_report(report, top=5)
    tool_lines = [line for line in printed.splitlines()
                  if line.startswith("BrokenTool")]
    assert len(tool_lines) == 1, printed
    assert "no non-error call" in tool_lines[0]
    assert "0 / 0" not in tool_lines[0]
    assert not any(re.match(r"^BrokenTool\s+\d+ / 0\b", line)
                   for line in printed.splitlines())
    # The corpus line says so too: the day's denominator really is zero.
    assert "no non-error call in this corpus" in printed


def test_the_emitted_row_set_states_the_corpus_it_measured(corpus):
    """Files scanned and total calls travel with the rates.

    Three files, one of them truncated JSON (a run killed mid-write leaves exactly
    this shape): `files_scanned` counts what was opened, `files_unreadable` says
    what could not be read, `total_calls` is the denominator behind the corpus
    rate. This is what makes a zero-hit sweep legible — the answer to "is that 0 a
    clean corpus or an empty one" is in the same object as the 0.
    """
    sessions, store = corpus
    write_session(sessions, "20261006_000000_chat_one.json", doc(
        call_row("i1", "Glob", {"pattern": "*.py"}),
        result_row("i1", "app/paths.py\n", ok_stats("app/paths.py\n"))))
    write_session(sessions, "20261005_000000_chat_two.json", doc(
        call_row("i2", "Glob", {"pattern": "*.txt"}),
        result_row("i2", "(no matches)", ok_stats("(no matches)"))))
    (sessions / "20261004_000000_chat_truncated.json").write_text(
        '{"session_id": "x", "messages": [{"role": ', encoding="utf-8")
    report = tq.sweep(sessions_dir=sessions, out_dir=store, days=7, today=DAY)
    assert report["corpus"] == {
        "files_scanned": 3, "files_unreadable": 1, "files_undated_in_dir": 0,
        "total_calls": 2, "total_empty": 1, "total_degenerate_args": 0,
        "total_degenerate_errored": 0}
    assert report["corpus_rates"]["empty"] == "1 / 2"


# ────────── clause 5: read-only, one dated file, self-pruning store ──────────
def test_the_sweep_leaves_every_input_byte_for_byte_unchanged(corpus):
    """Read-only means read-only, including over the files it cannot parse.

    Every file hashed before and after, over the shapes the sweep must survive
    without touching: a normal transcript, a truncated one, a document with no
    `messages` key, a non-`.json` sibling it never opens, and a `.tool-results/`
    directory holding a screenshot the sweep has no reason to read. The only
    writes are the sweep's own dated file and the deletions `prune_store` makes
    inside its own store directory.
    """
    sessions, store = corpus
    write_session(sessions, "20261006_000000_chat_ok.json",
                  tq.control_transcript())
    (sessions / "20261006_000001_chat_trunc.json").write_text(
        '{"messages": [{"role"', encoding="utf-8")
    write_session(sessions, "20261006_000002_chat_nomessages.json",
                  {"session_id": "s", "message_count": 0})
    (sessions / "notes.txt").write_text("not a transcript\n", encoding="utf-8")
    (sessions / "20261006_000003_chat_ok.tool-results").mkdir()
    (sessions / "20261006_000003_chat_ok.tool-results"
     / "shot.png").write_bytes(b"\x89PNG\r\n")

    def fingerprint() -> "dict[str, str]":
        return {path.relative_to(sessions).as_posix():
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(sessions.rglob("*")) if path.is_file()}

    before = fingerprint()
    tq.sweep(sessions_dir=sessions, out_dir=store, days=7, today=DAY)
    after = fingerprint()
    assert len(before) == 5
    assert after == before


def test_the_sweep_writes_exactly_one_dated_file_named_by_today(corpus,
                                                                monkeypatch):
    """One `<YYYY-MM-DD>.json` under the store, named from the pinned clock.

    Run twice on one day and the store still holds one file: the report is a
    day's measurement, re-written by the day's last run, not a log that grows per
    invocation. The default location is `app.paths.TOOL_QUALITY_DIR`, resolved at
    CALL time — which this asserts by monkeypatching that constant, a thing that
    only works because `sweep` reads the module attribute rather than an
    import-time copy, so a process whose data root moved never sweeps one tree and
    writes another.
    """
    sessions, store = corpus
    write_session(sessions, "20261006_000000_chat_a.json", doc(
        call_row("j1", "Glob", {"pattern": "*.py"}),
        result_row("j1", "app/paths.py\n", ok_stats("app/paths.py\n"))))
    first = tq.sweep(sessions_dir=sessions, out_dir=store, days=7, today=DAY)
    second = tq.sweep(sessions_dir=sessions, out_dir=store, days=7, today=DAY)
    assert first["written_to"] == second["written_to"]
    assert [path.name for path in sorted(store.iterdir())] == [day_name(ANCHOR)]
    assert json.loads(Path(second["written_to"]).read_text(
        encoding="utf-8"))["date"] == DAY

    moved = store.parent / "moved-store"
    monkeypatch.setattr(app_paths, "TOOL_QUALITY_DIR", moved)
    default_run = tq.sweep(sessions_dir=sessions, days=7, today=DAY)
    assert default_run["written_to"] == str(moved / day_name(ANCHOR))
    assert (moved / day_name(ANCHOR)).is_file()


def test_tool_quality_dir_is_a_data_root_path_beside_the_baselines():
    """The constant exists, and it cannot point inside the code tree.

    `TOOL_QUALITY_DIR` sits beside `EVAL_BASELINES_DIR` under `DATA_ROOT`, which is
    the 2026-09-22 rule that a `rm -r` or a fixture aimed at `~/lloyd` must not be
    able to reach runtime state: a store declared inside the code tree is a store
    one clean build deletes.
    """
    assert app_paths.TOOL_QUALITY_DIR == (
        app_paths.DATA_ROOT / "eval" / "tool-quality")
    assert (app_paths.TOOL_QUALITY_DIR.parent
            == app_paths.EVAL_BASELINES_DIR.parent)
    assert app_paths.TOOL_QUALITY_DIR != (
        app_paths.LLOYD_HOME / "eval" / "tool-quality")


def test_each_run_prunes_its_own_dated_files_past_the_window(corpus):
    """The store bounds itself: past-window dated files go, the rest is untouched.

    Seeded at 40, 30, 8, 3 and 0 days old with `retention_days=7`: the cutoff is
    `today - 6`, so the 3-day and today's files survive byte-identical and the
    other three are deleted. A retained day keeps its bytes — this prunes age, it
    does not rewrite history — and the run records its own deletion inside the
    dated file it writes, so a reader can see the store bounding itself without
    consulting a log elsewhere.
    """
    sessions, store = corpus
    store.mkdir(parents=True)
    seeded = seed_dated_files(store, ages=(40, 30, 8, 3, 0))
    write_session(sessions, "20261006_000000_chat_a.json", doc(
        call_row("k1", "Glob", {"pattern": "*.py"}),
        result_row("k1", "app/paths.py\n", ok_stats("app/paths.py\n"))))
    report = tq.sweep(sessions_dir=sessions, out_dir=store, days=7, today=DAY,
                      retention_days=7)

    assert sorted(path.name for path in store.iterdir()) == [
        "2026-10-03.json", day_name(ANCHOR)]
    assert report["retention"]["days"] == 7
    assert report["retention"]["prune"]["files"] == 3
    for age in (3, 0):
        name = day_name(ANCHOR - timedelta(days=age))
        assert seeded[name].read_bytes() == (store / name).read_bytes()
    written = json.loads((store / day_name(ANCHOR)).read_text(encoding="utf-8"))
    assert written["retention"]["prune"]["files"] == 3


def test_pruning_touches_only_this_stores_own_dated_json(corpus):
    """A name that is not this store's is never this store's to delete.

    `README.md`, a dated `.jsonl`, an impossible date (`2026-13-45.json`), a dated
    file in the wrong shape (`2026.09.01.json`), a bare `notadate.json`, and a
    dated file in a sibling directory all survive a prune whose window would
    otherwise have covered them. A prune that could eat a human's note dropped
    beside the data is not a prune.
    """
    store = corpus[1]
    store.mkdir(parents=True)
    survivors = {
        "README.md": "notes about the store\n",
        "2026-09-01.jsonl": '{"old": 1}\n',
        "2026-13-45.json": "{}\n",
        "2026.09.01.json": "{}\n",
        "notadate.json": "{}\n",
    }
    for name, body in survivors.items():
        (store / name).write_text(body, encoding="utf-8")
    neighbour = store.parent / "elsewhere" / "2025-01-01.json"
    neighbour.parent.mkdir(parents=True)
    neighbour.write_text("{}\n", encoding="utf-8")

    assert tq.prune_store(store, today=DAY, days=7) == {
        "files": 0, "bytes": 0, "errors": 0}
    assert sorted(path.name for path in store.iterdir()) == sorted(survivors)
    assert neighbour.is_file()


def test_a_window_of_zero_or_less_deletes_nothing(corpus):
    """An unbounded store beats a mistyped window that deletes every measurement.

    `app/component_manifest.py` reaches the same conclusion from its config key: a
    non-positive window disables pruning rather than meaning "keep nothing",
    because the failure this guards against is the one where the numbers a later
    reader needs are already gone, with no undo.
    """
    store = corpus[1]
    store.mkdir(parents=True)
    seed_dated_files(store, ages=(40, 30, 8))
    assert tq.prune_store(store, today=DAY, days=0) == {
        "files": 0, "bytes": 0, "errors": 0}
    assert tq.prune_store(store, today=DAY, days=-1) == {
        "files": 0, "bytes": 0, "errors": 0}
    assert len(list(store.iterdir())) == 3


# ──────────────── out of scope, pinned so it stays out ───────────────────────
def test_no_duration_is_read_from_a_transcript():
    """Instant-return is out, and this is what stops it creeping back in.

    `build_tool_result_entry` writes no duration field, and `_tool_pair` hands the
    call row and the result row the SAME timestamp string — measured 2026-10-06:
    1599/1599 call/result pairs in the 40 newest session JSONs share a timestamp,
    and `app/harness/loop.py:3230`'s `duration_ms` is dropped by both persistence
    paths. So no duration is computable here, and a row seeded with a `duration_ms`
    key must still produce no timing signal: same five fields, no `instant` in
    `ROW_KEYS`, no `is_instant_return` to import. Emitting that field is a
    dispatch-side change on a harness write path and its own round.
    """
    document = doc(
        call_row("p1", "Grep", {"pattern": "x"}, duration_ms=1),
        result_row("p1", "app/paths.py:1: x\n",
                   dict(ok_stats("app/paths.py:1: x\n"), duration_ms=1)),
    )
    rows = tq.classify_transcript(document)
    assert rows["Grep"] == {"tool": "Grep", "calls": 1, "empty": 0,
                            "degenerate_args": 0, "degenerate_errored": 0}
    assert "instant" not in tq.ROW_KEYS
    assert not hasattr(tq, "is_instant_return")
    assert not any("instant" in key or "duration" in key
                   for row in tq.tool_rows(rows) for key in row)
