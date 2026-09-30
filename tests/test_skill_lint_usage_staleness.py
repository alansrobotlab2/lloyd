"""The STALE bucket's usage half is measured now, through the real telemetry (#1603).

`scripts/skill_lint.py` used to print what it *would* measure — "Usage-based
staleness detection (N queries where this skill scored > 0) requires injection-level
telemetry not yet emitted — see #334 follow-ups" — and the sentence was regenerated
into the nightly report every run because the report is rewritten wholesale. It
became false on 2026-09-24, when #435 landed as `3774e27b` and
`prefetch._emit_skill_match_events` started appending one `prefetch.skill_match` row
per offered skill per turn. `app.skill_telemetry.skill_injection_counts` is the
reader, and #1603 wired it to the report.

These tests run against a synthetic event log in `tmp_path`, so the counts asserted
below are the fixture's counts, not the live machine's. Two things make that more
than a smoke test: the numbers come out of the real reader rather than a stub, so
the seam between the report and `app.skill_telemetry` is exercised; and the shape
that matters — a library skill the telemetry never saw — is a shape the live window
is full of (93 of 193 skills on 2026-09-28) and cannot be produced by any fixture
that only writes rows for the skills it names.

What is NOT pinned here is a retirement threshold. The window is 30 days because
that is the span `skill_injection_counts` takes, and the rows begin 2026-09-25, so
thirty days of them do not exist until ~2026-10-24 (#1603's owed list) — but the
reason the table yields no cutoff at all is the reporting ceiling, not that date:
`SKILL_REPORT_TOP_K = 8` (`app/prefetch.py`) means a skill below rank 8 contributes
no rows to ANY window, so no depth makes it measurable (#1815). The report prints
counts; it does not decide which skill is dead.

Since #1815 the same file also holds the section's *span* to account, which is the
other half of not lying with a true number. The heading used to read "Usage over the
last 30 days" while the rows beneath it reached back four days, so 123 `loaded share`
percentages were four-day rates wearing thirty-day labels. `first_event` (the reader's
measured depth) has to be the oldest row COUNTED and `None` over an empty store —
pinned in `tests/test_skill_injection_telemetry.py` — and what the section prints with
it is pinned here: the measured span on the line carrying the counts, the shortfall
said in words when it exceeds a day, the declared window and no notice when the store
really is that deep, and no date literal anywhere in the code that produces the depth.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: `app.prefetch` is the WRITER at the other end of the event log this file reads, so
#: the seam nodes below emit through it rather than hand-writing its rows.
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "skill_lint_usage_staleness", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sl = _load()

#: Inside the window the reader computes from `days`, so a fixture timestamp has to
#: be recent — a row older than the window is a row that was never written, and the
#: test below would silently be testing the no-data path.
NOW = datetime.now(timezone.utc)


def _ts(offset_minutes: int = 5) -> str:
    return (NOW - timedelta(minutes=offset_minutes)).strftime(
        "%Y-%m-%dT%H:%M:%S.") + "000Z"


def _match_row(skill: str, *, landed: bool, score: float = 15.9,
               ts: str | None = None) -> str:
    return json.dumps({
        "ts": ts or _ts(), "session_id": "s-one", "event": "prefetch.skill_match",
        "data": {"skill": skill, "score": score, "landed": landed,
                 "injected_body": landed},
    })


def _write_log(root: Path, rows: list[str], session: str = "s-one") -> Path:
    """`<session_id>.events.jsonl` — the flat layout `skill_injection_counts` globs."""
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{session}.events.jsonl").write_text(
        "\n".join(rows) + "\n", encoding="utf-8")
    return root


def _stale_report(result_over: dict) -> str:
    """A rendered report whose STALE section is the thing under test."""
    result = {
        "generated_at": "2026-09-28T00:00:00Z", "skill_roots": ["~/obsidian/skills/"],
        "total": 3, "scanned": ["a", "b", "c"], "dead": [], "drift": [],
        "stale": [], "duplicates": [], "missing_desc": [], "missing_author": [],
        "oversized": [], "unbounded": [], "phantom": [], "missing_script": [],
        "phantom_unbounded": {"count": 0, "cap": 0, "note": "", "examples": []},
        "stale_context": {"threshold_days": sl.STALE_DAYS, "marked_active": 2,
                          "unmarked": 1, "total": 3, "max_age_days": 34,
                          "oldest_could_trip": False},
    }
    result.update(result_over)
    report = sl.render_report(result)
    start = report.index("## STALE")
    rest = report[start:]
    end = rest.find("\n## ", 1)
    return rest if end < 0 else rest[:end]


def test_the_stale_section_prints_offers_and_loaded_over_the_window(tmp_path):
    """Clause 5, first half: the numbers are in the report, per skill, and they came
    from `app.skill_telemetry.skill_injection_counts`.

    The fixture writes three offers and two loads for one skill, one offer that did
    not render for another, and nothing at all for a third. Those three states are
    the whole vocabulary the section has to keep apart: loaded, offered-and-ignored,
    and unmeasured. Getting them out of the real reader rather than a stub is the
    point — the seam between `scripts/skill_lint.py` and `app/skill_telemetry` is a
    process-boundary-shaped one (a nightly job on one side, a module the agent loads
    on the other), and the failure it would otherwise hide is the report going green
    over counts that stopped existing upstream.
    """
    _write_log(tmp_path, [
        _match_row("voice-mode", landed=True, score=15.9),
        _match_row("voice-mode", landed=True, score=12.0),
        _match_row("voice-mode", landed=False, score=9.0),
        _match_row("youtube-transcript", landed=False, score=8.0),
    ])
    usage = sl.collect_usage_counts(days=7, root=tmp_path)
    assert not usage.get("error"), usage
    assert usage["events"] == 4, usage
    assert usage["no_telemetry"] is False, usage
    assert usage["skills"]["voice-mode"] == {
        "offers": 3, "loaded": 2, "ignored": 1,
        "max_score": usage["skills"]["voice-mode"]["max_score"]}, usage["skills"]
    assert usage["skills"]["youtube-transcript"]["loaded"] == 0, usage["skills"]

    sec = _stale_report({"usage": usage, "names": ["voice-mode",
                                                   "youtube-transcript", "never-seen"]})
    # Per-skill counts, printed rather than promised.
    assert "`voice-mode`" in sec, sec
    assert "| 3 | 2 | 1 |" in sec, sec
    assert "`youtube-transcript`" in sec, sec
    assert "| 1 | 0 | 1 |" in sec, sec
    # The window the counts are over, stated, because "offers: 3" with no span is
    # not a rate and a reader will read it as one.
    assert "last 7 days" in sec, sec
    assert "prefetch.skill_match" in sec, sec
    assert "skill_injection_counts" in sec, sec


def test_a_skill_with_no_row_is_listed_as_unmeasured_not_as_zero(tmp_path):
    """Clause 5, second half: absence of a row is not a measurement of zero.

    `prefetch._emit_skill_match_events` writes a row only for skills the scorer put
    in the turn's report, and `SKILL_REPORT_TOP_K = 8` (`app/prefetch.py`) caps that
    at eight per turn. A skill scoring below rank 8 therefore emits nothing however
    often it would have applied — 93 of the 193 library skills were in that state on
    2026-09-28. So the no-row skills must be named under a label that says the
    instrument could not see them, and must not appear as a table row reading `0`,
    because a `0` in an offers column is a claim about usage the emitter never
    supports. This is the distinction the owed scope call on retirement thresholds
    turns on, which is why it is pinned here rather than in prose.
    """
    _write_log(tmp_path, [_match_row("voice-mode", landed=True)])
    usage = sl.collect_usage_counts(days=7, root=tmp_path)
    lines = sl.usage_lines(usage, ["voice-mode", "never-seen-a", "never-seen-b"])
    body = "\n".join(lines)

    assert sl.USAGE_UNMEASURED_LABEL in body, body
    assert "2 of 3 scanned skills" in body, body
    for missing in ("never-seen-a", "never-seen-b"):
        assert f"`{missing}`" in body, body
        assert f"| `{missing}` | 0" not in body, (
            f"{missing} printed as a zero-offer row: the table would be claiming a "
            "usage count the emitter cannot support for a skill below rank "
            "`SKILL_REPORT_TOP_K`")
    # The cause is printed beside the list, so the label is self-explaining to a
    # reader who has not read this file.
    assert "SKILL_REPORT_TOP_K" in body, body

    # Positive control on the control: the skills that DO have rows are excluded from
    # the unmeasured list, so the list is a set difference and not "everything".
    measured_line = [ln for ln in lines if "no row in the window" in ln]
    assert measured_line and "`voice-mode`" not in measured_line[0], lines
    assert usage["skills"]["voice-mode"]["loaded"] == 1, usage


def test_an_unusable_window_says_not_measured_rather_than_printing_an_empty_table(
        tmp_path, monkeypatch):
    """Both silent-failure shapes, refused in words.

    An empty window and a failed read look identical in a report that only prints
    rows: an absent table. The first is legitimately empty (`no_telemetry`, which
    `skill_injection_counts` reports and #1603 must not paper over), and the second
    is an instrument that stopped working. Neither may print as a blank section, and
    neither may repeat the retired claim that the telemetry does not exist — which
    is the sentence this whole file exists to keep out of the report.
    """
    empty = sl.collect_usage_counts(days=7, root=tmp_path)      # no files at all
    assert empty["no_telemetry"] is True, empty
    sec = _stale_report({"usage": empty, "names": ["voice-mode"]})
    assert "NOT MEASURED" in sec, sec
    assert "not the old claim that the telemetry does not exist" in sec, sec
    for retired in ("not yet " + "emitted", "telemetry that is " + "not emitted"):
        assert retired not in sec, retired

    def _boom(*a, **k):
        raise OSError("event log unreadable")
    import app.skill_telemetry as st
    monkeypatch.setattr(st, "skill_injection_counts", _boom)
    broken = sl.collect_usage_counts(days=7, root=tmp_path)
    assert "read failed" in broken["error"], broken
    sec2 = _stale_report({"usage": broken, "names": ["voice-mode"]})
    assert "NOT MEASURED" in sec2 and "event log unreadable" in sec2, sec2


def test_the_report_reads_the_window_through_the_shared_reader(tmp_path, monkeypatch):
    """The seam itself, as a call: `collect_usage_counts` delegates to
    `app.skill_telemetry.skill_injection_counts` with the days it was given.

    A locally re-implemented parser would pass every count assertion above while
    making the two halves of the instrument disagree — `skill_telemetry`'s docstring
    and the report would then be counting different populations, which is the drift
    the shared reader exists to prevent. So the delegation is pinned with the window
    and the root it forwards, not with a source grep.
    """
    seen: dict[str, object] = {}

    def fake(root, days):
        seen["root"] = Path(root)
        seen["days"] = days
        return {"window_days": days, "events": 0, "sessions": 0, "skills": {},
                "no_telemetry": True, "note": "synthetic"}

    import app.skill_telemetry as st
    monkeypatch.setattr(st, "skill_injection_counts", fake)
    out = sl.collect_usage_counts(days=13, root=tmp_path)
    assert seen == {"root": tmp_path, "days": 13}, seen
    assert out["days"] == 13, (
        f"the renderer prints the window from the dict: {out}")


#: Minutes-per-day, so a fixture's age is written as the span it is meant to be.
MINUTES_PER_DAY = 1440.0

#: An ISO calendar date standing in code, of the kind #1815 forbids in the span.
_DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")


def _rows_spanning(root: Path, ages_days: list[float], skill: str = "voice-mode") -> Path:
    """One counted row per age in `ages_days`, oldest first.

    Ages are in days back from `NOW`, which is the fixture's own clock, not the
    reader's — `until` is `now()` at read time, so a span asserted to one decimal
    here is stable by the three-order-of-magnitude margin between a fixture's age and
    the milliseconds it takes pytest to get as far as the reader.
    """
    return _write_log(root, [
        _match_row(skill, landed=(i % 2 == 0),
                   ts=_ts(int(round(age * MINUTES_PER_DAY))))
        for i, age in enumerate(ages_days)])


def _line_with(lines: list[str], needle: str) -> str:
    hits = [ln for ln in lines if needle in ln]
    assert hits, f"no printed line carries {needle!r}:\n" + "\n".join(lines)
    return hits[0]


def _heading(lines: list[str]) -> str:
    return _line_with(lines, "### Usage over the")


def test_a_two_day_store_asked_for_thirty_days_prints_the_two_day_span(tmp_path):
    """Clause 4, shallow half, and clause 3: the section states the span it measured.

    The fixture's oldest counted row is two days back and the report asks for thirty,
    so the pre-#1815 section printed a heading of "the last 30 days" over a two-day
    measurement and a table of `loaded share` percentages that were two-day rates. The
    three claims that had to change are checked separately: the heading names the
    measured span and not the requested one, the counts line names it too (so the span
    travels with the numbers rather than living in a heading a quoted table row leaves
    behind), and the shortfall is said in words with the measured depth in it.
    """
    _rows_spanning(tmp_path, [2.0, 0.5, 0.01])
    usage = sl.collect_usage_counts(days=30, root=tmp_path)
    assert not usage.get("error"), usage
    assert usage["events"] == 3, usage
    lines = sl.usage_lines(usage, ["voice-mode"])
    body = "\n".join(lines)

    heading = _heading(lines)
    # The printed span is derived here rather than written as the literal "2.0", so the
    # node cannot tick over into a flake on a slow run (rows age from a module-level NOW
    # while the reader's `until` is its own now()). What makes it non-tautological is the
    # range: `first_event` pinned to `since`, which is the failure it guards, would derive
    # ~30 days and fail here before the string comparison is ever reached.
    span = (datetime.fromisoformat(usage["until"])
            - datetime.fromisoformat(usage["first_event"])).total_seconds() / 86400.0
    assert 1.99 <= span < 2.05, \
        f"the store is not two days deep, so this node measures something else: {usage}"
    shown = f"{span:.1f}"
    assert f"{shown} measured days" in heading, heading
    assert "last 30 days" not in heading, \
        f"the heading still asserts the requested span as the measured one: {heading}"

    counts = _line_with(lines, "rows over")
    assert f"measured span {shown} days" in counts, counts
    assert "3 rows" in counts and "1 sessions" in counts and "1 skills" in counts, counts

    notice = _line_with(lines, "shallower")
    assert f"{shown} days" in notice and "30" in notice, notice
    assert f"{shown} measured days" in body, body
    assert not any("Usage over the last 30 days" in ln for ln in lines), \
        "the section presents a 30-day span as the span of these counts"


def test_a_store_reaching_the_asked_window_prints_the_declared_window_and_no_notice(
        tmp_path):
    """Clause 4, deep half: the notice is about the store, and retires itself.

    The rows here are written across forty days, which is the case the item named: the
    window is genuinely covered, because the reader counts only rows inside it, so the
    OLDEST COUNTED row is inside a day of `since` even though the file is older. Two
    things are being pinned at once. The section must go quiet — a notice that fired
    whether or not the store was deep would be wallpaper, and wallpaper gets skipped.
    And the two forty-day rows must NOT deepen the reported span: if an excluded row
    could move `first_event`, this section would claim depth the counts do not have,
    which is the same overstatement wearing the other coat.

    Ages come from the module-level `NOW` while the reader's `until` is its own
    `now()`, so the measured span can only GROW by however long the run takes: the
    notice cannot fire spuriously from that drift, and the span below is asserted as a
    range rather than an exact string for the same reason.
    """
    _rows_spanning(tmp_path, [40.0, 30.5, 29.9, 13.9, 2.8, 0.003])
    usage = sl.collect_usage_counts(days=30, root=tmp_path)
    assert not usage.get("error"), usage
    assert usage["events"] == 4, \
        f"rows outside the window were counted: {usage['events']}"
    lines = sl.usage_lines(usage, ["voice-mode"])

    heading = _heading(lines)
    assert "### Usage over the last 30 days" in heading, heading
    assert "measured days" not in heading, heading
    assert not any("shallower" in ln for ln in lines), \
        "the shallow notice fired on a window the store does cover:\n" + "\n".join(lines)
    counts = _line_with(lines, "rows over")
    span = re.search(r"measured span ([0-9.]+) days", counts)
    assert span, counts
    # 29.9 is the age of the oldest COUNTED row, not of the file: the 40.0-day and
    # 30.5-day rows are outside the window, and an excluded row deepening the span
    # would be the overstatement again.
    assert 29.9 <= float(span.group(1)) < 30.0, \
        f"the printed span is not the counted rows': {counts}"


def _strings_outside_docstrings(src: str) -> list[str]:
    """Every string literal in `src` that is not the source's own docstring.

    One scanner, shared by the scan and by its control (`test_a_literal_date_would_be_
    caught`), because a control that re-implements the walk proves only that the
    re-implementation works. Takes source text rather than a function object so a test
    can hand it a snippet the repo does not contain — which is the only way to show this
    scanner is not blind without putting a date into production code to prove it. The
    first statement is skipped when it is a bare string, which is where a docstring
    lives in both a module source and a `def` source; a date in prose about the code is
    not a date the code computes from.
    """
    tree = ast.parse(textwrap.dedent(src))
    head = tree.body[0].body[0] if isinstance(tree.body[0], ast.FunctionDef) else None
    skip = {id(head.value)} if (head is not None and isinstance(head, ast.Expr)
                                and isinstance(head.value, ast.Constant)
                                and isinstance(head.value.value, str)) else set()
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in skip]


#: The production functions that turn an asked-for window into a reported span. If a
#: date is going to be smuggled into the depth, it happens in one of these.
def _span_producers() -> list:
    return [sl.measured_span_days, sl.usage_heading, sl.is_shallow_window,
            sl._span_text, sl.usage_lines]


def test_the_span_is_measured_and_no_date_literal_produces_it():
    """Clause 3's other half: the depth comes from the rows, so nothing here can rot.

    The bug was a sentence in prose — a heading whose span came from `USAGE_WINDOW_DAYS`
    while the rows underneath came from the store. Pinning the fix to a date would
    recreate it one edit later, so what is pinned is that the code which turns a
    window into a span contains no date at all. The scan is `_strings_outside_docstrings`
    over each producer's source; `test_a_literal_date_would_be_caught` is the positive
    control on that same scanner, without which "no date found" would be
    indistinguishable from a scanner that reads nothing.
    """
    found = {fn.__name__: _strings_outside_docstrings(inspect.getsource(fn))
             for fn in _span_producers()}
    # A pure predicate (`is_shallow_window`) may honestly hold no string at all, so
    # blindness is not asserted per function: it is answered by the corpus having
    # literals to read at all, and by `test_a_literal_date_would_be_caught`, which runs
    # the same scanner over a snippet whose code holds a date. The exemption is
    # load-bearing too — `usage_lines` names 2026-09-25 in its own docstring, so a scan
    # that stopped exempting prose would go red on a true sentence.
    assert any(found.values()), f"the scan read no literals anywhere: {found}"
    prose_dates = [d for fn in _span_producers() for d in [fn.__doc__ or ""]
                   if _DATE.search(d)]
    assert prose_dates, \
        "no producer mentions a date in its docstring any more, so the exemption the " \
        "control below pins has nothing to exempt here and could silently stop applying"
    for name, literals in found.items():
        for lit in literals:
            assert not _DATE.search(lit), \
                f"{name} builds its span from a date literal {lit!r}; the depth must " \
                "come from the rows, or it goes stale exactly like the heading did"


def test_a_literal_date_would_be_caught():
    """Positive control on the scan, through the same scanner the scan uses.

    Two edges, both able to fail. A date in *code* must be found: without this, the
    scan's silence would be indistinguishable from a scanner that parses nothing, and
    the item's "no date literal anywhere" would be a claim about a regex nobody checked.
    A date in the *docstring* must NOT be found: `usage_lines` says 2026-09-25 in prose,
    so an exemption that silently stopped applying would turn this suite red on a true
    sentence, and the next editor's fix would be to delete the honest prose.
    """
    code_date = _strings_outside_docstrings(textwrap.dedent('''
            def heading(days):
                """Prose mentioning 2026-10-24 is exempt."""
                return f"### Usage over the {'2026-10-24'} days"
            '''))
    assert any(_DATE.search(s) for s in code_date), \
        f"the scanner cannot see a date inside an f-string; literals were {code_date}"

    doc_only = _strings_outside_docstrings(textwrap.dedent('''
            def heading(days):
                """Prose mentioning 2026-10-24 is exempt."""
                return f"### Usage over the last {days} days"
            '''))
    assert not [s for s in doc_only if _DATE.search(s)], \
        f"the scanner reported a docstring date as code: {doc_only}"


def test_an_unmeasurable_span_says_unmeasured_rather_than_reprinting_the_ask(
        tmp_path):
    """The defensive branch, so the honest path is not the only one that is tested.

    The real reader cannot reach this state — it reports `first_event` whenever it
    counts a row — but `usage_lines` takes a dict, and a caller that hands it counts
    without an oldest row must not get a heading that asserts the requested span as if
    it had been verified. This is the shape of every future caller that fabricates the
    dict, and it is pinned here so the fallback stays a statement of ignorance rather
    than decaying into the old confident heading.
    """
    hand = {"days": 30, "events": 7, "sessions": 2, "no_telemetry": False,
            "skills": {"voice-mode": {"offers": 7, "loaded": 3, "ignored": 4,
                                      "max_score": 9.0}},
            "first_event": None, "until": None}
    lines = sl.usage_lines(hand, ["voice-mode", "never-seen"])
    heading = _heading(lines)
    assert "span unconfirmed" in heading, heading
    assert "2.0 measured" not in heading and "last 30 days (span unconfirmed)" in heading
    counts = _line_with(lines, "rows over")
    assert "span unmeasured" in counts, counts
    assert not any("shallower" in ln for ln in lines), \
        "a notice about depth fired off a depth that was never measured"


def test_the_depth_fix_adds_no_retirement_threshold_and_no_verdict_about_a_skill():
    """Clause 5's second half: the window got readable, nothing got decided.

    #1815 is only allowed to make the depth legible. A staleness or retirement cutoff
    is owed and is a scope call (#1603's owed entry), and no depth settles it either:
    `SKILL_REPORT_TOP_K = 8` means a skill below rank 8 contributes no rows to any
    window at all, which is the reason the table yields no cutoff (#1815). So this
    pins the two constants that could carry such a cutoff at the values they already
    had, and pins that the new sentences about depth say nothing about any skill's
    fate. The wording is checked as a negation: the section may, and does, explain
    that no threshold is applied — what it may not do is start applying one.
    """
    assert sl.USAGE_WINDOW_DAYS == 30, sl.USAGE_WINDOW_DAYS
    assert sl.USAGE_SPAN_NOTICE_SLACK_DAYS == 1.0, sl.USAGE_SPAN_NOTICE_SLACK_DAYS
    body = "\n".join(sl.usage_lines(
        {"days": 30, "events": 3, "sessions": 1, "no_telemetry": False,
         "first_event": "2026-09-25T01:17:43+00:00",
         "until": "2026-09-29T10:34:11+00:00",
         "skills": {"voice-mode": {"offers": 3, "loaded": 1, "ignored": 2,
                                   "max_score": 9.0}}},
        ["voice-mode", "never-seen"]))
    says_none_applied = "no retirement threshold follows from this table at any window depth"
    assert says_none_applied in body, body
    # The one allowed mention is the sentence saying no cutoff exists; strip it, and
    # nothing else may talk about a skill's fate.
    rest = body.replace(says_none_applied, "").lower()
    for verdict in ("retire", "dead", "obsolete", "recommend removal", "stale"):
        assert verdict not in rest, \
            f"the depth section now renders a verdict ({verdict}) about a skill; the " \
            "retirement cutoff is owed, not implemented"


def test_the_window_the_report_asks_for_is_thirty_days():
    """The span the section prints, pinned to the constant rather than to a copy of
    it in prose — and stated here so nobody mistakes it for a threshold.

    Thirty days is the span `skill_injection_counts` is written to take. It is not a
    staleness cutoff and never becomes one: `SKILL_REPORT_TOP_K = 8` means a
    low-scoring skill contributes no rows to any window at all (#1815), and the rows
    beginning 2026-09-25 only says the window is not 30 days deep before
    ~2026-10-24 (#1603's owed entry) — a fact about the span, never about a verdict.
    """
    assert sl.USAGE_WINDOW_DAYS == 30, sl.USAGE_WINDOW_DAYS
    assert sl.USAGE_UNMEASURED_LABEL == "unmeasured"
    assert sl.USAGE_UNMEASURED_LABEL not in ("zero", "never used", "dead")


def test_the_emitter_censors_a_below_rank_8_skill_and_the_table_says_unmeasured(
        tmp_path, monkeypatch):
    """#1923 clauses 3 and 4, across the whole boundary: emitter → JSONL → reader →
    rendered section.

    Twelve offers scoring above `SKILL_REPORT_FLOOR` are handed to the real
    `prefetch._emit_skill_match_events`, which writes a row for eight of them and
    nothing at all for four — that is `scored[:SKILL_REPORT_TOP_K]` in
    `app/prefetch.py` doing the censoring, not a fixture's choice. Two more rows are
    then appended at 29.9 and
    14.0 days back so the store genuinely covers the whole 30-day window: a reader who
    found the section shallow-notice-free has no depth left to blame, and that is
    precisely the state in which a cutoff gets re-proposed. The answer must still be
    `USAGE_UNMEASURED_LABEL` beside the named ceiling and never a zero row, because
    rank 9 emits nothing at any depth (#1815).

    Three ways this can fail, one per thing it pins: drop the slice in
    `_reported_offers` and the four censored skills start emitting, which reddens the
    row-count assertions; re-word the no-threshold sentence back to the store's depth
    and the ceiling assertion reddens; let a fate word (retire, dead, obsolete) into
    any other rendered line and the closing negation reddens. The negation is the one
    `test_the_depth_fix_adds_no_retirement_threshold_and_no_verdict_about_a_skill`
    pins over a shallow hand-built dict; repeated here over the deep one because that
    is the render where a cutoff would next be proposed.
    """
    from app import prefetch

    log_root = tmp_path / "event_logs"
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", log_root)
    monkeypatch.setattr("app.event_log.BLOBS_DIR", log_root / "blobs")
    offers = [(12.0 - 0.1 * i, {"name": f"sk{i:02d}"}) for i in range(12)]
    prefetch._emit_skill_match_events("s-one", offers, [])

    with (log_root / "s-one.events.jsonl").open("a", encoding="utf-8") as fh:
        for age_days in (29.9, 14.0):
            fh.write(_match_row("sk00", landed=False,
                                ts=_ts(int(round(age_days * MINUTES_PER_DAY)))) + "\n")

    usage = sl.collect_usage_counts(days=30, root=log_root)
    assert not usage.get("error"), usage
    assert usage["events"] == 8 + 2, \
        f"rows reached the log past the top-8 ceiling: {usage['events']}"
    assert sorted(usage["skills"]) == [f"sk{i:02d}" for i in range(8)], usage["skills"]

    names = [f"sk{i:02d}" for i in range(12)]
    lines = sl.usage_lines(usage, names)
    body = "\n".join(lines)
    assert not any("shallower" in ln for ln in lines), \
        "the store covers the whole window, so depth is not the reason any more:\n" + body

    listed = _line_with(lines, f"{sl.USAGE_UNMEASURED_LABEL}: ")
    for i in range(8, 12):
        assert f"`sk{i:02d}`" in listed, \
            f"sk{i:02d} is below rank 8 and so was never measured: it must read " \
            f"{sl.USAGE_UNMEASURED_LABEL}, not be absent from the section\n{listed}"
    assert "`sk00`" not in listed, listed
    for i in range(8, 12):
        assert f"| `sk{i:02d}` | 0" not in body, \
            f"sk{i:02d} printed as a zero-offer row, about which the emitter wrote nothing"

    no_threshold = _line_with(lines, "no retirement threshold")
    assert "SKILL_REPORT_TOP_K = 8" in no_threshold, \
        f"the no-threshold sentence must name the ceiling, not the depth: {no_threshold}"
    assert not _DATE.search(no_threshold), \
        f"the no-threshold sentence dates itself: {no_threshold}"
    # The one allowed mention is that sentence. Strip it and no other rendered line of
    # a 30-day-deep section may talk about a skill's fate either — the re-wording added
    # its reason, not a verdict.
    rest = body.replace("no retirement threshold follows from this table at any "
                        "window depth", "").lower()
    for verdict in ("retire", "dead", "obsolete", "recommend removal", "stale"):
        assert verdict not in rest, \
            f"the section renders a verdict ({verdict}) about a skill; the cutoff is " \
            f"owed, not implemented\n{body}"


def _window_comment_block() -> str:
    """The `#:` comment block above `USAGE_WINDOW_DAYS`, sliced out of the source.

    A `#:` comment is unreachable to the AST walk `test_the_span_is_measured_and_no_
    date_literal_produces_it` uses, so slicing the file's text between the block's own
    opening line and the assignment it documents is the only way to hold this prose to
    account. The two `index` calls raise rather than returning empty if either
    boundary moves, so the slice cannot silently read nothing.
    """
    src = (ROOT / "scripts" / "skill_lint.py").read_text(encoding="utf-8")
    start = src.index("#: The window the STALE bucket reports usage over.")
    end = src.index("USAGE_WINDOW_DAYS = 30")
    assert start < end, "the comment no longer sits above the constant it documents"
    return src[start:end]


def test_the_usage_window_comment_names_the_ceiling_as_the_reason_and_not_the_depth():
    """#1923 clause 2: the comment a reader meets `USAGE_WINDOW_DAYS` at is the prose
    no report shows, so it is pinned by slicing it out of the file `sl` was loaded from.

    What has to be in it: the ceiling `SKILL_REPORT_TOP_K = 8` named as the reason no
    staleness or retirement cutoff follows, the statement that a below-rank-8 skill
    contributes no rows to ANY window, the #1815 citation, and the ~2026-10-24 depth
    note confined to the span the printed counts cover. What has to be gone is the old
    licence — `NOT yet` a threshold, which read as a date on which a cutoff became
    derivable. Restoring that sentence turns this node red, and so does deleting the
    ceiling half, which is the thing the clause exists to add.
    """
    block = _window_comment_block()
    assert "Thirty days because" in block, block
    assert "no depth of rows makes it one" in block, block
    assert "SKILL_REPORT_TOP_K = 8" in block, block
    assert "not this store's shallowness" in block, block
    assert "no rows to ANY window" in block, block
    assert "#1815" in block, block
    assert "NOT yet" not in block, f"the comment licenses a cutoff again: {block}"

    depth_note = block[block.index("The depth note"):]
    assert "about this window alone" in depth_note, depth_note
    assert "2026-10-24" in depth_note, depth_note
    assert "bounds the span the printed counts cover" in depth_note, depth_note
    assert "threshold" not in depth_note, \
        f"the depth note reached back into the verdict: {depth_note}"


def test_the_values_the_ceiling_ruling_rests_on_are_unchanged():
    """#1923 clause 4: the ruling is that no cutoff follows, so nothing may have
    quietly become one while the prose saying so was rewritten.

    All four values in one node because they are one claim: `prefetch.SKILL_REPORT_TOP_K`
    is the ceiling that censors the rows, `USAGE_WINDOW_DAYS` is the span the counts
    cover, `USAGE_SPAN_NOTICE_SLACK_DAYS` is the only depth tolerance in the file and
    it gates a sentence about the report rather than a verdict about a skill, and the
    label a no-row skill gets says the instrument could not see it. The rendered half
    below is the other half of the clause: the label still prints beside the ceiling
    that produces it, so the word explains itself to a reader of the report alone.
    """
    from app import prefetch

    assert prefetch.SKILL_REPORT_TOP_K == 8, prefetch.SKILL_REPORT_TOP_K
    assert sl.USAGE_WINDOW_DAYS == 30, sl.USAGE_WINDOW_DAYS
    assert sl.USAGE_SPAN_NOTICE_SLACK_DAYS == 1.0, sl.USAGE_SPAN_NOTICE_SLACK_DAYS
    assert sl.USAGE_UNMEASURED_LABEL == "unmeasured", sl.USAGE_UNMEASURED_LABEL

    body = "\n".join(sl.usage_lines(
        {"days": 30, "events": 3, "sessions": 1, "no_telemetry": False,
         "first_event": "2026-09-25T01:17:43+00:00",
         "until": "2026-09-29T10:34:11+00:00",
         "skills": {"voice-mode": {"offers": 3, "loaded": 1, "ignored": 2,
                                   "max_score": 9.0}}},
        ["voice-mode", "never-seen"]))
    assert f"{sl.USAGE_UNMEASURED_LABEL}: `never-seen`" in body, body
    assert "SKILL_REPORT_TOP_K = 8" in body, body
