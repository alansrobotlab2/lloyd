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
that is the span `skill_injection_counts` takes, and the rows only begin 2026-09-25,
so a 30-day window is not 30 days deep until ~2026-10-24 (#1603's owed list). The
report prints counts; it does not decide which skill is dead.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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


def test_the_window_the_report_asks_for_is_thirty_days():
    """The span the section prints, pinned to the constant rather than to a copy of
    it in prose — and stated here so nobody mistakes it for a threshold.

    Thirty days is the span `skill_injection_counts` is written to take. It is not a
    staleness cutoff, and cannot be one yet: the rows begin 2026-09-25, so the
    window is not 30 days deep before ~2026-10-24, and `SKILL_REPORT_TOP_K = 8`
    means a low-scoring skill contributes no rows to any window at all.
    """
    assert sl.USAGE_WINDOW_DAYS == 30, sl.USAGE_WINDOW_DAYS
    assert sl.USAGE_UNMEASURED_LABEL == "unmeasured"
    assert sl.USAGE_UNMEASURED_LABEL not in ("zero", "never used", "dead")
