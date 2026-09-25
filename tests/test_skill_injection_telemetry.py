"""Reader tests for the #435 skill-injection telemetry.

`prefetch._emit_skill_match_events` writes the rows (pinned end-to-end in
`tests/test_prefetch.py`); this file pins the other half — `app.skill_telemetry`
turning an event-log root plus a day-window into per-skill offer / loaded /
ignored counts, and refusing to read an empty store as "nothing was ignored".

Hermetic: every case seeds its own `<root>/<session_id>.events.jsonl` under
`tmp_path`, so nothing here reads or writes the production event log.

Run: .venvs/lloyd/bin/python -m pytest tests/test_skill_injection_telemetry.py -q
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.skill_telemetry import SKILL_MATCH_EVENT, skill_injection_counts  # noqa: E402


def _stamp(**delta) -> str:
    """An event timestamp in the exact shape `app.event_log` writes: UTC,
    millisecond precision, trailing `Z`."""
    ts = datetime.now(timezone.utc) - timedelta(**delta)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def _match(name, score, landed, *, body=False, ts=None):
    return {
        "ts": ts or _stamp(minutes=5),
        "event": SKILL_MATCH_EVENT,
        "data": {"skill": name, "score": score, "landed": landed,
                 "injected_body": body},
    }


def _seed(root: Path, session_id: str, rows: list[dict]) -> None:
    """Append `rows` the way the writer does: one JSON object per line, the
    per-session filename the reader has to discover on its own."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{session_id}.events.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_all(res: dict) -> dict:
    return res["skills"]


# ── clause 4: per-skill offer / loaded / ignored over an N-day window ─────────

def test_three_seeded_skills_come_back_with_their_three_counts(tmp_path):
    """One full-body load, one excerpt load, one offer that never landed.

    The three states are what the retirement rule turns on: `ignored` is the
    count that has never had a source before #435, because the offered-but-not
    -injected set was discarded inside `_search_skills`.
    """
    _seed(tmp_path, "s1", [
        _match("full", 9.0, True, body=True),
        _match("full", 8.0, True, body=True),
        _match("excerpt", 5.0, True, body=False),
        _match("excerpt", 5.0, False),
        _match("ghost", 3.2, False),
    ])

    res = skill_injection_counts(tmp_path, 7)

    assert _read_all(res) == {
        "full": {"offers": 2, "loaded": 2, "ignored": 0, "max_score": 9.0},
        "excerpt": {"offers": 2, "loaded": 1, "ignored": 1, "max_score": 5.0},
        "ghost": {"offers": 1, "loaded": 0, "ignored": 1, "max_score": 3.2},
    }, res
    assert res["events"] == 5, res      # the five rows seeded above, all in window
    assert res["no_telemetry"] is False


def test_the_window_spans_every_session_file_under_the_root(tmp_path):
    """The reader is given a *root*, not one session's log: offers are spread
    across sessions arbitrarily, so a per-file count would be a different
    quantity than the one the retirement rule needs."""
    _seed(tmp_path, "s1", [_match("shared", 7.0, False)])
    _seed(tmp_path, "s2", [_match("shared", 6.0, True, body=True),
                           _match("only-here", 4.0, False)])

    skills = _read_all(skill_injection_counts(tmp_path, 7))

    assert skills["shared"] == {"offers": 2, "loaded": 1, "ignored": 1,
                                "max_score": 7.0}
    assert skills["only-here"] == {"offers": 1, "loaded": 0, "ignored": 1,
                                   "max_score": 4.0}


def test_rows_outside_the_day_window_are_not_counted(tmp_path):
    """A 14-day window sees the old turn; a 7-day window must not, or "used 30
    times this week" is a statement about some other week."""
    old = _stamp(days=10)
    _seed(tmp_path, "s1", [
        _match("steady", 8.0, False, ts=old),
        _match("steady", 8.0, False, ts=old),
        _match("steady", 8.0, False),
    ])

    week = skill_injection_counts(tmp_path, 7)
    fortnight = skill_injection_counts(tmp_path, 14)

    assert week["skills"]["steady"]["offers"] == 1, week
    assert fortnight["skills"]["steady"]["offers"] == 3, fortnight


def test_other_event_types_in_the_same_file_are_not_offers(tmp_path):
    """`prefetch.skill_match` shares the file with the brain1/inner_voice
    events, so the filter is on the event name, not on the file existing."""
    _seed(tmp_path, "s1", [
        {"ts": _stamp(minutes=1), "event": "brain1.tool_call_proposed",
         "data": {"name": "skills_read", "params": {"name": "unrelated"}}},
        _match("real", 4.0, False),
    ])

    res = skill_injection_counts(tmp_path, 7)

    assert list(res["skills"]) == ["real"], res
    assert res["skills"]["real"]["offers"] == 1


def test_an_unparseable_line_is_skipped_not_silently_dropped(tmp_path):
    """`read_events` skips malformed lines; the reader may not stop at one, and
    may not count it as an offer either — it has to say it skipped it."""
    _seed(tmp_path, "s1", [_match("real", 4.0, False)])
    path = tmp_path / "s1.events.jsonl"
    path.write_text(path.read_text() + '{"ts": "2026-0', encoding="utf-8")

    res = skill_injection_counts(tmp_path, 7)

    assert res["skills"]["real"]["offers"] == 1, res
    assert res["skipped"] == 1, res


def test_a_skill_match_row_without_a_usable_skill_name_is_skipped(tmp_path):
    """A row missing `data.skill` cannot be attributed to a skill. Counting it
    would put a `""` row into the per-skill table, which downstream looks like a
    real (very popular) skill."""
    _seed(tmp_path, "s1", [
        {"ts": _stamp(minutes=2), "event": SKILL_MATCH_EVENT,
         "data": {"score": 5.0, "landed": False}},
        _match("real", 4.0, False),
    ])

    res = skill_injection_counts(tmp_path, 7)

    assert list(res["skills"]) == ["real"], res
    assert res["skipped"] == 1, res


# ── clause 5: an empty store is "no telemetry", never a column of zeros ───────

def test_an_empty_store_reports_no_telemetry(tmp_path):
    """The failure mode this clause exists for: a retire rule reading
    `ignored == 0` out of a store that holds nothing would retire the skills
    that were simply never measured — the mistake `tests/
    test_skill_lint_report_trust.py:71` already pins for skill-lint's STALE=0."""
    res = skill_injection_counts(tmp_path, 7)

    assert res["no_telemetry"] is True, res
    assert res["skills"] == {}, (
        "an empty store must not mint a zero-offer row per skill")
    assert res["events"] == 0
    assert "no telemetry" in res["note"].lower(), res


def test_a_store_full_of_other_events_is_still_no_telemetry(tmp_path):
    """The sharper half of the clause: the directory is not empty — it holds
    thousands of tool-call events — so "the store exists" cannot be the test
    for whether skill injection was ever recorded."""
    _seed(tmp_path, "s1", [
        {"ts": _stamp(hours=3), "event": "brain1.tool_call_proposed",
         "data": {"name": "Read", "params": {"file_path": "x"}}},
        {"ts": _stamp(hours=3), "event": "inner_voice.decision",
         "data": {"action": "none"}},
    ])

    res = skill_injection_counts(tmp_path, 7)

    assert res["no_telemetry"] is True, res
    assert res["skills"] == {}
    assert res["events"] == 0


def test_old_rows_only_is_no_telemetry_for_that_window(tmp_path):
    """`no_telemetry` is scoped to the window: a store whose only skill rows
    predate it says so, rather than reporting zeros for this window."""
    _seed(tmp_path, "s1", [_match("aged", 9.0, False, ts=_stamp(days=40))])

    res = skill_injection_counts(tmp_path, 7)

    assert res["no_telemetry"] is True, res
    assert res["skills"] == {}
    # Positive control: the row is there, the window is what excludes it.
    assert skill_injection_counts(tmp_path, 60)["skills"]["aged"]["offers"] == 1


def test_the_reader_does_not_touch_the_uptake_instrument_it_complements():
    """`app/uptake.py` still owns the *load* half, through its own proxy:
    `SKILL_PRESENCE_PROXY = "proxy:skills_read+injected_context (#435 pending)"`
    at `app/uptake.py:173`, with `SKILL_PRESENCE_NOTE` at `:187-191` calling it
    "Not per-injection telemetry".

    This module adds only the offer half, so pin the two boundaries this change
    must not cross: it neither imports uptake nor writes under `eval/uptake/`
    (those tables are uptake's own artifacts), and the proxy constant is
    untouched — retiring its `(#435 pending)` marker needs a day of real
    `prefetch.skill_match` traffic first, which is the post-landing person
    check, and deleting the constant would silently drop `presence_source` from
    every skill row of every table uptake writes.
    """
    src = (ROOT / "app" / "skill_telemetry.py").read_text(encoding="utf-8")
    assert "import uptake" not in src and "eval/uptake" not in src, (
        "the offer-half reader reached into the load-half instrument's artifacts")

    uptake = (ROOT / "app" / "uptake.py").read_text(encoding="utf-8")
    assert ('SKILL_PRESENCE_PROXY = "proxy:skills_read+'
            'injected_context (#435 pending)"') in uptake, (
        "the proxy constant moved or lost its marker; the marker is a person's "
        "post-landing edit once live traffic confirms the events, not this "
        "round's to retire")


def test_counts_are_not_clipped_by_the_event_reader_default_page(tmp_path,
                                                                  monkeypatch):
    """`app.event_log.read_events` is that module's paginated UI reader:
    keyword-only `(session_id, *, offset, limit=200)`, no root parameter. A
    30-day window at `SKILL_REPORT_TOP_K` = 8 is comfortably more than 200 rows
    for one session, so a counts reader written on top of it would report the
    tail of each session and drop the rest — numbers low enough to read as a
    dead skill, the exact misreading this item was filed against. Hence the
    hand-rolled scan.

    Seeded: 250 offers for one skill inside the window, plus one 40 days back
    that the window must still exclude (otherwise "not clipped" could be
    satisfied by ignoring the window entirely).
    """
    rows = [_match("deep-history", 3.0, False) for _ in range(250)]
    _seed(tmp_path, "s1", rows + [_match("deep-history", 3.0, False,
                                         ts=_stamp(days=40))])

    res = skill_injection_counts(tmp_path, 30)

    assert res["skills"]["deep-history"]["offers"] == 250, (
        f"count clipped, not windowed: {res['skills']['deep-history']}")
    assert res["events"] == 251 - 1, res
    assert res["window_days"] == 30
    # Positive control for the clip itself: the paginated reader really does
    # stop at 200, so the assert above is about this reader, not about a page
    # size nobody would have tripped.
    from app.event_log import read_events
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path)
    page = read_events("s1")
    assert len(page) == 200, f"read_events' default page changed: {len(page)}"


def test_the_writer_and_the_reader_name_one_event_type():
    """The emitter's constant lives in `prefetch`, this reader's in
    `app.skill_telemetry` — deliberately duplicated, because importing
    `prefetch` from a script drags in the MCP tool modules. A drift between the
    two is the worst failure this instrument can have: the writer keeps writing
    and the reader keeps returning `no_telemetry` over a store that is actually
    full, which is indistinguishable from "nobody prefetched anything".
    """
    import prefetch

    assert prefetch.SKILL_MATCH_EVENT == SKILL_MATCH_EVENT == "prefetch.skill_match"


# ── P5: the pull arm, and reads counted as loads ──────────────────────────────

def _read_call(skill, *, ts=None, args_as_dict=False):
    args = {"summary": "Reading a skill", "name": skill}
    return {"ts": ts or _stamp(minutes=4), "event": "brain1.tool_call_proposed",
            "data": {"tool_call_id": "c1", "name": "skills_read",
                     "args": args if args_as_dict else json.dumps(args)}}


def test_pull_arm_still_reports_offers(tmp_path, monkeypatch):
    """`prefetch.skills.push: false` renders no skill body, and the offer rows
    are still written — every one `landed: false` — so the pull arm's
    telemetry says what the turn was offered and that none of it was pushed.
    The model's own `skills_read` then shows up as `loaded_by_read`."""
    import prefetch
    from app.config import CONFIG

    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path)
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "blobs")
    monkeypatch.setitem(CONFIG, "prefetch", {"skills": {"push": False}})
    offers = [(9.0, {"name": "alpha", "raw": "alpha body"}),
              (8.5, {"name": "beta", "raw": "beta body"})]

    plan = prefetch._skill_injection_plan(prefetch._injectable_skills(offers))
    assert plan == []
    assert "<skill" not in prefetch._format_context(offers, [], show_skill_hint=False)
    prefetch._emit_skill_match_events("pull-sess", offers, plan)
    _seed(tmp_path, "pull-sess", [_read_call("alpha")])

    res = skill_injection_counts(tmp_path, 7)
    assert res["no_telemetry"] is False
    assert res["skills"]["alpha"] == {"offers": 1, "loaded": 0, "ignored": 1,
                                      "max_score": 9.0}
    assert res["skills"]["beta"]["offers"] == 1 and res["skills"]["beta"]["loaded"] == 0
    assert res["loaded_by_read"] == {"alpha": 1}


def test_loaded_by_read_counts_skills_read_calls_in_the_window(tmp_path):
    """Reads are their own mapping: an offer entry keeps its exact shape, a
    read-only skill is not minted as a zero-offer measured skill, and reads
    alone do not turn `no_telemetry` off (that flag is about offers)."""
    _seed(tmp_path, "s1", [
        _read_call("alpha"), _read_call("alpha", args_as_dict=True),
        _read_call("beta", ts=_stamp(days=40)),          # outside the window
        {"ts": _stamp(minutes=3), "event": "brain1.tool_call_proposed",
         "data": {"name": "skills_search", "args": '{"query": "alpha"}'}},
        {"ts": _stamp(minutes=3), "event": "brain1.tool_call_proposed",
         "data": {"name": "skills_read", "args": "{not json"}},
    ])

    res = skill_injection_counts(tmp_path, 7)
    assert res["loaded_by_read"] == {"alpha": 2}
    assert res["reads"] == 2
    assert res["skills"] == {} and res["no_telemetry"] is True
