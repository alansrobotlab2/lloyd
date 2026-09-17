r"""`skills/morning-brief-and-triage/SKILL.md` must derive its dates, not write them.

Backlog #1197. Autonomy task #68 runs this skill every ~15 minutes and injects a
brief. The brief of 2026-09-16 10:32Z printed `Brief + Triage — 2026-09-17 02:53
PDT` and `📅 Today: (no events)` while Ben's Birthday sat confirmed on the 16th.
Not a stale value and not a timezone bug: the run had no clock anywhere in its
context, so it invented one. The template it filled in was a literal
`YYYY-MM-DD`, and the *same* hand-typed value fed both the display header and the
`calendar_events` window arguments — so the invented day selected which day got
queried, the query returned `[]` legitimately, and `status: success` was honest.
A transcript pass over `autonomy-task:68` sessions found 58 of 147 dated headers
(39%) naming a calendar day other than their own run's; at ~34 runs per 6 h that
mis-reports the calendar roughly 130 times a day.

The skill half of the fix is what these tests pin: the run fetches the wall clock,
and every date it emits is a paste of that fetch. The other half — the server
stamping each ambient delivery with its own reading — is
`tests/test_prefetch.py`; the last test here checks the two halves agree on the
words, because a skill that promises a stamp the delivery path does not send is
the same defect wearing a different hat.

These assertions read the live vault, so they can go red from a nightly skills
pass rather than from the change under review. That is deliberate and is the same
trade `tests/test_skill_tool_names.py` and `tests/test_yaml_fix_skill_claims.py`
make: the gate runner hardcodes `-m "not live_vault"`, so a marked check is
deselected from the run meant to enforce it and pins nothing. Nothing here skips,
and a missing vault fails the read rather than passing an absence.

Measured denominators, so a later reader can tell a real 0 from an empty pattern:
`grep -c 'YYYY-MM-DD'` over this file was 5 before the change (`:88` `:89` `:154`
`:178` `:181`) and is 0 after; `grep -nE "`date|%Y-%m-%d|now\(\)|TZ=" SKILL.md`
before the change hit exactly one line — `datetime.utcnow()` feeding only
`state.json`'s `last_run` — which is why there was no clock to copy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL = Path.home() / "obsidian" / "skills" / "morning-brief-and-triage" / "SKILL.md"
NO_VAULT = "the live vault is not at"


def _text() -> str:
    if not SKILL.exists():
        pytest.fail(f"{NO_VAULT} {SKILL.parent.parent.parent}")
    return SKILL.read_text(encoding="utf-8")


# ── Clause 3: the emitted dates are fetched, not typed ───────────────────────

def test_the_skill_fetches_the_wall_clock_before_it_states_any_date():
    """There must be a `date` invocation in the box's own zone ahead of every
    date the run emits. Before this change the only clock read in the file was
    `datetime.utcnow()` inside Step 5's state writer — after the header and the
    calendar call — so nothing existed to copy and the model composed a day."""
    body = _text()
    fetches = [m.start() for m in
               re.finditer(r"TZ=America/Los_Angeles date '", body)]
    assert len(fetches) >= 2, (
        "the skill must fetch the clock (Step 0, and again immediately before "
        f"composing); found {len(fetches)} fetches")
    first_fetch = fetches[0]
    window = body.find("calendar_events(", first_fetch)
    assert window > first_fetch, "the calendar call must come after the fetch"
    header = body.find("Brief + Triage — <", first_fetch)
    assert header > first_fetch, "the header template must come after the fetch"


def test_no_literal_date_template_is_left_in_the_skill():
    """`YYYY-MM-DD` is the shape that was copied into both the header and the
    window arguments. Zero hits, not "every hit is guarded": a placeholder that
    sits next to an instruction to paste a real value still gets completed from
    the pattern when the run is many turns past the tool output that held it.
    """
    body = _text()
    hits = [i + 1 for i, line in enumerate(body.splitlines())
            if "YYYY-MM-DD" in line]
    assert hits == [], f"literal date templates remain on lines {hits}"


def test_the_calendar_window_is_the_fetched_values_and_carries_no_typed_offset():
    """The window arguments are where the birthday was lost, so they are the
    load-bearing pair: `START`/`END` pasted from Step 0, and no hand-written
    `-07:00` — that offset is only true for half the year and a remembered one
    is exactly the kind of unmeasured value this item is about.
    """
    body = _text()
    call = re.search(r"calendar_events\(\s*\n(.*?)\)", body, re.S)
    assert call, "the calendar_events call template is missing"
    args = call.group(1)
    assert "<START from Step 0" in args and "<END from Step 0" in args, args
    assert "-07:00" not in args and "-08:00" not in args, args
    assert re.search(r"\d{4}", args) is None, f"a numeric date survived: {args}"


def test_the_window_queried_is_echoed_beside_the_count():
    """A negative result has to name its denominator, or a query aimed at the
    wrong day is indistinguishable from a quiet one in the artifact."""
    body = _text()
    echo = re.search(r"calendar window queried:.*?events returned", body, re.S)
    assert echo, "Step 2 must echo the queried window with the returned count"
    assert "Step 0" in echo.group(0), "the echo must be the fetched day, not a typed one"


# ── Clause 4: an empty `Today:` prints its window and count on the same line ──

def test_an_empty_today_line_carries_its_denominator():
    """At least one *form* the run may emit for an empty day must print the
    window and the count on that same line. Matched over every occurrence of the
    zero-event string, because the page also narrates the incident with the same
    words — and it is the emitted form, not the narration, that has to carry the
    denominator.
    """
    body = _text()
    candidates = [line for line in body.splitlines()
                  if "📅 Today: (no events)" in line]
    assert candidates, "the composed-signal block lost the zero-event form"
    carried = [line for line in candidates
               if re.search(r"\b0 events\b", line)
               and re.search(r"window <TODAY", line)]
    assert carried, (
        "no zero-event form prints its count and window on the same line; the "
        f"forms present were: {candidates}")


# ── The two halves speak the same words ──────────────────────────────────────

def test_the_stamp_the_skill_promises_is_the_stamp_the_server_sends():
    """The skill tells the run to expect a server-computed clock beside its own
    signal. That sentence is only true if both delivery paths actually render
    one, so this checks the phrases against the real renderers rather than
    against each other: the `<ambient-signals>` drain and the `<ambient …>`
    envelope.
    """
    body = _text()
    assert "server clock when queued" in body
    assert "server_clock=" in body

    import prefetch
    from app.routers import messages as M
    from app.sessions_io import AmbientPrefetchEntry, ambient_clock_stamp

    rendered = prefetch._format_context(
        skills=[], fact_lines=[],
        ambient_entries=[AmbientPrefetchEntry(
            source="autotriage", summary="Brief + Triage", enqueued_at=1.0)])
    assert "server clock when queued" in rendered

    envelope_src = Path(M.__file__).read_text(encoding="utf-8")
    start = envelope_src.index("async def build_ambient_turn")
    end = envelope_src.index("async def enqueue_ambient", start)
    assert "server_clock=" in envelope_src[start:end]

    # And the formatter both paths share is the one the skill's prose describes:
    # a box-local zone abbreviation, not a bare UTC instant.
    stamp = ambient_clock_stamp(1773032767.0)  # 2026-03-04T05:06:07Z
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} [A-Z]{3,4}", stamp), stamp
    assert stamp.endswith(("PST", "PDT")), stamp


def test_the_skill_still_loads_through_the_real_loader():
    """Front matter must keep parsing — `agent_mcp/skills.py:_parse_frontmatter`
    swallows every exception, so a degraded page is served with a truncated
    description and nothing raises. Prefetch injects `skill["body"]`, so this
    page's text is what an autonomy run reads.
    """
    from agent_mcp import skills

    if not SKILL.exists():
        pytest.fail(f"{NO_VAULT} {SKILL.parent.parent.parent}")
    loaded = skills._load_skill(SKILL.parent)
    assert loaded, f"{SKILL} no longer loads through _load_skill"
    assert loaded["name"] == "morning-brief-and-triage"
    assert "TZ=America/Los_Angeles date" in loaded["raw"]
