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

The five checks under `# ── #1177 calendar rules ──` (backlog #2091) pin the four
calendar clauses vault commit `fd5d7eb6` landed on this same page: the Step 2
extraction pair, Step 4d's modality-from-the-field rule, Step 4c's already-ended
rule (including the all-day end), Step 4b.3's re-read-clock rule, and the
one-row-per-VEVENT list. They read the delivered text, and every one of them also
asserts its own pattern is ABSENT from `925fff79`, the page as it stood before
`fd5d7eb6` — so a check that has silently stopped discriminating goes red at the
gate rather than restating a rule that was always there. Measured denominators for
that comparison: the delivered page is 71,403 B against `925fff79`'s 55,993 B, and
`925fff79` holds 0 hits for `onlineMeetingURL` (5 lines / 6 occurrences delivered),
`comma-joined` (3 delivered), `not a header stamp` (1), `Step 0a's copy` (2) and
`ENDED … ago` (2). To run the five cold against that pre-image instead of the
delivered page — the check that they fail, not merely that they pass — set
`LLOYD_SKILL_PRE1177=1`, which points every read here at that one blob.
"""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

SKILL = Path.home() / "obsidian" / "skills" / "morning-brief-and-triage" / "SKILL.md"
NO_VAULT = "the live vault is not at"
VAULT = SKILL.parents[2]
SKILL_REL = "skills/morning-brief-and-triage/SKILL.md"

# `fd5d7eb6` (#1177: calendar rows carry the source's location, never a guessed
# modality, never an ended event) and its parent. The parent is the pre-image the
# five #1177 checks must NOT match, which is what makes them witnesses of that
# commit instead of witnesses of the page in general.
PRE_1177_REV = "925fff79"
PRE_1177_ENV = "LLOYD_SKILL_PRE1177"

# Positive control for the pre-image read: the thing #1177 deleted. A `git show`
# that returned an empty string would otherwise satisfy every absence-check below
# and report a green discriminator that never read a page.
PRE_1177_CONTROL = "[event 1 at time],[event 2 at time]"


@lru_cache(maxsize=1)
def _pre_1177_text() -> str:
    """The page as `git -C ~/obsidian show 925fff79:<skill>` holds it.

    Fails rather than returning `""`: every #1177 check below asserts this text
    does NOT match its pattern, so a silent empty read would turn five
    discriminators into five free passes. The control is the comma-joined calendar
    template `fd5d7eb6` removed (1 hit in this blob, 0 in the delivered page), so
    the read is proven to have returned the old page, not just some bytes.
    """
    if not VAULT.joinpath(".git").exists():
        pytest.fail(f"{NO_VAULT} {VAULT}")
    proc = subprocess.run(
        ["git", "-C", str(VAULT), "show", f"{PRE_1177_REV}:{SKILL_REL}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"`git -C {VAULT} show {PRE_1177_REV}:{SKILL_REL}` exited "
                    f"{proc.returncode}: {proc.stderr.strip()}")
    blob = proc.stdout
    if blob.count(PRE_1177_CONTROL) != 1:
        pytest.fail(
            f"pre-#1177 read is not the old page: {len(blob.encode()):,} B with "
            f"{blob.count(PRE_1177_CONTROL)} hits for the comma-joined calendar "
            f"template (expected 55,993 B and 1)")
    return blob


def _text() -> str:
    if os.environ.get(PRE_1177_ENV) == "1":
        return _pre_1177_text()
    if not SKILL.exists():
        pytest.fail(f"{NO_VAULT} {SKILL.parent.parent.parent}")
    return SKILL.read_text(encoding="utf-8")


def _pin_1177_rule(*pairs: tuple[str, str]) -> None:
    """Assert each `(pattern, why)` matches the page under test and not the pre-image.

    Two assertions per pattern, and the second is the one that keeps the first
    honest. A pattern that matches `925fff79` too is not pinning anything #1177
    did — it is pinning prose that was already standing, and the page could lose
    the rule while the check stays green. With `LLOYD_SKILL_PRE1177=1` the page
    under test IS that blob, so every check routed through here fails on the first
    assertion: that run is the proof these five discriminate.

    Flags are `re.M | re.S` for every pattern here: `re.M` for the `^📅 Today:$`
    heading, `re.S` for the rules whose sentences the page wraps across lines.
    """
    body, pre = _text(), _pre_1177_text()
    for pattern, why in pairs:
        rx = re.compile(pattern, re.M | re.S)
        assert rx.search(body), f"{why}\npattern {pattern!r} did not match {_source()}:"
        assert not rx.search(pre), (
            f"{why}\n{PRE_1177_REV} (the page before fd5d7eb6/#1177) matches pattern "
            f"{pattern!r} too, so this check restates prose that predates the rule "
            "instead of pinning it")


def _source() -> str:
    return (f"{VAULT}@{PRE_1177_REV}:{SKILL_REL} (named by {PRE_1177_ENV}=1)"
            if os.environ.get(PRE_1177_ENV) == "1" else str(SKILL))


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

    from app import prefetch
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


# ── #1177 calendar rules: the four clauses fd5d7eb6 landed, pinned in code ─────
#
# The delivered brief that #68 would have produced is unreachable while that task
# stays parked (Alan's ruling of 2026-09-17), so no pre-landing check can ever
# confirm a calendar line quoted its source `location` verbatim. These five are
# therefore the only witness that the clauses survive a later skills pass, and
# each one is written to fail on `925fff79` — see `_pin_1177_rule`. Patterns are
# copied from the delivered page, not from the backlog item that drafted them:
# three of the drafted probe strings do not exist in the text that shipped
# (`Never one comma-joined line` is lowercase, `re-read wins` is split by markdown
# bold and a newline, and the Step 4d table lists the URL row before the location
# row). Matching a string nobody wrote is a green check over nothing.

def test_the_extraction_list_names_the_location_and_the_meeting_url_pair():
    """Step 2 has to carry `location` and `onlineMeetingURL` forward, and has to
    say an empty string is a value rather than a gap.

    The 2026-09-15 brief printed `(both online)` for two meetings whose payloads
    held `"location":""` and `"onlineMeetingURL":null`: with no source field in
    hand the run invented a modality from the words `Discussion Group`. Before
    `fd5d7eb6` the extraction list named title, time and calendar only, so there
    was nothing to copy — 0 hits for `onlineMeetingURL` in that page, 5 lines / 6
    occurrences in this one.
    """
    _pin_1177_rule(
        (r"Extract:[\s\S]{0,600}?\*\*`location`\*\*,\s*and\s*\*\*`onlineMeetingURL`\*\*",
         "Step 2's extraction list must name both source fields for *where* a meeting is"),
        (r"\*\*including the empty string and the bare `null`, which are values and not\s+gaps\*\*",
         "an empty `location` must be carried as a value, not read as a gap to fill"),
        (r'`"location":""` with `"onlineMeetingURL":null`',
         "the page must record what the tool actually returns for an event with no venue"),
        (r"`location` and `onlineMeetingURL` are both empty",
         "the both-empty case needs a named form, or the run fills the gap with a guess"),
    )


def test_a_modality_word_requires_a_non_empty_source_field_in_the_table_s_row_order():
    """Step 4d: a row may say `online`, `in person`, `hybrid` or `(both online)`
    only when that event's own payload carries a source, and the source is copied
    verbatim instead of being categorised.

    Pinned against the table as it shipped: `onlineMeetingURL` non-empty is the
    first row and `location` non-empty the second, so one pattern over all three
    rows fixes the order too — a table that quietly swapped them would still be
    fine, but one that dropped the both-empty row would let a guess back in, and
    that row is the third term of the same pattern.
    """
    _pin_1177_rule(
        (r"A calendar row may assert modality[\s\S]{0,900}?"
         r"\|\s*`onlineMeetingURL` non-empty[\s\S]{0,400}?"
         r"\|\s*`location` non-empty[\s\S]{0,400}?"
         r"\|\s*both empty, absent, or `null` \| \*\*nothing about modality\.\*\*",
         "Step 4d must gate every modality word on a non-empty field, URL row first"),
        (r"the row simply has no `@ …` half",
         "both-empty must render as silence, not as a placeholder that reads as a venue"),
        (r"\*\*The value is copied, never categorised\.\*\*",
         "a URL renders as the URL and an address as the address, never as a category"),
        (r"\*\*An empty `location` is not evidence of anything\.\*\*",
         "the page must say out loud that an empty field licenses no claim"),
    )


def test_an_event_that_ended_before_composition_is_dropped_or_marked_ended():
    """Step 4c: every row is tested against the clock re-read immediately before
    composing, an ended event defaults to omission, and an all-day event's end is
    the end of its converted day — never the midnight its `endDate` prints.

    The 2026-09-16T03:00:12Z inject said `📅 Tonight: DPRG Weekly Meeting @ 5:30
    PM, ROS Discussion Group @ 7:00 PM` under a header six hours older, with both
    events ended in the same minute the inject arrived. The all-day half is the
    same failure inverted: this calendar returns the *same* midnight in both
    `startDate` and `endDate` for an all-day event, so a literal `endDate <= now`
    test retires every birthday at 00:01. `ENDED … ago` is 0 hits in `925fff79`
    and exactly 2 here: the Step 4c past-marker row and the Step 6 checklist
    restatement.
    """
    body = _text()
    hits = [m.group(0) for m in re.finditer(r"ENDED [^\n]*ago", body)]
    assert len(hits) == 2, (
        "the past marker must appear in Step 4c's table and once more in the Step 6 "
        f"checklist; found {len(hits)}: {hits}")
    _pin_1177_rule(
        (r"\(ENDED \[converted end\] — \[N\]h ago\)",
         "when naming an ended event carries information, the marker must be explicit"),
        (r"\|\s*end is at or before now\s*\|\s*\*\*omit it\*\* — the default\. "
         r"A finished event is not a plan",
         "omission must be the default row, not one option among several"),
        (r"re-read \*\*immediately before\s+composing\*\*",
         "the comparison instant is Step 4's re-read, the one the header is stamped with"),
        (r"\*\*An all-day event's end is the end of its converted calendar day\*\*,\s*"
         r"not the instant its\s+`endDate` field prints",
         "an all-day row must run to the end of its converted day"),
        (r"event is in progress until\s+23:59 in its converted day and is never "
         r"`ENDED` while that day is `TODAY`",
         "an all-day event must never be retired as ended while its day is today"),
        (r"never to the midnight its `endDate` field prints",
         "the Step 6 checklist must restate the all-day rule, not just Step 4c"),
    )
    assert _pre_1177_text().count("ENDED") == 0, (
        "positive control: the pre-image must hold no `ENDED` marker at all, or the "
        "count above is measuring an unread page")


def test_a_relative_label_is_computed_from_the_re_read_clock_not_a_stamped_one():
    """Step 4b.3: `today` / `tonight` / `tomorrow` come from the instant Step 4
    re-read, and where that disagrees with Step 0a's copy the re-read wins.

    A brief of 2026-09-16 called an event `tomorrow Sep 19` in a run whose own
    `started_at` was 2026-09-16T14:13:39Z while the payload said
    `2026-09-19T17:00:00.000Z` — two days out, from a label chosen to fit a
    heading. Note the shipped text writes the resolution as `the **re-read` /
    `wins**` across a line break, so the drafted probe string `re-read wins`
    matches nothing here and this file pins the split form instead.
    """
    _pin_1177_rule(
        (r"not Step 0a's copy of it, not a header stamp",
         "Step 4b.3 must name the two stale instants it forbids, in one sentence"),
        (r"\*\*re-read\s+wins\*\*",
         "a disagreement between the two clocks must resolve toward the re-read"),
        (r"never Step 0a's copy or a remembered header stamp",
         "the Step 6 checklist must restate the re-read-clock rule for the header word"),
    )


def test_the_calendar_section_is_a_list_and_never_a_comma_joined_line():
    """One row per VEVENT, and a merge has to announce itself.

    The page #1177 replaced emitted `📅 Today: [event 1 at time],[event 2 at
    time],...` — one line for the whole section, which is exactly where the
    `ROS Discussion group` / `HBRC ROS Discussion Group` collapse hid: two
    VEVENTs sharing 19:00–20:00 PDT rendered identically to one commitment, so
    the artifact could not distinguish a quiet calendar from a merged one. That
    comma-joined template is this file's positive control for the pre-image read
    (`_pre_1177_text`), 1 hit there and 0 here; `comma-joined` itself is 0 hits in
    `925fff79` and 3 here, all of them the rule against it.
    """
    _pin_1177_rule(
        (r"^📅 Today:$",
         "the heading must stand alone above the rows, carrying no events itself"),
        (r"📅 Today:\n- \[title\] — \[converted start–end\] @ \[location verbatim\]\n"
         r"- \[title\] — \[converted start–end\]\n```",
         "the composed block must show two rows, the second the both-empty form"),
        (r"The 📅 block above is a \*\*list\*\*: one row per VEVENT, in start-time order",
         "the section must be declared a list, in start-time order"),
        (r"\*\*One row per VEVENT — never one comma-joined line for the whole section\.\*\*",
         "the rule against a comma-joined section must be stated as a rule"),
        (r"`2 entries merged: <title A> / <title B>`",
         "a merge the page does not license must at least be readable as a merge"),
    )
