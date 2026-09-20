"""A self-contradictory date cannot be injected at a human (#1149).

On 2026-09-14 the morning brief delivered one source email's deadline two ways
fourteen minutes apart: `deadline Fri Sep 18 at 11:59 PM`, then `nominating
window closes Fri Sept 19`. The source said `Nominations will be accepted until
Friday, September 18, 2026 at 11:59 p.m.` The second value is not a reading of
anything — 2026-09-19 is a Saturday, so the pair refutes itself without any
calendar lookup — and it is worse than a wrong header clock (#1079) because it
arrives well-formed and hands the reader a day of slack the source never granted.

These tests do two things. They put a test client across the HTTP seam the way
the real producer crosses it — the brief POSTs to `/inject` or
`/inject-prefetch` through `session_inject_context`, and the guard's answer has
to reach the producer, not just the log — and they pin the skill rule the guard
backs (`_skill()` below): that a date reaches a brief only as a literal copied
from a document the run read, that a truncated literal is `unverified`, and that
a day heading comes from converting the source's instant. The guard refuses a
pair this run invented; only the skill keeps the run from inventing one, so the
two halves are one contract and are pinned in one file.

The skill assertions read the live vault, deliberately unmarked, for the reason
`tests/test_brief_triage_clock_skill.py` gives: the gate runner hardcodes
`-m "not live_vault"`, so a marked check is deselected from the very run meant to
enforce it and pins nothing. A missing vault fails the read rather than passing
an absence.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import sessions_io as sio
from app.date_fidelity import find_contradictory_dates, refusal_detail
from app.routers import sessions as R

SKILL = Path.home() / "obsidian" / "skills" / "morning-brief-and-triage" / "SKILL.md"


def _skill() -> str:
    """The live brief skill. Fails rather than passing an absence."""
    if not SKILL.exists():
        pytest.fail(f"the live vault is not at {SKILL.parent.parent.parent}")
    return SKILL.read_text(encoding="utf-8")

# Pinned so the year the guard assumes is the year these fixtures were written
# for: `Fri Sept 18` is a true pair in 2026 and a false one in 2027, and a test
# that silently changed meaning in January would be worse than no test.
TODAY = datetime.date(2026, 9, 16)          # a Wednesday
BAD = "Brief + Triage — 2026-09-16 15:27 PDT\n\n🔔 Act now:\n- Chaffee Fourth District — kindness awards → deadline closes Fri Sept 19"
GOOD = "Brief + Triage — 2026-09-16 15:30 PDT\n\n🔔 Act now:\n- Chaffee Fourth District — kindness awards → \"Nominations will be accepted until Friday, September 18, 2026 at 11:59 p.m.\""
DATE_FREE = "Brief + Triage — 3 emails kept, 1 escalated to act-now, 2 calendar events."
# The same ordinal with no weekday word to contradict it: nothing here is
# decidable as false, so the guard must stay out of it.
BARE_ORDINAL = "SRS Monthly Meeting — Sept 19, 10am–1pm PDT"
# Two statements, the first ending on a weekday word, the second opening with an
# ordinal. True as written — "since Monday" is not a claim that Sept 19 is a
# Monday — and the first version of the pattern matched `Monday. September 19`
# across the full stop and refused a brief that had done nothing wrong.
SENTENCE_BOUNDARY = ("Nominations were open since Monday. September 19 is when the "
                     "window reopens")
# The shipped artifact sitting inside prose instead of standing alone. Every
# other control here is a single date phrase, so multi-clause text gets both
# directions pinned: this one must still be refused, and it must be named by the
# fragment alone rather than by the sentence around it.
BAD_IN_PROSE = ("Brief + Triage — 2026-09-16 15:27 PDT\n\n🔔 Act now:\n"
                "- Chaffee Fourth District — kindness awards. The nominating window "
                "closes Fri Sept 19, one day later than the source says.")


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    """The guard's default year comes from the server's clock; pin it."""
    monkeypatch.setattr("app.date_fidelity._today", lambda: TODAY)
    # The prefetch queue is a module global; isolate every test from the last.
    monkeypatch.setattr(sio, "_ambient_prefetch_queue", {})


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    """One readable user session plus one worker session nobody reads."""
    monkeypatch.setattr(R, "SESSIONS_DIR", tmp_path)
    for sid, platform in (("human", "mission-control"), ("bot", "worker")):
        (tmp_path / f"{sid}.json").write_text(json.dumps(
            {"session_id": sid, "platform": platform, "messages": []}))
    return tmp_path


@pytest.fixture
def client(sessions_dir):
    app = FastAPI()
    app.include_router(R.router)
    with TestClient(app) as c:
        yield c


# --- the matcher's own controls ------------------------------------------------
# The 09-17 verification pass of this item published 94 offenders instead of 6
# because its month table mapped "sept" to October and so called every
# "Monday, Sept. 14" false. A guard that is wrong in that direction would reject
# real briefs, so both directions are pinned here, one string at a time.

@pytest.mark.parametrize("text", [
    "Monday, Sept. 14",                       # correct, abbreviated month
    "deadline Fri Sep 18 at 11:59 PM",        # the correct restatement of the source
    "Friday, September 18, 2026 at 11:59 p.m.",  # the source's own sentence
    "Wed Sep 16, 2026",
    "Sat Sep 19 10am",                        # 2026-09-19 really is a Saturday
])
def test_a_true_weekday_and_ordinal_pass(text):
    assert find_contradictory_dates(text) == []


@pytest.mark.parametrize("text", [
    "Fri Sept 19",                            # the shipped defect
    "Friday Sept 19",
    "Friday, September 19, 2026",
])
def test_a_weekday_word_that_refutes_its_own_ordinal_is_found(text):
    bad = find_contradictory_dates(text)
    assert len(bad) == 1
    assert bad[0].fragment == text
    assert bad[0].iso == "2026-09-19"
    assert bad[0].actual_weekday == "Saturday"
    assert bad[0].stated_weekday == "Friday"


def test_a_stated_year_is_honoured_rather_than_the_servers():
    # 2025-09-19 genuinely was a Friday, so quoting it is not the defect.
    assert find_contradictory_dates("Fri Sept 19, 2025") == []
    assert len(find_contradictory_dates("Fri Sept 19, 2027")) == 1


def test_an_ordinal_with_no_weekday_word_is_not_guessed_at():
    assert find_contradictory_dates(BARE_ORDINAL) == []


def test_a_relative_word_is_left_to_the_skill():
    """`tomorrow Sep 19` is one of #1149's artifacts, and this guard still ignores it.

    `Tonight! Sept 15` written at 22:49 on Sept 14 is legitimate span-midnight
    usage, and 4 of the 6 offenders in the 173 task-#68 reports since 09-14 are
    relative-label cases — rejecting those would refuse real briefs. Day-bucket
    drift is the skill's rule (Step 4b), not this boundary's.
    """
    assert find_contradictory_dates("SRS Monthly Meeting — tomorrow Sep 19") == []


def test_an_unreadable_month_or_day_is_reported_as_no_verdict():
    """Each of the three skip paths is named by the string that reaches it.

    The first version of this check asserted `"BlorpSept 19 on Friday" == []` and
    passed for the wrong reason: that string is one token to the matcher, so it
    fell into the unknown-*weekday* skip and left the unknown-*month* skip never
    exercised. `Friday Blorp 19` is the string that does exercise it — the weekday
    resolves, the month does not.
    """
    assert find_contradictory_dates("Friday Blorp 19") == []      # month unknown
    assert find_contradictory_dates("Blorp Sept 19") == []        # weekday unknown
    assert find_contradictory_dates("Fri Feb 30, 2026") == []     # no such day
    assert refusal_detail([""]) is None


def test_a_full_stop_between_two_statements_is_not_read_as_one_date():
    """`since Monday. September 19` is two true statements, not a false pair.

    The gap between the weekday word and the month admits spaces, commas, and an
    abbreviation period — `Fri.Sept 19` — but not a period that ends a sentence,
    which is a period with whitespace after it. Without that restriction the
    matcher spans the full stop and a correct brief is refused; `Mon. September 19`
    is missed as a result, and a missed pair costs a brief only its guard.
    """
    assert find_contradictory_dates(SENTENCE_BOUNDARY) == []
    # And the same shape inside a longer brief is still clean at the boundary.
    assert find_contradictory_dates(
        "The office was closed since Monday. September 19, 2026 is the payment date"
    ) == []
    # The abbreviation form is still a pair, and still a false one.
    assert [b.fragment for b in find_contradictory_dates("Friday, Sept. 19")] == \
        ["Friday, Sept. 19"]


@pytest.mark.parametrize("fragment,iso", [
    ("Fri Sept 19", "2026-09-19"),        # the shipped restatement, #1149
    ("Friday Sept 19", "2026-09-19"),     # the same drift, third cycle
    ("Sat Aug 30", "2026-08-30"),         # 2026-08-30 is a Sunday
    ("Sun Aug 31", "2026-08-31"),         # 2026-08-31 is a Monday
    ("Tue Sep 9", "2026-09-09"),          # 2026-09-09 is a Wednesday
    ("Sat Sep 13", "2026-09-13"),         # 2026-09-13 is a Sunday
])
def test_every_fragment_pulled_from_the_real_reports_is_refused_as_written(fragment, iso):
    """The six weekday↔ordinal fragments in the 1,541 task-#68 reports on disk.

    Scanning the `## Response` half of every report with this matcher yields
    exactly these six, in five reports, and all six are genuine contradictions —
    that scan is the guard's false-positive measurement over real traffic, so the
    same six strings are pinned here as the corpus can't be (new reports land in
    the directory every few minutes).
    """
    bad = find_contradictory_dates(f"deadline closes {fragment} at 11:59 PM")
    assert len(bad) == 1
    assert bad[0].fragment == fragment
    assert bad[0].iso == iso


def test_one_pass_names_every_offender_in_a_payload():
    detail = refusal_detail([BAD, "and also Friday, September 19, 2026"])
    assert "Fri Sept 19" in detail
    assert "Friday, September 19, 2026" in detail


# --- clause 1: /inject --------------------------------------------------------

def test_inject_refuses_a_self_contradictory_date_naming_the_fragment(client):
    r = client.post("/api/sessions/human/inject",
                    json={"text": BAD, "source": "autonomy:morning-brief-and-triage"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "Fri Sept 19" in detail
    assert "Saturday" in detail and "Friday" in detail


def test_inject_refusal_does_not_enqueue_a_turn(client, monkeypatch):
    """A 400 must be a non-event: the brief has to fail, not half-deliver."""
    seen: list[str] = []

    async def capture_turn(session_id, text, **kw):
        seen.append(text)
        return {"turn_id": "should-not-exist"}

    async def capture_enqueue(session_id, turn):
        seen.append(f"enqueued:{turn['turn_id']}")
        return {"turn_id": "should-not-exist"}

    monkeypatch.setattr("app.routers.messages.build_ambient_turn", capture_turn)
    monkeypatch.setattr("app.routers.messages.enqueue_ambient", capture_enqueue)

    r = client.post("/api/sessions/human/inject",
                    json={"text": BAD, "source": "autonomy:morning-brief-and-triage"})
    assert r.status_code == 400
    assert seen == []


def test_inject_accepts_the_coherent_pair_unchanged(client, monkeypatch):
    seen = {}

    async def capture_turn(session_id, text, **kw):
        seen["text"] = text
        seen["kw"] = kw
        return {"turn_id": "t-1"}

    async def capture_enqueue(session_id, turn):
        return {"turn_id": "t-1", "dropped": [], "deduped": False}

    monkeypatch.setattr("app.routers.messages.build_ambient_turn", capture_turn)
    monkeypatch.setattr("app.routers.messages.enqueue_ambient", capture_enqueue)

    r = client.post("/api/sessions/human/inject",
                    json={"text": GOOD, "source": "autonomy:morning-brief-and-triage",
                          "priority": "notable"})
    assert r.status_code == 200
    assert seen["text"] == GOOD                      # byte-for-byte, no rewriting
    assert seen["kw"]["priority"] == "notable"
    assert r.json()["turn_id"] == "t-1"


def test_inject_accepts_a_date_free_payload_unchanged(client, monkeypatch):
    seen = {}

    async def capture_turn(session_id, text, **kw):
        seen["text"] = text
        return {"turn_id": "t-2"}

    async def capture_enqueue(session_id, turn):
        return {"turn_id": "t-2", "dropped": [], "deduped": False}

    monkeypatch.setattr("app.routers.messages.build_ambient_turn", capture_turn)
    monkeypatch.setattr("app.routers.messages.enqueue_ambient", capture_enqueue)

    r = client.post("/api/sessions/human/inject", json={"text": DATE_FREE})
    assert r.status_code == 200
    assert seen["text"] == DATE_FREE


def test_a_payload_defect_is_reported_before_the_target_defect(client):
    """400 on a worker session, not 409: the text is wrong whatever it targets.

    `tests/test_active_session_resolution.py` pins the 409 for well-formed text
    aimed at a machine session; this pins that a malformed date is the louder
    answer, so a producer fixing its 400 does not chase a session-routing bug.
    """
    r = client.post("/api/sessions/bot/inject", json={"text": BAD})
    assert r.status_code == 400
    assert "Fri Sept 19" in r.json()["detail"]


def test_inject_refuses_the_bad_pair_found_inside_prose(client):
    """A real brief is paragraphs, not a date phrase standing alone.

    The fragment has to survive being surrounded by sentences: refused, named
    exactly, and the rest of the payload not blamed for it.
    """
    r = client.post("/api/sessions/human/inject",
                    json={"text": BAD_IN_PROSE, "priority": "notable"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "'Fri Sept 19' cannot be right" in detail
    assert "closes" not in detail            # names the pair, not the sentence


def test_inject_accepts_two_sentences_the_matcher_might_have_misread(client, monkeypatch):
    """`…since Monday. September 19…` must arrive, not take the 400.

    This is the false-positive direction, and it is the expensive one: a refused
    injection is a brief the user never sees, and `session_inject_context`
    records the delivery as failed.
    """
    seen = {}

    async def capture_turn(session_id, text, **kw):
        seen["text"] = text
        return {"turn_id": "t-3"}

    async def capture_enqueue(session_id, turn):
        return {"turn_id": "t-3", "dropped": [], "deduped": False}

    monkeypatch.setattr("app.routers.messages.build_ambient_turn", capture_turn)
    monkeypatch.setattr("app.routers.messages.enqueue_ambient", capture_enqueue)

    r = client.post("/api/sessions/human/inject",
                    json={"text": SENTENCE_BOUNDARY, "priority": "notable"})
    assert r.status_code == 200
    assert seen["text"] == SENTENCE_BOUNDARY


def test_inject_prefetch_accepts_two_sentences_the_matcher_might_have_misread(client):
    r = client.post("/api/sessions/human/inject-prefetch",
                    json={"source": "autonomy:morning-brief-and-triage",
                          "summary": "Brief + Triage",
                          "content": SENTENCE_BOUNDARY})
    assert r.status_code == 200
    assert sio.peek_ambient_prefetch("human")[0].content == SENTENCE_BOUNDARY


# --- clause 1 continued: /inject-prefetch -------------------------------------

def test_inject_prefetch_refuses_a_bad_date_in_the_content(client):
    r = client.post("/api/sessions/human/inject-prefetch",
                    json={"source": "autonomy:morning-brief-and-triage",
                          "summary": "Brief + Triage — 1 act-now",
                          "content": BAD})
    assert r.status_code == 400
    assert "Fri Sept 19" in r.json()["detail"]
    assert sio.peek_ambient_prefetch("human") == []      # nothing queued


def test_inject_prefetch_refuses_a_bad_date_in_the_summary(client):
    """The summary alone reaches the <context> block, so it is judged too."""
    r = client.post("/api/sessions/human/inject-prefetch",
                    json={"source": "autonomy:morning-brief-and-triage",
                          "summary": "Brief — deadline Fri Sept 19",
                          "content": DATE_FREE})
    assert r.status_code == 400
    assert "Fri Sept 19" in r.json()["detail"]
    assert sio.peek_ambient_prefetch("human") == []


def test_inject_prefetch_accepts_the_coherent_pair_unchanged(client):
    r = client.post("/api/sessions/human/inject-prefetch",
                    json={"source": "autonomy:morning-brief-and-triage",
                          "summary": "Brief + Triage — 1 act-now",
                          "content": GOOD})
    assert r.status_code == 200
    queued = sio.peek_ambient_prefetch("human")
    assert len(queued) == 1
    assert queued[0].content == GOOD


def test_inject_prefetch_accepts_a_date_free_payload_unchanged(client):
    r = client.post("/api/sessions/human/inject-prefetch",
                    json={"source": "autonomy:morning-brief-and-triage",
                          "summary": "Brief + Triage",
                          "content": DATE_FREE})
    assert r.status_code == 200
    assert len(sio.peek_ambient_prefetch("human")) == 1


def test_inject_prefetch_accepts_an_ordinal_it_cannot_judge(client):
    r = client.post("/api/sessions/human/inject-prefetch",
                    json={"source": "autonomy:morning-brief-and-triage",
                          "summary": "Brief + Triage",
                          "content": BARE_ORDINAL})
    assert r.status_code == 200
    assert sio.peek_ambient_prefetch("human")[0].content == BARE_ORDINAL


# --- the producer's end of the seam -------------------------------------------

def test_the_producer_sees_the_refusal_as_a_failure(client, monkeypatch):
    """`session_inject_context` must report ok=false with the fragment.

    Both ambient priorities route through this one MCP tool — prefetch for
    `ambient`, a turn for `notable`/`urgent` — so if the client swallowed the 400
    the brief's own run would record a delivered notification for a date that was
    never sent.
    """
    from agent_mcp import ambient

    async def fake_post_json(path, body, timeout=10.0):
        r = client.post(path, json=body)
        return r.status_code, r.json()

    async def fake_get_json(path, timeout=5.0):
        return 200, {"session_id": "human"}

    monkeypatch.setattr(ambient, "_post_json", fake_post_json)
    monkeypatch.setattr(ambient, "_get_json", fake_get_json)

    for priority in ("ambient", "notable"):
        out = asyncio.run(ambient._tool_session_inject_context({
            "source": "autonomy:morning-brief-and-triage",
            "summary": "Brief + Triage — 2026-09-16 15:27 PDT",
            "content": BAD,
            "priority": priority,
            "session_id": "human",
        }))
        result = json.loads(out.content[0].text)
        assert result["ok"] is False, priority
        assert result["status"] == 400, priority
        assert "Fri Sept 19" in json.dumps(result["server_response"]), priority


# --- clauses 2 and 3: the skill half of the same contract ---------------------
# The guard above only refuses a pair that refutes itself. It cannot stop a brief
# inventing a date that happens to be internally coherent, and that is the
# original defect (`deadline Fri Sep 18 at 11:59 PM` — a coherent copy — vs
# `closes Fri Sept 19`, refused here; both shipped). The rule that keeps the run
# from inventing one in the first place lives in the brief skill, so it is pinned
# in this file next to the boundary that backs it.

def _flat() -> str:
    """The skill with whitespace runs collapsed, so a re-wrap is not a red run."""
    return " ".join(_skill().split())


def test_the_skill_names_where_a_date_in_a_brief_may_come_from():
    """Clause 2: copied from a source the run read, never reformulated."""
    flat = _flat()
    assert "Every other date in the block has exactly one legal origin: a literal " \
        "**copied** out of a document this run actually read (Step 4a)" in flat
    assert "**Copied, never reformulated.**" in flat
    # The three named sources, one of them the calendar tool's ISO field — the
    # data the 2026-09-16 brief already had in hand and did not use.
    assert "the email subject, or the body returned by Step 3.5a's `email_read`, " \
        "or the calendar tool's own ISO field from Step 2" in flat


def test_the_skill_requires_a_deadline_line_to_quote_the_source_verbatim():
    """Clause 2: the reader can check the date without opening the mail."""
    flat = _flat()
    assert "**A deadline entry carries the source's sentence verbatim**" in flat
    assert "Where the source gives both a weekday word and an ordinal, carry " \
        "**both** and they must agree" in flat


def test_a_truncated_literal_is_unverified_and_never_reported_as_the_source_s_words():
    """Clause 2: #1149 was filed on a false provenance claim of exactly this kind.

    The item's own headline asserted its quotation was "verbatim, confirmed in
    both `text` and `html` body reads" when both reads stopped at the ellipsis, so
    the day number came from inference and was reported as the document's.
    """
    flat = _flat()
    assert "**A literal this run could not see is `unverified`, never the source's " \
        "words.**" in flat
    assert "**A truncated render is `unverified`, not `verbatim`.**" in flat
    assert "if the read truncated before the day number it is `unverified` " \
        "(Step 4a rule 3)" in flat


def test_step_4_forbids_a_typed_date_and_permits_a_copied_one():
    """Clause 2's amendment: the two-date rule and Step 3.5a no longer contradict.

    The #1197 edit made Step 4 say HEADER and TODAY were the only two dates the
    run could state, while Step 3.5a required a quoted body date inside that same
    block — read literally, the skill banned the very copy this item asks for. The
    resolution is the word `produce`: the two measured lines are the only dates the
    run *produces*, and every other date is one it *copies*.
    """
    flat = _flat()
    assert "the only two dates this run is allowed to **produce**" in flat
    assert "Typed, computed, remembered, or reformatted is the same defect " \
        "whatever it is quoting: a date that came from prose." in flat
    assert "what Step 3.5a permits and this step demands is the copy, and what " \
        "this step forbids is the type." in flat
    # The pre-amendment wording, which is what made the two steps collide.
    assert "only two dates this run is allowed to state" not in flat


def test_day_headings_come_from_converting_the_source_instant():
    """Clause 3: Today/Tonight/Tomorrow are derived, not chosen to fit a section."""
    flat = _flat()
    assert "### 4b — Today / Tonight / Tomorrow are converted, never labelled" \
        in _skill()
    assert "Convert it to local time, `America/Los_Angeles`, in this run:" in flat
    # The literal is the Step 4b.3 sentence as it stands in the skill after vault
    # commit `fd5d7eb6` (#1177) rewrote it: `TODAY` is now named as the clock
    # re-read at Step 4 rather than "measured in that same run" — the same
    # requirement, stated with the instant that answers it. The claim this pins
    # (the converted day is compared against a TODAY this run obtained) is
    # unchanged; only its wording moved, and the file's own header says why the
    # pin reads the live vault unmarked. (#1313)
    assert "Compare the **converted calendar day** with `TODAY` from the clock " \
        "**re-read at Step 4**" in flat


def test_an_entry_whose_converted_day_differs_is_filed_truly_or_dropped():
    """Clause 3: the heading agrees with the conversion, never the reverse."""
    flat = _flat()
    assert "filed under its **true converted day or dropped**" in flat
    assert "An entry is **never relabelled to fit the heading**" in flat
    assert "the heading is the thing that has to agree with the conversion, " \
        "not the other way round" in flat


def test_the_two_halves_disagree_only_about_the_invented_value():
    """The skill's worked example and this guard must call the same three strings the
    same way, or the rule the run reads and the refusal it would get are two
    different rules — the failure `tests/test_brief_triage_clock_skill.py` closes
    for the clock stamp.

    All three literals below are quoted in the skill's Step 4a rule 1, so a skill
    edit that changes the example forces this assertion to be re-read against the
    guard rather than quietly desynchronised.
    """
    flat = _flat()
    copied = "`Friday, September 18, 2026 at 11:59 p.m.` stays exactly that"
    correct_restatement = "`deadline Fri Sep 18 at 11:59 PM` (a copy)"
    invented = "`nominating window closes Fri Sept 19` (invented)"
    assert copied in flat and correct_restatement in flat and invented in flat
    assert find_contradictory_dates("Friday, September 18, 2026 at 11:59 p.m.") == []
    assert find_contradictory_dates("deadline Fri Sep 18 at 11:59 PM") == []
    bad = find_contradictory_dates("nominating window closes Fri Sept 19")
    assert [b.fragment for b in bad] == ["Fri Sept 19"]


def test_a_year_on_the_next_line_is_not_pulled_up_into_the_fragment():
    """A bullet list must not donate its second line to the date on the first.

    `Fri Sept 19` on one line and `2025` on the next would, with a whitespace
    hungry year slot, read as `Fri Sept 19, 2025` — a genuinely true pair — and
    the invented date would walk free. The year slot takes spaces and tabs only,
    so the fragment stays year-less and is judged against the server's year.
    """
    bad = find_contradictory_dates("deadline closes Fri Sept 19\n2025 reopened it")
    assert [b.fragment for b in bad] == ["Fri Sept 19"]
    assert bad[0].iso == "2026-09-19"
