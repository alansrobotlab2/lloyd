"""Claims `architecture/background-runs.md` must not make: the chat summarizer
(#1228) and the mis-labelled class's archive age (#2651).

The #1228 paragraph said `_summarize_chat` ran on every `POST /api/mc/state`,
opened every file with no ceiling, and cost 238 ms at 1,420 files. The first two
were false and the third was a carried-over figure from a directory five months
younger. What it may say is pinned here as behaviour the doc must describe.

The #2651 nodes below pin the other correction: the page dated the Background
tab's never-aged-out exemption at 30 days in four places, and the sweep archives
that class at 90.

The #2917 nodes at the bottom take the third: §8's census sentence named one
in-tree route that removes a transcript where the router declares two, and read a
missing witness out of a store that holds no transcript by any name for nine days
either side of it.

The #4114 nodes take the half of that reading the server disproves: uvicorn's INFO
access logger records a call to either route with method, path, peer and status, so
"neither leaves a per-session record" was false for the DELETE route and true only in
a narrower way for the clear route (its id is in the body). What is left to pin is
the asymmetry, and the record's rotation-bounded life inside `server.log` — the bound
that actually explains why the vanished twelve cannot be accounted for.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "background-runs.md"


def _paragraph() -> str:
    text = DOC.read_text(encoding="utf-8")
    assert text.count("_summarize_chat") >= 1, "the doc no longer names the chat summarizer"
    for para in re.split(r"\n\s*\n", text):
        if "_summarize_chat" in para:
            return " ".join(para.split())
    raise AssertionError("unreachable")


def test_the_falsified_claims_are_gone():
    text = DOC.read_text(encoding="utf-8")
    for stale in ("238 ms", "1,420", "#1017", "POST /api/mc/state"):
        assert stale not in text, stale
    para = _paragraph()
    for stale in ("no ceiling", "opens every file"):
        assert stale not in para, stale


def test_the_paragraph_states_its_call_path_scaling_and_a_dated_figure():
    para = _paragraph()
    assert "only from `mc_navigate`" in para
    assert "scales with the background share" in para
    ms = re.search(r"~?\d+(?:\.\d+)? ms", para)
    assert ms, para
    assert re.search(r"\b20\d\d-\d\d-\d\d\b", para), "a timing needs the date it was measured on"


# ---------------------------------------------------------------------------
# #2651 — when the mis-labelled class actually expires
# ---------------------------------------------------------------------------
#
# The page said, in four places, that the Background tab's "never aged out"
# exemption for a four-part id carrying a human platform lasts 30 days, because
# the retention sweep archives a background session at
# `BACKGROUND_SESSION_ARCHIVE_AGE_DAYS`. `_archive_age_for`
# (`scripts/groundskeeper/retention-sweep.py:607-619`) takes that age from the
# `platform` field, and the class is defined by its platform NOT being in
# `NON_USER_PLATFORMS = ("autonomy", "worker")`, so every member gets
# `SESSION_ARCHIVE_AGE_DAYS = 90`. Measured through the sweep's own function over
# synthetic four-part sessions: at 45 days `mission-control`, `""` and `browser`
# each returned `archive_age=90` and were not archived, while `autonomy` returned
# 30 and was; at 91 days the class member went. The nodes below pin what the page
# may now say — nothing here imports the sweep, because the sweep is not what
# changed and is not what these nodes can break.


def _section(heading: str) -> str:
    """The body of one `## ` section, up to the next one.

    Pinning a section rather than the page is what keeps the correction honest:
    § Retention (inside §8) has always been right about the 30/90 split, and a
    page-wide absence check would have to dodge it.
    """
    text = DOC.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(heading)}", text, re.M)
    assert m, f"the page has no section headed '## {heading}...'"
    nxt = re.search(r"^## ", text[m.end():], re.M)
    return text[m.end(): m.end() + (nxt.start() if nxt else len(text))]


def _flat(s: str) -> str:
    """One line, so a claim split across the page's 79-column wrap still matches."""
    return " ".join(s.split())


def test_the_page_no_longer_dates_the_classs_expiry_at_a_thirty_day_archive():
    """Clause 1. `holds for 30 days` was §8's sentence; the class is on the
    conversation clock, so the phrase has to be gone page-wide and §8 has to say
    what actually bounds the exemption: live, then archived at 90 days."""
    text = _flat(DOC.read_text(encoding="utf-8"))
    assert "holds for 30 days" not in text, (
        "a member of the class is archived at SESSION_ARCHIVE_AGE_DAYS, not the "
        "30-day background age — `_archive_age_for` keys the age on `platform`")

    s8 = _flat(_section("8. The history split"))
    assert "90-day conversation age" in s8, s8
    assert "stays in the Background listing for as long as its file is live" in s8, s8
    assert "leaves every `*.json` listing only when" in s8, s8


def test_section_13_names_the_conversation_age_and_the_platform_rule_behind_it():
    """Clause 2, with its reason: not merely a different number but the field the
    sweep reads. `reachable for a month` is the understatement by a factor of
    three, and it is what a reader would have acted on."""
    s13 = _flat(_section("13. Known limits"))
    assert "reachable for a month" not in s13
    assert "reachable for a" not in s13, "the understatement, in any wrap"
    assert "90-day" in s13, s13
    assert "conversation" in s13, s13
    assert "not the 30-day background age" in s13, (
        "the corrected limit must name the one it replaces, or the next reader "
        "re-derives the wrong rule from § Retention")
    assert "NON_USER_PLATFORMS" in s13, "the reason, not just the number"


def test_neither_the_section_11_parenthetical_nor_the_review_log_says_a_thirty_day_archive_expired_the_class():
    """Clause 3. §11 said archiving a member "ages it out after all" with no age
    attached, which read as the same 30 days; the review log is a record of what
    was filed on the day and is not rewritten, so a dated correction line beside
    the entry is what settles it."""
    s11 = _flat(_section("11. The Background tab"))
    assert "for 30 days" not in s11, s11
    assert "90-day" in s11, "the parenthetical has to carry the corrected age"

    log = _flat(_section("Review log"))
    entry = "never-aged-out exemption expiring at the sweep's 30-day archive"
    if entry in log:
        tail = log[log.index(entry):]
        assert "Correction" in tail, (
            "the log entry stands as filed only with a dated correction beside it")
        assert re.search(r"20\d\d-\d\d-\d\d", tail), "a correction needs its date"
    else:
        assert "30-day" not in log, "the entry was edited, so no correction is owed"


def test_section_8_reattributes_the_eight_archived_ids_to_ordinary_worker_runs():
    """Clause 4. The eight were classified from their FILENAME (`_autonomy_`) and
    their head has always said `worker`; a writer cannot rewrite a platform of an
    existing session, so they were never in the class and were archived on the
    machine age like every other background run."""
    s8 = _section("8. The history split")
    paras = [p for p in re.split(r"\n\s*\n", s8) if "never members" in p]
    assert len(paras) == 1, f"expected one reattribution paragraph, got {len(paras)}"
    para = _flat(paras[0])
    assert "20260909_1538xx_autonomy_" in para, "it must name the ids it is correcting"
    assert "platform: worker" in para, para
    assert "42" in para, "the census count the claim rests on"
    assert "machine age" in para, para
    assert re.search(r"20\d\d-\d\d-\d\d", para), "a census needs the date it was taken"


def test_one_dated_sentence_records_the_emptiness_and_the_missing_witness():
    """Clause 5. Twelve four-part `mission-control` sessions left the store with
    no archive behind them. The sweep never deletes a transcript and no writer
    rewrites an existing session's platform, so the in-tree candidates are the
    two routes that remove one (#2917: not the DELETE route alone, which is what
    this node's first version said). What the twelve have in common with those two
    routes is only that no writer records their departure; the removal routes are
    logged, at least as to path (#4114, and the three nodes above this one), so the
    absence is a missing witness about the clear route and the archive window, not a
    cause — and this page must not promote it into one."""
    s8 = _section("8. The history split")
    cand = [p for p in re.split(r"\n\s*\n", s8) if "no writer records" in p]
    assert len(cand) == 1, f"expected exactly one such sentence, got {len(cand)}"
    para = _flat(cand[0])
    assert "DELETE /api/sessions/{session_id}" in para, para
    assert "gzips and never deletes" in para, para
    assert "asserts no cause" in para, para
    assert re.search(r"20\d\d-\d\d-\d\d", para), "the emptiness is a measurement"
    for cause in ("were deleted", "the sweep deleted", "someone deleted",
                  "a DELETE call removed"):
        assert cause not in para, f"the page asserted a cause: {cause}"


# ── #2648: §12 must describe the bench family's life, not its old absence ─────

def _bench_section() -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.find("### The bench family has a life of its own now")
    assert start != -1, (
        "clause 5: §12 no longer has the section that describes what #2648 changed — "
        "the page went back to describing the absence the code no longer has")
    section = text[start:]
    nxt = section.find("\n## ", 5)
    return " ".join((section[:nxt] if nxt != -1 else section).split())


def test_the_page_states_the_bench_archive_age_and_the_default_hidden_listing():
    section = _bench_section()
    assert "BENCH_SESSION_ARCHIVE_AGE_DAYS = 7" in section, section[:200]
    assert "`?include_bench=true`" in section
    assert "bench_hidden" in section
    # The two names, because a page that says only `bench` describes half the class.
    assert "`bench`" in section and "`bench-mine`" in section


def test_the_stated_numbers_are_the_ones_the_code_ships():
    """A doc figure that no test re-reads is a figure that will rot (#2648 clause 5).

    The archive age and the withheld-count field are read out of the modules the page
    describes, so a change to either reddens this node instead of quietly leaving a
    sentence that names a constant nobody sets any more.
    """
    import sys
    import importlib.util

    sys.path.insert(0, str(ROOT))
    from app import sessions_io

    spec = importlib.util.spec_from_file_location(
        "rs_doc", ROOT / "scripts" / "groundskeeper" / "retention-sweep.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)

    section = _bench_section()
    assert f"BENCH_SESSION_ARCHIVE_AGE_DAYS = {rs.BENCH_SESSION_ARCHIVE_AGE_DAYS}" \
        in section
    assert f"{rs.BACKGROUND_SESSION_ARCHIVE_AGE_DAYS} days" in section
    assert set(sessions_io.BENCH_SOURCES) == {"bench", "bench-mine"}


def test_the_page_keeps_the_measured_grid_day_and_drops_the_wrong_share():
    """The row's original 2,544 / 24% counted 960 `.tool-results` dirs as sessions.

    The corrected figures are 1,590 + 424 live transcripts and one grid-day of 273;
    the page must keep a dated measurement rather than the inflated one it shipped with.
    """
    section = _bench_section()
    # The corrected figures live in the section #2648 wrote; the §12 table row that
    # still carries the 2,544 / 24% pair is not this clause's surface and is not
    # rewritten here (it landed at HEAD from the review that filed the item). What the
    # section must not do is repeat the inflated share as if it were a session count.
    assert "24% of `sessions/`" not in section, (
        "the 24% share counted `.tool-results` directories as sessions — #2648's triage")
    # The burst is stated as its two components, because they are two writers: 249
    # `bench` from the runner plus 24 `bench-mine` from the mining worker = 273
    # transcripts, against a listing 150 rows wide.
    assert "249 `bench`" in section and "24 `bench-mine`" in section, section[:400]
    assert "2026-10-07" in section, (
        "a burst-driven number needs the day it was measured on")


# ── #2917: §8's census sentence names ONE transcript-removing route ─────────
#
# The page says `DELETE /api/sessions/{session_id}` "is the only in-tree route
# that removes a transcript". The router contradicts it: `POST
# /api/sessions/clear` removes one too, and it is the *worse* witness — its
# handler is a bare `meta_path.unlink()` of the id in the request body, with no
# cancel, no drain, no spill dir and not one `logger` call beside it, where
# `delete_session` cancels the turn, drains both tiers and `rmtree`s the
# `<id>.tool-results` directory. A sweep of `unlink()` and `rmtree` over `app/`
# finds no third route that touches a session transcript, so the candidate set
# is exactly these two — which is why the sentence was understating its own
# candidates, not failing to explain the emptiness.
#
# The route set is read off the router's decorator lines and filtered on the
# handler unlinking a `SESSIONS_DIR` file, so what the page must name is what the
# tree ships. Path-anchored deliberately: `app/routers/sessions.py` declares two
# `@router.delete(` — the session's goal route comes first — and twelve
# `@router.post(`, so "the first decorator of the verb" names the wrong
# endpoint. No line number from that file is transcribed into this one; the two
# the doc cites are checked against the decorators, below.

ROUTER = ROOT / "app" / "routers" / "sessions.py"
TRANSCRIPT_ROUTE = "DELETE /api/sessions/{session_id}"
CLEAR_ROUTE = "POST /api/sessions/clear"
GOAL_ROUTE = "DELETE /api/sessions/{session_id}/goal"

_DECORATOR = re.compile(r'^@router\.(delete|post)\("([^"]+)"', re.M)


def _router_source() -> str:
    text = ROUTER.read_text(encoding="utf-8")
    assert len(text.splitlines()) > 100, f"{ROUTER} has stopped looking like the router"
    return text


def _declared_routes() -> dict[str, int]:
    """`{"VERB /path": decorator line}` for every delete/post route in the router.

    The line is derived from the match offset, which is the only way a doc
    citation to a route can be checked at all: nothing here holds a number
    somebody had to keep current by hand.
    """
    src = _router_source()
    out: dict[str, int] = {}
    for m in _DECORATOR.finditer(src):
        out[f"{m.group(1).upper()} {m.group(2)}"] = src.count("\n", 0, m.start()) + 1
    assert out, f"no @router.delete/@router.post decorator parsed out of {ROUTER}"
    return out


def _transcript_removing_routes() -> dict[str, int]:
    """The declared routes whose handler unlinks something under `SESSIONS_DIR`.

    "Removes a transcript" is a property of the handler, so it is read off the
    handler: the text between one decorator and the next either contains a
    `.unlink()` of a `SESSIONS_DIR` path or it does not. A third such route would
    land in here and demand a third name in the doc's sentence.
    """
    src = _router_source()
    out: dict[str, int] = {}
    for m in _DECORATOR.finditer(src):
        nxt = src.find("\n@router.", m.end())
        body = src[m.end(): len(src) if nxt == -1 else nxt]
        if ".unlink()" in body and "SESSIONS_DIR" in body:
            out[f"{m.group(1).upper()} {m.group(2)}"] = src.count("\n", 0, m.start()) + 1
    assert out, f"no route in {ROUTER} unlinks a session file — this section's premise moved"
    return out


def _census_paragraph() -> str:
    """§8's census paragraph, selected exactly as the #2651 node above selects it.

    Both files pick the paragraph as the only §8 paragraph carrying `"no writer
    records"`, so #2651's carried phrases and #2917's two-route sentence cannot
    drift into grading two adjacent blocks of prose.
    """
    s8 = _section("8. The history split")
    cand = [p for p in re.split(r"\n\s*\n", s8) if "no writer records" in p]
    assert len(cand) == 1, f"expected exactly one, got {len(cand)}"
    return _flat(cand[0])


def _sentences(flat: str) -> list[str]:
    """Sentences of an already-flattened block.

    A period ends a sentence only when the next word starts capitalised or with a
    backtick, because this page is full of `app/sessions_io.py:1363-1348` spans
    and `<id>.tool-results` names whose periods are not full stops.
    """
    return [s.strip() for s in re.split(r"(?<=\.) (?=[A-Z`])", flat) if s.strip()]


def test_the_census_sentence_names_both_routes_that_remove_a_transcript():
    """Clause 1. `only in-tree` was the exclusivity, and the router disproves it,
    so the phrase is gone page-wide and both endpoints appear by the exact path
    strings their own decorators declare."""
    routes = _transcript_removing_routes()
    assert set(routes) == {TRANSCRIPT_ROUTE, CLEAR_ROUTE}, (
        f"the router removes transcripts on {sorted(routes)}; §8's sentence names "
        "that set, so the page and this pin have to be re-read together")
    page = DOC.read_text(encoding="utf-8")
    assert "only in-tree" not in page, (
        "§8 called the DELETE route 'the only in-tree route that removes a "
        "transcript' — POST /api/sessions/clear is a second one, so the "
        "exclusivity is false rather than merely incomplete (#2917)")
    para = _census_paragraph()
    assert "in-tree" in para, (
        "positive control for the absence above: were 'in-tree' gone from the "
        "sentence as well, a zero-hit grep for 'only in-tree' would say nothing "
        "about this page")
    for route in sorted(routes):
        assert route in para, f"§8's census sentence never names {route}: {para[:160]}"


def test_the_census_sentence_says_what_separates_the_two_routes():
    """Clause 2. Naming two routes and leaving them interchangeable would be its
    own error, and the difference is the reason the candidate set mattered:
    `delete_session` cancels, drains both tiers and takes the spill dir, while
    the clear route leaves no trace beyond the missing file. It has to sit in the
    paragraph #2651 pins — a separation stated one paragraph away leaves the
    census sentence standing as the exclusivity claim this item exists to drop.
    """
    para = _census_paragraph()
    assert "cancels the current turn" in para, para[:160]
    assert "drains the queued turns in **both** tiers" in para, para[:160]
    assert "<id>.tool-results" in para and "spill dir" in para, para[:160]
    assert "bare `meta_path.unlink()`" in para, (
        "the clear route is a witness only because nothing surrounds its unlink: "
        + para[:160])
    for carried in ("no writer records", "gzips and never deletes", "asserts no cause"):
        assert carried in para, f"#2651's carried phrase {carried!r} left the paragraph"
    assert re.search(r"[Mm]easured 20\d\d-\d\d-\d\d", para), (
        "the emptiness is a measurement and keeps the date it was taken")


#: Where the rotation ceiling the census paragraph quotes comes from. Read rather than
#: transcribed for the reason #2917 filed this file: a page that still says one figure
#: after the conf moved is a false measurement with a date on it.
BACKEND_CONF = ROOT / "agent-services" / "supervisor" / "conf.d" / "lloyd-backend.conf"


def _stdout_rotation_ceiling() -> str:
    """The `stdout_logfile_maxbytes` value the backend's supervisor block sets."""
    lines = [ln for ln in BACKEND_CONF.read_text(encoding="utf-8").splitlines()
             if ln.startswith("stdout_logfile_maxbytes")]
    assert len(lines) == 1, (
        f"{BACKEND_CONF} sets stdout_logfile_maxbytes {len(lines)} times; the page "
        "quotes one value and this helper can only hand over one")
    return lines[0].split("=", 1)[1].strip()


def test_the_census_paragraph_names_the_access_line_both_removals_leave():
    """Clause 1, and the correction #4114 exists for. The sentence this node used to
    pin — "Neither leaves a per-session record a later reader could find" — was false
    for the DELETE route: uvicorn boots at `log_level="info"` with no `access_log` or
    `LOGGING_CONFIG` override in the tree, so its access logger records every request
    to either route, with method, path, peer and status, into the stdout file
    supervisor rotates. Saying otherwise is what generated the "no witness" items
    (#4110 proposed an in-handler `logger.warning` that would have landed in that same
    file and expired with it), so the paragraph now states the witness and this node
    forbids the sentence's return.
    """
    para = _census_paragraph()
    assert "Neither leaves a per-session record" not in para, (
        "the falsified sentence is back: an INFO access line names both removal "
        "routes, so a later reader does find a record for at least one of them")
    sents = _sentences(para)
    witness = [s for s in sents if "access" in s.lower()]
    assert witness, (
        "the paragraph no longer names any access log; it has gone back to the "
        f"unwitnessed claim: {para[:200]}")
    joined = " ".join(witness)
    assert "INFO" in joined, (
        "the record is an INFO access line and the level is what makes it live at "
        "`log_level=\"info\"`: " + joined[:220])
    for field in ("method", "path", "status"):
        assert field in joined, (
            f"the access line is described without its {field!r} field: " + joined[:220])
    assert "peer" in joined, (
        "the peer address is half of what makes the line evidence about a caller: "
        + joined[:220])
    assert "server.log" in joined, (
        "the witness has to name the file it lands in, or a reader cannot find it: "
        + joined[:220])


def test_the_census_paragraph_states_which_route_is_a_per_session_witness():
    """Clause 2. The two routes are not equally good records, and collapsing them into
    "both are logged" would be the same error as the sentence it replaces, mirrored:
    on the DELETE route the id is in the path, so the line identifies a transcript;
    on the clear route the id is in the body and the handler returns success for a
    request that removed nothing, so the line identifies neither.
    """
    para = _census_paragraph()
    sents = _sentences(para)
    path_side = [s for s in sents if "rides in the path" in s]
    body_side = [s for s in sents if "in the request body" in s]
    assert len(path_side) == 1 and len(body_side) == 1, (
        f"the asymmetry is one pair of sentences and the paragraph carries "
        f"{len(path_side)} path sides and {len(body_side)} body sides")
    assert "per-session" in path_side[0], (
        "the path case is the one that names a transcript, which is the whole "
        "difference: " + path_side[0][:200])
    assert "never the session" in body_side[0], (
        "the body case has to say the line does not identify the vanished "
        "transcript: " + body_side[0][:200])
    assert "success" in body_side[0], (
        "a 200 from the clear route is not evidence a transcript went, because the "
        "handler answers the same when the file was absent: " + body_side[0][:220])


def test_the_witness_is_bounded_by_rotation_and_dated_not_stated_as_durable():
    """Clause 3. The record exists, and it survives about eleven days, which is the
    actual reason the vanished twelve cannot be explained from it. Without the bound a
    reader draws the opposite conclusion from the same paragraph — that a checkable
    record of a removal is sitting there to be read — so the ceiling is pinned against
    the supervisor block that sets it and the span against the date it was measured.
    """
    para = _census_paragraph()
    ceiling = _stdout_rotation_ceiling()
    bound = [s for s in _sentences(para) if "rotation" in s or "backup" in s]
    assert bound, "the paragraph stopped bounding the witness by rotation"
    joined = " ".join(bound)
    assert ceiling in joined, (
        f"the page quotes a rotation ceiling other than the {ceiling!r} the backend's "
        "supervisor block sets: " + joined[:220])
    assert re.search(r"(eleven|11)\b", joined), (
        "the window is a count of files, not an adjective — the page says eleven "
        "(the live file plus ten backups): " + joined[:220])
    assert re.search(r"[Mm]easured 20\d\d-\d\d-\d\d", joined), (
        "the span is a box state that traffic volume moves, so it keeps its date and "
        "the owed-check re-derives it: " + joined[:220])
    assert "eleven days" in joined or re.search(r"~?\d+ days", joined), (
        "the bound has to be stated in days as well as bytes, since days is what a "
        "reader of a nine-day gap needs: " + joined[:220])
    assert "in-handler" in joined and "rotation" in joined, (
        "the paragraph has to say why the fix is prose: a new in-handler logger call "
        "expires in the same rotation")


def test_the_cause_ban_still_covers_the_rewritten_census_paragraph():
    """Clause 4. Everything #2651 and #2917 needed from this paragraph survives the
    reword: the page still asserts no cause, and the four cause phrasings stay out of
    all of §8 — checked over the section rather than the sentence, because a page with
    one incident has one place to say a thing about it. The witness paragraph adds a
    real mechanism ("an access line naming the route"), which is exactly the kind of
    sentence that quietly becomes a cause, so this is the rung that says it may not.
    """
    para = _census_paragraph()
    assert "asserts no cause" in para, para[:160]
    s8 = _flat(_section("8. The history split"))
    for cause in ("were deleted", "the sweep deleted", "someone deleted",
                  "a DELETE call removed"):
        assert cause not in s8, f"§8 asserted a cause: {cause!r}"
    assert "no writer records" in para and "gzips and never deletes" in para, (
        "#2651's carried phrases left with the reword: " + para[:200])


def test_neither_removal_handler_gained_a_logger_call_for_the_witness():
    """Clause 5. #4110's proposed remedy was a `logger.warning` in each handler; it
    would have landed in the same rotating file and bought no lifetime, which is why
    this round is prose. The router is read rather than trusted: it has exactly one
    `logger.` call — the spill-dir `rmtree` failure warning — and one `getLogger`
    binding, so the count of `logger.` call sites is the whole of what the handlers
    log. Both grep spellings are pinned with their numbers because they differ: a bare
    `logger.` matches the module-level `logger = logging.getLogger(...)` binding as
    well (2 hits), and only `logger.` with the dot escaped counts calls (1 hit), which
    is the number the page's account rests on.
    """
    src = _router_source()
    calls = re.findall(r"\blogger\.\w+\(", src)
    escaped = len(re.findall(r"logger\.", src))
    bare = len(re.findall(r"logger.", src))
    assert escaped == 1, (
        f"the router now makes {escaped} logger calls, not the one spill-dir warning §8 "
        "describes — either a handler logs removals now and the page under-states it, "
        "or it logs them and this round's reason for prose over code needs re-reading")
    assert bare == 2, (
        f"the router reads as {bare} bare `logger.` hits, not 2; the extra one is the "
        "module's own `logger = logging.getLogger(...)` binding, so the file's logging "
        "shape moved and both spellings above have to be re-read")
    assert calls == ["logger.warning("], (
        f"the one call is no longer the spill-dir warning alone: {calls}")


def test_exactly_one_dated_sentence_records_the_store_wide_name_date_gap():
    """Clause 4. The four ids minted after the 2026-09-09 sandbox batch are dated
    2026-09-16 and 2026-09-17 — #1064 counted all twelve while the files were
    still on disk — and the store holds nothing at all named in
    2026-09-13..2026-09-21. So an absent transcript in that window says nothing
    about this class. The sentence must name its basis: mtime in that store is the
    *archive* time, and a reader who re-derives by mtime gets a different answer.
    """
    s8 = _flat(_section("8. The history split"))
    hits = [s for s in _sentences(s8) if "2026-09-13" in s]
    assert len(hits) == 1, (
        f"§8 carries {len(hits)} sentences about the 2026-09-13 window; the item "
        "asked for exactly one dated sentence")
    sent = hits[0]
    assert "2026-09-21" in sent and "of any suffix" in sent, sent
    assert "2026-09-16" in sent and "2026-09-17" in sent, (
        "the sentence has to place the four post-2026-09-09 ids inside the gap, "
        "at the dates they were counted at: " + sent)
    assert "2026-09-09" in sent and "2026-09-12" in sent and "2026-09-22" in sent, (
        "the flanks are what make the gap store-wide rather than class-shaped: "
        + sent)
    assert "are present" in sent, sent
    assert "name" in sent and "mtime" in sent, (
        "the basis has to be named — an mtime re-derivation gives another number: "
        + sent)
    assert "not evidence about this class" in sent, sent
    assert re.search(r"[Mm]easured 20\d\d-\d\d-\d\d", sent), (
        "the gap is a measurement and needs the day it was taken: " + sent)
    for cause in ("were deleted", "the sweep deleted", "someone deleted",
                  "a DELETE call removed"):
        assert cause not in sent, f"the gap sentence asserted a cause: {cause!r}"


def test_the_route_pin_derives_from_the_router_and_not_from_a_transcribed_line():
    """Clause 5's guard on this file. It closes two ways for this section to stop
    meaning anything: a line number copied in by hand, which rots the first time
    anybody edits the router above it, and a derivation that took the first
    decorator of a verb, which in this router is the session's goal route rather
    than the transcript route.
    """
    src = Path(__file__).read_text(encoding="utf-8")
    assert re.search(r"sessions\.py:[0-9]", src) is None, (
        "a sessions.py line number was transcribed into this file — take it out of "
        "_transcript_removing_routes() instead of writing it down")
    declared = _declared_routes()
    assert GOAL_ROUTE in declared, (
        "the goal route is what proves this lookup is path-anchored: it shares the "
        "@router.delete( verb with the transcript route and is declared first, so "
        "a first-of-the-verb derivation names it. It has left the router and this "
        "node is no longer discriminating anything.")
    removing = _transcript_removing_routes()
    assert TRANSCRIPT_ROUTE in removing and GOAL_ROUTE not in removing, (
        f"the unlink-of-a-session-file filter selected {sorted(removing)}, which is "
        "no longer the transcript route and nothing else")


def test_each_route_citation_in_the_census_sentence_points_at_its_decorator():
    """#2917's fourth point, as a mechanism rather than a wish: the sentence cites
    each route at its `app/routers/sessions.py:<line>`, and each number is checked
    against the decorator the router declares there now. A citation outliving its
    route is the exact failure this file exists to catch, and here it reddens
    instead of sending a reader to somebody else's endpoint.
    """
    src_lines = _router_source().splitlines()
    para = _census_paragraph()
    for route in sorted(_transcript_removing_routes()):
        verb, path = route.split(" ", 1)
        cited = re.search(re.escape(f"`{route}`")
                          + r" \(`app/routers/sessions\.py:(\d+)`\)", para)
        assert cited, (
            f"{route} is not followed by its `app/routers/sessions.py:<line>` "
            f"citation: {para[:200]}")
        line = src_lines[int(cited.group(1)) - 1]
        assert line.startswith(f'@router.{verb.lower()}("{path}")'), (
            f"§8 cites {route} at line {cited.group(1)}, which reads {line!r}")


def test_the_name_date_gap_the_sentence_records_is_still_the_store():
    """The new sentence is a measurement of a live store, so this node takes the
    measurement rather than only the prose. The window is closed history — no
    writer stamps a transcript with a name nine days old — so a non-empty window
    means §8's sentence has been falsified, and the sentence is the thing to
    amend, not the count.

    The store is opened through `app.data_root.ACCOUNT_HOME`, the passwd-derived
    account home, never through `$HOME` or `$LLOYD_DATA`: `tests/conftest.py`
    points `LLOYD_DATA` at a scratch root for every run, and inside a gate
    `$HOME` is the round's symlink farm, which carries no `lloyd-data` by design.
    Same resolution, and same reason, as
    `tests/test_architecture_coverage_doc_claims.py`'s `~/lloyd-data/...` citations.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from app.data_root import ACCOUNT_HOME

    store = ACCOUNT_HOME / "lloyd-data" / "sessions"
    assert store.is_dir(), f"{store} is not on disk — there is no store to measure"
    names = [p.name for p in store.iterdir()]
    assert names, f"{store} reads as empty, which is not the store §8 measured"

    def named(pattern: str) -> list[str]:
        rx = re.compile(pattern)
        return sorted(n for n in names if rx.match(n))

    window = named(r"202609(1[3-9]|2[01])")
    assert window == [], (
        f"{len(window)} entries now carry a name-date inside "
        f"2026-09-13..2026-09-21 (first: {window[:3]}) — §8's discontinuity "
        "sentence is falsified and has to be amended, not this count")
    assert named(r"202609(09|1[012])"), (
        "the pre-window flank is empty as well, so the archives §8 says are "
        "present are no longer there")
    assert named(r"202609(2[2-9]|30)"), (
        "the post-window flank is empty as well, so the discontinuity §8 records "
        "is no longer bounded by anything")
