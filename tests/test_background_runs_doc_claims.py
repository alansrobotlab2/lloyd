"""Claims `architecture/background-runs.md` must not make: the chat summarizer
(#1228) and the mis-labelled class's archive age (#2651).

The #1228 paragraph said `_summarize_chat` ran on every `POST /api/mc/state`,
opened every file with no ceiling, and cost 238 ms at 1,420 files. The first two
were false and the third was a carried-over figure from a directory five months
younger. What it may say is pinned here as behaviour the doc must describe.

The #2651 nodes below pin the other correction: the page dated the Background
tab's never-aged-out exemption at 30 days in four places, and the sweep archives
that class at 90.
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
    rewrites an existing session's platform, so the DELETE route is the only
    in-tree candidate and nothing logs it — which is a missing witness, not a
    cause, and this page must not promote it into one."""
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
