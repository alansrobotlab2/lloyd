"""#1204 clause 5: the dashboard's board-size and latency comments are claims,
not decor.

Two comments in `app/routers/dashboard.py` described a board that no longer
exists. One said the backlog is "300+ markdown files" against a real 1,142; the
other said `board_health` costs "~2 s on the live board" against a measured
4–9 s. Both were true when written. A future session inherits them as facts —
that is the sentence in the item, and it is the reason the fix's diff has to
carry the comment change rather than a follow-up.

The literal strings are only the half that could be grepped for. The durable
half is the rule, applied to **comment blocks only**: a comment that asserts a
count of the board/ledger with a unit, or a duration, must also carry the
evidence for it — an ISO date it was taken at, the word `measured`, or a
backticked command at least three words long that re-measures it. Two filters
keep the rule on the board rather than over the whole file: docstrings are out
(the file's narrative prose, including `board_health`'s own re-measured claim
in `scripts/automod/backlog.py`, is graded where it lives), and the block must
name the backlog, the board or the ledger — which is why the dated size claims
at `:184`, `:200` and `:269`, all about the sessions directory, are not
flagged. A comment block that says the board is some size with no date and no
command is exactly the artifact this file exists to catch at write time.

The positive controls matter here more than anywhere: a `grep -c` that returns
0 because the pattern is malformed against the corpus is the false-zero the box
already has a catalogued habit of, so each assertion is paired with a count of
a string known to be present, and the file's line count is printed so an empty
file cannot pass as a clean one.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parent.parent / "app" / "routers" / "dashboard.py"
FRONTEND = Path(__file__).resolve().parent.parent / "web" / "src" / "components" / "pages" / "DashboardPage.tsx"

# The two stale claims by their exact wording, so the assertion cannot drift
# into "some comment that looks stale". Denominator at triage (2026-09-16):
# 1 hit each, at :56 and ~:672.
STALE = ("300+", "~2 s on the live board")

# A claim about *quantity of the thing the dashboard walks*, stated with a
# unit; or any stated duration. `~?\d+(\.\d+)? (s|ms|seconds)` for the second
# half. The first half allows up to two modifier words between the number and
# its unit, which is not generosity — it is the file's own idiom: the comment
# this item was filed about reads "the backlog is 300+ markdown files", and a
# variant phrasing the review rung named explicitly, "the board is 300+ item
# files", has a bare singular noun (`item`) standing between the count and the
# plural unit. A unit alternation glued to `\s+` matched neither, so a
# reintroduced claim in exactly that phrasing passed the structural rule
# unflagged. `test_the_quantity_rule_fires_on_…` now pins all three spellings.
QUANTITY = re.compile(
    r"\d[\d,]*\+?\s+(?:[a-z]+\.?[a-z]*\s+){0,2}"
    r"(markdown|files|items|rows|front matters|bytes)", re.I)
DURATION = re.compile(r"~?\s*\d[\d,]*(?:\.\d+)?\s*(s|ms|seconds)\b", re.I)
SUBJECT = re.compile(r"\b(backlog|board|ledger)\b", re.I)

# What makes a stated number auditable later. Two things count: a date, which
# decays visibly, or a backticked span with two spaces in it, i.e. a command and
# not an identifier — bare `` `board_health` `` and `` `up_next` `` are names,
# and an earlier draft let the ~2 s comment through on them.
#
# The word "measured" by itself used to be on this list, and the review rung
# removed the excuse (`SM_20260917_064456`: "EVIDENCE accepts a bare 'measured'
# without a date or command"). It was the weakest possible thing on the list:
# "measured on the live board" is the exact sentence this item exists to kill —
# it is a claim of provenance with nothing behind it, and it is exactly what a
# future session writes when it inherits a number it did not take. A number now
# needs a date, or the command that re-takes it. `
# test_the_quantity_rule_fires_on_the_board_claims_it_was_filed_for` pins that
# bare "measured" no longer satisfies the rule, alongside the flag/no-flag cases.
EVIDENCE = re.compile(
    r"(20\d\d-\d\d-\d\d|`[^`\n]*\s[^`\n]*\s[^`\n]*`)", re.I)


def _comment_blocks(text: str) -> list[tuple[int, str]]:
    """Contiguous `#` comment blocks, as (first line number, joined text)."""
    out, block = [], []
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            block.append((lineno, line))
        else:
            if block:
                out.append((block[0][0], " ".join(l for _, l in block)))
                block = []
    if block:
        out.append((block[0][0], " ".join(l for _, l in block)))
    return out


@pytest.fixture
def source() -> str:
    assert DASHBOARD.is_file(), f"{DASHBOARD} is not a file"
    return DASHBOARD.read_text(encoding="utf-8")


def test_the_two_stale_board_claims_are_gone(source: str):
    """Clause 5, literal: `grep -n "300+\\|~2 s on the live board"` → 0 hits."""
    lines = source.splitlines()
    print(f"{DASHBOARD} lines: {len(lines)}")
    assert len(lines) > 100, (
        f"{DASHBOARD} is {len(lines)} lines; a truncated file has no stale "
        f"comments either, and must not be read as a fixed one"
    )
    # Positive control: a string that must survive, so a 0 for the stale
    # patterns cannot be a read-the-wrong-file artifact.
    assert "_VAULT_SCAN_TTL_S" in source and "_SCORECARD_TTL_S" in source, (
        "the TTL constants are absent, so this is not the dashboard router "
        "the stale comments lived in"
    )

    offenders = []
    for needle in STALE:
        hits = [n for n, line in enumerate(lines, 1) if needle in line]
        print(f"{needle!r}: {len(hits)} hit(s)")
        if hits:
            offenders.append(f"{needle!r} still at line(s) {hits}")
    assert not offenders, (
        "the stale board claims are still in the file: " + "; ".join(offenders) +
        ". Rule already in force on this box: a 'currently ~N seconds' comment "
        "is a measurement with a decay date — state the board size it was "
        "taken at, or name the command that re-measures it."
    )


def test_every_board_or_ledger_quantity_claim_in_a_comment_carries_evidence(source: str):
    """The structural half: no comment may assert a board size or a latency bare.

    A comment block that mentions the backlog/board/ledger *and* states a
    counted quantity or a duration must also state when it was measured, or how
    to measure it. Triage denominator: 2 blocks flagged, both the two stale
    ones above; the replacement comments carry `pytest tests/
    test_dashboard_cold_render.py` and a dated count, so this file has a
    positive control of its own — it is not merely the negation of one grep.
    """
    blocks = _comment_blocks(source)
    print(f"comment blocks: {len(blocks)}")
    assert blocks, f"{DASHBOARD} has no comment blocks — nothing to grade"

    bare = []
    flagged = 0
    for lineno, text in blocks:
        if not SUBJECT.search(text):
            continue
        if not (QUANTITY.search(text) or DURATION.search(text)):
            continue
        flagged += 1
        if not EVIDENCE.search(text):
            bare.append(f":{lineno}: {text[:110]}")
    print(f"quantity/latency comment blocks about the board: {flagged}")
    assert not bare, (
        f"{len(bare)} comment block(s) assert a board/ledger size or a latency "
        f"with no date and no re-measuring command:\n" + "\n".join(bare) +
        "\nEither name the size the figure was taken at with its date, or name "
        "the command that re-measures it."
    )
    # And the rule must actually be exercised: at least one board-shaped
    # quantity claim exists in the file and passes.
    assert flagged >= 1, (
        "no comment block in the file mentions a board size or a latency, so "
        "the rule above asserted over an empty set. A denominator of zero is "
        "not a passing check."
    )


def test_the_quantity_rule_fires_on_the_board_claims_it_was_filed_for():
    """The rule must catch the phrasings this item was filed about.

    Advisory finding on round `SM_20260917_055508`: "QUANTITY's unit alternation
    cannot match the file's own idiom — 'the board is 300+ item files' does not
    match because 'items' needs the plural and the words are reversed — so a
    reintroduced bare board-size claim in that phrasing passes the structural
    rule unflagged". A structural rule that never fires on the artifact that
    motivated it is a rule asserted over an empty set, and
    `test_every_board_or_ledger_quantity_claim_in_a_comment_carries_evidence`
    cannot see that: it counts what it flags in the real file, so a rule that
    matches nothing fails on `flagged >= 1` at worst and reports nothing at
    better. These are synthetic comment blocks, run through the same
    `_comment_blocks` → SUBJECT → QUANTITY/DURATION → EVIDENCE pipeline, so a
    regression in the pattern goes red here and not on the next board claim.

    Advisory finding on the following round, `SM_20260917_064456`: "EVIDENCE
    accepts a bare 'measured' without a date or command". `_BARE_MEASURED` below
    is that case pinned — a latency claim whose only evidence is the WORD
    `measured`, which must NOT be accepted as evidenced. It is the weakest thing
    the old pattern accepted and the exact sentence this item exists to delete.

    So: five blocks must flag (a claim with no date and no re-measuring command),
    three must not (the same claim with a date, with a command, and one with no
    number at all). A rule that flags everything passes for free and would have
    this file deleted, so the negatives are as load-bearing as the positives.
    """
    must_flag = {
        # The literal comment at app/routers/dashboard.py:56 that #1204 was
        # filed to remove, as it was written across two comment lines.
        "original 300+ markdown files": (
            "# Sections that walk the vault are cached: the backlog is 300+ markdown\n"
            "# files and its status counts do not change between 2-second polls."),
        # The reviewer's counter-example verbatim: singular `item`, unit after it.
        "300+ item files": (
            "# Cached because the board is 300+ item files and the counts do not move."),
        # The other half of clause 5: a stated latency with no evidence.
        "original ~2 s": (
            "# item and the ledger (~2 s on the live board), so it sits on the\n"
            "# scorecard's minute rather than the vault scan's ten seconds."),
        # A plural-unit spelling with the number detached by a comma-group.
        "1,500 front matters": (
            "# The ledger is cheap here; the board is 1,500 front matters."),
        # Round SM_20260917_064456's advisory: the word `measured` was itself on
        # the EVIDENCE list, so a claim whose only backing is that word passed.
        # This is the sentence #1204 exists to delete — a claim of provenance
        # with nothing under it.
        "bare 'measured', no date, no command": (
            "# the ledger is ~2 s on the live board, measured on the current\n"
            "# tree, because the board is 1,142 item files."),
    }
    flagged = {}
    for label, block in must_flag.items():
        blocks = _comment_blocks(block)
        assert len(blocks) == 1, f"{label}: block splitter returned {len(blocks)}"
        text = blocks[0][1]
        assert SUBJECT.search(text), f"{label}: no subject match — {text[:80]}"
        quantity = bool(QUANTITY.search(text))
        duration = bool(DURATION.search(text))
        assert quantity or duration, (
            f"{label}: neither QUANTITY nor DURATION matched the very claim this "
            f"rule exists to catch — {text[:90]!r}. The pattern is the bug, not the "
            f"comment.")
        assert not EVIDENCE.search(text), (
            f"{label}: the block carries evidence it should not, so the negative "
            f"control below proves nothing about this one")
        flagged[label] = True

    # The negatives. Three separate ones, because one sample holding BOTH a date
    # and a command cannot tell you which half of EVIDENCE is load-bearing — and
    # it was the command half that stayed and the bare-word half that the review
    # rung cut.
    negatives = {
        # Date only, no command: the shape the live file's own comments use at
        # :32 and :69 after this item's fix.
        "date only": (
            "# Cached because the board is 1,500 item files, measured 2026-09-17."),
        # Command only, no date: decays by being re-run rather than by the clock.
        "command only": (
            "# the board is 1,142 item files; re-measure with\n"
            "# `pytest tests/test_dashboard_cold_render.py -q -s`."),
    }
    for label, text in negatives.items():
        block = _comment_blocks(text)[0][1]
        assert QUANTITY.search(block) or DURATION.search(block), (
            f"negative {label!r} is no longer a quantity/latency claim, so it "
            f"proves nothing about acceptance")
        assert EVIDENCE.search(block), (
            f"negative {label!r}: the rule no longer accepts a {label}, so it "
            f"would flag every comment the fix wrote — a rule that flags "
            f"everything passes for free")

    # And a block that is about the board but states no number: out of scope, and
    # must not flag. Without this the negatives only prove that evidenced claims
    # pass, never that the scope gate has a floor.
    out_of_scope = _comment_blocks(
        "# Sections that walk the board are cached; see the scorecard TTL\n"
        "# above for why the poll interval is what it is.")[0][1]
    assert SUBJECT.search(out_of_scope), "the out-of-scope sample lost its subject"
    assert not (QUANTITY.search(out_of_scope) or DURATION.search(out_of_scope)), (
        "the out-of-scope sample now carries a quantity or duration, so the "
        "negative below tests the wrong thing")


def test_the_ttl_comments_name_both_numbers_they_reconcile(source: str):
    """Recommendation E: the TTL against the poll rate has to be a decision.

    #1204 section 5 is that `_VAULT_SCAN_TTL_S = 10.0` against the frontend's
    2-second poll is the reason a hitch fires every fifth request. The fix is
    not to pick a magic number, it is to write the pair down; this test makes
    the pair unreadable-by-accident by requiring both constants' values in the
    comment above each TTL assignment.
    """
    assert FRONTEND.is_file(), f"{FRONTEND} is not a file"
    fe = FRONTEND.read_text(encoding="utf-8")
    poll = re.search(r"POLL_MS\s*=\s*(\d+)", fe)
    assert poll, "POLL_MS is gone from DashboardPage.tsx, so the pairing has no anchor"
    poll_s = int(poll.group(1)) / 1000
    print(f"frontend POLL_MS: {poll.group(1)} ({poll_s:g} s)")

    lines = source.splitlines()
    for const in ("_VAULT_SCAN_TTL_S", "_SCORECARD_TTL_S"):
        m = re.search(rf"^{const}\s*=\s*([\d.]+)", source, re.M)
        assert m, f"{const} is no longer assigned in {DASHBOARD}"
        ttl = float(m.group(1))
        # The comment block immediately above the assignment.
        assign_line = source[:m.start()].count("\n") + 1
        block = []
        n = assign_line - 1
        while n >= 1 and lines[n - 1].lstrip().startswith("#"):
            block.insert(0, lines[n - 1])
            n -= 1
        joined = " ".join(block)
        print(f"{const} = {ttl}: {len(block)} comment lines above it")
        assert joined, f"{const} has no comment above it naming the pairing"
        assert str(int(ttl)) in joined or str(ttl) in joined, (
            f"the comment above {const} = {ttl} does not state its own value")
        assert re.search(rf"\b{poll_s:g}[\s-]?(s|second)", joined, re.I) or f"{poll.group(1)}" in joined, (
            f"the comment above {const} = {ttl} does not name the frontend poll "
            f"interval ({poll.group(1)} ms = {poll_s:g} s in DashboardPage.tsx). "
            f"The 10 s TTL against a 2 s poll is what made a hitch fire every "
            f"fifth request; a TTL that outlives the poll by an unstated "
            f"multiple is the accident #1204 section 5 describes.")



# ── #1273: the section-error rule, and one definition of the section list ──
#
# Two facts the dashboard depended on and nothing checked.
#
# First, message derivation. `sectionOk` answers "no" for a section that is
# **absent** as well as one that came back failed, so the else branch of every
# `sectionOk` ternary has to render `sectionError(section)` — which accepts
# `undefined` — and never `section.error`, which dereferences it inside render.
# An exception there is caught by the ErrorBoundary in `web/src/App.tsx`, which
# replaces the *whole* dashboard: the page must not be the second thing to
# break.
#
# Second, the section list lived twice — as the `_gather(...)` call in
# `app/routers/dashboard.py` and as the prose table in CLAUDE.md — and the copy
# had already drifted by exactly one section (`automod`, added by `f6e25ac` on
# 2026-09-10, listed nowhere for nine days). The equality below is what makes
# "eleven of them" a number a run re-measures instead of a sentence it trusts.
#
# Every zero-count assertion carries a negative control, because a regex that
# matches nothing against the corpus looks identical to a clean tree — the
# false zero this box has a catalogued habit of.

ROOT = Path(__file__).resolve().parent.parent
CLAUDE = ROOT / "CLAUDE.md"
ARCH_DOC = ROOT / "architecture" / "mission-control.md"
WEB = ROOT / "web"
API_SPEC = WEB / "src" / "api.test.ts"

# The bug's shape and the fix's shape, as separate patterns so one regex can
# never be satisfied by both.
ERRORPANEL_RAW = re.compile(r"error=\{([A-Za-z_$][\w$]*)\.error\}")
# Deliberately whitespace-tolerant inside the tag: `</* reformat */>`-shaped
# churn must not trip a rule about *derivation*, and the fix itself wraps one
# of these panels across lines.
ERRORPANEL_CALL = re.compile(
    r'<ErrorPanel\s+what="([^"]*)"\s+error=\{\s*sectionError\(\s*([A-Za-z_$][\w$]*)\s*\)\s*\}',
    re.S)
GATHER_SECTION = re.compile(r"_gather\(\s*\"([^\"]+)\"")
# The source cell is captured loosely so an *empty* one parses as a row and
# then fails the assertion below; a `\S` here would skip such a row silently.
TABLE_ROW = re.compile(r"^\|\s*`([a-z_]+)`\s*\|\s*(.*?)\s*\|\s*$", re.M)

# Spelled out rather than read back from either file: the point of the
# equality is that table, endpoint and stated count are three claims, and a
# constant copied from one of them would assert nothing.
DASHBOARD_SECTIONS = {
    "host", "vllm", "primary", "recent", "agents", "services",
    "workers", "autonomy", "backlog", "automod", "usage",
    # #628: the egress destination inventory, twelfth of the twelve.
    "network",
}

# The three sentences #1273 falsifies, as `architecture/mission-control.md`
# carried them. Matched by their claim, not their full wording, so a reworded
# regression is caught; the second element is the former text, quoted so the
# failure names the sentence rather than a keyword.
STALE_ARCH_CLAIMS = {
    "the rule is not held yet": "**The rule is not held yet:** nine `ErrorPanel` sites",
    "is the only one done right": "The newest section, `automod`, is the only one done right.",
    "still does not list": "still does not list it",
}

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15}


@pytest.fixture
def frontend_source() -> str:
    assert FRONTEND.is_file(), f"{FRONTEND} is not a file"
    return FRONTEND.read_text(encoding="utf-8")


@pytest.fixture
def arch_text() -> str:
    assert ARCH_DOC.is_file(), f"{ARCH_DOC} is not a file"
    return ARCH_DOC.read_text(encoding="utf-8")


def _panels(source: str) -> list[tuple[str, str]]:
    """`(what, section)` for every rendered `ErrorPanel` site.

    Split rather than one global scan, so a site that has *no*
    `error={sectionError(…)}` at all — the case that matters — is reported as
    a missing panel instead of silently not matching.
    """
    found = ERRORPANEL_CALL.findall(source)
    sites = source.count("<ErrorPanel")
    assert len(found) == sites, (
        f"{sites} `<ErrorPanel` sites in DashboardPage.tsx but only {len(found)} "
        "render through `sectionError(...)`: a panel outside the pattern has an "
        "error message this rule no longer covers. `function ErrorPanel` at the "
        "top of the file is a definition and is not counted here.")
    return found


def test_the_errorpanel_patterns_fire_on_the_shapes_they_name():
    """Negative control for both ErrorPanel patterns, before either is trusted.

    The assertions that follow count matches and assert a zero; either is
    satisfied by a pattern that has quietly stopped matching, which is how a
    check becomes decoration. Both literals are byte-exact copies of what the
    file contained before and after #1273.
    """
    raw = ERRORPANEL_RAW.search('<ErrorPanel what="Host metrics" error={host.error} />')
    assert raw and raw.group(1) == "host", (
        "ERRORPANEL_RAW no longer matches `error={host.error}`, so the "
        "zero-count assertion below would pass on any file")
    oneline = ERRORPANEL_CALL.search(
        '<ErrorPanel what="Host metrics" error={sectionError(host)} />')
    assert oneline and oneline.groups() == ("Host metrics", "host"), (
        "ERRORPANEL_CALL no longer matches the fixed form, so the panel-set "
        "assertion below would fail on the fix itself")
    wrapped = ERRORPANEL_CALL.search('<ErrorPanel\n  what="Host metrics"\n'
                                     '  error={\n    sectionError(host)\n  }\n/>')
    assert wrapped and wrapped.groups() == ("Host metrics", "host"), (
        "ERRORPANEL_CALL does not survive a reformatted tag, so it is pinning "
        "formatting rather than derivation")
    assert not ERRORPANEL_CALL.search('<ErrorPanel what="Host metrics" error={host.error} />'), (
        "ERRORPANEL_CALL matches the raw form too, so it cannot tell the fix "
        "from the bug and the set assertion proves nothing")


def test_every_dashboard_errorpanel_goes_through_sectionerror(frontend_source: str):
    """Clause 1: no panel reads `section.error` raw, and every section has one.

    Nine sites read `x.error` at triage (2026-09-19) — 1275, 1340, 1374, 1426,
    1489, 1504, 1507, 1510, 1526 — each the else branch of a `sectionOk`
    ternary, which is precisely the branch an *absent* section takes.
    """
    raw = sorted(ERRORPANEL_RAW.findall(frontend_source))
    assert not raw, (
        f"ErrorPanel sites read a section's `.error` raw again: {raw}. "
        "`sectionOk` starts with `!!section`, so a section missing from the "
        "payload lands in this branch and `section.error` throws inside "
        "render — the ErrorBoundary then blanks the whole dashboard. Use "
        "`error={sectionError(section)}`.")

    panels = _panels(frontend_source)
    print(f"DashboardPage.tsx: {len(panels)} ErrorPanel sites, "
          f"{len(frontend_source.splitlines())} lines")
    rendered = {section for _, section in panels}
    assert rendered == DASHBOARD_SECTIONS, (
        f"panels render {sorted(rendered)}, expected the "
        f"{len(DASHBOARD_SECTIONS)} payload sections "
        f"{sorted(DASHBOARD_SECTIONS)} — a section with no error row is a "
        "section whose failure is invisible")


def test_the_automod_panel_is_not_gated_behind_an_undefined_check(frontend_source: str):
    """Clause 2: an absent `automod` shows a row, it does not vanish.

    The wrapper `{automod !== undefined && (…)}` was the newest section's own
    fix and it fails the same way, just quietly: an absent section is *hidden*
    rather than reported, so a backend that stops emitting one loses the panel
    that would have said so. `sectionError` already answers for `undefined`, so
    the plain `sectionOk` ternary is enough.

    Scoped to the payload sections on purpose. A `!== undefined` guard over an
    ordinary prop is a normal thing to write in a `.tsx`, and flagging one
    under a rule about dashboard sections would point the failure at the wrong
    file.
    """
    guarded = {name for name in
               re.findall(r"([A-Za-z_$][\w$]*)\s*!==\s*undefined\s*&&", frontend_source)}
    offending = sorted(guarded & DASHBOARD_SECTIONS)
    print(f"section-typed `!== undefined` guards: {offending}; "
          f"other `!== undefined` guards: {sorted(guarded - DASHBOARD_SECTIONS)}")
    assert not offending, (
        f"`{offending[0]} !== undefined &&` is back, so that section renders "
        "nothing when the payload lacks it — a missing section has to be a "
        "visible amber row, not a vanished panel")

    # Whitespace-tolerant and anchored on the section name rather than on a
    # whole literal block, so re-indentation is not a clause-2 failure while a
    # different `what` label or a different error source still is.
    flat = re.sub(r"\s+", "", frontend_source)
    ternary = ('sectionOk(automod)?<AutomodPanelautomod={automod}/>'
               ':<ErrorPanelwhat="Automodscorecard"error={sectionError(automod)}/>')
    assert ternary in flat, (
        "automod no longer renders through the same sectionOk ternary as "
        'its siblings with the `what="Automod scorecard"` error row, so an '
        "absent section has nothing to degrade to")


def test_the_table_row_pattern_reads_a_sourceless_row_as_a_failure():
    """Negative control for `TABLE_ROW`, so the source-cell assertion can fail.

    The row regex has to parse a row whose second cell is empty — otherwise an
    empty cell simply produces no row, the `named == DASHBOARD_SECTIONS` set
    complains about a *missing* section, and the real defect (a table that has
    become a name roster) goes unreported. It also has to ignore the `|---|---|`
    separator, which a looser row pattern would count as a section named `---`.
    """
    sample = ("| Section | Source |\n|---|---|\n"
              "| `automod` | `app/routers/dashboard.py::_automod` |\n"
              "| `host` |  |\n")
    rows = TABLE_ROW.findall(sample)
    assert ("automod", "`app/routers/dashboard.py::_automod`") in rows, (
        "TABLE_ROW does not read a normal row, so the table assertion below "
        "is asserting against nothing")
    assert ("host", "") in rows, (
        f"TABLE_ROW skips a row with an empty source cell ({rows}); the "
        "empty-cell assertion can never fire and the table could degrade into "
        "a name roster unnoticed")
    assert not any(name == "---" for name, _ in rows), (
        "TABLE_ROW reads the markdown separator as a section, so the set "
        "equality below could never match")


def test_the_section_table_and_the_endpoint_name_the_same_sections(source: str):
    """Clause 4: one definition of the section list, checked across two files.

    The table in CLAUDE.md is the prose a session reads first, and it had been
    missing `automod` since the section shipped. Requiring a non-empty source
    cell is what keeps the fix from becoming a name roster: a row that says
    only `| automod |` re-opens the drift it closes.
    """
    assert CLAUDE.is_file(), f"{CLAUDE} is not a file"
    claude = CLAUDE.read_text(encoding="utf-8")
    heading = re.search(r"^## Mission Control dashboard\b.*?(?=^## )",
                        claude, re.M | re.S)
    assert heading, ("CLAUDE.md has no `## Mission Control dashboard` section, "
                     "so the table this test exists to pin has no subject")
    rows = TABLE_ROW.findall(heading.group(0))
    print(f"CLAUDE.md dashboard section: {len(heading.group(0).splitlines())} "
          f"lines, {len(rows)} table rows")
    assert rows, "no table rows parsed — the regex is empty against the corpus"
    named = {name for name, _ in rows}
    assert named == DASHBOARD_SECTIONS, (
        f"CLAUDE.md's table lists {sorted(named)}, the endpoint's section list "
        f"is {sorted(DASHBOARD_SECTIONS)}; the two are one fact stated twice")
    for name, origin in rows:
        assert origin, (
            f"the `{name}` row has no source cell, so the table is a name list "
            "and not a map — the drift this assertion exists to catch starts "
            "with a row that names a section and says nothing about it")
    automod_rows = [origin for name, origin in rows if name == "automod"]
    assert automod_rows and "_automod" in automod_rows[0] \
        and "dashboard.py" in automod_rows[0], (
        f"the `automod` row does not name its source "
        f"(`app/routers/dashboard.py::_automod`): {automod_rows}")

    gathered = set(GATHER_SECTION.findall(source))
    assert gathered == DASHBOARD_SECTIONS, (
        f"`_gather(...)` names {sorted(gathered)}; either a section was added "
        "and CLAUDE.md's table is stale again, or one was dropped and this "
        "test's constant needs the same commit that changes the endpoint")


def test_the_architecture_doc_no_longer_reports_the_rule_as_unheld(arch_text: str):
    """Clause 5: the doc may not assert a gap the fix closed.

    It carried three: "**The rule is not held yet:** nine `ErrorPanel` sites",
    "`automod` is the only one done right", and the table "still does not list"
    `automod`. A reader who trusts any of them stops looking — a stale copy
    outlives the fix precisely because it reads as a current state report.
    """
    flat = re.sub(r"\s+", " ", arch_text).lower()
    for claim, former in STALE_ARCH_CLAIMS.items():
        assert claim not in flat, (
            f"architecture/mission-control.md still states {claim!r} "
            f"(it read: {former!r}). #1273 makes it false; a doc that reports a "
            "closed gap as open is the stale copy this item is about.")
    # And the section still states the rule those sentences were apologising
    # for, so deleting the bullet cannot pass as satisfying the clause.
    assert "sectionerror(section)" in flat, (
        "the doc no longer tells a writer to render sectionError(section) — "
        "removing the stale status sentence is not the same as removing the rule")
    assert "sectionok(section)" in flat, "sectionOk is no longer named as the test"


def test_the_architecture_doc_stated_count_is_the_real_one(arch_text: str, source: str):
    """"eleven of them" has to be a measurement, not a human's running total.

    Honest about what it is: eleven was already the true count when this was
    written, so this assertion does not discriminate the fix — it is the guard
    for the *next* section, which is the failure the merged half of the item
    describes ("the next section added to `_gather` makes both the count and
    the table wrong at once").
    """
    match = re.search(r"[—-]\s*(\w+)\s+of them\b", re.sub(r"\s+", " ", arch_text))
    assert match, (
        "architecture/mission-control.md no longer states how many sections "
        "the endpoint has, so the constant below has nothing to check against")
    word = match.group(1).lower()
    assert word in NUMBER_WORDS, f"unparsable section count {word!r} in the doc"
    stated = NUMBER_WORDS[word]
    gathered = set(GATHER_SECTION.findall(source))
    print(f"arch doc states {stated} sections; `_gather` names {len(gathered)}")
    assert stated == len(gathered), (
        f"architecture/mission-control.md says {stated} sections, "
        f"`_gather(...)` names {len(gathered)}: {sorted(gathered)}")


def test_the_section_error_helpers_have_a_spec_the_suite_actually_runs():
    """Clause 3, reachable from the runner the gate and CI use.

    `web/src/api.test.ts` is the spec that pins `sectionOk`/`sectionError`, and
    the gate's `frontend` rung does run it (`gate._vitest_run`, after symlinking
    the live `node_modules` into the worktree). It does not say so in a node id
    the pytest runner can name, so this node runs the same binary over the same
    file and reports vitest's own test names — the behaviour stays pinned by a
    node the suite itself executes, and the two runners cannot disagree about
    what `sectionError(undefined)` returns.

    It skips when the toolchain is absent rather than failing, for the reason
    `gate._vitest_run` gives: `web/node_modules` is untracked, and a checkout
    where nobody has run `npm install` since vitest landed must not turn a
    missing binary into a red suite. The skip is loud and names the command.
    """
    assert API_SPEC.is_file(), (
        f"{API_SPEC} is missing — the clause-3 spec is the whole point of this "
        "node, so its absence is a failure and not a skip")
    spec = API_SPEC.read_text(encoding="utf-8")
    for claim in ('sectionOk(undefined)', 'sectionError(undefined)',
                  'sectionError({ error: "boom" })'):
        assert claim in spec, (
            f"web/src/api.test.ts no longer exercises {claim}, so the clause-3 "
            "behaviour has no spec for this node to run")

    vitest = WEB / "node_modules" / ".bin" / "vitest"
    if not vitest.exists():
        pytest.skip(f"{vitest} is not installed — run `npm install` in ~/lloyd/web, "
                    "then `cd ~/lloyd/web && npx vitest run src/api.test.ts`. "
                    "The gate's frontend rung runs it whenever it can, and this "
                    "node runs it whenever the symlink that rung makes is there.")

    from app.lint_findings import node_env  # the same env the gate's rung builds
    proc = subprocess.run(
        [str(vitest), "run", "src/api.test.ts", "--reporter=json", "--silent"],
        cwd=str(WEB), env=node_env(), capture_output=True, text=True, timeout=300)
    # `--reporter=json` writes the report to stdout and nothing else does, so
    # the payload is one object; the find/rfind pair only guards against a
    # toolchain warning landing in front of it.
    payload, start, end = proc.stdout, proc.stdout.find("{"), proc.stdout.rfind("}")
    assert start >= 0 and end > start, (
        f"vitest produced no JSON report (rc={proc.returncode}): "
        f"{(proc.stdout + proc.stderr)[-600:]}")
    report = json.loads(payload[start:end + 1])
    results = [t for file in report.get("testResults", []) for t in file.get("assertionResults", [])]
    names = sorted(t["title"] for t in results)
    print(f"vitest src/api.test.ts: {len(names)} tests — {names}")
    assert results, "vitest reported zero tests, so the green below would be vacuous"
    failed = sorted(t["title"] for t in results if t.get("status") not in ("passed", "pending"))
    assert proc.returncode == 0 and not failed, (
        f"vitest failed sectionError/sectionOk spec: {failed or proc.returncode}")
    # The spec must still name the behaviours clause 3 states, so a spec
    # rewritten to assert nothing cannot pass by being emptied. One case per
    # branch of the pair: what a *missing* section reports, and what a *failed*
    # one reports.
    joined = " ".join(names).lower()
    for behaviour in ("missing", "failed"):
        assert behaviour in joined, (
            f"no api.test.ts case covers {behaviour!r} any more: the spec that "
            "this node runs has stopped pinning clause 3")
