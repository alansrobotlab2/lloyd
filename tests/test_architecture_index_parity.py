"""`architecture/index.md` is a hand-written list over `architecture/*.md`, and
nothing made the two agree (#1103).

`workers/sources/arch_review.py::doc_slugs` builds the review picklist by
globbing the directory; `index.md` is prose over the same directory, read as
"every doc there is". The listing was 22/22 clean by luck when the item was
filed, and it drifted exactly as predicted: `desktop.md` landed on 2026-09-23
and was in no table of the index the next day. Same shape as
`tests/test_mc_tab_parity.py` — a fifth hand-maintained list, pinned the same
way, in both directions with a readable diff.

The second half is the counts. `index.md` once restated "32 scheduled jobs"
and "a 9-rung gate" in a summary table, and both rotted inside 24 hours of
the doc's own `date:`; each has an owner (the autonomy board, the gate's
ladder list) that `autonomy-jobs.md` refuses to copy for the same reason. So
no Covers cell may count a set its owner enumerates — and the guard counts the
*noun*, not the spelling: #2700 found it had matched a bare integer over two
nouns for its whole life, which is how "the five due-gates" and "the four
lists" stayed green while their owners had already grown past both. Digits and
one-through-twelve therefore count now, over the nouns these tables use.

What is exempt is decided by what the check is fed: the `| [[doc]] | Covers |`
rows, and nothing else. That leaves the three places a number belongs on this
page alone — the GPU allocation table, whose row count is pinned by
`tests/test_infrastructure_gpu_inventory.py` and whose prose says on purpose
how many cards the box has; the Review log, which quotes the counts a past
correction retired and may not be rewritten to satisfy a guard; and the
retired-name list under "Retired" (a fixed historical set). A tilde'd order of
magnitude — a suite size, a rate, anything with a margin rather than an
enumerable owner — is a measurement and not a count of a set, so the noun list
never names it either.

The third half is multiplicity. Both directions above read the index into a `set`
of stems, so a doc listed once and a doc listed twice are the same green — which is
how "The agent" table came to carry `[[desktop]]` in two rows, at lines 48 and 59 of
that revision, with two different and equally true Covers cells until `8ac21903`
(2026-10-14) deleted the second. So one row per doc is asserted over the rows themselves, counted over
table rows only: `index.md` also names `[[desktop]]` in the Review-log entry that
records that deletion, and a whole-file count would report the survivor as the
duplicate and invite a "fix" that rewrites the log.

The fourth half is a figure that travels. #3249 found the index quoting a suite integer
this page had already retired while claiming, in the present tense, that the doc which
owns the measurement measured it — a figure the owner had long since re-measured, and a
second copy of one measurement kept in prose nothing re-measures. So a present-tense
attribution of a tilde'd or comma-grouped figure to another architecture doc is refused
across `architecture/*.md` and this file's own docstrings, and a bare suite size may sit
in a live body only in the doc that owns it plus a closed register of dated growth
records. `## Review log` history stays standing, and both guards are shown red by
planting the retired prose into the shipped pages themselves — never by being handed a
sentence typed by the same hand that wrote the guard.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "architecture"
INDEX = ARCH / "index.md"
#: This module's own docstring, captured so a node can hold it to what it promises.
#: It is the only prose in this file that states what the count guard exempts, and
#: prose that states an exemption by naming one example rots the way the example did.
MODULE_DOC = __doc__ or ""

_LINK = re.compile(r"\[\[([a-z0-9-]+)\]\]")
#: How a count is written when someone types it from memory: digits, or the word.
#: #2700 is the second half of this rule and the half that was missing: the guard
#: matched `\d+` alone, so "the five due-gates" sat in a Covers cell for a month
#: while its owner (`_is_task_due`) had grown past it, and `architecture/index.md`
#: advertised "the four lists" for the tabs while `test_mc_tab_parity.py` opens by
#: naming six. A spelled-out number is the same second definition of a set, and
#: rotting is not sensitive to how it was spelled.
_NUMBER = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
#: The sets these tables actually advertise, each of which an owner enumerates:
#: `_is_task_due`'s refusals, `SEAMS` in `app/djev_shadow.py`, the djev tool
#: registry, its two-tier trust flags, `context_window`'s layers, the gate's
#: floors, `test_mc_tab_parity.py`'s lists, the research producers, the token
#: meter's readers, `memory_*`'s tools. Cards belong here too: `nvidia-smi -L`
#: enumerated five long before `index.md`'s own Review log caught up (#2697), and
#: the inventory that owns the number is
#: `tests/test_infrastructure_gpu_inventory.py`, not this page.
_OWNED_NOUNS = (r"scheduled\s+jobs?|rungs?|due-?gates?|shadow\s+seams?|tools?|"
                r"trust\s+flags?|layers?|floors?|lists?|producers?|readers?|"
                r"verbs?|cards?")
#: A hand-typed count of a set that has an owner elsewhere in the tree.
_OWNED_COUNT = re.compile(rf"\b{_NUMBER}\s*-?\s*(?:{_OWNED_NOUNS})\b", re.I)
#: A link-table row, split into the doc and the Covers cell that follows it. Only
#: these two cells are the index's claim about what a doc covers; the GPU table's
#: first cell is a digit and is not a `[[link]]`, so it never enters.
_COVERS_ROW = re.compile(
    r"^[ \t]*\|[ \t]*\[\[([a-z0-9-]+)\]\][ \t]*\|([^\n]*)$", re.M)
#: A `[[slug]]` heading a table row — the first cell, and nothing else in it.
#: `[ \t]` rather than `\s` so a match cannot cross a line boundary: multiplicity
#: below is a property of rows, and `_LINK` over the whole text would also count
#: the Review-log prose that records a duplicate being deleted.
_ROW_LINK = re.compile(r"^[ \t]*\|[ \t]*\[\[([a-z0-9-]+)\]\][ \t]*\|", re.M)


def _docs_on_disk() -> set[str]:
    """Same walk as `arch_review.doc_slugs`: top level only, `.archive/` never."""
    return {p.stem for p in ARCH.glob("*.md") if p.stem != "index"}


def _docs_linked() -> set[str]:
    return set(_LINK.findall(INDEX.read_text(encoding="utf-8")))


def _row_first_cells(text: str) -> list[str]:
    """Every `[[slug]]` that heads a table row, in file order, repeats included.

    A list and not a set, deliberately: `_docs_linked()` is a set, so a doc listed
    once and a doc listed twice are the same green to every node above it. That is
    how `index.md` came to carry `[[desktop]]` in two rows of "The agent" table, at
    lines 48 and 59 of that revision, with two different Covers cells, both true, so
    no reader could tell which one describes the doc. `8ac21903` (2026-10-14) deleted
    the second of them;
    the repeats-included list is what makes that shape checkable at all.
    """
    return _ROW_LINK.findall(text)


def _covers_cells(text: str) -> list[tuple[str, str]]:
    """Every `(doc, Covers cell)` the index's link tables advertise, in file order.

    Rows, not the file. The index says two different kinds of sentence: a table row
    states what a doc covers *now*, and the Review log below it records what somebody
    believed on a date — which is why `index.md` may still quote the counts a past
    correction retired there (the list this file checks verbatim, in `EXEMPT_COUNTS`)
    while a live Covers cell may not count a thing (#2329 pinned the log against being
    rewritten by a correction, and
    `test_the_row_scope_is_what_keeps_the_review_log_from_reading_as_a_duplicate` is
    this file's other half of the same rule). The GPU-allocation prose names its own
    count for the same reason and points at its owner in the next sentence. Scanning
    the whole text would fire on all of it, so the count check is fed rows.
    """
    return [(doc, cell) for doc, cell in _COVERS_ROW.findall(text)]


def _owned_count_offenders(text: str) -> list[str]:
    """Each `[[doc]]: count` whose Covers cell restates a set its owner enumerates.

    Names the doc, not just the count: the reader's job is to open that one row and
    reword it, and a bare list of numbers would not say which of thirty-five rows to
    open. Empty means the tables describe their sets; a match list is only trustworthy
    because `test_the_parity_check_can_see_a_count_in_a_covers_row` shows the pair
    extractor finds rows at all.
    """
    return [f"[[{doc}]]: {m.group(0)}"
            for doc, cell in _covers_cells(text)
            for m in _OWNED_COUNT.finditer(cell)]


def _section(text: str, heading: str) -> str:
    """The body of one `## ` section of the index, up to the next heading.

    A reader with a boundary, not "everything below this line": a node that wants to
    say a phrase sits in the GPU-allocation prose has to say where that section ends,
    or the same claim passes when the phrase has moved past it.
    """
    start = text.index(heading)
    rest = text[start + len(heading):]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _duplicate_row_cells(text: str) -> list[str]:
    """Each stem heading more than one table row, once each, sorted.

    Empty means one row per doc; it names the repeated stem, never just a count,
    because the reader's job is to go merge two Covers cells and needs to know
    which doc to open. Scoped to rows so the Review-log sentence that records a
    duplicate having been deleted does not read as a second listing of it.
    """
    counts = Counter(_row_first_cells(text))
    return sorted(stem for stem, seen in counts.items() if seen > 1)


#: The section that records a retirement. Its heading is matched by prefix so the
#: parenthetical about where the files live can be reworded without blinding this.
_RETIRED_HEADING = "## Retired"
#: How that section records one: a name in backticks. A `[[slug]]` is a promise
#: that `architecture/<slug>.md` is behind it, so the retired list may not use it.
_BACKTICKED = re.compile(r"`([a-z0-9][a-z0-9-]*)`")


def _retired_section(text: str) -> str:
    """The Retired section's body, up to the next `##` heading.

    Asserts the section is there: a green must not come from a scan that found
    nowhere to look.
    """
    start = text.find(_RETIRED_HEADING)
    assert start >= 0, "index.md has no Retired section to record a retirement in"
    rest = text[start:]
    end = rest.find("\n## ", len(_RETIRED_HEADING))
    return rest if end < 0 else rest[:end]


def _retirement_problems(text: str) -> list[str]:
    """Every way `index.md` breaks the retirement convention, as messages.

    A retired doc is *named* under Retired and linked from no table, so the two
    forms must stay disjoint: a retired name that is also a `[[link]]` is the
    stale promise that went red in #1704, and a retired name whose doc is back on
    disk belongs in a table. The list itself is asserted non-empty — a retired
    section emptied out would make both checks pass on nothing.
    """
    retired = set(_BACKTICKED.findall(_retired_section(text)))
    assert retired, (
        "the Retired section names no doc at all, so there is no convention "
        "here to check — the list cannot be empty and the check be meaningful")
    linked, on_disk = _docs_linked(), _docs_on_disk()
    return (
        [f"{name}: retired under 'Retired' and still linked as [[{name}]]"
         for name in sorted(retired & linked)]
        + [f"{name}: retired under 'Retired' but architecture/{name}.md is on disk"
           for name in sorted(retired & on_disk)])


def test_every_architecture_doc_is_linked_from_the_index():
    unlisted = _docs_on_disk() - _docs_linked()
    assert unlisted == set(), (
        f"architecture/*.md not linked as [[slug]] from index.md: {sorted(unlisted)} — "
        "add a row to the right table; a doc the index does not list is one a "
        "reader does not find")


def test_every_index_link_names_a_doc_on_disk():
    dangling = _docs_linked() - _docs_on_disk()
    assert dangling == set(), (
        f"index.md links [[slug]]s with no architecture/<slug>.md behind them: "
        f"{sorted(dangling)} — a retired doc goes under 'Retired', not in a table")


def test_no_doc_is_listed_in_more_than_one_table_row():
    """One doc, one row: the set-based checks above cannot see a doc listed twice.

    `index.md` had `[[desktop]]` heading two rows of "The agent" table with two
    different Covers cells — "the lease only a human grants" and "a lease a human
    grants from the Desktop tab" — both true, so which cell describes the doc was
    unanswerable. It arose by re-adding the row instead of editing it, which is
    exactly the drift a membership test cannot see: the node above asks only whether
    the stem is in the link set at all.
    """
    dupes = _duplicate_row_cells(INDEX.read_text(encoding="utf-8"))
    assert dupes == [], (
        f"architecture docs heading more than one table row in index.md: {dupes} — "
        "fold the two Covers cells into one row; a doc with two rows has no single "
        "description, and a reader cannot tell which row is stale")


def test_the_row_scope_is_what_keeps_the_review_log_from_reading_as_a_duplicate():
    """The count is over rows, and that scoping is load-bearing on today's index.

    `[[desktop]]` is named twice in `index.md`: once as the first cell of its row in
    "The agent" table, once in the Review log recording that the duplicate row had
    been deleted. A count over the file would report the survivor as a duplicate and
    invite someone to "fix" it by rewriting the Review-log entry — which
    `test_the_retired_section_says_git_history_is_the_only_copy_left` exists to
    forbid. So the log stays untouched, and this node is the proof that the row
    anchor, not the prose, is what makes the check above green.
    """
    text = INDEX.read_text(encoding="utf-8")
    cells = _row_first_cells(text)
    named, rows = _LINK.findall(text).count("desktop"), cells.count("desktop")
    assert rows == 1, (
        f"[[desktop]] heads {rows} table row(s) of index.md, expected exactly 1 — "
        + ("two rows is the shape this file exists to catch" if rows > 1 else
           "the desktop row has left the tables, so this node no longer measures "
           "what its name claims and belongs with the retired-name checks"))
    # What a whole-file count would have seen beside it: the same stem named more
    # often than it heads a row, i.e. prose the row anchor correctly excludes.
    assert named > rows, (
        f"index.md names [[desktop]] {named} time(s) and rows head it {rows} time(s): "
        "with no Review-log mention beside the row there is nothing for the row "
        "anchor to exclude, so this node's scoping is asserted but never exercised")


def test_the_index_states_no_count_of_a_set_with_an_owner():
    """No Covers cell counts a set an owner enumerates — spelled any way.

    The rule is in `architecture/autonomy-jobs.md` §Scheduled jobs: a copy of a
    count here is wrong within a week, because the board and the code move and this
    page does not. What #2700 adds is the *spelling*: the guard read as "a bare
    integer", so it passed over "the five due-gates" and "the four lists" while the
    owners had grown past both. A Covers cell describes a set — "its due-gates", "the
    lists that must agree on them" — and leaves the count to the doc that enumerates
    it.
    """
    offenders = _owned_count_offenders(INDEX.read_text(encoding="utf-8"))
    assert offenders == [], (
        f"index.md restates a count that an owner already enumerates: {offenders} — "
        "describe the set, do not count it (autonomy-jobs.md §Scheduled jobs "
        "says why: a copy here would be wrong within a week)")


def test_the_parity_check_can_see_a_missing_doc(tmp_path, monkeypatch):
    """Negative control: a 31st doc dropped into the directory with no index row
    must be the exact set difference, so an empty result above means parity and
    not a scanner that matched nothing."""
    import sys
    fake_arch = tmp_path / "architecture"
    fake_arch.mkdir()
    (fake_arch / "index.md").write_text("| [[alpha]] | a |\n", encoding="utf-8")
    (fake_arch / "alpha.md").write_text("# alpha\n", encoding="utf-8")
    (fake_arch / "beta.md").write_text("# beta\n", encoding="utf-8")
    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, "ARCH", fake_arch)
    monkeypatch.setattr(mod, "INDEX", fake_arch / "index.md")
    assert _docs_on_disk() - _docs_linked() == {"beta"}
    assert _docs_linked() - _docs_on_disk() == set()
    assert _OWNED_COUNT.findall("each of the 32 scheduled jobs; the 9-rung gate; 9 rungs") == [
        "32 scheduled jobs", "9-rung", "9 rungs"]
    # A tilde'd order of magnitude is a measurement with a margin, not a count of a
    # set with an owner, so the noun list never names it whatever the figure is.
    assert _OWNED_COUNT.findall("the ~9,600-test suite; 3 rows; 2026-09-11") == []


def test_the_mission_control_row_names_the_tab_lists_without_counting_them():
    """The row that started this, pinned against both ways of fixing it.

    `index.md` advertised [[mission-control]] as covering "the tabs and the four
    lists that must agree on them". Six lists do: `test_mc_tab_parity.py` opens by
    naming six, the sixth (`Layout.tsx`'s `PAGES`) arrived in `2a58a8e5` (#1274)
    long after the row was written, and `architecture/mission-control.md` says so in
    its own words. The digits-only guard could not see the stale word, which is
    #2700. Two ways out, and the row must stay out of both: re-typing the count as
    "six" just installs the next rot, and deleting the clause loses what the doc is
    for. So the cell names the set and no number beside it — while the count itself
    stays where it has an owner, in the doc and the test that enumerate the lists.
    """
    text = INDEX.read_text(encoding="utf-8")
    cells = dict(_covers_cells(text))
    assert "mission-control" in cells, "[[mission-control]] no longer heads a table row"
    cell = cells["mission-control"]
    assert "lists" in cell, (
        f"the row stopped naming the tab-list set at all: {cell!r} — the guard is "
        "there to stop counting it, not to make it unmentionable")
    typed = re.findall(rf"\b{_NUMBER}\b", cell)
    assert typed == [], (
        f"[[mission-control]]'s Covers cell types a count ({typed}): {cell!r} — "
        "six is correct today and wrong the next time a tab lands, which is the "
        "reason `test_mc_tab_parity.py` enumerates the lists instead")
    assert not re.search(r"\b(?:four|six)\s+lists\b", text, re.I), (
        "index.md still counts the tab lists somewhere — the stale row is what this "
        "#2700 review found, and a grep for 'four lists|six lists' is how a reader "
        "checks it went")


#: Every retired count the shipped page still quotes outside its link tables.
#: Each is one the widened pattern matches when it is handed the string, so the
#: green on the real index can only come from the row scoping — and each is
#: wording a correction may not rewrite (`test_the_index_states_no_count_of_a_set_with_an_owner`
#: going green by deleting them is the failure mode this list exists to catch).
#: Each phrase paired with the heading of the ONE section that may hold it, so the
#: node below asserts a location rather than "somewhere below a heading".
EXEMPT_COUNTS = (
    ("ten rungs", "## Review log"),             # 2026-09-14: the ladder grew again
    ("five cards", "## Review log"),            # 2026-10-14: the GPU table's history
    ("three shadow seams", "## Review log"),    # 2026-10-14: superseded by four
    ("the five due-gates", "## Review log"),    # 2026-10-14: the #2700 finding itself
    ("Five cards, not three", "## GPU allocation"),  # names its owner next sentence
)


def test_the_exempt_regions_still_carry_their_counts_and_stay_outside_the_guard():
    """The shipped page's exemptions are real, not a scan that found nothing.

    Widening `_OWNED_COUNT` and scoping it to rows in one commit has a failure mode
    the green above cannot see on its own: if the scoping were wrong, or someone
    deleted the quoted history to get to green, the page would pass and the record of
    why these rules exist would be gone. So this node reads the shipped bytes and
    asserts all three halves at once — the counts are still on the page verbatim, the
    widened pattern still fires on each of them in isolation (so the page is not
    green because the guard got narrower), and the guard as wired reports none of
    them, because none of them sits in a Covers cell.
    """
    text = INDEX.read_text(encoding="utf-8")
    for phrase, heading in EXEMPT_COUNTS:
        section = _section(text, heading)
        assert phrase in section, (
            f"{phrase!r} is not inside {heading!r} of index.md: that count is either "
            "the Review log's record of a retired number (which a correction may not "
            "rewrite) or the GPU prose that names its owner, and both are exempt by "
            "scoping, not by deletion")
        assert _OWNED_COUNT.findall(phrase), (
            f"the widened pattern no longer matches {phrase!r} at all, so the page "
            "passing is the guard going blind again rather than the scoping working")
    assert _owned_count_offenders(text) == [], (
        "a retired count moved into a Covers cell, where it is a live claim and not "
        "history")


#: The suite-size figure this module's docstring used to name as its exemption
#: example. Written as a concatenation so the node below can forbid the literal
#: anywhere in this file without this line containing it.
ROTTED_FIGURE = "8," + "700"


def test_the_module_states_its_exemption_as_a_class_and_not_as_a_figure():
    """The exemption this file promises is a rule, not a number that will age.

    Its docstring used to name one suite-size figure as the exemption's whole meaning,
    and that number went stale inside the same window #2700 was open in. Dated now,
    because the sentence used to say it in the present tense (#3249): on 2026-09-24 the
    `tests` rung collected 9,583–9,620 across three sampled rounds and
    `architecture/testing.md` opened at the same order of magnitude; by the 2026-10-24
    rounds that same rung collected over twice that. A present-tense reading of the old
    sentence would have concluded the doc was wrong while the doc was right — the
    identical failure the guard exists to prevent, committed in the guard's own prose,
    and it went unnoticed for the same reason the rotted Covers cells
    did: nothing read the sentence. So the exemption is now stated as a class (a tilde'd
    order of magnitude is a measurement with a margin, not a count of an owned set), and
    this node keeps it that way in both directions:

    1. the rotted figure may not come back, in the docstring or anywhere in this file;
    2. the class wording must stay, because without it the next reader has only an
       example to generalise from, which is how the sentence got a number in it;
    3. whatever the current figure is, the guard must still refuse to match it — the
       negative control in `test_the_parity_check_can_see_a_count_in_a_covers_row`
       carries the measurement, and this asserts the class, not a specific suite size.
       That citation used to name a node that does not exist, which is what
       `test_every_node_this_file_cites_in_prose_is_a_node_this_file_defines` now stops.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    assert ROTTED_FIGURE not in MODULE_DOC, (
        "the module docstring names the stale suite-size figure again — it exempts a "
        "class of measurement, not a number this file would then have to re-measure")
    assert ROTTED_FIGURE not in source, (
        f"the figure {ROTTED_FIGURE!r} is back somewhere in this file; the suite size "
        "this module is measured against lives in `architecture/testing.md`, not here")
    flat = " ".join(MODULE_DOC.split())  # the docstring wraps; the phrase may not sit on one line
    assert "tilde" in flat and "order of magnitude" in flat, (
        "the docstring no longer states WHY a suite size is exempt, so the next "
        "correction has nothing to re-derive the rule from and will paste a number in")
    assert _OWNED_COUNT.findall("the ~9,600-test suite") == [], (
        "a tilde'd suite size now trips the count guard, which means the exemption is "
        "no longer a class but whatever figure happens to be current")


#: ── #3249: a figure that lives in one doc may be *owned* there and restated nowhere ──
#:
#: The count guard above stops `index.md` re-typing a set its owner enumerates. This is
#: the other half of the same rot, which that guard cannot see because a suite size is
#: deliberately exempt from it: prose in one file asserting, **in the present tense**,
#: what a *different* architecture doc measures. Three of those existed at once —
#: `index.md`'s Review log said `([[testing]] measures ~9,600)` twenty-four hours after
#: the correcting pass had moved the doc to over twice that; this module's own docstring
#: said `both measure ~9,600 now`; and `testing.md`'s live body asserted it was `still
#: what architecture/index.md quotes this doc for`, which was true only until the index
#: stopped quoting it. Nothing read the sentences, so all three rotted silently. A
#: dated-past clause is allowed to quote a dead figure — that is history, and the Review
#: log is made of it — so the discriminator is an *undated present-tense* attribution.

#: How prose points at another architecture doc: the wiki stem the link tables use, or
#: the backticked path. Both spellings occur in the sentences this guard had to catch.
_XDOC_REF = r"(?:\[\[[a-z0-9-]+\]\]|`architecture/[a-z0-9-]+\.md`)"
#: Present-tense attribution verbs, and only those. A past form ("measured", "restated",
#: "advertised") puts the figure in the tense the Review log lives in, and a spelled-out
#: count belongs to the owned-set guard above, not here. The trailing `\s` is what makes
#: this a verb list: the index's own Review-log sentence — `…names no count), and`
#: followed two tokens later by the retired suite-size integer — sits one character away
#: from tripping it on the **noun** "count", which is how a guard meant for attributions
#: would fire on half the Review log.
_ATTRIB_VERB = (r"(?:measures?|says|quotes?|counts?|reports?|gives?|carries?|"
                r"advertises?|states?)\s")
#: A figure written with a margin or a thousands separator — the shape a suite size, a
#: rate or a token count takes. Spelled numbers are out on purpose: `[[djev]] says four`
#: is the #2700 case, which has its own guard.
_FIGURE = r"(?:~\s?\d[\d,]*|\d{1,3},\d{3,})"
#: A gap inside one sentence: prose wraps at ~80 columns, so newlines are allowed and a
#: sentence-ending period is not (a period before a letter is an abbreviation).
_GAP = r"(?:[^.]|\.(?=[a-z0-9])){0,%d}"
_XDOC_FIGURE = re.compile(
    rf"(?:{_XDOC_REF}{_GAP % 80}{_ATTRIB_VERB}{_GAP % 40}{_FIGURE}"
    rf"|{_FIGURE}{_GAP % 60}{_XDOC_REF}{_GAP % 40}{_ATTRIB_VERB})",
    re.I)
#: An explicit day stamp turns a claim into a measurement of its date, which is what
#: makes a quoted figure survivable instead of rotten.
_DATE_STAMP = re.compile(r"\b20\d\d-\d\d-\d\d\b")
#: A sentence end: a period with whitespace or end-of-text after it. A period before a
#: letter is an abbreviation or a version number and does not close a sentence — the same
#: rule `_GAP` follows inside a match.
_SENT_BREAK = re.compile(r"\.(?=\s|\Z)")


def _sentences(text: str) -> list[str]:
    """`text` split into sentences, so a node can reason one sentence at a time.

    The unit is the point: #3249's fix rewrote a sentence to stop attributing a figure to
    another doc, and an `== []` over a whole file cannot tell that achievement apart from
    a page that simply stopped naming the owner. Split per sentence and a node can demand
    both halves of the fix — the pointer is still there, and no sentence holding it carries
    a number.
    """
    out, head = [], 0
    for brk in _SENT_BREAK.finditer(text):
        out.append(text[head:brk.end()])
        head = brk.end()
    if head < len(text):
        out.append(text[head:])
    return out


def _live_body(text: str) -> str:
    """Everything above a page's `## Review log`: the prose a reader takes as current.

    One definition for both guards that need it — the cross-doc figure scan and the
    suite-size placement check — because if the two disagree about where "live" ends, one
    of them is exempting text the other indicts.
    """
    return text.split("## Review log")[0]


def _xdoc_figure_claims(text: str) -> list[str]:
    """Present-tense cross-doc figure claims in `text`, dated clauses excluded.

    The date window is ±60 characters around the match, so a clause can carry its stamp on
    either side. Refusing to exempt a *dated* clause is what would make this guard fire on
    every Review-log entry, and those are the record #3249's own clause forbids rewriting.

    The window is characters, not sentences, and that was measured rather than assumed:
    scoping the carve-out to the claim's own sentence looked stricter and reddened a dated
    Review-log entry in `testing.md` — the very record #3249 is told to leave standing —
    because log prose wraps mid-sentence, and a wrapped bullet is not a sentence. What the
    narrower scope would buy is covered from the other side instead: the corpus node
    asserts that no LIVE sentence of `index.md` carries a figure at all, so a stamp in that
    page can shield nothing there, and the `far` control below pushes the date past sixty
    characters and requires the guard to speak.
    """
    claims = []
    for match in _XDOC_FIGURE.finditer(text):
        window = text[max(0, match.start() - 60):match.end() + 60]
        if _DATE_STAMP.search(window):
            continue
        claims.append(" ".join(match.group(0).split()))
    return claims


#: The one doc that measures the suite: `scripts/automod/gate.py`'s `tests` rung collects
#: it, and `testing.md` is where that rung's numbers are recorded. No other doc owns a
#: re-measurement, which is why no other doc may state the size as a present fact.
SUITE_SIZE_OWNER = "testing.md"
#: Suite sizes that appear in some other doc's live body, and are allowed to, because each
#: is the record of a suite **once** having that size — `automod.md`'s gate-latency history
#: of the suite growing. A measurement of the past cannot move, which is what makes naming
#: these by *value* safe where naming their line numbers would not be: the numbers below
#: stay true no matter how far the corpus grows downward, and an unregistered figure is
#: red whether it is a new copy of the present count or a fourth growth record, in which
#: case it joins the list with its own history sentence beside it.
SUITE_SIZE_RECORDS = {
    "automod.md": ["~5,830", "3,271", "5,900"],
}
#: A suite size planted by the controls: not a figure any page holds, so a report of it can
#: only have come from the plant, and the registered records beside it stay untouched.
PLANTED_FIGURE = "~12,345"
#: A suite size: the figure, immediately followed by "test"/"tests" (optionally
#: hyphenated into it, as in the retired integer the index hand-typed and then quoted).
#: Anchored on the noun rather than on the word "suite" so a count of something else
#: sitting beside that word — "529 lines in test_testing_md", "26 test files" — is not
#: read as a restatement of the suite's size.
_SUITE_FIGURE = re.compile(rf"({_FIGURE})(?=\s*-?\s*tests?\b|\s*-?\s*test\b)", re.I)


def _live_suite_figures(doc) -> list[str]:
    """Suite sizes in `doc`'s live body — everything above its `## Review log`.

    The Review log is exempt as the dated record of retired numbers: the 2026-10-14 entry
    is *about* a hand-typed count that rotted, and #3249 is about the cross-doc clause in
    that same sentence, not about the log keeping it.
    """
    text = doc.read_text(encoding="utf-8").split("## Review log")[0]
    return _SUITE_FIGURE.findall(text)


def _suite_placement_offenders(arch_dir: Path) -> dict[str, tuple[list[str], list[str]]]:
    """Per doc under `arch_dir`: (figures with no registration, figures found at all).

    Factored out of the corpus loop so the planted-figure control can run the WALK instead
    of a copy of its idea of a walk: a node asserting over a corpus and a control asserting
    against a hand-built list are two different guards, and only the first is watching the
    shipped pages. Both readings come back because the loop needs both — an unregistered
    figure is rot, and a registered value that has stopped being found means the register
    is now describing a sentence nobody wrote. `SUITE_SIZE_OWNER` is skipped for one reason
    only: it holds the live count, which is what makes it the sole home by clause 5.
    """
    seen = {}
    for doc in sorted(arch_dir.glob("*.md")):
        if doc.name == SUITE_SIZE_OWNER:
            continue
        found = _live_suite_figures(doc)
        seen[doc.name] = (_unregistered(doc.name, found), found)
    return seen


def _unregistered(name: str, figures: list[str]) -> list[str]:
    """Which of `figures` `name` has no registration for. Split out so the register can be
    handed a planted figure and shown to be red: an `== []` over a corpus is the same green
    as a pattern that matches nothing, which is the failure this file has been bitten by
    twice already (#2700's digits-only pattern, #2701's set-based extractors)."""
    return [f for f in figures if f not in SUITE_SIZE_RECORDS.get(name, [])]


def _module_docstrings() -> str:
    """Every docstring in this file: the module's and each test node's.

    The fixtures deliberately hand the count guard a stale-looking suite size, so the
    scan is of docstrings — the prose a reader trusts — and not of the source.
    """
    import sys

    parts = [MODULE_DOC]
    for name, obj in sorted(vars(sys.modules[__name__]).items()):
        if name.startswith("test_") and callable(obj) and obj.__doc__:
            parts.append(obj.__doc__)
    return "\n".join(parts)


#: The pre-fix sentence, and the one site in the shipped page it can come back to. Both
#: are planted into copies of the REAL `index.md` below rather than handed to the guard on
#: their own: a string typed out beside the pattern proves the pattern matches a string,
#: not that the page is clean, and the page is what #3249 is about.
PRE_FIX_INDEX_CLAUSE = "[[testing]] measures ~9,600"
#: The clause #3249 wrote in its place, located by its own words rather than by a line
#: number, so re-wrapping the page cannot hide the site from either control below. Both
#: read it as a literal substring of the shipped bytes, and both read the WHOLE page: the
#: clause sits under `## Review log`, which is exactly why the cross-doc scan is whole-file
#: while the suite-size placement check stops at that heading.
INDEX_CLAUSE_HERE = "The page that measured"


#: The three sentences as they stood before #3249, one per surface. The index one is
#: assembled without the retired integer because this file forbids that literal.
PRE_FIX_CLAIMS = (
    PRE_FIX_INDEX_CLAUSE,
    "`architecture/testing.md` line 11 and the gate's own collected count both "
    "measure ~9,600 now",
    "was ~9,600 (still what `architecture/index.md` quotes this doc for)",
)
#: `index.md`'s neighbouring sentence: an attribution with a doc reference and a number
#: in it that is not a tilde'd or comma-grouped figure. The guard must stay quiet here,
#: or it fires on half the Review log.
CLEAN_ADJACENT_CLAIM = "with this page's own [[djev]] row, which says GPU 2 is djev's"
#: The date carve-out's controls: the same claim with a stamp inside its own sentence,
#: without one at all, and with a stamp one sentence away. All three directions are
#: asserted: a carve-out with no control is a hole with a docstring, and the guard's
#: silence must be attributable to the date and not to the pattern having stopped matching.
DATED_SILENT_CLAIM = "[[testing]] measures ~9,600 as of 2026-09-24"
UNDATED_FIRING_CLAIM = "[[testing]] measures ~9,600"


def test_no_architecture_doc_restates_another_docs_figure_in_the_present_tense():
    """No file may assert, undated and in the present tense, what another doc measures.

    #3249 found three such sentences live at once. Two of them named `[[testing]]`'s
    suite size from somewhere else, and the third was `architecture/testing.md` itself
    claiming what the index quoted it for — a claim that dies the moment the index is
    fixed, which is how a correction written in one file becomes a falsehood in the
    file next to it. What each of them should have said instead is the rule: the
    document that measures a figure owns the number, and everywhere else points at the
    owner or dates the quote. Both corpus halves are scanned — every
    `architecture/*.md`, and this module's own docstrings, whose last sentence is the
    one that rotted first.
    """
    offenders = []
    for doc in sorted(ARCH.glob("*.md")):
        for claim in _xdoc_figure_claims(doc.read_text(encoding="utf-8")):
            offenders.append(f"{doc.name}: {claim!r}")
    for claim in _xdoc_figure_claims(_module_docstrings()):
        offenders.append(f"this module's docstrings: {claim!r}")
    assert offenders == [], (
        "an architecture doc or this module attributes a figure to another doc in the "
        "present tense: " + "; ".join(offenders) +
        " — state the rule (that doc owns the number) or date the quote; a present-tense "
        "copy is wrong the next time the owner re-measures")

    # The scan's most important reader has to be shown reading something. A corpus `== []`
    # is also what a live body that stopped mentioning the owner at all would produce, and
    # the point of #3249's fix was to keep the pointer while dropping the number — so the
    # pointer's presence, and the sentence's figure-freeness, are asserted, and both are
    # shown red below rather than left as a shape the guard happens to be silent on.
    idx_sentences = [s for s in _sentences(_live_body(INDEX.read_text(encoding="utf-8")))
                     if re.search(_XDOC_REF, s)]
    assert idx_sentences, (
        "index.md's LIVE body no longer names another architecture doc in any sentence, "
        "so the corpus scan above has nothing left to read in the page #3249 was about — "
        "delete the pointer and the guard stays green while the parity it enforces dies")
    numbered = [s for s in idx_sentences if re.search(_FIGURE, s)]
    assert numbered == [], (
        f"index.md attributes a figure to another doc inside its own live sentence: "
        f"{[s.strip()[:90] for s in numbered]} — that is the #3249 sentence again")
    # …and the scan has to be READING this page. A corpus `== []` is equally what a page
    # that dropped the pointer altogether would produce, and clause 1 asks for the pointer
    # to survive with no number beside it — so #3249's own sentence goes back into the
    # shipped bytes and must come out of the scan. Whole page, because the clause lives
    # under `## Review log`, which the placement guard cuts and this one must not.
    shipped = INDEX.read_text(encoding="utf-8")
    revived = shipped.replace(INDEX_CLAUSE_HERE, "The suite is " + PRE_FIX_INDEX_CLAUSE)
    assert revived != shipped, (
        "the clause #3249 rewrote is gone from index.md, so this scan has no site left in "
        "the page it was written for — deleting the pointer is the other way the parity "
        "dies, and it would leave the corpus green")
    assert _xdoc_figure_claims(revived), (
        "restoring #3249's sentence into the shipped index.md reported nothing: the corpus "
        "scan is not reading this page, and clause 1 is unenforced")


def test_the_cross_doc_figure_guard_fires_on_the_rotten_sentences():
    """Negative control: all three pre-fix sentences must fire, and the neighbour must not.

    A corpus `== []` is the same green a regex that matches nothing gives, and this file
    has already been bitten twice by exactly that (#2700's digits-only pattern, #2701's
    set-based extractors). So the guard is handed the three sentences as they stood
    before #3249 — one per surface, the index's assembled without the retired integer
    this file bans — and must report each. The fourth assertion is the other direction:
    `index.md`'s own next sentence attributes GPU 2 to a doc in the present tense with a
    spelled-out number in the neighbourhood, and must stay out of the report, because
    the rule is about figures with a margin, not about every mention of another doc.
    """
    for sentence in PRE_FIX_CLAIMS:
        claims = _xdoc_figure_claims(sentence)
        assert len(claims) == 1, (
            f"the guard let the pre-fix sentence through: {sentence!r} reported "
            f"{claims} — if this goes quiet, the corpus green above is the pattern "
            "having stopped matching, not the prose having been fixed")
        assert claims[0] in " ".join(sentence.split()), (
            f"the guard reported {claims[0]!r}, which is not the sentence it was handed")
    assert _xdoc_figure_claims(CLEAN_ADJACENT_CLAIM) == [], (
        "the guard now fires on a spelled-out attribution with no tilde'd or "
        "comma-grouped figure, which is the owned-count case #2700 owns and would make "
        f"the Review log unreadable: {CLEAN_ADJACENT_CLAIM!r}")
    # Both directions of the date carve-out, so its silence is attributable to the stamp
    # and not to the pattern having stopped matching. The two strings are the same
    # sentence — `PRE_FIX_CLAIMS[0]` — with and without a date inside the ±60-char window.
    assert _xdoc_figure_claims(DATED_SILENT_CLAIM) == [], (
        f"{DATED_SILENT_CLAIM!r} is the dated form the acceptance text allows and should "
        "be silent; if it fires, no dated Review-log sentence is safe either and the "
        "guard gets widened away")
    assert len(_xdoc_figure_claims(UNDATED_FIRING_CLAIM)) == 1, (
        f"{UNDATED_FIRING_CLAIM!r} is the rot itself and must fire — this is the same "
        "sentence as the dated control above, so a green here with a red above is the "
        "carve-out working and a red here is the pattern dying")
    # …and the stamp only carves out while it is *in the clause*. The gap is measured in
    # the assertion, so the 60-character bound is the thing under test, not an assumption.
    far = UNDATED_FIRING_CLAIM + (" x" * 40) + " re-measured 2026-09-24"
    gap = far.index("2026-09-24") - (far.index("~9,600") + len("~9,600"))
    assert gap > 60, f"the control put the stamp only {gap} chars from the figure"
    assert len(_xdoc_figure_claims(far)) == 1, (
        f"a date {gap} chars from the figure silenced the guard; the carve-out is a "
        "clause-length bound, not a whole-paragraph one")

    # The third direction, and the one #3249's own fix could have hidden inside. The
    # rewritten clause retires a figure under a `2026-10-14` stamp; if that stamp fell
    # within the window, the page would be excused from re-inserting the very number the
    # sentence exists to drop. The probe is the SHIPPED bytes, because a string typed out
    # beside the pattern proves only that the pattern matches a string — the vacuity this
    # file has been bitten by twice (#2700's digits-only pattern, #2701's set extractors).
    shipped = INDEX.read_text(encoding="utf-8")
    at = shipped.index(INDEX_CLAUSE_HERE)
    near = shipped[max(0, at - 60):at + len(INDEX_CLAUSE_HERE) + 60]
    stamps = [d.group(0) for d in _DATE_STAMP.finditer(near)]
    assert not stamps, (
        f"a day stamp ({stamps}) sits within sixty characters of the clause #3249 rewrote, "
        "so the carve-out would excuse a figure planted back at that site — the sentence "
        "has to keep its retirement stamp out of the window, not merely beside it")
    planted = shipped.replace(INDEX_CLAUSE_HERE,
                              "The suite is " + PRE_FIX_INDEX_CLAUSE)
    assert _xdoc_figure_claims(planted), (
        "planting the retired clause back at the shipped site came back silent, so the "
        "rewritten sentence has crept inside its own date carve-out")


def test_only_the_doc_that_measures_the_suite_states_a_bare_suite_size():
    """The live figure has one home; everywhere else it is quoted history or nothing.

    #3249 is not a request to move `architecture/testing.md`'s number into the index —
    that would be the third definition the item is about. So placement is pinned and the
    value is not: hard-coding the current figure here would restart the rot this whole
    file exists to stop.

    The rule, measured rather than assumed: `testing.md` is the only doc whose live body
    states the suite's size as a plain present fact, and everywhere else a suite size may
    appear only as the record of a suite that **once** had that size. `index.md` is held to
    more than that — its live body may carry none at all, because the #3249 fix was to
    point at [[testing]] and quote nothing — and it carries the retired integer only inside
    `## Review log`, where a past correction is on record removing it. Three growth records
    in `automod.md`'s gate-latency history are registered in `SUITE_SIZE_RECORDS` by value,
    which cannot rot the way a line number does, and the node's own control shows an
    unregistered figure in that same doc is red. Whether `automod.md`'s three should also
    have their dates moved into the sentence rather than sitting in the table beside them is
    filed on #3249 as a finding, not taken here.

    The exemptions the shipped page depends on are re-checked rather than trusted,
    because deleting them is the easy way to make every guard above go green.
    """
    # The register has to be shown capable of red, or a green below is the pattern
    # matching nothing. A copy of the owner's current figure is the planted case, because
    # that is the third definition #3249 exists to stop.
    planted = ["~5,830", "3,271", "5,900", "~12,345"]
    assert _unregistered("automod.md", planted) == ["~12,345"], (
        f"the register let a fourth figure through: {_unregistered('automod.md', planted)}")
    assert _unregistered("index.md", ["~12,345"]) == ["~12,345"], (
        "the register treats index.md as having a registered home for suite sizes, so the "
        "page whose prose #3249 fixed is now unguarded")

    for doc_name, (unexpected, found) in _suite_placement_offenders(ARCH).items():
        assert not unexpected, (
            f"{doc_name}'s live prose states {unexpected} as a suite size with no "
            f"registration in SUITE_SIZE_RECORDS — outside {SUITE_SIZE_OWNER} a suite size "
            "is a definition nobody re-measures; point at the owner instead")
        assert sorted(set(found)) == sorted(SUITE_SIZE_RECORDS.get(doc_name, [])), (
            f"{doc_name}'s registered growth records are no longer the ones on file "
            f"(found {sorted(set(found))}, registered "
            f"{sorted(SUITE_SIZE_RECORDS.get(doc_name, []))}) — a record may be added with "
            "its history sentence, and a deleted one is a rewritten history")
    assert _live_suite_figures(ARCH / SUITE_SIZE_OWNER), (
        f"{SUITE_SIZE_OWNER}'s live body no longer states a suite size at all, so the "
        "docs that must not restate one have no owner to defer to")
    index = INDEX.read_text(encoding="utf-8")
    assert _live_suite_figures(INDEX) == [], (
        "index.md's live prose states a suite size again — the #3249 fix was to point at "
        "[[testing]] and quote no figure outside the log")
    assert ROTTED_FIGURE in _section(index, "## Review log"), (
        "the retired suite-size integer this page hand-typed has been scrubbed out of "
        "the Review log — #3249 rewrote that sentence's cross-doc clause, and a "
        "correction may not delete the quoted history it is recording")
    for phrase, heading in EXEMPT_COUNTS:
        assert phrase in _section(index, heading), (
            f"{phrase!r} has gone from {heading!r}, so this page's exemptions now "
            "survive only because the guard stopped looking")
    assert _row_first_cells(index).count("desktop") == 1, (
        "[[desktop]] heads more than one row again, or the row was deleted to satisfy a "
        "figure guard — #2701 is the duplicate, this item is not")
    # The register is CLOSED, not merely populated: one doc, the three figures this item
    # names. Each entry added is another live copy of a measurement only the owner may hold,
    # so the exception list is part of the rule and is pinned as one.
    assert list(SUITE_SIZE_RECORDS) == ["automod.md"], (
        f"suite sizes are registered as exceptions in {list(SUITE_SIZE_RECORDS)}, not in "
        "automod.md alone — every doc added here is a second home for the owner's "
        "measurement, which is the thing clause 5 refuses")
    assert SUITE_SIZE_RECORDS["automod.md"] == ["~5,830", "3,271", "5,900"], (
        f"automod.md's registered records are {SUITE_SIZE_RECORDS['automod.md']}, not the "
        "three growth figures on file (~5,830 at its line 1351, 3,271 at 2176, 5,900 at "
        "2938); a widened register is the placement rule quietly removed")
    assert index.count("[[desktop]]") >= 2, (
        "the Review-log mention of the deleted duplicate row has gone, and "
        "test_the_row_scope_is_what_keeps_the_review_log_from_reading_as_a_duplicate "
        "exists to keep the log from being 'fixed'")


def test_the_placement_guard_fires_on_a_figure_planted_into_a_real_page(tmp_path,
                                                                       monkeypatch):
    """A suite size planted into copies of the SHIPPED pages is reported, in both of them.

    A corpus `== []` is not evidence about the corpus: it reads identically when the walk
    works and nothing is out of place and when `_SUITE_FIGURE` has stopped matching a suite
    at all, and this file has been bitten by that exact vacuity twice (#2700's digits-only
    pattern, #2701's set-based extractors). So the guard runs over a staged tree carrying the
    real bytes of the pages #3249 is about — `index.md`, which hand-typed the retired count,
    and `automod.md`, whose three growth records are the only registered exceptions — with
    one suite-size sentence added to each live body.

    What must come back is the walk's own verdict, per doc: the planted figure and nothing
    else, so the registered records neither swallow it nor are reported with it. Clause 5
    asks for a plant into a real corpus file through the module's staging pattern rather
    than a list handed to `_unregistered`, which is what the `_unregistered` asserts at the
    top of the corpus node are: a unit check on the register, not a check that the walk
    reaches the pages.
    """
    import sys
    mod = sys.modules[__name__]
    fake_arch = tmp_path / "architecture"
    fake_arch.mkdir()
    for name in ("index.md", "automod.md", SUITE_SIZE_OWNER):
        (fake_arch / name).write_text(
            _live_body((ARCH / name).read_text(encoding="utf-8"))
            + f"\nThe suite now holds {PLANTED_FIGURE} tests under `tests/` for this "
              "sentence only.\n",
            encoding="utf-8")
    monkeypatch.setattr(mod, "ARCH", fake_arch)

    offenders = {n: u for n, (u, _f) in _suite_placement_offenders(fake_arch).items() if u}
    assert set(offenders) >= {"index.md", "automod.md"}, (
        f"planting a suite-size sentence into the staged live bodies of index.md and "
        f"automod.md produced {offenders} — on a page now stating {PLANTED_FIGURE} as the "
        "suite's size the guard is silent, so its silence over the real corpus is not "
        "evidence that the corpus is clean")
    for name in ("index.md", "automod.md"):
        assert offenders.get(name) == [PLANTED_FIGURE], (
            f"{name} reported {offenders.get(name)!r} instead of exactly "
            f"[{PLANTED_FIGURE!r}]: automod.md's three registered growth records must pass "
            "the plant untouched, and a plant that is reported twice or not at all means "
            "the register and the walk are not the same code")


def test_every_node_this_file_cites_in_prose_is_a_node_this_file_defines():
    """A citation to a test that does not exist is a pointer into nothing.

    Docstrings are what a reader of a red suite reads first, so a name in one is a promise.
    At main, line 411 of this file sent the reader to a node that is a strict PREFIX of the
    real one — three words short of
    `test_the_parity_check_can_see_a_count_in_a_covers_row`, and no such node exists here.
    The prose that documents the exemption carried a broken pointer, which is the same
    defect #3249 is about one level up: a claim about a source nobody can re-open.

    This docstring therefore does not spell the dead name out — this node would otherwise
    indict its own sentence, exactly as it indicts the one below it, and a guard that fires
    on its own explanation is a guard someone deletes. The name is rebuilt from the live
    node's name plus nothing, by trimming the real one back to the prefix main cited.
    """
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    defined = {n.name for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    assert defined, "this module defines no tests, so every citation in it is vacuous"

    prose = "\n".join(d for d in (
        [ast.get_docstring(tree) or ""]
        + [ast.get_docstring(n) or "" for n in tree.body
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]))

    def undefined_in(text: str) -> list[str]:
        out = []
        for name in sorted(set(re.findall(r"\btest_[a-z0-9_]+\b", text))):
            if name in defined:
                continue
            # `test_mc_tab_parity.py` and `test_infrastructure_gpu_inventory.py` are OTHER
            # files and legitimately named here; a bare `test_x` with no `.py` glued to it
            # is a promise about a node in this module.
            if not re.search(re.escape(name) + r"\.py", text):
                out.append(name)
        return out

    # Its own prefix-truncated citation, rebuilt from a live name so this docstring never
    # has to contain the dead string, has to be reported — or the scan is reading a corpus
    # that can no longer produce a hit and the `== []` below is the same green as no guard.
    live = "test_the_parity_check_can_see_a_count_in_a_covers_row"
    assert live in defined, f"{live} was renamed, and this node's control is stale"
    dead = live[: -len("_in_a_covers_row")]
    assert undefined_in(f"see `{dead}` for the negative control") == [dead], (
        "the scan went silent on a citation truncated to the exact shape main's line 411 "
        "had, so it enforces nothing and the green below is an artefact")

    bad = undefined_in(prose)
    assert bad == [], (
        f"prose cites node(s) {bad} that this module does not define — the citation sends "
        "a reader to a node that is not here; name the real one or say which file")


def test_the_parity_check_can_see_a_count_in_a_covers_row():
    """Negative control for the count half, in both directions it must fail in.

    The green above is a pair of nested extracts — rows out of the page, counts out of
    the Covers cell — and either one matching nothing looks exactly like a clean index.
    #2700 is the proof that the second one was matching nothing: its pattern read
    `\\d+` over two nouns, so every count on this page that a person spelled out was
    invisible to it for as long as the page existed. Hence three assertions:

    1. the number half fires on words as well as digits, over the nouns the tables
       use. Today's page contains all four of these shapes' clean versions, and none
       of the counted versions, so an `== []` above cannot be told from this line
       going quiet;
    2. a count planted in a Covers cell is reported **naming its doc**, which is what
       makes the message above actionable;
    3. the same phrase under `## Review log`, and again in the GPU-allocation prose,
       is not a Covers cell and is not reported. That is the scoping the count check
       needs and did not have: both regions carry retired counts on purpose, as the
       record of an error, and a guard that fired on them could only be silenced by
       rewriting history — which `test_the_retired_section_says_git_history_is_the_only_copy_left`
       and the Review-log node above exist to forbid.
    """
    spelled = _OWNED_COUNT.findall(
        "the five due-gates; nine rungs; two trust flags; the four lists")
    assert spelled == ["five due-gates", "nine rungs", "two trust flags", "four lists"], (
        f"the count guard saw {spelled} in four spelled-out counts — a count written "
        "as a word is the same second definition of a set as one written as a digit, "
        "and this is the hole #2700 files")
    assert _OWNED_COUNT.findall("the nine due-gates") == ["nine due-gates"], (
        "the guard no longer counts due-gates at all, so an empty result on the real "
        "index means the scan is blind, not that the page is clean")

    seeded = (
        "## Runtime\n\n"
        "| Doc | Covers |\n|---|---|\n"
        "| [[alpha]] | the mechanism: the nine due-gates, backoff, run records |\n"
        "| [[beta]] | the mechanism: its due-gates, backoff, run records |\n\n"
        "## GPU allocation\n\n"
        "Five cards, not three, since 2026-10-07. The prose names the nine due-gates.\n\n"
        "| GPU | Hardware |\n|---|---|\n| 0 | RTX 3090 24 GB |\n\n"
        "## Review log\n\n"
        "- **2026-09-14 — `stale`.** The page used to say the nine due-gates.\n")
    pairs = _covers_cells(seeded)
    assert [doc for doc, _ in pairs] == ["alpha", "beta"], (
        f"the row extractor found {pairs} in a page with two link rows, so an empty "
        "offender list above is the extractor being blind, not the index being clean")
    offenders = _owned_count_offenders(seeded)
    assert offenders == ["[[alpha]]: nine due-gates"], (
        f"a count planted in [[alpha]]'s Covers cell was not reported by that doc's "
        f"name, and [[beta]]'s count-free wording must stay clean: {offenders}")
    # The asymmetry, measured on the same bytes: the phrase occurs three times in
    # that fixture — [[alpha]]'s cell, the GPU prose, the Review-log entry — and the
    # row-scoped check reports exactly one of them. Hand the whole text to the pattern
    # and the scoping is what makes the difference, so this cannot be green because
    # the fixture stopped containing the phrase.
    assert _OWNED_COUNT.findall(seeded).count("nine due-gates") == 3, (
        f"a whole-text scan of the fixture finds "
        f"{_OWNED_COUNT.findall(seeded)} — the planted count is supposed to occur in "
        "the Covers cell, the GPU prose and the Review-log entry, so a different "
        "number here means the fixture lost an exempt region and the len()==1 below "
        "proves nothing")
    assert len(offenders) == 1, (
        "the Review-log entry or the GPU-allocation prose was fed to the count guard; "
        "to go green there someone would have to delete a retired count, which is the "
        f"record this file's other nodes protect: {offenders}")


def test_the_parity_check_can_see_a_duplicate_row():
    """Negative control, duplicate direction: the check must be able to fail.

    The control above only exercises the missing-row direction, which is how the
    set-based extractors stayed a green blind spot for the whole life of the
    duplicate desktop row. So the same discipline here, on a synthetic index: a stem
    heading two rows must be reported by name, one row must not be, and the extractor
    must be shown to see rows at all — otherwise `_duplicate_row_cells` returning `[]`
    everywhere, including from a regex that matched nothing, would satisfy both.
    """
    twice = "| [[alpha]] | a |\n| [[alpha]] | different cell |\n"
    assert _row_first_cells(twice) == ["alpha", "alpha"], (
        "the row extractor matched nothing, so an empty duplicate list below is "
        "the scanner being blind, not the index being clean")
    assert _duplicate_row_cells(twice) == ["alpha"], (
        "two rows heading with [[alpha]] and the check reported nothing")
    assert _duplicate_row_cells("| [[alpha]] | a |\n| [[beta]] | b |\n") == [], (
        "one row per doc must report nothing; a check that fires on the clean shape "
        "is a check that gets deleted")
    # The row anchor, not the file: a stem named in prose beside its single row is
    # the live index's own shape (the desktop row and its Review-log entry).
    assert _duplicate_row_cells(
        "| [[alpha]] | a |\n\nProse carried [[alpha]] in two rows.\n") == [], (
        "prose naming a stem is being counted as a second table row, which would "
        "make the Review log read as a duplicate of the row it describes")


def test_a_retired_doc_is_named_under_retired_and_linked_from_no_table():
    """A retirement is a name under Retired, never a `[[link]]` (#1704).

    `4a6cdd54` deleted `architecture/recall-research-2026-09-24.md` and left its
    table row in place, so the index kept promising a doc that had stopped
    existing — the failure above. The remedy it names is the convention checked
    here, over every retired name rather than that one: each must appear under
    Retired and in no link table, so recording a retirement cannot quietly
    re-create the stale promise, and a doc that came back cannot hide in the
    retired list.
    """
    assert _retirement_problems(INDEX.read_text(encoding="utf-8")) == []


def test_the_retirement_check_fires_on_a_retired_name_that_is_live_again(tmp_path,
                                                                        monkeypatch):
    """Negative control: both problem branches must fire, and the guards that
    make this check meaningful must refuse an unreadable section rather than
    report an empty problem list.
    """
    import sys
    fake_arch = tmp_path / "architecture"
    fake_arch.mkdir()
    # One problem each: alpha is retired yet still linked (no file behind it, so
    # only that branch fires); beta is retired and genuinely gone, which is the
    # clean case; gamma is retired while its doc is back on the directory.
    (fake_arch / "index.md").write_text(
        "| [[alpha]] | a |\n\n" + _RETIRED_HEADING + " (untracked)\n\n"
        "`alpha` and `beta` and `gamma` went.\n\n## Review log\n", encoding="utf-8")
    (fake_arch / "gamma.md").write_text("# gamma\n", encoding="utf-8")
    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, "ARCH", fake_arch)
    monkeypatch.setattr(mod, "INDEX", fake_arch / "index.md")
    problems = _retirement_problems((fake_arch / "index.md").read_text(encoding="utf-8"))
    assert sorted(p.split(":")[0] for p in problems) == ["alpha", "gamma"], problems
    assert any("still linked" in p for p in problems), problems
    assert any("on disk" in p for p in problems), problems
    # A retired section with no names, and no retired section at all, are both
    # hard failures: neither may read as "nothing to complain about".
    with pytest.raises(AssertionError, match="names no doc"):
        _retirement_problems("## Retired\n\nnothing named here\n")
    with pytest.raises(AssertionError, match="no Retired section"):
        _retirement_problems("# index\n\nno retired section at all\n")


#: The eleven docs `b94be171` retired, typed rather than read out of index.md.
#: The section under test is the only place in the repo that lists them, so a node
#: that asked that section what it names would be asking it the same question twice
#: and could not notice a rewrite that quietly dropped one.
RETIRED_ROSTER = (
    "agents", "background-monitoring", "evaluation-engine", "exploration-engine",
    "harness-comparison", "improvement-planner", "intelligence-pipeline",
    "nightly-vault-maintenance", "staged-pipeline", "usage-tracking",
    "verification-system",
)


def test_the_retired_section_says_git_history_is_the_only_copy_left():
    """#2329 clauses 3 and 4. The heading used to read
    `## Retired (in `.archive/`, or in git history alone)`, which offered a
    directory as the first option, and the section's own closing sentence said the
    twelfth name was "not in `.archive/` either". `architecture/.archive/` is not in
    this checkout and `git ls-files` shows nothing tracked under it, so the heading
    named a place a reader cannot go.

    The section now says the eleven are in git history alone and names the commit
    whose parent holds them. Pinned as a set of facts about the section, because
    `test_a_retired_doc_is_named_under_retired_and_linked_from_no_table` is about
    the *convention* (a name in backticks, never a `[[link]]`) and stays green no
    matter where the section claims the files are — which is the gap this round
    could otherwise have walked straight through.
    """
    text = INDEX.read_text(encoding="utf-8")
    section = _retired_section(text)
    heading = section.splitlines()[0]
    assert heading.startswith(_RETIRED_HEADING), (
        f"the section no longer opens with {_RETIRED_HEADING!r}, so the parity "
        f"check above has nothing to find: {heading!r}")
    assert ".archive" not in heading, (
        f"the heading offers .archive/ as a place the retired docs live again: "
        f"{heading!r} — the directory is not in this checkout")
    for name in RETIRED_ROSTER:
        assert f"`{name}`" in section, (
            f"{name} is no longer named in backticks under Retired; the roster is "
            "what a reader searches when looking for a doc that has gone missing")
    assert ".archive" not in section, (
        "the Retired section still mentions .archive/ somewhere besides its "
        "heading — the claim is what misleads, not only the heading")
    assert "b94be171" in section, (
        "the section names no commit, so a reader is told the text is in history "
        "with no way of finding which commit's parent holds it")
    assert "`git show b94be171^:architecture/<slug>.md`" in section, (
        "the section does not give the recovery command in the one form a reader "
        "can copy, with <slug> as the only variable part")
    assert text.count(".archive") == 2, (
        f"index.md mentions .archive {text.count('.archive')} times, not 2: the two "
        "that belong here are the standing retirement convention above the tables "
        "and the dated Review-log line that records the old claims being measured. "
        "A third is a live claim about where files are; a lost one is history "
        "erased rather than corrected")
    log = text[text.index("## Review log"):]
    assert "`.archive/` claims" in log, (
        "the Review-log entry that records how the .archive claims were measured "
        "has been edited; a correction may not rewrite the record of the error")
