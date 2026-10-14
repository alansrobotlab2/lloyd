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
    and that number went stale inside the same window #2700 was open in:
    `architecture/testing.md` line 11 and the gate's own collected count both measure
    ~9,600 now (9,583–9,620 across rounds). That is the identical failure the guard
    exists to prevent, committed in the guard's own prose — and it went unnoticed for
    the same reason the rotted Covers cells
    did: nothing read the sentence. So the exemption is now stated as a class (a tilde'd
    order of magnitude is a measurement with a margin, not a count of an owned set), and
    this node keeps it that way in both directions:

    1. the rotted figure may not come back, in the docstring or anywhere in this file;
    2. the class wording must stay, because without it the next reader has only an
       example to generalise from, which is how the sentence got a number in it;
    3. whatever the current figure is, the guard must still refuse to match it — the
       negative control in `test_the_parity_check_can_see_a_count` carries the
       measurement, and this asserts the class, not a specific suite size.
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
