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

Scoping to rows cuts both ways, and #4143 is the direction that was never taken: the
multiplicity half reads rows, while the membership half reads the whole file. A doc whose
row was deleted survives in that whole-file set as long as any `[[slug]]` of it is left in
prose — the GPU-allocation lead-in, or a Review-log entry — so "every doc has a row, one
row" is enforced only against a second row, never against a missing one, and the count
guard cannot see it either because it is fed rows and an unrowed doc simply stops being a
candidate. So membership is asserted over rows as well, by the same `_ROW_LINK` anchor and
with the same whole-file node left standing beside it: that node is the dangling-link
direction's mate, and the prose it counts is exactly what the row-scoped one excludes.

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

The same rule reaches one scope wider below the Review-log nodes, over the
vault-protection layer set. The index row for it was de-counted under this file's rule,
and the four other places that typed the same count — `README.md`'s table row,
`CLAUDE.md`'s vault section, and one sentence each in
`architecture/threat-model.md` and `architecture/background-runs.md` — were left
stating a number the tree had outgrown, because the count check was fed only
`index.md`'s rows. They now cite §2 of `architecture/vault-protection.md` as the list
and state no figure, and the one figure that survives is the owner page's own, pinned
there against its own §2 headings.
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


def _unrowed_docs(text: str) -> list[str]:
    """Every doc on disk that heads no table row of this page: `_docs_on_disk() -
    set(_row_first_cells(text))`, sorted by name.

    Membership at the scope the invariant is stated at. `_docs_linked()` is
    `set(_LINK.findall(whole file))`, which is the right scope for a dangling link and the
    wrong one for a row: a `[[slug]]` left in the GPU-allocation lead-in or in a Review-log
    entry keeps a deleted doc a member of that set, so `test_every_architecture_doc_is_linked_from_the_index`
    stays green over a page that lists the doc in no table and so carries no Covers cell for
    it. #4143. The count guard is no backstop either: it is fed rows, and a doc with no row
    simply stops being a candidate.

    Names the doc, never a count — the reader's job is to open that one table and put the
    row back. Nothing is subtracted from the on-disk side: a retired doc is named in
    backticks under Retired, and `_retirement_problems` already refuses a retired name that
    is on disk or linked, so the retired set has no member in `_docs_on_disk()` to except.
    """
    return sorted(_docs_on_disk() - set(_row_first_cells(text)))


def _without_row(text: str, slug: str) -> str:
    """The shipped page with the one table row heading `[[slug]]` deleted.

    A deletion from the real bytes, not a typed fixture: what #4143 claims is that THIS
    page, with THAT row gone, still reads complete to the whole-file check — and only the
    page itself can say where else the slug is named. Asserts exactly one line left, so a
    slug that headed no row or headed two cannot silently seed a control about a
    single-row doc.
    """
    pattern = re.compile(r"^[ \t]*\|[ \t]*\[\[" + re.escape(slug) + r"\]\][ \t]*\|")
    lines = text.splitlines(keepends=True)
    kept = [ln for ln in lines if not pattern.match(ln)]
    assert len(kept) == len(lines) - 1, (
        f"expected [[{slug}]] to head exactly one line of index.md, so deleting a row "
        f"is a one-line edit; found {len(lines) - len(kept)} — the seeded copy would "
        "then be deleting something other than the row this control is about")
    return "".join(kept)


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


def test_every_architecture_doc_heads_a_table_row():
    """Every doc on disk heads a row of a link table: membership, at row scope.

    `test_every_architecture_doc_is_linked_from_the_index` asks only whether the stem occurs
    as a `[[slug]]` somewhere in the page, and that is the whole of #4143. Delete the
    `[[workers]]` row from the shipped page and its Covers cell goes with it, while the slug
    is still named in the GPU-allocation lead-in and in a Review-log entry — measured on
    those bytes, the presence node reports nothing, the duplicate node reports nothing, the
    count guard reports nothing (it is fed rows, and an unrowed doc stops being a candidate),
    and the index advertises a doc it does not list. So the page's stated invariant, "every
    doc has a row, one row", was enforced only against a second row. This node is the missing
    direction, on the same `_ROW_LINK` anchor the node above already uses.

    The whole-file node stays, and stays whole-file: it is the dangling-link direction's
    mate, and a prose `[[slug]]` is precisely what that check has to keep finding. Widening
    `_ROW_LINK` to prose instead would buy the same blindness back on this node and re-break
    the scoping that stops the Review-log entry from reading as a second `[[desktop]]`.

    No retired name is excepted from the on-disk side, so no retired name is subtracted. A
    retirement is a backticked name under Retired and `_retirement_problems` refuses one that
    is on disk or linked, so the retired set has no member in `_docs_on_disk()` to take away,
    and the check below passes with the `## Retired` section standing as shipped — the three
    assertions after it are that claim measured, not asserted away.
    """
    text = INDEX.read_text(encoding="utf-8")
    assert _docs_on_disk(), (
        "architecture/ holds no doc besides index.md, so an empty unrowed list below is a "
        "walk that found nowhere to look and not a complete index")
    unrowed = _unrowed_docs(text)
    assert unrowed == [], (
        f"architecture docs heading no table row of index.md: {unrowed} — add a row, with a "
        "Covers cell, to the right table. A `[[slug]]` named in prose is not a listing: it is "
        "what keeps the whole-file presence node green over a doc no table carries")
    # Clause 4, measured. The Retired section is there, it names docs, and none of those
    # names is on disk or heads a row — so a subtraction would be a code path with nothing
    # to subtract, and a name showing up here is a broken retirement, not a reason to widen
    # the expression.
    retired = set(_BACKTICKED.findall(_retired_section(text)))
    assert retired, (
        "the Retired section names no doc, so this node's claim that retired names need no "
        "subtraction is a claim about an empty set")
    clashing = sorted(retired & (_docs_on_disk() | set(_row_first_cells(text))))
    assert clashing == [], (
        f"retired name(s) {clashing} are on disk or heading a row, which is "
        "`_retirement_problems`'s defect and not a reason to subtract the retired set here")
    # And the node is blind to that section by construction: the answer is read off rows, so
    # cutting the whole Retired block out of the text it is computed over cannot move it.
    body = _retired_section(text)
    assert text.count(body) == 1, (
        "the Retired section's body occurs more than once in index.md, so removing it to "
        "test this node's scope would remove something else too")
    assert _unrowed_docs(text.replace(body, "", 1)) == unrowed, (
        "coverage changed when the Retired section came out, so `_unrowed_docs` is reading "
        "prose it was written to ignore — the row anchor is not what is deciding the answer")


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


def test_a_deleted_row_stays_invisible_to_the_whole_file_presence_check():
    """The coverage gap, planted in the shipped page: rows out, the file still complete.

    #2701's control had to show that a set-based check could not see a SECOND row; this is
    the same blindness one direction over, shown the same way — on the shipped bytes, never
    on a sentence typed beside the pattern. For every doc the page names outside the row it
    heads (`[[workers]]`, `[[infrastructure]]`, `[[vllm]]` and the rest — the GPU-allocation
    lead-in, the Retired paragraphs and the Review log each mention some of them), deleting
    that one row:

    1. makes `_unrowed_docs` report exactly that doc, which is the point of it, and
    2. leaves `_docs_on_disk() - set(_LINK.findall(seed))` empty, which is the defect: the
       whole-file read `test_every_architecture_doc_is_linked_from_the_index` lives on cannot
       see the row go, because the prose mention outlives it.

    The pair, on the same bytes, is the whole of #4143: the two reads differ, so the file
    stays the dangling-link direction's mate and rows become the scope membership is checked
    at. Every one of the prose-named docs is seeded, not one, because the claim is about the
    read scope and not about a particular slug.
    """
    text = INDEX.read_text(encoding="utf-8")
    on_disk = _docs_on_disk()
    whole, rows = _LINK.findall(text), _row_first_cells(text)
    prose_named = sorted(d for d in on_disk if whole.count(d) > rows.count(d))
    assert prose_named, (
        "no doc on disk is named in prose outside a row it heads, so there is nothing left to "
        "seed: with no surviving mention the whole-file read would see a deleted row go, and "
        "this control would be proving a hole the page no longer has")
    for slug in prose_named:
        seed = _without_row(text, slug)
        assert _row_first_cells(seed).count(slug) == 0, (
            f"[[{slug}]]'s row was deleted from the seed and it still heads a row, so the "
            "row extractor is not reading the bytes this control edited")
        assert _unrowed_docs(seed) == [slug], (
            f"deleting [[{slug}]]'s row from the shipped page gave "
            f"{_unrowed_docs(seed)}: expected exactly that one doc, so either the row "
            "deletion took more than the row or the coverage read cannot see a row go")
        assert on_disk - set(_LINK.findall(seed)) == set(), (
            f"[[{slug}]] vanished from the whole page when its row went, so this seed no "
            "longer holds the prose mention the control exists to measure — the gap would be "
            "simulated, not reproduced")
        # Nothing else in this file reports the missing row either: the other two row-scoped
        # nodes stay green on the same bytes, which is why the new node had to be added
        # rather than one of these widened.
        assert _duplicate_row_cells(seed) == [], (
            f"deleting [[{slug}]]'s row made some other doc look duplicated, so the seed is "
            "no longer the page this control is about")
        assert _owned_count_offenders(seed) == [], (
            f"deleting [[{slug}]]'s row made the count guard fire, which the shipped page "
            "asserts it never does")


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


# --------------------------------------------------------------------------- #
# The same rule one scope wider: the vault-protection layer set.  (#4441)
# --------------------------------------------------------------------------- #

#: The page that owns the vault-protection layer set, and the one file the guard below
#: exempts.
#:
#: allowed: its own lead figure. Why a count is allowed there: `vault-protection.md` is
#: the only doc that enumerates the set as sections, so a number on that page sits beside
#: the thing it counts, and
#: `test_the_owner_pages_lead_figure_matches_its_own_enumeration` refuses it the moment §2
#: stops holding that many headings. Every other doc cites §2 and types no number.
VAULT_DOC = ARCH / "vault-protection.md"
README = ROOT / "README.md"
CLAUDE = ROOT / "CLAUDE.md"
#: `README.md`'s vault-protection table row, located by prefix rather than by line number:
#: the shipped assertion and the planted control below must land on the same row after
#: someone re-wraps the page.
README_VAULT_ROW = "| [vault-protection]("
#: `CLAUDE.md`'s vault section, named by the clause the item names it by.
CLAUDE_VAULT_HEADING = "## The vault is protected at the tool layer"

#: A count of *this* set: the shipped rot was the spelled word `four`, and #2700's lesson is
#: that a digits-only pattern watches that happen, so the figure is `_NUMBER` itself,
#: composed the way `_OWNED_COUNT` composes it. The noun must be PLURAL, because "one layer
#: up" is how this tree says "further up the stack" — `interaction-loop.md`,
#: `editing-safeguards.md` and `desktop-use.md` each use that idiom, one of them in a
#: paragraph that cites `vault-protection` — so a singular match would report a distance in
#: the stack as a count of this set. The `-?` covers the attributive hyphen ("four-layer").
_VAULT_LAYER_COUNT = re.compile(rf"\b{_NUMBER}\s*-?\s*layers\b", re.I)
#: How a doc points at the owner. Both shipped spellings: the wikilink
#: `[[vault-protection]]` that architecture prose uses, README's table form
#: `[vault-protection](architecture/vault-protection.md)`, and bare prose such as "the
#: vault-protection follow-up".
_VAULT_POINTER = re.compile(r"vault[\s-]*protection", re.I)
#: The figure spelled as a word, for the one place a count is allowed and must be checked.
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

#: Which mechanism each §2 ordinal is, as the shortest token a list must contain for that
#: mechanism to be NAMED. Keyed by the ordinal so the table cannot drift from the section it
#: indexes. Each marker matches its own §2 body: the lists are written at deliberately
#: different lengths — §2.2's heading is "Every trial is recorded", README's cell says "trial
#: recording", `CLAUDE.md`'s bullet says "Every bench or eval trial is recorded" — so the
#: marker is the word all three share, not any one of those sentences.
_VAULT_SET_MARKERS: dict[str, re.Pattern[str]] = {
    "2.1": re.compile(r"sandbox", re.I),
    "2.2": re.compile(r"trial", re.I),
    "2.3": re.compile(r"wholesale|protected_paths", re.I),
    "2.4": re.compile(r"tripwire|vaultwatch", re.I),
    "2.5": re.compile(r"snapshot", re.I),
    "2.6": re.compile(r"sync[- ]registration", re.I),
}

#: The counts of OTHER sets, quoted from the shipped files and located by that quote. Four
#: different sets share the noun "layer" and none of them is this one, so each row asserts
#: three things at once: the phrase is still in the file, `_vault_layer_count` still matches
#: the phrase alone (or the green below would only prove the pattern stopped matching), and
#: the wired guard reports nothing at that line.
UNRELATED_LAYER_SETS: tuple[tuple[str, str], ...] = (
    ("README.md", "Four layers fit a session to the model's window"),
    ("architecture/context-window.md", "Four layers, cheapest first"),
    ("architecture/context-window.md", "this doc's four layers run at turn start"),
    ("architecture/data-home.md", "A stray in the tree: four layers"),
    ("architecture/data-home.md", "Four layers, in the order a turn meets them"),
    ("architecture/data-home.md", "That gap is why the next two layers exist"),
    ("architecture/subliminal.md", "Three layers of context injection operate independently"),
)
#: Where the injection corpus keeps the layer count the guard must never reach, and the
#: count itself: quoted prompt material about transformer offload, in a tree the guard's
#: scope cannot name.
EVAL_CORPUS_DOC = "eval/injection_canary/corpus/persistence-digest-webpage.md"
EVAL_CORPUS_FIGURE = "last 4 layers"

#: The retired figure, planted into shipped bytes rather than typed beside the pattern: an
#: `== []` over a corpus is the same green as a guard that stopped matching, which is the
#: failure this file has been bitten by twice (#2700's digits-only pattern, #2701's
#: set-based extractors). Both spellings, because the word is the form that rotted and the
#: digit is the form a fresh edit would type.
PLANTED_VAULT_COUNTS: tuple[str, ...] = ("The four layers of this set are below.",
                                        "There are 6 layers here.")


def _vault_layer_count(text: str) -> str | None:
    """The figure `_VAULT_LAYER_COUNT` matches in `text`, or None. A function rather than the
    pattern so a caller that wants the matched words gets them in one step."""
    hit = _VAULT_LAYER_COUNT.search(text)
    return hit.group(0) if hit else None


def _vault_guard_files() -> list[Path]:
    """Every file the vault-layer guard reads: `README.md`, `CLAUDE.md`, and each
    `architecture/*.md` except the owner page, sorted. `eval/` is absent by construction,
    not by a filter: the injection corpus under it is quoted prompt material whose "last 4
    layers" is about transformer offload, and a guard that walked it would turn an edit to
    a fixture into a documentation failure.
    """
    return sorted([README, CLAUDE, *(p for p in ARCH.glob("*.md") if p != VAULT_DOC)])


def _vault_layer_offenders(name: str, text: str) -> list[str]:
    """Each `<figure> layers` in `name` that counts the vault-protection set, as
    `name:line: figure in context`.

    Pointer-scoped by section — the rule that keeps `index.md`'s rows honest (#2700),
    widened from a table row to a heading-delimited block because prose has no rows. Both
    narrower and wider scopes are wrong on the shipped tree, and both were measured before
    this was written: whole-FILE scope fires on `README.md`'s context-window row, sixteen
    lines from the vault-protection row in the same file, whose four layers belong to
    `context-window.md`; a paragraph-only window misses `CLAUDE.md`'s vault lead line, whose
    only pointer is the `Long version:` sentence below the list it introduces. A section is
    the unit both edges agree on, and `## Review log` keeps quoted history out of prose.
    """
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if re.match(r"^#{1,6} ", ln)]
    hits = []
    for i, ln in enumerate(lines):
        figure = _vault_layer_count(ln)
        if not figure:
            continue
        above = [h for h in heads if h <= i]
        below = [h for h in heads if h > i]
        block = "\n".join(lines[above[-1] if above else 0: below[0] if below else len(lines)])
        if _VAULT_POINTER.search(block):
            hits.append(f"{name}:{i + 1}: {figure!r} in {ln.strip()[:72]!r}")
    return hits


def _vault_layer_sites() -> list[str]:
    """`_vault_layer_offenders` over every shipped file in `_vault_guard_files()`."""
    out: list[str] = []
    for path in _vault_guard_files():
        out += _vault_layer_offenders(path.relative_to(ROOT).as_posix(),
                                      path.read_text(encoding="utf-8"))
    return out


def _section_two() -> dict[str, str]:
    """The owner page's §2, keyed by ordinal in insertion order, values the section body
    including its heading. The enumeration every list is checked against, so a mechanism
    added to §2 becomes a hole in every doc that fails to name it without any number being
    typed anywhere.
    """
    lines = VAULT_DOC.read_text(encoding="utf-8").splitlines()
    heads = [(i, ln) for i, ln in enumerate(lines) if re.match(r"^### 2\.\d+ ", ln)]
    out: dict[str, str] = {}
    for k, (i, ln) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        out[re.match(r"^### (2\.\d+) ", ln).group(1)] = "\n".join(lines[i:end])
    return out


def _numbered_items(text: str) -> list[str]:
    """The numbered items of the first ordered list in `text`, continuation lines folded in."""
    items: list[str] = []
    for ln in text.splitlines():
        if re.match(r"^\d+\. ", ln):
            items.append(ln.strip())
        elif items and ln.startswith("  ") and ln.strip():
            items[-1] += " " + ln.strip()
    return items


def _planted_row(text: str, prefix: str, clause: str) -> str:
    """`text` with `clause` appended to the Covers cell of the first table row starting with
    `prefix`, in place: the control is planted into the REAL page so a green shipped-tree
    node and a red planted one differ only by the plant.
    """
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.startswith(prefix):
            body, pipe, _tail = ln.rstrip("\n").rpartition("|")
            assert pipe, f"row {prefix!r} has no closing pipe to plant before"
            lines[i] = f"{body}{clause} {pipe}\n"
            return "".join(lines)
    raise AssertionError(f"no table row starts with {prefix!r}")


def test_no_shipped_doc_counts_the_vault_protection_layers():
    """No shipped doc outside the owner page says how many layers protect the vault.

    Five sites used to, and #4441 names four of them: `README.md`'s vault-protection table
    row, `CLAUDE.md`'s vault-section lead, `architecture/threat-model.md`'s pointer to the
    owner, and two sentences in `architecture/background-runs.md` that leaned on the list to
    say what the grant gate is NOT. All five said four. Four was wrong twice over: §2 grew a
    mechanism that no list mentioned (§2.2, every trial recorded) and had always carried
    another the lists left out (§2.6, no tool call may change the sync registration) — the
    second is the omission this item calls the worse half, because `app/harness/`
    `sync_registration.py` exists only because Lloyd deleted the sync registration twice on
    2026-09-14 from a Mission Control chat, and the one page a newcomer reads for the layer
    set never named it. `CLAUDE.md` is the copy an agent actually reads, which is why its
    list now carries all six inside the numbered items rather than demoting the sync guard
    to a trailing aside.

    The fix is the ruling #3131 reached for `check_bash_command`'s lane chain: the number is
    removed rather than corrected, and each site cites §2 of the owner page as the list —
    which is what `index.md`'s own row has done since #2700, and why its row was the only
    one of the six that stayed true. An empty list is the invariant; the controls below are
    what make it mean something.
    """
    offenders = _vault_layer_sites()
    assert offenders == [], "shipped docs count the vault-protection layers: " + "; ".join(offenders)


def test_the_vault_layer_guard_fires_on_a_count_planted_into_readmes_row():
    """The guard is wired: the page it reads green goes red on a planted count, at that row.

    `README.md`'s vault-protection row is where the retired number comes back first — it is
    the sentence a newcomer reads — so the control is planted there, into the shipped bytes,
    at the row the guard is about. Each spelling alone must produce exactly one offender, and
    it must name README at the vault-protection row's line and nowhere else in the file; the
    unplanted page must be clean in the same call, or the control proves nothing about the
    shipped tree.
    """
    text = README.read_text(encoding="utf-8")
    assert _vault_layer_offenders("README.md", text) == [], (
        "README.md already trips the vault-layer guard, so a planted count there could "
        "not be attributed to the plant")
    row_line = next(ln for ln in text.splitlines() if ln.startswith(README_VAULT_ROW))
    for planted in PLANTED_VAULT_COUNTS:
        hits = _vault_layer_offenders("README.md", _planted_row(text, README_VAULT_ROW, planted))
        assert len(hits) == 1, f"planting {planted!r} into README's row produced {hits}"
        line_no = int(hits[0].split(":")[1])
        assert row_line[:24] in text.splitlines()[line_no - 1], (
            f"the planted count was reported at line {line_no}, which is not the "
            "vault-protection row, so the guard is firing on some other site")


def test_the_vault_layer_guard_stays_off_the_other_layer_sets():
    """Seven other counts of layers ship in these files, and this guard touches none of them.

    One noun, four sets. `architecture/context-window.md` heads a section "Four layers,
    cheapest first" and says "this doc's four layers run at turn start" — its compaction
    stack, whose owner is that doc; `architecture/data-home.md` counts its tree-strays guards
    three times, including "That gap is why the next two layers exist" under its own
    Protection heading; `architecture/subliminal.md` has "Three layers of context injection";
    and `README.md`'s context-window table row still says "Four layers fit a session to the
    model's window", sixteen lines from the vault-protection row this guard owns — which is
    the whole reason the rule is pointer-scoped and not file-scoped. That row is the same rot
    one line away (`architecture/index.md`'s row for it is de-counted while
    `context-window.md`'s own section still heads with its count); it is a different set and a
    different item, recorded on #4441 rather than fixed here.

    `EVAL_CORPUS_DOC` is out of scope by construction and asserted so: `_vault_guard_files()`
    names `README.md`, `CLAUDE.md` and `architecture/*.md` and nothing else, so no edit to
    the injection corpus can ever be read as a documentation failure.
    """
    scoped = {p.relative_to(ROOT).as_posix() for p in _vault_guard_files()}
    for rel, phrase in UNRELATED_LAYER_SETS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert phrase in text, f"{rel} no longer ships {phrase!r}, so this control moved"
        assert _vault_layer_count(phrase), (
            f"{phrase!r} is no longer matched by the count pattern, so the guard's silence "
            f"on {rel} proves nothing about the guard")
        at = next(i for i, ln in enumerate(text.splitlines(), 1) if phrase in ln)
        near = [h for h in _vault_layer_offenders(rel, text) if h.startswith(f"{rel}:{at}:")]
        assert near == [], f"the guard fired on an unrelated set: {near}"
    corpus = (ROOT / EVAL_CORPUS_DOC).read_text(encoding="utf-8")
    assert EVAL_CORPUS_FIGURE in corpus, (
        "the eval corpus no longer holds the transformer-layer figure the scope carve-out "
        "exists for, so the assertion below has nothing to keep out")
    assert EVAL_CORPUS_DOC not in scoped, "the vault-layer guard's scope reached the corpus"
    assert not any(p.startswith("eval/") for p in scoped), sorted(scoped)
    assert {"README.md", "CLAUDE.md", "architecture/index.md"} <= scoped, sorted(scoped)
    assert "architecture/vault-protection.md" not in scoped, (
        "the owner page is inside the scope its lead figure is allowed in")


def test_readmes_vault_row_names_the_whole_set_without_counting_it():
    """README's row names all six mechanisms and states no number, so it cannot rot by growth.

    The row used to advertise "the four layers built after them: tool sandbox,
    wholesale-delete refusal, guardian tripwire, 15-minute snapshots". Half of that was
    missing rather than miscounted: trial recording (§2.2) and the sync-registration guard
    (§2.6) were absent, and the second omission is the one that matters, because the guard
    that answers a deleted sync registration is exactly what a reader would go looking for
    after an incident. It now lists all six and cites §2, matching `architecture/index.md`'s
    row, and holds no figure — so when §2 gains a seventh mechanism this row goes red for
    not naming it (`_VAULT_SET_MARKERS`), not for being one short of a number nobody typed.
    """
    row = next(ln for ln in README.read_text(encoding="utf-8").splitlines()
               if ln.startswith(README_VAULT_ROW))
    assert _vault_layer_count(row) is None, (
        f"the README vault row counts the set again ({_vault_layer_count(row)!r}); the "
        "ruling on this set is that no doc but the owner page states a figure")
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert len(cells) == 2, f"the row no longer parses as link-cell plus Covers-cell: {row!r}"
    cell = cells[1]
    assert "layers" in cell, (
        "the row stopped naming the set at all, which de-counts it by deleting the claim "
        "rather than by citing the owner")
    assert "§2" in cell, (
        "the row names mechanisms but cites no section, so a reader has no way to check "
        "the list against the page that owns it")
    missing = [o for o, rx in _VAULT_SET_MARKERS.items() if not rx.search(cell)]
    assert missing == [], (
        f"§2.{','.join(missing)} is not named in README's row: the cell must name every "
        "mechanism the owner page enumerates")


def test_claude_md_lists_every_mechanism_the_owner_page_enumerates():
    """`CLAUDE.md`'s numbered list is §2 in §2's order: six items, no count, no trailing aside.

    This is the file a turn actually reads, and it was the worst of the five sites: its lead
    said "Four layers now, none of which trusts the model or the caller", trial recording
    (§2.2) appeared nowhere, and the sync-registration guard (§2.6) survived only as an
    unnumbered sentence below the list, where nothing indexes it. The list now has one item
    per §2 heading, in §2's order, each carrying that mechanism's marker, and the lead states
    no figure — so the item count and the section count are pinned together and the file
    cannot quietly drop back to four. The retired aside sentence is asserted absent, because
    a list that grows back a trailing paragraph has not been fixed, only re-wrapped.
    """
    text = CLAUDE.read_text(encoding="utf-8")
    assert CLAUDE_VAULT_HEADING in text, (
        f"CLAUDE.md no longer heads its vault section {CLAUDE_VAULT_HEADING!r}, so this "
        "node is reading the wrong section")
    body = _section(text, CLAUDE_VAULT_HEADING)
    for ln in body.splitlines():
        assert _vault_layer_count(ln) is None, (
            f"the vault section counts the set: {ln.strip()[:72]!r}")
    sections = _section_two()
    items = _numbered_items(body)
    assert len(items) == len(sections), (
        f"CLAUDE.md's vault list has {len(items)} numbered items and §2 enumerates "
        f"{len(sections)} mechanisms: one item per mechanism, in §2's order")
    for (ordinal, sbody), item in zip(sections.items(), items):
        assert _VAULT_SET_MARKERS[ordinal].search(item), (
            f"item {ordinal} of CLAUDE.md's list does not name mechanism §2.{ordinal}: "
            f"{item[:80]!r} against {sbody.splitlines()[0]!r}")
    assert "The sync registration is guarded the same way" not in text, (
        "the sync-registration guard is back in a trailing aside below the list; §2.6 is a "
        "numbered item here or the count is wrong again")


def test_the_three_prose_sites_cite_the_owners_section_instead_of_counting():
    """The three sentences that leaned on the list now cite §2 of the owner page as the list.

    `architecture/threat-model.md` pointed at "the vault's four layers";
    `architecture/background-runs.md` said "vault protection is now four layers below any
    hook" and then named four mechanisms, and its Known-limits bullet closed with "the grant
    gate is none of those four layers". Each sentence did real work — the last one says what
    the grant gate is NOT — so each keeps its point and loses its number, and now reaches §2
    by section rather than by page: a pointer to the doc as a whole is how a count survived
    beside a de-counted index for two months. The phrases are located by the words that
    replaced them, so a re-wrap cannot hide a site from this node, and the citation must sit
    in the same few lines as the sentence that needs it.
    """
    for rel, phrase in (("architecture/threat-model.md",
                         "the vault's layers are §2 of"),
                        ("architecture/background-runs.md", "is the list of them"),
                        ("architecture/background-runs.md",
                         "the grant gate is none of the layers")):
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        at = [i for i, ln in enumerate(lines) if phrase in ln]
        assert at, f"{rel} no longer says {phrase!r}: the sentence this node pins moved"
        window = "\n".join(lines[max(0, at[0] - 2): at[0] + 3])
        assert "§2" in window, f"{rel}: {phrase!r} cites the page but not §2 — {window!r}"
        assert _VAULT_POINTER.search(window), (
            f"{rel}: {phrase!r} cites a §2 with no owner page named beside it")


def test_the_owner_pages_lead_figure_matches_its_own_enumeration():
    """The one count this file allows is checked against the headings it counts.

    `architecture/vault-protection.md`'s lead says "Six mechanisms, in §2.1–§2.6" and its §2
    holds six `### 2.N` headings. That is the only place a figure survives, and it survives
    because it is checkable where it is read: the sentence is on the same page as the
    enumeration, so unlike the four that rotted in README, CLAUDE.md and the two prose sites
    it cannot disagree with anything. The node refuses both ways — a lead figure that stops
    matching the headings, and a lead that stops stating a figure at all, which would leave
    the exemption below with nothing to exempt.

    The exemption is then shown to be a scope decision and not a pattern that misses the
    page: the owner's own front block still carries the retired figure inside quoted history
    ("It said \"four layers\" while §2.2 and §2.6 existed"), so the same function, run on the
    same bytes, does report it — and the file is not in `_vault_guard_files()`, which is what
    keeps that report from being a failure.
    """
    text = VAULT_DOC.read_text(encoding="utf-8")
    front = text.split("\n## ")[0]
    stated = None
    for hit in re.finditer(_NUMBER, front, re.I):
        if re.match(r"^\s+mechanisms\b", front[hit.end():]):
            word = hit.group(0).lower()
            stated = _WORD_NUMBERS.get(word, None) if not word.isdigit() else int(word)
            break
    sections = _section_two()
    assert stated is not None, (
        "the owner page's lead states no figure beside `mechanisms`, so the one count this "
        "file allows has nothing left to check against §2")
    assert stated == len(sections), (
        f"the owner page's lead says {stated} mechanisms and §2 holds {len(sections)} "
        "headings; the figure and the enumeration are on one page and must agree")
    quoted = _vault_layer_offenders("architecture/vault-protection.md", text)
    assert quoted, (
        "the owner page's front block no longer holds a count its own prose points at, so "
        "excluding it from the guard is no longer a decision anyone can see")
    #: allowed: the owner page's own lead figure. Why a count is allowed here when no other
    #: doc may type one: §2 is on the same page as the sentence stating the figure, so this
    #: number is checkable where it is read, which is the one thing the counts in README,
    #: CLAUDE.md and the two prose sites never had. The exemption is paid for by the
    #: assertion above, which refuses the page a figure that disagrees with its own headings.
    assert "architecture/vault-protection.md" not in {
        p.relative_to(ROOT).as_posix() for p in _vault_guard_files()
    }, "the owner page is inside the scope its lead figure is allowed to stand in"


def test_the_vault_set_marker_table_indexes_every_section_two_mechanism():
    """A mechanism added to §2 is a hole in every list the moment it is added.

    The marker table is the only thing that turns §2's growth into a red node in README and
    `CLAUDE.md`, so it is pinned to §2 from both ends: the table's keys are exactly the
    ordinals §2 heads, and each marker matches the body of its own section — proof that the
    token is about the mechanism it indexes and not merely about the page. §2.2 and §2.6 are
    how this set rotted once: they existed, and the four-item lists beside them stayed at
    four because nothing compared a list against the section.
    """
    sections = _section_two()
    assert list(sections) == sorted(sections), (
        f"§2's headings are not in ordinal order ({list(sections)}), so 'one item per "
        "section, in order' in CLAUDE.md's list would be pinned to nothing")
    assert set(sections) == set(_VAULT_SET_MARKERS), (
        f"§2 enumerates {sorted(sections)} and the marker table knows {sorted(_VAULT_SET_MARKERS)}")
    for ordinal, body in sections.items():
        assert _VAULT_SET_MARKERS[ordinal].search(body), (
            f"the marker for §2.{ordinal} matches nothing in §2.{ordinal}'s own section, so "
            "it does not identify that mechanism")
