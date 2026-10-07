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
the index may not state a fleet or a rung count as a bare integer. Exempt, and
deliberately not matched by the pattern: the GPU table (hardware, three rows),
the retired-name list under "Retired" (a fixed historical set), and the
`~8,700-test suite` figure, which is an order of magnitude with a tilde, not
a count of a set with an enumerable owner.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "architecture"
INDEX = ARCH / "index.md"

_LINK = re.compile(r"\[\[([a-z0-9-]+)\]\]")
#: A hand-typed count of a set that has an owner elsewhere in the tree.
_OWNED_COUNT = re.compile(r"\b\d+\s*-?\s*(?:scheduled\s+jobs?|rungs?)\b", re.I)


def _docs_on_disk() -> set[str]:
    """Same walk as `arch_review.doc_slugs`: top level only, `.archive/` never."""
    return {p.stem for p in ARCH.glob("*.md") if p.stem != "index"}


def _docs_linked() -> set[str]:
    return set(_LINK.findall(INDEX.read_text(encoding="utf-8")))


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


def test_the_index_states_no_count_of_a_set_with_an_owner():
    hits = [m.group(0) for m in _OWNED_COUNT.finditer(INDEX.read_text(encoding="utf-8"))]
    assert hits == [], (
        f"index.md restates a count that an owner already enumerates: {hits} — "
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
    assert _OWNED_COUNT.findall("the ~8,700-test suite; 3 rows; 2026-09-11") == []


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
