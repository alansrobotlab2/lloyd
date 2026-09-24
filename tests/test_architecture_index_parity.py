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
