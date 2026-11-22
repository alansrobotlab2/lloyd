"""The promise a folded skill publishes about its siblings is checked, and can fail.

`skills/autonomy-data-pipeline/SKILL.md` ends with the sentence that names this file as
its enforcement, and item #4496 found the enforcement missing: the citation resolved to
no path in the checkout, which is three red nodes in
`tests/test_skill_tool_names.py` and the exact defect that module exists to catch. The
modules that *read* the folded skill had also been reading one file and reporting moved
prose as deleted prose (#4496's other 37 nodes), which `tests/skill_folder.py` now
repairs; this module is what keeps a future fold from re-breaking it silently.

The check is asked of a folder that PUBLISHES the promise — a ``Full text: `…` `` pointer
line in its `SKILL.md` — and not of every sibling-carrying skill on the box. Measured
here 2026-11-22: 20 skills carry a sibling at all, and `kg-mention-classifier` alone held
23 `run-log-*.md` files, 20 of them named by no index bullet or pointer, so a fleet-wide
index rail would red the next author for a fold nobody ruled in — which is how a check
gets disabled, the argument `tests/test_skill_tool_names.py` makes for its own ledger.
`test_the_fleet_of_skills_…` is the fleet-wide half: every skill that made the promise is
clean. The unnamed-sibling census is printed by that node rather than asserted, for the
same reason.

These nodes read the live vault and are deliberately UNMARKED, not `live_vault`, like
every other module in this family (`test_skill_tool_names.py`, `test_yaml_fix_skill_claims.py`,
`test_skill_stranded_ruled_sets.py` all say so and cite the same trade): a guard that
the gate deselects enforces nothing, and a fold is a vault write the vault-side gate
runs these over.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.skill_sibling_reachability import (  # noqa: E402
    INDEX_BULLET_RE, folder_violations, missing_index_files, pointer_rows,
    publishes_reachability_claim, unreachable)
from tests.skill_folder import splice_union  # noqa: E402

#: The vault's skill root. `Path.home()`, like every other module that reads these files
#: (`test_vault_commit_identity.py`'s `corpus_dir`, `test_skill_step25_label_pin.py:80`):
#: inside a gate run `$HOME` is the round's home, whose `obsidian` is a symlink into the
#: house vault — `scripts/automod/worktree.py::HOME_LINK_SKIP` skips `lloyd` and
#: `lloyd-data`, not `obsidian` — so the name resolves there too, whereas a path derived
#: from this file's own location would point into the round's checkout and find no vault.
SKILLS_DIR = Path.home() / "obsidian" / "skills"

#: The skill whose SKILL.md names this file as its enforcement.
SKILL_DIR = SKILLS_DIR / "autonomy-data-pipeline"
SKILL = SKILL_DIR / "SKILL.md"

#: A two-section skill with one section folded out, used as the shape the negative
#: controls break. `## Kept` stays in `SKILL.md`; `## Folded` lives in `sibling.md`.
MAIN = """# Index

## Kept

The rule that stayed behind.

## Folded

Full text: `sibling.md` — read before running this step.
"""

SIBLING = """## Kept

Not here.

## Folded

The moved rule: print the denominator or the step did not happen.
"""


def _siblings(main: str, *pairs: tuple[str, str]) -> dict[str, str]:
    return dict(pairs)


def test_the_folded_skill_keeps_every_pointered_section_reachable():
    """The live claim: every `Full text:` pointer resolves to text under its heading.

    Three shapes are one assertion because all three make the same reader blind — a
    sibling that was never written, one that carries the section under a reworded
    heading (`splice_union` joins by title, so the prose lands in a tail append and a
    section-scoped reader never sees it), and a heading left empty. The skill carries 17
    `Full text:` pointers across 9 files, measured 2026-11-22, so the count assertion
    below is what stops a fold that deleted every pointer reading as a pass.
    """
    assert SKILL.is_file(), f"no {SKILL}: this module polices that file, so it reports"
    violations = folder_violations(SKILL_DIR)
    assert violations == [], (
        "the fold published in the skill's last line is not true of the folder:\n  "
        + "\n  ".join(violations))
    rows = pointer_rows(SKILL.read_text(encoding="utf-8"))
    assert len(rows) == 17, (
        f"the skill carries {len(rows)} `Full text:` pointers, expected 17 as measured "
        "2026-11-22 — a different count means sections moved again, and this "
        "node's silence about a *missing* pointer is the hole to keep shut")


def test_a_pointer_to_a_sibling_that_is_not_in_the_folder_is_named(tmp_path):
    """Non-vacuity, shape one: the file the pointer names does not exist.

    This is the #85 failure class in prose form — a doc naming a file the checkout
    lacks — and without this control the clean live result above would only prove the
    checker never found anything to look at."""
    folder = tmp_path / "folded"
    folder.mkdir()
    (folder / "SKILL.md").write_text(MAIN, encoding="utf-8")
    assert pointer_rows(MAIN) == [("Folded", "sibling.md")], (
        "the fixture stopped being the shape the checker reads")
    problems = unreachable(MAIN, _siblings(MAIN))
    assert problems == ["`sibling.md` named under `Folded` is not in the folder"], problems
    assert folder_violations(folder) == problems, (
        "the disk route disagrees with the pure checker, so the live node above is "
        "grading a different rule than the controls here")


def test_a_relabelled_heading_in_the_sibling_is_named_naming_both_titles():
    """Non-vacuity, shape two: the sibling exists but retitled the section.

    #4192's finding is the reason this is the important control: `splice_union` joins a
    sibling's body to `SKILL.md` only on an exact stripped-title match, so
    `## Folded, rewritten` against `## Folded` puts the prose in the WHOLE-FILE union
    (appended, where a keyword scan still finds it) but not in the section span — every
    section-scoped assertion loses it while every keyword assertion passes. The message
    from `unreachable` must name both titles, because the fix is to re-copy the heading
    line verbatim."""
    moved = SIBLING.replace("## Folded\n", "## Folded, rewritten\n")
    problems = unreachable(MAIN, _siblings(MAIN, ("sibling.md", moved)))
    assert len(problems) == 1, problems
    assert "sibling.md" in problems[0] and "Folded" in problems[0], problems[0]
    assert "Folded, rewritten" in problems[0], (
        f"the message must name the heading the sibling actually carries: {problems[0]}")
    union = splice_union(MAIN, [("sibling.md", moved)])
    assert "The moved rule" in union, (
        "the retitled section's text should survive somewhere in the whole-file union — "
        "that is what makes a keyword assertion pass on a broken fold")
    span = union[union.index("## Folded\n"):]
    span = span[:span.index("\n## ")] if "\n## " in span else span
    assert "The moved rule" not in span, (
        f"a section-scoped reader is supposed to lose a retitled sibling's text; it "
        f"didn't, so this control no longer describes the defect: {span!r}")


def test_a_heading_the_sibling_carries_with_no_text_under_it_is_named():
    """Non-vacuity, shape three: an empty section is a landmark, not the promised text."""
    empty = SIBLING.replace(
        "The moved rule: print the denominator or the step did not happen.", "")
    problems = unreachable(MAIN, _siblings(MAIN, ("sibling.md", empty)))
    assert problems == ["`sibling.md` carries `Folded` with no text under it"], problems


def test_an_index_naming_a_sibling_the_folder_lacks_is_named(tmp_path):
    """The index half: the `Files in this skill` list is an instruction to go read a file.

    A run that obeys the index and finds nothing improvises, which is the behaviour
    #1240 was filed for. The live folder's 11 index bullets and its 11 sibling files
    agree (measured 2026-11-22), so this control is what proves the direction is
    actually enforced rather than trivially satisfied by a folder in agreement."""
    folder = tmp_path / "indexed"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        MAIN + "\n## Files in this skill\n\n- `sibling.md` — the folded rule\n"
               "- `never-written.md` — a rule nobody moved\n", encoding="utf-8")
    assert missing_index_files(MAIN, _siblings(MAIN, ("sibling.md", SIBLING))) == [], (
        "the control's positive arm must hold, or the negative arm proves nothing")
    problems = folder_violations(folder)
    assert any("never-written.md" in p for p in problems), (
        f"an index bullet naming an absent file went unreported: {problems}")


def test_the_fleet_of_skills_that_publish_the_promise_are_all_reachable():
    """Every skill on this box that left a `Full text:` pointer is clean.

    Scoped by the promise, not by the filesystem: `publishes_reachability_claim` is what
    puts a folder inside this loop, so a skill that keeps all its prose in `SKILL.md` is
    not judged by a contract it never signed, and a skill that folds a section tomorrow
    is judged the moment it says so. A sibling the `Files in this skill` index never
    named is PRINTED rather than asserted — the census printed 20 unnamed
    `run-log-*.md` files in `kg-mention-classifier` on 2026-11-22, and those are daily run
    logs, whose growing count is the point of having them, so a rail there is a rail that
    gets disabled.
    """
    claiming, census = _skill_folders()
    assert claiming, (
        "no skill folder carries a `Full text:` pointer line, which means the fold route "
        "this check polices has been un-done everywhere and the check is dead")
    bad = {name: rows for name, rows in
           ((name, folder_violations(path)) for name, path in sorted(claiming.items()))
           if rows}
    assert bad == {}, ("folded skills whose published pointer text is not reachable:\n"
                       + "\n".join(f"  {name}: {rows}" for name, rows in sorted(bad.items())))
    print(f"skills publishing the promise: {len(claiming)}; siblings the index does "
          f"not name (reported, not asserted): {census or 'none'}")


def _skill_folders() -> tuple[dict[str, Path], dict[str, list[str]]]:
    """`(name -> folder)` for the skills that publish the promise, plus the unnamed census."""
    claiming: dict[str, Path] = {}
    census: dict[str, list[str]] = {}
    for folder in sorted(p for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file()):
        main = (folder / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        named = set(INDEX_BULLET_RE.findall(main)) | {
            name for _title, name in pointer_rows(main)}
        unlisted = sorted(p.name for p in folder.glob("*.md")
                          if p.name != "SKILL.md" and p.name not in named)
        if unlisted:
            census[folder.name] = unlisted
        if publishes_reachability_claim(main):
            claiming[folder.name] = folder
    return claiming, census
