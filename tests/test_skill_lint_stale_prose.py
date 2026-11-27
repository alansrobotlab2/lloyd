"""Nothing in either instruction surface may state an age ceiling (#4617 clauses 3-5).

Two documents teach a reader what `stale_context`'s six fields mean, and both were
writing the age arm as inert while the report they describe said otherwise:

- `scripts/skill_lint.py`'s `stale_context` docstring — the docstring of the very
  function that publishes `oldest_could_trip` — said "no file in the tree is old
  enough to trip the threshold", beside a run whose own dict read `max_age_days: 95`
  against `threshold_days: 90` with `oldest_could_trip: True`.
- `skills/nightly-skills-management/SKILL.md`'s `stale_context` bullet said
  "`max_age_days` was 67 against a 90-day threshold with `oldest_could_trip: false`,
  so nothing is stale, and `unmarked: 1` is a scan-coverage figure rather than a
  defect", and the step below it then commanded the run to "Report all three with
  those figures" — so job 83 was instructed to print, every night, numbers its own
  input contradicted.

The reason both survived is the one this repo has now recorded three times: a landed
state sentence rots, and the next run cites it instead of re-reading the tree. #4614
rendered the report's STALE section and `CATEGORY_TRUST["STALE"]`; #4615 fixed the
STALE row of `skills/skill-lint/SKILL.md`; each quoted this docstring as belonging to
it and neither opened it. So the fix is prose that carries no age and no threshold
figure in either direction and sends the reader to the run's own dict — and the guard
below is what stops the replacement rotting back into a number.

The matcher is ONE function, imported by `tests/test_skill_lint_report_trust.py` for
the code-side clauses as well: a ban that lives on one of two write surfaces is a ban
the other surface can walk past, and this item is precisely two surfaces.

The skill file is read from the live vault, deliberately unmarked rather than
`live_vault`, following
`tests/test_frontmatter_check_route.py` and
`tests/test_skill_tool_names.py::test_no_active_skill_or_task_names_a_path_absent_from_the_checkout`:
the vault is one shared tree, a round's base cannot make it stale, and the only way
this goes red is a real regression in the instruction a nightly run reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.automod import vault_round as VR  # noqa: E402
from tests.skill_folder import disk_skill_folder  # noqa: E402

VAULT = Path.home() / "obsidian"
SKILL_REL = "skills/nightly-skills-management/SKILL.md"
SKILL_FILE = VAULT / SKILL_REL

#: The heading whose section is graded, verbatim. #4617 renames no heading:
#: `tests/test_frontmatter_check_route.py` re-reads this file through `_section(...)`
#: anchors, and a renamed heading would break those anchors while looking like a
#: harmless restructure here.
SECTION_HEADING = ("### Read the skill-lint report before creating or updating any skill"
                   " (added 2026-11-04 — #3453)")

#: The body size of this skill measured immediately before #4617's edit
#: (`_skill_desc_and_body(SKILL_FILE.read_text())[1]`, the vault route's own definition,
#: on the tree at vault commit `43b1c030b`). The route refuses to GROW an over-cap skill
#: (`SKILL_BODY_CHAR_CAP`, `scripts/automod/vault_round.py:2908` and the size row at
#: `:3019`), so this is not a new rule: it is the bound that gate enforces at land time,
#: re-checked from the repo, so a landing route nobody ran cannot quietly undo it.
PRE_4617_BODY_CHARS = 104_803

#: The six keys `stale_context` publishes, and the phrase the bullet must keep saying:
#: the field list is the durable part of the sentence, the figures were not.
STALE_CONTEXT_FIELDS = ("threshold_days", "marked_active", "unmarked", "total",
                        "max_age_days", "oldest_could_trip")

#: Every spelling in which a frozen age figure has actually appeared in these two
#: surfaces. Naming one literal (`67`) would rot into the thing it polices, because the
#: next run writes its own number in a different shape.
FROZEN_AGE_FIGURE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b\d[\d,]*\s*-?\s*days?\b", "an age written as a number"),
    (r"max_age_days[^\n]{0,24}?\b\d[\d,]*\b", "a number following `max_age_days`"),
    (r"\bthreshold\w*[^\n]{0,24}?\b\d[\d,]*\b", "a number beside `threshold`"),
    (r"oldest_could_trip\s*[:=]\s*(false|true)", "`oldest_could_trip` given a value"),
    (r"unmarked\s*[:=]\s*\d+", "`unmarked` given a number"),
)


def frozen_age_figures(text: str) -> list[str]:
    """Which frozen-figure shapes `text` commits. Shared with the code-side clauses."""
    return [label for pattern, label in FROZEN_AGE_FIGURE_PATTERNS
            if re.search(pattern, text, re.IGNORECASE)]


#: The retired sentence, kept so the matcher above can be proven to bite on the text it
#: was written for rather than passing on an empty pattern list.
RETIRED_BULLET = """not a count of stale skills: on that scan
     `max_age_days` was 67 against a 90-day threshold with `oldest_could_trip: false`, so
     nothing is stale, and `unmarked: 1` is a scan-coverage figure rather than a defect."""


def _stale_section() -> str:
    """The section's text as its whole FOLDER, SKILL.md spliced with its siblings.

    A fold moves a section's body and leaves its heading behind, so a reader of
    `SKILL.md` alone would report the moved figures as deleted ones — the #4190
    misreading `tests/skill_folder.py` exists to prevent, and this skill folds
    (`live-vault-preflight-history.md` sits beside it).
    """
    assert SKILL_FILE.is_file(), f"no skill at {SKILL_FILE}: this file polices it"
    union = disk_skill_folder(SKILL_FILE.parent, encoding="utf-8")[1]
    start = union.index(SECTION_HEADING)
    rest = union[start + len(SECTION_HEADING):]
    nxt = re.search(r"^#{2,4} ", rest, re.MULTILINE)
    return rest[:nxt.start()] if nxt else rest


def test_the_frozen_age_figure_matcher_bites_on_the_sentence_it_replaces():
    """Positive control for every absence assertion below.

    A `not in text` test whose matcher matches nothing passes on any text, including a
    skill that still carries the retired figures — the failure mode
    `test_the_ban_on_the_unemitted_telemetry_claim_bites` exists to close, and the same
    trap is one edit away here. So the retired sentence is kept as a fixture and the
    matcher has to name four of its five shapes, and stay silent on the replacement.
    """
    assert len(FROZEN_AGE_FIGURE_PATTERNS) == 5, FROZEN_AGE_FIGURE_PATTERNS
    fired = frozen_age_figures(RETIRED_BULLET)
    assert len(fired) >= 4, (
        f"the matcher only found {fired} in the very sentence it was written for: it "
        "cannot be trusted to fail a future rewrite of it")
    assert frozen_age_figures(
        "read `oldest_could_trip`, `max_age_days` and `marked_active` of `total` off "
        "this run's own report; the age arm is measured, never asserted") == [], (
        "the matcher fires on field names alone, so no prose naming the fields can "
        "ever pass it and the clause would be unsatisfiable")


def test_the_nightly_skill_states_no_age_and_no_threshold_figure():
    """#4617 clause 3: the four falsified figures are gone from the section, not merely
    moved. Graded over the spliced folder, because a fold into a sibling would otherwise
    read as a fix while the run still reaches the figures under the same heading.
    """
    section = _stale_section()
    assert section, f"nothing under {SECTION_HEADING!r} — the section was deleted"
    for literal in ("67", "90-day", "oldest_could_trip: false", "unmarked: 1"):
        assert literal not in section, (
            f"the section still tells a nightly run `{literal}`, which the live "
            f"`stale_context` dict no longer says")
    assert frozen_age_figures(section) == [], (
        f"the section still carries {frozen_age_figures(section)}: an age figure in "
        "job 83's own instruction surface is the number the next pass cites instead of "
        "re-reading the report")


def _report_instruction(section: str) -> str:
    """The paragraph that tells the run what to report — the line #4617 clause 4 re-points.

    Scoped, not whole-file: the clause is about the instruction that read "Report all
    three with those figures", and a keyword scanned across the section would pass on
    the bullet's own field list while that sentence still ordered the run to recite
    frozen numbers (the reasoning `tests/test_retrieval_eval_skill_ci_contract.py`'s
    `section()` records for the same kind of scoping).
    """
    hits = [block for block in section.split("\n\n") if "Report all three" in block]
    assert len(hits) == 1, (
        f"expected exactly one 'Report all three' instruction, found {len(hits)}: the "
        "clause names that sentence and cannot be graded against a section that moved "
        "it, duplicated it, or dropped it")
    return hits[0]


def test_the_nightly_skill_keeps_the_six_fields_and_sends_the_run_to_its_own_dict():
    """#4617 clause 4: what the bullet must still say. Dropping the figures is only a
    fix if the field names and the pointer survive — otherwise the sentence has been
    made vague rather than made true, and the next reader has nothing to go and measure.
    """
    section = _stale_section()
    for field in STALE_CONTEXT_FIELDS:
        assert field in section, f"`{field}` vanished from the section with the figures"
    assert "not a count of stale skills" in section, section[:400]

    instruction = _report_instruction(section)
    assert "with those figures" not in instruction, (
        "the run is still ordered to report figures it did not measure, which is how a "
        "frozen number survived in this surface in the first place")
    assert re.search(r"(this|its) run's own", instruction), (
        "the instruction no longer sends the run to its own report — the only place the "
        "age verdict lives now")
    for field in ("oldest_could_trip", "max_age_days", "marked_active"):
        assert field in instruction, (
            f"`{field}` is the per-run source clause 4 requires the instruction to name")
    assert "of `total`" in instruction, (
        "`marked_active` without its denominator is the count that misleads: 205 marked "
        "of how many scanned?")



def test_the_landed_vault_edit_did_not_grow_the_over_cap_skill():
    """#4617 clause 5, measured through the route's own definition and its own gate.

    `nightly-skills-management` is at `over_char_cap: True` in the live lint report, so
    the size row at `vault_round.py:3019` refuses any landing that GROWS it: the fix had
    to be net-shrinking or it could not land at all. Asserted three ways — the cap, the
    route's body measurement against the pre-edit figure, and the gate itself returning
    no `skill body size` row for this skill — because a `would_refuse` row is only
    absent if the guard is armed, and an empty list from a disabled check is the green
    that reads as clean while nobody looks.
    """
    assert VR.SKILL_CONTRACT_ENFORCE is True, (
        "the skill contract route is switched off, so the empty finding list below "
        "would prove nothing")
    assert VR.SKILL_BODY_CHAR_CAP == 12_000, VR.SKILL_BODY_CHAR_CAP

    body_chars = VR._skill_desc_and_body(SKILL_FILE.read_text(encoding="utf-8"))[1]
    assert body_chars <= PRE_4617_BODY_CHARS, (
        f"{SKILL_REL} is {body_chars:,} body chars against the {PRE_4617_BODY_CHARS:,} "
        "measured before #4617: the age arm's fix grew an over-cap skill, which the "
        "vault route refuses for every future landing of this file")
    assert body_chars > VR.SKILL_BODY_CHAR_CAP, (
        f"{SKILL_REL} is now UNDER the {VR.SKILL_BODY_CHAR_CAP:,} cap, so this node's "
        "non-growing claim is no longer the binding constraint — retire it against the "
        "spill that got it there rather than leaving a test that checks nothing")

    rows = [row for row in VR.skill_contract_findings([SKILL_REL])
            if row.get("skill") == "nightly-skills-management"
            and row.get("gate") == "skill body size"]
    assert rows == [], rows
