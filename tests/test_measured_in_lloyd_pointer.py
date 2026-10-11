"""#2554 — the `Measured in Lloyd` convention reaches a CLOSING actor, not just a writer.

`~/obsidian/knowledge/KNOWLEDGE_SCHEMA.md:107` defines `## Measured in Lloyd`: the one
optional section a knowledge note gains when the backlog item it filed closes and a
dated measurement artifact exists for it. Since it landed (vault `0050139c`, 2026-10-08
15:41) no qualifying closure has written one, and the reason is structural rather than
laziness, measured on the base of this round (`git grep` at 5096c448):

    git grep -n KNOWLEDGE_SCHEMA -- scripts/automod/          -> 0 lines
    git grep -n "knowledge/" -- workers/sources/autocode.py   -> 0 lines
    grep -c 'Measured in Lloyd' skills/automod-change-own-code/SKILL.md  -> 0
    grep -c 'KNOWLEDGE_SCHEMA'  skills/automod-change-own-code/SKILL.md  -> 0

So the entry sits only on the surfaces a note AUTHOR loads, and a closing session meets
the string only when some item body happens to quote it. That last clause is what makes
it a convention with no writer: a scan of the live `~/lloyd-data/sessions` on 2026-10-11
(2666 `*_review_*`, 635 `*_autocode_*`, 574 `*_boardsteward_*` session files) found 38,
24 and 2 files mentioning `KNOWLEDGE_SCHEMA`, and every `messages[0]` hit was a quoted
artifact arriving with an item that named the schema — `20260924_103823_review_8dce`
carries the #872 okf_taxonomy docstring, `20260924_114913_autocode_658b` carries #431's
split-item text, `20261008_142009_boardsteward_eec4` carries #2438's own acceptance
line. Nothing loads the rule on its own.

The fix is a POINTER, at the two surfaces a closing actor actually loads: one
trigger-bearing line in `skills/automod-change-own-code/SKILL.md`, and the schema path
inside the closure turn's `final_schema_prompt` (pinned in
`tests/test_backlog_unattended.py`). A pointer, deliberately — `#431` ruled the writer
Prose Rules must be INLINED rather than pointed at
(`tests/test_prose_rules_contract.py`), because a ban-list has to be held across every
sentence as it is written. This rule fires at one decision — does this closure warrant a
section? — so a trigger plus a path is executable there, and the section's placement,
citation and carry-across rules stay single-sourced in the schema, which is also what
keeps the two surfaces from drifting apart from a rule they cannot both own.

What is NOT pinned here, because no test can: whether a closing session that sees the
pointer then consults the entry. Zero qualifying closures had occurred when this round
was filed, so the payoff is measured after landing by owed-check, not by a node below.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

VAULT = Path.home() / "obsidian"
pytestmark = pytest.mark.skipif(not VAULT.is_dir(), reason="no vault on this box")

SKILL = VAULT / "skills/automod-change-own-code/SKILL.md"
SCHEMA = VAULT / "knowledge/KNOWLEDGE_SCHEMA.md"

#: The heading form, with its `##`. Distinct from the bare phrase the pointer carries:
#: a note that HAS a section carries the heading, and `knowledge/` is where grep -rl
#: finds one. The pointer must name the rule without posing as a note that has it.
HEADING = "## Measured in Lloyd"

#: The three notes that carry the heading today, per triage 2026-10-11. The two
#: carriers predate the convention that names them (`git log -S'## Measured in Lloyd'`:
#: `04277b23` 2026-10-03, `e8f7a0f5` 2026-10-07, convention `0050139c` 2026-10-08 15:41
#: — those two are the #1164 residue writes that motivated the entry), which is exactly
#: why "never retro-fit" is in it.
EXISTING_CARRIERS = {
    "knowledge/KNOWLEDGE_SCHEMA.md",
    ("knowledge/research/2026-09-24-self-question-retrieval-drafting-recall-queries-"
     "from-an-agen.md"),
    ("knowledge/youtube/AI_Engineer/20260915-tolan-voice-first-ai-companion-"
     "paula-dozsa-tolan.md"),
}

#: Operative sentences of the schema entry, quoted from its own text. A pointer states
#: WHEN the rule applies and WHERE it lives; carrying any of these would put a second
#: copy of a rule into a file that cannot maintain it.
NOT_IN_A_POINTER = (
    "open questions",                           # the placement rule's anchor heading
    "closing commit",                           # the citation rule's triple
    "never retro-fit",                          # the no-sweep rule
    "two measured ideas, two sections",         # the one-per-technique rule
)


def _text(path: Path) -> str:
    assert path.is_file(), f"{path} is not there; this pin has no subject"
    return path.read_text(encoding="utf-8")


def _pointer_lines(text: str) -> list[str]:
    """Every line of the skill that names the convention. Self-scoping, so the test
    cannot pass by pointing at some other paragraph."""
    return [ln for ln in text.splitlines() if "Measured in Lloyd" in ln]


def test_the_closing_skill_names_the_convention_and_where_it_lives():
    """Clause 1: a closing round reading `automod-change-own-code` meets the rule's
    name, its trigger, and the path that holds the rule.

    `grep -c 'Measured in Lloyd'` on this skill was 0 at filing, which is the whole
    premise: the surface the round is told to load never mentions the one thing a
    closure can owe a note.
    """
    lines = _pointer_lines(_text(SKILL))
    assert lines, (
        "the closure skill never mentions the `Measured in Lloyd` convention, so a "
        "round that closes a note-filed item is never told the rule exists")
    named = [ln for ln in lines if "knowledge/KNOWLEDGE_SCHEMA.md" in ln]
    assert named, (
        f"the skill names the convention without saying where the rule lives: {lines}")
    line = " ".join(named)
    # The trigger must be ONE sentence carrying BOTH halves and their conjunction.
    # Testing the tokens against the whole bullet was the hole here: the bold opener
    # says "A closing can owe…", so a trigger sentence naming only the artifact half
    # kept every separate token assertion green while telling a closer half a rule.
    trigger = [sent for sent in re.split(r"(?<=[.!?])\s+", line)
               if "clos" in sent.lower() and "dated measurement" in sent.lower()]
    assert trigger, (
        "no single sentence of the pointer states the whole trigger — that the item "
        "the note filed is closing AND that a dated measurement artifact exists for "
        f"it. The rule fires on the conjunction, so a pointer naming one half "
        f"authorises a section on one half: {line}")
    assert re.search(r"\bAND\b|\band\b", trigger[0]), (
        "the trigger sentence carries both halves but not as a conjunction, so a "
        f"closer cannot tell from it whether one alone is enough: {trigger[0]}")


def test_the_skill_line_is_a_pointer_and_not_a_second_copy_of_the_rule():
    """Clause 1's other half: it adds the trigger and the path, and NOTHING else.

    The entry is one rule with one owner. Duplicating its placement, citation or
    carry-across text into a skill would buy drift, and it would also quietly answer
    #431's ruling in the wrong direction — that ruling banned inlining for rules a
    writer must hold across every sentence, and the answer to it is that this rule
    fires at one decision, not that a second copy is fine.
    """
    line = " ".join(_pointer_lines(_text(SKILL)))
    assert line, "no pointer line exists, so there is nothing to check for a copy"
    lowered = line.lower()
    copied = [frag for frag in NOT_IN_A_POINTER if frag in lowered]
    assert copied == [], (
        f"the pointer carries the rule's own text ({copied}); the placement, "
        f"citation, carry-across and no-retro-fit rules stay in the schema alone")
    # And the single source really is single: the entry is still in the schema, once.
    assert _text(SCHEMA).count("Measured in Lloyd") == 1, (
        "the schema no longer holds exactly one copy of the convention, so 'the "
        "pointer points at the single source' is no longer true either")


def test_no_file_in_knowledge_gained_the_section_heading():
    """Clause 4: no retro-fit. Exactly the three notes that carry it today.

    The convention's own last bullet says an absent section is the correct state for an
    unmeasured idea, so a round that repoints a citation must not sweep notes to
    "helpfully" add one. This is a non-regression pin: it is true at the base and must
    stay true, and it is what would refuse a retro-fit landed alongside the pointer.
    """
    found = sorted(
        p.relative_to(VAULT).as_posix()
        for p in (VAULT / "knowledge").rglob("*.md")
        if HEADING in p.read_text(encoding="utf-8")
    )
    assert set(found) == EXISTING_CARRIERS, (
        f"the `## Measured in Lloyd` carrier set moved off the three notes that had "
        f"it at filing: {found}")
    assert len(found) == 3, found


def test_the_pointer_does_not_pose_as_a_note_that_has_a_section():
    """The skill gains the phrase, never the heading.

    `## Measured in Lloyd` is what a note that HAS the section carries, and
    `grep -rl` over `knowledge/` is how the convention is counted. The pointer line
    names the rule inside prose, so it cannot be mistaken for an instance of it — which
    is a property of MY OWN line, not of the vault: write `## Measured in Lloyd` into
    the skill and this node is the one that refuses.
    """
    skill = _text(SKILL)
    assert HEADING not in skill, (
        "the skill now carries the section heading itself, so any sweep that counts "
        "`## Measured in Lloyd` finds a 'measured' note where none exists")
    # …and no line of it claims one in prose either. The only reason this file may
    # name the section is to point elsewhere for it, so every mention has to name the
    # file holding the rule. "This skill carries a `Measured in Lloyd` section" passes
    # the heading check above and is still a false claim sitting in the file — which is
    # the half this node is named for and the heading check alone cannot see.
    claiming = [ln for ln in skill.splitlines()
                if "Measured in Lloyd" in ln
                and "knowledge/KNOWLEDGE_SCHEMA.md" not in ln]
    assert claiming == [], (
        "the skill mentions the section without pointing at the file that holds the "
        f"rule, which reads as the skill having one: {claiming}")
