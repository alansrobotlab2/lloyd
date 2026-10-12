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
import subprocess
from pathlib import Path

import pytest

VAULT = Path.home() / "obsidian"
pytestmark = pytest.mark.skipif(not VAULT.is_dir(), reason="no vault on this box")

SKILL = VAULT / "skills/automod-change-own-code/SKILL.md"
SCHEMA = VAULT / "knowledge/KNOWLEDGE_SCHEMA.md"

#: The heading form, with its `##`. Distinct from the bare phrase the pointer carries:
#: a note that HAS a section carries the heading as a line of its own, and `knowledge/`
#: is where grep -rl finds one. The pointer must name the rule without posing as a note
#: that has it.
HEADING = "## Measured in Lloyd"

#: A note CARRIES an instance of the section only when the heading is a line of its own.
#: The rule's owning file, `knowledge/KNOWLEDGE_SCHEMA.md`, holds the string too — as its
#: own heading `### `## Measured in Lloyd`` (that exact form is pinned by name in
#: tests/test_prose_rules_contract.py). Matching the substring made the single source read
#: as an instance of itself, and the 2026-10-11 triage's `grep -rl` sweep duly reported it
#: as one of "the three notes that carry the heading", so that carrier set was never the
#: set of notes that have a section.
INSTANCE = re.compile(r"^##\s+Measured in Lloyd\s*$")

#: The vault commit that recorded the convention: `0050139c`, 2026-10-08 15:41:07 -0700,
#: "Record the `## Measured in Lloyd` convention in KNOWLEDGE_SCHEMA.md (#2438)". A
#: heading whose write predates it is #1164 residue — the writes the entry exists to
#: describe — so the exemption is a date cut taken from the note's own history, not a
#: name that has to be kept current by hand.
CONVENTION_COMMIT = "0050139c"

#: The two notes whose heading arrived before that commit, per
#: test_the_residue_carriers_are_the_two_pre_convention_writes: the Tolan source note
#: (`04277b23`, 2026-10-03 22:16) and the self-question research note (`e8f7a0f5`,
#: 2026-10-07 01:47).
RESIDUE_CARRIERS = {
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


def _vault_git(*args: str) -> str:
    """One git call against the live vault.

    The vault is the corpus these pins read and its history is what decides a section's
    origin, so a git failure is reported rather than absorbed — `_retro_fit` refuses on an
    unknown origin, which is the loud outcome. `subprocess` over `git -C ~/obsidian` is
    how tests/test_prose_rules_contract.py already reads the same tree, and under the gate
    `~/obsidian` is symlinked to the real vault (`worktree.ensure_round_home`), so this
    resolves there exactly as it does here.
    """
    proc = subprocess.run(("git", "-C", str(VAULT), *args),
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"`git -C ~/obsidian {' '.join(args)}` exited {proc.returncode}: "
        f"{proc.stderr.strip()[:200]!r}. The no-retro-fit pin is decided by vault "
        "provenance, so it cannot be answered without git.")
    return proc.stdout.strip()


def _carriers() -> list[Path]:
    """Every note under `knowledge/` carrying an instance of the section."""
    return sorted(p for p in (VAULT / "knowledge").rglob("*.md")
                  if any(INSTANCE.match(ln) for ln in _text(p).splitlines()))


def _rel(path: Path) -> str:
    return path.relative_to(VAULT).as_posix()


def _section(text: str) -> str:
    """The section's own text: its heading line through to the next `## ` heading."""
    keep, out = False, []
    for line in text.splitlines():
        if keep and line.startswith("## "):
            break
        keep = keep or bool(INSTANCE.match(line))
        if keep:
            out.append(line)
    return "\n".join(out)


def _heading_write(rel_path: str) -> tuple[str, int, str]:
    """(short sha, committer unix time, full message) of the most recent vault commit
    that changed how many times `## Measured in Lloyd` appears in that note — in practice,
    the write that put the section there. `git log -S` fires on a change of count, so an
    edit that rewords the body and leaves the heading alone does not move it.
    """
    line = _vault_git("log", "-1", "-S", HEADING, "--format=%h%x1f%ct%x1f%B",
                      "--", rel_path)
    assert line, f"no vault commit ever changed a `## Measured in Lloyd` count in {rel_path}"
    sha, _, rest = line.partition("\x1f")
    ts_txt, _, message = rest.partition("\x1f")
    return sha, int(ts_txt), message


def _convention_ts() -> int:
    """Committer time of the convention's own commit, as a unix timestamp."""
    return int(_vault_git("log", "-1", "--format=%ct", CONVENTION_COMMIT))


def _retro_fit(rel_path: str, text: str) -> str | None:
    """Why this carrier is not a legitimate write, or None when it is one.

    Two origins are legitimate, and the schema names both. Before `0050139c` the rule did
    not exist, so a heading written earlier is the #1164 residue the entry was written to
    describe. After it the Trigger is "the backlog item the note filed closes", and such a
    write is recognisable by exactly what the Trigger says: the commit that wrote the
    heading names an item, and the section cites that same item as the one it closed.

    A maintenance sweep is the refused case. It has no closing item whose number belongs
    in the note, so its commit names the round that ran it while the section names the
    idea it "helpfully" marked, and the two sets of numbers do not intersect.
    """
    sha, ts, message = _heading_write(rel_path)
    if ts < _convention_ts():
        return None
    written = set(re.findall(r"#(\d+)", message))
    cited = set(re.findall(r"#(\d+)", _section(text)))
    if written & cited:
        return None
    return (f"{rel_path}: its heading arrived in {sha} "
            f"({message.splitlines()[0][:72]!r}), whose message names "
            f"{sorted(written) or 'no backlog item'} while the section cites "
            f"{sorted(cited) or 'nothing'}. A `## Measured in Lloyd` written into a note "
            "by work other than the closure of the item the section records is the "
            "retro-fit the convention forbids. To land one legitimately, cite the closing "
            "item's number in the section and name that same number in the vault commit.")


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
    """Clause 4: no retro-fit. A section may arrive only with the work that measured the
    idea, never from a sweep over notes that already exist.

    The convention's own last bullet says an absent section is the correct state for an
    unmeasured idea, so a round that repoints a citation must not sweep notes to
    "helpfully" add one. This is the pin that refuses it.

    As filed it was a three-name snapshot of `knowledge/`, taken at triage 2026-10-11, and
    the snapshot rather than the rule is what went red at base `8525c8ab`, where the sweep
    returned four paths. One of them, `knowledge/KNOWLEDGE_SCHEMA.md`, is the rule's
    DEFINITION (`### `## Measured in Lloyd``) and has never carried an instance of the
    section — `INSTANCE` is here to stop counting it. The fourth,
    `knowledge/thinking-effort-per-job-class-2026-10-11.md`, gained the section on
    2026-10-11 in vault commit `cd565534` ("#2576: report the class table, the two-run
    ladder and the prefix-cache price"), 71 minutes after `a3bf00c4` created the note from
    the same item's measurement work: the closure of #2576 answering the pointer #2554
    planted, which is the outcome the convention exists to produce, not the sweep this node
    refuses.

    A hand-kept name-list cannot make that distinction, and it reddens the suite at every
    further qualifying closure until someone re-cuts the list — which is how this item came
    to exist. So the pin now decides origin, the thing the list was standing in for:
    `_retro_fit` exempts a heading written before the convention commit `0050139c` and
    requires every heading written after it to arrive in a commit whose message names the
    item the section cites. `test_the_residue_carriers_are_the_two_pre_convention_writes`
    holds the history the exemption rests on.
    """
    offenders = [msg for path in _carriers()
                 if (msg := _retro_fit(_rel(path), _text(path)))]
    assert offenders == [], "\n".join(offenders)


def test_the_residue_carriers_are_the_two_pre_convention_writes():
    """The exemption's factual basis is fixed history, and this is what notices if it
    stops being fixed.

    `0050139c` recorded the convention on 2026-10-08 15:41, and exactly two notes already
    had a section: the Tolan source note (`04277b23`, 2026-10-03 22:16, "#2159 (#1164
    residue): record the measured rejection in the Tolan source note") and the
    self-question research note (`e8f7a0f5`, 2026-10-07 01:47, "#2336 Mark the measured
    rejection in the self-question research note"). Those are the writes the entry names as
    its own motivation for "never retro-fit".

    Vault history only grows at the end, so a third member here is not a normal event: it
    means the convention's own commit moved, or a heading was committed with a backdated
    date — and either way the date cut in `_retro_fit` is no longer the exemption it claims
    to be, which is the thing worth a red test.
    """
    residue = {_rel(p) for p in _carriers()
               if _heading_write(_rel(p))[1] < _convention_ts()}
    assert residue == RESIDUE_CARRIERS, (
        "the pre-convention carriers moved off the two #1164 residue writes, so the date "
        f"cut in `_retro_fit` exempts something else now: {sorted(residue)}")


def test_a_heading_written_by_a_different_item_s_closure_is_refused():
    """The negative control, on real provenance: the origin rule does refuse.

    `knowledge/thinking-effort-per-job-class-2026-10-11.md` is the one carrier whose
    heading postdates `0050139c`; `cd565534` wrote it for the closure of #2576, and its own
    text passes. Give that same real commit provenance a section citing a different item —
    precisely what a sweep over an existing note leaves behind — and the rule names the
    mismatch instead of passing. The note's bytes are only read, never written: the synthetic
    body lives in this call, not in the vault.
    """
    rel = "knowledge/thinking-effort-per-job-class-2026-10-11.md"
    assert _retro_fit(rel, _text(VAULT / rel)) is None, (
        "the one closure-attributed carrier stopped passing, so a refusal below would say "
        "nothing about the mismatch it is meant to catch")
    refused = _retro_fit(rel, f"{HEADING}\n\nCloses #1234, measured 2026-10-11.\n")
    assert refused, (
        "a section whose cited item is not the item whose commit wrote it was accepted, so "
        "the retro-fit pin cannot fail and guards nothing")
    assert ("1234" in refused and "2576" in refused), refused


def test_the_pointer_does_not_pose_as_a_note_that_has_a_section():
    """The skill gains the phrase, never the heading.

    `## Measured in Lloyd` on a line of its own — `INSTANCE` — is what a note that HAS the
    section carries, and a `grep -rl` sweep over `knowledge/` is how they get counted (a
    substring sweep also reports the schema, which defines the heading rather than
    carrying one). The pointer line
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
