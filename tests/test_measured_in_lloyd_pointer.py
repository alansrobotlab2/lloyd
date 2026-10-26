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
import sys
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


# ── #3367: a FIRST-EVER section, which has no heading commit until the land makes one ──
#
# Everything above reads the live vault, where all three carriers already have a heading
# commit behind them (the newest, `cd565534`, predates this file's own `50bba19b`, which
# is why the shape below was never exercised). #3366 was refused by a draw over a proposal
# whose `## Measured in Lloyd` was real, current and unciteable — verbatim at
# `~/.local/state/lloyd-automod/promotions.jsonl`, row `"created_at": "2026-10-26T07:59:25Z"`,
# `"event": "vault_land"`, `"ok": false`, `"item_id": 3366`:
#
#   tests/test_measured_in_lloyd_pointer.py::test_no_file_in_knowledge_gained_the_section_heading:
#   fails against the vault as proposed (every one of 2 draws) and passes against the vault
#   before this land (every one of 2 draws), in the default selection at
#   /home/alansrobotlab/lloyd@89e1904a
#
# The three nodes below re-create exactly that draw: a real git vault, a tracked note with
# NO heading in its history, the section in its working tree. They write only into a temp
# vault, never into this one.

CITED_SECTION = (f"{HEADING}\n\n"
                 "Closes #3366, measured 2026-10-26.\n"
                 "WER over 202 utterances: 11.5 % (baseline 15.3 %).\n")
UNCITED_SECTION = (f"{HEADING}\n\n"
                   "Measured 2026-10-26 over 202 utterances: 11.5 % (baseline 15.3 %).\n")
LAND_MESSAGE = "#3366 add the Lloyd measurement to the jesse-hall live note"


def _first_ever_note(tmp_path, monkeypatch):
    """A git vault holding one tracked note whose history contains no `## Measured in Lloyd`.

    Stands for the digest note #3366 wanted to section: `git log -- <path>` is non-empty, so
    the note is not new, and `git log -S"## Measured in Lloyd" -- <path>` is empty, so no
    commit has ever moved the heading count. `_convention_ts` is pinned to 0 because
    `0050139c` does not exist in a temp vault, and pinning it to 0 states the case: every
    commit here postdates the convention, so nothing is exempt as pre-convention residue and
    a legitimacy verdict can only come from the section's own citation.
    """
    root = tmp_path / "vault"
    (root / "knowledge" / "research").mkdir(parents=True)
    subprocess.run(("git", "init", "-q", "-b", "main", str(root)), check=True,
                   capture_output=True)
    for key, val in (("user.email", "t@e.com"), ("user.name", "t")):
        subprocess.run(("git", "-C", str(root), "config", key, val), check=True,
                       capture_output=True)
    note = root / "knowledge" / "research" / "2026-10-26-note.md"
    note.write_text("# A note the digest worker wrote\n\nNo section yet.\n")
    subprocess.run(("git", "-C", str(root), "add", "-A"), check=True, capture_output=True)
    subprocess.run(("git", "-C", str(root), "commit", "-q", "-m", "note without a section"),
                   check=True, capture_output=True)
    # `sys.modules[__name__]`, not the dotted name: pytest imports a test file by PATH as
    # `test_measured_in_lloyd_pointer`, so importing `tests.test_measured_in_lloyd_pointer`
    # would hand back a second module object and the patch would land on globals this node's
    # own `_heading_write` never reads — which is what a green-looking fixture with a red
    # assert on the live vault looks like.
    mine = sys.modules[__name__]
    monkeypatch.setattr(mine, "VAULT", root)
    monkeypatch.setattr(mine, "_convention_ts", lambda: 0)
    rel = "knowledge/research/2026-10-26-note.md"
    assert not _vault_git("log", "-1", "-S", HEADING, "--format=%h", "--", rel).strip(), (
        "fixture: this note must have no heading in its history, or the case below is a "
        "second-section case and proves nothing about a first one")
    return root, rel


def _commit(root: Path, message: str, paths: list[str]) -> list[str]:
    """Stage exactly `paths`, commit them under `message`, and return that commit's files."""
    for p in paths:
        subprocess.run(("git", "-C", str(root), "add", "--", p), check=True,
                       capture_output=True)
    subprocess.run(("git", "-C", str(root), "commit", "-q", "-m", message), check=True,
                   capture_output=True)
    listed = subprocess.run(("git", "-C", str(root), "show", "--pretty=format:",
                             "--name-only", "HEAD"),
                            capture_output=True, text=True, check=True).stdout
    return sorted(line for line in listed.splitlines() if line.strip())


def test_a_first_ever_section_is_legitimate_once_the_land_commits_its_own_message(tmp_path,
                                                                                  monkeypatch):
    """Clause 2: the section that CANNOT be refused for provenance is the one that can be landed.

    #3367's defect is that `_heading_write` demands the vault commit which introduced the
    heading while the code-agreement probe runs BEFORE the route makes any commit
    (`agreement()` at `scripts/automod/vault_round.py:1874`, the commit at :1980), so a
    first-ever section has no such commit to find and the assert at :157 fires — every
    proposed draw red, every pre-land draw green. The three steps below are that sequence,
    in order, and each one is the reason the next is worth asserting.

    1. the section in the working tree with no commit behind it — the refused draw, red;
    2. `VG.commit_proposal`'s mirror commit: the declared paths, the land's own message,
       nothing else in it — which is what the proposed mirror now carries;
    3. `_retro_fit` on the result: legitimate, because the message names #3366 and the
       section cites #3366, so `_retro_fit`'s `written & cited` is non-empty.

    The residue exemption is checked too, because the clause names both nodes and only its
    second one asks about residue: the commit is today's, so it does NOT date under
    `0050139c`, and this section earns its place by citing rather than by being old.
    """
    root, rel = _first_ever_note(tmp_path, monkeypatch)
    (root / rel).write_text(CITED_SECTION)

    with pytest.raises(AssertionError, match="no vault commit ever changed"):
        _heading_write(rel)

    from scripts.automod import vault_guards as VG
    staged = VG.commit_proposal(root, [rel], LAND_MESSAGE)
    assert staged["ok"] is True, staged["reason"]
    assert _commit_files(root) == [rel], (
        "the mirror commit must carry exactly the paths the land declared, or a proposal "
        "would be judged on somebody else's bytes")

    sha, ts, message = _heading_write(rel)
    assert message.strip() == LAND_MESSAGE.strip(), (
        f"the commit the guard reads is not the land's message: {message!r}")
    assert ts >= _convention_ts(), (
        f"a commit made today ({sha}) would be exempted as pre-convention residue, so this "
        "node would be passing on the wrong rule")
    assert _retro_fit(rel, CITED_SECTION) is None, (
        "a first-ever section citing the item the land's message names was still refused, "
        "so the fix did not reach the case #3366 was refused for")


def _commit_files(root: Path) -> list[str]:
    listed = subprocess.run(("git", "-C", str(root), "show", "--pretty=format:",
                             "--name-only", "HEAD"),
                            capture_output=True, text=True, check=True).stdout
    return sorted(line for line in listed.splitlines() if line.strip())


def test_a_first_ever_section_citing_nothing_is_refused_by_its_own_text(tmp_path, monkeypatch):
    """Clause 3: the new path cannot launder a maintenance sweep, because the refusal it
    produces is about the SECTION and not about a missing commit.

    The pre-fix failure mode and the refused-forever failure mode both read as "the node is
    red", so they are distinguished here: this commit EXISTS, `_heading_write` is silent, and
    `_retro_fit` still refuses — which can only be the citation intersection,
    `written & cited`, coming out empty. The message names #3366, exactly as an honest land's
    does; `written = {"3366"}` and `cited = {}` do not intersect, and the text the sweep would
    have to write to pass is a number it does not have: that is what keeps the sweep refused
    now that a first-ever section can be landed.
    """
    root, rel = _first_ever_note(tmp_path, monkeypatch)
    (root / rel).write_text(UNCITED_SECTION)

    from scripts.automod import vault_guards as VG
    assert VG.commit_proposal(root, [rel], LAND_MESSAGE)["ok"] is True
    sha, ts, message = _heading_write(rel)   # the commit exists: nothing missing to blame
    assert sha and message.strip() == LAND_MESSAGE.strip(), (
        f"the guard is not reading the mirror commit: {sha!r} {message!r}")

    refused = _retro_fit(rel, UNCITED_SECTION)
    assert refused, (
        "a `## Measured in Lloyd` citing no item was accepted once a commit carried it, so "
        "the route that let a first-ever section through also let a sweep through")
    assert "3366" in refused and "nothing" in refused, refused


def test_a_first_ever_section_lands_refused_when_the_message_names_no_item(tmp_path,
                                                                          monkeypatch):
    """Clause 3's other half: an EMPTY or item-less land message cannot excuse a section
    either, which is why `message` defaults to "" and why the keyword can be passed without
    a lever.

    `commit_proposal` labels a nameless commit `unlabelled`, and an unlabelled commit names
    no item, so `written` is empty for every proposal in a vault whose land passed no message
    — every section would be refused. That is the safe direction (a probe with no message can
    cost a landing, never grant one), and it is pinned here so a later change cannot make the
    fallback permissive by treating an empty message as a wildcard.
    """
    root, rel = _first_ever_note(tmp_path, monkeypatch)
    (root / rel).write_text(CITED_SECTION)     # cites #3366, as an honest section does

    from scripts.automod import vault_guards as VG
    assert VG.commit_proposal(root, [rel], "nightly: refresh the digest notes")["ok"] is True
    sha, ts, message = _heading_write(rel)
    # The commit carries the land's prose verbatim and names no item in it, which is the
    # state under test. (A land that passed no message at all is labelled `unlabelled` by
    # `commit_proposal`; either way `written` comes out empty.)
    assert sha and message.strip() == "nightly: refresh the digest notes", message

    refused = _retro_fit(rel, CITED_SECTION)
    assert refused, (
        "a commit message naming no backlog item was treated as naming whatever the section "
        "cited, which would let any sweep through by passing no message at all")
    assert "no backlog item" in refused, refused
