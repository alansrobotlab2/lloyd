"""Backlog #436 — nightly reflection reports must be archived before they are
overwritten, and the archive name must not come from a hand-written stamp.

Why this is a code problem and not a skill problem
--------------------------------------------------
The nightly chain writes three live reports in place under
``~/lloyd/_pipeline/reflection/``: ``signals-latest.md``,
``tool-patterns-latest.md`` and ``conversation-patterns-latest.md``. That
directory is gitignored (``~/lloyd/.gitignore:25`` ``/_pipeline/``;
``git ls-files _pipeline | wc -l`` → 0), so an overwrite with no prior copy
destroys the only version that has ever existed — the same irreversibility that
made the 2026-08-22 entity-graph destruction unrecoverable.

Measured at triage (2026-09-12T10:15Z), against ~22 nightly cycles since 08-22:
``signals-latest*`` → 7 files (6 dated copies), and **zero** dated copies for
either pattern file. So the retention rule was enforced by nothing.

The first fix (vault commit ``3514dac1``, 2026-09-12) added an archive ``cp`` to
one skill and only for ``signals-latest.md``. Prose one model reads once per run
is what the remaining cycles were still relying on, and #436's own merged finding
says so in the skill's own words: "A Read-before-Write discipline that preserves
a file only while the reader remembers to copy it is not a retention policy."
This file is the instrument: the archive step is now pinned, and deleting it is
refused at the writer that would land the change.

Where the rule lives, and where it runs
--------------------------------------
The rule itself is ``scripts/reflection_archive.py`` — one definition, two consumers.
``scripts/automod/vault_round.py`` calls it on every ``skills/**/SKILL.md`` a
vault round touches, so dropping an archive step is refused at
``automod_vault_land`` before the commit and the round's paths are reverted.
**That is the automated rung this rule is enforced at**, and
``test_the_vault_writer_refuses_a_skill_that_dropped_its_archive_step`` below
drives ``validate`` end to end to prove the wiring, not just the helper.

The assertions in this file that read ``~/obsidian/skills`` carry
``live_vault`` (``pytest.ini:6-12``): skill prose is state no round under test
controls, and the gate's hard ``tests`` rung runs ``-m "not live_vault"`` so an
autoresearch promotion or a nightly rewrite of some unrelated skill cannot fail
the next author's round for the previous writer's wording. Being honest about the
rest: the full suite *command* runs the marked group, and no job in the fleet runs
the full suite command — they name individual files — so those assertions are
executed by hand and by the gate's review rung, and backlog #979 owns giving
``live_vault`` a rung of its own. That is why the enforcement is at the writer and
not here; it is the architecture ``pytest.ini`` already states for the identity
surface ("Enforcement lives at the writers … this mark is the reporting copy"),
and ``reflection_archive`` exists so this rule can follow it. The unmarked tests
here — every mutation that proves the rule can fail, the writer wiring, and the
git-ignore pin — run on every rung.

What each clause is pinned by
-----------------------------
1. *Every live report gets a dated copy before its overwrite, on every cycle* →
   ``test_every_skill_that_writes_a_reflection_report_archives_it_first``, over
   the writer set discovered by ``classify_all``. Clause 1's specific
   requirement — §2e of ``nightly-reflection-knowledge-write`` for both pattern
   files — is asserted separately in
   ``test_knowledge_write_section_2e_archives_both_pattern_files``.
2. *A test in the ``tests/test_skill_*.py`` family, failing when such a line is
   removed* → the same test, plus ``test_the_archive_check_can_actually_fail``,
   which mutates synthetic skill bodies and asserts the check reports each
   removal class, and the two writer-wiring tests that show a removal is
   *refused* and not merely reported. A pin that cannot fail is not a pin.
3. *The stamp must be UTC-derived, never a `generated:` field read out of the
   previous report* → ``test_archive_stamp_is_derived_from_a_utc_command`` (the
   destination is ``$STAMP``/``$(date …)`` and that variable is assigned from
   ``date -u … +%Y-%m-%d-%H%M`` on the same logical line) and
   ``test_no_skill_builds_an_archive_name_from_a_generated_field``.
4. *No gitignored file under ``~/lloyd`` is touched* → the diff is this file, one
   module and one validator hunk, all tracked paths, and
   ``test_pipeline_stays_gitignored_so_retention_is_copies_not_tracking`` fails if
   ``_pipeline/`` ever stops being ignored — which is the other route #436 names
   and deliberately does not take (it is Alan's decision, not a side effect).

How writers are recognised, and why the carve-outs are honest
-------------------------------------------------------------
A *writer* is a skill whose text instructs a write to the live report. Three
tiers, discovered from the same classifier, and asserted to equal the sets
recorded below — a new writer in **any** tier fails the suite rather than being
silently uncovered, and any mention that is neither a write nor a read lands in
``unclassified``, which must be empty, so an under-inclusive rule cannot hide a
writer.

Tier 1 is the tier the archive rule binds: a real ``Write``/``Edit`` call on the
path. Tier 2 is a write prescription that cannot land or cannot be machine-bound:
either routed through ``vault_write``/``mem_write`` — which reject any ``~/lloyd/``
target with ``PATH_ESCAPE`` (recorded in ``nightly-reflection-signals`` Phase 0,
``nightly-reflection-knowledge-analysis`` step 1, and MEMORY.md) — or stated as
prose ("Merge all ``signals-*.md`` into …"). Tier-2 entries carry a reason and are
re-verified every run: if one ever gains a real ``Write`` call it moves to tier 1
and must have an archive step. Both tier-2 families are recorded as findings on
#436 rather than widened into this diff.

The attribution boundary, stated plainly: a mention counts when it names the path
(``reflection/<name>-latest.md``). A skill that refers to ``test-results-latest.md``
by bare filename somewhere else is not attributed, because a bare filename cannot
be told apart from one in a different directory. That is the reason
``PROSE_WRITE`` carries verbs of instruction and not nouns like "output" — the
nouns are what descriptive annotations use, and three of the read-list items in
``autonomy-reflection-pipeline`` Step 6.1 are annotated that way.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import reflection_archive as ra
from scripts.reflection_archive import archive_problems, classify, classify_all

SKILLS_DIRS = [Path.home() / "obsidian" / "skills", ROOT / "skills"]


def _active_skills() -> dict[str, str]:
    """Every SKILL.md the prompt actually advertises → its body.

    Mirrors prompt_builder._load_skills_index: dot-prefixed directories are the
    archive, quarantined skills are out of circulation. The quarantine filter
    lives here and not in ``reflection_archive`` because the vault writer is
    handed the one path a round touched and must judge that file whether or not
    the index advertises it — a skill being un-listed is not a licence to lose a
    report.
    """
    from prompt_builder import _is_quarantined_skill

    out: dict[str, str] = {}
    for root in SKILLS_DIRS:
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in out:
                continue
            skill_file = entry / "SKILL.md"
            if skill_file.exists() and not _is_quarantined_skill(skill_file):
                out[entry.name] = skill_file.read_text(encoding="utf-8", errors="replace")
    return out


@pytest.fixture(scope="module")
def skills() -> dict[str, str]:
    """Active skill bodies. Every test using this reads state no round controls, so
    those tests carry `live_vault`; an unreadable skill tree is a failure, not a skip
    — a green run that read nothing is exactly the vacuity this file argues against."""
    bodies = _active_skills()
    assert bodies, (
        f"no SKILL.md found under {SKILLS_DIRS} — the skills tree is what this check "
        "reads, so a run that found none proves nothing and must not pass"
    )
    return bodies


# --- Tier 1: writers that the archive rule binds. -------------------------------
# Discovered and asserted as a set, so a new tool-writer of any reflection report
# has to be looked at rather than quietly inheriting "no archive step".
EXPECTED_TOOL_WRITERS: dict[str, set[str]] = {
    "nightly-reflection-signals": {"signals-latest"},
    "nightly-reflection-knowledge-write": {
        "tool-patterns-latest",
        "conversation-patterns-latest",
    },
}

# --- Tier 2: write prescriptions that are not machine-bindable, and why. --------
# Each entry is re-verified to still be true by
# test_declared_write_prescriptions_are_still_accurate, and the discovered tier-2
# set must equal this set. If a skill here gains a real Write call it moves to
# tier 1 and owes an archive step; both families are recorded as findings on #436.
#: Write prescriptions routed through vault_write / mem_write, which reject any
#: ~/lloyd/ target with PATH_ESCAPE — the instruction cannot destroy a report, so
#: the absence of an archive step costs nothing today.
#: (`autonomy-reflection-pipeline` names prompt-audit-latest and test-results-latest
#: only inside Step 6.1's *read* list, so they classify as reads; it prescribes
#: producing them by bare filename, which no path-shaped check can attribute.)
EXPECTED_VAULT_WRITE_WRITERS: dict[str, set[str]] = {
    "autonomy-reflection-pipeline": {"signals-latest", "day-end-synthesis-latest"},
    "nightly-prompt-audit": {"prompt-audit-latest"},
    "nightly-behavior-test": {"test-results-latest"},
    "nightly-day-end-synthesis": {"day-end-synthesis-latest"},
}

#: The disclosed debt, checked family by family by
#: test_the_prose_writer_carve_out_is_debt_recorded_on_436. These skills really do
#: instruct overwriting the live reports and really do have no archive step; this
#: check cannot bind them because it binds instructions a run executes, and prose is
#: not one. Recorded on #436 (round SM_20260912_155333 findings) as the item's scope,
#: not this test's exemption: the set is asserted equal, so it can only shrink.
EXPECTED_PROSE_WRITERS: dict[str, set[str]] = {
    "historical-knowledge-refresh": {
        "signals-latest",
        "tool-patterns-latest",
        "conversation-patterns-latest",
    },
}

EXPECTED_UNARCHIVED_WRITERS: dict[str, set[str]] = {
    name: set(stems) for name, stems in EXPECTED_VAULT_WRITE_WRITERS.items()
}
for _name, _stems in EXPECTED_PROSE_WRITERS.items():
    EXPECTED_UNARCHIVED_WRITERS.setdefault(_name, set()).update(_stems)


# --- Clause 1 + 2 ---------------------------------------------------------------
@pytest.mark.live_vault
def test_every_skill_that_writes_a_reflection_report_archives_it_first(skills):
    """The acceptance clause: every live nightly reflection report gets a dated
    copy before it is overwritten, on every cycle, pinned here rather than by the
    skill prose alone."""
    offenders = {
        f"{name}:{stem}": problems
        for (name, stem), tier in classify_all(skills).items()
        if tier in ra.BOUND_TIERS
        for problems in [archive_problems(skills[name], stem)]
        if problems
    }
    assert offenders == {}, (
        "skills write a nightly reflection report in place without an archive "
        f"step for that same path: {offenders}. `_pipeline/` is gitignored, so "
        "the overwritten report is unrecoverable (#436)."
    )


@pytest.mark.live_vault
def test_discovered_tool_writers_are_exactly_the_expected_set(skills):
    """Anti-drift for the classifier itself: a new tool-writer, or one that
    disappears, is a decision someone has to record here."""
    found: dict[str, set[str]] = {}
    for (name, stem), tier in classify_all(skills).items():
        if tier in ra.BOUND_TIERS:
            found.setdefault(name, set()).add(stem)
    assert found == EXPECTED_TOOL_WRITERS, (
        f"reflection-report writers changed: found {found}, expected "
        f"{EXPECTED_TOOL_WRITERS}. Every entry must archive before overwriting."
    )


@pytest.mark.live_vault
def test_knowledge_write_section_2e_archives_both_pattern_files(skills):
    """Clause 1 by name: §2e of nightly-reflection-knowledge-write, for each of
    the two pattern files — Read the existing file, copy it to a dated path, then
    write. This is the half that had never landed: zero dated copies of either
    pattern file across ~22 nightly cycles."""
    body = skills["nightly-reflection-knowledge-write"]
    section = body.split("### 2e.", 1)
    assert len(section) == 2, "nightly-reflection-knowledge-write lost its §2e"
    section = section[1].split("\n### ", 1)[0]
    for stem in ("tool-patterns-latest", "conversation-patterns-latest"):
        assert f"_pipeline/reflection/{stem}.md" in section, (
            f"§2e no longer writes {stem}.md"
        )
        assert not archive_problems(section, stem), (
            f"§2e must Read and dated-copy {stem}.md before overwriting it: "
            f"{archive_problems(section, stem)}"
        )
        # Per file, not per section: clause 1 says "for each of", and one Read
        # anywhere in §2e would satisfy a section-wide search while leaving the
        # other pattern file's overwrite unread.
        assert re.search(
            r"(?<![A-Za-z_])\bRead\s*\(\s*[\"'`][^\"'`]*" + re.escape(stem) + r"\.md", section
        ), (
            f"§2e must Read {stem}.md itself: Write refuses a file it has not read "
            "in the session, so the instruction has to name this file"
        )


# --- Clause 3 ------------------------------------------------------------------
@pytest.mark.live_vault
def test_archive_stamp_is_derived_from_a_utc_command(skills):
    """The archive name must come from `date -u … +%Y-%m-%d-%H%M` (or the source's
    mtime) — never a hand-typed date."""
    problems: dict[str, list[str]] = {}
    for (name, stem), tier in classify_all(skills).items():
        if tier not in ra.BOUND_TIERS:
            continue
        bad = [p for p in archive_problems(skills[name], stem) if "UTC" in p]
        if bad:
            problems[f"{name}:{stem}"] = bad
    assert problems == {}, f"archive stamps are not UTC-derived: {problems}"


def _sentences_mentioning(text: str, token: str) -> list[str]:
    """Sentences containing `token`, on text with blockquote markers stripped.

    Sentence boundaries are rough by design (`.md` contains a period), which is
    safe here: the rule only ever applies to a sentence that also names a
    filename, and a fragment that loses the negation loses the `.md` with it.
    """
    flat = re.sub(r"^\s*> ?", "", text, flags=re.MULTILINE)
    flat = re.sub(r"\s+", " ", flat)
    return [s for s in re.split(r"(?<=[.!?:])\s+", flat) if token in s]


def _generated_field_offenders(text: str) -> list[str]:
    """Sentences that instruct naming a file from a `generated:` header.

    A negation is the difference between a rule and a prohibition on the rule: the
    skills now spend several lines explaining why the old wording was wrong, and
    those sentences name both the field and a filename.
    """
    return [
        s for s in _sentences_mentioning(text, "generated:")
        if ".md" in s and not re.search(r"\b(?:never|not|no longer|isn't|cannot)\b", s, re.I)
    ]


@pytest.mark.live_vault
def test_no_skill_builds_an_archive_name_from_a_generated_field(skills):
    """Clause 3's negative half. The report's own `generated:` header is a local
    (PST) clock reading, so a copy named from it sorts 8 hours out from the
    UTC-named siblings in the same directory and cannot be re-derived when the
    field is missing. A skill may *say* this — with a negation — but must not
    instruct it."""
    offenders: dict[str, list[str]] = {}
    for (name, stem), tier in classify_all(skills).items():
        if tier not in ra.BOUND_TIERS:
            continue
        bad = _generated_field_offenders(skills[name])
        if bad:
            offenders[name] = bad
    assert offenders == {}, (
        f"skills derive an archive filename from a report's generated: field: "
        f"{offenders}. Use `date -u +%Y-%m-%d-%H%M` (#436 clause 3)."
    )


# --- Clause 4 ------------------------------------------------------------------
def test_pipeline_stays_gitignored_so_retention_is_copies_not_tracking():
    """Clause 4: retention is achieved by dated copies (or, later, by a vault
    relocation Alan decides), not by un-ignoring `_pipeline/` and committing
    generated reports into the repo. If someone un-ignores it, the choice has
    changed and this pin — plus #436's open question for Alan — has to be
    re-decided, not silently implemented from a test."""
    probe = "_pipeline/reflection/signals-latest.md"
    # No skip here: this asserts repo state the round does control, and a run that
    # cannot ask git is a failure worth seeing, not a green line.
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", probe],
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"`{probe}` is no longer gitignored. #436 keeps the nightly reports out of "
        "the repo and archives them as dated copies; committing generated reports "
        "into ~/lloyd is Alan's decision, not a side effect of this test file."
    )


# --- The enforcement point: the vault writer, not a test rung -------------------
_SKILL_FRONT = "---\nname: test-skill\ndescription: A skill that overwrites a report\n---\n\n"
_STEM = "tool-patterns-latest"
_LIVE = f"/home/alansrobotlab/lloyd/_pipeline/reflection/{_STEM}.md"
# Written the way the skills write it: one instruction wrapped over three lines
# with backslash continuations, so this fixture also exercises logical_lines.
_READ_LINE = f'Read("{_LIVE}")'
_COPY_LINE = (
    'Bash("STAMP=$(date -u +%Y-%m-%d-%H%M) && \\\n'
    f"      cp {_LIVE} \\\n"
    f"         {_LIVE[:-3]}-$STAMP.md\")"
)
_WRITE_LINE = f'- `Write(file_path="{_LIVE}", content=…)` — new content'
_GOOD_BODY = _SKILL_FRONT + "\n".join(
    ["### 2e. Pattern Output Files\n", _READ_LINE, _COPY_LINE, "", _WRITE_LINE, ""]
)
_BAD_BODY = _SKILL_FRONT + "\n".join(
    ["### 2e. Pattern Output Files\n", _READ_LINE, "", _WRITE_LINE, ""]
)


@pytest.fixture
def scratch_vault(tmp_path, monkeypatch):
    """A vault tree the test owns, plus the loader subprocess stubbed out.

    `skills/**` is a *validated* vault path, so `validate` would otherwise shell
    out to `agent_mcp.skills._load_skill` against this scratch tree — which is a
    real check of a different question (does the skill load) and belongs to
    tests/test_automod_vault_round.py. Stubbing it keeps this assertion about the
    retention branch and nothing else, the same way every case in that file
    stubs it.
    """
    from scripts.automod import vault_round

    monkeypatch.setattr(vault_round, "VAULT", tmp_path)
    monkeypatch.setattr(vault_round, "loader_errors", lambda paths: [])
    return tmp_path


def _write_skill(vault: Path, name: str, body: str) -> str:
    rel = f"skills/{name}/SKILL.md"
    target = vault / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return rel


def test_the_vault_writer_refuses_a_skill_that_dropped_its_archive_step(scratch_vault):
    """The wiring, not the helper: `validate()` — the function `land()` and
    therefore `automod_vault_land` actually call — must reach the retention branch.

    Calling `reflection_archive.skill_rule_violations` directly pins the rule; it
    does not pin the call site, and a correct helper that nothing calls is the
    exact shape of "a guard whose input nothing wired up" that this item exists to
    close. So this drives the lander: a skill edit that loses the archive `cp`
    comes back refused, naming the report and the missing step, while the same
    skill with its `cp` intact lands.
    """
    from scripts.automod import vault_round

    bad = _write_skill(scratch_vault, "nightly-reflection-knowledge-write", _BAD_BODY)
    errors, _buckets = vault_round.validate([bad])
    assert len(errors) == 1, f"the lander let an unarchived overwrite through: {errors}"
    assert _STEM in errors[0], f"the refusal must name the report: {errors[0]}"
    assert "dated-copy archive step" in errors[0], (
        f"the refusal must name what is missing: {errors[0]}"
    )
    assert errors[0].startswith(f"{bad}: "), f"the error must name the file: {errors[0]}"

    good = _write_skill(scratch_vault, "nightly-reflection-knowledge-write", _GOOD_BODY)
    assert vault_round.validate([good])[0] == [], (
        "a skill that archives before overwriting must be landable: "
        f"{vault_round.validate([good])[0]}"
    )


def test_the_vault_writer_only_judges_the_skills_a_round_touched(scratch_vault):
    """Scoping, which is what keeps this off an unrelated round's critical path.

    A governed skill sitting unarchived in the vault must not block a round that
    lands a different file — the same reason `contract_errors` reads only the two
    identity files it is named for. Without this, adding the check to `validate`
    would make every skill edit in the tree hostage to the worst skill in it.
    """
    from scripts.automod import vault_round

    _write_skill(scratch_vault, "nightly-reflection-knowledge-write", _BAD_BODY)
    unrelated = _write_skill(
        scratch_vault, "some-other-skill",
        _SKILL_FRONT + "Nothing about reflection reports here.\n",
    )
    assert vault_round.validate([unrelated])[0] == [], (
        "an untouched skill was judged for a condition it did not change"
    )
    # And a touched skill that never mentions a report is judged and clean.
    assert vault_round.reflection_archive_errors([unrelated]) == []


def test_the_writer_and_the_test_share_one_definition():
    """Static pin: the lander must call the same rule this file asserts.

    Mirrors `tests/test_prompt_surface_guard.py::test_both_writers_call_the_shared_invariants`
    for the same reason — a mark and a helper can both exist while the call site
    quietly disappears, and then the rule is documentation again.
    """
    src = (ROOT / "scripts" / "automod" / "vault_round.py").read_text(encoding="utf-8")
    assert "import reflection_archive" in src, (
        "vault_round no longer imports the retention rule: the writer stopped "
        "enforcing it and only the live_vault reporting copy remains"
    )
    assert "reflection_archive.skill_rule_violations" in src, (
        "vault_round no longer calls the retention rule"
    )
    validate_src = src.split("def validate(", 1)[1].split("\ndef ", 1)[0]
    assert "reflection_archive_errors(paths)" in validate_src, (
        "reflection_archive_errors is defined but validate() no longer calls it — "
        "the lander would compute nothing and land everything"
    )


# --- Non-vacuity: the pin must be able to fail ---------------------------------
def _mutate(body: str, old: str, new: str) -> str:
    assert old in body, "mutation target absent — the fixture drifted, fix the fixture"
    return body.replace(old, new)


def test_the_archive_check_can_actually_fail():
    """Every way this guard was asked to fail, on synthetic bodies. Without this
    the suite could be green on a check that matches nothing."""
    assert classify(_GOOD_BODY, _STEM) == "tool-write", (
        "the well-formed skill body must classify as a writer — otherwise every "
        "archive assertion below is vacuous"
    )
    assert archive_problems(_GOOD_BODY, _STEM) == [], "the well-formed case must be clean"
    assert ra.skill_rule_violations("s", _GOOD_BODY) == [], (
        "the writer's entry point reported a problem on a skill that archives"
    )

    bad = _BAD_BODY
    assert ra.skill_rule_violations("s", bad), (
        "the writer's entry point passed a skill with no archive step"
    )
    assert classify(bad, _STEM) == "tool-write", (
        "removing the cp must not have removed the writer — the skill still "
        "overwrites the report, which is the whole problem"
    )

    good = _GOOD_BODY
    cases = {
        "the cp instruction deleted": good.replace(_COPY_LINE, ""),
        "the cp replaced by a no-op": _mutate(good, "      cp ", "      echo "),
        "the stamp taken from the local clock": _mutate(
            good, "STAMP=$(date -u +%Y-%m-%d-%H%M)", "STAMP=$(date +%Y-%m-%d-%H%M)"
        ),
        "the stamp a `<cycle>` placeholder": _mutate(
            good, f"{_LIVE[:-3]}-$STAMP.md", f"{_LIVE[:-3]}-<cycle>.md"
        ),
        "the stamp a hand-typed date": _mutate(
            good, f"{_LIVE[:-3]}-$STAMP.md", f"{_LIVE[:-3]}-2026-09-10-2230.md"
        ),
        "the copy pointed at a different file": _mutate(
            good,
            f"{_LIVE[:-3]}-$STAMP.md",
            "/home/alansrobotlab/lloyd/_pipeline/reflection/other-latest-$STAMP.md",
        ),
        # "before it is overwritten" is part of the clause, not a suggestion: the
        # same two instructions in the other order copy the new report and lose the
        # old one, so a check blind to order would pass a body that saves nothing.
        "the archive step written below the overwrite": "\n".join(
            ["### 2e. Pattern Output Files\n", _READ_LINE, "", _WRITE_LINE, _COPY_LINE, ""]
        ),
        "the read written below the overwrite": "\n".join(
            ["### 2e. Pattern Output Files\n", _COPY_LINE, "", _WRITE_LINE, _READ_LINE, ""]
        ),
    }
    for label, body in cases.items():
        problems = archive_problems(body, _STEM)
        assert problems, f"the check passed a body whose archive step is broken ({label})"
        assert ra.skill_rule_violations("s", body), (
            f"the writer would have landed a body with a broken archive step ({label})"
        )

    # A body that reads the file but never writes it is not a writer, so it must
    # not be demanded an archive step.
    assert classify(_READ_LINE + "\n", _STEM) == "read"
    # A write prescribed through vault_write is a broken instruction, not a writer.
    vault = f'Write to `vault_write(path="~/lloyd/_pipeline/reflection/{_STEM}.md")`:\n'
    assert classify(vault, _STEM) == "vault-write"

    # A line that reads *and* destroys the file is a write. If `read` won here, a
    # skill could say "Read the previous report and merge everything into <path>",
    # classify as a reader, and escape the archive rule, the tier-2 set equality and
    # the unclassified alarm all at once — so this case is the leak, pinned.
    read_merge = (
        f'Read the previous report and merge everything into '
        f'`~/lloyd/_pipeline/reflection/{_STEM}.md`\n'
    )
    assert classify(read_merge, _STEM) == "prose-write", (
        "a read-and-merge instruction was classified as a harmless read"
    )
    # Its inverse: a genuine read annotated with a write that belongs to another
    # file further down the line stays a reader. Without this pair the rule would
    # simply be "any line mentioning a write verb is a writer", which is wrong.
    annotated_read = (
        f'1. Signal report: `Read(file_path="/home/alansrobotlab/lloyd/_pipeline/'
        f'reflection/{_STEM}.md")` — absolute path, not `vault_read`. Same rule for '
        f"the handoff write below and for reading it back.\n"
    )
    assert classify(annotated_read, _STEM) == "read", (
        "a reader was recruited as a writer by a noun belonging to another file"
    )


def test_the_generated_field_check_can_actually_fail():
    """The pre-#436 wording, verbatim from vault commit 3514dac1, must be caught."""
    old_wording = (
        "> **Archive before the skeleton write, every run.** `<cycle>` is the "
        "previous report's own `generated:` stamp, so "
        "`signals-latest-2026-09-10-2230.md`.\n"
    )
    assert _generated_field_offenders(old_wording), (
        "the check accepted an instruction to build the name from generated:"
    )

    allowed = (
        "> `STAMP` is never read out of the previous report's `generated:` field. "
        "The `generated:` stamp is not usable here: the copy named "
        "`signals-latest-2026-09-10-2230.md` sorts 8 hours out.\n"
    )
    flagged = _generated_field_offenders(allowed)
    assert flagged == [], f"a permitted, negated mention was flagged: {flagged}"


def test_the_rule_survives_a_wrapped_shell_instruction():
    """The seam between a wrapped shell instruction and a line-matching rule.

    The archive step is one `Bash("cp …` written over three markdown lines with
    backslash continuations. If `logical_lines` stopped splicing them, the `cp`
    and the dated destination would be on different lines and every real archive
    step in the vault would read as missing — so the splice is asserted, in both
    directions, on a body that is wrapped exactly as §2e's is.
    """
    wrapped = ra.logical_lines(_COPY_LINE)
    assert len(wrapped) == 1, f"the continuation was not spliced: {wrapped}"
    assert ra.COPY.search(wrapped[0]) and ra.archive_dest(_STEM).search(wrapped[0]), (
        "the spliced line no longer matches as a copy to a dated destination"
    )
    assert ra.utc_stamp(wrapped[0]), "the spliced line no longer carries the UTC stamp"
    # The negative control: the same instruction un-spliced matches nothing, which
    # is what the splice is for.
    assert not any(
        ra.COPY.search(ln) and ra.archive_dest(_STEM).search(ln)
        for ln in _COPY_LINE.splitlines()
    ), "an un-spliced body matched, so the splice is not what makes the check work"


# --- Live-vault reporting copy: drift guards on the classifier itself -----------
@pytest.mark.live_vault
def test_classification_survives_an_unclassifiable_mention(skills):
    """Every mention must be attributed. A path that is neither written nor read
    nor prose-written lands in `unclassified`, which is how an under-inclusive
    rule would leak a writer — so it is a failure, not a shrug."""
    unclassified = {
        f"{name}:{stem}" for (name, stem), tier in classify_all(skills).items()
        if tier == "unclassified"
    }
    assert unclassified == set(), (
        f"reflection-report mentions the classifier could not attribute: "
        f"{unclassified}. Classify them (and if one writes the file, it owes an "
        "archive step) rather than letting the writer scan go narrow."
    )


@pytest.mark.live_vault
def test_declared_write_prescriptions_are_still_accurate(skills):
    """The tier-2 carve-outs must still describe reality, and no new one may
    appear unlisted. A carve-out that outlives its reason would quietly exempt a
    live writer from the archive rule."""
    classified = classify_all(skills)
    found: dict[str, set[str]] = {}
    for (name, stem), tier in classified.items():
        if tier in ("vault-write", "prose-write"):
            found.setdefault(name, set()).add(stem)
    assert found == EXPECTED_UNARCHIVED_WRITERS, (
        f"unarchived write prescriptions changed: found {found}, expected "
        f"{EXPECTED_UNARCHIVED_WRITERS}. A tool-call writer must archive; a "
        "prose/vault_write one is a finding on #436, not an exemption."
    )
    for name, stems in EXPECTED_UNARCHIVED_WRITERS.items():
        assert name in skills, f"{name} is no longer an active skill — prune its entry"
        for stem in stems:
            assert classified.get((name, stem)) in ("vault-write", "prose-write"), (
                f"{name}:{stem} changed classification to "
                f"{classified.get((name, stem))}; if it now writes with Write/Edit "
                "it must archive before overwriting"
            )


@pytest.mark.live_vault
def test_the_prose_writer_carve_out_is_debt_recorded_on_436(skills):
    """The honest limit of this guard, stated as an assertion rather than a comment.

    This file binds three write prescriptions and leaves four instructions alone:
    `historical-knowledge-refresh/SKILL.md` instructs merging into all three live
    reports, and nothing in the repo executes a `cp` for it, because the check
    binds instructions a run *executes* and that skill's write is a sentence. It is
    carved out as a finding on #436 (round SM_20260912_155333), not silently covered
    by a rule that reads "every writer". Asserted family by family so the debt can
    only shrink: a new prose writer has to be added here, which is a decision with a
    name on it, and a prose writer that is fixed disappears from it.
    """
    classified = classify_all(skills)
    prose: dict[str, set[str]] = {}
    vault: dict[str, set[str]] = {}
    for (name, stem), tier in classified.items():
        if tier == "prose-write":
            prose.setdefault(name, set()).add(stem)
        elif tier == "vault-write":
            vault.setdefault(name, set()).add(stem)
    assert prose == EXPECTED_PROSE_WRITERS, (
        f"prose writers changed: found {prose}, expected {EXPECTED_PROSE_WRITERS}. "
        "Either one was fixed (delete its entry) or a new skill writes reports in "
        "prose — which is a finding on #436, not an entry added to keep green."
    )
    assert vault == EXPECTED_VAULT_WRITE_WRITERS, (
        f"vault_write-shaped prescriptions changed: found {vault}, expected "
        f"{EXPECTED_VAULT_WRITE_WRITERS}"
    )
    # The debt is on the item, not just in this file: if #436 closes while a prose
    # writer still has no archive step, whoever closed it read this and disagreed.
    # Asserted, not guarded — a renamed or pruned item file must stop this test
    # dead rather than leave its final assertion silently unrun (review finding,
    # round SM_20260912_155333).
    items = sorted((Path.home() / "obsidian" / "backlog").glob("436-*.md"))
    assert items, (
        "backlog/436-*.md is gone, so the prose-writer carve-out has nowhere to "
        "live: re-record it on the item that owns #436's remaining scope, then "
        "point this assertion at it"
    )
    body = items[0].read_text(encoding="utf-8", errors="replace")
    for skill in prose:
        assert skill in body, (
            f"{skill} is unarchived prose-write debt on #436 but is not named in "
            "the item — the carve-out has become invisible to whoever closes it"
        )
