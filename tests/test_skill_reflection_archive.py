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
This file is the instrument: the archive step is now pinned, and deleting it
turns the suite red instead of losing a cycle.

What each clause is pinned by
-----------------------------
1. *Every live report gets a dated copy before its overwrite, on every cycle* →
   ``test_every_skill_that_writes_a_reflection_report_archives_it_first``, over
   the writer set discovered by ``_classify_writers``. Clause 1's specific
   requirement — §2e of ``nightly-reflection-knowledge-write`` for both pattern
   files — is asserted separately in
   ``test_knowledge_write_section_2e_archives_both_pattern_files``.
2. *A test in the ``tests/test_skill_*.py`` family, failing when such a line is
   removed* → the same test, plus ``test_the_archive_check_can_actually_fail``,
   which mutates synthetic skill bodies and asserts the check reports each
   removal class. A pin that cannot fail is not a pin.
3. *The stamp must be UTC-derived, never a `generated:` field read out of the
   previous report* → ``test_archive_stamp_is_derived_from_a_utc_command`` (the
   destination is ``$STAMP``/``$(date …)`` and that variable is assigned from
   ``date -u … +%Y-%m-%d-%H%M`` on the same logical line) and
   ``test_no_skill_builds_an_archive_name_from_a_generated_field``.
4. *No gitignored file under ``~/lloyd`` is touched* → this test reads skill
   markdown only; it never opens ``_pipeline/`` (see
   ``test_the_check_needs_no_pipeline_directory``), so retention can never be
   "achieved" by tracking the gitignored tree.

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
prose ("Merge all ``signals-*.md`` into …"). Tier 2 entries carry a reason and are
re-verified every run: if one ever gains a real ``Write`` call it moves to tier 1
and must have an archive step. Both tier-2 families are recorded as findings on
#436 rather than widened into this diff.

The attribution boundary, stated plainly: a mention counts when it names the path
(``reflection/<name>-latest.md``). A skill that refers to ``test-results-latest.md``
by bare filename somewhere else is not attributed, because a bare filename cannot
be told apart from one in a different directory. That is the reason
``_PROSE_WRITE`` carries verbs of instruction and not nouns like "output" — the
nouns are what descriptive annotations use, and three of the read-list items in
``autonomy-reflection-pipeline`` Step 6.1 are annotated that way.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SKILLS_DIRS = [Path.home() / "obsidian" / "skills", ROOT / "skills"]

#: A live report path under the reflection directory. `-latest` is the in-place
#: convention this test exists to police; the per-day artifacts in the same
#: directory (`knowledge-handoff-2026-09-12.md`, `knowledge-write-*`) already
#: carry their date in the name and need no check.
_LATEST_PATH = re.compile(r"_pipeline/reflection/([A-Za-z0-9_-]+-latest)\.md")

#: A write that can actually land the file: a `Write`/`Edit` call whose target is
#: a literal path. `Write to` alone is not enough — the older skills say
#: "Write to `vault_write(path=…)`", which is this next pattern's business.
_TOOL_WRITE = re.compile(
    r"(?<![A-Za-z_])\bWrite\s*\(\s*(?:file_path\s*=|[\"'`])"
    r"|(?<![A-Za-z_])\bEdit\s*\("
)

#: Tools scoped to the vault. On a `~/lloyd/` target they return PATH_ESCAPE, so a
#: skill naming them is a broken prescription, not a writer that loses data.
#: Checked before _TOOL_WRITE's "Write to" prose form for exactly that reason.
_VAULT_WRITE = re.compile(r"\bvault_write\s*\(|\bmem_write\s*\(")

#: Prose instruction to produce the file, naming no tool that could do it.
_WRITE_TO = re.compile(r"\bWrite\s+(?:this|to|the|pattern|both)\b|^\s*[-*]?\s*\*\*Write\b")

_READ = re.compile(
    r"(?<![A-Za-z_])\bRead\s*\(|\bvault_read\s*\(|\bmem_get\s*\(|\bRead (?:this|these)\b"
    r"|\bInput:|\bprevious (?:audit|report)\b",
    re.IGNORECASE,
)

#: Prose that prescribes producing the file, with no tool call to bind. Deliberately
#: verbs of *instruction* only: "Output: …" and "— prompt audit output (Subagent B)"
#: annotate a file, they do not write it, and the second is the annotation on an
#: item of a **read** list (`autonomy-reflection-pipeline` Step 6.1), so a noun here
#: would read a reader as a writer.
_PROSE_WRITE = re.compile(
    r"\b(?:merge|merges|consolidate|consolidates|regenerate|regenerates"
    r"|write|writes|written)\b",
    re.IGNORECASE,
)

#: An archive destination: `<stem>-<stamp>.md` where the stamp is produced by the
#: shell. A `$STAMP` variable is accepted (its assignment is checked separately by
#: _utc_stamp), a `$(date …)` substitution must carry `-u` inline, and a
#: `<cycle>`/`<YYYY-MM-DD>` placeholder or a hand-typed date is not: those are
#: exactly the "name comes from prose or from someone's typing" case clause 3
#: forbids. A bare `<stem>.md` overwrite does not match either.
def _archive_dest(stem: str) -> re.Pattern[str]:
    stamp = r"(?:\$STAMP|\$\{STAMP\}|\$\(date\s+-u[^)]*\))"
    return re.compile(re.escape(stem) + "-" + stamp + r"\.md")


_COPY = re.compile(r"(?<![A-Za-z0-9_])cp\b|shutil\.copy")


def _utc_stamp(line: str) -> bool:
    """True when the line derives a stamp from the UTC clock.

    Order-independent so `STAMP=$(date -u +%Y-%m-%d-%H%M)` and the mtime form
    `date -u -r "$src" +%Y-%m-%d-%H%M` (clause 3 allows both) both count. A local
    `date +%Y-%m-%d-%H%M` does not: it is the defect that made
    `signals-latest-2026-09-10-2230.md`, a file written at 2026-09-12 05:03 UTC
    whose name sorts 8 hours behind that.
    """
    return "date" in line and "-u" in line and "%Y-%m-%d-%H%M" in line


def _logical_lines(body: str) -> list[str]:
    """Body with backslash-newline continuations spliced into single lines.

    Skill code blocks wrap long `Bash("cp …` invocations; the archive step is one
    instruction, and matching per raw line would miss every wrapped copy.
    """
    return re.sub(r"\\\s*\n\s*", " ", body).splitlines()


def _active_skills() -> dict[str, str]:
    """Every SKILL.md the prompt actually advertises → its body.

    Mirrors prompt_builder._load_skills_index: dot-prefixed directories are the
    archive, quarantined skills are out of circulation.
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
    bodies = _active_skills()
    if not bodies:
        pytest.skip("no skills directory on this machine")
    return bodies


def _mention_lines(body: str, stem: str) -> list[tuple[int, str]]:
    """(line index, text) for every logical line naming this live report."""
    needle = f"_pipeline/reflection/{stem}.md"
    return [(i, ln) for i, ln in enumerate(_logical_lines(body)) if needle in ln]


def _heading_verb(body_lines: list[str], idx: int) -> str | None:
    """Verb from the section a bare path bullet sits in.

    `Write pattern files from the artifact:` followed by two backticked paths is a
    write prescription; `Read these files:` followed by seven is not — and in
    `autonomy-reflection-pipeline` Step 6.1 the read list is seven items long, while
    in `historical-knowledge-refresh` the write list nests two levels
    (`**Consolidate patterns:** Merge all pattern analysis into:` then two sub-bullets).
    So this walks back through the list, taking the first verb it meets, and gives
    up at a blank line, a heading, or six non-blank lines — far enough for a nested
    bullet, not far enough to borrow a verb from the previous section.
    """
    walked = 0
    for j in range(idx - 1, -1, -1):
        prev = body_lines[j]
        if not prev.strip():
            return None
        if prev.lstrip().startswith("#"):
            return None
        if _TOOL_WRITE.search(prev) or _VAULT_WRITE.search(prev) or _WRITE_TO.search(prev) or _PROSE_WRITE.search(prev):
            return "prose-write"
        if _READ.search(prev):
            return "read"
        walked += 1
        if walked >= 6:
            return None
    return None


def _line_tier(body_lines: list[str], idx: int, line: str) -> str:
    """One mention line's tier.

    `vault_write` is asked before the "Write to" prose form because the skills that
    use that form are writing *to vault_write* — an instruction that cannot reach
    the path. The direct-call test comes first of all so a skill that names both
    (`Write(file_path=…)` — "`Write` not `vault_write`") still reads as a writer.
    """
    if _TOOL_WRITE.search(line):
        return "tool-write"
    if _VAULT_WRITE.search(line):
        return "vault-write"
    if _READ.search(line):
        return "read"
    if _WRITE_TO.search(line) or _PROSE_WRITE.search(line):
        return "prose-write"
    return _heading_verb(body_lines, idx) or "unclassified"


def _classify(body: str, stem: str) -> str:
    """The strongest tier any mention of `stem` carries.

    A skill with one real write and three reads is a writer: the reads cannot undo
    the write, and the archive step is required by the write.
    """
    lines = _logical_lines(body)
    mentions = _mention_lines(body, stem)
    if not mentions:
        return "absent"
    tiers = {_line_tier(lines, idx, line) for idx, line in mentions}
    for tier in ("tool-write", "vault-write", "prose-write", "read", "unclassified"):
        if tier in tiers:
            return tier
    return "unclassified"


def _classify_all(skills: dict[str, str]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for name, body in skills.items():
        for stem in sorted(set(_LATEST_PATH.findall(body))):
            out[(name, stem)] = _classify(body, stem)
    return out


def _archive_problems(body: str, stem: str) -> list[str]:
    """Why this skill does not archive `stem` before overwriting it, or []."""
    problems: list[str] = []
    copied = [ln for ln in _logical_lines(body) if _COPY.search(ln) and _archive_dest(stem).search(ln)]
    if not copied:
        problems.append(
            f"no dated-copy archive step for {stem}: no cp to `{stem}-<stamp>.md`"
        )
        return problems
    if not any(f"{stem}.md" in ln for ln in copied):
        problems.append(
            f"archive step for {stem} never names the live path it is copying from"
        )
    if not any(_utc_stamp(ln) for ln in copied):
        problems.append(
            f"archive stamp for {stem} is not derived from a UTC command "
            f"(`date -u … +%Y-%m-%d-%H%M`, or the source's mtime)"
        )
    return problems


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
EXPECTED_UNARCHIVED_WRITERS: dict[str, set[str]] = {
    # Prescribes the write through vault_write / mem_write, which reject any
    # ~/lloyd/ target with PATH_ESCAPE — the instruction cannot destroy a report.
    # (`autonomy-reflection-pipeline` names prompt-audit-latest and
    # test-results-latest only inside Step 6.1's *read* list, so they classify as
    # reads; it prescribes producing them by bare filename, which no path-shaped
    # check can attribute.)
    "autonomy-reflection-pipeline": {"signals-latest", "day-end-synthesis-latest"},
    "nightly-prompt-audit": {"prompt-audit-latest"},
    "nightly-behavior-test": {"test-results-latest"},
    "nightly-day-end-synthesis": {"day-end-synthesis-latest"},
    # Prose writes ("**Consolidate signals:** Merge all `signals-YYYY-MM-DD.md`
    # into …", lines 94-98) — a real loss vector when the refresh is run, with no
    # dated copy and no tool call for this check to bind. Finding on #436.
    "historical-knowledge-refresh": {
        "signals-latest",
        "tool-patterns-latest",
        "conversation-patterns-latest",
    },
}


# --- Clause 1 + 2 ---------------------------------------------------------------
def test_every_skill_that_writes_a_reflection_report_archives_it_first(skills):
    """The acceptance clause: every live nightly reflection report gets a dated
    copy before it is overwritten, on every cycle, pinned here rather than by the
    skill prose alone."""
    classified = _classify_all(skills)
    offenders = {
        f"{name}:{stem}": _archive_problems(skills[name], stem)
        for (name, stem), tier in classified.items()
        if tier == "tool-write"
        for bad in [_archive_problems(skills[name], stem)]
        if bad
    }
    assert offenders == {}, (
        "skills write a nightly reflection report in place without an archive "
        f"step for that same path: {offenders}. `_pipeline/` is gitignored, so "
        "the overwritten report is unrecoverable (#436)."
    )


def test_discovered_tool_writers_are_exactly_the_expected_set(skills):
    """Anti-drift for the classifier itself: a new tool-writer, or one that
    disappears, is a decision someone has to record here."""
    classified = _classify_all(skills)
    found: dict[str, set[str]] = {}
    for (name, stem), tier in classified.items():
        if tier == "tool-write":
            found.setdefault(name, set()).add(stem)
    assert found == EXPECTED_TOOL_WRITERS, (
        f"reflection-report writers changed: found {found}, expected "
        f"{EXPECTED_TOOL_WRITERS}. Every entry must archive before overwriting."
    )


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
        assert not _archive_problems(section, stem), (
            f"§2e must Read and dated-copy {stem}.md before overwriting it: "
            f"{_archive_problems(section, stem)}"
        )
        assert re.search(r"(?<![A-Za-z_])\bRead\s*\(", section), (
            "§2e must Read the existing pattern file: Write refuses an unread "
            "file, so an instruction that omits it was refused on every run"
        )


# --- Clause 3 ------------------------------------------------------------------
def test_archive_stamp_is_derived_from_a_utc_command(skills):
    """The archive name must come from `date -u … +%Y-%m-%d-%H%M` (or the source's
    mtime) — never a hand-typed date."""
    problems: dict[str, list[str]] = {}
    for (name, stem), tier in _classify_all(skills).items():
        if tier != "tool-write":
            continue
        bad = [p for p in _archive_problems(skills[name], stem) if "UTC" in p]
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


def test_no_skill_builds_an_archive_name_from_a_generated_field(skills):
    """Clause 3's negative half. The report's own `generated:` header is a local
    (PST) clock reading, so a copy named from it sorts 8 hours out from the
    UTC-named siblings in the same directory and cannot be re-derived when the
    field is missing. A skill may *say* this — with a negation — but must not
    instruct it."""
    offenders: dict[str, list[str]] = {}
    for (name, stem), tier in _classify_all(skills).items():
        if tier != "tool-write":
            continue
        bad = [
            s for s in _sentences_mentioning(skills[name], "generated:")
            if ".md" in s and not re.search(r"\b(?:never|not|no longer|isn't|cannot)\b", s, re.I)
        ]
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
    import subprocess

    probe = "_pipeline/reflection/signals-latest.md"
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-q", probe],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git unavailable; cannot check the ignore rule")
    assert proc.returncode == 0, (
        f"`{probe}` is no longer gitignored. #436 keeps the nightly reports out of "
        "the repo and archives them as dated copies; committing generated reports "
        "into ~/lloyd is Alan's decision, not a side effect of this test file."
    )


# --- Non-vacuity: the pin must be able to fail ---------------------------------
_STEM = "tool-patterns-latest"
_LIVE = f"/home/alansrobotlab/lloyd/_pipeline/reflection/{_STEM}.md"
# Written the way the skills write it: one instruction wrapped over three lines
# with backslash continuations, so this fixture also exercises _logical_lines.
_READ_LINE = f'Read("{_LIVE}")'
_COPY_LINE = (
    'Bash("STAMP=$(date -u +%Y-%m-%d-%H%M) && \\\n'
    f"      cp {_LIVE} \\\n"
    f"         {_LIVE[:-3]}-$STAMP.md\")"
)
_WRITE_LINE = f'- `Write(file_path="{_LIVE}", content=…)` — new content'
_GOOD = "\n".join(["### 2e. Pattern Output Files\n", _READ_LINE, _COPY_LINE, "", _WRITE_LINE, ""])


def _mutate(old: str, new: str) -> str:
    assert old in _GOOD, "mutation target absent — the fixture drifted, fix the fixture"
    return _GOOD.replace(old, new)


def test_the_archive_check_can_actually_fail():
    """Every way this guard was asked to fail, on synthetic bodies. Without this
    the suite could be green on a check that matches nothing."""
    assert _classify(_GOOD, _STEM) == "tool-write", (
        "the well-formed skill body must classify as a writer — otherwise every "
        "archive assertion below is vacuous"
    )
    assert _archive_problems(_GOOD, _STEM) == [], "the well-formed case must be clean"

    cases = {
        "the cp instruction deleted": _GOOD.replace(_COPY_LINE, ""),
        "the cp replaced by a no-op": _mutate("      cp ", "      echo "),
        "the stamp taken from the local clock": _mutate(
            "STAMP=$(date -u +%Y-%m-%d-%H%M)", "STAMP=$(date +%Y-%m-%d-%H%M)"
        ),
        "the stamp a `<cycle>` placeholder": _mutate(
            f"{_LIVE[:-3]}-$STAMP.md", f"{_LIVE[:-3]}-<cycle>.md"
        ),
        "the stamp a hand-typed date": _mutate(
            f"{_LIVE[:-3]}-$STAMP.md", f"{_LIVE[:-3]}-2026-09-10-2230.md"
        ),
        "the copy pointed at a different file": _mutate(
            f"{_LIVE[:-3]}-$STAMP.md",
            "/home/alansrobotlab/lloyd/_pipeline/reflection/other-latest-$STAMP.md",
        ),
    }
    for label, body in cases.items():
        problems = _archive_problems(body, _STEM)
        assert problems, f"the check passed a body whose archive step is broken ({label})"

    # A body that reads the file but never writes it is not a writer, so it must
    # not be demanded an archive step.
    reader = f'{_READ_LINE}\n'
    assert _classify(reader, _STEM) == "read"
    # A write prescribed through vault_write is a broken instruction, not a writer.
    vault = (
        f'Write to `vault_write(path="~/lloyd/_pipeline/reflection/{_STEM}.md")`:\n'
    )
    assert _classify(vault, _STEM) == "vault-write"


def test_the_generated_field_check_can_actually_fail():
    """The pre-#436 wording, verbatim from vault commit 3514dac1, must be caught."""
    old_wording = (
        "> **Archive before the skeleton write, every run.** `<cycle>` is the "
        "previous report's own `generated:` stamp, so "
        "`signals-latest-2026-09-10-2230.md`.\n"
    )
    hits = [
        s
        for s in _sentences_mentioning(old_wording, "generated:")
        if ".md" in s and not re.search(r"\b(?:never|not|no longer|isn't|cannot)\b", s, re.I)
    ]
    assert hits, "the check accepted an instruction to build the name from generated:"

    allowed = (
        "> `STAMP` is never read out of the previous report's `generated:` field. "
        "The `generated:` stamp is not usable here: the copy named "
        "`signals-latest-2026-09-10-2230.md` sorts 8 hours out.\n"
    )
    flagged = [
        s
        for s in _sentences_mentioning(allowed, "generated:")
        if ".md" in s and not re.search(r"\b(?:never|not|no longer|isn't|cannot)\b", s, re.I)
    ]
    assert flagged == [], f"a permitted, negated mention was flagged: {flagged}"


def test_classification_survives_an_unclassifiable_mention(skills):
    """Every mention must be attributed. A path that is neither written nor read
    nor prose-written lands in `unclassified`, which is how an under-inclusive
    rule would leak a writer — so it is a failure, not a shrug."""
    unclassified = {
        f"{name}:{stem}" for (name, stem), tier in _classify_all(skills).items()
        if tier == "unclassified"
    }
    assert unclassified == set(), (
        f"reflection-report mentions the classifier could not attribute: "
        f"{unclassified}. Classify them (and if one writes the file, it owes an "
        "archive step) rather than letting the writer scan go narrow."
    )


def test_declared_write_prescriptions_are_still_accurate(skills):
    """The tier-2 carve-outs must still describe reality, and no new one may
    appear unlisted. A carve-out that outlives its reason would quietly exempt a
    live writer from the archive rule."""
    classified = _classify_all(skills)
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
