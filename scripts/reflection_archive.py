"""Nightly reflection reports must be archived before they are overwritten —
stated once, and checked where writes land.

Why this is a module and not only a test
----------------------------------------
Backlog #436 first fixed the loss in skill prose: vault commit ``3514dac1``
(2026-09-12) added an archive ``cp`` to ``nightly-reflection-signals`` for
``signals-latest.md`` alone. Prose one model reads once per run is what the
other two reports were still relying on, and the skill says so in its own
words: "A Read-before-Write discipline that preserves a file only while the
reader remembers to copy it is not a retention policy."

The obvious instrument — a test that reads ``~/obsidian/skills`` — is the wrong
place for the *enforcement*, for the reason ``prompt_surface.py`` documents:
that tree is state no round under test controls, so on the gate's hard `tests`
rung it punishes the next author for the previous writer's wording. Round
``SM_20260912_155333`` learned the other half of the same lesson: marked
``live_vault`` instead, and the review rung refused it because nothing
automated evaluated it either. Both refusals describe one missing piece — the
rule lived in a test file, so there was nowhere it *ran*.

So the definition moves here and gets three consumers, exactly like the
identity surface's invariants:

* ``scripts/automod/vault_round.py`` runs :func:`skill_rule_violations` on every
  ``skills/**/SKILL.md`` a vault round touches — the writer. Removing an archive
  step from a governed skill is refused at ``automod_vault_land``, before the
  commit, and the round's paths are reverted. This is the automated rung.
* ``tests/test_skill_reflection_archive.py`` imports it. Its mutations prove the
  rule can fail (unmarked, so every gate rung runs them), and its live-vault
  group — the reporting copy ``pytest.ini`` describes — asserts the real skills
  against it.
* The rule is scoped to the diff like every other vault validator: a
  pre-existing condition in an untouched skill must not block an unrelated
  round.

Stdlib only, and no import of ``prompt_builder``, for the same reason
``prompt_surface`` states it: ``vault_round`` executes its validators with the
round's interpreter warm and the loaders in a fresh one, and a heavy import is
paid on both paths. The skill-tree scan and the quarantine filter stay in the
test, which is the only consumer that needs to enumerate skills at all.

What "archived" means here is clause 3 of #436: the destination is
``<stem>-<stamp>.md`` where the stamp comes from the UTC clock
(``date -u … +%Y-%m-%d-%H%M``, or the source file's mtime), never from a field
inside the previous report. ``signals-latest-2026-09-10-2230.md`` — a file
written at 2026-09-12 05:03 UTC, named from its own local-time ``generated:``
header — is the defect that rule forbids: it sorts eight hours behind the
UTC-named siblings in the same directory, and cannot be re-derived at all when
the header is missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: A live report path under the reflection directory. `-latest` is the in-place
#: convention this rule polices; the per-day artifacts in the same directory
#: (`knowledge-handoff-2026-09-12.md`, `knowledge-write-*`) already carry their
#: date in the name and need no check.
LATEST_PATH = re.compile(r"_pipeline/reflection/([A-Za-z0-9_-]+-latest)\.md")

#: A write that can actually land the file: a `Write`/`Edit` call whose target is
#: a literal path. `Write to` alone is not enough — the older skills say
#: "Write to `vault_write(path=…)`", which is this next pattern's business.
TOOL_WRITE = re.compile(
    r"(?<![A-Za-z_])\bWrite\s*\(\s*(?:file_path\s*=|[\"'`])"
    r"|(?<![A-Za-z_])\bEdit\s*\("
)

#: Tools scoped to the vault. On a `~/lloyd/` target they return PATH_ESCAPE, so a
#: skill naming them is a broken prescription, not a writer that loses data.
#: Checked before TOOL_WRITE's "Write to" prose form for exactly that reason.
VAULT_WRITE = re.compile(r"\bvault_write\s*\(|\bmem_write\s*\(")

#: Prose instruction to produce the file, naming no tool that could do it.
WRITE_TO = re.compile(r"\bWrite\s+(?:this|to|the|pattern|both)\b|^\s*[-*]?\s*\*\*Write\b")

READ = re.compile(
    r"(?<![A-Za-z_])\bRead\s*\(|\bvault_read\s*\(|\bmem_get\s*\(|\bRead (?:this|these)\b"
    r"|\bInput:|\bprevious (?:audit|report)\b",
    re.IGNORECASE,
)

#: Prose that prescribes producing the file, with no tool call to bind. Deliberately
#: verbs of *instruction* only: "Output: …" and "— prompt audit output (Subagent B)"
#: annotate a file, they do not write it, and the second is the annotation on an
#: item of a **read** list (`autonomy-reflection-pipeline` Step 6.1), so a noun here
#: would read a reader as a writer.
PROSE_WRITE = re.compile(
    r"\b(?:merge|merges|consolidate|consolidates|regenerate|regenerates"
    r"|write|writes|written)\b",
    re.IGNORECASE,
)

COPY = re.compile(r"(?<![A-Za-z0-9_])cp\b|shutil\.copy")

#: Tiers, strongest first. `tool-write` and `prose-write` are the tiers the
#: archive rule binds (see `BOUND_TIERS`). `vault-write` is exempt for a reason
#: that is not a loophole: those calls reject a `~/lloyd/` target with
#: `PATH_ESCAPE`, so nothing they prescribe can overwrite anything.
#: `unclassified` is a failure of *this* rule, not a verdict about the skill,
#: which is why the test alarms on it instead of letting an under-inclusive rule
#: hide a writer.
TIER_ORDER = ("tool-write", "vault-write", "prose-write", "read", "unclassified")

#: The tiers that destroy the report when no dated copy precedes them, and that
#: the vault writer therefore refuses to land without one.
#:
#: Both of them, deliberately. A prose instruction ("Merge all `signals-*.md`
#: into …") is a sentence rather than a call a run executes, so this rule cannot
#: make the run copy the file first — but #436's acceptance quantifier is about
#: *skills that write a report*, not about calls, and a rule binding only
#: `tool-write` would let a skill destroy all three reports in prose and still
#: read as covered. So a prose writer is bound the only way a prose writer can be
#: bound: the archive step must be named in the same skill text, in the shape the
#: nightly writers use, and the vault writer refuses to land the skill without
#: it. Narrowing this tuple is narrowing the clause, which is why
#: `tests/test_skill_reflection_archive.py` asserts it by value.
BOUND_TIERS = ("tool-write", "prose-write")


def archive_dest(stem: str) -> re.Pattern[str]:
    """`<stem>-<stamp>.md` where the stamp is produced by the shell.

    A `$STAMP` variable is accepted (its assignment is checked separately by
    :func:`utc_stamp`), a `$(date …)` substitution must carry `-u` inline, and a
    `<cycle>`/`<YYYY-MM-DD>` placeholder or a hand-typed date is not: those are
    exactly the "name comes from prose or from someone's typing" case clause 3
    forbids. A bare `<stem>.md` overwrite does not match either.
    """
    stamp = r"(?:\$STAMP|\$\{STAMP\}|\$\(date\s+-u[^)]*\))"
    return re.compile(re.escape(stem) + "-" + stamp + r"\.md")


def utc_stamp(line: str) -> bool:
    """True when the line derives a stamp from the UTC clock.

    Order-independent so `STAMP=$(date -u +%Y-%m-%d-%H%M)` and the mtime form
    `date -u -r "$src" +%Y-%m-%d-%H%M` (clause 3 allows both) both count. A local
    `date +%Y-%m-%d-%H%M` does not: it is the defect that made
    `signals-latest-2026-09-10-2230.md`.
    """
    return "date" in line and "-u" in line and "%Y-%m-%d-%H%M" in line


def logical_lines(body: str) -> list[str]:
    """Body with backslash-newline continuations spliced into single lines.

    Skill code blocks wrap long `Bash("cp …` invocations; the archive step is one
    instruction written over three markdown lines, and matching per raw line
    would miss every wrapped copy — i.e. it would report every real archive step
    as missing.
    """
    return re.sub(r"\\\s*\n\s*", " ", body).splitlines()


def mention_lines(body: str, stem: str) -> list[tuple[int, str]]:
    """(line index, text) for every logical line naming this live report."""
    needle = f"_pipeline/reflection/{stem}.md"
    return [(i, ln) for i, ln in enumerate(logical_lines(body)) if needle in ln]


def _heading_verb(body_lines: list[str], idx: int) -> str | None:
    """Verb from the section a bare path bullet sits in.

    `Write pattern files from the artifact:` followed by two backticked paths is a
    write prescription; `Read these files:` followed by seven is not — and in
    `autonomy-reflection-pipeline` Step 6.1 the read list is seven items long, while
    in `historical-knowledge-refresh` the write list nests two levels
    (`**Consolidate patterns:** Merge all pattern analysis into:` then two
    sub-bullets). So this walks back through the list, taking the first verb it
    meets, and gives up at a blank line, a heading, or six non-blank lines — far
    enough for a nested bullet, not far enough to borrow a verb from the previous
    section.
    """
    walked = 0
    for j in range(idx - 1, -1, -1):
        prev = body_lines[j]
        if not prev.strip():
            return None
        if prev.lstrip().startswith("#"):
            return None
        if (TOOL_WRITE.search(prev) or VAULT_WRITE.search(prev)
                or WRITE_TO.search(prev) or PROSE_WRITE.search(prev)):
            return "prose-write"
        if READ.search(prev):
            return "read"
        walked += 1
        if walked >= 6:
            return None
    return None


def prose_write_before_path(line: str, needle: str) -> bool:
    """A prose instruction governing *this* path: the verb must precede the path.

    Position is what separates an instruction from an annotation. "Read the previous
    report and merge everything into `<path>`" is a write that also reads, and must
    not be talked out of being one by the word `Read` at its head; while
    "`Read(file_path="<path>")` — Same rule for the handoff write below" mentions a
    write that belongs to a different file, and reading a report does not make a
    reader a writer. An instruction puts its verb in front of its object.
    """
    at = line.find(needle)
    if at < 0:
        return False
    head = line[:at]
    return bool(WRITE_TO.search(head) or PROSE_WRITE.search(head))


def line_tier(body_lines: list[str], idx: int, line: str, needle: str) -> str:
    """One mention line's tier.

    `vault_write` is asked before the "Write to" prose form because the skills that
    use that form are writing *to vault_write* — an instruction that cannot reach
    the path. The direct-call test comes first of all so a skill that names both
    (`Write(file_path=…)` — "`Write` not `vault_write`") still reads as a writer.
    Write tiers are then asked **before** `read`: a line that both reads and writes
    the path destroys the report, which is the thing being guarded.
    """
    if TOOL_WRITE.search(line):
        return "tool-write"
    if VAULT_WRITE.search(line):
        return "vault-write"
    if prose_write_before_path(line, needle):
        return "prose-write"
    if READ.search(line):
        return "read"
    return _heading_verb(body_lines, idx) or "unclassified"


def classify(body: str, stem: str) -> str:
    """The strongest tier any mention of `stem` carries.

    A skill with one real write and three reads is a writer: the reads cannot undo
    the write, and the archive step is required by the write.
    """
    lines = logical_lines(body)
    needle = f"_pipeline/reflection/{stem}.md"
    mentions = mention_lines(body, stem)
    if not mentions:
        return "absent"
    tiers = {line_tier(lines, idx, line, needle) for idx, line in mentions}
    for tier in TIER_ORDER:
        if tier in tiers:
            return tier
    return "unclassified"


def classify_all(skills: dict[str, str]) -> dict[tuple[str, str], str]:
    """{(skill name, report stem): tier} for every reflection report any skill names."""
    out: dict[tuple[str, str], str] = {}
    for name, body in skills.items():
        for stem in sorted(set(LATEST_PATH.findall(body))):
            out[(name, stem)] = classify(body, stem)
    return out


def archive_problems(body: str, stem: str) -> list[str]:
    """Why this skill does not archive `stem` before overwriting it, or []."""
    problems: list[str] = []
    lines = logical_lines(body)
    needle = f"_pipeline/reflection/{stem}.md"
    copy_at = [
        i for i, ln in enumerate(lines) if COPY.search(ln) and archive_dest(stem).search(ln)
    ]
    if not copy_at:
        problems.append(
            f"no dated-copy archive step for {stem}: no cp to `{stem}-<stamp>.md`"
        )
        return problems
    copied = [lines[i] for i in copy_at]
    if not any(f"{stem}.md" in ln for ln in copied):
        problems.append(
            f"archive step for {stem} never names the live path it is copying from"
        )
    if not any(utc_stamp(ln) for ln in copied):
        problems.append(
            f"archive stamp for {stem} is not derived from a UTC command "
            f"(`date -u … +%Y-%m-%d-%H%M`, or the source's mtime)"
        )
    # "Before it is overwritten" is the clause, so order is part of what is pinned:
    # an archive step written *below* the write instruction copies the new report and
    # loses the old one exactly as completely as no copy at all. Asked over every
    # bound tier, not just `Write(` calls — a skill whose overwrite is a prose
    # instruction has the same ordering hazard, and it is the tier #436's quantifier
    # is mainly about.
    write_at = [
        i for i, ln in enumerate(lines)
        if needle in ln and line_tier(lines, i, ln, needle) in BOUND_TIERS
    ]
    if write_at and min(copy_at) > min(write_at):
        problems.append(
            f"archive step for {stem} appears after the instruction that overwrites it"
        )
    read_at = [
        i for i, ln in enumerate(lines)
        if needle in ln and re.search(r"(?<![A-Za-z_])\bRead\s*\(", ln)
    ]
    if read_at and write_at and min(read_at) > min(write_at):
        problems.append(
            f"{stem} is written before it is read, so Write was refused on every run"
        )
    return problems


def stems_written(body: str) -> list[str]:
    """Reports this skill body overwrites, in any tier the rule binds.

    Both a `Write(file_path=…)` and a prose "Merge … into <path>" are here: which
    of the two it is decides how the archive step has to be written, not whether
    one is owed.
    """
    return sorted(
        stem for stem in set(LATEST_PATH.findall(body))
        if classify(body, stem) in BOUND_TIERS
    )


def skill_rule_violations(name: str, body: str) -> list[str]:
    """Why this skill must not land as written, or []. Scoped to bound tiers.

    This is the vault writer's entry point: it is called per touched
    `skills/<name>/SKILL.md`, so it must answer for one body alone and name what
    it refused in terms a reader of `automod_vault_land`'s error can act on.
    """
    out: list[str] = []
    for stem in stems_written(body):
        for problem in archive_problems(body, stem):
            out.append(f"{name}:{stem}: {problem}")
    return out


# ── Copy gaps: the one check that reads the directory (#1227) ────────────────
#
# Everything above reads skill *text*, so a run that ignores its skill and
# overwrites a `-latest` report without the `cp` loses that cycle with no signal
# anywhere. The evidence that an overwrite happened is the writer's own report
# naming the copy it made; a copy that report names and the directory lacks is a
# lost cycle. The expectation deliberately comes from that pointer and never
# from a calendar: copies are stamped in UTC beside locally-dated siblings, and
# a pattern-file cycle exists only on nights the run actually wrote the file
# (2026-09-14 and -16 have a knowledge-write report, no tool-patterns copy, and
# lost nothing). A report that names no copy is therefore not a gap.

#: The line the nightly writers emit, e.g. signals-latest.md's
#: "Archive copied before this write: `signals-latest-2026-09-20-0517.md`".
ARCHIVE_POINTER = re.compile(
    r"Archive copied before this write:\s*`?([A-Za-z0-9_./~-]+\.md)`?")

#: A dated copy named anywhere in a report body. The time component is optional
#: because the family already holds `signals-latest-2026-08-31.md` beside
#: fourteen `…-HHMM` copies, and a single-format parser would skip it silently.
DATED_COPY = re.compile(
    r"(?<![A-Za-z0-9_-])((?:[A-Za-z0-9_.~-]+/)*[A-Za-z0-9_-]+-latest"
    r"-\d{4}-\d{2}-\d{2}(?:-\d{4})?\.md)")


@dataclass(frozen=True)
class CopyGap:
    """A report that names an archive copy the reflection directory lacks."""
    report: Path
    copy: str


def named_copies(body: str) -> list[str]:
    """Basenames of every archive copy `body` names, first mention first."""
    seen: dict[str, None] = {}
    for rx in (ARCHIVE_POINTER, DATED_COPY):
        for m in rx.finditer(body):
            seen.setdefault(Path(m.group(1)).name, None)
    return list(seen)


def copy_gaps(reflection_dir: Path, reports: Mapping[Path, str]) -> list[CopyGap]:
    """One `CopyGap` per (report, named copy) absent from `reflection_dir`.

    The directory is an argument so the check runs against a tmp fixture on the
    gate's `not live_vault` rung; the live directory is the caller's business
    (`scripts/memory/knowledge-health-report.py`).
    """
    reflection_dir = Path(reflection_dir)
    gaps = []
    for report in sorted(reports, key=str):
        for name in named_copies(reports[report]):
            if not (reflection_dir / name).is_file():
                gaps.append(CopyGap(Path(report), name))
    return gaps


def reports_in(reflection_dir: Path) -> dict[Path, str]:
    """The reports that carry copy pointers: the dated knowledge-write reports
    and the live `-latest` files. Dated copies themselves are left out — each
    one is a frozen earlier report whose pointer was judged in its own cycle."""
    reflection_dir = Path(reflection_dir)
    out: dict[Path, str] = {}
    for pattern in ("knowledge-write-*.md", "*-latest.md"):
        for p in reflection_dir.glob(pattern):
            if p.is_file():
                try:
                    out[p] = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
    return out
