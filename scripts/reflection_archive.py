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
place for the *enforcement*, for the reason ``app/prompt_surface.py`` documents:
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
from datetime import datetime, timezone
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
#: item of a **read** list in `autonomy-reflection-pipeline` Step 6.1 (archived
#: 2026-10-08, vault commit `42144a4b`), so a noun here would read a reader as a
#: writer. The example outlives the skill; the tense has to say so, and
#: `tests/test_skill_reflection_archive.py`
#: holds every sentence here that names a retired skill to that rule.
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
    `autonomy-reflection-pipeline` Step 6.1 (archived 2026-10-08, vault commit
    `42144a4b`) the read list was seven items long, while in
    `historical-knowledge-refresh` the write list nests two levels
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


# ── Pattern-file currency: the second directory check (#2413) ────────────────
#
# `copy_gaps` above asks one question — did a cycle lose a copy *its own report
# names*? — and deliberately never asks a calendar question, because pattern
# files are only rewritten on nights the run wrote them. That discipline is why
# it cannot see the shape #2413 caught: task 39's run of 2026-10-07 skipped §2e
# entirely, so it named no copy, `copy_gaps` returned nothing, and the
# `tool-patterns-latest.md` on disk still read "written 2026-10-06" while the run
# reported `status: success` with `stop_reason: "stop"`. The only thing that
# caught it was a step-conformance replay over a 14-day window.
#
# So this asks the question the pointer check structurally cannot: is the newest
# dated archive of a file §2e mandates archiving on *every* cycle older than the
# newest completed nightly cycle? That is a state comparison, not a per-report
# one, and it keeps reporting while the stale state persists rather than for the
# one cycle that caused it — which is the point: the 2026-10-07 gap was still
# live on 2026-10-08 when the replay caught it.
#
# Where the two checks disagree, this one is not contradicting the other. A night
# that wrote no pattern file at all makes no copy and loses nothing per
# `copy_gaps`, and still shows here until the next cycle rewrites the file. That
# is the cost of the shape, and it is bounded: the finding clears as soon as one
# §2e step runs.

#: The §2e pattern output files, named in
#: `skills/nightly-reflection-knowledge-write/steps-2b-2f.md` §2e, which requires
#: one `Read` + one `cp` + one `Write` per file on every cycle. Named as a
#: constant instead of scanned out of the skill for the reason in this module's
#: docstring: the vault is state no round under test controls, so a rule read from
#: it at check time would punish the next author for the previous writer's wording.
#: §2e names exactly these two; `signals-latest.md` is another skill's archive
#: contract (`nightly-reflection-signals` Phase 0) and is left to `copy_gaps`.
PATTERN_STEMS = ("tool-patterns-latest", "conversation-patterns-latest")

#: A dated archive copy of a governed stem, `<stem>-<YYYY-MM-DD>[-HHMM].md`. The
#: time is optional because the family already holds date-only names
#: (`signals-latest-2026-08-31.md`) beside `-HHMM` siblings, and a single-format
#: parser skips the other shape silently. Both are UTC, per §2e's `date -u` rule.
_STAMPED_COPY = re.compile(r"^(%s)-(\d{4})-(\d{2})-(\d{2})(?:-(\d{2})(\d{2}))?\.md$"
                           % "|".join(PATTERN_STEMS))

#: The report a completed nightly cycle leaves. Anchored on the date so
#: `knowledge-write-error-2026-09-24.md` — a cycle that died before §2e — is not
#: read as one that owed a pattern-file archive.
KNOWLEDGE_WRITE_REPORT = re.compile(r"^knowledge-write-(\d{4})-(\d{2})-(\d{2})\.md$")


@dataclass(frozen=True)
class PatternStaleness:
    """A governed pattern file whose newest archive predates the newest report.

    `newest_copy` is None when the directory holds no dated copy of the stem at
    all, which is the shape the chain lived in from 2026-08-22 to 2026-09-12
    (#436): nightly cycles, not one archive.
    """
    stem: str
    newest_copy: str | None
    report: str


def _utc_key(year: str, month: str, day: str,
             hour: str | None = None, minute: str | None = None) -> tuple[int, ...]:
    """Sortable UTC key for a filename stamp; a missing time reads as 00:00."""
    return (int(year), int(month), int(day), int(hour or 0), int(minute or 0))


def _listing(d: Path) -> list[str]:
    """The file names in `d`, or an empty list when `d` is not a readable directory.

    `copy_gaps` never had to ask this question: it only ever stats a name a report
    handed it, so a missing directory reads as "nothing named is missing". This check
    lists the tree, so the absent tree is its case to handle — a reflection directory
    that does not exist yet owes no archive, and must not raise out of a nightly
    report. `tests/test_knowledge_health_stale_facts.py` runs the report's `main()`
    against an empty tmp tree and caught exactly this.
    """
    try:
        return sorted(p.name for p in d.iterdir() if p.is_file())
    except OSError:
        return []


#: The floor a completion note has to clear before either §2e check may read it as
#: *the newest report*. Deliberately the same number as `app/autonomy.py`'s
#: `_ARTIFACT_MIN_BYTES`: the two readers are looking at one file, and a note small
#: enough to fail the engine's test of whether the run that wrote it produced anything
#: must not also be big enough to become the bar the next cycle is graded against.
REPORT_MIN_BYTES = 512

#: A completion note's own claim about its state, written by §Phase 0 of the
#: knowledge-write skill: `in-progress` while the run is live, then `complete` or
#: `incomplete`. Only the first is a claim that a writer is still at the keyboard.
_NOTE_STATUS = re.compile(r"(?m)^status:[ \t]*([A-Za-z-]+)[ \t]*$")
_NOTE_IN_PROGRESS = "in-progress"


class Unmeasured(list):
    """An empty result that does not mean "clean" — and still compares equal to empty.

    `pattern_staleness` and `pointer_mtime_lag` return findings, so an empty result is
    printed as *"0 — every §2e archive is current"*. That sentence is right when the
    check ran and found nothing, and wrong when the check had nothing to measure
    against. This carries the reason without changing the emptiness every existing
    reader and test compares against.
    """

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason


def _stub_reason(name: str, body: str) -> str | None:
    """Why note `name` is not a report, or None when it is one.

    Two tests, both taken from the note that motivated them: size, because the run of
    2026-10-30 (`run_39_20261030_080034`, stopped at `max_turns` after 91 turns against
    a cap of 90) left 68 bytes of heading where the four prior nights are
    8,291-13,494 bytes; and `status: in-progress`, the word that stub was still
    carrying, which claims a writer is still working here.
    """
    defects: list[str] = []
    size = len(body.encode("utf-8", errors="replace"))
    if size < REPORT_MIN_BYTES:
        defects.append(f"it holds {size} bytes, under the {REPORT_MIN_BYTES}-byte floor a "
                       "completion note has to clear to be read as a report")
    m = _NOTE_STATUS.search(body)
    if m and m.group(1).strip().lower() == _NOTE_IN_PROGRESS:
        defects.append(f"it still reads `status: {_NOTE_IN_PROGRESS}`, which claims a run "
                       "that has not finished")
    if not defects:
        return None
    return f"{name} is not a report: " + "; and ".join(defects)


def _newest_complete_note(names: list[str], reflection_dir: Path
                          ) -> tuple[str | None, tuple[int, ...] | None, str, str]:
    """The newest dated note that is a report: its (name, stamp key, body, why-not).

    Walks the dated notes newest first and returns the first that is not a stub, so the
    §2e bar is the last cycle that actually reported — which is what "the newest
    report" has always been taken to mean by the person reading the row. The fourth
    element is "" when a complete note was found, and otherwise the reason none was,
    naming the newest stub and how many were set aside, so the report can print a
    *not measured* line someone can act on instead of a row that reads clean. A
    directory with no dated note at all returns four empties, which is a first run or a
    swept tree, not a stub — those callers keep their existing silence.
    """
    dated = sorted(((_utc_key(*m.groups()), n)
                    for n in names if (m := KNOWLEDGE_WRITE_REPORT.match(n))),
                   key=lambda pair: pair[0], reverse=True)
    newest_stub = ""
    skipped = 0
    for key, name in dated:
        try:
            body = (reflection_dir / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        reason = _stub_reason(name, body)
        if reason is None:
            return name, key, body, ""
        skipped += 1
        newest_stub = newest_stub or reason
    if skipped:
        return (None, None, "",
                f"the {skipped} dated completion note(s) on disk are not reports: "
                + newest_stub)
    return None, None, "", ""


def _newest_matching(names: list[str], rx: re.Pattern[str]) -> tuple[str | None, tuple[int, ...] | None]:
    """The name in `names` matching `rx` with the highest stamp, and that stamp.

    Judged on the stamp in the name and never on mtime: §2e's own rule is that the
    stamp comes from `date -u`, and a copy restored by hand has an mtime that
    describes the restoring, not the cycle.
    """
    newest_name: str | None = None
    newest_key: tuple[int, ...] | None = None
    for name in names:
        m = rx.match(name)
        if not m:
            continue
        key = _utc_key(*m.groups())
        if newest_key is None or key > newest_key:
            newest_name, newest_key = name, key
    return newest_name, newest_key


def pattern_staleness(reflection_dir: Path) -> list[PatternStaleness]:
    """One finding per governed pattern file whose archive fell behind the cycles.

    The comparison is a `>=` on UTC keys: a copy stamped the same day as the newest
    report is current, because the report's name carries a date and §2e's stamp
    carries the hour, so "later that same night" and "the previous evening" are the
    same day and only the day is shared between the two name shapes.

    Two shapes never fire. A directory with no dated `knowledge-write-*` report has
    completed no cycle, so nothing is owed — a first run, or a tree whose reports
    have been swept, must read as silence and not as a gap. And a governed stem
    with no dated copy and only one report is a first cycle: §2e's `test -f` gate
    exists precisely because there is nothing to archive the first time. With two
    or more reports behind it, zero copies is the #436 shape and does fire.
    

    The bar is the newest COMPLETE note (`_newest_complete_note`), not the newest name.
    On 2026-10-30 the newest note was the 68-byte stub `run_39_20261030_080034` left
    reading `status: in-progress`, and grading the copies against that day printed
    "behind their newest report: 3" for archives that were in fact current — a run that
    landed nothing had silently moved the bar forward a day. With no complete note at
    all the result is an `Unmeasured`, which is empty to every existing reader and
    carries the reason for the report to print instead of a zero.
    """
    reflection_dir = Path(reflection_dir)
    names = _listing(reflection_dir)
    report, report_key, _body, unmeasured = _newest_complete_note(names, reflection_dir)
    if report is None or report_key is None:
        return Unmeasured(unmeasured) if unmeasured else []
    reports = sum(1 for n in names if KNOWLEDGE_WRITE_REPORT.match(n))
    out: list[PatternStaleness] = []
    for stem in PATTERN_STEMS:
        copy_rx = re.compile(
            r"^%s-(\d{4})-(\d{2})-(\d{2})(?:-(\d{2})(\d{2}))?\.md$" % re.escape(stem))
        if f"{stem}.md" not in names:
            continue
        copy, copy_key = _newest_matching(names, copy_rx)
        if copy_key is None:
            if reports >= 2:
                out.append(PatternStaleness(stem, None, report))
            continue
        if copy_key < report_key:
            out.append(PatternStaleness(stem, copy, report))
    return out


#: A governed pattern pointer, matched by shape rather than by name. `PATTERN_STEMS`
#: cannot be the enumeration here: it holds `tool-patterns-latest` and
#: `conversation-patterns-latest` only, and the one pointer measured stale on the live
#: tree (#2832, five days behind on 2026-10-18) is `tool-failure-patterns-latest.md`,
#: which is not in that tuple. Whether that file is §2e-governed is a separate ruling;
#: this check reports what the filesystem says about every pointer of the shape.
_POINTER_LATEST = re.compile(r"^(.+-patterns-latest)\.md$")

#: A completed cycle's note prints one of these per pointer it advanced (§2e). The
#: count is the witness that the pointer write happened; the pointers' mtimes are the
#: measurement, and this is the report that says whether the note said so.
MTIME_WITNESS_MARKER = "mtime="

#: §2e writes two governed pointers, so a full cycle's note carries two witness lines.
EXPECTED_MTIME_WITNESSES = 2


@dataclass(frozen=True)
class PointerMtimeLag:
    """A pattern pointer whose mtime is older than the newest completion note's day.

    `stem` is the pointer's name without `.md`, `mtime_date` the UTC date its mtime
    carries, `report` the newest note's file name and `report_day` the date inside it.
    """
    stem: str
    mtime_date: str
    report: str
    report_day: str


@dataclass(frozen=True)
class MtimeWitness:
    """How many `mtime=` lines the newest completion note carries.

    `note` is None when the directory holds no readable dated note at all, which the
    report must render as unmeasured rather than as zero — a note that could not be
    read is not a note that said nothing.
    """
    note: str | None
    count: int


def pointer_mtime_lag(reflection_dir: Path) -> list[PointerMtimeLag]:
    """One finding per `*-patterns-latest.md` whose mtime predates the newest report.

    This is not `pattern_staleness` and does not replace it. That check compares the
    stamp in an archive copy's NAME against the stamp in the report's name and never
    stats the pointer — deliberately, per `_newest_matching`, because a hand-restored
    copy's mtime describes the restoring. The cost is #2832: reproducing it took a bare
    `cp` of a stem, which makes the NAME current and writes no pointer, and the report
    then printed "Pattern files behind the newest report: 0 — every §2e archive is
    current." over a pointer five days stale. The §2e `mtime=` lines were supposed to
    be the second witness and printed zero times on three consecutive cycles, so this
    row reads the one fact that cannot be satisfied by a file name: the pointer's own
    mtime.

    Read-only — it lists and stats, and never writes or `utime`s anything, because it
    runs inside a nightly report over the live corpus. Like `pattern_staleness`, a
    directory with no dated `knowledge-write-*` note, or none at all, returns nothing:
    no completed cycle means no pointer is behind anything. Also like
    `pattern_staleness`, the day it compares against is the newest COMPLETE note: the
    2026-10-30 stub made its own day the bar, so two pointers that were current against
    the last real report read as lag because a run that wrote nothing had stamped the
    directory with a new note.
    """
    reflection_dir = Path(reflection_dir)
    names = _listing(reflection_dir)
    report, report_key, _body, unmeasured = _newest_complete_note(names, reflection_dir)
    if report is None or report_key is None:
        return Unmeasured(unmeasured) if unmeasured else []
    report_day = "%04d-%02d-%02d" % report_key[:3]
    out: list[PointerMtimeLag] = []
    for name in names:
        m = _POINTER_LATEST.match(name)
        if not m:
            continue
        try:
            stamp = (reflection_dir / name).stat().st_mtime
        except OSError:
            continue
        mtime_day = datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%d")
        if mtime_day < report_day:
            out.append(PointerMtimeLag(m.group(1), mtime_day, report, report_day))
    return out


def mtime_witness(reflection_dir: Path) -> MtimeWitness:
    """Count the `mtime=` witness lines in the newest completion note.

    The pointer mtimes above are the measurement; this counts how loudly the cycle that
    produced them said so. Zero means the note is silent about writes that may or may
    not have happened, which is a reporting gap and never a statement about the corpus.
    A directory with no COMPLETE dated note — none at all, one that cannot be read, or
    only stubs — returns `note=None` so the report can say *not measured* instead of
    reading a missing note as a silent one (#1227). Counting a stub's lines was the
    third half of the 2026-10-30 shape: the row printed "0 of 2 expected" against a
    68-byte note whose silence means only that the run stopped before writing anything,
    which is not a statement about the note's author having a reporting gap.
    """
    reflection_dir = Path(reflection_dir)
    report, _, body, _why = _newest_complete_note(_listing(reflection_dir), reflection_dir)
    if report is None:
        return MtimeWitness(None, 0)
    return MtimeWitness(report, body.count(MTIME_WITNESS_MARKER))
