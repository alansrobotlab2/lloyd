#!/usr/bin/env python3
"""Validate the nightly reflection handoff markdown artifact.

Purpose: Catch structural drift in the knowledge-handoff-YYYY-MM-DD.md emitted
by the Knowledge Analysis stage BEFORE the Write stage tries to consume it.

Exit code 0 = valid; 1 = missing required sections, malformed bullets, or a
frontmatter `generated_at` that claims a time after the file was written.

Intentionally forgiving: the 35B local model drifts on exact phrasing, so we
check for required section *presence* and flag parse issues — we do NOT reject
on cosmetic variation. We are equally silent about a stamp that merely LAGS its
file's mtime: that is the ordinary amend-in-place case and most of the corpus is
that way. Only a stamp claiming to exist before the file does is a finding
(`check_clock`, #2610).
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# How far a `generated_at` may sit ahead of the file's own mtime before the stamp
# is dishonest rather than merely a few seconds early in the same call sequence.
# A tool call cannot interpolate a shell value into its own arguments, so the
# instant has to come from the call BEFORE the write — a `date -u` in the
# preceding call is the honest shape, and it lands a few seconds short of the
# write it describes. Zero tolerance would fail every real artifact for harness
# latency; 120 s is generous against what this exists to catch, which is a clock
# typed from a guess: +1063, +937 and +430 s in the three handoffs it would have
# failed, minutes off.
CLOCK_AHEAD_MAX_SECONDS = 120

# Required top-level (## H2) sections
REQUIRED_H2 = [
    # At least one `## Person: <name>` must exist — checked separately
    "## Vault Propagations",
    "## Tool Patterns — Failures",
    "## Tool Patterns — Successes",
    "## Conversation Patterns",
    "## Priority Order",
]

# Required H3 subsections under each `## Person:` block
REQUIRED_PERSON_H3 = [
    "### Mental Model — Decision Patterns",
    "### Mental Model — Communication Preferences",
    "### Mental Model — Technical Preferences",
    "### Mental Model — Project Prioritization",
    "### MEMORY.md Additions",
    "### MEMORY.md Removals",
    "### MEMORY.md Relocations",
    "### Profile Updates",
    "### Missing Files",
]

# Bullet patterns we care about (for sanity, not strict enforcement)
FREQUENCY_BARE_INT = re.compile(r"^\s*frequency:\s*(\d+)\s*$")
FREQUENCY_WITH_PAREN = re.compile(r"^\s*frequency:\s*(\d+)\s*[(\[].*")


def split_person_blocks(text: str) -> list[tuple[str, str]]:
    """Return list of (person_name, block_text) from ## Person: <name> sections."""
    blocks: list[tuple[str, str]] = []
    pattern = re.compile(r"^## Person:\s*(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Stop at next ## H2 (non-Person)
        tail = text[start:end]
        next_h2 = re.search(r"^##\s+(?!#)", tail, re.MULTILINE)
        if next_h2 and not tail[next_h2.start() : next_h2.start() + 10].startswith(
            "## Person:"
        ):
            tail = tail[: next_h2.start()]
        blocks.append((name, tail))
    return blocks


GENERATED_AT_KEY = re.compile(r"^generated_at:[ \t]*(.+?)[ \t]*$", re.MULTILINE)


def _parse_instant(raw: str) -> "datetime | None":
    """Parse a frontmatter instant, or None when it is not one.

    Three shapes occur in the live corpus: `...Z`, `...+00:00`, and (from a
    hand-typed pass) no offset at all. `datetime.fromisoformat` takes the first two
    natively on 3.12; `Z` is normalised because it is what most handoffs emit. A
    naive instant is read as UTC rather than local — this box runs at -0700, so
    reading it as local would move every honest stamp by seven hours and invent a
    failure out of a correct file, and UTC is what the key has always meant here.
    """
    text = raw.strip()
    if not text:
        return None
    if text[-1] in "zZ":
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def check_clock(path: Path) -> list[str]:
    """One finding when `generated_at` claims a time after the file exists.

    A comparison rather than a fourth prose rule, because the rules already
    existed and stopped nothing: three instructions across the reflection skills
    say never to type a clock by hand, and the validator still printed
    `[ok] … structure valid` and exited 0 over a stamp 1063 s inside its own file's
    future (#2610).

    Silent in the other three cases, each of which would be a fresh false failure.
    No `generated_at` key at all — structure does not require it, and signals
    reports name the same idea `generated:`. A stamp that will not parse — there is
    no instant there, so there is no claim to judge. And a stamp BEHIND its mtime by
    any amount, which is the ordinary amend-in-place and addendum-after-write case:
    nine of the fifteen real handoffs sit that way, so firing backwards would
    retroactively fail most of the corpus and stop the nightly chain. Bounding the
    stale direction (measured at -82,928 s in a signals report) is a separate
    ruling owed on #2610, not this check.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        # An unreadable file is main()'s own report; a clock finding on top of that
        # would be two messages about one broken read.
        return []
    match = GENERATED_AT_KEY.search(text)
    if not match:
        return []
    raw = match.group(1)
    generated = _parse_instant(raw)
    if generated is None:
        return []
    mtime = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    delta = int((generated - mtime).total_seconds())
    if delta <= CLOCK_AHEAD_MAX_SECONDS:
        return []
    return [
        f"frontmatter `generated_at` is {delta:+d} s ahead of the file's own mtime, "
        f"past the {CLOCK_AHEAD_MAX_SECONDS} s tolerance: generated_at={raw} vs "
        f"mtime={mtime.strftime('%Y-%m-%dT%H:%M:%SZ')}. A file cannot be generated "
        f"after it was last written, so that stamp was typed rather than read. Take "
        f"`date -u` in the call immediately before the write — a tool call cannot "
        f"interpolate a shell value into its own arguments, so the preceding call is "
        f"the closest honest instant — and re-validate."
    ]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_handoff.py <path-to-handoff.md>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1]).expanduser()
    if not path.exists():
        print(f"[fail] handoff artifact not found: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    issues: list[str] = []

    # Frontmatter sanity
    if not text.lstrip().startswith("---"):
        issues.append("missing YAML frontmatter delimiter at top (`---`)")

    # H1 title check
    if not re.search(r"^# Knowledge Handoff", text, re.MULTILINE):
        issues.append("missing `# Knowledge Handoff` H1 header")

    # At least one Person block
    person_blocks = split_person_blocks(text)
    if not person_blocks:
        issues.append("no `## Person: <name>` section found")

    # Required H2 sections
    for heading in REQUIRED_H2:
        if heading not in text:
            issues.append(f"missing required section: `{heading}`")

    # Per-person H3 sections
    for name, block in person_blocks:
        for h3 in REQUIRED_PERSON_H3:
            if h3 not in block:
                issues.append(f"person `{name}`: missing `{h3}`")

    # Frequency fields must be bare integers (catches "frequency: 4 (tasks ...)" drift)
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("frequency:"):
            if FREQUENCY_WITH_PAREN.match(line):
                issues.append(
                    f"line {lineno}: `frequency:` has non-integer trailing text "
                    f"({line.strip()!r}). Move narrative to a separate bullet."
                )
            elif not FREQUENCY_BARE_INT.match(line):
                issues.append(
                    f"line {lineno}: `frequency:` is not a bare integer "
                    f"({line.strip()!r})"
                )

    # The clock last and into the same list, not printed and returned early: a
    # file with a missing section AND a future stamp has two problems, and
    # reporting one teaches the reader to expect one (#2610).
    issues.extend(check_clock(path))

    if issues:
        print(f"[fail] {len(issues)} validation issue(s) in {path.name}:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print(f"[ok] {path.name} — structure valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
