#!/usr/bin/env python3
"""Validate the nightly reflection handoff markdown artifact.

Purpose: Catch structural drift in the knowledge-handoff-YYYY-MM-DD.md emitted
by the Knowledge Analysis stage BEFORE the Write stage tries to consume it.

Exit code 0 = valid; 1 = missing required sections or malformed bullets.

Intentionally forgiving: the 35B local model drifts on exact phrasing, so we
check for required section *presence* and flag parse issues — we do NOT reject
on cosmetic variation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

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

    if issues:
        print(f"[fail] {len(issues)} validation issue(s) in {path.name}:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print(f"[ok] {path.name} — structure valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
