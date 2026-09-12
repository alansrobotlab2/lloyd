#!/usr/bin/env python3
"""The only thing that appends to ``knowledge/_log.md`` — and the reason it can be trusted to.

Backlog #873. ``~/obsidian/knowledge/_log.md`` is the one change-history file Lloyd actually
has, and every one of the four things that make a log usable was broken at once: two entries
shared a single line (``…gaussian-splatting-vs-nerf.md## [2026-06-19] research | Real-time
depth estimation…``), the dates were not newest-first, every heading was ``## [YYYY-MM-DD]``
where OKF §9 requires ISO ``YYYY-MM-DD``, and the schema line pointing at the file named no
owner — so whoever wrote into it followed a template that produced exactly these defects. The
file had 10 entries; 342 commits had touched ``knowledge/`` since the last one was written.

Hand-editing a markdown file is not the problem, but hand-editing *this* file is, because a
missing newline is invisible in the writer and fatal in the reader: an append that forgets its
trailing newline does not fail, it silently joins two entries and makes every per-line count
disagree with every per-occurrence count. So the contract is: one owner
(``nightly-reflection-knowledge-write``, named in ``knowledge/KNOWLEDGE_SCHEMA.md``), and that
owner reaches the file through this module and nothing else.

Three guarantees, each pinned in ``tests/test_knowledge_log_contract.py``:

* **Newest-first.** The entry goes above the first heading older than it, so order is a property
  of the writer, not of the operator's memory.
* **One entry per line, always.** Body and kind are flattened to one line and a value that would
  open a second heading is refused rather than written; the file is written with exactly one
  trailing newline.
* **Checkable without prose.** ``check`` re-measures the two counts that disagreed (occurrences
  of ``"## "`` vs lines starting with ``"## "``), the bracketed form, the ordering and the
  trailing newline, and names the entry that broke.

Usage::

    python3 scripts/vault/knowledge_log.py append --kind nightly-reflection-knowledge-write \
        --text "run run_39_20260911_080010 — knowledge write: 2 new knowledge notes (…)"
    python3 scripts/vault/knowledge_log.py check
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

#: The live log. Not overridable by the environment on purpose: an owner that quietly
#: appended to a different file would look like a working contract while the real file kept
#: rotting. Tests pass ``--log`` / the ``path`` argument instead.
DEFAULT_LOG = Path(os.path.expanduser("~")) / "obsidian" / "knowledge" / "_log.md"

#: A conformant date heading: the ISO date and nothing else on the line.
DATE_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2})$")
#: The pre-#873 form OKF §9 does not allow, still prescribed by four skill templates
#: until this item retired them.
BRACKETED_HEADING = re.compile(r"^## \[\d{4}-\d{2}-\d{2}\]")
_HEADING_ANYWHERE = "## "
_HEADING_START = re.compile(r"^## ")
_KIND = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class LogShapeError(ValueError):
    """The value handed to ``append_entry`` could not be written without breaking the file."""


def _one_line(value: str, what: str) -> str:
    """Flatten to a single line, and refuse a value that would open a second heading."""
    flat = " ".join(value.split())
    if not flat:
        raise LogShapeError(f"{what} is empty")
    if flat.startswith("#"):
        raise LogShapeError(
            f"{what} starts with '#': it would open a second heading on the body line"
        )
    if _HEADING_ANYWHERE in flat:
        raise LogShapeError(
            f"{what} contains '#{'#'} ': a second heading would land on the entry's line, "
            "which is the defect that made _log.md unparseable"
        )
    return flat


def _valid_day(day: str) -> str:
    try:
        parsed = datetime.strptime(day, "%Y-%m-%d")
    except ValueError as exc:
        raise LogShapeError(f"date {day!r} is not ISO YYYY-MM-DD: {exc}") from exc
    return parsed.strftime("%Y-%m-%d")


def read_entries(path: Path) -> list[tuple[str, str]]:
    """Return ``[(date, body), …]`` in file order. A heading with no body yields ``""``."""
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out: list[tuple[str, str]] = []
    index: int | None = None
    for i, line in enumerate(lines):
        match = DATE_HEADING.match(line)
        if not match:
            continue
        if index is not None:
            out.append((lines[index][3:], " ".join(" ".join(lines[index + 1:i]).split())))
        index = i
    if index is not None:
        out.append((lines[index][3:], " ".join(" ".join(lines[index + 1:]).split())))
    return out


def _insert_index(lines: list[str], day: str) -> int:
    """Where a new entry belongs: above the first heading older than it, else at the end."""
    for i, line in enumerate(lines):
        match = DATE_HEADING.match(line)
        if match and match.group(1) < day:
            return i
    return len(lines)


def append_entry(path: Path, text: str, kind: str = "", day: str | None = None) -> str | None:
    """Insert one entry newest-first. Returns the heading written, or ``None`` if it is already there.

    Re-appending the same ``(date, body)`` is a no-op, so a retry of a run that already logged
    cannot produce a duplicate entry.
    """
    path = Path(path)
    if not path.exists():
        raise LogShapeError(f"log file does not exist: {path}")
    day = _valid_day(day or datetime.now().strftime("%Y-%m-%d"))
    if kind and not _KIND.match(kind):
        raise LogShapeError(f"kind {kind!r} must match {_KIND.pattern}")
    body = f"{kind} | {_one_line(text, 'text')}" if kind else _one_line(text, "text")

    entry_text = path.read_text(encoding="utf-8")
    for existing_date, existing_body in read_entries(path):
        if existing_date == day and existing_body == body:
            return None

    lines = entry_text.splitlines()
    at = _insert_index(lines, day)
    if at == len(lines):
        # Older than every entry: it goes below them, blank-separated from the last.
        block = ["", f"## {day}", body] if lines else [f"## {day}", body]
    else:
        block = [f"## {day}", body, ""]
    lines[at:at] = block
    # Exactly one trailing newline: an append that forgets it is what joined two entries once.
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    return f"## {day}"


def check(path: Path) -> list[str]:
    """Problems with the log's shape. Empty means the append contract still holds."""
    path = Path(path)
    if not path.exists():
        return [f"{path}: missing"]
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    problems: list[str] = []

    occurrences = raw.count(_HEADING_ANYWHERE)
    line_initial = sum(1 for line in lines if _HEADING_START.match(line))
    if occurrences != line_initial:
        problems.append(
            f"{occurrences} '#{'#'} ' occurrences vs {line_initial} heading lines: an entry "
            "shares a line with another"
        )
    for line in lines:
        if BRACKETED_HEADING.match(line):
            problems.append(f"{line}: bracketed date heading; OKF §9 wants '## YYYY-MM-DD'")

    dates = [m.group(1) for m in (DATE_HEADING.match(l) for l in lines) if m]
    if not dates:
        problems.append("no ISO date heading at all")
    else:
        if dates != sorted(dates, reverse=True):
            offenders = [
                f"{dates[i]} above {dates[i + 1]}"
                for i in range(len(dates) - 1)
                if dates[i] < dates[i + 1]
            ]
            problems.append(f"newest-first is broken: {'; '.join(offenders)}")
        if dates[0] != max(dates):
            problems.append(f"newest entry is {dates[0]} but {max(dates)} exists")
    for date, body in read_entries(path):
        if _HEADING_ANYWHERE in body:
            problems.append(f"{date}: body line carries a second heading")
    if raw and not raw.endswith("\n"):
        problems.append("file does not end with a newline: the next append would collide")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["append", "check"])
    parser.add_argument("--kind", default="", help="writer or job, e.g. nightly-reflection-knowledge-write")
    parser.add_argument("--text", default="", help="one-line description of the change")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="log file (default: the live vault log)")
    args = parser.parse_args(argv)
    path = Path(os.path.expanduser(args.log))

    if args.command == "check":
        problems = check(path)
        for problem in problems:
            print(f"SHAPE {problem}")
        if not problems:
            print(f"OK {path} — {len(read_entries(path))} entries, newest-first, one entry per line")
        return 1 if problems else 0

    try:
        written = append_entry(path, args.text, kind=args.kind, day=args.date)
    except LogShapeError as exc:
        print(f"REFUSED {exc}")
        return 1
    print(f"EXISTS ## {args.date or datetime.now().strftime('%Y-%m-%d')} (already logged)"
          if written is None else f"WROTE {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
