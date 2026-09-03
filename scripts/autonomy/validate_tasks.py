#!/usr/bin/env python3
"""validate_tasks.py — fail-loud linter for ~/obsidian/autonomy task files.

The autonomy scheduler (autonomy._parse_task_file) silently drops any task whose
frontmatter fails yaml.safe_load — the task vanishes from the schedule with no
failure record and no alert. On 2026-05-28 a bulk edit corrupted the `tags` field
of 34/40 task files (inline list followed by orphan block-list items), which
dormant-killed ~85% of the autonomy system for six days before anyone noticed.

This script makes that failure mode loud. Run it in CI / a healthcheck / before
any bulk edit of the autonomy dir. Exit codes:
    0  all task files parse and pass structural checks
    1  one or more task files are UNPARSEABLE (scheduler-invisible) — critical
    2  files parse but have structural problems (bad depends_on, no skill, ...)

Usage:
    python scripts/autonomy/validate_tasks.py [--autonomy-dir DIR] [--strict]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

DEFAULT_DIR = Path.home() / "obsidian" / "autonomy"
RUNNABLE_STATUSES = {"up_next", "in_progress", "draft", "paused", "archived", "done",
                     # `failed` = disabled after max_retries consecutive
                     # failures; a human re-enables it by setting up_next.
                     "failed", "archived"}


def _parse(path: Path):
    """Mirror autonomy._parse_task_file: split on '---\\n' and yaml.safe_load."""
    content = path.read_text(encoding="utf-8")
    parts = content.split("---\n", 2)
    if len(parts) < 3:
        return None, "no frontmatter (need leading '---' block)"
    try:
        fm = yaml.safe_load(parts[1])
    except Exception as e:  # noqa: BLE001 — we want the raw yaml error text
        first = str(e).splitlines()[0]
        return None, f"yaml parse error: {first}"
    if not isinstance(fm, dict):
        return None, "frontmatter is not a mapping"
    return fm, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autonomy-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--strict", action="store_true",
                    help="treat structural warnings as failures (exit 2)")
    args = ap.parse_args()

    d = Path(args.autonomy_dir).expanduser()
    if not d.is_dir():
        print(f"error: {d} is not a directory", file=sys.stderr)
        return 2

    # Task files follow the NN-name.md convention. Other .md files in the dir
    # (reports, notes) are ignored by the scheduler because _parse_task_file
    # returns None for them; we mirror that by only linting numbered files.
    files = sorted(
        (p for p in d.glob("*.md") if re.match(r"\d+-", p.name)),
        key=lambda p: int(re.match(r"\d+", p.name).group()),
    )

    unparseable: list[tuple[str, str]] = []
    parsed: dict[str, dict] = {}      # id -> fm
    by_file: list[tuple[Path, dict]] = []

    for p in files:
        fm, err = _parse(p)
        if err:
            unparseable.append((p.name, err))
            continue
        by_file.append((p, fm))
        tid = str(fm.get("id", "")).strip()
        if tid:
            parsed[tid] = fm

    warnings: list[str] = []
    for p, fm in by_file:
        status = str(fm.get("status", "") or "").strip()
        if status and status not in RUNNABLE_STATUSES:
            warnings.append(f"{p.name}: unknown status '{status}'")
        if status in ("up_next", "in_progress"):
            skill = str(fm.get("skill_name", "") or "").strip()
            spath = str(fm.get("skill_path", "") or "").strip()
            if not skill and not spath:
                warnings.append(f"{p.name}: runnable but has no skill_name/skill_path")
        dep = fm.get("depends_on")
        if dep and str(dep).strip().lower() not in ("", "null", "none"):
            if str(dep).strip() not in parsed:
                warnings.append(
                    f"{p.name}: depends_on '{dep}' not found among parseable tasks"
                )

    print(f"Scanned {len(files)} task files in {d}")
    print(f"  parseable:   {len(by_file)}")
    print(f"  UNPARSEABLE: {len(unparseable)}")

    if unparseable:
        print("\n🔴 CRITICAL — these tasks are INVISIBLE to the scheduler:")
        for name, err in unparseable:
            print(f"   {name}\n       {err}")

    if warnings:
        print(f"\n⚠️  {len(warnings)} structural warning(s):")
        for w in warnings:
            print(f"   {w}")

    if unparseable:
        return 1
    if warnings and args.strict:
        return 2
    print("\n✅ all task files parse" + (" and pass structural checks" if not warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
