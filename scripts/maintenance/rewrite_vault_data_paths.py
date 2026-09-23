#!/usr/bin/env python3
"""Point the vault's instructions at ~/lloyd-data instead of the code tree.

The runtime data moved out of ~/lloyd on 2026-09-22 (architecture/data-home.md).
Skills and autonomy tasks are instructions Lloyd follows, and ~80 of them name
`~/lloyd/_pipeline/...`, `~/lloyd/workers.db`, `~/lloyd/sessions` and so on. A
skill that still says `sqlite3 ~/lloyd/workers.db` creates an empty database in
the tree the first time it runs.

Only `autonomy/*.md` and `skills/**` are instructions. `backlog/`, `memory/`,
`knowledge/` and `projects/` are the record of what was true when they were
written, and are left alone.

    rewrite_vault_data_paths.py            # dry run: files and line counts
    rewrite_vault_data_paths.py --apply
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VAULT = Path.home() / "obsidian"
SCOPES = ("autonomy/*.md", "skills/**/*.md")

#: Names that moved, under their new relative path.
MOVED = {
    "_pipeline": "_pipeline",
    "sessions": "sessions",
    "event_logs": "event_logs",
    "autonomy-runs": "autonomy-runs",
    "logs": "logs",
    "eval/baselines": "eval/baselines",
    "voice_profiles": "voice_profiles",
    "usage.db": "usage.db",
    "workers.db": "workers.db",
    "research.db": "research.db",
    "mc-state.json": "mc-state.json",
    "data/tool_overrides.yaml": "data/tool_overrides.yaml",
    "agent-services/logs": "logs/services",
}
_NAMES = "|".join(re.escape(n) for n in sorted(MOVED, key=len, reverse=True))
PATTERN = re.compile(r"(~|\$HOME|\$\{HOME\}|/home/alansrobotlab)/lloyd/(" + _NAMES + r")(?![\w.-])")


def rewrite(text: str) -> tuple[str, int]:
    return PATTERN.subn(lambda m: f"{m.group(1)}/lloyd-data/{MOVED[m.group(2)]}", text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--vault", type=Path, default=VAULT)
    args = ap.parse_args(argv)
    files = sorted({p for s in SCOPES for p in args.vault.glob(s) if p.is_file()})
    total = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        new, n = rewrite(text)
        if not n:
            continue
        total += n
        print(f"{n:4d}  {f.relative_to(args.vault)}")
        if args.apply:
            f.write_text(new, encoding="utf-8")
    print(f"{total} references {'rewritten' if args.apply else 'would be rewritten'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
