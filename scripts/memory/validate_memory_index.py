#!/usr/bin/env python3
"""Is `lloyd/MEMORY.md` a bounded, typed index whose links resolve? Exit 1 if not.

Review 2026-09-24, P4. MEMORY.md is meant to be an index — one typed line per
entry, `- [type] … → topics/<slug>` — with the detail in
`lloyd/memory/<slug>.md` topic files that `memory_read(file="topics/<slug>")`
pulls on demand. Two jobs keep it that way and both run this:

* the dream-consolidation skill (#47) owns index tightness and runs it after its
  pass, refusing to call the pass done on exit 1;
* `tests/test_prompt_surface_budget.py`'s live section runs it over the live vault.

Two modes, because the index is built before it is deployed:

  structure  every `→ topics/<slug>` link resolves; every topic file has a legal
             slug and fits `TOPIC_FILE_CEILING_BYTES`; MEMORY.md fits its ceiling.
             Holds for today's un-indexed file, so it is what the live test runs
             until the index ceiling is deployed.
  full       structure, plus: MEMORY.md at or under `--tightness` (80%) of the
             ceiling, every top-level entry typed, every index line at or under
             `INDEX_LINE_MAX_CHARS`. What the dream pass must leave behind.

The ceiling defaults to `prompt_surface.MEMORY_MD_CEILING_BYTES` — the live one —
so the day it is lowered to `MEMORY_MD_INDEX_CEILING_BYTES` every caller of this
script is judged against the new number with no edit here. An eval overlay passes
`--ceiling 25600` explicitly.

Stdlib only (the dream skill runs it with the system python3), read-only, and it
never follows a link outside `<root>/memory/`.

Usage:
  python3 ~/lloyd/scripts/memory/validate_memory_index.py            # live, full
  python3 ~/lloyd/scripts/memory/validate_memory_index.py --mode structure
  python3 …/validate_memory_index.py --root <overlay> --ceiling 25600 --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prompt_surface as ps  # noqa: E402
from app import memory_ceiling as mc  # noqa: E402

#: Longest legal index line, in characters. An index line is a pointer with a
#: hook, not the entry: past this it is the detail again, paid for every turn.
INDEX_LINE_MAX_CHARS = 300

#: Share of the ceiling a consolidated index may fill. The rest is the room the
#: nightly writers append into before the next dream pass tightens it again.
DEFAULT_TIGHTNESS = 0.80

#: `→ topics/<slug>` (the arrow is the index's own grammar; `->` is accepted too).
LINK_RE = re.compile(r"(?:→|->)\s*topics/([A-Za-z0-9_./-]+?)(?:\.md)?(?=[\s,;)`]|$)")


def links(text: str) -> list[str]:
    """Every topic slug an index line links to, in file order (raw, unvalidated)."""
    return [m.group(1) for m in LINK_RE.finditer(ps.body(text))]


def check(root: Path, *, ceiling: int, mode: str = "full",
          tightness: float = DEFAULT_TIGHTNESS) -> dict:
    """The findings for the memory root `root`. `ok` is False on any violation."""
    errors: list[str] = []
    memory_md = root / "MEMORY.md"
    text = memory_md.read_text(encoding="utf-8") if memory_md.exists() else ""
    size = len(text.encode("utf-8"))
    report: dict = {"root": str(root), "mode": mode, "ceiling": ceiling,
                    "memory_md_bytes": size, "errors": errors}
    if not memory_md.exists():
        errors.append(f"{memory_md} does not exist")
    if size > ceiling:
        errors.append(f"MEMORY.md is {size:,} B, over the {ceiling:,} B ceiling")

    tdir = root / mc.TOPICS_SUBDIR
    topics = sorted(tdir.glob("*.md")) if tdir.is_dir() else []
    report["topic_files"] = len(topics)
    for p in topics:
        if not mc.TOPIC_SLUG_RE.fullmatch(p.stem):
            errors.append(f"topic file {p.name}: slug is not [a-z0-9-]{{1,48}}")
        tb = p.stat().st_size
        if tb > mc.TOPIC_FILE_CEILING_BYTES:
            errors.append(f"topic file {p.name} is {tb:,} B, over the "
                          f"{mc.TOPIC_FILE_CEILING_BYTES:,} B topic ceiling")

    linked = links(text)
    report["links"] = len(linked)
    for slug in sorted(set(linked)):
        if not mc.TOPIC_SLUG_RE.fullmatch(slug):
            errors.append(f"link → topics/{slug}: not a legal slug")
        elif not mc.topic_path(root, slug).is_file():
            errors.append(f"link → topics/{slug}: {mc.TOPICS_SUBDIR}/{slug}.md does not exist")

    if mode == "full":
        limit = int(ceiling * tightness)
        report["tight_limit"] = limit
        if size > limit:
            errors.append(f"MEMORY.md is {size:,} B, over {tightness:.0%} of its ceiling "
                          f"({limit:,} B) — the index is not consolidated")
        untyped = ps.untyped_entry_count(text)
        report["untyped_entries"] = untyped
        if untyped:
            errors.append(f"{untyped} top-level entries carry no [type] tag "
                          f"({', '.join(ps.ENTRY_TYPES)})")
        long = [ln for ln in ps.body(text).split("\n")
                if ln.startswith(("- ", "* ")) and len(ln) > INDEX_LINE_MAX_CHARS]
        report["long_lines"] = len(long)
        if long:
            errors.append(f"{len(long)} index lines over {INDEX_LINE_MAX_CHARS} chars; "
                          f"first: {long[0][:80]}…")
    report["ok"] = not errors
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=mc.MEMORIES_DIR,
                    help="memory root holding MEMORY.md and memory/ (default: live)")
    ap.add_argument("--ceiling", type=int, default=ps.MEMORY_MD_CEILING_BYTES)
    ap.add_argument("--mode", choices=("full", "structure"), default="full")
    ap.add_argument("--tightness", type=float, default=DEFAULT_TIGHTNESS)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = check(args.root, ceiling=args.ceiling, mode=args.mode,
                   tightness=args.tightness)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{'OK' if report['ok'] else 'FAIL'}: {report['root']}/MEMORY.md "
              f"{report['memory_md_bytes']:,} B / {report['ceiling']:,} B ceiling, "
              f"{report['topic_files']} topic files, {report['links']} links ({args.mode})")
        for e in report["errors"]:
            print(f"  - {e}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
