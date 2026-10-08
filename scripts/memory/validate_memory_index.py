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

  structure  every `→ topics/<slug>` link resolves, and every topic file under
             `memory/` is named by at least one index line; every topic file has a
             legal slug and fits `TOPIC_FILE_CEILING_BYTES`; MEMORY.md fits its
             ceiling. Holds for today's un-indexed file, so it is what the live test
             runs until the index ceiling is deployed.
  full       structure, plus: MEMORY.md at or under `--tightness` (80%) of the
             ceiling, every top-level entry typed, every index line at or under
             `INDEX_LINE_MAX_CHARS`. What the dream pass must leave behind.

Both directions of the link are checked, and the reverse one is the whole point of
the pair. The index is the only route a topic file has into a prompt — a
`→ topics/<slug>` hook line is what `memory_read` is reached by — so a file on disk
that no index line names is a rule that exists and is never applied. Checking only
index→topic let exactly that happen: dream consolidation #47 found six such files
on 2026-10-07 (five class rules the nightly knowledge write shipped 2026-10-01 →
10-06, plus `memory-md-ledger`), unread by every prompt for a week while this script
printed `OK` nightly (#2399). So an unlinked topic file is an error in both modes,
with no baseline exemption for files already orphaned when the check was written —
the count over the live vault is 0, and a warning nobody acts on is how the six
became six.

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

from app import prompt_surface as ps  # noqa: E402
from app import memory_ceiling as mc  # noqa: E402

# Both bounds are DEFINED under `app/` (#1895) — `app/memory_ceiling.py` is the
# write guard that has to refuse the same two conditions this script reports, and
# the day the numbers live here the writer and the reader disagreed by construction
# and only the nightly run noticed. Re-exported under the old names so
# `consolidate_memory_index.py`, this file's own `--tightness` default and every
# caller keep resolving them, and so `git grep -n INDEX_LINE_MAX_CHARS` now points
# at `app/` first.
INDEX_LINE_MAX_CHARS = mc.INDEX_LINE_MAX_CHARS

#: Share of the ceiling a consolidated index may fill. The rest is the room the
#: nightly writers append into before the next dream pass tightens it again.
DEFAULT_TIGHTNESS = mc.MEMORY_TIGHTNESS

#: `→ topics/<slug>` (the arrow is the index's own grammar; `->` is accepted too).
LINK_RE = re.compile(r"(?:→|->)\s*topics/([A-Za-z0-9_./-]+?)(?:\.md)?(?=[\s,;)`]|$)")


def links(text: str) -> list[str]:
    """Every topic slug an index line links to, in file order (raw, unvalidated)."""
    return [m.group(1) for m in LINK_RE.finditer(ps.body(text))]


def _memory_ledger():
    """The sibling ledger instrument, imported lazily and by path.

    Lazy and by path, not `import memory_ledger`: the module lives in this same
    directory, which is not on `sys.path` for a caller that imports this file as
    `scripts.memory.validate_memory_index` or runs it as a script from elsewhere, and
    the dream skill invokes it with the system python3, so a top-level import would
    put a second file's import errors in front of the check that is supposed to
    report on the first. Every reader of a ledger in this repo asks
    `memory_ledger.status()`; the two-readers defect is what #2212 was about, so this
    one does not re-implement the anchor join.

    A function, not a module-level alias, so a caller can substitute it — the count
    crossing this boundary is the thing #2415's test pins.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "memory_ledger.py"
    spec = importlib.util.spec_from_file_location("memory_ledger_for_validator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ledger_counts(root: Path) -> tuple[dict, str]:
    """Per loaded index file, what its live ledger says — never a verdict, never fatal.

    Returns `(report, why_unread)`. `why_unread` is non-empty when the read itself
    failed, because a validator that dies on a missing sibling script or a vault
    half-written by the nightly would be a worse instrument than the blindness it is
    fixing, and a silent absence is worse still: an absent count is exactly how
    `orphan rows: 28` read for a day, as a check that printed nothing.

    The counts are REPORT-ONLY by decision, not by omission (#2415 owed ruling 2):
    nothing here appends to `errors`, so the exit status is whatever the byte, link
    and topic findings already produced. A fatal count on the day this landed would
    have failed every nightly from dream #47's fold onward for a defect in a file this
    check does not even police — the correct sequence is to print first, let one fold
    consume the number, then decide whether nonzero means red.
    """
    try:
        return _memory_ledger().status(root), ""
    except Exception as exc:  # noqa: BLE001 — report-only, so nothing to re-raise
        return {}, f"orphan rows unread ({type(exc).__name__}: {exc})"


#: The shape `report["ledger"]` carries, and the only shape `_ledger_suffix` reads:
#: `status()`'s lists already reduced to counts, so the JSON report and the printed
#: line cannot disagree about what a number means.
def _ledger_view(counts: dict) -> dict:
    view = {}
    for name, rep in counts.items():
        view[name] = ({"missing": True} if rep.get("missing") else
                      {"bytes": rep["bytes"], "entries": rep["entries"],
                       "rows": rep["rows"], "orphan_rows": len(rep["orphan_rows"])})
    return view


def _ledger_suffix(ledger: dict, name: str = "MEMORY.md") -> str:
    """` — ledger: 32 rows, 0 orphan rows`, or the reason there is none, to hang off
    the byte figure this script already prints. The two numbers ride one string
    because the fold's failure was a reader who stopped at the bytes: `19,135 B` and
    `OK` were both true on the night 28 rows detached."""
    if not ledger:
        return ""
    rep = ledger.get(name) or {}
    if not rep or rep.get("missing"):
        return " — ledger: no loaded text for this file"
    return (f" — ledger: {rep['rows']} rows, {rep['orphan_rows']} orphan rows "
            f"(report-only)")


def _ledger_block(ledger: dict, unread: str) -> list[str]:
    """One indented line per loaded index file — the count beside that file's own
    byte figure, including for `USER.md`, whose bytes this script never otherwise
    prints and which is the file the curation trigger actually fires on."""
    if unread:
        return [f"  ledger: {unread}"]
    return [f"  {name}: no loaded text at this root" if rep.get("missing") else
            f"  {name}: {rep['bytes']:,} B, {rep['entries']} entries, {rep['rows']} "
            f"ledger rows, {rep['orphan_rows']} orphan rows"
            for name, rep in ledger.items()]


def check(root: Path, *, ceiling: int, mode: str = "full",
          tightness: float = DEFAULT_TIGHTNESS) -> dict:
    """The findings for the memory root `root`. `ok` is False on any violation."""
    errors: list[str] = []
    memory_md = root / "MEMORY.md"
    text = memory_md.read_text(encoding="utf-8") if memory_md.exists() else ""
    size = len(text.encode("utf-8"))
    # Read once, over the same root, and carried in the report so `main` prints from
    # the same numbers the findings quote rather than asking a second instrument a
    # second question about a tree that may have moved between the two.
    counts, ledger_unread = _ledger_counts(root)
    report: dict = {"root": str(root), "mode": mode, "ceiling": ceiling,
                    "memory_md_bytes": size, "errors": errors}
    ledger = {} if ledger_unread else _ledger_view(counts)
    if ledger_unread:
        report["ledger_unread"] = ledger_unread
    else:
        report["ledger"] = ledger
    suffix = _ledger_suffix(ledger)
    if not memory_md.exists():
        errors.append(f"{memory_md} does not exist")
    if size > ceiling:
        errors.append(f"MEMORY.md is {size:,} B, over the {ceiling:,} B ceiling"
                      + suffix)

    tdir = root / mc.TOPICS_SUBDIR
    topics = sorted(tdir.glob("*.md")) if tdir.is_dir() else []
    report["topic_files"] = len(topics)
    for p in topics:
        if not mc.TOPIC_SLUG_RE.fullmatch(p.stem):
            errors.append(f"topic file {p.name}: slug is not [a-z0-9-]{{1,48}}")
        tb = p.stat().st_size
        # #2212: ask for the bound, do not copy it. A ledger is measured at
        # LEDGER_MULTIPLIER x the ceiling of the file it audits, and this report is
        # the surface a nightly reads — comparing here against the shared constant
        # would have flagged red the very ledger this change exists to let finish,
        # which is #1010's "one constant" rule broken in a second place.
        topic_bound = mc.topic_ceiling(f"{mc.TOPIC_PREFIX}{p.stem}")
        if tb > topic_bound:
            errors.append(f"topic file {p.name} is {tb:,} B, over the "
                          f"{topic_bound:,} B topic ceiling")

    linked = links(text)
    report["links"] = len(linked)
    for slug in sorted(set(linked)):
        if not mc.TOPIC_SLUG_RE.fullmatch(slug):
            errors.append(f"link → topics/{slug}: not a legal slug")
        elif not mc.topic_path(root, slug).is_file():
            errors.append(f"link → topics/{slug}: {mc.TOPICS_SUBDIR}/{slug}.md does not exist")

    # The reverse direction, which this function did not have until #2399: a set
    # difference over the same two collections the loop above already reads. It is
    # the direction that matters to a READER — an index line pointing at nothing
    # loses one detail, an unlinked file loses a whole rule silently, which is how
    # five class rules and `memory-md-ledger` sat unread for a week. No baseline:
    # the live count is 0, so an exemption would only ever shelter a new orphan.
    unlinked = sorted({p.stem for p in topics} - set(linked))
    report["topic_files_unlinked"] = len(unlinked)
    report["unlinked_topic_files"] = unlinked
    for slug in unlinked:
        errors.append(f"topic file {mc.TOPICS_SUBDIR}/{slug}.md: no index line links "
                      f"it (add a line ending → topics/{slug})")

    if mode == "full":
        limit = int(ceiling * tightness)
        report["tight_limit"] = limit
        if size > limit:
            # The other byte figure this script prints, and the one the fold actually
            # reads on a night it consolidates: a size finding quoted without the row
            # count is how `20,458 → 19,135 B` looked like the whole story.
            errors.append(f"MEMORY.md is {size:,} B, over {tightness:.0%} of its ceiling "
                          f"({limit:,} B) — the index is not consolidated" + suffix)
        untyped = ps.untyped_entry_count(text)
        report["untyped_entries"] = untyped
        if untyped:
            errors.append(f"{untyped} top-level entries carry no [type] tag "
                          f"({', '.join(ps.ENTRY_TYPES)})")
        # The same predicate the write guard refuses on, not a copy of it.
        long = mc.overlong_index_lines(text)
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
        # The unlinked count rides the same line as the linked one, so `0 unlinked`
        # is readable as "every file is hooked" and not as "the check did not run".
        # #2415: the ledger's orphan count rides it too, and the per-file block under
        # it exists so `USER.md`, whose bytes appear nowhere else in this output,
        # cannot be the one loaded file nobody counts — it is the file the curation
        # trigger actually fires on, and it is the one whose ledger had 0 orphans.
        print(f"{'OK' if report['ok'] else 'FAIL'}: {report['root']}/MEMORY.md "
              f"{report['memory_md_bytes']:,} B / {report['ceiling']:,} B ceiling, "
              f"{report['topic_files']} topic files, {report['links']} links, "
              f"{report['topic_files_unlinked']} unlinked ({args.mode})"
              + _ledger_suffix(report.get("ledger", {})))
        for line in _ledger_block(report.get("ledger", {}),
                                  report.get("ledger_unread", "")):
            print(line)
        for e in report["errors"]:
            print(f"  - {e}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
