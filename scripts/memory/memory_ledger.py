#!/usr/bin/env python3
"""Coverage of the loaded-memory rationale ledger (#1488), and its retire step (#1996).

`status` is read-only. `retire` is the one writer here, and it writes only the
ledgers and their archive — never a loaded file.

`lloyd/USER.md` and the `lloyd/MEMORY.md` index are loaded into every prompt
and sit at a fixed byte ceiling (`prompt_surface`). The proposal in
`scripts/maintenance/vault-user-md-rationale-curation.patch` gives every loaded
line a ledger row — why it is loaded, where it came from, and the checkable
condition that would make it false — in a topic file the prompt never loads
(`lloyd/memory/user-md-ledger.md`, `lloyd/memory/memory-md-ledger.md`), so the
nightly curator can make headroom by retiring the lines whose reason no longer
holds instead of refusing the next addition.

This script is the deterministic half the curator runs first. `status` reads,
never writes — the loaded files are written only through the memory tools and
`Edit`, which enforce the ceiling — and answers three questions:

* which loaded lines have no ledger row (backfill owed);
* which loaded lines have an anchor that NO row can carry (reported apart,
  because they are not curator backlog — writing more rows changes nothing);
* which ledger rows name no line that is still loaded (an orphan: the line was
  edited or removed without its row following);
* how far the file is from its ceiling, and which rows are the stalest checked.

A row is one bullet whose first field is the line's anchor, the first
`ANCHOR_CHARS` characters of the entry text after the bullet marker. The field
runs to the first `FIELD_SEP`, the delimiter every later field already uses, and
is written unwrapped:

    - anchor: **Scope**: agent memory, knowledge and research notes go in | why: … | origin: … | retire_when: … | check: `…` | checked: 2026-09-25

The delimiter is not a backtick, and #1730 is why: an anchor is the entry's own
prose, which in these two files is full of code spans, so a backtick-delimited
capture silently truncated every anchor that crossed one — 19 of the 54 loaded
`USER.md` lines and 24 of the 85 `MEMORY.md` lines — and a row written in good
faith from `anchor_of()`'s output came back shorter than it went in and read as
a permanent orphan. Two shapes still cannot be written, so `status` counts them
as `unrepresentable` rather than as lines awaiting a row: an anchor containing
the delimiter, and one that opens AND closes on a backtick, which is
indistinguishable from the legacy backtick-wrapped row form that
`_anchor_field` still unwraps.

    python3 scripts/memory/memory_ledger.py status            # both files
    python3 scripts/memory/memory_ledger.py status --json

**The retire step (#1996): the ledger is bounded by lifecycle, and this step makes
no room.** A ledger lives in the topic directory, but since #2212 it is not measured
by the topic ceiling every other file there gets: `app/memory_ceiling.ledger_ceiling`
derives its bound as `LEDGER_MULTIPLIER` times the ceiling of the file it audits
(76,800 B for `memory-md-ledger`, 49,152 B for `user-md-ledger`), because a row is
written per loaded line and the old shared bound would have stopped the audit with
lines still uncovered. Its only bound is still `memory_add`'s `topic_size_error`
refusal — now a message that names that derived bound and says not to split the
file. `retire` moves a row out of the live
ledger into an archive under `lloyd/reviews/` — outside the topic directory, so
the archive is not ceiling-checked and `LEDGERS` cannot read it — when, and only
when, that row's anchor no longer joins any line in the loaded file: the line was
retired, relocated or rewritten, and the row is an orphan that covers nothing.
That is the one retirement that needs no judgement. Deciding that a row's
`retire_when` has happened means running its `check` and is the curator's call
(step-2a-ter-curation §2-§3), which deletes the row by hand when it retires the
line; what it leaves behind is what this step sweeps.

    python3 scripts/memory/memory_ledger.py retire            # both ledgers
    python3 scripts/memory/memory_ledger.py retire --dry-run  # print, move nothing

What `retire` is NOT, and must never become: a way to make room. #1881's rule is
"never make room" — do not shorten anchors, drop fields, rewrite the ledger
shorter or open a second live ledger to fit another row — and it stands. `retire`
takes no size, no target and no "until it fits" flag; it is never called from a
write path; a row whose line is still loaded is never moved however full the
ledger is; and a ledger write that would cross the ceiling is still refused while
archivable rows sit in the file. Do not add such a flag later. `LEDGERS` keeps
exactly one live file per loaded file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.memory_ceiling import tight_limit  # noqa: E402

#: How much of an entry identifies it. Long enough that two entries in one
#: file never share it (checked: 0 collisions in USER.md and MEMORY.md on
#: 2026-09-25), short enough that an edit to the end of a line keeps its row.
ANCHOR_CHARS = 60
#: Below the file's TIGHT limit (`app.memory_ceiling.tight_limit` — 80% of its
#: ceiling for MEMORY.md, the line `validate_memory_index.py --mode full` holds the
#: index to) the curator runs, so it acts while there is still room to write: one
#: KiB of headroom (#1895 moved the trigger down from the full ceiling, which sat
#: 4,096 B past the line the index is already red over — live `status` printed no
#: marker for a 19,918 B MEMORY.md with 562 B left before the next append trips
#: the validator). Every file that printed CURATE under the old trigger still
#: prints it: this only widens who is asked.
HEADROOM_BYTES = 1024

LEDGERS = {"USER.md": "user-md-ledger.md", "MEMORY.md": "memory-md-ledger.md"}
#: What ends the anchor field. A pipe, not a backtick: the anchor is the loaded
#: line's own first 60 characters, code spans and all, and every later field in
#: the row is already pipe-separated (#1730 — see the module docstring).
FIELD_SEP = "|"
# `memory_add` writes `- [project] (YYYY-MM-DD) <entry>`, so the type tag and
# date stamp it prepends are optional in front of `anchor:`.
_ROW = re.compile(r"^\s*-\s+(?:\[[a-z]+\]\s+)?(?:\(\d{4}-\d{2}-\d{2}\)\s+)?"
                  r"anchor:\s*(?P<field>[^|]*)(?:\s*\|\s*(?P<rest>.*))?$")
_CHECKED = re.compile(r"checked:\s*(\d{4}-\d{2}-\d{2})")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def anchor_of(entry_line: str) -> str:
    """The anchor for one loaded entry line."""
    body = entry_line.strip()
    body = re.sub(r"^(?:[-*]|>)\s*", "", body)
    return _norm(body)[:ANCHOR_CHARS].rstrip()


def _anchor_field(field: str) -> str:
    """The anchor a row's `anchor:` field carries.

    Rows written before #1730 wrapped the anchor in backticks, so one pair of
    wrapping backticks is stripped to keep those rows covering their line. The
    strip is greedy on purpose: for an anchor that ends on a code-span backtick
    the old wrapper's closing backtick WAS the anchor's own last character, and
    a non-greedy strip would take that character away as well.
    """
    f = field.strip()
    if len(f) >= 2 and f.startswith("`") and f.endswith("`"):
        return f[1:-1]
    return f


def representable(anchor: str) -> bool:
    """Whether a row can carry `anchor` and `ledger_rows` read it back whole.

    Two shapes cannot: one containing the field delimiter, which the row parser
    cuts at, and one that both opens and closes on a backtick, which no reader
    can tell from the legacy wrapped form. A line with either is not curator
    backlog — no row will ever cover it — so `status` counts it separately.
    """
    return FIELD_SEP not in anchor and not (anchor.startswith("`")
                                            and anchor.endswith("`"))


def loaded_entries(text: str) -> list[str]:
    """Bullet lines outside the front matter: the entries a ledger row covers.

    Blockquote paragraphs (USER.md's header) are the file's own rules about
    itself and carry no row."""
    out, in_fm = [], False
    for i, line in enumerate(text.splitlines()):
        if i == 0 and line.strip() == "---":
            in_fm = True
            continue
        if in_fm:
            in_fm = line.strip() != "---"
            continue
        if line.lstrip().startswith("- "):
            out.append(line)
    return out


def ledger_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        m = _ROW.match(line)
        if m:
            c = _CHECKED.search(m.group("rest") or "")
            rows.append({"anchor": _norm(_anchor_field(m.group("field"))),
                         "checked": c.group(1) if c else None})
    return rows


def status(memories_dir: Path) -> dict:
    from app.prompt_surface import memory_ceiling
    # `tight_limit` derives the same number the write guard refuses growth past
    # and the validator reports the index red over — one definition, three readers.
    report = {}
    for name, ledger_name in LEDGERS.items():
        path = memories_dir / name
        if not path.exists():
            report[name] = {"missing": True}
            continue
        text = path.read_text(encoding="utf-8")
        size = len(text.encode("utf-8"))
        ceiling = memory_ceiling(name)
        entries = [anchor_of(e) for e in loaded_entries(text)]
        lpath = memories_dir / "memory" / ledger_name
        rows = ledger_rows(lpath.read_text(encoding="utf-8")) if lpath.exists() else []
        have = {r["anchor"] for r in rows}
        live = set(entries)
        # A line whose anchor no row can carry is not backfill owed: keeping it
        # out of `without_row` is what stops the curator's bounded work being
        # spent re-attempting a row that would orphan the moment it landed.
        unrepresentable = [a for a in entries if not representable(a)]
        unreachable = set(unrepresentable)
        unchecked = sorted((r for r in rows if r["anchor"] in live),
                           key=lambda r: r["checked"] or "")
        limit = tight_limit(name)
        report[name] = {
            "bytes": size, "ceiling": ceiling, "tight_limit": limit,
            "headroom": None if ceiling is None else ceiling - size,
            "curate": limit is not None and size > limit - HEADROOM_BYTES,
            "entries": len(entries), "ledger": str(lpath), "rows": len(rows),
            "without_row": [a for a in entries if a not in have and a not in unreachable],
            "unrepresentable": unrepresentable,
            "orphan_rows": sorted(have - live),
            "stalest_checked": [r["anchor"] for r in unchecked[:10]],
            "duplicate_anchors": sorted({a for a in entries if entries.count(a) > 1}),
        }
    return report


#: Where retired rows go: beside the curator's other archives (§3 of the curation
#: step), outside `lloyd/memory/` so the topic ceiling does not apply to it and
#: nothing in `LEDGERS` reads it back.
ARCHIVE_SUBDIR = "reviews"


def archive_path(memories_dir: Path, ledger_name: str) -> Path:
    return memories_dir / ARCHIVE_SUBDIR / f"{Path(ledger_name).stem}-archive.md"


def retire(memories_dir: Path, *, dry_run: bool = False, today: str | None = None) -> dict:
    """Move orphan rows out of each live ledger into its archive.

    A row is eligible only when its anchor joins no line in the loaded file. A row
    whose line is still loaded is kept, byte for byte, whatever the ledger's size.
    Lines of the ledger that are not rows (the heading, blank lines, notes) are
    never touched. The archive is appended to BEFORE the ledger is rewritten, so a
    crash between the two leaves a row in both places rather than in neither.

    Returns `{loaded file: {"moved": [anchor…], "kept": n, "archive": path}}`.
    """
    from datetime import date
    stamp = today or date.today().isoformat()
    report = {}
    for name, ledger_name in LEDGERS.items():
        loaded, lpath = memories_dir / name, memories_dir / "memory" / ledger_name
        if not loaded.exists() or not lpath.exists():
            # No loaded file means no line can be shown to be gone: retire nothing.
            report[name] = {"moved": [], "kept": 0, "archive": None,
                            "skipped": "no loaded file" if not loaded.exists() else "no ledger"}
            continue
        live = {anchor_of(e) for e in loaded_entries(loaded.read_text(encoding="utf-8"))}
        text = lpath.read_text(encoding="utf-8")
        keep_lines, moved_lines, moved, kept = [], [], [], 0
        for line in text.splitlines(keepends=True):
            m = _ROW.match(line.rstrip("\n"))
            if m is None:
                keep_lines.append(line)
                continue
            anchor = _norm(_anchor_field(m.group("field")))
            if anchor in live:
                keep_lines.append(line)
                kept += 1
            else:
                moved_lines.append(line if line.endswith("\n") else line + "\n")
                moved.append(anchor)
        apath = archive_path(memories_dir, ledger_name)
        if moved and not dry_run:
            apath.parent.mkdir(parents=True, exist_ok=True)
            head = "" if apath.exists() else (
                f"# {Path(ledger_name).stem} — retired rows\n\n"
                f"Rows `scripts/memory/memory_ledger.py retire` moved out of "
                f"`lloyd/memory/{ledger_name}` because the line they anchored is no "
                f"longer in `lloyd/{name}`. Verbatim; nothing reads this file back.\n")
            with apath.open("a", encoding="utf-8") as fh:
                fh.write(f"{head}\n## Retired {stamp}\n\n" + "".join(moved_lines))
            tmp = lpath.with_name(lpath.name + ".retire-tmp")
            tmp.write_text("".join(keep_lines), encoding="utf-8")
            tmp.replace(lpath)
        report[name] = {"moved": moved, "kept": kept, "archive": str(apath),
                        "ledger": str(lpath), "dry_run": dry_run}
    return report


def _print_retire(rep: dict) -> None:
    for name, r in rep.items():
        if r.get("skipped"):
            print(f"{name}: nothing retired ({r['skipped']})")
            continue
        verb = "would archive" if r["dry_run"] else "archived"
        for anchor in r["moved"]:
            print(f"{name}: {verb} -> {r['archive']}: {anchor}")
        print(f"{name}: {len(r['moved'])} row(s) {verb}, {r['kept']} kept "
              f"(line still loaded)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_dir = str(Path.home() / "obsidian" / "lloyd")
    s = sub.add_parser("status")
    s.add_argument("--memories-dir", default=default_dir)
    s.add_argument("--json", action="store_true")
    r = sub.add_parser("retire", help="archive ledger rows whose line is no longer "
                                      "loaded; never makes room for a new row")
    r.add_argument("--memories-dir", default=default_dir)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "retire":
        rep = retire(Path(args.memories_dir).expanduser(), dry_run=args.dry_run)
        if args.json:
            print(json.dumps(rep, indent=1, ensure_ascii=False))
        else:
            _print_retire(rep)
        return 0
    rep = status(Path(args.memories_dir).expanduser())
    if args.json:
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        return 0
    for name, r in rep.items():
        if r.get("missing"):
            print(f"{name}: missing")
            continue
        print(f"{name}: {r['bytes']} / {r['ceiling']} B (headroom {r['headroom']}, "
              f"tight limit {r['tight_limit']}), "
              f"{r['entries']} entries, {r['rows']} ledger rows"
              + ("  -> CURATE" if r["curate"] else ""))
        print(f"  without a row: {len(r['without_row'])}; "
              f"unrepresentable: {len(r['unrepresentable'])}; "
              f"orphan rows: {len(r['orphan_rows'])}; duplicate anchors: {len(r['duplicate_anchors'])}")
        for a in r["without_row"][:10]:
            print(f"    no row: {a}")
        for a in r["unrepresentable"][:10]:
            print(f"    unrepresentable: {a}")
        for a in r["orphan_rows"][:10]:
            print(f"    orphan: {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
