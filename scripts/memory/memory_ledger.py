#!/usr/bin/env python3
"""Coverage of the loaded-memory rationale ledger (#1488). Read-only.

`lloyd/USER.md` and the `lloyd/MEMORY.md` index are loaded into every prompt
and sit at a fixed byte ceiling (`prompt_surface`). The proposal in
`scripts/maintenance/vault-user-md-rationale-curation.patch` gives every loaded
line a ledger row — why it is loaded, where it came from, and the checkable
condition that would make it false — in a topic file the prompt never loads
(`lloyd/memory/user-md-ledger.md`, `lloyd/memory/memory-md-ledger.md`), so the
nightly curator can make headroom by retiring the lines whose reason no longer
holds instead of refusing the next addition.

This script is the deterministic half the curator runs first. It reads, never
writes — the loaded files are written only through the memory tools and `Edit`,
which enforce the ceiling — and answers three questions:

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--memories-dir", default=str(Path.home() / "obsidian" / "lloyd"))
    s.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
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
