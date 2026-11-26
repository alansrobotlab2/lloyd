#!/usr/bin/env python3
"""Report which dated witness artifacts never say when their bytes were captured — #4262.

THE RULING this checks (#4022's owed-check, 2026-11-19): a `backlog/data` filename's
date is the READING / demand date — the day a demand named the file — and never a
capture date. The capture instant lives INSIDE the artifact (a `witness_meta` table in
a SQLite extract) or in a sibling witness note beside it. Capture-date equality is
deliberately NOT checked: `2026-11-07.3907-lloyd-data-workers.db` holds bytes captured
2026-11-10T00:05:30Z and is correctly named. The harm is the other reading — a
reading-dated name taken as a capture date: `2026-10-14.2692-lloyd-data-workers.db`
read as "the queue" is 2,609 rows over 16 kinds against 14,433 over 17 in the 11-07
extract, 5.5x under and missing `failure-investigation` entirely. So the check is
whether the self-declaration EXISTS, never whether the date in the name matches it.

What it reports, over the vault's COMMITTED `backlog/data` files (`git ls-files`):

  * stores — dated `.db` / `.sqlite` / `.sqlite3` artifacts with no `witness_meta`
    table inside and no sibling note. Detection is by table NAME only: the two
    committed stamps use different schemas (key/value in
    `2026-11-06.3850-lloyd-data-usage.db`, columnar in
    `2026-11-07.3907-lloyd-data-workers.db`), so a column or arity check would miss one.
    Each store is opened `mode=ro&immutable=1`, which creates no `-wal`/`-shm`.
  * text — every other dated non-`.md` artifact with no sibling note. A text file has
    no in-artifact stamp convention, so only the sibling arm can clear it; the two
    figures are printed apart because which of them is "the report" is owed-check's to
    rule from a first run's real output, not this module's.
  * sidecars — `-wal` / `-shm` / `-journal` files beside a tracked store, found by a
    directory scan (`os.listdir`), because the vault's `.gitignore` hides them from
    `git status` (`~/obsidian/.gitignore:48-49`, `:68-69`): a porcelain read counts 0.

A sibling note is any committed `*-witness.md` or `*.witness.md` in the same directory
that shares the artifact's `<date>.<id>` prefix, or whose own stem names the artifact
(`usage.db.witness.md` for `….usage.db`) — notes are not required to carry the prefix,
and 31 of the 85 committed on 2026-11-18 do not.

REPORT-ONLY, structurally: it renames nothing, writes nothing, runs no git command that
takes the index lock (`--no-optional-locks ls-files` only), and neither commit door —
`scripts/automod/vault_round.py` nor `scripts/util/vault-commit.sh` — imports it, so no
land can be blocked by it. The item-id-mismatch arm the item first asked for is not
here: nothing maps an artifact to the item that owes it (#4262's findings).

    python -m scripts.maintenance.witness_naming_report [--vault DIR] [--json]

Exit 0 always: the output is the report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

DATA_DIR = "backlog/data"
DATED_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.")
#: `<date>.<id>-`: the key a note and an artifact share when both carry it.
DATE_ID_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}\.\d+)-")
STORE_EXTS = (".db", ".sqlite", ".sqlite3")
NOTE_SUFFIXES = (".witness.md", "-witness.md")
SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
STAMP_TABLE = "witness_meta"


def _committed(vault: Path, data_dir: str) -> list[str]:
    out = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(vault), "ls-files", "-z", "--", data_dir],
        capture_output=True, check=True).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def has_capture_stamp(path: Path) -> bool:
    """True when the SQLite file at `path` holds a `witness_meta` table (any schema)."""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error:
        return False
    try:
        row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                          (STAMP_TABLE,)).fetchone()
        return row is not None
    except sqlite3.Error:
        return False
    finally:
        con.close()


def _note_stem(name: str) -> str | None:
    for suf in NOTE_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return None


def _names_of(artifact: str) -> set[str]:
    """Every spelling of `artifact` an unprefixed note's stem may use."""
    bare = DATE_ID_RE.sub("", artifact)
    out = {artifact, bare}
    for n in (artifact, bare):
        root, ext = os.path.splitext(n)
        if ext:
            out.add(root)
    return out


def has_sibling_note(artifact: str, notes: list[str]) -> bool:
    m = DATE_ID_RE.match(artifact)
    for note in notes:
        stem = _note_stem(note)
        if stem is None:
            continue
        if m and note.startswith(m.group(1) + "-"):
            return True
        if stem in _names_of(artifact):
            return True
    return False


def report(vault: Path, data_dir: str = DATA_DIR) -> dict:
    vault = Path(vault)
    committed = _committed(vault, data_dir)
    by_dir: dict[str, list[str]] = {}
    for rel in committed:
        d, _, base = rel.rpartition("/")
        by_dir.setdefault(d, []).append(base)
    stores, text, dated = [], [], 0
    for d, names in sorted(by_dir.items()):
        notes = [n for n in names if _note_stem(n) is not None]
        for name in sorted(names):
            if not DATED_RE.match(name) or name.endswith(".md"):
                continue
            dated += 1
            rel = f"{d}/{name}" if d else name
            if has_sibling_note(name, notes):
                continue
            if name.endswith(STORE_EXTS):
                if not has_capture_stamp(vault / rel):
                    stores.append(rel)
            else:
                text.append(rel)
    tracked = set(committed)
    sidecars = []
    for d in sorted(by_dir):
        try:
            listing = os.listdir(vault / d)
        except OSError:
            continue
        for name in sorted(listing):
            for suf in SIDECAR_SUFFIXES:
                primary = f"{d}/{name[: -len(suf)]}"
                if name.endswith(suf) and primary in tracked and primary.endswith(STORE_EXTS):
                    sidecars.append(f"{d}/{name}")
                    break
    return {"dated": dated, "unstamped_stores": stores, "unstamped_text": text,
            "sidecars": sidecars, "sidecars_indexed": sorted(set(sidecars) & tracked)}


def render(r: dict) -> str:
    lines = [f"{r['dated']} dated non-.md artifacts committed under {DATA_DIR}.",
             f"{len(r['unstamped_stores'])} dated SQLite stores declare no capture instant "
             f"(no `{STAMP_TABLE}` table, no sibling witness note):"]
    lines += [f"  {p}" for p in r["unstamped_stores"]]
    lines.append(f"{len(r['unstamped_text'])} dated text artifacts have no sibling witness "
                 f"note (text has no in-artifact stamp convention):")
    lines += [f"  {p}" for p in r["unstamped_text"]]
    lines.append(f"{len(r['sidecars'])} SQLite sidecars beside tracked stores (directory "
                 f"scan; {len(r['sidecars_indexed'])} indexed):")
    lines += [f"  {p}" for p in r["sidecars"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--vault", default=None, help="vault root (default: app.paths.VAULT_ROOT)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.vault is None:
        from app import paths
        a.vault = str(paths.VAULT_ROOT)
    r = report(Path(a.vault))
    print(json.dumps(r, indent=2) if a.json else render(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
