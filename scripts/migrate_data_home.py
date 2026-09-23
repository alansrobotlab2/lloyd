#!/usr/bin/env python3
"""Move Lloyd's runtime data out of the code tree into ~/lloyd-data.

On 2026-09-22 a pytest fixture teardown deleted ~/lloyd, and every session,
database and log inside it went with the code. `app.paths.DATA_ROOT` now names
a root outside the tree (architecture/data-home.md); this is the one-shot move
of what the tree still holds. Stdlib only, so it runs with any python and never
imports the `app.paths` whose resolution it exists to satisfy.

    migrate_data_home.py            # dry run: what would move, and how much
    migrate_data_home.py --apply    # do it (the stack must be stopped)

`--apply`, in order, stopping at the first failure with nothing half-done in
the tree:
  1. refuses while any process holds a file under a source open, or while the
     data root already carries its marker (a second run would double-move);
  2. checkpoints every SQLite WAL, so each database is one file;
  3. copies each source into the root with `cp -a --reflink=auto` (instant
     and space-free on btrfs; the copy is a real file, not a link);
  4. verifies file count, bytes, the sha256 of every database and
     `PRAGMA integrity_check` on each copy;
  5. only then moves each original to ~/lloyd-data-migration-hold/<stamp>/,
     outside the tree — deleted by nobody, ever, from this script;
  6. writes `.lloyd-data-root` last, which is what lets production start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

try:
    import pwd
    HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
except (ImportError, KeyError):  # pragma: no cover
    HOME = Path.home()

TREE = HOME / "lloyd"
DATA = Path(os.environ.get("LLOYD_DATA") or HOME / "lloyd-data")
HOLD = HOME / "lloyd-data-migration-hold"
MARKER = ".lloyd-data-root"

#: (path in the tree, path under the data root). Same names, except the
#: engines' logs, which join the rest of the logs.
MOVES: tuple[tuple[str, str], ...] = (
    ("sessions", "sessions"),
    ("event_logs", "event_logs"),
    ("_pipeline", "_pipeline"),
    ("autonomy-runs", "autonomy-runs"),
    ("logs", "logs"),
    ("agent-services/logs", "logs/services"),
    ("eval/baselines", "eval/baselines"),
    ("voice_profiles", "voice_profiles"),
    ("data", "data"),
    ("usage.db", "usage.db"),
    ("usage.db-wal", "usage.db-wal"),
    ("usage.db-shm", "usage.db-shm"),
    ("workers.db", "workers.db"),
    ("workers.db-wal", "workers.db-wal"),
    ("workers.db-shm", "workers.db-shm"),
    ("research.db", "research.db"),
    ("research.db-wal", "research.db-wal"),
    ("research.db-shm", "research.db-shm"),
    ("mc-state.json", "mc-state.json"),
)
#: In the tree, not data, and not worth a copy: moved to the hold only.
STRAYS = ("None",)
DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")


def _files(p: Path) -> list[Path]:
    if p.is_file() or p.is_symlink():
        return [p]
    out = []
    for dirpath, _dirs, names in os.walk(p):
        out.extend(Path(dirpath) / n for n in names)
    return out


def _measure(p: Path) -> tuple[int, int]:
    files = _files(p)
    return len(files), sum(f.lstat().st_size for f in files)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_holders(paths: list[Path]) -> list[str]:
    """Processes holding a file under any of `paths` open (via /proc)."""
    prefixes = tuple(str(p.resolve()) + ("/" if p.is_dir() else "") for p in paths)
    exact = {str(p.resolve()) for p in paths}
    hits = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            target = target.removesuffix(" (deleted)")
            if target in exact or target.startswith(prefixes):
                try:
                    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")[:120]
                except OSError:
                    cmd = b"?"
                hits.append(f"pid {pid} {cmd.decode(errors='replace')} -> {target}")
                break
    return hits


def _databases(root: Path) -> list[Path]:
    return [f for f in _files(root) if f.suffix in DB_SUFFIXES and f.is_file()]


def _checkpoint(db: Path) -> None:
    con = sqlite3.connect(str(db), timeout=30)
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()


def _integrity(db: Path) -> str:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        return str(con.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        con.close()


def plan(tree: Path, data: Path) -> list[tuple[Path, Path]]:
    return [(tree / s, data / d) for s, d in MOVES if (tree / s).exists() or (tree / s).is_symlink()]


def apply(tree: Path, data: Path, hold_root: Path) -> dict:
    if (data / MARKER).exists():
        raise SystemExit(f"{data} already carries {MARKER}: this migration has run. "
                         "Nothing moved.")
    if not data.is_dir():
        raise SystemExit(f"{data} does not exist. Create it first "
                         f"(`btrfs subvolume create {data}`). Nothing moved.")
    holders = _open_holders([s for s, _ in plan(tree, data)])
    if holders:
        raise SystemExit("refusing: these processes hold files under the sources open "
                         "(stop the stack first):\n  " + "\n  ".join(holders[:20]))
    for dst in (d for _, d in plan(tree, data)):
        if dst.exists() and (dst.is_file() or any(dst.iterdir())):
            raise SystemExit(f"refusing: {dst} already exists and is not empty. Nothing moved.")

    # 2. one file per database. Planned AFTER: closing the last connection to a
    # WAL database removes its -wal and -shm files.
    for src, _ in plan(tree, data):
        for db in _databases(src):
            _checkpoint(db)
    moves = plan(tree, data)

    # 3 + 4. copy, then verify everything before touching an original
    report: dict = {"tree": str(tree), "data": str(data), "moves": []}
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_dir() and not any(dst.iterdir()):
            dst.rmdir()
        subprocess.run(["cp", "-a", "--reflink=auto", str(src), str(dst)], check=True)
        before, after = _measure(src), _measure(dst)
        if before != after:
            raise SystemExit(f"verify failed for {src} -> {dst}: {before} != {after}. "
                             "Originals untouched; remove the copies and retry.")
        dbs = []
        for db in _databases(src):
            copy = dst / db.relative_to(src) if src.is_dir() else dst
            if _sha(db) != _sha(copy):
                raise SystemExit(f"verify failed: {copy} differs from {db}. Originals untouched.")
            check = _integrity(copy)
            if check != "ok":
                raise SystemExit(f"integrity_check on {copy}: {check}. Originals untouched.")
            dbs.append(str(copy))
        report["moves"].append({"from": str(src), "to": str(dst), "files": after[0],
                                "bytes": after[1], "databases_checked": dbs})

    # 5. originals aside, outside the tree
    hold = hold_root / time.strftime("%Y%m%d-%H%M%S")
    for src, _ in moves:
        target = hold / src.relative_to(tree)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, target)
    for name in STRAYS:
        p = tree / name
        if p.is_file() and p.stat().st_size == 0:
            target = hold / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(p, target)
    report["hold"] = str(hold)

    # 6. the marker, last
    (data / MARKER).write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "by": "scripts/migrate_data_home.py",
        "migrated_from": str(tree),
    }) + "\n")
    (hold / "migration-report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="move the data (default: dry run)")
    ap.add_argument("--tree", type=Path, default=TREE)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--hold", type=Path, default=HOLD)
    args = ap.parse_args(argv)
    if not args.apply:
        for src, dst in plan(args.tree, args.data):
            n, b = _measure(src)
            print(f"{src} -> {dst}  ({n} files, {b / 1e6:.1f} MB)")
        strays = [args.tree / s for s in STRAYS if (args.tree / s).exists()]
        for s in strays:
            print(f"{s} -> hold (stray)")
        print(f"marker: {args.data / MARKER} "
              f"({'present' if (args.data / MARKER).exists() else 'absent'})")
        return 0
    report = apply(args.tree, args.data, args.hold)
    for m in report["moves"]:
        print(f"moved {m['from']} -> {m['to']} ({m['files']} files, {m['bytes'] / 1e6:.1f} MB, "
              f"{len(m['databases_checked'])} databases ok)")
    print(f"originals held at {report['hold']}; marker written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
