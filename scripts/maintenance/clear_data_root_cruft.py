#!/usr/bin/env python3
"""Delete ONE explicitly-named, provably-inert cruft stub from the data root's top level.

#3210 gave the guardian eyes for a 0-byte stub standing at the data root — a file named
`false` or `sqlite3`, the residue of a shell redirection or a `sqlite3 <path>` typed a
half-second too early — and, held to #3109's report-only contract for anything under this
root, it only reports. Its own docstring says the two stubs it named on 2026-10-24 are
"owed to whoever deletes them", and on 2026-10-27 both were still here: 0 bytes, `file`
says empty, `lsof` shows no handle, and the only repo reference to either name is a comment.
This script is the other half of that pair: the deletion, held to a standard a report never
had to meet, run by a person who is looking at the root.

Why a person runs it and the watchdog does not: deleting a file is not a measurement.
#3109 settled that the leg over this root reports and never touches, and the one write the
guardian does have (`datawatch.quarantine_inert`) moves code-tree residue INTO the
root, never the reverse. This program is therefore the only thing in the repo that unlinks a
data-root name, and it is reachable only from a terminal: `datawatch.py` and `guardian.py`
do not import it, do not shell out to it, and gain no deletion of their own. The
`test_no_guardian_route_reaches_the_remover_or_deletes_a_data_root_name` node pins that.

Its entire value is that it refuses far more readily than it deletes. The refusals, in the
order they are asked — every one of them a class the live root held a member of on
2026-10-27, so none is hypothetical:

1. **The name is not one bare, non-dot basename.** No path, no `..`, no glob character. The
   item's rule "never a glob, never recursive" is a statement about this program's own
   reach: it unlinks the names it was handed, one `os.unlink` per name it accepted, and a
   `*` in an argument means somebody meant a sweep, which is a different and louder act.
2. **It is not a 0-byte regular file.** A directory, a symlink, a socket, a fifo or anything
   with bytes in it is not this class. `os.lstat` answers size and type in one call, and it is
   `lstat` and not `stat` so a symlink is refused as what it is rather than resolved into the
   file it points at.
3. **A declared store's basename.** Drawn from the modules that declare the stores —
   `app.data_root.KG_DB_RELATIVE` plus `USAGE_DB`/`WORKERS_DB`/`FAILURE_LEDGER_DB`/
   `RESEARCH_DB` in `app/paths.py` — never from a list typed out here, because a list here
   rots the first time a store is added and the whole point is that this program cannot know
   better than the code that opens these files. `usage.db` sits at the live root right now
   with a live `-wal` and `-shm` beside it.
4. **Any name ending in a database suffix** — `datawatch._DB_SUFFIXES`. That class is
   `data_root_shadows`' to name and it has a different remedy: a 0-byte `kg.sqlite` at the
   root may be shadowing a populated nested store, and deleting it is a decision about which
   store wins, not a cleaning.
5. **A SQLite sidecar of a store standing in the same scan**, by the pairing predicate the
   hourly leg already uses (`datawatch._sidecar_of_store_in_this_scan`, which reuses
   `_SQLITE_SIDECARS`). This is the one that makes the class non-trivial: the live
   `workers.db-wal` was 0 bytes at the 2026-10-26 17:59 checkpoint, and it grows and empties
   on its own. Its own name has no `-wal` sibling (`workers.db-wal-wal` never exists), so a
   checker that only looks for the candidate's own siblings waves it through, and
   `'workers.db-wal'.endswith('.db')` is False, so the suffix test waves it through too. It
   is a live database file whose bytes happen to be zero right now.
6. **A `*.lock` name, or any name whose own base file stands at the root.** The live
   `next-session-notes.json.lock` is 0 bytes, is 31 days old, and is a lock: some code owns
   it or will. `.lock` appears nowhere in `datawatch.py`, so no existing constant carries it
   and this file is the only place that spells it — with the base-file test beside it, which
   catches the other lock spellings (`<name>.lock` is refused twice over, once per rule).
7. **Anything with an open handle.** `lsof -t` on the exact path. The size and sibling tests
   are properties of the name; this one is the question "is anybody using it right now", and
   it is asked last because it is the only one that costs a subprocess. An `lsof` that
   errors is a refusal, not a pass: no answer is not "no handles".

Exit codes: 0 every name was removed or was not there to begin with; 1 a name was refused
(and its reason printed) or the baseline prune had to refuse something; 2 the run could not
establish which root it was standing in. A dry run refuses with the same exit 1 — the point
of the dry run is to be the pre-flight that disagrees with you in the same words the real run
would use.

Stdlib-only, and it reaches the two modules it reuses the same way
`scripts/groundskeeper/retention-sweep.py`'s resolver does, for the reason stated there: a
   cron or
`sh -c` child has no project venv, and a resolver that cannot be imported is a resolver that
gets hand-rolled a second time (#1525's shape).
"""
from __future__ import annotations

import argparse
import os
import stat as _stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "agent-services" / "guardian")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from app.data_root import (DATA_ROOT_MARKER, KG_DB_RELATIVE,
                               resolve_data_root_for_tree)
except ImportError:  # pragma: no cover - `sys.path` holding app/ itself
    from data_root import (DATA_ROOT_MARKER, KG_DB_RELATIVE,
                           resolve_data_root_for_tree)

import datawatch as DW                                          # noqa: E402

#: A lock's own suffix, spelled nowhere else in the repo: `.lock` appears in no constant in
#: `datawatch.py`, so a checker that wants to refuse one has to carry the word itself.
LOCK_SUFFIX = ".lock"

#: Glob and path characters an argument may not contain. `?` and `*` and `[` are the intent
#: of a sweep; `/` is a path, and the root this program acts under is decided once.
_FORBIDDEN_CHARS = ("*", "?", "[", "]", "/", "\\")

EXIT_OK = 0
#: A name was looked at and declined. Distinct from 2 because the program DID start and DID
#: judge: an operator triaging a red exit needs to know whether the run happened.
EXIT_REFUSED = 1
#: 2 is what the shell convention gives a usage error, and both routes below are that: the
#: root named is not a data root, or the arguments are not names. Both mean the same thing to
#: the reader — NO name was considered, so nothing was declined on its merits.
EXIT_NO_ROOT = 2
EXIT_USAGE = 2


def data_root(tree: "str | None" = None, override: "str | None" = None) -> Path:
    """The one root this program may act under, resolved through `app.data_root`.

    Not `app.paths.DATA_ROOT`: that constant is a reader's root, which answers even in the
    state the watchdog exists to alert on (a production root that lost its marker), and a
    write route wants the opposite property — `resolve_data_root_for_tree` RAISES rather than
    guess, which is why the stray-move route resolves through it too. With no `--data-root`
    it resolves for the checkout this script lives in, so a worktree copy of this file acts
    under its own root and can never reach another tree's. An explicit `--data-root` is
    honoured as given, which is the seam a test uses; `DataRootUnresolvedError` from the
    resolver becomes exit 2 rather than a guess.
    """
    if override:
        return Path(override).expanduser()
    return Path(resolve_data_root_for_tree(Path(tree) if tree else None))


def _kind_of(st) -> str:
    """What a name actually IS, for the refusal that says so.

    `S_ISREG` alone answers "not a regular file", which is the one message that tells an
    operator nothing: a symlink, a directory and a socket each need a different next action.
    The symlink is the hazard worth naming — resolving one is how a root-local unlink turns
    into a deletion somewhere else — which is why the accept-set calls `lstat` and not `stat`.
    """
    for test, name in ((_stat.S_ISLNK, "a symlink, which would be followed out of the root"),
                       (_stat.S_ISDIR, "a directory: never recursive"),
                       (_stat.S_ISFIFO, "a fifo"),
                       (_stat.S_ISSOCK, "a socket"),
                       (_stat.S_ISBLK, "a block device"),
                       (_stat.S_ISCHR, "a character device")):
        if test(st.st_mode):
            return name
    return "not a regular file"


def declared_store_names() -> "frozenset[str]":
    """Basenames a declared store claims at the data root, read from the declaring modules.

    `app.data_root` declares one (`KG_DB_RELATIVE`; `QUARANTINE_DIR_RELATIVE` is a directory,
    not a file name) and `app/paths.py` declares the rest of the ones actually standing at
    the root. Read by attribute so adding a store extends this list by itself — a name list
    maintained here would be a second opinion about which files are somebody's database.
    """
    import app.paths as P
    names = {Path(str(KG_DB_RELATIVE)).name}
    for attr in ("USAGE_DB", "WORKERS_DB", "FAILURE_LEDGER_DB", "RESEARCH_DB"):
        val = getattr(P, attr, None)
        if val is not None:
            names.add(Path(str(val)).name)
    return frozenset(n for n in names if n)


def has_open_handle(path: Path) -> "tuple[bool, str]":
    """(held, reason). `lsof -t` on the exact path: its exit 1 is the no-handle answer.

    An unusable `lsof` is refused rather than skipped. "Could not ask" is not "nobody is
    using it", and the whole claim of this program is that every name it deletes was proven
    inert — an unanswered question keeps the file where it is.
    """
    try:
        proc = subprocess.run(["lsof", "-t", "--", str(path)], capture_output=True,
                              text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return True, f"could not ask whether a handle is open ({type(exc).__name__}: {exc})"
    if proc.returncode > 1:
        return True, (f"the handle probe said nothing useful (lsof exit {proc.returncode}: "
                      f"{(proc.stderr or '').strip()[:120]})")
    pids = [p for p in (proc.stdout or "").split() if p.strip()]
    if pids:
        return True, f"open handle(s) held by pid(s) {', '.join(pids[:8])}"
    return False, ""


def _bad_name(raw: str) -> str:
    """Why this argument is not a bare name, or "" when it is.

    Checked on the raw argument, before it is joined to anything: `/` and `*` are the two
    characters that turn a name list into a path or a sweep, and an argument carrying either is
    evidence about the operator's intent that the accept-set must not quietly overrule by
    answering "no such file".
    """
    if not raw:
        return "is empty"
    if raw.startswith("-"):
        return "reads as a flag, not a name"
    if "/" in raw or "\\" in raw:
        return ("is a path: this program is handed the names it deletes, one at a time, and "
                "resolves nothing")
    if any(c in raw for c in "*?["):
        return ("carries a glob character, and a shell that had expanded it would have left "
                "no trace of the sweep you meant")
    if any(c.isspace() for c in raw):
        # Quoting gone wrong (`--apply "false sqlite3"`) reads as one name to the shell and as
        # a name that does not exist to this program; answering "not there" would tell the
        # operator the root is clean when nothing was ever looked for.
        return "contains whitespace, so it is not one bare name"

    # A dot-name is NOT handled here: `./.lloyd-data-root` typed as a path is, but the bare
    # marker name is a legitimate thing to hand this program, and classify refuses it on its
    # merits (exit 1, "a name was looked at and declined"). Reporting it here would be a usage
    # error about a correct invocation.
    return ""


def classify(name: str, root: Path, scan: "dict[str, float]",
             stores: "frozenset[str]") -> "tuple[str, str]":
    """One of ("remove", "noop", "refuse") and the reason an operator reads.

    `scan` is the root's own name→mtime map from `datawatch.top_level_regular_files`, which
    is what "standing in the same scan" means in the sidecar predicate's own words — the
    hourly leg's snapshot, not a re-read, so this program and the leg cannot disagree about
    which stores are present.
    """
    if not name or name in (".", "..") or name.startswith(".") \
            or any(c in name for c in _FORBIDDEN_CHARS):
        return "refuse", (f"{name!r} is not one bare, non-dot basename; this program takes "
                          f"the names it deletes as arguments and never sweeps")

    st = None
    try:
        st = (root / name).lstat()
    except FileNotFoundError:
        return "noop", "not there"
    except OSError as exc:
        return "refuse", f"could not even stat it ({exc})"

    if not _stat.S_ISREG(st.st_mode):
        return "refuse", f"{_kind_of(st)}: this program unlinks a 0-byte regular file, " \
                         "nothing else"
    if st.st_size != 0:
        return "refuse", f"is {st.st_size} bytes, not 0 — the class is an empty stub"
    if _stat.S_IMODE(st.st_mode) & 0o002:
        return "refuse", "is world-writable, which a stub a program created is not"

    if name in stores:
        return "refuse", ("is a declared store's basename: some code opens it, and a 0-byte "
                          "answer from a database is a state to read, not cruft to clear")
    if name.endswith(DW._DB_SUFFIXES):
        return "refuse", ("ends in a database suffix, which is data_root_shadows' class to "
                          "name: deleting it can decide which of two stores wins")
    if name.endswith(LOCK_SUFFIX):
        return "refuse", (f"is a {LOCK_SUFFIX} name: a lock is owned by whatever takes it, "
                          "empty or not")
    # `splitext` answers `("false", "")` for a name with no extension, and a name with no
    # extension has no base to be a sibling OF — comparing the candidate's own stem against the
    # root compares it with itself and refuses it. That is not a corner case: `false` and
    # `sqlite3`, the two names this entire class is made of, are extensionless, so a rule that
    # fires on an empty extension refuses exactly the names the remover exists for and passes
    # every unit node that only ever hands it `usage.db-wal`.
    base, ext = os.path.splitext(name)
    if ext and base and (root / base).is_file():
        return "refuse", (f"is derived from {base!r}, which stands at the root as a real "
                          "file; a name beside a file belongs to it")
    own = [name + sfx for sfx in DW._SQLITE_SIDECARS if (root / (name + sfx)).exists()]
    if own:
        return "refuse", (f"has live database sidecar(s) beside it: {', '.join(own)}. A file "
                          "with a -wal/-shm/-journal sibling is a store in use, not a stub")
    if DW._sidecar_of_store_in_this_scan(name, scan):
        return "refuse", ("is a SQLite sidecar of a store standing in the same scan, which is "
                          "a database file whose bytes happen to be 0 right now")
    if name not in scan:
        return "refuse", ("is not in the scan the sidecar test was answered from, so no "
                          "inertness claim about it is measurable")

    held, reason = has_open_handle(root / name)
    if held:
        return "refuse", reason or "an open handle was reported without a reason"
    return "remove", "0-byte regular file, no declared owner, no sidecar, no handle"


def run(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        prog="clear_data_root_cruft",
        description="Remove explicitly-named, provably-inert 0-byte cruft from the data "
                    "root's top level. Dry run unless --apply.")
    ap.add_argument("names", nargs="+", help="the bare basename(s) to remove, spelled out")
    ap.add_argument("--apply", action="store_true",
                    help="actually unlink; without this nothing is deleted")
    ap.add_argument("--data-root", default="",
                    help="act under this root instead of resolving it (a test's fixture)")
    ap.add_argument("--tree", default="",
                    help="resolve the data root of this checkout instead of production's")
    ap.add_argument("--state-dir", default="",
                    help="the guardian state dir holding the arrivals baseline "
                         "(default: LLOYD_GUARDIAN_STATE)")
    args = ap.parse_args(argv)

    try:
        root = data_root(args.tree, args.data_root)
    except Exception as exc:                                   # noqa: BLE001
        print(f"refused to run: the data root is not established ({exc})")
        return EXIT_NO_ROOT
    print(f"data root: {root}")
    # The declined flag is named in the rehearsal's own line: an operator reading this back has
    # to be able to tell a run that found nothing from one that found something and was not
    # asked to delete it.
    print(f"mode: {'APPLY' if args.apply else 'dry run — pass --apply to delete anything'}")

    # A malformed argument is a bad INVOCATION, not a name that failed the accept-set, and it
    # is answered before any name is looked at. Mixing the two would be the misleading option:
    # `sub/false` alongside a name that really would be removed exits red either way, so an
    # operator who then read the output has to be told "you did not type what you meant", not
    # "a name was refused" — and a refusal reported per-name would imply the others proceeded.
    bad = [(raw, _bad_name(raw)) for raw in args.names]
    bad = [(raw, why) for raw, why in bad if why]
    if bad:
        # Every malformed argument at once: an operator who mistyped two of them should not
        # have to run the command twice to learn that.
        for raw, why in bad:
            print(f"refused before any name was read: {raw!r} {why}")
        print(f"result: {len(bad)} of {len(args.names)} arguments were not bare names, so NO "
              "name was looked at; this program takes names, never a path or a glob")
        return EXIT_USAGE

    # The marker is the root's own declaration that it is a data root. A mistyped
    # `--data-root` (a home directory, a repo) is refused here, before any name is looked at,
    # because the one property every deletion below shares is that it happens inside a root
    # that says what it is.
    if not (root / DATA_ROOT_MARKER).is_file():
        print(f"refused to run: {root} carries no {DATA_ROOT_MARKER} marker, so this program "
              "cannot tell it is a data root rather than a directory that happens to be here")
        return EXIT_NO_ROOT

    try:
        scan = DW.top_level_regular_files(str(root))
    except OSError as exc:
        print(f"refused to run: the root could not be scanned ({exc})")
        return EXIT_NO_ROOT
    stores = declared_store_names()

    removed: "list[str]" = []
    refused = False
    for name in args.names:
        verdict, reason = classify(name, root, scan, stores)
        if verdict == "noop":
            print(f"nothing to remove: {name} ({reason})")
            continue
        if verdict == "refuse":
            print(f"refused: {name}: {reason}")
            refused = True
            continue
        if not args.apply:
            print(f"would remove: {name}: {reason}")
            continue
        try:
            (root / name).unlink()
        except FileNotFoundError:
            print(f"nothing to remove: {name} (it went away between the test and the unlink)")
            continue
        except OSError as exc:
            print(f"refused: {name}: the unlink itself failed ({exc})")
            refused = True
            continue
        removed.append(name)
        print(f"removed: {name}: {reason}")

    if removed:
        # `write_arrivals_baseline` merges monotonically (`seen = {**previous, **files}`) and
        # `new_top_level_names` reports only names NOT already in it, so a name that was
        # `seen` before it was deleted can never be reported again: delete `false` today and
        # re-create it tomorrow and the hourly leg that exists to name a recurrence is silent
        # about it forever. Pruning exactly these names is what keeps the leg able to do its
        # job; `root=` is passed so a name still standing at the root is refused by the prune
        # rather than re-labelled as an arrival.
        state_dir = Path(args.state_dir or DW.V.GUARDIAN_STATE)
        pruned, pr_refused = DW.prune_arrivals_baseline(state_dir, removed, now=None,
                                                        root=str(root))
        for n in pruned:
            print(f"arrivals baseline pruned: {n} (a recurrence of this name is nameable "
                  "again)")
        for n, why in pr_refused:
            print(f"arrivals baseline NOT pruned: {n}: {why}")
            refused = True

    if refused:
        print("result: at least one name was refused; nothing about it was changed")
        return EXIT_REFUSED
    print(f"result: {len(removed)} removed"
          + ("" if args.apply else " (dry run: 0 deleted)"))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(run())
