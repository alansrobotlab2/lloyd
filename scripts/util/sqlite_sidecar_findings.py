#!/usr/bin/env python3
"""Refuse a live SQLite store's companion files out of a vault COMMIT — #3236.

The vault tracks 33 store primaries under `backlog/data/`
(`git ls-files backlog/data/ | grep -cE '[.](db|sqlite|sqlite3)$'` on 2026-10-24;
`kg.sqlite` is one of the 33, not a 34th), and some of those stores are LIVE:
`ls -l backlog/data/usage.db*` shows a `-wal` and a `-shm` beside the tracked
339,968-byte primary: 0 B and 32,768 B as measured at 2026-10-24T10:55Z, while
the same store's live WAL under `~/lloyd-data` was 4,169,472 B one minute later.
Sizes are therefore no part of this rule — the file's NAME is the only stable
fact about it, and a 0-byte WAL committed beside a DB is exactly as wrong as a
4 MB one. A SQLite database committed together with a copy of its
own `-wal` and `-shm` is worse than a snapshot: restoring that pair against any
other copy of the primary is a corruption recipe, because the sidecars carry the
transaction history of the instant they were read, not of the page image next to
them.

`#3230` closed part of this with a `.gitignore` pair, and that pair is still
load-bearing — this module does NOT replace it and the vault ignore file is not
touched here. An ignore pattern is a hand-list over an open set, which is the
2026-09-20 class rule, and it fails two ways that are both measurable in
`~/obsidian` today:

  * suffix coverage. `git check-ignore -v backlog/data/usage.db-wal` exits 0
    (matched at `.gitignore:48`); `git check-ignore -v backlog/data/usage.db-journal`
    exits 1. SQLite's rollback-journal family — `*.db-journal`, the file a
    non-WAL transaction leaves behind — has no pattern, so it is an ordinary
    dirty file that `vault-commit.sh`'s own `git add -A --` would stage.
  * directory coverage. Both patterns are anchored at `backlog/data/`, while
    `git ls-files | grep -E '\\.(db|sqlite|sqlite3)$'` names store primaries
    elsewhere in the vault (`knowledge/graph.db`), whose companions are unignored.

And the ignore file has no authority over a path that is already in the index:
one `git add -f` by any writer and `*.db-wal` is committed anyway. So this
matcher reads `git diff --cached` — the index, which is what a commit is made
of — and not `.gitignore`.

Two properties the shape of the fix is boxed in by, both from the file it lands
in:

  * It keys on the COMPANION suffixes only, never on `*.db` / `*.sqlite` /
    `*.sqlite3`. A deliberate witness copy of a store's primary is wanted in the
    index — that is what #3223 asks for, and 33 witnesses already sit there — so
    a guard that keyed on the primary suffix would hide a witness from the very
    `git add` meant to archive it.
  * It unstages and reports; it never refuses the commit. `vault-commit.sh:481-488`
    (the same words sat at `:412-419` until the sidecar rung's +69 lines pushed them
    down) records why the wrapper's pre-flight rungs are report-only: the wrapper also
    commits OTHER writers' state, so one job's bad file must not strand every
    later job's snapshot — and #2245 is the scar of exactly that stranding
    (~20 minutes of task #4's commits). So a companion is dropped from the
    index, named on stderr with its primary store and the `git rm --cached`
    command that would remove it if it were tracked, and the rest of the commit
    goes through.

ONE MATCHER, TWO DOORS. The vault is committed by two routes that never call
each other: `scripts/util/vault-commit.sh` (every nightly writer) and
`scripts/automod/vault_round.py`'s `land()`, which has its own `git add` and its
own `git commit` precisely because the wrapper's pathset can never be empty.
`#2468` was paid for a guard that covered one door and called itself coverage;
both doors import these functions, and `tests/test_vault_commit_sidecar_refusal.py`
plus the one node in `tests/test_automod_vault_round.py` walk each of them.

CLI (the wrapper's call, one subprocess, which also performs the unstage):

    sqlite_sidecar_findings.py --vault DIR [--path P]... --unstage

    stdout  nothing, unless `--json`, which puts the `{"path", "suffix",
            "primary"}` records on it as one array
    stderr  ALWAYS one human refusal line per companion found — what the clause
            names: the path, its primary store, and the `git rm --cached` that
            untracks it. Both commit doors are non-interactive and their stdout is
            a transcript or a command substitution, so stdout is reserved for
            machine output and the readable refusal lives on stderr. A caller that
            wants the lines on stdout asks with `2>&1`.
    exit    0 no companion found, 1 at least one — informational, not a guard:
            the door decides, and the door commits the rest either way.

Without `--unstage` nothing is written, so the same command answers "what WOULD
this commit take?".
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# The companion files SQLite maintains beside a primary. The guard keys on these
# and on nothing else: a store primary is a witness, a companion never is.
COMPANION_SUFFIXES = ("-wal", "-shm", "-journal")

# The spellings of a store primary this repo actually archives. Used only to
# word the refusal precisely ("the WAL of `usage.db`" vs "the `-wal` companion of
# `notes.md`") — an unrecognized stem is still refused, because refusing an
# odd name costs nothing and versioning a real WAL is the expensive error.
PRIMARY_SUFFIXES = (".db", ".sqlite", ".sqlite3")

_COMPANION_KIND = {
    "-wal": "write-ahead log",
    "-shm": "shared-memory sidecar",
    "-journal": "rollback journal",
}


def companion_suffix(path: str) -> str | None:
    """The SQLite companion suffix `path` ends with, else None.

    Case-folded: a store opened from Windows tooling can come back `-WAL`, and
    the property has to hold over the names, not over one author's typing.
    """
    low = str(path).lower()
    for suffix in COMPANION_SUFFIXES:
        if low.endswith(suffix):
            return suffix
    return None


def primary_of(path: str) -> str:
    """The primary file a companion belongs to: the path minus its suffix.

    ``backlog/data/usage.db-wal`` -> ``backlog/data/usage.db``. Returns "" for a
    path that is not companion-shaped, and callers must not ask a non-companion.
    """
    suffix = companion_suffix(path)
    return "" if suffix is None else str(path)[: -len(suffix)]


def is_store_primary(path: str) -> bool:
    """True if `path` is spelled like a SQLite database file.

    Nothing in this module refuses on the strength of this — see the module
    docstring for why keying the guard on it would hide a witness. It exists for
    the wording of a refusal and for the tests that pin primaries staying put.
    """
    return str(path).lower().endswith(PRIMARY_SUFFIXES)


def companions_in(paths) -> list[dict]:
    """`{"path", "suffix", "primary"}` for every companion-shaped path in `paths`.

    Order preserved, duplicates collapsed: the same path can arrive twice when
    it is both named on the command line and sitting in the index, and one
    refusal per commit per file is the report's contract with the reader.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for raw in paths:
        path = str(raw)
        suffix = companion_suffix(path)
        if suffix is None or path in seen:
            continue
        seen.add(path)
        out.append({"path": path, "suffix": suffix, "primary": primary_of(path)})
    return out


def _git(vault: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(vault), *args],
                          capture_output=True, text=True)


def staged_companions(vault: str) -> list[dict]:
    """The companion-shaped paths git's INDEX reports as staged.

    `diff --cached --name-only`, not `status`, and never `.gitignore`: a path
    force-added past an ignore pattern is in the index, and the index is what
    `git commit` is about to turn into an object. `--no-renames` so a rename
    cannot smuggle a companion out of the list as one side of a pair.
    """
    return companions_in(_git(vault, "diff", "--cached", "--name-only",
                              "--no-renames").stdout.splitlines())


def unstage(vault: str, paths: list[str]) -> None:
    """Pull `paths` back out of the index, leaving every file on disk.

    `git reset -q -- <path>` is the right verb for both shapes of the problem:
    with no HEAD entry the path drops to untracked (its bytes untouched, which
    is what a live store's WAL needs — deleting it would be the destructive
    version of this fix), and with a HEAD entry it reverts the index to that
    entry, which is what a companion that is ALREADY tracked needs. One call per
    path: a single bad pathspec must not cancel the unstaging of the others,
    which is #2245's failure mode one layer down.
    """
    for path in paths:
        _git(vault, "reset", "-q", "--", path)


def refusal_line(path: str, vault: str) -> str:
    """The one-line refusal a reader of the job log needs, with no other context.

    Three things are mandatory and the tests say so: the offending path, the
    primary store it belongs to, and the exact `git rm --cached` that would
    remove it from tracking — the reader who finds this line is the reader who
    has to decide whether the store should be tracked at all. Written at column
    0 with a leading token: `tests/test_vault_commit_attribution.py:158` reads
    the wrapper's stderr and treats an indented line under a block header as a
    path entry, so an indented refusal would be scored as an unattributed dirty
    file. The file is left on disk, and the line says so, because "dropped from
    the commit" must never be mistaken for "deleted".
    """
    suffix = companion_suffix(path) or ""
    primary = primary_of(path)
    kind = _COMPANION_KIND.get(suffix, f"{suffix} companion")
    of = (f"the {kind} of the store primary {primary}" if is_store_primary(primary)
          else f"the {kind} companion of {primary}")
    return (f"sidecar REFUSED: {path} — {of}. Unstaged from this commit; the file "
            f"is left on disk. If it is already tracked, untrack it with: "
            f"git rm --cached {path} (run from {vault})")


MEASUREMENT_SEAM = "LLOYD_VAULT_SIDECAR_REFUSAL"


def refusal_disabled_by_environment() -> bool:
    """True only when a test harness asked for the guard to stand down.

    `#2960`'s and `#3230`'s controls have to SHOW the leak — run the wrapper with
    the ignore lines deleted and require the WAL+SHM in its commit — and a refusal
    keyed on the suffix sits in front of the very `git add` those controls measure,
    so they would refuse for the reason they exist to reproduce. The seam isolates
    the ignore layer for them. Production never sets it: no caller in `app/`,
    `agent_mcp/`, `agent-services/`, `config.yaml` or the autonomy tasks writes it,
    and `tests/test_vault_commit_sidecar_refusal.py` pins both halves — that the
    seam does what it says, and that nothing on the live path sets it.
    """
    return (os.environ.get(MEASUREMENT_SEAM) or "").strip().lower() in {
        "off", "0", "no", "false"}


def refuse_staged(vault: str, extra_paths=(), *, perform: bool = True) -> list[dict]:
    """Find and (by default) unstage every companion this commit would take.

    The union of what the index already holds and `extra_paths` — the paths the
    caller NAMED, which may not be in the index yet in a race, and which the
    caller may be about to `git add -A` a sibling of. `perform=False` is the
    dry-read a person or a test uses to ask the question without answering it.
    Returns the records in command order; the human lines are the caller's to
    print, via `refusal_line`.
    """
    # Paths in, records out — `staged_companions` already classified the index, so
    # re-feeding its RECORDS into `companions_in` would stringify each dict into a
    # path-shaped string ending in a quote, match no suffix, and the guard would
    # report a clean index while a WAL sat in it. The two sources are named in the
    # order the reader expects: what the caller asked for first, then what the index
    # holds that nobody asked for, which is the force-added case.
    if refusal_disabled_by_environment():
        # Measurement only — see the docstring. Nothing is unstaged and nothing is
        # reported, which is what lets #2960's and #3230's controls show the leak.
        return []
    found = companions_in([*extra_paths,
                           *[rec["path"] for rec in staged_companions(vault)]])
    if perform and found:
        unstage(vault, [rec["path"] for rec in found])
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vault", required=True,
                    help="vault directory whose index is about to be committed")
    ap.add_argument("--path", action="append", default=[], dest="paths",
                    help="a path the caller intends to stage (repeatable); "
                         "the index is read in addition to these, never instead")
    ap.add_argument("--json", action="store_true",
                    help="dump the records as one JSON array on stdout; the human "
                         "refusal lines go to stderr either way, so this is "
                         "additive — for a caller that wants the paths as data")
    ap.add_argument("--unstage", action="store_true",
                    help="actually pull each companion out of the index — the "
                         "doors pass this; without it the command only reports")
    args = ap.parse_args(argv)

    found = refuse_staged(args.vault, args.paths, perform=args.unstage)
    if args.json:
        print(json.dumps(found, separators=(",", ":")))
    for rec in found:
        print(refusal_line(rec["path"], args.vault), file=sys.stderr, flush=True)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
