"""Data-root tripwire: `~/lloyd-data` is watched the way the vault is.

Lloyd's runtime data — transcripts, `workers.db`, `usage.db`, the knowledge
graph, the logs — lived inside the code tree until 2026-09-22, when a pytest
fixture teardown deleted `~/lloyd` and all of it with the code. It moved to
`~/lloyd-data` (`architecture/data-home.md`), out of reach of anything aimed at
the tree. This is the tick-by-tick half of guarding it; the hourly read-only
btrfs snapshots are the other.

Same thresholds and latch as `vaultwatch` (it reuses `measure`/`evaluate`),
plus the two things specific to the data root:

* the root must carry `.lloyd-data-root`. `app.paths` refuses to start
  production without it, so a missing marker is either a deletion or a swap;
* a watch that has never seen the root is not armed. A machine that has not
  been cut over yet has no data root to lose, and a tripwire that fires on day
  one teaches everyone to clear it without reading it.

A trip pauses the worker pool and halts promotions (there is no sync to stop),
writes `data-tripped.json`, and alerts critical. `snapshot-data.sh` refuses
while the marker exists, so a gutted root never becomes the newest snapshot.

`stray_in_tree` is the drift check: a runtime name reappearing inside
`~/lloyd` means some writer still resolves its path off the code tree.

`snapshot_report` is the other half of guarding the root: the hourly snapshot
is the layer that catches what this tripwire cannot, and it fails silently —
both of `snapshot-data.sh`'s refusals `exit 0` by design, and the unit is
`Type=oneshot`, so it still reports `Result=success` after refusing. How old the
newest snapshot in that directory is is the only check that separates alive from
dead, because pruning never deletes the newest one whatever its age, so the
entry *count* looks healthy forever (#1416).

Stdlib only. CLI:  datawatch.py status | clear | snapshot-gate | snapshot-age | strays | inert
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import stat
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import policy as P
import vaultwatch as V

#: Where this process's READ-ONLY data lives: `DataWatch`'s tripwire root (`:643`),
#: `snapshot_gate` (`:755`) and the shadow check's scan root (`:490`). One resolution with
#: `policy.DATA_ROOT` (#3208) — this line used to hand-roll the same answer a second time,
#: reading the environment over an account-named absolute default, and two independent
#: copies of one resolution are how a single root move ends up with two answers (#1525's
#: shape, one constant later). It is a reader's root and it always answers, including in
#: the state the tripwire below exists to alert on: a production root that lost its marker,
#: where the resolver refuses by design and a raise at import would take the watchdog with
#: it. The spell-and-scan node in `tests/test_guardian_inert_stray.py` is what keeps a
#: hand-rolled spelling from coming back here, prose included.
#:
#: The stray MOVE does not use it. `resolve_move_root` is the write route, and it answers
#: for the tree the residue was found in — refusing rather than falling back, because a
#: move creates a file under the root and unlinks one from the tree.
DATA_ROOT = P.DATA_ROOT
TREE = os.environ.get("LLOYD_TREE", "/home/alansrobotlab/lloyd")
ROOT_MARKER = ".lloyd-data-root"

#: What used to live in the tree and must not come back there. Anything the
#: gitignore also hides: a writer that still resolves off `__file__` recreates
#: one of these silently, because git never shows it.
#:
#: Since #1541 this is no longer the candidate list — `stray_in_tree` asks the
#: tree — and the list survives for the one thing no enumeration can do: name
#: the runtime stores that sit *below* the top level that scan reads
#: (`data/tool_overrides.yaml`, `agent-services/logs`), which `git status`
#: cannot show either. Everything else in it is history worth keeping rather
#: than a prediction of what a writer will create next. `"None"` included: it is
#: the scar of a writer that once resolved a path to `None` and created
#: `~/lloyd/None` (added by `6426668b`); no such file exists today, the writer
#: was never identified, and deleting the name deletes the only breadcrumb to it.
RUNTIME_NAMES = ("sessions", "event_logs", "logs", "autonomy-runs", "_pipeline",
                 "usage.db", "workers.db", "research.db", "mc-state.json",
                 "data/tool_overrides.yaml", "voice_profiles", "eval/baselines",
                 "agent-services/logs", "None")

#: How long a top-level folder must have been under watch before losing it is an
#: event (#3153), and the other half of the pair `RUNTIME_NAMES` is: that list names
#: the folders age can never excuse, this bounds the folders it can. A folder this
#: watcher first sighted more recently than this is a folder it watched ARRIVE between
#: two of its own measurements, and no store this root has ever declared was created
#: between two ticks — so its disappearance is a job cleaning up, not a wipe, and
#: `DataWatch._excuse_vanished` says so instead of halting promotions.
#:
#: WHY A LIFETIME AND NOT A NAME. The 2026-10-20 trips were a scratch worktree and
#: #3095 excused that by the `.git` inside it; the same shape without a `.git` (a job
#: that runs `mkdir ~/lloyd-data/<job>-<id>`, works, and deletes on success) has no
#: byte that says scratch, and `VaultWatch.tick` measures every tick, so any job that
#: outlives one gap is in the baseline before it is deleted. Names are free text, which
#: is why #2160 rules out a pattern; "the watcher saw this folder come into being" is a
#: reading over the open set instead. See `vaultwatch.Snapshot.first_seen`.
#:
#: THE NUMBER IS A PERSON'S, NOT THIS CONSTANT'S. It has to be at least the longest
#: legitimate scratch job, and only post-landing traffic shows that — it is recorded on
#: backlog #3153 as an owed ruling. Two things it is NOT free to become: below
#: `vaultwatch.WINDOW_SECONDS` (900.0), because then the stamp it is compared against
#: would be capped by the history ring it is read out of, which the horizon node in
#: `tests/test_guardian_datawatch.py` pins; and above a span in which a real store could
#: plausibly be created and then vanish unnoticed, which is the reason it is a day and
#: not a month.
SCRATCH_HORIZON_SECONDS = 24 * 3600.0

#: Top-level entries a real `~/lloyd` checkout carries that are not runtime
#: data. `architecture/data-home.md` names the rebuildable-cache half of this
#: set ("code, build output or a rebuildable cache, not data": `.venvs/`,
#: `qmd/`, `node_modules`, `graphify-out/`, `__pycache__`); the rest is editor
#: and CI tooling, the gitignored local config, and `.claude/worktrees`, which
#: holds the trees an open automod round is running.
#:
#: An EXCLUSION list, deliberately the mirror image of `RUNTIME_NAMES`: a
#: top-level entry is a candidate stray unless git tracks it or it is named
#: here, so a writer's new directory is caught without anyone having predicted
#: its name — the class rule that a hand-maintained allowlist cannot close a
#: property over an open set. What cannot be closed is the other direction, and
#: it is a decision rather than an oversight: a new *tooling* directory alarms
#: until a person decides which side of this constant it belongs on. For a
#: tripwire that is the right way round, and the alert names this set so the
#: decision has somewhere to land.
KNOWN_GOOD_TOPLEVEL = frozenset({
    ".git",            # git never lists its own repository directory
    ".claude",         # agent scratch; `.claude/worktrees` is an open round
    ".env",            # local config, gitignored, never committed
    ".pytest_cache",   # ignored only by its own nested .gitignore, not ours
    ".venvs",
    ".vscode",
    "__pycache__",
    "graphify-out",    # graphify's default output, only from a hand-run build now
                       # (`code_graph.refresh` writes to `CODE_GRAPH_DIR`)
    "qmd",             # the qmd fork: its own clone, ignored by `.gitignore`
    "node_modules",
    "llama.cpp",       # vendored trees; neither one is gitignored
})


def _ls_files(tree: str, names: tuple[str, ...] = (), *flags: str) -> set[str] | None:
    """The paths `git ls-files` reports under `names` — every path in the index
    when `names` is empty — or None when git cannot answer (not a checkout, no
    git, a hung index lock)."""
    try:
        out = subprocess.run(["git", "-C", tree, "ls-files", "-z", *flags,
                              *(("--", *names) if names else ())],
                             capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return {p for p in out.stdout.decode("utf-8", "replace").split("\0") if p}


def _under(paths, name: str) -> bool:
    """Whether any path in `paths` is `name` or sits below it.

    Hoisted out of `stray_in_tree`, where it was a closure, because the reachability
    predicate below needs the same comparison and a second spelling of "is under" is
    one place for the two readers of this rule to disagree."""
    return any(p == name or p.startswith(name + "/") for p in paths)


def _top_level(tree: str) -> list[str]:
    """The tree's own top-level entries, minus `KNOWN_GOOD_TOPLEVEL`.

    This is where the candidates come from now, and the point of it is that
    nobody listed them: `os.scandir`, not a tuple of names someone predicted,
    and not `git ls-files --others --exclude-standard`, which the item proposed
    and which answers 0 paths on this tree — `--exclude-standard` excludes
    exactly the gitignored writers (`usage.db`, `workers.db`, `research.db`,
    `mc-state.json`) the check exists to catch. The scan is the top level only:
    one level deeper is the 367,353 ignored paths the full ignored set holds,
    and nested runtime names are what `RUNTIME_NAMES` is still for."""
    try:
        return sorted(e.name for e in os.scandir(tree)
                      if e.name not in KNOWN_GOOD_TOPLEVEL)
    except OSError:
        return []


def _runtime_name(rel: str) -> "str | None":
    """The retained `RUNTIME_NAMES` entry `rel` is or sits below, or None.

    This is what makes a nested runtime name reachable at all: `data/tool_overrides.yaml`
    and `agent-services/logs` are two-segment names the top-level scan cannot see, so
    the list is the only thing that reaches them (#1541), and "is this path under one of
    them" is the one question that decides which half of the rule judges it."""
    for name in RUNTIME_NAMES:
        if rel == name or rel.startswith(name + "/"):
            return name
    return None


def reachable_by_stray_check(path: str, tree: str = TREE,
                             tracked: "set[str] | None" = None) -> bool:
    """Could `stray_in_tree` report this path? One path in, one answer out.

    The whole per-path decision the alert makes, exported for the one other reader
    that has to say the same thing about a path: `agent_mcp._bash_tree_strays` labels
    every row it appends to its own journal (`app.paths.TREE_STRAY_JOURNAL_PATH`) with
    the instrument that could act on it (#2172), and it labels through HERE rather than
    from a copy of the two exclusion sets. Two hand-maintained lists drift, and a drift
    on that journal is an accusation — nightly reflection read four rows, one of them
    `web/tsconfig.node.tsbuildinfo`, sitting under an `ALERT.md` `cleared:` stamp and
    filed #2169 at priority high as a silent-clear alerting gap, for a path the alert
    could never have reached at all. One-constant discipline, as #2110 established for
    `QUARANTINE_DIR_RELATIVE`.

    The decision is `stray_in_tree`'s, unchanged and in both its halves:

    * a path is judged by the NAME the alert would report: its retained
      `RUNTIME_NAMES` entry if it is or sits below one (`logs/x.log` is reported as
      `logs`, and `data/tool_overrides.yaml` is a name in its own right), otherwise
      its top-level entry — the scan is one `os.scandir` level deep, so a nested path
      under a tracked directory can only ever be named through that directory;
    * a name `KNOWN_GOOD_TOPLEVEL` explains is never reported: the tree's own
      tooling and rebuildable caches;
    * a retained name is reported unless git tracks everything under it, and a
      tracked retained name is reported again once something untracked lands beside
      the committed records (9e98d0df's hourly false alarm);
    * any other name is reported only when git tracks nothing at all under it, which
      is what keeps `app/__pycache__`, `scripts/selfmod` and `web/.vite` off the alert.

    So `web/x.tsbuildinfo` is NOT reachable — `web` is tracked, so the alert names
    neither the file nor its directory — while `workers.db` and
    `sessions/20261004_0232.json` both are, and `cache/x` (an untracked directory
    nobody listed) is, because the alert would alarm on `cache`.

    EXISTENCE IS DELIBERATELY NOT PART OF THE QUESTION, and that is what makes the
    answer usable twice. `stray_in_tree` can only ever report something its scan found,
    so `_candidates` filters on `lexists` and this predicate answers the rule alone —
    which is exactly what a reader holding a HISTORICAL path needs: a `removed` journal
    row names a file the session already deleted, and "would the alert have been able to
    name this" must not turn into "it cannot, because it is gone", or the one row that
    proves the stray was actioned reads as unreachable.

    `path` is accepted in the two shapes its two callers hold it in: tree-relative
    (`web/x.tsbuildinfo`, which is what the bracket journals, because git reports it that
    way) or absolute (what a person pastes out of an alert). A path outside `tree` — an
    absolute one elsewhere, or a relative one that walks out with `..` — is not something
    this check can ever name, and the tree itself answers True, trivially. `tracked` is the index read the caller
    already holds — `stray_in_tree` passes its own so a whole scan costs one
    `git ls-files`, and a caller with none reads it here. When git cannot answer at all
    an open-set name is called unreachable and a retained name reachable: unknown may
    never become an alert that did not have to fire, the same direction of failure
    #1541 chose. A retained name whose `--others` read fails is reachable on the index
    alone — the judgement degrades per name, which is what one predicate per decision
    buys, and it degrades toward the alert firing, which is the side that gets read."""
    p = os.path.normpath(str(path))
    t = os.path.normpath(str(tree))
    if os.path.isabs(p):
        if not (p == t or p.startswith(t + os.sep)):
            return False
        rel = "" if p == t else p[len(t) + 1:]
    elif p == "." or p == "":
        rel = ""
    elif p.startswith(".." + os.sep) or p == "..":
        return False                       # a relative path that leaves the tree
    else:
        rel = p                            # tree-relative, the shape the bracket journals
    if not rel:
        return True
    name = _runtime_name(rel) or rel.split("/", 1)[0]
    if name in KNOWN_GOOD_TOPLEVEL:
        return False
    if tracked is None:
        tracked = _ls_files(t)
    if name not in RUNTIME_NAMES:
        return tracked is not None and not _under(tracked, name)
    if tracked is None or not _under(tracked, name):
        return True
    untracked = _ls_files(t, (name,), "--others")
    return True if untracked is None else _under(untracked, name)


def _candidates(tree: str) -> list[str]:
    """The names worth asking `reachable_by_stray_check` about.

    The retained runtime names that exist, plus the tree's own top-level entries from
    `_top_level` — scanned, not predicted, which is the whole point of #1541. Whether
    a candidate is REPORTED is the predicate's decision and nobody else's, including
    the subtraction of `KNOWN_GOOD_TOPLEVEL`, which `reachable_by_stray_check` applies
    itself and `_top_level` pre-applies as a cheap filter on the scan."""
    retained = [n for n in RUNTIME_NAMES if os.path.lexists(os.path.join(tree, n))]
    return retained + [n for n in _top_level(tree) if n not in retained]


def stray_in_tree(tree: str = TREE) -> list[str]:
    """Runtime data that has come back inside the code tree. Empty is healthy.

    The candidate set is the tree, not a list: `KNOWN_GOOD_TOPLEVEL` subtracted
    from its top-level entries, plus the retained `RUNTIME_NAMES` whose nested
    paths the top-level scan cannot reach. What decides a candidate is
    `reachable_by_stray_check` — the same single-path predicate
    `agent_mcp._bash_tree_strays` labels a journal row with (#2172), so the label
    cannot drift from the alert that would have had to fire, the way two
    hand-maintained reach lists would. It applies, unchanged from before this
    extraction, the two rules that decide a candidate:

    * a retained runtime name is a stray unless git tracks everything under it.
      A name git tracks is committed on purpose, not written by a stray writer:
      `eval/baselines/` holds measurement records the harness reviews commit as
      evidence (#600, P4-P9), and it alerted every hour for them (9e98d0df).
      Such a name is a stray again once it also holds something untracked —
      ignored files included, because `--others` is called *without*
      `--exclude-standard` precisely so the gitignore cannot hide a writer;
    * any other candidate is a stray only when git tracks nothing at all under
      it. That half is what keeps an ordinary source directory which merely
      holds a build or test cache (`app/__pycache__`, `scripts/selfmod`,
      `web/.vite`, `chrome-extension/manifest.json`) off the alert.

    When git cannot answer, the open set stays shut and only the retained names
    are judged, on presence alone, as before #1541: with no index there is no
    way to tell `.venvs` from a stray writer, and a check that guesses wrong
    alerts every hour until someone turns it off."""
    tracked = _ls_files(tree)
    return sorted(n for n in _candidates(tree)
                  if reachable_by_stray_check(n, tree=tree, tracked=tracked))


#: An empty file younger than this may be a store its writer has only just opened
#: (`sqlite3` creates the file on connect and writes the header on first use), so it
#: is left for the next check rather than moved from under a live handle.
INERT_MIN_AGE_SECONDS = 600.0
def resolve_move_root(tree: str):
    """`(root, source, why)` for inert residue found in `tree` — the root it belongs to.

    The same three rules `app.paths` uses, read out of that tree's `app/data_root.py` by
    path, which is the file `policy.quarantine_dir` already reads for the destination's
    last segments. `LLOYD_DATA` wins; else the marked production root; else
    `<tree>/.lloyd-data` — so a sandbox, a round's worktree or a gate's prefetch checkout
    that finds residue in its own tree moves it into its OWN root instead of into the live
    one, which is what this root being an account-named literal used to make impossible
    (#3208).

    None is an answer, not a crash: no resolver in that tree, or a resolver that refused
    (`DataRootMissing` on an unmarked production root). The caller refuses the move; the
    stray stays put and keeps alerting.
    """
    return P.data_root_for_tree(tree, repo=tree)


#: Where inert residue goes, under the data root — read from the repo's data-layout
#: module, never spelled here. A move, so the hourly snapshots carry it and nothing is
#: ever deleted on this check's say-so.
#:
#: Until #2110 this was a two-segment `os.path.join` spelled inline in this file, and a
#: backlog contract of the same week named a different root for the same purpose, so a
#: reader chasing a moved stray had two candidate directories, neither of which any code
#: could be shown to have written. `policy.quarantine_dir` returns None when it cannot
#: read the layout, and this module treats that as a REFUSAL to move rather than a
#: default: relocating a file on an unverifiable path is the un-witnessed action the item
#: is about, so `_quarantine_dir` below raises and the stray keeps alerting, loudly.
#:
#: #3208 applies that same rule one level up, to the ROOT. `_quarantine_dir` used to
#: default `data_root` to this module's `DATA_ROOT`, and a read-only resolution always
#: answers — including from its fallback literal — which is exactly the guess a move may
#: not make. There is no default here: the root comes from `resolve_move_root` above, and
#: a caller with no answer gets a refusal.
def _quarantine_dir(tree: str, data_root: str) -> str:
    dest = P.quarantine_dir(repo=tree, data_root=data_root)
    if dest is None:
        raise RuntimeError(
            f"{tree}/app/data_root.py exposes no QUARANTINE_DIR_RELATIVE: refusing to "
            f"move inert residue to a destination this process could not read from the "
            "repo's own layout module — the file stays put and keeps alerting (#2110)")
    return dest
_SQLITE_SIDECARS = ("-wal", "-shm", "-journal")


def inert_residue(tree: str, names, data_root: str = DATA_ROOT,
                  now: float | None = None) -> list[str]:
    """The strays in `names` that are provably nothing: empty, idle, and a second
    copy of a store that lives in the data root.

    On 2026-10-02 a nightly task ran `cd ~/lloyd && sqlite3 workers.db "select …"`
    against a path it had guessed. `sqlite3` created the file to open it, the query
    failed, and a 0-byte `workers.db` sat in the tree for seven hours: an alert every
    hour, two backlog items, and both parked on a human because a file git ignores
    yields no diff for a round to land. Nothing was ever going to be learned from the
    file itself — every fact about it was in the first `lstat`.

    Every condition is a measurement of the file, and all must hold:

    * a retained runtime name at the top of the tree — the open-set strays
      (`.t`, a new tooling directory) are a classification a person makes, never this;
    * a regular file, not a link, one name, zero bytes: there is no content to lose;
    * untouched for `INERT_MIN_AGE_SECONDS`, with no SQLite sidecar beside it: no
      writer is mid-open;
    * the same name exists in the data root: the real store is elsewhere, so this
      one is a copy a wrong path made, not the only one there is.

    Anything else — one byte of data, a directory, a name nobody listed — still
    alerts exactly as before."""
    now = time.time() if now is None else now
    out: list[str] = []
    for name in names:
        if name not in RUNTIME_NAMES or "/" in name:
            continue
        path = os.path.join(tree, name)
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_size != 0 or st.st_nlink != 1:
            continue
        if now - st.st_mtime < INERT_MIN_AGE_SECONDS:
            continue
        if any(os.path.lexists(path + suffix) for suffix in _SQLITE_SIDECARS):
            continue
        if not os.path.lexists(os.path.join(data_root, name)):
            continue
        out.append(name)
    return out


def quarantine_inert(tree: str, names, data_root: "str | None" = None,
                     now: float | None = None) -> list[tuple[str, str]]:
    """Move each `inert_residue` file out of the tree. Returns `(name, destination)`
    for the ones that moved; a file that could not be moved is simply not in the list
    and goes on to alert.

    The move is an empty file created at the destination and the source unlinked,
    in that order, because the data root is its own subvolume and `rename` across it
    is EXDEV. The source is re-read immediately before the unlink and must be the
    same inode, still empty — a writer that arrived between the two reads keeps its
    file. One JSONL line per move records what the file was, so the quarantine
    directory is its own audit and not just a pile of empty files.

    `data_root` is the root the residue belongs to and there is deliberately no default
    for it (#3208). This function creates a file under that root, writes an audit line
    beside it, and `os.unlink`s the tree's copy — so the answer cannot come from a
    read-only fallback, which always answers even when the resolver refused. Left unset,
    it is resolved for `tree` by `resolve_move_root`, and an unresolvable root raises: a
    refusal is the same shape as `_quarantine_dir`'s missing-layout refusal, the file
    stays in the tree, and the hourly check keeps alerting on it.
    """
    now = time.time() if now is None else now
    if data_root is None:
        data_root, _source, why = resolve_move_root(tree)
        if data_root is None:
            raise RuntimeError(
                f"refusing to move inert residue out of {tree}: no data root resolves "
                f"for that tree ({why}) and the move would have to guess one — the file "
                "stays put and keeps alerting (#3208)")
    moved: list[tuple[str, str]] = []
    # Resolved lazily and deliberately: with nothing to move there is nothing to
    # relocate, and a tick that finds the tree clean must not raise because this process
    # cannot read a layout constant it does not currently need. The refusal fires on the
    # tick that WOULD move something, which is the only tick that needs an address.
    qdir: str | None = None
    for name in inert_residue(tree, names, data_root, now):
        src = os.path.join(tree, name)
        try:
            before = os.lstat(src)
            if qdir is None:
                qdir = _quarantine_dir(tree, data_root)
            os.makedirs(qdir, exist_ok=True)
            stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = os.path.join(qdir, f"{stamp}-{name}")
            os.close(os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            os.utime(dest, (before.st_atime, before.st_mtime))
            again = os.lstat(src)
            if (again.st_ino, again.st_dev, again.st_size) != (
                    before.st_ino, before.st_dev, 0) or not stat.S_ISREG(again.st_mode):
                os.unlink(dest)
                continue
            os.unlink(src)
            with open(os.path.join(qdir, "log.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "at": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
                    "source": src, "destination": dest, "size": 0,
                    "inode": before.st_ino,
                    "mtime": datetime.fromtimestamp(before.st_mtime, timezone.utc)
                    .isoformat(timespec="seconds"),
                }, sort_keys=True) + "\n")
        except OSError:
            continue
        moved.append((name, dest))
    return moved


#: Top-level folders of the data root that are rebuildable caches, not data:
#: never counted, so one emptying (a forced rebuild, a prune) cannot trip.
#: `code-graph` is the code graph's per-tree store (`app.paths.CODE_GRAPH_DIR`),
#: a nested subvolume the hourly snapshots skip for the same reason.
CACHE_DIRS = frozenset({"code-graph"})


#: The file extensions a data-root scan treats as a database. The SQLite sidecars above
#: are excluded by SUFFIX, so a 0-byte `usage.db-shm` never enters as a name at all.
_DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")
#: How deep under the data root the shadow scan looks, and how many directory entries it
#: reads before refusing to answer. Both bound a tick of a watchdog that must not block,
#: and neither is a limit the live root comes near: a full-depth walk of it on 2026-10-21
#: touched 115,307 directories and 663,207 files (778,514 entries) in 1.75 s, and its 90
#: database files all sit at depth 5 or shallower. They are runaway guards — a tree that
#: trips either bound gets an error, never a clean answer it did not measure, which is why
#: a bound sits above the measurement rather than at it: a root that gains a level of
#: nesting must not start reporting "no shadows" because the scan stopped early.
SHADOW_MAX_DEPTH = 6
SHADOW_MAX_ENTRIES = 1_000_000

#: There is deliberately NO byte bound on the content probe, and the reason is the
#: acceptance check this item was written against.
#:
#: An earlier draft carried `SHADOW_PROBE_MAX_BYTES = 8 MiB`, reasoned from the cost of
#: opening the root's populated stores — `workers.db` at 32 MB, `usage.db` at 40 MB, the
#: declared KG store at 172 MB. It was wrong in the one place that matters: the probe is
#: also how the check decides WHICH copy of a name to trust, the declared store is exactly
#: such a copy, and a declared store over the cap reads as "not measurable", so the pairing
#: fell back to a 24,576-byte witness copy under `_scratch/` and reported the live stray as
#: `declared: false` — the acceptance check's stated shape, missed by the cap. Measured on
#: the live root the same day: all 94 database-suffixed regular files, 9,460,412,416 bytes
#: of them, probed in 0.066 s total with none declining, because the query is
#: `select exists(select 1 from t)`, which costs one b-tree page read whatever the table
#: holds and returns at the first table that has a row. A cost bound on a probe whose work
#: does not scale with file size buys a false answer and no speed.

#: Tables that hold bookkeeping rather than content, so a store whose only row lives in
#: one of them is still EMPTY TO A READER (#3481).
#:
#: `meta` is `app/kg_store.py:_SCHEMA`'s key/value table, and `_init_schema` writes
#: `('schema_version', '1')` into it as part of PROVISIONING a database — so every store
#: `KGStore(path)` ever created has exactly one row and answers `select count(*) from
#: facts_idx` with a clean `0`. Counting that row would make the content test unsatisfiable
#: at the exact byte size of the live incident (81,920 bytes, one `meta` row, four empty
#: data tables), which is the shape this check was written for and did not see. Only
#: `meta` is here, on purpose: a name is decided by its copies under this root, so a store
#: that carries a table no sibling has is a different store, and its content counts.
SHADOW_BOOKKEEPING_TABLES = frozenset({"meta"})

#: `struct flock` for the `F_GETLK` query, in NATIVE alignment: `l_type` and `l_whence`
#: (2 bytes each), padding to 8, then `l_start`, `l_len` (8 each) and `l_pid` at offset 24.
#: 28 bytes on this x86_64 build, and the size is asserted rather than assumed: on an ABI
#: where the format stops being 28 the query declines to answer (`_flock_holder` checks
#: `calcsize`) instead of reading a layout it did not measure.
#:
#: What makes this constant a proof and not a guess: `fcntl.lockf` has no test mode on
#: Linux — `F_TEST` is a FreeBSD constant absent from its `fcntl.h`, and the first draft of
#: this probe died on that `AttributeError` — and `F_TLOCK` would ACQUIRE a write lock on
#: somebody's live database to learn the answer, so a wrong struct size here is not a loud
#: failure: it reads "free" for every candidate, which is a rail that never fires. Measured
#: on this box 2026-10-29 against a lock this process does not hold, a child taking
#: `F_SETLK(F_WRLCK)` made the query answer `l_type = 1` with `l_pid` 1194437, and the same
#: query answered `l_type = 2` (`F_UNLCK`) once the child died. The two child-process nodes
#: in `tests/test_guardian_data_root_shadow.py` re-run exactly that pair.
_FLOCK_STRUCT = "hhqqi"
_FLOCK_SIZE = struct.calcsize(_FLOCK_STRUCT)      # 28 on this x86_64 build


def _declared_db_relatives(repo: str) -> dict[str, str]:
    """`{basename: root-relative path}` for every `*_DB_RELATIVE` the repo declares.

    `app/data_root.py` spells the data root's store layout once — `KG_DB_RELATIVE` today,
    and `QUARANTINE_DIR_RELATIVE` beside it for a different purpose — so that a root or
    layout move is one edit (#1525, #2110, #3095: a stored command naming the old root
    returns a clean-looking non-answer). A check that re-spelled `_pipeline/vault-derived/
    kg.sqlite` would keep pointing at the retired layout after such a move and call a
    relocated store missing, which is the defect `policy.kg_db_path` was written for.
    Enumerating the `*_DB_RELATIVE` suffix rather than naming `KG_DB_RELATIVE` means the
    next store the layout module declares is consulted with no change here.

    Empty dict when the module cannot be loaded, and that is not an all-clear: the
    shadow verdict comes from the scan, and this only says WHICH store the repo itself
    declares for a name.
    """
    mod = P._data_root_module(repo)
    if mod is None:
        return {}
    out: dict[str, str] = {}
    for attr in dir(mod):
        if not attr.endswith("_DB_RELATIVE"):
            continue
        value = getattr(mod, attr, None)
        if value is None:
            continue
        rel = os.path.normpath(str(value))
        if not rel or rel == "." or os.path.isabs(rel) or rel.startswith(".."):
            continue                      # not a path under the root: not a store layout
        out.setdefault(os.path.basename(rel), rel)
    return out


def _flock_holder(path: str) -> tuple[bool, int]:
    """`(locked, pid)` for the POSIX record lock another process holds on `path`.

    (`False`, 0) answers only "no conflicting lock was found at the instant of the query".
    A SQLite connection takes an `F_SETLK` write lock on the main file while it commits and
    holds one on its `-shm` index for the life of the connection, so this is the query that
    distinguishes a database being written from one whose `-wal`/`-shm` pair a killed
    process left behind — which the bare existence test cannot do, and the live
    `/home/alansrobotlab/lloyd-data/kg.sqlite` proves: it has both sidecars, stamped
    2026-10-27 07:32 and 07:35, and no descriptor anywhere on the box holds either.

    Returns (`locked`, `pid_of_holder`), `pid` 0 when unlocked. `F_GETLK` answers for
    another process's lock only: POSIX record locks do not conflict within one process, so
    this can never be exercised by taking a lock here — which is why the two nodes that pin
    it spawn a child that locks the file and holds it across the check.

    The pid is measured and dropped rather than used. `F_GETLK` reports the pid of the
    process that owns the conflicting lock, and reporting it on an alert would be the
    oldest stale-data failure in this tree: a pid is reused, and a daily note naming one
    hours later accuses an innocent process (`_stale_pid_text`). The fact a human needs is
    the boolean — somebody is writing this file, do not touch it.

    (`False`, 0) is also the answer for a file that cannot be opened or queried. That is
    deliberately the same answer as "free": this function only ever EXCLUDES, so an
    unreadable file stays a candidate and the content probe, which must open it anyway, is
    the one that decides.
    """
    if _FLOCK_SIZE != 28:                    # not the ABI this layout was measured on
        return False, 0
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False, 0
    try:
        # `l_start` 0 and `l_len` 0 is the whole file, which is what makes the answer the
        # strict one: SQLite byte-range locks (a shared reader at one offset, the write
        # lock at another) all fall inside it, so any lock another connection holds counts.
        query = struct.pack(_FLOCK_STRUCT, fcntl.F_WRLCK, 0, 0, 0, 0)
        reply = struct.unpack(_FLOCK_STRUCT,
                              fcntl.fcntl(fd, fcntl.F_GETLK, query))
        held = reply[0] != fcntl.F_UNLCK
        return held, (reply[4] if held else 0)
    except OSError:
        return False, 0
    finally:
        os.close(fd)


def _held_by_another_process(path: str) -> bool:
    """Does some process hold `path`, or a SQLite sidecar beside it, under a record lock?

    This REPLACES the 0-byte class's `os.path.lexists(shadow + suffix)` test for the
    candidates this check now measures, and the reason is a measurement, not a preference:
    `/home/alansrobotlab/lloyd-data/kg.sqlite` carries `kg.sqlite-wal` (0 bytes, stamped
    2026-10-27 07:32) and `kg.sqlite-shm` (32,768 bytes, 07:35), and no descriptor on this
    box has either — a WAL connection that died leaves the pair standing over a store nobody
    has open, and SQLite unlinks them only when the LAST connection closes cleanly. Reading
    their existence as "a writer has it open" therefore suppressed the one finding #3481
    exists to report, so file-not-exists and file-not-held are now different questions and
    only the second one excludes.

    The lock query still excludes exactly what it used to where it mattered: a live SQLite
    writer holds a `F_SETLK` lock on the main file while it commits and one on the `-shm`
    index for the life of its connection, so both are asked. `tests/test_guardian_data_root_
    shadow.py` pins both directions with a child process that really takes the lock, because
    a same-process lock would not conflict with this process's own query and the rail would
    pass while doing nothing.

    A candidate the query cannot ask about (unreadable, vanished) is NOT excluded here:
    `_content_rows` has to open the same file to decide anything, and it declines on the
    same failure, so an unreadable candidate ends up unreported by that route rather than by
    a lock answer nobody measured.
    """
    for probe in (path, *(path + suffix for suffix in _SQLITE_SIDECARS)):
        if os.path.lexists(probe) and _flock_holder(probe)[0]:
            return True
    return False


def _read_open(path: str) -> "str | None":
    """The one URI query `path` may be measured with, or `None` to decline measuring it.

    Exactly one open per candidate, chosen from what stands beside the file, because the
    two safe-in-one-way opens are each unsafe in the other. Measured 2026-10-29 against
    hand-built fixtures on BOTH SQLite builds this box runs — 3.54.0 (`/usr/bin/python3`,
    which is what the guardian executes) and 3.53.1 (the venv, which is what the tests
    execute) — and the two agree on every row:

    * **No `-wal`, or a `-wal` with no bytes in it** → `mode=ro&immutable=1`. A log that
      holds no bytes holds no frames — WAL content is appended after a 32-byte header — so
      the main file is the whole store and reading it alone is the true answer (measured on
      a WAL-mode store whose rows were checkpointed and whose `-wal` was then re-created
      empty: `immutable=1` answered 3 rows, correct). It takes no lock and, unlike the
      other open, **creates nothing**: measured with the sidecar set absent before the open
      and empty after. The alternative open on the same file created `kg.sqlite-wal` AND
      `kg.sqlite-shm` — which is the review's finding on clause 3 of #3481, and the reason
      the WAL-aware open is no longer tried first: a probe that manufactures a sidecar
      writes into a person's data root and leaves a file the next tick has to explain.
    * **A `-wal` with bytes in it, and a `-shm` beside it** → `mode=ro`. Only a WAL-aware
      open applies the log, and `immutable=1` on such a file answers **0 rows for a store
      with 3 committed un-checkpointed rows in it** (measured, both builds). That is
      clause 2's forbidden error — an alert telling a person to delete a populated store —
      so the cheaper open is not offered here even though it creates nothing. With the
      index already standing there, this open creates nothing either (measured: sidecar set
      identical before and after, and the store was not checkpointed by the read).
    * **A `-wal` with bytes in it and NO `-shm`** → `None`, declined. `immutable=1` would
      lie about it (measured: 0 rows, same file `mode=ro` answers 3), and `mode=ro` has to
      BUILD the index, which creates `<db>-shm` (measured: `-shm` absent before the open,
      32,768 bytes of it after). This check does not write into the data root, so the
      candidate goes unmeasured rather than being measured wrongly: an unanswered question
      is not emptiness (#2056), and this is the re-entry case the module already refuses on
      for a read that fails partway. The shape is a WAL connection that died between
      writing its log and materialising its index; it is re-measured every tick, and the
      `test_a_wal_log_without_its_index_is_declined_and_never_written_into` node pins it, with the
      lying open and the writing open each reproduced as a control inside that same node.

    `os.stat` and not `os.path.lexists` for the log's bytes: one syscall answers both
    "is it there" and "does it hold anything", and a zero-length `-wal` is the ordinary
    residue of a checkpointed-and-killed connection — the live
    `/home/alansrobotlab/lloyd-data/kg.sqlite` has exactly one, 0 bytes, stamped
    2026-10-27 07:32, which is why the empty-log branch is the load-bearing one.
    """
    try:
        wal_bytes = os.stat(path + "-wal").st_size
    except OSError:
        return "mode=ro&immutable=1"       # no log beside it: the main file is the store
    if wal_bytes == 0:
        return "mode=ro&immutable=1"       # a log with no frames in it holds no content
    return "mode=ro" if os.path.lexists(path + "-shm") else None


def _content_rows(path: str, size: int) -> tuple[bool, int, int]:
    """Measure whether `path` holds content, opening it read-only: `(ok, tables, rows)`.

    `rows` is whether any data table holds a row, reported as 0 or 1 — the only question
    the caller asks is "is any of this readable content", and
    `select exists(select 1 from t)` costs one b-tree page read whatever the table holds,
    so a populated store is answered as cheaply as an empty one and the probe's cost does
    not grow with file size (which is why no byte bound was kept; see the note above
    `SHADOW_BOOKKEEPING_TABLES`). `ok=False` means the open was declined or could not be
    made or the query did not answer: not a database, unreadable, a header too torn to
    parse, or a WAL log standing without its index (`_read_open`). Then `tables`/`rows`
    are meaningless, and an unanswered question is NOT emptiness — #2056's rule applied to
    content. `size` is the caller's measured byte count and is not used: after #3481 bytes
    decide nothing here, and the parameter stays because every caller passes its own
    measurement and a probe that re-`stat`ed would be reading a different instant.

    Which single open is chosen, and why exactly one, is `_read_open`'s subject: the safe
    choice depends on whether a write-ahead log beside the file holds bytes, and both of
    the two candidate opens is wrong for one of those shapes — one creates a sidecar, the
    other reads a populated store as empty. A read that fails partway is likewise NOT an
    answer of zero rows: the WAL-aware open reads real file locks, so on a store another
    process is mid-commit in it can raise `database is locked`, and a lock error recorded
    as "empty" would hand a human a POPULATED store to delete. Such a candidate is
    declined and re-measured on the next tick — which is also why `_held_by_another_process`
    asks the kernel about the lock first: a live writer should be excluded before its file
    is opened at all.

    Threading mode is not assumed: SQLite's URI docs name `SQLITE_OPEN_READONLY` and
    `SQLITE_OPEN_FULLMUTEX` without saying which mode URI `mode` selects, so the connection
    is used on this thread, closed before returning, and never shared. `immutable=1` also
    asserts the file cannot change underneath the read; if that is false for a file, the
    query raises and `ok=False` turns the candidate away, which is the direction that
    cannot call a populated store empty.
    """
    query = _read_open(path)
    if query is None:
        return False, 0, 0
    try:
        con = sqlite3.connect("file:" + _uri_encode_ro(path) + "?" + query, uri=True)
    except sqlite3.Error:
        return False, 0, 0                  # this open cannot be made: nothing is known
    try:
        names = [row[0] for row in con.execute(
            "select name from main.sqlite_master "
            "where type = 'table' and name not like 'sqlite_%'")]
        data_tables = [n for n in names if n not in SHADOW_BOOKKEEPING_TABLES]
        rows = 0
        for name in data_tables:
            # A table cannot be parameterised, so the identifier is quoted the only
            # safe way: doubled, and inside double quotes SQLite treats it as one
            # identifier whatever it contains. A malicious name cannot reach here
            # anyway — `sqlite_master` reports it only for a file that really is a
            # SQLite database, which this probe just proved.
            if con.execute('select exists(select 1 from main."%s")'
                           % name.replace('"', '""')).fetchone()[0]:
                rows = 1
                break                  # populated: the answer, no further table asked
        return True, len(data_tables), rows
    except sqlite3.Error:
        return False, 0, 0
    finally:
        _close_quietly(con)


def _uri_encode_ro(path: str) -> str:
    """Percent-encode everything sqlite3's URI parser would eat: `?`, `#`, `%`, `"` and
    the control bytes, leaving `/` and every other byte intact. sqlite3's own
    `filename_uri` test suite (test/e_data1.test) is what pins its grammar to
    scheme + path + `?`-parameters, and `?` in a path segment is the realistic failure: a
    file named `usage.db?mode=ro` would otherwise ask for a parameter twice.
    """
    out = []
    for byte in os.fsencode(path):
        if byte in (0x3F, 0x23, 0x25, 0x22) or byte < 0x20 or byte == 0x7F:
            out.append("%%%02X" % byte)
        else:
            out.append(chr(byte))
    return "".join(out)


def _close_quietly(con) -> None:
    try:
        con.close()
    except Exception:      # pragma: no cover - close on a read-only handle has no failure
        pass


def data_root_shadows(data_root: str = DATA_ROOT, repo: str = TREE,
                      now: float | None = None) -> list[dict]:
    """Every content-empty database in the data root whose name holds content elsewhere
    under the same root, as `(shadow, trusted)` pairs. READ-ONLY: never creates, moves,
    truncates or unlinks anything, and opens every candidate it tests in a mode where a
    write is not possible.

    Two live shapes of one hazard, both named here. (1) `kg.sqlite` at 0 bytes at the top
    while the knowledge graph the code opens — `app.paths.VAULT_KG_DB`,
    `_pipeline/vault-derived/kg.sqlite` — holds 160,882,688 bytes. A 0-byte file is a valid
    *empty* SQLite database, so a probe that opened the bare name from the wrong cwd does
    not get an error; it gets 0 rows, and prints "the KG is empty" about a populated store.
    (2) Since #3481, the same file at 81,920 bytes: `KGStore('/home/alansrobotlab/lloyd-data/
    kg.sqlite')` run from an agent's Bash heredoc at 2026-10-26 23:47 laid the SHIPPED
    schema plus a `meta` version row and no data, which answers `select count(*) from
    facts_idx` with a clean `0` while the declared store held 178,001 facts — the identical
    false reading, at a byte size the 0-byte predicate could not see. Both came from the
    same origin as the 2026-10-02 in-tree `workers.db` story: `sqlite3` creates the file on
    connect, and `KGStore(path)` provisions an absent path by design (#1236's guard is
    `_require_database`, which covers the READ route only). The class is therefore content,
    not size; the 2026-08-23/24 incident named above is what the check exists for, and it
    happened twice in opposite directions (`_pipeline/workers.db` empty beside the populated
    queue store at the top), which is why the check runs both ways.

    Why a new predicate and not the existing two. `stray_in_tree` is scoped to the CODE
    tree by construction — its candidates are git's untracked entries, and the data root
    is not a repo, so it returns `[]` here and always will. `inert_residue` does the
    right arithmetic but cannot reach this file from either side: a name must be in
    `RUNTIME_NAMES` (a prediction, and `kg.sqlite` is not in it), nested names are skipped
    outright, and the "the real store is elsewhere" test looks only at the data root's
    TOP level (`os.path.join(data_root, name)`), which is never where a
    `_pipeline/vault-derived/` store lives. Handing it the data root as its tree returns
    `[]` for both live pairs. The property is a name at one depth whose populated
    counterpart sits at another depth under one root, so the fix enumerates the root and
    compares names — no name list to maintain, and no new entry in `RUNTIME_NAMES`, which
    is the axis the item rules out.

    Both halves of a pair must be measurements. The populated half is a file with bytes in
    it and at least one row; the empty half is a file the probe could open and that answered
    no row for any data table, at ANY byte size (#3481: the 0-byte-only rule was the blind
    spot this item exists to close). The rest of the exclusions are the ones `inert_residue`
    earns its right to act on: a regular file, one link, untouched for
    `INERT_MIN_AGE_SECONDS`, and no SQLite writer holding it — which for a candidate that
    holds bytes is a `F_GETLK` query (`_flock_holder`) and not the bare existence of a
    sidecar. Existence is what the 0-byte rule used, and it would have kept this finding
    unreported: the live `kg.sqlite` has a 0-byte `-wal` and a 32,768-byte `-shm` beside it
    from a WAL connection that died on 2026-10-27, with no descriptor on either. `CACHE_DIRS`
    is not scanned (a rebuild in progress is not a shadow), and the scan does not follow a
    symlink or descend past `SHADOW_MAX_DEPTH`.

    Reading content is why this check opens files it used to only `lstat`, so the four
    things that make that safe are each pinned by a node in
    `tests/test_guardian_data_root_shadow.py`: the one open it uses is chosen by whether a
    write-ahead log beside the candidate holds bytes (`_read_open`), so the probe never
    writes — a `-wal`/`-shm` pair is not created beside a candidate, which the WAL-aware
    open alone would do, and the pair is this check's residue to explain rather than
    something it may manufacture — while a store whose newest rows are still sitting
    un-checkpointed in its log is measured as POPULATED and never as empty, which the
    sidecar-safe `immutable=1` open alone would do; a candidate that can only be measured
    by writing an index into the root is DECLINED, not measured; a file the probe cannot
    parse is never called empty, so a database-named file whose bytes are not a database
    stays untouched; and a database another PROCESS holds with an `F_SETLK` lock is left
    alone, which is the rail the existence rule was standing in for.

    The trusted half is the populated copy the ROOT itself declares — whichever store
    `app/data_root.py` names for that basename via a `*_DB_RELATIVE` constant — and it
    wins over every other populated copy of that name even a shallower or larger one,
    because the finding's purpose is to send a human to the file the code actually opens.
    Only when the layout module names no store for the basename does the check fall back to
    the shallowest populated copy (then largest), because a root's own top level is what its
    code conventionally opens (`WORKERS_DB` and `USAGE_DB` are `DATA_ROOT / name`).
    `declared` in each finding says which of the two it was, so the alert never guesses
    where the real store is.

    A root that cannot be read RAISES rather than returning `[]`. #2056 is the reason:
    substituting an empty list for a failed measurement retracted three open sections of
    `memory/2026-10-02.md` while the same title was still alerting an hour later. The
    caller catches this and says nothing in either direction.

    Report-only on purpose. `quarantine_inert` MOVES code-tree residue into
    `QUARANTINE_DIR_RELATIVE`; a data-root file is a person's call — the item's own
    reasoning, and `guardian.py`'s existing `needs_human` split for a retained name. The
    removal of the two live 0-byte files is a human action under `~/lloyd-data`."""
    now = time.time() if now is None else now
    root = os.path.abspath(data_root)
    if not os.path.isdir(root):
        raise OSError(f"data root is not a readable directory: {root}")
    declared = _declared_db_relatives(repo)

    # One bounded walk: basename -> [(depth, size, path)]. Depth is levels below the
    # root, so a top-level file is 1 and `_pipeline/workers.db` is 2.
    found: dict[str, list[tuple[int, int, str]]] = {}
    seen = 0
    truncated = False
    stack: list[tuple[str, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for de in entries:
                seen += 1
                if de.is_dir(follow_symlinks=False):
                    if depth == 0 and de.name in CACHE_DIRS:
                        continue
                    if depth + 1 < SHADOW_MAX_DEPTH:
                        stack.append((de.path, depth + 1))
                    continue
                if not de.name.endswith(_DB_SUFFIXES):
                    continue
                try:
                    st = de.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    found.setdefault(de.name, []).append(
                        (depth + 1, st.st_size, de.path))
        if seen > SHADOW_MAX_ENTRIES:
            truncated = True
            break

    out: list[dict] = []
    for name, recs in sorted(found.items()):
        populated = sorted(((p, d, sz) for d, sz, p in recs if sz > 0),
                           key=lambda r: (r[1], -r[2], r[0]))
        if not populated:
            # Every copy of this name is byte-empty: there is no store here to shadow,
            # and no reader can be pointed at a populated one. (#3481 moved the empty half
            # of the pair off bytes and onto the row probe below, which is why the byte
            # test survives on the POPULATED side: a file with no bytes cannot hold a row.)
            continue                      # one side of the pair is missing: no shadow
        # The store to trust is the one the repo DECLARES for this name whenever that copy
        # is itself populated; the shallowest-then-largest sort is only the fallback for a
        # name the layout module does not declare. The order is not cosmetic: a populated
        # `sub/kg.sqlite` at depth 2 outranks the declared `_pipeline/vault-derived/
        # kg.sqlite` at depth 3 on depth alone (and a bigger undeclared copy outranks it on
        # size), and the alert's whole job is to name the file a human should open and
        # count. Sorting first and looking up the declaration afterwards also mislabels the
        # finding `declared=False` for a root that does declare the store, which is the
        # difference between "this came from the layout module" and "the check guessed".
        # Both sides are compared by `normpath` because a declared relative path and a
        # scandir path are two spellings of one file.
        rel = declared.get(name)
        declared_path = os.path.normpath(os.path.join(root, rel)) if rel else None
        # One measurement per path per name per tick (`measured`): the trusted pick and the
        # candidate sweep read through this, so three copies of a name cost three probes.
        measured: dict[str, tuple[bool, int, int]] = {}

        def content(path: str, size: int) -> tuple[bool, int, int]:
            if path not in measured:
                measured[path] = _content_rows(path, size)
            return measured[path]

        # The store to TRUST is the copy that holds rows, and the declared one wins when it
        # does. #3481's row measure has to reach this pick too: `populated` is sorted by
        # bytes, and a zero-row copy of a name can be the biggest file under it (the live
        # `kg.sqlite` at 81,920 bytes sat in `populated[0]` in a fixture where the real
        # store was smaller), so trusting bytes here would name an empty file as the store
        # a human should count — inverting the one thing the alert exists to say. The
        # declared copy is tried first so `declared: True` means what #3127 says it means;
        # the (depth, then size) order is the fallback for a name the layout does not
        # declare, and it is not cosmetic — a 2-deep copy outranks a 3-deep one on depth.
        # The age, link-count and lock rails below are NOT applied to this pick: they exist
        # because a finding invites a person to delete, and nothing is invited here. A torn
        # read of a database mid-commit makes `content` decline, which leaves the name
        # unpaired for this tick and re-measured on the next one.
        ordered = populated
        if declared_path:
            ordered = sorted(
                populated,
                key=lambda r: (os.path.normpath(r[0]) != declared_path, r[1], -r[2], r[0]))
        measured_populated = [(p, d, sz) for p, d, sz in ordered
                              if content(p, sz)[0] and content(p, sz)[2]]
        if measured_populated:
            trusted = measured_populated[0]
        else:
            # Nothing under this name is a readable database WITH rows in it. The copies may
            # still be databases that are all empty, or files that are not SQLite at all —
            # `state.db` ships as a Berkeley DB (`policy.RUNTIME_DATA_NAMES`), and its probe
            # declines for both halves — so this falls back to the pre-#3481 answer, the
            # byte-populated copy the sort puts first. Requiring a readable populated half
            # would silently un-report a 0-byte `state.db` shadowing a real Berkeley one,
            # which is coverage this check has had since #3109 and this round does not own.
            # A name whose copies are ALL readable-and-empty lands here too, and the
            # candidate sweep below reports each non-trusted copy on its own measurement.
            trusted = ordered[0]
        trusted_path, trusted_depth, trusted_size = trusted
        trusted_norm = os.path.normpath(trusted_path)
        # Every copy of the name except the trusted one is a candidate, and content is what
        # decides — the byte size only ever ordered the old rule.
        for shadow, depth, size in sorted(((p, d, sz) for d, sz, p in recs),
                                          key=lambda r: (r[0], r[1], r[2])):
            if os.path.normpath(shadow) == trusted_norm:
                continue                  # the store to trust is never its own shadow
            try:
                st = os.lstat(shadow)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                continue
            if now - st.st_mtime < INERT_MIN_AGE_SECONDS:
                continue
            if st.st_size != size:
                # The file changed between the walk and the decision, so this tick's
                # measurement is not this file's measurement. Skip; the next tick re-measures.
                continue
            if _held_by_another_process(shadow):
                continue
            readable, data_tables, data_rows = content(shadow, size)
            if not readable or data_rows:
                # Either the probe could not answer — an unreadable file is not an empty
                # one, #2056 — or the copy holds content, which makes it a store.
                continue
            out.append({
                "shadow": shadow, "name": name, "depth": depth,
                "trusted": trusted_path, "trusted_depth": trusted_depth,
                "trusted_size": trusted_size,
                "declared": declared_path is not None
                and trusted_norm == declared_path,
                "declared_relative": rel,
                # Measured content, and the bytes it sits in. Both go on the alert: after
                # #3481 a finding's byte size says nothing about its emptiness, so the row
                # count is the number a person decides on.
                "shadow_size": size, "data_tables": data_tables,
                "data_rows": data_rows,
            })
    out.sort(key=lambda f: f["shadow"])
    if truncated:
        # The bound was hit, so an absence after here is a scan that stopped, not a
        # root that is clean. Loud rather than silent, and never a claim of all-clear.
        raise OSError(f"data-root shadow scan stopped after {seen} entries under {root} "
                      f"(SHADOW_MAX_ENTRIES); its silence would be a partial measure")
    return out


# ---------------------------------------------------------------------------
# Names that newly appear at the data root's top level (#3210)
# ---------------------------------------------------------------------------
#
# The four detectors that look at this directory answer four other questions. This one
# asks WHAT IS NEW HERE, which none of them asks, and answers it as data for a person.
#
# `data_root_shadows` above answers "is this 0-byte file SHADOWING a populated store of
# the same name", and to do that cheaply it drops every name that does not end in a
# database suffix (`_DB_SUFFIXES`, applied before the pairing test) — so a file named
# `false` or `sqlite3`, which is what a mistyped command actually leaves behind, is never
# even learned by that walk. `_bash_tree_strays` snapshots the root's top level per Bash
# call and then narrows to new directories holding a checkout and new 0-byte files whose
# name ends in a database extension, ruling the rest out on purpose ("a placeholder… left
# alone"). `inert_residue`/`quarantine_inert` is scoped to the CODE tree. And the tripwire
# only ever speaks of names that VANISHED.
#
# So `~/lloyd-data/false` (0 bytes, mtime 2026-10-12) and `~/lloyd-data/sqlite3` (0 bytes,
# mtime 2026-10-22) have stood at the live root's top level for weeks, inside every hourly
# snapshot, named by nothing. The residue of a mistyped `sqlite3 <db>` is not dangerous;
# it is informative — it says a command in this system is writing a file nobody intended —
# and the root's inventory is the thing the snapshot timer, the tripwire's baseline and
# `data_root_shadows` all measure, so a name that arrives silently moves all three.
#
# Report-only, and that is load-bearing: #3109 ruled the same for the half of this walk
# that found shadows, because the guardian cannot know whether a file in a person's root
# is theirs or a mistake. `quarantine_inert`'s "provably inert" tests are for git-ignored
# CODE-TREE residue and are never offered a data-root name. Nothing here unlinks, renames,
# moves or creates; the decision to delete the live pair stays a human action under
# `~/lloyd-data`, exactly as `data_root_shadows`' docstring rules for the pair it named.

#: The leg's own previous scan, persisted. Not the tripwire's snapshot: `VaultWatch.tick`
#: runs every `TICK_SECONDS = 5.0`, so a snapshot would already hold a file created
#: minutes before this hourly leg runs, and "absent from the previous measurement" would
#: degrade to "created in the last five seconds" — naming almost nothing, forever.
ARRIVALS_STATE_FILE = "data-root-arrivals.json"


def top_level_regular_files(data_root: str = DATA_ROOT) -> dict[str, float]:
    """{name: mtime} for every regular file standing directly under `data_root`.

    No name filter of any kind — that is the whole point of this leg, whose subject is a
    name shaped like `false` or `sqlite3`. Symlinks, directories, sockets and devices are
    not regular files and are not this question; `follow_symlinks=False` so a link is
    never followed or read, and the size is deliberately NOT part of the key because
    `false` and `sqlite3` are 0 bytes and the class does not stop being interesting when
    it grows.

    #2056's rule, and #3109's for exactly this walk: a root that cannot be read RAISES.
    An empty mapping here would mean "nothing is new", which is the value that CLOSES an
    alert — the same confusion that made the shadow walk raise instead of answering `[]`,
    where a root that could not be opened reported a root that had nothing to quarantine.
    A missing root raises too: `app.data_root` exists because an absent data root reading
    as empty is the larger version of this same bug.
    """
    out: dict[str, float] = {}
    with os.scandir(data_root) as it:                  # OSError propagates: unreadable ≠ clean
        for de in it:
            try:
                st = de.stat(follow_symlinks=False)
            except OSError:
                continue                               # a name that will not stat is not a finding
            if stat.S_ISREG(st.st_mode):
                out[de.name] = st.st_mtime
    return out


def read_arrivals_baseline(state_dir, data_root: str = DATA_ROOT) -> "dict | None":
    """The names this leg has already seen, or `None` when it has never run.

    `None` is a real third state and must not collapse to `{}`: a leg that re-armed from
    an empty baseline would, on every restart, report the entire legitimate inventory
    (sessions marker, `workers.db`, the snapshot marker) as a fresh human decision, which
    is the noise that gets a new alert switched off. The recorded root is part of the
    state — a baseline measured against one root is not a baseline for another, and #1415
    is about a guardian answering about the wrong root — so a root change re-arms.
    """
    p = Path(state_dir) / ARRIVALS_STATE_FILE
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or state.get("root") != os.path.realpath(data_root):
        return None
    seen = state.get("seen")
    return seen if isinstance(seen, dict) else None


def write_arrivals_baseline(state_dir, files: "dict[str, float]", now: float,
                            data_root: str = DATA_ROOT) -> None:
    """Record what the leg saw, MONOTONICALLY: the union of what it saw before and now.

    A snapshot of this hour alone would make a sqlite sidecar that legitimately comes and
    goes (`usage.db-wal` removed at checkpoint, recreated by the next writer) a recurring
    human-decision alert. A name the leg has once seen stays seen, so an arrival really is
    a first appearance. `V._atomic_write` is the tmp+rename the rest of this file's state
    uses, so a crash cannot leave a half-written baseline that would re-arm the leg.
    """
    state_dir = Path(state_dir)
    previous = read_arrivals_baseline(state_dir, data_root) or {}
    seen = {**previous, **files}
    state_dir.mkdir(parents=True, exist_ok=True)
    # tmp + `os.replace`, the same shape `VaultWatch._persist` uses for its own state file:
    # a partially-written baseline would parse as corrupt, and a corrupt baseline re-arms
    # the leg, which is the one failure that hides a real arrival for an hour.
    tmp = state_dir / f"{ARRIVALS_STATE_FILE}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({
        "root": os.path.realpath(data_root), "seen": seen, "updated": now}, indent=1),
        encoding="utf-8")
    os.replace(tmp, state_dir / ARRIVALS_STATE_FILE)


def _days_ago(mtime: float, now: float) -> int:
    return max(0, int((now - mtime) // 86400))


def _sidecar_of_store_in_this_scan(this_name: str, files: "dict[str, float]") -> bool:
    """True when `this_name` is `<base><suffix>` for a store standing in this same scan.

    SQLite makes these names itself: opening a store in WAL mode creates `<store>-wal` and
    `<store>-shm` beside it, and the last connection to close cleanly removes them again.
    So a first-time `usage.db-wal` beside a `usage.db` the root already carries is not a
    name that arrived — it is a bookkeeping file of a store the alert would name by its own
    base, and the decision it asks for cannot be made by the person reading it: by the time
    they do, the file is either gone on its own or it is a live database's write-ahead log,
    where deleting it loses committed writes (#3229, which filed `#3227` for
    `research.db-wal`/`-shm` and cost the two-day-old snapshot incident its ALERT.md slot).

    The pairing is the rule, NOT the suffix. #3210's clause 1 rejects a suffix filter
    outright, and a suffix-only mute would be that filter wearing a different name: it would
    also hush a `-wal` whose store is nowhere at the root, which is exactly the unexplained
    name this leg exists for. `base in files` is therefore required, from the SAME scan the
    findings come out of — no new filesystem call, and no reliance on what `previous` held:
    a store that is new too has its OWN name reported, which is the news worth having.
    """
    for suffix in _SQLITE_SIDECARS:
        if this_name.endswith(suffix):
            base = this_name[:-len(suffix)]
            return bool(base) and base in files
    return False


def new_top_level_names(files: "dict[str, float]", previous: "dict | None",
                        *, now: "float | None" = None) -> list[dict]:
    """The regular files at the root's top level that `previous` does not already hold.

    Novelty is the rule, with the one exception `_sidecar_of_store_in_this_scan` explains:
    a WAL sidecar whose own store stands in this same scan is that store, not a new name.
    `data_root_shadows` gates on `_DB_SUFFIXES` and on a populated same-basename copy
    elsewhere, `_bash_tree_strays` gates its file half on `EMPTY_DB_SUFFIXES`; both gates
    are the reason the live `false` and `sqlite3` are unnamed today, so this leg must not
    re-introduce either — and the sidecar exception is not a suffix test, because it holds
    only for a base the same scan also carries.

    `previous=None` means no previous measurement, and the answer is NOTHING: the leg
    ARMS on the world as it stands and reports no arrivals. That is what clause 5's "a
    name already present when the watcher armed" requires, and the alternative is worse
    than silent — a baseline seeded from `KNOWN_GOOD_TOPLEVEL` would report the live
    root's seventeen legitimate residents (the marker, `workers.db`, `usage.db`, a dozen
    state files) as news on the first pass, and a wall of expected names buries the two
    this leg exists for. Consequence, stated rather than hidden: two stubs that were
    standing BEFORE this leg shipped are never named by it. `false` and `sqlite3` are
    therefore owed to whoever deletes them, which is where #3210's own acceptance already
    puts them ("removal of the two live 0-byte files is a human action", datawatch.py:543).
    A first pass must still WRITE the baseline, or every guardian restart would re-arm
    against a fresh empty file and the leg would never see a second measurement.
    """
    if previous is None:
        return []
    # The sidecar mute reads the CURRENT scan (`files`), not the findings: the pairing that
    # explains the name is a fact about what stands at the root now. And muting here cannot
    # lose the baseline, which guardian.py writes from its own `arrivals_files` scan rather
    # than from this list — so a muted name is still recorded as seen, and never re-reported.
    out = [{"name": name, "mtime": mtime}
           for name, mtime in sorted(files.items())
           if name not in previous and not _sidecar_of_store_in_this_scan(name, files)]
    if out and now is not None:
        for f in out:
            f["age_days"] = _days_ago(f["mtime"], now)
    return out


def arrivals_alert(arrivals: list[dict]) -> "dict | None":
    """The `state["strays"]` dict for these findings, or None when there are none.

    `needs_human=True` is the #3109 routing and it is the whole alert: a name in a
    person's data root is their decision, and the guardian's job is to say which name and
    since when, not to act. No `quarantine` key and no `staged` key is emitted, so there
    is nothing for the resolve path to consume — those two keys are what a STRAY carries,
    and an arrival is not a stray. The date is rendered because a stub that has been
    standing for nine days and a file made this turn are different questions asked of
    different people; a bare "a new file appeared" about a ten-day-old `false` invites the
    correct answer, which is "then it is not news".
    """
    if not arrivals:
        return None
    detail = "; ".join(
        f"{f['name']} — mtime "
        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(f['mtime']))}"
        + (f" ({f['age_days']} days ago)" if "age_days" in f else "")
        for f in arrivals)
    return {
        "needs_human": True,
        "reason": (f"{len(arrivals)} new name(s) at the data root's top level: {detail}. "
                   "Report-only: the guardian moves nothing, and the decision is yours."),
        "evidence": detail,
    }


class DataWatch(V.VaultWatch):
    what = "data"
    skip_top = CACHE_DIRS
    state_file = "data_watch.json"
    marker_file = "data-tripped.json"

    def __init__(self, root: str = DATA_ROOT, state_dir: Path = V.GUARDIAN_STATE):
        super().__init__(root, state_dir)

    @property
    def armed(self) -> bool:
        return bool(self.history)

    def _excuse_vanished(self, name: str, count: int,
                         before: V.Snapshot) -> "str | None":
        """A top-level folder this root measured as a git checkout, or watched come
        into being, is scratch rather than store. Two exemptions, one floor, and the
        vault has none of them.

        WHY IT IS ALLOWED HERE AND NOT FOR THE VAULT. The data root is the one
        watched root a session is TOLD to write into, and since 2026-10-20 it is
        also where `tests/conftest.py:241-247` sends a session that needs a
        checkout: the refusal that fires when the suite is invoked from the
        production tree literally prints
        `git worktree add --detach ~/lloyd-work/check-$$ HEAD`. A session that
        instead runs `git worktree add ~/lloyd-data/triage-3087/check HEAD` has
        created a top-level folder of thousands of files that it is *expected* to
        delete when it finishes, and the folder rule was built to fire on exactly
        that shape. On 2026-10-20 14:24:14 it did: `triage-3087` (3347 files) went
        from present to absent, promotions halted, the worker pool paused,
        snapshots refused — and the before/after measurement differed in nothing
        else but `event_logs` (+1). No data was missing.

        WHY A MEASUREMENT AND NOT A NAME PATTERN. `triage-<id>` and `check-<pid>`
        are named by whoever runs the command, so matching them is a
        hand-maintained list over an open set, which is the standing ruling from
        #2160 and the reason `skip_top` was not extended here. The flag comes from
        `measure` seeing a `.git` entry inside the folder (depth bounded by
        `vaultwatch.CHECKOUT_MAX_DEPTH`), so it is true of a folder because of
        what is IN it.

        WHY `RUNTIME_NAMES` IS A FLOOR, NOT AN ALLOWLIST. A repo cloned two levels
        down inside a real store would flag that store as a checkout —
        `_pipeline/` is where a tool cloning something would land — and losing
        `_pipeline` is the event this tripwire exists to catch. So a name the root
        already knows as a runtime store is never excused, whatever its flag says.
        That list NARROWS the exemption, which is the opposite direction of failure
        from the allowlist #2160 forbids: it can only ever make the tripwire
        stricter, and every store on it was named before this rule existed.

        WHY A LIFETIME ALSO EXCUSES (#3153). The checkout flag above is one bit wide:
        it covers the 2026-10-20 folder because that one held a `.git`, and nothing
        else. A job that runs `mkdir ~/lloyd-data/<job>-<id>`, writes a few thousand
        files, and deletes them on success leaves no such entry, so the folder rule
        still read its clean-up as a wiped store and still halted promotions — measured
        2026-10-22 by running the real watcher over that exact shape. What distinguishes
        it is not in the folder's bytes at all, it is in the watcher's own record:
        `Snapshot.first_seen` says which measurement first saw this name, and a folder
        the watcher watched come into being between two measurements is not one of the
        stores this root has always had. So the second exemption reads the age of that
        sighting against `SCRATCH_HORIZON_SECONDS`, which is why it is behind the
        checkout branch and not in front of it — where both would excuse a folder, the
        journal should carry the stronger of the two reasons.

        WHY AN ABSENT STAMP IS NOT A YOUNG FOLDER. `VaultWatch._stamp_births` records a
        sighting only for a name the previous measurement did not have, and records
        nothing at all at arming. So a folder that was already here when the guardian
        started has no entry, the same entry a state file from before this field
        carries for every folder, and both read as no exemption. The exemption is
        granted by a measurement this watcher made, and the absence of that measurement
        is not evidence of anything (#3153, and the same schema-addition rule
        `checkouts` follows above).
        """
        if name in RUNTIME_NAMES:
            return None
        if name in before.checkouts:
            return (f"top-level folder {name!r} ({count} files) went missing; not a trip — "
                    f"the last measurement saw a {V.CHECKOUT_ENTRY} entry inside it, so it "
                    f"was a git checkout and not one of this root's stores (#3095)")
        # The non-checkout half of the same shape (#3153), bounded by the lifetime the
        # watcher measured rather than by anything the folder is called. An absent stamp
        # is not a young folder: it is a folder this watcher never saw arrive, which
        # includes every folder on a root whose state predates this field.
        born = before.first_seen.get(name)
        if born is None:
            return None
        age = before.ts - born
        # A negative age is a record that cannot be true — a state file edited by hand,
        # or the boot clock jump of the shape 2026-10-18 produced — and the reading of an
        # impossible record is the trip, not a pass.
        if not 0.0 <= age < SCRATCH_HORIZON_SECONDS:
            return None
        # Says nothing about WHICH of the folder rule's two states this is — the hook is
        # called for a vanished folder and an emptied one alike, and `evaluate`'s own
        # reason carries that word.
        return (f"top-level folder {name!r} ({count} files) is no longer counted; "
                f"not a trip — "
                f"this watcher first saw the folder {age:.0f}s before that measurement, "
                f"inside the {SCRATCH_HORIZON_SECONDS / 3600.0:g} h scratch horizon, so "
                f"it came into being and went away while the watcher was running, and no "
                f"store of this root's was ever created between two measurements "
                f"(#3153)")

    def judge(self, snap: V.Snapshot | None,
              excused: "list[str] | None" = None) -> str | None:
        if not self.armed:
            return None
        if snap is not None and not os.path.isfile(os.path.join(self.root, ROOT_MARKER)):
            return f"{self.root} lost its {ROOT_MARKER} marker"
        return super().judge(snap, excused)

    def tick(self, now: float | None = None):
        # Not armed and nothing there yet: nothing to watch, nothing to record.
        if not self.armed and not os.path.isfile(os.path.join(self.root, ROOT_MARKER)):
            return None, None
        return super().tick(now)


def snapshot_gate(root: str = DATA_ROOT, state_dir: Path = V.GUARDIAN_STATE) -> tuple[bool, str]:
    """May a snapshot be taken? Not while tripped, not of a root that shrank."""
    w = DataWatch(root, state_dir)
    marker = w.tripped()
    if marker:
        return False, f"data tripwire is set ({marker.get('reason')})"
    if not os.path.isfile(os.path.join(root, ROOT_MARKER)):
        return False, f"{root} has no {ROOT_MARKER}"
    snap = V.measure(root, skip_top=CACHE_DIRS)
    if snap is None:
        return False, f"{root} does not exist"
    if w.history:
        why = V.evaluate(w.history, snap, window=float("inf"), what="data")
        if why:
            return False, f"the data root does not look like the last healthy measurement: {why}"
    return True, f"{snap.total} files"


#: What `snapshot-data.sh` names a snapshot (`date -u +%Y%m%dT%H%M%SZ`), and the
#: same shape `prune-data-snapshots.sh` matches with `-name '20*T*Z'`.
SNAPSHOT_STAMP_FMT = "%Y%m%dT%H%M%SZ"


def _stamp_epoch(name: str) -> float | None:
    """Seconds since the epoch for a snapshot stamp; None if it is not one."""
    try:
        return datetime.strptime(name, SNAPSHOT_STAMP_FMT).replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def newest_snapshot(snaps_dir: str) -> tuple[str, float] | None:
    """The newest snapshot under `snaps_dir`, as (stamp, epoch). None if it
    holds none, or cannot be read.

    Judged by the stamp in the name and not by mtime: a copied or restored
    directory has a fresh mtime over an old snapshot inside it, and the prune
    job decides what survives by that same stamp — so a name it would not match
    is not a snapshot here either, and cannot silence this check.
    """
    try:
        entries = list(os.scandir(snaps_dir))
    except OSError:
        return None
    newest: tuple[str, float] | None = None
    for e in entries:
        ts = _stamp_epoch(e.name)
        if ts is None:
            continue
        try:
            if not e.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        if newest is None or ts > newest[1]:
            newest = (e.name, ts)
    return newest


def age_text(seconds: float) -> str:
    """An age in the units a person reads an alert in: days and hours past a
    day, hours and minutes below that, minutes below an hour."""
    whole = max(0, int(seconds))
    days, rem = divmod(whole, 86400)
    hours, rem = divmod(rem, 3600)
    if days:
        return f"{days} d {hours} h"
    if hours:
        return f"{hours} h {rem // 60} m"
    return f"{rem // 60} m"


def snapshot_state(snaps_dir: str, max_age_seconds: float,
                   now: float | None = None) -> dict:
    """The measurement behind every sentence about snapshot age: the newest stamp
    the directory holds, how old it is, whether that is inside
    `max_age_seconds`, and one line saying so.

    `snapshot_report` and `datawatch.py snapshot-age --tsv` are both formatted
    from here, so the age an operator prints from a restore listing and the age
    the watchdog would have alerted on are one reading of the directory and not
    two reads that could disagree. What a stale answer *costs* is the caller's
    decision: the guardian alerts, the listing warns.
    """
    now = time.time() if now is None else now
    newest = newest_snapshot(snaps_dir)
    if newest is None:
        return {"stamp": None, "age": None, "fresh": False,
                "text": f"{snaps_dir}: no snapshot directory, or nothing in it is one"}
    stamp, ts = newest
    age = now - ts
    fresh = age <= max_age_seconds
    text = f"{snaps_dir}: newest snapshot {stamp} is {age_text(age)} old"
    if not fresh:
        text += f", past the {max_age_seconds / 3600:g} h limit"
    return {"stamp": stamp, "age": age, "fresh": fresh, "text": text}


def snapshot_report(snaps_dir: str, max_age_seconds: float,
                    now: float | None = None) -> tuple[bool, str]:
    """Is the newest snapshot fresh, and the line that says so, naming its stamp
    and its age. See `snapshot_state`."""
    state = snapshot_state(snaps_dir, max_age_seconds, now)
    return state["fresh"], state["text"]


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    w = DataWatch()
    if cmd == "status":
        snap = V.measure(w.root, skip_top=CACHE_DIRS)
        print(json.dumps({"armed": w.armed, "tripped": w.tripped(),
                          "last_good": w.history[-1].to_dict() if w.history else None,
                          "now": snap.to_dict() if snap else None,
                          "stray_in_tree": stray_in_tree()}, indent=2))
        return 0
    if cmd == "clear":
        existed = w.clear()
        print("data tripwire cleared; baseline reset to the data root as it is now" if existed
              else "data tripwire was not set; baseline refreshed")
        return 0
    if cmd == "snapshot-gate":
        ok, why = snapshot_gate()
        print(("ok: " if ok else "REFUSING: ") + why, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    if cmd == "snapshot-age":
        # Imported here, not at module scope: the threshold and the directory
        # are guardian tunables, and this module's import stays dependency-free
        # for everything the tick runs. `--tsv <dir>` is what restore-data.sh
        # reads, so the age it prints is the age that would have alerted.
        import policy
        rest = [a for a in argv[2:] if not a.startswith("-")]
        snaps = rest[0] if rest else str(policy.DATA_SNAPSHOTS)
        st = snapshot_state(snaps, policy.SNAPSHOT_MAX_AGE_SECONDS)
        if "--tsv" in argv[2:]:
            age = "-" if st["age"] is None else age_text(st["age"])
            print(f"{st['stamp'] or '-'}\t{age}\t{0 if st['fresh'] else 1}")
        else:
            print(("fresh: " if st["fresh"] else "STALE: ") + st["text"])
        return 0
    if cmd == "strays":
        found = stray_in_tree()
        for n in found:
            print(os.path.join(TREE, n))
        return 1 if found else 0
    if cmd == "inert":
        # Read-only: which of the current strays the hourly check would move.
        for n in inert_residue(TREE, stray_in_tree()):
            print(os.path.join(TREE, n))
        return 0
    print(f"usage: {argv[0]} status|clear|snapshot-gate|snapshot-age|strays|inert",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
