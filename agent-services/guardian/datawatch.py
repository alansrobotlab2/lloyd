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

import json
import os
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import policy as P
import vaultwatch as V

DATA_ROOT = os.environ.get("LLOYD_DATA", "/home/alansrobotlab/lloyd-data")
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


def quarantine_inert(tree: str, names, data_root: str = DATA_ROOT,
                     now: float | None = None) -> list[tuple[str, str]]:
    """Move each `inert_residue` file out of the tree. Returns `(name, destination)`
    for the ones that moved; a file that could not be moved is simply not in the list
    and goes on to alert.

    The move is an empty file created at the destination and the source unlinked,
    in that order, because the data root is its own subvolume and `rename` across it
    is EXDEV. The source is re-read immediately before the unlink and must be the
    same inode, still empty — a writer that arrived between the two reads keeps its
    file. One JSONL line per move records what the file was, so the quarantine
    directory is its own audit and not just a pile of empty files."""
    now = time.time() if now is None else now
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


def data_root_shadows(data_root: str = DATA_ROOT, repo: str = TREE,
                      now: float | None = None) -> list[dict]:
    """Every 0-byte database in the data root whose name is populated elsewhere under
    the same root, as `(shadow, trusted)` pairs. READ-ONLY: never creates, moves,
    truncates or unlinks anything.

    The shape, from the live root: `kg.sqlite` at 0 bytes at the top while the knowledge
    graph the code opens — `app.paths.VAULT_KG_DB`, `_pipeline/vault-derived/kg.sqlite` —
    holds 160,882,688 bytes. A 0-byte file is a valid *empty* SQLite database, so a probe
    that opened the bare name from the wrong cwd does not get an error; it gets 0 rows,
    and prints "the KG is empty" about a populated store. That is the 2026-08-23/24
    incident class this item names, and the same origin as the 2026-10-02 in-tree
    `workers.db` story: `sqlite3` creates the file on connect. It happened twice in
    opposite directions (`_pipeline/workers.db` at 0 bytes beside the populated queue
    store at the top), which is why the check runs both ways.

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

    Both halves of a pair must be measurements, and the exclusions are the same ones
    `inert_residue` earns its right to act on: a regular file, one name, zero bytes,
    untouched for `INERT_MIN_AGE_SECONDS`, no SQLite sidecar beside it. `CACHE_DIRS` is
    not scanned (a rebuild in progress is not a shadow), and the scan does not follow a
    symlink or descend past `SHADOW_MAX_DEPTH`.

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
        empties = sorted((p, d) for d, sz, p in recs if sz == 0)
        populated = sorted(((p, d, sz) for d, sz, p in recs if sz > 0),
                           key=lambda r: (r[1], -r[2], r[0]))
        if not empties or not populated:
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
        trusted_path, trusted_depth, trusted_size = next(
            (r for r in populated
             if declared_path and os.path.normpath(r[0]) == declared_path),
            populated[0])
        trusted_norm = os.path.normpath(trusted_path)
        for shadow, depth in empties:
            try:
                st = os.lstat(shadow)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_size != 0 or st.st_nlink != 1:
                continue
            if now - st.st_mtime < INERT_MIN_AGE_SECONDS:
                continue
            if any(os.path.lexists(shadow + suffix) for suffix in _SQLITE_SIDECARS):
                continue
            out.append({
                "shadow": shadow, "name": name, "depth": depth,
                "trusted": trusted_path, "trusted_depth": trusted_depth,
                "trusted_size": trusted_size,
                "declared": declared_path is not None
                and trusted_norm == declared_path,
                "declared_relative": rel,
            })
    out.sort(key=lambda f: f["shadow"])
    if truncated:
        # The bound was hit, so an absence after here is a scan that stopped, not a
        # root that is clean. Loud rather than silent, and never a claim of all-clear.
        raise OSError(f"data-root shadow scan stopped after {seen} entries under {root} "
                      f"(SHADOW_MAX_ENTRIES); its silence would be a partial measure")
    return out


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
        """A top-level folder this root measured as a git checkout is scratch, not store.

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
        """
        if name in RUNTIME_NAMES:
            return None
        if name not in before.checkouts:
            return None
        return (f"top-level folder {name!r} ({count} files) went missing; not a trip — "
                f"the last measurement saw a {V.CHECKOUT_ENTRY} entry inside it, so it "
                f"was a git checkout and not one of this root's stores (#3095)")

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
