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


def stray_in_tree(tree: str = TREE) -> list[str]:
    """Runtime data that has come back inside the code tree. Empty is healthy.

    The candidate set is the tree, not a list: `KNOWN_GOOD_TOPLEVEL` subtracted
    from its top-level entries, plus the retained `RUNTIME_NAMES` whose nested
    paths the top-level scan cannot reach. Two rules then decide a candidate:

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
    retained = [n for n in RUNTIME_NAMES if os.path.lexists(os.path.join(tree, n))]
    present = retained + [n for n in _top_level(tree) if n not in retained]
    if not present:
        return []
    tracked = _ls_files(tree)
    if tracked is None:
        return retained

    def under(paths: set[str], name: str) -> bool:
        return any(p == name or p.startswith(name + "/") for p in paths)

    untracked = _ls_files(tree, tuple(retained), "--others") if retained else set()
    if untracked is None:
        return retained
    kept = set(retained)
    return sorted(n for n in present
                  if not under(tracked, n)
                  or (n in kept and under(untracked, n)))


#: An empty file younger than this may be a store its writer has only just opened
#: (`sqlite3` creates the file on connect and writes the header on first use), so it
#: is left for the next check rather than moved from under a live handle.
INERT_MIN_AGE_SECONDS = 600.0
#: Where inert residue goes, under the data root. A move, so the hourly snapshots
#: carry it and nothing is ever deleted on this check's say-so.
QUARANTINE_SUBDIR = os.path.join("quarantine", "tree-strays")
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
    qdir = os.path.join(data_root, QUARANTINE_SUBDIR)
    for name in inert_residue(tree, names, data_root, now):
        src = os.path.join(tree, name)
        try:
            before = os.lstat(src)
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

    def judge(self, snap: V.Snapshot | None) -> str | None:
        if not self.armed:
            return None
        if snap is not None and not os.path.isfile(os.path.join(self.root, ROOT_MARKER)):
            return f"{self.root} lost its {ROOT_MARKER} marker"
        return super().judge(snap)

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
