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

Stdlib only. CLI:  datawatch.py status | clear | snapshot-gate | snapshot-age | strays
"""

from __future__ import annotations

import json
import os
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
    "graphify-out",    # the code graph's per-tree cache (`agent_mcp/code_graph.py`)
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


class DataWatch(V.VaultWatch):
    what = "data"
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
    snap = V.measure(root)
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
        snap = V.measure(w.root)
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
    print(f"usage: {argv[0]} status|clear|snapshot-gate|snapshot-age|strays", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
