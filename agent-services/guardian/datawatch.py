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
RUNTIME_NAMES = ("sessions", "event_logs", "logs", "autonomy-runs", "_pipeline",
                 "usage.db", "workers.db", "research.db", "mc-state.json",
                 "data/tool_overrides.yaml", "voice_profiles", "eval/baselines",
                 "agent-services/logs", "None")


def _ls_files(tree: str, names: list[str], *flags: str) -> set[str] | None:
    """The paths `git ls-files` reports under `names`, or None when git cannot
    answer (not a checkout, no git, a hung index lock)."""
    try:
        out = subprocess.run(["git", "-C", tree, "ls-files", "-z", *flags, "--", *names],
                             capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return {p for p in out.stdout.decode("utf-8", "replace").split("\0") if p}


def stray_in_tree(tree: str = TREE) -> list[str]:
    """Runtime names present inside the code tree. Empty is healthy.

    A name git tracks is committed on purpose, not written by a stray writer:
    `eval/baselines/` holds measurement records the harness reviews commit as
    evidence (#600, P4-P9), and it alerted every hour for them. Such a name is a
    stray only when it also holds something untracked (ignored files included —
    the gitignore hiding a writer is the case this check exists for). When git
    cannot answer, presence alone decides, as before."""
    present = [n for n in RUNTIME_NAMES if os.path.lexists(os.path.join(tree, n))]
    if not present:
        return []
    tracked = _ls_files(tree, present)
    untracked = _ls_files(tree, present, "--others")
    if tracked is None or untracked is None:
        return present

    def under(paths: set[str], name: str) -> bool:
        return any(p == name or p.startswith(name + "/") for p in paths)

    return [n for n in present if not under(tracked, n) or under(untracked, n)]


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
