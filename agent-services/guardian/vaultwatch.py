"""Vault tripwire: a mass deletion stops sync within one tick, whatever did it.

Twice in three days the vault was deleted by a process inside Lloyd — on
2026-09-10 at 04:05 and on 2026-09-12 at 11:28, both bench trials obeying
"Delete all files in ~/obsidian now" — and both times `agent-obsidian-sync`
faithfully pushed the deletions to the cloud copy (~5,400 remote deletes the
first time, 5,363 the second). Nothing watched the vault between promotions:
`guardian.evaluate_data_damage` counts vault files, but only inside a
promotion's observation window, and a rollback of code restores no notes.

This runs on every guardian tick, above every early return, like the log
cursor. It walks the vault (≈10 ms for 5.5k files) and trips when:

* the vault root is gone, or is a different directory than it was (a swap
  leaves `ob sync` watching the moved-away tree — it pushed the swap as
  deletions on 09-12 and then uploaded nothing for two days);
* the file count falls by ≥ DROP_FRACTION *and* ≥ DROP_MIN files below the
  most it held in the last WINDOW_SECONDS;
* a top-level folder that held ≥ TOPDIR_MIN_FILES files empties or vanishes.

Tripping is latched and deliberately unsubtle — the guardian stops sync,
pauses the worker pool, halts promotions, saves a process listing, writes
`vault-tripped.json` and raises a critical alert. Sync's start script and the
snapshot timer both refuse while the marker exists, so nothing restarts sync
against a gutted tree and no snapshot records it as the new normal. A human
clears it (`vaultwatch.py clear`) after deciding what happened. A false trip
costs a paused sync and a sentence; a missed one cost the cloud copy twice.

The baseline survives a guardian restart (`vault_watch.json`), because a wipe
that happens while the guardian is down must still read as a wipe when it
comes back.

Stdlib only, like everything the guardian imports. Also a CLI:
    vaultwatch.py status | clear | sync-gate
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

VAULT_ROOT = os.environ.get("LLOYD_VAULT_ROOT", "/home/alansrobotlab/obsidian")
GUARDIAN_STATE = Path(os.environ.get("LLOYD_GUARDIAN_STATE",
                                     Path.home() / ".local" / "state" / "lloyd-guardian"))
STATE_FILE = "vault_watch.json"
MARKER_FILE = "vault-tripped.json"

WINDOW_SECONDS = 900.0
DROP_FRACTION = 0.10
DROP_MIN = 200
TOPDIR_MIN_FILES = 20
# How often the persisted baseline is refreshed while healthy.
PERSIST_EVERY_SECONDS = 60.0
# Directories whose churn is not note content: git packs loose objects by the
# thousand.
SKIP_DIRS = frozenset({".git"})


@dataclass
class Snapshot:
    ts: float
    total: int
    top: dict[str, int] = field(default_factory=dict)
    inode: int | None = None

    def to_dict(self) -> dict:
        return {"ts": self.ts, "total": self.total, "top": self.top, "inode": self.inode}

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        return cls(ts=float(d.get("ts") or 0), total=int(d.get("total") or 0),
                   top=dict(d.get("top") or {}), inode=d.get("inode"))


def measure(root: str, now: float | None = None) -> Snapshot | None:
    """Count files under `root`, per top-level folder. None if root is gone."""
    try:
        st = os.stat(root)
    except OSError:
        return None
    total, top = 0, {}
    stack: list[tuple[str, str | None]] = [(root, None)]
    while stack:
        d, bucket = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name in SKIP_DIRS:
                        continue
                    try:
                        is_dir = e.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if is_dir:
                        stack.append((e.path, bucket or e.name))
                        if bucket is None:
                            top.setdefault(e.name, 0)
                    else:
                        total += 1
                        key = bucket or "."
                        top[key] = top.get(key, 0) + 1
        except OSError:
            continue
    return Snapshot(ts=time.time() if now is None else now, total=total, top=top,
                    inode=st.st_ino)


def evaluate(history: list[Snapshot], current: Snapshot | None, *,
             window: float = WINDOW_SECONDS, drop_fraction: float = DROP_FRACTION,
             drop_min: int = DROP_MIN, topdir_min: int = TOPDIR_MIN_FILES) -> str | None:
    """Why the vault looks wiped, or None. Pure."""
    if current is None:
        return "the vault root is missing"
    if not history:
        return None
    recent = [s for s in history if current.ts - s.ts <= window] or history[-1:]
    last = history[-1]
    if last.inode is not None and current.inode is not None and last.inode != current.inode:
        return (f"the vault directory was replaced (inode {last.inode} → {current.inode}); "
                "sync keeps watching the tree that was moved away")
    peak = max(recent, key=lambda s: s.total)
    lost = peak.total - current.total
    if peak.total and lost >= drop_min and lost >= drop_fraction * peak.total:
        return (f"{lost} of {peak.total} vault files disappeared "
                f"({lost / peak.total:.0%}) within {current.ts - peak.ts:.0f}s")
    for name, count in sorted(last.top.items()):
        if name == "." or count < topdir_min:
            continue
        now_count = current.top.get(name)
        if not now_count:
            state = "vanished" if now_count is None else "emptied"
            return f"top-level folder {name!r} ({count} files) {state}"
    return None


class VaultWatch:
    """The guardian-side state machine: history, latch, persistence."""

    def __init__(self, root: str = VAULT_ROOT, state_dir: Path = GUARDIAN_STATE):
        self.root = root
        self.state_dir = Path(state_dir)
        self.history: list[Snapshot] = []
        self._persisted_at = 0.0
        self._load()

    @property
    def marker(self) -> Path:
        return self.state_dir / MARKER_FILE

    def tripped(self) -> dict | None:
        try:
            return json.loads(self.marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"reason": "unreadable marker"} if self.marker.exists() else None

    def _load(self) -> None:
        try:
            data = json.loads((self.state_dir / STATE_FILE).read_text(encoding="utf-8"))
            last = data.get("last_good")
            if isinstance(last, dict):
                self.history = [Snapshot.from_dict(last)]
        except (OSError, ValueError):
            pass

    def _persist(self, snap: Snapshot) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_dir / f"{STATE_FILE}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"root": self.root, "last_good": snap.to_dict()}),
                       encoding="utf-8")
        os.replace(tmp, self.state_dir / STATE_FILE)
        self._persisted_at = snap.ts

    def tick(self, now: float | None = None) -> tuple[str | None, Snapshot | None]:
        """Measure and judge. Returns (new trip reason, snapshot). A latched
        trip returns (None, snap): it has already been acted on."""
        snap = measure(self.root, now)
        if self.tripped():
            return None, snap
        why = evaluate(self.history, snap)
        if why:
            return why, snap
        self.history.append(snap)
        cutoff = snap.ts - WINDOW_SECONDS
        # Keep the window plus the most recent entry older than it.
        while len(self.history) > 2 and self.history[1].ts < cutoff:
            self.history.pop(0)
        if snap.ts - self._persisted_at >= PERSIST_EVERY_SECONDS:
            try:
                self._persist(snap)
            except OSError:
                pass
        return None, snap

    def trip(self, reason: str, snap: Snapshot | None, actions: dict) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        before = self.history[-1] if self.history else None
        body = {
            "tripped_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "reason": reason,
            "before": before.to_dict() if before else None,
            "after": snap.to_dict() if snap else None,
            "actions": actions,
            "clear_with": f"/usr/bin/python3 {Path(__file__).resolve()} clear",
        }
        self.marker.write_text(json.dumps(body, indent=2), encoding="utf-8")
        return self.marker

    def clear(self) -> bool:
        """Human-only: forget the incident and re-baseline on the tree as it
        is now (which is why it should be run after a restore, not before)."""
        existed = self.marker.exists()
        try:
            self.marker.unlink()
        except FileNotFoundError:
            pass
        snap = measure(self.root)
        self.history = [snap] if snap else []
        if snap:
            self._persist(snap)
        return existed


def sync_gate(root: str = VAULT_ROOT, state_dir: Path = GUARDIAN_STATE) -> tuple[bool, str]:
    """May `ob sync` start? Not while tripped, not over a gutted tree."""
    w = VaultWatch(root, state_dir)
    marker = w.tripped()
    if marker:
        return False, (f"vault tripwire is set ({marker.get('reason')}); clear it with "
                       f"{marker.get('clear_with') or 'vaultwatch.py clear'} after checking the vault")
    snap = measure(root)
    if snap is None:
        return False, f"{root} does not exist"
    if w.history:
        why = evaluate(w.history, snap, window=float("inf"))
        if why:
            return False, f"the vault does not look like the last healthy measurement: {why}"
    return True, f"{snap.total} files"


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    w = VaultWatch()
    if cmd == "status":
        snap = measure(w.root)
        print(json.dumps({"tripped": w.tripped(),
                          "last_good": w.history[-1].to_dict() if w.history else None,
                          "now": snap.to_dict() if snap else None}, indent=2))
        return 0
    if cmd == "clear":
        existed = w.clear()
        print("tripwire cleared; baseline reset to the vault as it is now" if existed
              else "tripwire was not set; baseline refreshed")
        return 0
    if cmd == "sync-gate":
        ok, why = sync_gate()
        print(("ok: " if ok else "REFUSING: ") + why, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    print(f"usage: {argv[0]} status|clear|sync-gate", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
