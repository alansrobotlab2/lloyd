"""On-disk state for the self-modification loop.

Everything lives under ``~/.local/state/lloyd-selfmod/`` — deliberately
**outside the repo**, because the guardian must read it while the repo is
being rewritten. `_pipeline/` would not do: it is gitignored but still inside
the tree, so a `git clean -fdx` would take it.

Two files carry the contract, and they have different jobs:

  * ``last_known_good.json`` — one small object, read under duress by a
    stdlib-only watchdog with the backend dead. Must parse in one read.
  * ``promotions.jsonl`` — the append-only audit trail.

**LKG is advanced only by the guardian**, after a promotion survives its full
observation window. The promoter writes `current.json` and never touches
`last_known_good.json`. That invariant is what makes "last known good" mean
*observed healthy in production* rather than *passed a pre-flight*, and it
guarantees a rollback always targets a commit that already ran clean.

Why this module does not reuse ``scripts.autoresearch.common.ledger_append``:
that function is documented "best-effort — never raises", with no fsync and no
locking. Defensible for a research ledger; wrong for the audit record of what
code is running in production, where a silently dropped line means you cannot
reconstruct what landed. `append_event` here fsyncs and raises.
``tests/test_selfmod_state.py`` asserts the two behave differently so nobody
later refactors them together.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_DIR = Path(
    os.environ.get("LLOYD_SELFMOD_STATE", Path.home() / ".local" / "state" / "lloyd-selfmod")
)

LKG_PATH = STATE_DIR / "last_known_good.json"
CURRENT_PATH = STATE_DIR / "current.json"
LEDGER_PATH = STATE_DIR / "promotions.jsonl"
LOCK_PATH = STATE_DIR / "lock"
PAUSE_PATH = STATE_DIR / "pause"
HALTED_PATH = STATE_DIR / "promotions-halted"
BROKEN_PATH = STATE_DIR / "BROKEN"
DENIED_PATH = STATE_DIR / "denied.json"
BROKEN_DIR = STATE_DIR / "broken"
ROUNDS_DIR = STATE_DIR / "rounds"

SCHEMA = 1


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    for d in (STATE_DIR, BROKEN_DIR, ROUNDS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Atomic JSON
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    """Write `payload` atomically, fsyncing both the file and its directory.

    The directory fsync matters: without it a crash can leave the rename
    unrecorded and the old file in place, which for `last_known_good.json`
    means rolling back to the wrong commit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=False)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_verified(path: Path, payload: dict) -> dict:
    """Write `payload` and read it straight back, raising if it did not land.

    This is the direct fix for the defect class documented at
    `tests/test_autoresearch_promotion.py:362`: `snapshot_current_prompts`
    mkdirs unconditionally, never verifies the copy landed, and `promote()`
    overwrites live state anyway — 26 of 83 ledger promotions have no matching
    snapshot and therefore no rollback point. Nothing in this package mutates
    the live tree until its rollback point has been read back from disk.
    """
    write_json(path, payload)
    back = read_json(path)
    if back is None:
        raise RuntimeError(f"rollback point did not land: {path} is unreadable after write")
    for key in ("commit", "rollback_target", "round_id"):
        if key in payload and back.get(key) != payload[key]:
            raise RuntimeError(
                f"rollback point did not round-trip: {path} {key}="
                f"{back.get(key)!r} != {payload[key]!r}"
            )
    return back


# ---------------------------------------------------------------------------
# Ledger — fsyncs, and RAISES (unlike autoresearch's best-effort append)
# ---------------------------------------------------------------------------

def append_event(entry: dict, path: Path | None = None) -> None:
    """Append one JSON line to the promotions ledger. Raises on failure."""
    target = path or LEDGER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), "created_at": now_iso(), **entry}
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with open(target, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_events(path: Path | None = None, limit: int = 100) -> list[dict]:
    target = path or LEDGER_PATH
    if not target.exists():
        return []
    out: list[dict] = []
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        return []
    return out[-limit:]


# ---------------------------------------------------------------------------
# Last known good
# ---------------------------------------------------------------------------

def read_lkg() -> dict | None:
    return read_json(LKG_PATH)


def write_lkg(commit: str, *, floor: str | None = None, health: dict | None = None,
              eval_baseline: dict | None = None) -> dict:
    existing = read_lkg() or {}
    payload = {
        "schema": SCHEMA,
        "commit": commit,
        "recorded_at": now_iso(),
        # The floor is set once, at install, and never moves: no rollback may
        # land on a tree that predates the guardian's own existence.
        "floor": floor or existing.get("floor") or commit,
        "health": health if health is not None else existing.get("health", {}),
        "eval": eval_baseline if eval_baseline is not None else existing.get("eval", {}),
    }
    return write_verified(LKG_PATH, payload)


def read_current() -> dict | None:
    return read_json(CURRENT_PATH)


def clear_current() -> None:
    try:
        CURRENT_PATH.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def is_halted() -> bool:
    return HALTED_PATH.exists()


def set_halted(reason: str) -> None:
    HALTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    HALTED_PATH.write_text(f"{now_iso()} {reason}\n", encoding="utf-8")


def clear_halted() -> None:
    try:
        HALTED_PATH.unlink()
    except FileNotFoundError:
        pass


def is_broken() -> bool:
    return BROKEN_PATH.exists()


def pause_remaining() -> float:
    """Seconds left on the maintenance lease, 0 if none/expired."""
    try:
        expiry = float(PAUSE_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0.0
    return max(0.0, expiry - time.time())


def set_pause(seconds: float, cap: float = 1800.0) -> float:
    """Take a maintenance lease so the guardian observes but does not act.

    Capped so a forgotten lease cannot disable the watchdog indefinitely. The
    cap the *guardian* enforces lives in its pinned snapshot, not here — this
    one is only a courtesy to callers.
    """
    seconds = max(0.0, min(float(seconds), cap))
    PAUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    expiry = time.time() + seconds
    PAUSE_PATH.write_text(str(expiry), encoding="utf-8")
    return expiry


def clear_pause() -> None:
    try:
        PAUSE_PATH.unlink()
    except FileNotFoundError:
        pass


def read_denied() -> dict:
    return read_json(DENIED_PATH) or {"commits": [], "trees": []}


def deny(commit: str, tree_hash: str | None = None) -> None:
    """Record a reverted commit so the same change is not re-landed.

    Keyed by SHA *and* by the tree-hash of its changed paths, so a change
    re-derived under a new SHA is caught too. This is the anti-ping-pong
    mechanism.
    """
    d = read_denied()
    if commit and commit not in d["commits"]:
        d["commits"].append(commit)
    if tree_hash and tree_hash not in d["trees"]:
        d["trees"].append(tree_hash)
    write_json(DENIED_PATH, d)


def is_denied(commit: str | None = None, tree_hash: str | None = None) -> bool:
    d = read_denied()
    return bool((commit and commit in d["commits"]) or (tree_hash and tree_hash in d["trees"]))


# ---------------------------------------------------------------------------
# Lock — one round / promotion / rollback at a time
# ---------------------------------------------------------------------------

class LockHeld(RuntimeError):
    """Another self-modification operation holds the lock."""


class Lock:
    """flock-based mutex whose holder is identifiable and stealable when dead.

    `flock` releases automatically if the holder dies, so staleness is handled
    by the kernel; the JSON payload exists so a human (or an alert) can see
    *who* holds it.
    """

    def __init__(self, path: Path | None = None, owner: str = ""):
        self.path = path or LOCK_PATH
        self.owner = owner
        self._fd: int | None = None

    def acquire(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = ""
            try:
                holder = os.read(fd, 4096).decode("utf-8", "replace").strip()
            except OSError:
                pass
            os.close(fd)
            raise LockHeld(f"self-modification lock held: {holder or 'unknown holder'}")
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(
            {"pid": os.getpid(), "owner": self.owner, "since": now_iso()}
        ).encode())
        os.fsync(fd)
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "Lock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
