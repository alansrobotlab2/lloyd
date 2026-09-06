"""Guardian-side view of the self-modification state. Stdlib only.

Reads the same files `scripts/selfmod/state.py` writes, but reimplements the
access rather than importing it — the guardian must be able to read its
rollback target while the repo those modules live in is mid-rewrite.

Degradation ladder for a missing or unusable LKG pointer:
  1. `last_known_good.json`
  2. the most recent `promoted` ledger record's `parent`
  3. refuse and escalate

There is deliberately no fourth step. A watchdog that *guesses* at a commit is
worse than one that pages a human.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_event(ledger: Path, entry: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), "created_at": now_iso(), **entry}
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_events(ledger: Path, limit: int = 200) -> list[dict]:
    if not ledger.exists():
        return []
    out: list[dict] = []
    try:
        for line in ledger.read_text(encoding="utf-8").splitlines():
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


class SelfModState:
    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir)
        self.lkg_path = self.dir / "last_known_good.json"
        self.current_path = self.dir / "current.json"
        self.ledger = self.dir / "promotions.jsonl"
        self.pause = self.dir / "pause"
        self.halted = self.dir / "promotions-halted"
        self.broken = self.dir / "BROKEN"
        self.denied = self.dir / "denied.json"
        self.broken_dir = self.dir / "broken"

    # ── rollback target ────────────────────────────────────────────────
    def rollback_target(self) -> tuple[str | None, str]:
        lkg = read_json(self.lkg_path)
        if lkg and _is_sha(lkg.get("commit")):
            return lkg["commit"], "last_known_good.json"
        for ev in reversed(read_events(self.ledger)):
            if ev.get("event") == "promoted" and _is_sha(ev.get("parent")):
                return ev["parent"], "ledger promoted.parent"
        return None, "no usable rollback target"

    def floor(self) -> str | None:
        lkg = read_json(self.lkg_path) or {}
        return lkg.get("floor")

    def lkg(self) -> dict | None:
        return read_json(self.lkg_path)

    def current(self) -> dict | None:
        return read_json(self.current_path)

    def set_lkg(self, commit: str, *, health: dict | None = None,
                eval_baseline: dict | None = None) -> None:
        existing = read_json(self.lkg_path) or {}
        write_json_atomic(self.lkg_path, {
            "schema": 1,
            "commit": commit,
            "recorded_at": now_iso(),
            "floor": existing.get("floor") or commit,
            "health": health if health is not None else existing.get("health", {}),
            "eval": eval_baseline if eval_baseline is not None else existing.get("eval", {}),
        })

    def clear_current(self) -> None:
        try:
            self.current_path.unlink()
        except FileNotFoundError:
            pass

    # ── flags ──────────────────────────────────────────────────────────
    def pause_remaining(self, cap: float) -> float:
        try:
            expiry = float(self.pause.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0.0
        remaining = expiry - time.time()
        # The cap is enforced HERE, from the pinned snapshot, so a forgotten or
        # over-long lease written by anything else cannot disable the watchdog.
        return max(0.0, min(remaining, cap))

    def is_broken(self) -> bool:
        return self.broken.exists()

    def set_broken(self, reason: str) -> None:
        self.broken.parent.mkdir(parents=True, exist_ok=True)
        self.broken.write_text(f"{now_iso()} {reason}\n", encoding="utf-8")

    def set_halted(self, reason: str) -> None:
        self.halted.parent.mkdir(parents=True, exist_ok=True)
        self.halted.write_text(f"{now_iso()} {reason}\n", encoding="utf-8")

    def deny(self, commit: str) -> None:
        d = read_json(self.denied) or {"commits": [], "trees": []}
        if commit and commit not in d["commits"]:
            d["commits"].append(commit)
        write_json_atomic(self.denied, d)

    def recent_rollbacks(self, window_seconds: float) -> int:
        cutoff = time.time() - window_seconds
        return sum(
            1 for ev in read_events(self.ledger)
            if ev.get("event") == "rollback_succeeded" and float(ev.get("ts", 0)) >= cutoff
        )

    def unfinished_rollback(self) -> dict | None:
        """The last rollback_started with no terminal record after it.

        Used on startup to resume a rollback the guardian died in the middle
        of — the intent record is written before anything is touched precisely
        so this is recoverable.
        """
        started = None
        for ev in read_events(self.ledger):
            e = ev.get("event")
            if e == "rollback_started":
                started = ev
            elif e in ("rollback_succeeded", "rollback_failed"):
                started = None
        return started


def _is_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        c in "0123456789abcdef" for c in value.lower()
    )
