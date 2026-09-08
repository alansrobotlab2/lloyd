"""What this turn changed on disk, and how to put it back.

Ordinary chat turns in `~/lloyd` are deploys — a saved file is live. The
self-modification loop has a worktree, a gate and an automatic rollback; a
chat turn that edits three files has none of that, no line saying which three,
and no undo. This is the missing half: one record per (session, turn) naming
every file written, with the pre-image beside it.

Shape
-----
    sessions/<sid>.changes/<turn_id>/index.json
    sessions/<sid>.changes/<turn_id>/<sha1(realpath)>.pre

The layout follows `app/harness/tool_result_spill.py` — a sibling directory
per session — because both are per-turn side data that must survive an
aggregator restart and be findable from a session id alone.

Rules that are not obvious
--------------------------
* **First writer per realpath per turn wins.** The pre-image is what the file
  looked like when *this turn* started touching it, not before the last of
  five edits. A turn that edits one file ten times reverts to where it came
  in. `begin` loads the index from disk on a miss, so an aggregator restart
  mid-turn does not restart the rule.
* **Revert refuses a file that moved since.** If the current sha does not
  match what this turn wrote, something else has written since and restoring
  the pre-image would destroy that instead. Refused, named, and reported —
  never silently skipped.
* **Keys are realpath**, so a symlink and its target are one entry, matching
  the read-tracking gate next door.
* **A create reverts by unlinking.** Documented imperfection: a create made
  *through* a dangling symlink unlinks the symlink path rather than the
  target it pointed at.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path

from app.atomic_io import atomic_write_text
from app.paths import SESSIONS_DIR

logger = logging.getLogger("lloyd-change-ledger")

# Where the per-turn directories live. Module-level so tests can point it at
# tmp_path without touching app.paths.
CHANGES_ROOT: Path = SESSIONS_DIR

# Above this, the pre-image is not kept and revert refuses with "no snapshot".
# A model does not hand-edit an 8 MiB file; something generated it.
SNAPSHOT_MAX_BYTES = 8 * 1024 * 1024

# In-memory mirror, bounded. The index on disk is the durable copy.
_TURNS_MAX = 512

_DEFAULTS = {"enabled": True, "retention_days": 7, "max_total_mb": 256}


def config() -> dict:
    try:
        from app.config import CONFIG
        raw = (CONFIG.get("harness") or {}).get("change_ledger") or {}
    except Exception:
        raw = {}
    out = dict(_DEFAULTS)
    if isinstance(raw, dict):
        out.update({k: v for k, v in raw.items() if v is not None})
    return out


def enabled() -> bool:
    return bool(config()["enabled"])


@dataclass
class Entry:
    path: str                       # as the model gave it
    real: str                       # os.path.realpath
    op: str                         # edit | write | create
    call_id: str = ""
    via_session: str = ""           # set when a subagent made the change
    ts: float = 0.0
    pre_sha256: str = ""
    post_sha256: str = ""
    writes: int = 0
    snapshot: str = "none"          # ok | too_large | failed | none
    reverted_at: float | None = None


@dataclass
class TurnLedger:
    session_id: str
    turn_id: str
    entries: "OrderedDict[str, Entry]" = field(default_factory=OrderedDict)


_turns: "OrderedDict[tuple[str, str], TurnLedger]" = OrderedDict()
_lock = threading.Lock()
_last_prune = 0.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def turn_dir(session_id: str, turn_id: str) -> Path:
    return CHANGES_ROOT / f"{session_id}.changes" / turn_id


def _index_path(session_id: str, turn_id: str) -> Path:
    return turn_dir(session_id, turn_id) / "index.json"


def _pre_path(session_id: str, turn_id: str, real: str) -> Path:
    digest = hashlib.sha1(real.encode("utf-8", "replace")).hexdigest()
    return turn_dir(session_id, turn_id) / f"{digest}.pre"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    try:
        with open(path, "rb") as f:
            h = hashlib.sha256()
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def scope(session_id: str, turn_id: str) -> tuple[str, str] | None:
    """The (session, turn) a change should be recorded against.

    A subagent runs under a `task:*` session with no turn of its own, and
    nothing ever reads that session's ledger — the chat that spawned it is
    where a human would look for "what did this turn change". So a `task:*`
    session is redirected to the parent scope; everything else is itself.

    Returns None when there is nothing to attribute to, which is the normal
    case for a worker turn or a direct `run_query` caller: no turn id, no
    ledger. The feature is for turns a human is watching.
    """
    if session_id.startswith("task:"):
        try:
            from agent_mcp import _subagent_registry
            parent = _subagent_registry.parent_scope(session_id)
        except Exception:
            parent = None
        if not parent or not parent[0] or not parent[1]:
            return None
        return parent
    if not session_id or not turn_id:
        return None
    return session_id, turn_id


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _load(session_id: str, turn_id: str) -> TurnLedger:
    """In-memory ledger for a turn, loading the on-disk index on a miss."""
    key = (session_id, turn_id)
    hit = _turns.get(key)
    if hit is not None:
        _turns.move_to_end(key)
        return hit
    ledger = TurnLedger(session_id=session_id, turn_id=turn_id)
    try:
        raw = json.loads(_index_path(session_id, turn_id).read_text())
        for row in raw.get("entries") or []:
            entry = Entry(**{k: v for k, v in row.items()
                             if k in Entry.__dataclass_fields__})
            ledger.entries[entry.real] = entry
    except Exception:
        pass
    _turns[key] = ledger
    _turns.move_to_end(key)
    while len(_turns) > _TURNS_MAX:
        _turns.popitem(last=False)
    return ledger


def _write_index(ledger: TurnLedger) -> None:
    path = _index_path(ledger.session_id, ledger.turn_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps({
            "session_id": ledger.session_id,
            "turn_id": ledger.turn_id,
            "entries": [asdict(e) for e in ledger.entries.values()],
        }, indent=1))
    except Exception:
        logger.warning("change ledger: could not write %s", path, exc_info=True)


def begin(scope_pair: tuple[str, str], *, real: str, path: str, op: str,
          call_id: str = "", via_session: str = "") -> Entry | None:
    """Open (or find) this turn's entry for `real`. First writer wins."""
    session_id, turn_id = scope_pair
    with _lock:
        ledger = _load(session_id, turn_id)
        existing = ledger.entries.get(real)
        if existing is not None:
            existing.writes += 1
            return existing
        entry = Entry(path=path, real=real, op=op, call_id=call_id,
                      via_session=via_session, ts=time.time(), writes=1)
        ledger.entries[real] = entry
        return entry


def snapshot_pre(scope_pair: tuple[str, str], entry: Entry,
                 pre_bytes: bytes | None) -> None:
    """Write the pre-image once, on the first write of this turn."""
    if entry.writes != 1:
        return                      # already snapshotted by the first writer
    if entry.op == "create" or pre_bytes is None:
        entry.snapshot = "none"
        return
    if len(pre_bytes) > SNAPSHOT_MAX_BYTES:
        entry.snapshot = "too_large"
        entry.pre_sha256 = sha256_bytes(pre_bytes)
        return
    dest = _pre_path(*scope_pair, entry.real)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pre_bytes)
        os.replace(tmp, dest)
        entry.snapshot = "ok"
        entry.pre_sha256 = sha256_bytes(pre_bytes)
    except Exception:
        logger.warning("change ledger: snapshot failed for %s", entry.real,
                       exc_info=True)
        entry.snapshot = "failed"


def commit(scope_pair: tuple[str, str], entry: Entry, post_sha: str) -> None:
    """Record the post-image hash and flush the index."""
    session_id, turn_id = scope_pair
    with _lock:
        entry.post_sha256 = post_sha
        # A later write in the same turn supersedes the earlier post-image,
        # which is right: revert compares against what is on disk NOW.
        ledger = _load(session_id, turn_id)
        _write_index(ledger)
    _maybe_prune()


# ---------------------------------------------------------------------------
# Reading and reverting
# ---------------------------------------------------------------------------


def list_changes(session_id: str, turn_id: str) -> list[dict]:
    with _lock:
        ledger = _load(session_id, turn_id)
        return [asdict(e) for e in ledger.entries.values()]


def revert(session_id: str, turn_id: str,
           paths: list[str] | None = None) -> list[dict]:
    """Put back what this turn wrote. Per file, and never silently."""
    wanted = {os.path.realpath(p) for p in paths} if paths else None
    results: list[dict] = []
    with _lock:
        ledger = _load(session_id, turn_id)
        for entry in ledger.entries.values():
            if wanted is not None and entry.real not in wanted \
                    and entry.path not in (paths or []):
                continue
            results.append(_revert_one(session_id, turn_id, entry))
        if results:
            _write_index(ledger)
    return results


def _revert_one(session_id: str, turn_id: str, entry: Entry) -> dict:
    def out(status: str, reason: str = "") -> dict:
        return {"path": entry.path, "real": entry.real, "op": entry.op,
                "status": status, "reason": reason}

    if entry.reverted_at is not None:
        return out("skipped", "already reverted")

    if entry.op == "create":
        if not os.path.exists(entry.real):
            entry.reverted_at = time.time()
            return out("skipped", "already gone")
        current = sha256_file(entry.real)
        if entry.post_sha256 and current != entry.post_sha256:
            return out("refused", "file changed since this turn wrote it")
        try:
            os.unlink(entry.real)
        except OSError as exc:
            return out("refused", f"could not remove: {exc}")
        entry.reverted_at = time.time()
        return out("deleted")

    if entry.snapshot != "ok":
        return out("refused", f"no snapshot ({entry.snapshot})")
    if not os.path.exists(entry.real):
        return out("refused", "file no longer exists")
    current = sha256_file(entry.real)
    if entry.post_sha256 and current != entry.post_sha256:
        # Restoring here would revert whoever wrote after this turn, which is
        # the exact damage the whole feature is meant to prevent.
        return out("refused", "file changed since this turn wrote it")
    src = _pre_path(session_id, turn_id, entry.real)
    try:
        data = src.read_bytes()
        tmp = Path(entry.real).with_name(f"{Path(entry.real).name}.{os.getpid()}.revert")
        tmp.write_bytes(data)
        os.replace(tmp, entry.real)
    except Exception as exc:
        return out("refused", f"could not restore: {exc}")
    entry.reverted_at = time.time()
    return out("restored")


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------


def _maybe_prune() -> None:
    """Opportunistic, at most hourly, so a busy turn never pays for it twice."""
    global _last_prune
    now = time.time()
    if now - _last_prune < 3600:
        return
    _last_prune = now
    try:
        prune()
    except Exception:
        logger.warning("change ledger: prune failed", exc_info=True)


def prune() -> dict:
    """Drop old turn dirs, then the oldest until under the size cap.

    Only ever walks `*.changes/` under CHANGES_ROOT. The sessions directory
    also holds the transcripts and the spilled tool results, and a prune that
    reached those would be deleting the record it exists to protect.
    """
    cfg = config()
    cutoff = time.time() - float(cfg["retention_days"]) * 86400
    max_bytes = float(cfg["max_total_mb"]) * 1024 * 1024
    removed, kept = 0, []
    try:
        roots = sorted(CHANGES_ROOT.glob("*.changes"))
    except OSError:
        return {"removed": 0, "bytes": 0}

    import shutil
    for root in roots:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            try:
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                mtime = d.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
                continue
            kept.append((mtime, size, d))

    total = sum(s for _m, s, _d in kept)
    kept.sort(key=lambda t: t[0])
    i = 0
    while total > max_bytes and i < len(kept):
        _m, size, d = kept[i]
        shutil.rmtree(d, ignore_errors=True)
        total -= size
        removed += 1
        i += 1

    # Leave no empty `<sid>.changes` shells behind.
    for root in roots:
        try:
            if root.is_dir() and not any(root.iterdir()):
                root.rmdir()
        except OSError:
            pass
    return {"removed": removed, "bytes": int(total)}


def stats() -> dict:
    return {
        "turns_in_memory": len(_turns),
        "enabled": enabled(),
        "root": str(CHANGES_ROOT),
    }


def reset() -> None:
    """Tests only."""
    global _last_prune
    with _lock:
        _turns.clear()
    _last_prune = 0.0
