"""On-disk state for the self-modification loop.

Everything lives under ``~/.local/state/lloyd-autoimplement/`` — deliberately
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
``tests/test_autoimplement_state.py`` asserts the two behave differently so nobody
later refactors them together.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


STATE_DIR = Path(
    os.environ.get("LLOYD_AUTOIMPLEMENT_STATE", Path.home() / ".local" / "state" / "lloyd-autoimplement")
)

LKG_PATH = STATE_DIR / "last_known_good.json"
CURRENT_PATH = STATE_DIR / "current.json"
# Written by the guardian when a promotion settles. `current.json` is DELETED
# at settle, so anything that wants to ask "what landed recently, and what did
# it replace?" after the 15-minute window has nothing to read. That is not
# hypothetical: the nightly regression check keyed on `current.json` and a
# 24-hour job therefore found a promotion under observation essentially never.
LAST_SETTLED_PATH = STATE_DIR / "last_settled.json"
# A rollback the guardian should perform on someone else's behalf. Nothing in
# the backend or the aggregator may roll back inline: both are stopped by the
# rollback itself, so the process doing it dies partway through. The guardian
# is the one component outside that failure domain, and it already owns
# evidence preservation, retry, the denylist and flap protection.
ROLLBACK_REQUEST_PATH = STATE_DIR / "rollback_request.json"
# Last measured quality baseline, written by the autoimplement-regression worker and
# READ (never written) by the guardian, which folds it into the LKG record at
# settle. Keeps "the guardian is the only writer of last_known_good.json"
# true while still letting a venv-only measurement reach it.
EVAL_LAST_PATH = STATE_DIR / "eval_last.json"
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


def read_last_settled() -> dict | None:
    """The most recent promotion that survived its observation window.

    Carries the promotion's own `parent`, which is what a post-landing quality
    check must compare against — by the time it runs, the LKG pointer has
    already advanced to the promoted commit, so comparing against the LKG
    would compare a commit with itself.
    """
    return read_json(LAST_SETTLED_PATH)


def read_eval_last() -> dict | None:
    return read_json(EVAL_LAST_PATH)


def write_eval_last(payload: dict) -> None:
    write_json(EVAL_LAST_PATH, payload)


# ---------------------------------------------------------------------------
# Rollback requests — the only way a process inside the blast radius asks for
# a revert
# ---------------------------------------------------------------------------

def request_rollback(*, reason: str, trigger: str, target: str | None = None,
                     commit: str | None = None,
                     changed_paths: list | None = None) -> dict:
    """Ask the guardian to roll back. Returns the request as written.

    `target` is optional: omitted, the guardian resolves it the way it would
    for a crash (the promotion's own recorded rollback_target first). Passing
    one is for a manual revert to a specific commit, and the guardian still
    validates it against the floor and the object store before acting.
    """
    payload = {
        "requested_at": now_iso(),
        "ts": time.time(),
        "reason": str(reason)[:2000],
        "trigger": str(trigger)[:100],
        "target": target,
        "commit": commit,
        # Carried so a request about an ALREADY-SETTLED promotion can still be
        # reverted surgically: by then `current.json` is gone, and without the
        # bad commit the guardian can only reset bluntly to the target.
        "changed_paths": list(changed_paths or []),
        "pid": os.getpid(),
    }
    write_json(ROLLBACK_REQUEST_PATH, payload)
    append_event({"event": "rollback_requested", "trigger": trigger,
                  "reason": str(reason)[:1000], "target": target, "commit": commit})
    return payload


def read_rollback_request() -> dict | None:
    return read_json(ROLLBACK_REQUEST_PATH)


def clear_rollback_request() -> None:
    try:
        ROLLBACK_REQUEST_PATH.unlink()
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


def changed_tree_hash(repo, commit: str, paths: list[str]) -> str | None:
    """Content hash of `paths` as they stand at `commit`.

    This is the half of the denylist that makes it mean something. Keying only
    on the SHA catches re-landing the *same commit*, which nothing was ever
    going to do — a round that is re-cut produces a new SHA for identical
    content and sails straight past. Hashing the blob ids of the paths the
    promotion touched catches the change however it is re-derived.

    Returns None when git cannot answer, and callers must treat that as "not
    denied" rather than as a denial: refusing to promote because git hiccuped
    would be a worse failure than the one this prevents.
    """
    if not paths:
        return None
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "ls-tree", "-r", "--full-tree", commit, "--", *paths],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return hashlib.sha1(r.stdout.encode("utf-8", "replace")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Detached execution
# ---------------------------------------------------------------------------

def write_gate_report(round_id: str, report: dict) -> Path:
    """`gate.json` is what `land` reads its base and head from. Written by
    `run_gate` and, after a rebase-and-retest inside the promoter, by
    `promote` — one writer for the shape, or the two drift."""
    out = ROUNDS_DIR / round_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gate.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def update_run_spec_base(round_id: str, base: str) -> bool:
    """Move the round's recorded base after a rebase.

    `run_gate` reads `code.base_commit` from `run_spec.yaml`, and `changed_paths`
    is `base...HEAD`. Leave the old base in place after a rebase and the next
    `autoimplement_gate` call computes the round's diff against a commit that is no
    longer its parent — sweeping every file the human committed in between
    into the round's changed paths, and from there into scope checks, the
    tree hash and the promotion record. False if there is no spec to update.
    """
    path = ROUNDS_DIR / round_id / "run_spec.yaml"
    if not path.exists():
        return False
    import yaml
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    code = spec.setdefault("code", {})
    if code.get("base_commit") == base:
        return True
    code["base_commit"] = base
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return True


def spawn_detached(argv: list[str], log_path: Path, cwd=None) -> int:
    """Run `argv` in its own session, and return its pid.

    A landing restarts `lloyd-mcp` and `lloyd-backend`. When it is driven from
    an MCP tool the promoter is running *inside* `lloyd-mcp`, and both confs
    set `stopasgroup`/`killasgroup` — so supervisord signals the whole process
    group and kills the promoter mid-flight. What that leaves behind is the
    worst of all the states: the tree fast-forwarded onto the new commit, the
    aggregator stopped (an intentional stop is not auto-restarted), the backend
    still serving old code with no tools, and `current.json` frozen at
    `landing`, which the guardian is specifically told to ignore. Nothing is
    watching, and nothing rolls back.

    `start_new_session=True` puts the child in a new session and therefore a
    new process group, which is exactly what a group signal cannot reach. The
    CLI path survived this by accident, because the Bash tool already spawns
    its children that way.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            [str(a) for a in argv],
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    finally:
        fh.close()
    return proc.pid


# ---------------------------------------------------------------------------
# The master switch
# ---------------------------------------------------------------------------

class AutoimplementDisabled(RuntimeError):
    """`autoimplement.enabled` is false in config.yaml."""


def is_enabled(repo=None) -> bool:
    """Read `autoimplement.enabled` straight from config.yaml.

    Read raw rather than through `app.config` so this works from a stdlib-ish
    context and cannot be affected by overlay expansion. It is an interlock
    against accident, not a sandbox: anything holding the Bash tool can edit
    config.yaml. Its value is that neither the MCP tools NOR the CLI can be
    invoked into a live promotion without a deliberate human edit — before,
    only the tool wrapper checked, and the CLI is what the skill's own worked
    examples use.
    """
    root = Path(repo) if repo else Path(__file__).resolve().parent.parent.parent
    try:
        import yaml
        raw = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    return bool((raw.get("autoimplement") or {}).get("enabled", False))


def require_enabled(action: str, repo=None) -> None:
    if not is_enabled(repo):
        raise AutoimplementDisabled(
            f"autoimplement.enabled is false in config.yaml — refusing to {action}. "
            "The loop ships inert; enabling it is a deliberate human decision.")


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
