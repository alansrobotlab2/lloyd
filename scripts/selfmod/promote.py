"""Land a gated candidate on the live tree, reversibly.

Ordering is the whole design, and one step is non-negotiable:

**The rollback point is written and read back BEFORE anything is mutated.**
`scripts/autoresearch/promote.py::snapshot_current_prompts` mkdirs
unconditionally, never verifies the copy landed, and `promote()` overwrites
live state regardless — which is why 26 of 83 historical promotions have no
snapshot and therefore no way back (see
`tests/test_autoresearch_promotion.py:362`, an xfail documenting the live
defect). Here `write_verified` raises unless the state file round-trips, and
nothing touches the tree until it has.

The promoter never advances `last_known_good`. It writes `current.json` with
an observation window; the **guardian** promotes that to LKG only after the
window passes clean. So "last known good" means *observed healthy in
production*, and a rollback always targets a commit that already survived a
full window.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.selfmod import state as S, worktree as W

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

IDLE_POLL_SECONDS = 2.0
IDLE_QUIET_POLLS = 3
IDLE_MAX_WAIT = 900.0
DRAIN_TTL = 180.0
LIVENESS_WINDOW = 120.0     # guardian watches for a crash this long
ERRORS_WINDOW = 900.0       # ...and for an error spike this long
RESTART_LEASE = 120.0

BACKEND = "http://127.0.0.1:8080"
MCP_HEALTH = "http://127.0.0.1:8500/health"


class PromoteError(RuntimeError):
    pass


def _get(url: str, timeout: float = 5.0):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(url), timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", "replace"))
        except Exception:
            return e.code, None
    except Exception:
        return None, None


def _post(url: str, payload: dict, timeout: float = 5.0) -> bool:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def set_drain(on: bool, ttl: float = DRAIN_TTL) -> bool:
    return _post(f"{BACKEND}/api/selfmod/drain", {"on": on, "ttl_s": ttl})


def wait_idle(max_wait: float = IDLE_MAX_WAIT) -> tuple[bool, str]:
    """Require N consecutive quiet polls before touching anything."""
    deadline = time.time() + max_wait
    quiet = 0
    while time.time() < deadline:
        status, body = _get(f"{BACKEND}/health")
        if status == 200 and body:
            turns = body.get("turns") or {}
            if turns.get("active", 1) == 0 and turns.get("queued", 1) == 0:
                quiet += 1
                if quiet >= IDLE_QUIET_POLLS:
                    return True, f"idle for {quiet} consecutive polls"
            else:
                quiet = 0  # a turn appearing resets the counter
        else:
            quiet = 0
        time.sleep(IDLE_POLL_SECONDS)
    return False, f"backend never went idle within {max_wait:.0f}s"


def count_kg_rows() -> int | None:
    import sqlite3
    db = LIVE_ROOT / "_pipeline" / "vault-derived" / "kg.sqlite"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        try:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            return sum(con.execute(f"SELECT count(*) FROM '{n}'").fetchone()[0] for n in names)
        finally:
            con.close()
    except Exception:
        return None


def count_vault_files() -> int | None:
    root = Path.home() / "obsidian"
    if not root.is_dir():
        return None
    try:
        return sum(1 for p in root.rglob("*") if p.is_file())
    except OSError:
        return None


def _err_size() -> int:
    try:
        return (LIVE_ROOT / "logs" / "server.err").stat().st_size
    except OSError:
        return 0


def promote(round_id: str, worktree: Path, base: str, *,
            gate_report: dict | None = None, dry_run: bool = False) -> dict:
    live = LIVE_ROOT
    head = W.head(Path(worktree))
    if not head:
        raise PromoteError("cannot read the candidate HEAD")

    if S.is_halted():
        raise PromoteError(f"promotions are halted: {S.HALTED_PATH}")
    if S.is_broken():
        raise PromoteError(f"guardian is in a BROKEN state: {S.BROKEN_PATH}")
    if S.is_denied(commit=head):
        raise PromoteError(f"{head[:8]} is on the rollback denylist")
    if not W.is_clean(live):
        raise PromoteError("live tree is dirty — refusing to land onto uncommitted work")

    live_head = subprocess.run(["git", "-C", str(live), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
    if live_head != base:
        raise PromoteError(f"live HEAD moved since the gate ran ({live_head[:8]} != {base[:8]})")

    changed = W.changed_paths(Path(worktree), base)
    result: dict = {"round_id": round_id, "commit": head, "parent": live_head,
                    "changed_paths": changed, "dry_run": dry_run}
    if dry_run:
        result["would_promote"] = True
        return result

    # ── the rollback point, verified before anything moves ─────────────
    now = time.time()
    current = {
        "schema": 1,
        "round_id": round_id,
        "commit": head,
        "parent": live_head,
        "rollback_target": live_head,
        "branch": f"selfmod/{round_id}",
        "landed_at": S.now_iso(),
        "landed_ts": now,
        "liveness_until_ts": now + LIVENESS_WINDOW,
        "errors_until_ts": now + ERRORS_WINDOW,
        "changed_paths": changed,
        "venv_swapped": False,
        "touched_guardian": any(p.startswith("agent-services/guardian/") for p in changed),
        "err_offset": _err_size(),
        "kg_rows": count_kg_rows(),
        "vault_files": count_vault_files(),
        "gate": (gate_report or {}).get("rungs"),
    }
    _, body = _get(f"{BACKEND}/health")
    current["boot_id"] = (body or {}).get("boot_id")
    S.write_verified(S.CURRENT_PATH, current)   # raises unless it round-trips

    # ── idle gate + drain ──────────────────────────────────────────────
    ok, why = wait_idle()
    if not ok:
        S.clear_current()
        raise PromoteError(why)
    set_drain(True, DRAIN_TTL)
    try:
        status, body = _get(f"{BACKEND}/health")
        turns = (body or {}).get("turns") or {}
        if turns.get("active") or turns.get("queued"):
            raise PromoteError("a turn started during the drain handshake")

        # ── land ───────────────────────────────────────────────────────
        S.set_pause(RESTART_LEASE)   # the guardian must not read our own restart as a crash
        merge = subprocess.run(
            ["git", "-C", str(live), "merge", "--ff-only", f"selfmod/{round_id}"],
            capture_output=True, text=True)
        if merge.returncode != 0:
            raise PromoteError(f"fast-forward failed: {merge.stderr.strip()[:300]}")

        venv_clone = Path(worktree) / ".venvs" / "lloyd"
        if venv_clone.exists():
            import os
            venvs = live / ".venvs"
            os.rename(venvs / "lloyd", venvs / "lloyd.prev")
            os.rename(venv_clone, venvs / "lloyd")
            current["venv_swapped"] = True
            S.write_verified(S.CURRENT_PATH, current)

        # ── restart, MCP first ─────────────────────────────────────────
        from app.supervisor_client import restart_process
        for program, health in (("lloyd-mcp", MCP_HEALTH), ("lloyd-backend", f"{BACKEND}/health")):
            ok, msg = restart_process(program)
            if not ok:
                raise PromoteError(f"restart {program} failed: {msg}")
            if not _wait_health(health, 90.0):
                raise PromoteError(f"{program} never became healthy after restart")

        # ── prove the RUNNING code changed ─────────────────────────────
        _, body = _get(f"{BACKEND}/health")
        actual = (body or {}).get("commit")
        if actual != head:
            raise PromoteError(f"backend reports commit {actual}, expected {head} "
                               "— the restart did not pick up the new code")
        if current.get("boot_id") and (body or {}).get("boot_id") == current["boot_id"]:
            raise PromoteError("backend boot_id unchanged — the process was never replaced")

        current["boot_id"] = (body or {}).get("boot_id")
        S.write_verified(S.CURRENT_PATH, current)
        S.append_event({"event": "promoted", "round_id": round_id, "commit": head,
                        "parent": live_head, "changed_paths": changed,
                        "errors_until": current["errors_until_ts"]})
        result["promoted"] = True
        return result

    except Exception:
        # Any failure between the merge and the verification: revert now
        # rather than waiting for the guardian's next tick.
        S.clear_pause()
        try:
            _rollback_inline(live, live_head)
            S.append_event({"event": "rollback_succeeded", "trigger": "promote_failed",
                            "commit": head, "restored": live_head})
        except Exception as exc:
            S.append_event({"event": "rollback_failed", "trigger": "promote_failed",
                            "commit": head, "error": str(exc)[:400]})
        S.clear_current()
        raise
    finally:
        set_drain(False)
        S.clear_pause()


def _wait_health(url: str, budget: float) -> bool:
    deadline = time.time() + budget
    while time.time() < deadline:
        status, _ = _get(url, 3.0)
        if status == 200:
            return True
        time.sleep(1.0)
    return False


def _rollback_inline(live: Path, target: str) -> None:
    """Reuse the guardian's rollback rather than reimplementing it."""
    import importlib.util
    guardian_dir = live / "agent-services" / "guardian"
    spec_ = importlib.util.spec_from_file_location("_g_rollback", guardian_dir / "rollback.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)

    from app.supervisor_client import restart_process, stop_process
    for program in ("lloyd-backend", "lloyd-mcp"):
        stop_process(program, wait=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mod.preserve_evidence(str(live), S.BROKEN_DIR / stamp, f"guardian-broken-{stamp}")
    mod.restore_tree(str(live), target, ("app", "agent_mcp", "workers", "scripts",
                                         "eval", "tests"),
                     ("app", "agent_mcp", "workers", "scripts"))
    mod.verify_tree(str(live), target)
    for program, health in (("lloyd-mcp", MCP_HEALTH), ("lloyd-backend", f"{BACKEND}/health")):
        restart_process(program)
        _wait_health(health, 90.0)
