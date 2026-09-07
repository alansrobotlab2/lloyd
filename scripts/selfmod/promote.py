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
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.supervisor_client import restart_process, stop_process
from scripts.selfmod import state as S, worktree as W

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

IDLE_POLL_SECONDS = 2.0
IDLE_QUIET_POLLS = 3
IDLE_MAX_WAIT = 900.0
DRAIN_TTL = 180.0
# The observation window. Liveness is watched for ALL of it, not for some
# shorter sub-window: a build that crashes at minute ten is exactly as bad as
# one that crashes at minute one. There was a `LIVENESS_WINDOW = 120.0` here
# and a `liveness_until_ts` written into every promotion record, and nothing
# ever read either — the guardian applies liveness whenever a promotion is
# under observation. A constant that looks like a bound and bounds nothing is
# worse than no constant.
ERRORS_WINDOW = 900.0
RESTART_LEASE = 120.0

SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"
SUPERVISORCTL = Path.home() / ".local/share/uv/tools/supervisor/bin/supervisorctl"
SUPERVISORD_CONF = LIVE_ROOT / "agent-services" / "supervisor" / "supervisord.conf"

BACKEND = "http://127.0.0.1:8080"
MCP_HEALTH = "http://127.0.0.1:8500/health"
# Vite dev server, serving the live tree over HTTPS with a private cert. Not
# restarted by a landing: HMR picks the fast-forward up on its own.
FRONTEND_URL = "https://127.0.0.1:5173/"


class PromoteError(RuntimeError):
    pass


def _frontend_alive(url: str = FRONTEND_URL, budget: float = 30.0) -> tuple[bool, str]:
    """Does the Vite dev server still answer after a frontend landing?

    Liveness only, and deliberately so: a broken `src` change is a
    browser-side error the dev server serves with a 200. The gate's `vite
    build` is what verifies the change; this catches the one thing the
    landing itself could do to the frontend — leave it unreachable."""
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    deadline = time.time() + budget
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0, context=ctx) as resp:
                if 200 <= resp.status < 400:
                    return True, f"HTTP {resp.status}"
                last = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except Exception as exc:  # refused, TLS, timeout
            last = type(exc).__name__
        time.sleep(2.0)
    return False, last or "no answer"


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
    """Require N consecutive quiet polls before touching anything.

    "Quiet" means all three counters are zero. `harness_runs` is the one that
    was missing: it counts agent loops in flight by ANY caller, and worker jobs
    never enter a session queue. A landing that restarts the backend during a
    ten-minute research job kills it, and the connection failures it logs on
    the way down land inside the window the error-rate detector is watching —
    so the promotion is reverted for the damage its own landing caused.
    """
    deadline = time.time() + max_wait
    quiet = 0
    busiest = ""
    while time.time() < deadline:
        status, body = _get(f"{BACKEND}/health")
        if status == 200 and body:
            turns = body.get("turns") or {}
            busy = (turns.get("active", 1) or turns.get("queued", 1)
                    or turns.get("harness_runs", 0))
            if not busy:
                quiet += 1
                if quiet >= IDLE_QUIET_POLLS:
                    return True, f"idle for {quiet} consecutive polls"
            else:
                quiet = 0  # a turn appearing resets the counter
                busiest = (f"active={turns.get('active')} queued={turns.get('queued')} "
                           f"harness_runs={turns.get('harness_runs')}")
        else:
            quiet = 0
        time.sleep(IDLE_POLL_SECONDS)
    return False, (f"backend never went idle within {max_wait:.0f}s"
                   + (f" (last: {busiest})" if busiest else ""))


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


def _run(argv: list, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True,
                          timeout=timeout, check=False)


def _apply_service_changes(changed: list[str]) -> list[str]:
    """Make changed service DEFINITIONS take effect. Returns human-readable notes.

    `spec.py` calls the supervisor confs, the systemd units and the guardian
    "protected" — allowed to change, provided the drill passes. But nothing
    ever *applied* such a change: supervisord includes conf.d straight out of
    the repo and needs `reread`/`update` to notice, systemd units are copies
    under ~/.config and the repo edit reached nothing at all, and the guardian
    runs from a pinned snapshot that is only re-staged on a unit restart. So a
    round could pass a full drill, land, be marked healthy, and leave the
    running system on the old definition indefinitely — the change looked
    delivered and was not.

    Never fatal to the promotion: the code is already live and verified by the
    time this runs, and a failure to reload a conf is worth an alert, not a
    revert.
    """
    notes: list[str] = []
    touched_supervisor = any(p.startswith("agent-services/supervisor/") for p in changed)
    touched_units = [p for p in changed if p.startswith("agent-services/systemd/")]
    touched_guardian = any(p.startswith("agent-services/guardian/") for p in changed)

    if touched_supervisor and SUPERVISORCTL.exists():
        for verb in ("reread", "update"):
            r = _run([SUPERVISORCTL, "-c", SUPERVISORD_CONF, verb], timeout=120)
            notes.append(f"supervisorctl {verb}: "
                         f"{'ok' if r.returncode == 0 else r.stderr.strip()[:120]}")

    for rel in touched_units:
        src = LIVE_ROOT / rel
        if not src.is_file():
            continue
        try:
            SYSTEMD_USER_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, SYSTEMD_USER_DIR / src.name)
            notes.append(f"installed {src.name}")
        except OSError as exc:
            notes.append(f"FAILED to install {src.name}: {exc}")
    if touched_units:
        r = _run(["systemctl", "--user", "daemon-reload"])
        notes.append("daemon-reload: "
                     f"{'ok' if r.returncode == 0 else r.stderr.strip()[:120]}")

    # The guardian runs a pinned snapshot, re-staged by ExecStartPre. Without
    # this a landed guardian change is inert until something else restarts the
    # unit. Staging declines a candidate that does not compile or fails its own
    # selftest, so the worst case is that the previous snapshot keeps running.
    if touched_guardian or any(p.endswith("lloyd-guardian.service") for p in touched_units):
        r = _run(["systemctl", "--user", "restart", "lloyd-guardian"], timeout=120)
        notes.append("guardian restarted: "
                     f"{'ok' if r.returncode == 0 else r.stderr.strip()[:120]}")
    return notes


def _announce_promoted(round_id: str, commit: str, changed: list) -> None:
    """Say out loud that the loop just landed code on itself.

    Until now the self-modification loop only ever spoke when it *failed*:
    every notify-send in the tree hung off a guardian alert. A loop that can
    rewrite the running system in the background and is silent when it works
    is the wrong way round — the successful landings are the ones nobody is
    watching a terminal for.

    Routed through the guardian's `Notifier.announce` rather than a private
    notify-send so it shares the one fan-out, and guarded end to end: an
    announcement must never be able to fail a promotion that already
    succeeded and is being observed.
    """
    try:
        import sys
        gdir = Path(__file__).resolve().parents[2] / "agent-services" / "guardian"
        if not (gdir / "notify.py").is_file():
            return
        if str(gdir) not in sys.path:
            sys.path.insert(0, str(gdir))
        import gstate, notify as notify_mod, policy
        notifier = notify_mod.Notifier(
            ledger=gstate.SelfModState(Path(policy.SELFMOD_STATE)).ledger,
            state_dir=Path(policy.GUARDIAN_STATE),
            vault_root=policy.VAULT_ROOT,
            voice_window=policy.VOICE_REPEAT_SECONDS,
        )
        n = len(changed)
        notifier.announce(
            f"Promoted {round_id}",
            f"{n} file{'' if n == 1 else 's'} changed. "
            f"Watching for {int(ERRORS_WINDOW // 60)} minutes.",
        )
    except Exception:
        pass


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

    # The gate judged a specific commit. Anything committed into the worktree
    # afterwards would land completely ungated: `land` passes the base along
    # but the candidate HEAD was re-read from the worktree here, so an extra
    # commit made after a passing gate was indistinguishable from the one that
    # passed it.
    gate_head = (gate_report or {}).get("head")
    if gate_head and gate_head != head:
        raise PromoteError(
            f"worktree HEAD moved since the gate ran ({head[:8]} != gated "
            f"{gate_head[:8]}) — re-gate before landing")

    # One promotion under observation at a time. A second landing overwrote
    # `current.json`, so the first promotion never settled, never advanced the
    # LKG, and — worse — the new record's rollback target became a commit that
    # had never survived a window, quietly breaking the invariant the whole
    # promoter/guardian split exists to guarantee.
    observed = S.read_current()
    if observed and observed.get("state") in ("landing", "observing"):
        until = float(observed.get("errors_until_ts") or 0)
        left = max(0.0, until - time.time())
        raise PromoteError(
            f"{str(observed.get('commit'))[:8]} is still under observation "
            f"({observed.get('state')}, {left / 60:.1f} min left) — a promotion must "
            "settle before the next one lands")

    changed_preview = W.changed_paths(Path(worktree), base)
    tree_hash = S.changed_tree_hash(worktree, head, changed_preview)
    if S.is_denied(commit=head, tree_hash=tree_hash):
        raise PromoteError(
            f"{head[:8]} is on the rollback denylist"
            + (" (matched by content, not SHA — this change was reverted before "
               "and has been re-derived)" if tree_hash and not S.is_denied(commit=head)
               else ""))
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
    current = {
        "schema": 1,
        "round_id": round_id,
        "commit": head,
        "parent": live_head,
        "rollback_target": live_head,
        "branch": f"selfmod/{round_id}",
        "state": "landing",
        "landed_at": None,
        "landed_ts": None,
        # Deliberately not started yet. The idle gate can wait up to 15
        # minutes for a turn to finish, and starting the observation clock
        # here would burn most of the window before the code is even live —
        # the guardian would then settle a promotion it had barely watched.
        # Both are set after the restart verifies, below.
        "errors_until_ts": None,
        "changed_paths": changed,
        "tree_hash": tree_hash,
        "venv_swapped": False,
        "touched_guardian": any(p.startswith("agent-services/guardian/") for p in changed),
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
        if turns.get("active") or turns.get("queued") or turns.get("harness_runs"):
            raise PromoteError(f"a turn started during the drain handshake: {turns}")

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

        # Service definitions the diff changed must reach the running system
        # BEFORE the restart, or the restart re-reads the old ones.
        service_notes = _apply_service_changes(changed)

        # ── restart, MCP first ─────────────────────────────────────────
        for program, health in (("lloyd-mcp", MCP_HEALTH), ("lloyd-backend", f"{BACKEND}/health")):
            # Refresh the lease before each leg. The lease is 120s and this
            # loop can legitimately spend 90s per service waiting on health,
            # so a single lease taken before the merge could expire mid-restart
            # and hand the guardian its own deploy to judge.
            S.set_pause(RESTART_LEASE)
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

        if any(p.startswith("web/") for p in changed):
            alive, note = _frontend_alive()
            if not alive:
                raise PromoteError(f"frontend unreachable after landing ({note}) at {FRONTEND_URL}")

        # The code is live and verified: start the clock now.
        landed = time.time()
        current["state"] = "observing"
        current["landed_at"] = S.now_iso()
        current["landed_ts"] = landed
        current["errors_until_ts"] = landed + ERRORS_WINDOW
        current["boot_id"] = (body or {}).get("boot_id")
        current["service_changes"] = service_notes
        S.write_verified(S.CURRENT_PATH, current)
        result["service_changes"] = service_notes
        S.append_event({"event": "promoted", "round_id": round_id, "commit": head,
                        "parent": live_head, "changed_paths": changed,
                        "tree_hash": tree_hash, "service_changes": service_notes,
                        "errors_until": current["errors_until_ts"]})
        _announce_promoted(round_id, head, changed)
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
