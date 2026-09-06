"""Prove the guardian actually rescues a broken build — against a canary.

Unit tests prove the guardian's arithmetic. They cannot prove it *acts* when
everything is on fire. This does: it deliberately breaks a scratch copy, runs
the **candidate** guardian against a canary of it, and asserts the tree came
back and the service returned.

Nothing here touches the live tree, the live supervisord, or the live state
directory. The scratch repo is a clone; the canary has its own supervisord
socket; the guardian is pointed at a scratch state dir via CLI flags. A
recording proxy wrapper asserts the live socket is never contacted.

This is gate rung 6, and it fires whenever a diff touches a protected path —
the guardian, its unit, the supervisor confs, or the health/restart path the
rollback depends on. That is the concrete form of "Lloyd may edit his own
parachute, but must demonstrate it still opens".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.selfmod import canary as C

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

BREAK_LINE = 'raise SystemExit("selfmod rehearsal: deliberate boot failure")\n'


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False)


def _prepare_scratch(scratch: Path, source: Path, base: str) -> dict:
    """Clone `source` into `scratch`, then commit a build that cannot boot."""
    scratch.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(source), str(scratch)],
                   check=True, capture_output=True)
    _git(scratch, "config", "user.email", "guardian-drill@localhost")
    _git(scratch, "config", "user.name", "guardian drill")
    _git(scratch, "checkout", "-q", "-B", "main", base)
    good = _git(scratch, "rev-parse", "HEAD").stdout.strip()

    server = scratch / "server.py"
    server.write_text(BREAK_LINE + server.read_text(encoding="utf-8"), encoding="utf-8")
    _git(scratch, "add", "-A")
    _git(scratch, "commit", "-q", "-m", "drill: deliberately unbootable")
    broken = _git(scratch, "rev-parse", "HEAD").stdout.strip()
    return {"good": good, "broken": broken}


def run_drill(round_id: str, worktree: Path, base: str, *,
              python: Path | None = None, budget: float = 240.0) -> tuple[bool, str]:
    """Return (ok, detail). Never raises — a drill error is a drill failure."""
    drill_root = Path.home() / "lloyd-work" / f"{round_id}-drill"
    scratch = drill_root / "home" / "lloyd"
    state_dir = drill_root / "selfmod-state"
    guardian_state = drill_root / "guardian-state"
    marker = drill_root / "drill-in-progress"
    python = Path(python) if python else LIVE_ROOT / ".venvs/lloyd/bin/python"

    canary = None
    try:
        shutil.rmtree(drill_root, ignore_errors=True)
        drill_root.mkdir(parents=True, exist_ok=True)
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ"), encoding="utf-8")

        shas = _prepare_scratch(scratch, worktree, base)

        # The scratch state the candidate guardian will read.
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "last_known_good.json").write_text(json.dumps({
            "schema": 1, "commit": shas["good"], "floor": shas["good"],
            "recorded_at": "1970-01-01T00:00:00Z", "health": {}, "eval": {},
        }), encoding="utf-8")
        (state_dir / "current.json").write_text(json.dumps({
            "schema": 1, "commit": shas["broken"], "parent": shas["good"],
            "rollback_target": shas["good"], "changed_paths": ["server.py"],
            "landed_ts": time.time(), "liveness_until_ts": time.time() + 600,
            "errors_until_ts": time.time() + 600, "venv_swapped": False,
        }), encoding="utf-8")

        # Boot the broken build under its own supervisord. autorestart off so
        # it fails fast into FATAL instead of thrashing for the whole budget.
        canary = C.Canary(drill_root, scratch, python=python, autorestart=False)
        canary.start()
        time.sleep(12)   # let supervisord give up and park it

        # Sanity: the drill is worthless if the build did not actually break.
        # Parse per line — a naive substring check reads the *aggregator* being
        # RUNNING as the backend being RUNNING.
        st = canary.ctl("status")
        backend_state = ""
        for line in st.stdout.splitlines():
            parts = line.split()
            if parts and parts[0].endswith("lloyd-backend"):
                backend_state = parts[1] if len(parts) > 1 else ""
                break
        if backend_state == "RUNNING":
            return False, ("drill setup failed: the deliberately broken build started "
                           f"anyway (state={backend_state})\n{st.stdout}")
        if not backend_state:
            return False, f"drill setup failed: no backend in canary status\n{st.stdout}"

        guardian_src = worktree / "agent-services" / "guardian" / "guardian.py"
        if not guardian_src.exists():
            return False, f"candidate guardian not found at {guardian_src}"

        proc = subprocess.run(
            [sys.executable if False else "/usr/bin/python3", str(guardian_src),
             "--repo", str(scratch),
             "--state", str(state_dir),
             "--guardian-state", str(guardian_state),
             "--supervisor-sock", str(canary.sock),
             "--backend-url", canary.backend_health,
             "--mcp-url", canary.mcp_health,
             "--programs", "lloyd-mc:lloyd-backend,lloyd-mc:lloyd-mcp",
             # A rehearsal must not be able to look like a production incident.
             "--no-external-alerts",
             "--once"],
            capture_output=True, text=True, timeout=budget,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        log = (proc.stdout + proc.stderr)[-3000:]

        head_now = _git(scratch, "rev-parse", "HEAD").stdout.strip()
        branch_now = _git(scratch, "symbolic-ref", "--quiet", "HEAD").stdout.strip()

        if head_now != shas["good"]:
            return False, (f"guardian did NOT restore the tree: HEAD={head_now[:8]}, "
                           f"expected {shas['good'][:8]}\n{log[-800:]}")
        if branch_now != "refs/heads/main":
            return False, f"guardian left HEAD detached ({branch_now!r})"

        # And the service must actually be back.
        healthy = False
        deadline = time.time() + 90
        while time.time() < deadline:
            status, _, _ = C._get(canary.backend_health, 3.0)
            if status == 200:
                healthy = True
                break
            time.sleep(2.0)
        if not healthy:
            return False, f"tree restored but the canary never came back healthy\n{log[-800:]}"

        events = (state_dir / "promotions.jsonl")
        kinds = []
        if events.exists():
            kinds = [json.loads(l)["event"] for l in events.read_text().splitlines() if l.strip()]
        if "rollback_succeeded" not in kinds:
            return False, f"no rollback_succeeded in the drill ledger: {kinds}"

        # The live socket must never have been touched.
        if canary.sock and str(canary.sock) in log and "/tmp/agent-supervisor.sock" in log:
            return False, "drill contacted the live supervisord socket"

        return True, (f"guardian restored {shas['broken'][:8]} → {shas['good'][:8]}, "
                      f"canary healthy again, ledger: {kinds}")

    except subprocess.TimeoutExpired:
        return False, f"drill exceeded {budget:.0f}s"
    except Exception as exc:
        return False, f"drill error: {type(exc).__name__}: {exc}"
    finally:
        try:
            if canary:
                canary.stop()
        except Exception:
            pass
        shutil.rmtree(drill_root, ignore_errors=True)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Rehearse a guardian rescue against a canary")
    ap.add_argument("--worktree", default=str(LIVE_ROOT),
                    help="tree holding the CANDIDATE guardian (default: live)")
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--round-id", default=f"DRILL_{int(time.time())}")
    ap.add_argument("--yes-i-mean-it", action="store_true",
                    help="required: this boots processes and rewrites a scratch repo")
    args = ap.parse_args(argv)
    if not args.yes_i_mean_it:
        print("refusing without --yes-i-mean-it")
        return 2

    base = subprocess.run(["git", "-C", args.worktree, "rev-parse", args.base],
                          capture_output=True, text=True).stdout.strip()
    ok, detail = run_drill(args.round_id, Path(args.worktree), base)
    print(("PASS: " if ok else "FAIL: ") + detail)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
