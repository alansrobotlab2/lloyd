"""All guardian tunables, in one pure-data module.

This file is read from the **pinned snapshot** at
``~/.local/state/lloyd-guardian/bin/policy.py``, never from the live repo.
That is deliberate and it is what makes "Lloyd may edit the guardian" safe:
editing the thresholds or the protected-path list in the working tree has no
effect until a staged copy has compiled and passed `selftest.py`. A bound on
how the parachute changes, not a ban on changing it.

Stdlib only. Nothing here may import from `app/`, `agent_mcp/` or `scripts/`.
"""

# ── Polling ────────────────────────────────────────────────────────────────
TICK_SECONDS = 5.0

# Grace after a (re)start during which a process cannot be judged down. The
# backend's own startsecs is 25s; this is the probe-side equivalent.
BOOT_GRACE = {
    "lloyd-mc:lloyd-backend": 45.0,
    "lloyd-mc:lloyd-mcp": 30.0,
}
DEFAULT_BOOT_GRACE = 30.0

# Consecutive failed HTTP probes before a RUNNING process is called down.
#
# Split by cause, because they mean different things. A REFUSED connection
# means nothing is listening — the process is gone, and 3 x 5s = 15s is
# plenty. A TIMEOUT means the socket accepted but the app did not answer in
# time, which for this backend is routine: /health is served by the same
# asyncio loop as the agent's own work, and an hourly autoresearch round fans
# 77 bench trials through it.
#
# This is not hypothetical. On 2026-09-06 an autoresearch round starting at
# 11:29:18 made /health miss three consecutive 2s probes, and the guardian
# reverted a perfectly good promotion at 11:30:39. A watchdog that reverts
# good code every time the machine gets busy is worse than no watchdog.
PROBE_FAIL_STREAK = 3            # refused / http-error
PROBE_TIMEOUT_STREAK = 24        # 24 x 5s = 2 minutes of no answer

# supervisord itself unreachable for this many consecutive ticks before we
# try to restart the unit. Never a code-rollback trigger.
SUPERVISORD_DOWN_STREAK = 3

# A crash loop that never reaches FATAL: the spawn timestamp advancing N times
# inside a window means the process is cycling even if every individual sample
# says RUNNING. This is the predicate that catches what `autorestart=true`
# plus a too-small `startsecs` produces.
CRASH_LOOP_STARTS = 3
CRASH_LOOP_WINDOW_SECONDS = 180.0

# ── Watched processes ──────────────────────────────────────────────────────
# Group-qualified, because supervisord's XML-RPC rejects bare names for
# grouped programs (Fault 10 BAD_NAME).
WATCHED = ("lloyd-mc:lloyd-backend", "lloyd-mc:lloyd-mcp")
# Warn-only: a dead Vite dev server does not justify rewriting history.
ADVISORY = ("lloyd-mc:lloyd-frontend",)

RESTART_ORDER = ("lloyd-mc:lloyd-mcp", "lloyd-mc:lloyd-backend")

# ── Endpoints ──────────────────────────────────────────────────────────────
BACKEND_HEALTH_URL = "http://127.0.0.1:8080/health"
MCP_HEALTH_URL = "http://127.0.0.1:8500/health"
# Generous: this is how long a BUSY event loop may take to answer, not how
# long a healthy one does (measured at 0.37ms).
PROBE_TIMEOUT_SECONDS = 10.0
HEALTH_WAIT_MCP = 60.0
HEALTH_WAIT_BACKEND = 90.0

# ── Maintenance lease ──────────────────────────────────────────────────────
# The promoter takes a short lease around its own restart. Capped here, in the
# snapshot, so a forgotten or malicious lease cannot disable the watchdog.
PAUSE_MAX_SECONDS = 1800.0

# ── Rollback ───────────────────────────────────────────────────────────────
REPO = "/home/alansrobotlab/lloyd"
# Paths cleaned after a reset. PATH-SCOPED, never the repo root: the root
# holds usage.db, workers.db, mc-state.json, .env and .venvs/, all gitignored
# and none of them replaceable. A bare `git clean -fdx` here is a data-loss
# event.
CLEAN_PATHS = ("app", "agent_mcp", "workers", "scripts", "eval", "tests")
PYCACHE_PATHS = ("app", "agent_mcp", "workers", "scripts")
ROLLBACK_MAX_ATTEMPTS = 2
ROLLBACK_RETRY_SECONDS = 60.0
INDEX_LOCK_STALE_SECONDS = 60.0
WRITER_DRAIN_SECONDS = 20.0
# Quiet period after a rollback in which only liveness may fire — an
# error-rate or eval trigger immediately post-rollback is almost certainly
# measuring the rollback itself.
POST_ROLLBACK_QUIET_SECONDS = 600.0

# ── Flap protection ────────────────────────────────────────────────────────
FLAP_WINDOW_SECONDS = 6 * 3600.0
FLAP_HALT_AFTER = 2      # halt promotions
FLAP_STOP_AFTER = 3      # also stop the backend

# ── Error-rate detection ───────────────────────────────────────────────────
LOG_FILES = (
    "/home/alansrobotlab/lloyd/logs/server.err",
    "/home/alansrobotlab/lloyd/logs/mcp.err",
)
# NOT server.log / mcp.log. server.py's logging.basicConfig writes to stderr,
# which supervisord maps to *.err, so all application logs (INFO through
# CRITICAL, and every traceback) land there. server.log is uvicorn's access
# log and contains zero error-shaped lines — a watchdog grepping it would find
# nothing forever.
LOG_READ_CAP_BYTES = 4 * 1024 * 1024
CHRONIC_MIN_DISTINCT_HOURS = 3
CHRONIC_LOOKBACK_DAYS = 7
NOVEL_SIGNATURE_THRESHOLD = 5           # one novel signature this many times
NOVEL_FATAL_DISTINCT_THRESHOLD = 3      # distinct novel tracebacks
NOVEL_IN_CHANGED_PATH_THRESHOLD = 2     # novel + names a file the promo touched
ERROR_RATE_FLOOR_PER_MIN = 20.0
ERROR_RATE_MULTIPLIER = 8.0

# ── Worker failure CUSUM ───────────────────────────────────────────────────
WORKERS_DB = "/home/alansrobotlab/lloyd/workers.db"
CUSUM_P1 = 0.30
CUSUM_THRESHOLD = 4.6                   # ~1% false alarm
CUSUM_P0_FLOOR = 0.01

# ── Data-damage tripwire ───────────────────────────────────────────────────
# The one failure class `git reset --hard` structurally cannot undo: the KG
# and the vault are gitignored, so a change that deletes rows or notes boots
# fine, logs nothing, and survives the revert.
DATA_DROP_FRACTION = 0.05
KG_DB = "/home/alansrobotlab/lloyd/_pipeline/vault-derived/kg.sqlite"
VAULT_ROOT = "/home/alansrobotlab/obsidian"

# ── Paths ──────────────────────────────────────────────────────────────────
import os as _os
from pathlib import Path as _Path

SELFMOD_STATE = _Path(
    _os.environ.get("LLOYD_SELFMOD_STATE", _Path.home() / ".local/state/lloyd-selfmod")
)
GUARDIAN_STATE = _Path(
    _os.environ.get("LLOYD_GUARDIAN_STATE", _Path.home() / ".local/state/lloyd-guardian")
)
SUPERVISOR_SOCK = _os.environ.get("LLOYD_SUPERVISOR_SOCK", "/tmp/agent-supervisor.sock")
# Must exceed stopwaitsecs (15s) — a blocking stopProcess(wait=True) legitimately
# takes that long, and a shorter client timeout reports "error: timed out" for a
# stop that is actually working, so the rollback proceeds without knowing whether
# the writers are down. Observed exactly that during the 11:30 rollback.
SUPERVISOR_RPC_TIMEOUT = 45.0
SUPERVISORD_UNIT = "agent-supervisord.service"

SELFTEST_INTERVAL_SECONDS = 24 * 3600.0
HEARTBEAT_NAME = "heartbeat.json"
