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
# Three causes, three budgets. Collapsing them is how a busy box gets its
# code reverted (see the 2026-09-06 incident) and how a merely *degraded*
# aggregator gets read as a dead one.
PROBE_FAIL_STREAK = 3            # REFUSED: nothing is listening. 3 x 5s = 15s.
PROBE_TIMEOUT_STREAK = 24        # TIMEOUT: accepted but busy. 24 x 5s = 2 min.
# HTTP_ERROR: the socket answered with a non-200. The process is demonstrably
# alive and serving, so this is never death — but a backend parked in
# `degraded` (a router that failed to mount, startup that never completed) is
# a real failure a promotion can cause, so it gets its own, wider budget.
# For the AGGREGATOR this counter is not consulted at all: a 503 there means
# "some module is degraded", which `mcp_degraded_is_fatal` judges properly
# against the last-known-good baseline. Counting it here instead reverted on
# any three consecutive ticks of a Thunderbird bridge being closed.
PROBE_HTTP_ERROR_STREAK = 36     # 36 x 5s = 3 minutes of answering, badly

# supervisord itself unreachable for this many consecutive ticks before we
# try to restart the unit. Never a code-rollback trigger.
SUPERVISORD_DOWN_STREAK = 3

# Consecutive ticks an aggregator verdict must hold before it reverts code.
# `mcp_degraded_is_fatal` fires on a body reporting zero tools, and an
# aggregator answering mid-restart parses to exactly that — so without a
# streak a single bad response reverts a promotion on its own. Every other
# detector here confirms across ticks; this one reached that verdict only
# via a 503, which hid how sharp it was.
MCP_FATAL_STREAK = 3

# A crash loop that never reaches FATAL: the spawn timestamp advancing N times
# inside a window means the process is cycling even if every individual sample
# says RUNNING. This is the predicate that catches what `autorestart=true`
# plus a too-small `startsecs` produces.
CRASH_LOOP_STARTS = 3
CRASH_LOOP_WINDOW_SECONDS = 180.0

# ── Watched processes ──────────────────────────────────────────────────────
# Group-qualified, because supervisord's XML-RPC rejects bare names for
# grouped programs (Fault 10 BAD_NAME).
# `lloyd-frontend` is deliberately absent and stays absent: a dead Vite dev
# server does not justify rewriting history, and there was an `ADVISORY` tuple
# here that named it and that nothing ever read — a knob an operator would
# reasonably think was doing something.
WATCHED = ("lloyd-mc:lloyd-backend", "lloyd-mc:lloyd-mcp")

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
# holds .env and .venvs/, gitignored and not replaceable. A bare
# `git clean -fdx` here is a data-loss event. (The runtime data — sessions,
# the databases, logs — moved out to DATA_ROOT after 2026-09-22.)
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

# ── Runtime data root ──────────────────────────────────────────────────────
# Sessions, databases and logs, outside the code tree since 2026-09-22
# (`app/paths.py` resolves the same place; `datawatch.py` guards it).
DATA_ROOT = __import__("os").environ.get("LLOYD_DATA", "/home/alansrobotlab/lloyd-data")
# How often the tree is checked for a runtime name that came back into it.
STRAY_CHECK_SECONDS = 3600.0
# How often the snapshot directory is asked how old its newest snapshot is. The
# snapshots arrive hourly, so asking faster than a quarter of that buys nothing.
SNAPSHOT_CHECK_SECONDS = 900.0
# A snapshot older than this means the hourly timer has stopped delivering:
# three periods, so one skipped hour — a refusal while the data tripwire is set,
# a machine that was asleep — is not an alarm. A run of silent refusals is what
# this catches and nothing else does: `snapshot-data.sh` exits 0 on a refusal by
# design, its `Type=oneshot` unit still reports `Result=success`, and pruning
# never deletes the newest snapshot, so the entry count stays >= 1 forever
# (backlog #1416).
SNAPSHOT_MAX_AGE_SECONDS = 3 * 3600.0

# ── Error-rate detection ───────────────────────────────────────────────────
LOG_FILES = (
    f"{DATA_ROOT}/logs/server.err",
    f"{DATA_ROOT}/logs/mcp.err",
)
# NOT server.log / mcp.log. server.py's logging.basicConfig writes to stderr,
# which supervisord maps to *.err, so all application logs (INFO through
# CRITICAL, and every traceback) land there. server.log is uvicorn's access
# log and contains zero error-shaped lines — a watchdog grepping it would find
# nothing forever.
LOG_READ_CAP_BYTES = 4 * 1024 * 1024
CHRONIC_MIN_DISTINCT_HOURS = 3
# The chronic set is learned from a bounded backward scan and CACHED. Without
# an expiry it is learned exactly once, on the first boot after the state dir
# is created, and every steady-state error that starts happening afterwards is
# "novel" forever — so the next promotion is reverted for a recurring failure
# it did not cause. Re-learn daily.
CHRONIC_REFRESH_SECONDS = 24 * 3600.0
NOVEL_SIGNATURE_THRESHOLD = 5           # one novel signature this many times
NOVEL_FATAL_DISTINCT_THRESHOLD = 3      # distinct novel tracebacks
NOVEL_IN_CHANGED_PATH_THRESHOLD = 2     # novel + names a file the promo touched

# ── Rollback requests ──────────────────────────────────────────────────────
# A process inside the blast radius (the backend, the aggregator) cannot roll
# back inline — the rollback stops it partway through. It writes a request and
# the guardian performs it. Stale requests are discarded rather than obeyed:
# the state they were reasoning about is long gone.
ROLLBACK_REQUEST_MAX_AGE_SECONDS = 900.0

# ── Data-damage tripwire ───────────────────────────────────────────────────
# The one failure class `git reset --hard` structurally cannot undo: the KG
# and the vault are gitignored, so a change that deletes rows or notes boots
# fine, logs nothing, and survives the revert.
DATA_DROP_FRACTION = 0.05

#: Only the fallback branch of `kg_db_path` below uses this, and only because that
#: branch is the one where no resolver could be loaded at all. Everything else
#: that needs this layout reads `app.data_root.KG_DB_RELATIVE`.
_KG_DB_RELPATH = "_pipeline/vault-derived/kg.sqlite"


def _data_root_module(repo: str):
    """The stdlib-only data-root resolver, imported by file path, or None.

    By path and not by name for the reason `speak.py::_load_shaping_module`
    spells out: this file runs from the pinned snapshot under
    `~/.local/state/lloyd-guardian/bin` on `/usr/bin/python3`, where the repo is
    not on `sys.path`. The module's own header says nothing here may import from
    `app/`, and this is not an exception to that rule so much as the case the rule
    is written around: `app/data_root.py` is the stdlib-only half (#1415) with no
    import of its own and no venv behind it — `app.paths`, which does need one,
    stays out. Every failure shape (no repo, no module, a loader that refuses, a
    module that raises while executing) returns None rather than propagating,
    because an exception here is a watchdog that never starts.
    """
    path = __import__("pathlib").Path(repo) / "app" / "data_root.py"
    if not path.is_file():
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("lloyd_data_root", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:      # noqa: BLE001 - see the docstring: never propagate
        return None


def kg_db_path(repo: str = REPO, fallback_root: str = DATA_ROOT) -> str:
    """Where the knowledge-graph store the tripwire counts lives.

    Restating `_pipeline/vault-derived/kg.sqlite` against `DATA_ROOT` was the
    defect: the data root moved to `~/lloyd-data` on 2026-09-22, and a copy of a
    path whose root moved returns a clean-looking non-answer — here a missing file
    reads as an unreadable graph, which is what #1525 is about. The resolver is
    asked instead, so a `LLOYD_DATA` set for the watchdog is followed by both it
    and `app.paths`, and the layout is spelled in one place (`KG_DB_RELATIVE`).

    The fallback is that pre-#1525 literal, and it stays reachable on purpose: a
    watchdog that dies because its resolver is missing or refuses
    (`DataRootMissing` on a production root with no marker) is worse than one
    counting against a path it can name. `datawatch.py:133` sets the precedent
    for degrading rather than raising, and the caller says out loud which store it
    read.
    """
    resolver = _data_root_module(repo)
    if resolver is not None:
        try:
            return str(resolver.kg_store_for_tree(__import__("pathlib").Path(repo)))
        except Exception:  # noqa: BLE001 - DataRootMissing and anything like it
            pass
    return str(__import__("pathlib").Path(fallback_root) / _KG_DB_RELPATH)
KG_DB = kg_db_path()
VAULT_ROOT = "/home/alansrobotlab/obsidian"

# ── Vault tripwire (every tick, not only while observing) ──────────────────
# `vaultwatch.py` owns the thresholds; these name what the guardian acts on.
# Sync is stopped first because it is the one writer that turns a local wipe
# into a remote one — it did exactly that on 2026-09-10 and 2026-09-12.
OBSIDIAN_SYNC_PROGRAM = "agent-obsidian-sync"
WORKERS_PAUSE_URL = "http://127.0.0.1:8080/api/workers/pause"

# ── Paths ──────────────────────────────────────────────────────────────────
import os as _os
from pathlib import Path as _Path

AUTOMOD_STATE = _Path(
    _os.environ.get("LLOYD_AUTOMOD_STATE", _Path.home() / ".local/state/lloyd-automod")
)
GUARDIAN_STATE = _Path(
    _os.environ.get("LLOYD_GUARDIAN_STATE", _Path.home() / ".local/state/lloyd-guardian")
)
# Where `snapshot-data.sh` writes the hourly read-only snapshots of DATA_ROOT:
# the same resolution as the script's `${LLOYD_DATA_SNAPSHOTS:-$HOME/.lloyd-data-snapshots}`,
# because the alert that sends a person to look for a restore point has to name
# a directory that exists. The literal this replaces,
# `/home/.lloyd-data-snapshots`, has never existed on any machine (#1416).
DATA_SNAPSHOTS = _os.environ.get("LLOYD_DATA_SNAPSHOTS",
                                 str(_Path.home() / ".lloyd-data-snapshots"))
SUPERVISOR_SOCK = _os.environ.get("LLOYD_SUPERVISOR_SOCK", "/tmp/agent-supervisor.sock")
# Must exceed stopwaitsecs (15s) — a blocking stopProcess(wait=True) legitimately
# takes that long, and a shorter client timeout reports "error: timed out" for a
# stop that is actually working, so the rollback proceeds without knowing whether
# the writers are down. Observed exactly that during the 11:30 rollback.
SUPERVISOR_RPC_TIMEOUT = 45.0
SUPERVISORD_UNIT = "agent-supervisord.service"

# Suppress an identical alert title repeating inside this window.
ALERT_REPEAT_SECONDS = 900.0
# Speech repeats far more slowly than the toast. A toast you have already seen
# costs a glance; a sentence you have already heard costs the whole sentence,
# and the nag re-announces an unresolved BROKEN state every 15 minutes for as
# long as it lasts. At 900s that is four utterances an hour, indefinitely —
# which is how a useful alarm becomes one that gets muted for good.
VOICE_REPEAT_SECONDS = 3600.0

SELFTEST_INTERVAL_SECONDS = 24 * 3600.0
# A failed daily selftest is asked again this soon, not in a day. Its first run
# is the guardian's first tick, which on a cold boot is ~2 s after systemd
# starts it — before supervisord's socket exists or either service answers — so
# the three stack-dependent checks fail by construction. Every cold boot from
# 2026-09-15 to 09-23 paged "Guardian self-test failed" (needs_human, every
# channel) that way, and the heartbeat then published `selftest: false` for the
# next 24 h over a stack that had been healthy since minute one.
SELFTEST_RETRY_SECONDS = 120.0
# Inside this window after the guardian starts, a failed selftest is logged and
# retried, never alerted. Ten minutes covers a cold boot of the stack it probes
# (backend + aggregator, ~1 min) with room; a watchdog that truly cannot act is
# still reported ten minutes after boot rather than within seconds of it.
SELFTEST_BOOT_GRACE_SECONDS = 600.0
HEARTBEAT_NAME = "heartbeat.json"
