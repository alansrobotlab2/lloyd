"""Liveness and readiness for the backend process.

Two endpoints with deliberately different costs:

  * ``GET /health`` — pure in-memory, target < 5ms, safe to poll every few
    seconds. This is what `agent-services/guardian` watches and what the
    self-modification promoter uses as its idle gate.
  * ``GET /health/deep`` — adds an MCP probe and a git-dirty check, ~50ms.
    Gate-only; never polled.

Three design decisions worth not relitigating:

**Root path, not `/api/health`.** `server.py`'s auth middleware only inspects
paths under `/api/`, so a root-mounted `/health` stays reachable from a tailnet
device with no client fingerprint. It also mirrors the aggregator's existing
`/health` (`agent_mcp/main.py`), so a watchdog can treat both identically.

**`/health` must NOT probe MCP.** The backend recovers from an absent
aggregator lazily — the pool opens per turn and poisoned pools get evicted — so
a dead aggregator is not a sick backend. A watchdog polling every few seconds
would otherwise flap on every routine `lloyd-mcp` restart and could roll back
entirely unrelated code. The aggregator publishes its own `/health`; the
watchdog polls that directly and attributes failures correctly.

**`commit` is the load-bearing field.** It reports which commit *this running
process* booted from. `git reset --hard` proves the filesystem changed;
only `/health.commit == last_known_good` proves the service changed. Captured
once at import via `app.gitinfo` (no subprocess — see that module).
"""

from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import CONFIG, MODEL_CONFIGS, service_url
from app.gitinfo import head_branch, head_commit
from app.paths import LLOYD_HOME
from app.sessions_io import active_turn_summary

router = APIRouter()
logger = logging.getLogger("lloyd-health")


# Captured once, at import, for the lifetime of the process.
BOOT_ID = uuid.uuid4().hex
STARTED_AT = time.time()
BOOT_COMMIT = head_commit(LLOYD_HOME)
BOOT_BRANCH = head_branch(LLOYD_HOME)

# Routes that must be mounted for this process to be considered serving.
# An explicit list rather than a count threshold: a count is brittle (it moves
# every time anyone adds an endpoint) and it cannot say *which* router failed.
REQUIRED_ROUTES = frozenset({
    "/health",
    "/api/models",
    "/api/sessions",
    "/api/message/stream",
    "/api/services",
    "/api/tools",
})

# Set by the startup hook registered last in server.py. Distinguishes
# "socket bound, lifespan still running" from "ready" — which is exactly the
# difference the canary boot probe and the post-restart verification need.
_startup_complete = False


def mark_startup_complete() -> None:
    global _startup_complete
    _startup_complete = True
    logger.info("startup complete — /health will report ready")


def startup_complete() -> bool:
    return _startup_complete


def _missing_routes(request: Request) -> list[str]:
    try:
        mounted = {getattr(r, "path", None) for r in request.app.routes}
    except Exception:
        return sorted(REQUIRED_ROUTES)
    return sorted(REQUIRED_ROUTES - mounted)


def _health_payload(request: Request) -> tuple[dict, int]:
    checks_failed: list[str] = []

    config_loaded = bool(CONFIG) and bool(MODEL_CONFIGS)
    if not config_loaded:
        checks_failed.append("config_loaded")

    missing = _missing_routes(request)
    if missing:
        checks_failed.append("routers_mounted")

    if not _startup_complete:
        checks_failed.append("startup_complete")

    turns = active_turn_summary()
    workers_enabled = bool((CONFIG.get("workers") or {}).get("enabled", False))

    if not _startup_complete and not missing and config_loaded:
        status = "starting"
    elif checks_failed:
        status = "degraded"
    else:
        status = "ok"

    payload = {
        "status": status,
        "pid": os.getpid(),
        "boot_id": BOOT_ID,
        "commit": BOOT_COMMIT,
        "branch": BOOT_BRANCH,
        "started_at": STARTED_AT,
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "startup_complete": _startup_complete,
        "config": {
            "loaded": config_loaded,
            "models": len(MODEL_CONFIGS or {}),
            "overlay": os.environ.get("LLOYD_CONFIG_OVERLAY"),
        },
        "routers": {"required_present": not missing, "missing": missing},
        "turns": turns,
        "workers": {"enabled": workers_enabled},
        "checks_failed": checks_failed,
    }
    return payload, (200 if status == "ok" else 503)


@router.get("/health")
async def health(request: Request):
    """Cheap liveness. 200 when serving, 503 while starting or degraded."""
    payload, code = _health_payload(request)
    return JSONResponse(payload, status_code=code)


@router.get("/health/deep")
async def health_deep(request: Request):
    """Liveness plus an MCP probe and a git-dirty check. Gate-only, ~50ms."""
    payload, code = _health_payload(request)

    mcp_url = service_url("lloyd_mcp", "http://127.0.0.1:8500/mcp")
    health_url = mcp_url.rsplit("/mcp", 1)[0] + "/health"
    mcp_info: dict = {"url": mcp_url, "status": "unreachable",
                      "tools": 0, "degraded_modules": [], "latency_ms": None}
    try:
        import httpx
        started = time.time()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(health_url)
        mcp_info["latency_ms"] = round((time.time() - started) * 1000, 2)
        body = resp.json()
        mcp_info["status"] = body.get("status", "unknown")
        mcp_info["tools"] = body.get("tools", 0)
        mcp_info["degraded_modules"] = body.get("degraded_modules", [])
        mcp_info["per_module"] = {
            name: entry.get("tools", 0)
            for name, entry in (body.get("discovery") or {}).items()
        }
    except Exception as exc:
        mcp_info["error"] = str(exc)[:200]

    if mcp_info["status"] != "ok":
        payload["checks_failed"] = list(payload["checks_failed"]) + ["mcp"]
        code = 503

    dirty = None
    try:
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(LLOYD_HOME), "status", "--porcelain"],
            capture_output=True, text=True, timeout=3.0,
        )
        dirty = bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        dirty = None

    payload["mcp"] = mcp_info
    payload["git"] = {"dirty": dirty, "commit": BOOT_COMMIT, "branch": BOOT_BRANCH}
    if payload["checks_failed"]:
        payload["status"] = "degraded"
    return JSONResponse(payload, status_code=code)
