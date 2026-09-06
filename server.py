#!/usr/bin/env python3
"""Lloyd Mission Control Server — FastAPI app factory.

Creates the FastAPI app, mounts every router under app/routers/, wires the
autonomy startup ticker, and starts uvicorn. All business logic lives in app/.
"""

import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import CONFIG
from app.lifecycle import shutdown_cleanup

from app.routers import skills as _skills_router
from app.routers import memory as _memory_router
from app.routers import architecture as _architecture_router
from app.routers import entities as _entities_router
from app.routers import backlog as _backlog_router
from app.routers import services as _services_router
from app.routers import tools as _tools_router
from app.routers import autonomy as _autonomy_router
from app.routers import sessions as _sessions_router
from app.routers import models as _models_router
from app.routers import voice as _voice_router
from app.routers import messages as _messages_router
from app.routers import workers as _workers_router
from app.routers import inner_voice as _inner_voice_router
from app.routers import mc_ui as _mc_ui_router
from app.routers import system as _system_router
from app.routers import ide as _ide_router
from app.routers import lsp as _lsp_router
from app.routers import dashboard as _dashboard_router
from app.routers import health as _health_router
from app.routers import selfmod as _selfmod_router


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("lloyd-server")


REPO_ROOT = Path(__file__).resolve().parent
CLIENTS_JSON = REPO_ROOT / "agent-services" / "cert" / "clients.json"


def _load_allowlist() -> dict[str, str]:
    """Read clients.json and return {fingerprint_uppercase: name}.

    Re-read on every request so revocations take effect without a backend
    restart. This file is small (one entry per device) and the I/O is cheap.
    """
    try:
        data = json.loads(CLIENTS_JSON.read_text() or "{}")
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    out: dict[str, str] = {}
    for name, entry in data.items():
        fp = (entry.get("fingerprint") or "").upper().replace(":", "")
        if fp:
            out[fp] = name
    return out


app = FastAPI(title="Lloyd Mission Control")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _require_client_cert(request: Request, call_next):
    """Per-device client-cert allowlist on /api/* routes — now optional.

    mTLS was dropped 2026-06-14: iOS Chrome (and other third-party iOS
    browsers) can't present keychain identities for mutual TLS, so Vite no
    longer requests a client cert. Tailscale is the access boundary now —
    only tailnet devices can reach the frontend.

    We still honor the allowlist when a fingerprint IS forwarded (e.g. a
    browser with the Lloyd client cert installed, or if Vite mTLS is
    re-enabled), but no longer reject requests that lack one.
    """
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)

    # Loopback bypass: same-host services (LiveKit worker, autonomy ticker)
    # POST directly to 127.0.0.1:8080 without going through Vite's TLS layer.
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1"):
        return await call_next(request)

    fp = (request.headers.get("x-client-fingerprint") or "").upper().replace(":", "")
    if fp:
        # A cert was presented — keep enforcing the allowlist so revocation
        # still works for cert-bearing clients.
        allowlist = _load_allowlist()
        if fp not in allowlist:
            return JSONResponse(
                {"detail": "client cert revoked or unknown"},
                status_code=403,
            )
        # Stash the cert identity for handlers that want to log/use it.
        request.state.client_name = allowlist[fp]
        request.state.client_fingerprint = fp
    return await call_next(request)

app.include_router(_messages_router.router)
app.include_router(_sessions_router.router)
app.include_router(_models_router.router)
app.include_router(_autonomy_router.router)
app.include_router(_skills_router.router)
app.include_router(_memory_router.router)
app.include_router(_architecture_router.router)
app.include_router(_entities_router.router)
app.include_router(_backlog_router.router)
app.include_router(_services_router.router)
app.include_router(_tools_router.router)
app.include_router(_voice_router.router)
app.include_router(_workers_router.router)
app.include_router(_inner_voice_router.router)
app.include_router(_mc_ui_router.router)
app.include_router(_system_router.router)
app.include_router(_ide_router.router)
app.include_router(_lsp_router.router)
app.include_router(_dashboard_router.router)
app.include_router(_health_router.router)
app.include_router(_selfmod_router.router)

app.on_event("startup")(_autonomy_router.start_autonomy_ticker)
app.on_event("startup")(_workers_router.start_worker_pool)
app.on_event("shutdown")(shutdown_cleanup)


@app.on_event("startup")
async def _start_file_watcher() -> None:
    """Attach the running event loop to the IDE file watcher and rebind
    to whatever folder was open before the restart."""
    import asyncio
    from app import file_watcher, mc_state
    file_watcher.attach_loop(asyncio.get_running_loop())
    snap = mc_state.get_ide_snapshot() or {}
    folder = snap.get("open_folder")
    if folder:
        file_watcher.bind(folder)


@app.on_event("startup")
async def _sync_secondary_llm_state() -> None:
    """Reconcile the secondary vLLM supervisord process with config.yaml's
    `secondary_enabled` flag. Idempotent — safe to call on every boot."""
    import logging
    from app.config import CONFIG
    from app.supervisor_client import start_process, stop_process

    log = logging.getLogger("lloyd-server")
    # A canary boots from a worktree but the supervisord socket path is a
    # process-wide constant, so without this flag a gate run would reach the
    # LIVE supervisord and stop the live secondary vLLM. Canary configs set
    # `services.sync_secondary_llm: false`.
    if not (CONFIG.get("services") or {}).get("sync_secondary_llm", True):
        log.info("services.sync_secondary_llm=false → skipping secondary reconcile")
        return
    enabled = bool(CONFIG.get("secondary_enabled", False))
    proc = "agent-llm-secondary"
    if enabled:
        ok, msg = start_process(proc)
        log.info("secondary_enabled=true → start %s: %s (ok=%s)", proc, msg, ok)
    else:
        ok, msg = stop_process(proc)
        log.info("secondary_enabled=false → stop %s: %s (ok=%s)", proc, msg, ok)


@app.on_event("startup")
async def _mark_ready() -> None:
    """Flip /health from `starting` to `ok`.

    Registered after every other startup hook, so readiness means "the whole
    lifespan finished", not merely "the socket is bound". The canary boot
    probe and the post-restart verification both key off this.
    """
    _health_router.mark_startup_complete()


@app.on_event("shutdown")
async def _stop_file_watcher() -> None:
    from app import file_watcher
    file_watcher.shutdown()


if __name__ == "__main__":
    host = CONFIG.get("server", {}).get("host", "0.0.0.0")
    port = CONFIG.get("server", {}).get("port", 8080)
    uvicorn.run(app, host=host, port=port, log_level="info")
