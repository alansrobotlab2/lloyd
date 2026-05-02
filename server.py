#!/usr/bin/env python3
"""Lloyd Mission Control Server — FastAPI app factory.

Applies the claude_agent_sdk parse_message monkeypatch, creates the app,
mounts every router under app/routers/, wires the autonomy startup ticker,
and starts uvicorn. All business logic lives in app/.
"""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.sdk_patch  # noqa: F401 — applies SDK parse_message monkeypatch on import
from app.config import CONFIG
from app.lifecycle import reap_orphaned_sdk_subprocesses, shutdown_cleanup

from app.routers import skills as _skills_router
from app.routers import memory as _memory_router
from app.routers import architecture as _architecture_router
from app.routers import entities as _entities_router
from app.routers import backlog as _backlog_router
from app.routers import services as _services_router
from app.routers import tools as _tools_router
from app.routers import pipelines as _pipelines_router
from app.routers import autonomy as _autonomy_router
from app.routers import sessions as _sessions_router
from app.routers import models as _models_router
from app.routers import usage as _usage_router
from app.routers import voice as _voice_router
from app.routers import messages as _messages_router
from app.routers import workers as _workers_router
from app.routers import inner_voice as _inner_voice_router


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("lloyd-server")


app = FastAPI(title="Lloyd Mission Control")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(_messages_router.router)
app.include_router(_sessions_router.router)
app.include_router(_models_router.router)
app.include_router(_usage_router.router)
app.include_router(_autonomy_router.router)
app.include_router(_skills_router.router)
app.include_router(_memory_router.router)
app.include_router(_architecture_router.router)
app.include_router(_entities_router.router)
app.include_router(_backlog_router.router)
app.include_router(_services_router.router)
app.include_router(_tools_router.router)
app.include_router(_pipelines_router.router)
app.include_router(_voice_router.router)
app.include_router(_workers_router.router)
app.include_router(_inner_voice_router.router)

app.on_event("startup")(reap_orphaned_sdk_subprocesses)
app.on_event("startup")(_autonomy_router.start_autonomy_ticker)
app.on_event("startup")(_workers_router.start_worker_pool)
app.on_event("shutdown")(shutdown_cleanup)


if __name__ == "__main__":
    host = CONFIG.get("server", {}).get("host", "0.0.0.0")
    port = CONFIG.get("server", {}).get("port", 8080)
    uvicorn.run(app, host=host, port=port, log_level="info")
