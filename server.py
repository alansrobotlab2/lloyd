#!/usr/bin/env python3
"""Lloyd Mission Control Server — FastAPI app factory.

Creates the FastAPI app, mounts every router under app/routers/, wires the
autonomy startup ticker, and starts uvicorn. All business logic lives in app/.
"""

import ipaddress
import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
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
from app.routers import automod as _automod_router
from app.routers import browser as _browser_router
from app.routers import desktop as _desktop_router


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


# Tailscale's CGNAT range: every address on this tailnet sits inside it, and
# nothing else on this host does. That is what makes it a usable default for
# the trusted set — the tailnet was always meant to be the boundary (mTLS was
# dropped 2026-06-14 because iOS Chrome cannot present a keychain identity);
# what never happened was *checking* it on the backend. Widen it with
# `server.trusted_networks: [100.64.0.0/10, 192.168.50.0/24]` if a LAN-only
# device has to keep working.
DEFAULT_TRUSTED_NETWORKS = "100.64.0.0/10"

# The whole of the pre-auth surface, as exact paths rather than a prefix: a
# watchdog has to be able to ask "is it alive" from a peer the gate refuses,
# and `/health` (root-mounted, not `/api/health`) is what both the guardian and
# the promoter's idle gate poll. `/health/deep` is the gate-only sibling on the
# same handler. Anything under `/api/` — including a future `/api/health` — is
# gated by construction, never by an exemption list nobody keeps current.
PRE_AUTH_PATHS = frozenset({"/health", "/health/deep"})

_trusted_nets_cache: list | None = None


def _parse_networks(raw) -> list:
    """Parse a config value (comma-separated string or list) into networks.

    Returns [] when nothing usable was configured; the caller then falls back
    to the default rather than to an empty set, because an empty trusted set
    over a typo in config.yaml would lock every browser out of Mission Control.
    An unparseable entry is logged and dropped — it must not silently widen or
    silently vanish.
    """
    if isinstance(raw, str):
        items: list = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = []
    nets: list = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        try:
            nets.append(ipaddress.ip_network(text, strict=False))
        except ValueError:
            logger.error("server.trusted_networks: ignoring unparseable entry %r", text)
    return nets


def _trusted_networks() -> list:
    """The configured networks, parsed once and cached.

    Cached because the middleware asks on every request and `CONFIG` is
    boot-read-only by contract — unlike `clients.json`, which is re-read so a
    revocation lands without a restart, this value cannot change until reboot.
    """
    global _trusted_nets_cache
    if _trusted_nets_cache is None:
        raw = (CONFIG.get("server") or {}).get("trusted_networks")
        nets = _parse_networks(raw)
        if not nets:
            nets = _parse_networks(DEFAULT_TRUSTED_NETWORKS)
        _trusted_nets_cache = nets
    return _trusted_nets_cache


def _is_trusted_peer(host: str) -> bool:
    """Whether `host` — the ASGI peer address and nothing else — may call /api/*.

    Never a `Host`, `X-Forwarded-For` or `X-Real-IP` header. uvicorn runs
    `proxy_headers` by default with `forwarded_allow_ips="127.0.0.1"`, and
    `web/vite.config.ts` proxies `/api` with `xfwd: true`, so for a request
    arriving from Vite those headers *replace* `scope["client"]` with the
    browser's own address rather than evidencing it — `proxy_headers.py:32`
    honours them only from an address already in `forwarded_allow_ips`. Reading
    the header would both let any client name the peer it wanted and lock the
    UI out, because the real browser arrives as `100.93.123.77`, not as
    `127.0.0.1`. An unparseable address (starlette's literal `testclient`, an
    address that is simply malformed) is refused, not passed: fail closed means
    the case the code cannot read is a refusal.
    """
    text = (host or "").strip()
    if not text:
        return False
    if "%" in text:  # IPv6 link-local scope id, e.g. fe80::1%eth0
        text = text.split("%", 1)[0]
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return False
    if addr.is_loopback:
        return True
    return any(addr in net for net in _trusted_networks())


app = FastAPI(title="Lloyd Mission Control")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# The refusal, in one place: the HTTP body and the WebSocket close reason are
# the same claim, and they must not be able to drift apart.
REFUSAL_DETAIL = "peer not permitted: /api/* requires a loopback or trusted-network client"

# ASGI close code for a refused upgrade. 4000-4999 are reserved for libraries
# and applications (RFC 6455 §7.4.2), and 4403 mirrors the HTTP status the same
# decision produces, so a client that logs the code says the same thing a
# browser's 403 body would.
REFUSAL_CLOSE_CODE = 4403


class ApiPeerGate:
    """Fail closed on /api/*: the peer must be verified, not merely present.

    Until 2026-09-20 this middleware enforced the per-device allowlist **only
    when a `x-client-fingerprint` header arrived** and called `call_next` for
    everyone else — so it vetted the clients who volunteered identity and
    passed the ones who did not. `config.yaml` binds the backend on
    `0.0.0.0:8080`, and `curl http://192.168.50.108:8080/api/sessions` answered
    200 with real session data from the office LAN with no cert, no token and
    no tailnet. Backlog #683, measured there twice.

    The decision is the peer address: loopback (same-host daemons — the
    autonomy ticker, the LiveKit worker, the self-mod promoter) or a network in
    `server.trusted_networks`, default Tailscale's CGNAT range, which is what a
    tailnet browser presents after Vite's `xfwd` rewrite. See
    `_is_trusted_peer` for why a header can never be the evidence, and
    `PRE_AUTH_PATHS` for the two health probes that stay reachable.

    The allowlist still runs for cert-bearing clients, *after* the network
    rule, so revocation keeps working while a fingerprint copied out of
    `clients.json` — which is not proof of possession — buys nothing on its own.

    Why a pure ASGI middleware instead of `@app.middleware("http")`. That
    decorator wraps the handler in starlette's `BaseHTTPMiddleware`, whose
    `__call__` returns early for every scope that is not `http`
    (`starlette/middleware/base.py:101-104`) — an HTTP-only gate therefore
    passed WebSocket scopes through untouched. `app/routers/lsp.py:102` serves
    `@router.websocket("/api/lsp/{language}")` and each accepted connection
    spawns a language-server subprocess rooted at a caller-supplied `workspace`
    path, so leaving the upgrade path ungated would have kept a `/api/*`
    control reachable from exactly the peers this class exists to refuse
    (measured on #683's round: a LAN peer was accepted and asked to spawn while
    the HTTP sibling of the same route answered 403). One middleware, both
    scope types, one decision — and the pass-through to non-`/api/` scopes
    stays on the same one field.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket"):  # lifespan and anything else
            await self.app(scope, receive, send)
            return

        # A CORS preflight carries no identity by design — browsers send it
        # without custom headers — so refusing it would break a trusted
        # cross-origin client while proving nothing. Only a *real* preflight
        # gets the pass: a bare `OPTIONS /api/...`, which no browser sends, is
        # gated like any other request, because letting it through reaches the
        # router and hands an untrusted peer a 405-vs-403 oracle for which API
        # paths exist. `access-control-request-method` is the field that
        # distinguishes the two, and it is what starlette's own CORS middleware
        # keys on (`middlewares/cors.py`, `is_preflight`).
        if scope_type == "http" and scope.get("method") == "OPTIONS" \
                and b"access-control-request-method" in {
                    name for name, _ in scope.get("headers") or []}:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in PRE_AUTH_PATHS or not path.startswith("/api/"):
            await self.app(scope, receive, send)
            return

        # Peer address only — never a Host or forwarded header. Same-host
        # services (LiveKit worker, autonomy ticker) POST directly to
        # 127.0.0.1:8080 without Vite's TLS layer and are the loopback case.
        client = scope.get("client")
        client_host = client[0] if client else ""
        if not _is_trusted_peer(client_host):
            logger.warning("api-gate: refused %s %s from peer %r",
                           scope_type, path, client_host or "<unknown>")
            if scope_type == "websocket":
                # Closing before accepting is the ASGI rejection: uvicorn
                # answers the handshake with an HTTP 403 and the route — and so
                # the language-server spawn — is never entered.
                await send({"type": "websocket.close",
                            "code": REFUSAL_CLOSE_CODE,
                            "reason": REFUSAL_DETAIL})
                return
            await JSONResponse({"detail": REFUSAL_DETAIL},
                               status_code=403)(scope, receive, send)
            return

        fp = ""
        if scope_type == "http":
            fp = _cert_fingerprint(scope)
            if fp:
                allowlist = _load_allowlist()
                if fp not in allowlist:
                    # A cert was presented — keep enforcing the allowlist so
                    # revocation still works for cert-bearing clients.
                    await JSONResponse({"detail": "client cert revoked or unknown"},
                                       status_code=403)(scope, receive, send)
                    return
                # Stash the cert identity in the scope for handlers that want
                # to log/use it (`app/routers/system.py` reads it off
                # `request.state`, which is this dict).
                scope.setdefault("state", {})
                scope["state"]["client_name"] = allowlist[fp]
                scope["state"]["client_fingerprint"] = fp

        await self.app(scope, receive, send)


# Added after CORSMiddleware, and starlette's `add_middleware` inserts at the
# head of the user list (`starlette/applications.py:101`), so this gate is the
# outermost of the app's own middleware — only starlette's ServerErrorMiddleware
# is outside it — and it sees a request before CORS or any router can answer.
# That is the position `@app.middleware("http")` occupied before it was replaced
# (#683), so the ordering is unchanged by the swap.
app.add_middleware(ApiPeerGate)


def _cert_fingerprint(scope) -> str:
    """The normalised `x-client-fingerprint` from a raw ASGI header list.

    Same normalisation the old `Request`-based code did (upper-case, colons
    stripped) — `clients.json` stores `AA:BB:…` while Vite forwards the hex
    either way.
    """
    for name, value in scope.get("headers") or []:
        if name == b"x-client-fingerprint":
            return value.decode("latin-1").upper().replace(":", "")
    return ""

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
app.include_router(_automod_router.router)
app.include_router(_browser_router.router)
app.include_router(_desktop_router.router)

app.on_event("startup")(_autonomy_router.start_autonomy_ticker)
app.on_event("startup")(_workers_router.start_worker_pool)
app.on_event("shutdown")(shutdown_cleanup)

# The primary's KV usage and request count, sampled in the background. The
# pool's KV gate, the prefix-miss alert and the dashboard's KV p90 all read
# it, and with no sample yet each treats the engine as unpressured
# (app/engine_pressure.py), so its order against the pool does not matter.
from app import engine_pressure as _engine_pressure  # noqa: E402
app.on_event("startup")(_engine_pressure.start)
app.on_event("shutdown")(_engine_pressure.stop)


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
async def _sync_llm_slots() -> None:
    """Reconcile each optional LLM slot's supervisord process against its
    config.yaml flag. Idempotent — safe to call on every boot.

    One hook for both slots rather than one per slot, because the canary guard
    below, the "only one may run" rule and the start/stop call are identical
    for each and a second copy is how the two come to disagree.
    """
    import logging
    from app.config import CONFIG
    from app.supervisor_client import start_process, stop_process

    log = logging.getLogger("lloyd-server")
    # A canary boots from a worktree but the supervisord socket path is a
    # process-wide constant, so without this flag a gate run would reach the
    # LIVE supervisord and stop the live engines. Canary configs set
    # `services.sync_secondary_llm: false`. The key keeps its old name: it
    # has always meant "may this process drive supervisord's engine slots",
    # and renaming it would silently un-guard every canary already on disk.
    if not (CONFIG.get("services") or {}).get("sync_secondary_llm", True):
        log.info("services.sync_secondary_llm=false → skipping LLM slot reconcile")
        return

    from app import llm_slots

    wanted = llm_slots.enabled_slots(CONFIG)
    if len(wanted) > 1:
        # Both flags true is a config error, not a request. GPU 2 holds either
        # 21.7 GiB of llama.cpp or 17.6 GiB of DiffusionGemma plus its KV, so
        # starting the second one OOMs the card — and an OOM on this box has
        # twice cost the whole supervisord unit. Start neither and say so:
        # picking a winner here would look like the flag being ignored.
        log.error(
            "both %s are enabled and they share GPU 2; starting neither. "
            "Set exactly one to true.",
            " and ".join(flag for flag, _ in wanted),
        )
        for _flag, proc, _on in llm_slots.slots(CONFIG):
            ok, msg = stop_process(proc)
            log.error("conflict → stop %s: %s (ok=%s)", proc, msg, ok)
        return

    for flag, proc, on in llm_slots.slots(CONFIG):
        if on:
            ok, msg = start_process(proc)
            log.info("%s=true → start %s: %s (ok=%s)", flag, proc, msg, ok)
        else:
            ok, msg = stop_process(proc)
            log.info("%s=false → stop %s: %s (ok=%s)", flag, proc, msg, ok)


@app.on_event("startup")
async def _verify_model_identity() -> None:
    """Check each slot actually serves the model config claims.

    Runs detached: the sweep retries around a cold engine for minutes and
    readiness must not wait on it. Reports only — a mismatch is logged at
    ERROR and surfaced on GET /api/models/identity rather than acted on,
    because restarting a slot is exactly the operation that would have
    swapped the model in the first place.
    """
    import asyncio
    import logging
    from app.config import CONFIG

    log = logging.getLogger("lloyd-server")
    # Same guard as the secondary reconcile: a canary boots from a worktree
    # against the live ports and its verdict would be about someone else's
    # engines.
    if not (CONFIG.get("services") or {}).get("sync_secondary_llm", True):
        log.info("services.sync_secondary_llm=false → skipping model identity check")
        return

    from app import model_identity

    async def _sweep() -> None:
        try:
            await model_identity.verify_models_with_retry()
        except Exception as exc:
            log.warning("model identity check failed: %s", exc)

    asyncio.create_task(_sweep())


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
