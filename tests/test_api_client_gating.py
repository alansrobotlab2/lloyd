"""#683 — `/api/*` fails closed: the peer must be verified, not merely present.

Before this change `server._require_client_cert` enforced the per-device
allowlist **only when a `x-client-fingerprint` header was present** and
otherwise called `call_next`. A control that vets the clients who volunteer
identity and passes everyone else is a control that has been switched off, and
the backend binds `0.0.0.0:8080`, so the passing path was the whole LAN —
`curl http://192.168.50.108:8080/api/sessions` answered 200 with real session
data (measured at triage, 2026-09-20).

The replacement decision is taken from the ASGI peer address alone:
loopback (same-host daemons: the autonomy ticker, the LiveKit worker, the
promoter) or a configured trusted network (default Tailscale CGNAT
`100.64.0.0/10`, which is what a tailnet browser presents after Vite's `xfwd`
rewrite). Everything else is refused before the handler runs. The named
pre-auth exemption is exactly `/health` and `/health/deep` — exact paths, not a
prefix, so `/api/health` is not exempt (and is not a route either).

Two boundaries this file has to cross, not just one.

1. **The transport seam.** The gate is not the first thing a request meets:
   uvicorn wraps the app in `ProxyHeadersMiddleware` by default
   (`proxy_headers=True`, `forwarded_allow_ips` defaulting to `127.0.0.1`) and
   `web/vite.config.ts` proxies `/api` with `xfwd: true`, so for a request
   arriving from Vite that middleware *replaces* `scope["client"]` with the
   address in `X-Forwarded-For` before the gate reads it
   (`uvicorn/middleware/proxy_headers.py:58`). That is why the real browser
   arrives as its tailnet address `100.93.123.77` rather than as `127.0.0.1`,
   and why a gate that read the header itself would both forge the loopback
   bypass and lock Mission Control out. Triage measured the first half live: a
   loopback `POST /api/automod/drain` carrying `X-Forwarded-For: 203.0.113.9`
   got 403 where the same request without the header got 200. The
   `TestProducerComposesWithTheProxy` class drives the real uvicorn stack, so
   the peer the gate decides on is the peer the producer computed, including
   its right-to-left scan over a forged header chain.
2. **The scope-type seam.** `/api/*` is not HTTP-only:
   `app/routers/lsp.py:102` serves `@router.websocket("/api/lsp/{language}")`
   and every accepted connection spawns a language-server subprocess rooted at
   a caller-supplied `workspace`. An `@app.middleware("http")` gate cannot see
   those at all — starlette's `BaseHTTPMiddleware.__call__` returns early for
   any scope that is not `http` (`starlette/middleware/base.py:101-104`). The
   `TestWebSocketScope` class drives raw ASGI websocket scopes.

No lifespan runs, so the startup hooks stay off and every side-effecting
handler is replaced by a recorder. A request that was refused must leave the
recorder empty — asserting only the status code would also pass on a gate that
let the request through to a handler that then returned 403 for its own
reasons. Refusals are additionally asserted to carry the gate's own detail
text, so a route's 403 cannot be mistaken for this gate's.

Peers are the four the triage differential used: `127.0.0.1` (same-host
daemons), `100.93.123.77` (alansrobotlabs-imac on the tailnet, the real browser
address after the rewrite), `192.168.50.77` (office LAN, which is routable to
the box today), `203.0.113.9` (TEST-NET-3, a public address — nothing on this
host should ever accept it).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path

import httpx
import pytest
from uvicorn.config import Config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import server  # noqa: E402
from app.routers import automod as automod_router  # noqa: E402
from app.routers import backlog as backlog_router  # noqa: E402
from app.routers import services as services_router  # noqa: E402
from app.routers import workers as workers_router  # noqa: E402

LOOPBACK = "127.0.0.1"
TAILNET = "100.93.123.77"
LAN = "192.168.50.77"
PUBLIC = "203.0.113.9"

UNTRUSTED = [LAN, PUBLIC]
TRUSTED = [LOOPBACK, TAILNET]

# A `/api/*` path that is not a route: reaching it yields the router's 404, so
# it separates "the gate answered" from "something under the gate answered".
NOT_A_ROUTE = "/api/health"

# The five side-effecting routes named in the acceptance: pause the worker
# pool, run an autonomy task, restart a supervisor unit, arm the self-mod
# drain, write a backlog item. Each returns 200 from *any* peer before the
# gate, which is what the refusals below are differential against.
SIDE_EFFECTING = [
    pytest.param("POST", "/api/workers/pause", {"paused": True}, id="workers-pause"),
    pytest.param("POST", "/api/autonomy/run", {"task_id": 999999}, id="autonomy-run"),
    pytest.param("POST", "/api/services/action",
                 {"serviceId": "lloyd-backend", "action": "restart"},
                 id="services-restart"),
    pytest.param("POST", "/api/automod/drain", {"on": False}, id="automod-drain"),
    pytest.param("POST", "/api/backlog/task-create",
                 {"name": "gating probe", "board": "lloyd", "status": "draft"},
                 id="backlog-create"),
]

# Four of those five are routes a browser drives. `/api/automod/drain` (index
# 3) is not: it keeps its own loopback-only guard
# (`app/routers/automod.py::_is_loopback`, asserted present by
# `tests/test_automod_hardening.py`), because its only caller is the promoter on
# this box and a tab that could arm the drain is a tab that can refuse every
# user turn for up to 10 minutes. This item puts a network gate in front of that
# control; widening it is not this item's to do, so the tailnet half of clause 2
# is asserted over these four and drain's tailnet refusal is pinned to its own
# guard in `test_the_drain_refusal_from_the_tailnet_comes_from_its_own_guard`.
BROWSER_SIDE_EFFECTING = SIDE_EFFECTING[:3] + SIDE_EFFECTING[4:]

# The one `/api/*` websocket route (only `@router.websocket` in the app, at
# `app/routers/lsp.py:102`). Query carries a workspace because the handler
# spawns a language server against it; the recorder below replaces the spawn.
LSP_WS = "/api/lsp/python"


def _client(peer: str, app=None) -> httpx.AsyncClient:
    """An app with a chosen *socket* peer address and no lifespan.

    `httpx.ASGITransport` puts `peer` straight into the ASGI scope's `client`,
    which is the same field uvicorn fills from the socket — so this is the
    value a producer hands the middleware stack, with no real socket needed.
    """
    transport = httpx.ASGITransport(app=app or server.app, client=(peer, 5555))
    return httpx.AsyncClient(transport=transport, base_url="http://lloyd-test")


async def _ws_handshake(app, path: str, peer: str) -> tuple[list[str], list[dict]]:
    """Drive one websocket scope through `app` from socket `peer`.

    Returns the messages the app sent back and, for a refusal, nothing else
    ran; the caller asserts on the send sequence. `websocket.close` *before*
    `websocket.accept` is the ASGI rejection — uvicorn turns that into an HTTP
    403 on the handshake, and the route body never executes.
    """
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "scheme": "ws",
        "server": ("lloyd-test", 80),
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"workspace=%2Ftmp",
        "root_path": "",
        "subprotocols": [],
        "headers": [(b"host", b"lloyd-test"),
                    (b"connection", b"upgrade"),
                    (b"upgrade", b"websocket"),
                    (b"sec-websocket-key", b"cQ2lu6f8UV0eEhZVaWl0Lg=="),
                    (b"sec-websocket-version", b"13")],
        "client": (peer, 5555),
    }
    sent: list[dict] = []
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put({"type": "websocket.connect"})

    async def receive() -> dict:
        return await queue.get()

    async def send(message: dict) -> None:
        sent.append(message)
        if message["type"] == "websocket.accept":
            # Accepted: hang up so the handler's pumps end instead of blocking.
            await queue.put({"type": "websocket.disconnect", "code": 1000})

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return [m["type"] for m in sent], sent


class _FakePool:
    def __init__(self, calls: list) -> None:
        self._calls = calls
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self, paused: bool = True, owner: str = "operator") -> None:
        self._paused = paused
        self._calls.append(("workers/pause", paused))


@pytest.fixture
def reached(monkeypatch, tmp_path):
    """Neutralise every side effect and record what the handlers did.

    Empty `reached` after a request means the middleware refused it on the way
    in — the assertion that makes a pass mean "gated" rather than "answered".
    """
    calls: list = []

    monkeypatch.setattr(workers_router, "get_pool", lambda: _FakePool(calls))
    monkeypatch.setattr(services_router, "restart_process",
                        lambda name: (calls.append(("services/restart", name)) or (True, "simulated")))
    monkeypatch.setattr(services_router, "start_process",
                        lambda name: (calls.append(("services/start", name)) or (True, "simulated")))
    monkeypatch.setattr(services_router, "stop_process",
                        lambda name: (calls.append(("services/stop", name)) or (True, "simulated")))

    async def _fake_run_task(task_id):
        calls.append(("autonomy/run", task_id))
        return {"ok": True, "task_id": task_id}

    monkeypatch.setattr("autonomy.run_task", _fake_run_task)

    def _fake_set_drain(on, ttl_s=180.0):  # noqa: ARG001
        calls.append(("automod/drain", on))
        return 0.0

    monkeypatch.setattr(automod_router, "set_drain", _fake_set_drain)

    # Backlog create scans the whole board corpus for the board map and writes
    # into the vault; both are replaced so the test stays hermetic, and the
    # write itself is recorded rather than performed.
    monkeypatch.setattr(backlog_router, "_BACKLOG_DIR", tmp_path / "backlog")
    monkeypatch.setattr(backlog_router, "_backlog_board_map", lambda: {"lloyd": 1})

    def _record_write(filepath, fm, body):  # noqa: ANN001
        calls.append(("backlog/write", filepath.name))

    monkeypatch.setattr(backlog_router, "_write_task_file", _record_write)

    # The LSP websocket handler spawns a language server per connection rooted
    # at the query's workspace. Replaced with a recorder, so an accepted
    # connection proves the scope reached the route without launching pyright.
    async def _fake_spawn_ls(language, workspace):  # noqa: ARG001
        calls.append(("lsp/spawn", language))
        return None

    monkeypatch.setattr("app.routers.lsp._spawn_ls", _fake_spawn_ls)

    return calls


async def _hit(client, method, path, body, headers=None):
    if method == "POST":
        return await client.post(path, json=body, headers=headers)
    return await client.get(path, headers=headers)


def _detail_is_the_gate(response) -> str:
    """The refusal body text, so a caller can assert *this* gate said no."""
    return response.text


# ── clause 1: unverified peer is refused on the side-effecting surface ───────

@pytest.mark.parametrize("peer", UNTRUSTED, ids=["lan", "public"])
@pytest.mark.parametrize(("method", "path", "body"), SIDE_EFFECTING)
async def test_untrusted_peer_is_refused_before_the_handler_runs(
        reached, peer, method, path, body):
    """No credential, no fingerprint, peer outside loopback and the trusted
    networks: refused with 401/403, and the handler never ran."""
    async with _client(peer) as client:
        r = await _hit(client, method, path, body)
    assert r.status_code in (401, 403), (
        f"{method} {path} from {peer} returned {r.status_code}; "
        "an unverified peer must not reach a side-effecting route")
    assert server.REFUSAL_DETAIL in _detail_is_the_gate(r), (
        f"{method} {path} from {peer} answered {r.status_code} {r.text[:200]}; "
        "expected this gate's refusal detail, not a route's own 403")
    assert reached == [], f"{path} executed for untrusted peer {peer}: {reached}"


@pytest.mark.parametrize("peer", UNTRUSTED, ids=["lan", "public"])
async def test_untrusted_peer_is_refused_on_a_read_route(reached, peer):
    """The gate covers reads too — `/api/sessions` is what leaked real session
    data from the LAN address at triage."""
    async with _client(peer) as client:
        r = await client.get("/api/sessions")
    assert r.status_code in (401, 403)
    assert server.REFUSAL_DETAIL in _detail_is_the_gate(r), r.text[:200]
    assert reached == []


# ── clause 2: loopback and the tailnet keep working ─────────────────────────

@pytest.mark.parametrize("peer", TRUSTED, ids=["loopback", "tailnet"])
@pytest.mark.parametrize(("method", "path", "body"), BROWSER_SIDE_EFFECTING)
async def test_trusted_peers_still_reach_the_same_routes(
        reached, peer, method, path, body):
    """127.0.0.1 is the autonomy ticker and LiveKit worker's path;
    100.93.123.77 is what the tailnet browser presents after Vite's `xfwd`
    rewrite. Neither may see 401/403, and both must actually execute."""
    async with _client(peer) as client:
        r = await _hit(client, method, path, body)
    assert r.status_code not in (401, 403), (
        f"{method} {path} from trusted peer {peer} got {r.status_code}: {r.text[:200]}")
    assert r.status_code == 200, f"{path} from {peer}: {r.status_code} {r.text[:200]}"
    assert reached, f"{path} reported 200 from {peer} without executing anything"


async def test_the_drain_still_executes_for_its_only_real_caller(reached):
    """Loopback is the promoter's path and the gate must not touch it: a gate
    that refused the promoter's own drain call would stop the self-modification
    loop from landing anything at all."""
    async with _client(LOOPBACK) as client:
        r = await _hit(client, "POST", "/api/automod/drain", {"on": True})
    assert r.status_code == 200, r.text[:200]
    assert reached == [("automod/drain", True)], reached


async def test_the_drain_refusal_from_the_tailnet_comes_from_its_own_guard(reached):
    """A tailnet peer is refused on `/api/automod/drain` — but by the route's
    pre-existing loopback-only guard, not by this gate.

    The distinction is what clause 2 is actually about: the new middleware must
    not be the thing that breaks a working caller, and the drain was never
    reachable from off-loopback (`app/routers/automod.py::_is_loopback` refused
    it first, and `tests/test_automod_hardening.py` pins that guard). Relaxing
    that guard to "trusted network" would widen the most dangerous control in
    the app for the sake of a test expectation, so the refusal is kept and its
    *author* is pinned here: the middleware let the request through to the
    route, and the route said no in its own words.
    """
    async with _client(TAILNET) as client:
        r = await client.post("/api/automod/drain", json={"on": True})
    assert r.status_code == 403, r.text[:200]
    assert "loopback-only" in r.text, r.text[:200]
    assert server.REFUSAL_DETAIL not in r.text, r.text[:200]
    assert reached == [], f"drain executed anyway: {reached}"


# ── clause 3: the decision is the peer address, never a header ──────────────

async def test_x_forwarded_for_cannot_buy_the_loopback_bypass(reached):
    """A LAN peer claiming `X-Forwarded-For: 127.0.0.1` is still refused.

    What this pins is that the gate reads one field and no header. Under
    `httpx.ASGITransport` uvicorn's `proxy_headers` middleware never runs, so
    the header is inert here by construction — and inert in production too for
    this peer, since uvicorn only honours it from an address already in
    `forwarded_allow_ips`. It is the header that would matter to a gate written
    the obvious wrong way, which is the way it must not be written: triage saw
    `X-Forwarded-For: 203.0.113.9` flip a loopback `POST /api/automod/drain`
    from 200 to 403. `TestProducerComposesWithTheProxy` runs the same claim
    with the real producer in front of the gate.

    `workers/pause` carries the assertion rather than `automod/drain` because
    its only guard is this middleware — drain would answer 403 from its own
    loopback check and the test would pass at base for a reason it does not
    test.
    """
    async with _client(LAN) as client:
        r = await client.post("/api/workers/pause", json={"paused": True},
                              headers={"X-Forwarded-For": LOOPBACK})
    assert r.status_code in (401, 403), (
        f"peer {LAN} with X-Forwarded-For: 127.0.0.1 got {r.status_code}")
    assert server.REFUSAL_DETAIL in _detail_is_the_gate(r), r.text[:200]
    assert reached == []


@pytest.mark.parametrize("header_value", [TAILNET, LOOPBACK], ids=["tailnet", "loopback"])
async def test_a_header_claiming_a_forged_origin_does_not_evict_a_trusted_peer(
        reached, header_value):
    """The mirror image: a loopback peer carrying a header naming another
    address is still served, so the gate reads one field and nothing else.
    Same route as the test above — `workers/pause`, whose only guard is this
    middleware.

    Note the `127.0.0.1` case is not the same claim as the test above: with the
    real producer running, uvicorn would rewrite this peer to the header's
    value, and it happens to land on loopback, which is permitted either way.
    What this file asserts *with* the producer running is
    `test_a_forged_xff_prefix_loses_to_the_right_to_left_scan`.
    """
    async with _client(LOOPBACK) as client:
        r = await client.post("/api/workers/pause", json={"paused": True},
                              headers={"X-Forwarded-For": header_value})
    assert r.status_code == 200, r.text[:200]
    assert reached == [("workers/pause", True)]


# ── clause 4: the pre-auth exemption is the two named health probes ─────────

@pytest.mark.parametrize("path", ["/health", "/health/deep"], ids=["health", "health-deep"])
@pytest.mark.parametrize("peer", UNTRUSTED + TRUSTED,
                         ids=["lan", "public", "loopback", "tailnet"])
async def test_health_probes_stay_reachable_from_every_peer(path, peer):
    """A watchdog must be able to ask "is it alive" from anywhere. Both probes
    answer with the health payload from every peer — 200 once startup is
    complete, 503 while it is not (no lifespan runs here) — and never with the
    gate's 401/403. `commit` in the body is the positive control that the
    request reached the health handler rather than some other 2xx/5xx."""
    async with _client(peer) as client:
        r = await client.get(path)
    assert r.status_code not in (401, 403), (
        f"{path} from {peer} was gated: {r.status_code} {r.text[:200]}")
    assert server.REFUSAL_DETAIL not in r.text, r.text[:200]
    assert "commit" in r.json(), f"{path} did not reach the health handler: {r.text[:200]}"


async def test_the_exemption_is_not_a_prefix_match(reached):
    """`/api/health` is not a route and is not exempt — an untrusted peer gets
    the gate's refusal, not the 404 a pass-through would produce. `/health`
    being exempt cannot leak to any `/api/*` path however it is named.

    The 404 is what an untrusted peer *would* see if the exemption were written
    as a prefix match on `/health`, so the assertion is on the status *and* the
    author of the refusal.
    """
    async with _client(LAN) as client:
        r = await client.get(NOT_A_ROUTE)
    assert r.status_code in (401, 403), (
        f"{NOT_A_ROUTE} from {LAN} returned {r.status_code}; expected the gate's "
        "refusal rather than the route's 404")
    assert server.REFUSAL_DETAIL in _detail_is_the_gate(r), r.text[:200]
    assert reached == []


async def test_the_rest_of_the_api_stays_gated_from_an_untrusted_peer(reached):
    """The exemption is two exact paths; the read surface around it is not."""
    async with _client(LAN) as client:
        for path in ("/api/sessions", "/api/autonomy/tasks", "/api/backlog/tasks",
                     "/api/workers/queue", "/api/services"):
            r = await client.get(path)
            assert r.status_code in (401, 403), f"{path} from {LAN}: {r.status_code}"
            assert server.REFUSAL_DETAIL in _detail_is_the_gate(r), f"{path}: {r.text[:200]}"
    assert reached == []


async def test_a_cors_preflight_passes_and_a_bare_options_does_not(reached):
    """The CORS bypass admits a preflight, not every OPTIONS.

    Browsers send a preflight with no custom headers, so refusing one would
    break a trusted cross-origin client for no gain — but `OPTIONS` without
    `Access-Control-Request-Method` is something no browser sends, and passing
    it through reaches the router, which answers 405 for a real path and 404 for
    a made-up one. That is a path-existence oracle handed to exactly the peers
    this gate exists to refuse, on the one HTTP method the gate otherwise let
    through untouched.

    So both halves are pinned from the untrusted peer — the preflight is served
    (200, with the CORS answer), the bare OPTIONS is refused by the gate — and
    the trusted peer's bare OPTIONS is confirmed to still be the router's 405,
    so the narrowing cost a legitimate caller nothing it had before.
    """
    preflight = {"Origin": "https://goliath.test",
                 "Access-Control-Request-Method": "POST"}
    async with _client(LAN) as client:
        r = await client.request("OPTIONS", "/api/sessions", headers=preflight)
        assert r.status_code == 200, r.text[:200]
        assert r.headers.get("access-control-allow-methods"), (
            "the preflight was not answered by the CORS middleware")
        bare = await client.options("/api/sessions")
    assert bare.status_code in (401, 403), (
        f"bare OPTIONS from {LAN} returned {bare.status_code}; expected the "
        "gate's refusal rather than the router's answer")
    assert server.REFUSAL_DETAIL in _detail_is_the_gate(bare), bare.text[:200]
    assert reached == []
    async with _client(TAILNET) as client:
        trusted_bare = await client.options("/api/sessions")
    assert trusted_bare.status_code == 405, (
        f"bare OPTIONS from {TAILNET} returned {trusted_bare.status_code}; the "
        "router's 405 is its pre-existing answer and the gate must not change it")


# ── clause 5: the control that works today keeps working ────────────────────

@pytest.fixture
def one_device_allowlist(tmp_path, monkeypatch):
    """A `clients.json` with exactly one enrolled device."""
    cert_dir = tmp_path / "cert"
    cert_dir.mkdir()
    (cert_dir / "clients.json").write_text(json.dumps({
        "alansrobotlabs-imac": {"fingerprint": "AA:BB:CC:DD"},
    }))
    monkeypatch.setattr(server, "CLIENTS_JSON", cert_dir / "clients.json")
    return "AABBCCDD"


@pytest.mark.parametrize("peer", TRUSTED, ids=["loopback", "tailnet"])
async def test_a_known_fingerprint_from_a_trusted_network_is_still_served(
        reached, one_device_allowlist, peer):
    async with _client(peer) as client:
        r = await client.get("/api/sessions",
                             headers={"x-client-fingerprint": one_device_allowlist})
    assert r.status_code == 200, r.text[:200]


async def test_an_unknown_fingerprint_from_a_trusted_network_is_still_refused(
        reached, one_device_allowlist):
    """Revocation semantics, unchanged: a cert-bearing client whose
    fingerprint is absent from `clients.json` gets 403 with the same detail.
    No behavioural test covered this branch before #683. The detail is the cert
    branch's own, so the assertion also rules out the network rule firing."""
    async with _client(TAILNET) as client:
        r = await client.get("/api/sessions",
                             headers={"x-client-fingerprint": "DEADBEEF"})
    assert r.status_code == 403
    assert "client cert revoked or unknown" in r.json()["detail"]
    assert server.REFUSAL_DETAIL not in r.text
    assert reached == []


async def test_a_known_fingerprint_does_not_open_an_untrusted_network(
        reached, one_device_allowlist):
    """The allowlist is not proof of possession, so it cannot substitute for
    the network check: an enrolled fingerprint arriving from the LAN is still
    refused, on the network rule rather than the cert rule."""
    async with _client(LAN) as client:
        r = await client.get("/api/sessions",
                             headers={"x-client-fingerprint": one_device_allowlist})
    assert r.status_code in (401, 403)
    assert server.REFUSAL_DETAIL in _detail_is_the_gate(r), r.text[:200]
    assert reached == []


async def test_an_enrolled_device_is_still_named_to_the_route(reached,
                                                              one_device_allowlist):
    """The recognised name has to survive the gate to reach `request.state`.

    This is the half of the allowlist the *refusal* tests cannot see. The gate
    used to set `client_name` through starlette's `Request`; as a plain-ASGI
    middleware it writes `scope["state"]` and the route reads it back through
    `request.state`, and if the two disagreed no request would fail — `name`
    would simply come back null. Two live behaviours depend on it:
    `/api/system/identity` tells a device who the server thinks it is
    (`app/routers/system.py:79-84`), and `POST /api/clients/revoke` compares the
    caller against the entry it is about to delete and refuses to let a device
    revoke itself (`system.py:176-180`, "cannot revoke the cert you're currently
    using") — a null caller disables that guard silently, which is how a device
    ends up locking its own owner out of the allowlist it is trying to fix.

    Asserted through `/api/system/identity` because that route's whole response
    *is* the propagation, so there is no wrapper between the middleware's write
    and the read to take it on trust.
    """
    async with _client(TAILNET) as client:
        r = await client.get("/api/system/identity",
                             headers={"x-client-fingerprint": one_device_allowlist})
    assert r.status_code == 200, r.text[:200]
    assert r.json() == {
        "name": "alansrobotlabs-imac",
        # `clients.json` stores "AA:BB:CC:DD" and Vite forwards the hex either
        # way, so the gate normalises to upper-case with the separators stripped
        # (`server._cert_fingerprint`) and that is what a handler sees.
        "fingerprint": "AABBCCDD",
    }


# ── seam: the ASGI scope type — /api/* is not HTTP-only ─────────────────────

class TestWebSocketScope:
    """`/api/lsp/{language}` must fall to the same rule as its HTTP siblings.

    `app/routers/lsp.py:102` accepts a websocket on an `/api/*` path and spawns
    a language-server subprocess against a workspace path taken from the query
    string. The gate that replaced `_require_client_cert` used to be registered
    with `@app.middleware("http")`, which starlette wraps in
    `BaseHTTPMiddleware` — and that class returns before dispatching anything
    that is not an `http` scope (`starlette/middleware/base.py:101-104`). With
    the HTTP gate in place and the websocket path ungated, a LAN peer was
    accepted and asked to spawn while the same peer's `GET /api/sessions`
    answered 403 (measured on this round's first commit).
    """

    @pytest.mark.parametrize("peer", UNTRUSTED, ids=["lan", "public"])
    async def test_an_untrusted_peer_cannot_open_the_lsp_socket(self, reached, peer):
        types, sent = await _ws_handshake(server.app, LSP_WS, peer)
        assert types[:1] == ["websocket.close"], (
            f"{LSP_WS} from {peer} produced {types}; expected a close before any "
            "accept, which is the ASGI form of a refused handshake")
        assert "websocket.accept" not in types, f"{peer} was accepted: {sent}"
        assert sent[0]["code"] == server.REFUSAL_CLOSE_CODE, sent
        assert server.REFUSAL_DETAIL in sent[0]["reason"], sent
        assert reached == [], f"the LSP route ran for {peer}: {reached}"

    @pytest.mark.parametrize("peer", TRUSTED, ids=["loopback", "tailnet"])
    async def test_a_trusted_peer_still_opens_the_lsp_socket(self, reached, peer):
        types, sent = await _ws_handshake(server.app, LSP_WS, peer)
        assert types[:1] == ["websocket.accept"], (
            f"{LSP_WS} from trusted peer {peer} produced {types}: {sent}")
        assert reached == [("lsp/spawn", "python")], (
            f"accepted from {peer} but never reached the route: {reached}")

    async def test_the_refusal_closes_before_accept_rather_than_after(self, reached):
        """Ordering is the whole difference between "refused" and "accepted,
        then dropped": `websocket.close` before `websocket.accept` is what
        uvicorn answers with an HTTP 403, after it is a 101 followed by a
        close, which would have run the handler.
        """
        types, sent = await _ws_handshake(server.app, LSP_WS, LAN)
        assert types.index("websocket.close") < (
            types.index("websocket.accept") if "websocket.accept" in types else len(types))
        assert types == ["websocket.close"], sent
        assert reached == []


# ── seam: the transport producer — uvicorn's ProxyHeadersMiddleware ─────────

@pytest.fixture(scope="module")
def producer_stack():
    """The app exactly as uvicorn boots it, not as the test would like it.

    `server.py:362` calls `uvicorn.run(app, ...)`, and `uvicorn.config.Config`
    wraps the app in `ProxyHeadersMiddleware` whenever `proxy_headers` is true,
    which is the default, with `forwarded_allow_ips` defaulting to
    `127.0.0.1` (`uvicorn/config.py:207, 339-340, 476-477`). That middleware
    rewrites `scope["client"]` — the single field the gate is required to read
    — so a test of the gate that skips it is testing a stack production does not
    run. `Config.load()` builds the real one; the two asserts here say out loud
    that it still has the shape this file depends on, so if uvicorn ever stops
    applying it by default these tests report that instead of quietly passing
    against the bare app.
    """
    cfg = Config(app=server.app, host="0.0.0.0", port=8080, log_config=None)
    cfg.load()
    assert cfg.proxy_headers is True, "uvicorn no longer applies proxy_headers by default"
    assert type(cfg.loaded_app).__name__ == "ProxyHeadersMiddleware", (
        f"the app is no longer fronted by the producer this file pins: "
        f"{type(cfg.loaded_app).__name__}")
    return cfg.loaded_app


class TestProducerComposesWithTheGate:
    """The peer the gate decides on is the peer uvicorn computed.

    Vite's `/api` proxy runs with `xfwd: true`
    (`web/vite.config.ts:139-147`), so it hands the backend a loopback socket
    plus `X-Forwarded-For: <browser address>`; uvicorn honours that because
    loopback is in `forwarded_allow_ips`, and scans the list right-to-left for
    the first address it does not trust (`proxy_headers.py:131-146`,
    `_TrustedHosts.get_trusted_client_host`). Two consequences, one in each
    direction: the tailnet browser must be *allowed* although its socket is
    loopback-adjacent, and a client that prepends a trusted address to the
    chain must not get it.
    """

    async def test_the_tailnet_browser_path_through_the_producer_is_served(
            self, reached, producer_stack):
        """Socket 127.0.0.1 (Vite) + `X-Forwarded-For: 100.93.123.77` (Chrome on
        alansrobotlabs-imac) is the live Mission Control request. Rewritten peer
        is the tailnet, which is in the default trusted set.
        """
        async with _client(LOOPBACK, producer_stack) as client:
            r = await client.post("/api/workers/pause", json={"paused": True},
                                  headers={"X-Forwarded-For": TAILNET})
        assert r.status_code == 200, r.text[:200]
        assert server.REFUSAL_DETAIL not in r.text
        assert reached == [("workers/pause", True)]

    @pytest.mark.parametrize("xff", [LAN, PUBLIC], ids=["lan-browser", "public-browser"])
    async def test_a_non_trusted_browser_behind_the_same_producer_is_refused(
            self, reached, producer_stack, xff):
        """The same loopback socket, a different browser. This is the
        differential that a bare-app test cannot produce: without the producer
        the peer is `127.0.0.1` and the request is served, so only with
        uvicorn's rewrite in place does the gate see the LAN/public address —
        which is also what makes the gate's loopback bypass *not* a hole for a
        LAN-sourced browser session. Losing LAN browser sessions is needs-a-
        person clause 2 on the item.
        """
        async with _client(LOOPBACK, producer_stack) as client:
            r = await client.get("/api/sessions", headers={"X-Forwarded-For": xff})
        assert r.status_code in (401, 403), (
            f"/api/sessions via the producer with XFF {xff} returned "
            f"{r.status_code}; the gate must decide on the rewritten peer")
        assert server.REFUSAL_DETAIL in _detail_is_the_gate(r), r.text[:200]
        assert reached == []

    async def test_a_forged_xff_prefix_loses_to_the_right_to_left_scan(
            self, reached, producer_stack):
        """`X-Forwarded-For: 127.0.0.1, 192.168.50.77` resolves to the LAN.

        A chain is appended to by each proxy, so uvicorn scans it from the
        right and takes the first address it does not trust — the client's own
        entry is the leftmost and therefore the last thing consulted. If the
        scan ran left-to-right, or the gate read the header itself, this
        request would be a loopback peer and Mission Control's most
        side-effecting route would answer. `proxy_headers.py:131-136`.
        """
        async with _client(LOOPBACK, producer_stack) as client:
            r = await client.post("/api/workers/pause", json={"paused": True},
                                  headers={"X-Forwarded-For": f"{LOOPBACK}, {LAN}"})
        assert r.status_code in (401, 403), (
            f"a chain naming loopback first returned {r.status_code}; "
            "uvicorn's scan should have resolved the peer to the LAN address")
        assert server.REFUSAL_DETAIL in _detail_is_the_gate(r), r.text[:200]
        assert reached == []

    async def test_an_xff_from_an_untrusted_socket_is_ignored_entirely(
            self, reached, producer_stack):
        """A direct-from-LAN client cannot launder itself onto the tailnet:
        `forwarded_allow_ips` is loopback only, so the producer leaves the
        scope's peer at the socket address and the header is inert. This is
        clause 3's claim re-run with the producer actually present.
        """
        async with _client(LAN, producer_stack) as client:
            r = await client.post("/api/workers/pause", json={"paused": True},
                                  headers={"X-Forwarded-For": TAILNET})
        assert r.status_code in (401, 403), r.status_code
        assert server.REFUSAL_DETAIL in _detail_is_the_gate(r), r.text[:200]
        assert reached == []

    async def test_the_websocket_suffering_through_the_producer_is_the_same_decision(
            self, reached, producer_stack):
        """Both seams at once: the scope type is `websocket` and the peer is
        the producer's rewrite of a header chain. A tailnet browser's LSP
        connection through Vite must survive the two together, or the IDE tab's
        language features break even though every HTTP call works.
        """
        app = producer_stack
        # ProxyHeadersMiddleware passes `websocket` scopes through the same
        # client-rewrite branch, so the scope below arrives at the gate with
        # the tailnet peer the header names.
        scope_app = app

        async def drive(peer: str):
            return await _ws_handshake_with_xff(scope_app, LSP_WS, peer, TAILNET)

        types, sent = await drive(LOOPBACK)
        assert types[:1] == ["websocket.accept"], (
            f"tailnet browser LSP through the producer produced {types}: {sent}")
        assert reached == [("lsp/spawn", "python")], reached

    async def test_a_websocket_from_a_forged_producer_chain_is_refused(
            self, reached, producer_stack):
        """The refusal half of the same composition: LAN socket chain naming
        loopback first must not open the LSP socket, so no language server is
        spawned for a peer the HTTP gate would refuse.
        """
        types, sent = await _ws_handshake_with_xff(
            producer_stack, LSP_WS, LOOPBACK, f"{LOOPBACK}, {LAN}")
        assert types[:1] == ["websocket.close"], (
            f"forged chain produced {types}: {sent}")
        assert "websocket.accept" not in types, sent
        assert server.REFUSAL_DETAIL in sent[0]["reason"], sent
        assert reached == [], reached


async def _ws_handshake_with_xff(app, path: str, peer: str, xff: str):
    """`_ws_handshake` with an `X-Forwarded-For` header in the upgrade request.

    Vite's websocket proxy forwards the upgrade with `xfwd`, so this is the
    shape the backend actually sees for a proxied LSP connection.
    """
    sent: list[dict] = []
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put({"type": "websocket.connect"})
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "scheme": "ws",
        "server": ("lloyd-test", 80),
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"workspace=%2Ftmp",
        "root_path": "",
        "subprotocols": [],
        "headers": [(b"host", b"lloyd-test"),
                    (b"connection", b"upgrade"),
                    (b"upgrade", b"websocket"),
                    (b"x-forwarded-for", xff.encode()),
                    (b"sec-websocket-key", b"cQ2lu6f8UV0eEhZVaWl0Lg=="),
                    (b"sec-websocket-version", b"13")],
        "client": (peer, 5555),
    }

    async def receive() -> dict:
        return await queue.get()

    async def send(message: dict) -> None:
        sent.append(message)
        if message["type"] == "websocket.accept":
            await queue.put({"type": "websocket.disconnect", "code": 1000})

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return [m["type"] for m in sent], sent
