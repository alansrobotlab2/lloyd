"""The retained desktop capture expires, and only this machine may feed it.

Backlog #1418. ``agent_mcp/desktop`` POSTs every capture to
``POST /api/desktop/state`` — the whole JPEG as ``image_b64`` plus up to 200 element
roles and names (``desktop.max_elements: 200``) — and ``app/routers/desktop.py``
kept that body in a module global with no expiry. ``GET /api/desktop/frame`` splatted the
global to whoever the peer gate let through, and the state POST accepted a frame
from any peer that gate trusted, so a capture of Alan's screen stayed readable
long after the session that took it had ended, and a peer that never captured
anything could put a frame on the Desktop tab.

The gate in front of these routes is ``server.ApiPeerGate``'s peer-address rule
(loopback, or ``server.trusted_networks``, default Tailscale's CGNAT range), the
backend binds ``0.0.0.0``, and ``agent-services/cert/clients.json`` enrols no
devices — so no read carried an identity and nothing bounded the retention. What
code can do is stop retaining the screen past the turn that captured it and stop
letting a non-local peer feed it. Gating the *reads* on a real device identity
stays a human action (enrol the devices); the live SSE stream therefore remains
reachable to a trusted peer, and the TTL is the mitigation, not a claim that the
read path is closed.

Two boundaries, both crossed rather than grepped.

1. **The peer boundary.** Production does not hand the router the socket peer:
   uvicorn fronts the app with ``ProxyHeadersMiddleware`` by default
   (``proxy_headers=True``, ``forwarded_allow_ips="127.0.0.1"``) and
   ``web/vite.config.ts`` proxies ``/api`` with ``xfwd: true``, so for a tab's
   request that middleware *replaces* ``scope["client"]`` with the browser's own
   address before the router reads it. ``_peer_is_loopback`` reads exactly that
   field, so the stack that computes it is in the test — the same
   ``uvicorn.config.Config(...).load()`` producer ``tests/test_api_client_gating.py``
   pins, and for the same reason: without it a test of a peer rule decides on a
   peer production never produced. The decisive pair is here: the aggregator's
   own loopback POST (no forwarded header, straight to ``127.0.0.1:8080``) must
   still be accepted, and the tailnet browser's rewritten peer must be refused —
   if the rewrite had ever resolved the publisher to something non-loopback the
   frame pipeline would be dead, and if a header could resolve a peer to
   loopback the gate would be a no-op.
2. **The model's read path.** ``mc_navigate(tab="desktop")`` puts
   ``latest_frame_summary()`` (``app/routers/mc_ui.py``) into the model's
   context, and that path is not gated the way ``desktop_*`` is. Every stale
   assertion here checks the summary as well as the route: an expiry honoured in
   one read path and not the other would leave the last window title readable
   after the frame itself was gone.

Refusals are asserted by *detail text*, not just status, so a 403 from
``ApiPeerGate`` or from request validation can never be mistaken for this
route's own refusal.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx
import pytest
from uvicorn.config import Config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CONFIG  # noqa: E402
from app.routers import desktop as desktop_router  # noqa: E402

FRAME_ROUTE = "/api/desktop/frame"
STATE_ROUTE = "/api/desktop/state"
LEASE_ROUTE = "/api/desktop/lease"

# The peers, named by who they are on this box.
LOOPBACK_HOST = "127.0.0.1"
# alansrobotlabs-imac on the tailnet: what Mission Control's browser presents
# after Vite's xfwd rewrite, and the peer ``ApiPeerGate`` *trusts* — the one a
# filesystem-less remote reader actually is.
TAILNET = "100.93.123.77"
# TEST-NET-2: routable-looking, trusted by nothing here.
PUBLIC = "198.51.100.7"

# What a stale answer must not carry: the JPEG, the element names, the rendered
# tree, and the window title the summary used to hand the model.
LEAKED_KEYS = ("image_b64", "elements", "summary", "window")

# This route's own refusal, and the peer gate's, kept apart for the asserts.
PUBLISHER_REFUSAL = "published by the local aggregator"
PEER_GATE_REFUSAL = "peer not permitted"


def frame(**over) -> dict:
    """A frame shaped like the one ``agent_mcp/desktop._push_frame`` builds."""
    f = {
        "tool": "desktop_capture", "ts": time.time(), "capture_id": "cap-1",
        "window": {"title": "Invoice — LibreOffice Calc", "class": "libreoffice",
                   "address": "0x1a", "workspace": "3", "focused": True},
        "width": 1350, "height": 1457, "scale": 0.729, "kind": "window",
        "mime": "image/jpeg",
        "image_b64": "QUJD" * 400,
        "elements": [{"index": 1, "role": "push button", "name": "Send",
                      "bounds": [4, 8, 92, 30]}],
        "summary": "the accessibility tree, rendered",
    }
    f.update(over)
    return f


@pytest.fixture(autouse=True)
def _mirror_starts_empty():
    desktop_router._latest = None
    desktop_router._subscribers.clear()
    yield
    desktop_router._latest = None
    desktop_router._subscribers.clear()


def _bare_app():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(desktop_router.router)
    return app


def _asgi_client(peer: str, port: int = 5555, app=None) -> httpx.AsyncClient:
    """A client whose *socket* peer is `peer`, with no lifespan.

    ``httpx.ASGITransport`` writes `peer` into the ASGI scope's ``client``, the
    same field uvicorn fills from the socket and the only field
    ``_peer_is_loopback`` reads.
    """
    transport = httpx.ASGITransport(app=app or _bare_app(), client=(peer, port))
    return httpx.AsyncClient(transport=transport, base_url="http://lloyd-test")


def _client(peer: str):
    from fastapi.testclient import TestClient

    return TestClient(_bare_app(), client=(peer, 5555))


@pytest.fixture
def local():
    with _client(LOOPBACK_HOST) as c:
        yield c


@pytest.fixture(scope="module")
def producer_stack():
    """The app exactly as uvicorn boots it, so the peer is the one production made.

    Same producer ``tests/test_api_client_gating.py`` uses: ``Config.load()``
    applies the middleware chain uvicorn would, and the two asserts here say out
    loud that it still has the shape this file depends on — if uvicorn ever
    stopped fronting the app with ``ProxyHeadersMiddleware``, or the default
    proxy trust moved off loopback, these tests report that instead of quietly
    deciding on a peer nobody produces.
    """
    import server

    cfg = Config(app=server.app, host="0.0.0.0", port=8080, log_config=None)
    cfg.load()
    assert cfg.proxy_headers is True, "uvicorn no longer applies proxy_headers by default"
    assert type(cfg.loaded_app).__name__ == "ProxyHeadersMiddleware", (
        f"the app is no longer fronted by the producer this file pins: "
        f"{type(cfg.loaded_app).__name__}")
    assert cfg.forwarded_allow_ips == LOOPBACK_HOST, (
        f"forwarded_allow_ips is {cfg.forwarded_allow_ips!r}; who counts as a "
        "trusted proxy is exactly what decides the publisher's loopback peer")
    # The router under test inside that stack is the module whose `_latest` the
    # asserts below read — a second import would make every assert here vacuous.
    assert server._desktop_router is desktop_router
    return cfg.loaded_app


# ── clause 1: the frame stops being readable once it is stale ──────────────

def test_a_frame_inside_the_window_is_served_exactly_as_before(local):
    """The fix must not cost the Desktop tab anything while a capture is live.

    The tab reads this route on mount (``web/src/.../DesktopPage.tsx``), so a
    fresh frame keeps carrying the image, the elements and the lease.
    """
    published = local.post(STATE_ROUTE, json=frame(ts=time.time()))
    assert published.json()["ok"] is True

    body = local.get(FRAME_ROUTE).json()
    assert body["active"] is True
    assert body["capture_id"] == "cap-1"
    assert body["image_b64"] == "QUJD" * 400
    assert len(body["elements"]) == 1
    assert body["window"]["title"] == "Invoice — LibreOffice Calc"
    assert "lease" in body
    assert desktop_router.latest_frame_summary()["active"] is True


STALE = [
    pytest.param(121.0, id="one_second_past_the_120s_default"),
    pytest.param(86_400.0, id="a_day_old"),
    pytest.param(None, id="no_ts_at_all"),
    pytest.param("yesterday", id="unparseable_ts"),
]


@pytest.mark.parametrize("age", STALE)
def test_a_stale_frame_is_served_as_if_nobody_had_captured(local, age):
    """Every stale shape answers ``active: false`` with the screen stripped.

    A missing or unparseable ``ts`` counts as stale, not fresh: the check fails
    closed, so an unreadable stamp is never evidence that the screen is current.
    """
    ts = time.time() - age if isinstance(age, float) else age
    assert local.post(STATE_ROUTE, json=frame(ts=ts)).json()["ok"] is True

    body = local.get(FRAME_ROUTE).json()
    assert body["active"] is False, body
    for key in LEAKED_KEYS:
        assert key not in body, f"{key} survived the TTL"
    assert "lease" in body, "the tab still needs the lease to render its toggle"

    summary = desktop_router.latest_frame_summary()
    assert summary["active"] is False, summary
    assert "Invoice" not in str(summary), "mc_navigate would still read the title"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_stamp_is_not_a_fresh_capture(local, literal):
    """A stamp that is not a real point in time is stale, never fresh.

    Crossed over HTTP, not on the accessor: ``json.loads`` accepts the ``NaN``,
    ``Infinity`` and ``-Infinity`` literals, so a POSTed body really can carry
    one — and a plain ``now - ts <= ttl`` subtraction would read ``Infinity`` as
    the freshest frame on the box. The raw body is the only way to send one, and
    the 200 below is the assertion that the store accepted it: if the literal
    were ever rejected at validation this request would never reach the frame
    route, and the test would pass without testing anything.
    """
    raw = '{"ts": ' + literal + ', "image_b64": "QUJD", "capture_id": "cap-1"}'
    posted = local.post(STATE_ROUTE, content=raw.encode(),
                        headers={"Content-Type": "application/json"})
    assert posted.status_code == 200, posted.text
    assert desktop_router._latest.get("capture_id") == "cap-1", "body not stored"
    assert local.get(FRAME_ROUTE).json()["active"] is False
    assert desktop_router.latest_frame_summary()["active"] is False


def test_the_ttl_is_a_config_value_so_no_config_yaml_edit_is_needed(local, monkeypatch):
    """``desktop.frame_ttl_seconds`` is read through CONFIG on every call.

    The clause's "no config.yaml edit" is the reason the value is looked up
    where it is used instead of baked in: the monkeypatched section below is the
    only change, and one frame's fate flips with it. An override that is not a
    usable positive number falls back to 120 rather than to no expiry.
    """
    assert desktop_router.DEFAULT_FRAME_TTL_SECONDS == 120.0
    monkeypatch.setitem(CONFIG, "desktop", {})
    assert desktop_router._frame_ttl_seconds() == 120.0

    old = frame(ts=time.time() - 600.0)
    monkeypatch.setitem(CONFIG, "desktop", {"frame_ttl_seconds": 3600})
    assert local.post(STATE_ROUTE, json=old).json()["ok"] is True
    assert local.get(FRAME_ROUTE).json()["active"] is True
    assert desktop_router.latest_frame_summary()["active"] is True

    monkeypatch.setitem(CONFIG, "desktop", {"frame_ttl_seconds": "soon"})
    assert desktop_router._frame_ttl_seconds() == 120.0
    assert local.get(FRAME_ROUTE).json()["active"] is False
    assert desktop_router.latest_frame_summary()["active"] is False


# ── clause 2: only the local publisher may feed the mirror ─────────────────

def test_a_remote_peer_is_refused_and_changes_nothing(local):
    """403 from a non-loopback peer, with the retained frame and every
    subscriber untouched — and the same body accepted from loopback.

    The refusal is asserted to carry *this* route's detail: a bare 403 would
    also be returned by request validation or by ``ApiPeerGate``, and the point
    of the clause is that the frame route itself says no.
    """
    good = frame(ts=time.time())
    assert local.post(STATE_ROUTE, json=good).json()["ok"] is True

    seat = asyncio.Queue(maxsize=16)
    desktop_router._subscribers.add(seat)
    try:
        forged = frame(ts=time.time(), capture_id="forged",
                       image_b64="Rk9SR0VE" * 64,
                       window={"title": "A window Alan never opened"})

        refused = _client(PUBLIC).post(STATE_ROUTE, json=forged)
        assert refused.status_code == 403, refused.text
        assert PUBLISHER_REFUSAL in refused.text, refused.text
        assert PEER_GATE_REFUSAL not in refused.text, refused.text
        assert desktop_router._latest == good, "the retained frame was replaced"
        assert seat.empty(), "a subscriber was pushed a frame nobody captured"
        assert local.get(FRAME_ROUTE).json()["capture_id"] == "cap-1"

        accepted = local.post(STATE_ROUTE, json=forged)
        assert accepted.json() == {"ok": True, "subscribers": 1}
        event, data = seat.get_nowait()
        assert event == "state" and data["capture_id"] == "forged"
    finally:
        desktop_router._subscribers.discard(seat)


def test_a_remote_peer_cannot_open_a_mirror_that_is_not_there_yet(local):
    """Refused before the first store, not after: an absent frame stays
    absent, so a peer cannot make the tab show a capture at all."""
    assert desktop_router._latest is None
    refused = _client(PUBLIC).post(STATE_ROUTE, json=frame(capture_id="forged"))
    assert refused.status_code == 403 and PUBLISHER_REFUSAL in refused.text
    assert desktop_router._latest is None
    assert local.get(FRAME_ROUTE).json()["active"] is False


def test_the_lease_route_still_answers_a_trusted_peer(local):
    """This round narrows the frame mirror, not the lease.

    The lease GET is what the Desktop tab reads from a browser over the tailnet,
    and gating reads on a device identity is the enrolled-devices decision the
    item leaves to a person — so it must keep answering a peer this route does
    not gate.
    """
    assert _client(TAILNET).get(LEASE_ROUTE).status_code == 200
    assert local.get(LEASE_ROUTE).status_code == 200


# ── seam: the producer that decides who the publisher is ───────────────────

class TestPublisherIdentityUnderTheProducer:
    """The peer this route decides on is the peer uvicorn computed.

    ``_push_frame`` posts from the MCP-server process straight to
    ``service_url("backend")`` — measured ``http://127.0.0.1:8080``, with no Vite
    in front and no forwarded header — while Mission Control's tab arrives as a
    loopback socket *plus* ``X-Forwarded-For: <browser>`` that uvicorn honours.
    Those two facts are what make a loopback rule both sufficient and safe; each
    test below is one way that composition could stop holding.
    """

    async def test_the_local_publishers_own_post_is_still_accepted(self, producer_stack):
        """No forwarded header, loopback socket: the frame is stored.

        This is the frame pipeline's happy path under the real stack. If
        uvicorn's rewrite, or a proxy in front of the aggregator, ever made this
        post look non-local, ``desktop_capture`` would silently stop feeding the
        Desktop tab — and only this test, run against the producer, would say so.
        """
        async with _asgi_client(LOOPBACK_HOST, app=producer_stack) as client:
            r = await client.post(STATE_ROUTE, json=frame(ts=time.time()))
            assert r.status_code == 200, r.text[:200]
            assert r.json() == {"ok": True, "subscribers": 0}
            served = (await client.get(FRAME_ROUTE)).json()
        assert served["active"] is True and served["capture_id"] == "cap-1"
        assert desktop_router.latest_frame_summary()["active"] is True

    async def test_the_browsers_rewritten_peer_is_refused_by_this_route(
            self, producer_stack):
        """The Desktop tab's own shape: loopback socket, tailnet browser behind
        Vite's ``xfwd``. Uvicorn rewrites the peer to the tailnet, and the frame
        POST is refused — by this route, not by the peer gate, which trusts that
        address and is the reason the tab keeps working.
        """
        async with _asgi_client(LOOPBACK_HOST, app=producer_stack) as client:
            seed = await client.post(STATE_ROUTE, json=frame(ts=time.time()))
            assert seed.json()["ok"] is True

        # Attached after the seed, so `seat.empty()` below can only mean one
        # thing: nothing was pushed by the refused request.
        seat = asyncio.Queue(maxsize=16)
        desktop_router._subscribers.add(seat)
        try:
            async with _asgi_client(LOOPBACK_HOST, app=producer_stack) as client:
                r = await client.post(
                    STATE_ROUTE, json=frame(capture_id="forged"),
                    headers={"X-Forwarded-For": TAILNET})
                assert r.status_code == 403, r.text[:200]
                assert PUBLISHER_REFUSAL in r.text, r.text
                assert PEER_GATE_REFUSAL not in r.text, (
                    "the peer gate answered, so this proves nothing about the route")
                assert (await client.get(FRAME_ROUTE)).json()["capture_id"] == "cap-1"
        finally:
            desktop_router._subscribers.discard(seat)
        assert desktop_router._latest["capture_id"] == "cap-1"
        assert seat.empty(), "a subscriber was pushed a frame nobody captured"

    async def test_a_trusted_peer_cannot_launder_itself_into_loopback(
            self, producer_stack):
        """A tailnet socket claiming ``X-Forwarded-For: 127.0.0.1`` is refused.

        ``forwarded_allow_ips`` is loopback only, so uvicorn leaves that header
        inert and the peer stays the tailnet — which is the whole reason this
        gate reads the scope's peer rather than a header. Read the header and
        this request would be the local publisher.
        """
        async with _asgi_client(TAILNET, app=producer_stack) as client:
            r = await client.post(
                STATE_ROUTE, json=frame(capture_id="forged"),
                headers={"X-Forwarded-For": LOOPBACK_HOST})
        assert r.status_code == 403, r.text[:200]
        assert PUBLISHER_REFUSAL in r.text, r.text
        assert desktop_router._latest is None

    async def test_a_forged_chain_naming_loopback_first_loses(
            self, producer_stack):
        """``X-Forwarded-For: 127.0.0.1, <tailnet>`` resolves to the tailnet.

        Each proxy appends, so uvicorn scans right-to-left for the first address
        it does not trust (`proxy_headers.py`); the client's own leftmost entry is
        consulted last. A gate that read the header, or took the chain's first
        entry, would call this request loopback and store the frame.
        """
        async with _asgi_client(LOOPBACK_HOST, app=producer_stack) as client:
            r = await client.post(
                STATE_ROUTE, json=frame(capture_id="forged"),
                headers={"X-Forwarded-For": f"{LOOPBACK_HOST}, {TAILNET}"})
        assert r.status_code == 403, r.text[:200]
        assert PUBLISHER_REFUSAL in r.text, r.text
        assert desktop_router._latest is None

    async def test_the_tab_still_gets_a_live_frame_over_the_producer(
            self, producer_stack):
        """Locking the publisher in must not lock the reader out.

        The tailnet browser's GET of the frame route answers with the frame the
        aggregator published — the acceptance's "without locking out the
        Mission Control Desktop tab", on the same stack rather than on a bare
        test app.
        """
        async with _asgi_client(LOOPBACK_HOST, app=producer_stack) as client:
            await client.post(STATE_ROUTE, json=frame(ts=time.time()))
        async with _asgi_client(TAILNET, app=producer_stack) as reader:
            body = (await reader.get(FRAME_ROUTE)).json()
        assert body["active"] is True, body
        assert body["image_b64"] == "QUJD" * 400
