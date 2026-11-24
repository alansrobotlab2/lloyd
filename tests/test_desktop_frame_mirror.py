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
backend binds ``0.0.0.0``, and ``agent-services/cert/clients.json`` is ``{}`` —
so no read carried an identity and nothing bounded the retention. What code
could do, it did: stop retaining the screen past the turn that captured it, and
stop letting a non-local peer feed it. Gating the *reads* on a real device
identity is not owed — #683 ruled "no per-device token for now" on 2026-09-27,
by which point mutual TLS had already been dropped from the Vite dev server on
2026-06-14 — so no read path can carry one, and the peer-address gate is the
designed boundary rather than a placeholder for a stricter one. The live SSE
stream therefore stays reachable to a trusted peer, and
``desktop.frame_ttl_seconds`` plus the loopback-only publisher are the
mitigation, not a claim that the read path is closed. A tab that is already open
keeps the last frame it was pushed until it reloads, by decision.

Three boundaries, each crossed rather than grepped.

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
3. **The grant peer boundary** (backlog #4533). ``op: grant`` on
   ``/api/desktop/lease`` lends Alan's keyboard and mouse for up to
   ``desktop_lease.MAX_TTL_MINUTES``, and until this round the only thing in
   front of it was the address rule that also admits the tailnet — while the
   frame *publish* beside it demanded the seat. ``TestGrantPeerBoundary``
   crosses it the same way as boundary one: a tailnet peer's grant is refused by
   this route before the lease file, the toast or the pushed event; the seat's
   own POST still grants, which is what Vite's rewrite leaves a tab at
   ``https://localhost:5173`` (``127.0.0.1``); and ``op: revoke`` from that same
   tailnet peer still revokes. That asymmetry is the decision, not an oversight:
   a device across the house is how the seat gets taken back.

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

    The lease GET is what the Desktop tab reads from a browser over the tailnet.
    Its boundary is the peer address ``server.ApiPeerGate`` decides on (loopback,
    or ``server.trusted_networks``), not a device identity: #683 ruled "no
    per-device token for now" on 2026-09-27, so there is no stricter gate this
    route is waiting on. It must keep answering a peer the gate trusts.
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


# ── who may grant the lease (backlog #4533) ───────────────────────────────
#
# The frame publish two sections above has refused a non-loopback peer since
# #1418; the grant beside it did not, so the write that lends the keyboard and
# mouse for up to `desktop_lease.MAX_TTL_MINUTES` was reachable by any peer
# ApiPeerGate trusts — the whole tailnet by default, because config.yaml never
# sets server.trusted_networks (server.py's DEFAULT_TRUSTED_NETWORKS is
# 100.64.0.0/10). What follows is the corrected half of that asymmetry.

#: This route's own refusal of a grant, kept apart from ``PEER_GATE_REFUSAL``
#: for the same reason ``PUBLISHER_REFUSAL`` is: an assert has to be able to
#: tell "this route decided" from "the network gate decided", and under the
#: producer stack a tailnet peer passes the gate, so only the route can be what
#: refused it.
GRANT_REFUSAL = "can only be granted from this machine"


@pytest.fixture
def seat(monkeypatch, tmp_path):
    """The lease file on a temp path, and the seat's toast recorded rather than run.

    Every grant node below would otherwise write the real
    ``~/lloyd-data/desktop/lease.json`` and pop a real ``hyprctl notify`` on
    Alan's desktop from inside a test run. ``_toast`` is the router's only
    compositor call, so replacing it both keeps the seat out of the suite and
    *is* the instrument for "no notify ran" — and
    ``test_the_same_trusted_peer_keeps_the_revoke_kill_switch`` asserts it
    records a real toast, so an empty list elsewhere is a refusal, not a
    broken spy.
    """
    from app import desktop_lease

    monkeypatch.setattr(desktop_lease, "DESKTOP_LEASE_PATH", tmp_path / "lease.json")
    toasts: list[str] = []
    monkeypatch.setattr(desktop_router, "_toast", toasts.append)
    return desktop_lease, toasts


class TestGrantPeerBoundary:
    """A grant is a seat decision; a revoke is available from anywhere trusted.

    Clauses 2 and 4 of #4533 name ``tests/test_desktop_module.py``, and their
    nodes are here instead: that file's skips are in the machine-fact population
    of the before-inventory, and
    ``tests/test_skip_inventory.py::test_no_machine_fact_skip_was_deleted_to_buy_headroom``
    refuses a diff that edits any file in it, whatever the edit is for. The
    library-level half of clause 4 — the in-process pin still working — is pinned
    there by ``test_a_grant_pinned_to_a_session_covers_only_it``, untouched by
    this round. What is new here is the route's side of both clauses, which
    belongs beside the other lease-route nodes in this file anyway.
    """

    def test_a_trusted_peer_is_refused_a_grant_before_it_changes_anything(
            self, seat):
        """#4533 clause 1: the 403 has nothing behind it.

        Alan lent the seat by hand first, so the lease file exists and is live:
        a refusal that still wrote would be visible as a bumped ``epoch``, and
        one that toasted or pushed would be visible to the seat and to any
        open Desktop tab. The peer is ``TAILNET``, which ``ApiPeerGate`` trusts
        — under the bare app here the gate is not even in the stack, and the
        assert on ``PEER_GATE_REFUSAL`` is what proves the refusal below is the
        route's and not a gate that happened to be standing in the way.
        """
        desktop_lease, toasts = seat
        desktop_lease.grant(30)
        before = desktop_lease.read()
        assert before["holder"] == "agent", before

        pushed: asyncio.Queue = asyncio.Queue(maxsize=8)
        desktop_router._subscribers.add(pushed)
        refused = _client(TAILNET).post(
            LEASE_ROUTE, json={"op": "grant", "minutes": 240})
        assert refused.status_code == 403, refused.text
        assert GRANT_REFUSAL in refused.text, refused.text
        assert PEER_GATE_REFUSAL not in refused.text, (
            "the peer gate refused this, so the route's own check is untested")
        after = desktop_lease.read()
        assert after["holder"] == before["holder"], after
        assert after["epoch"] == before["epoch"], (
            "the grant ran and the 403 lied: the epoch moved")
        assert after["expires_at"] == before["expires_at"], after
        assert toasts == [], f"the refusal toasted the seat: {toasts}"
        assert pushed.empty(), "a Desktop tab was pushed a lease nobody granted"

    def test_an_untrusted_peer_is_refused_the_same_way_by_this_route(self, seat):
        """The refusal is the loopback test, not a favour from the address rule.

        ``PUBLIC`` is trusted by nothing, so a 403 there alone would be
        ambiguous; what matters is that the detail text is *this* route's
        refusal, which is only reachable if the peer got past the gate — and in
        the bare app here there is no gate, so both peers reach the same line.
        """
        desktop_lease, toasts = seat
        before = desktop_lease.read()
        refused = _client(PUBLIC).post(
            LEASE_ROUTE, json={"op": "grant", "minutes": 240})
        assert refused.status_code == 403, refused.text
        assert GRANT_REFUSAL in refused.text, refused.text
        assert desktop_lease.read()["epoch"] == before["epoch"]
        assert toasts == []

    def test_the_same_trusted_peer_keeps_the_revoke_kill_switch(self, seat):
        """#4533 clause 3: narrowing the grant must not cost the kill switch.

        A tailnet device is how Alan cuts Lloyd off from the other side of the
        house, so ``op: revoke`` from ``TAILNET`` still revokes, still bumps the
        epoch, and still toasts — that last assert is the positive control that
        makes the empty ``toasts`` in the refusal nodes a fact about the refusal
        rather than about a spy that never records anything.
        """
        desktop_lease, toasts = seat
        lent = desktop_lease.grant(30)
        revoked = _client(TAILNET).post(LEASE_ROUTE, json={"op": "revoke"})
        assert revoked.status_code == 200, revoked.text
        body = revoked.json()
        assert body["holder"] == "human", body
        assert body["epoch"] == lent["epoch"] + 1, body
        assert not desktop_lease.agent_may_act()[0]
        assert toasts == ["Lloyd's desktop lease revoked"], toasts

    def test_a_seat_grant_still_lends_the_lease(self, seat):
        """The seat keeps working, over the stack that computes its peer.

        A tab at ``https://localhost:5173`` is on this box, so Vite's ``xfwd``
        rewrite hands the backend a ``127.0.0.1`` peer (``forwarded_allow_ips``
        is loopback, pinned in ``test_the_composition_under_test_is_the_right_one``),
        and the grant behind it is unchanged: the agent holds the lease and
        ``agent_may_act`` says yes.
        """
        desktop_lease, toasts = seat
        granted = _client(LOOPBACK_HOST).post(
            LEASE_ROUTE, json={"op": "grant", "minutes": 30})
        assert granted.status_code == 200, granted.text
        assert granted.json()["holder"] == "agent", granted.text
        assert desktop_lease.agent_may_act()[0]
        assert toasts == [
            "Lloyd may use the desktop for 30 min — move the mouse to take it back"], toasts

    def test_a_seat_grant_beyond_the_cap_still_lends_and_still_clamps(self, seat):
        """#4533 clause 2, on the arm the gate had to leave alone.

        A tab asking for more than ``MAX_TTL_MINUTES`` is the interesting case
        because three things have to be true of it at once, none of them changed
        by #4533: the agent holds the lease, the epoch moved by exactly one (a
        grant that silently did nothing would still answer 200), and the expiry
        is the clamp rather than the number in the body. ``remaining_s`` is read
        off the response, so this is the seat's own view and not a second copy of
        the constant.
        """
        desktop_lease, toasts = seat
        before = desktop_lease.read()
        granted = _client(LOOPBACK_HOST).post(
            LEASE_ROUTE, json={"op": "grant", "minutes": 10_000})
        assert granted.status_code == 200, granted.text
        view = granted.json()
        assert view["holder"] == "agent", view
        assert view["epoch"] == before["epoch"] + 1, view
        assert 0 < view["remaining_s"] <= desktop_lease.MAX_TTL_MINUTES * 60, view
        # One toast, and the grant's. Prefix-matched on purpose: the line quotes
        # the *requested* minutes over a lease that expires at the clamp, so
        # pinning it whole would pin a wrong number — see the item's findings.
        assert len(toasts) == 1 and toasts[0].startswith(
            "Lloyd may use the desktop"), toasts

    def test_a_grant_body_naming_a_session_stores_no_pin(self, seat):
        """#4533 clause 4: the pin is not the caller's to choose.

        ``session_id`` in the body used to be written straight into the lease,
        and a pinned lease covers only the session it names — so a caller this
        route cannot identify got to decide which of Alan's sessions could drive
        his desktop, in either direction. The route ignores the field now: the
        stored pin is empty, and a session nobody named still holds, which is the
        unpinned behaviour every real grant has had anyway since
        ``web/src/api.ts`` only ever posts ``{op, minutes}``.
        """
        desktop_lease, _toasts = seat
        granted = _client(LOOPBACK_HOST).post(
            LEASE_ROUTE, json={"op": "grant", "minutes": 30,
                               "session_id": "someone_elses_session"})
        assert granted.status_code == 200, granted.text
        assert desktop_lease.read()["session_id"] == "", desktop_lease.read()
        assert desktop_lease.agent_may_act("any_other_session")[0], (
            "an unnamed session is locked out, so the pin was honoured after all")

    async def test_a_seat_browsers_grant_arrives_on_loopback_and_grants(self, seat,
                                                                        producer_stack):
        """Boundary three, crossed through the producer that rewrites the peer.

        Exactly one thing differs between this node and the next: the address
        ``ProxyHeadersMiddleware`` resolves out of ``X-Forwarded-For``. This is
        the seat browser — a loopback socket and a loopback forwarded address,
        which is what a tab at ``https://localhost:5173`` presents once Vite's
        ``xfwd`` has had it — and it still gets the lease.
        """
        desktop_lease, _toasts = seat
        async with _asgi_client(LOOPBACK_HOST, app=producer_stack) as client:
            r = await client.post(
                LEASE_ROUTE, json={"op": "grant", "minutes": 30},
                headers={"X-Forwarded-For": LOOPBACK_HOST})
            assert r.status_code == 200, r.text[:200]
            assert r.json()["holder"] == "agent", r.text[:200]
        assert desktop_lease.agent_may_act()[0]

    async def test_a_tailnet_browsers_grant_arrives_as_its_own_address_and_is_refused(
            self, seat, producer_stack):
        """The other half of the pair, and the reason no frontend change is owed.

        The same POST from a browser on the imac is rewritten to that browser's
        tailnet address, which the gate trusts and this route does not: 403 with
        the route's own detail, and no lease written. That is the contingency the
        item had to measure before landing the line — a loopback test on the
        grant really does refuse a tailnet tab, which is the point, because only
        the machine itself hands over its own keyboard.
        """
        desktop_lease, toasts = seat
        before = desktop_lease.read()
        async with _asgi_client(LOOPBACK_HOST, app=producer_stack) as client:
            r = await client.post(
                LEASE_ROUTE, json={"op": "grant", "minutes": 240},
                headers={"X-Forwarded-For": TAILNET})
            assert r.status_code == 403, r.text[:200]
            assert GRANT_REFUSAL in r.text, r.text[:200]
            assert PEER_GATE_REFUSAL not in r.text, (
                "the peer gate answered, so this proves nothing about the route")
            assert desktop_lease.read()["epoch"] == before["epoch"], r.text[:200]
        assert toasts == []

    async def test_a_tailnet_peer_cannot_launder_itself_into_a_seat_grant(self, seat,
                                                                         producer_stack):
        """A peer the socket names as tailnet cannot claim loopback in a header.

        ``forwarded_allow_ips`` is loopback only, so the header stays inert and
        the peer remains the tailnet — the same shape as
        ``test_a_trusted_peer_cannot_launder_itself_into_loopback`` on the frame
        publish, now on the write that lends the keyboard instead of the one that
        shows a picture.
        """
        desktop_lease, toasts = seat
        before = desktop_lease.read()
        async with _asgi_client(TAILNET, app=producer_stack) as client:
            r = await client.post(
                LEASE_ROUTE, json={"op": "grant", "minutes": 240},
                headers={"X-Forwarded-For": LOOPBACK_HOST})
            assert r.status_code == 403, r.text[:200]
            assert GRANT_REFUSAL in r.text, r.text[:200]
            assert desktop_lease.read()["epoch"] == before["epoch"], r.text[:200]
        assert toasts == []


# ── the prose that describes the boundary ─────────────────────────────────
#
# Backlog #1671. #1418 shipped its access boundary described as unfinished
# business: this file and ``app/routers/desktop.py`` both told the next reader
# that gating the reads on a real device identity was still owed. The ruling
# that settled the question is #683 (2026-09-27: no per-device token for now),
# and because the sentence outlived the ruling, an owed-human sweep read the
# sentence instead of the ruling and regenerated the action. Prose is the only
# surface that claim can live on, so the pin below is prose too — over the same
# extent the sweep greps, with a positive control beside it.

#: The declined recommendation, assembled rather than spelled out. The check is
#: a case-insensitive substring over the *whole* source of both files, so a
#: literal here would put the token back into one of the two files it measures.
DECLINED_DEVICE_GATE = "en" + "rol"

#: What each of the two module docstrings must now carry: the gate and the
#: network rule that together are the boundary, and both halves of the
#: mitigation that actually exists.
BOUNDARY_TOKENS = ("ApiPeerGate", "server.trusted_networks", "frame_ttl_seconds",
                   "loopback", "publisher")

DESKTOP_ROUTER_SRC = ROOT / "app" / "routers" / "desktop.py"
THIS_TEST_SRC = Path(__file__).resolve()


def test_the_desktop_docs_state_the_peer_boundary_ruling_out_a_device_gate():
    """#1671: the boundary these routes have is the one that is built.

    The declined sentence rested on a premise that no longer holds, measured
    rather than remembered: the Vite dev server dropped mutual TLS on
    2026-06-14 (the ``mTLS dropped 2026-06-14`` comment over ``httpsConfig`` in
    ``web/vite.config.ts``), so a tab's request carries no
    client certificate and the still-wired ``clientCertHeaders()`` there
    injects no ``x-client-fingerprint``; ``agent-services/cert/clients.json``
    is ``{}`` at 3 bytes; and ``ApiPeerGate`` decides on the peer address,
    running the allowlist only after the network rule and only for
    cert-bearing peers, so its own docstring says a copied fingerprint
    "buys nothing on its own". No read path can carry a device identity even
    in principle, so naming one as owed is a false instruction.

    Half one is the absence, over the whole source text of both files — the
    same extent the sweep greps — and each positive control says the 0 hits
    are an absence, not an unread or emptied file. Half two is the substance,
    read off the runtime ``__doc__`` of each module and of
    :func:`test_the_lease_route_still_answers_a_trusted_peer`, so the assert
    grades the docstrings themselves rather than a copy: they name the gate,
    and they cite #683 inside the statement that rules the device identity
    out. Deleting any of that wording turns this node red.
    """
    sources = {
        "app/routers/desktop.py": DESKTOP_ROUTER_SRC.read_text(),
        "tests/test_desktop_frame_mirror.py": THIS_TEST_SRC.read_text(),
    }
    for label, src in sources.items():
        assert DECLINED_DEVICE_GATE not in src.lower(), (
            f"{label} still names the declined device-identity step as a human "
            "action that is owed")
    assert "/api/desktop/lease" in sources["app/routers/desktop.py"], (
        "positive control: the router source read back empty, so the absence "
        "above proves nothing")
    assert "def test_the_lease_route_still_answers_a_trusted_peer" \
        in sources["tests/test_desktop_frame_mirror.py"], (
        "positive control: this file read back empty, so the absence above "
        "proves nothing")

    docstrings = {
        "app/routers/desktop.py": desktop_router.__doc__ or "",
        "tests/test_desktop_frame_mirror.py": __doc__ or "",
    }
    for label, doc in docstrings.items():
        flat = " ".join(doc.split())
        for token in BOUNDARY_TOKENS:
            assert token in flat, f"{label}'s module docstring omits {token}"
        cited = [s for s in flat.split(".") if "#683" in s]
        assert cited and any("device" in s or "identity" in s for s in cited), (
            f"{label} cites #683 but not inside the statement that rules a "
            "device identity out")

    lease_doc = " ".join(
        test_the_lease_route_still_answers_a_trusted_peer.__doc__.split())
    assert "ApiPeerGate" in lease_doc, (
        "the lease route's own test does not name the gate that answers the peer")
    assert "peer address" in lease_doc.lower(), (
        "the lease route's own test does not state the boundary as a peer address")
    lease_cited = [s for s in lease_doc.split(".") if "#683" in s]
    assert lease_cited and any("device" in s or "identity" in s for s in lease_cited), (
        "the lease route's own test cites #683 but not as the reason its reads "
        "carry no device identity")
    for deferred in ("human action", "leaves to a person", "left to a person"):
        assert deferred not in lease_doc.lower(), (
            f"the lease route's own test still defers the boundary ({deferred!r})")


DESKTOP_DOC = ROOT / "architecture" / "desktop.md"


def _doc_section(start: str, stop: str) -> str:
    """One section of ``architecture/desktop.md``, between two headings.

    Cut by heading pair rather than read whole: the doc's HISTORY section
    legitimately records what used to be true, and a node that priced the entire
    file would fail on a dated entry that is allowed to say the old thing while
    claiming the new one is false.
    """
    text = DESKTOP_DOC.read_text()
    i = text.index(start)
    return text[i:text.index(stop, i)]


def test_the_docs_state_that_a_grant_is_seat_only_and_a_revoke_is_not():
    """#4533 clause 5: the sentence that said the grant enforced nothing is the
    claim this round made false, and restated prose is where a stale instruction
    lives (#1671), so the new wording is pinned on both surfaces, not trusted.

    Two facts must survive in the router's module docstring and in
    ``architecture/desktop.md`` §3 alike: only the seat (a loopback peer) may
    grant, and a revoke stays open to a trusted peer. Both are asserted
    sentence-scoped, the way #1671 asserts its own citation, so wording that
    mentions ``loopback`` somewhere and grants somewhere else stays red. §3 must
    additionally have stopped saying the route enforces *nothing* for a peer,
    with the section's own ``ApiPeerGate`` sentence as the positive control that
    the section was read at all.
    """
    for label, flat in (
            ("app/routers/desktop.py",
             " ".join((desktop_router.__doc__ or "").split())),
            ("architecture/desktop.md §3",
             " ".join(_doc_section("## 3.", "\n## 4.").split()))):
        grants = [s for s in flat.split(".") if "grant" in s]
        assert grants and any("loopback" in s for s in grants), (
            f"{label} states a grant without naming the loopback peer that may "
            "ask for one")
        revokes = [s for s in flat.split(".") if "revoke" in s]
        assert revokes and any("trust" in s for s in revokes), (
            f"{label} states a revoke without naming the trusted peer that keeps "
            "it")

    s3 = _doc_section("## 3.", "\n## 4.")
    assert "nothing: `post_desktop_lease`" not in s3, (
        "§3 still says what the route enforces for a peer is nothing — #4533 "
        "closed that, and the sentence is now a false instruction")
    assert "ApiPeerGate" in s3, (
        "positive control: §3 read back empty, so the absence above proves nothing")
