"""#2207 — `/api/system/identity` says what it can prove, and the Settings card
stops presenting a caller-supplied header as a verified connection.

Two false claims lived here.

**"This response is identity."** `GET /api/system/identity` answered
`{"name", "fingerprint"}` and nothing else, so a value the server inferred from
the caller's own `x-client-fingerprint` header was byte-identical to a value it
had attested itself. Nothing on this box attests anything any more: no server
requests a client certificate since `5e1351f3` (2026-06-14 — iOS Chrome cannot
present a keychain identity), which the `httpsConfig` block in
`web/vite.config.ts` still records by carrying `key`/`cert` only, with no
`requestCert` and no `ca`. The absence is
pinned below against a positive control, because a grep that finds nothing is
worth nothing until something proves it can find something: `git grep -n
requestCert -- web/vite.config.ts` lands only inside the `clientCertHeaders`
docstring that records the absence — the config object itself never carries that
key — and `git grep -n "https: httpsConfig" -- web/vite.config.ts` resolves the
symbol to the `server:` block that consumes it.
So `scope["client_cert"]` is never populated, `server._cert_fingerprint` is the
only identity writer (`server.py:359-369`), and the name it stores is
`allowlist[fp]` (`server.py:344`) — a lookup of a header value in a file, not
proof of possession. `verified` is therefore False for every caller today, and
the last node here pins that it is *read* from the transport's certificate field
rather than typed as a literal, which is what makes "false today" a statement
about the transport instead of a tautology that hides the mechanism.

**"The caller cannot revoke their own cert."** `revoke_client` compared the
target against `request.state.client_name` and answered 400 with "cannot revoke
the cert you're currently using". That sentence describes a server that knows
whose certificate you are using, which is exactly what stopped being true at
`5e1351f3`: `tests/test_api_client_gating.py::test_an_enrolled_device_is_still_named_to_the_route`
proves the guard *does* fire for a caller who presents the enrolled fingerprint,
and nothing stops the same caller from omitting the header and revoking it. A
guard a liar walks past is not a guard, and the enforceable thing in a DELETE is
a body the server reads: the `confirm` field, whose refusal text and comment are
what clause 3 makes honest.

The Settings half is copy, so the `node` vitest environment cannot render it and
the pin is a source-text guard over `web/src` in pytest — the precedent is
`tests/test_api_contracts.py` and `tests/test_stale_mtls_comment_claims.py`. The
sentence that always rendered (`SettingsPage.tsx:769-771`, "Lloyd uses mutual TLS
— every device needs a client cert (signed by the on-host CA) to reach the API")
was flatly false: the boundary has been `ApiPeerGate`'s peer-address rule since
`1b050d82` (#683, 2026-09-20). The identity line under a green `ShieldCheck`
("You are connected as …") sat behind `identity?.name`, which cannot render while
`clients.json` is `{}`, so it is banned by wording as well.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server  # noqa: E402
from app.routers import system as system_router  # noqa: E402

LOOPBACK = "127.0.0.1"
ENROLLED_NAME = "alansrobotlabs-imac"
#: `clients.json` stores the colon-separated form and the gate normalises it to
#: upper-case hex with the separators stripped (`server._cert_fingerprint`), so
#: this is the value a handler sees.
ENROLLED_FP = "AABBCCDD"

SYSTEM_SRC = "app/routers/system.py"
SETTINGS_PAGE = "web/src/components/pages/SettingsPage.tsx"
WEB_API = "web/src/api.ts"

#: The three files this clause is about, named verbatim rather than by a glob:
#: a directory pattern over `web/src/**` would sweep in 40-odd components this
#: guard was not written for and silently shrink to nothing if the card moved.
CORPUS = (SYSTEM_SRC, SETTINGS_PAGE, WEB_API)

#: The commit that dropped client-certificate auth. The honest sentences must
#: keep naming it: "no verified caller identity exists" is a claim with a date,
#: and the date is what makes it checkable when someone re-enables mTLS.
MTLS_DROP_SHA = "5e1351f3"

# ── the sentences as they shipped, kept as control fixtures ──────────────────
#
# Every absence asserted below is a ban on a *shape*, and a ban that matches
# nothing is green forever. Each pattern therefore has to match the wording that
# actually shipped, quoted from the pre-change tree.

#: `app/routers/system.py:176` and `:179` before #2207 — the comment above the
#: comparison and the detail it raised. Two separate strings because the base has a
#: statement between them, and "byte-for-byte" has to mean the lines are each
#: literally in that file rather than spliced into a shape it never held.
OLD_SELF_REVOKE_COMMENT = (
    "    # Don't let the caller revoke their own cert (locks them out instantly)"
)
OLD_SELF_REVOKE_DETAIL = (
    '        raise HTTPException(400, "cannot revoke the cert you\'re currently using")'
)
#: `SettingsPage.tsx:769-770` before #2207 — the CardDescription, byte-for-byte
#: including the ten-space indent and the line break the JSX text node had.
OLD_CARD_DESCRIPTION = (
    "          Lloyd uses mutual TLS — every device needs a client cert (signed by the on-host CA)\n"
    "          to reach the API. Mint one cert per device and install it in that device's keystore."
)
#: `SettingsPage.tsx:816-817` before #2207 — the ShieldCheck icon and the line it
#: introduced, byte-for-byte including both className attributes.
OLD_IDENTITY_LINE = (
    '              <ShieldCheck className="w-4 h-4 text-emerald-400" />\n'
    '              You are connected as <span className="font-mono text-foreground">{identity.name}</span>'
)
#: `SettingsPage.tsx:750` before #2207 — the browser confirm() prompt, exactly as
#: the base blob has it (the em-dash-free ternary branch, backticks included).
OLD_YOU_PROMPT = (
    "`'${name}' is the cert YOU are using. Revoking it will lock you out "
    "immediately. Continue?`"
)

#: "the server knows whose cert you are". Legitimate prose about the drop is
#: allowed to say plenty; this is only the claim that a *caller's* identity is
#: known to the server by certificate.
CALLER_ATTESTATION_CLAIM = re.compile(
    r"cert you'?re currently using|your own cert|the cert you are using"
    r"|cert YOU are using",
    re.IGNORECASE,
)
#: A cert presented to the API by every device, which is the mechanism
#: `1b050d82` left behind. True prose may say a device *may* install one; what
#: is banned is the requirement.
CERT_REQUIRED_CLAIM = re.compile(
    r"mutual\s+TLS|needs?\s+a\s+client\s+cert[^.]*to reach", re.IGNORECASE
)
#: A name presented as settled fact about who is connected.
VERIFIED_YOU_CLAIM = re.compile(r"you\s+are\s+connected\s+as", re.IGNORECASE)


def _tracked_text(rel_path: str) -> str:
    """Read one corpus file, asserting first that it is really in the corpus.

    A missing or empty file would make every absence below pass for the wrong
    reason, so the guard runs before any searching: tracked in this worktree, and
    with bytes in it.
    """
    path = ROOT / rel_path
    assert path.is_file(), f"{rel_path} is not a file — the corpus moved"
    assert path.stat().st_size > 0, f"{rel_path} is empty — nothing to grep"
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel_path],
        capture_output=True, text=True,
    )
    assert tracked.returncode == 0, f"{rel_path} is not tracked by git: {tracked.stderr.strip()}"
    return path.read_text(encoding="utf-8")


#: The commit #2207's changes sit on top of — `5f4d378e`, the base the round was
#: opened from. Named, not derived: `HEAD~1` was the pre-change tree while this
#: round had one commit, and stopped being it the moment the review fixes were
#: committed on top. A control that silently starts reading the round's own output
#: is worse than no control, which is exactly what the gate caught on `b45cbd6e`.
BASE_SHA = "5f4d378e"


def _base_blob(rel_path: str) -> str:
    """`rel_path` as `BASE_SHA` had it — the tree before #2207.

    Used only to prove an assertion is not vacuous: the shipped file itself is the
    control's source, read from the commit that shipped it.
    """
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_SHA}:{rel_path}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"{rel_path} is not readable at {BASE_SHA} (the round's base commit): "
        f"{r.stderr.strip()}")
    assert r.stdout, f"{rel_path} is empty at {BASE_SHA}"
    return r.stdout


# ── the ban patterns are proven to fire on the wording they were written for ──


def test_the_bans_fire_on_the_wording_that_shipped():
    """Positive control for every absence asserted below, in two parts.

    First the fixtures are checked against the pre-change tree itself — each string
    must appear byte-for-byte in the file and commit it claims — so the word
    "verbatim" above is an assertion rather than a comment someone can quietly stop
    believing. Then each pattern must match its own sentence: a ban that stops
    matching the reason it exists is green forever, and this node is the only place
    that would notice.
    """
    router_base = _base_blob(SYSTEM_SRC)
    page_base = _base_blob(SETTINGS_PAGE)
    for fixture in (OLD_SELF_REVOKE_COMMENT, OLD_SELF_REVOKE_DETAIL):
        assert fixture in router_base, (
            f"a control fixture is not the bytes app/routers/system.py shipped: "
            f"{fixture[:60]!r}…")
    for fixture in (OLD_CARD_DESCRIPTION, OLD_IDENTITY_LINE, OLD_YOU_PROMPT):
        assert fixture in page_base, (
            f"a control fixture is not the bytes SettingsPage.tsx shipped: "
            f"{fixture[:60]!r}…")

    assert CALLER_ATTESTATION_CLAIM.search(OLD_SELF_REVOKE_DETAIL), (
        "the attestation ban no longer matches 'cannot revoke the cert you're "
        "currently using' — it cannot catch it coming back")
    assert CALLER_ATTESTATION_CLAIM.search(OLD_YOU_PROMPT), (
        "the attestation ban no longer matches the UI's 'is the cert YOU are using'")
    assert CERT_REQUIRED_CLAIM.search(OLD_CARD_DESCRIPTION), (
        "the requirement ban no longer matches 'Lloyd uses mutual TLS — every "
        "device needs a client cert … to reach the API'")
    assert VERIFIED_YOU_CLAIM.search(OLD_IDENTITY_LINE), (
        "the identity ban no longer matches 'You are connected as'")


# ── clause 1: the response carries the flag, and the flag is false ───────────


@pytest.fixture
def one_device(tmp_path, monkeypatch):
    """A `clients.json` with one enrolled device, for both readers of it.

    `server._load_allowlist` decides whether a fingerprint-bearing request is
    served at all; `app.routers.system` reads the same file for its own listing.
    Both module constants move, so nothing here can reach
    `agent-services/cert/clients.json`.
    """
    cert_dir = tmp_path / "cert"
    cert_dir.mkdir()
    live = cert_dir / "clients.json"
    live.write_text(json.dumps({ENROLLED_NAME: {"fingerprint": "AA:BB:CC:DD"}}))
    monkeypatch.setattr(server, "CLIENTS_JSON", live)
    monkeypatch.setattr(system_router, "CLIENTS_JSON", live)
    monkeypatch.setattr(system_router, "CLIENTS_DIR", cert_dir / "clients")
    return live


def _client(app=None) -> httpx.AsyncClient:
    """An app whose socket peer is loopback — a trusted peer, so the identity
    route is reached and only the header evidence varies between calls."""
    transport = httpx.ASGITransport(app=app or server.app, client=(LOOPBACK, 5555))
    return httpx.AsyncClient(transport=transport, base_url="http://lloyd-test")


async def test_an_enrolled_fingerprint_is_reported_as_unverified(one_device):
    """Clause 1, the case that used to be indistinguishable: the caller presents
    a fingerprint that IS in the allowlist, so the route has a name to return —
    and it must still say it did not verify it. Asserted as exact equality so a
    future field cannot appear without being noticed here, and `is False` rather
    than `== False` so a truthy placeholder (a string, `0`) is not mistaken for
    the boolean."""
    async with _client() as client:
        r = await client.get("/api/system/identity",
                             headers={"x-client-fingerprint": ENROLLED_FP})
    assert r.status_code == 200, r.text[:200]
    assert r.json() == {"name": ENROLLED_NAME, "fingerprint": ENROLLED_FP,
                        "verified": False}
    assert r.json()["verified"] is False, "verified must be a boolean, not falsy"


async def test_a_caller_sending_nothing_is_unverified_too(one_device):
    """Clause 1, the other half — and the only answer this box can give a real
    browser today. No header means `server.py:329-345` stores nothing, so both
    values are null; `verified` has to be present and false anyway, because a
    missing key is the shape the UI used to read as "no identity" and render as
    "you are nobody", which is not the same claim as "nobody verified you"."""
    async with _client() as client:
        r = await client.get("/api/system/identity")
    assert r.status_code == 200, r.text[:200]
    assert r.json() == {"name": None, "fingerprint": None, "verified": False}
    assert r.json()["verified"] is False


async def test_a_forged_client_cn_header_buys_no_identity_and_no_verification(
        one_device):
    """The header the item's body named is not an identity source at all: the
    backend has never read `x-client-cn` (`git grep -n "x-client-cn" -- '*.py'`
    → 0 code hits, `git log -S'x-client-cn' -- server.py` → 0 commits; the only
    writer is the still-wired `clientCertHeaders()` plugin in
    `web/vite.config.ts`, which sets no header for browser traffic today).
    So a caller that forges
    it gets a null name — it cannot even become unverified identity, let alone
    verified. Kept as a node because the shape is the one worth pinning: no
    header on this route can produce `verified: true`."""
    async with _client() as client:
        r = await client.get("/api/system/identity",
                             headers={"x-client-cn": ENROLLED_NAME})
    assert r.status_code == 200, r.text[:200]
    assert r.json() == {"name": None, "fingerprint": None, "verified": False}


async def test_verified_is_read_from_the_certificate_field_not_hardcoded():
    """The other half of what makes clause 1 mean something. A `verified: False`
    typed as a literal would satisfy every assertion above while proving nothing:
    it would still be saying "not verified" on the day mTLS came back, and the flag
    would be decoration. So the value is read from `scope["client_cert"]` — the one
    place a TLS-terminating server could hand the app a verified peer certificate —
    and that key is a convention this app defines rather than one this stack
    implements: uvicorn 0.44.0 and Starlette 1.3.1 never mention it (`grep -rn
    --include=*.py client_cert .venvs/lloyd/lib/python3.12/site-packages/uvicorn/
    .venvs/lloyd/lib/python3.12/site-packages/starlette/` → 0 hits). The certificate
    is therefore synthesised here, because no transport on this box produces one —
    the same fact the module docstring's absence check records, seen from the other
    side. This node does NOT decide whether mTLS should be re-enabled; that ruling
    is owed. It pins only which way the field reads: certificate evidence makes it
    true, and its absence makes it false."""
    scope = {
        "type": "http", "method": "GET", "path": "/api/system/identity",
        "headers": [(b"host", b"lloyd-test")],
        "query_string": b"", "client": (LOOPBACK, 5555), "state": {},
        "client_cert": b"-----BEGIN CERTIFICATE-----\nsynthetic\n",
    }
    # Direct call, not through `server.app`: the point is to hand the handler the
    # one input the real transports cannot, and `ApiPeerGate` would never set it.
    r = await system_router.identity(Request(scope))
    body = json.loads(r.body)
    assert body["verified"] is True, (
        "`verified` is not following the transport's certificate evidence — a "
        "literal False here is the dishonesty this clause was written to remove, "
        "pointing the other way")

    del scope["client_cert"]
    r2 = await system_router.identity(Request(scope))
    assert json.loads(r2.body)["verified"] is False, (
        "`verified` is true without a client certificate in the scope")


# ── clause 3: the refusal text and the comment describe the real check ───────


async def test_the_revoke_refusal_states_the_enforceable_check(one_device):
    """Clause 3: a DELETE refused for want of a `confirm` body must tell the
    caller what will actually be enforced, and must not imply the server could
    tell whose certificate they were holding. Asserted on the wire, not on the
    constant, because the detail text is the surface a client author reads."""
    async with _client() as client:
        r = await client.delete("/api/system/clients/alansrobotlabs-imac")
    assert r.status_code == 400, r.text[:200]
    detail = r.json()["detail"]
    assert "confirm" in detail, f"the refusal never names the field it wants: {detail!r}"
    assert MTLS_DROP_SHA in detail, (
        f"the refusal must name the commit that removed verified caller identity: {detail!r}")
    assert "enforce" in detail.lower(), (
        f"the refusal must say the confirm body is what is enforced: {detail!r}")
    banned = CALLER_ATTESTATION_CLAIM.search(detail)
    assert not banned, (
        f"the refusal still claims the server knows whose cert this is: {banned.group(0)!r}")


async def test_the_revoke_comment_and_the_identity_docstring_stay_honest():
    """Clause 3, the prose half. The comment over the self-revocation comparison
    has to keep naming the sha that made it advisory and say the compared name is
    the caller's own header; `identity`'s docstring has to say the response is
    not attested identity. Pinned against the source because a comment is exactly
    the thing a later edit rewrites back into the comfortable version."""
    text = _tracked_text(SYSTEM_SRC)
    assert MTLS_DROP_SHA in text, (
        f"{SYSTEM_SRC} no longer names {MTLS_DROP_SHA} — the honest sentences are gone")
    assert "x-client-fingerprint" in text, (
        f"{SYSTEM_SRC} must name the header the compared name actually comes from")
    banned = CALLER_ATTESTATION_CLAIM.search(text)
    assert not banned, (
        f"{SYSTEM_SRC} claims the server knows the caller's own cert again: "
        f"{banned.group(0)!r}")

    revoke_src = _route_body(text, "async def revoke_client(")
    assert "confirm" in revoke_src, "the route must name the field it enforces"
    assert MTLS_DROP_SHA in revoke_src, (
        "the comment over the caller comparison must keep naming the commit that "
        "made it advisory, or the next reader has to rediscover why")

    ident = _identity_docstring(text).lower()
    assert "header" in ident, (
        "identity()'s docstring must say the name it returns is header-derived — "
        "that is the sentence a reader mistakes for 'this is who you are'")
    assert "verified" in ident, "identity()'s docstring must explain its own field"


def _route_body(src: str, signature: str) -> str:
    """One route's own source: `signature` up to the next top-level statement.

    Slicing to end-of-file was the first version, and the review of 2026-10-05
    named what that allowed: prose written *after* the route — a later function's
    comment, or a note at the bottom of the module — would satisfy "the route
    names the sha" without the route saying anything. Cutting at the next
    zero-indent `@router`/`def`/`class` means the assertion can only be satisfied
    by the route's own body, and the two checks below are what stop a bounded slice
    that silently missed the route (no decorator) or swallowed the next one (two
    signatures).
    """
    start = src.find(signature)
    assert start >= 0, f"{signature!r} is gone from the router"
    body = src[start:]
    nxt = re.search(r"\n(?:@router\.|async def |def |class )", body[1:])
    bounded = body[:1 + nxt.start()] if nxt else body
    assert bounded.strip(), f"{signature!r} has an empty body"
    assert bounded.count("async def ") == 1, (
        f"the slice for {signature!r} spans more than one route: {bounded!r}")
    return bounded


def _identity_docstring(src: str) -> str:
    start = src.find("async def identity(")
    assert start >= 0, "`identity` vanished from the router — the route was deleted"
    body = src[start:]
    opened = body.find('"""')
    assert opened >= 0, "identity() has no docstring to hold the claim"
    closed = body.find('"""', opened + 3)
    assert closed >= 0, "identity()'s docstring is unterminated"
    return body[opened + 3:closed]


# ── clause 5: the card's copy ────────────────────────────────────────────────


def test_the_settings_card_stops_asserting_a_verified_connection():
    """Clause 5: the cert card may no longer render `identity.name` under a
    shield as who you are, and its header may no longer state that every device
    needs a client cert to reach the API. Every file in `CORPUS` is banned over,
    each one proven tracked and non-empty before it is searched, so the 0-hit
    result cannot be a grep that searched nothing and the tuple cannot read wider
    than the check is; `identity.name` is then asserted to be STILL rendered in the
    card — the fix is to label it, not to delete the feature."""
    # The ban runs over every file in the corpus rather than over the one file the
    # sentence moved out of. A copy of `CORPUS` that no guard dereferences would
    # let the tuple read wider than the check is: `mutual TLS` coming back into
    # `api.ts`'s `getIdentity` docstring is the same false claim, and the card's
    # copy is free to move files that no per-file ban would then follow.
    for rel_path in CORPUS:
        text = _tracked_text(rel_path)
        for pattern, why in (
                (VERIFIED_YOU_CLAIM, "presents a header-derived name as who you are"),
                (CERT_REQUIRED_CLAIM, "claims a client cert is required to reach the API"),
                (CALLER_ATTESTATION_CLAIM, "claims the server knows whose cert this caller holds")):
            hit = pattern.search(text)
            assert not hit, f"{rel_path} {why}: {hit.group(0)!r}"

    text = _tracked_text(SETTINGS_PAGE)
    assert "identity.name" in text, (
        f"{SETTINGS_PAGE} stopped rendering the reported name at all — the clause "
        "is to label it as unverified, not to hide it")
    assert "identity.verified" in text, (
        f"{SETTINGS_PAGE} must branch on the server's `verified` flag rather than "
        "decide the wording on its own")
    assert "unverified" in text.lower(), (
        f"{SETTINGS_PAGE} must say the reported name is unverified")


def test_the_client_layer_still_names_the_identity_route_it_calls():
    """Corpus control for the file above: `web/src/api.ts` is tracked, non-empty
    and still the caller of both endpoints this round changed, so a guard that
    reads it is reading the layer a revoke actually goes through.

    The second half is the coverage proof the corpus needs and a restatement of its
    own tuple could never supply: the sentences the bans exist to keep out shipped
    in files that are *inside* this corpus. Read from the base blob, so if the cert
    card ever moves to a file the guard does not read, the guard's own history stops
    matching it and this node goes red rather than the ban going quiet.
    """
    text = _tracked_text(WEB_API)
    assert "/system/identity" in text, "api.ts no longer calls the identity route"
    assert "/system/clients/" in text, "api.ts no longer calls the clients route"

    page_base = _base_blob(SETTINGS_PAGE)
    covered = sum(1 for phrase in (OLD_IDENTITY_LINE, OLD_CARD_DESCRIPTION, OLD_YOU_PROMPT)
                  if phrase in page_base)
    assert covered == 3, (
        f"only {covered} of the three false claims this round removed were shipped "
        "in SettingsPage.tsx, the file the guard reads — either the corpus does not "
        "cover what it bans, or the card moved out of it")


# ── acceptance: no route was deleted ─────────────────────────────────────────


def test_every_cert_surface_is_still_registered():
    """The acceptance check's "with every route kept": `cert/ca`, clients
    GET/POST, the `.p12` download and revoke must all still be mounted. Counted
    off the live app rather than off the source text, and with the count asserted
    so an accidental `@router.delete` removal cannot hide behind a route that was
    renamed — the item's whole ruling was "keep the surface, bound it"."""
    routes = {(m, r.path)
              for r in server.app.routes
              for m in (getattr(r, "methods", None) or [])}
    kept = {
        ("GET", "/api/system/identity"),
        ("GET", "/api/system/lan-info"),
        ("GET", "/api/system/cert/ca"),
        ("GET", "/api/system/clients"),
        ("POST", "/api/system/clients"),
        ("GET", "/api/system/clients/{name}/p12"),
        ("DELETE", "/api/system/clients/{name}"),
    }
    missing = kept - routes
    assert not missing, f"routes deleted from /api/system: {sorted(missing)}"
    system_paths = {p for (_, p) in routes if p.startswith("/api/system/")}
    assert system_paths == {p for (_, p) in kept}, (
        f"the /api/system surface drifted from the seven routes this item names: "
        f"{sorted(system_paths)}")
