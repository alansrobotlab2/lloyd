"""#2207 clause 2 — `DELETE /api/system/clients/{name}` cannot succeed unless the
request body names that client in a `confirm` field.

The route used to read no body at all: `system.py:163-191` (pre-#2207) validated
the name's shape, looked it up, compared it against a header-derived caller name,
and removed it. The only thing standing between a stray `fetch` and a revoked
device was a browser `confirm()` dialog (`SettingsPage.tsx:752`) over a bodyless
`fetch` (`web/src/api.ts:1868-1871`) — client-side ceremony, which is to say
nothing, to anything that is not the browser.

The server-side check that replaces it is deliberately the *confirm body*, not
caller identity, because caller identity is not available to check: no server on
this box requests a client certificate since `5e1351f3`, so the one name a caller
can present arrives in `x-client-fingerprint` and is looked up in the same file
the request is asking to edit (`server.py:329-345`, `server._cert_fingerprint`).
A rule of the form "revoke X only if you are not X" is a rule the caller decides.
"revoke X only if you spelled X twice" is a rule the server decides, and it is the
one this file pins: absent body, non-JSON body and a mismatched value all answer
400; only a match answers 204 and removes the entry.

Everything runs against a fixture `clients.json` under `tmp_path`. The store the
route must *not* touch is asserted untouched around a successful revoke, and it
is a populated decoy rather than the real path:
`agent-services/cert/clients.json` is gitignored (`.gitignore:99`) and so absent
from every worktree, where a before/after compare there reads `None` against
`None` and can only catch a writer that creates a file. The decoy
(`decoy_cert_store`) carries that load with bytes and key material in it, and the
real path is still compared, with its weakness stated where it is asserted. The
fixture also asserts the module constants moved off the repo paths, and the
`.crt`/`.key`/`.p12` cleanup is proven scoped to the temporary directory rather
than to `agent-services/cert/clients/`.

Requests go through `server.app` with an `httpx.ASGITransport`, so the peer gate
is on the path exactly as it is in production; every call comes from loopback,
which is trusted, so a 400 can only be the route's own.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server  # noqa: E402
from app.routers import system as system_router  # noqa: E402

LOOPBACK = "127.0.0.1"

#: Two enrolled devices, so a `confirm` for the wrong one is distinguishable from
#: a `confirm` for the right one, and a refusal can be shown not to have deleted
#: a neighbour. Fingerprints are the colon form `clients.json` really stores; the
#: gate compares the normalised form (`server._cert_fingerprint`).
DEVICES = {
    "studio-mini": {"fingerprint": "AA:AA:AA:AA"},
    "hallway-pi": {"fingerprint": "BB:BB:BB:BB"},
}


@pytest.fixture
def fixture_cert_dir(tmp_path, monkeypatch):
    """A throwaway cert dir, with on-disk material for both devices.

    Returns the directory. The module constants the route touches are all moved
    onto it, and the assertion here is that none of them still points at the
    repository: the destructive half of `revoke_client` is the unlink loop, and a
    test that let it run against `agent-services/cert/` would be deleting real key
    material to prove a point.
    """
    cert_dir = tmp_path / "cert"
    cert_dir.mkdir()
    clients_file = cert_dir / "clients.json"
    clients_file.write_text(json.dumps(DEVICES, indent=2))
    clients_dir = cert_dir / "clients"
    clients_dir.mkdir()
    for name in DEVICES:
        for ext in ("crt", "key", "p12"):
            (clients_dir / f"{name}.{ext}").write_text(f"fixture {name} {ext}\n")

    live_json = system_router.CLIENTS_JSON
    live_dir = system_router.CLIENTS_DIR
    monkeypatch.setattr(system_router, "CLIENTS_JSON", clients_file)
    monkeypatch.setattr(system_router, "CLIENTS_DIR", clients_dir)
    monkeypatch.setattr(server, "CLIENTS_JSON", clients_file)

    assert system_router.CLIENTS_JSON != live_json, "the JSON path did not move off the repo"
    assert system_router.CLIENTS_DIR != live_dir, "the cert dir did not move off the repo"
    return cert_dir


#: A third device, present only in the decoy store, so the decoy's own entry
#: count is not the thing at risk when the revoke under test targets one of the
#: two fixture devices.
DECOY_NAME = "kitchen-tablet"


@pytest.fixture
def decoy_cert_store(tmp_path, monkeypatch):
    """A second, fully populated cert store, parked where `server`'s copy of the
    path points — the store this revoke must not touch.

    Why a decoy and not the real file: `agent-services/cert/clients.json` is
    gitignored (`.gitignore:99`), so it does not exist in a worktree, and a
    before/after compare against it is `None == None` on both sides. This fixture
    is the review finding of 2026-10-05 made into a check with a non-zero
    denominator: it has JSON bytes, it lists the very client the revoke names,
    and it has `.crt`/`.key`/`.p12` material on disk.

    Why *this* location: `server.py:52` and `app/routers/system.py:23` spell the
    same file out independently, and the peer gate reads the former
    (`_load_allowlist`, `server.py:55-62`) while the routes read the latter
    (`_save_clients`). "The router wrote through the other module's constant" is
    therefore a live drift risk, and this is the store that would notice.

    Ordering that the node depends on: `fixture_cert_dir` patches
    `server.CLIENTS_JSON` onto the fixture first, so any test requesting both must
    list `fixture_cert_dir` first in its signature and this one wins after it.
    """
    decoy_dir = tmp_path / "decoy" / "cert"
    decoy_dir.mkdir(parents=True)
    decoy_json = decoy_dir / "clients.json"
    decoy_json.write_text(json.dumps(
        {**DEVICES, DECOY_NAME: {"fingerprint": "CC:CC:CC:CC"}}, indent=2))
    decoy_clients = decoy_dir / "clients"
    decoy_clients.mkdir()
    for name in (*DEVICES, DECOY_NAME):
        for ext in ("crt", "key", "p12"):
            (decoy_clients / f"{name}.{ext}").write_text(f"decoy {name} {ext}\n")

    monkeypatch.setattr(server, "CLIENTS_JSON", decoy_json)
    assert server.CLIENTS_JSON == decoy_json, "the decoy did not take"
    assert system_router.CLIENTS_JSON != decoy_json, "fixture and decoy are the same store"
    return decoy_dir


def _live_cert_paths() -> tuple[Path, Path]:
    """The real cert paths, rebuilt from the repository root — what the module
    constants point at before the fixture moves them."""
    return (ROOT / "agent-services" / "cert" / "clients.json",
            ROOT / "agent-services" / "cert" / "clients")


async def _delete(name: str, *, content=None, headers=None) -> httpx.Response:
    transport = httpx.ASGITransport(app=server.app, client=(LOOPBACK, 5555))
    async with httpx.AsyncClient(transport=transport,
                                base_url="http://lloyd-test") as client:
        return await client.request("DELETE", f"/api/system/clients/{name}",
                                    content=content, headers=headers)


def _entries(cert_dir: Path) -> dict:
    return json.loads((cert_dir / "clients.json").read_text())


def _json_body(name: str) -> tuple[bytes, dict]:
    return (json.dumps({"confirm": name}).encode(),
            {"content-type": "application/json"})


async def test_a_revoke_with_no_body_is_refused_and_deletes_nothing(fixture_cert_dir):
    """Clause 2, the absent-body case — the exact request the old UI sent. Must
    be refused, and the refusal must be observable in the file: entry still
    listed, `.p12` still on disk. A 400 that had already unlinked the material
    would be the worse bug."""
    before = _entries(fixture_cert_dir)
    r = await _delete("studio-mini")
    assert r.status_code == 400, r.text[:200]
    assert _entries(fixture_cert_dir) == before, "an entry vanished on a refused revoke"
    assert (fixture_cert_dir / "clients" / "studio-mini.p12").is_file(), \
        "the refused revoke deleted key material on its way out"


async def test_a_revoke_whose_body_is_not_json_is_refused(fixture_cert_dir):
    """Clause 2, the non-JSON case. A form-encoded body is the shape an unwrapped
    `fetch` produces, and it must not be silently read as "no confirm given, but
    that's fine" — nor may it 500 on the decode."""
    r = await _delete("studio-mini", content=b"confirm=studio-mini",
                      headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 400, r.text[:200]
    assert set(_entries(fixture_cert_dir)) == {"studio-mini", "hallway-pi"}, "a non-JSON body revoked a device"


async def test_a_revoke_whose_confirm_names_another_client_is_refused(fixture_cert_dir):
    """Clause 2, the mismatch case, and the one that makes the check mean
    something beyond "a body arrived": confirming `hallway-pi` must not revoke
    `studio-mini`. Both entries have to survive, since a partial write would turn
    a refusal into a different outage."""
    body = json.dumps({"confirm": "hallway-pi"}).encode()
    r = await _delete("studio-mini", content=body,
                      headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text[:200]
    assert set(_entries(fixture_cert_dir)) == {"studio-mini", "hallway-pi"}, "the refused revoke removed an entry"
    assert (fixture_cert_dir / "clients" / "studio-mini.p12").is_file()


@pytest.mark.parametrize("body", [
    pytest.param(None, id="no-body-at-all"),
    pytest.param(b"", id="empty-body"),
    pytest.param(b"not json at all", id="non-json"),
    pytest.param(b"[]", id="json-array"),
    pytest.param(b'{"confirm": 7}', id="confirm-not-a-string"),
    pytest.param(b'{"name": "studio-mini"}', id="confirm-absent"),
    pytest.param(b'{"confirm": "studio-mini "}', id="confirm-not-the-path-name"),
])
async def test_every_refused_shape_answers_400(fixture_cert_dir, body):
    """Clause 2 as a table. `{"confirm": "studio-mini "}` is in it on purpose: the
    compared value is the path name, and a trailing space trimmed on one side only
    would let a mismatch through the door."""
    r = await _delete("studio-mini", content=body,
                      headers={"content-type": "application/json"})
    assert r.status_code == 400, f"{body!r} -> {r.status_code} {r.text[:120]}"
    assert set(_entries(fixture_cert_dir)) == {"studio-mini", "hallway-pi"}


async def test_a_revoke_with_the_matching_confirm_removes_only_that_entry(fixture_cert_dir):
    """Clause 2's green path: 204, the entry gone, the neighbour intact, and the
    target's key material unlinked — which is the behaviour the old route had and
    must keep, since a revoked cert whose key survives on disk is a revoked cert
    that can still be re-imported by hand."""
    content, headers = _json_body("studio-mini")
    r = await _delete("studio-mini", content=content, headers=headers)
    assert r.status_code == 204, r.text[:200]
    assert r.content == b"", "204 must not carry a body"
    assert set(_entries(fixture_cert_dir)) == {"hallway-pi"}
    for ext in ("crt", "key", "p12"):
        assert not (fixture_cert_dir / "clients" / f"studio-mini.{ext}").exists(), ext
        assert (fixture_cert_dir / "clients" / f"hallway-pi.{ext}").is_file(), ext


async def test_the_live_cert_store_is_never_touched(fixture_cert_dir, decoy_cert_store):
    """Clause 2's "never the real `clients.json`", measured against a store that
    has bytes in it — `fixture_cert_dir` first so this node's `server.CLIENTS_JSON`
    is the decoy, per that fixture's docstring.

    The compare is only worth reading if the denominator is non-zero, so the
    decoy's own state is asserted before the request rather than assumed, and the
    fixture's mutation is asserted after it: without that second half, "nothing
    changed over there" is equally satisfied by a revoke that did nothing anywhere.

    The real path is still compared, but its power is stated rather than implied.
    In a worktree it does not exist, so the only thing that line can catch is a
    writer that creates it; on the live checkout, where the file is `{}`, it would
    additionally catch a rewrite to different bytes. The load is carried by the
    decoy.
    """
    decoy_json = decoy_cert_store / "clients.json"
    decoy_clients = decoy_cert_store / "clients"
    listed_before = sorted(p.name for p in decoy_clients.iterdir())
    assert decoy_json.stat().st_size > 2, "the decoy store is empty — this compare proves nothing"
    assert DECOY_NAME in json.loads(decoy_json.read_text()), "the decoy has no entries"
    assert listed_before, "the decoy has no key material"
    bytes_before = decoy_json.read_bytes()

    live_json, live_dir = _live_cert_paths()
    live_before = live_json.read_bytes() if live_json.exists() else None
    live_listed_before = sorted(p.name for p in live_dir.iterdir()) if live_dir.is_dir() else []

    content, headers = _json_body("studio-mini")
    r = await _delete("studio-mini", content=content, headers=headers)
    assert r.status_code == 204, r.text[:200]

    assert set(_entries(fixture_cert_dir)) == {"hallway-pi"}, \
        "the revoke did not land in the fixture, so the comparisons below prove nothing"
    assert decoy_json.read_bytes() == bytes_before, (
        "a fixture-scoped revoke wrote through server.CLIENTS_JSON — the two modules "
        "name the same file independently and only one of them was patched")
    assert sorted(p.name for p in decoy_clients.iterdir()) == listed_before, (
        "a fixture-scoped revoke deleted material out of the store it was not pointed at")
    assert (decoy_clients / "studio-mini.p12").is_file(), (
        "the decoy lost the target's .p12 while the fixture kept its own entry list intact")

    assert (live_json.read_bytes() if live_json.exists() else None) == live_before, \
        "the live clients.json changed under a fixture-scoped revoke"
    live_listed_after = sorted(p.name for p in live_dir.iterdir()) if live_dir.is_dir() else []
    assert live_listed_before == live_listed_after, \
        "the live cert dir changed under a fixture-scoped revoke"


async def test_a_valid_confirm_for_an_unknown_client_is_404(fixture_cert_dir):
    """The order of the two checks, pinned rather than left to chance: a body that
    satisfies the confirm rule but names nobody gets the router's own 404, not a
    400 about a field it did send. Without this node a refactor that moved the
    confirm check after the lookup would silently change what a mistyped name
    looks like to the UI."""
    content, headers = _json_body("no-such-device")
    r = await _delete("no-such-device", content=content, headers=headers)
    assert r.status_code == 404, r.text[:200]
    assert "no-such-device" in r.json()["detail"]
    assert set(_entries(fixture_cert_dir)) == {"studio-mini", "hallway-pi"}


async def test_a_name_the_router_would_never_take_is_refused_on_the_name(fixture_cert_dir):
    """`NAME_RE` still gates the shape: a name the route would never accept is a
    400 about the name even with a well-formed confirm sitting in the body, which
    keeps the new check from answering first and teaching a caller nothing."""
    content, headers = _json_body("bad name")
    r = await _delete("bad name", content=content, headers=headers)
    assert r.status_code == 400, r.text[:200]
    assert "name" in r.json()["detail"].lower(), r.json()["detail"]
    assert set(_entries(fixture_cert_dir)) == {"studio-mini", "hallway-pi"}


async def test_the_header_derived_self_revoke_refusal_is_still_there_and_honest(
        fixture_cert_dir):
    """The guard's behaviour is kept (an enrolled caller presenting the enrolled
    fingerprint is still refused for its own entry), and so is the honesty of its
    text: `tests/test_api_client_gating.py::test_an_enrolled_device_is_still_named_to_the_route`
    shows the comparison does fire, so the claim that was wrong was never "does it
    run" but "is it a control" — the same caller with no header gets through, and
    that bypass is asserted right below so it stays a documented behaviour rather
    than a rediscovered surprise. The refusal must name the sha that made it
    advisory and must not say the server knows whose certificate this is."""
    content, headers = _json_body("studio-mini")
    r = await _delete("studio-mini", content=content,
                      headers={**headers, "x-client-fingerprint": "AAAAAAAA"})
    assert r.status_code == 400, (
        "the self-revocation refusal disappeared: a caller presenting its own "
        "enrolled fingerprint revoked itself. This node keeps the guard; only its "
        "wording was wrong")
    detail = r.json()["detail"]
    assert "5e1351f3" in detail, f"the refusal must name the drop: {detail!r}"
    assert "x-client-fingerprint" in detail, (
        f"the refusal must say where the compared name came from: {detail!r}")

    content2, headers2 = _json_body("hallway-pi")
    r2 = await _delete("hallway-pi", content=content2, headers=headers2)
    assert r2.status_code == 204, (
        "the confirm path regressed: with no header the honest revoke must proceed")
    assert set(_entries(fixture_cert_dir)) == {"studio-mini"}


# ── clause 4, seen from the server side: does the shipped client pass this check? ──

WEB_API = "web/src/api.ts"


def _tracked(rel_path: str) -> str:
    """Read a repo file, proving first that git tracks it and that it has bytes —
    the same rule the honesty guard runs, restated here so this file does not
    depend on the other module's private helpers."""
    path = ROOT / rel_path
    assert path.is_file(), f"{rel_path} is not a file — the client layer moved"
    assert path.stat().st_size > 0, f"{rel_path} is empty — nothing to read"
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel_path],
                             capture_output=True, text=True)
    assert tracked.returncode == 0, f"{rel_path} is not tracked: {tracked.stderr.strip()}"
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


def _client_fn(rel_path: str, fn: str) -> str:
    """The body of one `api` method, sliced between its own signature and the
    closing brace that ends it — so an assertion cannot be satisfied by prose in
    another method or in a comment."""
    text = _tracked(rel_path)
    start = text.find(fn)
    assert start >= 0, f"{fn} is gone from {rel_path} — the client layer moved"
    end = text.find("\n  },", start)
    assert end > start, f"{fn} in {rel_path} is not followed by its own close"
    return text[start:end]


def _client_request_shape(rel_path: str) -> tuple[str, str]:
    """`(confirm_key, media_type)` as the shipped `revokeClient` actually spells
    them, read out of its own `headers`/`body` lines.

    Returned rather than hardcoded so the node below sends what the browser would
    send: a client that renames the field changes the bytes under test and the
    route refuses them, instead of the test quietly asserting on a name the client
    no longer uses. The value expression is checked to be the function's own
    parameter, which is the difference between sending the caller's string and
    sending something else that happens to match.
    """
    body = _client_fn(rel_path, "revokeClient: async (name: string)")
    param = re.search(r"revokeClient:\s*async\s*\(\s*([A-Za-z_$][\w$]*)\s*:", body)
    assert param, f"cannot read revokeClient's own parameter out of: {body!r}"
    match = re.search(
        r"body:\s*JSON\.stringify\(\{\s*([A-Za-z_$][\w$]*)\s*:\s*([A-Za-z_$][\w$]*)\s*\}\)",
        body)
    assert match, f"revokeClient sends no single-field JSON.stringify body: {body!r}"
    key, value = match.group(1), match.group(2)
    assert value == param.group(1), (
        f"the body carries {value!r}, not the caller's own {param.group(1)!r} — the "
        "route compares the value against the decoded path segment")
    media = re.search(r"'Content-Type':\s*'([^']+)'", body)
    assert media, f"revokeClient sets no Content-Type, so the route's reader sees none: {body!r}"
    return key, media.group(1)


async def test_the_request_the_shipped_client_builds_is_the_request_this_route_accepts(
        fixture_cert_dir):
    """Clause 4 and clause 2 met in one executed request — the half neither runner
    on this box can reach alone.

    Vitest stubs `fetch` and never opens a socket; pytest drives
    `httpx.ASGITransport` and never reads the client. "A revoke started from the
    Settings card passes the new server check" sat between the two suites, and the
    review of 2026-10-05 named that gap. This node closes the part of it that is
    decidable here: the confirm key and the media type are read out of the shipped
    `api.ts`, turned into the bytes `JSON.stringify` puts on the wire (compact, no
    spaces, like the browser's) and sent. If the client renames the field or drops
    the JSON content type, the route refuses and this fails — not a coincidence
    that both files spell `confirm` today.

    The control is the pre-change client read from the named base commit: it built
    no body, and the request built from it must now be refused. Without it the 204
    would only show the route accepts some body.

    The leg still untested is the Vite dev proxy in front of the app
    (`web/vite.config.ts`, which #2206 owns and this round may not edit): nothing
    here proves a body-carrying DELETE survives that hop. It goes to the owed
    post-landing browser revoke, where a proxy that dropped the body would show up
    as exactly this route's 400 naming `confirm`.
    """
    key, media = _client_request_shape(WEB_API)
    content = json.dumps({key: "studio-mini"}, separators=(",", ":")).encode()
    r = await _delete("studio-mini", content=content, headers={"content-type": media})
    assert r.status_code == 204, (
        f"the shipped client's own request shape was refused: {r.status_code} {r.text[:160]}")
    assert set(_entries(fixture_cert_dir)) == {"hallway-pi"}, "the accepted revoke deleted nothing"

    base_text = _base_blob(WEB_API)
    start = base_text.find("revokeClient: async (name: string)")
    assert start >= 0, "the base blob has no revokeClient to control against"
    base_body = base_text[start:base_text.find("\n  },", start)]
    assert "body:" not in base_body, (
        "the pre-change client already built a body, so the control below proves "
        f"nothing: {base_body!r}")
    r2 = await _delete("hallway-pi", content=None, headers=None)
    assert r2.status_code == 400, (
        f"the request the base client built is accepted again — the confirm check is "
        f"not enforcing: {r2.status_code}")
    assert set(_entries(fixture_cert_dir)) == {"hallway-pi"}, "a refused revoke deleted an entry"


def test_the_web_client_sends_the_confirm_this_route_demands():
    """Clause 4, pinned where the gate can always run it.

    `web/src/api.test.ts` is the real behaviour test and stays the clause's named
    file, but vitest is skipped wherever `web/node_modules` is absent — the gate's
    own snapshot is such a tree, which is why the first review could only *read*
    this seam, and why its second attempt downgraded the clause for a node id that
    lived in a file its runner never collects. The route is the half that now
    refuses a request, so this file — which the diff changes and pytest always
    collects — asserts the other half's shipped bytes: the method is DELETE, the path is
    percent-encoded while the body carries the raw name (Starlette hands the handler
    the decoded `{name}`, so `confirm: encodeURIComponent(name)` would mismatch on
    any name needing escaping), the content type is JSON, and the key the browser
    writes is the key `_confirm_field` reads.

    The control is the pre-change client, read from the base blob: it sent no body,
    and it must fail this node — otherwise every assertion above would be matching
    something that was already true.
    """
    body = _client_fn(WEB_API, "revokeClient: async (name: string)")
    assert re.search(r"method:\s*'DELETE'", body), f"no DELETE in {body!r}"
    assert re.search(r"'Content-Type':\s*'application/json'", body), (
        "the revoke sends no JSON content type — the route's own reader answers "
        f"None when the media type is not json: {body!r}")
    assert re.search(r"body:\s*JSON\.stringify\(\{\s*confirm:\s*name\s*\}\)", body), (
        "the revoke body does not carry the raw `name` under `confirm`: "
        f"{body!r}")
    assert "encodeURIComponent(name)" in body, (
        "the path no longer escapes the name, so a name outside NAME_RE could "
        f"reach the router as a path: {body!r}")
    assert "confirm: encodeURIComponent(name)" not in body, (
        "the confirm value is escaped while the route compares it against the "
        "already-decoded path parameter — they would never match for an escaping name")

    base_text = _base_blob(WEB_API)
    start = base_text.find("revokeClient: async (name: string)")
    assert start >= 0, "the base blob has no revokeClient to control against"
    base_body = base_text[start:base_text.find("\n  },", start)]
    assert "confirm" not in base_body and "body:" not in base_body, (
        "the pre-change client already sent a confirm body, so the assertions above "
        f"prove nothing: {base_body!r}")
