"""System endpoints — TLS CA download, client cert minting/listing/revoking, LAN info."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response


router = APIRouter()


REPO_ROOT = Path(__file__).resolve().parents[2]
CERT_DIR = REPO_ROOT / "agent-services" / "cert"
CA_CERT = CERT_DIR / "ca.crt"
CLIENTS_DIR = CERT_DIR / "clients"
CLIENTS_JSON = CERT_DIR / "clients.json"
MINT_SCRIPT = REPO_ROOT / "scripts" / "mint-client-cert.sh"

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

#: The commit that dropped client-certificate auth (2026-06-14: iOS Chrome cannot
#: present a keychain identity). Everything below that uses the word "verified"
#: means it in the transport sense, and this sha is the date the transport stopped
#: offering it — named rather than paraphrased so a reader can re-check the claim
#: the day someone re-enables mTLS and `verified` starts coming back true.
MTLS_DROP_SHA = "5e1351f3"

#: Why a revoke cannot be authorised by who is asking. No server on this box
#: requests a client certificate (`web/vite.config.ts` carries `key`/`cert` only —
#: no `requestCert`, no `ca`), so `scope["client_cert"]` is never populated and
#: `ApiPeerGate` derives a name from the `x-client-fingerprint` header alone
#: (`server._cert_fingerprint`), looked up in the very file these endpoints edit.
#: A name a caller chooses cannot authorise an action on the record it names.
NO_VERIFIED_CALLER = (
    "there is no verified caller identity since " + MTLS_DROP_SHA
    + " — nothing requests a client certificate, so the only name a caller can "
    "present is its own x-client-fingerprint header"
)

#: The refusal a revoke gives without a usable `confirm`, and the only check on
#: that route the server performs rather than trusts.
REVOKE_CONFIRM_DETAIL = (
    'revoke requires a JSON body whose "confirm" field names this client; '
    + NO_VERIFIED_CALLER
    + ", so the confirm body is what the server can enforce"
)


def _self_revoke_detail(name: str) -> str:
    """Why the caller-comparison refusal says what it says.

    Kept as a function so the sentence and the guard cannot drift apart: the
    comparison is real (`tests/test_api_client_gating.py::test_an_enrolled_device_is_still_named_to_the_route`
    proves the name does arrive), and what was wrong about it was ever being called
    a protection.
    """
    return (
        f"refusing to revoke '{name}': the name this caller arrives under is the "
        "same one, and since " + MTLS_DROP_SHA + " that name is whatever the "
        "caller's own x-client-fingerprint header says rather than anything the "
        "server verified — so this stops an accidental self-revoke and has never "
        "stopped a deliberate one"
    )


def _cert_attested(request: Request) -> bool:
    """Was this caller's identity produced by the transport rather than by them?

    Reads the one place a TLS-terminating server could hand the app a verified peer
    certificate: the ``client_cert`` scope key. It is a convention this app defines,
    not one this stack implements — uvicorn 0.44.0 and Starlette 1.3.1 never mention
    it (`grep -rn --include=*.py client_cert
    .venvs/lloyd/lib/python3.12/site-packages/uvicorn/
    .venvs/lloyd/lib/python3.12/site-packages/starlette/` → 0 hits), and no front end
    in front of this app terminates TLS with a CA of ours. So the answer is False for
    every request this box serves today, for the transport reason in
    `NO_VERIFIED_CALLER` rather than because someone typed a constant.

    It is written as a read for exactly that reason: a literal `False` would keep
    saying "not verified" on the day mTLS came back and quietly turn the response
    field into decoration, which is the same dishonesty in the other direction.
    Whether `verified` should ever be reachable — whether mTLS returns at all, which
    `web/vite.config.ts` records as impossible for iOS Chrome — is a ruling this
    route does not make. Both edges are pinned by
    `tests/test_system_identity_honesty.py::test_verified_is_read_from_the_certificate_field_not_hardcoded`.
    """
    return request.scope.get("client_cert") is not None


async def _confirm_field(request: Request) -> str | None:
    """The body's `confirm` value, or None when the body never answers the question.

    A body that is present but unreadable is refused here rather than returned as
    None, so "I sent you something" and "I sent nothing" share one refusal instead
    of one of them surfacing as a 500 out of the decoder.
    """
    raw = await request.body()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(400, REVOKE_CONFIRM_DETAIL) from None
    if not isinstance(parsed, dict):
        raise HTTPException(400, REVOKE_CONFIRM_DETAIL)
    value = parsed.get("confirm")
    return value if isinstance(value, str) else None


def _load_clients() -> dict[str, dict]:
    try:
        return json.loads(CLIENTS_JSON.read_text() or "{}")
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_clients(data: dict) -> None:
    CLIENTS_JSON.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(CLIENTS_JSON, 0o600)
    except OSError:
        pass


def _detect_lan_ip() -> str | None:
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "route", "get", "1.1.1.1"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        parts = out.split()
        if "src" in parts:
            return parts[parts.index("src") + 1]
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("1.1.1.1", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return None


@router.get("/api/system/lan-info")
async def lan_info():
    ip = _detect_lan_ip()
    return JSONResponse({
        "lan_ip": ip,
        "hostname": os.uname().nodename,
        "https_url": f"https://{ip}:5173/" if ip else None,
        "ca_available": CA_CERT.exists(),
    })


@router.get("/api/system/identity")
async def identity(request: Request):
    """Report the name this caller is known by, and whether that is a fact.

    `name`/`fingerprint` are the values `ApiPeerGate` derived from the caller's own
    `x-client-fingerprint` header, and `verified` is the answer to the question the
    rest of this response used to leave implied: the server checked nothing, so the
    caller is introducing itself. False for a header-bearing caller and for one
    that sent nothing, and it stays False for every caller until something in front
    of this app asks for a certificate — see `_cert_attested` and `MTLS_DROP_SHA`.
    """
    return JSONResponse({
        "name": getattr(request.state, "client_name", None),
        "fingerprint": getattr(request.state, "client_fingerprint", None),
        "verified": _cert_attested(request),
    })


@router.get("/api/system/cert/ca")
async def download_ca():
    if not CA_CERT.exists():
        raise HTTPException(404, "CA cert not found. Run: bash scripts/gen-cert.sh")
    return FileResponse(
        path=str(CA_CERT),
        media_type="application/x-x509-ca-cert",
        filename="lloyd-ca.crt",
    )


@router.get("/api/system/clients")
async def list_clients():
    data = _load_clients()
    return JSONResponse({
        "clients": [
            {"name": name, **entry} for name, entry in sorted(data.items())
        ],
    })


@router.post("/api/system/clients")
async def mint_client(request: Request):
    """Mint a new client cert. Returns the .p12 bundle inline as base64 + metadata.

    Request body: {"name": "<device-name>", "passphrase": "<optional>"}
    """
    body = await request.json() if (await request.body()) else {}
    name = (body.get("name") or "").strip()
    passphrase = (body.get("passphrase") or "lloyd").strip() or "lloyd"

    if not name or not NAME_RE.match(name):
        raise HTTPException(400, "name must be alphanumeric (with - or _)")
    if name in _load_clients():
        raise HTTPException(409, f"client '{name}' already exists — revoke it first")
    if not MINT_SCRIPT.exists():
        raise HTTPException(500, f"mint script not found at {MINT_SCRIPT}")

    proc = subprocess.run(
        ["bash", str(MINT_SCRIPT), name, passphrase],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        raise HTTPException(500, f"mint failed: {proc.stderr.strip() or proc.stdout.strip()}")

    p12_path = CLIENTS_DIR / f"{name}.p12"
    if not p12_path.exists():
        raise HTTPException(500, "mint succeeded but .p12 not found")
    entry = _load_clients().get(name, {})

    return JSONResponse({
        "name": name,
        "fingerprint": entry.get("fingerprint"),
        "issued_at": entry.get("issued_at"),
        "passphrase": passphrase,
        "p12_url": f"/api/system/clients/{name}/p12",
    })


@router.get("/api/system/clients/{name}/p12")
async def download_client_p12(name: str):
    if not NAME_RE.match(name):
        raise HTTPException(400, "invalid name")
    p12 = CLIENTS_DIR / f"{name}.p12"
    if not p12.exists():
        raise HTTPException(404, f"no .p12 for '{name}'")
    return FileResponse(
        path=str(p12),
        media_type="application/x-pkcs12",
        filename=f"lloyd-{name}.p12",
    )


@router.delete("/api/system/clients/{name}")
async def revoke_client(name: str, request: Request):
    """Revoke a client cert by removing its fingerprint from the allowlist.

    The allowlist entry is what `ApiPeerGate` checks, so removing it is the
    revocation; the `clients/{name}.crt`/`.key`/`.p12` files are then unlinked
    best-effort, which is the part that is not recoverable from this endpoint.

    The body must carry `confirm` equal to `{name}`. That is the enforceable
    check on this route, and the only one: `NO_VERIFIED_CALLER` spells out why a
    caller cannot be authorised by identity here. The browser's `confirm()` dialog
    in Settings is a courtesy in front of it, not the guard.
    """
    if not NAME_RE.match(name):
        raise HTTPException(400, "invalid name")

    # First, because it is the one thing on this route the server decides. The
    # path already named the client; the body has to name it again, so a request
    # that deletes a device has to state which one it means twice, in two places
    # a mis-clicked UI tab does not both control.
    if await _confirm_field(request) != name:
        raise HTTPException(400, REVOKE_CONFIRM_DETAIL)

    clients = _load_clients()
    if name not in clients:
        raise HTTPException(404, f"no client '{name}'")

    # Advisory only, and honest about being advisory: this compares the target
    # against the name this caller's `x-client-fingerprint` header resolved to. No
    # verified caller identity has existed since 5e1351f3, so that name is a claim
    # the caller makes, and withholding the header walks straight past this branch.
    # It stays because the accident it catches — a device revoking the entry it is
    # signed with, locking its own owner out mid-session — is still worth refusing
    # on the honest path. The check that protects the file is the one above.
    caller = getattr(request.state, "client_name", None)
    if caller == name:
        raise HTTPException(400, _self_revoke_detail(name))

    del clients[name]
    _save_clients(clients)

    # Best-effort cleanup of the on-disk material
    for ext in ("crt", "key", "p12"):
        try:
            (CLIENTS_DIR / f"{name}.{ext}").unlink()
        except FileNotFoundError:
            pass

    return Response(status_code=204)
