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
    """Tell the caller which client cert they're using (CN + fingerprint)."""
    return JSONResponse({
        "name": getattr(request.state, "client_name", None),
        "fingerprint": getattr(request.state, "client_fingerprint", None),
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

    The cert's keypair files remain on disk for forensics — only the
    allowlist entry is removed, which is what the auth middleware checks.
    """
    if not NAME_RE.match(name):
        raise HTTPException(400, "invalid name")
    clients = _load_clients()
    if name not in clients:
        raise HTTPException(404, f"no client '{name}'")

    # Don't let the caller revoke their own cert (locks them out instantly)
    caller = getattr(request.state, "client_name", None)
    if caller == name:
        raise HTTPException(400, "cannot revoke the cert you're currently using")

    del clients[name]
    _save_clients(clients)

    # Best-effort cleanup of the on-disk material
    for ext in ("crt", "key", "p12"):
        try:
            (CLIENTS_DIR / f"{name}.{ext}").unlink()
        except FileNotFoundError:
            pass

    return Response(status_code=204)
