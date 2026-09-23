#!/usr/bin/env bash
# Generate a self-signed CA and server cert for Lloyd MC mTLS.
#
# Outputs:
#   agent-services/cert/ca.crt          (public CA cert — install on every device)
#   agent-services/cert/ca.key          (CA private key, chmod 600 — keep on server)
#   agent-services/cert/lloyd.crt       (server cert, signed by CA)
#   agent-services/cert/lloyd.key       (server key, chmod 600)
#   agent-services/cert/clients.json    (empty allowlist — minted by mint-client-cert.sh)
#   agent-services/cert/clients/        (per-device cert bundles)
#
# Usage:
#   bash scripts/gen-cert.sh                # idempotent — skip if files exist
#   bash scripts/gen-cert.sh --force        # regenerate CA + server (invalidates ALL existing client certs)
#   bash scripts/gen-cert.sh --print-sans   # print the SAN string the server leaf would get; write nothing
#
# Extra SANs for server cert (e.g. WAN hostname):
#   LLOYD_CERT_EXTRA_SANS="DNS:lloyd.example.com,IP:1.2.3.4" bash scripts/gen-cert.sh
#
# Where the cert dir lives: agent-services/cert, or $LLOYD_CERT_DIR when set —
# tests/test_gen_cert_sans.py mints a throwaway CA + leaf into a temp tree with it,
# so a test can never reach the live agent-services/cert.
#
# Trust: the "install on every device" above was advice nobody had automated. Every
# path of this script that leaves a CA on disk now hands it to scripts/install-ca.sh,
# which puts it in the invoking user's NSS store (~/.pki/nssdb, nickname "Lloyd CA").
# That store is what lets Chromium load the frontend whenever vite serves this private
# leaf — web/vite.config.ts prefers a Tailscale-issued cert for the MagicDNS name and
# falls back to lloyd.crt when there is none, and agent_mcp/browser.py passes no
# ignore_https_errors, so an unprovisioned store means
# net::ERR_CERT_AUTHORITY_INVALID on that path (backlog #1241). --print-sans returns
# before this step: inspecting what a mint would contain must never change a device's
# trust. The machine-wide half — a real anchor under
# /etc/ca-certificates/trust-source/anchors/ plus update-ca-trust — needs root and is
# not done here.
#
# The tailnet is deliberately part of the server SAN set. lloyd-frontend is
# reached over the tailnet (the "mTLS dropped 2026-06-14" comment in
# web/vite.config.ts makes the tailnet the access boundary), so a leaf naming only
# localhost/hostname/LAN-IP makes every off-localhost client fail a hard TLS name
# check — curl exit 60 with the CA installed, which is why installing the CA on a
# device never helped (backlog #1045). Both the Tailscale IPv4 and the ts.net
# MagicDNS name are read from `tailscale` below; a host that is not on a tailnet
# mints exactly the SANs it minted before.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_CA="$SCRIPT_DIR/install-ca.sh"
CERT_DIR="${LLOYD_CERT_DIR:-$REPO/agent-services/cert}"
CA_CRT="$CERT_DIR/ca.crt"
CA_KEY="$CERT_DIR/ca.key"
SRV_CRT="$CERT_DIR/lloyd.crt"
SRV_KEY="$CERT_DIR/lloyd.key"
CLIENTS_DIR="$CERT_DIR/clients"
ALLOWLIST="$CERT_DIR/clients.json"

FORCE=0
PRINT_SANS=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --print-sans) PRINT_SANS=1 ;;
    *)
      echo "[gen-cert] unknown argument: $arg (expected --force or --print-sans)" >&2
      exit 2
      ;;
  esac
done

# ── Server SAN set ────────────────────────────────────────────────────────
# Computed before the cert dir is created or its files checked, so --print-sans
# below can answer without touching anything, and shared with the signing step,
# so the printed string IS the string the leaf receives.
HOSTNAME_FQDN="${HOSTNAME:-$(uname -n)}"
LAN_IP="$(ip -4 -o route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") { print $(i+1); exit }}')"
if [[ -z "$LAN_IP" ]]; then
  echo "[gen-cert] WARNING: could not auto-detect LAN IP" >&2
fi

# Tailscale IPv4 — the address tailnet clients actually connect to ("ss -tnp" on
# this host shows :5173 sockets whose local address is it). Empty means no
# tailscale binary, an unreachable socket, or a logged-out node: no tailnet, so
# no SAN, exactly as before this existed.
TS_IP="$(tailscale ip -4 2>/dev/null \
  | sed -n -E 's/^[[:space:]]*([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})[[:space:]]*$/\1/p' \
  | head -1 || true)"

# The MagicDNS names this tailnet will issue Let's Encrypt certs for — i.e.
# exactly the names `tailscale cert <name>` accepts, so exactly the names worth
# naming in the leaf. An empty list means tailnet HTTPS is off for this node.
TS_STATUS="$(tailscale status --json 2>/dev/null || true)"
TS_CERT_DOMAINS=""
if [[ -n "$TS_STATUS" ]]; then
  if TS_CERT_DOMAINS="$(printf '%s\n' "$TS_STATUS" \
      | python3 -c 'import json,sys; [print(n) for n in json.load(sys.stdin).get("CertDomains") or []]' 2>/dev/null)"; then
    if [[ -z "$TS_CERT_DOMAINS" ]]; then
      echo "[gen-cert] NOTE: this tailnet issues no Let's Encrypt certs (CertDomains is empty), so the leaf will not name a ts.net hostname." >&2
    fi
  else
    echo "[gen-cert] WARNING: could not parse 'tailscale status --json'; the leaf will NOT name a ts.net hostname." >&2
  fi
fi

SANS="DNS:localhost,DNS:${HOSTNAME_FQDN},IP:127.0.0.1"
if [[ -n "$LAN_IP" ]]; then
  SANS="${SANS},IP:${LAN_IP}"
fi
if [[ -n "$TS_IP" ]]; then
  SANS="${SANS},IP:${TS_IP}"
fi
while IFS= read -r ts_name; do
  if [[ -n "$ts_name" ]]; then
    SANS="${SANS},DNS:${ts_name}"
  fi
done <<<"$TS_CERT_DOMAINS"
if [[ -n "${LLOYD_CERT_EXTRA_SANS:-}" ]]; then
  SANS="${SANS},${LLOYD_CERT_EXTRA_SANS}"
fi

if [[ $PRINT_SANS -eq 1 ]]; then
  # No mkdir, no existence check, no --force invalidation warning, no write of any
  # kind: inspecting what a mint WOULD contain must never be able to invalidate a
  # device's trust (that is what --force does at the client allowlist).
  printf '%s\n' "$SANS"
  exit 0
fi

mkdir -p "$CERT_DIR" "$CLIENTS_DIR"

if [[ -f "$CA_CRT" && -f "$CA_KEY" && -f "$SRV_CRT" && -f "$SRV_KEY" && $FORCE -eq 0 ]]; then
  echo "[gen-cert] CA + server cert already exist — skipping (pass --force to regenerate)"
  echo "          CA fingerprint:"
  openssl x509 -in "$CA_CRT" -noout -fingerprint -sha256
  # Provision trust here as well as after the mint, and before this exit: this is the
  # path every box that was ever set up takes, so a step appended at the bottom of the
  # file would never run on exactly the machines that still need it (#1241 — a new
  # user, or a rebuilt store, on a host whose certs are already in place).
  bash "$INSTALL_CA" "$CA_CRT"
  exit 0
fi

if [[ $FORCE -eq 1 && -d "$CLIENTS_DIR" && -n "$(ls -A "$CLIENTS_DIR" 2>/dev/null)" ]]; then
  echo "[gen-cert] WARNING: --force will invalidate every existing client cert in $CLIENTS_DIR"
  echo "          Existing client certs are signed by the OLD CA and will be rejected after this."
fi

echo "[gen-cert] hostname:    $HOSTNAME_FQDN"
echo "[gen-cert] LAN IP:      ${LAN_IP:-<none>}"
if [[ -n "$TS_IP" ]]; then
  echo "[gen-cert] tailnet IP:  $TS_IP"
fi
if [[ -n "$TS_CERT_DOMAINS" ]]; then
  echo "[gen-cert] cert names:  ${TS_CERT_DOMAINS//$'\n'/ }"
fi
echo "[gen-cert] server SANs: $SANS"

# ── CA ────────────────────────────────────────────────────────────────────
echo "[gen-cert] generating CA…"
openssl genrsa -out "$CA_KEY" 4096 2>/dev/null
chmod 600 "$CA_KEY"
openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 \
  -out "$CA_CRT" \
  -subj "/CN=Lloyd CA" \
  -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
  -addext "keyUsage=critical,keyCertSign,cRLSign" 2>/dev/null
chmod 644 "$CA_CRT"

# ── Server cert ───────────────────────────────────────────────────────────
echo "[gen-cert] generating server cert…"
SRV_CONF="$(mktemp)"
SRV_CSR="$(mktemp)"
trap 'rm -f "$SRV_CONF" "$SRV_CSR"' EXIT

cat >"$SRV_CONF" <<EOF
[req]
distinguished_name = dn
prompt = no
req_extensions = v3_req

[dn]
CN = lloyd

[v3_req]
basicConstraints = critical,CA:FALSE
keyUsage         = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName   = ${SANS}
EOF

openssl genrsa -out "$SRV_KEY" 2048 2>/dev/null
chmod 600 "$SRV_KEY"
openssl req -new -key "$SRV_KEY" -out "$SRV_CSR" -config "$SRV_CONF" 2>/dev/null
# 397 days, NOT 3650: Apple platforms (iOS 13.4+/Safari) reject TLS *server*
# certs with validity > 398 days as invalid ("this connection is not private"),
# even when the signing CA is trusted. The CA itself (above) is exempt and stays
# long-lived, so this leaf can be re-minted without re-installing the CA on devices.
openssl x509 -req -in "$SRV_CSR" \
  -CA "$CA_CRT" -CAkey "$CA_KEY" -CAcreateserial \
  -out "$SRV_CRT" -days 397 -sha256 \
  -extfile "$SRV_CONF" -extensions v3_req 2>/dev/null
chmod 644 "$SRV_CRT"

# Initialise allowlist if missing
if [[ ! -f "$ALLOWLIST" ]]; then
  echo "{}" > "$ALLOWLIST"
fi

echo
echo "[gen-cert] wrote $CA_CRT"
echo "[gen-cert] wrote $CA_KEY"
echo "[gen-cert] wrote $SRV_CRT"
echo "[gen-cert] wrote $SRV_KEY"
echo
echo "CA fingerprint:"
openssl x509 -in "$CA_CRT" -noout -fingerprint -sha256
echo
# The mint path's share of the trust step (the skip branch above does its own). After
# a --force re-mint this is also what retires the previous CA's entry, which would
# otherwise sit in the store still trusted for SSL — see scripts/install-ca.sh.
bash "$INSTALL_CA" "$CA_CRT"
echo
echo "Next: mint at least one client cert before enabling mTLS in Vite, e.g."
echo "  bash scripts/mint-client-cert.sh host-browser"
