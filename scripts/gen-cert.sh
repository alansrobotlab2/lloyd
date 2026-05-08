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
#   bash scripts/gen-cert.sh           # idempotent — skip if files exist
#   bash scripts/gen-cert.sh --force   # regenerate CA + server (invalidates ALL existing client certs)
#
# Extra SANs for server cert (e.g. WAN hostname):
#   LLOYD_CERT_EXTRA_SANS="DNS:lloyd.example.com,IP:1.2.3.4" bash scripts/gen-cert.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CERT_DIR="$REPO/agent-services/cert"
CA_CRT="$CERT_DIR/ca.crt"
CA_KEY="$CERT_DIR/ca.key"
SRV_CRT="$CERT_DIR/lloyd.crt"
SRV_KEY="$CERT_DIR/lloyd.key"
CLIENTS_DIR="$CERT_DIR/clients"
ALLOWLIST="$CERT_DIR/clients.json"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

mkdir -p "$CERT_DIR" "$CLIENTS_DIR"

if [[ -f "$CA_CRT" && -f "$CA_KEY" && -f "$SRV_CRT" && -f "$SRV_KEY" && $FORCE -eq 0 ]]; then
  echo "[gen-cert] CA + server cert already exist — skipping (pass --force to regenerate)"
  echo "          CA fingerprint:"
  openssl x509 -in "$CA_CRT" -noout -fingerprint -sha256
  exit 0
fi

if [[ $FORCE -eq 1 && -d "$CLIENTS_DIR" && -n "$(ls -A "$CLIENTS_DIR" 2>/dev/null)" ]]; then
  echo "[gen-cert] WARNING: --force will invalidate every existing client cert in $CLIENTS_DIR"
  echo "          Existing client certs are signed by the OLD CA and will be rejected after this."
fi

HOSTNAME_FQDN="${HOSTNAME:-$(uname -n)}"
LAN_IP="$(ip -4 -o route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") { print $(i+1); exit }}')"
if [[ -z "$LAN_IP" ]]; then
  echo "[gen-cert] WARNING: could not auto-detect LAN IP" >&2
fi

SANS="DNS:localhost,DNS:${HOSTNAME_FQDN},IP:127.0.0.1"
if [[ -n "$LAN_IP" ]]; then
  SANS="${SANS},IP:${LAN_IP}"
fi
if [[ -n "${LLOYD_CERT_EXTRA_SANS:-}" ]]; then
  SANS="${SANS},${LLOYD_CERT_EXTRA_SANS}"
fi

echo "[gen-cert] hostname:    $HOSTNAME_FQDN"
echo "[gen-cert] LAN IP:      ${LAN_IP:-<none>}"
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
openssl x509 -req -in "$SRV_CSR" \
  -CA "$CA_CRT" -CAkey "$CA_KEY" -CAcreateserial \
  -out "$SRV_CRT" -days 3650 -sha256 \
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
echo "Next: mint at least one client cert before enabling mTLS in Vite, e.g."
echo "  bash scripts/mint-client-cert.sh host-browser"
