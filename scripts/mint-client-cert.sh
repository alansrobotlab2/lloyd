#!/usr/bin/env bash
# Mint a client cert for one device, signed by the Lloyd CA, plus a .p12
# bundle ready to import into a browser / OS keystore.
#
# Usage:
#   bash scripts/mint-client-cert.sh <name> [p12-passphrase]
#
# <name> must match [a-zA-Z0-9_-]+ (used as CN and as the filename stem).
# Default passphrase if omitted: lloyd
#
# Outputs (under agent-services/cert/clients/):
#   <name>.crt      PEM cert
#   <name>.key      PEM key (chmod 600)
#   <name>.p12      PKCS#12 bundle for browser/OS import (chmod 600)
#
# Side effects:
#   Adds an entry to agent-services/cert/clients.json with the cert's
#   sha256 fingerprint. The backend uses this allowlist to authorise
#   API requests (Vite enforces CA-signed at TLS layer; backend enforces
#   fingerprint at HTTP layer).

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <name> [p12-passphrase]" >&2
  exit 1
fi

NAME="$1"
PASS="${2:-lloyd}"

if [[ ! "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "error: name must be alphanumeric (with - or _)" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CERT_DIR="$REPO/agent-services/cert"
CA_CRT="$CERT_DIR/ca.crt"
CA_KEY="$CERT_DIR/ca.key"
CLIENTS_DIR="$CERT_DIR/clients"
ALLOWLIST="$CERT_DIR/clients.json"

if [[ ! -f "$CA_CRT" || ! -f "$CA_KEY" ]]; then
  echo "error: CA not found. Run: bash scripts/gen-cert.sh" >&2
  exit 1
fi

mkdir -p "$CLIENTS_DIR"
KEY="$CLIENTS_DIR/$NAME.key"
CRT="$CLIENTS_DIR/$NAME.crt"
P12="$CLIENTS_DIR/$NAME.p12"

if [[ -f "$CRT" || -f "$KEY" || -f "$P12" ]]; then
  echo "error: cert for '$NAME' already exists. Revoke it first or pick a different name." >&2
  exit 1
fi

CSR="$(mktemp)"
CONF="$(mktemp)"
trap 'rm -f "$CSR" "$CONF"' EXIT

cat >"$CONF" <<EOF
[req]
distinguished_name = dn
prompt = no
req_extensions = v3_req

[dn]
CN = $NAME

[v3_req]
basicConstraints = critical,CA:FALSE
keyUsage         = critical,digitalSignature,keyEncipherment
extendedKeyUsage = clientAuth
EOF

openssl genrsa -out "$KEY" 2048 2>/dev/null
chmod 600 "$KEY"
openssl req -new -key "$KEY" -out "$CSR" -config "$CONF" 2>/dev/null
openssl x509 -req -in "$CSR" \
  -CA "$CA_CRT" -CAkey "$CA_KEY" -CAcreateserial \
  -out "$CRT" -days 3650 -sha256 \
  -extfile "$CONF" -extensions v3_req 2>/dev/null
chmod 644 "$CRT"

openssl pkcs12 -export \
  -inkey "$KEY" -in "$CRT" -certfile "$CA_CRT" \
  -name "Lloyd-$NAME" \
  -out "$P12" \
  -passout "pass:$PASS" \
  -macalg sha256 2>/dev/null
chmod 600 "$P12"

# Capture fingerprint (uppercase, colons stripped) for the allowlist
FP="$(openssl x509 -in "$CRT" -noout -fingerprint -sha256 | sed 's/^.*Fingerprint=//' | tr -d ':')"
ISSUED="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# Append to clients.json (jq if available, else python)
if command -v jq >/dev/null 2>&1; then
  TMP="$(mktemp)"
  jq --arg name "$NAME" --arg fp "$FP" --arg issued "$ISSUED" \
    '.[$name] = {fingerprint: $fp, issued_at: $issued}' \
    "$ALLOWLIST" > "$TMP"
  mv "$TMP" "$ALLOWLIST"
else
  python3 - "$ALLOWLIST" "$NAME" "$FP" "$ISSUED" <<'PY'
import json, sys
path, name, fp, issued = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    data = json.load(open(path))
except FileNotFoundError:
    data = {}
data[name] = {"fingerprint": fp, "issued_at": issued}
json.dump(data, open(path, "w"), indent=2)
PY
fi

echo "[mint-client-cert] minted '$NAME'"
echo "  cert:        $CRT"
echo "  key:         $KEY"
echo "  bundle:      $P12  (passphrase: $PASS)"
echo "  fingerprint: $FP"
echo
echo "Next: install $P12 in the device's browser/OS keystore."
