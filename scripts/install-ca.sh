#!/usr/bin/env bash
# Provision a Lloyd CA certificate into the invoking user's NSS trust store, so
# Chromium trusts the leaf lloyd-frontend serves.
#
# Backlog #1241. `agent_mcp/browser.py` passes no `ignore_https_errors`, so the
# browser arm verifies whatever certificate the frontend presents — and
# web/vite.config.ts serves this CA's `lloyd.crt` whenever no Tailscale-issued cert
# is present, which is the state of every box that has never run `tailscale cert`.
# Until this script existed, the trust that made that load work was hand-seeded
# machine state: `certutil -A` run by a person, recorded in no script
# (`git grep -ln -E "certutil|nssdb|update-ca-certificates" HEAD` returned no files).
# A rebuilt box therefore got net::ERR_CERT_AUTHORITY_INVALID from the browser arm
# with nothing in the checkout naming the cause. scripts/gen-cert.sh calls this on
# every path of its own, so minting the CA and trusting it are one step.
#
# Usage:
#   bash scripts/install-ca.sh                      # install $LLOYD_CERT_DIR/ca.crt
#   bash scripts/install-ca.sh /path/to/ca.crt      # install a named CA instead
#
# Where the trust store lives: $LLOYD_NSS_DB when set, else $HOME/.pki/nssdb — the
# NSS shared database Chromium reads on Linux. The override exists so
# tests/test_gen_cert_ca_install.py can exercise the whole step against a temp store
# without ever touching a real ~/.pki/nssdb.
#
# Trust flags "CT,C,C" — trusted CA for SSL, e-mail and object signing. That is what
# makes NSS treat the file as a web anchor, and it is the form the hand-seeded store
# on goliath carries.
#
# Rootless by design and sufficient for the browser arm: verified on this box by
# navigating to a throwaway-CA-signed fixture with HOME set to a temp dir whose nssdb
# held nothing but this install. The machine-wide half — a real anchor under
# /etc/ca-certificates/trust-source/anchors/ plus update-ca-trust, so other users and
# browsers on the host trust it too and a trust re-anchor cannot drop the generated
# /etc/ca-certificates/trust-source/Lloyd_CA.p11-kit, which currently has no source —
# needs root and is NOT this script's job (backlog #1241's needs-a-person clause).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CA_CRT="${1:-${LLOYD_CERT_DIR:-$REPO/agent-services/cert}/ca.crt}"
NICK="Lloyd CA"
TRUST="CT,C,C"

if ! command -v certutil >/dev/null 2>&1; then
  # A box without nss-tools / libnss3-tools cannot hold an NSS store at all. Warn and
  # succeed: the caller (scripts/gen-cert.sh) has its own job to finish, and failing
  # here would take cert minting down with a missing optional package.
  echo "[install-ca] WARNING: certutil is not on PATH (Debian/Ubuntu: libnss3-tools, Fedora: nss-tools)." >&2
  echo "[install-ca]          Skipped installing $CA_CRT into the NSS trust store; until it is" >&2
  echo "[install-ca]          installed, Chromium fails certificate verification on the" >&2
  echo "[install-ca]          certificates this CA signs (net::ERR_CERT_AUTHORITY_INVALID)." >&2
  exit 0
fi

# Resolve the store location. HOME is only read when there is no explicit override,
# and an unset HOME with no override is reported rather than guessed at.
if [[ -n "${LLOYD_NSS_DB:-}" ]]; then
  NSS_DB="$LLOYD_NSS_DB"
elif [[ -n "${HOME:-}" ]]; then
  NSS_DB="$HOME/.pki/nssdb"
else
  echo "[install-ca] WARNING: HOME is unset and LLOYD_NSS_DB is not set — no trust store location to install into." >&2
  exit 0
fi
DB="sql:$NSS_DB"

if [[ ! -f "$CA_CRT" ]]; then
  echo "[install-ca] ERROR: no CA certificate at $CA_CRT (run scripts/gen-cert.sh first)" >&2
  exit 1
fi

der_sha() { sha256sum | awk '{ print $1 }'; }
WANT_SHA="$(openssl x509 -in "$CA_CRT" -outform DER | der_sha)"

# NSS needs the directory to exist before it will create a database in it
# (certutil -N otherwise answers SEC_ERROR_BAD_DATABASE), and a fresh user on a
# rebuilt box has no ~/.pki at all.
mkdir -p "$NSS_DB"
if [[ ! -e "$NSS_DB/cert9.db" && ! -e "$NSS_DB/cert8.db" ]]; then
  echo "[install-ca] creating NSS database $NSS_DB"
  certutil -N -d "$DB" --empty-password
fi

if certutil -L -d "$DB" -n "$NICK" >/dev/null 2>&1; then
  # Check before adding. `certutil -A` for a subject the store already holds does not
  # add a row — it overwrites that row's trust bits — so a re-run would silently
  # rewrite whatever trust a person or an earlier install chose. #1241 measured it: a
  # second `-A -t "C,,"` over a `-t "CT,C,C"` entry left one row reading `C,,`.
  HAVE_SHA="$(certutil -L -d "$DB" -n "$NICK" -r | der_sha)"
  if [[ "$HAVE_SHA" == "$WANT_SHA" ]]; then
    echo "[install-ca] $NICK already in $NSS_DB, same certificate — trust attributes left untouched"
    exit 0
  fi
  # Same nickname, different key: this CA was re-minted (`--force`), so the stale
  # certificate has to go before the new one is trusted. Deliberate replacement, not
  # a silent trust rewrite, and it cannot leave a second nickname behind.
  echo "[install-ca] $NICK in $NSS_DB is a different certificate — replacing it"
  certutil -D -d "$DB" -n "$NICK"
fi

certutil -A -d "$DB" -n "$NICK" -t "$TRUST" -i "$CA_CRT"
echo "[install-ca] installed $CA_CRT as '$NICK' with trust $TRUST in $NSS_DB"
certutil -L -d "$DB" | awk -v n="$NICK" 'index($0, n) == 1 { printf "[install-ca]   %s\n", $0 }'
