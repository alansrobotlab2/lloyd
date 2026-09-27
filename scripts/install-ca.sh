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
#   bash scripts/install-ca.sh --check              # compare the store to the CA, write nothing
#   bash scripts/install-ca.sh --check /path/ca.crt # ... against a named CA
#
# OPERATOR STEP: run `bash scripts/install-ca.sh --check` after ANY CA re-mint, on
# every box that has the browser arm. `scripts/gen-cert.sh` is the only caller of
# this script (scripts/gen-cert.sh:50), and normal ops never run it, so a CA minted
# out of band leaves the old CA trusted and the new one untrusted with nothing to
# say so. That is what happened on goliath on 2026-09-22: the CA was re-minted,
# nickname 'Lloyd CA' kept holding the retired key, and the browser arm's fallback
# leaf (web/vite.config.ts serves lloyd.crt whenever no Tailscale cert is present)
# verified nowhere — visible only as a failed navigation (#1668, #1241). `--check`
# exits 0 when the store holds the CA this script would install and 1 naming both
# fingerprints when it holds a different key or no such nickname at all.
#
# `--check` reads the NSS store and nothing else: it never writes a store, and it
# never reads or polices /etc/ca-certificates/**, whose Lloyd_CA.p11-kit /
# cadir entries are machine-wide state that #1241's ruling of 2026-09-27 declined
# to touch (#1668 owed 3).
#
# Where the trust store lives: $LLOYD_NSS_DB when set, else $HOME/.pki/nssdb — the
# NSS shared database Chromium reads on Linux. The override exists so
# tests/test_gen_cert_ca_install.py can exercise the whole step against a temp store
# without ever touching a real ~/.pki/nssdb. It is honoured in --check too, which is
# how that suite runs the check without resolving to this user's store.
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
NICK="Lloyd CA"
TRUST="CT,C,C"

# `--check` is a mode, not a second script: it shares the argument parsing, the
# store resolution and the CA selection with the install path, because the drift it
# detects is precisely "what this script would install" vs "what the store holds".
CHECK=0
CA_CRT=""
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    -*)
      echo "[install-ca] ERROR: unknown argument $arg (usage: install-ca.sh [--check] [ca.crt])" >&2
      exit 2
      ;;
    *) CA_CRT="$arg" ;;
  esac
done
CA_CRT="${CA_CRT:-${LLOYD_CERT_DIR:-$REPO/agent-services/cert}/ca.crt}"

if ! command -v certutil >/dev/null 2>&1; then
  # A box without nss-tools / libnss3-tools cannot hold an NSS store at all. Warn and
  # succeed: the caller (scripts/gen-cert.sh) has its own job to finish, and failing
  # here would take cert minting down with a missing optional package.
  echo "[install-ca] WARNING: certutil is not on PATH (Debian/Ubuntu: libnss3-tools, Fedora: nss-tools)." >&2
  if [[ "$CHECK" == 1 ]]; then
    # `--check` exists to answer "is the trusted CA stale?", and a store that cannot
    # be read answers nothing. Exiting 0 here would be the same defect this script is
    # about: a guard reporting the verdict it has no way to justify.
    echo "[install-ca] CHECK INCONCLUSIVE: no certutil, so no NSS store could be read —" >&2
    echo "[install-ca]          reporting no verdict rather than 'up to date'." >&2
    exit 2
  fi
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
  if [[ "$CHECK" == 1 ]]; then
    exit 2
  fi
  exit 0
fi
DB="sql:$NSS_DB"

if [[ ! -f "$CA_CRT" ]]; then
  echo "[install-ca] ERROR: no CA certificate at $CA_CRT (run scripts/gen-cert.sh first)" >&2
  exit 1
fi

der_sha() { sha256sum | awk '{ print $1 }'; }
# The human-readable form, which is what an operator comparing two boxes needs and
# what #1668's own check was run with: `openssl x509 -noout -fingerprint -sha256`.
# Parsed by parameter expansion so one openssl process answers it: the suite runs
# these scripts with PATH stripped to tests/test_gen_cert_ca_install.py::TOOLCHAIN,
# and every tool this branch reaches for beyond `openssl`, `sha256sum` and `awk`
# would be another entry that tuple has to keep carrying.
pem_fp() {
  local line
  line="$(openssl x509 -noout -fingerprint -sha256 2>/dev/null)" || return 0
  printf '%s\n' "${line##*=}"
}
WANT_SHA="$(openssl x509 -in "$CA_CRT" -outform DER | der_sha)"
WANT_FP="$(openssl x509 -in "$CA_CRT" | pem_fp)"

if [[ "$CHECK" == 1 ]]; then
  # Read-only by construction: this branch returns before the `mkdir -p`, the
  # `certutil -N` and the `-D`/`-A` below, so on a fresh store it leaves no cert9.db
  # and on a store holding a stale CA it leaves that entry's bytes and trust bits
  # exactly as it found them (#1668 clause 3).
  if certutil -L -d "$DB" -n "$NICK" >/dev/null 2>&1; then
    HAVE_SHA="$(certutil -L -d "$DB" -n "$NICK" -r | der_sha)"
    HAVE_FP="$(certutil -L -d "$DB" -n "$NICK" -a | pem_fp)"
    if [[ "$HAVE_SHA" == "$WANT_SHA" ]]; then
      echo "[install-ca] CHECK OK: '$NICK' in $NSS_DB is $CA_CRT"
      echo "[install-ca]   stored   sha256 $HAVE_FP"
      echo "[install-ca]   expected sha256 $WANT_FP"
      exit 0
    fi
    echo "[install-ca] CHECK FAILED: '$NICK' in $NSS_DB is a different certificate than $CA_CRT" >&2
    echo "[install-ca]   stored   sha256 $HAVE_FP" >&2
    echo "[install-ca]   expected sha256 $WANT_FP" >&2
    echo "[install-ca]   Run: bash scripts/install-ca.sh $CA_CRT" >&2
    exit 1
  fi
  echo "[install-ca] CHECK FAILED: no '$NICK' entry in $NSS_DB (store absent, empty, or unreadable)" >&2
  echo "[install-ca]   stored   sha256 none (no such nickname)" >&2
  echo "[install-ca]   expected sha256 $WANT_FP" >&2
  echo "[install-ca]   Run: bash scripts/install-ca.sh $CA_CRT" >&2
  exit 1
fi

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
