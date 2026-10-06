#!/usr/bin/env bash
# Renew the tailnet leaf that Vite prefers for the MC frontend (#1727).
#
# `goliath.taile37041.ts.net.crt` is a 90-day Let's Encrypt leaf handed out by
# `tailscale cert`, and web/vite.config.ts selects it by `fs.existsSync` rather
# than by validity — so when it expires Vite keeps serving the dead leaf instead
# of falling back to the still-valid `lloyd.crt`, and every tailnet client fails
# TLS. Nothing renewed it: no timer, no cron, no supervisor program. This script
# is the owner; agent-services/systemd/lloyd-cert-renew.timer runs it daily.
#
# What it does, in order:
#   1. Read the leaf's notAfter with `openssl x509 -enddate -noout`.
#   2. If that is more than CERT_RENEW_WINDOW_DAYS away, exit 0 touching nothing.
#   3. Otherwise mint a replacement with `tailscale cert` into a staging tree
#      BESIDE the cert dir, so no unverified byte can reach it, and only publish
#      the pair by renaming it into place once the new leaf parses and its
#      notAfter is later than the one on disk.
#   4. Restart the frontend, which is what Vite needs to pick the new pair up —
#      it reads the cert at startup (SETUP.md:1513-1514).
#   5. When the mint did NOT produce a replacement and the pair still on disk is
#      inside CERT_QUARANTINE_WITHIN_SECONDS (24 h by default) of its notAfter,
#      move that pair out of the cert dir to $CERT_DIR/../.expired-$HOST/ and
#      restart the frontend once, then exit non-zero anyway. Vite selects the
#      tailnet pair by `fs.existsSync` alone (web/vite.config.ts:20), so a dead
#      pair left standing in the cert dir is a dead leaf served tailnet-wide and
#      the valid `lloyd.crt` fallback is never reached; taking the pair out is
#      the only lever that selection responds to. The next daily run then lands
#      in the "no leaf at all" branch below and mints a replacement. (#2209)
#
# Failure paths never overwrite a byte: a refused mint, a zero-byte file, an
# unparseable leaf, or a minted notAfter that is not later than the current one
# all leave the existing .crt/.key bytes exactly as they were and exit non-zero,
# which is what makes `systemctl --user --failed` (and the journal, which is
# where the unit's stderr goes) the read-only surface that says the renewal did
# not happen. The one thing a failed mint may still do is the #2209 quarantine in
# step 5, and it MOVES rather than deletes: the pair's bytes stay on disk under
# .expired-$HOST beside the cert dir, for the next run to mint over and for a
# human to read.
# It never truncates or replaces in place, never re-mints the CA (that
# is scripts/gen-cert.sh's job), never writes under /etc/ca-certificates/ and
# never runs update-ca-trust (#1241's ruling).
#
# Minting needs to be the tailscale operator. On this box `tailscale debug prefs`
# already reports OperatorUser=$USER (SETUP.md:1504 granted it once). If a mint
# comes back refused for want of that, this script says so in one line naming
# `sudo tailscale set --operator=$USER` and exits non-zero — it does not
# attempt sudo itself, so the run record states whether the owner is the timer
# or a human.
#
# Everything the script touches is overridable for the tests in
# tests/test_renew_tailnet_cert.py: CERT_DIR, TAILNET_CERT_HOST,
# CERT_RENEW_WINDOW_DAYS, CERT_QUARANTINE_WITHIN_SECONDS, SUPERVISORD_CONF,
# MC_FRONTEND_PROGRAM.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CERT_DIR="${CERT_DIR:-$REPO/agent-services/cert}"
HOST="${TAILNET_CERT_HOST:-goliath.taile37041.ts.net}"
WINDOW_DAYS="${CERT_RENEW_WINDOW_DAYS:-14}"
# How close to notAfter a pair this script failed to renew has to be before it is
# moved out of the cert dir. 24 h, deliberately far tighter than WINDOW_DAYS:
# inside the window a mint is merely worth attempting, whereas a leaf a day from
# expiry is one Vite will still be serving when it stops validating.
QUARANTINE_WITHIN="${CERT_QUARANTINE_WITHIN_SECONDS:-86400}"
SUPERVISORD_CONF="${SUPERVISORD_CONF:-$REPO/agent-services/supervisor/supervisord.conf}"
MC_PROGRAM="${MC_FRONTEND_PROGRAM:-lloyd-mc:lloyd-frontend}"
SUPERVISORCTL="${SUPERVISORCTL:-supervisorctl}"
OPENSSL="${OPENSSL:-openssl}"

CRT="$CERT_DIR/$HOST.crt"
KEY="$CERT_DIR/$HOST.key"
# A systemd user unit's environment carries HOME and LOGNAME but not USER, and
# `set -u` would kill the one message that tells an operator how to fix a
# refused mint. Ask the kernel instead of trusting the variable.
WHOAMI="${USER:-$(id -un)}"

warn() { echo "renew-tailnet-cert: $*" >&2; }
info() { echo "renew-tailnet-cert: $*"; }

# The leaf's notAfter as an epoch, or non-zero when it cannot be read. The
# command is the one the operator would type, so a leaf this cannot parse is one
# Vite will serve broken and no window arithmetic is safe on it.
not_after_epoch() {
  local file="$1" raw
  [[ -s "$file" ]] || return 1
  raw="$("$OPENSSL" x509 -enddate -noout -in "$file" 2>/dev/null)" || return 1
  raw="${raw#notAfter=}"
  date -d "$raw" +%s 2>/dev/null || return 1
}

# The fallback for a mint that did not happen (#2209). Vite selects the tailnet
# leaf by `fs.existsSync` alone (web/vite.config.ts:20), so while a dead pair sits
# in the cert dir it keeps being served tailnet-wide and the still-valid
# `lloyd.crt` is never reached — removing the pair is the only lever Vite responds
# to. It is MOVED, never deleted, and moved BESIDE the cert dir for the same
# reason the staging tree is (see STAGE below): a rename across filesystems is a
# copy, and a half-written key at a live path is worse than the dead leaf. Then
# the frontend restarts exactly once, because Vite reads the cert at startup and
# a fallback nobody restarted is still the dead leaf being served. The caller
# still exits non-zero: the renewal did not happen and the unit has to say so.
#
# A pair with more than QUARANTINE_WITHIN left is deliberately not moved — Vite is
# serving it correctly today and the next daily run gets another chance — and so
# is an absent pair, which is the case Vite has already fallen back for.
quarantine_dead_leaf() {
  if [[ "$HAVE_LEAF" != 1 ]]; then
    return 0
  fi
  if ! (( CURRENT < NOW + QUARANTINE_WITHIN )); then
    return 0
  fi
  local qdir="$CERT_DIR/../.expired-$HOST" moved="" f dest
  if ! mkdir -p "$qdir"; then
    warn "cannot create the quarantine dir $qdir; leaving the dead leaf where it is"
    return 0
  fi
  # Resolve it before logging, so the journal names one unambiguous place rather
  # than a path with `..` in it.
  qdir="$(cd "$qdir" && pwd -P)" || qdir="$CERT_DIR/../.expired-$HOST"
  for f in "$CRT" "$KEY"; do
    [[ -e "$f" ]] || continue
    dest="$qdir/${f##*/}"
    if mv -f "$f" "$dest"; then
      moved="$moved $f -> $dest"
    else
      # A half-finished move still has to be reported: one file left behind is
      # enough for Vite to fall back, so this is not a swallowed failure — the
      # restart below is what makes the fallback real, and the journal says which
      # file is still sitting at a live path.
      warn "cannot move $f to $dest; it is still at the path Vite reads"
    fi
  done
  # Vite requires both to exist (`fs.existsSync(tsCert) && fs.existsSync(tsKey)`,
  # web/vite.config.ts:20), so a pair with either half gone is a pair it no longer
  # selects. Nothing left the cert dir then there is no fallback to restart into.
  if [[ -z "$moved" ]] || { [[ -e "$CRT" ]] && [[ -e "$KEY" ]]; }; then
    return 0
  fi
  warn "quarantined the dead pair so Vite falls back to lloyd.crt:"
  warn "  moved$moved"
  warn "  quarantined leaf notAfter $(date -u -d "@$CURRENT" '+%Y-%m-%dT%H:%M:%SZ')"
  if ! "$SUPERVISORCTL" -c "$SUPERVISORD_CONF" restart "$MC_PROGRAM"; then
    warn "the dead pair is out of $CERT_DIR but the restart failed; Vite keeps serving it until $MC_PROGRAM restarts"
  fi
  return 0
}

# Whether there is a readable leaf on disk at all. `quarantine_dead_leaf` acts on
# a pair this script could not replace, and an absent pair is not one: Vite has
# already fallen back to lloyd.crt for it, and there is nothing to move.
HAVE_LEAF=1
CURRENT="$(not_after_epoch "$CRT")" || CURRENT=""
if [[ -z "$CURRENT" ]]; then
  HAVE_LEAF=0
  if [[ -e "$CRT" ]]; then
    # Clause 3: an unreadable leaf is not a leaf to replace. The pair stays as
    # it is and the unit goes to failed; a human reads the journal.
    warn "cannot read notAfter from $CRT; leaving it untouched"
    exit 1
  fi
  # No leaf at all is the inside-window case: Vite is already serving lloyd.crt
  # and there is nothing here that a mint could damage.
  warn "no tailnet leaf at $CRT; minting one"
  CURRENT=0
fi

NOW="$(date +%s)"
DEADLINE=$((NOW + WINDOW_DAYS * 86400))
if (( CURRENT > DEADLINE )); then
  info "notAfter $(date -u -d "@$CURRENT" '+%Y-%m-%dT%H:%M:%SZ') is more than $WINDOW_DAYS days out; nothing to do"
  exit 0
fi
info "notAfter $(date -u -d "@$CURRENT" '+%Y-%m-%dT%H:%M:%SZ') is inside the $WINDOW_DAYS-day window; renewing"

# Stage BESIDE the cert dir, not inside it and not in /tmp: the publish has to be
# a rename, and a rename across filesystems (/tmp is tmpfs, the repo is btrfs) is
# a copy. The staging tree mirrors the repo layout so the mint command this
# script issues is the documented one — `tailscale cert --cert-file
# agent-services/cert/<host>.crt --key-file agent-services/cert/<host>.key
# <host>` — applied to the staging copy, and the live pair is never an argument
# to anything until it is verified.
STAGE="$(mktemp -d "$CERT_DIR/../.renew-$HOST.XXXXXXXX")"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT
mkdir -p "$STAGE/agent-services/cert"
S_CRT="$STAGE/agent-services/cert/$HOST.crt"
S_KEY="$STAGE/agent-services/cert/$HOST.key"

MINT_ERR="$STAGE/mint.err"
: >"$MINT_ERR"
set +e
TAILSCALE_OUT="$(tailscale cert \
  --cert-file "$S_CRT" \
  --key-file "$S_KEY" \
  "$HOST" 2>"$MINT_ERR")"
MINT_RC=$?
set -e
MINT_MSG="$(tr '\n' ' ' <"$MINT_ERR") ${TAILSCALE_OUT:-}"

if (( MINT_RC != 0 )); then
  # tailscale refuses the mint for one of several reasons; only one of them has
  # a fix the operator can apply in one command, and saying which keeps the run
  # record actionable. Exactly one line names the grant, and this script never
  # runs sudo itself.
  if grep -qi 'operator' <<<"$MINT_MSG$TAILSCALE_OUT"; then
    warn "mint refused: $WHOAMI is not the tailscale operator — fix with: sudo tailscale set --operator=$WHOAMI"
  else
    warn "tailscale cert failed (exit $MINT_RC): ${MINT_MSG:-no output}"
  fi
  quarantine_dead_leaf
  exit 1
fi

[[ -s "$S_CRT" && -s "$S_KEY" ]] || { warn "minted material is empty; leaving the current pair in place"; quarantine_dead_leaf; exit 1; }
NEW="$(not_after_epoch "$S_CRT")" || { warn "minted leaf does not parse; leaving the current pair in place"; quarantine_dead_leaf; exit 1; }
if (( NEW <= CURRENT )); then
  warn "minted notAfter $(date -u -d "@$NEW" '+%Y-%m-%dT%H:%M:%SZ') is not later than the current one; leaving it in place"
  quarantine_dead_leaf
  exit 1
fi

# Verified, so publish. The key keeps tailscale's owner-only mode; chmod before
# the rename so there is no window where a group-readable key sits at the live
# path.
chmod 600 "$S_KEY"
chmod 644 "$S_CRT"
mv -f "$S_CRT" "$CRT"
mv -f "$S_KEY" "$KEY"
trap - EXIT
cleanup
info "leaf renewed to $(date -u -d "@$NEW" '+%Y-%m-%dT%H:%M:%SZ'); restarting $MC_PROGRAM"

if ! "$SUPERVISORCTL" -c "$SUPERVISORD_CONF" restart "$MC_PROGRAM"; then
  # The cert dir is already good here, so this is not an inert path: say plainly
  # that the new leaf is on disk and Vite is still serving the old one until
  # something restarts it, rather than pretending nothing happened.
  warn "minted a leaf valid to $(date -u -d "@$NEW" '+%Y-%m-%dT%H:%M:%SZ') but the restart failed; Vite is still serving the previous cert until $MC_PROGRAM restarts"
  exit 1
fi
info "done"
