#!/bin/bash
# Obsidian Headless continuous sync for the lloyd vault (~/obsidian).
#
# This is the SINGLE Obsidian Sync client for this device. The desktop app's
# Sync core plugin MUST be disabled — running both on the same vault causes
# data conflicts (per Obsidian docs: "Do not use both the desktop app Sync and
# Headless Sync on the same device").
#
# Prereqs (interactive, run once by the user):
#   ob login            # Obsidian account email + password + MFA
#   ob sync-setup --vault "<name>" --path ~/obsidian   # + E2E encryption password
#
# obsidian-headless installed via: npm install -g obsidian-headless
#
# Install location varies: a user-prefix install lands in ~/.npm-global/bin/ob,
# a root/system install in /usr/bin/ob. Resolve rather than hardcode — pinning
# one path is what left this service FATAL after the 2026-08-22 rebuild.
set -euo pipefail

VAULT_PATH=/home/alansrobotlab/obsidian

OB="${OB:-}"
if [[ -z "$OB" ]]; then
  for cand in /home/alansrobotlab/.npm-global/bin/ob /usr/bin/ob "$(command -v ob 2>/dev/null || true)"; do
    if [[ -n "$cand" && -x "$cand" ]]; then OB="$cand"; break; fi
  done
fi

if [[ -z "$OB" || ! -x "$OB" ]]; then
  echo "[start-obsidian-sync] ERROR: 'ob' not found." >&2
  echo "  install:  npm install -g obsidian-headless" >&2
  echo "  override: OB=/path/to/ob in the supervisor conf" >&2
  exit 1
fi

if ! "$OB" sync-list-local </dev/null 2>/dev/null | grep -q "$VAULT_PATH"; then
  echo "[start-obsidian-sync] ERROR: no vault configured for ${VAULT_PATH}." >&2
  echo "  These are interactive and must be run once by hand:" >&2
  echo "    ob login" >&2
  echo "    ob sync-setup --vault \"<name>\" --path ${VAULT_PATH}" >&2
  exit 1
fi

echo "[start-obsidian-sync] $(date -Is) launching continuous sync for ${VAULT_PATH}"
exec "${OB}" sync --path "${VAULT_PATH}" --continuous
