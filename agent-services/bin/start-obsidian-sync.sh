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
set -euo pipefail

OB=/home/alansrobotlab/.npm-global/bin/ob
VAULT_PATH=/home/alansrobotlab/obsidian

echo "[start-obsidian-sync] $(date -Is) launching continuous sync for ${VAULT_PATH}"
exec "${OB}" sync --path "${VAULT_PATH}" --continuous
