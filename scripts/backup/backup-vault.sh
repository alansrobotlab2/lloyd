#!/usr/bin/env bash
# Snapshot the vault into a git repository that lives OUTSIDE the vault.
#
# Why this exists: the vault was deleted on 2026-09-10 and again on
# 2026-09-12, both times together with its own `.git`, and both times Obsidian
# Sync pushed the deletions to the cloud copy. What saved it was a tarball the
# deleting process happened to write first. Recovery should not depend on the
# culprit's manners.
#
# Every 15 minutes (lloyd-vault-backup.timer) the whole vault — notes, the
# vault's untracked files, `.obsidian`; never the vault's own `.git` — is
# committed to ~/.local/state/lloyd-vault-backup/vault.git. Git deduplicates,
# so a snapshot costs roughly what changed since the last one.
#
# Two refusals, both so a wipe is never recorded as the new normal:
#   * the guardian's vault tripwire is set (vault-tripped.json) — an incident
#     is open, and the last snapshot is the one a restore wants;
#   * the vault holds under 90% of the files the last snapshot recorded — the
#     tripwire should have fired; this is the second net.
# A refusal exits 0 after saying why, so the timer does not flap to failed
# and hide the next real error; `journalctl --user -u lloyd-vault-backup`
# shows it.
#
# Restore with scripts/backup/restore-vault.sh (into a side directory, never
# in place).
set -euo pipefail

VAULT="${LLOYD_VAULT_ROOT:-$HOME/obsidian}"
REPO="${LLOYD_VAULT_BACKUP_REPO:-$HOME/.local/state/lloyd-vault-backup/vault.git}"
MARKER="${LLOYD_VAULT_TRIP_MARKER:-$HOME/.local/state/lloyd-guardian/vault-tripped.json}"
MIN_FRACTION_PCT=90

log() { echo "backup-vault: $*"; }

if [[ -f "$MARKER" ]]; then
  log "vault tripwire is set ($MARKER) — not snapshotting a vault under incident"
  exit 0
fi
if [[ ! -d "$VAULT" ]]; then
  log "ERROR: $VAULT does not exist"
  exit 1
fi

g() { git --git-dir="$REPO" --work-tree="$VAULT" "$@"; }

if [[ ! -d "$REPO" ]]; then
  mkdir -p "$(dirname "$REPO")"
  git init --quiet --bare "$REPO"
  git --git-dir="$REPO" config core.bare false
  git --git-dir="$REPO" config user.name "lloyd-vault-backup"
  git --git-dir="$REPO" config user.email "vault-backup@localhost"
  # The vault's own ignore rules are for its own repo; a backup keeps it all.
  git --git-dir="$REPO" config core.excludesFile /dev/null
  git --git-dir="$REPO" config gc.auto 256
  log "initialised $REPO"
fi

count=$(find "$VAULT" -name .git -prune -o -type f -print | wc -l)
last_file="$REPO/lloyd-last-count"
if [[ -f "$last_file" ]]; then
  last=$(cat "$last_file")
  if [[ "$last" =~ ^[0-9]+$ ]] && (( last > 0 )) && (( count * 100 < last * MIN_FRACTION_PCT )); then
    log "REFUSING: vault holds $count files, the last snapshot held $last — not recording a shrunken vault"
    exit 0
  fi
fi

# --force: include files the vault's own .gitignore excludes. Git never adds a
# directory named .git, so the vault's repository is not copied into this one.
g add --all --force .
if g diff --cached --quiet 2>/dev/null && g rev-parse --verify -q HEAD >/dev/null; then
  log "no change ($count files)"
else
  g commit --quiet --no-verify -m "snapshot $(date -Is) files=$count"
  log "committed $(g rev-parse --short HEAD) ($count files)"
fi
echo "$count" > "$last_file"
g gc --auto --quiet || true
