#!/usr/bin/env bash
# qmd-watcher.sh — Watch obsidian vault for .md changes, trigger qmd update + embed
# Uses inotifywait with a 2-second debounce to batch rapid edits
set -euo pipefail

QMD="$HOME/.bun/bin/qmd"
VAULT="$HOME/obsidian"
DEBOUNCE_SEC=2

log() { echo "$(date '+%H:%M:%S') $*"; }

log "Watching $VAULT for .md changes (debounce: ${DEBOUNCE_SEC}s)"

inotifywait -m -r -e close_write,create,delete,moved_to,moved_from \
  --include '\.md$' "$VAULT" |
while read -r; do
  # Drain additional events within the debounce window
  while read -r -t "$DEBOUNCE_SEC"; do :; done

  log "Change detected, updating index..."
  if "$QMD" update 2>&1; then
    log "Index updated, embedding new documents..."
    "$QMD" embed 2>&1 || log "WARNING: embed failed"
  else
    log "WARNING: update failed"
  fi
  log "Ready"
done
