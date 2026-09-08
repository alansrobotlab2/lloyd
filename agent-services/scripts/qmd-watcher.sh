#!/usr/bin/env bash
# qmd-watcher.sh — Watch obsidian vault for .md changes, trigger qmd update + embed
# Uses inotifywait with a 2-second debounce to batch rapid edits
set -euo pipefail

QMD="$HOME/.bun/bin/qmd"
VAULT="$HOME/obsidian"
# Session transcripts are exported outside the vault, to the derived tree
# (app/post_capture.py::_export_session_markdown). The qmd `sessions`
# collection points there, so it has to be watched too. Before 2026-09-08 it
# was not, and neither was the collection: it pointed at ~/obsidian/sessions,
# which the 2026-05-08 vault reorg emptied. Watching only the vault would
# leave indexing to whatever unrelated vault write happened next -- usually
# the daily-note append from the same capture, except when that capture is
# skipped as TRIVIAL and no vault write follows the export at all.
SESSIONS="$HOME/lloyd/_pipeline/vault-derived/sessions"
DEBOUNCE_SEC=2

log() { echo "$(date '+%H:%M:%S') $*"; }

log "Watching $VAULT and $SESSIONS for .md changes (debounce: ${DEBOUNCE_SEC}s)"

mkdir -p "$SESSIONS"

inotifywait -m -r -e close_write,create,delete,moved_to,moved_from \
  --include '\.md$' "$VAULT" "$SESSIONS" |
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
