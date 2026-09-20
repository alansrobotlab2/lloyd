#!/usr/bin/env bash
# qmd-watcher.sh — Watch obsidian vault for .md changes, trigger qmd update + embed
# Uses inotifywait with a 2-second debounce to batch rapid edits
set -euo pipefail

# The DAEMON serves this index from the fork in ~/lloyd/qmd (see
# agent-services/supervisor/conf.d/agent-qmd-daemon.conf), so the writer must be
# the same build. Until 2026-09-19 this was `$HOME/.bun/bin/qmd`, the published
# @tobilu/qmd 2.8.3 (facd35e) installed by setup-all.sh, while the daemon ran the
# fork (a7b5425) -- same version string, different commit. The fork's changes are
# serve-side (in-memory vector index, rerank window, daemon knobs, skipRerank on
# REST), so the published build's `update`/`embed` did write a compatible index;
# what it cost was a second definition of how this index gets built, which would
# diverge silently the first time the fork touches chunking or embedding.
QMD_CLI=("/usr/bin/node" "$HOME/lloyd/qmd/dist/cli/qmd.js")
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
# A no-op `qmd update` is not cheap: it re-hashes all ~15,800 files across all
# 15 collections to discover that one of them changed, and that costs 7.5s
# (measured 2026-09-19, live index). The debounce alone does not bound how
# often that runs -- it only waits for a quiet gap, and with the automod loop
# at depth 4 writing backlog activity logs continuously there is a fresh gap
# every few seconds. 213 cycles were logged in one evening.
#
# COOLDOWN_SEC is the floor between two walks: after a cycle, events are
# consumed without triggering work until it expires. Worst-case staleness is
# COOLDOWN_SEC + the cycle itself; `backlog_similar` already assumes qmd may
# not have seen an item created in the last ten minutes (its rule B covers
# exactly this window with a lexical leg), so a minute of lag costs nothing
# that is not already designed for.
COOLDOWN_SEC=60
# And the debounce needs a ceiling of its own, or it starves in the other
# direction: `while read -t 2` only exits on a 2s quiet gap, so a vault under
# sustained sub-2s writes would never reach the update at all.
MAX_DEBOUNCE_SEC=10

log() { echo "$(date '+%H:%M:%S') $*"; }

# Consume events until a $DEBOUNCE_SEC quiet gap, or $MAX_DEBOUNCE_SEC total.
debounce() {
  local end=$(( SECONDS + MAX_DEBOUNCE_SEC ))
  while (( SECONDS < end )); do
    read -r -t "$DEBOUNCE_SEC" || return 0
  done
}

# Consume events for $1 seconds without acting on them. Returns early only if
# the whole remaining window was quiet, in which case the outer blocking read
# is equivalent -- either way no walk starts before the floor has passed.
drain_for() {
  local end=$(( SECONDS + $1 )) remaining
  while (( (remaining = end - SECONDS) > 0 )); do
    read -r -t "$remaining" || return 0
  done
}

log "Watching $VAULT and $SESSIONS for .md changes (debounce: ${DEBOUNCE_SEC}s, max ${MAX_DEBOUNCE_SEC}s; cooldown: ${COOLDOWN_SEC}s)"

mkdir -p "$SESSIONS"

inotifywait -m -r -e close_write,create,delete,moved_to,moved_from \
  --include '\.md$' "$VAULT" "$SESSIONS" |
while read -r; do
  debounce

  log "Change detected, updating index..."
  if "${QMD_CLI[@]}" update 2>&1; then
    log "Index updated, embedding new documents..."
    "${QMD_CLI[@]}" embed 2>&1 || log "WARNING: embed failed"
  else
    log "WARNING: update failed"
  fi
  log "Ready (cooldown ${COOLDOWN_SEC}s)"
  drain_for "$COOLDOWN_SEC"
done
