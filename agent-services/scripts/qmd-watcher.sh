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
SESSIONS="${LLOYD_DATA:-$HOME/lloyd-data}/_pipeline/vault-derived/sessions"
# The checkout's architecture docs, qmd's `lloyd-architecture` collection
# (2026-11-10). A landing rewrites them in ~/lloyd and nothing else in the watch
# moves with it, so without this the collection would serve the docs as of the
# last unrelated vault write.
ARCH_DOCS="$HOME/lloyd/architecture"
DEBOUNCE_SEC=2
# A no-op `qmd update` is not cheap: it re-hashes every file of every collection
# `index.yml` declares before it discovers that one of them changed. No count is
# stated here, because one goes stale -- the size this block used to quote was
# unreproducible within three weeks of being stamped (#3161). Two commands
# measure what the walk covers instead, and they are the only things that can:
#   sqlite3 ~/.cache/qmd/index.sqlite "select count(*) from documents where active=1"
# for the indexed documents, and each collection's own `pattern`/`ignore` in
# `index.yml` for the files it re-hashes. The figure this file used to quote was
# 7.5 s for such a walk, measured once on 2026-09-19 against the live index; read
# that as the day's observation it was, not as today's cost. The watcher's own
# supervisor log, measured on 2026-10-22: median 2 s, mean 1.8 s, max 39 s.
# That log is the derivation, and it is re-runnable: `stdout_logfile` in
# agent-services/supervisor/conf.d/agent-qmd-watcher.conf names the file, one
# walk is the gap from an `HH:MM:SS Change detected` stamp to the next
# `HH:MM:SS Index updated`, and the window is every such pair since that file's
# last 10 MB rotation. The stamps carry no date, and that costs the pairing twice,
# so the derivation drops two kinds of pair first: a negative gap is a walk that
# crossed midnight, and a pair straddling a `Watching` restart banner is a gap
# between two processes rather than one walk. The missing date also means
# 2026-10-22 is the day these were computed, not the span they cover, and the next
# rotation discards this window -- the derivation is the part meant to outlive it,
# and running it on whatever the log holds then is how the next reader gets
# today's figure. What the floor below insures against is that 39 s tail, not the
# 2 s median: at the median a minute is a 30x margin, and the window held exactly
# one cycle of 39 s.
# Retained derivation, committed: backlog/data/2026-10-22.3165-services-agent-qmd-watcher.log
#
# The debounce alone does not bound how often that runs -- it
# only waits for a quiet gap, and with the automod loop at depth 4 writing
# backlog activity logs continuously there is a fresh gap every few seconds.
# 213 cycles were logged in one evening.
#
# COOLDOWN_SEC is the floor between two walks: after a cycle, events are
# consumed without triggering work until it expires. An event consumed there —
# or one that arrived while the walk itself ran, which waits in the pipe and is
# read by the same drain — is REMEMBERED, and one more cycle runs as soon as
# the floor has passed. Until 2026-11-10 those events were read and dropped,
# and nothing re-triggered: an edit landing during a walk stayed unindexed until
# some unrelated write happened, which on a quiet vault could be hours, and the
# old "worst case = cooldown + cycle" here was not true. Worst-case staleness
# is now one cycle + COOLDOWN_SEC + MAX_DEBOUNCE_SEC + one cycle.
# `backlog_similar` already assumes qmd may not have seen an item created in the
# last ten minutes (its rule B covers exactly this window with a lexical leg), so
# a minute of lag costs nothing that is not already designed for.
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

# Consume events for $1 seconds without acting on them, and set DRAINED=1 if
# any arrived (including those buffered while the walk ran). Returns early only
# if the whole remaining window was quiet -- either way no walk starts before
# the floor has passed.
DRAINED=0
drain_for() {
  local end=$(( SECONDS + $1 )) remaining
  DRAINED=0
  while (( (remaining = end - SECONDS) > 0 )); do
    read -r -t "$remaining" || return 0
    DRAINED=1
  done
}

# One walk: update, then embed what the update found.
run_cycle() {
  log "Change detected, updating index..."
  if "${QMD_CLI[@]}" update 2>&1; then
    log "Index updated, embedding new documents..."
    "${QMD_CLI[@]}" embed 2>&1 || log "WARNING: embed failed"
  else
    log "WARNING: update failed"
  fi
}

# The loop, reading one event per line on stdin until EOF. A change seen during
# a walk or its cooldown is owed one more walk, which runs without waiting for
# a fresh event.
watch_loop() {
  local pending=0
  while :; do
    if [[ $pending == 0 ]]; then
      read -r || return 0
    fi
    debounce
    run_cycle
    log "Ready (cooldown ${COOLDOWN_SEC}s)"
    drain_for "$COOLDOWN_SEC"
    pending=$DRAINED
    if [[ $pending == 1 ]]; then
      log "Changes arrived during the last walk; re-running"
    fi
  done
}

# Sourced by tests/test_qmd_watcher_rerun.py to drive watch_loop with a stub CLI.
if [[ ${QMD_WATCHER_LIB:-0} == 1 ]]; then
  return 0 2>/dev/null || exit 0
fi

log "Watching $VAULT, $SESSIONS and $ARCH_DOCS for .md changes (debounce: ${DEBOUNCE_SEC}s, max ${MAX_DEBOUNCE_SEC}s; cooldown: ${COOLDOWN_SEC}s)"

mkdir -p "$SESSIONS"

inotifywait -m -r -e close_write,create,delete,moved_to,moved_from \
  --include '\.md$' "$VAULT" "$SESSIONS" "$ARCH_DOCS" |
watch_loop
