#!/usr/bin/env bash
# Take a read-only btrfs snapshot of Lloyd's data root (~/lloyd-data).
#
# Why this exists: on 2026-09-22 a pytest fixture teardown deleted ~/lloyd,
# and every session, database and log inside it went too. The data moved out of
# the tree to its own btrfs subvolume (architecture/data-home.md); this is the
# undo for the next thing that reaches it.
#
# Every hour (lloyd-data-snapshot.timer) the subvolume is snapshotted READ-ONLY
# into ~/.lloyd-data-snapshots/<UTC stamp>. The snapshot is atomic, so SQLite in
# WAL mode opens it the way it opens after a power cut; nothing needs .backup.
# And it is out of reach of everything running as this user: @home is not
# mounted user_subvol_rm_allowed, so `btrfs subvolume delete` is EPERM and
# `rm -rf` is EROFS. Pruning therefore needs root —
# scripts/backup/prune-data-snapshots.sh, run by a system timer.
#
# Two refusals, both so a wipe is never recorded as the newest snapshot:
#   * the guardian's data tripwire is set (data-tripped.json);
#   * datawatch's snapshot gate says the root shrank below its last healthy
#     measurement, or lost its .lloyd-data-root marker.
# A refusal exits 0 after saying why, so the timer does not flap to failed.
#
# Restore with scripts/backup/restore-data.sh (into a side directory).
set -euo pipefail

DATA="${LLOYD_DATA:-$HOME/lloyd-data}"
DEST="${LLOYD_DATA_SNAPSHOTS:-$HOME/.lloyd-data-snapshots}"
GSTATE="${LLOYD_GUARDIAN_STATE:-$HOME/.local/state/lloyd-guardian}"
# The pinned guardian copy first: it is what the running watchdog uses.
DATAWATCH="$GSTATE/bin/datawatch.py"
[[ -f "$DATAWATCH" ]] || DATAWATCH="$(dirname "$0")/../../agent-services/guardian/datawatch.py"

log() { echo "snapshot-data: $*"; }

if [[ -f "$GSTATE/data-tripped.json" ]]; then
  log "data tripwire is set ($GSTATE/data-tripped.json) — not snapshotting a root under incident"
  exit 0
fi
if ! gate=$(LLOYD_DATA="$DATA" /usr/bin/python3 "$DATAWATCH" snapshot-gate 2>&1); then
  log "REFUSING: $gate"
  exit 0
fi
mkdir -p "$DEST"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
btrfs subvolume snapshot -r "$DATA" "$DEST/$stamp" >/dev/null
log "snapshot $DEST/$stamp (${gate#ok: })"
