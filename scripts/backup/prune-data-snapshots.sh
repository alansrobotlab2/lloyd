#!/usr/bin/env bash
# Prune ~alansrobotlab/.lloyd-data-snapshots to 48 hourly + 14 daily. ROOT ONLY.
#
# The snapshots are read-only subvolumes that nothing running as the user can
# delete (that is the point: see snapshot-data.sh). So pruning runs as root,
# from the system timer lloyd-data-snapshot-prune.timer, installed once with:
#   sudo install -m 0755 -o root -g root scripts/backup/prune-data-snapshots.sh /usr/local/sbin/
#   sudo install -m 0644 agent-services/systemd/system/lloyd-data-snapshot-prune.* /etc/systemd/system/
#   sudo systemctl daemon-reload && sudo systemctl enable --now lloyd-data-snapshot-prune.timer
# The installed copy is root-owned, so an edit to this file in the repo changes
# nothing until a person re-installs it.
#
# Keeps: every snapshot from the last 48 hours, plus the newest snapshot of each
# of the last 14 days. Never deletes the newest snapshot, whatever its age.
set -euo pipefail

DEST="${1:-/home/alansrobotlab/.lloyd-data-snapshots}"
HOURLY_HOURS=48
DAILY_DAYS=14
DRY="${DRY_RUN:-0}"

[[ -d "$DEST" ]] || { echo "prune: $DEST does not exist"; exit 0; }
now=$(date -u +%s)
mapfile -t snaps < <(find "$DEST" -mindepth 1 -maxdepth 1 -type d -name '20*T*Z' -printf '%f\n' | sort)
(( ${#snaps[@]} )) || exit 0
newest="${snaps[-1]}"
declare -A keep_day=()
for (( i=${#snaps[@]}-1; i>=0; i-- )); do
  s="${snaps[$i]}"
  ts=$(date -u -d "${s:0:4}-${s:4:2}-${s:6:2} ${s:9:2}:${s:11:2}:${s:13:2}" +%s) || continue
  age=$(( now - ts ))
  day="${s:0:8}"
  if [[ "$s" == "$newest" ]] || (( age <= HOURLY_HOURS * 3600 )); then
    keep_day[$day]=1; continue
  fi
  if (( age <= DAILY_DAYS * 86400 )) && [[ -z "${keep_day[$day]:-}" ]]; then
    keep_day[$day]=1; continue
  fi
  if [[ "$DRY" == 1 ]]; then echo "would delete $DEST/$s"
  else btrfs subvolume delete "$DEST/$s" >/dev/null && echo "deleted $DEST/$s"; fi
done
