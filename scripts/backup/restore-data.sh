#!/usr/bin/env bash
# Restore a data-root snapshot into a SIDE directory. Never touches ~/lloyd-data.
#
#   scripts/backup/restore-data.sh                  # list snapshots
#   scripts/backup/restore-data.sh <stamp> [dest]   # copy one out
#
# The copy is `cp -a --reflink=always`: instant and space-free on btrfs, and
# writable, unlike the snapshot. Every SQLite file in it is integrity-checked.
#
# Swapping it in stays a human step, with the stack stopped:
#   1. systemctl --user stop lloyd-guardian agent-supervisord
#   2. mv ~/lloyd-data ~/lloyd-data.broken-<stamp>; mv <dest> ~/lloyd-data
#      (the restored copy is a plain directory; to make it a subvolume again,
#      `btrfs subvolume create ~/lloyd-data` and cp --reflink the contents in)
#   3. /usr/bin/python3 ~/.local/state/lloyd-guardian/bin/datawatch.py clear
#   4. systemctl --user start agent-supervisord lloyd-guardian
set -euo pipefail

SNAPS="${LLOYD_DATA_SNAPSHOTS:-$HOME/.lloyd-data-snapshots}"
LIVE="${LLOYD_DATA:-$HOME/lloyd-data}"
if [[ $# -eq 0 ]]; then
  ls -1 "$SNAPS" 2>/dev/null | tail -n 60
  exit 0
fi
SRC="$SNAPS/$1"
DEST="${2:-$HOME/lloyd-data-restore-$(date +%Y%m%d-%H%M%S)}"
[[ -d "$SRC" ]] || { echo "restore-data: no snapshot $SRC" >&2; exit 1; }
case "$(realpath -m "$DEST")" in
  "$(realpath -m "$LIVE")"|"$(realpath -m "$LIVE")"/*)
    echo "restore-data: refusing to restore into the live data root; pick a side directory" >&2
    exit 1 ;;
esac
if [[ -e "$DEST" ]]; then
  echo "restore-data: $DEST already exists" >&2
  exit 1
fi
cp -a --reflink=always "$SRC" "$DEST"
bad=0
while IFS= read -r -d '' db; do
  r=$(sqlite3 "file:$db?mode=ro" 'PRAGMA integrity_check;' 2>&1 | head -1)
  [[ "$r" == ok ]] && echo "ok       $db" || { echo "DAMAGED  $db: $r"; bad=1; }
done < <(find "$DEST" \( -name '*.db' -o -name '*.sqlite' \) -type f -print0)
echo "restored $SRC -> $DEST"
exit $bad
