#!/usr/bin/env bash
# Restore a vault snapshot into a SIDE directory. Never touches ~/obsidian.
#
#   scripts/backup/restore-vault.sh                 # list recent snapshots
#   scripts/backup/restore-vault.sh <rev> [dest]    # extract one
#
# <rev> is a commit from the list (or HEAD, HEAD~3, a date via
# `@{2026-09-12 11:00}`). The snapshot is checked out into
# dest (default ~/vault-restore-<stamp>/obsidian) through a throwaway index, so
# the backup repository's own state is untouched.
#
# Swapping it in stays a human step, and the order matters because sync is
# bidirectional (see agent-services/guardian/vaultwatch.py):
#   1. keep agent-obsidian-sync stopped (the tripwire already stopped it);
#   2. mv ~/obsidian ~/obsidian.broken-<stamp>; mv <dest> ~/obsidian
#   3. /usr/bin/python3 ~/.local/state/lloyd-guardian/bin/vaultwatch.py clear
#   4. decide what the cloud copy should be before starting sync again — it
#      holds whatever the incident pushed.
set -euo pipefail

REPO="${LLOYD_VAULT_BACKUP_REPO:-$HOME/.local/state/lloyd-vault-backup/vault.git}"
if [[ ! -d "$REPO" ]]; then
  echo "restore-vault: no backup repository at $REPO" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  git --git-dir="$REPO" log --format='%h  %ad  %s' --date=iso -n 30
  exit 0
fi

REV="$1"
DEST="${2:-$HOME/vault-restore-$(date +%Y%m%d-%H%M%S)/obsidian}"
case "$(realpath -m "$DEST")" in
  "$(realpath -m "$HOME/obsidian")"|"$(realpath -m "$HOME/obsidian")"/*)
    echo "restore-vault: refusing to restore into the live vault; pick a side directory" >&2
    exit 1 ;;
esac
if [[ -e "$DEST" ]] && [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
  echo "restore-vault: $DEST exists and is not empty" >&2
  exit 1
fi

COMMIT=$(git --git-dir="$REPO" rev-parse --verify "$REV^{commit}")
mkdir -p "$DEST"
INDEX=$(mktemp)
trap 'rm -f "$INDEX"' EXIT
rm -f "$INDEX"
GIT_INDEX_FILE="$INDEX" git --git-dir="$REPO" --work-tree="$DEST" read-tree "$COMMIT"
GIT_INDEX_FILE="$INDEX" git --git-dir="$REPO" --work-tree="$DEST" checkout-index -a -f
echo "restore-vault: $(git --git-dir="$REPO" log -1 --format='%h %ad %s' --date=iso "$COMMIT")"
echo "restore-vault: $(find "$DEST" -type f | wc -l) files in $DEST"
echo "restore-vault: nothing was swapped in — see the header of this script for the steps"
