#!/usr/bin/env bash
# Re-link Obsidian Sync to a NEW remote vault after an incident — safely.
#
#   agent-services/bin/obsidian-sync-relink.sh <new-remote-name>
#
# Interactive: `ob` prompts for the end-to-end encryption password twice (once
# to create the remote, once to set up this device). Run it from a terminal.
#
# Why a new remote and not the old one: every `ob` mode downloads remote
# changes, and a remote that received an incident's deletions will delete the
# restored local vault the moment sync starts. That is why 566dda12… was retired
# on 2026-09-14 (it held the 09-12 wipe's 5,363 deletions), as c73df6a0… was on
# 2026-09-10. An empty remote has nothing to download, so the first pass can
# only upload. See architecture/vault-protection.md.
#
# Order, and each step refuses on anything unexpected:
#   1. the vault tripwire is clear and the sync gate passes;
#   2. agent-obsidian-sync is stopped;
#   3. a fresh out-of-vault snapshot is taken;
#   4. create the remote, set up this device, apply the settings used before;
#   5. ONE non-continuous pass, whose output must contain no deletion and no
#      download — otherwise stop before continuous sync ever starts;
#   6. start agent-obsidian-sync under supervisord.
set -euo pipefail

NAME="${1:-}"
if [[ -z "$NAME" ]]; then
  echo "usage: $0 <new-remote-name>" >&2
  exit 2
fi

VAULT=/home/alansrobotlab/obsidian
LLOYD=/home/alansrobotlab/lloyd
SUPERVISORCTL=(/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl
               -c "$LLOYD/agent-services/supervisor/supervisord.conf")
VAULTWATCH="$HOME/.local/state/lloyd-guardian/bin/vaultwatch.py"
[[ -f "$VAULTWATCH" ]] || VAULTWATCH="$LLOYD/agent-services/guardian/vaultwatch.py"
OB="$(command -v ob)"

step() { echo; echo "== $*"; }

step "1. vault gate"
/usr/bin/python3 "$VAULTWATCH" sync-gate

step "2. stop agent-obsidian-sync"
"${SUPERVISORCTL[@]}" stop agent-obsidian-sync || true
if pgrep -f "sync --path $VAULT" >/dev/null; then
  echo "an 'ob sync' for $VAULT is still running — stop it first" >&2
  exit 1
fi

step "3. snapshot"
"$LLOYD/scripts/backup/backup-vault.sh"

step "4. remote + device"
if "$OB" sync-list-remote 2>/dev/null | grep -q "\"$NAME\""; then
  echo "a remote vault named \"$NAME\" already exists — pick a new name; this script only links EMPTY remotes" >&2
  exit 1
fi
"$OB" sync-create-remote --name "$NAME" --encryption e2ee
"$OB" sync-setup --vault "$NAME" --path "$VAULT" --device-name "goliath-headless"
"$OB" sync-config --path "$VAULT" --conflict-strategy merge \
  --file-types image,audio,pdf,video --configs "" --excluded-folders .git

step "5. one pass (upload only expected)"
before=$(find "$VAULT" -name .git -prune -o -type f -print | wc -l)
log="$(mktemp)"
"$OB" sync --path "$VAULT" 2>&1 | tee "$log"
after=$(find "$VAULT" -name .git -prune -o -type f -print | wc -l)
if grep -qiE '^(Deleting|Downloading|New remote file)' "$log"; then
  echo "REFUSING to start continuous sync: the first pass deleted or downloaded something (see $log)" >&2
  exit 1
fi
if (( after < before )); then
  echo "REFUSING: local file count fell during the pass ($before → $after)" >&2
  exit 1
fi
echo "first pass clean: $(grep -c '^Upload complete' "$log" || true) uploads, local files $before → $after"

step "6. continuous sync"
"${SUPERVISORCTL[@]}" start agent-obsidian-sync
sleep 5
"${SUPERVISORCTL[@]}" status agent-obsidian-sync
