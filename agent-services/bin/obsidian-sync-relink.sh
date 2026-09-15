#!/usr/bin/env bash
# Re-link Obsidian Sync after an incident — safely.
#
#   agent-services/bin/obsidian-sync-relink.sh <new-remote-name>
#   agent-services/bin/obsidian-sync-relink.sh --existing <remote-vault-id>
#
# Interactive: `ob` prompts for the end-to-end encryption password (twice for a
# new remote: once to create it, once to set up this device). Run it from a
# terminal.
#
# Two situations, two modes:
#
# * **A new remote** (`<new-remote-name>`) when the remote received an
#   incident's deletions. Every `ob` mode downloads remote changes, and such a
#   remote will delete the restored local vault the moment sync starts. That is
#   why 566dda12… was retired on 2026-09-14 (it held the 09-12 wipe's 5,363
#   deletions), as c73df6a0… was on 2026-09-10. An empty remote has nothing to
#   download, so the first pass can only upload.
# * **The existing remote** (`--existing <id>`) when only this device's
#   registration was lost and the remote is sound — 2026-09-14 14:15, when the
#   health check's round-trip probe `sync-unlink`ed the live registration by
#   vault id while the client, still running from memory, reported `Fully
#   synced`. Re-uploading the whole vault to a fresh remote there costs quota
#   for nothing. A two-way first pass could push a deletion before any check
#   saw it, so this mode's first pass is **pull-only** — it cannot change the
#   remote — and it is judged by a content manifest of the local vault, not by
#   log lines: identical before and after, or continuous sync never starts.
#
# Order, and each step refuses on anything unexpected:
#   1. the vault tripwire is clear and the sync gate passes;
#   2. agent-obsidian-sync is stopped;
#   3. a fresh out-of-vault snapshot is taken;
#   4. new: create the remote; existing: confirm the id is listed and not
#      retired. Then set up this device and apply the settings used before;
#   5. ONE non-continuous pass — new: output must contain no deletion and no
#      download; existing: pull-only, and the local content must not change;
#   6. start agent-obsidian-sync under supervisord.
# See architecture/vault-protection.md.
set -euo pipefail

RETIRED_IDS=(566dda1217a7a8dfddbac33d4f280b3a c73df6a055405245f95f936685bb3683)

MODE=new
TARGET="${1:-}"
if [[ "$TARGET" == "--existing" ]]; then
  MODE=existing
  TARGET="${2:-}"
fi
if [[ -z "$TARGET" ]]; then
  echo "usage: $0 <new-remote-name>" >&2
  echo "       $0 --existing <remote-vault-id>" >&2
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

# Content of every file ob syncs, by path. `.git` is excluded from sync, and
# `.obsidian/.sync.lock` is the client's own lock file.
manifest() {
  (cd "$VAULT" && find . -path ./.git -prune -o -path ./.obsidian/.sync.lock -prune \
     -o -type f -print0 | sort -z | xargs -0 sha1sum)
}

if [[ "$MODE" == existing ]]; then
  for retired in "${RETIRED_IDS[@]}"; do
    if [[ "$TARGET" == "$retired" ]]; then
      echo "$TARGET is a retired remote that holds an incident's deletions — never re-link it" >&2
      exit 1
    fi
  done
fi

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
if [[ "$MODE" == new ]]; then
  if "$OB" sync-list-remote 2>/dev/null | grep -q "\"$TARGET\""; then
    echo "a remote vault named \"$TARGET\" already exists — pick a new name, or use --existing <id>" >&2
    exit 1
  fi
  "$OB" sync-create-remote --name "$TARGET" --encryption e2ee
else
  if ! "$OB" sync-list-remote 2>/dev/null | grep -qE "^[[:space:]]*$TARGET[[:space:]]"; then
    echo "no remote vault with id $TARGET on this account (ob sync-list-remote)" >&2
    exit 1
  fi
fi
"$OB" sync-setup --vault "$TARGET" --path "$VAULT" --device-name "goliath-headless"
"$OB" sync-config --path "$VAULT" --conflict-strategy merge \
  --file-types image,audio,pdf,video --configs "" --excluded-folders .git

log="$(mktemp)"
if [[ "$MODE" == new ]]; then
  step "5. one pass (upload only expected)"
  before=$(find "$VAULT" -name .git -prune -o -type f -print | wc -l)
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
else
  step "5. one pull-only pass (the local vault must come out identical)"
  # Lloyd's workers write into the vault; a write landing during the pass would
  # read as the pass changing a file. Pause new jobs for the pass (a pause
  # somebody else set is left alone), and name what changed if anything did.
  POOL=http://127.0.0.1:8080/api/workers
  if curl -s -m 5 "$POOL/status" | grep -q '"paused": *false'; then
    curl -s -m 5 -X POST -H 'Content-Type: application/json' -d '{"paused": true}' \
      "$POOL/pause" >/dev/null && trap \
      "curl -s -m 5 -X POST -H 'Content-Type: application/json' -d '{\"paused\": false}' $POOL/pause >/dev/null" EXIT
  fi
  before="$(mktemp)"; after="$(mktemp)"
  manifest > "$before"
  "$OB" sync-config --path "$VAULT" --mode pull-only
  "$OB" sync --path "$VAULT" 2>&1 | tee "$log"
  manifest > "$after"
  if ! diff -q "$before" "$after" >/dev/null; then
    echo "REFUSING to start continuous sync: local files changed during the pull-only pass:" >&2
    diff "$before" "$after" | grep '^[<>]' | awk '{print "  " $1 " " $3}' | sort -u -k2 | head -20 >&2
    echo "  (full: diff $before $after; pass log: $log)" >&2
    echo "  Sync is left STOPPED in pull-only mode. If those are Lloyd's own writes, not" >&2
    echo "  downloads, re-run this script. Otherwise step 3's snapshot restores them:" >&2
    echo "  scripts/backup/restore-vault.sh" >&2
    exit 1
  fi
  "$OB" sync-config --path "$VAULT" --mode bidirectional
  echo "first pass clean: $(wc -l < "$after") files unchanged; mode back to bidirectional"
fi

step "6. continuous sync"
"${SUPERVISORCTL[@]}" start agent-obsidian-sync
sleep 5
"${SUPERVISORCTL[@]}" status agent-obsidian-sync
