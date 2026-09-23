#!/usr/bin/env bash
# One-shot production cutover: runtime data out of ~/lloyd into ~/lloyd-data.
# architecture/data-home.md is the design; this is the operator's half, run once
# by a person (Alan), with the stack's consent to go down for ~25 minutes.
#
#   bash ~/lloyd-wt/data-home/scripts/maintenance/cutover_data_home.sh
#
# Why the whole supervisord unit restarts, not `round restart`: supervisord
# reopens a program's log only when it restarts that program, and its own log
# only when the unit restarts, and every log moves. The guardian is stopped
# first so the outage is not read as a crash.
#
# Order, and what is reversible:
#   1. preflight (nothing changed)             4. fast-forward ~/lloyd
#   2. create the ~/lloyd-data subvolume        5. migrate (originals HELD, not deleted)
#   3. stop guardian, timers, the unit          6. qmd config, vault instructions
#   7. start the unit, wait for health, verify the running commit
#   8. start guardian + timers + snapshot timer, first snapshot, bless + floor
# A failure before step 5 restarts the old stack as it was. From step 5 on the
# data is already in ~/lloyd-data and the old tree layout is in the hold dir;
# the script stops and says where it stopped.
set -euo pipefail

BRANCH_REPO="${BRANCH_REPO:-$HOME/lloyd-sandbox}"
BRANCH="${BRANCH:-data-home}"
LIVE="$HOME/lloyd"
DATA="$HOME/lloyd-data"
PY="$LIVE/.venvs/lloyd/bin/python"
SC=("$HOME/.local/share/uv/tools/supervisor/bin/supervisorctl" -c "$LIVE/agent-services/supervisor/supervisord.conf")
TIMERS=(lloyd-guardian-nag.timer lloyd-graph-backup.timer lloyd-groundskeeper-survey.timer lloyd-qmd-cleanup.timer lloyd-vault-backup.timer)
LOG="$HOME/lloyd-data-cutover-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n!! %s\n!! transcript: %s\n' "$*" "$LOG"; exit 1; }
mem_gib() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
MIGRATED=0
restore_old_stack() {
  if (( MIGRATED )); then
    echo "!! stopped AFTER the migration — the data is in $DATA, originals in ~/lloyd-data-migration-hold/."
    echo "!! Fix the cause and re-run from the step that failed; do not restart the old commit."
    return
  fi
  echo "!! failed before the migration: bringing the old stack back as it was"
  systemctl --user start agent-supervisord || true
  systemctl --user start lloyd-guardian || true
  systemctl --user start "${TIMERS[@]}" || true
}
trap 'rc=$?; (( rc )) && restore_old_stack; exit $rc' EXIT

# ── 1. preflight ─────────────────────────────────────────────────────────────
say "1. preflight"
TARGET=$(git -C "$BRANCH_REPO" rev-parse "$BRANCH")
HEAD=$(git -C "$LIVE" rev-parse HEAD)
echo "live HEAD $HEAD, target $TARGET ($BRANCH in $BRANCH_REPO)"
git -C "$BRANCH_REPO" merge-base --is-ancestor "$HEAD" "$TARGET" \
  || die "live main moved past the branch base; rebase $BRANCH onto live main first"
[[ -z "$(git -C "$LIVE" status --porcelain --untracked-files=no)" ]] || die "live tree has uncommitted tracked edits"
[[ -f "$DATA/.lloyd-data-root" ]] && die "$DATA already carries its marker — this cutover has run"
if pgrep -f "scripts.automod.round (land|gate)" >/dev/null; then die "an automod gate or landing is running"; fi
for m in "$HOME"/.local/state/lloyd-automod/rounds/*/gate.running "$HOME"/.local/state/lloyd-automod/rounds/*/land.running; do
  [[ -e "$m" ]] || continue
  pid=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('pid',''))" "$m" 2>/dev/null || true)
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && die "live marker $m (pid $pid)"
done
curl -sf localhost:8080/api/workers/status | python3 -c '
import json,sys; p=json.load(sys.stdin)["pool"]
print("pool paused:", p["paused"], p.get("paused_by"), "in flight:", p["in_flight_count"])
sys.exit(0 if p["in_flight_count"] == 0 else 3)' || die "worker jobs are in flight; pause the pool and wait"
[[ "$(findmnt -no FSTYPE -T "$HOME")" == btrfs ]] || die "$HOME is not btrfs"
echo "preflight ok"

# ── 2. the subvolume ─────────────────────────────────────────────────────────
say "2. $DATA"
if [[ -d "$DATA" ]]; then
  [[ -z "$(ls -A "$DATA")" ]] || die "$DATA exists and is not empty"
  btrfs subvolume show "$DATA" >/dev/null 2>&1 || die "$DATA exists but is not a btrfs subvolume"
else
  btrfs subvolume create "$DATA"
fi

# ── 3. stop ──────────────────────────────────────────────────────────────────
say "3. stopping the guardian, timers and agent-supervisord"
systemctl --user stop lloyd-guardian
systemctl --user stop "${TIMERS[@]}"
systemctl --user stop agent-supervisord
for i in $(seq 1 120); do
  systemctl --user is-active --quiet agent-supervisord || break; sleep 1
done
systemctl --user is-active --quiet agent-supervisord && die "agent-supervisord did not stop"
for i in $(seq 1 60); do
  pgrep -f "$LIVE/(\.venvs|agent-services|qmd)" >/dev/null || break; sleep 1
done
pgrep -af "$LIVE/(\.venvs|agent-services|qmd)" && die "processes from $LIVE are still running"
echo "stack is down"

# ── 4. fast-forward ──────────────────────────────────────────────────────────
say "4. fast-forward ~/lloyd to $TARGET"
git -C "$LIVE" fetch -q "$BRANCH_REPO" "$BRANCH"
git -C "$LIVE" merge --ff-only FETCH_HEAD
[[ "$(git -C "$LIVE" rev-parse HEAD)" == "$TARGET" ]] || die "HEAD is not the target after the merge"

# ── 5. migrate ───────────────────────────────────────────────────────────────
say "5. migrating runtime data"
/usr/bin/python3 "$LIVE/scripts/migrate_data_home.py"
/usr/bin/python3 "$LIVE/scripts/migrate_data_home.py" --apply
MIGRATED=1
mkdir -p "$DATA/logs/services" "$DATA/data" "$DATA/eval/baselines" "$DATA/voice_profiles"

# ── 6. qmd config and vault instructions ─────────────────────────────────────
say "6. qmd collections and vault instructions"
QMD="$HOME/.config/qmd/index.yml"
cp -a "$QMD" "$QMD.pre-data-home"
sed -i "s|path: $HOME/lloyd/autonomy-runs|path: $HOME/lloyd-data/autonomy-runs|; s|path: $HOME/lloyd/_pipeline/vault-derived/sessions|path: $HOME/lloyd-data/_pipeline/vault-derived/sessions|" "$QMD"
grep -n "lloyd-data" "$QMD"
if grep -q "path: $HOME/lloyd/" "$QMD"; then die "qmd still names a path in the tree"; fi
bash "$LIVE/scripts/backup/backup-vault.sh"
"$PY" "$LIVE/scripts/maintenance/rewrite_vault_data_paths.py" --apply | tail -3

# ── 7. start and verify ──────────────────────────────────────────────────────
say "7. starting agent-supervisord (waiting for MemAvailable >= 150 GiB first; now $(mem_gib) GiB)"
for i in $(seq 1 120); do (( $(mem_gib) >= 150 )) && break; sleep 5; done
(( $(mem_gib) >= 150 )) || die "MemAvailable stayed under 150 GiB — not booting the primary beside a stale table"
systemctl --user start agent-supervisord
for i in $(seq 1 90); do
  body=$(curl -sf localhost:8080/health || true)
  [[ -n "$body" ]] && break; sleep 5
done
running=$(printf '%s' "${body:-}" | python3 -c 'import json,sys; d=sys.stdin.read(); print(json.loads(d).get("commit","") if d.strip() else "")')
[[ "$running" == "$TARGET" ]] || die "backend reports commit '$running', expected $TARGET"
echo "backend up at $running"
curl -sf localhost:8080/api/workers/status | python3 -c '
import json,sys; p=json.load(sys.stdin)["pool"]; print("pool paused:", p["paused"], p.get("paused_by"))'
echo "waiting for the primary (up to 25 min)"
for i in $(seq 1 300); do curl -sf localhost:8096/health >/dev/null && break; sleep 5; done
curl -sf localhost:8096/health >/dev/null && echo "primary healthy" || echo "!! primary not healthy yet — check ${DATA}/logs/services/agent-llm-primary.err"
"${SC[@]}" status || true

# ── 8. guardian, timers, snapshots, bless ────────────────────────────────────
say "8. guardian, timers, snapshots, bless"
for u in lloyd-data-snapshot.service lloyd-data-snapshot.timer; do
  ln -sfn "$LIVE/agent-services/systemd/$u" "$HOME/.config/systemd/user/$u"
done
systemctl --user daemon-reload
systemctl --user start lloyd-guardian
systemctl --user start "${TIMERS[@]}"
systemctl --user enable --now lloyd-data-snapshot.timer
sleep 10   # one guardian tick arms the data watch before the first snapshot
systemctl --user start lloyd-data-snapshot.service
journalctl --user -u lloyd-data-snapshot.service -n 3 --no-pager | tail -2
cd "$LIVE"
"$PY" -m scripts.automod.round bless --note "data-home cutover: runtime data moved to ~/lloyd-data" | tail -3 || echo "!! bless refused — see above"
"$PY" - <<PYEOF
from scripts.automod import state as S
lkg = S.read_lkg() or {}
if lkg.get("commit") == "$TARGET":
    S.write_lkg("$TARGET", floor="$TARGET")
    print("LKG floor raised to $TARGET: no rollback may land on the in-tree data layout")
else:
    print("!! LKG is", lkg.get("commit"), "- floor NOT raised")
PYEOF
/usr/bin/python3 "$HOME/.local/state/lloyd-guardian/bin/datawatch.py" status | head -20
if /usr/bin/python3 "$HOME/.local/state/lloyd-guardian/bin/datawatch.py" strays; then echo "no runtime data left in $LIVE"; else echo "!! runtime names still in $LIVE (above)"; fi
say "done — transcript: $LOG"
