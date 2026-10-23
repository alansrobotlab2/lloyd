#!/usr/bin/env bash
# Snapshot the vault into a git repository that lives OUTSIDE the vault.
#
# Why this exists: the vault was deleted on 2026-09-10 and again on
# 2026-09-12, both times together with its own `.git`, and both times Obsidian
# Sync pushed the deletions to the cloud copy. What saved it was a tarball the
# deleting process happened to write first. Recovery should not depend on the
# culprit's manners.
#
# Every 15 minutes (lloyd-vault-backup.timer) the whole vault — notes, the
# vault's untracked files, `.obsidian`; never the vault's own `.git` — is
# committed to ~/.local/state/lloyd-vault-backup/vault.git. Git deduplicates,
# so a snapshot costs roughly what changed since the last one. This repository
# is the only copy of the vault that is not the vault: the btrfs snapshots Alan
# cited when closing #1142 cover ~/lloyd-data only
# (scripts/backup/snapshot-data.sh takes DATA="$HOME/lloyd-data"), and there is
# no remote by ruling.
#
# Two refusals, both so a wipe is never recorded as the new normal:
#   * the guardian's vault tripwire is set (vault-tripped.json) — an incident
#     is open, and the last snapshot is the one a restore wants;
#   * the vault holds under 90% of the files the last snapshot recorded — the
#     tripwire should have fired; this is the second net.
# A refusal exits 0 after saying why, so the timer does not flap to failed
# and hide the next real error; `journalctl --user -u lloyd-vault-backup`
# shows it.
#
# A third condition stops the run too, and it exits NON-ZERO on purpose: a git
# lock in the backup repo that this run may not safely remove. A killed
# `git add` leaves `index.lock` and a killed `git gc` leaves `gc.pid.lock`, and
# from then on every run dies at `git add` with "File exists" — status=128, 370
# runs in a row from 2026-10-16 11:03 to backlog #3078, the snapshot frozen at
# 2026-10-16 10:45 with 8145 files while the live vault grew to 8516, and the
# timer guard (#1891) still green because the timer really was firing: only the
# runs were failing. `lock_guard` removes a lock on two proofs and refuses
# without either, because "a lock is wedged so nothing can be snapshotted" is a
# failure the timer should be seen to have.
#
# Restore with scripts/backup/restore-vault.sh (into a side directory, never
# in place).
set -euo pipefail

VAULT="${LLOYD_VAULT_ROOT:-$HOME/obsidian}"
REPO="${LLOYD_VAULT_BACKUP_REPO:-$HOME/.local/state/lloyd-vault-backup/vault.git}"
MARKER="${LLOYD_VAULT_TRIP_MARKER:-$HOME/.local/state/lloyd-guardian/vault-tripped.json}"
MIN_FRACTION_PCT=90
# One timer period — `OnCalendar=*:0/15` on lloyd-vault-backup.timer, restated
# here because this script cannot read its own unit. It is the oldest a run that
# is still in flight can possibly be, which is what makes a lock older than this
# one dead. tests/test_backup_vault_locks.py re-derives the number from the unit
# and fails if the two ever disagree.
LOCK_STALE_SECONDS=900

log() { echo "backup-vault: $*"; }

if [[ -f "$MARKER" ]]; then
  log "vault tripwire is set ($MARKER) — not snapshotting a vault under incident"
  exit 0
fi
if [[ ! -d "$VAULT" ]]; then
  log "ERROR: $VAULT does not exist"
  exit 1
fi

g() { git --git-dir="$REPO" --work-tree="$VAULT" "$@"; }

if [[ ! -d "$REPO" ]]; then
  mkdir -p "$(dirname "$REPO")"
  git init --quiet --bare "$REPO"
  git --git-dir="$REPO" config core.bare false
  git --git-dir="$REPO" config user.name "lloyd-vault-backup"
  git --git-dir="$REPO" config user.email "vault-backup@localhost"
  # The vault's own ignore rules are for its own repo; a backup keeps it all.
  git --git-dir="$REPO" config core.excludesFile /dev/null
  git --git-dir="$REPO" config gc.auto 256
  log "initialised $REPO"
fi

# ── stale git locks, cleared ahead of any git call a lock can stop ────────────
# The git calls above run only when the repo does not exist yet, and a repo that
# does not exist cannot have a lock in it, so this is ahead of every call one can
# block. It sweeps `*.lock` rather than two named files because that is the
# property: any lock git may leave in the repo dir (`HEAD.lock`,
# `packed-refs.lock`, `config.lock`, and whatever a later git adds) has the same
# cause and the same two proofs available.
lock_holder_reason() {
  # Echo why a live process owns $1; echo nothing and return 1 when none does.
  local f=$1 pids pid pid_file
  if command -v fuser >/dev/null 2>&1; then
    if pids=$(fuser "$f" 2>/dev/null); then
      pids="${pids//[$'\n' ]/}"
      if [[ -n "$pids" ]]; then
        echo "open by pid $pids"
        return 0
      fi
    fi
  else
    # psmisc missing is not "no holder": without a probe, clearing is a guess.
    echo "fuser is not installed, so a holder cannot be ruled out"
    return 0
  fi
  # git's own pid file, and ONLY for a `*.pid.lock`: git renames such a lock to
  # `<base>.pid` holding the live gc's pid, so a gc in flight may have nothing
  # open by now and fuser would report an empty field. The suffix test is not
  # decoration — strip `.lock` from `index.lock` and the name you get is
  # `<repo>/index`, git's staging index, which is a binary file and not anyone's
  # pid file. Same-user truth only: kill -0 answers EPERM for a process owned by
  # someone else, and this timer and its git are one user.
  pid_file="${f%.lock}"
  if [[ "$f" == *.pid.lock && -f "$pid_file" ]]; then
    pid=$(<"$pid_file")
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "pid $pid in $(basename "$pid_file") is alive"
      return 0
    fi
  fi
  return 1
}

lock_guard() {
  # Clear a lock this run may safely take the repo over; refuse without proof.
  local f mtime age why pid_file
  for f in "$REPO"/*.lock; do
    [[ -e "$f" ]] || continue            # the glob stays literal with no match
    pid_file="${f%.lock}"
    if ! mtime=$(stat -c %Y -- "$f" 2>/dev/null); then
      log "REFUSING: $f exists but cannot be stat'ed, so its age cannot be proved"
      exit 1
    fi
    age=$(( $(date +%s) - mtime ))
    if why=$(lock_holder_reason "$f"); then
      log "REFUSING: $(basename "$f") in $REPO is held by a live process ($why) — not clearing a lock another run owns"
      exit 1
    fi
    if (( age < LOCK_STALE_SECONDS )); then
      log "REFUSING: $(basename "$f") is ${age}s old, inside the ${LOCK_STALE_SECONDS}s timer period — a snapshot started this cycle may still be writing it"
      exit 1
    fi
    rm -f -- "$f"
    log "cleared stale $(basename "$f") (${age}s old, past the ${LOCK_STALE_SECONDS}s timer period, no live holder)"
    # A pid file goes with the lock it belongs to and nowhere else, and only a
    # `*.pid.lock` has one: `index.lock`'s prefix-stripped name is `index`, git's
    # staging index, and deleting that would be the corruption this script exists
    # to survive rather than the thing it clears. A pid that no longer exists is
    # nothing to refuse on, and leaving a dead pid file behind lets a recycled
    # pid refuse this run forever on a number that has nothing to do with git.
    if [[ "$f" == *.pid.lock && -f "$pid_file" ]]; then
      rm -f -- "$pid_file"
      log "cleared stale $(basename "$pid_file") with it"
    fi
  done
}

lock_guard

count=$(find "$VAULT" -name .git -prune -o -type f -print | wc -l)
last_file="$REPO/lloyd-last-count"
if [[ -f "$last_file" ]]; then
  last=$(cat "$last_file")
  if [[ "$last" =~ ^[0-9]+$ ]] && (( last > 0 )) && (( count * 100 < last * MIN_FRACTION_PCT )); then
    log "REFUSING: vault holds $count files, the last snapshot held $last — not recording a shrunken vault"
    exit 0
  fi
fi

# --force: include files the vault's own .gitignore excludes. Git never adds a
# directory named .git, so the vault's repository is not copied into this one.
g add --all --force .
if g diff --cached --quiet 2>/dev/null && g rev-parse --verify -q HEAD >/dev/null; then
  log "no change ($count files)"
else
  g commit --quiet --no-verify -m "snapshot $(date -Is) files=$count"
  log "committed $(g rev-parse --short HEAD) ($count files)"
fi
# This write is the producer's per-run beat, not just bookkeeping: the guardian's
# snapshot check (agent-services/guardian/vaultwatch.py `snapshot_age_state`,
# #3184) weighs this file's mtime beside HEAD's %ct, so a `no change` delivery
# above still proves the timer is delivering. That reading only means anything
# because every refusal above — tripwire, lock_guard, shrink — exits before this
# line, so a stale or missing beat means a stopped producer and nothing else.
echo "$count" > "$last_file"
# Not `|| true`. That is how the 2026-10-14 17:15 gc lock went unnoticed: the
# commit succeeded, gc printed "fatal: Unable to create '…/gc.pid.lock': File
# exists", `|| true` turned that into exit 0, and a snapshot that had just lost
# its next gc looked healthy in the journal. The commit above stands either way,
# so this warns rather than fails — but it warns out loud, and the next run's
# lock_guard is what clears what a killed gc leaves.
g gc --auto --quiet || log "WARNING: git gc --auto failed; the snapshot above is committed, and the next run clears any lock this left"
