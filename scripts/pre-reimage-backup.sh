#!/usr/bin/env bash
# Collect everything that exists ONLY on this disk before reimaging.
#
# The vault's on-box copies — lloyd-vault-backup.timer's 15-minute git
# snapshots and lloyd-data-snapshot.timer's hourly btrfs snapshots — sit on
# the same disk as the vault, and nothing ships them anywhere (off-box is
# #1142). This script gathers them and every untracked runtime asset below
# into one directory you can copy off-box.
#
# Runtime data (sessions, _pipeline/, the databases, baselines, voice profiles,
# tool overrides, logs) lives in the data root, `${LLOYD_DATA:-~/lloyd-data}`
# (app/paths.py), and is copied WHOLE into <dest>/lloyd-data/ — from the newest
# read-only btrfs snapshot when one exists, since a snapshot is atomic and its
# SQLite files open like after a power cut. See the block below for the rest.
#
# Usage:
#   bash scripts/pre-reimage-backup.sh /run/media/you/external/lloyd-preimage
#   INCLUDE_MODELS=1 bash scripts/pre-reimage-backup.sh <dest>   # +311GB of LLM weights
#   INCLUDE_SESSIONS=0 bash scripts/pre-reimage-backup.sh <dest> # skip 725MB of sessions
#   INCLUDE_PIPELINE_BULK=1 bash scripts/pre-reimage-backup.sh <dest>
#       # + ~60MB of regenerable autoresearch variants and _debug dumps
#
# Everything here is documented in SETUP.md Part 0.

set -euo pipefail

DEST="${1:-}"
if [[ -z "$DEST" ]]; then
    echo "usage: $0 <destination-directory>" >&2
    exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR="$HOME"
DATA="${LLOYD_DATA:-$HOME/lloyd-data}"
SNAPSHOTS="${LLOYD_DATA_SNAPSHOTS:-$HOME/.lloyd-data-snapshots}"
INCLUDE_MODELS="${INCLUDE_MODELS:-0}"
INCLUDE_SESSIONS="${INCLUDE_SESSIONS:-1}"
INCLUDE_PIPELINE_BULK="${INCLUDE_PIPELINE_BULK:-0}"

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"

if [[ "$DEST" == "$HOME_DIR"/* ]]; then
    echo "WARNING: $DEST is inside \$HOME — it will be erased by the reimage."
    echo "         Pass an external mount point instead."
    if [[ -t 0 ]]; then
        read -rp "Continue anyway? [y/N] " yn
        [[ "$yn" == [yY] ]] || exit 1
    fi
fi

log() { echo "[pre-reimage] $*"; }
missing=()

# copy <label> <source> <dest-subpath> [extra rsync args…]
copy() {
    local label="$1" src="$2" sub="$3"
    shift 3
    if [[ ! -e "$src" ]]; then
        missing+=("$label ($src)")
        log "SKIP  $label — not found at $src"
        return
    fi
    log "COPY  $label"
    mkdir -p "$DEST/$(dirname "$sub")"
    rsync -a --delete "$@" "$src" "$DEST/$sub"
}

log "destination: $DEST"

# ── Vault: notes + local git history (no remote exists) + plugin state ──
copy "obsidian vault"          "$HOME_DIR/obsidian/"                              "obsidian/"

# ── Secrets and gitignored config ──
copy ".env"                    "$REPO/.env"                                       "lloyd/.env"
copy "qmd collections"         "$HOME_DIR/.config/qmd/index.yml"                  "config/qmd/index.yml"

# ── mTLS material (regenerating the CA invalidates every enrolled device) ──
copy "mTLS certs"              "$REPO/agent-services/cert/"                       "lloyd/agent-services/cert/"

# ── Wake-word models are now TRACKED in the repo (force-added past the
#    unanchored models/ ignore rule), so they need no backup. Copied anyway as
#    cheap insurance — 6 MB, and they are unrecoverable if the repo is ever lost.
copy "wakeword models"         "$REPO/agent-services/models/"                     "lloyd/agent-services/models/"

# ── Vendored TTS repo: cloned voice (untracked) + patches (tracked, re-captured) ──
copy "TTS voice library"       "$REPO/agent-services/services/tts/qwen3-tts/voice_library/" \
                               "lloyd/agent-services/services/tts/qwen3-tts/voice_library/"

TTS_DIR="$REPO/agent-services/services/tts/qwen3-tts"
# The patch is committed at agent-services/services/tts/qwen3-tts-local.patch;
# re-capture here so the backup reflects the tree as it stands right now.
if [[ -d "$TTS_DIR/.git" ]]; then
    log "COPY  TTS local patches (diff vs upstream)"
    mkdir -p "$DEST/lloyd/agent-services/services/tts"
    git -C "$TTS_DIR" diff > "$DEST/lloyd/agent-services/services/tts/qwen3-tts-local.patch"
    git -C "$TTS_DIR" log --oneline -1 \
        > "$DEST/lloyd/agent-services/services/tts/qwen3-tts-upstream-commit.txt"
else
    missing+=("TTS local patches ($TTS_DIR/.git)")
    log "SKIP  TTS local patches — $TTS_DIR is not a git checkout"
fi

# ── The data root, whole ──
# Everything that used to be copied piece by piece out of the tree — voice
# profiles, tool overrides, the autoresearch ledger/rounds/snapshots, the
# memory-graph evidence, trajectories, metrics, sessions — is under one root now,
# so it goes as one rsync. What is left out, and why:
#
#   * The live databases when no snapshot exists. kg.sqlite, usage.db,
#     workers.db and research.db are WAL-mode, and rsync of a database taken
#     mid-write is not restorable (SETUP.md Part 0). A btrfs snapshot is atomic,
#     so from one they are copied like everything else; from the live root they
#     are skipped and named in the manifest. The knowledge graph also always has
#     its own vehicle, the newest daily tarball below (#947).
#   * `_pipeline/backups/daily/` except its newest tarball: the 30-day window
#     stays on-box, and a restore wants the newest one.
#   * `memory-graph/store-backups/` (1.2GB of raw db copies): the tarball holds a
#     restorable store.
#   * Opt-outs: `sessions/` under INCLUDE_SESSIONS=0, and the regenerable
#     autoresearch `variants/` + `_debug/` unless INCLUDE_PIPELINE_BULK=1.
DATA_SRC=""
if [[ -d "$SNAPSHOTS" ]]; then
    newest_snap="$(ls -1 "$SNAPSHOTS" 2>/dev/null | sort | tail -1)"
    [[ -n "$newest_snap" ]] && DATA_SRC="$SNAPSHOTS/$newest_snap"
fi
data_excludes=(--exclude=/_pipeline/backups/daily/ --exclude=store-backups/)
if [[ -n "$DATA_SRC" ]]; then
    log "data root: newest snapshot $DATA_SRC"
else
    DATA_SRC="$DATA"
    log "data root: no snapshot under $SNAPSHOTS — copying live $DATA without its databases"
    data_excludes+=(--exclude='*.sqlite' --exclude='*.sqlite-wal' --exclude='*.sqlite-shm'
                    --exclude='*.db' --exclude='*.db-wal' --exclude='*.db-shm')
    if [[ -d "$DATA" ]]; then
        missing+=("live databases under $DATA (no btrfs snapshot to copy them from consistently)")
    fi
fi
if [[ "$INCLUDE_SESSIONS" != "1" ]]; then
    data_excludes+=(--exclude=/sessions/)
    log "SKIP  sessions (INCLUDE_SESSIONS=0)"
fi
if [[ "$INCLUDE_PIPELINE_BULK" != "1" ]]; then
    data_excludes+=(--exclude=/_pipeline/research/variants/ --exclude=/_pipeline/research/_debug/)
    log "SKIP  autoresearch variants + _debug (INCLUDE_PIPELINE_BULK=1 to include, ~60MB, both regenerable)"
fi
copy "data root"                "$DATA_SRC/"                                       "lloyd-data/" "${data_excludes[@]}"

# ── Knowledge-graph daily snapshot (#947): one consistent file carrying the
#    staged store, its json export, memory-graph/ and facts/, built nightly by
#    scripts/backup/backup-graph.sh from KGStore.backup().
PIPELINE="$DATA/_pipeline"
if compgen -G "$PIPELINE/backups/daily/graph-*.tar.gz" >/dev/null; then
    latest_graph="$(ls -1t "$PIPELINE"/backups/daily/graph-*.tar.gz | head -1)"
    copy "KG daily snapshot"      "$latest_graph"                                  "lloyd-data/_pipeline/backups/daily/$(basename "$latest_graph")"
else
    missing+=("KG daily snapshot ($PIPELINE/backups/daily/graph-*.tar.gz)")
    log "SKIP  KG daily snapshot — no graph-*.tar.gz in $PIPELINE/backups/daily"
fi

# ── The vault's 15-minute git snapshots (lloyd-vault-backup.timer) ──
# The vault copy above carries the vault's own .git; this repository is the
# history a wipe of ~/obsidian cannot reach, and it exists nowhere else. Its
# absence is MISSING rather than SKIP: every other gap here is optional or
# regenerable, this one means the timer has never run on this box.
VAULT_BACKUP_REPO="${LLOYD_VAULT_BACKUP_REPO:-$HOME_DIR/.local/state/lloyd-vault-backup/vault.git}"
if [[ -d "$VAULT_BACKUP_REPO" ]]; then
    copy "vault snapshot repo"  "$VAULT_BACKUP_REPO/"                              "vault-backup/vault.git/"
else
    missing+=("vault snapshot repo ($VAULT_BACKUP_REPO) — lloyd-vault-backup.timer has never run here")
    log "MISSING vault snapshot repo — $VAULT_BACKUP_REPO does not exist; lloyd-vault-backup.timer has never run here" >&2
fi

# ── LLM weights (opt-in: ~311GB) ──
if [[ "$INCLUDE_MODELS" == "1" ]]; then
    copy "LLM models"          "$REPO/agent-services/llm/models/"                 "lloyd/agent-services/llm/models/"
    copy "TTS model weights"   "$TTS_DIR/models/"                                 "lloyd/agent-services/services/tts/qwen3-tts/models/"
else
    log "SKIP  LLM + TTS model weights (INCLUDE_MODELS=1 to include, ~315GB)"
fi

# ── Manifest ──
{
    echo "Lloyd pre-reimage backup"
    echo "created: $(date -Is)"
    echo "host:    $(uname -n)"
    echo "repo:    $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "data:    $DATA_SRC"
    echo "switches: INCLUDE_MODELS=$INCLUDE_MODELS INCLUDE_SESSIONS=$INCLUDE_SESSIONS INCLUDE_PIPELINE_BULK=$INCLUDE_PIPELINE_BULK"
    echo
    echo "Restore instructions: SETUP.md"
    echo
    echo "Contents:"
    du -sh "$DEST"/* 2>/dev/null | sed 's/^/  /'
    # The destination gets read later, from another machine, where the terminal
    # scrollback that carried the SKIP lines no longer exists.
    if (( ${#missing[@]} )); then
        echo
        echo "NOT captured (confirm each is genuinely absent before wiping):"
        printf '  - %s\n' "${missing[@]}" | sed 's/^/  /'
    fi
} > "$DEST/MANIFEST.txt"

log "done — $(du -sh "$DEST" | cut -f1) at $DEST"

if (( ${#missing[@]} )); then
    echo
    log "The following were NOT captured:"
    printf '  - %s\n' "${missing[@]}"
    log "Confirm each is genuinely absent before wiping the disk."
fi

echo
log "Verify the copy, then confirm it is readable from another machine."
