#!/usr/bin/env bash
# Collect everything that exists ONLY on this disk before reimaging.
#
# The daily backup (scripts/backup.sh) covers ~/obsidian and ~/lloyd/scripts,
# but writes to ~/backups on the same disk — and it misses the untracked
# runtime assets below. This script gathers all of it into one directory you
# can copy off-box.
#
# Usage:
#   bash scripts/pre-reimage-backup.sh /run/media/you/external/lloyd-preimage
#   INCLUDE_MODELS=1 bash scripts/pre-reimage-backup.sh <dest>   # +311GB of LLM weights
#   INCLUDE_SESSIONS=0 bash scripts/pre-reimage-backup.sh <dest> # skip 725MB of sessions
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
INCLUDE_MODELS="${INCLUDE_MODELS:-0}"
INCLUDE_SESSIONS="${INCLUDE_SESSIONS:-1}"

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

# copy <label> <source> <dest-subpath>
copy() {
    local label="$1" src="$2" sub="$3"
    if [[ ! -e "$src" ]]; then
        missing+=("$label ($src)")
        log "SKIP  $label — not found at $src"
        return
    fi
    log "COPY  $label"
    mkdir -p "$DEST/$(dirname "$sub")"
    rsync -a --delete "$src" "$DEST/$sub"
}

log "destination: $DEST"

# ── Vault: notes + local git history (no remote exists) + plugin state ──
copy "obsidian vault"          "$HOME_DIR/obsidian/"                              "obsidian/"

# ── Secrets and gitignored config ──
copy ".env"                    "$REPO/.env"                                       "lloyd/.env"
copy "tool_overrides.yaml"     "$REPO/data/tool_overrides.yaml"                   "lloyd/data/tool_overrides.yaml"
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

# ── Voice profiles (speaker identification) ──
copy "voice profiles"          "$REPO/voice_profiles/"                            "lloyd/voice_profiles/"

# ── Conversation history ──
if [[ "$INCLUDE_SESSIONS" == "1" ]]; then
    copy "sessions"            "$REPO/sessions/"                                  "lloyd/sessions/"
else
    log "SKIP  sessions (INCLUDE_SESSIONS=0)"
fi

# ── Latest daily backup archive ──
if compgen -G "$HOME_DIR/backups/backup_*.tar.gz" >/dev/null; then
    latest="$(ls -1t "$HOME_DIR"/backups/backup_*.tar.gz | head -1)"
    log "COPY  latest daily backup ($(basename "$latest"))"
    mkdir -p "$DEST/backups"
    cp -a "$latest" "$DEST/backups/"
    [[ -f "$latest.sha256" ]] && cp -a "$latest.sha256" "$DEST/backups/"
else
    missing+=("daily backup archive ($HOME_DIR/backups/backup_*.tar.gz)")
    log "SKIP  daily backup archive — none found"
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
    echo "models:  INCLUDE_MODELS=$INCLUDE_MODELS"
    echo
    echo "Restore instructions: SETUP.md"
    echo
    echo "Contents:"
    du -sh "$DEST"/* 2>/dev/null | sed 's/^/  /'
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
