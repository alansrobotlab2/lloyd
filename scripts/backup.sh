#!/bin/bash
# Robust backup script with verification, logging, and retention
# Backs up critical directories to compressed archives with rotation

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────
BACKUP_BASE="/home/alansrobotlab/backups"
LOG_FILE="$BACKUP_BASE/backup.log"
DATE=$(date +%Y%m%d_%H%M%S)
DATE_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
BACKUP_FILE="$BACKUP_BASE/backup_$DATE.tar.gz"
CHECKSUM_FILE="$BACKUP_BASE/backup_$DATE.tar.gz.sha256"

# Directories to back up (add/remove as needed)
SOURCE_DIRS=(
    "/home/alansrobotlab/obsidian"
    "/home/alansrobotlab/lloyd/scripts"
)

# Retention: keep last N backups
KEEP_LAST=14

# Notification (optional: set EMAIL or leave empty)
EMAIL=""

# ── Helpers ────────────────────────────────────────────────────────
log() {
    echo "[$DATE_HUMAN] $*" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $*" >&2
}

send_alert() {
    if [ -n "$EMAIL" ]; then
        echo "$1" | mail -s "Backup Alert: $2" "$EMAIL" 2>/dev/null || true
    fi
}

# ── Setup ────────────────────────────────────────────────────────
mkdir -p "$BACKUP_BASE"

# Verify source directories exist
for dir in "${SOURCE_DIRS[@]}"; do
    if [ ! -e "$dir" ]; then
        error "Source directory does not exist: $dir"
        send_alert "Backup failed: missing directory $dir" "Missing Directory"
        exit 1
    fi
done

# ── Backup ────────────────────────────────────────────────────────
log "Starting backup: $BACKUP_FILE"

# Create tar archive with relative paths for clean extraction
# Use -C to make archives extractable into backups/
tar_args=()
for dir in "${SOURCE_DIRS[@]}"; do
    tar_args+=("-C" "$(dirname "$dir")" "$(basename "$dir")")
done

if tar -czf "$BACKUP_FILE" "${tar_args[@]}"; then
    log "Archive created: $(du -h "$BACKUP_FILE" | cut -f1)"
else
    error "tar command failed"
    send_alert "Backup archive creation failed" "Archive Failure"
    rm -f "$BACKUP_FILE"
    exit 1
fi

# ── Verification ─────────────────────────────────────────────────
# 1. SHA-256 checksum
if sha256sum "$BACKUP_FILE" > "$CHECKSUM_FILE"; then
    log "Checksum generated"
else
    error "Checksum generation failed"
    send_alert "Backup checksum generation failed" "Checksum Failure"
    exit 1
fi

# 2. Test archive integrity (list contents without extracting)
if tar -tzf "$BACKUP_FILE" >/dev/null 2>&1; then
    log "Archive integrity verified"
else
    error "Archive integrity check FAILED"
    send_alert "Backup archive is corrupted" "Integrity Failure"
    rm -f "$BACKUP_FILE" "$CHECKSUM_FILE"
    exit 1
fi

# ── Cleanup old backups ─────────────────────────────────────────
log "Rotating backups (keeping last $KEEP_LAST)"
cd "$BACKUP_BASE" || exit 1

# Remove old tar.gz files beyond retention count
ls -1t backup_*.tar.gz 2>/dev/null | tail -n +$((KEEP_LAST + 1)) | while read -r old; do
    rm -f "$old" "${old}.sha256"
    log "Removed old backup: $old"
done

log "Backup complete. Total backups: $(ls -1 backup_*.tar.gz 2>/dev/null | wc -l)"
echo "[$DATE_HUMAN] Backup successful: $BACKUP_FILE" >> "$LOG_FILE"