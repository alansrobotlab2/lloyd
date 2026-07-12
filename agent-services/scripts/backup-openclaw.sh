#!/bin/bash
# OpenClaw Configuration Backup Script
# Backs up OpenClaw configuration to ~/lloyd/agent-services/config-backups/

set -e

BACKUP_DIR="$HOME/lloyd/agent-services/config-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_SUBDIR="$BACKUP_DIR/openclaw_$TIMESTAMP"

mkdir -p "$BACKUP_SUBDIR"

# Copy OpenClaw configuration
if [ -d "$HOME/.openclaw" ]; then
    cp -r "$HOME/.openclaw" "$BACKUP_SUBDIR/"
    echo "Backed up $HOME/.openclaw to $BACKUP_SUBDIR/"
else
    echo "No OpenClaw configuration found at $HOME/.openclaw"
    exit 1
fi

# Copy any cookie files
if [ -f "$HOME/.openclaw/cookies.json" ]; then
    cp "$HOME/.openclaw/cookies.json" "$BACKUP_SUBDIR/"
    echo "Backed up cookies.json to $BACKUP_SUBDIR/"
fi

echo "Backup complete: $BACKUP_SUBDIR"
echo "Contents:"
ls -la "$BACKUP_SUBDIR/"