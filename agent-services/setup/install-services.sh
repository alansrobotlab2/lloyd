#!/usr/bin/env bash
set -euo pipefail

# Installs systemd user service files by symlinking from systemd/ to ~/.config/systemd/user/
# Then reloads the systemd daemon.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEMD_SRC="$PROJECT_DIR/systemd"
SYSTEMD_DST="$HOME/.config/systemd/user"

echo "=== Installing Systemd Service Files ==="

mkdir -p "$SYSTEMD_DST"

for f in "$SYSTEMD_SRC"/*.service; do
    name="$(basename "$f")"
    src="$(realpath "$f")"
    dst="$SYSTEMD_DST/$name"

    if [ -L "$dst" ] && [ "$(readlink "$dst")" = "$src" ]; then
        echo "  $name — already linked"
    else
        if [ -e "$dst" ]; then
            echo "  $name — replacing existing file"
            rm "$dst"
        fi
        ln -sf "$src" "$dst"
        echo "  $name — linked"
    fi
done

echo ""
echo "Reloading systemd daemon..."
systemctl --user daemon-reload

echo ""
echo "Done. Services installed:"
for f in "$SYSTEMD_SRC"/*.service; do
    name="$(basename "$f" .service)"
    status=$(systemctl --user is-active "$name" 2>/dev/null || echo "inactive")
    echo "  $name — $status"
done

echo ""
echo "Start all: systemctl --user start lloyd-llm lloyd-tts lloyd-voice-mode lloyd-voice-mcp lloyd-tool-mcp openclaw-gateway openclaw-cert lloyd-qmd-daemon lloyd-qmd-watcher"
echo "Enable at boot: systemctl --user enable lloyd-llm lloyd-tts lloyd-voice-mode lloyd-voice-mcp lloyd-tool-mcp openclaw-gateway openclaw-cert lloyd-qmd-daemon lloyd-qmd-watcher"
