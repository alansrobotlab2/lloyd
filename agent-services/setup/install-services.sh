#!/usr/bin/env bash
set -euo pipefail

# Installs the systemd user unit by symlinking from systemd/ to ~/.config/systemd/user/,
# then reloads the systemd daemon.
#
# There are TWO units, and the second one is a deliberate exception.
#
#   agent-supervisord.service — runs supervisord, which manages every Lloyd
#     service (see supervisor/conf.d/). Individual per-service systemd units
#     were retired; do not expect lloyd-llm.service etc.
#
#   lloyd-guardian.service — the self-modification rollback watchdog. It
#     cannot live under supervisord: agent-supervisord.service sets
#     KillMode=control-group, so every supervisord child dies when that unit
#     restarts or supervisord crashes, which is precisely the situation the
#     guardian exists to survive. It also needs to be able to restart
#     supervisord itself, and a child cannot restart its own supervisor and
#     live. Plus supervisord parks a program in FATAL after startretries and
#     never un-parks it, while systemd Restart=always never gives up.
#
# Anything that is not a watchdog still belongs in supervisor/conf.d/.
#
# Lingering must also be enabled or supervisord dies at logout and never starts
# at boot:
#   sudo loginctl enable-linger "$USER"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEMD_SRC="$PROJECT_DIR/systemd"
SYSTEMD_DST="$HOME/.config/systemd/user"

echo "=== Installing Systemd Service Files ==="

mkdir -p "$SYSTEMD_DST"

for f in "$SYSTEMD_SRC"/*.service "$SYSTEMD_SRC"/*.timer; do
    [ -e "$f" ] || continue
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

LINGER=$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo "no")
if [ "$LINGER" != "yes" ]; then
    echo ""
    echo "WARNING: lingering is not enabled for $USER."
    echo "         supervisord will stop at logout and will not start at boot."
    echo "         Enable it with:  sudo loginctl enable-linger $USER"
fi

echo ""
echo "Enable and start:"
echo "  systemctl --user enable --now agent-supervisord.service"
echo ""
echo "Then check the services supervisord manages:"
echo "  $HOME/.local/share/uv/tools/supervisor/bin/supervisorctl \\"
echo "    -c $PROJECT_DIR/supervisor/supervisord.conf status"
