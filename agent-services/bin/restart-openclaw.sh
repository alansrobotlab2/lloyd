#!/usr/bin/env bash
set -euo pipefail

# Kill stale OpenClaw gateway process that blocks systemd restart,
# then restart the service cleanly.

# Find and kill any orphaned openclaw-gateway processes
PIDS=$(pgrep -f 'openclaw-gateway' 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    echo "Killing stale openclaw-gateway processes: $PIDS"
    kill $PIDS 2>/dev/null || true
    sleep 2
fi

# Also kill any openclaw processes holding the port
PORT_PID=$(ss -tlnp 2>/dev/null | grep ':18789' | grep -oP 'pid=\K[0-9]+' || true)
if [ -n "$PORT_PID" ]; then
    echo "Killing process holding port 18789: $PORT_PID"
    kill $PORT_PID 2>/dev/null || true
    sleep 2
fi

echo "Restarting openclaw-gateway.service..."
systemctl --user restart openclaw-gateway.service
sleep 5

if systemctl --user is-active --quiet openclaw-gateway.service; then
    echo "OpenClaw gateway is running."
    systemctl --user status openclaw-gateway.service --no-pager | head -6
else
    echo "ERROR: OpenClaw gateway failed to start."
    journalctl --user -u openclaw-gateway.service --since "10 sec ago" --no-pager
    exit 1
fi
