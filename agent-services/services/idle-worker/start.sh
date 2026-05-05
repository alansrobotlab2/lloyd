#!/bin/bash
# Start the idle worker service

SERVICE_DIR="/home/alansrobotlab/lloyd/agent-services/services/idle-worker"
LOG_DIR="/home/alansrobotlab/lloyd/agent-services/logs"
LOG_FILE="$LOG_DIR/idle-worker.log"

echo "=== Lloyd Idle Worker Service Starter ==="

# Check if already running via systemd
if systemctl is-active --quiet lloyd-idle-worker.service 2>/dev/null; then
    echo "Service is already running (systemctl status lloyd-idle-worker.service)"
    exit 0
fi

# Check for stale PID file
PID_FILE="$SERVICE_DIR/idle-worker.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Daemon process found (PID: $PID)"
        echo "If this is stale, stop with: systemctl stop lloyd-monitoring.service"
        exit 1
    else
        echo "Removing stale PID file..."
        rm -f "$PID_FILE"
    fi
fi

# Check if daemon script exists
if [ ! -f "$SERVICE_DIR/idle-worker.py" ]; then
    echo "ERROR: Daemon script not found: $SERVICE_DIR/idle-worker.py"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

echo "Starting monitoring daemon..."
echo "Log file: $LOG_FILE"

# Start daemon
nohup python3 "$SERVICE_DIR/idle-worker.py" >> "$LOG_FILE" 2>&1 &
DAEMON_PID=$!

echo "$DAEMON_PID" > "$PID_FILE"
echo "Started idle worker with PID: $DAEMON_PID"
echo ""
echo "Commands:"
echo "  Status: systemctl status lloyd-idle-worker.service"
echo "  Logs: journalctl -u lloyd-idle-worker.service -f"
echo "  Stop: systemctl stop lloyd-idle-worker.service"
