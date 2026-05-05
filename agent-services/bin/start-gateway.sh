#!/usr/bin/env bash
set -euo pipefail

# Starts the OpenClaw Gateway on port 18789
# Loads secrets from ~/.openclaw/.env

# Source the openclaw env file — gateway needs these as env vars for SecretRef resolution
if [[ -f "$HOME/.openclaw/.env" ]]; then
  set -a
  source "$HOME/.openclaw/.env"
  set +a
fi

export DISPLAY="${DISPLAY:-:1}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}"

export NODE_EXTRA_CA_CERTS="$HOME/.openclaw/certs/mc.crt"
export NODE_PATH="$HOME/.npm-global/lib/node_modules:$HOME/.npm-global/lib/node_modules/openclaw/node_modules"
export OPENCLAW_HOME="$HOME"
export OPENCLAW_CONFIG_PATH="$HOME/.openclaw/openclaw.json"
export OPENCLAW_STATE_DIR="$HOME/.openclaw"
export OPENCLAW_GATEWAY_PORT=18789
export OPENCLAW_NO_RESPAWN=1

# Kill any orphaned process on port 18789
if ss -tlnp 2>/dev/null | grep -q ":18789 "; then
  ORPHAN_PID=$(ss -tlnp | grep ":18789 " | grep -oP 'pid=\K[0-9]+' | head -1)
  if [[ -n "$ORPHAN_PID" ]]; then
    echo "Killing orphan process $ORPHAN_PID on port 18789"
    kill "$ORPHAN_PID" 2>/dev/null || true
    sleep 2
    kill -0 "$ORPHAN_PID" 2>/dev/null && kill -9 "$ORPHAN_PID" 2>/dev/null || true
    sleep 1
  fi
fi

rm -f /tmp/openclaw-gateway.lock

# Trim bloated session stores before startup (saves ~5-6s of structuredClone)
if [[ -x "$HOME/lloyd/agent-services/bin/session-store-trim.sh" ]]; then
  "$HOME/lloyd/agent-services/bin/session-store-trim.sh" 2>/dev/null || true
fi

exec "$HOME/.npm-global/bin/openclaw" gateway --port 18789
