#!/usr/bin/env bash
set -euo pipefail

# Self-hosted LiveKit server for Lloyd voice/RTC.
# Downloads the binary on first run if missing, then execs it under supervisord.

LIVEKIT_VERSION="${LIVEKIT_VERSION:-1.11.0}"
LIVEKIT_BIN="${LIVEKIT_BIN:-$HOME/.local/bin/livekit-server}"
LIVEKIT_CONFIG="${LIVEKIT_CONFIG:-/home/alansrobotlab/lloyd/agent-services/conf/livekit.yaml}"

if [[ ! -x "$LIVEKIT_BIN" ]]; then
    echo "[start-livekit-server] $LIVEKIT_BIN missing — downloading v${LIVEKIT_VERSION}" >&2
    mkdir -p "$(dirname "$LIVEKIT_BIN")"
    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT
    url="https://github.com/livekit/livekit/releases/download/v${LIVEKIT_VERSION}/livekit_${LIVEKIT_VERSION}_linux_amd64.tar.gz"
    curl -fsSL "$url" -o "$tmpdir/livekit.tar.gz"
    tar -xzf "$tmpdir/livekit.tar.gz" -C "$tmpdir"
    install -m 0755 "$tmpdir/livekit-server" "$LIVEKIT_BIN"
fi

exec "$LIVEKIT_BIN" --config "$LIVEKIT_CONFIG"
