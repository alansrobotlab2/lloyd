#!/usr/bin/env bash
set -euo pipefail

# Self-hosted LiveKit server for Lloyd voice/RTC.
# Downloads the binary on first run if missing.
# Sources secrets from <repo>/.env and renders the committed livekit.yaml
# (which uses ${VAR} placeholders) into a runtime file with values
# substituted, then execs the server with that.

LIVEKIT_VERSION="${LIVEKIT_VERSION:-1.11.0}"
LIVEKIT_BIN="${LIVEKIT_BIN:-$HOME/.local/bin/livekit-server}"

REPO_ROOT="/home/alansrobotlab/lloyd"
TEMPLATE="${LIVEKIT_CONFIG:-$REPO_ROOT/agent-services/conf/livekit.yaml}"
RUNTIME="$REPO_ROOT/agent-services/conf/livekit.yaml.runtime"
ENV_FILE="$REPO_ROOT/.env"

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

# Pull only the LIVEKIT_API_* values from .env. Don't `source` the file —
# other entries (e.g. model option names) may contain spaces and would
# fail to parse as bash assignments.
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r k v; do
        [[ -z "$k" || "$k" == \#* ]] && continue
        # Strip surrounding quotes if present
        v="${v%\"}"; v="${v#\"}"
        v="${v%\'}"; v="${v#\'}"
        case "$k" in
            LIVEKIT_API_KEY|LIVEKIT_API_SECRET) export "$k=$v" ;;
        esac
    done < "$ENV_FILE"
fi

if [[ -z "${LIVEKIT_API_KEY:-}" || -z "${LIVEKIT_API_SECRET:-}" ]]; then
    echo "[start-livekit-server] LIVEKIT_API_KEY / LIVEKIT_API_SECRET missing." >&2
    echo "[start-livekit-server] Run: bash scripts/gen-livekit-secrets.sh" >&2
    exit 1
fi

# Render the runtime config — only ${LIVEKIT_*} placeholders are substituted,
# leaving any other unrelated $vars intact.
envsubst '$LIVEKIT_API_KEY $LIVEKIT_API_SECRET' < "$TEMPLATE" > "$RUNTIME"
chmod 600 "$RUNTIME"

exec "$LIVEKIT_BIN" --config "$RUNTIME"
