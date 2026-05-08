#!/usr/bin/env bash
# Generate fresh LIVEKIT_API_KEY / LIVEKIT_API_SECRET into <repo>/.env.
#
# These are consumed by:
#   - app/config.py + agent-services/livekit_worker.py (via ${VAR} expansion in config.yaml)
#   - agent-services/bin/start-livekit-server.sh (envsubst into the LiveKit YAML)
#
# Existing LIVEKIT_API_* lines in .env are replaced; other lines are preserved.
# After running this, restart the affected services:
#   supervisorctl restart lloyd-mc:lloyd-backend lloyd-agent-worker agent-livekit-server
#
# Usage:
#   bash scripts/gen-livekit-secrets.sh         # rotate
#   bash scripts/gen-livekit-secrets.sh --print # show without writing
#   bash scripts/gen-livekit-secrets.sh --keep  # only generate if missing

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO/.env"

ACTION="${1:-rotate}"

KEY="lloyd_$(openssl rand -hex 6)"
SECRET="$(openssl rand -base64 48 | tr -d '\n=' | tr '/+' '_-' | head -c 48)"

if [[ "$ACTION" == "--print" ]]; then
    echo "LIVEKIT_API_KEY=$KEY"
    echo "LIVEKIT_API_SECRET=$SECRET"
    exit 0
fi

if [[ "$ACTION" == "--keep" && -f "$ENV_FILE" ]] && grep -q "^LIVEKIT_API_KEY=" "$ENV_FILE" && grep -q "^LIVEKIT_API_SECRET=" "$ENV_FILE"; then
    echo "[gen-livekit-secrets] LIVEKIT_API_* already present in $ENV_FILE — skipping (--keep)"
    exit 0
fi

# Preserve everything else in .env, swap out the LIVEKIT_API_* lines.
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
if [[ -f "$ENV_FILE" ]]; then
    grep -v "^LIVEKIT_API_" "$ENV_FILE" > "$TMP" || true
fi
{
    echo "LIVEKIT_API_KEY=$KEY"
    echo "LIVEKIT_API_SECRET=$SECRET"
} >> "$TMP"
mv "$TMP" "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo "[gen-livekit-secrets] wrote LIVEKIT_API_KEY + LIVEKIT_API_SECRET to $ENV_FILE"
echo "[gen-livekit-secrets] Restart services to pick up the new values:"
echo "  supervisorctl restart lloyd-mc:lloyd-backend lloyd-agent-worker agent-livekit-server"
