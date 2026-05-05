#!/usr/bin/env bash
set -euo pipefail

# One-time setup for Orpheus TTS venv.
# Creates venvs/orpheus/ (Python 3.11) and installs all dependencies.
#
# After running this, start the server with:
#   bash bin/start-orpheus-tts.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/venvs/orpheus"
PY="$VENV/bin/python"

echo "=== Orpheus TTS Setup ==="

echo "Creating venv at $VENV (Python 3.11)..."
uv venv "$VENV" --python 3.11 --seed

echo "Installing PyTorch (cu128)..."
uv pip install --python "$PY" \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo "Installing orpheus-speech (pulls in vllm + snac)..."
# orpheus-speech installs vllm as a dependency; let it pick the version it needs
"$PY" -m pip install orpheus-speech

echo "Installing FastAPI, uvicorn, scipy..."
uv pip install --python "$PY" fastapi uvicorn scipy

echo "Verifying imports..."
"$PY" -c "
from orpheus_tts import OrpheusModel, tokens_decoder_sync
import fastapi, uvicorn, scipy
print('  All imports OK')
print('  orpheus_tts: OrpheusModel, tokens_decoder_sync')
print('  fastapi, uvicorn, scipy: OK')
"

echo ""
echo "Done. Orpheus venv ready at $VENV"
echo "Run: bash bin/start-orpheus-tts.sh"
