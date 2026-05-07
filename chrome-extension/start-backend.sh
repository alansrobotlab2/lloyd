#!/bin/bash
# Start the Lloyd YouTube Capture backend server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo "Starting Lloyd YouTube Capture backend..."
echo "Server will run on http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if port is available
if lsof -i :8080 &> /dev/null; then
    echo "Error: Port 8080 is already in use"
    lsof -i :8080
    exit 1
fi

# Start the server
cd "$BACKEND_DIR"
python3 server.py
