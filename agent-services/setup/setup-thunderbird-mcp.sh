#!/usr/bin/env bash
# Setup script for Thunderbird MCP server
# Clones TKasperczyk/thunderbird-mcp into lloyd-services/services/thunderbird-mcp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$SCRIPT_DIR/../services/thunderbird-mcp"
REPO_URL="https://github.com/TKasperczyk/thunderbird-mcp.git"

echo "=== Thunderbird MCP Setup ==="

# Check prerequisites
command -v node >/dev/null 2>&1 || { echo "ERROR: Node.js is required (18+). Install with: sudo apt install nodejs"; exit 1; }
command -v thunderbird >/dev/null 2>&1 || { echo "ERROR: Thunderbird is required (102+)."; exit 1; }

NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "ERROR: Node.js 18+ required, found v$NODE_VERSION"
    exit 1
fi

echo "Prerequisites OK:"
echo "  Node.js: $(node -v)"
echo "  Thunderbird: $(thunderbird --version 2>/dev/null || echo 'installed')"

# Clone or update repo
if [ -d "$SERVICE_DIR/.git" ]; then
    echo "Updating existing clone..."
    cd "$SERVICE_DIR" && git pull --ff-only
else
    echo "Cloning thunderbird-mcp..."
    rm -rf "$SERVICE_DIR"
    git clone "$REPO_URL" "$SERVICE_DIR"
fi

# Install node dependencies (if any package.json exists)
if [ -f "$SERVICE_DIR/package.json" ]; then
    echo "Installing Node.js dependencies..."
    cd "$SERVICE_DIR" && npm install --production
fi

echo ""
echo "=== Clone complete ==="
echo ""
echo "Next steps:"
echo "  1. Install the Thunderbird extension:"
echo "     Open Thunderbird → Tools → Add-ons → Install from File"
echo "     Select: $SERVICE_DIR/dist/thunderbird-mcp.xpi"
echo "     Restart Thunderbird"
echo ""
echo "  2. Verify extension loaded:"
echo "     Thunderbird → Tools → Developer Tools → Error Console"
echo "     Look for: 'Thunderbird MCP server listening on port 8765'"
echo ""
echo "  3. Test the bridge:"
echo "     echo '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}' | node $SERVICE_DIR/mcp-bridge.cjs"
echo ""
echo "  4. Add to OpenClaw config (see docs/thunderbird-mcp-openclaw.json)"
echo ""
echo "Done."
