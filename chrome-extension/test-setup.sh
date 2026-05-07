#!/bin/bash
# Test script for Lloyd YouTube Capture setup

echo "=== Lloyd YouTube Capture Setup Test ==="
echo ""

# Check uv
echo "1. Checking uv..."
if command -v uv &> /dev/null; then
    echo "   ✓ uv is installed: $(uv --version)"
else
    echo "   ✗ uv is NOT installed"
    echo "   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check Python version
echo ""
echo "2. Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✓ $PYTHON_VERSION"
else
    echo "   ✗ Python 3 not found"
    exit 1
fi

# Check if backend directory exists
echo ""
echo "3. Checking backend directory..."
if [ -f ~/lloyd/chrome-extension/backend/server.py ]; then
    echo "   ✓ Backend server exists"
else
    echo "   ✗ Backend server not found"
    exit 1
fi

# Check if vault paths exist
echo ""
echo "4. Checking vault paths..."
if [ -d ~/obsidian/knowledge/ai ]; then
    echo "   ✓ Vault knowledge path exists"
else
    echo "   ⚠ Creating vault knowledge path..."
    mkdir -p ~/obsidian/knowledge/ai
    echo "   ✓ Created ~/obsidian/knowledge/ai"
fi

# Check if port 8087 is available
echo ""
echo "5. Checking port 8087..."
if lsof -i :8087 &> /dev/null; then
    echo "   ⚠ Port 8087 is already in use"
    echo "   Process: $(lsof -i :8087 | grep LISTEN | awk '{print $1}' | head -1)"
    echo "   Kill it with: kill $(lsof -i :8087 | grep LISTEN | awk '{print $2}')"
else
    echo "   ✓ Port 8087 is available"
fi

# Check if LLM endpoint is reachable
echo ""
echo "6. Checking LLM endpoint (localhost:8091)..."
if curl -s http://localhost:8091/health &> /dev/null || curl -s http://localhost:8091/v1/models &> /dev/null; then
    echo "   ✓ LLM endpoint is reachable"
    MODELS=$(curl -s http://localhost:8091/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1)
    if [ ! -z "$MODELS" ]; then
        echo "   Models: $MODELS"
    fi
else
    echo "   ⚠ LLM endpoint NOT reachable at http://localhost:8091"
    echo "   Make sure your local LLM is running"
fi

# Test backend server start
echo ""
echo "7. Testing backend server start..."
cd ~/lloyd/chrome-extension/backend
timeout 3 python3 server.py &
SERVER_PID=$!
sleep 2

if kill -0 $SERVER_PID 2>/dev/null; then
    echo "   ✓ Backend server starts successfully"
    kill $SERVER_PID 2>/dev/null
    sleep 1
else
    echo "   ✗ Backend server failed to start"
    exit 1
fi

echo ""
echo "=== Setup Test Complete ==="
echo ""
echo "Next steps:"
echo "1. Load the Chrome extension:"
echo "   - Open chrome://extensions/"
echo "   - Enable Developer mode"
echo "   - Click 'Load unpacked'"
echo "   - Select ~/lloyd/chrome-extension"
echo ""
echo "2. Start the backend server:"
echo "   ~/lloyd/chrome-extension/start-backend.sh"
echo ""
echo "3. Test with a YouTube video!"
