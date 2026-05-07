# Lloyd Web Capture Chrome Extension

A Chrome extension that captures **any web page** (YouTube videos, articles, documentation), summarizes them with your local LLM, and saves the knowledge to your Obsidian vault.

## Architecture

```
Chrome Extension (popup) 
    → HTTP POST → Backend Server (localhost:8087)
    → Detects page type:
      - YouTube → transcript extraction
      - Article → content scraping
    → Local LLM (Qwen3.5-35B-A3B on localhost:8091)
    → Knowledge vault (~/obsidian/knowledge/ai/ or knowledge/software/)
```

## Prerequisites

- **uv** - Python package manager (`~/.local/bin/uv`)
- **Python 3.10+**
- **Local LLM** running on `http://localhost:8091` (Qwen3.5-35B-A3B)
- **youtube-transcript-api** (installed via uv when needed)

## Installation

### 1. Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify uv is in PATH
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

### 2. Load the Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top right)
3. Click **Load unpacked**
4. Select the `~/lloyd/chrome-extension` directory
5. The extension icon (blue circle with "L") should appear in your toolbar

### 3. Start the Backend Server

```bash
cd ~/lloyd/chrome-extension
chmod +x start-backend.sh
./start-backend.sh
```

The server will run on `http://localhost:8087`. Keep it running while using the extension.

### 4. Verify Setup

```bash
# Test YouTube capture
curl -X POST http://localhost:8087/capture \
  -H "Content-Type: application/json" \
  -d '{
    "pageType": "youtube",
    "videoId": "dQw4w9WgXcQ",
    "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "title": "Test Video"
  }'

# Test article capture
curl -X POST http://localhost:8087/capture \
  -H "Content-Type: application/json" \
  -d '{
    "pageType": "article",
    "url": "https://example.com/article",
    "title": "Test Article"
  }'
```

You should get an error about the content not existing (not a connection error).

## Usage

1. **Navigate to a YouTube video** you want to capture
2. **Click the extension icon** in your Chrome toolbar
3. The popup will show the video title and a "Capture to Vault" button
4. **Click "Capture to Vault"**
5. Wait for the status message:
   - ✅ "Captured to ~/obsidian/knowledge/ai/..." on success
   - ❌ Error message if something failed

## Output Format

Captured videos are saved to `~/obsidian/knowledge/ai/<slug>.md` with:

```yaml
---
type: reference
tags: [youtube, transcript, summary, ai]
source: https://youtube.com/watch?v=VIDEO_ID
video_id: VIDEO_ID
date: YYYY-MM-DD
summary: "First 150 chars of summary..."
---
```

Followed by:
- Full summary from LLM
- Complete transcript
- Metadata (URL, video ID, capture timestamp)

## Logs

Capture activity is logged to:
- `~/lloyd/_pipeline/youtube-capture.log`

## Troubleshooting

### Extension won't load
- Check that you selected the correct directory (`~/lloyd/chrome-extension`)
- Look for errors in `chrome://extensions/` (click "Details" on the extension)

### "Not on a YouTube page"
- Make sure you're on `youtube.com` or `youtu.be`
- The extension only works on YouTube domains

### Backend connection error
- Check if the backend server is running (`ps aux | grep server.py`)
- Verify port 8087 is available (`lsof -i :8087`)
- Check the log file for errors

### Transcript fetch failed
- Some videos don't have captions available
- Try a different video to confirm the setup works
- Check that `uv` can install packages

### LLM summarization failed
- Verify your local LLM is running on `http://localhost:8091`
- Check that the model `Qwen3.5-35B-A3B` is loaded
- Look at the backend logs for specific errors

### Vault save failed
- Check that `~/obsidian/knowledge/ai/` exists
- Verify write permissions
- Check the log file for the exact error

## Development

### Hot Reload

The extension doesn't auto-reload. After changing files:
1. Go to `chrome://extensions/`
2. Click the refresh icon on the Lloyd extension

### Backend Development

```bash
# Run server in debug mode
cd ~/lloyd/chrome-extension/backend
python3 -m pdb server.py

# Or just run directly
python3 server.py
```

### Testing Without Extension

```bash
# Manually trigger a capture
curl -X POST http://localhost:8087/capture \
  -H "Content-Type: application/json" \
  -d '{
    "videoId": "dQw4w9WgXcQ",
    "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "title": "Test Video"
  }'
```

## Future Enhancements

- [ ] Add transcription language selection
- [ ] Support for playlists
- [ ] Direct integration with Lloyd MCP (instead of HTTP)
- [ ] Customizable output paths and tags
- [ ] Batch capture from history
- [ ] Summary customization (length, style)

## License

MIT (same as the rest of your Lloyd setup)
