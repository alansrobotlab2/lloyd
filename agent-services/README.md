# Lloyd Services — Voice Pipeline MCP Server

Lloyd services provide a comprehensive voice pipeline for OpenClaw, including wake word detection, speech recognition, speaker diarization, and text-to-speech synthesis.

## Quick Start

### 1. Start the MCP Server

```bash
cd ~/Projects/lloyd-services
python3 voice_mcp_server.py
```

The MCP server starts on `http://127.0.0.1:8092` by default.

### 2. Start the TUI (Optional)

```bash
skill: lloyd-voice-tui start
```

Or directly:
```bash
cd ~/Projects/lloyd-services
python3 voice_tui.py
```

## Components

### voice_mcp_server.py

FastMCP server exposing voice pipeline capabilities:

- **Status**: `voice_status()` — Get pipeline state, last transcript, speaker info
- **Toggle**: `voice_toggle()` — Enable/disable the voice pipeline
- **ASR**: `voice_last_utterance()` — Get last heard utterance with transcript
- **Correction**: `voice_correct_transcript()` — Fix ASR errors
- **Diarization**: `voice_enroll_speaker()` / `voice_list_speakers()` — Speaker identification
- **TTS**: `voice_say()` — Text-to-speech playback
- **Config**: `voice_get_config()` / `voice_set_config()` — Inspect/adjust settings

### voice_bridge.py

HTTP API server that wires up the voice pipeline:

```
openWakeWord → Silero VAD → Moonshine STT → LLM → TTS
```

Endpoints:
- `GET /v1/status` — Current pipeline state
- `POST /v1/voice/toggle` — Enable/disable voice mode
- `POST /v1/say` — TTS playback
- `POST /v1/utterance/last` — Get last utterance
- `POST /v1/speaker/enroll` — Enroll new speaker
- `GET /v1/speakers` — List enrolled speakers
- `POST /v1/transcript/correct` — Correct ASR transcript
- `GET /v1/config` — Get configuration
- `POST /v1/config` — Update configuration

### voice_tui.py

Textual-based terminal UI with:

- Real-time status indicators (IDLE, LISTENING, PROCESSING, SPEAKING)
- Rolling transcript display with speaker labels
- `v` keybinding to toggle voice mode
- OpenClaw integration: ASR → gateway chat, `<summary>` → TTS

## Configuration

Edit `voice_bridge_config.json` to customize:

```json
{
  "vad": {
    "model": "silero_vad",
    "threshold": 0.5,
    "min_silence_duration_s": 1.0
  },
  "wakeword": {
    "model": "openwakeword",
    "threshold": 0.6,
    "smoothing": true
  },
  "asr": {
    "model": "moonshine_stt",
    "language": "en"
  },
  "tts": {
    "model": "cosyvoice3",
    "voice_id": "default"
  },
  "diarization": {
    "enabled": false,
    "threshold": 0.75
  }
}
```

## OpenClaw Integration

### ASR → Gateway Chat

When voice mode is enabled:
1. Final ASR utterances are forwarded to the OpenClaw gateway chat
2. This appears as user input, allowing the agent to respond
3. Transcript history is maintained (50 entries max)

### TTS Response Hook

When OpenClaw responds:
1. The response is parsed for `<summary>...</summary>` blocks
2. Content inside the summary tags is extracted
3. That text is sent to `voice_say` for TTS playback
4. The spoken text is logged for debugging

## Skills

### lloyd-voice-tui

Provides commands to control the voice pipeline:

```bash
# Start the TUI
skill: lloyd-voice-tui start

# Enable voice mode (no TUI)
skill: lloyd-voice-tui enable

# Disable voice mode
skill: lloyd-voice-tui disable
```

## Testing

### Manual API Testing

```bash
# Check status
curl -s http://127.0.0.1:8092/v1/status | jq

# Toggle voice mode
curl -s -X POST http://127.0.0.1:8092/v1/voice/toggle -H "Content-Type: application/json" -d '{}' | jq

# List speakers
curl -s http://127.0.0.1:8092/v1/speakers | jq

# Get last utterance
curl -s -X POST http://127.0.0.1:8092/v1/utterance/last | jq
```

### Run Test Continuity Script

```bash
cd ~/Projects/lloyd/scripts
python3 test_continuity.py
```

## Dependencies

- **Python 3.10+**
- **textual** — Terminal UI framework
- **httpx** — HTTP client
- **openwakeword** — Wake word detection
- **silero-vad** — Voice activity detection
- **moonshine-stt** — Speech-to-text
- **tts models** — CosyVoice3, Qwen3-TTS, or Orpheus

## Troubleshooting

### MCP server not starting

Check logs:
```bash
python3 voice_mcp_server.py --verbose
```

Ensure all dependencies are installed:
```bash
pip install textual httpx openwakeword silero-vad moonshine-stt
```

### Voice mode not working

1. Check if MCP server is running:
   ```bash
   curl -s http://127.0.0.1:8092/v1/status
   ```

2. Check pipeline state in status response
3. Verify wake word model is loaded
4. Check VAD threshold is appropriate

### TTS not playing

1. Verify TTS model is configured in `voice_bridge_config.json`
2. Check that voice mode is enabled
3. Look for errors in MCP server logs

## QMD Search (Vault Memory)

QMD provides full-text (FTS5) and vector (embedding) search over the Obsidian vault — the agent's long-term memory backbone.

### Services

| Service | Description |
|---|---|
| `lloyd-qmd-daemon` | HTTP MCP server on port 8181 (Streamable HTTP at `/mcp`) |
| `lloyd-qmd-watcher` | inotifywait on vault, auto-reindexes on `.md` changes (~2s debounce) |

Both services pin to **GPU 0** via `CUDA_VISIBLE_DEVICES=0`.

### Setup

```bash
bash setup/setup-qmd.sh
systemctl --user enable --now lloyd-qmd-daemon lloyd-qmd-watcher
```

### Manual Commands

```bash
qmd update          # Re-index FTS (0.4s full scan)
qmd embed           # Generate/update vector embeddings
qmd search "query"  # FTS keyword search
qmd vsearch "query" # Vector similarity search
qmd query "query"   # Hybrid search with expansion + reranking
qmd status          # Index health
```

## Project Structure

```
~/Projects/lloyd-services/
├── voice_mcp_server.py      # FastMCP server
├── voice_bridge.py          # Voice pipeline HTTP API
├── voice_bridge_config.json # Pipeline configuration
├── voice_tui.py             # Textual TUI application
├── tool_services.py         # MCP tool services (qmd search, etc.)
├── scripts/
│   └── qmd-watcher.sh       # Vault filesystem watcher
├── setup/
│   ├── setup-all.sh          # Master setup script
│   ├── setup-qmd.sh          # QMD search setup
│   ├── setup-llm.sh          # LLM server setup
│   ├── setup-orpheus.sh      # Orpheus TTS setup
│   ├── setup-cosyvoice.sh    # CosyVoice TTS setup
│   └── install-services.sh   # Systemd service installer
├── systemd/
│   ├── lloyd-qmd-daemon.service   # QMD HTTP MCP (port 8181, GPU 0)
│   ├── lloyd-qmd-watcher.service  # Vault auto-reindex watcher
│   └── ...                        # Other service files
├── README.md                # This file
└── .git/                    # Git repository
```

## License

Proprietary — Alan's Robot Lab