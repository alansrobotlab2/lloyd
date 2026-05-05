# Voice Mode Integration

Bidirectional bridge between the voice pipeline TUI (`~/Projects/lloyd-services/voice_mode.py`) and OpenClaw.

## Architecture

```
                        ┌─────────────────────┐
                        │   OpenClaw Gateway   │
                        │   localhost:18789    │
                        └──────┬────┬─────────┘
            POST /hooks/agent  │    │  message_sending hook
            (ASR transcript)   │    │  (extract <summary>)
                               │    │
                        ┌──────┴────▼─────────┐
                        │    voice-tools       │
                        │    plugin (TS)       │
                        └──────┬────┬─────────┘
                               │    │
                               │    │ POST /v1/say
                               │    │ (summary text)
                        ┌──────▼────▼─────────┐
  Mic → Wake Word →     │   voice_mode.py      │ → Speaker
  VAD → STT → Speaker → │   HTTP :8092        │
  ID → transcript        │   TUI + Pipeline    │
                        └─────────────────────┘
```

## Dataflow

### ASR → OpenClaw (Phase 1)

1. Wake word detected → VAD → STT produces transcript
2. `on_transcript(text, speaker)` callback fires in voice_mode.py
3. TUI POSTs to `http://127.0.0.1:18789/hooks/agent`:
   ```json
   {
     "message": "[Alan]: hello world",
     "sessionKey": "agent:main:main"
   }
   ```
4. OpenClaw processes as user input in the main session
5. Agent responds normally through its channels

### OpenClaw → TTS (Phase 2)

1. Agent response contains `<summary>spoken text here</summary>`
2. voice-tools plugin `message_sending` hook extracts summary content
3. Plugin POSTs to `http://127.0.0.1:8092/v1/say`:
   ```json
   { "text": "spoken text here" }
   ```
4. voice_mode.py receives, streams through CosyVoice3 TTS at port 8090
5. Audio plays through the configured output device

### Summary Filtering (Phase 3)

1. Same `message_sending` hook strips `<summary>...</summary>` tags
2. Returns `{ content: strippedText }` so tags don't appear on screen
3. The spoken text only reaches the user via audio, not cluttering the chat

## Skills

The `lloyd-voice-tui` skill (`~/obsidian/lloyd/skills/lloyd-voice-tui/`) provides:

| Command | Description |
|---------|-------------|
| `skill: lloyd-voice-tui start` | Launch the voice TUI (starts pipeline + HTTP server on :8092) |
| `skill: lloyd-voice-tui enable` | Enable voice mode via HTTP toggle |
| `skill: lloyd-voice-tui disable` | Disable voice mode via HTTP toggle |

The TUI must be running for enable/disable to work (it serves :8092).

## Files Modified

| File | What changed |
|------|-------------|
| `~/Projects/lloyd-services/voice_mode.py` | OpenClaw injection + HTTP server (/v1/say, /v1/status, /v1/voice/toggle) |
| `~/.openclaw/extensions/voice-tools/index.ts` | `message_sending` hook: summary → TTS + strip from display |
| `~/.openclaw/openclaw.json` | `hooks.allowRequestSessionKey: true` |
| `~/obsidian/lloyd/skills/lloyd-voice-tui/voice-tui.sh` | Fixed start command (--config not --mcp-url) |
| `~/obsidian/lloyd/skills/lloyd-voice-tui/SKILL.md` | Updated to reflect new architecture |

## HTTP Endpoints (voice_mode.py :8092)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/say` | POST | TTS playback: `{"text": "..."}` |
| `/v1/status` | GET | Pipeline state, voice enabled, last transcript |
| `/v1/voice/toggle` | POST | Toggle voice mode on/off |

## Configuration

`voice_bridge_config.json` keys used:
- `use_openclaw: true` — enable ASR → OpenClaw forwarding
- `openclaw_url` — gateway base URL (stripped to `http://127.0.0.1:18789`)
- `openclaw_token` — Bearer token for hooks endpoint
- `mcp_api_port: 8092` — HTTP server port for TTS/status/toggle

`openclaw.json` hooks section:
- `allowRequestSessionKey: true` — allow voice TUI to target `agent:main:main`

## Verification

1. Start voice_tui: `cd ~/Projects/lloyd-services && python3 voice_mode.py`
2. Confirm HTTP server: `curl -s http://127.0.0.1:8092/v1/status`
3. Test TTS: `curl -X POST http://127.0.0.1:8092/v1/say -H 'Content-Type: application/json' -d '{"text":"hello"}'`
4. Test ASR injection: speak wake word + phrase, check OpenClaw session logs
5. Test summary routing: trigger response with `<summary>` tags, confirm TTS plays and tags stripped from display
