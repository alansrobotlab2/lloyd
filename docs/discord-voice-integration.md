# Discord Voice Integration via Existing Voice Pipeline

## Context

OpenClaw's built-in Discord voice handler has no plugin hooks for audio interception — STT, TTS, and audio routing are all internal. Rather than fight the built-in system, we disable OpenClaw's Discord channel entirely and create a standalone Discord voice bot (`discord.py`) that routes audio through the existing voice pipeline. This reuses the proven wake word → VAD → STT → speaker ID → TTS infrastructure already working for local mic and Mission Control.

**Decision:** Use the same bot token (disable OpenClaw Discord, move token to voice config). Text chat forwarding via the custom bot is a Phase 2 item.

**Decision:** Keep wake word detection — users say "Hey Lloyd" in Discord voice, same as other modes.

## Architecture

```
Discord Voice Channel
  ↓ Opus 48kHz stereo          ↑ Opus 48kHz stereo
discord_voice.py (new adapter)
  ↓ int16 PCM 16kHz            ↑ float32 PCM 24kHz
WebSocketAudioServer (:8095)
  ↓                             ↑
PipelineRunner (wake word → VAD → STT → speaker ID)
  ↓
on_transcript → POST /hooks/wake → OpenClaw → response
  ↓
<summary> tags → _play_tts_text() → CosyVoice3 → WS audio back
```

The Discord bot connects as a WebSocket client to port 8095 — the same protocol the browser uses via Mission Control. No changes to PipelineRunner needed.

## Implementation

### 1. Disable OpenClaw Discord channel

**File:** `openclaw.json`
- Set `channels.discord.enabled: false` (entire Discord section, not just voice)
- Frees the bot token for discord.py and avoids the `baseUrl` config validation error

### 2. Create `discord_voice.py`

**File:** `~/Projects/lloyd-services/discord_voice.py`

Single new file with three components:

**`DiscordAudioSink`** — receives per-user Opus-decoded PCM from Discord:
- Discord provides 48kHz stereo int16 PCM (20ms frames)
- Convert stereo → mono (average channels)
- Resample 48kHz → 16kHz via `scipy.signal.decimate(audio, 3)` (exact 3:1 ratio)
- Send int16 PCM bytes over WebSocket to pipeline
- Mix multiple users' audio into single stream (speaker ID handles identification)

**`DiscordAudioSource(discord.AudioSource)`** — plays TTS audio in Discord:
- Receives float32 PCM at 24kHz from pipeline WebSocket
- Resample 24kHz → 48kHz via `scipy.signal.resample_poly(audio, 2, 1)`
- Convert float32 → int16, mono → stereo
- Buffer into 20ms frames (960 samples at 48kHz)
- `read()` returns frames to discord.py's AudioPlayer

**`DiscordVoiceBot(discord.Client)`** — main bot class:
- On ready: join configured voice channel, connect WebSocket to :8095, send `{"type": "start"}`
- WebSocket receive loop: `tts_start` → prepare audio source, binary data → feed to `DiscordAudioSource`, `tts_end` → stop playback, `transcript` → optionally post to text channel
- Audio receive: `VoiceClient.listen()` with `DiscordAudioSink`
- Graceful shutdown on SIGTERM

**CLI:** `--config voice_bridge_config.json` (reuses existing config)

### 3. Add Discord config to `voice_bridge_config.json`

```json
"discord_voice": {
    "enabled": true,
    "token": "<bot token>",
    "guild_id": "1478527693489438830",
    "auto_join_channel": null,
    "post_transcripts_channel_id": null
}
```

### 4. Add dependencies to `pyproject.toml`

```toml
[project.optional-dependencies]
discord = ["discord.py[voice]>=2.4.0", "websockets>=13.0"]
```

### 5. Create systemd service

`systemd/lloyd-discord-voice.service` — runs `discord_voice.py --config voice_bridge_config.json` inside the lloyd distrobox.

## Reuse Summary

| Component | Source | Reuse |
|-----------|--------|-------|
| Wake word, VAD, STT, speaker ID | `voice_pipeline.py` PipelineRunner | Full — no changes |
| WebSocket protocol | `voice_pipeline.py` WebSocketAudioServer | Full — Discord bot is just another WS client |
| TTS synthesis | `_play_tts_text()` in voice_mode.py | Full — TTS audio returns via WS |
| OpenClaw injection | `/hooks/wake` endpoint | Full — same POST as local/MC modes |
| Config loading | `voice_pipeline.py` `load_config()` | Full — extend same config file |
| Resampling | `scipy.signal` | Already a dependency |

## Constraints & Future Work

- **Single WS client:** WebSocketAudioServer allows only one client. Discord and Mission Control voice cannot run simultaneously. Phase 2: extend to multiple named clients.
- **No Discord text chat:** Disabling OpenClaw Discord loses text chat. Phase 2: add text message forwarding in the custom bot.
- **discord.py voice_recv:** The `listen()`/sink API varies between discord.py forks. Start with standard `discord.py[voice]`; fall back to `py-cord` if needed.

## Verification

1. `pip install "discord.py[voice]"` in the lloyd venv
2. Disable Discord in openclaw.json, add `discord_voice` section to voice_bridge_config.json
3. Restart gateway — confirm clean startup
4. Start voice pipeline: `python voice_mode.py --config voice_bridge_config.json --headless`
5. Start Discord bot: `python discord_voice.py --config voice_bridge_config.json`
6. Join Discord voice channel, say "Hey Lloyd" — verify wake word → transcript → OpenClaw response → TTS playback in Discord
7. Test speaker ID with a second person
