---
segment: architecture
relations:
  related-to: []
tags: [architecture]
summary: 'Voice pipeline architecture: wake word detection,VAD,Whisper STT,speaker
  ID,Qwen3-TTS,and dual local-mic/WebSocket input modes.'
type: reference

---

















# Voice Pipeline

Self-contained voice processing pipeline for wake word detection,speech recognition,text-to-speech,and speaker identification. All source code lives at `~/Projects/lloyd-services/`.

Two-file architecture: `voice_pipeline.py` handles audio processing,`voice_mode.py` handles orchestration and the HTTP API.

Supports two input modes: **local microphone** (default) and **browser WebSocket** (Mission Control).

## Architecture -- Local Mic Mode (`input_mode: local`)

```mermaid
graph TB
    Mic["Local Microphone<br/>(sounddevice)"] --> VP["voice_pipeline.py<br/>(Wake Word + VAD + STT + Speaker ID)"]
    VP --> Transcript["Transcript"]

    VP <-->|"HTTP :8092"| VM["voice_mode.py<br/>(Headless / TUI)"]

    VMCP["voice_services.py<br/>MCP :8094 SSE"] -->|"HTTP :8092"| VM
    VPlugin["voice-tools plugin"] -->|SSE| VMCP
    GW["OpenClaw Gateway"] --> VPlugin

    GW -->|"message_sending hook"| VPlugin
    VPlugin -->|"POST /v1/say"| VM
    VM --> TTS["Qwen3-TTS :8090<br/>(GPU1)"]
    TTS --> Speakers["Local Speakers<br/>(sd.OutputStream)"]
```

## Architecture -- Browser WebSocket Mode (`input_mode: websocket`)

```mermaid
graph TB
    BrowserMic["Browser Microphone<br/>(getUserMedia)"] -->|"PCM Int16 16kHz mono"| WS["WebSocket :8095<br/>(WebSocketAudioServer)"]
    WS --> VP["voice_pipeline.py<br/>(VAD + STT + Speaker ID)"]
    VP --> Transcript["Transcript"]

    VP <-->|"HTTP :8092"| VM["voice_mode.py<br/>(Headless)"]

    VMCP["voice_services.py<br/>MCP :8094 SSE"] -->|"HTTP :8092"| VM
    VPlugin["voice-tools plugin"] -->|SSE| VMCP
    GW["OpenClaw Gateway :18790"] --> VPlugin

    GW -->|"message_sending hook"| VPlugin
    VPlugin -->|"POST /v1/say"| VM
    VM --> TTS["Qwen3-TTS :8090<br/>(GPU1)"]
    TTS -->|"Float32 PCM 24kHz"| WS
    WS -->|"Binary audio frames"| Browser["MC Browser<br/>(Web Audio API)"]
    TTS --> Speakers["Local Speakers<br/>(sd.OutputStream)"]

    MC["Mission Control<br/>(HTTPS :18790)"] --> BrowserMic
    MC --> Browser

    subgraph "MC Frontend"
        VoicePanel["VoicePanel.tsx"]
        UseVoice["useVoiceStream.ts"]
        VoicePanel --> UseVoice
        UseVoice --> BrowserMic
        UseVoice --> Browser
    end
```

## Webhook Integration

When a transcript is recognized,`voice_mode.py` forwards it to the OpenClaw gateway via a webhook POST to `/hooks/wake`. This triggers a new conversation turn with the transcribed text.

## Components

### voice_pipeline.py -- Core Processing

The main audio processing pipeline.

- **Wake word detection:** openWakeWord (custom-trained "Hey Lloyd" / "Lloyd")
- **VAD:** Silero VAD (512-sample frames at 16kHz)
- **STT/ASR:** Whisper via onnxruntime-gpu
- **Speaker identification:** Resemblyzer + enrolled voice profiles
- **Pipeline states:** IDLE -> LISTENING -> PROCESSING -> IDLE (+ SPEAKING for TTS)
- **Wakeword ring buffer:** 3 seconds of audio
- **Silence detection:** 1000ms threshold
- **Minimum utterance:** 0.3 seconds
- **GPU:** ONNX Runtime with CUDA support (pre-loads NVIDIA CUDA 12 libs)

#### Audio Reader Abstraction

Pipeline functions accept a `reader_factory` parameter for swappable audio sources:
- `_AudioReader` -- wraps `_SafeMicStream` (sounddevice callback API,avoids Pa_ReadStream heap corruption)
- `_WebSocketAudioReader` -- pulls frames from `WebSocketAudioServer.frame_queue`

Both implement: `read(timeout) -> np.ndarray | None`,context manager protocol.

#### WebSocketAudioServer

- Runs in a background thread with its own asyncio event loop
- `send_message(dict)` -- thread-safe JSON push to browser
- `send_binary(bytes)` -- thread-safe binary push (TTS audio)
- `make_reader()` -- creates a `_WebSocketAudioReader` (drains stale frames first)
- `has_client` -- whether a browser is connected

### voice_mode.py -- Orchestration / HTTP API

Headless mode for supervisord operation (also supports Textual TUI). Exposes HTTP API on port 8092.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/status` | GET | Pipeline state,last transcript,speakers |
| `/v1/say` | POST | TTS playback (local speakers + browser if connected) |
| `/v1/voice/toggle` | POST | Enable/disable voice |
| `/v1/voice/ws-status` | GET | WebSocket input mode status and port |

Forwards transcripts to the gateway via webhook POST to `/hooks/wake`.

### voice_services.py -- Voice MCP Server

- **Framework:** FastMCP on port 8094 (SSE transport)
- **Supervisord service:** `lloyd-voice-mcp` (inside distrobox `lloyd-services`)
- **Command:** `uv run voice_services.py --transport sse --port 8094`
- **Role:** Proxies tool calls to the voice HTTP API (port 8092)
- **Tools:** `voice_status`,`voice_last_utterance`,`voice_enroll_speaker`,`voice_list_speakers`
- **