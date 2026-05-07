#!/usr/bin/env python3
"""
Voice Mode TUI — terminal interface for Lloyd's voice pipeline.

Runs the voice pipeline directly (wake word, VAD, STT, speaker ID) and
forwards transcripts to OpenClaw. Displays pipeline state, transcripts,
and responses in real time.
"""

import argparse
import json
import queue
import time as _time
import threading
from collections import deque
from datetime import datetime
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import requests as http_requests
import sounddevice as sd

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Header, Footer, Static, DataTable, Select, Button, Label

from voice_pipeline import (
    PipelineCallbacks, PipelineRunner, State, load_config, list_audio_devices,
    make_aec_processor, TTS_WRITE_BLOCK_SIZE, VAD, VAD_FRAME_SIZE,
    PIPELINE_SAMPLE_RATE,
)

import numpy as np

TRANSCRIPT_HISTORY_SIZE = 50
DEFAULT_CONFIG = "voice_bridge_config.json"


# ---------------------------------------------------------------------------
# HTTP API handler — serves /v1/say, /v1/status, /v1/voice/toggle
# ---------------------------------------------------------------------------

class _VoiceHTTPHandler(BaseHTTPRequestHandler):
    """Minimal HTTP API so OpenClaw (and MCP tools) can reach the TUI."""

    tui: "VoiceTUI | None" = None  # set before server starts

    def log_message(self, fmt, *args):
        pass  # suppress stderr noise

    def handle_one_request(self):
        # Silence noisy BrokenPipeError tracebacks from polling clients
        # (MC frontend disconnects before we finish writing the response).
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json_response(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected, nothing to do

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    # -- GET /v1/status -------------------------------------------------------

    def do_GET(self):
        if self.path == "/v1/status":
            app = _VoiceHTTPHandler.tui
            state = app.state if app else "UNKNOWN"
            enabled = app.voice_enabled if app else False
            last = ""
            if app and app.transcript_history:
                last = app.transcript_history[-1].get("transcript", "")
            self._json_response(200, {
                "state": state,
                "voice_enabled": enabled,
                "last_transcript": last,
            })
        elif self.path == "/v1/voice/ws-status":
            app = _VoiceHTTPHandler.tui
            if not app or not app._pipeline:
                self._json_response(503, {"error": "pipeline not ready"})
                return
            ws_active = app._pipeline.input_mode == "websocket"
            ws_port = 8093
            if app._pipeline.ws_server:
                ws_port = app._pipeline.ws_server.port
            has_client = bool(
                app._pipeline.ws_server and app._pipeline.ws_server.has_clients
            )
            self._json_response(200, {
                "ws_active": ws_active,
                "ws_port": ws_port,
                "has_client": has_client,
                "voice_enabled": app.voice_enabled,
                "state": app.state,
            })
        elif self.path == "/v1/last_utterance":
            app = _VoiceHTTPHandler.tui
            if not app or not app._pipeline:
                self._json_response(503, {"error": "pipeline not ready"})
                return
            utt = app._pipeline.get_last_utterance()
            if not utt:
                self._json_response(200, {"error": "no utterance recorded"})
            else:
                self._json_response(200, utt)
        elif self.path == "/v1/speakers":
            app = _VoiceHTTPHandler.tui
            if not app or not app._pipeline or not app._pipeline.speaker_id:
                self._json_response(200, {"profiles": [], "profiles_dir": "speakers/"})
                return
            sid = app._pipeline.speaker_id
            profiles = [
                {"name": name, "embedding_dim": len(emb)}
                for name, emb in sid._profiles.items()
            ]
            self._json_response(200, {
                "profiles": profiles,
                "profiles_dir": str(sid.profiles_dir),
            })
        elif self.path == "/v1/speaker_stats":
            app = _VoiceHTTPHandler.tui
            if not app or not app._pipeline:
                self._json_response(503, {"error": "pipeline not ready"})
                return
            self._json_response(200, app._pipeline.get_speaker_stats())
        elif self.path == "/v1/uncertain_speakers":
            app = _VoiceHTTPHandler.tui
            if not app or not app._pipeline:
                self._json_response(503, {"error": "pipeline not ready"})
                return
            self._json_response(200, {
                "uncertain": app._pipeline.get_uncertain_speakers()
            })
        elif self.path == "/v1/config":
            app = _VoiceHTTPHandler.tui
            if not app or not app._pipeline:
                self._json_response(503, {"error": "pipeline not ready"})
                return
            cfg = dict(app._pipeline.config)
            # Redact sensitive fields
            for key in ("openclaw_token", "diarization_hf_token"):
                if key in cfg:
                    cfg[key] = "***"
            self._json_response(200, cfg)
        else:
            self._json_response(404, {"error": "not found"})

    # -- POST /v1/say | /v1/voice/toggle --------------------------------------

    def do_POST(self):
        if self.path == "/v1/say":
            self._handle_say()
        elif self.path == "/v1/voice/toggle":
            self._handle_toggle()
        elif self.path == "/v1/correct_transcript":
            self._handle_correct_transcript()
        elif self.path == "/v1/enroll_speaker":
            self._handle_enroll_speaker()
        elif self.path == "/v1/delete_speaker":
            self._handle_delete_speaker()
        elif self.path == "/v1/config":
            self._handle_set_config()
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_correct_transcript(self) -> None:
        app = _VoiceHTTPHandler.tui
        body = self._read_json()
        corrected = body.get("corrected", "").strip()
        if not corrected:
            self._json_response(400, {"error": "no corrected text"})
            return
        if not app or not app._pipeline:
            self._json_response(503, {"error": "pipeline not ready"})
            return
        result = app._pipeline.correct_transcript(corrected)
        if result is None:
            self._json_response(404, {"error": "no utterance to correct"})
        else:
            self._json_response(200, result)

    def _handle_enroll_speaker(self) -> None:
        app = _VoiceHTTPHandler.tui
        body = self._read_json()
        name = body.get("name", "").strip()
        if not name:
            self._json_response(400, {"error": "no name provided"})
            return
        if not app or not app._pipeline or not app._pipeline.speaker_id:
            self._json_response(503, {"error": "speaker ID not available"})
            return
        pipeline = app._pipeline
        with pipeline._last_utterance_lock:
            audio = pipeline._last_utterance.get("audio_int16")
        if audio is None or len(audio) < PIPELINE_SAMPLE_RATE:
            self._json_response(400, {"error": "no recent utterance (need at least 1s)"})
            return
        pipeline.speaker_id.enroll(name, audio)
        self._json_response(200, {
            "enrolled": name,
            "audio_duration_s": len(audio) / PIPELINE_SAMPLE_RATE,
            "profile_path": str(pipeline.speaker_id.profiles_dir / f"{name}.npy"),
        })

    def _handle_delete_speaker(self) -> None:
        app = _VoiceHTTPHandler.tui
        body = self._read_json()
        name = body.get("name", "").strip()
        if not name:
            self._json_response(400, {"error": "no name provided"})
            return
        if not app or not app._pipeline:
            self._json_response(503, {"error": "pipeline not ready"})
            return
        deleted = app._pipeline.delete_speaker(name)
        if deleted:
            self._json_response(200, {"deleted": name})
        else:
            self._json_response(404, {"error": f"speaker '{name}' not found"})

    def _handle_set_config(self) -> None:
        app = _VoiceHTTPHandler.tui
        body = self._read_json()
        if not app or not app._pipeline:
            self._json_response(503, {"error": "pipeline not ready"})
            return
        pipeline = app._pipeline
        updated = {}
        rejected = {}
        allowed = {
            "wake_word_threshold": (float, 0.0, 1.0),
            "silence_duration_ms": (int, 100, 5000),
            "wake_word_smoothing_window": (int, 1, 10),
            "wake_word_min_hits": (int, 1, 10),
            "speaker_id_threshold": (float, 0.0, 1.0),
            "smart_turn_threshold": (float, 0.0, 1.0),
            "barge_in_vad_threshold": (float, 0.0, 1.0),
            "barge_in_consecutive_chunks": (int, 1, 20),
            "barge_in_grace_period_ms": (int, 0, 2000),
            "barge_in_enabled": (bool, None, None),
            "smart_turn_enabled": (bool, None, None),
            "aec_enabled": (bool, None, None),
            "audio_feedback_enabled": (bool, None, None),
        }
        for key, val in body.items():
            if key not in allowed:
                rejected[key] = "unknown config key"
                continue
            typ, lo, hi = allowed[key]
            if typ == bool:
                pipeline.config[key] = bool(val)
                updated[key] = bool(val)
            else:
                try:
                    v = typ(val)
                    if lo is not None and v < lo:
                        rejected[key] = f"must be >= {lo}"
                        continue
                    if hi is not None and v > hi:
                        rejected[key] = f"must be <= {hi}"
                        continue
                    pipeline.config[key] = v
                    updated[key] = v
                except (ValueError, TypeError):
                    rejected[key] = f"expected {typ.__name__}"
        # Apply specific runtime updates
        if "silence_duration_ms" in updated:
            from voice_pipeline import VAD_FRAME_SIZE as _VFS
            vad_frame_ms = _VFS / PIPELINE_SAMPLE_RATE * 1000
            pipeline.silence_frames_threshold = int(updated["silence_duration_ms"] / vad_frame_ms)
        if "speaker_id_threshold" in updated and pipeline.speaker_id:
            pipeline.speaker_id.threshold = updated["speaker_id_threshold"]
            pipeline.speaker_threshold = updated["speaker_id_threshold"]
        if "wake_word_threshold" in updated and pipeline.wake_word:
            pipeline.wake_word.threshold = updated["wake_word_threshold"]
        if "smart_turn_threshold" in updated and pipeline.smart_turn:
            pipeline.smart_turn.threshold = updated["smart_turn_threshold"]
        if "barge_in_vad_threshold" in updated:
            pipeline._barge_in_vad_threshold = updated["barge_in_vad_threshold"]
        if "barge_in_consecutive_chunks" in updated:
            pipeline._barge_in_consec_chunks = updated["barge_in_consecutive_chunks"]
        if "barge_in_grace_period_ms" in updated:
            pipeline._barge_in_grace_period = updated["barge_in_grace_period_ms"] / 1000.0
        if "barge_in_enabled" in updated:
            pipeline._barge_in_enabled = updated["barge_in_enabled"]
        if "aec_enabled" in updated:
            pipeline._aec_enabled = updated["aec_enabled"]
        self._json_response(200, {"updated": updated, "rejected": rejected})

    def _handle_toggle(self) -> None:
        app = _VoiceHTTPHandler.tui
        if not app or not app._pipeline:
            self._json_response(503, {"error": "pipeline not ready"})
            return
        # Toggle via the main thread (synchronous in headless mode)
        app.call_from_thread(app.action_toggle_voice)
        # Return the new state — app.voice_enabled has already flipped
        self._json_response(200, {"voice_enabled": app.voice_enabled})

    def _handle_say(self) -> None:
        app = _VoiceHTTPHandler.tui
        body = self._read_json()
        text = body.get("text", "").strip()
        if not text:
            self._json_response(400, {"error": "no text provided"})
            return
        if not app or not app._pipeline or not app._pipeline.tts:
            self._json_response(503, {"error": "TTS not available"})
            return

        t0 = _time.time()
        tts = app._pipeline.tts
        audio_q: queue.Queue = queue.Queue()
        stop_ev = threading.Event()

        def _synth():
            tts.synthesize_streaming(text, audio_q, stop_ev)
            audio_q.put(None)  # sentinel

        threading.Thread(target=_synth, daemon=True).start()

        sample_rate = tts.sample_rate
        output_dev = getattr(app._pipeline, "output_device", None)
        if output_dev is None:
            # PortAudio's `sd.default.device` is a (in, out) tuple that on this
            # daemon often resolves to a hw:N,X HDMI sink with a fixed native
            # rate, which makes `device=None` either raise "Input and output
            # device are different" or "Invalid sample rate" for the 24 kHz
            # TTS stream. Pick a PipeWire/PulseAudio virtual sink that
            # resamples internally; fall through to None if nothing accepts
            # the rate (paplay fallback below will handle it).
            for _i, _d in enumerate(sd.query_devices()):
                if _d["max_output_channels"] <= 0:
                    continue
                _lname = _d["name"].lower()
                if not any(tag in _lname for tag in ("pipewire", "pulse", "default")):
                    continue
                try:
                    sd.check_output_settings(
                        device=_i, samplerate=sample_rate,
                        channels=1, dtype="float32",
                    )
                    output_dev = _i
                    break
                except Exception:
                    continue

        def _play():
            ws = getattr(app._pipeline, 'ws_server', None)
            ws_active = ws is not None and ws.has_clients
            pipeline = app._pipeline

            # Notify pipeline that TTS is starting
            if pipeline:
                pipeline.notify_tts_start()

            if ws_active:
                ws.send_message({"type": "tts_start", "sample_rate": sample_rate})

            # Phase 2/3: Set up AEC + barge-in monitoring.
            # No separate InputStream — we read from the pipeline's always-on
            # shared mic queue. The wake-word loop has paused itself for the
            # duration of TTS (gates on _tts_playing), so barge-in is the
            # only consumer of _mic_q while Lloyd is talking.
            aec_process = None
            barge_in_vad = None
            barge_in_mic_q = None
            if pipeline and pipeline._barge_in_enabled and pipeline.input_mode == "local":
                try:
                    aec_process = make_aec_processor(PIPELINE_SAMPLE_RATE)
                    barge_in_vad = VAD()
                    barge_in_mic_q = pipeline._mic_q
                    pipeline._barge_in_mic_pos = 0
                    # Drop any chunks captured before TTS started so AEC's
                    # mic_pos starts aligned with the first TTS reference frame.
                    pipeline._drain_mic_q()
                    print(f"  [barge-in] monitor armed (aec={aec_process is not None}, thr={pipeline._barge_in_vad_threshold}, consec={pipeline._barge_in_consec_chunks})", flush=True)
                except Exception as e:
                    if not getattr(_VoiceHTTPHandler, "_barge_in_warned", False):
                        print(f"  Barge-in setup failed (will retry on next utterance): {e}", flush=True)
                        _VoiceHTTPHandler._barge_in_warned = True
                    aec_process = None
                    barge_in_mic_q = None

            # Barge-in detection runs in its own thread so VAD/AEC inference
            # never blocks the audio output writer. Previous design ran
            # check_barge_in() between each TTS_WRITE_BLOCK_SIZE write, which
            # starved sd.OutputStream's buffer when inference time approached
            # block playback time → audible stutter (#287 regression).
            barge_in_monitor_thread = None
            barge_in_monitor_stop = threading.Event()

            def _barge_in_monitor():
                """Poll mic queue + run AEC/VAD off the playback thread.

                Exits on: explicit stop event, TTS already requested to stop,
                or barge-in detected. Detection signals via request_tts_stop()
                which the playback loop already polls between block writes.
                """
                try:
                    while not barge_in_monitor_stop.is_set():
                        if pipeline.is_tts_stop_requested():
                            return
                        try:
                            if pipeline.check_barge_in(
                                barge_in_mic_q, barge_in_vad, aec_process
                            ):
                                print("  [voice barge-in]", flush=True)
                                pipeline.request_tts_stop()
                                stop_ev.set()
                                return
                        except Exception:
                            pass
                        # 20ms cadence: well under VAD frame interval (~32ms),
                        # so a barge-in is detected within one frame of arrival
                        # without busy-waiting.
                        _time.sleep(0.02)
                except Exception:
                    pass

            if barge_in_mic_q is not None and barge_in_vad is not None:
                barge_in_monitor_thread = threading.Thread(
                    target=_barge_in_monitor, daemon=True
                )
                barge_in_monitor_thread.start()

            # Try to open local audio output (optional — may fail)
            local_stream = None
            paplay_proc = None
            try:
                # latency='high' tells PortAudio to allocate a generous output
                # buffer (~hundreds of ms instead of the default ~10ms). Qwen3-
                # TTS produces audio at ~0.84x realtime in bursts of ~600ms
                # gaps; without this cushion the hardware buffer underruns
                # between bursts and you hear clicks/stutter.
                local_stream = sd.OutputStream(
                    samplerate=sample_rate, channels=1, dtype="float32",
                    device=output_dev, latency='high',
                )
                local_stream.start()
            except Exception as e:
                # PortAudio under supervisord loses the PulseAudio host API
                # (no `pipewire`/`default` virtual sinks visible — only ALSA
                # hw:N,X devices that reject the 24 kHz TTS rate). Log once,
                # then fall through silently to the paplay fallback below.
                if not getattr(_VoiceHTTPHandler, "_local_stream_warned", False):
                    print(f"  Local audio output unavailable (using paplay fallback): {e}", flush=True)
                    _VoiceHTTPHandler._local_stream_warned = True
                local_stream = None
                # Fall back to PulseAudio via paplay (handles resampling automatically)
                try:
                    import subprocess as _subprocess
                    paplay_proc = _subprocess.Popen(
                        [
                            "paplay",
                            f"--rate={sample_rate}",
                            "--channels=1",
                            "--format=float32le",
                            "--raw",
                            "--client-name=lloyd-voice-tts",
                        ],
                        stdin=_subprocess.PIPE,
                        stdout=_subprocess.DEVNULL,
                        stderr=_subprocess.DEVNULL,
                    )
                except Exception as pe:
                    print(f"  paplay fallback unavailable: {pe}", flush=True)
                    paplay_proc = None

            interrupted = False
            # Track audio duration written so we can hold _tts_playing set
            # until the audio actually finishes — paplay accepts pipe writes
            # faster than it plays, so without this, notify_tts_end fires
            # while the user can still hear Lloyd talking.
            total_samples_written = 0
            t_first_write: float | None = None
            try:
                while True:
                    # Check if pipeline requested TTS stop (wake word interrupt)
                    if pipeline and pipeline.is_tts_stop_requested():
                        stop_ev.set()  # signal synthesizer to stop too
                        break
                    try:
                        chunk = audio_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if chunk is None:
                        break

                    chunk_arr = np.asarray(chunk, dtype=np.float32)
                    if t_first_write is None:
                        t_first_write = _time.monotonic()
                    total_samples_written += len(chunk_arr)

                    # Phase 2: Track TTS reference for AEC
                    if pipeline:
                        pipeline.append_tts_reference(chunk_arr, sample_rate)

                    # Phase 3: Write in sub-blocks. Barge-in detection runs in
                    # the dedicated monitor thread above; the inner loop only
                    # polls the cheap is_tts_stop_requested() flag so the
                    # output buffer never goes hungry.
                    if local_stream is not None:
                        try:
                            for i in range(0, len(chunk_arr), TTS_WRITE_BLOCK_SIZE):
                                if pipeline and pipeline.is_tts_stop_requested():
                                    interrupted = True
                                    break
                                block = chunk_arr[i:i + TTS_WRITE_BLOCK_SIZE]
                                local_stream.write(block.reshape(-1, 1))
                        except Exception:
                            pass
                    elif paplay_proc is not None and paplay_proc.stdin is not None:
                        # PulseAudio fallback path. Same rationale as above:
                        # barge-in handled by monitor thread, not inline.
                        try:
                            for i in range(0, len(chunk_arr), TTS_WRITE_BLOCK_SIZE):
                                if pipeline and pipeline.is_tts_stop_requested():
                                    interrupted = True
                                    break
                                block = chunk_arr[i:i + TTS_WRITE_BLOCK_SIZE]
                                paplay_proc.stdin.write(block.tobytes())
                        except Exception:
                            pass
                    else:
                        # No local output — barge-in still handled by monitor
                        # thread, just propagate any stop request so the loop
                        # can exit on the WS-only path.
                        if pipeline and pipeline.is_tts_stop_requested():
                            interrupted = True

                    if ws_active:
                        ws.send_audio(chunk_arr, sample_rate)

                    if interrupted:
                        break
            finally:
                # Stop the barge-in monitor first so it doesn't try to read
                # from the mic queue / VAD while we tear them down.
                barge_in_monitor_stop.set()
                if barge_in_monitor_thread is not None:
                    try:
                        barge_in_monitor_thread.join(timeout=1.0)
                    except Exception:
                        pass
                # Drain any audio captured during TTS so the wake-word loop
                # doesn't process TTS bleed-through when it resumes.
                if pipeline and pipeline.input_mode == "local":
                    pipeline._drain_mic_q()
                if local_stream is not None:
                    try:
                        local_stream.stop()
                        local_stream.close()
                    except Exception:
                        pass
                if paplay_proc is not None:
                    try:
                        if paplay_proc.stdin is not None:
                            paplay_proc.stdin.close()
                        paplay_proc.wait(timeout=30)
                    except Exception:
                        try:
                            paplay_proc.terminate()
                        except Exception:
                            pass
                if ws_active:
                    ws.send_message({"type": "tts_end"})
                # Hold _tts_playing set until the audio is actually expected
                # to finish. paplay (and to a lesser extent sounddevice) buffer
                # writes ahead of playback, so if we clear immediately the
                # navbar flips out of SPEAKING while the user can still hear
                # Lloyd talking. Skip on barge-in — playback was cut short.
                if (not interrupted
                        and t_first_write is not None
                        and total_samples_written > 0
                        and sample_rate > 0):
                    expected_end = t_first_write + (total_samples_written / sample_rate)
                    # Small fudge for paplay's hardware-buffer flush.
                    residual = expected_end - _time.monotonic() + 0.15
                    if residual > 0:
                        _time.sleep(min(residual, 30.0))
                # Notify pipeline that TTS is done
                if pipeline:
                    pipeline.notify_tts_end()

        play_thread = threading.Thread(target=_play, daemon=True)
        play_thread.start()
        play_thread.join(timeout=120)

        elapsed = _time.time() - t0
        self._json_response(200, {
            "text": text,
            "duration_s": round(elapsed, 2),
        })


class SettingsScreen(ModalScreen):
    """Modal settings screen for selecting audio devices."""

    CSS = """
    SettingsScreen {
        align: center middle;
    }

    #settings-dialog {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #settings-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    .device-label {
        margin-top: 1;
        text-style: bold;
    }

    #settings-buttons {
        margin-top: 1;
        height: 3;
        align: center middle;
    }

    #settings-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_settings", "Close", show=True),
    ]

    def __init__(
        self,
        current_input: int | None,
        current_output: int | None,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._current_input = current_input
        self._current_output = current_output
        self._devices = list_audio_devices()

    def compose(self) -> ComposeResult:
        input_options = [
            (dev["name"], dev["index"]) for dev in self._devices["input"]
        ]
        output_options = [
            (dev["name"], dev["index"]) for dev in self._devices["output"]
        ]

        with Vertical(id="settings-dialog"):
            yield Label("Audio Device Settings", id="settings-title")

            yield Label("Input Device (Microphone)", classes="device-label")
            input_value = (
                self._current_input
                if self._current_input is not None and any(
                    d["index"] == self._current_input for d in self._devices["input"]
                )
                else (input_options[0][1] if input_options else Select.BLANK)
            )
            yield Select(
                input_options,
                value=input_value,
                id="input-device",
                allow_blank=not input_options,
            )

            yield Label("Output Device (Speaker)", classes="device-label")
            output_value = (
                self._current_output
                if self._current_output is not None and any(
                    d["index"] == self._current_output for d in self._devices["output"]
                )
                else (output_options[0][1] if output_options else Select.BLANK)
            )
            yield Select(
                output_options,
                value=output_value,
                id="output-device",
                allow_blank=not output_options,
            )

            with Horizontal(id="settings-buttons"):
                yield Button("Apply", variant="primary", id="apply-btn")
                yield Button("Close", variant="default", id="close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-btn":
            input_select = self.query_one("#input-device", Select)
            output_select = self.query_one("#output-device", Select)
            self.dismiss({
                "input": input_select.value,
                "output": output_select.value,
            })
        elif event.button.id == "close-btn":
            self.dismiss(None)

    def action_dismiss_settings(self) -> None:
        self.dismiss(None)


class VoiceTUI(App):
    """Voice pipeline TUI — runs pipeline directly, forwards to OpenClaw."""

    TITLE = "Lloyd Voice Mode"

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        width: 100%;
        height: 100%;
        layout: vertical;
    }

    #status-bar {
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 2;
        text-align: center;
        text-style: bold;
    }

    #transcript-area {
        width: 100%;
        height: 1fr;
        margin: 1 0;
    }

    #transcript-table {
        width: 100%;
        height: 1fr;
    }

    #controls {
        height: 3;
        background: $surface;
        padding: 1 2;
    }

    #controls-label {
        text-align: center;
        color: $text-muted;
        text-style: italic;
    }

    #voice-toggle {
        width: 12;
        height: 2;
        margin: 0 1;
    }

    #voice-toggle.button--enabled {
        background: $success;
        color: $text;
    }

    #voice-toggle.button--disabled {
        background: $error;
        color: $text;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("v", "toggle_voice", "Toggle Voice"),
        Binding("s", "settings", "Settings"),
        Binding("c", "clear_transcript", "Clear"),
    ]

    # --- Custom messages for thread-safe UI updates ---

    class PipelineStateChanged(Message):
        def __init__(self, state: State) -> None:
            super().__init__()
            self.state = state

    class TranscriptReceived(Message):
        def __init__(self, text: str, speaker: str, is_continuity: bool, client_id: str | None = None) -> None:
            super().__init__()
            self.text = text
            self.speaker = speaker
            self.is_continuity = is_continuity
            self.client_id = client_id

    class InitProgressUpdate(Message):
        def __init__(self, component: str) -> None:
            super().__init__()
            self.component = component

    class InitComplete(Message):
        pass

    class ContinuityStatus(Message):
        def __init__(self, msg: str) -> None:
            super().__init__()
            self.msg = msg

    class PipelineError(Message):
        def __init__(self, error: str) -> None:
            super().__init__()
            self.error = error

    def __init__(self, config_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_path = config_path
        self.voice_enabled = False
        self.transcript_history: deque[dict] = deque(maxlen=TRANSCRIPT_HISTORY_SIZE)
        self.state = "INITIALIZING"
        self._pipeline: PipelineRunner | None = None
        self._initialized = False
        # Lloyd transcript injection
        self._lloyd_url: str | None = None
        self._lloyd_session_key: str = "voice-main"
        self._inject_enabled: bool = False
        self._http_server: HTTPServer | None = None

    def compose(self) -> ComposeResult:
        with Container(id="main-container"):
            yield Header()
            yield Static("Initializing...", id="status-bar")

            with ScrollableContainer(id="transcript-area"):
                yield DataTable(id="transcript-table")

            with Horizontal(id="controls"):
                yield Static("Voice Mode: ", id="voice-label")
                yield Static("DISABLED", id="voice-toggle",
                             classes="button--disabled")
                yield Static("", id="controls-label")

            yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#transcript-table", DataTable)
        table.add_column("Time", width=12)
        table.add_column("Speaker", width=15)
        table.add_column("Transcript")

        self._update_voice_toggle()

        # Load config and start pipeline init in background
        try:
            config = load_config(self._config_path)
        except FileNotFoundError:
            self._update_status_bar(f"Config not found: {self._config_path}")
            return

        # Lloyd transcript injection config
        self._inject_enabled = config.get("lloyd_inject_enabled", True)
        self._lloyd_url = config.get(
            "lloyd_inject_url", "http://127.0.0.1:8080/api/voice/inject"
        )
        self._lloyd_session_key = config.get("lloyd_session_key", "voice-main")

        self._pipeline = PipelineRunner(config, self._make_callbacks())
        threading.Thread(
            target=self._pipeline.init_components, daemon=True
        ).start()

        # Start HTTP API server for TTS / status / toggle
        api_port = config.get("mcp_api_port", 8092)
        _VoiceHTTPHandler.tui = self
        self._http_server = ThreadingHTTPServer(("127.0.0.1", api_port), _VoiceHTTPHandler)
        threading.Thread(
            target=self._http_server.serve_forever, daemon=True
        ).start()

    # --- Callbacks from pipeline thread ---

    def _make_callbacks(self) -> PipelineCallbacks:
        app = self

        class Callbacks:
            def on_state_changed(self, state: State) -> None:
                app.call_from_thread(
                    lambda s=state: app.post_message(
                        VoiceTUI.PipelineStateChanged(s)
                    )
                )

            def on_init_progress(self, component: str) -> None:
                app.call_from_thread(
                    lambda c=component: app.post_message(
                        VoiceTUI.InitProgressUpdate(c)
                    )
                )

            def on_init_complete(self) -> None:
                app.call_from_thread(
                    lambda: app.post_message(VoiceTUI.InitComplete())
                )

            def on_transcript(self, text: str, speaker: str,
                              is_continuity: bool, client_id: str | None = None) -> None:
                app.call_from_thread(
                    lambda t=text, s=speaker, ic=is_continuity, cid=client_id:
                        app.post_message(
                            VoiceTUI.TranscriptReceived(t, s, ic, cid)
                        )
                )

            def on_continuity_status(self, msg: str) -> None:
                app.call_from_thread(
                    lambda m=msg: app.post_message(
                        VoiceTUI.ContinuityStatus(m)
                    )
                )

            def on_error(self, error: str) -> None:
                app.call_from_thread(
                    lambda e=error: app.post_message(
                        VoiceTUI.PipelineError(e)
                    )
                )

        return Callbacks()

    # --- Message handlers ---

    def on_voice_tui_pipeline_state_changed(
        self, msg: PipelineStateChanged
    ) -> None:
        self.state = msg.state.name
        self._update_status_bar()

    def on_voice_tui_init_progress_update(
        self, msg: InitProgressUpdate
    ) -> None:
        self._update_status_bar(f"Loading: {msg.component}...")

    def on_voice_tui_init_complete(self, msg: InitComplete) -> None:
        self._initialized = True
        self._update_status_bar("Ready — press 'v' to enable voice mode")

    def on_voice_tui_transcript_received(
        self, msg: TranscriptReceived
    ) -> None:
        prefix = "[+] " if msg.is_continuity else ""
        self._add_transcript_row(
            f"{prefix}{msg.text}", msg.speaker
        )
        # Look up session key from WS server
        session_key = None
        if msg.client_id and self._pipeline and self._pipeline.ws_server:
            session_key = self._pipeline.ws_server.get_client_session_key(msg.client_id)
        self._inject_to_lloyd(msg.text, msg.speaker, session_key=session_key)

    def on_voice_tui_continuity_status(self, msg: ContinuityStatus) -> None:
        label = self.query_one("#controls-label", Static)
        label.update(msg.msg)

    def on_voice_tui_pipeline_error(self, msg: PipelineError) -> None:
        self.notify(f"Pipeline error: {msg.error}", severity="error")

    # --- Actions ---

    # --- Lloyd integration ---

    def _inject_to_lloyd(self, text: str, speaker: str, session_key: str | None = None) -> None:
        """POST transcript to Lloyd /api/voice/inject (fire-and-forget)."""
        if not self._inject_enabled or not self._lloyd_url:
            return

        payload = {
            "text": text,
            "speaker": speaker or "",
            "session_key": session_key or self._lloyd_session_key,
        }

        def _post():
            try:
                http_requests.post(
                    self._lloyd_url, json=payload, timeout=180,
                )
            except Exception:
                pass

        threading.Thread(target=_post, daemon=True).start()

    def action_quit(self) -> None:
        if self._http_server:
            self._http_server.shutdown()
        if self._pipeline:
            self._pipeline.stop()
        self.exit()

    def action_toggle_voice(self) -> None:
        if not self._initialized:
            self.notify("Pipeline still loading...", severity="warning")
            return

        self.voice_enabled = not self.voice_enabled

        if self.voice_enabled:
            self._pipeline.voice_enabled.set()
            self._pipeline.start()
        else:
            self._pipeline.voice_enabled.clear()

        self._update_voice_toggle()
        self.notify(
            f"Voice mode {'enabled' if self.voice_enabled else 'disabled'}",
            severity="information",
        )

    def action_settings(self) -> None:
        current_input = None
        current_output = None
        if self._pipeline:
            current_input = self._pipeline.mic_device
            current_output = self._pipeline.output_device
        self.push_screen(
            SettingsScreen(current_input, current_output),
            callback=self._on_settings_result,
        )

    def _on_settings_result(self, result: dict | None) -> None:
        if result is None:
            return

        devices = list_audio_devices()
        dev_map = {}
        for dev in devices["input"] + devices["output"]:
            dev_map[dev["index"]] = dev

        input_idx = result["input"]
        output_idx = result["output"]

        if self._pipeline and input_idx is not None:
            dev = dev_map.get(input_idx, {})
            rate = dev.get("rate", 16000)
            self._pipeline.set_input_device(input_idx, rate)
            self.notify(f"Input: {dev.get('name', input_idx)}")

        if self._pipeline and output_idx is not None:
            self._pipeline.set_output_device(output_idx)
            dev = dev_map.get(output_idx, {})
            self.notify(f"Output: {dev.get('name', output_idx)}")

        self._save_device_config(input_idx, output_idx, dev_map)

    def _save_device_config(
        self, input_idx: int | None, output_idx: int | None,
        dev_map: dict,
    ) -> None:
        config_path = Path(self._config_path)
        if not config_path.exists():
            return
        try:
            config = json.loads(config_path.read_text())
            if input_idx is not None and input_idx in dev_map:
                config["mic_device_name"] = dev_map[input_idx]["name"]
            if output_idx is not None and output_idx in dev_map:
                config["output_device_name"] = dev_map[output_idx]["name"]
            config_path.write_text(json.dumps(config, indent=4) + "\n")
        except Exception:
            self.notify("Failed to save device config", severity="warning")

    def action_clear_transcript(self) -> None:
        table = self.query_one("#transcript-table", DataTable)
        table.clear()
        self.transcript_history.clear()

    # --- UI helpers ---

    def _update_status_bar(self, text: str | None = None) -> None:
        status_bar = self.query_one("#status-bar", Static)
        if text is not None:
            status_bar.update(text)
            return
        voice_icon = "ON" if self.voice_enabled else "OFF"
        status_bar.update(f"State: {self.state:<15} | Voice: {voice_icon}")

    def _update_voice_toggle(self) -> None:
        toggle = self.query_one("#voice-toggle", Static)
        if self.voice_enabled:
            toggle.update("ENABLED")
            toggle.set_classes({"button--disabled": False,
                                "button--enabled": True})
        else:
            toggle.update("DISABLED")
            toggle.set_classes({"button--disabled": True,
                                "button--enabled": False})

    def _add_transcript_row(self, text: str, speaker: str) -> None:
        table = self.query_one("#transcript-table", DataTable)
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.transcript_history.append({
            "timestamp": timestamp,
            "speaker": speaker,
            "transcript": text,
        })

        table.add_row(timestamp, speaker, text)
        table.scroll_end()


# ---------------------------------------------------------------------------
# Headless mode — runs pipeline + HTTP API without Textual TUI
# ---------------------------------------------------------------------------

class HeadlessVoiceMode:
    """Lightweight replacement for VoiceTUI when running as a systemd service."""

    def __init__(self, config_path: str):
        self._config_path = config_path
        self.voice_enabled = False
        self.transcript_history: deque[dict] = deque(maxlen=TRANSCRIPT_HISTORY_SIZE)
        self.state = "INITIALIZING"
        self._pipeline: PipelineRunner | None = None
        self._initialized = False
        self._lloyd_url: str | None = None
        self._lloyd_session_key: str = "voice-main"
        self._inject_enabled: bool = False
        self._http_server: HTTPServer | None = None
        self._lock = threading.Lock()

    def call_from_thread(self, fn):
        """Direct call — no Textual event loop in headless mode."""
        fn()

    def action_toggle_voice(self) -> None:
        if not self._initialized:
            return
        self.voice_enabled = not self.voice_enabled
        if self.voice_enabled:
            self._pipeline.voice_enabled.set()
            self._pipeline.start()
        else:
            self._pipeline.voice_enabled.clear()

    def run(self):
        import signal

        config = load_config(self._config_path)

        # Lloyd transcript injection config
        self._inject_enabled = config.get("lloyd_inject_enabled", True)
        self._lloyd_url = config.get(
            "lloyd_inject_url", "http://127.0.0.1:8080/api/voice/inject"
        )
        self._lloyd_session_key = config.get("lloyd_session_key", "voice-main")

        # Build pipeline callbacks (headless — just log + update state)
        callbacks = self._make_callbacks()
        self._pipeline = PipelineRunner(config, callbacks)
        threading.Thread(
            target=self._pipeline.init_components, daemon=True
        ).start()

        # Start HTTP API
        api_port = config.get("mcp_api_port", 8092)
        _VoiceHTTPHandler.tui = self
        self._http_server = ThreadingHTTPServer(("127.0.0.1", api_port), _VoiceHTTPHandler)

        print(
            f"Voice Mode (headless) — HTTP API on 127.0.0.1:{api_port}",
            flush=True,
        )

        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())

        threading.Thread(
            target=self._http_server.serve_forever, daemon=True
        ).start()

        stop.wait()

        self._http_server.shutdown()
        if self._pipeline:
            self._pipeline.stop()

    def _make_callbacks(self):
        host = self

        class Callbacks:
            def on_state_changed(self, state: State) -> None:
                host.state = state.name

            def on_init_progress(self, component: str) -> None:
                print(f"  Loading: {component}", flush=True)

            def on_init_complete(self) -> None:
                host._initialized = True
                host.voice_enabled = True
                if host._pipeline:
                    host._pipeline.voice_enabled.set()
                    host._pipeline.start()
                print("  Pipeline ready — voice enabled", flush=True)

            def on_transcript(self, text: str, speaker: str,
                              is_continuity: bool, client_id: str | None = None) -> None:
                ts = datetime.now().strftime("%H:%M:%S")
                host.transcript_history.append({
                    "timestamp": ts,
                    "speaker": speaker,
                    "transcript": text,
                })
                # Look up session key from WS server
                session_key = None
                if client_id and host._pipeline and host._pipeline.ws_server:
                    session_key = host._pipeline.ws_server.get_client_session_key(client_id)
                host._inject_to_lloyd(text, speaker, session_key=session_key)

            def on_continuity_status(self, msg: str) -> None:
                pass

            def on_error(self, error: str) -> None:
                print(f"  Pipeline error: {error}", flush=True)

        return Callbacks()

    def _inject_to_lloyd(self, text: str, speaker: str, session_key: str | None = None) -> None:
        """POST transcript to Lloyd /api/voice/inject (fire-and-forget)."""
        if not self._inject_enabled or not self._lloyd_url:
            return
        payload = {
            "text": text,
            "speaker": speaker or "",
            "session_key": session_key or self._lloyd_session_key,
        }

        def _post():
            try:
                http_requests.post(
                    self._lloyd_url, json=payload, timeout=180,
                )
            except Exception:
                pass

        threading.Thread(target=_post, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(
        description="Lloyd Voice Mode TUI"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to voice config JSON (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without TUI (systemd service mode)",
    )
    args = parser.parse_args()

    if args.headless:
        app = HeadlessVoiceMode(config_path=args.config)
    else:
        app = VoiceTUI(config_path=args.config)
    app.run()


if __name__ == "__main__":
    main()
