#!/usr/bin/env python3
"""
Voice pipeline — self-contained voice processing components and pipeline runner.

Extracted from lloyd/voice_bridge.py and lloyd/scripts/test_continuity.py.
Runs wake word detection, VAD, STT, speaker identification, and conversational
continuity in a background thread. Communicates with a TUI (or other frontend)
via callbacks.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# --- Prevent heap corruption from concurrent native thread pools ---
# Must be set before importing numpy, scipy, torch, or onnxruntime.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Pre-load pip-installed NVIDIA CUDA 12 libs into the process so
# onnxruntime-gpu can find libcublas, libcufft, libcudart, libcurand, libcudnn, etc.
def _preload_cuda_libs():
    import ctypes
    import glob as _glob
    site_pkgs = _os.path.join(
        _os.path.dirname(__import__("onnxruntime").__file__), _os.pardir,
    )
    nvidia_dir = _os.path.join(site_pkgs, "nvidia")
    if not _os.path.isdir(nvidia_dir):
        return
    for so in sorted(_glob.glob(_os.path.join(nvidia_dir, "*/lib/lib*.so.*"))):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
_preload_cuda_libs()

import importlib.util
import io
import json
import queue
import re
import uuid

import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import asyncio
from typing import Any, Callable, Protocol

import numpy as np
import requests
import sounddevice as sd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_SAMPLE_RATE = 16000
VAD_FRAME_SIZE = 512       # Silero VAD frame size at 16kHz
WW_FRAME_SIZE = 640        # openWakeWord frame size at 16kHz
SILENCE_DURATION_MS = 1000
MIN_UTTERANCE_SAMPLES = 4800  # minimum 0.3s of audio
MAX_UTTERANCE_SAMPLES = 30 * PIPELINE_SAMPLE_RATE  # hard cut at 30s so a stuck Smart Turn can't trap the loop forever

# --- Phase 1: Smart Turn v3 constants ---
SMART_TURN_MAX_SAMPLES = 8 * PIPELINE_SAMPLE_RATE  # 8 seconds max input
SMART_TURN_DEFAULT_THRESHOLD = 0.5  # prob >= threshold = end of turn

# --- Phase 2: AEC constants ---
AEC_FRAME_SIZE = 160       # 10ms frames at 16kHz (speexdsp native)
AEC_FILTER_LENGTH = 4800   # 300ms echo tail (16kHz * 0.3s)

# --- Phase 3: Barge-in constants ---
BARGE_IN_VAD_THRESHOLD = 0.8      # stricter than normal 0.5
BARGE_IN_CONSEC_CHUNKS = 5        # 5 × 32ms = 160ms sustained speech
BARGE_IN_GRACE_PERIOD_S = 0.5     # ignore first 500ms of TTS
TTS_WRITE_BLOCK_SIZE = 4096       # samples per write (barge-in check between)

# --- Phase 6: Audio feedback constants ---
CHIME_SAMPLE_RATE = 24000
CHIME_DURATION_S = 30.0
CHIME_TICK_EVERY_S = 1.5

WAKEWORD_RING_BUFFER_SECONDS = 3
WAKEWORD_RING_BUFFER_FRAMES = int(
    WAKEWORD_RING_BUFFER_SECONDS * PIPELINE_SAMPLE_RATE / WW_FRAME_SIZE
)

DEFAULT_LISTEN_WINDOW_S = 10


# ---------------------------------------------------------------------------
# Enums & helpers
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE = auto()        # Waiting for wake word
    LISTENING = auto()   # Recording speech (VAD active)
    PROCESSING = auto()  # STT / speaker ID
    SPEAKING = auto()    # Playing TTS audio


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def list_audio_devices() -> dict[str, list[dict]]:
    """Return available audio input and output devices."""
    devices = sd.query_devices()
    inputs: list[dict] = []
    outputs: list[dict] = []
    for idx, dev in enumerate(devices):
        info = {
            "index": idx,
            "name": dev["name"],
            "rate": int(dev["default_samplerate"]),
        }
        if dev["max_input_channels"] > 0:
            inputs.append(info)
        if dev["max_output_channels"] > 0:
            outputs.append(info)
    return {"input": inputs, "output": outputs}


def resolve_mic_device(config: dict, retries: int = 5, retry_delay: float = 2.0) -> tuple[int, int]:
    """Resolve mic device index and native sample rate from config.

    Retries on failure to tolerate cold-start races where supervisord launches
    voice-mode before the host audio devices are exposed into the container.
    """
    name_pattern = config.get("mic_device_name")
    if not name_pattern:
        device_idx = config["mic_device"]
        native_rate = config.get("mic_native_rate", config["sample_rate"])
        return device_idx, native_rate

    last_available: list[str] = []
    for attempt in range(retries):
        devices = sd.query_devices()
        last_available = [d["name"] for d in devices if d["max_input_channels"] > 0]
        for idx, dev in enumerate(devices):
            if (dev["max_input_channels"] > 0
                    and name_pattern.lower() in dev["name"].lower()):
                native_rate = (config.get("mic_native_rate")
                               or int(dev["default_samplerate"]))
                return idx, native_rate
        if attempt < retries - 1:
            time.sleep(retry_delay)
    raise RuntimeError(
        f"No input device matching '{name_pattern}' after {retries} attempts. "
        f"Available: {last_available}"
    )


def resolve_output_device(config: dict) -> int | None:
    """Resolve output device index from `output_device_name` (substring match).

    Returns None if no name configured, no match, or the matched device does
    not accept the TTS sample rate — caller falls back to the PortAudio
    default device, which on this system is PipeWire and resamples internally.

    Strips a trailing `(hw:N,X)` from the configured name before matching,
    since ALSA card indices renumber across reboots.
    """
    name_pattern = config.get("output_device_name")
    if not name_pattern:
        return None
    cleaned = re.sub(r"\s*\(hw:\d+,\d+\)\s*$", "", name_pattern).strip()
    tts_rate = int(config.get("tts_sample_rate", 24000))
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] <= 0:
            continue
        if cleaned.lower() not in dev["name"].lower():
            continue
        try:
            sd.check_output_settings(
                device=idx, samplerate=tts_rate, channels=1, dtype="float32",
            )
        except Exception:
            # ALSA hw devices reject non-native rates; skip so caller falls
            # back to the resampling default sink.
            return None
        return idx
    return None


def resample_int16(audio_int16: np.ndarray, target_len: int) -> np.ndarray:
    """Resample int16 audio to target length (mic native rate -> pipeline rate).

    Uses numpy linear interpolation instead of scipy FFT resampling to avoid
    heap corruption from scipy's internal memory allocations conflicting with
    PortAudio's audio callback threads.
    """
    old_len = len(audio_int16)
    if old_len == target_len:
        return audio_int16
    old_indices = np.arange(old_len)
    new_indices = np.linspace(0, old_len - 1, target_len)
    resampled = np.interp(new_indices, old_indices, audio_int16.astype(np.float32))
    return resampled.astype(np.int16)


# ---------------------------------------------------------------------------
# Component classes
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """openWakeWord-based wake word detection with score smoothing."""

    def __init__(self, model_names: list[str], threshold: float = 0.5,
                 inference_framework: str = "onnx",
                 smoothing_window: int = 1, min_hits: int = 1):
        from openwakeword.model import Model
        self._ensure_resource_models()

        self.threshold = threshold
        self.smoothing_window = smoothing_window
        self.min_hits = min_hits
        self._score_buffer: deque[float] = deque(maxlen=smoothing_window)
        self.model = Model(
            wakeword_models=model_names,
            inference_framework=inference_framework,
            enable_speex_noise_suppression=True,
        )
        self.model_names = model_names

    @staticmethod
    def _ensure_resource_models():
        """Copy bundled openwakeword ONNX models if missing from package."""
        import openwakeword
        pkg_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        local_dir = Path(__file__).parent / "models" / "openwakeword"
        for name in ("melspectrogram.onnx", "embedding_model.onnx"):
            dest = pkg_dir / name
            src = local_dir / name
            if not dest.exists() and src.exists():
                import shutil
                shutil.copy2(src, dest)

    def process(self, audio_chunk: np.ndarray) -> bool:
        """Process audio chunk. Returns True if wake word detected."""
        predictions = self.model.predict(audio_chunk)
        score = max(predictions.values()) if predictions else 0.0
        self._score_buffer.append(score)
        hits = sum(1 for s in self._score_buffer if s > self.threshold)
        return hits >= self.min_hits

    def get_last_score(self) -> float:
        return self._score_buffer[-1] if self._score_buffer else 0.0

    def reset(self):
        self.model.reset()
        self._score_buffer.clear()


class VAD:
    """Silero VAD for speech boundary detection (pure ONNX, no torch)."""

    _CONTEXT_SIZE = 64
    _STATE_SHAPE = (2, 1, 128)

    def __init__(self):
        import onnxruntime as ort

        spec = importlib.util.find_spec("silero_vad")
        model_path = str(Path(spec.origin).parent / "data" / "silero_vad.onnx")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.sample_rate = 16000
        self.speech_threshold = 0.5
        self.reset()

    def speech_probability(self, audio_chunk: np.ndarray) -> float:
        """Process float32 chunk (512 samples). Returns speech probability 0-1."""
        if audio_chunk.ndim != 1 or len(audio_chunk) != VAD_FRAME_SIZE:
            self.reset()
            return 0.0
        x = audio_chunk.reshape(1, -1).astype(np.float32)
        x_in = np.concatenate([self._context, x], axis=1)
        out, new_state = self._session.run(
            None,
            {
                "input": x_in,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        self._state = new_state
        self._context = x_in[:, -self._CONTEXT_SIZE:]
        return float(out[0, 0])

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Process float32 chunk (512 samples). Returns True if speech."""
        return self.speech_probability(audio_chunk) > self.speech_threshold

    def reset(self):
        self._state = np.zeros(self._STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, self._CONTEXT_SIZE), dtype=np.float32)


class SpeechToText:
    """Moonshine ONNX-based speech-to-text."""

    def __init__(self, model_name: str = "moonshine/base", gpu_device: int = 0):
        import onnxruntime as ort
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        from moonshine_onnx.model import _get_onnx_weights

        self.tokenizer = load_tokenizer()

        # Build constrained session options *before* MoonshineOnnxModel creates
        # its default sessions — unconstrained sessions spawn ~128 threads and
        # cause glibc heap corruption.
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3  # suppress memcpy warnings
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            use_providers = [
                ("CUDAExecutionProvider", {"device_id": gpu_device}),
                "CPUExecutionProvider",
            ]
        else:
            use_providers = ["CPUExecutionProvider"]

        # Monkeypatch InferenceSession so MoonshineOnnxModel's __init__
        # creates single-threaded GPU-enabled sessions from the start.
        _OrigSession = ort.InferenceSession
        def _constrained_session(path, *args, **kwargs):
            kwargs["sess_options"] = opts
            kwargs["providers"] = use_providers
            return _OrigSession(path, *args, **kwargs)
        ort.InferenceSession = _constrained_session
        try:
            self.model = MoonshineOnnxModel(model_name=model_name)
        finally:
            ort.InferenceSession = _OrigSession

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio (int16 PCM at 16kHz) to text."""
        audio_float = audio.astype(np.float32) / 32768.0
        audio_batch = audio_float[np.newaxis, :]
        tokens = self.model.generate(audio_batch, max_len=192)
        text = self.tokenizer.decode_batch(tokens)[0]
        return text.strip()


class WhisperSTT:
    """Faster-whisper-based speech-to-text."""

    def __init__(self, model_size: str = "small", gpu_device: int = 0,
                 hotwords_file: str | None = None):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            model_size, device="cuda", device_index=gpu_device,
            compute_type="float16",
        )
        self.hotwords: str | None = None
        if hotwords_file:
            try:
                path = Path(hotwords_file).expanduser()
                lines = path.read_text().splitlines()
                if lines and lines[0].strip() == "---":
                    end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
                    if end is not None:
                        lines = lines[end + 1:]
                words = [w.strip() for w in lines if w.strip()]
                if words:
                    self.hotwords = ", ".join(words)
                    logger.info("Loaded %d ASR hotwords from %s", len(words), path)
                else:
                    logger.warning("Hotwords file is empty: %s", path)
            except FileNotFoundError:
                logger.warning("Hotwords file not found: %s", hotwords_file)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio (int16 PCM at 16kHz) to text."""
        audio_float = audio.astype(np.float32) / 32768.0
        kwargs = dict(beam_size=5, language="en", vad_filter=True)
        if self.hotwords is not None:
            kwargs["hotwords"] = self.hotwords
        segments, _ = self.model.transcribe(audio_float, **kwargs)
        return " ".join(seg.text for seg in segments).strip()


class SpeakerIdentifier:
    """Match speakers against enrolled voice profiles using Resemblyzer."""

    def __init__(self, profiles_dir: str, threshold: float = 0.75,
                 unknown_label: str = "Unknown", gpu_device: int = 0):
        import torch
        torch.set_num_interop_threads(1)
        from resemblyzer import VoiceEncoder
        device = f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu"
        self._encoder = VoiceEncoder(device=device)
        self.profiles_dir = Path(profiles_dir)
        self.threshold = threshold
        self.unknown_label = unknown_label
        self._profiles: dict[str, np.ndarray] = {}
        self._load_profiles()

    def _load_profiles(self) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.profiles_dir.glob("*.npy")):
            self._profiles[path.stem] = np.load(path)

    def extract_embedding(self, audio_int16: np.ndarray) -> np.ndarray:
        from resemblyzer import preprocess_wav
        wav = audio_int16.astype(np.float32) / 32768.0
        wav = preprocess_wav(wav, source_sr=16000)
        return self._encoder.embed_utterance(wav)

    def identify(self, audio_int16: np.ndarray) -> tuple[str, float]:
        if not self._profiles:
            return self.unknown_label, 0.0
        embedding = self.extract_embedding(audio_int16)
        best_name = self.unknown_label
        best_score = -1.0
        for name, profile in self._profiles.items():
            score = float(np.dot(embedding, profile))
            if score > best_score:
                best_score = score
                best_name = name
        if best_score < self.threshold:
            return self.unknown_label, best_score
        return best_name, best_score

    def enroll(self, name: str, audio_int16: np.ndarray) -> None:
        embedding = self.extract_embedding(audio_int16)
        out_path = self.profiles_dir / f"{name}.npy"
        np.save(out_path, embedding)
        self._profiles[name] = embedding

    def delete_profile(self, name: str) -> bool:
        """Delete a speaker profile. Returns True if deleted."""
        path = self.profiles_dir / f"{name}.npy"
        if path.exists():
            path.unlink()
            self._profiles.pop(name, None)
            return True
        return False

    def get_stats(self) -> dict[str, dict]:
        """Return speaker profiles and their embedding info."""
        stats = {}
        for name, emb in self._profiles.items():
            stats[name] = {
                "embedding_dim": len(emb),
                "profile_path": str(self.profiles_dir / f"{name}.npy"),
            }
        return stats


# ---------------------------------------------------------------------------
# Phase 1: Smart Turn v3 — end-of-turn detection
# ---------------------------------------------------------------------------

class SmartTurnDetector:
    """pipecat-ai/smart-turn-v3 ONNX model for end-of-turn probability estimation.

    Uses Whisper's mel-spectrogram feature extraction on the last 8 seconds
    of audio. Returns a probability (0-1) that the speaker's turn is complete.
    """

    def __init__(self, model_path: str, threshold: float = SMART_TURN_DEFAULT_THRESHOLD,
                 gpu_device: int = 0):
        import onnxruntime as ort
        import torch
        from transformers import WhisperFeatureExtractor

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Smart Turn model not found: {model_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        # Prefer CUDA for the tiny end-of-turn ONNX model; CPU fallback if unavailable.
        providers = []
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": gpu_device}))
        providers.append("CPUExecutionProvider")
        self._session = ort.InferenceSession(
            model_path, providers=providers, sess_options=opts
        )
        self._extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        # Pin mel-spectrogram feature extraction to CUDA. Running it on CPU triggers
        # libtorch_cpu's BLAS dispatch, which gets hijacked by libnvblas.so when
        # CUDA libs are on LD path and there is no CPU BLAS fallback → NULL pointer.
        self._fe_device = f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold

    def predict(self, audio_float32: np.ndarray) -> float:
        """Return end-of-turn probability for the given audio.

        Args:
            audio_float32: Audio samples as float32, 16kHz mono.

        Returns:
            Float 0-1. >= threshold means turn is complete.
        """
        # Use last 8 seconds max
        audio = audio_float32[-SMART_TURN_MAX_SAMPLES:]
        features = self._extractor(
            audio, sampling_rate=PIPELINE_SAMPLE_RATE,
            max_length=SMART_TURN_MAX_SAMPLES,
            padding="max_length", return_attention_mask=False,
            return_tensors="np",
            device=self._fe_device,
        )
        prob = float(
            self._session.run(
                None,
                {"input_features": features.input_features.astype(np.float32)},
            )[0].flatten()[0]
        )
        return prob

    def is_turn_complete(self, audio_float32: np.ndarray) -> bool:
        """Convenience: True if predicted end-of-turn probability >= threshold."""
        return self.predict(audio_float32) >= self.threshold


# ---------------------------------------------------------------------------
# Phase 2: Speex Echo Cancellation via ctypes
# ---------------------------------------------------------------------------

class SpeexEchoCanceller:
    """Echo cancellation using libspeexdsp via ctypes.

    Wraps speex_echo_state_init / speex_echo_cancellation to subtract
    a known reference signal (TTS playback) from the mic signal.
    """

    def __init__(self, frame_size: int = AEC_FRAME_SIZE,
                 filter_length: int = AEC_FILTER_LENGTH,
                 sample_rate: int = PIPELINE_SAMPLE_RATE):
        import ctypes
        import glob as _glob

        # Find the bundled libspeexdsp from speexdsp_ns package
        so_candidates = _glob.glob(
            str(Path(__file__).parent / ".venvs" / "voice-mode" / "lib" / "python*"
                / "site-packages" / "speexdsp_ns.libs" / "libspeexdsp*.so*")
        )
        if not so_candidates:
            # Fallback: search in the same venv as the running python
            import site
            for sp in site.getsitepackages():
                so_candidates.extend(
                    _glob.glob(str(Path(sp) / "speexdsp_ns.libs" / "libspeexdsp*.so*"))
                )
        if not so_candidates:
            raise RuntimeError("libspeexdsp not found. Install speexdsp-ns package.")

        self._lib = ctypes.CDLL(so_candidates[0])
        self._ctypes = ctypes

        self.frame_size = frame_size
        self.filter_length = filter_length

        # speex_echo_state_init(frame_size, filter_length)
        self._lib.speex_echo_state_init.restype = ctypes.c_void_p
        self._lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]

        # speex_echo_cancellation(st, rec, play, out)
        self._lib.speex_echo_cancellation.restype = None
        self._lib.speex_echo_cancellation.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
        ]

        # speex_echo_state_destroy(st)
        self._lib.speex_echo_state_destroy.restype = None
        self._lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]

        # speex_echo_ctl(st, request, ptr)
        self._lib.speex_echo_ctl.restype = ctypes.c_int
        self._lib.speex_echo_ctl.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
        ]

        # Create echo canceller state
        self._state = self._lib.speex_echo_state_init(frame_size, filter_length)

        # Set sample rate (SPEEX_ECHO_SET_SAMPLING_RATE = 24)
        sr = ctypes.c_int(sample_rate)
        self._lib.speex_echo_ctl(self._state, 24, ctypes.byref(sr))

    def process(self, mic_int16: np.ndarray, ref_int16: np.ndarray) -> np.ndarray:
        """Cancel echo from mic signal using reference signal.

        Both inputs must be int16, length = frame_size.
        Returns cleaned int16 audio.
        """
        assert len(mic_int16) == self.frame_size, f"mic must be {self.frame_size} samples"
        assert len(ref_int16) == self.frame_size, f"ref must be {self.frame_size} samples"

        mic_buf = mic_int16.astype(np.int16)
        ref_buf = ref_int16.astype(np.int16)
        out_buf = np.zeros(self.frame_size, dtype=np.int16)

        self._lib.speex_echo_cancellation(
            self._state,
            mic_buf.ctypes.data_as(self._ctypes.POINTER(self._ctypes.c_int16)),
            ref_buf.ctypes.data_as(self._ctypes.POINTER(self._ctypes.c_int16)),
            out_buf.ctypes.data_as(self._ctypes.POINTER(self._ctypes.c_int16)),
        )
        return out_buf

    def reset(self):
        """Reset echo canceller state (call between TTS sessions)."""
        # SPEEX_ECHO_RESET = 1
        self._lib.speex_echo_ctl(self._state, 1, None)

    def __del__(self):
        if hasattr(self, "_state") and self._state:
            self._lib.speex_echo_state_destroy(self._state)
            self._state = None


def make_aec_processor(sample_rate: int = PIPELINE_SAMPLE_RATE):
    """Factory: create a fresh AEC processor for one TTS playback session.

    Returns a function: process(mic_float32, ref_float32) -> cleaned_float32
    Both inputs are float32 arrays of arbitrary length at 16kHz.
    """
    try:
        aec = SpeexEchoCanceller(
            frame_size=AEC_FRAME_SIZE,
            filter_length=AEC_FILTER_LENGTH,
            sample_rate=sample_rate,
        )
    except Exception as e:
        logger.warning("AEC init failed: %s — barge-in will use raw mic", e)
        return None

    def process(mic_float32: np.ndarray, ref_float32: np.ndarray) -> np.ndarray:
        """Process mic and reference signals through AEC. Returns cleaned float32."""
        # Convert to int16 for speexdsp
        mic_i16 = (mic_float32 * 32767).clip(-32768, 32767).astype(np.int16)
        ref_i16 = (ref_float32 * 32767).clip(-32768, 32767).astype(np.int16)

        # Ensure same length
        min_len = min(len(mic_i16), len(ref_i16))
        mic_i16 = mic_i16[:min_len]
        ref_i16 = ref_i16[:min_len]

        cleaned = np.zeros(min_len, dtype=np.int16)

        # Process in AEC_FRAME_SIZE chunks
        for i in range(0, min_len, AEC_FRAME_SIZE):
            end = i + AEC_FRAME_SIZE
            if end > min_len:
                # Pad last frame
                mic_frame = np.zeros(AEC_FRAME_SIZE, dtype=np.int16)
                ref_frame = np.zeros(AEC_FRAME_SIZE, dtype=np.int16)
                remaining = min_len - i
                mic_frame[:remaining] = mic_i16[i:min_len]
                ref_frame[:remaining] = ref_i16[i:min_len]
                result = aec.process(mic_frame, ref_frame)
                cleaned[i:min_len] = result[:remaining]
            else:
                cleaned[i:end] = aec.process(mic_i16[i:end], ref_i16[i:end])

        return cleaned.astype(np.float32) / 32768.0

    return process


def _get_ref_segment(tts_concat: np.ndarray, position: int, length: int) -> np.ndarray:
    """Get a segment of TTS reference audio at the given position."""
    if position >= len(tts_concat):
        return np.zeros(length, dtype=np.float32)
    seg = tts_concat[position:position + length]
    if len(seg) < length:
        return np.concatenate([seg, np.zeros(length - len(seg), dtype=np.float32)])
    return seg


# ---------------------------------------------------------------------------
# Phase 4: Real-time diarization
# ---------------------------------------------------------------------------

class DiarizationEngine:
    """Speaker diarization using pyannote.audio.

    Runs offline per-utterance: given a complete audio buffer, returns
    time-aligned speaker segments.
    """

    def __init__(self, hf_token: str, gpu_device: int = 0,
                 min_speakers: int = 1, max_speakers: int = 4):
        import torch
        from pyannote.audio import Pipeline

        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._device = f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu"

        logger.info("Loading pyannote diarization pipeline on %s...", self._device)
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        self._pipeline.to(torch.device(self._device))
        logger.info("Diarization pipeline loaded.")

    def diarize(self, audio_int16: np.ndarray,
                sample_rate: int = PIPELINE_SAMPLE_RATE) -> list[dict]:
        """Diarize an audio buffer.

        Returns list of segments: [{"start": float, "end": float, "speaker": str}, ...]
        Speaker labels are "SPEAKER_00", "SPEAKER_01", etc.
        """
        import torch

        audio_float = audio_int16.astype(np.float32) / 32768.0
        waveform = torch.tensor(audio_float).unsqueeze(0)

        diarization = self._pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })
        return segments

    def diarize_with_speaker_id(
        self, audio_int16: np.ndarray, speaker_id: "SpeakerIdentifier",
        sample_rate: int = PIPELINE_SAMPLE_RATE,
    ) -> list[dict]:
        """Diarize and map pyannote labels to enrolled speaker names.

        Returns segments with resolved speaker names instead of SPEAKER_XX.
        """
        segments = self.diarize(audio_int16, sample_rate)
        if not segments or speaker_id is None:
            return segments

        # For each unique pyannote label, extract audio and run speaker ID
        label_to_name: dict[str, str] = {}
        for seg in segments:
            label = seg["speaker"]
            if label in label_to_name:
                continue
            # Extract audio for this segment
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            seg_audio = audio_int16[start_sample:end_sample]
            if len(seg_audio) < sample_rate // 2:  # need at least 0.5s
                label_to_name[label] = speaker_id.unknown_label
                continue
            name, score = speaker_id.identify(seg_audio)
            label_to_name[label] = name
            logger.info("Diarization: %s -> %s (%.2f)", label, name, score)

        # Apply resolved names
        for seg in segments:
            seg["speaker"] = label_to_name.get(seg["speaker"], seg["speaker"])
        return segments


# ---------------------------------------------------------------------------
# Phase 6: Audio feedback (chime/tick)
# ---------------------------------------------------------------------------

def make_chime(duration: float = CHIME_DURATION_S,
               tick_every: float = CHIME_TICK_EVERY_S,
               sample_rate: int = CHIME_SAMPLE_RATE) -> np.ndarray:
    """Generate a chime + periodic tick audio buffer.

    Returns float32 audio: two-tone chime head followed by periodic ticks.
    """
    total_samples = int(duration * sample_rate)
    buf = np.zeros(total_samples, dtype=np.float32)

    # --- Chime head: 880Hz (90ms) + 30ms gap + 1320Hz (100ms) ---
    def _tone(freq, dur_s, amp=0.6):
        n = int(dur_s * sample_rate)
        t = np.arange(n, dtype=np.float32) / sample_rate
        envelope = np.sin(np.pi * np.arange(n) / n)  # Hann envelope
        return amp * envelope * np.sin(2 * np.pi * freq * t)

    tone1 = _tone(880, 0.09)
    gap = int(0.03 * sample_rate)
    tone2 = _tone(1320, 0.10)
    chime_head = np.concatenate([tone1, np.zeros(gap, dtype=np.float32), tone2])
    chime_len = len(chime_head)
    buf[:chime_len] = chime_head

    # --- Periodic ticks: 550Hz, 40ms, amplitude 0.18 ---
    tick = _tone(550, 0.04, amp=0.18)
    tick_samples = int(tick_every * sample_rate)
    # Start ticks after chime head + a small gap
    tick_start = chime_len + int(0.2 * sample_rate)
    for pos in range(tick_start, total_samples - len(tick), tick_samples):
        buf[pos:pos + len(tick)] = tick

    return buf


class TextToSpeech:
    """TTS via HTTP API server. Supports 'orpheus' (CosyVoice3), 'fish' (Fish Speech), and 'qwen3' backends."""

    def __init__(self, api_url: str = "http://127.0.0.1:8090",
                 sample_rate: int = 24000, speed: float = 1.0,
                 backend: str = "orpheus", reference_id: str = "cullen"):
        self.api_url = api_url.rstrip("/")
        self.sample_rate = sample_rate
        self.speed = speed
        self.backend = backend
        self.reference_id = reference_id
        self._bytes_per_sample = 2

        if self.backend == "fish":
            import msgpack  # noqa: F401 — ensure available
            self._msgpack = msgpack
            # Fish Speech has no /health endpoint; just check connectivity
            try:
                requests.get(self.api_url, timeout=5)
            except Exception:
                pass
        else:
            try:
                r = requests.get(f"{self.api_url}/health", timeout=5)
                info = r.json()
                self.sample_rate = info.get("sample_rate", sample_rate)
            except Exception:
                pass

    # -- Fish Speech backend ---------------------------------------------------

    def _synthesize_fish(self, text: str, audio_q: queue.Queue,
                         stop_event: threading.Event | None = None):
        """Fish Speech TTS — msgpack request, WAV response."""
        silence_pad = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)

        payload = self._msgpack.packb({
            "text": text.strip(),
            "reference_id": self.reference_id,
            "format": "wav",
            "streaming": True,
        })

        try:
            resp = requests.post(
                f"{self.api_url}/v1/tts",
                data=payload,
                headers={"Content-Type": "application/msgpack"},
                stream=True, timeout=(10, 120),
            )
            resp.raise_for_status()
        except requests.RequestException:
            return

        # Accumulate the full response (streaming WAV may include multiple
        # headers per chunk, so it's safest to collect then parse).
        chunks: list[bytes] = []
        for raw_chunk in resp.iter_content(chunk_size=8192):
            if stop_event and stop_event.is_set():
                resp.close()
                return
            if raw_chunk:
                chunks.append(raw_chunk)

        if not chunks or (stop_event and stop_event.is_set()):
            return

        wav_data = b"".join(chunks)

        try:
            with wave.open(io.BytesIO(wav_data), "rb") as wf:
                sr = wf.getframerate()
                sw = wf.getsampwidth()
                nframes = wf.getnframes()
                raw_pcm = wf.readframes(nframes)
        except Exception:
            return

        # Update sample rate from the WAV header
        self.sample_rate = sr

        # Convert PCM bytes to float32
        if sw == 2:
            samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            # 32-bit int PCM
            samples = np.frombuffer(raw_pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return

        # Re-create silence pad at correct sample rate
        silence_pad = np.zeros(int(sr * 0.2), dtype=np.float32)
        audio_q.put(silence_pad)

        # Feed in manageable chunks (~100ms)
        chunk_size = int(sr * 0.1)
        for i in range(0, len(samples), chunk_size):
            if stop_event and stop_event.is_set():
                return
            audio_q.put(samples[i:i + chunk_size])

        audio_q.put(silence_pad)

    # -- Qwen3-TTS backend -----------------------------------------------------

    def _synthesize_qwen3(self, text: str, audio_q: queue.Queue,
                          stop_event: threading.Event | None = None):
        """Qwen3-TTS — OpenAI-compatible API, streaming raw PCM."""
        t0 = time.time()
        silence_pad = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)
        total_samples = 0
        pcm_buffer = b""
        read_size = int(self.sample_rate * 0.1) * self._bytes_per_sample

        try:
            resp = requests.post(
                f"{self.api_url}/v1/audio/speech",
                json={
                    "input": text.strip(),
                    "voice": f"clone:{self.reference_id}",
                    "response_format": "pcm",
                    "stream": True,
                },
                stream=True, timeout=(10, 120),
            )
            resp.raise_for_status()
            audio_q.put(silence_pad)

            for raw_chunk in resp.iter_content(chunk_size=read_size):
                if stop_event and stop_event.is_set():
                    resp.close()
                    break
                if not raw_chunk:
                    continue
                pcm_buffer += raw_chunk
                usable = len(pcm_buffer) - (len(pcm_buffer) % self._bytes_per_sample)
                if usable == 0:
                    continue
                pcm_data = pcm_buffer[:usable]
                pcm_buffer = pcm_buffer[usable:]
                samples = np.frombuffer(
                    pcm_data, dtype=np.int16
                ).astype(np.float32) / 32768.0
                total_samples += len(samples)
                audio_q.put(samples)

            # Flush remaining buffer
            if pcm_buffer and not (stop_event and stop_event.is_set()):
                usable = len(pcm_buffer) - (len(pcm_buffer) % self._bytes_per_sample)
                if usable > 0:
                    samples = np.frombuffer(
                        pcm_buffer[:usable], dtype=np.int16
                    ).astype(np.float32) / 32768.0
                    total_samples += len(samples)
                    audio_q.put(samples)

            audio_q.put(silence_pad)
        except requests.RequestException:
            pass

    # -- Orpheus / CosyVoice backend -------------------------------------------

    def _synthesize_orpheus(self, text: str, audio_q: queue.Queue,
                            stop_event: threading.Event | None = None):
        """Original streaming PCM int16 backend."""
        silence_pad = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)
        total_samples = 0
        pcm_buffer = b""
        read_size = int(self.sample_rate * 0.1) * self._bytes_per_sample

        try:
            resp = requests.post(
                f"{self.api_url}/v1/tts",
                json={"text": text.strip(), "speed": self.speed},
                stream=True, timeout=(10, 120),
            )
            resp.raise_for_status()
            audio_q.put(silence_pad)

            for raw_chunk in resp.iter_content(chunk_size=read_size):
                if stop_event and stop_event.is_set():
                    resp.close()
                    break
                if not raw_chunk:
                    continue
                pcm_buffer += raw_chunk
                usable = len(pcm_buffer) - (len(pcm_buffer) % self._bytes_per_sample)
                if usable == 0:
                    continue
                pcm_data = pcm_buffer[:usable]
                pcm_buffer = pcm_buffer[usable:]
                samples = np.frombuffer(
                    pcm_data, dtype=np.int16
                ).astype(np.float32) / 32768.0
                total_samples += len(samples)
                audio_q.put(samples)

            usable = len(pcm_buffer) - (len(pcm_buffer) % self._bytes_per_sample)
            if usable > 0 and not (stop_event and stop_event.is_set()):
                pcm_data = pcm_buffer[:usable]
                samples = np.frombuffer(
                    pcm_data, dtype=np.int16
                ).astype(np.float32) / 32768.0
                total_samples += len(samples)
                audio_q.put(samples)

            if not (stop_event and stop_event.is_set()):
                audio_q.put(silence_pad)

        except requests.RequestException:
            return

    # -- Public API ------------------------------------------------------------

    def synthesize_streaming(self, text: str, audio_q: queue.Queue,
                             stop_event: threading.Event | None = None):
        if self.backend == "fish":
            self._synthesize_fish(text, audio_q, stop_event)
        elif self.backend == "qwen3":
            self._synthesize_qwen3(text, audio_q, stop_event)
        else:
            self._synthesize_orpheus(text, audio_q, stop_event)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        chunks: list[np.ndarray] = []
        chunk_q: queue.Queue = queue.Queue()
        self.synthesize_streaming(text, chunk_q)
        while not chunk_q.empty():
            chunks.append(chunk_q.get())
        if not chunks:
            return np.array([], dtype=np.float32), self.sample_rate
        return np.concatenate(chunks), self.sample_rate


# ---------------------------------------------------------------------------
# Audio capture — callback API (Pa_ReadStream has a heap corruption bug)
# ---------------------------------------------------------------------------

class _AudioReader:
    """Wraps sd.InputStream using the callback API to avoid Pa_ReadStream.

    PortAudio's blocking Pa_ReadStream corrupts the glibc heap on this
    system (PortAudio 19.7.0, PipeWire backend). The callback API is
    unaffected, so we funnel audio through a queue instead.
    """

    def __init__(self, samplerate: int, device: int, blocksize: int):
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="int16",
            device=device,
            blocksize=blocksize,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata[:, 0].copy())

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Return the next audio block, or None on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *args):
        self._stream.stop()
        self._stream.close()


class _WebSocketAudioReader:
    """Audio reader that pulls PCM frames from a WebSocket queue."""

    def __init__(self, frame_queue: queue.Queue, frame_size: int = VAD_FRAME_SIZE):
        self._q = frame_queue
        self._frame_size = frame_size
        self._buffer = np.array([], dtype=np.int16)

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        deadline = time.time() + timeout
        while len(self._buffer) < self._frame_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                chunk = self._q.get(timeout=min(remaining, 0.1))
                self._buffer = np.concatenate([self._buffer, chunk])
            except queue.Empty:
                continue
        frame = self._buffer[:self._frame_size]
        self._buffer = self._buffer[self._frame_size:]
        return frame

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Discord uses the same queue-based reader as WebSocket
_DiscordAudioReader = _WebSocketAudioReader


def wait_for_wake_word_with_buffer(
    wake_word: WakeWordDetector,
    mic_device: int,
    mic_native_rate: int,
    needs_resample: bool,
    mic_ww_blocksize: int,
    stop_event: threading.Event | None = None,
    voice_enabled: threading.Event | None = None,
    reader_factory: Callable[[], _WebSocketAudioReader | _AudioReader | _DiscordAudioReader] | None = None,
    tts_playing: threading.Event | None = None,
    drain_mic_q: Callable[[], None] | None = None,
) -> np.ndarray | None:
    """Listen for wake word while keeping a ring buffer of recent audio.

    Returns the ring buffer contents as int16 PCM at 16kHz when wake word
    is detected, or None if stopped.

    If `tts_playing` is set, the loop pauses consuming so the barge-in
    monitor in voice_mode._handle_say can own the shared mic queue during
    TTS. After TTS ends, `drain_mic_q` (if provided) clears the bleed and
    the wake-word detector state resets.
    """
    wake_word.reset()
    ring_buffer: deque[np.ndarray] = deque(maxlen=WAKEWORD_RING_BUFFER_FRAMES)

    reader_ctx = reader_factory() if reader_factory else _AudioReader(mic_native_rate, mic_device, mic_ww_blocksize)
    with reader_ctx as reader:
        while True:
            if stop_event and stop_event.is_set():
                return None
            if voice_enabled and not voice_enabled.is_set():
                return None

            if tts_playing is not None and tts_playing.is_set():
                # Stand aside; barge-in monitor owns the shared mic queue.
                while tts_playing.is_set():
                    if stop_event and stop_event.is_set():
                        return None
                    if voice_enabled and not voice_enabled.is_set():
                        return None
                    time.sleep(0.05)
                if drain_mic_q is not None:
                    drain_mic_q()
                wake_word.reset()
                ring_buffer.clear()
                continue

            audio_chunk = reader.read(timeout=0.5)
            if audio_chunk is None:
                continue
            if needs_resample:
                audio_chunk = resample_int16(audio_chunk, WW_FRAME_SIZE)

            ring_buffer.append(audio_chunk.copy())

            if wake_word.process(audio_chunk):
                wake_word.reset()
                if ring_buffer:
                    return np.concatenate(list(ring_buffer))
                return None


def record_utterance(
    vad: VAD,
    mic_device: int,
    mic_native_rate: int,
    needs_resample: bool,
    mic_vad_blocksize: int,
    silence_frames_threshold: int,
    stop_event: threading.Event | None = None,
    reader_factory: Callable[[], _WebSocketAudioReader | _AudioReader | _DiscordAudioReader] | None = None,
    smart_turn: SmartTurnDetector | None = None,
) -> np.ndarray | None:
    """Record speech using VAD for boundary detection. Returns int16 PCM or None.

    If smart_turn is provided, after silence threshold is reached, checks
    end-of-turn probability before stopping. If prob < threshold, resets
    silence counter and keeps listening.
    """
    audio_buffer: list[bytes] = []
    silent_frames = 0
    speech_started = False
    pre_speech_buffer: deque[bytes] = deque(maxlen=10)

    reader_ctx = reader_factory() if reader_factory else _AudioReader(mic_native_rate, mic_device, mic_vad_blocksize)
    with reader_ctx as reader:
        while True:
            if stop_event and stop_event.is_set():
                break

            raw_frame = reader.read(timeout=0.5)
            if raw_frame is None:
                continue
            if needs_resample:
                frame = resample_int16(raw_frame, VAD_FRAME_SIZE)
            else:
                frame = raw_frame
            frame_float = frame.astype(np.float32) / 32768.0
            is_speech = vad.is_speech(frame_float)

            if not speech_started:
                pre_speech_buffer.append(frame.tobytes())
                if is_speech:
                    speech_started = True
                    silent_frames = 0
                    for buf in pre_speech_buffer:
                        audio_buffer.append(buf)
            else:
                audio_buffer.append(frame.tobytes())
                if is_speech:
                    silent_frames = 0
                else:
                    silent_frames += 1
                    if silent_frames >= silence_frames_threshold:
                        # Phase 1: Smart Turn check before finalizing
                        if smart_turn and audio_buffer:
                            all_pcm = np.frombuffer(b"".join(audio_buffer), dtype=np.int16)
                            all_float = all_pcm.astype(np.float32) / 32768.0
                            prob = smart_turn.predict(all_float)
                            buffered_samples = sum(len(b) // 2 for b in audio_buffer)
                            print(f"  [smart_turn] prob={prob:.2f} thr={smart_turn.threshold:.2f} dur={buffered_samples/PIPELINE_SAMPLE_RATE:.1f}s", flush=True)
                            if prob < smart_turn.threshold and buffered_samples < MAX_UTTERANCE_SAMPLES:
                                # Not end of turn — keep listening (until hard cap)
                                silent_frames = 0
                                continue
                        break

    vad.reset()

    if not audio_buffer:
        return None

    pcm_bytes = b"".join(audio_buffer)
    return np.frombuffer(pcm_bytes, dtype=np.int16)


def active_listen(
    vad: VAD,
    mic_device: int,
    mic_native_rate: int,
    needs_resample: bool,
    mic_vad_blocksize: int,
    window_seconds: float,
    silence_frames_threshold: int,
    stop_event: threading.Event | None = None,
    reader_factory: Callable[[], _WebSocketAudioReader | _AudioReader | _DiscordAudioReader] | None = None,
    smart_turn: SmartTurnDetector | None = None,
) -> np.ndarray | None:
    """Listen for speech within a time window.

    If speech detected within window_seconds, records the full utterance
    and returns it. If no speech before timeout, returns None.
    """
    t0 = time.time()
    audio_buffer: list[bytes] = []
    silent_frames = 0
    speech_started = False
    pre_speech_buffer: deque[bytes] = deque(maxlen=10)

    reader_ctx = reader_factory() if reader_factory else _AudioReader(mic_native_rate, mic_device, mic_vad_blocksize)
    with reader_ctx as reader:
        while True:
            if stop_event and stop_event.is_set():
                vad.reset()
                return None

            elapsed = time.time() - t0

            if not speech_started and elapsed >= window_seconds:
                vad.reset()
                return None

            raw_frame = reader.read(timeout=0.5)
            if raw_frame is None:
                continue
            if needs_resample:
                frame = resample_int16(raw_frame, VAD_FRAME_SIZE)
            else:
                frame = raw_frame
            frame_float = frame.astype(np.float32) / 32768.0
            is_speech = vad.is_speech(frame_float)

            if not speech_started:
                pre_speech_buffer.append(frame.tobytes())
                if is_speech:
                    speech_started = True
                    silent_frames = 0
                    for buf in pre_speech_buffer:
                        audio_buffer.append(buf)
            else:
                audio_buffer.append(frame.tobytes())
                if is_speech:
                    silent_frames = 0
                else:
                    silent_frames += 1
                    if silent_frames >= silence_frames_threshold:
                        # Phase 1: Smart Turn check
                        if smart_turn and audio_buffer:
                            all_pcm = np.frombuffer(b"".join(audio_buffer), dtype=np.int16)
                            all_float = all_pcm.astype(np.float32) / 32768.0
                            prob = smart_turn.predict(all_float)
                            buffered_samples = sum(len(b) // 2 for b in audio_buffer)
                            print(f"  [smart_turn:follow] prob={prob:.2f} thr={smart_turn.threshold:.2f} dur={buffered_samples/PIPELINE_SAMPLE_RATE:.1f}s", flush=True)
                            if prob < smart_turn.threshold and buffered_samples < MAX_UTTERANCE_SAMPLES:
                                silent_frames = 0
                                continue
                        break

    vad.reset()

    if not audio_buffer:
        return None

    pcm_bytes = b"".join(audio_buffer)
    return np.frombuffer(pcm_bytes, dtype=np.int16)


# ---------------------------------------------------------------------------
# WebSocket audio server
# ---------------------------------------------------------------------------

@dataclass
class ClientState:
    """Per-client state for multi-client WebSocket audio server."""
    websocket: Any
    frame_queue: queue.Queue  # per-client audio frames
    control_queue: queue.Queue  # per-client control messages
    send_queue: asyncio.Queue | None = None  # per-client send queue
    session_key: str | None = None
    sender_task: asyncio.Task | None = None


class WebSocketAudioServer:
    """WebSocket server that accepts PCM audio from multiple browser clients."""

    MAX_CLIENTS = 8

    def __init__(self, port: int = 8095, callbacks: PipelineCallbacks | None = None):
        self.port = port
        self._clients: dict[str, ClientState] = {}
        self._clients_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self.cb = callbacks

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        import websockets
        async with websockets.serve(self._handler, "127.0.0.1", self.port):
            await asyncio.Future()  # run forever

    async def _handler(self, websocket):
        import sys; print(f"VOICE_DEBUG: _handler called from {websocket.remote_address}", file=sys.stderr, flush=True)
        # Reject if at capacity
        with self._clients_lock:
            if len(self._clients) >= self.MAX_CLIENTS:
                await websocket.close(1013, "Max clients reached")
                return

        client_id = str(uuid.uuid4())

        # Wait for hello message (5 second timeout)
        try:
            import sys; print("VOICE_DEBUG: waiting for hello...", file=sys.stderr, flush=True)
            hello_raw = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            logger.warning(f"WS handler: received hello_raw: {hello_raw[:100] if isinstance(hello_raw, str) else type(hello_raw)}")
        except (asyncio.TimeoutError, Exception):
            await websocket.close(1008, "Expected hello message")
            return

        try:
            hello = json.loads(hello_raw) if isinstance(hello_raw, str) else {}
        except (json.JSONDecodeError, TypeError):
            hello = {}

        if hello.get("type") != "hello":
            await websocket.close(1008, "First message must be hello")
            return

        # Create per-client state
        send_queue: asyncio.Queue = asyncio.Queue()
        client_state = ClientState(
            websocket=websocket,
            frame_queue=queue.Queue(),
            control_queue=queue.Queue(),
            send_queue=send_queue,
            session_key=hello.get("sessionKey"),
        )

        with self._clients_lock:
            self._clients[client_id] = client_state
            logger.warning(f"WS handler: registered client {client_id[:8]}, total: {len(self._clients)}")

        # Create sender task
        sender = asyncio.create_task(self._sender(client_id, websocket, send_queue))
        client_state.sender_task = sender

        # Send welcome with assigned client_id
        try:
            await websocket.send(json.dumps({
                "type": "welcome",
                "clientId": client_id,
            }))
        except Exception:
            pass

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    samples = np.frombuffer(message, dtype=np.int16)
                    client_state.frame_queue.put(samples)
                else:
                    try:
                        cmd = json.loads(message)
                        cmd["_client_id"] = client_id
                        client_state.control_queue.put(cmd)
                    except json.JSONDecodeError:
                        pass
        finally:
            sender.cancel()
            with self._clients_lock:
                logger.warning(f"WS handler: client {client_id[:8]} disconnected, removing")
                self._clients.pop(client_id, None)

    async def _sender(self, client_id: str, websocket, send_queue: asyncio.Queue):
        """Send queued messages to a specific client."""
        while True:
            msg = await send_queue.get()
            try:
                if isinstance(msg, bytes):
                    await websocket.send(msg)
                else:
                    await websocket.send(json.dumps(msg))
            except Exception:
                break

    def send_message(self, msg: dict, client_id: str | None = None):
        """Thread-safe: queue a JSON message to one or all clients."""
        if not self._loop:
            return
        if client_id is not None:
            with self._clients_lock:
                cs = self._clients.get(client_id)
            if cs and cs.send_queue:
                self._loop.call_soon_threadsafe(cs.send_queue.put_nowait, msg)
        else:
            with self._clients_lock:
                clients = list(self._clients.values())
            for cs in clients:
                if cs.send_queue:
                    self._loop.call_soon_threadsafe(cs.send_queue.put_nowait, msg)

    def send_binary(self, data: bytes, client_id: str | None = None):
        """Thread-safe: queue binary data to one or all clients."""
        if not self._loop:
            return
        if client_id is not None:
            with self._clients_lock:
                cs = self._clients.get(client_id)
            if cs and cs.send_queue:
                self._loop.call_soon_threadsafe(cs.send_queue.put_nowait, data)
        else:
            with self._clients_lock:
                clients = list(self._clients.values())
            for cs in clients:
                if cs.send_queue:
                    self._loop.call_soon_threadsafe(cs.send_queue.put_nowait, data)

    def send_audio(self, chunk: np.ndarray, sample_rate: int, client_id: str | None = None):
        """Send a TTS audio chunk (float32 PCM) to one or all browser clients."""
        if not self.has_clients:
            return
        audio = np.asarray(chunk, dtype=np.float32)
        self.send_binary(audio.tobytes(), client_id=client_id)

    def get_control(self, client_id: str) -> dict | None:
        """Get next control message from a specific client."""
        with self._clients_lock:
            cs = self._clients.get(client_id)
        if not cs:
            return None
        try:
            return cs.control_queue.get_nowait()
        except queue.Empty:
            return None

    def get_client_ids(self) -> set[str]:
        """Return snapshot of currently connected client IDs."""
        with self._clients_lock:
            return set(self._clients.keys())

    @property
    def has_clients(self) -> bool:
        with self._clients_lock:
            return bool(self._clients)

    def get_client_session_key(self, client_id: str) -> str | None:
        """Look up the session key for a connected client."""
        with self._clients_lock:
            cs = self._clients.get(client_id)
        return cs.session_key if cs else None

    def make_reader(self, client_id: str, frame_size: int = VAD_FRAME_SIZE) -> _WebSocketAudioReader:
        """Create a reader for a specific client's audio frames."""
        with self._clients_lock:
            cs = self._clients.get(client_id)
        if not cs:
            # Return a reader on an empty queue; will timeout harmlessly
            return _WebSocketAudioReader(queue.Queue(), frame_size)
        # Drain stale frames from between pipeline phases
        while not cs.frame_queue.empty():
            try:
                cs.frame_queue.get_nowait()
            except queue.Empty:
                break
        return _WebSocketAudioReader(cs.frame_queue, frame_size)


# ---------------------------------------------------------------------------
# Pipeline callbacks protocol
# ---------------------------------------------------------------------------

class PipelineCallbacks(Protocol):
    def on_state_changed(self, state: State) -> None: ...
    def on_init_progress(self, component: str) -> None: ...
    def on_init_complete(self) -> None: ...
    def on_transcript(self, text: str, speaker: str, is_continuity: bool, client_id: str | None = None) -> None: ...
    def on_continuity_status(self, msg: str) -> None: ...
    def on_error(self, error: str) -> None: ...


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Runs the full voice pipeline in a background thread."""

    def __init__(self, config: dict, callbacks: PipelineCallbacks):
        self.config = config
        self.cb = callbacks
        self.voice_enabled = threading.Event()
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Components (set by init_components)
        self.wake_word: WakeWordDetector | None = None
        self.vad: VAD | None = None
        self.stt: SpeechToText | WhisperSTT | None = None
        self.speaker_id: SpeakerIdentifier | None = None
        self.tts: TextToSpeech | None = None
        self.smart_turn: SmartTurnDetector | None = None  # Phase 1
        self.diarization: DiarizationEngine | None = None  # Phase 4

        # Last utterance state (for ASR correction)
        self._last_utterance: dict = {}
        self._last_utterance_lock = threading.Lock()

        # Phase 5: Uncertain speaker tracking
        self._uncertain_speakers: deque[dict] = deque(maxlen=50)
        self._uncertain_speakers_lock = threading.Lock()

        # TTS coordination
        self._tts_playing = threading.Event()  # set while TTS is playing
        self._tts_stop = threading.Event()     # set to request TTS stop (wake word interrupt)

        # Local-mode shared mic stream + queue. Opened once at init_components
        # and kept open for the daemon's lifetime so wake-word detection,
        # utterance recording, and barge-in monitoring all consume from one
        # source — ALSA hw devices reject multiple concurrent readers, so we
        # serialize access through this queue. Only one consumer should pull
        # from it at a time (state-machine driven).
        self._mic_q: queue.Queue[np.ndarray] = queue.Queue()
        self._mic_stream: object | None = None  # sd.InputStream

        # Conversation mode — after TTS completes, accept ASR from the same
        # speaker for a window without requiring a new wake word.
        self._conversation_mode_enabled: bool = config.get(
            "conversation_mode_enabled", True)
        self._conversation_window_s: float = float(config.get(
            "conversation_window_s", 8.0))
        self._conversation_tts_wait_s: float = float(config.get(
            "conversation_tts_wait_s", 30.0))
        self._conversation_max_turns: int = int(config.get(
            "conversation_max_turns", 10))

        # Phase 2/3: AEC and barge-in state
        self._aec_enabled: bool = config.get("aec_enabled", True)
        self._barge_in_enabled: bool = config.get("barge_in_enabled", True)
        self._barge_in_vad_threshold: float = config.get(
            "barge_in_vad_threshold", BARGE_IN_VAD_THRESHOLD)
        self._barge_in_consec_chunks: int = config.get(
            "barge_in_consecutive_chunks", BARGE_IN_CONSEC_CHUNKS)
        self._barge_in_grace_period: float = config.get(
            "barge_in_grace_period_ms", BARGE_IN_GRACE_PERIOD_S * 1000) / 1000.0
        # Shared mic queue for barge-in during TTS (local input mode)
        self._mic_queue: queue.Queue | None = None
        # TTS reference buffer (float32 at 16kHz) accumulated during TTS playback
        self._tts_ref_buffer: list[np.ndarray] = []
        self._tts_ref_lock = threading.Lock()
        self._tts_play_start: float = 0.0

        # Phase 6: Audio feedback
        self._chime_enabled: bool = config.get("audio_feedback_enabled", False)
        self._chime_audio: np.ndarray | None = None

        # Mic config
        self.mic_device: int = 0
        self.mic_native_rate: int = PIPELINE_SAMPLE_RATE
        self.needs_resample: bool = False
        self.mic_ww_blocksize: int = WW_FRAME_SIZE
        self.mic_vad_blocksize: int = VAD_FRAME_SIZE
        self.silence_frames_threshold: int = 31  # ~1000ms at 32ms/frame
        self.speaker_threshold: float = 0.75
        self.listen_window_s: float = DEFAULT_LISTEN_WINDOW_S

        # Output device (for future TTS playback)
        self.output_device: int | None = None

        # Input mode: "local" (mic), "websocket" (browser), or "discord"
        self.input_mode: str = config.get("input_mode", "local")
        self._ws_server: WebSocketAudioServer | None = None

        # Per-client pipeline threads (websocket mode)
        self._client_threads: dict[str, threading.Thread] = {}
        self._client_threads_lock = threading.Lock()
        self._stt_lock = threading.Lock()
        self._speaker_lock = threading.Lock()

        # Discord voice bridge
        self._discord_reader_queue: queue.Queue | None = None
        self._discord_reader_factory: Callable | None = None

    @property
    def ws_server(self) -> WebSocketAudioServer | None:
        return self._ws_server

    @property
    def discord_audio_queue(self) -> queue.Queue | None:
        return self._discord_reader_queue

    def notify_tts_start(self) -> None:
        """Signal that TTS playback has started."""
        self._tts_playing.set()
        self._tts_stop.clear()
        self._tts_play_start = time.monotonic()
        with self._tts_ref_lock:
            self._tts_ref_buffer.clear()
        # Reset per-utterance state. mic_pos/consec_speech are also reset on
        # first ref chunk arrival in append_tts_reference, but clearing here
        # too keeps things tidy if AEC is disabled.
        self._barge_in_mic_pos = 0
        self._barge_in_consec_speech = 0
        # Reset per-utterance debug flags so check_barge_in's heartbeat fires
        # once per TTS turn instead of just once per process lifetime.
        self._bi_no_ref_warned = False
        self._bi_active_warned = False
        self.cb.on_state_changed(State.SPEAKING)
        if self._ws_server and self._ws_server.has_clients:
            self._ws_server.send_message({"type": "state", "state": "SPEAKING"})

    def notify_tts_end(self) -> None:
        """Signal that TTS playback has ended."""
        self._tts_playing.clear()
        self._tts_stop.clear()
        with self._tts_ref_lock:
            self._tts_ref_buffer.clear()
        # Send state update so UI leaves SPEAKING
        ws = self._ws_server
        if ws and ws.has_clients:
            ws.send_message({"type": "state", "state": "ACTIVE_LISTEN"})
        self.cb.on_state_changed(State.IDLE)

    def request_tts_stop(self) -> None:
        """Request TTS to stop (e.g. wake word interrupt or barge-in)."""
        self._tts_stop.set()

    def is_tts_stop_requested(self) -> bool:
        """Check if TTS stop has been requested."""
        return self._tts_stop.is_set()

    def append_tts_reference(self, audio_float32: np.ndarray, tts_sr: int) -> None:
        """Accumulate TTS reference audio for AEC (resampled to 16kHz).

        Called by TTS playback code as chunks are generated. On the first
        chunk, drain the mic queue and reset `_barge_in_mic_pos` so AEC
        starts with mic and ref samples aligned at index 0 — without this,
        the queue contains ~100-500 ms of pre-TTS mic data and AEC ends
        up subtracting the wrong window of the reference signal.
        """
        if not self._aec_enabled:
            return
        # Resample to pipeline rate if needed
        if tts_sr != PIPELINE_SAMPLE_RATE:
            n_out = int(len(audio_float32) * PIPELINE_SAMPLE_RATE / tts_sr)
            idx = np.linspace(0, len(audio_float32) - 1, n_out)
            ref_16k = np.interp(idx, np.arange(len(audio_float32)), audio_float32).astype(np.float32)
        else:
            ref_16k = audio_float32.astype(np.float32)
        with self._tts_ref_lock:
            first_chunk = len(self._tts_ref_buffer) == 0
            self._tts_ref_buffer.append(ref_16k)
        if first_chunk and self.input_mode == "local":
            self._drain_mic_q()
            self._barge_in_mic_pos = 0
            self._barge_in_consec_speech = 0

    def check_barge_in(self, mic_queue: queue.Queue, vad: VAD,
                       aec_process=None) -> bool:
        """Check if user is speaking during TTS playback (voice barge-in).

        Drains mic_queue, runs AEC + VAD with elevated threshold.
        Returns True if barge-in detected. Mic chunks are expected at
        `self.mic_native_rate` and resampled to 16 kHz before AEC/VAD.
        """
        if not self._barge_in_enabled:
            return False

        # Grace period: ignore first N ms of TTS
        if time.monotonic() - self._tts_play_start < self._barge_in_grace_period:
            return False

        # Get TTS reference
        with self._tts_ref_lock:
            if not self._tts_ref_buffer:
                # One-shot debug for the no-ref-yet case
                if not getattr(self, "_bi_no_ref_warned", False):
                    print("  [barge-in] no tts_ref yet, skipping", flush=True)
                    self._bi_no_ref_warned = True
                return False
            tts_concat = np.concatenate(self._tts_ref_buffer)

        # One-shot heartbeat: confirm we're past early-exits at least once
        if not getattr(self, "_bi_active_warned", False):
            print(f"  [barge-in] active: q_size={mic_queue.qsize()} ref_len={len(tts_concat)}", flush=True)
            self._bi_active_warned = True

        # consec_speech persists across polls — the monitor calls this every
        # 20 ms but the queue typically has 0-1 chunks per call, so a local
        # counter would never accumulate to threshold.
        consec_speech = getattr(self, "_barge_in_consec_speech", 0)
        mic_pos = getattr(self, "_barge_in_mic_pos", 0)

        while not mic_queue.empty():
            try:
                mic_chunk = mic_queue.get_nowait()
            except queue.Empty:
                break

            # Native-rate chunks → 16 kHz for AEC/VAD.
            if self.needs_resample:
                target_len = int(len(mic_chunk) * PIPELINE_SAMPLE_RATE / self.mic_native_rate)
                if target_len < VAD_FRAME_SIZE:
                    continue
                mic_chunk = resample_int16(mic_chunk, target_len)

            if len(mic_chunk) < VAD_FRAME_SIZE:
                continue

            frame_float = mic_chunk.astype(np.float32) / 32768.0

            # Apply AEC if available. Speex AEC has no double-talk detection,
            # so it can over-cancel when user speech overlaps with TTS — we
            # therefore evaluate VAD on both raw and cleaned audio and fire on
            # whichever crosses threshold (cleaned for clean-voice barge-in,
            # raw with a stricter requirement for cases where AEC eats the
            # user's voice).
            if aec_process is not None:
                ref = _get_ref_segment(tts_concat, mic_pos, len(frame_float))
                mic_pos += len(frame_float)
                cleaned = aec_process(frame_float, ref)
            else:
                cleaned = frame_float
                mic_pos += len(cleaned)

            cleaned_for_vad = cleaned[:VAD_FRAME_SIZE] if len(cleaned) >= VAD_FRAME_SIZE else np.pad(cleaned, (0, VAD_FRAME_SIZE - len(cleaned)))
            raw_for_vad = frame_float[:VAD_FRAME_SIZE] if len(frame_float) >= VAD_FRAME_SIZE else np.pad(frame_float, (0, VAD_FRAME_SIZE - len(frame_float)))
            prob_clean = vad.speech_probability(cleaned_for_vad)
            prob_raw = vad.speech_probability(raw_for_vad)

            # RMS comparison: when the user adds voice on top of Lloyd's
            # speaker output, raw mic energy goes up. If raw RMS substantially
            # exceeds AEC residual RMS, we have evidence of user voice that
            # Speex couldn't subtract.
            raw_rms = float(np.sqrt(np.mean(raw_for_vad ** 2)) + 1e-12)
            clean_rms = float(np.sqrt(np.mean(cleaned_for_vad ** 2)) + 1e-12)

            prob = max(prob_clean, prob_raw)  # use the larger; cleaned still wins when AEC is good

            if prob_clean > 0.3 or prob_raw > 0.85:  # log interesting frames
                print(f"  [barge-in] clean={prob_clean:.2f} raw={prob_raw:.2f} thr={self._barge_in_vad_threshold:.2f} consec={consec_speech}/{self._barge_in_consec_chunks} rms_raw={raw_rms:.3f} rms_clean={clean_rms:.3f}", flush=True)

            # Fire if cleaned VAD crosses threshold (clean-voice barge-in) OR
            # raw VAD is very high AND raw RMS exceeds clean RMS (= user voice
            # adding energy on top of Lloyd's room playback).
            cleaned_fires = prob_clean > self._barge_in_vad_threshold
            raw_fires = (prob_raw > 0.92 and raw_rms > clean_rms * 1.3)

            if cleaned_fires or raw_fires:
                consec_speech += 1
                if consec_speech >= self._barge_in_consec_chunks:
                    self._barge_in_mic_pos = mic_pos
                    self._barge_in_consec_speech = 0  # reset for next time
                    return True
            else:
                consec_speech = 0

        self._barge_in_mic_pos = mic_pos
        self._barge_in_consec_speech = consec_speech
        return False

    # --- Phase 5: Uncertain speaker tracking ---

    def record_uncertain_speaker(self, transcript: str, speaker: str,
                                  confidence: float, audio_duration: float) -> None:
        """Track uncertain speaker identifications for review."""
        with self._uncertain_speakers_lock:
            self._uncertain_speakers.append({
                "timestamp": time.time(),
                "transcript": transcript[:100],
                "speaker": speaker,
                "confidence": confidence,
                "audio_duration": audio_duration,
            })

    def get_uncertain_speakers(self) -> list[dict]:
        """Return recent uncertain speaker identifications."""
        with self._uncertain_speakers_lock:
            return list(self._uncertain_speakers)

    # --- Phase 5: Enhanced speaker operations ---

    def delete_speaker(self, name: str) -> bool:
        """Delete an enrolled speaker profile."""
        if self.speaker_id is None:
            return False
        return self.speaker_id.delete_profile(name)

    def get_speaker_stats(self) -> dict:
        """Return speaker profile statistics."""
        if self.speaker_id is None:
            return {}
        return self.speaker_id.get_stats()

    def init_components(self) -> None:
        """Load all models. Call from a background thread."""
        config = self.config

        if self.input_mode == "local":
            # Mic
            self.cb.on_init_progress("Microphone")
            self.mic_device, self.mic_native_rate = resolve_mic_device(config)
            self.needs_resample = self.mic_native_rate != PIPELINE_SAMPLE_RATE

            if self.needs_resample:
                ratio = self.mic_native_rate / PIPELINE_SAMPLE_RATE
                self.mic_ww_blocksize = int(WW_FRAME_SIZE * ratio)
                self.mic_vad_blocksize = int(VAD_FRAME_SIZE * ratio)
            else:
                self.mic_ww_blocksize = WW_FRAME_SIZE
                self.mic_vad_blocksize = VAD_FRAME_SIZE

            # Output device (optional — falls back to system default if unset/missing)
            out_idx = resolve_output_device(config)
            if out_idx is not None:
                self.output_device = out_idx

            # Always-on shared mic stream. Pushes int16 chunks into _mic_q at
            # native rate; consumers (wake-word, VAD, barge-in) read from
            # _mic_q via reader_factory. Block size is the VAD-aligned size
            # (smaller of the two consumer frame sizes), which gives ~32 ms
            # callback cadence at 16 kHz native.
            self._open_local_mic_stream()

        if self.input_mode == "websocket":
            self.cb.on_init_progress("WebSocket Audio Server")
            self._ws_server = WebSocketAudioServer(
                port=config.get("ws_audio_port", 8095),
                callbacks=self.cb,
            )
            self._ws_server.start()
            self.needs_resample = False

        if self.input_mode == "discord":
            self.cb.on_init_progress("Discord Audio Bridge")
            self._discord_reader_queue = queue.Queue()
            self._discord_reader_factory = lambda frame_size: _DiscordAudioReader(
                self._discord_reader_queue, frame_size
            )
            self.needs_resample = False

        vad_frame_ms = VAD_FRAME_SIZE / PIPELINE_SAMPLE_RATE * 1000
        silence_ms = config.get("silence_duration_ms", SILENCE_DURATION_MS)
        self.silence_frames_threshold = int(silence_ms / vad_frame_ms)
        self.speaker_threshold = config.get("speaker_id_threshold", 0.75)
        self.listen_window_s = config.get("listen_window_s", DEFAULT_LISTEN_WINDOW_S)
        self.max_followup_turns = config.get("max_followup_turns", 3)

        # Wake word
        self.cb.on_init_progress("Wake Word (openWakeWord)")
        ww_models = config.get("wake_word_models", [])
        self.wake_word = WakeWordDetector(
            model_names=ww_models,
            threshold=config.get("wake_word_threshold", 0.5),
            smoothing_window=config.get("wake_word_smoothing_window", 1),
            min_hits=config.get("wake_word_min_hits", 1),
        )

        # VAD
        self.cb.on_init_progress("VAD (Silero)")
        self.vad = VAD()

        # STT
        gpu_device = config.get("gpu_device", 0)
        stt_engine = config.get("stt_engine", "whisper")
        if stt_engine == "whisper":
            stt_model = config.get("stt_model", "small")
            hotwords_file = config.get("stt_hotwords_file")
            self.cb.on_init_progress(f"STT (Whisper {stt_model})")
            self.stt = WhisperSTT(
                model_size=stt_model, gpu_device=gpu_device,
                hotwords_file=hotwords_file,
            )
        else:
            stt_model = config.get("stt_model", "moonshine/base")
            self.cb.on_init_progress(f"STT (Moonshine {stt_model})")
            self.stt = SpeechToText(model_name=stt_model, gpu_device=gpu_device)

        # Speaker ID
        if config.get("speaker_id_enabled", False):
            self.cb.on_init_progress("Speaker ID (Resemblyzer)")
            self.speaker_id = SpeakerIdentifier(
                profiles_dir=config.get("speaker_profiles_dir", "speakers"),
                threshold=self.speaker_threshold,
                unknown_label=config.get("speaker_id_unknown_label", "Unknown"),
                gpu_device=gpu_device,
            )

        # TTS (optional)
        tts_url = config.get("tts_url", "")
        tts_backend = config.get("tts_backend", "orpheus")
        if tts_url:
            labels = {"fish": "TTS (Fish Speech)", "qwen3": "TTS (Qwen3-TTS)", "orpheus": "TTS (Orpheus)"}
            label = labels.get(tts_backend, "TTS (CosyVoice3)")
            self.cb.on_init_progress(label)
            self.tts = TextToSpeech(
                api_url=tts_url,
                sample_rate=config.get("tts_sample_rate", 24000),
                backend=tts_backend,
                reference_id=config.get("tts_reference_id", "cullen"),
            )

        # Phase 1: Smart Turn v3
        if config.get("smart_turn_enabled", True):
            smart_turn_model = config.get(
                "smart_turn_model",
                str(Path(__file__).parent / "models" / "smart_turn" / "smart_turn_v3.2_cpu.onnx")
            )
            if Path(smart_turn_model).exists():
                self.cb.on_init_progress("Smart Turn v3 (ONNX)")
                try:
                    self.smart_turn = SmartTurnDetector(
                        model_path=smart_turn_model,
                        threshold=config.get("smart_turn_threshold", SMART_TURN_DEFAULT_THRESHOLD),
                        gpu_device=gpu_device,
                    )
                    logger.info("Smart Turn v3 loaded (threshold=%.2f)", self.smart_turn.threshold)
                except Exception as e:
                    logger.warning("Smart Turn v3 failed to load: %s", e)
            else:
                logger.info("Smart Turn model not found at %s, skipping", smart_turn_model)

        # Phase 4: Diarization
        if config.get("diarization_enabled", False):
            hf_token = config.get("diarization_hf_token", "")
            if hf_token:
                self.cb.on_init_progress("Diarization (pyannote)")
                try:
                    self.diarization = DiarizationEngine(
                        hf_token=hf_token,
                        gpu_device=config.get("diarization_gpu_device", gpu_device),
                        min_speakers=config.get("diarization_min_speakers", 1),
                        max_speakers=config.get("diarization_max_speakers", 4),
                    )
                except Exception as e:
                    logger.warning("Diarization failed to load: %s", e)

        # Phase 6: Audio feedback
        if self._chime_enabled:
            self.cb.on_init_progress("Audio Feedback (chime)")
            self._chime_audio = make_chime()

        self.cb.on_init_complete()

    def set_input_device(self, device_index: int, sample_rate: int) -> None:
        """Change the input device. Takes effect on the next pipeline cycle."""
        self.mic_device = device_index
        self.mic_native_rate = sample_rate
        self.needs_resample = sample_rate != PIPELINE_SAMPLE_RATE
        if self.needs_resample:
            ratio = sample_rate / PIPELINE_SAMPLE_RATE
            self.mic_ww_blocksize = int(WW_FRAME_SIZE * ratio)
            self.mic_vad_blocksize = int(VAD_FRAME_SIZE * ratio)
        else:
            self.mic_ww_blocksize = WW_FRAME_SIZE
            self.mic_vad_blocksize = VAD_FRAME_SIZE

    def set_output_device(self, device_index: int) -> None:
        """Set the output device for TTS playback."""
        self.output_device = device_index

    # --- Local-mode shared mic plumbing ---

    def _open_local_mic_stream(self) -> None:
        """Open the always-on local mic InputStream that fans out to _mic_q."""
        if self._mic_stream is not None:
            return

        def _cb(indata, frames, time_info, status):
            try:
                self._mic_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        self._mic_stream = sd.InputStream(
            samplerate=self.mic_native_rate,
            channels=1, dtype="int16",
            device=self.mic_device,
            blocksize=self.mic_vad_blocksize,
            callback=_cb,
        )
        self._mic_stream.start()

    def _close_local_mic_stream(self) -> None:
        if self._mic_stream is None:
            return
        try:
            self._mic_stream.stop()
            self._mic_stream.close()
        except Exception:
            pass
        self._mic_stream = None

    def _drain_mic_q(self) -> None:
        """Discard any buffered chunks. Called when handing off between
        consumers (e.g. after TTS ends, before wake-word resumes) so we
        don't process stale audio that may include TTS bleed-through."""
        try:
            while True:
                self._mic_q.get_nowait()
        except queue.Empty:
            pass

    def _local_reader_factory(self, frame_size: int):
        """Return a queue-based reader pinned to the shared mic queue."""
        return lambda: _WebSocketAudioReader(self._mic_q, frame_size=frame_size)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._close_local_mic_stream()
        # Join client pipeline threads
        with self._client_threads_lock:
            threads = list(self._client_threads.values())
        for t in threads:
            t.join(timeout=5)

    def _run_loop(self) -> None:
        """Main pipeline loop."""
        if self.input_mode == "websocket":
            # Multi-client mode: poll for new clients, spawn per-client threads
            while not self.stop_event.is_set():
                ws = self._ws_server
                if not ws:
                    time.sleep(0.2)
                    continue
                # Check for new clients
                current_clients = ws.get_client_ids()
                with self._client_threads_lock:
                    tracked = set(self._client_threads.keys())
                new_clients = current_clients - tracked
                for cid in new_clients:
                    t = threading.Thread(
                        target=self._client_pipeline_thread,
                        args=(cid,), daemon=True,
                    )
                    with self._client_threads_lock:
                        self._client_threads[cid] = t
                    t.start()
                time.sleep(0.2)
            return  # exit _run_loop for websocket mode

        while not self.stop_event.is_set():
            if self.input_mode in ("local", "discord"):
                if not self.voice_enabled.wait(timeout=0.2):
                    continue

            try:
                if self.input_mode == "discord":
                    self._pipeline_iteration_discord()
                else:
                    self._pipeline_iteration()
            except Exception as e:
                self.cb.on_error(str(e))
                self.vad.reset()
                time.sleep(1)

    def _pipeline_iteration(self) -> None:
        """One full wake-word → conversation → return-to-idle cycle."""
        # Phase 6: Play chime on wake word detection
        chime_stream = None

        # 1. IDLE — wait for wake word
        self.cb.on_state_changed(State.IDLE)
        ww_audio = wait_for_wake_word_with_buffer(
            self.wake_word, self.mic_device, self.mic_native_rate,
            self.needs_resample, self.mic_ww_blocksize,
            stop_event=self.stop_event,
            voice_enabled=self.voice_enabled,
            reader_factory=self._local_reader_factory(self.mic_ww_blocksize),
            tts_playing=self._tts_playing,
            drain_mic_q=self._drain_mic_q,
        )
        if ww_audio is None or self.stop_event.is_set():
            return
        if not self.voice_enabled.is_set():
            return

        # Phase 6: Start chime playback
        if self._chime_enabled and self._chime_audio is not None:
            try:
                chime_stream = sd.OutputStream(
                    samplerate=CHIME_SAMPLE_RATE,
                    channels=1, dtype="float32",
                    device=self.output_device,
                )
                chime_stream.start()
                # Write chime in background (non-blocking)
                chime_stream.write(self._chime_audio[:int(CHIME_SAMPLE_RATE * 0.25)])
            except Exception as e:
                logger.debug("Chime playback failed: %s", e)
                chime_stream = None

        # 2. Speaker ID from wake word audio
        ww_speaker_name = None
        ww_embedding = None
        if (self.speaker_id is not None
                and ww_audio is not None
                and len(ww_audio) >= PIPELINE_SAMPLE_RATE):
            ww_speaker_name, _ = self.speaker_id.identify(ww_audio)
            ww_embedding = self.speaker_id.extract_embedding(ww_audio)

        # 3. LISTENING — record utterance (with Smart Turn)
        self.cb.on_state_changed(State.LISTENING)
        # Drain any chunks that piled up between wake-word detection and now.
        self._drain_mic_q()
        pcm_data = record_utterance(
            self.vad, self.mic_device, self.mic_native_rate,
            self.needs_resample, self.mic_vad_blocksize,
            self.silence_frames_threshold,
            stop_event=self.stop_event,
            reader_factory=self._local_reader_factory(self.mic_vad_blocksize),
            smart_turn=self.smart_turn,
        )
        if pcm_data is None or len(pcm_data) < MIN_UTTERANCE_SAMPLES:
            if chime_stream:
                chime_stream.stop(); chime_stream.close()
            return

        # Stop chime before processing
        if chime_stream:
            chime_stream.stop(); chime_stream.close()

        # 4. PROCESSING — STT + diarization + speaker ID
        self.cb.on_state_changed(State.PROCESSING)

        # Phase 4: Diarization (if enabled and audio long enough)
        diarized_transcript = None
        if self.diarization is not None and len(pcm_data) >= PIPELINE_SAMPLE_RATE:
            try:
                segments = self.diarization.diarize_with_speaker_id(
                    pcm_data, self.speaker_id)
                if segments and len(segments) > 1:
                    # Multi-speaker: build diarized transcript
                    diarized_parts = []
                    for seg in segments:
                        start_s = int(seg["start"] * PIPELINE_SAMPLE_RATE)
                        end_s = int(seg["end"] * PIPELINE_SAMPLE_RATE)
                        seg_audio = pcm_data[start_s:end_s]
                        if len(seg_audio) >= MIN_UTTERANCE_SAMPLES:
                            seg_text = self.stt.transcribe(seg_audio)
                            if seg_text.strip():
                                diarized_parts.append(f"[{seg['speaker']}]: {seg_text}")
                    if diarized_parts:
                        diarized_transcript = "\n".join(diarized_parts)
            except Exception as e:
                logger.warning("Diarization failed: %s", e)

        # Standard STT (always run for single-speaker or fallback)
        transcript = self.stt.transcribe(pcm_data)
        if not transcript.strip():
            return

        # Use diarized transcript if available
        if diarized_transcript:
            transcript = diarized_transcript

        utt_speaker = "Unknown"
        utt_embedding = None
        utt_confidence = 0.0
        if self.speaker_id is not None:
            utt_speaker, utt_confidence = self.speaker_id.identify(pcm_data)
            utt_embedding = self.speaker_id.extract_embedding(pcm_data)
            # Phase 5: Track uncertain identifications
            if self.speaker_id.threshold * 0.67 < utt_confidence < self.speaker_id.threshold:
                self.record_uncertain_speaker(
                    transcript, utt_speaker, utt_confidence,
                    len(pcm_data) / PIPELINE_SAMPLE_RATE)

        # 5. Store utterance for ASR correction API
        import datetime as _dt
        with self._last_utterance_lock:
            self._last_utterance = {
                "timestamp": time.time(),
                "timestamp_iso": _dt.datetime.now().isoformat(),
                "raw_transcript": transcript,
                "speaker": utt_speaker,
                "speaker_confidence": utt_confidence,
                "audio_int16": pcm_data,
                "duration_s": len(pcm_data) / PIPELINE_SAMPLE_RATE,
            }

        self.cb.on_transcript(transcript, utt_speaker, False)

        # 6. Active listening loop (conversational continuity)
        current_speaker = utt_speaker
        current_embedding = utt_embedding

        while not self.stop_event.is_set() and self.voice_enabled.is_set():
            self.cb.on_continuity_status("Listening for follow-up...")

            self._drain_mic_q()
            follow_audio = active_listen(
                self.vad, self.mic_device, self.mic_native_rate,
                self.needs_resample, self.mic_vad_blocksize,
                self.listen_window_s,
                self.silence_frames_threshold,
                stop_event=self.stop_event,
                reader_factory=self._local_reader_factory(self.mic_vad_blocksize),
                smart_turn=self.smart_turn,
            )

            if follow_audio is None:
                self.cb.on_continuity_status("No follow-up, returning to wake word")
                break

            if len(follow_audio) < MIN_UTTERANCE_SAMPLES:
                break

            # STT + ASR correction
            self.cb.on_state_changed(State.PROCESSING)
            follow_text = self.stt.transcribe(follow_audio)
            if not follow_text.strip():
                break
            # Speaker check
            same_speaker = True
            if self.speaker_id is not None and current_embedding is not None:
                follow_name, _ = self.speaker_id.identify(follow_audio)
                follow_embedding = self.speaker_id.extract_embedding(follow_audio)
                cosine_sim = float(np.dot(current_embedding, follow_embedding))
                same_speaker = (
                    follow_name == current_speaker
                    or cosine_sim > self.speaker_threshold
                )

                if not same_speaker:
                    self.cb.on_continuity_status(
                        "Different speaker, returning to wake word"
                    )
                    break

            # Store follow-up utterance
            with self._last_utterance_lock:
                self._last_utterance = {
                    "timestamp": time.time(),
                    "timestamp_iso": _dt.datetime.now().isoformat(),
                    "raw_transcript": follow_text,
                    "speaker": current_speaker,
                    "audio_int16": follow_audio,
                    "duration_s": len(follow_audio) / PIPELINE_SAMPLE_RATE,
                }

            # Accept as continuation
            self.cb.on_transcript(follow_text, current_speaker, True)

        # 7. Conversation mode — after Lloyd finishes speaking, give the
        # same speaker a window to follow up without saying the wake word.
        if (self._conversation_mode_enabled
                and current_embedding is not None
                and self.input_mode == "local"
                and not self.stop_event.is_set()
                and self.voice_enabled.is_set()):
            self._conversation_loop(current_embedding, current_speaker)

    def _conversation_loop(self, anchor_embedding, anchor_speaker: str) -> None:
        """Post-TTS speaker-locked follow-up listener.

        Loops:
          1. Wait (bounded) for TTS playback to start, then for it to end.
             If TTS never starts (e.g. backend errored, TTS disabled), times
             out and returns to wake-word.
          2. Drain the shared mic queue (TTS bleed).
          3. active_listen() with `conversation_window_s` — if the same
             speaker speaks, transcribe + inject and loop back to step 1
             for the next response. If silent or a different voice, exit.
        """
        import datetime as _dt

        for turn in range(self._conversation_max_turns):
            if self.stop_event.is_set() or not self.voice_enabled.is_set():
                return

            # Wait for TTS to start (Lloyd is generating + summarizing).
            wait_deadline = time.monotonic() + self._conversation_tts_wait_s
            while not self._tts_playing.is_set():
                if self.stop_event.is_set() or not self.voice_enabled.is_set():
                    return
                if time.monotonic() > wait_deadline:
                    return  # no TTS this turn — fall through to wake word
                time.sleep(0.1)

            # Wait for TTS to finish.
            while self._tts_playing.is_set():
                if self.stop_event.is_set() or not self.voice_enabled.is_set():
                    return
                time.sleep(0.1)

            # Discard chunks captured during TTS (bleed-through, paplay echoes).
            self._drain_mic_q()

            self.cb.on_continuity_status(
                f"Conversation mode (window {self._conversation_window_s:.0f}s)"
            )
            # Show LISTENING in the navbar while the conversation window is
            # open — IDLE here would make it look like wake-word mode and is
            # the wrong signal for "still actively listening to you".
            self.cb.on_state_changed(State.LISTENING)

            follow_audio = active_listen(
                self.vad, self.mic_device, self.mic_native_rate,
                self.needs_resample, self.mic_vad_blocksize,
                self._conversation_window_s,
                self.silence_frames_threshold,
                stop_event=self.stop_event,
                reader_factory=self._local_reader_factory(self.mic_vad_blocksize),
                smart_turn=self.smart_turn,
            )

            if follow_audio is None or len(follow_audio) < MIN_UTTERANCE_SAMPLES:
                self.cb.on_continuity_status("Conversation window expired")
                return

            self.cb.on_state_changed(State.PROCESSING)
            follow_text = self.stt.transcribe(follow_audio)
            if not follow_text.strip():
                return

            # Speaker match against the anchor embedding (the speaker who
            # triggered the wake word). Cosine similarity threshold mirrors
            # the standard speaker_threshold from the ID layer.
            if self.speaker_id is not None:
                follow_embedding = self.speaker_id.extract_embedding(follow_audio)
                sim = float(np.dot(anchor_embedding, follow_embedding))
                print(f"  [conversation] speaker_sim={sim:.2f} thr={self.speaker_threshold:.2f}", flush=True)
                if sim < self.speaker_threshold:
                    self.cb.on_continuity_status(
                        f"Different speaker (sim={sim:.2f}); exiting conversation mode"
                    )
                    return

            # Same speaker — accept as a new turn. Persist + emit; the
            # callback handler injects to the backend like any utterance.
            with self._last_utterance_lock:
                self._last_utterance = {
                    "timestamp": time.time(),
                    "timestamp_iso": _dt.datetime.now().isoformat(),
                    "raw_transcript": follow_text,
                    "speaker": anchor_speaker,
                    "audio_int16": follow_audio,
                    "duration_s": len(follow_audio) / PIPELINE_SAMPLE_RATE,
                }
            self.cb.on_transcript(follow_text, anchor_speaker, True)
            # Loop back: wait for Lloyd's next response, then listen again.

        self.cb.on_continuity_status(
            f"Conversation max turns ({self._conversation_max_turns}) reached"
        )

    # --- WebSocket pipeline: per-client thread management ---

    def _client_pipeline_thread(self, client_id: str):
        """Per-client pipeline thread."""
        try:
            self._pipeline_iteration_websocket(client_id)
        except Exception as e:
            self.cb.on_error(f"Client {client_id[:8]}: {e}")
        finally:
            with self._client_threads_lock:
                self._client_threads.pop(client_id, None)

    def _client_alive(self, client_id: str) -> bool:
        """Check if a client is still connected."""
        ws = self._ws_server
        return ws is not None and client_id in ws.get_client_ids()

    # --- WebSocket pipeline iteration (per-client) ---

    def _pipeline_iteration_websocket(self, client_id: str) -> None:
        """One full wake-word -> conversation -> return-to-idle cycle over WebSocket.

        Runs in a per-client thread. Each client gets its own WakeWordDetector
        and VAD instances. STT is shared with a lock.
        """
        ws = self._ws_server

        # Per-client wake word and VAD instances (not thread-safe for concurrent use)
        config = self.config
        wake_word = WakeWordDetector(
            model_names=config.get("wake_word_models", []),
            threshold=config.get("wake_word_threshold", 0.5),
            smoothing_window=config.get("wake_word_smoothing_window", 1),
            min_hits=config.get("wake_word_min_hits", 1),
        )
        vad = VAD()

        # Wait for "start" control message
        self.cb.on_state_changed(State.IDLE)
        ws.send_message({"type": "state", "state": "IDLE"}, client_id)

        while not self.stop_event.is_set() and self._client_alive(client_id):
            ctrl = ws.get_control(client_id)
            if ctrl and ctrl.get("type") == "start":
                # Update session key if provided in start command
                if ctrl.get("sessionKey"):
                    with ws._clients_lock:
                        if client_id in ws._clients:
                            ws._clients[client_id].session_key = ctrl["sessionKey"]
                break
            time.sleep(0.05)
        else:
            if self.stop_event.is_set():
                return
            # Client disconnected before sending start
            return

        # Push-to-talk loop: wait for "start" -> listen -> process -> wait for next "start"
        # No wake word in websocket mode — browser mic button is the trigger
        while not self.stop_event.is_set() and self._client_alive(client_id) and self.voice_enabled.is_set():
            # No wake word audio in push-to-talk mode
            ww_audio = None
            ww_speaker_name = None
            ww_embedding = None

            # 1. LISTENING — record utterance (go straight to recording)
            self.cb.on_state_changed(State.LISTENING)
            ws.send_message({"type": "state", "state": "LISTENING"}, client_id)

            pcm_data = record_utterance(
                vad, self.mic_device, self.mic_native_rate,
                False, VAD_FRAME_SIZE,
                self.silence_frames_threshold,
                stop_event=self.stop_event,
                reader_factory=lambda: ws.make_reader(client_id, VAD_FRAME_SIZE),
                smart_turn=self.smart_turn,
            )
            if pcm_data is None or len(pcm_data) < MIN_UTTERANCE_SAMPLES:
                # No valid audio — go back to IDLE and wait for next "start"
                break
            if not self._client_alive(client_id):
                return

            # 2. PROCESSING — STT + Speaker ID
            self.cb.on_state_changed(State.PROCESSING)
            ws.send_message({"type": "state", "state": "PROCESSING"}, client_id)

            with self._stt_lock:
                transcript = self.stt.transcribe(pcm_data)
            if not transcript.strip():
                break

            utt_speaker = "Unknown"
            utt_embedding = None
            if self.speaker_id is not None:
                with self._speaker_lock:
                    utt_speaker, _ = self.speaker_id.identify(pcm_data)
                    utt_embedding = self.speaker_id.extract_embedding(pcm_data)

            # 3. Store utterance + send transcript
            import datetime as _dt
            with self._last_utterance_lock:
                self._last_utterance = {
                    "timestamp": time.time(),
                    "timestamp_iso": _dt.datetime.now().isoformat(),
                    "raw_transcript": transcript,
                    "speaker": utt_speaker,
                    "audio_int16": pcm_data,
                    "duration_s": len(pcm_data) / PIPELINE_SAMPLE_RATE,
                }

            self.cb.on_transcript(transcript, utt_speaker, False, client_id=client_id)
            ws.send_message({
                "type": "transcript",
                "text": transcript,
                "speaker": utt_speaker,
                "is_continuity": False,
            }, client_id)

            # 4. Wait for TTS to start, then wait for it to finish
            current_speaker = utt_speaker
            current_embedding = utt_embedding

            # Wait up to 15s for TTS to begin
            tts_wait_deadline = time.time() + 15.0
            while not self._tts_playing.is_set() and not self.stop_event.is_set():
                if not self._client_alive(client_id):
                    return
                if time.time() > tts_wait_deadline:
                    break
                time.sleep(0.1)

            # TTS is playing — wait for it to finish
            while self._tts_playing.is_set() and not self.stop_event.is_set():
                if not self._client_alive(client_id):
                    return
                time.sleep(0.1)

            if self.stop_event.is_set():
                return

            # 5. Active listening loop with speaker continuity
            followup_count = 0
            while not self.stop_event.is_set() and self._client_alive(client_id) and self.voice_enabled.is_set():
                if followup_count >= self.max_followup_turns:
                    self.cb.on_continuity_status("Max follow-ups reached, returning to idle")
                    break
                ctrl = ws.get_control(client_id)
                if ctrl and ctrl.get("type") == "stop":
                    return

                ws.send_message({"type": "state", "state": "ACTIVE_LISTEN"}, client_id)
                self.cb.on_continuity_status("Listening for follow-up...")

                follow_audio = active_listen(
                    vad, self.mic_device, self.mic_native_rate,
                    False, VAD_FRAME_SIZE,
                    self.listen_window_s,
                    self.silence_frames_threshold,
                    stop_event=self.stop_event,
                    reader_factory=lambda: ws.make_reader(client_id, VAD_FRAME_SIZE),
                    smart_turn=self.smart_turn,
                )

                if follow_audio is None:
                    self.cb.on_continuity_status("No follow-up, returning to idle")
                    break

                if len(follow_audio) < MIN_UTTERANCE_SAMPLES:
                    break

                # STT
                self.cb.on_state_changed(State.PROCESSING)
                ws.send_message({"type": "state", "state": "PROCESSING"}, client_id)

                with self._stt_lock:
                    follow_text = self.stt.transcribe(follow_audio)
                if not follow_text.strip():
                    break

                # Speaker continuity check
                same_speaker = True
                if self.speaker_id is not None and current_embedding is not None:
                    with self._speaker_lock:
                        follow_name, _ = self.speaker_id.identify(follow_audio)
                        follow_embedding = self.speaker_id.extract_embedding(follow_audio)
                    cosine_sim = float(np.dot(current_embedding, follow_embedding))
                    same_speaker = (
                        follow_name == current_speaker
                        or cosine_sim > self.speaker_threshold
                    )

                    if not same_speaker:
                        self.cb.on_continuity_status("Different speaker, returning to idle")
                        break

                # Store follow-up
                with self._last_utterance_lock:
                    self._last_utterance = {
                        "timestamp": time.time(),
                        "timestamp_iso": _dt.datetime.now().isoformat(),
                        "raw_transcript": follow_text,
                        "speaker": current_speaker,
                        "audio_int16": follow_audio,
                        "duration_s": len(follow_audio) / PIPELINE_SAMPLE_RATE,
                    }

                self.cb.on_transcript(follow_text, current_speaker, True, client_id=client_id)
                ws.send_message({
                    "type": "transcript",
                    "text": follow_text,
                    "speaker": current_speaker,
                    "is_continuity": True,
                }, client_id)

                # Wait for TTS to finish before listening for next follow-up
                tts_wait_deadline = time.time() + 15.0
                while not self._tts_playing.is_set() and not self.stop_event.is_set():
                    if not self._client_alive(client_id):
                        return
                    if time.time() > tts_wait_deadline:
                        break
                    time.sleep(0.1)

                while self._tts_playing.is_set() and not self.stop_event.is_set():
                    if not self._client_alive(client_id):
                        return
                    time.sleep(0.1)

                followup_count += 1

            # Turn complete — return to IDLE and wait for next "start" command
            self.cb.on_state_changed(State.IDLE)
            ws.send_message({"type": "state", "state": "IDLE"}, client_id)

            # Wait for next "start" push-to-talk trigger (or "stop" to disconnect)
            while not self.stop_event.is_set() and self._client_alive(client_id):
                ctrl = ws.get_control(client_id)
                if ctrl and ctrl.get("type") == "stop":
                    return
                if ctrl and ctrl.get("type") == "start":
                    if ctrl.get("sessionKey"):
                        with ws._clients_lock:
                            if client_id in ws._clients:
                                ws._clients[client_id].session_key = ctrl["sessionKey"]
                    break  # back to top of push-to-talk loop
                time.sleep(0.05)

    # --- Discord pipeline iteration ---

    def _pipeline_iteration_discord(self) -> None:
        """One full wake-word -> conversation -> return-to-idle cycle (Discord input).

        Mirrors _pipeline_iteration() but reads audio from the Discord bridge
        HTTP endpoint instead of the local microphone. The Node.js bridge sends
        16 kHz mono PCM so no resampling is needed.
        """
        factory = self._discord_reader_factory
        if not factory:
            time.sleep(1)
            return

        # 1. IDLE — wait for wake word
        self.cb.on_state_changed(State.IDLE)
        ww_audio = wait_for_wake_word_with_buffer(
            self.wake_word, 0, PIPELINE_SAMPLE_RATE,
            False, WW_FRAME_SIZE,
            stop_event=self.stop_event,
            voice_enabled=self.voice_enabled,
            reader_factory=lambda: factory(WW_FRAME_SIZE),
        )
        if ww_audio is None or self.stop_event.is_set():
            return
        if not self.voice_enabled.is_set():
            return

        # 2. Speaker ID from wake word audio
        ww_speaker_name = None
        ww_embedding = None
        if (self.speaker_id is not None
                and ww_audio is not None
                and len(ww_audio) >= PIPELINE_SAMPLE_RATE):
            ww_speaker_name, _ = self.speaker_id.identify(ww_audio)
            ww_embedding = self.speaker_id.extract_embedding(ww_audio)

        # 3. LISTENING — record utterance (with Smart Turn)
        self.cb.on_state_changed(State.LISTENING)
        pcm_data = record_utterance(
            self.vad, 0, PIPELINE_SAMPLE_RATE,
            False, VAD_FRAME_SIZE,
            self.silence_frames_threshold,
            stop_event=self.stop_event,
            reader_factory=lambda: factory(VAD_FRAME_SIZE),
            smart_turn=self.smart_turn,
        )
        if pcm_data is None or len(pcm_data) < MIN_UTTERANCE_SAMPLES:
            return

        # 4. PROCESSING — STT + diarization + Speaker ID
        self.cb.on_state_changed(State.PROCESSING)

        # Phase 4: Diarization
        diarized_transcript = None
        if self.diarization is not None and len(pcm_data) >= PIPELINE_SAMPLE_RATE:
            try:
                segments = self.diarization.diarize_with_speaker_id(
                    pcm_data, self.speaker_id)
                if segments and len(segments) > 1:
                    diarized_parts = []
                    for seg in segments:
                        start_s = int(seg["start"] * PIPELINE_SAMPLE_RATE)
                        end_s = int(seg["end"] * PIPELINE_SAMPLE_RATE)
                        seg_audio = pcm_data[start_s:end_s]
                        if len(seg_audio) >= MIN_UTTERANCE_SAMPLES:
                            seg_text = self.stt.transcribe(seg_audio)
                            if seg_text.strip():
                                diarized_parts.append(f"[{seg['speaker']}]: {seg_text}")
                    if diarized_parts:
                        diarized_transcript = "\n".join(diarized_parts)
            except Exception as e:
                logger.warning("Diarization failed: %s", e)

        transcript = self.stt.transcribe(pcm_data)
        if not transcript.strip():
            return
        if diarized_transcript:
            transcript = diarized_transcript

        utt_speaker = ww_speaker_name or "Unknown"
        utt_confidence = 0.0
        if self.speaker_id is not None:
            utt_speaker, utt_confidence = self.speaker_id.identify(pcm_data)
            if self.speaker_id.threshold * 0.67 < utt_confidence < self.speaker_id.threshold:
                self.record_uncertain_speaker(
                    transcript, utt_speaker, utt_confidence,
                    len(pcm_data) / PIPELINE_SAMPLE_RATE)

        # 5. Store + emit transcript
        import datetime as _dt
        with self._last_utterance_lock:
            self._last_utterance = {
                "timestamp": time.time(),
                "timestamp_iso": _dt.datetime.now().isoformat(),
                "raw_transcript": transcript,
                "speaker": utt_speaker,
                "speaker_confidence": utt_confidence,
                "audio_int16": pcm_data,
                "duration_s": len(pcm_data) / PIPELINE_SAMPLE_RATE,
            }
        self.cb.on_transcript(transcript, utt_speaker, False)

        # 6. Active listening loop (conversational continuity)
        current_speaker = utt_speaker
        current_embedding = ww_embedding
        followup_count = 0

        while not self.stop_event.is_set() and self.voice_enabled.is_set():
            if followup_count >= self.max_followup_turns:
                self.cb.on_continuity_status("Max follow-ups reached, returning to wake word")
                break
            self.cb.on_continuity_status("Listening for follow-up...")

            follow_audio = active_listen(
                self.vad, 0, PIPELINE_SAMPLE_RATE,
                False, VAD_FRAME_SIZE,
                self.listen_window_s,
                self.silence_frames_threshold,
                stop_event=self.stop_event,
                reader_factory=lambda: factory(VAD_FRAME_SIZE),
                smart_turn=self.smart_turn,
            )

            if follow_audio is None:
                self.cb.on_continuity_status("No follow-up, returning to wake word")
                break

            if len(follow_audio) < MIN_UTTERANCE_SAMPLES:
                break

            # STT
            self.cb.on_state_changed(State.PROCESSING)
            follow_text = self.stt.transcribe(follow_audio)
            if not follow_text.strip():
                break

            # Speaker check
            same_speaker = True
            if self.speaker_id is not None and current_embedding is not None:
                follow_name, _ = self.speaker_id.identify(follow_audio)
                follow_embedding = self.speaker_id.extract_embedding(follow_audio)
                cosine_sim = float(np.dot(current_embedding, follow_embedding))
                same_speaker = (
                    follow_name == current_speaker
                    or cosine_sim > self.speaker_threshold
                )

                if not same_speaker:
                    self.cb.on_continuity_status(
                        "Different speaker, returning to wake word"
                    )
                    break

            # Store follow-up utterance
            with self._last_utterance_lock:
                self._last_utterance = {
                    "timestamp": time.time(),
                    "timestamp_iso": _dt.datetime.now().isoformat(),
                    "raw_transcript": follow_text,
                    "speaker": current_speaker,
                    "audio_int16": follow_audio,
                    "duration_s": len(follow_audio) / PIPELINE_SAMPLE_RATE,
                }

            # Accept as continuation
            self.cb.on_transcript(follow_text, current_speaker, True)
            followup_count += 1

    # --- ASR correction ---

    def get_last_utterance(self) -> dict | None:
        """Return a copy of the last utterance (without audio data)."""
        with self._last_utterance_lock:
            if not self._last_utterance:
                return None
            utt = {k: v for k, v in self._last_utterance.items()
                   if k != "audio_int16"}
            return utt

    def correct_transcript(self, corrected: str) -> dict | None:
        """Store a corrected transcript for the last utterance.

        Also adds new words from the correction to the ASR corrector vocabulary
        so future transcripts benefit from the correction.

        Returns dict with original and corrected text, or None if no utterance.
        """
        with self._last_utterance_lock:
            if not self._last_utterance:
                return None
            original = self._last_utterance.get("raw_transcript", "")
            self._last_utterance["corrected_transcript"] = corrected

        return {"original": original, "corrected": corrected}
