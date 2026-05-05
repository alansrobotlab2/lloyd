#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp[cli]",
#   "httpx",
#   "requests",
#   "uvicorn",
#   "starlette",
#   "sse-starlette",
# ]
# ///
"""
Lloyd Voice Tools — MCP Server

Exposes Lloyd's voice pipeline capabilities as MCP tools over stdio transport.
Proxies to the voice bridge HTTP API (default port 8092).

Capabilities:
  - Wakeword detection (status, threshold tuning)
  - VAD / speech recording status
  - Speaker diarization & identification
  - ASR (last utterance transcript)
  - ASR cleaning (LLM-powered transcript correction with domain vocabulary)
  - TTS (text-to-speech playback)
  - ASR correction (post-transcription fixes)

Run with:
  uv run voice_mcp_server.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import httpx
import requests
from mcp.server.fastmcp import FastMCP

VOICE_API_URL = os.environ.get("LLOYD_VOICE_API_URL", "http://127.0.0.1:8092")
VLLM_URL = os.environ.get("LLOYD_VLLM_URL", "http://127.0.0.1:8091/v1/chat/completions")
VLLM_MODEL = os.environ.get("LLOYD_VLLM_MODEL", "Qwen3.5-35B-A3B")
ASR_VOCAB_PATH = os.environ.get("LLOYD_ASR_VOCAB_PATH", "~/obsidian/tags.md")

mcp = FastMCP("lloyd-voice")


def _api_get(path: str) -> dict:
    resp = httpx.get(f"{VOICE_API_URL}{path}", timeout=10.0)
    return resp.json()


def _api_post(path: str, body: dict) -> dict:
    resp = httpx.post(f"{VOICE_API_URL}{path}", json=body, timeout=60.0)
    return resp.json()


def _connect_error_msg() -> str:
    return "Error: voice bridge not running (cannot reach HTTP API on port 8092)"


# ---------------------------------------------------------------------------
# Status & Pipeline State
# ---------------------------------------------------------------------------

@mcp.tool()
def voice_status() -> str:
    """Get the current voice pipeline status.

    Returns the pipeline state (IDLE/LISTENING/PROCESSING/SPEAKING),
    whether voice is enabled, the last transcript heard, identified speakers,
    and the timestamp of the last utterance.

    Use this to check if Lloyd is listening, speaking, or processing.
    """
    try:
        data = _api_get("/v1/status")
        if "error" in data:
            return f"Error: {data['error']}"
        lines = [
            f"State: {data.get('state', 'unknown')}",
            f"Voice enabled: {data.get('voice_enabled', 'unknown')}",
        ]
        transcript = data.get("last_transcript", "")
        if transcript:
            lines.append(f"Last transcript: \"{transcript}\"")
            speakers = data.get("last_speaker_ids", [])
            if speakers:
                lines.append(f"Speakers: {', '.join(speakers)}")
        else:
            lines.append("No utterance recorded yet.")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _connect_error_msg()
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# ASR — Speech Recognition
# ---------------------------------------------------------------------------

@mcp.tool()
def voice_last_utterance() -> str:
    """Get the last voice utterance heard by Lloyd.

    Returns the transcript with speaker identification (e.g. "[Alan]: hello"),
    raw transcript (pre-diarization), any corrected transcript, timestamp,
    duration, and speaker segments from diarization.

    This is the primary tool for seeing what was just said and who said it.
    """
    try:
        data = _api_get("/v1/last_utterance")
        if "error" in data:
            return f"Error: {data['error']}"
        return json.dumps(data, indent=2, ensure_ascii=False)
    except httpx.ConnectError:
        return _connect_error_msg()
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# ASR Correction
# ---------------------------------------------------------------------------

@mcp.tool()
def voice_correct_transcript(corrected: str) -> str:
    """Submit a corrected version of the last utterance's transcript.

    Use this when you notice the ASR (Moonshine STT) made an error —
    e.g. misheard a name, technical term, or homophone. The correction
    is stored alongside the original so future references use the
    corrected text.

    Args:
        corrected: The corrected transcript text.
    """
    try:
        data = _api_post("/v1/correct_transcript", {"corrected": corrected})
        if "error" in data:
            return f"Error: {data['error']}"
        return (
            f"Transcript corrected.\n"
            f"Original: \"{data.get('original', '')}\"\n"
            f"Corrected: \"{data.get('corrected', '')}\""
        )
    except httpx.ConnectError:
        return _connect_error_msg()
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# ASR Cleaner — LLM-powered transcript correction
# ---------------------------------------------------------------------------

_ASR_CLEANER_SYSTEM_PROMPT = (
    "You are an ASR post-processor. You receive a raw speech-to-text "
    "transcript and a list of known domain terms. Your job:\n"
    "1. Fix misheard words using the domain terms as reference "
    "(spelling, word-boundary errors, homophones).\n"
    "2. Add proper punctuation.\n"
    "3. Fix capitalisation (sentence case, proper nouns, acronyms).\n"
    "Only output the corrected transcript — nothing else. "
    "If no corrections are needed, return the original text unchanged. "
    "Do not rephrase, do not explain, do not add commentary."
)


def _load_vocab(path: str) -> list[str]:
    """Load domain vocabulary from a tags file (one #tag per line)."""
    p = Path(path).expanduser()
    if not p.exists():
        return []
    vocab: list[str] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") and len(line) > 1:
            vocab.append(line[1:])
    return vocab


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


@mcp.tool()
def asr_cleaner(transcript: str) -> str:
    """Clean and correct a raw ASR transcript using a local LLM.

    Takes a raw speech-to-text transcript and corrects it for:
    - Misheard domain-specific terms (matched against a known vocabulary
      sourced from Obsidian tags)
    - Punctuation (adds periods, commas, question marks, etc.)
    - Capitalisation (sentence case, proper nouns, technical terms)

    The correction uses a local vLLM endpoint with domain vocabulary context.

    Args:
        transcript: The raw ASR transcript to clean and correct.

    Returns:
        The corrected transcript with proper punctuation and capitalisation.
    """
    if not transcript.strip():
        return transcript

    vocab = _load_vocab(ASR_VOCAB_PATH)
    vocab_str = ", ".join(vocab) if vocab else "(no domain terms loaded)"

    prompt = (
        f"Domain terms: {vocab_str}\n\n"
        f"Raw transcript: {transcript}"
    )

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": _ASR_CLEANER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(
            VLLM_URL, json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        corrected = data["choices"][0]["message"]["content"].strip()
        corrected = _strip_think_tags(corrected)
        return corrected or transcript
    except requests.ConnectionError:
        return f"Error: cannot reach vLLM at {VLLM_URL}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Speaker Diarization & Identification
# ---------------------------------------------------------------------------

@mcp.tool()
def voice_enroll_speaker(name: str, speaker_label: str = "") -> str:
    """Enroll a speaker using audio from the last utterance.

    Creates a voice profile (Resemblyzer GE2E embedding) so Lloyd can
    identify this person by name in future utterances. The last utterance
    must contain at least 1 second of clear speech.

    Args:
        name: Name to assign to the speaker (e.g. "Alan", "Sarah").
              Overwrites any existing profile with the same name.
        speaker_label: Optional diarization label (e.g. "SPEAKER_00") to
                       isolate one speaker from a multi-speaker utterance.
                       If omitted, uses the full utterance audio.

    Takes effect immediately — the next utterance will attempt to identify
    the speaker by this name.
    """
    try:
        body: dict = {"name": name}
        if speaker_label:
            body["speaker_label"] = speaker_label
        data = _api_post("/v1/enroll_speaker", body)
        if "error" in data:
            return f"Error: {data['error']}"
        return (
            f"Enrolled '{data['enrolled']}' successfully.\n"
            f"Audio used: {data['audio_duration_s']:.1f}s\n"
            f"Profile: {data['profile_path']}"
        )
    except httpx.ConnectError:
        return _connect_error_msg()
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def voice_list_speakers() -> str:
    """List all enrolled speaker profiles.

    Shows the names of all speakers that Lloyd can identify by voice,
    along with their embedding dimensions. Uses Resemblyzer (GE2E model)
    for speaker verification.
    """
    try:
        data = _api_get("/v1/speakers")
        if "error" in data:
            return f"Error: {data['error']}"
        profiles = data.get("profiles", [])
        if not profiles:
            return (
                "No speakers enrolled.\n"
                f"Profiles directory: {data.get('profiles_dir', 'speakers/')}"
            )
        lines = [f"Enrolled speakers ({len(profiles)}):"]
        for p in profiles:
            lines.append(f"  - {p['name']} ({p['embedding_dim']}d embedding)")
        lines.append(f"\nProfiles directory: {data.get('profiles_dir', 'speakers/')}")
        return "\n".join(lines)
    except httpx.ConnectError:
        return _connect_error_msg()
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# TTS — Text-to-Speech
# ---------------------------------------------------------------------------

@mcp.tool()
def voice_say(text: str) -> str:
    """Synthesize text and play it through the speakers.

    Uses the currently active TTS engine (CosyVoice3, Qwen3-TTS, or Orpheus)
    to speak the given text aloud. The audio is played immediately through
    the default output device.

    Args:
        text: The text to speak aloud. Keep it concise for natural speech.
    """
    try:
        data = _api_post("/v1/say", {"text": text})
        if "error" in data:
            return f"Error: {data['error']}"
        return (
            f"Spoken: \"{data.get('text', text)}\"\n"
            f"Duration: {data.get('duration_s', 0):.1f}s "
            f"(synthesized in {data.get('elapsed_s', 0):.1f}s)"
        )
    except httpx.ConnectError:
        return _connect_error_msg()
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Voice Pipeline Control
# ---------------------------------------------------------------------------

@mcp.tool()
def voice_toggle() -> str:
    """Toggle the voice pipeline on or off.

    When disabled, Lloyd stops listening for the wake word and ignores
    all microphone input. When re-enabled, listening resumes immediately.

    Returns the new state (enabled or disabled).
    """
    try:
        data = _api_post("/v1/voice/toggle", {})
        if "error" in data:
            return f"Error: {data['error']}"
        enabled = data.get("voice_enabled", False)
        return f"Voice pipeline {'enabled' if enabled else 'disabled'}."
    except httpx.ConnectError:
        return _connect_error_msg()
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Wakeword & VAD Configuration
# ---------------------------------------------------------------------------

@mcp.tool()
def voice_get_config() -> str:
    """Get the current voice pipeline configuration.

    Returns all config values including wake word models and thresholds,
    VAD settings, silence duration, diarization state, TTS sample rate,
    and runtime status. Sensitive fields (tokens) are excluded.

    Use this to inspect current settings before making changes.
    """
    try:
        data = _api_get("/v1/config")
        if "error" in data:
            return f"Error: {data['error']}"
        return json.dumps(data, indent=2, ensure_ascii=False)
    except httpx.ConnectError:
        return _connect_error_msg()
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def voice_set_config(
    wake_word_threshold: float | None = None,
    silence_duration_ms: int | None = None,
    wake_word_smoothing_window: int | None = None,
    wake_word_min_hits: int | None = None,
    speaker_id_threshold: float | None = None,
) -> str:
    """Update voice pipeline settings at runtime (no restart needed).

    All parameters are optional — only provide the ones you want to change.

    Args:
        wake_word_threshold: Detection threshold for wake word (0.0-1.0).
                             Lower = more sensitive, higher = fewer false positives.
                             Current default: 0.10.
        silence_duration_ms: Milliseconds of silence before end-of-speech.
                             Higher = waits longer for pauses. Default: 1000.
        wake_word_smoothing_window: Number of frames in the score smoothing
                                    window. Default: 3.
        wake_word_min_hits: Minimum frames above threshold within the
                            smoothing window to trigger. Default: 2.
        speaker_id_threshold: Cosine similarity threshold for speaker
                              identification (0.0-1.0). Default: 0.75.
    """
    body = {}
    if wake_word_threshold is not None:
        body["wake_word_threshold"] = wake_word_threshold
    if silence_duration_ms is not None:
        body["silence_duration_ms"] = silence_duration_ms
    if wake_word_smoothing_window is not None:
        body["wake_word_smoothing_window"] = wake_word_smoothing_window
    if wake_word_min_hits is not None:
        body["wake_word_min_hits"] = wake_word_min_hits
    if speaker_id_threshold is not None:
        body["speaker_id_threshold"] = speaker_id_threshold

    if not body:
        return "No config values provided. Pass at least one parameter to update."

    try:
        data = _api_post("/v1/config", body)
        if "error" in data:
            return f"Error: {data['error']}"
        lines = []
        updated = data.get("updated", {})
        if updated:
            lines.append("Updated:")
            for k, v in updated.items():
                lines.append(f"  {k} = {v}")
        rejected = data.get("rejected", {})
        if rejected:
            lines.append("Rejected:")
            for k, reason in rejected.items():
                lines.append(f"  {k}: {reason}")
        return "\n".join(lines) if lines else "No changes applied."
    except httpx.ConnectError:
        return _connect_error_msg()
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lloyd Voice MCP Server")
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio", "sse"],
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE bind host")
    parser.add_argument("--port", type=int, default=8094, help="SSE bind port")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port

    print(
        f"Lloyd Voice MCP Server started\n"
        f"  Voice API: {VOICE_API_URL}\n"
        f"  Transport: {args.transport}"
        + (f"\n  SSE: {args.host}:{args.port}" if args.transport == "sse" else ""),
        file=sys.stderr,
    )
    try:
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        pass
