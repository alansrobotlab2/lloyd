"""Synchronous calls to the secondary LLM (port 8091) for post-session jobs.

The primary model talks to the user; the secondary model runs on a separate
GPU and handles summary/fact/focus extraction in the background so the
primary generation path stays unblocked.
"""

import json
import logging
import re
import urllib.request
from typing import Optional


logger = logging.getLogger("lloyd-server")

_POST_CAPTURE_MODEL_URL = "http://127.0.0.1:8091/v1/chat/completions"
_POST_CAPTURE_MODEL_NAME = "secondary"


_FACT_EXTRACTION_PROMPT = """\
Analyze this conversation transcript and extract 3-5 durable facts worth \
remembering across sessions. Focus ONLY on:
- User preferences or decisions ("prefers X over Y", "decided to use X")
- System/project state changes ("switched X to Y", "port N now runs Z")
- Technical decisions ("using asyncio for X", "chose library Y because Z")
- Project milestones ("Phase 2 started", "feature X shipped")

Skip: greetings, transient debug output, questions without answers, opinions \
about things outside the user's control.

Return one fact per line, prefixed with the entity name in brackets. Example:
[Lloyd] Vault search added to prefetch pipeline
[Alfie] Switched from ROS1 to ROS2 for motor control
[vLLM] Running on GPU1 (RTX PRO 6000)

If no durable facts exist, return exactly: NONE

Transcript:
"""


def _sync_secondary_capture_call(transcript: str) -> Optional[str]:
    """Call secondary model synchronously for post-session summary extraction."""
    prompt = (
        "Analyze this conversation transcript and produce a concise summary "
        "(2-4 sentences) of what was discussed, decided, or accomplished. "
        "Focus on outcomes: decisions made, problems solved, preferences expressed, "
        "system changes, and action items. If the conversation is trivial "
        "(greetings, small talk, no substantive content), return exactly: TRIVIAL\n\n"
        f"Transcript:\n{transcript}"
    )

    payload = {
        "model": _POST_CAPTURE_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a concise conversation summarizer. Return only the summary text, no JSON or formatting."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 500,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        req = urllib.request.Request(
            _POST_CAPTURE_MODEL_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"secondary capture call failed: {e}")
        return None


def _sync_secondary_fact_extraction(transcript: str) -> list[dict]:
    """Call secondary model to extract durable facts from a session transcript."""
    payload = {
        "model": _POST_CAPTURE_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract durable facts from conversations. Return only fact lines, no commentary."},
            {"role": "user", "content": _FACT_EXTRACTION_PROMPT + transcript},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        req = urllib.request.Request(
            _POST_CAPTURE_MODEL_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()

        if text.upper() == "NONE":
            return []

        facts = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and "]" in line:
                bracket_end = line.index("]")
                entity = line[1:bracket_end].strip()
                fact_text = line[bracket_end + 1:].strip().lstrip("- :")
                if entity and fact_text:
                    facts.append({"entity": entity, "fact": fact_text})
            else:
                fact_text = line.lstrip("- ")
                if fact_text:
                    facts.append({"entity": "Lloyd", "fact": fact_text})
        return facts[:5]

    except Exception as e:
        logger.warning(f"secondary fact extraction call failed: {e}")
        return []


_VOICE_SUMMARY_SYSTEM = (
    "You rewrite Lloyd's text reply for spoken delivery. Preserve Lloyd's "
    "first-person voice. Strip markdown, code blocks, bullet lists, links, "
    "file paths, and stage directions. Keep it conversational and brief "
    "(1-3 sentences). If the reply is already short and conversational, "
    "return it unchanged. Output only the spoken text — no preamble, no "
    "labels, no quotes."
)


# Markers that signal structured / non-conversational content. Anything that
# matches gets routed through the secondary so code, paths, lists, and tables
# can be flattened into something speakable.
_NON_TRIVIAL_PATTERN = re.compile(
    r"```"                       # fenced code block
    r"|`[^`]+`"                  # inline code (paths, snippets)
    r"|^\s*[-*+]\s"              # bullet list
    r"|^\s*\d+\.\s"              # numbered list
    r"|^\s*#{1,6}\s"             # markdown heading
    r"|\|.*\|"                   # table row
    r"|\[[^\]]+\]\([^)]+\)",     # markdown link
    re.MULTILINE,
)

_TRIVIAL_MAX_CHARS = 300


def _is_trivially_speakable(text: str) -> bool:
    """True when `text` is short, plain prose that can be TTS'd as-is.

    The secondary's voice rewrite costs ~0.5-2s per call. For one- or two-
    sentence replies with no markdown/code, the secondary's own prompt just
    echoes the input back anyway, so we skip the round-trip and let the
    LiveKit worker speak the primary text directly."""
    if not text:
        return False
    s = text.strip()
    if len(s) > _TRIVIAL_MAX_CHARS:
        return False
    return _NON_TRIVIAL_PATTERN.search(s) is None


def _sync_secondary_voice_summary(primary_text: str, timeout: float = 15.0) -> Optional[str]:
    """Call secondary to rewrite a primary response for TTS playback.

    Returns the spoken-form text, or None on failure (caller should fall
    back to the original primary text).
    """
    text = (primary_text or "").strip()
    if not text:
        return None

    payload = {
        "model": _POST_CAPTURE_MODEL_NAME,
        "messages": [
            {"role": "system", "content": _VOICE_SUMMARY_SYSTEM},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        req = urllib.request.Request(
            _POST_CAPTURE_MODEL_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip() or None
    except Exception as e:
        logger.warning(f"secondary voice summary failed: {e}")
        return None


def _sync_secondary_focus_extraction(transcript: str) -> list[str]:
    """Call secondary model to extract 3-5 topic phrases from recent conversation."""
    payload = {
        "model": _POST_CAPTURE_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Extract the main topics from this conversation. Return 3-5 short topic phrases (2-4 words each), one per line. No numbering, no explanation."},
            {"role": "user", "content": transcript},
        ],
        "temperature": 0.0,
        "max_tokens": 100,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        req = urllib.request.Request(
            _POST_CAPTURE_MODEL_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()

        topics = [line.strip().lstrip("0123456789.-) ") for line in text.splitlines() if line.strip()]
        return [t for t in topics if 2 <= len(t.split()) <= 6][:5]

    except Exception as e:
        logger.debug(f"35B focus extraction call failed: {e}")
        return []
