"""Synchronous calls to the secondary LLM for post-session jobs.

The primary model talks to the user; the secondary model handles
summary/fact/focus extraction in the background so the primary generation
path stays unblocked. When `secondary_enabled: false` in config.yaml, the
alias resolver routes these calls to the primary instead.

Per-job routing lives here, not in config.yaml, because the two knobs mean
different things: `secondary_enabled` is "does the box have a secondary
engine at all" (it also decides whether `agent-llm-secondary` is started —
`server.py:172`), while `JOBS_ON_PRIMARY` is "this job's measured quality on
the cheap engine was not good enough". Flipping the first to fix the second
would take the engine down for every other job.
"""

import json
import logging
import re
import urllib.request
from typing import Optional


logger = logging.getLogger("lloyd-server")

#: Job names whose work goes to the primary even while the secondary is
#: enabled. The five routable jobs are `title`, `capture`, `facts`, `focus`
#: and `voice`.
#:
#: It must not be filled in by hand: the only legitimate content is the set
#: of jobs `eval/secondary_routing_eval.py` pairs primary against secondary
#: for and sends outside its stated tolerance, written by that script's
#: `--pin`, and `tests/test_secondary_routing_eval.py` fails if this set and
#: the decisions in that eval's checked-in artifact
#: (`eval/secondary-routing/decisions.yaml`) disagree. What is in here now —
#: `title` — is the decision four decision-grade runs (2026-09-16 twice,
#: 09-17, 09-18) reached independently, each of them putting `title` more
#: than the 5.0-point margin outside tolerance. A routing change that nobody
#: measured is exactly the defect item #551 exists to remove — the secondary
#: was chosen for throughput and never re-checked against a quality number.
JOBS_ON_PRIMARY: frozenset = frozenset({'title'})


def _engine_for(job: str) -> str:
    """Alias for one routed job: `secondary`, or `primary` if it was flipped.

    Returns the *alias*, so `resolve_model_alias` still has the last word and
    `secondary_enabled: false` keeps overriding everything — a flipped job
    asks for primary, a kept job asks for secondary, and both land on primary
    when the engine is off.
    """
    return "primary" if job in JOBS_ON_PRIMARY else "secondary"


def _endpoint(job: str) -> tuple[str, str]:
    """Resolve (chat_completions_url, model_name) for one routed job.

    `job` is required with no default on purpose. A default of "secondary"
    would resolve happily for a caller that forgot to name its job, and a
    newly-pinned job that forgot to pass its name would then quietly measure
    and run on the cheap engine while reading as routed. There is no correct
    default to pick: the whole point of item #551 is that the engine per job
    is a measured decision, not a fallback.
    """
    from app.config import resolve_model_alias, _get_model_cfg
    name = resolve_model_alias(_engine_for(job))
    cfg = _get_model_cfg(name) or {}
    base = cfg.get("base_url") or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")
    return f"{base.rstrip('/')}/v1/chat/completions", name


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

    url, model_name = _endpoint("capture")
    payload = {
        "model": model_name,
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
            url,
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
    url, model_name = _endpoint("facts")
    payload = {
        "model": model_name,
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
            url,
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

    url, model_name = _endpoint("voice")
    payload = {
        "model": model_name,
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
            url,
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
    url, model_name = _endpoint("focus")
    payload = {
        "model": model_name,
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
            url,
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


_TITLE_SYSTEM = (
    "You name conversations. Given a transcript, reply with a title of 3-6 "
    "words naming the specific subject — the system, file, bug, or decision "
    "the conversation is actually about. No quotes, no punctuation at the "
    "end, no preamble, no explanation. Output the title and nothing else."
)


def _sync_secondary_title(transcript: str, timeout: float = 30.0) -> Optional[str]:
    """Call the secondary for a few-word title describing a session.

    Returns the model's raw text — cleaning and validation belong to
    `app.session_titles`, which is the only caller and the only place that
    knows what a usable title looks like.
    """
    if not transcript.strip():
        return None

    url, model_name = _endpoint("title")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _TITLE_SYSTEM},
            {"role": "user", "content": (
                "Title this conversation in 3-6 words.\n\n"
                f"Transcript:\n{transcript}"
            )},
        ],
        "temperature": 0.2,
        # Room for a long-ish title and nothing more. A model that wants to
        # explain itself gets cut off rather than indulged.
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip() or None
    except Exception as e:
        logger.warning(f"secondary title call failed: {e}")
        return None
