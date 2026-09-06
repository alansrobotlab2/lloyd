#!/usr/bin/env python3
"""
Lloyd MCP Server: Session — agent persistent memory + cross-session recall.

Tools:
    memory_read, memory_add, memory_replace, memory_remove, session_recall
    (5 tools)

Memory files: ~/obsidian/lloyd/MEMORY.md, ~/obsidian/lloyd/USER.md
Session transcripts: ~/lloyd/sessions/*.json

Split out of agent_mcp/memory.py as part of Task #340 PR 5. Owns:
    - The agent's persistent self-memory (MEMORY.md / USER.md)
    - Prompt-injection guardrails on memory writes
    - Session transcript indexing and tokenized search for cross-session
      recall ("what did we work on yesterday?")
"""

import datetime
import json
import re
import time
from pathlib import Path
from typing import Optional

from mcp.types import Tool

from agent_mcp._shared import (
    ErrorCode,
    _ENTITY_STOPWORDS,
    _err,
    _wrap,
)

# ── Constants ────────────────────────────────────────────────────────────────

MEMORIES_ROOT = Path.home() / "obsidian" / "lloyd"
MEMORY_FILES = {"MEMORY.md", "USER.md"}

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"you\s+are\s+now\s+a", re.I),
    re.compile(r"disregard\s+(your\s+)?(previous\s+)?instructions", re.I),
    re.compile(r"new\s+system\s+prompt", re.I),
    re.compile(r"pretend\s+you\s+are", re.I),
    re.compile(r"\x00|​|‌|‍|⁠|﻿", re.I),
]

from app.paths import SESSIONS_DIR  # anchored to LLOYD_HOME, not $HOME/lloyd
_SESSION_INDEX_TTL = 120       # cache session index for 2 min
_SESSION_CORPUS_MAX = 5000     # max chars of searchable text per session

_session_index_cache: Optional[tuple] = None  # (monotonic_ts, max_days, {filename: metadata})


# ── Session memory tools ─────────────────────────────────────────────────────

def _check_injection(text: str) -> Optional[str]:
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return "Potential prompt injection detected in entry"
    return None


def _memory_read(params: dict) -> dict:
    file = params.get("file", "MEMORY.md").strip()
    if file not in MEMORY_FILES:
        return _err(f"Invalid file. Must be one of: {', '.join(sorted(MEMORY_FILES))}", ErrorCode.INVALID_PARAM)
    filepath = MEMORIES_ROOT / file
    if not filepath.exists():
        return {"content": "", "file": file}
    return {"content": filepath.read_text(encoding="utf-8"), "file": file}


def _memory_add(params: dict) -> dict:
    file = params.get("file", "MEMORY.md").strip()
    entry = params.get("entry", "").strip()
    if file not in MEMORY_FILES:
        return _err(f"Invalid file. Must be one of: {', '.join(sorted(MEMORY_FILES))}", ErrorCode.INVALID_PARAM)
    if not entry:
        return _err("entry is required", ErrorCode.MISSING_PARAM)
    injection_msg = _check_injection(entry)
    if injection_msg:
        return _err(injection_msg, ErrorCode.INJECTION)
    MEMORIES_ROOT.mkdir(parents=True, exist_ok=True)
    filepath = MEMORIES_ROOT / file
    existing = filepath.read_text(encoding="utf-8") if filepath.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    filepath.write_text(existing + entry + "\n", encoding="utf-8")
    return {"success": True, "file": file}


def _memory_replace(params: dict) -> dict:
    file = params.get("file", "MEMORY.md").strip()
    old_text = params.get("old_text", "")
    new_text = params.get("new_text", "")
    if file not in MEMORY_FILES:
        return _err(f"Invalid file. Must be one of: {', '.join(sorted(MEMORY_FILES))}", ErrorCode.INVALID_PARAM)
    if not old_text:
        return _err("old_text is required", ErrorCode.MISSING_PARAM)
    injection_msg = _check_injection(new_text)
    if injection_msg:
        return _err(injection_msg, ErrorCode.INJECTION)
    filepath = MEMORIES_ROOT / file
    if not filepath.exists():
        return _err(f"{file} does not exist", ErrorCode.NOT_FOUND)
    content = filepath.read_text(encoding="utf-8")
    if old_text not in content:
        return _err("old_text not found in file", ErrorCode.NO_MATCH, matched=False)
    filepath.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return {"success": True, "file": file}


def _memory_remove(params: dict) -> dict:
    file = params.get("file", "MEMORY.md").strip()
    entry = params.get("entry", "").strip()
    if file not in MEMORY_FILES:
        return _err(f"Invalid file. Must be one of: {', '.join(sorted(MEMORY_FILES))}", ErrorCode.INVALID_PARAM)
    if not entry:
        return _err("entry is required", ErrorCode.MISSING_PARAM)
    filepath = MEMORIES_ROOT / file
    if not filepath.exists():
        return _err(f"{file} does not exist", ErrorCode.NOT_FOUND)
    content = filepath.read_text(encoding="utf-8")
    if entry not in content:
        return _err("entry not found in file", ErrorCode.NO_MATCH, matched=False)
    updated = content.replace(entry, "", 1)
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    filepath.write_text(updated, encoding="utf-8")
    return {"success": True, "file": file}


# ── Session recall ───────────────────────────────────────────────────────────

def _extract_msg_text(msg: dict) -> str:
    """Extract plain text from a session message, skipping injected context."""
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        text = " ".join(t for t in parts if t)
    elif isinstance(content, str):
        text = content
    else:
        return ""
    stripped = text.strip()
    if any(stripped.startswith(p) for p in (
        "<context>", "<system-reminder>", "<memory>", "<daily_notes>",
        "[cron:", "[System Message]", "[autonomy:",
    )):
        return ""
    return text


def _load_session_index(max_days: int = 14) -> dict:
    """Load session metadata from recent JSON files. Cached with TTL.

    Returns {filename: {session_id, date_str, time_str, created_at, model,
    preview, message_count, platform, corpus, user_snippets}}.
    """
    global _session_index_cache
    now = time.monotonic()

    # The cache is only reusable when it was built with a window at least
    # as wide as the one requested. prefetch asks for 3 days and
    # session_recall for 14; before this check, whichever ran first
    # served the other for the TTL — a 3-day index silently answering a
    # 14-day recall. Callers filter by date themselves, so a wider cached
    # window is always safe to hand back.
    if (_session_index_cache
            and (now - _session_index_cache[0]) < _SESSION_INDEX_TTL
            and _session_index_cache[1] >= max_days):
        return _session_index_cache[2]

    cutoff = (datetime.datetime.now() - datetime.timedelta(days=max_days)).strftime("%Y%m%d")
    index: dict[str, dict] = {}

    if not SESSIONS_DIR.exists():
        _session_index_cache = (now, max_days, index)
        return index

    for f in SESSIONS_DIR.iterdir():
        if not f.name.endswith(".json") or f.name.startswith("autonomy_"):
            continue
        parts = f.name.split("_")
        if len(parts) < 3 or len(parts[0]) != 8:
            continue
        if parts[0] < cutoff:
            continue

        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("platform") == "autonomy":
                continue

            user_texts: list[str] = []
            asst_texts: list[str] = []
            for msg in data.get("messages", []):
                text = _extract_msg_text(msg)
                if not text.strip():
                    continue
                if msg.get("role") == "user":
                    user_texts.append(text[:500])
                elif msg.get("role") == "assistant":
                    asst_texts.append(text[:300])

            corpus = " ".join(user_texts + asst_texts).lower()[:_SESSION_CORPUS_MAX]

            index[f.name] = {
                "filename": f.name,
                "session_id": data.get("session_id", f.stem),
                "date_str": parts[0],
                "time_str": parts[1] if len(parts) > 1 else "",
                "created_at": data.get("created_at", ""),
                "model": data.get("model", ""),
                "preview": data.get("preview", ""),
                "message_count": data.get("message_count", 0),
                "platform": data.get("platform", ""),
                "corpus": corpus,
                "user_snippets": [t[:300] for t in user_texts[:8]],
            }
        except Exception:
            continue

    _session_index_cache = (now, max_days, index)
    return index


def _score_session(session: dict, query_tokens: set) -> float:
    """Score a session against query tokens using term frequency."""
    corpus = session.get("corpus", "")
    if not corpus or not query_tokens:
        return 0.0
    score = 0.0
    for token in query_tokens:
        count = corpus.count(token)
        if count > 0:
            score += 1.0 + 0.3 * min(count - 1, 4)
    return score / len(query_tokens)


def _session_recall(params: dict) -> dict:
    """Search recent session transcripts for topics, decisions, or discussions."""
    query = params.get("query", "").strip()
    if not query:
        return _err("query is required", ErrorCode.MISSING_PARAM, sessions=[])

    days = int(params.get("days", 7))
    limit = int(params.get("limit", 5))

    index = _load_session_index(max_days=max(days, 14))

    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
    sessions = [s for s in index.values() if s["date_str"] >= cutoff]

    query_tokens = {w for w in re.findall(r"\w+", query.lower())
                    if w not in _ENTITY_STOPWORDS and len(w) >= 2}

    if not query_tokens:
        # No meaningful tokens — return most recent sessions
        sessions.sort(key=lambda s: s["date_str"] + s.get("time_str", ""), reverse=True)
        results = [{
            "session_id": s["session_id"],
            "created_at": s["created_at"],
            "model": s["model"],
            "preview": s["preview"][:200],
            "message_count": s["message_count"],
            "snippets": s["user_snippets"][:3],
        } for s in sessions[:limit]]
        return {"query": query, "sessions": results, "total_searched": len(sessions)}

    scored = []
    for s in sessions:
        score = _score_session(s, query_tokens)
        if score > 0.2:
            scored.append((score, s))
    scored.sort(key=lambda x: -x[0])

    results = []
    for score, s in scored[:limit]:
        snippets = []
        for text in s.get("user_snippets", []):
            if any(t in text.lower() for t in query_tokens):
                snippets.append(text[:300])
                if len(snippets) >= 3:
                    break
        results.append({
            "session_id": s["session_id"],
            "created_at": s["created_at"],
            "model": s["model"],
            "preview": s["preview"][:200],
            "message_count": s["message_count"],
            "match_score": round(score, 3),
            "snippets": snippets or [s["preview"][:200]],
        })

    return {"query": query, "sessions": results, "total_searched": len(sessions)}


# ── MCP registration ─────────────────────────────────────────────────────────

async def list_tools():
    return [
        Tool(name="memory_read", description="Read the cross-session memory files. MEMORY.md holds durable working notes; USER.md holds standing facts about the user. Returns the whole file.", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"], "description": "Which file to read"}}, "required": []}),
        Tool(name="memory_add", description="Append one entry to a cross-session memory file. Appends only — use memory_replace to change an existing line and memory_remove to drop one.", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"], "description": "MEMORY.md for durable working notes, USER.md for facts about the user (default MEMORY.md)"}, "entry": {"type": "string", "description": "Text to append"}}, "required": ["entry"]}),
        Tool(name="memory_replace", description="Replace text in a cross-session memory file by substring match. Fails if old_text is absent, so a stale edit is reported rather than silently skipped.", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"], "description": "MEMORY.md for durable working notes, USER.md for facts about the user (default MEMORY.md)"}, "old_text": {"type": "string", "description": "Existing text to find (substring, must appear exactly once)"}, "new_text": {"type": "string", "description": "Replacement text"}}, "required": ["old_text", "new_text"]}),
        Tool(name="memory_remove", description="Remove an entry from MEMORY.md or USER.md (substring match).", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"], "description": "MEMORY.md for durable working notes, USER.md for facts about the user (default MEMORY.md)"}, "entry": {"type": "string", "description": "Text to remove"}}, "required": ["entry"]}),
        Tool(name="session_recall", description="Search recent session transcripts for topics, decisions, or discussions from past sessions. Use for cross-session context like 'what did we work on today?' or 'what was decided about X?'", inputSchema={
            "type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "days": {"type": "integer", "description": "Days back to search (default: 7)"}, "limit": {"type": "integer", "description": "Max results (default: 5)"}}, "required": ["query"]}),
    ]


async def call_tool(name: str, arguments: dict):
    handlers = {
        "memory_read": _memory_read, "memory_add": _memory_add,
        "memory_replace": _memory_replace, "memory_remove": _memory_remove,
        "session_recall": _session_recall,
    }
    handler = handlers.get(name)
    if handler:
        return _wrap(handler(arguments))
    return _wrap(_err(f"Unknown tool: {name}", ErrorCode.UNKNOWN_TOOL))
