#!/usr/bin/env python3
"""
Lloyd MCP Server: Session — agent persistent memory + cross-session recall.

Tools:
    memory_read, memory_add, memory_replace, memory_remove, session_recall
    (5 tools)

Memory files: ~/obsidian/lloyd/MEMORY.md, ~/obsidian/lloyd/USER.md, and topic
files ~/obsidian/lloyd/memory/<slug>.md named `topics/<slug>` (review
2026-09-24, P4: pulled by memory_read, never rendered into a prompt)
Session transcripts: ~/lloyd-data/sessions/*.json

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
    _fit_not_found,
    _near_match_hint,
    _wrap,
)

# Stdlib-only and dependency-free by design (see its docstring, and the same
# argument in `prompt_surface`): the byte ceiling has to be ONE definition shared by
# the memory tools, `Write`/`Edit`, `vault_write`, the vault-round validator and the
# tests. `prompt_builder` is too heavy to import into this process, and a second
# copy of the number is how two writers end up disagreeing about what is bounded —
# which is exactly the state #1010 was filed on.
from app.memory_ceiling import (
    TOPIC_SLUG_RE,
    TOPICS_SUBDIR,
    memory_write_error,
    topic_path,
    topic_slug,
)
from prompt_surface import ENTRY_TYPES

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

from app.atomic_io import commit_lock, write_text_durable
from app.paths import SESSIONS_DIR  # anchored to LLOYD_HOME, not $HOME/lloyd
from app.sessions_io import is_user_session
_SESSION_INDEX_TTL = 120       # cache session index for 2 min

_session_index_cache: Optional[tuple] = None  # (monotonic_ts, max_days, {filename: metadata})


# ── Session memory tools ─────────────────────────────────────────────────────

def _check_injection(text: str) -> Optional[str]:
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return "Potential prompt injection detected in entry"
    return None


_INVALID_FILE = ("Invalid file. Must be MEMORY.md, USER.md, or topics/<slug> "
                 "(slug: 1-48 of a-z, 0-9, '-')")


def _resolve_file(file: str) -> Optional[Path]:
    """The path a memory tool's `file` names, or None when it names nothing legal.

    MEMORY.md and USER.md by name; `topics/<slug>` → `<root>/memory/<slug>.md`
    (review 2026-09-24, P4). The slug grammar in `app.memory_ceiling` is the whole
    traversal defence: `topics/../SOUL`, `topics/a/b` and `topics/X` all fail it.
    """
    if file in MEMORY_FILES:
        return MEMORIES_ROOT / file
    slug = topic_slug(file)
    return topic_path(MEMORIES_ROOT, slug) if slug else None


def _topic_names() -> list[str]:
    tdir = MEMORIES_ROOT / TOPICS_SUBDIR
    if not tdir.is_dir():
        return []
    return sorted(f"topics/{p.stem}" for p in tdir.glob("*.md")
                  if TOPIC_SLUG_RE.fullmatch(p.stem))


def _memory_read(params: dict) -> dict:
    file = params.get("file", "MEMORY.md").strip()
    filepath = _resolve_file(file)
    if filepath is None:
        return _err(_INVALID_FILE, ErrorCode.INVALID_PARAM)
    if not filepath.exists():
        if file in MEMORY_FILES:
            return {"content": "", "file": file}
        # A dangling index link is the one miss worth answering with the list: the
        # model followed a `→ topics/<slug>` line, and the names that do exist are
        # the cheapest way back to the right one.
        return {"content": "", "file": file, "exists": False, "topics": _topic_names()}
    return {"content": filepath.read_text(encoding="utf-8"), "file": file}


def _date_stamp_enabled() -> bool:
    """`memory_tools.date_stamp_entries` in config.yaml (default off).

    #622: once a consolidation pass rewrites MEMORY.md, file position stops
    saying which of two conflicting lines is newer, and the model then acts on
    whichever sits LAST — 28 of 33 trials when that was the superseded line,
    0 of 27 when it was not (`eval/measurements/stale-fact-2026-09-24.md`). A
    written-on date is a recency cue a rewrite carries along with the line.
    """
    try:
        from app.config import CONFIG
        return bool((CONFIG.get("memory_tools") or {}).get("date_stamp_entries", False))
    except Exception:
        return False


def _entry_date() -> datetime.date:
    """The date a `memory_add` entry is stamped with (UTC). Its own function so
    an eval can plant entries on the dates it means to."""
    return datetime.datetime.now(datetime.timezone.utc).date()


def _date_stamped(entry: str) -> str:
    """`- text` -> `- (2026-09-02) text`; a line with no bullet gets the prefix
    at its start. Only the first line is stamped: one call is one entry."""
    stamp = f"({_entry_date().isoformat()}) "
    m = re.match(r"(\s*(?:[-*+]|\d+\.)\s+)", entry)
    return entry[:m.end()] + stamp + entry[m.end():] if m else stamp + entry


def _typed_entries_enabled() -> bool:
    """`memory_tools.typed_entries` in config.yaml (code default off).

    Review 2026-09-24, P4: an entry carries its kind — `user`, `feedback`,
    `project`, `reference` — so a consolidation pass can tell Alan's rulings (the
    lines an index must never lose) from working state it may move into a topic
    file. Off here for the same reason as the date stamp: the writer-lane tests
    assert entries byte for byte.
    """
    try:
        from app.config import CONFIG
        return bool((CONFIG.get("memory_tools") or {}).get("typed_entries", False))
    except Exception:
        return False


_BULLET_RE = re.compile(r"(\s*(?:[-*+]|\d+\.)\s+)")
_TYPE_TAG_RE = re.compile(r"\[(" + "|".join(ENTRY_TYPES) + r")\]\s+")
_DATE_TAG_RE = re.compile(r"\(\d{4}-\d{2}-\d{2}\)\s+")


def _default_entry_type(file: str) -> str:
    """USER.md holds who Alan is; everything else defaults to working state."""
    return "user" if file == "USER.md" else "project"


def _typed_entry(entry: str, etype: str, *, stamp: bool) -> str:
    """`text` -> `- [feedback] (2026-09-24) text`.

    A bullet is added when the entry has none — the typed form is a top-level
    bullet by definition, which is also what keeps it one `app.uptake` entry
    (`_MEMORY_DOC_RE`). A tag or date the writer already wrote is kept, never
    doubled: a model that copies an index line's shape into its entry should not
    get `[project] [project]`.
    """
    m = _BULLET_RE.match(entry)
    head, rest = (entry[:m.end()], entry[m.end():]) if m else ("- ", entry)
    tag = _TYPE_TAG_RE.match(rest)
    if tag:
        prefix, rest = rest[:tag.end()], rest[tag.end():]
    else:
        prefix = f"[{etype}] "
    if stamp and not _DATE_TAG_RE.match(rest):
        rest = f"({_entry_date().isoformat()}) " + rest
    return head + prefix + rest


def _memory_add(params: dict) -> dict:
    file = params.get("file", "MEMORY.md").strip()
    entry = params.get("entry", "").strip()
    etype = str(params.get("type") or "").strip().lower() or _default_entry_type(file)
    if etype not in ENTRY_TYPES:
        return _err(f"Invalid type {etype!r}. Must be one of: {', '.join(ENTRY_TYPES)}",
                    ErrorCode.INVALID_PARAM)
    if entry:
        if _typed_entries_enabled():
            entry = _typed_entry(entry, etype, stamp=_date_stamp_enabled())
        elif _date_stamp_enabled():
            entry = _date_stamped(entry)
    filepath = _resolve_file(file)
    if filepath is None:
        return _err(_INVALID_FILE, ErrorCode.INVALID_PARAM)
    if not entry:
        return _err("entry is required", ErrorCode.MISSING_PARAM)
    injection_msg = _check_injection(entry)
    if injection_msg:
        return _err(injection_msg, ErrorCode.INJECTION)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The read belongs inside the lock. Locking only the write turns a lost
        # update into a slightly later lost update, and MEMORY.md/USER.md have
        # three lanes writing them — the memory tools, Write/Edit (what the
        # nightly knowledge-write job uses), and vault_write.
        with commit_lock(filepath):
            existing = filepath.read_text(encoding="utf-8") if filepath.exists() else ""
            if not existing and file not in MEMORY_FILES:
                # A new topic opens with its name, so a `memory_read` of it
                # says what it is before the first entry does.
                existing = f"# {file}\n\n"
            if existing and not existing.endswith("\n"):
                existing += "\n"
            updated = existing + entry + "\n"
            # #1010/#507: the ceiling is enforced HERE, at the write site, and not
            # only in a test that reads the live vault. `lloyd/USER.md` grew
            # 48,068 B → 95,302 B in five days of appends and no check could refuse
            # one, because there was no byte constant to cross; a hard-rung test on
            # the result would have punished the next author rather than the writer
            # (the reason `prompt_surface` exists at all). Priced on the exact bytes
            # the line below writes, inside the lock, so the size refused is the
            # size that would have landed and not a recomputation of it.
            ceiling_msg = memory_write_error(filepath, updated)
            if ceiling_msg:
                return _err(ceiling_msg, ErrorCode.INVALID_PARAM)
            write_text_durable(filepath, updated)
    except TimeoutError as exc:
        return _err(str(exc), ErrorCode.LOCK_TIMEOUT)
    return {"success": True, "file": file}


def _memory_replace(params: dict) -> dict:
    file = params.get("file", "MEMORY.md").strip()
    old_text = params.get("old_text", "")
    new_text = params.get("new_text", "")
    filepath = _resolve_file(file)
    if filepath is None:
        return _err(_INVALID_FILE, ErrorCode.INVALID_PARAM)
    if not old_text:
        return _err("old_text is required", ErrorCode.MISSING_PARAM)
    injection_msg = _check_injection(new_text)
    if injection_msg:
        return _err(injection_msg, ErrorCode.INJECTION)
    try:
        with commit_lock(filepath):
            if not filepath.exists():
                return _err(f"{file} does not exist", ErrorCode.NOT_FOUND)
            content = filepath.read_text(encoding="utf-8")
            if old_text not in content:
                # Same report `Edit` gives, from the bytes already under the lock;
                # `code` and `matched` are the pre-existing companions callers read.
                return _err(_fit_not_found(
                    "old_text not found in file",
                    _near_match_hint(content, old_text, label="old_text")),
                    ErrorCode.NO_MATCH, matched=False)
            updated = content.replace(old_text, new_text, 1)
            # A replace is an append wearing a disguise: a longer `new_text` grows
            # the same file `memory_add` is refused for, and the trim route that has
            # to keep working is the shrinking one — which `memory_write_error`
            # allows even above the ceiling, so a file can always be cut back to its
            # limit by the same tool that would refuse to grow it.
            ceiling_msg = memory_write_error(filepath, updated)
            if ceiling_msg:
                return _err(ceiling_msg, ErrorCode.INVALID_PARAM)
            write_text_durable(filepath, updated)
    except TimeoutError as exc:
        return _err(str(exc), ErrorCode.LOCK_TIMEOUT)
    return {"success": True, "file": file}


def _memory_remove(params: dict) -> dict:
    file = params.get("file", "MEMORY.md").strip()
    entry = params.get("entry", "").strip()
    filepath = _resolve_file(file)
    if filepath is None:
        return _err(_INVALID_FILE, ErrorCode.INVALID_PARAM)
    if not entry:
        return _err("entry is required", ErrorCode.MISSING_PARAM)
    try:
        with commit_lock(filepath):
            if not filepath.exists():
                return _err(f"{file} does not exist", ErrorCode.NOT_FOUND)
            content = filepath.read_text(encoding="utf-8")
            if entry not in content:
                return _err("entry not found in file", ErrorCode.NO_MATCH, matched=False)
            updated = content.replace(entry, "", 1)
            updated = re.sub(r"\n{3,}", "\n\n", updated)
            write_text_durable(filepath, updated)
    except TimeoutError as exc:
        return _err(str(exc), ErrorCode.LOCK_TIMEOUT)
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
    preview, message_count, platform, corpus, turns, user_snippets}}.

    `corpus` is the whole lowercased user+assistant text of the session, turn
    by turn in message order, and `turns` is one (message index, role, start,
    end) row per turn with `corpus[start:end]` equal to that turn's own text.
    Both belong to `_load_session_index` rather than to `_session_recall`
    because `prefetch.py:842` builds this same index and scores it with the
    same `_score_session` to render the per-turn `<recent-sessions>` block —
    the recall tool is not the only consumer of what is indexed here.

    Text carried by a role other than user/assistant is never indexed, and
    `_extract_msg_text` drops injected-context messages and non-text content
    blocks, which is what keeps model reasoning out of this cache: the
    transcript's thinking rows are `role="thinking"` with their text in
    `reasoning`, so neither the role filter nor the extractor can reach them.
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
            # Recall is over what the USER said. A worker turn arrives through
            # the chat path, so without this the model's own background jobs
            # were being recalled back to it as the user's conversations.
            if not is_user_session(data):
                continue

            # One row per indexed turn: (message index, role, start, end) into
            # `corpus`, with the WHOLE turn in the corpus rather than a clip of
            # it. The corpus used to be cut per message (500 chars user, 300
            # assistant) and then cut again from the head at 5,000 chars, so the
            # tail of every long session was unsearchable and text past the
            # per-message clip was unsearchable even in a short one. Scoring
            # read only that string, so such a term could not be found by any
            # path — and the empty result looked exactly like a session that
            # never mentioned it. The rows are what let a hit report where in
            # the transcript it came from instead of quoting the head.
            #
            # No cap replaces them: a fixed cap leaves the same blind spot on
            # whatever the long tail grows to. Cost was measured, not assumed —
            # on 2026-09-22 the live 14-day window held 88 sessions and 0.59 MB
            # of searchable text (0.93 MB for the whole index, keys and turn
            # rows included), building it ran 1.4x FASTER than the clipping
            # version (9.3 ms vs 14.0 ms best-of-3 over the same parsed files),
            # and a full scoring pass got 0.14 ms slower. The cap was buying
            # latency nobody was paying.
            turns: list[tuple] = []
            user_texts: list[str] = []
            chunks: list[str] = []
            pos = 0
            for i, msg in enumerate(data.get("messages", [])):
                role = msg.get("role")
                if role not in ("user", "assistant"):
                    continue
                text = _extract_msg_text(msg)
                if not text.strip():
                    continue
                lowered = text.lower()
                if chunks:
                    pos += 1              # the single space joining turns
                chunks.append(lowered)
                turns.append((i, role, pos, pos + len(lowered)))
                pos += len(lowered)
                if role == "user":
                    user_texts.append(text)

            corpus = " ".join(chunks)

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
                "turns": turns,
                "user_snippets": [t[:300] for t in user_texts[:8]],
            }
        except Exception:
            continue

    _session_index_cache = (now, max_days, index)
    return index


def _score_session(session: dict, query_tokens: set) -> float:
    """Score a session against query tokens using term frequency.

    Reads `corpus`, which since #1090 is the session's whole turn text rather
    than a head-truncated digest — so this scores more, not differently, and
    `prefetch.py` needs no change to get the wider coverage.
    """
    corpus = session.get("corpus", "")
    if not corpus or not query_tokens:
        return 0.0
    score = 0.0
    for token in query_tokens:
        count = corpus.count(token)
        if count > 0:
            score += 1.0 + 0.3 * min(count - 1, 4)
    return score / len(query_tokens)


# How much of a matched turn a result quotes, and how many matched turns it
# quotes. A turn can be tens of thousands of characters, so the quote is a
# window that starts a third of it before the earliest hit in that turn rather
# than the whole turn.
_SNIPPET_WINDOW = 300
_SNIPPET_MAX = 3


def _match_evidence(session: dict, query_tokens: set,
                    limit: int = _SNIPPET_MAX) -> tuple[list[str], dict]:
    """Return (snippets, location) for the turns of a session that hold a token.

    The snippets are windows around the hits and so always contain the query
    term; `location` names the FIRST matched turn as its index in the session's
    own `messages` list, its role, and its `char_offset` into `corpus` — an
    offset in characters into that string, not a byte offset into the JSON file.
    Together they say where in the transcript a match came from.

    Before #1090 the snippets were read out of `user_snippets` — the first eight
    user turns, clipped — so a match anywhere else in the session was reported
    with a quote that did not contain the term, and the caller was left with the
    session preview as its only clue.

    A query token is `\\w+` shaped and turns are joined by a single space, so a
    token present in `corpus` lies inside one turn's slice and cannot straddle
    two of them; that is why a session that scored above threshold always yields
    at least one snippet here.

    One stated trade-off: a quote is taken from `corpus`, so it comes back
    lowercased. Quoting original case would mean holding the session's text
    twice in the cached index — the two copies cost more than a quote's
    capitalisation is worth, and `preview`/`user_snippets` still carry the
    original case for the head of the session.
    """
    corpus = session.get("corpus", "")
    snippets: list[str] = []
    location: dict = {}
    for msg_index, role, start, end in session.get("turns", []):
        turn = corpus[start:end]
        hits = [turn.find(token) for token in query_tokens if token in turn]
        if not hits:
            continue
        if not location:
            location = {"turn_index": msg_index, "role": role, "char_offset": start}
        cut = min(hits)
        window = max(0, cut - _SNIPPET_WINDOW // 3)
        snippets.append(turn[window:window + _SNIPPET_WINDOW])
        if len(snippets) >= limit:
            break
    return snippets, location


# ── session_recall as a qmd query (#1485, OFF) ───────────────────────────────
#
# "tokens" is the scorer below: substring counts over the user sessions' JSON
# in the window. "qmd" asks the daemon's lex+vec hybrid over the `sessions`
# collection (the markdown exports of the same chats), keeps the 5-result shape
# and the date window, and drops a transcript that echoes the query or is the
# caller's own session (#1511). Measured and not deployed
# (`eval/measurements/episodic-recall-2026-09-25.md`): where both backends can see
# the chat (JSON era, n=10) hit@5 0.9 tokens vs 0.8 qmd, diff -0.1 [-0.4, +0.2],
# at 0.7 ms vs 62 ms p50. Its 0.72-vs-0 win on older chats (n=32) is only the
# 2026-09-22 wipe: those chats' JSON is gone and their exports survived.
SESSION_RECALL_BACKEND = "tokens"
_SESSION_QMD_OVERFETCH = 4
_SESSION_ID_RE = re.compile(r"^(\d{8})_(\d{6})_")


def _session_recall_qmd(query: str, days: int, limit: int) -> dict:
    from agent_mcp import _task_registry
    from agent_mcp.transcript_self_hit import note_path, self_hit_reason
    from agent_mcp.vault import RECALL_LEX_MODE, _qmd_daemon_search, strip_qmd_snippet

    rows = _qmd_daemon_search(query, max(limit, 1) * _SESSION_QMD_OVERFETCH, ["sessions"],
                              skip_rerank=True, legs=("lex", "vec"), lex_mode=RECALL_LEX_MODE)
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
    sid = str(_task_registry.current_session_id.get("") or "") or None
    results, seen = [], set()
    for r in rows:
        rel = note_path(str(r.get("file") or ""))
        if rel is None:
            continue
        stem = Path(rel).stem
        m = _SESSION_ID_RE.match(stem)
        if not m or m.group(1) < cutoff or stem in seen:
            continue
        if self_hit_reason(query, r, sid):
            continue
        seen.add(stem)
        snippet, _ = strip_qmd_snippet(r.get("snippet", ""))
        d, t = m.group(1), m.group(2)
        results.append({
            "session_id": stem,
            "created_at": f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}",
            "model": "",
            "preview": snippet[:200],
            "message_count": 0,
            "match_score": round(float(r.get("score", 0) or 0), 3),
            "snippets": [snippet[:_SNIPPET_WINDOW]],
            "match_location": {},
        })
        if len(results) >= limit:
            break
    return {"query": query, "sessions": results, "total_searched": len(rows),
            "backend": "qmd"}


def _session_recall(params: dict) -> dict:
    """Search recent session transcripts for topics, decisions, or discussions."""
    query = params.get("query", "").strip()
    if not query:
        return _err("query is required", ErrorCode.MISSING_PARAM, sessions=[])
    if SESSION_RECALL_BACKEND == "qmd":
        try:
            return _session_recall_qmd(query, int(params.get("days", 7)),
                                       int(params.get("limit", 5)))
        except Exception:  # noqa: BLE001 — a daemon outage falls back to the scorer
            pass

    days = int(params.get("days", 7))
    limit = int(params.get("limit", 5))

    index = _load_session_index(max_days=max(days, 14))

    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
    sessions = [s for s in index.values() if s["date_str"] >= cutoff]

    query_tokens = {w for w in re.findall(r"\w+", query.lower())
                    if w not in _ENTITY_STOPWORDS and len(w) >= 2}

    if not query_tokens:
        # No meaningful tokens — return most recent sessions. These rows carry
        # no `match_location`: nothing matched, so there is no position to
        # report, and their snippets stay the head-of-session `user_snippets`
        # that suit a "what was I working on" listing.
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
        # Snippets now come from the turns that actually hold a query token,
        # and the result says which turn that was. The preview fallback is kept
        # for a session with no turn rows at all — an index row built without
        # them — so the shape never loses its `snippets` key.
        snippets, location = _match_evidence(s, query_tokens)
        results.append({
            "session_id": s["session_id"],
            "created_at": s["created_at"],
            "model": s["model"],
            "preview": s["preview"][:200],
            "message_count": s["message_count"],
            "match_score": round(score, 3),
            "snippets": snippets or [s["preview"][:200]],
            "match_location": location,
        })

    return {"query": query, "sessions": results, "total_searched": len(sessions)}


# ── MCP registration ─────────────────────────────────────────────────────────

# No `enum` since P4: `topics/<slug>` is open-ended. The pattern states the same
# grammar `_resolve_file` enforces; the handler is what refuses.
_FILE_PATTERN = r"^(MEMORY\.md|USER\.md|topics/[a-z0-9-]{1,48}(\.md)?)$"
_FILE_PARAM = {"type": "string", "pattern": _FILE_PATTERN,
               "description": "MEMORY.md for durable working notes, USER.md for facts "
                              "about the user, topics/<slug> for a topic file's detail "
                              "(default MEMORY.md)"}
_FILE_PARAM_READ = {**_FILE_PARAM, "description": "Which file to read: MEMORY.md, "
                    "USER.md, or topics/<slug> (default MEMORY.md)"}

async def list_tools():
    return [
        Tool(name="memory_read", description="Use to check what is already remembered before memory_add; to search past chats use session_recall. Read the cross-session memory files. MEMORY.md holds durable working notes; USER.md holds standing facts about the user; topics/<slug> is a topic file an index line points at (`→ topics/<slug>`), holding the detail that line summarises. Returns the whole file.", inputSchema={
            "type": "object", "properties": {"file": _FILE_PARAM_READ}, "required": []}),
        Tool(name="memory_add", description="Use to record a durable note or user fact; to change an existing entry use memory_replace instead. Append one entry to a cross-session memory file. Appends only — use memory_replace to change an existing line and memory_remove to drop one.", inputSchema={
            "type": "object", "properties": {"file": _FILE_PARAM, "entry": {"type": "string", "description": "Text to append"}, "type": {"type": "string", "enum": list(ENTRY_TYPES), "description": "What kind of entry: feedback (a ruling or correction from the user), user (a fact about the user), project (working state; default for MEMORY.md and topics), reference (where something lives). Default user for USER.md."}}, "required": ["entry"]}),
        Tool(name="memory_replace", description="Replace text in a cross-session memory file by substring match. Fails if old_text is absent, so a stale edit is reported rather than silently skipped.", inputSchema={
            "type": "object", "properties": {"file": _FILE_PARAM, "old_text": {"type": "string", "description": "Existing text to find (substring, must appear exactly once)"}, "new_text": {"type": "string", "description": "Replacement text"}}, "required": ["old_text", "new_text"]}),
        Tool(name="memory_remove", description="Use to drop a memory entry that is wrong or obsolete; to correct it in place use memory_replace. Remove an entry from MEMORY.md, USER.md or a topics/<slug> file (substring match).", inputSchema={
            "type": "object", "properties": {"file": _FILE_PARAM, "entry": {"type": "string", "description": "Text to remove"}}, "required": ["entry"]}),
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
