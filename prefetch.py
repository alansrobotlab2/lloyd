#!/usr/bin/env python3
"""
prefetch.py — Automatic context prefetch layer.

Extracts keywords from the user message, searches skills, facts, backlog
refs, recent sessions, and vault documents in parallel, then prepends a
<context> block so the agent gets relevant content without needing
explicit tool calls.

Maintains per-session conversation focus via keyword accumulation with
exponential decay. Short/vague messages benefit from accumulated context
about what the conversation is actually about.

Called by the chat/voice routers before every run_query() invocation —
use `prefetch_context_async` from coroutine code so the budgeted wait
happens off the event loop.
"""

import asyncio
import json
import os
import re
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from agent_mcp.skills import SKILLS_DIRS, _iter_skills, _score_skill, _query_tokens
from agent_mcp._shared import _ENTITY_STOPWORDS
from agent_mcp.facts import _extract_entities_from_query, _get_facts_sync
from agent_mcp.session import _load_session_index, _score_session
from agent_mcp.vault import _qmd_daemon_search, _qmd_strip_stopwords

logger = logging.getLogger("lloyd.prefetch")

# ── Tuning ────────────────────────────────────────────────────────────────────

SKILL_THRESHOLD_FIRST = 3.0     # minimum score to inject first skill (full body)
SKILL_THRESHOLD_SECOND = 4.0    # minimum score to inject second skill (excerpt only)
SKILL_BODY_MAX = 6000           # chars, first skill
SKILL_EXCERPT_MAX = 500         # chars, second skill
FACT_MAX_ENTITIES = 2           # top N entities to look up
FACT_MAX_PER_ENTITY = 3         # top N facts per entity (by confidence)
MIN_MESSAGE_LEN = 10            # skip prefetch for very short messages

# Hard latency budget for the parallel prefetch phase. Any subtask not
# finished within this window is dropped for the current turn.
#
# Measured 2026-09-03 on this host, warm process (see architecture/subliminal.md):
#   skills     ~1ms      token sets memoized on the cached skill dicts
#                        (was ~83ms re-tokenizing 1.5MB of bodies per turn)
#   facts      ~5ms      warm; ~150ms once per process for the entity index
#   backlog    ~2ms      incremental mtime scan (was ~216ms yaml rebuild
#                        every 60s — the main reason facts got starved)
#   sessions   <5ms      temporal queries only
#   vault lex  6ms hot / 80-160ms per novel call; the leg runs 1-4 short
#                        sub-queries and stops issuing new ones near the
#                        deadline, so it lands with whatever it has
#   vault vec  1.1-2.6s  NEVER lands in 300ms. The query embedding
#                        dominates and qmd's embedding cache only brings a
#                        repeated query down to ~1.1s. It runs as a
#                        straggler and its result carries over to the next
#                        turn (VAULT_CARRY_MAX_AGE_S).
# 300ms keeps first-token latency predictable for the voice pipeline.
PREFETCH_BUDGET_MS = 300

# Vault search settings
VAULT_MAX_RESULTS = 5           # top N vault documents to inject
VAULT_SNIPPET_MAX = 500         # chars per vault snippet
VAULT_MIN_SCORE = 0.5           # minimum relevance score (0.3 was too noisy)
VAULT_MIN_QUERY_LEN = 25        # skip vault search for short/vague messages
# Collections the prefetch vault leg searches. facts and skills have
# dedicated workers; personal/people are left to the explicit tool.
# architecture/autonomy/backlog were added 2026-09-03 — 8 of the 20 eval
# queries expect docs there, and the agent's own design docs and task notes
# are exactly what it needs mid-task. Warm lex cost 6→9 collections: +5-30ms.
VAULT_COLLECTIONS = ["memory", "knowledge", "projects", "lloyd", "work", "sessions",
                     "architecture", "autonomy", "backlog"]
VAULT_CARRY_MAX_AGE_S = 900.0   # max age of a carried-over vault result (either leg)
# The lex leg is waited on for at most this long once every other required
# worker has landed. A term the daemon has never seen costs 0.5-1.3s on
# first sight (FTS pages cold) and 6-50ms after — so on the first turn of a
# new topic the leg straggles and its result carries over, like the hybrid
# leg, instead of pinning the turn at the full budget for nothing.
VAULT_LEX_SOFT_WAIT_MS = 150
VAULT_LEX_STRAGGLE_MAX_S = 2.0  # ladder deadline once it is straggling
# Slots in the injected vault section reserved for carried-over hits when
# both fresh and carried results exist. Lex scores are normalized per
# query, so five confident-looking lex misses used to crowd out a correct
# semantic hit carried from the previous turn (2 of 20 eval queries).
VAULT_CARRIED_MIN_SLOTS = 2
# qmd's lex leg is FTS5 with implicit AND: every term must match, `OR` is
# not passed through, and one unmatched term returns nothing (measured
# 2026-09-03: "alfie servo shoulder pid" → 2 hits, "+ zzzqqq" → 0). So the
# in-budget lex leg runs a few SHORT sub-queries and, when one returns
# nothing, drops its lowest-weight term and retries.
VAULT_LEX_MAX_TERMS = 4         # terms per lex sub-query
VAULT_LEX_MIN_TERMS = 2         # ladder floor — don't go below this
VAULT_LEX_MAX_CALLS = 6         # hard cap on daemon round-trips per turn

# Backlog task-ID reference detection. Task IDs in user text ("what's left on
# #294", "302 is resolved") frequently lose the fact-lookup competition
# against entities that share common words ("deep", "prompt"). Backlog ref
# injection is a precision-first parallel source: cast a wide net with a
# regex, filter by live-task existence, inject current title/status/priority
# directly. No dependency on whether a fact entity exists for the task.
# Regex captures `#(\d{2,4})` OR plain 3-4 digit numeric tokens surrounded
# by non-digits (so `20260421` yields no matches — both sides digit).
_TASK_REF_RE = re.compile(r'(?:#(\d{2,4})|(?<!\d)(\d{3,4})(?!\d))')
# Bare (un-hashed) numbers are additionally rejected when the surrounding
# text marks them as a unit or a code rather than a task id. Live task ids
# span 9..387, which collides with HTTP status codes and millisecond values
# ("returned a 302", "PREFETCH_BUDGET_MS = 300"). `#NNN` is never filtered.
_TASK_REF_UNIT_AFTER = re.compile(
    r"^\s?(?:ms|s|secs?|seconds?|mins?|minutes?|hrs?|hours?|px|pt|%|k|kb|mb|gb|tb|"
    r"fps|hz|khz|mhz|ghz|tok|tokens?|chars?|lines?|bytes?|rows?|cols?|files?|"
    r"turns?|msgs?|messages?|items?|entries|results?|skills?|sessions?)\b",
    re.IGNORECASE,
)
_TASK_REF_CTX_BEFORE = re.compile(
    r"(?:\bhttps?|\bstatus|\bcode|\berror|\bport|\bpid|\bgpu|\bcuda|\bversion|"
    r"\bv|\breturn(?:ed|s)?|\bexit(?:ed)?|\bline|\bpage|\bbudget(?:_ms)?|=)"
    r"\s*(?:(?:a|an|the|with|as|is|was|of)\s+)?$",
    re.IGNORECASE,
)
BACKLOG_MAX_REFS = 3            # cap per turn to avoid token bloat
BACKLOG_CACHE_TTL = 60.0        # re-stat the backlog dir every minute
BACKLOG_BODY_EXCERPT_MAX = 300  # chars of task body to include in injection

# Continuation messages don't benefit from a skills_search nudge.
# Matches openers like "ok", "please continue", "yes please", "let's go".
# Only the closed forms of "please …" and "let's …" count: "please review
# the module" and "let's build a new feature" start new work, and the old
# `please\s+\w+` / `let'?s\s+\w+` alternatives swallowed them.
_CONTINUATION_RE = re.compile(
    r"^\s*(ok\.?|okay\.?|yes\.?|yeah\.?|sure\.?|yep\.?|yup\.?|nope\.?|great\.?|"
    r"perfect\.?|alright\.?|got\s+it|sounds\s+good|"
    r"please\s+(?:continue|proceed|go\s+ahead|carry\s+on|keep\s+going)|"
    r"continue|proceed|carry\s+on|"
    r"let'?s\s+(?:go|continue|proceed|carry\s+on|keep\s+going|move\s+on|do\s+(?:it|that|this))|"
    r"go\s+ahead|thank\s+you|thanks\.?|cool\.?|"
    r"we'?re\s+back|we'?re\s+good|that'?s\s+(right|correct|good|fine)|"
    r"makes\s+sense|sounds\s+right)\b",
    re.IGNORECASE,
)

# Conversation focus tracking
FOCUS_DECAY = 0.75              # decay factor per turn for keyword weights
FOCUS_TOP_K = 6                 # top N focus keywords to enrich vault queries
FOCUS_MIN_WEIGHT = 0.3          # minimum weight to include a keyword
FOCUS_EXTRACT_INTERVAL = 5      # extract secondary-model topics every N turns
FOCUS_MAX_SESSIONS = 50         # max tracked sessions (LRU eviction)

# Noise words for focus keyword extraction, tuned for conversational
# messages rather than search queries.
_FOCUS_NOISE = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "its", "his", "her",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "how", "when", "where", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "between",
    "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    "just", "also", "very", "really", "quite", "too", "much",
    "ok", "okay", "yeah", "yes", "nah", "sure", "right",
    "lets", "let", "go", "going", "get", "got", "getting",
    "want", "wants", "wanted", "know", "knows", "knew",
    "think", "thinks", "thought", "look", "looking", "looked",
    "take", "takes", "took", "make", "makes", "made",
    "now", "some", "any", "all", "each", "every", "both",
    "up", "out", "over", "down", "off", "away",
    "here", "there", "thing", "things", "stuff",
    "left", "done", "next", "back", "ready", "still", "already",
    "tell", "show", "give", "put", "run", "running", "ran",
    "come", "came", "see", "saw", "seen", "say", "said",
    "try", "tried", "use", "used", "using", "work", "working", "worked",
    "start", "started", "stop", "stopped", "keep", "kept",
    "set", "check", "handle", "update", "updated", "well", "good",
    "bit", "lot", "way", "something", "anything", "everything",
    "like", "going", "first", "last", "new", "old", "one", "two",
}


def _extract_focus_keywords(text: str) -> list[str]:
    """Extract content-bearing keywords from a message, filtering noise.

    Trailing punctuation is stripped so a sentence-final word ("servo.")
    accumulates onto the same key as its mid-sentence form ("servo").
    Internal dots/hyphens are kept (config.yaml, plan-mode).
    """
    words = re.findall(r"[a-z][a-z0-9_\-\.]{1,}", text.lower())
    out = []
    for w in words:
        w = w.rstrip(".-_")
        if len(w) >= 2 and w not in _FOCUS_NOISE:
            out.append(w)
    return out


class SessionFocus:
    """Tracks conversation focus for a single session via keyword accumulation.

    Instances are touched from the event-loop thread (update, topic
    extraction) and from prefetch worker threads (enrich_query, vault
    carry-over), so every mutation goes through `_lock`.
    """

    __slots__ = ("keyword_weights", "turn_count", "topics", "topics_turn",
                 "last_access", "pending_vault", "_lock")

    def __init__(self):
        self.keyword_weights: dict[str, float] = {}
        self.turn_count: int = 0
        self.topics: list[str] = []       # secondary-model-extracted topic phrases
        self.topics_turn: int = 0          # turn when topics were last attempted
        self.last_access: float = time.monotonic()
        # leg -> (monotonic_ts, results) from vault stragglers (lex and/or
        # hybrid) — consumed by the next turn and merged into its own hits.
        self.pending_vault: dict[str, tuple[float, list[dict]]] = {}
        self._lock = threading.Lock()

    def update(self, text: str):
        """Add keywords from new message, decay old weights."""
        with self._lock:
            self.turn_count += 1
            self.last_access = time.monotonic()

            # Decay all existing weights
            for k in list(self.keyword_weights):
                self.keyword_weights[k] *= FOCUS_DECAY
                if self.keyword_weights[k] < 0.1:
                    del self.keyword_weights[k]

            # Add new keywords with weight 1.0
            for kw in _extract_focus_keywords(text):
                self.keyword_weights[kw] = self.keyword_weights.get(kw, 0) + 1.0

    def _top_keywords_locked(self, n: int) -> list[str]:
        if not self.keyword_weights:
            return []
        sorted_kw = sorted(self.keyword_weights.items(), key=lambda x: -x[1])
        return [kw for kw, w in sorted_kw[:n] if w >= FOCUS_MIN_WEIGHT]

    def top_keywords(self, n: int = FOCUS_TOP_K) -> list[str]:
        """Return top N keywords by accumulated weight."""
        with self._lock:
            return self._top_keywords_locked(n)

    def enrich_query(self, text: str) -> str:
        """Combine current message with accumulated focus for better vault search.

        If the message is short/vague, focus keywords provide the missing context.
        If the message is already specific, focus keywords reinforce the topic.
        Keywords already present in the message are not appended again —
        the old version re-added every word of the current message as its
        own "focus", doubling the query for no gain.
        """
        parts = [text]
        text_tokens = set(re.findall(r"\w+", text.lower()))
        with self._lock:
            topics = list(self.topics)
            top = self._top_keywords_locked(FOCUS_TOP_K)

        # Secondary-model topics first (highest quality signal), then keywords
        if topics:
            parts.append(" ".join(topics))
        top = [k for k in top if k not in text_tokens]
        if top:
            parts.append(" ".join(top))

        return " ".join(parts)

    def lex_subqueries(self, text: str) -> list[list[str]]:
        """Short term lists for the AND-semantics lex leg, best first.

        1. The message's own content terms, highest focus weight first
           (recurring topic words win ties), capped at VAULT_LEX_MAX_TERMS.
        2. The top focus keywords, when they differ from (1) — this is what
           gives "what about the D term?" its servo/shoulder context.
        3. Up to two secondary-model topic phrases.
        """
        msg_terms = _lex_terms(text)
        with self._lock:
            weights = dict(self.keyword_weights)
            top = self._top_keywords_locked(VAULT_LEX_MAX_TERMS * 3)
            topics = list(self.topics[:2])
        order = {t: i for i, t in enumerate(msg_terms)}
        msg_terms.sort(key=lambda t: (-weights.get(t, 0.0), order[t]))
        queries: list[list[str]] = []
        if msg_terms:
            queries.append(msg_terms[:VAULT_LEX_MAX_TERMS])
        # Prior-turn context: the strongest focus keywords NOT in this
        # message. Fresh words carry weight 1.0 and decayed ones ≤0.75, so
        # the raw top-K is dominated by the current message and would just
        # repeat sub-query 1.
        msg_set = set(msg_terms)
        prior = [k for k in top if k not in msg_set][:VAULT_LEX_MAX_TERMS]
        if prior:
            queries.append(prior)
        for phrase in topics:
            terms = _lex_terms(phrase)
            if terms and all(set(terms) != set(q) for q in queries):
                queries.append(terms[:VAULT_LEX_MAX_TERMS])
        return queries

    def needs_topic_extraction(self) -> bool:
        """Should we fire a secondary-model topic extraction call?"""
        return (
            self.turn_count >= 3  # need enough context
            and (self.turn_count - self.topics_turn) >= FOCUS_EXTRACT_INTERVAL
        )

    def mark_topic_attempt(self):
        """Record that an extraction call was made, whatever it returned.

        Without this, an empty/failed extraction left `topics_turn` at its
        old value and `needs_topic_extraction()` re-fired the model call on
        every subsequent turn instead of every FOCUS_EXTRACT_INTERVAL.
        """
        with self._lock:
            self.topics_turn = self.turn_count

    def set_topics(self, topics: list[str]):
        """Set secondary-model-extracted topics."""
        with self._lock:
            self.topics = topics[:5]
            self.topics_turn = self.turn_count

    def stash_vault(self, results: list[dict], leg: str = "hybrid"):
        """Store a vault leg's result for the next turn."""
        with self._lock:
            self.pending_vault[leg] = (time.monotonic(), results)

    def take_vault(self, max_age_s: float = VAULT_CARRY_MAX_AGE_S) -> list[dict]:
        """Pop and merge the carried-over vault results that are still fresh.
        Hybrid first (it is the superset), lex fills in anything it missed."""
        with self._lock:
            pv = self.pending_vault
            self.pending_vault = {}
        now = time.monotonic()
        merged: dict[str, dict] = {}
        for leg in ("hybrid", "lex"):
            item = pv.get(leg)
            if not item or (now - item[0]) > max_age_s:
                continue
            for r in item[1]:
                key = r.get("file") or r.get("title") or ""
                if key not in merged or r.get("score", 0) > merged[key].get("score", 0):
                    merged[key] = r
        return sorted(merged.values(), key=lambda r: -float(r.get("score", 0) or 0))


# Per-session focus state (in-memory, lost on restart — that's fine)
_session_focus: dict[str, SessionFocus] = {}
_focus_lock = threading.Lock()


def _get_session_focus(session_id: str | None) -> SessionFocus | None:
    """Get or create focus tracker for a session. Returns None if no session_id."""
    if not session_id:
        return None

    with _focus_lock:
        # LRU eviction
        if len(_session_focus) >= FOCUS_MAX_SESSIONS and session_id not in _session_focus:
            oldest = min(_session_focus, key=lambda k: _session_focus[k].last_access)
            del _session_focus[oldest]

        focus = _session_focus.get(session_id)
        if focus is None:
            focus = _session_focus[session_id] = SessionFocus()
        return focus


# ── Skill cache ───────────────────────────────────────────────────────────────
#
# The skill list is rebuilt only when a SKILL.md changes on disk. A
# signature of (dir name, SKILL.md mtime) over both skill roots costs ~1-2ms
# for ~280 entries and is re-checked at most every _SKILL_CACHE_CHECK_S, so
# edits show up within seconds and unchanged trees never pay the ~140ms
# reload (+ the one-off tokenization memo, see skills._skill_token_sets).

_skill_cache: list[dict] = []
_skill_cache_sig: tuple = ()
_skill_cache_checked: float = 0.0
_SKILL_CACHE_CHECK_S = 15.0
_skill_cache_lock = threading.Lock()
_SKILL_CACHE_TTL = _SKILL_CACHE_CHECK_S  # legacy name, kept for callers/docs


def _skills_signature() -> tuple:
    sig = []
    for d in SKILLS_DIRS:
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name.startswith(".") or not e.is_dir():
                        continue
                    try:
                        mt = os.stat(os.path.join(e.path, "SKILL.md")).st_mtime_ns
                    except OSError:
                        mt = -1
                    sig.append((e.name, mt))
        except OSError:
            continue
    sig.sort()
    return tuple(sig)


def _get_skills_cached() -> list[dict]:
    global _skill_cache, _skill_cache_sig, _skill_cache_checked
    now = time.monotonic()
    if _skill_cache and now - _skill_cache_checked < _SKILL_CACHE_CHECK_S:
        return _skill_cache
    with _skill_cache_lock:
        now = time.monotonic()
        if _skill_cache and now - _skill_cache_checked < _SKILL_CACHE_CHECK_S:
            return _skill_cache
        sig = _skills_signature()
        if sig != _skill_cache_sig or not _skill_cache:
            _skill_cache = list(_iter_skills())
            _skill_cache_sig = sig
        _skill_cache_checked = now
        return _skill_cache


# ── Backlog index cache ───────────────────────────────────────────────────────
#
# task_id -> {title, status, priority, board, body_excerpt}. The directory is
# re-stat'ed every BACKLOG_CACHE_TTL seconds (~1-2ms for ~330 files) and only
# files whose mtime changed are re-read and re-parsed. The previous
# implementation re-read and yaml-parsed every file each rebuild (~216ms of
# GIL-held CPU landing inside the prefetch budget once a minute).

_backlog_id_cache: dict[int, dict] = {}
_backlog_id_cache_ts: float = 0.0
_backlog_file_cache: dict[str, tuple[int, int, dict]] = {}  # name -> (mtime_ns, tid, entry)
_backlog_lock = threading.Lock()
_BACKLOG_NAME_RE = re.compile(r'^(\d+)[-_].*\.md$')


def _parse_backlog_frontmatter(content: str) -> tuple[dict, str]:
    """Silent YAML-frontmatter split. Unlike agent_mcp.backlog.parse_frontmatter,
    this does NOT print to stderr on parse failures (several backlog files
    have malformed YAML and we rebuild the index every BACKLOG_CACHE_TTL — a
    noisy warning per rebuild would flood server.err). Returns ({}, content)
    on any parse error so the body-level title extraction still works."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        import yaml
        fm = yaml.safe_load(parts[1]) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    return fm, parts[2].strip()


def _parse_backlog_file(path: str) -> dict:
    """Read one backlog task file into an index entry."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    fm, body = _parse_backlog_frontmatter(content)
    # Title = first non-empty line of body, stripping leading "# "
    title = ""
    body_lines = body.splitlines()
    for line in body_lines:
        stripped = line.strip()
        if stripped:
            title = stripped.lstrip("#").strip()
            break
    if not title:
        title = Path(path).stem
    # Body excerpt: everything after the title line, up to
    # BACKLOG_BODY_EXCERPT_MAX chars.
    body_rest = "\n".join(body_lines[1:]).strip() if body_lines else ""
    return {
        "title":        title,
        "status":       fm.get("status", "?"),
        "priority":     fm.get("priority", "?"),
        "board":        fm.get("board", "?"),
        "body_excerpt": body_rest[:BACKLOG_BODY_EXCERPT_MAX],
    }


def _get_backlog_index() -> dict[int, dict]:
    """Return {task_id: {title, status, priority, board, body_excerpt}} for
    all live backlog tasks. Re-stat'ed every BACKLOG_CACHE_TTL seconds and
    re-parsed per changed file; stale cache retained on rescan failure
    rather than breaking prefetch."""
    global _backlog_id_cache, _backlog_id_cache_ts, _backlog_file_cache
    now = time.monotonic()
    if now - _backlog_id_cache_ts <= BACKLOG_CACHE_TTL and _backlog_id_cache:
        return _backlog_id_cache
    with _backlog_lock:
        now = time.monotonic()
        if now - _backlog_id_cache_ts <= BACKLOG_CACHE_TTL and _backlog_id_cache:
            return _backlog_id_cache
        try:
            backlog_dir = Path.home() / "obsidian" / "backlog"
            new_cache: dict[int, dict] = {}
            new_files: dict[str, tuple[int, int, dict]] = {}
            if backlog_dir.exists():
                with os.scandir(backlog_dir) as it:
                    for e in it:
                        m = _BACKLOG_NAME_RE.match(e.name)
                        if not m or not e.is_file():
                            continue
                        try:
                            mtime = e.stat().st_mtime_ns
                        except OSError:
                            continue
                        prev = _backlog_file_cache.get(e.name)
                        if prev is not None and prev[0] == mtime:
                            _, tid, entry = prev
                        else:
                            tid = int(m.group(1))
                            try:
                                entry = _parse_backlog_file(e.path)
                            except Exception:
                                continue
                        new_files[e.name] = (mtime, tid, entry)
                        new_cache[tid] = entry
            _backlog_id_cache = new_cache
            _backlog_file_cache = new_files
            _backlog_id_cache_ts = now
        except Exception:
            pass  # keep stale cache rather than break prefetch
        return _backlog_id_cache


# ── Search helpers ────────────────────────────────────────────────────────────

def _search_skills(query_tokens: set[str]) -> list[tuple[float, dict]]:
    """Return scored skills sorted descending.

    Uses the metadata-hit-required scoring (see skills._score_skill). A skill
    with zero name/desc/tag match scores 0.0 regardless of body accidents —
    fixes #311 where generic stopword queries pulled powerpoint/youtube skills
    into graph-classifier sessions.
    """
    scored = []
    for skill in _get_skills_cached():
        score = _score_skill(skill, query_tokens, require_metadata_hit=True)
        if score >= SKILL_THRESHOLD_FIRST:
            scored.append((score, skill))
    scored.sort(key=lambda x: -x[0])
    return scored


def _task_ref_candidates(text: str) -> set[int]:
    """Task-id candidates in `text`. `#NNN` always qualifies; a bare number
    is rejected when its neighbours mark it as a unit or a code."""
    candidates: set[int] = set()
    for m in _TASK_REF_RE.finditer(text):
        if m.group(1):
            candidates.add(int(m.group(1)))
            continue
        raw = m.group(2)
        if not raw:
            continue
        before = text[max(0, m.start() - 24):m.start()]
        after = text[m.end():m.end() + 12]
        if _TASK_REF_UNIT_AFTER.match(after) or _TASK_REF_CTX_BEFORE.search(before):
            continue
        candidates.add(int(raw))
    return candidates


def _search_backlog_refs(text: str) -> list[str]:
    """Detect task-ID references in `text` and return formatted lines for
    any that correspond to live backlog tasks. Unknown numbers are dropped
    silently — existence against the backlog is the precision gate."""
    candidates = _task_ref_candidates(text)
    if not candidates:
        return []
    index = _get_backlog_index()
    hits = sorted(tid for tid in candidates if tid in index)
    if not hits:
        return []
    lines = []
    for tid in hits[:BACKLOG_MAX_REFS]:
        t = index[tid]
        title = (t.get("title") or "").strip()
        if len(title) > 100:
            title = title[:97] + "..."
        line = (
            f'- [Task #{tid}] "{title}" — {t["status"]}, {t["priority"]} priority, '
            f'{t["board"]} board'
        )
        excerpt = (t.get("body_excerpt") or "").strip()
        if excerpt:
            if len(excerpt) >= BACKLOG_BODY_EXCERPT_MAX:
                excerpt = excerpt[:BACKLOG_BODY_EXCERPT_MAX] + "…"
            line += f"\n  {excerpt}"
        lines.append(line)
    return lines


def _search_facts(query: str) -> list[str]:
    """Return fact bullet lines for top matching entities."""
    lines = []
    entity_matches = _extract_entities_from_query(query)[:FACT_MAX_ENTITIES]
    for entity, _ in entity_matches:
        result = _get_facts_sync(entity)
        facts = result.get("facts", [])
        # Sort by confidence descending, take top N
        facts.sort(key=lambda f: f.get("confidence", 0.0), reverse=True)
        for f in facts[:FACT_MAX_PER_ENTITY]:
            fact_text = f.get("fact", "").strip()
            conf = f.get("confidence", 0.0)
            if fact_text:
                lines.append(f"- [{entity}] {fact_text} (confidence: {conf})")
    return lines


def _lex_terms(text: str) -> list[str]:
    """Content terms of `text` for the lex leg: qmd's stopword strip plus
    the conversational noise list, deduped, order preserved."""
    stripped = _qmd_strip_stopwords(text)
    out: list[str] = []
    seen: set[str] = set()
    for w in re.findall(r"\w+", stripped.lower()):
        if w in _FOCUS_NOISE or w in seen or (len(w) < 2 and not w.isdigit()):
            continue
        seen.add(w)
        out.append(w)
    return out


def _search_vault(query: str, focus: SessionFocus | None = None,
                  legs: tuple[str, ...] = ("lex", "vec"),
                  min_query_len: int = VAULT_MIN_QUERY_LEN,
                  lex_query: str | None = None) -> list[dict]:
    """Search vault via QMD daemon for cross-session context.

    If a SessionFocus is provided, enriches the query with accumulated
    conversation keywords — this lets short messages like "what about the
    PID gains?" inherit context from the broader conversation about servos.

    `legs` picks the qmd search legs. Prefetch runs the lex leg inside its
    budget (see `_search_vault_lex`) and the full hybrid as a straggler
    whose result is carried to the next turn — the vec leg costs 1.1-2.6s.
    """
    # Enrich query with conversation focus
    effective_query = focus.enrich_query(query) if focus else query

    # Skip vault search for short/vague messages with no focus context
    if len(effective_query.strip()) < min_query_len:
        return []
    try:
        # Prefetch is latency-critical — skip the reranker in favour of
        # RRF-only ranking. Explicit vault_recall calls keep rerank on for
        # higher quality. See _qmd_daemon_search for phase breakdown.
        results = _qmd_daemon_search(effective_query, VAULT_MAX_RESULTS,
                                      VAULT_COLLECTIONS, skip_rerank=True,
                                      legs=legs, lex_query=lex_query)
        if not results:
            return []
        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", "")[:VAULT_SNIPPET_MAX],
                "score": r.get("score", 0),
                "file": r.get("file", ""),
            }
            for r in results
            if r.get("score", 0) >= VAULT_MIN_SCORE
        ]
    except Exception:
        return []  # Non-fatal


# Don't start another lex round-trip when fewer than this many seconds of
# the budget remain — a novel call costs 80-160ms and would drop the whole
# leg, including hits already in hand.
VAULT_LEX_DEADLINE_MARGIN_S = 0.09


def _search_vault_lex(text: str, focus: SessionFocus | None,
                      deadline: float | None = None) -> list[dict]:
    """In-budget lexical leg. Runs a few short AND sub-queries (see
    SessionFocus.lex_subqueries) and, when one returns nothing, drops its
    lowest-weight term and retries down to VAULT_LEX_MIN_TERMS. Results are
    merged by file with the best score kept. Bounded by
    VAULT_LEX_MAX_CALLS daemon round-trips (6ms hot, 80-160ms for a novel
    term set, serialized at the daemon) and by `deadline` (monotonic
    seconds): no new call starts within VAULT_LEX_DEADLINE_MARGIN_S of it,
    so the leg returns partial results instead of overrunning the budget."""
    queries = focus.lex_subqueries(text) if focus else (
        [_lex_terms(text)[:VAULT_LEX_MAX_TERMS]] if _lex_terms(text) else []
    )
    merged: dict[str, dict] = {}
    calls = 0
    out_of_time = False
    for terms in queries:
        n = len(terms)
        # A sub-query that only has one term IS the query ("qmd",
        # "vault_recall") — the floor only stops a longer query from being
        # laddered down to a single generic word.
        floor = min(VAULT_LEX_MIN_TERMS, len(terms))
        while n >= floor and n >= 1 and calls < VAULT_LEX_MAX_CALLS:
            if deadline is not None and time.monotonic() > deadline - VAULT_LEX_DEADLINE_MARGIN_S:
                out_of_time = True
                break
            calls += 1
            hits = _search_vault(" ".join(terms[:n]), None, ("lex",), min_query_len=1)
            if hits:
                for h in hits:
                    key = h.get("file") or h.get("title") or ""
                    if key not in merged or h.get("score", 0) > merged[key].get("score", 0):
                        merged[key] = h
                break
            n -= 1
        if calls >= VAULT_LEX_MAX_CALLS or out_of_time:
            break
    out = sorted(merged.values(), key=lambda r: -float(r.get("score", 0) or 0))
    return out[:VAULT_MAX_RESULTS]


def _search_vault_lex_and_stash(text: str, focus: SessionFocus | None,
                                deadline: float | None = None) -> list[dict]:
    """`_search_vault_lex` that also stashes its result on `focus`, so a
    cold-term ladder that misses the soft wait still reaches the next turn."""
    res = _search_vault_lex(text, focus, deadline)
    if focus is not None:
        focus.stash_vault(res, leg="lex")
    return res


def _search_vault_hybrid_and_stash(query: str, focus: SessionFocus | None,
                                   after=None) -> list[dict]:
    """Full lex+vec vault search. Stashes its result on `focus` *before*
    returning so the outcome is deterministic either way: if it lands
    inside the budget the caller consumes the stash immediately; if it
    straggles, the next turn picks the stash up as a carry-over.

    The lex leg gets the message's short term list (AND semantics), the
    vec leg the focus-enriched sentence — see `_qmd_daemon_search`.

    `after` is the lex-leg future. The qmd daemon is a single node process
    that handles requests serially — a lex query fired 50ms *after* a vec
    query waits ~1.2s behind it (measured 2026-09-03), which would push the
    fast leg past the budget too. Waiting for the lex leg to return first
    guarantees it gets the daemon to itself.
    """
    if after is not None:
        try:
            after.result(timeout=VAULT_LEX_STRAGGLE_MAX_S + 1.0)
        except Exception:
            pass  # lex leg failed or timed out — still worth running hybrid
    lex_q = None
    subs = focus.lex_subqueries(query) if focus is not None else []
    if subs:
        lex_q = " ".join(subs[0])
    res = _search_vault(query, focus, legs=("lex", "vec"), lex_query=lex_q)
    if focus is not None:
        focus.stash_vault(res)
    return res


def _merge_vault_results(fresh: list[dict], carried: list[dict]) -> list[dict]:
    """Union of this turn's hits and the carried-over ones, deduped by file,
    capped at VAULT_MAX_RESULTS. Carried entries are flagged so the
    rendering can say where they came from.

    Ordering: by score, except that up to VAULT_CARRIED_MIN_SLOTS carried
    hits are guaranteed a place when there are any — lex scores are
    normalized per query, so a full page of confident-looking lex misses
    must not evict the one semantic hit the previous turn found.
    """
    seen: set[str] = set()
    fresh_u: list[dict] = []
    for r in fresh:
        key = r.get("file") or r.get("title") or ""
        if key in seen:
            continue
        seen.add(key)
        fresh_u.append(r)
    carried_u: list[dict] = []
    for r in carried:
        key = r.get("file") or r.get("title") or ""
        if key in seen:
            continue
        seen.add(key)
        carried_u.append({**r, "carried": True})
    score = lambda r: -float(r.get("score", 0) or 0)  # noqa: E731
    fresh_u.sort(key=score)
    carried_u.sort(key=score)
    if not carried_u:
        return fresh_u[:VAULT_MAX_RESULTS]
    reserved = min(VAULT_CARRIED_MIN_SLOTS, len(carried_u))
    head = fresh_u[:VAULT_MAX_RESULTS - reserved] + carried_u[:reserved]
    rest = sorted(fresh_u[VAULT_MAX_RESULTS - reserved:] + carried_u[reserved:], key=score)
    out = head + rest[:max(0, VAULT_MAX_RESULTS - len(head))]
    out.sort(key=score)
    return out


# ── Session search ───────────────────────────────────────────────────────────

# Temporal query patterns — triggers session search in prefetch
_TEMPORAL_RE = re.compile(
    r"\b(today|yesterday|earlier|last\s+session|this\s+morning|this\s+afternoon|"
    r"this\s+evening|what\s+did\s+we|what\s+have\s+we|recent(?:ly)?|"
    r"previous(?:ly)?|before\s+this|last\s+time|worked\s+on|talked\s+about|"
    r"discussed|decided|earlier\s+today|what\s+was)\b", re.I
)

SESSION_PREFETCH_LIMIT = 3
SESSION_PREFETCH_SNIPPET_MAX = 200
SESSION_PREFETCH_DAYS = 3


def _search_recent_sessions(query: str) -> list[dict]:
    """Search recent sessions for prefetch context injection.

    Only fires for temporal queries ('today', 'yesterday', 'what did we', etc.)
    to avoid adding session noise to non-temporal queries.
    """
    if not _TEMPORAL_RE.search(query):
        return []

    try:
        import datetime as _dt

        index = _load_session_index(max_days=SESSION_PREFETCH_DAYS)
        cutoff = (_dt.datetime.now() - _dt.timedelta(days=SESSION_PREFETCH_DAYS)).strftime("%Y%m%d")
        sessions = [s for s in index.values() if s["date_str"] >= cutoff]

        if not sessions:
            return []

        # Tokenize query for scoring
        query_tokens = {w for w in re.findall(r"\w+", query.lower())
                        if w not in _FOCUS_NOISE and w not in _ENTITY_STOPWORDS and len(w) >= 2}

        if not query_tokens:
            # Pure temporal query like "what did we work on today?" — return most recent
            sessions.sort(key=lambda s: s["date_str"] + s.get("time_str", ""), reverse=True)
            return sessions[:SESSION_PREFETCH_LIMIT]

        # Score sessions, with a baseline boost since the query is temporal
        scored = []
        for s in sessions:
            score = _score_session(s, query_tokens) + 0.3  # temporal baseline
            scored.append((score, s))
        scored.sort(key=lambda x: -x[0])

        return [s for _, s in scored[:SESSION_PREFETCH_LIMIT]]
    except Exception:
        return []


# ── Context block formatting ──────────────────────────────────────────────────

def _format_context(skills: list[tuple[float, dict]], fact_lines: list[str],
                    vault_results: list[dict] = None,
                    session_results: list[dict] = None,
                    ambient_entries: list = None,
                    backlog_refs: list[str] = None,
                    show_skill_hint: bool = True) -> str:
    parts = []

    # Ambient prefetch drain — background signals from autonomy/cron/etc.
    # Rendered first so the agent sees them as current-but-passive context
    # BEFORE skills/facts/vault. Agent is free to reference or ignore.
    # (#295 Mechanism 1)
    if ambient_entries:
        amb_lines = []
        for entry in ambient_entries:
            line = f"- **[{entry.source}]** {entry.summary}"
            if entry.content:
                # Indent content so it's clearly nested under the summary
                body = entry.content[:800]
                if len(entry.content) > 800:
                    body += "…"
                line += f"\n  > {body}"
            amb_lines.append(line)
        parts.append(
            "<ambient-signals>\n"
            "Background producers queued these signals for you. The user did NOT ask — "
            "reference them only if naturally relevant to what they're saying now.\n"
            + "\n".join(amb_lines)
            + "\n</ambient-signals>"
        )

    # First skill: full body
    if skills:
        score, skill = skills[0]
        body = skill["raw"][:SKILL_BODY_MAX]
        if len(skill["raw"]) > SKILL_BODY_MAX:
            body += "\n[... truncated]"
        parts.append(f'<skill name="{skill["name"]}" score="{score:.1f}">\n{body}\n</skill>')

    # Second skill: excerpt only
    if len(skills) >= 2 and skills[1][0] >= SKILL_THRESHOLD_SECOND:
        score2, skill2 = skills[1]
        excerpt = skill2["raw"][:SKILL_EXCERPT_MAX]
        if len(skill2["raw"]) > SKILL_EXCERPT_MAX:
            excerpt += "\n[... truncated]"
        parts.append(f'<skill name="{skill2["name"]}" score="{score2:.1f}" excerpt="true">\n{excerpt}\n</skill>')

    # Backlog refs — live task-summary lookups for `#NNN` / bare-number
    # references in the user message. More authoritative than fact-store
    # snapshots (current title/status/priority) so rendered ahead of facts.
    if backlog_refs:
        parts.append("<backlog-refs>\n" + "\n".join(backlog_refs) + "\n</backlog-refs>")

    if fact_lines:
        parts.append("<facts>\n" + "\n".join(fact_lines) + "\n</facts>")

    # Vault search results — cross-session context from daily notes, knowledge, projects
    if vault_results:
        vault_lines = []
        for vr in vault_results:
            title = vr.get("title", "")
            snippet = vr.get("snippet", "").strip()
            score = vr.get("score", 0)
            if not snippet:
                continue
            if vr.get("carried"):
                vault_lines.append(
                    f"- **{title}** (score: {score:.2f}, semantic hit from the previous turn's query): {snippet}"
                )
            else:
                vault_lines.append(f"- **{title}** (score: {score:.2f}): {snippet}")
        if vault_lines:
            parts.append("<vault-context>\n" + "\n".join(vault_lines) + "\n</vault-context>")

    # Recent session context — cross-session recall for temporal queries
    if session_results:
        session_lines = []
        for sr in session_results:
            # `created_at` is usually an ISO string; some session-recall sources
            # return it as a unix-timestamp float, which crashes the slice
            # below and aborts the whole turn. Coerce defensively.
            created = str(sr.get("created_at", "") or "")[:16]
            preview = str(sr.get("preview", "") or "")[:SESSION_PREFETCH_SNIPPET_MAX]
            msg_count = sr.get("message_count", 0)
            model = sr.get("model", "")
            snippets = sr.get("user_snippets", [])[:2]
            snippet_text = " | ".join(s[:150] for s in snippets) if snippets else preview
            session_lines.append(f"- [{created}] ({model}, {msg_count} msgs): {snippet_text}")
        if session_lines:
            parts.append("<recent-sessions>\n" + "\n".join(session_lines) + "\n</recent-sessions>")

    # Skill-hint nudge: fires only when no skill matched AND the message is a
    # genuine new task (not a continuation). Low-confidence auto-picks are
    # trusted by the model ~95% of the time regardless of a hint, so the
    # low-conf branch was dropped (pure noise, ~65 tokens/firing).
    if show_skill_hint and not skills:
        parts.append(
            "<skill-hint>No skills matched automatically. "
            "If this looks like a repeatable workflow, call skills_search "
            "before proceeding.</skill-hint>"
        )

    if not parts:
        return ""

    return "<context>\n" + "\n".join(parts) + "\n</context>"


# ── Public API ────────────────────────────────────────────────────────────────

def _read_plan_mode(session_id: str | None) -> bool:
    """Fallback for callers that don't pass `plan_mode`: read it off disk."""
    if not session_id:
        return False
    try:
        from app.paths import SESSIONS_DIR as _SD
        _meta = _SD / f"{session_id}.json"
        if _meta.exists():
            _data = json.loads(_meta.read_text())
            return bool((_data.get("plan") or {}).get("plan_mode"))
    except Exception:
        pass  # non-fatal; prefetch continues with plain query_tokens
    return False


def _prefetch_prepare(text: str, session_id: str | None,
                      plan_mode: bool | None) -> tuple | None:
    """Cheap, loop-thread-safe half of prefetch: ambient drain, focus
    update, plan-mode flag. Returns None when there is nothing to do
    (message too short and no ambient entries), else the tuple that
    `_prefetch_run` consumes.

    Kept on the caller's thread on purpose: the ambient queue is mutated
    by producers on the event loop, and the focus tracker's turn counter
    should advance in request order.
    """
    # Drain ambient queue FIRST — even a short message should surface
    # queued background context. Don't let the MIN_MESSAGE_LEN guard
    # suppress injections the producer already decided were worth showing.
    ambient_entries = []
    if session_id:
        try:
            from app.sessions_io import drain_ambient_prefetch
            ambient_entries = drain_ambient_prefetch(session_id)
        except Exception:
            ambient_entries = []  # Non-fatal

    if len(text.strip()) < MIN_MESSAGE_LEN and not ambient_entries:
        return None

    # Update conversation focus tracker
    focus = _get_session_focus(session_id)
    if focus:
        focus.update(text)

    if plan_mode is None:
        plan_mode = _read_plan_mode(session_id)

    return ambient_entries, focus, bool(plan_mode)


def _prefetch_run(text: str, ambient_entries: list, focus: SessionFocus | None,
                  plan_mode: bool) -> str:
    """Blocking half of prefetch: budgeted parallel search + formatting.
    Safe to run in a worker thread."""
    query_tokens = _query_tokens(text)

    # Plan B — synthetic skill-match tokens. When the session is in
    # plan_mode, augment the query with tokens that match plan-mode-authoring
    # (and the related `plan` / `writing-plans` skills) so the playbook
    # for drafting a high-quality plan auto-loads as <context> regardless
    # of what the user actually typed. Same shape as the natural query
    # tokens — the matcher doesn't need to know they're synthetic.
    if plan_mode:
        # `plan-mode-authoring` tags include `plan-mode` and `planning`;
        # name tokenizes to {plan, mode, authoring}. Adding these matches
        # it strongly (name 3.0× + tags 1.5×).
        query_tokens = query_tokens | {
            "plan", "mode", "authoring", "planning",
            "plan-mode", "plan-mode-authoring",
        }

    # Run skill, fact, vault, session, and backlog-ref search in parallel
    skills_result: list[tuple[float, dict]] = []
    facts_result: list[str] = []
    vault_lex_result: list[dict] = []
    vault_hybrid_result: list[dict] | None = None
    session_result: list[dict] = []
    backlog_result: list[str] = []
    dropped: list[str] = []
    landed: dict[str, int] = {}   # worker -> ms after t0 when its result landed
    t0 = time.monotonic()

    # Short messages still get ambient injection but skip the
    # skill/fact/vault search (too little signal to query with).
    if len(text.strip()) >= MIN_MESSAGE_LEN:
        # Note: pool is NOT used as a context manager. `with ThreadPoolExecutor`
        # blocks on __exit__ until every submitted future finishes — which
        # would defeat PREFETCH_BUDGET_MS for the slow vault vec leg.
        # Instead we submit, wait up to the budget, and leave stragglers
        # running in a daemon-like pool that GCs after their threads finish.
        budget_s = PREFETCH_BUDGET_MS / 1000.0
        pool = ThreadPoolExecutor(max_workers=6)
        f_skills = pool.submit(_search_skills, query_tokens)
        f_facts = pool.submit(_search_facts, text)
        f_vault_lex = pool.submit(_search_vault_lex_and_stash, text, focus,
                                  t0 + VAULT_LEX_STRAGGLE_MAX_S)
        f_vault_hybrid = pool.submit(_search_vault_hybrid_and_stash, text, focus, f_vault_lex)
        f_sessions = pool.submit(_search_recent_sessions, text)
        f_backlog = pool.submit(_search_backlog_refs, text)
        named = (
            ("skills", f_skills), ("facts", f_facts), ("vault", f_vault_lex),
            ("vault_vec", f_vault_hybrid), ("sessions", f_sessions),
            ("backlog", f_backlog),
        )

        # Block at most `budget_s` total across the *required* futures. Any
        # still running at the deadline is abandoned for this turn. The
        # hybrid vault leg is not required: it is the carry-over producer
        # and is expected to straggle, so waiting on it would pin every
        # turn at the full budget (which is exactly what the old
        # single-vault-future design did — 20/20 logged turns at 300ms+).
        pending = {f for name, f in named if name != "vault_vec"}
        soft_s = VAULT_LEX_SOFT_WAIT_MS / 1000.0
        while pending:
            elapsed = time.monotonic() - t0
            remaining = budget_s - elapsed
            if pending == {f_vault_lex}:
                # Everything else is in; give a cold lex ladder only the
                # soft wait, then let it straggle and carry over.
                remaining = min(remaining, soft_s - elapsed)
            if remaining <= 0:
                break
            done, pending = wait(pending, timeout=remaining,
                                 return_when=FIRST_COMPLETED)
            if not done:
                break  # timed out
            now_ms = int((time.monotonic() - t0) * 1000)
            for future in done:
                for name, f in named:
                    if f is future:
                        landed[name] = now_ms
                try:
                    if future is f_skills:
                        skills_result = future.result()
                    elif future is f_facts:
                        facts_result = future.result()
                    elif future is f_vault_lex:
                        vault_lex_result = future.result()
                    elif future is f_vault_hybrid:
                        vault_hybrid_result = future.result()
                    elif future is f_sessions:
                        session_result = future.result()
                    elif future is f_backlog:
                        backlog_result = future.result()
                except Exception:
                    pass  # Individual failure — non-fatal

        # If the hybrid leg happened to finish while we waited on the
        # others (fast daemon, cached embedding), take it now.
        if f_vault_hybrid.done():
            try:
                vault_hybrid_result = f_vault_hybrid.result()
            except Exception:
                vault_hybrid_result = None

        # Record which required subtasks missed the budget for
        # observability. Both vault legs are allowed to straggle (their
        # results carry over), so they are reported in the debug line only.
        for name, fut in named:
            if not fut.done() and name not in ("vault_vec", "vault"):
                dropped.append(name)

        # shutdown(wait=False) lets stragglers (vault vec leg) complete
        # in the background without blocking the caller. Python will GC
        # the pool once every thread returns.
        pool.shutdown(wait=False)

        if dropped:
            logger.info("prefetch budget=%dms exceeded, dropped=%s",
                        PREFETCH_BUDGET_MS, ",".join(dropped))

    # Vault: this turn's best landed leg (hybrid is a superset of lex) plus
    # whatever the previous turn's stragglers stashed. A leg that landed
    # this turn also stashed itself; the merge dedupes by file so those
    # copies vanish rather than showing up as "carried".
    fresh = vault_hybrid_result if vault_hybrid_result is not None else vault_lex_result
    carried = focus.take_vault() if focus is not None else []
    vault_result = _merge_vault_results(fresh, carried)
    carried = [r for r in vault_result if r.get("carried")]

    # Suppress the no-match skill hint for continuation messages ("ok", "yes
    # please", "let's go", etc.) — they're never starting a new workflow, so
    # the nudge is pure noise.  Also suppress for very short messages (<15
    # chars) that lack enough signal to warrant a skills_search prompt.
    is_continuation = (
        len(text.strip()) < 15
        or bool(_CONTINUATION_RE.match(text.strip()))
    )

    context = _format_context(skills_result, facts_result, vault_result,
                              session_result, ambient_entries=ambient_entries,
                              backlog_refs=backlog_result,
                              show_skill_hint=not is_continuation)

    # IDE state — what folder/file the user has open in the IDE tab. Tiny,
    # always-fresh, helps the agent answer "what file am I looking at?"
    # without needing a tool call. Folded into the existing <context>
    # envelope when present so it sits next to skills/facts/vault.
    ide_block = _format_ide_state()
    if ide_block:
        if context:
            # Splice <ide_state>...</ide_state> just before the closing
            # </context> tag so the agent sees it as part of the same
            # injection block.
            context = context.rsplit("</context>", 1)
            context = context[0] + ide_block + "\n</context>"
        else:
            context = "<context>\n" + ide_block + "\n</context>"

    logger.debug(
        "prefetch %dms: skills=%d facts=%d backlog=%d vault=%d(%s%s) sessions=%d ambient=%d landed=%s",
        int((time.monotonic() - t0) * 1000), len(skills_result), len(facts_result),
        len(backlog_result), len(vault_result),
        "hybrid" if vault_hybrid_result is not None else ("lex" if "vault" in landed else "lex-straggling"),
        f"+{len(carried)} carried" if carried else "",
        len(session_result), len(ambient_entries),
        " ".join(f"{k}@{v}ms" for k, v in sorted(landed.items(), key=lambda kv: kv[1])),
    )

    if not context:
        return text

    return context + "\n\n" + text


def prefetch_context(text: str, session_id: str | None = None,
                     plan_mode: bool | None = None) -> str:
    """
    Return the user message with a <context> block prepended if relevant
    skill/fact/vault content was found. Returns original text unchanged if
    nothing matches or the message is too short to bother.

    If session_id is provided, maintains conversation focus state to improve
    vault search quality across turns. Short/vague messages benefit from
    accumulated topic keywords. Also drains the session's ambient prefetch
    queue (#295 Mechanism 1) so background producer signals are folded
    into this turn's context.

    `plan_mode` lets a caller that already loaded the session pass the flag
    through; when None it is read from the session JSON.

    Blocks the calling thread for up to PREFETCH_BUDGET_MS. From coroutine
    code use `prefetch_context_async` instead.
    """
    prep = _prefetch_prepare(text, session_id, plan_mode)
    if prep is None:
        return text
    return _prefetch_run(text, *prep)


async def prefetch_context_async(text: str, session_id: str | None = None,
                                 plan_mode: bool | None = None) -> str:
    """`prefetch_context` for coroutine callers: the cheap bookkeeping runs
    on the loop thread (in request order), the budgeted search phase runs
    in a worker thread so the event loop keeps serving other sessions,
    SSE streams, and Inner Voice while we wait."""
    prep = _prefetch_prepare(text, session_id, plan_mode)
    if prep is None:
        return text
    return await asyncio.to_thread(_prefetch_run, text, *prep)


def _format_ide_state() -> str:
    """Render the current IDE block as a compact tag, or "" if none.

    Reads the in-memory MC state mirror — same process as the FastAPI
    backend, no HTTP round-trip.
    """
    try:
        from app.mc_state import get_ide_snapshot
        snap = get_ide_snapshot()
    except Exception:
        return ""
    if not snap:
        return ""
    open_folder = snap.get("open_folder")
    visible_file = snap.get("visible_file")
    open_tabs = snap.get("open_tabs") or []
    lines = ["<ide_state>"]
    if open_folder:
        lines.append(f"  open_folder: {open_folder}")
    if visible_file:
        lines.append(f"  visible_file: {visible_file}")
    if open_tabs:
        lines.append(f"  open_tabs: [{', '.join(open_tabs)}]")
    if len(lines) == 1:
        return ""
    lines.append("</ide_state>")
    return "\n".join(lines)
