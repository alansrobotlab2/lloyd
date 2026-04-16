#!/usr/bin/env python3
"""
prefetch.py — Automatic context prefetch layer.

Extracts keywords from the user message, searches skills, facts, and
vault documents in parallel, then prepends a <context> block so the
agent gets relevant content without needing explicit tool calls.

Maintains per-session conversation focus via keyword accumulation with
exponential decay. Short/vague messages benefit from accumulated context
about what the conversation is actually about.

Called by server.py before every query() invocation.
"""

import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_mcp.skills import _iter_skills, _score_skill, _tokenize
from agent_mcp.memory import (
    _extract_entities_from_query, _get_facts_sync,
    _qmd_daemon_search, _qmd_strip_stopwords,
    _load_session_index, _score_session, _ENTITY_STOPWORDS,
)

logger = logging.getLogger("lloyd.prefetch")

# ── Tuning ────────────────────────────────────────────────────────────────────

SKILL_THRESHOLD_FIRST = 3.0     # minimum score to inject first skill (full body)
SKILL_THRESHOLD_SECOND = 4.0    # minimum score to inject second skill (excerpt only)
SKILL_BODY_MAX = 6000           # chars, first skill
SKILL_EXCERPT_MAX = 500         # chars, second skill
FACT_MAX_ENTITIES = 2           # top N entities to look up
FACT_MAX_PER_ENTITY = 3         # top N facts per entity (by confidence)
MIN_MESSAGE_LEN = 10            # skip prefetch for very short messages

# Vault search settings
VAULT_MAX_RESULTS = 5           # top N vault documents to inject
VAULT_SNIPPET_MAX = 500         # chars per vault snippet
VAULT_MIN_SCORE = 0.5           # minimum relevance score (0.3 was too noisy)
VAULT_MIN_QUERY_LEN = 25        # skip vault search for short/vague messages
VAULT_COLLECTIONS = ["memory", "knowledge", "projects", "lloyd", "work", "sessions"]  # skip facts (entity lookup), skills (skill search)

# Conversation focus tracking
FOCUS_DECAY = 0.75              # decay factor per turn for keyword weights
FOCUS_TOP_K = 6                 # top N focus keywords to enrich vault queries
FOCUS_MIN_WEIGHT = 0.3          # minimum weight to include a keyword
FOCUS_EXTRACT_INTERVAL = 5      # extract 35B topics every N turns
FOCUS_MAX_SESSIONS = 50         # max tracked sessions (LRU eviction)

# Noise words for focus keyword extraction (superset of subliminal's list,
# tuned for conversational messages rather than search queries)
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
    """Extract content-bearing keywords from a message, filtering noise."""
    words = re.findall(r"[a-z][a-z0-9_\-\.]{1,}", text.lower())
    return [w for w in words if w not in _FOCUS_NOISE]


class SessionFocus:
    """Tracks conversation focus for a single session via keyword accumulation."""

    __slots__ = ("keyword_weights", "turn_count", "topics", "topics_turn", "last_access")

    def __init__(self):
        self.keyword_weights: dict[str, float] = {}
        self.turn_count: int = 0
        self.topics: list[str] = []       # 35B-extracted topic phrases
        self.topics_turn: int = 0          # turn when topics were last set
        self.last_access: float = time.monotonic()

    def update(self, text: str):
        """Add keywords from new message, decay old weights."""
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

    def top_keywords(self, n: int = FOCUS_TOP_K) -> list[str]:
        """Return top N keywords by accumulated weight."""
        if not self.keyword_weights:
            return []
        sorted_kw = sorted(self.keyword_weights.items(), key=lambda x: -x[1])
        return [kw for kw, w in sorted_kw[:n] if w >= FOCUS_MIN_WEIGHT]

    def enrich_query(self, text: str) -> str:
        """Combine current message with accumulated focus for better vault search.

        If the message is short/vague, focus keywords provide the missing context.
        If the message is already specific, focus keywords reinforce the topic.
        """
        parts = [text]

        # Add 35B-extracted topics if available (highest quality signal)
        if self.topics:
            parts.append(" ".join(self.topics))

        # Add top accumulated keywords
        top = self.top_keywords()
        if top:
            parts.append(" ".join(top))

        enriched = " ".join(parts)
        return enriched

    def needs_topic_extraction(self) -> bool:
        """Should we fire a 35B topic extraction call?"""
        return (
            self.turn_count >= 3  # need enough context
            and (self.turn_count - self.topics_turn) >= FOCUS_EXTRACT_INTERVAL
        )

    def set_topics(self, topics: list[str]):
        """Set 35B-extracted topics."""
        self.topics = topics[:5]
        self.topics_turn = self.turn_count


# Per-session focus state (in-memory, lost on restart — that's fine)
_session_focus: dict[str, SessionFocus] = {}


def _get_session_focus(session_id: str | None) -> SessionFocus | None:
    """Get or create focus tracker for a session. Returns None if no session_id."""
    if not session_id:
        return None

    # LRU eviction
    if len(_session_focus) > FOCUS_MAX_SESSIONS and session_id not in _session_focus:
        oldest = min(_session_focus, key=lambda k: _session_focus[k].last_access)
        del _session_focus[oldest]

    if session_id not in _session_focus:
        _session_focus[session_id] = SessionFocus()
    return _session_focus[session_id]


# Simple skill cache: list of loaded skill dicts, refreshed every 5 min
_skill_cache: list[dict] = []
_skill_cache_ts: float = 0.0
_SKILL_CACHE_TTL = 300.0


def _get_skills_cached() -> list[dict]:
    global _skill_cache, _skill_cache_ts
    now = time.monotonic()
    if now - _skill_cache_ts > _SKILL_CACHE_TTL:
        _skill_cache = list(_iter_skills())
        _skill_cache_ts = now
    return _skill_cache


# ── Search helpers ────────────────────────────────────────────────────────────

def _search_skills(query_tokens: set[str]) -> list[tuple[float, dict]]:
    """Return scored skills sorted descending."""
    scored = []
    for skill in _get_skills_cached():
        score = _score_skill(skill, query_tokens)
        if score >= SKILL_THRESHOLD_FIRST:
            scored.append((score, skill))
    scored.sort(key=lambda x: -x[0])
    return scored


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


def _search_vault(query: str, focus: SessionFocus | None = None) -> list[dict]:
    """Search vault via QMD daemon for cross-session context.

    If a SessionFocus is provided, enriches the query with accumulated
    conversation keywords — this lets short messages like "what about the
    PID gains?" inherit context from the broader conversation about servos.
    """
    # Enrich query with conversation focus
    effective_query = focus.enrich_query(query) if focus else query

    # Skip vault search for short/vague messages with no focus context
    if len(effective_query.strip()) < VAULT_MIN_QUERY_LEN:
        return []
    try:
        results = _qmd_daemon_search(effective_query, VAULT_MAX_RESULTS, VAULT_COLLECTIONS)
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
                    had_query: bool = True) -> str:
    parts = []

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

    if fact_lines:
        parts.append("<facts>\n" + "\n".join(fact_lines) + "\n</facts>")

    # Vault search results — cross-session context from daily notes, knowledge, projects
    if vault_results:
        vault_lines = []
        for vr in vault_results:
            title = vr.get("title", "")
            snippet = vr.get("snippet", "").strip()
            score = vr.get("score", 0)
            if snippet:
                vault_lines.append(f"- **{title}** (score: {score:.2f}): {snippet}")
        if vault_lines:
            parts.append("<vault-context>\n" + "\n".join(vault_lines) + "\n</vault-context>")

    # Recent session context — cross-session recall for temporal queries
    if session_results:
        session_lines = []
        for sr in session_results:
            created = sr.get("created_at", "")[:16]
            preview = sr.get("preview", "")[:SESSION_PREFETCH_SNIPPET_MAX]
            msg_count = sr.get("message_count", 0)
            model = sr.get("model", "")
            snippets = sr.get("user_snippets", [])[:2]
            snippet_text = " | ".join(s[:150] for s in snippets) if snippets else preview
            session_lines.append(f"- [{created}] ({model}, {msg_count} msgs): {snippet_text}")
        if session_lines:
            parts.append("<recent-sessions>\n" + "\n".join(session_lines) + "\n</recent-sessions>")

    # No skill matched but the message was long enough to warrant a search —
    # nudge the agent to consider an explicit skills_search call.
    if not skills and had_query:
        parts.append(
            "<skill-hint>No skills matched automatically. "
            "If this task involves a repeatable workflow, consider calling "
            "skills_search to check for applicable skills.</skill-hint>"
        )

    if not parts:
        return ""

    return "<context>\n" + "\n".join(parts) + "\n</context>"


# ── Public API ────────────────────────────────────────────────────────────────

def prefetch_context(text: str, session_id: str | None = None) -> str:
    """
    Return the user message with a <context> block prepended if relevant
    skill/fact/vault content was found. Returns original text unchanged if
    nothing matches or the message is too short to bother.

    If session_id is provided, maintains conversation focus state to improve
    vault search quality across turns. Short/vague messages benefit from
    accumulated topic keywords.
    """
    if len(text.strip()) < MIN_MESSAGE_LEN:
        return text

    # Update conversation focus tracker
    focus = _get_session_focus(session_id)
    if focus:
        focus.update(text)

    query_tokens = _tokenize(text)

    # Run skill, fact, vault, and session search in parallel
    skills_result: list[tuple[float, dict]] = []
    facts_result: list[str] = []
    vault_result: list[dict] = []
    session_result: list[dict] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_skills = pool.submit(_search_skills, query_tokens)
        f_facts = pool.submit(_search_facts, text)
        f_vault = pool.submit(_search_vault, text, focus)
        f_sessions = pool.submit(_search_recent_sessions, text)
        for future in as_completed([f_skills, f_facts, f_vault, f_sessions]):
            try:
                if future is f_skills:
                    skills_result = future.result()
                elif future is f_facts:
                    facts_result = future.result()
                elif future is f_vault:
                    vault_result = future.result()
                else:
                    session_result = future.result()
            except Exception:
                pass  # Prefetch failures are non-fatal

    context = _format_context(skills_result, facts_result, vault_result,
                              session_result, had_query=True)
    if not context:
        return text

    return context + "\n\n" + text
