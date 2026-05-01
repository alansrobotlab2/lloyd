"""Shared helpers for agent_mcp knowledge-graph and vault servers.

Extracted from agent_mcp/memory.py as part of Task #340 PR 1. No behavior
change — pure module split for maintainability.

Contents:
    - Path constants (VAULT, FACTS_ROOT, ALIASES_PATH)
    - Stopword sets (_ENTITY_STOPWORDS, _SCORING_STOPWORDS, _QUERY_STOPWORDS,
      _FACT_QUERY_STOPWORDS, _SKILLS_QUERY_STOPWORDS)
    - Pure helpers (_token_overlap, _levenshtein, _fuzzy_entity_match)
    - Fact frontmatter helpers (_parse_fact_frontmatter, _write_fact_frontmatter)
    - Entity resolution (_find_entity_dir, _load_aliases, _save_aliases,
      _get_entity_dirs_cached, _resolve_entity)
    - Cache invalidation (_invalidate_entity_dirs_cache)

Anything that touches the relationships index, qmd daemon, vault audit log,
session memory injection patterns, or fact ranking stays in its owning module
(facts/vault/session after PR 5).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

import yaml
from mcp.types import TextContent

# ── Path constants ───────────────────────────────────────────────────────────

VAULT = Path.home() / "obsidian"
FACTS_ROOT = VAULT / "facts"
ALIASES_PATH = FACTS_ROOT / "entity-aliases.json"


# ── Result types & dispatch helpers (#340 PR 4) ──────────────────────────────
#
# All handlers in agent_mcp.facts / .vault / .session return a Python dict.
# The per-module call_tool dispatcher wraps each return in TextContent via
# _wrap() — json.dumps happens exactly once per call, not 3+ times per
# handler.
#
# Shape policy:
#   - Success: handler-specific data, NOT wrapped in an envelope. The agent
#     reads tool output as text; an extra {"ok": true, "data": {...}} layer
#     adds visual noise on every successful call. Pre-existing success
#     shapes ({"facts": [...]}, {"path": ..., "text": ...}, etc.) are
#     preserved verbatim.
#
#   - Error: standardized via _err() — always {"error": str, "code": str,
#     ...extra}. Extra keys preserve pre-existing companion fields like
#     {"facts": []} that callers may expect on the error path.
#
# Why no Result envelope:
#   The original audit (#340 background) proposed
#   {"ok": bool, "data": ..., "error": ..., "code": ...}. PR 4's callsite
#   audit found zero programmatic consumers (prefetch.py uses internal
#   helpers below the json layer; post_capture.py discards the return).
#   The only "consumer" is the LLM agent reading the JSON visually, where
#   verbosity is a tax. We get the type-safety, single-dumps, and code-
#   field wins from this approach without taxing every successful call.

class Result(TypedDict, total=False):
    """Type contract for MCP handler returns.

    All handlers in facts.py / vault.py / session.py return Result dicts
    (Python dicts). The dispatcher serializes via _wrap() exactly once.

    Two valid shapes:
      Success: arbitrary handler-specific keys (no envelope).
      Error:   {"error": str, "code": str, ...extra}

    Use _err() to construct errors so the shape stays consistent.
    """
    error: str
    code: str


class ErrorCode:
    """Standardized error-code constants for handler returns.

    Use these instead of inline strings so callers can branch on the code
    without parsing the human-readable error message.
    """
    MISSING_PARAM = "MISSING_PARAM"        # Required parameter omitted/empty
    INVALID_PARAM = "INVALID_PARAM"        # Parameter present but malformed
    NOT_FOUND = "NOT_FOUND"                # Entity/file/resource doesn't exist
    PATH_ESCAPE = "PATH_ESCAPE"            # Path traversal outside vault
    INJECTION = "INJECTION"                # Prompt-injection guardrail tripped
    NO_MATCH = "NO_MATCH"                  # Substring not found in target
    INTERNAL = "INTERNAL"                  # Caught exception during handler
    UNKNOWN_TOOL = "UNKNOWN_TOOL"          # Dispatcher could not route name


def _err(message: str, code: str = ErrorCode.INTERNAL, **extra: Any) -> dict:
    """Construct a standardized error result.

    Example:
        return _err("entity is required", ErrorCode.MISSING_PARAM, facts=[])

    The order is fixed: error first, code second, then any caller-supplied
    extras (e.g. empty list/dict companions that pre-existing callers may
    expect on the error path).
    """
    return {"error": message, "code": code, **extra}


def _wrap(result: dict) -> list[TextContent]:
    """Serialize a handler return to a TextContent envelope.

    Exactly one json.dumps per tool call. default=str handles datetimes
    and other non-JSON-native types that may slip through (e.g. event_date
    values from YAML-parsed fact frontmatter).
    """
    return [TextContent(type="text", text=json.dumps(result, default=str))]


# ── Stopword sets ────────────────────────────────────────────────────────────

_ENTITY_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "what", "which", "who", "how", "when", "where", "why",
}

# Scoring stopwords — broader than _ENTITY_STOPWORDS. "task" and "backlog" as
# bare tokens are noise for the scorer (the task-ID regex handles numeric
# references explicitly).
_SCORING_STOPWORDS = _ENTITY_STOPWORDS | {
    "task", "backlog", "item", "issue", "ticket",
}

# Query-phrasing stopwords (#327) — superset of _ENTITY_STOPWORDS used ONLY
# by qmd lex cleanup (FTS5 BM25 leg). DO NOT use for entity scoring, fact
# ranking, or anywhere a short query token could be a legitimate match target.
_QUERY_STOPWORDS = _ENTITY_STOPWORDS | {
    # Conversational pronouns / self-reference
    "me", "us", "our", "ours", "my", "mine", "your", "yours",
    "myself", "yourself", "ourselves", "them", "their", "theirs",
    # Question-framing verbs (content-empty when used as prefix/filler)
    "walk", "tell", "show", "describe", "explain", "let", "lets",
    "give", "ask", "share", "summarize",
    # Third-person-s question framing verbs (#332). Restricted to -s forms
    # so root-verb uses in content queries remain searchable.
    "happens", "sends", "occurs", "goes",
    # Discourse fillers / intensifiers
    "just", "also", "really", "actually", "basically", "pretty",
    "please", "kindly", "maybe", "probably", "perhaps",
    # NOTE deliberately NOT stripped — can be meaningful in architectural
    # content: enable, handle, return, work, function, operate, build,
    # ship, shipped, run, start, stop, fail, failed.
}

# Fact-side query stopwords (#322 fact ranking). Used by
# agent_mcp.facts._fact_query_tokens to keep fact ranking responsive to
# query-token overlap without aggressive trimming. Lightweight on purpose:
# aggressive stopword removal hurts when the query is short
# ("how does fact_path work"). Kept as a frozenset because it's hot-path.
_FACT_QUERY_STOPWORDS = frozenset({
    "what", "how", "when", "where", "why", "who", "which",
    "the", "and", "for", "with", "from", "into", "over", "under",
    "does", "did", "do", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "can", "could", "will", "would", "should", "may",
    "might", "must", "shall",
    "this", "that", "these", "those", "it", "its", "them", "they", "their",
    "our", "ours", "my", "mine", "your", "yours", "his", "her", "hers",
    "we", "you", "us", "me", "i",
    "about", "also", "just", "than", "then", "there", "here", "through",
    "during", "between", "among", "across", "within", "without", "after",
    "before", "while", "until", "since",
    "of", "in", "on", "to", "at", "by", "as", "or", "nor", "not", "so",
    "if", "else", "but", "yet", "though", "although",
    "tell", "show", "explain", "describe", "walk", "give", "get", "use",
    "using", "used", "make", "made", "see", "seen", "know", "need",
})

# Skill-matching query stopwords. Used by agent_mcp.skills._query_tokens
# to filter conversational fillers from the *query* side of skill scoring
# (skill body side is unfiltered — the asymmetry is deliberate). Per
# the original docstring in skills.py: a query like "lets dig into 311"
# should NOT fire skills just because "lets", "dig", "into" appear in
# arbitrary skill bodies.
#
# Broader than _QUERY_STOPWORDS — includes many more conversational
# tokens (ok, yeah, sure, getting, looked, made, etc.). Distinct purpose,
# distinct shape. Kept as a separate constant rather than aliased.
_SKILLS_QUERY_STOPWORDS = {
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
    "try", "tried", "use", "used", "using",
    "start", "started", "stop", "stopped", "keep", "kept",
    "set", "well", "good",
    "bit", "lot", "way", "something", "anything", "everything",
    "like", "first", "last", "new", "old", "one", "two",
    "dig", "really",
}


# ── Pure helpers ─────────────────────────────────────────────────────────────

def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of word tokens, case-insensitive."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _levenshtein(s1: str, s2: str) -> int:
    """Standard Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _fuzzy_entity_match(name: str, candidates: list[str], threshold: float = 0.85) -> Optional[str]:
    """Fuzzy match an entity name against known entities.

    History: previously used threshold=0.7 with a substring-match boost to 0.8,
    which poisoned the alias table with entries like
      'agent prompt constraint' -> 'agent'
      'autonomy-system.md'      -> 'System'
    because any short canonical name containing a token of the query got the
    substring boost. See backlog #310 Tier 4.

    Changes:
    - Threshold bumped 0.7 → 0.85 (tight similarity required).
    - Substring boost removed entirely — Levenshtein + token overlap drive
      the match.
    - Additional guard: block matches where the length ratio is extreme
      (>2×), which reliably indicates a short canonical swallowing a long
      specific name.
    """
    name_lower = name.lower().strip()
    best_match: Optional[str] = None
    best_score = 0.0
    for candidate in candidates:
        cand_lower = candidate.lower().strip()
        if name_lower == cand_lower:
            return candidate
        # Block extreme length asymmetry — "agent prompt constraint" shouldn't
        # match "agent".
        if name_lower and cand_lower:
            len_ratio = max(len(name_lower), len(cand_lower)) / min(
                len(name_lower), len(cand_lower)
            )
            if len_ratio > 2.0:
                continue
        overlap = _token_overlap(name_lower, cand_lower)
        max_len = max(len(name_lower), len(cand_lower))
        lev_score = 1.0 - (_levenshtein(name_lower, cand_lower) / max_len) if max_len > 0 else 0.0
        combined = 0.4 * overlap + 0.6 * lev_score
        if combined > best_score:
            best_score = combined
            best_match = candidate
    return best_match if best_score >= threshold else None


# ── Fact frontmatter helpers ────────────────────────────────────────────────

def _parse_fact_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from the head of a markdown string.

    Returns {} for any malformed input (no leading ---, no closing ---).
    """
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    return yaml.safe_load(content[3:end]) or {}


def _write_fact_frontmatter(data: dict) -> str:
    """Emit YAML frontmatter block with leading and trailing --- markers."""
    return f"---\n{yaml.dump(data, default_flow_style=False, sort_keys=False)}---\n"


# ── Entity directory cache ──────────────────────────────────────────────────

_entity_dirs_cache: Optional[tuple[float, list[str]]] = None
_ENTITY_DIRS_TTL = 60


def _get_entity_dirs_cached() -> list[str]:
    """List entity directory names under FACTS_ROOT, cached for 60s."""
    global _entity_dirs_cache
    now = time.monotonic()
    if _entity_dirs_cache is not None and (now - _entity_dirs_cache[0]) < _ENTITY_DIRS_TTL:
        return _entity_dirs_cache[1]
    if not FACTS_ROOT.exists():
        _entity_dirs_cache = (now, [])
        return []
    names = [d.name for d in FACTS_ROOT.iterdir() if d.is_dir()]
    _entity_dirs_cache = (now, names)
    return names


def _invalidate_entity_dirs_cache() -> None:
    """Clear the entity-dir cache. Call after creating a new entity dir."""
    global _entity_dirs_cache
    _entity_dirs_cache = None


# ── Entity resolution ───────────────────────────────────────────────────────

def _find_entity_dir(entity: str) -> Optional[Path]:
    """Find an entity directory under FACTS_ROOT, case-insensitive."""
    if not FACTS_ROOT.exists():
        return None
    entity_lower = entity.lower()
    for entry in FACTS_ROOT.iterdir():
        if entry.is_dir() and entry.name.lower() == entity_lower:
            return entry
    return None


def _load_aliases() -> dict:
    """Load the entity alias map from disk. Returns {} on missing/corrupt."""
    if not ALIASES_PATH.exists():
        return {}
    try:
        return json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_aliases(aliases: dict) -> None:
    """Persist the entity alias map. Creates parent dirs as needed."""
    ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIASES_PATH.write_text(
        json.dumps(aliases, indent=2, sort_keys=True), encoding="utf-8"
    )


def _resolve_entity(name: str, *, mode: Literal["read", "write"]) -> tuple[str, bool]:
    """Resolve an entity name to its canonical form.

    Returns (canonical_name, is_new). `is_new` is True when no resolution
    happened (the caller's input is returned verbatim).

    Resolution order depends on mode:

    Read mode (mode="read"):
        1. Exact case-insensitive directory match
        2. Alias table lookup (original case or lowercase)
        3. Fuzzy match against known entities (in-memory only)
        Returns the caller's input verbatim if nothing resolves.
        NO DISK WRITES — pure read-side resolution.

    Write mode (mode="write"):
        1. Exact case-insensitive directory match
        2. Alias table lookup (original case or lowercase)
        Returns the caller's input verbatim if nothing resolves.
        FUZZY MATCHING IS DISABLED. The caller writes to the exact name
        they specified — this is the fix for the silent fuzzy-merge bug.

    Why mode is required (and kw-only):
        Task #340 PR 3 introduces this parameter to fix a silent data-
        corruption bug: previously fuzzy match ran regardless of
        auto_create, so fact_add(entity="Lloyd", ...) could silently
        land on "lloyd-mc" if the fuzzy threshold passed. Reads also
        wrote to the alias table as a side effect.

        Forcing mode= keyword-only and required makes every callsite
        explicit about whether it expects existing data (read) or is
        creating something new (write). No silent default.

    Mode="write" no longer creates the entity directory — that's still
    the caller's responsibility (see _fact_add). It also no longer
    pre-registers an alias mapping, because the caller's input IS the
    canonical name.
    """
    name = name.strip()
    if not name:
        return name, True

    # 1. Exact directory match (both modes).
    entity_dir = _find_entity_dir(name)
    if entity_dir:
        return entity_dir.name, False

    # 2. Alias table lookup (both modes).
    aliases = _load_aliases()
    # Check both original-case and lowercase keys — Tier 1 sweep writes both.
    canonical = aliases.get(name) or aliases.get(name.lower())
    if canonical and _find_entity_dir(canonical):
        return canonical, False

    # 3. Fuzzy match — read mode ONLY, in-memory, no persistence.
    if mode == "read":
        known_entities = _get_entity_dirs_cached()
        fuzzy_match = _fuzzy_entity_match(name, known_entities)
        if fuzzy_match:
            return fuzzy_match, False

    # Nothing resolved — return verbatim, mark as new.
    return name, True
