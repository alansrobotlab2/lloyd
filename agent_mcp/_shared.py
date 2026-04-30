"""Shared helpers for agent_mcp knowledge-graph and vault servers.

Extracted from agent_mcp/memory.py as part of Task #340 PR 1. No behavior
change — pure module split for maintainability.

Contents:
    - Path constants (VAULT, FACTS_ROOT, ALIASES_PATH)
    - Stopword sets (_ENTITY_STOPWORDS, _SCORING_STOPWORDS, _QUERY_STOPWORDS)
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
from typing import Literal, Optional

import yaml

# ── Path constants ───────────────────────────────────────────────────────────

VAULT = Path.home() / "obsidian"
FACTS_ROOT = VAULT / "facts"
ALIASES_PATH = FACTS_ROOT / "entity-aliases.json"


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
