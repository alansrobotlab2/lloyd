#!/usr/bin/env python3
"""
prefetch.py — Automatic context prefetch layer.

Extracts keywords from the user message, searches skills and facts in
parallel, and prepends a <context> block so the agent gets relevant
skill content and facts without needing to call skills_search first.

Called by server.py before every query() invocation.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add mcp_server to path so we can import helpers directly
sys.path.insert(0, str(Path(__file__).parent / "mcp_server"))

from skills import _iter_skills, _score_skill, _tokenize  # noqa: E402
from memory import _extract_entities_from_query, _get_facts_sync  # noqa: E402

# ── Tuning ────────────────────────────────────────────────────────────────────

SKILL_THRESHOLD_FIRST = 3.0     # minimum score to inject first skill (full body)
SKILL_THRESHOLD_SECOND = 4.0    # minimum score to inject second skill (excerpt only)
SKILL_BODY_MAX = 6000           # chars, first skill
SKILL_EXCERPT_MAX = 500         # chars, second skill
FACT_MAX_ENTITIES = 2           # top N entities to look up
FACT_MAX_PER_ENTITY = 3         # top N facts per entity (by confidence)
MIN_MESSAGE_LEN = 10            # skip prefetch for very short messages

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


# ── Context block formatting ──────────────────────────────────────────────────

def _format_context(skills: list[tuple[float, dict]], fact_lines: list[str]) -> str:
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

    if not parts:
        return ""

    return "<context>\n" + "\n".join(parts) + "\n</context>"


# ── Public API ────────────────────────────────────────────────────────────────

def prefetch_context(text: str) -> str:
    """
    Return the user message with a <context> block prepended if relevant
    skill/fact content was found. Returns original text unchanged if nothing
    matches or the message is too short to bother.
    """
    if len(text.strip()) < MIN_MESSAGE_LEN:
        return text

    query_tokens = _tokenize(text)

    # Run skill search and fact search in parallel
    skills_result: list[tuple[float, dict]] = []
    facts_result: list[str] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_skills = pool.submit(_search_skills, query_tokens)
        f_facts = pool.submit(_search_facts, text)
        for future in as_completed([f_skills, f_facts]):
            try:
                if future is f_skills:
                    skills_result = future.result()
                else:
                    facts_result = future.result()
            except Exception:
                pass  # Prefetch failures are non-fatal

    context = _format_context(skills_result, facts_result)
    if not context:
        return text

    return context + "\n\n" + text
