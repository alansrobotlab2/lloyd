#!/usr/bin/env python3
"""Shared retrieval core for the knowledge graph.

Owns the pieces both `agent_mcp.facts` (the fact tools) and
`agent_mcp.vault` (vault_recall) need:

    - Entity extraction from a query (extract_entities_from_query)
    - The relationships index with mtime-based caching
      (load_relationships / save_relationships / invalidate_relationships_cache)
    - Entity-to-entity graph traversal + weighted expansion
      (graph_expand_entities / graph_weighted_neighbors)
    - Query-aware fact reading and ranking
      (get_facts_sync / fact_query_tokens / fact_matches_tokens / fact_score)

Extracted from facts.py (2026-06-11 architecture review, Tier 2.1) —
vault.py previously imported six underscore-private functions from
facts.py, an invisible contract that made any facts refactor a silent
vault_recall breakage. These names are the public API; treat changes to
their signatures as cross-module breaking changes.
"""

import json
import math
import re
import time
from typing import Optional

from agent_mcp._shared import (
    FACTS_ROOT,
    atomic_write_text,
    _FACT_QUERY_STOPWORDS,
    _SCORING_STOPWORDS,
    _find_entity_dir,
    _get_entity_dirs_cached,
    _parse_fact_frontmatter,
    _resolve_entity,
)

# ── Constants ────────────────────────────────────────────────────────────────

# Edge-type weights for weighted graph expansion in vault_recall.
# Typed semantic edges dominate; cooccurrence-style edges are down-weighted so
# they still contribute but don't drown out real relationships.
# Keep this in sync with the vocabulary emitted by the relation classifier
# (scripts/memory/classify-relationships.py, Phase 1B of backlog #294).
EDGE_TYPE_WEIGHTS = {
    # semantic, high-confidence
    "uses": 1.0,
    "depends_on": 1.0,
    "implements": 1.0,
    "supersedes": 0.9,
    "part_of": 0.85,
    "created_by": 0.8,
    "discusses": 0.75,
    "competes_with": 0.7,
    "related_to": 0.6,
    # cooccurrence / weak signals
    "wiki_link_co_occurrence": 0.4,
    "co_mentioned": 0.35,
    "mentions": 0.3,
}
_DEFAULT_EDGE_WEIGHT = 0.3  # unknown types fall back to same weight as mentions

# Task-ID extractor. Matches "Task #299", "Task 299", "#299", "task_310",
# "backlog_120", "backlog_item_18", "backlog_task_41". Captures the numeric
# ID so we can dispatch to whichever naming convention exists.
_TASK_ID_RE = re.compile(
    r"(?:\btask\s*[#_ ]?|#|\bbacklog[_ -]?(?:item[_ -]?|task[_ -]?)?)(\d{1,4})\b",
    re.IGNORECASE,
)

# Cache for entity degree (non-expired edge count) — used as a deterministic
# tie-break signal in extract_entities_from_query. 60s TTL keeps the hot
# prefetch path fast without going fully stale.
_edge_count_cache: Optional[tuple] = None
_EDGE_COUNT_TTL = 60

RELATIONSHIPS_PATH = FACTS_ROOT / "_relationships.json"

# In-memory cache for the relationships index (#340 PR 2).
#
# Layout: (mtime_ns, parsed_data) | None
#
# Invalidation strategy:
#   - Reads stat() the file and compare mtime_ns. Mismatch → reload.
#   - save_relationships() refreshes the cache with the new mtime.
#   - This handles both in-process mutation (load → mutate → save) and
#     cross-process writes (autonomy classifier writes file, MCP server
#     picks up via stat-check on next read).
#
# Mutation contract: callers that mutate the returned dict MUST follow
# with save_relationships(). Between load and save, the cache and the
# caller share the same object — there's no defensive deep-copy because
# the MCP server is single-threaded and the file is small enough to
# parse but large enough (2.3 MB / 4.6k edges) that copying defeats
# the cache.
_relationships_cache: Optional[tuple[int, dict]] = None

# Fact ranking config (Fix A+C for #322).
# God-node threshold: entities with more than this many facts are treated as
# buckets of loosely-related debris. A query-token match is required to pull
# any of their facts.
FACT_GODNODE_THRESHOLD = 50

# How many ranked facts to return from seed entities and from graph-expanded
# neighbors respectively.
FACT_RANK_CAP_SEED = 10
FACT_RANK_CAP_GRAPH = 5


# ── Fact reading ─────────────────────────────────────────────────────────────

def get_facts_sync(entity: str, category: str = None, as_of: str = None,
                   include_expired: bool = False) -> dict:
    resolved, _ = _resolve_entity(entity, mode="read")
    entity_dir = _find_entity_dir(resolved)
    if not entity_dir:
        return {"error": f"Entity not found: {entity}", "facts": []}
    facts = []
    if category:
        fact_file = entity_dir / f"{resolved}-{category}.md"
        if not fact_file.exists():
            fact_file = entity_dir / f"{entity}-{category}.md"
        if fact_file.exists():
            frontmatter = _parse_fact_frontmatter(fact_file.read_text(encoding="utf-8"))
            facts = frontmatter.get("facts", [])
    else:
        for fact_file in entity_dir.glob("*.md"):
            frontmatter = _parse_fact_frontmatter(fact_file.read_text(encoding="utf-8"))
            facts.extend(frontmatter.get("facts", []))
    if not include_expired:
        if as_of:
            # Return facts valid at a specific point in time
            facts = [f for f in facts
                     if (not f.get("valid_at") or f["valid_at"] <= as_of)
                     and (not f.get("expired_at") or f["expired_at"] > as_of)
                     and (not f.get("invalid_at") or f["invalid_at"] > as_of)]
        else:
            # Default: only current facts
            facts = [f for f in facts if not f.get("expired_at") and not f.get("invalid_at")]
    return {"entity": resolved, "category": category, "facts": facts}


def get_entity_edge_counts() -> dict:
    global _edge_count_cache
    now = time.monotonic()
    if _edge_count_cache is not None and (now - _edge_count_cache[0]) < _EDGE_COUNT_TTL:
        return _edge_count_cache[1]
    counts: dict[str, int] = {}
    rel_path = FACTS_ROOT / "_relationships.json"
    if rel_path.exists():
        try:
            data = json.loads(rel_path.read_text(encoding="utf-8"))
            for edge in data.get("edges", []):
                if edge.get("expired_at") or edge.get("invalid_at"):
                    continue
                s, t = edge.get("source"), edge.get("target")
                if s:
                    counts[s] = counts.get(s, 0) + 1
                if t:
                    counts[t] = counts.get(t, 0) + 1
        except Exception:
            pass
    _edge_count_cache = (now, counts)
    return counts


# ── Entity extraction ────────────────────────────────────────────────────────

def extract_entities_from_query(query: str) -> list:
    """Rank known entities by how well they match the query.

    Prior implementation (pre-#312) scored binary 2-or-3 with tie-breaks
    resolved by arbitrary dict iteration order. With FACT_MAX_ENTITIES=2
    downstream that meant legitimate entities got dropped on ties. Trace
    evidence from #306 showed ~50% weak-match rate.

    New scoring:
      - Task-ID references (#299, backlog_18, task_310) dispatch to canonical
        `Task #N` / legacy forms with a fixed high score.
      - Full-name substring (entity name appears verbatim in query) gets a
        strong bonus scaled by length.
      - Token overlap is scored (overlap^2) / (entity_tokens * query_tokens)
        so 1-token matches on multi-token entities don't win.
      - Ties broken deterministically: score → longer canonical → higher
        graph degree → alphabetical.
    """
    if not FACTS_ROOT.exists():
        return []

    q_lower = query.lower()
    q_tokens = {
        w for w in re.findall(r"\b\w+\b", q_lower)
        if w not in _SCORING_STOPWORDS and len(w) >= 2
    }

    entities = _get_entity_dirs_cached()
    entity_lookup = {e.lower(): e for e in entities}

    scores: dict[str, float] = {}

    def _bump(name: str, score: float) -> None:
        # Resolve to canonical case if we have a directory for it.
        canonical = entity_lookup.get(name.lower(), name)
        if scores.get(canonical, 0.0) < score:
            scores[canonical] = score

    # 1. Task-ID direct dispatch (highest-priority signal).
    for m in _TASK_ID_RE.finditer(q_lower):
        tid = m.group(1)
        # Try every known naming convention in the vault.
        for candidate in (
            f"Task #{tid}", f"Task {tid}", f"Task_{tid}",
            f"backlog_{tid}", f"backlog_item_{tid}", f"backlog_task_{tid}",
            tid,
        ):
            hit = entity_lookup.get(candidate.lower())
            if hit:
                _bump(hit, 10.0)

    # 2. Full-name match at word boundaries.
    for e_lower, e_cased in entity_lookup.items():
        if len(e_lower) < 3:
            continue
        if re.search(r"(?<!\w)" + re.escape(e_lower) + r"(?!\w)", q_lower):
            _bump(e_cased, 5.0 + min(len(e_lower) / 20.0, 2.0))

    # 3. Token-overlap scoring. Reward specificity on both sides.
    if q_tokens:
        q_norm = max(len(q_tokens), 2)
        for e_lower, e_cased in entity_lookup.items():
            e_tokens = {
                w for w in re.findall(r"\b\w+\b", e_lower)
                if w not in _SCORING_STOPWORDS and len(w) >= 2
            }
            if not e_tokens:
                continue
            overlap = e_tokens & q_tokens
            if not overlap:
                continue
            score = (len(overlap) ** 2) / (len(e_tokens) * q_norm)
            if len(e_tokens) == 1 and len(overlap) == 1:
                score = max(score, 0.5)
            if score >= 0.25:
                _bump(e_cased, score)

    if not scores:
        return []

    edge_counts = get_entity_edge_counts()
    ranked = sorted(
        scores.items(),
        key=lambda kv: (
            -kv[1],
            -len(kv[0]),
            -edge_counts.get(kv[0], 0),
            kv[0].lower(),
        ),
    )
    return ranked


# ── Relationship store ───────────────────────────────────────────────────────

def load_relationships() -> dict:
    """Load the relationships index, with mtime-based caching.

    Returns the parsed dict. On missing file or parse error, returns the
    empty schema and clears the cache.
    """
    global _relationships_cache
    if not RELATIONSHIPS_PATH.exists():
        _relationships_cache = None
        return {"edges": [], "schema_version": 1}
    try:
        mtime_ns = RELATIONSHIPS_PATH.stat().st_mtime_ns
    except OSError:
        return {"edges": [], "schema_version": 1}
    if _relationships_cache is not None and _relationships_cache[0] == mtime_ns:
        return _relationships_cache[1]
    try:
        data = json.loads(RELATIONSHIPS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"edges": [], "schema_version": 1}
    _relationships_cache = (mtime_ns, data)
    return data


def save_relationships(data: dict) -> None:
    """Persist the relationships index and refresh the cache."""
    global _relationships_cache
    RELATIONSHIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        RELATIONSHIPS_PATH,
        json.dumps(data, indent=2, sort_keys=False),
        fsync=True,
    )
    try:
        new_mtime = RELATIONSHIPS_PATH.stat().st_mtime_ns
        _relationships_cache = (new_mtime, data)
    except OSError:
        _relationships_cache = None


def invalidate_relationships_cache() -> None:
    """Clear the relationships cache. Used by tests and forced reloads."""
    global _relationships_cache
    _relationships_cache = None


# ── Graph expansion ──────────────────────────────────────────────────────────

def graph_expand_entities(seed_entities: list[str], hops: int = 1) -> list[str]:
    """Expand a set of seed entities via relationship graph traversal."""
    if not RELATIONSHIPS_PATH.exists():
        return []
    try:
        data = load_relationships()
        adj: dict[str, set[str]] = {}
        for edge in data["edges"]:
            if edge.get("expired_at"):
                continue
            adj.setdefault(edge["source"], set()).add(edge["target"])
            adj.setdefault(edge["target"], set()).add(edge["source"])
        expanded = set()
        current = set(seed_entities)
        for _ in range(hops):
            next_layer = set()
            for entity in current:
                for neighbor in adj.get(entity, set()):
                    if neighbor not in seed_entities and neighbor not in expanded:
                        next_layer.add(neighbor)
            expanded.update(next_layer)
            current = next_layer
        return list(expanded)
    except Exception:
        return []


def graph_weighted_neighbors(
    seed_entities: list[str], top_k: int = 3, hops: int = 1
) -> list[tuple[str, float]]:
    """Weighted graph expansion: return top-k neighbors scored by
    edge confidence × EDGE_TYPE_WEIGHTS[type].

    Multiple edges to the same neighbor sum their contributions so entities
    connected by multiple typed relationships rise to the top. Seed entities
    are excluded from results.

    Returns [(entity, weight)] sorted by weight desc.
    """
    if not RELATIONSHIPS_PATH.exists() or not seed_entities:
        return []
    try:
        data = load_relationships()
    except Exception:
        return []

    adj: dict[str, list[tuple[str, str, float]]] = {}
    for edge in data.get("edges", []):
        if edge.get("expired_at"):
            continue
        src, tgt, etype = edge.get("source"), edge.get("target"), edge.get("type", "")
        conf = float(edge.get("confidence", 0.5))
        if not src or not tgt:
            continue
        adj.setdefault(src, []).append((tgt, etype, conf))
        adj.setdefault(tgt, []).append((src, etype, conf))

    seed_set = set(seed_entities)
    scores: dict[str, float] = {}
    current = set(seed_entities)
    visited = set(seed_entities)

    for hop in range(hops):
        hop_decay = 1.0 if hop == 0 else 0.5 ** hop
        next_layer = set()
        for entity in current:
            for neighbor, etype, conf in adj.get(entity, []):
                if neighbor in seed_set:
                    continue
                w = EDGE_TYPE_WEIGHTS.get(etype, _DEFAULT_EDGE_WEIGHT) * conf * hop_decay
                # No cap here — accumulate raw evidence weight first so multiple
                # typed edges to the same neighbor compound. God-node penalty
                # applied below.
                scores[neighbor] = scores.get(neighbor, 0.0) + w
                if neighbor not in visited:
                    next_layer.add(neighbor)
        visited.update(next_layer)
        current = next_layer

    # God-node penalty: divide by log(degree+e) so high-degree entities
    # (e.g. "lloyd" with ~991 edges, log≈7) don't dominate over specific
    # entities (degree<10, log≈2.5). Without this, broad nodes saturated at 1.0.
    edge_counts = get_entity_edge_counts()
    for entity in list(scores.keys()):
        degree = max(edge_counts.get(entity, 1), 1)
        scores[entity] = scores[entity] / math.log(degree + math.e)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return ranked[:top_k]


# ── Fact ranking ─────────────────────────────────────────────────────────────

def fact_query_tokens(query: str) -> list[str]:
    """Extract scorable tokens from the query. Lowercased, length-3+,
    stopwords stripped. Preserves identifier-like tokens (with _, -, #)."""
    if not query:
        return []
    raw = re.findall(r"[A-Za-z0-9_#-]+", query.lower())
    return [t for t in raw if len(t) >= 3 and t not in _FACT_QUERY_STOPWORDS]


def _fact_blob(fact: dict) -> str:
    """Flatten a fact into searchable lowercased text."""
    parts = []
    for key in ("fact", "category", "provenance", "event_date"):
        val = fact.get(key)
        if val:
            parts.append(str(val))
    return " ".join(parts).lower()


def fact_matches_tokens(fact: dict, tokens: list[str]) -> bool:
    """True if any query token appears in the fact's searchable text."""
    if not tokens:
        return False
    blob = _fact_blob(fact)
    return any(t in blob for t in tokens)


def fact_score(fact: dict, tokens: list[str]) -> float:
    """Fraction of query tokens that appear in the fact's searchable text."""
    if not tokens:
        return 0.0
    blob = _fact_blob(fact)
    hits = sum(1 for t in tokens if t in blob)
    return hits / len(tokens)
