#!/usr/bin/env python3
"""Shared retrieval core for the knowledge graph.

Owns the pieces both `agent_mcp.facts` (the fact tools) and
`agent_mcp.vault` (vault_recall) need:

    - Entity extraction from a query (extract_entities_from_query)
    - A read view of the edge graph (load_relationships), which since the
      2026-09 migration lives in app.kg_store, not a JSON file
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

import math
import re
from typing import Optional

from app.kg_store import StoreUnavailable, store

from agent_mcp._shared import (
    FACTS_ROOT,
    _FACT_QUERY_STOPWORDS,
    _SCORING_STOPWORDS,
    _find_entity_dir,
    _get_entity_dirs_cached,
    _parse_fact_frontmatter,
    _resolve_entity,
)

# ── Constants ────────────────────────────────────────────────────────────────

# The edge graph is unreadable. Kept as an alias so the guards written for the
# JSON era read the same: a store that will not open is not an empty graph, and
# a writer that treats it as one persists a 3-edge graph over 6,539 real ones.
RelationshipsCorrupt = StoreUnavailable


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

# Degree and adjacency are memoised inside app.kg_store, keyed on the store's
# `PRAGMA data_version` — so a commit from the classifier process invalidates
# this process's cache on the next read. That replaces the 60s TTL and the
# mtime checks the JSON readers used, both of which could serve a stale graph
# for up to a minute after a nightly run rewrote it.

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

# Parsed fact-file cache: path → (mtime_ns, facts).
#
# God-node entities make the uncached path brutal: `QMD` carries 4,328 facts
# and re-parsing its YAML cost 1,338 ms on EVERY vault_recall that seeded on
# it (2026-08-06, #380). FACT_GODNODE_THRESHOLD limits how many of those facts
# reach results, but only after the whole file set has already been parsed.
#
# Invalidated by mtime, same contract as _relationships_cache above.
#
# MUTATION CONTRACT: the returned fact dicts are shared with the cache.
# Callers must not mutate them in place — copy first. `vault._collect` already
# does (`{**f, "entity": ...}`).
_fact_file_cache: dict[str, tuple[int, list]] = {}
_FACT_FILE_CACHE_MAX = 3000


def _read_facts_cached(fact_file) -> list:
    """Parse one fact file's `facts:` list, memoised on (path, mtime_ns)."""
    key = str(fact_file)
    try:
        mtime_ns = fact_file.stat().st_mtime_ns
    except OSError:
        return []
    hit = _fact_file_cache.get(key)
    if hit is not None and hit[0] == mtime_ns:
        return hit[1]
    try:
        frontmatter = _parse_fact_frontmatter(fact_file.read_text(encoding="utf-8"))
    except OSError:
        return []
    facts = frontmatter.get("facts") or []
    if len(_fact_file_cache) >= _FACT_FILE_CACHE_MAX:
        _fact_file_cache.clear()
    _fact_file_cache[key] = (mtime_ns, facts)
    return facts


def invalidate_fact_file_cache() -> None:
    """Drop the parsed fact-file cache (used by writers and tests)."""
    _fact_file_cache.clear()


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
            facts = list(_read_facts_cached(fact_file))
    else:
        for fact_file in entity_dir.glob("*.md"):
            facts.extend(_read_facts_cached(fact_file))
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
    """Entity name → number of active edges touching it (its degree).

    Memoised in the store on `PRAGMA data_version`. Raises
    `StoreUnavailable` when the graph cannot be read — callers that only rank
    with this should use `_edge_counts_or_empty()`.
    """
    return store().edges.degree()


def get_entity_edge_counts_ci() -> dict:
    """Lowercased entity name → summed degree.

    `vault._graph_rerank` keys its voters on the lowercased name, so against
    the cased map every lookup missed, every degree read as 1, and the
    god-node penalty divided every voter by the same constant — a no-op
    dressed as a penalty (2026-09-03 review).
    """
    return store().edges.degree_ci()


def _edge_counts_or_empty(ci: bool = False) -> dict:
    """Degree map for ranking-only call sites, `{}` when the graph is unreadable.

    Degree is a tie-break and a god-node penalty here, never a stored value,
    so a corrupt store should cost ranking quality — not a failed
    `vault_recall`.
    """
    try:
        return get_entity_edge_counts_ci() if ci else get_entity_edge_counts()
    except StoreUnavailable:
        return {}


def invalidate_edge_count_cache() -> None:
    """Drop the store's derived caches (used by writers and tests)."""
    try:
        store().invalidate_caches()
    except StoreUnavailable:
        pass


# ── Entity extraction ────────────────────────────────────────────────────────

# ── Entity match index ───────────────────────────────────────────────────────
# extract_entities_from_query used to walk all entity names twice per query:
# once compiling a fresh `re.escape(name)` pattern each (step 2) and once
# re-tokenising each name (step 3). At 64k entities that was ~128k regex
# compilations per query — far past CPython's 512-entry pattern cache, so every
# one was a full parse+compile. Profiling on 2026-08-06 (#380) put it at ~1.8s
# of a 2.7s call.
#
# Both loops are inverted here against a token index built once per entity-dir
# cache refresh (60s), not per query. Candidate sets are provably supersets of
# the originals, so scoring is unchanged:
#   - step 2 needs the entity's full name inside the query, which implies its
#     first token is present → first-token index is sound.
#   - step 3 skips any entity with zero token overlap → token index is sound.
_entity_index_cache: Optional[tuple] = None
_ENTITY_PATTERN_CACHE: dict[str, "re.Pattern"] = {}
_ENTITY_PATTERN_CACHE_MAX = 4096


def _entity_match_index(entities: list) -> tuple[dict, dict, dict]:
    """Return (first_token→names, token→names, name→token_set) for `entities`.

    Cached on the identity of the list returned by `_get_entity_dirs_cached()`,
    which is stable for the 60s TTL and replaced wholesale on refresh.
    """
    global _entity_index_cache
    if _entity_index_cache is not None and _entity_index_cache[0] is entities:
        return _entity_index_cache[1], _entity_index_cache[2], _entity_index_cache[3]

    first_tok: dict[str, list] = {}
    tok_index: dict[str, list] = {}
    tok_sets: dict[str, set] = {}
    word_re = re.compile(r"\b\w+\b")
    for e in entities:
        el = e.lower()
        toks = word_re.findall(el)
        if not toks:
            continue
        first_tok.setdefault(toks[0], []).append(el)
        scored = {w for w in toks if w not in _SCORING_STOPWORDS and len(w) >= 2}
        tok_sets[el] = scored
        for w in scored:
            tok_index.setdefault(w, []).append(el)

    _entity_index_cache = (entities, first_tok, tok_index, tok_sets)
    return first_tok, tok_index, tok_sets


def _entity_pattern(e_lower: str) -> "re.Pattern":
    """Word-boundary pattern for an entity name, memoised across queries."""
    pat = _ENTITY_PATTERN_CACHE.get(e_lower)
    if pat is None:
        if len(_ENTITY_PATTERN_CACHE) >= _ENTITY_PATTERN_CACHE_MAX:
            _ENTITY_PATTERN_CACHE.clear()
        pat = re.compile(r"(?<!\w)" + re.escape(e_lower) + r"(?!\w)")
        _ENTITY_PATTERN_CACHE[e_lower] = pat
    return pat


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

    first_tok, tok_index, tok_sets = _entity_match_index(entities)
    # Unfiltered query tokens — an entity's first token may be a stopword
    # ("the vault") or single-char, which q_tokens drops.
    q_tokens_all = set(re.findall(r"\b\w+\b", q_lower))

    # 2. Full-name match at word boundaries.
    for tok in q_tokens_all:
        for e_lower in first_tok.get(tok, ()):
            if len(e_lower) < 3:
                continue
            if _entity_pattern(e_lower).search(q_lower):
                _bump(entity_lookup[e_lower], 5.0 + min(len(e_lower) / 20.0, 2.0))

    # 3. Token-overlap scoring. Reward specificity on both sides.
    if q_tokens:
        q_norm = max(len(q_tokens), 2)
        candidates = set()
        for w in q_tokens:
            candidates.update(tok_index.get(w, ()))
        for e_lower in candidates:
            e_cased = entity_lookup[e_lower]
            e_tokens = tok_sets.get(e_lower)
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

    edge_counts = _edge_counts_or_empty()
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


# ── Edge graph ───────────────────────────────────────────────────────────────

def load_relationships() -> dict:
    """Read view of the whole edge graph in the legacy JSON shape.

    Kept for readers that still want `{"edges": [...]}`; the store is the
    source of truth and there is no matching save. Mutating this dict changes
    nothing — that load-mutate-save contract, spread across six programs with
    no lock between them, is what the store replaces. Writers call
    `store().edges.add/expire/retype/rewrite_endpoint`.
    """
    return {"edges": store().edges.all(), "schema_version": 1}


def invalidate_relationships_cache() -> None:
    """Clear the store's derived caches. Used by tests and forced reloads."""
    invalidate_edge_count_cache()


# ── Graph expansion ──────────────────────────────────────────────────────────

def graph_expand_entities(seed_entities: list[str], hops: int = 1) -> list[str]:
    """Expand a set of seed entities via relationship graph traversal."""
    if not seed_entities:
        return []
    try:
        adj = store().edges.adjacency()
    except StoreUnavailable:
        return []
    seed_set = set(seed_entities)
    expanded: set[str] = set()
    current = set(seed_entities)
    for _ in range(hops):
        next_layer = set()
        for entity in current:
            for edge in adj.get(entity, ()):
                neighbor = edge["target"] if edge["source"] == entity else edge["source"]
                if neighbor not in seed_set and neighbor not in expanded:
                    next_layer.add(neighbor)
        expanded.update(next_layer)
        current = next_layer
    return list(expanded)


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
    if not seed_entities:
        return []
    try:
        adj = store().edges.adjacency()
    except StoreUnavailable:
        return []

    seed_set = set(seed_entities)
    scores: dict[str, float] = {}
    current = set(seed_entities)
    visited = set(seed_entities)

    for hop in range(hops):
        hop_decay = 1.0 if hop == 0 else 0.5 ** hop
        next_layer = set()
        for entity in current:
            for edge in adj.get(entity, ()):
                neighbor = edge["target"] if edge["source"] == entity else edge["source"]
                if neighbor in seed_set:
                    continue
                w = (EDGE_TYPE_WEIGHTS.get(edge["type"], _DEFAULT_EDGE_WEIGHT)
                     * edge["confidence"] * hop_decay)
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
    edge_counts = _edge_counts_or_empty()
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
