#!/usr/bin/env python3
"""
Lloyd MCP Server: Facts — knowledge-graph facts and typed relationships.

Tools:
    fact_get, fact_add, fact_profile, fact_check, fact_resolve,
    fact_invalidate, fact_relate, fact_relationships, fact_path,
    fact_neighbors  (10 tools)

Data root: ~/obsidian/facts/
Relationships index: ~/obsidian/facts/_relationships.json

Split out of agent_mcp/memory.py as part of Task #340 PR 5. Owns:
    - Entity-keyed fact files (~/obsidian/facts/<Entity>/<Entity>-<category>.md)
    - The relationships index (with mtime-based caching from PR 2)
    - Entity-to-entity graph traversal + weighted expansion
    - Query-aware fact ranking (FACT_GODNODE_THRESHOLD etc.)
"""

import datetime
import json
import re
import time
import uuid
from typing import Optional

from mcp.server import Server
from mcp.types import Tool

from agent_mcp._shared import (
    FACTS_ROOT,
    ErrorCode,
    _SCORING_STOPWORDS,
    _err,
    _find_entity_dir,
    _get_entity_dirs_cached,
    _invalidate_entity_dirs_cache,
    _parse_fact_frontmatter,
    _resolve_entity,
    _token_overlap,
    _wrap,
    _write_fact_frontmatter,
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

# Used by _detect_contradictions_sync.
_OPPOSING_PAIRS = [
    ("yes", "no"), ("true", "false"), ("enabled", "disabled"),
    ("active", "inactive"), ("supported", "unsupported"),
    ("working", "broken"), ("success", "failure"),
]

# Task-ID extractor. Matches "Task #299", "Task 299", "#299", "task_310",
# "backlog_120", "backlog_item_18", "backlog_task_41". Captures the numeric
# ID so we can dispatch to whichever naming convention exists.
_TASK_ID_RE = re.compile(
    r"(?:\btask\s*[#_ ]?|#|\bbacklog[_ -]?(?:item[_ -]?|task[_ -]?)?)(\d{1,4})\b",
    re.IGNORECASE,
)

# Cache for entity degree (non-expired edge count) — used as a deterministic
# tie-break signal in _extract_entities_from_query. 60s TTL keeps the hot
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
#   - _save_relationships() refreshes the cache with the new mtime.
#   - This handles both in-process mutation (load → mutate → save) and
#     cross-process writes (autonomy classifier writes file, MCP server
#     picks up via stat-check on next read).
#
# Mutation contract: callers that mutate the returned dict MUST follow
# with _save_relationships(). Between load and save, the cache and the
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

# Stopwords for query tokenization — question words, auxiliaries, common
# function words. Kept lightweight; aggressive stopword removal hurts when
# the query itself is short ("how does fact_path work").
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

app = Server("lloyd-facts")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _generate_fact_id(category: str) -> str:
    return f"{category[:4]}-{uuid.uuid4().hex[:4]}"


def _get_facts_sync(entity: str, category: str = None, as_of: str = None,
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


def _get_entity_edge_counts() -> dict:
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


def _extract_entities_from_query(query: str) -> list:
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

    edge_counts = _get_entity_edge_counts()
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


def _detect_contradictions_sync(entity: str, category: str = None) -> dict:
    facts = _get_facts_sync(entity, category).get("facts", [])
    contradictions = []
    for i, f1 in enumerate(facts):
        for f2 in facts[i + 1:]:
            t1, t2 = f1.get("fact", "").lower(), f2.get("fact", "").lower()
            reason = None
            for pair in _OPPOSING_PAIRS:
                if (pair[0] in t1 and pair[1] in t2) or (pair[1] in t1 and pair[0] in t2):
                    reason = f"opposing_terms:{pair[0]}/{pair[1]}"
                    break
            if not reason and _token_overlap(t1, t2) > 0.6:
                reason = "high_overlap_potential_update"
            if reason:
                contradictions.append({"fact1": f1, "fact2": f2, "reason": reason})
    return {"entity": entity, "category": category, "contradictions": contradictions, "checked": len(facts)}


# ── Tool handlers ────────────────────────────────────────────────────────────

def _fact_get(params: dict) -> dict:
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM, facts=[])
    category = params.get("category") or None
    as_of = params.get("as_of") or None
    include_expired = bool(params.get("include_expired", False))
    try:
        return _get_facts_sync(entity, category, as_of=as_of,
                               include_expired=include_expired)
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, facts=[])


def _fact_add(params: dict) -> dict:
    raw_entity = params.get("entity", "").strip()
    category = params.get("category", "").strip()
    fact_text = params.get("fact", "").strip()
    if not raw_entity or not category or not fact_text:
        return _err("entity, category, and fact are required", ErrorCode.MISSING_PARAM)
    confidence = float(params.get("confidence", 0.9))
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        # mode="write" — exact + alias only, no fuzzy match. The fact lands
        # on the literal name the caller specified. (#340 PR 3 — fixes the
        # silent fuzzy-merge data-corruption bug.)
        entity, is_new = _resolve_entity(raw_entity, mode="write")
        entity_dir = _find_entity_dir(entity)
        if not entity_dir:
            entity_dir = FACTS_ROOT / entity
            entity_dir.mkdir(parents=True, exist_ok=True)
        fact_file = entity_dir / f"{entity}-{category}.md"
        if fact_file.exists():
            frontmatter = _parse_fact_frontmatter(fact_file.read_text(encoding="utf-8"))
        else:
            frontmatter = {"type": "facts", "entity": entity, "category": category, "facts": []}
        fact_id = _generate_fact_id(category)
        provenance = params.get("provenance", "STATED")
        if provenance not in ("STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"):
            provenance = "STATED"
        new_fact = {"fact": fact_text, "confidence": confidence, "category": category, "id": fact_id, "created_at": now_iso, "valid_at": params.get("valid_at"), "invalid_at": None, "expired_at": None, "provenance": provenance, "source_doc": params.get("source_doc")}
        frontmatter.setdefault("facts", []).append(new_fact)
        frontmatter["last_updated"] = now_iso
        body = f"\n# {entity} - {category}\n\n**Entity:** {entity}\n**Category:** {category}\n**Fact Count:** {len(frontmatter['facts'])}\n"
        fact_file.write_text(_write_fact_frontmatter(frontmatter) + body, encoding="utf-8")
        _invalidate_entity_dirs_cache()
        result: dict = {"success": True, "fact_id": fact_id, "entity": entity, "category": category}
        if entity != raw_entity:
            result["resolved_from"] = raw_entity
        return result
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _fact_profile(params: dict) -> dict:
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM)
    try:
        facts = _get_facts_sync(entity).get("facts", [])
        categories: dict = {}
        for fact in facts:
            cat = fact.get("category", "general")
            categories.setdefault(cat, []).append(fact)
        lines = [f"Profile for: {entity}"]
        for cat, cat_facts in categories.items():
            lines.append(f"\n{cat.upper()}:")
            for f in cat_facts[:3]:
                lines.append(f"  - {f.get('fact', '')}")
        return {"entity": entity, "categories": categories, "fact_count": len(facts), "summary": "\n".join(lines)}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _fact_check(params: dict) -> dict:
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM, contradictions=[], checked=0)
    try:
        return _detect_contradictions_sync(entity, params.get("category"))
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, contradictions=[], checked=0)


def _fact_resolve(params: dict) -> dict:
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        result = _detect_contradictions_sync(entity)
        contradictions = result.get("contradictions", [])
        resolved = 0
        if params.get("auto_resolve", True) and contradictions:
            entity_dir = _find_entity_dir(entity)
            if entity_dir:
                for contradiction in contradictions:
                    f1, f2 = contradiction.get("fact1", {}), contradiction.get("fact2", {})
                    c1, c2 = f1.get("confidence", 0.5), f2.get("confidence", 0.5)
                    invalidate_id = f2.get("id") if c1 >= c2 else f1.get("id")
                    if not invalidate_id:
                        continue
                    for fact_file in entity_dir.glob("*.md"):
                        content = fact_file.read_text(encoding="utf-8")
                        frontmatter = _parse_fact_frontmatter(content)
                        if "facts" not in frontmatter:
                            continue
                        changed = False
                        for f in frontmatter["facts"]:
                            if f.get("id") == invalidate_id:
                                f["invalid_at"] = now_iso
                                f["expired_at"] = now_iso
                                changed = True
                        if changed:
                            body_start = content.find("---", 3)
                            body = content[body_start + 3:] if body_start != -1 else ""
                            fact_file.write_text(_write_fact_frontmatter(frontmatter) + body, encoding="utf-8")
                            resolved += 1
        return {"entity": entity, "resolved": resolved, "remaining": len(contradictions) - resolved}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _fact_invalidate(params: dict) -> dict:
    """Expire facts that are no longer current (were true, now outdated)."""
    entity = params.get("entity", "").strip()
    ended = params.get("ended", "").strip()
    if not entity or not ended:
        return _err("entity and ended (ISO date) are required", ErrorCode.MISSING_PARAM)
    category = params.get("category") or None
    fact_substring = params.get("fact_substring", "").strip().lower()
    reason = params.get("reason", "").strip()
    try:
        resolved, _ = _resolve_entity(entity, mode="read")
        entity_dir = _find_entity_dir(resolved)
        if not entity_dir:
            return _err(f"Entity not found: {entity}", ErrorCode.NOT_FOUND, expired_count=0)
        expired_count = 0
        matched_facts = []
        files_to_scan = []
        if category:
            fact_file = entity_dir / f"{resolved}-{category}.md"
            if not fact_file.exists():
                fact_file = entity_dir / f"{entity}-{category}.md"
            if fact_file.exists():
                files_to_scan.append(fact_file)
        else:
            files_to_scan = list(entity_dir.glob("*.md"))
        for fact_file in files_to_scan:
            content = fact_file.read_text(encoding="utf-8")
            frontmatter = _parse_fact_frontmatter(content)
            if "facts" not in frontmatter:
                continue
            changed = False
            for f in frontmatter["facts"]:
                if f.get("expired_at") or f.get("invalid_at"):
                    continue
                if fact_substring and fact_substring not in f.get("fact", "").lower():
                    continue
                f["expired_at"] = ended
                if reason:
                    f["expired_reason"] = reason
                changed = True
                expired_count += 1
                matched_facts.append({"id": f.get("id"), "fact": f.get("fact", "")[:80]})
            if changed:
                body_start = content.find("---", 3)
                body = content[body_start + 3:] if body_start != -1 else ""
                fact_file.write_text(_write_fact_frontmatter(frontmatter) + body, encoding="utf-8")
        return {"success": True, "entity": resolved, "expired_count": expired_count, "matched_facts": matched_facts}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, expired_count=0)


# ── Relationship store ───────────────────────────────────────────────────────

def _load_relationships() -> dict:
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


def _save_relationships(data: dict) -> None:
    """Persist the relationships index and refresh the cache."""
    global _relationships_cache
    RELATIONSHIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RELATIONSHIPS_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=False), encoding="utf-8"
    )
    try:
        new_mtime = RELATIONSHIPS_PATH.stat().st_mtime_ns
        _relationships_cache = (new_mtime, data)
    except OSError:
        _relationships_cache = None


def _invalidate_relationships_cache() -> None:
    """Clear the relationships cache. Used by tests and forced reloads."""
    global _relationships_cache
    _relationships_cache = None


def _fact_relate(params: dict) -> dict:
    """Add a typed relationship edge between two entities."""
    source = params.get("source", "").strip()
    target = params.get("target", "").strip()
    rel_type = params.get("type", "").strip()
    if not source or not target or not rel_type:
        return _err("source, target, and type are required", ErrorCode.MISSING_PARAM)
    confidence = float(params.get("confidence", 0.9))
    provenance = params.get("provenance", "STATED")
    if provenance not in ("STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"):
        provenance = "STATED"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        # mode="write" so edges land on the literal names the caller
        # specified, not fuzzy-matched neighbours. (#340 PR 3.)
        src_resolved, _ = _resolve_entity(source, mode="write")
        tgt_resolved, _ = _resolve_entity(target, mode="write")
        data = _load_relationships()
        for edge in data["edges"]:
            if (edge["source"] == src_resolved and edge["target"] == tgt_resolved
                    and edge["type"] == rel_type and not edge.get("expired_at")):
                return {"success": True, "action": "already_exists",
                        "source": src_resolved, "target": tgt_resolved, "type": rel_type}
        new_edge = {
            "source": src_resolved, "target": tgt_resolved, "type": rel_type,
            "confidence": confidence, "provenance": provenance,
            "created_at": now_iso, "expired_at": None,
            "source_doc": params.get("source_doc"),
        }
        data["edges"].append(new_edge)
        _save_relationships(data)
        return {"success": True, "action": "created",
                "source": src_resolved, "target": tgt_resolved, "type": rel_type}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _fact_relationships(params: dict) -> dict:
    """Get all relationships for an entity (inbound + outbound)."""
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM, edges=[])
    direction = params.get("direction", "both")
    rel_type = params.get("type") or None
    try:
        resolved, _ = _resolve_entity(entity, mode="read")
        data = _load_relationships()
        edges = []
        for edge in data["edges"]:
            if edge.get("expired_at"):
                continue
            match = False
            if direction in ("out", "both") and edge["source"] == resolved:
                match = True
            if direction in ("in", "both") and edge["target"] == resolved:
                match = True
            if match and rel_type and edge["type"] != rel_type:
                match = False
            if match:
                edges.append(edge)
        return {"entity": resolved, "edges": edges, "count": len(edges)}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, edges=[])


def _fact_path(params: dict) -> dict:
    """Find shortest path between two entities via BFS on relationship graph."""
    source = params.get("source", "").strip()
    target = params.get("target", "").strip()
    max_hops = int(params.get("max_hops", 3))
    if not source or not target:
        return _err("source and target are required", ErrorCode.MISSING_PARAM)
    try:
        src_resolved, _ = _resolve_entity(source, mode="read")
        tgt_resolved, _ = _resolve_entity(target, mode="read")
        data = _load_relationships()
        adj: dict[str, list[tuple[str, dict]]] = {}
        for edge in data["edges"]:
            if edge.get("expired_at"):
                continue
            s, t = edge["source"], edge["target"]
            adj.setdefault(s, []).append((t, edge))
            adj.setdefault(t, []).append((s, edge))
        from collections import deque
        queue = deque([(src_resolved, [src_resolved], [])])
        visited = {src_resolved}
        while queue:
            node, path, edges_path = queue.popleft()
            if node == tgt_resolved:
                return {"found": True, "path": path, "edges": edges_path, "hops": len(edges_path)}
            if len(path) > max_hops:
                continue
            for neighbor, edge in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor],
                                  edges_path + [{"source": edge["source"], "target": edge["target"], "type": edge["type"]}]))
        return {"found": False, "path": [], "edges": [], "hops": -1}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _fact_neighbors(params: dict) -> dict:
    """Get neighborhood subgraph around an entity within N hops."""
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM)
    hops = int(params.get("hops", 1))
    min_confidence = float(params.get("min_confidence", 0.5))
    try:
        resolved, _ = _resolve_entity(entity, mode="read")
        data = _load_relationships()
        adj: dict[str, list[tuple[str, dict]]] = {}
        for edge in data["edges"]:
            if edge.get("expired_at") or edge.get("confidence", 1.0) < min_confidence:
                continue
            s, t = edge["source"], edge["target"]
            adj.setdefault(s, []).append((t, edge))
            adj.setdefault(t, []).append((s, edge))
        from collections import deque  # noqa: F401
        visited = {resolved}
        current_layer = [resolved]
        all_edges = []
        for _ in range(hops):
            next_layer = []
            for node in current_layer:
                for neighbor, edge in adj.get(node, []):
                    edge_key = (edge["source"], edge["target"], edge["type"])  # noqa: F841
                    all_edges.append({"source": edge["source"], "target": edge["target"],
                                      "type": edge["type"], "confidence": edge.get("confidence", 1.0)})
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.append(neighbor)
            current_layer = next_layer
        seen_edges = set()
        unique_edges = []
        for e in all_edges:
            key = (e["source"], e["target"], e["type"])
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)
        return {"entity": resolved, "nodes": sorted(visited),
                "edges": unique_edges, "node_count": len(visited), "edge_count": len(unique_edges)}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


# ── Graph traversal (used by vault.py for vault_recall) ──────────────────────

def _graph_expand_entities(seed_entities: list[str], hops: int = 1) -> list[str]:
    """Expand a set of seed entities via relationship graph traversal."""
    if not RELATIONSHIPS_PATH.exists():
        return []
    try:
        data = _load_relationships()
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


def _graph_weighted_neighbors(
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
        data = _load_relationships()
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
                scores[neighbor] = min(1.0, scores.get(neighbor, 0.0) + w)
                if neighbor not in visited:
                    next_layer.add(neighbor)
        visited.update(next_layer)
        current = next_layer

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return ranked[:top_k]


# ── Fact ranking helpers (used by vault.py for vault_recall) ─────────────────

def _fact_query_tokens(query: str) -> list[str]:
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


def _fact_matches_tokens(fact: dict, tokens: list[str]) -> bool:
    """True if any query token appears in the fact's searchable text."""
    if not tokens:
        return False
    blob = _fact_blob(fact)
    return any(t in blob for t in tokens)


def _fact_score(fact: dict, tokens: list[str]) -> float:
    """Fraction of query tokens that appear in the fact's searchable text."""
    if not tokens:
        return 0.0
    blob = _fact_blob(fact)
    hits = sum(1 for t in tokens if t in blob)
    return hits / len(tokens)


# ── MCP registration ─────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools():
    return [
        Tool(name="fact_get", description="Retrieve structured facts for a named entity. Supports temporal queries with as_of and include_expired.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "category": {"type": "string"}, "as_of": {"type": "string", "description": "ISO date — return facts valid at this point in time"}, "include_expired": {"type": "boolean", "description": "If true, include expired/invalidated facts"}}, "required": ["entity"]}),
        Tool(name="fact_add", description="Add a structured fact for a named entity/category.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "category": {"type": "string"}, "fact": {"type": "string"}, "confidence": {"type": "number"}, "valid_at": {"type": "string"}, "provenance": {"type": "string", "enum": ["STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"], "description": "How the fact was derived (default: STATED)"}, "source_doc": {"type": "string"}}, "required": ["entity", "category", "fact"]}),
        Tool(name="fact_profile", description="Get synthesized profile for an entity — all facts grouped by category.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}}, "required": ["entity"]}),
        Tool(name="fact_check", description="Detect contradictions in stored facts for an entity.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "category": {"type": "string"}}, "required": ["entity"]}),
        Tool(name="fact_resolve", description="Resolve contradictions by keeping higher-confidence fact.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "auto_resolve": {"type": "boolean"}}, "required": ["entity"]}),
        Tool(name="fact_invalidate", description="Expire facts that are no longer current. Sets expired_at on matching facts.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "category": {"type": "string"}, "fact_substring": {"type": "string", "description": "Match facts containing this text"}, "ended": {"type": "string", "description": "ISO date when fact stopped being true"}, "reason": {"type": "string", "description": "Why the fact was expired"}}, "required": ["entity", "ended"]}),
        Tool(name="fact_relate", description="Add a typed relationship edge between two entities.", inputSchema={
            "type": "object", "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "type": {"type": "string", "description": "Relationship type (e.g. built_on, uses, part_of, related_to)"}, "confidence": {"type": "number"}, "provenance": {"type": "string", "enum": ["STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"]}, "source_doc": {"type": "string"}}, "required": ["source", "target", "type"]}),
        Tool(name="fact_relationships", description="Get all relationships for an entity (inbound + outbound edges).", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "direction": {"type": "string", "enum": ["in", "out", "both"]}, "type": {"type": "string"}}, "required": ["entity"]}),
        Tool(name="fact_path", description="Find shortest path between two entities via relationship graph.", inputSchema={
            "type": "object", "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "max_hops": {"type": "integer"}}, "required": ["source", "target"]}),
        Tool(name="fact_neighbors", description="Get neighborhood subgraph around an entity within N hops.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "hops": {"type": "integer"}, "min_confidence": {"type": "number"}}, "required": ["entity"]}),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    handlers = {
        "fact_get": _fact_get, "fact_add": _fact_add, "fact_profile": _fact_profile,
        "fact_check": _fact_check, "fact_resolve": _fact_resolve,
        "fact_invalidate": _fact_invalidate,
        "fact_relate": _fact_relate, "fact_relationships": _fact_relationships,
        "fact_path": _fact_path, "fact_neighbors": _fact_neighbors,
    }
    handler = handlers.get(name)
    if handler:
        return _wrap(handler(arguments))
    return _wrap(_err(f"Unknown tool: {name}", ErrorCode.UNKNOWN_TOOL))
