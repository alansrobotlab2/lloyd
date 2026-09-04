#!/usr/bin/env python3
"""
Lloyd MCP Server: Facts — knowledge-graph facts and typed relationships.

Tools:
    fact_get, fact_add, fact_profile, fact_check, fact_resolve,
    fact_invalidate, fact_relate, fact_relationships, fact_path,
    fact_neighbors  (10 tools)

Data root: app.paths.VAULT_FACTS_ROOT
    (currently ~/lloyd/_pipeline/vault-derived/facts/)
Edge graph, aliases, entity registry, fact index: app.kg_store

Split out of agent_mcp/memory.py as part of Task #340 PR 5. Owns the
fact tool handlers and the entity-keyed fact files
(<FACTS_ROOT>/<Entity>/<Entity>-<category>.md).

The retrieval core (entity extraction, relationships index + cache,
graph expansion, fact ranking) lives in agent_mcp.retrieval — shared
with vault.py via public names. The underscore aliases imported below
keep this module's call sites and external consumers stable.
"""

import datetime
import uuid

from mcp.server import Server
from mcp.types import Tool

try:
    from app.entity_naming import looks_like_junk_entity as _is_junk_entity
except Exception:  # pragma: no cover - defensive import
    def _is_junk_entity(name: str) -> bool:
        return False

from agent_mcp._shared import (
    FACTS_ROOT,
    ErrorCode,
    atomic_write_text,
    _err,
    _find_entity_dir,
    _invalidate_entity_dirs_cache,
    _parse_fact_frontmatter,
    _resolve_entity,
    _token_overlap,
    _wrap,
    _write_fact_frontmatter,
)
from app.kg_store import StoreUnavailable, store as _store
from agent_mcp.retrieval import (  # noqa: F401  (re-exported compat names)
    EDGE_TYPE_WEIGHTS,
    RelationshipsCorrupt,
    FACT_GODNODE_THRESHOLD,
    FACT_RANK_CAP_GRAPH,
    FACT_RANK_CAP_SEED,
    extract_entities_from_query as _extract_entities_from_query,
    fact_matches_tokens as _fact_matches_tokens,
    fact_query_tokens as _fact_query_tokens,
    fact_score as _fact_score,
    get_entity_edge_counts as _get_entity_edge_counts,
    get_facts_sync as _get_facts_sync,
    graph_expand_entities as _graph_expand_entities,
    graph_weighted_neighbors as _graph_weighted_neighbors,
    invalidate_relationships_cache as _invalidate_relationships_cache,
    load_relationships as _load_relationships,
)

# ── Constants ────────────────────────────────────────────────────────────────

# Used by _detect_contradictions_sync.
_OPPOSING_PAIRS = [
    ("yes", "no"), ("true", "false"), ("enabled", "disabled"),
    ("active", "inactive"), ("supported", "unsupported"),
    ("working", "broken"), ("success", "failure"),
]

app = Server("lloyd-facts")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _generate_fact_id(category: str) -> str:
    return f"{category[:4]}-{uuid.uuid4().hex[:4]}"


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
    if _is_junk_entity(raw_entity):
        return _err(
            f"'{raw_entity}' looks like a filename or code fragment, not an entity; "
            "use a concept/project/person name",
            ErrorCode.INVALID_PARAM,
        )
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
        atomic_write_text(fact_file, _write_fact_frontmatter(frontmatter) + body)
        _invalidate_entity_dirs_cache()
        # The store learns about the entity and the new fact here rather than
        # waiting for the nightly reindex, so a fact added in chat is visible
        # to the router and to `facts_idx` immediately.
        try:
            st = _store()
            st.entities.register(entity)
            st.facts_idx.update_file(fact_file, root=FACTS_ROOT)
        except StoreUnavailable as exc:
            # The markdown write already succeeded and is the fact layer; the
            # index is derived and `kg reindex` rebuilds it. Say so, don't fail.
            result_note = f"fact written; store index not updated ({exc})"
        else:
            result_note = None
        result: dict = {"success": True, "fact_id": fact_id, "entity": entity, "category": category}
        if result_note:
            result["warning"] = result_note
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
                            atomic_write_text(fact_file, _write_fact_frontmatter(frontmatter) + body)
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
                atomic_write_text(fact_file, _write_fact_frontmatter(frontmatter) + body)
        return {"success": True, "entity": resolved, "expired_count": expired_count, "matched_facts": matched_facts}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, expired_count=0)


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
    try:
        # mode="write" so edges land on the literal names the caller
        # specified, not fuzzy-matched neighbours. (#340 PR 3.)
        src_resolved, _ = _resolve_entity(source, mode="write")
        tgt_resolved, _ = _resolve_entity(target, mode="write")
        if src_resolved == tgt_resolved:
            return _err(
                f"source and target both resolve to {src_resolved!r}",
                ErrorCode.INVALID_PARAM,
            )
        st = _store()
        existing = st.edges.find_active(src_resolved, tgt_resolved, rel_type)
        if existing is not None:
            return {"success": True, "action": "already_exists", "edge_id": existing["id"],
                    "source": src_resolved, "target": tgt_resolved, "type": rel_type}
        edge_id = st.edges.add({
            "source": src_resolved, "target": tgt_resolved, "type": rel_type,
            "confidence": confidence, "provenance": provenance,
            "source_doc": params.get("source_doc"),
            "evidence": params.get("evidence"),
        }, origin="fact_relate")
        return {"success": True, "action": "created", "edge_id": edge_id,
                "source": src_resolved, "target": tgt_resolved, "type": rel_type}
    except StoreUnavailable as exc:
        # Writing over an unreadable graph is how 6,539 edges become 1.
        return _err(f"edge store is unreadable, refusing to write: {exc}", ErrorCode.INTERNAL)
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
        st = _store()
        types = [rel_type] if rel_type else None
        if direction == "out":
            edges = st.edges.active(source=resolved, types=types)
        elif direction == "in":
            edges = st.edges.active(target=resolved, types=types)
        else:
            edges = st.edges.active(either=resolved, types=types)
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
        adj = _store().edges.adjacency()
        from collections import deque
        queue = deque([(src_resolved, [src_resolved], [])])
        visited = {src_resolved}
        while queue:
            node, path, edges_path = queue.popleft()
            if node == tgt_resolved:
                return {"found": True, "path": path, "edges": edges_path, "hops": len(edges_path)}
            if len(path) > max_hops:
                continue
            for edge in adj.get(node, ()):
                neighbor = edge["target"] if edge["source"] == node else edge["source"]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor],
                                  edges_path + [{"source": edge["source"], "target": edge["target"],
                                                 "type": edge["type"]}]))
        return {"found": False, "path": [], "edges": [], "hops": -1}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


# Soft caps on fact_neighbors result size. The harness spills any result
# above ~50KB to disk (app.harness.tool_result_spill), so the model never
# loses information — but explicit caps + a `hint` field steer the model
# toward narrower queries (hops=1, higher min_confidence) BEFORE it pays
# the spill cost. Defaults are generous; hub-entity hops=2 still hits them.
_FACT_NEIGHBORS_MAX_NODES = 1000
_FACT_NEIGHBORS_MAX_EDGES = 2000


def _fact_neighbors(params: dict) -> dict:
    """Get neighborhood subgraph around an entity within N hops.

    Truncates at ``_FACT_NEIGHBORS_MAX_NODES`` / ``_FACT_NEIGHBORS_MAX_EDGES``
    to bound result size. When truncated, sets ``truncated=True`` and adds
    a ``hint`` field telling the caller to narrow with ``hops=1`` or a
    higher ``min_confidence`` threshold. Without this cap, hub-entity
    queries return tens of thousands of nodes in a single tool result.
    """
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM)
    hops = int(params.get("hops", 1))
    min_confidence = float(params.get("min_confidence", 0.5))
    max_nodes = int(params.get("max_nodes", _FACT_NEIGHBORS_MAX_NODES))
    max_edges = int(params.get("max_edges", _FACT_NEIGHBORS_MAX_EDGES))
    try:
        resolved, _ = _resolve_entity(entity, mode="read")
        adj = _store().edges.adjacency(min_confidence=min_confidence)
        visited = {resolved}
        current_layer = [resolved]
        all_edges: list[dict] = []
        truncated = False
        for _ in range(hops):
            next_layer = []
            for node in current_layer:
                for edge in adj.get(node, ()):
                    if len(all_edges) >= max_edges:
                        truncated = True
                        break
                    all_edges.append({"source": edge["source"], "target": edge["target"],
                                      "type": edge["type"], "confidence": edge["confidence"]})
                    neighbor = edge["target"] if edge["source"] == node else edge["source"]
                    if neighbor not in visited:
                        if len(visited) >= max_nodes:
                            truncated = True
                            continue
                        visited.add(neighbor)
                        next_layer.append(neighbor)
                if truncated and len(all_edges) >= max_edges:
                    break
            if truncated and len(all_edges) >= max_edges:
                break
            current_layer = next_layer
        seen_edges = set()
        unique_edges = []
        for e in all_edges:
            key = (e["source"], e["target"], e["type"])
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)
        result: dict = {
            "entity": resolved,
            "nodes": sorted(visited),
            "edges": unique_edges,
            "node_count": len(visited),
            "edge_count": len(unique_edges),
        }
        if truncated:
            result["truncated"] = True
            result["hint"] = (
                f"Result truncated at {max_nodes} nodes / {max_edges} edges. "
                f"Hub entity '{resolved}' has too many connections at hops={hops}. "
                f"Retry with hops=1, raise min_confidence (currently {min_confidence}), "
                f"or pass smaller max_nodes/max_edges."
            )
        return result
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


# ── Graph traversal (used by vault.py for vault_recall) ──────────────────────

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
        Tool(name="fact_neighbors", description="Get neighborhood subgraph around an entity within N hops. Truncates at 200 nodes / 500 edges by default; hub entities at hops=2 will be truncated — narrow with hops=1 or higher min_confidence.", inputSchema={
            "type": "object", "properties": {
                "entity": {"type": "string"},
                "hops": {"type": "integer"},
                "min_confidence": {"type": "number"},
                "max_nodes": {"type": "integer", "description": "Cap on returned nodes (default 200)"},
                "max_edges": {"type": "integer", "description": "Cap on returned edges (default 500)"},
            }, "required": ["entity"]}),
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
