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
from app.atomic_io import locked_file
from app.fact_ids import assign_ids as _assign_fact_ids, category_prefix, next_fact_id
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

def _generate_fact_id(category: str, existing_ids=()) -> str:
    """The next `<prefix>-NNN` for this file.

    Was `f"{category[:4]}-{uuid4().hex[:4]}"`, which produced a second,
    incompatible ID scheme in the same files the extractor numbered
    sequentially. One scheme now — see app.fact_ids.
    """
    return next_fact_id(existing_ids, category_prefix(category))


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
    source_doc = params.get("source_doc")
    if _is_junk_entity(raw_entity, source_doc):
        return _err(
            f"'{raw_entity}' looks like a filename, a code fragment or a pipeline "
            "run, not an entity; use a concept/project/person name",
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
        provenance = params.get("provenance", "STATED")
        if provenance not in ("STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"):
            provenance = "STATED"

        # The lock covers read-modify-write. Four extractor threads write
        # these same files; without it whichever finishes last wins and the
        # other's facts are gone.
        with locked_file(fact_file):
            if fact_file.exists():
                raw = fact_file.read_text(encoding="utf-8")
                frontmatter = _parse_fact_frontmatter(raw)
                if not frontmatter and raw.strip():
                    # A file that exists, is non-empty, and will not parse is
                    # corrupt. Writing here would replace an entity's whole
                    # history with this one fact.
                    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
                    quarantine = fact_file.with_name(f"{fact_file.name}.corrupt-{stamp}")
                    fact_file.rename(quarantine)
                    return _err(
                        f"{fact_file.name} is corrupt and was quarantined as "
                        f"{quarantine.name}; nothing was overwritten. Retry to "
                        "start a fresh file.",
                        ErrorCode.INTERNAL,
                    )
            else:
                frontmatter = {}
            if not frontmatter:
                frontmatter = {"type": "facts", "entity": entity, "category": category, "facts": []}
            existing = frontmatter.setdefault("facts", [])
            fact_id = _generate_fact_id(category, [f.get("id") for f in existing if isinstance(f, dict)])
            new_fact = {"fact": fact_text, "confidence": confidence, "category": category,
                        "id": fact_id, "created_at": now_iso, "valid_at": params.get("valid_at"),
                        "invalid_at": None, "expired_at": None, "provenance": provenance,
                        "source_doc": source_doc}
            existing.append(new_fact)
            _assign_fact_ids(existing, category)
            frontmatter["last_updated"] = now_iso
            body = (f"\n# {entity} - {category}\n\n**Entity:** {entity}\n"
                    f"**Category:** {category}\n**Fact Count:** {len(existing)}\n")
            atomic_write_text(fact_file, _write_fact_frontmatter(frontmatter) + body)
        _invalidate_entity_dirs_cache()
        # The store learns about the entity and the new fact here rather than
        # waiting for the nightly reindex, so a fact added in chat is visible
        # to the router and to `facts_idx` immediately.
        result: dict = {"success": True, "fact_id": fact_id, "entity": entity, "category": category}
        try:
            st = _store()
            st.entities.register(entity)
            st.facts_idx.update_file(fact_file, root=FACTS_ROOT)
        except StoreUnavailable as exc:
            # The markdown write already succeeded and is the fact layer; the
            # index is derived and `kg reindex` rebuilds it. Say so, don't fail.
            result["warning"] = f"fact written; store index not updated ({exc})"
        if entity != raw_entity:
            result["resolved_from"] = raw_entity
        return result
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _fact_profile(params: dict) -> dict:
    """All of an entity's facts, grouped by category.

    Capped at `FACT_RANK_CAP_SEED` per category. Uncapped, `fact_profile` on
    a god node returned every fact it had — `Lloyd` alone carries 5,489, and
    the whole list went into the model's context to answer one question.
    With a `query`, each category is ranked by token overlap and the cap
    keeps the most relevant; without one it keeps the most recent.
    """
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM)
    query = (params.get("query") or "").strip()
    cap = int(params.get("limit_per_category", FACT_RANK_CAP_SEED))
    try:
        facts = _get_facts_sync(entity).get("facts", [])
        categories: dict = {}
        for fact in facts:
            cat = fact.get("category", "general")
            categories.setdefault(cat, []).append(fact)

        tokens = _fact_query_tokens(query) if query else []
        truncated: dict = {}
        for cat, cat_facts in categories.items():
            if tokens:
                cat_facts.sort(key=lambda f: (-_fact_score(f, tokens),
                                              str(f.get("created_at") or "")), reverse=False)
            else:
                cat_facts.sort(key=lambda f: str(f.get("created_at") or ""), reverse=True)
            if len(cat_facts) > cap:
                truncated[cat] = len(cat_facts)
                categories[cat] = cat_facts[:cap]

        lines = [f"Profile for: {entity}"]
        for cat, cat_facts in categories.items():
            total = truncated.get(cat, len(cat_facts))
            header = f"\n{cat.upper()}:" + (f"  (showing {len(cat_facts)} of {total})" if cat in truncated else "")
            lines.append(header)
            for f in cat_facts[:3]:
                lines.append(f"  - {f.get('fact', '')}")
        result = {"entity": entity, "categories": categories, "fact_count": len(facts),
                  "summary": "\n".join(lines)}
        if truncated:
            result["truncated_categories"] = truncated
            result["hint"] = (
                f"{entity} has {len(facts)} facts; each category is capped at {cap}. "
                "Pass `query` to rank by relevance, or `fact_get(entity, category=…)` "
                "for one category in full."
            )
        return result
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
    """Report contradictions; optionally mark the weaker side invalid.

    `auto_resolve` now defaults to FALSE. It defaulted to true, so a bare
    `fact_resolve(entity=...)` — which reads like a query — silently expired
    facts, and its contradiction detector fires on `_token_overlap > 0.6`,
    which is two facts phrased similarly, not two facts that disagree.

    When it does act it sets `invalid_at` only, not `expired_at`. The two
    mean different things: expired is "was true, no longer is"; invalid is
    "should not have been recorded". A same-confidence pair is left alone —
    there is no basis to pick a winner.

    Refuses outright on an entity above FACT_GODNODE_THRESHOLD facts, where
    the pairwise scan is O(n²) and the overlap heuristic produces mostly
    false positives.
    """
    entity = params.get("entity", "").strip()
    if not entity:
        return _err("entity is required", ErrorCode.MISSING_PARAM)
    auto_resolve = bool(params.get("auto_resolve", False))
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        detection = _detect_contradictions_sync(entity)
        contradictions = detection.get("contradictions", [])
        if not auto_resolve:
            return {"entity": entity, "resolved": 0,
                    "contradictions": contradictions[:20],
                    "remaining": len(contradictions),
                    "hint": ("Reporting only. Pass auto_resolve=true to mark the "
                             "lower-confidence side invalid, or use fact_invalidate "
                             "to expire a specific fact.")}
        if detection.get("checked", 0) > FACT_GODNODE_THRESHOLD:
            return _err(
                f"{entity} has {detection['checked']} facts; auto_resolve is refused "
                f"above {FACT_GODNODE_THRESHOLD} because the overlap heuristic yields "
                "mostly false positives at that size. Use fact_invalidate on specific facts.",
                ErrorCode.INVALID_PARAM, resolved=0, remaining=len(contradictions))

        entity_dir = _find_entity_dir(entity)
        resolved = 0
        if entity_dir:
            to_invalidate: dict[str, str] = {}
            for contradiction in contradictions:
                f1, f2 = contradiction.get("fact1", {}), contradiction.get("fact2", {})
                c1, c2 = f1.get("confidence", 0.5), f2.get("confidence", 0.5)
                if c1 == c2:
                    continue     # no basis to pick a winner
                loser = f2 if c1 > c2 else f1
                if loser.get("id"):
                    to_invalidate[loser["id"]] = contradiction.get("reason", "contradiction")
            for fact_file in entity_dir.glob("*.md"):
                content = fact_file.read_text(encoding="utf-8")
                frontmatter = _parse_fact_frontmatter(content)
                if "facts" not in frontmatter:
                    continue
                changed = False
                for f in frontmatter["facts"]:
                    reason = to_invalidate.get(f.get("id"))
                    if reason and not f.get("invalid_at"):
                        f["invalid_at"] = now_iso
                        f["invalid_reason"] = f"fact_resolve: {reason}"
                        changed = True
                        resolved += 1
                if changed:
                    body_start = content.find("---", 3)
                    body = content[body_start + 3:] if body_start != -1 else ""
                    with locked_file(fact_file):
                        atomic_write_text(fact_file, _write_fact_frontmatter(frontmatter) + body)
                    try:
                        _store().facts_idx.update_file(fact_file, root=FACTS_ROOT)
                    except StoreUnavailable:
                        pass
        return {"entity": entity, "resolved": resolved,
                "remaining": len(contradictions) - resolved}
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
        Tool(name="fact_profile", description=f"Synthesized profile for an entity: facts grouped by category, capped at {FACT_RANK_CAP_SEED} per category. Pass `query` to rank each category by relevance instead of recency.", inputSchema={
            "type": "object", "properties": {
                "entity": {"type": "string"},
                "query": {"type": "string", "description": "Rank facts by relevance to this text"},
                "limit_per_category": {"type": "integer", "description": f"Facts kept per category (default {FACT_RANK_CAP_SEED})"},
            }, "required": ["entity"]}),
        Tool(name="fact_check", description="Detect contradictions in stored facts for an entity.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "category": {"type": "string"}}, "required": ["entity"]}),
        Tool(name="fact_resolve", description="Report contradictions between an entity's facts. Reports only unless auto_resolve=true, which marks the lower-confidence side invalid (never expired).", inputSchema={
            "type": "object", "properties": {
                "entity": {"type": "string"},
                "auto_resolve": {"type": "boolean", "description": "Mark the weaker side invalid (default false)"},
            }, "required": ["entity"]}),
        Tool(name="fact_invalidate", description="Expire facts that are no longer current. Sets expired_at on matching facts.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "category": {"type": "string"}, "fact_substring": {"type": "string", "description": "Match facts containing this text"}, "ended": {"type": "string", "description": "ISO date when fact stopped being true"}, "reason": {"type": "string", "description": "Why the fact was expired"}}, "required": ["entity", "ended"]}),
        Tool(name="fact_relate", description="Add a typed relationship edge between two entities.", inputSchema={
            "type": "object", "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "type": {"type": "string", "description": "Relationship type (e.g. built_on, uses, part_of, related_to)"}, "confidence": {"type": "number"}, "provenance": {"type": "string", "enum": ["STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"]}, "source_doc": {"type": "string"}}, "required": ["source", "target", "type"]}),
        Tool(name="fact_relationships", description="Get all relationships for an entity (inbound + outbound edges).", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "direction": {"type": "string", "enum": ["in", "out", "both"]}, "type": {"type": "string"}}, "required": ["entity"]}),
        Tool(name="fact_path", description="Find shortest path between two entities via relationship graph.", inputSchema={
            "type": "object", "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "max_hops": {"type": "integer"}}, "required": ["source", "target"]}),
        Tool(name="fact_neighbors", description=f"Neighborhood subgraph around an entity within N hops. Truncates at {_FACT_NEIGHBORS_MAX_NODES} nodes / {_FACT_NEIGHBORS_MAX_EDGES} edges; hub entities at hops=2 will truncate — narrow with hops=1 or a higher min_confidence.", inputSchema={
            "type": "object", "properties": {
                "entity": {"type": "string"},
                "hops": {"type": "integer", "description": "Traversal depth (default 1)"},
                "min_confidence": {"type": "number", "description": "Drop edges below this confidence (default 0.5)"},
                "max_nodes": {"type": "integer", "description": f"Cap on returned nodes (default {_FACT_NEIGHBORS_MAX_NODES})"},
                "max_edges": {"type": "integer", "description": f"Cap on returned edges (default {_FACT_NEIGHBORS_MAX_EDGES})"},
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
