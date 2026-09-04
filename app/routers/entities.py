"""Entity facts/graph endpoints.

Everything structural is read from `app.kg_store`; the markdown tree is only
opened for the things it alone holds (an entity's overview prose and its
fact text). Before the store this router re-walked 23,560 directories and
re-parsed 60,622 markdown files per cold request, and it read the whole
relationship JSON per call to answer "which edges touch this entity".
"""

import datetime as _dt
import re
import time
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.entity_naming import normalize as _normalize_entity
from app.kg_store import StoreUnavailable, store as _store
from app.paths import VAULT_FACTS_ROOT

# Prefer the C-based YAML loader when available (~10x faster than safe_load).
# All frontmatter we parse here is trusted vault content, so SafeLoader-level
# restrictions suffice.
try:
    from yaml import CSafeLoader as _YamlLoader  # type: ignore
except ImportError:  # pragma: no cover
    from yaml import SafeLoader as _YamlLoader  # type: ignore


def _yaml_load(text: str):
    return yaml.load(text, Loader=_YamlLoader)


def _jsonable(v):
    """Coerce yaml-parsed values into JSON-safe scalars (dates → ISO strings)."""
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    return v


router = APIRouter()

_FACTS_ROOT = VAULT_FACTS_ROOT

# Edge types that are co-occurrence noise rather than a stated relationship.
# Kept out of the graph view; `fact_neighbors` still returns them.
_STRUCTURAL_EDGE_TYPES = frozenset({"mentions", "co_mentioned", "wiki_link_co_occurrence"})

# Response caches. Keyed on the store's version plus its last reindex, so a
# commit from any process — a classifier run, a fact added in chat —
# invalidates them on the next request.
_GRAPH_CACHE: dict = {}
_ENTITIES_CACHE: dict = {}
_CACHE_MAX = 8


def _facts_signature() -> tuple:
    """Cheap cache key. Falls back to the facts-tree mtimes when the store is
    unavailable, so the page still renders."""
    try:
        st = _store()
        return ("store", st.version(), st.facts_idx.last_reindex())
    except StoreUnavailable:
        try:
            mtimes = [d.stat().st_mtime for d in _FACTS_ROOT.iterdir() if d.is_dir()]
        except FileNotFoundError:
            return ("fs", 0, 0.0)
        return ("fs", len(mtimes), max(mtimes) if mtimes else 0.0)


def _cache_get(cache: dict, key):
    hit = cache.get(key)
    return hit[1] if hit else None


def _cache_put(cache: dict, key, payload):
    if len(cache) >= _CACHE_MAX:
        cache.clear()
    cache[key] = (time.time(), payload)
    return payload


def _load_active_edges() -> list[dict]:
    """Every currently-active edge. `[]` when the store cannot be read — the
    Memory page degrades to facts-only rather than 500ing."""
    try:
        return _store().edges.active()
    except StoreUnavailable:
        return []


def _overview_file(entity_dir: Path) -> Path:
    return entity_dir / f"{entity_dir.name}-overview.md"


def _read_overview(entity_dir: Path) -> tuple[str | None, str | None]:
    """Return (definition, summary) from {entity}-overview.md, or (None, None)."""
    path = _overview_file(entity_dir)
    if not path.exists():
        return None, None
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None, None
    if not content.startswith("---"):
        return None, None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None, None
    try:
        fm = _yaml_load(parts[1]) or {}
    except Exception:
        fm = {}
    definition = fm.get("definition")
    body = parts[2].lstrip("\n")
    if body.startswith("# Summary"):
        body = body.split("\n", 1)[1].lstrip("\n") if "\n" in body else ""
    summary = body.strip() or None
    return definition, summary


def _read_definition(entity_dir: Path) -> str | None:
    """Cheap: read only frontmatter definition, for graph tooltips."""
    path = _overview_file(entity_dir)
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 2:
        return None
    try:
        fm = _yaml_load(parts[1]) or {}
    except Exception:
        return None
    return fm.get("definition")


# ── /api/entities ────────────────────────────────────────────────────────────

def _build_entity_list(q: str, limit: int, offset: int) -> dict:
    st = _store()
    counts = st.facts_idx.entity_fact_counts()
    kinds = st.entities.kinds()
    names = st.entities.search(q, limit=10000) if q else st.entities.all()
    # Entities with facts first, then by fact count. An alphabetical list of
    # 23,564 names put `#160` and `_append_messages` at the top of the
    # sidebar and the entities anyone would look for thousands of rows down.
    names.sort(key=lambda n: (-counts.get(n, 0), n.lower()))
    total = len(names)
    page = names[offset:offset + limit]
    entities = [{
        "name": n,
        "factCount": counts.get(n, 0),
        "kind": kinds.get(n, "entity"),
        "categories": st.facts_idx.categories_for(n),
    } for n in page]
    return {"entities": entities, "total": total, "offset": offset,
            "limit": limit, "returned": len(entities), "query": q or None}


@router.get("/api/entities")
async def list_entities(
    q: str = Query("", description="Filter by name substring"),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """Entity list, most-populated first.

    `limit` was accepted by the frontend's URL and ignored by the handler, so
    every load shipped all 23,564 entities (2.2 MB) to render a sidebar.
    """
    try:
        _store()
    except StoreUnavailable:
        return JSONResponse({"entities": [], "total": 0, "offset": 0,
                             "limit": limit, "returned": 0, "query": q or None})
    key = (_facts_signature(), q, limit, offset)
    cached = _cache_get(_ENTITIES_CACHE, key)
    if cached is not None:
        return JSONResponse(cached)
    payload = await run_in_threadpool(_build_entity_list, q, limit, offset)
    return JSONResponse(_cache_put(_ENTITIES_CACHE, key, payload))


# ── /api/entity ──────────────────────────────────────────────────────────────

def _build_entity_detail(name: str, include_expired: bool) -> dict:
    st = _store()
    rows = st.facts_idx.for_entity(name, include_expired=include_expired)
    facts = [{
        "id": r["fact_id"] or "",
        "fact": r["fact"] or "",
        "confidence": r["confidence"] if r["confidence"] is not None else 0,
        "category": r["category"],
        "created_at": r["created_at"],
        "source_doc": r["source_doc"],
        "provenance": r["provenance"],
        "expired_at": r["expired_at"],
        "invalid_at": r["invalid_at"],
        "event_date": r["valid_at"],
    } for r in rows]

    entity_dir = _FACTS_ROOT / name
    definition, summary = _read_overview(entity_dir) if entity_dir.exists() else (None, None)
    row = st.entities.get(name) or {}
    if not definition:
        definition = row.get("definition")

    inbound, outbound = [], []
    for edge in st.edges.active(either=name):
        if edge["type"] in _STRUCTURAL_EDGE_TYPES:
            continue
        item = {
            "source": edge["source"], "target": edge["target"], "type": edge["type"],
            "score": edge["confidence"], "provenance": edge["provenance"],
            "created_at": edge["created_at"], "source_doc": edge["source_doc"],
            "evidence": edge["evidence"],
            # The other end, so the UI can render an inbound edge without
            # having to work out which side the entity is on. It used to
            # print `target` regardless, so every inbound edge displayed the
            # entity's own name.
            "other": edge["target"] if edge["source"] == name else edge["source"],
        }
        (outbound if edge["source"] == name else inbound).append(item)

    return {
        "name": name,
        "kind": row.get("kind") or "entity",
        "facts": facts,
        "factCount": len(facts),
        # Flat list kept for the existing UI; the split is the useful shape.
        "relationships": outbound + inbound,
        "outbound": outbound,
        "inbound": inbound,
        "aliases": st.aliases.for_canonical(name),
        "definition": definition,
        "summary": summary,
        "includeExpired": include_expired,
    }


@router.get("/api/entity")
async def entity_detail(
    name: str = "",
    include_expired: bool = Query(False, description="Include expired/invalidated facts"),
):
    """Entity facts and relationships.

    Facts come from the store's index, which means expired and invalidated
    ones are filtered by default — the page used to show them with no
    indication, so a fact the graph had retired still read as current.
    """
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    # Resolve aliases before touching anything so callers don't have to know
    # the canonical casing / form.
    name = _normalize_entity(name)
    try:
        _store()
    except StoreUnavailable:
        return JSONResponse({"name": name, "kind": "entity", "facts": [], "factCount": 0,
                             "relationships": [], "outbound": [], "inbound": [],
                             "aliases": [], "definition": None, "summary": None,
                             "includeExpired": include_expired})
    return JSONResponse(await run_in_threadpool(_build_entity_detail, name, include_expired))


# ── /api/entity-graph ────────────────────────────────────────────────────────

def _build_entity_graph(include_isolated: bool, limit: int, min_confidence: float) -> dict:
    st = _store()
    kinds = st.entities.kinds()
    counts = st.facts_idx.entity_fact_counts()

    # Collapse directional duplicates into one edge per unordered pair,
    # keeping the highest-confidence direction as dominant. `bidirectional`
    # means both directions genuinely exist — it used to be set whenever a
    # second edge of ANY direction landed on the pair, including two edges
    # the same way round.
    pair_edges: dict[tuple[str, str], dict] = {}
    seen_directions: dict[tuple[str, str], set] = {}
    for edge in st.edges.active():
        src, tgt = edge["source"], edge["target"]
        if not src or not tgt or src == tgt:
            continue
        if edge["type"] in _STRUCTURAL_EDGE_TYPES:
            continue
        conf = float(edge["confidence"] or 0.0)
        if conf < min_confidence:
            continue
        key = (min(src, tgt), max(src, tgt))
        seen_directions.setdefault(key, set()).add((src, tgt))
        existing = pair_edges.get(key)
        if existing is None or conf > existing["weight"]:
            pair_edges[key] = {
                "source": src, "target": tgt, "type": edge["type"], "weight": conf,
                "provenance": edge["provenance"], "created_at": edge["created_at"],
                "bidirectional": False,
            }
    for key, e in pair_edges.items():
        e["bidirectional"] = len(seen_directions.get(key, ())) > 1

    edges = sorted(pair_edges.values(), key=lambda e: -e["weight"])[:limit]
    connected = {n for e in edges for n in (e["source"], e["target"])}

    def _node(name: str) -> dict:
        return {
            "id": name,
            "label": name,
            # The registry's kind, not the entity's first fact category. The
            # old value made a legend of category names (`state`, `goal`) and
            # called that a node type.
            "type": kinds.get(name, "entity"),
            "factCount": counts.get(name, 0),
            # Definitions are read from disk one file at a time; at 23,565
            # nodes that was 23,565 opens per cold graph build. The UI fetches
            # one via /api/entity when a node is selected.
            "definition": None,
        }

    names = sorted(connected)
    if include_isolated:
        names = sorted(connected | set(st.entities.all()))
    return {"nodes": [_node(n) for n in names], "edges": edges,
            "nodeCount": len(names), "edgeCount": len(edges),
            "includeIsolated": include_isolated, "minConfidence": min_confidence}


@router.get("/api/entity-graph")
async def entity_graph(
    include_isolated: bool = Query(False, description="Include entities with no edges"),
    limit: int = Query(3000, ge=1, le=20000, description="Max edges returned"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
):
    """Entity graph from the typed edge store.

    Nodes default to those with at least one edge. Including the 20,000
    isolated entities made a 5.7 MB payload the browser then laid out, and
    nothing could be seen in it.
    """
    try:
        _store()
    except StoreUnavailable:
        return JSONResponse({"nodes": [], "edges": [], "nodeCount": 0, "edgeCount": 0,
                             "includeIsolated": include_isolated,
                             "minConfidence": min_confidence})
    key = (_facts_signature(), include_isolated, limit, min_confidence)
    cached = _cache_get(_GRAPH_CACHE, key)
    if cached is not None:
        return JSONResponse(cached)
    payload = await run_in_threadpool(_build_entity_graph, include_isolated, limit, min_confidence)
    return JSONResponse(_cache_put(_GRAPH_CACHE, key, payload))
