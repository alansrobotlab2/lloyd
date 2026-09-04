"""Entity facts/graph endpoints."""

import datetime as _dt
import json
import re
import time
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
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
# The typed knowledge graph lives in app.kg_store (SQLite). Semantic edge
# types (uses, part_of, implements, depends_on, …) with confidence,
# provenance and temporal bookkeeping.

# Graph cache. Key: mtime-signature of the facts root. Invalidated automatically
# when any entity dir is touched. Computing the graph over ~1k entities / ~3.5k
# fact files costs ~500ms cold, so caching repeat UI loads is worth it.
_GRAPH_CACHE: dict = {"sig": None, "payload": None, "built_at": 0.0}

# Entity-list cache. Cold scan (~150ms) globs each entity dir twice; warm hits
# this cache and returns immediately. Same invalidation key as the graph since
# both depend on the facts/ tree shape.
_ENTITIES_CACHE: dict = {"sig": None, "payload": None}


def _facts_signature() -> tuple:
    """Cheap cache key: the store's version plus its last reindex. Any commit
    from any process (a nightly classifier run, a fact added in chat) moves
    `PRAGMA data_version`, so the UI never serves a graph the store has
    already replaced. Falls back to the facts-tree mtimes when the store is
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


def _load_active_edges() -> list[dict]:
    """Every currently-active edge. `[]` when the store cannot be read — the
    Memory page degrades to facts-only rather than 500ing."""
    try:
        return _store().edges.active()
    except StoreUnavailable:
        return []


# Entity name normalization lives in app.entity_naming (shared across writers
# and readers). Imported at the top of this file as `_normalize_entity`.


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


@router.get("/api/entities")
async def list_entities():
    if not _FACTS_ROOT.exists():
        return JSONResponse({"entities": [], "total": 0})
    sig = _facts_signature()
    if _ENTITIES_CACHE["sig"] == sig and _ENTITIES_CACHE["payload"] is not None:
        return JSONResponse(_ENTITIES_CACHE["payload"])
    entities = []
    for d in sorted(_FACTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        fact_count = 0
        cats: set[str] = set()
        for f in d.glob("*.md"):
            fact_count += 1
            stem = f.stem
            if "-" in stem:
                cats.add(stem.split("-", 1)[-1])
        entities.append({"name": d.name, "factCount": fact_count, "categories": list(cats)})
    payload = {"entities": entities, "total": len(entities)}
    _ENTITIES_CACHE["sig"] = sig
    _ENTITIES_CACHE["payload"] = payload
    return JSONResponse(payload)


@router.get("/api/entity")
async def entity_detail(name: str = ""):
    """Get entity facts and relationships."""
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    # Resolve aliases before touching the filesystem so callers don't have to
    # know the canonical casing / form.
    name = _normalize_entity(name)
    entity_dir = _FACTS_ROOT / name
    if not entity_dir.exists():
        return JSONResponse({"name": name, "facts": [], "relationships": [], "definition": None, "summary": None})
    facts = []
    for f in sorted(entity_dir.glob("*.md")):
        try:
            content = f.read_text(encoding="utf-8")
            if not content.startswith("---"):
                continue
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue
            fm = _yaml_load(parts[1]) or {}
            # Overview files carry type=overview and are not part of the fact list.
            if fm.get("type") == "overview":
                continue
            for fact_item in (fm.get("facts") or []):
                if isinstance(fact_item, dict):
                    facts.append({
                        "id": fact_item.get("id", ""),
                        "fact": fact_item.get("fact", ""),
                        "confidence": fact_item.get("confidence", 0),
                        "category": fact_item.get("category", fm.get("category", "")),
                        "event_date": _jsonable(fact_item.get("event_date")),
                    })
        except Exception:
            continue
    definition, summary = _read_overview(entity_dir)
    # Pull typed edges from the authoritative classifier-maintained graph.
    # Exact-match on source/target — no substring matching.
    relationships = []
    for edge in _load_active_edges():
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        etype = edge.get("type", "related_to")
        if etype == "mentions":
            continue
        if src == name or tgt == name:
            relationships.append({
                "source": src,
                "target": tgt,
                "type": edge.get("type", "related_to"),
                "score": edge.get("confidence", 1.0),
            })
    return JSONResponse({
        "name": name,
        "facts": facts,
        "relationships": relationships,
        "definition": definition,
        "summary": summary,
    })


@router.get("/api/entity-graph")
async def entity_graph():
    """Build entity graph from the typed-relationship classifier output.

    Nodes: one per entity dir under app.paths.VAULT_FACTS_ROOT, plus any
    endpoint that appears in the active relationship graph without a local
    fact dir (these are rendered with factCount=0).

    Edges: active (non-expired) edges from <FACTS_ROOT>/_relationships.json
    with their real semantic type (uses, part_of, implements, etc.). No
    substring co-occurrence, no fabricated "has-facts" edges.
    """
    if not _FACTS_ROOT.exists():
        return JSONResponse({"nodes": [], "edges": []})
    # Serve from cache when neither the facts tree nor the typed graph changed.
    sig = _facts_signature()
    if _GRAPH_CACHE["sig"] == sig and _GRAPH_CACHE["payload"] is not None:
        return JSONResponse(_GRAPH_CACHE["payload"])

    # Build node index from facts/ dirs (for type, factCount, definition).
    nodes_by_id: dict[str, dict] = {}
    for d in sorted(_FACTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        fact_count = 0
        categories = []
        for f in d.glob("*.md"):
            stem_suffix = f.stem.split("-", 1)[-1] if "-" in f.stem else ""
            if stem_suffix == "overview":
                continue
            fact_count += 1
            if stem_suffix:
                categories.append(stem_suffix)
        node_type = categories[0] if categories else "entity"
        nodes_by_id[name] = {
            "id": name,
            "label": name,
            "type": node_type,
            "factCount": fact_count,
            "definition": _read_definition(d),
        }

    # Pull typed edges. Collapse directional duplicates into one edge per
    # unordered pair, keeping the highest-confidence direction as dominant.
    # Skip co-occurrence "mentions" edges — they're structural noise.
    pair_edges: dict[tuple[str, str], dict] = {}
    for edge in _load_active_edges():
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if not src or not tgt or src == tgt:
            continue
        etype = edge.get("type", "related_to")
        if etype == "mentions":
            continue
        conf = float(edge.get("confidence", 1.0))
        key = (min(src, tgt), max(src, tgt))
        existing = pair_edges.get(key)
        if existing is None or conf > existing["weight"]:
            pair_edges[key] = {
                "source": src,
                "target": tgt,
                "type": etype,
                "weight": conf,
                "bidirectional": False,
            }
        elif existing is not None:
            existing["bidirectional"] = True

    # Add placeholder nodes for edge endpoints without their own fact dir
    # (orphan entities referenced by the classifier but not yet extracted).
    for edge in pair_edges.values():
        for endpoint in (edge["source"], edge["target"]):
            if endpoint not in nodes_by_id:
                nodes_by_id[endpoint] = {
                    "id": endpoint,
                    "label": endpoint,
                    "type": "entity",
                    "factCount": 0,
                    "definition": None,
                }

    nodes = sorted(nodes_by_id.values(), key=lambda n: n["id"])
    edges = sorted(pair_edges.values(), key=lambda e: -e["weight"])
    payload = {"nodes": nodes, "edges": edges}
    _GRAPH_CACHE["sig"] = sig
    _GRAPH_CACHE["payload"] = payload
    _GRAPH_CACHE["built_at"] = time.time()
    return JSONResponse(payload)
