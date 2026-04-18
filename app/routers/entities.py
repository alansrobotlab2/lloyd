"""Entity facts/graph endpoints."""

import json
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse


router = APIRouter()

_FACTS_ROOT = Path.home() / "obsidian" / "facts"
_RELATIONS_INDEX = Path.home() / "lloyd" / "_pipeline" / "relations-index.json"


@router.get("/api/entities")
async def list_entities():
    if not _FACTS_ROOT.exists():
        return JSONResponse({"entities": [], "total": 0})
    entities = []
    for d in sorted(_FACTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        fact_count = sum(1 for _ in d.glob("*.md"))
        categories = list(set(f.stem.split("-", 1)[-1] for f in d.glob("*.md") if "-" in f.stem))
        entities.append({"name": d.name, "factCount": fact_count, "categories": categories})
    return JSONResponse({"entities": entities, "total": len(entities)})


@router.get("/api/entity")
async def entity_detail(name: str = ""):
    """Get entity facts and relationships."""
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    entity_dir = _FACTS_ROOT / name
    if not entity_dir.exists():
        return JSONResponse({"name": name, "facts": [], "relationships": []})
    facts = []
    for f in sorted(entity_dir.glob("*.md")):
        try:
            content = f.read_text(encoding="utf-8")
            if not content.startswith("---"):
                continue
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue
            fm = yaml.safe_load(parts[1]) or {}
            for fact_item in (fm.get("facts") or []):
                if isinstance(fact_item, dict):
                    facts.append({
                        "id": fact_item.get("id", ""),
                        "text": fact_item.get("fact", ""),
                        "confidence": fact_item.get("confidence", 0),
                        "category": fact_item.get("category", fm.get("category", "")),
                        "eventDate": fact_item.get("event_date"),
                    })
        except Exception:
            continue
    relationships = []
    if _RELATIONS_INDEX.exists():
        try:
            rel_data = json.loads(_RELATIONS_INDEX.read_text(encoding="utf-8"))
            for edge in rel_data.get("edges", []):
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                name_lower = name.lower()
                if name_lower in src.lower() or name_lower in tgt.lower():
                    relationships.append({
                        "source": src,
                        "target": tgt,
                        "type": edge.get("type", "related-to"),
                        "score": edge.get("weight", edge.get("score", 1.0)),
                    })
        except Exception:
            pass
    return JSONResponse({"name": name, "facts": facts, "relationships": relationships})


@router.get("/api/entity-graph")
async def entity_graph():
    """Build entity graph from facts directory using cross-entity references."""
    if not _FACTS_ROOT.exists():
        return JSONResponse({"nodes": [], "edges": []})
    nodes = []
    entity_names: set[str] = set()
    entity_facts: dict[str, list[str]] = {}
    for d in sorted(_FACTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        entity_names.add(name)
        fact_count = 0
        fact_texts = []
        categories = []
        for f in d.glob("*.md"):
            fact_count += 1
            if "-" in f.stem:
                categories.append(f.stem.split("-", 1)[-1])
            try:
                content = f.read_text(encoding="utf-8")[:4000]
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        fm = yaml.safe_load(parts[1]) or {}
                        for fact_item in (fm.get("facts") or []):
                            if isinstance(fact_item, dict):
                                fact_texts.append(fact_item.get("fact", ""))
            except Exception:
                pass
        entity_facts[name] = fact_texts
        node_type = categories[0] if categories else "entity"
        nodes.append({
            "id": name,
            "label": name,
            "type": node_type,
            "factCount": fact_count,
        })
    edge_counts: dict[tuple[str, str], int] = {}
    searchable = {n for n in entity_names if len(n) >= 3}
    name_lower_map = {n.lower(): n for n in searchable}
    for entity, facts in entity_facts.items():
        all_text = " ".join(facts).lower()
        for other_lower, other in name_lower_map.items():
            if other == entity:
                continue
            if other_lower in all_text:
                key = (min(entity, other), max(entity, other))
                edge_counts[key] = edge_counts.get(key, 0) + 1
    edges = [
        {"source": src, "target": tgt, "type": "has-facts", "weight": float(count)}
        for (src, tgt), count in sorted(edge_counts.items(), key=lambda x: -x[1])
    ]
    return JSONResponse({"nodes": nodes, "edges": edges})
