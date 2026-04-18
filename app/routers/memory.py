"""Memory (Obsidian vault) browse/search/read/save endpoints."""

import json
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse


router = APIRouter()

_VAULT = Path.home() / "obsidian"
_VAULT_SEGMENTS = ["memory", "knowledge", "projects", "agents", "personal", "work", "skills"]


@router.get("/api/memory/stats")
async def memory_stats():
    if not _VAULT.exists():
        return JSONResponse({"docCount": 0, "tagCount": 0, "types": {}, "topTags": [], "lastRefresh": ""})
    types: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    doc_count = 0
    for seg in _VAULT_SEGMENTS:
        seg_dir = _VAULT / seg
        if not seg_dir.is_dir():
            continue
        count = 0
        for f in seg_dir.rglob("*.md"):
            count += 1
            try:
                head = f.read_text(encoding="utf-8")[:2000]
                if head.startswith("---"):
                    parts = head.split("---", 2)
                    if len(parts) >= 3:
                        fm = yaml.safe_load(parts[1]) or {}
                        for t in (fm.get("tags") or []):
                            if isinstance(t, str):
                                tag_counts[t] = tag_counts.get(t, 0) + 1
            except Exception:
                pass
        types[seg] = count
        doc_count += count
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:20]
    return JSONResponse({
        "docCount": doc_count,
        "tagCount": len(tag_counts),
        "types": types,
        "topTags": [{"tag": t, "count": c} for t, c in top_tags],
        "lastRefresh": datetime.now().isoformat(),
    })


@router.get("/api/memory/search")
async def memory_search(q: str = "", limit: int = 10, scope: str = ""):
    if not q:
        return JSONResponse({"query": q, "results": []})
    payload = json.dumps({
        "searches": [{"type": "lex", "query": q}, {"type": "vec", "query": q}],
        "limit": limit,
        "collections": scope.split(",") if scope else _VAULT_SEGMENTS,
    }).encode()
    req = urllib.request.Request("http://localhost:8181/query", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        results = [{"path": r.get("file", ""), "title": r.get("title", ""), "score": r.get("score", 0), "snippet": r.get("snippet", ""), "summary": r.get("summary", "")} for r in data.get("results", [])]
        return JSONResponse({"query": q, "results": results})
    except Exception as e:
        return JSONResponse({"query": q, "error": str(e), "results": []})


@router.get("/api/memory/browse")
async def memory_browse(path: str = ""):
    """Browse vault directory structure."""
    browse_dir = _VAULT / path if path else _VAULT
    if not browse_dir.exists() or not browse_dir.is_dir():
        return JSONResponse({"path": path, "entries": []})
    entries = []
    for entry in sorted(browse_dir.iterdir()):
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        if entry.is_dir():
            children = sum(1 for _ in entry.iterdir() if not _.name.startswith("."))
            entries.append({"name": entry.name, "type": "dir", "children": children})
        elif entry.suffix == ".md":
            title = entry.stem
            try:
                head = entry.read_text(encoding="utf-8")[:500]
                if head.startswith("---"):
                    parts = head.split("---", 2)
                    if len(parts) >= 3:
                        fm = yaml.safe_load(parts[1]) or {}
                        title = fm.get("title", title)
            except Exception:
                pass
            entries.append({"name": entry.name, "type": "file", "size": entry.stat().st_size, "title": title})
    return JSONResponse({"path": path, "entries": entries})


@router.get("/api/memory/read")
async def memory_read(path: str = ""):
    """Read a vault markdown file with frontmatter."""
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    filepath = _VAULT / path
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    content = filepath.read_text(encoding="utf-8")
    fm = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
    return JSONResponse({
        "path": path,
        "frontmatter": fm,
        "content": body,
        "lineCount": content.count("\n") + 1,
    })


@router.post("/api/memory/save")
async def memory_save(request: Request):
    """Save a vault markdown file."""
    data = await request.json()
    path = data.get("path", "")
    content = data.get("content", "")
    frontmatter = data.get("frontmatter")
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    filepath = _VAULT / path
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if frontmatter:
        out = f"---\n{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}---\n\n{content}"
    else:
        out = content
    filepath.write_text(out, encoding="utf-8")
    return JSONResponse({"ok": True})
