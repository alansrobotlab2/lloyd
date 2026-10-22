"""Memory (Obsidian vault) browse/search/read/save endpoints."""

import asyncio
import re
from datetime import date, datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from agent_mcp.vault import VAULT_SEGMENTS as _VAULT_SEGMENTS

# The searchable segment set is the recall's own, imported and not restated.
# #3160: this file and `mc_ui.py` each carried a hand-maintained copy of seven
# names, six of them members of this list and one — `agents` — a name no qmd
# index has ever declared (it appears in both `~/.config/qmd/index.yml` and the
# tracked `agent-services/conf/qmd-index.yml` only on `subliminal`'s `ignore:`
# list) and no vault has ever held. qmd answers an undeclared collection with
# HTTP 200 and `{"results": []}` instead of an error (measured on the live daemon
# 2026-10-22), so the copy spent a search leg on nothing while leaving
# `architecture`, `lloyd`, `autonomy`, `backlog` and `people` unreachable from
# the Memory tab — in the query below, in the stats walk, and in the navigate
# brief's `vault_file_count`.
#
# It is an import rather than a copy because the copy could not fail: adding a
# twelfth segment here used to leave Mission Control searching eleven, and the
# only symptom was an empty search box. A name added here that the daemon does
# not have is now caught by
# `tests/test_qmd_index_template.py::test_every_collection_the_memory_route_names_is_declared_by_the_template`.

router = APIRouter()


def _jsonable(obj):
    """Recursively convert yaml.safe_load output to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


# The closing delimiter of a YAML front-matter block: `---` alone on its own
# line. Anchored, because `str.split("---")` also fires on a `---` inside a
# quoted value — a markdown table row (`|---|---|`) is the common one — which
# cuts the block mid-scalar and turns a well-formed note into a parse error.
# Same matcher and same reasoning as `app/routers/dashboard.py::_FM_END_RE`.
_FM_END_RE = re.compile(r"^---[ \t]*$", re.M)


def _fm_span(text: str) -> "tuple[str, str] | None":
    """Split `text` at its front-matter fence, returning `(raw_yaml, body)`.

    None when the text does not open with a `---` line or that block never
    closes. Whole-text and per-line by construction: no byte cap, so a note
    whose front matter runs past any prefix length is still found (measured
    2026-09-22 over the seven stats segments: 18 notes closed their fence past
    the 2,000-char cap this file used, the largest past byte 5,152), and no
    substring match, so a `---` inside a quoted value is not mistaken for the
    fence.
    """
    if not text.startswith("---"):
        return None
    opening = text.find("\n")
    if opening < 0:
        return None
    end = _FM_END_RE.search(text, opening)
    if end is None:
        return None
    return text[opening + 1:end.start()], text[end.end():]


def _parse_fm(raw: str) -> dict:
    """Front-matter YAML as a dict; {} if it is not a mapping that parses.

    JSON-safe (`_jsonable`) because every caller here serialises the result:
    now that the fence is found in long front matter too, a `timestamp:
    2026-05-20` reaching `title` would otherwise 500 the whole route.
    """
    try:
        fm = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}
    # A block that parses to a list or a bare string is not front matter;
    # returning it would blow up on the caller's first `.get`.
    return _jsonable(fm) if isinstance(fm, dict) else {}


def _split_frontmatter(text: str) -> "tuple[dict, str]":
    """`(frontmatter, body)` for a markdown note, front matter included or not.

    No front matter — or front matter whose YAML will not parse — yields
    `({}, text)`, which is the caller's existing fallback (filename as title,
    file as body) and the right answer for both: neither is something the
    Memory tab can fix. Callers that must tell those two apart parse `_fm_span`
    themselves, the way `/api/memory/save` has to to keep rejecting invalid
    YAML.
    """
    span = _fm_span(text)
    if span is None:
        return {}, text
    raw, body = span
    return _parse_fm(raw), body


_VAULT = Path.home() / "obsidian"

# Cached stats payload. Keyed by a mtime signature over the segment dirs.
# Warm is the signature walk, not the parse: ~0.06s over the eleven segments on
# 2026-10-22. The cold cost is reading every .md in full and parsing its YAML,
# and #3160 moved it by widening the walk from the six directories the router's
# own seven-name list actually found (2.4s, 4,753 notes, 40.3 MB) to the eleven
# the search searches (17.0s, 8,157 notes, 107.6 MB) — both measured on the live
# vault the same day, which is why `_build_stats_payload` runs in a thread. An
# earlier note here sized this cache against ~1.9s for the 4,135-file / ~30MB
# vault of 2026-09-22, and the widened walk is not sized for that any more:
# whether 17.0s on a cold cache is itself acceptable is the ruling #3160 owes,
# and it needs a post-restart measurement, not a diff.
_STATS_CACHE: dict = {"sig": None, "payload": None}


def _stats_signature() -> tuple:
    """Cheap signature: per-segment (recursive max mtime, file count). Recomputed
    each call by walking the tree and stat()ing each .md — still much cheaper
    than re-reading each file and parsing its YAML."""
    parts: list = []
    for seg in _VAULT_SEGMENTS:
        seg_dir = _VAULT / seg
        if not seg_dir.is_dir():
            parts.append((seg, 0, 0.0))
            continue
        max_mtime = 0.0
        count = 0
        for f in seg_dir.rglob("*.md"):
            try:
                m = f.stat().st_mtime
            except OSError:
                continue
            count += 1
            if m > max_mtime:
                max_mtime = m
        parts.append((seg, count, max_mtime))
    return tuple(parts)


def _build_stats_payload() -> dict:
    """Every searchable segment's note count and tag tally, by reading each note.

    Blocking and expensive, which is why the route below hands it to a thread:
    measured over the live vault on 2026-10-22, 17.0s cold across the eleven
    segments — 8,157 notes, 107.6 MB read whole and YAML-parsed — against 2.4s
    for the six directories the seven-name copy this replaced actually found
    (4,753 notes, 40.3 MB). An `async def` that walked the vault inline held the
    event loop for the whole walk, so at 2.4s that was already a chat-stream
    stall and at 17s it is one every cold cache would cause; whether 17s on a
    cold cache is itself acceptable is the ruling #3160 owes, and it is not a
    reason to block the loop either way.
    """
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
                fm, _ = _split_frontmatter(f.read_text(encoding="utf-8"))
            except Exception:
                # Undecodable or unreadable: counted as a document, no tags.
                continue
            for t in (fm.get("tags") or []):
                if isinstance(t, str):
                    tag_counts[t] = tag_counts.get(t, 0) + 1
        types[seg] = count
        doc_count += count
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:20]
    return {
        "docCount": doc_count,
        "tagCount": len(tag_counts),
        "types": types,
        "topTags": [{"tag": t, "count": c} for t, c in top_tags],
        "lastRefresh": datetime.now().isoformat(),
    }


@router.get("/api/memory/stats")
async def memory_stats():
    if not _VAULT.exists():
        return JSONResponse({"docCount": 0, "tagCount": 0, "types": {}, "topTags": [], "lastRefresh": ""})
    sig = _stats_signature()
    if _STATS_CACHE["sig"] == sig and _STATS_CACHE["payload"] is not None:
        return JSONResponse(_STATS_CACHE["payload"])
    # Off the event loop, for the reason `_build_stats_payload` gives — the same
    # reason the search below threads its blocking HTTP call. The signature walk
    # above stays here: ~0.06s over the eleven segments, stat() only.
    payload = await asyncio.to_thread(_build_stats_payload)
    _STATS_CACHE["sig"] = sig
    _STATS_CACHE["payload"] = payload
    return JSONResponse(payload)


def _search_collections(scope: str) -> list[str]:
    """The collections this route may name in a qmd query, and nothing else.

    Empty scope → the recall's set, `_VAULT_SEGMENTS` (the shape Mission Control
    sends: `web/src/api.ts` fetches `/memory/search` with `q` and `limit` only).

    A `scope` is restricted exactly the way the recall restricts its own — see
    `agent_mcp/vault.py:1255-1261`: a collection is a scope's FIRST path
    component (`knowledge/youtube` lives in `knowledge`, #670), a name outside
    the segment set is dropped, and a scope whose names are all unknown searches
    everything rather than nothing. That last fallback is the point: an unknown
    name costs a leg and returns no rows with no error, so a retired collection or
    a typo used to read here as "the vault has nothing on that", which is the
    failure #3160 is about, one caller deeper.
    """
    if not scope:
        return list(_VAULT_SEGMENTS)
    segs = [s.strip().strip("/").split("/")[0] for s in scope.split(",") if s.strip()]
    return list(dict.fromkeys(s for s in segs if s in _VAULT_SEGMENTS)) or list(_VAULT_SEGMENTS)


@router.get("/api/memory/search")
async def memory_search(q: str = "", limit: int = 10, scope: str = ""):
    if not q:
        return JSONResponse({"query": q, "results": []})
    # The recall's door and the recall's sanitizer (#1498): a `-term` reaching
    # the vec leg is an HTTP 500 from qmd (#325), and a request opened here
    # rather than through `qmd_query` was a reply `qmd_health` never saw.
    from agent_mcp.vault import _qmd_sanitize, qmd_query
    clean = _qmd_sanitize(q)
    if not clean:
        return JSONResponse({"query": q, "results": []})
    payload = {
        "searches": [{"type": "lex", "query": clean}, {"type": "vec", "query": clean}],
        "limit": limit,
        "collections": _search_collections(scope),
    }
    try:
        # Off the event loop: a blocking HTTP call here stalls every chat stream.
        data = await asyncio.to_thread(qmd_query, payload)
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
                fm, _ = _split_frontmatter(entry.read_text(encoding="utf-8"))
            except Exception:
                fm = {}
            if "title" in fm:
                title = fm["title"]
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
    span = _fm_span(content)
    if span is None:
        # No front matter: the body is the file, unstripped, as before.
        fm, body = {}, content
    else:
        raw, body = span
        fm = _parse_fm(raw)
        body = body.strip()
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
        # Validate embedded frontmatter when content contains its own --- block
        span = _fm_span(out)
        if span is not None:
            try:
                yaml.safe_load(span[0])
            except yaml.YAMLError as e:
                raise HTTPException(status_code=422, detail=f"Invalid frontmatter YAML: {e}")
    filepath.write_text(out, encoding="utf-8")
    return JSONResponse({"ok": True})
