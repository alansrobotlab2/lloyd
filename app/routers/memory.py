"""Memory (Obsidian vault) browse/search/read/save endpoints."""

import asyncio
import logging
import re
from datetime import date, datetime
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from agent_mcp.vault import VAULT_SEGMENTS as _VAULT_SEGMENTS
from app.data_root import vault_root

logger = logging.getLogger(__name__)

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


_VAULT = vault_root()

# What is cached here is now ONLY the tag tally. `docCount` and `types` — the two
# numbers the Memory tab actually renders — are answered straight from
# `_stats_signature()`'s per-segment counts, so they cost the stat walk and open no
# note file (#3503). This dict holds the one figure that genuinely needs the note
# contents, plus the two fields that keep its rebuild single and unreaped.
#
# Measured 2026-10-27 over the live vault: a cold `/api/memory/stats` on the
# running backend was 21.7s, the same build out of process 17.6s, of which 17.1s is
# `yaml.safe_load` over 8,528 front-matter blocks and `backlog/` alone is 15.3s —
# against 0.036s for the stat-only signature walk over the same 11 segments and
# 8,608 notes. So 99.8% of what the request used to pay bought nothing but
# `tagCount`. An earlier note here sized this cache against a much smaller vault
# (4,135 files / ~30MB, 2026-09-22) and left the widened walk's cold cost as an
# open question; #3503 closes it by not paying the parse for the counts at all.
#
# And cold is the NORMAL state, not the exception: the signature keys on each
# segment's recursive max mtime, and 11 vault notes were written in the 60 minutes
# before those readings. Hence the hand-off below — a stale signature serves the
# LAST completed tally and queues the rebuild behind the response, rather than
# making whoever happened to open the tab wait 17.6s for it.
_STATS_CACHE: dict = {
    "sig": None,          # signature last built, or last attempted and not retried until the vault moves
    "tally": None,        # {"tagCount", "topTags", "lastRefresh"}; None before the first build lands
    "in_flight": False,   # one rebuild at a time; cleared in a `finally` — see `_rebuild_tag_tally`
    "task": None,         # strong ref: the loop may garbage-collect a task nothing holds
}


def _stats_signature() -> tuple:
    """Per-segment `(count, recursive max mtime, directory exists)`, and the served
    counts themselves.

    Recomputed each call by walking the tree and stat()ing each `.md`: 0.036s over
    the eleven segments and 8,608 notes on 2026-10-27, against 17.6s to read and
    parse the same set. It began as only an invalidation key, which is why the shape
    is a tuple of tuples rather than a dict; since #3503 it also answers `docCount`
    and `types`, which is why each part carries `exists` as well as the count. A
    count of 0 cannot tell "no such segment" from "segment, no notes in it", and
    `_build_stats_payload` has always treated those as different answers — so the
    flag is what lets `_counts_from_signature` reproduce its map exactly instead of
    inventing a `0` entry (#3503 clause 2).
    """
    parts: list = []
    for seg in _VAULT_SEGMENTS:
        seg_dir = _VAULT / seg
        if not seg_dir.is_dir():
            parts.append((seg, 0, 0.0, False))
            continue
        max_mtime = 0.0
        count = 0
        for f in seg_dir.rglob("*.md"):
            # Counted before the stat, not after: a note deleted between the listing
            # and the stat is still one more `.md` than the segment had a moment ago,
            # and `_build_stats_payload` counts it too. The count has to survive the
            # OSError here or the two walks answer differently about the same vault.
            count += 1
            try:
                m = f.stat().st_mtime
            except OSError:
                continue
            if m > max_mtime:
                max_mtime = m
        parts.append((seg, count, max_mtime, True))
    return tuple(parts)


def _counts_from_signature(sig: tuple) -> "tuple[int, dict[str, int]]":
    """`(docCount, types)` taken off the signature — no note is read to make them.

    A segment whose directory does not exist is ABSENT from `types`, which is the
    shape `_build_stats_payload` has always served (it `continue`s before
    `types[seg] = count`), so the signature-derived map and the walked map are
    identical and nothing new appears in the response. Serving `0` for a missing
    directory was the other candidate and is the one this pins down as wrong: it
    would make a renamed or unmounted segment indistinguishable from an empty one,
    which is exactly the silent-zero failure #3160 is about.
    """
    types = {seg: count for seg, count, _mtime, exists in sig if exists}
    return sum(types.values()), types


def _build_stats_payload() -> dict:
    """The tag tally over every searchable segment, by reading and parsing each note.

    The expensive half of the stats route, and now the ONLY thing it still costs a
    requester: measured over the live vault on 2026-10-27, 17.6s out of process here
    (21.7s cold through the running backend) across the eleven segments — 8,608
    notes read whole and YAML-parsed, 17.1s of it `yaml.safe_load`, `backlog/` alone
    15.3s — against 0.036s for `_stats_signature()`, which answers `docCount` and
    `types` without opening a single note (#3503).

    Two properties survive from before that change and both are pinned in
    `tests/test_memory_router_stats.py`. It stays off the event loop: it is `async
    def`-adjacent CPU work, and an inline walk held the loop for its whole duration,
    which at 17.6s is a chat-stream stall per rebuild — so it goes to
    `asyncio.to_thread`. And it stays off the REQUEST path: `memory_stats` hands it
    to a worker thread and serves the previous tally meanwhile, so no request's
    latency is this parse. Its `docCount`/`types` keys are still computed and
    returned — the tests below compare the served map against this walked one — but
    the route no longer reads them.
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


def _empty_tally() -> dict:
    """The tally shape before any build has ever finished: zeros, not a missing key.

    A fresh backend answers its first `/api/memory/stats` with `tagCount: 0` while
    the first build runs. `web/src/components/pages/MemoryPage.tsx` renders
    `stats.tagCount` unguarded, so the key has to be present and numeric on that
    first response — `undefined` would print as an empty span.
    """
    return {"tagCount": 0, "topTags": [], "lastRefresh": ""}


def _kick_tag_tally_rebuild(sig: tuple) -> None:
    """Queue the tag build on a worker thread and return to the requester.

    Not awaited — that is the whole of #3503 clause 3. The alternative that was in
    place, `await asyncio.to_thread(...)` inside the handler, made the first request
    after any vault write pay the 17.6s parse, and any vault write is the normal
    state (see the cache note above). The rebuild still leaves the loop:
    `_rebuild_tag_tally` awaits it through `asyncio.to_thread`, exactly as the
    handler used to.

    `in_flight` is what keeps a 17.6s parse to one at a time no matter how many
    requests find a stale signature; `_STATS_CACHE["task"]` is the strong reference
    that keeps the loop from garbage-collecting the fire-and-forget task while it is
    still running.
    """
    if _STATS_CACHE["in_flight"]:
        return
    _STATS_CACHE["in_flight"] = True
    _STATS_CACHE["task"] = asyncio.create_task(_rebuild_tag_tally(sig))


async def _rebuild_tag_tally(sig: tuple) -> None:
    """Build the tally in a thread, publish it, and always release the flag.

    The `finally` is the difference between a vault that is briefly unreadable and a
    Memory tab that never recovers (#3503 clause 4): a raising build that left
    `in_flight` set would make every later request skip the rebuild, wedging the
    tally at whatever it last was — or at zeros, forever, if that was the first
    attempt.

    Recording `sig` on the failure path too is deliberate and is what keeps this a
    retry rather than a hammer: without it, every later request re-queues a parse
    that is known to fail, and a 17.6s YAML walk burning CPU in the background of a
    live chat stream is not something to repeat per tab-open. So a failed signature
    is retried when the vault moves next, which is when it may well succeed.
    """
    try:
        payload = await asyncio.to_thread(_build_stats_payload)
        _STATS_CACHE["tally"] = {
            "tagCount": payload["tagCount"],
            "topTags": payload["topTags"],
            "lastRefresh": payload["lastRefresh"],
        }
        _STATS_CACHE["sig"] = sig
    except Exception:
        _STATS_CACHE["sig"] = sig
        # Best-effort by construction: the served counts are already correct without
        # this, so a failed tally keeps the previous one rather than failing the
        # route. Say so, because otherwise a wedged tag count looks like a vault
        # that has no tags.
        logger.warning("memory stats: tag tally rebuild failed; serving the previous tally", exc_info=True)
    finally:
        _STATS_CACHE["in_flight"] = False
        _STATS_CACHE["task"] = None


@router.get("/api/memory/stats")
async def memory_stats():
    if not _VAULT.exists():
        return JSONResponse({"docCount": 0, "types": {}, **_empty_tally()})
    # stat() only — 0.036s over the eleven segments, and the whole of what this
    # route now does synchronously. Both numbers the Memory tab renders come off it.
    sig = _stats_signature()
    doc_count, types = _counts_from_signature(sig)
    tally = _STATS_CACHE["tally"]
    if tally is None or _STATS_CACHE["sig"] != sig:
        # Deliberately not awaited: the counts above are already this vault's truth,
        # and the tag tally the parse produces is served from the last completed
        # build while the new one runs behind the response.
        _kick_tag_tally_rebuild(sig)
        if tally is None:
            tally = _empty_tally()
    return JSONResponse({"docCount": doc_count, "types": types, **tally})


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
