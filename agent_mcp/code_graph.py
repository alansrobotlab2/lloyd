#!/usr/bin/env python3
"""Structural code navigation over graphify's deterministic extraction.

Six `graph_*` tools that answer "who calls this", "what breaks if I change
it", "how do these two things connect" from a real import/call graph
instead of from a grep for the symbol's spelling.

Why a module and not a second MCP server
----------------------------------------
graphify ships `graphify-mcp`, and mounting it would have been one config
line. It is the wrong shape here:

- Lloyd advertises every server's tools under bare names and
  `app/harness/tool_schema.py::build_tool_list` raises on a cross-server
  name collision, so a second server is a permanent collision risk.
- Task subagents pin `DEFAULT_LLOYD_MCP_SERVERS`, so they would not see it.
- `tests/test_mcp_layer.py` requires every configured server to be
  discoverable at test time — including inside a autoimplement worktree, where a
  second daemon is not running.
- `agent-services/supervisor/**` is a protected autoimplement path, so Lloyd
  could never add or repair the program that runs it.
- graphify-mcp has no `affected`, which is the one query a change most
  needs.

So this reads graphify's `graph.json` directly and shells out to the binary
only to *build* it.

The graph is blind across process seams
---------------------------------------
It is an AST extraction of one tree. An HTTP call from the backend to the
aggregator, or an MCP dispatch from `run_query` into a tool handler, is not
an edge — there is no path from `run_prompt_in_session` to `run_query`.
Keep using Grep for string keys, route paths and config names; use this for
symbols.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.types import Tool

from agent_mcp._shared import text_result
from app.paths import LLOYD_HOME

logger = logging.getLogger("lloyd-code-graph")

GRAPH_DIRNAME = "graphify-out"
GRAPH_FILENAME = "graph.json"

# Relations that propagate a change. A reverse walk over these answers
# "what breaks if I change this".
IMPACT_RELATIONS: frozenset[str] = frozenset({
    "calls", "indirect_call", "imports", "imports_from", "references", "uses",
    "inherits", "extends", "implements", "dynamic_import", "re_exports",
    "mixes_in", "embeds", "requires",
})

# Structure, not dependency: a file "contains" a function, a class has a
# "method". Reported on their own line rather than mixed into the call
# lists, and excluded from `graph_path` (every symbol is two hops from
# every other one through its file otherwise).
CONTAINMENT_RELATIONS: frozenset[str] = frozenset({"contains", "method", "defines"})

# graphify's own commentary nodes. Never a code answer.
RATIONALE_RELATIONS: frozenset[str] = frozenset({"rationale_for"})

# Suffixes whose uncommitted edits make the graph stale. `.md` is in
# because graphify extracts document headings too.
SOURCE_SUFFIXES = (".py", ".ts", ".tsx", ".js", ".jsx", ".md")

# A round id, as `scripts/autoimplement/round.py` mints them.
ROUND_ID_RE = re.compile(r"^SM_\d{8}_\d{6}$")

_DEFAULTS: dict[str, Any] = {
    "graphify_bin": "~/.local/bin/graphify",
    "auto_refresh": True,
    "refresh_timeout_s": 120,
    "min_refresh_interval_s": 30,
    "max_cached_roots": 4,
    "max_lines": 60,
}


def _cfg() -> dict:
    """Read config at call time so tests can monkeypatch.setitem CONFIG."""
    try:
        from app.config import CONFIG
        raw = CONFIG.get("code_graph") or {}
    except Exception:
        raw = {}
    out = dict(_DEFAULTS)
    if isinstance(raw, dict):
        out.update({k: v for k, v in raw.items() if v is not None})
    return out


def _binary() -> str:
    return os.path.expanduser(str(_cfg()["graphify_bin"]))


# ---------------------------------------------------------------------------
# Subprocess helpers — never `subprocess.run`, this runs on the event loop
# ---------------------------------------------------------------------------


async def _exec(argv: list[str], *, cwd: str | None = None,
                timeout: float = 30.0) -> tuple[int, str, str]:
    """Run a command off the loop. Returns (rc, stdout, stderr).

    rc is -1 when the binary is missing and -2 on timeout, so callers can
    tell "not installed" from "hung" without parsing stderr.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError:
        return -1, "", f"{argv[0]}: not found"
    except Exception as exc:  # pragma: no cover - defensive
        return -1, "", f"{argv[0]}: {exc}"
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        from agent_mcp.builtin_bash import _kill_proc_tree
        _kill_proc_tree(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        raise
    return (
        proc.returncode or 0,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


async def _exec_or_timeout(argv: list[str], *, cwd: str | None = None,
                           timeout: float = 30.0) -> tuple[int, str, str]:
    try:
        return await _exec(argv, cwd=cwd, timeout=timeout)
    except asyncio.TimeoutError:
        return -2, "", f"{argv[0]}: timed out after {timeout}s"


async def _git(root: Path, *args: str, timeout: float = 20.0) -> tuple[int, str]:
    rc, out, _err = await _exec_or_timeout(
        ["git", "-C", str(root), *args], timeout=timeout)
    return rc, out.strip()


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------


class RootError(ValueError):
    """A root argument that cannot be resolved to a directory."""


def resolve_root(root: str | None) -> Path:
    """Resolve the `root` argument to a real directory.

    Explicit, never inferred from the calling session. Nothing on disk
    links a chat session to an open autoimplement round — `round_start` ledger
    rows carry no session id — so a "bound session's worktree" default
    would silently answer about the wrong checkout.
    """
    raw = (root or "").strip()
    if not raw:
        return Path(os.path.realpath(LLOYD_HOME))
    if ROUND_ID_RE.match(raw):
        from scripts.autoimplement.worktree import worktree_path
        p = worktree_path(raw)
        if not p.is_dir():
            raise RootError(f"round {raw} has no worktree at {p}")
        return Path(os.path.realpath(p))
    expanded = os.path.expanduser(os.path.expandvars(raw))
    if not os.path.isabs(expanded):
        raise RootError(
            f"root must be absolute, an SM_… round id, or omitted for "
            f"{LLOYD_HOME}; got {raw!r}")
    if not os.path.isdir(expanded):
        raise RootError(f"root is not a directory: {expanded}")
    return Path(os.path.realpath(expanded))


def graph_path_for(root: Path) -> Path:
    return root / GRAPH_DIRNAME / GRAPH_FILENAME


# ---------------------------------------------------------------------------
# Loading and caching
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    root: str
    mtime_ns: int
    size: int
    graph: Any                        # networkx.DiGraph
    built_at_commit: str
    n_nodes: int
    n_edges: int
    by_label: dict[str, list[str]] = field(default_factory=dict)
    by_file: dict[str, list[str]] = field(default_factory=dict)
    loaded_at: float = 0.0


_CACHE: "OrderedDict[str, Entry]" = OrderedDict()
_LOCKS: dict[str, asyncio.Lock] = {}
_LAST_REFRESH: dict[str, float] = {}
_WORKTREES: dict[str, Any] = {"at": 0.0, "paths": []}
_VERSION_CACHE: dict[str, str] = {}


def _lock_for(root: str) -> asyncio.Lock:
    lock = _LOCKS.get(root)
    if lock is None:
        lock = _LOCKS[root] = asyncio.Lock()
    return lock


def norm_label(s: str) -> str:
    """Bare comparable form of a node label.

    graphify labels callables `run_query()` and methods `.get_text()`, so a
    model asking for `run_query` must still land on the node.
    """
    s = (s or "").strip()
    while s.endswith("()"):
        s = s[:-2]
    s = s.lstrip(".")
    return s.casefold()


def _is_file_node(attrs: dict) -> bool:
    if (attrs.get("metadata") or {}).get("kind") == "file":
        return True
    return (
        not attrs.get("_callable")
        and attrs.get("file_type") == "code"
        and attrs.get("source_location") == "L1"
    )


def _is_rationale(attrs: dict) -> bool:
    return attrs.get("file_type") == "rationale"


def _build_graph(data: dict) -> Any:
    """Build a DiGraph from graphify's node-link JSON.

    Two shapes are accepted. A finished build writes `links`; an
    interrupted `graphify update --no-cluster` leaves the raw extraction
    keyed `edges` with no `community` on the nodes. Reading only `links`
    made a half-finished refresh look like an empty graph.

    Edges collapse: one (u, v) pair can carry several relations, so each
    edge keeps a `rels` list rather than a single relation, and every
    render walks it.
    """
    import networkx as nx

    g = nx.DiGraph()
    for node in data.get("nodes") or []:
        nid = node.get("id")
        if not nid:
            continue
        attrs = {k: v for k, v in node.items() if k != "id"}
        g.add_node(nid, **attrs)

    raw_edges = data.get("links")
    if not raw_edges:
        raw_edges = data.get("edges") or []
    for link in raw_edges:
        u, v = link.get("source"), link.get("target")
        if not u or not v:
            continue
        # graphify can name an endpoint that never became a node.
        if u not in g:
            g.add_node(u, label=u, file_type="code")
        if v not in g:
            g.add_node(v, label=v, file_type="code")
        rel = {
            "relation": link.get("relation") or link.get("type") or "related",
            "source_file": link.get("source_file") or "",
            "source_location": link.get("source_location") or "",
            "confidence": link.get("confidence") or "",
            "context": link.get("context") or "",
        }
        if g.has_edge(u, v):
            g[u][v]["rels"].append(rel)
        else:
            g.add_edge(u, v, rels=[rel])
    return g


def _index(g: Any) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    by_label: dict[str, list[str]] = {}
    by_file: dict[str, list[str]] = {}
    for nid, attrs in g.nodes(data=True):
        key = norm_label(attrs.get("label") or nid)
        by_label.setdefault(key, []).append(nid)
        src = attrs.get("source_file") or ""
        if src:
            by_file.setdefault(src, []).append(nid)
    return by_label, by_file


def _load_sync(root: Path) -> Entry | None:
    gp = graph_path_for(root)
    try:
        st = gp.stat()
    except OSError:
        return None
    try:
        data = json.loads(gp.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        logger.warning("code_graph: unreadable %s: %s", gp, exc)
        return None
    if not isinstance(data, dict):
        return None
    g = _build_graph(data)
    by_label, by_file = _index(g)
    return Entry(
        root=str(root),
        mtime_ns=st.st_mtime_ns,
        size=st.st_size,
        graph=g,
        built_at_commit=str(data.get("built_at_commit") or ""),
        n_nodes=g.number_of_nodes(),
        n_edges=g.number_of_edges(),
        by_label=by_label,
        by_file=by_file,
        loaded_at=time.time(),
    )


async def load_graph(root: Path) -> Entry | None:
    """Cached load, re-read when graph.json's (mtime, size) moved.

    Parsing 10 MB of JSON into a graph is ~1 s, so it happens on a worker
    thread; the dashboard and every other turn share this loop.
    """
    key = str(root)
    gp = graph_path_for(root)
    try:
        st = gp.stat()
    except OSError:
        _CACHE.pop(key, None)
        return None
    hit = _CACHE.get(key)
    if hit is not None and hit.mtime_ns == st.st_mtime_ns and hit.size == st.st_size:
        _CACHE.move_to_end(key)
        return hit
    entry = await asyncio.to_thread(_load_sync, root)
    if entry is None:
        _CACHE.pop(key, None)
        return None
    _CACHE[key] = entry
    _CACHE.move_to_end(key)
    while len(_CACHE) > int(_cfg()["max_cached_roots"]):
        _CACHE.popitem(last=False)
    return entry


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


@dataclass
class Staleness:
    stale: bool
    reasons: list[str]
    head: str = ""
    is_git: bool = True
    dirty_only: bool = False
    age_s: float | None = None


async def staleness(root: Path, entry: Entry | None) -> Staleness:
    rc, head = await _git(root, "rev-parse", "HEAD")
    is_git = rc == 0 and bool(head)
    reasons: list[str] = []
    age = None
    if entry is None:
        return Staleness(True, ["no graph built for this root"], head if is_git else "",
                         is_git, False, None)
    try:
        age = time.time() - graph_path_for(root).stat().st_mtime
    except OSError:
        age = None

    if not is_git:
        return Staleness(False, [], "", False, False, age)

    commit_moved = bool(entry.built_at_commit) and entry.built_at_commit != head
    if commit_moved:
        reasons.append(
            f"HEAD moved ({entry.built_at_commit[:8]} -> {head[:8]})")
    elif not entry.built_at_commit:
        reasons.append("graph records no commit")

    dirty = await _dirty_sources_newer_than_graph(root)
    if dirty:
        shown = ", ".join(dirty[:3])
        more = f" (+{len(dirty) - 3} more)" if len(dirty) > 3 else ""
        reasons.append(f"uncommitted edits newer than the graph: {shown}{more}")

    return Staleness(
        stale=bool(reasons),
        reasons=reasons,
        head=head,
        is_git=True,
        dirty_only=bool(dirty) and not commit_moved and bool(entry.built_at_commit),
        age_s=age,
    )


async def _dirty_sources_newer_than_graph(root: Path) -> list[str]:
    """Uncommitted source files modified after graph.json was written.

    Mandatory, not a nicety: inside a autoimplement round HEAD does not move
    while the model edits, so a commit-only staleness rule would call a
    graph fresh for the entire round it is most wrong in.
    """
    rc, out = await _git(root, "status", "--porcelain")
    if rc != 0 or not out:
        return []
    try:
        graph_mtime = graph_path_for(root).stat().st_mtime
    except OSError:
        return []
    newer: list[str] = []
    for line in out.splitlines():
        rel = line[3:].strip() if len(line) > 3 else ""
        if " -> " in rel:                      # rename
            rel = rel.split(" -> ", 1)[1]
        rel = rel.strip('"')
        if not rel or not rel.endswith(SOURCE_SUFFIXES):
            continue
        try:
            if (root / rel).stat().st_mtime > graph_mtime:
                newer.append(rel)
        except OSError:
            continue
    return sorted(newer)


async def _open_worktrees() -> list[str]:
    """Autoimplement worktrees currently checked out, cached for 10 s."""
    now = time.time()
    if now - float(_WORKTREES["at"]) < 10.0:
        return list(_WORKTREES["paths"])
    rc, out = await _git(Path(os.path.realpath(LLOYD_HOME)), "worktree", "list", "--porcelain")
    paths: list[str] = []
    if rc == 0:
        work_root = str(Path.home() / "lloyd-work")
        for line in out.splitlines():
            if line.startswith("worktree "):
                p = line.split(" ", 1)[1].strip()
                if p.startswith(work_root):
                    paths.append(p)
    _WORKTREES["at"] = now
    _WORKTREES["paths"] = paths
    return list(paths)


async def binary_version() -> str:
    b = _binary()
    if b in _VERSION_CACHE:
        return _VERSION_CACHE[b]
    rc, out, _ = await _exec_or_timeout([b, "--version"], timeout=10.0)
    v = out.strip().splitlines()[0] if (rc == 0 and out.strip()) else ""
    if rc == -1:
        v = "missing"
    _VERSION_CACHE[b] = v or "unknown"
    return _VERSION_CACHE[b]


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


async def refresh(root: Path, *, force: bool = False) -> dict:
    """Rebuild the graph for `root`. Returns a result dict (may carry `error`)."""
    cfg = _cfg()
    key = str(root)
    before = await load_graph(root)
    previous = before.built_at_commit if before else ""
    timeout = float(cfg["refresh_timeout_s"])
    b = _binary()
    started = time.monotonic()

    async with _lock_for(key):
        # A caller that queued behind another build must not build again:
        # the one it waited for may already have done the work.
        if not force:
            again = await load_graph(root)
            st = await staleness(root, again)
            if again is not None and not st.stale:
                return {
                    "root": key, "built_at_commit": again.built_at_commit,
                    "previous_commit": previous, "nodes": again.n_nodes,
                    "edges": again.n_edges, "seconds": 0.0,
                    "note": "already fresh (rebuilt while this call waited)",
                }
        rc, out, err = await _exec_or_timeout(
            [b, "update", key, "--no-cluster"], cwd=key, timeout=timeout)
        if rc == -1:
            return {"error": f"graphify binary not found at {b}; "
                             f"set code_graph.graphify_bin in config.yaml"}
        if rc == -2:
            return {"error": f"graphify update timed out after {timeout}s"}
        if rc != 0:
            return {"error": f"graphify update failed (rc={rc}): "
                             f"{(err or out).strip()[:400]}"}
        rc2, out2, err2 = await _exec_or_timeout(
            [b, "cluster-only", key, "--no-label", "--no-viz"],
            cwd=key, timeout=timeout)
        if rc2 == -2:
            return {"error": f"graphify cluster-only timed out after {timeout}s"}
        if rc2 not in (0, -1):
            # Clustering is cosmetic; the extraction already landed.
            logger.warning("code_graph: cluster-only rc=%s: %s",
                           rc2, (err2 or out2).strip()[:200])
        _LAST_REFRESH[key] = time.time()
        _CACHE.pop(key, None)
        entry = await load_graph(root)

    if entry is None:
        return {"error": f"graphify wrote no readable graph under {key}"}
    return {
        "root": key,
        "built_at_commit": entry.built_at_commit,
        "previous_commit": previous,
        "nodes": entry.n_nodes,
        "edges": entry.n_edges,
        "seconds": round(time.monotonic() - started, 1),
    }


async def ensure_graph(root: Path, *, allow_refresh: bool = True
                       ) -> tuple[Entry | None, Staleness, list[str]]:
    """Load, and rebuild when stale and allowed. Returns (entry, staleness, notes)."""
    cfg = _cfg()
    notes: list[str] = []
    entry = await load_graph(root)
    st = await staleness(root, entry)

    if not st.stale or not allow_refresh or not cfg["auto_refresh"]:
        if st.stale and entry is not None:
            why = "; ".join(st.reasons)
            suffix = " (refresh=false)" if not allow_refresh else ""
            notes.append(f"STALE: {why}{suffix}")
        return entry, st, notes

    if not st.is_git and entry is not None:
        # Nothing to compare against; never auto-rebuild a non-git tree.
        return entry, st, notes

    interval = float(cfg["min_refresh_interval_s"])
    last = _LAST_REFRESH.get(str(root), 0.0)
    if st.dirty_only and entry is not None and (time.time() - last) < interval:
        notes.append(
            f"STALE: uncommitted edits (debounced, rebuilds after {int(interval)}s "
            f"or graph_refresh(force=true))")
        return entry, st, notes

    result = await refresh(root)
    if "error" in result:
        if entry is None:
            return None, st, [f"cannot refresh: {result['error']}"]
        notes.append(f"cannot refresh: {result['error']}; answering from the "
                     f"existing graph, which is STALE: " + "; ".join(st.reasons))
        return entry, st, notes
    entry = await load_graph(root)
    st = await staleness(root, entry)
    return entry, st, notes


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------


@dataclass
class Resolution:
    node: str | None = None
    candidates: list[str] = field(default_factory=list)
    nearest: list[str] = field(default_factory=list)


def _looks_like_path(q: str) -> bool:
    return "/" in q or q.endswith(SOURCE_SUFFIXES)


def resolve_symbol(entry: Entry, query: str, file_hint: str = "") -> Resolution:
    """Resolve `query` to one node id, or to a candidate list.

    Ambiguity is an answer, not an error: `main` exists in a dozen files
    and a listing lets the model pick, where an error makes it guess again.
    """
    g = entry.graph
    q = (query or "").strip()
    if not q:
        return Resolution()

    if q in g:
        return Resolution(node=q)

    if _looks_like_path(q):
        hits = [nid for src, ids in entry.by_file.items()
                if src == q or src.endswith("/" + q.lstrip("/"))
                for nid in ids if _is_file_node(g.nodes[nid])]
        if len(hits) == 1:
            return Resolution(node=hits[0])
        if hits:
            return Resolution(candidates=sorted(hits))

    owner = ""
    bare = q
    if "." in q and not _looks_like_path(q):
        owner, bare = q.rsplit(".", 1)

    ids = list(entry.by_label.get(norm_label(bare), []))
    ids = [n for n in ids if not _is_rationale(g.nodes[n])]

    if owner:
        wanted = norm_label(owner)
        kept = []
        for nid in ids:
            for pred in g.predecessors(nid):
                rels = {r["relation"] for r in g[pred][nid]["rels"]}
                if rels & CONTAINMENT_RELATIONS and \
                        norm_label(g.nodes[pred].get("label") or pred) == wanted:
                    kept.append(nid)
                    break
        if kept:
            ids = kept

    if file_hint:
        fh = file_hint.strip()
        narrowed = [n for n in ids
                    if (g.nodes[n].get("source_file") or "").endswith(fh)
                    or fh.endswith(g.nodes[n].get("source_file") or "\0")]
        if narrowed:
            ids = narrowed

    if len(ids) == 1:
        return Resolution(node=ids[0])
    if ids:
        return Resolution(candidates=sorted(ids))

    near_keys = difflib.get_close_matches(norm_label(bare),
                                          list(entry.by_label.keys()), n=6, cutoff=0.7)
    nearest: list[str] = []
    for k in near_keys:
        for nid in entry.by_label.get(k, [])[:2]:
            if not _is_rationale(g.nodes[nid]):
                nearest.append(nid)
    return Resolution(nearest=nearest[:8])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _site(attrs: dict) -> str:
    f = attrs.get("source_file") or ""
    loc = attrs.get("source_location") or ""
    if f and loc:
        return f"{f}:{loc}"
    return f or "?"


def describe(g: Any, nid: str) -> str:
    a = g.nodes[nid]
    return f"{a.get('label') or nid} — {_site(a)}"


def _edge_lines(g: Any, arrow: str, pairs: list[tuple[str, dict]]) -> list[str]:
    out = []
    for other, edata in pairs:
        for rel in edata["rels"]:
            if rel["relation"] in CONTAINMENT_RELATIONS | RATIONALE_RELATIONS:
                continue
            site = ""
            if rel["source_file"]:
                site = f" {rel['source_file']}"
                if rel["source_location"]:
                    site += f":{rel['source_location']}"
            oa = g.nodes[other]
            out.append(f"  {arrow} {oa.get('label') or other} "
                       f"[{rel['relation']}]{site}")
    return sorted(set(out))


def _cap(lines: list[str], limit: int) -> str:
    if limit > 0 and len(lines) > limit:
        cut = len(lines) - limit
        return "\n".join(lines[:limit]) + \
            f"\n... {cut} more line(s) omitted; raise `limit` to see them"
    return "\n".join(lines)


def _limit_of(args: dict, cfg: dict) -> int:
    try:
        v = int(args.get("limit") or 0)
    except (TypeError, ValueError):
        v = 0
    return v if v > 0 else int(cfg["max_lines"])


async def _header_lines(root: Path, entry: Entry | None, st: Staleness,
                        notes: list[str]) -> list[str]:
    lines: list[str] = []
    if entry is not None:
        age = ""
        if st.age_s is not None:
            age = f", built {_ago(st.age_s)}"
        commit = entry.built_at_commit[:8] or "no-commit"
        lines.append(f"root: {root} (commit {commit}, {entry.n_nodes} nodes / "
                     f"{entry.n_edges} edges{age})")
    else:
        lines.append(f"root: {root} (no graph)")
    if not st.is_git:
        lines.append("note: not a git repo: staleness unknown")
    for n in notes:
        lines.append(f"note: {n}")
    if os.path.realpath(root) == os.path.realpath(LLOYD_HOME):
        for wt in await _open_worktrees():
            lines.append(f"note: open autoimplement worktree at {wt}; "
                         f"pass root={wt} if you are editing there")
    return lines


def _ago(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 90:
        return f"{s}s ago"
    if s < 5400:
        return f"{s // 60}m ago"
    if s < 172800:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def _ambiguous(g: Any, cands: list[str], query: str, limit: int) -> list[str]:
    lines = [f"'{query}' matches {len(cands)} nodes — retry with file= or the id:"]
    for nid in cands[:limit]:
        lines.append(f"  {describe(g, nid)} (id: {nid})")
    if len(cands) > limit:
        lines.append(f"  ... {len(cands) - limit} more")
    return lines


# ---------------------------------------------------------------------------
# Tool bodies
# ---------------------------------------------------------------------------


async def _prepare(args: dict) -> tuple[Path, Entry | None, Staleness, list[str], str | None]:
    root = resolve_root(args.get("root"))
    allow = args.get("refresh")
    allow = True if allow is None else bool(allow)
    entry, st, notes = await ensure_graph(root, allow_refresh=allow)
    if entry is None:
        return root, None, st, notes, json.dumps({
            "error": f"no code graph for {root}: " + "; ".join(notes or st.reasons),
        })
    return root, entry, st, notes, None


async def _explain(args: dict) -> str:
    cfg = _cfg()
    limit = _limit_of(args, cfg)
    try:
        root, entry, st, notes, err = await _prepare(args)
    except RootError as exc:
        return json.dumps({"error": str(exc)})
    if err:
        return err
    g = entry.graph
    sym = args.get("symbol") or ""
    res = resolve_symbol(entry, sym, args.get("file") or "")
    head = await _header_lines(root, entry, st, notes)
    if res.candidates:
        return _cap(head + _ambiguous(g, res.candidates, sym, limit), limit)
    if not res.node:
        return json.dumps({
            "error": f"no node matching {sym!r} in {root}",
            "nearest": [describe(g, n) for n in res.nearest],
        })

    nid = res.node
    a = g.nodes[nid]
    lines = list(head)
    lines.append("")
    lines.append(f"{a.get('label') or nid} — {_site(a)} (id: {nid})")

    inbound = [(p, g[p][nid]) for p in g.predecessors(nid)]
    outbound = [(s, g[nid][s]) for s in g.successors(nid)]
    in_lines = _edge_lines(g, "<--", inbound)
    out_lines = _edge_lines(g, "-->", outbound)

    lines.append(f"inbound ({len(in_lines)}):")
    lines.extend(in_lines or ["  (none)"])
    lines.append(f"outbound ({len(out_lines)}):")
    lines.extend(out_lines or ["  (none)"])

    parents = [p for p in g.predecessors(nid)
               if {r["relation"] for r in g[p][nid]["rels"]} & CONTAINMENT_RELATIONS]
    members = [s for s in g.successors(nid)
               if {r["relation"] for r in g[nid][s]["rels"]} & CONTAINMENT_RELATIONS]
    if parents:
        lines.append("contained by: " + ", ".join(
            g.nodes[p].get("label") or p for p in parents[:6]))
    if members:
        lines.append(f"members ({len(members)}): " + ", ".join(
            (g.nodes[m].get("label") or m) for m in members[:12]))
    return _cap(lines, limit)


def _impact_preds(g: Any, nid: str) -> list[str]:
    out = []
    for p in g.predecessors(nid):
        if {r["relation"] for r in g[p][nid]["rels"]} & IMPACT_RELATIONS:
            out.append(p)
    return out


async def _affected(args: dict) -> str:
    cfg = _cfg()
    limit = _limit_of(args, cfg)
    try:
        root, entry, st, notes, err = await _prepare(args)
    except RootError as exc:
        return json.dumps({"error": str(exc)})
    if err:
        return err
    g = entry.graph
    sym = args.get("symbol") or ""
    try:
        depth = min(4, max(1, int(args.get("depth") or 2)))
    except (TypeError, ValueError):
        depth = 2
    res = resolve_symbol(entry, sym, args.get("file") or "")
    head = await _header_lines(root, entry, st, notes)
    if res.candidates:
        return _cap(head + _ambiguous(g, res.candidates, sym, limit), limit)
    if not res.node:
        return json.dumps({
            "error": f"no node matching {sym!r} in {root}",
            "nearest": [describe(g, n) for n in res.nearest],
        })

    # Seed with the node's own members, so graph_affected("Gate") reaches
    # the callers of Gate.run rather than only whoever names the class.
    seeds = {res.node}
    for s in g.successors(res.node):
        if {r["relation"] for r in g[res.node][s]["rels"]} & CONTAINMENT_RELATIONS:
            seeds.add(s)

    # Bound the walk independently of `limit`: `limit` shortens the *report*,
    # and tying the traversal to it would silently change the answer when a
    # caller only wanted a shorter one.
    cap = 5 * int(cfg["max_lines"])
    seen = set(seeds)
    frontier = list(seeds)
    levels: list[list[str]] = []
    truncated = False
    for _d in range(depth):
        nxt: list[str] = []
        for nid in frontier:
            for p in _impact_preds(g, nid):
                if p in seen:
                    continue
                if len(seen) >= cap:
                    truncated = True
                    break
                seen.add(p)
                nxt.append(p)
            if truncated:
                break
        if not nxt:
            break
        levels.append(nxt)
        frontier = nxt
        if truncated:
            break

    lines = list(head)
    lines.append("")
    lines.append(f"reverse impact of {describe(g, res.node)} (depth {depth}, "
                 f"{sum(len(x) for x in levels)} node(s))")
    if len(seeds) > 1:
        lines.append(f"seeded with {len(seeds) - 1} member(s) of the symbol")
    if truncated:
        # Stated before the listing, not after: the listing is what gets cut
        # by `limit`, and a truncation notice the reader never sees is worse
        # than no notice at all.
        lines.append(f"(stopped expanding at {cap} nodes — the reach is wider "
                     f"than this; narrow with a smaller depth)")
    for i, level in enumerate(levels, start=1):
        lines.append(f"depth {i} ({len(level)}):")
        for nid in sorted(level, key=lambda n: describe(g, n))[:limit]:
            lines.append(f"  {describe(g, nid)}")
        if len(level) > limit:
            lines.append(f"  ... {len(level) - limit} more at this depth")
    if not levels:
        lines.append("  (nothing depends on it in this graph)")

    files: dict[str, int] = {}
    for level in levels:
        for nid in level:
            f = g.nodes[nid].get("source_file") or "?"
            files[f] = files.get(f, 0) + 1
    if files:
        top = sorted(files.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append(f"files ({len(files)}): " + " ".join(
            f"{f} x{c}" for f, c in top[:20]))
    return _cap(lines, limit)


async def _path(args: dict) -> str:
    import networkx as nx

    cfg = _cfg()
    limit = _limit_of(args, cfg)
    try:
        root, entry, st, notes, err = await _prepare(args)
    except RootError as exc:
        return json.dumps({"error": str(exc)})
    if err:
        return err
    g = entry.graph
    head = await _header_lines(root, entry, st, notes)

    ends = []
    for key, fkey in (("a", "a_file"), ("b", "b_file")):
        q = args.get(key) or ""
        res = resolve_symbol(entry, q, args.get(fkey) or "")
        if res.candidates:
            return _cap(head + _ambiguous(g, res.candidates, q, limit), limit)
        if not res.node:
            return json.dumps({
                "error": f"no node matching {q!r} in {root}",
                "nearest": [describe(g, n) for n in res.nearest],
            })
        ends.append(res.node)

    def keep(u, v):
        rels = {r["relation"] for r in g[u][v]["rels"]}
        return bool(rels - CONTAINMENT_RELATIONS - RATIONALE_RELATIONS)

    view = nx.subgraph_view(g, filter_edge=keep)
    if args.get("undirected"):
        view = view.to_undirected(as_view=False)
    try:
        hops = nx.shortest_path(view, ends[0], ends[1])
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        hops = None

    lines = list(head)
    lines.append("")
    if not hops:
        lines.append(f"no path from {describe(g, ends[0])} to {describe(g, ends[1])}"
                     + (" (try undirected=true)" if not args.get("undirected") else ""))
        return _cap(lines, limit)
    lines.append(f"{len(hops) - 1} hop(s):")
    for i, nid in enumerate(hops):
        prefix = "  " if i == 0 else "  -> "
        rel = ""
        if i:
            prev = hops[i - 1]
            edata = g[prev][nid] if g.has_edge(prev, nid) else g[nid][prev]
            rel = " [" + ",".join(sorted({r["relation"] for r in edata["rels"]})) + "]"
        lines.append(f"{prefix}{describe(g, nid)}{rel}")
    return _cap(lines, limit)


async def _hubs(args: dict) -> str:
    cfg = _cfg()
    limit = _limit_of(args, cfg)
    try:
        root, entry, st, notes, err = await _prepare(args)
    except RootError as exc:
        return json.dumps({"error": str(exc)})
    if err:
        return err
    g = entry.graph
    try:
        top = max(1, int(args.get("top") or 10))
    except (TypeError, ValueError):
        top = 10
    prefix = (args.get("prefix") or "").strip()

    rows = []
    for nid, a in g.nodes(data=True):
        if a.get("file_type") != "code" or _is_file_node(a) or _is_rationale(a):
            continue
        if prefix and not (a.get("source_file") or "").startswith(prefix):
            continue
        ind = sum(1 for p in g.predecessors(nid)
                  if {r["relation"] for r in g[p][nid]["rels"]} & IMPACT_RELATIONS)
        outd = sum(1 for s in g.successors(nid)
                   if {r["relation"] for r in g[nid][s]["rels"]} & IMPACT_RELATIONS)
        if ind + outd:
            rows.append((ind + outd, ind, outd, nid))
    rows.sort(key=lambda r: (-r[0], r[3]))

    lines = await _header_lines(root, entry, st, notes)
    lines.append("")
    lines.append(f"top {min(top, len(rows))} of {len(rows)} connected code nodes"
                 + (f" under {prefix}" if prefix else ""))
    for _tot, ind, outd, nid in rows[:top]:
        lines.append(f"  {describe(g, nid)} — in {ind} / out {outd}")
    if not rows:
        lines.append("  (no code nodes matched)")
    return _cap(lines, max(limit, top + 4))


async def _status(args: dict) -> str:
    """Report, never build — this is what you call when a build is suspect."""
    try:
        root = resolve_root(args.get("root"))
    except RootError as exc:
        return json.dumps({"error": str(exc)})
    entry = await load_graph(root)
    st = await staleness(root, entry)
    out = {
        "root": str(root),
        "graph": str(graph_path_for(root)),
        "exists": entry is not None,
        "nodes": entry.n_nodes if entry else 0,
        "edges": entry.n_edges if entry else 0,
        "built_at_commit": entry.built_at_commit if entry else "",
        "head": st.head,
        "is_git_repo": st.is_git,
        "stale": st.stale,
        "stale_reasons": st.reasons,
        "age_seconds": round(st.age_s, 1) if st.age_s is not None else None,
        "graphify_version": await binary_version(),
        "auto_refresh": bool(_cfg()["auto_refresh"]),
        "open_worktrees": await _open_worktrees(),
    }
    return json.dumps(out, indent=2)


async def _refresh_tool(args: dict) -> str:
    try:
        root = resolve_root(args.get("root"))
    except RootError as exc:
        return json.dumps({"error": str(exc)})
    result = await refresh(root, force=bool(args.get("force")))
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------

_ROOT_DESC = ("Tree to answer about. Omit for the live checkout; pass an "
              "SM_… round id or an absolute path to ask about a autoimplement "
              "worktree. Never inferred from the session.")
_REFRESH_DESC = ("Rebuild first when the graph is stale (default true). "
                 "false answers from the existing graph and says it is stale.")


async def list_tools():
    return [
        Tool(
            name="graph_explain",
            description=(
                "Who calls this symbol and what it calls, from the code graph "
                "(AST-extracted, not grep). Use before hand-searching for "
                "callers or importers. Blind across HTTP/MCP process seams."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description":
                               "Function, class, Class.method, file path or node id"},
                    "file": {"type": "string", "description":
                             "Narrow an ambiguous symbol to this file (suffix match)"},
                    "root": {"type": "string", "description": _ROOT_DESC},
                    "refresh": {"type": "boolean", "description": _REFRESH_DESC},
                    "limit": {"type": "integer", "description": "Max output lines"},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="graph_affected",
            description=(
                "Blast radius: everything that transitively depends on a "
                "symbol, grouped by depth, with the file list. Call this "
                "before changing a shared function."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol about to change"},
                    "file": {"type": "string", "description": "Disambiguate by file"},
                    "root": {"type": "string", "description": _ROOT_DESC},
                    "refresh": {"type": "boolean", "description": _REFRESH_DESC},
                    "depth": {"type": "integer", "description":
                              "Reverse hops to walk, 1-4 (default 2)"},
                    "limit": {"type": "integer", "description": "Max lines per depth"},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="graph_path",
            description=(
                "Shortest dependency path between two symbols, hop by hop with "
                "call sites. 'no path' is an answer, not an error."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "string", "description": "Start symbol"},
                    "b": {"type": "string", "description": "End symbol"},
                    "a_file": {"type": "string", "description": "Disambiguate a"},
                    "b_file": {"type": "string", "description": "Disambiguate b"},
                    "root": {"type": "string", "description": _ROOT_DESC},
                    "refresh": {"type": "boolean", "description": _REFRESH_DESC},
                    "undirected": {"type": "boolean", "description":
                                   "Ignore edge direction (finds 'related via' links)"},
                    "limit": {"type": "integer", "description": "Max output lines"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="graph_hubs",
            description=(
                "Most-connected code symbols, ranked. Orients you in an "
                "unfamiliar area; `prefix` scopes it to a subtree."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": _ROOT_DESC},
                    "refresh": {"type": "boolean", "description": _REFRESH_DESC},
                    "top": {"type": "integer", "description": "How many (default 10)"},
                    "prefix": {"type": "string", "description":
                               "Repo-relative path prefix, e.g. 'app/harness/'"},
                    "limit": {"type": "integer", "description": "Max output lines"},
                },
                "required": [],
            },
        ),
        Tool(
            name="graph_status",
            description=(
                "Is the graph there and is it current? Node/edge counts, the "
                "commit it was built at, HEAD, why it is stale, and any open "
                "autoimplement worktree. Never builds."
            ),
            inputSchema={
                "type": "object",
                "properties": {"root": {"type": "string", "description": _ROOT_DESC}},
                "required": [],
            },
        ),
        Tool(
            name="graph_refresh",
            description=(
                "Rebuild the graph for a tree (~15s on this repo, no LLM "
                "calls). Query tools rebuild on their own when stale; call "
                "this to force one, e.g. right after opening a round."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": _ROOT_DESC},
                    "force": {"type": "boolean", "description":
                              "Rebuild even when the graph looks current"},
                },
                "required": [],
            },
        ),
    ]


_HANDLERS = {
    "graph_explain": _explain,
    "graph_affected": _affected,
    "graph_path": _path,
    "graph_hubs": _hubs,
    "graph_status": _status,
    "graph_refresh": _refresh_tool,
}


async def call_tool(name: str, arguments: dict):
    fn = _HANDLERS.get(name)
    if fn is None:
        return text_result(json.dumps({"error": f"Unknown tool: {name}"}))
    try:
        text = await fn(arguments or {})
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("code_graph: %s failed", name)
        text = json.dumps({"error": f"{name} failed: {exc}"})
    return text_result(text)


async def shutdown() -> None:
    _CACHE.clear()


def stats() -> dict:
    return {
        "cached_roots": list(_CACHE.keys()),
        "last_refresh": dict(_LAST_REFRESH),
    }
