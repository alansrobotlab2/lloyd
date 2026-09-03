#!/usr/bin/env python3
"""
Lloyd MCP Server: Vault — obsidian vault read/write/search and hybrid recall.

Tools:
    vault_read, vault_write, vault_overview, vault_search, vault_recall (5 tools)

Vault root: ~/obsidian/
QMD daemon: http://localhost:8181/query

Split out of agent_mcp/memory.py as part of Task #340 PR 5. Owns:
    - Vault file read/write with case-insensitive resolution
    - Audit log for writes (~/obsidian/memory/audit/writes.jsonl)
    - QMD daemon hybrid search (BM25 + vector) with stopword cleanup
    - vault_recall, the parallel doc + fact retrieval entry point

Imports from agent_mcp.retrieval for the fact-side of vault_recall
(entity extraction, graph expansion, fact ranking) — the shared core
also used by agent_mcp.facts.
"""

import asyncio
import concurrent.futures
import datetime
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from app.config import service_url
from mcp.server import Server
from mcp.types import Tool

from agent_mcp._shared import (
    VAULT,
    ErrorCode,
    _QUERY_STOPWORDS,
    _err,
    _wrap,
)
from agent_mcp.retrieval import (
    FACT_GODNODE_THRESHOLD,
    FACT_RANK_CAP_SEED,
    FACT_RANK_CAP_GRAPH,
    extract_entities_from_query,
    fact_matches_tokens,
    fact_query_tokens,
    fact_score,
    get_entity_edge_counts,
    get_facts_sync,
    graph_weighted_neighbors,
)
import math

# ── Constants ────────────────────────────────────────────────────────────────

AUDIT_LOG_DIR = VAULT / "memory" / "audit"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "writes.jsonl"
QMD_BIN = Path.home() / ".bun" / "bin" / "qmd"
QMD_DAEMON_URL = service_url("qmd", "http://localhost:8181/query")

VAULT_SEGMENTS = [
    "facts", "memory", "knowledge", "projects", "personal", "work", "skills",
    "architecture", "lloyd", "autonomy", "backlog", "people",
]
VAULT_EXCLUDE_DIRS = {"templates", "images"}
VAULT_EXCLUDE_FILES = {"tags.md"}

# Demotion patterns for auto-generated "memory churn" files. These all
# lexically contain whatever the user was discussing recently and crowd
# out canonical knowledge sources. Covers:
#  - `memory/YYYY-MM-DD.md` — daily session transcripts
#  - `memory/pipeline/**` — auto-generated pipeline state / skill candidate files
# Other memory/ subdirs (learnings, alan, autonomy-pipeline) NOT demoted —
# they tested as containing genuinely-useful aggregate content.
# Tunable via `demote_daily_logs` + `demote_factor` params on _vault_recall.
_DAILY_LOG_RE = re.compile(
    r"(?:^|/)memory/(\d{4}-\d{2}-\d{2}\.md|pipeline/)"
)
DAILY_LOG_DEMOTE_FACTOR = 0.4

# Canonical-source prefixes for graph_lookup boost. When a graph-derived
# entity name resolves to a file under one of these prefixes, treat it as
# a strong signal — the user almost certainly wants this file, not the
# memory log that mentions it.
_CANONICAL_PREFIXES = (
    "autonomy/", "skills/", "backlog/", "architecture/", "knowledge/", "facts/",
)

# Source-code grep fallback. QMD's index covers `~/obsidian/` only, so
# questions about Lloyd internals (`vault_recall`, `FACT_GODNODE_THRESHOLD`,
# etc.) return nothing relevant. When the query mentions identifier-like
# tokens AND QMD's hit count is thin, fall back to ripgrep over the
# code roots and merge results.
LLOYD_HOME = Path(__file__).resolve().parent.parent
LLOYD_CODE_ROOTS = [
    LLOYD_HOME / "agent_mcp",
    LLOYD_HOME / "app",
    LLOYD_HOME / "scripts",
    LLOYD_HOME / "workers",
]
LLOYD_CODE_PREFIX = str(LLOYD_HOME) + "/"
# Match Python-style identifiers >=4 chars that look code-like:
#   - have an underscore (vault_recall, _relationships)
#   - OR have 2+ uppercase chars (FACT_GODNODE, KGMentionClassifier)
#   - OR have a dot-extension (vault.py, relationships.json)
_IDENT_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b")
_DOTTED_FILE_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\.(?:py|json|md|yaml|yml|toml|sh)\b")


def _identifier_tokens(query: str) -> list[str]:
    """Pull identifier-like tokens from a query for the grep fallback.
    Returns dedup'd list preserving order; bounded to 4 to cap subprocess work.
    """
    out, seen = [], set()
    for m in _DOTTED_FILE_RE.finditer(query):
        t = m.group(0)
        if t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    for m in _IDENT_RE.finditer(query):
        t = m.group(0)
        if "_" in t or sum(1 for c in t if c.isupper()) >= 2:
            if t.lower() not in seen:
                seen.add(t.lower()); out.append(t)
    return out[:4]


def _grep_lloyd_code(query: str, limit: int = 8, timeout: float = 2.0) -> list[dict]:
    """Ripgrep the Lloyd code roots for identifier-like tokens in `query`.
    Returns QMD-shaped result dicts so they merge cleanly. Empty if no
    identifier tokens or rg fails."""
    idents = _identifier_tokens(query)
    if not idents:
        return []
    roots = [str(r) for r in LLOYD_CODE_ROOTS if r.exists()]
    if not roots:
        return []
    found: dict[str, dict] = {}
    for rank, ident in enumerate(idents):
        try:
            proc = subprocess.run(
                ["rg", "--files-with-matches", "--type-add=src:*.{py,ts,tsx,js,sh,yaml,yml,toml,json}",
                 "--type", "src", "-F", ident, *roots],
                capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        for path in (proc.stdout or "").splitlines():
            if not path:
                continue
            rel = path.removeprefix(LLOYD_CODE_PREFIX)
            if rel in found:
                continue
            # Score: 0.5 for top match per ident, decaying; behind QMD rank-1 (1.0)
            # but ahead of QMD rank-3 (0.33). Multiple-ident match files rise.
            base_score = 0.5 / (1 + 0.15 * rank)
            found[rel] = {
                "file": rel,
                "title": Path(path).name,
                "snippet": f"[code match: {ident}]",
                "score": round(base_score, 4),
            }
            if len(found) >= limit:
                break
        if len(found) >= limit:
            break
    return list(found.values())

CONSOLIDATION_MIN_RESULTS = 4
CONSOLIDATION_TIMEOUT = 10


def _consolidation_endpoint() -> tuple[str, str]:
    """Resolve (chat_completions_url, model_name) for vault consolidation.

    Uses the alias resolver so when `secondary_enabled: false` the call
    routes to primary instead of the dead :8091 endpoint.
    """
    try:
        from app.config import resolve_model_alias, _get_model_cfg
    except Exception:
        return ("", "")
    name = resolve_model_alias("secondary")
    cfg = _get_model_cfg(name) or {}
    base = cfg.get("base_url") or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")
    if not base:
        return ("", name)
    return (f"{base.rstrip('/')}/v1/chat/completions", name)

CONSOLIDATION_SYSTEM_PROMPT = (
    "You are a memory consolidation engine. Your job is to take raw search results "
    "from a knowledge vault and produce a concise, deduplicated, well-structured "
    "consolidation that directly answers the user's query.\n\n"
    "Rules:\n1. Deduplicate overlapping content.\n2. Preserve specific facts, dates, names, numbers.\n"
    "3. Return a JSON object with a 'summary' key.\n4. Be concise — under 400 words."
)

_INTENT_FACTUAL_RE = re.compile(
    r"(?:what\s+port|where\s+is|how\s+to|which\s+file|what\s+is\s+the|config\s+for|path\s+to|url\s+for|command\s+to)", re.I)
_INTENT_TEMPORAL_RE = re.compile(
    r"(?:last\s+week|yesterday|today|when\s+did|recent(?:ly)?|latest|last\s+month|last\s+time|\d{4}-\d{2}-\d{2})", re.I)
_INTENT_CONCEPTUAL_RE = re.compile(
    r"(?:approaches?\s+to|how\s+does\s+\w+\s+compare|explain\s|overview\s+of|strategy\s+for|principles?\s+of)", re.I)

app = Server("lloyd-vault")


# ── QMD helpers ──────────────────────────────────────────────────────────────

def _qmd_sanitize(query: str) -> str:
    """Strip control chars and collapse qmd query-syntax operators to spaces.

    qmd's vec/hyde parser treats `-term` as negation (Google-style) and throws
    HTTP 500 when it sees one. Replace hyphens (and other operator chars qmd
    treats as syntax) with spaces before the query hits either the vec or lex
    leg. The lex path already tokenizes on non-word boundaries so this is a
    no-op there; the vec path gets identical clean input. Fixes #325.
    """
    q = re.sub(r"[\x00-\x1f\x7f]", " ", query)
    q = re.sub(r"[-+\"]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _qmd_strip_stopwords(query: str) -> str:
    """Strip stopwords from the FTS5 lex-leg query.

    Uses the wider _QUERY_STOPWORDS (see #327) rather than _ENTITY_STOPWORDS
    so natural-language question framing gets fully removed from BM25 signal.

    Short tokens (len<2) are kept when purely digits so version numbers like
    "Qwen3.5" → tokens [qwen3, 5] don't lose the fractional part.
    """
    words = [
        w for w in re.findall(r"\b\w+\b", query.lower())
        if w not in _QUERY_STOPWORDS and (len(w) >= 2 or w.isdigit())
    ]
    return " ".join(words) if words else query


def _qmd_log(msg: str) -> None:
    print(f"[qmd] {msg}", file=sys.stderr, flush=True)


def _qmd_post(payload: dict) -> list:
    req = urllib.request.Request(
        QMD_DAEMON_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return [
        {
            "file": r.get("file", ""),
            "title": r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "score": r.get("score", 0),
        }
        for r in data.get("results", [])
    ]


def _qmd_daemon_search(query: str, limit: int, collections: list,
                      skip_rerank: bool = True,
                      legs: tuple[str, ...] = ("lex", "vec"),
                      lex_query: Optional[str] = None) -> Optional[list]:
    """Send a lex and/or vec query to the qmd daemon.

    DEFAULT: skip_rerank=True. The reranker rarely changes top-1 and
    shuffles within top-5; not worth the tax for LLM-context consumers.
    Pass skip_rerank=False explicitly only when ordering within the top-5
    matters for a human scanning results.

    `legs` selects which search legs run. Measured 2026-09-03 on this host
    (6 collections, skipRerank): lex-only 10-80ms; anything including the
    vec leg 1.1-2.6s — the query embedding dominates and the daemon's
    embedding cache only brings a *repeated* query down to ~1.1s. The
    prefetch path runs `("lex",)` inside its latency budget and the full
    hybrid as a straggler whose result carries over to the next turn.

    `lex_query`, when given, replaces the lex leg's text. The lex leg is
    FTS5 with implicit AND (every term must match), so it wants a short,
    high-signal term list, while the vec leg benefits from the full
    focus-enriched sentence. Without this both legs got the same string and
    the hybrid's lex component returned nothing on enriched queries.
    """
    query = _qmd_sanitize(query)
    if not query:
        return []
    # Apply the same stopword strip to BOTH legs. Conversational framing
    # ("tell me about X", "show me Y") drifts the vec embedding away from
    # content. Identical inputs are fine — lex (BM25) and vec (embedding)
    # do fundamentally different matching, so they still produce
    # complementary signal.
    stripped = _qmd_strip_stopwords(query)
    lex_q = stripped
    if lex_query:
        lex_q = _qmd_strip_stopwords(_qmd_sanitize(lex_query)) or stripped
    payload = {
        "searches": [{"type": leg, "query": lex_q if leg == "lex" else stripped}
                     for leg in legs],
        "limit": limit,
        "collections": collections,
    }
    if skip_rerank:
        payload["skipRerank"] = True

    try:
        return _qmd_post(payload)
    except urllib.error.HTTPError as e:
        # Rerank context OOM manifests as HTTP 500. One-shot retry without
        # rerank so callers see documents instead of silent zero-hits.
        if e.code == 500 and not payload.get("skipRerank"):
            try:
                payload["skipRerank"] = True
                _qmd_log("rerank failed (HTTP 500) — retrying with skipRerank")
                return _qmd_post(payload)
            except Exception as e2:
                _qmd_log(f"skipRerank retry also failed: {e2!r}")
                return None
        _qmd_log(f"HTTPError {e.code}: {e.reason}")
        return None
    except Exception as e:
        _qmd_log(f"search failed: {e!r}")
        return None


def _qmd_subprocess_search(query: str, limit: int, collections: list) -> list:
    if not QMD_BIN.exists():
        return []
    try:
        coll_args = []
        for c in collections:
            coll_args.extend(["-c", c])
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"}
        proc = subprocess.run([str(QMD_BIN), "query", query, *coll_args, "-n", str(limit), "--json"],
                              capture_output=True, text=True, timeout=30, env=env)
        if proc.returncode != 0:
            return []
        return json.loads(proc.stdout)
    except Exception:
        return []


def _consolidate_results(query: str, results: list) -> Optional[dict]:
    if len(results) < CONSOLIDATION_MIN_RESULTS:
        return None
    url, model_name = _consolidation_endpoint()
    if not url:
        return None
    parts = [f"Query: {query}\n\nSearch Results:\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"--- Result {i} (score: {r.get('score', 'N/A')}) ---")
        parts.append(f"File: {r.get('citation', r.get('path', ''))}")
        snippet = r.get("snippet", "")[:2000]
        parts.append(f"Content:\n{snippet}\n")
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(parts)}],
        "temperature": 0.0, "max_tokens": 1000,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=CONSOLIDATION_TIMEOUT) as resp:
            data = json.loads(resp.read())
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            return None
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        return json.loads(content)
    except Exception:
        return None


def _rrf_fuse(ranked_lists: list, k: int = 60) -> list:
    scores, items = {}, {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            path = item.get("path") or item.get("file", "")
            rrf_score = 1.0 / (k + rank + 1)
            scores[path] = scores.get(path, 0.0) + rrf_score
            existing = items.get(path)
            if not existing or float(item.get("score", 0)) > float(existing.get("score", 0)):
                items[path] = item
    fused = []
    for path, rrf_score in sorted(scores.items(), key=lambda x: -x[1]):
        result = dict(items[path])
        result["rrf_score"] = round(rrf_score, 6)
        fused.append(result)
    return fused


def _graph_rerank(
    documents: list[dict],
    seed_entities: list[str],
    weighted_neighbors: list,
    alpha: float = 0.5,
) -> list[dict]:
    """Re-rank documents using fact-graph topological voting (TGS-RAG Phase 3).

    For each doc, extract entities mentioned in its title+snippet. Each
    seed/neighbor entity contributes a vote weighted by its graph weight
    (seeds=1.0, neighbors=their graph_weighted_neighbors weight) and
    divided by log(1+degree) so god-nodes (e.g. "lloyd" with ~991 edges)
    don't dominate. Final score = alpha * QMD_score + (1-alpha) * normalized_topo.

    Returns a new list of dicts sorted by combined score, with `_qmd_score`
    and `_topo_score` annotations preserved for inspection/eval.
    """
    if not documents or (not seed_entities and not weighted_neighbors):
        return documents

    voters: dict[str, float] = {}
    for s in seed_entities or []:
        if s:
            voters[s.lower()] = 1.0
    for ent, w in (weighted_neighbors or []):
        key = (ent or "").lower()
        if key and key not in voters:
            voters[key] = float(w)

    if not voters:
        return documents

    # Precompute per-voter score contribution (degree penalty baked in)
    # and a word-boundary regex. This collapses what used to be a full
    # `extract_entities_from_query(doc_text)` call per doc — which
    # iterates ~2,700 entity dirs — into a tight regex scan over ~15
    # voter terms. Drops rerank latency by ~10x.
    edge_counts = get_entity_edge_counts()
    voter_contribs: dict[str, float] = {}
    voter_patterns: dict[str, re.Pattern] = {}
    for vk, vw in voters.items():
        if len(vk) < 2:
            continue
        degree = max(edge_counts.get(vk, 1), 1)
        voter_contribs[vk] = vw / math.log(1 + degree + math.e)
        voter_patterns[vk] = re.compile(r"(?<!\w)" + re.escape(vk) + r"(?!\w)", re.IGNORECASE)

    def _topo(text: str) -> float:
        if not text:
            return 0.0
        score = 0.0
        for vk, pat in voter_patterns.items():
            if pat.search(text):
                score += voter_contribs[vk]
        return score

    raw = []
    for d in documents:
        text = " ".join([
            str(d.get("path") or ""),
            str(d.get("title") or ""),
            str(d.get("snippet") or ""),
        ])
        raw.append(_topo(text))

    max_topo = max(raw) if raw else 0.0
    rescored = []
    for d, t in zip(documents, raw):
        norm = (t / max_topo) if max_topo > 0 else 0.0
        qmd = float(d.get("score", 0) or 0)
        combined = alpha * qmd + (1 - alpha) * norm
        rescored.append({
            **d,
            "score": round(combined, 6),
            "_qmd_score": round(qmd, 6),
            "_topo_score": round(norm, 6),
        })
    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored


def _run_vault_search(query: str, max_results: int, min_score: float, scope: str, consolidate: bool) -> dict:
    scope_prefixes = []
    if scope:
        for item in scope.split(","):
            item = item.strip()
            if item:
                scope_prefixes.append(item.rstrip("/") + "/")

    coll_list = VAULT_SEGMENTS
    if scope:
        scope_segs = [s.strip().rstrip("/") for s in scope.split(",") if s.strip()]
        coll_list = [s for s in scope_segs if s in VAULT_SEGMENTS] or VAULT_SEGMENTS

    # Run QMD search across the requested collections AND the source-code
    # grep fallback in parallel (lever 3) when not scope-restricted.
    def _do_qmd():
        if len(coll_list) == 1:
            return _qmd_daemon_search(query, max_results, coll_list) or []
        def _search_one(coll):
            return _qmd_daemon_search(query, max_results, [coll]) or []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(coll_list), 4)) as pool:
            batches = list(pool.map(_search_one, coll_list))
        merged = {}
        for batch in batches:
            for r in batch:
                fpath = r.get("file", "")
                if fpath not in merged or float(r.get("score", 0)) > float(merged[fpath].get("score", 0)):
                    merged[fpath] = r
        return sorted(merged.values(), key=lambda x: float(x.get("score", 0)), reverse=True)

    def _do_grep():
        # Skip grep when caller restricted scope — they want vault-only results.
        if scope_prefixes:
            return []
        return _grep_lloyd_code(query, limit=max(max_results, 8))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        qmd_fut = pool.submit(_do_qmd)
        grep_fut = pool.submit(_do_grep)
        all_raw = qmd_fut.result()
        grep_results = grep_fut.result()

    if grep_results:
        existing = {r.get("file", "") for r in all_raw}
        for gr in grep_results:
            if gr.get("file") not in existing:
                all_raw.append(gr)

    # Lever 1: demote memory daily logs so they don't dominate canonical sources.
    for r in all_raw:
        path_for_demote = r.get("file", "").removeprefix("qmd://").removeprefix("obsidian/")
        if _DAILY_LOG_RE.search(path_for_demote):
            r["_pre_demote_score"] = r.get("score", 0)
            r["score"] = round(float(r.get("score", 0)) * DAILY_LOG_DEMOTE_FACTOR, 6)
    all_raw.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

    parsed = []
    for r in all_raw:
        file_val = r.get("file", "")
        path = file_val.removeprefix("qmd://")
        if path.startswith("obsidian/"):
            path = path.removeprefix("obsidian/")
        score = float(r.get("score", 0))
        if score < min_score:
            continue
        if scope_prefixes and not any(path.startswith(p) for p in scope_prefixes):
            continue
        snippet = re.sub(r"@@[^@]*@@\s*(?:\([^)]*\)\s*)?", "", r.get("snippet", "")).strip()
        snippet = re.sub(r"^\d+:\s*", "", snippet, flags=re.MULTILINE).strip()
        parsed.append({"path": path, "score": round(score, 4), "snippet": snippet[:300], "citation": path})

    trimmed = parsed[:max_results]

    consolidated = None
    if consolidate and len(trimmed) >= CONSOLIDATION_MIN_RESULTS:
        consolidated = _consolidate_results(query, trimmed)

    return {
        "results": trimmed, "mode": "hybrid",
        "consolidated": consolidated is not None,
        "consolidated_summary": consolidated,
        "collections_searched": coll_list,
    }


# ── Vault helpers ────────────────────────────────────────────────────────────

def _resolve_case_insensitive(rel_path: str) -> Optional[Path]:
    current = VAULT
    for segment in Path(rel_path).parts:
        exact = current / segment
        if exact.exists():
            current = exact
            continue
        try:
            matches = [e for e in current.iterdir() if e.name.lower() == segment.lower()]
        except OSError:
            return None
        if not matches:
            return None
        current = matches[0]
    return current if current.is_file() else None


def _normalize_vault_path(path: str) -> tuple[str | None, str | None]:
    """Reduce a caller-supplied path to a vault-relative POSIX path.

    Callers routinely pass the documented ``~/obsidian/...`` or absolute
    ``/home/.../obsidian/...`` forms. ``Path.__truediv__`` does NOT expand
    ``~`` (only ``expanduser()`` does), so ``VAULT / "~/obsidian/x"`` silently
    creates a literal ``~`` directory *inside* the vault — the escape guard
    passes because the path is still under VAULT. That produced the recurring
    stray ``~/obsidian/`` tree (2026-07 incident). Strip the known vault
    prefixes and reject anything that points outside the vault.

    Returns ``(relative_path, None)`` on success or ``(None, error_message)``.
    """
    raw = path.strip()
    if not raw:
        return None, "path is required"
    vault_str = str(VAULT)
    p = raw
    if p.startswith("~/obsidian/"):
        p = p[len("~/obsidian/"):]
    elif p in ("~/obsidian", "~"):
        p = ""
    elif p == vault_str or p.startswith(vault_str + "/"):
        p = p[len(vault_str):]
    elif p.startswith("/") or p.startswith("~"):
        # Absolute path outside the vault, or a ~/<other-root> home path
        # (e.g. ~/lloyd/...). These are never valid vault targets.
        return None, f"path must be vault-relative or under {vault_str}, got {raw!r}"
    p = p.lstrip("/")
    if not p:
        return None, "path resolves to the vault root, not a file"
    parts = Path(p).parts
    if "~" in parts or ".." in parts:
        return None, f"invalid path segment in {raw!r}"
    return p, None


def _audit_write(path: str, byte_count: int) -> None:
    try:
        AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "agent_id": "lloyd", "path": path, "bytes": byte_count, "action": "write"}
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ── Tool handlers ────────────────────────────────────────────────────────────

def _vault_read(params: dict) -> dict:
    path, norm_err = _normalize_vault_path(params.get("path", ""))
    if norm_err:
        return _err(norm_err, ErrorCode.MISSING_PARAM if "required" in norm_err else ErrorCode.PATH_ESCAPE)
    try:
        target = VAULT / path
        if not target.resolve().is_relative_to(VAULT.resolve()):
            return _err("path escapes vault root", ErrorCode.PATH_ESCAPE)
        if not target.exists():
            resolved = _resolve_case_insensitive(path)
            if resolved is None:
                return _err(f"File not found: {path}", ErrorCode.NOT_FOUND)
            target = resolved
        text = target.read_text(encoding="utf-8", errors="replace")
        start_line = int(params.get("start_line", 0))
        num_lines = int(params.get("num_lines", 0))
        if start_line > 0 or num_lines > 0:
            lines = text.splitlines()
            start = max(0, start_line - 1)
            end = (start + num_lines) if num_lines > 0 else len(lines)
            text = "\n".join(lines[start:end])
        return {"path": path, "text": text or "(empty file)"}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _vault_write(params: dict) -> dict:
    path, norm_err = _normalize_vault_path(params.get("path", ""))
    if norm_err:
        return _err(norm_err, ErrorCode.MISSING_PARAM if "required" in norm_err else ErrorCode.PATH_ESCAPE)
    content = params.get("content", "")
    try:
        target = VAULT / path
        if not target.resolve().is_relative_to(VAULT.resolve()):
            return _err("path escapes vault root", ErrorCode.PATH_ESCAPE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        byte_count = len(content.encode("utf-8"))
        _audit_write(path, byte_count)
        return {"success": True, "path": path, "bytes": byte_count}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _vault_overview(params: dict) -> dict:
    try:
        if not VAULT.exists():
            return _err(f"Vault not found: {VAULT}", ErrorCode.NOT_FOUND)
        totals: dict = {}
        grand_total = 0
        for segment in VAULT_SEGMENTS:
            seg_dir = VAULT / segment
            if not seg_dir.is_dir():
                totals[segment] = 0
                continue
            count = sum(1 for f in seg_dir.rglob("*.md") if f.name not in VAULT_EXCLUDE_FILES and not any(p in VAULT_EXCLUDE_DIRS for p in f.parts))
            totals[segment] = count
            grand_total += count
        return {"total_files": grand_total, "segments": totals}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _vault_search(params: dict) -> dict:
    query = params.get("query", "").strip()
    if not query:
        return _err("query is required", ErrorCode.MISSING_PARAM, results=[])
    try:
        return _run_vault_search(query, int(params.get("max_results", 10)), float(params.get("min_score", 0.0)), params.get("scope", ""), params.get("consolidate", True))
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, results=[])


def _vault_recall(params: dict) -> dict:
    query = params.get("query", "").strip()
    if not query:
        return _err("query is required", ErrorCode.MISSING_PARAM, documents=[], facts=[])
    limit = int(params.get("limit", 20))
    include_facts = params.get("include_facts", True)
    expand_graph = params.get("expand_graph", False)
    # graph_rerank default-on as of 2026-05-12 — the perf optimization
    # (regex over voters instead of full entity scan) drops latency from
    # ~2s extra to near-zero, while the MRR lift (+6-13%) is consistent.
    # Alpha defaults to 0.3 (graph-heavy) per the May 11 alpha sweep.
    graph_rerank = bool(params.get("graph_rerank", True))
    rerank_alpha = float(params.get("rerank_alpha", 0.3))
    demote_daily_logs = bool(params.get("demote_daily_logs", True))
    demote_factor = float(params.get("demote_factor", DAILY_LOG_DEMOTE_FACTOR))
    # Graph expansion breadth/depth. Both were hardcoded (top_k=5, hops=1)
    # until 2026-08-06 (#380): with only 5 neighbour slots, low-weight edge
    # types can never surface — a `mentions` edge scores 0.3 × 0.8 = 0.24
    # against typed edges at 0.4–0.6, so density changes were invisible to
    # retrieval and to the eval. hops=1 also meant the "multi-hop" eval
    # category was in fact being served by single-hop expansion.
    # Defaults preserve the historic behaviour exactly.
    graph_top_k = int(params.get("graph_top_k", 5))
    graph_hops = int(params.get("graph_hops", 1))

    # Take top-10 seeds (was 5). Ties at low scores can knock out the
    # canonical entity; e.g. "Knowledge Graph Consistency" and "Knowledge
    # Graph System" both score 0.27 for the same query, and with k=5 the
    # canonical one can be cut off.
    seed_entities = [e for e, _ in extract_entities_from_query(query)[:10]]

    # If graph_rerank is requested, we need neighbors regardless of expand_graph,
    # because rerank uses them as voters. Force graph expansion in that case.
    need_neighbors = expand_graph or graph_rerank
    weighted_neighbors: list[tuple[str, float]] = []
    if need_neighbors and seed_entities:
        weighted_neighbors = graph_weighted_neighbors(
            seed_entities, top_k=graph_top_k, hops=graph_hops
        )

    def _do_search():
        # Daemon is the only search path. Subprocess fallback was removed
        # 2026-04-20: on daemon failure the CLI subprocess hits the same
        # broken state, then eats 30s before returning empty.
        result = _qmd_daemon_search(query, limit, VAULT_SEGMENTS)
        return result or []

    def _do_code_grep():
        if not params.get("grep_code", True):
            return []
        return _grep_lloyd_code(query, limit=8)

    def _do_graph_lookup():
        """Phase 2 (reframed): for each top seed + graph neighbor, look
        up its canonical file via a focused QMD search of the entity name.
        Surfaces autonomy/N-*.md, skills/<slug>/SKILL.md, backlog files
        that lexically match the entity but don't share the original
        query's tokens.

        Eval verdict (2026-05-12): on average HURTS single-entity queries
        by injecting alternatives that displace the perfect QMD top-1
        match. HELPS the 'hard' cross-domain category. Net regression
        when default-on (-12% MRR overall). Default-off, opt-in via
        `graph_lookup: True`."""
        if not params.get("graph_lookup", False):
            return []
        # Use top 4 seeds + top 4 neighbors. Cap entity name length to skip
        # noisy compound names like "autonomy tasks API" that just retrieve
        # the original query's results again.
        entity_pool = []
        for e in (seed_entities or [])[:4]:
            if e and 3 <= len(e) <= 60 and e not in entity_pool:
                entity_pool.append(e)
        if need_neighbors:
            for e, _w in (weighted_neighbors or [])[:4]:
                if e and 3 <= len(e) <= 60 and e not in entity_pool:
                    entity_pool.append(e)
        if not entity_pool:
            return []

        def _lookup_one(ent):
            try:
                hits = _qmd_daemon_search(ent, 2, VAULT_SEGMENTS) or []
            except Exception:
                hits = []
            return [(ent, h) for h in hits[:2]]

        # Parallel per-entity lookup. Cap workers to avoid swamping QMD.
        canonical: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(entity_pool), 6)
        ) as pool:
            for pairs in pool.map(_lookup_one, entity_pool):
                for ent, h in pairs:
                    f = h.get("file", "")
                    if not f or f in canonical:
                        continue
                    rel_path = f.removeprefix("qmd://").removeprefix("obsidian/")
                    is_canonical = any(rel_path.startswith(p) for p in _CANONICAL_PREFIXES)
                    qmd_score = float(h.get("score", 0) or 0)
                    if is_canonical:
                        # Promote canonical files into top-N but don't
                        # overtake QMD rank-1 (1.0). Cap at 0.85 so a
                        # legitimate query→canonical lexical match (1.0)
                        # still wins, but graph-derived canonicals beat
                        # demoted memory logs (0.4) and QMD rank-2 (0.5).
                        boosted = min(0.85, qmd_score + 0.35)
                    else:
                        # Weak promote: don't outrank ANY strong QMD hit.
                        boosted = min(0.5, qmd_score * 0.6)
                    canonical[f] = {
                        **h,
                        "score": round(boosted, 4),
                        "_via_graph_entity": ent,
                        "_qmd_score_at_lookup": qmd_score,
                        "_canonical_boost": is_canonical,
                    }
        return list(canonical.values())

    def _do_facts():
        if not include_facts:
            return [], []
        # Query-aware fact ranking (Fix A, #322).
        qtoks = fact_query_tokens(query)

        def _collect(entity_names: list[str], godnode_threshold: int) -> list[dict]:
            """Pull facts from entities, applying god-node guardrail (Fix C).
            Tags each fact with its source `entity` so downstream consumers
            (re-ranking, eval, UI) can attribute facts back to a node.
            """
            out: list[dict] = []
            for ent in entity_names:
                try:
                    entity_data = get_facts_sync(ent)
                except Exception:
                    continue
                ef = entity_data.get("facts") or []
                if not ef:
                    continue
                if len(ef) > godnode_threshold and qtoks:
                    kept = [f for f in ef if fact_matches_tokens(f, qtoks)]
                    if not kept:
                        continue
                    ef = kept
                resolved_entity = entity_data.get("entity") or ent
                out.extend({**f, "entity": resolved_entity} for f in ef)
            return out

        def _rank(candidates: list[dict], cap: int) -> list[dict]:
            if not candidates:
                return []
            scored = [
                (fact_score(f, qtoks), float(f.get("confidence", 0.5)), idx, f)
                for idx, f in enumerate(candidates)
            ]
            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
            return [f for _s, _c, _i, f in scored[:cap]]

        seed_pool = _collect(seed_entities, FACT_GODNODE_THRESHOLD)
        facts = _rank(seed_pool, FACT_RANK_CAP_SEED)

        graph_facts: list[dict] = []
        if expand_graph and weighted_neighbors:
            neighbor_names = [e for e, _w in weighted_neighbors[:graph_top_k]]
            graph_pool = _collect(neighbor_names, FACT_GODNODE_THRESHOLD)
            graph_facts = _rank(graph_pool, FACT_RANK_CAP_GRAPH)

        return facts, graph_facts

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            search_fut = pool.submit(_do_search)
            facts_fut = pool.submit(_do_facts)
            grep_fut = pool.submit(_do_code_grep)
            graph_lookup_fut = pool.submit(_do_graph_lookup)
            raw_results = search_fut.result()
            facts, graph_facts = facts_fut.result()
            grep_results = grep_fut.result()
            graph_lookup_results = graph_lookup_fut.result()
        # Merge alternate-retrieval sources: dedupe by file path; first
        # wins (QMD primary, then grep, then graph-lookup).
        existing_files = {r.get("file", "") for r in raw_results}
        for gr in grep_results:
            if gr.get("file") not in existing_files:
                raw_results.append(gr)
                existing_files.add(gr.get("file", ""))
        for gl in graph_lookup_results:
            if gl.get("file") not in existing_files:
                raw_results.append(gl)
                existing_files.add(gl.get("file", ""))
        documents = []
        # When graph_rerank is on (or demote_daily_logs is on, since post-
        # processing may swap docs) we over-fetch from QMD then re-sort.
        pool_size = max(limit * 3, limit) if (graph_rerank or demote_daily_logs) else limit
        prerank_pool = raw_results[:pool_size]
        for r in prerank_pool:
            path = r.get("file", "").removeprefix("qmd://")
            if path.startswith("obsidian/"):
                path = path.removeprefix("obsidian/")
            documents.append({
                "path": path,
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "score": r.get("score", 0),
            })
        if demote_daily_logs:
            for d in documents:
                if _DAILY_LOG_RE.search(d.get("path") or ""):
                    d["_pre_demote_score"] = d.get("score", 0)
                    d["score"] = round(float(d.get("score", 0)) * demote_factor, 6)
            documents.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
        if graph_rerank:
            documents = _graph_rerank(documents, seed_entities, weighted_neighbors, alpha=rerank_alpha)[:limit]
        else:
            documents = documents[:limit]
        result = {"documents": documents, "facts": facts, "query": query}
        if graph_facts:
            result["graph_expanded_facts"] = graph_facts
        if weighted_neighbors:
            result["graph_neighbors_used"] = [
                {"entity": e, "weight": round(w, 3)} for e, w in weighted_neighbors
            ]
        # NOTE: datetime event_date values from YAML-parsed fact frontmatter
        # are non-JSON-native. _wrap() in the dispatcher uses default=str to
        # serialize them. (Pre-existing data issue in the facts tree,
        # surfaced by #322 returning more facts.)
        return result
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, documents=[], facts=[])


# ── MCP registration ─────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools():
    return [
        Tool(name="vault_read", description="Read a file from the obsidian vault. Path is vault-relative (e.g. 'memory/learnings/DAILY_NOTES.md'); a leading '~/obsidian/' is stripped automatically.", inputSchema={
            "type": "object", "properties": {"path": {"type": "string", "description": "Vault-relative path, e.g. 'knowledge/agents/foo.md'. Do not prefix with '~/' or an absolute home path."}, "start_line": {"type": "integer"}, "num_lines": {"type": "integer"}}, "required": ["path"]}),
        Tool(name="vault_write", description="Write content to a vault file. Audit-logged. Path is vault-relative (e.g. 'memory/learnings/DAILY_NOTES.md'); a leading '~/obsidian/' is stripped automatically.", inputSchema={
            "type": "object", "properties": {"path": {"type": "string", "description": "Vault-relative path, e.g. 'knowledge/agents/foo.md'. Do not prefix with '~/' or an absolute home path."}, "content": {"type": "string"}}, "required": ["path", "content"]}),
        Tool(name="vault_overview", description="Get vault statistics: file counts per segment.", inputSchema={
            "type": "object", "properties": {"detail": {"type": "string", "enum": ["summary", "hubs"]}}}),
        Tool(name="vault_search", description="Hybrid BM25+vector search across the obsidian vault.", inputSchema={
            "type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}, "min_score": {"type": "number"}, "scope": {"type": "string"}, "consolidate": {"type": "boolean"}}, "required": ["query"]}),
        Tool(name="vault_recall", description="Combined recall: vault search + entity fact retrieval in parallel. Use expand_graph=true to include facts from related entities.", inputSchema={
            "type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "include_facts": {"type": "boolean"}, "expand_graph": {"type": "boolean", "description": "Expand results via relationship graph (1 hop)"}}, "required": ["query"]}),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    handlers = {
        "vault_read": _vault_read, "vault_write": _vault_write, "vault_overview": _vault_overview,
        "vault_search": _vault_search, "vault_recall": _vault_recall,
    }
    handler = handlers.get(name)
    if handler:
        # Handlers are sync and do subprocess/urllib I/O (QMD search, rg,
        # consolidation LLM call) with multi-second timeouts — run them in a
        # worker thread so the shared event loop never stalls.
        return _wrap(await asyncio.to_thread(handler, arguments))
    return _wrap(_err(f"Unknown tool: {name}", ErrorCode.UNKNOWN_TOOL))
