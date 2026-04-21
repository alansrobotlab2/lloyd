#!/usr/bin/env python3
"""
Lloyd MCP Server: Memory — knowledge graph, facts, and vault tools.

Fact tools: fact_get, fact_add, fact_profile, fact_check, fact_resolve
Vault tools: vault_get, vault_write, vault_overview, vault_search, vault_recall

Facts data: ~/obsidian/facts/
Vault root: ~/obsidian/
QMD daemon: http://localhost:8181/query
"""

import asyncio
import concurrent.futures
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

import yaml
from mcp.server import Server
from mcp.types import Tool, TextContent

# ── Constants ─────────────────────────────────────────────────────────────────

MEMORIES_ROOT = Path.home() / "obsidian" / "lloyd"
MEMORY_FILES = {"MEMORY.md", "USER.md"}

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"you\s+are\s+now\s+a", re.I),
    re.compile(r"disregard\s+(your\s+)?(previous\s+)?instructions", re.I),
    re.compile(r"new\s+system\s+prompt", re.I),
    re.compile(r"pretend\s+you\s+are", re.I),
    re.compile(r"\x00|\u200b|\u200c|\u200d|\u2060|\ufeff", re.I),
]

VAULT = Path.home() / "obsidian"
FACTS_ROOT = Path.home() / "obsidian" / "facts"
AUDIT_LOG_DIR = VAULT / "memory" / "audit"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "writes.jsonl"
QMD_BIN = Path.home() / ".bun" / "bin" / "qmd"
QMD_DAEMON_URL = "http://localhost:8181/query"

VAULT_SEGMENTS = [
    "facts", "memory", "knowledge", "projects", "personal", "work", "skills",
    "architecture", "lloyd", "autonomy", "backlog", "people",
]
VAULT_EXCLUDE_DIRS = {"templates", "images"}
VAULT_EXCLUDE_FILES = {"tags.md"}

# Edge-type weights for weighted graph expansion in vault_recall.
# Typed semantic edges dominate; cooccurrence-style edges are down-weighted so
# they still contribute but don't drown out real relationships.
# Keep this in sync with the vocabulary emitted by the relation classifier
# (scripts/memory/classify-relationships.py, Phase 1B of backlog #294).
EDGE_TYPE_WEIGHTS = {
    # semantic, high-confidence
    "uses": 1.0,
    "depends_on": 1.0,
    "implements": 1.0,
    "supersedes": 0.9,
    "part_of": 0.85,
    "created_by": 0.8,
    "discusses": 0.75,
    "competes_with": 0.7,
    "related_to": 0.6,
    # cooccurrence / weak signals
    "wiki_link_co_occurrence": 0.4,
    "co_mentioned": 0.35,
    "mentions": 0.3,
}
_DEFAULT_EDGE_WEIGHT = 0.3  # unknown types fall back to same weight as mentions

CONSOLIDATION_ENDPOINT = "http://localhost:8091/v1/chat/completions"
CONSOLIDATION_MODEL = "Qwen3.5-35B-A3B"
CONSOLIDATION_MIN_RESULTS = 4
CONSOLIDATION_TIMEOUT = 10

ALIASES_PATH = FACTS_ROOT / "entity-aliases.json"

_ENTITY_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "what", "which", "who", "how", "when", "where", "why",
}

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

_OPPOSING_PAIRS = [
    ("yes", "no"), ("true", "false"), ("enabled", "disabled"),
    ("active", "inactive"), ("supported", "unsupported"),
    ("working", "broken"), ("success", "failure"),
]

app = Server("lloyd-memory")

# ── Caches ────────────────────────────────────────────────────────────────────

_entity_dirs_cache: Optional[tuple] = None
_ENTITY_DIRS_TTL = 60


def _get_entity_dirs_cached() -> list:
    global _entity_dirs_cache
    now = time.monotonic()
    if _entity_dirs_cache is not None and (now - _entity_dirs_cache[0]) < _ENTITY_DIRS_TTL:
        return _entity_dirs_cache[1]
    if not FACTS_ROOT.exists():
        _entity_dirs_cache = (now, [])
        return []
    names = [d.name for d in FACTS_ROOT.iterdir() if d.is_dir()]
    _entity_dirs_cache = (now, names)
    return names


# ── Fact helpers ──────────────────────────────────────────────────────────────

def _parse_fact_frontmatter(content: str) -> dict:
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    return yaml.safe_load(content[3:end]) or {}


def _write_fact_frontmatter(data: dict) -> str:
    return f"---\n{yaml.dump(data, default_flow_style=False, sort_keys=False)}---\n"


def _find_entity_dir(entity: str) -> Optional[Path]:
    if not FACTS_ROOT.exists():
        return None
    entity_lower = entity.lower()
    for entry in FACTS_ROOT.iterdir():
        if entry.is_dir() and entry.name.lower() == entity_lower:
            return entry
    return None


def _load_aliases() -> dict:
    if not ALIASES_PATH.exists():
        return {}
    try:
        return json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_aliases(aliases: dict) -> None:
    ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIASES_PATH.write_text(json.dumps(aliases, indent=2, sort_keys=True), encoding="utf-8")


def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _fuzzy_entity_match(name: str, candidates: list[str], threshold: float = 0.7) -> Optional[str]:
    name_lower = name.lower().strip()
    best_match = None
    best_score = 0.0
    for candidate in candidates:
        cand_lower = candidate.lower().strip()
        if name_lower == cand_lower:
            return candidate
        overlap = _token_overlap(name_lower, cand_lower)
        max_len = max(len(name_lower), len(cand_lower))
        lev_score = 1.0 - (_levenshtein(name_lower, cand_lower) / max_len) if max_len > 0 else 0.0
        combined = 0.4 * overlap + 0.6 * lev_score
        if name_lower in cand_lower or cand_lower in name_lower:
            combined = max(combined, 0.8)
        if combined > best_score:
            best_score = combined
            best_match = candidate
    return best_match if best_score >= threshold else None


def _resolve_entity(name: str, auto_create: bool = False) -> tuple[str, bool]:
    name = name.strip()
    if not name:
        return name, True
    entity_dir = _find_entity_dir(name)
    if entity_dir:
        return entity_dir.name, False
    aliases = _load_aliases()
    name_lower = name.lower()
    if name_lower in aliases:
        canonical = aliases[name_lower]
        if _find_entity_dir(canonical):
            return canonical, False
    known_entities = _get_entity_dirs_cached()
    fuzzy_match = _fuzzy_entity_match(name, known_entities)
    if fuzzy_match:
        aliases[name_lower] = fuzzy_match
        _save_aliases(aliases)
        return fuzzy_match, False
    if auto_create:
        aliases[name_lower] = name
        _save_aliases(aliases)
    return name, True


def _generate_fact_id(category: str) -> str:
    return f"{category[:4]}-{uuid.uuid4().hex[:4]}"


def _get_facts_sync(entity: str, category: str = None, as_of: str = None,
                     include_expired: bool = False) -> dict:
    resolved, _ = _resolve_entity(entity)
    entity_dir = _find_entity_dir(resolved)
    if not entity_dir:
        return {"error": f"Entity not found: {entity}", "facts": []}
    facts = []
    if category:
        fact_file = entity_dir / f"{resolved}-{category}.md"
        if not fact_file.exists():
            fact_file = entity_dir / f"{entity}-{category}.md"
        if fact_file.exists():
            frontmatter = _parse_fact_frontmatter(fact_file.read_text(encoding="utf-8"))
            facts = frontmatter.get("facts", [])
    else:
        for fact_file in entity_dir.glob("*.md"):
            frontmatter = _parse_fact_frontmatter(fact_file.read_text(encoding="utf-8"))
            facts.extend(frontmatter.get("facts", []))
    if not include_expired:
        if as_of:
            # Return facts valid at a specific point in time
            facts = [f for f in facts
                     if (not f.get("valid_at") or f["valid_at"] <= as_of)
                     and (not f.get("expired_at") or f["expired_at"] > as_of)
                     and (not f.get("invalid_at") or f["invalid_at"] > as_of)]
        else:
            # Default: only current facts
            facts = [f for f in facts if not f.get("expired_at") and not f.get("invalid_at")]
    return {"entity": resolved, "category": category, "facts": facts}


def _extract_entities_from_query(query: str) -> list:
    if not FACTS_ROOT.exists():
        return []
    query_words = {w for w in re.findall(r"\b\w+\b", query.lower()) if w not in _ENTITY_STOPWORDS and len(w) >= 2}
    matches = []
    for entity in _get_entity_dirs_cached():
        entity_words = {w for w in re.findall(r"\b\w+\b", entity.lower())} - _ENTITY_STOPWORDS
        if entity.lower() in query_words:
            matches.append((entity, 3))
        elif entity_words & query_words:
            matches.append((entity, 2))
    matches.sort(key=lambda x: -x[1])
    return matches


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


# ── QMD helpers ───────────────────────────────────────────────────────────────

def _qmd_sanitize(query: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f]", " ", query).strip()


def _qmd_strip_stopwords(query: str) -> str:
    words = [w for w in re.findall(r"\b\w+\b", query.lower()) if w not in _ENTITY_STOPWORDS and len(w) >= 2]
    return " ".join(words) if words else query


# qmd rerank can fail (e.g. GPU OOM on context creation) — when it does, we
# fall back to skipRerank=true so hybrid search still returns results instead
# of a blanket HTTP 500. One-shot retry per request — previously we stuck
# skipRerank=true for 5 min after any 500, which silently killed rerank
# quality across the fleet. If GPU is consistently under pressure we'd rather
# see the errors than blind-degrade.
def _qmd_log(msg: str) -> None:
    # MCP server uses stdout for JSON-RPC; stderr is safe for diagnostics
    # and ends up in agent-services logs alongside other MCP chatter.
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
                      skip_rerank: bool = True) -> Optional[list]:
    """Send a hybrid lex+vec query to the qmd daemon.

    Phase cost (post 2026-04-20 binary-quantization patch, 157K vectors, 12 colls):
      lex       ~15ms   (BM25 FTS5)
      embed      ~5ms   (cached) / ~20ms cold
      vec       ~80ms   (sqlite-vec bit-KNN + blob rescore)
      chunk      ~0ms   (skipped when skip_rerank=True)
      rerank   500-750ms on novel queries (Qwen3-reranker, batched cross-encode)
               ~1ms on cache hit

    DEFAULT: skip_rerank=True (~100ms warm end-to-end).

    Evidence (2026-04-20 bench, 15 novel queries): the reranker never changes
    the #1 result, shuffles ~1-3 of positions 2-5 on ~67% of queries, and
    never surfaces anything outside the RRF top-20 candidate pool. 500ms for
    a within-top-5 reorder is not worth the latency tax — especially when the
    downstream consumer (LLM context) reads all 5 anyway.

    Pass skip_rerank=False explicitly only when ordering within the top-5
    matters for a human scanning results.
    """
    query = _qmd_sanitize(query)
    if not query:
        return []
    payload = {
        "searches": [
            {"type": "lex", "query": _qmd_strip_stopwords(query)},
            {"type": "vec", "query": query},
        ],
        "limit": limit,
        "collections": collections,
    }
    if skip_rerank:
        payload["skipRerank"] = True

    try:
        return _qmd_post(payload)
    except urllib.error.HTTPError as e:
        # Rerank context OOM manifests as HTTP 500. One-shot retry without
        # rerank so callers see documents instead of silent zero-hits. Not
        # sticky — if GPU pressure is persistent we'd rather surface the
        # errors than silently degrade rerank quality fleet-wide.
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
    parts = [f"Query: {query}\n\nSearch Results:\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"--- Result {i} (score: {r.get('score', 'N/A')}) ---")
        parts.append(f"File: {r.get('citation', r.get('path', ''))}")
        snippet = r.get("snippet", "")[:2000]
        parts.append(f"Content:\n{snippet}\n")
    payload = json.dumps({
        "model": CONSOLIDATION_MODEL,
        "messages": [{"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(parts)}],
        "temperature": 0.0, "max_tokens": 1000,
    }).encode()
    req = urllib.request.Request(CONSOLIDATION_ENDPOINT, data=payload, headers={"Content-Type": "application/json"}, method="POST")
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


def _run_vault_search(query: str, max_results: int, min_score: float, scope: str, consolidate: bool) -> str:
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

    # QMD search
    all_raw = []
    if len(coll_list) == 1:
        result = _qmd_daemon_search(query, max_results, coll_list)
        if result is not None:
            all_raw = result
    else:
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
        all_raw = sorted(merged.values(), key=lambda x: float(x.get("score", 0)), reverse=True)

    if not all_raw:
        all_raw = _qmd_subprocess_search(query, max_results, coll_list)

    # Parse results
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

    return json.dumps({
        "results": trimmed, "mode": "hybrid",
        "consolidated": consolidated is not None,
        "consolidated_summary": consolidated,
        "collections_searched": coll_list,
    })


# ── Vault helpers ─────────────────────────────────────────────────────────────

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


def _audit_write(path: str, byte_count: int) -> None:
    try:
        AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "agent_id": "lloyd", "path": path, "bytes": byte_count, "action": "write"}
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ── Tool handlers ─────────────────────────────────────────────────────────────

def _fact_get(params: dict) -> str:
    entity = params.get("entity", "").strip()
    if not entity:
        return json.dumps({"error": "entity is required", "facts": []})
    category = params.get("category") or None
    as_of = params.get("as_of") or None
    include_expired = bool(params.get("include_expired", False))
    try:
        return json.dumps(_get_facts_sync(entity, category, as_of=as_of,
                                          include_expired=include_expired))
    except Exception as exc:
        return json.dumps({"error": str(exc), "facts": []})


def _fact_add(params: dict) -> str:
    raw_entity = params.get("entity", "").strip()
    category = params.get("category", "").strip()
    fact_text = params.get("fact", "").strip()
    if not raw_entity or not category or not fact_text:
        return json.dumps({"error": "entity, category, and fact are required"})
    confidence = float(params.get("confidence", 0.9))
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        entity, is_new = _resolve_entity(raw_entity, auto_create=True)
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
        fact_file.write_text(_write_fact_frontmatter(frontmatter) + body, encoding="utf-8")
        global _entity_dirs_cache
        _entity_dirs_cache = None
        result = {"success": True, "fact_id": fact_id, "entity": entity, "category": category}
        if entity != raw_entity:
            result["resolved_from"] = raw_entity
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _fact_profile(params: dict) -> str:
    entity = params.get("entity", "").strip()
    if not entity:
        return json.dumps({"error": "entity is required"})
    try:
        facts = _get_facts_sync(entity).get("facts", [])
        categories = {}
        for fact in facts:
            cat = fact.get("category", "general")
            categories.setdefault(cat, []).append(fact)
        lines = [f"Profile for: {entity}"]
        for cat, cat_facts in categories.items():
            lines.append(f"\n{cat.upper()}:")
            for f in cat_facts[:3]:
                lines.append(f"  - {f.get('fact', '')}")
        return json.dumps({"entity": entity, "categories": categories, "fact_count": len(facts), "summary": "\n".join(lines)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _fact_check(params: dict) -> str:
    entity = params.get("entity", "").strip()
    if not entity:
        return json.dumps({"error": "entity is required", "contradictions": [], "checked": 0})
    try:
        return json.dumps(_detect_contradictions_sync(entity, params.get("category")))
    except Exception as exc:
        return json.dumps({"error": str(exc), "contradictions": [], "checked": 0})


def _fact_resolve(params: dict) -> str:
    entity = params.get("entity", "").strip()
    if not entity:
        return json.dumps({"error": "entity is required"})
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
                            fact_file.write_text(_write_fact_frontmatter(frontmatter) + body, encoding="utf-8")
                            resolved += 1
        return json.dumps({"entity": entity, "resolved": resolved, "remaining": len(contradictions) - resolved})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _fact_invalidate(params: dict) -> str:
    """Expire facts that are no longer current (were true, now outdated)."""
    entity = params.get("entity", "").strip()
    ended = params.get("ended", "").strip()
    if not entity or not ended:
        return json.dumps({"error": "entity and ended (ISO date) are required"})
    category = params.get("category") or None
    fact_substring = params.get("fact_substring", "").strip().lower()
    reason = params.get("reason", "").strip()
    try:
        resolved, _ = _resolve_entity(entity)
        entity_dir = _find_entity_dir(resolved)
        if not entity_dir:
            return json.dumps({"error": f"Entity not found: {entity}", "expired_count": 0})
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
                # Skip already expired/invalid facts
                if f.get("expired_at") or f.get("invalid_at"):
                    continue
                # Match by substring if provided, otherwise match all
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
                fact_file.write_text(_write_fact_frontmatter(frontmatter) + body, encoding="utf-8")
        return json.dumps({"success": True, "entity": resolved, "expired_count": expired_count, "matched_facts": matched_facts})
    except Exception as exc:
        return json.dumps({"error": str(exc), "expired_count": 0})


# ── Relationship helpers ─────────────────────────────────────────────────────

RELATIONSHIPS_PATH = FACTS_ROOT / "_relationships.json"


def _load_relationships() -> dict:
    if not RELATIONSHIPS_PATH.exists():
        return {"edges": [], "schema_version": 1}
    try:
        return json.loads(RELATIONSHIPS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"edges": [], "schema_version": 1}


def _save_relationships(data: dict) -> None:
    RELATIONSHIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RELATIONSHIPS_PATH.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def _fact_relate(params: dict) -> str:
    """Add a typed relationship edge between two entities."""
    source = params.get("source", "").strip()
    target = params.get("target", "").strip()
    rel_type = params.get("type", "").strip()
    if not source or not target or not rel_type:
        return json.dumps({"error": "source, target, and type are required"})
    confidence = float(params.get("confidence", 0.9))
    provenance = params.get("provenance", "STATED")
    if provenance not in ("STATED", "EXTRACTED", "INFERRED", "AMBIGUOUS"):
        provenance = "STATED"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        # Resolve entity names
        src_resolved, _ = _resolve_entity(source, auto_create=True)
        tgt_resolved, _ = _resolve_entity(target, auto_create=True)
        data = _load_relationships()
        # Check for duplicate
        for edge in data["edges"]:
            if (edge["source"] == src_resolved and edge["target"] == tgt_resolved
                    and edge["type"] == rel_type and not edge.get("expired_at")):
                return json.dumps({"success": True, "action": "already_exists",
                                   "source": src_resolved, "target": tgt_resolved, "type": rel_type})
        new_edge = {
            "source": src_resolved, "target": tgt_resolved, "type": rel_type,
            "confidence": confidence, "provenance": provenance,
            "created_at": now_iso, "expired_at": None,
            "source_doc": params.get("source_doc"),
        }
        data["edges"].append(new_edge)
        _save_relationships(data)
        return json.dumps({"success": True, "action": "created",
                           "source": src_resolved, "target": tgt_resolved, "type": rel_type})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _fact_relationships(params: dict) -> str:
    """Get all relationships for an entity (inbound + outbound)."""
    entity = params.get("entity", "").strip()
    if not entity:
        return json.dumps({"error": "entity is required", "edges": []})
    direction = params.get("direction", "both")  # "in", "out", "both"
    rel_type = params.get("type") or None
    try:
        resolved, _ = _resolve_entity(entity)
        data = _load_relationships()
        edges = []
        for edge in data["edges"]:
            if edge.get("expired_at"):
                continue
            match = False
            if direction in ("out", "both") and edge["source"] == resolved:
                match = True
            if direction in ("in", "both") and edge["target"] == resolved:
                match = True
            if match and rel_type and edge["type"] != rel_type:
                match = False
            if match:
                edges.append(edge)
        return json.dumps({"entity": resolved, "edges": edges, "count": len(edges)})
    except Exception as exc:
        return json.dumps({"error": str(exc), "edges": []})


def _fact_path(params: dict) -> str:
    """Find shortest path between two entities via BFS on relationship graph."""
    source = params.get("source", "").strip()
    target = params.get("target", "").strip()
    max_hops = int(params.get("max_hops", 3))
    if not source or not target:
        return json.dumps({"error": "source and target are required"})
    try:
        src_resolved, _ = _resolve_entity(source)
        tgt_resolved, _ = _resolve_entity(target)
        data = _load_relationships()
        # Build adjacency list from active edges
        adj: dict[str, list[tuple[str, dict]]] = {}
        for edge in data["edges"]:
            if edge.get("expired_at"):
                continue
            s, t = edge["source"], edge["target"]
            adj.setdefault(s, []).append((t, edge))
            adj.setdefault(t, []).append((s, edge))
        # BFS
        from collections import deque
        queue = deque([(src_resolved, [src_resolved], [])])
        visited = {src_resolved}
        while queue:
            node, path, edges_path = queue.popleft()
            if node == tgt_resolved:
                return json.dumps({"found": True, "path": path, "edges": edges_path, "hops": len(edges_path)})
            if len(path) > max_hops:
                continue
            for neighbor, edge in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor],
                                  edges_path + [{"source": edge["source"], "target": edge["target"], "type": edge["type"]}]))
        return json.dumps({"found": False, "path": [], "edges": [], "hops": -1})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _fact_neighbors(params: dict) -> str:
    """Get neighborhood subgraph around an entity within N hops."""
    entity = params.get("entity", "").strip()
    if not entity:
        return json.dumps({"error": "entity is required"})
    hops = int(params.get("hops", 1))
    min_confidence = float(params.get("min_confidence", 0.5))
    try:
        resolved, _ = _resolve_entity(entity)
        data = _load_relationships()
        # Build adjacency list
        adj: dict[str, list[tuple[str, dict]]] = {}
        for edge in data["edges"]:
            if edge.get("expired_at") or edge.get("confidence", 1.0) < min_confidence:
                continue
            s, t = edge["source"], edge["target"]
            adj.setdefault(s, []).append((t, edge))
            adj.setdefault(t, []).append((s, edge))
        # BFS up to N hops
        from collections import deque
        visited = {resolved}
        current_layer = [resolved]
        all_edges = []
        for _ in range(hops):
            next_layer = []
            for node in current_layer:
                for neighbor, edge in adj.get(node, []):
                    edge_key = (edge["source"], edge["target"], edge["type"])
                    all_edges.append({"source": edge["source"], "target": edge["target"],
                                      "type": edge["type"], "confidence": edge.get("confidence", 1.0)})
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.append(neighbor)
            current_layer = next_layer
        # Deduplicate edges
        seen_edges = set()
        unique_edges = []
        for e in all_edges:
            key = (e["source"], e["target"], e["type"])
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)
        return json.dumps({"entity": resolved, "nodes": sorted(visited),
                           "edges": unique_edges, "node_count": len(visited), "edge_count": len(unique_edges)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _vault_get(params: dict) -> str:
    path = params.get("path", "").strip()
    if not path:
        return json.dumps({"error": "path is required"})
    try:
        target = VAULT / path
        if not target.resolve().is_relative_to(VAULT.resolve()):
            return json.dumps({"error": "path escapes vault root"})
        if not target.exists():
            resolved = _resolve_case_insensitive(path)
            if resolved is None:
                return json.dumps({"error": f"File not found: {path}"})
            target = resolved
        text = target.read_text(encoding="utf-8", errors="replace")
        start_line = int(params.get("start_line", 0))
        num_lines = int(params.get("num_lines", 0))
        if start_line > 0 or num_lines > 0:
            lines = text.splitlines()
            start = max(0, start_line - 1)
            end = (start + num_lines) if num_lines > 0 else len(lines)
            text = "\n".join(lines[start:end])
        return json.dumps({"path": path, "text": text or "(empty file)"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _vault_write(params: dict) -> str:
    path = params.get("path", "").strip()
    content = params.get("content", "")
    if not path:
        return json.dumps({"error": "path is required"})
    try:
        target = VAULT / path
        if not target.resolve().is_relative_to(VAULT.resolve()):
            return json.dumps({"error": "path escapes vault root"})
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        byte_count = len(content.encode("utf-8"))
        _audit_write(path, byte_count)
        return json.dumps({"success": True, "path": path, "bytes": byte_count})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _vault_overview(params: dict) -> str:
    try:
        if not VAULT.exists():
            return json.dumps({"error": f"Vault not found: {VAULT}"})
        totals, grand_total = {}, 0
        for segment in VAULT_SEGMENTS:
            seg_dir = VAULT / segment
            if not seg_dir.is_dir():
                totals[segment] = 0
                continue
            count = sum(1 for f in seg_dir.rglob("*.md") if f.name not in VAULT_EXCLUDE_FILES and not any(p in VAULT_EXCLUDE_DIRS for p in f.parts))
            totals[segment] = count
            grand_total += count
        return json.dumps({"total_files": grand_total, "segments": totals})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _vault_search(params: dict) -> str:
    query = params.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required", "results": []})
    try:
        return _run_vault_search(query, int(params.get("max_results", 10)), float(params.get("min_score", 0.0)), params.get("scope", ""), params.get("consolidate", True))
    except Exception as exc:
        return json.dumps({"error": str(exc), "results": []})


def _graph_expand_entities(seed_entities: list[str], hops: int = 1) -> list[str]:
    """Expand a set of seed entities via relationship graph traversal."""
    if not RELATIONSHIPS_PATH.exists():
        return []
    try:
        data = _load_relationships()
        adj: dict[str, set[str]] = {}
        for edge in data["edges"]:
            if edge.get("expired_at"):
                continue
            adj.setdefault(edge["source"], set()).add(edge["target"])
            adj.setdefault(edge["target"], set()).add(edge["source"])
        expanded = set()
        current = set(seed_entities)
        for _ in range(hops):
            next_layer = set()
            for entity in current:
                for neighbor in adj.get(entity, set()):
                    if neighbor not in seed_entities and neighbor not in expanded:
                        next_layer.add(neighbor)
            expanded.update(next_layer)
            current = next_layer
        return list(expanded)
    except Exception:
        return []


def _graph_weighted_neighbors(
    seed_entities: list[str], top_k: int = 3, hops: int = 1
) -> list[tuple[str, float]]:
    """Weighted graph expansion: return top-k neighbors scored by
    edge confidence × EDGE_TYPE_WEIGHTS[type].

    Multiple edges to the same neighbor sum their contributions so entities
    connected by multiple typed relationships rise to the top. Seed entities
    are excluded from results.

    Returns [(entity, weight)] sorted by weight desc.
    """
    if not RELATIONSHIPS_PATH.exists() or not seed_entities:
        return []
    try:
        data = _load_relationships()
    except Exception:
        return []

    # adjacency with edge metadata
    adj: dict[str, list[tuple[str, str, float]]] = {}
    for edge in data.get("edges", []):
        if edge.get("expired_at"):
            continue
        src, tgt, etype = edge.get("source"), edge.get("target"), edge.get("type", "")
        conf = float(edge.get("confidence", 0.5))
        if not src or not tgt:
            continue
        adj.setdefault(src, []).append((tgt, etype, conf))
        adj.setdefault(tgt, []).append((src, etype, conf))

    seed_set = set(seed_entities)
    # scores[entity] = accumulated weight, capped at 1.0 to avoid runaway
    scores: dict[str, float] = {}
    current = set(seed_entities)
    visited = set(seed_entities)

    for hop in range(hops):
        # decay by hop so 2-hop neighbors weigh less than 1-hop
        hop_decay = 1.0 if hop == 0 else 0.5 ** hop
        next_layer = set()
        for entity in current:
            for neighbor, etype, conf in adj.get(entity, []):
                if neighbor in seed_set:
                    continue
                w = EDGE_TYPE_WEIGHTS.get(etype, _DEFAULT_EDGE_WEIGHT) * conf * hop_decay
                scores[neighbor] = min(1.0, scores.get(neighbor, 0.0) + w)
                if neighbor not in visited:
                    next_layer.add(neighbor)
        visited.update(next_layer)
        current = next_layer

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return ranked[:top_k]


def _vault_recall(params: dict) -> str:
    query = params.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required", "documents": [], "facts": []})
    limit = int(params.get("limit", 20))
    include_facts = params.get("include_facts", True)
    expand_graph = params.get("expand_graph", False)

    # Extract seed entities once — used by fact expansion below.
    seed_entities = [e for e, _ in _extract_entities_from_query(query)[:5]]

    # Weighted graph expansion: rank neighbors by edge_type × confidence.
    # Currently drives fact expansion only. Document augmentation via
    # additional qmd queries was tried (Phase 1A prototype, 2026-04-20) but
    # regressed recall and tripled latency — reverted. Phase 2 (PPR + typed
    # edges after the classifier lands) should produce a measurable lift.
    weighted_neighbors: list[tuple[str, float]] = []
    if expand_graph and seed_entities:
        weighted_neighbors = _graph_weighted_neighbors(seed_entities, top_k=5, hops=1)

    def _do_search():
        result = _qmd_daemon_search(query, limit, VAULT_SEGMENTS)
        if result is None:
            result = _qmd_subprocess_search(query, limit, VAULT_SEGMENTS)
        return result or []

    def _do_facts():
        if not include_facts:
            return [], []
        facts = []
        for entity in seed_entities:
            try:
                entity_data = _get_facts_sync(entity)
                if entity_data.get("facts"):
                    top = sorted(entity_data["facts"], key=lambda f: f.get("confidence", 0.5), reverse=True)
                    facts.extend(top[:3])
            except Exception:
                pass
        # Graph expansion: also fetch top facts from weighted neighbors.
        # Neighbors are already sorted by edge_weight × confidence desc, so
        # stronger semantic ties surface first.
        graph_facts = []
        if expand_graph and weighted_neighbors:
            for entity, _w in weighted_neighbors[:5]:
                try:
                    entity_data = _get_facts_sync(entity)
                    if entity_data.get("facts"):
                        top = sorted(entity_data["facts"], key=lambda f: f.get("confidence", 0.5), reverse=True)
                        graph_facts.extend(top[:2])
                except Exception:
                    pass
        return facts, graph_facts

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            search_fut = pool.submit(_do_search)
            facts_fut = pool.submit(_do_facts)
            raw_results = search_fut.result()
            facts, graph_facts = facts_fut.result()
        documents = []
        for r in raw_results[:limit]:
            path = r.get("file", "").removeprefix("qmd://")
            if path.startswith("obsidian/"):
                path = path.removeprefix("obsidian/")
            documents.append({
                "path": path,
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "score": r.get("score", 0),
            })
        result = {"documents": documents, "facts": facts, "query": query}
        if graph_facts:
            result["graph_expanded_facts"] = graph_facts
        if weighted_neighbors:
            result["graph_neighbors_used"] = [
                {"entity": e, "weight": round(w, 3)} for e, w in weighted_neighbors
            ]
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc), "documents": [], "facts": []})


# ── Session memory tools ─────────────────────────────────────────────────────

def _check_injection(text: str) -> Optional[str]:
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return f"Potential prompt injection detected in entry"
    return None


def _memory_read(params: dict) -> str:
    file = params.get("file", "MEMORY.md").strip()
    if file not in MEMORY_FILES:
        return json.dumps({"error": f"Invalid file. Must be one of: {', '.join(sorted(MEMORY_FILES))}"})
    filepath = MEMORIES_ROOT / file
    if not filepath.exists():
        return json.dumps({"content": "", "file": file})
    return json.dumps({"content": filepath.read_text(encoding="utf-8"), "file": file})


def _memory_add(params: dict) -> str:
    file = params.get("file", "MEMORY.md").strip()
    entry = params.get("entry", "").strip()
    if file not in MEMORY_FILES:
        return json.dumps({"error": f"Invalid file. Must be one of: {', '.join(sorted(MEMORY_FILES))}"})
    if not entry:
        return json.dumps({"error": "entry is required"})
    err = _check_injection(entry)
    if err:
        return json.dumps({"error": err})
    MEMORIES_ROOT.mkdir(parents=True, exist_ok=True)
    filepath = MEMORIES_ROOT / file
    existing = filepath.read_text(encoding="utf-8") if filepath.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    filepath.write_text(existing + entry + "\n", encoding="utf-8")
    return json.dumps({"success": True, "file": file})


def _memory_replace(params: dict) -> str:
    file = params.get("file", "MEMORY.md").strip()
    old_text = params.get("old_text", "")
    new_text = params.get("new_text", "")
    if file not in MEMORY_FILES:
        return json.dumps({"error": f"Invalid file. Must be one of: {', '.join(sorted(MEMORY_FILES))}"})
    if not old_text:
        return json.dumps({"error": "old_text is required"})
    err = _check_injection(new_text)
    if err:
        return json.dumps({"error": err})
    filepath = MEMORIES_ROOT / file
    if not filepath.exists():
        return json.dumps({"error": f"{file} does not exist"})
    content = filepath.read_text(encoding="utf-8")
    if old_text not in content:
        return json.dumps({"error": "old_text not found in file", "matched": False})
    filepath.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return json.dumps({"success": True, "file": file})


def _memory_remove(params: dict) -> str:
    file = params.get("file", "MEMORY.md").strip()
    entry = params.get("entry", "").strip()
    if file not in MEMORY_FILES:
        return json.dumps({"error": f"Invalid file. Must be one of: {', '.join(sorted(MEMORY_FILES))}"})
    if not entry:
        return json.dumps({"error": "entry is required"})
    filepath = MEMORIES_ROOT / file
    if not filepath.exists():
        return json.dumps({"error": f"{file} does not exist"})
    content = filepath.read_text(encoding="utf-8")
    if entry not in content:
        return json.dumps({"error": "entry not found in file", "matched": False})
    updated = content.replace(entry, "", 1)
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    filepath.write_text(updated, encoding="utf-8")
    return json.dumps({"success": True, "file": file})


# ── Session recall ───────────────────────────────────────────────────────────

SESSIONS_DIR = Path.home() / "lloyd" / "sessions"
_SESSION_INDEX_TTL = 120       # cache session index for 2 min
_SESSION_CORPUS_MAX = 5000     # max chars of searchable text per session

_session_index_cache: Optional[tuple] = None  # (monotonic_ts, {filename: metadata})


def _extract_msg_text(msg: dict) -> str:
    """Extract plain text from a session message, skipping injected context."""
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        text = " ".join(t for t in parts if t)
    elif isinstance(content, str):
        text = content
    else:
        return ""
    stripped = text.strip()
    if any(stripped.startswith(p) for p in (
        "<context>", "<system-reminder>", "<memory>", "<daily_notes>",
        "[cron:", "[System Message]", "[autonomy:",
    )):
        return ""
    return text


def _load_session_index(max_days: int = 14) -> dict:
    """Load session metadata from recent JSON files. Cached with TTL.

    Returns {filename: {session_id, date_str, time_str, created_at, model,
    preview, message_count, platform, corpus, user_snippets}}.
    """
    global _session_index_cache
    now = time.monotonic()

    if _session_index_cache and (now - _session_index_cache[0]) < _SESSION_INDEX_TTL:
        return _session_index_cache[1]

    cutoff = (datetime.datetime.now() - datetime.timedelta(days=max_days)).strftime("%Y%m%d")
    index: dict[str, dict] = {}

    if not SESSIONS_DIR.exists():
        _session_index_cache = (now, index)
        return index

    for f in SESSIONS_DIR.iterdir():
        if not f.name.endswith(".json") or f.name.startswith("autonomy_"):
            continue
        parts = f.name.split("_")
        if len(parts) < 3 or len(parts[0]) != 8:
            continue
        if parts[0] < cutoff:
            continue

        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("platform") == "autonomy":
                continue

            user_texts: list[str] = []
            asst_texts: list[str] = []
            for msg in data.get("messages", []):
                text = _extract_msg_text(msg)
                if not text.strip():
                    continue
                if msg.get("role") == "user":
                    user_texts.append(text[:500])
                elif msg.get("role") == "assistant":
                    asst_texts.append(text[:300])

            corpus = " ".join(user_texts + asst_texts).lower()[:_SESSION_CORPUS_MAX]

            index[f.name] = {
                "filename": f.name,
                "session_id": data.get("session_id", f.stem),
                "date_str": parts[0],
                "time_str": parts[1] if len(parts) > 1 else "",
                "created_at": data.get("created_at", ""),
                "model": data.get("model", ""),
                "preview": data.get("preview", ""),
                "message_count": data.get("message_count", 0),
                "platform": data.get("platform", ""),
                "corpus": corpus,
                "user_snippets": [t[:300] for t in user_texts[:8]],
            }
        except Exception:
            continue

    _session_index_cache = (now, index)
    return index


def _score_session(session: dict, query_tokens: set) -> float:
    """Score a session against query tokens using term frequency."""
    corpus = session.get("corpus", "")
    if not corpus or not query_tokens:
        return 0.0
    score = 0.0
    for token in query_tokens:
        count = corpus.count(token)
        if count > 0:
            score += 1.0 + 0.3 * min(count - 1, 4)
    return score / len(query_tokens)


def _session_recall(params: dict) -> str:
    """Search recent session transcripts for topics, decisions, or discussions."""
    query = params.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required", "sessions": []})

    days = int(params.get("days", 7))
    limit = int(params.get("limit", 5))

    index = _load_session_index(max_days=max(days, 14))

    # Filter to requested date range
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
    sessions = [s for s in index.values() if s["date_str"] >= cutoff]

    # Tokenize query (reuse existing stopwords)
    query_tokens = {w for w in re.findall(r"\w+", query.lower())
                    if w not in _ENTITY_STOPWORDS and len(w) >= 2}

    if not query_tokens:
        # No meaningful tokens — return most recent sessions
        sessions.sort(key=lambda s: s["date_str"] + s.get("time_str", ""), reverse=True)
        results = [{
            "session_id": s["session_id"],
            "created_at": s["created_at"],
            "model": s["model"],
            "preview": s["preview"][:200],
            "message_count": s["message_count"],
            "snippets": s["user_snippets"][:3],
        } for s in sessions[:limit]]
        return json.dumps({"query": query, "sessions": results, "total_searched": len(sessions)})

    # Score and rank
    scored = []
    for s in sessions:
        score = _score_session(s, query_tokens)
        if score > 0.2:
            scored.append((score, s))
    scored.sort(key=lambda x: -x[0])

    results = []
    for score, s in scored[:limit]:
        # Extract matching snippets
        snippets = []
        for text in s.get("user_snippets", []):
            if any(t in text.lower() for t in query_tokens):
                snippets.append(text[:300])
                if len(snippets) >= 3:
                    break
        results.append({
            "session_id": s["session_id"],
            "created_at": s["created_at"],
            "model": s["model"],
            "preview": s["preview"][:200],
            "message_count": s["message_count"],
            "match_score": round(score, 3),
            "snippets": snippets or [s["preview"][:200]],
        })

    return json.dumps({"query": query, "sessions": results, "total_searched": len(sessions)})


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
        Tool(name="fact_neighbors", description="Get neighborhood subgraph around an entity within N hops.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "hops": {"type": "integer"}, "min_confidence": {"type": "number"}}, "required": ["entity"]}),
        Tool(name="vault_get", description="Read a file from the obsidian vault by vault-relative path.", inputSchema={
            "type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "num_lines": {"type": "integer"}}, "required": ["path"]}),
        Tool(name="vault_write", description="Write content to a vault file. Audit-logged.", inputSchema={
            "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
        Tool(name="vault_overview", description="Get vault statistics: file counts per segment.", inputSchema={
            "type": "object", "properties": {"detail": {"type": "string", "enum": ["summary", "hubs"]}}}),
        Tool(name="vault_search", description="Hybrid BM25+vector search across the obsidian vault.", inputSchema={
            "type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}, "min_score": {"type": "number"}, "scope": {"type": "string"}, "consolidate": {"type": "boolean"}}, "required": ["query"]}),
        Tool(name="vault_recall", description="Combined recall: vault search + entity fact retrieval in parallel. Use expand_graph=true to include facts from related entities.", inputSchema={
            "type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "include_facts": {"type": "boolean"}, "expand_graph": {"type": "boolean", "description": "Expand results via relationship graph (1 hop)"}}, "required": ["query"]}),
        Tool(name="memory_read", description="Read MEMORY.md or USER.md session memory files.", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"], "description": "Which file to read"}}, "required": []}),
        Tool(name="memory_add", description="Append an entry to MEMORY.md or USER.md.", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"]}, "entry": {"type": "string", "description": "Text to append"}}, "required": ["entry"]}),
        Tool(name="memory_replace", description="Replace text in MEMORY.md or USER.md (substring match).", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"]}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["old_text", "new_text"]}),
        Tool(name="memory_remove", description="Remove an entry from MEMORY.md or USER.md (substring match).", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"]}, "entry": {"type": "string", "description": "Text to remove"}}, "required": ["entry"]}),
        Tool(name="session_recall", description="Search recent session transcripts for topics, decisions, or discussions from past sessions. Use for cross-session context like 'what did we work on today?' or 'what was decided about X?'", inputSchema={
            "type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "days": {"type": "integer", "description": "Days back to search (default: 7)"}, "limit": {"type": "integer", "description": "Max results (default: 5)"}}, "required": ["query"]}),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    handlers = {
        "fact_get": _fact_get, "fact_add": _fact_add, "fact_profile": _fact_profile,
        "fact_check": _fact_check, "fact_resolve": _fact_resolve,
        "fact_invalidate": _fact_invalidate,
        "fact_relate": _fact_relate, "fact_relationships": _fact_relationships,
        "fact_path": _fact_path, "fact_neighbors": _fact_neighbors,
        "vault_get": _vault_get, "vault_write": _vault_write, "vault_overview": _vault_overview,
        "vault_search": _vault_search, "vault_recall": _vault_recall,
        "memory_read": _memory_read, "memory_add": _memory_add,
        "memory_replace": _memory_replace, "memory_remove": _memory_remove,
        "session_recall": _session_recall,
    }
    handler = handlers.get(name)
    if handler:
        return [TextContent(type="text", text=handler(arguments))]
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

