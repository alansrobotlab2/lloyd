#!/usr/bin/env python3
"""
Lloyd MCP Server: Memory — knowledge graph, facts, and vault tools.

Fact tools: fact_get, fact_add, fact_profile, fact_check, fact_resolve
Vault tools: vault_get, vault_write, vault_overview, vault_search, vault_recall

Facts data: ~/obsidian/facts/
Vault root: ~/obsidian/
QMD daemon: http://localhost:8181/query
"""

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

# Path constants and shared helpers extracted to agent_mcp/_shared.py (#340 PR 1).
# Re-imported here to preserve the existing memory.X public surface for callers
# and tests during the multi-PR refactor.
from agent_mcp._shared import (  # noqa: E402,F401  (re-exports)
    VAULT,
    FACTS_ROOT,
    ALIASES_PATH,
    _ENTITY_STOPWORDS,
    _SCORING_STOPWORDS,
    _QUERY_STOPWORDS,
    _token_overlap,
    _levenshtein,
    _fuzzy_entity_match,
    _parse_fact_frontmatter,
    _write_fact_frontmatter,
    _find_entity_dir,
    _load_aliases,
    _save_aliases,
    _get_entity_dirs_cached,
    _invalidate_entity_dirs_cache,
    _resolve_entity,
)

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


def _generate_fact_id(category: str) -> str:
    return f"{category[:4]}-{uuid.uuid4().hex[:4]}"


def _get_facts_sync(entity: str, category: str = None, as_of: str = None,
                     include_expired: bool = False) -> dict:
    resolved, _ = _resolve_entity(entity, mode="read")
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


# Task-ID extractor. Matches "Task #299", "Task 299", "#299", "task_310",
# "backlog_120", "backlog_item_18", "backlog_task_41". Captures the numeric
# ID so we can dispatch to whichever naming convention exists.
_TASK_ID_RE = re.compile(
    r"(?:\btask\s*[#_ ]?|#|\bbacklog[_ -]?(?:item[_ -]?|task[_ -]?)?)(\d{1,4})\b",
    re.IGNORECASE,
)

# Cache for entity degree (non-expired edge count) — used as a deterministic
# tie-break signal in _extract_entities_from_query. 60s TTL keeps the hot
# prefetch path fast without going fully stale.
_edge_count_cache: Optional[tuple] = None
_EDGE_COUNT_TTL = 60


def _get_entity_edge_counts() -> dict:
    global _edge_count_cache
    now = time.monotonic()
    if _edge_count_cache is not None and (now - _edge_count_cache[0]) < _EDGE_COUNT_TTL:
        return _edge_count_cache[1]
    counts: dict[str, int] = {}
    rel_path = FACTS_ROOT / "_relationships.json"
    if rel_path.exists():
        try:
            data = json.loads(rel_path.read_text(encoding="utf-8"))
            for edge in data.get("edges", []):
                if edge.get("expired_at") or edge.get("invalid_at"):
                    continue
                s, t = edge.get("source"), edge.get("target")
                if s:
                    counts[s] = counts.get(s, 0) + 1
                if t:
                    counts[t] = counts.get(t, 0) + 1
        except Exception:
            pass
    _edge_count_cache = (now, counts)
    return counts


def _extract_entities_from_query(query: str) -> list:
    """Rank known entities by how well they match the query.

    Prior implementation (pre-#312) scored binary 2-or-3 with tie-breaks
    resolved by arbitrary dict iteration order. With FACT_MAX_ENTITIES=2
    downstream that meant legitimate entities got dropped on ties. Trace
    evidence from #306 showed ~50% weak-match rate — e.g. "deep dive into
    299" grabbed `Deep Agent` over a (missing) `Task #299`, and "classifier
    prompt tweaks" grabbed `Fact Extraction 2B Model`.

    New scoring:
      - Task-ID references (#299, backlog_18, task_310) dispatch to canonical
        `Task #N` / legacy forms with a fixed high score.
      - Full-name substring (entity name appears verbatim in query) gets a
        strong bonus scaled by length.
      - Token overlap is scored (overlap^2) / (entity_tokens * query_tokens)
        so 1-token matches on multi-token entities don't win.
      - Ties broken deterministically: score → longer canonical → higher
        graph degree → alphabetical.
    """
    if not FACTS_ROOT.exists():
        return []

    q_lower = query.lower()
    q_tokens = {
        w for w in re.findall(r"\b\w+\b", q_lower)
        if w not in _SCORING_STOPWORDS and len(w) >= 2
    }

    entities = _get_entity_dirs_cached()
    entity_lookup = {e.lower(): e for e in entities}

    scores: dict[str, float] = {}

    def _bump(name: str, score: float) -> None:
        # Resolve to canonical case if we have a directory for it.
        canonical = entity_lookup.get(name.lower(), name)
        if scores.get(canonical, 0.0) < score:
            scores[canonical] = score

    # 1. Task-ID direct dispatch (highest-priority signal).
    for m in _TASK_ID_RE.finditer(q_lower):
        tid = m.group(1)
        # Try every known naming convention in the vault.
        for candidate in (
            f"Task #{tid}", f"Task {tid}", f"Task_{tid}",
            f"backlog_{tid}", f"backlog_item_{tid}", f"backlog_task_{tid}",
            tid,
        ):
            hit = entity_lookup.get(candidate.lower())
            if hit:
                _bump(hit, 10.0)

    # 2. Full-name match at word boundaries (entity appears verbatim in
    #    query as a whole word / phrase, not as a substring of a larger
    #    token — "dee" inside "deep" must not match "Dee").
    for e_lower, e_cased in entity_lookup.items():
        if len(e_lower) < 3:
            continue
        if re.search(r"(?<!\w)" + re.escape(e_lower) + r"(?!\w)", q_lower):
            _bump(e_cased, 5.0 + min(len(e_lower) / 20.0, 2.0))

    # 3. Token-overlap scoring. Reward specificity on both sides.
    if q_tokens:
        q_norm = max(len(q_tokens), 2)
        for e_lower, e_cased in entity_lookup.items():
            e_tokens = {
                w for w in re.findall(r"\b\w+\b", e_lower)
                if w not in _SCORING_STOPWORDS and len(w) >= 2
            }
            if not e_tokens:
                continue
            overlap = e_tokens & q_tokens
            if not overlap:
                continue
            # Quadratic reward for multi-token matches; normalized by both
            # sides so a single common token against a huge query doesn't
            # bubble up.
            score = (len(overlap) ** 2) / (len(e_tokens) * q_norm)
            # Floor: single-token entity exactly matched to a query token is
            # legitimate (e.g. query "prefetch" ↔ entity "prefetch").
            if len(e_tokens) == 1 and len(overlap) == 1:
                score = max(score, 0.5)
            if score >= 0.25:
                _bump(e_cased, score)

    if not scores:
        return []

    # Deterministic tie-break: score desc → longer canonical → higher graph
    # degree → alphabetical. Longer canonical first deliberately prefers
    # specific entities ("Lloyd Agent") over generic parents ("Lloyd") when
    # all else is equal — the specific one usually carries more relevant
    # facts for a query that mentioned it.
    edge_counts = _get_entity_edge_counts()
    ranked = sorted(
        scores.items(),
        key=lambda kv: (
            -kv[1],
            -len(kv[0]),
            -edge_counts.get(kv[0], 0),
            kv[0].lower(),
        ),
    )
    return ranked


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
    """Strip control chars and collapse qmd query-syntax operators to spaces.

    qmd's vec/hyde parser treats `-term` as negation (Google-style) and throws
    HTTP 500 when it sees one — "Negation (-term) is not supported in vec/hyde
    queries." Questions like "next-gen memory subsystem", "SMPL-X pipeline",
    "three-layer architecture", or "end-to-end flow" hit this silently: hybrid
    search returns zero results and the caller sees no documents.

    We replace hyphens (and other operator chars qmd treats as syntax) with
    spaces before the query hits either the vec or lex leg. The lex path
    already tokenizes on non-word boundaries so this is a no-op there; the
    vec path gets identical clean input and embeddings tokenize robustly
    across the substitution (`SMPL-X` ≈ `SMPL X`). Fixes #325.
    """
    q = re.sub(r"[\x00-\x1f\x7f]", " ", query)
    # qmd query-syntax operators → space. `-` is negation; `+`/`"` also reserved.
    q = re.sub(r"[-+\"]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _qmd_strip_stopwords(query: str) -> str:
    """Strip stopwords from the FTS5 lex-leg query.

    Uses the wider _QUERY_STOPWORDS (see #327) rather than _ENTITY_STOPWORDS
    so natural-language question framing gets fully removed from BM25 signal.
    Fall-through to the original query if everything strips — protects short
    all-stopword inputs ("what is it?") from becoming empty.
    """
    words = [w for w in re.findall(r"\b\w+\b", query.lower()) if w not in _QUERY_STOPWORDS and len(w) >= 2]
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

    # Subprocess fallback removed 2026-04-20 — see note in _do_search above
    # (_vault_recall). Empty daemon result means no hits, not "try the slow
    # cold-start CLI subprocess for 30s."

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
        # default=str handles datetime event_date values parsed natively from
        # YAML (pre-existing data issue; see vault_recall for precedent at
        # line ~1481).
        return json.dumps(_get_facts_sync(entity, category, as_of=as_of,
                                          include_expired=include_expired),
                          default=str)
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
        # mode="write" — exact + alias only, no fuzzy match. The fact lands
        # on the literal name the caller specified. (#340 PR 3 — fixes the
        # silent fuzzy-merge data-corruption bug.)
        entity, is_new = _resolve_entity(raw_entity, mode="write")
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
        _invalidate_entity_dirs_cache()
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
        return json.dumps({"entity": entity, "categories": categories, "fact_count": len(facts), "summary": "\n".join(lines)}, default=str)
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
        resolved, _ = _resolve_entity(entity, mode="read")
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

# In-memory cache for the relationships index (#340 PR 2).
#
# Layout: (mtime_ns, parsed_data) | None
#
# Invalidation strategy:
#   - Reads stat() the file and compare mtime_ns. Mismatch → reload.
#   - _save_relationships() refreshes the cache with the new mtime.
#   - This handles both in-process mutation (load → mutate → save) and
#     cross-process writes (autonomy classifier writes file, MCP server
#     picks up via stat-check on next read).
#
# Mutation contract: callers that mutate the returned dict MUST follow
# with _save_relationships(). Between load and save, the cache and the
# caller share the same object — there's no defensive deep-copy because
# the MCP server is single-threaded and the file is small enough to
# parse but large enough (2.3 MB / 4.6k edges) that copying defeats
# the cache.
_relationships_cache: Optional[tuple[int, dict]] = None


def _load_relationships() -> dict:
    """Load the relationships index, with mtime-based caching.

    Returns the parsed dict. On missing file or parse error, returns the
    empty schema and clears the cache.
    """
    global _relationships_cache
    if not RELATIONSHIPS_PATH.exists():
        _relationships_cache = None
        return {"edges": [], "schema_version": 1}
    try:
        mtime_ns = RELATIONSHIPS_PATH.stat().st_mtime_ns
    except OSError:
        return {"edges": [], "schema_version": 1}
    if _relationships_cache is not None and _relationships_cache[0] == mtime_ns:
        return _relationships_cache[1]
    try:
        data = json.loads(RELATIONSHIPS_PATH.read_text(encoding="utf-8"))
    except Exception:
        # Don't poison the cache with an empty placeholder; just return one.
        return {"edges": [], "schema_version": 1}
    _relationships_cache = (mtime_ns, data)
    return data


def _save_relationships(data: dict) -> None:
    """Persist the relationships index and refresh the cache."""
    global _relationships_cache
    RELATIONSHIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RELATIONSHIPS_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=False), encoding="utf-8"
    )
    try:
        new_mtime = RELATIONSHIPS_PATH.stat().st_mtime_ns
        _relationships_cache = (new_mtime, data)
    except OSError:
        _relationships_cache = None


def _invalidate_relationships_cache() -> None:
    """Clear the relationships cache. Used by tests and forced reloads."""
    global _relationships_cache
    _relationships_cache = None


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
        # Resolve entity names — mode="write" so edges land on the literal
        # names the caller specified, not fuzzy-matched neighbours. (#340
        # PR 3.)
        src_resolved, _ = _resolve_entity(source, mode="write")
        tgt_resolved, _ = _resolve_entity(target, mode="write")
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
        resolved, _ = _resolve_entity(entity, mode="read")
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
        src_resolved, _ = _resolve_entity(source, mode="read")
        tgt_resolved, _ = _resolve_entity(target, mode="read")
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
        resolved, _ = _resolve_entity(entity, mode="read")
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


# ── Fact ranking (Fix A+C for #322) ─────────────────────────────────────────
# God-node threshold: entities with more than this many facts are treated as
# buckets of loosely-related debris. A query-token match is required to pull
# any of their facts. Tuned on Phase 1B benchmark: lloyd=2149, agent=181,
# system=103, so threshold ~50 filters out catch-all entities while letting
# normal entities (typical <20 facts) through untouched.
FACT_GODNODE_THRESHOLD = 50

# How many ranked facts to return from seed entities and from graph-expanded
# neighbors respectively. Previously 3 per entity (so 5 entities × 3 = 15
# unranked). Now a ranked pool capped at these totals.
FACT_RANK_CAP_SEED = 10
FACT_RANK_CAP_GRAPH = 5

# Stopwords for query tokenization — question words, auxiliaries, common
# function words. Kept lightweight; aggressive stopword removal hurts when
# the query itself is short ("how does fact_path work").
_FACT_QUERY_STOPWORDS = frozenset({
    "what", "how", "when", "where", "why", "who", "which",
    "the", "and", "for", "with", "from", "into", "over", "under",
    "does", "did", "do", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "can", "could", "will", "would", "should", "may",
    "might", "must", "shall",
    "this", "that", "these", "those", "it", "its", "them", "they", "their",
    "our", "ours", "my", "mine", "your", "yours", "his", "her", "hers",
    "we", "you", "us", "me", "i",
    "about", "also", "just", "than", "then", "there", "here", "through",
    "during", "between", "among", "across", "within", "without", "after",
    "before", "while", "until", "since",
    "of", "in", "on", "to", "at", "by", "as", "or", "nor", "not", "so",
    "if", "else", "but", "yet", "though", "although",
    "tell", "show", "explain", "describe", "walk", "give", "get", "use",
    "using", "used", "make", "made", "see", "seen", "know", "need",
})


def _fact_query_tokens(query: str) -> list[str]:
    """Extract scorable tokens from the query. Lowercased, length-3+,
    stopwords stripped. Preserves identifier-like tokens (with _, -, #)."""
    if not query:
        return []
    raw = re.findall(r"[A-Za-z0-9_#-]+", query.lower())
    return [t for t in raw if len(t) >= 3 and t not in _FACT_QUERY_STOPWORDS]


def _fact_blob(fact: dict) -> str:
    """Flatten a fact into searchable lowercased text. Mirrors the benchmark
    scorer in `_pipeline/memory-graph/run_benchmark.py::_fact_text` so our
    ranking is scoring the same field set that gets judged."""
    parts = []
    for key in ("fact", "category", "provenance", "event_date"):
        val = fact.get(key)
        if val:
            parts.append(str(val))
    return " ".join(parts).lower()


def _fact_matches_tokens(fact: dict, tokens: list[str]) -> bool:
    """True if any query token appears in the fact's searchable text."""
    if not tokens:
        return False
    blob = _fact_blob(fact)
    return any(t in blob for t in tokens)


def _fact_score(fact: dict, tokens: list[str]) -> float:
    """Fraction of query tokens that appear in the fact's searchable text.
    Returns 0.0 if tokens is empty (preserves confidence-tiebreak as the
    signal for contentless queries)."""
    if not tokens:
        return 0.0
    blob = _fact_blob(fact)
    hits = sum(1 for t in tokens if t in blob)
    return hits / len(tokens)


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
        # Daemon is the only search path. Subprocess fallback was removed
        # 2026-04-20: on daemon failure the CLI subprocess hits the same
        # broken state, then eats 30s (timeout=30 in _qmd_subprocess_search)
        # before returning empty. 4/20 benchmark queries were pinned at 30s
        # entirely due to this path. Daemon-down + return-empty is strictly
        # better than daemon-down + 30s-wait + empty.
        result = _qmd_daemon_search(query, limit, VAULT_SEGMENTS)
        return result or []

    def _do_facts():
        if not include_facts:
            return [], []
        # Query-aware fact ranking (Fix A, #322).
        # Old behavior: top-3-by-confidence per seed entity. Confidence is
        # ~1.0 for most STATED facts so ties resolved by insertion order,
        # returning effectively-random facts unrelated to the query.
        # New behavior: pull all facts from each seed entity, rank the whole
        # pool by keyword-overlap with the query, tie-break on confidence,
        # return top N overall. Matches the benchmark's substring-based
        # fact_hit_rate scoring.
        qtoks = _fact_query_tokens(query)

        def _collect(entity_names: list[str], godnode_threshold: int) -> list[dict]:
            """Pull facts from entities, applying god-node guardrail (Fix C)."""
            out: list[dict] = []
            for ent in entity_names:
                try:
                    entity_data = _get_facts_sync(ent)
                except Exception:
                    continue
                ef = entity_data.get("facts") or []
                if not ef:
                    continue
                # Fix C: god-nodes (>threshold facts) dump unrelated debris
                # into the pool. Require a query-token match to include any
                # of their facts; skip the entity if nothing matches.
                if len(ef) > godnode_threshold and qtoks:
                    kept = [f for f in ef if _fact_matches_tokens(f, qtoks)]
                    if not kept:
                        continue
                    ef = kept
                out.extend(ef)
            return out

        def _rank(candidates: list[dict], cap: int) -> list[dict]:
            if not candidates:
                return []
            scored = [
                (_fact_score(f, qtoks), float(f.get("confidence", 0.5)), idx, f)
                for idx, f in enumerate(candidates)
            ]
            # Sort: score desc, confidence desc, insertion idx asc (stable)
            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
            return [f for _s, _c, _i, f in scored[:cap]]

        seed_pool = _collect(seed_entities, FACT_GODNODE_THRESHOLD)
        facts = _rank(seed_pool, FACT_RANK_CAP_SEED)

        graph_facts: list[dict] = []
        if expand_graph and weighted_neighbors:
            neighbor_names = [e for e, _w in weighted_neighbors[:5]]
            graph_pool = _collect(neighbor_names, FACT_GODNODE_THRESHOLD)
            graph_facts = _rank(graph_pool, FACT_RANK_CAP_GRAPH)

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
        # default=str handles datetime event_date values that slipped
        # through YAML parsing as native objects (pre-existing data issue
        # in ~/obsidian/facts/, surfaced by #322 returning more facts).
        return json.dumps(result, default=str)
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

