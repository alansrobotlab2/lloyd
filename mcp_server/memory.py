#!/usr/bin/env python3
"""
Lloyd MCP Server: Memory — knowledge graph, facts, and vault tools.

Fact tools: fact_get, fact_add, fact_profile, fact_check, fact_resolve
Vault tools: vault_get, vault_write, vault_overview, vault_search, vault_recall

Facts data: ~/obsidian/memory/_pipeline/facts/
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
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

import yaml
from mcp.server import Server
from mcp.types import Tool, TextContent

# ── Constants ─────────────────────────────────────────────────────────────────

MEMORIES_ROOT = Path.home() / "lloyd" / "memories"
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
FACTS_ROOT = VAULT / "memory" / "_pipeline" / "facts"
AUDIT_LOG_DIR = VAULT / "memory" / "audit"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "writes.jsonl"
QMD_BIN = Path.home() / ".bun" / "bin" / "qmd"
QMD_DAEMON_URL = "http://localhost:8181/query"

VAULT_SEGMENTS = ["memory", "knowledge", "projects", "agents", "personal", "work", "skills"]
VAULT_EXCLUDE_DIRS = {"templates", "images", "_pipeline"}
VAULT_EXCLUDE_FILES = {"tags.md"}

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


def _get_facts_sync(entity: str, category: str = None) -> dict:
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


def _qmd_daemon_search(query: str, limit: int, collections: list) -> Optional[list]:
    query = _qmd_sanitize(query)
    if not query:
        return []
    payload = json.dumps({
        "searches": [{"type": "lex", "query": _qmd_strip_stopwords(query)}, {"type": "vec", "query": query}],
        "limit": limit, "collections": collections,
    }).encode()
    req = urllib.request.Request(QMD_DAEMON_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return [{"file": r.get("file", ""), "title": r.get("title", ""), "snippet": r.get("snippet", ""), "score": r.get("score", 0)} for r in data.get("results", [])]
    except Exception:
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
    try:
        return json.dumps(_get_facts_sync(entity, category))
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
        new_fact = {"fact": fact_text, "confidence": confidence, "category": category, "id": fact_id, "created_at": now_iso, "valid_at": params.get("valid_at"), "invalid_at": None, "expired_at": None, "source_doc": params.get("source_doc")}
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


def _vault_recall(params: dict) -> str:
    query = params.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required", "documents": [], "facts": []})
    limit = int(params.get("limit", 20))
    include_facts = params.get("include_facts", True)

    def _do_search():
        result = _qmd_daemon_search(query, limit, VAULT_SEGMENTS)
        if result is None:
            result = _qmd_subprocess_search(query, limit, VAULT_SEGMENTS)
        return result or []

    def _do_facts():
        if not include_facts:
            return []
        facts = []
        for entity, _ in _extract_entities_from_query(query)[:5]:
            try:
                entity_data = _get_facts_sync(entity)
                if entity_data.get("facts"):
                    top = sorted(entity_data["facts"], key=lambda f: f.get("confidence", 0.5), reverse=True)
                    facts.extend(top[:3])
            except Exception:
                pass
        return facts

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            search_fut = pool.submit(_do_search)
            facts_fut = pool.submit(_do_facts)
            raw_results = search_fut.result()
            facts = facts_fut.result()
        documents = []
        for r in raw_results[:limit]:
            path = r.get("file", "").removeprefix("qmd://")
            if path.startswith("obsidian/"):
                path = path.removeprefix("obsidian/")
            documents.append({"path": path, "title": r.get("title", ""), "snippet": r.get("snippet", ""), "score": r.get("score", 0)})
        return json.dumps({"documents": documents, "facts": facts, "query": query})
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


# ── MCP registration ─────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools():
    return [
        Tool(name="fact_get", description="Retrieve structured facts for a named entity.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "category": {"type": "string"}}, "required": ["entity"]}),
        Tool(name="fact_add", description="Add a structured fact for a named entity/category.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "category": {"type": "string"}, "fact": {"type": "string"}, "confidence": {"type": "number"}, "valid_at": {"type": "string"}, "source_doc": {"type": "string"}}, "required": ["entity", "category", "fact"]}),
        Tool(name="fact_profile", description="Get synthesized profile for an entity — all facts grouped by category.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}}, "required": ["entity"]}),
        Tool(name="fact_check", description="Detect contradictions in stored facts for an entity.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "category": {"type": "string"}}, "required": ["entity"]}),
        Tool(name="fact_resolve", description="Resolve contradictions by keeping higher-confidence fact.", inputSchema={
            "type": "object", "properties": {"entity": {"type": "string"}, "auto_resolve": {"type": "boolean"}}, "required": ["entity"]}),
        Tool(name="vault_get", description="Read a file from the obsidian vault by vault-relative path.", inputSchema={
            "type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "num_lines": {"type": "integer"}}, "required": ["path"]}),
        Tool(name="vault_write", description="Write content to a vault file. Audit-logged.", inputSchema={
            "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
        Tool(name="vault_overview", description="Get vault statistics: file counts per segment.", inputSchema={
            "type": "object", "properties": {"detail": {"type": "string", "enum": ["summary", "hubs"]}}}),
        Tool(name="vault_search", description="Hybrid BM25+vector search across the obsidian vault.", inputSchema={
            "type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}, "min_score": {"type": "number"}, "scope": {"type": "string"}, "consolidate": {"type": "boolean"}}, "required": ["query"]}),
        Tool(name="vault_recall", description="Combined recall: vault search + entity fact retrieval in parallel.", inputSchema={
            "type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "include_facts": {"type": "boolean"}}, "required": ["query"]}),
        Tool(name="memory_read", description="Read MEMORY.md or USER.md session memory files.", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"], "description": "Which file to read"}}, "required": []}),
        Tool(name="memory_add", description="Append an entry to MEMORY.md or USER.md.", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"]}, "entry": {"type": "string", "description": "Text to append"}}, "required": ["entry"]}),
        Tool(name="memory_replace", description="Replace text in MEMORY.md or USER.md (substring match).", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"]}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["old_text", "new_text"]}),
        Tool(name="memory_remove", description="Remove an entry from MEMORY.md or USER.md (substring match).", inputSchema={
            "type": "object", "properties": {"file": {"type": "string", "enum": ["MEMORY.md", "USER.md"]}, "entry": {"type": "string", "description": "Text to remove"}}, "required": ["entry"]}),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    handlers = {
        "fact_get": _fact_get, "fact_add": _fact_add, "fact_profile": _fact_profile,
        "fact_check": _fact_check, "fact_resolve": _fact_resolve,
        "vault_get": _vault_get, "vault_write": _vault_write, "vault_overview": _vault_overview,
        "vault_search": _vault_search, "vault_recall": _vault_recall,
        "memory_read": _memory_read, "memory_add": _memory_add,
        "memory_replace": _memory_replace, "memory_remove": _memory_remove,
    }
    handler = handlers.get(name)
    if handler:
        return [TextContent(type="text", text=handler(arguments))]
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

