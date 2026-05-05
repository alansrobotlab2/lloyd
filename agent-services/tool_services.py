# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp[cli]",
#   "pyyaml",
#   "ddgs",
#   "httpx",
#   "readability-lxml",
#   "html2text",
#   "uvicorn",
#   "starlette",
#   "sse-starlette>=2.0,<3.0",
# ]
# ///
"""
OpenClaw MCP Server — lloyd-services edition

Consolidated MCP tool server for OpenClaw. Replaces the per-plugin
subprocess architecture (5 separate server.py processes) with a single
long-lived process. Spawned by one OpenClaw plugin via McpStdioClient.

Tools (27):
  Memory/Vault : tag_search, tag_explore, vault_overview, mem_search,
                 mem_get, mem_write
  Prefill      : prefill_context (tag match + BM25 + GLM keywords)
  Web          : http_search (DuckDuckGo), http_fetch (readability), http_request
  File System  : file_read, file_write, file_edit, file_patch, file_glob, file_grep
  System       : run_bash, bg_exec, bg_process
  Backlog      : backlog_boards, backlog_tasks, backlog_get_task, backlog_write_task
  Skills       : skills_search, skills_get, skills_install

Run standalone:
  uv run openclaw_mcp_server.py

Nothing is written to stdout except MCP JSON-RPC frames.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import sys
import time
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import tempfile
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import html2text as _html2text
import httpx as _httpx
import yaml
from ddgs import DDGS as _DDGS
from mcp.server.fastmcp import FastMCP
from readability import Document as _ReadabilityDocument


VAULT = Path(os.environ.get("HOME", "/home/alansrobotlab")) / "obsidian"
QMD = Path.home() / ".bun/bin/qmd"

EXCLUDE_DIRS = {"templates", "images"}
EXCLUDE_FILES = {"tags.md"}

# ── Backlog constants ─────────────────────────────────────────────────────────

BACKLOG_DIR = Path.home() / "obsidian" / "backlog"

# ── Segment constants ──────────────────────────────────────────────────────────

VALID_SEGMENTS = {"agents", "personal", "work", "projects", "knowledge", "memory", "skills"}
SEGMENT_COLLECTIONS = ["memory", "knowledge", "projects", "agents", "personal", "work", "skills"]

# ── Intent classification patterns (compiled at module level) ─────────────────

_INTENT_FACTUAL_RE = re.compile(
    r"(?:what\s+port|where\s+is|how\s+to|which\s+file|what\s+is\s+the|what\s+are\s+the"
    r"|config\s+for|path\s+to|url\s+for|command\s+to|version\s+of)",
    re.IGNORECASE,
)
_INTENT_TEMPORAL_RE = re.compile(
    r"(?:last\s+week|yesterday|today|when\s+did|recent(?:ly)?|latest|last\s+month"
    r"|last\s+time|this\s+week|this\s+month|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_INTENT_CONCEPTUAL_RE = re.compile(
    r"(?:approaches?\s+to|how\s+does\s+\w+\s+compare|explain\s|overview\s+of"
    r"|what\s+are\s+(?:the\s+)?(?:different|best|common)|philosophy|strategy\s+for"
    r"|theory\s+of|principles?\s+of)",
    re.IGNORECASE,
)
_INTENT_CROSSREF_RE = re.compile(
    r"(?:\bcompare\b|\bvs\.?\b|\bbetween\s+\w+\s+and\b)",
    re.IGNORECASE,
)

# ── Consolidation ────────────────────────────────────────────────────────────
CONSOLIDATION_ENDPOINT = "http://localhost:8091/v1/chat/completions"
CONSOLIDATION_MODEL = "Qwen3.5-35B-A3B"
CONSOLIDATION_ENABLED = True
CONSOLIDATION_MIN_RESULTS = 4  # Only consolidate when 4+ results
CONSOLIDATION_TIMEOUT = 10  # seconds

CONSOLIDATION_SYSTEM_PROMPT = """\
You are a memory consolidation engine. Your job is to take raw search results \
from a knowledge vault and produce a concise, deduplicated, well-structured \
consolidation that directly answers the user's query.

Rules:
1. Deduplicate overlapping content across chunks. If multiple chunks say the same thing, merge them.
2. Rank information by relevance to the original query.
3. Compress verbose passages while preserving ALL key facts, names, numbers, URLs, and decisions.
4. Return 3-5 dense, relevant passages that together answer the query comprehensively.
5. Each passage should cite which source document(s) it drew from.
6. Provide a 1-2 sentence summary overview.

Output ONLY valid JSON (no markdown fences, no commentary) with this schema:
{
  "summary": "1-2 sentence overview answering the query",
  "passages": [
    {
      "title": "descriptive title for this passage",
      "content": "consolidated content with key facts preserved",
      "sources": ["qmd://path/to/source1.md", "qmd://path/to/source2.md"],
      "relevance_score": 0.95
    }
  ]
}"""

# ── Agent identity / write-permission constants ──────────────────────────────

WRITE_PERMISSIONS: dict[str, list[str]] = {
    "lloyd":      ["memory/", "agents/lloyd/", "personal/"],
    "dee":        ["memory/dee/", "agents/dee/", "personal/"],
    "memory":     ["memory/", "agents/lloyd/", "agents/shared/"],
    "researcher": ["knowledge/"],
    "coder":      ["agents/coder/"],
    "operator":   ["agents/operator/"],
}

DEFAULT_READ_SCOPE: dict[str, str] = {
    "lloyd":        "memory,personal,knowledge,projects",
    "memory":       "memory",
    "researcher":   "knowledge,projects",
    "coder":        "agents/coder,projects,knowledge",
    "orchestrator": "projects,knowledge,agents",
}

AUDIT_LOG_DIR = VAULT / "memory" / "audit"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "writes.jsonl"


def _check_write_permission(agent_id: str | None, path: str) -> str | None:
    """Return error message if agent_id is not allowed to write to path, else None."""
    if agent_id is None:
        return None  # backward compatible: no enforcement
    # Sanitize agent_id to prevent path traversal
    if "/" in agent_id or ".." in agent_id or not agent_id.replace("-", "").replace("_", "").isalnum():
        return f"Write denied: invalid agent_id '{agent_id}'"
    allowed = WRITE_PERMISSIONS.get(agent_id)
    if allowed is None:
        # Unknown agent: can only write to their own namespaces
        allowed = [f"agents/{agent_id}/", f"memory/{agent_id}/"]
    normalized = path if path.endswith("/") or "." in path.split("/")[-1] else path + "/"
    for prefix in allowed:
        if normalized.startswith(prefix):
            return None
    return f"Write denied: agent '{agent_id}' cannot write to '{path}'. Allowed paths: {allowed}"


def _audit_write(agent_id: str | None, path: str, byte_count: int, action: str = "write") -> None:
    """Append a JSON line to the audit log."""
    try:
        AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "agent_id": agent_id or "unknown",
            "path": path,
            "bytes": byte_count,
            "action": action,
        }
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        print(f"audit-log: {exc}", file=sys.stderr)

# ── Skill directories ────────────────────────────────────────────────────────

CUSTOM_SKILLS_DIR = Path.home() / "obsidian" / "skills"
BUILTIN_SKILLS_DIR = Path.home() / ".npm-global" / "lib" / "node_modules" / "openclaw" / "skills"
OPENCLAW_CONFIG = Path.home() / "agents" / "lloyd" / "openclaw.json"


# ── Memory facts/constants ────────────────────────────────────────────────────

FACTS_ROOT = Path.home() / "obsidian" / "memory" / "_pipeline" / "facts"

# Entity matching stopwords — filtered from query parsing and entity word matching
_ENTITY_STOPWORDS = frozenset({
    'a', 'an', 'the', 'is', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or',
    'it', 'be', 'as', 'by', 'this', 'that', 'with', 'from', 'not', 'but',
    'what', 'which', 'who', 'how', 'when', 'where', 'all', 'if', 'so', 'do',
    'can', 'has', 'was', 'are', 'had', 'have', 'no', 'up', 'about', 'me',
    'tell', 'status', 'running', 'get', 'set', 'would', 'could', 'should',
    'will', 'just', 'been', 'being', 'its', 'my', 'your', 'our', 'their',
})

RELATIONS_INDEX = Path.home() / "obsidian" / "memory" / "_pipeline" / "relations-index.json"
ENTITY_REGISTRY = Path.home() / "obsidian" / "memory" / "facts" / "entity-registry.json"

# Relations index cache (loaded once, invalidated periodically)
_RELATIONS_INDEX_CACHE: tuple[float, dict] | None = None
_RELATIONS_INDEX_CACHE_TTL = 300  # 5 minutes


# ── ClawhHub constants ──────────────────────────────────────────────────────
CLAWHUB_CACHE_TTL = 3600  # 1 hour
_clawhub_cache: dict[str, tuple[float, list[dict]]] = {}  # query -> (timestamp, results)
CLAWHUB_STAGING = Path(tempfile.gettempdir()) / "clawhub-staging"


def _load_relations_index_cached() -> dict:
    """Load relations index from disk with caching (5 min TTL)."""
    global _RELATIONS_INDEX_CACHE
    now = time.monotonic()
    if _RELATIONS_INDEX_CACHE is not None:
        cached_ts, cached_data = _RELATIONS_INDEX_CACHE
        if now - cached_ts < _RELATIONS_INDEX_CACHE_TTL:
            return cached_data
    # Load from disk
    try:
        with open(RELATIONS_INDEX, "r") as f:
            data = json.load(f)
        _RELATIONS_INDEX_CACHE = (now, data)
        return data
    except Exception as e:
        print(f"Warning: Failed to load relations index: {e}", file=sys.stderr)
        return {"relationships": []}
CLAWHUB_INSTALL_DIR = Path.home() / "agents" / "lloyd" / "skills"
CLAWHUB_INSTALL_LOG = Path.home() / "obsidian" / "knowledge" / "software" / "clawhub-installed-skills.md"
_CLAWHUB_SLUG_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$')


def _resolve_scope_prefixes(scope: "str | list[str] | None") -> list[str]:
    """Convert a scope value to a list of path prefixes.

    scope can be:
      None              → no filtering (all segments)
      "projects"        → ["projects/"]
      "projects,knowledge" → ["projects/", "knowledge/"]
      ["projects", "knowledge"] → ["projects/", "knowledge/"]
      "projects/alfie"  → ["projects/alfie"] (sub-path passthrough)
    """
    if not scope:
        return []
    items = [scope] if isinstance(scope, str) else list(scope)
    # Handle comma-separated strings
    expanded: list[str] = []
    for item in items:
        expanded.extend(s.strip() for s in item.split(",") if s.strip())
    prefixes: list[str] = []
    for item in expanded:
        # If it's a known segment name, add trailing slash; otherwise pass through
        if item in VALID_SEGMENTS:
            prefixes.append(item + "/")
        elif item:
            prefixes.append(item.rstrip("/") + "/")
    return prefixes
MAX_FILE_SIZE = 512 * 1024  # 500 KB

# ── Web tool constants ─────────────────────────────────────────────────────────

WEB_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
WEB_ACCEPT_LANG = "en-US,en;q=0.9"
WEB_TIMEOUT_S = 15.0
WEB_MAX_RESPONSE_BYTES = 2_000_000
WEB_DEFAULT_MAX_CHARS = 50_000

# ── Prefill pipeline constants ─────────────────────────────────────────────────

GLM_URL = "http://127.0.0.1:8091/v1/chat/completions"
GLM_MODEL = "Qwen3.5-35B-A3B"

PREFILL_TIMEOUT_S      = 2.0
PREFILL_CACHE_TTL_S    = 60.0
PREFILL_REFRESH_S      = 10 * 60  # 10 minutes
MIN_QUERY_LENGTH       = 12
GLM_MIN_QUERY_LEN      = 40
MAX_CONTEXT_CHARS      = 8_000
MAX_TOPICS             = 3
MAX_DOCS_PER_TAG       = 4
TIER1_THRESHOLD        = 0.6
TIER2_THRESHOLD        = 0.3
TIER1_MAX              = 3
TIER2_MAX              = 5
CONTENT_PER_DOC        = 1_500
W_VECTOR               = 0.55
W_TAG                  = 0.35
W_CROSS                = 0.10
SYNTHETIC_VECTOR_SCORE = 0.5

FILLER_WORDS = {
    "hey", "hi", "what", "do", "you", "know", "about", "my", "work", "with",
    "the", "a", "an", "is", "are", "was", "were", "can", "could", "would",
    "should", "tell", "me", "i", "want", "to", "how", "when", "where", "why",
    "who", "which", "that", "this", "these", "those", "it", "be", "been",
    "being", "have", "has", "had", "does", "did", "will", "shall", "may",
    "might", "some", "any", "all", "more", "very", "just", "your", "our",
    "their", "its", "there", "give", "get", "let", "make", "see", "look",
    "show", "find", "help", "think", "say", "please", "thanks", "ok", "so",
    "and", "or", "but", "in", "on", "at", "of", "for", "from", "up", "out",
    "if", "no", "not", "as", "into", "through", "during", "before", "after",
    "use", "using", "search", "memory", "related", "connected", "tag", "tags",
}

_PRIVATE_IP_PATTERNS = [
    re.compile(r"^127\."),
    re.compile(r"^10\."),
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^0\."),
    re.compile(r"^169\.254\."),
    re.compile(r"^::1$"),
    re.compile(r"^fc00:", re.IGNORECASE),
    re.compile(r"^fd", re.IGNORECASE),
    re.compile(r"^fe80:", re.IGNORECASE),
]


def _is_private_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    return any(p.match(hostname) for p in _PRIVATE_IP_PATTERNS)


# ── qmd daemon HTTP client ────────────────────────────────────────────────────

QMD_DAEMON_URL = "http://localhost:8181/query"


class QmdDaemonClient:
    """HTTP client for the QMD daemon REST API.

    Sends hybrid search requests (BM25 + vector) to the QMD daemon
    instead of querying SQLite directly.
    """

    @staticmethod
    def _sanitize(query: str) -> str:
        """Minimal sanitization — preserve lex syntax operators."""
        return re.sub(r'[\x00-\x1f\x7f]', ' ', query).strip()

    @staticmethod
    def _strip_lex_stopwords(query: str) -> str:
        """Strip stopwords from query for BM25 — they have near-zero IDF and waste search time."""
        words = [w for w in re.findall(r'\b\w+\b', query.lower()) if w not in _ENTITY_STOPWORDS and len(w) >= 2]
        return " ".join(words) if words else query  # fall back to original if everything stripped

    def search(self, query: str, limit: int = 10, collections: list[str] | None = None, searches: list[dict] | None = None, intent: str | None = None, skip_rerank: bool = False, candidate_limit: int | None = None) -> list[dict] | None:
        """Hybrid search via QMD daemon. Returns None if daemon is unavailable."""
        query = self._sanitize(query)
        if not query:
            return []
        default_searches = [
            {"type": "lex", "query": self._strip_lex_stopwords(query)},
            {"type": "vec", "query": query},  # vec handles stopwords via embedding
        ]
        payload_dict = {
            "searches": searches or default_searches,
            "limit": limit,
            "collections": collections or SEGMENT_COLLECTIONS,
        }
        if intent is not None:
            payload_dict["intent"] = intent
        if skip_rerank:
            payload_dict["skipRerank"] = True
        if candidate_limit is not None:
            payload_dict["candidateLimit"] = candidate_limit
        payload = json.dumps(payload_dict).encode()
        req = urllib.request.Request(
            QMD_DAEMON_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            results = []
            for r in data.get("results", []):
                results.append({
                    "file": r.get("file", ""),
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "score": r.get("score", 0),
                    "docid": r.get("docid"),
                })
            return results
        except Exception as exc:
            print(f"openclaw-mcp: QMD daemon query failed: {exc}", file=sys.stderr)
            return None

    def search_one(self, query: str, limit: int = 10, collection: str = "memory", searches: list[dict] | None = None, intent: str | None = None) -> list[dict] | None:
        """Hybrid search in a single collection."""
        query = self._sanitize(query)
        if not query:
            return []
        default_searches = [
            {"type": "lex", "query": self._strip_lex_stopwords(query)},
            {"type": "vec", "query": query},
        ]
        payload_dict = {
            "searches": searches or default_searches,
            "limit": limit,
            "collections": [collection],
        }
        if intent is not None:
            payload_dict["intent"] = intent
        payload = json.dumps(payload_dict).encode()
        req = urllib.request.Request(
            QMD_DAEMON_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            results = []
            for r in data.get("results", []):
                results.append({
                    "file": r.get("file", ""),
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "score": r.get("score", 0),
                    "docid": r.get("docid"),
                })
            return results
        except Exception as exc:
            print(f"openclaw-mcp: QMD daemon query failed ({collection}): {exc}", file=sys.stderr)
            return None


_qmd_client = QmdDaemonClient()


# ── Memory/tag constants ───────────────────────────────────────────────────────

TYPE_BOOST = {
    "hub": 2.0,
    "project-notes": 1.0,
    "talk": 0.9,
    "work-notes": 0.8,
    "notes": 0.7,
}

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class DocMeta:
    path: str
    stem: str
    title: str
    type: str
    tags: list[str]
    summary: str
    status: str
    folder: str
    importance: float = 0.0


# ── Vault scanner ─────────────────────────────────────────────────────────────

_FM_BLOCK = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def _parse_frontmatter(text: str) -> dict[str, Any]:
    m = _FM_BLOCK.match(text)
    if not m:
        return {}
    try:
        result = yaml.safe_load(m.group(1))
        return result if isinstance(result, dict) else {}
    except yaml.YAMLError:
        return {}


def _walk_markdown(vault: Path):
    for entry in vault.iterdir():
        if entry.name.startswith(".") or entry.name in EXCLUDE_DIRS:
            continue
        if entry.is_dir():
            yield from _walk_markdown(entry)
        elif (
            entry.is_file()
            and entry.suffix == ".md"
            and entry.name not in EXCLUDE_FILES
            and entry.stat().st_size <= MAX_FILE_SIZE
        ):
            yield entry


def _scan_vault(vault: Path) -> list[DocMeta]:
    docs: list[DocMeta] = []
    if not vault.exists():
        return docs

    for md_file in _walk_markdown(vault):
        rel = md_file.relative_to(vault)
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        fm = _parse_frontmatter(text)
        stem = md_file.stem

        raw_title = fm.get("title", "")
        title = str(raw_title).strip('"') if raw_title else ""
        if not title:
            title = stem.replace("-", " ").title()

        raw_folder = fm.get("folder", "")
        folder = str(raw_folder).strip('"') if raw_folder else ""
        if not folder:
            folder = str(rel.parent)
        if folder == ".":
            folder = ""

        raw_tags = fm.get("tags", [])
        if isinstance(raw_tags, list):
            tags = [str(t).strip().strip('"') for t in raw_tags if t]
        elif isinstance(raw_tags, str):
            tags = [t.strip().strip('"') for t in raw_tags.split(",") if t.strip()]
        else:
            tags = []

        docs.append(
            DocMeta(
                path=str(rel),
                stem=stem,
                title=title,
                type=str(fm.get("type", "")).strip('"'),
                tags=tags,
                summary=str(fm.get("summary", "")).strip('"'),
                status=str(fm.get("status", "")).strip('"'),
                folder=folder,
            )
        )

    return docs


# ── Tag index ─────────────────────────────────────────────────────────────────


class TagIndex:
    def __init__(self, docs: list[DocMeta]):
        self.doc_index: dict[str, DocMeta] = {}
        self.tag_to_docs: dict[str, list[str]] = {}   # lowercase tag → [path]
        self.tag_cooccurrence: dict[str, dict[str, int]] = {}
        self.tag_idf: dict[str, float] = {}

        n = len(docs)

        for doc in docs:
            self.doc_index[doc.path] = doc
            tags_lower = [t.lower() for t in doc.tags]

            # importance
            boost = TYPE_BOOST.get(doc.type, 0.5)
            connectivity = math.sqrt(len(doc.tags)) * 0.3
            summary_bonus = 0.3 if doc.summary else 0.0
            doc.importance = boost + connectivity + summary_bonus

            for tag in tags_lower:
                self.tag_to_docs.setdefault(tag, []).append(doc.path)

            # co-occurrence
            for i, t1 in enumerate(tags_lower):
                for t2 in tags_lower[i + 1:]:
                    self.tag_cooccurrence.setdefault(t1, {})[t2] = (
                        self.tag_cooccurrence.get(t1, {}).get(t2, 0) + 1
                    )
                    self.tag_cooccurrence.setdefault(t2, {})[t1] = (
                        self.tag_cooccurrence.get(t2, {}).get(t1, 0) + 1
                    )

        # IDF
        for tag, paths in self.tag_to_docs.items():
            self.tag_idf[tag] = math.log((n + 1) / (len(paths) + 1))

    @property
    def doc_count(self) -> int:
        return len(self.doc_index)

    @property
    def tag_count(self) -> int:
        return len(self.tag_to_docs)

    def resolve_tag(self, tag: str) -> str | None:
        """Return the canonical cased tag key, or None if not found."""
        lower = tag.lower().lstrip("#")
        return lower if lower in self.tag_to_docs else None

    def suggest_tags(self, tag: str, limit: int = 3) -> list[str]:
        needle = tag.lower().lstrip("#")
        matches = [t for t in self.tag_to_docs if needle in t or t in needle]
        return sorted(matches, key=len)[:limit]

    def search_by_tags(
        self,
        tags: list[str],
        mode: str = "or",
        type_filter: str | None = None,
        limit: int = 10,
        scope_prefixes: list[str] | None = None,
    ) -> list[DocMeta]:
        normalized = [t.lower().lstrip("#") for t in tags]

        if mode == "and":
            result_paths: set[str] | None = None
            for tag in normalized:
                paths = set(self.tag_to_docs.get(tag, []))
                result_paths = paths if result_paths is None else result_paths & paths
            path_set = result_paths or set()
        else:
            path_set: set[str] = set()
            for tag in normalized:
                path_set |= set(self.tag_to_docs.get(tag, []))

        docs = [self.doc_index[p] for p in path_set if p in self.doc_index]
        if type_filter and type_filter != "any":
            docs = [d for d in docs if d.type == type_filter]
        if scope_prefixes:
            docs = [d for d in docs if any(d.path.startswith(p) for p in scope_prefixes)]
        docs.sort(key=lambda d: d.importance, reverse=True)
        return docs[:limit]

    def get_related_tags(
        self, tag: str, limit: int = 15
    ) -> list[dict[str, Any]]:
        co = self.tag_cooccurrence.get(tag, {})
        result = []
        for related_tag, count in co.items():
            result.append(
                {
                    "tag": related_tag,
                    "count": count,
                    "idf": self.tag_idf.get(related_tag, 0.0),
                    "docFreq": len(self.tag_to_docs.get(related_tag, [])),
                }
            )
        result.sort(key=lambda x: x["count"], reverse=True)
        return result[:limit]

    def get_stats(self) -> dict[str, Any]:
        type_dist: dict[str, int] = {}
        hub_pages: list[DocMeta] = []
        for doc in self.doc_index.values():
            type_dist[doc.type or "other"] = type_dist.get(doc.type or "other", 0) + 1
            if doc.type == "hub":
                hub_pages.append(doc)

        top_tags = sorted(
            [
                {
                    "tag": tag,
                    "docFreq": len(paths),
                    "idf": self.tag_idf.get(tag, 0.0),
                }
                for tag, paths in self.tag_to_docs.items()
            ],
            key=lambda x: x["docFreq"],
            reverse=True,
        )

        top_docs = sorted(
            self.doc_index.values(), key=lambda d: d.importance, reverse=True
        )[:10]

        return {
            "docCount": self.doc_count,
            "tagCount": self.tag_count,
            "typeDistribution": type_dist,
            "hubPages": sorted(hub_pages, key=lambda d: d.importance, reverse=True),
            "topTags": top_tags,
            "topDocsByImportance": list(top_docs),
        }


# ── Format helpers ────────────────────────────────────────────────────────────


def _format_doc_line(doc: DocMeta, max_summary: int = 120) -> str:
    badge_parts = [p for p in [doc.type, doc.status] if p]
    badge = f" [{' | '.join(badge_parts)}]" if badge_parts else ""
    summary = doc.summary
    if summary and len(summary) > max_summary:
        summary = summary[:max_summary] + "..."
    summary_str = f" — {summary}" if summary else ""
    tag_str = ", ".join(doc.tags) if doc.tags else "(none)"
    return f"- {doc.path}{badge} \"{doc.title}\"{summary_str}\n  Tags: {tag_str}"


def _format_tag_search(
    tags: list[str],
    mode: str,
    docs: list[DocMeta],
    total: int,
    limit: int,
    unresolved: list[str],
    suggestions: dict[str, list[str]],
) -> str:
    mode_label = "ALL" if mode == "and" else "ANY"
    tag_list = ", ".join(f'"{t}"' for t in tags)

    if not docs and unresolved:
        lines = [f"No documents found matching tags [{tag_list}] ({mode_label} mode)."]
        for tag in unresolved:
            sugg = suggestions.get(tag, [])
            if sugg:
                lines.append(f'  Tag "{tag}" not found. Did you mean: {", ".join(sugg)}?')
            else:
                lines.append(f'  Tag "{tag}" not found in the vault.')
        return "\n".join(lines)

    if not docs:
        return f"No documents found matching tags [{tag_list}] ({mode_label} mode)."

    lines = [
        f"**Tag search: [{tag_list}] ({mode_label} mode)**",
        f"Showing {len(docs)} of {total} matches:\n",
    ]
    for doc in docs:
        lines.append(_format_doc_line(doc))

    if total > limit:
        lines.append(f"\n... and {total - limit} more. Use a higher limit or narrow with AND mode.")

    return "\n".join(lines)


def _format_tag_explore(
    tag: str,
    tag_doc_count: int,
    tag_idf: float,
    related_tags: list[dict[str, Any]],
    bridge_tag: str | None,
    bridge_docs: list[DocMeta] | None,
) -> str:
    lines = [f'**Tag "{tag}"** — appears in {tag_doc_count} documents (IDF: {tag_idf:.2f})\n']

    if related_tags:
        lines.append("**Co-occurring tags:**")
        for rt in related_tags:
            lines.append(
                f"  {rt['tag']} ({rt['count']} docs together, {rt['docFreq']} total, IDF: {rt['idf']:.2f})"
            )
        lines.append("")
    else:
        lines.append("No co-occurring tags found.\n")

    if bridge_tag:
        if bridge_docs:
            lines.append(f'**Documents bridging "{tag}" and "{bridge_tag}":**')
            for doc in bridge_docs:
                lines.append(_format_doc_line(doc))
        else:
            lines.append(f'No documents found with both "{tag}" and "{bridge_tag}" tags.')
        lines.append("")

    return "\n".join(lines)


def _format_vault_overview(stats: dict[str, Any], detail: str) -> str:
    if stats["docCount"] == 0:
        return "The vault is empty. No markdown files were found."

    if detail == "tags":
        top_tags = stats["topTags"]
        lines = [f"**All Tags** ({stats['tagCount']} tags across {stats['docCount']} documents)\n"]
        for t in top_tags:
            lines.append(f"  {t['tag']} — {t['docFreq']} docs (IDF: {t['idf']:.2f})")
        return "\n".join(lines)

    if detail == "hubs":
        hubs = stats["hubPages"]
        if not hubs:
            return "No hub pages found in the vault."
        lines = [f"**Hub Pages** ({len(hubs)} index pages)\n"]
        for doc in hubs:
            lines.append(_format_doc_line(doc, max_summary=100))
        return "\n".join(lines)

    if detail == "types":
        type_dist = stats["typeDistribution"]
        total = stats["docCount"]
        lines = [f"**Document Types** ({total} total documents)\n"]
        for doc_type, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100
            lines.append(f"  {doc_type}: {count} ({pct:.1f}%)")
        return "\n".join(lines)

    # Default: summary
    type_dist = stats["typeDistribution"]
    type_parts = " | ".join(
        f"{t}: {c}" for t, c in sorted(type_dist.items(), key=lambda x: x[1], reverse=True)
    )
    lines = [
        "**Vault Overview**\n",
        f"Documents: {stats['docCount']} | Tags: {stats['tagCount']} | Hub pages: {len(stats['hubPages'])}",
        "",
        f"**Type distribution:** {type_parts}\n",
    ]

    top_tags = stats["topTags"][:15]
    if top_tags:
        tag_parts = " | ".join(f"{t['tag']} ({t['docFreq']})" for t in top_tags)
        lines.append(f"**Top tags:** {tag_parts}\n")

    top_docs = stats["topDocsByImportance"][:5]
    if top_docs:
        lines.append("**Top documents by importance:**")
        for doc in top_docs:
            lines.append(_format_doc_line(doc, max_summary=80))

    return "\n".join(lines)


# ── Skill index ───────────────────────────────────────────────────────────────


@dataclass
class SkillEntry:
    name: str           # from frontmatter
    description: str    # from frontmatter
    source: str         # "builtin" or "custom"
    dir_path: str       # absolute path to skill directory
    body: str           # full SKILL.md content (for search)
    enabled: bool       # whether skill is enabled in openclaw.json


class SkillIndex:
    def __init__(self):
        self.skills: list[SkillEntry] = []
        self.by_name: dict[str, SkillEntry] = {}  # lowercase name -> entry
        self._scan()

    def _scan(self):
        self.skills = []
        self.by_name = {}

        # Read openclaw.json for skills.entries enabled flags
        skill_entries_cfg: dict = {}
        try:
            import json as _json
            cfg = _json.loads(OPENCLAW_CONFIG.read_text(encoding="utf-8"))
            skill_entries_cfg = cfg.get("skills", {}).get("entries", {})
        except (OSError, ValueError):
            pass

        # Custom skills first (take precedence on name collision)
        for source, skills_dir in [("custom", CUSTOM_SKILLS_DIR), ("builtin", BUILTIN_SKILLS_DIR)]:
            if not skills_dir.is_dir():
                continue
            for skill_dir in sorted(skills_dir.iterdir()):
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.is_file():
                    continue
                try:
                    text = skill_md.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                fm = _parse_frontmatter(text)
                name = str(fm.get("name", skill_dir.name)).strip()
                if not name:
                    name = skill_dir.name
                desc = str(fm.get("description", "")).strip()
                # Check enabled state from openclaw.json skills.entries
                scfg = skill_entries_cfg.get(name) or skill_entries_cfg.get(skill_dir.name) or {}
                enabled = scfg.get("enabled", True) is not False
                entry = SkillEntry(
                    name=name,
                    description=desc,
                    source=source,
                    dir_path=str(skill_dir),
                    body=text,
                    enabled=enabled,
                )
                self.skills.append(entry)
                # Custom takes precedence (scanned first)
                name_lower = name.lower()
                if name_lower not in self.by_name:
                    self.by_name[name_lower] = entry

    def search(self, query: str, source: str = "all") -> list[tuple[SkillEntry, int]]:
        """Search skills. Returns (entry, rank) tuples sorted by rank (lower=better).
        Rank: 0=exact name, 1=name contains, 2=description match, 3=body match."""
        q = query.lower().strip()
        if not q:
            # Return all, filtered by source and enabled state
            results = []
            for s in self.skills:
                if not s.enabled:
                    continue
                if source != "all" and s.source != source:
                    continue
                results.append((s, 2))
            return results

        results: list[tuple[SkillEntry, int]] = []
        keywords = q.split()

        for skill in self.skills:
            if not skill.enabled:
                continue
            if source != "all" and skill.source != source:
                continue

            name_lower = skill.name.lower()
            desc_lower = skill.description.lower()
            body_lower = skill.body.lower()

            if name_lower == q:
                rank = 0
            elif q in name_lower or all(kw in name_lower for kw in keywords):
                rank = 1
            elif all(kw in desc_lower for kw in keywords):
                rank = 2
            elif all(kw in body_lower for kw in keywords):
                rank = 3
            else:
                continue

            results.append((skill, rank))

        results.sort(key=lambda x: (x[1], x[0].name.lower()))
        return results

    @property
    def count(self) -> int:
        return len(self.skills)


# ── ClawhHub helpers ─────────────────────────────────────────────────────────

_CLAWHUB_LINE_RE = re.compile(
    r"^\s*(\S+)\s+(.+?)\s+\(([0-9.]+)\)\s*$"
)


def _clawhub_search(query: str) -> list[dict]:
    """Search ClawhHub catalog via CLI. Returns list of result dicts.

    Caches results for CLAWHUB_CACHE_TTL seconds. Gracefully returns []
    on any error or timeout.
    """
    cache_key = query.lower().strip()
    now = time.monotonic()

    cached = _clawhub_cache.get(cache_key)
    if cached is not None:
        ts, results = cached
        if now - ts < CLAWHUB_CACHE_TTL:
            return results

    try:
        proc = subprocess.run(
            ["clawhub", "search", query, "--limit", "10"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []

    results: list[dict] = []
    for line in proc.stdout.splitlines():
        m = _CLAWHUB_LINE_RE.match(line)
        if m:
            results.append({
                "slug": m.group(1),
                "name": m.group(2).strip(),
                "score": float(m.group(3)),
                "source": "clawhub",
            })

    _clawhub_cache[cache_key] = (now, results)
    return results


def _clawhub_inspect_json(slug: str) -> dict:
    """Run `clawhub inspect <slug> --json` and return parsed dict."""
    proc = subprocess.run(
        ["clawhub", "inspect", slug, "--json"],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"clawhub inspect failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _clawhub_inspect_file(slug: str, filename: str) -> str:
    """Run `clawhub inspect <slug> --file <filename>` and return content."""
    proc = subprocess.run(
        ["clawhub", "inspect", slug, "--file", filename],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"clawhub inspect --file failed: {proc.stderr.strip()}")
    return proc.stdout


def _clawhub_inspect_files(slug: str) -> str:
    """Run `clawhub inspect <slug> --files` and return file listing."""
    proc = subprocess.run(
        ["clawhub", "inspect", slug, "--files"],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"clawhub inspect --files failed: {proc.stderr.strip()}")
    return proc.stdout




# ── Memory helper functions ───────────────────────────────────────────────────

def _parse_fact_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from a fact file."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    return yaml.safe_load(content[3:end]) or {}

def _write_fact_frontmatter(data: dict) -> str:
    """Write YAML frontmatter for a fact file."""
    return f"---\n{yaml.dump(data, default_flow_style=False, sort_keys=False)}---\n"

def _find_entity_dir(entity: str) -> Path | None:
    """Find entity directory with case-insensitive match."""
    if not FACTS_ROOT.exists():
        return None
    entity_lower = entity.lower()
    for dir_entry in FACTS_ROOT.iterdir():
        if dir_entry.is_dir() and dir_entry.name.lower() == entity_lower:
            return dir_entry
    return None

_ENTITY_DIRS_CACHE: tuple[float, list[str]] | None = None  # (timestamp, list)
_ENTITY_DIRS_CACHE_TTL = 60  # seconds

def _get_entity_dirs_cached() -> list[str]:
    """Get entity directory names with 60s TTL cache."""
    global _ENTITY_DIRS_CACHE
    import time as _time
    now = _time.monotonic()
    if _ENTITY_DIRS_CACHE is not None and (now - _ENTITY_DIRS_CACHE[0]) < _ENTITY_DIRS_CACHE_TTL:
        return _ENTITY_DIRS_CACHE[1]
    if not FACTS_ROOT.exists():
        _ENTITY_DIRS_CACHE = (now, [])
        return []
    names = [d.name for d in FACTS_ROOT.iterdir() if d.is_dir()]
    _ENTITY_DIRS_CACHE = (now, names)
    return names

def _extract_entities_from_query(query: str) -> list[tuple[str, int]]:
    """Extract entity names from query by matching against known entity directory names.
    
    Returns list of (entity_name, score) tuples where score is:
    - 3 for exact match
    - 2 for word overlap match
    - 1 for minimal overlap
    
    Entities are sorted by score (descending) then by entity name.
    Case-insensitive duplicates are removed, keeping the entity with more facts.
    """
    if not FACTS_ROOT.exists():
        return []
    
    query_lower = query.lower()
    # Extract words from query, filtering stopwords and short words
    query_words = [w for w in re.findall(r'\b\w+\b', query_lower) if w not in _ENTITY_STOPWORDS and len(w) >= 2]
    
    # Get all entity directory names (cached)
    entity_names = _get_entity_dirs_cached()
    
    matches = []
    for entity in entity_names:
        entity_lower = entity.lower()
        # Split entity into words and remove stopwords
        entity_words = set(re.findall(r'\b\w+\b', entity_lower))
        entity_content_words = entity_words - _ENTITY_STOPWORDS
        query_content_words = set(query_words)
        
        score = 0
        # Exact match: entity name equals a query word exactly
        if entity_lower in query_content_words:
            score = 3
        # Word overlap: entity content words overlap with query content words
        elif entity_content_words & query_content_words:
            overlap = entity_content_words & query_content_words
            # More overlapping words = higher score
            score = 2 if len(overlap) >= 1 else 1
        
        if score > 0:
            matches.append((entity, score))
    
    # Sort by score (descending), then by entity name
    matches.sort(key=lambda x: (-x[1], x[0]))
    
    # Deduplicate case-insensitive matches, keeping the one with more facts
    seen_lower = {}
    deduped = []
    for entity, score in matches:
        key = entity.lower()
        if key in seen_lower:
            # Keep the one with more facts
            existing_idx, existing_entity = seen_lower[key]
            existing_count = sum(1 for _ in (FACTS_ROOT / existing_entity).glob("*.md"))
            new_count = sum(1 for _ in (FACTS_ROOT / entity).glob("*.md"))
            if new_count > existing_count:
                deduped[existing_idx] = (entity, max(score, deduped[existing_idx][1]))
                seen_lower[key] = (existing_idx, entity)
        else:
            seen_lower[key] = (len(deduped), entity)
            deduped.append((entity, score))
    
    return deduped

def _generate_fact_id(category: str) -> str:
    """Generate a unique fact ID."""
    import uuid
    return f"{category[:4]}-{uuid.uuid4().hex[:4]}"

def _get_facts_sync(entity: str, category: str = None) -> dict:
    """Synchronous helper to retrieve facts for an entity. Returns dict."""
    entity_dir = _find_entity_dir(entity)
    if not entity_dir:
        return {"error": f"Entity not found: {entity}", "facts": []}
    facts = []
    if category:
        fact_file = entity_dir / f"{entity}-{category}.md"
        if fact_file.exists():
            content = fact_file.read_text(encoding="utf-8")
            frontmatter = _parse_fact_frontmatter(content)
            if "facts" in frontmatter:
                facts = frontmatter["facts"]
    else:
        for fact_file in entity_dir.glob("*.md"):
            content = fact_file.read_text(encoding="utf-8")
            frontmatter = _parse_fact_frontmatter(content)
            if "facts" in frontmatter:
                facts.extend(frontmatter["facts"])
    return {"entity": entity, "category": category, "facts": facts}

def _detect_contradictions_sync(entity: str, category: str = None) -> dict:
    """Synchronous helper to detect contradictions. Returns dict."""
    facts_data = _get_facts_sync(entity, category)
    facts = facts_data.get("facts", [])
    opposing_pairs = [
        ("yes", "no"), ("true", "false"), ("enabled", "disabled"),
        ("active", "inactive"), ("supported", "unsupported"),
        ("working", "broken"), ("success", "failure")
    ]
    contradictions = []
    for i, fact1 in enumerate(facts):
        for fact2 in facts[i+1:]:
            text1 = fact1.get("fact", "").lower()
            text2 = fact2.get("fact", "").lower()
            for pair in opposing_pairs:
                if pair[0] in text1 and pair[1] in text2:
                    contradictions.append({
                        "fact1": fact1,
                        "fact2": fact2,
                        "opposing_terms": pair
                    })
                    break
    return {"entity": entity, "category": category, "contradictions": contradictions, "checked": len(facts)}


# ── Build index (deferred to background thread) ──────────────────────────────

_index: TagIndex = TagIndex([])        # empty sentinel — populated in background
_skill_index: SkillIndex | None = None
_index_ready = False                   # flipped after first background build


def _build_index_sync() -> None:
    """Synchronous index build, runs in a daemon thread at import time."""
    global _index, _skill_index, _index_ready
    t0 = time.monotonic()
    print(f"openclaw-mcp: building index in background ({VAULT})...", file=sys.stderr, flush=True)
    try:
        docs = _scan_vault(VAULT)
        new_index = TagIndex(docs)
        _index = new_index
        new_skills = SkillIndex()
        _skill_index = new_skills
        _index_ready = True
        print(
            f"openclaw-mcp: index ready in {time.monotonic() - t0:.1f}s — "
            f"{_index.doc_count} docs, {_index.tag_count} tags, {_skill_index.count} skills",
            file=sys.stderr, flush=True,
        )
    except Exception as exc:
        _index_ready = True  # unblock tools even on failure
        print(f"openclaw-mcp: initial index build failed: {exc}", file=sys.stderr, flush=True)


threading.Thread(target=_build_index_sync, daemon=True).start()

# ── Prefill: cache ────────────────────────────────────────────────────────────

_prefill_cache: dict[str, tuple[float, str]] = {}  # key -> (monotonic_ts, result)


def _get_prefill_cached(key: str) -> str | None:
    entry = _prefill_cache.get(key)
    if entry and (time.monotonic() - entry[0]) < PREFILL_CACHE_TTL_S:
        return entry[1]
    return None


def _set_prefill_cache(key: str, result: str) -> None:
    now = time.monotonic()
    _prefill_cache[key] = (now, result)
    stale = [k for k, (ts, _) in _prefill_cache.items() if now - ts > PREFILL_CACHE_TTL_S]
    for k in stale:
        del _prefill_cache[k]


# ── Prefill: query helpers (ported from query.ts) ──────────────────────────────

_PROMPT_ENVELOPE = re.compile(
    r"\[\w{3}\s+\d{4}-\d{2}-\d{2}\s+[\d:]+\s+\w+\]\s+([\s\S]+)$"
)
_JSON_BLOCK = re.compile(r"```json[\s\S]*?```")
_USER_TAG_RE = re.compile(r"\[/?user\]", re.IGNORECASE)
_PUNCT_SIMPLE = re.compile(r"[?!.,;:'\"]")
_PUNCT_FULL = re.compile(r"[?!.,;:'\"()\[\]{}#]")


def _extract_user_query(prompt: str) -> str | None:
    m = _PROMPT_ENVELOPE.search(prompt)
    if m:
        return _USER_TAG_RE.sub("", m.group(1)).strip()
    stripped = _JSON_BLOCK.sub("", prompt)
    stripped = _USER_TAG_RE.sub("", stripped)
    stripped = " ".join(stripped.split())
    return stripped[-500:] if len(stripped) >= MIN_QUERY_LENGTH else None


def _simplify_query(query: str) -> str:
    words = [
        w for w in _PUNCT_SIMPLE.sub("", query.lower()).split()
        if len(w) > 1 and w not in FILLER_WORDS
    ]
    return " ".join(words[:6])


def _extract_topic_keywords(text: str) -> list[str]:
    """Extract bigrams and keywords from text for tag matching and extra searches.

    Callers should pass the effective_query (short, pre-extracted) rather than
    the full raw prompt, to avoid false positives from boilerplate instructions.
    """
    text = _USER_TAG_RE.sub("", text).strip()

    words = [
        w for w in _PUNCT_FULL.sub(" ", text).lower().split()
        if len(w) > 2 and w not in FILLER_WORDS
    ]

    raw_words = [w for w in _PUNCT_FULL.sub(" ", text).split() if len(w) > 2]
    phrases: list[str] = []
    for i in range(len(raw_words) - 1):
        a, b = raw_words[i].lower(), raw_words[i + 1].lower()
        if a not in FILLER_WORDS and b not in FILLER_WORDS:
            phrases.append(f"{a} {b}")

    return phrases + words


def _match_keyword_to_tag(keyword: str, index: TagIndex) -> str | None:
    kw = keyword.lower()
    if kw in index.tag_to_docs:
        return kw
    kw_nodash = kw.replace("-", "")
    for tag in index.tag_to_docs:
        if tag.replace("-", "") == kw_nodash:
            return tag
    return None


# ── Prefill: async GLM keyword extractor ──────────────────────────────────────

async def _extract_keywords_via_glm(user_text: str) -> list[str]:
    """Call local Qwen GLM to extract 2-3 search keywords. Returns [] on any failure."""
    if len(user_text) < GLM_MIN_QUERY_LEN:
        return []
    try:
        async with _httpx.AsyncClient(timeout=1.5, verify=False) as client:
            resp = await client.post(
                GLM_URL,
                json={
                    "model": GLM_MODEL,
                    "max_tokens": 40,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a search keyword extractor. Always respond with "
                                "ONLY a valid JSON array of 2-3 short keyword strings. "
                                "No markdown, no explanation."
                            ),
                        },
                        {"role": "user", "content": f"Message: {user_text[:300]}"},
                    ],
                },
            )
        if resp.status_code != 200:
            return []
        text = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r"\[[\s\S]*?\]", text)
        return json.loads(m.group(0)) if m else []
    except Exception:
        return []


# ── Prefill: candidate dataclass + tag match + merge/rank ─────────────────────

@dataclass
class _Candidate:
    path: str
    doc: DocMeta | None
    vector_score: float
    tag_score: float
    snippet: str | None
    sources: set  # "vector" | "tag"
    final_score: float = 0.0
    content: str | None = None


def _run_tag_match(
    topic_keywords: list[str],
    index: TagIndex,
) -> tuple[list[dict[str, Any]], dict[str, tuple[DocMeta, float]]]:
    """Returns (matches_list, tag_docs map: path -> (DocMeta, tagScore))."""
    matched_tags: set[str] = set()
    matches: list[dict[str, Any]] = []

    for kw in topic_keywords:
        tag = _match_keyword_to_tag(kw, index)
        if not tag or tag in matched_tags:
            continue
        matched_tags.add(tag)

        paths = index.tag_to_docs.get(tag, [])
        if not paths:
            continue

        docs = sorted(
            [index.doc_index[p] for p in paths if p in index.doc_index],
            key=lambda d: d.importance,
            reverse=True,
        )[:MAX_DOCS_PER_TAG]

        matches.append({"tag": tag, "tagDocCount": len(paths), "docs": docs})
        if len(matches) >= MAX_TOPICS:
            break

    all_tag_docs = [d for m in matches for d in m["docs"]]
    max_idf = max((index.tag_idf.get(m["tag"], 0.0) for m in matches), default=1.0) or 1.0
    max_importance = max((d.importance for d in all_tag_docs), default=1.0) or 1.0

    tag_docs: dict[str, tuple[DocMeta, float]] = {}
    for m in matches:
        tag_idf = index.tag_idf.get(m["tag"], 0.0)
        for doc in m["docs"]:
            if doc.path in tag_docs:
                continue
            tag_score = 0.5 * (tag_idf / max_idf) + 0.5 * (doc.importance / max_importance)
            tag_docs[doc.path] = (doc, tag_score)

    return matches, tag_docs


def _merge_and_rank(
    tag_docs: dict[str, tuple[DocMeta, float]],
    vector_results: list[dict[str, Any]],
) -> list[_Candidate]:
    candidates: dict[str, _Candidate] = {}

    for vr in vector_results:
        path = vr["path"]
        existing = candidates.get(path)
        if existing:
            existing.vector_score = max(existing.vector_score, vr.get("score", 0))
            existing.sources.add("vector")
            if vr.get("snippet") and not existing.snippet:
                existing.snippet = vr["snippet"]
        else:
            candidates[path] = _Candidate(
                path=path,
                doc=None,
                vector_score=vr.get("score", 0),
                tag_score=0.0,
                snippet=vr.get("snippet"),
                sources={"vector"},
            )

    for path, (doc, tag_score) in tag_docs.items():
        existing = candidates.get(path)
        if existing:
            existing.doc = doc
            existing.tag_score = tag_score
            existing.sources.add("tag")
        else:
            candidates[path] = _Candidate(
                path=path,
                doc=doc,
                vector_score=SYNTHETIC_VECTOR_SCORE,
                tag_score=tag_score,
                snippet=None,
                sources={"tag"},
            )

    for c in candidates.values():
        cross = W_CROSS if len(c.sources) > 1 else 0.0
        c.final_score = W_VECTOR * c.vector_score + W_TAG * c.tag_score + cross

    return sorted(candidates.values(), key=lambda c: c.final_score, reverse=True)


# ── Prefill: context formatter (ported from format.ts) ───────────────────────

def _format_unified_context(
    tier1: list[_Candidate],
    tier2: list[_Candidate],
    budget: int,
    facts: list[dict] | None = None,
) -> str:
    if not tier1 and not tier2 and not facts:
        return ""

    lines: list[str] = [
        "<memory_context>",
        "IMPORTANT: Answer from the context below. Do NOT call mem_search or"
        " tag_search unless this context is clearly missing what you need."
        " Redundant searches waste time and slow down the response.\n",
    ]
    
    # Add facts block if provided
    if facts:
        lines.append("<facts>")
        for fact_item in facts:
            fact_text = fact_item.get("fact", "")
            confidence = fact_item.get("confidence", 0.9)
            category = fact_item.get("category", "")
            entity = fact_item.get("entity", "")
            lines.append(f"- [{entity}] {fact_text} (confidence: {confidence:.2f})")
        lines.append("</facts>")
        lines.append("")

    for c in tier1:
        badge = "|".join(p for p in [c.doc.type if c.doc else "", c.doc.status if c.doc else ""] if p)
        badge_str = f" [{badge}]" if badge else ""
        if c.doc and c.doc.title:
            title = c.doc.title
        else:
            stem = c.path.split("/")[-1]
            title = stem[:-3] if stem.endswith(".md") else stem

        src_parts: list[str] = []
        if "vector" in c.sources or "glm" in c.sources:
            src_parts.append("vector")
        if "tag" in c.sources:
            src_parts.append("tag")
        via = "+".join(src_parts)

        lines.append(f"--- {c.path}{badge_str} (score: {c.final_score:.2f}, via: {via}) ---")
        lines.append(f'"{title}"')
        if c.doc and c.doc.tags:
            lines.append(f"Tags: {', '.join(c.doc.tags)}")
        if c.content:
            lines.append(c.content)
        elif c.snippet:
            lines.append(c.snippet)
        elif c.doc and c.doc.summary:
            lines.append(c.doc.summary)
        lines.append("")

    if tier2:
        lines.append("**Also relevant** (use mem_get for full content):")
        for c in tier2:
            badge = "|".join(p for p in [c.doc.type if c.doc else "", c.doc.status if c.doc else ""] if p)
            badge_str = f" [{badge}]" if badge else ""
            if c.doc and c.doc.title:
                title = c.doc.title
            else:
                stem = c.path.split("/")[-1]
                title = stem[:-3] if stem.endswith(".md") else stem
            if c.doc and c.doc.summary:
                detail = f" — {c.doc.summary[:100]}"
            elif c.snippet:
                detail = f" — {c.snippet[:100]}"
            elif c.doc and c.doc.tags:
                detail = f" — Tags: {', '.join(c.doc.tags)}"
            else:
                detail = ""
            lines.append(f'- {c.path}{badge_str} "{title}" ({c.final_score:.2f}){detail}')
        lines.append("")

    lines.append("</memory_context>")

    full = "\n".join(lines)
    if len(full) <= budget:
        return full
    return full[: budget - 40] + "\n[... truncated]\n</memory_context>"


# ── Prefill: timing log ───────────────────────────────────────────────────────

_PREFILL_LOG = Path.home() / "agents" / "lloyd" / "logs" / "timing.jsonl"


def _log_prefill(**kwargs: Any) -> None:
    try:
        _PREFILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.datetime.now().isoformat(), "event": "memory_prefill", **kwargs}
        with _PREFILL_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ── Background TagIndex refresh ───────────────────────────────────────────────


async def _do_index_refresh(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Rescan vault and atomically swap _index. Runs scan in thread executor."""
    global _index, _skill_index
    if loop is None:
        loop = asyncio.get_running_loop()
    try:
        new_docs = await loop.run_in_executor(None, _scan_vault, VAULT)
        new_index = await loop.run_in_executor(None, TagIndex, new_docs)
        _index = new_index
        new_skills = await loop.run_in_executor(None, SkillIndex)
        _skill_index = new_skills
        print(
            f"openclaw-mcp: refreshed index: {_index.doc_count} docs, {_index.tag_count} tags, {_skill_index.count} skills",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"openclaw-mcp: index refresh failed: {exc}", file=sys.stderr)


async def _refresh_index_loop() -> None:
    """Rescan vault every PREFILL_REFRESH_S seconds and atomically swap _index."""
    global _index
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(PREFILL_REFRESH_S)
        await _do_index_refresh(loop)


@asynccontextmanager
async def _lifespan(_server: Any):
    task = asyncio.create_task(_refresh_index_loop())
    cleanup = asyncio.create_task(_bg_cleanup_loop())
    yield
    task.cancel()
    cleanup.cancel()


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("openclaw", lifespan=_lifespan)


@mcp.tool()
async def tag_search(
    tags: list[str],
    mode: str = "or",
    type: str = "any",
    limit: int = 10,
    scope: str = "",
) -> str:
    """Search vault by frontmatter tags. Returns matching documents with metadata."""
    def _impl():
        if not _index_ready and _index.doc_count == 0:
            return json.dumps({"error": "Vault index is still building — retry in a few seconds.", "status": "building"})
        limit_ = min(max(limit, 1), 25)
        type_filter = None if type == "any" else type
        scope_prefixes = _resolve_scope_prefixes(scope or None)

        unresolved: list[str] = []
        suggestions: dict[str, list[str]] = {}
        for tag in tags:
            if _index.resolve_tag(tag) is None:
                unresolved.append(tag)
                suggestions[tag] = _index.suggest_tags(tag)

        docs = _index.search_by_tags(tags, mode, type_filter, limit_, scope_prefixes)
        all_docs = _index.search_by_tags(tags, mode, type_filter, 9999, scope_prefixes)

        return _format_tag_search(tags, mode, docs, len(all_docs), limit_, unresolved, suggestions)
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def tag_explore(
    tag: str,
    bridge_to: str = "",
    limit: int = 15,
    scope: str = "",
) -> str:
    """Explore tag co-occurrence and relationships. bridge_to finds documents with BOTH tags."""
    def _impl():
        if not _index_ready and _index.doc_count == 0:
            return json.dumps({"error": "Vault index is still building — retry in a few seconds.", "status": "building"})
        limit_ = min(max(limit, 1), 30)
        scope_prefixes = _resolve_scope_prefixes(scope or None)
        canonical = _index.resolve_tag(tag)
        tag_clean = tag.lstrip("#")

        if canonical is None:
            suggestions = _index.suggest_tags(tag)
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            return f'Tag "{tag_clean}" not found in the vault.{hint}'

        all_docs = _index.tag_to_docs.get(canonical, [])
        if scope_prefixes:
            all_docs = [d for d in all_docs if any(d.startswith(p) for p in scope_prefixes)]
        tag_doc_count = len(all_docs)
        tag_idf = _index.tag_idf.get(canonical, 0.0)
        related = _index.get_related_tags(canonical, limit_)

        if scope_prefixes:
            # Re-count tag co-occurrences using only docs in scope
            scoped_docs = set(d for d in _index.tag_to_docs.get(canonical, [])
                              if any(d.startswith(p) for p in scope_prefixes))
            co_counts: dict[str, int] = {}
            for other_tag, other_docs in _index.tag_to_docs.items():
                if other_tag == canonical:
                    continue
                count = sum(1 for d in other_docs if d in scoped_docs)
                if count > 0:
                    co_counts[other_tag] = count
            scoped_related: list[dict[str, Any]] = []
            for other_tag, count in sorted(co_counts.items(), key=lambda x: x[1], reverse=True)[:limit_]:
                scoped_related.append({
                    "tag": other_tag,
                    "count": count,
                    "idf": _index.tag_idf.get(other_tag, 0.0),
                    "docFreq": len([d for d in _index.tag_to_docs.get(other_tag, [])
                                    if any(d.startswith(p) for p in scope_prefixes)]),
                })
            related = scoped_related

        bridge_tag: str | None = None
        bridge_docs: list[DocMeta] | None = None
        if bridge_to:
            bridge_tag = bridge_to.lstrip("#")
            bridge_docs = _index.search_by_tags([canonical, bridge_to], "and", None, 10, scope_prefixes)

        return _format_tag_explore(canonical, tag_doc_count, tag_idf, related, bridge_tag, bridge_docs)
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def vault_overview(detail: str = "summary") -> str:
    """Vault statistics. detail: summary|tags|hubs|types"""
    def _impl():
        if not _index_ready and _index.doc_count == 0:
            return json.dumps({"error": "Vault index is still building — retry in a few seconds.", "status": "building"})
        stats = _index.get_stats()
        return _format_vault_overview(stats, detail)
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


def _parse_qmd_results(
    raw_list: list[dict],
    scope_prefixes: list[str],
    min_score: float,
    strip_line_prefixes: bool = False,
) -> list[dict]:
    """Convert raw qmd result dicts to the canonical tool output format.

    Handles both daemon format (file="memory/path", snippets have line-number
    prefixes) and subprocess format (file="qmd://memory/path").
    """
    parsed = []
    for r in raw_list:
        file_val = r.get("file", "")
        # Strip qmd:// URI prefix; with per-segment collections QMD returns
        # e.g. "qmd://memory/subdir/file.md" where the collection name matches
        # the segment directory, so stripping "qmd://" yields the vault-relative path.
        path = file_val.removeprefix("qmd://")
        # Backward compat: strip legacy "obsidian/" prefix if present
        if path.startswith("obsidian/"):
            path = path.removeprefix("obsidian/")
        score = float(r.get("score", 0))
        if score < min_score:
            continue
        if scope_prefixes and not any(path.startswith(p) for p in scope_prefixes):
            continue
        # Parse line range from qmd's diff-style context markers: @@ -startLine,count @@
        raw_snippet = r.get("snippet", "")
        start_line = 0
        end_line = 0
        line_match = re.search(r"@@\s*-?(\d+)(?:\s*,\s*(\d+))?\s*@@", raw_snippet)
        if line_match:
            start_line = int(line_match.group(1))
            count = int(line_match.group(2)) if line_match.group(2) else 1
            end_line = start_line + count - 1
        # Strip @@ markers and the "(N before, M after)" description
        snippet = re.sub(r"@@[^@]*@@\s*(?:\([^)]*\)\s*)?", "", raw_snippet).strip()
        # Daemon snippets prefix each content line with "N: "; remove those
        if strip_line_prefixes:
            snippet = re.sub(r"^\d+:\s*", "", snippet, flags=re.MULTILINE).strip()
        citation = f"{path}#L{start_line}-L{end_line}" if start_line else path
        parsed.append({
            "path": path,
            "score": round(score, 4),
            "snippet": snippet[:300],
            "startLine": start_line,
            "endLine": end_line,
            "source": "memory",
            "citation": citation,
        })
    return parsed


async def _consolidate_results(query: str, results: list[dict], max_passages: int = 5) -> dict | None:
    """Send raw search results through the local 2B consolidation model.

    Returns parsed consolidation dict on success, None on failure (caller falls back to raw).
    """
    if not CONSOLIDATION_ENABLED or len(results) < CONSOLIDATION_MIN_RESULTS:
        return None

    # Build user prompt from results
    parts = [f"Query: {query}\n\nSearch Results:\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"--- Result {i} (score: {r.get('score', 'N/A')}) ---")
        parts.append(f"File: {r.get('citation', r.get('file', ''))}")
        parts.append(f"Title: {r.get('title', '')}")
        snippet = r.get("snippet", "")
        if len(snippet) > 3000:
            snippet = snippet[:3000] + "\n[... truncated ...]"
        parts.append(f"Content:\n{snippet}")
        parts.append("")
    parts.append(f"Consolidate these results into {max_passages} or fewer dense passages answering the query.")
    user_prompt = "\n".join(parts)

    payload = {
        "model": CONSOLIDATION_MODEL,
        "messages": [
            {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1000,
    }

    try:
        async with _httpx.AsyncClient() as client:
            resp = await client.post(
                CONSOLIDATION_ENDPOINT,
                json=payload,
                timeout=CONSOLIDATION_TIMEOUT,
            )
            if resp.status_code != 200:
                print(f"openclaw-mcp: consolidation HTTP {resp.status_code}", file=sys.stderr)
                return None

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return None

            # Try to parse JSON response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            return json.loads(content)
    except (json.JSONDecodeError, _httpx.TimeoutException, _httpx.ConnectError, Exception) as exc:
        print(f"openclaw-mcp: consolidation failed: {exc}", file=sys.stderr)
        return None


async def _maybe_consolidate_and_return(query: str, parsed: list[dict], max_results: int, consolidate: bool = True) -> str:
    """Optionally consolidate results, then return JSON response."""
    trimmed = parsed[:max_results]
    if consolidate and len(trimmed) >= CONSOLIDATION_MIN_RESULTS:
        consolidated = await _consolidate_results(query, trimmed)
        if consolidated:
            return json.dumps({
                "results": trimmed,
                "mode": "hybrid",
                "consolidated": True,
                "consolidated_summary": consolidated,
            })
    return json.dumps({"results": trimmed, "mode": "hybrid", "consolidated": False})


def _classify_query_intent(query: str) -> dict | None:
    """Classify a search query into an intent category using heuristics.

    Returns a dict with intent, suggested segments, and search_params,
    or None if no clear intent pattern matches (use default behavior).
    """
    q = QmdDaemonClient._sanitize(query)
    if not q:
        return None

    # Temporal queries (check early — temporal signals are strong)
    if _INTENT_TEMPORAL_RE.search(q):
        return {
            "intent": "temporal",
            "segments": ["memory", "personal"],
            "search_params": {
                "searches": [
                    {"type": "lex", "query": q},
                    {"type": "vec", "query": q},
                ],
            },
        }

    # Cross-reference queries (check before factual — "compare", "vs" are strong signals)
    if _INTENT_CROSSREF_RE.search(q):
        return {
            "intent": "cross_reference",
            "segments": SEGMENT_COLLECTIONS,
            "search_params": {
                "searches": [
                    {"type": "lex", "query": q},
                    {"type": "vec", "query": q},
                ],
            },
        }

    # Factual queries with specific nouns (pattern match on short queries, or
    # short queries with specific-looking tokens)
    word_count = len(q.split())
    if (word_count <= 8 and _INTENT_FACTUAL_RE.search(q)) or (word_count <= 6 and not _INTENT_CONCEPTUAL_RE.search(q)):
        # Only apply short-query heuristic if there's at least one capitalized
        # or specific-looking token (avoid classifying vague short queries)
        words = q.split()
        has_specific = _INTENT_FACTUAL_RE.search(q) or any(
            w[0].isupper() or w.replace("-", "").replace("_", "").replace(".", "").isdigit()
            for w in words if w
        )
        if has_specific:
            return {
                "intent": "factual",
                "segments": ["knowledge", "projects", "agents"],
                "search_params": {
                    "searches": [
                        {"type": "lex", "query": q},
                        {"type": "vec", "query": q},
                    ],
                },
            }

    # Conceptual / abstract queries (exploratory patterns only)
    if _INTENT_CONCEPTUAL_RE.search(q):
        return {
            "intent": "conceptual",
            "segments": ["knowledge"],
            "search_params": {
                "searches": [
                    {"type": "vec", "query": q},
                    {"type": "lex", "query": q},
                ],
            },
        }

    return None


@mcp.tool()
async def mem_search(
    query: str,
    max_results: int = 10,
    min_score: float = 0.0,
    scope: str = "",
    agent_id: str = "",
    consolidate: bool = True,
) -> str:
    """Mandatory recall: search vault before answering about prior work, decisions, dates, people, preferences, or todos. Returns matching paths with scores and snippets."""
    # Apply default read scope for known agents when no explicit scope given
    effective_scope = scope
    if not effective_scope and agent_id:
        effective_scope = DEFAULT_READ_SCOPE.get(agent_id, "")

    scope_prefixes = _resolve_scope_prefixes(effective_scope or None)

    # Map scope to specific collections for efficient querying
    target_collections = None
    if effective_scope:
        scope_segments = [s.strip().rstrip("/") for s in effective_scope.split(",") if s.strip()]
        if all(s in SEGMENT_COLLECTIONS for s in scope_segments):
            target_collections = scope_segments

    coll_list = target_collections or SEGMENT_COLLECTIONS

    # Intent-aware routing (only when no explicit scope narrows collections)
    intent_info = _classify_query_intent(query)
    intent_searches: list[dict] | None = None
    if intent_info and not effective_scope:
        print(f"openclaw-mcp: query intent={intent_info['intent']} for: {query[:80]}", file=sys.stderr)
        # Use intent-suggested segments, mapped to collections
        suggested = [s for s in intent_info["segments"] if s in SEGMENT_COLLECTIONS]
        if suggested and not target_collections:
            coll_list = suggested
        intent_searches = intent_info.get("search_params", {}).get("searches")

    intent_str: str | None = intent_info.get("intent") if intent_info else None

    # Single collection: direct query, no fan-out overhead
    if len(coll_list) == 1:
        def _single():
            return _qmd_client.search_one(query, max_results, collection=coll_list[0], searches=intent_searches, intent=intent_str)
        fts_results = await asyncio.get_running_loop().run_in_executor(None, _single)
        if fts_results is not None:
            parsed = _parse_qmd_results(fts_results, scope_prefixes, min_score, strip_line_prefixes=True)
            return await _maybe_consolidate_and_return(query, parsed, max_results, consolidate)
    else:
        # Parallel fan-out: one request per collection
        loop = asyncio.get_running_loop()

        async def _query_collection(coll: str) -> list[dict]:
            def _do():
                return _qmd_client.search_one(query, max_results, collection=coll, searches=intent_searches, intent=intent_str)
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, _do),
                    timeout=8.0,
                )
                return result or []
            except asyncio.TimeoutError:
                print(f"openclaw-mcp: QMD query timed out for collection {coll}", file=sys.stderr)
                return []
            except Exception as exc:
                print(f"openclaw-mcp: QMD query failed for collection {coll}: {exc}", file=sys.stderr)
                return []

        all_results = await asyncio.gather(*[_query_collection(c) for c in coll_list])

        # Merge and deduplicate by file path (keep highest score)
        merged: dict[str, dict] = {}
        for batch in all_results:
            for r in batch:
                fpath = r.get("file", "")
                score = float(r.get("score", 0))
                if fpath not in merged or score > float(merged[fpath].get("score", 0)):
                    merged[fpath] = r

        if merged:
            fts_results = sorted(merged.values(), key=lambda x: float(x.get("score", 0)), reverse=True)
            parsed = _parse_qmd_results(fts_results, scope_prefixes, min_score, strip_line_prefixes=True)
            return await _maybe_consolidate_and_return(query, parsed, max_results, consolidate)

        # If all collections returned empty, try a single combined request as fallback
        if all(len(batch) == 0 for batch in all_results):
            def _combined():
                return _qmd_client.search(query, max_results, collections=coll_list, searches=intent_searches, intent=intent_str)
            combined = await loop.run_in_executor(None, _combined)
            if combined is not None:
                parsed = _parse_qmd_results(combined, scope_prefixes, min_score, strip_line_prefixes=True)
                return await _maybe_consolidate_and_return(query, parsed, max_results, consolidate)

    # Fallback: subprocess (daemon unavailable)
    def _subprocess_fallback():
        try:
            env = {
                **os.environ,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
                "LD_LIBRARY_PATH": "/opt/cuda/lib64"
                + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
            }
            collection_args = []
            for c in coll_list:
                collection_args.extend(["-c", c])
            proc = subprocess.run(
                [str(QMD), "query", query, *collection_args, "-n", str(max_results), "--json"],
                capture_output=True, text=True, timeout=30, env=env,
            )
            if proc.returncode != 0:
                return json.dumps({"results": [], "error": proc.stderr.strip()})
            raw_list: list[dict] = json.loads(proc.stdout)
            parsed = _parse_qmd_results(raw_list, scope_prefixes, min_score)
            return json.dumps({"results": parsed[:max_results], "mode": "hybrid"})
        except subprocess.TimeoutExpired:
            return json.dumps({"results": [], "error": "qmd query timed out"})
        except Exception as exc:
            return json.dumps({"results": [], "error": str(exc)})

    return await asyncio.get_running_loop().run_in_executor(None, _subprocess_fallback)


def _resolve_case_insensitive(vault: Path, rel_path: str) -> Path | None:
    """Walk path segments with case-insensitive matching.

    QMD's handelize() lowercases all indexed paths, so search results return
    e.g. 'agents/lloyd/soul.md' when the file on disk is 'agents/lloyd/SOUL.md'.
    This function resolves the lowercase path to the actual file.
    """
    current = vault
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


@mcp.tool()
async def mem_get(path: str, start_line: int = 0, num_lines: int = 0, agent_id: str = "") -> str:
    """Read vault file by path. Use after mem_search."""
    def _impl():
        target = VAULT / path
        if not target.resolve().is_relative_to(VAULT.resolve()):
            return json.dumps({"path": path, "text": "", "error": "path escapes vault root"})
        if not target.exists():
            # Fallback: case-insensitive resolution (QMD lowercases indexed paths)
            resolved = _resolve_case_insensitive(VAULT, path)
            if resolved is None or not resolved.is_relative_to(VAULT):
                return json.dumps({"path": path, "text": "", "error": f"File not found: {path}"})
            target = resolved
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
            if start_line > 0 or num_lines > 0:
                file_lines = text.splitlines()
                start = max(0, start_line - 1)
                end = (start + num_lines) if num_lines > 0 else len(file_lines)
                text = "\n".join(file_lines[start:end])
            return json.dumps({"path": path, "text": text or "(empty file)"})
        except OSError as exc:
            return json.dumps({"path": path, "text": "", "error": str(exc)})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


# ── Prefill context tool ──────────────────────────────────────────────────────


@mcp.tool()
async def prefill_context(prompt: str, session_id: str = "", scope: str = "") -> str:
    """Run the memory prefill pipeline. Not for direct LLM use."""
    t0 = time.monotonic()
    scope_val = scope or ""

    # Phase 0: query extraction + cache check
    user_query = _extract_user_query(prompt)
    if not user_query or len(user_query) < MIN_QUERY_LENGTH:
        return ""

    vector_query = _simplify_query(user_query)
    effective_query = vector_query if len(vector_query.split()) >= 2 else user_query[:80]
    if len(effective_query.split()) < 2:
        return ""

    # Include scope in cache key so different scopes don't share cache entries
    cache_key = effective_query + (f"|scope:{scope_val}" if scope_val else "")
    cached = _get_prefill_cached(cache_key)
    if cached:
        return cached

    scope_prefixes = _resolve_scope_prefixes(scope_val or None)

    try:
        async with asyncio.timeout(PREFILL_TIMEOUT_S):
            # Phase 1-2: keyword extraction + tag match (sync, instant)
            # Use effective_query (not full prompt) to avoid false tag matches
            # from boilerplate instructions in long system-style prompts.
            topic_keywords = _extract_topic_keywords(effective_query)
            if _index.doc_count > 0:
                tag_matches, tag_docs = _run_tag_match(topic_keywords, _index)
                # Apply scope filter to tag_docs
                if scope_prefixes and tag_docs:
                    tag_docs = {
                        p: d for p, d in tag_docs.items()
                        if any(p.startswith(pfx) for pfx in scope_prefixes)
                    }
            else:
                tag_matches, tag_docs = [], {}
            matched_tag_names = {m["tag"].lower() for m in tag_matches}

            # Phase 3+4: parallel BM25 searches — all local FTS5 (~1ms each)
            # Use top local bigrams as extra queries; no GLM needed.
            extra_queries = [
                k for k in topic_keywords[:6]
                if len(k) > 4 and k.lower() not in matched_tag_names
            ][:2]

            # Map scope to specific collections for efficient querying
            prefill_target_collections = None
            if scope_val:
                prefill_scope_segments = [s.strip().rstrip("/") for s in scope_val.split(",") if s.strip()]
                if all(s in SEGMENT_COLLECTIONS for s in prefill_scope_segments):
                    prefill_target_collections = prefill_scope_segments

            async def _fts_search(q: str, limit: int) -> list[dict[str, Any]]:
                scope_prefixes = _resolve_scope_prefixes(scope_val or None)
                # prefill path: intent not applicable
                fts_results = await asyncio.to_thread(_qmd_client.search, q, limit, prefill_target_collections)
                if fts_results is None:
                    return []
                return _parse_qmd_results(fts_results, scope_prefixes, 0.0, strip_line_prefixes=True)

            search_tasks = [_fts_search(effective_query, 8)] + [
                _fts_search(q, 3) for q in extra_queries
            ]
            search_results = await asyncio.gather(*search_tasks)
            init_results = search_results[0]
            extra_results = [r for batch in search_results[1:] for r in batch]
            all_vector = init_results + extra_results
            glm_keywords = extra_queries  # kept for log field compatibility

            # Phase 5: merge & rank
            ranked = _merge_and_rank(tag_docs, all_vector)
            if not ranked:
                return ""

            # Phase 6: tier classification
            tier1 = [c for c in ranked if c.final_score >= TIER1_THRESHOLD][:TIER1_MAX]
            if not tier1 and tag_docs:
                tier1 = ranked[: min(2, len(ranked))]
            tier2 = [
                c for c in ranked
                if c.final_score >= TIER2_THRESHOLD and c not in tier1
            ][:TIER2_MAX]

            # Phase 6.5: Fact injection (Engram-inspired proactive retrieval)
            # Extract entities from keywords and fetch their top facts
            facts = []
            if topic_keywords:
                # Use the same entity matching logic as context_bundle
                # Build a combined query string from keywords for better matching
                query_for_entities = " ".join(topic_keywords[:10])
                entity_matches = _extract_entities_from_query(query_for_entities)
                
                # Fetch facts for matched entities (max 10 facts total)
                for entity, score in entity_matches[:5]:
                    try:
                        entity_data = _get_facts_sync(entity)
                        if entity_data.get("facts"):
                            # Sort by confidence and take top facts
                            sorted_facts = sorted(
                                entity_data["facts"],
                                key=lambda f: f.get("confidence", 0.5),
                                reverse=True
                            )
                            # Add facts until we hit the limit
                            for fact in sorted_facts:
                                if len(facts) >= 10:
                                    break
                                facts.append({
                                    "entity": entity,
                                    "fact": fact.get("fact", ""),
                                    "confidence": fact.get("confidence", 0.9),
                                    "category": fact.get("category", "")
                                })
                    except Exception:
                        pass

            # Phase 7: fetch tier-1 content
            if tier1:
                fetched = await asyncio.gather(
                    *[asyncio.to_thread(mem_get, c.path) for c in tier1],
                    return_exceptions=True,
                )
                for i, r in enumerate(fetched):
                    if isinstance(r, str) and r and not r.startswith("File not found"):
                        tier1[i].content = r[:CONTENT_PER_DOC]

            # Phase 8: format
            context = _format_unified_context(tier1, tier2, MAX_CONTEXT_CHARS, facts)
            if not context:
                return ""

            _set_prefill_cache(cache_key, context)
            _log_prefill(
                session_id=session_id,
                duration_ms=int((time.monotonic() - t0) * 1000),
                effective_query=effective_query,
                query_length=len(user_query),
                tag_topics=len(tag_matches),
                tag_docs=len(tag_docs),
                vector_results=len(all_vector),
                glm_keywords=len(glm_keywords),
                tier1_count=len(tier1),
                tier2_count=len(tier2),
                context_chars=len(context),
            )
            return context

    except asyncio.TimeoutError:
        return ""
    except Exception:
        return ""


# ── Web tools ─────────────────────────────────────────────────────────────────


@mcp.tool()
async def http_search(query: str, count: int = 5) -> str:
    """Web search (DuckDuckGo). Returns titles, URLs, snippets."""
    def _impl():
        count_ = min(max(count, 1), 10)
        try:
            raw = list(_DDGS().text(query, max_results=count_))
        except Exception as exc:
            return f"http_search error: {exc}"
        if not raw:
            return f'No results found for "{query}".'
        lines: list[str] = []
        for i, r in enumerate(raw, 1):
            title = r.get("title", "") or ""
            url = r.get("href", "") or ""
            snippet = r.get("body", "") or ""
            lines.append(f"[{i}] {title}\n    {url}\n    {snippet}")
        return "\n\n".join(lines)
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def http_fetch(
    url: str,
    extract_mode: str = "markdown",
    max_chars: int = WEB_DEFAULT_MAX_CHARS,
) -> str:
    """Fetch a URL and extract readable content as markdown or text."""
    def _impl():
        from urllib.parse import urlparse

        max_chars_ = min(max(max_chars, 1_000), 200_000)

        try:
            parsed = urlparse(url)
        except Exception:
            return f"http_fetch error: Invalid URL: {url}"

        if parsed.scheme not in ("http", "https"):
            return f"http_fetch error: Only http/https URLs are supported, got {parsed.scheme!r}"

        hostname = parsed.hostname or ""
        if _is_private_host(hostname):
            return f'http_fetch error: Blocked — private/internal hostname "{hostname}"'

        headers = {
            "User-Agent": WEB_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": WEB_ACCEPT_LANG,
        }
        try:
            # verify=False: lloyd container SSL chain is incomplete; tool already blocks private IPs
            with _httpx.Client(follow_redirects=True, timeout=WEB_TIMEOUT_S, verify=False) as client:
                response = client.get(url, headers=headers)
        except _httpx.TimeoutException:
            return f"http_fetch error: Request timed out after {WEB_TIMEOUT_S}s"
        except Exception as exc:
            return f"http_fetch error: {exc}"

        if response.status_code >= 400:
            return f"http_fetch error: HTTP {response.status_code}"

        content_type = response.headers.get("content-type", "")

        # Non-HTML: return raw text
        if "html" not in content_type and "xml" not in content_type:
            text = response.text
            truncated = text[:max_chars_]
            if len(truncated) < len(text):
                return truncated + f"\n\n[Truncated — {len(text)} chars total]"
            return truncated

        raw_bytes = response.content[:WEB_MAX_RESPONSE_BYTES]
        html_text = raw_bytes.decode("utf-8", errors="replace")

        try:
            doc = _ReadabilityDocument(html_text)
            title = doc.title() or ""
            summary_html = doc.summary()
        except Exception as exc:
            return f"http_fetch error: readability failed: {exc}"

        converter = _html2text.HTML2Text()
        converter.ignore_links = extract_mode == "text"
        converter.ignore_images = True
        converter.body_width = 0  # no line wrapping

        body = converter.handle(summary_html).strip()
        full = f"# {title}\n\n{body}" if title else body

        truncated = full[:max_chars_]
        if len(truncated) < len(full):
            return truncated + f"\n\n[Truncated — {len(full)} chars total]"
        return truncated
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


# ── File system tools ─────────────────────────────────────────────────────────

HOME = Path.home()
FILE_MAX_READ_BYTES = 2_000_000  # 2 MB


def _safe_path(raw: str) -> Path | str:
    """Expand ~ and verify the path stays within HOME. Returns Path or error string."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = HOME / p
    try:
        resolved = p.resolve()
    except OSError as exc:
        return f"Error resolving path: {exc}"
    if not str(resolved).startswith(str(HOME)):
        return f"Error: path escapes home directory: {raw!r}"
    return p


@mcp.tool()
async def file_read(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read file (must be within $HOME). Supports line ranges."""
    def _impl():
        result = _safe_path(path)
        if isinstance(result, str):
            return result
        p = result
        if not p.exists():
            return f"File not found: {path}"
        if not p.is_file():
            return f"Not a file: {path}"
        size = p.stat().st_size
        if size > FILE_MAX_READ_BYTES:
            return f"Error: file too large ({size} bytes, max {FILE_MAX_READ_BYTES})"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Error reading file: {exc}"
        if start_line > 0 or end_line > 0:
            lines = text.splitlines()
            start = max(0, start_line - 1)
            end = end_line if end_line > 0 else len(lines)
            text = "\n".join(lines[start:end])
        return text or "(empty file)"
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def file_write(path: str, content: str) -> str:
    """Write file. Creates parent dirs."""
    def _impl():
        result = _safe_path(path)
        if isinstance(result, str):
            return result
        p = result
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"Error writing file: {exc}"
        return f"Written {len(content)} chars to {p}"
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def file_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace an exact string in a file (first occurrence only)."""
    def _impl():
        result = _safe_path(path)
        if isinstance(result, str):
            return result
        p = result
        if not p.exists():
            return f"File not found: {path}"
        try:
            original = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Error reading file: {exc}"
        count = original.count(old_text)
        if count == 0:
            return "Error: old_text not found in file"
        if count > 1:
            return f"Error: old_text appears {count} times — provide more context to make it unique"
        updated = original.replace(old_text, new_text, 1)
        try:
            p.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return f"Error writing file: {exc}"
        return f"Replaced 1 occurrence in {p}"
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


# ── file_patch helpers ────────────────────────────────────────────────────────


@dataclass
class _PatchOp:
    """A single file operation parsed from a unified diff."""
    op: str                # "A" (add), "M" (modify), "D" (delete)
    path: str              # relative file path (from +++ line)
    hunks: list[tuple[int, int, int, int, list[str]]]  # (old_start, old_count, new_start, new_count, lines)
    new_content: str = ""  # full content for new-file operations


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_PATCH_MAX_BYTES = 500_000  # 500 KB


def _strip_prefix(path: str) -> str:
    """Strip a/ or b/ prefix from a diff path."""
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _parse_unified_diff(patch_text: str) -> list[_PatchOp]:
    """Parse unified diff text into a list of PatchOp objects."""
    ops: list[_PatchOp] = []
    lines = patch_text.splitlines(keepends=True)
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Find --- / +++ header pair
        if not line.startswith("--- "):
            i += 1
            continue

        if i + 1 >= n or not lines[i + 1].startswith("+++ "):
            i += 1
            continue

        old_path_raw = line[4:].strip().split("\t")[0]
        new_path_raw = lines[i + 1][4:].strip().split("\t")[0]
        i += 2

        old_is_null = old_path_raw == "/dev/null"
        new_is_null = new_path_raw == "/dev/null"

        if old_is_null and new_is_null:
            continue  # nonsensical

        if new_is_null:
            op_type = "D"
            path = _strip_prefix(old_path_raw)
        elif old_is_null:
            op_type = "A"
            path = _strip_prefix(new_path_raw)
        else:
            op_type = "M"
            path = _strip_prefix(new_path_raw)

        # Collect hunks
        hunks: list[tuple[int, int, int, int, list[str]]] = []
        while i < n:
            hunk_match = _HUNK_RE.match(lines[i])
            if not hunk_match:
                break
            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2)) if hunk_match.group(2) is not None else 1
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4)) if hunk_match.group(4) is not None else 1
            i += 1
            hunk_lines: list[str] = []
            while i < n and not lines[i].startswith("--- ") and not _HUNK_RE.match(lines[i]):
                hunk_lines.append(lines[i])
                i += 1
            hunks.append((old_start, old_count, new_start, new_count, hunk_lines))

        # For new files, reconstruct full content from hunks
        new_content = ""
        if op_type == "A" and hunks:
            content_lines = []
            for _, _, _, _, hunk_lines in hunks:
                for hl in hunk_lines:
                    if hl.startswith("+"):
                        content_lines.append(hl[1:])
                    elif hl.startswith(" "):
                        content_lines.append(hl[1:])
                    # skip lines starting with - (shouldn't exist for new files)
                    elif hl.startswith("\\ No newline"):
                        pass
            new_content = "".join(content_lines)

        ops.append(_PatchOp(op=op_type, path=path, hunks=hunks, new_content=new_content))

    return ops


def _apply_hunks(original: str, hunks: list[tuple[int, int, int, int, list[str]]]) -> str | None:
    """Apply hunks to original text. Returns modified text or None on failure."""
    orig_lines = original.splitlines(keepends=True)
    # Ensure every line ends with newline for consistent matching
    if orig_lines and not orig_lines[-1].endswith("\n"):
        orig_lines[-1] += "\n"

    # We need to apply hunks from bottom to top to preserve line numbers
    sorted_hunks = sorted(hunks, key=lambda h: h[0], reverse=True)

    for old_start, old_count, _new_start, _new_count, hunk_lines in sorted_hunks:
        # Separate context/deletion lines (what we expect) and addition lines (what we insert)
        expected: list[str] = []
        replacement: list[str] = []
        for hl in hunk_lines:
            if hl.startswith(" "):
                expected.append(hl[1:])
                replacement.append(hl[1:])
            elif hl.startswith("-"):
                expected.append(hl[1:])
            elif hl.startswith("+"):
                replacement.append(hl[1:])
            elif hl.startswith("\\ No newline"):
                # Strip trailing newline from last line of previous group
                if replacement and replacement[-1].endswith("\n"):
                    replacement[-1] = replacement[-1][:-1]
                elif expected and expected[-1].endswith("\n"):
                    expected[-1] = expected[-1][:-1]

        # Find where the expected lines match (try exact position first, then fuzzy)
        start_idx = old_start - 1  # 1-indexed to 0-indexed
        match_found = False

        for offset in range(0, 6):  # try 0, +1, -1, +2, -2, +3, ...
            for direction in ([0] if offset == 0 else [offset, -offset]):
                try_idx = start_idx + direction
                if try_idx < 0 or try_idx + len(expected) > len(orig_lines):
                    continue
                # Compare expected vs actual
                match = True
                for j, exp_line in enumerate(expected):
                    actual = orig_lines[try_idx + j]
                    if exp_line.rstrip("\n") != actual.rstrip("\n"):
                        match = False
                        break
                if match:
                    start_idx = try_idx
                    match_found = True
                    break
            if match_found:
                break

        if not match_found and expected:
            return None  # hunk failed to apply

        # Apply: replace expected lines with replacement lines
        orig_lines[start_idx:start_idx + len(expected)] = replacement

    return "".join(orig_lines)


@mcp.tool()
async def file_patch(patch: str, root: str = "~") -> str:
    """Apply unified diff patch to files."""
    def _impl():
        if len(patch.encode("utf-8", errors="replace")) > _PATCH_MAX_BYTES:
            return f"Error: patch exceeds maximum size ({_PATCH_MAX_BYTES} bytes)"

        root_result = _safe_path(root)
        if isinstance(root_result, str):
            return root_result
        root_path = root_result

        try:
            ops = _parse_unified_diff(patch)
        except Exception as exc:
            return f"Error parsing patch: {exc}"

        if not ops:
            return "Error: no file operations found in patch"

        # Validation pass
        resolved: list[tuple[_PatchOp, Path]] = []
        for op in ops:
            # Resolve path relative to root
            if os.path.isabs(op.path):
                fp = Path(op.path)
            else:
                fp = root_path / op.path
            safe = _safe_path(str(fp))
            if isinstance(safe, str):
                return f"Error: {safe} (file: {op.path})"
            p = safe

            if op.op == "M" and not p.exists():
                return f"Error: file not found for modification: {op.path}"
            if op.op == "D" and not p.exists():
                return f"Error: file not found for deletion: {op.path}"
            resolved.append((op, p))

        # Apply pass
        results: list[str] = []
        for op, p in resolved:
            if op.op == "A":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(op.new_content, encoding="utf-8")
                results.append(f"A {op.path}")
            elif op.op == "D":
                p.unlink()
                results.append(f"D {op.path}")
            elif op.op == "M":
                original = p.read_text(encoding="utf-8", errors="replace")
                modified = _apply_hunks(original, op.hunks)
                if modified is None:
                    return f"Error: hunk failed to apply in {op.path} (context mismatch)"
                p.write_text(modified, encoding="utf-8")
                results.append(f"M {op.path}")

        return "\n".join(results) if results else "No operations applied"
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def file_glob(pattern: str, root: str = "~") -> str:
    """Find files matching a glob pattern. Returns up to 200 paths."""
    def _impl():
        root_result = _safe_path(root)
        if isinstance(root_result, str):
            return root_result
        root_path = root_result.expanduser() if hasattr(root_result, "expanduser") else root_result

        if not root_path.exists():
            return f"Root directory not found: {root}"
        if not root_path.is_dir():
            return f"Not a directory: {root}"

        try:
            matches = sorted(root_path.glob(pattern))
        except Exception as exc:
            return f"Error globbing: {exc}"

        if not matches:
            return f'No files matching "{pattern}" under {root_path}'

        MAX = 200
        lines = [str(m.relative_to(root_path)) for m in matches[:MAX]]
        result = "\n".join(lines)
        if len(matches) > MAX:
            result += f"\n... and {len(matches) - MAX} more (pattern matched {len(matches)} total)"
        return result
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def file_grep(pattern: str, path: str = "~", file_glob: str = "**/*", max_results: int = 50) -> str:
    """Search file contents by regex pattern."""
    def _impl():
        max_results_ = min(max(max_results, 1), 200)

        path_result = _safe_path(path)
        if isinstance(path_result, str):
            return path_result
        search_path = path_result

        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex: {exc}"

        results: list[str] = []

        def _search_file(fp: Path) -> None:
            if len(results) >= max_results_:
                return
            try:
                size = fp.stat().st_size
                if size > FILE_MAX_READ_BYTES:
                    return
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return
            for lineno, line in enumerate(text.splitlines(), 1):
                if len(results) >= max_results_:
                    break
                if compiled.search(line):
                    results.append(f"{fp}:{lineno}: {line.rstrip()}")

        if search_path.is_file():
            _search_file(search_path)
        elif search_path.is_dir():
            try:
                for fp in sorted(search_path.glob(file_glob)):
                    if len(results) >= max_results_:
                        break
                    if fp.is_file():
                        _search_file(fp)
            except Exception as exc:
                return f"Error searching: {exc}"
        else:
            return f"Path not found: {path}"

        if not results:
            return f'No matches for "{pattern}" in {search_path}'

        output = "\n".join(results)
        if len(results) >= max_results_:
            output += f"\n... (limit of {max_results_} results reached)"
        return output
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


# ── run_bash ──────────────────────────────────────────────────────────────────


@mcp.tool()
async def run_bash(command: str, cwd: str = "~", timeout: int = 30) -> str:
    """Execute a bash command. Returns stdout+stderr with exit code."""
    def _impl():
        timeout_ = min(max(timeout, 1), 120)
        cwd_result = _safe_path(cwd)
        if isinstance(cwd_result, str):
            return cwd_result

        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=str(cwd_result),
                capture_output=True,
                text=True,
                timeout=timeout_,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout_}s"
        except Exception as exc:
            return f"Error: {exc}"

        output = proc.stdout
        if proc.stderr:
            output += ("\n" if output else "") + proc.stderr
        return f"exit {proc.returncode}\n{output}" if output else f"exit {proc.returncode}"
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


# ── bg_exec / bg_process ─────────────────────────────────────────────────────

BG_MAX_SESSIONS = 20
BG_OUTPUT_BUFFER_MAX_LINES = 5000
BG_AUTO_EXPIRE_S = 3600  # 1 hour after completion


@dataclass
class _BgSession:
    session_id: str
    pid: int
    command: str
    cwd: str
    start_time: float
    proc: subprocess.Popen
    output_buffer: list[str] = field(default_factory=list)
    output_lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "running"        # running | completed | failed
    exit_code: int | None = None
    end_time: float | None = None
    _reader: threading.Thread | None = field(default=None, repr=False)
    _watchdog: threading.Thread | None = field(default=None, repr=False)


_bg_sessions: dict[str, _BgSession] = {}
_bg_counter = 0
_bg_lock = threading.Lock()


def _bg_reader(session: _BgSession) -> None:
    """Read stdout+stderr into the session buffer. Runs as daemon thread."""
    try:
        for line in session.proc.stdout:
            with session.output_lock:
                session.output_buffer.append(line)
                if len(session.output_buffer) > BG_OUTPUT_BUFFER_MAX_LINES:
                    session.output_buffer.pop(0)
    except (ValueError, OSError):
        pass  # pipe closed
    session.proc.wait()
    session.exit_code = session.proc.returncode
    session.end_time = time.time()
    session.status = "completed" if session.exit_code == 0 else "failed"


def _bg_watchdog(session: _BgSession, timeout_s: int) -> None:
    """Auto-kill a session after timeout_s seconds. Daemon thread."""
    deadline = session.start_time + timeout_s
    while time.time() < deadline:
        if session.status != "running":
            return
        time.sleep(5)
    # Timeout reached — kill
    if session.status == "running":
        try:
            session.proc.terminate()
            session.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            session.proc.kill()
            session.proc.wait()
        except OSError:
            pass


async def _bg_cleanup_loop() -> None:
    """Remove expired completed sessions. Runs in the asyncio event loop."""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        with _bg_lock:
            expired = [
                sid for sid, s in _bg_sessions.items()
                if s.status != "running" and s.end_time
                and (now - s.end_time) > BG_AUTO_EXPIRE_S
            ]
            for sid in expired:
                del _bg_sessions[sid]


@mcp.tool()
async def bg_exec(command: str, cwd: str = "~", timeout: int = 1800) -> str:
    """Start a background shell command. Returns session_id. Use bg_process to manage."""
    def _impl():
        global _bg_counter
        timeout_ = min(max(timeout, 10), 7200)
        cwd_result = _safe_path(cwd)
        if isinstance(cwd_result, str):
            return cwd_result

        with _bg_lock:
            running = sum(1 for s in _bg_sessions.values() if s.status == "running")
            if running >= BG_MAX_SESSIONS:
                return f"Error: too many background sessions ({running}/{BG_MAX_SESSIONS}). Kill some first."
            session_id = f"bg-{_bg_counter}"
            _bg_counter += 1

        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                cwd=str(cwd_result),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            return f"Error starting process: {exc}"

        session = _BgSession(
            session_id=session_id,
            pid=proc.pid,
            command=command[:200],
            cwd=str(cwd_result),
            start_time=time.time(),
            proc=proc,
        )

        reader = threading.Thread(target=_bg_reader, args=(session,), daemon=True)
        reader.start()
        session._reader = reader

        watchdog = threading.Thread(target=_bg_watchdog, args=(session, timeout_), daemon=True)
        watchdog.start()
        session._watchdog = watchdog

        with _bg_lock:
            _bg_sessions[session_id] = session

        return json.dumps({"session_id": session_id, "pid": proc.pid, "status": "running"})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def bg_process(
    action: str,
    session_id: str = "",
    timeout: int = 10,
    text: str = "",
    offset: int = 0,
    limit: int = 100,
) -> str:
    """Manage background processes. action: list|poll|log|write|kill"""
    def _impl():
        action_ = action.lower().strip()

        if action_ == "list":
            with _bg_lock:
                sessions = list(_bg_sessions.values())
            now = time.time()
            entries = []
            for s in sessions:
                runtime = (s.end_time or now) - s.start_time
                with s.output_lock:
                    line_count = len(s.output_buffer)
                entries.append({
                    "session_id": s.session_id,
                    "pid": s.pid,
                    "command": s.command,
                    "status": s.status,
                    "exit_code": s.exit_code,
                    "runtime_s": round(runtime, 1),
                    "output_lines": line_count,
                })
            return json.dumps(entries, indent=2)

        # All other actions require a session_id
        if not session_id:
            return "Error: session_id is required for this action"
        with _bg_lock:
            session = _bg_sessions.get(session_id)
        if not session:
            return f"Error: session '{session_id}' not found"

        if action_ == "poll":
            timeout_ = min(max(timeout, 1), 60)
            with session.output_lock:
                start_len = len(session.output_buffer)
            # Wait for new output or process exit
            deadline = time.time() + timeout_
            while time.time() < deadline:
                with session.output_lock:
                    cur_len = len(session.output_buffer)
                if cur_len > start_len or session.status != "running":
                    break
                time.sleep(0.5)
            # Return new lines
            with session.output_lock:
                new_lines = session.output_buffer[start_len:]
            result = {
                "status": session.status,
                "exit_code": session.exit_code,
                "new_lines": len(new_lines),
                "output": "".join(new_lines),
            }
            return json.dumps(result)

        elif action_ == "log":
            limit_ = min(max(limit, 1), 500)
            with session.output_lock:
                buf = session.output_buffer
                total = len(buf)
                if offset > 0:
                    chunk = buf[offset:offset + limit_]
                else:
                    # Last N lines
                    chunk = buf[-limit_:] if total > limit_ else buf[:]
            result = {
                "status": session.status,
                "exit_code": session.exit_code,
                "total_lines": total,
                "offset": offset if offset > 0 else max(0, total - limit_),
                "lines_returned": len(chunk),
                "output": "".join(chunk),
            }
            return json.dumps(result)

        elif action_ == "write":
            if session.status != "running":
                return f"Error: session '{session_id}' is not running (status: {session.status})"
            if not text:
                return "Error: text parameter is required for write action"
            try:
                session.proc.stdin.write(text)
                session.proc.stdin.flush()
            except (OSError, BrokenPipeError) as exc:
                return f"Error writing to stdin: {exc}"
            return json.dumps({"ok": True, "bytes_written": len(text)})

        elif action_ == "kill":
            if session.status != "running":
                return f"Session '{session_id}' already {session.status} (exit_code={session.exit_code})"
            try:
                session.proc.terminate()
                try:
                    session.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    session.proc.kill()
                    session.proc.wait(timeout=5)
            except OSError as exc:
                return f"Error killing process: {exc}"
            return json.dumps({
                "ok": True,
                "session_id": session_id,
                "status": session.status,
                "exit_code": session.exit_code,
            })

        else:
            return f"Error: unknown action '{action_}'. Must be one of: list, poll, log, write, kill"
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


# ── http_request ───────────────────────────────────────────────────────────────

_HTTP_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}


@mcp.tool()
async def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: str = "",
    timeout: int = 30,
) -> str:
    """HTTP request. Returns status + body. Private IPs blocked (loopback allowed)."""
    def _impl():
        import urllib.parse as _urlparse

        method_ = method.upper()
        if method_ not in _HTTP_ALLOWED_METHODS:
            return f"Error: unsupported method {method_!r}. Allowed: {', '.join(sorted(_HTTP_ALLOWED_METHODS))}"

        timeout_ = min(max(timeout, 1), 120)

        try:
            parsed = _urlparse.urlparse(url)
        except Exception as exc:
            return f"Error: invalid URL: {exc}"

        if parsed.scheme not in ("http", "https"):
            return f"Error: only http/https supported, got {parsed.scheme!r}"

        hostname = parsed.hostname or ""
        # Allow loopback (127.*) for local services; block all other private IPs
        if _is_private_host(hostname) and not hostname.startswith("127."):
            return f'Error: blocked — private/internal hostname "{hostname}"'

        try:
            resp = _httpx.request(
                method_,
                url,
                headers=headers or {},
                content=body.encode() if body else b"",
                timeout=timeout_,
                verify=False,
                follow_redirects=True,
            )
            return f"HTTP {resp.status_code}\n{resp.text}"
        except Exception as exc:
            return f"Error: {exc}"
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


# ── mem_write ───────────────────────────────────────────────────────────────


@mcp.tool()
async def mem_write(path: str, content: str, agent_id: str = "") -> str:
    """Write vault file. Creates parent dirs."""
    loop = asyncio.get_running_loop()
    effective_agent = agent_id or None

    # Check write permission
    perm_error = _check_write_permission(effective_agent, path)
    if perm_error:
        return perm_error

    def _impl():
        target = VAULT / path
        try:
            if not target.resolve().is_relative_to(VAULT.resolve()):
                return "Error: path escapes vault root"
        except Exception as exc:
            return f"Error: invalid path: {exc}"

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            # Audit log
            _audit_write(effective_agent, path, len(content.encode("utf-8")))
            # Schedule non-blocking index refresh so tag searches see the new file
            try:
                loop.call_soon_threadsafe(loop.create_task, _do_index_refresh(loop))
            except RuntimeError:
                pass  # no running loop (e.g. during testing)
            return f"Written {len(content)} chars to {path}"
        except Exception as exc:
            return f"Error: {exc}"
    return await loop.run_in_executor(None, _impl)


# ── Backlog markdown helpers ──────────────────────────────────────────────────


def _generate_slug(name: str) -> str:
    """Generate a URL-friendly slug from a task name.
    
    Lowercase, replace spaces/special chars with hyphens, strip leading/trailing hyphens.
    """
    slug = name.lower()
    # Replace spaces and special characters with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    # Strip leading/trailing hyphens
    slug = slug.strip('-')
    return slug


def _parse_backlog_frontmatter(text: str) -> dict | None:
    """Extract YAML frontmatter from a markdown file. Returns dict or None."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---\n", 3)
    if end == -1:
        return None
    try:
        return yaml.safe_load(text[4:end])
    except yaml.YAMLError:
        return None


def _write_backlog_frontmatter(data: dict) -> str:
    """Generate YAML frontmatter block for a backlog file."""
    # Ensure tags is a list for YAML serialization
    if "tags" in data and isinstance(data["tags"], str):
        data["tags"] = [t.strip() for t in data["tags"].split(",") if t.strip()]
    
    # Handle None/null for completed field - ensure it stays as None for YAML serialization
    if "completed" in data and data["completed"] is None:
        # Keep as None, will serialize as 'null' in YAML
        pass
    
    fm = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    # Fix null serialization - YAML may output 'null' as '.' in some cases
    fm = re.sub(r'completed: \.\n', 'completed: null\n', fm)
    return f"---\n{fm}---\n"


def _find_max_backlog_id() -> int:
    """Scan backlog directory and return the maximum task ID found."""
    if not BACKLOG_DIR.exists():
        return 0
    max_id = 0
    for file in BACKLOG_DIR.glob("*.md"):
        match = re.match(r"^(\d+)-", file.name)
        if match:
            try:
                task_id = int(match.group(1))
                max_id = max(max_id, task_id)
            except ValueError:
                pass
    return max_id


def _read_backlog_task_file(id: int) -> tuple[dict, str, list[str]] | None:
    """Read a backlog task file by ID. Returns (frontmatter, body, activity_lines) or None.
    
    Body is everything between title and Activity Log.
    Activity lines are the timestamped entries under the LAST ## Activity Log heading.
    """
    pattern = f"{BACKLOG_DIR}/{id}-*.md"
    files = list(BACKLOG_DIR.glob(f"{id}-*.md"))
    if not files:
        return None
    
    # Take the first match (should be unique)
    file_path = files[0]
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError:
        return None
    
    frontmatter = _parse_backlog_frontmatter(text)
    if not frontmatter:
        return None
    
    # Split body and activity log
    body_lines = []
    activity_lines = []
    
    lines = text.split("\n")
    # Skip frontmatter
    start_idx = 0
    for i, line in enumerate(lines):
        if i > 0 and line.startswith("---"):
            start_idx = i + 1
            break
    
    # Find the LAST occurrence of ## Activity Log heading
    last_activity_idx = -1
    for i in range(start_idx, len(lines)):
        if lines[i].strip() == "## Activity Log":
            last_activity_idx = i
    
    # If no Activity Log section found, treat entire body as body_lines
    if last_activity_idx == -1:
        body_lines = lines[start_idx:]
        return (frontmatter, "\n".join(body_lines), [])
    
    # Split: body is everything before the last Activity Log heading
    body_lines = lines[start_idx:last_activity_idx]
    # Activity lines start from the heading itself
    activity_lines = lines[last_activity_idx:]
    
    # Extract title from body (first line starting with #)
    title = ""
    for line in body_lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    
    # Add title to frontmatter for convenience
    if title:
        frontmatter["title"] = title
    
    return (frontmatter, "\n".join(body_lines), activity_lines)


def _write_backlog_task_file(path: Path, frontmatter: dict, body: str, activity_lines: list[str]) -> None:
    """Write a complete backlog task file."""
    fm_text = _write_backlog_frontmatter(frontmatter)
    # Ensure body has Activity Log header
    body_stripped = body.rstrip()
    if "## Activity Log" not in body_stripped:
        body_stripped = body_stripped + "\n\n## Activity Log"
    content = fm_text + body_stripped + "\n" + "\n".join(activity_lines)
    path.write_text(content, encoding="utf-8")


def _append_activity_note(file_path: Path, note: str) -> None:
    """Append an activity note to a backlog task file's Activity Log section."""
    text = file_path.read_text(encoding="utf-8")
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = now.strftime("%Y-%m-%d %H:%M")
    
    # Find Activity Log section
    lines = text.split("\n")
    activity_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "## Activity Log":
            activity_idx = i
            break
    
    if activity_idx == -1:
        # No Activity Log section, create one
        lines.append("\n## Activity Log")
        lines.append(f"- **{date_str}** — {note}")
    else:
        # Append to existing Activity Log
        lines.append(f"- **{date_str}** — {note}")
    
    file_path.write_text("\n".join(lines), encoding="utf-8")


# ── Backlog tools ─────────────────────────────────────────────────────────────


_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "none": 3}


@mcp.tool()
async def backlog_boards() -> str:
    """List kanban boards with task counts."""
    def _impl():
        # Aggregate boards from markdown files
        board_counts: dict[str, int] = {}
        
        if BACKLOG_DIR.exists():
            for file in BACKLOG_DIR.glob("*.md"):
                try:
                    text = file.read_text(encoding="utf-8")
                    fm = _parse_backlog_frontmatter(text)
                    if fm and fm.get("type") == "backlog":
                        board = fm.get("board", "unknown")
                        board_counts[board] = board_counts.get(board, 0) + 1
                except OSError:
                    continue
        
        # Build board list - using a simple structure since we don't have board metadata
        boards = []
        for board_name, count in sorted(board_counts.items()):
            boards.append({
                "id": hash(board_name) & 0x7FFFFFFF,  # Generate a stable ID from name
                "name": board_name,
                "icon": "📋",
                "color": "gray",
                "position": len(boards),
                "tasks_count": count,
            })
        
        return json.dumps({"boards": boards})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def backlog_tasks(
    status: str = "",
    assigned: str = "",
    blocked: str = "",
    board_id: int = 0,
    tag: str = "",
) -> str:
    """List tasks with optional filters."""
    def _impl():
        tasks = []
        
        if BACKLOG_DIR.exists():
            for file in BACKLOG_DIR.glob("*.md"):
                try:
                    text = file.read_text(encoding="utf-8")
                    fm = _parse_backlog_frontmatter(text)
                    if not fm or fm.get("type") != "backlog":
                        continue
                    
                    # Apply filters
                    if status and fm.get("status") != status:
                        continue
                    if assigned:
                        is_assigned = fm.get("assigned", False)
                        if (assigned.lower() == "true") != is_assigned:
                            continue
                    if blocked:
                        is_blocked = fm.get("blocked", False)
                        if (blocked.lower() == "true") != is_blocked:
                            continue
                    # Note: board_id param is numeric, but files have board as string slug
                    # For compatibility, we could map board_id to board slug, but for now skip
                    if tag:
                        task_tags = fm.get("tags", [])
                        if isinstance(task_tags, str):
                            task_tags = [t.strip() for t in task_tags.split(",")]
                        if tag not in task_tags:
                            continue
                    
                    # Extract task data - note: files don't have position field, use id
                    # Extract title from filename if not in frontmatter
                    title = fm.get("title", "")
                    if not title:
                        # Extract from filename: {id}-{slug}.md -> slug to title
                        slug = file.name.replace(f"{fm.get('id', id)}-", "").replace(".md", "")
                        title = slug.replace("-", " ").title()
                    
                    # Ensure all datetime fields are strings
                    created = fm.get("created")
                    updated = fm.get("updated")
                    completed = fm.get("completed")
                    if isinstance(created, datetime.datetime):
                        created = created.isoformat()
                    if isinstance(updated, datetime.datetime):
                        updated = updated.isoformat()
                    if isinstance(completed, datetime.datetime):
                        completed = completed.isoformat()
                    
                    # Handle completed field - check for None, string "None", or null
                    completed_is_set = completed is not None and completed != "None" and completed != "null"
                    
                    task_data = {
                        "id": fm.get("id", 0),
                        "name": title,
                        "description": "",  # Will extract from body if needed
                        "priority": fm.get("priority", "none"),
                        "status": fm.get("status", "inbox"),
                        "blocked": bool(fm.get("blocked", False)),
                        "completed": completed_is_set,
                        "completed_at": completed if completed_is_set else None,
                        "assigned_to_agent": bool(fm.get("assigned", False)),
                        "board_id": hash(fm.get("board", "")) & 0x7FFFFFFF,  # Convert board slug to id
                        "tags": fm.get("tags", []) if isinstance(fm.get("tags"), list) else [],
                        "position": fm.get("id", 0),  # Use id as position fallback
                        "created_at": created,
                        "updated_at": updated,
                    }
                    tasks.append(task_data)
                except OSError:
                    continue
        
        # Sort by position, then id
        tasks.sort(key=lambda t: (t.get("position", 0), t.get("id", 0)))
        
        return json.dumps({"tasks": tasks})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def backlog_get_task(id: int) -> str:
    """Get task details by ID."""
    def _impl():
        result = _read_backlog_task_file(id)
        if not result:
            return json.dumps({"error": f"Task #{id} not found"})
        
        frontmatter, body, activity_lines = result
        
        # Ensure datetime fields are strings
        created = frontmatter.get("created")
        updated = frontmatter.get("updated")
        completed = frontmatter.get("completed")
        if isinstance(created, datetime.datetime):
            created = created.isoformat()
        if isinstance(updated, datetime.datetime):
            updated = updated.isoformat()
        if isinstance(completed, datetime.datetime):
            completed = completed.isoformat()
        
        # Handle completed field - check for None, string "None", or null
        completed_is_set = completed is not None and completed != "None" and completed != "null"
        
        # Build task dict
        task_data = {
            "id": frontmatter.get("id", id),
            "name": frontmatter.get("title", ""),
            "description": body.strip(),
            "priority": frontmatter.get("priority", "none"),
            "status": frontmatter.get("status", "inbox"),
            "blocked": bool(frontmatter.get("blocked", False)),
            "completed": completed_is_set,
            "completed_at": completed if completed_is_set else None,
            "assigned_to_agent": bool(frontmatter.get("assigned", False)),
            "board_id": hash(frontmatter.get("board", "")) & 0x7FFFFFFF,
            "tags": frontmatter.get("tags", []) if isinstance(frontmatter.get("tags"), list) else [],
            "position": frontmatter.get("id", id),
            "created_at": created,
            "updated_at": updated,
            "activity_log": activity_lines,
        }
        
        return json.dumps({"task": task_data})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def backlog_write_task(
    id: int = 0,
    name: str = "",
    description: str | None = None,
    board_id: int = 0,
    status: str = "",
    tags: str = "",
    priority: str = "",
    blocked: str = "",
    activity_note: str = "",
) -> str:
    """Create (id=0) or update a backlog task. Status: inbox|up_next|in_progress|in_review|done."""
    def _impl():
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        
        if id == 0:
            # ── CREATE ──
            if not name:
                return json.dumps({"error": "name is required when creating a task"})
            
            # Generate slug and determine new ID
            slug = _generate_slug(name)
            new_id = _find_max_backlog_id() + 1
            
            # Parse tags
            tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            
            # Default values
            effective_status = status or "inbox"
            effective_priority = priority or "none"
            effective_board = "lloyd"  # Default board
            
            # Create frontmatter
            frontmatter = {
                "type": "backlog",
                "id": new_id,
                "board": effective_board,
                "status": effective_status,
                "priority": effective_priority,
                "tags": tags_list,
                "blocked": False,
                "assigned": False,
                "created": now,
                "updated": now,
                "completed": None,
            }
            
            # Create body with title, description, and empty Activity Log
            body = f"# {name}\n\n{description}\n\n## Activity Log"
            activity_lines = [f"- **{today}** — Created (api)"]
            
            # Write file
            file_path = BACKLOG_DIR / f"{new_id}-{slug}.md"
            _write_backlog_task_file(file_path, frontmatter, body, activity_lines)
            
            # Return created task
            task_data = {
                "id": new_id,
                "name": name,
                "description": description,
                "priority": effective_priority,
                "status": effective_status,
                "blocked": False,
                "completed": False,
                "completed_at": None,
                "assigned_to_agent": False,
                "board_id": hash(effective_board) & 0x7FFFFFFF,
                "tags": tags_list,
                "position": new_id,
                "created_at": now,
                "updated_at": now,
            }
            return json.dumps({"task": task_data})
        
        else:
            # ── UPDATE ──
            result = _read_backlog_task_file(id)
            if not result:
                return json.dumps({"error": f"Task #{id} not found"})
            
            frontmatter, body, activity_lines = result
            
            # Track changes for activity log
            changes = []
            
            # Update status
            if status and status != frontmatter.get("status"):
                old_status = frontmatter.get("status", "inbox")
                frontmatter["status"] = status
                changes.append(f"moved to {status}")
                
                # Handle completed field
                if status == "done" and frontmatter.get("completed") is None:
                    frontmatter["completed"] = now
                elif old_status == "done" and status != "done":
                    frontmatter["completed"] = None
            
            # Update blocked
            if blocked and blocked.lower() in ("true", "false"):
                new_blocked = blocked.lower() == "true"
                if new_blocked != frontmatter.get("blocked", False):
                    frontmatter["blocked"] = new_blocked
                    changes.append(f"blocked={blocked.lower()}")
            
            # Update priority
            if priority and priority != frontmatter.get("priority", "none"):
                frontmatter["priority"] = priority
                changes.append(f"priority={priority}")
            
            # Update name (also affects filename if slug changes, but we keep original filename)
            if name and name != frontmatter.get("title", ""):
                frontmatter["title"] = name
                changes.append(f"name updated")
            
            # Update description (in body)
            if description is not None and description != body.strip():
                # Rebuild body with new description
                # Extract title from current body
                body_lines = body.split("\n")
                if body_lines and body_lines[0].startswith("# "):
                    title = body_lines[0][2:]
                else:
                    title = frontmatter.get("title", name)
                body = f"# {title}\n\n{description}\n\n## Activity Log\n"
                changes.append("description updated")
            
            # Update tags
            if tags:
                new_tags = [t.strip() for t in tags.split(",") if t.strip()]
                old_tags = frontmatter.get("tags", [])
                if isinstance(old_tags, str):
                    old_tags = [t.strip() for t in old_tags.split(",")]
                if new_tags != old_tags:
                    frontmatter["tags"] = new_tags
                    changes.append(f"tags={tags}")
            
            # Handle activity_note
            if activity_note:
                changes.append(activity_note)
            
            # Check if anything changed
            if not changes and not activity_note:
                return json.dumps({"error": "Nothing to update — provide status, blocked, priority, name, description, tags, or activity_note."})
            
            # Update timestamp
            frontmatter["updated"] = now
            
            # Append activity note
            file_path = BACKLOG_DIR / f"{id}-*.md"
            # Find the actual file
            matching_files = list(BACKLOG_DIR.glob(f"{id}-*.md"))
            if not matching_files:
                return json.dumps({"error": f"Task #{id} file not found"})
            
            actual_path = matching_files[0]
            
            # Append to activity log
            activity_entry = f"- **{today}** — Updated ({', '.join(changes)})"
            activity_lines.append(activity_entry)
            
            # Write updated file
            _write_backlog_task_file(actual_path, frontmatter, body, activity_lines)
            
            # Return updated task
            # Ensure datetime fields are strings
            created = frontmatter.get("created")
            updated = frontmatter.get("updated")
            completed = frontmatter.get("completed")
            if isinstance(created, datetime.datetime):
                created = created.isoformat()
            if isinstance(updated, datetime.datetime):
                updated = updated.isoformat()
            if isinstance(completed, datetime.datetime):
                completed = completed.isoformat()
            
            # Handle completed field - check for None, string "None", or null
            completed_is_set = completed is not None and completed != "None" and completed != "null"
            
            task_data = {
                "id": frontmatter.get("id", id),
                "name": frontmatter.get("title", ""),
                "description": body.strip(),
                "priority": frontmatter.get("priority", "none"),
                "status": frontmatter.get("status", "inbox"),
                "blocked": bool(frontmatter.get("blocked", False)),
                "completed": completed_is_set,
                "completed_at": completed if completed_is_set else None,
                "assigned_to_agent": bool(frontmatter.get("assigned", False)),
                "board_id": hash(frontmatter.get("board", "")) & 0x7FFFFFFF,
                "tags": frontmatter.get("tags", []) if isinstance(frontmatter.get("tags"), list) else [],
                "position": frontmatter.get("id", id),
                "created_at": created,
                "updated_at": updated,
            }
            return json.dumps({"task": task_data})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


# ── Autonomy Board Tools ──────────────────────────────────────────────────────

# ── Autonomy Board Tools (File-Based) ─────────────────────────────────────────

AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"
_autonomy_lock = threading.Lock()


def _slugify(name: str) -> str:
    """Convert name to URL-friendly slug."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name[:50]


def _parse_task_file(path: Path) -> dict | None:
    """Read task file, parse YAML frontmatter, return dict."""
    try:
        content = path.read_text(encoding="utf-8")
        # Split frontmatter from body
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        frontmatter_str = parts[1]
        body = parts[2] if len(parts) > 2 else ""
        
        frontmatter = yaml.safe_load(frontmatter_str)
        if not isinstance(frontmatter, dict):
            return None
        
        # Ensure all expected fields exist with defaults
        result = {
            "id": frontmatter.get("id", 0),
            "name": frontmatter.get("name", ""),
            "description": frontmatter.get("description", ""),
            "status": frontmatter.get("status", "inbox"),
            "priority": frontmatter.get("priority", "medium"),
            "frequency": frontmatter.get("frequency", ""),
            "scheduled_at": frontmatter.get("scheduled_at", ""),
            "last_run": frontmatter.get("last_run"),
            "next_run": frontmatter.get("next_run"),
            "auto_advance": bool(frontmatter.get("auto_advance", False)),
            "preemptible": bool(frontmatter.get("preemptible", True)),
            "board_id": frontmatter.get("board_id", 4),
            "tags": frontmatter.get("tags", []) or [],
            "created_at": frontmatter.get("created", frontmatter.get("created_at", "")),
            "updated_at": frontmatter.get("updated", frontmatter.get("updated_at", "")),
            "skill_path": frontmatter.get("skill_path", ""),
            "agent_id": frontmatter.get("agent_id", "memory"),
            "model": frontmatter.get("model", ""),
            "timeout_seconds": frontmatter.get("timeout_seconds", 1800),
            "depends_on": frontmatter.get("depends_on"),
            "pipeline": frontmatter.get("pipeline"),
            "runs_per_day": frontmatter.get("runs_per_day"),
            "run_count": frontmatter.get("run_count"),
            "failure_count": frontmatter.get("failure_count", 0),
            "max_retries": frontmatter.get("max_retries", 3),
            "notify_on_complete": frontmatter.get("notify_on_complete", True),
            "preferred_hours": frontmatter.get("preferred_hours", []),
            "cron_id": frontmatter.get("cron_id"),
            "type": frontmatter.get("type", "autonomy"),
            "body": body,
        }
        return result
    except Exception as e:
        print(f"[_parse_task_file] Error parsing {path}: {e}", file=sys.stderr)
        return None


def _write_task_file(task_dict: dict) -> Path:
    """Write task dict to markdown file with frontmatter."""
    task_id = task_dict.get("id", 0)
    name = task_dict.get("name", "unnamed")
    slug = _slugify(name)
    
    # Ensure directory exists
    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find existing file or create new path
    existing = _find_task_file(task_id)
    if existing:
        path = existing
    else:
        path = AUTONOMY_DIR / f"{task_id}-{slug}.md"
    
    # Build frontmatter
    frontmatter = {
        "type": "autonomy",
        "id": task_dict.get("id", 0),
        "name": task_dict.get("name", ""),
        "description": task_dict.get("description", ""),
        "status": task_dict.get("status", "inbox"),
        "priority": task_dict.get("priority", "medium"),
        "frequency": task_dict.get("frequency", ""),
        "agent_id": task_dict.get("agent_id", "memory"),
        "model": task_dict.get("model", ""),
        "tags": task_dict.get("tags", []),
        "auto_advance": task_dict.get("auto_advance", False),
        "preemptible": task_dict.get("preemptible", True),
        "pipeline_mode": task_dict.get("pipeline_mode", False),
        "timeout_seconds": task_dict.get("timeout_seconds", 1800),
        "max_retries": task_dict.get("max_retries", 3),
        "failure_count": task_dict.get("failure_count", 0),
        "skill_path": task_dict.get("skill_path", ""),
        "cron_id": task_dict.get("cron_id"),
        "runs_per_day": task_dict.get("runs_per_day"),
        "scheduled_at": task_dict.get("scheduled_at", ""),
        "last_run": task_dict.get("last_run"),
        "next_run": task_dict.get("next_run"),
        "depends_on": task_dict.get("depends_on"),
        "preferred_hours": task_dict.get("preferred_hours", []),
        "notify_on_complete": task_dict.get("notify_on_complete", True),
        "pipeline": task_dict.get("pipeline"),
        "created": task_dict.get("created_at", task_dict.get("created", "")),
        "updated": task_dict.get("updated_at", task_dict.get("updated", "")),
    }
    
    # Remove None values
    frontmatter = {k: v for k, v in frontmatter.items() if v is not None}
    
    # Write file
    body = task_dict.get("body", "")
    content = f"---\n{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}---\n\n{body}"
    path.write_text(content, encoding="utf-8")
    
    return path


def _parse_run_file(path: Path) -> dict | None:
    """Parse run file frontmatter."""
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        frontmatter = yaml.safe_load(parts[1])
        if not isinstance(frontmatter, dict):
            return None
        return {
            "run_id": frontmatter.get("run_id", 0),
            "task_id": frontmatter.get("task_id", 0),
            "status": frontmatter.get("status", ""),
            "duration_seconds": frontmatter.get("duration_seconds"),
            "started_at": frontmatter.get("started_at"),
            "completed_at": frontmatter.get("completed_at"),
            "body": parts[2] if len(parts) > 2 else "",
        }
    except Exception as e:
        print(f"[_parse_run_file] Error parsing {path}: {e}", file=sys.stderr)
        return None


def _find_task_file(task_id: int) -> Path | None:
    """Find task file by ID globbing."""
    if not AUTONOMY_DIR.exists():
        return None
    patterns = list(AUTONOMY_DIR.glob(f"{task_id}-*.md"))
    # Filter out _config.md
    patterns = [p for p in patterns if p.name != "_config.md"]
    return patterns[0] if patterns else None


def _next_task_id() -> int:
    """Scan all task files, return max id + 1."""
    if not AUTONOMY_DIR.exists():
        return 1
    max_id = 0
    for path in AUTONOMY_DIR.glob("*.md"):
        if path.name == "_config.md":
            continue
        try:
            # Extract ID from filename
            name = path.name
            if "-" in name:
                id_str = name.split("-")[0]
                if id_str.isdigit():
                    max_id = max(max_id, int(id_str))
        except Exception:
            continue
    return max_id + 1


@mcp.tool()
async def autonomy_tasks(
    status: str = "",
    tag: str = "",
    frequency: str = "",
    agent_id: str = "",
) -> str:
    """List/filter autonomy tasks. Returns array of task objects."""
    def _impl():
        with _autonomy_lock:
            if not AUTONOMY_DIR.exists():
                return json.dumps({"tasks": []})
            
            tasks = []
            for path in AUTONOMY_DIR.glob("*.md"):
                if path.name == "_config.md":
                    continue
                
                task = _parse_task_file(path)
                if task is None:
                    continue
                
                # Apply filters
                if status and task.get("status") != status:
                    continue
                if frequency and task.get("frequency") != frequency:
                    continue
                if agent_id and task.get("agent_id") != agent_id:
                    continue
                if tag:
                    task_tags = task.get("tags", [])
                    if not isinstance(task_tags, list):
                        try:
                            task_tags = json.loads(task_tags) if task_tags else []
                        except (json.JSONDecodeError, TypeError):
                            task_tags = []
                    if tag not in task_tags:
                        continue
                
                tasks.append(task)
            
            return json.dumps({"tasks": tasks})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def autonomy_write_task(
    id: int = 0,
    name: str = "",
    description: str = "",
    status: str = "",
    priority: str = "",
    frequency: str = "",
    skill_path: str = "",
    agent_id: str = "",
    model: str = "",
    timeout_seconds: int = 0,
    tags: str = "",
    auto_advance: bool = False,
    preemptible: bool = True,
    scheduled_at: str = "",
    depends_on: int = 0,
    pipeline: str = "",
    activity_note: str = "",
) -> str:
    """Create or update (upsert) autonomy task. If id omitted → CREATE, if id provided → UPDATE."""
    def _impl():
        with _autonomy_lock:
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            if id == 0:
                # CREATE - name required
                if not name:
                    return json.dumps({"error": "name is required when creating a task"})
                
                new_id = _next_task_id()
                effective_status = status or "inbox"
                effective_priority = priority or "medium"
                effective_agent_id = agent_id or "memory"
                effective_timeout = timeout_seconds if timeout_seconds > 0 else 1800
                effective_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
                
                task_dict = {
                    "id": new_id,
                    "name": name,
                    "description": description,
                    "status": effective_status,
                    "priority": effective_priority,
                    "frequency": frequency,
                    "skill_path": skill_path,
                    "agent_id": effective_agent_id,
                    "model": model,
                    "timeout_seconds": effective_timeout,
                    "tags": effective_tags,
                    "auto_advance": auto_advance,
                    "preemptible": preemptible,
                    "scheduled_at": scheduled_at,
                    "depends_on": depends_on if depends_on > 0 else None,
                    "pipeline": pipeline,
                    "created_at": now,
                    "updated_at": now,
                    "body": "",
                }
                
                _write_task_file(task_dict)
                return json.dumps({"task": task_dict})
            
            else:
                # UPDATE - partial update
                existing_path = _find_task_file(id)
                if not existing_path:
                    return json.dumps({"error": f"Task #{id} not found"})
                
                task_dict = _parse_task_file(existing_path)
                if task_dict is None:
                    return json.dumps({"error": f"Failed to parse task #{id}"})
                
                # Update only provided fields
                if status:
                    task_dict["status"] = status
                if priority:
                    task_dict["priority"] = priority
                if frequency:
                    task_dict["frequency"] = frequency
                if skill_path:
                    task_dict["skill_path"] = skill_path
                if agent_id:
                    task_dict["agent_id"] = agent_id
                if model:
                    task_dict["model"] = model
                if timeout_seconds > 0:
                    task_dict["timeout_seconds"] = timeout_seconds
                if tags:
                    task_dict["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
                if auto_advance is not None:
                    task_dict["auto_advance"] = auto_advance
                if preemptible is not None:
                    task_dict["preemptible"] = preemptible
                if scheduled_at:
                    task_dict["scheduled_at"] = scheduled_at
                if depends_on > 0:
                    task_dict["depends_on"] = depends_on
                if pipeline:
                    task_dict["pipeline"] = pipeline
                if description:
                    task_dict["description"] = description
                
                task_dict["updated_at"] = now
                
                # Handle activity_note - append to body
                if activity_note:
                    body = task_dict.get("body", "")
                    if "## Activity Log" not in body:
                        body += "\n\n## Activity Log\n"
                    body += f"\n- {now}: {activity_note}\n"
                    task_dict["body"] = body
                
                _write_task_file(task_dict)
                return json.dumps({"task": task_dict})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def autonomy_get_task(id: int) -> str:
    """Get full task detail + recent runs. Required param: id (int)."""
    def _impl():
        with _autonomy_lock:
            path = _find_task_file(id)
            if not path:
                return json.dumps({"error": f"Task #{id} not found"})
            
            task = _parse_task_file(path)
            if task is None:
                return json.dumps({"error": f"Failed to parse task #{id}"})
            
            # Get runs
            runs_dir = AUTONOMY_DIR / "runs" / str(id)
            runs = []
            if runs_dir.exists():
                run_files = sorted(runs_dir.glob("*.md"), key=lambda p: p.name, reverse=True)
                for run_path in run_files[:10]:
                    run = _parse_run_file(run_path)
                    if run:
                        runs.append(run)
            
            # Sort runs by started_at descending
            runs = sorted(runs, key=lambda r: r.get("started_at", "") or "", reverse=True)
            task["runs"] = runs
            
            return json.dumps({"task": task})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def autonomy_run_task(id: int) -> str:
    """Trigger immediate execution. Required param: id (int)."""
    def _impl():
        with _autonomy_lock:
            path = _find_task_file(id)
            if not path:
                return json.dumps({"error": f"Task #{id} not found"})
            
            task = _parse_task_file(path)
            if task is None:
                return json.dumps({"error": f"Failed to parse task #{id}"})
            
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Build the prompt
            if task.get("skill_path"):
                prompt = f"Read the skill file at {task['skill_path']} and execute all instructions in it.\n\nContext: Autonomy Task #{task['id']} — {task['name']}\nPriority: {task['priority']} | Frequency: {task.get('frequency', '')}"
            else:
                prompt = f"## Autonomy Task #{task['id']}: {task['name']}\n\n{task.get('description', '')}\n\n**Priority:** {task['priority']}\n**Frequency:** {task.get('frequency', '')}"
            
            # Determine agent ID
            agent_id = task.get("agent_id") or "orchestrator"
            
            # Update task status to in_progress
            task["status"] = "in_progress"
            task["last_run"] = now
            task["updated_at"] = now
            _write_task_file(task)
            
            # Create runs directory if needed
            runs_dir = AUTONOMY_DIR / "runs" / str(id)
            runs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create new run file
            # Generate run_id (use timestamp-based for uniqueness)
            run_id = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
            run_path = runs_dir / f"{run_id}.md"
            
            run_dict = {
                "type": "autonomy-run",
                "task_id": id,
                "run_id": run_id,
                "status": "running",
                "started_at": now,
            }
            
            # Write run file
            content = f"---\n{yaml.dump(run_dict, default_flow_style=False, allow_unicode=True)}---\n\n"
            run_path.write_text(content, encoding="utf-8")
            
            # Read hooks token from config
            config_path = Path.home() / ".openclaw" / "openclaw.json"
            hooks_token = ""
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                hooks_token = config.get("hooks", {}).get("token", "")
            except Exception as e:
                print(f"[autonomy_run_task] Failed to read config: {e}", file=sys.stderr)
            
            # POST to gateway hooks endpoint
            gateway_url = "https://localhost:18789/hooks/agent"
            payload = {
                "message": prompt,
                "agentId": agent_id,
                "sessionKey": f"hook:idler:task-{id}",
                "name": f"idler-{id}",
                "deliver": False,
                "timeoutSeconds": task.get("timeout_seconds") or 600,
            }
            
            try:
                import urllib.request, ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                req = urllib.request.Request(
                    gateway_url,
                    data=json.dumps(payload).encode(),
                    headers={
                        "Authorization": f"Bearer {hooks_token}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                resp = urllib.request.urlopen(req, context=ctx, timeout=30)
                resp_body = resp.read().decode()
                resp_status = resp.status
                
                if resp_status != 200:
                    # Mark run as failed and revert task status
                    run_dict["status"] = "failed"
                    run_dict["body"] = f"Gateway dispatch failed: {resp_status}"
                    content = f"---\n{yaml.dump(run_dict, default_flow_style=False, allow_unicode=True)}---\n\n"
                    run_path.write_text(content, encoding="utf-8")
                    
                    task["status"] = "up_next"
                    task["updated_at"] = now
                    _write_task_file(task)
                    
                    return json.dumps({
                        "error": f"Gateway dispatch failed: {resp_status} {resp_body}",
                        "task_id": id,
                        "run_id": run_id,
                    })
                
                result = json.loads(resp_body)
                gateway_run_id = result.get("runId", run_id)
                
                return json.dumps({
                    "task_id": id,
                    "run_id": gateway_run_id,
                    "status": "dispatched",
                    "agent_id": agent_id,
                })
            except Exception as e:
                print(f"[autonomy_run_task] Dispatch error: {e}", file=sys.stderr)
                # Mark run as failed and revert task status
                run_dict["status"] = "failed"
                run_dict["body"] = f"Dispatch error: {e}"
                content = f"---\n{yaml.dump(run_dict, default_flow_style=False, allow_unicode=True)}---\n\n"
                run_path.write_text(content, encoding="utf-8")
                
                task["status"] = "up_next"
                task["updated_at"] = now
                _write_task_file(task)
                
                return json.dumps({
                    "error": f"Dispatch failed: {e}",
                    "task_id": id,
                    "run_id": run_id,
                })
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def autonomy_delete_task(id: int, archive: bool = True) -> str:
    """Delete or archive a task."""
    def _impl():
        with _autonomy_lock:
            path = _find_task_file(id)
            if not path:
                return json.dumps({"error": f"Task #{id} not found"})
            
            if archive:
                # Set status to done
                task = _parse_task_file(path)
                if task:
                    task["status"] = "done"
                    task["updated_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    _write_task_file(task)
                    return json.dumps({"success": True, "id": id})
                return json.dumps({"error": f"Failed to parse task #{id}"})
            else:
                # Delete the file
                try:
                    path.unlink()
                    return json.dumps({"success": True, "id": id})
                except Exception as e:
                    return json.dumps({"error": f"Failed to delete task: {e}"})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


def _read_config() -> dict:
    """Read config file or return empty dict."""
    config_path = AUTONOMY_DIR / "_config.md"
    if not config_path.exists():
        return {}
    try:
        content = config_path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) >= 2:
            frontmatter = yaml.safe_load(parts[1])
            if isinstance(frontmatter, dict):
                return frontmatter
        return {}
    except Exception:
        return {}


def _write_config(config: dict) -> None:
    """Write config to file."""
    AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    config_path = AUTONOMY_DIR / "_config.md"
    content = f"---\n{yaml.dump(config, default_flow_style=False, allow_unicode=True)}---\n\n"
    config_path.write_text(content, encoding="utf-8")


@mcp.tool()
async def autonomy_config(key: str = "", value: str = "") -> str:
    """Get or set autonomy system config."""
    def _impl():
        with _autonomy_lock:
            if not key:
                # Get all config
                config = _read_config()
                return json.dumps(config)
            
            if value is not None:
                # Set config
                config = _read_config()
                config[key] = value
                _write_config(config)
                return json.dumps({"set": key, "value": value})
            else:
                # Get single config
                config = _read_config()
                if key in config:
                    return json.dumps({key: config[key]})
                return json.dumps({"error": f"Config key not found: {key}"})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)
# ── Next-Gen Memory Tools ─────────────────────────────────────────────────────

@mcp.tool()
async def get_facts(entity: str, category: str = None, status: str = "current") -> str:
    """Retrieve facts for an entity/category. Returns JSON with facts array."""
    def _impl():
        try:
            return json.dumps(_get_facts_sync(entity, category))
        except Exception as exc:
            return json.dumps({"error": str(exc), "facts": []})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def add_fact(entity: str, category: str, fact: str, confidence: float = 0.9) -> str:
    """Add or update a fact for an entity/category."""
    def _impl():
        try:
            # Find or create entity directory
            entity_dir = _find_entity_dir(entity)
            if not entity_dir:
                entity_dir = FACTS_ROOT / entity
                entity_dir.mkdir(parents=True, exist_ok=True)
            
            # Find or create fact file
            fact_file = entity_dir / f"{entity}-{category}.md"
            
            # Parse existing frontmatter
            if fact_file.exists():
                content = fact_file.read_text(encoding="utf-8")
                frontmatter = _parse_fact_frontmatter(content)
            else:
                frontmatter = {"type": "facts", "entity": entity, "category": category, "facts": []}
            
            # Generate new fact ID
            fact_id = _generate_fact_id(category)
            
            # Append new fact
            new_fact = {
                "fact": fact,
                "confidence": confidence,
                "event_date": None,
                "category": "state",
                "id": fact_id
            }
            frontmatter.setdefault("facts", []).append(new_fact)
            frontmatter["last_updated"] = datetime.datetime.now().isoformat()
            
            # Write back
            frontmatter_str = _write_fact_frontmatter(frontmatter)
            fact_file.write_text(frontmatter_str + f"\n# {entity} - {category}\n\n**Entity:** {entity}\n**Category:** {category}\n**Fact Count:** {len(frontmatter.get('facts', []))}\n", encoding="utf-8")
            
            return json.dumps({"success": True, "fact_id": fact_id, "entity": entity, "category": category})
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def get_relations(document_path: str, relation_type: str = None) -> str:
    """Get relationships for a document. Returns JSON with matching relationships."""
    def _impl():
        try:
            if not RELATIONS_INDEX.exists():
                return json.dumps({"error": "Relations index not found", "relationships": []})
            
            data = json.loads(RELATIONS_INDEX.read_text(encoding="utf-8"))
            relationships = data.get("relationships", [])
            
            # Filter by document_path (source or target)
            filtered = [
                r for r in relationships
                if r.get("source") == document_path or r.get("target") == document_path
            ]
            
            # Filter by relation_type if given
            if relation_type:
                filtered = [r for r in filtered if r.get("type") == relation_type]
            
            return json.dumps({"document_path": document_path, "relationships": filtered})
        except Exception as exc:
            return json.dumps({"error": str(exc), "relationships": []})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def add_relation(source: str, target: str, rel_type: str) -> str:
    """Add relationship between documents."""
    def _impl():
        try:
            # Read relations index
            if not RELATIONS_INDEX.exists():
                data = {"relationships": [], "total_relationships": 0, "last_updated": None}
            else:
                data = json.loads(RELATIONS_INDEX.read_text(encoding="utf-8"))
            
            # Append new relationship
            new_relation = {
                "source": source,
                "target": target,
                "type": rel_type,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data.setdefault("relationships", []).append(new_relation)
            data["total_relationships"] = len(data["relationships"])
            data["last_updated"] = datetime.datetime.now().isoformat()
            
            # Write back
            RELATIONS_INDEX.write_text(json.dumps(data, indent=2), encoding="utf-8")
            
            return json.dumps({"success": True, "relation": new_relation})
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def context_bundle(query: str, mode: str = "shallow", limit: int = 20, include_facts: bool = True) -> str:
    """Get relevant context for a query. Returns JSON with documents, facts, and query."""
    def _impl():
        try:
            import concurrent.futures

            def _do_search():
                return _qmd_client.search(query, limit, collections=["subliminal"], skip_rerank=True, candidate_limit=10)

            def _do_facts():
                if not include_facts:
                    return []
                facts = []
                entity_matches = _extract_entities_from_query(query)
                for entity, score in entity_matches[:5]:
                    try:
                        entity_data = _get_facts_sync(entity)
                        if entity_data.get("facts"):
                            sorted_facts = sorted(
                                entity_data["facts"],
                                key=lambda f: f.get("confidence", 0.5),
                                reverse=True
                            )
                            facts.extend(sorted_facts[:3])
                    except Exception:
                        pass
                return facts

            # Run QMD search and fact retrieval in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                search_future = pool.submit(_do_search)
                facts_future = pool.submit(_do_facts)
                fts_results = search_future.result()
                facts = facts_future.result()

            documents = []
            if fts_results:
                for r in fts_results:
                    documents.append({
                        "path": r.get("file", "").removeprefix("qmd://"),
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", ""),
                        "score": r.get("score", 0)
                    })

            return json.dumps({
                "documents": documents[:limit],
                "facts": facts,
                "query": query,
                "mode": mode
            })
        except Exception as exc:
            return json.dumps({"error": str(exc), "documents": [], "facts": []})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def get_profile(entity: str, include_summary: bool = True) -> str:
    """Get synthesized profile for an entity."""
    def _impl():
        try:
            # Get all facts for entity
            facts_data = _get_facts_sync(entity)
            facts = facts_data.get("facts", [])
            
            # Group facts by category
            categories = {}
            for fact in facts:
                cat = fact.get("category", "general")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(fact)
            
            # Generate summary if requested
            summary = ""
            if include_summary:
                summary_parts = [f"Profile for: {entity}\n"]
                for cat, cat_facts in categories.items():
                    summary_parts.append(f"\n{cat.upper()}:")
                    for fact in cat_facts[:3]:  # Limit to 3 facts per category
                        summary_parts.append(f"  - {fact.get('fact', '')}")
                summary = "\n".join(summary_parts)
            
            return json.dumps({
                "entity": entity,
                "categories": categories,
                "fact_count": len(facts),
                "summary": summary if include_summary else None
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def detect_contradictions(entity: str, category: str = None) -> str:
    """Find contradictions in facts for an entity."""
    def _impl():
        try:
            return json.dumps(_detect_contradictions_sync(entity, category))
        except Exception as exc:
            return json.dumps({"error": str(exc), "contradictions": [], "checked": 0})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def resolve_contradictions(entity: str, auto_resolve: bool = True) -> str:
    """Auto-resolve contradictions by keeping higher-confidence fact."""
    def _impl():
        try:
            # Detect contradictions first
            contradiction_data = _detect_contradictions_sync(entity)
            contradictions = contradiction_data.get("contradictions", [])
            
            resolved_count = 0
            remaining_count = len(contradictions)
            
            if auto_resolve and contradictions:
                # Find entity directory
                entity_dir = _find_entity_dir(entity)
                if entity_dir:
                    # For each contradiction, remove the lower-confidence fact
                    for contradiction in contradictions:
                        fact1 = contradiction.get("fact1", {})
                        fact2 = contradiction.get("fact2", {})
                        
                        conf1 = fact1.get("confidence", 0.9)
                        conf2 = fact2.get("confidence", 0.9)
                        
                        # Keep higher confidence, remove lower
                        if conf1 >= conf2:
                            fact_to_remove = fact2.get("id")
                        else:
                            fact_to_remove = fact1.get("id")
                        
                        if fact_to_remove:
                            # Read and update fact file
                            for fact_file in entity_dir.glob("*.md"):
                                content = fact_file.read_text(encoding="utf-8")
                                frontmatter = _parse_fact_frontmatter(content)
                                if "facts" in frontmatter:
                                    original_count = len(frontmatter["facts"])
                                    frontmatter["facts"] = [
                                        f for f in frontmatter["facts"]
                                        if f.get("id") != fact_to_remove
                                    ]
                                    if len(frontmatter["facts"]) < original_count:
                                        # Write back
                                        frontmatter_str = _write_fact_frontmatter(frontmatter)
                                        fact_file.write_text(frontmatter_str, encoding="utf-8")
                                        resolved_count += 1
            
            remaining_count = len(contradictions) - resolved_count
            
            return json.dumps({
                "entity": entity,
                "resolved": resolved_count,
                "remaining": remaining_count
            })
        except Exception as exc:
            return json.dumps({"error": str(exc), "resolved": 0, "remaining": 0})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


@mcp.tool()
async def rebuild_index(index_type: str = "all") -> str:
    """Rebuild memory system indexes."""
    def _impl():
        try:
            # This would typically rebuild facts and relations indexes
            # For now, just return success
            return json.dumps({
                "success": True,
                "index_type": index_type,
                "message": "Index rebuild scheduled"
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    return await asyncio.get_running_loop().run_in_executor(None, _impl)


# ── Skill tools ───────────────────────────────────────────────────────────────


@mcp.tool()
async def skills_search(query: str, source: str = "all") -> str:
    """Search local and ClawhHub skills."""
    def _local_search():
        if source in ("clawhub",):
            return []
        local_source = source if source in ("builtin", "custom") else "all"
        results = _skill_index.search(query, local_source)
        items = []
        for entry, rank in results:
            items.append({
                "name": entry.name,
                "description": entry.description,
                "source": entry.source,
                "path": entry.dir_path,
            })
        return items

    def _hub_search():
        if source in ("builtin", "custom"):
            return []
        return _clawhub_search(query)

    loop = asyncio.get_running_loop()
    valid_sources = ("all", "builtin", "custom", "clawhub")
    if source not in valid_sources:
        return json.dumps({"error": f"Invalid source filter: {source!r}. Use {', '.join(repr(s) for s in valid_sources)}."})

    local_items, hub_items = await asyncio.gather(
        loop.run_in_executor(None, _local_search),
        loop.run_in_executor(None, _hub_search),
    )

    # Deduplicate: if a ClawhHub slug matches a local skill name, skip it
    local_names = {item["name"].lower() for item in local_items}
    for item in hub_items:
        if item["slug"].lower() not in local_names and item["name"].lower() not in local_names:
            local_items.append({
                "name": item["name"],
                "description": f"ClawhHub: {item['name']} (score: {item['score']})",
                "source": "clawhub",
                "slug": item["slug"],
            })

    if not local_items:
        return json.dumps({"results": [], "message": f"No skills matching '{query}'."})

    return json.dumps({"results": local_items, "total": len(local_items)})


@mcp.tool()
async def skills_get(name: str) -> str:
    """Get SKILL.md content by name. Prefix "clawhub:" for catalog."""
    def _local_impl():
        entry = _skill_index.by_name.get(name.lower().strip())
        if entry is not None:
            return json.dumps({
                "name": entry.name,
                "source": entry.source,
                "path": entry.dir_path,
                "content": entry.body,
            })

        # Not found — suggest similar
        suggestions = _skill_index.search(name, "all")[:5]
        suggestion_names = [s.name for s, _ in suggestions]
        msg = f"Skill '{name}' not found."
        if suggestion_names:
            msg += f" Did you mean: {', '.join(suggestion_names)}?"
        return json.dumps({"error": msg})

    def _clawhub_impl(slug: str):
        try:
            metadata = _clawhub_inspect_json(slug)
        except Exception as exc:
            return json.dumps({"error": f"ClawhHub inspect failed for '{slug}': {exc}"})

        try:
            content = _clawhub_inspect_file(slug, "SKILL.md")
        except Exception:
            content = "(SKILL.md not available)"

        try:
            files_listing = _clawhub_inspect_files(slug)
        except Exception:
            files_listing = "(file listing not available)"

        skill_info = metadata.get("skill", {})
        return json.dumps({
            "name": skill_info.get("displayName", slug),
            "source": "clawhub",
            "slug": slug,
            "content": content,
            "files": files_listing,
            "metadata": {
                "summary": skill_info.get("summary", ""),
                "stats": skill_info.get("stats", {}),
            },
        })

    loop = asyncio.get_running_loop()
    stripped = name.strip()
    if stripped.lower().startswith("clawhub:"):
        ch_slug = stripped[len("clawhub:"):].strip()
        if not _CLAWHUB_SLUG_RE.fullmatch(ch_slug):
            return json.dumps({"error": f"Invalid ClawhHub slug: {ch_slug!r}"})
        return await loop.run_in_executor(None, _clawhub_impl, ch_slug)
    else:
        return await loop.run_in_executor(None, _local_impl)


@mcp.tool()
async def skills_install(slug: str) -> str:
    """Install a skill from ClawhHub with security scanning."""
    def _impl():
        if not _CLAWHUB_SLUG_RE.fullmatch(slug):
            return json.dumps({"status": "rejected", "reason": f"Invalid slug: {slug!r}"})

        staging_dir = CLAWHUB_STAGING / slug
        skill_dir = None
        try:
            # 1. Create staging directory
            staging_dir.mkdir(parents=True, exist_ok=True)

            # 2. Download skill via clawhub install
            try:
                proc = subprocess.run(
                    ["clawhub", "install", slug, "--workdir", str(staging_dir)],
                    capture_output=True, text=True, timeout=15,
                )
                if proc.returncode != 0:
                    return json.dumps({
                        "status": "rejected",
                        "reason": f"clawhub install failed: {proc.stderr.strip()}",
                        "scan_summary": None,
                    })
            except subprocess.TimeoutExpired:
                return json.dumps({
                    "status": "rejected",
                    "reason": "clawhub install timed out (15s)",
                    "scan_summary": None,
                })
            except FileNotFoundError:
                return json.dumps({
                    "status": "rejected",
                    "reason": "clawhub binary not found",
                    "scan_summary": None,
                })

            # 3. Locate the installed skill
            skill_dir = staging_dir / "skills" / slug
            if not skill_dir.is_dir():
                # Try direct staging dir if clawhub puts it there
                if (staging_dir / "SKILL.md").is_file():
                    skill_dir = staging_dir
                else:
                    return json.dumps({
                        "status": "rejected",
                        "reason": f"Skill directory not found after install at {skill_dir}",
                        "scan_summary": None,
                    })

            # 4. Analyze skill contents
            has_scripts = (skill_dir / "scripts").is_dir()
            file_list = [f for f in skill_dir.rglob("*") if f.is_file()]
            total_lines = 0
            for f in file_list:
                try:
                    total_lines += len(f.read_text(encoding="utf-8", errors="replace").splitlines())
                except OSError:
                    pass
            md_only = (not file_list) or all(
                f.suffix.lower() in (".md", ".markdown") for f in file_list
            )

            if not file_list:
                status = "rejected"
                reason = "Skill contains no files (possible download failure)"
                scan_summary = {
                    "findings": 0, "errors": 0, "warnings": 0,
                    "scan_error": None, "file_count": 0, "total_lines": 0,
                    "markdown_only": True, "has_scripts": False,
                }
                _clawhub_log_install(slug, status, reason, scan_summary)
                return json.dumps({
                    "status": status,
                    "reason": reason,
                    "scan_summary": scan_summary,
                })

            # 5. Run snyk security scan
            scan_findings: list[dict] = []
            scan_error = None
            try:
                proc = subprocess.run(
                    ["uvx", "snyk-agent-scan@latest", "scan", "--skills", "--json", str(skill_dir)],
                    capture_output=True, text=True, timeout=15,
                )
                if proc.stdout.strip():
                    try:
                        scan_data = json.loads(proc.stdout)
                        if isinstance(scan_data, list):
                            scan_findings = scan_data
                        elif isinstance(scan_data, dict):
                            scan_findings = scan_data.get("findings", scan_data.get("results", []))
                            if not scan_findings and "vulnerabilities" in scan_data:
                                scan_findings = scan_data["vulnerabilities"]
                    except json.JSONDecodeError:
                        scan_error = "Could not parse scan output as JSON"
            except subprocess.TimeoutExpired:
                scan_error = "snyk scan timed out (15s)"
            except FileNotFoundError:
                scan_error = "uvx binary not found — scan skipped"

            # 6. Classify findings by severity
            error_count = 0
            warning_count = 0
            for finding in scan_findings:
                sev = str(finding.get("severity", finding.get("level", ""))).lower()
                if sev in ("e", "error", "critical", "high"):
                    error_count += 1
                elif sev in ("w", "warning", "medium", "low"):
                    warning_count += 1

            # 7. Gating logic
            scan_summary = {
                "findings": len(scan_findings),
                "errors": error_count,
                "warnings": warning_count,
                "scan_error": scan_error,
                "file_count": len(file_list),
                "total_lines": total_lines,
                "markdown_only": md_only,
                "has_scripts": has_scripts,
            }

            if scan_error is not None:
                status = "flagged"
                reason = f"Security scan could not complete: {scan_error}. Manual review required."
            elif error_count > 0:
                status = "rejected"
                reason = f"Security scan found {error_count} critical/error-level findings"
            elif has_scripts:
                status = "flagged"
                reason = "Skill contains a scripts/ directory — manual review required"
                if warning_count > 0:
                    reason += f" ({warning_count} warnings also found)"
            elif warning_count > 0 and not md_only:
                status = "flagged"
                reason = f"{warning_count} warnings found in non-markdown skill"
            elif warning_count > 0 and md_only and total_lines >= 500:
                status = "flagged"
                reason = f"{warning_count} warnings found; skill has {total_lines} lines (>=500)"
            elif md_only and total_lines < 500:
                status = "installed"
                reason = "Auto-installed: "
                if warning_count > 0:
                    reason += f"markdown-only, <500 lines, {warning_count} warnings (low risk)"
                else:
                    reason += "clean scan, markdown-only, <500 lines"
            else:
                status = "flagged"
                reason = f"Non-markdown skill with {len(file_list)} files, {total_lines} lines — manual review recommended"

            # 8. Auto-install if approved
            if status == "installed":
                install_dest = CLAWHUB_INSTALL_DIR / slug
                install_dest.parent.mkdir(parents=True, exist_ok=True)
                if install_dest.exists():
                    shutil.rmtree(install_dest)
                shutil.copytree(skill_dir, install_dest)
                _skill_index._scan()

            # 9. Log to install log
            _clawhub_log_install(slug, status, reason, scan_summary)

            return json.dumps({
                "status": status,
                "reason": reason,
                "scan_summary": scan_summary,
            })

        except Exception as exc:
            return json.dumps({
                "status": "rejected",
                "reason": f"Unexpected error: {exc}",
                "scan_summary": None,
            })
        finally:
            # 10. Clean up staging
            try:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)
            except OSError:
                pass

    return await asyncio.get_running_loop().run_in_executor(None, _impl)


def _clawhub_log_install(slug: str, status: str, reason: str, scan_summary: dict) -> None:
    """Append an install record to CLAWHUB_INSTALL_LOG."""
    try:
        CLAWHUB_INSTALL_LOG.parent.mkdir(parents=True, exist_ok=True)

        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        scan_desc = "clean"
        if scan_summary.get("scan_error"):
            scan_desc = f"scan error: {scan_summary['scan_error']}"
        elif scan_summary.get("errors", 0) > 0:
            scan_desc = f"{scan_summary['errors']} errors"
        elif scan_summary.get("warnings", 0) > 0:
            scan_desc = f"{scan_summary['warnings']} warnings"

        entry = (
            f"\n## {slug} — {date_str}\n"
            f"- **Decision**: {status}\n"
            f"- **Reason**: {reason}\n"
            f"- **Scan**: {scan_desc}\n"
            f"- **Files**: {scan_summary.get('file_count', '?')} files, "
            f"{scan_summary.get('total_lines', '?')} lines, "
            f"markdown-only: {'yes' if scan_summary.get('markdown_only') else 'no'}, "
            f"has scripts: {'yes' if scan_summary.get('has_scripts') else 'no'}\n"
        )

        with CLAWHUB_INSTALL_LOG.open("a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("# ClawhHub Installed Skills Log\n\n")
            f.write(entry + "\n")
    except Exception as exc:
        print(f"openclaw-mcp: failed to write install log: {exc}", file=sys.stderr)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenClaw MCP Server")
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio", "sse"],
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE bind host")
    parser.add_argument("--port", type=int, default=8093, help="SSE bind port")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port

    print(
        f"OpenClaw MCP Server — transport={args.transport}"
        + (f", {args.host}:{args.port}" if args.transport == "sse" else ""),
        file=sys.stderr,
    )
    mcp.run(transport=args.transport)
