# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp[cli]",
#   "pyyaml",
#   "ddgs",
#   "httpx",
#   "readability-lxml",
#   "html2text",
# ]
# ///
"""
OpenClaw MCP Server

Exposes tools via the MCP stdio protocol. Fully standalone —
no dependency on the OpenClaw gateway or any external service.

  - tag_search      : frontmatter tag-based document search (in-process index)
  - tag_explore     : tag co-occurrence discovery (in-process index)
  - vault_overview  : vault statistics and hub pages (in-process index)
  - memory_search   : BM25 full-text search via qmd CLI (~/.bun/bin/qmd)
  - memory_get      : direct filesystem read from ~/obsidian/
  - prefill_context : full memory prefill pipeline (tag match + BM25 + GLM)
  - web_search      : DuckDuckGo search via ddgs
  - web_fetch       : HTTP GET + readability-lxml + html2text content extraction
  - file_read/write/edit/glob/grep : filesystem tools sandboxed to $HOME
  - run_bash      : shell command execution (sandboxed to $HOME, max 120s)
  - http_request  : full HTTP client (GET/POST/PUT/PATCH/DELETE/HEAD; loopback allowed)
  - memory_write  : create/overwrite a file in the Obsidian vault

Run with:
  uv run server.py

Nothing is written to stdout except MCP JSON-RPC frames.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import math
import os
import re
import subprocess
import sys
import time
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

# ── Constants ─────────────────────────────────────────────────────────────────

VAULT = Path(os.environ.get("HOME", "/home/alansrobotlab")) / "obsidian"
QMD = Path.home() / ".bun/bin/qmd"

EXCLUDE_DIRS = {"templates", "images"}
EXCLUDE_FILES = {"tags.md"}
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
GLM_MODEL = "Qwen3-30B-A3B-Instruct-2507"

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


# ── Build index ───────────────────────────────────────────────────────────────

print(f"openclaw-mcp: scanning vault at {VAULT}", file=sys.stderr)
_docs = _scan_vault(VAULT)
_index = TagIndex(_docs)
print(
    f"openclaw-mcp: indexed {_index.doc_count} docs, {_index.tag_count} tags",
    file=sys.stderr,
)

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
_PUNCT_SIMPLE = re.compile(r"[?!.,;:'\"]")
_PUNCT_FULL = re.compile(r"[?!.,;:'\"()\[\]{}#]")


def _extract_user_query(prompt: str) -> str | None:
    m = _PROMPT_ENVELOPE.search(prompt)
    if m:
        return m.group(1).strip()
    stripped = _JSON_BLOCK.sub("", prompt)
    stripped = " ".join(stripped.split())
    return stripped[-500:] if len(stripped) >= MIN_QUERY_LENGTH else None


def _simplify_query(query: str) -> str:
    words = [
        w for w in _PUNCT_SIMPLE.sub("", query.lower()).split()
        if len(w) > 1 and w not in FILLER_WORDS
    ]
    return " ".join(words[:6])


def _extract_topic_keywords(prompt: str) -> list[str]:
    m = _PROMPT_ENVELOPE.search(prompt)
    text = m.group(1).strip() if m else prompt.strip()

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
) -> str:
    if not tier1 and not tier2:
        return ""

    lines: list[str] = [
        "<memory_context>",
        "Pre-fetched memory context. Use this to answer directly if sufficient"
        " — use memory_search or tag_search only if you need more.\n",
    ]

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
        lines.append("**Also relevant** (use memory_get for full content):")
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

_PREFILL_LOG = Path.home() / ".openclaw" / "logs" / "timing.jsonl"


def _log_prefill(**kwargs: Any) -> None:
    try:
        _PREFILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.datetime.now().isoformat(), "event": "memory_prefill", **kwargs}
        with _PREFILL_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ── Background TagIndex refresh ───────────────────────────────────────────────

async def _refresh_index_loop() -> None:
    """Rescan vault every PREFILL_REFRESH_S seconds and atomically swap _index."""
    global _index
    while True:
        await asyncio.sleep(PREFILL_REFRESH_S)
        try:
            new_docs = _scan_vault(VAULT)
            _index = TagIndex(new_docs)
            print(
                f"openclaw-mcp: refreshed index: {_index.doc_count} docs, {_index.tag_count} tags",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"openclaw-mcp: index refresh failed: {exc}", file=sys.stderr)


@asynccontextmanager
async def _lifespan(_server: Any):
    task = asyncio.create_task(_refresh_index_loop())
    yield
    task.cancel()


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("openclaw", lifespan=_lifespan)


@mcp.tool()
def tag_search(
    tags: list[str],
    mode: str = "or",
    type: str = "any",
    limit: int = 10,
) -> str:
    """Search the Obsidian knowledge vault by frontmatter tags.

    Returns documents matching the specified tag(s) with title, summary,
    tags, type, and status. Use AND mode to find documents at the intersection
    of multiple topics, OR mode for broader searches.

    Examples:
      tag_search(["alfie"])
      tag_search(["ai", "rag"], mode="and")
      tag_search(["robotics"], type="hub")
    """
    limit = min(max(limit, 1), 25)
    type_filter = None if type == "any" else type

    unresolved: list[str] = []
    suggestions: dict[str, list[str]] = {}
    for tag in tags:
        if _index.resolve_tag(tag) is None:
            unresolved.append(tag)
            suggestions[tag] = _index.suggest_tags(tag)

    docs = _index.search_by_tags(tags, mode, type_filter, limit)
    all_docs = _index.search_by_tags(tags, mode, type_filter, 9999)

    return _format_tag_search(tags, mode, docs, len(all_docs), limit, unresolved, suggestions)


@mcp.tool()
def tag_explore(
    tag: str,
    bridge_to: str = "",
    limit: int = 15,
) -> str:
    """Explore tag relationships in the Obsidian vault.

    Given a tag, shows co-occurring tags ranked by frequency.
    Optionally provide bridge_to to find documents that have BOTH tags.
    Use this to discover connections between topics and navigate the
    vault's knowledge structure.
    """
    limit = min(max(limit, 1), 30)
    canonical = _index.resolve_tag(tag)
    tag_clean = tag.lstrip("#")

    if canonical is None:
        suggestions = _index.suggest_tags(tag)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f'Tag "{tag_clean}" not found in the vault.{hint}'

    tag_doc_count = len(_index.tag_to_docs.get(canonical, []))
    tag_idf = _index.tag_idf.get(canonical, 0.0)
    related = _index.get_related_tags(canonical, limit)

    bridge_tag: str | None = None
    bridge_docs: list[DocMeta] | None = None
    if bridge_to:
        bridge_tag = bridge_to.lstrip("#")
        bridge_docs = _index.search_by_tags([canonical, bridge_to], "and", None, 10)

    return _format_tag_explore(canonical, tag_doc_count, tag_idf, related, bridge_tag, bridge_docs)


@mcp.tool()
def vault_overview(detail: str = "summary") -> str:
    """Show Obsidian vault statistics and structure.

    detail options:
      summary  — overview: doc counts, type distribution, top tags, top docs (default)
      tags     — all tags with document frequencies and IDF scores
      hubs     — hub/index pages with summaries
      types    — document type breakdown with percentages
    """
    stats = _index.get_stats()
    return _format_vault_overview(stats, detail)


@mcp.tool()
def memory_search(query: str, max_results: int = 10, json_output: bool = False) -> str:
    """BM25 full-text search across the Obsidian vault.

    Uses the qmd CLI (~/.bun/bin/qmd) for keyword-based search.
    Returns matching document paths, relevance scores, and snippets.
    Standalone — does not require the OpenClaw gateway.

    json_output: if True, return structured JSON {"results": [{path, score, snippet}]}
                 instead of human-readable text (used internally by the prefill hook).
    """
    try:
        proc = subprocess.run(
            [str(QMD), "search", query, "-c", "obsidian", "-n", str(max_results), "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return '{"results": []}' if json_output else f"qmd search error: {proc.stderr.strip()}"
        results: list[dict] = json.loads(proc.stdout)
        if not results:
            return '{"results": []}' if json_output else "No results found."

        parsed = []
        for r in results:
            path = r.get("file", "").removeprefix("qmd://obsidian/")
            score = r.get("score", 0)
            # Strip qmd's diff-style context markers (@@...@@) from snippets
            snippet = re.sub(r"@@[^@]*@@\s*", "", r.get("snippet", "")).strip()
            parsed.append({"path": path, "score": score, "snippet": snippet[:200]})

        if json_output:
            return json.dumps({"results": parsed})

        lines = [f"- {r['path']} (score: {r['score']:.2f})\n  {r['snippet']}" for r in parsed]
        return "\n".join(lines)

    except subprocess.TimeoutExpired:
        return '{"results": []}' if json_output else "Error: qmd search timed out"
    except Exception as exc:
        return '{"results": []}' if json_output else f"Error: {exc}"


@mcp.tool()
def memory_get(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a specific file from the Obsidian vault by relative path.

    path: relative path from vault root, e.g. "projects/alfie/alfie.md"
    Standalone — reads directly from ~/obsidian/, no gateway required.
    """
    target = VAULT / path
    if not target.exists():
        return f"File not found: {path}"
    if not target.is_relative_to(VAULT):
        return "Error: path escapes vault root"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
        if start_line > 0 or end_line > 0:
            file_lines = text.splitlines()
            start = max(0, start_line - 1)
            end = end_line if end_line > 0 else len(file_lines)
            text = "\n".join(file_lines[start:end])
        return text or "(empty file)"
    except OSError as exc:
        return f"Error reading file: {exc}"


# ── Prefill context tool ──────────────────────────────────────────────────────


@mcp.tool()
async def prefill_context(prompt: str, session_id: str = "") -> str:
    """Run the memory prefill pipeline for a user prompt.

    Executes tag matching (in-memory, instant) and BM25 vector search in
    parallel, then optionally runs a GLM keyword extraction pass for extra
    searches. Merges and ranks results, fetches content for top hits, and
    returns a formatted <memory_context> block.

    Returns empty string if no relevant context is found or pipeline times out.
    Called by the before_prompt_build hook; not intended for direct LLM use.
    """
    t0 = time.monotonic()

    # Phase 0: query extraction + cache check
    user_query = _extract_user_query(prompt)
    if not user_query or len(user_query) < MIN_QUERY_LENGTH:
        return ""

    vector_query = _simplify_query(user_query)
    effective_query = vector_query if len(vector_query.split()) >= 2 else user_query[:80]
    if len(effective_query.split()) < 2:
        return ""

    cached = _get_prefill_cached(effective_query)
    if cached:
        return cached

    try:
        async with asyncio.timeout(PREFILL_TIMEOUT_S):
            # Phase 1-2: keyword extraction + tag match (sync, instant)
            topic_keywords = _extract_topic_keywords(prompt)
            if _index.doc_count > 0:
                tag_matches, tag_docs = _run_tag_match(topic_keywords, _index)
            else:
                tag_matches, tag_docs = [], {}
            matched_tag_names = {m["tag"].lower() for m in tag_matches}

            # Phase 3+4: parallel BM25 search + GLM (pipelined)
            # GLM extra searches chain directly off the GLM promise so they
            # start as soon as GLM resolves — same pattern as the TS prefill.

            async def _initial_search() -> list[dict[str, Any]]:
                raw = await asyncio.to_thread(memory_search, effective_query, 8, True)
                try:
                    return json.loads(raw).get("results", [])
                except Exception:
                    return []

            async def _glm_and_extra() -> tuple[list[str], list[dict[str, Any]]]:
                kws = await _extract_keywords_via_glm(user_query)
                glm_keywords = [k for k in kws if k and len(k) > 2][:2]
                extra_queries = [k for k in glm_keywords if k.lower() not in matched_tag_names]
                extra_results: list[dict[str, Any]] = []
                if extra_queries:
                    searches = await asyncio.gather(
                        *[asyncio.to_thread(memory_search, q, 3, True) for q in extra_queries],
                        return_exceptions=True,
                    )
                    for r in searches:
                        if isinstance(r, str):
                            try:
                                extra_results.extend(json.loads(r).get("results", []))
                            except Exception:
                                pass
                return glm_keywords, extra_results

            (init_results, (glm_keywords, extra_results)) = await asyncio.gather(
                _initial_search(),
                _glm_and_extra(),
            )
            all_vector = init_results + extra_results

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

            # Phase 7: fetch tier-1 content
            if tier1:
                fetched = await asyncio.gather(
                    *[asyncio.to_thread(memory_get, c.path) for c in tier1],
                    return_exceptions=True,
                )
                for i, r in enumerate(fetched):
                    if isinstance(r, str) and r and not r.startswith("File not found"):
                        tier1[i].content = r[:CONTENT_PER_DOC]

            # Phase 8: format
            context = _format_unified_context(tier1, tier2, MAX_CONTEXT_CHARS)
            if not context:
                return ""

            _set_prefill_cache(effective_query, context)
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
def web_search(query: str, count: int = 5) -> str:
    """Search the web using DuckDuckGo. Returns a numbered list of results with title, URL, and snippet.

    query: search terms
    count: number of results to return (1–10, default 5)
    """
    count = min(max(count, 1), 10)
    try:
        raw = list(_DDGS().text(query, max_results=count))
    except Exception as exc:
        return f"web_search error: {exc}"
    if not raw:
        return f'No results found for "{query}".'
    lines: list[str] = []
    for i, r in enumerate(raw, 1):
        title = r.get("title", "") or ""
        url = r.get("href", "") or ""
        snippet = r.get("body", "") or ""
        lines.append(f"[{i}] {title}\n    {url}\n    {snippet}")
    return "\n\n".join(lines)


@mcp.tool()
def web_fetch(
    url: str,
    extract_mode: str = "markdown",
    max_chars: int = WEB_DEFAULT_MAX_CHARS,
) -> str:
    """Fetch a URL and extract its readable content via readability-lxml + html2text.

    url: the URL to fetch (http or https only)
    extract_mode: "markdown" or "text" (default "markdown")
    max_chars: maximum characters to return (default 50000, max 200000)
    """
    from urllib.parse import urlparse

    max_chars = min(max(max_chars, 1_000), 200_000)

    try:
        parsed = urlparse(url)
    except Exception:
        return f"web_fetch error: Invalid URL: {url}"

    if parsed.scheme not in ("http", "https"):
        return f"web_fetch error: Only http/https URLs are supported, got {parsed.scheme!r}"

    hostname = parsed.hostname or ""
    if _is_private_host(hostname):
        return f'web_fetch error: Blocked — private/internal hostname "{hostname}"'

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
        return f"web_fetch error: Request timed out after {WEB_TIMEOUT_S}s"
    except Exception as exc:
        return f"web_fetch error: {exc}"

    if response.status_code >= 400:
        return f"web_fetch error: HTTP {response.status_code}"

    content_type = response.headers.get("content-type", "")

    # Non-HTML: return raw text
    if "html" not in content_type and "xml" not in content_type:
        text = response.text
        truncated = text[:max_chars]
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
        return f"web_fetch error: readability failed: {exc}"

    converter = _html2text.HTML2Text()
    converter.ignore_links = extract_mode == "text"
    converter.ignore_images = True
    converter.body_width = 0  # no line wrapping

    body = converter.handle(summary_html).strip()
    full = f"# {title}\n\n{body}" if title else body

    truncated = full[:max_chars]
    if len(truncated) < len(full):
        return truncated + f"\n\n[Truncated — {len(full)} chars total]"
    return truncated


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
def file_read(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a file from the filesystem.

    path: absolute path or ~/relative path (must be within $HOME)
    start_line: first line to return (1-indexed; 0 = beginning)
    end_line: last line to return (0 = end of file)

    Returns the file contents as text, optionally sliced to a line range.
    """
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


@mcp.tool()
def file_write(path: str, content: str) -> str:
    """Write (create or overwrite) a file.

    path: absolute path or ~/relative path (must be within $HOME)
    content: text content to write

    Creates parent directories if they don't exist.
    Returns a confirmation message.
    """
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


@mcp.tool()
def file_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace an exact string in a file (first occurrence only).

    path: absolute path or ~/relative path (must be within $HOME)
    old_text: exact text to find (must appear exactly once)
    new_text: replacement text

    Fails if old_text appears 0 or more than 1 time.
    """
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


@mcp.tool()
def file_glob(pattern: str, root: str = "~") -> str:
    """Find files matching a glob pattern.

    pattern: glob pattern, e.g. "**/*.py", "*.md", "src/**/*.ts"
    root: directory to search from (default: $HOME); must be within $HOME

    Returns a sorted list of matching paths (up to 200), relative to root.
    """
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


@mcp.tool()
def file_grep(pattern: str, path: str = "~", file_glob: str = "**/*", max_results: int = 50) -> str:
    """Search file contents with a regular expression.

    pattern: Python regex pattern to search for
    path: directory or file to search (default: $HOME); must be within $HOME
    file_glob: glob pattern to filter which files are searched (default "**/*")
    max_results: maximum matching lines to return (default 50, max 200)

    Returns matching lines with filename and line number.
    """
    max_results = min(max(max_results, 1), 200)

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
        if len(results) >= max_results:
            return
        try:
            size = fp.stat().st_size
            if size > FILE_MAX_READ_BYTES:
                return
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for lineno, line in enumerate(text.splitlines(), 1):
            if len(results) >= max_results:
                break
            if compiled.search(line):
                results.append(f"{fp}:{lineno}: {line.rstrip()}")

    if search_path.is_file():
        _search_file(search_path)
    elif search_path.is_dir():
        try:
            for fp in sorted(search_path.glob(file_glob)):
                if len(results) >= max_results:
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
    if len(results) >= max_results:
        output += f"\n... (limit of {max_results} results reached)"
    return output


# ── run_bash ──────────────────────────────────────────────────────────────────


@mcp.tool()
def run_bash(command: str, cwd: str = "~", timeout: int = 30) -> str:
    """Execute a shell command and return combined stdout+stderr with exit code.

    command: bash command string (passed to bash -c)
    cwd: working directory (default: $HOME; must be within $HOME)
    timeout: max seconds to wait (default 30, max 120)

    Returns: "exit 0\\n<stdout>" or "exit N\\n<stderr>" or error string.
    """
    timeout = min(max(timeout, 1), 120)
    cwd_result = _safe_path(cwd)
    if isinstance(cwd_result, str):
        return cwd_result

    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=str(cwd_result),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"

    output = proc.stdout
    if proc.stderr:
        output += ("\n" if output else "") + proc.stderr
    return f"exit {proc.returncode}\n{output}" if output else f"exit {proc.returncode}"


# ── http_request ───────────────────────────────────────────────────────────────

_HTTP_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}


@mcp.tool()
def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: str = "",
    timeout: int = 30,
) -> str:
    """Make an HTTP request and return status + response body.

    method: GET, POST, PUT, PATCH, DELETE, HEAD
    url: full URL (http or https only)
    headers: optional request headers dict
    body: optional request body string (set Content-Type header yourself for JSON)
    timeout: seconds (default 30, max 120)

    Returns: "HTTP <status>\\n<body>" or error string.
    Note: 127.0.0.1 (loopback) is allowed for local container services.
    Other private/internal IPs are blocked.
    """
    import urllib.parse as _urlparse

    method = method.upper()
    if method not in _HTTP_ALLOWED_METHODS:
        return f"Error: unsupported method {method!r}. Allowed: {', '.join(sorted(_HTTP_ALLOWED_METHODS))}"

    timeout = min(max(timeout, 1), 120)

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
            method,
            url,
            headers=headers or {},
            content=body.encode() if body else b"",
            timeout=timeout,
            verify=False,
            follow_redirects=True,
        )
        return f"HTTP {resp.status_code}\n{resp.text}"
    except Exception as exc:
        return f"Error: {exc}"


# ── memory_write ───────────────────────────────────────────────────────────────


@mcp.tool()
def memory_write(path: str, content: str) -> str:
    """Create or overwrite a file in the Obsidian vault.

    path: vault-relative path, e.g. "projects/alfie/notes.md"
    content: text content to write

    Creates parent directories automatically.
    """
    target = VAULT / path
    try:
        if not target.resolve().is_relative_to(VAULT.resolve()):
            return "Error: path escapes vault root"
    except Exception as exc:
        return f"Error: invalid path: {exc}"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except Exception as exc:
        return f"Error: {exc}"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
