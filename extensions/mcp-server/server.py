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

Exposes seven tools via the MCP stdio protocol. Fully standalone —
no dependency on the OpenClaw gateway or any external service.

  - tag_search    : frontmatter tag-based document search (in-process index)
  - tag_explore   : tag co-occurrence discovery (in-process index)
  - vault_overview: vault statistics and hub pages (in-process index)
  - memory_search : BM25 full-text search via qmd CLI (~/.bun/bin/qmd)
  - memory_get    : direct filesystem read from ~/obsidian/
  - web_search    : DuckDuckGo search via ddgs
  - web_fetch     : HTTP GET + readability-lxml + html2text content extraction

Run with:
  uv run server.py

Nothing is written to stdout except MCP JSON-RPC frames.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
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

# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("openclaw")


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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
