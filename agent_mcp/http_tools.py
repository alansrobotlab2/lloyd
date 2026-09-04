#!/usr/bin/env python3
"""
Lloyd MCP Server: HTTP Tools — web search, fetch, and generic requests.

Tools: http_search, http_fetch, http_request

Extraction: `http_fetch` runs trafilatura, which keeps the document's shape —
headings, lists, tables, and crucially the href of every link, so the model can
fetch an index page and then follow it. The hand-rolled `_TextExtractor` below
survives only as the fallback for pages trafilatura declines to parse; on its
own it returned one whitespace-joined blob with every URL discarded, which is
why callers kept giving up on this tool and shelling out to `curl`.
"""

import json
import re
import urllib.parse
from html.parser import HTMLParser

import httpx
import trafilatura

from agent_mcp._shared import make_sync_http_client, text_result
from ddgs import DDGS
from mcp.types import Tool

WEB_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
WEB_TIMEOUT_S = 15.0
WEB_MAX_RESPONSE_BYTES = 2_000_000

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


def _is_loopback_host(hostname: str) -> bool:
    """The machine itself, by any of its names.

    `http_request` deliberately allows loopback so the agent can drive local
    services; `127.0.0.1` and `localhost` are the same host and must be
    treated identically. Previously only `127.` was allowed, so
    `http://localhost:8080/x` was blocked while `http://127.0.0.1:8080/x`
    went through.
    """
    h = hostname.lower()
    return h == "localhost" or h.startswith("127.") or h == "::1"


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip_tags = {"script", "style", "head", "meta", "link", "nav", "header", "footer"}
        self._in_skip = 0
        self._in_body = False
        self._title = ""
        self._title_tag = False

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._in_skip += 1
        if tag == "title":
            self._title_tag = True
        if tag in {"body", "article", "main", "section"}:
            self._in_body = True

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._in_skip = max(0, self._in_skip - 1)
        if tag == "title":
            self._title_tag = False

    def handle_data(self, data):
        # <title> lives inside <head>, which is in _skip_tags — so the title
        # has to be read before the skip check, or it is always empty.
        if self._title_tag:
            text = data.strip()
            if text and not self._title:
                self._title = text
            return
        if self._in_skip > 0:
            return
        text = data.strip()
        if text:
            if self._in_body or not self._title:
                self.text_parts.append(text)

    def get_text(self):
        return " ".join(self.text_parts)


def _http_search(query: str, count: int = 5) -> str:
    count_ = min(max(count, 1), 10)
    try:
        raw = list(DDGS().text(query, max_results=count_))
    except Exception as exc:
        return json.dumps({"error": f"http_search error: {exc}"})
    if not raw:
        return json.dumps({"error": f'No results found for "{query}".'})
    results = []
    for i, r in enumerate(raw, 1):
        results.append({
            "rank": i,
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        })
    return json.dumps({"results": results})


EXTRACT_MODES = ("markdown", "text")


_MD_LINK = re.compile(r"\[([^\]]*)\]\((?!https?://|mailto:|#)([^)\s]+)\)")


def _absolutize_links(markdown: str, base_url: str) -> str:
    """Rewrite relative markdown link targets against the page URL.

    Trafilatura emits hrefs exactly as the page wrote them, so a docs index
    comes back full of `../topic/packaging/`. Those are unfollowable once the
    content leaves the page context — and following a link off an index page
    is the main reason markdown mode keeps hrefs at all.
    """
    def repl(m):
        try:
            return f"[{m.group(1)}]({urllib.parse.urljoin(base_url, m.group(2))})"
        except Exception:
            return m.group(0)
    return _MD_LINK.sub(repl, markdown)


def _extract_html(html_text: str, extract_mode: str) -> tuple[str, str]:
    """(title, content) for an HTML document.

    `extract_mode` is a real branch: "markdown" keeps headings, list markers,
    inline code and `[text](href)` links; "text" is the same content flattened
    to prose with the link URLs dropped. Before 2026-09-04 this argument was
    advertised, accepted, and then never read — both values returned identical
    bytes, so the fallback chain the skills library documented ("retry with
    extract_mode: text") could not do anything.
    """
    title = ""
    try:
        meta = trafilatura.extract_metadata(html_text)
        title = ((meta.title if meta else "") or "").strip()
    except Exception:
        title = ""

    content = None
    try:
        content = trafilatura.extract(
            html_text,
            output_format="markdown" if extract_mode == "markdown" else "txt",
            include_links=(extract_mode == "markdown"),
            include_tables=True,
            include_comments=False,
        )
    except Exception:
        content = None

    if not (content or "").strip():
        # Trafilatura declines documents it reads as boilerplate or as too
        # short to score. Falling back keeps a thin answer better than none.
        parser = _TextExtractor()
        parser.feed(html_text)
        content = parser.get_text()
        title = title or (parser._title or "")

    content = re.sub(r"\n\s*\n+", "\n\n", content or "").strip()
    return title, content


def _http_fetch(url: str, extract_mode: str = "markdown", max_chars: int = 50000) -> str:
    max_chars_ = min(max(max_chars, 1000), 200000)
    extract_mode_ = (extract_mode or "markdown").strip().lower()
    if extract_mode_ not in EXTRACT_MODES:
        return json.dumps({
            "error": f'Invalid extract_mode "{extract_mode}" — expected one of {", ".join(EXTRACT_MODES)}'
        })
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return json.dumps({"error": f"Invalid URL: {url}"})
    if parsed.scheme not in ("http", "https"):
        return json.dumps({"error": f"Only http/https URLs supported"})
    hostname = parsed.hostname or ""
    if _is_private_host(hostname):
        return json.dumps({"error": f'Blocked — private/internal hostname "{hostname}"'})
    headers = {"User-Agent": WEB_USER_AGENT, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}
    try:
        with make_sync_http_client(timeout=WEB_TIMEOUT_S, follow_redirects=True, verify=True) as client:
            response = client.get(url, headers=headers)
    except httpx.TimeoutException:
        return json.dumps({"error": f"Timed out after {WEB_TIMEOUT_S}s"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    if response.status_code >= 400:
        return json.dumps({"error": f"HTTP {response.status_code}"})
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "xml" not in content_type:
        text = response.text
        truncated = text[:max_chars_]
        return json.dumps({"url": url, "content": truncated, "truncated": len(truncated) < len(text)})
    try:
        raw_bytes = response.content[:WEB_MAX_RESPONSE_BYTES]
        html_text = raw_bytes.decode("utf-8", errors="replace")
        title, content = _extract_html(html_text, extract_mode_)
        if extract_mode_ == "markdown":
            # Resolve against the FINAL url so relative links on a redirected
            # page resolve to where the content actually came from.
            base = str(getattr(response, "url", "") or url)
            content = _absolutize_links(content, base)
        # The title is a heading only in markdown mode, and only when the
        # extracted content does not already open with it — trafilatura
        # usually keeps the page's own <h1>, and printing both reads as a
        # duplicated heading.
        first_line = content.lstrip().split("\n", 1)[0].lstrip("# ").strip()
        # Either direction counts as "already there": trafilatura keeps the
        # page's <h1> ("PEP 723 - Inline script metadata") while the <title>
        # carries a site suffix ("... | peps.python.org"), so neither string
        # contains the other outright.
        t_low, f_low = title.strip().lower(), first_line.lower()
        needs_heading = (
            title
            and extract_mode_ == "markdown"
            and f_low != ""
            and t_low not in f_low
            and f_low not in t_low
        )
        full = f"# {title}\n\n{content}" if needs_heading else content
        truncated = full[:max_chars_]
        return json.dumps({
            "url": url,
            "title": title,
            "extract_mode": extract_mode_,
            "content": truncated,
            "truncated": len(truncated) < len(full),
        })
    except Exception as exc:
        return json.dumps({"error": f"Extraction failed: {exc}"})


def _http_request(method: str, url: str, headers: dict | None = None, body: str = "", timeout: int = 30) -> str:
    method_ = method.upper()
    allowed = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}
    if method_ not in allowed:
        return json.dumps({"error": f"Unsupported method {method_!r}"})
    timeout_ = min(max(timeout, 1), 120)
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return json.dumps({"error": f"Invalid URL: {url}"})
    if parsed.scheme not in ("http", "https"):
        return json.dumps({"error": f"Only http/https supported"})
    hostname = parsed.hostname or ""
    loopback = _is_loopback_host(hostname)
    if _is_private_host(hostname) and not loopback:
        return json.dumps({"error": f'Blocked — private/internal hostname "{hostname}"'})
    # TLS verification is on for everything except the machine's own loopback,
    # where local services legitimately serve self-signed certificates. It used
    # to be off for every request, which meant no certificate was ever checked
    # on any outbound call (added by an auto-generated commit, 2026-04-11).
    try:
        with make_sync_http_client(timeout=timeout_, verify=not loopback, follow_redirects=True) as client:
            resp = client.request(method_, url, headers=headers or {}, content=body.encode() if body else b"")
            return json.dumps({"status_code": resp.status_code, "headers": dict(resp.headers), "body": resp.text})
    except Exception as exc:
        return json.dumps({"error": f"Request failed: {exc}"})


async def list_tools():
    return [
        Tool(name="http_search", description=(
            "Search the public web (DuckDuckGo) and get back ranked titles, URLs and snippets. "
            "This is the way to look something up online — reach for it before Bash whenever the "
            "answer is on the internet rather than on this machine, including when you do not yet "
            "know which URL you need. Pair it with http_fetch to read a result in full. Do not shell "
            "out to curl or wget for web search."
        ), inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {"type": "integer", "description": "Number of results (1-10, default 5)"},
            },
            "required": ["query"],
        }),
        Tool(name="http_fetch", description=(
            "Fetch a public http(s) URL and read it as clean markdown or plain text, keeping headings, "
            "lists, tables and link URLs while dropping navigation and boilerplate. Use it for any web "
            "page, article or documentation page — in markdown mode the links come back as "
            "[text](href), so you can fetch an index page and then follow it. Prefer this over running "
            "curl in Bash, which returns raw HTML you then have to strip yourself. Use http_request "
            "instead for non-GET verbs, custom headers, or a JSON/XML API whose raw body you want; use "
            "Bash + curl for localhost, which this tool blocks by design. If a page comes back near-empty "
            "it is probably JavaScript-rendered — switch to browser_navigate + browser_snapshot rather "
            "than retrying."
        ), inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "extract_mode": {"type": "string", "enum": ["markdown", "text"], "description": "markdown (default) keeps headings, lists, tables and [text](href) links; text returns flat prose with link URLs dropped"},
                "max_chars": {"type": "integer", "description": "Max characters to return; clamped to 1000-200000, default 50000. Out-of-range values are clamped, not rejected"},
            },
            "required": ["url"],
        }),
        Tool(name="http_request", description=(
            "Make a raw HTTP request with any verb, custom headers and a body, and get back the status "
            "code, response headers and the unparsed body. Use it for REST/GraphQL APIs, for POST/PUT/PATCH/DELETE, and whenever you want JSON or XML exactly as the server sent it rather than extracted "
            "prose. For reading a human-facing web page use http_fetch; to find a URL first use http_search."
        ), inputSchema={
            "type": "object",
            "properties": {
                "method": {"type": "string", "description": "HTTP method (GET, POST, PUT, PATCH, DELETE, HEAD)"},
                "url": {"type": "string", "description": "URL to request"},
                "headers": {"type": "object", "description": "Request headers"},
                "body": {"type": "string", "description": "Request body"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (1-120)"},
            },
            "required": ["method", "url"],
        }),
    ]


async def call_tool(name: str, arguments: dict):
    if name == "http_search":
        return text_result(_http_search(arguments.get("query", ""), arguments.get("count", 5)))
    elif name == "http_fetch":
        return text_result(_http_fetch(arguments.get("url", ""), arguments.get("extract_mode", "markdown"), arguments.get("max_chars", 50000)))
    elif name == "http_request":
        return text_result(_http_request(arguments.get("method", "GET"), arguments.get("url", ""), arguments.get("headers"), arguments.get("body", ""), arguments.get("timeout", 30)))
    return text_result(json.dumps({"error": f"Unknown tool: {name}"}))

