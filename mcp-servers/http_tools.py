#!/usr/bin/env python3
"""
Lloyd MCP Server: HTTP Tools — web search, fetch, and generic requests.

Tools: http_search, http_fetch, http_request
"""

import asyncio
import json
import re
import urllib.parse
from html.parser import HTMLParser

import httpx
from ddgs import DDGS
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

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

app = Server("lloyd-http-tools")


def _is_private_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    return any(p.match(hostname) for p in _PRIVATE_IP_PATTERNS)


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
        if self._in_skip > 0:
            return
        text = data.strip()
        if text:
            if self._title_tag and not self._title:
                self._title = text
            elif self._in_body or not self._title:
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


def _http_fetch(url: str, extract_mode: str = "markdown", max_chars: int = 50000) -> str:
    max_chars_ = min(max(max_chars, 1000), 200000)
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
        with httpx.Client(follow_redirects=True, timeout=WEB_TIMEOUT_S, verify=False) as client:
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
        parser = _TextExtractor()
        parser.feed(html_text)
        title = parser._title or ""
        content = parser.get_text()
        content = re.sub(r"\n\s*\n+", "\n\n", content).strip()
        full = f"# {title}\n\n{content}" if title else content
        truncated = full[:max_chars_]
        return json.dumps({"url": url, "title": title, "content": truncated, "truncated": len(truncated) < len(full)})
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
    if _is_private_host(hostname) and not hostname.startswith("127."):
        return json.dumps({"error": f'Blocked — private/internal hostname "{hostname}"'})
    try:
        with httpx.Client(timeout=timeout_, verify=False, follow_redirects=True) as client:
            resp = client.request(method_, url, headers=headers or {}, content=body.encode() if body else b"")
            return json.dumps({"status_code": resp.status_code, "headers": dict(resp.headers), "body": resp.text})
    except Exception as exc:
        return json.dumps({"error": f"Request failed: {exc}"})


@app.list_tools()
async def list_tools():
    return [
        Tool(name="http_search", description="Web search via DuckDuckGo. Returns titles, URLs, and snippets.", inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {"type": "integer", "description": "Number of results (1-10, default 5)"},
            },
            "required": ["query"],
        }),
        Tool(name="http_fetch", description="Fetch a URL and extract readable content.", inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "extract_mode": {"type": "string", "description": "Extraction mode: markdown or text"},
                "max_chars": {"type": "integer", "description": "Max characters to return (1000-200000)"},
            },
            "required": ["url"],
        }),
        Tool(name="http_request", description="Generic HTTP request. Returns status code, headers, and body.", inputSchema={
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


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "http_search":
        return [TextContent(type="text", text=_http_search(arguments.get("query", ""), arguments.get("count", 5)))]
    elif name == "http_fetch":
        return [TextContent(type="text", text=_http_fetch(arguments.get("url", ""), arguments.get("extract_mode", "markdown"), arguments.get("max_chars", 50000)))]
    elif name == "http_request":
        return [TextContent(type="text", text=_http_request(arguments.get("method", "GET"), arguments.get("url", ""), arguments.get("headers"), arguments.get("body", ""), arguments.get("timeout", 30)))]
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
