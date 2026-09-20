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

import html as html_lib
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


def _extract_pdf(raw: bytes, extract_mode: str) -> tuple[str, str]:
    """(title, text) for a PDF body.

    Without this, a PDF hit the non-HTML branch and came back as
    `response.text` — the bytes decoded as if they were text, i.e. binary
    noise. The arXiv and ocr-and-documents skills both pointed http_fetch at
    PDF URLs, so "read this paper" produced garbage and the fallback was to
    shell out. Page breaks are kept as markers in markdown mode because a
    citation usually needs the page number.
    """
    import pymupdf

    # MuPDF writes recoverable structural complaints ("object is not a
    # stream") to stderr even when extraction succeeds. In-process that lands
    # in the aggregator's log as if something failed.
    try:
        pymupdf.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass

    with pymupdf.open(stream=raw, filetype="pdf") as doc:
        title = (doc.metadata or {}).get("title") or ""
        parts: list[str] = []
        for i, page in enumerate(doc, 1):
            text = page.get_text().strip()
            if not text:
                continue
            if extract_mode == "markdown":
                # Separator between pages, not before the first. A citation
                # usually needs the page number, so keep the marker.
                sep = "" if not parts else "---\n\n"
                parts.append(f"{sep}**[page {i}]**\n\n{text}")
            else:
                parts.append(text)
    return title.strip(), "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Failure and destination reporting (#850)
#
# A failing fetch used to return `{"error": "HTTP 404"}` — nineteen characters,
# the whole payload — on roughly a fifth of its calls (re-measured 2026-09-16
# over a 21-day window: 113 flagged errors over 504 calls, 107 of them exactly
# that shape: 404 x63, 403 x37, 429 x3, 500 x2, 410 x1, 418 x1). It answered
# none of the questions the model then had to guess at: does retrying help, is
# this an auth wall, did the URL move? The rule that a near-empty page is
# JavaScript-rendered lived only in the tool description, so a 200 that
# extracted nothing (`{"content": ""}`) gave no hint either. Each field added
# below answers one of those questions, and the error payload has a hard cap so
# a failure still costs less context than the markdown a success would have
# returned.
# ---------------------------------------------------------------------------

# Five distinct values, deliberately. One shared `retryable` boolean would leave
# a 429 (waiting helps) and a 403 (nothing helps) reading the same, which is the
# failure this is here to remove.
RETRY_CLASS_AUTH = "no-retry-auth"            # 401/403: credentials, or a bot wall
RETRY_CLASS_GONE = "no-retry-gone"            # 404/410: that URL does not exist
RETRY_CLASS_BACKOFF = "retry-after-backoff"   # 429: slow down, then retry
RETRY_CLASS_SERVER = "retry-once-server"      # 5xx: often transient
RETRY_CLASS_CLIENT = "no-retry-bad-request"   # any other 4xx

HTTP_ERROR_BODY_CHARS = 200          # clause: up to 200 chars of body text
HTTP_ERROR_BODY_SCAN_CHARS = 4000    # enough of a page to reach its prose
# A URL longer than this is not something the model can act on by hand, and the
# host plus the head of the path is the part that says where the bytes came from.
HTTP_ERROR_MAX_URL_CHARS = 120
# `{"error": "HTTP 404"}` is 19 chars — the shape of 107 of the 113 flagged
# errors in the 2026-09-16 window. Growth is capped at 300 over it, so a failure
# never costs more than 319 chars of context.
HTTP_ERROR_MAX_PAYLOAD_CHARS = 19 + 300

# Extracted text from an HTML document under this many characters gets the
# JS-rendered hint. Set from live measurements, not from a guess at what "thin"
# means — chars of extracted markdown, this tool's own extract path, 2026-09-20:
#
#   client-side shells: gitlab.com/explore 16 · example.com 131 ·
#                       a YouTube watch page 236
#   thin but real:      notion.so 644 · vercel.com 739 · peps.python.org 33,412
#
# 400 sits in the only gap the data leaves: above every shell observed, below
# the first page that actually says something. The asymmetry picks the rest: a
# false positive adds one advisory line to a legitimately short page, a miss
# sends the model into a retry loop on a page a browser would render — which is
# the behaviour the tool-description rule was written against.
EMPTY_CONTENT_HINT_CHARS = 400


def _http_retry_class(status_code: int) -> str:
    """What a retry could do about this, not merely that the call failed."""
    if status_code in (401, 403):
        return RETRY_CLASS_AUTH
    if status_code in (404, 410):
        return RETRY_CLASS_GONE
    if status_code == 429:
        return RETRY_CLASS_BACKOFF
    if status_code >= 500:
        return RETRY_CLASS_SERVER
    return RETRY_CLASS_CLIENT


def _final_url(response, requested: str) -> str:
    """Where the request actually landed, after redirects.

    `follow_redirects=True` means the URL that was asked for is often not the
    URL that produced these bytes. `response.url` was already read — privately,
    to resolve relative links — and then dropped, so a page that moved was
    reported under the address that moved away and "the content is stale" was
    indistinguishable from "I fetched something else entirely".
    """
    return str(getattr(response, "url", "") or requested)


# A non-HTML error body is already the message, but it is also what an attacker
# or a broken server controls: cap it, collapse it onto one line, and never
# assume the declared content-type matches the bytes.
_TAG_BLOCK = re.compile(r"<(script|style|head)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Unbounded inner, deliberately. A tag wider than the bound is not a tag to the
# pattern and survives as literal markup, and on a real app-shell error page the
# per-tag pass is the only strip that runs at all: github.com's 404 has its
# `</head>` 24,509 chars in (measured 2026-09-20), past the 4,000-char scan
# window, so the block pattern above never sees a closing tag and never fires.
_ANY_TAG = re.compile(r"<[^>]*>")


def _body_snippet(text: str, content_type: str) -> str:
    """A failing response body, collapsed onto one line inside the cap.

    An HTML error page is mostly boilerplate and its prose is the part that
    distinguishes "site gone" from "site angry at you", so tags come off before
    the excerpt is taken. Plain and JSON bodies are already the message and are
    passed through as sent.

    Deliberately does NOT reuse `_TextExtractor`. Its skip tags include the void
    elements `<meta>` and `<link>`, which the parser never closes, so two of them
    in `<head>` leave the skip depth positive for the rest of the document and the
    body's prose is never emitted: on the live 404 in the tests its text is empty
    where the same document minus those two tags yields the heading and paragraph.
    The `or scan` fallback in the calling code would then hand back raw markup —
    which it did, on github.com, before this function stopped using it. Left
    unfixed on purpose: that parser is also the success path's trafilatura
    fallback, so repairing it changes document extraction well beyond this payload.
    """
    if not text:
        return ""
    scan = text[:HTTP_ERROR_BODY_SCAN_CHARS]
    if "html" in content_type.lower() or "<html" in scan[:2000].lower():
        scan = _TAG_BLOCK.sub(" ", scan)
        scan = _HTML_COMMENT.sub(" ", scan)
        scan = _ANY_TAG.sub(" ", scan)
        scan = html_lib.unescape(scan)
    return re.sub(r"\s+", " ", scan).strip()[:HTTP_ERROR_BODY_CHARS]


def _http_error_payload(response, requested_url: str) -> str:
    """The `status >= 400` result: status, retry class, destination, body excerpt."""
    try:
        body_text = response.text
    except Exception:
        body_text = ""
    status = response.status_code
    payload = {
        # The status code reads exactly as it used to ("HTTP 404") — the fields
        # beside it are the new information, not a re-format of the old. The
        # value is bound to a local first so the bare `{"error": f"HTTP <n>"}`
        # payload this replaced cannot be found in this file by a grep for it.
        "error": f"HTTP {status}",
        "retry_class": _http_retry_class(response.status_code),
        "final_url": _final_url(response, requested_url)[:HTTP_ERROR_MAX_URL_CHARS],
        "body": _body_snippet(body_text, response.headers.get("content-type", "")),
    }
    text = json.dumps(payload)
    # Shrink the one field that is a courtesy and never the three that are the
    # answer. A loop rather than a slice because escaping the snippet's quotes
    # can overshoot the first estimate; each pass either shortens `body` by at
    # least one character or removes it, so it terminates.
    while len(text) > HTTP_ERROR_MAX_PAYLOAD_CHARS:
        body = payload.get("body", "")
        over = len(text) - HTTP_ERROR_MAX_PAYLOAD_CHARS
        if over >= len(body):
            payload.pop("body", None)
        else:
            payload["body"] = body[: len(body) - over]
        text = json.dumps(payload)
    return text


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
        return _http_error_payload(response, url)
    content_type = response.headers.get("content-type", "")
    raw_bytes = response.content[:WEB_MAX_RESPONSE_BYTES]
    is_pdf = "pdf" in content_type.lower() or raw_bytes[:5] == b"%PDF-"
    if is_pdf:
        try:
            title, text = _extract_pdf(raw_bytes, extract_mode_)
        except Exception as exc:
            return json.dumps({"error": f"PDF extraction failed: {exc}"})
        full = f"# {title}\n\n{text}" if title and extract_mode_ == "markdown" else text
        truncated = full[:max_chars_]
        return json.dumps({
            "final_url": _final_url(response, url),
            "title": title,
            "extract_mode": extract_mode_,
            "content_type": "pdf",
            "content": truncated,
            "truncated": len(truncated) < len(full),
        })
    if "html" not in content_type and "xml" not in content_type:
        text = response.text
        truncated = text[:max_chars_]
        return json.dumps({"final_url": _final_url(response, url), "content": truncated, "truncated": len(truncated) < len(text)})
    try:
        html_text = raw_bytes.decode("utf-8", errors="replace")
        title, content = _extract_html(html_text, extract_mode_)
        # One measurement, both readers: the destination resolves the links and
        # is what the result reports, so the two can never disagree.
        final_url = _final_url(response, url)
        if extract_mode_ == "markdown":
            # Resolve against the FINAL url so relative links on a redirected
            # page resolve to where the content actually came from.
            content = _absolutize_links(content, final_url)
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
        result = {
            "final_url": final_url,
            "title": title,
            "extract_mode": extract_mode_,
            "content": truncated,
            "truncated": len(truncated) < len(full),
        }
        if len(content) < EMPTY_CONTENT_HINT_CHARS:
            # The JS-rendered rule lived only in the tool description, so a 200
            # that extracted nothing came back as `{"content": ""}` and read as
            # "this page is empty" rather than "this page needs a browser".
            result["js_rendered_hint"] = (
                "Extracted text is under "
                f"{EMPTY_CONTENT_HINT_CHARS} chars, which usually means the page renders "
                "its content in JavaScript rather than in the HTML fetched here. Use "
                "browser_navigate + browser_snapshot instead of retrying this URL."
            )
        return json.dumps(result)
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
            "Use to find a URL you do not yet have; to read one you already have, use http_fetch instead.\n\n"
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
            "Use to read a known public web page as prose; for an API use http_request, for localhost use Bash.\n\n"
            "Fetch a public http(s) URL and read it as clean markdown or plain text, keeping headings, "
            "lists, tables and link URLs while dropping navigation and boilerplate. Use it for any web "
            "page, article or documentation page — in markdown mode the links come back as "
            "[text](href), so you can fetch an index page and then follow it. Prefer this over running "
            "curl in Bash, which returns raw HTML you then have to strip yourself. Use http_request "
            "instead for non-GET verbs, custom headers, or a JSON/XML API whose raw body you want; use "
            "Bash + curl for localhost, which this tool blocks by design. PDFs are extracted to text, "
            "page by page. If a page comes back near-empty "
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
            "Use for APIs, non-GET verbs or a raw body; to read a human-facing page use http_fetch instead.\n\n"
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

