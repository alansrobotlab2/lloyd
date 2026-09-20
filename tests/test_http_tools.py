"""Contract tests for agent_mcp.http_tools.

This module had no tests at all, which is how three defects survived into
production and taught the agent to distrust the tool:

  * `extract_mode` was advertised in the schema, threaded into `_http_fetch`,
    and then never read — "markdown" and "text" returned identical bytes, so
    the retry chain the skills library documented was a guaranteed no-op.
  * The returned `title` was always empty: `<title>` lives inside `<head>`,
    and `head` is in the extractor's skip-set, so the text was dropped before
    it could be captured.
  * Extraction was `" ".join(parts)` — headings, list items, table cells and
    every link href collapsed into one blob, so `http_fetch` could not be used
    to fetch an index page and follow its links.

Network is never touched: the HTTP client is stubbed and the extraction path
is exercised directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_mcp import http_tools


# A realistically-sized document. Trafilatura scores very short pages as
# boilerplate and drops to a crude path, so a two-line fixture would test the
# fallback rather than the extractor.
PAGE = """<html><head><title>Install Guide</title></head><body>
<nav>Home About Contact</nav>
<article>
<h1>Installing Foo</h1>
<p>Foo requires Python 3.11 or newer. This paragraph is long enough that the
extractor treats the article as real content rather than navigation chrome,
which matters because short documents take a different code path entirely.</p>
<h2>Steps</h2>
<ol><li>Run <code>pip install foo</code> from a shell</li>
<li>Set <code>FOO_KEY</code> in your environment before starting the server</li></ol>
<p>See the <a href="https://example.com/api">API reference</a> for the full
option list and for details about configuring the retry behaviour.</p>
<table><tr><th>Flag</th><th>Default</th></tr><tr><td>--fast</td><td>off</td></tr></table>
</article>
<footer>Copyright 2026</footer>
</body></html>"""


class _FakeResponse:
    def __init__(self, *, text="", content=b"", status_code=200, content_type="text/html",
                 url="https://example.com/guide"):
        self.text = text
        self.content = content or text.encode()
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        # httpx exposes the FINAL url after redirects; relative links resolve
        # against it, not against what was requested.
        self.url = url


class _FakeClient:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None):
        return self._response


@pytest.fixture
def served(monkeypatch):
    """Serve a canned response to `_http_fetch` without touching the network."""
    def _serve(text=PAGE, content_type="text/html", status_code=200):
        resp = _FakeResponse(text=text, status_code=status_code, content_type=content_type)
        monkeypatch.setattr(http_tools, "make_sync_http_client",
                            lambda **kw: _FakeClient(resp))
        return resp
    return _serve


# ---------------------------------------------------------------------------
# extract_mode is a real branch
# ---------------------------------------------------------------------------

def test_extract_mode_markdown_and_text_differ(served):
    served()
    md = json.loads(http_tools._http_fetch("https://example.com/guide", "markdown"))
    txt = json.loads(http_tools._http_fetch("https://example.com/guide", "text"))
    assert md["content"] != txt["content"], "extract_mode must actually branch"
    assert md["extract_mode"] == "markdown"
    assert txt["extract_mode"] == "text"


def test_markdown_mode_keeps_structure_and_link_urls(served):
    served()
    out = json.loads(http_tools._http_fetch("https://example.com/guide", "markdown"))
    content = out["content"]
    # Headings survive.
    assert "# Installing Foo" in content
    assert "## Steps" in content
    # Link hrefs survive — this is what makes "fetch the index, then follow a
    # link" possible at all.
    assert "https://example.com/api" in content
    # Table cells stay distinguishable from prose.
    assert "--fast" in content and "Default" in content


def test_text_mode_drops_link_urls_but_keeps_prose(served):
    served()
    out = json.loads(http_tools._http_fetch("https://example.com/guide", "text"))
    content = out["content"]
    assert "API reference" in content
    assert "https://example.com/api" not in content


def test_boilerplate_is_stripped(served):
    served()
    out = json.loads(http_tools._http_fetch("https://example.com/guide", "markdown"))
    assert "Home About Contact" not in out["content"]


def test_invalid_extract_mode_is_rejected(served):
    served()
    out = json.loads(http_tools._http_fetch("https://example.com/guide", "raw"))
    assert "error" in out
    assert "extract_mode" in out["error"]


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

def test_title_is_populated(served):
    served()
    out = json.loads(http_tools._http_fetch("https://example.com/guide", "markdown"))
    assert out["title"], "title must not be empty for a page with <title>"
    assert "Install" in out["title"]


def test_text_extractor_captures_title_despite_head_skip():
    """The fallback extractor must read <title> before the <head> skip-set."""
    parser = http_tools._TextExtractor()
    parser.feed("<html><head><title>Hello</title></head><body><p>Body text</p></body></html>")
    assert parser._title == "Hello"


# ---------------------------------------------------------------------------
# max_chars clamping — clamped, never an error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requested", [1, 10, 999, 10**9, -5])
def test_max_chars_out_of_range_is_clamped_not_rejected(served, requested):
    served()
    out = json.loads(http_tools._http_fetch("https://example.com/guide", "markdown", requested))
    assert "error" not in out
    assert len(out["content"]) <= 200000


def test_truncation_is_reported(served):
    served()
    out = json.loads(http_tools._http_fetch("https://example.com/guide", "markdown", 1000))
    assert out["truncated"] is (len(out["content"]) == 1000)


# ---------------------------------------------------------------------------
# SSRF guard and URL validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://localhost:8080/health",
    "http://127.0.0.1:8500/mcp",
    "http://10.0.0.5/",
    "http://192.168.1.20/admin",
    "http://169.254.169.254/latest/meta-data/",
])
def test_private_hosts_are_blocked(url):
    out = json.loads(http_tools._http_fetch(url))
    assert "error" in out
    assert "private" in out["error"].lower() or "blocked" in out["error"].lower()


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "notaurl"])
def test_non_http_schemes_are_rejected(url):
    out = json.loads(http_tools._http_fetch(url))
    assert "error" in out


def test_non_html_content_type_is_passed_through(served):
    served(text='{"ok": true}', content_type="application/json")
    out = json.loads(http_tools._http_fetch("https://example.com/api.json"))
    assert out["content"] == '{"ok": true}'


def test_http_error_status_is_reported(served):
    """A failing fetch names its status, its retry class and where it landed.

    This node used to assert `out["error"] == "HTTP 404"` — the entire payload,
    one field, no way to tell a gone page from a rate limit from an auth wall.
    Item #850's acceptance makes that shape wrong on purpose: the clause reads
    "a result for 401, 403, 404, 429 and 5xx carries a retry-class field" and
    "a result with status >=400 also carries up to 200 characters of the
    response body text beside the status code", so a payload that is only
    `HTTP 404` can no longer satisfy it.
    """
    served(status_code=404)
    out = json.loads(http_tools._http_fetch("https://example.com/missing"))
    assert out["error"].startswith("HTTP 404"), out["error"]
    assert out["retry_class"] == http_tools.RETRY_CLASS_GONE
    assert "body" in out
    # `_FakeResponse` lands on /guide while the request asked for /missing:
    # the result must report the destination, not the request.
    assert out["final_url"] == "https://example.com/guide"


# ---------------------------------------------------------------------------
# Retry classes (clause 1: 401/403 vs 404 vs 429 vs 5xx must not collide)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    (401, http_tools.RETRY_CLASS_AUTH),
    (403, http_tools.RETRY_CLASS_AUTH),
    (404, http_tools.RETRY_CLASS_GONE),
    (410, http_tools.RETRY_CLASS_GONE),
    (429, http_tools.RETRY_CLASS_BACKOFF),
    (500, http_tools.RETRY_CLASS_SERVER),
    (502, http_tools.RETRY_CLASS_SERVER),
    (503, http_tools.RETRY_CLASS_SERVER),
])
def test_error_status_carries_its_retry_class(served, status, expected):
    served(status_code=status)
    out = json.loads(http_tools._http_fetch("https://example.com/x"))
    assert out["retry_class"] == expected, (status, out)


def test_retry_classes_are_four_distinct_values(served):
    """Auth, gone, backoff and server-error must read as four different values.

    One shared "retryable: false" flag would leave the model exactly where
    `HTTP 404` left it: unable to tell a Cloudflare interstitial (retrying
    never helps) from a 429 (retrying after a delay does).
    """
    seen = {}
    for status in (401, 404, 429, 500):
        served(status_code=status)
        seen[status] = json.loads(http_tools._http_fetch("https://example.com/x"))["retry_class"]
    assert len(set(seen.values())) == 4, seen
    assert seen[401] != seen[404] != seen[429] != seen[500]


# ---------------------------------------------------------------------------
# Body snippet (clause 2: up to 200 chars beside the status; empty body ok)
# ---------------------------------------------------------------------------

def test_error_status_carries_a_body_snippet(served):
    """Enough of the response to decide, without shipping the whole error page."""
    body = ("<html><body><h1>404 Not Found</h1><p>The page you requested "
            "was moved or deleted. Check the URL and try a search instead.</p>"
            "</body></html>")
    served(text=body, status_code=404)
    out = json.loads(http_tools._http_fetch("https://example.com/missing"))
    assert "404 Not Found" in out["body"]
    assert "moved or deleted" in out["body"]
    assert len(out["body"]) <= 200
    # One line: a raw HTML body would otherwise arrive as tags and newlines.
    assert "\n" not in out["body"]


# Verbatim shape of a live 404 body (example.com, 2026-09-20): the explanation
# sits in a <body> that follows a <link> and a <meta> in <head>.
REAL_404 = ('<!doctype html><html lang="en"><head><title>Example Domain</title>'
            '<link rel="icon" href="data:,"><meta name="viewport" content="width=device-width">'
            '<style>body{background:#eee;width:60vw;margin:15vh auto}</style></head>'
            '<body><div><h1>Example Domain</h1><p>This domain is for use in documentation '
            'examples without needing permission.</p></div></body></html>')


def test_error_snippet_is_prose_not_markup_on_a_real_error_page(served):
    """The excerpt has to be readable, not the first 200 bytes of the response.

    Routing the excerpt through the module's own `_TextExtractor` returns an
    empty string on this document: its skip tags include the void elements
    `<link>` and `<meta>`, which the parser never closes, so the skip depth is
    still positive at `</body>` and the `<body>` prose below is never emitted —
    the same document with those two tags cut extracts the heading and the
    sentence intact. Stripping tags here instead keeps that sentence in the
    payload and keeps the markup out of it.
    """
    served(text=REAL_404, content_type="text/html", status_code=404)
    out = json.loads(http_tools._http_fetch("https://example.com/nope"))
    assert "documentation examples" in out["body"], out["body"]
    assert "<" not in out["body"], "an excerpt of raw markup is not a snippet"


def test_error_snippet_survives_a_content_type_that_lies(served):
    """Servers send text/plain for HTML error pages; the strip keys off the bytes."""
    served(text=REAL_404, content_type="text/plain", status_code=503)
    out = json.loads(http_tools._http_fetch("https://example.com/nope"))
    assert "documentation examples" in out["body"], out["body"]
    assert "<style>" not in out["body"]


def test_error_snippet_drops_a_tag_longer_than_any_small_attribute_bound(served):
    """A tag wider than the pattern's inner bound is not a tag to the pattern.

    Defensive rather than observed: the first version of this strip bounded the
    pattern to 400 inner characters, and a 700-char tag — the length a
    theme/state data-attribute pair reaches on a modern app shell — passes
    straight through it as literal markup. The live case that first showed raw
    markup in this field was github.com/nope/nope on 2026-09-20, and its cause
    was the extractor fallback, not this bound (see
    test_error_snippet_is_prose_not_markup_on_a_real_error_page).
    """
    long_tag = '<link data-color-theme="' + "x" * 700 + '">'
    served(text=f"<html><body>{long_tag}<p>404 - page not found</p></body></html>",
            content_type="text/html", status_code=404)
    out = json.loads(http_tools._http_fetch("https://example.com/nope"))
    assert out["body"] == "404 - page not found", out["body"]


def test_error_snippet_drops_a_comment_block(served):
    """A Cloudflare challenge page is mostly comment and script; the prose that
    is left has to be what survives, not the comment's contents."""
    served(text="<html><body><!-- wait a few moments and refresh -->"
                "<p>Attention Required! | Cloudflare</p></body></html>",
           content_type="text/html", status_code=403)
    out = json.loads(http_tools._http_fetch("https://example.com/blocked"))
    assert "Attention Required" in out["body"], out["body"]
    assert "wait a few moments" not in out["body"], out["body"]


def test_error_status_snips_a_body_longer_than_the_limit(served):
    served(text="E" * 5000, content_type="text/plain", status_code=500)
    out = json.loads(http_tools._http_fetch("https://example.com/boom"))
    assert len(out["body"]) == 200


def test_error_status_with_an_empty_body_yields_an_empty_snippet(served):
    """An empty body is the common case for a HEAD-ish 403 and must not raise."""
    served(text="", status_code=403)
    out = json.loads(http_tools._http_fetch("https://example.com/forbidden"))
    assert out["body"] == ""
    assert out["retry_class"] == http_tools.RETRY_CLASS_AUTH


# ---------------------------------------------------------------------------
# JS-rendered pages (clause 3: the hint must be in the result, not only the
# tool description, which is the only place it lived before)
# ---------------------------------------------------------------------------

SPA_SHELL = ("<html><head><title>Acme Console</title></head>"
             "<body><div id=\"root\"></div>"
             "<script>window.__env={};fetch('/api/bootstrap')</script>"
             "</body></html>")


def test_thin_html_extraction_says_the_page_is_js_rendered(served):
    served(text=SPA_SHELL)
    out = json.loads(http_tools._http_fetch("https://app.example.com/console"))
    assert len(out["content"]) < http_tools.EMPTY_CONTENT_HINT_CHARS
    hint = out["js_rendered_hint"]
    # The two things the model needs: what happened, and the tool that works.
    assert "browser_navigate" in hint and "browser_snapshot" in hint
    assert "JavaScript" in hint
    # The hint is the one field a successful fetch adds beyond `final_url`, and
    # it appears only on a page whose content is under 400 chars — so bounding
    # it bounds worst-case growth on a success: 400 chars of content + hint +
    # final_url still costs less than any real article this tool returns.
    assert len(hint) <= 300, len(hint)


def test_a_real_page_gets_no_js_render_hint(served):
    served()
    out = json.loads(http_tools._http_fetch("https://example.com/guide"))
    assert "js_rendered_hint" not in out
    assert len(out["content"]) > http_tools.EMPTY_CONTENT_HINT_CHARS


def test_short_non_html_body_gets_no_js_render_hint(served):
    """A 40-char JSON API answer is short and correct; only HTML gets the hint."""
    served(text='{"ok": true}', content_type="application/json")
    out = json.loads(http_tools._http_fetch("https://example.com/api.json"))
    assert "js_rendered_hint" not in out


# ---------------------------------------------------------------------------
# final_url (clause 4: all three success returns and the >=400 path)
# ---------------------------------------------------------------------------

def test_final_url_reports_the_redirect_destination(served):
    """A redirected fetch must report where it landed, not what was asked for.

    `follow_redirects=True` was on and `response.url` was already read at
    private scope to absolutize links, then discarded: a page that moved was
    reported under the URL that moved away, so "the content is stale" and "I
    fetched something else" were indistinguishable.
    """
    served()
    out = json.loads(http_tools._http_fetch("https://short.example/old-page"))
    assert out["final_url"] == "https://example.com/guide"
    assert "url" not in out, "the requested URL is the model's own input; the result reports the destination"


def test_final_url_on_every_success_path(monkeypatch):
    """pdf, non-HTML and HTML all report the destination."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "paper body")
    raw = doc.tobytes()
    doc.close()
    for text, ctype, needle in [
        (raw, "application/pdf", "paper body"),
        ('{"ok": true}', "application/json", '"ok"'),
        (PAGE, "text/html", "Installing Foo"),
    ]:
        resp = _FakeResponse(text=text, content=raw if isinstance(text, bytes) else "",
                             content_type=ctype, url="https://cdn.example.net/landed")
        monkeypatch.setattr(http_tools, "make_sync_http_client", lambda **kw: _FakeClient(resp))
        out = json.loads(http_tools._http_fetch("https://short.example/x"))
        assert needle in out["content"], ctype
        assert out["final_url"] == "https://cdn.example.net/landed", ctype


# ---------------------------------------------------------------------------
# Payload growth caps (clause 5)
# ---------------------------------------------------------------------------

def test_error_payload_growth_over_the_old_shape_is_capped(monkeypatch):
    """>=400 payloads grew by at most 300 chars over `{"error": "HTTP 404"}`."""
    resp = _FakeResponse(text="x" * 20000, content_type="text/plain", status_code=500,
                         url="https://example.com/" + "a" * 400)
    monkeypatch.setattr(http_tools, "make_sync_http_client", lambda **kw: _FakeClient(resp))
    raw = http_tools._http_fetch("https://example.com/boom")
    assert len(raw) <= 19 + 300, len(raw)
    assert len(raw) <= http_tools.HTTP_ERROR_MAX_PAYLOAD_CHARS
    assert http_tools.HTTP_ERROR_MAX_PAYLOAD_CHARS == 19 + 300


def test_error_payload_keeps_the_snippet_when_it_fits(monkeypatch):
    """The cap must not silently delete the body it was asked to carry."""
    resp = _FakeResponse(text="y" * 20000, content_type="text/plain", status_code=500,
                         url="https://example.com/boom")
    monkeypatch.setattr(http_tools, "make_sync_http_client", lambda **kw: _FakeClient(resp))
    out = json.loads(http_tools._http_fetch("https://example.com/boom"))
    assert len(out["body"]) >= 150, out["body"]


def test_success_payload_grows_only_by_final_url(served):
    """A known-good markdown fetch costs at most 200 extra characters."""
    out = json.loads(http_tools._http_fetch("https://example.com/guide", "markdown"))
    old_shape = {k: v for k, v in out.items() if k not in ("final_url", "js_rendered_hint")}
    growth = len(json.dumps(out)) - len(json.dumps(old_shape))
    assert growth <= 200, growth
    assert "js_rendered_hint" not in out


# ---------------------------------------------------------------------------
# Error payloads must reach the harness as isError=True (P1-2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_payload_sets_is_error():
    result = await http_tools.call_tool("http_fetch", {"url": "http://localhost/x"})
    # mcp 2.x exposes model fields snake_case on read, even though the
    # constructor still accepts isError.
    assert result.is_error is True, (
        "a blocked fetch must surface as a tool error, not a success whose "
        "text happens to contain an error key"
    )


@pytest.mark.asyncio
async def test_the_text_the_model_receives_carries_the_retry_class(monkeypatch):
    """The payload survives the MCP seam, not just the in-process call.

    `_http_fetch` returns a string that `text_result` wraps into a
    `CallToolResult` and ships across the process boundary to the aggregator,
    which replays `content[0].text` to the model verbatim. A field that only
    exists on the dict inside this module is worth nothing to the model, so the
    contract is asserted on the serialized content on the far side of
    `call_tool` — the same bytes the model reads.
    """
    resp = _FakeResponse(text="<html><body>Forbidden: rate limit exceeded for token</body></html>",
                         status_code=403)
    monkeypatch.setattr(http_tools, "make_sync_http_client", lambda **kw: _FakeClient(resp))
    result = await http_tools.call_tool("http_fetch", {"url": "https://example.com/private"})
    assert result.is_error is True, "an HTTP error must still read as a tool error"
    out = json.loads(result.content[0].text)
    assert out["retry_class"] == http_tools.RETRY_CLASS_AUTH
    assert "rate limit exceeded" in out["body"]
    assert out["final_url"] == "https://example.com/guide"


@pytest.mark.asyncio
async def test_success_payload_is_not_an_error(served):
    served()
    result = await http_tools.call_tool(
        "http_fetch", {"url": "https://example.com/guide", "extract_mode": "markdown"}
    )
    assert result.is_error is False


# ---------------------------------------------------------------------------
# Advertised schema matches the implementation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_matches_implementation():
    tools = {t.name: t for t in await http_tools.list_tools()}
    assert set(tools) == {"http_search", "http_fetch", "http_request"}

    fetch = tools["http_fetch"].input_schema["properties"]
    # The enum must match EXTRACT_MODES or the model is told about a mode the
    # implementation will reject.
    assert set(fetch["extract_mode"]["enum"]) == set(http_tools.EXTRACT_MODES)

    for name, tool in tools.items():
        assert len(tool.description) >= 60, f"{name} description is a label, not a description"
        for param, spec in tool.input_schema.get("properties", {}).items():
            assert (spec.get("description") or "").strip(), f"{name}.{param} undocumented"


@pytest.mark.asyncio
async def test_descriptions_steer_away_from_curl():
    """The tool must say when to use it, not only what it is.

    With `Bash` always in the ToolSearch baseline alongside these three, the
    description is the only thing distinguishing them at selection time.
    """
    tools = {t.name: t.description.lower() for t in await http_tools.list_tools()}
    assert "curl" in tools["http_search"], "http_search must name the alternative it replaces"
    assert "curl" in tools["http_fetch"]
    # The localhost carve-out has to survive: Bash + curl is correct there,
    # because _http_fetch blocks private hosts.
    assert "localhost" in tools["http_fetch"]


# ---------------------------------------------------------------------------
# Link absolutization and heading de-duplication
# ---------------------------------------------------------------------------

def test_relative_links_are_absolutized():
    """A relative href is unfollowable once the content leaves the page, and
    following a link off an index page is the point of keeping hrefs."""
    out = http_tools._absolutize_links(
        "see [a](../topic/packaging/) and [b](/abs/path) and [c](https://x.test/q)",
        "https://peps.python.org/pep-0723/",
    )
    assert "https://peps.python.org/topic/packaging/" in out
    assert "https://peps.python.org/abs/path" in out
    # Already-absolute links and in-page fragments are left alone.
    assert "https://x.test/q" in out


def test_fragment_and_mailto_links_are_left_alone():
    out = http_tools._absolutize_links(
        "[f](#section) [m](mailto:a@b.test)", "https://example.com/page/"
    )
    assert "(#section)" in out
    assert "(mailto:a@b.test)" in out


def test_title_is_not_duplicated_as_a_heading(served):
    """Trafilatura keeps the page's own <h1>; prepending the <title> on top of
    it renders as the same heading twice."""
    served()
    out = json.loads(http_tools._http_fetch("https://example.com/guide", "markdown"))
    assert out["content"].count("# Installing Foo") == 1


def test_title_is_prepended_when_content_lacks_it(served):
    page = PAGE.replace("<h1>Installing Foo</h1>", "")
    served(text=page)
    out = json.loads(http_tools._http_fetch("https://example.com/guide", "markdown"))
    if out["title"]:
        assert out["content"].lstrip().startswith("#")


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _one_page_pdf(text: str = "Hello from page one") -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    raw = doc.tobytes()
    doc.close()
    return raw


def test_pdf_is_extracted_not_dumped_as_bytes(served, monkeypatch):
    """A PDF used to fall through to the non-HTML branch and come back as
    response.text — the bytes decoded as if they were text."""
    raw = _one_page_pdf()
    resp = _FakeResponse(content=raw, content_type="application/pdf")
    monkeypatch.setattr(http_tools, "make_sync_http_client", lambda **kw: _FakeClient(resp))
    out = json.loads(http_tools._http_fetch("https://example.com/paper.pdf", "markdown"))
    assert "error" not in out
    assert out["content_type"] == "pdf"
    assert "Hello from page one" in out["content"]


def test_pdf_detected_by_magic_bytes_when_content_type_lies(monkeypatch):
    """Plenty of servers send application/octet-stream for a PDF."""
    raw = _one_page_pdf("Served as octet-stream")
    resp = _FakeResponse(content=raw, content_type="application/octet-stream")
    monkeypatch.setattr(http_tools, "make_sync_http_client", lambda **kw: _FakeClient(resp))
    out = json.loads(http_tools._http_fetch("https://example.com/x", "markdown"))
    assert out.get("content_type") == "pdf"
    assert "Served as octet-stream" in out["content"]


def test_pdf_page_markers_only_in_markdown_mode(monkeypatch):
    raw = _one_page_pdf()
    resp = _FakeResponse(content=raw, content_type="application/pdf")
    monkeypatch.setattr(http_tools, "make_sync_http_client", lambda **kw: _FakeClient(resp))
    md = json.loads(http_tools._http_fetch("https://example.com/p.pdf", "markdown"))
    tx = json.loads(http_tools._http_fetch("https://example.com/p.pdf", "text"))
    assert "[page 1]" in md["content"]
    assert "[page 1]" not in tx["content"]
    # No separator before the first page.
    assert not md["content"].lstrip().startswith("---")


def test_corrupt_pdf_reports_an_error_not_a_crash(monkeypatch):
    resp = _FakeResponse(content=b"%PDF-1.4 not really a pdf", content_type="application/pdf")
    monkeypatch.setattr(http_tools, "make_sync_http_client", lambda **kw: _FakeClient(resp))
    out = json.loads(http_tools._http_fetch("https://example.com/broken.pdf"))
    assert "error" in out and "PDF" in out["error"]
