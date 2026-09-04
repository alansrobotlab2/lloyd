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
    served(status_code=404)
    out = json.loads(http_tools._http_fetch("https://example.com/missing"))
    assert out["error"] == "HTTP 404"


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
