"""The aggregator refuses a request that did not come through the harness.

Every gate that decides whether a tool call may run lives upstream of the MCP
server: the grant/authority gate is a PreToolUse hook
(`app.harness.policy.install_policy_hook`), plan mode only removes tools from the
advertised list, the effect ledger is fed from harness `_meta`, and the bench/eval
read-only sandbox is keyed on a session id. A process on this box that posts
straight to `http://127.0.0.1:8500/mcp` passed none of them. Verified on the live
box before this file existed: `tools/list` answered 200 with 152 tools and
`tools/call backlog_boards {}` returned live board data with no Authorization
header, no `_meta` and no session id — `stateless_http=True` plus
`json_response=True` mean not even an `initialize` handshake is required — and
`tools/call vault_write {}` came back with the tool's own `MISSING_PARAM`
validator, which is how we know the request reached the module handler rather
than a gate.

These tests pin the control: one ASGI-layer credential check over every route,
asserted where it is installed and not in a mock that re-decides the same
question. The one path left open is `GET /health`, which supervisord, the
promotion gate (`scripts/automod/promote.py`) and the guardian
(`agent-services/guardian/policy.py`) probe without holding anything.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

import agent_mcp.aggregator_auth as A


# ── the shared credential ────────────────────────────────────────────────────

def test_the_first_reader_mints_a_0600_file_and_every_later_reader_agrees(tmp_path):
    path = tmp_path / "state" / "aggregator-token"
    A.reset_for_tests()
    token = A.read_token(env={A.TOKEN_FILE_ENV: str(path)})
    assert token and len(token) >= 16, "a server with no credential must mint one"
    # The umask on this box is 007, which would otherwise leave the file
    # unreadable to the other legitimate reader process.
    assert path.stat().st_mode & 0o077 == 0o000, "the token file is readable by anyone"
    A.reset_for_tests()
    assert A.read_token(publish=False, env={A.TOKEN_FILE_ENV: str(path)}) == token


def test_a_client_never_mints_a_credential_the_server_has_not_agreed_to(tmp_path):
    # A backend that minted its own token on a cold boot would send a value the
    # aggregator then refuses, and every tool call in the product would fail.
    missing = tmp_path / "nothing-here"
    assert A.read_token(publish=False, env={A.TOKEN_FILE_ENV: str(missing)}) is None
    assert not missing.exists(), "publish=False must not create the file"


def test_an_existing_token_is_never_rewritten(tmp_path):
    # The aggregator restarts first on a landing (promote.py restarts lloyd-mcp
    # before lloyd-backend). Rewriting here would orphan whatever the backend had
    # already learned and refuse every call until it re-read.
    path = tmp_path / "aggregator-token"
    path.write_text("already-agreed-value-0123456789\n")
    A.reset_for_tests()
    assert A.read_token(env={A.TOKEN_FILE_ENV: str(path)}) == "already-agreed-value-0123456789"


def test_token_matches_refuses_both_sides_being_empty():
    assert A.token_matches("", "") is False, "no header must not match no expectation"
    assert A.token_matches("abcdef0123456789", "abcdef0123456789") is True
    assert A.token_matches("abcdef0123456788", "abcdef0123456789") is False


# ── the transport-layer check, over the real route table ────────────────────

TOKEN = "test-aggregator-credential-0123456789abcdef"


class Recorder:
    """Stands in for a tool module and records whether it was ever reached.

    The item's check is a *sequencing* claim — the refusal must happen before
    `mod.call_tool` (`agent_mcp/main.py`) — so a test that only asserted a 401
    would pass even if the server had dispatched the call and answered 401 on the
    way out. Every refusal assertion below is paired with "the handler recorded
    nothing".
    """

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        from agent_mcp.main import TextContent
        return [TextContent(type="text", text="THE MODULE HANDLER RAN")]


@pytest.fixture
async def guarded(tmp_path, monkeypatch):
    """The real aggregator app, wrapped by the real middleware, with a
    credential the tests can quote.

    `_dispatch` is patched *after* discovery has run: `call_tool` rebuilds the
    table whenever it is empty, which would silently restore the live modules and
    dispatch a real `vault_write`.
    """
    import agent_mcp.main as M

    # The app is guarded at import with `token_fn=read_token`, and the guard
    # resolves the credential per request rather than caching it. So handing the
    # fixture a pre-written token *file* — instead of re-wrapping the app with a
    # lambda — makes the production guard present exactly TOKEN, which is what a
    # deployed aggregator does and what the test then quotes back. Re-wrapping
    # here would have tested a second guard stacked on the first and left the
    # wiring in `agent_mcp.main` untested.
    token_file = tmp_path / "aggregator-token"
    token_file.write_text(TOKEN + "\n")
    os.chmod(token_file, 0o600)
    monkeypatch.delenv(A.TOKEN_ENV, raising=False)
    monkeypatch.setenv(A.TOKEN_FILE_ENV, str(token_file))
    A.reset_for_tests()
    await M.list_tools()
    rec = Recorder()
    table = dict(M._dispatch)
    table["backlog_boards"] = rec        # read-class, live board data pre-fix
    table["vault_write"] = rec           # write-class, the tool that wiped the vault
    monkeypatch.setattr(M, "_dispatch", table)
    transport = httpx.ASGITransport(app=M.starlette_app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://127.0.0.1:8500") as client:
        yield client, rec


def _mcp(method: str, params: dict) -> dict:
    """One stateless JSON-RPC request — the body `urllib` sent, verbatim."""
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}


MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


# ── clause 1: a tools/call with no credential is refused before the handler ──

@pytest.mark.parametrize("tool,args", [
    ("backlog_boards", {}),
    ("vault_write", {"path": "knowledge/pinned.md", "content": "written by a script"}),
])
async def test_tools_call_without_a_credential_never_reaches_the_handler(guarded, tool, args):
    """Refused with the MCP-shaped deny, for one read-class call and one
    write-class call whose arguments are well-formed enough to have dispatched.

    The reach of the `rec.calls == []` leg below, stated plainly: this fixture
    drives the app through `httpx.ASGITransport` **without entering its lifespan**,
    and Streamable HTTP's session manager starts its anyio task group in the
    lifespan — so without it every `tools/call` dies at
    `mcp/server/streamable_http_manager.py:170-171` (`RuntimeError: Task group is
    not initialized`) before `mod.call_tool`, guard or no guard. Nor can this
    fixture be given one: that manager may be run once per instance, so a per-test
    entry fails on the second test. A dispatch that did happen would still be
    recorded here, but the claim that a handler call is observable in this shape at
    all is carried by
    `tests/test_mcp_transport.py::test_a_refused_tools_call_is_recorded_as_never_reaching_the_handler`,
    where the same recorder is read on both sides of the one credential header over
    a real socket and the credentialed leg provably arrives.
    """
    client, rec = guarded
    r = await client.post("/mcp", json=_mcp(
        "tools/call", {"name": tool, "arguments": args}), headers=MCP_HEADERS)
    assert r.status_code == 401, r.text[:300]
    assert r.json()["code"] == A.AUTH_ERROR_CODE
    assert rec.calls == [], f"{tool} was dispatched before the credential check"


# ── clause 2: tools/list and both mutating custom routes ────────────────────

async def test_tools_list_without_a_credential_is_refused(guarded):
    client, _ = guarded
    r = await client.post("/mcp", json=_mcp("tools/list", {}), headers=MCP_HEADERS)
    assert r.status_code == 401, r.text[:300]
    assert "boards" not in r.text


async def test_changes_revert_without_a_credential_is_refused_not_400(guarded):
    """The pre-fix answer to a well-formed body was 200 and a real revert.

    `POST /changes/revert` takes *caller-supplied* session and turn ids, so any
    local process could undo another session's recorded writes by naming them.
    A 400 from the handler's own validation would not be a control: a body that
    passes validation used to reach `_change_ledger.revert`.
    """
    client, _ = guarded
    r = await client.post("/changes/revert",
                          json={"session": "20260918_084244_someone_elses_session",
                                "turn": "turn-1", "paths": ["~/obsidian/SEED.md"]})
    assert r.status_code == 401, r.text[:300]
    assert r.status_code != 200


async def test_browser_navigate_without_a_credential_is_refused(guarded):
    client, _ = guarded
    r = await client.post("/browser/navigate", json={"url": "http://example.com"})
    assert r.status_code == 401, r.text[:300]


async def test_state_and_changes_without_a_credential_are_refused(guarded):
    """/state is the dashboard's and /changes the turn footer's: both carry
    live data, neither carried a credential before this."""
    client, _ = guarded
    for path in ("/state", "/changes?session=s&turn=t"):
        r = await client.get(path)
        assert r.status_code == 401, f"{path} -> {r.status_code} {r.text[:120]}"


async def test_a_wrong_credential_is_refused_too(guarded):
    client, rec = guarded
    r = await client.post("/mcp", json=_mcp(
        "tools/call", {"name": "vault_write",
                       "arguments": {"path": "x", "content": "y"}}),
        headers={**MCP_HEADERS, A.AUTH_HEADER: "guess-my-token-000000"})
    assert r.status_code == 401, r.text[:200]
    assert rec.calls == []


# ── clause 3: liveness stays open ───────────────────────────────────────────

async def test_health_is_served_with_no_credential(guarded):
    """supervisord, `app/routers/health.py` and `promote.MCP_HEALTH` probe this
    with no credential and no way to get one; refusing it would turn a healthy
    aggregator into a restart storm."""
    client, _ = guarded
    r = await client.get("/health")
    assert r.status_code == 200, r.text[:200]
    assert r.json()["tools"] > 0


# ── the positive half: the credential is what makes it work ─────────────────

async def test_a_request_carrying_the_credential_reaches_the_route_handler(guarded):
    """The positive half, over a real route instead of a mock.

    `/changes` is a custom Starlette route on the same app table as
    `/changes/revert`, so a 200 here says the middleware let the request through
    to the handler — the same path `/changes/revert` takes one route away. The
    MCP transport is not used for this leg: `tools/call` through a
    transport-only client needs the session manager's task group, which only the
    server's own lifespan starts, and the harness-side positive legs are
    `tests/test_tool_sandbox.py` (a credentialed, session-bearing write still
    dispatches) and the pool/proxy credential tests below.
    """
    client, _ = guarded
    r = await client.get("/changes?session=no-such-session&turn=t1",
                         headers={A.AUTH_HEADER: TOKEN})
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["files"] == [] and body["session_id"] == "no-such-session"
    r = await client.get("/state", headers={A.AUTH_HEADER: TOKEN})
    assert r.status_code == 200, r.text[:200]
    assert "tool_sandbox" in r.json(), "the bench runner's pre-flight key moved"


async def test_the_middleware_forwards_the_untouched_scope_to_the_inner_app():
    seen: list[dict] = []

    async def inner(scope, receive, send):
        seen.append(scope)
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{\"ok\": true}"})

    app = A.CredentialMiddleware(inner, token_fn=lambda: TOKEN).app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://127.0.0.1:8500") as client:
        r = await client.post("/changes/revert", json={"session": "s", "turn": "t"},
                              headers={A.AUTH_HEADER: TOKEN})
    assert r.status_code == 200 and len(seen) == 1
    assert seen[0]["path"] == "/changes/revert"
    assert seen[0]["method"] == "POST"


async def test_no_open_path_leaks_beyond_health():
    assert A.OPEN_PATHS == frozenset({"/health"}), (
        "every path not listed here must carry the credential; a second open "
        "path needs its own reason and its own test")


# ── the callers: every legitimate route across the seam still gets through ──

def test_a_loopback_url_gets_the_credential_and_a_remote_one_never_does(tmp_path,
                                                                       monkeypatch):
    """The credential is a loopback secret.

    `MCPPool` is generic: a `mcp_servers:` entry can name a host on the
    internet. Attaching this header to every HTTP server the pool opens would
    hand the aggregator's credential to a third party, so it goes out only to a
    host the secret could already reach.
    """
    token_file = tmp_path / "aggregator-token"
    token_file.write_text(TOKEN + "\n")
    os.chmod(token_file, 0o600)
    monkeypatch.delenv(A.TOKEN_ENV, raising=False)
    monkeypatch.setenv(A.TOKEN_FILE_ENV, str(token_file))
    A.reset_for_tests()

    assert A.headers_for_url("http://127.0.0.1:8500/mcp") == {A.AUTH_HEADER: TOKEN}
    assert A.headers_for_url("http://localhost:8500/state") == {A.AUTH_HEADER: TOKEN}
    for remote in ("https://api.example.com/mcp", "http://100.100.100.100:8500/mcp",
                   "http://goliath.example.ts.net:8500/mcp"):
        assert A.headers_for_url(remote) == {}, f"leaked the secret to {remote}"
    A.reset_for_tests()


def test_every_backend_proxy_sends_the_credential_it_needs():
    """The three backend proxies and the bench pre-flight, by source.

    Asserted on the resolved URLs rather than on a hardcoded port: each proxy
    derives its URL from `services.lloyd_mcp` through
    `app.aggregator_config`, so the header and the endpoint cannot drift apart —
    which is the failure mode this file's own bug history is full of.

    Source text is not the only pin for any of these five. Three run the real
    caller and check the header it puts on the wire — the changes/revert proxies
    (`tests/test_files_changed_surface.py::test_the_backend_proxies_send_the_credential`,
    which asserts `headers == A.auth_headers()` on the recording fake), the browser
    navigate leg (`tests/test_browser_panel.py`), and the bench pre-flight, which is
    the only one driven against a live guarded server
    (`tests/test_aggregator_auth_callers.py::test_the_bench_preflight_sends_the_credential_the_guard_accepts`)
    because the refusal is observable there, as is the dashboard's `/state` read
    (`tests/test_mcp_transport.py::test_the_dashboard_state_read_carries_the_credential`,
    whose positive leg runs `_agent_state()` itself and whose negative leg proves
    the route refuses without it). The pool's three HTTP client sites are
    the remaining shape this file can only count: the persistent one is now opened
    for real against the guarded server
    (`tests/test_mcp_transport.py::test_the_persistent_session_opener_carries_the_credential`),
    and the legacy SSE branch is still grep-only, because no SSE server exists
    anywhere on this box to open a session against.
    """
    from app.aggregator_config import route

    assert route("browser_navigate").endswith("/browser/navigate")
    assert route("changes_revert").endswith("/changes/revert")
    assert route("changes").endswith("/changes")
    for rel, needle in (
        ("app/routers/browser.py", "auth_headers_for(_MCP_NAVIGATE_URL)"),
        ("app/routers/sessions.py", "auth_headers_for(revert_url)"),
        ("app/routers/messages.py", "auth_headers_for(changes_url)"),
        ("app/routers/dashboard.py", "auth_headers_for(_MCP_STATE_URL)"),
        ("scripts/autoresearch/bench_runner_sdk.py", "auth_headers_for(state_url)"),
    ):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert needle in src, f"{rel} no longer sends the credential ({needle})"


def test_the_harness_pool_sends_the_credential_on_both_http_paths():
    """`_http_session` (per-call) and `_open_session` (persistent) are two
    separate client constructions; covering one and missing the other would
    break discovery or dispatch depending on which config the pool is running,
    which is exactly the split this repo keeps tripping over.

    The count is a backstop, not the primary pin, because two of the three sites
    are now proven by running them against the guarded server: the per-call path
    is the one `tests/test_mcp_transport.py`'s `pool` fixture dispatches through,
    and the persistent one is opened directly by
    `test_the_persistent_session_opener_carries_the_credential` (measured on this
    branch: mutating `app/harness/mcp_pool.py:665` to `headers=None` fails exactly
    that node, and mutating the per-call `headers = aggregator_headers(url)` at
    :347 leaves `tests/test_mcp_transport.py` + `tests/test_mcp_layer.py` at 1
    failure and 9 errors instead of 59 passed). The legacy SSE branch at :676 is
    the one leg with no server to open against — nothing on this box speaks SSE —
    so for it the count is the only pin."""
    src = (ROOT / "app" / "harness" / "mcp_pool.py").read_text(encoding="utf-8")
    assert "from agent_mcp.aggregator_auth import headers_for_url as aggregator_headers" in src
    assert src.count("aggregator_headers(") >= 3, (
        "the pool has three HTTP client construction sites (the per-call streamable "
        "path, the persistent streamable path, and the legacy SSE path); one of them "
        f"sends no credential — found {src.count('aggregator_headers(')} call sites")
    assert "create_mcp_http_client" in src


def test_the_promotion_probes_stay_on_the_open_path():
    """`scripts/automod/promote.py` and `app/routers/health.py` probe `/health`,
    and the guardian's copy of that URL lives at
    `agent-services/guardian/policy.py:84`. None of them can hold a credential:
    the promotion probe runs from the round process, the guardian runs from a
    detached copy, and a refusal there reads as a dead aggregator."""
    from agent_mcp import main as mcp_main

    src = (ROOT / "scripts" / "automod" / "promote.py").read_text(encoding="utf-8")
    assert "/health" in src, "the promotion liveness probe moved off /health"
    assert mcp_main.PORT == 8500
    assert A.OPEN_PATHS == frozenset({"/health"})
