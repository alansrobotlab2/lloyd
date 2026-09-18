"""The two legitimate callers whose credential path no booted-server test reaches.

`tests/test_mcp_transport.py` boots the shipped aggregator and is where the
refusal is proven over a socket; it cannot also cover these two, for two
concrete reasons:

* The bench runner's pre-flight (`scripts/autoresearch/bench_runner_sdk.py:
  require_tool_sandbox`) opens its **own** `httpx.AsyncClient`, so it can only be
  exercised against a server on a real port — and `agent_mcp.main.starlette_app`
  is a module singleton whose Streamable HTTP session manager initialises once,
  so a second file cannot boot it in the same process. The stub below is a plain
  Starlette app wrapped in the *same* guard (`require_credential`), which is what
  makes the leg worth testing: it proves the header the bench runner sends is the
  header the guard accepts, and it lets the sandbox map say `enforced: true`
  without the host's bubblewrap. Its URL is built by the same
  `_get_mcp_servers()` → `.rsplit('/mcp')` derivation as the live path, redirected
  to the stub port, so that derivation is what is being checked, not a copy of it.
* The canary runs two processes — `scripts/automod/canary_config.py` points
  `HOME` at the worktree (`scripts/automod/gate.py:_boot_canary_services` sets
  `HOME=round_home` and `LLOYD_AUTOMOD_ROOT`) — and `token_path()` is
  `Path.home() / ".local/state/lloyd/aggregator-token"`. No in-process test has
  two processes, so the canary's whole boot is unpinned. This file's second test
  starts the two real processes with `HOME` redirected the way the canary does and
  asserts they agree on one credential.

The third seam the review named — `app/routers/health.py`,
`scripts/automod/promote.py:MCP_HEALTH`, and
`agent-services/guardian/policy.py:MCP_HEALTH_URL` all probe `/health` without a
credential — is pinned in `tests/test_mcp_transport.py`, where the shipped app is
booted and the answer to "does that probe still get a 200" is a real response.
"""

from __future__ import annotations

import asyncio
import os
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_mcp.aggregator_auth as A  # noqa: E402

#: A credential of its own, so this file can never pass by accidentally reading
#: the box's real `~/.local/state/lloyd/aggregator-token`.
STUB_TOKEN = "stub-side-credential-0123456789abcdef"

_SANDBOX_MAP = {"enforced": True, "bwrap": True, "background_slugs": ["bench"]}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stub_app(seen: list):
    """A `/state` route, behind the real credential guard.

    `seen` collects the `Host` and credential header each request actually
    carried, so a test can distinguish "the guard refused" from "the route
    answered" by what arrived rather than by which code raised.
    """
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def state(request):
        seen.append({"host": request.headers.get("host", ""),
                     "token": request.headers.get(A.AUTH_HEADER, "")})
        return JSONResponse({"tool_sandbox": dict(_SANDBOX_MAP)})

    inner = Starlette(routes=[Route("/state", state, methods=["GET"])])
    return A.require_credential(inner)


@pytest.fixture
def guarded_stub():
    """Serve the guarded stub on a loopback port for one test."""
    import uvicorn

    seen: list = []
    port = _free_port()
    config = uvicorn.Config(_stub_app(seen), host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)

    def run():
        asyncio.run(server.serve())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = 400
    while deadline:
        deadline -= 1
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            import time
            time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", seen
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# ── the bench runner's `/state` pre-flight ───────────────────────────────────

@pytest.mark.asyncio
async def test_the_bench_preflight_sends_the_credential_the_guard_accepts(
        guarded_stub, monkeypatch, tmp_path):
    """`require_tool_sandbox` crosses the seam; removing the credential stops it.

    Pre-fix this GET needed nothing. Post-fix a pre-flight that sent no header
    would be refused, `ToolSandboxUnavailable` would be raised, and every trial
    would be skipped — a bench that silently cannot run. So both directions are
    pinned: the header the runner builds is the one the guard accepts, and its
    absence is a refusal rather than a pass.
    """
    import scripts.autoresearch.bench_runner_sdk as bench

    base, seen = guarded_stub
    monkeypatch.setenv(A.TOKEN_ENV, STUB_TOKEN)
    monkeypatch.delenv(A.TOKEN_FILE_ENV, raising=False)
    A.reset_for_tests()
    # Redirect the *source* of the URL, so the construction under test stays the
    # runner's own: `_get_mcp_servers()` → `base.rsplit('/mcp', 1)[0] + '/state'`.
    monkeypatch.setattr("app.mcp_discovery._get_mcp_servers",
                        lambda: {"lloyd-mcp": {"url": f"{base}/mcp"}})

    monkeypatch.setattr(bench, "_sandbox_verified", False)
    await bench.require_tool_sandbox()          # must not raise: sandbox map says enforced
    assert seen and seen[-1]["token"] == STUB_TOKEN, (
        f"the pre-flight reached the route without the credential: {seen}")

    # And the same pre-flight with nothing to send is a refusal, not a silent
    # pass. Empty env *and* an unreadable token file is the state a canary or a
    # fresh box is really in; deleting the env alone would just fall back to this
    # box's own token, which the guard would happily accept — that leg would pass
    # whether or not the runner sent anything. The token file is redirected to an
    # empty tmp path so the guard's mint-on-read cannot land on the real one.
    seen.clear()
    monkeypatch.setenv(A.TOKEN_ENV, "")
    monkeypatch.setenv(A.TOKEN_FILE_ENV, str(tmp_path / "absent-token"))
    A.reset_for_tests()
    monkeypatch.setattr(bench, "_sandbox_verified", False)
    with pytest.raises(bench.ToolSandboxUnavailable) as exc:
        await bench.require_tool_sandbox()
    # The guard answers with the refusal body, which carries no `tool_sandbox`
    # key, so the pre-flight's own check is what refuses the trial. The point is
    # that it refuses: the route never saw the request, and no trial ran.
    assert seen == [], "the route answered a credentialless pre-flight"
    assert "does not report an enforced read-only tool sandbox" in str(exc.value), exc.value


# ── the canary: two processes, one redirected HOME ───────────────────────────

#: One process's view of the credential, run with `HOME` where the canary puts it.
#: The two modules are imported from where production imports them — the client
#: half from `app.aggregator_config`, the minting half from
#: `agent_mcp.aggregator_auth`, which is what the aggregator's own middleware
#: calls (`read_token(publish=True)`, `expected()`) — so the probe cannot pass by
#: reimplementing either side.
_CANARY_PROBE = """
import sys, json
sys.path.insert(0, {root!r})
import agent_mcp.aggregator_auth as A
from app.aggregator_config import auth_headers_for, route
tok = A.read_token(publish={publish!r})
print(json.dumps({{"token": tok,
                   "path": str(A.token_path()),
                   "header": auth_headers_for(route("state")).get(A.AUTH_HEADER, "")}}))
"""


def _spawn_canary_role(root: Path, home: Path, publish: bool) -> dict:
    """Run one process's view of the credential, with `HOME` where the canary puts it."""
    code = _CANARY_PROBE.format(root=str(root), publish=publish)
    env = dict(os.environ)
    env["HOME"] = str(home)                      # what gate.py:_boot_canary_services sets
    env.pop(A.TOKEN_ENV, None)                   # the canary sets no token env
    env.pop(A.TOKEN_FILE_ENV, None)
    out = subprocess.run([sys.executable, "-c", code], cwd=str(root), env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    return __import__("json").loads(out.stdout.strip().splitlines()[-1])


def test_the_canary_backend_and_mcp_agree_on_one_credential(tmp_path):
    """Server publishes, client reads, under a `HOME` with no `STATE_DIR` symlink.

    The canary's backend (`promote.py:337-339`) and its aggregator
    (`canary_config.mcp_command()`, which inherits the same env) are separate
    processes and their round home is a fresh tree — `round.py` symlinks
    `.venvs`/`_pipeline`/`sessions`, not `.local`. Two independently minted
    secrets would mean the canary aggregator refuses its own backend, and
    `promote.py` reads a `tools>0` health map as the requirement, so the failure
    surfaces as a round refused with no explanation.
    """
    home = tmp_path / "canary-home"
    server = _spawn_canary_role(ROOT, home, publish=True)
    client = _spawn_canary_role(ROOT, home, publish=False)
    token = server["token"]
    assert token and len(token) >= 16, f"the server half minted nothing: {server}"
    assert client["token"] == token, (
        "the canary backend and aggregator would disagree and refuse each other")
    assert client["header"] == token, "the client would send a different value"
    token_file = home / ".local" / "state" / "lloyd" / "aggregator-token"
    assert token_file.is_file(), "the credential is not where the client looks"
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600, (
        "a group- or world-readable credential is readable by the model's "
        "children too, which is the whole point of the file being 0600")


def test_a_canary_that_cannot_mint_leaves_the_client_with_nothing(tmp_path):
    """Unwritable state is a refusal, not an accidental agreement on the empty string.

    The canary worktree sits under `/tmp` for the gate's run; a read-only mount,
    or the same uid losing the directory, means no credential exists at all. What
    must not happen is the client sending an empty header and the server's
    `expected()` being empty too — two nothings matching is an open door.
    """
    home = tmp_path / "unwritable-home"
    home.mkdir()
    home.chmod(0o500)                            # no write permission: mint fails
    try:
        server = _spawn_canary_role(ROOT, home, publish=True)
        client = _spawn_canary_role(ROOT, home, publish=False)
    finally:
        home.chmod(0o700)
    assert server["token"] is None, f"a mint that cannot write reported {server}"
    assert client["token"] is None, "the client invented a credential"
    # The guard's own rule, stated where the comparison happens rather than in a
    # mock of it: an empty expectation matches nothing, header or not.
    assert A.token_matches("", "") is False
    assert A.token_matches("anything", "") is False
