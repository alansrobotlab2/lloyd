"""The credential every request to the aggregator must carry (#1053).

The aggregator owns every tool: the built-ins (`Bash`, `Write`, `Edit`) and the
domain ones (`vault_write`, `email_send`, `grant_create`, `automod_land`). Until
this module existed it decided *what* to run and never asked *who* asked. The
bind is loopback (`main.PORT`, resolved from `services.lloyd_mcp`) and the Host
check (`main._security`) validates loopback names only, so the only thing
between another local process and the tool surface was a port number. That is a
limit on *who can reach it*, not on *what runs*: a `urllib` script inside one
`Bash` call POSTed `tools/call backlog_boards {}` with no auth, no `_meta` and
no session id, and got live board data back — 200 on `tools/list` (152 tools),
200 on the call, and `vault_write {}` returned the tool's own `MISSING_PARAM`
validator rather than a refusal, which is how we know the request travelled
every gate and reached `mod.call_tool`. `stateless_http=True` plus
`json_response=True` mean not even an `initialize` handshake is required.

Every gate that decides whether a call may run lives upstream of this process —
the grant gate is a PreToolUse hook (`app.harness.policy.install_policy_hook`),
plan mode only removes tools from the advertised list — so an out-of-band call
skipped all of them. It also skipped the bench/eval read-only sandbox, which is
keyed solely on a session id the caller simply omitted.

So the boundary is a request credential, checked in the ASGI layer above every
route rather than in a tool wrapper: an MCP JSON-RPC `tools/call` that arrives
here has, by construction, already bypassed every gate the harness owns, and
`/changes/revert` is a mutating route that trusts *caller-supplied* session and
turn ids — any local process could revert another session's recorded writes by
naming them.

What the credential does and does not buy
-----------------------------------------
The aggregator and the backend run as the same uid, so a secret in a file that
process can read is readable by a state-changing child of `Bash` too. This is
therefore **not containment against code that already has shell-as-this-user**.
What it does buy, and what the item asked for, is the boundary "a request that
did not come through the harness": it kills the unsupervised/accidental path —
a script, a notebook, a subagent shell — and every gate-skipping that does not
first go and read the secret. The sandbox keeps the rest: a sandboxed session
is refused the non-read-only tools even after reading this file, because the
sandbox is keyed on the session id and, since this module, a state-changing
call with *no* session id is refused outright rather than reading as
"not sandboxed".

Where the secret lives
----------------------
`~/.local/state/lloyd/aggregator-token`, mode `0600`, written once by the first
process to look (normally the aggregator at boot, which restarts before the
backend — `scripts/automod/promote.py` restarts `lloyd-mcp` first, so the
aggregator publishes and the backend reads). Not `config.yaml`, not a
supervisor conf: both are read at boot, so a token there could not survive a
landing without a restart that is refused from inside a self-modification round.
The aggregator writes it from inside its ASGI app — a file whose content was
already on disk is left alone, so a second process never clobbers it, and no
process's environment is ever rewritten (which would orphan a token another
process had already learned). Override the location with `LLOYD_MCP_TOKEN_FILE`,
or supply the value directly with `LLOYD_MCP_TOKEN`.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import stat
import threading
from pathlib import Path
from urllib.parse import urlparse

#: Header carrying the credential. Dash-case like every other Lloyd header, and
#: not `Authorization`: an `Authorization` header on a loopback request would be
#: logged by some middlewares as a bearer secret.
AUTH_HEADER = "X-Lloyd-Aggregator-Token"

#: Error code in every refusal body, so a caller can tell "wrong credential"
#: from a transport error rather than reading prose out of a status.
AUTH_ERROR_CODE = "AGGREGATOR_UNAUTHENTICATED"

TOKEN_ENV = "LLOYD_MCP_TOKEN"
TOKEN_FILE_ENV = "LLOYD_MCP_TOKEN_FILE"

#: `~/.local/state/lloyd/` — the same state root the automod and guardian
#: ledgers use (`scripts/automod/state.py`, `agent-services/guardian/policy.py`).
DEFAULT_TOKEN_PATH = Path.home() / ".local" / "state" / "lloyd" / "aggregator-token"

#: Paths served with no credential. `GET /health` is supervisord's and the
#: gate's liveness probe (`promote.MCP_HEALTH`, `app/routers/health.py`,
#: `agent-services/guardian/policy.MCP_HEALTH_URL`): all three run before or
#: independently of anything that holds the secret, and the body is a module
#: health map, not user data. Everything else — `/mcp`, `/state`,
#: `/changes`, `/changes/revert`, `/browser/navigate` — needs the credential.
#: `/state` is not on the list because the dashboard proxies it every 2 s and
#: the bench runner reads it before a trial; both are processes that can read
#: the file.
OPEN_PATHS = frozenset({"/health"})

_TOKEN_BYTES = 32
_MIN_TOKEN_CHARS = 16

_lock = threading.Lock()
#: The value this process published to the token file, so the write happens at
#: most once per process and a later read is a plain file read.
_published: str | None = None


def token_path(env: dict | None = None) -> Path:
    """Where the credential lives, honouring `LLOYD_MCP_TOKEN_FILE`."""
    raw = (env if env is not None else os.environ).get(TOKEN_FILE_ENV, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_TOKEN_PATH


def _looks_valid(token: str) -> bool:
    return len(token) >= _MIN_TOKEN_CHARS and all(
        c.isalnum() or c in "-_" for c in token)


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _publish(path: Path) -> str:
    """Write a fresh credential to `path`, or return an existing one.

    Two rules make this safe to call from a process that is not alone on the
    box: a file whose content already validates is never rewritten (so the
    aggregator coming up after the backend cannot orphan a token the backend
    has already learned), and the write is create-exclusive — the sole
    remaining race is two processes creating the file in the same instant,
    where the loser re-reads the winner's value.
    """
    existing = _read_file(path)
    if _looks_valid(existing):
        return existing
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _read_file(path)
    except OSError as exc:
        import logging
        logging.getLogger("lloyd-mcp").error(
            "aggregator_auth: cannot write the credential at %s: %s — every "
            "tools/call will be refused until it exists or %s is set",
            path, exc, TOKEN_ENV)
        return ""
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
        # An inherited umask (007 here) would otherwise leave the file
        # unreadable to the *other* legitimate reader process.
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except OSError:
        return ""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            path.chmod(0o600)
    except OSError:
        pass
    return token


def read_token(publish: bool = True,
               env: dict | None = None,
               path: Path | None = None) -> str | None:
    """The shared credential this process can use, or None if there is none.

    `publish=False` is the client form: a caller must never mint a credential
    the server has not agreed to, because the two halves could disagree and
    every call would then be refused. A server (`publish=True`) mints one on
    first read so the requirement fails closed rather than open.
    """
    global _published
    source = env if env is not None else os.environ
    from_env = (source.get(TOKEN_ENV) or "").strip()
    if from_env:
        return from_env
    target = path if path is not None else token_path(source)
    if not publish:
        token = _read_file(target)
        return token if _looks_valid(token) else None
    with _lock:
        if _published is not None:
            return _published
        token = _read_file(target)
        if not _looks_valid(token):
            token = _publish(target)
        if not _looks_valid(token):
            return None
        _published = token
        return token


def reset_for_tests() -> None:
    """Forget what this process published. Test-only seam."""
    global _published
    with _lock:
        _published = None


def token_matches(presented: str, expected: str) -> bool:
    """Constant-time compare, and False when either side is missing.

    An unset expected value refuses everything rather than everything-with-no-
    header: `read_token(publish=True)` is called by the middleware on every
    request, so the ordinary boot path is that the file exists by then.
    """
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented, expected)


def _refusal_body() -> bytes:
    return json.dumps({
        "error": (
            "the aggregator requires a credential in the "
            f"{AUTH_HEADER} header (#1053): this request did not come through "
            "the harness, so none of the harness's gates — grant authority, "
            "plan mode, the effect ledger, the bench/eval sandbox — were "
            "applied to it."
        ),
        "code": AUTH_ERROR_CODE,
    }).encode("utf-8")


async def _send_http(send, status: int, body: bytes, content_type: str) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", content_type.encode("latin-1")),
                            (b"content-length", str(len(body)).encode("ascii"))]})
    await send({"type": "http.response.body", "body": body})


class CredentialMiddleware:
    """Refuse any request to `app` that does not carry the credential.

    A raw ASGI middleware rather than a Starlette `BaseHTTPMiddleware`: it sits
    above the route table *and* above the SDK's Host/Origin check, and it
    decides by path rather than by endpoint — which is the property the item
    asked for: no per-endpoint call site can be forgotten. A non-HTTP scope is
    decided too rather than passed through by silence: the lifespan scope is
    forwarded (the server's own startup must still run), a websocket is closed.

    `token_fn` is a callable, not a value, so a token that does not exist when
    the app is constructed can still exist by the first request.
    """

    def __init__(self, app, token_fn=read_token):
        self.app = app
        self.token_fn = token_fn

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope.get("type") != "http":
            if scope.get("type") == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return
        if scope.get("path", "") in OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}
        expected = self.token_fn()
        if expected and token_matches(headers.get(AUTH_HEADER.lower(), ""),
                                      expected):
            await self.app(scope, receive, send)
            return
        await _send_http(send, 401, _refusal_body(), "application/json")


def require_credential(app, token_fn=read_token):
    """Put `CredentialMiddleware` on a Starlette app and return that same app.

    `add_middleware` rather than a wrapper object, so the module-level
    `agent_mcp.main.starlette_app` stays the Starlette instance it always was —
    `uvicorn.run` and the SDK both introspect `.routes` on it, and a wrapper
    would read as a plain function there. Starlette's own stack puts this
    outermost, above the router that holds both the MCP transport and the
    custom `/changes/revert` and `/browser/navigate` routes.
    """
    app.add_middleware(CredentialMiddleware, token_fn=token_fn)
    return app


def guard(app, token_fn=read_token):
    """Wrap any ASGI callable in `CredentialMiddleware`, without mutating it.

    For tests and for an app that is not a Starlette instance: the decision is
    the same class the server runs, so a test that wraps a fake app is testing
    the real control, not a copy of it.
    """
    return CredentialMiddleware(app, token_fn=token_fn)



# ── the caller side ─────────────────────────────────────────────────────────

def auth_headers(publish: bool = False) -> dict[str, str]:
    """Headers for a request to the aggregator, or `{}` if there is no credential.

    `{}` rather than a refusal, because the callers have to stay constructible:
    the dashboard and the proxies build their header dict at module import on
    some paths, and raising there would take the backend down over a missing
    file. The refusal belongs on the server, where it is a fact about the
    request; here it would be a crash with no request in it. An empty dict means
    the request goes out with no credential and is refused with 401 /
    `AGGREGATOR_UNAUTHENTICATED`, which names the condition in the response.

    `publish=False` is deliberate for every caller: minting on the client side
    could produce a value the server never agreed to, and every tool call in the
    product would then be refused by its own plumbing.
    """
    token = read_token(publish=publish)
    return {AUTH_HEADER: token} if token else {}


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def headers_for_url(url: str, publish: bool = False) -> dict[str, str]:
    """Credential headers for a request to `url`, or `{}` if it is not this box.

    The credential is a loopback secret. The MCP pool is generic — a
    `mcp_servers:` entry can name a remote host — and attaching this header to
    every HTTP server it opens would hand the aggregator's credential to a
    third party on the internet. So it goes out only to a loopback host, which is
    the only place it means anything.
    """
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return {}
    if host not in LOOPBACK_HOSTS:
        return {}
    return auth_headers(publish=publish)

