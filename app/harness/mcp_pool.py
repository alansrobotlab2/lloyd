"""Persistent MCP client for the harness.

The lloyd-mcp aggregator at http://127.0.0.1:8500/mcp owns every MCP
tool — built-ins (Bash, Read, Edit, Write, Grep, Glob, Task) plus the
existing domain modules. The harness `loop.py` reuses one process-wide
pool keyed by mcp_servers config (see `get_or_open_pool`); cleanup is
handled at FastAPI shutdown via `lifecycle.shutdown_cleanup`.

Streamable HTTP, legacy SSE and stdio transports are all wired.
lloyd-mcp speaks Streamable HTTP (MCP 2026-07-28); SSE remains for any
third-party server still on the deprecated transport. stdio is what the
Thunderbird bridge runs on (`agent_mcp.thunderbird`) — it speaks MCP over
a pipe, so it gets an SDK client rather than the hand-rolled JSON-RPC
exchange it used to have.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Sequence

from mcp import ClientSession, MCPError
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

try:  # the SDK's own client factory; private path, so the fallback is the SDK default
    import httpx2  # the httpx fork mcp 2.x is built on
    from mcp.shared._httpx_utils import create_mcp_http_client
except Exception:  # pragma: no cover - older SDK
    httpx2 = None
    create_mcp_http_client = None

from app.config import CONFIG, service_url
from app.exception_text import root_cause
from app.harness.errors import ToolDiscoveryError, ToolDispatchError

# The aggregator's request credential (#1053). Stdlib-only module, and the one
# piece of `agent_mcp` a client is allowed to import: everything else in that
# package is server-side tool code. `agent_mcp.thunderbird` imports this file,
# so the import must stay acyclic — aggregator_auth imports nothing from `app`.
from agent_mcp.aggregator_auth import headers_for_url as aggregator_headers

logger = logging.getLogger("lloyd-harness-mcp-pool")

# Default URL for the unified lloyd-mcp aggregator (agent_mcp/main.py).
DEFAULT_LLOYD_MCP_URL = service_url("lloyd_mcp", "http://127.0.0.1:8500/mcp")

# The canonical server-config for the aggregator. Callers that need a pool
# without going through `app.mcp_discovery` (the background-task drain, the
# harness's own fallback, the Task subagent) must use this rather than
# writing the dict inline.
#
# Three separate callsites used to hardcode `{"type": "sse", "url":
# DEFAULT_LLOYD_MCP_URL}`. When the aggregator moved to Streamable HTTP the
# URL constant updated and the transport did not, so an SSE client GET'd
# the /mcp endpoint and hung waiting for an `endpoint` event that never
# came — inside `get_or_open_pool`, which holds the process-wide pool-cache
# lock across `open()`. One stale literal therefore froze every agent turn
# in the process while /health stayed green.
DEFAULT_LLOYD_MCP_SERVERS: dict[str, dict[str, Any]] = {
    "lloyd-mcp": {"type": "streamable-http", "url": DEFAULT_LLOYD_MCP_URL},
}

# `_meta` key carrying the harness session id. Must match
# agent_mcp.main.META_SESSION_ID.
META_SESSION_ID = "lloyd/session_id"

# `_meta` keys carrying the CALLING turn's model and endpoint. The Task
# tool runs inside the aggregator process, which has no other way to know
# which model the turn that invoked it is running on — so a subagent used
# to be pinned to whatever `subagents.<type>.model` said (always
# `primary`), and a turn on the secondary silently delegated to the
# primary. Must match agent_mcp.main.META_MODEL / META_BASE_URL.
META_MODEL = "lloyd/model"
META_BASE_URL = "lloyd/base_url"

# `_meta` key carrying the model's own one-line caption for THIS call (the
# injected `summary` display parameter). It travels here rather than in
# `args` for the same reason the session id does — `args` is what the
# handler receives (nothing validates it against the inputSchema; unknown
# keys reach the handler as if they were parameters) — but also because a caption
# in `args` is a caption in `tool_input`, which is what the repetition guard
# hashes: two identical calls reworded would stop comparing equal.
#
# Two tools need it on the far side. `Bash(run_in_background=true)` labels
# the row it opens in the background-task list, and `Task` labels its
# subagent row. Both used to ask the model for that label with a *second*
# argument of their own — Bash's `description`, Task's `description` — whose
# wording ("Short human-readable description (informational only)") restated
# what `summary` already asks for. The model answered the question once, and
# on 2026-09-07 it started answering it in the wrong field: sessions
# `..._235236_backlogs_a8fd` and `..._000804_backlogi_3828` emitted
# `{"command": ..., "description": "check"}` for 49 consecutive Bash calls
# and no `summary` at all. Nothing errored — `description` was a real Bash
# argument, so the call was valid, and the caption vanished into a field
# only background tasks read. Must match agent_mcp.main.META_SUMMARY.
META_SUMMARY = "lloyd/summary"

# `_meta` keys identifying the chat turn and the individual tool call. The
# change ledger records a file write against (session, turn) so a chat can
# show "changed 2 files" and offer a revert; `call_id` is what ties an entry
# back to the Edit that made it. They ride in `_meta` for the same reason
# everything else here does — `args` is what the handler receives (unknown
# keys are not rejected, they reach it), and is what the repetition guard
# hashes. Must match
# agent_mcp.main.META_TURN_ID / META_CALL_ID.
META_TURN_ID = "lloyd/turn_id"
META_CALL_ID = "lloyd/call_id"

# `_meta` key carrying the #544 effect scope — the queue item a worker turn is
# running, read off `policy.current_effect_scope` by the loop. It is the
# aggregator's only way to know that this call belongs to a retry of an item it
# already served: a retried attempt gets a fresh session id, so without a scope
# carried per request there is nothing stable across the retry to key an
# idempotency ledger on. Must match agent_mcp.main.META_EFFECT_SCOPE.
META_EFFECT_SCOPE = "lloyd/effect_scope"
# The calling turn's tool surface ("chat"/"worker"), so a Task subagent runs on
# the same one. Must match agent_mcp.main.META_SURFACE.
META_SURFACE = "lloyd/surface"
# The calling turn's #534 grant scope and its live deny list, so a `Task`
# child runs under the same authority gate and cannot call what its parent
# could not (review 2026-09-24, D4). Must match agent_mcp.main.META_GRANT_SCOPE
# / META_DISALLOWED.
META_GRANT_SCOPE = "lloyd/grant_scope"
META_DISALLOWED = "lloyd/disallowed_tools"

# Ceiling on a single tools/call round trip. Sits above the Bash tool's own
# 600s hard cap so a legitimately long command finishes on its own terms and
# this only fires when something is genuinely wedged. Without it a hung tool
# blocks the harness indefinitely — there is no default in the MCP client.
CALL_TIMEOUT_SECONDS = 660.0

# The HTTP client's READ timeout has to sit above that ceiling, and it did
# not: `streamable_http_client(url)` with no client builds the SDK default,
# `httpx.Timeout(30, read=300)`. A tool that stays silent for five minutes —
# `automod_gate` with its review rung runs seven to twelve — died at the
# transport, and the retry below re-sent the request, which started a SECOND
# gate on the same round while the first was still grading. On 2026-09-11
# every review-gated round hit it: 18 rounds, 17 aborts, 0 landings, and the
# model's first sight of any review finding was the "abort and report" line.
# `read` is the gap between bytes on the response stream; the connect and
# write budgets stay short because those really are blips.
HTTP_READ_TIMEOUT_SECONDS = CALL_TIMEOUT_SECONDS + 30.0

# The per-tool call budget a server declares in a tool's `_meta` (P12). Must
# match agent_mcp.annotations.META_TIMEOUT_SECONDS. It rides in `Tool._meta`
# and not on `ToolAnnotations`, which the installed SDK (mcp 2.1) models with
# pydantic's default `extra="ignore"`: a `lloyd/timeoutSeconds` set there is
# dropped at construction and never reaches the wire. `_meta` is the field
# the spec reserves for implementation metadata, and it round-trips.
META_TIMEOUT_SECONDS = "lloyd/timeoutSeconds"

# How often a turn start re-asks the HTTP servers for tools/list, when
# `harness.mcp_pool.discovery_ttl_s` does not say. See `MCPPool.ensure_fresh`.
DEFAULT_DISCOVERY_TTL_S = 300.0

# A refresh runs on the turn path, so it is bounded: an aggregator that
# accepts the connection and then says nothing costs a turn this much, once
# per TTL, and the turn proceeds on the catalog it already had.
DISCOVERY_REFRESH_TIMEOUT_S = 10.0

# Transports that carry the 2026-07-28 stateless protocol. Nothing is pinned
# to a connection for these, so the pool does not hold one open.
HTTP_TRANSPORT_TYPES = ("http", "streamable-http", "streamable_http", "sse")


# ---------------------------------------------------------------------------
# SDK field-name compatibility
# ---------------------------------------------------------------------------
#
# mcp 2.x renames every model field to snake_case in Python (the wire format
# stays camelCase). Construction is unaffected — the models set
# populate_by_name, so `CallToolResult(isError=...)` still works — but
# ATTRIBUTE READS are not aliased, and that asymmetry is dangerous here:
#
#     getattr(result, "isError", False)
#
# returns False on a 2.x CallToolResult rather than raising, which would
# silently mark every failed tool call a success and quietly undo the
# is_error plumbing. Read through these helpers instead.


def _is_error(result: Any) -> bool:
    """`isError` (mcp 1.x) / `is_error` (mcp 2.x) from a CallToolResult."""
    value = getattr(result, "is_error", None)
    if value is None:
        value = getattr(result, "isError", None)
    return bool(value)


def _input_schema(tool: Any) -> dict[str, Any]:
    """`inputSchema` (mcp 1.x) / `input_schema` (mcp 2.x) from a Tool."""
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    return schema or {"type": "object", "properties": {}}



def _flatten_result(result: Any) -> dict[str, Any]:
    """Flatten an MCP ``CallToolResult`` into the pool's result dict.

    Text parts are joined into ``content``. ``ImageContent`` parts are
    carried out separately under ``images`` (``[{data, mime_type}]``, base64
    as the server sent it) so the loop can persist them and decide per model
    whether they reach the engine (``app/harness/tool_images.py``). The key
    is ABSENT when there are no images, so every caller that reads only
    ``content``/``is_error`` sees exactly the dict it always has. Any other
    block type (embedded resource, audio) still renders as a type stub.
    """
    text_parts: list[str] = []
    images: list[dict[str, str]] = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text is not None:
            text_parts.append(text)
            continue
        kind = getattr(item, "type", "?")
        data = getattr(item, "data", None)
        if kind == "image" and isinstance(data, str) and data:
            images.append({
                "data": data,
                "mime_type": (getattr(item, "mime_type", None)
                              or getattr(item, "mimeType", None) or "image/png"),
            })
            continue
        text_parts.append(json.dumps({"type": kind}))
    out: dict[str, Any] = {"content": "".join(text_parts), "is_error": _is_error(result)}
    if images:
        out["images"] = images
    return out

def discovery_ttl_s() -> float:
    """`harness.mcp_pool.discovery_ttl_s`; 0 (or less) turns refresh off."""
    try:
        cfg = ((CONFIG.get("harness") or {}).get("mcp_pool") or {})
        value = cfg.get("discovery_ttl_s", DEFAULT_DISCOVERY_TTL_S)
        return float(DEFAULT_DISCOVERY_TTL_S if value is None else value)
    except Exception:
        return DEFAULT_DISCOVERY_TTL_S


def per_tool_timeouts_enabled() -> bool:
    """`harness.mcp_pool.per_tool_timeouts` (default on)."""
    try:
        cfg = ((CONFIG.get("harness") or {}).get("mcp_pool") or {})
        return bool(cfg.get("per_tool_timeouts", True))
    except Exception:
        return True


@dataclass(frozen=True)
class _Catalog:
    """Everything discovery produced, as one value that is swapped, never edited.

    `open()`, `_reopen()` and `ensure_fresh()` each build a whole new catalog
    and assign it in one statement. The old code cleared four dicts in place
    and refilled them across several awaits, so a call resolving its route in
    that window read "no server claims tool" from a pool that had every tool a
    moment earlier — the empty-pool failure (CLAUDE.md) in miniature, for one
    call. A reader holding the old catalog keeps a complete one.

    The containers are plain dicts and lists for the callers that read them
    (`discovered` goes straight to `build_tool_list`), and are not mutated
    once the catalog is published. `_register()` without a builder, which
    tests use, copies into a new catalog like everything else.
    """

    discovered: list = field(default_factory=list)   # [(server, [tool dict])]
    routes: dict = field(default_factory=dict)       # bare -> server
    schemas: dict = field(default_factory=dict)      # bare -> inputSchema
    annotations: dict = field(default_factory=dict)  # bare -> ToolAnnotations dict
    timeouts: dict = field(default_factory=dict)     # bare -> declared seconds

    @property
    def tool_count(self) -> int:
        return sum(len(tools) for _srv, tools in self.discovered)

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.discovered, sort_keys=True,
                                       default=str).encode()).hexdigest()


def _build_catalog(discovered: list[tuple[str, list[dict[str, Any]]]]) -> _Catalog:
    """Fold per-server tool lists into one catalog; first server wins a name."""
    routes: dict[str, str] = {}
    schemas: dict[str, dict[str, Any]] = {}
    annotations: dict[str, dict[str, Any]] = {}
    timeouts: dict[str, float] = {}
    for server_name, tools in discovered:
        for tool in tools:
            bare = tool["name"]
            if bare in routes:
                logger.warning(
                    "mcp_pool: tool name collision on %r — %s wins over %s",
                    bare, routes[bare], server_name,
                )
                continue
            routes[bare] = server_name
            schema = tool.get("inputSchema")
            if isinstance(schema, dict):
                schemas[bare] = schema
            ann = tool.get("annotations")
            if isinstance(ann, dict):
                annotations[bare] = ann
            timeout = tool.get("timeoutSeconds")
            if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) \
                    and timeout > 0:
                timeouts[bare] = float(timeout)
    return _Catalog(discovered=list(discovered), routes=routes, schemas=schemas,
                    annotations=annotations, timeouts=timeouts)


# The current call's timing, filled by `_invoke` and read back by `call_tool`.
# A contextvar rather than a return value or an argument because `_invoke` is
# the seam half the tests replace with a five-argument fake; a fake simply
# never fills it, and the result carries no `timing`, as before.
_CALL_TIMING: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "mcp_pool_call_timing", default=None)


def _new_stats() -> dict[str, Any]:
    return {
        "calls": 0,
        "handshakes": 0,
        "handshake_ms_total": 0,
        "handshake_ms_max": 0,
        "refreshes": 0,
        "refresh_changes": 0,
        "refresh_failures": 0,
        "last_refresh_error": None,
    }


class MCPPool:
    """One-process pool that holds open MCP client sessions keyed by
    server name. `aclose()` tears them all down.

    The pool is server-config-aware: pass the same shape that
    `app.mcp_discovery._get_mcp_servers()` returns — a dict of
    {server_name: {"type": "streamable-http"|"sse"|"stdio",
                   "url"|"command"|"args": ...}}.

    All tools advertise to the model under bare MCP names. The pool
    resolves them by asking each server for its tools/list and building
    a bare-name → server map. On dispatch, bare names route to whichever
    server claims them.
    """

    def __init__(self, server_configs: dict[str, dict[str, Any]]):
        self._configs = server_configs
        # HTTP servers are connected per call; stdio servers keep a
        # persistent session held by the owner task. See `_owner_loop`.
        self._http_configs = {
            n: c for n, c in server_configs.items()
            if c.get("type", "stdio") in HTTP_TRANSPORT_TYPES
        }
        self._stdio_configs = {
            n: c for n, c in server_configs.items()
            if c.get("type", "stdio") not in HTTP_TRANSPORT_TYPES
        }
        self._sessions: dict[str, ClientSession] = {}
        # Routes, schemas, annotations (read by `_retry_safe`: a transport
        # failure mid-call is retried only for a tool the server itself calls
        # read-only or idempotent) and declared timeouts, swapped as a unit.
        self._catalog = _Catalog()
        self._catalog_at = time.monotonic()
        # What the stdio owner task discovered for the `open()` in progress.
        self._stdio_found: list[tuple[str, list[dict[str, Any]]]] = []
        self._stdio_failed: list[str] = []
        self._refresh_lock = asyncio.Lock()
        self._stats = _new_stats()
        self._opened = False
        self._open_lock = asyncio.Lock()
        self._reopen_lock = asyncio.Lock()
        # One lock per stdio server, serialising `call_tool` on its shared
        # session. Kept in a dict that outlives `_reopen`, which replaces the
        # sessions but not this map — a lock recreated under a waiter would
        # let two frames onto the same pipe.
        self._stdio_locks: dict[str, asyncio.Lock] = {}
        # Owner-task pattern: a single dedicated task holds the AsyncExitStack
        # for the SSE clients and ClientSessions. All cleanup happens in that
        # same task, avoiding anyio's "cancel scope exited in a different task"
        # error that previously left the task group spinning in
        # _deliver_cancellation at 100 % CPU.
        self._owner_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._opened_event = asyncio.Event()
        self._open_error: BaseException | None = None
        self._poisoned = False

    async def open(self) -> None:
        """Open sessions to every configured server and discover tools.

        Idempotent — concurrent callers wait on the lock and see the
        already-opened pool.
        """
        if self._opened:
            return
        async with self._open_lock:
            if self._opened:
                return

            # HTTP servers: connect once, in THIS task, purely to discover
            # tools, then close. Nothing is held afterwards — under the
            # stateless protocol each call brings its own context, and a
            # connection held open across tasks is precisely what made the
            # anyio cancel scopes fragile.
            #
            # Everything found goes into a local list and becomes the pool's
            # catalog in ONE assignment at the end (P12). A `_reopen` keeps the
            # previous catalog until then, so a call resolving its route while
            # discovery runs sees the old table, never a half-built or empty one.
            failed: list[str] = []
            found: list[tuple[str, list[dict[str, Any]]]] = []
            for server_name, cfg in self._http_configs.items():
                try:
                    async with self._http_session(cfg) as session:
                        tools = await self._list_tools(server_name, session)
                except Exception as exc:
                    logger.warning(
                        "mcp_pool: failed to discover %s: %s", server_name, exc
                    )
                    failed.append(server_name)
                    continue
                found.append((server_name, tools))

            # stdio servers: a subprocess must outlive the call, so those
            # keep the owner-task pattern.
            self._stdio_found = []
            self._stdio_failed = []
            if self._stdio_configs:
                self._owner_task = asyncio.create_task(
                    self._owner_loop(), name="mcp_pool_owner"
                )
                await self._opened_event.wait()
                if self._open_error is not None:
                    err = self._open_error
                    self._open_error = None
                    raise err
                found.extend(self._stdio_found)
                failed.extend(self._stdio_failed)

            # Checked after BOTH transports have had their turn, so a failed
            # HTTP server does not mask a healthy stdio one.
            #
            # One broken server among several must not take the whole tool
            # surface down — that is what `continue` above is for. But a
            # pool that discovered NOTHING is not degraded, it is useless,
            # and marking it `_opened` caches that uselessness process-wide
            # forever: `get_or_open_pool` short-circuits on `_opened`, and
            # nothing else ever retries discovery.
            #
            # The cost of getting this wrong is not an obvious outage. An
            # empty catalog makes `client.stream_chat` omit `tools` from the
            # request, so vLLM never engages the tool parser, and the model
            # degrades into narrating tool calls as prose instead of making
            # them (2026-09-06, twice in one afternoon, ~30 min each).
            #
            # Raise instead: `get_or_open_pool` already evicts a pool whose
            # `open()` raises, so the next caller rebuilds and re-discovers.
            # Raising here also leaves the previous catalog in place, which is
            # what a `_reopen` that fails wants.
            catalog = _build_catalog(found)
            if failed and not catalog.routes:
                raise ToolDiscoveryError(
                    "MCP discovery yielded no tools; "
                    f"failed server(s): {', '.join(failed)}",
                    servers=failed,
                )
            self._catalog = catalog
            self._catalog_at = time.monotonic()
            self._opened = True

    def _register(self, server_name: str, tools: list[dict[str, Any]]) -> None:
        """Add a server's tools to the catalog, by building a new one."""
        cur = self._current_catalog()
        self._catalog = _build_catalog(cur.discovered + [(server_name, tools)])

    # -- the catalog, and the attribute names callers and tests already use --

    def _current_catalog(self) -> _Catalog:
        # A pool built with `MCPPool.__new__` (several tests) never ran
        # `__init__`; give it an empty catalog of its own on first touch.
        cat = self.__dict__.get("_catalog")
        if cat is None:
            cat = self._catalog = _Catalog()
        return cat

    def _replace_catalog(self, **changes: Any) -> None:
        cur = self._current_catalog()
        self._catalog = _Catalog(**{**cur.__dict__, **changes})

    @property
    def _tool_routes(self) -> dict[str, str]:
        return self._current_catalog().routes

    @_tool_routes.setter
    def _tool_routes(self, value: dict[str, str]) -> None:
        self._replace_catalog(routes=value)

    @property
    def _schemas(self) -> dict[str, dict[str, Any]]:
        return self._current_catalog().schemas

    @_schemas.setter
    def _schemas(self, value: dict[str, dict[str, Any]]) -> None:
        self._replace_catalog(schemas=value)

    @property
    def _annotations(self) -> dict[str, dict[str, Any]]:
        return self._current_catalog().annotations

    @_annotations.setter
    def _annotations(self, value: dict[str, dict[str, Any]]) -> None:
        self._replace_catalog(annotations=value)

    def timeout_for(self, bare: str) -> float:
        """The call budget for `bare`: its declared timeout, clamped.

        A server declares a tool's budget as `lloyd/timeoutSeconds` in the
        tool's `_meta` (`agent_mcp/annotations.py::TIMEOUT_SECONDS`). Never
        above `CALL_TIMEOUT_SECONDS` — the HTTP read timeout is sized against
        that ceiling, so a longer declared budget would die at the transport
        and read as a transport error — and never below one second. An
        undeclared tool, or `harness.mcp_pool.per_tool_timeouts: false`, gets
        the ceiling, which is today's behaviour.
        """
        declared = self._current_catalog().timeouts.get(bare)
        if declared is None or not per_tool_timeouts_enabled():
            return CALL_TIMEOUT_SECONDS
        return max(1.0, min(float(declared), CALL_TIMEOUT_SECONDS))

    def _retry_safe(self, bare: str) -> bool:
        """May a call to `bare` be re-sent after a transport failure?

        Only when the server annotated it `readOnlyHint` or `idempotentHint`.
        A transport error says nothing about whether the server ran the
        call — for a long one it almost certainly did — and re-sending a
        mutating call is how one `automod_gate` became two concurrent gates
        of the same round. An unannotated tool is not retried: that is the
        `annotations.py` contract (a server that sets no hints qualifies
        nothing), and the cost of being wrong that way is one tool error
        the model can read, not a duplicated side effect it cannot see.
        """
        ann = self._annotations.get(bare) or {}
        return bool(ann.get("readOnlyHint") or ann.get("idempotentHint"))

    @asynccontextmanager
    async def _http_session(self, cfg: dict[str, Any]):
        """An MCP session over HTTP, entered and exited in the caller's task.

        Deliberately short-lived. The 2026-07-28 core is stateless — no
        session is pinned to the server — so a connection buys nothing and
        costs the thing that made this file hard: `streamable_http_client`
        runs an anyio task group, and holding one across the boundary
        between the task that entered it and the task that uses it is what
        anyio's cancel scopes will not tolerate. Reconnecting measures in
        single-digit milliseconds.
        """
        url = cfg["url"]
        # The aggregator refuses a request that carries no credential (#1053);
        # this is the harness, so it is the caller the credential is issued to.
        # Resolved per session, never cached at import: the aggregator mints the
        # value on its own first boot, and a header dict built once at import
        # would freeze "absent" across that boot for the life of the backend.
        headers = aggregator_headers(url)
        async with AsyncExitStack() as stack:
            if cfg.get("type") == "sse":
                ctx = sse_client(url, headers=headers or None)
            else:
                # A client we hand in is a client we own: the SDK closes only
                # the one it built itself.
                client = _http_client(headers=headers)
                if client is not None:
                    await stack.enter_async_context(client)
                ctx = streamable_http_client(url, http_client=client)
            read_stream, write_stream = await stack.enter_async_context(ctx)
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream))
            await session.initialize()
            yield session

    async def _owner_loop(self) -> None:
        """Hold every SSE/ClientSession context for the pool's lifetime.

        Opens all configured servers under one AsyncExitStack, signals the
        opener via `_opened_event`, then parks on `_shutdown_event`. When
        shutdown fires (or any context raises), the AsyncExitStack unwinds
        in this same task — keeping anyio's cancel scopes consistent.
        """
        try:
            async with AsyncExitStack() as stack:
                for server_name, cfg in self._stdio_configs.items():
                    try:
                        session = await self._open_session(stack, server_name, cfg)
                    except Exception as exc:
                        logger.warning(
                            "mcp_pool: failed to open %s: %s", server_name, exc
                        )
                        self._stdio_failed.append(server_name)
                        continue
                    self._sessions[server_name] = session
                    # Collected, not registered: `open()` publishes the whole
                    # catalog at once when this task signals it is done.
                    self._stdio_found.append(
                        (server_name, await self._list_tools(server_name, session)))
                self._opened_event.set()
                await self._shutdown_event.wait()
        except BaseException as exc:
            # Open failed, or a child task in one of the SSE task groups
            # propagated. Surface to whoever's awaiting open(), then let
            # the AsyncExitStack finish unwinding in this task.
            self._open_error = exc
            self._opened_event.set()
            self._poisoned = True
            if not isinstance(exc, Exception):
                raise

    async def aclose(self) -> None:
        """Signal the owner task to tear down, then await it.

        Cleanup runs in the owner task — never in the caller's task — so
        anyio cancel scopes always exit in the task that entered them.
        """
        self._shutdown_event.set()
        if self._owner_task is not None:
            try:
                await self._owner_task
            except Exception as exc:
                logger.warning("mcp_pool: owner task exited with %s", exc)
            self._owner_task = None
        self._sessions.clear()
        self._catalog = _Catalog()
        self._opened = False

    @property
    def discovered(self) -> list[tuple[str, list[dict[str, Any]]]]:
        """List of (server_name, mcp_tools_list) pairs ready for
        `app.harness.tool_schema.build_tool_list`.

        The current catalog's list. A caller that reads it several times in
        one step should read it once: a refresh by another turn may publish a
        new catalog between two reads (never an emptier one on failure).
        """
        return self._current_catalog().discovered

    async def ensure_fresh(self, ttl_s: float | None = None) -> bool:
        """Re-run tools/list on the HTTP servers when the catalog is older than
        `ttl_s` (default `harness.mcp_pool.discovery_ttl_s`). True when a new
        catalog was published.

        Called at a turn boundary only (`loop._build_pool`), so the tools a
        turn advertised stay fixed for the whole turn. Discovery used to run
        once per pool, i.e. once per backend process: a tool added, removed or
        re-described in the aggregator was invisible until the backend
        restarted, while the aggregator itself caches its list for 60 s.

        The rules, all about never making things worse than a stale list:

        - a server that fails keeps its previous entry, and a refresh that
          would leave no tools at all publishes nothing — the empty pool is
          the worst failure in the system, and a refresh must not be a new way
          to reach it;
        - unchanged discovery keeps the current catalog object;
        - the attempt is stamped either way, so a down aggregator costs one
          bounded attempt per TTL, not one per turn;
        - a turn that finds another turn refreshing does not wait for it;
        - stdio servers are not re-listed: their session is held open and
          `_reopen` rediscovers them.

        `notifications/tools/list_changed` is not implementable here: an HTTP
        session lives for exactly one call (`_http_session`), so there is no
        open session for the server to notify. A TTL poll is the substitute,
        and the aggregator's own `ttl_ms` on tools/list is the same idea from
        the other side.
        """
        ttl = discovery_ttl_s() if ttl_s is None else float(ttl_s)
        if ttl <= 0 or not self._opened or not self._http_configs:
            return False
        if time.monotonic() - self._catalog_at < ttl:
            return False
        if self._refresh_lock.locked():
            return False
        async with self._refresh_lock:
            if time.monotonic() - self._catalog_at < ttl:
                return False
            self._catalog_at = time.monotonic()
            stats = self._stat()
            stats["refreshes"] += 1
            old = self._current_catalog()
            previous = dict(old.discovered)
            found: list[tuple[str, list[dict[str, Any]]]] = []
            errors: list[str] = []
            for server_name, cfg in self._http_configs.items():
                try:
                    async with asyncio.timeout(DISCOVERY_REFRESH_TIMEOUT_S):
                        async with self._http_session(cfg) as session:
                            tools = await self._list_tools(server_name, session)
                except Exception as exc:
                    errors.append(f"{server_name}: {root_cause(exc)}")
                    if server_name in previous:
                        found.append((server_name, previous[server_name]))
                    continue
                found.append((server_name, tools))
            found.extend((n, t) for n, t in old.discovered
                         if n in self._stdio_configs)
            if errors:
                stats["refresh_failures"] += 1
                stats["last_refresh_error"] = "; ".join(errors)[:300]
                logger.warning("mcp_pool: discovery refresh failed (%s); keeping "
                               "the previous entries", "; ".join(errors))
            new = _build_catalog(found)
            if not new.routes:
                return False
            if new.fingerprint() == old.fingerprint():
                return False
            old_names, new_names = set(old.routes), set(new.routes)
            logger.info(
                "mcp_pool: discovery refresh published %d tools (+%s -%s)",
                new.tool_count, sorted(new_names - old_names)[:10],
                sorted(old_names - new_names)[:10],
            )
            self._catalog = new
            stats["refresh_changes"] += 1
            return True

    def _stat(self) -> dict[str, Any]:
        stats = self.__dict__.get("_stats")
        if stats is None:
            stats = self._stats = _new_stats()
        return stats

    def stats(self) -> dict[str, Any]:
        """Discovery and handshake counters, for `/health/deep`."""
        stats = dict(self._stat())
        cat = self._current_catalog()
        n = stats["handshakes"]
        stats["handshake_ms_avg"] = round(stats["handshake_ms_total"] / n, 1) if n else None
        stats["tools"] = len(cat.routes)
        stats["declared_timeouts"] = len(cat.timeouts)
        stats["servers"] = sorted(self._configs)
        at = self.__dict__.get("_catalog_at")
        stats["catalog_age_s"] = (round(time.monotonic() - at, 1)
                                  if at is not None else None)
        stats["opened"] = bool(self.__dict__.get("_opened"))
        return stats

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session_id: str = "",
        model: str = "",
        base_url: str = "",
        summary: str = "",
        turn_id: str = "",
        call_id: str = "",
        effect_scope: str = "",
        surface: str = "",
        grant_scope: str = "",
        disallowed_tools: Sequence[str] = (),
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Dispatch a tool call to the right server.

        `name` is normally a bare tool name (`"Bash"`, `"email_recent"`).
        The legacy ``mcp__server__tool`` form is still accepted so old
        persisted session JSON replays cleanly. Returns
        ``{"content": str, "is_error": bool}``. Raises ToolDispatchError
        when routing or transport fails — caller maps to a tool_result
        with ``is_error=True``.

        `session_id`, `model`, `base_url` and `summary` travel in the
        request's ``_meta``, not in ``args``.
        Nothing validates ``args`` against the tool's inputSchema: top-level
        primitives are coerced only (``_coerce_args``) and unknown keys reach
        the handler, so anything injected there arrives as though it were a
        real parameter; ``_meta`` is the field the spec reserves for
        exactly this kind of implementation metadata.

        `effect_scope` (#544) rides in ``_meta`` for the same reason and is
        the queue item a worker turn is running; the aggregator's effect
        ledger keys on it so a retried attempt cannot fire the same side
        effect twice. Empty means no ledger for this call — an interactive
        turn, or any caller that is not a retryable queue item.

        `grant_scope` and `disallowed_tools` are what a `Task` child inherits:
        the authority scope the calling turn's grant gate checks, and the deny
        list in force for this iteration. Both are omitted when empty.
        """
        if not self._opened:
            await self.open()

        # Resolve to (server_name, bare_tool_name).
        if name.startswith("mcp__"):
            rest = name[len("mcp__"):]
            sep = rest.find("__")
            if sep > 0:
                server_name = rest[:sep]
                bare = rest[sep + 2:]
            else:
                raise ToolDispatchError(name, "malformed namespaced tool name")
        else:
            bare = name
            server_name = self._tool_routes.get(bare, "")
            if not server_name:
                raise ToolDispatchError(
                    name, f"no server claims tool {bare!r}"
                )

        if server_name not in self._http_configs and server_name not in self._sessions:
            # HTTP servers have no held session by design; only a stdio
            # server can be "not open".
            raise ToolDispatchError(name, f"server {server_name!r} not open")

        coerced = _coerce_args(args, self._schemas.get(bare))
        meta: dict[str, Any] = {}
        if session_id:
            meta[META_SESSION_ID] = session_id
        if model:
            meta[META_MODEL] = model
        if base_url:
            meta[META_BASE_URL] = base_url
        if summary:
            meta[META_SUMMARY] = summary
        if turn_id:
            meta[META_TURN_ID] = turn_id
        if call_id:
            meta[META_CALL_ID] = call_id
        if effect_scope:
            meta[META_EFFECT_SCOPE] = effect_scope
        if surface:
            meta[META_SURFACE] = surface
        if grant_scope:
            meta[META_GRANT_SCOPE] = grant_scope
        if disallowed_tools:
            meta[META_DISALLOWED] = list(disallowed_tools)
        # A caller's explicit budget wins; otherwise the tool's own declared
        # one (P12), which is never above the ceiling.
        budget = timeout_seconds if timeout_seconds is not None else self.timeout_for(bare)
        timing: dict[str, int] = {}
        timing_token = _CALL_TIMING.set(timing)

        try:
            result = await self._invoke(
                server_name, bare, coerced, budget, meta or None
            )
        except MCPError as exc:
            # A protocol-level error from a live server: it was reachable
            # and it answered. That is a failed *call*, not a failed
            # transport — surface it as a tool error and leave the pool be.
            logger.info("mcp_pool: %s on %s returned an MCP error: %s", bare, server_name, exc)
            return {"content": f"MCP error calling {bare}: {exc}", "is_error": True}
        except Exception as exc:
            # Transport-shaped failure. Under the 2026-07-28 stateless core
            # there is no session pinned to the server, so this is just a
            # dropped connection: reopen and try once more. Reconnecting
            # costs single-digit milliseconds, which is why the old
            # behaviour — poison the pool, evict it from the cache, tear
            # down the session every concurrent turn was sharing — is no
            # longer a trade worth making.
            #
            # ...for a call that is safe to send twice. The server may well
            # have RUN the first one — a read timeout fires while the tool is
            # still working — and a second copy of a mutating call is a
            # duplicated side effect the model never sees. Those are
            # surfaced as a tool error that says so, and not retried.
            #
            # The cause goes through `root_cause`, not the raw exception: a
            # collapsed server-side TaskGroup stringifies to "unhandled errors
            # in a TaskGroup (1 sub-exception)" and nothing else (#936), which
            # tells the caller neither what failed nor whether "read its state
            # back" has anything to look for. The group stays on the chain via
            # `from exc`, so the traceback is still intact for the log.
            cause = root_cause(exc)
            if not self._retry_safe(bare):
                logger.warning(
                    "mcp_pool: %s on %s failed (%s); not retried — the tool is not "
                    "annotated read-only or idempotent and may have run",
                    bare, server_name, cause,
                )
                raise ToolDispatchError(
                    name,
                    f"transport error: {cause}. The call was NOT retried because {bare} "
                    f"is not idempotent and the server may have run it — read its "
                    f"state back (status, log, marker file) before calling it again.",
                ) from exc
            logger.warning(
                "mcp_pool: %s on %s failed (%s); reconnecting and retrying once",
                bare, server_name, cause,
            )
            try:
                await self._reopen()
                result = await self._invoke(
                server_name, bare, coerced, budget, meta or None
            )
            except MCPError as retry_exc:
                return {
                    "content": f"MCP error calling {bare}: {retry_exc}",
                    "is_error": True,
                }
            except Exception as retry_exc:
                # Still failing after a fresh connection: the server is
                # genuinely down, not a blip. Evict so the next turn builds
                # a new pool rather than reusing this one.
                #
                # Same unwrap as the site above (#936): this is the error that
                # concludes the server is down, and a group summary here said
                # both "down" and nothing about why.
                self._poisoned = True
                _evict_pool(self)
                self._shutdown_event.set()
                retry_cause = root_cause(retry_exc)
                logger.warning(
                    "mcp_pool: %s on %s failed again after reconnect (%s); pool evicted",
                    bare, server_name, retry_cause,
                )
                raise ToolDispatchError(
                    name, f"transport error: {retry_cause}"
                ) from retry_exc
        finally:
            _CALL_TIMING.reset(timing_token)

        out = _flatten_result(result)
        stats = self._stat()
        stats["calls"] += 1
        if timing:
            # P11's `tool_result.handshake_ms` reads exactly this key.
            out["timing"] = dict(timing)
            hs = timing.get("handshake_ms")
            if hs is not None:
                stats["handshakes"] += 1
                stats["handshake_ms_total"] += hs
                stats["handshake_ms_max"] = max(stats["handshake_ms_max"], hs)
        return out

    async def _invoke(
        self,
        server_name: str,
        bare: str,
        args: dict[str, Any],
        budget: float,
        meta: dict[str, Any] | None,
    ):
        """One tools/call, on a fresh HTTP session or the stdio one.

        The HTTP path opens a session per call and is safe to run
        concurrently. The stdio path shares ONE `ClientSession` over one pair
        of pipes, and two concurrent `call_tool`s on it interleave their
        JSON-RPC frames — so it takes a per-server lock. The lock is keyed by
        server name and kept in a dict that survives `_reopen`, which
        replaces the sessions but not this map.

        The HTTP path times its two halves (P12): `handshake_ms` is entering
        `_http_session` — connect plus `initialize()`, paid on every call
        because the session is per call — and `call_ms` is the tools/call
        itself. They land in `_CALL_TIMING` for `call_tool` to return as
        `result["timing"]`. stdio has no per-call handshake and reports none.
        """
        cfg = self._http_configs.get(server_name)
        if cfg is not None:
            timing = _CALL_TIMING.get()
            started = time.perf_counter()
            async with self._http_session(cfg) as session:
                entered = time.perf_counter()
                if timing is not None:
                    timing["handshake_ms"] = max(0, int((entered - started) * 1000))
                result = await session.call_tool(
                    bare, args, read_timeout_seconds=budget, meta=meta,
                )
                if timing is not None:
                    timing["call_ms"] = max(0, int((time.perf_counter() - entered) * 1000))
                return result
        lock = self._stdio_locks.get(server_name)
        if lock is None:
            lock = self._stdio_locks[server_name] = asyncio.Lock()
        async with lock:
            session = self._sessions.get(server_name)
            if session is None:
                raise ToolDispatchError(bare, f"server {server_name!r} not open")
            return await session.call_tool(
                bare, args, read_timeout_seconds=budget, meta=meta,
            )

    async def _reopen(self) -> None:
        """Tear down and rebuild every session, in place.

        Keeps this MCPPool instance (and therefore its cache entry and its
        tool routes) valid, so callers holding a reference keep working. The
        catalog is NOT cleared: `open()` swaps the rediscovered one in whole,
        and a reopen that fails leaves the previous one standing (P12). The
        old in-place `.clear()` of the routes left a window, across the owner
        task's teardown and the whole rediscovery, in which every concurrent
        call read "no server claims tool".
        """
        async with self._reopen_lock:
            if not self._stdio_configs:
                # HTTP-only pool: every call already opens its own session,
                # so there is nothing to rebuild — just clear the failure
                # flag and let the retry go out on a fresh connection.
                self._poisoned = False
                return
            self._shutdown_event.set()
            if self._owner_task is not None:
                try:
                    await self._owner_task
                except Exception as exc:
                    logger.debug("mcp_pool: owner task exit during reopen: %s", exc)
                self._owner_task = None
            self._sessions.clear()
            self._opened = False
            self._poisoned = False
            self._shutdown_event = asyncio.Event()
            self._opened_event = asyncio.Event()
            self._open_error = None
            await self.open()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _open_session(
        self,
        stack: AsyncExitStack,
        server_name: str,
        cfg: dict[str, Any],
    ) -> ClientSession:
        """Enter the SSE client + ClientSession contexts on the supplied
        ``stack``. The stack belongs to the owner task, so cleanup runs in
        the same task that entered the contexts.
        """
        server_type = cfg.get("type", "stdio")
        if server_type in ("http", "streamable-http", "streamable_http"):
            # Same credential as the per-call path above: an un-credentialed
            # session here would fail discovery, not just dispatch, and would
            # read as "this server advertises nothing".
            client = _http_client(headers=aggregator_headers(cfg["url"]))
            if client is not None:
                await stack.enter_async_context(client)
                ctx = streamable_http_client(cfg["url"], http_client=client)
            else:
                ctx = streamable_http_client(cfg["url"])
        elif server_type == "sse":
            # Legacy HTTP+SSE transport, deprecated upstream as of
            # 2026-07-28 with a 12-month removal window. Kept for any
            # third-party server that hasn't moved; lloyd-mcp is on
            # streamable HTTP.
            ctx = sse_client(cfg["url"], headers=aggregator_headers(cfg["url"]) or None)
        elif server_type == "stdio":
            command = cfg.get("command")
            if not command:
                raise ValueError(f"stdio server {server_name!r} has no 'command'")
            env = cfg.get("env")
            ctx = stdio_client(
                StdioServerParameters(
                    command=command,
                    args=list(cfg.get("args") or []),
                    env=dict(env) if env else None,
                    cwd=cfg.get("cwd") or None,
                )
            )
        else:
            raise ValueError(
                f"mcp_pool: unknown transport {server_type!r} for {server_name!r}"
            )

        read_stream, write_stream = await stack.enter_async_context(ctx)
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        return session

    async def _list_tools(
        self, server_name: str, session: ClientSession
    ) -> list[dict[str, Any]]:
        result = await session.list_tools()
        return [_tool_dict(t) for t in result.tools]


def _tool_dict(t: Any) -> dict[str, Any]:
    """One tools/list entry as the plain dict the catalog and `build_tool_list`
    read. `timeoutSeconds` is present only when the server declared one."""
    out = {
        "name": t.name,
        "description": t.description or "",
        "inputSchema": _input_schema(t),
        # Carried through, not dropped. `readOnlyHint` is what lets a
        # consumer decide whether a batch of calls can run
        # concurrently without asking a second, private list of tool
        # names — which is the pattern `agent_mcp/annotations.py` was
        # written to replace. A server that sets no hints qualifies
        # nothing, which is the contract.
        "annotations": _annotations(t),
    }
    timeout = _declared_timeout(t)
    if timeout is not None:
        out["timeoutSeconds"] = timeout
    return out


def _declared_timeout(tool: Any) -> float | None:
    """`lloyd/timeoutSeconds` from a Tool's `_meta` (`meta` on mcp 2.x)."""
    meta = getattr(tool, "meta", None)
    if meta is None:
        meta = getattr(tool, "_meta", None)
    if not isinstance(meta, dict):
        return None
    value = meta.get(META_TIMEOUT_SECONDS)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _http_client(headers: dict[str, str] | None = None):
    """An httpx client whose read timeout outlives `CALL_TIMEOUT_SECONDS`.

    None when the SDK's factory is unavailable, which hands
    `streamable_http_client` its own default (read=300) — the pre-2026-09-11
    behaviour, kept only as the fallback for an SDK without the helper. That
    fallback sends no credential, so a server enforcing #1053 refuses the call
    rather than accepting an unauthenticated one.
    """
    if create_mcp_http_client is None or httpx2 is None:
        return None
    return create_mcp_http_client(
        headers=headers or None,
        timeout=httpx2.Timeout(60.0, read=HTTP_READ_TIMEOUT_SECONDS))


def _annotations(tool: Any) -> dict[str, Any]:
    """A tool's ToolAnnotations as a plain dict, or {} when it has none.

    Both SDK naming conventions, like `_input_schema` next door: mcp 2.x
    renamed the model fields to snake_case in Python while the wire keeps
    camelCase, and a pool talking to a server on the other version must not
    silently read every tool as unannotated.
    """
    ann = getattr(tool, "annotations", None)
    if ann is None:
        return {}
    out: dict[str, Any] = {}
    for wire, snake in (("readOnlyHint", "read_only_hint"),
                        ("destructiveHint", "destructive_hint"),
                        ("idempotentHint", "idempotent_hint"),
                        ("openWorldHint", "open_world_hint")):
        value = getattr(ann, snake, None)
        if value is None:
            value = getattr(ann, wire, None)
        if value is not None:
            out[wire] = bool(value)
    return out


# ---------------------------------------------------------------------------
# Argument coercion
# ---------------------------------------------------------------------------


def _coerce_args(args: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce primitive args to match the tool's inputSchema.

    vLLM occasionally emits string-shaped scalars (`"5"` for an integer
    field, `"true"` for a boolean) and a handler reading a typed field
    would get the wrong type. We coerce best-effort using the declared
    `type` of each top-level property; anything ambiguous is passed through
    untouched. This is the only shape defense on the path: nothing
    validates arguments against the inputSchema, and unknown keys reach the
    handler, whose own checks are what surface a bad argument.
    """
    if not isinstance(args, dict) or not isinstance(schema, dict):
        return args
    props = schema.get("properties")
    if not isinstance(props, dict):
        return args
    out = dict(args)
    for k, v in args.items():
        prop = props.get(k)
        if not isinstance(prop, dict):
            continue
        target = prop.get("type")
        if isinstance(target, list):
            target = next((t for t in target if t != "null"), None)
        out[k] = _coerce_one(v, target)
    return out


def _coerce_one(v: Any, target: Any) -> Any:
    if v is None or target is None:
        return v
    if target == "integer":
        if isinstance(v, bool) or isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if s.lstrip("-").isdigit():
                try:
                    return int(s)
                except ValueError:
                    return v
        return v
    if target == "number":
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            try:
                return float(v.strip())
            except ValueError:
                return v
        return v
    if target == "boolean":
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1", "yes"):
                return True
            if s in ("false", "0", "no", ""):
                return False
        if isinstance(v, int):
            return bool(v)
        return v
    if target == "string":
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        return v
    return v


# ---------------------------------------------------------------------------
# Process-wide pool cache
# ---------------------------------------------------------------------------

_POOL_CACHE: dict[str, MCPPool] = {}
_POOL_CACHE_LOCK = asyncio.Lock()


def _config_key(server_configs: dict[str, dict[str, Any]]) -> str:
    """Stable hashable key for a server-config dict."""
    return json.dumps(server_configs, sort_keys=True, default=str)


async def get_or_open_pool(server_configs: dict[str, dict[str, Any]]) -> MCPPool:
    """Return a process-wide MCPPool for `server_configs`, opening on first use.

    Concurrent callers serialize on a lock so the SSE handshake +
    `tools/list` only runs once per unique config. Subsequent turns reuse
    the open sessions.

    Pools that hit transport failures are marked `_poisoned` and evicted
    via `_evict_pool` from `call_tool`. If a poisoned pool is somehow
    still in the cache when a new caller arrives (race: evicted but the
    same instance got re-cached), we treat it as missing and rebuild.
    """
    key = _config_key(server_configs)
    async with _POOL_CACHE_LOCK:
        pool = _POOL_CACHE.get(key)
        if pool is None or pool._poisoned:
            pool = MCPPool(server_configs)
            _POOL_CACHE[key] = pool
        elif pool._opened:
            return pool

    # Open OUTSIDE the process-wide cache lock.
    #
    # This lock used to be held across `await pool.open()`, which made any
    # interruption of a first open a permanent, process-wide deadlock:
    # `run_query` is an async generator, and `autonomy.run_task` iterates
    # it under `asyncio.timeout`. When that timeout fires mid-open the
    # generator is abandoned while suspended — its `async with` never
    # unwinds, so the lock is never released — and every later caller,
    # including every user turn, blocks on it forever with no error
    # anywhere. `MCPPool.open()` is idempotent and serialized by its own
    # per-pool lock, so the global one only needs to guard the dict.
    try:
        await pool.open()
    except BaseException:
        # Includes CancelledError: never leave a half-open pool cached for
        # the next caller to inherit.
        async with _POOL_CACHE_LOCK:
            if _POOL_CACHE.get(key) is pool:
                _POOL_CACHE.pop(key, None)
        raise
    return pool


def _evict_pool(pool: MCPPool) -> None:
    """Drop a poisoned pool from the cache.

    Synchronous best-effort: drop the cache entry only. Closing the pool
    is handled by the owner task once `_shutdown_event` is set (see
    ``MCPPool.call_tool``); we never await across tasks here, so we don't
    need to do any async work in this function.

    Two concurrent evictions race harmlessly: dict ops are atomic under
    the GIL, and the second pop sees the entry already gone.
    """
    for k, p in list(_POOL_CACHE.items()):
        if p is pool:
            _POOL_CACHE.pop(k, None)
            return


def pool_stats() -> list[dict[str, Any]]:
    """`MCPPool.stats()` for every cached pool (P12, `/health/deep`)."""
    out = []
    for pool in list(_POOL_CACHE.values()):
        try:
            out.append(pool.stats())
        except Exception as exc:  # a stats bug must not take the probe down
            out.append({"error": str(exc)[:200]})
    return out


async def close_all_pools() -> None:
    """Close every cached pool. Call from FastAPI shutdown."""
    async with _POOL_CACHE_LOCK:
        for pool in list(_POOL_CACHE.values()):
            try:
                await pool.aclose()
            except Exception as exc:
                logger.warning("close_all_pools: %s", exc)
        _POOL_CACHE.clear()
