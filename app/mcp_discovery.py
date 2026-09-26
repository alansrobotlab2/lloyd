"""MCP server configuration + tool discovery.

`_get_mcp_servers()` builds the live server-config dict for the harness
from `CONFIG`. We deliberately do not cache this at module scope —
callers invoke the function each request so config edits via
`/api/tool-toggle` take effect immediately without requiring any
module-level rebinding.

All tools (built-in + domain) are advertised to the model under their
bare MCP name. To disable any tool, add its bare name to
`mcp_servers.<server>.disabled_tools`.
"""

import asyncio

from app.config import CONFIG
# The unwrapper is shared with tool dispatch in `app/harness/mcp_pool.py`,
# which had the same opaque "unhandled errors in a TaskGroup" string in front
# of the model and never unwrapped it (#936). One implementation, not one per
# caller — see the module docstring for why it sits outside `app.harness`.
from app.exception_text import root_cause

_MCP_SERVER_META: dict[str, dict] = {
    "lloyd-mcp": {
        "label": "Lloyd MCP",
        "description": "Unified aggregator: built-in tools (Bash/Read/Write/Edit/Grep/Glob/Task) + domain modules (autonomy, backlog, browser, facts, vault, mission control, HTTP, Thunderbird, pipeline, ambient, autoresearch, skills, session)",
    },
}

_tools_cache: dict[str, dict] = {}  # {server_name: {tools, error, ts}}
_TOOLS_CACHE_TTL = 300.0  # 5 minutes


# Tool → category mapping. Categories are derived from the source agent_mcp
# module each tool lives in (e.g., facts.py → "Memory: Facts"). Built-in
# tools are bare-named; everything else uses a stable module prefix.
# Order matters for prefix rules — first match wins.
_TOOL_EXACT_CATEGORY: dict[str, str] = {
    "Bash": "Shell",
    "Read": "Filesystem",
    "Write": "Filesystem",
    "Edit": "Filesystem",
    "Grep": "Filesystem",
    "Glob": "Filesystem",
    "Task": "Agents",
    "session_recall": "Memory: Session",
    "session_inject_context": "Memory: Session",
    "ambient_decide": "Ambient",
}

_TOOL_PREFIX_CATEGORY: list[tuple[str, str]] = [
    ("fact_", "Memory: Facts"),
    ("vault_", "Memory: Vault"),
    ("memory_", "Memory: Session"),
    ("ambient_", "Ambient"),
    ("autonomy_", "Autonomy"),
    ("autoresearch_", "Autoresearch"),
    ("backlog_", "Backlog"),
    ("browser_", "Browser"),
    ("discord_", "Discord"),
    ("chat_", "Mission Control"),
    ("skills_", "Skills"),
    ("http_", "HTTP"),
    ("tb_", "Thunderbird"),
    ("email_", "Email"),
    ("calendar_", "Calendar"),
    ("contacts_", "Contacts"),
]


def _categorize_tool(name: str) -> str:
    """Return the user-facing category label for a tool name."""
    if name in _TOOL_EXACT_CATEGORY:
        return _TOOL_EXACT_CATEGORY[name]
    for prefix, label in _TOOL_PREFIX_CATEGORY:
        if name.startswith(prefix):
            return label
    return "Other"


# Transports MCPPool._open_session knows how to open. Kept here so an
# unrecognized `type:` fails loudly at config-read time.
HTTP_TRANSPORTS = ("http", "streamable-http", "streamable_http", "sse")
STDIO_TRANSPORTS = ("stdio",)


def _get_mcp_servers() -> dict[str, dict]:
    """Build MCP server configs for the harness pool, skipping disabled servers.

    An unknown `type:` raises rather than falling through to stdio. The
    fall-through used to be silent, and it was: when lloyd-mcp moved to
    `type: streamable-http`, this function's whitelist still only knew
    ("sse", "http"), so it emitted `{"command": "python", "args": []}` —
    a config that spawns a bare Python REPL and hangs the pool open
    forever. `/health` stayed green throughout, because the aggregator
    itself was fine; only the client's view of it was broken.
    """
    servers: dict[str, dict] = {}
    for name, cfg in CONFIG.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        server_type = cfg.get("type", "stdio")
        if server_type in HTTP_TRANSPORTS:
            url = cfg.get("url")
            if not url:
                raise ValueError(f"mcp server {name!r} has type {server_type!r} but no url")
            servers[name] = {"type": server_type, "url": url}
        elif server_type in STDIO_TRANSPORTS:
            entry = {
                "type": "stdio",
                "command": cfg.get("command", "python"),
                "args": list(cfg.get("args") or []),
            }
            if cfg.get("env"):
                entry["env"] = dict(cfg["env"])
            if cfg.get("cwd"):
                entry["cwd"] = cfg["cwd"]
            servers[name] = entry
        else:
            raise ValueError(
                f"mcp server {name!r} has unknown transport type {server_type!r}; "
                f"expected one of {HTTP_TRANSPORTS + STDIO_TRANSPORTS}"
            )
    return servers


# Plan B — tools that primary cannot use while in plan mode. The plan
# ritual is research-only: read tools, ToolSearch, TodoWrite, and the
# plan-mode tools themselves stay allowed; the actuator tools are
# blocked until ExitPlanMode commits or cancel flips plan_mode off.
#
# This tuple is now only the FLOOR. The real list is derived from the
# `readOnlyHint` annotations in `agent_mcp.annotations` (see
# `_plan_mode_blocked`), which covers every actuator rather than the three
# that happened to be listed here — `email_send`, `vault_write`,
# `discord_send`, `fact_add` and ~55 others used to sail straight through
# a "read-only" plan-mode turn.
PLAN_MODE_BLOCKED_TOOLS = ("Write", "Edit", "Bash")

# Tool-name universe, recorded by the harness after MCP discovery. Plan
# mode needs to know which tools exist in order to block the mutating
# ones; discovery is the only place that knows, and it happens after the
# routers have already constructed their RunOptions.
_TOOL_UNIVERSE: set[str] = set()


def record_tool_universe(names) -> None:
    """Record the discovered tool names for annotation-derived gating.

    Called by the harness once per pool open. Idempotent and cheap; a
    superset across servers is fine because the gate only ever removes
    names that are actually advertised.
    """
    if names:
        _TOOL_UNIVERSE.update(names)


def _plan_mode_blocked() -> list[str]:
    """Actuator tools to block while plan mode is active.

    Falls back to the three-tool floor until discovery has run — better a
    narrow gate than an empty one, and the very next turn has the full
    universe recorded.
    """
    if not _TOOL_UNIVERSE:
        return list(PLAN_MODE_BLOCKED_TOOLS)
    from agent_mcp.annotations import plan_mode_blocked_tools

    return plan_mode_blocked_tools(_TOOL_UNIVERSE)


def _get_disallowed_tools(plan_mode: bool = False) -> list[str]:
    """Build disallowed_tools list from config for harness RunOptions.

    Tools are advertised by bare name, so the disallow list uses bare
    names too. The legacy ``mcp__server__tool`` form is also recognized
    by ``build_tool_list`` for any rolled-forward configs.

    `plan_mode` (Plan B) — when true, append the actuator tools
    (Write, Edit, Bash) so primary cannot mutate state while drafting
    a plan. Read tools, TodoWrite, ToolSearch, EnterPlanMode, and
    ExitPlanMode all remain allowed.
    """
    disallowed: list[str] = []
    for _server_name, cfg in CONFIG.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        for tool_name in cfg.get("disabled_tools", []):
            disallowed.append(tool_name)
    if plan_mode:
        for name in _plan_mode_blocked():
            if name not in disallowed:
                disallowed.append(name)
    return disallowed


DISCOVERY_TIMEOUT_SECONDS = 30.0


def _get_harness_kwargs() -> dict:
    """Resolve `harness.*` config into RunOptions kwargs.

    Splatted into RunOptions(**...) at every construction site (chat
    streaming, ambient, sync, voice, autonomy). Defaults align with
    RunOptions's own dataclass defaults so missing config keys behave
    sanely.
    """
    harness = CONFIG.get("harness") or {}
    cfg = harness.get("tool_search") or {}
    out: dict = {}
    if "preserve_thinking_iterations" in harness:
        out["preserve_thinking_iterations"] = int(
            harness["preserve_thinking_iterations"]
        )
    if "stream_chunk_timeout_seconds" in harness:
        out["stream_chunk_timeout_s"] = float(
            harness["stream_chunk_timeout_seconds"]
        )
    retry = harness.get("stream_retry") or {}
    if "max_attempts" in retry:
        out["stream_retry_max"] = max(0, int(retry["max_attempts"]))
    if "backoff_seconds" in retry:
        out["stream_retry_backoff_s"] = max(0.0, float(retry["backoff_seconds"]))
    if "tool_call_summaries" in harness:
        out["tool_call_summaries"] = bool(harness["tool_call_summaries"])
    if "enabled" in cfg:
        out["tool_search_enabled"] = bool(cfg["enabled"])
    if "threshold_tools" in cfg:
        out["tool_search_threshold_tools"] = int(cfg["threshold_tools"])
    if cfg.get("baseline_tools"):
        out["tool_search_baseline"] = list(cfg["baseline_tools"])
    if "max_results_default" in cfg:
        out["tool_search_max_results_default"] = int(cfg["max_results_default"])
    if "max_results_cap" in cfg:
        out["tool_search_max_results_cap"] = int(cfg["max_results_cap"])
    # Finalizer budgets. `final_schema` itself is never config — it is a
    # per-caller contract — so only the two knobs come from here.
    par = harness.get("parallel_tool_calls") or {}
    if "enabled" in par:
        out["parallel_tool_calls_enabled"] = bool(par["enabled"])
    if "max_concurrency" in par:
        out["parallel_tool_calls_max_concurrency"] = max(1, int(par["max_concurrency"]))
    out["parallel_safe_task_profiles"] = parallel_safe_task_profiles()
    fin = harness.get("finalizer") or {}
    if "max_tokens" in fin:
        out["finalizer_max_tokens"] = int(fin["max_tokens"])
    if "timeout_seconds" in fin:
        out["finalizer_timeout_s"] = float(fin["timeout_seconds"])
    echo = harness.get("echo_guard") or {}
    if echo.get("mode") in ("nudge", "tool_choice"):
        out["echo_guard_mode"] = str(echo["mode"])
    out.update(max_turns_wrapup_kwargs())
    out.update(intra_turn_compaction_kwargs())
    return out


def parallel_safe_task_profiles() -> frozenset[str]:
    """`subagents.<type>.parallel_safe: true`, as the set the loop checks (P8).

    Read from the same `subagents:` block `agent_mcp/builtin_task.py` loads
    its profiles from, so the parent that decides to overlap a batch and the
    child that holds itself to the read-only set agree on which profiles
    those are. Literally `true` only: a truthy string is a typo, not a vote.
    """
    profiles = CONFIG.get("subagents") or {}
    return frozenset(
        str(name) for name, prof in profiles.items()
        if isinstance(prof, dict) and prof.get("parallel_safe") is True
    )


def max_turns_wrapup_kwargs() -> dict:
    """`harness.max_turns_wrapup`, resolved to the engines it may run on (P6).

    `models` names the slots whose engine is verified to honour
    `tool_choice: "none"` with the tools array present. Resolved here to base
    URLs because the loop only knows the URL it was handed, and a Task child
    or a turn on another slot must get a per-call answer from the same list.
    A slot not listed — the llama.cpp secondary today — never wraps up.
    """
    cfg = ((CONFIG.get("harness") or {}).get("max_turns_wrapup") or {})
    if not cfg.get("enabled"):
        return {}
    models = CONFIG.get("models") or {}
    urls = tuple(
        str((models.get(alias) or {}).get("base_url") or "").rstrip("/")
        for alias in (cfg.get("models") or [])
    )
    urls = tuple(u for u in urls if u)
    return {"max_turns_wrapup": bool(urls), "max_turns_wrapup_base_urls": urls}


def intra_turn_compaction_kwargs() -> dict:
    """`compaction.microcompact`'s two fractions, for the in-turn pass.

    The turn-start pass (`app.compaction`) and the in-turn pass
    (`loop._intra_turn_microcompact`) share their threshold arithmetic so they
    cannot disagree about where the wall is — but only the first ever read the
    config. The second ran on `RunOptions` defaults no config reached, so
    lowering `trigger_fraction` would have moved the wall for the first
    request of a turn and not for the sixty after it, which is where a long
    worker round spends its context. A function of its own because
    `workers/sources/_common._worker_run_options` needs these and nothing
    else from here.
    """
    mc = ((CONFIG.get("compaction") or {}).get("microcompact") or {})
    out: dict = {}
    if "trigger_fraction" in mc:
        out["intra_turn_microcompact_trigger_fraction"] = float(mc["trigger_fraction"])
    if "target_fraction" in mc:
        out["intra_turn_microcompact_target_fraction"] = float(mc["target_fraction"])
    # Deny-list mode (D10): present — even empty — means every tool result
    # may be cleared except these; absent or null keeps the allow-list.
    if mc.get("non_compactable_tools") is not None:
        out["intra_turn_microcompact_non_compactable"] = tuple(
            str(x) for x in mc["non_compactable_tools"]
        )
    # #1514, off unless config turns it on (the turn-start pass in
    # `app.compaction` reads the same key).
    if "name_session_record" in mc:
        out["intra_turn_microcompact_name_session_record"] = bool(
            mc["name_session_record"])
    out.update(context_relief_kwargs())
    return out


def context_relief_kwargs() -> dict:
    """`harness.context_relief.*`, for the relief ladder.

    Rides in `intra_turn_compaction_kwargs` rather than only in
    `_get_harness_kwargs` for the reason that function's docstring gives:
    `workers/sources/_common._worker_run_options` builds its options from
    the narrower seam, and a worker round is exactly the turn that fills its
    window. Plumbing a context knob through one of the two seams would have
    it apply to chats and not to the rounds it was written for.
    """
    cr = ((CONFIG.get("harness") or {}).get("context_relief") or {})
    out: dict = {}
    if "enabled" in cr:
        out["context_relief_enabled"] = bool(cr["enabled"])
    if "terminal_floor_tokens" in cr:
        out["context_relief_terminal_floor_tokens"] = int(cr["terminal_floor_tokens"])
    if "min_completion_tokens" in cr:
        out["context_relief_min_completion_tokens"] = int(cr["min_completion_tokens"])
    if "reasoning_keep_under_pressure" in cr:
        out["context_relief_reasoning_keep_under_pressure"] = int(
            cr["reasoning_keep_under_pressure"]
        )
    if "shrink_arguments" in cr:
        out["context_relief_shrink_arguments"] = bool(cr["shrink_arguments"])
    if "shrink_arguments_min_chars" in cr:
        out["context_relief_shrink_arguments_min_chars"] = int(
            cr["shrink_arguments_min_chars"]
        )
    if cr.get("shrink_arguments_tools"):
        out["context_relief_shrink_arguments_tools"] = tuple(
            str(x) for x in cr["shrink_arguments_tools"]
        )
    if "max_overflow_recoveries" in cr:
        out["max_context_overflow_recoveries"] = max(
            0, int(cr["max_overflow_recoveries"])
        )
    return out


async def _discover_mcp_tools(server_name: str, cfg: dict) -> tuple[list[dict], str | None]:
    """Discover tools from an MCP server. Supports SSE/HTTP and stdio.

    Both transports go through the SDK client. The stdio path used to be a
    hand-rolled JSON-RPC exchange — write a framed initialize, read one
    line, write tools/list, read one line — which pinned the protocol
    version at 2024-11-05, never drained stderr, and matched no request
    ids. It was the same shape of code as the Thunderbird bridge client,
    with the same defects; there is no reason to keep a second copy.

    Returns (tools, error). Never raises: a server that is down should
    render as an empty, explained row in the Tools page, not a 500.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client

    server_type = cfg.get("type", "stdio")

    async def _query() -> list[dict]:
        # Dispatch off the same HTTP_TRANSPORTS list `_get_mcp_servers`
        # validates against. This branch used to carry its own hardcoded
        # ("sse", "http"), so `streamable-http` fell through to stdio and
        # tried to spawn a bare `python` — the Tools page then sat on
        # "Discovering tools..." until the timeout, with nothing logged.
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            if server_type in HTTP_TRANSPORTS:
                url = cfg.get("url", "")
                # The aggregator refuses a request that carries no credential
                # (#1053), and this is a fifth caller of it that #1053 did not
                # list: it names the server by its `mcp_servers:` URL, so a
                # grep for the port or for `services.lloyd_mcp` never finds
                # it. The day the guard landed the Tools page read
                # "Server returned an error response" — the SDK's rendering of
                # a 401 — and `test_discovery_resolves_the_configured_transport`
                # went red on main for every round's tests rung, which the
                # change's own gate could not see because that test talks to
                # the LIVE aggregator and the live one was still unguarded.
                # The pool's helpers, not a second copy: loopback only, read
                # per call, and a client we hand in is one we close.
                from app.harness.mcp_pool import _http_client, aggregator_headers
                headers = aggregator_headers(url)
                if server_type == "sse":
                    ctx = sse_client(url, headers=headers or None)
                else:
                    client = _http_client(headers=headers)
                    if client is not None:
                        await stack.enter_async_context(client)
                    ctx = streamable_http_client(url, http_client=client)
            elif server_type in STDIO_TRANSPORTS:
                ctx = stdio_client(StdioServerParameters(
                    command=cfg.get("command", "python"),
                    args=list(cfg.get("args") or []),
                    env=dict(cfg["env"]) if cfg.get("env") else None,
                    cwd=cfg.get("cwd") or None,
                ))
            else:
                raise ValueError(f"unknown transport type {server_type!r}")
            read_stream, write_stream = await stack.enter_async_context(ctx)
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            result = await session.list_tools()
            return [
                {"name": t.name, "description": t.description or ""}
                for t in result.tools
            ]

    try:
        # Bound the whole exchange rather than the read: ClientSession only
        # accepts a per-request timeout on call_tool, and a server that
        # accepts the connection then never speaks would hang discovery —
        # which the Tools page blocks on.
        return await asyncio.wait_for(_query(), timeout=DISCOVERY_TIMEOUT_SECONDS), None
    except asyncio.TimeoutError:
        return [], f"Timeout querying {server_name}"
    except Exception as exc:
        return [], root_cause(exc)
