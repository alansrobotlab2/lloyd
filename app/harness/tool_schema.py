"""MCP tool definitions ↔ OpenAI tool definitions.

vLLM's `/v1/chat/completions` expects the OpenAI tools schema:

    [{"type": "function",
      "function": {"name": "Bash",
                   "description": "...",
                   "parameters": {<JSON Schema>}}}]

MCP `tools/list` returns:

    [{"name": "Bash", "description": "...", "inputSchema": {<JSON Schema>}}]

Tools are advertised under their bare MCP name. The historical
`mcp__<server>__<tool>` prefix was a Claude Agent SDK artifact — with the
in-process harness we control naming, and skill docs / SOUL.md / persisted
sessions all reference tools by bare name. ``build_tool_list`` raises
``ToolDiscoveryError`` if two MCP servers ever export the same bare name;
today only ``lloyd-mcp`` exists, but we want fail-loud rather than silent
shadowing.

OpenAI's spec caps tool names at 64 chars. ``mcp_tool_to_openai`` refuses a
longer one; ``build_tool_list`` leaves that tool out of the catalog and logs
a warning naming it, so the rest of the catalog still ships.
"""

from __future__ import annotations

import logging
from typing import Any

from app.harness.errors import ToolDiscoveryError

logger = logging.getLogger(__name__)

OPENAI_TOOL_NAME_MAX = 64

# ---------------------------------------------------------------------------
# The `summary` display parameter
# ---------------------------------------------------------------------------

# Every advertised tool carries one extra string parameter the model fills
# in with a short phrase describing what the call is doing. It is display
# metadata only: `_commit_tool_calls` lifts it off the parsed arguments and
# onto the `tool_call` event, and it never reaches the MCP server. Nothing
# validates `arguments` against a tool's inputSchema — the pool coerces
# top-level primitives only (`mcp_pool._coerce_args`) and unknown keys reach
# the handler — so keeping it out is the harness's job, not the server's:
# a leaked caption would land in the handler's arguments unannounced.
SUMMARY_ARG = "summary"

SUMMARY_PROPERTY: dict[str, Any] = {
    "type": "string",
    "description": (
        "Short present-tense summary of what this call is doing, for the "
        "human watching the transcript — e.g. \"Reading server.py\", "
        "\"Restarting the backend\", \"Searching the vault for voice notes\". "
        "Under ~8 words, no trailing period, and don't repeat the tool name. "
        "Display only: stripped before the tool runs."
    ),
}


def add_summary_param(tools: list[dict[str, Any]]) -> set[str]:
    """Add the `summary` display parameter to each advertised tool.

    Returns the set of tool names that actually received it, which is what
    the loop uses to decide whether a `summary` key in a model's arguments
    is ours to strip. Two tools must never be treated the same way here:

      * A tool that already declares its own top-level ``summary`` —
        ``session_inject_context`` does, and it is *required* — keeps it untouched
        and is left out of the returned set. Popping that value before
        dispatch would silently delete a real argument.
      * Everything else gets the injected copy and is stripped on the way
        out.

    The `parameters` object is replaced rather than mutated. It arrives as
    the very ``inputSchema`` dict held in ``MCPPool.discovered``, which is
    process-shared and reused for the life of the pool: mutating it in
    place would leave `summary` in the pool's own copy, so the *second*
    call would read it as the tool's own parameter, skip injection, and
    stop stripping it — handing the aggregator an argument no tool
    declares.
    """
    injected: set[str] = set()
    for tool in tools:
        fn = tool.get("function") or {}
        name = fn.get("name") or ""
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        props = params.get("properties") or {}
        if SUMMARY_ARG in props:
            continue
        # First, not last, in both lists. Property order is the order the
        # schema is rendered to the model and the order it tends to emit
        # arguments in, so a caption placed after Bash's `command` is a
        # caption written after a 40-line heredoc. Cheap to state up front,
        # easy to trail off at the end.
        new_props = {SUMMARY_ARG: dict(SUMMARY_PROPERTY), **props}
        new_params = dict(params)
        new_params["properties"] = new_props
        required = list(new_params.get("required") or [])
        if SUMMARY_ARG not in required:
            required.insert(0, SUMMARY_ARG)
        new_params["required"] = required
        fn["parameters"] = new_params
        injected.add(name)
    return injected


def pop_summary(args: dict[str, Any]) -> str:
    """Remove and return the injected `summary` argument, if the model sent one.

    Non-string values are dropped rather than coerced: a model that emits
    ``summary: {...}`` has misunderstood the field, and rendering
    ``[object Object]`` beside a tool name is worse than rendering nothing.
    """
    value = args.pop(SUMMARY_ARG, None)
    if not isinstance(value, str):
        return ""
    return value.strip()


def mcp_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate one MCP tool definition into the OpenAI tools schema."""
    name = tool["name"]
    if len(name) > OPENAI_TOOL_NAME_MAX:
        raise ValueError(
            f"tool name '{name}' exceeds OpenAI's {OPENAI_TOOL_NAME_MAX}-char limit"
        )
    parameters = tool.get("inputSchema") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description") or "",
            "parameters": parameters,
        },
    }


def resolve_tool_name(advertised_name: str) -> tuple[str | None, str]:
    """Map a name the model emitted back to (server_name | None, bare_name).

    The advertised form is bare. The legacy ``mcp__server__tool`` form is
    still parsed so historical session JSON replays cleanly. Returns
    ``(server_name | None, bare_name)``; ``server_name`` is informational
    only — dispatch always goes through the single aggregator pool.
    """
    if advertised_name.startswith("mcp__"):
        rest = advertised_name[len("mcp__"):]
        sep = rest.find("__")
        if sep > 0:
            return rest[:sep], rest[sep + 2:]
    return None, advertised_name


def build_tool_list(
    discovered: list[tuple[str, list[dict[str, Any]]]],
    disallowed: set[str],
) -> list[dict[str, Any]]:
    """Build the OpenAI ``tools=[...]`` payload from MCP discovery output.

    ``discovered`` is a list of ``(server_name, tools_list)`` pairs.
    ``disallowed`` is the set of bare tool names to skip; the legacy
    ``mcp__server__tool`` form is also accepted so old config can be
    rolled forward.

    Raises ``ToolDiscoveryError`` if two servers export the same bare tool
    name — bare-name advertise gives us no way to disambiguate, and we want
    a loud error rather than silent shadowing. It is a discovery failure,
    named as one, so `run_query`'s caller sees the same error class as for
    an empty catalog (D13). A tool whose name is over the 64-char limit is
    left out with a warning, not silently.
    """
    tools: list[dict[str, Any]] = []
    seen: dict[str, str] = {}  # bare_name -> server_name
    for server_name, mcp_tools in discovered:
        for mcp_tool in mcp_tools:
            bare = mcp_tool["name"]
            # Internal-only tools — harness can dispatch them via direct
            # pool.call_tool, but we never advertise them to the model.
            # Background-task drain is the first user (#async-bash).
            if bare.startswith("_"):
                continue
            if bare in disallowed:
                continue
            if f"mcp__{server_name}__{bare}" in disallowed:
                continue
            if bare in seen and seen[bare] != server_name:
                raise ToolDiscoveryError(
                    f"tool name collision: {bare!r} exported by both "
                    f"{seen[bare]!r} and {server_name!r}",
                    servers=[seen[bare], server_name],
                )
            try:
                openai_tool = mcp_tool_to_openai(mcp_tool)
            except ValueError as exc:
                logger.warning(
                    "tool_schema: not advertising %r from %r: %s",
                    bare, server_name, exc,
                )
                continue
            seen[bare] = server_name
            tools.append(openai_tool)
    return tools
