"""MCP tool definitions ↔ OpenAI tool definitions.

vLLM's `/v1/chat/completions` expects the OpenAI tools schema:

    [{"type": "function",
      "function": {"name": "Bash",
                   "description": "...",
                   "parameters": {<JSON Schema>}}}]

MCP `tools/list` returns:

    [{"name": "Bash", "description": "...", "inputSchema": {<JSON Schema>}}]

Two design notes:

1. Bare-name aliasing. The SDK previously exposed Bash/Read/etc. as
   built-in tool names without any prefix. Persisted session JSON,
   SOUL.md, and `inner_voice.pretooluse_deny` rules all reference these
   bare names. We advertise the same bare name to vLLM (the
   `mcp__lloyd-mcp__` prefix that `_get_disallowed_tools` uses for
   namespacing is stripped at the model boundary). The dispatch side
   accepts either form.

2. OpenAI's spec caps tool names at 64 chars. The longest current name
   in Lloyd is well under, but this is validated at translation time so
   a future MCP server with a long tool name fails loudly instead of
   silently.
"""

from __future__ import annotations

from typing import Any

OPENAI_TOOL_NAME_MAX = 64

# Tools that should be re-exported under their bare name to the model.
# These are the "built-ins" we recreated as MCP tools — they need to
# match the names used in persisted sessions and SOUL.md prompts.
BUILTIN_BARE_NAMES = {
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Grep",
    "Glob",
    "Task",
}


def mcp_tool_to_openai(server_name: str, tool: dict[str, Any]) -> dict[str, Any]:
    """Translate one MCP tool definition into the OpenAI tools schema.

    `server_name` is the MCP server's identifier (e.g. "lloyd-mcp").
    Names are flattened to bare for built-in tools, otherwise namespaced
    as `mcp__<server>__<tool>`.
    """
    name = tool["name"]
    if name not in BUILTIN_BARE_NAMES:
        name = f"mcp__{server_name}__{name}"
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
    """Map a name the model emitted back to (server_name, mcp_tool_name).

    Bare built-in names (e.g. "Bash") return (None, "Bash"); the
    dispatcher then knows to look it up against the lloyd-mcp aggregator.
    Namespaced names (`mcp__server__tool`) are split apart.

    Returns (server_name | None, bare_tool_name). The server_name is
    informational — the dispatcher always routes through the single
    aggregator MCP pool.
    """
    if advertised_name.startswith("mcp__"):
        # mcp__server__tool — split on the first two underscores after
        # the prefix. Server names may contain underscores so we use
        # the first valid split.
        rest = advertised_name[len("mcp__"):]
        sep = rest.find("__")
        if sep > 0:
            server = rest[:sep]
            tool = rest[sep + 2:]
            return server, tool
        # Malformed — fall through to bare interpretation.
    return None, advertised_name


def build_tool_list(
    discovered: list[tuple[str, list[dict[str, Any]]]],
    disallowed: set[str],
) -> list[dict[str, Any]]:
    """Build the OpenAI `tools=[...]` payload from MCP discovery output.

    `discovered` is a list of `(server_name, tools_list)` pairs as
    returned by `app.mcp_discovery._discover_mcp_tools`. `disallowed` is
    the set of tool names (in either bare or namespaced form) to skip.

    Returns the OpenAI-shaped tools list, ready to send as the request's
    `tools` field.
    """
    tools: list[dict[str, Any]] = []
    for server_name, mcp_tools in discovered:
        for mcp_tool in mcp_tools:
            try:
                openai_tool = mcp_tool_to_openai(server_name, mcp_tool)
            except ValueError:
                continue
            advertised = openai_tool["function"]["name"]
            bare = mcp_tool["name"]
            if advertised in disallowed or bare in disallowed:
                continue
            namespaced = f"mcp__{server_name}__{bare}"
            if namespaced in disallowed:
                continue
            tools.append(openai_tool)
    return tools
