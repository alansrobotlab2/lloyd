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
sessions all reference tools by bare name. ``build_tool_list`` raises if
two MCP servers ever export the same bare name; today only ``lloyd-mcp``
exists, but we want fail-loud rather than silent shadowing.

OpenAI's spec caps tool names at 64 chars. Validated at translation time
so a future MCP server with a long name fails loudly instead of silently.
"""

from __future__ import annotations

from typing import Any

OPENAI_TOOL_NAME_MAX = 64


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

    Raises ``ValueError`` if two servers export the same bare tool name —
    bare-name advertise gives us no way to disambiguate, and we want a
    loud error rather than silent shadowing.
    """
    tools: list[dict[str, Any]] = []
    seen: dict[str, str] = {}  # bare_name -> server_name
    for server_name, mcp_tools in discovered:
        for mcp_tool in mcp_tools:
            bare = mcp_tool["name"]
            if bare in disallowed:
                continue
            if f"mcp__{server_name}__{bare}" in disallowed:
                continue
            if bare in seen and seen[bare] != server_name:
                raise ValueError(
                    f"tool name collision: {bare!r} exported by both "
                    f"{seen[bare]!r} and {server_name!r}"
                )
            try:
                openai_tool = mcp_tool_to_openai(mcp_tool)
            except ValueError:
                continue
            seen[bare] = server_name
            tools.append(openai_tool)
    return tools
