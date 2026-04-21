#!/usr/bin/env python3
"""
Test 3: Custom MCP server with Claude Agent SDK.

Tests whether the SDK can connect to a simple custom MCP server
and the local model can call its tools. This validates the
plugin-as-MCP-server migration path.
"""

import asyncio
import os
import tempfile

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage, SystemMessage

os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:8096"
os.environ["ANTHROPIC_API_KEY"] = "no-key-required"
os.environ["ANTHROPIC_CUSTOM_MODEL_OPTION"] = "primary"
os.environ["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] = "Primary Local"

# Create a minimal test MCP server inline
MCP_SERVER_CODE = '''
import asyncio
import json
import sys
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("test-tools")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_current_time",
            description="Returns the current date and time",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="calculate",
            description="Evaluates a simple math expression",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate, e.g. '2+2'"}
                },
                "required": ["expression"],
            },
        ),
    ]

@app.call_tool()
async def call_tool(name, arguments):
    if name == "get_current_time":
        return [TextContent(type="text", text=f"Current time: {datetime.now().isoformat()}")]
    elif name == "calculate":
        expr = arguments.get("expression", "0")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return [TextContent(type="text", text=f"Result: {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream)

asyncio.run(main())
'''


async def main():
    print("=" * 60)
    print("Test 3: Custom MCP server with local primary")
    print("=" * 60)
    print()

    # Write temp MCP server file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="test_mcp_") as f:
        f.write(MCP_SERVER_CODE)
        mcp_server_path = f.name

    print(f"MCP server: {mcp_server_path}")

    venv_python = os.path.expanduser("~/agent-services/.venvs/claude-agent-sdk/bin/python")

    options = ClaudeAgentOptions(
        model="primary",
        system_prompt="You are a helpful assistant. Use the MCP tools (get_current_time, calculate) to answer questions. Do NOT use Bash.",
        allowed_tools=["mcp__test-tools__get_current_time", "mcp__test-tools__calculate"],
        max_turns=3,
        permission_mode="bypassPermissions",
        mcp_servers={
            "test-tools": {
                "type": "stdio",
                "command": venv_python,
                "args": [mcp_server_path],
            }
        },
    )

    prompt = "What time is it right now? Also, what is 137 * 42? Use your get_current_time and calculate tools."

    print(f"Prompt: {prompt}")
    print("-" * 40)

    tool_calls = 0
    mcp_tool_calls = 0
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, SystemMessage):
                print(f"[system] subtype={message.subtype}")

            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text.strip():
                        print(f"[assistant] {block.text[:300]}")
                    elif hasattr(block, "name"):
                        tool_calls += 1
                        name = block.name
                        if "get_current_time" in name or "calculate" in name:
                            mcp_tool_calls += 1
                        print(f"[tool_use] {name}({str(block.input)[:100]})")

            elif isinstance(message, ResultMessage):
                print(f"\n[result] subtype={message.subtype}")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    print()
    print(f"Total tool calls: {tool_calls}")
    print(f"MCP tool calls: {mcp_tool_calls}")
    print(f"MCP integration {'WORKS' if mcp_tool_calls > 0 else 'FAILED — model did not call MCP tools'}")
    print("=" * 60)

    # Cleanup
    os.unlink(mcp_server_path)


if __name__ == "__main__":
    asyncio.run(main())
