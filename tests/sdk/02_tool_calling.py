#!/usr/bin/env python3
"""
Test 2: Tool calling with local Qwen model.

Tests whether the SDK's built-in tools (Read, Bash) work when
routed through the local primary model. This is the critical test —
if the local model can handle Claude Code's tool schemas, the
migration path opens up significantly.
"""

import asyncio
import os

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage, SystemMessage

os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:8096"
os.environ["ANTHROPIC_API_KEY"] = "no-key-required"
os.environ["ANTHROPIC_CUSTOM_MODEL_OPTION"] = "primary"
os.environ["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] = "Primary Local"


async def main():
    print("=" * 60)
    print("Test 2: Tool calling with local primary")
    print("=" * 60)
    print()

    options = ClaudeAgentOptions(
        model="primary",
        system_prompt="You are a helpful assistant. Use tools when asked to perform tasks.",
        allowed_tools=["Bash", "Read"],
        max_turns=3,
        permission_mode="bypassPermissions",
        cwd=os.path.expanduser("~/.hermes"),
    )

    prompt = "Read the file config.yaml and tell me what the default model is set to."

    print(f"Model: {options.model}")
    print(f"Tools: {options.allowed_tools}")
    print(f"Prompt: {prompt}")
    print("-" * 40)

    tool_calls = 0
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, SystemMessage):
                print(f"[system] subtype={message.subtype}")

            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text.strip():
                        print(f"[assistant] {block.text[:200]}")
                    elif hasattr(block, "name"):
                        tool_calls += 1
                        args_preview = str(block.input)[:100]
                        print(f"[tool_use] {block.name}({args_preview})")

            elif isinstance(message, ResultMessage):
                print(f"\n[result] subtype={message.subtype}")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    print()
    print(f"Tool calls made: {tool_calls}")
    print(f"Tool calling {'WORKS' if tool_calls > 0 else 'FAILED — model did not invoke tools'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
