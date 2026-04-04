#!/usr/bin/env python3
"""
Test 4: Streaming event inspection.

Dumps all message types and content blocks from the SDK to understand
the exact event stream we'd need to bridge to mc_server.py's SSE format.
"""

import asyncio
import json
import os

from claude_code_sdk import query, ClaudeCodeOptions

os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:8096"
os.environ["ANTHROPIC_API_KEY"] = "no-key-required"
os.environ["ANTHROPIC_CUSTOM_MODEL_OPTION"] = "Qwen3.5-122B-A10B"
os.environ["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] = "Qwen 122B Local"


async def main():
    print("=" * 60)
    print("Test 4: Streaming event inspection")
    print("=" * 60)
    print()

    options = ClaudeCodeOptions(
        model="Qwen3.5-122B-A10B",
        system_prompt="You are a helpful assistant.",
        allowed_tools=["Bash"],
        max_turns=2,
        permission_mode="bypassPermissions",
        cwd=os.path.expanduser("~"),
    )

    prompt = "Run 'echo hello world' in bash and tell me the output."
    print(f"Prompt: {prompt}")
    print("-" * 40)

    msg_count = 0
    try:
        async for message in query(prompt=prompt, options=options):
            msg_count += 1
            msg_type = type(message).__name__

            print(f"\n--- Message #{msg_count}: {msg_type} ---")
            print(f"  type attr: {getattr(message, 'type', 'N/A')}")
            print(f"  subtype: {getattr(message, 'subtype', 'N/A')}")

            if hasattr(message, "session_id"):
                print(f"  session_id: {message.session_id}")

            if hasattr(message, "content"):
                content = message.content
                if isinstance(content, list):
                    for i, block in enumerate(content):
                        block_type = type(block).__name__
                        print(f"  content[{i}]: {block_type}")
                        if hasattr(block, "text"):
                            print(f"    text: {block.text[:200]}")
                        if hasattr(block, "name"):
                            print(f"    name: {block.name}")
                            print(f"    input: {str(block.input)[:200]}")
                        if hasattr(block, "id"):
                            print(f"    id: {block.id}")
                        if hasattr(block, "tool_use_id"):
                            print(f"    tool_use_id: {block.tool_use_id}")
                        if hasattr(block, "content"):
                            print(f"    content: {str(block.content)[:200]}")
                elif isinstance(content, str):
                    print(f"  content: {content[:200]}")

            # Dump any extra attributes
            for attr in ["cost", "usage", "duration_ms", "duration_api_ms",
                         "is_error", "num_turns", "result"]:
                if hasattr(message, attr):
                    print(f"  {attr}: {getattr(message, attr)}")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    print()
    print(f"Total messages: {msg_count}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
