#!/usr/bin/env python3
"""
Test 1: Basic query() with local Qwen model via ANTHROPIC_BASE_URL.

Tests whether the Claude Agent SDK can route to our local 122B model
at 127.0.0.1:8096 using the OpenAI-compatible endpoint.
"""

import asyncio
import os

from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, ResultMessage, SystemMessage

# Route SDK to local model via Anthropic-compatible endpoint
# Note: vLLM serves both OpenAI and Anthropic API formats
# ANTHROPIC_BASE_URL should NOT include /v1 — the SDK appends /v1/messages
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:8096"
os.environ["ANTHROPIC_API_KEY"] = "no-key-required"
os.environ["ANTHROPIC_CUSTOM_MODEL_OPTION"] = "Qwen3.5-122B-A10B"
os.environ["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] = "Qwen 122B Local"


async def main():
    print("=" * 60)
    print("Test 1: Basic query() with local Qwen 122B")
    print("=" * 60)
    print()

    options = ClaudeCodeOptions(
        model="Qwen3.5-122B-A10B",
        system_prompt="You are a helpful assistant. Respond concisely.",
        max_turns=1,
        permission_mode="bypassPermissions",
    )

    print(f"Model: {options.model}")
    print(f"Endpoint: {os.environ['ANTHROPIC_BASE_URL']}")
    print()
    print("Prompt: What is 2+2? Answer in one sentence.")
    print("-" * 40)

    session_id = None
    try:
        async for message in query(
            prompt="What is 2+2? Answer in one sentence.",
            options=options,
        ):
            if isinstance(message, SystemMessage):
                session_id = getattr(message, "session_id", None)
                print(f"[system] session_id={session_id}, subtype={message.subtype}")

            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text"):
                        print(f"[assistant] {block.text}")
                    elif hasattr(block, "name"):
                        print(f"[tool_use] {block.name}({block.input})")

            elif isinstance(message, ResultMessage):
                print(f"\n[result] subtype={message.subtype}")
                if hasattr(message, "cost"):
                    print(f"[result] cost={message.cost}")
                if hasattr(message, "usage"):
                    print(f"[result] usage={message.usage}")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print(f"Session ID: {session_id}")
    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
