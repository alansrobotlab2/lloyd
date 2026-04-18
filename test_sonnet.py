"""Quick smoke tests for claude-agent-sdk against Sonnet."""
import asyncio
import os
import sys

# Clear local-model env vars so we hit the real Anthropic API
for var in ["ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY",
            "ANTHROPIC_CUSTOM_MODEL_OPTION", "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"]:
    os.environ.pop(var, None)

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage
from claude_agent_sdk.types import TextBlock, StreamEvent
from claude_agent_sdk._internal import client as _sdk_client
from claude_agent_sdk._internal.message_parser import parse_message as _orig_parse

# Patch: SDK 0.0.25 doesn't handle rate_limit_event — patch it to a no-op.
# Must patch the reference in client.py (which imported by-name at load time).
def _patched_parse(data):
    try:
        return _orig_parse(data)
    except Exception:
        return StreamEvent(uuid=data.get("uuid",""), session_id=data.get("session_id",""), event=data)
_sdk_client.parse_message = _patched_parse

MODEL = "claude-sonnet-4-6"
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


async def run(label, prompt, *, check):
    print(f"  {label} ... ", end="", flush=True)
    try:
        text = ""
        result = None
        async for msg in query(prompt=prompt, options=ClaudeAgentOptions(model=MODEL, max_turns=3)):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text += block.text
            elif isinstance(msg, ResultMessage):
                result = msg
        ok = check(text, result)
        print(PASS if ok else FAIL, f"  [{text[:60].strip()!r}]")
        return ok
    except Exception as e:
        print(FAIL, f"  exception: {e}")
        return False


async def main():
    print(f"\nSonnet SDK smoke tests ({MODEL})\n")
    results = []

    results.append(await run(
        "1. basic reply",
        "Reply with exactly the word: pong",
        check=lambda t, r: "pong" in t.lower(),
    ))

    results.append(await run(
        "2. arithmetic",
        "What is 17 * 23? Reply with only the number.",
        check=lambda t, r: "391" in t,
    ))

    results.append(await run(
        "3. result message present",
        "Say hello.",
        check=lambda t, r: r is not None and not r.is_error,
    ))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed", "✓" if passed == total else "✗")
    sys.exit(0 if passed == total else 1)


asyncio.run(main())
