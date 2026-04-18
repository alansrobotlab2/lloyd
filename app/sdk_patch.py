"""Monkeypatch for claude_agent_sdk 0.0.25.

The SDK's `parse_message` raises `MessageParseError` on new Anthropic API
message types (e.g. `rate_limit_event`), which kills the stream mid-turn.
We wrap it to fall back to a bare StreamEvent instead. The patch must be
applied to `client.parse_message` — that module captured the reference
by-name at import time, so patching the source module alone is not enough.

Import this module once at process start to apply the patch.
"""

from claude_agent_sdk.types import StreamEvent
from claude_agent_sdk._internal import client as _sdk_client
from claude_agent_sdk._internal.message_parser import parse_message as _sdk_parse_original


def _sdk_parse_patched(data):
    try:
        return _sdk_parse_original(data)
    except Exception:
        return StreamEvent(
            uuid=data.get("uuid", ""),
            session_id=data.get("session_id", ""),
            event=data,
        )


_sdk_client.parse_message = _sdk_parse_patched
