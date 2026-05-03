"""Local-only agent harness.

Replaces `claude_agent_sdk.query()` with an in-process loop that streams
against vLLM (`/v1/chat/completions`), accumulates structured tool_calls,
dispatches them through the existing `agent_mcp/` MCP aggregator, and
yields a normalized event stream that downstream code consumes the same
way it consumed SDK message objects.

Public surface:
    run_query(messages, options) -> AsyncIterator[NormalizedEvent]
    RunOptions, HookRegistry, NormalizedEvent
    ParseError, ToolDispatchError, MaxTurnsExceeded
"""

from app.harness.errors import (
    HarnessError,
    ParseError,
    ToolDispatchError,
    MaxTurnsExceeded,
)
from app.harness.events import NormalizedEvent
from app.harness.hooks import HookRegistry, HookCallback
from app.harness.loop import run_query
from app.harness.options import RunOptions

__all__ = [
    "HarnessError",
    "HookCallback",
    "HookRegistry",
    "MaxTurnsExceeded",
    "NormalizedEvent",
    "ParseError",
    "RunOptions",
    "ToolDispatchError",
    "run_query",
]
