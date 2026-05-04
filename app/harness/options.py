"""Replacement for `claude_agent_sdk.ClaudeAgentOptions`.

`RunOptions` is a dataclass that mirrors the SDK's options surface minus
SDK-only kwargs (resume, stderr, setting_sources, agents, skills,
plugins, cli_path, fork_session, etc.). Adds `history` for OpenAI-style
message lists (the harness builds these from `compaction.load_and_compact_session`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.harness.hooks import HookRegistry


@dataclass
class RunOptions:
    # Model + endpoint
    model: str
    base_url: str = "http://127.0.0.1:8096"

    # Prompting
    system_prompt: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    # Loop control
    max_turns: int = 60
    permission_mode: str = "bypassPermissions"
    disallowed_tools: list[str] = field(default_factory=list)

    # MCP / tools
    mcp_servers: dict[str, dict] = field(default_factory=dict)

    # Request-level
    env: dict[str, str] = field(default_factory=dict)
    request_timeout_s: float = 600.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    api_key: str = "no-key-required"

    # Hooks
    hooks: "HookRegistry | None" = None

    # Shared chat messages buffer. When supplied, the harness uses this
    # list directly instead of copying `history` into a private buffer.
    # Lets the Inner Voice observer mutate (e.g. inject a system message)
    # between iterations, with the harness picking it up on the next loop pass.
    chat_messages_handle: list[dict[str, Any]] | None = None

    # Cancellation — the consumer (messages._run_turn) sets this to
    # interrupt mid-stream. The loop checks between SSE chunks AND
    # before each tool dispatch.
    cancel_event: asyncio.Event | None = None

    # Session correlation — closure-bound into hook callbacks so
    # heuristics/intra_turn write to the right event log file.
    session_id: str = ""

    # Forwarded to vLLM unchanged. Most local deployments ignore these,
    # but we keep them so config-driven thinking knobs can survive into
    # extra_body if ever needed.
    effort: str | None = None
    thinking: dict | None = None

    # Tool search / progressive disclosure. When activated, the harness
    # advertises a small baseline + ToolSearch instead of the full catalog,
    # and the model loads tool schemas on demand. See app/harness/tool_search.py.
    tool_search_enabled: bool = True
    tool_search_threshold_tools: int = 30
    tool_search_baseline: list[str] = field(default_factory=list)
    tool_search_max_results_default: int = 5
    tool_search_max_results_cap: int = 20
