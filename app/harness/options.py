"""Replacement for `claude_agent_sdk.ClaudeAgentOptions`.

`RunOptions` is a dataclass that mirrors the SDK's options surface minus
SDK-only kwargs (resume, stderr, setting_sources, agents, skills,
plugins, cli_path, fork_session, etc.). Adds `history` for OpenAI-style
message lists (the harness builds these from `compaction.load_and_compact_session`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

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

    # Optional per-iteration refresher for the disallowed tools list.
    # When set, the harness calls this at the top of each loop iteration
    # to get the live disallowed list — used both to filter advertised
    # tools (visible_tools) and to gate dispatch. The catalog itself is
    # built once at turn start using `disallowed_tools` (the static base);
    # the refresher only ever ADDS or REMOVES from that catalog at runtime,
    # not changes the universe of tools the harness can see.
    #
    # Plan B uses this so ExitPlanMode flipping plan_mode=false within a
    # turn unblocks Write/Edit/Bash on the next iteration of the same
    # turn, instead of waiting for a fresh user turn to rebuild options.
    disallowed_tools_refresh: Callable[[], list[str]] | None = None

    # MCP / tools
    mcp_servers: dict[str, dict] = field(default_factory=dict)

    # Request-level
    env: dict[str, str] = field(default_factory=dict)
    request_timeout_s: float = 600.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    api_key: str = "no-key-required"

    # vLLM scheduling priority (lower = higher priority). Requires vLLM
    # launched with `--scheduling-policy priority`. Convention: 0 for
    # interactive (chat, inner voice), 1 for background (workers, autonomy).
    priority: int | None = None

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

    # Background-task notification drain. When set, the loop calls this
    # at the top of each iteration; the callable returns a list of
    # OpenAI-format messages (typically role: "user" with a
    # <task_notification> XML body) to splice into chat_messages before
    # the next vLLM request. The callback is also expected to persist
    # the messages into the session JSON, since the harness no longer
    # owns persistence past run_query's entry.
    notification_drain: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None

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

    # Mid-turn microcompaction. When tool results pile up within a single
    # turn (the temporal-knowledge stall hit 30+ in one turn), the
    # harness clears stale ones in-place via app.harness.microcompact so
    # the primary's next iteration sees a manageable context. Spill-aware:
    # any older result with a `<persisted-output>` marker also gets cleared
    # since its content is already on disk and re-readable via Read.
    intra_turn_microcompact_enabled: bool = True
    intra_turn_microcompact_threshold: int = 15  # tool results before considering
    intra_turn_microcompact_keep_recent: int = 15  # keep this many most-recent inline
    # Budget gate, added 2026-09-05. The tool-count threshold above is now
    # only a cheap pre-check: clearing happens solely when the prompt is
    # actually pressing on the context window. Session 20260905_024955_iv5f05
    # ran 70 tool calls in one turn at a peak of 106,802 tokens against a
    # 210,144 threshold, and this pass held it to 5 inline tool results the
    # whole way, because it counted tools and never looked at tokens.
    intra_turn_microcompact_trigger_fraction: float = 0.8
    intra_turn_microcompact_target_fraction: float = 0.6
    intra_turn_microcompact_min_chars: int = 2_000
