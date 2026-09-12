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

    # Inter-chunk stall deadline, in seconds; 0 disables. Bounds the gap
    # BETWEEN SSE lines once the engine has started producing — not
    # time-to-first-line, which is `request_timeout_s`'s job (prefill emits
    # no bytes, and the llama.cpp secondary serialises behind `--parallel 1`,
    # so pre-first-line silence is normal). `client.stream_chat` sets httpx
    # `read=None`, so without this a wedged engine mid-generation hangs the
    # turn until the client gives up. Fed from
    # `harness.stream_chunk_timeout_seconds`.
    stream_chunk_timeout_s: float = 0.0

    # Hooks
    hooks: "HookRegistry | None" = None

    # Shared chat messages buffer. When supplied, the harness uses this
    # list directly instead of copying `history` into a private buffer.
    # Lets the Inner Voice observer mutate (e.g. inject a system message)
    # between iterations, with the harness picking it up on the next loop pass.
    chat_messages_handle: list[dict[str, Any]] | None = None

    # The exact `tools` array the loop last advertised, written into this
    # caller-owned list when one is supplied (#529).
    #
    # `finalizer.py` already knows the rule — send the identical tools array
    # with `tool_choice: "none"`, or Qwen's template re-renders the prompt from
    # token zero and vLLM re-prefills the whole conversation. What it never had
    # was a way for a caller OUTSIDE the loop to honour it: the array is built
    # per iteration from the MCP pool plus the live disallowed set, so nothing
    # on the outside can reconstruct it. `app/harness/run_state.py` asks for a
    # state patch after a segment ends and retries a rejected patch once, and a
    # re-ask that guesses the array re-prefills the very prompt the retry
    # exists to protect. Written by slice assignment so the caller's list
    # object stays the one it handed over. Ignored when unset, which is every
    # other caller, including the interactive loop.
    visible_tools_capture: list[dict[str, Any]] | None = None

    # Cancellation — the consumer (messages._run_turn) sets this to
    # interrupt mid-stream. The loop checks between SSE chunks AND
    # before each tool dispatch.
    cancel_event: asyncio.Event | None = None

    # Session correlation — closure-bound into hook callbacks so
    # heuristics/intra_turn write to the right event log file.
    session_id: str = ""

    # The chat turn this run belongs to. Travels to the aggregator in the
    # request `_meta` so the change ledger can record file writes against the
    # turn a human will look at. Empty for workers and direct `run_query`
    # callers, which turns the ledger off for them — the footer and the revert
    # button are chat surfaces, and a worker turn has no reader.
    turn_id: str = ""

    # #544: the queue item this turn is running for (`item:<source>:<id>`),
    # carried to the aggregator in `_meta` so its effect ledger can tell a
    # retry's second `email_send` from a new one. The pool binds
    # `policy.current_effect_scope` in its own task; a session-backed worker
    # turn runs in the backend's task after a loopback POST, where that
    # contextvar is empty, so the router sets this from the payload instead
    # and the loop prefers it over the contextvar. Empty for a chat turn,
    # which keeps it out of the ledger by design.
    effect_scope: str = ""

    # Background-task notification drain. When set, the loop calls this
    # at the top of each iteration; the callable returns a list of
    # OpenAI-format messages (typically role: "user" with a
    # <task_notification> XML body) to splice into chat_messages before
    # the next vLLM request. The callback is also expected to persist
    # the messages into the session JSON, since the harness no longer
    # owns persistence past run_query's entry.
    notification_drain: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None

    # Per-iteration state re-anchor. Session state that lives OUTSIDE the
    # conversation — the todo list, the plan, the goal — is rendered into
    # the system prompt once at turn start and then goes stale: the system
    # prompt sits at position 0 and the loop only ever appends, so
    # rewriting it mid-turn would invalidate the whole cached prefix on
    # every iteration. This callback re-anchors that state by APPENDING
    # instead, which keeps the prefix intact.
    #
    # Called with the 1-based iteration number; returns messages to splice
    # in, or [] for "nothing to say". Unlike `notification_drain` these
    # are ephemeral scaffolding and must NOT be persisted to the session
    # JSON — they describe state as of this iteration and would be stale
    # (and duplicated) on the next turn's reconstruction.
    #
    # Motivating case: session 20260905_151355_iv5174 called TodoWrite at
    # iteration 6 of 52 and never again. The `<active_todos>` block was
    # empty at turn start, so for 46 iterations the only trace of the list
    # was a tool call 130k tokens back.
    state_anchor: Callable[[int], Awaitable[list[dict[str, Any]]]] | None = None

    # Tool-call summaries. Adds one string parameter, `summary`, to every
    # advertised tool: a short phrase the model writes describing what the
    # call is doing, rendered beside the tool name in the transcript. It is
    # display metadata — the harness strips it before dispatch and before
    # replaying the call back as history, so no tool ever sees it. Turning
    # this off removes the parameter from every schema; the UI then falls
    # back to showing the tool name alone. Fed from
    # `harness.tool_call_summaries`.
    tool_call_summaries: bool = True

    # Preserved thinking. Carry this many recent iterations' reasoning
    # back into `chat_messages` as the assistant messages' `reasoning`
    # field, which Qwen3.8-Flash-Next's template renders into each prior
    # turn's <think> block. 0 disables (the pre-2026-09-05 behaviour:
    # every historical turn shows an empty <think>, and the model
    # re-derives its own conclusions each iteration).
    preserve_thinking_iterations: int = 0

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

    # ---- context pressure -------------------------------------------------
    # Live view of how much of the window this turn has spent. Caller-owned
    # and mutable, the same pattern as `chat_messages_handle`: the router
    # builds one, hands it to the loop AND to the Inner Voice observer, and
    # both read the one figure. The loop builds a private meter when this is
    # None, so a bare `run_query` caller still gets relief — it just has no
    # `<context>` anchor, because nothing outside the loop can see the meter.
    context_meter: Any | None = None

    # Relief ladder. Microcompaction clears tool results and nothing else,
    # so a turn whose residue is reasoning and `Write` bodies has no valve:
    # on 2026-09-11 a pass freed 1.0 tool result (254k -> 252k against a 109k
    # target) and the round died at the wall with finished work uncommitted.
    context_relief_enabled: bool = True
    # An observer inject on a terminal iteration continues the loop. Below
    # this much headroom the next request has no room to answer in, so the
    # turn ends as `context_exhausted` instead of spending an iteration
    # producing a truncated completion.
    context_relief_terminal_floor_tokens: int = 12_000
    # Pre-request floor: relieve before `stream_chat` when the completion
    # would have less than this to write into.
    context_relief_min_completion_tokens: int = 6_000
    # Second reasoning window, applied ONLY under pressure. The turn-entry
    # window (`preserve_thinking_iterations`) stays what it is: pruning
    # mid-turn invalidates the cached prefix (#520), which is a real cost
    # and worth paying only when the alternative is losing the turn.
    context_relief_reasoning_keep_under_pressure: int = 2
    # Spill `Write`/`Edit` bodies out of assistant tool_call arguments once
    # their result has landed. A 16k-char file body rides in the prompt for
    # the rest of the turn otherwise, and no existing rung can reach it.
    context_relief_shrink_arguments: bool = True
    context_relief_shrink_arguments_min_chars: int = 2_000
    context_relief_shrink_arguments_tools: tuple[str, ...] = ("Write", "Edit")
    # Opt-in A/B only: reserve completion room with an explicit max_tokens.
    # Off because a cap truncates a long think on a short answer, and
    # llama.cpp truncates silently rather than erroring.
    context_relief_send_max_tokens_reservation: bool = False

    # Structured final answer. When `final_schema` is set and the turn ends
    # of its own accord, the loop runs ONE extra completion that restates the
    # answer as a JSON object matching this schema, and reports it on the
    # `result` event as `structured`.
    #
    # It cannot be applied to the turn itself: a guided-decoding grammar
    # constrains `content`, and during the loop the model has to be free to
    # emit qwen3_xml tool calls. See app/harness/finalizer.py — in particular
    # why the extra request must send the identical `tools` list with
    # `tool_choice: "none"` rather than dropping tools.
    #
    # Inert when unset, which is every caller but the worker sources that
    # want a machine-readable verdict.
    # Concurrent tool dispatch, for READ-ONLY batches only. A batch runs
    # concurrently when every call in it is annotated `readOnlyHint` (or is a
    # parse error, or ToolSearch); anything that can write — Bash, Edit,
    # Write, a mutating MCP tool — makes the whole batch sequential.
    #
    # Read-only-only is not caution, it is the only classification available
    # that is not a guess. `agent_mcp/annotations.py` is the declared table
    # and a server that sets no hints qualifies nothing, which is that file's
    # contract. A deny-list is the pattern it was written to replace, and
    # `Bash(cat ...)` is exactly the shell-command classification the safety
    # hook deliberately refuses to play at.
    parallel_tool_calls_enabled: bool = False
    parallel_tool_calls_max_concurrency: int = 4

    final_schema: dict[str, Any] | None = None
    final_schema_prompt: str = ""
    # 4096, not 1024: this is the WHOLE completion budget and the grammar only
    # applies after `</think>`, so reasoning tokens are spent before the
    # object starts. At 1024, 14 of the first 34 triage verdicts (41%) came
    # back as well-formed JSON cut mid-string inside `check`/`evidence` — 13
    # of them `confirmed`, the verbose verdicts that matter — and fell back
    # to the regex parser. `finalizer.py` names a truncation as such now.
    # 8192 since the review grader's object (five clauses with notes, a
    # test-honesty list, a seams list) cut off at exactly 4096 on its second
    # calibration case. A cap, not a spend. config.yaml carries the same
    # number and a test pins the two together.
    finalizer_max_tokens: int = 8192
    finalizer_timeout_s: float = 180.0
