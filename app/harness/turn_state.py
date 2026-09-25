"""The state one `run_query` turn carries, and the state of one iteration.

A pure move (P13.1). These were ~40 locals of a 1,000-line `run_query`; the
phases it is now split into (`_open_turn`, `_prelude`, `_stream_iteration`,
`_dispatch_batch`, `_after_batch`, `_close_turn` in `loop.py`) read and write
them here instead. Nothing in this module decides anything: every field keeps
the name, the initial value and the meaning it had as a local, and the
comments that explained them moved with them.

`TurnState` lives for the turn; `Iteration` is rebuilt at the head of every
loop pass, exactly where the old locals were re-initialised. A field that was
initialised once before the loop (`broken_stream`, `forced_tool_choice`) is on
`TurnState` even when it reads per-iteration, because that is when it was
reset — which is to say never, or only where the loop reset it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.harness.context_meter import ContextMeter
from app.harness.mcp_pool import MCPPool
from app.harness.options import RunOptions
from app.harness.tool_search import LoadedToolSet


@dataclass(eq=False)
class TurnState:
    """Everything one turn of `run_query` holds across iterations."""

    options: RunOptions
    # The list every request is sent from — the caller's handle when one was
    # passed (Inner Voice appends to it), a private copy otherwise.
    chat_messages: list[dict[str, Any]]
    meter: ContextMeter
    pool: MCPPool
    started_at: float
    session_id: str
    # The surface's hidden tools plus whatever an allow-list leaves out; joins
    # every iteration's dispatch set (see `_surface_hidden`).
    surface_hidden: set[str]
    # The tools whose schema received the injected `summary` parameter.
    summary_tools: set[str]
    loaded_set: LoadedToolSet
    # The preserved-thinking window, applied at turn entry only (#520).
    keep_reasoning: int
    # A private relief latch per turn (#800); see `loop._ReliefLatch`.
    relief_latch: Any

    accumulated_text: str = ""
    # Two distinct usage trackers:
    #   iteration_usage (on `Iteration`) — ONE chat-completion's tokens.
    #   total_usage — cross-iteration aggregate. `input_tokens` and
    #     `cache_read` are a PEAK pair and `*_sum` the summed pair, so
    #     either ratio is a real fraction; other counters are summed.
    #     Reported on the final `result` event — see
    #     `_accumulate_iteration_usage`.
    total_usage: dict[str, int] = field(default_factory=dict)
    last_iteration_usage: dict[str, int] = field(default_factory=dict)
    # Caption bookkeeping — see `_CAPTION_NUDGE` and events.result.
    caption_total: int = 0
    caption_present: int = 0
    caption_nudged: bool = False
    num_turns: int = 0
    stop_reason: str = "stop"
    context_overflow_recoveries: int = 0
    max_context_overflow_recoveries: int = 2
    # One multimodal rejection latches image attachment off for the rest of
    # THIS run (#1419).
    images_latched: bool = False
    echo_guard_reprompts: int = 0
    # Broken-stream retries spent this turn (D7), and whether the iteration
    # in hand ended on one that could not be retried.
    stream_retries: int = 0
    broken_stream: bool = False
    # P6: the `tool_choice` the NEXT request carries when it is not "auto"
    # (the echo guard's "required"); consumed by that request.
    forced_tool_choice: str | None = None
    # P6b: "" -> "requested" (wrap-up message appended, the toolless request
    # is owed) -> "done". A retry of the wrap-up request re-enters as
    # "requested" and does not append again.
    wrapup_state: str = ""
    # Held so the finalizer can send the IDENTICAL array. Initialised before
    # the loop: a turn that breaks on its first check never assigns it.
    last_visible_tools: list[dict[str, Any]] = field(default_factory=list)
    # The iteration whose head (notification drain + state anchor) has
    # already run; an overflow/multimodal retry re-enters the head with the
    # SAME iteration number and must not append a second anchor (D9).
    prelude_done_for: int = 0


@dataclass(eq=False)
class Iteration:
    """One pass of the loop: one request, its stream, and its batch."""

    started_at: float
    # This iteration's disallowed set (plan-mode refresh + surface).
    current_disallowed: set[str]
    iteration_usage: dict[str, int] = field(default_factory=dict)
    assistant_text: str = ""
    thinking_text: str = ""
    # Wall time spent producing reasoning: first chunk to last. Not the
    # iteration's own clock, which also covers prefill.
    thinking_started_at: float | None = None
    thinking_last_at: float = 0.0
    tool_calls_acc: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None
    # The list length as the request went out; what the meter attributes the
    # engine's reported prompt size to.
    request_msgs_len: int = 0
    # P11: the request's own clock, started after the pre-request relief.
    request_started_at: float = 0.0
    first_chunk_at: float | None = None
    request_tool_choice: str = "auto"
    tool_calls_committed: list[dict[str, Any]] = field(default_factory=list)
    # `len(chat_messages)` before the assistant_message hook fired: an
    # observer inject lands after it, and it is where the batch begins (D8).
    batch_base: int = 0
    observer_injected: bool = False
    # How the phase that ran last wants the loop to go on: "" (fall through
    # to the next phase), "continue" (next pass) or "break" (end the turn).
    flow: str = ""
