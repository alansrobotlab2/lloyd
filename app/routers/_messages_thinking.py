"""Per-phase thinking capture.

The harness emits one `thinking_done` per agent-loop iteration
(`app/harness/loop.py`), correctly timed from the first reasoning chunk to
the last. The router used to hold that in a single `accumulated_thinking`
buffer that each new phase *replaced*, and only flushed to disk when an
iteration produced both tool calls and non-empty text. A tool-only
iteration — the common shape — never flushed, so a forty-iteration turn
persisted exactly one reasoning phase: the last one. The rest survived
only in the event log as `brain1.thinking_block_emitted`.

Each phase is now its own message entry with `role="thinking"`, following
the `role="subliminal"` precedent in `_messages_subliminal.py`.

Two properties of the shape are load-bearing:

  - **The role is what keeps reasoning out of the transcripts, and it is
    the only thing that does.** Every transcript generated from a session
    log dispatches on role first — the vault exporter and
    `_build_capture_transcript` in `app/post_capture.py`,
    `app/session_titles.py::build_transcript`, the `scripts/memory/*`
    renderers, `scripts/extract-trajectories.py`, and the `session_recall`
    corpus in `agent_mcp/session.py` — and none of them has a branch for
    `"thinking"`, so the row renders as nothing everywhere. Measured, not
    assumed: with the role changed to `"assistant"` six of the seven
    producers leak the reasoning; with the role left alone, filling
    `content` leaks from none of them.

    `content` therefore stays empty as a second layer rather than the
    first — it is what protects against a future producer that iterates
    content blocks without checking role, which is the shape a leak would
    most plausibly arrive in. Do not read the empty content as the reason
    this works and conclude the role is free to change.
    `tests/test_thinking_trace_transcripts.py` pins both directions.

  - **The row is written when `thinking_done` arrives**, which the loop
    yields before it commits tool calls and before the `assistant_message`
    that flushes a text segment. Appending there lands the row ahead of
    that iteration's tool and text rows, so the timeline is in
    chronological order with no sorting logic.

`role="thinking"` is not in the conversation roles kept by
`app/compaction.py`, so these rows are dropped before history is rebuilt
and never re-enter the prompt. Preserved thinking (`loop.py`) remains the
separate in-flight mechanism it has always been.
"""

from __future__ import annotations

from app.sessions_io import SessionTurn


def _build_thinking_entry(
    turn: SessionTurn,
    text: str,
    duration_ms: int,
    seq: int,
    iteration: int,
    timestamp: str,
) -> dict:
    """Shape one reasoning phase as a message entry. Pure, for testability.

    `seq` orders phases within a turn and keeps ids unique; `iteration` is
    the agent-loop iteration the phase belongs to, which is what a reader
    correlating a thought with the tool call it produced actually wants.
    """
    return {
        "id": f"think_{turn.turn_id}_{seq}",
        "role": "thinking",
        # Deliberately empty — see the module docstring. The reasoning text
        # lives in `reasoning`, which no transcript producer reads.
        "content": [],
        "timestamp": timestamp,
        "reasoning": text,
        "reasoning_ms": duration_ms,
        "thinking": {
            "chars":     len(text),
            "iteration": iteration,
            "turn_id":   turn.turn_id,
        },
    }
