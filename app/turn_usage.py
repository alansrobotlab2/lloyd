"""The turn-level token row: one unit decision, applied by every writer.

Backlog #859, closing instrument for #520.

A turn is several chat completions. Each completion reports its own prompt
size and its own prefix-cache hits, and vLLM guarantees
``cached_tokens <= prompt_tokens`` for a single request — so a *per-iteration*
stats row has always been self-consistent. What was not self-consistent was
the *turn* row built on top of it: `app/harness/loop.py`
``_accumulate_iteration_usage`` took the **max** of ``input_tokens`` across
iterations (the peak — the meaning every consumer in the tree reads it by)
and the **sum** of ``cache_read``. Dividing the two is how the dashboard's
usage panel, ``usage_store.summary()`` and #520's cached-fraction measurement
all compute a hit rate, so a five-iteration turn reported five iterations'
cached tokens against one iteration's prompt tokens. On 2026-09-11 that was
665 persisted ``assistant.stats`` rows, the worst at
``cache_read: 17,564,800`` against ``input_tokens: 202,467``.

The decision, made once here so no writer can re-split the units:

* ``input_tokens`` / ``cache_read`` — **peak** pair. Commensurable, so the
  ratio cannot exceed 100%, and ``input_tokens`` keeps the meaning it already
  has everywhere else (``_maybe_finalize`` deliberately declines to fold the
  finalizer's prompt into it for exactly that reason).
* ``prompt_tokens_sum`` / ``cache_read_sum`` — **sum** pair, under names that
  say so. The turn's real cost, still bounded at 100% because it is a sum over
  a sum.

Both pairs are on the row. Whichever one you divide, the answer cannot exceed
100% — that is the contract, and it is the reason the sums are named rather
than implied.

This module exists because two separate writers build the row:
``app/routers/messages.py`` (the chat path, and therefore mission-control /
browser / voice traffic) and ``app/run_recorder.py`` (background runs: worker,
autonomy, session-distill). A rule enforced in one of them is a rule the other
one breaks on its next edit.
"""

from __future__ import annotations

from typing import Any


def _int(source: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    return 0


def turn_usage_row(result_usage: dict[str, Any] | None) -> dict[str, int]:
    """Bounded token figures for one turn's persisted usage row.

    Accepts the usage dict the harness publishes on its final ``result``
    event. Tolerates OpenAI-style key spellings and a missing sum — a turn
    produced before this change, or a finalizer-only turn whose prompt is
    carried under ``finalizer_input_tokens`` — by falling back to the peak
    rather than to a zero, which would read as a cold cache.

    Every returned pair satisfies ``cached <= prompt``. Clamping is deliberate
    and loses nothing an operator needs: the exact per-request numbers are
    still on the session's per-iteration rows, so an engine reading more
    cached than prompted is visible there while the aggregate stays a number
    that can be divided.
    """
    usage = result_usage or {}

    peak_prompt = _int(usage, "input_tokens", "prompt_tokens")
    peak_cache = _int(usage, "cache_read", "prompt_tokens_cached")

    sum_prompt = _int(usage, "prompt_tokens_sum") or peak_prompt
    # The peak is inside the sum by construction; enforce it anyway so a
    # truncated or hand-constructed dict cannot put them out of order.
    sum_prompt = max(sum_prompt, peak_prompt)
    sum_cache = _int(usage, "cache_read_sum") or peak_cache

    return {
        "input_tokens": peak_prompt,
        "cache_read": min(peak_cache, peak_prompt),
        "prompt_tokens_sum": sum_prompt,
        "cache_read_sum": min(sum_cache, sum_prompt),
    }


# ---------------------------------------------------------------------------
# P11: the turn's harness telemetry, folded once for both writers
# ---------------------------------------------------------------------------
#
# The same two writers again, and the same reason to fold here: a column one
# of them fills and the other leaves NULL is a dashboard that reads the chat
# path as healthy and background runs as unmeasured, or the reverse.

# The harness events the row counts, keyed by column. Emitted through
# `app.harness.telemetry.log_harness_event` by the loop (overflow), hooks.py
# (a gate that raised, D5) and the stream retry (D7); counted off the
# context-bound tally, so neither emitter has to know a usage row exists.
_COUNTED_EVENTS = {
    "stream_retries": "harness.stream_retried",
    "overflow_recoveries": "harness.overflow_recovered",
    "hook_raised": "harness.hook_raised",
}


class TurnTelemetry:
    """Folds one turn's events into the P11 columns of its usage row.

    Construct it before driving `run_query` — construction binds the
    context's harness-event tally — then hand it every event (`note`), and
    splat `row()` into `usage_store.record_usage`. NULL means unmeasured
    throughout: a turn that never received a first chunk has no TTFT, and a
    turn whose engine never reported reasoning tokens has no reasoning count,
    rather than zeros that read as fast and thoughtless.
    """

    def __init__(self) -> None:
        from app.harness.telemetry import bind_event_counts

        self._events = bind_event_counts()
        self.ttfts: list[int] = []
        self.reasoning_tokens: int | None = None
        self.tool_calls = 0
        self.tool_errors = 0
        self.tool_errors_by_class: dict[str, int] = {}
        self.tool_ms_total: int | None = None
        self.stop_reason: str | None = None
        self.wrapped_up: bool | None = None
        self.result_reasoning_tokens: int | None = None
        # Running token totals from the per-iteration events, for a turn
        # that never reaches `result` (D12): peak prompt, summed output.
        self._peak_input = 0
        self._peak_cache_read = 0
        self._output_sum = 0
        self._cache_create_sum = 0
        self._iterations = 0

    def note(self, evt: dict[str, Any]) -> None:
        """Dispatch on the event type; anything else is ignored."""
        etype = evt.get("type")
        if etype == "assistant_message":
            self.note_assistant(evt)
        elif etype == "tool_result":
            self.note_tool_result(evt)
        elif etype == "result":
            self.note_result(evt)

    def note_assistant(self, evt: dict[str, Any]) -> None:
        self._iterations += 1
        ttft = evt.get("ttft_ms")
        if isinstance(ttft, int) and not isinstance(ttft, bool):
            self.ttfts.append(max(0, ttft))
        usage = evt.get("usage") or {}
        rt = usage.get("reasoning_tokens")
        if isinstance(rt, int) and not isinstance(rt, bool):
            self.reasoning_tokens = (self.reasoning_tokens or 0) + max(0, rt)
        self._peak_input = max(self._peak_input, _int(
            usage, "input_tokens", "prompt_tokens"))
        self._peak_cache_read = max(self._peak_cache_read, _int(
            usage, "cache_read", "prompt_tokens_cached"))
        self._output_sum += _int(usage, "output_tokens", "completion_tokens")
        self._cache_create_sum += _int(usage, "cache_create")

    def note_tool_result(self, evt: dict[str, Any]) -> None:
        self.tool_calls += 1
        if evt.get("is_error"):
            self.tool_errors += 1
            cls = str(evt.get("error_class") or "tool_error")
            self.tool_errors_by_class[cls] = self.tool_errors_by_class.get(cls, 0) + 1
        ms = evt.get("duration_ms")
        if isinstance(ms, int) and not isinstance(ms, bool):
            self.tool_ms_total = (self.tool_ms_total or 0) + max(0, ms)

    def note_result(self, evt: dict[str, Any]) -> None:
        self.stop_reason = str(evt.get("stop_reason") or "") or None
        self.wrapped_up = bool(evt.get("wrapped_up"))
        rt = (evt.get("usage") or {}).get("reasoning_tokens")
        if isinstance(rt, int) and not isinstance(rt, bool):
            self.result_reasoning_tokens = max(0, rt)

    def row(self) -> dict[str, Any]:
        """The P11 columns, as `usage_store.record_usage` keyword arguments."""
        reasoning = (self.result_reasoning_tokens
                     if self.result_reasoning_tokens is not None
                     else self.reasoning_tokens)
        out: dict[str, Any] = {
            "stop_reason": self.stop_reason,
            "reasoning_tokens": reasoning,
            "ttft_ms_first": self.ttfts[0] if self.ttfts else None,
            "ttft_ms_max": max(self.ttfts) if self.ttfts else None,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "tool_errors_by_class": dict(self.tool_errors_by_class),
            "tool_ms_total": self.tool_ms_total,
            "wrapped_up": (None if self.wrapped_up is None
                           else int(self.wrapped_up)),
        }
        for column, event in _COUNTED_EVENTS.items():
            out[column] = int(self._events.get(event, 0))
        return out

    def partial_row(self) -> dict[str, int]:
        """Token totals from the iterations seen so far, for a turn that died
        before its `result` event (D12). Same units as `turn_usage_row`:
        prompt and cache_read are peaks, output and cache_create are sums."""
        return {
            "input_tokens": self._peak_input,
            "output_tokens": self._output_sum,
            "cache_read": min(self._peak_cache_read, self._peak_input),
            "cache_create": self._cache_create_sum,
            "num_turns": self._iterations,
        }
