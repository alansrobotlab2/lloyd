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
