"""How much of the context window this turn has actually used.

One number, three readers: the loop (to decide when to relieve pressure and
whether a terminal inject can still be answered), the `<context>` state
anchor (to tell the model), and the Inner Voice observer (to stop nudging a
turn that has no room to act on a nudge).

**Why a shared object rather than three private estimators.** Every reader
needs the same figure and each had its own partial view of it:
`_intra_turn_microcompact` recomputed the estimate every iteration and threw
it away, the anchor had no view at all, and the observer judged a turn at
241k tokens exactly as it judged one at 40k. On 2026-09-11 three autocode
rounds (866-a, 869, 875) died at the wall with finished work in a worktree;
875's completions were capped at `262_144 - prompt` and shrank 3147 -> 1760
-> 1175 tokens while it retried the same heredoc, and the observer then
injected on a silent terminal iteration with zero room left to answer in.

**Anchoring.** vLLM reports `input_tokens` for the prompt it just processed:
the real number, including the system prompt and the tool schemas, neither of
which is in `chat_messages`. The estimator (`app.compaction`) sees only
`chat_messages`. Carrying the difference as a fixed `offset` keeps both halves
in the same units — the same arithmetic `_intra_turn_microcompact` already
uses, and for the same reason: triggering on the real figure while budgeting
against the estimate means triggering and then clearing nothing.

**Before the first usage report the meter is unmeasured**, and every reader
must fail open on that: `measured` is False, `used` is 0 and nothing fires.
An iteration-1 turn with no report is not a turn with no context.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Fallback when `app.compaction` cannot be imported (bench scripts, tests
# that stub the module out). Matches compaction.DEFAULT_CONTEXT_WINDOW.
_DEFAULT_WINDOW = 128_000


def _compaction():
    """Lazy import, the pattern `_intra_turn_microcompact` already uses.

    `app.compaction` lazy-imports this package in the other direction; a
    module-level edge here would close the cycle.
    """
    from app import compaction

    return compaction


def context_window_for(model: str) -> int:
    """`models.<model>.context_length`, or the default. Never raises."""
    try:
        return int(_compaction().get_context_window(model or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("context_meter: window lookup failed for %r: %s", model, exc)
        return _DEFAULT_WINDOW


def _estimate(messages: list[dict[str, Any]]) -> int:
    try:
        return int(_compaction().estimate_conversation_tokens(messages, ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("context_meter: estimate failed: %s", exc)
        return 0


class ContextMeter:
    """Live view of one turn's prompt size.

    `used` = the last reported prompt size + an estimate of everything
    appended since. Callers mutate `chat_messages` in place all through an
    iteration, so the meter is told about it rather than watching:

        observe_usage(usage, len(chat_messages))   # a request came back
        observe_append(chat_messages)              # something was appended
        resync(chat_messages)                      # the list was rewritten

    `observe_append` recomputes rather than accumulates, so calling it twice
    for one append is a no-op instead of a double count.
    """

    __slots__ = (
        "window", "_reported", "_reported_at_len", "_appended", "_offset",
    )

    def __init__(
        self,
        window: int,
        reported_input_tokens: int = 0,
        reported_at_len: int = 0,
        appended_estimate: int = 0,
    ) -> None:
        self.window = int(window) if window and window > 0 else _DEFAULT_WINDOW
        self._reported = max(0, int(reported_input_tokens))
        self._reported_at_len = max(0, int(reported_at_len))
        self._appended = max(0, int(appended_estimate))
        # Everything in the prompt that is not in `chat_messages` — the
        # system prompt and the tool schemas. Learned on the first
        # `observe_append`, which is the first moment both the reported
        # figure and the list are in hand.
        self._offset = 0

    # -- state ------------------------------------------------------------

    @property
    def measured(self) -> bool:
        """True once an engine has reported a prompt size for this turn."""
        return self._reported > 0

    @property
    def used(self) -> int:
        if not self.measured:
            return 0
        return self._reported + self._appended

    @property
    def headroom(self) -> int:
        """Tokens left before the window. 0 when unmeasured or over."""
        if not self.measured:
            return self.window
        return max(0, self.window - self.used)

    @property
    def fraction(self) -> float:
        """`used` as a fraction of the whole window. 0.0 when unmeasured."""
        if not self.measured or self.window <= 0:
            return 0.0
        return self.used / self.window

    @property
    def threshold(self) -> int:
        """The compaction wall — where the turn-start pass starts truncating."""
        try:
            return int(_compaction().truncation_threshold(self.window))
        except Exception:  # noqa: BLE001
            return max(1_000, self.window - 52_000)

    @property
    def threshold_fraction(self) -> float:
        """`used` against the compaction wall rather than the window.

        What the observer's pressure note keys on: past this the turn-start
        pass would already be truncating, so an iteration here is one the
        loop is holding open by clearing tool results.
        """
        thr = self.threshold
        if not self.measured or thr <= 0:
            return 0.0
        return self.used / thr

    # -- observation ------------------------------------------------------

    def observe_usage(self, usage: dict[str, Any] | None, n_messages: int) -> None:
        """Record what the engine said about the request it just answered.

        `n_messages` is `len(chat_messages)` as the request went out, so
        everything appended afterwards is attributed to `_appended` and not
        double-counted against the reported figure.
        """
        try:
            reported = int((usage or {}).get("input_tokens", 0) or 0)
        except (TypeError, ValueError):
            reported = 0
        if reported <= 0:
            return
        self._reported = reported
        self._reported_at_len = max(0, int(n_messages))
        self._appended = 0

    def observe_append(self, chat_messages: list[dict[str, Any]]) -> None:
        """Re-estimate everything appended since the last usage report."""
        if not self.measured:
            return
        tail = chat_messages[self._reported_at_len:]
        self._appended = _estimate(tail) if tail else 0
        # First sight of both halves: learn the fixed cost the estimator
        # cannot see. Clamped at 0 so an over-reporting estimate cannot
        # invent headroom.
        prefix_est = max(0, _estimate(chat_messages) - self._appended)
        self._offset = max(0, self._reported - prefix_est)

    def resync(self, chat_messages: list[dict[str, Any]]) -> None:
        """Re-anchor after an in-place rewrite (relief, microcompact).

        The reported figure describes a prompt that no longer exists, so it
        is replaced by `offset + estimate(list)` and the meter carries on
        from there as though the engine had reported that.
        """
        if not self.measured:
            return
        self._reported = max(1, self._offset + _estimate(chat_messages))
        self._reported_at_len = len(chat_messages)
        self._appended = 0

    # -- diagnostics ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """What rides on an `assistant_message` event and into the logs."""
        return {
            "input_tokens": self.used,
            "context_window": self.window,
            "headroom": self.headroom,
            "fraction": round(self.fraction, 4),
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        if not self.measured:
            return f"<ContextMeter window={self.window} unmeasured>"
        return (
            f"<ContextMeter {self.used}/{self.window} "
            f"({self.fraction:.0%}) headroom={self.headroom}>"
        )


__all__ = ["ContextMeter", "context_window_for"]
