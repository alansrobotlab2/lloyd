"""The wall-clock anchor: a budget the model cannot see is a deadline it
cannot meet.

`RunOptions.max_turns` is warned about by the chat path's iteration anchor,
but iterations are not the budget most unattended work dies on. An autonomy
task is bounded by `asyncio.timeout`; a worker turn is bounded by
`run_prompt_in_session`'s own `wait_for`. Neither clock was visible to the
model until #80/#78/#24 (autonomy, 2026-09-08) and #446 (selfmod, 2026-09-09)
each died holding an answer they were never asked to write down.

#446 is the sharpest case: it committed 757 lines into its worktree at
06:43:36 and the wall clock killed it at 06:43:50 — fourteen seconds, one
`selfmod_gate` call, short of the verdict that would have landed it.

This module is the single definition. It lived privately in `autonomy.py`,
which is the wrong home once a second caller needs it: two copies of "how
close is the deadline" drift, and the one that drifts is the one nobody is
watching.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

# Two levels, because they ask for different things. Each fires once: a
# warning re-sent every iteration is one the model learns to skip.
BUDGET_WARN_FRACTIONS = (0.70, 0.90)


def build_deadline_anchor(
    timeout_s: float,
    *,
    what: str = "task",
) -> Callable[[int], Awaitable[list[dict[str, Any]]]] | None:
    """A `RunOptions.state_anchor` that announces the wall clock, once per level.

    The harness calls this at the top of each iteration, so resolution is one
    iteration: a single tool call longer than the remaining budget still
    overruns. That is the accepted limit — the failure being fixed is twenty
    short iterations past the stopping point, not one long one.

    `what` names the thing being bounded so the sentence reads right for the
    caller ("this task's", "this round's").
    """
    if not timeout_s or timeout_s <= 0:
        return None

    started = time.monotonic()
    fired: set[float] = set()

    async def anchor(iteration: int) -> list[dict[str, Any]]:
        elapsed = time.monotonic() - started
        out: list[dict[str, Any]] = []
        for frac in BUDGET_WARN_FRACTIONS:
            if frac in fired or elapsed < frac * timeout_s:
                continue
            fired.add(frac)
            left = max(0, int(timeout_s - elapsed))
            if frac >= 0.90:
                out.append({"role": "user", "content": (
                    f"<budget>{left}s of this {what}'s {int(timeout_s)}s budget remain. "
                    "Stop calling tools and write your report NOW, from what you "
                    "already have. A run cut off at the budget is recorded as a "
                    "failure and reports nothing, however much work it did — an "
                    "incomplete answer is worth far more than none. Say what you "
                    "found and what you did not get to.</budget>")})
            else:
                out.append({"role": "user", "content": (
                    f"<budget>{left}s of this {what}'s {int(timeout_s)}s budget remain. "
                    "Finish this run's own deliverable first; do not open new "
                    "lines of investigation. If the work is already done, report "
                    "now rather than verifying further.</budget>")})
        return out

    return anchor
