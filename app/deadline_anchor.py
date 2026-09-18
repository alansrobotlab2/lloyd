"""Budget anchors: a budget the model cannot see is a deadline it cannot meet.

An agent turn has two clocks and this module holds the sentence for each.

The **wall clock**. An autonomy task is bounded by `asyncio.timeout`; a worker
turn is bounded by `run_prompt_in_session`'s own `wait_for`. Neither was
visible to the model until #80/#78/#24 (autonomy, 2026-09-08) and #446
(automod, 2026-09-09) each died holding an answer they were never asked to
write down. #446 is the sharpest case: it committed 757 lines into its worktree
at 06:43:36 and the wall clock killed it at 06:43:50 — fourteen seconds, one
`automod_gate` call, short of the verdict that would have landed it.

The **iteration clock**, `RunOptions.max_turns`, which `app/harness/loop.py`
enforces and `app/harness/finalizer.py` records no verdict for. Its warning
lived as a closure inside `app/routers/messages.py::_build_state_anchor`, so it
reached the chat path and nowhere else — and a scheduled task with a 2400 s wall
clock and a 60-iteration cap died at `turns=61` having been warned about
neither (#1061: 20 such runs in `workers.db`, 2026-09-04 → 09-18).

So both builders are here and callers compose what they have. `compose_state_anchors`
returns the one callable the harness takes, because a caller with both clocks
that passed two anchors would be a caller with two chances to drift.

The reason this is a module rather than a private helper in each caller: two
copies of "how close is the budget" drift, and the one that drifts is the one
nobody is watching. That is true of the wording as much as the arithmetic — the
chat path's `<budget>Iteration N of M` bytes are a measurement surface, and
#769's anchor-firing ledger has to match on them.
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


# The two iteration levels, in whole percent so the crossing test is integer
# arithmetic: `iteration * 100 < pct * max_turns` cannot round. 75 and 90 are
# the chat path's long-standing pair (#278), and the wording below is its
# wording, unchanged — #769 counts firings by matching these bytes.
ITERATION_WARN_PERCENTAGES = (75, 90)


def build_iteration_anchor(
    max_turns: int,
) -> Callable[[int], Awaitable[list[dict[str, Any]]]] | None:
    """A `RunOptions.state_anchor` that announces the turn cap, once per level.

    The only input is `max_turns`, and it must be the same value the caller
    hands `RunOptions`: the warning is a fraction of a cap, and a fraction of a
    cap the harness does not enforce arrives either too early to act on or
    after the run is already dead.

    Returns None without a positive cap — the harness's own "nothing to say".
    It never invents one either: with a cap of 60 it speaks at iterations 45
    and 54, and at no other point, so a caller that has only this clock composes
    an anchor that says nothing about a wall clock it does not have.
    """
    if not max_turns or max_turns <= 0:
        return None

    fired: set[int] = set()

    async def anchor(iteration: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for pct in ITERATION_WARN_PERCENTAGES:
            if pct in fired or iteration * 100 < pct * max_turns:
                continue
            fired.add(pct)
            left = max(0, int(max_turns) - int(iteration))
            out.append({"role": "user", "content": (
                f"<budget>Iteration {iteration} of {max_turns}: {left} iteration(s) "
                "remain before this turn is stopped. A turn cut off at the budget "
                "ends with no report and lands nothing. If an automod round is open, "
                "gate and land it now (automod_gate, then automod_land) or abort it; "
                "otherwise finish — say what is done and what is not.</budget>")})
        return out

    return anchor


def compose_state_anchors(
    *anchors: Callable[[int], Awaitable[list[dict[str, Any]]]] | None,
) -> Callable[[int], Awaitable[list[dict[str, Any]]]] | None:
    """Fuse several anchors into the single callable `RunOptions.state_anchor`
    takes, emitting them in the order given.

    The harness has one hook and calls it once per iteration
    (`app/harness/loop.py`), so a caller holding two clocks has to choose: pass
    one and the other is invisible, or build its own wrapper and be the only
    place that order is written down. This is that place.

    `None` entries are dropped, which is what lets a caller pass builders that
    return `None` when the clock they describe does not exist. With nothing
    left, the answer is `None` — no callback, no warning, no invented budget.
    With exactly one, that anchor is returned unchanged rather than wrapped, so
    a caller with one clock behaves byte-for-byte as it did before the second
    clock existed.
    """
    parts = [a for a in anchors if a is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    async def anchor(iteration: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for part in parts:
            out.extend(await part(iteration))
        return out

    return anchor
