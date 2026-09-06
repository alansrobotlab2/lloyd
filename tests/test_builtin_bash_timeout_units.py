"""The Bash tool's `timeout` argument is milliseconds; models pass seconds.

`timeout` is documented as milliseconds ("Max ms before kill (default
120000, max 600000)"), but every checkpoint arrives with the opposite prior:
Anthropic's own bash toolset takes seconds
(`bash command timed out after {timeout}s`,
anthropic/lib/tools/agent_toolset.py:532).

On 2026-09-06 task #75 (AI Engineer YouTube monitor) on the 35B secondary
passed `timeout: 600` meaning ten minutes. The tool read 600 milliseconds,
SIGKILLed `ai-engineer-monitor.py --process-one` before the channel RSS fetch
finished, the model interpreted the failure as "nothing to do" and replied
[SILENT], and the run was recorded `status: success` — the channel went
unmonitored while the scheduler reported green.

The guard reads anything under 5 seconds as seconds. These assert both sides:
the rescue, and that a real millisecond timeout still kills.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_mcp import builtin_bash as B  # noqa: E402


async def _run(command: str, **args) -> tuple[float, str]:
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    out = await B._bash({"command": command, **args})
    return loop.time() - t0, out


def test_passing_seconds_no_longer_gutters_the_command():
    """`timeout: 600` from a model means ten minutes, not 0.6s."""

    async def scenario():
        return await _run("sleep 1.2; echo RAN", timeout=600)

    dur, out = asyncio.run(scenario())
    assert "RAN" in out, f"command was killed as it was on 2026-09-06: {out!r}"
    assert dur < 10, f"scaled to seconds but absurdly slow: {dur:.1f}s"


def test_tiny_values_scale_to_seconds():
    async def scenario():
        return await _run("sleep 1; echo TINY", timeout=100)

    _, out = asyncio.run(scenario())
    assert "TINY" in out, out


def test_values_at_or_above_the_ceiling_are_still_milliseconds():
    """The rescue must not swallow genuine millisecond budgets."""

    async def scenario():
        return await _run("sleep 8", timeout=5_000)

    dur, out = asyncio.run(scenario())
    assert "timed out after 5000ms" in out, (
        f"5000 is >= the ceiling and must be honoured as ms, got {out!r}"
    )
    assert dur < 7, f"should kill at ~5s, took {dur:.1f}s"


def test_default_and_cap_unchanged():
    assert B.DEFAULT_TIMEOUT_MS == 120_000
    assert B.MAX_TIMEOUT_MS == 600_000

    async def scenario():
        plain = await _run("echo OK")
        clamped = await _run("echo OK", timeout=99_999_999)
        zero = await _run("echo OK", timeout=0)
        return plain, clamped, zero

    (d1, o1), (d2, o2), (d3, o3) = asyncio.run(scenario())
    assert o1.strip() == "OK"
    assert o2.strip() == "OK", "over-cap timeout must clamp, not fail"
    assert o3.strip() == "OK", "timeout=0 must fall back to the default"


def test_guard_ceiling_leaves_room_for_a_real_millisecond_budget():
    """The ceiling must sit inside the legal range, so a caller who really
    does mean milliseconds and passes e.g. 30_000 is never rescaled, and a
    scaled seconds value still lands within the cap instead of overrunning
    it."""

    assert B._SECONDS_CEILING_MS < B.MAX_TIMEOUT_MS
    # A typical model-supplied second-budget (600 = ten minutes) scales to a
    # value the clamp can absorb.
    assert (600 * 1_000) <= B.MAX_TIMEOUT_MS
    # And anything pathological is clamped, not honoured.
    assert min(600_000_000, B.MAX_TIMEOUT_MS) == B.MAX_TIMEOUT_MS
