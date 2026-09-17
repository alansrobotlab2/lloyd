"""#1204: one cold `/api/dashboard` cycle over the real board, timed.

The page's render is gated on this one request (`DashboardPage.tsx:1146`
renders "Loading dashboard…" until the whole payload arrives), so page-render
time equals API time. Measured at triage, 2026-09-16: a cold cycle cost
**15.22 s over a 1,142-file board**, and the user saw "Loading dashboard…" for
exactly that long, recurring every 60 s on the scorecard TTL rather than once.

The contract's number is **under 6.0 s**, not the item's original sub-second
target: the loader swap alone was measured at 5.00 s cold, and anything below
that needs recommendation B (one shared board walk, analytics off the read
path), which is deferred to a follow-on round with #1199. Asserting 1.0 s here
would be a test that cannot pass, which is a defect in the contract rather than
in the code.

Three things keep this from being a stopwatch that always passes:

* the board file count is printed and asserted > 0 — the walk is the thing
  being timed, so a cycle over zero files is no measurement at all;
* the payload must actually carry the two heavy sections, because a `_gather`
  that swallowed an exception would return `{"error": ...}` in milliseconds and
  read as a speed-up;
* the cache is cleared first. A warm cycle costs 0.06 s and would trivially
  satisfy the bound while proving nothing about the cold path the user pays.
"""

from __future__ import annotations

import json
import time

import pytest

from app.routers import dashboard as dash
from board_presence import board_files_or_stop, timed_ledger_or_stop
from scripts.automod import backlog as B

#: Triage baseline (2026-09-16): 15.22 s cold over 1,142 files. The loader swap
#: alone measured 5.00 s, which is why the bound is 6.0 and not 1.0 — see the
#: module docstring and '## Findings (triage 2026-09-16)' on #1204.
COLD_BUDGET_S = 6.0


@pytest.fixture(autouse=True)
def _cold():
    """Every timing test starts from an empty cache, in both layers."""
    dash._cache.clear()
    B._ledger_cache_clear()
    yield
    dash._cache.clear()
    B._ledger_cache_clear()


async def _cold_cycle() -> dict:
    """One cold call through the real route handler, decoded."""
    response = await dash.get_dashboard()
    return json.loads(response.body.decode("utf-8"))


async def test_one_cold_dashboard_cycle_beats_the_budget(monkeypatch):
    # No `skipif` on the board directory: with one, a vault root that exists and
    # a board that has been emptied or moved skips silently and the budget reads
    # as not-run. `board_files_or_stop` fails on every state that leaves it
    # nothing to walk — moved board, emptied board, no vault. See tests/board_presence.py.
    files = board_files_or_stop(what="cold-cycle budget", numeric_names=True)
    ledger = timed_ledger_or_stop(monkeypatch, what="cold-cycle budget")
    print(f"board: {len(files)} item files; ledger {ledger}: "
          f"{ledger.stat().st_size} bytes")
    assert ledger.stat().st_size > 0, (
        f"{ledger} is empty, so this cycle decoded no ledger at all and the "
        f"budget does not constrain the ledger re-decode path"
    )

    started = time.perf_counter()
    payload = await _cold_cycle()
    elapsed = time.perf_counter() - started
    print(f"cold cycle: {elapsed:.2f}s over {len(files)} files")

    # Positive control: the two sections that cost the time must have run.
    backlog = payload.get("backlog") or {}
    automod = payload.get("automod") or {}
    assert isinstance(backlog, dict) and "error" not in backlog, (
        f"the backlog section errored out ({backlog}), so the timed cycle never "
        f"walked the board — a fast error is not a fast dashboard"
    )
    assert isinstance(automod, dict) and "error" not in automod, (
        f"the automod section errored out ({automod}), which is the section that "
        f"cost 13.07 s cold at triage; without it the measurement is of a "
        f"different request"
    )
    assert backlog.get("total", 0) > 0, (
        f"backlog.total reads {backlog.get('total')} after walking "
        f"{len(files)} files — the scan parsed nothing, so the budget would be "
        f"met by a broken reader"
    )

    assert elapsed < COLD_BUDGET_S, (
        f"one cold /api/dashboard cycle over {len(files)} board files took "
        f"{elapsed:.2f}s; the budget is {COLD_BUDGET_S}s. Triage baseline "
        f"15.22 s at 1,142 files, loader swap alone measured 5.00 s. The "
        f"remaining cost is named on #1204: re-walking the board per section "
        f"(recommendation B) and re-decoding the ledger (recommendation, "
        f"tests/test_backlog_ledger_cache.py)."
    )


async def test_the_warm_cycle_is_an_order_of_magnitude_cheaper_than_cold(monkeypatch):
    """The cold bound is only meaningful against a much cheaper warm baseline.

    Advisory finding on round `SM_20260917_055508`: "Docstring states a 10x
    warm/cold ratio contract but the code asserts only the absolute
    'warm < 1.0'; at the measured cold 1.69 s a 0.9 s warm cycle (1.9x) would
    pass while the stated contract is violated." Correct, and the fix is to
    measure both here rather than assert a number the neighbouring sentence
    never checks. Cold and warm are timed back to back in this one test, on the
    same board and the same ledger, so the ratio is a measured pair and not a
    pair of numbers from two runs.

    Measured on this branch at the 1,141-file board: cold 1.69 s, warm 0.073 s —
    a 23x ratio. The assert asks for 10x, the margin that says the caches are
    doing their job. #1204 section 5 measured the *old* warm path (a 2 s poll
    against `_VAULT_SCAN_TTL_S = 10.0`) and got every fifth request paying a ~2 s
    walk — a warm cycle at 1x the cold one, which is exactly what this fires on.

    The 10x is not a restatement of that 2 s/10 s pairing; it cannot be, since
    TTL-vs-poll is a schedule property and this is a wall-clock measurement. What
    it pins is the failure that pairing produced: a section recomputing on a poll
    whose TTL has not expired. If warm and cold ever read alike, the cache is
    dead and the cold test above has been timing the cache-miss path forever
    without saying so. The absolute `warm < 1.0` stays as the second half — it
    catches a machine where *both* cycles are slow, which a ratio cannot see.
    """
    files = board_files_or_stop(what="warm-vs-cold comparison", numeric_names=True)
    timed_ledger_or_stop(monkeypatch, what="warm-vs-cold comparison")
    cold_started = time.perf_counter()
    await _cold_cycle()
    cold = time.perf_counter() - cold_started
    started = time.perf_counter()
    await _cold_cycle()
    warm = time.perf_counter() - started
    ratio = cold / warm if warm > 0 else float("inf")
    print(f"cold {cold:.3f}s → warm {warm:.3f}s over the same {len(files)} item "
          f"files: {ratio:.1f}x")
    assert ratio >= 10, (
        f"the warm cycle took {warm:.3f}s against a cold cycle of {cold:.3f}s — "
        f"{ratio:.1f}x, under the 10x this clause states. Baseline on this branch: "
        f"cold 1.69 s, warm 0.073 s, 23x. A warm cycle costs milliseconds of dict "
        f"lookups, so a warm cycle anywhere near the cold one means a section is "
        f"recomputing on every poll regardless of TTL — the 2 s spike in #1204 "
        f"section 5, and it also means the cold test above has been timing the "
        f"cache-miss path while reading as a bound.")
    assert warm < 1.0, (
        f"the second cycle, with every section cache warm, took {warm:.2f}s. The "
        f"ratio above can be satisfied by a machine where cold and warm are both "
        f"slow; this is the absolute half of the same claim.")
