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
from pathlib import Path

import pytest

from app.routers import dashboard as dash
from scripts.automod import backlog as B

BOARD_DIR = Path.home() / "obsidian" / "backlog"

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


@pytest.mark.skipif(not BOARD_DIR.is_dir(),
                    reason=f"no board at {BOARD_DIR}: this test measures the real "
                           f"board and a stand-in would measure nothing")
async def test_one_cold_dashboard_cycle_beats_the_budget():
    files = [p for p in BOARD_DIR.glob("*.md") if p.name[:1].isdigit()]
    print(f"board {BOARD_DIR}: {len(files)} item files")
    assert files, (
        f"{BOARD_DIR} exists but matched no NN-*.md item files, so the cycle "
        f"below would walk nothing and time nothing. Zero files is not a fast "
        f"board, it is no board."
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


@pytest.mark.skipif(not BOARD_DIR.is_dir(),
                    reason=f"no board at {BOARD_DIR}")
async def test_the_warm_cycle_is_not_what_the_cold_test_measures():
    """The cold bound is only meaningful against a warm baseline ~100x cheaper.

    If warm and cold ever read the same, the cache is dead and the cold test
    above has been timing the cache-miss path forever without saying so.
    """
    files = [p for p in BOARD_DIR.glob("*.md") if p.name[:1].isdigit()]
    assert files, f"{BOARD_DIR} matched no item files"
    await _cold_cycle()
    started = time.perf_counter()
    await _cold_cycle()
    warm = time.perf_counter() - started
    print(f"warm cycle: {warm:.3f}s")
    assert warm < 1.0, (
        f"the second cycle, with every section cache warm, took {warm:.2f}s. A "
        f"warm cycle costs milliseconds of dict lookups; anything near a second "
        f"means a section is recomputing on every poll regardless of TTL, which "
        f"is the 2 s spike in #1204 section 5."
    )
