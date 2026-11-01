// Does a /api/memory/stats response need a second fetch, and how long after?
//
// Extracted from MemoryPage's mount effect so the decision is testable: the page has
// no test harness of its own, and the sibling pattern the gate runs is a pure helper
// plus a vitest node next to it (`backlogBoardDefault.ts` + `.test.ts`,
// `autonomyHealth.ts` + `.test.ts` — `vitest.config.ts` includes
// `components/pages/*.test.ts` beside the page globs).
//
// Why the question exists. `lastRefresh` is the tag tally's completion time and is
// `""` while no build has finished (app/routers/memory.py `_empty_tally`). On a freshly
// started backend the first request to the Memory tab is that case: it answers the
// document counts immediately and starts the build behind the response, so the tab
// renders `0 tags` — the one number on that page that is not live. Before #3682 a
// manual reload fixed it; after the floor a stale tally inside the floor starts nothing
// at all, so the viewer's only route to the real number is a fetch the page has to make
// itself. `stats.lastRefresh` is read nowhere in the UI today (only declared at
// web/src/api.ts), which is exactly why the tab cannot notice it is showing zeros.
//
// The delay is the cold build's own cost, measured 2026-10-27: 21.7s in the live
// backend, 17.6s in a plain out-of-process parse of the same 8,727-note vault. The wait
// has to clear the SLOWER of the two — the retry exists to outlast the parse, so at
// exactly 20s a re-fetch on the live box would routinely arrive while the build it is
// waiting for is still running and come back `lastRefresh: ""` again, which this helper
// deliberately does not re-schedule: the retry would be spent and the tab still at
// `0 tags`. Re-fetch sooner and the second response is empty for the same reason the
// first was, and the vault has been parsed twice for nothing.
// One delayed fetch, never a poll: an interval is the thing #3682 exists to stop.

import { api, type MemoryStats } from "../../api";

/** 22s: above the 21.7s live cold build (17.6s out of process), measured 2026-10-27. */
export const STATS_REFETCH_DELAY_MS = 22_000;

/** An empty `lastRefresh` is "no tally has completed", not "a tally with no time". */
export function tallyHasNeverCompleted(stats: Pick<MemoryStats, "lastRefresh"> | null | undefined): boolean {
  return !stats?.lastRefresh;
}

/**
 * Schedule the one re-fetch, or nothing.
 *
 * `schedule` is the caller's `setTimeout` and `apply` its `setStats`, both injected so
 * the nodes can read the delay it was handed, invoke or discard the callback, and see
 * what the response went to. The return value is whatever `schedule` returns, which is
 * the handle MemoryPage keeps so that an unmount clears a pending fetch. The fetch
 * itself fires only when the callback runs — returning here has not asked the server
 * for anything.
 *
 * `apply` is a parameter rather than an ignored `void` on the call because the point of
 * the re-fetch is the number on screen: a fetch whose response is dropped leaves the
 * tab exactly where it was, at `0 tags`, twenty seconds later and one vault parse
 * poorer. A re-fetch that lands empty again (a slow build) or fails just leaves the
 * header as drawn — this is one retry, not a poller.
 */
export function scheduleStatsRefetch<T>(
  stats: Pick<MemoryStats, "lastRefresh"> | null | undefined,
  schedule: (cb: () => void, ms: number) => T,
  apply: (fresh: MemoryStats) => void,
): T | null {
  if (!tallyHasNeverCompleted(stats)) return null;
  return schedule(() => {
    api.memoryStats().then(apply, () => {
      // Still no tally, or the endpoint is down: the header keeps what it shows. The
      // next mount asks again, which is the outer bound on how long a stale tally can
      // sit there from this side.
    });
  }, STATS_REFETCH_DELAY_MS);
}
