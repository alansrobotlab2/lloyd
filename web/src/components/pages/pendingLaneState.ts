/** The Review tab's read of the staged pile, as pure functions (#3548).
 *
 * Why this file exists rather than the logic living in `WorkersPage.tsx`: vitest runs
 * with `environment: "node"` (`web/vitest.config.ts`) and `web/package.json` is a
 * denied path for an automod round, so no DOM backend can be installed and the tab
 * cannot be mounted in a test — see the note at the head of `BackgroundPage.test.ts`,
 * which splits the same way with `runsPanelState.ts`. Everything that decides WHAT the
 * reviewer sees is here and is therefore pinned; only the JSX that renders it is not.
 *
 * Both numbers are load-bearing. `pending` is `workers.sources._common.staged_pending_count`
 * — the figure `/api/workers/health` publishes and the figure a source's
 * `max_staged_pending` bound is compared against, over which da5c43de halts enqueueing.
 * `returned` is the length of the list the server actually handed back. Printing the
 * second as though it were the first is the defect: the tab said
 * `all sources (300)` while health said 325 pending over a 150 cap, so an operator
 * looking for the pile the producer is waiting on was told a number that was neither.
 */

/** The ceiling `GET /api/workers/pending` accepts (`MAX_PENDING_LIMIT`), asked for in
 * full. The route's own cap is not what used to bind — the tab's 300 was: at 647 staged
 * notes a 300-row window reached 167 of the 325 pending and dropped 158 back to
 * 2026-09-22, and since da5c43de a pending note nobody can reach is not a backlog but a
 * halt, because nothing lowers the count except reviewing it. */
export const PENDING_LANE_LIMIT = 1000

/** Asked for by name rather than left to a default, so the tab says out loud that it is
 * reading the pile the bound counts and not the whole staging tree. */
export const PENDING_LANE_STATUS = 'pending'

export interface PendingCounts {
  /** `pending_count` off the envelope: the whole pile, wherever in the order it sits. */
  pending?: number
  /** `items.length` — how many notes THIS response carried. */
  returned: number
  /** `dropped_by_limit`: matching notes the cap cut. */
  dropped?: number
  /** `cap`: the ceiling that cut them. */
  cap?: number
}

/** The query `ReviewPanel` sends for its own list.
 *
 * A blank filter means ALL sources, so it must not reach the query at all — the route
 * compares `source` against a directory name, and `?source=%20%20%20` would match no
 * directory and answer an empty list over a pile the label has just said is 325 deep.
 * `trim()` because the all-sources control reaches this as `""`, and a stray space from
 * a future filter control must land on the same branch, not on a no-match query. */
export function pendingLaneParams(source: string): {
  source?: string
  limit: number
  status: string
} {
  const trimmed = source.trim()
  return {
    ...(trimmed ? { source: trimmed } : {}),
    limit: PENDING_LANE_LIMIT,
    status: PENDING_LANE_STATUS,
  }
}

/** `bench-mine — 220 pending, showing 220`, or with the cap named when it cut any.
 *
 * The dropped tail is spelled out rather than left to arithmetic because a truncated
 * list otherwise reads exactly like a cleared pile, and the difference is whether an
 * operator keeps reviewing. When the server predates the field the label says only what
 * it knows — `showing M` — rather than printing `returned` in the pending slot, which
 * is the sentence that got this written. */
export function pendingLaneLabel(counts: PendingCounts, scope = 'all sources'): string {
  const shown = `showing ${counts.returned}`
  if (counts.pending === undefined || counts.pending === null) {
    return `${scope} — ${shown} (pending count not reported)`
  }
  const head = `${scope} — ${counts.pending} pending, ${shown}`
  const dropped = counts.dropped ?? 0
  if (dropped <= 0) return head
  const byCap =
    counts.cap === undefined || counts.cap === null
      ? `${dropped} dropped by the cap`
      : `${dropped} dropped by the ${counts.cap}-row cap`
  return `${head} (${byCap})`
}
