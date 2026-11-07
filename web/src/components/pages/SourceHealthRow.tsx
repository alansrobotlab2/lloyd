/**
 * One source's outcome rollup, as the Background > Sources panel prints it.
 *
 * Extracted from `SourceCard` for the reason `RunsToolbar` is a component of its own:
 * the rose on a 50 %+ failure rate is a DECISION, and a decision written inline in a
 * page's JSX can only be tested by mounting the page — which no test in this repo can
 * do (`vitest.config.ts` pins `environment: "node"`, no DOM package is installable
 * because `web/package.json` is a denied path for a round, and `api.ts` reads
 * `import.meta.env` at module scope; the whole argument is in the header of
 * `BackgroundPage.test.ts`). Here the predicate is exported beside the markup that
 * consumes it, so a seeded `health` block renders on its own.
 *
 * The decision is #3640's. `fail_rate` is a ratio over a window and says nothing about
 * where inside that window its numerator sits: read 2026-10-28, youtube-digest's row
 * was 51 % over 100 runs whose every failure predated a landed revert by five days,
 * and #3523 was filed at `high` off exactly that payload. The queue has since put the
 * tail beside the rate — `last_failure` and `consecutive_failures` (#3527) — so the
 * panel can tell a source failing now from a burst that closed, which is the
 * difference between an alarm and a five-day-old fact.
 */
import { cn } from '@/lib/utils'
import { relative } from '@/lib/relativeTime'
import type { WorkerSourceHealth } from '../../api'

export type SourceHealth = NonNullable<WorkerSourceHealth['health']>

/** At or above this a rate is loud — but only while its own tail is still failing. */
export const ALARM_FAIL_RATE = 0.5

/** Is this source failing NOW, as opposed to having failed earlier in the window?
 *
 * `consecutive_failures` counts the source's most recent runs that were failures in a
 * row, so 0 means its newest run succeeded and the failures are behind it. A missing
 * value — `undefined` or `null` — is a backend older than `b22a1b3c`, which cannot
 * answer the question at all, and unknown is treated as live: the rose is this row's
 * only alarm, and disarming it for every backend that cannot prove its tail is clean
 * would trade a false alarm for a silent one. */
export function failureTailIsLive(h: SourceHealth): boolean {
  return h.consecutive_failures == null || h.consecutive_failures > 0
}

/** Should the rate be printed in the alarm colour? Both halves are required: a rate
 * that is merely high is history, and a live tail under 50 % is ordinary noise. */
/** The per-kind split as one line, `''` when there is nothing to split (#3906).
 *
 * `by_kind` is #3526's partition of the same window by `runs.kind`: one entry per kind,
 * each with that kind's own `total`/`failed`/`fail_rate`, the totals summing to the
 * source's `total`. It prints only where a source actually has halves: a backend older
 * than `64ff6c01` sends no key, an empty `{}` says the same thing, and ONE entry is that
 * source's own figures restated back at it — the sum is the total — so printing it would
 * put the same number on every row twice. `youtube-digest` is the one source the endpoint
 * reports with two kinds today; the rest print nothing. Kinds print in name order so two
 * reads of one window print one line.
 *
 * Context beside the rate, never the alarm: the decision above reads the source's own
 * blended `fail_rate`, and no kind's number is allowed to colour the row. A kind can be 1
 * failed of 1 — 100 % — on a single row, and alarming on that would report one run as a
 * source on fire, which is the reading the blend exists to prevent. */
export function kindSplitText(h: SourceHealth): string {
  const kinds = Object.entries(h.by_kind ?? {}).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
  if (kinds.length < 2) return ''
  return kinds
    .map(([kind, v]) => {
      const rate = v.fail_rate === null ? 'n/a' : `${Math.round(v.fail_rate * 100)}% failing`
      return `${kind}: ${v.total} run${v.total === 1 ? '' : 's'}, ${v.failed} failed, ${rate}`
    })
    .join(' · ')
}

export function rateIsAlarming(h: SourceHealth): boolean {
  return h.fail_rate !== null && h.fail_rate >= ALARM_FAIL_RATE && failureTailIsLive(h)
}

/** The failure's age, worded to sit beside the rate: `last failure 2d ago`, plus the
 * tail's streak when there is one. `''` when the window holds no failure (or the
 * stamp will not parse), which is the caller's cue to print nothing rather than a
 * label with no age after it. */
export function failureAgeText(h: SourceHealth): string {
  const age = relative(h.last_failure)
  if (!age) return ''
  const streak = h.consecutive_failures ?? 0
  return streak > 0 ? `last failure ${age} · ${streak} in a row` : `last failure ${age}`
}

export function SourceHealthRow({ h }: { h: SourceHealth }) {
  const alarming = rateIsAlarming(h)
  const age = failureAgeText(h)
  const kindSplit = kindSplitText(h)
  return (
    <>
      <span>{h.total} run{h.total === 1 ? '' : 's'}</span>
      {/* #1857's split, beside the run count it is a share of — 2 of 40
          and 2 of 2 are opposite readings. Only the deadline cut is the
          alarm; a shrunk matrix is the projection doing its job. */}
      {(h.deadline_cut ?? 0) > 0 && (
        <span className="text-rose-400">{h.deadline_cut} deadline-cut</span>
      )}
      {(h.matrix_shrunk ?? 0) > 0 && (
        <span>{h.matrix_shrunk} matrix-shrunk</span>
      )}
      <span className="text-emerald-400">{h.ok} ok</span>
      {h.failed > 0 && <span className="text-rose-400">{h.failed} failed</span>}
      {h.skipped > 0 && <span>{h.skipped} skipped</span>}
      <span className={cn(alarming && 'text-rose-400')}>
        {h.fail_rate === null ? '—' : `${Math.round(h.fail_rate * 100)}% failing`}
      </span>
      {/* #3640: the age of that rate's failure population, immediately beside it.
          Muted ink for a closed burst — the failures happened, they are not
          happening — and the alarm colour only when the rate itself is alarming,
          so the two rose marks are one reading and not two coincidences. */}
      {age !== '' && (
        <span className={cn(alarming && 'text-rose-400')}>{age}</span>
      )}
      {/* #3906: the by_kind split beside the blended rate it is a partition of, in muted
          ink and never in the alarm colour — which half of a two-kind source failed is
          context for the reader, not a second threshold. Absent when the payload has no
          `by_kind`, when its one kind is the whole source, or when it is empty. */}
      {kindSplit !== '' && <span className="text-zinc-500">{kindSplit}</span>}
      <span>{h.gpu_hours}h</span>
      {h.last_completed && <span>last {relative(h.last_completed)}</span>}
    </>
  )
}
