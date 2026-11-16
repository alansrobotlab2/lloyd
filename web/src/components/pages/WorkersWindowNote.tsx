/**
 * One line saying how wide the Background > Sources rollup actually is (#4108).
 *
 * `SourcesPanel` asks `/api/workers/health?days=7` and every source card prints its
 * own `total`/`ok`/`failed`/`gpu_hours` under a heading that names no span. `days` is
 * a REQUEST, and the route clamps it: when `workers.db` holds less history than the
 * window asks for, the payload it returns spans only `window_clamped_to_hours`. The
 * pair — that figure and `oldest_input`, the stamp of the oldest run row the verdict
 * read — is what #1401 put on `/api/autonomy/health` and #2127 copied here, for the
 * reason both routes state: a rate printed under a 7-day label is read as a 7-day
 * rate, and the label was never a measurement. The autonomy route's client has
 * carried the pair since (#`AutonomyHealth.fleet`); the workers client did not, so
 * the panel could not render what it had already thrown away.
 *
 * The pair is a FLEET bound, not a per-source measurement — `oldest_input` is
 * `MIN(completed_at)` over EVERY source (`app/routers/workers.py`, #2127) — so this
 * note belongs above the cards and never inside one: a young source whose own history
 * starts days after the store's oldest row would be printed as covered when it is not.
 *
 * Extracted for the reason `SourceHealthRow` is (`BackgroundPage.test.ts`'s header):
 * a phrase computed inside the panel's JSX can only be tested by mounting the panel,
 * and this repo's test env is `environment: "node"` with no DOM package installable.
 * Here the wording is one exported pure function, asserted on its return value, and
 * the component prints exactly what it returns.
 */

/** The requested window as a label: `7d`, `90d`. */
function dayLabel(days: number): string {
  return `${days}d`
}

/** The UTC date half of an ISO stamp — a full microsecond stamp in a 10 px span is
 *  unreadable, and the day is the unit the span is being judged in. `''` when the
 *  payload sent nothing parseable, which costs the date clause and not the hours. */
function oldestDay(stamp: string | null | undefined): string {
  if (!stamp) return ''
  const d = new Date(stamp)
  return Number.isNaN(d.getTime()) ? '' : d.toISOString().slice(0, 10)
}

/** The span phrase, or `''` when there is no span claim to make.
 *
 * `clampHours == null` and not `!clampHours`: `_health_window_fields` emits `0.0` as a
 * real clamp for a window that held no row reaching the tallies, so a falsy test would
 * print nothing for exactly the payload with the least evidence behind it. `days` is
 * the request the route echoed, so the phrase always names the label the reader is
 * already looking under. */
export function windowSpanText(
  clampHours: number | null | undefined,
  days: number,
  oldestInput?: string | null,
): string {
  if (clampHours == null) return ''
  const hours = Math.round(clampHours * 10) / 10
  const oldest = oldestDay(oldestInput)
  const base = `Sources rollup covers ${hours}h of the ${dayLabel(days)} window`
  return oldest ? `${base} (oldest input ${oldest})` : base
}

/** The note itself: nothing at all when the payload covers its window, one muted line
 *  when it does not. */
export function WorkersWindowNote({
  clampHours, days, oldestInput,
}: {
  clampHours: number | null | undefined
  days: number
  oldestInput?: string | null
}) {
  const text = windowSpanText(clampHours, days, oldestInput)
  if (text === '') return null
  return (
    <div className="rounded border border-border px-3 py-1.5 text-[10px] text-muted-foreground">
      {text}
    </div>
  )
}
