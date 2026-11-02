/**
 * "3d ago" for an ISO stamp, and `''` when there is nothing to date.
 *
 * Lifted out of `BackgroundPage.tsx` byte for byte by #3640, not rewritten: that
 * page still prints an age on every run row, and the per-source health row is now a
 * component of its own (`SourceHealthRow.tsx`) which prints two — a failure's age and
 * its newest completion — so the formatter can no longer live inside one page.
 *
 * The empty return is load-bearing rather than tidy. Every caller here renders its
 * age span only when this comes back non-empty, which is what stops a source with no
 * failure in the window (`last_failure: null`) or an unparseable stamp printing a
 * dangling "last failure ago".
 */
export function relative(iso: string | null | undefined): string {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  if (!Number.isFinite(then)) return ''
  const delta = (Date.now() - then) / 1000
  if (delta < 60) return 'just now'
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`
  return `${Math.floor(delta / 86400)}d ago`
}
