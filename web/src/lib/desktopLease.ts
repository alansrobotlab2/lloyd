/** The Desktop tab's lease countdown, e.g. "2m 05s". */
export function formatRemaining(s?: number | null): string {
  if (s == null) return ''
  const m = Math.floor(s / 60)
  const sec = s % 60
  return m > 0 ? `${m}m ${String(sec).padStart(2, '0')}s` : `${sec}s`
}
