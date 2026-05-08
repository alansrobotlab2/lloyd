// Per-client microphone gain. Persisted to localStorage so each device
// keeps its own tuning. Subscribers are notified synchronously when the
// value changes so the live VoiceRoom GainNode can update without a
// republish.

const STORAGE_KEY = 'lloyd.mic.gain'
const MIN_GAIN = 0
const MAX_GAIN = 4.0
const DEFAULT_GAIN = 1.0

const subscribers = new Set<(g: number) => void>()
let cached: number | null = null

function readFromStorage(): number {
  if (typeof window === 'undefined') return DEFAULT_GAIN
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (raw === null) return DEFAULT_GAIN
    const v = parseFloat(raw)
    if (!Number.isFinite(v) || v < MIN_GAIN) return DEFAULT_GAIN
    return Math.min(MAX_GAIN, v)
  } catch {
    return DEFAULT_GAIN
  }
}

export function getMicGain(): number {
  if (cached === null) cached = readFromStorage()
  return cached
}

export function setMicGain(g: number): void {
  const clamped = Math.max(MIN_GAIN, Math.min(MAX_GAIN, g))
  cached = clamped
  try {
    window.localStorage.setItem(STORAGE_KEY, String(clamped))
  } catch {
    /* private mode / disabled storage — keep in-memory only */
  }
  for (const cb of subscribers) {
    try { cb(clamped) } catch { /* swallow */ }
  }
}

export function subscribeMicGain(cb: (g: number) => void): () => void {
  subscribers.add(cb)
  return () => { subscribers.delete(cb) }
}

export function gainToDb(g: number): number {
  if (g <= 0) return -Infinity
  return 20 * Math.log10(g)
}

export const MIC_GAIN_RANGE = { min: MIN_GAIN, max: MAX_GAIN, default: DEFAULT_GAIN }
