// Half-duplex: mute the mic while Lloyd's voice is playing. Per device,
// persisted like the mic gain, because whether it is needed depends on the
// device's speakers, not on the session.
//
// Off by default since 2026-09-24. It existed to stop Lloyd's voice coming
// back through the mic, but his voice arrives as a LiveKit WebRTC track played
// through an <audio> element — the one playback path Chrome's AEC3 takes as
// its echo reference — and `echoCancellation` is on. With it on, barge-in is
// impossible: the mic is off exactly while he talks. The worker now guards the
// rest itself (sustained speech, a warm-up, pause-then-decide, and his own
// voice enrolled and rejected). Turn it back on for a device whose speakers
// leak past AEC, e.g. a laptop at full volume. `?half_duplex=1|0` sets it.

const STORAGE_KEY = 'lloyd.voice.half_duplex'

const subscribers = new Set<(on: boolean) => void>()
let cached: boolean | null = null

function readFromStorage(): boolean {
  if (typeof window === 'undefined') return false
  try {
    const q = new URL(window.location.href).searchParams.get('half_duplex')
    if (q === '1' || q === '0') {
      window.localStorage.setItem(STORAGE_KEY, q)
      return q === '1'
    }
    return window.localStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

export function getHalfDuplex(): boolean {
  if (cached === null) cached = readFromStorage()
  return cached
}

export function setHalfDuplex(on: boolean): void {
  cached = on
  try {
    window.localStorage.setItem(STORAGE_KEY, on ? '1' : '0')
  } catch {
    /* private mode — keep in-memory only */
  }
  for (const cb of subscribers) {
    try { cb(on) } catch { /* swallow */ }
  }
}

export function subscribeHalfDuplex(cb: (on: boolean) => void): () => void {
  subscribers.add(cb)
  return () => { subscribers.delete(cb) }
}
