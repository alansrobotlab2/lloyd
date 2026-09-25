/**
 * One definition of what the voice status pill and dot say.
 *
 * Both used to derive it separately, and neither knew about the microphone:
 * a browser that joined the room and never published its mic still read
 * "Say 'Lloyd'", which is an instruction that could not work. On 2026-09-18
 * that is what a four-minute session looked like — and the error VoiceRoom
 * had recorded for a failed mic was rendered nowhere at all.
 */

export type MicState = 'off' | 'starting' | 'live' | 'failed'

export type VoiceIndicator =
  | 'disconnected'
  | 'thinking'
  | 'speaking'
  | 'mic-failed'
  | 'mic-starting'
  | 'conversation'
  | 'listening'
  | 'idle'

export function voiceIndicator(s: {
  status: 'idle' | 'connecting' | 'connected' | 'failed'
  micState?: MicState
  agentThinking: boolean
  agentSpeaking: boolean
  wakeState: 'idle' | 'listening'
  /** 'conversation' when the wake word opened a conversation (no wake word
   *  needed until it closes); absent or anything else = the plain window. */
  wakeMode?: 'idle' | 'listening' | 'conversation'
}): VoiceIndicator {
  if (s.status !== 'connected') return 'disconnected'
  // Lloyd's own activity still wins: a typed turn is audible with no mic.
  if (s.agentThinking) return 'thinking'
  if (s.agentSpeaking) return 'speaking'
  // Below that, the mic decides whether "say Lloyd" is a thing you can do.
  if (s.micState === 'failed') return 'mic-failed'
  if (s.micState === 'starting') return 'mic-starting'
  if (s.wakeState === 'listening') {
    return s.wakeMode === 'conversation' ? 'conversation' : 'listening'
  }
  return 'idle'
}

/** "1:12" — how long a conversation has been open. */
export function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

/** How long the mic may take to start before the page says so. A permission
 *  prompt nobody has answered never resolves and never rejects. */
export const MIC_SLOW_MS = 8000

export const MIC_SLOW_MESSAGE =
  "Microphone hasn't started — check the browser's mic permission (address bar)."

/** A getUserMedia / publish failure in words a person can act on. */
export function micErrorMessage(e: unknown): string {
  const name = (e as { name?: string } | null)?.name
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone permission denied — allow it for this site, then reconnect.'
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No microphone found on this device.'
    case 'NotReadableError':
    case 'AbortError':
      return 'The microphone is busy or unavailable (another app may have it).'
  }
  const msg = e instanceof Error ? e.message : String(e)
  return `Microphone failed: ${msg}`
}
