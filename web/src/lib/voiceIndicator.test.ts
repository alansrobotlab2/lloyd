import { describe, expect, it } from 'vitest'
import { micErrorMessage, voiceIndicator, formatElapsed } from './voiceIndicator'

const base = {
  status: 'connected' as const,
  micState: 'live' as const,
  agentThinking: false,
  agentSpeaking: false,
  wakeState: 'idle' as const,
}

describe('voiceIndicator', () => {
  it('never tells you to say Lloyd while the mic is not live', () => {
    expect(voiceIndicator({ ...base, micState: 'starting' })).toBe('mic-starting')
    expect(voiceIndicator({ ...base, micState: 'failed' })).toBe('mic-failed')
    expect(voiceIndicator({ ...base, micState: 'failed', wakeState: 'listening' })).toBe('mic-failed')
  })

  it('says idle and listening only with a live mic', () => {
    expect(voiceIndicator(base)).toBe('idle')
    expect(voiceIndicator({ ...base, wakeState: 'listening' })).toBe('listening')
  })

  it("puts Lloyd's own activity above the mic, and disconnection above all", () => {
    expect(voiceIndicator({ ...base, micState: 'failed', agentSpeaking: true })).toBe('speaking')
    expect(voiceIndicator({ ...base, micState: 'failed', agentThinking: true })).toBe('thinking')
    expect(voiceIndicator({ ...base, status: 'connecting', agentSpeaking: true })).toBe('disconnected')
  })

  it('treats a listen-only room (no mic requested) like a live one', () => {
    expect(voiceIndicator({ ...base, micState: 'off' })).toBe('idle')
  })
})

describe('micErrorMessage', () => {
  it('names the fix for the common getUserMedia failures', () => {
    const err = (name: string) => Object.assign(new Error('x'), { name })
    expect(micErrorMessage(err('NotAllowedError'))).toMatch(/permission denied/)
    expect(micErrorMessage(err('NotFoundError'))).toMatch(/No microphone/)
    expect(micErrorMessage(err('NotReadableError'))).toMatch(/busy/)
  })

  it('falls back to the raw message for anything else', () => {
    expect(micErrorMessage(new Error('publish timed out'))).toBe('Microphone failed: publish timed out')
    expect(micErrorMessage('odd')).toBe('Microphone failed: odd')
  })
})

describe('conversation mode', () => {
  const base = {
    status: 'connected' as const,
    micState: 'live' as const,
    agentThinking: false,
    agentSpeaking: false,
    wakeState: 'listening' as const,
  }

  it('an open conversation reads as one, not as the short window', () => {
    expect(voiceIndicator({ ...base, wakeMode: 'conversation' })).toBe('conversation')
    expect(voiceIndicator({ ...base, wakeMode: 'listening' })).toBe('listening')
    expect(voiceIndicator(base)).toBe('listening')
  })

  it('Lloyd talking and a dead mic still outrank it', () => {
    expect(voiceIndicator({ ...base, wakeMode: 'conversation', agentSpeaking: true })).toBe('speaking')
    expect(voiceIndicator({ ...base, wakeMode: 'conversation', micState: 'failed' })).toBe('mic-failed')
  })

  it('formats elapsed time the way the pill shows it', () => {
    expect(formatElapsed(0)).toBe('0:00')
    expect(formatElapsed(72.9)).toBe('1:12')
    expect(formatElapsed(605)).toBe('10:05')
  })
})
