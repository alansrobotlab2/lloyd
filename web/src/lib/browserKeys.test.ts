import { describe, expect, it } from 'vitest'
import { playwrightKey } from './browserKeys'

const k = (key: string, mods: Partial<{ ctrl: boolean; meta: boolean; alt: boolean; shift: boolean }> = {}) => ({
  key,
  ctrlKey: !!mods.ctrl,
  metaKey: !!mods.meta,
  altKey: !!mods.alt,
  shiftKey: !!mods.shift,
})

describe('playwrightKey', () => {
  it('names the keys a login form needs', () => {
    expect(playwrightKey(k('Enter'))).toBe('Enter')
    expect(playwrightKey(k('Tab'))).toBe('Tab')
    expect(playwrightKey(k('Tab', { shift: true }))).toBe('Shift+Tab')
    expect(playwrightKey(k('Backspace'))).toBe('Backspace')
  })

  it('leaves a printable key to the text batch', () => {
    expect(playwrightKey(k('a'))).toBeNull()
    expect(playwrightKey(k('A', { shift: true }))).toBeNull()
    expect(playwrightKey(k(' '))).toBeNull()
  })

  it('sends Ctrl and Cmd shortcuts as ControlOrMeta', () => {
    expect(playwrightKey(k('a', { ctrl: true }))).toBe('ControlOrMeta+a')
    expect(playwrightKey(k('A', { meta: true, shift: true }))).toBe('ControlOrMeta+Shift+a')
  })

  it('ignores keys Playwright would not know', () => {
    expect(playwrightKey(k('Shift', { shift: true }))).toBeNull()
    expect(playwrightKey(k('Dead'))).toBeNull()
    expect(playwrightKey(k('Unidentified'))).toBeNull()
  })
})
