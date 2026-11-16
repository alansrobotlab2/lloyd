// The Browser tab's keyboard → Playwright key names (`page.keyboard.press`).
//
// A printable key without Ctrl/Cmd is not sent by name: the tab batches those
// into a `type` event. Everything here is a key Playwright knows by name, with
// modifiers in the order it accepts. Cmd and Ctrl both become `ControlOrMeta`,
// so Cmd+A on a Mac selects all in the agent's Linux Chromium.

const NAMED_KEYS = new Set([
  'Enter', 'Tab', 'Backspace', 'Delete', 'Escape', 'Insert',
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
  'Home', 'End', 'PageUp', 'PageDown',
  'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
])

export interface KeyLike {
  key: string
  ctrlKey: boolean
  metaKey: boolean
  altKey: boolean
  shiftKey: boolean
}

/** Playwright's name for a keydown, or null when it is not one we send. */
export function playwrightKey(e: KeyLike): string | null {
  const mods: string[] = []
  if (e.ctrlKey || e.metaKey) mods.push('ControlOrMeta')
  if (e.altKey) mods.push('Alt')
  if (e.shiftKey) mods.push('Shift')
  if (NAMED_KEYS.has(e.key)) return [...mods, e.key].join('+')
  if (e.key.length === 1 && (e.ctrlKey || e.metaKey)) {
    return [...mods, e.key === ' ' ? 'Space' : e.key.toLowerCase()].join('+')
  }
  return null
}
