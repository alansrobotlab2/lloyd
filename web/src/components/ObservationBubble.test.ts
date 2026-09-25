import { describe, expect, it } from 'vitest'
import { isLabelable, nextVerdict } from './ObservationBubble'

describe('observation verdict thumbs', () => {
  it('offers a label only on the levers that changed a turn', () => {
    for (const a of ['inject', 'cancel', 'ambient', 'clarify']) expect(isLabelable(a)).toBe(true)
    for (const a of ['noop', 'noop_budget_exhausted', 'acknowledge_complete']) expect(isLabelable(a)).toBe(false)
  })

  it('clicking the set thumb clears it, the other one replaces it', () => {
    expect(nextVerdict(null, 'up')).toBe('up')
    expect(nextVerdict('up', 'up')).toBe(null)
    expect(nextVerdict('up', 'down')).toBe('down')
    expect(nextVerdict(undefined, 'down')).toBe('down')
  })
})
