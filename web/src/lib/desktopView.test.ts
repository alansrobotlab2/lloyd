import { describe, expect, it } from 'vitest'
import { toolImageUrl } from '../api'
import { formatRemaining } from './desktopLease'

describe('toolImageUrl', () => {
  it('serves a screenshot from its session spill directory', () => {
    expect(toolImageUrl({ path: '/d/sessions/20260923_101010_ab.tool-results/c1.img0.jpg' }))
      .toBe('/api/sessions/20260923_101010_ab/tool-results/c1.img0.jpg')
  })
  it('an unchanged frame points at the image it repeats', () => {
    expect(toolImageUrl({ path: '/d/sessions/s.tool-results/c2.img0.jpg', deduped_from: 'c1.img0.jpg' }))
      .toBe('/api/sessions/s/tool-results/c1.img0.jpg')
  })
  it('refuses a path outside a spill directory', () => {
    expect(toolImageUrl({ path: '/etc/passwd' })).toBeNull()
  })
})

describe('formatRemaining', () => {
  it('formats the lease clock', () => {
    expect(formatRemaining(125)).toBe('2m 05s')
    expect(formatRemaining(9)).toBe('9s')
    expect(formatRemaining(null)).toBe('')
  })
})
