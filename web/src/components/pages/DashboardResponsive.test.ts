// Pins the class contract that keeps dashboard panels inside their grid track
// on a phone (#1685). These two assertions are source-text pins, not
// measurements — nothing here renders a pixel. The measured evidence lives in
// scripts/maintenance/dashboard_mobile_probe.py, which drives headless chromium
// at four phone widths and exits non-zero on overflow; before the fix that probe
// reported the Automation & work section overflowing its 288 px column by 360 px
// at 320 px, and Lloyd agent by 151 px.
//
// `?raw` rather than `node:fs`: this project's tsconfig ships no node types, so
// `node:fs` does not type-check here (same reason as AutonomyPage.test.ts and
// runSessions.test.ts), and a test that cannot compile pins nothing.
import { describe, expect, it } from 'vitest'
import pageSource from './DashboardPage.tsx?raw'

describe('dashboard mobile sizing (#1685)', () => {
  it('Panel floors its grid item at zero, not at its content', () => {
    const panel = pageSource.match(/function Panel\([\s\S]*?\n\}/)?.[0] ?? ''
    // Asserted on the root div's own class list, not anywhere in the function:
    // `min-w-0` on a child element would satisfy a loose contains() and still
    // leave every grid item floored at its content.
    const root = panel.match(/cn\(\s*'([^']*)'/)?.[1] ?? ''
    expect(root.split(/\s+/)).toContain('min-w-0')
  })

  it('worker stat row takes two columns on a phone, four from sm up', () => {
    expect(pageSource).toContain('grid grid-cols-2 gap-3 sm:grid-cols-4')
    expect(pageSource).not.toMatch(/className="grid grid-cols-4 gap-3"/)
  })
})
