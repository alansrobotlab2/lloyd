// Pins the class contract that keeps a dashboard row's only label reachable on
// a phone (#1742, re-landed as #2202). Like the two #1685 assertions below it,
// these are source-text pins, not measurements — nothing here renders a pixel.
// The measured evidence lives in scripts/maintenance/dashboard_mobile_probe.py,
// driven from tests/test_dashboard_responsive.py, which loads the dashboard
// seeded with deliberately long row labels at phone widths and asks, per label,
// whether it is clipped with nothing to get the full text back.
//
// The shape being pinned is a wrapping base plus a viewport-conditional ellipsis:
// `whitespace-normal break-words` at the base and `sm:truncate` from `sm` up.
// Both halves are load-bearing, and a reviewer reading only the class list
// cannot tell which one does what, so:
//
//   * `truncate` is `overflow:hidden; text-overflow:ellipsis; white-space:nowrap`
//     (verified against this project's Tailwind 4.2.2: node_modules/tailwindcss/
//     dist/lib.js registers it as exactly those three declarations — and
//     node_modules/tailwindcss/utilities.css is a single line, so it carries no
//     per-utility line to cite). The `white-space:nowrap` is what makes one long
//     task name a single clipped line, and it ships NO `overflow-wrap`, so
//     `break-words` is not implied by anything here — it has to be present.
//   * `whitespace-normal` removes that nowrap; `break-words` is
//     `overflow-wrap:break-word`, which is what lets an unbreakable run like
//     `scheduled-task:autonomy-self-improvement-pipeline-stage-two` split inside
//     a 232 px column instead of overflowing it.
//   * `sm:truncate` puts the ellipsis back from 640 px up, which is what keeps
//     the desktop dashboard's one-line rows one line. A bare `truncate` would
//     clip on a phone too and undo the fix.
//
// `?raw` rather than `node:fs`: this project's tsconfig ships no node types, so
// `node:fs` does not type-check here (same reason as AutonomyPage.test.ts and
// runSessions.test.ts), and a test that cannot compile pins nothing.
import { describe, expect, it } from 'vitest'
import pageSource from './DashboardPage.tsx?raw'

// The seven spans that are their row's ONLY text — the ones #1742 named, keyed
// by the interpolation each renders rather than by its class list, because the
// class list is the thing under test and a key made of it would move with the
// regression it is supposed to catch. `count` is how many such spans the page
// has: `{t.name}` renders in two of them (the Failed row and Backlog
// Recently-touched), so a per-key count is what stops a new `{t.name}` span
// slipping in uninspected, and the total is what stops a seventh becoming an
// eighth without a say.
const LABEL_SPANS: { key: string; count: number }[] = [
  { key: '{task.name}', count: 1 },            // TaskLine (upcoming / held / overdue)
  { key: '{r.kind || r.job_id}', count: 1 },   // Running now
  { key: '{t.name}', count: 2 },               // Failed row + Backlog Recently-touched
  { key: '{src.name}', count: 1 },             // Worker pool: sources
  { key: '{r.source}', count: 1 },              // Worker pool: recent runs
  { key: '{r.summary}', count: 1 },             // Worker pool: recent runs
]
const EXPECTED_SPANS = LABEL_SPANS.reduce((n, s) => n + s.count, 0)

// One span's own opening tag, ending at the interpolation it renders. The
// attribute group is `[^>]` so it cannot cross into the element's text, and the
// whole thing is anchored on the `>{key}</span>` tail so a `title=` written
// inside the tag is part of the match rather than an invisible neighbour.
function spanTags(source: string, key: string) {
  const needle = `>${key}</span>`
  const out: { tag: string; start: number }[] = []
  let at = source.indexOf(needle)
  while (at >= 0) {
    // Walk back to the `<` that opens this element. The nearest previous `>` is
    // where the previous tag closed, so the span's own `<` is the last one
    // before that — unless the span directly follows text, in which case the
    // last `>` is an ancestor's opener and this IS the span's `<`.
    const prevClose = source.lastIndexOf('>', at)
    const open = source.lastIndexOf('<', prevClose)
    if (open >= 0) out.push({ tag: source.slice(open, at + needle.length), start: open })
    at = source.indexOf(needle, at + needle.length)
  }
  return out
}

const indentOf = (source: string, start: number) =>
  (source.slice(source.lastIndexOf('\n', start) + 1, start).match(/^\s*/)?.[0] ?? '').length

/** Every way one of these spans can stop meeting the contract, as messages.
 *
 * A function over a string, not a set of assertions against the imported
 * source, because the contract's job is to REDEN: `it(...)` blocks that read the
 * one real file cannot be pointed at a broken copy of it, and a pin nobody has
 * watched fail is a pin nobody has tested. Each `it` below feeds this the real
 * source — where it must return no messages — and a mutated source built by the
 * test itself, where a specific message must come back. */
function auditLabelSpans(source: string): string[] {
  const problems: string[] = []
  let seen = 0
  for (const { key, count } of LABEL_SPANS) {
    const spans = spanTags(source, key)
    if (spans.length !== count) {
      problems.push(`${key}: expected ${count} label span(s), found ${spans.length}`)
      continue
    }
    seen += spans.length
    for (const { tag, start } of spans) {
      const cls = tag.match(/className="([^"]*)"/)?.[1] ?? ''
      const classes = cls.split(/\s+/)
      const label = `${key} [${cls}]`
      if (!classes.includes('whitespace-normal')) problems.push(`${label}: no whitespace-normal`)
      if (!classes.includes('break-words')) problems.push(`${label}: no break-words`)
      if (!classes.includes('sm:truncate')) problems.push(`${label}: no sm:truncate`)
      // A bare `truncate` is the regression, and also a way of satisfying the
      // three checks above while still clipping: `truncate sm:truncate` would
      // pass every "contains" test and fix nothing.
      if (classes.includes('truncate')) problems.push(`${label}: bare truncate still clips below sm`)
      if (/[\s"]title=/.test(tag)) problems.push(`${label}: title= is a tooltip, not a fix`)
      // An `a`/`button` WRAPPING the row is the other shape that makes a clipped
      // label reachable, so it must not sneak in as a substitute for wrapping.
      // Tag-only, matching the probe's predicate exactly: a `<div role="button">`
      // is not an exemption on either side of the pair. Indent is the nesting
      // proof — an `<a>`/`<button>` at a shallower indent above the span is an
      // ancestor, and one at the same or deeper indent is a sibling row.
      const before = source.slice(0, start)
      const lines = before.split('\n')
      const ownLine = lines.pop() ?? ''
      const myIndent = indentOf(source, start)
      // Same-line first: `<button><span>{x}</span></button>` is a wrapper too, and
      // has no shallower line above it to find. Everything on the span's own line
      // to the LEFT of the span is what it is nested inside on that line.
      // (`before` ends at the span's `<`, so `ownLine` is exactly the text on the
      // span's line to the left of the span.)
      if (/<a[\s>]/.test(ownLine) || /<button[\s>]/.test(ownLine)) {
        problems.push(`${label}: wrapped in a link/button on its own line`)
      }
      for (let i = lines.length - 1; i >= 0 && lines.length - i <= 8; i--) {
        const line = lines[i]
        if (!line.trim()) continue
        const indent = (line.match(/^\s*/)?.[0] ?? '').length
        if (indent >= myIndent) continue
        if (/<a[\s>]/.test(line) || /<button[\s>]/.test(line)) {
          problems.push(`${label}: wrapped in a link/button above :${i + 1}`)
          break
        }
      }
    }
  }
  if (seen !== EXPECTED_SPANS) {
    problems.push(`label spans: expected ${EXPECTED_SPANS} in all, found ${seen}`)
  }
  return problems
}

describe('dashboard row labels wrap on a phone (#1742, re-landed as #2202)', () => {
  it('all seven label spans carry a wrapping base and keep the ellipsis only from sm up', () => {
    expect(EXPECTED_SPANS).toBe(7)
    expect(auditLabelSpans(pageSource)).toEqual([])
  })

  it('the contract reddens when one span reverts to a bare truncate', () => {
    // The exact regression this round is re-landing against: #1742's fix was
    // written, review refused it, the round was abandoned, and every span went
    // back to a single clipped line. Mutation (a): the file cannot silently
    // return to that shape and keep a green suite.
    const reverted = pageSource.replace(
      '<span className="whitespace-normal break-words sm:truncate text-foreground">{task.name}</span>',
      '<span className="truncate text-foreground">{task.name}</span>',
    )
    expect(reverted).not.toBe(pageSource)
    const problems = auditLabelSpans(reverted)
    expect(problems.length).toBeGreaterThan(0)
    expect(problems.join('\n')).toContain('{task.name}')
    expect(problems.join('\n')).toContain('bare truncate')
  })

  it('the contract reddens when a label trades its wrap for a title attribute', () => {
    // Mutation (b): the cheapest wrong fix — keep one clipped line and hang a
    // tooltip on it. A `title` is a hover affordance, and a phone has no hover:
    // #1742 was refused once for exactly this substitution.
    const titled = pageSource.replace(
      '<span className="whitespace-normal break-words sm:truncate text-foreground">{task.name}</span>',
      '<span title={task.name} className="whitespace-normal break-words sm:truncate text-foreground">{task.name}</span>',
    )
    expect(titled).not.toBe(pageSource)
    expect(auditLabelSpans(titled).join('\n')).toContain('title= is a tooltip, not a fix')
  })

  it('the contract reddens when a label is wrapped in a button instead of wrapped in text', () => {
    // Mutation (c): the second wrong fix — make the row a `<button>` so the
    // clipped text is "reachable" by tapping through. That is a different
    // affordance than the one the page's rows have, and the probe's predicate
    // exempts it, so a wrapper here would turn the measured pin green while the
    // label stayed clipped.
    // Indentation matters here and is the point: the span moves to 8 spaces and
    // its new `<button>` parent to 6, which is how a wrapper is actually written
    // in this file, and shallower-than is the nesting proof the audit looks for.
    // A wrapper at the same indent as its child would be a formatting accident
    // the audit deliberately does not claim to read.
    const wrapped = pageSource.replace(
      '<span className="whitespace-normal break-words sm:truncate text-foreground">{task.name}</span>',
      '<button onClick={() => {}}>\n        <span className="whitespace-normal break-words sm:truncate text-foreground">{task.name}</span>\n      </button>',
    )
    expect(wrapped).not.toBe(pageSource)
    expect(auditLabelSpans(wrapped).join('\n')).toContain('wrapped in a link/button')
  })

  it('a label span appearing a second time over the wrong key is caught by count', () => {
    // Mutation (d): the denominator. A new row that interpolates `{src.name}`
    // with the old clipping class must not inherit the pass the five existing
    // spans earned.
    const extra = pageSource.replace(
      '<span className="whitespace-normal break-words sm:truncate text-muted-foreground">{r.source}</span>',
      '<span className="truncate text-muted-foreground">{src.name}</span>\n              <span className="whitespace-normal break-words sm:truncate text-muted-foreground">{r.source}</span>',
    )
    expect(extra).not.toBe(pageSource)
    expect(auditLabelSpans(extra).join('\n')).toContain('{src.name}: expected 1 label span(s), found 2')
  })
})

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
