// #2648 clause 4, EXECUTED: the Runs panel's bench reveal/hide cycle, the request each
// press causes, and the hidden count that comes off the response rather than the rows.
//
// What runs here, as opposed to being read off the source:
//
// * `loadRuns` — the exact function `RunsPanel` polls — through the REAL
//   `api.listBackgroundSessions`, with only `globalThis.fetch` replaced. Both request URLs
//   are asserted off the captured URL, not off a string in `api.ts`.
// * `panelReduce` — the exact reducer the panel mounts with `useReducer` — pressed with the
//   handler the button receives, for a full reveal-then-hide cycle.
// * `RunsToolbar` — the component the panel renders — invoked with props whose `onToggle`
//   feeds that reducer, its returned element tree walked to the real button, its real
//   `onClick` invoked, and its markup re-rendered from the resulting state. So "the control
//   reveals and hides again" is a state→markup loop that actually ran, not a pattern
//   matched over `?raw` text.
//
// The one seam that is NOT executed, and why: a full mounted `BackgroundPage` cannot be
// rendered in this repo's test environment at all. `vitest.config.ts` pins
// `environment: "node"`, `web/package.json` is a denied path for an automod round so no DOM
// package (`jsdom`, `happy-dom`, `linkedom` — none are in `web/node_modules`) can be added,
// and React 19 ships no server DOM (`react-dom/server` produces markup only, which is why
// the toolbar's render below asserts a serialized string). `BackgroundPage.test.tsx` was
// tried and dies on `import.meta.env` — the real `api.ts` reads it at module scope, so even
// `vi.mock` does not get a mounted panel past a config module. What those limits leave
// unexecuted is therefore exactly one binding inside `RunsPanel`: the JSX line that passes
// `makeToggleDispatcher(dispatch)` to the toolbar's `onToggle` prop. Its two ends run here
// (the factory, and the reducer it dispatches into), and the wiring is checked as text in
// `the panel wires the executed pieces together` below.

// What #3640's nodes below add to that route, and why they need the clock: the
// decision under test is a class on one span, so it is rendered (`SourceHealthRow` is
// the leaf `SourceCard` renders) and the one link that cannot be executed here is
// checked as text at the bottom of the block. `vi.setSystemTime` is pinned because
// `relative()` measures against now — every age asserted below is arithmetic on two
// seeded stamps, so "passes today" and "passes because the box's date drifted" are
// not the same claim.
//
// A span's colour is read out of the markup by `classOfSpanHolding` rather than by
// grepping the file for `text-rose-400`, because the closed-burst row legitimately
// DOES contain a rose span — `{h.failed} failed` is a rose count whatever the tail
// says — and an assertion over the whole string could not tell those two apart.

import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement } from 'react'
import source from './BackgroundPage.tsx?raw'
import panelSource from './runsPanelState.ts?raw'
import toolbarSource from './RunsToolbar.tsx?raw'
import rowSource from './SourceHealthRow.tsx?raw'
import { api } from '../../api'
import { RunsToolbar } from './RunsToolbar'
import {
  ALARM_FAIL_RATE,
  SourceHealthRow,
  failureAgeText,
  kindSplitText,
  rateIsAlarming,
  type SourceHealth,
} from './SourceHealthRow'
import {
  initialPanelState,
  loadRuns,
  makeToggleDispatcher,
  panelReduce,
  RUNS_WINDOW,
  toggleEvent,
  type PanelEvent,
  type PanelState,
  type RunsEnvelope,
} from './runsPanelState'

const realFetch = globalThis.fetch

afterEach(() => {
  globalThis.fetch = realFetch
})

/// A fetch stub that answers like the endpoint does: one window of rows, plus the count of
/// bench-family transcripts it withheld from that window. `withheld` is the default-request
/// case (rows are non-bench, 148 of them were set aside); `rows` is the opt-in case.
function stubFetch(rows: object[], withheld: number): string[] {
  const urls: string[] = []
  globalThis.fetch = (async (input: unknown, init?: { method?: string }) => {
    const url = String(input)
    if (init?.method && init.method !== 'GET') urls.push(`${init.method} ${url}`)
    else urls.push(url)
    const body = url.includes('include_bench=true')
      ? { sessions: rows, count: rows.length, bench_hidden: 0 }
      : { sessions: rows.slice(0, 2), count: Math.min(rows.length, 2), bench_hidden: withheld }
    return { ok: true, status: 200, statusText: 'OK', json: async () => body }
  }) as unknown as typeof fetch
  return urls
}

const row = (id: string, sourceName: string) => ({
  id, session_key: id, title: id, preview: '', platform: 'worker',
  source: sourceName, model: 'm', inner_voice: false, message_count: 3,
  last_active: '2026-10-13T00:00:00+00:00',
})

const benchRows = [row('bench-1', 'bench'), row('bench-2', 'bench-mine')]
const otherRows = [row('run-1', 'job-claim'), row('run-2', 'nightly-reflect')]

/// Walk a rendered element tree to the toolbar's button and hand back the node, so the
/// click below goes through the element `RunsToolbar` actually returned.
function findButton(node: unknown): { props: { onClick?: () => void; children?: unknown } } {
  const stack: unknown[] = [node]
  while (stack.length) {
    const n = stack.pop()
    if (Array.isArray(n)) { stack.push(...n); continue }
    if (!n || typeof n !== 'object') continue
    const el = n as { type?: unknown; props?: Record<string, unknown>; children?: unknown }
    if (el.type === 'button') return el as never
    if (el.props?.children) stack.push(el.props.children)
  }
  throw new Error('no <button> in the tree RunsToolbar returned')
}

describe('the bench control drives one request per class (#2648 clause 4, executed)', () => {
  it('asks for the non-bench window first and the bench family second, same window both times', async () => {
    const urls = stubFetch([...otherRows, ...benchRows], 148)
    const events: PanelEvent[] = []
    const dispatch = (e: PanelEvent) => { events.push(e) }

    // The panel's mount: `showBench` is false, so this is the class the endpoint hides by
    // default. Read off the captured URL, not off `api.ts`.
    await loadRuns(false, dispatch)
    expect(urls[0]).toContain('/api/background/sessions?limit=150')
    expect(urls[0]).not.toContain('include_bench')

    // The press: the handler the button holds, then the request the panel's effect makes
    // with the new state. The window is `RUNS_WINDOW` in both — the control cannot resize
    // it, which is the half of clause 4 that says hiding must not change which 150 rows
    // the endpoint returns.
    await loadRuns(panelReduce(initialPanelState, toggleEvent()).showBench, dispatch)
    expect(urls[1]).toContain('/api/background/sessions?limit=150')
    expect(urls[1]).toContain('include_bench=true')
    expect(urls[1]).toContain(`limit=${RUNS_WINDOW}`)

    // Pressing again is a third request for the hidden class — hide-again without a reload
    // — and it is the same URL the mount used.
    const back = panelReduce(panelReduce(initialPanelState, toggleEvent()), toggleEvent())
    expect(back.showBench).toBe(false)
    await loadRuns(back.showBench, dispatch)
    expect(urls[2]).toContain('/api/background/sessions?limit=150')
    expect(urls[2]).not.toContain('include_bench')

    // Three requests, no navigation: the panel never leaves the route it is on. Checked
    // where the toggle actually lives — the state machine, the toolbar, and the JSX element
    // the panel renders — because `BackgroundPage.tsx` legitimately does hold a
    // `window.location.reload()` elsewhere: the Sources tab's manual refresh button.
    expect(urls).toHaveLength(3)
    expect(panelSource).not.toMatch(/location|reload|history\.push/)
    expect(toolbarSource).not.toMatch(/location|reload/)
    const toolbarCall = source.slice(source.indexOf('<RunsToolbar'))
    expect(toolbarCall.slice(0, toolbarCall.indexOf('/>'))).not.toMatch(/location|reload/)

    // And each response lands in the reducer as the panel would put it: one `loaded` per
    // request, carrying the envelope the endpoint sent.
    expect(events.map(e => e.type)).toEqual(['loaded', 'loaded', 'loaded'])
  })

  it('takes the hidden count off the response, never off the rows it was sent', async () => {
    // The default response carries TWO non-bench rows and reports 148 trials withheld. A
    // panel that derived the number from `sessions` would count zero bench rows in it and
    // render "show 0 bench runs" — the exact misreport clause 3 exists to prevent, because
    // the rows are by construction only the ones that were NOT withheld.
    const urls = stubFetch([...otherRows, ...benchRows], 148)
    let state = initialPanelState
    const dispatch = (e: PanelEvent) => { state = panelReduce(state, e) }
    await loadRuns(false, dispatch)
    expect(urls).toHaveLength(1)
    expect(state.hiddenBench).toBe(148)
    expect(state.rows.map(r => r.source)).toEqual(['job-claim', 'nightly-reflect'])
    const countedFromRows = state.rows.filter(r => r.source === 'bench' || r.source === 'bench-mine').length
    expect(countedFromRows).toBe(0)
    expect(state.hiddenBench).not.toBe(countedFromRows)

    // The opt-in response carries the trials and reports nothing withheld.
    await loadRuns(true, dispatch)
    expect(state.hiddenBench).toBe(0)
    expect(state.rows).toHaveLength(4)
    expect(state.rows.filter(r => r.source === 'bench' || r.source === 'bench-mine')).toHaveLength(2)

    // A response that omits the field (the endpoint's two earliest returns do) reads as
    // "nothing withheld", not as a guess carried over from the previous poll.
    const noField = panelReduce(state, { type: 'loaded', out: { sessions: otherRows } })
    expect(noField.hiddenBench).toBe(0)
  })
})

describe('pressing the rendered toolbar moves the panel and re-renders its copy (#2648 clause 4, executed)', () => {
  it('clicks the real button and the markup follows the state it produced', () => {
    let state = panelReduce(initialPanelState, {
      type: 'loaded',
      out: { sessions: otherRows, count: 2, bench_hidden: 148 } satisfies RunsEnvelope,
    })
    const readState = () => state
    const dispatch = (e: PanelEvent) => { state = panelReduce(state, e) }
    const markup = (s: PanelState) => renderToStaticMarkup(createElement(RunsToolbar, {
      showBench: s.showBench, hiddenBench: s.hiddenBench, shown: s.rows.length,
      // The same handler the panel hands the toolbar, fed by this loop's dispatch.
      onToggle: makeToggleDispatcher(dispatch),
    }))

    const first = markup(readState())
    expect(first).toContain('show 148 bench runs')
    expect(first).toContain('148 bench withheld')

    // The click goes through the element `RunsToolbar` RETURNED from its own function body,
    // and the `onClick` it put on that element — not through a copy of the handler written
    // in this file. `RunsToolbar` takes no hooks, so calling it is rendering it.
    const button = findButton(RunsToolbar({
      showBench: readState().showBench, hiddenBench: readState().hiddenBench,
      shown: readState().rows.length, onToggle: makeToggleDispatcher(dispatch),
    }))
    expect(typeof button.props.onClick).toBe('function')
    button.props.onClick!()
    expect(readState().showBench).toBe(true)
    // Revealing is additive: the rows already on screen stay until the new answer lands.
    expect(readState().rows).toHaveLength(2)

    // The copy the browser would now show, rendered from the state the click produced. The
    // server has not been asked yet, so the count on screen is still the last one it gave.
    const second = markup(readState())
    expect(second).toContain('hide bench')
    expect(second).toContain('2 runs')
    expect(second).toContain('148 bench withheld')

    // The trials arrive, the count goes to zero, and the label flips back — the hide half
    // of the cycle, again driven by a click rather than by an assumption about one.
    state = panelReduce(readState(), {
      type: 'loaded',
      out: { sessions: [...otherRows, ...benchRows], count: 4, bench_hidden: 0 },
    })
    expect(markup(readState())).toContain('4 runs')
    expect(markup(readState())).not.toContain('withheld')
    button.props.onClick!()
    expect(readState().showBench).toBe(false)
    const third = markup(readState())
    expect(third).toContain('show 0 bench runs')
    expect(third).not.toContain('withheld')
    expect(third).toContain('4 runs')
  })

  it('renders the singular row count without mangling it', () => {
    const html = renderToStaticMarkup(createElement(RunsToolbar, {
      showBench: true, hiddenBench: 0, shown: 1, onToggle: () => {},
    }))
    expect(html).toContain('1 run')
    expect(html).not.toContain('1 runs')
  })
})

describe('the panel wires the executed pieces together (#2648 clause 4, wiring)', () => {
  // These are the ONE unexecuted link, checked as text: that `RunsPanel` mounts the reducer
  // this file exercised, renders the toolbar this file rendered, and hands the toolbar's
  // `onToggle` the handler factory this file pressed. See the file header for why a mounted
  // panel cannot be rendered here.
  const wiring = [
    ['mounts the reducer', /useReducer\(panelReduce,\s*initialPanelState\)/],
    ['polls through the extracted loader', /void loadRuns\(state\.showBench, dispatch\)/],
    ['re-asks on class change', /\}, \[state\.showBench\]\)/],
    ['renders the toolbar it exports', /<RunsToolbar/],
    ['hands the toolbar the toggling handler', /onToggle=\{makeToggleDispatcher\(dispatch\)\}/],
    // The withheld count reaches the control out of state, which the reducer filled from the
    // response. The draft of this row matched the COMMENT that said so
    // (/server's `bench_hidden`/), which is a prose assertion — it survived every way of
    // wiring the number wrong, including `hiddenBench={0}` at the call site. These two match
    // the props, so they cannot.
    ['passes the withheld count out of state', /hiddenBench=\{state\.hiddenBench\}/],
    ['passes the on-screen row count', /shown=\{state\.rows\.length\}/],
  ] as const
  for (const [what, re] of wiring) {
    it(`${what}`, () => {
      expect(source).toMatch(re)
    })
  }

  it('the toolbar is imported from the module the test renders, not a copy', () => {
    expect(source).toMatch(/import \{ RunsToolbar \} from '\.\/RunsToolbar'/)
    expect(source).toMatch(/panelReduce,\s*\n?\s*} from '\.\/runsPanelState'|runsPanelState'/)
  })

  it('leaves no second copy of the copy or the class list in the panel', () => {
    // The reason the toolbar is a component and not inline JSX: with the labels in
    // `benchRows` and the reducer in `runsPanelState`, the panel has one place left to put
    // them. A `setHiddenBench`/`useState(0)` pair is the shape that would let a second
    // count drift from the one the test above pins.
    expect(source).not.toMatch(/setHiddenBench|setShowBench/)
    expect(source).not.toMatch(/benchToggleLabel|hiddenBenchSuffix/)
    expect(panelSource).toMatch(/import \{ api, type BackgroundSession \} from '\.\.\/\.\.\/api'/)
  })
})

describe('the client builds the real URL (#2648 clause 4, executed)', () => {
  // The same call path the panel takes, with only the network replaced. `loadRuns` above
  // covers the panel's own calls; this pins the client's own surface (window clamping)
  // against the URL the panel's fixed window produces.
  const calls: string[] = []
  const stub = (body: object) => {
    globalThis.fetch = (async (input: unknown) => {
      calls.push(String(input))
      return { ok: true, status: 200, statusText: 'OK', json: async () => body }
    }) as unknown as typeof fetch
  }

  afterEach(() => { calls.length = 0 })

  it('omits include_bench entirely on the default call', async () => {
    stub({ sessions: [], count: 0, bench_hidden: 0 })
    const out = await api.listBackgroundSessions(150)
    expect(calls[0]).toBe('/api/background/sessions?limit=150')
    expect(out.bench_hidden).toBe(0)
  })

  it('appends include_bench=true and keeps the window when asked', async () => {
    stub({ sessions: [], count: 0, bench_hidden: 0 })
    await api.listBackgroundSessions(150, true)
    expect(calls[0]).toBe('/api/background/sessions?limit=150&include_bench=true')
  })

  it('passes the withheld count through instead of dropping it', async () => {
    stub({ sessions: [], count: 0, bench_hidden: 148 })
    const out = await api.listBackgroundSessions(150)
    expect(out.bench_hidden).toBe(148)
  })
})

// ── The per-source health row dates its own failures (#3640) ──────────────────
//
// `fail_rate` is a ratio over a window and says nothing about where inside that
// window its numerator sits. Read 2026-10-28, youtube-digest's row was 51 % over 100
// runs whose every failure predated a landed revert by five days — and #3523 was
// filed at `high` off exactly that payload, because dating it meant a hand-run
// aggregate over `workers.db` that nobody in front of a dashboard is going to do. The
// queue has since put the tail beside the rate (`last_failure` and
// `consecutive_failures`, #3527); these nodes pin the panel reading it, in both
// directions: a closed burst stops rendering as a live defect, and a live burst still
// does. The second direction is not decoration — over `?days=90` the live endpoint
// reports ZERO sources with `consecutive_failures > 0`, so this gate could ship as a
// colour that never fires again and nobody downstream would notice.

const SEEDED_NOW = '2026-11-02T12:00:00+00:00'

/// The 2026-10-28 payload #3523 was filed from, shaped as the route sends it: a rate
/// over the whole window, and a tail whose newest run is a success. `last_failure`
/// sits 2 d 19 h BEFORE `last_completed`, so `relative()` puts it at `2d ago` and the
/// completion at `1h ago`.
const CLOSED_BURST: SourceHealth = {
  total: 100, ok: 49, failed: 51, skipped: 0,
  fail_rate: 0.51, gpu_hours: 5.96,
  last_completed: '2026-11-02T11:00:00+00:00',
  last_failure: '2026-10-30T17:09:58+00:00',
  consecutive_failures: 0,
}

/// Same rate, opposite tail: three failures in a row at the end of the window, and a
/// newest failure NEWER than the newest completion. This is the row the rose exists
/// for, and it must survive whatever the closed-burst half of the gate does.
const LIVE_BURST: SourceHealth = {
  total: 100, ok: 49, failed: 51, skipped: 0,
  fail_rate: 0.51, gpu_hours: 5.96,
  last_completed: '2026-10-31T11:00:00+00:00',
  last_failure: '2026-11-02T11:30:00+00:00',
  consecutive_failures: 3,
}

/** The class attribute of the `<span>` whose entire text is `text`, or `''` when that
 *  span carries no class — which on this row is the muted ink, colour being opt-in.
 *  Scoped to one span deliberately: `{h.failed} failed` is rose in every seed below,
 *  so an assertion over the whole markup cannot tell "the rate is muted" from "there
 *  is no rose anywhere". */
function classOfSpanHolding(html: string, text: string): string {
  const esc = text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const m = html.match(new RegExp(`<span([^>]*)>${esc}</span>`))
  if (!m) throw new Error(`no <span> holds exactly ${JSON.stringify(text)} in:\n${html}`)
  const cls = /class="([^"]*)"/.exec(m[1])
  return cls ? cls[1] : ''
}

describe('SourceHealthRow — the failure rate and the age of its failures', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date'] })
    vi.setSystemTime(new Date(SEEDED_NOW))
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('the alarm threshold is the 50 % the clauses name', () => {
    expect(ALARM_FAIL_RATE).toBe(0.5)
  })

  it('a closed burst prints the age beside the rate and drops the rose', () => {
    const html = renderToStaticMarkup(createElement(SourceHealthRow, { h: CLOSED_BURST }))
    // Positive control on the formatter: 2 d 19 h back is `2d ago`, and a stamp the
    // payload never sent prints nothing rather than a label with no age after it.
    expect(failureAgeText(CLOSED_BURST)).toBe('last failure 2d ago')
    expect(failureAgeText({ ...CLOSED_BURST, last_failure: null })).toBe('')
    expect(html).toContain('51% failing')
    expect(html).toContain('last failure 2d ago')
    expect(html).toContain('last 1h ago')
    // The colour decision, at both levels.
    expect(rateIsAlarming(CLOSED_BURST)).toBe(false)
    expect(classOfSpanHolding(html, '51% failing')).toBe('')
    expect(classOfSpanHolding(html, 'last failure 2d ago')).toBe('')
    // …and the rose that DOES survive is the run count's, not the rate's.
    expect(classOfSpanHolding(html, '51 failed')).toContain('text-rose-400')
  })

  it('prints the age immediately after the rate, not somewhere in the row', () => {
    const html = renderToStaticMarkup(createElement(SourceHealthRow, { h: CLOSED_BURST }))
    const rateEnd = html.indexOf('51% failing') + '51% failing'.length
    const ageStart = html.indexOf('last failure 2d ago')
    expect(rateEnd).toBeGreaterThan('51% failing'.length)
    expect(ageStart).toBeGreaterThan(rateEnd)
    // Adjacency is the clause ("beside the rate"): the only bytes between the two
    // readings are the closing tag and the opening tag of the age's own span.
    expect(html.slice(rateEnd, ageStart)).toMatch(/^<\/span><span( class="")?>$/)
  })

  it('a live burst at the same rate still goes rose, streak and all', () => {
    const html = renderToStaticMarkup(createElement(SourceHealthRow, { h: LIVE_BURST }))
    expect(rateIsAlarming(LIVE_BURST)).toBe(true)
    expect(failureAgeText(LIVE_BURST)).toBe('last failure 30m ago · 3 in a row')
    expect(classOfSpanHolding(html, '51% failing')).toContain('text-rose-400')
    expect(classOfSpanHolding(html, 'last failure 30m ago · 3 in a row'))
      .toContain('text-rose-400')
  })

  it('a backend that cannot answer the tail question keeps the rose', () => {
    // `consecutive_failures` is optional because a backend older than `b22a1b3c`
    // sends no such key. Removing it must NOT disarm the alarm: unknown is read as
    // live, or every pre-#3527 backend silently loses its only alarm.
    const { consecutive_failures: removed, ...legacy } = LIVE_BURST
    expect(removed).toBe(3) // the destructure really did take the streak away
    expect(legacy).not.toHaveProperty('consecutive_failures')
    expect(rateIsAlarming(legacy)).toBe(true)
    const html = renderToStaticMarkup(createElement(SourceHealthRow, { h: legacy }))
    expect(classOfSpanHolding(html, '51% failing')).toContain('text-rose-400')
    // The age still prints; the streak is simply not invented for it.
    expect(html).toContain('last failure 30m ago')
    expect(html).not.toContain('in a row')
  })

  it('a rate under the threshold is never loud, live tail or not', () => {
    // The other half of the AND: `consecutive_failures` alone must not colour a row
    // whose failure rate is ordinary noise.
    expect(rateIsAlarming({ ...LIVE_BURST, fail_rate: 0.49 })).toBe(false)
    expect(rateIsAlarming({ ...LIVE_BURST, fail_rate: null })).toBe(false)
  })

  it('a source with no failure in the window prints no age at all', () => {
    // `last_failure: null` is the common case — over `?days=7` today `autoresearch`
    // and `backlog-cluster` report none — and a dangling "last failure" with nothing
    // after it is the failure mode `relative()` returning '' exists to prevent.
    const clean: SourceHealth = {
      ...CLOSED_BURST, ok: 100, failed: 0, fail_rate: 0,
      last_failure: null, consecutive_failures: 0,
    }
    const html = renderToStaticMarkup(createElement(SourceHealthRow, { h: clean }))
    expect(html).toContain('0% failing')
    expect(html).not.toContain('last failure')
    expect(html).not.toContain('text-rose-400')
  })

  // ── #3906: the by_kind split, printed as context and never as a second alarm ───────
  // Shaped off the live endpoint (`/api/workers/health?days=7`, 2026-11-07), where
  // `youtube-digest` is the one source reporting two kinds: 57 runs blended with 1 failed,
  // the video half 50 with 1 and the paper half 7 with 0. The blend is what gates; the
  // halves are what a reader needs, because the row on its own says only "1 of 57", which
  // is a different story from "the failure was a video" — and says nothing at all about a
  // paper half of 7 that would fail on its own rate alone.
  // The same 57 runs as the route sends them from a backend older than `64ff6c01`: every
  // figure the newer one sends, and no `by_kind` key. `TWO_KINDS` is this object plus the
  // partition, so the two seeds differ in exactly one field and the absence node below
  // renders a payload that never had the key rather than one it was stripped of.
  const NO_PARTITION: SourceHealth = {
    ...CLOSED_BURST, total: 57, ok: 56, failed: 1, fail_rate: 1 / 57,
    last_failure: '2026-11-06T09:00:00Z', consecutive_failures: 0,
  }

  const TWO_KINDS: SourceHealth = {
    ...NO_PARTITION,
    by_kind: {
      video: { total: 50, failed: 1, fail_rate: 0.02 },
      paper: { total: 7, failed: 0, fail_rate: 0.0 },
    },
  }

  it('a two-kind source prints both halves, each with its own rate', () => {
    // Both figures come out of the block itself, not off the blended row: a split that
    // printed `1 of 57` twice would be the same number the row already shows.
    expect(kindSplitText(TWO_KINDS)).toBe(
      'paper: 7 runs, 0 failed, 0% failing · video: 50 runs, 1 failed, 2% failing')
    const html = renderToStaticMarkup(createElement(SourceHealthRow, { h: TWO_KINDS }))
    expect(html).toContain('paper: 7 runs, 0 failed, 0% failing')
    expect(html).toContain('video: 50 runs, 1 failed, 2% failing')
    // Name order, so two reads of one window are byte-identical: the payload's own key
    // order is whatever `workers/queue.py` happened to see first.
    expect(html.indexOf('paper:')).toBeLessThan(html.indexOf('video:'))
  })

  it('the split is muted ink, and a kind at 100% still does not sound the alarm', () => {
    // Clause 4's other half, and the one that could be gotten wrong quietly: colouring
    // the split by the kind's own rate, or alarming on a kind at all. A kind can be 1
    // failed of 1 run — `__none__` alone in a window is exactly that — and the source is
    // not on fire, so the source-level blend is the only number the decision may read.
    const paperOnFire: SourceHealth = {
      ...TWO_KINDS,
      by_kind: {
        video: { total: 50, failed: 1, fail_rate: 0.02 },
        paper: { total: 1, failed: 1, fail_rate: 1.0 },
      },
    }
    const line = 'paper: 1 run, 1 failed, 100% failing · video: 50 runs, 1 failed, 2% failing'
    expect(kindSplitText(paperOnFire)).toBe(line)
    expect(rateIsAlarming(paperOnFire)).toBe(false)
    const html = renderToStaticMarkup(createElement(SourceHealthRow, { h: paperOnFire }))
    // One span, muted, and the row's only rose is the run count's — the same reading the
    // closed-burst node above pins for the source-level rate, now with a kind at 100 %.
    expect(classOfSpanHolding(html, line)).toBe('text-zinc-500')
    expect(classOfSpanHolding(html, '1 failed')).toContain('text-rose-400')
    expect(html.match(/text-rose-400/g)).toHaveLength(1)
    // The decision cannot see a kind's number even by accident: its body names the
    // source's own field and no partition of it. Read off the component's own bytes,
    // because an assert on a regex that did not match is an assert on `''`.
    const deciding = rowSource.match(/export function rateIsAlarming[\s\S]*?\n}/)?.[0]
    expect(deciding, 'no rateIsAlarming in the file this node reads').toBeTruthy()
    expect(deciding).toContain('fail_rate')
    expect(deciding).not.toContain('by_kind')
  })

  it('a payload with no split, or one that says nothing, prints no per-kind markup', () => {
    // Three absences, one behaviour, and every one of them decided on the RENDERED bytes:
    // no key at all is a backend older than `64ff6c01`; an empty block is the same claim
    // from a newer one; a ONE-entry block is that source's own figures restated at it,
    // since the per-kind totals sum to its total — every source but `youtube-digest` is
    // that case, and printing it would put the same number on every row twice.
    //
    // `paper:` / `video:` / `__none__:` are what a per-kind entry looks like once it is in
    // the markup — the kind's name and its colon, which no other span on this row prints —
    // so their absence in the rendered HTML is the no-markup claim itself, not a proxy for
    // a fixture that was built without the field. `__none__` is in the list because the
    // emitter keys an unclassified run exactly that way (`NO_KIND` in `workers/queue.py`),
    // and the component must not invent the group when the payload has none.
    const KIND_MARKERS = ['paper:', 'video:', '__none__:']
    const seeds: Array<[string, SourceHealth]> = [
      ['no by_kind key at all', NO_PARTITION],
      ['an empty by_kind block', { ...TWO_KINDS, by_kind: {} }],
      ['one kind, which is the whole source', {
        ...TWO_KINDS,
        by_kind: { video: { total: 57, failed: 1, fail_rate: 1 / 57 } },
      }],
    ]
    for (const [label, h] of seeds) {
      expect(kindSplitText(h), label).toBe('')
      const html = renderToStaticMarkup(createElement(SourceHealthRow, { h }))
      for (const marker of KIND_MARKERS) {
        expect(html, `${label} rendered ${marker}`).not.toContain(marker)
      }
      // The blended rate survives every one of them, and it is the ONLY rate the row
      // prints when there is no split to print: 1 failed of 57 rounds to 2 %.
      expect(html, label).toContain('2% failing')
      expect(html.match(/% failing/g), label).toHaveLength(1)
    }
  })

  it('SourceCard renders this component, so the decision above is the one on screen', () => {
    // The one link these nodes cannot execute (no DOM env — see this file's header):
    // that the page hands a source's own `health` to the leaf whose markup they just
    // rendered, rather than keeping a second inline copy of the row.
    expect(source).toContain("import { SourceHealthRow } from './SourceHealthRow'")
    expect(source).toMatch(/const h = source\.health/)
    expect(source).toMatch(/<SourceHealthRow h=\{h\} \/>/)
    // And the page must not still carry its own copy of either decision: a second
    // `>= 0.5` or a second `relative()` in this file is the drift these lines guard.
    expect(source).not.toMatch(/function relative\(/)
    expect(source).not.toMatch(/fail_rate >= 0\.5/)
    expect(source).toContain("import { relative } from '@/lib/relativeTime'")
    // `health: null` stays SourceCard's branch and stays worded as "no runs" — the
    // absence the panel must never render as a 0 % pass rate.
    expect(source).toContain('<span>no runs in the window</span>')
  })
})
