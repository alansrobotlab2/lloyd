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

import { describe, it, expect, afterEach } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement } from 'react'
import source from './BackgroundPage.tsx?raw'
import panelSource from './runsPanelState.ts?raw'
import toolbarSource from './RunsToolbar.tsx?raw'
import { api } from '../../api'
import { RunsToolbar } from './RunsToolbar'
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
