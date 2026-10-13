// The Runs panel's state machine, outside the component so it can be EXECUTED (#2648).
//
// `web/vitest.config.ts` pins `environment: "node"`, and the repo ships no DOM package
// (`jsdom`, `happy-dom` and `linkedom` are all absent from `web/node_modules`, and
// `web/package.json` is a denied path for an automod round), so a browser event cannot be
// dispatched over a mounted panel in a test here. The control's behaviour therefore does
// not live in markup the test can only pattern-match: the panel drives THIS reducer
// through `useReducer`, and `BackgroundPage.test.ts` runs the same reducer the click
// dispatches, through the same `loadRuns` the panel polls with, against the real
// `api.listBackgroundSessions`. What is left unexecuted is React's own event binding
// inside `RunsPanel` — one line, named as a seam in that test file.
//
// Why the panel needs a state machine at all: the endpoint hides the bench family by
// default (one day of one detached grid wrote 273 transcripts on 2026-10-07 against a
// listing 150 rows wide) and reports `bench_hidden`. So the panel holds a class choice
// (`showBench`) and a server-reported count (`hiddenBench`) that are not the same kind of
// thing as the rows, and the copy that makes the window honest is a function of all three.

import { api, type BackgroundSession } from '../../api'

// The row window the Runs panel asks the endpoint to fill. One name, because the tab's
// crowding problem is measured against it: 150 rows is the whole screen a grid competes
// with, and `bench_hidden` is only legible beside it.
export const RUNS_WINDOW = 150

// The envelope `GET /api/background/sessions` answers with, as far as the panel reads it.
// The fields are optional because that is the truth on the wire: `bench_hidden` is absent
// from the endpoint's two earliest returns (missing directory, unreadable directory), and a
// missing count reads as "nothing was withheld", never as a guess.
export type RunsEnvelope = {
  sessions?: BackgroundSession[] | null
  count?: number | null
  bench_hidden?: number | null
}

export type PanelState = {
  // Which CLASS the endpoint may send: false = the bench family is withheld (the default),
  // true = the trials join the page. A class choice, not a freshness flag — the rows still
  // come from one request to one endpoint.
  showBench: boolean
  rows: BackgroundSession[]
  // How many bench-family transcripts THIS response withheld, as the server counted them.
  hiddenBench: number
  loading: boolean
  error: string | null
}

export type PanelEvent =
  | { type: 'toggle' }
  | { type: 'loaded'; out: RunsEnvelope }
  | { type: 'failed'; message: string }

export type Dispatch = (event: PanelEvent) => void

export const initialPanelState: PanelState = {
  showBench: false,
  rows: [],
  hiddenBench: 0,
  loading: true,
  error: null,
}

// One step of the panel. Pure and total: every event returns a new state, and no event
// clears `rows` except a `loaded` that genuinely carries different ones.
export function panelReduce(state: PanelState, event: PanelEvent): PanelState {
  switch (event.type) {
    case 'toggle':
      // Only the class changes. `rows` and `hiddenBench` are the server's answer to the
      // PREVIOUS question and stay on screen until the new answer lands, so pressing the
      // control never blanks the list — and nothing in this reducer removes a row the
      // reader was looking at either. The server decides what is added or withheld
      // (#2648 clause 2's additive rule); this only changes what is asked for.
      return { ...state, showBench: !state.showBench }
    case 'loaded':
      return {
        ...state,
        rows: event.out.sessions ?? [],
        // The withheld count is the SERVER's number. It cannot come off the rows: they are
        // exactly the ones that were NOT withheld, so counting them for the class always
        // answers 0 — on a day the endpoint set aside 273 trials the button would read
        // "show 0 bench runs" (#2648 clause 3).
        hiddenBench: event.out.bench_hidden ?? 0,
        loading: false,
        error: null,
      }
    case 'failed':
      // The message is kept and the rows are not: a page about runs that went wrong
      // unattended must not lose what it already had on a failed poll.
      return { ...state, loading: false, error: event.message }
  }
  // Unreachable while `PanelEvent` is the three arms above; the arm exists so an event
  // added later without a case here leaves the panel showing what it had rather than
  // throwing inside a render.
  return state
}

// The event the Runs toolbar's button produces. A function rather than an inline object at
// the call site so a test can assert the exact event the control dispatches.
export function toggleEvent(): PanelEvent {
  return { type: 'toggle' }
}

// The handler the button gets. `RunsPanel` binds this to `useReducer`'s dispatch and passes
// it as `onToggle`; the test presses it and reads the state back through `panelReduce`,
// which is the same pair the browser runs.
export function makeToggleDispatcher(dispatch: Dispatch): () => void {
  return () => dispatch(toggleEvent())
}

// Ask the endpoint for one window of runs, for one class, and report the outcome.
//
// `showBench` is a parameter rather than a read of state so the caller — the panel's
// effect, or a test — decides which class is being asked about. The window (`RUNS_WINDOW`)
// is deliberately NOT a parameter: revealing the bench family must not be able to resize
// it, which is the half of clause 4 that says the hidden state cannot change which 150 rows
// come back.
export async function loadRuns(showBench: boolean, dispatch: Dispatch): Promise<void> {
  try {
    const out = await api.listBackgroundSessions(RUNS_WINDOW, showBench)
    dispatch({ type: 'loaded', out })
  } catch (e) {
    dispatch({ type: 'failed', message: e instanceof Error ? e.message : String(e) })
  }
}
