import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import {
  CheckButton,
  EmptyState,
  focusStateFromMessage,
  type FocusMessage,
} from './SidePanelApp'

// This file runs under vitest's `environment: "node"`, and `web/package.json` is
// a denied build input no round may touch, so no DOM package can be added: the
// panel's rendered half is pinned the way `components/VoiceInterruptButton.test.tsx`
// pins its own — renderToStaticMarkup for what a push shows, called per test so a
// broken render fails one assertion instead of the whole collection.
//
// Every render below is `panelFor`, which renders what `focusStateFromMessage`
// returned for the message. That is the point: the field under test is the push
// the service worker sent, never a hand-written prop, so a mutation to either
// half of the panel reddens this file. Dropping `!canCheck` from the button's
// `disabled` takes the flag away from the button alone; making the mapping stop
// reading `msg.canCheck` takes it away from the button and the empty-state copy
// together, whichever way it stops reading it.

const CHECKABLE_PUSH: FocusMessage = {
  type: 'focus-session',
  windowId: 3,
  tabId: 7,
  sessionKey: 'ext_seed',
  url: 'https://example.com/article',
  title: 'An Article',
  canCheck: true,
}

// The same push with the `canCheck` key left off — #3568's field. A tab switch
// onto an already-loaded tab used to send exactly this, and the panel read it as
// a page that cannot be checked.
const FLAG_OMITTED_PUSH: FocusMessage = {
  type: 'focus-session',
  windowId: 3,
  tabId: 7,
  sessionKey: 'ext_seed',
  url: 'https://example.com/article',
  title: 'An Article',
}

/** Panel state plus the two components that render it, all from one message. */
function panelFor(msg: FocusMessage) {
  const state = focusStateFromMessage(msg)
  return {
    state,
    button: renderToStaticMarkup(
      <CheckButton
        canCheck={state.canCheck}
        pending={false}
        onRequest={() => {}}
      />,
    ),
    empty: renderToStaticMarkup(<EmptyState canCheck={state.canCheck} />),
  }
}

describe('the panel a focus-session push renders', () => {
  it('enables the labelled check button for a checkable push', () => {
    const panel = panelFor(CHECKABLE_PUSH)
    expect(panel.button).toContain('<button')
    expect(panel.button).toContain('Check it out, Lloyd')
    expect(panel.button).not.toMatch(/disabled=""/)
  })

  it('reads an omitted canCheck as false', () => {
    expect(panelFor(FLAG_OMITTED_PUSH).state.canCheck).toBe(false)
  })

  it('renders the push that omitted canCheck as a disabled check button', () => {
    const panel = panelFor(FLAG_OMITTED_PUSH)
    expect(panel.button).toContain('Check it out, Lloyd')
    expect(panel.button).toMatch(/disabled=""/)
  })

  it('shows the refusal copy for the push that omitted canCheck', () => {
    const panel = panelFor(FLAG_OMITTED_PUSH)
    expect(panel.empty).toContain('No checkable page here')
    expect(panel.empty).not.toContain('Ready when you are')
  })

  it('shows the ready copy for a checkable page with no session yet', () => {
    const panel = panelFor({ ...CHECKABLE_PUSH, sessionKey: null })
    expect(panel.empty).toContain('Ready when you are')
    expect(panel.empty).not.toContain('No checkable page here')
  })

  it('carries sessionKey, url and title through from the push', () => {
    const { state } = panelFor(CHECKABLE_PUSH)
    expect([state.sessionKey, state.url, state.title]).toEqual([
      'ext_seed',
      'https://example.com/article',
      'An Article',
    ])
  })

  it('defaults url and title to null for a push carrying neither', () => {
    const state = focusStateFromMessage({
      type: 'focus-session',
      windowId: 3,
      tabId: 7,
      sessionKey: null,
    })
    expect([state.url, state.title]).toEqual([null, null])
  })
})
