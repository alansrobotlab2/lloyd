import { describe, expect, it } from 'vitest'

import pageSource from '../components/pages/BackgroundPage.tsx?raw'
import { runTranscriptIds, transcriptLabel } from './runSessions'

// The join `workers/pool.py` writes into `runs.meta_json` had no reader at all:
// a Background-tab run row rendered icon/summary/duration/time and stopped
// there, so "what did this run actually do" ended at a summary string. These pin
// the read half — one entry per recorded transcript, and an honest empty list
// for the sources that create no session.
describe('runTranscriptIds', () => {
  it('gives one entry per session id the run recorded, in recorded order', () => {
    expect(runTranscriptIds({
      meta_json: '{"session_ids": ["20260918_161828_autonomy_fcb5",'
        + ' "20260918_162410_autonomy_91a2"]}',
    })).toEqual([
      '20260918_161828_autonomy_fcb5',
      '20260918_162410_autonomy_91a2',
    ])
  })

  it('unions the collected list with the singular key a session-backed source stamps', () => {
    // Both keys are real, written by different files: the pool binds the
    // collected list (`workers/pool.py`), a source stamps its own `session_id`
    // (`workers/sources/arch_review.py`, `workers/sources/autocode.py`), and
    // neither knows the other exists — reading one key alone shows nothing for
    // whatever only the other one holds. No row count is quoted: the table is
    // pruned, so a number here would be stale before this test is next read.
    expect(runTranscriptIds({
      meta_json: '{"session_ids": ["a"], "session_id": "b"}',
    })).toEqual(['a', 'b'])
    expect(runTranscriptIds({ meta_json: '{"session_id": "solo"}' })).toEqual(['solo'])
  })

  it('lists a session once even when both keys name it', () => {
    expect(runTranscriptIds({
      meta_json: '{"session_ids": ["x", "x"], "session_id": "x"}',
    })).toEqual(['x'])
  })

  it('takes the parsed array the endpoint now ships, and keeps the blob as the fallback', () => {
    expect(runTranscriptIds({ session_ids: ['20260918_161828_autonomy_fcb5'] }))
      .toEqual(['20260918_161828_autonomy_fcb5'])
    // The endpoint ships both; the union must not double them.
    expect(runTranscriptIds({
      session_ids: ['a'],
      meta_json: '{"session_ids": ["a"], "session_id": "b"}',
    })).toEqual(['a', 'b'])
  })

  it('names no transcript for a row that names none', () => {
    // The empty case is those rows' normal shape, not the defect:
    // `architecture/background-runs.md` §12 lists the sources that record no
    // transcript by design (`automod-regression`, `autoresearch` among them), and
    // a run that dies before its first `create_session` names none. Per-source
    // row counts are deliberately not quoted here — the table is pruned, so a
    // number in a test comment is stale before the test is next read.
    for (const row of [
      {},
      { meta_json: null },
      { meta_json: '' },
      { meta_json: '{}' },
      { meta_json: '{"session_ids": []}' },
      { meta_json: '{"session_id": "   "}' },
      { meta_json: '[1, 2]' },
      { session_ids: [] },
      null,
      undefined,
    ]) {
      expect(runTranscriptIds(row)).toEqual([])
    }
  })

  it('survives a malformed meta_json without throwing', () => {
    // The column is written by the pool; a truncated or half-written blob must
    // cost that row its links, not the whole Sources panel.
    for (const meta of [
      'not json',
      '{"session_ids": ["a"',
      '{"session_ids": ',
      'null',
      'undefined',
      '{"session_ids": {"a": 1}}',
    ]) {
      expect(() => runTranscriptIds({ meta_json: meta })).not.toThrow()
      expect(runTranscriptIds({ meta_json: meta })).toEqual([])
    }
  })

  it('keeps a run that produced two transcripts navigable to both', () => {
    // One job can mint more than one session — a retry, or a handler that opens
    // a sub-session — which is why the field is a list and not a single id.
    const ids = runTranscriptIds({
      session_ids: ['20260922_010001_autonomy_1111', '20260922_011234_autonomy_2222'],
    })
    expect(ids).toHaveLength(2)
    expect(ids.map(transcriptLabel)).toEqual(['autonomy_1111', 'autonomy_2222'])
  })
})

describe('transcriptLabel', () => {
  it('drops the date and time the row already shows as its relative time', () => {
    expect(transcriptLabel('20260918_161828_autonomy_fcb5')).toBe('autonomy_fcb5')
    expect(transcriptLabel('20260923_074102_mission-control_20d56a50'))
      .toBe('mission-control_20d56a50')
  })

  it('renders an id that is not a minted session id whole', () => {
    expect(transcriptLabel('contract-test-1a2b')).toBe('contract-test-1a2b')
    expect(transcriptLabel('a_b')).toBe('a_b')
  })
})

// The click itself needs a React renderer this suite does not have (`environment:
// 'node'`, no jsdom in the project's dev dependencies, and `web/package.json` is
// a build input the automod loop may not land), so what CI can pin is that the
// page is wired to this module: it imports it, threads the page's own
// transcript-opening callback down to the Sources rows, and calls it per entry.
describe('the Background tab wires this module to its reader', () => {
  // `?raw` rather than `node:fs`: this project's tsconfig has no node types in
  // scope, and a Vite raw import is both how the bundler reads a file as text
  // and type-clean under `tsc --noEmit`. It binds the text itself, not `{ default }`.
  const page: string = pageSource

  it('imports the derivation rather than re-implementing it', () => {
    // `@/lib/...` is how every page here imports a pure lib module, the same
    // form `DashboardPage` and `InnerVoicePage` use for `sessionLabel`.
    expect(page).toMatch(/import\s*\{[^}]*runTranscriptIds[^}]*\}\s*from\s*'@\/lib\/runSessions'/)
  })

  it('hands the Sources panel the same callback the Runs panel already had', () => {
    expect(page).toMatch(/<SourcesPanel\b[^>]*onOpen=\{openInReader\}/)
    expect(page).toMatch(/<SourceCard\b[^>]*onOpen=\{onOpen\}/)
  })

  it('derives the row list from the run, not from anything the row already shows', () => {
    expect(page).toMatch(/const transcripts = runTranscriptIds\(run\)/)
  })

  it('renders the control inside the per-entry map, so a second transcript gets one too', () => {
    // A page that rendered only `transcripts[0]` would satisfy a grep for
    // `onOpen(sid)` and silently drop every later transcript — which is the half
    // of this clause the derivation cannot pin. So read the control's own
    // markup: the slice from the map's opening to the row's next static sibling
    // must hold exactly one button, keyed and opened by the map's own variable.
    const mapAt = page.indexOf('transcripts.map(sid => (')
    expect(mapAt).toBeGreaterThan(-1)
    const region = page.slice(mapAt, page.indexOf('<span', mapAt))
    expect((region.match(/<button/g) ?? []).length).toBe(1)
    expect(region).toMatch(/key=\{sid\}/)
    expect(region).toMatch(/onClick=\{\(\) => onOpen\(sid\)\}/)
    // And the shortcut that would have passed the loose grep stays unpresent.
    expect(page).not.toMatch(/transcripts\[\d\]/)
  })
})
