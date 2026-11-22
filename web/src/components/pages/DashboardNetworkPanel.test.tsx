// #4292 clauses 3 and 4: NetworkPanel has to say which arm it is counting.
//
// `agent_mcp.egress.network_report` ships the per-via split (#4044), and its own
// docstring is what makes a second instrument a readability problem rather than a new
// feature: every figure it returns except `per_via` counts `via='tool'` rows only —
// `total`, `hosts`, `by_decision` and the `per_destination`/`per_scope` rollups all
// pass `VIA_TOOL` into `_SINCE_TOOL` — and `app/routers/dashboard.py`'s `_network()`
// hands that payload over verbatim. Before this round the panel rendered neither the
// split nor the recorder's switch: the headline read `494 calls · 89 hosts · …` with
// no arm named, so the tool-only denominator was the number an operator armed
// `harness.egress_policy.enforce` on without ever seeing that a Bash arm existed or
// what it had caught.
//
// These nodes render the component; they do not grep it. `renderToStaticMarkup` under
// vitest's node environment is the device VoiceInterruptButton.test.tsx uses — there
// is no jsdom in this project, and what a panel *shows* is what ends up in that
// markup. Two fixtures carry the contract: a seeded `per_via` containing a
// `bash_proxy` row (clause 3 — traffic that cannot be produced yet, because
// config.yaml ships `harness.egress_policy.bash_record: false` and #4293 owns arming
// it), and the shape the live `/api/dashboard` returns today (clause 4: one
// `via: 'tool'` row, plus the older-backend case where the key is absent outright).
//
// The fixtures are annotated with `NetworkState['per_via']`'s element type, so tsc —
// which the gate's `frontend` rung runs as a delta over base — refuses this file if
// the typed reader and the emit site disagree about a key. The runtime half of that
// pin, which a type cannot do, lives in api.test.ts.
import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import type { NetworkState } from '../../api'
import { NetworkPanel } from './DashboardPage'

type ViaRow = NonNullable<NetworkState['per_via']>[number]

/** The tool arm, shaped from the live 2026-11-22 `/api/dashboard` row. */
const TOOL: ViaRow = {
  via: 'tool', count: 494, distinct_hosts: 89, denies: 0,
  last_at: '2026-11-22T14:33:46+00:00',
  scopes: [{ scope: 'worker:deep-research', count: 435 }],
}

/** A Bash arm. `denies: 2` is deliberate: per egress.py a denial under
 *  `via='bash_proxy'` is a recorder bug rather than a traffic finding, and the
 *  arm's own row still has to render. */
const BASH: ViaRow = {
  via: 'bash_proxy', count: 31, distinct_hosts: 4, denies: 2,
  last_at: '2026-11-22T13:02:11+00:00',
  scopes: [{ scope: 'interactive', count: 31 }],
}

/** Today's live policy block, five keys as the payload carries them. `per_via` is
 *  deliberately NOT set: the base case is the backend that predates the split and
 *  sends no such key, and each node opts in. */
function net(over: Partial<NetworkState> = {}): NetworkState {
  return {
    window_days: 7,
    total: 494,
    distinct_hosts: 89,
    by_decision: { allow: 494, deny: 0, 'grant-required': 0 },
    per_destination: [{ destination: 'api.github.com:443', host: 'api.github.com',
                        count: 435, denied: 0, grant_required: 0, scopes: 3,
                        last_at: '2026-11-22T14:33:46+00:00' }],
    per_scope: [{ scope: 'worker:deep-research', total: 435, denied: 0,
                  grant_required: 0, distinct_hosts: 12 }],
    database_present: true,
    policy: { telemetry: true, enforce: false, bash_record: false,
              allow_entries: 7, permanent_allow_entries: 7 },
    ...over,
  }
}

/** React separates adjacent JSX text nodes in the markup, and its comments and
 *  layout whitespace are React's, not the panel's — so a match reads on the words
 *  and "beside" can mean one element rather than one contiguous string. */
function text(html: string): string {
  return html.replace(/<!--[\s\S]*?-->/g, '').replace(/\s+/g, ' ')
}

/** The markup of each element that opens, so a claim that two things appear
 *  beside each other is about one element holding both. */
function elements(html: string): string[] {
  return html.split(/<(?:span|div)\b[^>]*>/).slice(1)
}

describe('NetworkPanel per-via split (#4292)', () => {
  // Clause 3: a seeded `bash_proxy` row puts the arm's own call count and a
  // recording-on word in the same element as the pre-existing policy word.
  it('renders the bash_proxy count and a recording-on word beside the policy word', () => {
    const html = text(renderToStaticMarkup(
      <NetworkPanel network={net({
        policy: { telemetry: true, enforce: true, bash_record: true,
                  allow_entries: 7, permanent_allow_entries: 7 },
        per_via: [TOOL, BASH],
      })} />,
    ))
    expect(html).toContain('bash recording')
    expect(html).toContain('31 bash calls')
    // "Beside" is the clause's word, so this is one element carrying the old
    // policy word, the recording word and the arm's count — and not the headline
    // figure: 494 is `total`, which egress.py computes over `via='tool'` rows, so
    // an element that showed `494` as the Bash arm's count would be labelling the
    // wrong measurement, which is the whole defect.
    const arm = elements(html).find(s => s.includes('enforcing')
                                     && s.includes('bash recording')
                                     && s.includes('31 bash calls'))
    expect(arm).toBeDefined()
    expect(arm).not.toContain('494')
  })

  // Clause 4, first arm of it: today's live shape — `per_via` present, one tool
  // row, recorder off. The panel still renders its body and reads the Bash arm as
  // off and zero, rather than omitting it or crashing on a missing row.
  it('reads the Bash arm as off and zero when per_via holds only a tool row', () => {
    const html = text(renderToStaticMarkup(
      <NetworkPanel network={net({ per_via: [TOOL] })} />,
    ))
    expect(html).toContain('bash off · 0 bash calls')
    expect(html).not.toContain('bash recording')
    // The body, not the missing-table placeholder: `database_present` is true
    // here, so the arm read sits on a rendered panel.
    expect(html).not.toContain('No egress table yet')
  })

  // Clause 4, second arm: a backend older than the split sends no `per_via` key
  // at all. `?? []` is what keeps the arm readable as zero instead of throwing
  // inside render and handing the whole panel to the section boundary.
  it('reads the Bash arm as off and zero when the backend sends no per_via', () => {
    const html = text(renderToStaticMarkup(<NetworkPanel network={net()} />))
    expect(html).toContain('bash off · 0 bash calls')
    expect(html).not.toContain('No egress table yet')
  })

  // The count and the switch are independent claims, and a reader has to be able
  // to tell them apart: the count is history the store holds, the word is the
  // live `bash_record` value. An arm that recorded and was then disarmed still
  // has destinations on record, and blanking that count because the switch is off
  // now would read as "Bash never reached anything".
  it('keeps a recorded bash_proxy count on screen while the recorder is off', () => {
    const html = text(renderToStaticMarkup(
      <NetworkPanel network={net({ per_via: [TOOL, BASH] })} />,
    ))
    expect(html).toContain('bash off · 31 bash calls')
    expect(html).not.toContain('bash recording')
  })

  // Why #4044 shipped `per_via` as its own key rather than folding the Bash rows
  // into the totals: the headline figures are one arm's, and the panel has to say
  // so on its face.
  it('names the arm the headline figures count', () => {
    const html = text(renderToStaticMarkup(
      <NetworkPanel network={net({ per_via: [TOOL, BASH] })} />,
    ))
    expect(html).toContain('494 tool calls')
    // And the two figures stay two claims: the arm's count never stands in for
    // the tool total in the figures element.
    const figures = elements(html).find(s => s.includes('494 tool calls'))
    expect(figures).toBeDefined()
    expect(figures).not.toContain('31 bash calls')
  })
})
