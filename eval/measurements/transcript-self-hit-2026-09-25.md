# #1511 — transcript self-hits in prefetch's `<vault-context>` (2026-09-25)

**Verdict: fixed, deployed on** (`prefetch.exclude_self_transcripts: true`).

## What was wrong

`_export_session_markdown` writes every chat into qmd's `sessions` collection,
which prefetch's vault leg searches. A prompt sent before retrieves the note that
holds it, with the previous run's tool result and answer beneath it. Reproduced
before the fix: the E2E probe ("E2E harness check … Read app/harness/client.py …")
put `qmd://sessions/2026-09-25/20260925_082940_iv30d7.md` at score 1.00 on the
hybrid leg, snippet opening with the byte-identical prompt.

## The rule (`agent_mcp/transcript_self_hit.py`)

Only `sessions/` hits are judged. Dropped when the note is this session's own
export (`own_session`), or when one of its user turns holds the prompt
near-verbatim (`verbatim`: ≥ 0.8 of the prompt's word 3-shingles AND ≥ 0.95 of
its distinct words; prompts under 6 words are never judged verbatim). The word
test keeps a reused template with a new argument ("pull the youtube transcript …
<different video>"), which the shingle test alone dropped. Filter cost: p50
0.33 ms, p90 1.16 ms, max 4.77 ms (n=35 turns).

## Measurements

Raw: `~/lloyd-data/eval/1511/` (`audit-all-turns.json`, `audit-before-after.json`,
`probe.json`). Script: `eval/run_transcript_self_hit_audit.py`.

**Clause 3 — the denominator** (`audit --all-turns`): every user turn of every
conversation session on disk. Only 19 three-part sessions survive the 2026-09-22
data-home wipe, so n = 35 turns (31 non-probe, 4 probe). Replayed as a first turn
through the deployed vault leg (lex ladder + hybrid, fused):

| group | top hit is a transcript | top hit is a self-hit | any self-hit in the 5 |
|---|---|---|---|
| non-probe, before | 6/31 (0.09–0.36) | 2/31 (0.02–0.21), both `own_session` | 7/31 (0.11–0.40) |
| non-probe, after | 5/31 | 0/31 (0–0.11) | 0/31 |
| probe, before | 0/4 | 0/4 | 4/4 (0.51–1.0) |
| probe, after | 0/4 | 0/4 | 0/4 |

(Wilson 95% in brackets.) The probe echo no longer tops the block today only
because backlog #1511 itself — which quotes the prompt — now outranks it; the
echo was still in all four blocks.

**Clauses 1 and 2 — probe re-run** (`probe`, primary under `flock -s`): the 4
probe sessions' prompts rendered through `prefetch_context` and run to their first
tool call (not dispatched). Echo in the rendered block: 4/4 before, **0/4 after**.
Pass condition = the first tool call is a file read naming the requested path:
**4/4 `Read`** (answer text is not consulted).

**Non-regression, gold set** (79 queries with `expect_docs`, prefetch's fused
first-turn leg, paired): the filter touched 1 of 79 queries; doc_hit 0.241 →
0.241, MRR 0.124 → 0.124, paired diff 0.000 [0.000, 0.000] on both.
