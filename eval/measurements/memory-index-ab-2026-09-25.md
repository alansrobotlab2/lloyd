# MEMORY.md as a typed index (P4), 2026-09-25

**Verdict: do not promote yet** — criterion (c) missed by one probe. The
ceiling stays at 73,728 B, `memory.render_overflow` stays `render_all`, the
vault is untouched and `scripts/maintenance/vault-memory-index-skills.patch`
stays unapplied. Typed entries and topic files (additive) are live.

Runner: `eval/run_memory_index_ab.py --with-trim-probes`, sandboxed
`run_bench_sdk` trials, blind grading; 50 probes × 3 arms. Arms: `canonical`
(MEMORY.md today, 72,562 chars in the prompt), `canonical_rep` (A/A), and
`indexed` (the consolidator's 20,104-char index + 24 topic files) — the whole
system prompt goes from 102,586 to 50,128 chars. Raw:
`eval/baselines/memory-index-ab-2026-09-25.json`.

| criterion | rule | result |
|---|---|---|
| (a) net loss within A/A noise | lost − gained ≤ A/A discordance | **ok** — net lost 0, A/A discordant 4 |
| (b) no feedback ruling lost | 0/10 | **ok** — 0 lost |
| (c) topic probes answered via `memory_read` | ≥ 8/10 | **miss** — 7/10 (all 10 passed; 3 were answered from the index line alone) |
| (d) live `memory_read` per user turn ≤ 0.5 | 7 d after deploy | pending (needs a deploy) |

## Reading

On answers alone the index is non-inferior (indexed failed 3 of 50 —
i05, f05, trim_p11; canonical failed 4; the A/A arm 4) at half the prompt.
The one miss is the pull-path check: three topic probes passed without a
`memory_read`, because the consolidator's index line carried the answer.
That is a probe-design question as much as a result, and promoting means
rewriting the live, synced MEMORY.md and two nightly skills in the vault —
so it is left for Alan: either accept (c) at 7/10 given 10/10 answered, or
tighten the three probes (i.e. keep the detail out of the index line) and
re-run. Deploy steps if accepted: one edit (`MEMORY_MD_CEILING_BYTES =
MEMORY_MD_INDEX_CEILING_BYTES`), `consolidate_memory_index.py` output written
into `~/obsidian/lloyd/`, `git -C ~/obsidian apply
scripts/maintenance/vault-memory-index-skills.patch`, restart both services.

## Tightened topic probes (for the re-run)

The three reads that did not happen were not index leaks. In the indexed arm's
traces, t02 (reaped round `SM_20260917_000427`), t07 (the HTTP create route
with no dedupe) and t08 (the halt that wrote no ledger row) were each answered
by grepping `promotions.jsonl`, git and the code: the facts are re-derivable
from the machine. t02's index line ("Automod round deaths outnumber round
starts…") carries nothing about the reaped round, and t08's claim is stale at
HEAD, since `ce2444bd` added `promotion_halt_set`/`_clear` rows. So the
consolidator stays as it is. A bold-lead hook is the entry's thesis, and the
ten index probes depend on it. The probes were re-pointed at incident history
instead:

| was | now | topic | answer term |
|---|---|---|---|
| t02_reaped_round | t02_gate_0908_suite_green | nightly-chain-agent-architecture-2026 | `2,495` |
| t07_http_create_no_dedupe | t07_djev_misdescribed | a-subsystem-with-no-memory-identity | `Django` |
| t08_halt_no_ledger_event | t08_bench010_quarantine | extracted-from-lloyd-claude-md-2026-09 | `quarantine` |

Every topic probe now names `answer_terms`, and `check_probe_anchors` refuses one
that the index, SOUL.md or USER.md carries (case-insensitive) or that is missing
from its topic file. On the rebuilt overlay (73,002 B → 20,466 B, unchanged),
all ten anchors and terms are absent from the prompt and present in their topic
file. Re-run (topic probes, all three arms), with the pool paused and the
primary lock held:

    flock <scratch>/primary.lock .venvs/lloyd/bin/python eval/run_memory_index_ab.py \
        --out eval/measurements/memory-index-ab-topics-rerun --only \
        t01_tencentdb_fragmented t02_gate_0908_suite_green t03_livekit_udp_proc \
        t04_release_watch_commits t05_task83_max_turns_floor t06_sub6s_failure \
        t07_djev_misdescribed t08_bench010_quarantine t09_merged_into_done \
        t10_clobber_cause_unknown

Deploy if (c) reads ≥ 8/10.
