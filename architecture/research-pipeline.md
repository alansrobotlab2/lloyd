---
segment: architecture
tags: [architecture, lloyd, research]
type: reference
status: implemented
date: 2026-09-19
---

# The research registry

How a question Lloyd cannot answer becomes a note in the vault. The registry in
the middle is what this document is about — the store, its seven states, and the
two producers that fill it. The consumer at the bottom is a worker job and its
entry is in [[workers-jobs]]. Despite the shared name, the `autoresearch` source
is not part of this loop: it runs prompt-variant rounds and never opens this
registry — `workers/sources/deep_research.py` is the only job that claims a
topic.

```
  nightly task #65            a chat turn
  research-queue-generator      (human)
        │                          │
        └──── research_propose ────┘
                     │
                     ▼
        research.db — app/research_store.py
        queued → researching → written
                   │        └→ nothing_found | duplicate
                   │           archived (the legacy import)  rejected
   enqueue_if_due  │  next()
                   ▼
        workers.db ── deep-research source ── one session turn
                   │      KV-gated; deep-dive-research skill, deny list,
                   │      IV off, disk decides `written`  → [[workers-jobs]]
                   ▼
   knowledge/research/YYYY-MM-DD-slug.md  +  fact_add
                   │
                   └── finish() ──► research_list(since_days) feeds #65 tomorrow
```

---

## 1. What this replaced, and why the size was not the problem

The two halves used to be joined by a markdown checklist,
`~/obsidian/lloyd/research-queue.md`. By the time it was retired it held
**2,839 checked items of which 314 were unique**: a producer that no longer
exists had written the same 82 topics up to 390 times each, under fabricated
2024 and 2025 dates. The generator's skill said to read it *in full* every
night, which cost 1.02 million tokens a run and timed the task out until it
disabled itself on 2026-09-04. It is archived rather than deleted — it sits at
`~/obsidian/lloyd/research-queue-archive.md` behind an `Archived` header — and
`tests/test_research_doc_claims.py` pins both halves of staying retired:
neither skill still reads it, and no module under `workers/` holds the old path
as a live string. Several docstrings still recount what reading it cost, which
is why that check walks the AST and exempts them.

Bounding the file would have fixed that and left the real problems:

- **A checkbox cannot hold an outcome.** Researching, written, nothing found,
  duplicate, gave-up-after-two-tries — all of it had to be inferred from a
  tick that only ever meant "a turn ran".
- **There was no feedback edge.** The skill's instruction for a dead end was
  "log it", to a place that did not exist. So the same dead ends came back
  night after night.
- **A tick was not evidence.** The consumer, `domain-research`, ticked the box
  whether or not the turn produced anything. 90 of its 142 notes have the body
  `(no response)`, and every one of those topics is now closed on disk. The
  source was retired with this change on 2026-09-08 and is gone from both
  `config.yaml` and `SOURCE_REGISTRY`, but its notes are not: all 142 are still
  staged under `~/lloyd/_pipeline/vault-derived/pending-research/domain-research/`
  — `app/paths.py::VAULT_PENDING_RESEARCH_DIR`, under `LLOYD_HOME` and not in the
  vault — unpromoted, and untriaged since 09-08 (#1278). That is why
  `app/routers/workers.py::_DEFAULT_DEST` still maps
  `domain-research` to `knowledge` — the source is history, the leftovers are
  not, and deleting the entry makes them unpromotable from the Review tab.
- **Four writers, whole-file read-modify-write, no lock.** The same shape that
  produced the 2026-08-22 knowledge-graph wipe.

The 314 unique topics were imported as `archived` rows. They are the dedup
corpus and nothing else: `archived` never counts toward a daily budget and
never appears in `recent()`.

---

## 2. The registry

`app/research_store.py` owns `~/lloyd/research.db`. **Nothing else opens it.**

It follows `workers/queue.py` rather than `app/kg_store.py` — a short-lived
connection per call under one lock — because two processes write it: the
backend's worker and the aggregator's MCP tools. That is exactly the situation
`workers.db` already solves, and cross-process safety comes from the SQL, not
from the lock, which only one process holds.

| state | meaning |
|---|---|
| `queued` | proposed, waiting. `not_before` may hold it back after a failure |
| `researching` | a worker has it. `reclaim_stale` returns what a crashed turn left stuck here, swept at twice the source's `max_duration_seconds` |
| `written` | a note exists on disk, verified by `finish` itself: `_require_real_note` refuses any `artifact_path` that is not a file holding at least `MIN_NOTE_BYTES` (400) bytes, raising `ValueError` before the `UPDATE` so the row stays `researching` and `reclaim_stale` can retry it. The worker's `deep_research._note_is_real` applies the same bar and is now belt-and-braces (#1276) |
| `nothing_found` | searched, found nothing — a real answer, and the one that stops it coming back |
| `duplicate` | the vault already covers it; names what does |
| `archived` | imported from the retired checklist; dedup corpus only |
| `rejected` | should not have been proposed |

Two statements carry the concurrency, and both were written the naive way
first:

- **`propose` is one `INSERT ... ON CONFLICT(key) DO NOTHING RETURNING id`.**
  `WorkQueue.enqueue` still does SELECT-then-INSERT, and across processes both
  sides can see "not there"; the loser takes an `IntegrityError`, which inside
  the nightly generator is a tool error in a task that disables itself after
  three of them.
- **`claim` returns the row from the UPDATE's own `RETURNING`.** Updating and
  then re-reading cannot tell your claim from someone else's — the re-read
  sees `researching` either way. The two-process test caught that handing all
  thirty topics to both workers.

A third rule keeps a chat turn out of the same race. Of the five `research_*`
tools the aggregator advertises — `propose`, `next`, `complete`, `list`,
`stats` — only `research_next` reads the claimable queue, and it is
**peek-only**. Only the worker claims: a claim taken by a chat turn that then
wanders off leaves the topic `researching` until `reclaim_stale` sweeps it, for
no gain, since a human researching something by hand records it afterwards with
`research_complete` either way. `agent_mcp/annotations.py` classifies
`research_next` read-only on exactly that reasoning.

`propose` also **refuses** rather than queues once `MAX_QUEUED` (40) topics are
waiting, returning the depth and "research them before proposing more" in place
of an id. The generator meets the same wall from the other side — a full queue
from `research_stats` means propose nothing and stop — because a cap is only
useful if the producer reads it as an answer rather than an error to retry
around.

`key` is the topic normalised (NFKC, lowercase, punctuation to space, legacy
suffixes stripped) and **never truncated**. The retired source keyed on a
50-character slug, which collides in the real corpus: "Multi-agent task
decomposition — hierarchical planning for" and the same line ending "for
complex robotic workflows" cut to the same string.

`similar()` is token overlap in Python, not FTS5. A few hundred rows scan in
under a millisecond; an external-content FTS table needs three triggers to
stay in sync, which is a migration story bought before there is a problem. The
signature is what matters — swapping the body later changes no caller.

---

## 3. What the consumer does with a topic

`workers/sources/deep_research.py` claims one topic per tick, runs the vault's
`deep-dive-research` skill against it in a real session, and records the
outcome. That job — the `daily_max` ceiling, the KV gate, why disk rather than
the model decides `written`, the deny list it needs for fetching arbitrary web
pages, and why its retries live **here** rather than in `workers.db` — is
[[workers-jobs]] § `deep-research`.

The one rule worth repeating on this side of the seam, because it is what the
registry exists to carry: `workers/pool.py` records an in-band
`{"status": "failed"}` as a **completed** run (`mark_completed` at `:717`,
immediately after the `run_status == "failed"` branch decides the log level) and
then reports the outcome. Only a *raised* exception — a timeout or a drain —
takes the `status="failed"` run-record path at `:744` / `:773`. So a source that
returns `failed` and expects a queue retry does not get one, and `release()` /
`exhaust()` here are the retry ladder: `max_attempts` from `workers.max_attempts`
(3), backoff of one `interval_seconds` (3600 s), and a
`DrainActive` that releases with no backoff at all.

---

## 4. Correlation

A topic row carries `queue_id` (the `workers.db` item that researched it) and
`session_id` (the transcript). `runs.queue_id` is the join back to the run
record, and it is what `autonomy` leans on too: `/api/autonomy/health` reads
`WorkQueue.list_runs_joined` (`workers/queue.py:677`, a `LEFT JOIN` on
`r.queue_id = q.id`) and `_row_task_id` (`autonomy.py:1953`) recovers the task
id from the joined queue payload when the run row carries none. The pool never
passes its `run_id` into `execute`, so the queue id is the link.

---

## 5. Known limits

- **`similar()` is lexical.** Two topics that share no vocabulary but ask the
  same question will both be researched. The skill's rule to act on `similar`
  is what closes most of that gap, and it is a prompt, not a guarantee.
- **A `queued` topic is never expired automatically.** `stats()` reports
  `stale_queued` past 60 days (`STALE_QUEUED_DAYS`); retiring one stays a
  human's decision. That threshold is also loose enough to hide the ordinary
  state: at 40 queued against a `daily_max` of 3, the head can sit for nearly
  two weeks with `stale_queued: 0`, and nothing compares `queued` to
  `MAX_QUEUED` (#1277).
- **The generator is still an autonomy task**, so it inherits that path's
  timeout semantics rather than the pool's — #65 at `timeout_seconds: 1500`,
  under `scheduled-task`'s 3600 s pool cap so the task's own timer is the one
  that fires. [[autonomy-jobs]] § inbound signal. Its product is whatever the
  cap allows: the 09-19 run took 300 s and reported success after proposing 3
  topics and being refused at the 4th.
- **No UI.** `research_stats` and `research_list` from a chat, or
  `python -m app.research_store stats` for a shell. Not `sqlite3` on
  `research.db`, whatever else section 2 says about who opens that file. The
  Workers page's Recent Runs shows what the source did, because the run
  summary names the topic and its outcome.

## Review log

- **2026-09-20 — `current`.** §2's `written` row moved the verification into the
  store: `ResearchStore.finish` now refuses a `written` whose `artifact_path` is
  not a real file of at least `MIN_NOTE_BYTES` bytes, before it writes, so the
  topic stays `researching` and retryable. That closes #1276 from the other side
  to how it was filed: the worker had always checked disk, and the caller that
  did not was this tool surface — `research_complete` is denied to the research
  turn itself (`deep_research._DENY`), so the only unverified way to settle a
  topic was a chat turn naming a path nothing had opened, which is the
  tick-without-evidence §1 says the registry exists to end. What the refusal
  also protects is `propose`: it keys on the topic line and returns "already
  known as #N (written)" for a settled row, so one phantom note suppressed a
  topic permanently. The two sentences that recorded the old state are corrected
  above and in the 09-19 entry below, which was true when written and is now
  superseded.
- **2026-09-19.** Every mechanism in the unit is live and the
  registry is doing its job (391 topics, 33 `written` since the 09-08 cutover,
  last note 09-19 05:20Z, #65 succeeding nightly). Corrections: the retired
  source's staging root is `~/lloyd/_pipeline/...`, not vault-relative; the
  pool's inversion now cites `workers/pool.py:717` / `:744` / `:773` and names
  the retry ladder's real numbers; §4's `compute_health` analogy is stated as
  the joined-row task-id recovery it actually is; the shell read path is
  `python -m app.research_store stats`, not `sqlite3` on a file §2 says nothing
  else opens; and the `written` row said where the disk check lived, the worker
  rather than `finish`, which was true when written and is superseded by the
  09-20 entry above (#1276 closed from the store side). Filed: #1276 (store
  accepts an unverified `written`), #1277 (a queue pinned at `MAX_QUEUED` reads
  healthy), #1278 (142 untriaged leftovers hold a retired source's router entry).
