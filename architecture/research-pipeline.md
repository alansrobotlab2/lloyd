---
segment: architecture
tags: [architecture, lloyd, research]
type: reference
status: implemented
date: 2026-09-11
---

# The research registry

How a question Lloyd cannot answer becomes a note in the vault. The registry in
the middle is what this document is about — the store, its seven states, and the
two producers that fill it. The consumer at the bottom is a worker job and its
entry is in [[workers-jobs]].

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
  staged under `_pipeline/vault-derived/pending-research/domain-research/`,
  unpromoted. That is why `app/routers/workers.py::_DEFAULT_DEST` still maps
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
| `written` | a note exists on disk, verified |
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
`{"status": "failed"}` and then calls `mark_completed` on the queue item
regardless. Only a *raised* exception reaches the queue's backoff. So a source
that returns `failed` and expects a retry does not get one, and `release()` /
`exhaust()` here are the retry ladder.

---

## 4. Correlation

A topic row carries `queue_id` (the `workers.db` item that researched it) and
`session_id` (the transcript). `runs.queue_id` is the join back to the run
record, which is the same recovery `autonomy.compute_health` uses. The pool
never passes its `run_id` into `execute`, so the queue id is the link.

---

## 5. Known limits

- **`similar()` is lexical.** Two topics that share no vocabulary but ask the
  same question will both be researched. The skill's rule to act on `similar`
  is what closes most of that gap, and it is a prompt, not a guarantee.
- **A `queued` topic is never expired automatically.** `stats()` reports
  `stale_queued` past 60 days; retiring one stays a human's decision.
- **The generator is still an autonomy task**, so it inherits that path's
  timeout semantics rather than the pool's — #65 at `timeout_seconds: 1500`,
  under `scheduled-task`'s 3600 s pool cap so the task's own timer is the one
  that fires. [[autonomy-jobs]] § inbound signal.
- **No UI.** `research_stats` and `research_list` from a chat, or `sqlite3`.
  The Workers page's Recent Runs shows what the source did, because the run
  summary names the topic and its outcome.
