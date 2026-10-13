---
segment: architecture
tags: [architecture, lloyd, workers, autonomy, inner-voice, sessions]
type: reference
status: implemented
date: 2026-10-13
---

# Background runs: recorded always, observed on request

Everything Lloyd does without being asked — scheduled autonomy tasks and the
worker sources in `architecture/workers.md` — now leaves the same record a
chat turn does: a session, a transcript, an event log, and a change ledger for
the files it wrote. Whether the Inner Voice observer *watches* one of those
runs is a separate, per-job decision. Chat history and background history are
split into two listings, and Mission Control has a **Background** tab for the
second one.

`CLAUDE.md` carries the short version under "Background runs are recorded;
watching them is opt-in". This is the long one.

---

## 1. Why

On 2026-09-10 an autonomy task was the prime suspect in a full wipe of
`~/obsidian`, and it could be neither confirmed nor cleared. Of the three
paths that run an agent loop unattended, only one persisted anything:

| path | before | after |
|---|---|---|
| `autonomy.run_task` | a 200-character summary in `autonomy-runs/<id>/<run>.md` | session, transcript, event log, change ledger |
| `_common.run_prompt_on_primary` | the collected text, nothing else | session, transcript, event log, change ledger |
| `_common.run_prompt_in_session` | session, transcript, event log | unchanged, plus the #534 grant gate |

The first two call `run_query` directly and consume the event stream
themselves. The tool calls they made — the thing you would want to read after
an incident — were never written anywhere.

## 2. Scale, measured

From `workers.db` for the 7 days to 2026-09-10 (`completed_at` 09-03 17:57 →
09-10 13:35), and the session directory on the same day:

| | week | per day |
|---|---:|---:|
| all background runs | 2,115 | 302 |
| `scheduled-task` (autonomy) | 851 | 122 |
| direct-path runs that used to leave nothing (`scheduled-task` + `session-distill` 337 + `bench-mine` 24; `gap-fill` 0) | 1,212 | 173 |
| session-backed worker runs, current names (`youtube-digest` 442, `autotriage` 19, `autocode` 15, `deep-research` 11) | 487 | 70 |
| worker sessions created | 479 | 68 |
| chat sessions created | 99 | 14 |

The session directory held 643 files: 479 `worker`, 161 chats (10 with
`platform: mission-control`, 151 with none, which reads the same), and 3
`e2e-harness`. **22 of the newest 24 by mtime were already background** before
a single autonomy run had been recorded. Recording adds ~173 a day more, so
the directory's growth is now roughly 240 background sessions a day against
14 chats. Most of what follows exists because of that ratio.

Hold that number against §8's "~300 background sessions a day", because they
are two different quantities and the page had better not blur them: ~240/day is
sessions *created* (the ~173 direct-path runs plus 68 session-backed worker
sessions from the table above), while the 306 and 298 in §8 are files whose
mtime *fell* on a day, counted by walking `sessions/` — the second also counts a
backfill or an edit touching an old file. Both are 2026-09 snapshots. Measured
again 2026-10-13 by the same directory walk: 266 sessions in the preceding 24 h
and 451 in the 24 h before that, of which 24 and 36 are autoresearch bench
trials, a class that did not exist as a writer when these tables were cut
(`run_bench_sdk` started recording on 2026-09-14 — §12).

## 3. Two axes

**Recording** and **observation** are different things and are switched
separately.

- *Recording* is cheap and universal: a few file appends per event. It is on
  for every background run, and its only switch is the kill switch
  `harness.background_recording.enabled`.
- *Observation* is the Inner Voice critic. It runs on the PRIMARY at priority
  1 and spends a goal-extraction call plus a critique per turn, in front of
  whatever a human is typing. It is opt-in per job (§9).

Conflating them was the design error to avoid. Turning observation on for
everything to get a transcript would have put ~300 critiques a day on the
primary; leaving recording tied to observation would have left most runs
unrecorded.

---

## 4. The recorder

`app/run_recorder.py::record_events(events, *, session_id, turn_id, prompt,
model, source, disallowed_tools)` is an async generator (`disallowed_tools` is
the writing turn's deny list, so a spilled tool result's recovery sentence
offers only what that run can call — #1066). It wraps an existing `run_query(...)`
stream, persists each event, and **re-yields every event unchanged**:

```python
async for evt in record_events(run_query(messages, options),
                               session_id=sid, turn_id=run_id,
                               prompt=prompt, source="autonomy"):
    ...                # the caller's loop, unchanged
```

It is a second reader of the stream, not a new owner of it. The caller's
`RunOptions`, `HookRegistry` and bookkeeping are untouched — which is the
whole reason it exists in this shape. The alternative, routing the direct
paths through `POST /api/message/stream` so the chat path persists them, was
tried on a stale `lloyd-sandbox` branch (`autonomy-sessions` @ `747a47e`, do
not merge) and costs two things: the chat endpoint does not install
`run_task`'s #534 policy hook on the caller's behalf, so the gate is lost; and
`saw_tool_call`, `tool_errors` and the timeout partial have to be re-plumbed
across the wire to get back what a direct `async for` already has.

Three properties are load-bearing:

- **Incremental.** Each thinking phase, tool pair and text segment is appended
  as it arrives. Nothing waits for the `result` event, because a run killed at
  its deadline never emits one — and that is how the runs worth reading end.
- **The final flush is shielded.** On any exit without a `result`, the
  recorder writes the unpersisted tool pairs and the partial text (marked
  `cancelled: true`) and logs `background.run_interrupted`. That write runs in
  `asyncio.shield(ensure_future(...))`, because the common way a background
  run ends is its own `asyncio.timeout` cancelling this task, and an
  unshielded await would be cancelled before it wrote anything.
- **Recording may never break the run.** Every append and every event-log
  write is wrapped; a failure logs a warning and the run continues. A full
  disk costs the record, not the work.

Events are mirrored into the event log under the same `brain1.*` names the
chat path uses (`user_prompt_received`, `query_started`,
`thinking_block_emitted`, `tool_call_proposed`, `tool_result_received`,
`result_message`), so the Inner Voice reader's raw event view works on either.

The recorder also writes the **usage row** these two paths never had, and
folds their prefix-cache misses on the way past. `_RunRecorder` holds a
`prefix_miss.TurnMissTracker` for the turn, feeds it each
`assistant_message`, and on `result` calls `usage_store.record_usage` with
the totals — the same accounting the chat path does, through the same
module, rather than a second private copy. Before this, `usage.db` and the
dashboard's token panel had never seen an autonomy task or a
`run_prompt_on_primary` job at all, so a per-turn miss count would have had
nowhere to live for two of the three paths that run an agent loop. Like
every other append here it is wrapped: a failed usage row costs the
accounting, not the run.

## 5. One transcript shape

`app/transcript_entries.py` owns the shape of every row a turn writes:
assistant text, tool call, tool result, thinking phase, and the opening user
prompt. It was built inline in `app/routers/messages.py` — five assistant
variants and three tool-pair variants across the streaming, cancel and error
paths — and a second writer would have needed a second private copy. Both
writers call the builders now, and `tests/test_transcript_entries.py` feeds
the same events to each and asserts the entries are identical.

Five fields are conditional rather than always-present, because every session
on disk predates them: `summary` on a tool call is omitted when empty, and
`is_error` and `raw_chars` on a tool result's stats are omitted when the
writer did not see the answer — the two reconstruct-from-the-log paths never
did, and writing `false` there would turn "we never saw it" into "it was
fine" (`raw_chars` is the length *before* the harness truncated or spilled
it, which is the only size that survives a spill, since `result_chars`
saturates at the cap). `persisted_path` is present only on a row that is a
pointer, and `turn_id` on a user, tool-call or tool-result row is omitted when
the writer does not know it. `role="thinking"` rows keep `content` empty; the
role is what keeps reasoning out of every transcript producer (see
`architecture/inner-voice.md` and `tests/test_thinking_trace_transcripts.py`).

A tool result over `TOOL_RESULT_MAX_CHARS` (2000) is no longer written cut
down to size. Since D1 (2026-09-24) `shape_tool_result_for_transcript` hands
it to the same `maybe_spill` the live loop uses, at that lower threshold, so
the row carries a `<persisted-output>` pointer into
`<sid>.tool-results/<call_id>.{txt,json}` and `stats.persisted_path` names the
file — one vocabulary with the live turn, which is also why microcompact can
shrink the row to its header without losing the path. The old cut at 2000 with
`...(truncated)` appended survives as the fallback only: spill switched off
(`compaction.transcript_spill.enabled`), no session or call id to name a file
by, or the spill write failing. So the transcript is faithful about *which*
calls were made, and a long result is one hop away rather than gone: an audit
that needs the whole of one follows the pointer.

## 6. One session writer

`sessions_io.create_session(session_id, *, platform, model, title, source,
inner_voice, preview)` creates every non-chat session. `workers.sources._common.
new_worker_session` goes through it too; it had its own copy of the shape,
which already disagreed with the chat path's — `id` where the chat path writes
`session_id`, and no `last_active`, so a worker session sorted by file mtime
while every chat sorted by its conversation. Both keys are written now.

| producer | id | `platform` | `source` | title |
|---|---|---|---|---|
| autonomy | `YYYYMMDD_HHMMSS_autonomy_<4hex>` | `autonomy` | `autonomy-task:<id>` | `#<id> <task name>` |
| direct-path worker | `YYYYMMDD_HHMMSS_<source>_<4hex>` | `worker` | the source's `NAME` | set by the caller |
| session-backed worker | `YYYYMMDD_HHMMSS_<source>_<4hex>` | `worker` | the source's `NAME` | set by the caller |

Ids are minted by `new_background_session_id(slug)` in local time, like chat
ids. The `source` slug is the source name with every non-alphanumeric
character dropped — which is what stops a slug smuggling an underscore in and
making a four-part id parse as five — and then cut to 12 characters.

**Titles are set at creation**, for two reasons that apply to different
sessions. A direct-path run never goes near the LLM titler — it is fired from
the chat path only — so the title set here is the only one it will ever have.
A session-backed worker does go through the chat path, and
`session_titles.should_title` now refuses it: the titler runs on the
single-tenant secondary, where every chat turn already queues, and those would
be ~70 model calls a day to relabel rows nobody asked about.

## 7. Joins

Four records per run, and each can reach the others.

- **Run record → transcript.** An autonomy run's `autonomy-runs/<id>/<run>.md`
  frontmatter carries `session_id` on every outcome: success, timeout,
  cancellation, empty response, exception.
- **Worker run row → transcripts.** `runs.meta_json.session_ids` lists every
  session the claimed job created. The pool binds an empty list in
  `sessions_io.current_run_sessions` around the job — beside `current_scope`
  and `current_effect_scope` — and `create_session` appends to it. Collected,
  not returned by the source: "the handler remembered to pass it back" is not
  a property worth depending on eleven times. The pool writes it on the timeout
  and exception branches too, and a session-backed source stamps a singular
  `session_id` into the same blob, so a reader must union the two keys or show
  nothing for the sources that only ever write one.
  **Where a human follows it: Background tab → Sources → a run row → the Inner
  Voice reader.** `app/routers/workers.py::run_session_ids` does that union and
  ships it as a top-level `session_ids` array on every row of
  `/api/workers/runs` and every `recent` row of `/api/workers/health`;
  `web/src/lib/runSessions.ts::runTranscriptIds` renders one control per
  transcript on the row, and activating one opens the session through the same
  `setPendingFocus` + `setCurrentTab` pair §11's Runs bullet uses, so a session
  the reader's own list has never seen still opens. An empty array is the honest
  answer, not a gap: §12 lists the sources that record no transcript by design —
  `automod-regression` among them (autoresearch *does* record, since
  `4abc079a`) — and a run that dies before
  `create_session` names none, so those rows render no control.
  The write half long predates the read half — until 2026-09-24 nothing parsed
  either key, and this bullet's claim that one record of a run names the
  transcripts it produced could only be followed by hand-typing an id into the
  reader. `tests/test_workers_router.py` pins the endpoint's field and
  `web/src/lib/runSessions.test.ts` the derivation plus the page's wiring to it.
- **Transcript → run.** The event log's `turn_id` is the autonomy `run_id`
  (`run_<task>_<utc stamp>`) or a fresh 12-hex id for a direct-path worker.
- **Change ledger.** Both direct paths now set `RunOptions.session_id` and
  `RunOptions.turn_id`. `turn_id` is the load-bearing one: it is what the
  aggregator's `agent_mcp/_change_ledger.py` keys on, so every file an
  unattended run writes now leaves a pre-image under
  `sessions/<sid>.changes/<turn_id>/` and can be reverted through
  `POST :8500/changes/revert`. These were the only turn paths with no undo.
  `session_id` routes the run's tool-result spills under its own session.

---

## 8. The history split

### Two predicates, because the two questions are not one

The split rests on **two** decisions in `sessions_io`, deliberately kept apart
because inverting one of them breaks the other.

- `NON_USER_PLATFORMS` = (`autonomy`, `worker`) with `is_user_session(data)` —
  a **deny-list**, so a platform the list has never heard of keeps receiving its
  ambient briefs rather than silently losing them. It answers "may this session
  receive a brief / is this ever *the user's* session", and nothing else.
- `INTERACTIVE_PLATFORMS` = (`mission-control`, `browser`) with
  `is_conversation_session(name, data)` — an **allow-list**, and it additionally
  refuses any four-part id whatever its platform says. It answers "is this a
  conversation a human will read", which is the only question that may decide
  being *listed* and being *embedded*.

The two lists cannot be collapsed into one bool, and the reason is on disk:
`e2e-harness` and the gate's `canary` are machine platforms no one ever meant to
deliver a brief to, yet they are absent from `NON_USER_PLATFORMS` — so under the
deny-list alone they read as user sessions, were listed in the chat history, and
were exported into the corpus qmd embeds, while a four-part run stamped
`mission-control` was dropped by both listings and appeared nowhere. Flipping
`NON_USER_PLATFORMS` into an allow-list to fix the second would silently stop
the first receiving briefs. An unknown platform is therefore still eligible for a
brief and still not a conversation; `POST /api/sessions/create` refuses an
unknown `platform` outright so the set stops growing by accident.

Six readers once hand-rolled `data.get("platform") == "autonomy"` and none of them
had learned about `worker`. Worker turns arrive through the chat path, so the
sessions this machine ran for itself were:

| reader | file | effect of the gap |
|---|---|---|
| chat history | `app/routers/sessions.py` | listed as the user's conversations |
| recent chats | `app/routers/dashboard.py` | counted on the landing tab |
| LLM titler | `app/session_titles.py` | titled on the secondary |
| post-capture | `app/post_capture.py` | summarised into the daily note, fact-extracted |
| session recall | `agent_mcp/session.py` | recalled to the model as the user's own words |
| agent's view of the chat tab | `app/routers/mc_ui.py` | described to the model as chats |

The four that decide *listing, titling, embedding and recall* now call
`is_conversation_session`; the ones that decide *delivery* —
`workers/handlers/ambient.py`, the Chat/Background tab summaries, the active-
session resolver — keep calling `is_user_session`, which is where the deny-list's
generosity belongs. `tests/test_session_platform_checks.py` greps `app/`,
`agent_mcp/`, `workers/` and `scripts/` for the literal, because a seventh reader
written next month is how this comes back — silently, since a background session
in the history looks like a session. `scripts/` was added by #1064: the backfill
exporter was reader number seven all along, sitting outside the swept roots with
its own `== "autonomy"`, and had written 469 `worker` transcripts into the
embedded chat corpus. The retention sweep restates the tuple (it is stdlib-only
and runs from cron with no venv), and the same test pins that the two agree.


### The id shape is a fast path — and in the chat listings, it decides

A background id has four underscore-separated parts
(`20260910_120001_autocode_9f2a`); a chat's has three
(`<ts>_<6hex>` from the chat path, `<ts>_iv<4hex>` from `POST /api/sessions/create`).
`sessions_io.is_background_session_name` tells them apart without opening the
file, and at ~300 background sessions a day against ~10 chats (measured over
2026-09-10 to 09-12: 178, 306, 298 background against 11, 13, 2 chats) that is
the difference between a listing that is bounded and one that grows with the
fleet's throughput.

Be precise about what that means, because it is not the same in every
listing:

- **`/api/sessions` and `_scan_recent_sessions` skip a four-part id unread.**
  For that shape the name is the whole decision, not a hint. That is safe only
  while nothing that creates a *user* session mints a four-part id, and
  `test_no_user_session_creator_mints_a_background_shaped_id` pins it at the
  creators: exactly three `session_id = f"..."` mints in the swept packages,
  two in the chat router and one in `POST /api/sessions/create`, both three-part.
  `tests/test_api_contracts.py` checks the create endpoint end to end.
- **Three-part ids are parsed and judged by `is_conversation_session`.** A
  chat-shaped name carrying a platform off the allow-list is still excluded — the
  3 `e2e-harness` automod smokes and the gate's `canary` turns are that shape, and
  until #1064 they sat in the chat history and the embedded corpus because the
  deny-list had never heard of them. The other side of that census is on disk
  too: 54 sessions
  carry `platform: browser` (ids `<ts>_iv<4hex>`, minted by the
  Chrome extension's side panel, which posts `platform: "browser",
  inner_voice: true` to `POST /api/sessions/create` from
  `chrome-extension/src/background/lloyd-client.ts:15`). These are not harness
  runs: the panel sends a real user turn per follow-up a person types, which is
  why `browser` is on the allow-list and they stay in the chat history and the
  embedded `sessions/` corpus; since #1143 the trajectory miner keeps them under
  their own `browser` class. An earlier draft of this page ascribed them to the
  observer's own bench harness — the same wrong premise as #493's — and the count
  above is a `platform` census of `sessions/`, not a claim about who typed. The
  endpoint that mints them now refuses a `platform` outside the named set, so a
  new caller has to say what it is instead of defaulting into the human corpus.
- **`/api/background/sessions` is the complement, not a name-based mirror.** It
  lists whatever `is_conversation_session` refuses, so the two listings cannot both
  show a session and cannot both hide it — hiding was the live failure, and it is why
  this endpoint no longer skips a three-part name unread. It also catches a
  chat-shaped run nobody classified (`e2e-harness`, `canary`), which the four-part
  name rule could not see at all.
  Membership is decided in two phases and the split is the substance: a four-part
  name is settled by the name rule — the shape answers membership, and that is what
  makes the second phase affordable — and the read budget
  (`_BACKGROUND_PLATFORM_SCAN_CEILING`, 600) is spent on the ambiguous chat-shaped
  half, 201 of the 4,045 files in `sessions/` on 2026-09-21. "Settled by the name"
  means membership needs no read; it does not mean the file goes unopened. A
  four-part candidate is still opened, 2 KB at a time, for two fields that exist
  only in the bytes: `title`/`preview` for the row, and the `source` the tab shows a
  mis-labelled run under — `source` is written on a background row and blanked on a
  chat row, so an unclassified four-part session carries no platform and no source
  and only its own file can say what it is. What the budget refuses is spending
  *parsing* reads in proportion to the directory: the ceiling caps the half whose
  membership the name cannot settle, membership is never capped. Be honest about
  what that leaves open — the four-part half is opened uncapped by design, one 2 KB
  head each, and that head is now the bulk of the cost: measured 2026-10-13 over
  `sessions/`, **10,514 four-part files, 20.3 MiB, 153 ms per walk**, against the
  55 ms over 3,854 files the endpoint's own comment still quotes from 2026-09-21,
  and `BackgroundPage.tsx` polls it every 10 s (#2646).
  Budgeting *reads* rather
  than *reach* is what makes this hold on live data instead of only in a test: a rule
  bounded by a scan window silently stops applying to everything older than the
  window, and all 12 orphan sessions below sit past rank 600 in mtime order (newest
  at rank 1234, oldest at rank 3630), so the previous "open the newest 600, judge
  what you read" loop could not have returned one of them with the predicate fixed
  or not. Measured over those 4,045 files at the tab's own limit: 521 ms for the
  old loop, which reached 594 candidate rows in its 600 reads, and 401 ms for the
  two-phase one, which finds 3,848 candidates and builds 150 rows.

What that changed was on disk: twelve sessions in four-part ids carried
`platform: mission-control` — eight named `20260909_1538xx_autonomy_<4hex>`,
created within 11 seconds by the sandbox experiment above, plus four minted
since, newest `20260917_105134_autocode_029c`. The chat listings skip a four-part
name unread, and the Background listing used to drop them for being
user-labelled, so they appeared in neither. They became the Background tab's,
and `_save_session_meta` stamps a newly created four-part session `worker`
instead, so no fresh run joins the set. They hold one or two user rows each and
are not conversations, which is why the Background tab is where they belong.

**That census no longer reproduces.** Re-measured 2026-10-13 over every readable
head in `sessions/`: no four-part id on live data carries a human platform at
all, and **no writer records a member's departure** — the sweep gzips and never
deletes, nothing rewrites the `platform` of an *existing* session
(`_save_session_meta` stamps it only in its create branch,
`app/sessions_io.py:1327-1345`), and `DELETE /api/sessions/{session_id}`
(`app/routers/sessions.py:1003`, unlink at `:1035`) is the only in-tree route
that removes a transcript — so this page records the emptiness and the missing
witness and asserts no cause for the vanished twelve, one of which,
`20260917_105134_autocode_029c`, exists under no suffix at all (#2651).

The eight `20260909_1538xx_autonomy_*` ids named above were never members of this
class. They were sorted by their *filename* — the `_autonomy_` in the name is
where the membership came from — while the head of each has always said
`platform: worker`, and measured 2026-10-13 all 42 archived four-part files on
the box report `platform: worker`, without exception. They are ordinary `worker`
runs (`source: autonomy-task` in every one of those heads), archived on the
machine age exactly like the background runs they always were.

What the exemption promises is longer than this page used to say, and the reason
is the field the sweep reads. `_archive_age_for`
(`scripts/groundskeeper/retention-sweep.py:607-619`) takes the archive age from
`platform`: `BACKGROUND_SESSION_ARCHIVE_AGE_DAYS` (30) for
`NON_USER_PLATFORMS = ("autonomy", "worker")`, `SESSION_ARCHIVE_AGE_DAYS` (90)
for everything else — including an unreadable head, which keeps the longer
window. This class is *defined* by a platform word outside that tuple:
`names_a_machine_platform` (`app/sessions_io.py:241-259`) returns exactly that
membership test, and its own docstring says the class is identified by failing
it. The sweep's predicate and the endpoint's exemption are complements over one
field, so the 30-day background age cannot reach a file the exemption exists for.
A member therefore stays in the Background listing for as long as its file is
live, and leaves every `*.json` listing only when the sweep gzips it at the
**90-day conversation age** (#2647, corrected by #2651). Treat the twelve as the
incident that motivated the rule, not as live rows to go looking for.

`_scan_recent_sessions` used to keep the newest 24 files by mtime and only
then drop non-user rows, which with 22 of the newest 24 already background
rendered a nearly empty panel. It now walks mtime order, skips by name, and
stops at `_RECENT_KEPT` user rows or `_RECENT_CEILING` (400) files opened.

The fifth reader is `mc_ui._summarize_chat`, the Chat tab's summary. It is
reached only from `mc_navigate` (`post_mc_navigate` → `_summarize_tab`); the
frontend's debounced state report never runs it. It walks mtime order, skips a
background-named file unread like the scans above (#1228), and stops at five
chat rows or 400 files opened, so its cost scales with the background share of
the directory rather than its size: 5 files opened and ~5 ms against 580
session files on 2026-09-24. `total_session_count` is a second glob, not a read.

### Post-capture and the titler

`_post_session_capture` and `maybe_title_session` are fired from
`app/routers/messages.py::_run_turn` and nowhere else, so the only background
sessions that ever reach them are the session-backed workers — ~200 a day
(worker sessions created: 142, 203, 212 on 2026-09-10 to 09-12).

- **The markdown export still runs for them**, into
  `_pipeline/vault-derived/sessions-background/` instead of `sessions/`.
  `agent-services/scripts/qmd-watcher.sh` indexes and *embeds* `sessions/` on
  every change; ~70 transcripts a day of the machine talking to itself against
  14 chats would drown the corpus that answers "what did we discuss".
  `sessions-background/` is outside the watch: exported, greppable, not
  embedded.
- **Then it stops**, marks the session `captured`, and skips the secondary
  summary, the daily note and fact extraction — all three are about the
  user's day.
- **Direct-path runs are not exported at all.** Their record is the session
  JSON, which the Background tab reads and `grep` can read. A second copy in
  the vault would buy a nicer grep target and cost a vault write per run, on
  a path whose rule is that recording must never be able to break the run.

### Retention

`scripts/groundskeeper/retention-sweep.py` gzips a background session after 30
days of inactivity, a conversation after 90. Same gzip-never-delete rule: an
archived run is out of every listing and still recoverable. The platform is
read from the file's first 4 KB, and an unreadable head keeps the *longer*
window — a few stale kilobytes beat losing a conversation from the history
two months early. That choice is a platform word and never the id shape, so a
four-part transcript carrying a human platform sits on the 90-day side whatever
its name looks like (§8). `workers.db` is on the same sweep: the `runs` table loses rows
whose `completed_at` is older than 30 days (`WORKER_RUN_MAX_AGE_DAYS`), reaching the
database through `workers.queue.configured_db_path()` so the sweep and the queue can
never name different files, and a database the live pool is holding reports
`SKIPPED (database locked …)` with exit 0 instead of a traceback. It is a DELETE with
no VACUUM — freed pages are reused by the next INSERT, which is what bounds the file,
and VACUUM would take an exclusive lock on a WAL database the pool is writing. The
sibling `queue` table is pruned on the same horizon since #1466: rows in a terminal
state (`completed`, `poisoned`, `quarantined`) older than 30 days
(`QUEUE_MAX_AGE_DAYS`) are deleted, and a live row (`queued`, `claimed`, `running`)
never is, whatever its age.

---

## 9. Observation, per job

### Worker sources

`workers.sources.<name>.inner_voice`, read through
`_common.source_inner_voice(source)`, is the only switch.
`run_prompt_in_session`'s `inner_voice` argument now defaults to `None`
meaning "ask the config", and no session-backed source passes it:
`deep-research` passed `inner_voice=False` as a literal until 2026-09-10,
which is to say it was not a setting, and a per-source switch a caller can
override with a literal reads as broken the one time somebody uses it.

| source | `inner_voice` | why |
|---|---|---|
| `autocode` | **true** since 2026-09-25 | rewrites production — was `true`, off 2026-09-12, back on with the worker harms fixed inside Inner Voice |
| `autotriage` | **true** since 2026-09-25 | judges items the loop will then implement; back on with `autocode` |
| `youtube-digest` | false | evaluates untrusted transcripts and files backlog items from them |
| `deep-research` | false | a human reads the note before anything acts on it |
| `arch-review` | false | reviews docs and edits the one doc it was given (`f80c9d0`) |
| `board-steward` | false | moves backlog statuses (`6d59b7c`); unset until #1015, so it was observed |
| `owed-check` | false | the seventh session-backed source; keyed off in config.yaml, and the table's "six" predates it |

**All six were off** from 2026-09-12 (`97a86cc`, cut 1 of
senses-not-supervision; `board-steward` since #1015), pinned by
`tests/test_background_inner_voice.py`:
the observer's measured effect on unattended turns was negative — round 874
abandoned at iteration 38 on an invented premise, sixteen false repetition
fires in one day — and what it provided there is now done by the context
anchors and the gate. **Since 2026-09-25 `autocode` and `autotriage` are on
again** (Alan: fix it inside Inner Voice, don't switch it off): every worker
review carries a PLATFORM/round note and a rail rewrites any "deliver the
report" inject — [[inner-voice]] § "Workers: back on for autocode and
autotriage". The override file wins over config.yaml for this key. The
deterministic turn guards (`app/harness/turn_guards.py`) run on every turn
regardless of the switch. Recording is untouched: all of these are still real
sessions in the Background tab.

These are the sources that call `run_prompt_in_session`, and only a turn that
runs through the chat endpoint can be observed at all. A session-backed source
that sets no key is **not observed**: `source_inner_voice`'s fallback is
`False` since #1015, so opting *in* is the thing config.yaml has to say. Until
then the fallback was `True`, and `board-steward`, added with no key, was
watched on primary every 15 minutes against the measured default while the
tests that should have caught it named five sources and four files by hand.
The test now discovers the session sources from `workers/sources/*.py` and
resolves each against the shipped config, so a source added without a key —
an eighth, once `owed-check` is counted — fails it.

`/api/workers/health` reports the key tri-state for the same reason — `null`
means "not set", which for a direct-path source means "not observable", and a
flat `false` would invite a knob that reads as broken. **Until 2026-09-11 the
key was set only on those four**, which is what made the tri-state honest.
`backlog-cluster` arrived on 2026-09-11 (`30be051`, the nightly clustering
source) carrying `inner_voice: false` on a source with no agent turn at all —
its `execute` is a `build_clusters` call in a thread — so health rendered it
`false` where `null` is the honest answer. #1015 removed the key, and with the
fallback now `False` a `null` on a session source means off, not observed.

It is UI-mutable through `data/tool_overrides.yaml` like `workers.enabled`;
the override merge honours that one key per source, refuses to introduce a
source that config.yaml does not already name, and touches nothing else.

### Autonomy tasks

A task file's `inner_voice:` frontmatter, falling back to
`autonomy.inner_voice` in config.yaml, which ships **off** — at ~120 runs a
day, a fleet default of on would put that many goal extractions and critiques
on the primary. Frontmatter wins in both directions, YAML-ish strings from
the degraded parser (`"true"`, `"off"`) read as the booleans they look like,
and the key is in `_parse_task_file`'s `fallback_fields` so a file that
needed the regex fallback keeps its opt-in. The fleet default is read through
`app.config.CONFIG`, not a raw file read, so a canary resolves its own overlay.

### How the flags reach the session

`create_session` writes `inner_voice_evaluate_user_turns` equal to `inner_voice`
when the caller passes no separate value, which is what every background path
does — `new_worker_session` hands it the job's one switch, and
`run_prompt_on_primary` creates its session with both off. The parameter exists
because the two are no longer the same decision for a *created* chat stub
(`POST /api/sessions/create` takes them apart, and refuses
`inner_voice_evaluate_user_turns: true` without `inner_voice: true`), so read
"equal by default" here, not "one field". They still have to move together
for a background job, because the two kinds of turn arrive differently: a worker's prompt is posted
to the chat endpoint as a *user* turn, where the observer fires only if
`inner_voice_evaluate_user_turns` is set; an autonomy run attaches with
`turn_source="ambient"`, where the master flag alone decides. A
`run_prompt_on_primary` session is always created with both off.

### The observer on the direct path

`autonomy.run_task` calls `attach_observer_for_turn(turn_source="ambient",
producer_source="autonomy", options=options, chat_messages_handle=messages,
cancel_event=...)` when the task opted in.

- **It keeps the grant gate.** `attach_observer_for_turn` creates a
  `HookRegistry` only when `options.hooks` is empty, so passing `run_task`'s
  own `task_hooks` adds the observer to the registry the #534 policy hook is
  already on. `tests/test_background_inner_voice.py` asserts both are there.
- **Its `cancel` lever reaches the loop.** `options.cancel_event` is wired to
  the observer's event; on the chat path `_run_turn` does the equivalent, and
  this path had nothing. For unattended work, a run going somewhere it should
  not is the case the lever exists for.
- **Its ambient and clarify callbacks are `None`.** Both exist to reach a
  human mid-turn, and nobody is reading. The observer's two clarify-driven
  cancels are both gated on the callback being set, so neither can fire here.
- **It closes in a `finally`**, however the run ends, and a failure to attach
  logs a warning and runs the task unobserved: watching is not the run.

Importing a router helper into `app/autonomy.py` is a layering smell, accepted
rather than relocated — the alternative is a second definition of how a turn
is watched.

---

## 10. The grant gate belongs to the endpoint

#534 gates tier-2 (durable-external) and tier-3 (hard-to-reverse) tools on a
live grant. It is a PreToolUse hook, so it exists only where whoever built the
turn's `HookRegistry` installed it. `autonomy.run_task` and
`_common._worker_run_options` did. `POST /api/message/stream` installs Inner
Voice, the destructive-Bash safety hook and skill dispatch, and did not — and
every session-backed worker posts there. Nothing re-checks at the tool layer.
So autocode, autotriage, deep-research and youtube-digest — the four that read
untrusted text or rewrite this repo — ran ungated, while the two paths that
built their own registry were covered.

`messages._authority_scope_for(session_id, data)` decides, and the caller
installs:

- **The trigger is the session's own platform.** A caller cannot forget,
  because a caller is not asked. A missing session file reads as a user
  session — it is a chat being created by this very request, and gating it
  would take the UI down. The fail-closed half lives inside the policy hook,
  which denies when the grant store cannot be evaluated.
- **A payload `grant_scope` also arms it, and wins.** `policy.current_scope`
  is a contextvar the pool binds around the claimed job; it is correct in the
  pool's task and gone across the loopback POST, so `run_prompt_in_session`
  sends it. It is the difference between a grant made to `autonomy-task:39`
  and one made to whatever runs on that source next. Without it the scope is
  `worker:<source>`.
- **`grant_create` is banned on those turns**, written back into the request
  body's `extra_disallowed` because `_refresh_disallowed_for_session` re-reads
  that key on every harness iteration. Enforced twice on purpose: not
  advertised, and denied by the hook if a local model emits the name anyway.
- **The automod tools come off every other worker turn the same way**
  (#709, `messages._ban_automod_for_workers`). `WORKER_AUTOMOD_BAN` used to
  be a per-call-site convention — `run_prompt_on_primary` baked it in, a
  session-backed source passed it as `extra_disallowed` or forgot to, and
  single and group triage forgot — so the source that decides which items get
  implemented was advertised `automod_start`. Now the endpoint unions the ban
  for any `NON_USER_PLATFORMS` session whose `source` is not in
  `AUTOMOD_DRIVER_SOURCES` (`autocode`, whose turn drives the loop on
  purpose), at the same site as the grant ban — one branch of
  `turn_options.build_turn_options`, applied to every kind but `voice`
  (`tests/test_grant_gate_session_path.py::test_every_site_that_bans_minting_also_bans_automod`).
- **One builder arms it, since P13.4.** This used to read "all three
  registry-building sites — the stream endpoint, the synchronous one, and
  `build_ambient_turn`", and the count was the pin. There is now exactly one:
  `app/routers/turn_options.py::build_turn_options` builds the `RunOptions`,
  the `HookRegistry`, the safety hook, the grant gate and both tool bans for
  every turn kind — `stream`, `ambient`, `flush` and `voice` — and the
  synchronous `POST /api/message` was deleted with that change. The
  `kind != "voice"` branch is deliberate and is the only unarmed kind: a
  spoken turn is always a person's, so it gets the safety floor and no grant
  gate and no platform. "The other endpoint is the ungated one" is still the
  bug being closed; the pin is now that `hooks = HookRegistry()`,
  `install_default_safety_hook(hooks)` and `install_policy_hook(hooks,
  scope=grant_scope)` each appear **once** in the builder
  (`tests/test_grant_gate_session_path.py`), with per-kind behaviour held in
  `tests/test_turn_options.py`. `_run_turn` still tops the registry up with
  the deterministic turn guards, because that is where the turn id exists and
  every kind passes through it.

**It denies nothing a worker does today.** Tier 1 is everything unclassified —
`Bash`, `Edit`, `Write`, `Read`, `backlog_write_task`, `vault_write` all pass —
because a gate that guesses wrong denies real work. What it gates is email,
calendar, contacts, tasks and `autonomy_delete_task`.

**It is therefore not vault protection.** When this was written only
`app/harness/safety.py`'s destructive-Bash patterns stood between an
unattended turn and `rm -rf`, and auditing them was the open follow-up. That
follow-up closed on 2026-09-14: vault protection is now four layers below any
hook — the bench/eval tool sandbox, `app/harness/protected_paths.py` enforced
at the aggregator's dispatch as well as the hook, the guardian's
`vaultwatch.py` tripwire, and 15-minute snapshots — [[vault-protection]] is the
long version. The grant gate is still none of them.

---

## 11. The Background tab

`web/src/components/pages/BackgroundPage.tsx`, registered in all four lists
`tests/test_mc_tab_parity.py` pins (`Page` + `NAV_ITEMS` in `Sidebar.tsx`,
`mc_state.VALID_TABS`, `mission_control_ui._VALID_TABS`,
`useMcNavigationEvents.ts`) and in `mc_ui._SUMMARIZERS`, which tells the agent
what it moved the user to as counts by source plus the newest eight — at this
volume a list would be a wall of near-identical titles in the model's context.

- **Runs** — `GET /api/background/sessions`, grouped by producer, polled every
  10 s. A row opens in the Inner Voice reader through the same
  `setPendingFocus` + `setCurrentTab` pair `mc_navigate` uses, so that page
  never has to know who asked; it already renders every row shape the
  recorder writes. A violet mark means the run was observed.
- **Sources** — `GET /api/workers/health`: per-source config, queue depth, a
  7-day outcome rollup aggregated in SQL (`WorkQueue.run_rollup_by_source`)
  and the last five runs. `/api/workers/status` reports what a source is
  *allowed* to do and how much is queued; nothing joined a source to its
  outcomes, so a source failing every run looked exactly like one succeeding
  at every run. `fail_rate` is `null` over zero runs, never 0.0 — "0% failing"
  for a source that has never run is the reading this panel exists to
  prevent. Each of those runs also carries one control per transcript it
  produced — the ids §7 joins, parsed by the endpoint rather than left inside
  `meta_json` — opening that session in the same reader the Runs bullet uses; a
  source that mints no session renders no control. Each section degrades
  independently on a failed fetch.

**The tab is a recent view, not an archive.** The page requests the newest
150 background sessions and the endpoint builds 150 rows, newest first, from the
candidates phase 1 settled — which at the ~240 a day of background sessions
measured for this page's first draft is the last ~15 hours on screen
and ~1½ days reachable. Re-measured 2026-10-13 from `sessions/` mtimes: 266
sessions created in the preceding 24 h and 451 in the 24 h before that (24 and 36
of them bench trials, §12), so the ordinary window is nearer 8-12 hours now.
Two exclusions keep that window honest rather than merely
recent. A row is not dropped for a value nobody read: `platform` sits ~165 bytes
into every session file, so a cheap head read decides whether a candidate is an
ordinary machine run (whose recency is all the tab has to offer) or a chat-shaped
or mis-labelled one, and only the ordinary pool is cut to `limit`. And the *reads*
stay budgeted — `_BACKGROUND_PLATFORM_SCAN_CEILING`, 600 files, spent on the
ambiguous chat-shaped half alone — so an aging directory cannot turn one refresh
into a multi-second request. Every four-part file is therefore still *considered*
for membership however old it is, which is how the 12 mis-labelled sessions got
listed at all; they are listed ahead of ordinary runs, since the tab's ordinary
window will not reach them otherwise and nothing else lists them. (That class is
empty on live data now, and archiving a member does age it out after all — at the
90-day conversation age and never the 30-day background one, because its platform
word is by construction outside `NON_USER_PLATFORMS`; §8, #2647 as corrected by
#2651.) Paging past
`limit` is not built; `unclassified` names the chat-shaped files the read budget
could not reach, so `count: 150` cannot be read as "there are 150". An archived
transcript is not gone: the sweep gzips and never deletes, so it stays openable by
id for as long as its `.json.gz` stands, and a member of this class is not even
archived until 90 days of inactivity.

---

## 12. Deliberately not recorded

| what | why |
|---|---|
| ~~autoresearch bench trials~~ **moved out of this table** | This row was the page's most consequential lie. `run_bench_sdk` used to persist nothing — "session quarantine", pinned by `test_trials_do_not_write_session_files`, a test that no longer exists. `4abc079a` (2026-09-14, the vault-protection landing) inverted it, because the two vault wipes this page exists to learn from were caused by trials that, by design, left no transcript naming a single command they ran. What is quarantined now is the *machine*: the trial's session id is one `agent_mcp/_tool_sandbox.py` recognises (read-only bwrap Bash, non-`readOnlyHint` tools refused), `require_tool_sandbox` refuses to start a trial without it, and **every trial is recorded** through `record_events` into a `platform: worker`, `source: bench` session — 2,544 of them live on 2026-10-13, 24% of `sessions/` (#2648). It still stays out of `messages._run_turn`: no prefetch, no post-session capture, no titler, no ambient producers. |
| `_common.run_prompt_with_run_state` (#529) | Keeps its own per-step `state-trace.ndjson`, and has no single transcript to persist because each step's transcript is deliberately dropped. Nothing in production calls it yet. |
| `automod-regression` | Runs its eval arms as subprocesses; there is no agent turn. |

## 13. Known limits

- The four-part ids carrying a human platform (§8) are gone from live data, so
  the class the Background tab's never-aged-out exemption exists for is empty
  today. The exemption is still in the endpoint, and its limit is an archive, not
  a month: a member of that class is archived at the 90-day **conversation** age,
  not the 30-day background age, because its platform word is by construction
  outside `NON_USER_PLATFORMS` — `_archive_age_for` takes the age from `platform`,
  so the background rule can never reach the class it would have cut short — and
  the endpoint globs `*.json`, so from that archive onwards it is listed by
  nothing (#2647, corrected by #2651).
- `backlog-cluster` does **not** carry the key any more. `#1015` removed it and
  `/api/workers/health` reports the source `inner_voice: null` (measured
  2026-10-13), which is the honest answer for a source with no agent turn. An
  earlier draft of this section said it still reported `false`; §9's account is
  the true one.
- The Background tab builds 150 rows, but membership is not windowed by a scan
  count: the 600 ceiling bounds only the reads of chat-shaped names, while every
  four-part file is opened for a 2 KB head on every 10 s poll — 10,514 reads and
  153 ms per walk on 2026-10-13, which is the limit that actually bites (§8,
  #2646).
- `workers.db` prunes `runs` at 30 d **and** the terminal `queue` rows at 30 d
  (`QUEUE_MAX_AGE_DAYS`, #1466); a live `queued`/`claimed`/`running` row is
  never deleted whatever its age (§8). This section used to say `queue` was
  still unpruned pending #1018, which the tree disproves.
- Session ids are local time; an autonomy `run_id` is UTC. Both are labels,
  and the join between them is explicit (§7), but they will not line up by eye.
- The vault-protection follow-up closed on 2026-09-14 (§10), and `safety.py` now
  delegates the delete/write lanes to `app/harness/protected_paths.py` with
  `tests/test_safety_gate_discrimination.py` exercising the gate; this page is
  not where its state is tracked — [[vault-protection]] is. What remains true is
  §10's point: the grant gate is none of those four layers.

## Files

| file | role |
|---|---|
| `app/run_recorder.py` | the passthrough recorder, kill switch |
| `app/transcript_entries.py` | every transcript row's shape |
| `app/sessions_io.py` | `create_session`, `new_background_session_id`, `is_background_session_name`, `is_user_session`, `is_conversation_session`, `known_platforms`, `current_run_sessions` |
| `app/autonomy.py` | `run_task` wiring, `_task_inner_voice`, observer attach |
| `workers/sources/_common.py` | `run_prompt_on_primary` wiring, `source_inner_voice`, `grant_scope` in the payload |
| `workers/pool.py` | `session_ids` on every run row |
| `workers/queue.py` | `run_rollup_by_source` |
| `app/routers/messages.py` | `_authority_scope_for`, `_ban_grant_minting`, the shared builders |
| `app/routers/turn_options.py` | `build_turn_options` — the one registry/`RunOptions` builder for all four turn kinds, where the grant gate and both tool bans are armed (§10) |
| `app/routers/sessions.py` | `/api/sessions` fast path, `/api/background/sessions` |
| `app/routers/workers.py` | `/api/workers/health`, `/api/workers/runs`, `run_session_ids` (the §7 join, parsed) |
| `app/post_capture.py`, `app/paths.py` | `sessions-background/` export |
| `app/config.py` | `workers.sources.<name>.inner_voice` override merge |
| `scripts/groundskeeper/retention-sweep.py` | 30/90-day archive |
| `web/src/components/pages/BackgroundPage.tsx` | the tab |
| `web/src/lib/runSessions.ts` | a run row's openable transcripts, and their label |

## Tests

| test | pins |
|---|---|
| `tests/test_transcript_entries.py` | both writers produce identical entries |
| `tests/test_run_recorder.py` | passthrough, both direct paths, partial transcript on a killed run, `role="thinking"`, both ids set, kill switch |
| `tests/test_grant_gate_session_path.py` | which turns are gated, the mint ban, that the one builder arms each hook exactly once (per-kind behaviour in `tests/test_turn_options.py`), fail-closed store |
| `tests/test_background_inner_voice.py` | frontmatter beats config, observer keeps the grant hook, cancel lever, closes on failure, one reader for the source switch, override merge |
| `tests/test_session_platform_checks.py` | no hand-rolled platform check, retention agrees, the id creators |
| `tests/test_dashboard_sections.py` | background runs do not starve the recent panel |
| `tests/test_api_contracts.py` | `/api/background/sessions`, `/api/workers/health`, the two listings are complements, created sessions are chat-shaped |
| `tests/test_workers_router.py` | the §7 join as a parsed `session_ids` field on `/api/workers/runs` and on `recent` in `/api/workers/health`: both meta keys unioned, one entry per transcript, `[]` for a row that names none — plus the Sources row's markup (one control per transcript, opening the mapped id), and this page's own §7 bullet naming the surface a human walks |
| `web/src/lib/runSessions.test.ts` | the Sources panel's derivation, and that the page threads its own reader callback to the run rows (the click itself is human-only: no renderer in this vitest config) |
| `tests/test_background_runs_doc_claims.py` | what this page may assert: the chat summarizer's call path, scaling and a dated figure (§8, #1228), and that the mis-labelled class expires at the 90-day conversation age rather than a 30-day archive in every section that dates it (§8, §11, §13, #2651) |

## Review log

- 2026-10-13 — **stale**. Checked every path, count, config key and named test in
  the page against `0cbb7ea49c62`. The mechanism sections still describe the tree
  (recorder, transcript shape, the two predicates, the joins, per-job
  observation, the grant gate); four things had moved underneath it. §5 still
  described a tool result as cut at 2 KB, when D1 made a long result a
  `<persisted-output>` pointer into `<sid>.tool-results/`; §10 still counted
  "three registry-building sites", when P13.4 collapsed them into
  `turn_options.build_turn_options` and deleted the synchronous route; §12 still
  said autoresearch bench trials persist nothing, when `4abc079a` inverted that
  in favour of quarantining the machine and recording every trial; and §13's
  Known limits contradicted §8, §9 and §10 in the same page on `queue` pruning,
  `backlog-cluster`'s key and the vault-protection follow-up. The twelve
  mis-labelled sessions §8 census are no longer on live data at all. Filed
  #2646 (the endpoint's uncapped four-part head walk), #2647 (the never-aged-out
  exemption expiring at the sweep's 30-day archive) and #2648 (the recorded
  `source: bench` class has no life).
- 2026-10-13 — **Correction to the entry above (#2651).** Its `#2647`
  parenthetical is wrong, and so was every sentence on this page that repeated it:
  `_archive_age_for` (`scripts/groundskeeper/retention-sweep.py:607-619`) picks the
  archive age from the `platform` field, and this class exists precisely because
  its platform is outside `NON_USER_PLATFORMS`, so a member is archived at the
  90-day conversation age, not at 30 — the exemption was never expiring a month
  after a run. §8, §11 and §13 now carry the measured policy; §8's eight
  `20260909_1538xx_autonomy_*` ids are reattributed to ordinary `worker` runs
  archived on the machine age, since all 42 archived four-part files on the box
  report `platform: worker`. The entry above is left exactly as filed.
