---
segment: architecture
tags: [architecture, lloyd, workers, autonomy, inner-voice, sessions]
type: reference
status: implemented
date: 2026-09-10
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
model, source)` is an async generator. It wraps an existing `run_query(...)`
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

## 5. One transcript shape

`app/transcript_entries.py` owns the shape of every row a turn writes:
assistant text, tool call, tool result, thinking phase, and the opening user
prompt. It was built inline in `app/routers/messages.py` — five assistant
variants and three tool-pair variants across the streaming, cancel and error
paths — and a second writer would have needed a second private copy. Both
writers call the builders now, and `tests/test_transcript_entries.py` feeds
the same events to each and asserts the entries are identical.

Two optional fields stay optional, because every session on disk predates
them: `summary` on a tool call is omitted when empty, and `is_error` on a tool
result's stats is omitted when the writer did not see the result — the two
reconstruct-from-the-log paths never did, and writing `false` there would turn
"we never saw it" into "it was fine". `role="thinking"` rows keep `content`
empty; the role is what keeps reasoning out of every transcript producer (see
`architecture/inner-voice.md` and `tests/test_thinking_trace_transcripts.py`).

## 6. One session writer

`sessions_io.create_session(session_id, *, platform, model, title, source,
inner_voice)` creates every non-chat session. `workers.sources._common.
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
ids. The `source` slug is the source name with non-alphanumerics dropped.

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
  a property worth depending on eleven times. The pool writes it on the
  timeout and exception branches too.
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

### One definition of "is a human reading this"

`sessions_io.is_user_session(data)` — false for any `platform` in
`NON_USER_PLATFORMS` (`autonomy`, `worker`), true for a missing platform, and
true for a platform the list has never heard of (a deny-list, on purpose).

Six readers hand-rolled `data.get("platform") == "autonomy"` and none of them
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

All six call `is_user_session` now. `tests/test_session_platform_checks.py`
greps `app/`, `agent_mcp/` and `workers/` for the literal, because a seventh
reader written next month is how this comes back — silently, since a
background session in the history looks like a session. The retention sweep
restates the tuple (it is stdlib-only and runs from cron with no venv), and
the same test pins that the two agree.

### The id shape is a fast path — and in the chat listings, it decides

A background id has four underscore-separated parts; a chat's has three
(`<ts>_<6hex>` from the chat path, `<ts>_iv<4hex>` from `POST /api/sessions/create`).
`sessions_io.is_background_session_name` tells them apart without opening the
file, and at ~240 background sessions a day against 14 chats that is the
difference between a listing that is bounded and one that grows with the
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
- **Three-part ids are parsed and judged by `platform`.** A chat-shaped name
  carrying a background platform is still excluded.
- **`/api/background/sessions` parses every four-part file** and judges it by
  `platform`, so there the name is a hint.

One consequence is already on disk: eight sessions named
`20260909_1538xx_autonomy_<4hex>`, created within 11 seconds by the sandbox
experiment above, carry `platform: mission-control` because the chat endpoint
created them. They are four-part, so the chat listings skip them, and
user-labelled, so the Background listing drops them. They appear in neither.
They hold one or two user rows each and are not conversations.

`_scan_recent_sessions` used to keep the newest 24 files by mtime and only
then drop non-user rows, which with 22 of the newest 24 already background
rendered a nearly empty panel. It now walks mtime order, skips by name, and
stops at `_RECENT_KEPT` user rows or `_RECENT_CEILING` (400) files opened.

### Post-capture and the titler

`_post_session_capture` and `maybe_title_session` are fired from
`app/routers/messages.py::_run_turn` and nowhere else, so the only background
sessions that ever reach them are the session-backed workers — ~70 a day.

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
two months early. `workers.db` itself still has no retention job.

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
| `autocode` | true | rewrites production |
| `autotriage` | true | judges items the loop will then implement |
| `youtube-digest` | true | evaluates untrusted transcripts and files backlog items from them |
| `deep-research` | false | a human reads the note before anything acts on it |

The key is set only on those four, because only a turn that runs through the
chat endpoint can be observed at all. `/api/workers/health` reports it
tri-state for that reason — `null` means "not set", which for a direct-path
source means "not observable", and a flat `false` would invite a knob that
reads as broken. It is UI-mutable through `data/tool_overrides.yaml` like
`workers.enabled`; the override merge honours that one key per source and
nothing else.

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

`create_session` writes `inner_voice` and `inner_voice_evaluate_user_turns`
equal to each other, from the job's switch. They have to move together
because the two kinds of turn arrive differently: a worker's prompt is posted
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

Importing a router helper into `autonomy.py` is a layering smell, accepted
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
- **All three registry-building sites arm it** — the stream endpoint, the
  synchronous one, and `build_ambient_turn`, where it cannot fire today
  because `/inject` refuses a non-user session with 409. "The other endpoint
  is the ungated one" is the bug being closed; the count is pinned in
  `tests/test_grant_gate_session_path.py`.

**It denies nothing a worker does today.** Tier 1 is everything unclassified —
`Bash`, `Edit`, `Write`, `Read`, `backlog_write_task`, `vault_write` all pass —
because a gate that guesses wrong denies real work. What it gates is email,
calendar, contacts, tasks and `autonomy_delete_task`.

**It is therefore not vault protection.** Only `app/harness/safety.py`'s
destructive-Bash patterns stand between an unattended turn and `rm -rf`, and
auditing those against `rm -rf ~/obsidian`, `tar … && rm -rf` chains and any
non-Bash deletion path is the open follow-up from the incident that started
this.

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
  prevent. Each section degrades independently on a failed fetch.

**The tab is a recent view, not an archive.** The page requests the newest
150 background sessions and the endpoint opens at most
`_BACKGROUND_SCAN_CEILING` (600) files, newest first. At ~240 a day that is
the last ~15 hours on screen and ~2½ days reachable. Older transcripts stay on
disk for 30 days, openable by id; paging the listing past the ceiling is not
built.

---

## 12. Deliberately not recorded

| what | why |
|---|---|
| autoresearch bench trials | `run_bench_sdk` persists nothing by design — "session quarantine", pinned by `test_trials_do_not_write_session_files`. A trial measures a prompt variant; it is not work anyone reviews, and hundreds of them would bury the runs that are. |
| `_common.run_prompt_with_run_state` (#529) | Keeps its own per-step `state-trace.ndjson`, and has no single transcript to persist because each step's transcript is deliberately dropped. Nothing in production calls it yet. |
| `automod-regression` | Runs its eval arms as subprocesses; there is no agent turn. |

## 13. Known limits

- The eight mis-labelled sandbox sessions (§8) appear in neither listing.
- The Background tab shows the newest 150 runs out of at most 600 scanned (§11).
- `workers.db` has no retention; session files do (§8).
- Session ids are local time; an autonomy `run_id` is UTC. Both are labels,
  and the join between them is explicit (§7), but they will not line up by eye.
- The vault-protection audit of `safety.py` is not done (§10).

## Files

| file | role |
|---|---|
| `app/run_recorder.py` | the passthrough recorder, kill switch |
| `app/transcript_entries.py` | every transcript row's shape |
| `app/sessions_io.py` | `create_session`, `new_background_session_id`, `is_background_session_name`, `is_user_session`, `current_run_sessions` |
| `autonomy.py` | `run_task` wiring, `_task_inner_voice`, observer attach |
| `workers/sources/_common.py` | `run_prompt_on_primary` wiring, `source_inner_voice`, `grant_scope` in the payload |
| `workers/pool.py` | `session_ids` on every run row |
| `workers/queue.py` | `run_rollup_by_source` |
| `app/routers/messages.py` | `_authority_scope_for`, `_ban_grant_minting`, the shared builders |
| `app/routers/sessions.py` | `/api/sessions` fast path, `/api/background/sessions` |
| `app/routers/workers.py` | `/api/workers/health` |
| `app/post_capture.py`, `app/paths.py` | `sessions-background/` export |
| `app/config.py` | `workers.sources.<name>.inner_voice` override merge |
| `scripts/groundskeeper/retention-sweep.py` | 30/90-day archive |
| `web/src/components/pages/BackgroundPage.tsx` | the tab |

## Tests

| test | pins |
|---|---|
| `tests/test_transcript_entries.py` | both writers produce identical entries |
| `tests/test_run_recorder.py` | passthrough, both direct paths, partial transcript on a killed run, `role="thinking"`, both ids set, kill switch |
| `tests/test_grant_gate_session_path.py` | which turns are gated, the mint ban, the three sites, fail-closed store |
| `tests/test_background_inner_voice.py` | frontmatter beats config, observer keeps the grant hook, cancel lever, closes on failure, one reader for the source switch, override merge |
| `tests/test_session_platform_checks.py` | no hand-rolled platform check, retention agrees, the id creators |
| `tests/test_dashboard_sections.py` | background runs do not starve the recent panel |
| `tests/test_api_contracts.py` | `/api/background/sessions`, `/api/workers/health`, the two listings are complements, created sessions are chat-shaped |
