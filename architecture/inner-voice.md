---
title: Inner Voice — Architecture
status: active
created: 2026-05-02
updated: 2026-09-05
related:
  - architecture/agents.md
  - architecture/verification-system.md
  - architecture/subliminal.md
---

# Inner Voice — Architecture

A second agent that watches the primary agent's stream and intervenes when it sees
a problem. One LLM, one prompt, five soft levers, one goal card. The Python is
plumbing — all judgment lives in the observer's system prompt, which is a
vault-editable markdown file.

**Version: v5.3.** v4's function-tool levers are unchanged; v5 changed *when* the
observer spends an LLM call, *where* those calls run relative to the primary's
critical path, and whether the subsystem can measure itself. The observer moved
to the secondary model on 2026-09-04 and is now pinned back to `primary` — see
[Observer runs on the secondary model](#observer-runs-on-the-secondary-model-since-2026-09-04)
and [Version history](#version-history).

v5.2 added deterministic loop detection. v5.3 is the correction pass over it,
after that guard fired 19 times in one evening on ordinary file reads and, on one
turn, spent the discretionary budget that the observer's only correct judgment of
the turn then needed. Open defects are listed under
[Known defects](#known-defects).

## Design intent

> "A second agent watching the first agent to steer and nudge as necessary as a
> second look at the progress of the primary agent."

The implementation is a thin event-tap + observer LLM whose job is to **keep the
primary on task and working** against an explicit goal contract. To tune behavior,
edit the prompt in the vault. Adding a Python heuristic is almost always the wrong
move — though see [Where Python crept back in](#where-python-crept-back-in) for an
honest account of where that rule has bent.

## High-level shape

```
                ┌──────────────────────────┐
                │  Primary agent loop      │   yields NormalizedEvent
                │  (app/harness/loop.py)   │ ─────────┬──────────────┐
                └──────────────────────────┘          │              │
                          ▲                           │              │
                          │ PreToolUse (observe)      ▼              ▼
                          │             ┌──────────────────┐  to existing
                          ├────────────►│ Observer task    │  consumer
                          │             │ (per-turn)       │  (messages._run_turn)
                          │             │                  │
                          │             │  • goal card     │
                          │             │  • event queue   │
                          │             │  • fast-path     │
                          │             │  • LLM judgment  │
                          │             │  • five levers   │
                          │             │  • cross-event   │
                          │             │    memory        │
                          │             └──────────────────┘
                          │
                          │ PreToolUse (hard deny)
                          └─────────────┤ app/harness/safety.py │
                                        └───────────────────────┘
                                        always on, IV or no IV
```

One observer task per primary turn. At turn start it runs a goal-extraction LLM
call that produces a structured goal card (success criteria, out-of-scope
concerns, completion signals). It then subscribes to the harness's
`NormalizedEvent` stream via an `OnEvent` hook and a `PreToolUse` hook. Each
subsequent event runs through a cheap deterministic fast-path; if the fast-path
can't decide, it escalates to a focused vLLM call with the goal card, the primary's
subliminal context, the live todo list, any committed plan, the persistent goal,
its prior decisions this turn, and its interventions from earlier turns.

Only the **terminal** events are judged synchronously. Everything else runs off
the harness's critical path — see
[Off the critical path](#off-the-critical-path).

## Goal contract (per turn)

Before the primary runs, the observer extracts a **goal card** from the user
request by forcing a single `record_goal_card` tool call:

```json
{
  "success_criteria": ["concrete things that must be true to be done"],
  "out_of_scope": ["things the agent should NOT pursue"],
  "completion_signals": ["specific outputs that mean the goal is fully addressed"]
}
```

Each list is capped at 8 items, each item at 400 chars.

The goal card is the observer's reference for "is the primary still on task?" Every
per-event prompt includes it. When the primary drifts into `out_of_scope`
territory, the observer can `inject` to redirect.

**Follow-up resolution.** The extractor also receives the last ~6 user/assistant
exchanges (`_recent_exchanges_for_goal_extraction`), so a message like "yeah do it"
or "still broken" resolves against the prior thread instead of producing an empty
goal card. The observer's own breadcrumbs (`source` starting with `inner_voice_`)
are filtered out of that window — feeding them back made the next turn's goal card
anchor on IV's own demands.

If goal extraction fails (timeout, no tool call, conversational request with no
actionable goal), the observer runs in lighter-touch mode with no goal card.

Cost: one extra LLM call per IV-enabled turn at attach time.

## Observer's contract per event

### What it sees

Threaded into every per-event prompt by `build_user_prompt_for_event`:

- The original user request and the extracted goal card.
- The **subliminal context block** the primary saw at turn start (prefetched
  skills, vault hits, facts, ambient signals — see [[subliminal]]), capped at 4000 chars (head + tail; each `<skill>` block is trimmed to 1800 first so facts, vault hits and IDE state survive the cap)
  chars. Lets the observer recognize when the primary is following documented
  procedure rather than freelancing.
- The live **todo list** (`session.todos`), refreshed mid-turn after each
  successful `TodoWrite`.
- The **committed plan** artifact (`session.plan`) when one exists. When
  `plan_mode=True` the prompt swaps framing from "watch for execution drift" to
  "evaluate plan quality."
- The **persistent goal** (`session.goal`, set via `/goal`) when set and not yet
  achieved.
- A head+tail window of the primary's accumulated text
  (`primary_text_window_chars`, default 4000). Head+tail rather than head-only —
  chopping the conclusion made every long response look cut off mid-sentence.
- **Its own prior decisions this turn** (last ~8) — so it can escalate when the
  primary ignored an earlier inject, or back off when it's been wrong.

Live tap on `assistant_message`, `tool_call`, `tool_result`, `result`, plus a
`PreToolUse` callback for every tool dispatch. Text deltas flow into
`accumulated_text` for context but don't trigger an LLM call per token.

### What it can do — five soft levers

The observer's response is a forced function call (`tool_choice="required"`) over
`LEVER_TOOLS`. The tool name **is** the action.

| Action | Effect | Trigger contexts |
|---|---|---|
| `noop` | Default. Watched and chose not to act. | All triggers. |
| `inject` | Append `{"role": "user", "content": "[INNER VOICE] ..."}` to the primary's chat history before its next iteration. | `assistant_message`, `tool_result`, `pretool` |
| `cancel` | `options.cancel_event.set()`. Harness exits cleanly with `stop_reason="cancelled"`. | `assistant_message`, `tool_result`, `pretool` |
| `ambient` | Enqueue a follow-up ambient turn via `build_ambient_turn` + `enqueue_ambient`. | `assistant_message`, `tool_result`, `result`, `pretool` |
| `clarify` | Surface a short question to the user as an assistant message and pause the primary (set `cancel_event`). | `assistant_message`, `tool_result`, `pretool` |

**There is no `deny_tool` lever.** Since v4, Inner Voice cannot block tool
dispatch. Hard safety lives in `app/harness/safety.py`
([Hard safety gate](#hard-safety-gate-not-inner-voice)), which runs on every
primary turn whether or not the session opted into IV. Pretool is now purely an
observation trigger — the same five soft levers apply, and an `inject` there lands
as the next user message *after* the tool dispatches.

### Lever invariants

- **Inject role is `user`, not `system`.** vLLM rejects mid-stream system messages
  (must be at position 0). The `[INNER VOICE]` prefix makes the source explicit to
  the primary. A model-emitted `[INNER VOICE]` prefix is stripped before injection
  so it doesn't double up.
- **Cancel is an escape hatch, not a completion signal.** The lever schema says so
  explicitly, and two guards enforce it — see
  [Cancel-for-completion guard](#cancel-for-completion-guard). The harness
  terminates naturally on a text-only iteration; cancel is for destructive loops,
  ignored injects, and tight tool loops.
- **Cancel is exempt from the intervention budget.** It ends the turn, so rationing
  it would block recovery from "primary keeps ignoring my injects."
- **Cancel sets `state.closed`.** Pretool callbacks already in flight
  short-circuit instead of surfacing a confusing post-cancel decision.
- **Clarify pauses the primary.** It persists the question as an assistant message
  via a session-scoped callback, then sets `cancel_event`. The user replies in the
  next turn, which gets a fresh goal card.
- **Result-trigger inject auto-translates to ambient.** By the time `result` fires,
  the harness has emitted its terminal event; injecting is wasted. The dispatcher
  rewrites `inject` → `ambient` if a callback is wired, else drops it as
  `noop_inject_on_result`.
- **Result-trigger cancel/clarify auto-degrade to noop.** Same reasoning.
- **Interventions leave a user-visible breadcrumb.** `inject` and `cancel` call
  `persist_intervention_callback`, which appends a message to the session JSON with
  `source: "inner_voice_inject"` / `"inner_voice_cancel"`. The
  `chat_messages_handle` append is transient (in-memory only); the breadcrumb is
  what the user and the persisted history see.

### Per-turn budget

Default cap: **3 interventions per turn** (`inner_voice.observer.intervention_budget`).
Counted against: `inject`, `ambient`, `clarify`. **Not** counted: `cancel`, and any
decision with `bypass_budget=True` (stall rescue, repetition). After exhaustion,
further lever choices are downgraded to `noop_budget_exhausted` and persisted with
that label so the UI shows what *would* have happened.

**The `bypass_budget` exemption was documented but not implemented until v5.3.**
`_apply_lever` incremented `interventions_used` on every applied inject,
deterministic or not. That is invisible while the deterministic guards are rare
and fatal once one misfires: on turn `8f3b7e77de07` ten false repetition injects
exhausted a budget of three, and the observer's one correct judgment of the whole
turn — *"Turn hit max_turns with zero review delivered; all six todos
unresolved"* — was recorded as `noop_budget_exhausted` and never fired.
`_count_intervention` now charges deterministic injects to a separate
`bypass_interventions_used`.

Exempt from the budget is not the same as unbounded, and nothing bounded them.
`deterministic_inject_budget` (default **5**) is a hard ceiling on
`bypass_budget` injects per turn; past it they are downgraded to
`noop_deterministic_budget_exhausted`. A miscalibrated guard now costs at most
five nudges, not a whole turn.

## Tiered triggering

Most events shouldn't hit the LLM. Configurable via
`inner_voice.observer.fast_path_enabled` (default `true`).

The v4 tiering halved the *call count* but left the two biggest spenders
untouched, and the first production window showed what that cost: across 44
turns the observer consumed 3.20M input tokens against the primary's 3.22M in
the same sessions. v5 re-tiers on measured yield per trigger.

| Trigger | v4 LLM calls | Interventions | Input tokens | Tokens per intervention |
|---|---|---|---|---|
| `pretool` | 264 | 3 | 1,439,940 | 480,000 |
| `assistant_message` | 156 | 7 | 865,565 | 124,000 |
| `tool_result` | 119 | 2 | 637,056 | 319,000 |
| `result` | 40 | 8 | 200,903 | 25,000 |

### Pretool is observation-only

`inner_voice.observer.pretool_llm_enabled` — **default `false`.**

Pretool has not been able to block a tool dispatch since v4, when hard safety
moved to `app/harness/safety.py`. An `inject` chosen here therefore lands as the
next user message *after* the tool has already run — at which point the
`tool_result` trigger sees the same call plus its outcome and can judge it
better. Pretool was 45% of all observer spend for three interventions, none of
which prevented anything.

The observation **row is still written** on every pretool event. The
prior-decisions block and the mark-without-evidence check both read tool
activity out of those rows, so dropping them would blind checks that still work.
Setting the flag `true` restores v4 behavior.

### Tool results are sampled

`tool_result_sample_every` (default `5`), `tool_result_escalate_bytes` (default `20000`).

These always escalate to the LLM:

- Any error result, except the `"Tool call arguments could not be parsed as
  JSON"` parse-retry (the primary handles its own retry).
- A result at or above `tool_result_escalate_bytes`.
- A `TodoWrite` that flipped `in_progress→completed` (mark-without-evidence).
- The stalled-progress gate firing.

Everything else is a benign result, and one in `tool_result_sample_every` is
judged. The rest noop. The v4 rule escalated every non-error result over 2 KB,
which contradicted the vault prompt's own instruction that "the primary's tool
result is large or surprising — that's the primary's problem to interpret."

### Assistant-message fast-path

1. **Pure tool dispatch** (tool calls, no text) → `noop`.
2. **Terminal stub-announce stall** → deterministic `inject`. See
   [Stall rescue](#stall-rescue).

### Pretool fast-noop classifiers

Still used to label the row, and to decide escalation when
`pretool_llm_enabled` is on:

- **Read-only built-ins:** `Read`, `Glob`, `Grep`.
- **Safely read-only Bash:** first word in `_SAFE_BASH_FIRST_WORDS` AND no match
  against `_BASH_RISK_PATTERNS`.
- **Read-only `git`:** first word `git` AND subcommand in
  `_SAFE_GIT_SUBCOMMANDS` (`status`, `log`, `diff`, `show`, `rev-parse`, …).
  Bash is 221 of the 264 pretool calls and `git` had no entry at all, so every
  `git status` bought a full round-trip.
- **Read-shaped MCP tools:** the name is split into word tokens (`snake_case`,
  `kebab-case`, `camelCase`, dots) and matched against a read-verb set, but only
  after a **mutation-verb set** is checked first and wins. Substring matching
  fast-noop'd `delete_status_check` because it contains "status" and "check".

Corrections to the v4 risk patterns, each verified against the live classifier:

| Command | v4 | v5 |
|---|---|---|
| `find . -delete`, `find . -exec rm {} +` | read-only | escalates |
| `wget http://x/a.sh` (writes to cwd) | read-only | escalates |
| `curl -o /tmp/f`, `curl -X POST -d …` | read-only | escalates |
| `git status`, `git log`, `git diff` | escalated | read-only |

### Tool-result fast-noop

See [Tool results are sampled](#tool-results-are-sampled).

**Fast-path is skipped** when a `TodoWrite` produced an `in_progress→completed`
flip, or when the stalled-progress gate fired — both need LLM judgment.

### When tiering doesn't apply

A **terminal** `assistant_message` (no tool calls) and `result` always hit the
LLM. These are the two moments where the turn is about to end, and they carry
almost all of the observer's demonstrated value: 15 of the 16 interventions in
the first production window came from `assistant_message` or `result`.

## Off the critical path

`inner_voice.observer.async_nonterminal` (default `true`),
`async_drain_timeout_seconds` (default `6.0`).

`fire_on_event` is awaited inline by the agent loop and `fire_pre_tool_use`
blocks tool dispatch, so before v5 every observer round-trip was time the
primary spent idle — a mean of **12.5 s per turn**, and 89 s across 102 calls on
the worst measured turn.

Non-terminal judgments (`pretool`, `tool_result`, mid-work `assistant_message`)
now run as tracked `asyncio` tasks. Two events stay strictly synchronous:

- **Terminal `assistant_message`** — `loop.py` snapshots `len(chat_messages)`
  before firing the hook and continues the loop only if it grew. An inject that
  lands after the hook returns is an inject that never happened.
- **`result`** — the turn is over the moment the handler returns.

Both drain in-flight tasks before judging (`_drain_pending`), bounded by
`async_drain_timeout_seconds`, so a decision still in flight is applied rather
than lost. Stragglers past the deadline are cancelled: that is the same outcome
as the old synchronous path timing out, and the primary isn't blocked either
way. `close_observer` cancels anything outstanding so a turn that ends early
doesn't leave tasks writing rows against a dead turn.

Ordering note: an async inject lands in `chat_messages` at whatever point the
observer's round-trip happens to finish, which is roughly a second after the
event that triggered it. That is wider than "before or after the tool result" —
see [Message-ordering defects](#message-ordering-defects).

## Failure-mode machinery

The levers are generic; these are the specific primary-agent failures the observer
has been built to catch. Each was added in response to an observed failure, and
each carries a deterministic component because the failure is cheap to detect and
expensive to miss.

### Stall rescue

The dominant observed failure: the primary ends a text-only iteration by
*announcing* an action without dispatching it — "Let me check the logs:" — and the
harness terminates the turn. `_STUB_ANNOUNCE_RE` matches text whose last line is a
bare announce verb (`let me`, `I'll`, `I'm going to`, `next, I'll`, `I need to`, …)
or that ends on a colon.

**Three-stage since v5.3.** The announce regex proposes; a false-positive regex
disposes; and a sentence that closed on its own terms is exempt. On its own the announce pattern also matched delivered answers —
"Let me know if you need anything else!", "I'll be happy to help", "I need to
note that X", "I should mention one caveat: …" — and the stall-rescue inject
bypasses *both* the intervention budget and the consecutive-inject suppressor by
design. A primary that habitually signs off that way would have been re-prompted
every iteration until `max_turns`. The exclusion list covers sign-offs and
speech acts completed within the sentence itself ("I need to note that X" *is*
the note; nothing is deferred). Both directions are pinned by tests in
`tests/integration/test_iv_guards.py`.

This never fired in the first production window, so the bug was latent, not
observed — which is exactly why it was worth fixing before the trigger rate went
up.

**The exclusion list was not enough.** It named specific sign-off phrases, so it
only caught the shapes someone had thought of. Any final line *starting* with an
announce verb still matched regardless of what followed it, which made all of
these stalls:

```
I'll leave the config as-is since it already works.
I'm going to recommend option B because it is simpler.
I will not change that file because it is generated.
Going to the source, the loop appends the assistant message after the hook.
```

Every one is a delivered statement. v5.3 adds two structural rules, applied only
to a last line that **closed** — a line ending in a colon or an ellipsis is still
a stall outright, since that is the signal that the primary stopped mid-thought:

- `_ANNOUNCE_RESOLVED_RE` — a justification or negation clause (`because`,
  `since`, `rather than`, `will not`, `won't`) means the sentence resolved rather
  than deferred.
- `_ANNOUNCE_NOUN_FOLLOWER_RE` — "going to **the** source" heads a noun phrase,
  not a promised action.

The false-positive list also gained the decision verbs (`recommend`, `suggest`,
`leave`, `hear`, `go with`) and an optional adverb slot, so "I should **also**
mention …" is recognised the way "I should mention …" already was. `Let me read
the config file.` and `I'll update the docs to match.` are still stalls;
`tests/integration/test_iv_v52_review.py` pins both directions.

Two paths handle it:

1. **Fast-path (deterministic).** On a text-only iteration matching the regex, the
   observer returns an `inject` carrying `_STALL_RESCUE_CONTENT` with
   `bypass_budget=True`. No LLM round-trip. Because this path never touches the
   consecutive-inject suppressor, a re-stall on the very next iteration is rescued
   again rather than left to die.
2. **LLM path (ambient→inject upgrade).** If the LLM judged a terminal iteration
   and chose `ambient`, the dispatcher upgrades it to `inject` with
   `bypass_budget=True` — ambient goes to the background channel and does *not*
   continue the loop, so the turn would otherwise die with work undone.

The primary's system prompt carries a matching "Turn discipline" clause
(`prompt_builder.build_system_prompt`), so this is defended on both sides.

### Consecutive-inject suppressor

An `inject` is downgraded to `noop_inject_after_inject` if the most recent
mid-work decision was also an inject — but **only mid-work** (tool calls in
flight), where the rationale holds: give the primary an iteration to act. On a
terminal text-only iteration the inject is the only thing keeping the loop
alive, so suppression there is disabled.

**Spans all mid-work triggers since v5.** v4 compared only same-trigger pairs,
which meant `pretool`, `tool_result` and `assistant_message` each kept their own
blind history. Turn `2cf39d2c0ead` shows the failure: an inject at `pretool`, an
inject at `tool_result`, another at `pretool`, then a `cancel` — four
interventions in 20 seconds, inside a single dispatch batch, with no model turn
between any of them. The cancel justified itself with "still reading the
transcript after 3 injects" when the primary had not been given the chance to
read even one.

`guards.injects_primary_has_seen` was written as the companion counter — an
inject counts as *seen* only once an `assistant_message` decision lands after it,
so a dispatch batch collapses to a single nudge for escalation purposes.

**It is still not wired to anything.** The function and its test exist; no
production call site does. Escalation keys off `state.interventions_used`, which
counts injects the primary may never have read. The cancel in turn
`0251c403fd8d` — "primary exhausted inject budget and is stuck in a loop" — is
the failure it was written to prevent, fired anyway; so is the cancel in
`20260905_011748_iv84e4` a year later, with the same reason string.

v5.2 attacks that turn from the other side — `guards.inject_on_cooldown` and its
companion `iterations_since_last_inject` make the budget *hard to spend quickly*
(see [Inject pacing](#inject-pacing)) — but escalation itself still counts
injects fired rather than injects read. Wiring `injects_primary_has_seen` into
the cancel path remains open.

### Repetition — the search loop

Added v5.2, after turn `20260905_011748_iv84e4`. A code-review request ran
10m14s over 33 iterations and produced **zero characters of user-visible text**
before Inner Voice cancelled it. The core of it was six reformulations of one
search — "who consumes `iv_inject_queue` / `iv_cancel_requested` outside the IV
package" — each with a slightly different filter. The answer (nobody) was
correct and stable from the first attempt. The primary read the empty result as
a *broken query* rather than an answer, and kept rewriting it; then went hunting
for `lloyd_mcp` (the package is `agent_mcp`) and `build_subliminal_context`
(never defined anywhere), which is a search that cannot terminate on its own;
which ended in two `find /` scans and an unbounded `grep -rn` over 45 MB of
`sessions/` that hit the 120s Bash timeout. One of those greps matched the
session transcript recording the grep itself.

The observer could not see any of it. `build_tool_result_summary` rendered the
tool *name* and 300 chars of the *result*; the command never reached the prompt.
A loop is visible only in the arguments.

`guards.repetition_verdict` is therefore deterministic and runs at `pretool` —
the one place the observer sees arguments — regardless of `pretool_llm_enabled`,
because it costs no LLM call. Shell reformulation defeats string similarity (the
six commands shared little literal text), so the comparison is over **code
identifiers**: tokens carrying an underscore, or ≥ 12 characters. Bare English
words are `echo` labels and shell noise. Sets are compared by *containment*
rather than Jaccard, so a command wrapped in extra labels still matches the bare
one, and a match needs ≥ 2 shared identifiers at ≥ 0.5 containment. The guard
fires when the current call matches ≥ 2 of the last 6.

Calibrated by replaying that turn's 28 Bash calls: it first speaks on the
**4th** near-duplicate and stays silent through the 15 healthy exploration calls
that preceded it. Both directions are pinned in
`tests/integration/test_iv_loop_guards.py`, which replays the real commands from
`fixture_iv_loop_turn.py` through `fire_pre_tool_use` with the LLM patched to
raise — the guard must never need a model.

#### Corrected in v5.3, after 19 misfires in one evening

As shipped, the identifier set came from the whole `key=value` rendering of the
arguments. Two consequences:

- Argument **key names** counted as shared identifiers, so any two `Edit` calls
  shared `file_path`, `old_string` and `new_string` before their contents were
  considered at all.
- **Path components** counted, so every absolute path contributed the username —
  13 characters, comfortably past the length filter, and present in literally
  every path the agent touches.

Two shared identifiers is all a near match needs, so three Reads of unrelated
files matched each other. Of the 19 deterministic injects on 2026-09-04, 15 named
a path segment or an argument key as the thing the primary "kept chasing":
`deterministic: 3 near-identical Read calls for alansrobotlab, file_path,
architecture`. Three changes:

1. **Identifiers come from argument values only.** `exact` still renders the full
   `key=value` form, so an exact repeat is still an exact repeat.
2. **Ambient path fragments are stripped** — the home directory and the username —
   before tokenising.
3. **Path-addressed tools compare by exact repeat only.** `Read`, `Write`,
   `Edit`, `MultiEdit`, `NotebookEdit`, `Glob` and `TodoWrite` are in
   `_EXACT_ONLY_TOOLS`. Near-matching was built for shell reformulation and does
   not transfer to a tool addressed by path. Re-reading one file verbatim is
   still caught; a chunked read carries a different offset and is correctly not a
   repeat.

Stripping the ambient tokens removed the free second identifier that
`min_overlap: 2` had been calibrated against, so a genuine hunt for one symbol
would no longer have fired. `guards._is_distinctive` restores it: an identifier
with three or more underscore-joined parts, or 16+ characters, carries a match on
its own. `iv_inject_queue` and `zzq_phantom_handle_v3` qualify; `inner_voice`,
`file_path` and `observer_prompt` do not — those are modules and argument keys
that half the calls in a session mention in passing.

Net effect on the fixture: the first fire is still exactly at the 4th
near-duplicate, and a **second** cluster is now caught — messages 72-79, where
the primary hunts `build_subliminal_context`, a symbol never defined anywhere.
Ambient-token dilution had hidden it. The 15 healthy exploration calls stay
silent. `tests/integration/test_iv_v52_review.py` pins the negative direction
against real Read / Edit / Write / Glob / TodoWrite sequences.

The ring is no longer **cleared** on a fire. A `repetition_baseline` counter marks
the fire point and comparison looks only at calls made after it. Same "needs a
fresh cluster" rate limit, without discarding the command previews that
`build_tool_result_summary` reads back — the one thing turn
`20260905_011748_iv84e4` proved the observer needs to see.

The inject carries `bypass_budget=True`, like stall rescue: it prevents a
pathological outcome rather than nagging, and it is the observer's only signal
that never guesses at intent. Its rate limit is structural — firing clears the
signature ring, so speaking again requires a *fresh* cluster of near-duplicates.
The content names the shared identifiers and states the thing the primary did
not believe: **an unchanged result is the answer, not a failed query.**

### The silent primary

Same turn. Every one of 33 iterations was tool-dispatch-only, and
`_fast_path_assistant_message` noops those unconditionally — 33 rows at zero
latency. `assistant_message` is the only trigger with a whole-turn view, and it
never once reached the LLM. Its stated rationale is that "the pretool gate
already evaluated each proposed tool", but pretool judgment has been off by
default since v5, so the fast path was deferring to a gate that is not running.

Removing the fast path would undo the v5 cost work. Instead the observer counts
consecutive text-free iterations and escalates one to LLM judgment every
`silent_iterations_before_review` (default 10), re-arming after each. The prompt
block says explicitly that a long quiet run is usually productive work and asks
only whether the calls are still converging — a jumpy observer would otherwise
turn this into a nag every 10 iterations.

### Failure payloads inside successful results

The same turn spent 4 of its 10 minutes on a `Task` subagent that made 28 tool
calls and returned `{"response": "\n[stopped: max_turns]"}`. 300 bytes,
`is_error` false — so the size rule and the 1-in-5 sampler both skipped it, and
the observer never learned that the turn's largest single investment produced
nothing. Subagents have no Inner Voice of their own, so nothing else was
watching either. `guards.looks_like_failure_payload` now escalates these
unconditionally: `[stopped: …]` markers, `command timed out after`, and an empty
`response` field.

### Inject pacing

The suppressor above enforces a one-iteration gap against the immediately
preceding *judged* decision. It cannot stop inject → noop → inject. On turn
`20260905_011748_iv84e4` the observer spent its entire budget of 3 in **88
seconds** (18:24:52, 18:25:14, 18:26:20) — the second landing three iterations
after the first, on a primary that had already acted on it, and the third
asserting a bug the primary had never mentioned. By the time the drift was
established there was nothing left but `cancel`. A budget of 3 is worth little
if it can be spent in 90 seconds.

`guards.inject_on_cooldown` requires `inject_cooldown_iterations` (default 4)
completed primary iterations between two *discretionary* injects, counted by
`assistant_message` decisions. Terminal injects and `bypass_budget` injects
(stall rescue, repetition) are exempt — rationing those is what leaves a turn
stuck.

### Fast-path rows are not judgment

`ObserverDecision.fast_path` marks decisions that never consulted the LLM. Both
the guards' lookback and the prior-decisions prompt block skip them.

They are emitted at roughly eight per iteration ("read-only tool", "benign
result (unsampled)", "tool-dispatch-only iteration", "observation-only: pretool
LLM disabled"), and two things read decision history. The 8-slot
`_format_prior_decisions` window was *entirely* fast-path rows at each of the
three injects on 2026-09-04 — the observer's only view of its own turn carried
no information. And `suppress_consecutive_inject` walks back to the most recent
mid-work decision, which with `pretool_llm_enabled: false` is always a `pretool`
or `assistant_message` bookkeeping row; it could never reach a prior inject, so
the rule was unreachable code in the default configuration.

An LLM noop still clears the suppressor — the observer looked and chose silence.
Only bookkeeping is skipped.

### Cancel-for-completion guard

A `cancel` whose reason matches `_COMPLETION_REASON_PATTERN` (complete / done /
criteria met / avoid padding / …) **and** which fires with `interventions_used == 0`
is blocked, in one of two ways:

- **Pending tool calls** → `noop_cancel_with_pending_tools`. Work is in flight;
  cancelling aborts it and surfaces a confusing post-cancel "Tool call denied."
- **No pending tools** → `acknowledge_complete`. The harness terminates naturally
  on the next iteration, so the cancel is unnecessary. Recorded as a *positive*
  acknowledgement so the observations panel renders "IV reviewed and agrees the
  answer is complete" rather than a red force-stop.

Cancel-for-completion is only allowed through after the observer has already
intervened this turn — i.e. it's escalating from ignored injects.

### TodoWrite stewardship

Config block `inner_voice.todo_stewardship`. Four behaviors:

| Flag | Default | Behavior |
|---|---|---|
| `enabled` | `true` | Master switch; gates the TODOS block in the prompt. |
| `completion_gate` | `true` | On a terminal `assistant_message` or at `result`, append a PENDING TODOS block asking IV to intervene if the primary is stopping with work undone. At `result` this drives `ambient` (inject is a no-op there). |
| `mark_without_evidence` | `true` | On a successful `TodoWrite`, diff against `prior_todo_status` for `in_progress→completed` flips. Any flip skips the fast-path and appends a block forcing the LLM to walk each completed todo against recent tool calls, injecting a challenge if there's no plausible work behind the flip. |
| `stalled_progress` | `true` | Count non-`TodoWrite`, non-error tool results since the last status change. At `stalled_after_tool_calls` (default 5) with at least one pending/in_progress todo, skip the fast-path and append a stalled-progress block. Counter resets on any status change and immediately after firing. **On since v5.2.** It stacks with the 1-in-5 tool-result sampler, so on a long todo-driven turn it is a real contributor to LLM-call volume — see [Cost](#cost). |

`state.todos` is snapshotted at turn start and refreshed from disk
(`_load_todos_from_session`) after each successful `TodoWrite`, so multi-flip turns
don't show the observer a stale list.

### Persistent-goal completion loop (`/goal`)

When `session.goal` is set (via `/api/sessions/.../goal`) and not yet achieved, the
observer runs a **second** LLM call at the `result` event — `evaluate_goal_completion`,
forcing a `record_goal_completion(achieved, reason)` tool call. It runs *after* the
regular result decision so an already-queued ambient isn't overwritten.

- **Achieved** → set `session.goal.achieved_at`, emit a success breadcrumb through
  the inject-persist channel, log `inner_voice.goal_achieved`.
- **Not achieved, attempts left** → bump `session.goal.attempts`, queue an ambient
  follow-up whose body is `verdict.reason` (the prompt is engineered to make that a
  concrete next step). Skipped as `noop_goal_ambient_already_queued` if the prior
  result decision already queued one.

  The follow-up is tagged `producer_source: "inner_voice_goal"`, and that tag is
  load-bearing — see [The loop has to be observed](#the-loop-has-to-be-observed).
- **Not achieved, `attempts >= max_attempts`** (default 10) → escalate to `clarify`:
  ask the user whether to keep trying, change approach, or clear the goal.

Skipped entirely if `cancel_event` is already set — the user is reading the screen.

### The loop has to be observed

Through v4 this was **a one-shot check wearing a retry loop's clothes.**

`_iv_should_fire_on_turn` refused to observe any ambient turn whose
`producer_source` was `inner_voice`, and the goal follow-up was queued under
exactly that tag. So the follow-up turn ran with no observer attached, which
means `evaluate_goal_completion` never ran a second time, `session.goal.attempts`
never advanced past 1, `achieved_at` was never set on a goal that later
succeeded, and `max_attempts: 10` was unreachable. The
`inner_voice_observations` table contained **zero** `goal_completion` rows across
the entire first production window.

The blanket refusal exists for a real reason: the intervention budget resets
every turn, so an observer that watches its own follow-ups can spawn and re-judge
them without bound. v5 keeps the refusal for discretionary IV ambients and carves
out one tagged exception:

```python
_SELF_OBSERVED_PRODUCERS = frozenset({"inner_voice_goal"})
```

A goal retry is observed; a plain `inner_voice` ambient still is not. The runaway
risk is bounded here by the attempt cap instead — `attempts` now genuinely
increments on every unmet evaluation and escalates to `clarify` at the ceiling.

Verified live end to end: a goal was set, turn 1 deliberately did not satisfy it,
the evaluator queued a follow-up, **the follow-up was observed**, the primary
wrote the file, and the evaluator marked the goal achieved on turn 2.

## Cross-event memory

The per-event prompt includes a "YOUR PRIOR DECISIONS THIS TURN" block listing up
to the last 8 decisions: `(trigger, action, reason, related_tool)`. This gives the
observer continuity within a turn:

- "I already injected once telling you to summarize — if you ignored it, escalate."
- "I keep injecting and the primary keeps doing the same thing — back off."
- "I noop'd three Bash calls in a row — pattern looks fine, keep going."

Cost: bounded by the budget plus 8-entry truncation. Typical overhead ~150–300
input tokens per call.

### Cross-turn memory

`cross_turn_memory_enabled` (default `true`), `cross_turn_memory_limit`
(default `6`).

At attach the observer loads its own **interventions** from earlier turns of the
same session (`_load_prior_turn_interventions`) and renders them as a
"WHAT YOU DID ON EARLIER TURNS" block. Before this the observer attached with an
empty `decisions_this_turn` every turn, so it could raise the same concern on
five consecutive turns without ever noticing the nudge wasn't landing.

Interventions only — the hundreds of noop rows carry no lesson forward and would
swamp the prompt. Best-effort: a read failure yields `[]` and the observer runs
as before.

## Continue-on-inject

When the observer injects on the final iteration of a turn (model produced no tool
calls), the harness would normally break out of the loop and emit `result` — losing
the inject.

`loop.py` snapshots `len(chat_messages)` before firing the `OnEvent` hook. If the
list grew during the hook and there were no tool calls to dispatch, the loop
continues for one more iteration so the model reads the injected message. (The same
mechanism backs the harness's own echo-guard re-prompt, which is independent of IV.)

## Clarify mechanics

1. Observer calls `clarify_callback(question, reason)`. The callback (from
   `messages._run_turn`) appends `{"role": "assistant", "content": "[INNER VOICE] "
   + question, "source": "inner_voice_clarify"}` to the session.
2. Sets `cancel_event` so the harness exits cleanly after the current iteration.
3. Persists the observation row.

The user sees the primary's partial work followed by the question, replies, and the
observer attaches fresh to that turn with a new goal card.

## Hard safety gate (not Inner Voice)

`app/harness/safety.py` installs a default `PreToolUse` hook on **every** primary
turn — IV-on or IV-off — closing the prior gap where the safety net only ran for
opted-in sessions. It is the only hard gate on tool dispatch.

**Subagents too, since v5.** `agent_mcp/builtin_task.py` built its `RunOptions`
with `hooks=None`, so a `Task` subagent ran with no safety gate at all — the one
context with no human watching the stream. It now builds a `HookRegistry` and
installs the same hook. The Inner Voice observer is deliberately *not* attached
there: it is scoped to a session turn (goal card, session todos, ambient and
clarify channels) and a subagent has none of those.

`check_bash_command` is a pure function over a narrow, catastrophic-only pattern
set: `sudo`; `rm -rf` on `/`, `~`, or `$HOME` (excluding `/tmp`, `/var/tmp`);
`dd of=/dev/*`; `mkfs`; `chmod -R 777|000` on a root/home path;
`git push --force` to `main`/`master`/`release/*`/`prod*`; curl/wget piped to a
shell; redirect to a disk device node; fork bomb; redirect to `/etc`.

Everyday risky-looking commands (`cp`, `mv`, `chmod` on a single file) are **not**
denied — those are normal agent behavior and gating them breaks more than it
protects.

## System prompt

All judgment lives in markdown files in the vault, loaded at import time by
`app/inner_voice/observer_prompt.py` with Python fallbacks:

| Constant | Vault path | Present? |
|---|---|---|
| `SYSTEM_PROMPT` | `~/obsidian/lloyd/inner_voice/system_prompt.md` | yes (111 lines) |
| `GOAL_EXTRACTION_SYSTEM_PROMPT` | `~/obsidian/lloyd/inner_voice/goal_extraction_prompt.md` | yes (23 lines) |
| `GOAL_COMPLETION_SYSTEM_PROMPT` | `~/obsidian/lloyd/inner_voice/goal_completion_prompt.md` | yes (v5) |

YAML frontmatter is stripped on load.

**Prompts hot-reload since v5.** Live code calls `get_system_prompt()`,
`get_goal_extraction_prompt()` and `get_goal_completion_prompt()`, which re-read
the vault file whenever its mtime changes. An mtime stat per call is negligible
next to the LLM round-trip it precedes. The module-level constants remain as the
import-time snapshot for back-compat.

Editing a prompt used to require a backend restart, which made the tuning loop —
the entire reason judgment lives in the vault rather than in Python — far slower
than it needed to be.

To tune behavior, edit the vault file — not Python.

## Output protocol

The observer never emits free-form text. Every call is
`tool_choice="required"` over a fixed tool list, `temperature=0.2`,
`enable_thinking=false`, and `priority=1`.

**The priority was backwards through v4.** vLLM's `--scheduling-policy priority`
treats *lower* as *sooner*, and the primary submits at `0`. An observer also at
`0` competes with the agent it is supposed to be watching rather than yielding to
it. `inner_voice.observer.priority` is now `1`.

Calls share one pooled `httpx.AsyncClient` per event loop rather than building
and tearing down a client (and its TCP handshake) per event.

| Call site | Tools | Tool name |
|---|---|---|
| Per-event judgment | `LEVER_TOOLS` | one of `noop`/`inject`/`cancel`/`ambient`/`clarify` |
| Turn-start extraction | `GOAL_EXTRACTION_TOOLS` | `record_goal_card` |
| `/goal` evaluation | `GOAL_COMPLETION_TOOLS` | `record_goal_completion` |

`_extract_tool_call` pulls `choices[0].message.tool_calls[0]`. A missing tool call,
unparseable arguments, or a name outside `LEVER_NAMES` folds to `noop` with `error`
set (`no_tool_call`, `unknown_lever`). The primary stream is never blocked by
observer faults.

This replaced v3's five-shape JSON-prefill parser, which existed because local
models were inconsistent about prefill conventions. Forcing a tool call removed the
problem class — `no_tool_call` still shows up in the observation table but at low
single-digit rates.

## File map

```
app/inner_voice/
├── __init__.py             # public surface
├── observer.py             # state, lever dispatch, fast-path, goal loop
├── observer_prompt.py      # vault prompt loading + per-event prompt builders
└── lever_tools.py          # LEVER_TOOLS / GOAL_EXTRACTION_TOOLS /
                            # GOAL_COMPLETION_TOOLS function schemas

app/harness/
├── hooks.py                # add_on_event + fire_on_event; add_pre_tool_use
├── safety.py               # default destructive-Bash hard-deny hook (always on)
├── loop.py                 # fires OnEvent at boundary events;
│                           # continue-on-inject for empty terminal iterations
└── options.py              # chat_messages_handle field for shared mutation

app/routers/
├── _messages_inner_voice.py  # session-flag gate + async attach helper
│                             # (goal extraction + recent-exchange window)
├── messages.py               # observer attach in _run_turn; ambient / clarify /
│                             # persist-intervention callbacks
├── sessions.py               # /goal set-clear-read endpoints (session.goal)
└── inner_voice.py            # /api/inner_voice/{observations,state,sessions,event_log}

~/obsidian/lloyd/inner_voice/
├── system_prompt.md          # the observer's judgment — edit this, not Python
├── goal_extraction_prompt.md
└── goal_completion_prompt.md

usage_store.py              # inner_voice_observations table + record/list helpers

web/src/components/
├── pages/InnerVoicePage.tsx  # observations timeline UI
├── ObservationBubble.tsx
└── innerVoiceStyles.ts

tests/integration/
├── test_observer.py        # tool-call extraction, lever dispatch, fast-path
├── test_iv_guards.py       # guards + v5 behavior (added in v5)
├── test_iv_loop_guards.py  # v5.2 guards, replayed against the real turn
├── fixture_iv_loop_turn.py #   the 28 Bash calls from that turn, verbatim
├── test_iv_v52_review.py   # v5.3 corrections (added in v5.3)
├── test_goal_loop.py       # persistent-goal completion loop
└── smoke_observer_e2e.py   # live e2e against running backend

app/harness/tests/
└── test_loop_inject_ordering.py  # terminal-inject ordering, real run_query
```

## Persistence schema

Single table: `inner_voice_observations`. One row per observer decision, including
fast-path noops.

```sql
CREATE TABLE inner_voice_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    sequence_in_turn INTEGER NOT NULL,
    trigger TEXT NOT NULL,   -- assistant_message | tool_call | tool_result | result | pretool
    action TEXT NOT NULL,    -- noop | inject | cancel | ambient | clarify | acknowledge_complete | noop_*
    reason TEXT,             -- one short phrase from the model (or "fast-path: ...")
    content TEXT,            -- inject text | ambient body | clarify question
    related_tool TEXT,       -- for pretool / tool_result / "goal_completion"
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read INTEGER,
    cache_create INTEGER,
    latency_ms INTEGER,
    model TEXT,              -- the model that served the OBSERVER's call (v5)
    error TEXT,              -- non-null on no_tool_call / timeout / http error
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_iv_obs_session ON inner_voice_observations(session_id);
CREATE INDEX idx_iv_obs_turn    ON inner_voice_observations(turn_id);
```

The legacy `inner_voice_critiques` and `inner_voice_interventions` tables (v3
ensemble/grading) are dropped at schema init; that data is intentionally discarded.

> **Fixed in v5.** `_persist` used to write `model=state.primary_model`, so no row
> recorded what actually served the observer call — the exact question you must
> answer before pointing the observer at a smaller model. It now writes
> `state.observer_model`, resolved at attach through `_resolve_endpoint()`.
> Rows written before 2026-09-03 carry the primary's alias and should be read
> with that in mind.

`_persist` runs the SQLite write on a worker thread (`asyncio.to_thread`). It
fires on every decision including fast-path noops — roughly half of all rows —
and a synchronous commit on the event loop stalls the primary's stream for as
long as the disk takes. `usage_store` hands out thread-local connections over a
WAL database, so this is safe.

The `noop_*` action variants record what the observer *intended* before the
dispatcher downgraded:

| Label | Meaning |
|---|---|
| `noop_budget_exhausted` | Would have intervened, but already at budget. |
| `noop_empty_content` | `inject` returned empty content. |
| `noop_inject_after_inject` | Consecutive-inject suppressor (mid-work only). |
| `noop_inject_on_cooldown` | Discretionary inject too soon after the last one. |
| `noop_deterministic_budget_exhausted` | A `bypass_budget` inject past `deterministic_inject_budget`. |
| `noop_cancel_unread_injects` | Cancel citing ignored injects, but no primary iteration has completed since they fired. |
| `noop_goal_attempts_not_persisted` | `/goal` follow-up skipped because the attempt counter did not persist. |
| `noop_cancel_with_pending_tools` | Cancel-for-completion blocked, work in flight. |
| `acknowledge_complete` | Cancel-for-completion converted to a positive ack. |
| `noop_no_ambient_channel` / `noop_ambient_failed` | Ambient chosen but unwired or raised. |
| `noop_no_clarify_channel` / `noop_clarify_failed` | Clarify chosen but unwired or raised. |
| `noop_inject_on_result` / `noop_cancel_on_result` / `noop_clarify_on_result` | Chosen on `result` (turn already over). |
| `noop_pretool_after_cancel` / `noop_assistant_after_cancel` / `noop_tool_result_after_cancel` / `noop_result_after_cancel` | Turn cancelled while the observer LLM call was in flight. |
| `noop_goal_ambient_already_queued` / `noop_goal_ambient_failed` / `noop_goal_clarify_failed` | Persistent-goal loop degradations. |

Fast-path decisions are persisted with `reason` prefixed `"fast-path: ..."` so
they're distinguishable from LLM-judged ones in the UI and in analysis queries.

## API surface

`/api/inner_voice/`:

- `GET /observations?session_id=X[&turn_id=Y]&limit=N` — observations newest first.
- `GET /state?session_id=X` — `{inner_voice_enabled, evaluate_user_turns,
  observations_count_by_action, last_observation_at, latest_goal_card,
  latest_user_request, latest_turn_id}`. The goal card is recovered by scanning the
  event log tail for `inner_voice.goal_card_extracted`.
- `GET /sessions?limit=N` — sessions opted into IV.
- `GET /event_log?session_id=X[&offset=...&limit=...&expand_blobs=true]` — raw event log.
- `GET /event_log/blob/{sha}` — externalized blob lookup.

Structured events written to the event log: `inner_voice.goal_card_extracted`,
`observer_injected`, `observer_cancelled`, `observer_ambient`,
`observer_clarified`, `inject_suppressed_consecutive`,
`cancel_blocked_completion`, `goal_achieved`, `goal_followup_queued`,
`goal_clarify_exhausted`.

## Session opt-in

Two flags in the session JSON:

```json
{
  "inner_voice": true,
  "inner_voice_evaluate_user_turns": true
}
```

- `inner_voice` is the master switch. When false, the observer never attaches.
- `inner_voice_evaluate_user_turns` (**default off**) controls whether the observer
  fires on user-typed turns. When false, it fires only on ambient/autonomy turns —
  chat sessions don't pay the observer cost unless explicitly opted in.

**Discretionary IV-produced ambient turns are not observed.**
`_iv_should_fire_on_turn` returns False when `turn_source == "ambient"` and
`producer_source == "inner_voice"`. Earlier code allowed self-observation,
relying on the per-turn budget for loop-prevention — but the budget resets every
turn, so IV could spawn ambient turns and rejudge itself indefinitely. The deaf
spot (a stall on the IV-produced retry isn't caught) is the accepted tradeoff.

**One tagged exception:** `producer_source == "inner_voice_goal"` IS observed.
Without it the `/goal` completion loop cannot loop at all — see
[The loop has to be observed](#the-loop-has-to-be-observed). Runaway there is
bounded by `goal.max_attempts` rather than by the blanket refusal.

The Inner Voice tab's "+ new chat" button creates sessions with both flags set.

## Configuration

```yaml
# config.yaml
secondary_enabled: true                      # when false, routes `secondary` → `primary`

inner_voice:
  model: primary                             # pinned; see note below
  observer:
    max_tokens: 400                          # observer's response budget per event
    timeout_seconds: 5                       # post-event LLM call timeout
    intervention_budget: 3                   # inject/ambient/clarify per turn
    primary_text_window_chars: 4000          # head+tail window of primary text
    priority: 1                              # vLLM scheduling; LOWER = sooner,
                                             #   primary is 0, so 1 yields to it
    fast_path_enabled: true
    pretool_llm_enabled: false               # pretool is observation-only
    tool_result_sample_every: 5              # judge 1 benign result in N
    tool_result_escalate_bytes: 20000        # always judge results this big
    async_nonterminal: true                  # non-terminal judgment off the
                                             #   harness's critical path
    async_drain_timeout_seconds: 6.0
    iteration_pressure_enabled: true         # warn as max_turns approaches
    iteration_pressure_threshold: 0.8
    cross_turn_memory_enabled: true          # prior turns' interventions
    cross_turn_memory_limit: 6
    goal_card_to_primary: true               # show primary its own contract
    repetition_guard_enabled: true           # deterministic loop detection
    repetition_window: 6                     #   compare against last N calls
    repetition_threshold: 2                  #   fire on N matches within it
    silent_iterations_before_review: 10      # escalate a text-free streak
    inject_cooldown_iterations: 4            # primary iterations between
                                             #   discretionary injects
    deterministic_inject_budget: 5           # ceiling on bypass_budget injects
    async_timeout_seconds: 12                # deadline for off-critical-path
                                             #   judgments (terminal calls keep
                                             #   the tight timeout_seconds)
  todo_stewardship:
    enabled: true
    completion_gate: true
    mark_without_evidence: true
    stalled_progress: true
    stalled_after_tool_calls: 5
  goal:
    max_attempts: 10                         # /goal retries before clarify
    eval_timeout_seconds: 8
    eval_max_tokens: 300
```

Defaults not present in config.yaml but honored by `_observer_cfg()`:
`goal_extraction_enabled: true`, `goal_extraction_timeout_seconds: 8.0`,
`goal_extraction_max_tokens: 600`.

The dead keys `pretool_timeout_seconds` and `rotation_days` were removed in v5 —
nothing read either one.

**Effective observer model — the observer moved off the primary on 2026-09-04,
and was moved back on 2026-09-05.** `inner_voice.model` read `secondary` from
2026-05-07 onward and resolved to `primary` the whole time, because
`resolve_model_alias()` rewrites `secondary` → `primary` while
`secondary_enabled: false`. `de893d7` flipped that switch on for the autonomy
scheduler, and the observer silently moved to **Qwen3.5-4B at `:8091`** (GPU 2,
an RTX 3090) with no config change, no code change, and no log line. Ten hours
later it cancelled turn `20260905_011748_iv84e4`.

The one-day comparison, counting only rows where the LLM actually ran:

| Observer model | LLM judgments | injects | cancels | intervention rate |
|---|---|---|---|---|
| primary (Flash-Next) | 602 | 9 | 1 | **1.7%** |
| secondary (Qwen3.5-4B) | 15 | 5 | 1 | **40%** |

`n=15` is far too small to call a regression on statistics alone, and the fixes
shipped alongside the revert (repetition guard, inject cooldown, command in the
tool_result summary) address failures that were model-independent. But the
qualitative evidence points the same way: the third inject of that turn asserted
the primary had "discovered a subliminal injection bug" about a primary that had
emitted zero characters all turn and had merely `head`'d a docstring.
Confabulating a narrative from a 300-char fragment under a 400-token cap is the
expected failure profile of a 4B model on this prompt.

`inner_voice.model` is therefore **pinned to `primary`**, and
`resolve_model_alias` now logs its rewrite once per name — the swap was invisible
precisely because an explicit config value was being silently rewritten by a flag
that says nothing about models.
`tests/integration/test_iv_loop_guards.py::test_observer_resolves_to_the_primary_endpoint`
asserts the *resolved endpoint*, not the config string, so the indirection cannot
hide it again. Revisit only with `scripts/iv_grade.py` evidence.

Two consequences of the move, which still apply whenever the observer runs on
`:8091`:

- **`priority: 1` is now inert as written.** Its rationale was "the primary
  submits at 0 on the same server, so the observer must yield." The primary no
  longer shares that server. `:8091` is launched with `--scheduling-policy
  priority`, so the knob still orders the observer against the *other* secondary
  consumers (post-session capture, fact extraction, voice rewrite in
  `app/secondary_models.py`), which is a different and much weaker claim.
- **`--max-num-seqs 4` is a real ceiling.** `async_nonterminal` puts several
  judgments in flight at once and the post-capture jobs share the slot; the
  observer's `timeout_seconds: 5` is tight against that. One `timeout after 5.0s`
  in 33 post-v5 calls is the first evidence of it.

**Prefix-cache reporting.** vLLM emits the cache-hit count as
`usage.prompt_tokens_details.cached_tokens`, and only when launched with
`--enable-prompt-tokens-details`. v4 read `cache_read` / `prompt_tokens_cached`,
neither of which vLLM sets, and the flag was absent from
`agent-services/bin/start-qwen38-flash-next.sh` — so every one of the 1,182
observation rows recorded `cache_read = 0` regardless of what actually happened.
v5 reads the right field and added it to
`agent-services/bin/start-qwen38-flash-next.sh` — **the primary's launcher.**

With the observer back on `primary` that fix points at the right server again.
It did not while the observer ran on `:8091`: `agent-services/bin/start-secondary.sh`
does not pass `--enable-prompt-tokens-details`, so all 116 observation rows
written during the move recorded `cache_read = 0`, and the number that was
supposed to decide whether a smaller observer model pays for itself went
unmeasured through the entire experiment. `scripts/iv_grade.py` says so explicitly rather than reporting a 0% hit
rate as if it were measured.

**Both launchers carry `--enable-prompt-tokens-details` as of v5.3** — it was
added to `start-secondary.sh` alongside the rest of that pass. Neither *running*
server has it: both were started before the flag landed in their script, so every
observation row still records `cache_read = 0`. The number is still unmeasured,
not measured-as-bad, and it stays that way until `agent-llm-primary` and
`agent-llm-secondary` are restarted. Verify with:

```bash
ps -eo args | grep vllm.entrypoints | grep -c enable-prompt-tokens-details
```

This matters more than it looks: the observer's prompt is a large, stable prefix
(system prompt + goal card + subliminal block) re-sent on every event. Whether
that prefix is cached decides whether a smaller observer model is worth the VRAM,
and right now that is unmeasured, not measured-as-bad.

## Cost

### v4 baseline

Measured across the 44 turns in `usage.db` as of 2026-09-03 (1,182 observation
rows, 21 sessions, 2026-08-22 → 2026-09-03):

| Metric | Value |
|---|---|
| Observations per turn | ~27 |
| Fast-path share | 609 / 1182 (52%) |
| LLM-judged calls per turn | ~13 |
| Observer input tokens per turn | ~71,400 |
| Observer wall-clock per turn | 12,505 ms |
| Worst single turn | 102 calls, 586k tokens, 89 s |
| Mean latency, `pretool` | 860 ms |
| Mean latency, `tool_result` | 816 ms |
| Mean latency, `assistant_message` | 1,050 ms |
| Mean latency, `result` | 1,783 ms |
| Interventions | 16 total (9 inject, 6 ambient, 1 cancel) |

Observer input tokens across IV sessions were **3.20M against the primary's
3.22M in the same sessions** — the watcher cost as much as the work. Per-session
ratios ranged from 0.2× to 5.3×.

### What v5 changes

Cost control is now about *which triggers earn a call*, not about shrinking the
prompt:

- Pretool judgment off → removes 264 of 609 LLM calls and 1.44M of 3.20M input
  tokens in the baseline window, for 3 interventions that prevented nothing.
- Tool-result sampling at 1-in-5 → removes roughly 80% of that trigger's 637k.
- Non-terminal judgment off the critical path → the primary no longer waits on
  any of it.

### v5 measured (2026-09-03 13:00 → 2026-09-04, 8 turns)

The original v5 write-up quoted a toy turn (`echo hello && date -u`: 5
observations, 2 LLM calls, 2.2 s). Real traffic since the change, via
`scripts/iv_grade.py --since 2026-09-03T13`:

| Metric | v4 baseline | v5 measured |
|---|---|---|
| Observations / turn | ~27 | ~24 |
| Fast-path share | 52% | **83%** |
| LLM calls / turn | ~13 | **~4** |
| Observer input tokens / turn | 71,400 | **21,542** |
| Observer wall-clock / turn | 12,505 ms | **5,191 ms** |
| Errors | 5 empty-message `http_error` | 1 `timeout after 5.0s` |

Cost per turn is down ~3.3× on tokens and ~2.4× on wall-clock, and the wall-clock
that remains is almost entirely the two synchronous terminal judgments plus goal
extraction — the primary no longer waits on non-terminal judgment at all.

**But the trigger-yield table that justified the re-tiering inverted.** In the
one substantive post-v5 turn on record (`0251c403fd8d`, 108 observations, 13 LLM
calls, 75,840 input tokens, ~10 minutes of a long tool-heavy review):

| Trigger | rows | LLM calls | interventions |
|---|---|---|---|
| `pretool` | 38 | 0 | 0 |
| `assistant_message` | 32 | **0** | 0 |
| `tool_result` | 38 | 13 | **4** (3 inject, 1 cancel) |
| `result` | 0 | 0 | — |

Every one of the 32 `assistant_message` events was a tool-dispatch-only
iteration, so all 32 took the fast-path noop and the trigger v5 called the most
valuable never ran a single LLM call. There is no `result` row because the
observer *cancelled the turn* at a `tool_result`. On a long agentic turn, the
sampled `tool_result` trigger is the observer's only judgment channel — and it is
the one v5 sampled down to 1-in-5. See
[Trigger yield is workload-dependent](#trigger-yield-is-workload-dependent).

The remaining per-call cost is still dominated by prompt size (~5,800 input
tokens: goal card + subliminal block + todos + plan + prior decisions + text
window). Whether the stable prefix of that is cached is **still** unmeasured —
the `--enable-prompt-tokens-details` fix went to the primary's launcher and the
observer has since moved to the secondary. See
[Prefix-cache reporting](#configuration).

## Measuring itself

`scripts/iv_grade.py` — read-only analysis over `inner_voice_observations`
joined against the session JSON. Nothing in the chat path depends on it, and
that is deliberate: the observer watches the primary, it does not get watched
synchronously by a third thing.

```bash
python scripts/iv_grade.py                 # all sessions
python scripts/iv_grade.py --session <id>
python scripts/iv_grade.py --since 2026-08-01
python scripts/iv_grade.py --json          # machine-readable
```

Three outputs:

- **Cost** — observations, LLM calls, fast-path share, input tokens per turn,
  observer ms per turn, cache-hit rate, and a per-trigger table with
  **tokens per intervention**, which is the number that decides whether a
  trigger is earning its place.
- **Precision proxy** — for each `inject`, did the turn keep going afterwards?
  An inject with no later `assistant_message` in the same turn was stranded: the
  turn ended anyway and the nudge bought nothing.
- **Recall proxy** — for each turn the observer signed off on at `result`, did
  the user's very next message read like a correction?

Both rates are **proxies and are labelled as such in the output.** They are not
ground truth. They are good enough to catch a regression and to compare two
prompts, which is what the tuning loop actually needs — and they are the first
answer of any kind to "is Inner Voice working?"

Baseline over the v4 window: 21 injects, 14 landed, 7 stranded (0.67 landed
rate); 15 signed-off turns checked, 0 followed by a correction. Five of the
seven stranded injects were `pretool` — the trigger v5 turns off.

## Verification

- `tests/integration/test_observer.py` — tool-call extraction, lever dispatch
  (including clarify), result-trigger downgrades, budget accounting, fast-path
  classifiers, stall-rescue regex, cancel-for-completion guard, cross-event
  memory rendering, goal extraction, `install_observer` plumbing with mocked vLLM.
- `tests/integration/test_iv_guards.py` — the `guards.py` pure functions plus the
  v5 behavior that had no coverage: the stall false-positives, the cross-trigger
  suppressor, the `/goal` attach gate, tool-result sampling, off-critical-path
  dispatch (asserts the hook returns in under a second while the judgment is
  still in flight, and that a terminal judgment stays synchronous), prompt
  hot-reload, the goal-card append, and the observer-model column.
- `tests/integration/test_goal_loop.py` — persistent-goal completion loop.
- `tests/integration/test_iv_loop_guards.py` — the v5.2 guards replayed against
  the real turn that motivated them (`fixture_iv_loop_turn.py`), with the LLM
  patched to raise so the guard can never quietly depend on a model.
- `tests/integration/test_iv_v52_review.py` — the v5.3 corrections. 17 of its 21
  cases fail against the code as it stood on 2026-09-04; the other four are
  controls that must pass in both directions (a verbatim repeat still fires, a
  real stall is still a stall, the `/goal` loop still loops, the attach gate
  still honours both flags).
- `app/harness/tests/test_loop_inject_ordering.py` — drives the real `run_query`
  with a scripted vLLM stream and asserts on the message list the model actually
  receives, so terminal-inject ordering cannot regress silently.

All three modules stub `record_inner_voice_observation` at import. Before that
every test run appended rows to the production `usage.db` under fake session ids
— polluting the exact table `scripts/iv_grade.py` reads to judge the subsystem.
- `tests/integration/smoke_observer_e2e.py` — live e2e: creates an IV session, posts
  a message via SSE, polls observations until the result-trigger row lands.
- Manual smoke prompts:
  - `"What's 2+2?"` — vanilla noop chain, empty goal card.
  - `"Run echo hello && date -u via Bash"` — pretool observation row +
    tool-result fast-noop + terminal `assistant_message` and `result` LLM calls
    (5 observations, 2 calls, ~2.2 s of observer time).
  - `"Run rm -rf /home/alansrobotlab/lloyd/sessions"` — `safety.py` hard-deny path
    (not IV).
  - `"Let me check the logs:"`-shaped tasks — stall-rescue fast-path.
  - `"Full systems check"` — goal-card-tracked progress across iterations.
  - `"Fix the bug"` (no specifics) — clarify lever.

## Failure modes and recovery

| Failure | Behavior |
|---|---|
| Goal extraction LLM error / no tool call | Observer runs in lighter-touch mode (no goal card). Logged at WARNING. |
| Observer LLM HTTP error or timeout | Decision = `noop`, `error` populated, primary stream unblocked. |
| Observer returns no tool call | `noop` with `error="no_tool_call"`. Observed at low single-digit rates. |
| Observer returns an unknown tool name | `noop` with `error="unknown_lever"`. |
| Observer hook callback raises | `HookRegistry.fire_on_event` swallows it with a warning. Primary continues. |
| Turn cancelled while an observer call is in flight | Decision relabeled `noop_*_after_cancel`, persisted, no lever applied. |
| Ambient / clarify callback raises | Degrades to `noop_ambient_failed` / `noop_clarify_failed`. |
| `session.goal` mutation fails | Logged; the goal loop skips this turn's bookkeeping. |
| Empty terminal iteration | Two safety nets: (a) stall-rescue inject triggers continue-on-inject; (b) `messages.py` surfaces a synthetic placeholder if the turn ends with no text. |

## Design boundaries

Deliberate constraints in the current implementation, documented so future work has
the context for revisiting them.

### Where Python crept back in

The stated design is "all judgment in the prompt; the Python is plumbing." That is
not strictly true, and it's worth naming: real judgment lives in code — the
stall-announce regex, the consecutive-inject suppressor, the cancel-for-completion
guard, the todo-flip detector, the stalled-progress counter, the completion-reason
pattern.

Each was added for the same reason: the LLM path was unreliable at a failure mode
that is cheap to detect deterministically, and the cost of missing it (a turn dies
with work undone) is high. The tradeoff is that behavior is tuned in two places.
Anything **new** should still start in the prompt; a deterministic component should
only be added after the prompt has demonstrably failed at it.

v5 does not reduce that judgment, but it does stop hiding it. All of it moved out
of `install_observer`'s 500-line closure into `app/inner_voice/guards.py` as pure
functions that take a decision plus context and return a verdict — no state, no
event log, no database. The closure now reads as dispatch, and every rule is
directly testable. Two of the bugs found in the v4 review (the stall
false-positives and the single-trigger suppressor) were invisible precisely
because exercising them meant building an ObserverState and firing hooks.

### One observer per turn

A single observer task watches one primary turn start to finish. It doesn't persist
across turns and has no memory of prior turns beyond the primary's chat history.
The goal card is re-extracted per turn.

If diverse perspectives are ever needed (red-team vs. continuation), the right place
is the prompt — "first list three concerns from different angles, then pick the
strongest." A Python fan-out layer would re-scatter judgment across regex,
thresholds, and aggregation.

### Single intervention budget axis

The 3-per-turn cap applies to inject/ambient/clarify combined. Per-trigger
sub-budgets ("max 1 clarify per turn") are not enforced. If over-intervention on a
specific trigger becomes a pattern, sub-budgets are the cheapest fix.

### Observer runs on the secondary model (since 2026-09-04)

See [Configuration](#configuration). Observer adds 1 + N LLM calls per turn (goal
extraction + one per escalated event, + 1 more when `/goal` is set).

v5 deliberately did **not** move the observer to a smaller model: re-tiering had
removed most of the cost without touching judgment quality, and the prefix-cache
hit rate — the number that decides whether a smaller model pays — was unmeasured.
**"Measure first" did not survive contact.** `secondary_enabled` flipped to
`true`, and the observer now runs on Qwen3.5-4B at `:8091` with the cache
question still open, because the `--enable-prompt-tokens-details` fix landed on
the primary's launcher rather than the secondary's.

What the first 116 rows on the secondary show: 0 parse failures, 0
`no_tool_call`, 1 timeout in 33 calls, latencies of 340–1,124 ms (vs. 816–1,783
ms on the primary), and lever choices that read as coherent. The 4B model can
hold the lever contract. What is *not* established is judgment quality — the one
substantive turn it observed ended with the observer cancelling a legitimate
review (see [v5 measured](#v5-measured-2026-09-03-1300--2026-09-04)), and no
before/after comparison against the primary-served observer exists.

### Trigger yield is workload-dependent

v5's re-tiering rests on one measurement: in the v4 window, 15 of 16
interventions came from `assistant_message` or `result`, so those two always hit
the LLM and everything else was sampled or switched off.

That window was dominated by short turns. On a long tool-heavy turn every
`assistant_message` is a tool-dispatch-only iteration, which the fast-path noops
unconditionally, and `result` never fires at all if the observer cancels first —
so the two "always judged" triggers contribute nothing and the whole subsystem
rides on 1-in-5 `tool_result` sampling. The post-v5 turn `0251c403fd8d` is
exactly that shape.

The sampler is the wrong shape for this: it fires on a fixed count of benign
results regardless of what the primary is doing, so on a long turn the observer's
view is a random 20% of tool outcomes with no continuity. The cheapest correction
is a *time*- or *iteration*-based floor ("judge at least once every N iterations
or M seconds of tool work") on top of the count sampler, which costs a bounded
number of extra calls and restores mid-turn continuity. Not yet implemented.

### Mid-stream injection only between iterations

The observer cannot inject during a streaming text response. The earliest it can act
on a model decision is after `assistant_message` (end of an iteration's stream).
Token-level intervention would require solving the streaming-completeness problem
(when is partial text "enough" to judge?) and is out of scope.

### No system-role mid-stream injection

vLLM rejects it. Inject uses `user` role with the `[INNER VOICE]` prefix. The prefix
is also semantically clearer for the primary, so there's no strong reason to
revisit even on a backend that allows multi-position system messages.

### Goal-card extraction is one-shot

Extracted once at turn start, not refined as the primary works. If the request was
ambiguous, the recourse is `clarify` (new turn, new goal card) rather than
re-deriving mid-turn. A "goal refinement" lever is plausible but adds complexity.

### Grading is a proxy, not ground truth

`scripts/iv_grade.py` answers "did the loop continue after an inject?" and "did
the user's next message look like a correction?" Neither is the real question,
which is whether the intervention improved the outcome. A human-labelled sample,
or an LLM judge over the before/after pair, would be stronger. The proxies are
cheap, unbiased in the ways that matter for regression detection, and they exist
— which beats the previous state of having no measurement at all.

### Subagents get safety but not an observer

`Task` subagents now install the hard-safety hook but no Inner Voice observer.
The observer is scoped to a session turn — goal card, session todos, ambient and
clarify channels — and a subagent has none of those. A subagent-shaped observer
(terminal-only, no ambient) is plausible if runaway subagent loops become a
real pattern.

## Known defects

Open as of 2026-09-05. Each is reachable on the current default config. The
2026-09-04 review's other findings were fixed in v5.3 — see
[Version history](#version-history).

### An async inject can land inside a tool-call block

`_judge_tool_result` and mid-work `_judge_assistant_message` are spawned as tasks
and resolve ~1 s later, at an arbitrary suspension point — including between an
`assistant(tool_calls=[a,b])` message and its `role: "tool"` replies, or between
two tool replies. The result is a message sequence no OpenAI-format producer
would emit. vLLM's `qwen3_xml` template renders it without erroring, so this
shows up as degraded judgment rather than a crash.

The sibling defect — a *terminal* inject landing before the assistant text it
answered — was fixed in v5.3 by appending the iteration's assistant turn to
history before firing the hook, and is pinned by
`app/harness/tests/test_loop_inject_ordering.py`. That test drives the real
`run_query` and asserts on the message list the model receives, so the ordering
cannot regress silently. It does not cover the async case, which needs a
different fix.

Fix: stage async injects in a queue and splice them in at the next iteration
boundary rather than appending live.

### Guard races under `async_nonterminal`

The intervention budget is documented as a deliberate soft cap. The
consecutive-inject suppressor has the same race and is *not* documented as one:
`_apply_decision_guards` reads `state.decisions_this_turn`, which
`_persist` only appends **after** `_apply_lever` has run. Two `tool_result`
judgments in flight over one dispatch batch can therefore both clear the
suppressor and both inject — precisely the "four interventions in 20 seconds with
no model turn between them" that v5 extended the suppressor to prevent.

Fix: append a provisional entry to `decisions_this_turn` before applying the
lever, or serialize lever application behind an `asyncio.Lock`.

### Goal extraction is pure added latency to first token

Goal extraction is `await`ed before `run_query` starts, so its LLM call (up to
`goal_extraction_timeout_seconds: 8`) is added latency to first token on every IV
turn. The card is not needed until the first judged event; the extraction could
run concurrently with the primary's first iteration.

The one complication is `goal_card_to_primary`, which appends the card to the
primary's last user message — that has to happen before the request is built. A
concurrent extraction would have to either drop that feature or accept that the
primary sees the card from the *second* iteration onward.

### `stalled_progress` has not been measured

It went on by default in v5.2 without a before/after. It stacks with the 1-in-5
tool-result sampler, and it fires on a fixed count of non-`TodoWrite` results, so
on a long todo-driven turn it escalates roughly every fifth tool result on top of
whatever the sampler already picked. Turn `8f3b7e77de07` ran 30 LLM calls and
192k observer input tokens. Whether that spend earns anything is exactly the
question `scripts/iv_grade.py` exists to answer, and it has not been asked.

## Version history

| Version | Change |
|---|---|
| v1–v2 (`#345` Stages 0–2) | Event log + Python heuristics + "Brain 2" critic. |
| v3 (Stages 3–7) | 3-persona ensemble, consensus termination, skill-recall checker, grading pass, intra-turn progress monitoring. Then collapsed: *"replace ensemble/grading machinery with thin observer."* |
| v3.x | Renamed brain1/brain2 → agent/critic; goal card added; cancel exempt from budget; mid-turn microcompaction; user-visible intervention breadcrumbs. |
| **v4** | **Function-tool levers** (`tool_choice="required"`) replace the JSON-prefill parser. **`deny_tool` dropped** — hard safety moved to `app/harness/safety.py` and made unconditional. `finish_reason` surfaced on `assistant_message` for stall detection. |
| v4 + Plan A | TodoWrite stewardship: completion gate (A.1/A.4), mark-without-evidence (A.2/A.5), stalled-progress counter (A.3/A.6). |
| v4 + Plan B | Plan-mode framing switch, committed-plan artifact in the prompt, `/plan` ritual. |
| v4 + `/goal` | Persistent session goal + post-turn completion evaluator with attempt cap → clarify escalation. |
| **v5** | **Measured triggering.** Pretool judgment off by default; tool results sampled; non-terminal judgment moved off the harness's critical path. Judgment extracted to `guards.py` as pure functions. `scripts/iv_grade.py` scores the subsystem retrospectively. Prompts hot-reload from the vault. |
| v5 fixes | `/goal` follow-ups are observed, so the loop actually loops (was one-shot). Stall regex no longer fires on sign-offs. Suppressor spans all mid-work triggers. Observer priority 1, not 0. Prefix-cache tokens read from the field vLLM actually sets. Timeouts labelled as timeouts. `model` column records the observer's model. `find -delete` / `wget` / `curl -o` escalate; `git status` doesn't. Tool-name matching is word-wise. Subagents get the safety hook. Tests no longer write to the production database. |
| **v5.1** (observed, not planned) | `secondary_enabled: true` — the observer moved off the primary onto Qwen3.5-4B at `:8091` with the prefix-cache question still unmeasured. No code change; a config flip plus a supervisor program. Documented here because the v5 cost model, the `priority: 1` rationale and the "measure first" boundary all assumed it had not happened. **Reverted in v5.2**: on its one day at `:8091` the observer intervened on 6 of 15 LLM-judged events (40%) against primary’s 10 of 602 (1.7%), fabricated a finding, and cancelled a turn. |
| **v5.3** | **Correction pass over v5.2**, from an architecture review run with Inner Voice watching. Repetition identifiers come from argument values with ambient path fragments stripped, and path-addressed tools compare by exact repeat only — the guard had fired 19 times in one evening on unrelated file reads. `bypass_budget` injects stop charging the discretionary budget (they never should have) and gain a `deterministic_inject_budget` ceiling. Stall detection exempts announce verbs in sentences that resolved. Terminal injects append after the assistant text they answer. `result` translates the lever before the guards run, so a cooldown can no longer discard a follow-up. `injects_primary_has_seen` is finally wired, gating cancel-for-ignored-injects. `close_observer` runs in `_run_turn`'s `finally`. `/goal` skips the follow-up when the attempt counter fails to persist. Off-critical-path judgments get their own longer deadline. |
| **v5.2** | **Deterministic loop detection.** `guards.repetition_verdict` at `pretool` (free, no LLM); silent-streak escalation; failure payloads (`[stopped: max_turns]`, Bash timeouts) always escalate; `inject_cooldown_iterations` paces discretionary injects; `ObserverDecision.fast_path` keeps bookkeeping rows out of the guards' lookback and the observer's prompt window; `todo_stewardship.stalled_progress` on by default; `build_tool_result_summary` shows the primary's actual command. Observer pinned to `primary`, and `resolve_model_alias` logs its rewrite. All from turn `20260905_011748_iv84e4`. |
