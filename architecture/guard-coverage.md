---
segment: architecture
tags: [architecture, safety, guards, coverage, injection]
type: reference
status: implemented
date: 2026-10-01
---

# Guard coverage — the union of what no guard on this box sees

Each guard on Lloyd's dispatch path documents *its own* exclusions well:
`app/harness/action_review.py` says it never sees assistant prose or tool
results, `app/harness/outbound_content.py` says it is handed `tool_input` and
nothing else, `app/harness/safety.py` says a `sudo`-shaped command is denied by
the hook and not at dispatch, `agent_mcp/_injection_probe.py` says it probes five
tools on background sessions. All true, all local to one file. What nobody had
was the **union**: the list of dispatch paths and content channels that no guard
on this machine sees at all. A reader who asks "what would an injected
instruction have to arrive inside to reach a worker untouched?" gets one partial
answer per guard and has to do the intersection themselves, which is the work
this page does once — and the first pass at it under-counted the guards, because
it read the guard directory instead of the refusal sites. The inventory below is
the re-derived one; §3, §8 and the two rows added on 2026-10-01 are what that
re-derivation found.

## How to read this page

Every exclusion states the **mechanism** — the line of code or the eligibility
rule that makes the path invisible — and the **command** that proves the sentence
against the tree you are reading. Run the command; do not trust the sentence. A
claim with no command is not allowed here, and neither is a claim copied in from
another document or a handoff: a landed-state sentence rots, and this page is
mostly about things whose absence is exactly what an attacker or a bad day
would exploit.

All commands were run on 2026-10-01 in the tree of `main` at `17f48780` plus the
diff that introduced this page (round `SM_20261001_060920`, item #1948), from the
repository root, and **re-run in full the same day at `b03c6c0b`** — every block
still hits, the two guards added on that re-run hit too, and nothing under this
page moved in between. Three conventions keep the greps honest:

- absence claims use `git grep`, and every absence command is paired with a
  **positive control** — the same pattern against a file where it must hit — so a
  zero that means "typo" never reads as a zero that means "absent";
- `git grep -c` prints nothing and exits 1 on a miss, which is the answer, not a
  failed command;
- a line number is a convenience, not the claim. Quote the symbol.

## The guards, and where each one is wired

| Guard | Question it answers | Sees | Wired at |
|---|---|---|---|
| `app/harness/safety.py` | is this Bash command catastrophic | the `command` string | the harness hook (`build_turn_options`) **and** the aggregator's `call_tool` with `at_dispatch=True` |
| `app/harness/policy.py` | who is authorised to make this side effect | tool name + tier + grant scope | `install_policy_hook`, per turn builder, only where a grant scope exists |
| `app/harness/outbound_content.py` | what is inside the arguments | string argument values, direction `out` | beside the safety hook, and per scoped turn builder |
| `app/harness/action_review.py` | is this call what the task asked for | the worker's prompt and the calls it has made | `build_turn_options`, `kind == "stream"`, worker platforms only |
| `agent_mcp/_injection_probe.py` | is this result instruction-shaped | the text of five tools' results, background sessions | `agent_mcp/main.call_tool` |
| `agent_mcp/session.py::_check_injection` | is this memory entry instruction-shaped | the `entry` of `memory_add`, the `new_text` of `memory_replace` | inside the two writers (`agent_mcp/session.py:299`, `:351`) — it **refuses**, `ErrorCode.INJECTION` |
| `agent_mcp/egress.py` | is this destination allowed for this scope | the host of four network tools | `agent_mcp/http_tools.py:119`, `:412`, `:523`, `agent_mcp/browser.py:694` — its `DECISION_DENY` branch is armed only by `harness.egress_policy.enforce`, a key **no `config.yaml` sets** (#1965, which also made the value parsed rather than `bool()`-coerced, so a quoted `"off"` cannot arm it). The posture is published by `egress.network_report()`'s `policy.enforce`, live at `GET /api/dashboard` → `network.policy` and rendered `enforcing` / `recording only` |

```
git grep -n "_check_injection" -- agent_mcp/session.py
git grep -n "egress.guard\|egress.aguard" -- agent_mcp/http_tools.py agent_mcp/browser.py
```

The last two rows are the 2026-10-01 re-derivation. They were missing from the
first version because it inventoried `app/harness/` — the guard directory — and
both refusals live elsewhere and refuse from inside a tool handler rather than
from a hook. A union page has to enumerate the **refusal sites**, not the
directory that smells like guards; that is the difference between §3 and §8
being true and being almost true.

```
git grep -n "install_default_safety_hook\|install_policy_hook\|install_outbound_content_gate\|_install_action_review" \
  -- app/routers/turn_options.py app/harness/safety.py app/autonomy.py \
  -- workers/sources/_common.py agent_mcp/builtin_task.py
```

That output is the guard-by-path matrix, and it is the reason the exclusions
below are shaped the way they are: **four** of the guards reach a worker's stream
turn through one builder — `install_default_safety_hook`, `install_policy_hook`
where a grant scope exists, `_install_action_review` on `kind == "stream"`, and
the outbound gate, which is armed from *inside* the safety installer at
`app/harness/safety.py:526` and so appears in that command only because
`app/harness/safety.py` is in its path list. The first version of this page said
"three", and its own table cell reading "beside the safety hook" contradicted it:
the command could not see the transitive arm, so the sentence was copied from the
command and inherited its blind spot. Three other turn builders construct their
own `HookRegistry` and install a different subset — `builtin_task.py`,
`autonomy.run_task`, `workers/sources/_common.py` — and the last of those three
installs no Bash hook at all, which is what §6's dispatch floor is for.

```
git grep -n "GATE_ARM_POINTS\|def stale_gate_arm_points\|def find_unarmed_dispatch_paths" \
  -- app/harness/outbound_content.py
```

That last block names the one **machine-checked** half of this table: the arm
points of the outbound gate are derived from the tree and re-measured on every
suite run. It covers one guard and runs only from tests (#1963) — which is why
this page is prose and why the prose has to be re-run by hand.

---

## 1. An interactive turn is the one turn nobody watches

Three separate eligibility rules point the same way, and they agree by accident
rather than by shared code:

**(a) No action reviewer.** `_install_action_review` returns `None` before
reading the mode when the session's platform is not in
`sessions_io.NON_USER_PLATFORMS`, and `build_turn_options` only calls it for
`kind == "stream"`:

```
git grep -n "if platform not in sessions_io.NON_USER_PLATFORMS" -- app/routers/messages.py
git grep -n "_install_action_review" -- app/routers/turn_options.py app/routers/messages.py
git grep -n "if kind == \"stream\"" -- app/routers/turn_options.py
```

The reviewer call is at `app/routers/turn_options.py:249`, inside the
`kind == "stream"` branch that opens at `:247` — so a `flush` or `voice` turn
never reaches the reviewer even on a worker session, and the eligibility test in
`messages.py` (the first command) is the second gate on the same path.

**(b) No injection probe.** `apply` returns the result untouched unless the
session is a background one, and a background session is an **id shape** — four
parts, an 8-digit date and a 6-digit time — while a chat id has three:

```
git grep -n "if name not in PROBED_TOOLS or not is_background" -- agent_mcp/_injection_probe.py
git grep -n "four parts" -- app/harness/service_control.py
```

**(c) No effect-ledger replay guard.** The #544 ledger dedupes a retry's second
`email_send` by the *queue item* a worker turn is running, carried in
`_meta` as `lloyd/effect_scope`. `main.py` records in its own comment that the
key is absent for an interactive turn, which leaves that turn unguarded on
purpose; the ledger then files it under a `turn:`-prefixed shadow scope that its
own queries exclude:

```
git grep -n "Absent for an interactive turn" -- agent_mcp/main.py
git grep -n 'SHADOW_SCOPE_PREFIX = ' -- agent_mcp/_tool_effects.py
```

The consequence is a standing property, not a bug report: **a person's chat has
no content-shaped guard on it at all.** It is also why the guard set is aimed at
`NON_USER_PLATFORMS` — an interactive turn has a human reading it, which is the
assumption every three of these exclusions rest on, and the assumption that a
prompt injected into a *chat's* fetched page has to defeat is a person, not a
regex.

## 2. The tool-result content channel

No guard reads a tool result and can act on it.

`outbound_content` says so in its own scope section:

```
git grep -n "It never sees a tool" -- app/harness/outbound_content.py
```

`action_review` reads the *first 400 characters* of a result only to classify
what the other gates did with the call, and the reviewer's whole view is built
from the prompt and the calls:

```
git grep -n "content = str(result_evt" -- app/harness/action_review.py
git grep -n "The reviewer's whole view" -- app/harness/action_review.py
```

So the text of a fetched page — the single largest channel by which somebody
else's words enter a worker's context — is read **for what it says** by exactly
one thing on this box, `_injection_probe`, and that thing's own exclusions are
§4. The second command above is the exception that makes the qualifier
necessary: `outcome_of` does read a result, and stops at a prefix test for
`Tool call denied:` and a fragment test for `disabled by configuration`
(`app/harness/action_review.py:189-197`) to work out what the *other* gates did.
It never asks what the page said. Say "reads a result" without that qualifier
and this section contradicts itself four lines after proving the reader exists.
This is deliberate in
`action_review` (a reviewer that reads the fetched content can be told what to
think by the page; pinned by
`tests/test_action_review.py::test_the_reviewer_sees_only_the_prompt_and_the_calls`)
and deliberate in `outbound_content` (a rule that reads a `contacts_get` response
fires on ordinary research). The union is the part that is not deliberate: the
two right-to-leave-alone reasons sit side by side and nothing previously named
the gap they leave together.

## 3. Model and provider traffic

Nothing on this box scans what a tool was **sent**, or what came back from the
provider. One thing does scan text the model authored, and it is not on this
path: `agent_mcp/session.py::_check_injection` reads the `entry` of `memory_add`
and the `new_text` of `memory_replace` against six families and
answers `ErrorCode.INJECTION` (`agent_mcp/session.py:299`, `:351`). It sits
between a model and loaded memory, never between a model and a call, so it
narrows this section's claim without weakening it. Its families and the probe's
come from one table, `agent_mcp/_injection_patterns.py`, and are **different
subsets on purpose** (#1959): the gate refuses, so it keeps its six narrow
shapes verbatim; the probe records, so it reads the broader ones. Only
`invisible_chars` — NUL included, which the probe did not see before — is read
by both. "Syncing" the two widens a refusal; the declared difference is pinned
in `tests/test_injection_pattern_relation.py`.

```
git grep -n "_check_injection" -- agent_mcp/session.py
```

The probe is wired at one call site, and it is a tool-dispatch site:

```
git grep -ln "_injection_probe" -- "*.py" ':!tests/*'
```

The only file that answer names is `agent_mcp/main.py`. Assistant prose is a
harness event the reviewer does not even branch on, and the same pattern finds
the event's own definition, so the miss is the reviewer's and not a typo:

```
git grep -n 'kind == "tool' -- app/harness/action_review.py
git grep -c "text_delta" -- app/harness/action_review.py   # nothing, exit 1
git grep -c "text_delta" -- app/harness/events.py          # control: the event exists
```

What that leaves open, stated plainly: a provider that echoes an injection back
as assistant prose, a poisoned model response, or a tool result the model
paraphrases into the next turn all travel the widest band of traffic on the box
with no guard on them. Closing it is not a wiring job — it is the inference
channel OpenAPPA's `FILE-MEDIATION.md` names as uncloseable on its side too,
because a check that reads model output and acts on it becomes a function of the
same content it is supposed to distrust.

## 4. `_injection_probe`'s own exclusions

The probe is the only content guard, so its limits are the union's floor. Five of
them, each from the module:

```
git grep -n "PROBED_TOOLS = \|SCAN_CHARS = \|if not text or is_error\|^PATTERNS" -- agent_mcp/_injection_probe.py
git grep -c '^    ("' -- agent_mcp/_injection_probe.py      # the family count
```

1. **Five tools.** `Read`, `http_fetch`, `http_request`, `vault_read`,
   `browser_snapshot`. A tool that is not on that allowlist dispatches its result
   to the model unprobed, and the list is hand-maintained while the tool surface
   is assembled at runtime from the MCP pool — so the allowlist does not grow with
   the box. See §5.
2. **Errors are skipped**: `if not text or is_error: return result`. An injection
   that arrives in an error payload — a refused call's explanation, a stack trace
   quoting a page — is never scanned.
3. **`SCAN_CHARS = 200_000`** of leading text. Content that matters at the tail of
   a large result is outside the scan.
4. **Eight regex families.** Every one is a spelling, not an intent:
   `role_header`, `ignore_instructions`, `you_must_now`, `run_the_following`,
   `conceal_from_user`, `persona_swap`, `new_system_prompt`, `invisible_chars`. A
   paraphrase of any of them ("first, please re-read the rules below and follow
   those") matches nothing. `tests/test_injection_probe.py` has a
   *misses-ordinary-prose* fixture list precisely because precision is what
   decides whether `warn` can ever be enabled; the same list is the inventory of
   what will not be caught.
5. **Background sessions only.** A `task:` child of one *is* probed: the single
   production call site always hands a resolver in
   (`agent_mcp/main.py`, `background = is_background_session(sid,
   parent_of=_safety_parent_of)` in the injection-probe block),
   so the helper classifies the subagent by its parent. The exclusion is the
   **miss**, not the missing argument: a `task:` id whose parent
   `agent_mcp/_subagent_registry` cannot resolve stays `False` there, so its
   fetched text is still not recorded — the miss costs a shadow row and nothing
   more, which is the one fail-open this guard can afford.
   The other four no longer share that answer (#1961). The desktop refusal
   (`agent_mcp/main.py`, the `name.startswith("desktop_")` branch) and the dispatch
   `Bash` service-control refusal (the `if name == "Bash"` branch,
   `check_bash_command`) now key on `service_control.classify_session(...)`, which
   answers three ways —
   `attended`, `background`, `unknown` — and an unresolvable `task:` id is
   `unknown`, which refuses: `unknown` is not evidence of a person, and
   `desktop_capture` has no lease behind it. Install provenance
   (`app/harness/supply_chain.py`, `_attended_by_session_id`) treats the same miss
   as unattended, because its attended branch returns before the registry is read
   and before any journal row exists. Every miss logs one
   `guard_parent_unresolved` warning naming the session id, the guard that asked
   and why, so a systematic miss is countable instead of reading as a clean
   window.
   The read-only sandbox is the fifth consumer and was the last inline copy
   (#2025): `_tool_sandbox.is_sandboxed_session` resolved a `task:` id itself and
   returned `False` on a miss, so a bench trial's subagent whose registry row had
   been evicted got the write toolbox. It now answers `True` on a miss and logs
   the same event with `guard=tool_sandbox`. It does not use `classify_session`:
   the question is "is the parent a bench or eval trial", not "is the parent
   unattended", and an ordinary worker's subagent must stay unsandboxed.

```
git grep -n "def _safety_parent_of" -- agent_mcp/main.py
git grep -n "parent_of=_safety_parent_of" -- agent_mcp/main.py
```

## 5. Content that reaches a worker from anywhere else

The consequence of §4.1 stated separately, because after §3 it is the widest band
left:

```
git grep -n "name in _injection_probe.PROBED_TOOLS" -- agent_mcp/main.py
git ls-files -- 'agent_mcp/builtin_*.py'
```

The first command is the allowlist test at the dispatch seam; the second is the
builtin tool modules in this tree. Two of the names the second one prints are the
highest-volume text channels the box has — `builtin_bash.py` hands back the stdout
of arbitrary commands, which is where a `curl`'s page, an installed package's
README and a build log's pasted issue comment arrive, and `builtin_task.py` hands
back a subagent's entire final response — and neither `Bash` nor `Task` is among
the five names §4's command printed. The email, calendar and contacts tools are
wider still and cannot be enumerated from here at all: they are not served from
this tree, so no `git grep` in this repository lists them — which is the point. A
hand-maintained allowlist of five names cannot be kept in step with a surface that
has sources outside git, so the set has to be read as a decision about *which*
channels are watched rather than as a description of the surface. Everything
absent from the list reaches the model unprobed. None of that is a hole the probe
was written to close; it is the area the probe was *not* written for, and a union
page is where it belongs.

## 6. `sudo`, and any other hook-only label

`safety.py`'s hard-deny table is consulted twice — by the hook, and again by the
aggregator at dispatch so a turn builder that forgot the hook is still covered —
but a label in `_HOOK_ONLY_LABELS` is skipped on the dispatch pass:

```
git grep -n "_HOOK_ONLY_LABELS" -- app/harness/safety.py
git grep -n "at_dispatch and label in _HOOK_ONLY_LABELS" -- app/harness/safety.py
```

The mechanism, from the comment above the definition: the pattern `\bsudo\b`
matches the word inside a `grep -rn sudo` or an `echo`, and the module records
that the paths which never installed the hook ran seven such commands harmlessly
before 2026-09-14, while `sudo` on this host needs a password — so the label
protects nothing at dispatch and would refuse ordinary research. The residue is
the honest one to publish: **for a label in that set, enforcement exists only on
a turn that installed the hook.** Adding a label to `_HOOK_ONLY_LABELS` moves it
out of the dispatch floor, so the set is worth keeping small on purpose.

## 7. A nested RPC call skips the hook layer

Programmatic tool calling (`architecture` of #P9, `app/harness/rpc_policy.py`)
lets a Bash child call read-only tools over a loopback socket. That path is not a
harness turn:

```
git grep -n "nested call skips every PreToolUse hook" -- app/harness/rpc_policy.py
```

The module says it plainly and keeps v1 read-only as the mitigation. What it
means for the union: `safety.py`, `policy.py` and `outbound_content.py` are all
PreToolUse hooks, so a nested call is seen by none of them — only by the
allowlist the socket stamped when the Bash call started, plus the part of
`agent_mcp/main.call_tool` it genuinely re-enters: the **probe**. The dispatch
`Bash` hard-deny pass does not apply here and cannot: `_rpc_call` re-enters
`call_tool`, but a nested `Bash` is refused a rung earlier by `FIXED_DENY`
(`app/harness/rpc_policy.py:76`, admitted at `agent_mcp/_rpc.py:219`), so the
`if name == "Bash"` branch at `agent_mcp/main.py:529` never sees the call. That
is a stronger statement than the one it replaces, not a weaker one — and read-only
is precisely the probe's class, so what a script pulls back *is* scanned.

The whole section is latent today: `harness.rpc.enabled: false` in `config.yaml`
(the `rpc:` block, `config.yaml:765`), and the module says the feature ships off.
Named anyway, because the exclusion belongs to the mechanism, and the mechanism
is one config key from live.

```
git grep -n "^FIXED_DENY" -- app/harness/rpc_policy.py
git grep -n "await call_tool(name, arguments" -- agent_mcp/main.py
```

## 8. Nothing on the content axis blocks the call an injection asked for

The floor below the floor. Both **injection** guards record; neither refuses.
(Without "injection" this heading was false on the day it was written: the
credential gate, the egress table and `session.py`'s memory check all refuse
content — every one of them keyed on a payload *shape*, which is the distinction
this section is actually making.)

```
git grep -n "^MODES = \|^DEPRECATED_MODES" -- app/harness/action_review.py agent_mcp/_injection_probe.py
```

`action_review`'s set is `("off", "shadow")` — `warn` was removed on the
measurement in that module's docstring (#1944: 174.1 would-be interruptions per
day, 1 of 963 positives above tier 1, `injected` the argmax in 0 of 57,325 rows).
`injection_probe` still *ships* `warn`, and `warn` is the only behaviour on the
box that puts a word back into the conversation. `config.yaml` does not have it
"off" — it sets `harness.injection_probe.mode: shadow` (`config.yaml:758`), which
records hits and stays out of the result — and the module says the decision to
turn `warn` on belongs to a measured hit sample, not to itself. So every
exclusion above is an exclusion from a *recorder*. A prompt injection that
reaches a worker today is written down after it acted, and everything standing
between an injected instruction and the call refuses on shape: `safety.py`'s
command shapes, `policy.py`'s tiers, `outbound_content.py`'s credential rules,
`egress.py`'s destination table (unarmed — #1965) and `session.py`'s
memory-entry check. Not one of them asks whether the call is what the task asked
for. That is the sentence this page exists to make unavoidable.

## What closed the "we cannot even tell" case (#1948)

Until 2026-10-01 both guards resolved their mode with
`mode if mode in MODES else DEFAULT_MODE`. A typo (`mode: shodow`) booted, the
guard ran at the default, and the only surface that could show the miss was the
corpus — which reads exactly like a clean window when the instrument stopped
recording. That is an *epistemic* hole under every exclusion above: with it open,
"no hits" meant either nothing arrived or nothing looked.

An unparseable value now stops the process at boot, naming the key, the value and
the valid set (`app/config.py::GUARD_MODE_KEYS`,
`validate_guard_modes`, refused at the import of `CONFIG` beside the existing
`$LLOYD_CONFIG_OVERLAY` refusals), and the per-turn read path still returns a
usable mode so a typo can never silently un-install a recorder mid-turn. Pinned
by `tests/test_config_guard_modes.py`, `tests/test_action_review.py` and
`tests/test_injection_probe.py`.

## What is **not** an exclusion (so nobody re-lists it)

- **Spoken and flushed turns are not hookless.** Every session turn kind goes
  through `build_turn_options`, which installs the safety hook unconditionally:
  `git grep -n "install_default_safety_hook" -- app/routers/turn_options.py`, and
  `git grep -n "build_turn_options" -- app/routers/voice.py` shows the voice path
  building its options there.
- **The autonomy and worker turn paths are not Bash-free zones.** The aggregator
  re-runs `check_bash_command` with `at_dispatch=True` for every Bash dispatch,
  "whatever hooks the caller installed" — see the command in §6 and the
  docstring beside it. The §6 residue is the hook-only labels, not the table.
- **Grant-gated tools are refused without a grant on scoped paths**, and
  `grant_create` is banned on every non-interactive scope (`install_policy_hook`
  in the §matrix command above names the builders that install it).

## File-gated safety state (#2024)

The guards above judge a call. Four *files* in `~/.local/state/lloyd-automod/`
gate the loop itself, and each is read with a bare `exists()` or a float parse —
so the file says a state holds and only the ledger (`promotions.jsonl`) can say
when it began, why, and who ended it. Every edge that is a write appends one
row from the call that makes it. The guardian cannot import
`scripts/automod/state.py`, so the event names are spelled on both sides and
pinned to each other in `tests/test_guardian_rollback.py`.

| Flag | Read by (what it gates) | Create edge → event | Remove edge → event |
|---|---|---|---|
| `BROKEN` | the guardian's tick (stands down), the promoter, the gate, `round start`, autocode's reaper, the dashboard | `gstate.AutomodState.set_broken`, from `guardian.py::escalate` → `broken_set` (`by`, `reason`, `already_broken`) | `state.clear_broken(by=…)`, the route `round recover` takes → `broken_cleared` (`by`) |
| `pause` | the guardian's tick only — it observes and does not act while `pause_remaining` is positive, capped from its own snapshot | `state.set_pause`, the promoter's and `round restart`'s restart legs → `pause_set` (`by`, `started_at`, `seconds` after the cap, `expires_at`, `refreshed`) | `state.clear_pause` → `pause_clear` (`by`). Expiry is not an edge: nothing unlinks a lapsed lease, so `expires_at` on the set row is what dates it |
| `promotions-halted` | the promoter, the gate, `round start`, autocode, the guardian's liveness predicate, the dashboard | `set_halted` on either surface (flap quarantine, vault tripwire) → `promotion_halt_set` | `clear_halted(by=…)` → `promotion_halt_clear` (#1365) |
| `rollback_request.json` | the guardian's tick (performs it), the promoter and autocode (hold while one is pending) | `state.request_rollback` → `rollback_requested` | `clear_rollback_request` on either surface → `rollback_request_cleared` (`by`) |

```
git grep -n "_EVENT = " -- scripts/automod/state.py agent-services/guardian/gstate.py
git grep -n "def set_broken\|def clear_broken\|def set_pause\|def clear_pause\|def clear_rollback_request" -- scripts/automod/state.py agent-services/guardian/gstate.py
git grep -n "BROKEN_PATH.unlink\|PAUSE_PATH.unlink\|HALTED_PATH.unlink" -- scripts app workers agent_mcp
```

What this does not cover: a flag removed by hand (`rm BROKEN`) leaves no row —
the escalation alert therefore names `round recover`, not the file — and the
guardian does not attest a flag it finds missing. A row is fail-open on the
edges that sit inside a restart or an escalation: the flag is written first and
a ledger that cannot be appended costs the row, never the lease or the BROKEN
state. Before 2026-10-01 only the halt pair and `rollback_requested` existed;
the ledger of that date held 28,351 rows and none of the other five kinds.

## Keeping this page honest

A new guard, a new eligible-session rule, or a new value in `GUARD_MODE_KEYS`
changes the union. When one lands: re-run every block here in the same commit,
update the stamp in "How to read this page", and add the exclusion the new guard
*creates* — a guard that sees channel X usually makes "everything except X"
worth naming.

And when the page itself is reviewed, re-derive the guard list from the
**refusal sites**, not from `app/harness/`: enumerate what can answer
`denied`/`refused`/`ErrorCode.INJECTION` and work backwards to who wrote it. The
one review this page had before this one found its two omissions that way, and
neither omission was in a file a reader would think to look in.

## Review log

- 2026-10-01 — **stale** (not drift: every command here still hits at
  `b03c6c0b`). The guard inventory was short of the two refusals living outside
  `app/harness/` — `agent_mcp/session.py::_check_injection`, which blocks a
  memory write, and `agent_mcp/egress.py`, which blocks a destination and is
  unarmed — so §3 and §8 were overstating; §4.5 published an exclusion the tree
  refutes (a `task:` child of a worker *is* probed — the real gap is the resolver
  *miss*, #1961); §7 claimed the dispatch `Bash` hard-deny covers a nested call,
  which `FIXED_DENY` makes unreachable; and "three of the five guards live in one
  builder" contradicted the table's own "beside the safety hook" row, because the
  matrix command could not see `app/harness/safety.py:526`. Filed #1959, #1960,
  #1961, #1962, #1963.
