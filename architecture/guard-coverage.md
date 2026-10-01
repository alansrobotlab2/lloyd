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
instruction have to arrive inside to reach a worker untouched?" gets five partial
answers and has to do the intersection themselves, which is the work this page
does once.

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
repository root. Three conventions keep the greps honest:

- absence claims use `git grep`, and every absence command is paired with a
  **positive control** — the same pattern against a file where it must hit — so a
  zero that means "typo" never reads as a zero that means "absent";
- `git grep -c` prints nothing and exits 1 on a miss, which is the answer, not a
  failed command;
- a line number is a convenience, not the claim. Quote the symbol.

## The five guards, and where each one is wired

| Guard | Question it answers | Sees | Wired at |
|---|---|---|---|
| `app/harness/safety.py` | is this Bash command catastrophic | the `command` string | the harness hook (`build_turn_options`) **and** the aggregator's `call_tool` with `at_dispatch=True` |
| `app/harness/policy.py` | who is authorised to make this side effect | tool name + tier + grant scope | `install_policy_hook`, per turn builder, only where a grant scope exists |
| `app/harness/outbound_content.py` | what is inside the arguments | string argument values, direction `out` | beside the safety hook, and per scoped turn builder |
| `app/harness/action_review.py` | is this call what the task asked for | the worker's prompt and the calls it has made | `build_turn_options`, `kind == "stream"`, worker platforms only |
| `agent_mcp/_injection_probe.py` | is this result instruction-shaped | the text of five tools' results, background sessions | `agent_mcp/main.call_tool` |

```
git grep -n "install_default_safety_hook\|install_policy_hook\|install_outbound_content_gate\|_install_action_review" \
  -- app/routers/turn_options.py app/autonomy.py workers/sources/_common.py agent_mcp/builtin_task.py
```

That output is the guard-by-path matrix, and it is the reason the exclusions
below are shaped the way they are: three of the five guards live in one builder,
and three other turn builders construct their own `HookRegistry` and install a
different subset.

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
else's words enter a worker's context — is read by exactly one thing on this box,
`_injection_probe`, and that thing's own exclusions are §4. This is deliberate in
`action_review` (a reviewer that reads the fetched content can be told what to
think by the page; pinned by
`tests/test_action_review.py::test_the_reviewer_sees_only_the_prompt_and_the_calls`)
and deliberate in `outbound_content` (a rule that reads a `contacts_get` response
fires on ordinary research). The union is the part that is not deliberate: the
two right-to-leave-alone reasons sit side by side and nothing previously named
the gap they leave together.

## 3. Model and provider traffic

Nothing on this box scans what the model sent, or what came back from the
provider.

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
5. **Background sessions only, and a `task:` child counts as one only if a parent
   resolver is handed in** (`is_background_session(..., parent_of=...)` returns
   False for a subagent with no resolver, by design, so the subagent's own tools
   are not probed even though its parent's are).

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
allowlist the socket stamped when the Bash call started, plus whatever
`agent_mcp/main.call_tool` still does (the Bash hard-deny pass, and the probe,
both of which run at dispatch and so do apply).

## 8. Nothing on the content axis blocks

The floor below the floor. Both content guards record; neither refuses:

```
git grep -n "^MODES = \|^DEPRECATED_MODES" -- app/harness/action_review.py agent_mcp/_injection_probe.py
```

`action_review`'s set is `("off", "shadow")` — `warn` was removed on the
measurement in that module's docstring (#1944: 174.1 would-be interruptions per
day, 1 of 963 positives above tier 1, `injected` the argmax in 0 of 57,325 rows).
`injection_probe` still *ships* `warn`, and `warn` is the only emitting behaviour
on the box; it is off in `config.yaml` and the module says the decision belongs to
a measured hit sample, not to itself. So every exclusion above is an exclusion
from a *recorder*. A prompt injection that reaches a worker today is written down
after it acted, and the only thing between an injected instruction and the call
is `safety.py`'s command shapes, `policy.py`'s tiers and `outbound_content.py`'s
credential rules — none of which asks whether the call is what the task asked
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

## Keeping this page honest

A new guard, a new eligible-session rule, or a new value in `GUARD_MODE_KEYS`
changes the union. When one lands: re-run every block here in the same commit,
update the stamp in "How to read this page", and add the exclusion the new guard
*creates* — a guard that sees channel X usually makes "everything except X"
worth naming.
