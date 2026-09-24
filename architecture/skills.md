---
segment: architecture
relations:
  related-to:
  - architecture/autonomy-jobs.md
  - architecture/subliminal.md
  - architecture/tools.md
  - architecture/automod.md
tags: [architecture]
summary: 'Skill system: SKILL.md procedures in ~/obsidian/skills/, the four surfaces
  that read them (prompt index, turn-start injector, dispatch-time deliverer, vault-round
  loader), and the phantom-tool gate.'
type: reference
date: '2026-09-20'
status: implemented

---

# Skill System

Skills are on-demand procedures Lloyd reads before executing. They are plain
markdown files, not code packages — a directory with a `SKILL.md` in it, and
optionally supporting files (`autolink/autolink.py`, `arxiv/scripts/`,
`github-pr-workflow/references/` + `templates/`).

**Until 2026-07 this document described a different system.** It was written
against OpenClaw, and named a marketplace (ClawhHub), an install tool
(`skills_install`), a security-validation pipeline, a built-in skill directory
under `~/.npm-global/lib/node_modules/openclaw/skills/`, an installed-skill
directory at `~/.openclaw/skills/`, and a claim that skills are *not* loaded
into the system prompt (`maxSkillsInPrompt=0`). None of that is true of Lloyd.
Neither path exists on this box, no marketplace or installer code exists in the
tree, and the skills index **is** in the system prompt on every turn. The doc
also named `skills_get` and `file_read` as the way to read a skill; both are on
the phantom-tool ban list (`scripts/skill_lint.py:211`,
`tests/test_skill_tool_names.py:39`) — this file was itself an instance of the
defect described under "Phantom tool names" below. Its skill inventory listed
34 skills, of which 8 have since been archived and 3 no longer exist at all.
The history is kept here rather than deleted because the OpenClaw vocabulary
still turns up in the vault, in old skills and in `nightly-skills-management`.

## Where skills live

| Location | What it is |
|----------|------------|
| `~/obsidian/skills/` | the library — 194 skills on disk, **189 advertised** |
| `~/obsidian/skills/.archived/` | 193 retired skills, kept on disk and in git |
| `~/lloyd/skills/` | second root, configured everywhere, **does not exist** |

**One definition of where skills live, since #1294.** `agent_mcp.skills.SKILLS_DIRS`
is the pair the code ships unioned with `config.yaml skills.directories` — built-in
roots first, so a config entry never silently outranks them — de-duplicated and
`~`-expanded at import. One walker, `agent_mcp.skills.iter_active_skills`, reads
that list, and the prompt index, `GET /api/skills`, the Mission Control tab count,
`scripts/skill_lint.py`, `prefetch`'s skill cache and the MCP tools all consume it.
Before this, five code paths and one test helper each did their own `iterdir()` over
the same directories and only two of them applied the quarantine rule, so the
surfaces disagreed about *what a live skill is* — 189 advertised, 187 listed, 194
counted, about one vault. `prompt_builder._CANON_SKILLS_DIRS` survives as a
test-facing patch point whose `None` means "whatever `SKILLS_DIRS` is at call time",
read per build rather than copied at import; it used to be a second hardcoded pair
anchored to the *repo* location, which inside an automod worktree resolved to
`<worktree>/home/obsidian/skills` — a directory that has never existed, so a prompt
built in a round advertised **no skills at all** while the Skills page listed the
live vault.

`~/lloyd/skills/` has never existed on this box. It is harmless — every reader
skips a missing directory — but it means the second root is not a tested path
against the live config; it is tested against a temp tree instead
(`tests/test_skills_single_walk.py`).

**Retirement is by directory, or by frontmatter, and the walker enforces both.** It
skips dot-prefixed directories and any skill whose `status:` is in
`_QUARANTINE_STATUSES`, so moving a skill into `.archived/` removes it from every
model-facing surface while keeping it on disk and in git history. The `status:`
quarantine is the other lever, and it is **not** the unused one the 2026-09-04 pass
found: 189 of the 194 skills on disk read `status: active`, and five
(`groundskeeper-loop`, `groundskeeper-research`, `nightly-behavior-test`,
`nightly-morning-briefing`, `nightly-prompt-audit`) read `status: archived`
while still sitting in the live directory. Those five are now absent from every
surface; until #1294 they were absent from the index and the MCP tools but still
listed, as `enabled: True`, by the Skills page (#1292).

## The four surfaces that read a skill

A skill can reach a turn four ways, and they are deliberately different
mechanisms rather than one with options.

### 1. The index, in the system prompt

`prompt_builder._load_skills_index` (`prompt_builder.py:626`) walks an optional
overlay root (`<overlay>/skills`, when `LLOYD_OVERLAY_DIR` is set — the
autoresearch bench runner passes one; what it may put there is bounded by
`_canonical_prompt_paths()` in `scripts/autoresearch/common.py`, today
SOUL.md / MEMORY.md / USER.md, so no variant has ever carried a skill)
plus both canonical roots,
dedupes by directory name (first root wins) and emits one line —
`Available skills: ai-engineer-monitor, alfie-monitoring, …` — wrapped by
`build_system_prompt` (`:274`) as:

```
<available_skills>
Available skills: …
</available_skills>
Note: relevant skill content is automatically injected into each
user message as <context> when matched.
```

It is names only: 189 of them, 4,028 chars, ~1k tokens. That is measured, not
estimated — the skills index is one of the named components in the
`PROMPT_BUDGET` line `log_prompt_size` writes once per build (`:73`), against
an 80,000-char tripwire. The note at the end exists because the index alone
would read as "call something to load these"; what actually happens is surface
2.

`include_skills_index` defaults to `True` and every production caller takes the
default (`app/routers/messages.py:1850`, `:2036`, `:2181`,
`app/routers/voice.py:222`, `autonomy.py:1605`). The `False` branch is for tests
that assert on the rest of the prompt.

**A quarantined skill is excluded from the index, and that is not cosmetic.**
`_is_quarantined_skill` (`:603`) reads the frontmatter `status:` and drops
anything in `_QUARANTINE_STATUSES` — `inactive`, `archived`, `disabled`,
`retired`, `quarantined`. The vocabulary is **imported from
`agent_mcp.skills`** (`:117`) rather than restated, with a hardcoded fallback
only for the case where MCP cannot be imported at all, because advertising a
skill the reader then declines is the same failure as naming a tool that does
not exist: the prompt promises something that does not work.
`tests/test_prompt_builder_overlay.py:196` pins the two lists together.

The parse is deliberately cheap — a 2,000-char head read and a line scan, not
YAML — because this runs for every skill on every prompt build.

### 2. The turn-start injector (prefetch)

`prefetch.py` scores the *user's message* against every skill and injects the
winner's body into a `<context>` block ahead of the turn. Thresholds
(`prefetch.py:41-44`):

| | value |
|---|---|
| `SKILL_THRESHOLD_FIRST` | 3.0 — inject full body |
| `SKILL_THRESHOLD_SECOND` | 4.0 — inject second skill as excerpt |
| `SKILL_BODY_MAX` | 6000 chars |
| `SKILL_EXCERPT_MAX` | 500 chars |
| `SKILL_CONSTRAINTS_MAX` / `SKILL_EXCERPT_CONSTRAINTS_MAX` | 1000 / 300 chars — hard-constraint lines carried from past the cut (#657) |

Rendered as `<skill name="…" score="…">` and, for the runner-up,
`<skill name="…" score="…" excerpt="true">` (`:930`, `:941`). When nothing
matches *and* the message is a genuine new task rather than a continuation, a
`<skill-hint>` nudges toward `skills_search` (`:1002`) — the low-confidence
branch was dropped as pure noise, because the model trusts a low-scoring
auto-pick ~95% of the time regardless.

Scoring lives in `agent_mcp/skills._score_skill` (`:237`), shared with the MCP
tool so the two cannot drift: name hits ×3, description ×2, tags ×1.5, body
×0.3 capped at `_BODY_HITS_CAP` = 4. **A body hit can never qualify a skill on
its own** — `require_metadata_hit=True` scores zero without a name/desc/tag
overlap. That rule is #311: generic English tokens in a query bag-of-words
matched arbitrary 5-15 KB bodies and pulled `powerpoint` onto graph-classifier
work twelve times.

Because the scorer weights the *name* ×3, a name sharing a token with the query
wins regardless of meaning, and two skills were renamed on 2026-09-04 for that
reason alone: `https-migration-gotchas` → `tls-migration-gotchas` (it matched
every prompt containing a URL, on the token "https", and was taking the top
slot on fetch prompts) and `web-lookup` → `web-search-and-fetch` (it carried
neither of its own trigger verbs in its name and lost to `youtube-content`,
7.2 vs 5.6, on "fetch <url> and summarize it"; it now wins at 8.6).

Two performance properties are load-bearing. Token sets are memoized on the
cached skill dict (`_skill_token_sets`, `:209`) — re-tokenizing ~1.5 MB of
bodies cost ~83 ms of GIL-held CPU on *every* turn and starved the other
prefetch legs; it is ~1 ms now. And the skill list is rebuilt only when a
`SKILL.md` mtime changes, checked at most every 15 s (`_skills_signature`,
`:397`).

`_stem` (`:139`) is a single-suffix plural collapse, not a real stemmer, and
exists for one logged failure: "full systems check" picked `claude-sdk-check`
over `system-health-check` because "systems" ≠ "system".

This surface is **chat-path only**. `autonomy.py` builds its own prompt and
never calls prefetch, so a scheduled task gets its skill from surface 4
instead.

### 3. Dispatch-time delivery (#536, default-off)

`app/harness/skill_dispatch.py`. The turn-start injector scores prose; what it
structurally cannot do is key on the tool the agent is about to call. On a
boilerplate worker prompt there is nothing protocol-shaped to match, so the
rule that matters — "never reach for yt-dlp on this box", "restart this unit,
not that one" — is absent at the only moment it is load-bearing.

This is a second injector, not a replacement. It runs **inside** the PreToolUse
walk on a drafted call, keyed on tool name plus argument pattern. The drafted
call is **not executed**: the loop returns a *non-error* synthetic tool result
carrying the matched `SKILL.md`, and the model re-issues the call informed.

**`is_error=False` is the entire point of the second outcome.** The same
intercept expressed as a deny comes back `is_error=True` and is booked into
`tool_errors` (appended at `autonomy.py:1792`, reported at `:1807`) — the very
number this feature exists to improve, so a deny would make the fleet look
sicker exactly where it is being taught something.
`HookRegistry.fire_pre_tool_use` recognises
`skillDeliver` alongside `deny` (`app/harness/hooks.py:148`) and `loop.py:1943`
renders it, in the same shape as the synthetic `ToolSearch` result above it.

**A deny beats a deliver regardless of registration order.** The walk holds a
deliver as provisional and keeps going (`hooks.py:133`, `:146-148`), so a catastrophic
`Bash` is blocked rather than answered with a protocol card by a deliverer that
happened to register first. `install_skill_dispatch_hook` is still called after
`install_default_safety_hook` (`app/routers/messages.py:1929`) — not because
order decides the outcome, but so the walk reads in the order that matters.
The Task subagent path installs it the same way, beside its safety hook in
`agent_mcp/builtin_task.py`, under the same flag and with nothing
`already_injected` because a subagent gets no turn-start skills (#750). The
two routes without it say why at the site: `build_ambient_turn` is a short
decide-and-stop turn that would pay the round-trip for nothing, and the sync
`post_message` has no caller in the tree.
`tests/test_task_subagent_skill_dispatch.py` pins the subagent install and
`tests/unit/test_skill_dispatch.py` the one-install-plus-two-exclusions shape
of `messages.py`.

Three rules ship (`:112`), ordered so the specific protocol precedes the
general one that also describes it:

| skill | trigger |
|---|---|
| `voice-mode` | an action verb against `agent-tts` / `agent-livekit-server` / `lloyd-voice-mode.service`, in either word order; or launching `voice_mode.py` directly |
| `restart-lloyd` | `supervisorctl … restart\|stop\|start\|signal`, or `systemctl --user restart …lloyd…` |
| `youtube-transcript` | `yt-dlp`, `youtube-transcript-api`, `transcriptExtractor` |

Every trigger costs the model one extra round-trip (~2.7 s TTFT here), so a
false positive is expensive and the patterns name failure modes that were
actually logged, not topics. All three are **action-gated**: `supervisorctl
status agent-tts` is every health check on this box, and delivering a protocol
card on a read is a pure false positive.

`DispatchRule.fields` is a whitelist, not a hint — a rule that names
`("command",)` cannot fire on a `Write` whose `content` happens to mention
`supervisorctl`. A skill already injected this turn by prefetch is not
delivered again (`injected_skill_names` parses the `<skill name="…">` tags
prefetch rendered, rather than reaching into its internals), and a skill is
held at most once per turn — otherwise the re-issued call is held again and the
turn budget burns on one protocol. A rule naming a skill that is not on disk
logs and passes the call through; it never swallows a call with an empty lesson
attached.

**Default-off, and the config key is absent.** `enabled()` reads
`harness.skill_dispatch.enabled`, where an absent key, an unreadable config or
a false flag all mean not installed. There is no `skill_dispatch` block in
`config.yaml` today, so the feature is off; `skills: [...]` restricts the
active rule set for a per-protocol rollout. Both keys are set by a human — the
automod preflight denies the runtime config.

`eval/run_skill_dispatch_probe.py` is the offline probe: it replays drafted
tool calls harvested from real session transcripts against the rule set with no
model and no tool, reporting `spurious_rate` (the acceptance metric verbatim:
triggering dispatches over total) and a per-protocol compliance rate. The
"after" number is labelled **predicted**, because a rule firing proves the
protocol reached the model, not that the model obeyed.
`tests/unit/test_skill_dispatch.py` pins the rest, including that a deny beats
a deliver in both registration orders.

### 4. Autonomy, by name

A scheduled task binds its skill in frontmatter: `skill_name` (a slug) or
`skill_path`. `autonomy._load_skill_content` (`autonomy.py:931`) resolves a
value containing `/` or ending `.md` as a path, and anything else as
`~/obsidian/skills/<slug>/SKILL.md`; the body is pasted into the task prompt
under "Follow the skill instructions below" (`:976`).

A task with neither is unrunnable, and says so once per task id rather than
silently dead-lettering (`:806-815`). `hold_reason` reports it as `"no skill"`
(`:882`). Both fields are in `_parse_task_file`'s `fallback_fields` (`:121`), so a task
file that needed the degraded parser does not lose its skill binding.

Note this path reads the file directly — it does **not** go through
`agent_mcp.skills._load_skill`, so it is the one surface a `status:` quarantine
does not reach.

## The MCP tools

Two, both read-only (`agent_mcp/annotations.py:49`), both served by the
`skills` module inside the lloyd-mcp aggregator (`agent_mcp/main.py:84`,
`:149`):

- **`skills_search(query, max_results=10)`** — ranked over name, description,
  tags and body, using the same `_score_skill` prefetch uses.
- **`skills_read(name)`** — the full raw `SKILL.md` for a directory name.

Both are in the ToolSearch baseline (`config.yaml:498-499`), so they are
advertised on every request rather than needing discovery.

There is no install tool, no catalog, no remote source, and no `skills_get` or
`skills_list` — those two names are *banned*, see below.

## Frontmatter

`_parse_frontmatter` (`agent_mcp/skills.py:42`) splits on the first `\n---`
after a leading `---` and `yaml.safe_load`s it; a parse failure degrades to an
empty mapping and a body, never an exception. Fields the loaders actually read:

| field | read by | note |
|---|---|---|
| `description` | scorer (×2), `/api/skills` | required by `skill_lint` |
| `tags` | scorer (×1.5) | required by `skill_lint` |
| `category` | scorer (not weighted), `/api/skills` | on 139 of 194 |
| `status` | both loaders | quarantine; 189 of 194 read `active`, 5 `archived` |
| `written_by` | `skill_lint` authorship count (#774) | `{job, date}` from a writer skill, or `interactive`; on 0 of 194 when introduced |

The directory name is the skill's `name` for every purpose that matters — the
scorer's ×3 weight is on the directory name, not the frontmatter `name:` field
— which is why a rename is a real intervention and why the two 2026-09-04
renames above worked.

`scripts/repair_skill_frontmatter.py` exists for the 2026-06 corruption family
that `skill_lint` flagged as DEAD: an indented `status:` after a flow-style
tags list, orphan col-0 list items, duplicate top-level keys, and a *second*
`---` block stacked under the first holding the real name and description. It
re-emits one valid block, verifies with `yaml.safe_load` before writing, and
leaves anything still broken untouched and reported. Dry-run by default;
`--apply` to write.

## Phantom tool names

**A skill naming a tool that does not exist is worse than no skill.** The model
calls it, gets an unknown-tool error, and takes whatever fallback the skill
documents. This is the defect this document itself carried.

The 2026-09-04 tool-choice investigation traced Lloyd's habit of shelling out
to `curl` to a skill literally named `websearch` — the one prefetch injected on
any web-shaped message — which instructed `web_search` and `web_fetch`. Neither
has ever existed in Lloyd; the tools are `http_search` and `http_fetch`. The
call failed, and the same skill named Bash + curl as the recovery path. The
same defect was in SOUL.md and MEMORY.md, which did not merely mention phantom
tools but *mandated* them: the output-shape gate told the model its skill
sequence must run through `skills_get`.

That is a class of defect, not one typo — an auto-generated skill can mint a
plausible tool name at any time, and nothing checked. The sweep found 40 skills
on the first pass and more against the live aggregator; **91 skills were
rewritten** onto the real tool with argument names fixed where they differed,
and three were archived as documenting tools that never existed.

Two gates now, deliberately at different cadences:

- **`tests/test_skill_tool_names.py`** is the hard one. It walks exactly what
  the prompt advertises — `_active_skill_files` mirrors
  `_load_skills_index`, dot-dirs and quarantine included — and fails on any
  banned name. `KNOWN_UNFIXED` is **empty and stays empty**: a new entry means
  a regression, not a grandfathering. A second test guards the guard, asserting
  against live aggregator discovery that nothing in `PHANTOM_TOOLS` has since
  become real. A third asserts `web-search-and-fetch` exists, names all three
  `http_*` tools, and keeps its localhost carve-out — Bash + curl is correct
  there, and the positive half of the fix has to stay.
- **`scripts/skill_lint.py`** carries the same list as a `PHANTOM_TOOL`
  category so drift shows up in the weekly report rather than only when someone
  runs pytest.

`terminal` — the OpenClaw name for Bash, found as a call in 69 places — is
matched only as `` `terminal` `` or `terminal(`, because it is also an ordinary
English word. `read_text`/`write_text` were deliberately **not** rewritten:
those are pathlib, not tools.

Seven skills are exempt in both gates, because their job is to say these names
are not real: `web-search-and-fetch`, `nightly-skills-management`,
`trajectory-skill-mining`, `nightly-skill-consolidation`, `create-hermes-plugin`,
`autonomy-task-diagnosis`, `pipeline-dispatch`.

## skill_lint

`scripts/skill_lint.py` is an advisory sweep writing
`~/obsidian/autonomy/skill-lint-report.md`. Its subject set is
`agent_mcp.skills.iter_active_skills()` (#1294) — the same live set every other
surface reports — where it used to walk its own hardcoded `~/obsidian/skills` with
no quarantine rule, so its weekly total counted retired skills the model can no
longer reach and never saw a second configured root. Narrowing the set is only
admissible because every finding is computed from the same walked records: a live
skill with a DEAD frontmatter or a PHANTOM_TOOL reference is still reported, and the
report now names the roots it scanned. Six categories: DEAD (unparseable
frontmatter, or description *and* tags both empty — the live scorer would
return 0 for any query), MISSING_DESC, DRIFT (description is output-framed
rather than trigger-framed), DUPLICATE (difflib name ratio ≥ 0.85 **and** description ratio ≥ 0.60), STALE (mtime
> 90 days and `status != active`), PHANTOM_TOOL.

Every count row carries a fourth column, "is this count trustworthy?", from
`CATEGORY_TRUST` in the script (#903): DRIFT and STALE are marked `no` — DRIFT
passes any ≤10-character opener untested, STALE returns before its age check for
every `status: active` skill — so an all-zero run renders a qualified verdict
naming them, never "✅ Clean". The qualification lives in code because the report
is rewritten wholesale and a hand-written column did not survive one run.

Beside the table, not in it, the report counts authorship (#774): live skills
whose `written_by.job` names an unattended writer, those marked `interactive`, and
those with no stamp — read off the front matter, so "which skills did a job
write?" no longer needs a walk over vault commit subjects. It is a measurement
rather than a category, so it never moves the verdict; the stamps come from the
writer skills' own write step, and files written before that step carry none.

Beside the table, never a row in it, the report carries a **SIZE** section
(#624): body lines and chars per live skill with front matter excluded, p50/p90/
max, the count over `MAX_BODY_LINES` (100), each skill's chars past the chat
injector's 6000-char cut and its largest `##`/`###` block, and — from the vault's
git history — the before/after embedded size of the five `SPILL_SAMPLE` skills.
Size is a cost only on the uncapped routes: the autonomy task prompt and the
worker prompt splice the whole file in, while the chat injector truncates. Each
route books what it embedded as a `skill.embedded` event on its session's event
log (`app/skill_embed.py`; routes `prefetch`, `prefetch_excerpt`,
`autonomy_task`, `worker_prompt`), so a report can say which route paid for what.
The 100-line cap is advisory; whether it becomes a failure is a person's call.

Advisory only — no automatic deletion or rewrites, and it exits 0 always so the
nightly pipeline does not fail on lint findings. Autonomy task #70 runs it
weekly.

## The automod vault route validates the loaders

The vault is a live shared tree with no worktree, and `prompt_builder` reads
SOUL.md and the skills straight from it, so a self-modification round touching
`skills/**` cannot be gated in isolation — an edit is live the moment it is
saved. `scripts/automod/vault_round.py` therefore validates, commits exactly
the named paths, and reverts on failure.

`skills/**` is one of three `VALIDATED_GLOBS` (`:60`, with `lloyd/**` and
`autonomy/**`) — the paths that feed a prompt or the scheduler, where the
loaders are actually run against the change:

1. Every changed `.md` opening with `---` must parse to a mapping. A skill
   whose front matter broke is a skill nobody can find.
2. The system prompt must still build (`build_system_prompt()` returning ≥ 500
   chars), and **each touched skill must still load** — through
   `agent_mcp.skills.skill_load_defect` (`:180-186`), not `_load_skill`.

Scoped to the diff on purpose: a pre-existing broken file elsewhere in the
vault must not block every round — the same delta principle as pyflakes and tsc
in the code gate. A loader failure reverts the round's paths; success commits
exactly those paths on `main` and records a `vault_land` ledger event.

Quarantine used to make this gate refuse both ways the skills lifecycle retires
a skill: the validator asked `_load_skill` whether each touched skill loads,
and `None` is also what a *quarantined* skill returns, so setting
`status: archived`, or renaming a skill into `.archived/`, was reported as
damage and the whole batch reverted. Fixed 2026-09-19 (`d49f6fd`): the mover
into a dot-tree is recognised as a retirement, and the question asked is now
"is this skill damaged", which a quarantine is not (#777).

## The Skills page

`web/src/components/pages/SkillsPage.tsx`, backed by two GET routes in
`app/routers/skills.py`: `/api/skills` (list) and `/api/skill-content?name=`
(raw text).

**The list route is the walker's output, since #1294.** `get_skills` is
`iter_active_skills()` rendered — dot-skip, `_QUARANTINE_STATUSES`, first root
wins — and `/api/skill-content` resolves through the same `skill_roots()`, so the
page cannot offer a skill whose own content endpoint then 404s. It used to be the
odd one out: its own `iterdir()` over `skills.directories`, its own frontmatter
split instead of `agent_mcp.skills._parse_frontmatter`, no quarantine rule, and a
`metadata.hermes` / `metadata.openclaw` fallback for description and category (the
last live remnant of the old vocabulary). That fallback is what made the page *lie
in both directions at once*: a null-valued `metadata.openclaw:` — which is what most
of the migrated skills carry — is `None`, `None.get("category")` raised, and the
route's bare `except Exception: continue` reported the skill as nonexistent. Seven
live skills (`alfie-monitoring`, `backlog-triage`, `email-calendar-monitoring`,
`github-watch`, `python-library-pipeline`, `ralph-loop`, `workspace-audit`) were
invisible on the page while the five `status: archived` ones were listed. The
fallback is gone rather than null-guarded because no live skill's description or
category lives only under those keys — `tests/test_skills_single_walk.py::test_no_live_skill_needs_the_hermes_fallback_the_route_dropped`
re-measures that against the vault, and a missing category is now an empty field
instead of a deleted row.

`enabled: True` remains hardcoded, because it is not state: nothing can turn a skill
off except retiring it, and a retired skill is now absent from the response on every
surface rather than listed and refusing (#1292).

**The page is read-only (#1293).** It used to call three routes that were never
registered — `POST /api/skill-toggle`, `POST /api/skill-content` and
`POST /api/skills/refresh` — and since `fetch` resolves on a 404/405 and none of
those calls checked `ok`, the toggle flipped and stayed flipped and Save closed
the editor as if SKILL.md had been written. The toggle, the editor and the three
client calls are gone; Refresh re-runs the `GET /api/skills` loader; a skill is
edited in the vault. `tests/test_api_contracts.py` now matches every
`${API_BASE}/…` path in `web/src/api.ts` against the app's registered routes, so
a client call to a route that does not exist fails the suite.

`app/routers/mc_ui.py:457` counts the same directories for the Mission Control
tab summary, which is a count only and does not go through either loader.

## What maintains the library

`nightly-skills-management` (autonomy task #83, daily) mines session
transcripts for procedural knowledge and creates or updates skills;
[[autonomy-jobs]] is the long version, under trace2skill. Its authoring rules, added
2026-09-04, are the direct descendants of the phantom-tool sweep: validate
every tool name against the live aggregator, never restate a tool's parameter
contract, and discard pre-2026-09-04 tool-failure candidates from the era when
no tool set `isError`.

## Review log

- 2026-09-21 — **one walker.** #1294: the five independent `iterdir()`s over the skill directories (prompt index, MCP discovery, `/api/skills`, the Mission Control tab count, `skill_lint`) collapsed onto `agent_mcp.skills.iter_active_skills`, which owns dot-skip + `_QUARANTINE_STATUSES` + first-root-wins, and `SKILLS_DIRS` now absorbs `config.yaml skills.directories` so the config key steers every surface instead of two. Rewrote *Where skills live*, *The Skills page* and *skill_lint* above; the "three separate definitions" paragraph and the route's "reports the on-disk count (194)" claim are gone with the code they described. Measured over the live vault after the change: prompt index, route, tab count and `skill_lint`'s total all 189, 54 rows carrying an empty `category` rather than vanishing. The route's `metadata.hermes`/`metadata.openclaw` fallback was deleted, not null-guarded, and a live-vault test pins why. Counts and line references had drifted wholesale: 194 skills on disk / 189 advertised, not "191 active", and five live-directory skills do carry `status: archived`, so quarantine is not the unused lever this doc claimed. Refreshed ~25 drifted `file:line` refs (`prompt_builder`, `prefetch`, `agent_mcp/skills`, `autonomy`, `vault_round`, `config.yaml`), corrected the DUPLICATE lint threshold to its two-threshold form, and rewrote the vault-route loader clause — `skill_load_defect` replaced the `_load_skill`-is-None check on 2026-09-19, which is what had made retirement unlandable (#777). Filed #1292 (`/api/skills` ignores quarantine and re-parses front matter itself), #1293 (three SkillsPage POSTs hit unregistered routes), #1294 (five independent walks over the skill dirs, only two apply quarantine); appended a re-verification to #750 (deliverer installed on one route only).
