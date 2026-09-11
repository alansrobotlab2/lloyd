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
the phantom-tool ban list (`scripts/skill_lint.py:214`,
`tests/test_skill_tool_names.py:38`) — this file was itself an instance of the
defect described under "Phantom tool names" below. Its skill inventory listed
34 skills, of which 8 have since been archived and 3 no longer exist at all.
The history is kept here rather than deleted because the OpenClaw vocabulary
still turns up in the vault, in old skills and in `nightly-skills-management`.

## Where skills live

| Location | What it is |
|----------|------------|
| `~/obsidian/skills/` | the library — 191 active skills |
| `~/obsidian/skills/.archived/` | 192 retired skills, kept on disk and in git |
| `~/lloyd/skills/` | second root, configured everywhere, **does not exist** |

**Three separate definitions of "where skills live", and they do not agree.**
`agent_mcp/skills.py:27` hardcodes `SKILLS_DIRS`; `prompt_builder.py:100`
hardcodes `_CANON_SKILLS_DIRS`, anchored to the *repo* location rather than
`Path.home()` so it resolves the same whoever runs the process; and
`config.yaml:685` `skills.directories` is read by only three call sites
(`app/routers/skills.py:19`, `:60` and `app/routers/mc_ui.py:457`) — the two
HTTP routes and the Mission Control tab summary. Nothing the *model* touches
reads the config key. All three currently list the same two paths, so the
divergence is latent: editing `skills.directories` moves the Skills page and
the tab summary and changes nothing about what reaches a turn.

`~/lloyd/skills/` has never existed on this box. It is harmless — every reader
skips a missing directory — but it means the second root is not a tested path.

**Retirement is by directory, not by frontmatter.** Both loaders skip
dot-prefixed directories (`prompt_builder.py:559`, `agent_mcp/skills.py:90`),
so moving a skill into `.archived/` removes it from every surface at once
while keeping it on disk and in git history. The `status:` quarantine below is
the other lever, and in practice it is the unused one: all 191 live skills read
`status: active`, and the archive is where the mixed values sit.

## The four surfaces that read a skill

A skill can reach a turn four ways, and they are deliberately different
mechanisms rather than one with options.

### 1. The index, in the system prompt

`prompt_builder._load_skills_index` (`prompt_builder.py:546`) walks both roots,
dedupes by directory name (first root wins) and emits one line —
`Available skills: ai-engineer-monitor, alfie-monitoring, …` — wrapped by
`build_system_prompt` (`:314`) as:

```
<available_skills>
Available skills: …
</available_skills>
Note: relevant skill content is automatically injected into each
user message as <context> when matched.
```

It is names only: 191 of them, 4,066 chars, ~1k tokens. That is measured, not
estimated — the skills index is one of the named components in the
`PROMPT_BUDGET` line `log_prompt_size` writes once per build (`:73`), against
an 80,000-char tripwire. The note at the end exists because the index alone
would read as "call something to load these"; what actually happens is surface
2.

`include_skills_index` defaults to `True` and every production caller takes the
default (`app/routers/messages.py:1736`, `:1908`, `:2032`,
`app/routers/voice.py:104`, `autonomy.py:943`). The `False` branch is for tests
that assert on the rest of the prompt.

**A quarantined skill is excluded from the index, and that is not cosmetic.**
`_is_quarantined_skill` (`:523`) reads the frontmatter `status:` and drops
anything in `_QUARANTINE_STATUSES` — `inactive`, `archived`, `disabled`,
`retired`, `quarantined`. The vocabulary is **imported from
`agent_mcp.skills`** (`:109`) rather than restated, with a hardcoded fallback
only for the case where MCP cannot be imported at all, because advertising a
skill the reader then declines is the same failure as naming a tool that does
not exist: the prompt promises something that does not work.
`tests/test_prompt_builder_overlay.py:193` pins the two lists together.

The parse is deliberately cheap — a 2,000-char head read and a line scan, not
YAML — because this runs for every skill on every prompt build.

### 2. The turn-start injector (prefetch)

`prefetch.py` scores the *user's message* against every skill and injects the
winner's body into a `<context>` block ahead of the turn. Thresholds
(`prefetch.py:40-43`):

| | value |
|---|---|
| `SKILL_THRESHOLD_FIRST` | 3.0 — inject full body |
| `SKILL_THRESHOLD_SECOND` | 4.0 — inject second skill as excerpt |
| `SKILL_BODY_MAX` | 6000 chars |
| `SKILL_EXCERPT_MAX` | 500 chars |

Rendered as `<skill name="…" score="…">` and, for the runner-up,
`<skill name="…" score="…" excerpt="true">` (`:903`, `:911`). When nothing
matches *and* the message is a genuine new task rather than a continuation, a
`<skill-hint>` nudges toward `skills_search` (`:961`) — the low-confidence
branch was dropped as pure noise, because the model trusts a low-scoring
auto-pick ~95% of the time regardless.

Scoring lives in `agent_mcp/skills._score_skill` (`:203`), shared with the MCP
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
cached skill dict (`_skill_token_sets`, `:175`) — re-tokenizing ~1.5 MB of
bodies cost ~83 ms of GIL-held CPU on *every* turn and starved the other
prefetch legs; it is ~1 ms now. And the skill list is rebuilt only when a
`SKILL.md` mtime changes, checked at most every 15 s (`_skills_signature`,
`:396`).

`_stem` (`:105`) is a single-suffix plural collapse, not a real stemmer, and
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
`tool_errors` (`autonomy.py:912`, `:927`) — the very number this feature exists
to improve, so a deny would make the fleet look sicker exactly where it is
being taught something. `HookRegistry.fire_pre_tool_use` recognises
`skillDeliver` alongside `deny` (`app/harness/hooks.py:148`) and `loop.py:1543`
renders it, in the same shape as the synthetic `ToolSearch` result above it.

**A deny beats a deliver regardless of registration order.** The walk holds a
deliver as provisional and keeps going (`hooks.py:132`, `:146-150`), so a catastrophic
`Bash` is blocked rather than answered with a protocol card by a deliverer that
happened to register first. `install_skill_dispatch_hook` is still called after
`install_default_safety_hook` (`app/routers/messages.py:1808`) — not because
order decides the outcome, but so the walk reads in the order that matters.

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
`skill_path`. `autonomy._load_skill_content` (`autonomy.py:545`) resolves a
value containing `/` or ending `.md` as a path, and anything else as
`~/obsidian/skills/<slug>/SKILL.md`; the body is pasted into the task prompt
under "Follow the skill instructions below" (`:577`).

A task with neither is unrunnable, and says so once per task id rather than
silently dead-lettering (`:428-439`). `hold_reason` reports it as `"no skill"`.
Both fields are in `_parse_task_file`'s `fallback_fields` (`:124`), so a task
file that needed the degraded parser does not lose its skill binding.

Note this path reads the file directly — it does **not** go through
`agent_mcp.skills._load_skill`, so it is the one surface a `status:` quarantine
does not reach.

## The MCP tools

Two, both read-only (`agent_mcp/annotations.py:49`), both served by the
`skills` module inside the lloyd-mcp aggregator (`agent_mcp/main.py:82`,
`:146`):

- **`skills_search(query, max_results=10)`** — ranked over name, description,
  tags and body, using the same `_score_skill` prefetch uses.
- **`skills_read(name)`** — the full raw `SKILL.md` for a directory name.

Both are in the ToolSearch baseline (`config.yaml:404-405`), so they are
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
| `category` | scorer (not weighted), `/api/skills` | on 136 of 191 |
| `status` | both loaders | quarantine; all 191 live read `active` |

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
`~/obsidian/autonomy/skill-lint-report.md`. Six categories: DEAD (unparseable
frontmatter, or description *and* tags both empty — the live scorer would
return 0 for any query), MISSING_DESC, DRIFT (description is output-framed
rather than trigger-framed), DUPLICATE (difflib ratio ≥ 0.85), STALE (mtime
> 90 days and `status != active`), PHANTOM_TOOL.

Advisory only — no automatic deletion or rewrites, and it exits 0 always so the
nightly pipeline does not fail on lint findings. Autonomy task #70 runs it
weekly.

## The automod vault route validates the loaders

The vault is a live shared tree with no worktree, and `prompt_builder` reads
SOUL.md and the skills straight from it, so a self-modification round touching
`skills/**` cannot be gated in isolation — an edit is live the moment it is
saved. `scripts/automod/vault_round.py` therefore validates, commits exactly
the named paths, and reverts on failure.

`skills/**` is one of three `VALIDATED_GLOBS` (`:57`, with `lloyd/**` and
`autonomy/**`) — the paths that feed a prompt or the scheduler, where the
loaders are actually run against the change:

1. Every changed `.md` opening with `---` must parse to a mapping. A skill
   whose front matter broke is a skill nobody can find.
2. The system prompt must still build (`build_system_prompt()` returning ≥ 500
   chars), and **each touched skill must still load through
   `agent_mcp.skills._load_skill`** (`:119-123`).

Scoped to the diff on purpose: a pre-existing broken file elsewhere in the
vault must not block every round — the same delta principle as pyflakes and tsc
in the code gate. A loader failure reverts the round's paths; success commits
exactly those paths on `main` and records a `vault_land` ledger event.

Note the interaction with quarantine: `_load_skill` returns `None` for a
quarantined skill, so setting `status: retired` on a skill *in the same round
that edits it* fails validation as "does not load". Archive by moving to
`.archived/` instead.

## The Skills page

`web/src/components/pages/SkillsPage.tsx`, backed by two GET routes in
`app/routers/skills.py`: `/api/skills` (list, with a `metadata.hermes` /
`metadata.openclaw` fallback for description and category — the last live
remnant of the old vocabulary) and `/api/skill-content?name=` (raw text).

**Three of the page's five API calls hit routes that do not exist.**
`api.skillToggle` → `POST /api/skill-toggle`, `api.skillContentSave` →
`POST /api/skill-content`, and `api.skillsRefresh` → `POST /api/skills/refresh`
are all defined in `web/src/api.ts:1074-1095` and called from the page
(`SkillsPage.tsx:96`, `:109`, `:120`), but only the two GETs are registered
(`server.py:124` includes a router carrying `@router.get` twice and nothing
else). The toggle, the in-place editor's save, and the refresh button are
therefore dead: `enabled: True` and `configured: True` are hardcoded in the
list response, so there is no toggle state to write anyway. Recorded, not
fixed — the page is read-useful as it stands, and a skill is edited in the
vault.

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
