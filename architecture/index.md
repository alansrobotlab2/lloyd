---
segment: architecture
tags: [architecture, lloyd, index]
type: reference
status: implemented
date: 2026-09-14
---

# Lloyd — architecture index

Lloyd is a fully local AI agent: an in-process agent loop ([[harness]])
against a local vLLM engine ([[vllm]]), every tool behind one MCP
aggregator ([[tools]]), a FastAPI + SSE backend with a React front end
([[mission-control]]), an Obsidian vault for memory and skills, and a worker
pool that does everything unasked ([[workers]]) — including rewriting its own
code behind a gate and a rollback watchdog ([[automod]]).

One doc per area. `status: implemented` docs describe what runs; a doc that
turns out to describe something gone or never built is moved to
`architecture/.archive/` (gitignored) rather than left to mislead. Every
tracked doc here carries `status: implemented` — the only accepted value — and
at least one date key, `date`, `updated` or `timestamp` (whichever the doc
already uses; arch-review refreshes that one).
`tests/test_architecture_doc_frontmatter.py` walks the tracked docs for both.

## Runtime

| Doc | Covers |
|---|---|
| [[infrastructure]] | host, GPUs, supervisord programs, systemd units, ports, remote access |
| [[vllm]] | the primary engine end to end: the served config, the GPU and its power clamp, FP8 KV, YaRN (built, off), the tuning knobs, the benchmarks |
| [[djev]] | GPU 2's structured-decision engine: serving, the wire protocol, the client and its two trust flags, three tools, the schema registry, three shadow seams, the eval arm, the calibration that says rank with it and do not gate on it |

## The agent

| Doc | Covers |
|---|---|
| [[harness]] | `run_query`: events, the position-0 rule, preserved thinking, tool pool, summaries, thinking trace, finalizer, subagents |
| [[tools]] | the lloyd-mcp aggregator: routes and credential, the dispatch path in order, tool properties and what each decides, every tool with its properties |
| [[desktop]] | desktop computer use: `desktop_capture`/`desktop_act` over Alan's real Hyprland desktop, the lease only a human grants, chat sessions only, screenshots as refs never base64 |
| [[editing-safeguards]] | read-before-edit gates, edit diagnostics and blast radius, the change and effect ledgers, the code graph |
| [[data-home]] | runtime data outside the code tree since the 09-22 deletion: `~/lloyd-data`, the `DATA_ROOT` resolver and who gets which root, the delete guard, hourly read-only snapshots, the data tripwire, restore |
| [[vault-protection]] | the 09-10/09-12 vault wipes and the four layers after them: the bench/eval tool sandbox, the wholesale-delete check, the guardian's vault tripwire and sync gate, 15-minute snapshots |
| [[inner-voice]] | deterministic turn guards on every turn, plus an opt-in terminal review and the loop that measures it |
| [[subliminal]] | pre-call context retrieval: skills, facts, vault docs, sessions, backlog |
| [[qmd]] | the vault search engine: Lloyd's fork, the daemon, index, config and models, its REST API, maintenance, eval pins |
| [[retrieval]] | vault recall's document leg: qmd fusion (OR keyword leg, Qwen3 vectors), djev ranking, the gold set, every measured change |
| [[recall-research-2026-09-24]] | the whole recall stack reviewed against the frontier on 2026-09-24: the map as measured, what the literature says, the proposals filed from it, what was not proposed and why |
| [[ambient-context-injection]] | how background producers surface context into the active chat |
| [[skills]] | on-demand SKILL.md procedures from the vault |
| [[voice]] | the whole voice-to-voice round trip: LiveKit transport, wake word, ASR, speaker id, the cloned TTS voice and its client-side shaping |
| [[desktop]] | desktop computer use: capture freely, act only on a lease a human grants from the Desktop tab; chat sessions only; screenshots as refs, never base64 |

## Memory and knowledge

| Doc | Covers |
|---|---|
| [[memory]] | the four-verb memory surface and the fact-improvement loop |
| [[knowledge-graph]] | markdown facts plus the one SQLite edge/alias/entity store behind `app.kg_store` |
| [[research-pipeline]] | the topic registry (`research.db`): its states, its two producers, and why retries live there rather than in the work queue |

## Unattended work

| Doc | Covers |
|---|---|
| [[workers]] | one SQLite queue, N asyncio workers in the backend, every source |
| [[workers-jobs]] | one entry per worker source: what wakes it, what it writes, and what it is actually doing |
| [[background-runs]] | every background run is recorded; Inner Voice observation is opt-in; the Background tab |
| [[autonomy]] | the scheduled-task mechanism: the five due-gates, failure backoff, the deadline anchor, run records, fleet health |
| [[autonomy-jobs]] | what each scheduled job is *for* — the reflection chain, trace2skill, the graph chain, vault hygiene, inbound signal |
| [[automod]] | self-modification: worktree, the gate ladder, review rung, promoter, guardian rollback, triage/implement, clustering, group triage |
| [[testing]] | the ~8,700-test suite: synthetic vs live-data tests and the rule for the second, when a live check is retired rather than skipped, the gate's three floors and why the skip cap is not the thing to raise, marks the gate deselects, parallel isolation |
| [[backlog]] | the markdown kanban at `~/obsidian/backlog/` and its tools |
| [[arch-review]] | the pass that keeps these docs honest: a picklist of docs and functional groups, one session each, the doc edits itself |

## Front end

| Doc | Covers |
|---|---|
| [[mission-control]] | the tabs and the four lists that must agree on them, the dashboard endpoint, sessions and titles, the agent's view of the UI, the browser SSRF guard |

## GPU allocation

| GPU | Hardware | Serves |
|---|---|---|
| 0 | RTX 3090 24 GB | TTS, voice worker, qmd |
| 1 | RTX PRO 6000 96 GB | primary LLM (vLLM) |
| 2 | RTX 3090 24 GB | secondary LLM (llama.cpp) |

## Retired (in `.archive/`, not tracked)

`agents`, `background-monitoring`, `evaluation-engine`, `exploration-engine`,
`harness-comparison`, `improvement-planner`, `intelligence-pipeline`,
`nightly-vault-maintenance`, `staged-pipeline`, `usage-tracking`,
`verification-system` — the OpenClaw-era agent roster and gateway, a daemon
that no longer exists, and five 2026-03 plans (#177–#183) whose ideas landed
later as [[automod]], [[workers]] and the research pipeline.

Three retirements on 2026-09-11 were folds rather than obsolescence, and the
distinction is worth keeping: nothing in any of them was wrong, it was in the
wrong place.

`gpu-power-limit-persist` went into [[vllm]] §2 — the power clamp only ever
mattered because of what runs on that card, and keeping the two apart meant
reading both to understand either.

`engines` went into [[infrastructure]] § Model slots and [[harness]] § Which
engine a turn reaches. It had become mostly a summary of its neighbours: the
GPU and program tables were already in [[infrastructure]], every "what keeps
the primary fast" bullet was already in [[vllm]], and `startsecs=900` was in
all three. What was genuinely its own — the slot-is-an-endpoint distinction
and the identity checks — belongs beside the programs, and the
model-resolution rules belong beside the loop that resolves them.

`groundskeeper`, `morning-briefing`, `nightly-reflection` and
`nightly-skills-management` went into [[autonomy-jobs]]. They were four docs
about a handful of the fleet's scheduled jobs, which left most of it with no
description anywhere — over-documenting a quarter of the fleet and ignoring the
rest. One doc per *job family* replaces them, and [[autonomy]] keeps the
mechanism.

## Review log

- **2026-09-14 — `stale`.** The counts had rotted in the three days since the
  curation: the fleet is 33 tasks, not 32, and the gate ladder is ten rungs, not
  nine (`prompt_surface` joined on 2026-09-12). Both numbers now come from their
  owners — the Covers rows say "each scheduled job" and "the gate ladder" and
  name no integer, because a hand-typed count here is a third definition of a
  set `GET /api/autonomy/tasks` and `gate.py`'s ladder list already own. The
  22-document list, the `.archive/` claims (gitignored, untracked, 11 retired +
  6 absorbed) and the GPU allocation table all still check out; the two rosters
  this page advertises are short by one member each — filed as #919 (rung
  ladder, in `automod.md`), #1015 (`board-steward`, in `workers-jobs.md`) and
  #1102 (#86, in `autonomy-jobs.md`). Nothing checks this page's own listing
  against the directory it describes, and six sibling docs carry no `status:`
  the convention recognises — #1103, #1104.
