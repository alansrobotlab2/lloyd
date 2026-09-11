---
segment: architecture
tags: [architecture, lloyd, index]
type: reference
status: implemented
date: 2026-09-11
---

# Lloyd — architecture index

Lloyd is a fully local AI agent: an in-process agent loop ([[harness]])
against a local vLLM engine ([[engines]]), every tool behind one MCP
aggregator ([[tools]]), a FastAPI + SSE backend with a React front end
([[mission-control]]), an Obsidian vault for memory and skills, and a worker
pool that does everything unasked ([[workers]]) — including rewriting its own
code behind a gate and a rollback watchdog ([[automod]]).

One doc per area. `status: implemented` docs describe what runs; a doc that
turns out to describe something gone or never built is moved to
`architecture/.archive/` (gitignored) rather than left to mislead.

## Runtime

| Doc | Covers |
|---|---|
| [[infrastructure]] | host, GPUs, supervisord programs, systemd units, ports, remote access |
| [[engines]] | the primary and secondary model slots, identity checks, what keeps the primary fast |
| [[vllm-throughput-mitigation]] | the 09-09 stall: cold re-prefills, FP8 KV, prefix-miss accounting, the KV gate |
| [[gpu-power-limit-persist]] | the GPU power clamp unit |

## The agent

| Doc | Covers |
|---|---|
| [[harness]] | `run_query`: events, the position-0 rule, preserved thinking, tool pool, summaries, thinking trace, finalizer, subagents |
| [[tools]] | the lloyd-mcp aggregator, the module and tool inventory, annotations |
| [[editing-safeguards]] | read-before-edit gates, edit diagnostics and blast radius, the change and effect ledgers, the code graph |
| [[inner-voice]] | the observer that watches the primary stream and steers it with five soft levers |
| [[subliminal]] | pre-call context retrieval: skills, facts, vault docs, sessions, backlog |
| [[ambient-context-injection]] | how background producers surface context into the active chat |
| [[skills]] | on-demand SKILL.md procedures from the vault |
| [[voice]] | wake word → STT → the cloned voice; LiveKit worker, client-side TTS shaping |

## Memory and knowledge

| Doc | Covers |
|---|---|
| [[memory]] | the four-verb memory surface and the fact-improvement loop |
| [[knowledge-graph]] | markdown facts plus the one SQLite edge/alias/entity store behind `app.kg_store` |
| [[groundskeeper]] | vault health scanning and enrichment |
| [[research-pipeline]] | the topic registry (`research.db`) fed nightly and drained by `deep-research` |

## Unattended work

| Doc | Covers |
|---|---|
| [[workers]] | one SQLite queue, N asyncio workers in the backend, every source |
| [[background-runs]] | every background run is recorded; Inner Voice observation is opt-in; the Background tab |
| [[autonomy-system]] | the scheduled task fleet in `~/obsidian/autonomy/` (predates the workers queue; being brought current) |
| [[automod]] | self-modification: worktree, nine-rung gate, review rung, promoter, guardian rollback, triage/implement, clustering, group triage |
| [[backlog]] | the markdown kanban at `~/obsidian/backlog/` and its tools |
| [[nightly-reflection]] | the nightly chain, rewritten from audited reality |
| [[nightly-skills-management]] | the 3 am skill mining / eval / dedup job |
| [[morning-briefing]] | the 7 am overnight synthesis |

## Front end

| Doc | Covers |
|---|---|
| [[mission-control]] | the sixteen tabs, the dashboard endpoint, sessions and titles, the agent's view of the UI, remote access |

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
