---
tags:
  - lloyd
  - architecture
type: reference
segment: projects
---

# Skill System

Skills are on-demand procedures that Lloyd reads before executing. They are simple markdown files, not code packages.

## Skill Locations

| Location | Path |
|----------|------|
| Custom skills | `~/obsidian/skills/` |
| ClawhHub installed | `~/.openclaw/skills/` |
| Built-in | `~/.npm-global/lib/node_modules/openclaw/skills/` |

## Discovery and Loading

- **Discovery:** `skills_search` [[tools|MCP tool]] queries both the local index and ClawhHub catalog in parallel
- Results tagged with `source: "local"` (custom/built-in) vs `source: "clawhub"`
- Local results returned first, ClawhHub appended (deduped by name/slug match)
- 1-hour cache TTL on ClawhHub results; graceful degradation on API errors
- **Source filter:** `source` param accepts `"all"` (default), `"builtin"`, `"custom"`, `"clawhub"`
- **Inspection:** `skills_get` supports `clawhub:<slug>` prefix for read-only inspection of catalog skills
- **Loading:** On-demand — Lloyd reads the `SKILL.md` via `file_read` before executing
- **Not loaded into the system prompt by default** (`maxSkillsInPrompt=0`); skills are loaded from config at `~/obsidian/skills/`

## Current Custom Skills (34)

| Skill | Path |
|-------|------|
| autolink | `skills/autolink/SKILL.md` |
| backlog-as-handoff | `skills/backlog-as-handoff/SKILL.md` |
| browser-audio-pipeline | `skills/browser-audio-pipeline/SKILL.md` |
| claude-code-subagent | `skills/claude-code-subagent/SKILL.md` |
| config-wipe-recovery | `skills/config-wipe-recovery/SKILL.md` |
| correction-handling | `skills/correction-handling/SKILL.md` |
| cron-stuck-recovery | `skills/cron-stuck-recovery/SKILL.md` |
| discord-social | `skills/discord-social/SKILL.md` |
| discord-voice-setup | `skills/discord-voice-setup/SKILL.md` |
| distrobox-service-setup | `skills/distrobox-service-setup/SKILL.md` |
| git-filter-repo | `skills/git-filter-repo/SKILL.md` |
| gpu-device-mismatch | `skills/gpu-device-mismatch/SKILL.md` |
| heartbeat | `skills/heartbeat/SKILL.md` |
| https-migration-gotchas | `skills/https-migration-gotchas/SKILL.md` |
| local-llm-gotchas | `skills/local-llm-gotchas/SKILL.md` |
| mc-ui-change | `skills/mc-ui-change/SKILL.md` |
| nightly-reflection | `skills/nightly-reflection/SKILL.md` |
| nightly-skills-management | `skills/nightly-skills-management/SKILL.md` |
| nightly-vault-maintenance | `skills/nightly-vault-maintenance/SKILL.md` |
| openclaw-plugin-authoring | `skills/openclaw-plugin-authoring/SKILL.md` |
| orchestrator-delegation | `skills/orchestrator-delegation/SKILL.md` |
| periodic-memory-capture | `skills/periodic-memory-capture/SKILL.md` |
| qmd-collection-management | `skills/qmd-collection-management/SKILL.md` |
| research-agent | `skills/research-agent/SKILL.md` |
| restart-openclaw | `skills/restart-openclaw/SKILL.md` |
| secret-migration | `skills/secret-migration/SKILL.md` |
| skill-harvest | `skills/skill-harvest/SKILL.md` |
| skill-tracking | `skills/skill-tracking/SKILL.md` |
| stale-process-cleanup | `skills/stale-process-cleanup/SKILL.md` |
| upstream-bug-triage | `skills/upstream-bug-triage/SKILL.md` |
| voice-clone-sample | `skills/voice-clone-sample/SKILL.md` |
| voice-mode | `skills/voice-mode/SKILL.md` |
| websearch | `skills/websearch/SKILL.md` |
| youtube-transcript | `skills/youtube-transcript/SKILL.md` |

## Skill Format

Each skill is a directory containing a `SKILL.md` file. Some skills include supporting scripts:

```
skills/
  autolink/
    SKILL.md
    autolink.py
  voice-mode/
    SKILL.md
  restart-openclaw/
    SKILL.md
  ...
```

## ClawhHub Integration

ClawhHub is the skill marketplace/catalog for the OpenClaw ecosystem. Integration is baked into the skill primitives — no separate agent needed.

- **Unified search:** `skills_search` queries local index and ClawhHub catalog in parallel
- **Installation:** `skills_install` tool handles download and security validation
- **Install log:** Installed skills tracked at `knowledge/software/clawhub-installed-skills.md`
- **Slug validation:** Regex blocks path traversal attempts

Three attack patterns defended against:
1. Social engineering prerequisites (e.g., "first run this shell command")
2. Prompt injection + malware convergence (malicious instructions embedded in skill text)
3. Remote instruction loading (skills that fetch and execute external URLs)

## Security Validation

All ClawhHub skills pass through a security validation pipeline before installation:

```
ANY E-level finding → REJECT (block, log, never install)
Scan failed/timed out → FLAG for manual review
Has scripts/ directory → FLAG regardless of scan result
Warnings + non-markdown → FLAG
Clean or low-risk warnings + markdown-only + <500 lines → AUTO-INSTALL
```

Security scanning powered by Snyk agent integration.

## Related Docs

- [[index]] — High-Level Architecture
- [[tools]] — MCP Tools Server (skills_search, skills_get, skills_install)
- [[nightly-skills-management]] — Skills Extraction Pipeline
- [[agents]] — Agent System
