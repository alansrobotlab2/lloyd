# OpenClaw Project

## Runtime Environment

OpenClaw runs inside the **`lloyd` distrobox container**.

To run openclaw commands, enter the container first:
```bash
distrobox enter lloyd
openclaw <command>
```

Or run a single command without entering interactively:
```bash
distrobox-enter lloyd -- openclaw <command>
```

The openclaw binary is at `/home/alansrobotlab/.npm-global/bin/openclaw` inside the container.

## Project Layout

- `openclaw.json` — main config (models, plugins, memory backend, gateway)
- `extensions/` — local plugins (TypeScript, loaded at startup)
- `agents/main/sessions/` — session JSONL transcripts
- `logs/timing.jsonl` — per-run timing metrics from the timing-profiler plugin
- `workspace/` — persistent workspace files (MEMORY.md, skills/, etc.)
- `tools/` — standalone utility scripts (e.g. timing-profile.js)

## Plugins

Active plugins (see `plugins.allow` in openclaw.json):
- `memory-core` — built-in QMD memory backend (Obsidian vault at `/home/alansrobotlab/obsidian`)
- `web-local` — local web search + fetch (extensions/web-local/)
- `timing-profiler` — per-run latency logging (extensions/timing-profiler/)
- `memory-prefetch` — pre-fetches memory before first LLM pass (extensions/memory-prefetch/)
