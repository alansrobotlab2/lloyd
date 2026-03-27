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

## Restarting the Gateway

The gateway is managed by **supervisord** inside the `lloyd` distrobox container.

- Config: `/home/alansrobotlab/agent-services/supervisor/supervisord.conf`
- Program config: `/home/alansrobotlab/agent-services/supervisor/conf.d/openclaw-gateway.conf`
- Logs: `/home/alansrobotlab/agent-services/logs/openclaw-gateway.log` (stderr: `.err`)

```bash
# Check status
distrobox-enter lloyd -- supervisorctl -c /home/alansrobotlab/agent-services/supervisor/supervisord.conf status openclaw-gateway

# Restart (kill stale port holder first if needed)
distrobox-enter lloyd -- bash -c "kill \$(lsof -ti :18789) 2>/dev/null; sleep 1"
distrobox-enter lloyd -- supervisorctl -c /home/alansrobotlab/agent-services/supervisor/supervisord.conf restart openclaw-gateway

# Watch logs
distrobox-enter lloyd -- tail -f /home/alansrobotlab/agent-services/logs/openclaw-gateway.log
```

**WARNING:** Do NOT run `openclaw doctor --fix` — it installs a systemd user service that conflicts with the supervisord setup, causing crash loops on port 18789.

## Project Layout

- `openclaw.json` — main config (models, plugins, memory backend, gateway)
- `extensions/` — local plugins (TypeScript, loaded at startup)
- `docs/` — analysis documents, evaluations, and design notes
- `agents/main/sessions/` — session JSONL transcripts
- `logs/timing.jsonl` — per-run timing metrics from the timing-profiler plugin
- `workspace/` — persistent workspace files (MEMORY.md, skills/, etc.)
- `tools/` — standalone utility scripts (e.g. timing-profile.js)
- `WORKLOG.md` — running log of analysis and changes (see below)

## Plugins

Active plugins (see `plugins.allow` in openclaw.json):
- `memory-core` — built-in QMD memory backend (Obsidian vault at `/home/alansrobotlab/obsidian`)
- `memory-graph` — tag-based vault index with IDF ranking (extensions/memory-graph/)
- `web-local` — local web search + fetch (extensions/web-local/)
- `timing-profiler` — per-run latency logging (extensions/timing-profiler/)
- `memory-prefetch` — pre-fetches memory before first LLM pass (extensions/memory-prefetch/)

## Memory & Vault Tools

All memory tools operate on the same Obsidian vault at `~/obsidian` via the QMD backend. Paths are relative to the vault root (e.g. `projects/alfie/alfie.md`).

- `mem_search` — QMD semantic/hybrid vector search across `~/obsidian/**/*.md`
- `mem_get` — QMD file reader by relative path (with optional line range)
- `tag_search` — frontmatter tag index lookup (structured, fast); returns title, summary, all tags
- `tag_explore` — tag co-occurrence discovery and bridging documents
- `vault_overview` — vault statistics (doc/tag counts, type distribution, hub pages)

All return the same relative paths. `tag_search` results are retrievable via `mem_get`.

## Work Log

Always append a summary entry to `WORKLOG.md` at the end of any session that involves analysis or code changes. Each entry should include:
- Date and short title
- Type (Analysis, Feature, Bugfix, Refactor, etc.)
- Files examined or modified
- Link to any docs produced (in `docs/`)
- Brief summary of findings or changes (3-5 bullet points max)
