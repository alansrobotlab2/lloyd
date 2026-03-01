# HEARTBEAT.md

# Add tasks below. Keep this file small to limit token burn.

## Daily Vault Snapshot

Check `memory/heartbeat-state.json` → `lastVaultSnapshot` timestamp.
If it's been more than 24 hours (or null), run:

```bash
cd /home/alansrobotlab/obsidian && git add -A && git diff --cached --quiet || git commit -m "snapshot"
```

Then update `lastVaultSnapshot` in heartbeat-state.json to the current unix timestamp.

## Open Threads

Ongoing items that need periodic check-in or follow-up. Review these each heartbeat — update status, close resolved items, surface anything overdue to Alan.

| Item | Last Status | Owner |
|------|------------|-------|
| Android app earbud button config | Wonky — Alan checking on it (2026-02-25) | Alan |
| Lloyd wake word training ("Hey Lloyd" / "Lloyd") | Active backlog | Lloyd |
| Inbox pattern for vault | Backlog — async drop zone for links/notes/ideas | Lloyd |
| "How Alan thinks" notes | Backlog — mental models / decision frameworks in vault | Lloyd |
| Per-project MD site log | Backlog — agent-maintained progress log inside each project dir | Lloyd |
| AUDIT.md checklist | Backlog — on-demand workspace health check file | Lloyd |
| Mistral embeddings for memory | Backlog — OpenClaw supports Mistral as cheaper semantic search layer on top of QMD | Lloyd |
| QMD memory settings reevaluation | Backlog — review MMR, temporal decay (halfLifeDays=30), session memory (retentionDays=30) after a few weeks of use. Enabled 2026-02-25. | Lloyd |
| daily-notes QMD collection persistence | Watch — `daily-notes-main` collection was manually added to index.yml (OpenClaw not generating it from paths[] config — possible bug). If daily notes stop surfacing in memory_search results after a gateway restart, re-add the collection to `~/.openclaw/agents/main/qmd/xdg-config/qmd/index.yml` and run `openclaw memory index`. | Lloyd |
| MEMORY.md consolidation routine | ✅ Active — cron job `f53ad621` runs nightly at 2am PST (isolated session, announces summary). Prunes stale entries, distills daily notes → MEMORY.md, keeps under 200 lines. | Lloyd |
