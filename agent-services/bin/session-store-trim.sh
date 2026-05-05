#!/usr/bin/env bash
# session-store-trim.sh — Keep session stores lean before gateway startup.
#
# Two phases:
#   1. Strip skillsSnapshot / systemPromptReport bloat from session store JSON
#   2. Archive old subagent JSONL files + prune their store entries
#
# Retention policy:
#   - main, discord-lloyd: keep 14 days
#   - all other agents (subagents): keep 3 days
#
# Archived sessions go to agents/_archived_sessions/<agent>/
# Safe to run while gateway is stopped.

set -euo pipefail

AGENTS_DIR="${OPENCLAW_STATE_DIR:-$HOME/.openclaw}/agents"

node -e '
const fs = require("fs");
const path = require("path");

const agentsDir = process.argv[1];
const now = Date.now();
const DAY = 86400000;

// Retention: main/discord-lloyd get 14 days, subagents get 3 days
const RETENTION = { main: 14, "discord-lloyd": 14 };
const DEFAULT_RETENTION_DAYS = 3;

const archiveRoot = path.join(agentsDir, "_archived_sessions");

const dirs = fs.readdirSync(agentsDir, { withFileTypes: true })
  .filter(d => d.isDirectory() && !d.name.startsWith("_"))
  .map(d => d.name);

let totalTrimmed = 0;
let totalArchived = 0;
let totalArchivedBytes = 0;

for (const agent of dirs) {
  const sessDir = path.join(agentsDir, agent, "sessions");
  const storePath = path.join(sessDir, "sessions.json");

  // --- Phase 1: Strip bloat fields from store ---
  let store = {};
  let storeRaw = "";
  try {
    storeRaw = fs.readFileSync(storePath, "utf-8");
    if (storeRaw.length > 10) store = JSON.parse(storeRaw);
  } catch { continue; }

  let storeChanged = false;
  for (const [, entry] of Object.entries(store)) {
    if (entry.skillsSnapshot) { delete entry.skillsSnapshot; storeChanged = true; }
    if ((entry.endedAt || entry.status === "ended" || entry.status === "archived") && entry.systemPromptReport) {
      delete entry.systemPromptReport; storeChanged = true;
    }
  }

  // --- Phase 2: Archive old JSONL files + prune store entries ---
  const retentionDays = RETENTION[agent] ?? DEFAULT_RETENTION_DAYS;
  const cutoffMs = retentionDays * DAY;

  // Build sessionId → store key + timestamp map
  const sessionMeta = new Map();
  for (const [key, entry] of Object.entries(store)) {
    if (entry.sessionId) sessionMeta.set(entry.sessionId, { key, updatedAt: entry.updatedAt || 0 });
  }

  let files = [];
  try { files = fs.readdirSync(sessDir).filter(f => f.endsWith(".jsonl")); } catch {}

  let agentArchived = 0;
  let agentArchivedBytes = 0;

  for (const f of files) {
    const sessionId = f.replace(".jsonl", "");
    const filePath = path.join(sessDir, f);
    let stat;
    try { stat = fs.statSync(filePath); } catch { continue; }

    const meta = sessionMeta.get(sessionId);
    const updatedAt = meta?.updatedAt || stat.mtimeMs;

    if ((now - updatedAt) > cutoffMs) {
      // Archive
      const agentArchiveDir = path.join(archiveRoot, agent);
      fs.mkdirSync(agentArchiveDir, { recursive: true });
      try {
        fs.renameSync(filePath, path.join(agentArchiveDir, f));
        agentArchived++;
        agentArchivedBytes += stat.size;
        // Remove store entry
        if (meta) { delete store[meta.key]; storeChanged = true; }
      } catch {}
    }
  }

  // Write store if changed
  if (storeChanged) {
    const newJson = JSON.stringify(store);
    const trimmed = storeRaw.length - newJson.length;
    if (trimmed > 0) totalTrimmed += trimmed;
    fs.writeFileSync(storePath, newJson);
  }

  if (agentArchived > 0) {
    totalArchived += agentArchived;
    totalArchivedBytes += agentArchivedBytes;
    console.log(agent + ": archived " + agentArchived + " sessions (" + (agentArchivedBytes/1024/1024).toFixed(0) + " MB)");
  }
}

if (totalTrimmed > 0) console.log("Store bloat trimmed: " + (totalTrimmed/1024/1024).toFixed(1) + " MB");
if (totalArchived > 0) console.log("Total archived: " + totalArchived + " files (" + (totalArchivedBytes/1024/1024).toFixed(0) + " MB)");
if (totalTrimmed === 0 && totalArchived === 0) console.log("Nothing to trim or archive.");
' "$AGENTS_DIR"
