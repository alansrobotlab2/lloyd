# Backlog Migration: Replace ClawDeck with Built-in Backlog System

**Date:** 2026-03-03
**Status:** Planning

## Why

ClawDeck is a Rails 8.1 app + PostgreSQL running as a separate systemd service on :3001, providing kanban task management for agent workflows. Since mission-control already has its own full kanban UI and all interaction happens through REST API calls, ClawDeck is an overweight dependency — a full Rails stack + database server just to store ~43 tasks and 3 boards.

**Removing ClawDeck eliminates:**
- Ruby on Rails runtime dependency
- PostgreSQL database server
- `lloyd-clawdeck.service` systemd unit
- The `extensions/clawdeck/` OpenClaw plugin
- Port 3001 allocation

**Replacing it with:**
- SQLite data layer in `tool_services.py` (Python, existing MCP server on :8093)
- Agent tools registered via MCP (not a separate plugin)
- Dashboard HTTP routes in `mission-control/index.ts` using `node:sqlite`
- Everything renamed from "clawdeck" to "backlog"

## Architecture

```
Mission Control UI ("Backlog" tab)
    |
    v
/api/mc/backlog/* routes (mission-control/index.ts, node:sqlite)
    |
    v
~/.openclaw/data/backlog.db  (SQLite, WAL mode)
    ^
    |
backlog_* MCP tools (tool_services.py, python sqlite3)
    ^
    |
Agent tool calls via MCP proxy (mcp-tools/index.ts)
```

Both Python and Node access the same SQLite file. WAL mode enables safe concurrent readers. Write volume is trivial.

---

## Phase 1: Build the Backend

### 1.1 SQLite data layer in `tool_services.py`

**File:** `~/Projects/lloyd-services/tool_services.py`

Constants:
```python
BACKLOG_DB = Path.home() / ".openclaw" / "data" / "backlog.db"
_backlog_lock = threading.Lock()
```

Schema (3 tables, initialized in `_lifespan`):
```sql
CREATE TABLE IF NOT EXISTS boards (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL, icon TEXT DEFAULT '📋', color TEXT DEFAULT 'gray',
  position INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL, description TEXT DEFAULT '',
  priority TEXT DEFAULT 'none',
  status TEXT DEFAULT 'inbox',
  blocked INTEGER DEFAULT 0, completed INTEGER DEFAULT 0,
  completed_at TEXT, due_date TEXT, position INTEGER,
  assigned_to_agent INTEGER DEFAULT 0,
  assigned_at TEXT, agent_claimed_at TEXT,
  board_id INTEGER NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
  tags TEXT DEFAULT '[]',
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS task_activities (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  action TEXT NOT NULL, field_name TEXT, old_value TEXT, new_value TEXT,
  source TEXT DEFAULT 'api', actor_type TEXT, actor_name TEXT, note TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
```

6 MCP tools following existing `@mcp.tool()` pattern with `_backlog_lock`:

| Tool | Purpose |
|------|---------|
| `backlog_boards` | List all boards with task counts |
| `backlog_tasks` | List tasks with filters (status, assigned, blocked, board_id, tag) |
| `backlog_next_task` | Get highest-priority assigned up_next task |
| `backlog_get_task` | Get full task details by ID |
| `backlog_update_task` | Update status/blocked/priority + activity note |
| `backlog_create_task` | Create new task on a board |

### 1.2 Register proxy tools in `mcp-tools/index.ts`

**File:** `~/.openclaw/extensions/mcp-tools/index.ts`

Add 6 `proxyTool()` calls mirroring the new tools. Timeout: `5_000ms`.

Add system event notifications: when `backlog_create_task` or `backlog_update_task` completes, fire `api.runtime.system.enqueueSystemEvent()` with debounce (5s per task ID).

### 1.3 Dashboard routes in `mission-control/index.ts`

**File:** `~/.openclaw/extensions/mission-control/index.ts`

**Delete** (~170 lines): `loadClawDeckConfig()`, `clawdeckProxy()`, all 5 `/api/mc/clawdeck/*` routes.

**Add** (~120 lines): 5 new `/api/mc/backlog/*` routes using `node:sqlite` `DatabaseSync`:
- `GET /api/mc/backlog/boards`
- `GET /api/mc/backlog/tasks`
- `POST /api/mc/backlog/task-update`
- `POST /api/mc/backlog/task-delete`
- `POST /api/mc/backlog/task-create`

DB opened once at plugin init with `PRAGMA journal_mode=WAL`.

**Update** tool group source: merge backlog tools into `mcp-tools` source group.
**Remove** clawdeck from `MANAGED_SERVICES`.

---

## Phase 2: Rename Frontend

### 2.1 Component rename

| File | Change |
|------|--------|
| `ClawDeckPage.tsx` → `BacklogPage.tsx` | Rename file, component, header text, API calls |
| `Sidebar.tsx` | `"clawdeck"` → `"backlog"`, label `"ClawDeck"` → `"Backlog"` |
| `Layout.tsx` | Update import + page map key |
| `ToolsPage.tsx` | Remove `"clawdeck"` from SOURCE_ICONS/SOURCE_COLORS |

### 2.2 API client rename

**File:** `api.ts`
- Interfaces: `ClawDeckBoard` → `BacklogBoard`, `ClawDeckTask` → `BacklogTask`
- Methods: `clawdeckBoards` → `backlogBoards`, etc.
- URL paths: `"/clawdeck/"` → `"/backlog/"`
- Delete `api.ts.bak`

---

## Phase 3: Config Updates

### 3.1 `openclaw.json`

Rename all tool references across agent configs:
- `clawdeck_boards` → `backlog_boards`
- `clawdeck_tasks` → `backlog_tasks`
- `clawdeck_next_task` → `backlog_next_task`
- `clawdeck_get_task` → `backlog_get_task`
- `clawdeck_update_task` → `backlog_update_task`
- `clawdeck_create_task` → `backlog_create_task`

Agents affected: main (lines 142-145), coder (lines 225-228), operator (lines 276-281).

Rename `"defaultSessionKey": "hook:clawdeck"` → `"hook:backlog"` (line 414).

Remove `clawdeck` from `plugins.allow`.

---

## Phase 4: Data Migration

One-time migration while Rails is still running:
1. `GET http://localhost:3001/api/v1/boards` → insert into SQLite boards table
2. `GET http://localhost:3001/api/v1/tasks` → insert into SQLite tasks table
3. Verify data integrity: count boards, count tasks, spot-check a few records
4. Confirm mission-control Backlog tab renders correctly from SQLite

---

## Phase 5: Kill ClawDeck & Full Reference Sweep

Once migration is verified, remove ClawDeck entirely and update all references.

### 5.1 Delete ClawDeck integration

- Delete `extensions/clawdeck/` entirely (index.ts, config.json, openclaw.plugin.json)
- `systemctl --user disable --now lloyd-clawdeck.service`
- Confirm port 3001 is free

### 5.2 Rebuild frontend

```bash
cd ~/.openclaw/extensions/mission-control/web && npm run build
```

### 5.3 Update Obsidian vault references (18 files)

**Active agent files:**
- `agents/lloyd/AGENTS.md` — update tool names + descriptions
- `agents/operator/AGENTS.md` — update tool names
- `agents/operator/MEMORY.md` — update tool references
- `agents/auditor/MEMORY.md` — update references
- `agents/coder/AGENTS.md` — update tool names
- `agents/orchestrator/AGENTS.md` — update tool names/descriptions
- `agents/lloyd/HEARTBEAT.md` — update any references
- `agents/lloyd/skills/research-agent/SKILL.md` — update tool list
- `agents/lloyd/knowledge/workflows/lloyd-delegation-pattern.md` — update
- `agents/lloyd/knowledge/workflows/tool-restriction-rollback.md` — update
- `agents/lloyd/memory/personal/2026-03-03.md` — update
- `agents/lloyd/memory/personal/2026-02-27.md` — update

**Archive files (update for consistency):**
- `agents/lloyd-legacy/operator/AGENTS.md`
- `agents/lloyd-archive/skills/research-agent/SKILL.md`
- `agents/lloyd-archive/knowledge/software/clawdeck.md` — rename/note as deprecated
- `agents/lloyd-archive/knowledge/software/openclaw-mission-control.md`
- `agents/lloyd-archive/knowledge/ai/openclaw-mission-control-dashboards.md`
- `agents/lloyd-archive/infrastructure/lloyd-services.md`

### 5.4 Update OpenClaw docs (10 files)

- `docs/memory-segmentation.md`
- `docs/agent-framework-gameplan.md`
- `docs/context-optimization-gameplan.md`
- `docs/system-prompt-full.md`
- `docs/system-prompt-breakdown.md`
- `docs/mcp-tool-migration.md`
- `docs/TOOL-ARCHITECTURE.md`
- `docs/SUBAGENTS-IMPLEMENTATION.md`
- `docs/services.md`
- `WORKLOG.md` — add migration entry

### 5.5 Update Claude Code memory

- `~/.claude/projects/-home-alansrobotlab--openclaw/memory/MEMORY.md`
  - Remove clawdeck plugin references
  - Add backlog system documentation
  - Update tool list under MCP-owned section

### 5.6 Update `tool_services.py` header

Change tool count from 19 → 25 and add backlog tools to the listing comment.

---

## Files Summary

| File | Action |
|------|--------|
| `~/Projects/lloyd-services/tool_services.py` | Add ~200 lines (schema + 6 tools) |
| `~/.openclaw/extensions/mcp-tools/index.ts` | Add ~60 lines (6 proxyTool + notifications) |
| `~/.openclaw/extensions/mission-control/index.ts` | Delete ~170, add ~120 lines |
| `~/.openclaw/extensions/mission-control/web/src/` | Rename + update 5 frontend files |
| `~/.openclaw/openclaw.json` | Rename tools, remove clawdeck plugin |
| `~/.openclaw/extensions/clawdeck/` | Delete entirely |
| `~/obsidian/agents/` | Update 18 vault files |
| `~/.openclaw/docs/` | Update 10 docs |
| `~/.claude/.../memory/MEMORY.md` | Update memory |

## Verification

1. Gateway starts clean — no clawdeck plugin errors, backlog tools register via MCP
2. Mission-control "Backlog" tab renders boards + tasks from SQLite
3. CRUD from UI — create, update, delete tasks persist across refresh
4. Agent tools — `backlog_tasks`, `backlog_create_task` work from agent session
5. System events — task create triggers notification to main agent
6. Port 3001 free, no Rails process
7. Services page — ClawDeck no longer listed
8. `grep -ri clawdeck ~/obsidian/ ~/.openclaw/` — zero hits outside archive/legacy
