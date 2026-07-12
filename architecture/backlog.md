---
segment: architecture
tags: [architecture,lloyd]
relations:
  related-to:
  - architecture/index.md
  - architecture/infrastructure.md
  - architecture/nightly-reflection.md
  - architecture/nightly-vault-maintenance.md
  - architecture/skills.md
  - architecture/tools.md
  - architecture/voice.md
  - architecture/memory.md
  - architecture/autonomy-system.md
  - architecture/evaluation-engine.md
  - projects/lloyd/plans/voice-async-protocol.md
  - projects/lloyd/plans/document-relations-retrieval.md
  - architecture/agents.md
  - architecture/backlog.md
tags: [architecture]
summary: Markdown-based kanban task management integrated into the MCP Tools Server,
  with 4 tools,QMD search integration,and file-based storage.
type: reference

---















# Backlog System

Markdown-based kanban task management integrated into the [[tools|MCP Tools Server]].

## Storage

- **Format:** Markdown files at `~/obsidian/backlog/{id}-{slug}.md`
- **Frontmatter:** YAML frontmatter for task metadata (id,status,priority,tags,etc.)
- **Body:** Markdown content with title,description,and Activity Log section
- **Search:** QMD collection for hybrid search (BM25 + vector)

## Task States

```mermaid
graph LR
    inbox --> up_next --> in_progress --> in_review --> done
```

## Task Properties

| Property | Type | Description |
|----------|------|-------------|
| id | integer | Unique task ID (in filename and frontmatter) |
| name | string | Task title (frontmatter + first heading) |
| description | string | Task details (markdown body) |
| status | enum | inbox,up_next,in_progress,in_review,done |
| board | string | Board association (frontmatter field) |
| priority | enum | none,low,medium,high |
| assigned | boolean | Whether task is assigned |
| blocked | boolean | Whether task is blocked |
| tags | array | Tags array (frontmatter) |
| activity notes | list | Activity Log entries (append-only section) |
| created | datetime | Creation timestamp (ISO format) |
| updated | datetime | Last update timestamp (ISO format) |
| completed | datetime/null | Completion timestamp (null if not done) |

## Boards

Multiple kanban boards,each with tasks identified by `board` field in frontmatter.

## MCP Tools (4)

| Tool | Description |
|------|-------------|
| `backlog_boards` | List all boards with task counts |
| `backlog_tasks` | List/filter tasks (by status,assigned,blocked,board,tag) |
| `backlog_get_task` | Full task details by ID (frontmatter + body + activity log) |
| `backlog_write_task` | Create or update a task (status,priority,blocked,name,description,tags,activity notes) |

See [[tools]] for full tool definitions and parameters.

## QMD Integration

Backlog tasks are indexed by QMD for search integration:

- **Collection:** `backlog`
- **Path:** `/home/alansrobotlab/obsidian/backlog`
- **Pattern:** `**/*.md`
- **Search:** Tasks are searchable via `mem_search`,`context_bundle`,and other QMD-powered tools

## Activity Log

Activity entries are appended to the `## Activity Log` section at the bottom of each task file:

```markdown
## Activity Log

- **2026-03-29 04:04** — Created (api)
- **2026-03-29 21:30** — Moved to up_next (Lloyd)
- **2026-03-29 22:15** — Updated description (alan)
```

The activity log parser finds the **LAST** `## Activity Log` heading in the file to avoid capturing description content that may contain the same heading as part of a code example.

## Access Rules

- Lloyd queries backlog directly via MCP tools -- never answers backlog questions from vault/memory (notes may be stale)
- Task updates are attributed to the agent in activity history
- All operations are file-based (frontmatter + markdown),no database

## Migration History

### Phase 1-2: SQLite → Markdown (Backlog #220)

Migrated from SQLite database (`~/.openclaw/data/backlog.db`) to markdown files:

- **Before:** SQLite with Python threading lock
- **After:** Markdown files with frontmatter + body structure
- **Tools:** All 4 MCP tools retained with identical interfaces,internals rewritten
- **Search:** QMD collection added for memory integration

### Previous System

The backlog was previously a Rails 8 app (Clawdeck) at `~/Development/clawdeck/`,port 3001,backed by PostgreSQL. It was migrated to the current markdown-based system integrated into the [[tools|MCP tools server]].

## Related Docs

- [[index]] — High-Level Architecture
- [[tools]] — MCP Tools Server (tool implementations)
- [[agents]] — Agent System (task delegation)
- [[infrastructure]] — Infrastructure

