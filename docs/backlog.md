---
tags:
  - lloyd
  - architecture
type: reference
segment: projects
---

# Backlog System

SQLite-based kanban task management integrated into the [[tools|MCP Tools Server]].

## Storage

- **Database:** `~/.openclaw/data/backlog.db` (SQLite)
- **Thread safety:** Python threading lock in `tool_services.py`

## Task States

```mermaid
graph LR
    inbox --> up_next --> in_progress --> in_review --> done
```

## Task Properties

| Property | Type | Description |
|----------|------|-------------|
| id | integer | Unique task ID |
| name | string | Task title |
| description | string | Task details |
| status | enum | inbox, up_next, in_progress, in_review, done |
| board_id | integer | Board association |
| priority | enum | none, low, medium, high |
| assigned | boolean | Whether task is assigned |
| blocked | boolean | Whether task is blocked |
| tags | string | Comma-separated tags |
| activity notes | list | Task history entries |

## Boards

Multiple kanban boards, each with its own task list.

## MCP Tools (4)

Previously 6 tools; consolidated to 4 by merging create/update into `backlog_write_task` and removing `backlog_next_task` (covered by `backlog_tasks` filters).

| Tool | Description |
|------|-------------|
| `backlog_boards` | List all boards |
| `backlog_tasks` | List/filter tasks (by status, assigned, blocked, board, tag) |
| `backlog_get_task` | Full task details by ID |
| `backlog_write_task` | Create or update a task (status, priority, blocked, activity notes) |

See [[tools]] for full tool definitions and parameters.

## Webhook Integration

Backlog triggers an OpenClaw webhook on task create/update:

- **Endpoint:** `POST /hooks/wake`
- **Auth:** `Authorization: Bearer <token>`
- **Token:** Configured in OpenClaw config (`hooks.token`) and backlog credentials

## Access Rules

- Lloyd queries backlog directly via MCP tools -- never answers backlog questions from vault/memory (notes may be stale)
- Task updates are attributed to the agent in activity history

## Previous System

The backlog was previously a Rails 8 app (Clawdeck) at `~/Development/clawdeck/`, port 3001, backed by PostgreSQL. It was migrated to the current SQLite-based system integrated into the [[tools|MCP tools server]].

## Related Docs

- [[index]] — High-Level Architecture
- [[tools]] — MCP Tools Server (tool implementations)
- [[agents]] — Agent System (task delegation)
- [[infrastructure]] — Infrastructure
