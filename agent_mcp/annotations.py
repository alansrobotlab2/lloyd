"""Tool annotations — declarative behaviour hints for every Lloyd tool.

MCP's `ToolAnnotations` (spec revision 2025-03-26, carried forward into
2026-07-28) let a tool declare what it does rather than having every
consumer maintain its own list of names:

    readOnlyHint     does not modify any state
    destructiveHint  may irreversibly destroy data
    idempotentHint   calling it twice with the same args changes nothing more
    openWorldHint    touches systems outside this machine

Why a central table rather than annotations on each `Tool(...)`:
the classification is security-relevant — `app.mcp_discovery` derives the
plan-mode block list from `readOnlyHint` — and reviewing one ordered file
is far easier than auditing 124 constructor calls spread over 20 modules.
A tool missing from this table is treated as NOT read-only, which is the
safe default: it gets blocked in plan mode until someone classifies it.

Sets are explicit rather than pattern-matched on purpose. A regex over
tool names gets `email_apply_filters` and `autonomy_get_task` wrong in
opposite directions, and a misclassification here silently widens what
the agent may do while drafting a plan.
"""

from __future__ import annotations

from mcp.types import Tool, ToolAnnotations

# ---------------------------------------------------------------------------
# Read-only: observation with no state change anywhere.
# ---------------------------------------------------------------------------
READ_ONLY: frozenset[str] = frozenset({
    # Filesystem
    "Read", "Grep", "Glob",
    # Internal
    "_BackgroundTaskDrain",
    # Knowledge graph
    "fact_get", "fact_profile", "fact_check", "fact_resolve",
    "fact_relationships", "fact_path", "fact_neighbors",
    # Vault + memory + sessions
    "vault_read", "vault_overview", "vault_search", "vault_recall",
    "memory_read", "session_recall", "chat_list_sessions", "chat_get_session",
    # Skills
    "skills_search", "skills_read",
    # Autonomy / backlog / research — inspection halves
    "autonomy_tasks", "autonomy_get_task", "autonomy_config", "autonomy_health",
    "backlog_boards", "backlog_tasks", "backlog_get_task",
    "autoresearch_status", "autoresearch_bench_list", "autoresearch_ledger_query",
    # Mission control
    "mc_get_state",
    # Web + browser observation
    "http_search", "http_fetch",
    "browser_snapshot", "browser_screenshot",
    # Mail / calendar / contacts — read halves
    "email_accounts", "email_account_access", "email_folders", "email_search",
    "email_read", "email_messages", "email_recent", "email_list_filters",
    "calendar_list", "calendar_events", "calendar_categories",
    "tasks_list", "contacts_search", "contacts_get",
    "discord_list_channels", "discord_get_home_channel",
})

# ---------------------------------------------------------------------------
# Destructive: may irreversibly destroy data. A strict subset of "not
# read-only" — used for UI badging and for reasoning about blast radius,
# not for the plan-mode gate (which keys off readOnlyHint).
# ---------------------------------------------------------------------------
DESTRUCTIVE: frozenset[str] = frozenset({
    "Write",                      # truncates an existing file
    "Bash",                       # arbitrary command execution
    "fact_invalidate",
    "memory_remove",
    "autonomy_delete_task",
    "autoresearch_rollback",
    "email_delete", "email_delete_filter", "email_delete_folder",
    "email_empty_trash", "email_empty_junk",
    "calendar_delete_event",
    "contacts_delete",
})

# ---------------------------------------------------------------------------
# Idempotent: repeating the call with identical arguments adds no further
# change. Setters and deletes qualify; appenders and senders do not.
# ---------------------------------------------------------------------------
IDEMPOTENT: frozenset[str] = frozenset({
    "Write", "SetGoal", "ClearGoal", "TodoWrite",
    "EnterPlanMode", "ExitPlanMode",
    "vault_write", "memory_replace", "memory_remove",
    "autonomy_delete_task", "autonomy_write_task",
    "email_delete", "email_delete_filter", "email_delete_folder",
    "email_empty_trash", "email_empty_junk",
    "calendar_delete_event", "calendar_update_event",
    "contacts_delete", "contacts_update",
    "tasks_update", "fact_invalidate",
    "ide_open_file", "ide_open_folder", "ide_close_tab",
    "mc_navigate", "mc_close_modal",
})

# ---------------------------------------------------------------------------
# Open world: reaches systems beyond this machine. Everything mail,
# calendar and contacts touches an IMAP/CalDAV account; browser and http
# reach the internet; Discord reaches a gateway; Task spawns a subagent
# with the full tool pool.
# ---------------------------------------------------------------------------
_OPEN_WORLD_PREFIXES = ("http_", "browser_", "email_", "calendar_",
                        "contacts_", "tasks_", "discord_")
_OPEN_WORLD_EXTRA: frozenset[str] = frozenset({"Task", "Bash"})


def is_open_world(name: str) -> bool:
    return name.startswith(_OPEN_WORLD_PREFIXES) or name in _OPEN_WORLD_EXTRA


def annotations_for(name: str) -> ToolAnnotations:
    """Build the ToolAnnotations for one tool name.

    Unlisted names default to read_only=False / destructive=False — the
    safe end for the gate, the conservative end for the badge.
    """
    return ToolAnnotations(
        readOnlyHint=name in READ_ONLY,
        destructiveHint=name in DESTRUCTIVE,
        idempotentHint=name in IDEMPOTENT or name in READ_ONLY,
        openWorldHint=is_open_world(name),
    )


def annotate(tool: Tool) -> Tool:
    """Return `tool` with annotations attached, preserving any it already has.

    A module that sets its own annotations wins — the table is a default
    for the 124 tools that don't, not an override.
    """
    if tool.annotations is not None:
        return tool
    tool.annotations = annotations_for(tool.name)
    return tool


def read_only_tool_names() -> frozenset[str]:
    """The canonical read-only set, for consumers deriving gates from it."""
    return READ_ONLY


# ---------------------------------------------------------------------------
# Plan-mode gate
# ---------------------------------------------------------------------------
#
# Plan mode exists so primary can draft a plan without mutating anything.
# It used to block exactly three tools — Write, Edit, Bash — which left
# `email_send`, `vault_write`, `fact_add`, `discord_send`, `browser_click`
# and every other actuator wide open. Deriving the gate from readOnlyHint
# closes that, but two corrections are needed before it is usable:
#
#   1. Session bookkeeping and the plan-mode control tools are technically
#      "not read-only" and must stay allowed, or plan mode blocks its own
#      exit. `ExitPlanMode` being unreachable is a deadlock, not a guard.
#   2. Tools that only move the local UI around (mission control panes, IDE
#      tabs) mutate no user data and reach nothing outside this machine.
#      Blocking them degrades how a plan is presented and guards nothing.
#
# Everything else that can write, send, spawn or click is blocked.
PLAN_MODE_ALWAYS_ALLOWED: frozenset[str] = frozenset({
    # Plan-mode control + session bookkeeping
    "EnterPlanMode", "ExitPlanMode", "TodoWrite", "SetGoal", "ClearGoal",
    "ToolSearch",
    # Local UI navigation — presentation, not mutation
    "mc_navigate", "mc_close_modal",
    "ide_open_file", "ide_open_folder", "ide_close_tab",
    # Ambient routing decision: records a routing choice for the current
    # turn, reaches nothing outside the process.
    "ambient_decide",
})


def plan_mode_blocked_tools(all_tool_names: frozenset[str] | set[str]) -> list[str]:
    """Tools primary may not call while plan mode is active.

    Derived from the annotation table rather than a hardcoded list, so a
    new actuator tool is gated the day it is added instead of the day
    someone remembers to update `mcp_discovery`.
    """
    return sorted(
        n for n in all_tool_names
        if n not in READ_ONLY
        and n not in PLAN_MODE_ALWAYS_ALLOWED
        and not n.startswith("_")
    )
