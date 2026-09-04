#!/usr/bin/env python3
"""
Lloyd MCP Server: Thunderbird — email, calendar, tasks and contacts.

The Thunderbird MCP bridge (`mcp-bridge.cjs`) is itself an MCP server
speaking JSON-RPC over stdio. This module runs it as one, through the
SDK's stdio client, and re-exports its tools under Lloyd's naming.

Requires Thunderbird running with the MCP extension (localhost:8765).
If the bridge can't start, discovery returns an empty tool list and the
aggregator carries on without these tools — see `list_tools`.

History
-------
This module used to hand-roll the JSON-RPC client: `subprocess.Popen`
plus `select.select` and blocking `readline()` called straight from the
async handlers. That carried six defects, all of them live:

  1. Blocking I/O on the aggregator's only event loop — a slow mail
     search froze all ~124 tools, every session, the Discord bot and the
     background-task drain for up to 30 seconds.
  2. No mutex: concurrent calls wrote to one stdin and read one stdout.
  3. Request ids were `int(time.time()*1000) % 1000000`, so two calls in
     the same millisecond collided and each accepted the other's result.
  4. Non-matching replies were discarded, consuming the *other* caller's
     response and leaving it to time out.
  5. `_bridge_receive` ignored ids entirely and returned the first
     parseable line, so discovery could return a tool call's result.
  6. `stderr=PIPE` was never drained — the bridge wedged forever once
     Node filled the 64 KB pipe buffer.

All six are gone by construction now: `MCPPool` owns an SDK
`ClientSession`, which does id correlation, framing and concurrency, and
`stdio_client` drains stderr to the log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, Tool

from agent_mcp._shared import text_result

logger = logging.getLogger("lloyd-thunderbird")

BRIDGE_PATH = Path.home() / "lloyd" / "agent-services" / "services" / "thunderbird-mcp" / "mcp-bridge.cjs"

# Bridge tool name -> Lloyd tool name.
#
# Everything the bridge exports is mapped. Anything unmapped would fall
# through to a `tb_<rawName>` passthrough, which is how 19 tools ended up
# in a second, camelCase namespace beside the snake_case one; `list_tools`
# now logs when that happens so the gap is visible instead of shipping.
TOOL_NAME_MAP = {
    # Mail
    "listAccounts": "email_accounts",
    "getAccountAccess": "email_account_access",
    "listFolders": "email_folders",
    "createFolder": "email_create_folder",
    "renameFolder": "email_rename_folder",
    "deleteFolder": "email_delete_folder",
    "moveFolder": "email_move_folder",
    "emptyTrash": "email_empty_trash",
    "emptyJunk": "email_empty_junk",
    "searchMessages": "email_search",
    "getMessage": "email_read",
    "getMessages": "email_messages",
    "getRecentMessages": "email_recent",
    "updateMessage": "email_update",
    "deleteMessages": "email_delete",
    "displayMessage": "email_display",
    "sendMail": "email_send",
    "saveDraft": "email_save_draft",
    "replyToMessage": "email_reply",
    "forwardMessage": "email_forward",
    # Filters
    "listFilters": "email_list_filters",
    "createFilter": "email_create_filter",
    "updateFilter": "email_update_filter",
    "deleteFilter": "email_delete_filter",
    "reorderFilters": "email_reorder_filters",
    "applyFilters": "email_apply_filters",
    # Calendar
    "listCalendars": "calendar_list",
    "createEvent": "calendar_create",
    "listEvents": "calendar_events",
    "updateEvent": "calendar_update_event",
    "deleteEvent": "calendar_delete_event",
    "listCategories": "calendar_categories",
    # Tasks
    "listTasks": "tasks_list",
    "createTask": "tasks_create",
    "updateTask": "tasks_update",
    # Contacts
    "searchContacts": "contacts_search",
    "getContact": "contacts_get",
    "createContact": "contacts_create",
    "updateContact": "contacts_update",
    "deleteContact": "contacts_delete",
}

REVERSE_MAP = {v: k for k, v in TOOL_NAME_MAP.items()}

# The bridge writes its own tool descriptions, and several are terse enough
# ("Read a contact by UID", 21 chars) that the model can't tell them apart
# from their neighbours in a 124-tool list. These replace the bridge text
# where it is too thin to disambiguate; everything else passes through, so
# an improved bridge description still wins by default.
DESCRIPTION_OVERRIDES = {
    "email_accounts": (
        "List the configured email accounts with their identities and "
        "addresses. Start here when you need an account id for another "
        "mail tool."
    ),
    "email_folders": (
        "List the mail folders for an account, with each folder's URI and "
        "message count. The URI is what email_search and email_messages "
        "take to scope a query."
    ),
    "email_read": (
        "Read one email message in full by id: headers, body and "
        "attachment list. Use email_search or email_recent to find the id."
    ),
    "email_create_filter": (
        "Create a mail filter rule on an account — matching conditions and "
        "the actions to apply. Filters run on new mail; use "
        "email_apply_filters to run them over existing messages."
    ),
    "email_delete_filter": (
        "Delete a mail filter by its index within the account's filter "
        "list. Indexes shift after a delete, so re-read email_list_filters "
        "before deleting another."
    ),
    "calendar_list": (
        "List the user's calendars with their ids and names. Use an id to "
        "scope calendar_events or to create an event on the right calendar."
    ),
    "calendar_events": (
        "List calendar events between two dates, across all calendars or "
        "one. Returns event ids, times, titles and locations."
    ),
    "calendar_delete_event": (
        "Delete a calendar event by id. Permanent — there is no undo and "
        "no trash for calendar items."
    ),
    "contacts_get": (
        "Read one contact in full by UID: name, email addresses, phone "
        "numbers and the other stored fields. Use contacts_search to find "
        "the UID."
    ),
    "contacts_create": (
        "Create a contact in an address book. Takes the address book id "
        "plus the contact's fields; returns the new contact's UID."
    ),
    "contacts_update": (
        "Update an existing contact's fields by UID. Only the properties "
        "you pass are changed; the rest are left as they are."
    ),
    "contacts_delete": (
        "Delete a contact from its address book by UID. Permanent — the "
        "contact is not recoverable from Thunderbird afterwards."
    ),
}

# Discovery cache. The bridge was previously re-queried on every
# tools/list — a blocking round trip on each new MCP client connection.
# The tool list only changes when the bridge restarts, so a TTL is ample.
DISCOVERY_TTL_SECONDS = 300.0

_pool: Any = None                       # MCPPool | None
_pool_lock = asyncio.Lock()
_cached_tools: list[Tool] = []
_cached_at = 0.0


def _lloyd_name(bridge_name: str) -> str:
    return TOOL_NAME_MAP.get(bridge_name, f"tb_{bridge_name}")


def _document_params(schema: dict) -> dict:
    """Ensure every advertised parameter carries a description.

    These schemas come from the bridge, which is outside this repo and can
    regress independently. Rather than let an undocumented parameter reach
    the model — or fail the schema-hygiene test over something we don't
    own — fill a minimal description from the parameter name and log it, so
    the gap is visible and fixable upstream.
    """
    props = schema.get("properties")
    if not isinstance(props, dict):
        return schema
    missing = [
        k for k, v in props.items()
        if isinstance(v, dict) and not (v.get("description") or "").strip()
    ]
    if not missing:
        return schema
    patched = dict(schema)
    patched["properties"] = {
        k: ({**v, "description": k.replace("_", " ")}
            if k in missing and isinstance(v, dict) else v)
        for k, v in props.items()
    }
    logger.info(
        "thunderbird: bridge schema omits descriptions for %s — filled from "
        "parameter names", ", ".join(sorted(missing)),
    )
    return patched


async def _get_pool():
    """Open (once) an MCP stdio session to the bridge.

    Reuses `MCPPool` rather than a bespoke client: it already implements
    the owner-task pattern that keeps anyio cancel scopes consistent, and
    it is the same code path the harness uses for every other server.
    """
    global _pool
    async with _pool_lock:
        if _pool is not None and not _pool._poisoned:
            return _pool
        if not BRIDGE_PATH.exists():
            raise FileNotFoundError(f"MCP bridge not found at {BRIDGE_PATH}")
        from app.harness.mcp_pool import MCPPool

        pool = MCPPool({
            "thunderbird": {
                "type": "stdio",
                "command": "node",
                "args": [str(BRIDGE_PATH)],
            }
        })
        await pool.open()
        _pool = pool
        return _pool


async def list_tools() -> list[Tool]:
    """Discover the bridge's tools, renamed into Lloyd's namespace.

    Degrades to an empty list when Thunderbird isn't running: these tools
    simply don't appear, and the other modules are unaffected. Cached for
    DISCOVERY_TTL_SECONDS so each new client connection doesn't pay for a
    round trip to Node.
    """
    global _cached_tools, _cached_at
    if _cached_tools and (time.monotonic() - _cached_at) < DISCOVERY_TTL_SECONDS:
        return _cached_tools

    try:
        pool = await _get_pool()
    except Exception as exc:
        logger.warning("thunderbird: bridge unavailable (%s); exporting no tools", exc)
        _cached_tools, _cached_at = [], time.monotonic()
        return []

    tools: list[Tool] = []
    unmapped: list[str] = []
    for _server, discovered in pool.discovered:
        for t in discovered:
            bridge_name = t["name"]
            name = _lloyd_name(bridge_name)
            if name.startswith("tb_"):
                unmapped.append(bridge_name)
            description = DESCRIPTION_OVERRIDES.get(name) or (
                t.get("description") or f"Thunderbird: {bridge_name}"
            )
            tools.append(Tool(
                name=name,
                description=description,
                inputSchema=_document_params(
                    t.get("inputSchema") or {"type": "object", "properties": {}}
                ),
            ))
    if unmapped:
        logger.warning(
            "thunderbird: %d bridge tool(s) have no name mapping and ship as "
            "tb_*: %s — add them to TOOL_NAME_MAP",
            len(unmapped), ", ".join(sorted(unmapped)),
        )

    _cached_tools, _cached_at = tools, time.monotonic()
    return tools


async def call_tool(name: str, arguments: dict) -> CallToolResult:
    bridge_name = REVERSE_MAP.get(name)
    if bridge_name is None:
        if name.startswith("tb_"):
            bridge_name = name[3:]
        else:
            return text_result(json.dumps({"error": f"Unknown tool: {name}"}), is_error=True)

    try:
        pool = await _get_pool()
        result = await pool.call_tool(bridge_name, arguments or {})
    except Exception as exc:
        logger.warning("thunderbird: %s failed: %s", bridge_name, exc)
        return text_result(json.dumps({"error": str(exc)}), is_error=True)

    return text_result(
        result["content"] or "(no result)",
        is_error=result["is_error"],
    )


async def shutdown() -> None:
    """Close the bridge session and reap the Node subprocess.

    Called from the aggregator's lifespan. Without it every restart left
    an orphaned `node mcp-bridge.cjs` behind.
    """
    global _pool, _cached_tools, _cached_at
    pool, _pool = _pool, None
    _cached_tools, _cached_at = [], 0.0
    if pool is not None:
        try:
            await pool.aclose()
        except Exception as exc:
            logger.warning("thunderbird: shutdown failed: %s", exc)
