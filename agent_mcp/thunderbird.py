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
import os
import re
import time
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, Tool

from agent_mcp._shared import text_result

logger = logging.getLogger("lloyd-thunderbird")

from app.paths import LLOYD_HOME as _LH
BRIDGE_PATH = _LH / "agent-services" / "services" / "thunderbird-mcp" / "mcp-bridge.cjs"

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
        "message count. Pass the URI as folderPath to email_search or "
        "email_recent to scope a query."
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
    # The compose-style tools each restated the skipReview policy in their
    # description AND in the parameter; the parameter now says it once.
    "email_send": (
        "Compose a new email and open it in a review window for the user to "
        "send. Use email_reply to answer an existing message."
    ),
    "email_reply": (
        "Reply to a message, quoting the original, in a review window for the "
        "user to send. Find the message with email_search or email_recent."
    ),
    "email_forward": (
        "Forward a message with its original content, in a review window for "
        "the user to send."
    ),
    "email_save_draft": (
        "Save a composed message to the identity's Drafts folder without "
        "sending it or opening a window."
    ),
    "calendar_create": (
        "Create a calendar event through a review dialog the user confirms. "
        "Use calendar_list for a calendar id."
    ),
    "tasks_create": (
        "Create a task (to-do) through a review dialog the user confirms."
    ),
    "email_search": (
        "Search message headers (subject, author, recipients, preview) and "
        "return ids and folder paths; read a hit with email_read. For the "
        "newest mail with no query, use email_recent."
    ),
    "email_update": (
        "Mark messages read/unread or flagged, add or remove tags, or move "
        "them to a folder or Trash. messageId for one message, messageIds "
        "for several. On IMAP, tagging and moving in one call can drop the "
        "tags on the moved copy."
    ),
}

# Parameter prose, by parameter name, applied to every mail tool that has
# that parameter. The bridge's own wording is kept wherever it is short; these
# replace the long outliers and the paragraphs the bridge repeats on several
# tools (skipReview on five, offset on two). Types, enums and `required` are
# never touched — only the words. Keyed by name because the same parameter
# means the same thing on every tool that carries it; a tool-specific
# exception goes in TOOL_PARAM_OVERRIDES.
PARAM_DESCRIPTION_OVERRIDES = {
    "skipReview": (
        "Skip the review window. Honoured only if the user has switched "
        "the default-on safety block off (default false)."
    ),
    "offset": (
        "Results to skip, for paging (default 0). When set, the result is "
        "{messages, totalMatches, offset, limit, hasMore}."
    ),
    "includeInlineImages": (
        "Also return inline images as image blocks (default false; 1 MiB "
        "each, 4 MiB total). Ignored with rawSource."
    ),
    "rawSource": (
        "Return the raw RFC 2822 source instead of the parsed body, e.g. "
        "for calendar invites (default false). Needs an offline copy on IMAP."
    ),
    "onlineMeeting": (
        "Add (true) or remove (false) a Teams meeting link. Office 365 "
        "accounts only."
    ),
    "showAs": (
        "'busy' (opaque) or 'free' (transparent); default busy. An explicit "
        "status wins. On update, null clears it."
    ),
    "dedupByMessageId": (
        "Collapse copies of one message in several folders into one row, "
        "listing the rest in dupLocations (default true)."
    ),
    "searchBody": (
        "Search full bodies too, not just the ~200-char preview (slower; "
        "needs query; IMAP needs offline sync)."
    ),
    "addTags": (
        "Tag keywords to add. Built-in: $label1 Important, $label2 Work, "
        "$label3 Personal, $label4 To Do, $label5 Later."
    ),
    "categories": (
        "Category names, case-sensitive; get them from calendar_categories. "
        "On update, [] clears them."
    ),
    "from": "Sender identity: an address or identity id from email_accounts.",
    "isHtml": "True if body is HTML (default false).",
    "attachments": "File paths, or inline {name, contentType, base64} objects.",
}

TOOL_PARAM_OVERRIDES = {
    ("email_search", "query"): (
        "Words that must all appear across subject, author, recipients and "
        "preview. Prefix from:, subject:, to: or cc: to search one field. "
        "Empty matches everything."
    ),
}

# The bridge's descriptions name its own camelCase tools ("use with
# getMessage", "from listFolders"), which do not exist under those names here.
_BRIDGE_NAME_RE = re.compile(
    r"\b(" + "|".join(sorted(TOOL_NAME_MAP, key=len, reverse=True)) + r")\b")
_OPTIONAL_RE = re.compile(r"\s*\((?:optional)\)")


def _lloyd_prose(text: str) -> str:
    """Bridge prose with its tool names translated and "(optional)" dropped.

    "(optional)" restates what `required` already says, on dozens of params.
    """
    text = _BRIDGE_NAME_RE.sub(lambda m: TOOL_NAME_MAP[m.group(1)], text or "")
    return _OPTIONAL_RE.sub("", text).strip()


def _shape_params(name: str, schema: dict) -> dict:
    """The bridge schema with Lloyd's parameter prose; structure untouched."""
    props = schema.get("properties")
    if not isinstance(props, dict):
        return schema
    shaped = {}
    for pname, spec in props.items():
        if not isinstance(spec, dict):
            shaped[pname] = spec
            continue
        desc = (TOOL_PARAM_OVERRIDES.get((name, pname))
                or PARAM_DESCRIPTION_OVERRIDES.get(pname)
                or _lloyd_prose(spec.get("description") or ""))
        shaped[pname] = {**spec, "description": desc} if desc else {
            k: v for k, v in spec.items() if k != "description"}
    return {**schema, "properties": shaped}

# Discovery cache. The bridge was previously re-queried on every
# tools/list — a blocking round trip on each new MCP client connection.
# The tool list only changes when the bridge restarts, so a TTL is ample.
DISCOVERY_TTL_SECONDS = 300.0

# A failed discovery is cached too, for less time, so a Thunderbird that
# comes back is picked up within a minute. Until 2026-09-18 the cache check
# was `if _cached_tools and …`, which an empty list never passes: with
# Thunderbird closed every discovery re-tried the bridge and waited ~5 s for
# it to fail, twice per tool call. From 18:11 on 09-17 to 08:57 on 09-18 a
# `Glob` took 10 s end to end and every agent iteration ran 2-4x slower.
UNAVAILABLE_RETRY_SECONDS = 60.0

_pool: Any = None                       # MCPPool | None
_pool_lock = asyncio.Lock()
_discovery_lock = asyncio.Lock()
_cached_tools: list[Tool] = []
_cached_at: float | None = None         # None = never discovered


def _cache_fresh() -> bool:
    if _cached_at is None:
        return False
    ttl = DISCOVERY_TTL_SECONDS if _cached_tools else UNAVAILABLE_RETRY_SECONDS
    return (time.monotonic() - _cached_at) < ttl


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
    round trip to Node, and a failure for UNAVAILABLE_RETRY_SECONDS.

    One discovery at a time: concurrent callers that all missed the cache
    used to queue on `_pool_lock` and each wait out their own failed open in
    turn. The second check under `_discovery_lock` hands them the answer the
    first one just got.
    """
    if _cache_fresh():
        return _cached_tools
    async with _discovery_lock:
        if _cache_fresh():
            return _cached_tools
        return await _discover()


async def _discover() -> list[Tool]:
    global _cached_tools, _cached_at
    try:
        pool = await _get_pool()
    except Exception as exc:
        logger.warning("thunderbird: bridge unavailable (%s); exporting no tools "
                       "for %.0f s", exc, UNAVAILABLE_RETRY_SECONDS)
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
            description = DESCRIPTION_OVERRIDES.get(name) or _lloyd_prose(
                t.get("description") or f"Thunderbird: {bridge_name}"
            )
            tools.append(Tool(
                name=name,
                description=description,
                inputSchema=_document_params(_shape_params(
                    name, t.get("inputSchema") or {"type": "object", "properties": {}}
                )),
            ))
    if unmapped:
        logger.warning(
            "thunderbird: %d bridge tool(s) have no name mapping and ship as "
            "tb_*: %s — add them to TOOL_NAME_MAP",
            len(unmapped), ", ".join(sorted(unmapped)),
        )

    _cached_tools, _cached_at = tools, time.monotonic()
    return tools


# ── Bulk-id count bound (#614) ──────────────────────────────────────────────
#
# One call to `email_delete` or `email_update` carries an array of message ids
# and nothing bounded how many: the extension declares `messageIds` with no
# `maxItems` and its handler checks only non-empty, the harness grant gate
# never fires on an attended turn (`_authority_scope_for` returns "" for a user
# session, and `check_grants` allows interactive scope outright), and a
# predicate-less grant authorizes any array size. Meta's named failure is
# precisely this: an email agent deleted 200 messages after its owner told it
# to stop, having been prompted in advance to confirm destructive actions.
#
# This is the only landable seam. The extension is gitignored
# (`agent-services/.gitignore:72`), so a `maxItems` in its schema cannot come
# through a round, and `call_tool` is the one choke point every mail tool
# crosses — so the refusal happens before the call is forwarded and therefore
# before any grant decision, for every scope.
#
# Lifting it is `LLOYD_MAIL_ID_CAP`, read at call time: no code edit, no
# config.yaml change, no restart for a human who means to do bulk mail work.
MAIL_ID_ARRAY_CAP = 50
MAIL_ID_CAP_ENV = "LLOYD_MAIL_ID_CAP"

# Lloyd tool name -> the argument carrying an id array. Tools that take no id
# array (`email_empty_trash`, `email_empty_junk`, `email_delete_folder`,
# `contacts_delete`, `calendar_delete_event`) cannot be count-bounded here and
# are deliberately absent: their exposure is the grant gate's tier-3 problem,
# not this one.
_ID_ARRAY_ARGS = {"email_delete": "messageIds", "email_update": "messageIds"}


def _id_array_cap() -> int:
    """The cap for this call: the constant, or a positive env override.

    An override that is empty, non-numeric or non-positive falls back to the
    constant — a malformed override must never read as "no bound".
    """
    raw = (os.environ.get(MAIL_ID_CAP_ENV) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return MAIL_ID_ARRAY_CAP
    return value if value > 0 else MAIL_ID_ARRAY_CAP


def _id_array_refusal(lloyd_name: str, arguments: dict) -> str | None:
    """Refusal text when this call carries more ids than the cap, else None."""
    field = _ID_ARRAY_ARGS.get(lloyd_name)
    if field is None:
        return None
    ids = arguments.get(field)
    if not isinstance(ids, list):
        return None  # a singular messageId, or the field absent: nothing to bound
    cap = _id_array_cap()
    if len(ids) <= cap:
        return None
    return (
        f"refusing {len(ids)} {field} for {lloyd_name}: the per-call cap is "
        f"{cap}. Pass at most {cap} ids per call, or set "
        f"{MAIL_ID_CAP_ENV}=<n> for deliberate bulk work."
    )


async def call_tool(name: str, arguments: dict) -> CallToolResult:
    bridge_name = REVERSE_MAP.get(name)
    if bridge_name is None:
        if name.startswith("tb_"):
            bridge_name = name[3:]
        else:
            return text_result(json.dumps({"error": f"Unknown tool: {name}"}), is_error=True)

    refusal = _id_array_refusal(_lloyd_name(bridge_name), arguments or {})
    if refusal is not None:
        logger.warning("thunderbird: %s — call not forwarded", refusal)
        return text_result(json.dumps({"error": refusal}), is_error=True)

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
    _cached_tools, _cached_at = [], None
    if pool is not None:
        try:
            await pool.aclose()
        except Exception as exc:
            logger.warning("thunderbird: shutdown failed: %s", exc)
