#!/usr/bin/env python3
"""
Lloyd MCP Server: Ambient — background-producer context injection.

Provides two tools for the autonomy → active session context pipeline
(#295):

  - `session_inject_context`: producers (autonomy tasks, cron jobs,
    pipeline stages) push context toward the user's active chat
    session. The `priority` field picks the delivery mechanism:
      - `ambient`  → cheap prefetch drain (no SDK turn)
      - `notable`  → synthetic agent turn (agent may stay silent via
                     `ambient_decide(surface=false)`)
      - `urgent`   → synthetic agent turn, framed for immediate surface

  - `ambient_decide`: agent-only tool, callable during an ambient turn.
    Lets the agent opt out of surfacing a background signal to the user
    without generating assistant output. Keeps 99% of nudges silent.

Both tools are thin MCP wrappers over server HTTP endpoints. The server
owns session identity, queue state, and transcript persistence; these
tools just mint requests.
"""

import json

import httpx

from agent_mcp._shared import make_http_client
from app.config import service_url
from mcp.server import Server
from mcp.types import Tool, TextContent

LLOYD_BACKEND = service_url("backend", "http://127.0.0.1:8080")

app = Server("lloyd-ambient")


async def _post_json(path: str, body: dict, timeout: float = 10.0) -> tuple[int, dict]:
    """POST JSON to the Lloyd backend. Returns (status, parsed_body_or_raw)."""
    url = f"{LLOYD_BACKEND}{path}"
    async with make_http_client(timeout=timeout) as client:
        r = await client.post(url, json=body)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text}


async def _get_json(path: str, timeout: float = 5.0) -> tuple[int, dict]:
    async with make_http_client(timeout=timeout) as client:
        r = await client.get(f"{LLOYD_BACKEND}{path}")
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text}


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="session_inject_context",
            description=(
                "Inject background-producer context into the user's active chat session. "
                "Use for email summaries, calendar nudges, background research results, "
                "autonomy-task findings, or any signal the user should eventually see. "
                "\n\nPriority picks the mechanism:\n"
                "  - 'ambient' (DEFAULT — 99% of uses): cheap, zero SDK cost. Queued and "
                "    folded into the <context> block on the user's next turn. No "
                "    interruption. The agent references it naturally if relevant.\n"
                "  - 'notable': fires a real agent turn that the agent can choose to stay "
                "    silent on via ambient_decide(surface=false). Use when a signal might "
                "    need surfacing within the current session but isn't time-critical.\n"
                "  - 'urgent': fires a real agent turn framed for immediate surface. Use "
                "    sparingly — every urgent costs a full SDK call.\n"
                "\nIf session_id is empty, routes to the user's most recent chat session "
                "(excluding autonomy's own sessions). If there is no active user session, "
                "returns a graceful no-op; producers do not need to handle this case."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Stable producer identifier, e.g. 'autonomy:task-42' or 'pipeline:research-117'. Appears in the context envelope so the agent knows what fired it.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "One-line summary (<120 chars ideal). This is what the agent will primarily see in the prefetch drain.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional fuller body. Rendered nested under the summary. Keep under ~800 chars or it'll be truncated.",
                        "default": "",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["ambient", "notable", "urgent"],
                        "description": "Delivery tier — see tool description. Default 'ambient'.",
                        "default": "ambient",
                    },
                    "dedup_key": {
                        "type": "string",
                        "description": "Collapse multiple firings into one. Default = source. Same producer re-firing replaces its previous unsent entry.",
                        "default": "",
                    },
                    "ttl_seconds": {
                        "type": "integer",
                        "description": "Drop the entry if the user hasn't engaged this many seconds later. Default 3600. Only applies to priority=ambient.",
                        "default": 3600,
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Target session. If empty, resolves to the user's most recent chat session.",
                        "default": "",
                    },
                },
                "required": ["source", "summary"],
            },
        ),
        Tool(
            name="ambient_decide",
            description=(
                "Agent-only decision tool — callable ONLY during a turn where "
                "source='ambient' (i.e. you were invoked by session_inject_context with "
                "priority='notable' or 'urgent'). Use this to opt out of surfacing the "
                "background signal to the user.\n\n"
                "- surface=false: Suppress your assistant output. A muted breadcrumb is "
                "  written to the transcript so the user knows a check happened. Call "
                "  this when the signal isn't actually worth the user's attention.\n"
                "- surface=true: Continue normally — your next assistant message will be "
                "  shown to the user, tagged as ambient-sourced. Only needed if you want "
                "  to explicitly log the decision; the default if you just reply is also "
                "  to surface.\n\n"
                "Server rejects the call if the current turn is not ambient. The "
                "session_id is embedded in the ambient envelope you received — copy it "
                "from there."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID from the <ambient session_id='...'> envelope.",
                    },
                    "surface": {
                        "type": "boolean",
                        "description": "True = surface to user (default behavior). False = stay silent, log breadcrumb only.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Optional — if surface=true, what you plan to say. Informational only; your actual next assistant output is what lands in the transcript.",
                        "default": "",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief why. Shown in the silent breadcrumb so the user can see what you decided and why.",
                        "default": "",
                    },
                },
                "required": ["session_id", "surface"],
            },
        ),
    ]


async def _resolve_session_id(session_id: str) -> str | None:
    """Return session_id if provided, else ask the server for active session."""
    sid = (session_id or "").strip()
    if sid:
        return sid
    try:
        status, body = await _get_json("/api/sessions/active")
        if status == 200:
            return body.get("session_id")
    except Exception:
        return None
    return None


async def _tool_session_inject_context(args: dict) -> list[TextContent]:
    source = (args.get("source") or "").strip()
    summary = (args.get("summary") or "").strip()
    if not source or not summary:
        return [TextContent(type="text", text=json.dumps({
            "error": "source and summary are required",
        }))]

    priority = (args.get("priority") or "ambient").strip().lower()
    if priority not in ("ambient", "notable", "urgent"):
        priority = "ambient"

    session_id = await _resolve_session_id(args.get("session_id") or "")
    if not session_id:
        return [TextContent(type="text", text=json.dumps({
            "skipped": True,
            "reason": "no active user session",
        }))]

    content = (args.get("content") or "").strip()
    dedup_key = (args.get("dedup_key") or source).strip()
    ttl_seconds = int(args.get("ttl_seconds") or 3600)

    if priority == "ambient":
        # Mechanism 1 — cheap prefetch drain, no SDK turn.
        status, body = await _post_json(
            f"/api/sessions/{session_id}/inject-prefetch",
            {
                "source": source,
                "summary": summary,
                "content": content,
                "dedup_key": dedup_key,
                "ttl_seconds": ttl_seconds,
            },
        )
    else:
        # Mechanism 2 — synthetic turn via existing /inject path.
        # The text fed to the agent is summary + content, with the envelope
        # and ambient_decide hint added server-side by build_ambient_turn.
        text_body = summary
        if content:
            text_body += "\n\n" + content
        status, body = await _post_json(
            f"/api/sessions/{session_id}/inject",
            {
                "text": text_body,
                "dedup_key": dedup_key,
                "priority": priority,
                "source": source,
                "summary": summary,
            },
        )

    result = {
        "ok": 200 <= status < 300,
        "status": status,
        "session_id": session_id,
        "priority": priority,
        "mechanism": "prefetch" if priority == "ambient" else "turn",
        "server_response": body,
    }
    return [TextContent(type="text", text=json.dumps(result))]


async def _tool_ambient_decide(args: dict) -> list[TextContent]:
    session_id = (args.get("session_id") or "").strip()
    if not session_id:
        return [TextContent(type="text", text=json.dumps({
            "error": "session_id is required — copy it from the <ambient session_id='...'> envelope",
        }))]

    if "surface" not in args:
        return [TextContent(type="text", text=json.dumps({
            "error": "surface (bool) is required",
        }))]

    body = {
        "surface": bool(args.get("surface")),
        "message": (args.get("message") or "").strip(),
        "reasoning": (args.get("reasoning") or "").strip(),
    }
    status, resp = await _post_json(
        f"/api/sessions/{session_id}/ambient-decide",
        body,
    )

    return [TextContent(type="text", text=json.dumps({
        "ok": 200 <= status < 300,
        "status": status,
        "server_response": resp,
    }))]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "session_inject_context":
        return await _tool_session_inject_context(arguments)
    if name == "ambient_decide":
        return await _tool_ambient_decide(arguments)
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
