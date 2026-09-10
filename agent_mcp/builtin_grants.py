#!/usr/bin/env python3
"""Lloyd MCP Server: authority grants (#534) — the human's mint path.

`grant_create` / `grant_list` / `grant_revoke` are the interactive half of
scope-bound expiring authority. The other half — `grants:` frontmatter on an
autonomy task file — needs no tool, because editing the task file IS the
approval, and the load-time validator turns a malformed block into a stopped
task.

Why a tool at all, when the answer to a denial could plausibly be a file
edit: the denial text names a grant shape, and the point of #534 is that the
human's interaction moves from interrupting a live run to a batched renewal
someone can action in one line. That only holds if the action is one call.

## Interactive scope only

A turn subject to the grant gate must not be able to write its way out of it.
So minting refuses anything that is not a session a human reads:

* no bound session → refused. `run_prompt_on_primary` turns carry no session
  at all, which is the case `tests/unit/test_grant_policy.py` pins.
* bound session whose `platform` is `worker` or `autonomy` → refused
  (`app.sessions_io.is_user_session`). Session-backed worker sources exist,
  and a session id alone does not make a turn interactive.
* `app/harness/policy.py::check_grants` independently denies `grant_create`
  for any non-interactive scope before the call is ever dispatched, and
  `_worker_run_options` does not advertise the tool. Three layers, because
  the failure mode of the inner two is "silently everything is allowed".

The store refuses a `minted_by`/`issued_by` naming a worker identity too, so
the audit red line — grants minted by a worker scope must count zero — holds
even if a future caller forgets all three.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.types import Tool

from agent_mcp._shared import get_bound_session, text_result

logger = logging.getLogger("lloyd-builtin-grants")


def _store():
    from app.harness.policy import default_store
    return default_store()


async def _refusal_if_not_human(session_id: str) -> str | None:
    """Why this turn may not mint, or None if it may."""
    if not session_id:
        return ("grant_create needs a session a human is present in, and this "
                "call carries none. Worker and autonomy turns cannot mint "
                "authority; ask a human to issue the grant.")
    from app.sessions_io import SESSIONS_DIR, is_user_session

    meta = SESSIONS_DIR / f"{session_id}.json"
    if not meta.exists():
        return f"grant_create: session {session_id} not found, cannot verify a human is present."
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"grant_create: could not read session metadata ({exc}); not minting."
    if not is_user_session(data):
        return ("grant_create: this session's platform is "
                f"{data.get('platform')!r} — an unattended turn, which may not "
                "issue its own authority.")
    return None


async def _grant_create(args: dict[str, Any]) -> str:
    refused = await _refusal_if_not_human(get_bound_session())
    if refused:
        logger.warning("[grants] refused mint from session=%r", get_bound_session())
        return json.dumps({"error": refused})

    try:
        row = _store().mint(
            scope=args.get("scope"),
            tool_pattern=args.get("tool"),
            arg_predicate=args.get("predicate") or "",
            quota=args.get("quota"),
            issued_by=args.get("issued_by"),
            expires_at=args.get("expires_at"),
            note=args.get("note") or "",
            minted_by="interactive-tool",
        )
    except Exception as exc:  # GrantError and nothing else is expected
        return json.dumps({"error": f"grant refused: {exc}"})

    logger.info("[grants] minted #%s scope=%s tool=%s expires=%s by %s",
                row["id"], row["scope"], row["tool_pattern"],
                row["expires_at"], row["issued_by"])
    return json.dumps({
        "granted": True, "grant_id": row["id"], "scope": row["scope"],
        "tool": row["tool_pattern"], "predicate": row["arg_predicate"],
        "quota": row["quota"], "expires_at": row["expires_at"],
        "note": ("This grant dies on its own date. Renewing it means minting a "
                 "new row — there is deliberately no renewal call."),
    })


async def _grant_list(args: dict[str, Any]) -> str:
    store = _store()
    try:
        rows = store.live(scope=args.get("scope") or None)
    except Exception as exc:
        return json.dumps({"error": f"grant store unreadable: {exc}"})
    return json.dumps({
        "live_grants": [
            {k: r.get(k) for k in ("id", "scope", "tool_pattern",
                                   "arg_predicate", "quota", "consumed",
                                   "issued_by", "expires_at", "minted_by")}
            for r in rows],
        "count": len(rows),
    })


async def _grant_revoke(args: dict[str, Any]) -> str:
    refused = await _refusal_if_not_human(get_bound_session())
    if refused:
        return json.dumps({"error": refused})
    grant_id = args.get("grant_id")
    if grant_id is None:
        return json.dumps({"error": "grant_id is required"})
    try:
        revoked = _store().revoke(int(grant_id))
    except (TypeError, ValueError):
        return json.dumps({"error": f"grant_id must be an integer, got {grant_id!r}"})
    if not revoked:
        return json.dumps({"error": f"grant #{grant_id} is already revoked or "
                                    "does not exist"})
    # Takes effect at the next dispatch, including inside a run that already
    # consumed this grant — a revocation that waited for the run to finish
    # would be a revocation that arrives after the thing it revokes.
    logger.warning("[grants] revoked grant #%s", grant_id)
    return json.dumps({"revoked": int(grant_id),
                       "effective": "next dispatch, including mid-run"})


_CREATE_DESC = """Mint an expiring, quota-bound authority grant (#534).

Interactive scope only: a worker or autonomy turn cannot call this, and the
grant store refuses a row whose issuer names a non-interactive identity. A
grant is what lets a tight gate still let real work through — the human
issues it in advance, it names one bounded scope, and it dies on its date.

## Args
- `scope` (required): whose authority this covers — `worker:<source>` or `autonomy-task:<id>`.
- `tool` (required): the bare tool name, e.g. `email_send`.
- `predicate` (optional): `len(field)<=N` or `field<=N`. The only grammar accepted; it is parsed, never evaluated.
- `quota` (optional): max executions. Unbounded if omitted — prefer setting it.
- `expires_at` (required): ISO date or datetime. Mandatory; there is no default and no infinity.
- `issued_by` (required): who authorized it. Names a human.

A denial that says "no grant for X from scope Y" renders this exact call shape."""

_LIST_DESC = """List live (unexpired, unrevoked) authority grants, optionally for one scope.

Read-only. This is what a renewal pass looks at: rows whose `expires_at` is
near, and rows nobody can say why they exist."""

_REVOKE_DESC = """Revoke an authority grant by id (#534).

Takes effect at the next dispatch, including a dispatch inside the run that
already consumed the grant. Revoking is the answer to 'that grant is broader
than I meant'; renewing is minting a new row, never extending this one."""


async def list_tools():
    return [
        Tool(name="grant_create", description=_CREATE_DESC, inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "tool": {"type": "string"},
                "predicate": {"type": "string"},
                "quota": {"type": "integer", "minimum": 1},
                "expires_at": {"type": "string",
                               "description": "ISO date (YYYY-MM-DD) or datetime. Required."},
                "issued_by": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["scope", "tool", "expires_at", "issued_by"],
        }, annotations={"readOnlyHint": False, "destructiveHint": False}),
        Tool(name="grant_list", description=_LIST_DESC, inputSchema={
            "type": "object",
            "properties": {"scope": {"type": "string"}},
        }, annotations={"readOnlyHint": True, "destructiveHint": False}),
        Tool(name="grant_revoke", description=_REVOKE_DESC, inputSchema={
            "type": "object",
            "properties": {"grant_id": {"type": "integer"}},
            "required": ["grant_id"],
        }, annotations={"readOnlyHint": False, "destructiveHint": True}),
    ]


async def call_tool(name: str, arguments: dict):
    args = arguments or {}
    if name == "grant_create":
        text = await _grant_create(args)
    elif name == "grant_list":
        text = await _grant_list(args)
    elif name == "grant_revoke":
        text = await _grant_revoke(args)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return text_result(text)
