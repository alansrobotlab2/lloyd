"""MCP tools for the self-modification loop.

Mirrors `agent_mcp/autoresearch.py`: the loop's real logic lives under
`scripts/selfmod/`, and this module is the thin surface the agent calls.

The proposal step is deliberately absent. There is no `selfmod_write_code`
tool — `selfmod_start` hands back a worktree path and Lloyd edits it with the
ordinary Edit/Write/Bash tools. The round is a wrapper around normal work, not
a special code-generation mode.

Every mutating tool refuses while `selfmod.enabled` is false, so the machinery
ships inert.
"""

from __future__ import annotations

import json
import logging

from mcp.types import TextContent, Tool

from agent_mcp._shared import text_result

logger = logging.getLogger("lloyd-mcp.selfmod")


def _enabled() -> bool:
    try:
        from app.config import CONFIG
        return bool((CONFIG.get("selfmod") or {}).get("enabled", False))
    except Exception:
        return False


def _err(message: str) -> str:
    """Errors are JSON with an `error` key — text_result sniffs that shape."""
    return json.dumps({"error": message})


async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="selfmod_start",
            description=(
                "Open a self-modification round: creates a git worktree off the current "
                "HEAD and returns its path. Edit that path with the normal Edit/Write/Bash "
                "tools, commit inside it, then call selfmod_gate. Only one round at a time."
            ),
            inputSchema={
                "type": "object",
                "properties": {"goal": {"type": "string",
                                        "description": "What this round is trying to change."}},
                "required": ["goal"],
            },
        ),
        Tool(
            name="selfmod_gate",
            description=(
                "Run the promotion gate on a round: static checks, the full test suite, a "
                "candidate venv if requirements changed, a canary boot on alternate ports, "
                "one real agent turn, and a guardian drill if the diff touches the rollback "
                "path. Returns a per-rung report. Takes minutes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string",
                                  "description": "Round id returned by selfmod_start."},
                    "skip_smoke": {"type": "boolean",
                                   "description": "Skip the live-LLM turn (rung 5). Use only "
                                                  "when vLLM is unavailable."},
                },
                "required": ["round_id"],
            },
        ),
        Tool(
            name="selfmod_land",
            description=(
                "Promote a round whose gate passed: waits for the backend to go idle, "
                "fast-forwards the live tree, restarts MCP then backend, and verifies the "
                "running commit changed. Refuses if the gate did not pass. The guardian "
                "then watches for 15 minutes before the commit becomes last-known-good."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string",
                                  "description": "Round id whose gate passed."},
                    "dry_run": {"type": "boolean",
                                "description": "Report what would be promoted without "
                                               "touching the live tree."},
                },
                "required": ["round_id"],
            },
        ),
        Tool(
            name="selfmod_status",
            description=("Current self-modification state: last known good commit, the "
                         "promotion being observed, halt/broken flags, and recent ledger events."),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="selfmod_abort",
            description=("Abandon a round. Removes its worktree but KEEPS the branch, which "
                         "is the only record of what was attempted."),
            inputSchema={
                "type": "object",
                "properties": {"round_id": {"type": "string",
                                            "description": "Round id to abandon."}},
                "required": ["round_id"],
            },
        ),
        Tool(
            name="selfmod_rollback",
            description=("Manually revert the live tree to the last known good commit and "
                         "restart. Normally the guardian does this automatically; use this "
                         "only when you need to force it."),
            inputSchema={
                "type": "object",
                "properties": {"reason": {"type": "string",
                                          "description": "Why a manual rollback is needed; "
                                                         "recorded in the ledger."}},
                "required": ["reason"],
            },
        ),
    ]


async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    import asyncio

    def _sync():
        from scripts.selfmod import round as R, state as S

        if name == "selfmod_status":
            return text_result(json.dumps(R.status(), indent=2, default=str))

        if not _enabled() and name != "selfmod_abort":
            return text_result(_err(
                "selfmod.enabled is false in config.yaml. The self-modification loop "
                "ships inert; enabling it is a deliberate human decision."))

        if name == "selfmod_start":
            goal = (arguments.get("goal") or "").strip()
            if not goal:
                return text_result(_err("goal is required"))
            return text_result(json.dumps(R.start(goal), indent=2))

        if name == "selfmod_gate":
            rid = arguments.get("round_id") or ""
            rep = R.run_gate(rid, skip_smoke=bool(arguments.get("skip_smoke")))
            return text_result(json.dumps(rep, indent=2))

        if name == "selfmod_land":
            rid = arguments.get("round_id") or ""
            return text_result(json.dumps(
                R.land(rid, dry_run=bool(arguments.get("dry_run"))), indent=2))

        if name == "selfmod_abort":
            return text_result(json.dumps(R.abort(arguments.get("round_id") or ""), indent=2))

        if name == "selfmod_rollback":
            reason = arguments.get("reason") or "manual"
            target, source = None, ""
            lkg = S.read_lkg()
            if lkg:
                target = lkg.get("commit")
            if not target:
                return text_result(_err("no last-known-good commit recorded — refusing to guess"))
            from scripts.selfmod.promote import _rollback_inline, LIVE_ROOT
            _rollback_inline(LIVE_ROOT, target)
            S.append_event({"event": "rollback_succeeded", "trigger": "manual",
                            "restored": target, "reason": reason})
            return text_result(f"rolled back to {target[:8]} ({reason})")

        return text_result(_err(f"Unknown tool: {name}"))

    try:
        return await asyncio.to_thread(_sync)
    except Exception as exc:
        logger.exception("selfmod tool %s failed", name)
        return text_result(_err(f"{type(exc).__name__}: {exc}"))
