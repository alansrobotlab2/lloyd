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


def _land_detached(round_id: str) -> dict:
    """Start the landing in its own session and return immediately.

    THE reason this tool cannot just call `promote()`. A landing restarts
    `lloyd-mcp` and `lloyd-backend`, and this module runs *inside* lloyd-mcp.
    Both confs set `stopasgroup`/`killasgroup`, so supervisord signals the
    whole process group and kills the promoter partway through its own
    restart sequence. What that leaves is worse than a failed landing: the
    tree already fast-forwarded onto the new commit, the aggregator stopped
    (an intentional stop is not auto-restarted), the backend still serving
    the old code with no tools at all, and `current.json` frozen in `landing`
    — the one state the guardian is explicitly told not to observe. Nothing
    watches it and nothing rolls it back.

    `spawn_detached` puts the promoter in a new session, which is exactly what
    a process-group signal cannot reach. The CLI path survived this only by
    accident, because the Bash tool already spawns that way — so the loop
    worked when a human drove it and would have failed the first time Lloyd
    did, which is the case it exists for.
    """
    from scripts.selfmod import state as S, worktree as W

    gate_path = S.ROUNDS_DIR / round_id / "gate.json"
    if not gate_path.exists():
        return {"error": f"{round_id} has no gate report — run selfmod_gate first"}
    report = json.loads(gate_path.read_text())
    if not report.get("ok"):
        failed = [r["name"] for r in report.get("rungs", []) if not r["ok"]]
        return {"error": f"gate did not pass (failed: {failed})"}
    if not W.worktree_path(round_id).exists():
        return {"error": f"no worktree for {round_id}"}

    log = S.ROUNDS_DIR / round_id / "land.log"
    python = W.LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
    pid = S.spawn_detached(
        [python, "-m", "scripts.selfmod.round", "land", round_id],
        log, cwd=W.LIVE_ROOT)
    return {
        "landing": round_id, "pid": pid, "log": str(log),
        "next": "END YOUR TURN NOW. Do not poll, do not call another tool.",
        "note": (
            "Detached, because landing restarts lloyd-mcp and would otherwise kill "
            "the promoter mid-flight. It now waits for the backend to be IDLE — and "
            "your own turn is what is keeping it busy. Polling selfmod_status in a "
            "loop starves the gate you are waiting on, and after 15 minutes the "
            "landing gives up. Stop talking and the landing proceeds within seconds. "
            "It will restart the backend, which ends this turn anyway. Check "
            "selfmod_status on your NEXT turn: `current.state` goes landing -> "
            "observing, and the guardian settles it to last-known-good 15 minutes "
            f"after that. Progress is logged to {log}."),
    }


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
                "Promote a round whose gate passed. Returns IMMEDIATELY: the landing "
                "runs detached, because it restarts this very process. It waits for the "
                "backend to go IDLE, fast-forwards, restarts MCP then backend, and "
                "verifies the running commit changed. END YOUR TURN once this returns — "
                "your own turn is what keeps the backend busy, so polling starves the "
                "gate the landing is waiting on. Check selfmod_status on your next "
                "turn. The guardian then watches for 15 minutes before the commit "
                "becomes last-known-good."
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
            description=("Ask the guardian to revert the live tree and restart. Returns "
                         "immediately; the guardian acts within seconds and picks the "
                         "target the same way it would for a crash. Normally it does "
                         "this on its own — use this only when you need to force it."),
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
            if bool(arguments.get("dry_run")):
                return text_result(json.dumps(R.land(rid, dry_run=True), indent=2))
            return text_result(json.dumps(_land_detached(rid), indent=2))

        if name == "selfmod_abort":
            return text_result(json.dumps(R.abort(arguments.get("round_id") or ""), indent=2))

        if name == "selfmod_rollback":
            reason = arguments.get("reason") or "manual"
            # Handed to the guardian, never performed here. A rollback stops
            # lloyd-mcp — this process — so doing it inline meant issuing the
            # stop that kills the caller and then never reaching `git reset`.
            # The stack went down and the tree did not move: the one outcome
            # worse than either doing it or not. The guardian is outside that
            # blast radius and already owns evidence preservation, retries,
            # the denylist and flap protection.
            req = S.request_rollback(reason=str(reason)[:1000], trigger="manual")
            return text_result(json.dumps({
                "requested": True,
                "note": ("The guardian performs this within a few seconds. It picks the "
                         "target the same way it would for a crash: the promotion's own "
                         "recorded rollback_target first, never a stranded pointer. "
                         "Poll selfmod_status for the outcome."),
                "request": req,
            }, indent=2))

        return text_result(_err(f"Unknown tool: {name}"))

    try:
        return await asyncio.to_thread(_sync)
    except Exception as exc:
        logger.exception("selfmod tool %s failed", name)
        return text_result(_err(f"{type(exc).__name__}: {exc}"))
