"""MCP tools for the self-modification loop.

Mirrors `agent_mcp/autoresearch.py`: the loop's real logic lives under
`scripts/autoimplement/`, and this module is the thin surface the agent calls.

The proposal step is deliberately absent. There is no `autoimplement_write_code`
tool — `autoimplement_start` hands back a worktree path and Lloyd edits it with the
ordinary Edit/Write/Bash tools. The round is a wrapper around normal work, not
a special code-generation mode.

Every mutating tool refuses while `autoimplement.enabled` is false, so the machinery
ships inert.
"""

from __future__ import annotations

import json
import logging

from mcp.types import TextContent, Tool

from agent_mcp._shared import get_bound_session, text_result

logger = logging.getLogger("lloyd-mcp.autoimplement")


def _enabled() -> bool:
    try:
        from app.config import CONFIG
        return bool((CONFIG.get("autoimplement") or {}).get("enabled", False))
    except Exception:
        return False


def _require_inner_voice() -> bool:
    """`autoimplement.require_inner_voice` in config.yaml, default true."""
    try:
        from app.config import CONFIG
        return bool((CONFIG.get("autoimplement") or {}).get("require_inner_voice", True))
    except Exception:
        return True


def _inner_voice_gate(action: str) -> dict | None:
    """Refuse to drive the loop from a turn with no observer attached.

    Returns an error payload to hand back, or None to proceed.

    **Enabling the flag is not sufficient, which is why this refuses rather
    than merely switching it on.** Inner Voice attaches at turn START:
    `attach_observer_for_turn` runs before `run_query` and is the only place
    the observer is installed. Flipping `inner_voice` from inside a tool call
    rewrites the session file for the *next* turn and does nothing for the one
    making the call. Same shape as the position-0 rule for the system prompt.
    A round driven inside a single turn would otherwise report itself observed
    while running blind, which is worse than being plainly unobserved.

    So: enable it, then refuse once. The retry runs observed, and the property
    is real rather than aspirational.

    Worth the friction for two independent reasons, and the second is the one
    that survives a quiet round.

    *Live:* a round is the one thing Lloyd does that rewrites production, and
    the observer is what notices the loop drifting off the request, repeating
    itself, or talking its way into a change nobody asked for.

    *Afterwards:* an IV session is the only kind the Inner Voice tab lists, so
    running rounds there is what makes self-modification **reviewable** — the
    transcript, the observations and the interventions all land somewhere a
    human can page back through. A round driven from a plain session leaves
    nothing but a ledger row and a diff, which tells you what changed and
    nothing about how the agent got there.

    `app/routers/messages.py` is the ONLY turn path that wires the observer —
    worker turns and Task subagents have none at all, which is why neither is
    allowed to drive the loop.

    No bound session means this is not a chat turn: the CLI, or the detached
    promoter. A human at a terminal is their own observer, so that path passes.
    """
    if not _require_inner_voice():
        return None
    session_id = get_bound_session()
    if not session_id:
        return None

    from app.paths import SESSIONS_DIR
    path = SESSIONS_DIR / f"{session_id}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Fail closed on the observer, not on the round: an unreadable session
        # file should not be able to block self-modification entirely.
        return None
    if data.get("inner_voice"):
        return None

    data["inner_voice"] = True
    data["inner_voice_evaluate_user_turns"] = True
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        enabled = True
    except OSError:
        enabled = False

    return {
        "error": f"inner voice is not attached to this turn — refusing to {action}",
        "inner_voice_enabled_for_next_turn": enabled,
        "why": ("A round rewrites production code, and the observer is what catches "
                "the loop drifting. It attaches at turn start, so switching it on "
                "now cannot cover this turn."),
        "next": ("End your turn and open the round again; the retry runs observed."
                 if enabled else
                 "Could not write the session file — enable Inner Voice on this "
                 "session and retry."),
    }


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
    from scripts.autoimplement import state as S, worktree as W

    gate_path = S.ROUNDS_DIR / round_id / "gate.json"
    if not gate_path.exists():
        return {"error": f"{round_id} has no gate report — run autoimplement_gate first"}
    report = json.loads(gate_path.read_text())
    if not report.get("ok"):
        failed = [r["name"] for r in report.get("rungs", []) if not r["ok"]]
        return {"error": f"gate did not pass (failed: {failed})"}
    if not W.worktree_path(round_id).exists():
        return {"error": f"no worktree for {round_id}"}

    log = S.ROUNDS_DIR / round_id / "land.log"
    python = W.LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
    pid = S.spawn_detached(
        [python, "-m", "scripts.autoimplement.round", "land", round_id],
        log, cwd=W.LIVE_ROOT)
    return {
        "landing": round_id, "pid": pid, "log": str(log),
        "next": "END YOUR TURN NOW. Do not poll, do not call another tool.",
        "note": (
            "Detached, because landing restarts lloyd-mcp and would otherwise kill "
            "the promoter mid-flight. It now waits for the backend to be IDLE — and "
            "your own turn is what is keeping it busy. Polling autoimplement_status in a "
            "loop starves the gate you are waiting on, and after 15 minutes the "
            "landing gives up. Stop talking and the landing proceeds within seconds. "
            "It will restart the backend, which ends this turn anyway. Check "
            "autoimplement_status on your NEXT turn: `current.state` goes landing -> "
            "observing, and the guardian settles it to last-known-good 15 minutes "
            f"after that. Progress is logged to {log}."),
    }


async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="autoimplement_start",
            description=(
                "Open a self-modification round: creates a git worktree off the current "
                "HEAD and returns its path. Edit that path with the normal Edit/Write/Bash "
                "tools, commit inside it, then call autoimplement_gate. Only one round at a time."
            ),
            inputSchema={
                "type": "object",
                "properties": {"goal": {"type": "string",
                                        "description": "What this round is trying to change."}},
                "required": ["goal"],
            },
        ),
        Tool(
            name="autoimplement_gate",
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
                                  "description": "Round id returned by autoimplement_start."},
                    "skip_smoke": {"type": "boolean",
                                   "description": "Skip the live-LLM turn (rung 5). Honoured "
                                                  "ONLY when the model endpoint is actually "
                                                  "unreachable — the gate probes it and runs "
                                                  "the rung anyway if it answers. Either way "
                                                  "the rung is recorded, so a skip is visible "
                                                  "in the promotion record rather than absent "
                                                  "from it."},
                },
                "required": ["round_id"],
            },
        ),
        Tool(
            name="autoimplement_land",
            description=(
                "Promote a round whose gate passed. Returns IMMEDIATELY: the landing "
                "runs detached, because it restarts this very process. It waits for the "
                "backend to go IDLE, fast-forwards, restarts MCP then backend, and "
                "verifies the running commit changed. END YOUR TURN once this returns — "
                "your own turn is what keeps the backend busy, so polling starves the "
                "gate the landing is waiting on. Check autoimplement_status on your next "
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
            name="autoimplement_status",
            description=("Current self-modification state: last known good commit, the "
                         "promotion being observed, halt/broken flags, and recent ledger events."),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="autoimplement_abort",
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
            name="autoimplement_vault_land",
            description=("Land a change to the Obsidian vault (~/obsidian) through the "
                         "self-modification loop. The vault is a live tree with no worktree, "
                         "so this runs AFTER you edit: it validates only the paths you name "
                         "(front matter must parse; a touched skill must still load, a touched "
                         "autonomy task must still parse, SOUL.md/memory/skill edits must still "
                         "build the system prompt), commits exactly those paths on the vault's "
                         "main, and records the sha in the autoimplement ledger. If validation fails "
                         "the paths are REVERTED (tracked files back to HEAD, new files deleted) "
                         "and nothing lands. .obsidian/**, .git/** and .trash/** are denied. "
                         "Requires Inner Voice on the turn, like autoimplement_start."),
            inputSchema={
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"},
                              "description": "Vault-relative paths you changed, e.g. "
                                             "skills/foo/SKILL.md"},
                    "message": {"type": "string", "description": "Commit message"},
                    "item_id": {"type": "integer",
                                "description": "Backlog item this implements, if any"},
                },
                "required": ["paths", "message"],
            },
        ),
        Tool(
            name="autoimplement_vault_revert",
            description=("Revert a vault commit made by autoimplement_vault_land (a plain git "
                         "revert), recorded in the ledger. For a landed vault change that "
                         "turned out wrong."),
            inputSchema={
                "type": "object",
                "properties": {"sha": {"type": "string", "description": "The commit to revert"},
                               "reason": {"type": "string",
                                          "description": "Why; recorded in the ledger"}},
                "required": ["sha"],
            },
        ),
        Tool(
            name="autoimplement_rollback",
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
        from scripts.autoimplement import round as R, state as S

        if name == "autoimplement_status":
            return text_result(json.dumps(R.status(), indent=2, default=str))

        if not _enabled() and name != "autoimplement_abort":
            return text_result(_err(
                "autoimplement.enabled is false in config.yaml. The self-modification loop "
                "ships inert; enabling it is a deliberate human decision."))

        if name == "autoimplement_start":
            goal = (arguments.get("goal") or "").strip()
            if not goal:
                return text_result(_err("goal is required"))
            gate = _inner_voice_gate("open a round")
            if gate:
                return text_result(json.dumps(gate, indent=2))
            return text_result(json.dumps(R.start(goal), indent=2))

        if name == "autoimplement_gate":
            rid = arguments.get("round_id") or ""
            rep = R.run_gate(rid, skip_smoke=bool(arguments.get("skip_smoke")))
            return text_result(json.dumps(rep, indent=2))

        if name == "autoimplement_land":
            rid = arguments.get("round_id") or ""
            if bool(arguments.get("dry_run")):
                return text_result(json.dumps(R.land(rid, dry_run=True), indent=2))
            return text_result(json.dumps(_land_detached(rid), indent=2))

        if name == "autoimplement_abort":
            return text_result(json.dumps(R.abort(arguments.get("round_id") or ""), indent=2))

        if name == "autoimplement_vault_land":
            from scripts.autoimplement import vault_round as VR
            gate = _inner_voice_gate("land a vault change")
            if gate:
                return text_result(json.dumps(gate, indent=2))
            paths = [str(x) for x in (arguments.get("paths") or []) if str(x).strip()]
            try:
                out = VR.land(paths, str(arguments.get("message") or ""),
                              item_id=arguments.get("item_id"))
            except VR.VaultRoundError as exc:
                return text_result(_err(str(exc)))
            out["note"] = ("Committed on the vault's main and live already — nothing "
                           "restarts. autoimplement_vault_revert undoes it.")
            return text_result(json.dumps(out, indent=2))

        if name == "autoimplement_vault_revert":
            from scripts.autoimplement import vault_round as VR
            try:
                out = VR.revert(str(arguments.get("sha") or ""),
                                str(arguments.get("reason") or "manual"))
            except VR.VaultRoundError as exc:
                return text_result(_err(str(exc)))
            return text_result(json.dumps(out, indent=2))

        if name == "autoimplement_rollback":
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
                         "Poll autoimplement_status for the outcome."),
                "request": req,
            }, indent=2))

        return text_result(_err(f"Unknown tool: {name}"))

    try:
        return await asyncio.to_thread(_sync)
    except Exception as exc:
        logger.exception("autoimplement tool %s failed", name)
        return text_result(_err(f"{type(exc).__name__}: {exc}"))
