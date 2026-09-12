"""MCP tools for the self-modification loop.

Mirrors `agent_mcp/autoresearch.py`: the loop's real logic lives under
`scripts/automod/`, and this module is the thin surface the agent calls.

The proposal step is deliberately absent. There is no `automod_write_code`
tool — `automod_start` hands back a worktree path and Lloyd edits it with the
ordinary Edit/Write/Bash tools. The round is a wrapper around normal work, not
a special code-generation mode.

Every mutating tool refuses while `automod.enabled` is false, so the machinery
ships inert.
"""

from __future__ import annotations

import json
import logging
import time

from mcp.types import TextContent, Tool

from agent_mcp._shared import get_bound_session, text_result

logger = logging.getLogger("lloyd-mcp.automod")


def _enabled() -> bool:
    try:
        from app.config import CONFIG
        return bool((CONFIG.get("automod") or {}).get("enabled", False))
    except Exception:
        return False


def _require_inner_voice() -> bool:
    """`automod.require_inner_voice` in config.yaml, default true."""
    try:
        from app.config import CONFIG
        return bool((CONFIG.get("automod") or {}).get("require_inner_voice", True))
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

    No bound session means this is not a chat turn: the CLI, or the detached
    promoter. A human at a terminal is their own observer, so that path passes.

    **A worker or autonomy session passes too, since 2026-09-12.** Both
    reasons above were written for a *chat* turn, and neither holds for an
    unattended one. *Live:* the observer's measured effect on rounds was
    negative — #874 abandoned at iteration 38 with 44 minutes left on an
    invented premise, sixteen false repetition fires in a day — and the
    drift it was meant to catch is caught deterministically now by the
    anchors and the gate. *Afterwards:* every background run has been
    recorded since 2026-09-10 (`app/run_recorder.py`) and the Background tab
    lists it, so a round driven from a worker session is exactly as
    reviewable as one driven from an IV session. Refusing here would have
    made `inner_voice: false` on the autocode source a switch that stops
    every round from opening, which is the opposite of a switch.
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
    try:
        from app.sessions_io import NON_USER_PLATFORMS
    except Exception:  # noqa: BLE001
        NON_USER_PLATFORMS = frozenset()
    if str(data.get("platform") or "") in NON_USER_PLATFORMS:
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
    from scripts.automod import state as S, worktree as W

    if S.gate_in_progress(round_id):
        return {"error": f"a gate is still running for {round_id} — automod_gate_wait first"}
    gate_path = S.ROUNDS_DIR / round_id / "gate.json"
    if not gate_path.exists():
        return {"error": f"{round_id} has no gate report — run automod_gate first"}
    report = json.loads(gate_path.read_text())
    if not report.get("ok"):
        failed = [r["name"] for r in report.get("rungs", []) if not r["ok"]]
        return {"error": f"gate did not pass (failed: {failed})"}
    if not W.worktree_path(round_id).exists():
        return {"error": f"no worktree for {round_id}"}

    log = S.ROUNDS_DIR / round_id / "land.log"
    python = W.LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
    pid = S.spawn_detached(
        [python, "-m", "scripts.automod.round", "land", round_id],
        log, cwd=W.LIVE_ROOT)
    return {
        "landing": round_id, "pid": pid, "log": str(log),
        "next": "END YOUR TURN NOW. Do not poll, do not call another tool.",
        "note": (
            "Detached, because landing restarts lloyd-mcp and would otherwise kill "
            "the promoter mid-flight. It now waits for the backend to be IDLE — and "
            "your own turn is what is keeping it busy. Polling automod_status in a "
            "loop starves the gate you are waiting on, and after 15 minutes the "
            "landing gives up. Stop talking and the landing proceeds within seconds. "
            "It will restart the backend, which ends this turn anyway. Check "
            "automod_status on your NEXT turn: `current.state` goes landing -> "
            "observing, and the guardian settles it to last-known-good 15 minutes "
            f"after that. Progress is logged to {log}."),
    }


def _background_tasks_for_session() -> list[dict]:
    """Background Bash children the bound session started and has not seen
    finish. A gate that runs while one is in flight kills it at abort or
    restarts under it at landing — #578 lost its only baseline run that way."""
    sid = get_bound_session()
    if not sid:
        return []
    from agent_mcp import _task_registry
    now = time.time()
    return [{"task_id": r.task_id, "description": r.description,
             "elapsed_s": round(now - r.started_at)}
            for r in _task_registry.list_active() if r.session_id == sid]


def _gate_detached(round_id: str, *, skip_smoke: bool = False) -> dict:
    """Start the gate in its own session and return at once.

    A gate with the review rung runs seven to twelve minutes, and the
    2026-09-11 lesson is that one tool call cannot sit silent on the wire
    that long: the transport's read timeout fired, the pool re-sent the
    request, and the same round was gated twice at once — both review
    attempts spent on one commit the model was never shown a finding for.
    So the gate runs detached, `gate.running` says so, and
    `automod_gate_wait` polls in bounded slices.
    """
    from scripts.automod import state as S, worktree as W

    if not W.worktree_path(round_id).exists():
        return {"error": f"no worktree for {round_id}"}
    if S.gate_in_progress(round_id):
        return _gate_wait(round_id, wait_seconds=0,
                          note="a gate is already running for this round; not started again")
    busy = _background_tasks_for_session()
    if busy:
        return {"error": ("refusing to gate while background task(s) of this session are "
                          "still running — a gate that aborts kills them and a landing "
                          "restarts the backend under them. Wait for them (they report on "
                          "a later iteration) or kill them, then gate."),
                "background_tasks": busy}
    log = S.ROUNDS_DIR / round_id / "gate.log"
    python = W.LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
    argv = [python, "-m", "scripts.automod.round", "gate", round_id]
    if skip_smoke:
        argv.append("--skip-smoke")
    pid = S.spawn_detached(argv, log, cwd=W.LIVE_ROOT)
    S.write_gate_marker(round_id, pid=pid, head=W.head(W.worktree_path(round_id)) or "",
                        by="automod_gate")
    return {
        "gate_started": round_id, "pid": pid, "log": str(log),
        "next": ("Call automod_gate_wait(round_id). It blocks up to four minutes and "
                 "returns the per-rung report when the ladder finishes, or a progress "
                 "line if it has not — call it again then. Do NOT edit, commit or run "
                 "anything in the worktree while the gate runs: the review grades a "
                 "snapshot of the commit you gated, and every commit it refuses spends "
                 "one of the round's two review attempts."),
        "note": ("Detached: with the review rung a gate runs seven to twelve minutes, "
                 "longer than one tool call may stay silent on the wire."),
    }


def _rungs_since(round_id: str, since_ts: float) -> list[str]:
    from scripts.automod import state as S
    rows = []
    for e in S.read_events(limit=400):
        if e.get("round_id") != round_id or float(e.get("ts") or 0) < since_ts - 1:
            continue
        if e.get("event") == "gate":
            rows.append(f"{'PASS' if e.get('ok') else 'FAIL'} {e.get('rung')} "
                        f"({e.get('seconds', 0)}s): {str(e.get('detail') or '')[:160]}")
        elif e.get("event") == "review":
            rows.append(f"review attempt {e.get('attempt')}: "
                        f"{e.get('kind') or e.get('error') or 'ran'}")
    return rows


def _gate_wait(round_id: str, *, wait_seconds: int = 240, note: str = "") -> dict:
    """Block until the detached gate finishes or `wait_seconds` pass.

    Bounded well under the transport's read timeout, so the call itself can
    never be the thing that times out. A marker whose process is gone with a
    `gate.json` newer than it is a finished gate; gone with no newer report
    is a gate that died, and says so.
    """
    from scripts.automod import state as S

    # `0` is a real answer ("just look"), so no `or 240` here.
    wait = max(0, min(int(240 if wait_seconds is None else wait_seconds), 540))
    deadline = time.time() + wait
    gate_path = S.ROUNDS_DIR / round_id / "gate.json"
    log = S.ROUNDS_DIR / round_id / "gate.log"

    def _report() -> dict:
        rep = json.loads(gate_path.read_text(encoding="utf-8"))
        rep["gate_finished"] = True
        # An external blocker is not a verdict on the diff, and the round's
        # right next move is to wait and gate again — but nothing said so, so
        # round 866-a ended its turn on a grader 503 caused by a SIBLING
        # round's landing and was reaped 30 minutes later with the work
        # intact in its worktree.
        failed = next((r for r in (rep.get("rungs") or [])
                       if not r.get("ok")), None) or {}
        data = failed.get("data") or {}
        if data.get("external_blocker"):
            after = int(data.get("retry_after_s") or 120)
            why = str(data.get("external_reason") or "outside your diff")
            rep["next"] = (
                f"This failure is {why} — outside your diff. Wait {after}s and "
                f"call automod_gate again, up to two more times. Do NOT abort "
                f"and do NOT end the turn on this: nothing about your change "
                f"has been judged yet."
            )
            rep["retry_after_s"] = after
        return rep

    while True:
        marker = S.read_gate_marker(round_id)
        if marker is None:
            if gate_path.exists():
                return _report()
            return {"error": f"no gate has been started for {round_id} — call automod_gate first"}
        started = float(marker.get("started_at") or 0)
        if not S.pid_alive(marker.get("pid") or 0):
            S.clear_gate_marker(round_id)
            if gate_path.exists() and gate_path.stat().st_mtime >= started - 1:
                return _report()
            return {"error": ("the gate process died before writing a report — read the "
                              "log, then gate again"), "log": str(log), "marker": marker}
        if time.time() >= deadline:
            out = {"running": True, "round_id": round_id, "pid": marker.get("pid"),
                   "elapsed_s": round(time.time() - started),
                   "rungs_so_far": _rungs_since(round_id, started),
                   "next": "call automod_gate_wait(round_id) again"}
            if note:
                out["note"] = note
            return out
        time.sleep(5)


def _amend_clause(round_id: str, clause, text: str, reason: str) -> dict:
    from scripts.automod import backlog as B, state as S
    import yaml
    spec_path = S.ROUNDS_DIR / round_id / "run_spec.yaml"
    if not spec_path.exists():
        return {"error": f"no run spec for {round_id}"}
    run_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    item_id = (run_spec.get("item") or {}).get("id")
    if not item_id:
        return {"error": f"{round_id} is not bound to a backlog item; there is no contract to amend"}
    try:
        rec = B.amend_clause(int(item_id), clause, text, reason, round_id=round_id)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"amended": rec, "item_id": int(item_id),
            "next": ("Gate again. The next review sees the amendment as an <amendments> "
                     "block and ratifies or refuses it — a refusal restores the old text. "
                     "Nothing needs committing for the amendment itself.")}


async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="automod_start",
            description=(
                "Open a self-modification round: creates a git worktree off the current "
                "HEAD and returns its path. Edit that path with the normal Edit/Write/Bash "
                "tools, commit inside it, then call automod_gate. Only one round at a time."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string",
                             "description": "What this round is trying to change."},
                    "item_id": {"type": "integer",
                                "description": ("The backlog item this round implements. "
                                                "Binds the round to its acceptance clauses, "
                                                "which the gate's review rung grades the diff "
                                                "against. Always pass it for an implement round.")},
                    "from_branch": {"type": "string",
                                    "description": ("Resume from this branch (e.g. automod/SM_…) "
                                                    "when a previous round was sent back by review: "
                                                    "the new worktree starts from it, rebased onto "
                                                    "live main, and the old branch is deleted.")},
                },
                "required": ["goal"],
            },
        ),
        Tool(
            name="automod_gate",
            description=(
                "Start the promotion gate on a round: static checks, the full test suite, "
                "the review rung (a second reader grading the diff against the item's "
                "clauses), a candidate venv if requirements changed, a canary boot on "
                "alternate ports, one real agent turn, and a guardian drill if the diff "
                "touches the rollback path. RETURNS IMMEDIATELY — the gate runs detached "
                "for seven to twelve minutes; call automod_gate_wait for the per-rung "
                "report. Refuses while a gate is already running for the round or while "
                "a background task of yours is in flight."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string",
                                  "description": "Round id returned by automod_start."},
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
            name="automod_gate_wait",
            description=(
                "Wait for the detached gate started by automod_gate. Blocks up to "
                "wait_seconds (default 240) and returns the per-rung report when the "
                "ladder has finished, or {running: true, rungs_so_far: [...]} if it has "
                "not — call again then. Read the failing rung's detail; it names the cause."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string",
                                 "description": "Round id returned by automod_start."},
                    "wait_seconds": {"type": "integer",
                                     "description": "How long to block, 0–540. Default 240."},
                },
                "required": ["round_id"],
            },
        ),
        Tool(
            name="automod_amend_clause",
            description=(
                "Amend ONE acceptance clause the review rung just judged `unsatisfiable` "
                "(no diff could satisfy it as written). Replaces the clause text on the "
                "backlog item and records the amendment as pending; the next review "
                "ratifies it or refuses it and restores the old text. Refused for any "
                "clause the last review did not mark unsatisfiable. Amend to the nearest "
                "clause that is satisfiable and still what the item asked for — never to "
                "something weaker."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string",
                                 "description": "Round id returned by automod_start."},
                    "clause": {"type": "integer", "description": "1-based clause index."},
                    "text": {"type": "string", "description": "The amended clause."},
                    "reason": {"type": "string",
                               "description": "Why the original could not be satisfied."},
                },
                "required": ["round_id", "clause", "text", "reason"],
            },
        ),
        Tool(
            name="automod_land",
            description=(
                "Promote a round whose gate passed. Returns IMMEDIATELY: the landing "
                "runs detached, because it restarts this very process. It waits for the "
                "backend to go IDLE, fast-forwards, restarts MCP then backend, and "
                "verifies the running commit changed. END YOUR TURN once this returns — "
                "your own turn is what keeps the backend busy, so polling starves the "
                "gate the landing is waiting on. Check automod_status on your next "
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
            name="automod_status",
            description=("Current self-modification state: last known good commit, the "
                         "promotion being observed, halt/broken flags, and recent ledger events."),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="automod_abort",
            description=("Abandon a round. Removes its worktree but KEEPS the branch, which "
                         "is the only record of what was attempted."),
            inputSchema={
                "type": "object",
                "properties": {"round_id": {"type": "string",
                                            "description": "Round id to abandon."},
                               "reason": {"type": "string",
                                          "description": ("Why — one line. When the review rung "
                                                          "sent the round back, its findings.")}},
                "required": ["round_id"],
            },
        ),
        Tool(
            name="automod_vault_land",
            description=("Land a change to the Obsidian vault (~/obsidian) through the "
                         "self-modification loop. The vault is a live tree with no worktree, "
                         "so this runs AFTER you edit: it validates only the paths you name "
                         "(front matter must parse; a touched skill must still load, a touched "
                         "autonomy task must still parse, SOUL.md/memory/skill edits must still "
                         "build the system prompt), commits exactly those paths on the vault's "
                         "main, and records the sha in the automod ledger. If validation fails "
                         "the paths are REVERTED (tracked files back to HEAD, new files deleted) "
                         "and nothing lands. .obsidian/**, .git/** and .trash/** are denied. "
                         "Requires Inner Voice on the turn, like automod_start."),
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
            name="automod_vault_revert",
            description=("Revert a vault commit made by automod_vault_land (a plain git "
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
            name="automod_rollback",
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
        from scripts.automod import round as R, state as S

        if name == "automod_status":
            return text_result(json.dumps(R.status(), indent=2, default=str))

        if not _enabled() and name != "automod_abort":
            return text_result(_err(
                "automod.enabled is false in config.yaml. The self-modification loop "
                "ships inert; enabling it is a deliberate human decision."))

        if name == "automod_start":
            goal = (arguments.get("goal") or "").strip()
            if not goal:
                return text_result(_err("goal is required"))
            gate = _inner_voice_gate("open a round")
            if gate:
                return text_result(json.dumps(gate, indent=2))
            item_id = arguments.get("item_id")
            try:
                item_id = int(item_id) if item_id not in (None, "") else None
            except (TypeError, ValueError):
                return text_result(_err(f"item_id must be an integer, got {item_id!r}"))
            from_branch = str(arguments.get("from_branch") or "").strip() or None
            return text_result(json.dumps(
                R.start(goal, item_id=item_id, from_branch=from_branch), indent=2))

        if name == "automod_gate":
            rid = arguments.get("round_id") or ""
            return text_result(json.dumps(
                _gate_detached(rid, skip_smoke=bool(arguments.get("skip_smoke"))), indent=2))

        if name == "automod_gate_wait":
            rid = arguments.get("round_id") or ""
            try:
                wait = int(arguments.get("wait_seconds", 240))
            except (TypeError, ValueError):
                wait = 240
            return text_result(json.dumps(_gate_wait(rid, wait_seconds=wait), indent=2))

        if name == "automod_amend_clause":
            return text_result(json.dumps(_amend_clause(
                str(arguments.get("round_id") or ""), arguments.get("clause"),
                str(arguments.get("text") or ""), str(arguments.get("reason") or "")), indent=2))

        if name == "automod_land":
            rid = arguments.get("round_id") or ""
            if bool(arguments.get("dry_run")):
                return text_result(json.dumps(R.land(rid, dry_run=True), indent=2))
            return text_result(json.dumps(_land_detached(rid), indent=2))

        if name == "automod_abort":
            return text_result(json.dumps(R.abort(arguments.get("round_id") or "",
                                                  reason=str(arguments.get("reason") or "")),
                                          indent=2))

        if name == "automod_vault_land":
            from scripts.automod import review as RV, vault_round as VR
            # The second reader for vault rounds, wired here and not at import
            # in vault_round: tests and the CLI must not reach for the network.
            if VR.GRADER is None:
                VR.GRADER = RV.grade_vault
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
                           "restarts. automod_vault_revert undoes it.")
            return text_result(json.dumps(out, indent=2))

        if name == "automod_vault_revert":
            from scripts.automod import vault_round as VR
            try:
                out = VR.revert(str(arguments.get("sha") or ""),
                                str(arguments.get("reason") or "manual"))
            except VR.VaultRoundError as exc:
                return text_result(_err(str(exc)))
            return text_result(json.dumps(out, indent=2))

        if name == "automod_rollback":
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
                         "Poll automod_status for the outcome."),
                "request": req,
            }, indent=2))

        return text_result(_err(f"Unknown tool: {name}"))

    try:
        return await asyncio.to_thread(_sync)
    except Exception as exc:
        logger.exception("automod tool %s failed", name)
        return text_result(_err(f"{type(exc).__name__}: {exc}"))
