#!/usr/bin/env python3
"""Lloyd MCP Server: Plan Mode (Plan B).

Two tools:

* **EnterPlanMode** — flips `session.plan.plan_mode = true` and tells
  primary to research before drafting. No args. Used as the auto-entry
  path; the slash command `/plan` flips the same flag from the server side.
* **ExitPlanMode** — commits a structured plan (or cancels) and atomically
  replaces `session.todos` so Plan A's stewardship machinery picks up the
  decomposed steps. Two paths:

  - `cancel=true`: just flip `plan_mode=false`, no plan written, todos untouched.
    Used when primary realizes mid-research that the request didn't actually
    need a plan.
  - commit (default): write `plan_md` to the vault at
    `~/obsidian/plans/<session_id>.md`, store stages in `session.plan`,
    replace `session.todos` (using the shared `validate_todos` helper that
    TodoWrite also uses), flip `plan_mode=false`. This is the bridge from
    Plan B → Plan A.

Plan markdown lives in the vault, not in the session JSON, so the user
can edit/diff/grep it as a real file. Session JSON holds `plan_md_path`
and the stages.

Session correlation: same pattern as builtin_todo — read
`_shared.get_bound_session()` — the contextvar bound by the
aggregator at dispatch time.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from agent_mcp._shared import get_bound_session
from agent_mcp._todo_validation import validate_todos
from app.sessions_io import mutate_session

logger = logging.getLogger("lloyd-builtin-plan")

app = Server("lloyd-builtin-plan")


# Vault location for plan markdown files. Resolved relative to the
# Lloyd repo root via the same convention as prompt_builder.py
# (LLOYD_HOME.parent / "obsidian"). Created on first use.
_LLOYD_HOME = Path(__file__).resolve().parent.parent
_PLANS_DIR = _LLOYD_HOME.parent / "obsidian" / "plans"


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def _validate_stages(raw: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate a stages payload. Returns (list, None) on success."""
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return None, "stages must be an array"
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return None, f"stages[{i}] must be an object"
        n = item.get("n")
        title = item.get("title")
        summary = item.get("summary", "")
        if not isinstance(n, int) or n < 1:
            return None, f"stages[{i}].n must be a positive integer"
        if not isinstance(title, str) or not title.strip():
            return None, f"stages[{i}].title must be a non-empty string"
        if not isinstance(summary, str):
            return None, f"stages[{i}].summary must be a string"
        out.append({"n": n, "title": title.strip(), "summary": summary.strip()})
    return out, None


async def _enter_plan_mode(_args: dict[str, Any]) -> str:
    session_id = get_bound_session()
    if not session_id:
        return json.dumps({
            "error": "EnterPlanMode called outside a session context — _session_id not bound",
        })

    def _apply(data: dict[str, Any]) -> None:
        existing = data.get("plan") or {}
        # Preserve plan_md_path if a prior plan already committed; flipping
        # plan_mode=true means primary is drafting a NEW plan (the user
        # re-/plan'd or auto-entry fired again). The existing plan stays
        # readable in the prompt until the new ExitPlanMode commits.
        data["plan"] = {
            "plan_mode": True,
            "plan_md_path": existing.get("plan_md_path", ""),
            "stages": existing.get("stages", []),
            "created_at": existing.get("created_at") or _now_iso(),
            "drafted_at": _now_iso(),
        }

    ok = await mutate_session(session_id, _apply)
    if not ok:
        return json.dumps({"error": f"Session {session_id} not found"})
    logger.info("[plan] EnterPlanMode session=%s", session_id)
    return (
        "Plan mode activated. Write tools (Write, Edit, Bash) are blocked "
        "until you commit. Research the request — read relevant files, "
        "skills, and prior context. Ask the user clarifying questions if "
        "the goal is ambiguous. Then call ExitPlanMode with: plan_md "
        "(structured markdown plan), stages (list of {n, title, summary}), "
        "and todos (the same shape TodoWrite takes — content, status, "
        "activeForm). Or call ExitPlanMode(cancel=true) if research shows "
        "the request was simpler than it looked."
    )


async def _exit_plan_mode(args: dict[str, Any]) -> str:
    session_id = get_bound_session()
    if not session_id:
        return json.dumps({
            "error": "ExitPlanMode called outside a session context — _session_id not bound",
        })

    cancel = bool(args.get("cancel", False))

    if cancel:
        # Cancel path: flip plan_mode=false, leave todos and any prior
        # committed plan untouched. The prior plan_md_path stays so the
        # next turn's system prompt still surfaces it (cancellation only
        # backs out of the *current* drafting session, not the prior plan).
        def _cancel_apply(data: dict[str, Any]) -> None:
            existing = data.get("plan") or {}
            data["plan"] = {
                **existing,
                "plan_mode": False,
                "cancelled_at": _now_iso(),
            }

        ok = await mutate_session(session_id, _cancel_apply)
        if not ok:
            return json.dumps({"error": f"Session {session_id} not found"})
        logger.info("[plan] ExitPlanMode(cancel) session=%s", session_id)
        return "Plan mode cancelled. Resume normal execution."

    # Commit path: validate inputs, write plan markdown to vault,
    # update session.plan + session.todos in one atomic mutate.
    plan_md = args.get("plan_md")
    if not isinstance(plan_md, str) or not plan_md.strip():
        return json.dumps({
            "error": "plan_md (string) is required when cancel=false",
        })
    stages, stage_err = _validate_stages(args.get("stages"))
    if stage_err is not None:
        return json.dumps({"error": stage_err})
    todos, todo_err = validate_todos(args.get("todos") or [])
    if todo_err is not None:
        return json.dumps({"error": todo_err})

    # Write the plan markdown to the vault. One canonical file per
    # session: `~/obsidian/plans/<session_id>.md`. Re-planning overwrites
    # the prior file (history beyond "current" is out of scope for B1).
    try:
        _PLANS_DIR.mkdir(parents=True, exist_ok=True)
        plan_path = _PLANS_DIR / f"{session_id}.md"
        plan_path.write_text(plan_md.strip() + "\n", encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"failed to write plan markdown: {e}"})

    def _commit_apply(data: dict[str, Any]) -> None:
        existing = data.get("plan") or {}
        data["plan"] = {
            "plan_mode": False,
            "plan_md_path": str(plan_path),
            "stages": stages,
            "created_at": existing.get("created_at") or _now_iso(),
            "committed_at": _now_iso(),
        }
        # Replace todos wholesale — same semantics as TodoWrite. This is
        # the explicit handoff to Plan A's stewardship machinery.
        data["todos"] = todos or []

    ok = await mutate_session(session_id, _commit_apply)
    if not ok:
        # Best effort: don't try to clean up the markdown file — the
        # mutate_session failure is rare and we'd rather leave a stale
        # plan file than crash recovery.
        return json.dumps({"error": f"Session {session_id} not found"})

    logger.info(
        "[plan] ExitPlanMode(commit) session=%s stages=%d todos=%d path=%s",
        session_id, len(stages), len(todos or []), plan_path,
    )
    return (
        f"Plan committed. {len(stages)} stages, {len(todos or [])} todos. "
        f"Markdown saved to {plan_path}.\n\n"
        f"IMPORTANT: stop this turn now. Write, Edit, and Bash remain "
        f"blocked for the rest of THIS turn (the harness pinned the "
        f"disallowed-tools list at turn start). Output a brief "
        f"acknowledgement summarizing the committed plan and stop. "
        f"On the user's next turn the actuator tools will be unblocked, "
        f"the committed plan will appear in your system prompt, and "
        f"TodoWrite will govern execution — advance each todo from "
        f"pending → in_progress → completed as you work the stages."
    )


_ENTER_DESC = """Enter plan mode — research-only mode for drafting a structured plan before execution.

## When to use
Call this when the user's request is genuinely multi-stage and would benefit from explicit planning before action: refactors that touch multiple files, features that span backend + frontend, investigations that branch across multiple sub-questions, or any task where the user's intent is unclear enough that you'd want to ask clarifying questions first.

## What it does
- Flips the session into plan mode: write tools (Write, Edit, Bash) are blocked until you commit.
- You retain read tools (Read, Grep, Glob, skills_search/skills_read), TodoWrite, and ToolSearch.
- Prepends a system reminder explaining the constraints.

## Workflow once in plan mode
1. Read relevant files, skills, and any prior session context.
2. If the goal is ambiguous, ask the user clarifying questions (the user can answer in their next turn — you'll see the answer when you're called next).
3. Draft a structured plan: a short markdown document describing goal, approach, stages, and acceptance criteria.
4. Decompose into todos: one or more per stage, in execution order.
5. Call ExitPlanMode with `plan_md` (the markdown), `stages` (the stage structure), and `todos` (the decomposed list).

## When not to use
- Single-step requests ("what time is it?", "show me file X", "run this query").
- Tasks where the skill protocol already prescribes the workflow.
- Tasks the user has already broken down — just call TodoWrite directly.

If during research you realize the task wasn't actually plan-worthy, call ExitPlanMode(cancel=true) and proceed with normal execution."""


_EXIT_DESC = """Exit plan mode — either commit a structured plan + todos, or cancel without committing.

## Commit path (default)
Required args: `plan_md` (the markdown plan body) and `todos` (the decomposed list, same shape as TodoWrite). Optional: `stages` (a list of `{n, title, summary}` for stage grouping).

What happens on commit:
- Plan markdown is written to `~/obsidian/plans/<session_id>.md`.
- `session.plan` is populated with `plan_md_path`, `stages`, `committed_at`.
- `session.todos` is REPLACED with the decomposed list — Plan A's TodoWrite stewardship governs execution from here on.
- Plan mode flag flips off; write tools are unblocked.

## Cancel path
Pass `cancel=true` (other args ignored). Used when:
- Research showed the request was simpler than it looked (a one-line change).
- The user changed direction during the research turn.
- You're stuck and want to abandon the plan ritual.

Cancel does NOT modify the prior committed plan (if one exists) — it only backs out of the current drafting session.

## Todo shape (same as TodoWrite)
Each todo: `{content: str, status: "pending"|"in_progress"|"completed", activeForm: str}`. Optional `stage: int` to associate with a stage from `stages`. Default first-stage todo to `in_progress` if you want to start work immediately on the next turn."""


@app.list_tools()
async def list_tools():
    # Note on `additionalProperties`: the top-level object schema does NOT
    # set additionalProperties=false, because the harness injects a
    # `_session_id` field into every tool call's args for session
    # correlation (stripped in agent_mcp/main.py before dispatch). A
    # strict top-level schema would reject `_session_id` and break the
    # whole plan-mode flow. Nested schemas (stages.items, todos.items)
    # keep additionalProperties=false because they're validated by the
    # model directly, not the harness's correlation plumbing.
    return [
        Tool(
            name="EnterPlanMode",
            description=_ENTER_DESC,
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="ExitPlanMode",
            description=_EXIT_DESC,
            inputSchema={
                "type": "object",
                "properties": {
                    "cancel": {
                        "type": "boolean",
                        "description": "True to abandon plan mode without committing.",
                    },
                    "plan_md": {
                        "type": "string",
                        "description": (
                            "Markdown body of the plan. Required when cancel "
                            "is false. Suggested structure: # Goal, ## "
                            "Approach, ## Stages, ## Acceptance criteria."
                        ),
                    },
                    "stages": {
                        "type": "array",
                        "description": (
                            "Optional stage structure. Each stage: "
                            "{n: int (1-indexed), title: str, summary: str}."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "n": {"type": "integer", "minimum": 1},
                                "title": {"type": "string", "minLength": 1},
                                "summary": {"type": "string"},
                            },
                            "required": ["n", "title"],
                            "additionalProperties": False,
                        },
                    },
                    "todos": {
                        "type": "array",
                        "description": (
                            "Decomposed todo list (same shape as TodoWrite). "
                            "Replaces session.todos on commit."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "minLength": 1},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                                "activeForm": {"type": "string", "minLength": 1},
                                "stage": {"type": "integer", "minimum": 1},
                            },
                            "required": ["content", "status", "activeForm"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "EnterPlanMode":
        text = await _enter_plan_mode(arguments)
    elif name == "ExitPlanMode":
        text = await _exit_plan_mode(arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return [TextContent(type="text", text=text)]
