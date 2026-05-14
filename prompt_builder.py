"""
Lloyd prompt builder — assembles system prompts for SDK sessions.

Combines:
- SOUL.md identity
- Memory (MEMORY.md, USER.md from memories/)
- Skills index
- Platform hints and timestamp
"""

import datetime
import os
from pathlib import Path

LLOYD_HOME = Path(__file__).parent
# Use LLOYD_HOME instead of Path.home() to avoid distrobox path mismatch issues
# where Path.home() resolves to the host home instead of the container's isolated home
_CANON_SOUL_PATH = LLOYD_HOME.parent / "obsidian" / "lloyd" / "SOUL.md"
_CANON_MEMORIES_DIR = LLOYD_HOME.parent / "obsidian" / "lloyd"
_CANON_SKILLS_DIRS = [
    LLOYD_HOME.parent / "obsidian" / "skills",
    LLOYD_HOME / "skills",
]


def _resolve_overlay(overlay_dir: str | Path | None) -> Path | None:
    """Normalize an overlay-dir argument (falls back to LLOYD_OVERLAY_DIR env var)."""
    raw = overlay_dir if overlay_dir is not None else os.environ.get("LLOYD_OVERLAY_DIR")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path if path.exists() else None


# Back-compat public names (some callers import these directly)
SOUL_PATH = _CANON_SOUL_PATH
MEMORIES_DIR = _CANON_MEMORIES_DIR
SKILLS_DIRS = _CANON_SKILLS_DIRS


def _format_active_todos(todos: list[dict] | None) -> str | None:
    """Render the persisted todo list as a compact prompt block.

    Anchors primary across turns even after compaction shrinks the message
    history. Cache-friendly: the block changes only when the todo list
    changes, not per-turn. Returns None when there are no todos.
    """
    if not todos:
        return None
    lines = ["<active_todos>", "Current progress (kept here so you stay anchored across turns):"]
    for t in todos:
        status = t.get("status", "?")
        content = (t.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"  - [{status}] {content}")
    if len(lines) == 2:
        return None
    lines.append("</active_todos>")
    return "\n".join(lines)


def _format_plan_block(plan: dict | None) -> str | None:
    """Render the session's plan state (Plan B).

    Two render modes:
      * `plan_mode=True`: emit a `<plan_mode_active>` banner reminding
        primary that write tools are blocked and pointing at the
        plan-mode-authoring skill (Phase B2 will inject that as
        `<context>` automatically; for B1 we just point at it by name).
      * Committed plan exists (`plan_md_path` set, `plan_mode=false`):
        load the markdown body and emit a `<plan>` block. Plan body
        changes only when the plan is re-committed, so cache stays
        warm across turns.
      * Otherwise: no block.
    """
    if not plan:
        return None
    if plan.get("plan_mode"):
        stage_summary = ""
        stages = plan.get("stages") or []
        if stages:
            lines = [f"  Stage {s.get('n','?')}: {s.get('title','')}" for s in stages if isinstance(s, dict)]
            stage_summary = "\nStages drafted so far:\n" + "\n".join(lines)
        return (
            "<plan_mode_active>\n"
            "You are in research-only PLAN MODE. Write, Edit, and Bash are "
            "blocked until you commit. Do not try to call them — they will "
            "return errors. You retain Read, Grep, Glob, skills_search, "
            "skills_read, and ToolSearch for research.\n\n"
            "DO NOT call TodoWrite directly while in plan mode. The proper "
            "commit path is `ExitPlanMode`, which atomically writes the "
            "plan markdown to the vault AND replaces session.todos with "
            "your decomposed list. Calling TodoWrite alone leaves the "
            "session stuck in plan_mode with no committed plan.\n\n"
            "Your workflow this turn:\n"
            "  1. Research the request — read relevant files, skills, prior context.\n"
            "  2. If the goal is ambiguous, ask the user clarifying questions "
            "(they answer in their next turn).\n"
            "  3. Draft a structured plan as markdown (goal, approach, stages, acceptance criteria).\n"
            "  4. Decompose into todos: one or more per stage, in execution order, "
            "with the first todo as `in_progress`.\n"
            "  5. Call `ExitPlanMode` with `plan_md`, `stages`, and `todos` to commit. "
            "If research shows the request was simpler than it looked, call "
            "`ExitPlanMode(cancel=true)` instead.\n"
            "  6. After ExitPlanMode succeeds, STOP this turn with a brief "
            "summary. Write/Edit/Bash stay blocked for the rest of this "
            "turn (harness pins disallowed_tools at turn start). The user's "
            "next turn rebuilds options, unblocks actuator tools, and "
            "surfaces the committed plan via your system prompt. Execution "
            "begins on that next turn, not this one."
            f"{stage_summary}\n"
            "</plan_mode_active>"
        )
    plan_path = plan.get("plan_md_path")
    if not plan_path:
        return None
    try:
        body = Path(plan_path).read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not body:
        return None
    stages = plan.get("stages") or []
    stage_lines = ""
    if stages:
        items = [f"  {s.get('n','?')}. {s.get('title','')}" for s in stages if isinstance(s, dict)]
        stage_lines = "\n\nSTAGES:\n" + "\n".join(items)
    return (
        "<plan>\n"
        f"PLAN COMMITTED at {plan.get('committed_at', '?')}.\n\n"
        f"{body}"
        f"{stage_lines}\n\n"
        "Stay focused on this plan. Use TodoWrite to track progress. Do "
        "not silently abandon stages — if you need to revise the plan, "
        "ask the user (only a fresh /plan can rewrite it)."
        "\n</plan>"
    )


def _format_goal_block(goal: dict | None) -> str | None:
    """Render the session's persistent goal (the /goal target).

    Anchors primary on the user's stated end condition across turns. The
    block is short and cache-friendly: changes only when the user
    sets/clears/achieves the goal. Returns None when no goal is set or
    the goal text is empty.
    """
    if not goal:
        return None
    text = (goal.get("text") or "").strip()
    if not text:
        return None
    achieved = goal.get("achieved_at")
    if achieved:
        return (
            "<goal achieved>\n"
            f"GOAL ACHIEVED at {achieved}: {text}\n"
            "The inner voice judged this goal met. Continue with whatever "
            "the user asks next; you do not need to keep working toward it."
            "\n</goal>"
        )
    attempts = int(goal.get("attempts") or 0)
    attempt_line = f"\nAttempts so far: {attempts}." if attempts else ""
    return (
        "<goal>\n"
        f"PERSISTENT GOAL (the user's /goal — keep working toward this until met):\n"
        f"{text}{attempt_line}\n\n"
        "After each turn the inner voice evaluates whether this goal is "
        "met. If not, it queues a follow-up turn with a short reason. "
        "Stay focused — every turn should advance the goal or surface a "
        "concrete blocker. If you cannot make progress, say so plainly "
        "and stop rather than padding."
        "\n</goal>"
    )


def build_system_prompt(
    include_skills_index: bool = True,
    overlay_dir: str | Path | None = None,
    todos: list[dict] | None = None,
    plan: dict | None = None,
    goal: dict | None = None,
) -> str:
    """Build the full system prompt for a Lloyd session.

    `overlay_dir` (or the LLOYD_OVERLAY_DIR env var as fallback) redirects
    SOUL.md / MEMORY.md / USER.md / skills/ reads to a variant sandbox, with
    fallthrough to the canonical vault for any file the overlay doesn't
    provide. Thread-safe when callers pass `overlay_dir` explicitly.

    `todos` (Plan A) — when non-empty, an `<active_todos>` block is appended
    so primary stays anchored to its committed plan even after compaction.
    Pass the list straight from `session.todos`.

    `plan` (Plan B) — when `plan_mode=True`, render a `<plan_mode_active>`
    banner; when a committed plan exists (`plan_md_path` set), render a
    `<plan>` block with the markdown body. Pass `session.plan` straight
    from disk.

    `goal` (the /goal target) — session-level persistent goal. When set,
    renders a `<goal>` block above plan + todos so primary stays anchored
    on the user's end condition. Pass `session.goal` straight from disk.
    """
    overlay = _resolve_overlay(overlay_dir)
    parts = []

    soul = _load_soul(overlay)
    if soul:
        parts.append(soul)

    memory = _load_memories(overlay)
    if memory:
        parts.append(f"<memory>\n{memory}\n</memory>")

    if include_skills_index:
        skills = _load_skills_index(overlay)
        if skills:
            parts.append(f"<available_skills>\n{skills}\n</available_skills>\nNote: relevant skill content is automatically injected into each user message as <context> when matched.")

    goal_block = _format_goal_block(goal)
    if goal_block:
        parts.append(goal_block)

    plan_block = _format_plan_block(plan)
    if plan_block:
        parts.append(plan_block)

    todos_block = _format_active_todos(todos)
    if todos_block:
        parts.append(todos_block)

    # Platform hints — NOTE: no timestamp here; a per-minute timestamp busts
    # vLLM's prefix cache, forcing full re-prefill of the system prompt every turn.
    # The model gets the current time via tool calls or conversation context instead.
    platform = (
        "Platform: Lloyd (Claude Agent SDK). "
        f"Home: {LLOYD_HOME}. "
        "Vault: ~/obsidian/. Knowledge notes go in ~/obsidian/knowledge/. "
        "All persistent notes, research output, and files created by the agent "
        "go in the vault (~/obsidian/), NOT in the lloyd project directory."
    )
    parts.append(platform)

    bg_tasks = (
        "Background bash tasks: pass run_in_background=true to Bash for any "
        "long-running command (builds, finds, deploys, monitoring loops). The "
        "tool returns a task_id and an output_file path immediately so you can "
        "keep working. When the command exits, a <task_notification> message "
        "appears in the conversation on a later turn carrying the same task_id, "
        "the final status, and the output_file path. Use Read on the output_file "
        "to inspect what the command produced. Do not respond directly to a "
        "<task_notification> unless its result changes your plan."
    )
    parts.append(bg_tasks)

    return "\n\n".join(parts)


def _load_soul(overlay: Path | None = None) -> str | None:
    """Load SOUL.md content, stripping any YAML frontmatter."""
    path = overlay / "SOUL.md" if (overlay and (overlay / "SOUL.md").exists()) else _CANON_SOUL_PATH
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    if content.startswith("---"):
        end = content.find("\n---\n", 3)
        if end != -1:
            content = content[end + 5:].strip()
    return content or None


def _load_memories(overlay: Path | None = None) -> str | None:
    """Load MEMORY.md and USER.md — overlay dir takes priority, canonical as fallback."""
    parts = []
    for filename in ("MEMORY.md", "USER.md"):
        content = None
        if overlay and (overlay / filename).exists():
            content = (overlay / filename).read_text(encoding="utf-8").strip()
        elif (_CANON_MEMORIES_DIR / filename).exists():
            content = (_CANON_MEMORIES_DIR / filename).read_text(encoding="utf-8").strip()
        if content:
            parts.append(f"## {filename}\n{content}")
    return "\n\n".join(parts) if parts else None


def _load_skills_index(overlay: Path | None = None) -> str | None:
    """Build a list of available skill names from skill directories (overlay first)."""
    dirs: list[Path] = []
    if overlay and (overlay / "skills").exists():
        dirs.append(overlay / "skills")
    dirs.extend(_CANON_SKILLS_DIRS)

    skill_names: list[str] = []
    seen: set[str] = set()
    for skills_dir in dirs:
        if not skills_dir.exists():
            continue
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name in seen:
                continue
            skill_file = entry / "SKILL.md"
            if skill_file.exists():
                skill_names.append(entry.name)
                seen.add(entry.name)
    if not skill_names:
        return None
    return "Available skills: " + ", ".join(skill_names)
