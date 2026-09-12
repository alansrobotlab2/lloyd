"""
Lloyd prompt builder — assembles system prompts for SDK sessions.

Combines:
- SOUL.md identity
- Memory (MEMORY.md, USER.md from memories/)
- Skills index
- Platform hints and timestamp
"""

import datetime
import logging
import os
from pathlib import Path

logger = logging.getLogger("lloyd.prompt")

# ---------------------------------------------------------------------------
# Prompt-size budget (#466).
#
# Item #110 set ~13K as the target for the identity surface; by 2026-09-08 the
# identity+memory files alone measured ~21.4K tokens and nothing in the code
# could see it — the MEMORY.md clobber (#464) added ~4.4K duplicate tokens per
# turn and sat unnoticed for a day because no number was ever computed. The
# budget is a growth tripwire, deliberately above the size measured the day it
# shipped (65,783 chars: SOUL 6.1K + memories 52.5K + skills index 4K + harness
# paragraphs), so a hit means "something grew", not "you are already over". The
# response to a hit is the component breakdown on the same line, which names the
# component that grew — USER.md at 48K was invisible before this existed.
#
# Nothing is truncated here. Silently cutting an identity file to fit a number
# is the same failure as never measuring it.
#
# The token figure is an estimate at ~4 chars/token, the right order for the
# Qwen tokenizer on this English/Markdown mix. Chars are the exact measurement;
# the estimate is labelled, never presented as a count.
# ---------------------------------------------------------------------------
PROMPT_BUDGET_CHARS = 80_000
CHARS_PER_TOKEN = 4

# A memory file that repeats this many consecutive contract lines has been
# pasted into, not annotated. Two quoted lines in a correction note is normal;
# a verbatim run of 40 is the #464 shape.
_MIN_DUPLICATED_CONTRACT_LINES = 40


def prompt_token_estimate(chars: int) -> int:
    """Rough token count from character count. Label it as an estimate."""
    return max(0, int(chars)) // CHARS_PER_TOKEN


def measure_prompt(components: dict[str, str]) -> dict:
    """Per-component char + estimated-token sizes and the total vs budget.

    `components` is ordered and named by what it is (SOUL.md, MEMORY.md,
    skills_index, …) so a log line or a dashboard can say which half of the
    prompt grew. Nothing is trimmed here — measuring is not enforcing.
    """
    parts = {
        name: {"chars": len(text or ""), "est_tokens": prompt_token_estimate(len(text or ""))}
        for name, text in components.items()
    }
    total = sum(p["chars"] for p in parts.values())
    return {
        "components": parts,
        "total_chars": total,
        "total_est_tokens": prompt_token_estimate(total),
        "budget_chars": PROMPT_BUDGET_CHARS,
        "over_budget": total > PROMPT_BUDGET_CHARS,
    }


def log_prompt_size(
    components: dict[str, str], *, session_id: str = "", platform: str = "",
) -> dict:
    """One INFO line per build: the component breakdown #466 never had.

    `platform=` is on the line because the prompt is no longer the same for
    every turn — a worker turn drops USER.md — and a `memories=` figure with
    no platform beside it cannot be read as either right or wrong.
    """
    report = measure_prompt(components)
    breakdown = "  ".join(
        f"{name}={size['chars']}c/{size['est_tokens']}t"
        for name, size in report["components"].items()
    )
    logger.info(
        "PROMPT_BUDGET session=%s platform=%s %s  total=%dc/%dt  budget=%dc  "
        "over_budget=%s",
        session_id or "-", platform or "user", breakdown, report["total_chars"],
        report["total_est_tokens"], PROMPT_BUDGET_CHARS, report["over_budget"],
    )
    return report


# The anti-compliance rules used to be duplicated here as
# ANTICOMPLIANCE_DIRECTIVE and injected ahead of SOUL.md while SOUL.md carried
# its own near-verbatim copy — two wordings of the same six rules every turn,
# already diverged, and only the vault one reachable by the promotion path
# (#465). They now live in one place: `## Anti-Compliance Directive` in SOUL.md.

LLOYD_HOME = Path(__file__).parent

# Anchor paths to the repo location rather than Path.home() so they resolve
# regardless of who/where the process runs as.
_CANON_SOUL_PATH = LLOYD_HOME.parent / "obsidian" / "lloyd" / "SOUL.md"
_CANON_MEMORIES_DIR = LLOYD_HOME.parent / "obsidian" / "lloyd"
_CANON_SKILLS_DIRS = [
    LLOYD_HOME.parent / "obsidian" / "skills",
    LLOYD_HOME / "skills",
]


# Same quarantine vocabulary the MCP skills module enforces — imported so the
# advertised index and the readable set cannot drift apart.
try:
    from agent_mcp.skills import _QUARANTINE_STATUSES
except Exception:  # pragma: no cover - prompt building must not hard-depend on MCP
    _QUARANTINE_STATUSES = {"inactive", "archived", "disabled", "retired", "quarantined"}


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
    session_id: str = "",
    platform: str = "",
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

    `session_id` — when non-empty, emit one `PROMPT_BUDGET` INFO line holding
    the per-component char/token breakdown for this build (#466). Pass the live
    session id from the chat/ambient/sync handler so there is one line per turn.
    Logging never changes the returned string.

    `platform` — the session's own platform. For a non-user platform
    (`autonomy`, `worker`) the memory files named in
    `harness.worker_prompt.drop_memory_files` are not loaded. USER.md is
    ~20k tokens describing the person Lloyd works for; an unattended round
    is judged against an acceptance contract and nobody reads its reply, so
    that is 20k tokens of a 262k window spent on something structurally
    irrelevant to the work. `sessions_io.NON_USER_PLATFORMS` is the one
    definition of "nobody is reading this", imported lazily so eval and
    bench scripts that call this function stay importable.
    """
    overlay = _resolve_overlay(overlay_dir)
    components: dict[str, str] = {}

    # SOUL.md is the whole identity frame. The anti-compliance rules used to be
    # prepended from a Python constant on top of SOUL.md's own copy, so the same
    # six rules arrived twice per turn in two wordings that had already diverged,
    # and only the vault copy was reachable by the promotion path (#465).
    soul = _load_soul(overlay)
    if soul:
        components["SOUL.md"] = soul

    memories = _load_memories(overlay, soul=soul, files=_memory_files_for(platform))
    if memories:
        components["memories"] = f"<memory>\n{memories}\n</memory>"

    if include_skills_index:
        skills = _load_skills_index(overlay)
        if skills:
            components["skills_index"] = (
                f"<available_skills>\n{skills}\n</available_skills>\n"
                "Note: relevant skill content is automatically injected into each "
                "user message as <context> when matched."
            )

    goal_block = _format_goal_block(goal)
    if goal_block:
        components["goal"] = goal_block

    plan_block = _format_plan_block(plan)
    if plan_block:
        components["plan"] = plan_block

    todos_block = _format_active_todos(todos)
    if todos_block:
        components["todos"] = todos_block

    parts: list[str] = list(components.values())

    # Platform hints — NOTE: no timestamp here; a per-minute timestamp busts
    # vLLM's prefix cache, forcing full re-prefill of the system prompt every turn.
    # The model gets the current time via tool calls or conversation context instead.
    # `platform_hints`, not `platform`: the parameter of that name is the
    # session's platform, and a local shadowing it here printed this whole
    # block into the PROMPT_BUDGET line as though it were the platform. The
    # memory-file decision above happens before this point, so the drop
    # itself was unaffected — which is exactly why this needs a name of its
    # own rather than an ordering rule nobody can see.
    platform_hints = (
        "Platform: Lloyd (Claude Agent SDK). "
        f"Home: {LLOYD_HOME}. "
        "Vault: ~/obsidian/. Knowledge notes go in ~/obsidian/knowledge/. "
        "All persistent notes, research output, and files created by the agent "
        "go in the vault (~/obsidian/), NOT in the lloyd project directory."
    )
    parts.append(platform_hints)

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

    # Web lookups. Bash gets an affordance paragraph directly above, and the
    # model has both it and the http_* tools in the ToolSearch baseline on
    # every request — so without this the only prompt-level guidance about
    # reaching the internet was Bash's. That asymmetry is how curl became the
    # habit (2026-09-04 tool-choice investigation).
    web_lookups = (
        "Web lookups: to reach the public internet use http_search (find pages "
        "by query) and http_fetch (read a page as markdown, links included, so "
        "you can follow them). Use http_request for APIs and non-GET verbs. Do "
        "not shell out to curl or wget to read the web — you would get raw HTML "
        "to strip yourself. Bash and curl remain correct for localhost and "
        "private hosts, which http_fetch blocks by design, and for the "
        "structured API pipelines individual skills document. If a fetched page "
        "comes back near-empty it is JavaScript-rendered: use browser_navigate "
        "then browser_snapshot rather than retrying the fetch."
    )
    parts.append(web_lookups)

    # Code navigation. The graph tools exist and the model still reaches for
    # Grep first, because Grep is what every transcript it has ever seen
    # does — the same asymmetry that made curl the habit above. Named here,
    # unconditionally: this module imports nothing from app.config (like the
    # web_lookups paragraph), and no per-turn value may appear in the system
    # prompt or vLLM re-prefills it every turn.
    code_graph = (
        "Code navigation: before grepping for callers, importers or blast "
        "radius, ask the code graph. graph_explain(symbol, file=) lists who "
        "calls a symbol and what it calls, each with the caller's own call "
        "site; graph_affected(symbol, depth=) is the reverse blast radius "
        "grouped by depth with the file list; graph_path, graph_hubs and "
        "graph_status round it out. It is an AST extraction of one tree, "
        "rebuilt on demand in seconds — accurate for symbols, and blind "
        "across process seams, so keep using Grep for string keys, route "
        "paths, config names and anything crossing HTTP or MCP. Inside a "
        "self-modification round pass root=<worktree path> or you will be "
        "reading about the live checkout instead of the one you are editing."
    )
    parts.append(code_graph)

    turn_discipline = (
        "Turn discipline: never end a turn on an unfulfilled announcement. If you "
        "say \"Let me …\", \"I'll …\", \"Now I'll …\", or end a sentence on a colon "
        "promising a next step, the matching tool call MUST be in that same message "
        "— do not stop and wait. Either do the thing now or don't announce it. When "
        "a task has multiple steps, keep working across iterations until it is "
        "actually complete, then state the result explicitly. Do not hand control "
        "back mid-task with work still pending."
    )
    parts.append(turn_discipline)

    # The four harness paragraphs are grouped as one measured component: they
    # are the instructions prompt_builder itself owns, as distinct from the
    # vault files it reads. Grouping them cannot change the joined string.
    components["harness_hints"] = "\n\n".join(parts[len(components):])
    if session_id:
        log_prompt_size(components, session_id=session_id, platform=platform)
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


def _longest_soul_run(text: str, soul: str) -> tuple[int, int]:
    """Raw line span [start, end) of the longest run of lines shared with `soul`.

    Blank lines do not break a run — the contract is markdown with a blank line
    between every paragraph, so a run measured over raw lines would never be
    longer than one paragraph and a real paste would slip through the guard.
    Returns (-1, -1) when there is no shared line.
    """
    soul_lines = {ln.strip() for ln in soul.split("\n") if ln.strip()}
    lines = text.split("\n")
    nonblank = [(i, ln.strip()) for i, ln in enumerate(lines) if ln.strip()]
    best = (0, 0, (-1, -1))  # (length, span_start_position, raw span)
    start = 0
    for pos in range(len(nonblank) + 1):
        shared = pos < len(nonblank) and nonblank[pos][1] in soul_lines
        if not shared:
            length = pos - start
            if length > best[0]:
                raw_start = nonblank[start][0] if start < len(nonblank) else len(lines)
                raw_end = nonblank[pos][0] if pos < len(nonblank) else len(lines)
                best = (length, start, (raw_start, raw_end))
            start = pos
    return best[2]


def _memory_duplicate_share(text: str, soul: str) -> tuple[float, int, int]:
    """(share of nonblank lines inside the longest shared run, run length, total).

    Longest-run, not total count: a memory file quoting the contract in a dozen
    one-line corrections has runs of 1, while a file pasted over with it has a
    run of the whole contract. That distinction is the difference between a note
    and a clobber.
    """
    start, end = _longest_soul_run(text, soul)
    if start == -1:
        return 0.0, 0, sum(1 for ln in text.split("\n") if ln.strip())
    run = sum(1 for ln in text.split("\n")[start:end] if ln.strip())
    total = sum(1 for ln in text.split("\n") if ln.strip())
    return (run / total if total else 0.0), run, total


def _drop_longest_duplicate_run(text: str, soul: str) -> str:
    start, end = _longest_soul_run(text, soul)
    if start == -1:
        return text
    lines = text.split("\n")
    return "\n".join(lines[:start] + lines[end:]).strip()


def _memory_files_for(platform: str) -> tuple[str, ...]:
    """Which memory files a turn on this platform should carry.

    Fails open in both directions that matter: an unreadable config keeps
    the full set, and an ImportError on `sessions_io` (a bench script with
    no app bootstrap) treats every platform as a user platform. Dropping
    USER.md from a chat turn would be a visible regression; carrying it on
    a worker turn is only a cost.
    """
    default = ("MEMORY.md", "USER.md")
    if not platform:
        return default
    try:
        from app.sessions_io import NON_USER_PLATFORMS
    except Exception:  # noqa: BLE001
        return default
    if platform not in NON_USER_PLATFORMS:
        return default
    try:
        from app.config import CONFIG

        drop = ((CONFIG.get("harness") or {}).get("worker_prompt") or {}).get(
            "drop_memory_files", ["USER.md"])
    except Exception:  # noqa: BLE001
        # The config is the kill switch. Unreadable means we cannot confirm
        # the drop was asked for, so keep everything: a worker turn carrying
        # USER.md costs tokens, and a `drop_memory_files: []` that stopped
        # holding because a yaml read failed would be a switch that does not
        # switch.
        return default
    dropped = {str(d) for d in (drop or [])}
    return tuple(f for f in default if f not in dropped)


def _load_memories(
    overlay: Path | None = None, *, soul: str | None = None,
    files: tuple[str, ...] = ("MEMORY.md", "USER.md"),
) -> str | None:
    """Load the memory files — overlay dir takes priority, canonical as fallback.

    `files` is the list to load, and it is a parameter because an unattended
    turn does not want all of them. USER.md is ~20k tokens of who Alan is,
    which is the single largest component of the system prompt and exactly
    the wrong thing to spend a worker round's window on: the round is judged
    against an acceptance contract, not against the user's preferences, and
    on 2026-09-11 three rounds died at the wall carrying it.

    When `soul` is supplied, a memory file that is a paste of the operating
    contract has the pasted run dropped rather than injected a second time.
    On 2026-09-08 MEMORY.md was overwritten with a verbatim copy of SOUL.md
    (#464): 17.6 KB of the file was the contract, so every turn carried it
    twice, and because nothing compared the two files it took a human audit to
    notice. Dropping the duplicate is safe — the identical text already reached
    the prompt from SOUL.md — but it is loud, because silently rewriting the
    memory surface is the same class of failure as the clobber itself.
    """
    parts = []
    for filename in files:
        content = None
        if overlay and (overlay / filename).exists():
            content = (overlay / filename).read_text(encoding="utf-8").strip()
        elif (_CANON_MEMORIES_DIR / filename).exists():
            content = (_CANON_MEMORIES_DIR / filename).read_text(encoding="utf-8").strip()
        if not content:
            continue
        if soul:
            share, run, total = _memory_duplicate_share(content, soul)
            if run >= _MIN_DUPLICATED_CONTRACT_LINES:
                logger.warning(
                    "PROMPT_BUDGET dropping %d of %d lines from %s (longest run "
                    "verbatim from SOUL.md, %.0f%% of the file) — the operating "
                    "contract was pasted into a memory file and is injected once "
                    "already (#464)", run, total, filename, share * 100,
                )
                content = _drop_longest_duplicate_run(content, soul)
                if not content:
                    logger.warning(
                        "PROMPT_BUDGET %s held nothing but the pasted contract; "
                        "nothing left to inject", filename,
                    )
                    continue
        parts.append(f"## {filename}\n{content}")
    return "\n\n".join(parts) if parts else None


def _is_quarantined_skill(skill_file: Path) -> bool:
    """True if the skill's frontmatter status pulls it from circulation.

    `agent_mcp.skills` already refuses to serve these, so advertising them
    here promised the model a skill that `skills_read` would then decline —
    the same "the prompt names something that does not work" failure as the
    phantom `web_search` tool references (2026-09-04).
    """
    try:
        head = skill_file.read_text(encoding="utf-8", errors="replace")[:2000]
    except OSError:
        return False
    if not head.startswith("---"):
        return False
    fm_end = head.find("\n---", 3)
    fm = head[3:fm_end] if fm_end != -1 else head
    for line in fm.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "status":
            return value.strip().strip("'\"").lower() in _QUARANTINE_STATUSES
    return False


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
            if skill_file.exists() and not _is_quarantined_skill(skill_file):
                skill_names.append(entry.name)
                seen.add(entry.name)
    if not skill_names:
        return None
    return "Available skills: " + ", ".join(skill_names)
