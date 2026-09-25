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
# The roots and the walk both come from `agent_mcp.skills` now (#1294). This pair
# used to be built from `LLOYD_HOME`, which is the *checkout's* location: inside an
# automod worktree it named `<worktree>/home/obsidian/skills`, a directory that has
# never existed, so a prompt built in a round advertised no skills at all while
# `GET /api/skills` listed 187 of the live vault and the Mission Control tab
# counted 194 of it — and `tests/test_archived_skill_artifacts.py:52-60` had to
# document that the prompt-side loader "would read nothing and pass". The walker's
# roots are anchored where the vault actually is, so the advertised index and every
# other surface are the same walk of the same directories.
#
# Same reason the quarantine vocabulary is imported rather than restated: the
# fallback literal that used to sit here was a second definition of the rule, kept
# honest only by one equality assertion in one test.
#
# This import used to be wrapped in `try/except Exception` so prompt building could
# survive `agent_mcp` being unimportable. The wrapper is gone because surviving it
# meant *silently advertising a different set of skills*: the fallback re-derived
# the roots from `LLOYD_HOME` and re-spelled the quarantine set, and nothing could
# see that it had taken over. `prefetch.py:29` has imported this module
# unconditionally all along, and every runtime path that builds a prompt imports
# `prefetch`, so the fallback protected a path that does not exist.
from agent_mcp import skills as _skills_module
from agent_mcp.skills import iter_active_skills, is_quarantined_skill_file

#: Patch point for the roots the advertised index walks; `None` means "whatever
#: `agent_mcp.skills.SKILLS_DIRS` is right now", which is the only arrangement that
#: keeps one definition (#1294) — and it is read per call, never copied at import,
#: so a caller that redirects the walker's roots redirects the index with it.
#: Tests that want a private pair set this; production leaves it `None`.
_CANON_SKILLS_DIRS: list[Path] | None = None


def _skill_walk_roots() -> list[Path]:
    """The roots to advertise from, read at call time rather than bound at import."""
    if _CANON_SKILLS_DIRS is not None:
        return list(_CANON_SKILLS_DIRS)
    return list(_skills_module.SKILLS_DIRS)


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


#: Where a turn's session state (goal, plan, todos) is rendered — P1.
#:   system_head — inside the system prompt, ahead of the harness hints
#:                 (today's layout, byte-identical).
#:   system_tail — inside the system prompt, after everything static, as one
#:                 `<session_state>` block. A todo edit then invalidates only
#:                 the tail of the system prompt, not the hints under it.
#:   user_tail   — not in the system prompt at all; the caller appends it to
#:                 the turn's user message (`app/prompt_layout.py`), so the
#:                 system prompt is byte-stable across state changes and the
#:                 whole previous conversation stays prefix-cached.
SESSION_STATE_LAYOUTS = ("system_head", "system_tail", "user_tail")


def session_state_layout() -> str:
    """`harness.prompt_layout.session_state`, validated. Never raises.

    Read lazily, like `_memory_files_for`: this module is imported by CLI
    scripts that never bring up the app package, and anything unreadable or
    unknown is today's layout — a layout nobody asked for is the one outcome
    worse than the old cache behaviour.
    """
    try:
        from app.config import CONFIG

        value = (((CONFIG.get("harness") or {}).get("prompt_layout") or {})
                 .get("session_state", "system_head"))
    except Exception:  # noqa: BLE001
        return "system_head"
    value = str(value or "system_head").strip()
    if value not in SESSION_STATE_LAYOUTS:
        logger.warning("harness.prompt_layout.session_state=%r is not one of %s; "
                       "using system_head", value, SESSION_STATE_LAYOUTS)
        return "system_head"
    return value


def build_session_state_block(
    todos: list[dict] | None = None,
    plan: dict | None = None,
    goal: dict | None = None,
) -> str | None:
    """The session's mutable state as one `<session_state>` block, or None.

    The same three renderers `system_head` uses, in the same order (goal,
    plan, todos), wrapped so the block can move as a unit to the tail of the
    system prompt or of the user message (P1).
    """
    blocks = [b for b in (_format_goal_block(goal), _format_plan_block(plan),
                          _format_active_todos(todos)) if b]
    if not blocks:
        return None
    return "<session_state>\n" + "\n\n".join(blocks) + "\n</session_state>"


def build_system_prompt(
    include_skills_index: bool = True,
    overlay_dir: str | Path | None = None,
    todos: list[dict] | None = None,
    plan: dict | None = None,
    goal: dict | None = None,
    session_id: str = "",
    platform: str = "",
    memories_text: str | None = None,
    session_state: str | None = None,
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

    `memories_text` — when not None, used verbatim as the memory body instead
    of reading MEMORY.md/USER.md (`""` means no memory block). This is how a
    session's frozen snapshot (`app/memory_snapshot.py`) reaches the prompt.

    `session_state` — the layout (`SESSION_STATE_LAYOUTS`); None reads
    `harness.prompt_layout.session_state`. `system_head` is today's output
    byte for byte; `user_tail` leaves the state out entirely.
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

    if memories_text is None:
        memories = _load_memories(overlay, soul=soul, files=_memory_files_for(platform))
    else:
        memories = memories_text or None
    if memories:
        components["memories"] = f"<memory>\n{memories}\n</memory>"

    if include_skills_index:
        skills = _load_skills_index(overlay)
        if skills:
            components["skills_index"] = (
                f"<available_skills>\n{skills}\n</available_skills>\n"
                + _skills_index_note(skills_push_enabled())
            )

    layout = session_state if session_state in SESSION_STATE_LAYOUTS \
        else session_state_layout()
    if layout == "system_head":
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
        "Platform: Lloyd (local harness). "
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

    # P9: only while `harness.rpc.enabled` — off, the prompt is byte for byte
    # what it was. A constant paragraph (the path is LLOYD_HOME's), so turning
    # it on costs one re-prefill, not one per turn.
    if rpc_hint_enabled():
        parts.append(_rpc_hint())

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
    # P1: the mutable state goes last, so a todo/plan/goal change moves only
    # the bytes after every static paragraph. In `user_tail` it is not here
    # at all — the caller appends it to the user message.
    if layout == "system_tail":
        state_block = build_session_state_block(todos, plan, goal)
        if state_block:
            components["session_state"] = state_block
            parts.append(state_block)
    if session_id:
        log_prompt_size(components, session_id=session_id, platform=platform)
        # #581: `log_prompt_size` reports each component's SIZE and then drops
        # the dict. The manifest needs the same named parts to hash them, and
        # this is the only place the complete dict exists — so this is the
        # handoff. `app/component_manifest.py` keeps it keyed by session id and
        # `app/harness/client.py` reads it back from inside the agent loop: one
        # module writing, another reading, no call between them. Imported here
        # rather than at module scope for the same reason `_load_non_user_platforms`
        # imports `app.sessions_io` inside its body — this module is imported by
        # CLI scripts that never bring up the app package. Never raises.
        from app.component_manifest import note_components
        note_components(session_id, components)
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
        raw = None
        if overlay and (overlay / filename).exists():
            raw = (overlay / filename).read_text(encoding="utf-8")
        elif (_CANON_MEMORIES_DIR / filename).exists():
            raw = (_CANON_MEMORIES_DIR / filename).read_text(encoding="utf-8")
        content = raw.strip() if raw is not None else None
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
        content = _bound_memory_render(filename, raw, content)
        parts.append(f"## {filename}\n{content}")
    return "\n\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Render-time overflow (review 2026-09-24, P4).
#
# The write-time ceiling (`app.memory_ceiling`) refuses every tool that would
# grow a loaded memory file past `prompt_surface.MEMORY_CEILINGS`, but a file can
# still arrive over it: a Bash heredoc (outside every tool handler by design), a
# vault sync, or the ceiling itself being lowered under a file that has not been
# consolidated yet — which is exactly the deploy step P4 ends in. This is what
# the prompt does with such a file.
#
# `memory.render_overflow` in config.yaml:
#   render_all  — inject the whole file, as before this existed (logged once a
#                 day). The DEFAULT until `eval/run_memory_index_ab.py` promotes
#                 the index, so production prompts stay byte-identical.
#   annotate    — cut at the last entry boundary under the ceiling and append a
#                 `<memory_overflow>` marker naming the file and the dropped
#                 bytes, so the model knows the tail exists and how to read it;
#                 `logger.error` and one guardian `announce()` a day per file.
#
# No second number: the limit is the file's own `prompt_surface` ceiling, the
# one every writer is refused at (#1010's defect was two of them).
# ---------------------------------------------------------------------------
_OVERFLOW_MODES = ("annotate", "render_all")
_overflow_noted: dict[tuple[str, str], str] = {}  # (filename, mode) -> ISO date noted


def _render_overflow_mode() -> str:
    try:
        from app.config import CONFIG

        mode = str((CONFIG.get("memory") or {}).get("render_overflow", "render_all"))
    except Exception:  # noqa: BLE001 — an unreadable config keeps today's prompt
        return "render_all"
    return mode if mode in _OVERFLOW_MODES else "render_all"


def _is_entry_start(line: str, prev: str) -> bool:
    """A line a cut may fall before: a heading, a top-level bullet, a paragraph."""
    return (line.startswith(("#", "- ", "* ")) or not prev.strip()) and bool(line.strip())


def _cut_at_entry_boundary(content: str, limit_bytes: int) -> tuple[str, int]:
    """(kept, dropped_bytes): the longest entry-aligned prefix within `limit_bytes`.

    Never mid-entry — half a ruling reads as a different ruling — so the kept text
    ends where the next heading, bullet or paragraph would have begun. A first
    entry larger than the whole limit keeps nothing rather than a fragment.
    """
    total = len(content.encode("utf-8"))
    if total <= limit_bytes:
        return content, 0
    lines = content.split("\n")
    kept_upto, size = 0, 0
    for i, line in enumerate(lines):
        if i and _is_entry_start(line, lines[i - 1]):
            if size <= limit_bytes:
                kept_upto = i
            else:
                break
        size += len(line.encode("utf-8")) + 1
    kept = "\n".join(lines[:kept_upto]).rstrip()
    return kept, total - len(kept.encode("utf-8"))


def _overflow_marker(filename: str, dropped: int) -> str:
    return (f'<memory_overflow file="{filename}" dropped_bytes="{dropped}">'
            f"{filename} is over its prompt ceiling; the entries after this point were "
            f'not loaded. memory_read(file="{filename}") returns the whole file.'
            f"</memory_overflow>")


def _overflow_announce(title: str, body: str) -> None:
    try:
        from app.prefix_miss import _announce

        _announce(title, body, False)
    except Exception as exc:  # noqa: BLE001 — the alert, never the prompt
        logger.warning("PROMPT_BUDGET could not announce memory overflow: %r", exc)


def _note_overflow_once(filename: str, mode: str) -> bool:
    today = datetime.date.today().isoformat()
    if _overflow_noted.get((filename, mode)) == today:
        return False
    _overflow_noted[(filename, mode)] = today
    return True


def _bound_memory_render(filename: str, raw: str, content: str) -> str:
    """`content`, or its entry-aligned cut plus a marker when the file is over."""
    from prompt_surface import memory_ceiling

    ceiling = memory_ceiling(filename)
    size = len(raw.encode("utf-8"))
    if ceiling is None or size <= ceiling:
        return content
    mode = _render_overflow_mode()
    if mode == "render_all":
        if _note_overflow_once(filename, mode):
            logger.warning(
                "PROMPT_BUDGET %s is %d bytes, over its %d-byte ceiling; rendered whole "
                "(memory.render_overflow: render_all)", filename, size, ceiling)
        return content
    kept, dropped = _cut_at_entry_boundary(content, ceiling)
    logger.error(
        "PROMPT_BUDGET %s is %d bytes, over its %d-byte ceiling; rendered %d and "
        "dropped %d at an entry boundary (memory.render_overflow: annotate)",
        filename, size, ceiling, len(kept.encode("utf-8")), dropped)
    if _note_overflow_once(filename, mode):
        _overflow_announce(
            f"Lloyd memory: {filename} over its ceiling",
            f"{filename} is {size:,} B against a {ceiling:,} B ceiling; the prompt "
            f"dropped {dropped:,} B of entries. Consolidate it "
            f"(scripts/memory/validate_memory_index.py names what is wrong).")
    return (kept + "\n\n" if kept else "") + _overflow_marker(filename, dropped)


# True when a skill's frontmatter `status:` pulls it from circulation.
# `agent_mcp.skills` already refuses to serve those, so advertising them here
# promised the model a skill that `skills_read` would then decline — the same
# "the prompt names something that does not work" failure as the phantom
# `web_search` tool references (2026-09-04).
#
# This is the walker's own function, not a re-implementation (#1294). The name used
# to be a local def that scanned the first 2000 characters line-by-line for a
# `status:` key — a second opinion rather than a copy of the rule, which called a
# nested `metadata:\n  openclaw:\n    status: archived` a retirement and missed a
# `status:` written below the 2000-character cut. The alias is kept because
# `tests/test_yaml_fix_skill_claims.py` and `tests/test_archived_skill_artifacts.py`
# reach for this name; the rule they are really testing now lives in one place.
_is_quarantined_skill = is_quarantined_skill_file


# ---------------------------------------------------------------------------
# P5: the index can say what each skill is for (`skills.index` in config.yaml).
#
# `descriptions: false` (the default) is today's names-only line, byte for byte —
# pinned by `tests/test_skills_index_descriptions.py`, because the index sits in
# the cached system-prompt prefix and a changed byte there re-prefills every
# session's next turn. On, the index is one `- name — description` line per
# skill under `budget_chars`, each description clipped at
# `max_description_chars`. Who gets a description is decided by 30-day use
# (`app.skill_telemetry`: offers + loaded + loaded_by_read), the tail stays
# names-only, and the lines are RENDERED ALPHABETICALLY whatever the rank: the
# ranking changes once a day, and an order that followed it would rewrite the
# prompt's position 0 every morning for no information the model can use.
#
# The description is the same field `GET /api/skills` shows (#1294) — the
# walker's parsed front matter, `str(fm.get("description") or "")` — so the
# model and the Skills page never describe one skill two ways.
#
# The closing note follows `prefetch.skills.push` (the other half of the arm):
# with push on, a matched skill's body is injected into the turn and the note
# says so (today's words); with it off, only the index is given and the note
# says to load a skill with `skills_read(name)`. The eval that decides both is
# `eval/run_prefetch_cost_eval.py --arms injected,desc_push,desc_pull`.
# ---------------------------------------------------------------------------
SKILLS_INDEX_BUDGET_CHARS = 12_000
SKILLS_INDEX_MAX_DESCRIPTION_CHARS = 100
#: The ranking window. `skill_telemetry.DEFAULT_DAYS` today; restated so a change
#: to that reader's default does not silently re-rank the prompt.
SKILLS_INDEX_RANK_DAYS = 30

_SKILLS_INDEX_PUSH_NOTE = (
    "Note: relevant skill content is automatically injected into each "
    "user message as <context> when matched."
)
_SKILLS_INDEX_PULL_NOTE = (
    "Note: skill bodies are not injected automatically. When a listed skill "
    "fits the task, load it with skills_read(name) before you start; files it "
    "bundles are read with Read."
)


def _skills_index_settings() -> dict:
    """`skills.index` from config.yaml, every key falling back to today's index.

    An unreadable config or a malformed value keeps the names-only line: this is
    the cached prefix, and a yaml hiccup must not be what changes it.
    """
    out = {"descriptions": False, "budget_chars": SKILLS_INDEX_BUDGET_CHARS,
           "max_description_chars": SKILLS_INDEX_MAX_DESCRIPTION_CHARS}
    try:
        from app.config import CONFIG

        cfg = ((CONFIG.get("skills") or {}).get("index") or {})
    except Exception:  # noqa: BLE001 — an unreadable config keeps today's prompt
        return out
    if not isinstance(cfg, dict):
        return out
    out["descriptions"] = cfg.get("descriptions") is True
    for key in ("budget_chars", "max_description_chars"):
        val = cfg.get(key)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            out[key] = val
    return out


def rpc_hint_enabled() -> bool:
    """`harness.rpc.enabled` (P9); False on any failure, today's prompt."""
    try:
        from app.harness.rpc_policy import enabled
    except Exception:  # noqa: BLE001
        return False
    return enabled()


def _rpc_hint() -> str:
    client = f"{LLOYD_HOME}/agent-services/bin/lloyd_rpc"
    lib = f"{LLOYD_HOME}/agent-services/rpc"
    return (
        "Programmatic tool calls: inside a Bash command you can call Lloyd's "
        "read-only tools from code and print only what you need, instead of one "
        "tool call per item. Command line: "
        f"`{client} call Read '{{\"file_path\": \"/abs/path\"}}'` prints the "
        f"result (exit 1 on a tool error). Python: `sys.path.insert(0, \"{lib}\"); "
        "import lloyd_rpc`, then `lloyd_rpc.call(\"Grep\", pattern=..., path=...)` "
        "returns the text and `lloyd_rpc.map(\"Read\", [{...}, ...], concurrency=4)` "
        "returns a list in order, an exception in place of each failure. Only "
        "read-only tools are available (Bash, Task, writes and anything this turn "
        "may not call are refused), every call must finish inside the Bash call's "
        "own timeout, and the Bash result ends with a one-line count of the calls "
        "made. Reach for it for many similar reads — twenty backlog items, ten "
        "files, a vault-wide frontmatter check — not for one or two."
    )


def skills_push_enabled() -> bool:
    """`prefetch.skills.push` — whether a matched skill's body is injected.

    The one reader, shared by `prefetch._skill_injection_plan` (which renders
    no body when it is off) and the index note above (which then tells the model
    to pull). Default and every failure is `True`, today's behaviour.
    """
    try:
        from app.config import CONFIG

        val = (((CONFIG.get("prefetch") or {}).get("skills") or {}).get("push", True))
    except Exception:  # noqa: BLE001
        return True
    return val is not False


def _skills_index_note(push: bool) -> str:
    return _SKILLS_INDEX_PUSH_NOTE if push else _SKILLS_INDEX_PULL_NOTE


def skill_index_description(frontmatter: dict) -> str:
    """The description the index shows: `/api/skills`' field, whitespace folded.

    Folded because a YAML block scalar keeps its newlines and one index entry
    must stay one line.
    """
    raw = frontmatter.get("description") if isinstance(frontmatter, dict) else None
    return " ".join(str(raw or "").split())


def clip_skill_description(desc: str, max_chars: int) -> str:
    """`desc` cut to at most `max_chars` characters, the cut marked with `…`."""
    if len(desc) <= max_chars:
        return desc
    return desc[: max(0, max_chars - 1)].rstrip() + "…"


_rank_cache: dict[str, dict[str, int]] = {}   # UTC date -> {skill: use count}


def _skill_rank_counts() -> dict[str, int]:
    """Per-skill 30-day use (offers + loaded + loaded_by_read), cached per UTC day.

    The scan reads every event log in the window (~0.8 s on 2026-09-24), so it
    runs once a day per process, not once per turn — which is also the cadence
    the rank is allowed to move the prefix at. Any failure is `{}`: every skill
    ties at zero and the rank falls back to alphabetical, the same answer as a
    window with no telemetry.
    """
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    cached = _rank_cache.get(today)
    if cached is not None:
        return cached
    counts: dict[str, int] = {}
    try:
        from app.paths import EVENT_LOGS_DIR
        from app.skill_telemetry import skill_injection_counts

        res = skill_injection_counts(EVENT_LOGS_DIR, SKILLS_INDEX_RANK_DAYS)
        for name, row in (res.get("skills") or {}).items():
            counts[name] = int(row.get("offers", 0)) + int(row.get("loaded", 0))
        for name, n in (res.get("loaded_by_read") or {}).items():
            counts[name] = counts.get(name, 0) + int(n)
    except Exception as exc:  # noqa: BLE001 — a rank is never worth a prompt
        logger.warning("skills index: telemetry rank unavailable (%s); "
                       "descriptions go to the alphabetical head", exc)
        counts = {}
    _rank_cache.clear()
    _rank_cache[today] = counts
    return counts


def _skills_index_lines(active, counts: dict[str, int] | None, *,
                        budget: int, max_desc: int) -> list[str]:
    """The description-arm index, one line per skill, alphabetical.

    `active` is `iter_active_skills` output (anything with `.name` and
    `.frontmatter`); `counts` ranks who gets a description — highest use first,
    name breaking ties, so an empty mapping is plain alphabetical. The header and
    every names-only line are always paid for: no skill is ever dropped from the
    index to make room for another's description. Descriptions are then granted
    in rank order while the whole index stays within `budget` characters; one
    that does not fit is skipped and the next, shorter one may still fit. A skill
    with no description is its name.
    """
    counts = counts or {}
    rows: dict[str, str] = {}
    for skill in active:
        rows.setdefault(skill.name, clip_skill_description(
            skill_index_description(skill.frontmatter), max_desc))
    names = sorted(rows)
    header = "Available skills (name — what it is for):"
    used = len(header) + sum(1 + len("- ") + len(n) for n in names)
    granted: set[str] = set()
    for name in sorted(names, key=lambda n: (-int(counts.get(n, 0)), n)):
        desc = rows[name]
        if not desc:
            continue
        extra = len(" — ") + len(desc)
        if used + extra <= budget:
            granted.add(name)
            used += extra
    return [header] + [f"- {n} — {rows[n]}" if n in granted else f"- {n}" for n in names]


def _load_skills_index(overlay: Path | None = None, *,
                       descriptions: bool | None = None) -> str | None:
    """Build the advertised skill index from skill directories (overlay first).

    The set is the single walker's (#1294). This function used to do its own
    `iterdir()` over the same roots with its own copy of the quarantine rule, which
    is how the advertised index, `GET /api/skills` and the Mission Control tab came
    to print 189, 187 and 194 about one vault — the model being told a set that no
    human-facing surface agreed with.

    `descriptions` overrides `skills.index.descriptions` (None reads config). Off
    is the names-only line exactly as it has always been rendered.
    """
    active = list(iter_active_skills(overlay=overlay, roots=_skill_walk_roots()))
    if not active:
        return None
    settings = _skills_index_settings()
    if descriptions is None:
        descriptions = settings["descriptions"]
    if not descriptions:
        return "Available skills: " + ", ".join(a.name for a in active)
    return "\n".join(_skills_index_lines(
        active, _skill_rank_counts(),
        budget=settings["budget_chars"], max_desc=settings["max_description_chars"]))
