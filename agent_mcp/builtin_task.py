#!/usr/bin/env python3
"""Lloyd MCP Server: Task built-in — in-process subagent.

Runs a nested `app.harness.run_query` call inside the current process.
The subagent gets the same lloyd-mcp tool pool minus `Task` itself
(recursion cap enforced via contextvars; currently 1 level deep).

Subagent profiles are defined under `subagents:` in config.yaml. If the
requested profile is absent, a minimal default is used (all tools, 20
turns). The `general-purpose` profile is the default.

Mounted into Server("lloyd") via agent_mcp/main.py MODULES.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import uuid
from typing import Any

from app.config import default_model_base_url
from mcp.types import Tool

from agent_mcp import _subagent_registry
from agent_mcp import _task_registry
from agent_mcp._shared import get_bound_session, text_result
from agent_mcp._subagent_registry import CallerScope

logger = logging.getLogger("lloyd-builtin-task")

# Recursion guard — depth of nested Task calls on the current call stack.
_task_depth: contextvars.ContextVar[int] = contextvars.ContextVar("_task_depth", default=0)
MAX_TASK_DEPTH = 1

# The calling turn's model and endpoint, bound by `agent_mcp.main.call_tool`
# from the request's `_meta` (see META_MODEL / META_BASE_URL there). This is
# the only channel that exists: Task runs inside the aggregator process, so
# without it a subagent has no idea which model spawned it and every profile
# fell back to `primary`. A turn on the secondary 35B delegated its
# subagents to the primary, which made the 35B untestable for exactly the
# fan-out work Task exists for.
current_parent_model: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_parent_model", default=""
)
current_parent_base_url: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_parent_base_url", default=""
)
# The calling turn's tool surface ("chat"/"worker", `lloyd/surface` in
# `_meta`), so a subagent a worker spawns is not handed the chat-only tools
# and one a chat spawns is not handed the worker-only ones.
current_parent_surface: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_parent_surface", default=""
)
# The calling turn's #534 grant scope (`lloyd/grant_scope`) and the deny list
# in force for the iteration that called Task (`lloyd/disallowed_tools`),
# bound by `agent_mcp.main.call_tool` like the three above. Before these a
# worker turn's subagent ran with no grant gate at all and with none of the
# parent's per-turn bans — the child could call what the parent was refused
# (review 2026-09-24, D4). An empty scope means a chat parent: no gate, the
# same as the chat route.
current_parent_grant_scope: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_parent_grant_scope", default=""
)
current_parent_disallowed: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "current_parent_disallowed", default=()
)


def current_caller_scope() -> CallerScope:
    """The identity whoever is acting inside this call, for the registry's
    authority check (#411).

    Reads the same two contextvars `_task` records as a row's parent scope,
    and they are non-empty only when a dispatch bound them from the request's
    `_meta` (`agent_mcp.main.call_tool`). So a control surface that names
    itself through this function can only claim the session its own dispatch
    arrived on — which is the whole policy, and why it is not a parameter a
    caller gets to pass.
    """
    return CallerScope(
        session_id=get_bound_session(),
        turn_id=_task_registry.current_turn_id.get(),
    )


# The system prompt a child gets when its profile names none (P8). A
# `general-purpose` child used to run with NO system prompt at all — a bare
# user message and a toolbox — so nothing told it that its last message is the
# whole deliverable, and a child that ended on "Let me check one more thing"
# handed that line back as its answer. Deliberately tool-agnostic: a profile's
# deny list decides what it may call, and a prompt that names tools drifts from
# that list (the `read-only` profile's prompt told it to use the Bash it was
# denied).
_DEFAULT_SUBAGENT_PROMPT = (
    "You are a subagent working for another agent on one scoped task. Your "
    "final message is returned to that agent verbatim and is all it sees of "
    "your work: none of your tool calls or intermediate text reach it. Do the "
    "task, then end with one self-contained answer. Be concise. Cite files by "
    "absolute path (with line numbers where they matter). Say plainly what "
    "you could not verify or did not get to, rather than implying you did."
)


def _load_subagent_profile(subagent_type: str) -> dict[str, Any]:
    """Read one `subagents:` profile from the live config.

    Goes through `app.config.CONFIG` rather than opening config.yaml
    directly. The direct read bypassed both `${VAR}` expansion and the
    merge of `data/tool_overrides.yaml`, so a subagent saw a different
    configuration than every other caller.
    """
    from app.config import CONFIG

    profiles = CONFIG.get("subagents") or {}
    profile = profiles.get(subagent_type) or {}
    return {
        "system_prompt": profile.get("system_prompt", ""),
        "max_turns": int(profile.get("max_turns", 20)),
        "disallowed_tools": list(profile.get("disallowed_tools") or []),
        "model": profile.get("model", ""),
        "base_url": profile.get("base_url", ""),
        # P8: the parent may overlap a batch of calls to this profile, so its
        # child is held to the read-only tool set (see `_task`). Literally
        # `true` only, matching `app.mcp_discovery.parallel_safe_task_profiles`
        # — the parent's reading and the child's must be the same.
        "parallel_safe": profile.get("parallel_safe") is True,
    }


def _parallel_safe_blocked(disallowed: list[str]) -> list[str]:
    """`disallowed` plus every discovered tool that is not read-only (P8).

    A parallel-safe child is overlapped with its siblings by the parent's loop
    (`app.harness.loop._is_task_fanout`), so what makes that safe has to be a
    property of the child, not of its prompt: every tool without `readOnlyHint`
    in `agent_mcp.annotations` is refused to it, the same derivation plan mode
    uses. Evaluated per iteration through `disallowed_tools_refresh`, because
    the tool universe is recorded when the child's own pool opens — after
    these options are built. `ToolSearch` and `_`-internal names stay usable;
    until a universe exists the three-tool floor applies, never nothing.
    """
    from agent_mcp.annotations import READ_ONLY
    from app.mcp_discovery import PLAN_MODE_BLOCKED_TOOLS, _TOOL_UNIVERSE

    out = list(disallowed)
    seen = set(out)
    names = set(_TOOL_UNIVERSE) or set(PLAN_MODE_BLOCKED_TOOLS)
    for name in sorted(names):
        if name in READ_ONLY or name.startswith("_") or name == "ToolSearch":
            continue
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _final_schema_arg(args: dict[str, Any]) -> tuple[dict | None, str]:
    """The optional `final_schema` argument, or why it was refused.

    The object goes to the harness finalizer as a guided-decoding grammar, so
    only a JSON-Schema *object* schema is accepted; anything else is reported
    back on the result as `structured_error` rather than failing the Task — the
    child's prose answer is still the answer.
    """
    raw = args.get("final_schema")
    if raw in (None, "", {}):
        return None, ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None, "final_schema is not valid JSON"
    if not isinstance(raw, dict) or raw.get("type") != "object":
        return None, 'final_schema must be a JSON Schema with "type": "object"'
    return raw, ""


def _base_url_for(model: str) -> str:
    """Endpoint that serves `model`, preferring the caller's own.

    The parent's base_url is authoritative when the model matches: it is
    the endpoint the turn is actually streaming from, which survives a
    caller that overrode the config default. Otherwise resolve the alias
    through `models:` in config, and only then fall back to the default
    model's endpoint.
    """
    if model and model == current_parent_model.get(""):
        parent_url = current_parent_base_url.get("")
        if parent_url:
            return parent_url
    from app.config import _get_model_cfg

    cfg = _get_model_cfg(model) or {}
    url = cfg.get("base_url") or (cfg.get("env") or {}).get("ANTHROPIC_BASE_URL", "")
    return url or default_model_base_url()


def _call_summary() -> str:
    """This call's caption, as lifted from `_meta` by ``main.call_tool``."""
    return (_task_registry.current_call_summary.get() or "").strip()


def _child_notification_drain(child_session_id: str):
    """Splice the child's own background completions into its next iteration. (#929)

    Same shape as the chat turn's drain in `app.routers.messages`, minus the
    session-message persistence: a subagent transcript is in-process (the
    registry row and the resume ring), not a `~/lloyd-data/sessions/<id>.json` file,
    so there is no home to persist a `bg_task_notification` message to. The
    record still reaches the model, the transcript and a resumed run — that is
    what the Bash tool's own wording promises ("a `<task_notification>` will
    appear on a later turn"), which before this was true only for a chat turn.

    The loop awaits the callback and extends `chat_messages` with what comes
    back (app/harness/loop.py), so this is a coroutine like the chat turn's.
    Only the child's own key is read: a record the close sweep already moved
    lives under the parent's key, where the parent's turn serves it.
    """
    async def drain() -> list[dict[str, Any]]:
        mine = await _task_registry.drain_completed_for_session(child_session_id)
        return [
            {"role": "user",
             "content": (_task_registry.format_notification(rec)
                         if isinstance(rec, _task_registry.TaskRecord)
                         else _task_registry.format_diagnostics_notification(rec))}
            for rec in mine
        ]

    return drain


def _hand_off_background_tasks(record: Any) -> None:
    """Route a closed child's background completions to its parent. (#929)

    Called while the run is still in `_subagent_registry`'s active table, so
    `parent_scope` resolves it. It also teaches `_task_registry` to re-key
    anything that arrives later for this child session, which is the common
    case: a subprocess routinely exits after the run that spawned it returned,
    and a `task:*` queue has no reader by then.
    """
    scope = _subagent_registry.parent_scope(record.session_id)
    parent_session_id = scope[0] if scope else ""
    if not parent_session_id:
        # Nothing to hand off to: a Task whose parent turn carried no session,
        # or a row the ring already evicted. Leaving the records queued is the
        # pre-#929 shape, so this loses nothing — but it says so rather than
        # reporting a hand-off that did not happen.
        logger.warning(
            "bg tasks for subagent %s have no parent session to hand off to",
            record.session_id)
        return
    moved = _task_registry.hand_off_to_parent(record.session_id, parent_session_id)
    if moved:
        logger.info(
            "bg tasks for closed subagent %s handed to parent %s: %s",
            record.session_id, parent_session_id,
            ",".join(r.task_id for r in moved if isinstance(r, _task_registry.TaskRecord)))


async def _task(args: dict[str, Any]) -> str:
    # The subagent row's label. Comes from the caption the model already
    # wrote for this call (request `_meta`; see main.META_SUMMARY) rather
    # than from a second argument asking the same question — Task's old
    # `description` ("Short label for the task (informational)") competed
    # with the injected `summary` display parameter for one answer, and a
    # model that spends it on the unadvertised half leaves the transcript
    # bubble blank. Still read from args so a session mid-drain that
    # learned the old shape keeps its labels.
    description = (args.get("description") or "").strip() or _call_summary()
    prompt = args.get("prompt", "")
    subagent_type = args.get("subagent_type", "general-purpose")
    resume_id = (args.get("task_id") or "").strip()
    final_schema, schema_error = _final_schema_arg(args)

    if not prompt:
        return json.dumps({"error": "prompt is required"})

    depth = _task_depth.get()
    if depth >= MAX_TASK_DEPTH:
        return json.dumps({
            "error": f"Task recursion limit ({MAX_TASK_DEPTH}) reached — nested Task calls are not allowed."
        })

    history = None
    if resume_id:
        try:
            history = _subagent_registry.claim_history(resume_id)
        except _subagent_registry.HistoryUnavailable as exc:
            return json.dumps({
                "error": f"Cannot resume task {resume_id!r}: {exc}",
                "task_id": resume_id,
            })

    if history is not None:
        # The stored run's identity wins. A `subagent_type` passed beside a
        # `task_id` would silently change the profile of a conversation
        # already half-done under the old one.
        if args.get("subagent_type") and args["subagent_type"] != history.subagent_type:
            logger.info("Task resume %s: ignoring subagent_type=%r (stored: %r)",
                        resume_id, args["subagent_type"], history.subagent_type)
        subagent_type = history.subagent_type
        profile = dict(history.profile)
        description = description or history.description
    else:
        profile = _load_subagent_profile(subagent_type)

    # Import here to avoid circular import at module load time.
    from app.harness.hooks import HookRegistry
    from app.harness.loop import run_query
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from app.harness.options import RunOptions
    from app.harness.policy import install_policy_hook
    from app.harness.safety import install_default_safety_hook
    from app.tool_bans import WORKER_AUTOMOD_BAN
    from app.harness.skill_dispatch import install_skill_dispatch_hook
    from app.harness.turn_guards import install_turn_guards
    from app.deadline_anchor import build_iteration_anchor

    # Resolve model and base_url.
    #
    # Precedence: an explicit `subagents.<type>.model` pin wins; otherwise
    # the subagent inherits the calling turn's model, so delegating from a
    # turn on the secondary keeps the work on the secondary. Falls back to
    # `primary` only when there is no parent context at all (Task invoked
    # outside a harness turn).
    #
    # base_url is resolved FROM the chosen model rather than from
    # `default_model_base_url()`, which always returns the primary's
    # endpoint: a profile pinned to `model: secondary` with an empty
    # `base_url` used to send "secondary" to the primary's port, where the
    # engine does not serve that name.
    if history is not None:
        # Keep the continuation where its KV cache already is. Consulting
        # `current_parent_model` here would move a half-finished conversation
        # to another engine and re-prefill all of it.
        model = history.model
        base_url = history.base_url
    else:
        model = profile["model"] or current_parent_model.get("") or "primary"
        base_url = profile["base_url"] or _base_url_for(model)

    # Config-level tool disables apply to subagents too. They did not
    # before: `disallowed` came from the profile alone, so any tool
    # switched off in the Tools page or listed in
    # `mcp_servers.<server>.disabled_tools` stayed fully callable inside
    # a Task — the one execution context with no human watching the
    # stream. Same reasoning that puts the safety hook here.
    from app.config import CONFIG
    from app.mcp_discovery import _get_disallowed_tools, max_turns_wrapup_kwargs

    disallowed = list(_get_disallowed_tools())
    for name in profile["disallowed_tools"]:
        if name not in disallowed:
            disallowed.append(name)
    # ...and whatever the PARENT could not call this iteration (plan mode, a
    # worker's per-turn bans, `grant_create` on a gated turn). A child is the
    # parent's delegate; handing it a tool the parent was refused is a way
    # round every ban the router arms.
    for name in current_parent_disallowed.get(()):
        if name not in disallowed:
            disallowed.append(name)
    # Subagent always disallows Task to prevent infinite recursion.
    if "Task" not in disallowed:
        disallowed.append("Task")

    # ...and never the self-modification loop. A subagent is spawned to do a
    # bounded piece of research or editing and reports back through a summary;
    # nothing about that shape suits opening a round, gating it, or landing
    # code on production. Landing from inside a Task would also restart the
    # aggregator the Task is running in. Same reasoning as the Task recursion
    # cap: the constraint belongs here, not in a prompt.
    # The list is the workers' shared one (`app/tool_bans.py`): this used to
    # be a private five-name copy that had fallen four names behind it.
    for name in WORKER_AUTOMOD_BAN:
        for spelling in (name, f"mcp__lloyd-mcp__{name}"):
            if spelling not in disallowed:
                disallowed.append(spelling)

    # Subagents ran with `hooks=None`, which meant the harness's default
    # destructive-Bash gate never installed inside a Task — the one place
    # with no human watching the stream. `safety.py` is documented as
    # running on every primary turn; a subagent is a harness run like any
    # other and gets the same floor.
    #
    # The Inner Voice observer is deliberately NOT attached here: it is
    # scoped to a session turn (goal card, session todos, ambient and
    # clarify channels) and a subagent has none of those.
    # P8: a parallel-safe profile's child may be running beside its siblings,
    # so it gets the read-only set, re-derived per iteration (see
    # `_parallel_safe_blocked`). A resume keeps the profile it was stored
    # with, so a stored parallel-safe run stays read-only too.
    disallowed_refresh = None
    if profile.get("parallel_safe"):
        base_disallowed = list(disallowed)
        disallowed = _parallel_safe_blocked(base_disallowed)
        disallowed_refresh = functools.partial(_parallel_safe_blocked, base_disallowed)

    task_hooks = HookRegistry()
    install_default_safety_hook(task_hooks)
    # The dispatch-time skill deliverer (#536), under the same
    # `harness.skill_dispatch.enabled` flag as the stream route and installed
    # after the safety gate for the same reason it is there. A subagent runs
    # exactly the repeated protocols the deliverer exists for, and it gets no
    # turn-start skills at all (the prompt is the raw `subagents.<type>`
    # profile), so `already_injected` is empty by construction (#750).
    install_skill_dispatch_hook(task_hooks)
    # The #534 grant gate, when the parent turn ran under one. It is a
    # PreToolUse hook, so it exists only where a registry installs it, and
    # this registry is the child's (a deny beats a deliver in either order).
    # A chat parent carries no scope and gets no gate — the same rule as the
    # chat route (`app/routers/messages.py`).
    grant_scope = current_parent_grant_scope.get("")
    if grant_scope:
        install_policy_hook(task_hooks, scope=grant_scope)
    # The deterministic senses are not the observer: a subagent that stalls
    # on "Let me check…" or re-runs one grep five times is the common case,
    # and nothing about them needs a session. Platform left empty on purpose —
    # a subagent's terminating text IS its answer, so "deliver" is right.
    install_turn_guards(task_hooks)

    # Per-invocation session id so each subagent run gets its own
    # tool_search LoadedToolSet — different disallowed_tools profiles
    # would otherwise share one cache entry. Bound here rather than
    # inline so the registry row and the harness agree on the id.
    # Reused on a resume so the continuation keeps its tool_search
    # LoadedToolSet and its spill directory.
    sub_session_id = (history.session_id if history is not None
                      else f"task:{subagent_type}:{uuid.uuid4().hex[:8]}")
    # #929: a resume reuses the id, so this run IS the reader of that key again.
    # The parent route recorded when the previous run closed has to go, or a
    # completion this child is told to expect would be written to the parent's
    # queue instead of the queue this run drains.
    _task_registry.unhand_off(sub_session_id)

    # The child's stop request (#411). ONE object, shared three ways: the
    # loop checks it between iterations, between SSE chunks and before every
    # tool dispatch (`app/harness/loop.py`), the registry row publishes it so
    # a control surface can set it, and the loop reads it off these options.
    # Built before `RunOptions` because the row has to carry the same event
    # the child does — a second Event nobody sets is not a stop button.
    cancel_evt = asyncio.Event()

    options = RunOptions(
        model=model,
        base_url=base_url,
        system_prompt=profile["system_prompt"] or _DEFAULT_SUBAGENT_PROMPT,
        max_turns=profile["max_turns"],
        disallowed_tools=disallowed,
        disallowed_tools_refresh=disallowed_refresh,
        # P8: the child sees its own cap coming, in the chat path's wording at
        # 75% and 90% — before this a child learned its budget from the cap.
        # Appended, never written into position 0.
        state_anchor=build_iteration_anchor(profile["max_turns"]),
        # P8: the caller's optional contract for the answer. Restated by the
        # finalizer after a clean stop only; `structured` rides the result.
        final_schema=final_schema,
        hooks=task_hooks,
        mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
        session_id=sub_session_id,
        surface=current_parent_surface.get(""),
        # Carried on so the child's own calls stamp it into `_meta` too.
        grant_scope=grant_scope,
        cancel_event=cancel_evt,
        # A subagent is an agent loop like any other and has the same
        # redundant-reasoning problem the main loop does — more so, since
        # it runs its whole investigation inside one turn.
        preserve_thinking_iterations=int(
            (CONFIG.get("harness") or {}).get("preserve_thinking_iterations", 0)
        ),
        # A wedged engine inside a subagent is worse than one on the main
        # turn: the caller is blocked on `_task` with nothing streaming and
        # no human watching. Same deadline as the parent.
        stream_chunk_timeout_s=float(
            (CONFIG.get("harness") or {}).get("stream_chunk_timeout_seconds", 0)
        ),
        # A subagent is the fan-out case parallel dispatch exists for — it
        # reads far more than it writes — and it constructs its own
        # RunOptions, so without this the flag would be on for every chat
        # turn and off for exactly the runs it helps most.
        parallel_tool_calls_enabled=bool(
            ((CONFIG.get("harness") or {}).get("parallel_tool_calls") or {})
            .get("enabled", False)
        ),
        # #929: the child's own background completions, drained against the
        # child's own session id. Nothing else ever reads a `task:*` queue —
        # the chat-turn drain is wired only for chat turns — so without this a
        # child that ran `Bash` with run_in_background=true was told a
        # `<task_notification>` would appear on a later turn and never saw one.
        # Deliberately NOT the chat builder in app.routers.messages: that one
        # also appends a session message to the session it is handed, and a
        # subagent transcript lives in this process, not in
        # `~/lloyd-data/sessions/<id>.json`.
        notification_drain=_child_notification_drain(sub_session_id),
        parallel_tool_calls_max_concurrency=max(1, int(
            ((CONFIG.get("harness") or {}).get("parallel_tool_calls") or {})
            .get("max_concurrency", 4)
        )),
        # P6b: a child that exhausts its budget asks once, toollessly, where
        # the work stands; that text becomes `final_text` below, marked
        # `truncated` because `stop_reason` stays `max_turns`. Decided per
        # call against the child's own `base_url`.
        **max_turns_wrapup_kwargs(),
    )

    # On a resume the follow-up is appended to the STORED list, which is then
    # handed back as `chat_messages_handle` — `run_query` ignores `messages`
    # entirely when the handle is non-empty (loop.py), so passing the
    # follow-up as `messages` would silently drop it.
    #
    # This same list object is what the registry row publishes below, so it is
    # both the resume mechanism and, since #411, the thing a steering append
    # goes into while the child is running. The loop reads it by reference.
    if history is not None:
        chat_messages = history.chat_messages
        chat_messages.append({"role": "user", "content": prompt})
        options.chat_messages_handle = chat_messages
        messages: list[dict[str, Any]] = []
    else:
        chat_messages = [{"role": "user", "content": prompt}]
        options.chat_messages_handle = chat_messages
        messages = list(chat_messages)

    # The subagent's answer is the text of its TERMINATING iteration —
    # the one that stopped without calling a tool. Accumulating every
    # `text_delta` instead (as this did until 2026-09-05) concatenates
    # each iteration's preamble, so a subagent that burns all its turns
    # mid-investigation returns its opening "I'll start by reading the
    # repo" as though that were the finished review. Session
    # 20260905_151355_iv5174 lost 231s to exactly that: 19 tool calls,
    # a preamble for an answer, and no signal to the caller that the
    # work never happened.
    final_text = ""            # terminal (tool-call-free) iteration's text
    last_iter_text = ""        # most recent iteration's text, terminal or not
    last_iter_thinking = ""    # ...and its reasoning, for diagnosing empty answers
    tool_calls_summary: list[str] = []
    stop_reason = "stop"
    num_turns = 0
    structured: dict | None = None
    structured_error = schema_error

    # Open the live row BEFORE the loop starts. A Task blocks its caller
    # for as long as it runs, so a row created on completion would only
    # ever describe runs that already finished — precisely the ones that
    # no longer need watching.
    record = _subagent_registry.register(
        subagent_type=subagent_type,
        description=description,
        prompt=prompt,
        parent_session_id=get_bound_session(),
        # A subagent has no turn of its own, and nothing ever reads its
        # ledger or its drain queue once Task returns. The parent's turn is
        # where a human looks for "what did this turn change".
        parent_turn_id=_task_registry.current_turn_id.get(),
        session_id=sub_session_id,
        model=model,
        max_turns=profile["max_turns"],
        task_id=history.task_id if history is not None else "",
        continuation_of=history.last_run_id if history is not None else "",
        # #411: the two handles that turn this row from a view into a control
        # surface. Published here, before `run_query` is entered, because a
        # row that only learns of its conversation at `_close` describes a
        # run nobody could still influence. `_close`/`finish` drops the row
        # out of `_active`, so a published handle can never outlive the run.
        chat_messages=chat_messages,
        cancel_event=cancel_evt,
    )
    runs_so_far = (history.runs if history is not None else 0) + 1

    def _close(status: str, **kw) -> None:
        """Close the row AND store the conversation, on every exit path.

        One helper because there are five exits and each one has to do both:
        a run that closed its row but stored nothing is a task_id the model
        is told about and cannot resume.
        """
        # #929: first thing, while the row still resolves to its parent via
        # `_subagent_registry.parent_scope` — `finish` below is what moves it out
        # of the active table. Whatever this child queued but never drained goes
        # to the parent's queue now, and whatever completes after this point is
        # re-keyed there too, because a `task:*` queue has no reader once `Task`
        # has returned.
        _hand_off_background_tasks(record)
        _subagent_registry.finish(record, status=status, **kw)
        try:
            _subagent_registry.store_history(
                task_id=record.task_id, subagent_type=subagent_type,
                profile=profile, model=model, base_url=base_url,
                session_id=sub_session_id, description=description,
                chat_messages=chat_messages, run_id=record.run_id,
                runs=runs_so_far,
            )
        except Exception:
            logger.warning("Task: could not store history for %s",
                           record.task_id, exc_info=True)

    token = _task_depth.set(depth + 1)
    try:
        async for evt in run_query(messages, options):
            if evt["type"] == "assistant_message":
                last_iter_text = evt.get("text") or ""
                last_iter_thinking = evt.get("thinking") or ""
                record.note_turn()
                if not evt.get("tool_calls"):
                    final_text = last_iter_text
            elif evt["type"] == "tool_call":
                tool_calls_summary.append(evt["name"])
                record.note_tool(evt["name"])
            elif evt["type"] == "result":
                stop_reason = evt.get("stop_reason", "stop")
                num_turns = int(evt.get("num_turns", 0) or 0)
                if final_schema is not None:
                    structured = evt.get("structured")
                    structured_error = str(evt.get("structured_error") or "")
    except asyncio.CancelledError:
        # Closed here rather than in `finally`: a finally runs before the
        # success paths below, and `finish` is idempotent, so a blanket
        # close there would stamp every completed run "cancelled" and
        # make the real status a no-op.
        _close("cancelled", stop_reason="cancelled")
        raise
    except Exception as exc:
        logger.exception("Task subagent error for prompt=%r", prompt[:120])
        _close("error", error=str(exc))
        return json.dumps({"error": f"Subagent failed: {exc}",
                           "task_id": record.task_id})
    finally:
        _task_depth.reset(token)

    # Stopped through its own `cancel_event` (#411). Neither of the branches
    # below describes this: it is not a failure to produce an answer, and no
    # exception fired — `app/harness/loop.py` breaks cleanly and reports
    # `stop_reason == "cancelled"`. It has to be answered explicitly, because
    # the default fall-through would report an uncancelled "stop" with an
    # empty response, and the caller would read a killed child as one that
    # finished with nothing to say.
    #
    # `_close` still runs: the row records the real status and the history is
    # stored, so the task_id stays resumable. Stopping a wedged child to
    # restart it with better instructions is the point; losing the work it had
    # already done would make the button worthless.
    if stop_reason == "cancelled":
        _close("cancelled", stop_reason="cancelled")
        cancelled: dict[str, Any] = {
            "error": ("Subagent was cancelled before it finished. Its "
                      "conversation is stored — resume it with `task_id`."),
            "stop_reason": "cancelled",
            "turns_used": num_turns,
            "tools_used": tool_calls_summary,
            "task_id": record.task_id,
        }
        if last_iter_text.strip():
            cancelled["partial_text"] = last_iter_text[:500]
        return json.dumps(cancelled)

    # A subagent that dispatched tools but never produced a closing
    # message did NOT do the job. Return an error rather than a
    # plausible-looking empty string — the caller cannot otherwise tell
    # "reviewed it, found nothing" from "never got there".
    if not final_text.strip() and tool_calls_summary:
        reason = _diagnose_empty_answer(
            stop_reason=stop_reason,
            num_turns=num_turns,
            max_turns=profile["max_turns"],
            thinking=last_iter_thinking,
        )
        logger.warning(
            "Task subagent produced no final answer (%s): stop_reason=%s "
            "turns=%d/%d tools=%d",
            reason, stop_reason, num_turns, profile["max_turns"],
            len(tool_calls_summary),
        )
        err: dict[str, Any] = {
            "error": (
                f"Subagent ran {len(tool_calls_summary)} tool calls but returned "
                f"no final answer ({reason}). The task was NOT completed — do "
                f"not treat any text below as the result."
            ),
            "stop_reason": stop_reason,
            "turns_used": num_turns,
            "max_turns": profile["max_turns"],
            "tools_used": tool_calls_summary,
        }
        if last_iter_text.strip():
            err["partial_text"] = last_iter_text[:500]
        if description:
            err["description"] = description
        if final_schema is not None or schema_error:
            err["structured"] = None
            err["structured_error"] = structured_error
        err["task_id"] = record.task_id
        _close("failed", stop_reason=stop_reason, error=reason)
        return json.dumps(err)

    result: dict[str, Any] = {"response": final_text}
    # Surface truncation even when there IS text — a max_turns run can
    # still end on a partial answer.
    if stop_reason not in ("stop", "end_turn"):
        result["stop_reason"] = stop_reason
        result["truncated"] = True
        result["turns_used"] = num_turns
        result["max_turns"] = profile["max_turns"]
    if tool_calls_summary:
        result["tools_used"] = tool_calls_summary
    if description:
        result["description"] = description
    # Only when the caller asked: `structured` is the object or None, and
    # `structured_error` says why it is None (skipped after a non-clean stop,
    # a bad schema, a finalizer failure) — never a silent absence.
    if final_schema is not None or schema_error:
        result["structured"] = structured
        result["structured_error"] = structured_error
    result["task_id"] = record.task_id
    _close("completed", stop_reason=stop_reason, response_chars=len(final_text))
    return json.dumps(result)


def _diagnose_empty_answer(
    *, stop_reason: str, num_turns: int, max_turns: int, thinking: str,
) -> str:
    """Short human-readable cause for a subagent that returned no answer."""
    if stop_reason == "max_turns" or num_turns >= max_turns:
        return f"exhausted its {max_turns}-turn budget"
    if stop_reason == "cancelled":
        return "was cancelled"
    if thinking.strip():
        return "emitted reasoning but no final message"
    return f"stopped with an empty final message (stop_reason={stop_reason})"


async def list_tools():
    return [
        Tool(
            name="Task",
            description=(
                "Use for a scoped subtask you need now; for tracked or recurring work use autonomy_write_task instead.\n\n"
                "Spawn a subagent to complete a task. The subagent runs in the "
                "same process with the full lloyd-mcp tool pool (minus Task). "
                "Returns the subagent's final response, the tools it used, and a "
                "`task_id`. Pass that `task_id` back with a follow-up `prompt` to "
                "CONTINUE the same subagent instead of starting over — it keeps "
                "its conversation, its tool results and its warm cache, which is "
                "what you want when one ran out of turns mid-investigation. "
                "Nested Task calls are not allowed (recursion cap = 1). "
                "Several `subagent_type: read-only` Tasks called in one "
                "message run at the same time; that profile cannot write."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task prompt for the subagent, or the "
                                       "follow-up when continuing one",
                    },
                    "subagent_type": {
                        "type": "string",
                        "description": "Profile name from config.yaml subagents section (default: general-purpose)",
                        "default": "general-purpose",
                    },
                    "final_schema": {
                        "type": "object",
                        "description": (
                            "Optional JSON Schema (type: object) for a "
                            "machine-readable answer. After the subagent "
                            "finishes cleanly it restates its answer as one "
                            "object matching this schema, returned as "
                            "`structured` beside `response`; "
                            "`structured_error` says why when there is none."
                        ),
                    },
                    "task_id": {
                        "type": "string",
                        "description": (
                            "Continue a previous Task instead of starting fresh. "
                            "Use the `task_id` it returned. The last few finished "
                            "subagents are kept for about 30 minutes and only "
                            "within this process, so an id from before an "
                            "aggregator restart is refused as 'unknown or "
                            "evicted'. `subagent_type` is ignored when this is set."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        ),
    ]


async def call_tool(name: str, arguments: dict):
    if name == "Task":
        text = await _task(arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return text_result(text)
