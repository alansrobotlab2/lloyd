"""Turn guards — the deterministic senses, on every turn.

Stall rescue, the repetition guard, failure-payload escalation, the open-todo
gate and the open-round gate used to live inside the Inner Voice observer's
`install_observer` closure. That made them exist only on a turn whose session
had opted into the observer, so the 2026-09-12 cut that switched the observer
off for every worker also switched off the one part of it that had been
demonstrably right: every intervention in the surviving window (09-22 → 09-24)
and most of the useful ones in the recovered transcripts were "the primary
stopped before finishing", and all of those are cheap to detect without a
model.

So they are a harness hook now, installed beside `install_default_safety_hook`
on the chat path (`messages._run_turn` and the sync route), on direct worker
turns (`workers/sources/_common.py`), on scheduled tasks (`autonomy.py`) and
inside every `Task` subagent (`agent_mcp/builtin_task.py`). No LLM, no session
flag. The judgment itself stays in `app/inner_voice/guards.py` as pure
functions; only the wiring moved.

How an inject works. The loop continues a turn that was about to end only if
an OnEvent hook grew the message list it reads (`loop.py`, "observer injected
on terminal iteration"), and it reorders a mid-batch append back into wire
order. `run_query` binds that list to the registry (`HookRegistry.bind_run`),
because a caller with no `chat_messages_handle` has only a private copy.

Each fire is recorded like an observer decision: an `inner_voice_observations`
row with `safeguard` set to the guard's name and no `model`, and an
`inner_voice.observer_injected` event with `deterministic: true`, so
`iv_grade` and `iv_outcome_score` keep reading them. On a chat turn the
router's breadcrumb callback also writes the `[INNER VOICE]` line into the
session, where the user sees it.

When the observer is attached to the same registry it does not run its own
copies; it listens (`add_listener`) so its suppressors and prompt history
still see what the guards did.

Kill switches: `inner_voice.turn_guards.enabled`, and one per guard.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.inner_voice import guards as _g

logger = logging.getLogger("lloyd-harness-turn-guards")

# Signatures kept for repetition comparison. One more than the widest
# comparison window, and never under the old observer ring.
_RING_FLOOR = 16

# A capability fault is announced at most this often per process: one toast
# says the pool is broken; ten say nothing more.
_FAULT_ANNOUNCE_COOLDOWN_S = 1800.0
_last_fault_announce: float = 0.0

_ROUND_OPEN_TOOLS = frozenset({"automod_start"})
_ROUND_CLOSE_TOOLS = frozenset({"automod_land", "automod_abort"})


def _cfg() -> dict[str, Any]:
    """`inner_voice.turn_guards`, with the old observer keys as fallback.

    The repetition knobs and the deterministic cap lived under
    `inner_voice.observer` while the guards did; a config that still sets them
    there keeps working.
    """
    try:
        from app.config import CONFIG

        iv = CONFIG.get("inner_voice") or {}
    except Exception:  # noqa: BLE001 — no config is the defaults
        iv = {}
    obs = iv.get("observer") or {}
    tg = dict(iv.get("turn_guards") or {})
    for key, default in (
        ("repetition_window", _g.REPETITION_WINDOW),
        ("repetition_threshold", _g.REPETITION_THRESHOLD),
        ("repetition_exempt_tools", []),
        ("deterministic_inject_budget", 5),
    ):
        tg.setdefault(key, obs.get(key, default))
    tg.setdefault("repetition", obs.get("repetition_guard_enabled", True))
    tg.setdefault("enabled", True)
    tg.setdefault("stall_rescue", True)
    tg.setdefault("failure_payload", True)
    tg.setdefault("todo_gate", True)
    tg.setdefault("round_open", True)
    tg.setdefault("capability_fault", True)
    return tg


def _is_unattended(platform: str) -> bool:
    """`sessions_io.NON_USER_PLATFORMS` is the one definition."""
    if not platform:
        return False
    try:
        from app.sessions_io import NON_USER_PLATFORMS
    except Exception:  # noqa: BLE001
        return False
    return platform in NON_USER_PLATFORMS


def _load_todos(session_id: str) -> list[dict[str, Any]]:
    if not session_id or session_id.startswith("task:"):
        return []
    try:
        from app.paths import SESSIONS_DIR

        p = SESSIONS_DIR / f"{session_id}.json"
        if not p.exists():
            return []
        return json.loads(p.read_text()).get("todos") or []
    except Exception:  # noqa: BLE001 — a todo list we cannot read is none
        return []


@dataclass
class GuardFire:
    """One deterministic inject, as the observer's listener sees it."""

    guard: str          # stall_rescue | repetition | failure_payload | todo_gate | round_open
    trigger: str        # pretool | assistant_message | tool_result
    reason: str
    content: str
    related_tool: str | None
    iteration: int


@dataclass
class TurnGuardState:
    hooks: Any
    session_id: str = ""
    turn_id: str = ""
    platform: str = ""
    source: str = ""
    cfg: dict[str, Any] = field(default_factory=dict)
    chat_messages: list[dict[str, Any]] | None = None
    persist_intervention_callback: Callable[[str, str, str], Awaitable[None]] | None = None
    recent: list[_g.ToolCallSignature] = field(default_factory=list)
    tool_calls_seen: int = 0
    repetition_baseline: int = 0
    fires: list[GuardFire] = field(default_factory=list)
    round_open: bool = False
    todo_write_seen: bool = False
    todo_gate_fired: bool = False
    round_gate_fired: bool = False
    failure_tools_fired: set[str] = field(default_factory=set)
    last_iteration: int = 0
    sequence: int = 0
    listeners: list[Callable[[GuardFire], None]] = field(default_factory=list)

    # -- identity, resolved lazily: a direct-path caller installs before the
    #    loop has bound its options.
    def _sid(self) -> str:
        if self.session_id:
            return self.session_id
        opts = getattr(self.hooks, "run_options", None)
        return str(getattr(opts, "session_id", "") or "")

    def _tid(self) -> str:
        if self.turn_id:
            return self.turn_id
        opts = getattr(self.hooks, "run_options", None)
        return str(getattr(opts, "turn_id", "") or "")

    @property
    def unattended(self) -> bool:
        return _is_unattended(self.platform)

    def next_sequence(self) -> int:
        """Shared with the observer's rows, so one turn's rows sort."""
        self.sequence += 1
        return self.sequence

    def add_listener(self, cb: Callable[[GuardFire], None]) -> None:
        self.listeners.append(cb)

    def fired_on_iteration(self, iteration: int) -> bool:
        return any(
            f.iteration == iteration and f.trigger == "assistant_message"
            for f in self.fires
        )

    def _target(self) -> list[dict[str, Any]] | None:
        if self.chat_messages is not None:
            return self.chat_messages
        return getattr(self.hooks, "chat_messages", None)

    async def fire(
        self, *, guard: str, trigger: str, reason: str, content: str,
        related_tool: str | None = None,
    ) -> bool:
        """Append the inject, record it, tell the listeners. False if capped."""
        target = self._target()
        if target is None:
            logger.warning("turn_guards: %s fired with no message list bound", guard)
            return False
        cap = int(self.cfg.get("deterministic_inject_budget", 5))
        if cap > 0 and len(self.fires) >= cap:
            logger.warning(
                "turn_guards: deterministic cap %d reached session=%s guard=%s",
                cap, self._sid(), guard,
            )
            await self._record(guard, trigger, "noop_deterministic_budget_exhausted",
                         reason + f" [deterministic inject cap of {cap} reached]",
                         content, related_tool)
            return False
        # role="user": vLLM rejects a non-leading system message.
        target.append({"role": "user", "content": "[INNER VOICE] " + content})
        f = GuardFire(guard, trigger, reason, content, related_tool, self.last_iteration)
        self.fires.append(f)
        logger.info("turn_guards: %s session=%s turn=%s reason=%s",
                    guard, self._sid(), self._tid(), reason)
        if self.persist_intervention_callback is not None:
            try:
                await self.persist_intervention_callback("inject", content, reason)
            except Exception as exc:  # noqa: BLE001
                logger.warning("turn_guards: breadcrumb failed: %s", exc)
        await self._record(guard, trigger, "inject", reason, content, related_tool)
        for cb in list(self.listeners):
            try:
                cb(f)
            except Exception as exc:  # noqa: BLE001 — a listener is not the guard
                logger.warning("turn_guards: listener raised: %s", exc)
        return True

    async def _record(self, guard: str, trigger: str, action: str, reason: str,
                      content: str, related_tool: str | None) -> None:
        sid = self._sid()
        if not sid:
            return
        tid = self._tid()
        try:
            from usage_store import record_inner_voice_observation

            # Off the event loop, like the observer's rows: a commit here
            # would stall the primary's stream for as long as the disk takes.
            await asyncio.to_thread(
                record_inner_voice_observation,
                session_id=sid, turn_id=tid, sequence_in_turn=self.next_sequence(),
                trigger=trigger, action=action, reason=reason, content=content,
                related_tool=related_tool, model=None, safeguard=guard,
            )
        except Exception as exc:  # noqa: BLE001 — the record is not the guard
            logger.warning("turn_guards: record failed: %s", exc)
        if action == "inject":
            try:
                from app import event_log

                event_log.log_event(
                    sid, "inner_voice.observer_injected",
                    {"trigger": trigger, "reason": reason, "content": content,
                     "related_tool": related_tool, "deterministic": True,
                     "guard": guard},
                    turn_id=tid or None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("turn_guards: event log failed: %s", exc)


def install_turn_guards(
    hooks: Any,
    *,
    session_id: str = "",
    turn_id: str = "",
    platform: str = "",
    source: str = "",
    chat_messages_handle: list[dict[str, Any]] | None = None,
    persist_intervention_callback: Callable[[str, str, str], Awaitable[None]] | None = None,
) -> TurnGuardState | None:
    """Install the deterministic senses on `hooks`. Idempotent per registry.

    A second call on the same registry fills in what the first did not know
    (the router installs before the breadcrumb callback exists; the observer
    installs with its own handle in tests) and returns the same state.
    Returns None when `inner_voice.turn_guards.enabled` is false.
    """
    existing = getattr(hooks, "turn_guards", None)
    if existing is not None:
        for attr, val in (
            ("session_id", session_id), ("turn_id", turn_id),
            ("platform", platform), ("source", source),
            ("chat_messages", chat_messages_handle),
            ("persist_intervention_callback", persist_intervention_callback),
        ):
            # `is None`/"" rather than truthiness: an empty list is a real
            # handle (the loop fills it), and must still be taken.
            if val is not None and val != "" and getattr(existing, attr) in (None, ""):
                setattr(existing, attr, val)
        return existing
    cfg = _cfg()
    if not cfg.get("enabled", True):
        return None
    state = TurnGuardState(
        hooks=hooks, session_id=session_id or "", turn_id=turn_id or "",
        platform=platform or "", source=source or "", cfg=cfg,
        chat_messages=chat_messages_handle,
        persist_intervention_callback=persist_intervention_callback,
    )
    window = int(cfg.get("repetition_window", _g.REPETITION_WINDOW))
    threshold = int(cfg.get("repetition_threshold", _g.REPETITION_THRESHOLD))
    exempt = _g.REPETITION_EXEMPT_TOOLS | frozenset(
        str(x) for x in (cfg.get("repetition_exempt_tools") or [])
    )
    ring_cap = max(_RING_FLOOR, window + 1)

    async def pretool_cb(input_data: dict[str, Any], _tid: Any, _ctx: Any) -> dict[str, Any]:
        tool = input_data.get("tool_name", "") or ""
        # `tool_summary` is deliberately not part of the signature: a caption
        # is the part of a call a looping model rewords each time.
        state.recent.append(_g.tool_call_signature(tool, input_data.get("tool_input") or {}))
        state.tool_calls_seen += 1
        if len(state.recent) > ring_cap:
            del state.recent[:-ring_cap]
        if not cfg.get("repetition", True):
            return {}
        # Compare only calls made SINCE the guard last spoke, so a second
        # fire needs a fresh cluster; ambient terms are judged over the whole
        # ring, which is longer than that slice.
        since = state.tool_calls_seen - state.repetition_baseline
        comparable = state.recent[-since:] if since > 0 else []
        rep = _g.repetition_verdict(
            comparable, window=window, threshold=threshold,
            ambient=_g.ubiquitous_identifiers(state.recent), exempt_tools=exempt,
        )
        if rep is not None:
            state.repetition_baseline = state.tool_calls_seen
            await state.fire(
                guard="repetition", trigger="pretool",
                reason=(f"deterministic: {rep.repeats + 1} near-identical {tool} "
                        f"calls for {', '.join(rep.shared_terms[:4])}"),
                content=_g.repetition_inject_content(rep), related_tool=tool,
            )
        return {}

    async def on_event_cb(evt: dict[str, Any]) -> None:
        etype = evt.get("type")
        if etype == "tool_result":
            await _on_tool_result(evt)
        elif etype == "assistant_message":
            await _on_assistant_message(evt)

    async def _on_tool_result(evt: dict[str, Any]) -> None:
        name = evt.get("name", "") or ""
        bare = name.rsplit("__", 1)[-1]
        is_error = bool(evt.get("is_error", False))
        if not is_error:
            if bare in _ROUND_OPEN_TOOLS:
                state.round_open = True
            elif bare in _ROUND_CLOSE_TOOLS:
                state.round_open = False
            if bare == "TodoWrite":
                state.todo_write_seen = True
        content = evt.get("content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        # A result that returned normally but reports the work did not
        # happen — `Task` → `[stopped: max_turns]` is 300 bytes, is_error
        # False. Once per tool name per turn: the second one says nothing new.
        if (
            cfg.get("failure_payload", True) and not is_error
            and bare not in state.failure_tools_fired
            and _g.looks_like_failure_payload(content)
        ):
            state.failure_tools_fired.add(bare)
            await state.fire(
                guard="failure_payload", trigger="tool_result",
                reason=f"deterministic: {bare} returned a failure payload",
                content=_g.failure_payload_content(bare), related_tool=name,
            )

    async def _on_assistant_message(evt: dict[str, Any]) -> None:
        state.last_iteration = int(evt.get("iteration", 0) or 0)
        if evt.get("tool_calls"):
            return
        text = evt.get("text", "") or ""
        # A tool call written as prose: the capability is missing, and more
        # text cannot supply it. Raised to a person, never injected.
        if cfg.get("capability_fault", True) and _g.looks_like_prose_tool_call(text):
            await _capability_fault(text)
            return
        # One inject per terminal iteration: the loop continues on any growth.
        if cfg.get("stall_rescue", True) and _g.is_terminal_stall(text):
            await state.fire(
                guard="stall_rescue", trigger="assistant_message",
                reason="deterministic: terminal stub-announce stall — forcing continuation",
                content=_g.stall_rescue_content(
                    unattended=state.unattended, round_open=state.round_open),
            )
            return
        if (
            cfg.get("round_open", True) and state.unattended and state.round_open
            and not state.round_gate_fired
        ):
            state.round_gate_fired = True
            await state.fire(
                guard="round_open", trigger="assistant_message",
                reason="deterministic: unattended turn ending with a round open",
                content=_g.UNATTENDED_ROUND_OPEN_CONTENT,
            )
            return
        if cfg.get("todo_gate", True) and state.todo_write_seen and not state.todo_gate_fired:
            open_items = [
                str(t.get("content") or "") for t in _load_todos(state._sid())
                if (t.get("status") or "") in ("pending", "in_progress")
                and t.get("content")
            ]
            if open_items:
                state.todo_gate_fired = True
                await state.fire(
                    guard="todo_gate", trigger="assistant_message",
                    reason=(f"deterministic: turn ending with {len(open_items)} "
                            f"open todo(s) it wrote this turn"),
                    content=_g.todo_gate_content(open_items),
                )

    async def _capability_fault(text: str) -> None:
        global _last_fault_announce
        sid = state._sid()
        logger.error(
            "turn_guards: capability fault — tool call written as prose "
            "session=%s turn=%s iter=%d (empty tool pool? see CLAUDE.md)",
            sid, state._tid(), state.last_iteration,
        )
        await state._record("capability_fault", "assistant_message",
                            "noop_capability_fault",
                            "deterministic: the primary wrote a tool call as text",
                            text[-600:], None)
        if sid:
            try:
                from app import event_log
                event_log.log_event(sid, "harness.capability_fault",
                                    {"iteration": state.last_iteration,
                                     "text_tail": text[-400:]},
                                    turn_id=state._tid() or None)
            except Exception:  # noqa: BLE001
                pass
        import time as _time
        now = _time.monotonic()
        if _last_fault_announce and now - _last_fault_announce < _FAULT_ANNOUNCE_COOLDOWN_S:
            return
        _last_fault_announce = now
        try:
            from app.prefix_miss import _announce
            await asyncio.to_thread(
                _announce, "Lloyd wrote a tool call as text",
                (f"Session {sid or '?'} ended an iteration with a tool call in "
                 f"its prose instead of dispatching it — usually an empty tool "
                 f"pool. Check `mcp_pool: failed to discover` in server.err."),
                False, "warning",
            )
        except Exception as exc:  # noqa: BLE001 — the toast is not the guard
            logger.warning("turn_guards: capability-fault announce failed: %s", exc)

    hooks.add_pre_tool_use(None, pretool_cb)
    hooks.add_on_event(on_event_cb)
    hooks.turn_guards = state
    return state


__all__ = ["GuardFire", "TurnGuardState", "install_turn_guards"]
