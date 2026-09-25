"""One builder for a turn's `RunOptions` (P13.4).

Four places used to build a turn by hand — the streaming chat POST
(`messages.post_message_stream`), the ambient builder (`build_ambient_turn`),
the memory-flush builder (`build_flush_turn`) and the voice setup
(`voice._voice_turn_setup`) — and a fifth, the sync `POST /api/message`, was
deleted with this change (P13.6). Every item that touched a turn's options had
to be threaded through all of them, and they drifted: the sync route had turn
guards the stream had in `_run_turn`, the voice turn had no hooks at all until
#1136. Each also parsed the session JSON again for every field it needed —
counted on base 8ac6f4a8, nine parses of a possibly multi-megabyte transcript
per chat POST and eleven per worker POST, on the event loop, before the meta
save's own read.

Two pieces:

* `SessionSnapshot.load` reads the session's small fields once
  (`sessions_io.read_session_fields`, cached per file version), and every
  decision below reads them from the snapshot rather than from disk.
* `build_turn_options(snapshot, body, kind)` returns a `TurnBuild`: the
  `RunOptions`, the system prompt, the hook registry, and the turn tail the
  caller appends to its user message. The kinds differ, and every difference is
  an explicit branch here rather than a difference between two copies:

  ======== ============================================================
  kind     what is different
  ======== ============================================================
  stream   body model/priority/budget/permission; context meter; final
           schema and effect scope (worker sessions only); skill dispatch
           (armed after the prefetch, `arm_skill_dispatch`); action review
  ambient  config budget, priority 0, `session_id` set; no skill dispatch
           (#750: a decide-and-stop turn)
  flush    memory tools only, tool search off, priority 1, no refresher,
           no Inner Voice registry, no turn tail
  voice    safety floor only — no grant gate, no platform (the prompt and
           the memory snapshot are built as a chat's), `extra_body` from
           the voice config, no surface, goal left out of the prompt
  ======== ============================================================

`tests/test_turn_options.py` holds every kind to a fixture recorded from the
four hand-built sites before this module existed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app import memory_snapshot as _memory_snapshot
from app import prompt_layout as _prompt_layout
from app.config import CONFIG, _get_model_env, _resolve_model_name
from app.harness import (
    HookRegistry,
    RunOptions,
    install_default_safety_hook,
    install_skill_dispatch_hook,
)
from app.harness.context_meter import ContextMeter, context_window_for
from app.harness.policy import install_policy_hook
from app.harness.skill_dispatch import injected_skill_names
from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs, _get_mcp_servers
from app.paths import SESSIONS_DIR
from app.sessions_io import read_session_fields
from prompt_builder import build_system_prompt

TurnKind = Literal["stream", "ambient", "flush", "voice"]
KINDS: tuple[str, ...] = ("stream", "ambient", "flush", "voice")

_DEFAULT_BASE_URL = "http://127.0.0.1:8096"


@dataclass(frozen=True)
class SessionSnapshot:
    """A session's small fields, read once for the turn being built.

    A missing or unreadable file is a brand-new chat: every field empty,
    `platform` read as `mission-control` (see `messages._session_identity`).
    `raw_platform` keeps the unset case distinguishable, because
    `_final_schema_for` logs what the file actually said.
    """

    session_id: str
    path: Path
    exists: bool = False
    model: str = ""
    raw_platform: str = ""
    source: str = ""
    todos: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    goal: dict = field(default_factory=dict)
    inner_voice: bool = False
    inner_voice_evaluate_user_turns: bool = False
    user_turns: int = 0

    @classmethod
    def load(cls, session_id: str, sessions_dir: Path | None = None) -> "SessionSnapshot":
        """One read of `<sessions_dir>/<session_id>.json` (none on a cache hit).

        `sessions_dir` defaults to this module's `SESSIONS_DIR`; the routers
        pass their own, which is what a test that points the router at a
        scratch directory patches.
        """
        base = Path(sessions_dir) if sessions_dir is not None else SESSIONS_DIR
        path = base / f"{session_id}.json"
        if not session_id:
            return cls(session_id=session_id, path=path)
        try:
            fields = read_session_fields(path)
        except Exception:  # noqa: BLE001 — unreadable reads as a new chat
            fields = None
        if not fields:
            return cls(session_id=session_id, path=path)
        return cls(
            session_id=session_id, path=path, exists=True,
            model=fields["model"], raw_platform=fields["platform"],
            source=fields["source"], todos=fields["todos"], plan=fields["plan"],
            goal=fields["goal"], inner_voice=fields["inner_voice"],
            # Meaningless without the master switch (`_session_iv_flags`).
            inner_voice_evaluate_user_turns=(
                fields["inner_voice"] and fields["inner_voice_evaluate_user_turns"]),
            user_turns=fields["user_turns"],
        )

    @property
    def platform(self) -> str:
        return self.raw_platform or "mission-control"

    @property
    def identity(self) -> tuple[str, str]:
        """`(platform, source)`, as `messages._session_identity` answers it."""
        return (self.platform, self.source) if self.exists else ("mission-control", "")

    @property
    def plan_mode(self) -> bool:
        return bool(self.plan.get("plan_mode"))

    def plan_mode_live(self) -> bool:
        """Plan mode as the file says NOW — `ExitPlanMode` inside the turn
        unblocks writes at the next iteration. Cached per file version, so the
        per-iteration refresher costs a `stat` unless the file changed."""
        try:
            fields = read_session_fields(self.path)
        except Exception:  # noqa: BLE001
            return False
        return bool(((fields or {}).get("plan") or {}).get("plan_mode"))

    @classmethod
    async def aload(cls, session_id: str,
                    sessions_dir: Path | None = None) -> "SessionSnapshot":
        """`load`, off the event loop: the one parse a POST pays is of a file
        its own previous turn just rewrote, so it is rarely a cache hit."""
        return await asyncio.to_thread(cls.load, session_id, sessions_dir)


@dataclass
class TurnBuild:
    """What `build_turn_options` hands back. `options.hooks is hooks`."""

    kind: str
    model: str
    options: RunOptions
    system_prompt: str
    hooks: HookRegistry
    #: Appended to the user message (`prompt_layout.append_turn_tail`); "" for
    #: a flush, which needs neither the session state nor the memory delta.
    turn_tail: str = ""


def resolve_model(snapshot: SessionSnapshot, body: dict | None = None) -> str:
    """The request's model, else the session's, else the default — resolved."""
    model = str((body or {}).get("model") or "") or snapshot.model \
        or CONFIG.get("model", {}).get("default", "")
    return _resolve_model_name(model)


def build_turn_options(snapshot: SessionSnapshot, body: dict, kind: TurnKind, *,
                       text: str = "") -> TurnBuild:
    """Build one turn's options. `body` is the request body for `stream` and a
    fresh dict otherwise; the grant and automod bans are written back into it
    (`extra_disallowed`), because the per-iteration refresher re-reads it.

    `text` is the user's (or worker's) own prompt, which the action reviewer is
    handed — never the prefetched context.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown turn kind {kind!r}")
    from app.routers import messages as _m  # the helpers live beside their tests

    session_id = snapshot.session_id
    identity = snapshot.identity
    platform = identity[0]
    model = resolve_model(snapshot, body if kind == "stream" else None)
    model_env = _get_model_env(model)
    agent_cfg = CONFIG.get("agent", {})

    # ── The system prompt and the turn tail ─────────────────────────────
    # Voice has always built its prompt as a chat's (no platform, no goal);
    # kept as it was, and named here rather than left to a second copy.
    if kind == "voice":
        frozen_mem, memory_note = _memory_snapshot.frozen_memories(session_id)
        system_prompt = build_system_prompt(
            session_id=session_id, todos=snapshot.todos, plan=snapshot.plan,
            **_prompt_layout.mem_kwargs(frozen_mem),
        )
        turn_tail = _prompt_layout.turn_tail(
            snapshot.todos, snapshot.plan, None, memory_note)
    else:
        frozen_mem, memory_note = _memory_snapshot.frozen_memories(session_id, platform)
        system_prompt = build_system_prompt(
            todos=snapshot.todos, plan=snapshot.plan, goal=snapshot.goal,
            session_id=session_id, platform=platform,
            **_prompt_layout.mem_kwargs(frozen_mem),
        )
        # A flush needs neither the session state nor the memory delta.
        turn_tail = "" if kind == "flush" else _prompt_layout.turn_tail(
            snapshot.todos, snapshot.plan, snapshot.goal, memory_note)

    # ── Who the turn is: the authority gate and the tool bans ──────────
    # Armed by the session's own platform, so a caller cannot forget (#534,
    # #709). Voice arms neither: a spoken turn is always a person's.
    grant_scope = ""
    if kind != "voice":
        grant_scope = _m._authority_scope_for(session_id, body, identity=identity)
        if grant_scope:
            _m._ban_grant_minting(body)
        _m._ban_automod_for_workers(body, session_id, identity=identity)

    def _refresh_disallowed() -> list[str]:
        """Per-iteration refresher: static config + the plan-mode actuator
        block as the file says now + the bans written into `body`."""
        return _get_disallowed_tools(plan_mode=snapshot.plan_mode_live()) + (
            body.get("extra_disallowed") or [])

    # ── Hooks ───────────────────────────────────────────────────────────
    # The Inner Voice registry is empty at build time (the observer installs
    # its callbacks per turn); a flush and a spoken turn get a plain one.
    if snapshot.inner_voice and kind in ("stream", "ambient"):
        from app.routers._messages_inner_voice import build_iv_hook_registry
        hooks = build_iv_hook_registry(session_id)
    else:
        hooks = HookRegistry()
    install_default_safety_hook(hooks)
    if grant_scope:
        install_policy_hook(hooks, scope=grant_scope)
    if kind == "stream":
        # P10: the action reviewer, shadow only, worker turns only.
        _m._install_action_review(hooks, platform=platform, text=text,
                                  source=identity[1], session_id=session_id)

    # ── Budget, priority ────────────────────────────────────────────────
    if kind == "stream":
        max_turns = _m._turn_budget(body, platform=platform)
        priority = _m._clamp_priority(body.get("priority", 0))
    elif kind == "flush":
        from app import memory_flush as _mf

        cfg = _mf.flush_cfg()
        max_turns = int(cfg.get("max_turns") or 6)
        priority = 1
    else:
        max_turns = agent_cfg.get("max_turns", 60)
        priority = 0

    harness_kwargs = dict(_get_harness_kwargs())
    extra: dict[str, Any] = {}
    if kind == "flush":
        # A flush that could reach Bash or the vault would be a second,
        # unobserved chat turn: memory tools only, tool search off.
        harness_kwargs["tool_search_enabled"] = False
        extra["allowed_tools"] = list(cfg.get("tools") or _mf.DEFAULT_TOOLS)
    else:
        extra["disallowed_tools_refresh"] = _refresh_disallowed
    if kind == "voice":
        from app.routers.voice import _voice_extra_body
        extra["extra_body"] = _voice_extra_body()
    else:
        extra["surface"] = _m._tool_surface(platform)
        extra["grant_scope"] = grant_scope  # D4: what a Task child inherits
    if kind != "stream":
        # The stream path's is stamped in `_run_turn`, with the cancel event.
        extra["session_id"] = session_id

    options = RunOptions(
        model=model,
        base_url=model_env.get("ANTHROPIC_BASE_URL", _DEFAULT_BASE_URL),
        system_prompt=system_prompt,
        max_turns=max_turns,
        mcp_servers=_get_mcp_servers(),
        disallowed_tools=_get_disallowed_tools(plan_mode=snapshot.plan_mode)
        + (body.get("extra_disallowed") or []),
        hooks=hooks,
        priority=priority,
        **extra,
        **harness_kwargs,
    )

    if kind == "stream":
        # One meter, three readers: the loop relieves against it, the
        # `<context>` anchor reports it to the model, and the Inner Voice
        # observer stops nudging a turn that has no room to act on a nudge.
        options.context_meter = ContextMeter(context_window_for(model))
        # A session with no file yet is a chat: no schema, and nothing logged.
        final_schema = _m._final_schema_for(
            session_id, body, platform=snapshot.raw_platform) if snapshot.exists else None
        if final_schema is not None:
            options.final_schema = final_schema
            options.final_schema_prompt = str(body.get("final_schema_prompt") or "")
        options.effect_scope = _m._effect_scope_for(session_id, body, identity=identity)

    return TurnBuild(kind=kind, model=model, options=options,
                     system_prompt=system_prompt, hooks=hooks, turn_tail=turn_tail)


def arm_skill_dispatch(build: TurnBuild, prefetched_text: str) -> None:
    """#536 — dispatch-time skill delivery, stream turns only.

    After the prefetch, because `already_injected` is the set the prefetch put
    in this turn's `<context>`, so the same body cannot land twice. Installed
    after the safety floor and the grant gate: a deny beats a deliver whatever
    the order, but the walk should read in the order that matters. One install
    covers the workers too — every worker source posts through the stream
    route. Ambient (#750) and voice turns are short decide-and-stop turns and
    never get it; the flush is memory tools only.
    """
    if build.kind != "stream":
        return
    install_skill_dispatch_hook(
        build.hooks, already_injected=injected_skill_names(prefetched_text),
    )
