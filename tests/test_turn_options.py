"""P13.4 — one options builder, and the turns it builds are the turns HEAD built.

Four places used to hand-build a turn's `RunOptions`: the streaming chat POST,
the ambient builder, the memory-flush builder and the voice setup (the sync
`POST /api/message` was a fifth and is deleted, P13.6). They drifted, and each
one parsed the session JSON again for every field it needed — nine parses per
POST on the event loop. `app/routers/turn_options.py` is the one builder now.

The refactor claims to change nothing a turn runs under, so the claim is
measured, not asserted: `tests/fixtures/turn_options_head.json` was recorded
by this file (`LLOYD_RECORD_TURN_OPTIONS=1`) against the tree BEFORE the
builder existed (base 8ac6f4a8), for every kind across five session shapes, and
the table test compares what the builder produces now against it field for
field — every `RunOptions` field, the system prompt's inputs, the prompt tail,
the installed hooks in order, the refreshed disallowed list. D13 removed
four fields nothing read (`env`, `permission_mode`, `history`,
`context_relief_send_max_tokens_reservation`) from `RunOptions`, and from
the fixture by deleting those keys — every other recorded value is as it was.

Everything that reads config or the live vault is replaced by a deterministic
stand-in that echoes its arguments, so the fixture pins what each site PASSED,
not what today's config.yaml happens to say. Stand-ins are patched on every
module that might look the name up (the router before, the builder after).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path

import pytest

from app.harness import RunOptions

FIXTURE = Path(__file__).parent / "fixtures" / "turn_options_head.json"


# ── The session shapes the table walks ────────────────────────────────────

def _user_rows(n: int) -> list[dict]:
    return [{"role": "user", "source": "user", "content": f"m{i}"} for i in range(n)]


SESSIONS: dict[str, dict | None] = {
    # A plain chat with todos and exactly 20 user turns, so the memory nudge fires.
    "chat": {"model": "primary", "platform": "mission-control",
             "todos": [{"content": "t1", "status": "pending"}],
             "messages": _user_rows(20)},
    # Plan mode on, a goal, Inner Voice on.
    "chat_plan": {"model": "secondary", "platform": "mission-control",
                  "todos": [{"content": "t2", "status": "in_progress"}],
                  "plan": {"plan_mode": True, "plan_md_path": "/p.md"},
                  "goal": {"text": "ship it"},
                  "inner_voice": True, "inner_voice_evaluate_user_turns": True,
                  "messages": _user_rows(3)},
    # A worker whose source is not the loop: automod banned, grant gate armed.
    "worker": {"model": "primary", "platform": "worker", "source": "autotriage",
               "messages": []},
    # The loop's own worker keeps the automod tools.
    "autocode": {"model": "", "platform": "worker", "source": "autocode",
                 "inner_voice": True, "messages": []},
    # No file yet: a brand-new chat.
    "new": None,
}

BODIES: dict[str, dict] = {
    "chat": {"text": "hello there"},
    "chat_plan": {"text": "plan it", "think": "high", "priority": 2,
                  "permission_mode": "default", "extra_disallowed": ["X"],
                  "model": "override-model"},
    "worker": {"text": "triage #1", "platform": "worker", "priority": 1,
               "effect_scope": "item:autotriage:7", "max_turns": 250,
               "deadline_seconds": 900, "extra_disallowed": ["http_request"],
               "final_schema": {"type": "object"},
               "final_schema_prompt": "restate"},
    "autocode": {"text": "implement #2", "grant_scope": "autonomy-task:9",
                 "effect_scope": "item:autocode:3", "final_schema": {"type": "object"}},
    "new": {"text": "first message"},
}

KINDS = ("stream", "ambient", "flush", "voice")


# ── Stand-ins ─────────────────────────────────────────────────────────────

def _modules():
    import app.routers.messages as messages
    import app.routers.voice as voice
    mods = [messages, voice]
    try:
        import app.routers.turn_options as turn_options
        mods.append(turn_options)
    except ImportError:
        pass
    return mods


def _tag(name, **cells):
    """A named callback whose closure carries `cells`, so `_describe_hooks`
    can record what the installer was handed."""
    frozen = {k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
              for k, v in cells.items()}

    async def cb(*a, **k):  # pragma: no cover — never called
        return frozen

    cb.__qualname__ = name
    cb._recorded = frozen
    return cb


def _fake_safety(hooks):
    hooks.add_pre_tool_use(None, _tag("safety"), fail_closed=True)


def _fake_policy(hooks, *, scope=None, **_):
    hooks.add_pre_tool_use(None, _tag("policy", scope=scope), fail_closed=True)


def _fake_skill_dispatch(hooks, *, already_injected=None, **_):
    hooks.add_pre_tool_use(None, _tag("skill_dispatch",
                                      already_injected=already_injected or set()))


def _fake_action_review(hooks, *, user_prompt, source="", session_id="", **_):
    hooks.add_on_event(_tag("action_review", user_prompt=user_prompt,
                            source=source, session_id=session_id))
    return object()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Deterministic stand-ins for everything config- or vault-shaped."""
    import app.component_manifest as component_manifest
    import app.memory_flush as memory_flush
    import app.memory_snapshot as memory_snapshot
    import app.prompt_layout as prompt_layout
    import app.routers._messages_inner_voice as iv
    import app.routers.automod as automod
    import app.sessions_io as sessions_io
    import app.skill_embed as skill_embed
    from app.config import CONFIG

    enqueued: list = []

    def everywhere(name, value):
        for mod in _modules():
            monkeypatch.setattr(mod, name, value, raising=False)

    async def _prefetch(text, session_id="", plan_mode=False, **_):
        return f"PF[{text}|plan_mode={plan_mode}]"

    async def _noop_async(*a, **k):
        return None

    async def _enqueue(session_id, turn, *, consumer_factory=None):
        enqueued.append(turn)
        return {}

    import app.paths as paths

    # `app.paths` too: HEAD's voice refresher imported the name from there at
    # call time, so a fixture that left it live recorded a plan mode read from
    # the wrong directory.
    for mod in (sessions_io, iv, paths):
        monkeypatch.setattr(mod, "SESSIONS_DIR", tmp_path)
    everywhere("SESSIONS_DIR", tmp_path)
    everywhere("build_system_prompt",
               lambda **kw: "SYS" + json.dumps(kw, sort_keys=True, default=str))
    everywhere("prefetch_context_async", _prefetch)
    everywhere("log_turn_prompt_budget", lambda *a, **k: None)
    everywhere("_save_session_meta", _noop_async)
    everywhere("enqueue_turn", _enqueue)
    everywhere("set_last_user_session", lambda sid: None)
    everywhere("_get_mcp_servers", lambda: {"lloyd-mcp": {"url": "u"}})
    everywhere("_get_disallowed_tools",
               lambda plan_mode=False: ["D0"] + (["PLAN_BLOCK"] if plan_mode else []))
    everywhere("_get_harness_kwargs",
               lambda: {"tool_search_enabled": True, "preserve_thinking_iterations": 3})
    everywhere("_get_model_env",
               lambda m: {"ANTHROPIC_BASE_URL": f"http://engine/{m}"})
    everywhere("_resolve_model_name", lambda m: f"R({m})")
    everywhere("context_window_for", lambda m: 1000 + len(m))
    everywhere("install_default_safety_hook", _fake_safety)
    everywhere("install_policy_hook", _fake_policy)
    everywhere("install_skill_dispatch_hook", _fake_skill_dispatch)
    everywhere("install_action_review_hook", _fake_action_review)
    everywhere("ambient_clock_stamp", lambda ts: "CLOCK")
    everywhere("voice_room_prefix", lambda sid: "VR:")
    everywhere("_voice_extra_body", lambda: {"voice": True})

    monkeypatch.setattr(memory_snapshot, "frozen_memories",
                        lambda sid, platform="": ("FROZEN-MEM", f"NOTE({platform})"))
    monkeypatch.setattr(prompt_layout, "mem_kwargs",
                        lambda m: {"memories_text": m} if m else {})
    monkeypatch.setattr(
        prompt_layout, "turn_tail",
        lambda todos=None, plan=None, goal=None, memory_note="":
            "TAIL" + json.dumps([todos, plan, goal, memory_note], sort_keys=True))
    monkeypatch.setattr(prompt_layout, "append_turn_tail",
                        lambda text, tail: f"{text}||{tail}" if tail else text)
    monkeypatch.setattr(component_manifest, "note_prefetch", lambda *a, **k: None)
    monkeypatch.setattr(skill_embed, "record_context_skills", lambda *a, **k: None)
    monkeypatch.setattr(memory_flush, "flush_cfg",
                        lambda: {"max_turns": 4, "tools": ["memory_add"]})
    monkeypatch.setattr(automod, "drain_active", lambda: False)
    monkeypatch.setitem(CONFIG, "agent", {
        "max_turns": 60, "max_turns_ceiling": 120, "max_turns_ceiling_worker": 200,
        "permission_mode": "bypassPermissions"})
    monkeypatch.setitem(CONFIG, "model", {"default": "primary"})

    def write(shape: str) -> str:
        sid = f"20260924_120000_{shape}"
        doc = SESSIONS[shape]
        if doc is not None:
            (tmp_path / f"{sid}.json").write_text(
                json.dumps({"session_id": sid, **doc}, indent=2))
        return sid

    return {"dir": tmp_path, "enqueued": enqueued, "write": write}


# ── Recording ─────────────────────────────────────────────────────────────

class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _describe_hooks(hooks) -> dict | None:
    if hooks is None:
        return None

    def one(cb):
        return {"name": getattr(cb, "__qualname__", repr(cb)),
                "args": getattr(cb, "_recorded", None)}

    return {
        "pre": [{"matcher": m, "fail_closed": fc, **one(cb)}
                for (m, cb), fc in zip(hooks._pre, hooks._pre_fail_closed)],
        "post": [one(cb) for cb in hooks._post],
        "post_failure": [one(cb) for cb in hooks._post_failure],
        "on_event": [one(cb) for cb in hooks._on_event],
    }


def _ser(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _ser(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ser(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_ser(v) for v in value)
    return f"<{type(value).__name__}>"


def _describe_options(opts: RunOptions) -> dict:
    out: dict = {}
    for f in dataclasses.fields(RunOptions):
        v = getattr(opts, f.name)
        if f.name == "disallowed_tools_refresh":
            out[f.name] = None if v is None else {"refreshed": list(v())}
        elif f.name == "hooks":
            out[f.name] = _describe_hooks(v)
        elif f.name == "context_meter":
            out[f.name] = None if v is None else {"window": v.window}
        elif f.name == "system_prompt":
            out[f.name] = v
        else:
            out[f.name] = _ser(v)
    return out


def _describe_payload(payload: dict) -> dict:
    return {k: (_ser(v) if k != "meta_path" else Path(v).name)
            for k, v in sorted(payload.items()) if k != "options"}


def capture(world, kind: str, shape: str) -> dict | str:
    """What `kind` builds for session shape `shape`, through the public entry."""
    import app.routers.messages as messages
    import app.routers.voice as voice

    sid = world["write"](shape)
    body = dict(BODIES[shape])
    if kind == "stream":
        body["session_id"] = sid
        world["enqueued"].clear()
        asyncio.run(messages.post_message_stream(_Req(body)))
        (turn,) = world["enqueued"]
        opts, payload = turn.payload["options"], turn.payload
        extra = {"body_extra_disallowed": body.get("extra_disallowed")}
    elif kind == "ambient":
        if SESSIONS[shape] is None:
            return "404"
        turn = asyncio.run(messages.build_ambient_turn(
            sid, "signal text", dedup_key="dk", priority="urgent",
            source="calendar", summary="sum"))
        opts, payload, extra = turn.payload["options"], turn.payload, {}
    elif kind == "flush":
        if SESSIONS[shape] is None:
            return "404"
        turn = asyncio.run(messages.build_flush_turn(sid))
        opts, payload, extra = turn.payload["options"], turn.payload, {}
    elif kind == "voice":
        setup = voice._voice_turn_setup(sid)
        opts = setup["options"]
        payload = {k: v for k, v in setup.items() if k != "options"}
        extra = {}
    else:  # pragma: no cover
        raise AssertionError(kind)
    return {"options": _describe_options(opts),
            "payload": _describe_payload(payload), **extra}


def _record_all(world) -> dict:
    return {f"{kind}/{shape}": capture(world, kind, shape)
            for kind in KINDS for shape in SESSIONS}


# ── The table test ────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.environ.get("LLOYD_RECORD_TURN_OPTIONS"),
                    reason="recording mode only (LLOYD_RECORD_TURN_OPTIONS=1)")
def test_record_the_fixture(world):
    FIXTURE.write_text(json.dumps(_record_all(world), indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", list(SESSIONS))
def test_each_kind_builds_what_head_built(world, kind, shape):
    """Field for field against the fixture recorded before the builder
    existed. A `RunOptions` field added after the recording must still hold
    its dataclass default on every kind — a new knob a builder sets is a
    decision, and it re-records the fixture on purpose."""
    want = json.loads(FIXTURE.read_text())[f"{kind}/{shape}"]
    got = capture(world, kind, shape)
    if isinstance(want, str):
        assert got == want
        return
    defaults = _describe_options(RunOptions(model=""))
    for name, value in got["options"].items():
        expected = want["options"].get(name, defaults[name])
        assert value == expected, (kind, shape, name)
    assert set(want["options"]) <= set(got["options"])
    assert got["payload"] == want["payload"], (kind, shape)
    assert {k: v for k, v in got.items() if k not in ("options", "payload")} == \
        {k: v for k, v in want.items() if k not in ("options", "payload")}


# ── One parse per POST ────────────────────────────────────────────────────

def _count_session_reads(monkeypatch, sid: str) -> list[str]:
    """Every `read_text` of `<sid>.json`, with the function that asked."""
    import sys

    reads: list[str] = []
    real = Path.read_text

    def spy(self, *a, **k):
        if self.name == f"{sid}.json":
            reads.append(sys._getframe(1).f_code.co_name)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    return reads


@pytest.mark.parametrize("shape", ["chat", "worker"])
def test_a_post_parses_the_session_json_once(world, monkeypatch, shape):
    """P13.4's claim, counted. Before the builder the stream POST read the
    transcript for the model, the todos, the plan, the goal, the turn count,
    the identity (five times, through the gate helpers), the final schema's
    platform and the Inner Voice flag — eleven parses of a 1 MB file on the
    event loop for a worker turn. Now: one read for the whole build, and the
    meta save's own read-modify-write, which has to re-read under its lock and
    runs off the loop (P13.5)."""
    import app.routers.messages as messages
    import app.sessions_io as sessions_io

    monkeypatch.setattr(messages, "_save_session_meta", sessions_io._save_session_meta)
    monkeypatch.setattr(sessions_io, "_FIELDS_CACHE", {})
    sid = world["write"](shape)
    reads = _count_session_reads(monkeypatch, sid)
    body = dict(BODIES[shape], session_id=sid)
    asyncio.run(messages.post_message_stream(_Req(body)))
    assert reads == ["read_session_fields", "_write"], reads

    # The same file version is not parsed again: a second build is a stat.
    # (With the racy window off: the file here is milliseconds old, and a file
    # that young is never cached — see the test below.)
    monkeypatch.setattr(sessions_io, "_FIELDS_RACY_NS", 0)
    reads.clear()
    from app.routers.turn_options import SessionSnapshot
    SessionSnapshot.load(sid, world["dir"])
    SessionSnapshot.load(sid, world["dir"]).plan_mode_live()
    assert reads == ["read_session_fields"], reads  # the save above moved the file
    reads.clear()
    SessionSnapshot.load(sid, world["dir"]).plan_mode_live()
    assert reads == []


def test_a_cached_snapshot_cannot_be_poisoned_by_its_reader(world, monkeypatch):
    import app.sessions_io as sessions_io
    from app.routers.turn_options import SessionSnapshot

    monkeypatch.setattr(sessions_io, "_FIELDS_RACY_NS", 0)

    sid = world["write"]("chat")
    SessionSnapshot.load(sid, world["dir"]).todos.append({"content": "leak"})
    assert SessionSnapshot.load(sid, world["dir"]).todos == SESSIONS["chat"]["todos"]


def test_a_rewritten_file_is_read_again(world):
    """The cache key is the file's version, and every writer replaces the file."""
    import app.sessions_io as sessions_io
    from app.routers.turn_options import SessionSnapshot

    sid = world["write"]("chat")
    assert not SessionSnapshot.load(sid, world["dir"]).plan_mode
    asyncio.run(sessions_io.mutate_session(
        sid, lambda d: d.__setitem__("plan", {"plan_mode": True}),
        path=world["dir"] / f"{sid}.json"))
    snap = SessionSnapshot.load(sid, world["dir"])
    assert snap.plan_mode and snap.plan_mode_live()


def test_a_racily_fresh_file_is_never_cached(world, monkeypatch):
    """File times tick at the kernel's coarse clock, so an in-place rewrite of
    the same size inside one tick keeps every part of the stat key. A file
    younger than the racy window is parsed and not cached, so a rewrite like
    that — here a plain `write_text` on the same inode, the case an atomic
    writer never produces — is still seen."""
    import app.sessions_io as sessions_io
    from app.routers.turn_options import SessionSnapshot

    monkeypatch.setattr(sessions_io, "_FIELDS_CACHE", {})
    sid = world["write"]("chat")
    path = world["dir"] / f"{sid}.json"
    assert not SessionSnapshot.load(sid, world["dir"]).plan_mode
    assert str(path) not in sessions_io._FIELDS_CACHE
    doc = json.loads(path.read_text())
    doc["plan"] = {"plan_mode": True}
    text = json.dumps(doc, indent=2)
    path.write_text(text)
    assert SessionSnapshot.load(sid, world["dir"]).plan_mode
