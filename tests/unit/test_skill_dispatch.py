"""Tests for dispatch-time skill delivery (#536) and its non-error outcome (#738).

The load-bearing cases are the two the acceptance check turns on: a triggered
call must NOT execute and must come back `is_error=False` (a deny-shaped
intercept would be booked into `tool_errors`, the number the feature is measured
with), and a skill that already reached the turn must not be delivered twice.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from app.harness import skill_dispatch as sd
from app.harness.hooks import HookRegistry
from app.harness.loop import _pre_dispatch
from app.harness.options import RunOptions

REPO = Path(__file__).resolve().parents[2]


def _run(coro):
    return asyncio.run(coro)


class _StubLoadedSet:
    """Enough of LoadedToolSet for `_pre_dispatch` to reach the hook walk.

    `enabled=False` short-circuits the ToolSearch intercept and the
    auto-load path, which is all of the function that sits between the top
    and PreToolUse.
    """

    enabled = False
    catalog: list = []
    loaded: set = set()

    def is_visible(self, name: str) -> bool:
        return True

    def mark_loaded(self, names) -> None:  # pragma: no cover - not reached
        raise AssertionError("stub must not be mutated")


def _tc(tool: str, args: dict) -> dict:
    return {
        "id": "call_test",
        "function": {"name": tool},
        "_args_dict": dict(args),
    }


def _install(hooks: HookRegistry, **kw):
    """Install with the deliverer forced on and bodies stubbed to a sentinel.

    Production skill bodies are ~10 KB; a test that greps for the sentinel says
    the same thing about plumbing without depending on the vault's contents.
    """
    rules = kw.pop("rules", sd.DISPATCH_RULES)
    ok = sd.install_skill_dispatch_hook(
        hooks, rules=rules, force_enabled=True, **kw
    )
    assert ok, "installer refused to install a forced-on deliverer"
    return hooks


@pytest.fixture(autouse=True)
def _clean_stats():
    sd.reset_stats()
    yield
    sd.reset_stats()


@pytest.fixture
def stub_body(monkeypatch):
    monkeypatch.setattr(sd, "skill_body", lambda name: f"BODY-OF[{name}]")
    return stub_body


# ---------------------------------------------------------------------------
# Rule matching: keyed on tool + argument pattern, not on topic
# ---------------------------------------------------------------------------


def test_rule_matches_argument_shape():
    cases = {
        # The 08-26 failure mode: reaching for the wrong extractor on a box
        # with no Node runtime.
        "python3 -c \"import subprocess; subprocess.run(['yt-dlp', url])\"": "youtube-transcript",
        "cd /tmp && python3 -m youtube_transcript_api --video X": "youtube-transcript",
        "page.evaluate(() => page.transcriptExtractor())": "youtube-transcript",
        # The restart that has to name the right unit.
        "supervisorctl -c cfg.conf restart lloyd-mc:lloyd-backend": "restart-lloyd",
        "systemctl --user restart lloyd-voice-mode.service": "voice-mode",
        "supervisorctl restart agent-tts": "voice-mode",
    }
    for command, expected in cases.items():
        rule = sd.match_rule("Bash", {"command": command}, rules=sd.DISPATCH_RULES)
        assert rule is not None, command
        assert rule.skill == expected, f"{command!r} -> {rule.skill}, wanted {expected}"
    print("test_rule_matches_argument_shape: OK")


def test_benign_dispatches_do_not_trigger():
    """The false-positive budget. A read is not a protocol event."""
    benign = [
        "supervisorctl -c cfg.conf status",           # health check, not a restart
        "supervisorctl pid lloyd-mc:lloyd-backend",
        "ls -la /tmp",
        "grep -Rn 'supervisorctl' ~/lloyd/docs/",     # mentions it, does not run it
        "echo hello",
        "systemctl --user status agent-tts",
        "supervisorctl -c cfg.conf status agent-tts",   # the health-check habit
        "systemctl --user is-active lloyd-voice-mode.service",
        "grep -Rn voice_mode.py ~/lloyd/",
        "curl -s http://127.0.0.1:8092/v1/status",
        "curl -s http://127.0.0.1:8096/v1/models",
    ]
    for command in benign:
        assert sd.match_rule("Bash", {"command": command}, rules=sd.DISPATCH_RULES) is None, command
    print("test_benign_dispatches_do_not_trigger: OK")


def test_rule_does_not_read_arguments_it_did_not_ask_for():
    """`fields` is a whitelist. A Write whose *content* discusses a rule's
    keywords must not be held — that would fire on half the vault."""
    assert sd.match_rule(
        "Write",
        {"file_path": "/tmp/note.md", "content": "run supervisorctl restart lloyd-mc:x"},
        rules=sd.DISPATCH_RULES,
    ) is None
    assert sd.match_rule("Read", {"file_path": "/tmp/yt-dlp.sh"}, rules=sd.DISPATCH_RULES) is None
    print("test_rule_does_not_read_arguments_it_did_not_ask_for: OK")


# ---------------------------------------------------------------------------
# Default-off
# ---------------------------------------------------------------------------


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(sd, "_config", lambda: {})
    hooks = HookRegistry()
    assert sd.enabled() is False
    assert sd.install_skill_dispatch_hook(hooks) is False
    assert hooks._pre == []

    monkeypatch.setattr(sd, "_config", lambda: {"enabled": False})
    assert sd.enabled() is False
    print("test_off_by_default: OK")


def test_on_only_for_named_skills(monkeypatch):
    monkeypatch.setattr(sd, "_config", lambda: {
        "enabled": True, "skills": ["restart-lloyd"],
    })
    rules = sd.rules_for(None)
    assert {r.skill for r in rules} == {"restart-lloyd"}
    hooks = HookRegistry()
    assert sd.install_skill_dispatch_hook(hooks) is True
    out = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    assert out == {}, "a rule outside the enabled set must not fire"
    print("test_on_only_for_named_skills: OK")


# ---------------------------------------------------------------------------
# The delivery itself
# ---------------------------------------------------------------------------


def test_delivery_is_a_non_error_tool_result(stub_body):
    """The crux of #738, and the reason the deliverer could not reuse deny.

    Before this change the only PreToolUse outcome beyond pass was `deny`, and
    `_pre_dispatch` answers a deny with `is_error=True` — which `autonomy.py`
    books into `tool_errors`. A protocol card that makes the fleet look sicker
    than it is would be measured out of existence.
    """
    hooks = _install(HookRegistry())
    evt = _run(_pre_dispatch(
        tc=_tc("Bash", {"command": "yt-dlp https://youtu.be/x"}),
        options=RunOptions(model="m", hooks=hooks),
        session_id="s", loaded_set=_StubLoadedSet(),
    ))
    assert evt is not None, "the drafted call should have been held back"
    assert evt["type"] == "tool_result", evt
    assert evt["is_error"] is False, "a delivery must not be booked as a tool error"
    assert "NOT executed" in evt["content"]
    assert "BODY-OF[youtube-transcript]" in evt["content"]
    assert evt["call_id"] == "call_test", evt
    print("test_delivery_is_a_non_error_tool_result: OK")


def test_non_matching_call_is_not_held(stub_body):
    hooks = _install(HookRegistry())
    evt = _run(_pre_dispatch(
        tc=_tc("Bash", {"command": "supervisorctl status"}),
        options=RunOptions(model="m", hooks=hooks),
        session_id="s", loaded_set=_StubLoadedSet(),
    ))
    assert evt is None, "a pass means dispatch proceeds, unchanged"
    print("test_non_matching_call_is_not_held: OK")


def test_deny_beats_deliver_regardless_of_registration_order(stub_body):
    """Safety first is not left to the installer's call order."""
    async def deny_bash(input_data, _tid, _ctx):
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "catastrophic",
        }}

    hooks = HookRegistry()
    _install(hooks)                       # deliverer FIRST
    hooks.add_pre_tool_use(None, deny_bash)   # safety second
    out = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    hso = out.get("hookSpecificOutput") or {}
    assert hso.get("permissionDecision") == "deny", hso
    assert not hso.get("skillDeliver")

    evt = _run(_pre_dispatch(
        tc=_tc("Bash", {"command": "yt-dlp https://youtu.be/x"}),
        options=RunOptions(model="m", hooks=hooks),
        session_id="s", loaded_set=_StubLoadedSet(),
    ))
    assert evt["is_error"] is True and "denied" in evt["content"], evt
    print("test_deny_beats_deliver_regardless_of_registration_order: OK")


# ---------------------------------------------------------------------------
# The "already injected this turn" guard
# ---------------------------------------------------------------------------


def test_already_injected_skill_is_not_delivered_again(stub_body):
    """prefetch.py put this body in the prompt before the turn started; a second
    copy in the same turn is pure prompt cost."""
    hooks = _install(HookRegistry(), already_injected={"youtube-transcript"})
    out = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    assert out == {}
    assert sd.STATS["skipped_already_injected"] >= 1
    print("test_already_injected_skill_is_not_delivered_again: OK")


def test_a_skill_is_held_at_most_once_per_turn(stub_body):
    """The re-issued call must actually run, or one protocol eats the turn
    budget repeating itself."""
    sd.reset_stats()
    hooks = _install(HookRegistry())
    first = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    second = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    assert first.get("hookSpecificOutput", {}).get("skillDeliver"), first
    assert second == {}, "a re-issued call must be dispatched, not held again"
    assert sd.STATS["delivered"] == 1
    assert sd.STATS["skipped_already_delivered"] == 1

    # The guard is per turn, not global: a fresh turn gets its own chance.
    fresh = _install(HookRegistry())
    out = _run(fresh.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    assert out.get("hookSpecificOutput", {}).get("skillDeliver"), out
    print("test_a_skill_is_held_at_most_once_per_turn: OK")


def test_injected_skill_names_parses_what_prefetch_renders():
    text = (
        '<context>\n<skill name="web-search-and-fetch" score="29.7">\nbody\n</skill>\n'
        '<skill name="youtube-transcript" score="4.1" excerpt="true">\nexcerpt\n</skill>\n'
        "- **269: @@ -268,4 @@** blah\n</context>"
    )
    assert sd.injected_skill_names(text) == {"web-search-and-fetch", "youtube-transcript"}
    assert sd.injected_skill_names("") == set()
    print("test_injected_skill_names_parses_what_prefetch_renders: OK")


# ---------------------------------------------------------------------------
# Size and safety of the payload
# ---------------------------------------------------------------------------


def test_body_is_capped(monkeypatch):
    monkeypatch.setattr(sd, "skill_body", lambda name: "x" * (sd.MAX_DELIVERY_CHARS * 3))
    hooks = _install(HookRegistry())
    out = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    content = out["hookSpecificOutput"]["skillDeliver"]["content"]
    assert len(content) <= sd.MAX_DELIVERY_CHARS + 600, len(content)
    assert "truncated" in content
    print("test_body_is_capped: OK")


def test_missing_body_never_swallows_the_call(monkeypatch):
    """A rule naming a skill that is not on disk must let the call through
    rather than answer it with an empty lesson."""
    monkeypatch.setattr(sd, "skill_body", lambda name: "")
    sd.reset_stats()
    hooks = _install(HookRegistry())
    out = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    assert out == {}
    assert sd.STATS["skipped_no_body"] == 1
    print("test_missing_body_never_swallows_the_call: OK")


def test_every_rule_names_a_skill_that_loads():
    """Guards against a rule whose skill was renamed, archived, or quarantined:
    that would silently degrade to no delivery."""
    for rule in sd.DISPATCH_RULES:
        body = sd.skill_body(rule.skill)
        assert body.strip(), f"rule {rule.label} names unloadable skill {rule.skill}"
    print("test_every_rule_names_a_skill_that_loads: OK")


# ---------------------------------------------------------------------------
# Wiring and the premise itself
# ---------------------------------------------------------------------------


def test_stream_route_installs_the_deliverer():
    """The route every worker source posts through
    (`workers/sources/_common.py:run_prompt_in_session` → /api/message/stream).
    Hooking only the interactive path would cover the half that already behaves
    best — the note in #536 is explicit that this must land where workers see it.
    """
    from app.routers import messages

    src = inspect.getsource(messages.post_message_stream)
    assert "install_skill_dispatch_hook(" in src
    assert "already_injected=injected_skill_names(" in src
    print("test_stream_route_installs_the_deliverer: OK")


def test_dispatch_path_references_skill_bodies():
    """The premise check from the item: this grep returned zero hits before
    #536. Pinned so a later refactor cannot quietly delete the deliverer and
    leave the item looking done."""
    hits = [
        p for p in (REPO / "app" / "harness").glob("*.py")
        if "SKILL.md" in p.read_text(encoding="utf-8", errors="replace")
        or "skill_body" in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert hits, "no SKILL.md/skill_body reference in app/harness/ — the deliverer is gone"
    print(f"test_dispatch_path_references_skill_bodies: OK ({len(hits)} file(s))")
