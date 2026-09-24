"""Tests for dispatch-time skill delivery (#536) and its non-error outcome (#738).

The load-bearing cases are the two the acceptance check turns on: a triggered
call must NOT execute and must come back `is_error=False` (a deny-shaped
intercept would be booked into `tool_errors`, the number the feature is measured
with), and a skill that already reached the turn must not be delivered twice.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path

import pytest

from app.harness import skill_dispatch as sd
from app.harness.hooks import HookRegistry
from app.harness.loop import _pre_dispatch
from app.harness.options import RunOptions
from app.harness.safety import install_default_safety_hook

REPO = Path(__file__).resolve().parents[2]

# The voice-mode verb/unit gap as it shipped in #536: unbounded, so it reached
# across shell statement separators (#751). The negatives below are pinned
# against THIS and only this, so `test_benign_dispatches_do_not_trigger` can
# never rot into a list of commands that no rule ever wanted to hold — a
# negative that the old rule also passed pins nothing.
_LEGACY_VOICE_PATTERNS = (
    re.compile(
        r"\b(?:restart|stop|start|enable|disable|signal|kill)\b[^\n]*"
        r"\b(?:agent-(?:tts|livekit-server)|lloyd-voice-mode\.service)\b"
    ),
    re.compile(
        r"\b(?:agent-(?:tts|livekit-server)|lloyd-voice-mode\.service)\b[^\n]*"
        r"\b(?:restart|stop|start|enable|disable)\b"
    ),
)

# Diagnostic reads lifted byte-for-byte out of the session transcripts, each one
# held by the old rule because an `echo` header naming a control verb sat in a
# different statement from the unit name the read was aimed at. The session file
# is named per entry so a later reader can re-harvest them.
TRANSCRIPT_READS_HELD_BY_THE_OLD_RULE: tuple[tuple[str, str], ...] = (
    # `tail -25 agent-livekit-server.err` under an
    # `echo "=== livekit rtc/bind lines since restart ==="` header.
    ("20260907_220430_iv15a0.json",
     'cd ~/lloyd/agent-services/logs; echo "=== livekit.err tail 25 ==="; tail -25 agent-livekit-server.err 2>/dev/null; echo; echo "=== livekit rtc/bind lines since restart ==="; grep -iE "rtc|udp|node_ip|Starting|error|port" agent-livekit-server.log 2>/dev/null | tail -12; echo; echo "=== voice worker (agent-worker) tail 15 ==="; tail -15 lloyd-agent-worker.err 2>/dev/null || tail -15 lloyd-agent-worker.log 2>/dev/null; echo; echo "=== compute apps ==="; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv 2>&1 | head -10; echo; echo "=== GPU uuid map ==="; nvidia-smi --query-gpu=index,uuid --format=csv,noheader'),
    # `cat bin/start-qwen3-tts.sh` — the verb is inside a filename.
    ("20260907_220430_iv15a0.json",
     'cd ~/lloyd/agent-services; echo "=== TTS start script ==="; cat bin/start-qwen3-tts.sh 2>/dev/null; echo; echo "=== TTS supervisor program block ==="; awk \'/\\[program:agent-tts\\]/,/^\\[/\' supervisor/*.conf 2>/dev/null | head -30; echo; echo "=== TTS env in running proc ==="; tr \'\\0\' \'\\n\' < /proc/1937169/environ 2>/dev/null | grep -iE \'TTS|CUDA|LAZY|MODEL|PORT\' | head -20'),
    # `curl -s http://localhost:8090/health | python3 -m json.tool`, multi-line,
    # held by the last line's `since restart ===` header reaching `agent-tts.log`.
    ("20260907_220430_iv15a0.json",
     'echo "=== TTS /health device block ==="; curl -s -m 10 http://localhost:8090/health | python3 -m json.tool\necho; echo "=== which physical GPU per pid ==="; nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader\necho "--- gpu uuid map ---"; nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv,noheader\necho; echo "=== TTS proc env + CPU/GPU split ==="; TTS_PID=$(pgrep -f "uvicorn api.main:app" | head -1); echo "pid=$TTS_PID"\ntr \'\\0\' \'\\n\' < /proc/$TTS_PID/environ | grep -E "CUDA_VISIBLE|TTS_|TTS_BACKEND|OMP_NUM|CUDA_MPS"\necho "--- cpu% and thread count ---"; ps -o pid,%cpu,nlwp,rss,etime -p $TTS_PID\necho; echo "=== TTS log: device / compile lines since restart ==="; grep -iE "device|cuda|cpu|compile|warmup|lazy|load" ~/lloyd/agent-services/logs/agent-tts.log | tail -20'),
    # the `ss -lnu` UDP block, held by `after last start) ===` reaching the
    # `grep -vE ... agent-livekit-server.err` two statements later.
    ("20260910_014618_iv6a62.json",
     'timeout 40 bash -c \'cd ~/lloyd/agent-services/logs; echo "=== any UDP >=50000 in full list ==="; ss -lnu | grep -oE ":5[0-9]{4}$" | sort -u | head; echo "(total UDP listeners: $(ss -lnu | grep -c UNCONN))"; echo; echo "=== livekit boot block (first 25 non-ListRooms lines after last start) ==="; grep -vE "ListRooms|GetNodeMetrics|service/twirp" agent-livekit-server.err | tail -25\''),
)

# Synthetic one-separator-per-case pins: the same leak shape, reduced to the
# separator that carried it. All four are reads; none may reach voice-mode.
STATEMENT_SEPARATED_READS: tuple[tuple[str, str], ...] = (
    (";", 'echo "=== livekit err lines since the restart ==="; tail -25 agent-livekit-server.err'),
    ("|", "journalctl --user -u agent-livekit-server --since -10m | grep -i restart"),
    ("&&", "awk '/agent-tts/ {print $1}' agent-services/supervisor/supervisord.conf "
           "&& systemctl --user list-units | grep -i start"),
    # The old gap already excluded a newline, so this one is a restatement of
    # the guarantee rather than a case it held. Pinned so the narrowed class
    # cannot drop it.
    ("newline", 'echo "=== a restart is not what this call does ==="\n'
                "tail -25 agent-livekit-server.err"),
)

# A genuine control call, multi-line, from 20260907_220430_iv15a0.json: the
# restart sits on its own line, so bounding the gap to one statement must not
# touch it.
TRANSCRIPT_CONTROL_CALL = (
    "cd ~/lloyd; CONF=agent-services/supervisor/supervisord.conf\n"
    "supervisorctl -c $CONF restart agent-tts 2>&1\n"
    "for i in $(seq 1 40); do\n"
    "  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8090/v1/health 2>/dev/null)\n"
    '  [ "$code" = "200" ] && { echo "health 200 after ${i}0s-ish (iter $i)"; break; }\n'
    "  sleep 3\n"
    "done\n"
    'echo "final health: $(curl -s -m 5 http://localhost:8090/v1/health)"\n'
    "supervisorctl -c $CONF status agent-tts"
)


def _held_skill(command: str) -> str | None:
    rule = sd.match_rule("Bash", {"command": command}, rules=sd.DISPATCH_RULES)
    return rule.skill if rule else None


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
        # The restart that has to name the right unit. Adjacency inside one
        # shell statement is what the voice-mode gate keys on (#751), so these
        # carry the realistic `-c <conf>` prefix: verb and unit in the same
        # statement, which is every control call in the transcripts.
        "supervisorctl -c cfg.conf restart lloyd-mc:lloyd-backend": "restart-lloyd",
        "systemctl --user restart lloyd-voice-mode.service": "voice-mode",
        "supervisorctl restart agent-tts": "voice-mode",
        "supervisorctl -c agent-services/supervisor/supervisord.conf restart agent-tts": "voice-mode",
        "supervisorctl -c cfg.conf restart agent-livekit-server": "voice-mode",
        TRANSCRIPT_CONTROL_CALL: "voice-mode",
        # The guard that outlives the narrowing: launching the script directly
        # is the failure the skill forbids, so the invocation alone triggers.
        "~/lloyd/.venvs/lloyd/bin/python ~/lloyd/voice_mode.py --serve": "voice-mode",
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
        # #751: a control verb in one shell statement and a voice unit in the
        # next is a diagnosis, not a control call. First the four reads as they
        # appear in the transcripts, then one synthetic case per separator.
        *(cmd for _session, cmd in TRANSCRIPT_READS_HELD_BY_THE_OLD_RULE),
        *(cmd for _separator, cmd in STATEMENT_SEPARATED_READS),
    ]
    for command in benign:
        assert sd.match_rule("Bash", {"command": command}, rules=sd.DISPATCH_RULES) is None, command
    print("test_benign_dispatches_do_not_trigger: OK")


def test_the_voice_read_negatives_are_the_dispatches_the_old_rule_held():
    """A negative that the old rule also passed would pin nothing, so each one
    is checked against the #536 pattern set — the unbounded `[^\n]*` gap — and
    against the shipped one. 4 transcript-exact reads + 3 synthetic separators
    are legacy leaks; the `newline` case is the exception named in its own
    comment (the old gap already excluded it) and is pinned only as must-not-fire.
    """
    legacy_leaks = [cmd for _session, cmd in TRANSCRIPT_READS_HELD_BY_THE_OLD_RULE]
    legacy_leaks += [cmd for sep, cmd in STATEMENT_SEPARATED_READS if sep != "newline"]
    assert len(legacy_leaks) == 7, f"fixture shape changed: {len(legacy_leaks)} cases"
    for command in legacy_leaks:
        assert any(p.search(command) for p in _LEGACY_VOICE_PATTERNS), (
            f"{command[:70]!r} was never held by the old rule: this negative pins nothing"
        )
        assert _held_skill(command) != "voice-mode", f"still held: {command[:70]!r}"
    for _sep, command in STATEMENT_SEPARATED_READS:
        assert _held_skill(command) != "voice-mode", f"{_sep}: still held"
    print("test_the_voice_read_negatives_are_the_dispatches_the_old_rule_held: OK")


def test_voice_mode_verb_unit_gap_is_bounded_to_one_statement():
    """#751 clause 3: the narrowing is adjacency, not deletion.

    Both word orders have to survive — deleting the reverse-order pattern alone
    would stop these leaks too, and would silently un-gate `agent-tts stop` —
    and the gap between verb and unit must exclude the shell statement
    separators and carry a bounded repeat.
    """
    rule = next(r for r in sd.DISPATCH_RULES if r.skill == "voice-mode")
    verb_unit, unit_verb, direct_script = rule.patterns
    unit_src = "agent-(?:tts"
    assert verb_unit.pattern.index("restart") < verb_unit.pattern.index(unit_src), (
        "the verb -> unit order is gone"
    )
    assert unit_verb.pattern.index(unit_src) < unit_verb.pattern.index("restart"), (
        "the unit -> verb order is gone"
    )
    gap_class = re.compile(r"\[\^([^\]]+)\]\{(\d+),(\d+)\}")
    for pattern in (verb_unit, unit_verb):
        match = gap_class.search(pattern.pattern)
        assert match, f"{pattern.pattern!r} carries no bounded gap class"
        excluded, low, high = match.group(1), int(match.group(2)), int(match.group(3))
        for separator in (r"\n", "|", ";", "&"):
            assert separator in excluded, f"[^{excluded}] does not exclude {separator!r}"
        assert low == 0 and 0 < high <= 60, match.group(0)
        assert "[^\\n]*" not in pattern.pattern, "the unbounded gap is back"
    # And the guard the narrowing must not reach: launching the script directly,
    # whole invocation, unbounded gap intact.
    assert direct_script.pattern == r"\bpython\S*[^\n]*\bvoice_mode\.py\b"
    print("test_voice_mode_verb_unit_gap_is_bounded_to_one_statement: OK")


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


def test_a_voice_diagnosis_read_reaches_the_loop_and_a_control_call_does_not(stub_body):
    """#751's consequence crosses a process boundary, so it is checked there.

    `match_rule` is a pure function, but its verdict decides whether a drafted
    `Bash` is executed at all: a held call never reaches the shell and comes back
    as a synthetic tool result (~2.7 s round-trip here), an unheld one dispatches.
    Both halves therefore go through `_pre_dispatch` — the real PreToolUse walk —
    with a transcript-exact diagnosis on one side and a same-statement restart on
    the other.
    """
    hooks = _install(HookRegistry())
    diagnosis = TRANSCRIPT_READS_HELD_BY_THE_OLD_RULE[0][1]
    evt = _run(_pre_dispatch(
        tc=_tc("Bash", {"command": diagnosis}),
        options=RunOptions(model="m", hooks=hooks),
        session_id="s", loaded_set=_StubLoadedSet(),
    ))
    assert evt is None, "a diagnostic read must reach Bash, not be held for a protocol"

    control = "supervisorctl -c agent-services/supervisor/supervisord.conf restart agent-tts"
    held = _run(_pre_dispatch(
        tc=_tc("Bash", {"command": control}),
        options=RunOptions(model="m", hooks=hooks),
        session_id="s2", loaded_set=_StubLoadedSet(),
    ))
    assert held is not None, "the restart must still be held for the voice protocol"
    assert held["is_error"] is False, held
    assert "BODY-OF[voice-mode]" in held["content"], held
    print("test_a_voice_diagnosis_read_reaches_the_loop_and_a_control_call_does_not: OK")


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


# ===========================================================================
# #779 — the deliverer must be attributable to the registry it sits on
#
# A paired with/without-skill bench arm withholds a SKILL.md body by not
# installing this deliverer on the without-arm's registry. Until the registry
# could report it, and until a caller could read which bodies the deliverer
# handed over, a without-arm that leaked was indistinguishable from a skill that
# did nothing — the experiment reported "inert" for a body it never withheld.
# `STATS` answers neither question: it is one process-global dict and
# `run_bench_sdk` runs trials concurrently under a semaphore, so a snapshot delta
# spans two trials. Hence two surfaces: a flag on the registry, and a set the
# caller owns.
# ===========================================================================


def test_registry_reports_the_dispatch_deliverer_registered_on_it():
    """Clause 3. The registry is the only possible witness.

    `add_pre_tool_use` takes an anonymous callback, so a deliverer and a safety
    gate are the same shape from the outside — nothing downstream of the
    installer can tell them apart by looking. The installer therefore says so,
    and a trial's trace reads the registry's answer instead of restating what
    the caller hoped it had built.
    """
    assert HookRegistry().skill_dispatch_installed is False

    safety_only = HookRegistry()
    install_default_safety_hook(safety_only)
    assert safety_only.skill_dispatch_installed is False, (
        "a registry carrying only the safety gate IS the without-arm condition; a "
        "trace that stamped True here would claim a deliverer it never built"
    )

    armed = HookRegistry()
    assert sd.install_skill_dispatch_hook(
        armed, rules=sd.DISPATCH_RULES, force_enabled=True) is True
    assert armed.skill_dispatch_installed is True


def test_a_refused_install_does_not_claim_the_deliverer():
    """The false half of the flag must be as trustworthy as the true half.

    Every case where the installer declines — config off, forced off, an empty
    rule set — is a trial that has to read `skill_dispatch_installed: false`, or
    the flag records intent rather than what was built.
    """
    forced_off = HookRegistry()
    assert sd.install_skill_dispatch_hook(
        forced_off, rules=sd.DISPATCH_RULES, force_enabled=False) is False
    assert forced_off.skill_dispatch_installed is False
    assert forced_off._pre == []

    no_rules = HookRegistry()
    assert sd.install_skill_dispatch_hook(
        no_rules, rules=(), force_enabled=True) is False
    assert no_rules.skill_dispatch_installed is False


def test_a_caller_owned_delivered_set_names_exactly_the_bodies_handed_over(stub_body):
    """Clause 4: per-trial attribution, which `STATS` structurally cannot give.

    The set belongs to the caller: the installer fills it, the caller reads it
    after the turn. One delivery per skill per turn already holds
    (`test_a_skill_is_held_at_most_once_per_turn`), so this is a list of bodies
    that arrived and not a count — and a skill the deliverer declined to hand
    over never appears in it.
    """
    mine: set[str] = set()
    hooks = _install(HookRegistry(), delivered=mine)
    assert mine == set(), "installing must not pre-populate what has not been delivered"

    first = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    assert first.get("hookSpecificOutput", {}).get("skillDeliver"), first
    assert mine == {"youtube-transcript"}

    # The re-issued call runs: a second entry for the same skill would turn the
    # set into a count of deliveries rather than a list of bodies that arrived.
    _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    assert mine == {"youtube-transcript"}

    # A different rule in the same turn is a second body; a call no rule matches
    # is not a body at all.
    _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash",
        tool_input={"command": "supervisorctl -c c.conf restart agent-tts"},
    ))
    assert mine == {"youtube-transcript", "voice-mode"}
    _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Read", tool_input={"file_path": "/tmp/x"},
    ))
    assert mine == {"youtube-transcript", "voice-mode"}


def test_a_skill_already_injected_at_turn_start_never_enters_the_delivered_set(stub_body):
    """The set is the leak channel, not a log of everything the module saw.

    A with-arm whose body came from turn-start prefetch was correctly not
    delivered again, and must show `skills_delivered: []`. Otherwise the two
    routes blur into one number and the artifact cannot name the route that
    delivered, which is the thing #779 asks the row to say.
    """
    mine: set[str] = set()
    hooks = _install(HookRegistry(), delivered=mine,
                     already_injected={"youtube-transcript"})
    out = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    assert out == {}
    assert mine == set()
    assert sd.STATS["skipped_already_injected"] == 1


def test_a_caller_that_passes_nothing_behaves_exactly_as_before(stub_body):
    """The parameter is additive: production's one install site
    (`app/routers/messages.py:install_skill_dispatch_hook`) passes no set and
    keeps today's behaviour, held call included."""
    hooks = _install(HookRegistry())
    out = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    assert out.get("hookSpecificOutput", {}).get("skillDeliver"), out
    assert sd.STATS["delivered"] == 1


def test_a_declined_install_leaves_a_supplied_set_untouched():
    """A caller reading its set after a refused install reads an empty one, not a
    leftover: nothing was installed, so nothing can arrive."""
    mine: set[str] = set()
    assert sd.install_skill_dispatch_hook(
        HookRegistry(), rules=sd.DISPATCH_RULES, force_enabled=False, delivered=mine
    ) is False
    assert mine == set()


def test_a_body_with_no_skill_body_on_disk_is_not_reported_as_delivered(monkeypatch):
    """`skipped_no_body` is a withhold, so it must not land in the delivered set —
    a set that counted attempts would call a missing file a delivered protocol."""
    monkeypatch.setattr(sd, "skill_body", lambda name: "")
    mine: set[str] = set()
    hooks = _install(HookRegistry(), delivered=mine)
    out = _run(hooks.fire_pre_tool_use(
        session_id="t", tool_name="Bash", tool_input={"command": "yt-dlp x"},
    ))
    assert out == {}
    assert mine == set()
    assert sd.STATS["skipped_no_body"] == 1


def test_the_dispatch_marker_and_its_recogniser_cannot_drift():
    """The bench recognises a dispatch-time body by the marker `render_delivery`
    writes, so the two are one contract: a body whose marker changed shape is a
    delivered skill the trial record silently omits — a leak reported as a
    withhold, the exact inversion #779 exists to prevent.
    """
    body = sd.render_delivery(sd.DISPATCH_RULES[2], "THE BODY")
    assert '<skill-dispatch name="youtube-transcript"' in body
    assert sd.delivered_skill_names(body) == {"youtube-transcript"}

    # The turn-start tag is NOT dispatch delivery: keyed on the shared regex, an
    # arm that injected at turn start would report a delivery it never made.
    assert sd.delivered_skill_names(
        '<skill name="restart-lloyd" score="9">\nBODY\n</skill>'
    ) == set()
    assert sd.delivered_skill_names("") == set()


def test_probe_reports_chars_measured_and_tokens_as_a_labelled_estimate(monkeypatch):
    """#752: the probe used to report `avg_injected_tokens_per_event` from
    chars/4 while the cap is in chars, so a capped body read as exactly 1500
    "tokens" — the cap's arithmetic. It now reports chars, an estimate named
    as one, and how many events hit the cap."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "skill_dispatch_probe_752",
        Path(__file__).resolve().parents[2] / "eval" / "run_skill_dispatch_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    monkeypatch.setattr(sd, "skill_body", lambda name: "x" * (sd.MAX_DELIVERY_CHARS * 2))
    report = probe.score([{"name": "Bash", "args": {"command": "yt-dlp x"}}],
                         scanned_sessions=1)
    row = next(r for r in report["per_protocol"].values() if r["triggered"])
    assert "avg_injected_tokens_per_event" not in row
    assert row["avg_injected_chars_per_event"] == sd.MAX_DELIVERY_CHARS
    assert row["est_tokens_per_event_chars_div_4"] == sd.MAX_DELIVERY_CHARS / 4
    assert row["capped_events"] == 1

