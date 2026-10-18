"""A background turn may not restart or stop an engine or a service.

On 2026-09-17 an autocode turn ran `round restart --only agent-llm-primary`
from Bash to fix a grader that had returned empty output, took the primary
down for five minutes, killed every turn in flight including itself, and
left its freshly reopened round with no owner. Refused now at both Bash
enforcement points, for background sessions only.

The same rule covers *booting* a supervised program by hand (`#1363`, 2026-09-22):
the launcher script is how the supervisor starts it, so hand-running one is
service control too — and for `agent-djev` it is the route by which a round
could boot a *chosen kernel variant* into the production GPU 2 ranker, which the
#1361 sweep reserves for an attended window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.harness import safety
from app.harness import service_control
from app.harness.service_control import (check_service_control, find_service_control,
                                         is_background_session)

WORKER = "20260916_165238_autocode_1b1a"
AUTONOMY = "20260910_120001_autonomy_9f2a"
CHAT = "20260916_164818_iv2eca"

REFUSED = [
    "cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.automod.round restart --only agent-llm-primary --reason \"grader empty\" 2>&1 | tail -3 &",
    "python -m scripts.automod.round recover",
    "/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf restart lloyd-mc:lloyd-backend",
    "supervisorctl stop agent-llm-primary",
    "supervisorctl -c x.conf reread; supervisorctl -c x.conf update",
    "systemctl --user restart lloyd-guardian",
    "systemctl stop agent-supervisord.service",
    "systemctl --user daemon-reload",
    "pkill -f vllm.entrypoints",
    "killall llama-server",
    "bash agent-services/bin/flash-next-run-arm.sh fp8",
    # #1363: the djev kernel bisect (#1361) is an Alan-attended window because
    # every trial boots a different kernel into the production ranker. Step 1 of
    # that runbook's hand-boot form, in the shape the sweep would actually run
    # it. Measured 2026-09-22 these four answered ALLOWED for a worker session:
    # of the 7 launchers supervisord's own conf.d names, exactly one
    # (`start-qwen38-flash-next.sh`) was guarded, so "no round may boot a
    # variant" held only because :8010/:8011 were already occupied.
    "bash agent-services/bin/start-djev.sh",
    "agent-services/bin/start-djev.sh",
    # #2505: `11fedc3d` added conf.d/agent-tts-omni.conf naming
    # start-qwen3-tts-omni.sh and `_LAUNCHERS` never learned it, so this hand-boot
    # answered ALLOWED for a background session. Measured red on main at 3bddc5bc
    # by the open-set test below, which is what it was built to catch. The bare
    # second form is how supervisor itself boots the program — that conf's
    # `command=` is the script path with no interpreter in front of it.
    "bash agent-services/bin/start-qwen3-tts-omni.sh",
    "agent-services/bin/start-qwen3-tts-omni.sh",
    "MOE_BACKEND=triton BATCH_INVARIANT=0 agent-services/bin/start-djev.sh",
    "MOE_BACKEND=triton BATCH_INVARIANT=0 bash agent-services/bin/start-djev.sh",
    "bash -c 'supervisorctl -c x.conf restart lloyd-mc:lloyd-mcp'",
    "python3 -c \"import subprocess; subprocess.run(['supervisorctl', '-c', 'x.conf', 'restart', 'lloyd-mc:lloyd-backend'])\"",
    "nohup timeout 900 .venvs/lloyd/bin/python -m scripts.automod.round restart --only lloyd-backend > /tmp/r.log 2>&1 &",
]
ALLOWED = [
    "supervisorctl -c x.conf status",
    "/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c x.conf tail -f lloyd-mc:lloyd-backend stderr",
    "systemctl --user status lloyd-guardian",
    "systemctl --user list-units | grep lloyd",
    "python -m scripts.automod.round status",
    # `round land` moved to tests/test_landing_killed.py (2026-09-17): a
    # foreground landing waits on the turn that ran it and dies at the Bash
    # timeout. The gate alone stays allowed.
    "python -m scripts.automod.round gate SM_1",
    "pkill -f my_probe_script.py",
    "curl -s localhost:8096/health",
    "grep -n 'supervisorctl restart' CLAUDE.md",
    "echo 'run: round restart --only lloyd-backend' >> notes.md",
    "python3 -c \"print('supervisorctl status')\"",
    "git log --oneline -3 -- scripts/automod/round.py",
    # The guard refuses a boot, never a read. #1363's own triage and every
    # health probe inspect these scripts by name, and `bash -n` is how a shell
    # script is linted without executing it.
    "cat agent-services/bin/start-djev.sh",
    "grep -n MOE_BACKEND agent-services/bin/start-djev.sh",
    "head -40 agent-services/bin/start-djev.sh",
    "bash -n agent-services/bin/start-djev.sh",
]


@pytest.mark.parametrize("command", REFUSED)
def test_a_worker_session_is_refused(command):
    assert find_service_control(command)
    assert check_service_control(command, WORKER)
    assert check_service_control(command, AUTONOMY)
    match = safety.check_bash_command(command, None, session_id=WORKER)
    assert match and match[0].startswith("service control:"), match
    match = safety.check_bash_command(command, None, at_dispatch=True, session_id=WORKER)
    assert match and match[0].startswith("service control:"), "dispatch refuses too"


@pytest.mark.parametrize("command", ALLOWED)
def test_read_only_forms_stay_allowed(command):
    assert check_service_control(command, WORKER) is None
    assert safety.check_bash_command(command, None, session_id=WORKER) is None


@pytest.mark.parametrize("command", REFUSED)
def test_a_chat_session_is_never_refused_here(command):
    assert check_service_control(command, CHAT) is None
    assert check_service_control(command, None) is None
    assert safety.check_bash_command(command, None, session_id=CHAT) is None
    assert safety.check_bash_command(command, None) is None, "no session known: skipped, not refused"


def test_a_subagent_is_classified_by_its_parent():
    parents = {"task:abc": WORKER, "task:def": CHAT, "task:loop": "task:loop"}
    assert is_background_session("task:abc", parent_of=parents.get)
    assert not is_background_session("task:def", parent_of=parents.get)
    assert not is_background_session("task:loop", parent_of=parents.get)
    assert not is_background_session("task:abc"), "no resolver: not refused"
    assert check_service_control("supervisorctl restart x", "task:abc", parent_of=parents.get)
    assert check_service_control("supervisorctl restart x", "task:def", parent_of=parents.get) is None


@pytest.mark.asyncio
async def test_dispatch_refuses_the_incident_command_for_the_worker_session(monkeypatch):
    import agent_mcp.main as M
    cmd = "cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.automod.round restart --only agent-llm-primary --reason x &"
    out = await M.call_tool("Bash", {"command": cmd}, {M.META_SESSION_ID: WORKER})
    content = getattr(out, "content", out)
    text = content[0].text if hasattr(content[0], "text") else str(content[0])
    assert "service control" in text and "background session" in text, text


def test_the_hook_passes_the_session_id(monkeypatch):
    import asyncio
    seen = {}

    def fake(command, cwd=None, *, at_dispatch=False, session_id=None, parent_of=None):
        seen["session_id"] = session_id
        return None
    monkeypatch.setattr(safety, "check_bash_command", fake)
    asyncio.run(safety._safety_pretool_cb(
        {"tool_name": "Bash", "tool_input": {"command": "ls"}, "session_id": WORKER}, None, None))
    assert seen["session_id"] == WORKER


CONF_DIR = Path(__file__).resolve().parents[1] / "agent-services" / "supervisor" / "conf.d"


def _supervised_launchers() -> list[str]:
    """Every `*.sh` supervisord launches itself, read out of its own conf.d.
    That is the corpus the guard has to cover, and the confs — not a hand-kept
    list — are what says which programs are owned."""
    found: list[str] = []
    for conf in sorted(CONF_DIR.glob("*.conf")):
        for line in conf.read_text().splitlines():
            if line.startswith("command="):
                found += [tok for tok in line[len("command="):].split()
                          if tok.endswith(".sh")]
    return found


def test_every_supervised_launcher_refuses_a_background_hand_boot():
    """The open-set half of #1363. `bash agent-services/bin/start-djev.sh`
    answering ALLOWED was not one missing entry so much as a list nobody had
    checked against the tree: measured 2026-09-22, conf.d names 7 launchers and
    the guard named one of them. Derived from the confs rather than restated
    here, so the day a program is added that the guard has never heard of, this
    test is red instead of the next round discovering it by booting onto a live
    engine.

    What it covers is the `.sh` programs — 9 of the 14 confs that name a
    `command=` (conf.d holds 15 files; `lloyd-mc.conf` is a group and has none).
    Up from 7 on 2026-09-22 because `11fedc3d` added `agent-tts-omni`. The other 5
    name a python or node entry point (`command=…/python …/server.py`,
    `…/python -m agent_mcp.main`, `…/node …/qmd.js`, `npm …`, the livekit worker),
    and this test does not reach them: refusing a hand-run `.py` would catch
    someone running that module under a test harness, which is a different
    question from booting an engine, and is left on #1363 rather than taken here.

    This is the drift it was written for, and the loop caught it rather than a
    round discovering it by booting onto a live engine: `11fedc3d` added the conf
    and never touched `app/harness/service_control.py`, so the test was red on
    `main` from 2026-10-09 until #2505, naming `start-qwen3-tts-omni.sh`.
    """
    launchers = _supervised_launchers()
    # A floor, not the count: the corpus is derived, so an empty or stubby read of
    # conf.d (a moved directory, a parser that stopped matching `command=`) has to
    # fail here rather than silently shrink the set this test checks. #1363
    # measured 7 `.sh` launchers in conf.d on 2026-09-22; it has only grown.
    assert len(launchers) >= 7, (
        f"the conf.d corpus collapsed to {launchers} — this test is worthless on a "
        f"set that small, so the corpus or its parser moved")
    for target in launchers:
        command = f"bash {target}"
        assert check_service_control(command, WORKER), f"{command} boots a supervised program"
        assert check_service_control(command, AUTONOMY), f"{command} boots it for autonomy too"
        assert check_service_control(command, CHAT) is None, f"a person may boot {target}"


def test_the_djev_sweep_boot_is_refused_because_it_boots_a_supervised_engine():
    """#1361's kernel sweep cannot be run by a round because every trial boots a
    different kernel into the production GPU 2 ranker, and #1363's acceptance is
    that this be enforced rather than incidental. Pinned here are both edges of
    the clause and the wording: the sweep's own env-prefixed hand-boot is
    refused for a background session and allowed for a chat session, and the
    reason names what a hand-boot does. The generic "restart or stop" sentence
    does not cover a boot, and the reader of this refusal is the turn that was
    about to run it."""
    boot = "MOE_BACKEND=triton BATCH_INVARIANT=0 bash agent-services/bin/start-djev.sh"
    why = check_service_control(boot, WORKER)
    assert why, f"{boot} must be refused for a background session"
    assert "start-djev.sh" in why, why
    assert "supervisorctl" in why, f"a boot races the supervisor that owns the pid: {why}"
    assert "background session" in why, why
    assert check_service_control(boot, CHAT) is None, "a person runs the attended window"


def test_the_tts_omni_hand_boot_is_refused_however_the_path_is_spelled():
    """#2505 — the guard fell behind the tree and the open-set test went red, as designed.

    `11fedc3d` added `agent-services/supervisor/conf.d/agent-tts-omni.conf`, whose
    `command=` is the bare path to `start-qwen3-tts-omni.sh` — the voice engine on
    :8093, GPU 3 — and `app/harness/service_control.py::_LAUNCHERS` was not widened
    with it. So `bash agent-services/bin/start-qwen3-tts-omni.sh` was answered
    ALLOWED for a worker session, and stayed that way until this item.

    What is pinned here beyond the corpus test above is the spelling. That test
    hands the guard the path exactly as `command=` carries it — the ABSOLUTE one —
    while a turn types the repo-relative one, and only a basename match makes them
    the same question. A guard keyed to either spelling alone would satisfy one of
    these two tests and still let through the command the other one names, so both
    are asserted, and the absolute one is taken from the conf corpus rather than
    written out, so the day the conf moves the script this test stops being a claim
    about a path nothing supervises.

    The read forms stay allowed for the reason the djev launcher's do: the turn
    diagnosing a boot has to be able to read the script, and `bash -n` parses it
    without putting a second engine on GPU 3. The guard refuses a boot, never a
    read — so that half is asserted here too, or a fix could pass by over-refusing.
    """
    relative = "agent-services/bin/start-qwen3-tts-omni.sh"
    conf_named = [t for t in _supervised_launchers() if t.endswith("/" + relative)]
    assert conf_named, (
        "conf.d no longer names start-qwen3-tts-omni.sh, so this test is pinning a "
        "name no supervisor owns — fix the conf path, not the assertion")

    for command in (f"bash {relative}", relative,
                    f"bash {conf_named[0]}", conf_named[0]):
        why = check_service_control(command, WORKER)
        assert why, f"{command} boots a supervised engine and was answered ALLOWED"
        assert "start-qwen3-tts-omni.sh" in why, f"the refusal must name the script: {why}"
        assert "supervisorctl" in why, f"a boot races the supervisor that owns the pid: {why}"
        assert "background session" in why, why
        assert check_service_control(command, AUTONOMY), f"{command} boots it for autonomy too"
        assert check_service_control(command, CHAT) is None, "a person runs the boot"

    for read_form in (f"cat {relative}", f"bash -n {relative}",
                      f"grep -n GPU {relative}"):
        assert check_service_control(read_form, WORKER) is None, (
            f"the guard refused a read of {read_form!r} rather than a boot of it")


# ── #1961: an unresolvable `task:*` id is UNKNOWN, not attended ─────────────
#
# One boolean, `is_background_session`, was handed to four guards. Its False
# covers both "a person is reading this" and "I could not tell", so a single
# missed row in `agent_mcp/_subagent_registry` granted a worker's subagent the
# chat surface: the desktop, the engine-restart pass, and (through
# `supply_chain._attended_by_session_id`) an install-provenance bypass that
# journalled nothing. `classify_session` splits the answer three ways so each
# guard picks its own side, and the old boolean is left exactly as it was so the
# injection probe keeps the fail-open its cost justifies.

UNKNOWN_TASK = "task:deadbeef"
NO_PARENTS: dict[str, str] = {}


def test_a_task_id_with_no_resolvable_parent_classifies_unknown():
    """Clause 1 — all three ways of failing to resolve, and no other id can be unknown.

    A `task:*` session is minted by the Task tool and is never the attended
    surface, so "I could not find the parent" is not evidence of a person. The
    three miss shapes are each asserted: no resolver supplied, a resolver that
    returns None, and a resolver that returns the id itself (a self-parent is not
    an ancestry — `task:loop` resolves to itself forever).
    """
    from app.harness.service_control import ATTENDED, BACKGROUND, UNKNOWN

    classify = service_control.classify_session

    def raising(session_id: str) -> str | None:
        raise RuntimeError("registry blew up")

    assert classify(UNKNOWN_TASK) is UNKNOWN, "no resolver supplied"
    assert classify(UNKNOWN_TASK, parent_of=NO_PARENTS.get) is UNKNOWN, \
        "resolver returned None"
    assert classify(UNKNOWN_TASK, parent_of=raising) is UNKNOWN, \
        "a resolver that raises is a miss, not a pass — and the exception must " \
        "not escape into dispatch"
    assert classify("task:loop", parent_of={"task:loop": "task:loop"}.get) is UNKNOWN, \
        "a self-parent is not an ancestry"

    # The answers that are NOT unknown, and the shape rule that makes them so.
    assert classify(CHAT) is ATTENDED
    assert classify(WORKER) is BACKGROUND
    assert classify(AUTONOMY) is BACKGROUND
    assert classify("") is ATTENDED, "no id at all is no privileges, as today"
    assert classify("task:abc", parent_of={"task:abc": CHAT}.get) is ATTENDED
    assert classify("task:abc", parent_of={"task:abc": WORKER}.get) is BACKGROUND
    assert classify("", parent_of=NO_PARENTS.get) is ATTENDED


def test_is_background_session_still_answers_exactly_as_it_did():
    """Clause 1's second half — the fix is a classifier beside the boolean, not a
    flipped boolean. The injection probe reads the boolean and its fail-open
    costs only a missing shadow row; #1170's pinned cases and the scored Bash
    corpus both read it too, so its answers are frozen here while the guards
    above move."""
    assert is_background_session("task:abc", parent_of={"task:abc": WORKER}.get)
    assert not is_background_session("task:def", parent_of={"task:def": CHAT}.get)
    assert not is_background_session("task:loop",
                                     parent_of={"task:loop": "task:loop"}.get)
    assert not is_background_session(UNKNOWN_TASK), "no resolver: not refused"
    assert not is_background_session(UNKNOWN_TASK, parent_of=NO_PARENTS.get)
    assert not is_background_session(CHAT)
    assert is_background_session(WORKER)


def test_the_service_control_refusal_fires_for_an_unknown_task_session():
    """Clause 3 — the rule exists to stop an unattended turn killing the engine,
    and a `task:*` id is never the attended surface whatever the registry says.

    Half two is that nothing else moves: chat allowed, worker refused, a task of
    a worker refused, a task of a chat allowed. Those four are the contexts the
    scored Bash corpus runs every command through, so a shift here is a shift in
    the corpus the item names as must-not-break.
    """
    from app.harness.service_control import ATTENDED, BACKGROUND, UNKNOWN

    cmd = "supervisorctl restart x"
    parents = {"task:abc": WORKER, "task:def": CHAT}
    assert service_control.classify_session(UNKNOWN_TASK,
                                           parent_of=NO_PARENTS.get) is UNKNOWN

    why = check_service_control(cmd, UNKNOWN_TASK, parent_of=NO_PARENTS.get)
    assert why, (
        "an unresolvable subagent got the engine-restart pass: this is the miss "
        "that the item names, and one registry row was all it took")

    assert check_service_control(cmd, CHAT) is None
    assert check_service_control(cmd, WORKER), "worker still refused"
    assert check_service_control(cmd, "task:abc", parent_of=parents.get), \
        "a task of a worker still refused"
    assert check_service_control(cmd, "task:def", parent_of=parents.get) is None, \
        "a task of a chat is still allowed"

    # The classifier is what the guard keys on, and the four resolvable shapes
    # are the four corpus contexts.
    classify = service_control.classify_session
    assert classify(CHAT) is ATTENDED and classify(WORKER) is BACKGROUND
    assert classify("task:def", parent_of=parents.get) is ATTENDED
    assert classify("task:abc", parent_of=parents.get) is BACKGROUND


def test_an_unresolvable_lookup_is_recorded_with_the_id_and_the_guard(caplog):
    """Clause 5 — a systematic miss must not read as a clean window.

    The miss is silent today: `parent_scope()` returning None produces no line
    anywhere, so a resolver that fails on every dispatch is indistinguishable
    from a window with no dispatches. One WARNING per miss, carrying the session
    id, the guard that asked and why resolution failed — the reason is in the
    record so `no resolver supplied` (a call site that forgot its argument) is
    not mistaken for a registry eviction.
    """
    with caplog.at_level("WARNING", logger="lloyd-harness-service-control"):
        service_control.classify_session(UNKNOWN_TASK, parent_of=NO_PARENTS.get,
                                         guard="desktop")
    text = caplog.text
    assert service_control.MISS_EVENT in text, (
        f"the miss left no record: {text!r}")
    assert UNKNOWN_TASK in text, "the record must name which session"
    assert "desktop" in text, "and which guard asked"

    # A resolved lookup records nothing: the count is the miss count, not a
    # dispatch count, or the signal is unreadable.
    caplog.clear()
    with caplog.at_level("WARNING", logger="lloyd-harness-service-control"):
        service_control.classify_session("task:abc",
                                         parent_of={"task:abc": CHAT}.get,
                                         guard="desktop")
    assert service_control.MISS_EVENT not in caplog.text, (
        "a resolved lookup was counted as a miss")


def test_the_tool_sandbox_records_its_misses_under_its_own_guard_name(caplog, monkeypatch):
    """#2025 — the fifth consumer is countable like the other four.

    One `guard_parent_unresolved` WARNING per miss, naming `guard=tool_sandbox`
    and the reason; a resolved lookup logs no WARNING and one debug witness, so
    a zero miss count is never mistaken for no traffic.
    """
    import logging
    from agent_mcp import _tool_sandbox as S

    for resolver, why in ((lambda sid: None, "resolver returned no parent"),
                          (lambda sid: sid, "resolver returned the id itself")):
        caplog.clear()
        monkeypatch.setattr(S, "_parent_of", resolver)
        with caplog.at_level(logging.DEBUG):
            assert S.is_sandboxed_session(UNKNOWN_TASK) is True
        misses = [r.getMessage() for r in caplog.records
                  if r.levelno == logging.WARNING
                  and service_control.MISS_EVENT in r.getMessage()]
        assert len(misses) == 1, misses
        assert UNKNOWN_TASK in misses[0] and "guard=tool_sandbox" in misses[0]
        assert why in misses[0], misses[0]

    caplog.clear()
    monkeypatch.setattr(S, "_parent_of", lambda sid: CHAT)
    with caplog.at_level(logging.DEBUG):
        assert S.is_sandboxed_session(UNKNOWN_TASK) is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text
    witness = [r for r in caplog.records if r.levelno == logging.DEBUG
               and r.name == "lloyd-tool-sandbox" and UNKNOWN_TASK in r.getMessage()]
    assert len(witness) == 1, caplog.text


# ── #2541: a `--now` flag form IS the unit operation it performs ─────────────
#
# Every command in `NOW_REFUSED` answered ALLOWED for a worker session at 01d0082b
# while the bare-verb equivalent against the same unit was refused. `--now` is a flag,
# and `enable`, `disable` and `reenable` are not in `_SYSTEMCTL_VERBS`, so none of the
# three axes saw the start or the stop systemd runs for it. The reachable target is not
# the `static` lloyd-persistence-window.service the item named (a unit with no
# `[Install]` section cannot be enabled by any form) — it is the enabled+active
# `agent-supervisord.service`, where `disable --now` is the stop half of a
# supervised-stack restart that the refused verb `stop` exists to block.

#: (command, the label the guard must now give it)
NOW_REFUSED = [
    ("systemctl --user enable --now agent-supervisord.service", "systemctl start"),
    ("systemctl --user reenable --now agent-supervisord.service", "systemctl start"),
    ("systemctl --user disable --now agent-supervisord.service", "systemctl stop"),
    # The flag may sit before the verb; the reading does not change.
    ("systemctl --user --now enable agent-supervisord.service", "systemctl start"),
    # `-N` is systemd's short form of `--now`.
    ("systemctl --user -N enable agent-supervisord.service", "systemctl start"),
    # No unit named at all: still the operation, never the exemption. A half-typed
    # command line is not evidence that this was only ever meant to arm a timer, which
    # is the same reason an unresolvable session id is UNKNOWN rather than attended.
    ("systemctl --user enable --now", "systemctl start"),
]

#: The `.timer` pass #2537 exists for: arming and unarming a schedule, which starts no
#: unit. `check-unit-enabledness.sh`'s FAIL line prints the first of these, so refusing
#: it would have a watchdog print a command its reader is refused — the boundary
#: tests/test_unit_enabledness_check.py::test_the_command_this_line_prints_is_runnable
#: crosses from the other side.
NOW_TIMER_ALLOWED = [
    "systemctl --user enable --now lloyd-persistence-window.timer",
    "systemctl --user disable --now lloyd-persistence-window.timer",
    "systemctl --user enable --now lloyd-cert-renew.timer",
]

#: The same shapes as data inside an interpreter one-liner — the third axis, which
#: `_INTERPRETER_FORM` could not see because it keys on a verb WORD.
NOW_ONELINER_REFUSED = [
    ("python3 -c \"import subprocess; subprocess.run("
     "['systemctl','--user','enable','--now','agent-supervisord.service'])\"",
     "systemctl start (in a one-liner)"),
    ("python3 -c \"import subprocess; subprocess.run("
     "['systemctl','--user','disable','--now','agent-supervisord.service'])\"",
     "systemctl stop (in a one-liner)"),
]


@pytest.mark.parametrize("command,performed", NOW_REFUSED)
def test_the_now_flag_form_of_a_service_start_is_refused(command, performed):
    """Clause 1: `enable --now` / `reenable --now` on a non-`.timer` unit is a unit
    START and `disable --now` is a unit STOP, and a background session is refused each
    with the service-control reason that already covers the bare verb.

    The label is asserted exactly, not just truthily: `systemctl start` / `systemctl
    stop` is what makes the refusal readable in the transcript and what the grant axis
    in `app/harness/safety.py` tiers, so a label that merely fired would let the two
    axes drift while every assertion here stayed green.
    """
    assert find_service_control(command) == performed, command
    assert check_service_control(command, WORKER), command
    assert check_service_control(command, AUTONOMY), command
    assert check_service_control(command, UNKNOWN_TASK), \
        "an unresolvable task id is UNKNOWN, and UNKNOWN is refused"
    match = safety.check_bash_command(command, None, session_id=WORKER)
    assert match and match[0].startswith("service control:"), match
    assert performed in match[0], f"the refusal must name the operation performed: {match}"
    dispatch = safety.check_bash_command(command, None, at_dispatch=True,
                                        session_id=WORKER)
    assert dispatch and dispatch[0].startswith("service control:"), \
        "the aggregator's dispatch refuses too, as `test_a_worker_session_is_refused` " \
        "requires of the verb-word shapes"


def test_a_timer_armed_inside_a_one_liner_is_not_refused_by_trailing_code():
    """The exemption on the axis where the tokens after `systemctl` are arbitrary source
    text rather than argv.

    The window this guard reads runs 160 characters, so anything the payload does
    afterwards is in it: `…['systemctl','--user','enable','--now',
    'lloyd-cert-renew.timer'])` followed by a print is the same arming command, and the
    first version of the fix read every non-flag token as a unit, so the trailing code
    became a second, unexempt "unit" and the route #2537 exists to keep reachable was
    refused. Only a token that NAMES a systemd unit type may claim the exemption now;
    `test_a_bare_enable_or_disable_without_the_flag_is_still_only_unit_file_state` holds
    the other side, where no typed unit is named at all and the operation stands.
    """
    armed = ("python3 -c \"import subprocess; subprocess.run("
             "['systemctl','--user','enable','--now','lloyd-cert-renew.timer']); "
             "print('armed')\"")
    assert find_service_control(armed) is None, find_service_control(armed)
    assert check_service_control(armed, WORKER) is None
    assert safety.check_bash_command(armed, None, session_id=WORKER) is None

    armed_first = ("python3 -c \"print('go'); import subprocess; subprocess.run("
                   "['systemctl','--user','disable','--now','lloyd-cert-renew.timer'])\"")
    assert find_service_control(armed_first) is None, find_service_control(armed_first)

    mixed = ("python3 -c \"import subprocess; subprocess.run("
             "['systemctl','--user','disable','--now','agent-supervisord.service']); "
             "subprocess.run(['systemctl','--user','enable','--now',"
             "'lloyd-cert-renew.timer'])\"")
    assert find_service_control(mixed) == "systemctl stop (in a one-liner)", \
        "one typed unit that is not a timer in the window is enough: the exemption is " \
        "for a command that arms nothing but schedules"


@pytest.mark.parametrize("command,expected", NOW_ONELINER_REFUSED)
def test_the_now_flag_form_in_an_interpreter_one_liner_is_refused(command, expected):
    """Clause 3: the same flag form written as data, not argv, is refused the way the
    `start` shape already was.

    Measured at 01d0082b this exact argv list, written into a `python3 -c` payload,
    answered `label=None refused=False`, while the identical list carrying `start`
    answered `systemctl start (in a one-liner)` and was refused — the third axis the
    item found and the one a fix to `_segment_label` alone would have left open.
    """
    assert find_service_control(command) == expected, command
    assert check_service_control(command, WORKER), command
    match = safety.check_bash_command(command, None, session_id=WORKER)
    assert match and match[0].startswith("service control:"), match


@pytest.mark.parametrize("command", NOW_TIMER_ALLOWED)
def test_arming_or_unarming_a_timer_with_now_stays_reachable(command):
    """Clause 4 (#2537, and the scope call the item settles): `--now` against a
    `.timer` stays tier 1 with no service-control label, so the enabledness watchdog's
    FAIL line and SETUP.md's timer routes remain commands an unattended turn may run.

    Arming a schedule starts no unit, and unarming kills nothing in flight — which is
    what this guard's own docstring says it exists to stop. The service half of the
    pair is `test_the_now_flag_form_of_a_service_start_is_refused` above; the two
    together are the whole rule, and neither half is the default.
    """
    assert find_service_control(command) is None, command
    assert check_service_control(command, WORKER) is None, command
    assert check_service_control(command, UNKNOWN_TASK) is None, command
    assert safety.check_bash_command(command, None, session_id=WORKER) is None, command
    assert safety.bash_command_tier(command) == 1, command


@pytest.mark.parametrize("command", [
    "systemctl --user enable lloyd-cert-renew.timer",
    "systemctl --user disable agent-supervisord.service",
])
def test_a_bare_enable_or_disable_without_the_flag_is_still_only_unit_file_state(command):
    """The other half of the fix's cost: `enable` and `disable` with no `--now` change
    which unit files are linked and run nothing, so they stay allowed to a worker turn.

    #2541 classifies a flag, not a word. Tiering the bare verbs would break the route
    `install-services.sh` and SETUP.md document for a rebuild, and it is not what the
    flag does: without `--now` the unit keeps the state it had.
    """
    assert find_service_control(command) is None, command
    assert check_service_control(command, WORKER) is None, command


# ── #2948: the candidate unit's own header says who "by hand" means ──────────
DECIDER_CONF = CONF_DIR / "agent-decider.conf"

#: The two paths the hand-boot paragraph has to name. Named here once so the node
#: below and the node that anti-rotates it cannot drift from each other.
GUARD_MODULE = "app/harness/service_control.py"
PREFLIGHT_SUITE = "tests/test_llm_slot_vram_preflight.py"


def _hand_boot_paragraph(text: str) -> str:
    """The unit's hand-boot block: from its CANDIDATE claim down to the line before
    the `autostart=false` paragraph.

    Slice boundaries chosen by TEXT, not by line number — this round is the proof that
    a `file:line` citation moves under an edit to the file it cites: the paragraph grew
    by 19 lines here and every number pointing below it in
    `tests/test_service_health_check_supervisor_state.py` had to be re-pointed. A node
    that re-created that bug by slicing on `:5-13` would rot the next time someone
    added a sentence.
    """
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if "A CANDIDATE, NOT A SERVICE YET" in ln)
    end = next(i for i, ln in enumerate(lines)
               if ln.startswith("; autostart=false for the same reason"))
    return "\n".join(lines[start:end])


def test_the_candidate_unit_documents_who_may_boot_it_and_where_the_refusal_lives():
    """#2948: "It is started and stopped by hand" named an actor nobody could find.

    The line was true and useless. The hands that may run those three `supervisorctl`
    commands are a person's or an attended chat session's, and the unit said nothing
    about the other kind of reader this file gets — a worker, autonomy or round turn
    that types the command, is refused by `app/harness/service_control.py`, and reads
    the refusal as missing plumbing because the file above it just told it the route
    is "by hand". #2716's owed check weighed that block on 2026-10-18 and kept it, so
    the surviving half of the ruling is to make the file say what the rail does.

    Every claim below is a claim the paragraph now makes and the guard really implements
    — `test_the_paragraph_agrees_with_the_guard_and_the_probe_it_names` checks them
    against the code rather than against this docstring.
    """
    para = _hand_boot_paragraph(DECIDER_CONF.read_text(encoding="utf-8"))

    # Clause 1: the rail is named, and named as refusing BOTH acts to a background turn.
    assert GUARD_MODULE in para, (
        "the hand-boot paragraph does not name the module that refuses a background "
        f"turn, so a reader still cannot tell who 'by hand' excludes: {para}")
    refuse = [s for s in para.replace("\n", " ").split(".") if "refus" in s.lower()]
    assert refuse, f"the paragraph names the module but never says what it does: {para}"
    both = [s for s in refuse if "supervisorctl" in s and "boot" in s.lower()]
    assert both, (
        "the paragraph must say the same turn is refused BOTH the supervisorctl "
        f"start/stop AND the hand boot of the launcher — naming one alone leaves the "
        f"other looking like a way through: {refuse}")
    for actor in ("worker", "autonomy", "round"):
        assert actor in para.lower(), (
            f"the paragraph omits {actor!r} from the list of turns that are refused, "
            f"so a {actor} turn reading it cannot recognise itself")
    allowed = [s for s in para.replace("\n", " ").split(".")
               if "attended" in s.lower() or "person" in s.lower()]
    assert allowed, (
        f"the paragraph never names the actor that MAY do it — the exclusion reads as "
        f"a blanket ban on touching the unit: {para}")
    assert "None" in para, (
        "the paragraph should state the attended edge the way the code does: the call "
        "returns None for an attended id, so the absence of a refusal is the answer")

    # Clause 2: the unattended route is the shimmed suite, and the live card is not it.
    assert PREFLIGHT_SUITE in para, (
        "the paragraph does not name the suite that can exercise the guard without the "
        f"card, so an unattended turn has nowhere to go: {para}")
    probe = para.replace("\n", " ")
    assert "nvidia-smi" in probe and "shim" in probe.lower(), (
        "the named suite is not described as a PATH-shimmed `nvidia-smi`, which is what "
        f"makes it a probe rather than a boot: {probe}")
    assert "fake engine" in probe.lower(), (
        f"the paragraph does not say the engine is faked: {probe}")
    assert "not that surface" in probe or "not the surface" in probe, (
        "the paragraph must say the live card is NOT the probe surface — naming a "
        f"suite while leaving the real card looking available is half a warning: {probe}")
    assert "djev" in probe.lower(), (
        "the thing a successful hand-boot would displace is the ranker holding the "
        f"shared card, and that cost is the reason for the rail: {probe}")


def test_the_paragraph_agrees_with_the_guard_and_the_probe_it_names():
    """Clause 3's anti-rot: the sentence is checked against the two things it cites.

    #1447 is the precedent that a comment about this card silently described a machine
    that no longer existed, and #2948's sentence is the same species of claim — it
    asserts a fact about a module and about a test suite. So this node does not merely
    grep for the two paths: it re-derives each half of the claim from the thing named,
    and the paragraph goes red if EITHER side moves.

    Renaming `decider_fakes`/`boot_decider`, dropping the shim, or removing the
    decider's launcher from `_LAUNCHERS` all make the conf's sentence false while the
    sentence itself still reads perfectly. A node that only grepped the file would stay
    green through all three, which is how the last placement comment rotted.
    """
    para = _hand_boot_paragraph(DECIDER_CONF.read_text(encoding="utf-8"))
    # CONF_DIR is `<root>/agent-services/supervisor/conf.d`.
    root = CONF_DIR.parents[2]

    # The sentence itself: both paths named, or there is nothing here to cross-check.
    assert GUARD_MODULE in para and PREFLIGHT_SUITE in para, (
        "the hand-boot paragraph stopped naming the guard and the probe surface it was "
        f"written to point at, so the claim this node exists to test is gone: {para}")

    # (a) The module the paragraph names is the module that is loaded and enforcing.
    assert (root / GUARD_MODULE).exists(), (
        f"{GUARD_MODULE} moved, and the paragraph still points at it")
    assert service_control.__name__ == "app.harness.service_control", service_control.__name__

    # (b) The launcher the unit boots is in that module's launcher set, and is refused
    #     for a background session and allowed for a chat — read out of THIS conf's own
    #     `command=`, so the node follows the unit if the script is ever renamed.
    command_line = next(ln for ln in DECIDER_CONF.read_text(encoding="utf-8").splitlines()
                        if ln.startswith("command="))
    launcher = next(tok for tok in command_line.split() if tok.endswith(".sh"))
    assert launcher.rsplit("/", 1)[-1] in service_control._LAUNCHERS, (
        f"{launcher} is what this unit runs, and it is not in _LAUNCHERS — so the "
        "paragraph's 'both the supervisorctl start/stop AND the hand boot' is false "
        "for the boot half")
    assert check_service_control(f"bash {launcher}", WORKER), (
        f"the paragraph promises a background turn is refused the hand boot of {launcher}")
    assert check_service_control(f"bash {launcher}", AUTONOMY), "autonomy too"
    assert check_service_control(f"bash {launcher}", CHAT) is None, (
        "the paragraph says an attended session may boot it, and the guard must agree")
    for verb in ("start", "stop"):
        assert check_service_control(f"supervisorctl -c x.conf {verb} agent-decider",
                                     WORKER), (
            f"the paragraph claims `supervisorctl {verb} agent-decider` is refused to a "
            f"background turn; it is not, and the sentence is now wrong")
        assert check_service_control(f"supervisorctl -c x.conf {verb} agent-decider",
                                     CHAT) is None

    # (c) The suite the paragraph points an unattended turn at still reaches the
    #     decider's guard through fakes rather than through the card.
    suite = root / PREFLIGHT_SUITE
    assert suite.exists(), f"{PREFLIGHT_SUITE} moved, and the paragraph still points at it"
    src = suite.read_text(encoding="utf-8")
    # Structural, not a grep for the word "shim": that word appears in prose and in
    # comments, so it survives the harness being deleted — measured, a run with all 68
    # occurrences of `shim` in the file replaced still passed this node. What carries
    # the claim is the fixtures that build the fake binary and hand it to the child, and
    # the two helpers the paragraph names by title.
    for helper in ("decider_fakes", "boot_decider", "shim_dir"):
        assert f"def {helper}" in src, (
            f"the paragraph rests on `{helper}` — as the way in, or as the fixture that "
            f"builds the fake `nvidia-smi` — and the suite no longer defines it")
    assert "nvidia-smi" in src, (
        "the fake binary this harness substitutes is no longer `nvidia-smi`, so the "
        "guard the paragraph points at is not the one being exercised")
    assert '{shim_dir}:{SYSTEM_PATH}' in src, (
        "the fake `nvidia-smi` reaches the script by sitting FIRST on the child's PATH; "
        "without that ordering the harness runs the real binary, which is the live-card "
        "run the paragraph exists to warn a round away from")
