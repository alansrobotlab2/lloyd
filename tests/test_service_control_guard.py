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

    What it covers is the `.sh` programs — 7 of the 12 confs. The other 5 name a
    python or node entry point (`command=…/python …/server.py`,
    `…/python -m agent_mcp.main`, `…/node …/qmd.js`, `npm …`, the livekit worker),
    and this test does not reach them: refusing a hand-run `.py` would catch
    someone running that module under a test harness, which is a different
    question from booting an engine, and is left on #1363 rather than taken here.
    """
    launchers = _supervised_launchers()
    assert len(launchers) >= 7, f"expected the 7 supervised launchers, found {launchers}"
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
