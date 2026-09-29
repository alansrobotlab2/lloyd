"""#1853 — the autouse guard that stops a test from running systemctl/pkill/killall.

`tests/conftest.py::_no_live_service_control_in_tests` refuses a `subprocess.run` or
`subprocess.call` whose command name is `systemctl`, `pkill` or `killall`, because
`tick()` issues a real `systemctl --user restart agent-supervisord.service` once its
supervisord streak reaches `SUPERVISORD_DOWN_STREAK` (3) at
`agent-services/guardian/guardian.py:1233-1236`, `_stop_sync` issues a real
`pkill -f "sync --path …"` at `guardian.py:1054`, and the promoter reaches
`systemctl --user daemon-reload` and `systemctl --user restart lloyd-guardian` at
`scripts/automod/promote.py:586` and `:595` — the second through the thin `_run`
wrapper at `promote.py:542`, which calls the module-global `subprocess.run`. Before this fixture the only protection
was each node remembering its own stub, so a new module re-inherited the trap.

Every node here also prepends a directory of inert stand-ins for the three commands to
`PATH` (`_inert_danger_bin`), and each stand-in appends to a log instead of acting. That
is what makes this file safe to run when the guard is ABSENT — which is the state a
regression puts it in, and the state the negative control was measured in: without the
fixture, `subprocess.run(["systemctl", "--user", "restart",
"agent-supervisord.service"])` resolves to the stub and returns, so the node below
fails on its own `pytest.raises` instead of restarting production's supervisor. The
log is also the proof of the *ordering* property: the guard must refuse **before** the
child is spawned, so a refused node leaves the log empty.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import conftest  # noqa: E402  the repo's only conftest, same object pytest loaded

STUB_LOG_VAR = "SUBPROCESS_STUB_LOG"
#: The unit the guardian restarts, and the real name behind it (`policy.py:293`).
SUPERVISORD_UNIT = "agent-supervisord.service"


@pytest.fixture
def danger_bin(tmp_path, monkeypatch) -> Path:
    """`systemctl`, `pkill` and `killall` stand-ins on PATH, logging instead of acting.

    Module-scoped names, function-scoped log: a node asks for this and gets a log file
    nothing else can write to, so "the log is empty" means *this node* spawned nothing.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "spawned.log"
    for name in ("systemctl", "pkill", "killall"):
        stub = bin_dir / name
        stub.write_text("#!/usr/bin/env bash\n"
                        "printf '%s %s\\n' \"$0\" \"$*\" >> \"${" + STUB_LOG_VAR + "}\"\n"
                        "exit 0\n", encoding="utf-8")
        stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv(STUB_LOG_VAR, str(log))
    return log


def _spawned() -> str:
    """What the stand-ins recorded, or the empty string when nothing was spawned."""
    log = os.environ.get(STUB_LOG_VAR)
    return Path(log).read_text() if log and Path(log).exists() else ""


# ------------------------------------------------ the refusal itself (clause 1) ----

def test_the_guard_refuses_a_systemctl_argv_and_names_that_argv(danger_bin):
    """The probe node the item names: this argv must fail, not return.

    Today (before #1853) `subprocess.run(["systemctl", "--user", "restart",
    "agent-supervisord.service"])` from a test *returns* — with the guard it raises, and
    the message carries the argv so a reader of a red node can see which command was
    reached for.
    """
    with pytest.raises(conftest.LiveServiceControlRefused) as got:
        subprocess.run(["systemctl", "--user", "restart", SUPERVISORD_UNIT],
                       capture_output=True, text=True, timeout=60, check=False)
    msg = str(got.value)
    assert "systemctl" in msg, msg
    assert "restart" in msg, msg
    assert SUPERVISORD_UNIT in msg, f"the argv is not in the message: {msg}"


def test_a_refused_command_is_never_spawned(danger_bin):
    """The guard refuses BEFORE executing, which is the whole point of it.

    A refusal that raised only after `Popen` had forked would still have restarted the
    unit; the stand-in log is the only witness that can tell those two apart, and it has
    to stay empty.
    """
    with pytest.raises(conftest.LiveServiceControlRefused):
        subprocess.run(["systemctl", "--user", "daemon-reload"])
    assert _spawned() == "", "the refused command reached the stub anyway"


def test_a_shell_string_is_refused_on_its_first_token(danger_bin):
    """`subprocess.run("systemctl …", shell=True)` is the same hazard in one string.

    `subprocess.run` takes either an argv list or a single string for a shell, so the
    command name has two spellings and both have to be read.
    """
    with pytest.raises(conftest.LiveServiceControlRefused) as got:
        subprocess.run("systemctl --user restart lloyd-guardian", shell=True)
    assert "systemctl" in str(got.value), str(got.value)
    assert _spawned() == ""


def test_pkill_killall_an_absolute_path_and_the_args_keyword_are_all_refused(danger_bin):
    """The other two names, and the three ways a test can still spell a command.

    `_stop_sync` reaches for `pkill` at `guardian.py:1054`, so the set is the item's
    three names — and the set is read off the COMMAND, not off argv[0]'s literal
    spelling: `/tmp/…/systemctl` is the same program as `systemctl`. The `args=` keyword
    is the same call spelled as a keyword (`subprocess.run` forwards it to `Popen`), so
    a guard that read only the positional would be a guard with a door left open.
    """
    for argv in (["pkill", "-f", "sync --path /nonexistent-for-this-node"],
                 ["killall", "-r", "lloyd-nothing-1853"],
                 [str(danger_bin / "systemctl"), "--user", "restart", SUPERVISORD_UNIT]):
        with pytest.raises(conftest.LiveServiceControlRefused) as got:
            subprocess.run(argv, capture_output=True, timeout=10)
        assert "restart" in str(got.value) or "pkill" in str(got.value) \
            or "killall" in str(got.value), str(got.value)
    with pytest.raises(conftest.LiveServiceControlRefused):
        subprocess.run(args=["pkill", "-f", "nothing-1853"], timeout=10)
    assert _spawned() == ""


def test_the_danger_set_is_exactly_the_three_named_commands():
    """`supervisorctl` stays out, and no token but the command name is ever read.

    Refusing the name `supervisorctl` would refuse `supervisorctl status`, which is a
    read the suite asks for on purpose (`tests/test_service_control_guard.py`'s ALLOWED
    list), so the set is the item's three words. The second half is what keeps
    `tests/test_data_home.py:1404` green: that node stops a whole fake fleet with
    `subprocess.run(["bash", "-c", program])` (its runner, `tests/test_data_home.py:1330`) and a STUB `systemctl` first on PATH, so
    its command name is `bash` and the word `systemctl` sits inside one quoted token of
    the program — reading any later token would refuse a node that never touches a unit.
    """
    assert conftest.DANGEROUS_SUBPROCESS_COMMANDS == {"systemctl", "pkill", "killall"}
    assert conftest._dangerous_subprocess_command(
        ["supervisorctl", "status"]) is None
    assert conftest._dangerous_subprocess_command(
        ["bash", "-c", "systemctl --user stop lloyd-guardian"]) is None
    assert conftest._dangerous_subprocess_command(
        ["/usr/bin/pkill", "-f", "sync"]) == "pkill"
    assert conftest._dangerous_subprocess_command("killall lloyd-x") == "killall"
    assert conftest._dangerous_subprocess_command(["bash", "-c", "echo ok"]) is None
    assert conftest._dangerous_subprocess_command([]) is None


# ------------------------------------------------- everything else still runs (2) ----

def test_every_other_command_still_runs_for_real(danger_bin):
    """The guard is a name filter, not a sandbox: `bash -c 'echo ok'` must execute.

    Clause 2's bare form. If this returns anything but rc 0 and stdout `ok`, the
    passthrough is broken and roughly every test that shells out is about to go red —
    including the four real-script call sites in `tests/test_guardian_vaultwatch.py`
    (`:158`, `:189`, `:217`, `:225`, reached from
    `test_sync_will_not_start_while_tripped_or_over_a_gutted_vault` and
    `test_snapshots_record_changes_refuse_a_shrunken_vault_and_restore_aside`), which are
    the nodes the item names as must-not-change.
    """
    done = subprocess.run(["bash", "-c", "echo ok"], capture_output=True,
                          text=True, timeout=30)
    assert done.returncode == 0, done
    assert done.stdout.strip() == "ok", done


def test_a_shell_string_that_mentions_a_danger_command_still_runs(danger_bin):
    """The half that must NOT be refused, run rather than asserted about.

    Same shape as `tests/test_data_home.py:1404`: the command name is `bash`, and
    `systemctl` only appears inside the program text. This node prints the word instead
    of executing it, so it proves the passthrough without needing the stub.
    """
    done = subprocess.run(["bash", "-c", "echo 'systemctl --user stop example'"],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done
    assert "systemctl" in done.stdout, done


# ---------------------------------------- the same door under a different name (4) ----

def test_check_call_and_call_are_refused_through_the_module_global(danger_bin):
    """`check_output` and `check_call` cannot walk past a guard on `run` alone.

    Both were claimed to "route through the module-global `run`", and measured on this
    interpreter (CPython 3.12.14) only half of that holds: `check_output` calls
    `subprocess.run`, while `check_call` calls `call`, which calls `Popen` — so a guard
    on `run` alone leaves `check_call` wide open. The fixture therefore wraps the two
    module-level entry points, `run` and `call`, and this node drives all four spellings.
    """
    with pytest.raises(conftest.LiveServiceControlRefused):
        subprocess.check_output(["systemctl", "--user", "restart", SUPERVISORD_UNIT])
    with pytest.raises(conftest.LiveServiceControlRefused):
        subprocess.check_call(["systemctl", "--user", "restart", SUPERVISORD_UNIT])
    with pytest.raises(conftest.LiveServiceControlRefused):
        subprocess.call(["pkill", "-f", "nothing-1853"])
    with pytest.raises(conftest.LiveServiceControlRefused):
        subprocess.check_output("killall lloyd-nothing-1853", shell=True)
    assert _spawned() == "", "one of the four spellings reached the stub"


# ------------------------------------------- an existing node's stub still wins (3) ----

def test_a_node_that_installs_its_own_stub_still_wins(monkeypatch, danger_bin):
    """The fixture is a floor, not a replacement for the per-node stubs.

    `tests/test_guardian_pool_watch.py:357` and `tests/test_guardian_predicates.py:1080`
    record the restart argv from their own fake, and one of those nodes drives
    `tick()` past `SUPERVISORD_DOWN_STREAK` (3) and asserts the fake saw
    `["systemctl", "--user", "restart", SUPERVISORD_UNIT]`. Those nodes were green
    before this fixture and must stay green, which holds because the node's
    `monkeypatch.setattr` is applied AFTER the autouse fixture's and `undo()` pops in
    reverse — so this is the same shape, and the guard stays out of the way.
    """
    seen: list[list[str]] = []

    def _fake_run(argv, **kw):
        seen.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    done = subprocess.run(["systemctl", "--user", "restart", SUPERVISORD_UNIT],
                          capture_output=True, timeout=60, check=False)
    assert done.returncode == 0, done
    assert seen == [["systemctl", "--user", "restart", SUPERVISORD_UNIT]], seen
    assert _spawned() == "", "a stubbed node fell through to the real command"


# ------------------------------------------- the promoter's own path to the unit (4) ----

def test_the_promoter_is_refused_before_it_restarts_the_live_guardian(monkeypatch,
                                                                      tmp_path):
    """`_apply_service_changes` with `_run` UNSTUBBED fails here, not on the unit.

    `scripts/automod/promote.py:542` `_run` is a thin `subprocess.run` wrapper, and the
    two nodes that exercise this function today (`tests/test_automod_hardening.py:1091`,
    `:1116`) defend the wrapper — a node that stubs neither would issue a real
    `systemctl --user restart lloyd-guardian` from inside the gate's pytest run. So the
    refusal has to reach that path through the wrapper, which it does because `_run`
    calls the module-global `subprocess.run`.

    `SYSTEMD_USER_DIR` is redirected into `tmp_path` first: the copy of the unit file
    into `~/.config/systemd/user` is a real write that this node has no business making,
    and asserting it happened is what proves the run got as far as the service-control
    call rather than bailing out somewhere earlier.
    """
    from scripts.automod import promote as P

    monkeypatch.setattr(P, "SYSTEMD_USER_DIR", tmp_path / "systemd")
    with pytest.raises(conftest.LiveServiceControlRefused) as got:
        P._apply_service_changes(["agent-services/systemd/lloyd-guardian.service"])
    assert "systemctl" in str(got.value), str(got.value)
    assert (tmp_path / "systemd" / "lloyd-guardian.service").is_file(), \
        "the unit was never installed, so this proves nothing about the restart"
    assert _spawned() == "", "daemon-reload reached systemd before the refusal fired"
