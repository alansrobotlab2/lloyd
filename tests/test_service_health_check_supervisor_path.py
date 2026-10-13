"""The checker's own supervisorctl must be a path, and a read that never happened must
not be billed as a service being down (#2652).

Two defects, one cause. `scripts/service_health_check.py` spelled argv[0] as the bare
`"supervisorctl"` in every one of its supervisor reads — the seven `SERVICES` command
lists and the two calls in `supervisor_fleet()` and `_ask_program()` — and this script
runs from a shell whose PATH is `/usr/local/bin:/usr/bin`, where that name is not. So
`subprocess.run` raised `FileNotFoundError` before supervisor was contacted, six
services that were RUNNING printed `[✗] error`, `Overall:` printed `15/23`, and the one
row that did explain why (`could not read the loaded program list ([Errno 2] No such
file or directory: 'supervisorctl')`) was counted as a fault too. The control at
2026-10-13T08:24Z: `which supervisorctl` empty under that PATH, while the same binary by
absolute path answers with the real fleet.

What this file pins:

1. argv[0]. Every supervisor argv list the module builds starts with an absolute path to
   an existing executable — including the two built inside functions, which a `SERVICES`
   grep cannot see. Because the existing tests replace `shc.subprocess` wholesale
   (`tests/test_service_health_check_supervisor_state.py:97`) and so never hand an argv[0]
   to the kernel, the rule here is that argv[0] is never made reachable by PATH: the
   binary is named by absolute path, either the box's real supervisorctl or one this file
   writes into a temp dir, and
   `test_the_stand_in_is_really_reached_and_its_answer_is_graded` execs one through the
   module's own unstubbed `subprocess.run`.

   Three nodes do rebind `shc.subprocess.run`, and none of them answers in place of a
   binary the way the state file's stub does:
   `test_every_supervisor_argv_the_module_actually_builds_is_absolute` and
   `test_a_fleet_that_was_never_read_asks_about_no_program_at_all` install a recorder that
   notes argv and then forwards to the real `subprocess.run` (recording what was passed
   and execing it are separate claims, and the exec has to be the one that proves a path
   is runnable); `test_a_read_that_timed_out_is_a_graded_answer_not_an_unverified_one`
   raises `subprocess.TimeoutExpired`, which is what a real hang raises and the only way
   to observe it without holding the suite for the module's own 5-second timeout. No node
   here replaces the `shc.subprocess` module object. That is why no test caught this.
2. What a read that never started is worth: `unverified`, printed with the reason the
   except arm had already computed, and left out of both sides of `Overall: N/M` the way
   an advisory row is. A row the checker never got to read is not evidence about the
   service named on it.

The stand-in binaries here are written to disk and exec'd for real. The one thing not
patched is `subprocess` itself.
"""
import ast
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MARKER = "supervisor-path-marker"

#: The PATH this script is invoked with by the harness shell (triage, 2026-10-13:
#: `/usr/local/bin:/usr/bin`). `supervisorctl` is not on it; that is the defect.
HARNESS_PATH = "/usr/local/bin:/usr/bin"

#: One program line, the shape `supervisorctl status` prints: name, state, pid.
ONE_RUNNING = "lloyd-mc:lloyd-mcp  RUNNING   pid 7, uptime 2:00:00"


def _load(monkeypatch, *, path=HARNESS_PATH):
    """The real script, imported with PATH set to the one the harness shell uses.

    The PATH matters: argv[0] is resolved at import, so importing under a PATH that
    happens to contain supervisorctl would pass whatever the module did.
    """
    monkeypatch.setenv("PATH", path)
    script = REPO / "scripts" / "service_health_check.py"
    text = script.read_text(encoding="utf-8")
    assert MARKER not in text
    code = compile(text + f"\n# {MARKER}\n", str(script), "exec")
    mod = types.ModuleType("shc_supervisor_path")
    mod.__dict__["__file__"] = str(script)
    exec(code, mod.__dict__)
    return mod


@pytest.fixture()
def shc(monkeypatch):
    mod = _load(monkeypatch)
    sys.modules.setdefault("shc_supervisor_path", mod)
    monkeypatch.setattr(mod, "_FLEET", None)
    return mod


def _stub(mod, monkeypatch, tmp_path, *, body, name="supervisorctl", executable=True):
    """A stand-in supervisor binary on disk, and the module pointed at it.

    Written to disk rather than wrapped in a fake `subprocess`, so the exec, the exit
    code and the argv the child receives are the real ones.
    """
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755 if executable else 0o644)
    assert bool(os.access(path, os.X_OK)) == executable
    monkeypatch.setattr(mod, "SUPERVISORCTL", str(path))
    return path


def _logging_stub(mod, monkeypatch, tmp_path, *, log_name, answer=ONE_RUNNING, rc=3):
    """A stand-in that records the argv it was exec'd with, then answers with `answer`."""
    log = tmp_path / log_name
    body = ('{ printf "%s\\n" "$0"; printf "%s\\n" "$@"; } > ' + str(log) + "\n"
            'printf "' + answer + '\\n"\n' + "exit " + str(rc) + "\n")
    _stub(mod, monkeypatch, tmp_path, body=body)
    return log


def _argv(log):
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines, f"{log} was never written: the child never ran"
    return lines


# ── clause 1: argv[0] is a resolved path, at the boundary, everywhere ────────

def test_under_the_harness_path_the_name_does_not_resolve_but_the_module_has_it(shc):
    """The premise and the fix in one view: the name is unfindable, the module has it.

    `shutil.which` is asked first inside the module, so if supervisorctl ever lands on
    the harness PATH the fallback stops being load-bearing and a green run here would say
    nothing about the fix — which is why the miss is asserted in the same environment
    rather than assumed from triage.
    """
    import shutil as _shutil
    assert _shutil.which("supervisorctl") is None, (
        "supervisorctl resolves on the harness PATH in this environment, so #2652 no "
        "longer reproduces here and this file needs a different control")
    assert os.path.isabs(shc.SUPERVISORCTL), shc.SUPERVISORCTL
    assert os.access(shc.SUPERVISORCTL, os.X_OK), shc.SUPERVISORCTL
    assert os.path.basename(shc.SUPERVISORCTL) == "supervisorctl", shc.SUPERVISORCTL


def test_no_supervisor_command_the_module_builds_starts_with_the_bare_name():
    """No supervisor argv list survives anywhere in the source with argv[0] spelled bare.

    An AST sweep rather than a grep, so a site inside a function is counted, and with a
    positive control: the same walk has to find the resolved form, or "zero bare sites"
    would be a pattern that matched nothing at all.
    """
    src = (REPO / "scripts" / "service_health_check.py").read_text(encoding="utf-8")
    bare, resolved = [], []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.List) and node.elts:
            head = node.elts[0]
            if isinstance(head, ast.Constant) and head.value == "supervisorctl":
                bare.append(node.lineno)
            elif isinstance(head, ast.Name) and head.id == "SUPERVISORCTL":
                resolved.append(node.lineno)
    assert not bare, f"bare argv[0] still built at line(s) {bare}"
    assert len(resolved) >= 9, (
        f"only {len(resolved)} argv lists name SUPERVISORCTL, expected the seven "
        "`SERVICES` entries plus the calls in supervisor_fleet() and _ask_program(); a "
        "'zero bare sites' answer out of a sweep this narrow is an empty scan, not a fix")


def test_every_supervisor_argv_the_module_actually_builds_is_absolute(shc, tmp_path,
                                                                     monkeypatch):
    """The seven `SERVICES` lists, the fleet read and the per-name re-ask, at the exec.

    Nothing is patched over the call itself: the stand-in records the argv the kernel
    received. A list literal inside a function is exactly where a re-spelled bare name
    would hide, and `SERVICES` alone cannot see it.
    """
    import shutil as _shutil
    assert _shutil.which("supervisorctl") is None, "see the control above"
    sup = [n for n, s in shc.SERVICES.items() if shc.is_supervisor_call(
        s.get("command") or [])]
    assert len(sup) == 7, (
        f"expected the seven supervisor `SERVICES` entries #2652 names, got {sup}")
    for name in sup:
        cmd = shc.SERVICES[name]["command"]
        assert cmd[0] != "supervisorctl", (name, cmd)
        assert os.path.isabs(cmd[0]) and os.access(cmd[0], os.X_OK), (name, cmd)

    log = _logging_stub(shc, monkeypatch, tmp_path, log_name="fleet.log")
    shc.supervisor_fleet()
    fleet_argv = _argv(log)
    assert fleet_argv[0] == str(tmp_path / "supervisorctl"), fleet_argv
    assert fleet_argv[1:3] == ["-c", str(shc.SUPervisor_CONF)], fleet_argv
    assert fleet_argv[3:] == ["status"], fleet_argv

    # The cache contract is the tuple `supervisor_fleet()` returns — (fleet, error) —
    # not the bare dict, so this seeds what a real second call would have read.
    monkeypatch.setattr(
        shc, "_FLEET",
        ({p.split()[0]: p.split()[1] for p in ONE_RUNNING.splitlines()}, None))
    monkeypatch.setattr(shc, "SUPervisor_CONF", tmp_path / "supervisord.conf")
    shc._ask_program("agent-tts")
    asked = _argv(log)
    assert asked[0] == str(tmp_path / "supervisorctl"), asked
    assert asked[3:] == ["status", "agent-tts"], asked


def test_the_stand_in_is_really_reached_and_its_answer_is_graded(shc, tmp_path,
                                                                monkeypatch):
    """The boundary this file claims to cross: a child ran, and its answer was graded.

    Without this node every assertion above could be satisfied by a module that never
    calls subprocess at all.
    """
    _stub(shc, monkeypatch, tmp_path,
          body='printf "' + ONE_RUNNING + '\\n"\nexit 3\n')
    fleet, error = shc.supervisor_fleet()
    assert error is None, error
    assert fleet == {"lloyd-mc:lloyd-mcp": "RUNNING"}, fleet


# ── clause 3: an unreadable list is unverified, not a fleet of faults ────────

def test_a_program_list_that_was_never_read_is_unverified_and_not_billed(shc, tmp_path,
                                                                        monkeypatch):
    """Exec fails -> nothing was learned -> the row says so and leaves the fraction.

    The stand-in is not executable, so the exec raises the same `OSError` family the real
    defect raised as `FileNotFoundError`. `rc != 0` is not this path — that is an answer,
    and clause 4's node is where the two are told apart.
    """
    _stub(shc, monkeypatch, tmp_path, body="exit 0\n", executable=False)
    rows = shc.derived_supervisor_rows()
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["unverified"] is True, row
    assert row["status"].startswith("unverified"), row
    assert ("no undeclared program was graded, so this run covered only the declared "
            "services") in row["status"], row
    healthy = {"name": "lloyd-backend", "category": "lloyd", "healthy": True,
               "status": "RUNNING", "exit_code": 0}
    assert shc._scored_rows([row, healthy]) == [healthy], "billed after all"
    assert shc._scored_rows([row]) == []


def test_an_unverified_row_still_prints_with_a_heading_of_its_own(shc, tmp_path,
                                                                 monkeypatch):
    """Excluded from the arithmetic must not mean invisible."""
    _stub(shc, monkeypatch, tmp_path, body="exit 0\n", executable=False)
    rows = shc.derived_supervisor_rows() + [
        {"name": "lloyd-backend", "category": "lloyd", "healthy": True,
         "status": "RUNNING", "exit_code": 0}]
    out = shc.format_text(rows, {"lloyd": "healthy"})
    assert "Overall: 1/1 services healthy" in out, out
    assert "Unverified: 1 row(s)" in out, out
    assert "[?] supervisorctl status" in out, out


# ── clause 5: the reason the except arm computed is printed ──────────────────

def test_a_row_whose_command_never_ran_prints_the_reason_not_just_the_word_error(shc,
                                                                                monkeypatch):
    """`output` was already computed; `format_text` was the half that dropped it.

    The command cannot exec, spelled the way the defect spelled it, so the printed line
    carries the string triage recorded — `[Errno 2] No such file or directory` — instead
    of the bare word `error`.
    """
    monkeypatch.setenv("PATH", HARNESS_PATH)
    row = shc.check_service("lloyd-backend", {
        "command": ["/nonexistent-dir/supervisorctl", "-c", "/nonexistent/conf",
                    "status", "lloyd-mc:lloyd-backend"],
        "category": "lloyd"})
    assert row["status"] == "unverified", row
    assert row["unverified"] is True, row
    assert "No such file or directory" in row["output"], row
    out = shc.format_text([row], {})
    # The `Unverified:` heading names the row too, so the row itself is the line that
    # opens with an icon.
    line = next(l for l in out.splitlines() if l.startswith("["))
    assert "lloyd-backend" in line, line
    assert "[?]" in line, line
    assert "[Errno 2] No such file or directory" in line, line
    assert "Unverified: 1 row(s)" in out, out


def test_a_command_that_ran_and_exited_nonzero_is_still_a_graded_row(shc, tmp_path,
                                                                    monkeypatch):
    """The other half of `unverified`: a command that ran is graded, whatever it exits.

    Without this pair the new marker would be a way to make any failure quietly vanish
    from `Overall:`.
    """
    _stub(shc, monkeypatch, tmp_path, body='printf "boom\\n" >&2\nexit 7\n')
    row = shc.check_service("lloyd-mcp", {
        "command": [str(tmp_path / "supervisorctl"), "-c", "/nonexistent/conf",
                    "status", "lloyd-mc:lloyd-mcp"],
        "category": "lloyd"})
    assert row.get("unverified") is None, row
    assert row["healthy"] is False and row["exit_code"] == 7, row
    assert "boom" in row["status"], row


# ── the two limits the new marker must not swallow ───────────────────────────

def test_a_fleet_that_was_never_read_asks_about_no_program_at_all(shc, tmp_path,
                                                                monkeypatch):
    """The short circuit is a real boundary, not an optimisation.

    `derived_supervisor_rows` returns the unverified row before its per-program loop, so
    one failed exec stays one failed exec. Without that order a box whose supervisor is
    unreachable would fire one probe per program — the same miss, paid for twelve times
    over and then reported as twelve rows. Pinned from both sides: the spy raises if a
    per-program probe is attempted, and the exec log counts the attempts that actually
    happen, so "one unread list, one row, one exec" is measured rather than inferred from
    the row count.
    """
    monkeypatch.setattr(shc, "SUPERVISORCTL", str(tmp_path / "gone" / "supervisorctl"))
    shc._FLEET = None

    def _no_probe(*a, **k):
        raise AssertionError(
            "an unread program list must not be probed per program")

    monkeypatch.setattr(shc, "_ask_program", _no_probe)
    log: list = []
    real_run = shc.subprocess.run

    def _record(*a, **k):
        log.append(list(a[0]) if a else list(k.get("args", [])))
        return real_run(*a, **k)

    monkeypatch.setattr(shc.subprocess, "run", _record)

    rows = shc.derived_supervisor_rows()
    assert len(rows) == 1, rows
    assert rows[0]["name"] == "supervisorctl status", rows[0]
    assert rows[0]["unverified"] is True, rows[0]
    assert len(log) == 1, f"the unread list cost {len(log)} execs, not one: {log}"
    assert log[0][3:] == ["status"], log[0]


def test_a_read_that_timed_out_is_a_graded_answer_not_an_unverified_one(shc, monkeypatch):
    """The boundary of `unverified`, on the exception side.

    A timeout is a process that started and did not answer: supervisor was reached, and
    its silence is information about the service. Execing nothing taught us nothing, and
    that is the only thing `unverified` claims (#2652). Without this node the marker could
    quietly widen to every exception, and a wedged service would leave `Overall:` instead
    of turning its row red.
    """
    import subprocess as _sp

    def _hang(*a, **k):
        raise _sp.TimeoutExpired(cmd=["supervisorctl", "status"], timeout=5)

    monkeypatch.setattr(shc.subprocess, "run", _hang)
    row = shc.check_service("lloyd-mcp", {
        "command": [shc.SUPERVISORCTL, "-c", "/nonexistent/conf", "status",
                    "lloyd-mc:lloyd-mcp"],
        "category": "lloyd"})
    assert row["status"] == "timeout", row
    assert "unverified" not in row, row
    assert row["healthy"] is False, row
    assert shc._scored_rows([row]) == [row], "a timeout left the fraction"
