"""#1649: the health check covers every program supervisor is running.

`SERVICES` in `scripts/service_health_check.py` had seven keys while
`supervisorctl -c CONF status` printed twelve programs, and those seven keys were
the whole scope of the default run. Measured on the round's base commit: the
script printed `Overall: 8/8 services healthy` and named 3 of the 12 programs
supervisor reported; `agent-livekit-server`, `agent-obsidian-sync`,
`agent-qmd-watcher`, `agent-tts` and `lloyd-agent-worker` were never read, so any
one of them sitting in FATAL kept the run green and the exit code at 0.

The fix derives coverage from `supervisorctl status`, so the boundary is what the
supervisor daemon has loaded rather than what this file remembers. The tests
below therefore feed a synthetic fleet that the real daemon does NOT have — an
`agent-brand-new-unit` the box has never run, three programs in states this box
is not in — because the property worth pinning is "a program nobody listed is
graded", which cannot be observed on a machine whose twelve programs are all
listed somewhere or another.

Loaded from the path, not imported: `scripts/` is not a package, and this is the
same route `tests/test_service_health_check_supervisor_state.py` uses.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"
SKILL = Path.home() / "obsidian" / "skills" / "service-health-check" / "SKILL.md"

# The twelve programs supervisor reports on this box today, verbatim, plus one
# unit this machine has never had. `agent-llm-secondary` is STOPPED by config, so
# `supervisorctl status` exits 3 for that reason alone on every real run.
FLEET_LINES = [
    "agent-djev                       RUNNING   pid 3439624, uptime 2 days, 14:34:33",
    "agent-livekit-server             RUNNING   pid 4262, uptime 3 days, 23:03:45",
    "agent-llm-primary                RUNNING   pid 2489, uptime 3 days, 23:03:52",
    "agent-llm-secondary              STOPPED   Not started",
    "agent-obsidian-sync              RUNNING   pid 2491, uptime 3 days, 23:03:52",
    "agent-qmd-daemon                 RUNNING   pid 2784521, uptime 2 days, 21:04:36",
    "agent-qmd-watcher                RUNNING   pid 2876803, uptime 3 days, 16:04:34",
    "agent-tts                        RUNNING   pid 2504, uptime 3 days, 23:03:52",
    "lloyd-agent-worker               RUNNING   pid 280881, uptime 3 days, 5:48:11",
    "lloyd-mc:lloyd-backend           RUNNING   pid 2186612, uptime 0:47:38",
    "lloyd-mc:lloyd-frontend          RUNNING   pid 2515, uptime 3 days, 23:03:52",
    "lloyd-mc:lloyd-mcp               RUNNING   pid 2173427, uptime 0:47:54",
    "agent-brand-new-unit             RUNNING   pid 999, uptime 0:00:04",
]
LIVE_PROGRAMS = [line.split()[0] for line in FLEET_LINES if "brand-new" not in line]
NEVER_READ = ["agent-livekit-server", "agent-obsidian-sync", "agent-qmd-watcher",
              "agent-tts", "lloyd-agent-worker"]


def _load():
    spec = importlib.util.spec_from_file_location("shc_cov", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Answer:
    """A stand-in `supervisorctl` that records every command it was asked to run.

    Faithful in the one way that matters: `status` with no name prints every loaded
    program, `status <name>` prints that program's line alone. Supervisor really
    behaves that way, and a stub that answered a per-name call with the whole fleet
    would hide #1040's bug rather than reproduce it — the script reads field 2 of a
    single line, and a multi-state blob has no field 2 that means anything.
    """

    def __init__(self, stdout, returncode=3, per_name=None):
        self.stdout, self.returncode = stdout, returncode
        # Per-name answers the fleet does not contain, for the one case that needs
        # them: a fleet line missing its state field that the named call can answer.
        self.per_name = dict(per_name or {})
        self.commands = []

    def answer_for(self, cmd):
        if len(cmd) > 4:                      # `status <program>`
            name = cmd[-1]
            if name in self.per_name:
                return self.per_name[name]
            lines = [ln for ln in self.stdout.splitlines()
                     if ln.split()[:1] == [name]]
            return lines[0] if lines else f"{name}    STOPPED   not started"
        return self.stdout

    def __call__(self, cmd, *a, **kw):
        self.commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.returncode,
                                           self.answer_for(cmd), "")


def _fleet(monkeypatch, lines, returncode=3, switched_off=("agent-llm-secondary",),
           per_name=None):
    """Load the script with `supervisorctl status` answered by `lines`."""
    mod = _load()
    answer = _Answer("\n".join(lines), returncode, per_name)
    monkeypatch.setattr(mod.subprocess, "run", answer)
    monkeypatch.setattr(mod, "_switched_off", lambda: set(switched_off))
    return mod, answer


# ---------------------------------------------------------------------- clause 1
def test_every_program_supervisor_reports_gets_a_row(monkeypatch):
    """The five units this check never read, plus one that is on no list at all.

    `agent-brand-new-unit` is the half the item calls "any program added later
    without a code edit": it appears in no pre-declared list anywhere in the
    script, and it is graded. The five named units are the ones whose absence the
    bug was about — they were on the box, in the supervisor config, and invisible.
    """
    mod, _ = _fleet(monkeypatch, FLEET_LINES)
    rows = {r["name"]: r for r in mod.derived_supervisor_rows()}

    for name in NEVER_READ:
        assert name in rows, f"{name} still has no row: {sorted(rows)}"
        assert rows[name]["healthy"] is True
    assert "agent-brand-new-unit" in rows, \
        "a program in no pre-declared list must still be graded"


def test_the_default_run_accounts_for_every_reported_program(monkeypatch):
    """One row per program, counting the rows the declared pass already owns.

    Coverage is a union: `agent-djev` is reported by its probe row, not by a
    second process-state row, and the three `lloyd-mc:` members are reported under
    their unqualified labels. What must not exist is a reported program that
    appears in neither — that silent skip is the whole of #1649.
    """
    mod, _ = _fleet(monkeypatch, FLEET_LINES)
    covered = {r["name"] for r in mod.derived_supervisor_rows()}
    covered |= {name for name in mod.SERVICES
                if name not in mod._switched_off()}
    covered |= {p for p in (mod._sup_program(s) for s in mod.SERVICES.values()) if p}

    missing = [p for p in mod.supervisor_fleet()[0] if p not in covered]
    assert missing == [], f"programs supervisor reports that no row graded: {missing}"


def test_a_program_the_fleet_read_cannot_parse_is_reported_not_skipped(monkeypatch):
    """The pass that exists to catch skipped programs must not skip itself.

    An unreachable daemon answers with a socket path on stdout and no program
    lines; a run that graded nothing then looks identical to a box with nothing
    undeclared, which is how a coverage check passes for the wrong reason.
    """
    mod, _ = _fleet(monkeypatch, ["unix:///run/supervisor/supervisor.sock"
                                 " refused connection"])
    rows = mod.derived_supervisor_rows()
    assert len(rows) == 1
    assert rows[0]["healthy"] is False
    assert "could not read" in rows[0]["status"]

    # And a fleet answer with one unparseable line in it is not a fleet answer.
    mod2, _ = _fleet(monkeypatch, FLEET_LINES[:2] + ["error: <class 'wrong'>"])
    assert mod2.derived_supervisor_rows()[0]["healthy"] is False


def test_a_fleet_that_reports_nothing_is_an_empty_answer_not_a_healthy_box(monkeypatch):
    mod, _ = _fleet(monkeypatch, [""])
    rows = mod.derived_supervisor_rows()
    assert rows and rows[0]["healthy"] is False, \
        "an empty answer must not read as twelve healthy programs"


def test_supervisor_exit_code_three_does_not_mask_the_fleet(monkeypatch):
    """`status` exits 3 whenever anything is not RUNNING, so rc is not the signal.

    On this box that is every single run, because `agent-llm-secondary` is stopped
    by config: a check that read `rc != 0` as "unreadable" would have graded zero
    undeclared programs on a healthy machine and only started working once
    something actually broke.
    """
    mod, _ = _fleet(monkeypatch, FLEET_LINES, returncode=3)
    assert len(mod.derived_supervisor_rows()) >= len(NEVER_READ)


# ---------------------------------------------------------------------- clause 2
# Every state in the table plus one that is not, through the derived row. The
# description fields are the ones supervisor actually prints for these states —
# BACKOFF carries an exit code, EXITED carries one too, so a reader tempted to go
# back to the exit code has them right there (#1040).
@pytest.mark.parametrize("state,expect_healthy", [
    ("RUNNING", True),
    ("STARTING", False),
    ("BACKOFF", False),
    ("STOPPING", False),
    ("STOPPED", False),
    ("EXITED", False),
    ("FATAL", False),
    ("UNKNOWN", False),
])
def test_an_undeclared_program_is_graded_from_the_state_field_alone(
        monkeypatch, state, expect_healthy):
    mod, _ = _fleet(monkeypatch, [f"agent-brand-new-unit    {state}   pid 1, uptime 0:00:01"],
                    switched_off=())
    row = mod.derived_supervisor_rows()[0]

    assert row["healthy"] is expect_healthy, \
        f"state {state}: {row['healthy']} / {row['status']}"
    assert state in row["status"], "the verdict must name the state it read"


def test_a_state_that_is_not_a_supervisord_state_is_a_broken_reading(monkeypatch):
    """`OK` is not a program state; calling it stopped would be an invention.

    A new supervisor release, a truncated line or a wrapper that prints its own
    banner all produce a field 2 that is not in the table. Reporting that as
    "STOPPED" would tell a person to start a service that may be running fine;
    reporting it healthy would be the bug that made this file (#1040). So the row
    says the reading itself is unusable — and is unhealthy, because an unusable
    check must never contribute to a green run.
    """
    mod, _ = _fleet(monkeypatch, ["agent-brand-new-unit    OK   pid 1"], switched_off=())
    row = mod.derived_supervisor_rows()[0]

    assert row["healthy"] is False
    assert "not a supervisord program state" in row["status"]
    # The word "stopped" appears in that sentence only to deny it; the test that
    # matters is that the check does not invent supervisor's own state name.
    assert "STOPPED" not in row["status"], \
        f"a broken reading must not be reported as the STOPPED state: {row['status']}"


def test_a_state_field_that_is_absent_is_a_broken_reading_too(monkeypatch):
    mod, _ = _fleet(monkeypatch, ["agent-brand-new-unit"], switched_off=())
    row = mod.derived_supervisor_rows()[0]
    assert row["healthy"] is False
    assert "no state field" in row["status"]


def test_a_state_in_the_wrong_position_is_not_read_from_field_one(monkeypatch):
    """Field 1 is the state. A line whose name is RUNNING is not a healthy program."""
    mod, _ = _fleet(monkeypatch, ["agent-brand-new-unit    FATAL   can't spawn process"],
                    switched_off=())
    assert mod.derived_supervisor_rows()[0]["healthy"] is False


# ---------------------------------------------------------------------- clause 3
def test_a_switched_off_program_is_expected_stopped_and_not_unhealthy(monkeypatch):
    """`agent-llm-secondary` today: stopped by config, graded by `_switched_off()`.

    #699 ruled a switched-off slot invisible rather than unhealthy, and the set is
    read from `app/llm_slots` so it moves when the config does. The row stays
    present and says `expected stopped` — a reader who wants to know why the GPU
    has one less server than they expected should be able to see it, on that slot
    and on any other a future config stops.
    """
    mod, _ = _fleet(monkeypatch, FLEET_LINES)
    rows = {r["name"]: r for r in mod.derived_supervisor_rows()}

    row = rows["agent-llm-secondary"]
    assert row["healthy"] is True, row["status"]
    assert row["status"].startswith("expected stopped")
    assert "config" in row["status"], "the row must say the stop was deliberate"


def test_a_switched_off_program_is_expected_stopped_in_any_non_running_state(monkeypatch):
    """BACKOFF on a switched-off slot is still config, not a fault.

    A slot that supervisor keeps retrying looks like a crash-loop from the socket,
    and the ruling is that a deliberate stop is not graded unhealthy. The name in
    `_switched_off()` is what decides, not which state string follows it.
    """
    for state in ("STOPPED", "BACKOFF", "EXITED", "STOPPING"):
        mod, _ = _fleet(monkeypatch, [f"agent-llm-secondary   {state}   Not started"])
        row = mod.derived_supervisor_rows()[0]
        assert row["healthy"] is True, f"{state}: {row['status']}"
        assert row["status"].startswith("expected stopped")


def test_the_expected_stopped_set_is_read_from_config_not_from_a_name_list(monkeypatch):
    """Clause 3's second half: no second hand-maintained list may appear.

    `EXPECTED_STOPPED`, `EXPECTED_DOWN`, `STOPPED_OK` — every one of those is a
    second copy of a fact `app/llm_slots` already owns, and #1649's triage names
    the hand list at `skills/system-health-check/system_health_check.py` as the
    rot this must not repeat: it still names four programs with no unit on this
    box. The test does not grep for a name though; it drives the set. Move the set
    and the grading has to move with it, which only a derived read can do.
    """
    mod, _ = _fleet(monkeypatch, [
        "agent-llm-secondary   STOPPED   Not started",
        "agent-some-new-slot     STOPPED   Not started",
    ], switched_off=("agent-some-new-slot",))       # config changed; the script didn't

    rows = {r["name"]: r for r in mod.derived_supervisor_rows()}
    assert rows["agent-some-new-slot"]["status"].startswith("expected stopped"), \
        "the new slot is the expected-stopped one now, and the script never changed"

    # And the name that used to be special is not special any more: it loses its
    # `expected stopped` row. It is a declared service, so the declared pass grades
    # it on its probe — coverage continues, it just stops being excused.
    assert "agent-llm-secondary" not in rows
    assert "agent-llm-secondary" in mod.SERVICES


def test_a_running_program_in_the_switched_off_set_still_gets_a_row(monkeypatch):
    """"Config says off, supervisor says up" is a fact, not a row to drop.

    #699 excuses a switched-off slot from being graded UNHEALTHY; it does not
    excuse it from being reported. The review rung refused the first cut of this fix
    for exactly that: a switched-off name whose state came back RUNNING produced no
    row at all, so the one situation where the config and the machine disagree — the
    situation worth a line — was the situation the report was silent about.
    """
    mod, _ = _fleet(monkeypatch, ["agent-llm-secondary   RUNNING   pid 7, uptime 0:01"])
    rows = mod.derived_supervisor_rows()

    assert [r["name"] for r in rows] == ["agent-llm-secondary"]
    assert rows[0]["healthy"] is True
    assert "expected stopped" not in rows[0]["status"]
    assert "RUNNING" in rows[0]["status"]


@pytest.mark.parametrize("off", [
    # The set `_switched_off()` can actually return: `app/llm_slots` derives it from
    # config.yaml's optional GPU-2 slots, and every name in it is optional.
    {"agent-llm-secondary", "agent-decider", "agent-voice-mcp",
     "agent-voice-mode", "agent-tts"},
    {"agent-tts"},
    set(),
])
def test_coverage_survives_every_config_the_switched_off_set_can_return(monkeypatch, off):
    """Clause 1 again, against the config this script has to survive.

    The default fixture's `off` set has one name, and one name is not enough to
    exercise a derived set: `agent-tts` is one of the five units #1649 names, and it
    is also an optional slot name, so the two facts overlap. If a switched-off name
    could cost a program its row, `agent-tts` would be missing on exactly the boxes
    where someone switched the speech synthesiser off — which is a regression from
    today's coverage, dressed up as a fix.
    """
    mod, _ = _fleet(monkeypatch, FLEET_LINES, switched_off=tuple(off))
    rows = {r["name"]: r for r in mod.derived_supervisor_rows()}

    for name in NEVER_READ:
        assert name in rows, f"off={sorted(off)} lost {name}: {sorted(rows)}"
        assert rows[name]["healthy"] is True
    assert "agent-brand-new-unit" in rows, f"off={sorted(off)} lost the undeclared unit"


@pytest.mark.parametrize("state", ["RUNNING", "STOPPED", "FATAL", "BACKOFF"])
def test_an_undeclared_switched_off_name_is_never_dropped_whatever_its_state(
        monkeypatch, state):
    """The same hole from the state side: four states, no missing row, ever.

    `agent-unloved-unit` is on no list anywhere in the script, so the only thing
    that can be said about it is config. Whatever that says, it stays in the report;
    only the verdict changes — `expected stopped` when it is down, its real state
    when it is up, and unhealthy for FATAL/BACKOFF, which no excuse covers.
    """
    mod, _ = _fleet(monkeypatch, [f"agent-unloved-unit   {state}   pid 1"],
                    switched_off=("agent-unloved-unit",))
    rows = mod.derived_supervisor_rows()

    assert [r["name"] for r in rows] == ["agent-unloved-unit"],         f"{state} with the name switched off produced no row"
    # #699 excuses every non-RUNNING state of a switched-off name, not just STOPPED:
    # a slot supervisor keeps retrying is indistinguishable from a crash-loop from
    # here, and the ruling is that a deliberate stop is not graded unhealthy. That is
    # pre-existing behaviour, unchanged by this round — asserted so a later "tighten
    # the excuse" change has to say so in a diff rather than discover it here.
    if state == "RUNNING":
        assert rows[0]["healthy"] is True and "RUNNING" in rows[0]["status"]
        assert "expected stopped" not in rows[0]["status"]
    else:
        assert rows[0]["healthy"] is True
        assert rows[0]["status"].startswith("expected stopped")


def test_declared_services_are_not_reported_twice(monkeypatch):
    """Coverage widens without duplicating: a probe row beats a process-state row.

    The three Mission Control members are the sharp case — their `SERVICES` keys
    are unqualified labels while supervisor prints `lloyd-mc:lloyd-backend`, so a
    derived set built from keys alone would report that program twice with two
    different verdicts, and `Overall:` would count it twice.
    """
    mod, _ = _fleet(monkeypatch, FLEET_LINES)
    derived = [r["name"] for r in mod.derived_supervisor_rows()]

    for name in ("agent-djev", "agent-qmd-daemon", "lloyd-mc:lloyd-backend",
                 "lloyd-mc:lloyd-frontend", "lloyd-mc:lloyd-mcp"):
        assert name not in derived, f"{name} would be reported twice"
    assert len(derived) == len(set(derived))


# ---------------------------------------------------------------------- clause 4
# #1141 is the incident this clause exists for: the health check used to prove the
# vault sync worked by making it do something, and on 09-14 that probe deleted the
# live sync registration. A check that can cost the thing it measures is worse than
# no check, so `agent-obsidian-sync` is graded on the state supervisor already
# tracks and on nothing else.

def _explode(*a, **k):
    raise AssertionError("a derived program must not reach this call site")


def _no_probing(monkeypatch, mod):
    """Make every way this script can reach a port blow up if used."""
    monkeypatch.setattr(mod, "http_check", _explode)
    monkeypatch.setattr(mod.socket, "socket", _explode)


def test_grading_the_vault_sync_asks_supervisor_for_state_and_nothing_else(monkeypatch):
    """Every command issued while grading the sync is a `status` call.

    The assertion is on the command list, not on the row's wording: a future edit
    that adds a `curl` of the sync's endpoint, a read of its lock file or a run of
    `obsidian-sync --probe` to make the row more convincing fails here even though
    the row itself would look healthier. The socket guard is the same claim from the
    other side — the derived pass never opens one.
    """
    mod, answer = _fleet(monkeypatch, FLEET_LINES)
    _no_probing(monkeypatch, mod)

    rows = {r["name"]: r for r in mod.derived_supervisor_rows()}
    assert rows["agent-obsidian-sync"]["healthy"] is True

    assert answer.commands, "the check must actually ask supervisor something"
    for cmd in answer.commands:
        assert cmd[:3] == ["supervisorctl", "-c", str(mod.SUPervisor_CONF)], cmd
        assert cmd[3] == "status", cmd
        assert len(cmd) <= 5, cmd


def test_a_sync_line_without_a_state_field_re_asks_by_name_only(monkeypatch):
    """The one escalation the sync is allowed: the same `status`, with its name.

    Still a state read, still no side effect. `supervisorctl status <program>`
    returns supervisor's own line for that program; it does not start, restart,
    signal or interrogate the process.
    """
    lines = [ln for ln in FLEET_LINES if "obsidian" not in ln]
    lines.append("agent-obsidian-sync")          # name only, no state field
    mod, answer = _fleet(
        monkeypatch, lines,
        per_name={"agent-obsidian-sync": "agent-obsidian-sync   RUNNING   "
                                          "pid 2491, uptime 3 days, 23:03:52"})
    _no_probing(monkeypatch, mod)

    rows = {r["name"]: r for r in mod.derived_supervisor_rows()}
    assert rows["agent-obsidian-sync"]["healthy"] is True, rows["agent-obsidian-sync"]
    asked = [c for c in answer.commands if c[3] == "status" and len(c) > 4]
    assert asked == [["supervisorctl", "-c", str(mod.SUPervisor_CONF), "status",
                      "agent-obsidian-sync"]], asked


def test_the_two_declared_port_probes_are_unchanged(monkeypatch):
    """Coverage widened; the probes did not move.

    qmd on 8181 is the declared entry that answers on a port and has no
    supervisorctl command — which is exactly why the old script printed six
    supervisor rows for seven declared services, and why #406 had to fix its port
    number. djev names a port too (8011) and is checked by its own supervisorctl
    call. Neither may be swallowed into a process-state-only row now that the
    derived pass exists: `agent-obsidian-sync` gets state-only because it has no
    declared probe, not because derived rows are state-only by policy.
    """
    mod, answer = _fleet(monkeypatch, FLEET_LINES)
    assert mod.SERVICES["agent-djev"]["port"] == 8011
    assert mod.SERVICES["agent-qmd-daemon"]["port"] == 8181

    derived = {r["name"] for r in mod.derived_supervisor_rows()}
    assert not ({"agent-djev", "agent-qmd-daemon"} & derived), \
        "a declared service must not also be graded as undeclared"
    # The derived pass is one fleet-wide `status`, whatever else happens later.
    assert [c[3:] for c in answer.commands] == [["status"]], answer.commands

    touched = []
    monkeypatch.setattr(mod, "http_check",
                        lambda port, host="localhost": (touched.append(port),
                                                        (True, "ok"))[1])
    r = mod.check_service("agent-qmd-daemon", mod.SERVICES["agent-qmd-daemon"])
    assert r["healthy"] is True, r
    assert touched == [8181], "qmd must still be probed on 8181, and only there"


def test_an_undeclared_program_row_carries_no_probe_of_any_kind(monkeypatch):
    """The derived row has no port, no url and no command — the shape enforces it.

    A row that gained a port key would be a round trip by the time `format_json`
    printed it, so the absence is asserted structurally rather than by reading the
    status text.
    """
    mod, _ = _fleet(monkeypatch, FLEET_LINES)
    for row in mod.derived_supervisor_rows():
        for key in ("port", "url", "command", "log_path"):
            assert key not in row, f"{row['name']} gained a {key}"


# --------------------------------------------------------------------- end to end
# The clause says "the default run yields", so the boundary that matters is a real
# `python3 scripts/service_health_check.py` across a real `subprocess.run` to a real
# `supervisorctl`. UNIT= names a stand-in script; the guard in the script only
# re-interprets argv under pytest so it can never do that on a live run.

_FLEET_SCRIPT = """#!/usr/bin/env python3
import os, sys
# The script under test passes `-c <conf>`; under pytest it is told to pass THIS
# file's path there (the UNIT seam, which only works under pytest), and that is how
# this stub knows it is being used as the conf path. Shift argv so the real script
# gets `[0] + argv[4:]`, which is what a real supervisorctl invocation looks like.
argv = sys.argv
if argv[3:4] == [os.path.abspath(__file__)]:
    argv = [argv[0]] + argv[4:]
LINES = {lines!r}
PER_NAME = {per_name!r}
name = argv[4] if len(argv) > 4 else None
out = PER_NAME.get(name) or ([ln for ln in LINES.split("\\n") if ln.split()[:1] == [name]] or [None])[0] \\
    if name else LINES
print(out or "")
# supervisor exits 3 when anything reported is not RUNNING, which is the steady
# state here because agent-llm-secondary is stopped by config.
sys.exit(3)
"""


def _stub(tmp_path, lines, per_name=None):
    """Write the stand-in supervisorctl and return (env, path).

    Two roles for one file, because the script reaches supervisor in two ways: by
    name on PATH, and by the conf path in `-c`. The UNIT seam points the second at
    this file, and the stub recognises its own path as the sentinel for "you are
    being used as the conf, shift argv".
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "supervisorctl"
    stub.write_text(_FLEET_SCRIPT.format(lines="\n".join(lines),
                                         per_name=per_name or {}))
    stub.chmod(0o755)
    env = dict(PATH=f"{bin_dir}:/usr/bin:/bin", UNIT=str(stub.resolve()),
               PYTEST_CURRENT_TEST="seam")
    return env, stub


def _cli(tmp_path, extra_args=(), lines=None, per_name=None):
    env, _ = _stub(tmp_path, lines or FLEET_LINES, per_name)
    r = subprocess.run([sys.executable, str(SCRIPT), "--format", "json", *extra_args],
                       capture_output=True, text=True, env=env, timeout=180)
    assert r.returncode in (0, 1), f"the run did not produce a verdict: {r.stderr[:300]}"
    return r, json.loads(r.stdout)


def test_the_default_cli_run_names_every_program(tmp_path):
    """The acceptance check itself: 3 of 12 programs before, all 12 after.

    Read as JSON rather than grepping text, because a program name can appear in
    another row's description and still not have been graded. `services` is the
    list of rows, and each row's `name` is the thing the check actually asked
    supervisor about.
    """
    r, doc = _cli(tmp_path)
    names = {s["name"] for s in doc["services"]}

    for program in NEVER_READ:
        assert program in names, f"{program} is still invisible to the default run"

    # The three Mission Control members are reported under their declared labels,
    # which are unqualified: `lloyd-backend`, not `lloyd-mc:lloyd-backend`.
    for label in ("lloyd-backend", "lloyd-frontend", "lloyd-mcp"):
        assert label in names, f"{label} (group-prefixed {label}) is unreported"

    # One program supervisor reported and nothing graded is the whole bug. A
    # group member counts as graded under either spelling of its name.
    ungraded = [p for p in {ln.split()[0] for ln in FLEET_LINES}
                if p not in names and p.split(":")[-1] not in names]
    assert ungraded == [], f"reported but graded by no row: {ungraded}"

    # And the switched-off slot is in the report, not excused from it.
    sec = [s for s in doc["services"] if s["name"] == "agent-llm-secondary"]
    assert sec and sec[0]["status"].startswith("expected stopped"), doc["services"]


def test_the_cli_run_is_a_health_report_that_can_fail(tmp_path):
    """`Overall:` counts what it graded, and the exit code is the verdict.

    Before the fix this printed `Overall: 8/8 services healthy` on a box with five
    ungraded programs. Now the total is the seven declared probes, the fleet rows
    and the deployed-copy pairs, and every one of them is in the count.
    """
    r, doc = _cli(tmp_path)
    assert r.returncode == 0, doc["services"]
    assert doc["total_services"] == len(doc["services"])
    assert doc["total_services"] >= 12, \
        f"only {doc['total_services']} rows; the fleet is 12 programs"
    assert doc["unhealthy"] == 0, [s for s in doc["services"] if not s["healthy"]]


def test_an_undeclared_program_in_fatal_reaches_the_verdict(tmp_path):
    """The point of the whole item: a FATAL unit nobody declared changes the answer.

    Before the fix such a unit was worth nothing to the verdict — `Overall: 8/8
    services healthy`, `Supervisor: healthy`, and anything reading this script saw
    an all-clear box. Now the row exists, it is unhealthy, `Overall:` counts it, and
    the category that holds it reads degraded.

    Not the exit code: `main` has never exited nonzero for an unhealthy service, so
    every caller today gets 0 whatever the fleet does. Changing that is a different
    decision with different blast radius (it is what the fleet watchdog and every
    skill run of this script would start seeing) and no clause of #1649 asks for it.
    """
    r, _ = _cli(tmp_path, lines=FLEET_LINES + [
        "agent-unloved-unit         FATAL   can't spawn process"])
    doc = json.loads(r.stdout)
    row = [s for s in doc["services"] if s["name"] == "agent-unloved-unit"]
    assert row and row[0]["healthy"] is False
    assert "FATAL" in row[0]["status"]
    # The two numbers a reader actually looks at: the count, and the category.
    assert doc["unhealthy"] >= 1, "the count must include the unit nobody declared"
    assert doc["healthy"] < doc["total_services"], \
        f"Overall: still reads as an all-clear: {doc['healthy']}/{doc['total_services']}"
    assert doc["summary"]["fleet"] == "degraded", doc["summary"]
    # ...and the existing headings are untouched by a unit they do not contain.
    assert doc["summary"]["supervisor"] == "healthy", doc["summary"]


def test_a_targeted_ask_gets_no_unsolicited_fleet_rows(tmp_path):
    """`--services lloyd-mcp` is a question about one service, not a fleet report.

    Without this guard the derived rows would be appended to a one-name ask, and
    the `--services` path — the one an operator reaches for when they already know
    what they are asking about — would print twelve rows they did not request.
    """
    _, doc = _cli(tmp_path, extra_args=("--services", "lloyd-mcp"))
    assert [s["name"] for s in doc["services"]] == ["lloyd-mcp"], doc["services"]


# ---------------------------------------------------------------- clause 5 (docs)
# Unmarked, deliberately: the gate runner hardcodes `-m "not live_vault"`, so a
# marked check is a check the gate never runs — and this one exists to stop the
# skill's prose drifting back to "seven services" after the script widened. Same
# reasoning as tests/test_brief_triage_clock_skill.py and
# tests/test_service_health_check_qmd.py, which pin this very file.

def test_skill_says_coverage_is_derived_and_names_what_it_excludes():
    """Clause 5: the doc tells the next reader how coverage is decided.

    The old text recorded the gap as a standing question with no owner. A reader
    who trusts the doc instead of the script would still conclude that a unit
    outside the seven is out of scope, which is how the gap survived #1233.
    """
    assert SKILL.is_file(), f"skill file missing: {SKILL}"
    text = SKILL.read_text()

    assert "supervisorctl status" in text
    for phrase in ("loaded", "derive"):
        assert phrase in text, f"the doc never says coverage is {phrase}"
    # The exclusion the derivation implies, stated rather than implied.
    assert "agent-decider" in text, "the unloaded-candidate case is undocumented"
    assert "autostart" in text or "not loaded" in text.lower()


def test_skill_no_longer_calls_the_coverage_gap_an_unowned_question():
    """No "no owner", no "no open item": the gap has both now.

    These strings were the doc's way of saying "nobody is going to fix this", and
    they were the reason the next pass did not fix it.
    """
    text = SKILL.read_text().lower()
    for dead in ("no owner", "no open item", "carries the question",
                 "open question"):
        assert dead not in text, f"the doc still says the gap has {dead!r}"


def test_skill_still_forbids_a_round_trip_probe_of_the_vault_sync():
    """The other doc half: the #1141 constraint has to outlive the rewrite.

    A doc that explains the new coverage and drops the incident is how the probe
    comes back — the check looks more convincing with a real round trip in it,
    which is exactly what made someone add it the first time.
    """
    text = SKILL.read_text().lower()
    assert "obsidian-sync" in text or "vault sync" in text
    assert "1141" in text, "the incident that forbids the probe must stay cited"
    assert "never" in text and "probe" in text


def test_the_default_run_re_asks_by_name_for_a_sync_line_with_no_state(tmp_path):
    """Clause 4's per-program path, through the default run rather than a helper.

    A fleet line carrying only a name is the sole reason the derived pass ever makes
    a second supervisor call, and the grader asked for it exercised at the CLI too:
    the whole point is that this one escalation is the sync's entire allowance. The
    named call comes back `... RUNNING ...`, so the row is healthy and the commands
    the run issued are one fleet `status` plus one `status agent-obsidian-sync` — no
    port, no script, no sentinel (#1141).
    """
    lines = [ln for ln in FLEET_LINES if "obsidian" not in ln] + ["agent-obsidian-sync"]
    r, doc = _cli(tmp_path, lines=lines, per_name={
        "agent-obsidian-sync": "agent-obsidian-sync   RUNNING   "
                               "pid 2491, uptime 3 days, 23:03:52"})
    row = [s for s in doc["services"] if s["name"] == "agent-obsidian-sync"]
    assert row and row[0]["healthy"] is True, doc["services"]
    assert "RUNNING" in row[0]["status"]
    assert r.returncode in (0, 1)
