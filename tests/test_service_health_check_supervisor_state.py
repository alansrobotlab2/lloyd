"""#1040: a crash-looping (BACKOFF) supervisor service must not read healthy.

`supervisorctl status` prints `name  STATENAME  description` per process
(supervisor 4.3.0, `supervisorctl.py:643-655`), and it *exits 0 for BACKOFF*:
`do_status` moves the exit status off SUCCESS only for
`states.STOPPED_STATES` (`supervisorctl.py:696-698`), and `states.py:14-22`
puts BACKOFF in RUNNING_STATES, not STOPPED_STATES. The script decided health
from substring tests over the whole answer plus a
`healthy = result.returncode == 0` fallback, so feeding the real
`check_service` the measured BACKOFF line with `returncode=0` returned
`healthy=True` — a process that is spawned, dies inside `startsecs` and is
being retried printed `[✓] healthy`. The same branch read
`agent-tts  FATAL  can't spawn process: RUNNING helper not found` as
`healthy=True, status="RUNNING"` (the substring hit inside the *description*),
and empty output as `healthy=True, status="unknown"`.

The state is field 2 of each status line, so field 2 is the only field read.
Each test names the acceptance clause it pins.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"

# Measured output, quoted verbatim from the item's /tmp supervisord run and
# from triage's replay through the real check_service().
BACKOFF_LINE = (
    "flaky                            BACKOFF   "
    "Exited too quickly (process log may have details)"
)
FATAL_WITH_RUNNING_IN_DESCRIPTION = (
    "agent-tts      FATAL   can't spawn process: RUNNING helper not found"
)
RUNNING_LINE = "lloyd-mc:lloyd-backend      RUNNING   pid 3781637, uptime 1:00:00"

# supervisor's own state names; every one of these except RUNNING is unhealthy.
NOT_RUNNING_STATES = ("STOPPED", "STARTING", "STOPPING", "BACKOFF", "EXITED", "FATAL", "UNKNOWN")

# A SERVICES-shaped entry with no `port`, so the supervisor branch alone
# decides the verdict — the same five of six live entries that are.
SERVICE_DEF = {
    "command": ["supervisorctl", "-c", "/nonexistent/supervisord.conf", "status", "flaky"],
    "category": "lloyd",
}


def _load_module():
    """Load the CLI script as a module.

    It has no package and nothing in the tree imports it — it is invoked as a
    subprocess from Bash — so loading it by path is how a unit test reaches
    `check_service` without inventing an import surface.
    """
    spec = importlib.util.spec_from_file_location("shc_supervisor_state", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shc = _load_module()


class _StubSubprocess:
    """Replaces `shc.subprocess`, so supervisor's answer and exit code are fixed.

    Only the genuinely external answer is stubbed: which state the daemon
    reports with which LSB exit code. The daemon itself cannot be crash-looped
    from a test, and the real-`subprocess` half of that boundary is covered by
    the CLI tests at the bottom of this file.
    """

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list] = []

    def run(self, command, **_kwargs):
        self.calls.append(list(command))
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, self.stderr)


def _verdict(monkeypatch, stdout: str, returncode: int = 0) -> dict:
    """Run the real `check_service` against a fixed supervisorctl answer."""
    monkeypatch.setattr(shc, "subprocess", _StubSubprocess(stdout, returncode))
    return shc.check_service("flaky", SERVICE_DEF)


# ---------------------------------------------------------------- clause 1
def test_backoff_line_with_a_zero_exit_code_is_not_healthy(monkeypatch):
    """Clause 1: the measured BACKOFF line, with the exit code supervisor really
    gives it (0), is unhealthy and the state survives into `status`."""
    result = _verdict(monkeypatch, BACKOFF_LINE, returncode=0)
    assert result["healthy"] is False, (
        f"a crash-looping process read as healthy: {result}"
    )
    assert "BACKOFF" in result["status"], result


# ---------------------------------------------------------------- clause 2
@pytest.mark.parametrize("returncode", [0, 3])
@pytest.mark.parametrize("state", NOT_RUNNING_STATES)
def test_every_non_running_state_is_unhealthy_whatever_the_exit_code(
    monkeypatch, state, returncode
):
    """Clause 2: the state decides, not the exit code. Each token as field 2 of
    a status line, presented with both the exit code supervisor gives BACKOFF
    (0) and the one it gives STOPPED/EXITED/FATAL/UNKNOWN (3)."""
    line = f"agent-x  {state}   Exited too quickly (process log may have details)"
    result = _verdict(monkeypatch, line, returncode=returncode)
    assert result["healthy"] is False, (
        f"state {state} with returncode={returncode} read as healthy: {result}"
    )
    assert state in result["status"], result


def test_running_state_is_healthy(monkeypatch):
    """Clause 2's other half, so the test above cannot be satisfied by an entry
    that is simply always unhealthy. `status` keeps the shape the script has
    always printed for a running service."""
    result = _verdict(monkeypatch, RUNNING_LINE, returncode=0)
    assert result["healthy"] is True, result
    assert result["status"] == "RUNNING", result


# ---------------------------------------------------------------- clause 3
def test_fatal_whose_description_mentions_running_is_unhealthy(monkeypatch):
    """Clause 3: the state is read from field 2, never matched as a substring
    anywhere in the line. This line was `healthy=True, status="RUNNING"`."""
    result = _verdict(monkeypatch, FATAL_WITH_RUNNING_IN_DESCRIPTION, returncode=0)
    assert result["healthy"] is False, (
        f"a FATAL process whose description contains the word RUNNING read as "
        f"healthy: {result}"
    )
    assert "FATAL" in result["status"], result


# ---------------------------------------------------------------- clause 4
def test_empty_output_is_unhealthy_and_names_the_missing_answer(monkeypatch):
    """Clause 4: an empty answer is not a healthy verdict. It was
    `healthy=True, status="unknown"`."""
    result = _verdict(monkeypatch, "", returncode=0)
    assert result["healthy"] is False, result
    assert result["status"].strip(), f"empty output left an empty status: {result}"
    assert "empty" in result["status"].lower(), result
    assert "unknown" in result["status"].lower(), result


def test_two_line_answer_with_one_backoff_line_is_unhealthy(monkeypatch):
    """Clause 4: a group namespec prints one line per process, so every line
    has to be RUNNING — the first line being RUNNING cannot carry the verdict."""
    answer = f"{RUNNING_LINE}\n{BACKOFF_LINE}"
    result = _verdict(monkeypatch, answer, returncode=0)
    assert result["healthy"] is False, (
        f"one BACKOFF process inside an otherwise RUNNING answer read as healthy: "
        f"{result}"
    )
    assert "BACKOFF" in result["status"], result


def test_two_line_answer_with_every_line_running_is_healthy(monkeypatch):
    """The all-lines-RUNNING positive, so the clause-4 test above cannot be
    satisfied by a parser that just always says unhealthy."""
    answer = f"{RUNNING_LINE}\nlloyd-mc:lloyd-mcp RUNNING pid 3781637, uptime 1:00:00"
    result = _verdict(monkeypatch, answer, returncode=0)
    assert result["healthy"] is True, result
    assert result["status"] == "RUNNING", result


@pytest.mark.parametrize(
    "answer",
    [
        "FAILED",
        f"{RUNNING_LINE}\nnot-a-status-line",
    ],
)
def test_a_line_without_a_state_field_is_unhealthy(monkeypatch, answer):
    """The acceptance check's "any unparsable line" half: fewer than two
    whitespace-separated fields means there is no state to read, so there is no
    healthy verdict — even when the first line said RUNNING."""
    result = _verdict(monkeypatch, answer, returncode=0)
    assert result["healthy"] is False, (
        f"answer with no readable state field read as healthy: {result}"
    )
    assert answer in result["status"], result


# ------------------------------------------- the real process boundary (Bash →
# ------------------------------------------- python → subprocess.run → child)
FAKE_SUPERVISORCTL = (
    "#!{python}\n"
    "import json, os, sys\n"
    "answer = json.loads(os.environ['FAKE_SUPERVISORCTL'])\n"
    "sys.stdout.write(answer['stdout'])\n"
    "sys.stderr.write(answer.get('stderr', ''))\n"
    "sys.exit(answer['returncode'])\n"
)


def _run_cli_with_fake_supervisorctl(tmp_path, stdout: str, returncode: int) -> dict:
    """Run the real script as a real subprocess against a stand-in
    `supervisorctl` that prints `stdout` and exits `returncode`.

    This is the boundary the unit tests above stub: the script's own
    `subprocess.run` call, a real child process, a real exit code, real stdout.
    `--category lloyd` selects the three port-less supervisor entries, so the
    verdict under test is the supervisor branch's alone.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "supervisorctl"
    fake.write_text(FAKE_SUPERVISORCTL.format(python=sys.executable))
    fake.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_SUPERVISORCTL"] = json.dumps(
        {"stdout": stdout, "returncode": returncode}
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--category", "lloyd", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, (
        f"the script's own exit status changed (it must not — that is #1029's "
        f"contract): rc={proc.returncode} stderr={proc.stderr[-500:]}"
    )
    return json.loads(proc.stdout)


def test_cli_reports_a_backoff_service_unhealthy_through_a_real_subprocess(tmp_path):
    """Same measured BACKOFF line, but delivered by a real child process with a
    real exit code 0 rather than by a stub: this is what a run of the
    service-health-check skill actually consumes."""
    payload = _run_cli_with_fake_supervisorctl(tmp_path, BACKOFF_LINE, 0)
    entries = payload["services"]
    assert len(entries) == 3, entries
    for entry in entries:
        assert entry["healthy"] is False, (
            f"real supervisorctl output BACKOFF with exit code 0 produced a "
            f"healthy entry: {entry}"
        )
        assert "BACKOFF" in entry["status"], entry
        assert entry["exit_code"] == 0, entry
    assert payload["healthy"] == 0, payload


def test_cli_reports_a_running_service_healthy_through_a_real_subprocess(tmp_path):
    """The positive control across the same boundary: the fake is not simply
    always unhealthy, and the RUNNING path still reports exit_code 0."""
    payload = _run_cli_with_fake_supervisorctl(tmp_path, RUNNING_LINE, 0)
    entries = payload["services"]
    assert len(entries) == 3, entries
    for entry in entries:
        assert entry["healthy"] is True, entry
        assert entry["status"] == "RUNNING", entry
        assert entry["exit_code"] == 0, entry
    assert payload["healthy"] == 3, payload


# =====================================================================================
# #1818: a hand-gated candidate unit is advisory, not a fleet fault.
#
# Measured on this round's base commit, one run of the script printed
# `[✗] agent-decider— STOPPED (process state only — no probe declared)` and
# `Categories: Fleet: degraded`, and agent-decider was the only unhealthy row in that
# category — so the word `degraded` was entirely this one unit. Its conf.d unit declares
# `autostart=false` at conf.d/agent-decider.conf:32, and the file's own header calls it
# "A CANDIDATE, NOT A SERVICE YET", hand-started against djev because GPU 2 holds one
# tenant. `_state_verdict` excused only `_switched_off()` names, so the declaration was
# never read.
#
# The excuse is read from the unit file, so every test below writes a throwaway
# supervisord.conf with its own `conf.d/` beside it and points the script at it. The
# program names are invented (`agent-candidate-unit`, `agent-watched-unit`) and appear
# nowhere in the script: a test whose candidate were a real program name would pass for
# a name list. The #1726 advisory machinery (the `[!]` icon, the `Advisory:` line, the
# json `advisory` count, the category arithmetic that skips advisory rows) already
# existed; what these tests pin is that the flag now gets SET, and only for a
# declared-off unit that is at rest.
# =====================================================================================

CANDIDATE = "agent-candidate-unit"        # in no list anywhere in the script
WATCHED = "agent-watched-unit"            # a derived row that is up, and graded
RUNNING_WATCHED = f"{WATCHED:<28}RUNNING   pid 4242, uptime 1:00:00"

UNIT_OFF = """; a candidate, not a service yet.
[program:{name}]
command=/bin/true
autostart=false
"""


def _conf_dir(tmp_path, units: dict) -> Path:
    """A throwaway supervisord.conf with `units` written into a `conf.d/` beside it.

    The layout is the one the script reads: `conf.d` is derived from the supervisord
    path, so pointing `UNIT` here is what redirects the unit read as well. No file here
    is the box's own config, and the box's real units are never opened by these tests.
    """
    conf = tmp_path / "supervisord.conf"
    conf.write_text("[supervisord]\nnodaemon=false\n")
    (tmp_path / "conf.d").mkdir(exist_ok=True)
    for fname, text in units.items():
        (tmp_path / "conf.d" / fname).write_text(text)
    return conf


class _Fleetctl:
    """Replaces the module's `subprocess`, answering `status` the way the daemon does.

    `status` prints every loaded program; `status <program>` prints that program's line
    alone. Replacing the attribute rather than `subprocess.run` keeps the patch inside
    the script under test, and a derived row opens no port, so nothing here reaches a
    socket (#1141).
    """

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, lines, per_name=None):
        self.stdout = "\n".join(lines)
        self.per_name = dict(per_name or {})
        self.commands: list[list] = []

    def run(self, command, **_kwargs):
        self.commands.append(list(command))
        out = self.stdout
        if len(command) > 4:                       # `status <program>`
            name = command[-1]
            out = self.per_name.get(name) or next(
                (ln for ln in self.stdout.splitlines() if ln.split()[:1] == [name]),
                f"{name}    STOPPED   not started")
        return subprocess.CompletedProcess(command, 3, out, "")


def _derived(tmp_path, monkeypatch, states: dict, units: dict, switched_off=()):
    """`derived_supervisor_rows()` over `states` with `units` as the conf.d.

    `SUPervisor_CONF` is the very constant the `UNIT` seam assigns (the script does
    `SUPervisor_CONF = os.environ["UNIT"]` under pytest), so this drives the same
    discovery the seam does; the end-to-end tests below go through the seam itself.
    """
    conf = _conf_dir(tmp_path, units)
    mod = _load_module()
    monkeypatch.setattr(mod, "SUPervisor_CONF", str(conf))
    monkeypatch.setattr(mod, "_switched_off", lambda: set(switched_off))
    ctl = _Fleetctl([f"{name:<28}{state}   pid 1, uptime 0:00:01"
                     for name, state in states.items()])
    monkeypatch.setattr(mod, "subprocess", ctl)
    return mod, {r["name"]: r for r in mod.derived_supervisor_rows()}


# ---------------------------------------------------------------- clause 1
def test_a_declared_off_derived_row_is_advisory_and_names_the_declaration(tmp_path, monkeypatch):
    """Clause 1: STOPPED + `autostart=false` + no SERVICES entry ⇒ advisory / `warn`,
    and the status quotes the declaration rather than guessing at it."""
    _, rows = _derived(tmp_path, monkeypatch, {CANDIDATE: "STOPPED"},
                      {f"{CANDIDATE}.conf": UNIT_OFF.format(name=CANDIDATE)})
    row = rows[CANDIDATE]

    assert row.get("advisory") is True, f"a hand-stopped candidate was graded a fault: {row}"
    assert row["verdict"] == "warn", row
    assert "autostart=false" in row["status"], \
        f"the row must name the declaration it read: {row['status']}"
    assert "STOPPED" in row["status"], row


# ---------------------------------------------------------------- clause 3
@pytest.mark.parametrize("state", ["STARTING", "EXITED", "FATAL", "BACKOFF"])
def test_autostart_false_excuses_only_a_stopped_unit(tmp_path, monkeypatch, state):
    """Clause 3: a candidate is expected to be DOWN, not to be failing.

    STARTING is supervisord bringing it up; BACKOFF, EXITED and FATAL mean a start was
    attempted and did not hold. The declaration says nothing about those, so the row
    stays a graded fault exactly as it was before #1818.
    """
    _, rows = _derived(tmp_path, monkeypatch, {CANDIDATE: state},
                      {f"{CANDIDATE}.conf": UNIT_OFF.format(name=CANDIDATE)})
    row = rows[CANDIDATE]

    assert row["healthy"] is False, f"{state} on a hand-started unit read healthy: {row}"
    assert not row.get("advisory"), f"{state} was excused by the unit file: {row}"
    assert state in row["status"], row


def test_the_advisory_row_is_still_reported_and_its_category_reads_healthy(tmp_path, monkeypatch):
    """Clause 1's "reported" half plus clause 2's category half, module-level.

    `Fleet: healthy` is the whole point of the change, and it only means something
    while the rows that ARE expected to be up are still graded: `agent-watched-unit`
    shares the category and is counted.
    """
    conf = _conf_dir(tmp_path, {f"{CANDIDATE}.conf": UNIT_OFF.format(name=CANDIDATE)})
    mod = _load_module()
    monkeypatch.setattr(mod, "SUPervisor_CONF", str(conf))
    monkeypatch.setattr(mod, "_switched_off", lambda: set())
    monkeypatch.setattr(mod, "subprocess",
                        _Fleetctl([f"{CANDIDATE}   STOPPED   Not started",
                                   RUNNING_WATCHED]))

    rows = {r["name"]: r for r in mod.derived_supervisor_rows()}
    assert rows[CANDIDATE].get("advisory") is True, rows[CANDIDATE]
    assert rows[WATCHED]["healthy"] is True, rows[WATCHED]

    # The graded input to the category arithmetic, which is the only thing this
    # function contributes to it: one row out of two. The word those rows produce is
    # `main`'s, and the two CLI tests below pin it across that boundary.
    assert [r["name"] for r in mod._scored_rows(list(rows.values()))] == [WATCHED], rows


# ---------------------------------------------------------------- clause 2
_STUB_SUPERVISORCTL = (
    "#!{python}\n"
    "import json, os, sys\n"
    "answer = json.loads(os.environ['STUB_FLEET'])\n"
    "name = sys.argv[4] if len(sys.argv) > 4 else None\n"
    "if name:\n"
    "    print(answer['per_name'].get(name)\n"
    "          or next((ln for ln in answer['fleet'].splitlines()\n"
    "                  if ln.split()[:1] == [name]),\n"
    "                  f'{{name}}    STOPPED   not started'))\n"
    "else:\n"
    "    print(answer['fleet'])\n"
    "sys.exit(3)\n"
)


def _default_run(tmp_path, fmt: str, units: dict, lines=None):
    """One real `python3 scripts/service_health_check.py`, across the `UNIT` seam.

    `UNIT` names the throwaway supervisord.conf and a stand-in `supervisorctl` answers
    on PATH, so the fleet is whatever `lines` says while the units under `conf.d/` are
    the ones written here. This is the boundary a unit test cannot reach: the seam
    guard at the top of the script, the argv it builds, and the formatters.
    """
    conf = _conf_dir(tmp_path, units)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "supervisorctl"
    stub.write_text(_STUB_SUPERVISORCTL.format(python=sys.executable))
    stub.chmod(0o755)
    env = dict(os.environ)
    env.update(PATH=f"{bin_dir}{os.pathsep}{env['PATH']}", UNIT=str(conf.resolve()),
               PYTEST_CURRENT_TEST="seam",
               STUB_FLEET=json.dumps({"fleet": "\n".join(
                   lines or [f"{CANDIDATE}   STOPPED   Not started",
                             RUNNING_WATCHED]), "per_name": {}}))
    proc = subprocess.run([sys.executable, str(SCRIPT), *(["--format", "json"]
                                                          if fmt == "json" else [])],
                          capture_output=True, text=True, timeout=180, env=env)
    assert proc.returncode in (0, 1), \
        f"the run produced no verdict: rc={proc.returncode} {proc.stderr[-400:]}"
    return proc


def test_the_default_run_reports_a_hand_stopped_candidate_without_degrading_fleet(tmp_path):
    """Clause 2 end to end: the only down unit in the fleet is declared-off, so
    `Fleet:` is healthy and the json carries the row under `advisory`, outside the
    graded total.

    Before this change the candidate's row was a graded fault and its category read
    `degraded` — on the live box, where it was the only unhealthy row in `Fleet:`
    (measured 2026-09-29: `[✗] agent-decider— STOPPED (process state only — no probe
    declared)` with `Categories: Fleet: degraded`). `agent-watched-unit` is RUNNING
    and is graded, which is what stops the category word from being an absence of
    evidence.
    """
    doc = json.loads(_default_run(tmp_path, "json",
                                 {f"{CANDIDATE}.conf": UNIT_OFF.format(name=CANDIDATE)}
                                 ).stdout)
    rows = {s["name"]: s for s in doc["services"]}
    cand = rows[CANDIDATE]

    assert cand.get("advisory") is True, cand
    assert cand["verdict"] == "warn", cand
    assert "autostart=false" in cand["status"], cand
    assert doc["summary"]["fleet"] == "healthy", doc["summary"]

    # The candidate is the only row in its category that is not graded, and the one
    # that is (agent-watched-unit, RUNNING) carries the category word. These two
    # assertions are what a missing exemption would break: without the fix the row is
    # graded, the list gains CANDIDATE, and `fleet` reads `degraded`.
    fleet = [s for s in doc["services"] if s["category"] == "fleet"]
    assert [s["name"] for s in fleet if not s.get("advisory")] == [WATCHED], fleet
    assert CANDIDATE not in [s["name"] for s in doc["services"]
                             if not s["healthy"] and not s.get("advisory")], \
        "the hand-stopped candidate was counted as a fault"

    # The clause's own arithmetic claim: the advisory count sits outside the graded
    # total, and the three numbers partition the rows. `ca-trust` rides every full
    # run (#1726) and is advisory here too, so the partition is asserted rather than
    # a fixed count.
    assert doc["healthy"] + doc["unhealthy"] + doc["advisory"] == doc["total_services"], doc
    assert doc["advisory"] >= 1 and doc["total_services"] > doc["advisory"], doc


def test_the_default_run_prints_an_advisory_marker_not_a_fault_marker(tmp_path):
    """Clause 1's `[!]` half, from the printed report rather than the json.

    The icon is what a person reads: `[✗]` on a unit nobody intended to start is the
    line that sent three passes of triage looking for a broken service.
    """
    out = _default_run(tmp_path, "text",
                       {f"{CANDIDATE}.conf": UNIT_OFF.format(name=CANDIDATE)}).stdout
    # The row itself, not the `Advisory:` summary line above the block, which also
    # carries the name.
    line = next(ln for ln in out.splitlines()
                if ln.startswith("[") and CANDIDATE in ln)

    assert line.startswith("[!]"), f"printed as a fault: {line}"
    assert "[✗]" not in line, line
    assert "autostart=false" in line, line
    assert "Fleet: healthy" in out, out


def test_a_fleet_of_nothing_but_advisory_rows_is_not_counted_as_healthy_either(tmp_path):
    """The other side of "not counted": the category does not silently go green.

    With every row in the category advisory there is no graded row to average, so the
    word is the rows' own verdict (#1726's machinery), never `degraded` or `unhealthy`
    — and never `healthy`, which would make the excuse a way to hide a whole category.
    """
    doc = json.loads(_default_run(
        tmp_path, "json",
        {f"{CANDIDATE}.conf": UNIT_OFF.format(name=CANDIDATE)},
        lines=[f"{CANDIDATE}   STOPPED   Not started"]).stdout)

    fleet = [s for s in doc["services"] if s["category"] == "fleet"]
    assert [s["name"] for s in fleet] == [CANDIDATE], fleet
    assert fleet[0].get("advisory") is True, fleet[0]
    # `warn`, never `degraded`/`unhealthy` (no fault was measured) and never
    # `healthy` (nothing was graded): #1726's verdict-word path.
    assert doc["summary"]["fleet"] == "warn", doc["summary"]


# ---------------------------------------------------------------- clause 4
def test_a_unit_declaring_autostart_true_is_still_a_fault(tmp_path, monkeypatch):
    """Clause 4: the flag comes from the declaration, so a unit that says it should
    start makes STOPPED a fault again. This is the test that fails if the read is
    replaced by a list of names — no name is involved here."""
    _, rows = _derived(tmp_path, monkeypatch, {CANDIDATE: "STOPPED"},
                      {f"{CANDIDATE}.conf": UNIT_OFF.format(name=CANDIDATE)
                       .replace("autostart=false", "autostart=true")})
    row = rows[CANDIDATE]

    assert row["healthy"] is False, f"autostart=true read as hand-started: {row}"
    assert not row.get("advisory"), row


def test_a_unit_with_no_autostart_line_is_still_a_fault(tmp_path, monkeypatch):
    """No declaration is not a declaration of rest: supervisord's own default is
    `true`, and the excuse is granted only on a line that exists."""
    _, rows = _derived(tmp_path, monkeypatch, {CANDIDATE: "STOPPED"},
                      {f"{CANDIDATE}.conf":
                       f"[program:{CANDIDATE}]\ncommand=/bin/true\n"})
    row = rows[CANDIDATE]

    assert row["healthy"] is False, f"a unit with no autostart line was excused: {row}"
    assert not row.get("advisory"), row


def test_the_declaration_is_read_from_the_named_program_s_own_section(tmp_path, monkeypatch):
    """Clause 4's section scope, against the shape the real conf.d has.

    One file, two programs. `agent-watched-unit`'s section declares `autostart=true`;
    `agent-candidate-unit`'s section carries only a COMMENT that mentions
    `autostart=false` — which is what conf.d/agent-decider.conf:14 does above its own
    section. So neither another program's declaration nor a line of prose may excuse
    this one: only a `key=value` inside the named section counts.
    """
    two_units = (
        f"[program:{WATCHED}]\ncommand=/bin/true\nautostart=true\n"
        f"\n[program:{CANDIDATE}]\ncommand=/bin/true\n"
        f"; autostart=false for the same reason as the other unit\n")
    _, rows = _derived(tmp_path, monkeypatch,
                       {CANDIDATE: "STOPPED", WATCHED: "STOPPED"},
                       {"two-units.conf": two_units})

    assert not rows[CANDIDATE].get("advisory"), \
        f"a comment excused a program: {rows[CANDIDATE]}"
    assert rows[CANDIDATE]["healthy"] is False, rows[CANDIDATE]
    assert not rows[WATCHED].get("advisory"), \
        f"another program's declaration excused this one: {rows[WATCHED]}"

    # And the positive control on the same file: the moment the candidate's own
    # section carries the line as a key, the same program is advisory. Without this
    # the assertions above would pass for a parser that never finds a declaration.
    _, rows2 = _derived(tmp_path, monkeypatch, {CANDIDATE: "STOPPED"},
                        {"two-units.conf": two_units.replace("; autostart=false",
                                                             "autostart=false")})
    assert rows2[CANDIDATE].get("advisory") is True, rows2[CANDIDATE]


def test_a_declaration_outside_the_program_s_own_section_does_not_excuse_it(tmp_path, monkeypatch):
    """Clause 4 from the other side: a global `autostart=false` above the first
    `[program:...]` header is not that program's declaration.

    The shape is a real one — a key ahead of any section, or one belonging to
    `[defaults]`, is supervisord's default for programs that do not say otherwise,
    and this check reads a declaration, not defaults. A file-wide search would find
    this line and excuse the candidate; the named section has no `autostart` line at
    all, so STOPPED stays a fault here.
    """
    _, rows = _derived(tmp_path, monkeypatch, {CANDIDATE: "STOPPED"},
                      {f"{CANDIDATE}.conf": (
                          f"autostart=false\n"
                          f"[program:{CANDIDATE}]\ncommand=/bin/true\n")})
    row = rows[CANDIDATE]

    assert not row.get("advisory"), \
        f"a declaration outside the program's section excused it: {row}"
    assert row["healthy"] is False, row


def test_another_key_mentioning_autostart_is_not_the_declaration(tmp_path, monkeypatch):
    """The key boundary, not just the section boundary.

    `environment=AUTOSTART=false` is a line real units carry — an env var for the
    process, nothing to do with whether supervisord starts it — and it is the shape
    that catches a lazy read: a search anywhere in the line finds `AUTOSTART=false`
    inside it and excuses a program nobody declared hand-started. Only a line whose
    own key is `autostart` is a declaration.
    """
    _, rows = _derived(tmp_path, monkeypatch, {CANDIDATE: "STOPPED"},
                      {f"{CANDIDATE}.conf": (
                          f"[program:{CANDIDATE}]\ncommand=/bin/true\n"
                          f"environment=DEVICE=AUTOSTART=false\n")})
    row = rows[CANDIDATE]

    assert not row.get("advisory"), \
        f"another key's value excused the program: {row}"
    assert row["healthy"] is False, row


def test_the_parser_reads_the_real_candidate_unit_it_was_written_for():
    """The real file, not a fixture shaped like it.

    Every other test in this section writes a throwaway unit, which proves the rule
    but not that the rule meets the box. The unit this item is about carries its
    look-alike at `agent-services/supervisor/conf.d/agent-decider.conf:14` — a header
    comment beginning `; autostart=false for the same reason as agent-djev`, well above
    the `[program:agent-decider]` header at `:27` — and its real declaration at `:32`.
    A parser that choked on that shape would keep the suite green and quietly never
    fire, which is the failure this test exists to prevent.

    It asserts the tracked file's current declaration, so it also fails if someone
    switches the candidate on at boot without noticing that the health report's
    wording is now wrong — at which point the expectation and the live run move
    together, deliberately.
    """
    text = (ROOT / "agent-services" / "supervisor" / "conf.d"
            / "agent-decider.conf").read_text()

    assert _load_module()._section_autostart(text, "agent-decider") == "false", \
        "the tracked unit's own declaration stopped being readable"
    # The comment alone must not be what carries it: strip the section's real line and
    # the prose stays prose.
    stripped = "\n".join(ln for ln in text.splitlines()
                         if ln.strip() != "autostart=false")
    assert _load_module()._section_autostart(stripped, "agent-decider") is None, \
        "the header comment alone excused the program"
    # And a different program's section in the same file is not this one's declaration.
    assert _load_module()._section_autostart(text, "agent-some-other-program") is None


def test_a_missing_conf_d_leaves_a_stopped_unit_a_fault(tmp_path, monkeypatch):
    """Fail closed on a failed read: no unit directory, no excuse.

    The direction matters — inventing an excuse would hide a service that really is
    down, which is the failure this skill exists to catch.
    """
    conf = tmp_path / "supervisord.conf"
    conf.write_text("[supervisord]\nnodaemon=false\n")        # no conf.d beside it
    mod = _load_module()
    monkeypatch.setattr(mod, "SUPervisor_CONF", str(conf))
    monkeypatch.setattr(mod, "_switched_off", lambda: set())
    monkeypatch.setattr(mod, "subprocess",
                        _Fleetctl([f"{CANDIDATE}   STOPPED   Not started"]))

    row = {r["name"]: r for r in mod.derived_supervisor_rows()}[CANDIDATE]
    assert not row.get("advisory"), f"an unread declaration excused a program: {row}"
    assert row["healthy"] is False, row


def test_a_declared_entry_never_takes_the_hand_started_path(tmp_path, monkeypatch):
    """Clause 2's other half: `agent-djev` and `agent-llm-secondary` both declare
    `autostart=false` in the real conf.d, and neither may borrow this excuse.

    A declared entry is one someone wrote a probe for because the box is meant to run
    it — for the two GPU-2 tenants `autostart=false` only means the boot reconcile
    owns the start. `declared=True` is what `derived_supervisor_rows()` passes for a
    program `SERVICES` refers to; the same fixture with `declared=False` is the
    candidate case, so the two calls differ only in that flag.
    """
    units = {"agent-djev.conf": UNIT_OFF.format(name="agent-djev")}
    conf = _conf_dir(tmp_path, units)
    mod = _load_module()
    monkeypatch.setattr(mod, "SUPervisor_CONF", str(conf))

    graded = mod._process_state_row("agent-djev", "STOPPED", declared=True)
    assert graded["healthy"] is False, graded
    assert not graded.get("advisory"), \
        f"a declared service was excused by its unit file: {graded}"
    assert "STOPPED" in graded["status"], graded

    candidate = mod._process_state_row("agent-djev", "STOPPED")
    assert candidate.get("advisory") is True, \
        "the same unit excuses the program only when no probe declares it"


def test_the_coverage_pass_tells_a_declared_program_from_a_candidate(tmp_path, monkeypatch):
    """Clause 2's wiring at the call site, which the direct call cannot pin.

    `derived_supervisor_rows()` decides the flag with `name in declared`, and a
    `_process_state_row` called with the argument missing would keep every other test
    here green. So this one watches the call: `agent-djev` is a declared service,
    switched off here so the declared pass skips it and the coverage pass reports it
    (that is the only route a declared program reaches this code by), and its unit in
    this throwaway conf.d declares `autostart=false` exactly as the real one does. It
    must arrive as declared — and keep `_state_verdict`'s `expected stopped` wording,
    which is config-derived and outranks a unit file that never changes.
    """
    conf = _conf_dir(tmp_path, {f"{CANDIDATE}.conf": UNIT_OFF.format(name=CANDIDATE),
                                "agent-djev.conf": UNIT_OFF.format(name="agent-djev")})
    mod = _load_module()
    monkeypatch.setattr(mod, "SUPervisor_CONF", str(conf))
    monkeypatch.setattr(mod, "_switched_off", lambda: {"agent-djev"})
    monkeypatch.setattr(mod, "subprocess", _Fleetctl(
        [f"{CANDIDATE}   STOPPED   Not started",
         "agent-djev                   STOPPED   Not started"]))

    seen = {}
    real_row = mod._process_state_row

    def spy(name, state, declared=False):
        seen[name] = declared
        return real_row(name, state, declared)

    monkeypatch.setattr(mod, "_process_state_row", spy)
    rows = {r["name"]: r for r in mod.derived_supervisor_rows()}

    assert seen == {CANDIDATE: False, "agent-djev": True}, \
        f"the coverage pass stopped telling them apart: {seen}"
    assert rows["agent-djev"]["status"].startswith("expected stopped"), rows["agent-djev"]
    assert not rows["agent-djev"].get("advisory"), \
        f"a declared service borrowed the hand-started excuse: {rows['agent-djev']}"
    assert rows[CANDIDATE].get("advisory") is True, rows[CANDIDATE]


def test_the_skill_says_a_declared_off_unit_is_reported_and_not_counted():
    """Clause 5: the paragraph a reader reaches for in the middle of an incident.

    Before #1818 this section said the opposite of what the code did — that coverage
    derived from `supervisorctl status` "cannot see" `agent-decider` at all, framing
    the exclusion as deliberate. The daemon is given that program, the check grades it,
    and the noise the paragraph claimed to be avoiding was `Fleet: degraded` on every
    single run. So the skill must state the ruling (reported with `[!]`, excluded from
    the graded arithmetic) and the stale claim must be gone, not merely supplemented.
    """
    # Three roots, because the promotion gate runs the suite with HOME pointed at a
    # round home that has no vault in it: `~` alone would make this skip exactly where
    # the clause has to be pinned, and a skipped test is not a pin. The passwd entry is
    # the operator's real home whatever HOME says; it is the same outside-the-tree read
    # the sibling coverage test already makes, just not fooled by the redirect.
    candidates = [Path(__file__).resolve().parents[1] / "skills",
                  Path.home() / "obsidian" / "skills"]
    try:
        import pwd
        candidates.append(Path(pwd.getpwuid(os.getuid()).pw_dir) / "obsidian" / "skills")
    except (KeyError, ImportError):        # no passwd entry / non-POSIX
        pass
    for root in candidates:
        skill = root / "service-health-check" / "SKILL.md"
        if skill.exists():
            break
    else:
        pytest.skip(f"no skill library under {candidates}")
    text = skill.read_text()

    assert "autostart=false" in text, "the skill never mentions the declaration"
    assert "not counted" in text, "reported-but-not-counted is not stated"
    assert "[!]" in text, "the skill does not say which marker such a row prints"
    for stale in ("cannot see", "What is not covered"):
        assert stale not in text, f"the skill still claims {stale!r}"
    # It must still name the two programs that declare the same thing and stay graded,
    # or a reader will conclude the excuse covers them too.
    assert "agent-djev" in text and "agent-llm-secondary" in text, text[-1200:]


def test_the_script_names_no_programs_to_excuse():
    """Clause 4's rail: the excuse must stay a read, so no name list may appear.

    `git grep -n 'EXPECTED_STOPPED\\|DECIDER' -- scripts/` is empty at this commit and
    must stay empty; asserting it over the file text is the same claim without depending
    on git, and it holds the stronger version — the script may not name the candidate
    even in prose, since the status line prints the unit file's name instead.
    """
    text = SCRIPT.read_text()

    assert "EXPECTED_STOPPED" not in text, "a second hand-maintained list appeared"
    for name in ("DECIDER", "decider"):
        assert name not in text, f"{name!r} is now hard-coded into the check"
