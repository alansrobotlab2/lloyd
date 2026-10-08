"""#2428 — the non-pool executor the persistence-arm window was shipped without.

`eval/run_persistence_arms.sh` landed in e654d56d with a hold that cannot be taken
from inside the pool it holds: `PersistenceHold.take()`
(`eval/persistence_arms.py:234`) pauses dispatch and then `wait_idle()` (`:271-287`)
refuses while any queue row is in flight, and the dispatched run of the window is
itself such a row — `IN_FLIGHT_STATES` at `:89` is read off the depth that
`workers/queue.py:1103-1111` builds with `SELECT source, state, COUNT(*) FROM queue
GROUP BY source, state`, which does not exclude the caller. So a pool job burns the
900 s drain and exits 2; a pytest round cannot run it either (tests/conftest.py
refuses the production tree, and the window needs the live aggregator and engine).
As of triage, 13 `lloyd-*` user units were installed and none of them, and none
tracked under `agent-services/systemd/`, was the missing caller — the measurement
#2194 is owed sat at three rows of `rep: 1`, unchanged since 2026-10-04.

These nodes pin the caller as configuration, the way `tests/test_cert_renew_units.py`
pins the renewal units: parsed through the INI shape systemd reads, the service must
be a oneshot that execs the shipped script at ten reps and overrides neither the rows
nor the report path; the timer must be a single recurring quiet-hour one whose comment
names the scheduled work the hour avoids; and `install-services.sh`'s symlink loop
must actually link both, which is tested by RUNNING the installer against a throwaway
`HOME` and a stub `systemctl` on the front of `PATH` — the same seam-crossing shape
`tests/test_unit_enabledness_check.py` uses, so the host's manager is never touched.

What is NOT pinned here, deliberately: that the window runs and lands a hold on the
live box. That needs the aggregator, the engine and ~2 h 30 m, and pausing this box's
pool from a test is exactly what the runner refuses to do to somebody else's landing.
It is owed after landing and named in the item.

Run: .venvs/lloyd/bin/python -m pytest tests/test_persistence_window_unit.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import run_injection_canary as RC  # noqa: E402

UNIT_DIR = ROOT / "agent-services" / "systemd"
SERVICE = UNIT_DIR / "lloyd-persistence-window.service"
TIMER = UNIT_DIR / "lloyd-persistence-window.timer"
INSTALLER = ROOT / "agent-services" / "setup" / "install-services.sh"
SETUP = ROOT / "SETUP.md"
WINDOW_SCRIPT = "eval/run_persistence_arms.sh"
REPS = "10"


def _parse(path: Path):
    """Read a unit file the way systemd's loader does: sections, `Key=value`, no
    interpolation (a `%h` in a value is systemd's own syntax, not configparser's) and
    key case preserved."""
    import configparser
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str
    cp.read(path, encoding="utf-8")
    return cp


def _env_lines(path: Path) -> list[str]:
    """Every `Environment=` line in the unit. systemd applies all of them while an INI
    read keeps only the last, so the rows/report assertion has to see them all."""
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip().startswith("Environment=")]


def _comments(path: Path) -> str:
    """The unit's comment prose — what the timer's choice of hour has to be argued in,
    since systemd reads none of it."""
    return "\n".join(l for l in path.read_text(encoding="utf-8").splitlines()
                     if l.strip().startswith("#"))


def _execstart() -> str:
    return _parse(SERVICE).get("Service", "ExecStart")


def _git_tracked(path: Path) -> bool:
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                        str(path.relative_to(ROOT))],
                       capture_output=True, text=True)
    return r.returncode == 0


def _systemd_seconds(value: str) -> int:
    """`4h`, `90s`, `1min 30s` -> seconds. Only the units this repo's units use."""
    total = 0
    for n, unit in re.findall(r"(\d+)\s*(us|ms|s|min|h|d)?", value):
        total += int(n) * {"": 1, "us": 1e-6, "ms": 1e-3, "s": 1,
                           "min": 60, "h": 3600, "d": 86400}[unit or ""]
    return int(total)


# ── clause 1: a oneshot that runs the shipped window at ten reps, on the defaults ──

def test_both_units_are_tracked_in_the_repo():
    """Untracked units are host drift: a fresh checkout or a rebuild would get no
    window at all, which is the state #2428 was filed against."""
    for path in (SERVICE, TIMER):
        assert path.exists(), path
        assert _git_tracked(path), f"{path.name} is not tracked in git"


def test_the_service_is_a_oneshot_that_runs_the_window_at_ten_reps():
    """Clause 1, first half: `Type=oneshot` and an `ExecStart` that runs the shipped
    script with the rep count as its argument. A timer on a service that never runs
    the window is the same gap with a clock attached."""
    cp = _parse(SERVICE)
    assert cp.has_section("Service"), SERVICE
    assert cp.get("Service", "Type") == "oneshot", "a long-running service would hold " \
        "the pool on every boot; this is one window per fire, then it is done"
    tokens = _execstart().split()
    assert tokens[0].endswith(WINDOW_SCRIPT), \
        f"ExecStart must run the shipped window script: {_execstart()!r}"
    assert tokens[1:] == [REPS], \
        f"the window must be invoked at ten reps, got {tokens[1:]!r}"


def test_the_script_systemd_execs_is_the_tracked_executable_one():
    """The unit names a path by suffix, so this is what proves the suffix resolves:
    the script is tracked, and executable, which is the difference between `exec`
    working and `EACCES` in the journal at 02:10 with nobody watching."""
    script = ROOT / WINDOW_SCRIPT
    assert script.is_file() and _git_tracked(script), script
    assert os.access(script, os.X_OK), f"{script} is not executable"


def test_the_unit_sets_neither_override_so_the_reps_land_on_the_tracked_corpus():
    """Clause 1, second half, across the seam it matters at.

    The unit's `Environment=` lines are the whole environment systemd hands the
    script, so neither `LLOYD_CANARY_ROWS` nor `LLOYD_CANARY_REPORT` may appear in
    them: with both unset the bench's own `rows_path()` resolves to the tracked
    `eval/measurements/injection-canary/rows.jsonl` — the file #2194's leak-rate
    denominator is read from — and the dated report keeps its tracked name pattern.
    A window writing a scratch path measures itself into nowhere, and the rows would
    never reach the grader that owes the answer."""
    keys = {l.split("=", 1)[1].split("=", 1)[0] for l in _env_lines(SERVICE)}
    for override in (RC.ROWS_ENV_VAR, RC.REPORT_ENV_VAR):
        assert override not in keys, f"the unit overrides the bench's own default: {override}"
    assert "LLOYD_CANARY" not in _execstart(), "an inline assignment in ExecStart is an " \
        "override by another name"

    monkey = pytest.MonkeyPatch()
    monkey.delenv(RC.ROWS_ENV_VAR, raising=False)
    monkey.delenv(RC.REPORT_ENV_VAR, raising=False)
    try:
        assert RC.rows_path() == RC.ROWS_PATH, "with no override the bench must land on " \
            "its tracked rows file"
        assert RC.ROWS_PATH.parent.name == "injection-canary", RC.ROWS_PATH
        assert _git_tracked(RC.ROWS_PATH), "the rows file the window appends to is tracked"
        assert RC.report_path("2026-10-04").name == "run-2026-10-04-persistence.md"
        assert RC.report_path("2026-10-04").parent == RC.OUT_DIR
    finally:
        monkey.undo()


# ── clause 2: systemd opens the window; no pool job could ──

def test_the_execstart_is_a_direct_exec_of_the_script_and_not_an_enqueue():
    """Clause 2: the window is opened by systemd, not by a job that its own running
    queue row would block. So the first token IS the script — no `bash -c`, no
    `python -c` — and nothing in the executable surface (ExecStart plus the
    environment) reaches for `curl`, an enqueue route or the pause API: the driver's
    own `PoolPause` is the only thing that may take a hold, because its resume is the
    one that refuses to lift the automod promoter's hold along with the operator's
    (`workers/pool.py:716-728`)."""
    execstart = _execstart()
    tokens = execstart.split()
    assert tokens[0].endswith(WINDOW_SCRIPT), execstart
    assert not re.search(r"(^|/)(ba)?sh$", tokens[0]), \
        f"wrapping the script in a shell re-derives the signal handling: {execstart!r}"
    forbidden = ["curl", "wget", "enqueue", "POST", "POST ", "/api/", "/api/workers",
                 "bash -c", "sh -c", "python -c"]
    surface = "\n".join([execstart, *_env_lines(SERVICE)])
    hits = [f for f in forbidden if f in surface]
    assert not hits, f"the unit reaches the pool itself ({hits}); that is the route the " \
        "window cannot use: its own claimed/running row is what wait_idle() refuses on"


# ── clause 3: one quiet hour, argued against the box's other schedules ──

def test_the_timer_fires_once_a_day_at_one_hour_and_catches_up():
    """Clause 3, first half: exactly one `OnCalendar=`, `Persistent=true` so a box
    that was asleep when the window was due still runs one, and `timers.target` so
    the user manager actually starts it once enabled."""
    cp = _parse(TIMER)
    lines = [l.strip() for l in TIMER.read_text(encoding="utf-8").splitlines()
             if l.strip().startswith("OnCalendar=")]
    assert len(lines) == 1, f"a window this long gets exactly one slot: {lines}"
    assert cp.get("Timer", "Persistent").lower() == "true"
    assert cp.get("Install", "WantedBy") == "timers.target"
    value = cp.get("Timer", "OnCalendar")
    assert value.startswith("*-*-* "), f"daily, not weekday-scoped: {value}"
    hh, mm = value.split()[-1].split(":")[:2]
    assert 1 <= int(hh) <= 2, \
        f"the slot must sit in the 01:00-02:59 band that has no nightly writes, no " \
        f"morning measurement and ~no user turns: {value}"
    assert int(mm) <= 15, f"start early: the ~2 h 30 m window must finish before 05:00 " \
        f"jobs and the morning human band: {value}"


def test_the_timer_comment_names_the_scheduled_work_the_hour_avoids():
    """Clause 3, second half: the hour is only defensible as prose. It must name the
    chain it waits out (the 22:00-04:00 write window the precedent
    `lloyd-qmd-cleanup.timer` documents, which measures out at 22:00 / 00:00 / 01:00
    in the run store) and the morning measurements it stays ahead of — the 04:00
    health report, the 05:00 qmd index maintenance, the 05:30 graph backup and the
    06:00 retrieval eval, which is itself an engine-heavy measurement. A number with
    no provenance behind it would be decoration, so the band's own human-turn count
    is quoted too."""
    prose = _comments(TIMER)
    assert prose.strip(), f"{TIMER.name} chose its hour in silence"
    for token, what in [("22:00", "the nightly write window"),
                        ("01:00", "the reflection writes it waits out"),
                        ("04:00", "#60 knowledge-health-report"),
                        ("05:30", "the graph backup"),
                        ("06:00", "#82 nightly-retrieval-eval")]:
        assert token in prose, f"the comment does not name {what} as avoided: {token}"
    assert re.search(r"2 h 30 m|~2 h 30", prose), \
        "the comment must state the wall clock the window occupies"


def test_the_service_states_the_wall_clock_and_outlives_the_manager_default():
    """Clause 3, third half, and the one that would have bitten silently: the user
    manager's default start timeout here is 90 s (`systemctl --user show default`
    reads `TimeoutStartUSec=1min 30s`), so an oneshot that inherits it is SIGTERM'd
    inside rep 1 — the driver resumes dispatch on the way out, so no hold leaks, but
    the timer records a clean exit and produces no rows. The unit must state the
    ~2 h 30 m budget (thirteen minutes of engine per rep, the driver's own figure)
    and set `TimeoutStartSec` above the 2 h 45 m worst case that budget allows."""
    cp = _parse(SERVICE)
    execstart = _execstart()
    text = SERVICE.read_text(encoding="utf-8")
    assert "2 h 30 m" in text, "the unit must state the wall clock it expects"
    assert "thirteen minutes" in text.lower(), "…with its per-rep budget named"
    assert execstart in text, "the wall clock must be documented beside the line it describes"
    declared = cp.get("Service", "TimeoutStartSec")
    assert _systemd_seconds(declared) > _systemd_seconds("2h 45m"), \
        f"TimeoutStartSec={declared} is under the window's own worst case"
    assert _systemd_seconds(declared) <= _systemd_seconds("8h"), \
        f"TimeoutStartSec={declared} is a licence to hang, not a budget"


# ── clause 4: the installer links them, and neither is system scope ──

def _run_installer(tmp_path: Path) -> tuple[Path, list[str]]:
    """Run `install-services.sh` for real, against a throwaway HOME and a stub
    `systemctl`/`loginctl` on the front of PATH.

    The symlink loop is the whole of clause 4, and it is a shell loop crossing a
    process boundary into systemd: grepping the unit names into the script would
    prove nothing about what the script DOES. The stub records every argv, and every
    caller below asserts the stub was consulted, so a script that resolved
    `/usr/bin/systemctl` by absolute path fails here instead of reloading the live
    manager of whatever box the suite ran on."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "systemctl.log"
    for tool, stdout in (("systemctl", ""), ("loginctl", "yes")):
        stub = bin_dir / tool
        stub.write_text("#!/usr/bin/env bash\n"
                        f'printf \'%s\\n\' "$*" >> "{log}"\n'
                        + (f"echo '{stdout}'\n" if stdout else "")
                        + "exit 0\n")
        stub.chmod(0o755)
    home = tmp_path / "home"
    # The environment this test overrides has to be replaced as a whole, not sparsely:
    # `install-services.sh` runs `set -u` and closes by reading `$USER` (`loginctl
    # show-user "$USER"`, line 89), so a runner whose environment carries no login
    # name — the gate's does not, which is how this was caught — fails the script for
    # a reason that has nothing to do with the units. HOME, PATH and USER are what a
    # login shell guarantees, so all three are named rather than inherited.
    env = dict(os.environ, HOME=str(home), USER="lloyd-window-test",
               PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
               STUB_LOG=str(log))
    proc = subprocess.run(["bash", str(INSTALLER)], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path), timeout=120)
    assert proc.returncode == 0, f"installer exited {proc.returncode}: {proc.stderr[-500:]}"
    assert log.is_file() and "--user daemon-reload" in log.read_text(), \
        "the installer never asked systemctl to reload the user manager"
    return home / ".config" / "systemd" / "user", log.read_text().splitlines()


def test_the_installer_links_both_new_units_and_reloads_the_daemon(tmp_path):
    """Clause 4, first half: one run of `install-services.sh` puts both files in
    `~/.config/systemd/user/` pointing back at the repo, with no manual step outside
    the repo. This is the loop as it runs, not a reading of it."""
    unit_dir, _ = _run_installer(tmp_path)
    for path in (SERVICE, TIMER):
        linked = unit_dir / path.name
        assert linked.is_symlink(), f"{path.name} was not linked by the installer"
        assert linked.resolve() == path.resolve(), \
            f"{path.name} links to somewhere other than the tracked unit"


def test_neither_new_unit_is_declared_system_scope():
    """Clause 4, second half: `SYSTEM_SCOPE_UNITS` is how a unit escapes the loop
    (`nvidia-power-limit.service` needs root, so a user-scope copy of it can only
    lie). A persistence-window unit that landed there would be linked by nothing at
    all, which is #1108's shape. The stale-scope assertion has its own positive
    control in the same parse: if the array were renamed, this fails rather than
    passing vacuously."""
    src = INSTALLER.read_text(encoding="utf-8")
    m = re.search(r"SYSTEM_SCOPE_UNITS=\(([^)]*)\)", src)
    assert m, "install-services.sh no longer declares SYSTEM_SCOPE_UNITS — re-check this"
    scope = re.findall(r"[\w.-]+", m.group(1))
    assert "nvidia-power-limit.service" in scope, f"parsed the wrong array: {scope}"
    for path in (SERVICE, TIMER):
        assert path.name not in scope, f"{path.name} is system scope, so nothing links it"


# ── the enable step, which is host state the loop cannot set ──

def test_setup_md_carries_the_enable_route():
    """A linked-but-never-enabled timer is the drift #1891 exists to catch:
    `install-services.sh` echoes the `systemctl --user enable` lines instead of
    running them, and `scripts/maintenance/check-unit-enabledness.sh` asserts every
    repo timer answers `enabled`. The enable itself is host state and cannot be set
    from a round (`systemctl` is refused at dispatch), so this pins the one place a
    human or a rebuild looks: the same page that carries the cert-renew line."""
    text = SETUP.read_text(encoding="utf-8")
    assert "systemctl --user enable --now lloyd-persistence-window.timer" in text, \
        "SETUP.md must carry the enable route, or the timer ships disabled"
