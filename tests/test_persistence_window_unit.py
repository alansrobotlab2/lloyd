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
be a oneshot that execs the shipped script at ten reps with a drain that outlasts a
measured autocode round (#2595: `--drain-wait 4200`, because 642 of the 898 autocode
runs in the 14 days to 2026-10-12 ran past the driver's 900 s default) and overrides
neither the rows nor the report path; the timer must be a single recurring quiet-hour
one whose comment names the scheduled work the hour avoids; and
`install-services.sh`'s symlink loop must actually link both, which is tested by
RUNNING the installer against a throwaway
`HOME` and a stub `systemctl` on the front of `PATH` — the same seam-crossing shape
`tests/test_unit_enabledness_check.py` uses, so the host's manager is never touched.

What is NOT pinned here, deliberately: that the window runs and lands a hold on the
live box. That needs the aggregator, the engine and the ~3 h 20 m of drain, reps and
grade that `ExecStart` now budgets, and pausing this box's
pool from a test is exactly what the runner refuses to do to somebody else's landing.
It is owed after landing and named in the item.

Run: .venvs/lloyd/bin/python -m pytest tests/test_persistence_window_unit.py
"""
from __future__ import annotations

import os
import re
import shlex
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
WINDOW_PATH = ROOT / WINDOW_SCRIPT
DRIVER = ROOT / "eval" / "persistence_arms.py"
REPS = "10"
DRAIN_FLAG = "--drain-wait"
DRAIN_WAIT_S = 4200       # the ExecStart budget; the driver's own default is 900 s
PER_REP_S = 13 * 60       # thirteen minutes: the driver's figure for one rep
WINDOW_S = DRAIN_WAIT_S + int(REPS) * PER_REP_S      # 4200 + 7800 = 12000 s = 3 h 20 m
RETRIEVAL_EVAL_S = 6 * 3600       # #82 nightly-retrieval-eval, the bound on this span


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


def _exec_argv() -> list[str]:
    """The `ExecStart` line as argv, split the way systemd splits it: POSIX quoting, so
    a quoted path holding a space stays one token where `str.split` would cut it in two
    and the test would be reading a different command than the one systemd runs."""
    return shlex.split(_execstart())


def _driver_args():
    """`ExecStart`'s own arguments, parsed by the DRIVER's `build_parser()` rather than
    by a regex beside it.

    This is the seam the drain lives on: a flag the driver does not know is an
    `SystemExit` here, a value it cannot cast is one too, and an argument the wrapper
    drops before the driver ever sees it arrives as `drain_wait is None`, which is
    precisely the driver's way of saying "fall back to the 900 s default"
    (`eval/persistence_arms.py:458`). Reading the unit through the code that will read it
    at 02:10 is what keeps a green suite from meaning a window that silently drains for
    900 s again."""
    sys.path.insert(0, str(ROOT / "eval"))
    import persistence_arms as PA
    return PA.build_parser().parse_args(_exec_argv()[1:])


def _unit_window_seconds() -> int:
    """The span the unit actually asks for: the drain it passes plus the reps it asks
    for, at the driver's own thirteen minutes each. Derived from `ExecStart`, never from
    the constants above, so a drain changed in the unit moves this number and the prose
    that has to match it."""
    ns = _driver_args()
    assert ns.drain_wait is not None, \
        f"ExecStart passes no {DRAIN_FLAG}, so the driver resolves the 900 s default at " \
        f"eval/persistence_arms.py:458 and there is no span to do arithmetic on: {_execstart()!r}"
    return int(ns.drain_wait + ns.reps * PER_REP_S)


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


def test_the_service_is_a_oneshot_running_ten_reps_with_a_drain_that_outlasts_a_round():
    """Clauses 1 and 2: `Type=oneshot`, and an `ExecStart` that runs the shipped script
    at ten reps with `--drain-wait 4200`. A timer on a service that never runs the
    window is the same gap with a clock attached.

    The argument list used to be pinned as exactly `[10]`, which is the whole reason the
    drain was still the driver's 900 s default: the drain this window runs is an
    `ExecStart` argument, and 642 of the 898 autocode runs in the 14 days to 2026-10-12
    outlived 900 s (longest 3552 s), with the queue busy at 02:10 on 18 of the 20 nights
    the run store covers — so the old default made the nightly fire a refusal, and the
    one fire this unit has had is exactly that. Pinned now, in this order: the script
    suffix, the rep count, the flag appearing exactly once, and the drain seconds AS THE
    DRIVER PARSES THEM — not as a substring of the line. A value argparse would refuse, a
    flag argparse never sees and a second flag silently overriding the first each turn
    this red rather than passing on text."""
    cp = _parse(SERVICE)
    assert cp.has_section("Service"), SERVICE
    assert cp.get("Service", "Type") == "oneshot", "a long-running service would hold " \
        "the pool on every boot; this is one window per fire, then it is done"
    argv = _exec_argv()
    assert argv[0].endswith(WINDOW_SCRIPT), \
        f"ExecStart must run the shipped window script: {_execstart()!r}"
    assert argv[1] == REPS, \
        f"the window must be invoked at ten reps, got {argv[1:]!r}"
    assert argv.count(DRAIN_FLAG) == 1, \
        f"exactly one {DRAIN_FLAG} may set the drain, got {argv[2:]!r}"
    ns = _driver_args()
    assert ns.reps == int(REPS), f"argparse read the rep count as {ns.reps}"
    assert ns.drain_wait == float(DRAIN_WAIT_S), \
        f"{DRAIN_FLAG} must hand the driver {DRAIN_WAIT_S} s, parsed {ns.drain_wait}"


def test_the_script_systemd_execs_is_the_tracked_executable_one():
    """The unit names a path by suffix, so this is what proves the suffix resolves:
    the script is tracked, and executable, which is the difference between `exec`
    working and `EACCES` in the journal at 02:10 with nobody watching."""
    script = ROOT / WINDOW_SCRIPT
    assert script.is_file() and _git_tracked(script), script
    assert os.access(script, os.X_OK), f"{script} is not executable"


def test_the_unit_arguments_reach_the_driver_through_the_shipped_wrapper(tmp_path):
    """Clause 5: the process boundary the unit's numbers cross before anything runs.

    `ExecStart` names the WRAPPER, which validates the rep count and then
    `exec "$PYTHON" "$ROOT/eval/persistence_arms.py" "$@"`
    (`eval/run_persistence_arms.sh:49`). A `"$@"` dropped there is a unit that still
    exits 0 and drains for 900 s, because the flag and not the exit code is what carries
    the budget — and a renamed flag is an argparse error at 02:10, in a journal nobody is
    reading. So this runs the tracked wrapper for real, passing the unit's OWN arguments,
    with a stub interpreter standing where the venv python would: the stub records argv
    and exits, so no window opens, no dispatch is held and no engine is touched, which is
    exactly what `tests/conftest.py` exists to keep a round from doing.

    Three things get pinned, because they fail separately: the forwarded tail is the
    unit's tail verbatim (a dropped or reordered argument), the driver's `build_parser()`
    accepts THAT tail and reads 4200.0 out of it (a flag the wrapper forwards but the
    driver does not know), and the no-flag case resolves to `drain_wait is None` — the
    driver's own way of saying it is about to use the 900 s default, which is the state
    this item was filed to end (`eval/persistence_arms.py:458`)."""
    assert _git_tracked(WINDOW_PATH) and '"$@"' in WINDOW_PATH.read_text(encoding="utf-8"), \
        "the wrapper must forward its arguments to the driver"

    argv = _exec_argv()
    stub, out = tmp_path / "stub-python", tmp_path / "argv.txt"
    stub.write_text("#!/usr/bin/env bash\n"
                    f'printf \'%s\\n\' "$@" > "{out}"\n'
                    "exit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    # 180 s, not the 60 s this node shipped with: at head db55d731 the gate's parallel
    # pass timed out here (`parallel_only_failures` names this node, `failed: 0` after
    # its serial re-run), while a second round's suite was live on the box at the same
    # time — 70+ unrelated nodes landed in that same list. The fork itself is sub-second
    # in isolation; the budget has to survive a contended machine, and the sibling
    # wrapper test already allows 90 s (`tests/test_run_persistence_arms.py:202`).
    proc = subprocess.run([str(WINDOW_PATH), *argv[1:]], capture_output=True,
                          text=True, timeout=180, cwd=str(tmp_path),
                          env=dict(os.environ, LLOYD_CANARY_PYTHON=str(stub)))
    assert proc.returncode == 0, \
        f"the wrapper refused the unit's own arguments ({proc.returncode}): " \
        f"{(proc.stderr or proc.stdout)[-300:]}"
    forwarded = out.read_text(encoding="utf-8").splitlines()
    assert forwarded and Path(forwarded[0]).resolve() == DRIVER.resolve(), \
        f"the wrapper did not exec the driver it names: {forwarded[:1]!r}"
    assert forwarded[1:] == argv[1:], \
        f"the wrapper dropped or reordered the unit's arguments: {forwarded[1:]!r} " \
        f"!= {argv[1:]!r}"

    sys.path.insert(0, str(ROOT / "eval"))
    import persistence_arms as PA
    ns = PA.build_parser().parse_args(forwarded[1:])
    assert ns.reps == int(REPS) and ns.drain_wait == float(DRAIN_WAIT_S), \
        f"the driver read reps={ns.reps} drain_wait={ns.drain_wait} off the forwarded " \
        f"tail {forwarded[1:]!r}"
    assert PA.build_parser().parse_args([REPS]).drain_wait is None, \
        "with no flag the driver falls back to the 900 s default — so the flag's " \
        "presence in ExecStart is load-bearing, not decorative"


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
    assert int(mm) <= 15, f"start early: from the latest start the 300 s jitter allows " \
        f"(02:15) the ~3 h 20 m span must still end clear of the 06:00 retrieval eval: " \
        f"{value}"


def test_the_timer_comment_names_the_scheduled_work_the_hour_avoids():
    """#2595 clause 4 (the second half of #2428's clause 3): the hour is only defensible
    as prose, and the prose has to describe the window the service actually runs.

    It must name the chain it waits out (the 22:00-04:00 write window the precedent
    `lloyd-qmd-cleanup.timer` documents, which measures out at 22:00 / 00:00 / 01:00 in
    the run store), the drain it exists to wait for, and the one measurement it stays
    ahead of — the 06:00 retrieval eval, itself engine-heavy, which is why the drain has
    a bound rather than a preference. The 04:00 health report and the 05:00 qmd index
    maintenance now fall INSIDE the span and the comment says so: both are
    pool-dispatched, so they queue behind the hold and run after it, whereas the 05:30
    graph backup is a user timer that does not queue, and is tolerable only because it
    copies the KG store and touches no engine. Saying "before the morning measurements"
    of a window that overlaps three of them would be a claim the schedule contradicts. A
    number with no provenance behind it would be decoration, so the band's own human-turn
    count is quoted too."""
    prose = _comments(TIMER)
    assert prose.strip(), f"{TIMER.name} chose its hour in silence"
    for token, what in [("22:00", "the nightly write window"),
                        ("01:00", "the reflection writes it waits out"),
                        ("04:00", "#60 knowledge-health-report"),
                        ("05:30", "the graph backup"),
                        ("06:00", "#82 nightly-retrieval-eval")]:
        assert token in prose, f"the comment does not name {what} as avoided: {token}"
    assert re.search(r"3 h 20 m|~3 h 20", prose), \
        "the comment must state the wall clock the window occupies"
    assert str(DRAIN_WAIT_S) in prose, \
        f"…and name the {DRAIN_WAIT_S} s drain it waits out, or the two units describe " \
        "two different windows read side by side"
    assert "02:15" in prose, \
        "the span only clears 06:00 from the latest start the jitter allows, so the " \
        "comment has to carry that start, not the nominal 02:10"
    assert "queue behind" in prose, \
        "the 04:00 and 05:00 jobs fall inside a span this long: the comment has to say " \
        "they queue behind the hold and run after it, not that the hour stays ahead"


def test_the_service_states_the_wall_clock_and_outlives_the_manager_default():
    """Clause 3, third half, and the one that would have bitten silently: the user
    manager's default start timeout here is 90 s (`systemctl --user show default`
    reads `TimeoutStartUSec=1min 30s`), so an oneshot that inherits it is SIGTERM'd
    inside rep 1 — the driver resumes dispatch on the way out, so no hold leaks, but
    the timer records a clean exit and produces no rows. The unit must state the
    ~3 h 20 m budget as the sum of its two terms — the 4200 s drain ExecStart passes and
    thirteen minutes of engine per rep, the driver's own figure — and set
    `TimeoutStartSec` above that worst case. The drain and the rep count are parsed out
    of `ExecStart` and multiplied at the driver's own thirteen minutes rather than read
    off the prose, because the drain is the term a future round will be tempted to raise
    and the 06:00 bound is what says by how much."""
    cp = _parse(SERVICE)
    execstart = _execstart()
    text = SERVICE.read_text(encoding="utf-8")
    assert "3 h 20 m" in text, "the unit must state the wall clock it expects"
    assert "thirteen minutes" in text.lower(), "…with its per-rep budget named"
    assert str(DRAIN_WAIT_S) in text, "…and the drain term beside it, not merged into the sum"
    window = _unit_window_seconds()
    assert window == WINDOW_S == 12000, \
        f"3 h 20 m is {DRAIN_WAIT_S} s of drain + {REPS} reps x {PER_REP_S} s = " \
        f"{WINDOW_S} s, but ExecStart asks for {window} s"
    assert execstart in text, "the wall clock must be documented beside the line it describes"
    declared = cp.get("Service", "TimeoutStartSec")
    assert _systemd_seconds(declared) > window, \
        f"TimeoutStartSec={declared} is under the window's own worst case ({window} s)"
    assert _systemd_seconds(declared) <= _systemd_seconds("8h"), \
        f"TimeoutStartSec={declared} is a licence to hang, not a budget"
    # The bound the drain was sized against, computed instead of asserted as prose: the
    # latest start the timer's `RandomizedDelaySec=300` allows is 02:15, and #82
    # nightly-retrieval-eval at 06:00 is the engine-heavy measurement a window running
    # beside it would contaminate. 4200 s is what is left of 02:15→06:00 once ten reps
    # take 7800 s and ~25 min is kept for the closing grade, so a longer drain spends
    # that grade room and a much longer one runs the reps past 06:00 by themselves.
    ends = 2 * 3600 + 15 * 60 + window
    assert ends < RETRIEVAL_EVAL_S, \
        f"from the latest jittered start the window ends {ends} s past the " \
        f"06:00 retrieval eval ({RETRIEVAL_EVAL_S} s)"
    assert RETRIEVAL_EVAL_S - ends >= 1500, \
        f"only {RETRIEVAL_EVAL_S - ends} s is left for the closing grade; 4200 s of " \
        "drain is the largest value that keeps half an hour ahead of 06:00"


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
    # 300 s, from 120 s: this node and the wrapper node above both timed out in the
    # gate's parallel pass at head db55d731 (`parallel_only_failures`, `failed: 0` on the
    # serial re-run) while a second round's suite was live on the box — 70+ unrelated
    # nodes are in that same list. The installer forks `systemctl --user` through the
    # stub on PATH, so the fork is the thing the contended machine was slow at, not the
    # installer's logic; the budget has to survive that.
    proc = subprocess.run(["bash", str(INSTALLER)], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path), timeout=300)
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
    repo timer answers `enabled`. The enable is host state, so this pins the one place
    a human or a rebuild looks: the same page that carries the cert-renew line. It is
    not beyond an unattended turn's reach, and #2537 corrected the older claim here
    that it was — though since #2541 (2026-10-11) the reason is narrower than
    "#2537 measured `enable --now` as tier 1 with no service-control hit". A `--now`
    command is now classified as the unit operation systemd performs for it
    (`systemctl start` for `enable`/`reenable --now`, `systemctl stop` for
    `disable --now`) and refused to a background session, so this route survives because
    its target is a `.timer` — arming a schedule starts no unit. The service form of the
    same flag is refused, which is the asymmetry
    tests/test_service_control_guard.py::test_arming_or_unarming_a_timer_with_now_stays_reachable
    and tests/test_grant_bash_tier.py::test_arming_a_timer_is_never_tiered_as_a_unit_operation
    hold open from both sides.

    What makes the exact string below worth pinning is that only the `--now` form
    ARMS — plain `enable` links the unit file and leaves the timer
    `ActiveState=inactive` until the user manager restarts — so this route has to keep
    the flag, and the guard's FAIL line now names the same command."""
    text = SETUP.read_text(encoding="utf-8")
    assert "systemctl --user enable --now lloyd-persistence-window.timer" in text, \
        "SETUP.md must carry the enable route, or the timer ships disabled"


def test_the_watchdog_hint_for_this_timer_is_that_arming_route():
    """Clause 2 as far as a test can hold it: the two routes cannot disagree.

    Two commands look like the fix and are not it. Plain `enable` links the unit file
    and leaves `ActiveState=inactive` until the user manager is restarted, so the
    02:10 fire never arrives while `check-unit-enabledness.sh` reads green; and
    `systemctl --user start` is a service-control verb, refused to an unattended turn.
    `systemctl --user enable --now lloyd-persistence-window.timer` is the one command
    that is both arming and reachable from here — so this pins that the remediation
    the watchdog prints for this unit and the route SETUP.md carries are the SAME
    command. The drift #2537 actually hit was a hint reading `enable` where the docs
    read `enable --now`: green check, unarmed timer, no window.

    The half this node cannot see — that the script really prints the line its source
    builds — is pinned by
    `test_the_remediation_arms_the_timer_and_points_by_content` in
    tests/test_unit_enabledness_check.py, which runs the guard against a stub
    `systemctl` answering one timer `disabled`. This one owns the cross-file identity,
    which is where the divergence lived.

    Still host state, read by hand and asserted by nothing here: that the timer
    answers `enabled` (`systemctl --user is-enabled lloyd-persistence-window.timer`)
    and that it is ARMED — `systemctl --user show lloyd-persistence-window.timer
    -p ActiveState` → `ActiveState=active`, which `enabled` alone does not give."""
    guard = (ROOT / "scripts" / "maintenance" / "check-unit-enabledness.sh")
    hint = re.search(r"systemctl --user enable( --now)? \$unit",
                     guard.read_text(encoding="utf-8"))
    assert hint, "the guard's FAIL line no longer builds a remediation from $unit"
    assert hint.group(1), (
        "the guard prints bare `enable`, which only links the unit file and leaves "
        "this timer ActiveState=inactive — the hint that let it ship unarmed (#2537)")

    command = f"systemctl --user enable --now {TIMER.name}"
    assert command in SETUP.read_text(encoding="utf-8"), \
        f"the watchdog's remediation {command!r} is not the documented route"


def test_the_arming_route_is_reachable_from_an_unattended_turn():
    """The premise #2537 was confirmed on, pinned against the two guards that decide
    it rather than against a sentence about them.

    The whole item rests on an asymmetry: an unattended turn may ARM this timer but may
    never START its service. `app.harness.safety.bash_command_tier` answers 1 for
    `systemctl --user enable --now <timer>` with nothing in the durable-external table
    to match, while `systemctl --user start <service>` is tier 2 AND a service-control
    label, which is what refuses it to a background session — a worker turn that
    restarted the unit owning the primary would kill every turn in flight, its own
    included. So the arming route is loop-runnable, and the string
    `test_setup_md_carries_the_enable_route` pins is the one the loop can actually
    run; a future widening of either table flips that silently, and this is the node
    that would say so.

    Measured with the same read-only probe #2537's triage used, at this commit:
    arm → tier 1, `match_durable_external` None, `find_service_control` None;
    start → tier 2, `systemctl state change (restart/stop a unit)`, `systemctl start`.
    Nothing here executes either command."""
    from app.harness.safety import bash_command_tier, match_durable_external
    from app.harness.service_control import find_service_control

    arm = f"systemctl --user enable --now {TIMER.name}"
    assert bash_command_tier(arm) == 1, \
        "arming this timer stopped being tier-1: an unattended turn can no longer arm it"
    assert match_durable_external(arm) is None, arm
    assert find_service_control(arm) is None, \
        f"`enable --now` became service control ({find_service_control(arm)!r}), " \
        "which is refused to a background session and would put this timer back " \
        "behind a human, as #2527 believed it was"

    start = f"systemctl --user start {SERVICE.name}"
    assert bash_command_tier(start) == 2, start
    assert find_service_control(start) == "systemctl start", start
