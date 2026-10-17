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

def test_the_timer_fires_once_a_week_on_monday_and_catches_up():
    """#2754 clause 2, replacing the node that asserted `value.startswith("*-*-* ")`
    with the message "daily, not weekday-scoped". That node did its job — it refused
    this change until someone rewrote it with the reason attached — and the daily form
    it enforced is the thing #2754 bounds: two fires measured 34 min of held dispatch
    and ~190 KB of tracked rows each, committed nightly forever.

    The weekly spec goes in the day-of-week field: `Mon *-*-* 02:10:00`. The other
    spelling of "weekly-ish" is a trap that reads like the fix — `*-*-1 02:10:00` puts
    the `1` in the day-of-MONTH field, so it is not weekly at all — which is why the
    date field has to stay three wildcards and the `Mon` token has to be there as a
    token, not inferred from an absence. `test_systemd_itself_reads_the_cadence_as_weekly_not_monthly`
    asks systemd to confirm both readings; the fullmatch here is what stops a regression
    from re-introducing the trap.

    What this node does NOT change is the rest of clause 3's arithmetic, because #2754
    is a cadence-only change: the 01:00-02:59 band, the minute bound that keeps the
    span clear of the 06:00 retrieval eval, `Persistent=true`, `timers.target`, and the
    300 s jitter the 02:15 latest start is computed from. The band is argued from the
    box's schedule (the comment the node below reads), not from how often the window
    runs, and `Persistent=true` matters more at one slot a week, not less: a suspend
    across Monday now costs the week's window rather than the night's."""
    cp = _parse(TIMER)
    lines = [l.strip() for l in TIMER.read_text(encoding="utf-8").splitlines()
             if l.strip().startswith("OnCalendar=")]
    assert len(lines) == 1, f"a window this long gets exactly one slot: {lines}"
    assert cp.get("Timer", "Persistent").lower() == "true"
    assert cp.get("Install", "WantedBy") == "timers.target"
    value = cp.get("Timer", "OnCalendar")
    assert re.fullmatch(r"Mon \*-\*-\* \d{2}:\d{2}:\d{2}", value), \
        f"weekly in the day-of-week field with the date field left wild: {value!r}"
    hh, mm = value.split()[-1].split(":")[:2]
    assert 1 <= int(hh) <= 2, \
        f"the slot must sit in the 01:00-02:59 band that has no nightly writes, no " \
        f"morning measurement and ~no user turns: {value}"
    assert int(mm) <= 15, f"start early: from the latest start the 300 s jitter allows " \
        f"(02:15) the ~3 h 20 m span must still end clear of the 06:00 retrieval eval: " \
        f"{value}"
    assert _systemd_seconds(cp.get("Timer", "RandomizedDelaySec")) == 300, \
        "the jitter is the 300 s every latest-start figure in these two units is " \
        "computed from; widening it moves 02:15 and with it the drain's headroom"


def test_systemd_itself_reads_the_cadence_as_weekly_not_monthly():
    """The unit file's grammar is systemd's, not mine, so the claim that distinguishes
    this round's spec from the trap spec is asked of systemd.

    `systemd-analyze calendar` is a pure parser — it normalizes an expression and prints
    when it would next elapse, touching no manager and no state — and its `Normalized
    form:` line is a deterministic function of the input. Three readings, all asserted:

      * the unit's own value normalizes to itself, day-of-week token intact;
      * `*-*-1 02:10:00` normalizes to `*-*-01 02:10:00`, a day-of-MONTH spec — twelve
        fires a year, on the 1st, of a window that means to run fifty-two;
      * `Mon *-*-* 02:10:00`, the form #2754 lands, keeps `Mon` in the normalized form.

    The `Next elapse:` line the same command prints is deliberately NOT asserted: it is
    a function of the wall clock, and a test that reads it goes red on its own without
    the unit changing. This is the boundary the item's triage measured by hand
    (`Normalized form: *-*-01`, next elapse 2026-11-01, against `Mon 2026-10-19` for the
    day-of-week form) and that a reader of this repo cannot otherwise re-check.
    """
    def normalized(expr: str) -> str:
        proc = subprocess.run(["systemd-analyze", "calendar", expr],
                              capture_output=True, text=True, timeout=30, check=True)
        line = next(l for l in proc.stdout.splitlines() if "Normalized form:" in l)
        return line.split("Normalized form:")[1].strip()

    value = _parse(TIMER).get("Timer", "OnCalendar")
    assert normalized(value) == value == "Mon *-*-* 02:10:00", \
        f"systemd reads the shipped cadence as something other than a Monday: {value!r}"
    assert normalized("*-*-1 02:10:00") == "*-*-01 02:10:00", \
        "the day-of-month form is monthly, and if systemd ever stops saying so this " \
        "node's reasoning needs re-deriving, not just its expected string"
    assert normalized("Mon *-*-* 02:10:00").split()[0] == "Mon"


def _setup_section(heading: str) -> str:
    """One `###` section of SETUP.md, heading to the next heading of the same level."""
    text = SETUP.read_text(encoding="utf-8")
    start = text.index(heading)
    rest = text[start + len(heading):]
    nxt = re.search(r"^### ", rest, re.MULTILINE)
    return rest[:nxt.start()] if nxt else rest


def test_the_cadence_prose_is_weekly_and_carries_the_measured_cost():
    """#2754 clause 3: the prose says what the timer now does, and the number that made
    the cadence worth bounding is quoted as a measurement.

    The old text was right about the hour and wrong about the cadence: `Description=Nightly
    Lloyd persistence-arm window`, and an argument that the slot was chosen because the
    queue "was busy at 02:10 on 18 of the 20 nights the run store covers" — a reason to
    pick 02:10, dressed as a reason to run at 02:10 every day. So three things are pinned
    per surface, and only the surfaces that describe this window:

      1. the `[Unit] Description` calls it weekly, not nightly;
      2. no line in either unit, or in SETUP.md's persistence section, claims THIS window
         is nightly or daily — except as a borrowed job title. `#82 nightly-retrieval-eval`
         and `#92 nightly-iv-metrics` are autonomy job NAMES, and the 06:00 and 05:00
         bounds they set are why the drain has a number instead of a preference; the item
         flagged the first as a pin a careless rewrite would break, and the second is the
         same class in the service's other paragraph. A blanket word ban would have to be
         broken to be satisfiable, so the exemption is these two titles and nothing else:
         the assert quotes the whole line, so a line that smuggles "nightly" in as
         cadence rather than as a name still fails.
      3. the cost that motivated the bound is stated with its figures: 34 min of held
         dispatch per fire (34m42s on 2026-10-13 and 34m01s on 2026-10-14, from
         `journalctl --user -u lloyd-persistence-window.service`), the 3 h 20 m the
         `ExecStart` arithmetic allows as the ceiling above it, and 30 rows / ~190 KB
         (190,226 B and 188,556 B) appended to the git-tracked rows corpus per window.

    Those two hold figures are not interchangeable and the old prose carried only the
    larger one. 12000 s is what the drain plus ten reps could occupy; 34 min is what the
    two fires actually held dispatch. A cadence argued from the ceiling is a cadence
    argued from a budget, and that is how a nightly commitment looked affordable here.

    SETUP.md's enable route is checked again here — with its cadence comment, not just
    its command — because `test_setup_md_carries_the_enable_route` pins the command
    string, and a rewrite that fixed the command's comment to say "daily" while the unit
    fired Mondays would leave that node green.
    """
    desc = _parse(TIMER).get("Unit", "Description")
    assert "nightly" not in desc.lower(), f"the Description still names a cadence: {desc!r}"
    assert re.search(r"(?i)weekly|monday", desc), \
        f"the Description must name the cadence the timer now has: {desc!r}"

    surfaces = {"timer": _comments(TIMER), "service": _comments(SERVICE),
                "SETUP.md": _setup_section("### The persistence-arm window")}
    for label, prose in surfaces.items():
        for line in prose.splitlines():
            if "night" in line.lower() or "daily" in line.lower():
                assert "nightly-retrieval-eval" in line, \
                    f"{label} claims a cadence for this window: {line.strip()}"
        assert "34 min" in prose, f"{label} never states the measured dispatch hold"
        assert "190" in prose, f"{label} never states the tracked bytes per window"
        assert re.search(r"(?i)week|monday", prose), \
            f"{label} still describes some other cadence than the shipped one"
    assert "3 h 20 m" in surfaces["timer"], \
        "the ceiling stays in the timer's own sizing — it is what the 06:00 bound is " \
        "checked against, and the measured 34 min is its subordinate, not its replacement"
    assert "30 rows" in surfaces["SETUP.md"], \
        "the row count is the figure a reader of SETUP.md can act on (retention, #2194's " \
        "denominator); the byte count alone hides that a window is 30 rows of corpus"
    assert re.search(r"enable --now lloyd-persistence-window\.timer\s+#\s*[^\n]*Monday",
                     _setup_section("### The persistence-arm window")), \
        "the enable route's comment must carry the new cadence; the command itself is " \
        "pinned verbatim by test_setup_md_carries_the_enable_route"


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
    """A linked-but-never-enabled timer is the drift #1891 exists to catch, and the
    shape that produced it was `install-services.sh` echoing the `systemctl --user
    enable` lines instead of running them — since #2774 the script executes them
    itself, which is what the route below now automates.
    `scripts/maintenance/check-unit-enabledness.sh` asserts every repo timer answers
    `enabled`. The enable is host state, so this pins the one place
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


# ── #2596: the report a timer-fired window has to leave behind ────────────────
#
# `eval/persistence_arms.py:362-365` writes the dated report only when
# `LLOYD_CANARY_REPORT` is non-blank, and this unit has never set that variable —
# correctly, since the unset is what keeps every rep on the tracked rows file. The
# consequence was that a timer-fired window produced rows and no report: as of the
# round that opened this section, `eval/measurements/injection-canary/` held exactly
# one report, the hand-written `run-2026-10-04-persistence.md`, and `git grep -n
# ExecStartPost -- .` returned nothing, so no systemd surface wrote one either. The fix
# is a second Exec line, and these five nodes pin it clause by clause.
#
# The line crosses a real process boundary in both directions: systemd forks the venv
# python with only the unit's own `Environment=` lines and a cwd of the user's home, and
# the report is written by a CLI, not by this test importing it. So the nodes below
# FORK — reading the rendered table out of the child's stdout rather than trusting a
# substring of the Exec line — because every failure mode here (a `-m` form that cannot
# resolve, a missing `--force` on a second same-day window, a date spelled out where
# systemd has no date specifier) is invisible to a test that only parses text.

REPORT_SCRIPT = "eval/run_injection_canary.py"      # named absolutely by the unit
VENV_PY_SUFFIX = ".venvs/lloyd/bin/python"          # the repo's own interpreter
# #2749 added two columns to this line — the window's raw `rows (attack/control)`
# denominator and its per-arm-key empty-terminal count — and relabelled the five scored
# columns `last rep per key`, because every `n` in them is `summarize`'s key count and a
# 30-row window was reading as `n=2`. #2857 appended two more at the END of the line —
# the same leak and benign-control figures summed over every row of the window, which on
# 2026-10-14 is `42/60` where the collapsed cell reads `3/6` — and left the eight cells
# before them in place, so the `[7]`-index reads of this table still find the
# empty-terminal figure. The constant is the whole line, so the forked-CLI stdout node
# and the write-behind node below both fail if the header and this copy drift.
REPORT_TABLE_HEADER = ("| window | rows (attack/control) | episodes ran "
                       "(last rep per key) | summary leak (95% CI, last rep "
                       "per key) | unattributed (last rep per key) | "
                       "attributed (last rep per key) | benign control "
                       "(last rep per key) | empty-terminal rows per arm "
                       "key | leak (all rows) | benign control (all rows) |")
WRITING_FLAGS = ("--write", "--force")              # dropped to make a run read-only


def _post_lines() -> list[str]:
    """Every `ExecStartPost=` value in the unit, in the order the FILE spells them.

    Read off the raw text, never out of `_parse`. The shipped parser is
    `strict=False`, and under it a repeated key is legal and LAST-WINS: measured on
    `"[Service]\\nExecStartPost=/a x\\nExecStartPost=/b y\\n"`, `strict=True` raises
    `DuplicateOptionError` while `strict=False` answers `'/b y'`. One `cp.get` after
    #2877 appended its commit step would therefore have returned the COMMIT line and
    silently hidden the report line from every node below, turning the four older
    post-start nodes red over a unit that is fine — or, worse, green while reading a
    command systemd never runs. systemd executes every `ExecStartPost` it is given, in
    file order, so file order is what this returns."""
    lines = [l.strip().split("=", 1)[1].strip()
             for l in SERVICE.read_text(encoding="utf-8").splitlines()
             if l.strip().startswith("ExecStartPost=")]
    assert lines, \
        "the unit has no ExecStartPost, so a completed window renders its report to the " \
        "journal and writes nothing: the one report in the tracked dir is still the " \
        "hand-written run-2026-10-04-persistence.md"
    return lines


def _execstartpost() -> str:
    """The REPORT step — the first `ExecStartPost` line, which is the one that must
    stay first: it is what writes the dated report the commit step then commits."""
    return _post_lines()[0]


def _post_argv() -> list[str]:
    """The report step's argv, split with POSIX quoting exactly as `ExecStart` is."""
    return shlex.split(_execstartpost())


def _service_env() -> dict[str, str]:
    """The environment systemd will hand `ExecStartPost`: the unit's own `Environment=`
    lines with `%h` expanded, and nothing else inherited.

    Passing the process nothing but these two variables is the point — a child that
    needs `VIRTUAL_ENV`, a login `PATH` or the caller's cwd is a child that works under
    `pytest` and fails at 02:10 with nobody reading the journal."""
    home = str(Path.home())
    env = {}
    for line in _env_lines(SERVICE):
        key, _, value = line.split("=", 1)[1].partition("=")
        env[key] = value.replace("%h", home)
    return env


#: The script the unit's `ExecStartPost` names, resolved into THIS checkout.
CANARY_SCRIPT = ROOT / "eval" / "run_injection_canary.py"


def _fork_head(argv: list[str]) -> list[str]:
    """The unit's interpreter, with its script token resolved into the tree under test.

    The shipped line says `%h/lloyd/eval/run_injection_canary.py`, so forking `argv`
    verbatim forks whatever checkout is INSTALLED — and a node that does that cannot pin
    a change to the file it exists to test. #2749 is the case: the diff relabelled the
    window table, the constant below carries the new header, and the fork still printed
    the old one because the live tree had not moved. That is not the unit failing, it is
    the fork pointing at the wrong tree — so the interpreter stays exactly as the unit
    spells it (the sibling node pins that it is absolute, executable, and the repo venv
    python, and that no `-m`/`--out` crept in) and only the script resolves into `ROOT`,
    with `LLOYD_HOME` moved with it so the fork reads the corpus this tree tracks.
    """
    assert CANARY_SCRIPT.is_file(), (
        f"{CANARY_SCRIPT} is not the shipped script path — a moved root silently turns "
        "these forks into no-ops, which is how #2397 first broke")
    return [argv[0], str(CANARY_SCRIPT)]


def _fork_env() -> dict[str, str]:
    """`_service_env()` with `LLOYD_HOME` pointed at the tree under test."""
    return {**_service_env(), "LLOYD_HOME": str(ROOT)}


def _utc_today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def test_the_unit_itself_writes_the_dated_report_after_the_window():
    """Clause 1: an `ExecStartPost` whose argv runs the repo's venv python on
    `eval/run_injection_canary.py report --write`, so the dated report is the unit's own
    output and not a side effect of an environment variable it is not allowed to set.

    The two tokens are pinned as things that exist, not as strings: the interpreter is a
    file this box can execute, the script is a file, and both live under the same
    checkout `ExecStart` runs — which is what makes a moved repo root a test failure
    rather than a unit that stops producing reports in silence. And because the unset of
    `LLOYD_CANARY_REPORT` is load-bearing on the rows side, the post line may not name it
    inline either: an `Environment=`-free unit that reaches the same variable through its
    own argv is the same override wearing a different hat."""
    argv = _post_argv()
    assert argv[0].endswith(VENV_PY_SUFFIX), \
        f"the post-start step must run the repo's venv python, got {argv[0]!r}"
    assert os.path.isabs(argv[0]) and Path(argv[0]).is_file() and \
        os.access(argv[0], os.X_OK), f"{argv[0]} is not an executable interpreter"
    assert argv[1].endswith(REPORT_SCRIPT), \
        f"the post-start step must run the canary CLI, got {argv[1:]!r}"
    assert Path(argv[1]).is_file(), f"{argv[1]} does not exist"

    repo_root = str(Path(_exec_argv()[0]).parents[1])     # ExecStart's own checkout
    for token in argv[:2]:
        assert token.startswith(repo_root + os.sep), \
            f"{token!r} is outside the checkout ExecStart runs ({repo_root}), so the " \
            "two Exec lines would report on different trees if either moved"

    assert argv[2] == "report", f"the subcommand must be report, got {argv[:3]!r}"
    assert "--write" in argv, \
        f"without --write the CLI prints and leaves the tree alone: {argv!r}"
    assert "LLOYD_CANARY" not in _execstartpost(), \
        "naming the override inline is an override; the report must come from this line"

    monkey = pytest.MonkeyPatch()
    monkey.delenv(RC.ROWS_ENV_VAR, raising=False)
    monkey.delenv(RC.REPORT_ENV_VAR, raising=False)
    try:
        today = _utc_today()
        assert RC.report_path(today) == RC.OUT_DIR / f"run-{today}{RC.REPORT_SUFFIX}", \
            "the target the flags resolve to is the dated report in the tracked dir"
        assert RC.OUT_DIR.name == "injection-canary", RC.OUT_DIR
        assert RC.report_path(today).parent == RC.OUT_DIR
    finally:
        monkey.undo()


def test_the_post_start_command_resolves_from_the_directory_systemd_starts_it_in():
    """Clause 2: the argv, run for real from a foreign cwd under nothing but the unit's
    own environment, exits 0 and prints the window table.

    systemd starts this unit in the user's home — the unit sets no `WorkingDirectory=` —
    and that is the ground the `-m` form falls through: `python -m
    eval.run_injection_canary report` from outside the checkout dies with
    `ModuleNotFoundError: No module named 'eval'`, while the absolute-script form exits 0
    from anywhere because `OUT_DIR` (`eval/run_injection_canary.py:80`) is derived from
    `LLOYD_HOME`, the script's own resolved parent, not from the cwd. `cwd="/"` here is
    stricter than `%h` on purpose: if it resolves from the filesystem root it resolves
    from home too.

    The writing flags are dropped so the fork only renders — that is what the operator
    contract in `_report_cmd`'s docstring says a bare `report` does — and stdout is read
    as the report it is: the table header and the 2026-10-04 row, the window the item
    says must stay on a line of its own.

    The interpreter token is the unit's own; the script token is resolved into this
    checkout by `_fork_head`, because a fork of whatever tree happens to be installed
    cannot pin a change to the renderer this node reads a header out of."""
    cp = _parse(SERVICE)
    wd = (cp.get("Service", "WorkingDirectory")
          if cp.has_option("Service", "WorkingDirectory") else None)
    argv = _post_argv()
    assert wd or (os.path.isabs(argv[0]) and os.path.isabs(argv[1])), \
        f"neither every path token is absolute nor is WorkingDirectory set: {argv!r}"
    assert not any(t == "-m" for t in argv), \
        "the -m form cannot resolve unless the unit sets WorkingDirectory=%h/lloyd"
    assert "--out" not in argv, "an --out here would be a path the unit invented per run"

    proc = subprocess.run(_fork_head(argv)
                          + [t for t in argv[2:] if t not in WRITING_FLAGS],
                          capture_output=True, text=True, env=_fork_env(),
                          cwd="/", timeout=180)
    assert proc.returncode == 0, \
        f"the post-start argv failed from cwd=/ ({proc.returncode}): " \
        f"{(proc.stderr or proc.stdout)[-400:]}"
    assert REPORT_TABLE_HEADER in proc.stdout, \
        f"no window table on stdout: {proc.stdout[:300]!r}"
    assert "| 2026-10-04 |" in proc.stdout, \
        "the 2026-10-04 window has to stay in the table the unit writes"
    assert "[not written]" in proc.stdout, \
        "with the writing flags dropped the CLI must refuse to touch the tree"
    assert proc.stderr == "", f"the report rendered but complained: {proc.stderr[-300:]}"


def test_the_post_start_carries_force_because_a_same_day_rerun_would_otherwise_refuse(tmp_path):
    """Clause 3: `--force`, no date literal, and a target that is `RC.report_path(<utc
    today>)` inside `RC.OUT_DIR`.

    The flag is not decoration. `_report_cmd` computes the target, sees it exists and
    returns 1 with `[refused] … already exists — pass --force to overwrite`
    (`eval/run_injection_canary.py:1735-1738`, the lines #2749's two new helpers pushed
    the refuse down by), which on a unit whose window actually
    succeeded reads as `Failed with result 'exit-code'` — a second completed window on
    one UTC day would fail the very run that earned the report. So this writes twice to
    one throwaway target and once more with the flag, through the child process, because
    the refusal is a child's exit status and not an exception this suite could catch.

    No date is spelled out, because systemd has no date specifier and `$(date -u +%F)`
    needs a shell this unit's own comment rules out; the dated name comes from
    `report_path()` (`:115-125`), which is the same function that named
    `run-2026-10-04-persistence.md`. Writes go to `tmp_path`, never to the tracked dir."""
    post = _execstartpost()
    argv = _post_argv()
    assert "--force" in argv, f"without --force a second window on one day exits 1: {argv!r}"
    assert not re.search(r"\d{4}-\d{2}-\d{2}", post), \
        f"a date literal in an Exec line cannot be interpolated: {post!r}"
    for shell_ism in ("$(", "${", "`", "%Y", "date "):
        assert shell_ism not in post, \
            f"{shell_ism!r} needs a shell, and this unit execs directly"

    today = _utc_today()
    target = RC.report_path(today)
    assert target == RC.OUT_DIR / f"run-{today}{RC.REPORT_SUFFIX}", target
    assert target.parent == RC.OUT_DIR, \
        "the dated name resolves inside the tracked report dir, beside the 2026-10-04 one"

    # The tracked dir before this node runs a child. What the node is owed about that
    # dir is that its three child writes — all aimed at `tmp_path` — left it exactly as
    # they found it, and the snapshot is how that is measured. The shipped node said it
    # as an absence (`not (RC.OUT_DIR / today).exists()`), which held only while today
    # had no report; `1cdb06bd` (#2397) committed real ones, and from the first day that
    # runs on this checkout after one of them the node went red over a file it had not
    # written. The subject of the node —
    # the `--force` the unit carries, and the same-day refusal it is there for — passed
    # throughout, which is why the claim below is about the writes and not about the
    # calendar.
    tracked_before = {p.name: p.read_bytes() for p in RC.OUT_DIR.iterdir()}

    head, target2 = _fork_head(argv), tmp_path / f"run-{today}{RC.REPORT_SUFFIX}"
    same_day = head + ["report", "--write", "--out", str(target2)]
    writes = [same_day,               # rep 10 finished, first window of the day
              same_day,               # a second one finished: this is the refusal
              same_day + ["--force"]]  # …which is the flag the unit carries
    env, ran = _fork_env(), []
    for i, cmd in enumerate(writes):
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd="/", timeout=180)
        ran.append(proc)
        assert proc.returncode == (0 if i != 1 else 1), \
            f"write {i + 1} of the same-day pair returned {proc.returncode}: " \
            f"{(proc.stderr or proc.stdout)[-300:]}"
    assert "already exists" in ran[1].stderr and "--force" in ran[1].stderr, \
        f"the second window was refused for a different reason: {ran[1].stderr[-300:]}"
    assert target2.is_file() and REPORT_TABLE_HEADER in target2.read_text(encoding="utf-8"), \
        "the third write, the one shaped like the unit's, overwrote and left a table"
    assert {p.name: p.read_bytes() for p in RC.OUT_DIR.iterdir()} == tracked_before, \
        "a write aimed at tmp_path touched the tracked measurement dir: the unit's own " \
        "target is that dir, and a node that dirties it cannot tell its own artifact " \
        "from a real window"


def test_the_unit_reaches_neither_override_on_any_surface_systemd_reads():
    """Clause 4: no environment surface — `Environment=`, `EnvironmentFile=` or
    `PassEnvironment=` — names either override, and `ExecStart`, `Type=oneshot` and
    `TimeoutStartSec=4h` are exactly as shipped.

    The two extra keys are the hole #2428's post-review finding left open: an INI read
    keeps only the last `Environment=`, so the older node could not see a second one, and
    neither `EnvironmentFile=` nor `PassEnvironment=` was named at all — either would
    carry `LLOYD_CANARY_REPORT` into the window without a single `Environment=` line
    changing. They are refused outright rather than searched for the two names, because
    a file systemd reads and this test's `_env_lines` does not is precisely the kind of
    gap that outlives a name-based blocklist.

    The preserved line is pinned by its shipped suffix and the timeout by its parsed
    seconds, so the new Exec line cannot pay for itself by quietly shortening either."""
    text = SERVICE.read_text(encoding="utf-8")
    surfaces = [l.strip() for l in text.splitlines()
                if l.strip().startswith(("Environment=", "EnvironmentFile=",
                                         "PassEnvironment="))]
    for override in (RC.ROWS_ENV_VAR, RC.REPORT_ENV_VAR):
        assert not [l for l in surfaces if override in l], \
            f"an environment surface names the bench's own default: {override}"
    assert not [l for l in surfaces if l.startswith(("EnvironmentFile=",
                                                    "PassEnvironment="))], \
        "EnvironmentFile=/PassEnvironment= would hand the window an override this node's " \
        "Environment= scan never sees (#2428's post-review finding)"

    for line in (_execstart(), *_post_lines()):
        assert "LLOYD_CANARY" not in line, f"inline assignment: {line!r}"
        for pool_ism in ("curl", "wget", "enqueue", "/api/", "bash -c", "sh -c",
                         "python -c"):
            assert pool_ism not in line, \
                f"the executable surface reaches the pool ({pool_ism!r}): {line!r}"

    cp = _parse(SERVICE)
    assert _execstart().endswith(f"/{WINDOW_SCRIPT} {REPS} {DRAIN_FLAG} {DRAIN_WAIT_S}"), \
        f"ExecStart changed: {_execstart()!r}"
    assert cp.get("Service", "Type") == "oneshot"
    assert _systemd_seconds(cp.get("Service", "TimeoutStartSec")) == 4 * 3600, \
        "TimeoutStartSec=4h is the window's hang guard; the report render costs 0.07 s"


def test_the_service_comment_describes_the_report_write_and_its_success_only_trigger():
    """Clause 5: the unit's own prose says where the report comes from, what the numbers
    are, and when the step runs.

    Three claims, because each is a thing a later reader could otherwise get wrong from
    the line alone. That it comes from `ExecStartPost` and never from
    `LLOYD_CANARY_REPORT` — the unit above states the unset is deliberate, so a report
    appearing by some other route would be the bug. That every figure is `summarize`'s and
    `_wilson`'s output unchanged, which is what keeps the 2026-10-04 window a row of its
    own with its own denominator (`persistence_windows` groups on the row's UTC day)
    rather than a number merged into the new window's. And that systemd runs
    `ExecStartPost` only after a successful `ExecStart`, so the refused drain this unit
    has actually produced — exit 2 — wrote no empty report and never will."""
    prose = _comments(SERVICE)
    assert "ExecStartPost" in prose, \
        "the unit must name the line that writes the report"
    assert "LLOYD_CANARY_REPORT" in prose and "never from" in prose, \
        "…and say the env var is not where it comes from"
    assert "summarize" in prose and "_wilson" in prose and "unchanged" in prose, \
        "the table's provenance has to be stated, not implied by a flag"
    assert "only after" in prose and "ExecStart" in prose, \
        "the success-only trigger is what makes a failed window write nothing"
    assert "exit 2" in prose, \
        "the refused-drain case is named by the status the one fire this unit had returned"
    assert "--force" in prose, \
        "…and the comment must say why overwriting a same-day file is the right behaviour"


#: The script the second `ExecStartPost` runs (#2877), and the ONE directory it may
#: stage — repeated here rather than imported so the node fails if the unit and the
#: script disagree about which corpus is theirs.
COMMIT_SCRIPT = "scripts/maintenance/commit_injection_canary_corpus.py"
COMMIT_PATHSPEC = "eval/measurements/injection-canary/"


def _commit_step_module():
    """The script the commit step names, imported by path from the tree under test.

    The seam this crosses is the one a grep cannot see: the unit's argv on one side, and
    the constants that decide what that argv will commit on the other. `scripts/` is not
    a package, so this is the same `spec_from_file_location` shape
    `tests/test_corpus_shape.py` uses for `scripts/maintenance/corpus_shape.py`.
    """
    import importlib.util
    script = ROOT / COMMIT_SCRIPT
    assert script.is_file(), \
        f"{script} is missing, so the unit's second ExecStartPost exits 2 and fails a " \
        "window that wrote its report correctly"
    spec = importlib.util.spec_from_file_location("commit_injection_canary_corpus",
                                                  script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_unit_commits_the_corpus_it_writes_after_rendering_the_report():
    """#2877 clause 1: a SECOND `ExecStartPost` that execs the repo's venv python on
    `scripts/maintenance/commit_injection_canary_corpus.py` by absolute path, after the
    unchanged `report --write --force` line — with BOTH lines read in file order from the
    raw text, because one `cp.get` under the shipped `strict=False` parser returns only
    the last duplicate (measured `'/b y'`).

    Why the order is load-bearing rather than tidy: `report --write` appends the fire's
    rows to `eval/measurements/injection-canary/rows.jsonl` and lands
    `run-<UTC date>-persistence.md` beside them, and the commit step records exactly that
    output. Run first, it would commit the PREVIOUS fire and leave this one's rows in
    `git status --porcelain` for another week — the failure this item exists to end.

    Why `is_file()` is checked on `ROOT / COMMIT_SCRIPT` and not on `argv[1]`: the unit
    spells its paths absolutely for the live checkout, so a round's worktree — where
    this node runs before the script has landed — holds the file at a different path.
    `_fork_head` reasons about the same trap for the report script; asserting the live
    path here would test the deployment instead of the diff. What IS pinned about
    `argv[1]` is that it is absolute, inside the same checkout `ExecStart` runs, ends in
    the script's repo-relative path, and equals `step.LIVE_REPO + "/" + COMMIT_SCRIPT` —
    i.e. the line and the script's own default agree on one tree.
    """
    import re
    raw = SERVICE.read_text(encoding="utf-8")
    lines = _post_lines()
    assert lines == [s.strip() for s in re.findall(r"(?m)^ExecStartPost=(.*)$", raw)], \
        "the helper is not reading the raw ExecStartPost lines in file order, and that " \
        "is the only way both of them are visible: `cp.get` with strict=False returns " \
        "the last duplicate alone, hiding the report line behind the commit line"
    assert len(lines) == 2, \
        f"expected the report step then the corpus commit step, got {len(lines)} " \
        f"ExecStartPost line(s): {lines!r}"
    assert lines[0].endswith("report --write --force"), \
        f"the report step must stay first, unchanged: {lines[0]!r}"

    step = _commit_step_module()
    argv = shlex.split(lines[1])
    assert argv[0].endswith(VENV_PY_SUFFIX) and os.path.isabs(argv[0]) \
        and Path(argv[0]).is_file() and os.access(argv[0], os.X_OK), \
        f"the commit step must run the repo's venv python, got {argv[0]!r}"
    repo_root = str(Path(_exec_argv()[0]).parents[1])         # ExecStart's own checkout
    assert argv[1] == step.LIVE_REPO + "/" + COMMIT_SCRIPT, \
        f"the line runs {argv[1]!r} but the script defaults to committing in " \
        f"{step.LIVE_REPO!r}: two trees, so the step commits nothing it wrote"
    for token in argv[:2]:
        assert token.startswith(repo_root + os.sep), \
            f"{token!r} is outside the checkout ExecStart runs ({repo_root})"
    assert step.LIVE_REPO == repo_root, \
        f"the script would commit {step.LIVE_REPO!r}, not {repo_root!r}"
    assert len(argv) == 2, \
        f"a token after the script would be a pathspec the unit invented instead of the " \
        f"script's own: {argv!r}"

    for shell_ism in ("$(", "${", "`", "sh -c", "bash -c", "python -c", "date "):
        assert shell_ism not in lines[1], \
            f"{shell_ism!r} needs a shell, and this unit execs directly"
    assert "%" not in lines[1], \
        "a `%` on this line is a systemd specifier, and the dated filename is the " \
        "script's own to derive from datetime.now(timezone.utc)"
    assert not re.search(r"\d{4}-\d{2}-\d{2}", lines[1]), \
        f"a date literal cannot be interpolated by systemd: {lines[1]!r}"

    assert step.CORPUS_PATHSPEC == COMMIT_PATHSPEC, \
        f"the step stages {step.CORPUS_PATHSPEC!r}, not {COMMIT_PATHSPEC!r}"
    assert step.CORPUS_PATHSPEC == f"{RC.OUT_DIR.relative_to(ROOT)}/", \
        f"the committed directory {step.CORPUS_PATHSPEC!r} is not the report's own " \
        f"output dir {RC.OUT_DIR}, so the fire's rows would stay uncommitted"


def test_the_timer_says_the_canary_store_counts_the_corpus_without_bounding_it():
    """#2877 clause 5: the timer's retention comment states what the sweep actually
    does, and nothing in either unit still claims the store is missing.

    The shipped sentence — ``(`scripts/groundskeeper/retention-sweep.py` has no
    ``injection-canary` entry)`` — was true when it was written and is false now: store
    18 was added by 13a8c238 (#2873, landing #2862's code half), and
    `sweep_injection_canary` is in that file. A comment that names a missing guard is
    worse than no comment here, because it is the reason a later reader keeps the cadence
    bound and does not go looking for the store: the true statement is narrower and has
    to be said exactly — the store COUNTS the corpus and does not BOUND it, because
    `apply` is accepted and ignored on purpose and deleting a git-tracked corpus is a
    repository write.

    The claims are checked against the sweep's own source, not just against the prose,
    because that is the boundary the sentence crosses: a comment can only be as true as
    the function it describes.
    """
    prose = _comments(TIMER)
    for label, text in (("timer", _comments(TIMER)), ("service", _comments(SERVICE))):
        for stale in ("has no `injection-canary` entry", "no retention store covering"):
            offender = [l.strip() for l in text.splitlines() if stale in l]
            assert not offender, \
                f"the {label} unit still claims the retention coverage is absent: " \
                f"{offender}"

    assert "sweep_injection_canary" in prose, \
        "the comment must name the store that does cover the corpus"
    assert "scripts/groundskeeper/retention-sweep.py" in prose, \
        "…and name the file it lives in"
    assert "counts it" in prose and "without bounding" in prose, \
        "the comment must say the store counts the corpus without bounding it, which is " \
        "the narrower true claim the cadence bound rests on now"
    assert "accepted and ignored on purpose" in prose and "git-tracked" in prose, \
        "…and why: `apply` is ignored because the corpus is git-tracked"
    assert "retention ruling" in prose, \
        "the owed retention ruling for the growing corpus must stay owed in the prose"

    sweep = (ROOT / "scripts" / "groundskeeper" / "retention-sweep.py").read_text(
        encoding="utf-8")
    assert "def sweep_injection_canary(" in sweep, \
        "the function the comment names is not in the file it names"
    assert "accepted and ignored, on purpose" in sweep, \
        "the comment repeats the store's own `apply` contract, so the source has to " \
        "still carry it"
    assert "reports without bounding" in sweep, \
        "the comment calls this the second store that reports without bounding; the " \
        "sweep's own docstring is where that is stated"
