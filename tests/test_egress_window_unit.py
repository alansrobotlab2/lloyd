"""#2555: the egress-arms window has an executor, and it is this pair of units.

`eval/egress_arms.py` landed complete and tested (b79835a4: the driver, the wrapper,
`tests/test_run_egress_arms.py`), and nothing on the box could fire it. Two reasons, one of
them structural:

* **A pool-dispatched window is impossible by construction.** The driver will not start its
  episodes while any queue row is in flight, and it learns that from `depth_by_source()`
  (`workers/queue.py:1229` — `SELECT source, state, COUNT(*) as n FROM queue GROUP BY
  source, state`) with no clause excluding its own row. A job dispatched to run the window
  IS the row it waits for, so it burns the whole drain and exits 2.
  `lloyd-persistence-window.service`'s first timer run is that failure on the record:
  2026-10-11T02:26:39-07:00, last poll `{'in_flight': 1, 'autocode:running': 1}`,
  `status=2/FAILURE`, "Consumed 1.236s CPU over 15min 2.499s wall clock" — 15 minutes being
  that unit's own 900 s drain, spent waiting on itself.
* **Nothing else fires it either.** No egress unit existed in `agent-services/systemd/` or
  `~/.config/systemd/user/` when this round opened, and a pytest round cannot run the
  window: `tests/conftest.py` refuses production subprocesses, and the window needs the
  live engine, the shared aggregator and the sandbox.

So the unit pair is the whole deliverable, and what is under test is its shape: what it
execs, what it refuses to set, the drain that outlasts a measured automod round, the run
root that keeps the ON arm's deny rows readable after the post-run leak check, the timeout
that keeps systemd from SIGTERMing it mid-drain, and the arming route that stops it
shipping linked-but-unarmed the way its sibling did (#2537).

Three nodes here assert against the unit files' own COMMENTS, not only their settings — the
move `tests/test_persistence_window_unit.py` makes at its timeout and timer nodes. This
unit's numbers are its whole argument (a 5 h timeout, a 3,600 s drain, a 06:23 hour), and a
setting whose comment no longer justifies it is exactly the drift that left the precedent
unsupportable: `lloyd-persistence-window.service` still cites `workers/queue.py:1103-1111`,
which is the boot-time interrupted-run recovery sweep now, while the query it means moved to
:1229. Whether to spend a round repairing that sibling comment is owed-check's ruling
(#2555 owed 5), not this round's.

What is NOT tested, and is owed instead (see the item): arming the timer and proving
`ActiveState` beside `is-enabled`; the first real run and the >=36 arm-stamped rows it
must write; the per-arm Wilson pair and the leak witness in `architecture/harness.md`. All
of those need the live engine and a `systemctl --user start`, which is a service-control
verb no round may issue.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UNIT_DIR = REPO / "agent-services" / "systemd"
SERVICE = UNIT_DIR / "lloyd-egress-window.service"
TIMER = UNIT_DIR / "lloyd-egress-window.timer"
SIBLING_SERVICE = UNIT_DIR / "lloyd-persistence-window.service"
SIBLING_TIMER = UNIT_DIR / "lloyd-persistence-window.timer"
SETUP = REPO / "SETUP.md"
INSTALLER = REPO / "agent-services" / "setup" / "install-services.sh"
DRIVER = REPO / "eval" / "egress_arms.py"
WRAPPER = REPO / "eval" / "run_egress_arms.sh"
SCENARIOS = REPO / "eval" / "injection_canary" / "scenarios.yaml"
QUEUE = REPO / "workers" / "queue.py"
ARMS_PY = REPO / "eval" / "persistence_arms.py"
CANARY = REPO / "eval" / "run_injection_canary.py"

# The window's own numbers, restated here so a test fails when the unit and its prose
# disagree — that disagreement is the drift this file exists to catch.
DRAIN_S = 3600          # the explicit --drain-wait
TIMEOUT_S = 5 * 3600    # TimeoutStartSec
EPISODE_BUDGET_S = 300  # the ceiling the unit's comment states per episode
ON_PORT = "8599"        # the documented example port, not a live one
ARM_ROOT = "/home/alansrobotlab/lloyd-data/egress-arms"


def _parse(text: str) -> dict[str, list[tuple[str, str]]]:
    """An ini view of a unit file: (key, value) pairs per section, comments dropped.

    Re-declared rather than imported from the sibling test file: pytest's rootdir-only
    conftest makes cross-test-file imports land as `tests.foo`, which the repo's
    module-import guard refuses.
    """
    out: dict[str, list[tuple[str, str]]] = {"_": []}
    sect = "_"
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            sect = line[1:-1]
            out.setdefault(sect, [])
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[sect].append((k.strip(), v.strip()))
    return out


def _comments(path: Path) -> str:
    return "\n".join(l.lstrip("#").strip() for l in path.read_text(encoding="utf-8").splitlines()
                     if l.lstrip().startswith("#"))


def _secs(value: str) -> int:
    """systemd's time syntax, in the forms these units use: `3600`, `600`, `600s`, `5h`."""
    v = value.strip()
    if v.isdigit():
        return int(v)
    m = re.fullmatch(r"(\d+)\s*(s|sec|secs|second|seconds|min|mins|minute|minutes|h|hr|hrs"
                     r"|hour|hours)?", v, re.I)
    assert m, f"unparsable systemd duration {value!r}"
    n, unit = int(m.group(1)), (m.group(2) or "s").lower()
    return n * (1 if unit.startswith("s") else 60 if unit.startswith("m") else 3600)


def _execstart(parsed) -> str:
    return next((v for k, v in parsed["Service"] if k == "ExecStart"), "")


def _env_keys(parsed) -> dict[str, str]:
    return {v.split("=", 1)[0]: v.split("=", 1)[1]
            for k, v in parsed["Service"] if k == "Environment" and "=" in v}


def _scenarios() -> list[str]:
    return re.findall(r"^  - key: (\S+)", SCENARIOS.read_text(encoding="utf-8"), re.M)


# ── clause 1: the unit exists, is a oneshot, execs the window, and sets neither flag ───

def test_the_unit_pair_is_tracked_and_the_service_is_a_oneshot():
    """#2555 clause 1(a): the executor exists in the repo, and it runs once per fire.

    `git ls-files`, not `exists()`: an untracked unit is a scratch file, and it is exactly
    what `install-services.sh` links nothing for. `ConditionPathExists` on the wrapper is
    the sibling's discipline — a box without the eval tree skips instead of failing at exec
    time — and `RemainAfterExit=no` is what makes `oneshot` mean "a run", not "a daemon".
    """
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files",
                              str(SERVICE.relative_to(REPO)), str(TIMER.relative_to(REPO))],
                             capture_output=True, text=True).stdout.split()
    assert len(tracked) == 2, tracked
    p = _parse(SERVICE.read_text(encoding="utf-8"))
    assert dict(p["Service"]).get("Type") == "oneshot", p["Service"]
    assert dict(p["Service"]).get("RemainAfterExit") == "no", p["Service"]
    assert dict(p["Unit"]).get("ConditionPathExists") == "/home/alansrobotlab/lloyd/eval/run_egress_arms.sh"


def test_execstart_is_a_direct_exec_of_the_tracked_wrapper_at_one_rep_both_arms():
    """#2555 clause 1(b): the command line, pinned argument by argument.

    `-` prefix, direct exec, absolute production path: the same three checks the sibling
    node `test_the_execstart_is_a_direct_exec_of_the_script_and_not_an_enqueue` applies,
    because the queue refuses `bash -c` for the same reason it refused that one. The script
    must be mode 0o755 AND `100755` in the index: `ExecStart` execs without a shell, and
    `install-services.sh` has a rail of its own for a unit whose ExecTarget is not
    executable.

    `1` is one rep, and the ABSENCE of `--arms` is what makes it both: the driver's default
    is the pair, so naming one arm would ship the single-sided fragment triage ruled
    unpublishable as a rate. The port is the documented 8599, which the driver refuses to
    run the ON arm without.
    """
    assert (WRAPPER.stat().st_mode & 0o777) == 0o755, oct(WRAPPER.stat().st_mode & 0o777)
    listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "-s",
                             str(WRAPPER.relative_to(REPO))],
                            capture_output=True, text=True).stdout.split("\n")[0].split()
    assert listed and listed[0] == "100755", listed

    arg = _execstart(_parse(SERVICE.read_text(encoding="utf-8")))
    # No leading `-`: systemd would read it as "ignore a non-zero exec exit", and for THIS
    # window the exit code is the signal — exit 2 is the drain refusal the sibling died on,
    # exit 1 is an arm that died mid-window. Swallowing either leaves a daily journal that
    # reads green with no rows behind it, which is the failure this item exists to end.
    assert not arg.startswith("-"), (
        f"ExecStart={arg!r} carries systemd's exec-failure-ignore prefix")
    argv = arg.split()
    assert argv[0] == "/home/alansrobotlab/lloyd/eval/run_egress_arms.sh", \
        f"ExecTarget must be the absolute production path: {argv[0]}"
    assert Path(argv[0]).name == "run_egress_arms.sh"
    assert argv[1] == "1", f"one rep, as the published sample says: {argv}"
    assert "--arms" not in argv, f"naming one arm would drop the comparison: {argv}"
    assert argv[argv.index("--on-port") + 1] == ON_PORT, argv


def test_the_unit_sets_neither_egress_override_so_the_arms_stay_the_driver_forked():
    """#2555 clauses 1(c): no enforce flag, no rows override, and only two keys total.

    `LLOYD_EGRESS_ENFORCE` is the thing UNDER test and it belongs only in the child: the
    module docstring (:14-17) says the ON arm's MCP subprocess is launched with the flag in
    ITS environment and the OFF arm's without it, and `start_on_arm_aggregator()`
    (eval/egress_arms.py:194) is where that scoping lives. A unit-level `Environment=` would put enforcement on the parent that supervises
    the OFF arm too; both arms would come back enforcing and the pair would read as "egress
    enforcement changed nothing" — the false negative in the shape of a clean result.

    `LLOYD_CANARY_ROWS` stays UNSET so the 36 armed rows land on the tracked corpus the
    canary `check` and `record` read. `run_root()` reads that same variable
    (`eval/egress_arms.py:154`), so setting it would move the rows beside the scratch root
    and the window would publish nothing anyone reads.

    `LLOYD_DATA` is absent for the mirror-image reason, and it is the one an editor is most
    likely to "helpfully" add: the PARENT must read the live store, because that is what the
    drain rule and the post-run leak witness both ask about. Setting it at unit level points
    both at the run root's empty store, which is how a window reports "egress was never
    blocked" on a night the ON arm denied plenty.

    The two keys the unit DOES set must be the only two `Environment=` lines: this pair is
    the window's only executor, so any override added later has to come through this node
    and say which arm it breaks.
    """
    parsed = _parse(SERVICE.read_text(encoding="utf-8"))
    env = _env_keys(parsed)
    assert sorted(env) == ["LLOYD_EGRESS_ARMS_ROOT", "LLOYD_PYTHON"], sorted(env)
    for forbidden in ("LLOYD_EGRESS_ENFORCE", "LLOYD_CANARY_ROWS", "LLOYD_DATA"):
        assert forbidden not in env, sorted(env)
        assert not re.search(rf"^\s*{forbidden}\s*=", SERVICE.read_text(encoding="utf-8"),
                             re.M), f"{forbidden} must not be assigned at unit level"

    arms = DRIVER.read_text(encoding="utf-8")
    assert "LLOYD_EGRESS_ENFORCE" in arms, "the flag the unit refuses to set must live in the driver"
    launch = re.search(r"async def start_on_arm_aggregator\(.*?\n(?=\nasync def |\ndef |\nclass )",
                       arms, re.S)
    assert launch is not None, (
        "the function that forks the ON arm's aggregator moved; re-find it before trusting "
        "this node — the flag's scope IS the clause")
    body = launch.group(0)
    assert "env = dict(os.environ)" in body and 'env[ARM_ENV_VAR] = "1"' in body, body


# ── clause 2: the run root is a directory the unit creates, not the live data root ─────

def test_the_run_root_is_a_fixed_dir_under_lloyd_data_created_before_the_window():
    """#2555 clause 2: `run_root()` neither mkdirs nor stamps a date, so the unit must do both.

    `eval/egress_arms.py:146-157` returns the `$LLOYD_EGRESS_ARMS_ROOT` value VERBATIM — no
    mkdir, no date — and `start_on_arm_aggregator()` hands that same path to the child as
    `LLOYD_DATA`
    (:207). A root of its own is what makes the leak witness leak-proof: the child's
    `workers.db` is a file inside it, so the ON arm's `decision=deny` rows for
    `collect.off-allow-list.invalid` are still readable when the post-run witness asks the
    LIVE aggregator the same question (#2435 published that pair: live store 0 deny / 0 rows
    for that host while the run root's own store reads >=1 deny; sharing the live root would
    let this window's own rows fake the second half).

    Fixed rather than dated, because a static `Environment=` line cannot carry a date and
    `bash -c` is the ExecStart form this directory rejects; the date belongs on the published
    report. Hence `ExecStartPre=/usr/bin/mkdir -p`, which is the mkdir the driver does not
    do — and this node re-reads `run_root()`'s own source to make sure that stays true,
    because the moment `run_root()` grows a mkdir the unit's `ExecStartPre` is dead weight
    and the moment it grows a date stamp this path stops being the one the child writes to.
    """
    parsed = _parse(SERVICE.read_text(encoding="utf-8"))
    root = _env_keys(parsed)["LLOYD_EGRESS_ARMS_ROOT"]
    assert root == ARM_ROOT, root
    assert root.startswith("/home/alansrobotlab/lloyd-data/"), root
    assert root != "/home/alansrobotlab/lloyd-data", \
        "the run root IS the live data root; the child would write into it"
    assert [v for k, v in parsed["Service"] if k == "ExecStartPre"] == \
        [f"/usr/bin/mkdir -p {root}"], parsed["Service"]

    src = DRIVER.read_text(encoding="utf-8")
    ast.parse(src, str(DRIVER))
    fn = next(n for n in ast.walk(ast.parse(src, str(DRIVER)))
              if isinstance(n, ast.FunctionDef) and n.name == "run_root")
    body = ast.get_source_segment(src, fn) or ""
    assert "mkdir" not in body and "strftime" not in body, (
        "run_root() grew mkdir/date logic, so this unit's ExecStartPre is redundant or its "
        "fixed path is wrong:\n" + body)
    assert re.search(r"DATA_ROOT_ENV\]\s*=\s*str\(root\)|DATA_ROOT_ENV\] = str\(root\)", src), \
        "the launcher no longer hands the run root to the child as LLOYD_DATA"


def test_the_wrapper_forwards_rep_and_flags_and_refuses_an_empty_rep():
    """#2555 clause 1(b), across the boundary the unit actually crosses.

    The unit execs the SHELL WRAPPER, not the driver, so arguments the wrapper drops are a
    unit that runs the wrong thing and still exits 0. Parsed from the script text rather
    than executed: running it here would launch a real MCP subprocess and hold the primary,
    which is precisely what conftest exists to refuse in a pytest round.
    """
    body = WRAPPER.read_text(encoding="utf-8")
    assert body.startswith("#!/usr/bin/env bash")
    assert '"$@"' in body, "the unit's --on-port/--drain-wait must reach the driver"
    assert '"$1"' in body, "the rep must be forwarded"
    # The empty-rep refusal is the TRACKED wrapper's job, so the tracked file is what gets
    # read and run here — the forwarding stub above only exists to keep these assertions
    # honest about argv. The guard is numeric, not `[ -z ]`: `[[ "$N" =~ ^[0-9]+$ ]]` plus
    # `10#$N < 1`, which refuses "", "0" and "abc" alike.
    real = WRAPPER.read_text(encoding="utf-8")
    assert '[[ "$N" =~ ^[0-9]+$ ]]' in real and "10#$N < 1" in real, \
        "the wrapper's rep guard must refuse a non-positive or non-numeric rep"
    for bad in ("", "0", "abc"):
        rc = subprocess.run([str(WRAPPER), bad, "--on-port", "1"],
                            capture_output=True, text=True, timeout=30)
        assert rc.returncode != 0, (
            f"rep {bad!r} was accepted: rc={rc.returncode} "
            f"err={(rc.stderr or rc.stdout)[:140]!r}")
    assert "egress_arms.py" in body, "the wrapper must exec the driver"
    # The <=8-turn bound is not a wrapper flag: it is the canary runner's own `--max-turns`
    # default, which the driver calls with. Pinning it here keeps the sample the published
    # rate is drawn from bounded at the figure the report states.
    cr = CANARY.read_text(encoding="utf-8")
    assert re.search(r'add_argument\("--max-turns", type=int, default=8\)', cr), \
        "the per-episode turn bound moved off the canary runner's default"


# ── clause 3: the timeout, with its arithmetic stated ─────────────────────────────────

def test_the_timeout_outlives_the_drain_plus_the_engine_work_and_states_the_sum():
    """#2555 clause 3: `TimeoutStartSec=` beats drain + canary work, and the comment shows the math.

    The user manager's default SIGTERMs a starting unit at 90 s, and this window spends its
    first 15 s in a drain poll — the sibling's first attempt died at 931 s, once its own
    900 s drain had run out. A unit that states a timeout but not the arithmetic behind it
    leaves the next reader to guess which term has to grow when the sample size changes, so
    this node re-derives the SUM from the unit's own two settings and requires both terms
    and the default to appear in the prose.
    """
    parsed = _parse(SERVICE.read_text(encoding="utf-8"))
    tvals = [v for k, v in parsed["Service"] if k == "TimeoutStartSec"]
    assert len(tvals) == 1, parsed["Service"]
    timeout = _secs(tvals[0])
    assert timeout == TIMEOUT_S, tvals[0]

    arg = _execstart(parsed)
    drain = _secs(arg.split("--drain-wait")[1].split()[0])
    assert drain == DRAIN_S, drain
    assert timeout > drain, (timeout, drain)

    n = len(_scenarios())
    assert n == 18, n
    assert timeout >= drain + 2 * n * EPISODE_BUDGET_S, (
        f"{timeout} s must exceed the drain ({drain}) plus 2 arms x {n} scenarios x "
        f"{EPISODE_BUDGET_S} s")

    c = _comments(SERVICE)
    assert re.search(rf"^TimeoutStartSec={re.escape(tvals[0])}\s*$",
                     SERVICE.read_text(encoding="utf-8"), re.M)
    assert str(drain) in c, f"the drain term must appear in the comment: {drain}"
    assert re.search(r"2 arms x 18 scenarios", c), \
        "the comment must name both multipliers: 2 arms and the 18 shipped scenarios"
    assert str(EPISODE_BUDGET_S) in c, "the per-episode budget must be stated, not implied"
    assert re.search(r"\b90 s\b", c), \
        "the default the unit outlives has to be named or the number is arbitrary"
    assert "39.4" in c, "the measured worst episode the 300 s ceiling is checked against"


# ── clause 4: the timer ───────────────────────────────────────────────────────────────

def test_the_timer_fires_once_a_day_with_catchup_and_bounded_jitter():
    """#2555 clause 4(a): `OnCalendar=`, `Persistent=true`, `RandomizedDelaySec=`.

    Once a day: one rep x 18 scenarios x 2 arms is the sample the rate pair is published
    from, and the primary is the scarce resource. `Persistent=true` catches a missed slot;
    the jitter keeps 36 episodes off the same contention pattern every night, bounded so
    the real start stays inside the same quarter-hour the comment reasons about.
    """
    d = dict(_parse(TIMER.read_text(encoding="utf-8"))["Timer"])
    cal = d["OnCalendar"]
    assert re.fullmatch(r"\*-\*-\* \d{2}:\d{2}:00", cal), cal
    assert d["Persistent"] == "true"
    assert _secs(d["RandomizedDelaySec"]) <= 300, d["RandomizedDelaySec"]
    assert dict(_parse(TIMER.read_text(encoding="utf-8"))["Install"]).get("WantedBy") \
        == "timers.target"


def test_the_timer_hour_is_after_every_named_fire_and_clear_of_the_sibling_ceiling():
    """#2555 clause 4(b): the hour justified against the other repo timers, read from their files.

    `06:23` is later than every other daily calendar in `agent-services/systemd/` (02:10
    persistence, 04:17 cert-renew, 04:45 qmd-cleanup) and clears the persistence window's
    own stated ceiling: that unit bounds its run with `TimeoutStartSec=4h` from a 02:10
    start, so 06:10 is the latest it can still hold the engine, and this window starts 13
    minutes past that. The minute is one no repo timer fires on, and the jitter ceiling
    (600 s) keeps it inside the same quarter-hour.

    Derived from the unit files rather than transcribed, because a comment that reads a
    sibling's fire time and ceiling out of the sibling is a comment that stays true when the
    sibling moves — and this directory's fire times have moved before.
    """
    d = dict(_parse(TIMER.read_text(encoding="utf-8"))["Timer"])
    hh, mm = [int(x) for x in d["OnCalendar"].split()[1].split(":")[:2]]

    others = {}
    for t in sorted(UNIT_DIR.glob("*.timer")):
        if t == TIMER:
            continue
        for k, v in _parse(t.read_text(encoding="utf-8")).get("Timer", []):
            # A daily calendar, and only a daily one: the date field is all wildcards.
            # #2754 moved the persistence window to `Mon *-*-* 02:10:00`, and since
            # systemd's grammar puts the day-of-week FIRST, that unit is no longer a
            # daily fire and no longer belongs in this set — it is covered below, by the
            # sibling's ceiling, which is the stronger statement. The floor drops with
            # it, 3 -> 2: the count is here to catch the family disappearing (a node
            # comparing one fire against nothing still passes its comparisons), and two
            # daily fires are what the directory has now.
            if k == "OnCalendar" and re.fullmatch(r"\*-\*-\* \d{2}:\d{2}:\d{2}", v):
                others[t.name] = v.split()[-1]
    assert len(others) >= 2, others
    assert SIBLING_TIMER.name not in others, \
        "the persistence window is weekly since #2754; if it is back in the daily set " \
        "the cadence changed and its hour has to re-appear in the comparisons below"
    for name, fire in others.items():
        oh, om = [int(x) for x in fire.split(":")[:2]]
        assert (hh, mm) > (oh, om), f"{TIMER.name} at {hh:02d}:{mm:02d} precedes {name} at {fire}"
        assert mm != om or hh != oh, f"{name} fires on the same minute"

    # `[-1]`, not `[1]`: with the day-of-week prefix #2754 added, index 1 is the date
    # field `*-*-*`, and `int("*")` raised ValueError out of this node — a cadence change
    # in another unit arriving as an error here, with nothing in the message saying which
    # field it tripped on.
    sh, sm = [int(x) for x in dict(_parse(SIBLING_TIMER.read_text(encoding="utf-8"))["Timer"])
              ["OnCalendar"].split()[-1].split(":")[:2]]
    sib_to = _secs(dict(_parse(SIBLING_SERVICE.read_text(encoding="utf-8"))["Service"])
                   ["TimeoutStartSec"])
    assert hh * 60 + mm >= sh * 60 + sm + sib_to // 60, (
        f"{hh:02d}:{mm:02d} falls inside the sibling's {sib_to // 3600} h ceiling from "
        f"{sh:02d}:{sm:02d}: the two windows' engine work would overlap")
    assert re.search(r"06:10|0?6:10", _comments(TIMER)), \
        "the ceiling this hour is timed against must appear in the comment"
    jitter = _secs(d["RandomizedDelaySec"])
    assert hh * 60 + mm + jitter // 60 < hh * 60 + 30, \
        "the jitter must keep the real start inside the same quarter-hour the comment names"


def test_the_timer_names_the_live_pool_query_and_refuses_to_claim_the_hour_avoids_rounds():
    """#2555 clause 4(c): the citation points at the query that exists, and the false claim is absent.

    The item, and the sibling unit's own header, both cite `workers/queue.py:1103-1111` for
    the pool-status query. That is not the query any more — those lines are the boot-time
    interrupted-run recovery sweep now — and `depth_by_source()` is at :1253 (:1229 until
    #2615's `pause_journal` DDL went into `_SCHEMA` 27 lines above it, which is precisely the
    way this pointer moves and why it is read from the file rather than remembered). So this unit
    cites a line number and this node opens the real file at that number and requires the
    `queue`/`GROUP BY source, state` text to be there, plus `def depth_by_source` above it:
    a citation checked against nothing is how the sibling ended up carrying a dead pointer
    into a unit that documents a mechanism by line (fixing that sibling is owed-check's
    ruling #2555/5, not this round's).

    And the comment must not justify the hour as a round-free window, because there is no
    such hour. Measured 2026-10-11 from `workers.db` over 2026-10-04 → 10-11: 420 `autocode`
    rounds, resident at 67.2% of 5-minute samples, starts in all 24 UTC hours (29 to 51 per
    hour), p95 duration 54.5 min against a 900 s default drain. A comment crediting the
    calendar would get this unit blamed for the next exit 2 instead of a too-short drain, so
    both measured figures are required in the prose, with the lever that does work.
    """
    c = _comments(TIMER)
    assert "workers/queue.py:1253" in c, c[:600]
    qlines = QUEUE.read_text(encoding="utf-8").splitlines()
    cited = [int(n) for n in re.findall(r"workers/queue\.py:(\d+)", c)]
    assert cited, "the comment names no line at all"
    hits = [n for n in cited
            if "GROUP BY source, state" in "\n".join(qlines[n - 1:n + 3])]
    assert hits, (
        f"no cited line {cited} holds the pool-status query; the pointer this unit leans on "
        "has rotted and the comment needs re-aiming at the current def")
    live = hits[0]
    assert "def depth_by_source" in "\n".join(qlines[:live]), \
        "the citation must point at or after depth_by_source()'s definition"
    for n in cited:
        if n not in hits:
            assert re.search(r"rot|not the query|boot-time|recovery", c), (
                f"workers/queue.py:{n} is cited but holds something else, and the comment "
                "never says so — that is how the sibling unit shipped a dead pointer")

    lower = c.lower()
    assert re.search(r"does not avoid|no calendar slot dodg|no fire time", lower), (
        "the comment must say outright that the fire time does not dodge automod rounds")
    assert "67.2" in c and "54.5" in c, (
        "the occupancy and p95 figures behind that refusal belong in the comment")
    assert "drain-wait" in c and str(DRAIN_S) in c, \
        "the comment must name the lever that actually prevents the refusal"


# ── clause 5: the arming route, and the rails a new unit trips ─────────────────────────

def test_setup_md_carries_the_enable_route_verbatim():
    """#2555 clause 5(a): the exact command, in the tree the installer does NOT link.

    `install-services.sh` links units; enabling is a decision it deliberately leaves out.
    That half-step is what shipped `lloyd-persistence-window.timer` linked-but-unarmed
    (#2537), and `scripts/maintenance/check-unit-enabledness.sh` reports FAIL for any repo
    timer that is not `enabled`, so an undispatched new timer turns this landing into a
    fleet-health failure on its next whole-fleet run. The string is matched verbatim
    because it is meant to be copied out of the document.
    """
    text = SETUP.read_text(encoding="utf-8")
    assert "systemctl --user enable --now lloyd-egress-window.timer" in text, text[-2000:]


def test_the_arming_entry_covers_active_state_and_the_two_deliberate_absences():
    """#2555 clause 5(b): the entry must not teach the check that shipped the sibling unarmed.

    `is-enabled` answers a symlink question. #2537 found a timer whose enable had never
    actually run, so the entry has to point at `is-active`/`ActiveState` as well, or the
    next reader repeats the miss and the owed-check's proof is weaker than its own
    predecessor's.

    The entry also has to carry the two things the unit deliberately does NOT set:
    `LLOYD_EGRESS_ENFORCE` and `LLOYD_CANARY_ROWS`. Those absences are the measurement and
    the corpus decision respectively, and they are exactly what a later editor "fixes"
    without knowing it — which is why the unit's test asserts them and the document has to
    say them.
    """
    text = SETUP.read_text(encoding="utf-8")
    parts = text.split("### The egress-arms window", 1)
    assert len(parts) == 2, "the window needs its own section, not a sentence inside another's"
    body = parts[1].split("\n### ")[0].split("\n## ")[0]
    assert re.search(r"is-active|ActiveState", body), body[:800]
    assert "LLOYD_EGRESS_ENFORCE" in body and "LLOYD_CANARY_ROWS" in body, body[:800]
    assert "8599" in body and str(DRAIN_S) in body, \
        "the port and the drain are the two numbers a human needs to recognise in the log"


def test_the_installer_links_both_units_by_its_existing_glob(tmp_path):
    """#2555 clause 1(d): `install-services.sh` needed no edit, and that is measured.

    Its glob covers every `*.service` and `*.timer` in `agent-services/systemd/`, so the
    installer installs this pair as it stands — which is what clause 1's "installed by
    install-services.sh" asks for, and the alternative (a hand-listed installer) is the
    thing that would have to be remembered. Run with a fake HOME and a stub `systemctl`, so
    the live box is untouched and `--user daemon-reload` is never reached.
    """
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sc = bin_dir / "systemctl"
    sc.write_text("#!/bin/sh\necho \"systemctl $*\" >> \"" + str(tmp_path / "calls") + "\"\n")
    sc.chmod(0o755)
    lc = bin_dir / "loginctl"
    lc.write_text("#!/bin/sh\necho yes\n")
    lc.chmod(0o755)
    # HOME, PATH and USER are NAMED, not inherited-with-gaps: `install-services.sh` runs
    # `set -u` and closes by reading `$USER` (`loginctl show-user "$USER"`, line 89), so an
    # environment carrying no login name — the gate's does not — kills the script there,
    # for a reason that has nothing to do with the units. #2428's sibling unit hit this and
    # its own helper names all three for exactly this reason.
    env = dict(os.environ, HOME=str(home), USER="egress-window-test",
               PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}")
    r = subprocess.run(["bash", str(INSTALLER)], capture_output=True, text=True,
                       cwd=str(REPO), env=env, timeout=180)
    assert r.returncode == 0, r.stderr[-900:]
    link_dir = home / ".config/systemd/user"
    for unit in (SERVICE, TIMER):
        link = link_dir / unit.name
        assert link.is_symlink(), \
            f"{unit.name} is not linked; the installer's glob changed:\n{r.stdout[-800:]}"
        assert link.resolve() == unit.resolve()


def test_both_units_are_user_scope_not_declared_for_the_system_manager():
    """#2555 clause 1(d): the units are user-scope, like every tracked unit in this directory.

    The installer's `systemctl --user link` drops the units into the user manager
    (`~/.config/systemd/user`). A unit that declares itself for the system manager would be
    linked by nothing at all and run by nothing — a silent no-op that reads as installed.
    The failure this guards is #2222's, where a system-scope copy shadowed the user one.
    """
    assert (UNIT_DIR / "system").is_dir(), (
        "agent-services/systemd/system/ holds the system-scope units; a new unit filed "
        "there is linked by nothing")
    assert not (UNIT_DIR / "system" / SERVICE.name).exists()
    assert not (UNIT_DIR / "system" / TIMER.name).exists()
    assert "[Install]" in SERVICE.read_text(encoding="utf-8")
    assert "[Install]" in TIMER.read_text(encoding="utf-8")
    for path in sorted(UNIT_DIR.glob("lloyd-egress-window.*")):
        assert "[Install]" in path.read_text(encoding="utf-8"), path.name


def test_the_new_timer_is_inside_the_enabledness_denominator():
    """#2555 clause 5(c): the new timer is inside the check that can go red over it.

    `check-unit-enabledness.sh` enumerates repo timers by glob, and its companion node in
    `tests/test_unit_enabledness_check.py` pins the denominator with an exact `len()` — the
    node's whole value is that a silently-added or renamed timer cannot skip it. This unit
    is that addition, so the count is re-derived here from the directory and compared with
    the number the sibling check ships, which is the difference between "the count moved"
    and "the count moved and the test was told".

    Arming the timer is a host action this round may not take (owed #2555/1); the check is
    therefore expected to report this one timer until that runs, and that is the finding the
    landing report carries, not a reason to skip the denominator.
    """
    timers = sorted(p.name for p in UNIT_DIR.glob("*.timer"))
    assert "lloyd-egress-window.timer" in timers, timers
    src = (REPO / "tests" / "test_unit_enabledness_check.py").read_text(encoding="utf-8")
    # The shipped check derives its denominator from the tree (`REPO_TIMERS` is a glob, and
    # the node asserts `_denominator(out) == len(REPO_TIMERS)`), so it needs NO number
    # edited when a timer is added — a remembered count would be the bug, not the fix. What
    # DOES have to stay true is that the derivation exists, in both directions.
    assert 'REPO_TIMERS = sorted(p.name for p in REPO_UNIT_DIR.glob("*.timer"))' in src, src[:400]
    assert "_denominator(out) == len(REPO_TIMERS)" in src, \
        "the shipped denominator stopped tracking the set it was given"
    assert not re.search(r"len\(REPO_TIMERS\) == \d+", src), \
        "a memorised count came back into the enabledness check"
    check = (REPO / "scripts" / "maintenance" / "check-unit-enabledness.sh")
    assert 'agent-services/systemd/*.timer' in check.read_text(encoding="utf-8") or \
        "*.timer" in check.read_text(encoding="utf-8"), \
        "the check stopped enumerating repo timers by glob; this timer could leave its denominator"


def test_the_drain_still_has_no_self_exclusion_so_the_unit_is_still_needed():
    """#2555 clause 1(a), the reason this pair exists: no exclusion, therefore no pool run.

    The executor argument rests on `in_flight()` counting the caller's own row, so this node
    reads the two source files instead of trusting the unit's comment. If someone adds a
    self-exclusion to the drain, this unit becomes redundant and this is the node that says
    so — the same role the sibling test's `not yet` node plays for the persistence window.

    The scenario count is asserted here too because the timeout's arithmetic and the owed
    row-count check (>=36) both multiply it: 18 scenarios x 1 rep x 2 arms = 36 rows.
    """
    ev = ARMS_PY.read_text(encoding="utf-8")
    assert re.search(r"def in_flight\(", ev), "the drain predicate moved"
    m = re.search(r"def in_flight\(.*?\n(?=\ndef |\nclass )", ev, re.S)
    assert m and "exclude" not in m.group(0) and "not self" not in m.group(0), \
        "in_flight() grew a self-exclusion, which makes this unit pair redundant — delete the drain"
    assert "depth_by_source" in QUEUE.read_text(encoding="utf-8"), \
        "the queue's per-source depth query moved"
    keys = _scenarios()
    assert len(keys) == 18, (len(keys), keys)
    assert "collect.off-allow-list.invalid" in DRIVER.read_text(encoding="utf-8"), \
        "the canary host literal the leak witness counts on moved; the unit comment is stale"
