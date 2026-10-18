"""#2555: the egress-arms window has an executor, and it is this pair of units.

`eval/egress_arms.py` landed complete and tested (b79835a4: the driver, the wrapper,
`tests/test_run_egress_arms.py`), and nothing on the box could fire it. Two reasons, one of
them structural:

* **A pool-dispatched window is impossible by construction.** The driver will not start its
  episodes while any queue row is in flight, and it learns that from `depth_by_source()`
  (`workers/queue.py` — `SELECT source, state, COUNT(*) as n FROM queue GROUP BY
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

Four nodes here assert against the unit files' own COMMENTS, and a fifth against the cadence
prose in SETUP.md and `architecture/infrastructure.md` — the move
`tests/test_persistence_window_unit.py` makes at its timeout and timer nodes. This
unit's numbers are its whole argument (a 5 h timeout, a 3,600 s drain, a 06:23 hour), and a
setting whose comment no longer justifies it is exactly the drift that left the precedent
unsupportable. The precedent's own failure was a CITATION, not a setting:
`lloyd-persistence-window.service` cited the pool-status query by line number long after
those lines had become the boot-time interrupted-run recovery sweep, and nothing noticed
because nothing re-read the file. #2772 retired both services' numbers in favour of naming
`depth_by_source()`, and `_resolve_pool_query_pointer` here re-reads every window unit's
pointer against `workers/queue.py`. #2971 retired every number corpus-wide, so each pointer
today is a symbol; the resolver's NUMBER branch is reachable only from its own red paths.

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

import pytest

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
INFRA = REPO / "architecture" / "infrastructure.md"

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


# ── the pool-status pointer both window units lean on ──────────────────────────────────
#
# Every "why this cannot be a pool job" paragraph in `agent-services/systemd/` ends at the
# same place: the queue's pool-status query. Citing it BY LINE NUMBER is what rotted — the
# egress service shipped pointing at a live line and #2615's `pause_journal` DDL moved the
# query below it without any test noticing, and the persistence service carried a dead
# number green from then until #2772. A symbol next to the file cannot drift without
# becoming a lie about a name, so that is the preferred form; a number is allowed only if
# this resolver can re-read it and find the query there.

POOL_QUERY_SQL = "GROUP BY source, state"        # the query's distinguishing text
POOL_QUERY_DEF = "def depth_by_source"           # the function that builds it
_QUEUE_ANCHOR = re.compile(r"workers/queue\.py")
_QUEUE_NUMERIC = re.compile(r"workers/queue\.py:(\d+)")
# `depth_by_source()` … workers/queue.py — the symbol bound to the file by the few
# non-digit words a comment can put between them ("(", " in ", " at "). Non-DIGIT rather
# than non-WORD: "`depth_by_source()` in workers/queue.py" has letters in the gap.
_QUEUE_SYMBOL = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)\(\)`\D{0,10}?workers/queue\.py")


def _resolve_pool_query_pointer(comment: str, qlines: list[str], *, src: str) -> list[str]:
    """Resolve every pointer `comment` makes to `workers/queue.py` against that file.

    Two forms resolve. A SYMBOL — `` `depth_by_source()` … workers/queue.py `` — needs
    `def <name>` to exist in the file. A NUMBER — ``workers/queue.py:<n>`` — needs the
    cited line's own window (that line plus the three below it, so a `:<n>-<m>` range
    counts) to hold the pool-status SQL, AND `def depth_by_source` to sit at or above it.

    Everything else raises, and there is deliberately no third form: an anchor naming the
    file with neither a line nor an adjacent symbol is unreadable, and — the part that
    matters — a stale number is NOT excused by the comment saying it is stale. The node
    this replaces forgave exactly that whenever the prose contained the word "rot", which
    is how a wrong number passed. `qlines` is a parameter rather than a read here so the
    failure modes below can be reached with a synthetic file instead of by damaging the
    real one (#2772 clauses 3 and 4).
    """
    resolved: list[str] = []
    covered: list[tuple[int, int]] = []

    for m in _QUEUE_NUMERIC.finditer(comment):
        n = int(m.group(1))
        window = "\n".join(qlines[n - 1:n + 3])
        assert POOL_QUERY_SQL in window, (
            f"{src}: `{m.group(0)}` is cited as the pool-status query, but lines "
            f"{n}–{n + 3} of workers/queue.py do not hold {POOL_QUERY_SQL!r}. The pointer "
            "has rotted: re-aim it at the symbol `depth_by_source()`, which cannot rot.")
        assert POOL_QUERY_DEF in "\n".join(qlines[:n]), (
            f"{src}: `{m.group(0)}` sits above {POOL_QUERY_DEF}, so the cited line cannot "
            "be inside the query the comment claims it is")
        resolved.append(m.group(0))
        covered.append(m.span())

    for m in _QUEUE_SYMBOL.finditer(comment):
        name = m.group(1)
        assert re.search(rf"\bdef {re.escape(name)}\b", "\n".join(qlines)), (
            f"{src}: `{m.group(0)}` names `{name}()` in workers/queue.py, which defines no "
            f"`def {name}`. The pointer names a symbol that is not there.")
        resolved.append(m.group(0))
        covered.append(m.span())

    uncovered = [comment[m.start():min(m.end() + 12, len(comment))]
                 for m in _QUEUE_ANCHOR.finditer(comment)
                 if not any(m.start() >= a and m.start() < b for a, b in covered)]
    assert not uncovered, (
        f"{src}: these mentions of workers/queue.py are neither a line number nor a "
        f"`symbol()` bound to the file, so nothing can be checked about them: {uncovered}")
    return resolved


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

def test_the_timer_fires_once_a_week_on_monday_with_catchup_and_bounded_jitter():
    """#2770 clause 1(a)/2, replacing the node that asserted
    `re.fullmatch(r"\\*-\\*\\* \\d{2}:\\d{2}:00", cal)` and so refused a weekday prefix by
    construction. That node did its job — it is the same shape #2754 had to replace on the
    sibling (`test_the_timer_fires_once_a_week_on_monday_and_catches_up` is that rewrite) —
    and the daily form it enforced is what #2770 bounds for this window.

    The weekly spec goes in the day-of-week field: `Mon *-*-* 06:23:00`. The trap spelling
    is `*-*-1 06:23:00` — the `1` sits in the day-of-MONTH field, systemd normalizes it to
    `*-*-01`, and a window that means Mondays gets the 1st of each month.
    `test_systemd_itself_reads_the_cadence_as_weekly_not_monthly` asks systemd to confirm
    both readings; the fullmatch here is what stops a regression from re-introducing the
    trap or the daily form.

    What #2770 does NOT change — cadence only, clause 2: `Persistent=true` (it matters more
    at one slot a week, not less: a suspend across Monday now costs the week's window
    rather than the night's), `RandomizedDelaySec=300` (the jitter every latest-start
    figure in this unit's comment is computed from), and `[Install] WantedBy=timers.target`
    — the arming route the enabledness check reads.
    """
    text = TIMER.read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines()
             if l.strip().startswith("OnCalendar=")]
    assert len(lines) == 1, f"this window gets exactly one slot: {lines}"
    d = dict(_parse(text)["Timer"])
    cal = d["OnCalendar"]
    assert re.fullmatch(r"Mon \*-\*-\* \d{2}:\d{2}:\d{2}", cal), \
        f"weekly in the day-of-week field with the date field left wild: {cal!r}"
    assert d["Persistent"] == "true"
    assert _secs(d["RandomizedDelaySec"]) <= 300, d["RandomizedDelaySec"]
    assert dict(_parse(text)["Install"]).get("WantedBy") == "timers.target"


def test_systemd_itself_reads_the_cadence_as_weekly_not_monthly():
    """#2770 clause 1(b): the grammar is systemd's, so the claim that separates the shipped
    spec from the monthly trap is asked of systemd — the way the sibling's identically
    named node asks it for 02:10.

    `systemd-analyze calendar` is a pure parser: it normalizes an expression and prints
    when it would next elapse, touching no manager and no unit state. Three readings, all
    asserted:

      * the unit's own value normalizes to itself, the `Mon` token intact;
      * `*-*-1 06:23:00` normalizes to `*-*-01 06:23:00`, a day-of-MONTH spec — twelve
        fires a year, on the 1st, of a window that means fifty-two Mondays;
      * the `Next elapse:` line for the shipped spec names a Monday. Unlike the date or
        the delta on that line — functions of the wall clock, which the sibling node
        refuses for exactly that reason — the weekday is a deterministic property of a
        weekday-scoped spec: the next Monday is always a Monday. That is clause 1's
        7-day interval, read from systemd rather than asserted against my arithmetic.
    """
    def analyzed(expr: str) -> dict[str, str]:
        proc = subprocess.run(["systemd-analyze", "calendar", expr],
                              capture_output=True, text=True, timeout=30, check=True)
        out: dict[str, str] = {}
        for line in proc.stdout.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
        return out

    cal = dict(_parse(TIMER.read_text(encoding="utf-8"))["Timer"])["OnCalendar"]
    got = analyzed(cal)
    assert got["Normalized form"] == cal == "Mon *-*-* 06:23:00", \
        f"systemd reads the shipped cadence as something other than a Monday: {cal!r}"
    assert analyzed("*-*-1 06:23:00")["Normalized form"] == "*-*-01 06:23:00", \
        "the day-of-month form is monthly, and if systemd ever stops saying so this " \
        "node's reasoning needs re-deriving, not just its expected string"
    assert got["Next elapse"].split()[0] == "Mon", \
        f"a Monday-scoped spec must next elapse on a Monday: {got['Next elapse']!r}"


def test_the_timer_hour_is_after_every_named_fire_and_clear_of_the_sibling_ceiling():
    """#2555 clause 4(b), #2770 clause 3: the hour justified against the other repo timers,
    read from their files, and both calendars parsed position-independently.

    `06:23` is later than every other all-wildcard-date (daily) calendar in
    `agent-services/systemd/` — since #2754 took the persistence window weekly and #2770
    takes this one the same way, that set is cert-renew 04:17 and qmd-cleanup 04:45 — and
    it clears the persistence window's own stated ceiling: the sibling's fire time and its
    `TimeoutStartSec=4h` are re-read from the sibling's two files here, so 06:10 is the
    latest that window can still hold the engine and this one starts 13 minutes past that.
    Both windows are Monday-scoped now, so the comparison is a time-of-day statement, and
    what the cadence change made true is that Monday's two windows are consecutive rather
    than concurrent.

    Position-independently (`[-1]`, not `[1]`): systemd's grammar puts the day-of-week
    FIRST, so with a weekday prefix index 1 is the date field `*-*-*` and `int("*")` raises
    ValueError. #2754 hit exactly that in this node's sibling read; #2770 would hit it in
    the self read the moment this unit gained its prefix — an error naming neither field.
    Nothing here transcribes an hour or a ceiling: the sibling's figures arrive by
    re-reading the sibling, and this unit's arrive by re-reading this unit.
    """
    d = dict(_parse(TIMER.read_text(encoding="utf-8"))["Timer"])
    hh, mm = [int(x) for x in d["OnCalendar"].split()[-1].split(":")[:2]]

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


def test_the_timer_comment_states_the_weekly_cadence_the_precedent_and_the_margin():
    """#2770 clause 4: the comment says what the timer now does, names the precedent, and
    keeps the sibling arithmetic that justifies the hour.

    The old paragraph listed the persistence window among "the daily calendars here" —
    a sentence `dfeec1e6` (#2754) had already falsified for the sibling — and never named
    a cadence for this unit at all. Three pins:

      * the cadence in prose: weekly, on Mondays, not only in the `OnCalendar=` line;
      * the precedent by number: #2754 is why Mondays, and the comment must say so;
      * the margin as arithmetic: the sibling's fire time and ceiling are re-read from
        the sibling's files here, and the comment must state the sum — the latest minute
        the sibling's engine work can still run (06:10 from 02:10 + 4 h) — and how many
        minutes past it this window starts. The figures are DERIVED here and matched
        against the comment, so a comment that transcribes them will rot when the
        sibling moves, which is clause 3's discipline applied to prose.
    """
    c = _comments(TIMER)
    assert re.search(r"(?i)weekly", c), "the comment never names the cadence"
    assert re.search(r"(?i)monday", c), "the comment never names the weekday it fires on"
    assert "#2754" in c, "the comment must name the precedent that made the sibling weekly"

    sh, sm = [int(x) for x in dict(_parse(SIBLING_TIMER.read_text(encoding="utf-8"))["Timer"])
              ["OnCalendar"].split()[-1].split(":")[:2]]
    sib_to = _secs(dict(_parse(SIBLING_SERVICE.read_text(encoding="utf-8"))["Service"])
                   ["TimeoutStartSec"])
    end = sh * 60 + sm + sib_to // 60
    latest = f"{end // 60:02d}:{end % 60:02d}"
    hh, mm = [int(x) for x in dict(_parse(TIMER.read_text(encoding="utf-8"))["Timer"])
              ["OnCalendar"].split()[-1].split(":")[:2]]
    past = hh * 60 + mm - end
    assert re.search(re.escape(latest), c), \
        f"the comment must state the latest minute the sibling's ceiling can run ({latest})"
    assert f"{past} minutes" in c, \
        f"the comment must state how far past that ceiling this window starts ({past} minutes)"
    assert re.search(r":23\s*\u2013\s*:28|:23\s*-\s*:28", c), \
        "the comment must bound the real start by the jitter it ships (:23–:28)"

    sib_fire = f"{sh:02d}:{sm:02d}"
    for line in c.splitlines():
        if "daily" in line.lower():
            assert ("persistence" not in line.lower() and sib_fire not in line), (
                f"the comment lists the sibling's {sib_fire} among daily calendars again, "
                f"the sentence #2754 had already falsified: {line.strip()!r}")


def test_no_surface_calls_this_window_a_daily_fire_anymore():
    """#2770 clause 5: every sentence that called this window a daily fire is updated, and
    the enable route survives the rewrite verbatim.

    Four surfaces describe this timer's cadence: the unit's `Description=`, the `[Timer]`
    block itself (pinned by the nodes above), SETUP.md's enable route, and the
    infrastructure unit table. The first two are graded from the unit file; the last two
    are read from the tree here. `test_setup_md_carries_the_enable_route_verbatim` pins
    the command string, so a rewrite that fixed the command's COMMENT to say "daily" while
    the unit fired Mondays would still pass it — the exact gap the sibling's #2754 closed
    by re-pinning its own enable route with its cadence comment, mirrored here.
    """
    desc = dict(_parse(TIMER.read_text(encoding="utf-8"))["Unit"])["Description"]
    assert "daily" not in desc.lower(), f"the Description still names a cadence: {desc!r}"
    assert re.search(r"(?i)weekly|monday", desc), \
        f"the Description must name the cadence the timer now has: {desc!r}"

    text = SETUP.read_text(encoding="utf-8")
    m = re.search(r"systemctl --user enable --now lloyd-egress-window\.timer"
                  r"[ \t]+#[ \t]*([^\n]*)", text)
    assert m, "the enable route must stay in SETUP.md verbatim through the rewrite"
    assert not re.search(r"(?i)\bdaily\b", m.group(1)), \
        f"the enable route's comment still claims a daily cadence: {m.group(1)!r}"
    assert re.search(r"(?i)monday|weekly", m.group(1)), \
        f"the enable route's comment must carry the shipped cadence: {m.group(1)!r}"
    body = text.split("### The egress-arms window", 1)[1]
    body = body.split("\n### ")[0].split("\n## ")[0]
    for line in body.splitlines():
        assert "daily" not in line.lower(), \
            f"SETUP.md claims a daily cadence for this window: {line.strip()!r}"
    assert re.search(r"(?i)weekly|monday", body), \
        "SETUP.md's egress section must say the cadence the unit now has"

    row = next((l for l in INFRA.read_text(encoding="utf-8").splitlines()
                if l.lstrip().startswith("|") and "lloyd-egress-window.timer" in l), None)
    assert row is not None, "the infrastructure unit table lost this timer's row"
    assert not re.search(r"(?i)\bdaily\b", row), \
        f"the infrastructure row still claims a daily fire: {row.strip()!r}"
    assert re.search(r"(?i)weekly|monday", row), \
        f"the infrastructure row must read weekly: {row.strip()!r}"


#: Every unit file in this directory that points at the queue's pool-status query.
#: Enumerated rather than listed below, because a fourth unit that grows such a pointer
#: has to be covered by the same rule and there is nothing to remember the name; the
#: three #2772 names are then asserted PRESENT, so the guard cannot quietly cover zero.
WINDOW_UNITS_WITH_POOL_POINTER = (SIBLING_SERVICE.name, SERVICE.name, TIMER.name)


def test_the_window_units_point_at_the_live_pool_query_and_refuse_to_claim_the_hour_avoids_rounds():
    """#2555 clause 4(c), #2772 clauses 1-4: every window unit's pointer resolves, and the
    false claim about the hour is absent.

    This node used to open ONE file — the egress timer — and to assert against it a
    literal line (`"workers/queue.py:<n>" in c`), which is the pattern the item forbids:
    the guard rotted with the number it quoted. It now walks every `.service`/`.timer` in
    the directory whose comments name `workers/queue.py`, hands each comment to
    `_resolve_pool_query_pointer`, and requires the three #2772 names to be in the set it
    resolved. All three point by symbol: #2971 retired the timer's surviving line number
    too, so the resolver's NUMBER branch is now reachable only from its own red-path tests.

    Why the number is unsafe and the symbol is not: the mechanism is on the record in the
    resolver's own header — a pointer aims at a line, `queue.py` grows above it, and
    nothing notices. That is what a dead citation costs: the persistence service shipped
    one, and the egress timer's comment had to spend a sentence explaining that its
    sibling's citation was stale. Clause 4's red paths are proven below by handing the
    resolver synthetic comments over a synthetic `queue.py`, so a wrong number in a real
    unit cannot pass on the strength of prose — the excusing clause this rewrite deletes
    forgave a dead number the moment the comment said the word "rot".

    Kept from the old node, and still about the timer: the comment must not justify the
    hour as a round-free window, because there is no such hour. Measured 2026-10-11 from
    `workers.db` over 2026-10-04 → 10-11: 420 `autocode` rounds, resident at 67.2% of
    5-minute samples, starts in all 24 UTC hours (29 to 51 per hour), p95 duration 54.5 min
    against a 900 s default drain. A comment crediting the calendar would get this unit
    blamed for the next exit 2 instead of a too-short drain, so both measured figures are
    required in the prose, with the lever that does work.
    """
    qlines = QUEUE.read_text(encoding="utf-8").splitlines()

    resolved: dict[str, list[str]] = {}
    for unit in sorted(UNIT_DIR.iterdir()):
        if unit.suffix not in (".service", ".timer"):
            continue
        c = _comments(unit)
        if _QUEUE_ANCHOR.search(c):
            resolved[unit.name] = _resolve_pool_query_pointer(c, qlines, src=unit.name)
    for name in WINDOW_UNITS_WITH_POOL_POINTER:
        assert name in resolved, (
            f"{name} no longer points at workers/queue.py at all — the pointer is what "
            "justifies this unit existing rather than being a pool job")
    assert resolved, "not one unit file in the directory names the pool-status query"

    # clause 1: the persistence service names the symbol, carries no line number, and the
    # sentence around it still says WHY — the query has no exclusion of the calling row, so
    # a pool-dispatched window waits on itself and burns its drain.
    sib = _comments(SIBLING_SERVICE)
    assert "`depth_by_source()`" in sib, \
        "the sibling unit must name the symbol, not a line, for the pool-status query"
    assert not _QUEUE_NUMERIC.search(sib), \
        "the sibling unit is back to citing workers/queue.py by line number — the form that " \
        "rotted once already (#2772)"
    assert re.search(r"no[\s\S]{0,40}?exclusion of the calling row", sib), \
        "the mechanism sentence went with the number: the query must be said to exclude nothing"
    # `[\s\S]` rather than a space wherever the sentence can re-wrap: a comment line break
    # is a reflow, not a change of meaning, and a guard that fails on one teaches people to
    # delete it.
    assert "drain" in sib and re.search(r"pool[\s\S]{0,4}run[\s\S]{0,20}?burns", sib), \
        "the sentence must still draw the consequence for a pool-dispatched window"

    # clause 2: same for this service, whose sentence ends in the exit code rather than a
    # burned drain, because this unit's drain is the hour that was set to beat a round.
    own = _comments(SERVICE)
    assert "`depth_by_source()`" in own, \
        "this unit must name the symbol, not a line, for the pool-status query"
    assert not _QUEUE_NUMERIC.search(own), \
        "this unit is back to citing workers/queue.py by line number (#2772)"
    assert re.search(r"no clause excluding the caller[\u2019']s own row", own), \
        "the mechanism sentence went with the number: the query must be said to exclude " \
        "the caller nothing"
    assert "exits 2" in own, "the consequence the dispatched window ends in must stay stated"

    # clause 4: the resolver's red paths, each reachable from a synthetic comment. Line
    # numbers are DERIVED from the synthetic bodies, never transcribed, so nothing here is
    # an assertion about where the real query lives.
    above_def = ["    rows = conn.execute(",
                 '        "SELECT source, state, COUNT(*) as n FROM queue GROUP BY source, state"',
                 "    return rows",
                 "",
                 f"    {POOL_QUERY_DEF}(self) -> int:  # defined BELOW the cited line"]
    # Derived from the synthetic body, never transcribed: which line of it holds the SQL.
    sql_line = next(i for i, ln in enumerate(above_def, 1) if POOL_QUERY_SQL in ln)
    with pytest.raises(AssertionError, match="above"):
        _resolve_pool_query_pointer(f"see workers/queue.py:{sql_line}", above_def,
                                    src="synthetic")
    no_sql = [f"    {POOL_QUERY_DEF}(self):", "        return {}"]
    with pytest.raises(AssertionError, match="do not hold"):
        _resolve_pool_query_pointer(f"see workers/queue.py:{sql_line}", no_sql,
                                    src="synthetic")
    with pytest.raises(AssertionError, match="defines no"):
        _resolve_pool_query_pointer("(`not_the_real_thing()` in workers/queue.py)",
                                    qlines, src="synthetic")
    with pytest.raises(AssertionError, match="neither a line number"):
        _resolve_pool_query_pointer("the queue file is workers/queue.py and that is all",
                                    qlines, src="synthetic")
    assert _resolve_pool_query_pointer("(`depth_by_source()` in workers/queue.py)",
                                       qlines, src="synthetic"), \
        "the symbol form the units now use must resolve against the real file"

    c = _comments(TIMER)
    lower = c.lower()
    assert re.search(r"does not avoid|no calendar slot dodg|no fire time", lower), (
        "the comment must say outright that the fire time does not dodge automod rounds")
    assert "67.2" in c and "54.5" in c, (
        "the occupancy and p95 figures behind that refusal belong in the comment")
    assert "drain-wait" in c and str(DRAIN_S) in c, \
        "the comment must name the lever that actually prevents the refusal"
    assert re.search(r"does not avoid|no calendar slot dodg|no fire time", lower), (
        "the comment must say outright that the fire time does not dodge automod rounds")
    assert "67.2" in c and "54.5" in c, (
        "the occupancy and p95 figures behind that refusal belong in the comment")
    assert "drain-wait" in c and str(DRAIN_S) in c, \
        "the comment must name the lever that actually prevents the refusal"


def test_systemd_itself_still_parses_every_window_unit_after_the_comment_edits():
    """#2772 clause 5: the comment edits leave the units loadable by the real parser.

    The only consumer of a systemd comment is systemd. Every other node here reads these
    files through `_parse`, which is this suite's own INI view, and the arming nodes cross
    the install route with a STUB `systemctl`; the one node that shells out to systemd
    (`test_systemd_itself_reads_the_cadence_as_weekly_not_monthly`) hands it a calendar
    expression, not the file. So a comment edit that broke the real grammar — an unbalanced
    continuation, a stray directive character — would have gone green here and failed at the
    next fire, which is the worst place to learn it. `systemd-analyze verify` is systemd's
    own loader, takes a path, and needs no manager, no engine and no `systemctl`, so it is
    the boundary a round is allowed to cross.

    It is run over all four window units, not just the three this round touched: the claim
    is that the shipped set loads, and a set pinned to "the files I edited" re-pins itself
    out of existence the next time somebody touches a fourth.
    """
    units = sorted(UNIT_DIR.glob("lloyd-*-window.*"))
    assert len(units) == 4, [u.name for u in units]
    for unit in units:
        proc = subprocess.run(["systemd-analyze", "verify", str(unit)],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, \
            f"systemd cannot load {unit.name} after the comment edits: {proc.stderr.strip()}"


# ── clause 5: the arming route, and the rails a new unit trips ─────────────────────────

def test_setup_md_carries_the_enable_route_verbatim():
    """#2555 clause 5(a): the exact command, in the tree the installer does NOT link.

    `install-services.sh` used to link units and leave enabling as a decision nobody took;
    since #2774 it arms every `.timer` it links, so the route below is what the script now
    runs rather than only what a human copies. The half-step is what shipped
    `lloyd-persistence-window.timer` linked-but-unarmed (#2537), and
    `scripts/maintenance/check-unit-enabledness.sh` reports FAIL for any repo timer that is
    not `enabled`, so an undispatched new timer turns this landing into a
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


# ── #2971: numeric `workers/queue.py:<n>` citations are retired corpus-wide ────────────
#
# `_resolve_pool_query_pointer` above can re-read a NUMBER, but only where a caller hands
# it the text: its one caller walks `agent-services/systemd` comments, and the absence it
# asserts is over the two `.service` files. The same citation sat on 17 lines across 13
# tracked files under the six roots, and 14 of those 17 pointed at lines that had moved.
# The drift mechanism is on the record: `691df24b` landed #2615's `pause_journal` DDL above
# `def depth_by_source` and pushed it from 1226 to 1253 with no test noticing — verified with
# `git show 691df24b^:workers/queue.py | grep -n 'def depth_by_source'` (1226) against
# `git show 691df24b:...` (1253). The two unit citations that resolve today resolve only
# because #2772 re-aimed them a week ago, and they would have rotted again at the next
# DDL above them.
#
# A resolver has to be handed the corpus, so it cannot outlast it; an absence can. The ban
# is therefore TOTAL over the six roots, including the citations that currently resolve —
# a partial retirement is not greppable, and today's live number is next release's dead
# one. What is NOT done here: the resolver's NUMBER branch is left in place (deleting it is
# an owed ruling on #2971), and `tests/test_architecture_coverage_doc_claims.py`'s span
# sweep is untouched — it asks whether a line falls inside some definition, which is a
# different property, and it passes all three dead architecture anchors today.

#: The six roots the ban covers. Exhaustive at filing: `git grep -c -E
#: 'workers/queue[.]py:[0-9]+'` over the base commit named 18 files — 13 of them here
#: (17 citations) and 5 under the record roots (12 citations). Nothing outside these
#: seventeen files, in `web/` or `docs/` or the repo root, carried one.
CORPUS_ROOTS = ("app", "agent-services", "architecture", "eval", "scripts", "tests")

#: The four excluded prefixes, and the ONLY exclusions. By directory prefix rather than a
#: list of files so no per-citation allowlist can grow: each is a frozen corpus the
#: citations are evidence INSIDE, and rewriting a baseline or a recovered row to satisfy a
#: guard would falsify the artifact. Their existence is asserted, so a renamed record root
#: cannot silently turn a record root into a free exemption for a new directory.
RECORD_ROOTS = ("eval/baselines", "eval/iv", "eval/cross_engine_disagreement_arm1",
                "tests/fixtures")

#: A line number handed to the synthetic corpus below, to prove the reader is not a no-op.
#: Spelled as a separate constant because THIS file is inside the corpus: the literal
#: `workers/queue.py:` + digits may not appear anywhere in it, and an f-string that builds
#: the synthetic citation from this number is the one spelling that is both honest and
#: invisible to the checker that reads this file.
SYNTHETIC_CITED_LINE = 999


def _is_derived(rel: str) -> bool:
    """True for a file the compiler wrote, not a file a person wrote a sentence into.

    `__pycache__` and bytecode are excluded from the walk, and they are the ONLY thing
    excluded besides the four record prefixes. They have to be: `quote`-style bytecode
    embeds every docstring and comment string constant of the module it was built from, so
    a stale `app/__pycache__/failure_ledger.cpython-313.pyc` left by a pre-retirement
    import still holds a `workers/queue.py:` citation that no tracked file contains — a red
    guard on a clean tree, which is the false alarm this whole item exists to end. The
    Filter by KIND (a compiled artifact), and the reason there is a filter at all is measured
    rather than assumed. Bytecode keeps a module's DOCSTRINGS and string constants but drops
    comments, so a citation survives compilation exactly when it sits in a docstring: compiling
    the thirteen pre-retirement Python files at 9c913df5 and grepping the resulting `.pyc` keeps
    8 of the 17 citations (`app/routers/dashboard.py`, `eval/persistence_arms.py`,
    `scripts/automod/landed_outcomes.py` and the four test files), while `app/failure_ledger.py`
    and `scripts/step_conformance.py` come back with 0 because theirs sat in a comment. A machine
    that imported any of the first group before this round holds a numeric citation that no
    tracked file contains any more, and a reader that reached it would be red on a clean tree
    depending on what was imported last — the false alarm this item exists to end. The second
    reason is the ban's own definition: the acceptance check is `git grep`, which reports no
    lines inside binary files either, and 850 files under these six roots begin with a NUL, so a
    reader that took every byte would be stricter than the instrument the item names and would
    redden on a local `agent-services/models/**` blob.

    Declining by name is still a hole — a hand-written `.py` parked under `__pycache__/` is
    declined too — and `_prose_declined_by_name` is what closes it. Its edges are proven on a
    planted tree in the synthetic node, because whether residue exists is a property of a working
    tree and not of a commit.
    """
    return "__pycache__" in rel.split("/") or rel.endswith((".pyc", ".pyo"))


def _compiled_by_content(path: Path) -> bool:
    """Whether the file's OWN BYTES say it is a compiled artifact, whatever its name says.

    CPython writes a four-byte magic number then a four-byte flags word that has been zero
    since 3.7, so every real `.pyc` has a NUL inside its first eight bytes and no text file
    does. Measured on the live checkout: of the 777 files the name filter declines under these
    six roots, none lacks that NUL — the two predicates agree over the whole real corpus, and
    the corpus node asserts the tracked half of that agreement rather than a count quoted here.
    This function deliberately never calls `_is_derived`, which is what lets it disagree with
    the name filter instead of restating it: restating it was the whole defect of the first
    witness.
    """
    return b"\x00" in path.read_bytes()[:8]


def _prose_declined_by_name(root: Path) -> list[str]:
    """Paths declined as bytecode whose own bytes are prose, so a sentence could live there.

    The witness this replaces computed the dropped set by subtracting two walks that shared
    `_is_derived`, so the difference was bytecode by construction and the assertion could not
    fail (review attempt 2). This asks each declined file. Who owns what comes back is the
    caller's business: `git` residue is not part of a tree a reviewer can read, while a path
    `git` TRACKS is corpus by definition and no directory name excuses it.
    """
    declined: list[str] = []
    for rel in CORPUS_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rp = path.relative_to(root).as_posix()
            if any(rp == ex or rp.startswith(ex + "/") for ex in RECORD_ROOTS):
                continue
            if _is_derived(rp) and not _compiled_by_content(path):
                declined.append(rp)
    return declined


def _tracked(root: Path) -> set[str]:
    """Paths `git` tracks under the six roots — the corpus a reader of the repo can see.

    This is the same corpus the item's own acceptance check reads, since `git grep` walks
    tracked files and nothing else, so the two instruments cannot disagree about what is banned.
    """
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", *CORPUS_ROOTS],
                         capture_output=True, text=True, check=True)
    return {p for p in out.stdout.split("\0") if p}


def _corpus_files(root: Path):
    """Yield `(relpath, path)` for every file of the ban's corpus inside `root`.

    The six roots walked with `rglob`, the four record prefixes dropped by path element (a
    file named `eval/iveil.py` is not under `eval/iv`) and compiled artifacts dropped by
    kind — see `_is_derived`, whose exclusions the corpus node witnesses one by one. Files
    are decoded with `errors="ignore"` rather than skipped when they are not valid UTF-8: a
    size or encoding exemption would be an unauthorised hole, and the walk has to be able to
    say "nothing here" only because nothing here matches.
    """
    for rel in CORPUS_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rp = path.relative_to(root).as_posix()
            if any(rp == ex or rp.startswith(ex + "/") for ex in RECORD_ROOTS):
                continue
            if _is_derived(rp):
                continue
            yield rp, path


def _numeric_queue_citations(root: Path) -> list[str]:
    """Every `workers/queue.py:<digits>` citation under the six roots of `root`.

    One regex, the same `_QUEUE_NUMERIC` the resolver above already uses, so the ban and
    the pointer checker cannot disagree about what a numeric citation is. Returns
    `relpath:line` plus the offending text, because a failure that names only a count is a
    failure the next reader has to go and find.
    """
    hits: list[str] = []
    for rp, path in _corpus_files(root):
        text = path.read_bytes().decode("utf-8", "ignore")
        for n, line in enumerate(text.splitlines(), 1):
            if _QUEUE_NUMERIC.search(line):
                hits.append(f"{rp}:{n}: {line.strip()[:100]}")
    return hits


def test_the_six_roots_hold_not_one_numeric_workers_queue_citation():
    """#2971 clauses 1-2: the corpus is clean, read from the tree rather than from a list.

    At filing the scoped grep returned 17 citations in 13 files, and 14 of the 17 were
    dead —
    `app/routers/dashboard.py` pointing at `mark_failed` on 394-430 when the def is at 925,
    `architecture/research-pipeline.md` pointing at `list_runs_joined` on 677 when it is at
    1577, four citations at 1353/1301 landing inside `claims_by_source` and
    `pending_wait_by_source` while `run_rollup_by_source` is at 1428. The property is
    re-derived from the filesystem here, not from that list, so a fifteenth citation added
    next month reddens this node without anybody updating it.

    The walk's own reach is asserted alongside the result: roots present, all four record
    prefixes still directories (so an exclusion cannot be "widened" by renaming the thing
    it excludes), and a file count large enough to show the walk is not standing in an
    empty directory. The checker's non-vacuity — the case where the corpus is clean because
    the reader reads nothing — is the next node's, over a synthetic tree.
    """
    for rel in CORPUS_ROOTS:
        assert (REPO / rel).is_dir(), f"the ban's corpus lost a root: {rel}"
    for rel in RECORD_ROOTS:
        assert (REPO / rel).is_dir(), (
            "record root " + rel + " is gone, and its absence would silently widen the "
            "exemption — the exclusion list must move only with a deliberate edit")
    scanned = [rp for rp, _ in _corpus_files(REPO)]
    assert len(scanned) > 2000, f"the walk reached only {len(scanned)} files"
    #: Nothing the walker declines by NAME may be a file git tracks. The first draft witnessed
    #: the kind filter with `set(all) - set(scanned)`, and `_corpus_files` IS that walk minus
    #: `_is_derived`, so the leftovers were bytecode by construction and the assertion could
    #: never fail (review attempt 2, finding 1). `_prose_declined_by_name` instead asks each
    #: declined file's own bytes, through a predicate that never calls `_is_derived`, and the
    #: rule below is the one that matters: compiler residue is nobody's corpus, but a TRACKED
    #: file declined by name is a sentence parked outside the ban. The agreement half of it is
    #: not vacuous on a box that has ever run the code — 777 files under these roots are
    #: declined by name, and every one of them has the header that says it was compiled — and
    #: both edges of the TRACKED half are proven on a planted tree in the next node, because
    #: whether residue exists is a property of a working tree, not of a commit.
    declined = _prose_declined_by_name(REPO)
    hiding = sorted(set(declined) & _tracked(REPO))
    assert hiding == [], (
        "the walker declines these as bytecode by name but git tracks them, so they are "
        "corpus and a citation can hide there: " + ", ".join(hiding[:10]))
    #: Files the ban must reach, named so a broken root shows up as a missing name rather
    #: than as a suspiciously clean corpus.
    for must_see in ("app/routers/dashboard.py", "architecture/threat-model.md",
                     "agent-services/systemd/lloyd-egress-window.timer",
                     "eval/persistence_arms.py", "scripts/step_conformance.py",
                     "tests/test_persistence_window_unit.py"):
        assert must_see in scanned, must_see
    assert not any(rp.startswith(("eval/baselines", "eval/iv",
                                  "eval/cross_engine_disagreement_arm1", "tests/fixtures"))
                   for rp in scanned), "a record prefix leaked back into the corpus"

    hits = _numeric_queue_citations(REPO)
    assert hits == [], (
        "numeric workers/queue.py line citations are banned under the six roots: they rot "
        "silently (691df24b moved depth_by_source 1226->1253). Re-aim each at the symbol "
        "its own sentence names. Found:\n" + "\n".join(hits))


def test_the_citation_reader_reddens_on_a_synthetic_corpus_and_its_four_prefixes(tmp_path):
    """#2971 clause 3: the checker is not a no-op, and its exemptions are exactly four.

    Proven the way #2772 proved the resolver's red paths — with a synthetic tree, so the
    real corpus never has to be damaged. Both directions are needed: a tree carrying one
    numeric citation must come back naming it (otherwise the clean real corpus could mean
    "the walk reached nothing"), and the same citation placed under each record prefix must
    come back clean (otherwise the exclusion is not what these nodes claim it is). A
    lookalike directory is included because prefix matching by string alone would exempt
    `eval/iveil/`: the exclusion is by path element, and that is what is asserted.

    The same node proves the two edges of the KIND filter, which no real tree can show at once
    (finding 1). A genuine `.pyc` is planted carrying the retired citation and must come back
    clean — bytecode keeps docstrings, and 8 of the 17 retired citations do survive compilation,
    so a reader that reached a stale one would be red on a clean tracked tree. And a prose `.py`
    planted in the same directory must be NAMED by
    `_prose_declined_by_name`, because declining it by name is exactly the hole a witness has to
    see. Then `git init` decides who owns it: untracked, it is residue; staged, the corpus node's
    rule must catch it.
    """
    def plant(rel: str, marker: str) -> Path:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        cited = f"workers/queue.py:{SYNTHETIC_CITED_LINE}"
        p.write_text(f"{marker} see {cited} for the query\n", encoding="utf-8")
        return p

    planted = plant("app/synthetic_reader.py", "#")
    plant("architecture/synthetic_reader.md", "<!--")
    plant("eval/synthetic/synthetic_reader.py", "#")
    for rel in RECORD_ROOTS:
        plant(f"{rel}/synthetic_reader.txt", "")
    #: A genuine bytecode header (this interpreter's four magic bytes, then the zeroed flags
    #: word and the mtime/size fields that follow) and then the retired citation, as a stale
    #: `.pyc` really does carry it. Local import because this file's header block is itself
    #: cited by line number elsewhere, and a guard against stale line numbers has no business
    #: moving one.
    import importlib.util
    pyc = tmp_path / "app/__pycache__/synthetic.cpython-312.pyc"
    pyc.parent.mkdir(parents=True, exist_ok=True)
    pyc.write_bytes(importlib.util.MAGIC_NUMBER + b"\x00" * 12
                    + f"see workers/queue.py:{SYNTHETIC_CITED_LINE} for the query\n".encode())

    hits = _numeric_queue_citations(tmp_path)
    assert len(hits) == 3, hits
    assert any(h.startswith("app/synthetic_reader.py:") for h in hits), hits
    assert str(planted.relative_to(tmp_path)).replace("\\", "/") in "".join(hits)
    #: The number the reader reports is the number in the text, not a bare count.
    assert any(f":{SYNTHETIC_CITED_LINE}" in h for h in hits), hits
    assert not any("synthetic_reader.txt" in h for h in hits), (
        f"a record prefix was not honoured: {hits}")
    #: Bytecode is declined, and on purpose: the planted `.pyc` above does carry the citation.
    assert not any("__pycache__" in h for h in hits), (
        f"the reader reached compiled residue, which is the false alarm on a clean tracked "
        f"tree that `_is_derived` exists to prevent: {hits}")
    #: And the witness agrees nothing was declined under a false name — the planted `.pyc` has
    #: a header that says it was compiled, so no prose is being swallowed here.
    assert _prose_declined_by_name(tmp_path) == [], (
        "a planted file whose bytes are a real bytecode header was reported as swallowed prose")
    #: Now the disguise: same directory, same citation, but the bytes are prose. The reader
    #: still declines it — the filter is by name — so `_prose_declined_by_name` naming this one
    #: file is the only thing between a sentence and the ban. This is the case the set
    #: difference could never have shown, whatever tree it ran on.
    disguised = tmp_path / "app/__pycache__/disguised_prose.py"
    disguised.write_text(f"# see workers/queue.py:{SYNTHETIC_CITED_LINE} for the query\n",
                         encoding="utf-8")
    assert _prose_declined_by_name(tmp_path) == ["app/__pycache__/disguised_prose.py"], (
        "the witness cannot see a prose file hidden under __pycache__, so the name filter is "
        "an unwitnessed exemption again and the corpus node's rule is vacuous")
    #: Untracked, that file is residue the ban does not own. Staged, it is corpus, and the
    #: corpus node's `set(declined) & _tracked(...)` is what stops a tracked sentence sitting
    #: outside the walk — so `_tracked` has to answer for a staged file, which is proven here
    #: rather than assumed.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert "app/__pycache__/disguised_prose.py" not in _tracked(tmp_path), (
        "git reports an unstaged file as tracked, so the corpus node's rule would excuse "
        "anything left in the working tree")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", "--",
                    "app/__pycache__/disguised_prose.py"], check=True)
    assert "app/__pycache__/disguised_prose.py" in _tracked(tmp_path), (
        "a staged file is not reported as tracked, so `set(declined) & _tracked(...)` could "
        "never fire and the kind filter is unwitnessed after all")
    #: And the near-miss directory is NOT exempt: `eval/iveil/` is under `eval/`, not under
    #: `eval/iv/`.
    plant("eval/iveil/synthetic_reader.py", "#")
    hits2 = _numeric_queue_citations(tmp_path)
    assert len(hits2) == 4, hits2
    assert any(h.startswith("eval/iveil/") for h in hits2), hits2


#: Each retired citation's file → the symbol(s) its sentence names, a phrase from the
#: mechanism that citation was attached to, and the term that must be found INSIDE the
#: named queue.py definition. All three are needed because the three failures are
#: different: a deleted citation (first check), a deleted explanation (second), and a
#: sentence that names a real symbol for a claim that symbol does not carry — which is
#: what the review caught in the first draft of this list, where `quarantine()` was named
#: for the poisoned/quarantined spelling although `quarantine()` moves an already-poisoned
#: row to `quarantined` and `mark_failed()` is what writes `poisoned` at
#: `attempts >= max_attempts`. The grounding term makes that a test failure, not a reread.
#:
#: Symbols and terms were measured at the base commit: mark_failed 925, claim_next 705
#: (its due-gate is `not_before`), mark_running 844, _RUN_INSERT_SQL 374, _insert_run 381,
#: run_rollup_by_source 1428 (`gpu_hours` and the failed/interrupted SUM inside it),
#: wm_get 1594 (`watermarks`), quarantine 1195 ("needs attention"), list_runs_joined 1577
#: (`LEFT JOIN`), depth_by_source 1253 (`GROUP BY source, state`).
#: What the second element of each entry is NOT: a term the file happens to contain.
#: `gpu_hours` occurs 8 times in `tests/test_landed_outcomes_by_author.py` and the
#: `gpu_hours|duration_seconds` disjunction occurs 15 times across
#: `scripts/automod/landed_outcomes.py`, so those two patterns stayed green with the retired
#: sentence deleted outright (review attempt 2, finding 3) — a pattern drawn from a file's
#: vocabulary witnesses the file's existence, not the sentence's. Each entry now holds ONE
#: FINGERPRINT PER RETIRED CITATION, and the node asserts of every one of them that it names
#: one of the entry's SYMBOLS and matches the file EXACTLY ONCE. Both halves are load-bearing:
#: the symbol is what makes the fingerprint that sentence rather than a keyword, and a single
#: match is what stops it being the file's idiom. Measured over the current tree: 17
#: fingerprints, one per retired citation, each with exactly one match.
RETIRED_CITATIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...], dict[str, ...]]] = {
    "app/failure_ledger.py": (
        ("mark_failed", "quarantine"),
        (r"mark_failed\(\)`[\s\S]{0,80}?`quarantine\(\)`",
         r"quarantine\(\)`[\s\S]{0,140}?quarantined[\s\S]{0,140}?needs attention"),
        {"mark_failed": "poisoned", "quarantine": "needs attention"}),
    "app/routers/dashboard.py": (
        ("mark_failed",),
        # The sentence's claim is that the DEF owns the ceiling. That is checkable, and it
        # is what the span says: `if attempts >= cap:` inside `mark_failed`, with `cap`
        # coming from the `max_attempts` argument. A re-aim that moved the decision to the
        # caller would be a sentence about a function that does not exist, so the grounding
        # terms are the condition itself, not just the state name.
        (r"mark_failed`[\s\S]{0,60}?writes `poisoned`[\s\S]{0,80}?attempt "
         r"ceiling[\s\S]{0,80}?`queued`",),
        {"mark_failed": ("poisoned", "attempts >=", "max_attempts")}),
    "architecture/research-pipeline.md": (
        ("list_runs_joined",),
        (r"list_runs_joined`[\s\S]{0,80}?LEFT JOIN[\s\S]{0,60}?r\.queue_id = q\.id",),
        {"list_runs_joined": "LEFT JOIN"}),
    "architecture/threat-model.md": (
        ("claim_next",),
        (r"due-gates[\s\S]{0,60}?claim_next\(\)",),
        {"claim_next": "not_before"}),
    "architecture/workers-jobs.md": (
        ("_insert_run", "_RUN_INSERT_SQL"),
        (r"rows are written only by[\s\S]{0,60}?_insert_run\(\)",),
        {"_insert_run": "_RUN_INSERT_SQL", "_RUN_INSERT_SQL": "INSERT INTO runs"}),
    "eval/persistence_arms.py": (
        ("claim_next", "mark_running", "wm_get"),
        (r"claim_next\(\)`[\s\S]{0,30}?state=.claimed.[\s\S]{0,40}?mark_running"
         r"\(\)` [\s\S]{0,30}?state=.running.",
         r"wm_get`[\s\S]{0,60}?workers/queue[.]py[\s\S]{0,80}?hand-rolled"),
        {"claim_next": "state='claimed'", "mark_running": "state='running'",
         "wm_get": "watermarks"}),
    "scripts/automod/landed_outcomes.py": (
        ("run_rollup_by_source",),
        (r"summed into `gpu_hours` per task by[\s\S]{0,60}?run_rollup_by_source",
         r"divided by 3600 into `gpu_hours` per TASK by[\s\S]{0,80}?run_rollup_"
         "by_source",
         r"prints as `gpu_hours`, via[\s\S]{0,80}?run_rollup_by_source"),
        {"run_rollup_by_source": "gpu_hours"}),
    "scripts/step_conformance.py": (
        ("run_rollup_by_source",),
        (r"rollup bills as failures[\s\S]{0,80}?run_rollup_by_source[\s\S]{0,120}?"
         r"SUM\(status IN",),
        {"run_rollup_by_source": "SUM(status IN ('failed','interrupted'))"}),
    "tests/test_egress_window_unit.py": (
        ("depth_by_source",),
        (r"A NUMBER[\s\S]{0,240}?def depth_by_source[\s\S]{0,60}?at or above it",),
        {"depth_by_source": "GROUP BY source, state"}),
    "tests/test_fleet_watchdog.py": (
        ("claim_next",),
        (r"claim_next`[\s\S]{0,80}?claimed_at[\s\S]{0,120}?workers/queue[.]py",),
        {"claim_next": "claimed_at"}),
    "tests/test_landed_outcomes_by_author.py": (
        ("run_rollup_by_source",),
        (r"calls `gpu_hours`[\s\S]{0,80}?run_rollup_by_source[\s\S]{0,120}?"
         "no GPU-seconds",),
        {"run_rollup_by_source": "gpu_hours"}),
    "tests/test_persistence_window_unit.py": (
        ("depth_by_source",),
        (r"depth_by_source\(\)` in `workers/queue[.]py`[\s\S]{0,140}?GROUP BY "
         r"source, state[\s\S]{0,80}?does not exclude the caller",),
        {"depth_by_source": "GROUP BY source, state"}),
    "agent-services/systemd/lloyd-egress-window.timer": (
        ("depth_by_source",),
        (r"depth_by_source\(\)`[\s\S]{0,40}?workers/queue[.]py[\s\S]{0,80}?"
         "counts the window's own row",),
        {"depth_by_source": "COUNT("}),
}


def _queue_def_span(sym: str) -> str | None:
    """The source of `workers/queue.py`'s definition of `sym`, or None if it has none.

    Methods and module-level constants both count: `_RUN_INSERT_SQL` is the SQL the
    retirement points at as much as `_insert_run()` is, and it is an assignment, not a def.
    """
    tree = ast.parse(QUEUE.read_text(encoding="utf-8"))
    lines = QUEUE.read_text(encoding="utf-8").splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == sym:
            return "".join(lines[node.lineno - 1:node.end_lineno])
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == sym for t in node.targets):
            return "".join(lines[node.lineno - 1:node.end_lineno])
    return None


def test_every_retired_citation_kept_its_mechanism_and_named_the_symbol_it_described():
    """#2971 clause 4: thirteen files lost a number and kept the sentence it explained.

    Three assertions per file, each against a different way the retirement could have been
    done badly. The file holds no numeric citation. It names the symbol in a POINTER form —
    backtick-quoted, `WorkQueue.sym`, or `sym()` — because the review caught that a bare
    `sym in text` is satisfied by any incidental use of the word, which is not a pointer.
    And the mechanism phrase the citation was attached to survives, in the file AND inside
    the definition it now names: a sentence that points at a real function for a claim that
    function does not make is exactly how the retired citation read: it pointed at the
    import block at the top of queue.py for a poisoned/quarantined distinction that lives in
    `mark_failed()` and `quarantine()`, some way down. The claim is therefore checked against
    the definition's own text, not against a name that merely exists.
    """
    for rel, (symbols, mechanisms, grounding) in RETIRED_CITATIONS.items():
        path = REPO / rel
        assert path.is_file(), f"{rel} is gone; the pair list needs updating, not deleting"
        text = path.read_text(encoding="utf-8")
        if rel == Path(__file__).relative_to(REPO).as_posix():
            # This file's OWN DATA BLOCK is not prose: a fingerprint stored in the list would
            # otherwise match its own source text and be counted a second time, so every
            # self-entry would report "matched twice" for a reason that has nothing to do with
            # the sentence it guards. The block is located by AST, never by a line number —
            # which would be exactly the thing this item is retiring.
            decl = next(n for n in ast.parse(text).body
                        if isinstance(n, ast.AnnAssign)
                        and getattr(n.target, "id", "") == "RETIRED_CITATIONS")
            text = text.replace(ast.get_source_segment(text, decl), "")
        assert not _QUEUE_NUMERIC.search(text), \
            f"{rel} is back to citing workers/queue.py by line number"
        for sym in symbols:
            assert (f"`{sym}" in text or f".{sym}" in text or f"{sym}()" in text), (
                f"{rel} does not point at `{sym}` in any pointer form; the sentence lost "
                "the symbol its retired citation was about")
            span = _queue_def_span(sym)
            assert span is not None, (
                f"{rel} now points at `{sym}`, which workers/queue.py does not define")
            for term in ([grounding[sym]] if isinstance(grounding[sym], str)
                         else grounding[sym]):
                assert term in span, (
                    f"{rel} attributes a claim to `{sym}` that its own definition in "
                    f"workers/queue.py does not make — {term!r} is not in that span")
        for pattern in mechanisms:
            assert any(sym in pattern for sym in symbols), (
                f"{rel}'s fingerprint {pattern!r} names none of its entry's symbols "
                f"{symbols}. A bare term is not a fingerprint of the retired sentence: "
                "`gpu_hours` occurs 8 times in tests/test_landed_outcomes_by_author.py, and "
                "the pattern built from it alone stayed green with the sentence deleted "
                "(review attempt 2), because it was matching the file's vocabulary.")
            found = list(re.finditer(pattern, text, re.S))
            assert len(found) == 1, (
                f"{rel}: fingerprint {pattern!r} matched {len(found)} times, so the node "
                "cannot tell which sentence the retired citation sat in — either it is file "
                "vocabulary rather than that sentence, or the sentence has been duplicated.")


def test_the_guard_illustrates_its_own_ban_and_the_timer_now_points_by_symbol():
    """#2971 clause 5: the ban applies to this file's prose and to the timer's pointer.

    Two self-references had to go, because a guard that quotes the forbidden form is a
    corpus that never goes clean: this file's `_resolve_pool_query_pointer` docstring
    illustrated a NUMBER with a real line, and the node above it described the OLD egress
    node as asserting a literal line, quoted. Both now read
    ``workers/queue.py:<n>`` — a placeholder, which is the point: the ban's reader cannot
    match it, which is asserted here rather than assumed, so the placeholder form is a
    spelling that is known safe and not a coincidence.

    The timer is the pointer the ban had to be shaped around. `_QUEUE_SYMBOL` binds a
    ``symbol()`` to the file across at most ten non-digit characters, so the comment reads
    The binding window is ten characters, so it reads "`depth_by_source()` at
    workers/queue.py" and not "the pool-status query at workers/queue.py" — the latter is a
    mention the resolver rejects as neither a line nor a bound symbol,
    which is the resolver's own red path and would have turned a comment tidy-up into a
    broken unit guard. The three named units are then asserted to resolve, as symbols.
    """
    #: Read off the two functions' own `__doc__`, not off a count in the file: an earlier
    #: draft asserted `own.count(placeholder) >= 2`, which this node's own docstring and
    #: assert literal already satisfy, so restoring the two real numbers elsewhere in them
    #: could not have reddened it. Each docstring must carry the placeholder AND hold no
    #: numeric citation, so putting a number back in either one is a failure.
    window_node = (
        test_the_window_units_point_at_the_live_pool_query_and_refuse_to_claim_the_hour_avoids_rounds
    )
    for fn in (_resolve_pool_query_pointer, window_node):
        doc = fn.__doc__ or ""
        assert "workers/queue.py:<n>" in doc, (
            f"{fn.__name__}'s docstring no longer illustrates the banned form with the "
            "<n> placeholder — if the real number came back, this is where it shows")
        assert not _QUEUE_NUMERIC.search(doc), (
            f"{fn.__name__}'s docstring cites workers/queue.py by line number again")
    #: A POSITIVE CONTROL on the regex the two assertions above lean on. Asking whether the
    #: placeholder literal itself looks numeric is a question about a string this file types,
    #: which is why the first draft of it was called unfalsifiable; what has to be true is that
    #: `_QUEUE_NUMERIC` matches the shape the placeholder stands for, because every
    #: `assert not _QUEUE_NUMERIC.search(...)` in this file — the pair list, the unit comments,
    #: the whole corpus — is only as strong as that match. Break the regex and the whole ban
    #: greens, so the shape it must catch is asserted here in the one file that owns it.
    assert _QUEUE_NUMERIC.search(f"workers/queue.py:{SYNTHETIC_CITED_LINE}"), (
        "_QUEUE_NUMERIC no longer matches the form it exists to catch, so every absence "
        "assertion in this file is a green light by default")
    assert not _QUEUE_NUMERIC.search("workers/queue.py:<n>"), (
        "the placeholder form the ban's prose uses has stopped being a placeholder — "
        "something has put a number back into it")

    qlines = QUEUE.read_text(encoding="utf-8").splitlines()
    timer_c = _comments(TIMER)
    bound = _QUEUE_SYMBOL.findall(timer_c)
    assert "depth_by_source" in bound, (
        "the timer's pointer must keep `depth_by_source()` bound to workers/queue.py: "
        f"the resolver reads only a bound symbol or a line number, and found {bound}")
    assert _resolve_pool_query_pointer(timer_c, qlines, src=TIMER.name), \
        "the timer's pointer stopped resolving against workers/queue.py"
    for name in WINDOW_UNITS_WITH_POOL_POINTER:
        unit = UNIT_DIR / name
        assert _QUEUE_NUMERIC.search(_comments(unit)) is None, (
            f"{name} is back to a numeric pointer, which the resolver can only re-read "
            "until the next DDL above the query")


#: Where a user-level systemd manager looks for a unit given only its name. Conventional and
#: fixed (`$XDG_CONFIG_HOME/systemd/user`), which is what makes it readable from a test — see
#: the node below for why the manager itself cannot be asked.
USER_UNIT_DIR = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "systemd" / "user"


def _loaded_by_name(name: str) -> Path | None:
    """The file a user systemd would load for the unit NAME, or None where it is not installed."""
    path = USER_UNIT_DIR / name
    return path if path.is_file() else None


def test_the_unit_systemd_loads_by_name_links_to_the_tracked_one_and_fires_the_pinned_cadence():
    """#2971's post-landing seam: the consumer of a unit comment is systemd, not pytest.

    The review's fourth finding, and the one thing no assertion inside the repository can show
    on its own: every node above reads `agent-services/systemd/` through this suite's own view,
    while what ACTS on those lines is a manager that resolves a unit name through its own load
    path. `#2555` is the precedent for what an unheld boundary costs — a node read unit settings
    from the repository and passed for weeks while systemd used a shadowed copy under
    `~/.config/systemd/user/`, so the settings the guard read and the settings the machine used
    were different files. Three things are therefore asked of the file at the load path:

    * it is a SYMLINK into an `agent-services/systemd/` directory, not a private copy. A copy is
      the shadowing case, and it is how a retired numeric citation could survive out of sight of
      a corpus node that walks the repository.
    * the cadence in it is the cadence in the file these guards read. Inside a review worktree
      those are genuinely two files — the edit is in the worktree, the load path points at the
      live checkout — so the comparison is not reading one file twice.
    * systemd's own parser, `systemd-analyze calendar`, turns that cadence into a Monday. The
      weekly cadence is what SETUP.md and the sibling unit claim, and a parser the machine
      agrees with is the only reading of a calendar expression that counts.

    What this node deliberately does NOT do is ask the manager. `systemctl` is refused inside the
    suite by `tests/conftest.py::_no_live_service_control_in_tests` (`DANGEROUS_SUBPROCESS_COMMANDS`,
    #1853) — read-only properties included, because the guard decides on argv[0] and a test that
    can read state is one `--user restart` away from acting on the box — and a `monkeypatch`
    stub of `systemctl show` would assert only its own return value, which is not a seam. So the
    identity half is taken at the load path and the parse half through systemd's parser, and the
    unit's live `ActiveState` and first real firing stay with the item's post-landing owed check,
    which is the only place they can be observed. `systemd-analyze verify` over these same files
    is already pinned by
    `test_systemd_itself_still_parses_every_window_unit_after_the_comment_edits`; that node is
    the grammar half, this one is the identity-and-cadence half.
    """
    installed = {u.name: _loaded_by_name(u.name) for u in (TIMER, SIBLING_SERVICE)}
    present = sorted(n for n, p in installed.items() if p is not None)
    if not present:
        pytest.skip(f"neither unit is installed under {USER_UNIT_DIR}, so there is no load path "
                    "to hold to the repository on this box")

    for name in present:
        path = installed[name]
        assert path.is_symlink(), (
            f"{name} is installed as a regular file at {path}, not a link into the repository — "
            "the shadowing shape that spent #2555: an edit to agent-services/systemd/ can never "
            f"reach the running system, and a retired citation could live in {path} unseen by "
            "the corpus node, which walks the repository")
        target = path.resolve()
        assert target.parent.name == "systemd" and target.parent.parent.name == "agent-services", (
            f"{name} links from {path} to {target}, which is not an agent-services/systemd "
            "directory, so the file these guards read is not the file systemd acts on")

    def setting(path: Path, key: str) -> str | None:
        m = re.search(rf"(?m)^{key}=(.+)$", path.read_text(encoding="utf-8"))
        return m.group(1).strip() if m else None

    repo_cal = setting(TIMER, "OnCalendar")
    assert repo_cal is not None, "the repo timer lost its OnCalendar line"
    loaded_timer = installed.get(TIMER.name)
    if loaded_timer is None:
        pytest.skip(f"{TIMER.name} is not installed, so the cadence systemd would fire cannot "
                    "be compared with the file these guards read")
    loaded_cal = setting(loaded_timer, "OnCalendar")
    assert repo_cal == loaded_cal, (
        f"the unit at systemd's load path fires on OnCalendar={loaded_cal!r} while the file "
        f"these guards read says {repo_cal!r}: the weekly cadence pinned in "
        "tests/test_persistence_window_unit.py and SETUP.md is not the cadence the machine has")
    import shutil
    analyze = shutil.which("systemd-analyze")
    if analyze is None:
        pytest.skip("systemd-analyze is absent, so the cadence cannot be asked of systemd's "
                    "own parser on this box")
    parsed = subprocess.run([analyze, "calendar", repo_cal],
                            capture_output=True, text=True, timeout=120)
    assert parsed.returncode == 0, (
        f"systemd cannot parse the cadence it is being handed, OnCalendar={repo_cal!r}: "
        f"{parsed.stderr.strip()[:200]}")
    assert re.search(r"(?i)\bmon\b", parsed.stdout), (
        f"systemd's own parse of OnCalendar={repo_cal!r} does not name a Monday, so the weekly "
        f"cadence the units and SETUP.md claim is not what will fire: {parsed.stdout[:300]}")
