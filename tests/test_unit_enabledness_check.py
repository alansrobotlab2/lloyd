"""#1891 — scripts/maintenance/check-unit-enabledness.sh, the guard on a placed-but-disabled timer.

A unit file that is installed but never enabled was invisible to every route this
repo owns when this guard was written: `install-services.sh` *echoed* the
`systemctl --user enable` lines instead of running them (#2774 closed that — the
script now executes the arming itself, and tests/test_install_services_arms_units.py
holds it), `promote.py` only copies a unit and daemon-reloads, and
`scripts/service_health_check.py` is supervisor-only —
`git grep -n "is-enabled" -- scripts app tests` returned 0 hits before this script
(positive control: 25 files match `systemctl` in the same scope). So
`lloyd-cert-renew.timer` reads `disabled` on goliath with the other four repo timers
`enabled`, and the enable has been handed forward from #1727 to #1793 to #1891 with
nothing noticing.

The script shells out, so these nodes cross that process boundary through a stub
`systemctl` written onto the front of `PATH` and a `--unit-dir` the test owns. The
stub mirrors the real command's semantics — it prints the unit's state and exits 0
only for `enabled` — and records every argv, which is how a check whose job is to
NOT ask about `.service` units and live-but-untracked timers can be pinned at all.
The host's `systemctl` is never reached: every run asserts the stub was consulted,
so a script that resolved an absolute path instead would fail here rather than
quietly grade whatever box it happens to run on — the same reason
tests/test_cert_renew_units.py declines to pin enabledness, and why no node in this
file asserts anything about the machine running it. The one host-adjacent read that
exists, `USER_UNIT_DIR`, asks only that a planted name is not a unit on this box — a
precondition on a string, never a verdict about what a unit here answers.

#2702 widened that boundary rather than adding a new one: the guard asks `systemctl --user
show` for arming evidence, and the same stub answers it (see STUB for the two tables and
the asymmetric defaults). #2943 widened the same seam again rather than adding another:
that question is now asked of EVERY timer, not only of one whose `is-enabled` verdict
already failed, because the verdict that needs the reading is the one that PASSES — an
`enabled` row over `ActiveState=inactive` with no fire instant is an installed, linked,
never-armed job, which is what this box's `lloyd-egress-window.timer` was while the guard
printed `checked 8 timers: 8 enabled, 0 not enabled` and exited 0. So the node #2702
shipped to pin the old cost limit
(`test_the_evidence_is_asked_only_of_the_timers_that_already_failed`) was REVISED, not
kept: it pinned `show == []` on a green run, which is exactly the green #2943 exists to
break, and it now pins the cost that did not grow — one `show` per timer, after its
`is-enabled` answer, never a third call for the unit. Nothing here changed the module's
refusal: the evidence a node can pin is
what the guard prints given a stated `show` answer, never what this box's timer
actually reports — `never-armed` is pinned by telling the stub the property is empty,
which is the only way a marker can appear at all.

Run: .venvs/lloyd/bin/python -m pytest tests/test_unit_enabledness_check.py
"""
import ast
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "maintenance" / "check-unit-enabledness.sh"
REPO_UNIT_DIR = ROOT / "agent-services" / "systemd"
CERT_UNITS_TEST = ROOT / "tests" / "test_cert_renew_units.py"
REPO_TIMERS = sorted(p.name for p in REPO_UNIT_DIR.glob("*.timer"))
CERT_TIMER = "lloyd-cert-renew.timer"
#: #2864 tracked the graph backup pair, so this guard's own fleet gained a member.
GRAPH_TIMER = "lloyd-graph-backup.timer"
#: The timer #2943 exists to arm: linked on disk, `ActiveState=inactive`,
#: `LastTriggerUSec=` empty, absent from `systemctl --user list-timers --all`, and
#: `journalctl --user -u lloyd-egress-window.service` → `-- No entries --`. The nodes below
#: plant the never-armed signature on a unit the guard really enumerates rather than on an
#: invented name, so if this unit ever leaves `agent-services/systemd/` the precondition
#: assert trips instead of the node quietly grading a timer nothing checks.
EGRESS_TIMER = "lloyd-egress-window.timer"
#: A name NO unit directory holds — planted here rather than borrowed from the box.
#: The node below needs a timer that is live-but-untracked, i.e. one the check must
#: refuse to ask about; `lloyd-graph-backup.timer` played that role until #2864
#: tracked the pair, and any name taken from box state afterwards is a name some rebuild
#: changes. An invented name is the only one that cannot quietly become real, and it
#: keeps the rule stated as a rule — membership over the repo set, never set equality
#: against `~/.config/systemd/user/timers.target.wants/`, which is what
#: `tests/test_data_home.py::_unresolved` rules for the cutover fleet too.
LIVE_TIMER_WITHOUT_REPO_UNIT = "planted-untracked-timer.timer"
#: Read by the one precondition node below, and by nothing else: a precondition on the
#: planted name is not a verdict about this machine's units, which stays the refusal
#: this module's docstring states.
USER_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"

STUB = """#!/usr/bin/env python3
import json, os, sys

# Stand-in for `systemctl`, with the real command's semantics: `is-enabled <unit>`
# prints the state and exits 0 only for `enabled`; any other state prints and exits
# non-zero. Since #2702 the guard asks a second question — `show -l -p LastTriggerUSec
# -p ActiveState <unit>` — and since #2943 it asks it of EVERY timer it checks rather than
# only of one whose `is-enabled` verdict failed, because the never-armed verdict is read off
# the rows that pass. So this stub answers that form too. It prints the properties SORTED BY
# NAME, which for
# the guard's request order is the reverse of it, because that is what the real command
# does on this box: `show -p LastTriggerUSec -p ActiveState` answers ActiveState first.
# Answering out of request order is what turns the guard's by-key parsing from an
# assumption into something a node can watch fail. Every argv is appended to STUB_LOG so
# a test can assert what was ASKED, not just what came back.
#
# STUB_STATES maps a unit name to the `is-enabled` answer; STUB_SHOW maps it to the
# properties `show` should print, or to the string "error" for a call that fails outright
# (non-zero exit, a message on stderr, nothing on stdout). Both tables have two lookups
# beyond an exact hit: a `*.suffix` key matches any unit ending in that suffix, and
# `__default__` answers everything else.
#
# The two defaults are NOT symmetric, on purpose. An unlisted unit answers `is-enabled`
# `enabled` — the pass case, so a node that cares about one unit need not describe the
# rest. An unlisted unit answers `show` as an ARMED timer (ActiveState=active plus a
# real-looking instant), so a node that expects `never-armed` or `no-answer` can only get
# there by asking for a no-value property: no default can produce a marker, which is what
# keeps a marker assertion from passing on a stub that was never configured. Since #2943
# that default carries a second job: the guard asks the question of EVERY timer now, so a
# node describing one unit must still see an all-green run for the rest — the ARMED default
# is what makes an un-described timer PASS rather than read as never-armed, and a default
# changed to a no-value property would turn every green node in this file red at once.
argv = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(json.dumps(argv) + "\\n")

states = json.loads(os.environ.get("STUB_STATES", "{}"))
shown = json.loads(os.environ.get("STUB_SHOW", "{}"))

ARMED = {"ActiveState": "active", "LastTriggerUSec": "Mon 2026-10-12 06:23:01 PDT"}


def lookup(table, unit, fallback):
    if unit in table:
        return table[unit]
    for key, value in table.items():
        if key.startswith("*") and unit.endswith(key[1:]):
            return value
    return table.get("__default__", fallback)


if argv[:2] == ["--user", "is-enabled"] and len(argv) == 3:
    state = lookup(states, argv[2], "enabled")
    print(state)
    raise SystemExit(0 if state == "enabled" else 1)

if argv[:2] == ["--user", "show"]:
    props, unit = [], ""
    rest = iter(argv[2:])
    for arg in rest:
        if arg in ("-p", "--property"):
            props.append(next(rest, ""))
        elif arg.startswith("--property="):
            props.append(arg.split("=", 1)[1])
        elif arg in ("-l", "--full", "--no-pager"):
            continue
        else:
            unit = arg
    answer = lookup(shown, unit, ARMED)
    if answer == "error":
        sys.stderr.write("stub systemctl: show failed for %r\\n" % (unit,))
        raise SystemExit(4)
    for prop in sorted(props):
        print("%s=%s" % (prop, answer.get(prop, "")))
    raise SystemExit(0)

sys.stderr.write("stub systemctl: unstubbed invocation %r\\n" % (argv,))
raise SystemExit(2)
"""


def _run(unit_dir: Path, states: dict, tmp_path: Path, tag: str = "run",
         require_query: bool = True, show: dict | None = None):
    """Execute the script for real, with a stub `systemctl` first on PATH.

    `states` answers `is-enabled` and `show` answers the arming-evidence `show` call
    (#2702); both default to the stub's own defaults, which are asymmetric on purpose —
    see STUB. Returns (CompletedProcess, combined stdout+stderr lines, argv the stub saw).
    """
    bin_dir = tmp_path / f"bin-{tag}"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "systemctl"
    stub.write_text(STUB, encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / f"queries-{tag}.jsonl"

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["STUB_LOG"] = str(log)
    env["STUB_STATES"] = json.dumps(states)
    env["STUB_SHOW"] = json.dumps(show or {})

    proc = subprocess.run([str(SCRIPT), "--unit-dir", str(unit_dir)],
                          capture_output=True, text=True, env=env, cwd=str(ROOT))
    queries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] \
        if log.exists() else []
    assert queries or not require_query, (
        "the script asked `systemctl` nothing at all: it never reached the stub on "
        "PATH, so this run measured no enabledness of anything")
    out = [line.strip() for line in (proc.stdout + proc.stderr).splitlines()
           if line.strip()]
    return proc, out, queries


def _unit_names(tmp_path: Path, *names: str) -> Path:
    """A unit dir holding exactly these unit files."""
    unit_dir = tmp_path / ("units-" + "-".join(names))
    unit_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (unit_dir / name).write_text("[Unit]\nDescription=stub\n", encoding="utf-8")
    return unit_dir


def _denominator(out: list[str]) -> int:
    """The count the total line names. Clause 1 makes it part of the contract, so a
    run with no total line is a failure here rather than a missing fact."""
    totals = [line for line in out if line.startswith("checked ")]
    assert len(totals) == 1, f"expected exactly one total line, got {totals}"
    match = re.fullmatch(r"checked (\d+) timers?:.*", totals[0])
    assert match, f"total line does not name how many timers were checked: {totals[0]}"
    return int(match.group(1))


def _fail_lines(out: list[str]) -> list[str]:
    return [line for line in out if line.startswith("FAIL ")]


def _fail_line_for(out: list[str], unit: str) -> str:
    """The exactly-one FAIL line naming `unit`.

    Every clause below is asserted against a line, so each node first has to prove the
    line exists and is unique: an `in` against a comprehension that came back empty is a
    green that proves nothing about a hint nobody was shown.
    """
    named = [line for line in _fail_lines(out) if unit in line]
    assert len(named) == 1, (
        f"expected exactly one FAIL line naming {unit}, got {named} "
        f"in {_fail_lines(out)}")
    return named[0]


def _row_lines(out: list[str]) -> list[str]:
    """The `unit<TAB>verdict` rows — the lines the health route parses into graded rows.

    The shape is `scripts/service_health_check.py`'s own `_UNITS_ROW_RE`
    (`^(\\S+\\.timer)\\t(\\S+)$`), so a line that gained any trailing text stops being a row
    here exactly where it stops being one on the health route.
    """
    return [line for line in out if re.fullmatch(r"\S+\.timer\t\S+", line)]


def _asked_about(queries: list, verb: str) -> list[str]:
    """The units the stub saw under `systemctl --user <verb> ...`, in call order."""
    return [q[-1] for q in queries if q[1] == verb]


def _queried_units(queries: list) -> list[str]:
    """Every unit the stub was asked about, one entry per query, in call order.

    These nodes used a positional read (`query[2]`), which was the unit name for every form
    the guard used to issue. #2943 made it ask `show` of EVERY timer rather than only of a
    miss, and that argv carries its `-l -p …` flags before the name, so the unit of a query
    is its LAST word for both verbs. The set-valued sites keep their exact meaning through
    this helper; the count-valued ones state how many queries per timer the run now costs —
    the number #2943 moved, so each of those sites writes it out rather than hiding it here.
    """
    return [q[-1] for q in queries]


def test_the_script_is_executable_and_prints_a_verdict_line_per_repo_timer(tmp_path):
    """Clause 1: every `agent-services/systemd/*.timer` gets its `is-enabled` verdict
    on a line of its own, and a total line names how many were checked.

    Run over the REAL repo unit dir (repo content, not host state) against a stub
    that answers `enabled`, so the expected denominator is whatever the tree
    actually tracks rather than a number this file memorised.

    The query-log assertion at the end counted ONE query per timer while `is-enabled` was
    the only question this guard asks. Since #2943 every timer also gets its one arming
    `show`, so the count it states is two per timer now — the same property (every repo
    timer asked, nothing outside the repo set asked) with the number the change moved. The
    read is `REPO_TIMERS * 2`, so a third query for one unit and a skipped unit both fail.
    """
    assert SCRIPT.exists(), SCRIPT
    assert SCRIPT.stat().st_mode & 0o111, f"{SCRIPT} is not executable"
    assert REPO_TIMERS, "the repo tracks no .timer units at all, so this node is vacuous"

    proc, out, queries = _run(REPO_UNIT_DIR, {}, tmp_path)

    assert proc.returncode == 0, f"all-enabled should pass:\n{proc.stdout}{proc.stderr}"
    assert _denominator(out) == len(REPO_TIMERS), out
    for name in REPO_TIMERS:
        own = [line for line in out if line.split()[:1] == [name]]
        assert own, f"{name} has no line of its own in:\n{out}"
        assert own[0].split()[1] == "enabled", own[0]
    assert sorted(_queried_units(queries)) == sorted(REPO_TIMERS * 2), (
        f"the run must ask exactly two queries per repo timer — its `is-enabled`, plus the "
        f"`show` every timer gets since #2943 — and never touch a unit outside the repo "
        f"set: {queries}")


def test_every_timer_answering_enabled_exits_zero(tmp_path):
    """Clause 2, first half: all-`enabled` is the pass case — and the denominator
    tracks the set it was given, which is what stops it being a remembered number."""
    unit_dir = _unit_names(tmp_path, "alpha.timer", "beta.timer", "gamma.timer")

    proc, out, queries = _run(unit_dir, {}, tmp_path)

    assert proc.returncode == 0, f"{proc.returncode}:\n{proc.stdout}{proc.stderr}"
    assert _denominator(out) == 3, out
    assert not _fail_lines(out), out
    assert _asked_about(queries, "is-enabled") == \
        ["alpha.timer", "beta.timer", "gamma.timer"], queries
    assert set(_queried_units(queries)) == {"alpha.timer", "beta.timer", "gamma.timer"}, \
        queries


def test_one_disabled_timer_exits_non_zero_and_names_only_that_unit(tmp_path):
    """Clause 2, second half: one `disabled` answer among enabled ones must fail, and
    name that unit on a line that names nothing else — a fleet-wide "something is
    wrong" line would leave an operator to bisect five units by hand."""
    unit_dir = _unit_names(tmp_path, "alpha.timer", "beta.timer", "gamma.timer")

    proc, out, queries = _run(unit_dir, {"beta.timer": "disabled"}, tmp_path)

    assert proc.returncode != 0, f"a disabled timer passed:\n{proc.stdout}"
    assert _denominator(out) == 3, out
    naming_beta = [line for line in out if "beta.timer" in line]
    assert any(line.startswith("FAIL ") for line in naming_beta), naming_beta
    assert not any(line.startswith("FAIL ")
                   and ("alpha.timer" in line or "gamma.timer" in line)
                   for line in out), out
    assert _asked_about(queries, "is-enabled").count("beta.timer") == 1, (
        f"beta was asked its enabledness question more than once: {queries}")


def test_a_service_answering_static_is_never_asked_and_cannot_fail_it(tmp_path):
    """Clause 3: the assertion covers `.timer` units only.

    `lloyd-cert-renew.service` reads `static` on this box because only its timer
    ever starts it, so a check that swept `.service` files in would report a defect
    that is not one. Pinned from both sides: the stub answers `static` for every
    service and the run still exits 0 with no service in the query log, and the same
    stub answering `static` for the TIMER fails — so the pass above is the check
    declining to ask, not the check ignoring every verdict.
    """
    assert (REPO_UNIT_DIR / "lloyd-cert-renew.service").is_file(), (
        "the co-located service this clause is about has moved")

    proc, out, queries = _run(REPO_UNIT_DIR, {"*.service": "static"}, tmp_path,
                              tag="services")

    assert proc.returncode == 0, f"a .service verdict failed the check:\n{proc.stdout}"
    asked = set(_queried_units(queries))
    assert not any(unit.endswith(".service") for unit in asked), asked
    assert CERT_TIMER in asked, f"{CERT_TIMER} was never asked about: {asked}"

    witness, witness_out, _ = _run(REPO_UNIT_DIR,
                                   {"*.service": "static", CERT_TIMER: "static"},
                                   tmp_path, tag="witness")
    assert witness.returncode != 0, (
        f"a .timer answering static passed, so the exit 0 above proves nothing: "
        f"{witness_out}")


def test_a_live_timer_with_no_repo_unit_cannot_fail_the_check(tmp_path):
    """Clause 4, first half: one-directional over the repo set.

    A user timer can be armed on a box with no tracked file behind it — that is what
    `lloyd-graph-backup.timer` was until #2864 tracked it, and
    `freeclip-a2dp-fix.service` is its live-only shape on this box today — so a check
    that walked `timers.target.wants/`, or compared it to the repo set, would fail on a
    unit this repository has no say in. Box state no longer supplies such a timer to
    prove it with, so the name here is planted, and the teeth move with it: the first run
    keeps the original statement (a name the stub knows as `disabled` is never asked, and
    the run passes), and the SECOND run is what makes the node trippable again — a unit
    dir holding ONLY the planted timer, over a box whose live user manager has eight
    timers. This box's `timers.target.wants/` is a strict subset of `REPO_TIMERS` (seven
    of the eight), so a guard that unioned the live set into the checked set would look
    identical in the first run; in the second it asks about nine units where this node
    demands exactly one. The two preconditions are what keep the name a name — if some
    future unit file ever spelled itself this, the node would be asserting that a checked
    unit is unchecked, which is a green worth nothing.

    The planted name is the only host-adjacent read this module makes, and it is a
    precondition on a string, not a verdict about the machine: no node here asserts
    what any unit on this box answers.
    """
    assert LIVE_TIMER_WITHOUT_REPO_UNIT not in REPO_TIMERS, (
        f"{LIVE_TIMER_WITHOUT_REPO_UNIT} is in {REPO_UNIT_DIR}, so this node is "
        f"asserting a repo timer is outside the repo set: {REPO_TIMERS}")
    assert not (USER_UNIT_DIR / LIVE_TIMER_WITHOUT_REPO_UNIT).exists(), (
        f"{LIVE_TIMER_WITHOUT_REPO_UNIT} exists under {USER_UNIT_DIR}, so the planted "
        "name has become box state and this node would be grading a live unit")

    proc, out, queries = _run(REPO_UNIT_DIR,
                              {LIVE_TIMER_WITHOUT_REPO_UNIT: "disabled"},
                              tmp_path, tag="onedir")

    assert proc.returncode == 0, f"{proc.returncode}:\n{proc.stdout}{proc.stderr}"
    asked = set(_queried_units(queries))
    assert LIVE_TIMER_WITHOUT_REPO_UNIT not in asked, asked
    assert asked == set(REPO_TIMERS), f"{asked ^ set(REPO_TIMERS)} is off-set"

    # The trippable half: the checked set is the directory the script was handed, never
    # that directory unioned with whatever the box's user manager has armed. The planted
    # timer is the ONLY file in this dir, and the stub answers every timer `enabled`, so
    # one row and exit 0 is the whole contract here.
    lone_dir = tmp_path / "lone-unit-dir"
    lone_dir.mkdir()
    (lone_dir / LIVE_TIMER_WITHOUT_REPO_UNIT).write_text(
        "[Timer]\nOnCalendar=*-*-* 05:30:00\n", encoding="utf-8")

    lone, lone_out, lone_queries = _run(lone_dir, {}, tmp_path, tag="loneplanted")

    assert lone.returncode == 0, f"{lone.returncode}:\n{lone.stdout}{lone.stderr}"
    lone_asked = set(_queried_units(lone_queries))
    assert lone_asked == {LIVE_TIMER_WITHOUT_REPO_UNIT}, (
        f"the checked set must be the unit dir it was handed and nothing else; "
        f"{lone_asked ^ {LIVE_TIMER_WITHOUT_REPO_UNIT}} came from somewhere else — a "
        f"guard that reads timers.target.wants/ fails here on any box")
    assert _denominator(lone_out) == 1, lone_out


def test_a_timer_tracked_into_the_repo_joins_the_checked_set_and_fails_nothing(tmp_path):
    """#2864: a unit entering `agent-services/systemd/` enters this guard's set, and
    enters it as one row — not as a new failure.

    `lloyd-graph-backup.timer` has armed this box since 2026-09-03 (the live file's own
    mtime) with no tracked file behind it, which is precisely the period over which this
    guard printed nothing about it. Tracking the pair
    has to move the denominator (pinned against `len(REPO_TIMERS)` by
    `test_the_script_is_executable_and_prints_a_verdict_line_per_repo_timer`, so the
    count is never memorised here) AND give the timer its own `unit<TAB>verdict` row:
    the health route parses those rows into graded rows, and a unit the script asks
    about without printing a row for is a unit whose verdict is dropped on the floor.
    The green half is load-bearing: over a stub that answers every timer `enabled` the
    run must exit 0 with NO FAIL line, so a unit joining the fleet cannot manufacture a
    red. What the box itself answers for that timer — `systemctl --user is-enabled
    lloyd-graph-backup.timer` — stays host state and unasserted, as everywhere else in
    this file.
    """
    assert GRAPH_TIMER in REPO_TIMERS, (
        f"{GRAPH_TIMER} is not tracked under {REPO_UNIT_DIR}, so #2864's tracking has "
        f"regressed and this guard is not grading it: {REPO_TIMERS}")

    proc, out, queries = _run(REPO_UNIT_DIR, {}, tmp_path, tag="graphjoin")

    assert proc.returncode == 0, f"{proc.returncode}:\n{proc.stdout}{proc.stderr}"
    rows = [line for line in _row_lines(out) if line.split("\t")[0] == GRAPH_TIMER]
    assert rows == [f"{GRAPH_TIMER}\tenabled"], (
        f"{GRAPH_TIMER} must print exactly one row of its own: {rows}")
    assert _asked_about(queries, "is-enabled").count(GRAPH_TIMER) == 1, queries
    assert not _fail_lines(out), out
    assert _denominator(out) == len(REPO_TIMERS), out


def test_a_unit_dir_holding_no_timers_fails_instead_of_passing_vacuously(tmp_path):
    """Clause 4, second half: a glob matching zero timers is a failure.

    Without `nullglob` the glob would stay the literal `*.timer` and be asked about
    as a unit name; with it and no guard the loop body never runs and an empty set
    exits 0. The witness is the same flag pointed at a dir holding one enabled timer:
    `--unit-dir` itself is not what fails here, the empty set is.
    """
    empty = tmp_path / "units-empty"
    empty.mkdir()

    proc, out, _ = _run(empty, {}, tmp_path, tag="empty", require_query=False)

    assert proc.returncode != 0, f"an empty unit dir passed:\n{proc.stdout}"
    assert _denominator(out) == 0, out
    assert _fail_lines(out), out

    one = _unit_names(tmp_path, "solo.timer")
    witness, witness_out, _ = _run(one, {}, tmp_path, tag="solo")
    assert witness.returncode == 0, f"{witness.returncode}: {witness_out}"
    assert _denominator(witness_out) == 1, witness_out


def test_the_cert_renew_units_docstring_blames_the_route_not_the_operator():
    """Clause 5, pinned as a doc claim (the pattern at
    tests/test_code_graph_doc_claims.py).

    "Installing and enabling them is host state and stays with the operator" is the
    sentence that let a placed-but-disabled timer survive three handoffs: it made the
    gap sound assigned, so no run ever re-checked it. Enabling is still not this
    module's to assert — a box without the units linked must not go red here — but
    the unpinned reason is now the SETUP.md enable route plus the check this round
    adds, and this module still asserts no host state at all.
    """
    src = CERT_UNITS_TEST.read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src)) or ""
    # Flattened so a rewrap of the prose cannot break the claim: what is pinned is
    # that the sentence says these things, not where it happens to break a line.
    flat = " ".join(doc.split())

    assert doc, f"{CERT_UNITS_TEST} lost its module docstring"
    assert "operator" not in flat, flat
    assert "does not pin enabledness" in flat, flat
    assert "SETUP.md route" in flat, flat
    # The pointer is pinned by what it points at, not by a line number (#1989):
    # a cited line moves with every edit above it, and then names something else.
    assert re.search(r"SETUP\.md:\d+ route", flat) is None, flat
    setup = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    assert f"systemctl --user enable --now {CERT_TIMER}" in setup
    assert "scripts/maintenance/check-unit-enabledness.sh" in flat, flat
    assert "is-enabled" not in src, (
        "this module has started asserting a unit is enabled, which is the host "
        "state it exists to refuse")


def test_the_remediation_arms_the_timer_and_points_by_content(tmp_path):
    """#2537: the command a red watchdog prints has to be one that works.

    `is-enabled` answers the unit FILE, so plain `systemctl --user enable X.timer`
    links X into `timers.target.wants/` and leaves `ActiveState=inactive` until the
    user manager is restarted — this check then goes green and the job still never
    fires. That is the hole #2428's window fell into: this script's FAIL line printed
    the bare form, and `lloyd-persistence-window.timer` sat installed, red-lined and
    unarmed (`journalctl --user -u lloyd-persistence-window.service --since
    2026-10-08` → `-- No entries --`) while its enable was handed forward.

    The arming form is `enable --now`, which is what SETUP.md carries for every repo
    timer it documents. Since #2541 (2026-10-11) the sentence "an unattended turn may
    run `--now`" is true of a `.timer` and false of anything else: systemd performs the
    start or stop for `--now` itself, so the flag form is now classified as the verb it
    performs (`systemctl start` for `enable`/`reenable --now`, `systemctl stop` for
    `disable --now`) and a non-`.timer` target is tier 2 and refused to a background
    session like the bare verb always was. What this line prints keeps working because
    its target is a timer — the exemption `test_the_script_prose_states_the_post_fix_now_rule`
    pins to the guards' own constant rather than to this paragraph.

    Pinned by RUNNING the script at a stub that answers one timer `disabled`, so what
    is graded is the line an operator actually reads rather than a grep of the source.
    The vacuity control is that a FAIL line naming that unit exists at all before any
    shape is asserted about it. The line must offer `enable --now <unit>`, must not
    offer the bare form for any unit, and no output line may carry a `SETUP.md:<n>`
    pointer — the no-line-number rule
    `test_the_cert_renew_units_docstring_blames_the_route_not_the_operator` applies to
    prose, for the reason it names: the number this script used to print had already
    drifted, and the line it named was a `/proc` environ capture, not an enable route.

    What this node still refuses to assert is unchanged: that some timer IS enabled or
    active on the box. That is host state, and the two commands that measure it are
    `systemctl --user is-enabled lloyd-persistence-window.timer` and `systemctl --user
    show lloyd-persistence-window.timer -p ActiveState`.
    """
    unit_dir = _unit_names(tmp_path, "alpha.timer", "beta.timer")

    proc, out, _ = _run(unit_dir, {"beta.timer": "disabled"}, tmp_path, tag="arming")

    assert proc.returncode != 0, f"a disabled timer passed:\n{proc.stdout}"
    assert _denominator(out) == 2, out
    fails = _fail_lines(out)
    named = [line for line in fails if "beta.timer" in line]
    assert named, f"the miss named no unit to fix, so the shape checks below prove " \
                  f"nothing about a hint nobody is shown: {fails}"
    assert not [line for line in fails if "alpha.timer" in line], fails
    for line in named:
        assert re.search(r"enable --now beta\.timer\b", line), line
    for line in fails:
        assert re.search(r"enable (?!--now)\S*\.timer", line) is None, \
            f"bare `enable` does not arm a timer: {line}"
    assert not [line for line in out if re.search(r"SETUP\.md:\d+", line)], out


def test_the_fail_line_names_the_last_trigger_instant_or_never_armed(tmp_path):
    """#2702 clause 1: the FAIL line carries the unit's `LastTriggerUSec`, and prints the
    literal `never-armed` instead of a blank when that property comes back with no value.

    Three miss units share one run so each rendering is the difference between answers
    rather than the only answer available: `fired.timer` has an instant, `never.timer`
    has the empty string systemd renders for an unset timestamp (which is what
    `systemctl --user show -l -p LastTriggerUSec lloyd-egress-window.timer` printed when
    #2702 was triaged), and `unset.timer` has `n/a`, systemd's other no-value spelling.
    Both no-value forms have to reach `never-armed`: this guard runs on goliath as well as
    here, and a marker that fires on only one box's rendering leaves the other reading
    `LastTriggerUSec=` and blaming the guard.

    Two things keep this falsifiable rather than tautological. The stub answers the two
    properties sorted by name — the reverse of the order the script asks for them, which
    is what the real command does on this box — so a line that renders BOTH fields under
    the right label is evidence the guard read them by key and not by position. And the
    stub's default for an unconfigured unit is an armed timer with a real instant, so
    `never-armed` cannot appear except from a property this node asked to be empty; a
    guard that printed the default instead of reading the answer would print the instant.

    The blank itself is ruled out by shape, not only by the marker being present: no FAIL
    line may carry `LastTriggerUSec=` followed by a non-space.
    """
    unit_dir = _unit_names(tmp_path, "alpha.timer", "fired.timer", "never.timer",
                           "unset.timer")
    show = {
        "fired.timer": {"ActiveState": "inactive",
                        "LastTriggerUSec": "Wed 2026-10-14 02:12:27 PDT"},
        "never.timer": {"ActiveState": "inactive", "LastTriggerUSec": ""},
        "unset.timer": {"ActiveState": "inactive", "LastTriggerUSec": "n/a"},
    }

    proc, out, _ = _run(
        unit_dir, {"fired.timer": "disabled", "never.timer": "disabled",
                   "unset.timer": "disabled"}, tmp_path, tag="trigger", show=show)

    assert proc.returncode != 0, out
    assert _denominator(out) == 4, out

    fired = _fail_line_for(out, "fired.timer")
    assert "LastTriggerUSec=Wed 2026-10-14 02:12:27 PDT" in fired, fired
    assert "ActiveState=inactive" in fired, fired
    for unit in ("never.timer", "unset.timer"):
        line = _fail_line_for(out, unit)
        assert "LastTriggerUSec=never-armed" in line, line
        assert "n/a" not in line, f"the raw no-value spelling leaked: {line}"
    for line in _fail_lines(out):
        assert re.search(r"LastTriggerUSec=\S", line), f"a blank instant: {line}"
    assert not [line for line in _fail_lines(out) if "alpha.timer" in line], out


def test_the_same_fail_line_separates_a_running_timer_from_a_stopped_one(tmp_path):
    """#2702 clause 2: `ActiveState` rides on that same FAIL line, so a unit FILE that is
    disarmed while the loaded timer still runs reads differently from a timer that is
    genuinely not running.

    The two miss units are given the SAME `LastTriggerUSec`, which is what makes this a
    comparison instead of a word-lookup: assert each line's `ActiveState` token and then
    assert the two lines are identical once the unit name and that one token are
    normalised away. A node that only checked the word appeared would still pass if the
    value were hardcoded, and would not notice the field being decorative — and the
    distinction it carries is the one #2702's triage had to reconstruct by hand: `disable`
    without `--now` leaves a `disabled` unit reading `ActiveState=active` until the user
    manager next restarts (a stop that has not happened yet), while an installed-but-
    never-armed timer reads `inactive` (an action that never happened).

    The third unit is the no-value case: `ActiveState=n/a` renders as the literal
    `no-answer` rather than as a blank, so an evidence read that could not ask still says
    so on the same line.
    """
    unit_dir = _unit_names(tmp_path, "running.timer", "stopped.timer", "blind.timer")
    same_instant = "Wed 2026-10-14 02:12:27 PDT"
    show = {
        "running.timer": {"ActiveState": "active", "LastTriggerUSec": same_instant},
        "stopped.timer": {"ActiveState": "inactive", "LastTriggerUSec": same_instant},
        "blind.timer": {"ActiveState": "n/a", "LastTriggerUSec": same_instant},
    }

    proc, out, _ = _run(unit_dir, {"*.timer": "disabled"}, tmp_path, tag="state",
                        show=show)

    assert proc.returncode != 0, out
    assert _denominator(out) == 3, out

    running = _fail_line_for(out, "running.timer")
    stopped = _fail_line_for(out, "stopped.timer")
    assert re.search(r"ActiveState=active\b", running) and \
        "ActiveState=inactive" not in running, running
    assert re.search(r"ActiveState=inactive\b", stopped) and \
        "ActiveState=active," not in stopped, stopped

    def normalised(line: str, unit: str) -> str:
        return line.replace(unit, "UNIT").replace(
            re.search(r"ActiveState=\S+", line).group(0), "STATE")

    assert normalised(running, "running.timer") == normalised(stopped, "stopped.timer"), \
        f"{running}\n{stopped}"

    blind = _fail_line_for(out, "blind.timer")
    assert "ActiveState=no-answer" in blind, blind
    assert "n/a" not in blind, blind
    for line in (running, stopped, blind):
        assert "LastTriggerUSec=Wed 2026-10-14 02:12:27 PDT" in line, line


def test_the_arming_read_fires_once_per_timer_after_its_enabledness_answer(tmp_path):
    """The guard's cost limit, re-pinned for the wider read #2943 gave it.

    This node shipped as `#2702's own cost limit`: the arming question asked of a timer
    only after `is-enabled` had already failed it, and `show == []` on a green run. #2943
    asks that same question of the rows that PASS — which IS the change — so the half
    claiming "an all-enabled fleet must ask `show` nothing" is now the false half and is
    replaced rather than deleted. The bound that survives, and the one this node exists to
    hold, is ONE `show` per timer even though two consumers want the reading (the
    never-armed verdict on a passing row, the evidence sentence on a FAIL line), so no run
    ever costs three `systemctl` calls for one unit. That is the guard's contract with the
    route that runs it: `service_health_check.py` gives it a 30-second timeout, and the
    whole script over this box's eight timers — sixteen calls — measures 0.115 s.

    Order is load-bearing as well as count, because a bare count would still pass on a
    guard that asked the arming question FIRST and graded enabledness off it. Both runs
    assert the `is-enabled`/`show` interleaving pair by pair, in call order, and the argv
    assertion keeps both property names on the request.
    """
    unit_dir = _unit_names(tmp_path, "alpha.timer", "beta.timer", "gamma.timer")
    want = ["alpha.timer", "beta.timer", "gamma.timer"]
    interleaved = ["is-enabled:alpha.timer", "show:alpha.timer",
                   "is-enabled:beta.timer", "show:beta.timer",
                   "is-enabled:gamma.timer", "show:gamma.timer"]

    def _pairs(queries):
        return [f"{q[1]}:{q[-1]}" for q in queries]

    green, green_out, green_queries = _run(unit_dir, {}, tmp_path, tag="greenask")
    assert green.returncode == 0, green_out
    assert _pairs(green_queries) == interleaved, green_queries

    red, red_out, red_queries = _run(unit_dir, {"beta.timer": "disabled",
                                                "gamma.timer": "disabled"},
                                     tmp_path, tag="redask")
    assert red.returncode != 0, red_out
    assert _pairs(red_queries) == interleaved, red_queries
    for argv in (q for q in red_queries if q[1] == "show"):
        assert argv[:2] == ["--user", "show"], argv
        assert "LastTriggerUSec" in argv and "ActiveState" in argv, argv
    for tag, queries in (("green", green_queries), ("red", red_queries)):
        asked = _asked_about(queries, "show")
        assert sorted(asked) == want, f"{tag} run asked `show` of {asked}, not {want}"
        assert len(asked) == len(set(asked)), f"{tag} run asked some timer twice: {asked}"


def test_an_evidence_read_that_fails_leaves_the_verdict_and_remediation_alone(tmp_path):
    """The new question must not be allowed to change the old answer.

    `show` can fail while `is-enabled` succeeds at saying `disabled` — a unit that is not
    loaded, a manager that is not listening for that query. The guard discards that call's
    stderr, because `service_health_check.py` folds this script's stderr into the same text
    it greps for rows and FAIL lines, so a tool error printed here would land in the parsed
    stream as a line about nothing. What the line must still do is name the miss, say in
    words that it could not ask, and print the arming command; and the run must still exit
    non-zero. A marker for "we could not read it" is a report, not a pass.

    The `stub systemctl` assertion is the load-bearing one for the stderr claim: the stub
    writes that message to stderr and exits non-zero for a unit whose `show` answer is
    `error`, so if the guard ever stopped discarding it, it would show up in the combined
    output this helper returns.

    #2943 reads this rule from the opposite side. Here a failed read rides on a row that was
    ALREADY red; there a failed read must leave a green row green, because "the question
    could not be asked" is not evidence that the timer never armed — see
    `test_only_the_never_armed_signature_moves_a_row_off_enabled`. The pair is why the new
    verdict keys on the exact reading (`ActiveState=inactive` AND no instant) and not on
    anything looser like "the arming read did not say armed", which would turn every box
    with a quiet `systemctl show` into a fleet of false alarms.
    """
    unit_dir = _unit_names(tmp_path, "alpha.timer", "broken.timer")

    proc, out, _ = _run(unit_dir, {"broken.timer": "disabled"}, tmp_path,
                        tag="brokenshow", show={"broken.timer": "error"})

    assert proc.returncode != 0, f"an unreadable evidence read passed the guard: {out}"
    assert _denominator(out) == 2, out
    assert _row_lines(out) == ["alpha.timer\tenabled", "broken.timer\tdisabled"], out
    assert len(_fail_lines(out)) == 1, out

    line = _fail_line_for(out, "broken.timer")
    assert "ActiveState=no-answer" in line, line
    assert "LastTriggerUSec=never-armed" in line, line
    assert re.search(r"enable --now broken\.timer\b", line), line
    assert not [ln for ln in out if "stub systemctl" in ln], (
        f"the evidence call's stderr reached the parsed output: {out}")


def test_an_enabled_timer_that_has_never_armed_fails_the_guard(tmp_path):
    """#2943 clause 1: `enabled` on the unit FILE stopped being a green row on its own.

    The state this node pins is the one this box sat in: `is-enabled` answers `enabled`,
    `systemctl --user show` answers `ActiveState=inactive` with no `LastTriggerUSec`. That
    is an installed, linked, never-armed job — `lloyd-egress-window.timer`, whose
    `journalctl --user -u lloyd-egress-window.service` is `-- No entries --` — and before
    #2943 the guard reported it as `enabled`, printed `checked 8 timers: 8 enabled, 0 not
    enabled` and exited 0.

    Three readings of that row are asserted, because each is a different consumer. The ROW
    carries `enabled-never-armed`, not the bare `enabled`, so the status
    `scripts/service_health_check.py` builds from it cannot read as a pass; the FAIL line
    carries `ActiveState=inactive`, the literal `never-armed` and the `enable --now`
    remediation; and the sentence is `is enabled but never armed`, never `is not enabled`,
    which is false for this unit and would send its reader to re-link a file that is
    already linked — the half-act that created this state.

    Both no-value renderings of `LastTriggerUSec` run through the same assertions: systemd
    prints the empty string for an unset timestamp and `n/a` is its other no-value spelling,
    and this guard ships to more than one box.
    """
    for no_value, tag in (("", "empty"), ("n/a", "na")):
        unit_dir = _unit_names(tmp_path, f"alpha-{tag}.timer", f"never-{tag}.timer")
        never = f"never-{tag}.timer"

        proc, out, _ = _run(unit_dir, {}, tmp_path, tag=f"neverarmed-{tag}",
                            show={never: {"ActiveState": "inactive",
                                          "LastTriggerUSec": no_value}})

        assert proc.returncode == 1, (
            f"[{tag}] an enabled but never-armed timer passed:\n{out}")
        assert _row_lines(out) == [f"alpha-{tag}.timer\tenabled",
                                  f"{never}\tenabled-never-armed"], out
        assert _denominator(out) == 2, out
        assert [ln for ln in out if ln.startswith("checked ")] == [
            "checked 2 timers: 1 enabled, 1 not enabled"], out

        line = _fail_line_for(out, never)
        assert "ActiveState=inactive" in line, line
        assert "never-armed" in line, line
        assert re.search(rf"— run: systemctl --user enable --now {re.escape(never)}\b",
                         line), line
        assert "is not enabled" not in line, (
            f"[{tag}] the unit FILE is enabled; saying otherwise sends the reader off to "
            f"re-link it: {line}")
        assert not [ln for ln in _fail_lines(out)
                    if f"alpha-{tag}.timer" in ln], out


def test_only_the_never_armed_signature_moves_a_row_off_enabled(tmp_path):
    """#2943 clause 2: the new red is keyed on the signature ALONE, so it cannot misfire.

    Five enabled timers, five different `show` answers, one run: exit 0, no FAIL line, and
    every row still the bare word `enabled`.

      * `unread.timer` — `show` exits non-zero and prints nothing. A read that could not be
        made is not evidence that no fire happened.
      * `blank.timer` — `show` answers both properties empty. `ActiveState` is not
        `inactive`, so the signature does not hold.
      * `armed.timer` — `active` plus a real instant: the ordinary healthy timer (and the
        stub's own default for any unit a node leaves undescribed).
      * `fresh.timer` — `active` with NO instant. This is a timer armed seconds ago, whose
        first fire has not happened yet. It is the case that must not go red: the guard's
        next green run after a successful arming would otherwise accuse the act that worked,
        which is the false alarm #2792's route would then chase.
      * `stopped.timer` — `inactive` WITH a real instant: a schedule that ran. "Keys on the
        never-armed signature alone" means the instant is exactly the reading that
        separates the two states, so this row stays green too.

    A guard that dropped the signature check entirely passes every assertion here, which is
    why this node is read beside `test_an_enabled_timer_that_has_never_armed_fails_the_guard`
    above: that one is the witness that the same helper, with a no-value instant and
    `ActiveState=inactive`, does go red.
    """
    names = ["unread.timer", "blank.timer", "armed.timer", "fresh.timer",
             "stopped.timer"]
    unit_dir = _unit_names(tmp_path, *names)
    instant = "Wed 2026-10-14 02:12:27 PDT"
    show = {
        "unread.timer": "error",
        "blank.timer": {"ActiveState": "", "LastTriggerUSec": ""},
        "armed.timer": {"ActiveState": "active", "LastTriggerUSec": instant},
        "fresh.timer": {"ActiveState": "active", "LastTriggerUSec": ""},
        "stopped.timer": {"ActiveState": "inactive", "LastTriggerUSec": instant},
    }

    proc, out, queries = _run(unit_dir, {}, tmp_path, tag="signature", show=show)

    assert proc.returncode == 0, f"a non-signature reading failed the guard:\n{out}"
    assert not _fail_lines(out), out
    assert _row_lines(out) == [f"{name}\tenabled" for name in sorted(names)], (
        f"rows are the unit dir's sorted set, and every one of these five readings must "
        f"still print the bare word `enabled`: {out}")
    assert _denominator(out) == 5, out
    assert not [ln for ln in out if "stub systemctl" in ln], (
        f"the failed `show` leaked its stderr into the parsed output: {out}")
    assert sorted(_asked_about(queries, "show")) == sorted(names), queries


def test_a_never_armed_row_leaves_the_row_and_denominator_contract_intact(tmp_path):
    """#2943 clause 3: the guard grew a verdict without growing a new output shape.

    Run over the REAL repo unit dir so the denominator is the tree's own count — eight
    timers since #2864 tracked `lloyd-graph-backup.timer` — and `len(REPO_TIMERS)` is where
    that number comes from, never a count this file memorised.
    `test_the_new_timer_is_inside_the_enabledness_denominator` in
    tests/test_egress_window_unit.py re-derives the same number from the directory and
    refuses a remembered one in either direction, so the count this node prints is a
    reading of the tree rather than a claim about it.

    One repo timer is planted with the never-armed signature and the whole parse contract
    is asserted over the run that results, in the four shapes
    `scripts/service_health_check.py` reads: one `<unit><TAB><verdict>` row per repo
    `.timer` and the planted one NOT the bare word `enabled`; exactly one
    `checked N timers: M enabled, K not enabled` line, still ahead of the FAIL lines;
    `K` equal to the number of FAIL lines actually printed; and the remediation still
    `systemctl --user enable --now <unit>`, never the bare form, with no `SETUP.md:<n>`
    pointer.
    """
    assert EGRESS_TIMER in REPO_TIMERS, (
        f"{EGRESS_TIMER} is no longer in agent-services/systemd/, so the signature planted "
        f"below names a unit the guard does not enumerate: {REPO_TIMERS}")
    total = len(REPO_TIMERS)

    proc, out, _ = _run(REPO_UNIT_DIR, {}, tmp_path, tag="reponever",
                        show={EGRESS_TIMER: {"ActiveState": "inactive",
                                             "LastTriggerUSec": ""}})

    assert proc.returncode == 1, f"a never-armed repo timer passed:\n{out}"
    rows = _row_lines(out)
    assert len(rows) == total, out
    assert sorted(row.split("\t")[0] for row in rows) == REPO_TIMERS, rows
    assert f"{EGRESS_TIMER}\tenabled-never-armed" in rows, rows
    assert f"{EGRESS_TIMER}\tenabled" not in rows, (
        "the row still reads the bare `enabled`, so the health route's status for this "
        "unit stays a false green even with this script exiting 1")
    assert sum(1 for row in rows if row.endswith("\tenabled")) == total - 1, rows

    totals = [line for line in out if line.startswith("checked ")]
    assert totals == [f"checked {total} timers: {total - 1} enabled, 1 not enabled"], out
    assert _denominator(out) == total, out
    fails = _fail_lines(out)
    not_enabled = int(re.fullmatch(
        r"checked \d+ timers: \d+ enabled, (\d+) not enabled", totals[0]).group(1))
    assert not_enabled == len(fails) == 1, (totals[0], fails)
    assert out.index(totals[0]) < min(out.index(line) for line in fails), out

    line = _fail_line_for(out, EGRESS_TIMER)
    assert re.search(rf"enable --now {re.escape(EGRESS_TIMER)}\b", line), line
    for fail in fails:
        assert re.search(r"enable (?!--now)\S*\.timer", fail) is None, \
            f"bare `enable` does not arm a timer: {fail}"
    assert not [line for line in out if re.search(r"SETUP\.md:\d+", line)], out


def test_a_grown_fail_line_leaves_the_row_and_denominator_contract_intact(tmp_path):
    """#2702 clause 3: the output the health route parses is byte-comparable with before.

    `scripts/service_health_check.py` reads this script four ways — `^(\\S+\\.timer)\\t(\\S+)$`
    for the rows, `^checked \\d+ timers?: .*$` for the denominator, `^FAIL (\\S+\\.timer) ` for
    which rows to fail, and the exit code for the verdict — and `test_unit_enabledness_route.py`
    pins that the rows carry the denominator line verbatim. The evidence therefore had to
    land on the FAIL line and nowhere else: appended to a row it would break the tab regex,
    whose verdict group cannot hold a space, and a second denominator-shaped line would
    make the single-line assertion below, and the route's copy of the count, go red.

    So this node runs one red fleet and pins the whole contract in one place: the rows are
    exactly the three `unit<TAB>is-enabled-answer` lines with nothing appended, there is
    exactly one `checked N timers: M enabled, K not enabled` line and it carries the true
    counts, it still precedes the FAIL lines, the exit code is 1, and each FAIL line still
    offers `enable --now <unit>` and never the bare form #2537 removed.
    """
    unit_dir = _unit_names(tmp_path, "alpha.timer", "beta.timer", "gamma.timer")

    proc, out, _ = _run(unit_dir, {"beta.timer": "disabled", "gamma.timer": "disabled"},
                        tmp_path, tag="contract")

    assert proc.returncode == 1, f"{proc.returncode}: {out}"
    assert sorted(_row_lines(out)) == ["alpha.timer\tenabled", "beta.timer\tdisabled",
                                       "gamma.timer\tdisabled"], out
    for line in _row_lines(out):
        assert "ActiveState" not in line and "LastTriggerUSec" not in line, line

    totals = [line for line in out if line.startswith("checked ")]
    assert totals == ["checked 3 timers: 1 enabled, 2 not enabled"], out
    assert _denominator(out) == 3, out

    fails = _fail_lines(out)
    assert len(fails) == 2, fails
    assert out.index(totals[0]) < min(out.index(line) for line in fails), out
    for unit in ("beta.timer", "gamma.timer"):
        assert re.search(rf"enable --now {re.escape(unit)}\b",
                         _fail_line_for(out, unit)), unit
    for line in fails:
        assert re.search(r"enable (?!--now)\S*\.timer", line) is None, \
            f"bare `enable` does not arm a timer: {line}"


def test_the_command_this_line_prints_is_runnable(tmp_path):
    """#2541: the remedy this script prints still has to survive the guards that now
    read `--now`.

    #2537's rule was that a red watchdog must print a command that works. #2541 moved
    the ground under it by classifying a `--now` command as the unit operation systemd
    performs for it — `systemctl start` for `enable`/`reenable --now`, `systemctl stop`
    for `disable --now` — so that rule holds today only because the target printed here
    is a `.timer`, the one unit type whose arming is not a unit start. Run the script at
    a stub that answers one timer `disabled`, lift the command out of the line an
    operator would actually read, and put THAT string through both axes as a background
    `task:` session: tier 1, no service-control label, no refusal.

    This is the boundary the owed check would otherwise measure against live traffic:
    watchdog output in, guard verdict out. The other half of the pair is
    `test_the_now_flag_form_of_a_service_start_is_refused` in
    tests/test_service_control_guard.py — the same flag form against a `.service` is
    refused — and if this node ever goes red, this script has started printing a command
    its own reader is not allowed to run.
    """
    from app.harness.safety import bash_command_tier, check_bash_command
    from app.harness.service_control import find_service_control

    unit_dir = _unit_names(tmp_path, "gamma.timer")
    proc, out, _ = _run(unit_dir, {"gamma.timer": "disabled"}, tmp_path, tag="runnable")
    assert proc.returncode != 0, out

    named = [line for line in _fail_lines(out) if "gamma.timer" in line]
    assert named, f"no FAIL line to take a command from: {_fail_lines(out)}"
    matches = [re.search(r"(systemctl --user enable --now \S+)", line) for line in named]
    assert all(matches), f"the FAIL line printed no runnable shape: {named}"
    command = matches[0].group(1)
    assert command == "systemctl --user enable --now gamma.timer", command

    assert bash_command_tier(command) == 1, \
        f"the watchdog's own remedy is now tiered, so an unattended turn reading this " \
        f"line is refused the command it was told to run: {command}"
    assert find_service_control(command) is None, command
    assert check_bash_command(command, session_id="task:watchdog") is None, command


def test_the_script_prose_states_the_post_fix_now_rule():
    """#2541 clause 5: the header that used to promise `enable --now` to any
    unattended turn states the post-fix rule, and states it in the guards' own
    vocabulary so it cannot drift from them.

    The block under `WHAT THE FAIL LINE NAMES` claimed `--now` was "the only form an
    unattended turn can act on at all", on the grounds that `systemctl --user start` is
    tier 2 and a service-control verb while `enable --now` was tier 1 with no label.
    That asymmetry was the bug #2541 fixes, not a property of the flag: the flag form
    starts or stops a unit, so it is now labelled and tiered as the verb it performs and
    refused to a background or unresolvable `task:*` session. This paragraph is prose
    both a human and a watchdog reader act on, so a stale sentence here would print a
    command the harness then refuses — which is exactly the failure
    `test_the_command_this_line_prints_is_runnable` catches for the FAIL line itself.

    Graded against `app.harness.service_control`'s own constants rather than this file's
    memory of them: every operation the map produces must be named in the block, the
    exempt unit type must be named, and the block must say the SERVICE form is denied.
    """
    from app.harness.service_control import SYSTEMCTL_NOW_UNIT_EXEMPT, SYSTEMCTL_NOW_VERBS

    text = SCRIPT.read_text(encoding="utf-8")
    assert "# WHAT THE FAIL LINE NAMES" in text and \
           "# tests/test_unit_enabledness_check.py drives" in text, text[:200]
    block = text.split("# WHAT THE FAIL LINE NAMES", 1)[1].split(
        "# tests/test_unit_enabledness_check.py drives", 1)[0]

    assert "#2541" in block, block
    assert SYSTEMCTL_NOW_UNIT_EXEMPT in block, \
        f"the block never names the exempt unit type {SYSTEMCTL_NOW_UNIT_EXEMPT!r}, so a " \
        f"reader cannot tell which `--now` survives: {block}"
    for operation in sorted(set(SYSTEMCTL_NOW_VERBS.values())):
        assert f"systemctl {operation}" in block, \
            f"the guard classifies a `--now` as `systemctl {operation}` and the block " \
            f"never says so: {block}"
    assert "<service>" in block and "tier 2" in block, \
        f"the block must say the service form is now denied, not available: {block}"
    assert "`enable --now` is tier 1" not in block, \
        "the blanket pre-fix claim is still standing as written"


def test_the_guard_module_docstring_states_the_same_rule():
    """#2541 clause 5's other half: the module whose verdict this script describes says
    the same thing in its own docstring.

    `app/harness/service_control.py` is where a reader who wants to know whether a
    command is allowed actually looks, and before #2541 its docstring enumerated the
    refused verbs without ever mentioning the flag that performed them — which is how
    the asymmetry read as deliberate. It now carries the rule, the `.timer` carve-out and
    the reason for it, graded against the same two constants so this node goes red if the
    prose and the guards part ways in either direction.
    """
    from app.harness import service_control
    from app.harness.service_control import (SYSTEMCTL_NOW_UNIT_EXEMPT,
                                             SYSTEMCTL_NOW_VERBS)

    doc = service_control.__doc__ or ""
    assert "#2541" in doc, "the module never records why the flag form counts"
    assert SYSTEMCTL_NOW_UNIT_EXEMPT in doc, \
        f"the module docstring does not name the exempt type {SYSTEMCTL_NOW_UNIT_EXEMPT!r}"
    # The prose renders these as `systemctl start`, so match the words without the
    # backticks: what must hold is that the docstring names each operation the map
    # produces, not how it punctuates it.
    for operation in sorted(set(SYSTEMCTL_NOW_VERBS.values())):
        assert f"systemctl {operation}" in doc.replace("`", ""), \
            f"the guard labels a `--now` `systemctl {operation}` and the docstring " \
            f"does not say so"
    assert "enable --now` is the form an unattended turn may always run" not in doc, \
        "the pre-fix blanket claim is back in the module's own words"


def test_the_guard_names_its_caller():
    """#1989 clause 5: the header's WHO RUNS IT names the health script, and the
    health script really does name this guard — a header that claims a caller the
    tree does not have is the same dangling pointer in the other direction."""
    text = SCRIPT.read_text(encoding="utf-8")
    who = text.split("# WHO RUNS IT.", 1)[1].split("# WHAT IT ASSERTS", 1)[0]
    assert "scripts/service_health_check.py" in who, who
    assert "Nobody automatically" not in who, who

    caller = (ROOT / "scripts" / "service_health_check.py").read_text(encoding="utf-8")
    assert '"check-unit-enabledness.sh"' in caller
