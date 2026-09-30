"""#1891 — scripts/maintenance/check-unit-enabledness.sh, the guard on a placed-but-disabled timer.

A unit file that is installed but never enabled is invisible to every route this
repo owns: `agent-services/setup/install-services.sh:99-100` *echoes* the
`systemctl --user enable` lines instead of running them, `promote.py` only copies a
unit and daemon-reloads, and `scripts/service_health_check.py` is supervisor-only —
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
file asserts anything about the machine running it.

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
# Live in ~/.config/systemd/user/timers.target.wants/ on goliath, tracked nowhere
# under agent-services/systemd/ — the reason the check must be membership and not
# set equality (tests/test_data_home.py rules the same thing for the cutover fleet).
LIVE_TIMER_WITHOUT_REPO_UNIT = "lloyd-graph-backup.timer"

STUB = """#!/usr/bin/env python3
import json, os, sys

# Stand-in for `systemctl`, with the real command's semantics: `is-enabled <unit>`
# prints the state and exits 0 only for `enabled`; any other state prints and exits
# non-zero. Every argv is appended to STUB_LOG so a test can assert what was ASKED,
# not just what came back.
#
# STUB_STATES maps a unit name to the state to answer, with two lookups beyond an
# exact hit: a `*.suffix` key matches any unit ending in that suffix, and
# `__default__` answers everything else (an unlisted unit answers `enabled`).
argv = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(json.dumps(argv) + "\\n")

states = json.loads(os.environ.get("STUB_STATES", "{}"))


def answer(unit):
    if unit in states:
        return states[unit]
    for key, state in states.items():
        if key.startswith("*") and unit.endswith(key[1:]):
            return state
    return states.get("__default__", "enabled")


if argv[:2] == ["--user", "is-enabled"] and len(argv) == 3:
    state = answer(argv[2])
    print(state)
    raise SystemExit(0 if state == "enabled" else 1)

sys.stderr.write("stub systemctl: unstubbed invocation %r\\n" % (argv,))
raise SystemExit(2)
"""


def _run(unit_dir: Path, states: dict, tmp_path: Path, tag: str = "run",
         require_query: bool = True):
    """Execute the script for real, with a stub `systemctl` first on PATH.

    Returns (CompletedProcess, combined stdout+stderr lines, argv the stub saw).
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


def test_the_script_is_executable_and_prints_a_verdict_line_per_repo_timer(tmp_path):
    """Clause 1: every `agent-services/systemd/*.timer` gets its `is-enabled` verdict
    on a line of its own, and a total line names how many were checked.

    Run over the REAL repo unit dir (repo content, not host state) against a stub
    that answers `enabled`, so the expected denominator is whatever the tree
    actually tracks rather than a number this file memorised.
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
    assert sorted(q[2] for q in queries) == REPO_TIMERS, queries


def test_every_timer_answering_enabled_exits_zero(tmp_path):
    """Clause 2, first half: all-`enabled` is the pass case — and the denominator
    tracks the set it was given, which is what stops it being a remembered number."""
    unit_dir = _unit_names(tmp_path, "alpha.timer", "beta.timer", "gamma.timer")

    proc, out, queries = _run(unit_dir, {}, tmp_path)

    assert proc.returncode == 0, f"{proc.returncode}:\n{proc.stdout}{proc.stderr}"
    assert _denominator(out) == 3, out
    assert not _fail_lines(out), out
    assert sorted(q[2] for q in queries) == ["alpha.timer", "beta.timer", "gamma.timer"]


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
    assert [q[2] for q in queries].count("beta.timer") == 1, queries


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
    asked = {q[2] for q in queries}
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

    `lloyd-graph-backup.timer` is live in goliath's `timers.target.wants/` and
    tracked nowhere in the repo, so a check that walked the wants dir — or compared
    it to the repo set — would fail on a unit this repository has no say in. The
    stub is told that unit is `disabled`; the check must never ask, and must pass.
    """
    assert not (REPO_UNIT_DIR / LIVE_TIMER_WITHOUT_REPO_UNIT).is_file(), (
        f"{LIVE_TIMER_WITHOUT_REPO_UNIT} is now tracked in the repo, so the "
        "membership-not-equality premise this node pins has changed")

    proc, out, queries = _run(REPO_UNIT_DIR,
                              {LIVE_TIMER_WITHOUT_REPO_UNIT: "disabled"},
                              tmp_path, tag="onedir")

    assert proc.returncode == 0, f"{proc.returncode}:\n{proc.stdout}{proc.stderr}"
    asked = {q[2] for q in queries}
    assert LIVE_TIMER_WITHOUT_REPO_UNIT not in asked, asked
    assert asked == set(REPO_TIMERS), f"{asked ^ set(REPO_TIMERS)} is off-set"


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
    assert "SETUP.md:1520" in flat, flat
    assert "scripts/maintenance/check-unit-enabledness.sh" in flat, flat
    assert "is-enabled" not in src, (
        "this module has started asserting a unit is enabled, which is the host "
        "state it exists to refuse")
