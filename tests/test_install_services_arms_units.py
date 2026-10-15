"""#2774 — `install-services.sh` arms the units it installs, and this file pins how.

The installer had two acts and performed one. It symlinked every `.service` and `.timer`
in `agent-services/systemd/` into `~/.config/systemd/user/`, ran `daemon-reload`, and then
PRINTED the `systemctl --user enable ...` commands as `echo` lines for a human to copy.
Placement is not activation: `scripts/maintenance/check-unit-enabledness.sh` has been
printing `FAIL lloyd-egress-window.timer ... never-armed` since the day #2555 landed that
timer, this box's `timers.target.wants/` holds a link for six of the seven repo timers,
`journalctl --user -u lloyd-egress-window.service` is `-- No entries --`, and the four
window units #1891, #2537, #2702 and #2771 each shipped into exactly that state. The fix is
for the script to run the `enable` itself, which it now does.

Every node here runs the REAL script in a subprocess against a throwaway `HOME` and a stub
`systemctl` written onto the front of `PATH`, which records every argv as JSON. That harness
is not new: `tests/test_persistence_window_unit.py::_run_installer` and
`tests/test_egress_window_unit.py::test_the_installer_links_both_units_by_its_existing_glob`
already run the installer that way, and this file keeps their contract — one real run must
still exit 0. Arming is a `systemctl` call, so the only honest place to look is the argv the
script produced. Grepping the word `enable` out of the file would pass on a script whose
every occurrence sits inside an `echo` argument, which is precisely the script this round
replaced; `test_a_script_whose_only_enable_sits_inside_an_echo_fails_the_predicate`
demonstrates that the predicate rejects it.

The four rules the script obeys, each learned from a measured failure, are in its own
comment block. What is pinned per rule:

  clause 1  an executed `enable` reaches every installed unit declaring `[Install]`, as
            argv, and an echo-only script is rejected by the same predicate.
  clause 2  every `.timer` is armed `enable --now` and no `.service` is, so nothing the
            script runs classifies as a unit operation to the guard that would otherwise
            refuse this fix to an unattended turn (`SYSTEMCTL_NOW_UNIT_EXEMPT = ".timer"`).
  clause 3  a unit with no `[Install]` section, a `SYSTEM_SCOPE_UNITS` name, and the service
            its own timer starts never reach `enable`; and a failing `enable` does not end
            the run under `set -euo pipefail`.
  clause 4  `ActiveState` prints beside `is-enabled` for every armed unit, so a timer that
            is enabled-but-inactive cannot read as green on stdout.

WHAT NO NODE HERE CLAIMS, because no diff can produce it: that this box's
`lloyd-egress-window.timer` now reads `enabled`/`ActiveState=active`, that
`timers.target.wants/` gained its eighth link, or that the Monday fire arrives. That is host
state, measured after the fix is actually run (the owed check), and `HOME` is redirected
into `tmp_path` in every node precisely so that nothing here touches it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "agent-services" / "setup" / "install-services.sh"
UNIT_DIR = ROOT / "agent-services" / "systemd"
GUARD = ROOT / "scripts" / "maintenance" / "check-unit-enabledness.sh"
HEALTH_CHECK = ROOT / "scripts" / "service_health_check.py"

REPO_TIMERS = sorted(p.name for p in UNIT_DIR.glob("*.timer"))
#: The unit #2770 armed and #2771 is waiting on: installed, linked, and the only repo timer
#: with no fire instant. Its name is load-bearing below because it is the unit this round
#: exists to arm; the COUNT of repo timers is derived from the tree, never recalled.
EGRESS_TIMER = "lloyd-egress-window.timer"
EGRESS_SERVICE = "lloyd-egress-window.service"

#: The six co-located services that carry no `[Install]` and therefore answer `static` or
#: `linked` on a live box. `enable` on one ERRORS, so under `set -euo pipefail` an
#: unfiltered arming loop would kill the installer mid-run.
SERVICES_WITHOUT_INSTALL = [
    "lloyd-cert-renew.service",
    "lloyd-data-snapshot.service",
    "lloyd-guardian-nag.service",
    "lloyd-persistence-window.service",
    "lloyd-qmd-cleanup.service",
    "lloyd-vault-backup.service",
]

STUB = """#!/usr/bin/env python3
# Stand-in for `systemctl`, with the semantics the installer depends on. Every argv is
# appended to STUB_LOG as one JSON array, which is what lets a node assert what the script
# ASKED systemd rather than what it printed. Four semantics matter: `is-enabled` prints the
# state and exits 0 only for `enabled` (the real command exits non-zero for every other
# state); `enable` is silent on success and non-zero on failure; `show -p KEY` prints
# `KEY=value` for each requested key; and an unknown invocation succeeds quietly. Nothing
# here reaches the machine — this is the only `systemctl` on PATH, and every node asserts
# the log is non-empty, so a script that resolved `/usr/bin/systemctl` outright would fail
# the node instead of arming the box the suite is supposed to leave alone.
import json, os, sys

argv = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(json.dumps(argv) + "\\n")

cfg = {}
cfg_path = os.environ.get("STUB_CONFIG")
if cfg_path and os.path.exists(cfg_path):
    with open(cfg_path) as fh:
        cfg = json.load(fh)

VERBS = ("is-enabled", "is-active", "daemon-reload", "enable", "disable",
         "start", "stop", "restart", "show", "status")
verb = next((t for t in argv if t in VERBS), "")
unit = next((t for t in argv if t.endswith((".service", ".timer", ".socket"))), "")

if verb == "is-enabled":
    state = cfg.get("states", {}).get(unit, cfg.get("state_default", "enabled"))
    print(state)
    sys.exit(0 if state == "enabled" else 1)

if verb == "is-active":
    state = cfg.get("active_service_default", "inactive")
    print(state)
    sys.exit(0 if state == "active" else 3)

if verb == "show":
    props = {"ActiveState": cfg.get("active_default", "active"),
             "LastTriggerUSec": "Sat 2026-10-10 06:23:01 PDT",
             "UnitFileState": cfg.get("states", {}).get(unit, "enabled")}
    for i, token in enumerate(argv):
        if token == "-p":
            for key in argv[i + 1].split(","):
                print("%s=%s" % (key, props.get(key, "")))
    sys.exit(0)

if verb in ("enable", "disable"):
    if unit in cfg.get("fail_enable", []):
        sys.stderr.write("Failed to %s %s: simulated refusal\\n" % (verb, unit))
        sys.exit(1)
    sys.exit(0)

sys.exit(0)
"""


def _declares_install(unit: Path) -> bool:
    """Whether a unit file has the section that makes `enable` mean anything."""
    return re.search(r"^\[Install\]", unit.read_text(encoding="utf-8"), re.M) is not None


def _started_by_a_timer(unit: Path) -> bool:
    """Whether a `.service` is the unit its co-located `.timer` fires."""
    return (unit.suffix == ".service"
            and (unit.parent / (unit.stem + ".timer")).is_file())


SYSTEM_SCOPE = set(re.search(r"^SYSTEM_SCOPE_UNITS=\(([^)]*)\)",
                             INSTALLER.read_text(encoding="utf-8"), re.M).group(1).split())

ALL_UNITS = sorted(p for p in UNIT_DIR.iterdir() if p.suffix in (".service", ".timer"))

#: The set the installer must hand to `systemctl --user enable`, derived from the tree the
#: same way the script derives it: picked up by the link glob, not system scope, `[Install]`
#: declared, and not a service its own timer starts. Derived rather than listed, so a unit
#: added to `agent-services/systemd/` is inside this expectation the moment it is committed
#: and the node cannot be quietly narrowed.
ARMABLE = sorted(p.name for p in ALL_UNITS
                 if p.name not in SYSTEM_SCOPE
                 and _declares_install(p)
                 and not _started_by_a_timer(p))

#: `[Install]`-bearing services the installer deliberately does NOT arm, because their own
#: timer starts them and `enable` would also pull them into their `WantedBy=` target at
#: every login. `lloyd-egress-window.service` is the case this rule was written for.
ARMED_ELSEWHERE = sorted(p.name for p in ALL_UNITS
                         if p.name not in SYSTEM_SCOPE
                         and _declares_install(p)
                         and _started_by_a_timer(p))


def _run(installer: Path, tmp_path: Path, *, config: dict | None = None,
         tag: str = "", unit_dir: Path | None = None):
    """One real run of `installer`, boxed: stub tools on PATH, a throwaway HOME.

    `HOME`, `PATH` and `USER` are NAMED, not inherited-with-gaps, for the reason
    `tests/test_persistence_window_unit.py::_run_installer` spells out: the installer runs
    `set -u` and closes by reading `$USER`, and the gate's environment carries no login
    name. `tag` separates the argv log when a node runs the same tree twice. `unit_dir`
    rebuilds the tree the installer globs when the caller needs a copy of the script that
    does not live in the repo — see the echo-only witness below.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    log = tmp_path / f"systemctl{tag}.log"
    systemctl = bin_dir / "systemctl"
    systemctl.write_text(STUB, encoding="utf-8")
    systemctl.chmod(0o755)
    loginctl = bin_dir / "loginctl"
    loginctl.write_text("#!/usr/bin/env python3\nprint('yes')\n", encoding="utf-8")
    loginctl.chmod(0o755)
    cfg_path = tmp_path / f"stub-config{tag}.json"
    cfg_path.write_text(json.dumps(config or {}), encoding="utf-8")
    if unit_dir is not None:
        # A script lifted out of the repo resolves PROJECT_DIR from its own location, so its
        # `systemd/` directory has to sit beside it — otherwise the glob finds no unit and
        # every assertion below would pass on an empty loop, which is a green worth nothing.
        (tmp_path / "systemd").mkdir(parents=True, exist_ok=True)
        for unit in sorted(unit_dir.iterdir()):
            if unit.suffix in (".service", ".timer"):
                staged = tmp_path / "systemd" / unit.name
                if not staged.is_symlink():
                    staged.symlink_to(unit.resolve())
    env = dict(os.environ,
               HOME=str(tmp_path / "home"),
               USER="install-arms-test",
               PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
               STUB_LOG=str(log),
               STUB_CONFIG=str(cfg_path))
    proc = subprocess.run(["bash", str(installer)], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path), timeout=300)
    calls = ([json.loads(line) for line in log.read_text().splitlines()]
             if log.exists() else [])
    return proc, calls


def _enable_calls(calls: list[list[str]]) -> list[list[str]]:
    """The argv of every executed `enable`, and only those.

    Matching is by exact token, so `["--user", "is-enabled", "x.timer"]` — which the same
    script issues right after each arming — is not an arming. This is the predicate an
    `echo`-only installer cannot satisfy: an echoed command never reaches the stub at all.
    """
    return [c for c in calls if "enable" in c and "--user" in c]


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory):
    """One run with every stub answer green — the state clauses 1 and 2 are read from."""
    return _run(INSTALLER, tmp_path_factory.mktemp("plain"))


def test_the_installer_executes_an_enable_for_every_unit_it_links_that_declares_install(plain_run):
    """#2774 clause 1: the `enable` is RUN, and it reaches a stub as argv.

    Set equality, not inclusion: the node fails both if an `[Install]`-bearing unit is left
    unarmed (the #2771 shape) and if the script arms something it was never given
    responsibility for. The `[--user, daemon-reload]` assertion is the positive control that
    this run reached the stub at all — an installer that resolved an absolute `systemctl`
    would arm the box the suite exists to leave alone, and fails here instead.
    """
    proc, calls = plain_run
    assert proc.returncode == 0, f"installer exited {proc.returncode}: {proc.stderr[-800:]}"
    assert ["--user", "daemon-reload"] in calls, "the stub was never consulted"
    assert ARMABLE, "no unit in the tree declares [Install]; this node would be vacuous"

    arms = _enable_calls(calls)
    armed = {c[-1] for c in arms}
    assert armed == set(ARMABLE), f"armed {sorted(armed)}, expected {ARMABLE}"
    assert all(c[0] == "--user" for c in arms), (
        f"an enable went at the system manager: {[c for c in arms if c[0] != '--user']}")
    assert EGRESS_TIMER in armed, "the timer #2770 waits on is still not armed by the script"


def test_a_script_whose_only_enable_sits_inside_an_echo_fails_the_predicate(tmp_path):
    """#2774 clause 1's teeth: the node can fail, and fails on the script this round replaced.

    `test_the_installer_executes_an_enable_...` would be worth nothing if its predicate also
    passed on the old behaviour. So one identical tree runs twice: once with the script as
    committed, and once with each executed `systemctl --user enable` rewritten into an `echo`
    of itself — the exact shape of the installer at HEAD before this round, whose only
    `enable` occurrences were two `echo` arguments at its foot. The rewritten run still exits
    0 and still links every unit, and records no arming whatsoever. That is what makes the
    predicate reject it rather than merely error on it.
    """
    src = INSTALLER.read_text(encoding="utf-8")
    executed = [line for line in src.splitlines()
                if re.match(r'^\s*systemctl --user enable\b', line)]
    assert len(executed) == 2, (
        f"expected two executed arming calls (the timer and service branches), got {executed}")

    home = tmp_path / "copy"
    copy = home / "setup" / "install-services.sh"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(src, encoding="utf-8")
    _, control = _run(copy, home, tag="-control", unit_dir=UNIT_DIR)
    assert {c[-1] for c in _enable_calls(control)} == set(ARMABLE), \
        "the control run of an unmodified copy stopped arming, so the harness is the variable"

    mutated = src
    for line in executed:
        mutated = mutated.replace(line, '        echo "would run: ' + line.strip() + '"')
    assert 'systemctl --user enable' in mutated, "the mutation left no enable to hide"
    assert not re.search(r'^\s*systemctl --user enable\b', mutated, re.M), \
        "the mutation left an executed enable standing"
    copy.write_text(mutated, encoding="utf-8")

    proc, calls = _run(copy, home, tag="-echo", unit_dir=UNIT_DIR)
    assert proc.returncode == 0, f"the echoed script died: {proc.stderr[-600:]}"
    assert (home / "home" / ".config" / "systemd" / "user" / EGRESS_TIMER).is_symlink(), \
        "the echoed script stopped linking units, so the two runs are not comparable"
    assert _enable_calls(calls) == [], (
        f"an echoed `enable` reached systemd as argv, so the node above cannot tell prose "
        f"from action: {_enable_calls(calls)}")


def test_every_timer_is_armed_with_now_and_no_service_is(plain_run):
    """#2774 clause 2: `--now` is the arming half of a timer and a START on a service.

    Bare `enable` links the unit file and leaves the loaded timer `ActiveState=inactive`
    until the user manager next restarts — #2537 measured that on
    `lloyd-persistence-window.timer`, armed by the bare form, reading `enabled`, never
    firing. So every repo timer must carry the flag. And the flag must appear on nothing
    else: since #2541 `enable --now <service>` IS `systemctl start` to the service-control
    guard, which refuses it to an unattended turn and would make this whole installer
    un-runnable by the session that reads it. The classifier is asked directly, so the claim
    is the guard's verdict and not this file's reading of it.
    """
    from app.harness.service_control import find_service_control

    _, calls = plain_run
    arms = _enable_calls(calls)
    assert arms, "no arming to grade"

    for call in arms:
        if "--now" in call:
            assert call[-1].endswith(".timer"), f"`--now` on a non-timer unit: {call}"

    for timer in REPO_TIMERS:
        matching = [c for c in arms if c[-1] == timer]
        assert matching, f"{timer} was never armed"
        assert all("--now" in c for c in matching), f"{timer} armed without --now: {matching}"

    services = [c for c in arms if c[-1].endswith(".service")]
    assert services, "no service arming to check the flag is absent from"
    assert all("--now" not in c for c in services), \
        f"`--now` on a service is a start: {[c for c in services if '--now' in c]}"

    for call in arms:
        label = find_service_control("systemctl " + " ".join(call))
        assert label is None, f"the guard reads {call} as {label}, so a turn cannot run it"
    # Positive control: the classifier is not blind. The service form of the same flag IS a
    # unit operation, which is what makes the loop above a constraint rather than a no-op.
    assert find_service_control("systemctl --user enable --now agent-supervisord.service"), \
        "the guard stopped labelling `enable --now <service>` as a start; this node is vacuous"


def test_units_without_an_install_section_and_system_scope_units_are_never_armed(plain_run):
    """#2774 clause 3, first half: the filter, and the two lists it has to respect.

    Six co-located services read `static`/`linked` because only their timer ever starts them;
    `enable` on one errors, and `set -e` would then end the installer before its own status
    print. `SYSTEM_SCOPE_UNITS` is the same refusal pointing the other way —
    `nvidia-power-limit.service` is not a user unit at all (#1108), so arming a user-scope
    copy of it would produce a unit that looks scheduled and clamps nothing.
    """
    proc, calls = plain_run
    assert proc.returncode == 0

    assert len(SERVICES_WITHOUT_INSTALL) == 6, SERVICES_WITHOUT_INSTALL
    for name in SERVICES_WITHOUT_INSTALL:
        assert not _declares_install(UNIT_DIR / name), \
            f"{name} grew an [Install] section; re-check whether it is now armable"
    assert SYSTEM_SCOPE == {"nvidia-power-limit.service"}, SYSTEM_SCOPE

    armed = {c[-1] for c in _enable_calls(calls)}
    assert not (armed & set(SERVICES_WITHOUT_INSTALL)), \
        f"armed a unit with no [Install]: {sorted(armed & set(SERVICES_WITHOUT_INSTALL))}"
    assert not (armed & SYSTEM_SCOPE), f"armed a system-scope unit: {armed & SYSTEM_SCOPE}"
    assert EGRESS_SERVICE not in armed, (
        "lloyd-egress-window.service was armed: enabling a timer-triggered oneshot into "
        "default.target runs the weekly canary window at every login as well")


def test_the_service_a_timer_starts_is_declared_untouchable_and_why():
    """#2774 clause 3, second half: the one unit the timer-twin rule exists for.

    `lloyd-egress-window.service` is the only `[Install]`-bearing service in the tree with a
    `.timer` beside it, and that combination is the trap. Its `[Install]` says
    `WantedBy=default.target`, so `enable` would start a `Type=oneshot` whose `ExecStart`
    drains the queue for up to an hour and holds the primary engine for up to five — at EVERY
    LOGIN, on top of the Monday 06:23 fire the timer exists to give it, writing ON-arm rows
    into the corpus #2771 publishes from at times no calendar chose. Its sibling
    `lloyd-persistence-window.service` escapes all of this by carrying no `[Install]` at all;
    this unit carries one, so the installer's filter is what stands in. The assertions below
    hold the four facts the filter reasons from, so if a future edit makes the service
    `static`, deletes its timer, or moves the target, the exception becomes visible here
    instead of quietly rotting into a rule nobody can justify.
    """
    assert ARMED_ELSEWHERE, (
        "no [Install]-bearing service has a timer twin any more, so the installer's "
        "timer-twin rule has nothing left to refuse — retire it in the same commit")
    assert EGRESS_SERVICE in ARMED_ELSEWHERE

    body = (UNIT_DIR / EGRESS_SERVICE).read_text(encoding="utf-8")
    assert re.search(r"^Type=oneshot$", body, re.M), "the unit is no longer a oneshot"
    assert re.search(r"^WantedBy=default\.target$", body, re.M), \
        "the [Install] target moved off default.target; re-check the login-time start"
    assert re.search(r"^Persistent=true$", (UNIT_DIR / EGRESS_TIMER).read_text(), re.M), \
        "the timer lost Persistent=true, so the arming rule's catch-up claim is stale"
    assert not _declares_install(UNIT_DIR / "lloyd-persistence-window.service"), \
        "the sibling grew an [Install]; the convention this rule leans on has changed"


def test_a_failing_enable_is_counted_and_the_run_still_finishes(tmp_path):
    """#2774 clause 3, last half: `set -euo pipefail` must not end the run mid-loop.

    Every arming call answers non-zero here, which is what a unit the manager refuses does.
    The script has to keep going, report each failure by name, and exit 0 on its own account:
    a half-armed box whose installer died halfway is the state this item exists to prevent,
    and the two existing stub-harness nodes would notice a non-zero exit before anything
    else did.
    """
    proc, calls = _run(INSTALLER, tmp_path, config={"fail_enable": ARMABLE})
    assert proc.returncode == 0, f"installer exited {proc.returncode}: {proc.stderr[-800:]}"

    arms = _enable_calls(calls)
    for unit in ARMABLE:
        assert [c for c in arms if c[-1] == unit], f"{unit} was never attempted"
        assert f"{unit} — ENABLE FAILED" in proc.stdout, proc.stdout[-1500:]
    assert "Armed 0 unit(s)" in proc.stdout, proc.stdout[-800:]
    summary = [ln for ln in proc.stdout.splitlines()
               if ln.strip().startswith("ENABLE FAILED:")]
    assert summary, proc.stdout[-800:]
    for unit in ARMABLE:
        assert unit in summary[0], f"the summary stopped naming {unit}: {summary[0]}"

    assert (tmp_path / "home" / ".config" / "systemd" / "user" / EGRESS_TIMER).is_symlink(), \
        "a refused enable stopped the link loop, which is the wrong coupling"
    assert ["--user", "daemon-reload"] in calls


def test_an_enabled_but_inactive_timer_is_visible_on_stdout(tmp_path):
    """#2774 clause 4: `is-enabled` alone is the verdict that already lied once.

    The stub is told the unit FILE is `enabled` and the loaded unit is `inactive` — the exact
    state #2537 found `lloyd-persistence-window.timer` in for its whole first life, and the
    state a bare `enable` leaves a timer in. On that answer a block printing `is-enabled`
    only would print `enabled` and read green; the arming block prints both halves, so the
    miss is on stdout where a human and the log both see it.
    """
    proc, _ = _run(INSTALLER, tmp_path,
                   config={"state_default": "enabled", "active_default": "inactive"})
    assert proc.returncode == 0

    lines = [ln for ln in proc.stdout.splitlines() if f"{EGRESS_TIMER} — armed" in ln]
    assert len(lines) == 1, proc.stdout[-1500:]
    assert "is-enabled=enabled" in lines[0], lines[0]
    assert "ActiveState=inactive" in lines[0], \
        f"ActiveState is missing or wrong, so an un-armed timer reads green: {lines[0]}"

    for unit in ARMABLE:
        unit_lines = [ln for ln in proc.stdout.splitlines() if f"{unit} — armed" in ln]
        assert unit_lines and "is-enabled=" in unit_lines[0] and "ActiveState=" in unit_lines[0], \
            f"{unit} has no line carrying both halves: {unit_lines}"


def test_the_guard_verdict_is_printed_but_its_exit_status_never_decides_the_installer(tmp_path):
    """#2774 clause 5's parenthetical, and the reason the exit code is only reported.

    The installer ends by running `scripts/maintenance/check-unit-enabledness.sh` and printing
    its verdict, so the arming arithmetic above and the fleet check cannot disagree — the item
    asked for the check's own exit status rather than a hand-rolled count. It is REPORTED and
    never propagated, because the two existing stub-harness nodes run this script against a
    stub that answers nothing: under such a stub every timer reads un-enabled and the guard
    exits 1, so propagating it would turn both of those green tests red and make the installer
    accuse a box that did nothing wrong. Both edges are pinned here: a red guard leaves the
    installer at 0, and a green guard says so.
    """
    # Derived, never recalled: the count in the two verdict lines below is this same number,
    # so the only thing to pin here is that the tree has timers to count at all. A `total`
    # of 0 would make both f-strings below `checked 0 timers: 0 enabled, 0 not enabled` and
    # pass on a guard that had silently stopped looking at anything.
    total = len(REPO_TIMERS)
    assert total > 0, "no repo timers to count; both verdict assertions below would be vacuous"

    red, _ = _run(INSTALLER, tmp_path / "red", config={"state_default": "disabled"}, tag="-red")
    assert red.returncode == 0, f"the installer inherited the guard's failure: {red.returncode}"
    assert f"checked {total} timers: 0 enabled, {total} not enabled" in red.stdout, \
        red.stdout[-1500:]
    assert GUARD.name in red.stdout, "the guard's verdict was never printed"
    assert "guard exit status: 1" in red.stdout, red.stdout[-1500:]

    green, _ = _run(INSTALLER, tmp_path / "green", config={"state_default": "enabled"},
                    tag="-green")
    assert green.returncode == 0
    assert f"checked {total} timers: {total} enabled, 0 not enabled" in green.stdout, \
        green.stdout[-1500:]
    assert "guard exit status: 0" in green.stdout, green.stdout[-600:]


def test_the_health_check_still_reads_the_scope_array_the_installer_declares():
    """The one Python seam this change crosses, tested across it.

    `scripts/service_health_check.py` neither imports nor runs the installer: it
    regex-extracts `SYSTEM_SCOPE_UNITS=(...)` out of the shell text (`_system_scope_units`,
    read by `check_unit_file_links`) so a unit added to the exclusion list is graded without
    anyone editing the checker (#2714/#2756). A rewrite of that declaration line — a line
    break inside the parentheses, a rename — reads fine to bash and silently changes what the
    health check files as `missing:`. The code graph is blind to this edge (bash is not in
    it), which is why the node asks the real reader instead of grepping for the array.
    """
    spec = importlib.util.spec_from_file_location("shc_arms_units", HEALTH_CHECK)
    shc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shc)

    assert shc._system_scope_units(INSTALLER) == {"nvidia-power-limit.service"}, \
        "the health check no longer reads the installer's exclusion list"
