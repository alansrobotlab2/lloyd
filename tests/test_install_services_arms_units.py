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

WHAT THIS FILE ALSO PINS SINCE #3300, on a different seam: a unit's `Documentation=`
target. `nvidia-power-limit.service` spent weeks pointing `systemctl status` at a copy of
the Xid 79 report that git never tracked, and no node here or anywhere else had ever
opened a `Documentation=` line to see whether its target existed. The last three nodes
parse every `file://` target in `agent-services/systemd/` — recursively, so the `system/`
subtree is inside the scan — and judge the parsed path by whether it exists, never by
comparing it to an absolute string, plus one planted-dead-target node so an empty scan
cannot be mistaken for a passing one.
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

#: The one unit whose `Documentation=` target was dead on this box: it named a copy
#: of the Xid 79 report inside the checkout that git never tracked (#3214/#3300).
NVIDIA_UNIT = UNIT_DIR / "nvidia-power-limit.service"

#: One `Documentation=` value per line, the way `systemd.unit(5)` reads the setting:
#: several lines are allowed and each holds one whitespace-separated URL.
_DOC_VALUE = re.compile(r"^Documentation=(\S+)", re.M)

_FILE_SCHEME = "file://"


def _main_checkout() -> Path:
    """The main worktree's root — the checkout a linked worktree is a copy of.

    A round runs these nodes from `~/lloyd-work/<round>/home/lloyd`, where `ROOT` is a
    worktree and an absolute path from a unit file is inside nothing there. Judging
    "inside the checkout" against `ROOT` would therefore call the dead
    `<home>/lloyd/gpu-xid79-falloff-report.md` target outside it and pass the node on
    the exact line this round exists to fix. `git rev-parse --git-common-dir` answers
    `<main>/.git` from any linked worktree, so its parent is that checkout — and
    nothing here spells the account name, which is what let the line rot: an absolute
    path written down once in a test would break the day the home directory moves.
    """
    proc = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"git rev-parse failed: {proc.stderr}"
    common = Path(proc.stdout.strip())
    if not common.is_absolute():
        common = ROOT / common
    return common.resolve().parent


def _doc_targets(unit_dir: Path) -> dict[Path, list[Path]]:
    """Every unit under this directory, keyed to the `file://` targets it declares.

    Recursive on purpose. `agent-services/systemd/system/lloyd-data-snapshot-prune.service`
    exists, so a top-level `iterdir()` (which `ALL_UNITS` above legitimately uses,
    because the installer's own glob is top-level) sees nine of the ten `file://`
    targets on this box and reports a clean bill of health on a tenth it never
    opened. Every unit scanned is a key, with an empty list when it declares
    nothing, so a caller can tell "looked and found no target" from "never looked".
    A value that is not a `file://` URL is not a path and is left to the node that
    counts the values, which asserts nothing was dropped without being graded.
    """
    found: dict[Path, list[Path]] = {}
    for unit in (sorted(unit_dir.rglob("*.service"))
                 + sorted(unit_dir.rglob("*.timer"))):
        values = _DOC_VALUE.findall(unit.read_text(encoding="utf-8"))
        found[unit] = [Path(v[len(_FILE_SCHEME):]) for v in values
                       if v.startswith(_FILE_SCHEME)]
    return found


def _dead_doc_targets(unit_dir: Path) -> list[tuple[Path, Path]]:
    """The `(unit, target)` pairs whose `Documentation=file://` target does not exist."""
    return [(unit, target)
            for unit, targets in _doc_targets(unit_dir).items()
            for target in targets
            if not target.exists()]


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


def test_every_documentation_file_target_in_the_unit_tree_resolves_on_this_box():
    """Clause 2: every `file://` target under `agent-services/systemd/`, `system/` included.

    The process boundary is systemd itself: a unit file is read by PID 1 long after
    this suite is green, and `systemctl status` is the only reader that ever complains
    — which is how `nvidia-power-limit.service` shipped a target that had never
    existed while this file stayed green (#3214 → #3300, three filings, no attempt).
    Nothing outside the repo is touched, so the honest test is to read the real tree
    the way systemd does and ask each parsed target whether it is there.

    Two things make the green mean something. The nested-coverage check below is what
    a top-level glob fails: without it, `iterdir()` would satisfy every other
    assertion in this node while silently leaving out the `system/` subtree. And the
    target count is asserted before the verdict, so a helper that stopped parsing
    (`file://` spelled differently, the regex anchored wrong) cannot report "zero
    dead targets" as a pass — the planted-dead case is in its own node, below.
    """
    scanned = _doc_targets(UNIT_DIR)
    n_targets = sum(len(targets) for targets in scanned.values())
    assert n_targets > 0, (
        "not one Documentation=file:// target was parsed under agent-services/systemd/, "
        "so the verdict below is a report about nothing")

    nested = sorted(p for p in UNIT_DIR.rglob("*")
                    if p.suffix in (".service", ".timer")
                    and p.is_file() and p.parent != UNIT_DIR)
    assert nested, f"{UNIT_DIR} has nothing below it, so the recursion claim is vacuous"
    unseen = [str(p.relative_to(UNIT_DIR)) for p in nested if p not in scanned]
    assert not unseen, (
        f"the scan stopped at the top level and never opened: {unseen} — a dead target "
        "there would be invisible to this node")

    dead = _dead_doc_targets(UNIT_DIR)
    assert not dead, "dead Documentation= targets: " + "; ".join(
        f"{unit.relative_to(UNIT_DIR)} -> {target}" for unit, target in dead)

    # Every value the parser saw was either graded as a path above or is a web link
    # that names no path. A third kind of scheme would have been dropped by
    # `_doc_targets` without a word, and this is where that shows up.
    values = [v for unit in scanned
              for v in _DOC_VALUE.findall(unit.read_text(encoding="utf-8"))]
    web = [v for v in values if v.startswith("https://")]
    assert len(values) == n_targets + len(web), (
        f"{len(values)} Documentation= values, {n_targets} graded as paths and "
        f"{len(web)} as links: something was skipped without being graded")
    assert web, "no web documentation left to compare against; the split above is vacuous"


def test_the_power_limit_unit_names_a_report_that_exists_outside_the_checkout():
    """Clauses 1 and 5: the parsed target resolves, and no copy of it was committed.

    No assertion below compares the target to a written-down path. That is the
    clause's point, and the mechanism of the original bug: `/home/alansrobotlab/lloyd/
    gpu-xid79-falloff-report.md` was a copy that was never committed, and a test that
    re-asserted that literal would have passed on the dead line and broken on a
    renamed home directory. So the value is parsed out of the unit and judged on what
    is true of the report and of nothing else that could stand in for it — it is a
    file, its first line names the fault the unit mitigates, and it is not inside the
    checkout. `git ls-files` then closes the cheapest wrong fix: committing a copy
    into the tree to make the target resolve, which #3214 ruled against (the report
    is RMA paperwork and lives beside the rest of it, in `~/rma`).
    """
    scanned = _doc_targets(UNIT_DIR)
    assert NVIDIA_UNIT in scanned, f"{NVIDIA_UNIT.name} was never scanned"
    targets = scanned[NVIDIA_UNIT]
    assert len(targets) == 1, f"expected one Documentation= target, parsed {targets}"
    target = targets[0]

    assert target.is_file(), (
        f"{NVIDIA_UNIT.name} points `systemctl status` at {target}, which is not a "
        "file on this box")
    assert "Xid 79" in target.read_text(encoding="utf-8").splitlines()[0], (
        f"{target} exists but its first line does not name Xid 79, so this is not the "
        "falloff report the unit exists to explain")

    main = _main_checkout()
    assert not target.is_relative_to(main), (
        f"{target} lies inside the checkout at {main}, where the report is not "
        f"tracked — a Documentation= line pointing into the tree is a dead path again")

    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files",
                             "*gpu-xid79-falloff-report*"],
                             capture_output=True, text=True, timeout=60)
    assert tracked.returncode == 0, tracked.stderr
    assert tracked.stdout.strip() == "", (
        "a copy of the report was committed to the checkout to make the target "
        f"resolve instead of the line being repointed: {tracked.stdout.strip()}")


def test_a_dead_target_planted_under_a_throwaway_unit_dir_is_the_one_reported(tmp_path):
    """Clause 3: the helper's silence means "no dead target", never "I found nothing".

    An absence assertion on the live tree is one broken parser away from permanent
    green, so the same two functions are pointed at a throwaway unit directory built
    here: one unit whose target is a file this repo really has, and one planted under
    `system/` — nested, because that subtree is where the live scan could go wrong —
    whose target is never written. The planted pair must be the exact report, and the
    real tree must come back clean in the same node, so a helper that reports nothing
    at all fails here on the day it breaks rather than in the next month a unit ships
    a dead path.
    """
    units = tmp_path / "units"
    (units / "system").mkdir(parents=True)
    live = units / "ok.service"
    live.write_text(f"[Unit]\nDocumentation={_FILE_SCHEME}{INSTALLER}\n", encoding="utf-8")
    planted = units / "system" / "planted-dead.service"
    missing = tmp_path / "report-nobody-wrote.md"
    planted.write_text(f"[Unit]\nDocumentation={_FILE_SCHEME}{missing}\n", encoding="utf-8")
    assert not missing.exists() and INSTALLER.is_file()

    scanned = _doc_targets(units)
    assert len(scanned) == 2, (
        f"the throwaway tree holds two units; the scan returned {sorted(scanned)}")
    dead = _dead_doc_targets(units)
    assert dead == [(planted, missing)], (
        "the planted dead target under system/ was not what the helper reported: "
        f"{dead} — a green on the live tree means nothing until this fires")

    assert _dead_doc_targets(UNIT_DIR) == [], \
        "the real unit tree reads as dead alongside a passing fixture: a real regression"
