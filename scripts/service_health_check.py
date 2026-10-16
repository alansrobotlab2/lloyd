#!/usr/bin/env python3
"""Service Health Check Skill

Bundle multiple service status checks into a single call.
Returns structured status for LLM, MCP, and other Lloyd services.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SUPervisor_CONF = "/home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf"

# Test seam (#1649). Under pytest, a test may point the supervisor path at a stand-in
# by exporting UNIT. Guarded on PYTEST_CURRENT_TEST so it cannot fire on a live run:
# this script is the thing a human runs when the box is broken, and an environment
# variable that silently redirects which supervisor it reads — into a nonexistent
# conf, or into one that answers what an attacker wants — would make every row it
# prints a lie about a machine that is fine. Outside pytest this line changes
# nothing, and the only test that uses it passes a path to a stub executable.
if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("UNIT"):
    SUPervisor_CONF = os.environ["UNIT"]

#: A supervisor read that never reached supervisor is reported as this (#2652).
#:
#: The distinction is the whole point, and the old word erased it. A read that raised
#: and a service that is down both used to land on `status: "error"`, and `_scored_rows`
#: counts every row that is not advisory — so the checker's own missing PATH was billed
#: to the fleet as six faults on a box whose six services were RUNNING. `unverified`
#: says what actually happened: nothing was asked, so nothing was graded. Such a row is
#: excluded from both sides of `Overall: N/M`, exactly as an advisory row is, and it
#: still prints, with the reason it could not be read.
UNVERIFIED_STATUS = "unverified"

SUPERVISOR_NAME = "supervisorctl"

#: Where this box keeps the binary, for a process whose PATH does not contain it.
_SUPERVISOR_INSTALLS = ("/home/alansrobotlab/.local/bin/supervisorctl",)


class UnreadableFleetRead(str):
    """The reason a fleet-wide `status` could not be read at all (#2652 clause 4).

    A `str` subclass rather than a new return shape, so every existing reader of
    `supervisor_fleet()`'s second element — the `if error is not None` test, the
    f-string that puts the reason into a row, and the tests that compare the text —
    keeps working unchanged. The type exists for one decision it cannot otherwise make:
    a read that *raised* never reached supervisor, so the rows it would have produced are
    `unverified`; a read that answered with something unusable (`no usable program lines
    (exit 3)`, this box's steady state while two hand-started units sit STOPPED) is an
    answer about the fleet and stays a graded fault. Grading the second like the first
    would let a healthy fleet be reported unreadable.
    """


def resolve_supervisor() -> str:
    """An absolute, executable path to supervisorctl; the bare name if nothing resolves.

    The bare name was the defect (#2652). This script runs from a shell whose PATH is
    `/usr/local/bin:/usr/bin`, and supervisorctl is installed in `~/.local/bin`, so
    every `subprocess.run(["supervisorctl", ...])` raised `FileNotFoundError` before
    supervisor was contacted — six RUNNING services printed `[✗] error` and `Overall:`
    billed the fleet for the checker's own miss. PATH is asked first, so an operator or
    a test that puts another supervisor on PATH still gets the answer it asked for; the
    known install is the fallback. The bare name survives only where nothing resolved at
    all, which is the one case that is the fleet's business rather than the checker's —
    and that row is reported `unverified`, not as a fault.
    """
    found = shutil.which(SUPERVISOR_NAME) or next(
        (p for p in _SUPERVISOR_INSTALLS if os.access(p, os.X_OK)), None)
    return found or SUPERVISOR_NAME


#: argv[0] for every supervisor read in this file: absolute where the binary was found,
#: `SUPERVISOR_NAME` where it was not, which is what makes a read raise and its row
#: land in `UNVERIFIED_STATUS` instead of the fraction.
SUPERVISORCTL = resolve_supervisor()


def is_supervisor_call(argv) -> bool:
    """True when this argv asks supervisor something, however argv[0] is spelled.

    By basename — not by equality, not by substring. Equality on argv[0] is what made
    the resolved path a silent regression: `_sup_program()` matched only the bare name,
    so once argv[0] carried a directory it returned None for `lloyd-mc:lloyd-backend`,
    that program fell out of `declared_programs()`, and the derived pass (#1649)
    re-reported a *declared* service as an undeclared one. A substring test over argv[0]
    (the other form in this file) matches a stand-in named `supervisorctl-stub` that is
    not supervisor at all. Neither form is about the program, so neither is kept.
    """
    return bool(argv) and os.path.basename(str(argv[0])) == SUPERVISOR_NAME


SERVICES = {
    # Supervisor services — status via supervisorctl, no HTTP check
    "lloyd-backend": {"command": [SUPERVISORCTL, "-c", SUPervisor_CONF, "status", "lloyd-mc:lloyd-backend"], "category": "lloyd"},
    "lloyd-frontend": {"command": [SUPERVISORCTL, "-c", SUPervisor_CONF, "status", "lloyd-mc:lloyd-frontend"], "category": "lloyd"},
    "lloyd-mcp": {"command": [SUPERVISORCTL, "-c", SUPervisor_CONF, "status", "lloyd-mc:lloyd-mcp"], "category": "lloyd"},

    # LLM inference servers — status via supervisorctl, no HTTP check
    "agent-llm-primary": {"command": [SUPERVISORCTL, "-c", SUPervisor_CONF, "status", "agent-llm-primary"], "category": "supervisor"},
    "agent-llm-secondary": {"command": [SUPERVISORCTL, "-c", SUPervisor_CONF, "status", "agent-llm-secondary"], "category": "supervisor"},
    # djev shares GPU 2 with the secondary, so at most one of these two is ever
    # enabled. Port 8011 is the structured decision API, which start-djev.sh
    # only launches once vLLM on 8010 answers — so an open 8011 means the whole
    # stack is serving, while an open 8010 during a cold boot does not.
    "agent-djev": {"command": [SUPERVISORCTL, "-c", SUPervisor_CONF, "status", "agent-djev"], "category": "supervisor", "port": 8011},

    # QMD retrieval daemon — the path every vault_search/vault_recall needs,
    # and until #406 this check could not see it at all. Port is the one
    # `agent-services/supervisor/conf.d/agent-qmd-daemon.conf` launches it on
    # (`qmd mcp --http --port 8181`); it serves a real /health route, so a
    # supervisor RUNNING line that no longer has a listener behind it — the
    # failure this entry exists to catch — now shows up here instead of in the
    # first failed vault_search of the day. This is also the first SERVICES
    # entry to declare a `port`, i.e. the first thing to ever exercise
    # http_check().
    "agent-qmd-daemon": {"command": [SUPERVISORCTL, "-c", SUPervisor_CONF, "status", "agent-qmd-daemon"], "category": "retrieval", "port": 8181},
}

# Every state supervisord can report for a program. A state outside this table is
# not a service that is down — it is a check that did not read anything (#1040
# learned that about exit codes; #1649 applies it to a program supervisor was
# never asked about at all).
SUP_PROGRAM_STATES = {
    "RUNNING", "STARTING", "STOPPING", "STOPPED", "BACKOFF", "FATAL",
    "UNKNOWN", "EXITED", "CREATING",
}

# The category a row graded on process state alone lands in: its own `Fleet:`
# heading, not one of the four existing ones. #1649's design note asks the
# implementer to choose this deliberately, because the per-category summary marks a
# category degraded when ANY member is unhealthy. `Supervisor:` already means
# "agent-djev" (the one declared entry carrying that category), so filing the vault
# sync, tts, livekit, the qmd watcher and the agent worker there would let tts
# stopping turn a heading whose red currently means "the decision engine is down" —
# the reader would go looking for djev. A distinct bucket keeps every existing
# heading meaning exactly what it meant before this landed, and `Fleet:` is the
# honest label: every program supervisor has loaded that no probe was written for.
# What these rows DO share with the declared ones is `Overall:` — an undeclared
# program in FATAL must be able to stop this run printing `8/8 services healthy`,
# which is the bug. Whether the new heading ever gets noisy is live traffic, so
# #1649 owes one post-landing read of a real run.
FLEET_CATEGORY = "fleet"


def _sup_program(svc: dict) -> Optional[str]:
    """The supervisord program a SERVICES entry refers to, or None.

    `SERVICES` is keyed by a label, not by a program name: the entry called
    `lloyd-backend` asks supervisor about `lloyd-mc:lloyd-backend`, and `lloyd-mc:`
    is supervisor's group prefix — it is the program name `supervisorctl status`
    prints. Coverage derived from the dict's KEYS would therefore report
    `lloyd-mc:lloyd-backend` as an undeclared program a second time, and the label
    `lloyd-backend` as a program supervisor never heard of.
    """
    cmd = svc.get("command") or []
    if len(cmd) >= 5 and is_supervisor_call(cmd) and "status" in cmd:
        return cmd[-1]
    return None


def declared_programs() -> set:
    """Programs the declared pass already reports, so the derived pass skips them.

    Both halves matter. The key is the program name for the entries whose command
    is not a supervisorctl call — `agent-llm-primary` curls :8000 — and the
    command's last word is it for the three `lloyd-mc:` group members, whose keys
    are unqualified labels. Deriving from keys alone would report
    `lloyd-mc:lloyd-backend` as undeclared, and derive from commands alone would
    report `agent-llm-primary` twice.
    """
    return set(SERVICES) | {p for p in (_sup_program(s) for s in SERVICES.values())
                            if p}


_FLEET: Optional[dict] = None


# A supervisord program name: `[group:]process`, alphanumerics and `._-`. Used to
# tell a status line from an error line — `unix:///run/supervisor/supervisor.sock`
# and `error:` both fail it, every name in `conf.d/` passes it.
_PROGRAM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?::[A-Za-z0-9._-]+)?\Z")


def supervisor_fleet() -> tuple:
    """`({program: state-or-None}, error_or_None)` over every loaded program.

    One fleet-wide `supervisorctl -c CONF status`, parsed positionally: field 0 is
    the program (group prefix included), field 1 the state. A line with no state
    field keeps its program with state None rather than being dropped, because a
    program the checker silently skipped is the bug #1649 is about.

    The error half is load-bearing: a script that could not reach supervisor and
    printed no undeclared rows would look exactly like a box with no undeclared
    programs, which is how a coverage check passes for the wrong reason.

    A nonzero exit is NOT the error signal here. `supervisorctl status` exits 3
    whenever any loaded program is not RUNNING, which on this box right now means
    EVERY run: `agent-llm-secondary` is STOPPED by config, so `rc != 0` is the
    steady state and reading it as "unreadable" would have made this fix grade
    zero programs on a healthy machine and only work once something broke. The
    answer is the parse: a line whose first field is not shaped like a program
    (`unix://… refused connection`, `error: <class …>`) is not a program, and a
    fleet answer containing one is not a fleet answer at all — so an unparseable
    line is reported as an unreadable fleet rather than invented as a service. The
    exit code is never consulted. Fail-loud in both directions is the point: a
    silently skipped line is the exact bug #1649 is about, so the parse refuses to
    guess which lines it did not understand.
    """
    global _FLEET
    if _FLEET is None:
        fleet, error = {}, None
        try:
            r = subprocess.run(
                [SUPERVISORCTL, "-c", SUPervisor_CONF, "status"],
                capture_output=True, text=True, timeout=10)
            noise = []
            for line in r.stdout.splitlines():
                fields = line.split()
                if not fields:
                    continue
                if not _PROGRAM.match(fields[0]):
                    noise.append(line.strip()[:80])
                    continue
                fleet[fields[0]] = fields[1] if len(fields) > 1 else None
            if not fleet or noise:
                detail = "; ".join(noise[:2]) or ((r.stdout or "") + " "
                                                  + (r.stderr or "")).strip()[:80]
                error = (f"no usable program lines (exit {r.returncode})"
                         if not fleet else
                         f"{len(noise)} line(s) were not program lines") + (
                    f": {detail}" if detail else "")
        except Exception as exc:
            # Typed, because the two failure shapes have to be told apart downstream:
            # this one is supervisor never having been reached, while the `not fleet`
            # branch above is an answer that made no sense (#2652 clause 4).
            error = UnreadableFleetRead(str(exc)[:160])
        _FLEET = (fleet, error)
    return _FLEET


def _ask_program(name: str, errors: Optional[list] = None) -> Optional[str]:
    """Re-ask supervisor for one program's state, when the fleet read named it but gave
    no usable state for it.

    The condition is narrower than "the fleet read was useless": when the read did not
    happen at all, `derived_supervisor_rows` returns the unverified row first and no
    program reaches this function — one failed exec must not become N probes (#2652).

    The one probe a derived program gets, and it is the same read with one name
    added: `supervisorctl -c CONF status <program>`, whose answer is one line and
    whose state is still field 2 (#1040). A fleet-wide `status` that garbled or
    dropped one program's state field should not be answered with a guess about a
    process that may be up.

    For `agent-obsidian-sync` in particular this is the entire allowance. #1141:
    the round-trip probe this skill used to run against the vault sync deleted the
    live sync registration on 09-14. A process-state read has no side effect — it
    opens no port, runs no script, reads no sentinel file and no lock, asks
    supervisor nothing it does not already track — so it cannot cost anything the
    service it is measuring owns.

    `errors` is an out-parameter, and it exists for one distinction: an answer with no
    state field is supervisor telling us something, while a read that never started is
    us failing to ask. `_process_state_row` grades the first and reports the second
    `unverified` (#2652), so a checker that cannot reach supervisor must not be able to
    turn a healthy fleet into a list of faults.
    """
    try:
        r = subprocess.run(
            [SUPERVISORCTL, "-c", SUPervisor_CONF, "status", name],
            capture_output=True, text=True, timeout=10)
        fields = (r.stdout or "").split()
        return fields[1] if len(fields) > 1 else None
    except Exception as exc:
        if errors is not None:
            errors.append(str(exc))
        return None


def _state_verdict(name: str, state: Optional[str]) -> tuple:
    """(healthy, status) for one program, from its state word and nothing else.

    No exit code, because a fleet-wide `status` does not carry one per program,
    and #1040's rule still holds: the state is the verdict. The declared seven
    keep their own per-name call (which does report an exit code, and is the
    difference between a service that crash-looped and one that exited once); this
    is the cheaper grade for the programs nobody declared, where the question is
    only "is the process up".
    """
    if state != "RUNNING" and name in _switched_off():
        # Deliberately stopped by config, which `app/llm_slots` owns; #699 ruled a
        # switched-off slot invisible rather than "stopped", and the set is read
        # from that module. There is no second list of names here to rot.
        return True, (f"expected stopped: {name} is switched off in config — "
                      "not a fault")
    if state is None:
        return False, ("no state field in supervisorctl status — nothing was "
                       "graded, which is a broken check, not a stopped service")
    if state not in SUP_PROGRAM_STATES:
        return False, (f"state {state!r} is not a supervisord program state, so "
                       "the answer could not be graded — a broken reading, not a "
                       "stopped service")
    if state == "RUNNING":
        return True, "RUNNING (process state only — no probe declared)"
    return False, f"{state} (process state only — no probe declared)"


# supervisord's own false words for a boolean option (supervisor/options.py
# `readboolean`), so only one of these counts as "declared hand-started". Anything
# else — `true`, a value that is neither, or no line at all — leaves STOPPED a
# graded fault, which is the fail-closed direction: the excuse is granted on a
# declaration that exists, never on a read that failed.
_AUTOSTART_OFF = ("false", "0", "no", "off")

_AUTOSTART_LINE = re.compile(r"autostart\s*[=:]\s*(\S+)", re.IGNORECASE)


def _unit_dir() -> Path:
    """The `conf.d` directory beside the supervisord.conf this run is reading.

    Derived from `SUPervisor_CONF`, never hard-coded, because that constant is what
    the `UNIT` seam re-points: a unit directory that stayed behind on the live tree
    would make every test that names a throwaway supervisord.conf either unable to
    reach its own units or, worse, graded against the box's real ones.
    """
    return Path(SUPervisor_CONF).resolve().parent / "conf.d"


def _section_autostart(text: str, program: str) -> Optional[str]:
    """The `autostart` value the `[program:<program>]` section of one unit declares.

    Section-scoped, because that is what makes the declaration mean something. The
    live candidate's own file carries a comment line reading
    `; autostart=false for the same reason as ...` above its section, and a
    whole-file substring test would read that as the declaration — which happens to
    agree today and would disagree the moment a unit is switched on while an
    explanatory comment survives. So: only a line inside the named program's own
    section counts, only a `key=value` line (a `;`/`#` comment is not a declaration),
    and a later line in the same section wins, as supervisord's own INI read does.

    Returns the raw value word, or None when no such line was found.
    """
    wanted = f"program:{program}"
    value: Optional[str] = None
    in_wanted = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_wanted = line[1:-1].strip() == wanted
            continue
        if not in_wanted or line[:1] in (";", "#"):
            continue
        found = _AUTOSTART_LINE.match(line)
        if found:
            value = found.group(1).strip().lower()
    return value


def _hand_started_unit(program: str) -> Optional[Path]:
    """The unit in `conf.d` that declares this program hand-started, else None.

    `autostart=false` is the operator's own declaration that supervisord must not
    start this program at boot. On this box that is how a candidate is held while it
    is measured against the service it would replace — GPU 2 carries one tenant, so
    the candidate stays down until someone stops the live one and starts it by hand.
    Such a program is loaded, so the derived pass rightly reports it (#1649); what it
    must not do is grade "nobody started the candidate" as a fleet fault, because
    that is the box working exactly as declared, and a `Fleet:` that reads degraded
    on every single run is the noise #699 warns a reader past.

    Read from the unit file rather than from a list of names in this script: the
    declaration is the fact, the list would be a copy of it, and a copy is what made
    the retired hand-maintained coverage list rot.
    """
    try:
        units = sorted(_unit_dir().glob("*.conf"))
    except OSError:
        return None
    for path in units:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _section_autostart(text, program) in _AUTOSTART_OFF:
            return path
    return None


def _process_state_row(name: str, state: Optional[str],
                       declared: bool = False) -> dict:
    """One derived row: `declared` says a `SERVICES` probe already reports it.

    Only a program with no `SERVICES` entry can be excused as hand-started. A
    declared entry in a non-RUNNING state is a fault whatever its unit declares:
    someone wrote a probe for it because the box is supposed to be running it, and
    `autostart=false` on such a unit (both GPU-2 tenants declare it) means the boot
    reconcile brings it up, not that nobody cares. `agent-llm-secondary` while its
    slot is switched off is the other excuse and it wins — `_state_verdict` owns
    that wording, and it is config-derived, which is a stronger claim about this box
    than a unit file that never changes.
    """
    if state is None:
        # The fleet-wide answer gave this program no state field. Ask for that one
        # program before calling the reading broken — same command with a name
        # appended, no new kind of probe, and a process that is up gets its true
        # state instead of a false alarm.
        asked = []
        state = _ask_program(name, asked)
        if state is None and asked:
            # The re-ask raised, so supervisor was never reached at all. Nothing was
            # learned about this program, and "nothing learned" must not be filed as
            # "this service is down" (#2652: a bare argv[0] made this the fate of every
            # derived row, on a box where every service was RUNNING).
            return {"name": name, "category": FLEET_CATEGORY, "healthy": False,
                    "status": f"{UNVERIFIED_STATUS} — supervisor was not asked "
                              f"({asked[0]})",
                    "exit_code": -1, "unverified": True}
    healthy, status = _state_verdict(name, state)
    row = {"name": name, "category": FLEET_CATEGORY, "healthy": healthy,
           "status": status, "exit_code": 0}
    # STOPPED is the only state a declaration can excuse, and it is the only one
    # this asks about: STARTING means supervisord is bringing it up, and BACKOFF,
    # EXITED or FATAL mean a start was attempted and failed — a hand-started unit is
    # expected to be down, never to be crash-looping. So the unit file is opened for
    # a candidate at rest and for nothing else.
    if not healthy and not declared and state == "STOPPED":
        unit = _hand_started_unit(name)
        if unit is not None:
            # `healthy` stays False — the row is not claiming a stopped process is
            # up — and `advisory` is what takes it out of the arithmetic
            # (`_scored_rows`, the `Overall:` count and the category words), on the
            # same declaration `ca-trust` uses (#1726). The word is the operator's:
            # it says the stop was declared and where, so the row can be read without
            # opening the conf.
            row["advisory"] = True
            row["verdict"] = "warn"
            row["status"] = (f"STOPPED by design — {unit.name} declares "
                             f"autostart=false, hand-started: reported, not counted")
    return row


def derived_supervisor_rows() -> list:
    """One row per loaded program that `SERVICES` does not refer to (#1649).

    This is the whole fix: the checker's coverage was a hand-maintained list of
    seven labels while supervisor ran twelve programs, so `agent-livekit-server`,
    `agent-obsidian-sync`, `agent-qmd-watcher`, `agent-tts` and
    `lloyd-agent-worker` were never read and could sit in FATAL beside an
    `Overall: 8/8 services healthy`. Derived from `status`, a program is graded the
    day supervisor starts it, with no edit to this file.

    Process state only, by ruling and by incident: `agent-obsidian-sync` gets
    `supervisorctl status` and nothing else. #1141 records that a round-trip probe
    of the vault sync deleted the live sync registration on 09-14 — a health check
    that costs the thing it measures is worse than no check, so no port
    connection, sentinel read or other side-effecting call is made for any derived
    program.
    """
    fleet, error = supervisor_fleet()
    if error is not None:
        row = {"name": "supervisorctl status", "category": FLEET_CATEGORY,
               "healthy": False, "exit_code": 0,
               "status": f"could not read the loaded program list ({error}) — no "
                         "undeclared program was graded, so this run covered only "
                         "the declared services"}
        if isinstance(error, UnreadableFleetRead):
            # Only a read that RAISED. The other shape of `error` — an answer carrying
            # no usable program lines, `(exit 3)` being this box's steady state while
            # two hand-started units sit STOPPED — is supervisor replying, and stays a
            # graded fault, or a healthy fleet could be reported unreadable (#2652
            # clause 4). The self-scope sentence is unchanged either way.
            row["status"] = f"{UNVERIFIED_STATUS}: {row['status']}"
            row["unverified"] = True
        return [row]
    if not fleet:
        return [{"name": "supervisorctl status", "category": FLEET_CATEGORY,
                 "healthy": False, "exit_code": 0,
                 "status": "reported no programs at all — that is not a healthy "
                           "fleet, it is an empty answer, so nothing outside the "
                           "declared services was graded"}]
    declared = declared_programs()
    off = _switched_off()
    rows = []
    for name, state in sorted(fleet.items()):
        # A program gets a row unless the declared pass is already reporting it —
        # and only then. `and name not in off` is the half that matters: `main`
        # removes every switched-off name from the list it checks, so a declared
        # service whose slot is off has no probe row, and skipping it here as well
        # loses the program from the report entirely. The review rung caught
        # exactly that: `_switched_off()` is config-derived and can name
        # `agent-tts`, `agent-voice-mcp` or `agent-voice-mode`, which would have put
        # this check back to not covering the units #1649 is about — and this time
        # the five rows would go missing conditionally, only on a box where someone
        # switched a slot off, so the suite could never see it.
        #
        # `_state_verdict` owns the wording: a switched-off program that is not
        # RUNNING reads `expected stopped`, and one that IS RUNNING reads RUNNING,
        # because "config says off, supervisor says up" is a fact a health report
        # must not silently drop.
        if name in declared and name not in off:
            continue
        # `name in declared` is passed down rather than re-derived there: a declared
        # entry whose slot is switched off lands here for the row the declared pass
        # was told to skip, and it must not be able to borrow the hand-started excuse
        # (#1818 clause 2's other half).
        rows.append(_process_state_row(name, state, name in declared))
    return rows


def _switched_off() -> set:
    """Supervisord programs config.yaml has deliberately switched off.

    The optional LLM slots share GPU 2, so one of them is always stopped on
    purpose, and reporting it as an unhealthy service every time this skill runs
    trains the reader past the one line that would matter. `app/llm_slots.py` is
    the single definition; this is a consumer of it, not a second copy.

    Fails OPEN — an unreadable config reports everything. Under-reporting hides
    a service that really is down, which is the failure this skill exists to
    catch; over-reporting only costs a line.
    """
    try:
        import sys
        sys.path.insert(0, "/home/alansrobotlab/lloyd")
        from app import llm_slots
        return {program for _flag, program, on in llm_slots.slots() if not on}
    except Exception:
        return set()


# The GPU power clamp runs as root, so its unit and script are not symlinked
# from the tree like every other unit: they are deployed by a hand `install`
# to root-owned paths (the unit's own header carries the two commands). Twice
# in one week (`bb0dbca` 09-11, `36ba27e` 09-17) the tracked copy was edited
# and the deployed one re-installed by hand the same minute, and had the second
# step been forgotten nothing would have said so — the next boot would simply
# have clamped the old watts. This is the gate (#1108): each tracked file
# against the copy that actually runs, reported beside the services.
_TREE = Path(__file__).resolve().parent.parent
DEPLOY_CATEGORY = "deploy"
DEPLOYED_COPIES = (
    (_TREE / "agent-services" / "systemd" / "nvidia-power-limit.service",
     Path("/etc/systemd/system/nvidia-power-limit.service")),
    (_TREE / "agent-services" / "bin" / "set-gpu-power-limit.sh",
     Path("/usr/local/sbin/set-gpu-power-limit.sh")),
)

# The other half of "is what runs the thing in the tree?" (#2714). The two pairs above
# are the only files the installer does NOT link, so they are compared byte for byte;
# every other unit in `agent-services/systemd/` is deployed as a symlink INTO the tree,
# which is what makes `architecture/infrastructure.md`'s "editing the repo copy is the
# deploy" true. A plain copy of a unit placed by hand before the last installer run is
# byte-identical to the tree and therefore invisible to a sha256 compare — which is
# precisely how five tracked units on this box (eight deployed files: `lloyd-cert-renew.*`,
# `lloyd-egress-window.*`, `lloyd-persistence-window.*`, `thunderbird.service`,
# `voxtype.service`) sat under a `Deploy: healthy` heading while a repo edit to any of
# them landed nowhere. For those units the property is file IDENTITY, not content.
USER_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
SYSTEMD_UNIT_TREE = _TREE / "agent-services" / "systemd"
INSTALL_SERVICES = _TREE / "agent-services" / "setup" / "install-services.sh"
#: `SYSTEM_SCOPE_UNITS=(nvidia-power-limit.service)`, the declaration beside
#: `SYSTEMD_SRC`/`SYSTEMD_DST` at the head of `install-services.sh` — anchored on its own
#: text rather than a line number, because #2774 grew an arming block below it and a
#: number here would have gone stale on the very commit that added arming. Read from the
#: installer instead of restated here, for `_switched_off`'s reason: the installer
#: owns that list, and a unit added to it must not need an edit to this file to stop
#: being flagged for the gap the installer leaves on purpose.
SYSTEM_SCOPE_RE = re.compile(r"^SYSTEM_SCOPE_UNITS=\(([^)]*)\)", re.MULTILINE)
#: `HOST_LOCAL_UNITS=(...)` in the same installer: one deployed unit name per line,
#: the one-line why as that line's `#` comment (#2756). Read for the same reason as
#: `SYSTEM_SCOPE_RE` above — the list owner carries the why, and recording a
#: disposition must never need an edit to this file. The two arrays are the
#: installer's two opposite gaps: `SYSTEM_SCOPE_UNITS` is a tracked unit the
#: installer refuses to place into the user manager; `HOST_LOCAL_UNITS` is a live
#: deployed unit the repo deliberately refuses to own.
HOST_LOCAL_RE = re.compile(r"^HOST_LOCAL_UNITS=\(([^)]*)\)", re.MULTILINE)
#: The installer's own two globs, one level deep — and one level is the exclusion:
#: `agent-services/systemd/system/` holds the root-scope units installed by hand
#: (`lloyd-data-snapshot-prune.*`), which no user unit directory ever receives.
#: The same one-level rule bounds the deployed-side walk (#2756): systemd's own
#: `timers.target.wants/` symlinks and `*.d` drop-in directories live below the
#: unit files, and a directory is never a unit.
UNIT_GLOBS = ("*.service", "*.timer")

# The private CA the browser arm has to trust. `scripts/install-ca.sh --check`
# existed with a test and no production caller (#1668), so the drift it detects —
# the CA re-minted on 09-22 while the NSS nickname kept the retired key — stayed
# invisible until a human ran the guard by hand (#1726). This entry is that caller.
#
# Its own category, for the reason #1649's `Fleet:` heading was chosen: the
# per-category summary marks a category degraded when ANY member is unhealthy, and
# `lloyd:` already means "the three Mission Control processes". Filing a stale trust
# store there would let a latent browser-arm problem turn a heading whose red means
# "MC is down" red for something else.
CA_TRUST_CATEGORY = "ca"
INSTALL_CA = _TREE / "scripts" / "install-ca.sh"

# The wattage the driver is enforcing right now versus the wattage the unit
# declares (#2135). Its own category, not `deploy`, for #1726's reason verbatim:
# a heading goes degraded when ANY member is unhealthy, so filing silicon drift
# under `deploy` would make "the installed bytes are stale" and "a hand
# `nvidia-smi -pl` moved the hardware off the unit" the same heading — and the
# deployed-copy check above compares sha256 of tracked-vs-installed BYTES, so it
# structurally cannot see the second one. `check_ca_trust` above and the
# single-row unit-enabledness category in `main` are the two precedents for a
# private category with no supervisor program behind it.
GPU_POWER_CATEGORY = "gpu-power"
GPU_POWER_UNIT = "nvidia-power-limit.service"
GPU_POWER_SCRIPT = "/usr/local/sbin/set-gpu-power-limit.sh"
#: The declared side is asked through a shell guard, not a call made here, because
#: this file's string constants are barred from naming the systemd verb at all
#: (#1951's rail, quoted on `_declared_power_limits`). `UNIT_ENABLEDNESS` above is the
#: same shape for the same reason.
GPU_POWER_GUARD = _TREE / "scripts" / "maintenance" / "read-gpu-power-limit.sh"
#: `GPU_POWER_LIMIT_W_1=450` names one card; bare `GPU_POWER_LIMIT_W=275` is the
#: fallback the script applies to every card without a per-index line.
GPU_POWER_PER_INDEX = re.compile(r"\bGPU_POWER_LIMIT_W_(\d+)=(\d+)\b")
GPU_POWER_FALLBACK = re.compile(r"\bGPU_POWER_LIMIT_W=(\d+)\b")
#: The live side arrives as `0, 275.00 W`: index, a decimal wattage, and a unit
#: suffix the declared side does not carry. Parsing the number out of the string
#: is the whole comparison — a raw string compare of `275.00 W` against a declared
#: `275` reports drift on a card that is exactly where the unit put it.
GPU_POWER_LIVE_ROW = re.compile(r"^\s*(\d+)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*W?\s*$")

# ---------------------------------------------------------------------------------------------
# The protected-path substrate (#2320). `agent_mcp/_path_sandbox.py` knows whether the Bash
# tool's bind-read-only sandbox is actually being built, and `agent_mcp/main.py` serves that
# answer on `/state` as `protected_path_sandbox` — `enforcing`, `fallback`, `bwrap`, `entries`,
# `error`. Nothing read it: `git grep -n "protected_path_sandbox" -- scripts/` returned a
# fixture and this file had no HTTP client at all (`http_check` below is a TCP connect that
# never reads a body), so a box whose bwrap cannot unshare a namespace ran every Bash call
# UNSANDBOXED — the deliberately fail-open branch whose only signal is a `logger.warning` no
# monitor greps — while this script printed `Overall: 23/23 services healthy`. The absence of a
# guard was indistinguishable from a guard that works, which is the failure this row closes.
#
# It reports the state and takes no position on #2109's still-open ruling (whether a fallback
# should refuse Bash outright the way `_tool_sandbox.refusal` does, or keep failing open with a
# warning): whichever way that goes, "is it enforcing" is the question this row answers, and
# only its wording may move afterwards.
PATH_SANDBOX_CATEGORY = "path-sandbox"
#: The key `/state` carries the answer under — named to READ it, never to copy out of it. The
#: PASS row's count is `len()` of the payload's own `entries`, so the bound list, the port and
#: the token each keep exactly one definition in this repo.
PATH_SANDBOX_KEY = "protected_path_sandbox"
#: A loopback GET of a process that answers in milliseconds. The bound exists because this
#: script is what an operator runs WHEN the box is broken: a hung aggregator must not become a
#: hung health check. A timeout is an unreadable `/state`, which is a graded FAIL naming itself.
PATH_SANDBOX_TIMEOUT_S = 10.0

# install-ca.sh's three verdict words, and the two fingerprints it prints beside
# them. Parsing the WORD rather than the exit code is the whole grade: exit 1 is
# also what `ERROR: no CA certificate at ...` returns, before the store has been
# read at all, and reporting that as drift would be #1726's clause 2 in the shape of
# a green row. So `warn` requires `CHECK FAILED` in the answer.
_CA_ANSWER = re.compile(r"CHECK (OK|FAILED|INCONCLUSIVE)")
# Every line the script prints is tagged `[install-ca] `, and the two words are spaced
# by hand (`stored   sha256 ` / `expected sha256 `), so this matches on the hex rather
# than the start of the line. A `^`-anchored pattern reads the real output and finds
# nothing: the fingerprint-less row it would produce still says `warn`, which is how
# that bug reaches production without failing a test.
_CA_FINGERPRINT = re.compile(r"\b(stored|expected)\s+sha256\s+([0-9A-Fa-f:]{8,})")


def _ca_head(text: str) -> str:
    """The line that carries the verdict, with the script's tag stripped off it."""
    line = next((l for l in (text or "").splitlines() if "CHECK " in l),
                (text or "").splitlines()[0] if text else "")
    return (line.split("] ", 1)[-1] if line.startswith("[install-ca]")
            else line).strip() or "no output"


def _ca_argv() -> list:
    """The one argv this entry ever builds: the check mode of the tree's script.

    `--check` is unconditional, because the same script without it WRITES the store
    (`certutil -D` then `-A`), and a health check that costs the thing it measures is
    the failure #1141 recorded. The CA path is deliberately NOT passed: `--check`
    resolves it the way the operator's command does, from `$LLOYD_CERT_DIR` or the
    tree, so this entry compares the CA the frontend would serve.
    """
    return ["bash", str(INSTALL_CA), "--check"]


def _ca_verdict(code: int, text: str) -> str:
    """`ok` / `warn` / `unknown` from the guard's own answer, never its exit code alone.

    `unknown` is the answer for every read that did not reach the store: exit 2 (no
    `certutil`, or no `HOME` and no override — install-ca.sh:88-89), an unreadable CA
    file, an unparseable answer, or a nonzero code that is not a named verdict. An
    inconclusive read is never `ok`, and never `warn` either: `warn` is a claim about
    a mismatch that was measured.
    """
    named = _CA_ANSWER.search(text or "")
    if named is None:
        return "unknown"
    word = named.group(1)
    if word == "OK":
        return "ok" if code == 0 else "unknown"
    if word == "FAILED":
        # `CHECK FAILED` with a code other than 1 is not the drift this row reports.
        return "warn" if code == 1 else "unknown"
    return "unknown"


def _ca_fingerprints(text: str) -> dict:
    found = {}
    for kind, value in _CA_FINGERPRINT.findall(text or ""):
        found.setdefault(kind, value.strip())
    return found


def _ca_detail(text: str) -> str:
    """The fingerprints as the operator reads them, naming any that the check withheld.

    A hash that IS present is always shown. The no-such-nickname branch prints
    `stored   sha256 none (no such nickname)`, which is not a hash, and a row that
    answered that by printing neither value would hide the one number the operator
    needs in order to tell a missing entry from a replaced CA.
    """
    found = _ca_fingerprints(text)
    parts = [f"{k} sha256 {found[k]}" for k in ("stored", "expected") if found.get(k)]
    missing = [k for k in ("stored", "expected") if not found.get(k)]
    shown = " — " + ", ".join(parts) if parts else ""
    if not missing:
        return shown
    if parts:
        return f"{shown} (the check printed no {missing[0]} fingerprint)"
    return " (the check printed neither fingerprint)"


def _ca_row(verdict: str, status: str, exit_code: int, output: str,
            stored: Optional[str] = None, expected: Optional[str] = None) -> dict:
    return {
        "name": "ca-trust",
        "status": status,
        "healthy": verdict == "ok",
        # Advisory rows are reported and never counted as a fault: clause 3 of
        # #1726 is that this row cannot turn an otherwise-healthy run into a
        # failure, and #1108's `drift:` rows are not the model here because a stale
        # trust store is latent (the frontend serves a Let's Encrypt leaf while
        # `web/vite.config.ts` finds one) where a missing root-owned script is live.
        "advisory": verdict != "ok",
        "verdict": verdict,
        "exit_code": exit_code,
        "output": output,
        "category": CA_TRUST_CATEGORY,
        "stored_sha256": stored,
        "expected_sha256": expected,
    }


def check_ca_trust(runner=subprocess.run) -> list:
    """One read-only row: does the NSS store hold the CA this tree would install?

    `bash scripts/install-ca.sh --check`, against the invoking user's real store:
    `LLOYD_NSS_DB` is stripped from the child environment, because that variable is
    how this repo's own tests sandbox the script, and a row that silently reported on
    a redirected store would be a health verdict about a database nobody uses.
    Nothing else in the environment is touched, and the check branch of the script
    writes nothing (#1668 clause 3).

    `runner` is the seam the suite grades through, the same shape
    `check_deployed_copies(pairs=...)` uses: a test that cannot force `CHECK FAILED`
    on a real store otherwise has to accept whatever state this box happens to be in.
    """
    argv = _ca_argv()
    env = {k: v for k, v in os.environ.items() if k != "LLOYD_NSS_DB"}
    try:
        proc = runner(argv, capture_output=True, text=True, timeout=20, env=env)
    except Exception as exc:
        # No bash, no subprocess, a timeout: nothing was read, so nothing is graded.
        return [_ca_row("unknown", f"unknown: the CA trust check could not be run "
                                   f"({str(exc)[:120]}) — nothing was read", -1, str(exc))]
    text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    verdict = _ca_verdict(proc.returncode, text)
    found = _ca_fingerprints(text)
    head = _ca_head(text)[:200]
    if verdict == "ok":
        status = f"ok: {head}"
    elif verdict == "warn":
        status = (f"warn: the NSS trust store does not hold this tree's Lloyd CA"
                  f"{_ca_detail(text)} — run: bash scripts/install-ca.sh")
    else:
        status = (f"unknown: the CA trust check returned no verdict "
                  f"(exit {proc.returncode}): {head}{_ca_detail(text)}")
    return [_ca_row(verdict, status, proc.returncode, text,
                    found.get("stored"), found.get("expected"))]


# Every repo-tracked user timer must actually be enabled (#1891). The guard,
# `scripts/maintenance/check-unit-enabledness.sh`, landed with a test and no caller,
# so a placed-but-disabled timer was still found only by someone remembering to run
# it (#1989). This entry is that caller, on the route a human actually runs.
#
# The .sh stays the single source of the ASSERTION. Nothing here asks systemd a
# question or decides which answer counts as a pass: a row is unhealthy only where the
# script exited non-zero AND printed its own `FAIL <unit>` line for that timer, and the
# denominator in every row is the script's own `checked N timers: ...` line, copied,
# never recounted.
#
# WHAT #2792 ADDED: the route now performs the act it used to only print. The guard's
# FAIL line ends `— run: systemctl --user enable --now <unit>`, and nothing in this
# repository ever ran it, so `lloyd-egress-window.timer` has sat disabled through four
# triages that each re-printed the command (#2537, #2702, #2774 put an executed
# `enable --now` into the installer, #2787, #2792). Three things make the act safe, and
# tests/test_unit_enabledness_route.py pins each one:
#
#   * ONE argv. `arm_timer_argv` builds it from one token tuple and accepts only a bare
#     `*.timer` name that the guard's own FAIL line named. No `disable`, `stop`,
#     `restart` or `daemon-reload` is nameable anywhere in this file, and a `.service`
#     name is refused. That rail is NOT inherited from the harness:
#     `app/harness/service_control.py` does exempt `enable --now <timer>` for an
#     unattended turn, but that hook classifies the Bash string an AGENT issues, and a
#     systemctl this script spawns never crosses it.
#   * An idle pool only. `agent-services/systemd/lloyd-egress-window.timer` carries
#     `Persistent=true` with no stamp dir on this box, so `enable --now` fires the
#     window at the arming instant, and its `.service` runs `--drain-wait 3600`: a
#     health route run from inside a worker turn would otherwise arm the thing that
#     drains the pool its own check is asking about. The reading is the two-part one
#     `eval/persistence_arms.py::in_flight` takes, and an unreadable reading defers too.
#   * The re-run is the verdict, never the act. A row says `enabled` only because the
#     guard's RE-RUN put the timer in an enabled row; an `enable --now` that exits 0 on
#     a timer the re-run still fails reads red, with that FAIL line in its status.
#
# So a whole-fleet or `--category units` run mutates host systemd state — which is the
# item's whole point, and why `--no-arm` exists: the 04:00 diagnostic stays read-only
# behind one flag. Under pytest the act is never real: systemctl goes to the same
# injectable `runner` the guard goes to, and a `main()`-driven test puts a recording
# stub `systemctl` first on PATH, exactly as tests/test_unit_enabledness_check.py does
# when it drives the guard itself.
UNITS_CATEGORY = "units"
UNIT_ENABLEDNESS = _TREE / "scripts" / "maintenance" / "check-unit-enabledness.sh"
_UNITS_DENOMINATOR_RE = re.compile(r"^checked \d+ timers?: .*$", re.M)
_UNITS_ROW_RE = re.compile(r"^(\S+\.timer)\t(\S+)$", re.M)
_UNITS_FAIL_RE = re.compile(r"^FAIL (\S+\.timer) ", re.M)

#: The one systemd argv this route can build, tokenised rather than written as a
#: sentence so the argv that runs and the argv a row prints cannot drift apart.
_ARM_TOKENS = ("systemctl", "--user", "enable", "--now")
_TIMER_SUFFIX = ".timer"
_ARM_TIMEOUT_S = 30

#: Queue states meaning a job is mid-flight — the same pair
#: `eval/persistence_arms.py` calls `IN_FLIGHT_STATES`.
_IN_FLIGHT_STATES = ("claimed", "running")
_POOL_TIMEOUT_S = 5


def _unit_enabledness_argv() -> list:
    """The guard's argv. Under pytest only, `UNIT_ENABLEDNESS_SCRIPT` names a stand-in.

    The same rule as the `UNIT` seam at the top of this file, for the same reason:
    guarded on PYTEST_CURRENT_TEST so no environment variable can redirect which
    guard a live run executes. The real script shells out to the host `systemctl`,
    and a test must never read the host's unit state.
    """
    script = str(UNIT_ENABLEDNESS)
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("UNIT_ENABLEDNESS_SCRIPT"):
        script = os.environ["UNIT_ENABLEDNESS_SCRIPT"]
    return ["bash", script]


def arm_timer_argv(unit) -> list:
    """The one argv this route can build: `systemctl --user enable --now <unit>`.

    Refuses anything that is not a bare timer unit name, which is the rail that makes
    "only the arming act is reachable" a property of the code rather than of good
    intentions. Names reach here out of `_UNITS_FAIL_RE`, whose `\\S+` will capture
    `/tmp/x.timer` or `--wait.timer` just as happily out of a FAIL line printed by a
    unit somebody named badly, and an argv list is only safe while its last element is
    a name: a leading `-` is parsed by systemctl as a flag, and a `/` stops being a
    unit name at all. A `.service` name cannot reach here (the regex yields `.timer`
    only) and would be refused anyway — arming a schedule starts no unit, and starting
    a unit is exactly what the harness reserves for a human.
    """
    if (not isinstance(unit, str) or not unit.endswith(_TIMER_SUFFIX)
            or unit == _TIMER_SUFFIX or "/" in unit or unit.startswith("-")
            or any(c.isspace() or c in "\\;&|$`'\"" for c in unit)):
        raise ValueError(
            f"refusing to build an arming argv for {unit!r}: only a bare *.timer unit "
            f"name, taken from the guard's own FAIL line, is armable")
    return [*_ARM_TOKENS, unit]


def _workers_status_url() -> str:
    """`/api/workers/status` on the backend this box runs.

    Resolved per call, for the reason the fetcher in `check_path_sandbox` is resolved
    at call time: an import-time constant freezes an environment someone may have
    moved. `LLOYD_BACKEND_URL` is the same override `eval/persistence_arms.py` honours.
    """
    base = os.environ.get("LLOYD_BACKEND_URL", "http://127.0.0.1:8080").rstrip("/")
    return f"{base}/api/workers/status"


def _workers_status_payload() -> dict:
    """One GET of `/api/workers/status`, parsed. stdlib: httpx is not importable here.

    No credential, because the route carries none — it is what `scripts/mitigation_drill.py`
    and the dashboard read unauthenticated, unlike `/state`, whose 401
    `check_path_sandbox` goes out of its way to name.
    """
    with urllib.request.urlopen(_workers_status_url(), timeout=_POOL_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pool_in_flight(fetch=None) -> Optional[dict]:
    """What the worker pool says is mid-flight: `{}` idle, non-empty busy, `None` unreadable.

    Two halves, because either one alone under-counts — the same reading
    `eval/persistence_arms.py::in_flight` takes. `pool.in_flight_count` is a job running
    inside a worker right now; a queue row in `claimed` or `running` is one the pool has
    taken and may not have published an exit for yet.

    `None` is a third answer and the caller must not collapse it into `{}`: an idle pool
    is the green light to arm, an unreadable pool is not one. A backend that is down or
    a body that is not JSON reads as "nothing is running" only if someone writes the
    `or {}`, and that is the false comfort every other guard in this file refuses.

    The same rule applies to a body that arrives whole. `app/routers/workers.py:131`
    answers `{"initialized": false}` — over HTTP 200 — when it cannot even construct the
    queue, and a reading built from that body with `.get(...) or {}` is an empty one: the
    backend that knows nothing about the pool would green-light arming a Persistent timer.
    So both halves have to be PRESENT to be believed — `initialized` true, a `pool` block
    carrying the `in_flight_count` key, and a `depth` block. Anything else is `None`, and
    `None` defers. The two type checks are the least load-bearing of the four: a block
    that is not a mapping raises into the `except` below and reaches the same `None`, so
    no input distinguishes them, and they stay because the answer being asserted is "this
    body cannot be interpreted", not the mechanism that happens to produce it —
    `test_a_status_body_that_says_nothing_about_the_pool_is_never_read_as_idle` pins that
    answer for every shape, the malformed ones included.
    """
    get = fetch or _workers_status_payload
    try:
        status = get()
        if not isinstance(status, dict):
            return None
        pool, depth = status.get("pool"), status.get("depth")
        if (not status.get("initialized") or not isinstance(pool, dict)
                or "in_flight_count" not in pool or not isinstance(depth, dict)):
            return None
        out: dict = {}
        count = int(pool.get("in_flight_count") or 0)
        if count:
            out["in_flight"] = count
        for source, states in depth.items():
            for state, n in (states or {}).items():
                if state in _IN_FLIGHT_STATES and int(n or 0) > 0:
                    out[f"{source}:{state}"] = int(n)
    except Exception:  # noqa: BLE001 — an unreadable reading is the answer, not a crash
        return None
    return out


def _run_units_guard(runner):
    """One guard invocation: `(exit code, its combined output)`. Raises what the runner raises."""
    proc = runner(_unit_enabledness_argv(), capture_output=True, text=True, timeout=30)
    return proc.returncode, ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()


def _arm_timers(units, runner) -> dict:
    """Run the guard's own remediation for each timer it named. Returns unit -> note.

    The note is a sentence for the row and the exit code in it is systemctl's own.
    Nothing here decides `healthy`: arming is an attempt, and only the guard's re-run
    is a verdict. A refusal or a failed spawn is recorded and carried on with — the
    re-run says what is true either way, which is the whole reason the re-run exists.
    """
    notes = {}
    for unit in sorted(units):
        try:
            argv = arm_timer_argv(unit)
        except ValueError as exc:
            notes[unit] = str(exc)
            continue
        try:
            proc = runner(argv, capture_output=True, text=True, timeout=_ARM_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — the re-run below is the arbiter
            notes[unit] = (f"{' '.join(_ARM_TOKENS)} could not be attempted "
                           f"({type(exc).__name__}: {str(exc)[:120]})")
            continue
        notes[unit] = f"{' '.join(argv)} exited {proc.returncode}"
    return notes


def _annotate_units(rows: list, units, note) -> list:
    """Append one sentence, built per timer, to the rows for the timers it was about.

    The guard's verdict word and its denominator stay first and untouched — the note
    goes after them, so the row still leads with what systemd answered and the fault
    count does not move.
    """
    for r in rows:
        unit = (r["name"][len("timer:"):] if r["name"].startswith("timer:")
                else r["name"])
        if unit in units:
            r["status"] = f"{r['status']} — {note(unit)}"
    return rows


def _units_rows(text: str, code: int) -> list:
    """The rows the guard's text and exit code imply — the #1989 parsing, verbatim.

    Kept whole so the first reading and the re-run after an arming are graded by one
    piece of code, not by two copies that can disagree about what a FAIL line means.
    """
    denominator = _UNITS_DENOMINATOR_RE.search(text)
    counted = denominator.group(0) if denominator else "the guard printed no denominator"
    failed = set(_UNITS_FAIL_RE.findall(text)) if code != 0 else set()
    rows = []
    for unit, verdict in _UNITS_ROW_RE.findall(text):
        rows.append(_units_row(f"timer:{unit}", unit not in failed,
                               f"{verdict} — {counted}", code, text))
    if code != 0 and not any(not r["healthy"] for r in rows):
        head = next((ln for ln in text.splitlines() if ln.startswith("FAIL")),
                    text.splitlines()[0] if text else "no output")
        rows.append(_units_row("unit-enabledness", False,
                               f"failed (exit {code}): {head[:200]} — {counted}",
                               code, text))
    elif not rows:
        rows.append(_units_row("unit-enabledness", False,
                               f"unknown: the guard exited 0 and named no timer — {counted}",
                               code, text))
    return rows


def _units_row(name: str, healthy: bool, status: str, exit_code: int, output: str) -> dict:
    # No `advisory` key, on purpose: a timer that never fires is a job that never
    # runs, which is a live fault and is graded like the `deployed:` rows.
    return {
        "name": name,
        "status": status,
        "healthy": healthy,
        "exit_code": exit_code,
        "output": output,
        "category": UNITS_CATEGORY,
    }


def check_unit_enabledness(runner=subprocess.run, pool_reader=None,
                           arm: bool = True) -> list:
    """One graded row per repo timer, from the guard's own output and exit code.

    Exit 0 is the only pass. On a non-zero exit the rows the script itself named in
    a `FAIL <unit>` line are unhealthy; if it failed without naming a timer (no unit
    directory, no systemctl, zero timers) one `unit-enabledness` row carries its
    first FAIL line, because a guard that looked at nothing is never a pass. A guard
    that could not be run at all is likewise an unhealthy row, not a missing one.

    On a red run that named at least one `*.timer`, the route then performs the
    remediation that FAIL line prints (#2792) — `arm_timer_argv`'s single argv, per
    named timer, and nothing else — and GRADES THE RE-RUN, not the act. Three exits
    from before the act, each of them leaving the rows from the reading that found the
    fault and naming in the row why no arming happened: `arm` is False (an operator
    asked for a read-only run with `--no-arm`), the pool's in-flight reading could not
    be read, or the pool has work in flight. An idle pool is the only green light.

    `runner` is the guard's parameter doing double duty on purpose: one injectable
    subprocess call means a test that can watch the guard can watch systemctl too, and
    the argv set a test records is the argv set a live run would execute.
    `pool_reader` is resolved at call time rather than bound as a default, for the
    reason `check_path_sandbox` gives: a test hands this function a reading and pins the
    deferred row with no live backend anywhere in the suite.
    """
    read_pool = pool_reader or _pool_in_flight
    try:
        code, text = _run_units_guard(runner)
    except Exception as exc:
        return [_units_row("unit-enabledness", False,
                           f"unknown: the timer enabledness guard could not be run "
                           f"({str(exc)[:120]}) — nothing was checked", -1, str(exc))]
    miss = set(_UNITS_FAIL_RE.findall(text)) if code != 0 else set()
    rows = _units_rows(text, code)
    if not miss:
        # Every timer the guard checked passed, or it failed without naming one — a
        # fault no arming answers. systemd is asked nothing further and the pool is not
        # read either: a run with nothing to fix has no business mutating anything.
        return rows
    if not arm:
        return _annotate_units(rows, miss, lambda unit: (
            "not armed: this run was asked to stay read-only (--no-arm)"))
    busy = read_pool()
    if busy is None:
        return _annotate_units(rows, miss, lambda unit: (
            "not armed: the worker pool's in-flight reading could not be read, and "
            "arming a Persistent timer from a box that cannot say the pool is idle is "
            "refused"))
    if busy:
        listed = ", ".join(f"{k}={v}" for k, v in sorted(busy.items()))
        return _annotate_units(rows, miss, lambda unit: (
            f"not armed: the worker pool has work in flight ({listed}), and this unit's "
            f"window drains the pool when it fires"))

    notes = _arm_timers(miss, runner)
    try:
        recheck_code, recheck_text = _run_units_guard(runner)
    except Exception as exc:
        # Arming may or may not have worked and nothing can now say so. The rows stay
        # the red ones that found the fault; the unverified arming rides along in them.
        # `why` is bound here rather than read inside the lambda: Python 3 unbinds the
        # `as exc` name at the end of the block, so a late-bound closure over it is a
        # warning waiting for the first reader who calls the lambda after the return.
        why = f"the guard could not be re-run to check it ({str(exc)[:120]})"
        return _annotate_units(rows, miss, lambda unit: (
            f"arming: {notes.get(unit, 'not attempted')}, and {why} — still failing"))

    still_failing = (set(_UNITS_FAIL_RE.findall(recheck_text))
                     if recheck_code != 0 else set())
    fail_lines = {}
    verdicts = dict(_UNITS_ROW_RE.findall(recheck_text))
    for line in recheck_text.splitlines():
        hit = _UNITS_FAIL_RE.match(line)
        if hit:
            fail_lines[hit.group(1)] = line

    def note(unit: str) -> str:
        attempted = notes.get(unit, "not attempted")
        if unit in still_failing:
            return (f"arming {attempted}, and the re-run still fails: "
                    f"{fail_lines.get(unit, 'the re-run named no FAIL line for it')[:300]}")
        return (f"arming {attempted}, and the re-run says "
                f"{verdicts.get(unit, 'nothing')}")

    return _annotate_units(_units_rows(recheck_text, recheck_code), miss, note)


CATEGORIES = {
    "llm": ["agent-llm-primary", "agent-llm-secondary", "agent-djev"],
    "lloyd": ["lloyd-backend", "lloyd-frontend", "lloyd-mcp"],
    "retrieval": ["agent-qmd-daemon"],
    # No supervisor program behind it: `main` answers this category from
    # `check_deployed_copies` instead.
    DEPLOY_CATEGORY: [],
    # Same shape: `main` answers it from `check_ca_trust`, which is neither a
    # supervisor program nor a deployed file.
    CA_TRUST_CATEGORY: [],
    # Same shape again: `main` answers it from `check_unit_enabledness`.
    UNITS_CATEGORY: [],
    # And again: `main` answers it from `check_gpu_power_limit`, which is neither a
    # supervisor program, a deployed file, nor a certificate — it is the wattage the
    # driver is enforcing versus what the unit declares (#2135). It is its own
    # heading rather than a member of `deploy` for the #1726 reason recorded on
    # GPU_POWER_CATEGORY.
    GPU_POWER_CATEGORY: [],
    # And again: `main` answers it from `check_path_sandbox`, which reads the aggregator's
    # /state rather than a supervisor program, a deployed file, a certificate or a wattage.
    PATH_SANDBOX_CATEGORY: [],
    "all": list(SERVICES.keys()),
}


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def check_deployed_copies(pairs=DEPLOYED_COPIES) -> list:
    """Compare each tracked file against its root-owned deployed copy.

    One result per pair, in the shape `check_service` returns, so the two
    formatters and the category summary carry it without a special case. The
    status is `ok` only when both files read and their bytes match, `drift`
    when they differ, and `missing` when either cannot be read — each naming
    the pair. An unreadable deployed copy is never `ok`: a check that read
    nothing verified nothing, and a green line from it would be exactly the
    false comfort the check exists to remove.
    """
    results = []
    for tracked, deployed in pairs:
        tracked, deployed = Path(tracked), Path(deployed)
        want, have = _sha256(tracked), _sha256(deployed)
        if want is None:
            status = f"missing: tracked copy {tracked} is unreadable"
        elif have is None:
            status = (f"missing: {deployed} is unreadable or not installed "
                      f"(tracked copy: {tracked})")
        elif want != have:
            status = (f"drift: {deployed} differs from tracked {tracked} "
                      f"— reinstall it from the tree")
        else:
            status = f"ok: {deployed} matches {tracked}"
        healthy = status.startswith("ok:")
        results.append({
            "name": f"deployed:{deployed.name}",
            "status": status,
            "healthy": healthy,
            "exit_code": 0 if healthy else 1,
            "output": status,
            "category": DEPLOY_CATEGORY,
            "tracked": str(tracked),
            "deployed": str(deployed),
        })
    return results


def _system_scope_units(installer: Path = INSTALL_SERVICES) -> Optional[set]:
    """The unit names `install-services.sh` declines to link, or `None` if unreadable.

    `None` is deliberately not an empty set. A checker that could not read the exclusion
    list and graded the whole directory anyway would file `missing:` against
    `nvidia-power-limit.service` — the one unit whose absence from the user manager is
    the intended state, graded on its real path by `check_deployed_copies` — and clause
    4 of #2714 exists to make that unreportable. The caller answers with one loud row.
    """
    try:
        text = Path(installer).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    line = SYSTEM_SCOPE_RE.search(text)
    if line is None:
        return None
    return {name.strip("\"'") for name in line.group(1).split() if name.strip("\"'")}


def _host_local_dispositions(installer: Path = INSTALL_SERVICES) -> dict:
    """The deployed units the repo deliberately does not own, as `{name: one-line why}`.

    The read shape is `SYSTEM_SCOPE_RE`'s own — regex-extract a declared array from
    the installer rather than restate it here — because the property that makes that
    work is that there is no second copy to forget: a disposition added to
    `install-services.sh` takes effect with no edit to this file (#2756). An
    unreadable installer or a missing array is the empty dict, not `None`: an empty
    disposition set grades every orphan `untracked:`, which is the right answer when
    nobody has recorded any disposition — and unlike the exclusion list it cannot
    misfile a deliberate gap, because the unit it declines to excuse is the unit the
    row then names. `check_unit_file_links` has already refused to grade anything at
    all when the installer could not be read, so this reader never runs blind.
    """
    try:
        text = Path(installer).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    block = HOST_LOCAL_RE.search(text)
    if block is None:
        return {}
    out = {}
    for line in block.group(1).splitlines():
        entry, _, why = line.partition("#")
        name = entry.strip().strip("\"'")
        if name:
            out[name] = why.strip()
    return out


def _link_target(link: Path) -> Path:
    """Where a symlink points, resolved the way `realpath` resolves it.

    `readlink` rather than `resolve()` on the link itself because the answer the row has
    to print is the target the link NAMES: resolving the link would follow an arbitrary
    chain and report the far end of it. A relative target is relative to the link's own
    directory, which is why it is joined to `link.parent` before being resolved.
    """
    raw = Path(os.readlink(link))
    return (raw if raw.is_absolute() else link.parent / raw).resolve()


def _unit_link_row(name: str, healthy: bool, status: str, tracked, deployed,
                   target: Optional[str] = None, prefix: str = "unitlink:") -> dict:
    """One graded row in `check_service`'s shape, so both formatters and the category
    summary carry it with no special case — #1108's reasoning, reused verbatim because
    these rows share the `deploy` heading those two do.

    `prefix` names the row after the unit file it graded. The one row that grades no
    unit file — the exclusion list could not be read — passes an empty one, so nobody
    can mistake "nothing was checked" for a verdict about a file.
    """
    row = {"name": f"{prefix}{name}",
           "status": status,
           "healthy": healthy,
           "exit_code": 0 if healthy else 1,
           "output": status,
           "category": DEPLOY_CATEGORY,
           "tracked": str(tracked),
           "deployed": str(deployed)}
    if target is not None:
        row["target"] = target
    return row


def check_unit_file_links(tree_dir: Path = SYSTEMD_UNIT_TREE,
                          deployed_dir: Path = USER_UNIT_DIR,
                          installer: Path = INSTALL_SERVICES) -> list:
    """One graded row per tracked user-scope unit file: is the copy that runs the tree's
    own file, or a stand-in that will never follow it?

    The installer's link loop (the `ln -sf` in `install-services.sh`, over
    `"$SYSTEMD_SRC"/*.service` and `*.timer`) links every unit in `tree_dir` into
    `deployed_dir`, so after it has run the deployed path is
    a symlink to the tracked file and editing the repo copy IS the deploy. Anything else
    at that path means the running unit is a different object from the tracked one:

    * a symlink resolving to the tracked file — `ok`. Resolution decides, not how the
      link was written, so a hand-made relative link passes.
    * a symlink resolving elsewhere — `wrongtarget`, naming the target. Same fault with a
      dangling target, which must not be read as healthy and must not raise.
    * a regular file — `copy`. The bytes usually match the tree (all eight on this box do
      today), which is exactly why this is not a byte question: the reader cannot fix it
      by reinstalling content, only by re-establishing the link.
    * nothing at all — `missing`. The installer never linked it, so the unit is not
      installed and never ran.

    Both globs are evaluated when the call is made, so a unit added to either
    directory is graded without anyone editing this file, and the walk runs both
    directions. Repo to deploy grades each tracked unit as above. Deploy to repo
    (#2756) grades every `*.service`/`*.timer` living in `deployed_dir` with no
    same-named tracked file as `orphanunit:<name>`: an `untracked:` fault — nothing
    in the repo owns the unit that runs — unless the name carries a disposition in
    the installer's `HOST_LOCAL_UNITS`, which reads ok and quotes the one-line why.
    The reverse walk reads unit files, not directories, and stays one level deep:
    systemd's armed-timer symlinks under `timers.target.wants/` and any `*.d`
    drop-in directories produce no row and no duplicate, and a tracked unit's own
    deployed symlink is graded once, by the forward walk. A `SYSTEM_SCOPE_UNITS`
    name always collides with its tracked tree file, so the reverse walk never
    reports one; anything under `system/` is excluded by the globs being one level
    deep, which is the same reason the installer never sees it.
    """
    skip = _system_scope_units(installer)
    if skip is None:
        # A check that cannot tell a deliberate gap from a stale copy has no business
        # reporting either way, and `check_unit_enabledness`'s "looked at nothing" row is
        # the precedent for saying so rather than grading everything or nothing.
        return [_unit_link_row(
            "unit-file-identity", False,
            f"unknown: the units the installer declines to link could not be read from "
            f"{installer} — no unit file was graded, so this row is the finding",
            installer, deployed_dir, prefix="")]

    tree_dir, deployed_dir = Path(tree_dir), Path(deployed_dir)
    # Every name the tree owns, exclusions included: a SYSTEM-scope name collides
    # with its tracked file here, which is what keeps the reverse walk from ever
    # filing an orphan row for a unit `check_deployed_copies` grades on its real
    # path. Evaluated at call time, like the forward glob.
    tracked_names = {p.name for pattern in UNIT_GLOBS for p in tree_dir.glob(pattern)}
    rows = []
    for pattern in UNIT_GLOBS:
        for tracked in sorted(tree_dir.glob(pattern)):
            if tracked.name in skip:
                continue
            want = tracked.resolve()
            deployed = deployed_dir / tracked.name
            if deployed.is_symlink():
                target = _link_target(deployed)
                if target == want:
                    rows.append(_unit_link_row(
                        tracked.name, True,
                        f"ok: {deployed} is a link to the tracked unit {tracked}",
                        tracked, deployed))
                else:
                    rows.append(_unit_link_row(
                        tracked.name, False,
                        f"wrongtarget: {deployed} is a link to {target}, not to the tree "
                        f"— the unit that runs is not {tracked}; re-link it with "
                        f"agent-services/setup/install-services.sh",
                        tracked, deployed, str(target)))
                continue
            if deployed.exists():
                what = ("a plain copy" if deployed.is_file()
                        else "an entry that is neither a link nor a regular file")
                rows.append(_unit_link_row(
                    tracked.name, False,
                    f"copy: {deployed} is {what}, not a link to the tree — editing "
                    f"{tracked} changes nothing that runs; re-link it with "
                    f"agent-services/setup/install-services.sh",
                    tracked, deployed))
            else:
                rows.append(_unit_link_row(
                    tracked.name, False,
                    f"missing: {deployed} is not installed, so the installer has never "
                    f"linked the tracked unit {tracked} — run "
                    f"agent-services/setup/install-services.sh",
                    tracked, deployed))

    dispositions = _host_local_dispositions(installer)
    for pattern in UNIT_GLOBS:
        for deployed in sorted(deployed_dir.glob(pattern)):
            if deployed.is_dir() or deployed.name in tracked_names:
                continue
            want = tree_dir / deployed.name
            why = dispositions.get(deployed.name)
            if why is None:
                rows.append(_unit_link_row(
                    deployed.name, False,
                    f"untracked: {deployed} is live in {deployed_dir} and tracked "
                    f"nowhere in the repo — no {want} exists, so nothing in git owns "
                    f"the unit that runs; fix it by tracking the unit under "
                    f"agent-services/systemd/ or recording a disposition with a "
                    f"one-line why in {installer}'s HOST_LOCAL_UNITS",
                    want, deployed, prefix="orphanunit:"))
            else:
                rows.append(_unit_link_row(
                    deployed.name, True,
                    f"ok: {deployed} is tracked nowhere in the repo by a disposition "
                    f"in {installer}" + (f": {why}" if why else
                                         " carrying no why on its line"),
                    want, deployed, prefix="orphanunit:"))
    return rows


def _declared_power_limits(runner=subprocess.run):
    """What the unit DECLARES, as `(per_index, default)` — watts keyed by GPU index.

    `GPU_POWER_GUARD` runs `systemctl show nvidia-power-limit.service
    --property=Environment`, which is the declared side: what the next boot and the
    next unit restart will enforce, the half nobody can see from the hardware. It goes
    through a shell guard rather than a direct call here for #1951's reason, which
    `tests/test_unit_enabledness_route.py::test_no_enabledness_assertion_is_reimplemented_in_python`
    enforces over this file's string constants: a systemd question answered in Python
    is a second implementation with no guard in front of it. `None` means the question
    cannot be answered — the unit is not installed, systemd could not be reached, or
    it declares no power-limit variable at all — and the caller reads that as unknown,
    never as zero or as drift. A missing `_0` beside a present bare
    `GPU_POWER_LIMIT_W` is NOT unknown: the bare name is that card's declared value.
    """
    try:
        out = runner(["bash", str(GPU_POWER_GUARD)],
                     capture_output=True, text=True, timeout=8)
    except Exception:
        return None
    if out.returncode != 0 or not (out.stdout or "").strip():
        return None
    text = out.stdout
    per_index = {int(i): float(w) for i, w in GPU_POWER_PER_INDEX.findall(text)}
    fallback = GPU_POWER_FALLBACK.search(text)
    return per_index, (float(fallback.group(1)) if fallback else None)


def _live_power_limits(runner=subprocess.run):
    """What the driver is enforcing NOW, as `{index: watts}`.

    `enforced.power.limit` is deliberately the field that reflects a hand
    `nvidia-smi -pl 400`: it is the limit in force, not the card's ceiling
    (`power.limit`, which `app/host_metrics.py` publishes for the layout
    fingerprint, is a different field and does not move the way this one does).
    Raises if `nvidia-smi` is absent or answers non-zero — the caller turns that
    into an unknown row; returns `{}` for a machine that answers with no GPU rows,
    which is a real answer and not a failure.
    """
    out = runner(["nvidia-smi", "--query-gpu=index,enforced.power.limit",
                  "--format=csv,noheader"],
                 capture_output=True, text=True, timeout=15)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout or "nvidia-smi failed").strip()[:200])
    limits = {}
    for line in (out.stdout or "").splitlines():
        row = GPU_POWER_LIVE_ROW.match(line)
        if row:
            limits[int(row.group(1))] = float(row.group(2))
    return limits


def check_gpu_power_limit(declared=None, live=None):
    """Live watts versus declared watts, per GPU index (#2135).

    Resolved at CALL time, not at import, so a test that monkeypatches
    `_declared_power_limits` on the module moves the check too — which is how
    `main()` is driven below without a GPU. The defaults are the two real readers.

    The failure this exists for happened twice and was invisible both times: a hand
    `nvidia-smi -pl` moves the hardware off the unit, and nothing in the loop is
    comparing the two — `check_deployed_copies` above hashes the unit's BYTES, and
    a byte-identical unit with a re-tuned card passes it. The unit edit and the root
    reinstall landed in `36ba27ec`; this is the probe #1107 owed, the one that makes
    the next hand edit show up in a health check instead of at the next boot.

    A comparison happens only where both sides are known. `declared`/`live` are
    injectable so the readers are the seam the tests stub — no root, no GPU, no
    `nvidia-smi` needed for the graded path. Three rules the shape of the output has
    to keep:

      * An index whose declared number it cannot find is not compared. Silence
        would be a missed check, so it is its own unknown row.
      * A card that answers `275.00 W` against a declared `275` is NOT drift: the
        suffix and the decimals are formatting, so watts are compared as numbers.
      * Nothing here can make a machine without a GPU go red. `advisory=True`
        follows `check_ca_trust`'s precedent, which keeps an unknown out of both the
        category's healthy/unhealthy arithmetic and `main`'s exit code, so `no GPU`
        reads as the word and not as an outage.
    """
    declared = declared or _declared_power_limits
    live = live or _live_power_limits

    results = []
    try:
        dec = declared()
    except Exception:  # noqa: BLE001 - a reader blowing up is unknown, never drift
        dec = None
    if dec is None:
        results.append({
            "name": f"{GPU_POWER_CATEGORY}:declared",
            "status": "declared limit unknown (unit not installed?)",
            "healthy": False, "exit_code": 0,
            "output": "read it with: bash %s  (which asks systemd for %s's "
                      "Environment= lines)" % (GPU_POWER_GUARD.name, GPU_POWER_UNIT),
            "category": GPU_POWER_CATEGORY, "verdict": "unknown", "advisory": True,
        })
        return results
    per_index, default = dec
    try:
        limits = live()
    except FileNotFoundError:
        limits = None
        why = "nvidia-smi not on PATH"
    except Exception as exc:  # noqa: BLE001 - absent, timed out, or refused: unknown
        limits = None
        why = str(exc)[:120] or "nvidia-smi failed"
    if limits is None:
        results.append({
            "name": f"{GPU_POWER_CATEGORY}:live",
            "status": "live limit unavailable",
            "healthy": False, "exit_code": 0,
            "output": why, "category": GPU_POWER_CATEGORY,
            "verdict": "unknown", "advisory": True,
        })
        return results
    if not limits:
        results.append({
            "name": f"{GPU_POWER_CATEGORY}:live",
            "status": "no GPU enumerated",
            "healthy": False, "exit_code": 0,
            "output": "nvidia-smi listed no GPU; nothing to compare",
            "category": GPU_POWER_CATEGORY, "verdict": "unknown", "advisory": True,
        })
        return results
    for index in sorted(limits):
        watts = limits[index]
        if index in per_index:
            want, source = per_index[index], "per-index"
        elif default is not None:
            want, source = default, "default"
        else:
            results.append({
                "name": f"{GPU_POWER_CATEGORY}:{index}",
                "status": f"GPU {index}: live {watts:g} W, no declared limit to "
                          f"compare against",
                "healthy": False, "exit_code": 0,
                "output": f"declared side carries neither "
                          f"GPU_POWER_LIMIT_W_{index} nor a GPU_POWER_LIMIT_W "
                          f"default",
                "category": GPU_POWER_CATEGORY, "verdict": "unknown",
                "advisory": True, "index": index, "live_w": watts,
            })
            continue
        ok = watts == want
        results.append({
            "name": f"{GPU_POWER_CATEGORY}:{index}",
            "status": (f"GPU {index}: {watts:g} W as declared" if ok else
                       f"GPU {index}: live {watts:g} W, declared {want:g} W"),
            "healthy": ok, "exit_code": 0 if ok else 1,
            "output": (f"live {watts:g} W == declared {want:g} W ({source}, "
                       f"{GPU_POWER_UNIT})" if ok else
                       f"live {watts:g} W != declared {want:g} W, which came from "
                       f"{source} on {GPU_POWER_UNIT}; fix with "
                       f"sudo {GPU_POWER_SCRIPT}"),
            "category": GPU_POWER_CATEGORY,
            "verdict": "ok" if ok else "drift",
            "index": index, "live_w": watts, "declared_w": want,
            "declared_source": source,
        })
    return results




def _aggregator_config():
    """`app.aggregator_config`, imported from the tree this script lives in.

    Lazy, exactly like `_switched_off`'s import of `app.llm_slots`: this file's own imports stay
    stdlib so it still runs under `/usr/bin/python3`, which is what `python3
    scripts/service_health_check.py` means on a machine where the venv is the broken part.
    """
    import sys
    tree = str(_TREE)
    if tree not in sys.path:
        sys.path.insert(0, tree)
    from app import aggregator_config
    return aggregator_config


def _state_url_and_headers() -> tuple:
    """(`route("state")`, `auth_headers_for(that url)`) — the pairing, from one module.

    `/state` is credential-bearing (#1053): an uncredentialed GET answers 401 with a body naming
    the `X-Lloyd-Aggregator-Token` header. Both halves come from `app.aggregator_config` rather
    than from constants here, because a check that hard-codes a port or a header name is a second
    definition that silently diverges from the server's own route table.
    """
    cfg = _aggregator_config()
    url = cfg.route("state")
    return url, cfg.auth_headers_for(url)


def _state_payload() -> dict:
    """One authenticated GET of `/state`, parsed. stdlib, because httpx is not available here.

    `python3 -c "import httpx"` is a ModuleNotFoundError under both `/usr/bin/python3` and the
    bare `python3` on PATH — the interpreter the fleet watchdog actually uses — so a row built on
    a third-party client would raise on every run and grade a false FAIL about the substrate from
    inside a missing dependency.
    """
    url, headers = _state_url_and_headers()
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=PATH_SANDBOX_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _path_sandbox_row(healthy: bool, status: str, exit_code: int = 0,
                      enforcing=None, fallback=None, entry_count=None,
                      error=None, output: str = "") -> dict:
    return {
        "name": "path-sandbox",
        "status": status,
        # No `advisory` key, on purpose — following `_units_row` rather than `_ca_row`. An
        # advisory row is dropped by `_scored_rows` before `format_text` and `format_json` count
        # anything, so a fallback reported as advisory would print under a separate `Advisory:`
        # heading and leave `Overall: 23/23` green: the exact blindness this item was filed
        # against. A substrate that is not enforcing is a live fault, not latent drift — every
        # Bash call on the box is running against protected paths unconfined.
        "healthy": healthy,
        "exit_code": exit_code,
        "output": output,
        "category": PATH_SANDBOX_CATEGORY,
        "enforcing": enforcing,
        "fallback": fallback,
        "entry_count": entry_count,
        "error": error,
    }


def check_path_sandbox(fetcher=None) -> list:
    """One graded row: is the protected-path Bash sandbox actually being enforced?

    The verdict is `/state`'s own `protected_path_sandbox` key, and nothing else — no second
    constant, no re-derived entry list, no local copy of what the serving process binds. The
    fetcher is a parameter for the reason `check_ca_trust(runner=...)` and
    `check_gpu_power_limit(declared=, live=)` are parameters, and resolved at CALL time rather
    than as a default argument: a test hands this function a payload and pins the red row with no
    live aggregator anywhere in the suite.

    Three answers, all of them graded:

      * `enforcing: true` — PASS, naming the count of the payload's own `entries` list. An
        enforcing answer with no entry list is a FAIL instead: a check whose denominator went
        missing is not a check (the standing rule #1726 states for a zero denominator).
      * `enforcing: false` — FAIL naming `.error`, the probe's own reason.
      * no usable answer at all — FAIL naming why. Connection refused, the 401 an uncredentialed
        request gets, a body that is not JSON, and a body without the key all land here. A row
        that went quietly is what this item exists to remove: silence read as health before, and
        would again.

    `exit_code` is 0 when the aggregator answered, -1 when the read itself could not be made —
    the same -1 `_units_row` carries for a guard that could not be run.
    """
    fetch = fetcher or _state_payload
    try:
        body = fetch()
    except Exception as exc:  # noqa: BLE001 — any failure to read is a FAIL naming its reason
        return [_path_sandbox_row(
            False,
            f"unknown: {PATH_SANDBOX_KEY} could not be read from /state "
            f"({type(exc).__name__}: {str(exc)[:180]}) — nothing was checked, so the "
            f"protected-path substrate is unverified",
            exit_code=-1)]

    sandbox = body.get(PATH_SANDBOX_KEY) if isinstance(body, dict) else None
    if not isinstance(sandbox, dict):
        return [_path_sandbox_row(
            False,
            f"unknown: the /state body carried no {PATH_SANDBOX_KEY} answer — nothing was "
            f"checked, so the protected-path substrate is unverified")]

    entries = sandbox.get("entries")
    count = len(entries) if isinstance(entries, list) else None
    error = sandbox.get("error")
    bwrap = "present" if sandbox.get("bwrap") else "absent"
    fallback = bool(sandbox.get("fallback"))
    shown = json.dumps(sandbox, sort_keys=True)[:400]

    if sandbox.get("enforcing") is True:
        if count is None:
            return [_path_sandbox_row(
                False,
                f"unknown: /state reports the substrate enforcing but the payload has no "
                f"entries list, so there is no denominator to print (bwrap {bwrap})",
                enforcing=True, fallback=fallback, error=None, output=shown)]
        return [_path_sandbox_row(
            True,
            f"enforcing — {count} protected path(s) bound read-only by bwrap "
            f"(bwrap {bwrap}, fallback={fallback})",
            enforcing=True, fallback=fallback, entry_count=count, output=shown)]

    reason = str(error) if error else "the payload named no error"
    denom = (f"{count} path(s) listed by the payload, none of them enforced"
             if count is not None else "the payload listed no entries")
    return [_path_sandbox_row(
        False,
        f"NOT enforcing — {reason} (bwrap {bwrap}, fallback={fallback}, {denom})",
        enforcing=bool(sandbox.get("enforcing")), fallback=fallback,
        entry_count=count, error=str(error) if error else None, output=shown)]


def _probe_targets(host: str, port: int) -> list:
    """Addresses worth trying for `host:port`, in the order curl would try them.

    `getaddrinfo` on this box returns `::1` *first* for "localhost", which is why
    `curl localhost:8181/health` succeeds against qmd while an IPv4-literal
    probe of the same port is refused. Loopback literals are appended for any
    family resolution did not yield, so a resolver hiccup cannot by itself
    produce an "unhealthy" verdict about a service.
    """
    targets = []
    try:
        for info in socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM):
            targets.append((info[0], info[4]))
    except Exception:
        pass
    resolved = {t[1][0] for t in targets}
    for family, literal in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        if literal not in resolved:
            addr = (literal, port, 0, 0) if family == socket.AF_INET6 else (literal, port)
            targets.append((family, addr))
    return targets


def http_check(port: int, host: str = "localhost") -> tuple:
    """Quick TCP connect check against a port. Returns (connected, error).

    Tries every address `host` resolves to rather than assuming the port is on
    IPv4 loopback. qmd is the reason: it binds `[::1]:8181` and nothing else, so
    a `127.0.0.1` probe answers ECONNREFUSED (111) from a daemon that is up and
    serving `/health` — a healthy service reported dead, which is a worse
    failure than not probing at all.
    """
    last_err = "no addresses to try"
    for family, addr in _probe_targets(host, port):
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
            s.settimeout(2)
            result = s.connect_ex(addr)
            s.close()
        except Exception as e:
            last_err = str(e)
            continue
        if result == 0:
            return (True, 0)
        last_err = result
    return (False, last_err)


# supervisor prints its status line as `name  STATENAME  description`, so the
# state is field 2 and only field 2 may be read as health. Reading anything
# else mis-verdicts, for two reasons measured on this box with the installed
# supervisor 4.3.0:
#
#   * `supervisorctl status` exits 0 for BACKOFF. `do_status` moves the exit
#     status off SUCCESS only for `states.STOPPED_STATES`
#     (supervisorctl.py:696-698), and states.py:14-22 files BACKOFF under
#     RUNNING_STATES — so a process that is spawned, dies inside `startsecs`
#     and is being retried used to satisfy the old
#     `healthy = result.returncode == 0` fallback and printed `[✓] healthy`.
#   * field 3 is free text, so `agent-tts  FATAL  can't spawn process: RUNNING
#     helper not found` satisfied the old `"RUNNING" in output` test and was
#     reported as `healthy=True, status="RUNNING"`.
#
# The exit code is deliberately not consulted: acting on it is #1029's
# contract, not this script's.
def _supervisor_verdict(output: str) -> tuple:
    """Decide health from supervisorctl's own state field.

    Returns `(status, healthy)`. Healthy only when every printed status line
    names exactly `RUNNING`: a group namespec (or a bare `supervisorctl
    status`) prints one line per process, so one BACKOFF process inside an
    otherwise RUNNING answer is not a healthy service. A line with no state
    field, and an empty answer, are unparsable and therefore unhealthy — the
    raw output is returned as `status` so the printed verdict still carries
    what supervisor actually said.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return ("unknown (empty supervisorctl status output)", False)
    for line in lines:
        fields = line.split()
        if len(fields) < 2 or fields[1] != "RUNNING":
            return (output, False)
    return ("RUNNING", True)


def check_service(name: str, service_def: dict) -> dict:
    """Check a single service status."""
    command = service_def["command"]
    category = service_def["category"]
    port = service_def.get("port")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout.strip() or result.stderr.strip()
        status = output or "unknown"
        healthy = False

        # Special handling for supervisorctl: the state field decides, not the
        # exit code and not a substring of the answer.
        if is_supervisor_call(command):
            status, healthy = _supervisor_verdict(output)

        # If supervisor says running, also check port is reachable
        extra = ""
        if healthy and port:
            connected, err = http_check(port)
            if connected:
                extra = f" (port {port} OK)"
            else:
                extra = f" (port {port} FAIL: {err})"
                healthy = False
                status += extra

        return {
            "name": name,
            "status": status,
            "healthy": healthy,
            "exit_code": result.returncode,
            "output": output,
            "category": category
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "status": "timeout",
            "healthy": False,
            "exit_code": -1,
            "output": "timeout",
            "category": category
        }
    except Exception as e:
        row = {"name": name, "status": "error", "healthy": False, "exit_code": -1,
               "output": str(e), "category": category}
        if isinstance(e, OSError):
            # The process never started, so this service was never measured: `error`
            # here was a claim the run had no right to make. A `timeout` above is NOT
            # this case — the binary was found, ran, and did not answer, which says
            # something about the service and stays counted (#2652).
            row["status"] = UNVERIFIED_STATUS
            row["unverified"] = True
        return row


def _advisory_rows(results: list) -> list:
    """Rows that report something without claiming a fault (#1726 clause 3).

    A row is advisory by its own declaration, never by its category name, so a future
    round cannot make a real fault invisible by filing it under `ca`.
    """
    return [r for r in results if r.get("advisory")]


def _unverified_rows(results: list) -> list:
    """The unverified rows no other counter has already claimed (#2693).

    `advisory` wins, and it has to: `_advisory_rows` selects on the `advisory` flag
    alone while `_scored_rows` excludes on either flag, so a row carrying both flags
    is already inside the `advisory` count. A counter spelled plainly as
    `sum(1 for r in results if r.get("unverified"))` would bill that row twice and
    let the four json counters sum ABOVE `total_services`.
    """
    return [r for r in results if r.get("unverified") and not r.get("advisory")]


def _scored_rows(results: list) -> list:
    """The rows the verdict is arithmetic over.

    Two kinds are left out, and for the same reason: neither claims a fault. An advisory
    row reports a condition the run declines to grade (#1726); an `unverified` row is one
    that never reached what it was checking, so counting it turned the checker's own miss
    into a fleet fault — six RUNNING services printed `Overall: 15/23` because of a
    missing PATH entry (#2652). Excluding it from BOTH sides is the point: leaving it in
    the denominator alone would hold the same sentence against a service nobody asked
    about. The rows still print, with their reason, above the categories.
    """
    return [r for r in results
            if not r.get("advisory") and not r.get("unverified")]


def format_text(results: list, summary: dict) -> str:
    """Format results as human-readable text."""
    lines = []
    scored = _scored_rows(results)
    advisory = _advisory_rows(results)
    healthy_count = sum(1 for r in scored if r["healthy"])
    total = len(scored)

    lines.append("=== Service Health Check ===")
    lines.append(f"Time: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Overall: {healthy_count}/{total} services healthy")
    if advisory:
        # Its own line, so `Overall:` keeps meaning "faults over what was graded"
        # and a stale trust store still cannot be missed under a separate heading.
        lines.append(f"Advisory: {len(advisory)} row(s) reported without a fault "
                     f"verdict: {', '.join(r['name'] for r in advisory)}")
    unverified = [r for r in results if r.get("unverified")]
    if unverified:
        # Named on its own line for the same reason advisory is: dropping these from
        # `Overall:` must not also make them invisible, or a run that read nothing
        # would look like a run that read nothing wrong (#2652).
        lines.append(f"Unverified: {len(unverified)} row(s) never reached what they "
                     f"check, graded as neither healthy nor down: "
                     f"{', '.join(r['name'] for r in unverified)}")
    lines.append("")

    for result in results:
        icon = ("[?]" if result.get("unverified")
                else "[!]" if result.get("advisory")
                else "[✓]" if result["healthy"] else "[✗]")
        # The reason a failed read computed goes on the row with it (#2652). It was
        # already there — `check_service`'s except arm put `str(e)` in `output` — and
        # `format_text` rendered only `status`, so the printed line said `error` about a
        # failure whose cause had been worked out one field earlier and thrown away.
        # `--json` always dumped the whole row, which is how this survived review.
        status = str(result["status"])
        reason = str(result.get("output") or "")
        if (result.get("unverified") or status in ("error", "timeout")) and reason \
                and reason not in status:
            status = f"{status} ({reason})"
        lines.append(f"{icon} {result['name']}\u2014 {status}")

    lines.append("")
    lines.append("Categories:")
    for cat, status in summary.items():
        lines.append(f"  {cat.capitalize()}: {status}")

    return "\n".join(lines)


def format_json(results: list, summary: dict) -> str:
    """Format results as JSON.

    `unhealthy` is faults only: an advisory row goes into its own count, so a caller
    that reads just `unhealthy` is not told the box is broken by a retired
    certificate, and a row whose command never started goes into `unverified` for the
    same reason — it is the checker's own miss, not a service being down (#2652). The
    four counters partition the rows, so
    `healthy + unhealthy + advisory + unverified == total_services`, where
    `unverified` counts the unverified rows that are not also advisory (a row
    carrying both flags is counted once, under `advisory`) and `total_services` is
    every entry in `services`. The row itself, with its `verdict` and its two
    fingerprints, is always in `services`.
    """
    scored = _scored_rows(results)
    advisory = _advisory_rows(results)
    unverified = _unverified_rows(results)
    healthy_count = sum(1 for r in scored if r["healthy"])

    output = {
        "check_time": datetime.now(timezone.utc).isoformat(),
        "total_services": len(results),
        "healthy": healthy_count,
        "unhealthy": len(scored) - healthy_count,
        "advisory": len(advisory),
        "unverified": len(unverified),
        "services": results,
        "summary": summary
    }
    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Check status of multiple services")
    parser.add_argument("--services", nargs="+", help="Specific services to check")
    parser.add_argument("--category", choices=list(CATEGORIES.keys()), help="Check by category")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--no-arm", action="store_true",
                        help="units category only: report a not-enabled timer without "
                             "running the `enable --now` its FAIL line prints. Default is "
                             "to arm it, but only when the worker pool reports nothing "
                             "in flight (#2792).")

    args = parser.parse_args()

    # Determine which services to check
    if args.services:
        # An explicit ask is answered whatever the config says: someone naming a
        # slot by hand wants to know about that slot, including that it is off.
        service_names = args.services
    else:
        service_names = CATEGORIES[args.category] if args.category else CATEGORIES["all"]
        off = _switched_off()
        service_names = [n for n in service_names if n not in off]

    # Run checks
    results = []
    # The coverage pass (#1649): every loaded program `SERVICES` does not refer to.
    # Only for a whole-fleet run — `--services foo` asks about foo, `--category llm`
    # asks about the LLMs, and neither wants twelve unsolicited rows. The declared
    # probes go second so the familiar rows print first.
    if not args.services and args.category in (None, "all"):
        results.extend(derived_supervisor_rows())
    for name in service_names:
        if name in SERVICES:
            results.append(check_service(name, SERVICES[name]))
    # The deployed-copy pairs ride with the full check and with their own
    # category; a `--services` ask names supervisor programs and gets only those.
    # The unit-file identity rows ride in the same block because they grade the same
    # question — is the deployed thing the tracked thing? — and the units the pairs above
    # deliberately exclude are exactly the ones it symlinks (#2714). They share the
    # `deploy` heading, so one heading is what a reader gets for both halves.
    if not args.services and args.category in (None, DEPLOY_CATEGORY):
        results.extend(check_deployed_copies())
        results.extend(check_unit_file_links())
    # The CA trust guard rides with the full check and with its own category, on the
    # same rule as the deployed copies: an ask that names supervisor programs gets
    # only those. No `LLOYD_NSS_DB` is set here, so the check reads the store the
    # operator's own `bash scripts/install-ca.sh --check` would read.
    if not args.services and args.category in (None, CA_TRUST_CATEGORY):
        results.extend(check_ca_trust())
    # The timer enabledness guard, on the same rule: whole-fleet run or its own
    # category, never a `--services` ask. Since #2792 it also arms what it finds
    # disabled — the `enable --now <timer>` its FAIL line has always printed — so this
    # is the one line of the health route that mutates host systemd state, and
    # `--no-arm` is how a run stays the read-only diagnostic it was.
    if not args.services and args.category in (None, UNITS_CATEGORY):
        results.extend(check_unit_enabledness(arm=not args.no_arm))
    # The GPU power guard, on the same rule again: whole-fleet run or its own
    # category, never a `--services` ask. It shells out to systemctl and nvidia-smi
    # with no root and no config, and on a box with neither it returns one advisory
    # unknown row that changes no verdict (#2135).
    if not args.services and args.category in (None, GPU_POWER_CATEGORY):
        results.extend(check_gpu_power_limit())
    # The protected-path substrate guard, on the same rule a fourth time: whole-fleet run or its
    # own category, never a `--services` ask. It is one authenticated loopback GET of /state, and
    # an unreadable /state is a graded FAIL — a box that cannot say whether Bash is sandboxed is
    # not a healthy box (#2320).
    if not args.services and args.category in (None, PATH_SANDBOX_CATEGORY):
        results.extend(check_path_sandbox())

    # Calculate summary
    summary = {}
    for result in results:
        cat = result["category"]
        if cat not in summary:
            # Only graded rows feed the arithmetic, so an advisory row cannot move a
            # category's word — clause 3 of #1726, for a category that later gains
            # both kinds of row. A category with nothing but advisory rows has no
            # fault to name and reports its own verdict word instead. `unverified`
            # rows are excluded by the same call for the same reason (#2652): a
            # heading whose red means "the decision engine is down" must not be
            # turned red by a check that never reached the engine, so the category
            # goes through `_scored_rows` rather than a second copy of the rule.
            graded = [r for r in _scored_rows(results) if r["category"] == cat]
            if graded:
                healthy_in_cat = sum(1 for r in graded if r["healthy"])
                if healthy_in_cat == len(graded):
                    summary[cat] = "healthy"
                elif healthy_in_cat > 0:
                    summary[cat] = "degraded"
                else:
                    summary[cat] = "unhealthy"
            else:
                verdicts = {r.get("verdict") for r in results
                            if r["category"] == cat}
                if verdicts == {"ok"}:
                    summary[cat] = "healthy"
                elif "warn" in verdicts:
                    summary[cat] = "warn"
                else:
                    summary[cat] = "unknown"

    # Output
    if args.format == "json":
        print(format_json(results, summary))
    else:
        print(format_text(results, summary))


if __name__ == "__main__":
    main()
