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
import socket
import subprocess
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

SERVICES = {
    # Supervisor services — status via supervisorctl, no HTTP check
    "lloyd-backend": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "lloyd-mc:lloyd-backend"], "category": "lloyd"},
    "lloyd-frontend": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "lloyd-mc:lloyd-frontend"], "category": "lloyd"},
    "lloyd-mcp": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "lloyd-mc:lloyd-mcp"], "category": "lloyd"},

    # LLM inference servers — status via supervisorctl, no HTTP check
    "agent-llm-primary": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "agent-llm-primary"], "category": "supervisor"},
    "agent-llm-secondary": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "agent-llm-secondary"], "category": "supervisor"},
    # djev shares GPU 2 with the secondary, so at most one of these two is ever
    # enabled. Port 8011 is the structured decision API, which start-djev.sh
    # only launches once vLLM on 8010 answers — so an open 8011 means the whole
    # stack is serving, while an open 8010 during a cold boot does not.
    "agent-djev": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "agent-djev"], "category": "supervisor", "port": 8011},

    # QMD retrieval daemon — the path every vault_search/vault_recall needs,
    # and until #406 this check could not see it at all. Port is the one
    # `agent-services/supervisor/conf.d/agent-qmd-daemon.conf` launches it on
    # (`qmd mcp --http --port 8181`); it serves a real /health route, so a
    # supervisor RUNNING line that no longer has a listener behind it — the
    # failure this entry exists to catch — now shows up here instead of in the
    # first failed vault_search of the day. This is also the first SERVICES
    # entry to declare a `port`, i.e. the first thing to ever exercise
    # http_check().
    "agent-qmd-daemon": {"command": ["supervisorctl", "-c", SUPervisor_CONF, "status", "agent-qmd-daemon"], "category": "retrieval", "port": 8181},
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
    if len(cmd) >= 5 and cmd[0] == "supervisorctl" and "status" in cmd:
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
                ["supervisorctl", "-c", SUPervisor_CONF, "status"],
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
            error = str(exc)[:160]
        _FLEET = (fleet, error)
    return _FLEET


def _ask_program(name: str) -> Optional[str]:
    """Re-ask supervisor for one program's state, when the fleet read gave no usable one.

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
    """
    try:
        r = subprocess.run(
            ["supervisorctl", "-c", SUPervisor_CONF, "status", name],
            capture_output=True, text=True, timeout=10)
        fields = (r.stdout or "").split()
        return fields[1] if len(fields) > 1 else None
    except Exception:
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
        state = _ask_program(name)
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
        return [{"name": "supervisorctl status", "category": FLEET_CATEGORY,
                 "healthy": False, "exit_code": 0,
                 "status": f"could not read the loaded program list ({error}) — no "
                           "undeclared program was graded, so this run covered only "
                           "the declared services"}]
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
# The .sh stays the single source of the assertion. Nothing here asks systemd
# anything or decides which answer counts as a pass: a row is unhealthy only where
# the script exited non-zero AND printed its own `FAIL <unit>` line for that timer,
# and the denominator in every row is the script's own `checked N timers: ...` line,
# copied, never recounted.
UNITS_CATEGORY = "units"
UNIT_ENABLEDNESS = _TREE / "scripts" / "maintenance" / "check-unit-enabledness.sh"
_UNITS_DENOMINATOR_RE = re.compile(r"^checked \d+ timers?: .*$", re.M)
_UNITS_ROW_RE = re.compile(r"^(\S+\.timer)\t(\S+)$", re.M)
_UNITS_FAIL_RE = re.compile(r"^FAIL (\S+\.timer) ", re.M)


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


def check_unit_enabledness(runner=subprocess.run) -> list:
    """One graded row per repo timer, from the guard's own output and exit code.

    Exit 0 is the only pass. On a non-zero exit the rows the script itself named in
    a `FAIL <unit>` line are unhealthy; if it failed without naming a timer (no unit
    directory, no systemctl, zero timers) one `unit-enabledness` row carries its
    first FAIL line, because a guard that looked at nothing is never a pass. A guard
    that could not be run at all is likewise an unhealthy row, not a missing one.
    """
    argv = _unit_enabledness_argv()
    try:
        proc = runner(argv, capture_output=True, text=True, timeout=30)
    except Exception as exc:
        return [_units_row("unit-enabledness", False,
                           f"unknown: the timer enabledness guard could not be run "
                           f"({str(exc)[:120]}) — nothing was checked", -1, str(exc))]
    text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    denominator = _UNITS_DENOMINATOR_RE.search(text)
    counted = denominator.group(0) if denominator else "the guard printed no denominator"
    failed = set(_UNITS_FAIL_RE.findall(text)) if proc.returncode != 0 else set()
    rows = []
    for unit, verdict in _UNITS_ROW_RE.findall(text):
        rows.append(_units_row(f"timer:{unit}", unit not in failed,
                               f"{verdict} — {counted}", proc.returncode, text))
    if proc.returncode != 0 and not any(not r["healthy"] for r in rows):
        head = next((ln for ln in text.splitlines() if ln.startswith("FAIL")),
                    text.splitlines()[0] if text else "no output")
        rows.append(_units_row("unit-enabledness", False,
                               f"failed (exit {proc.returncode}): {head[:200]} — {counted}",
                               proc.returncode, text))
    elif not rows:
        rows.append(_units_row("unit-enabledness", False,
                               f"unknown: the guard exited 0 and named no timer — {counted}",
                               proc.returncode, text))
    return rows


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
        if "supervisorctl" in command[0]:
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
        return {
            "name": name,
            "status": "error",
            "healthy": False,
            "exit_code": -1,
            "output": str(e),
            "category": category
        }


def _advisory_rows(results: list) -> list:
    """Rows that report something without claiming a fault (#1726 clause 3).

    A row is advisory by its own declaration, never by its category name, so a future
    round cannot make a real fault invisible by filing it under `ca`.
    """
    return [r for r in results if r.get("advisory")]


def _scored_rows(results: list) -> list:
    """The rows the verdict is arithmetic over."""
    return [r for r in results if not r.get("advisory")]


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
    lines.append("")

    for result in results:
        icon = ("[!]" if result.get("advisory")
                else "[✓]" if result["healthy"] else "[✗]")
        lines.append(f"{icon} {result['name']}\u2014 {result['status']}")

    lines.append("")
    lines.append("Categories:")
    for cat, status in summary.items():
        lines.append(f"  {cat.capitalize()}: {status}")

    return "\n".join(lines)


def format_json(results: list, summary: dict) -> str:
    """Format results as JSON.

    `unhealthy` is faults only: an advisory row goes into its own count, so
    `healthy + unhealthy + advisory == total_services` and a caller that reads just
    `unhealthy` is not told the box is broken by a retired certificate. The row
    itself, with its `verdict` and its two fingerprints, is always in `services`.
    """
    scored = _scored_rows(results)
    advisory = _advisory_rows(results)
    healthy_count = sum(1 for r in scored if r["healthy"])

    output = {
        "check_time": datetime.now(timezone.utc).isoformat(),
        "total_services": len(results),
        "healthy": healthy_count,
        "unhealthy": len(scored) - healthy_count,
        "advisory": len(advisory),
        "services": results,
        "summary": summary
    }
    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Check status of multiple services")
    parser.add_argument("--services", nargs="+", help="Specific services to check")
    parser.add_argument("--category", choices=list(CATEGORIES.keys()), help="Check by category")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

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
    if not args.services and args.category in (None, DEPLOY_CATEGORY):
        results.extend(check_deployed_copies())
    # The CA trust guard rides with the full check and with its own category, on the
    # same rule as the deployed copies: an ask that names supervisor programs gets
    # only those. No `LLOYD_NSS_DB` is set here, so the check reads the store the
    # operator's own `bash scripts/install-ca.sh --check` would read.
    if not args.services and args.category in (None, CA_TRUST_CATEGORY):
        results.extend(check_ca_trust())
    # The timer enabledness guard, on the same rule: whole-fleet run or its own
    # category, never a `--services` ask.
    if not args.services and args.category in (None, UNITS_CATEGORY):
        results.extend(check_unit_enabledness())

    # Calculate summary
    summary = {}
    for result in results:
        cat = result["category"]
        if cat not in summary:
            # Only graded rows feed the arithmetic, so an advisory row cannot move a
            # category's word — clause 3 of #1726, for a category that later gains
            # both kinds of row. A category with nothing but advisory rows has no
            # fault to name and reports its own verdict word instead.
            graded = [r for r in results
                      if r["category"] == cat and not r.get("advisory")]
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
