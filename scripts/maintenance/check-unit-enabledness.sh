#!/usr/bin/env bash
# Every repo-tracked user TIMER must actually be ENABLED (#1891).
#
# WHY THIS EXISTS. Placing a unit file and enabling it are two different acts, and
# the repo only ever performed the first. `agent-services/setup/install-services.sh`
# symlinks the units (:72) and daemon-reloads (:79), then at :99-100 *echoes* the
# `systemctl --user enable ...` commands instead of running them; `promote.py`
# copies a unit and reloads; and `scripts/service_health_check.py`, the health
# route a human actually runs, is supervisor-only and never calls `systemctl`. So a
# timer can sit installed in `~/.config/systemd/user/`, disabled, forever, and
# nothing in this repository says so. That is goliath today:
# `systemctl --user is-enabled lloyd-cert-renew.timer` → `disabled`, `list-timers`
# → `0 timers listed`, while the other four repo timers read `enabled`. #1727
# shipped the renewal units, #1793 and #1727 each recorded the enable as owed, and
# the command has been handed forward three times. Enabling is host state and no
# diff can produce it — but the DRIFT is checkable, and until this script nothing
# checked it.
#
# WHO RUNS IT. `scripts/service_health_check.py` (#1989): `check_unit_enabledness`
# runs this file through subprocess on every whole-fleet run and on
# `--category units`, and turns its lines into graded rows. It reimplements
# nothing — the verdict, the `FAIL <unit>` lines and the denominator are all read
# from this script's output, so change the assertion here and nowhere else. By
# hand it is still `scripts/maintenance/check-unit-enabledness.sh` from the repo root.
#
# WHAT IT ASSERTS, AND WHAT IT DELIBERATELY DOES NOT.
#   - `.timer` units ONLY. A co-located `.service` legitimately reads something
#     other than `enabled`: `lloyd-cert-renew.service` is `static` because only the
#     timer ever starts it, and `nvidia-power-limit.service` is #1118's root-only
#     unit. Asking about a service can only produce a non-`enabled` verdict that is
#     not a defect, so this check never asks.
#   - One direction only: repo `.timer` file → `enabled`. NEVER set equality against
#     `~/.config/systemd/user/timers.target.wants/`, because that directory can hold units
#     this repository has no say in — on goliath today `freeclip-a2dp-fix.service` is live
#     with nothing tracked behind it (its disposition is recorded in install-services.sh's
#     HOST_LOCAL_UNITS by #2949), and a hand-installed timer
#     is the same shape one level up. `tests/test_data_home.py::_unresolved` rules
#     membership-not-equality the same way for the cutover fleet. Since #2864 tracked
#     `lloyd-graph-backup.timer`, box state no longer supplies a live-only TIMER to
#     demonstrate this, so `test_a_live_timer_with_no_repo_unit_cannot_fail_the_check` in
#     `tests/test_unit_enabledness_check.py` pins the rule over a planted name instead. A
#     live unit with no repo unit is not this check's business.
#   - `agent-services/systemd/*.timer`, one level deep. `agent-services/systemd/system/`
#     holds SYSTEM-scope units, which `systemctl --user` cannot even see.
#   - A glob matching zero timers FAILS. An empty unit set means nothing was looked
#     at, which is the shape of every vacuous green.
#
# The denominator line is part of the contract, not decoration: #1891's clause asks
# for each verdict beside how many timers were checked, so a run that quietly
# checked one timer reads differently from a run that checked five.
#
# WHAT THE FAIL LINE NAMES, AND WHY IT IS `enable --now`. `is-enabled` answers the
# unit FILE, not the running timer: `systemctl --user enable X.timer` links X and
# leaves `ActiveState=inactive` until the user manager restarts, so `enabled` is a
# necessary half of "this job will run" and never the whole of it. A remediation that
# printed the bare form therefore sent the operator away with this check green and the
# timer still not armed — which is what #2537 found on lloyd-persistence-window.timer,
# installed by #2428 and never once fired (`journalctl --user -u
# lloyd-persistence-window.service --since 2026-10-08` → `-- No entries --`). Pinning
# the arming form is the least this guard can do. #2537 also recorded `ActiveState`
# beside `is-enabled` as owed, not taken; #2702 takes it for the FAIL line, the line a
# red run actually prints (next block). What stayed owed until #2943 was the other half
# of that sentence: a timer whose unit FILE is enabled but which has never armed read the
# bare word `enabled` on its own row, so the run exited 0 and the green named eight
# enabled timers while `lloyd-egress-window.timer` sat at `ActiveState=inactive`,
# `LastTriggerUSec=`, absent from `list-timers --all`, with `journalctl --user -u
# lloyd-egress-window.service` → `-- No entries --`. #2943 takes exactly that, and nothing
# wider: a row answering `enabled` whose `show` read says `ActiveState=inactive` with no
# fire instant becomes `enabled-never-armed`, joins the misses, and the run exits 1. A
# full `ActiveState` column on every row is still NOT taken — the row word moves for that
# one signature only, because a timer reading `inactive` WITH a real instant is a schedule
# that ran, a timer reading `active` has no instant yet either and fires anyway, and a
# timer whose question could not be asked is not evidence of a missing arm. Those three
# keep their `enabled` row and the exit 0; making them red would trade a false green for
# a false red.
#
# `--now` is also the only form an unattended turn can act on at all — and since #2541
# (2026-10-11) that is a guard verdict on the TIMER, not a blanket property of the flag.
# systemd performs a start or a stop for `--now` itself, so the flag form used to be the
# same operation as the bare verb while every axis keyed on the verb word
# (`app/harness/service_control.py`, `app/harness/safety.py`) and none read the flag.
# All three now classify it as the verb it performs — `systemctl start` for
# `enable`/`reenable --now`, `systemctl stop` for `disable --now` — and refuse it to a
# background or unresolvable `task:*` session, so `systemctl --user enable --now
# <service>` is tier 2 and denied exactly as `systemctl --user start <service>` was
# already. `.timer` is the exempt unit type: arming a schedule starts no unit, so
# `systemctl --user enable --now <timer>` and `disable --now <timer>` stay tier 1 with
# no service-control label, which is what keeps the line below runnable by the turn that
# reads it — pinned by tests/test_service_control_guard.py, tests/test_grant_bash_tier.py
# and tests/test_persistence_window_unit.py against the guards themselves.
#
# WHAT THE FAIL LINE ALSO NAMES (#2702). A unit name and an arming command answer
# "what is broken"; they do not answer "since when", and that gap is what made
# lloyd-egress-window.timer cost a full triage to classify. #2702 arrived asserting an
# enabled→disabled transition inside ~24 h, and the one reading that refutes it — that
# this timer has no fire instant because it never had one (`LastTriggerUSec=` empty; the
# unit's only add commit is 01d0082b on 2026-10-11; no egress entry exists in
# `~/.config/systemd/user/timers.target.wants/`) — was nowhere on the alarm, so the pass
# spent itself hunting a disable that had never happened. The FAIL line therefore carries
# the unit's own arming evidence, read with `systemctl --user show -p LastTriggerUSec
# -p ActiveState`: the last instant systemd recorded the timer firing, or the literal
# `never-armed` when that property comes back with no value (an unset timestamp renders as
# an empty string, and `n/a` is systemd's other no-value spelling), beside `ActiveState`.
#
# What the pair separates is the three states a bare `disabled` blurs. A real instant
# with `ActiveState=inactive` is a schedule that ran and then stopped. `ActiveState=active`
# means the unit FILE is disabled while the loaded timer still runs — `disable` without
# `--now` — so it keeps firing until the user manager next restarts and the red row is
# describing a future stop, not a present one. `ActiveState=inactive` with no instant is a
# unit that was installed and never armed: a missing action, not a broken job.
#
# `never-armed` claims exactly what the property holds and no more: systemd reports no
# fire instant for the unit it has loaded now, so a timer that fired before the user
# manager last restarted reads the same way. That still answers the question the alarm is
# asked, because a recorded fire would be present and would name itself.
#
# The `show` read is ONE call per timer, asked after that timer's `is-enabled` answer and
# reused for both jobs it now serves: the pass-row never-armed signature above, and the
# evidence sentence on a FAIL line. Before #2943 it was asked only of a miss — one call per
# miss, none at all on a green run, which the stub's query log pinned — and a check that
# reads the rows that PASS cannot be made without asking, so that node was revised rather
# than kept. What it pins now is the cost that did NOT grow: two `systemctl` calls per
# timer and never three, one `show` per unit even though two consumers want it. The read
# stays bounded the same way the route expects: `service_health_check.py` runs this script
# under a 30-second timeout, and the whole script over this box's eight timers — sixteen
# `systemctl` calls — ran in 0.114-0.120 s over five back-to-back runs on the box it ships
# to (0.118, 0.115, 0.114, 0.114, 0.120).
# Stderr is discarded on purpose: this is information beside a verdict, and a
# failed read degrades to the two markers below rather than adding a line that
# `scripts/service_health_check.py`, which folds this script's stderr into the same text it
# parses, could mistake for a verdict of its own.
#
# TWO FAIL SENTENCES, ONE FAIL SHAPE (#2943). The prefix `FAIL <unit> ` and the tail
# `— run: systemctl --user enable --now <unit>` do not move: `scripts/service_health_check.py`
# parses the prefix to decide which rows are unhealthy and which timers it may arm, and
# `test_the_command_this_line_prints_is_runnable` lifts the tail out of the line to put it
# through the harness guards. Only the middle clause changes with the reason — `is not
# enabled` for a unit `is-enabled` refused, `is enabled but never armed` for one it
# accepted and `show` says holds no fire instant. The second is the true sentence for that
# unit: its unit FILE IS linked, and printing `is not enabled` about it would send the
# reader off to re-link a file that is already linked, which is the act that produced this
# state in the first place (the installed link on this box came from a `--dry-run` that
# systemd 261 performed for real).
#
# tests/test_unit_enabledness_check.py drives this through a stub `systemctl` on
# PATH and `--unit-dir`, so the suite never reads the live host — the same reason
# tests/test_cert_renew_units.py refuses to pin enabledness.
set -uo pipefail

usage() {
  cat <<'EOF'
usage: check-unit-enabledness.sh [--unit-dir DIR] [--help]

Assert every *.timer in DIR (default <repo>/agent-services/systemd) answers
`systemctl --user is-enabled` with `enabled` AND, for the ones that do, that systemd
has armed it at least once: an enabled timer reading `ActiveState=inactive` with no
`LastTriggerUSec` is reported as `enabled-never-armed` and fails like a miss. Exit 0
only when every timer clears both; a DIR that holds no timer at all is itself
reported as a failure, not a pass. Each FAIL line also names that timer's arming
evidence — ActiveState and its last trigger instant, or `never-armed` when systemd
holds none.
EOF
}

# One property's value out of `systemctl show` output, by key. Parsed by key rather than
# by position because systemd answers the properties it was asked for in its own order,
# not the request's: `-p LastTriggerUSec -p ActiveState` comes back ActiveState first.
# Absent is not the same as empty — no line is an unreadable answer, and the caller
# renders both below rather than printing a blank.
show_prop() {
  local key="$1" text="$2" line
  line="$(printf '%s\n' "$text" | grep -m1 -- "^${key}=")" || return 1
  printf '%s' "${line#"${key}="}"
}

# The one arming read for one timer: both properties, in one call, stdout only. Every
# timer gets exactly one of these, asked after its `is-enabled` answer, and BOTH consumers
# parse this same capture — the pass-row signature and the FAIL-line evidence. That is what
# keeps "we now inspect the rows that pass" from becoming a second round trip per unit.
arm_show() {
  systemctl --user show -l -p LastTriggerUSec -p ActiveState "$1" 2>/dev/null
}

# The two properties rendered for a FAIL line. Both fields always render, so a line that
# could not read systemd still says so instead of printing `ActiveState=, LastTriggerUSec=`
# and reading like an empty value.
render_evidence() {
  local state="$1" trigger="$2"
  [[ -z "$state" || "$state" == "n/a" ]] && state="no-answer"
  [[ -z "$trigger" || "$trigger" == "n/a" ]] && trigger="never-armed"
  printf 'ActiveState=%s, LastTriggerUSec=%s' "$state" "$trigger"
}

# The never-armed signature, from the two readings: the loaded unit is not running AND
# systemd holds no fire instant for it. Both halves are required, which is the whole
# reason an unreadable answer cannot turn a healthy timer red — a failed `show` leaves
# `state` empty, and empty is not `inactive`. `active` is excluded on purpose too: a timer
# armed five seconds ago has no instant yet either, and it is armed. Returns 0 for the
# signature, 1 for everything else including `no answer at all`.
is_never_armed() {
  [[ "$1" == "inactive" && ( -z "$2" || "$2" == "n/a" ) ]]
}

UNIT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit-dir)
      UNIT_DIR="${2:-}"
      shift 2
      ;;
    --unit-dir=*)
      UNIT_DIR="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "FAIL unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$UNIT_DIR" ]]; then
  REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
  UNIT_DIR="$REPO/agent-services/systemd"
fi

if [[ ! -d "$UNIT_DIR" ]]; then
  echo "FAIL $UNIT_DIR is not a directory; nothing was checked"
  echo "checked 0 timers: 0 enabled, 0 not enabled"
  exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
  echo "FAIL no systemctl on PATH; every verdict below would be unverifiable"
  echo "checked 0 timers: 0 enabled, 0 not enabled"
  exit 1
fi

shopt -s nullglob
TIMERS=("$UNIT_DIR"/*.timer)
shopt -u nullglob

TOTAL=${#TIMERS[@]}
if (( TOTAL == 0 )); then
  echo "FAIL 0 *.timer in $UNIT_DIR — the check looked at nothing, which is not a pass"
  echo "checked 0 timers: 0 enabled, 0 not enabled"
  exit 1
fi

MISS=()
MISS_WHY=()
MISS_EVID=()
for path in "${TIMERS[@]}"; do
  unit="$(basename -- "$path")"
  # Combined capture: `is-enabled` writes the state to stdout for the known states
  # and to stderr for the ones it considers an error (`Unit ... not loaded`), and an
  # empty answer is its own verdict — "we could not ask" is never `enabled`.
  verdict="$(systemctl --user is-enabled "$unit" 2>&1 | head -n 1)"
  verdict="${verdict//[[:space:]]/}"
  [[ -z "$verdict" ]] && verdict="no-answer"
  # One arming read per timer, whatever `is-enabled` said, because the verdict that needs
  # it is the one that PASSES. Both consumers parse this single capture.
  show="$(arm_show "$unit")"
  state="$(show_prop ActiveState "$show")"
  trigger="$(show_prop LastTriggerUSec "$show")"
  why=""
  if [[ "$verdict" == "enabled" ]]; then
    # An `enabled` row is the row nobody used to look at twice, which is where the
    # installed-but-unarmed timer lived. The word on the row moves with the verdict:
    # `scripts/service_health_check.py` copies it into the row's status, so a row still
    # reading the bare `enabled` would keep the false green inside the health report even
    # after this script started exiting 1.
    if is_never_armed "$state" "$trigger"; then
      verdict="enabled-never-armed"
      why="never-armed"
    fi
  else
    why="not-enabled"
  fi
  printf '%s\t%s\n' "$unit" "$verdict"
  if [[ -n "$why" ]]; then
    MISS+=("$unit")
    MISS_WHY+=("$why")
    MISS_EVID+=("$(render_evidence "$state" "$trigger")")
  fi
done

printf 'checked %d timers: %d enabled, %d not enabled\n' \
  "$TOTAL" "$(( TOTAL - ${#MISS[@]} ))" "${#MISS[@]}"

if (( ${#MISS[@]} > 0 )); then
  # Three parallel arrays, one entry per miss: the name, the reason, the evidence read once
  # in the loop above. `K` in the denominator line is `${#MISS[@]}`, so it is the number of
  # FAIL lines this block prints and the route's unhealthy-row count moves with it — for a
  # never-armed timer that means it is counted under `not enabled`, which reads loosely for
  # a linked unit and is deliberately not re-worded: the line's shape is the contract
  # `scripts/service_health_check.py` parses and `tests/test_unit_enabledness_check.py`
  # pins, and the FAIL line beneath it is where the precise reason belongs.
  for i in "${!MISS[@]}"; do
    unit="${MISS[$i]}"
    reason="is not enabled"
    [[ "${MISS_WHY[$i]}" == "never-armed" ]] \
      && reason="is enabled but never armed"
    # `--now` is the arming half, not decoration. Plain `enable` only links the unit
    # file into `timers.target.wants/`; the timer keeps reading `ActiveState=inactive`
    # until the user manager is restarted, so a hint that prints the bare form sends
    # its reader away with the check green and no job scheduled. That is how
    # lloyd-persistence-window.timer (#2428) sat from install to #2537: the only
    # command this script ever printed was the half that does not arm it. `--now` is
    # also the form SETUP.md carries for every repo timer it documents, and — since
    # #2541 made `--now` count as the verb it performs — the only `--now` a worker turn
    # may run: `enable --now <service>` is now refused to a background session exactly
    # as `start <service>` always was, and `test_the_command_this_line_prints_is_runnable`
    # keeps the timer form, which is what prints here, reachable.
    #
    # The pointer is by content, never by line number: a `SETUP.md:<n>` moves with
    # every edit above it and then names something else, which is the rule
    # tests/test_unit_enabledness_check.py already applies to prose pointers. The one
    # this line used to print had drifted — the number it named was a `/proc` environ
    # capture, not an enable route.
    #
    # The evidence comes first and the command second, because the reader needs the
    # first to know what the second is for: on a timer whose `ActiveState` already reads
    # `active` the timer is running and only its unit FILE is disarmed, and on one that
    # never armed this line is the only place the missing action was ever written down.
    echo "FAIL $unit $reason — armed evidence: ${MISS_EVID[$i]}" \
         "— run: systemctl --user enable --now $unit" \
         "(SETUP.md carries the enable route for each repo timer)"
  done
  exit 1
fi

exit 0
