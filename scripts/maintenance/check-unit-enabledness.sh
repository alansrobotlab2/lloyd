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
#     `~/.config/systemd/user/timers.target.wants/`, because `lloyd-graph-backup.timer`
#     is live on goliath and tracked nowhere under `agent-services/systemd/`
#     (`tests/test_data_home.py` already rules membership-not-equality for that same
#     unit). A live unit with no repo unit is not this check's business.
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
# the arming form is the least this guard can do; checking `ActiveState` itself is a
# wider contract (#2537 records it as owed, not taken).
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
# tests/test_unit_enabledness_check.py drives this through a stub `systemctl` on
# PATH and `--unit-dir`, so the suite never reads the live host — the same reason
# tests/test_cert_renew_units.py refuses to pin enabledness.
set -uo pipefail

usage() {
  cat <<'EOF'
usage: check-unit-enabledness.sh [--unit-dir DIR] [--help]

Assert every *.timer in DIR (default <repo>/agent-services/systemd) answers
`systemctl --user is-enabled` with `enabled`. Exit 0 only when all of them do; a
DIR that holds no timer at all is itself reported as a failure, not a pass.
EOF
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
for path in "${TIMERS[@]}"; do
  unit="$(basename -- "$path")"
  # Combined capture: `is-enabled` writes the state to stdout for the known states
  # and to stderr for the ones it considers an error (`Unit ... not loaded`), and an
  # empty answer is its own verdict — "we could not ask" is never `enabled`.
  verdict="$(systemctl --user is-enabled "$unit" 2>&1 | head -n 1)"
  verdict="${verdict//[[:space:]]/}"
  [[ -z "$verdict" ]] && verdict="no-answer"
  printf '%s\t%s\n' "$unit" "$verdict"
  [[ "$verdict" != "enabled" ]] && MISS+=("$unit")
done

printf 'checked %d timers: %d enabled, %d not enabled\n' \
  "$TOTAL" "$(( TOTAL - ${#MISS[@]} ))" "${#MISS[@]}"

if (( ${#MISS[@]} > 0 )); then
  for unit in "${MISS[@]}"; do
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
    echo "FAIL $unit is not enabled — run: systemctl --user enable --now $unit" \
         "(SETUP.md carries the enable route for each repo timer)"
  done
  exit 1
fi

exit 0
