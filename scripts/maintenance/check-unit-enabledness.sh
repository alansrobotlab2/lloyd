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
# WHO RUNS IT. Nobody automatically, as of #1891: run it by hand as
# `scripts/maintenance/check-unit-enabledness.sh` from the repo root. Wiring it
# into a route someone actually executes (a health script, a nightly autonomy task)
# is the ruling #1891 left owed — that is a behaviour change on a route, not this
# check, and the item names it rather than pretending the file has a caller.
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
    echo "FAIL $unit is not enabled — run: systemctl --user enable $unit (SETUP.md:1520)"
  done
  exit 1
fi

exit 0
