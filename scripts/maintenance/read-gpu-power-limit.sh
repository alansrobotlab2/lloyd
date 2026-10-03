#!/usr/bin/env bash
# Print the GPU wattage the power-limit unit DECLARES, on stdout, verbatim.
#
# Why a script and not a `subprocess.run(["systemctl", ...])` inside
# `scripts/service_health_check.py`: `tests/test_unit_enabledness_route.py
# ::test_no_enabledness_assertion_is_reimplemented_in_python` walks that file's string
# constants and refuses the word `systemctl` in any of them, because a unit question
# answered in Python is a second, unguarded implementation of one. `#2135`'s probe reads
# the same source the fixer script does — this unit's `Environment=` lines — so it takes
# the same route: a shell guard asks systemd, Python parses and judges.
#
# Read-only. No root, no GPU needed: `systemctl show` answers for a unit that is merely
# installed, and this script changes nothing.
#
# Exit status is systemctl's, so the checker can tell "no such unit" (non-zero, unknown)
# apart from "the unit declares no wattage" (zero, an empty answer).
set -u

UNIT="${GPU_POWER_UNIT:-nvidia-power-limit.service}"

out=$(systemctl show "$UNIT" --property=Environment 2>&1)
code=$?
printf '%s\n' "$out"
exit "$code"
