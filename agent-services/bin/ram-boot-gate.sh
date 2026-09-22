#!/usr/bin/env bash
# =============================================================================
# The ONE definition of the host-RAM boot gate for `agent-llm-primary`.
#
# Two routes boot this engine and both have to answer the same question with the
# same gauge: has the PREVIOUS boot's host-RAM mapping been released, and is
# there enough room left to risk the next one? `supervisorctl stop` returns when
# the processes are *signalled*, not when the kernel has their memory back, and
# this engine holds a 95.37 GiB BF16 n-gram table in HOST ram. Start the next
# boot too soon and two of those coexist -- on 2026-09-08 that took a 251 GiB box
# to a 230 GiB peak and systemd-oomd killed the whole agent-supervisord unit, 953
# processes, every service on the machine and not just the engine being
# restarted, twice. Everything came back on autorestart, but the work was lost
# and the failure looked like whatever config was under test rather than the
# restart cadence.
#
# The two routes, and how each reads this file:
#   LANDING   `scripts.automod.round restart --only agent-llm-primary`
#             -> `scripts/automod/ram_gate.py` parses the numbers below and
#                `_restart_primary` (scripts/automod/promote.py) applies
#                PRIMARY_RAM_*.
#   SWEEP     `bin/flash-next-run-arm.sh` sources this file and calls
#                `ram_gate_wait_for_room` before it starts the engine, on
#                SWEEP_RAM_*.
# Sourcing this file is the shell route's way in; parsing it is the Python
# route's. Neither route keeps a copy of a number, which is the whole point:
# both pairs are machine- and day-specific and will move again, and one moving
# must not silently leave the other one stale. (#1340: 180/150 lived in
# promote.py and 150/120 in the arm script, with nothing between them.)
#
# WHAT THIS GATE IS NOT. Reading it as the oomd protection is how 2026-09-15 got
# misattributed to a qemu VM on the desktop. On 2026-09-17 this check PASSED at
# 198 GiB and the unit was killed 129 seconds later: ONE boot drives its own
# cgroup to ~226 GiB (a 170 GiB checkpoint read plus a 95 GiB shared mapping,
# page cache charged to the reader's cgroup), so the boot consumes the very
# thing this gauge measures -- MemAvailable fell to 79 GiB while the load ran.
# What keeps oomd off the unit is `Slice=lloyd.slice` on
# agent-supervisord.service (oomd watches only app.slice); see the comment there,
# including the MemoryHigh attempt that must not come back. Do not raise a number
# here expecting it to stop an oomd kill -- it cannot. What it does answer
# correctly is its original question, the one MemAvailable is the right gauge
# for: has the previous engine's shared mapping been released yet.
#
# WHY THE SWEEP PAIR SITS LOWER (150/120 under 180/150), which is intentional
# and not drift. The landing route runs unattended, inside a promotion whose
# other legs already hold the box's attention, and its only witness afterwards is
# an alert; a refused leg costs nothing but leaves the engine stopped. The sweep
# route is run by whoever is standing at the machine with the worker pool paused
# AND drained, the backend down and a guardian maintenance lease held -- all four
# preconditions the arm script's own header lists, each of which has bitten this
# box. A supervised sweep can afford the tighter margin because a human sees the
# refusal on the terminal. The landing floor is what a boot actually needs on top
# of whatever else the desktop holds: 150/120 was the first cut, and the first
# real use (2026-09-15 23:47Z) got through them and still lost the unit -- a
# 16 GiB qemu VM had joined the desktop, MemAvailable read 57 GiB with the engine
# up, the stop freed the table to just past the floor and the boot's own
# transient took the box to pressure, so systemd-oomd killed
# agent-supervisord.service at 23:52:46Z (memwatch snapshot 20260915_235555). The
# landing floor became 180 and its abort line became the old floor. The two pairs
# stay in one file so a person moving one sees the other and the reason.
#
# Every threshold below is a whole-GiB reading of /proc/meminfo's MemAvailable.
# =============================================================================

# --- LANDING route: `_restart_primary` in scripts/automod/promote.py ---------
# MemAvailable has to come back up to the floor before the next boot is started.
# If the wait budget runs out with the host still under the abort line, the leg
# leaves the engine STOPPED and reports a refusal instead of starting a boot into
# pressure; above the abort line but under the floor it boots anyway, which is
# what "waited as long as it could" means on this route too.
PRIMARY_RAM_FLOOR_GIB=180
PRIMARY_RAM_ABORT_GIB=150
# How long the landing route waits for that floor, refreshing the guardian
# maintenance lease on every pass so the pause reads as deliberate.
PRIMARY_RAM_WAIT_SECONDS=600

# --- SWEEP route: bin/flash-next-run-arm.sh ---------------------------------
# Wait for the previous arm's table to come back, refuse the arm below the abort
# line, and never start the engine on a refusal.
SWEEP_RAM_WAIT_GIB=150
SWEEP_RAM_ABORT_GIB=120

# --- how the sweep route waits -------------------------------------------------
# Not thresholds, cadence only, and overridable so a test can drive the gate
# without sitting through five minutes of sleep.
RAM_GATE_TRIES="${RAM_GATE_TRIES:-60}"
RAM_GATE_INTERVAL="${RAM_GATE_INTERVAL:-5}"

# ram_gate_available_gib [MEMINFO] -- MemAvailable in whole GiB, truncated. The
# file is a parameter (or the MEMINFO environment variable) so a test can drive
# the gate without writing to /proc/meminfo. An unmeasurable reading is reported
# as 0, because a gate that cannot measure has to refuse, not guess.
ram_gate_available_gib() {
  local n
  n=$(awk '/MemAvailable/{print int($2/1048576)}' "${1:-${MEMINFO:-/proc/meminfo}}" 2>/dev/null)
  [[ "$n" =~ ^[0-9]+$ ]] || { echo "ram-boot-gate: no MemAvailable reading, treating it as 0 GiB" >&2; n=0; }
  echo "$n"
}

# ram_gate_numbers_ok -- a half-read or corrupt definition must refuse, never
# boot on a remembered number. Also refuses a pair whose abort line stood at or
# above its own wait line: such a gate can only ever pass or only ever refuse,
# which is how a threshold edit silently disables one.
ram_gate_numbers_ok() {
  local name
  for name in PRIMARY_RAM_FLOOR_GIB PRIMARY_RAM_ABORT_GIB PRIMARY_RAM_WAIT_SECONDS \
              SWEEP_RAM_WAIT_GIB SWEEP_RAM_ABORT_GIB; do
    [[ "${!name:-}" =~ ^[0-9]+$ ]] \
      || { echo "ram-boot-gate: $name is '${!name:-unset}', expected an integer" >&2; return 1; }
  done
  (( PRIMARY_RAM_ABORT_GIB < PRIMARY_RAM_FLOOR_GIB )) \
    || { echo "ram-boot-gate: PRIMARY_RAM_ABORT_GIB must sit below PRIMARY_RAM_FLOOR_GIB" >&2; return 1; }
  (( SWEEP_RAM_ABORT_GIB < SWEEP_RAM_WAIT_GIB )) \
    || { echo "ram-boot-gate: SWEEP_RAM_ABORT_GIB must sit below SWEEP_RAM_WAIT_GIB" >&2; return 1; }
  return 0
}

# ram_gate_wait_for_room [label] -- the SWEEP route's gate. Waits up to
# RAM_GATE_TRIES x RAM_GATE_INTERVAL seconds for MemAvailable to reach
# SWEEP_RAM_WAIT_GIB, then returns 0 if it is at or over SWEEP_RAM_ABORT_GIB and
# 3 if it is not; 4 means the definition itself is unreadable. The caller must
# not start the engine on a non-zero status.
ram_gate_wait_for_room() {
  local label="${1:-arm}"
  local tries="${RAM_GATE_TRIES:-60}" interval="${RAM_GATE_INTERVAL:-5}"
  local avail=0 try
  ram_gate_numbers_ok || return 4
  echo -n "waiting for host RAM to be released "
  for ((try = 0; try < tries; try++)); do
    avail=$(ram_gate_available_gib)
    [[ "$avail" -ge "$SWEEP_RAM_WAIT_GIB" ]] && break
    echo -n "."
    sleep "$interval"
  done
  echo " ${avail} GiB available"
  if [[ "$avail" -lt "$SWEEP_RAM_ABORT_GIB" ]]; then
    echo "!! ABORT $label: only ${avail} GiB host RAM free; a 170 GiB load would risk the oomd kill documented in bin/ram-boot-gate.sh."
    return 3
  fi
  return 0
}
