#!/usr/bin/env bash
# Run one A/B arm on the primary slot: restart the engine with this arm's env,
# wait for it to serve, record what it ACTUALLY chose, then benchmark it.
#
#   bin/flash-next-run-arm.sh A2-b12x MOE_BACKEND=flashinfer_b12x
#   bin/flash-next-run-arm.sh A1-tier1 LANGUAGE_MODEL_ONLY=1 KV_CACHE_MEMORY_BYTES=12884901888
#
# PRECONDITIONS, all of which have bitten this box already:
#   - the worker pool is paused AND drained (POST /api/workers/pause), and the
#     backend is stopped. Neither is sufficient alone: on 2026-09-08 two batch
#     jobs launched from a second Claude Code session drove the engine at 8
#     concurrent for the whole window with the backend already down, and
#     nothing in the throughput numbers said so.
#   - the guardian holds a maintenance lease, or every restart here reads as an
#     incident and fans out to six channels including the spoken one.
#   - HEAD == last-known-good, so a restart cannot be mistaken for a bad
#     promotion and rolled back.
#
# The engine is started under supervisord so the arm runs the same supervision
# the production config does. supervisord passes no per-arm environment, so the
# arm's env is written to a file the program's own launcher sources.
set -uo pipefail

LABEL="${1:?usage: flash-next-run-arm.sh <label> [KEY=VAL ...]}"
shift

ROOT=/home/alansrobotlab/lloyd
SUP="/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c $ROOT/agent-services/supervisor/supervisord.conf"
ENVFILE="${LLOYD_DATA:-$HOME/lloyd-data}/logs/services/flash-next-arm.env"
LOG="${LLOYD_DATA:-$HOME/lloyd-data}/logs/services/agent-llm-primary.log"
RESULTS="${LLOYD_DATA:-$HOME/lloyd-data}/logs/services/flash-next-arms.jsonl"

# The host-RAM boot gate — its thresholds, its wait and the reason for both —
# lives in ONE file that the landing route (`round restart --only
# agent-llm-primary`) reads too. This script is the SWEEP route, so it uses the
# SWEEP_RAM_* pair, which sits lower than the landing pair by design; the
# definition's header says why and what the gate cannot do. No number in this
# file is the boot gate's, and if the definition cannot be read the arm is
# refused before anything is stopped.
RAM_GATE_SCRIPT="$ROOT/agent-services/bin/ram-boot-gate.sh"
if [[ ! -r "$RAM_GATE_SCRIPT" ]]; then
  echo "!! ABORT $LABEL: cannot read the boot-gate definition at $RAM_GATE_SCRIPT."
  exit 3
fi
# shellcheck source=/dev/null
source "$RAM_GATE_SCRIPT"

{
  echo "# arm: $LABEL   written $(date -Is)"
  for kv in "$@"; do echo "export $kv"; done
} > "$ENVFILE"

echo "=== arm $LABEL ==="
cat "$ENVFILE"

# Remember where the log ends, so bootfacts reads THIS boot and not the last
# one. Writing a marker line into the file does NOT work: supervisord owns that
# descriptor and an appended line does not survive its restart of the program.
# A byte offset needs nothing from supervisord and cannot be clobbered.
LOG_OFFSET=$(wc -c < "$LOG")

echo "--- restarting engine (cold boot reads 170 GiB; expect 3-5 min) ---"
$SUP stop agent-llm-primary

# WAIT FOR THE HOST TABLE TO BE RELEASED BEFORE STARTING THE NEXT BOOT. The
# gate — what it waits for, for how long, where it refuses, and the oomd
# history that explains what it is NOT — lives in `bin/ram-boot-gate.sh`, which
# the landing route reads out of the same file, so the two routes cannot disagree
# about what "room to boot" means. A non-zero status here means the room never
# came back: the engine stays stopped, this arm is refused, and nothing starts.
ram_gate_wait_for_room "$LABEL" || exit 3
# Backgrounded on purpose: startsecs=300 means `start` does not RETURN for five
# minutes even though the engine is usually serving by three, and this runs six
# times in a sweep. The launcher's own comment says it: poll :8096 instead.
$SUP start agent-llm-primary >/dev/null 2>&1 &
SUP_START_PID=$!

python3 - "$LABEL" <<'PY'
import sys, time, urllib.request
label = sys.argv[1]
t0 = time.time()
while time.time() - t0 < 1800:
    try:
        urllib.request.urlopen("http://127.0.0.1:8096/health", timeout=5)
        print(f"[{label}] serving after {time.time()-t0:.0f}s", flush=True)
        break
    except Exception:
        time.sleep(5)
else:
    print(f"[{label}] NEVER CAME UP in 1800s", flush=True)
    raise SystemExit(1)
PY
if [[ $? -ne 0 ]]; then
  echo "arm $LABEL failed to boot — read $LOG"
  wait "$SUP_START_PID" 2>/dev/null
  exit 1
fi
wait "$SUP_START_PID" 2>/dev/null   # let supervisord finish marking it RUNNING

echo "--- what the engine actually chose ---"
# +N is 1-indexed from the start, so offset+1 is the first byte of this boot.
# Then narrow again to the LAST "A/B config:" line. The byte offset alone is
# not enough: if the previous boot was still initialising when this arm ran
# `stop` (an OOM recovery, say), its tail lands inside the slice and the boot
# guard below counts it as a second engine init. The launcher prints that line
# exactly once per invocation, immediately before exec, so it is the precise
# start of THIS boot.
tail -c "+$((LOG_OFFSET + 1))" "$LOG" > /tmp/arm-slice-$$.log
if grep -qa "^A/B config:" /tmp/arm-slice-$$.log; then
  awk '/^A\/B config:/{n=NR} {a[NR]=$0} END{for(i=n;i<=NR;i++) print a[i]}' \
    /tmp/arm-slice-$$.log > /tmp/arm-boot-$$.log
else
  cp /tmp/arm-slice-$$.log /tmp/arm-boot-$$.log
fi
bash "$ROOT/agent-services/bin/flash-next-bootfacts.sh" /tmp/arm-boot-$$.log

# An arm that crashed and was resurrected by supervisord must never be
# benchmarked. The arm env is one-shot BY DESIGN (see the launcher), so an
# autorestart comes back on production defaults — and then the bench runs
# happily against the wrong config and reports baseline numbers under this
# arm's name. That is exactly what MOE_BACKEND=flashinfer_b12x did on
# 2026-09-08: it died in memory profiling with an illegal memory access,
# supervisord restarted it on defaults, and the arm "measured" 120.7 tok/s.
# Two independent tells, because either alone can be argued with.
BOOTS=$(grep -ac "Initializing a V1 LLM engine" /tmp/arm-boot-$$.log)
if [[ "$BOOTS" -ne 1 ]]; then
  echo "!! ABORT $LABEL: engine initialised $BOOTS times in this window."
  echo "   It crashed and supervisord restarted it, so the live config is NOT this arm."
  grep -a -m3 -E "Worker failed with error|EngineCore failed to start|AcceleratorError" /tmp/arm-boot-$$.log | cut -c1-200
  exit 2
fi
if grep -qa -E "EngineCore failed to start|Worker failed with error" /tmp/arm-boot-$$.log; then
  echo "!! ABORT $LABEL: engine reported a startup failure in this window."
  exit 2
fi
echo "boot guard: 1 engine init, no startup failures — this arm is what is serving."

# SKIP_BENCH=1 stops here with the arm serving. bench-flash-next.py measures
# decode and defeats the prefix cache, so it has nothing to say about an arm
# whose question is admission — the Layer 3 max_num_batched_tokens sweep
# (architecture/vllm.md) drives its own reproducer.
if [[ "${SKIP_BENCH:-0}" == "1" ]]; then
  echo "SKIP_BENCH=1 — arm $LABEL is serving; not benchmarking"
  echo "=== arm $LABEL done ==="
  exit 0
fi

echo "--- benchmarking ---"
"$ROOT/.venvs/lloyd/bin/python" "$ROOT/agent-services/bin/bench-flash-next.py" \
  "$LABEL" --out "$RESULTS"

echo "=== arm $LABEL done ==="
