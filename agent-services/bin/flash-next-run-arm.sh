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

# --- engine-output integrity canary (#1268 owed → #1625; persisted by #2163) -
# Everything above reads CONFIG: bootfacts greps the KV dtype and the pool size
# off the boot log, the boot guard counts engine inits. Neither one reads a
# token. AI21's two vLLM bugs — both in the Mamba state cache this hybrid runs on
# — produced confident WRONG output with no crash, no warning and no error line,
# so an arm could clear every check above and still be serving corrupted output
# under a config that looks identical in the log. This replays the committed
# 21-prompt corpus at temperature 0 against the engine that actually booted.
#
# NON-FATAL BY DESIGN, which is why both statuses are captured instead of being
# allowed to stand alone: exit 1 is "output moved past the floor" — a finding
# ABOUT the engine, which is what the bench is about to measure — and exit 2 is
# "the instrument could not decide" (engine busy on a probe-only arm, no
# logprobs, floor or reference file missing). Either way the arm prints a verdict
# and continues. A canary that aborts an arm makes the sweep depend on the
# instrument more than on the engine, and aborting on a busy port would cost the
# sweep its measurement while proving nothing about the build.
#
# The reference is NAMED, never defaulted: `compare --reference` defaults to "the
# newest other record in the current's directory", and every arm's record lands
# in one directory — so a defaulted sweep would grade arm N against arm N-1 and
# drift across a sweep would be invisible by construction. The named reference is
# one of the five committed idle repeats: the arm's replay is un-salted like
# those, and the tiered floor is leave-one-out clean at 0 of 90 on exactly these
# records. Override ENGINE_OUTPUT_REF after a rebaseline.
ENGINE_OUTPUT_PROBE="${ENGINE_OUTPUT_PROBE:-$ROOT/eval/engine_output_probe.py}"
ENGINE_OUTPUT_REF="${ENGINE_OUTPUT_REF:-$ROOT/eval/engine_output/idle/20260924T211316Z_idle-5.json}"
# THE VERDICT IS PERSISTED, not just echoed (#2163). What leaves this window has
# to be decided before the window closes, and the window is what goes away: an arm
# runs inside a primary restart, and `SKIP_BENCH=1` — how an admission sweep runs,
# since bench-flash-next.py measures decode and defeats the prefix cache — exits
# below, before the bench's `--out` write. So every canary decision this script
# ever printed went into scrollback and nothing else: no
# `~/lloyd-data/eval/baselines/engine_output/`, no `flash-next-arms.jsonl`, no
# `flash-next-arm.env`, and no `engine-output canary:` line anywhere under
# ~/lloyd-data/logs/, since the canary was wired in in dfb18646 (#1625, 2026-09-28).
# One JSONL line per arm, in the SAME runtime directory the bench results already
# use above ($RESULTS), so one `tail` shows a sweep's throughput and its integrity
# decisions side by side.
CANARY_LOG="${LLOYD_DATA:-$HOME/lloyd-data}/logs/services/flash-next-canary.jsonl"
# Per-process payload file, like the boot slices above. `compare` writes it only
# when it decides, so the file being ABSENT is itself the could-not-decide signal
# — a stale or empty one would read as a comparator that said 0 diverged.
CANARY_CMP_JSON="${TMPDIR:-/tmp}/flash-next-canary-$$.json"
# Each status is captured on the call itself (`|| RC=$?`) rather than by reading
# `$?` on the next line: that form aborts under `set -e`, and this script's header
# could gain the flag one day the way every other script's did. Same reason the
# record path is pulled out with one awk process instead of `sed | head -1` —
# under `pipefail` a `head` that closes the pipe early leaves the pipeline 141.
CANARY_RUN_RC=0
CANARY_RUN_OUT=$("$ROOT/.venvs/lloyd/bin/python" "$ENGINE_OUTPUT_PROBE" run \
                   --label "arm-$LABEL" 2>&1) || CANARY_RUN_RC=$?
CANARY_RECORD=$(printf '%s\n' "$CANARY_RUN_OUT" | awk '/^wrote /{print $2; exit}')
# Decision, stage and the instrument's last line are recorded, not just printed,
# because the line below has to say WHICH of the three outcomes happened and why.
# `CANARY_CMP_RC` stays empty when compare was never reached: an arm whose engine
# refused the corpus has no comparator decision, and writing 0 there would be the
# false clean pass this canary exists to prevent.
CANARY_DECISION=
CANARY_STAGE=
CANARY_LAST_OUTPUT=
CANARY_CMP_RC=
if [ "$CANARY_RUN_RC" -ne 0 ] || [ -z "$CANARY_RECORD" ]; then
  # `run` exits 2 when the engine refused the corpus, and prints no `wrote` line
  # when it wrote nothing. Neither is a verdict about the engine's output, so it
  # reports as the instrument not deciding rather than as a clean pass.
  CANARY_DECISION=could-not-decide
  CANARY_STAGE=run
  CANARY_LAST_OUTPUT=$(printf '%s\n' "$CANARY_RUN_OUT" | tail -1)
  echo "engine-output canary: arm $LABEL could-not-decide (run rc=$CANARY_RUN_RC, $CANARY_LAST_OUTPUT)"
else
  CANARY_CMP_RC=0
  CANARY_CMP_OUT=$("$ROOT/.venvs/lloyd/bin/python" "$ENGINE_OUTPUT_PROBE" compare \
                     --current "$CANARY_RECORD" --reference "$ENGINE_OUTPUT_REF" \
                     --json-out "$CANARY_CMP_JSON" 2>&1) \
    || CANARY_CMP_RC=$?
  CANARY_STAGE=compare
  CANARY_LAST_OUTPUT=$(printf '%s\n' "$CANARY_CMP_OUT" | tail -1)
  case "$CANARY_CMP_RC" in
    0) CANARY_DECISION=within-floor
       echo "engine-output canary: arm $LABEL within-floor ($CANARY_RECORD vs $ENGINE_OUTPUT_REF)" ;;
    1) CANARY_DECISION=past-floor
       echo "engine-output canary: arm $LABEL DIVERGED from $ENGINE_OUTPUT_REF: $(printf '%s\n' "$CANARY_CMP_OUT" | grep -m1 'past their .* floor')"
       printf '%s\n' "$CANARY_CMP_OUT" | sed 's/^/  canary: /' ;;
    *) CANARY_DECISION=could-not-decide
       echo "engine-output canary: arm $LABEL could-not-decide (compare rc=$CANARY_CMP_RC, $CANARY_LAST_OUTPUT)" ;;
  esac
fi

# The verdict is appended HERE, above the SKIP_BENCH exit, by the same rule that
# keeps the echoes non-fatal: an arm may not be lost to its own instrument. Here
# that rule has a second reason — this is the LAST point in the arm where a
# verdict can still be written on the canary-only route, so writing it after the
# bench (or after the exit) is what loses a sweep's integrity record today. A
# persistence failure is reported on stdout and swallowed: the engine is already
# booted, the bench has not run, and a lost line is worth less than a lost arm.
export CANARY_LABEL="$LABEL" CANARY_LOG CANARY_CMP_JSON CANARY_REF="$ENGINE_OUTPUT_REF" \
       CANARY_DECISION CANARY_STAGE CANARY_RUN_RC CANARY_CMP_RC CANARY_RECORD \
       CANARY_LAST_OUTPUT
CANARY_PERSIST_RC=0
python3 - <<'PY' || CANARY_PERSIST_RC=$?
import datetime, json, os


def _rc(value):
    """Empty means the call was never reached, which is not the same number as 0."""
    return int(value) if value else None


def _worst(rows):
    """compare_records returns its rows sorted worst first, so rows[0] is the
    prompt the comparator itself put on top of the table the operator just read.
    Carried with its reasons: a count without a prompt id and a first_divergence
    index is an alarm with nothing to act on."""
    if not rows:
        return None
    r = rows[0]
    return {"prompt": r.get("id"), "agreement": r.get("agreement"),
            "first_divergence": r.get("first_divergence"),
            "token_lp_delta": r.get("token_lp_delta"), "reasons": r.get("reasons")}


payload = None
try:
    # Only `compare` on a decision it made writes this file, so an absent or
    # unreadable one means the comparator never decided — never a clean 0.
    with open(os.environ["CANARY_CMP_JSON"], encoding="utf-8") as fh:
        payload = json.load(fh)
except (OSError, ValueError):
    pass
payload = payload or {}
os.makedirs(os.path.dirname(os.environ["CANARY_LOG"]), exist_ok=True)
with open(os.environ["CANARY_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "arm": os.environ.get("CANARY_LABEL", ""),
        "decision": os.environ.get("CANARY_DECISION", ""),
        "stage": os.environ.get("CANARY_STAGE", ""),
        "run_rc": _rc(os.environ.get("CANARY_RUN_RC", "")),
        "compare_rc": _rc(os.environ.get("CANARY_CMP_RC", "")),
        "reference": os.environ.get("CANARY_REF", ""),
        "record": os.environ.get("CANARY_RECORD") or None,
        "diverged": payload.get("diverged"),
        "worst_prompt": _worst(payload.get("rows")),
        "last_output": os.environ.get("CANARY_LAST_OUTPUT", ""),
    }) + "\n")
PY
if [ "$CANARY_PERSIST_RC" -ne 0 ]; then
  echo "engine-output canary: arm $LABEL verdict NOT persisted to $CANARY_LOG (rc=$CANARY_PERSIST_RC)"
fi

# --- forced-preemption load (#2162, #1268 owed 2) -----------------------------
#
# Its own executable, called unconditionally, because the assertions about what it
# hands the probe must not spend four minutes restarting the primary to be made —
# see its header for why an arm window is the ONLY valid caller (the probe refuses
# a non-idle engine outright, and `vllm:num_preemptions_total` is a per-boot
# counter, so the reboot above is what makes `before` zero) and for why every
# outcome it prints is a note. It gates itself: `PREEMPT_ARM=1` on this script's
# own command line arms it, and `PREEMPT_LOAD_PROMPT_WORDS` is its only size knob
# (exported vars reach it the way they reach every child here). Unset, it exits 0
# having done nothing, so an ordinary sweep does not see it at all.
#
# POSITION IS LOAD-BEARING IN BOTH DIRECTIONS. Below the boot guard: an arm whose
# engine crashed and was resurrected on production defaults must not drive a load
# on its way out, and both of the guard's `exit 2`s are above this line. Below the
# canary verdict persistence: the integrity decision is on disk whatever the load
# then does. And above the `SKIP_BENCH=1` exit, because bench-flash-next.py
# measures decode while the arm that could reach a preemption is an admission arm
# that skips the bench — an arm which must still get its load step.
#
# Run with `bash <path>`: this script is itself run that way (mode 644 in git,
# unusual among its bin/ siblings, and tests/test_flash_next_launcher.py invokes it
# as `bash <path>` too), and an interpreter-prefixed call does not care what the
# exec bit on the target says on any of those routes.
#
# The `||` is what keeps this a step and not a stage. The step always exits 0, so
# the only status that can arrive here is the shell failing to find the file at all
# — a tree that predates it, a checkout mid-copy, or a harness slicing this script
# around a tree with no bins (tests/test_flash_next_launcher.py does exactly that).
# This script runs `set -uo pipefail`, and a caller reads its rc as the arm's
# verdict, so an unguarded call would let "the step wasn't there" stand for "the
# arm failed" — landing last above the `SKIP_BENCH` exit, it would BE the script's
# rc. The step not running is a note; the arm's own stages decide its exit code.
#
# Read the arm's LOG, not its rc, for what the load did: because the step always
# exits 0, a load that crossed, one that did not, and one that never ran differ only
# in the line each prints. `preemption load:` in the arm log is the whole trail.
bash "$ROOT/agent-services/bin/flash-next-preempt-step.sh" "$LABEL" \
  || echo "preemption load: step did not run (rc=$?); non-fatal, continuing to bench"

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
