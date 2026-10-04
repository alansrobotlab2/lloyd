#!/usr/bin/env bash
# #2162 (#1268 owed 2): drive the forced-preemption load inside a flash-next arm
# window — and only there.
#
#   agent-services/bin/flash-next-preempt-step.sh <arm-label>
#
# Called unconditionally by `flash-next-run-arm.sh` after its boot guard and the
# #1625 engine-output canary. THE GATE IS THIS FILE'S OWN: with `PREEMPT_ARM`
# unset or not `1` it exits 0 without saying anything, because the arm route
# calls it on every arm and a chatty default would print into every sweep. Export
# `PREEMPT_ARM=1` and it drives
#
#   $ROOT/.venvs/lloyd/bin/python $ROOT/eval/engine_output_probe.py preempt \
#       --out-dir <dir> --label arm-<label> --load-prompt-words <N>
#
# exactly once, through the same interpreter the canary block above uses.
#
# WHY AN ARM WINDOW IS THE ONLY VALID CALLER. `preempt` fires `--load-requests`
# (8 by default) concurrent prompts of `--load-prompt-words` words each — not a
# measurement to run beside a serving box. The driver already refuses that:
# `eval/engine_output_probe.py:636-641` raises `ProbeRefused` and exits 2 without
# sending one load request unless the engine reports `num_requests_running` AND
# `num_requests_waiting` both zero. A session answering on the primary keeps that
# counter non-zero, which is the state the guard protects, so this step talks the
# driver out of nothing: it passes no flag, no argument and no environment
# variable that reaches that check, and it exports nothing to the child at all.
#
# The window is also why a non-zero record is reachable at all.
# `vllm:num_preemptions_total` is a PER-BOOT counter: the arm route stopped and
# rebooted the engine above, so `num_preemptions_before` is 0 here and the first
# eviction is a move (`after > before`). The one run that tried elsewhere —
# `eval/engine_output/preempt/preempt_2026-09-25T004044+0000_smoke.json`, a warm
# engine — recorded `2.0 -> 2.0`, `preemptions_reached: false`, peak kv 0.157.
# That is the record this step exists to beat.
#
# Always exiting 0 is also what lets the arm script call this with a `||` note —
# and it has to, because the arm runs `set -uo pipefail` and its last statement
# above the `SKIP_BENCH` exit IS its rc. The arm runs this file as `bash <path>`,
# so the exec bit is not a route to a silent 126, and the only status left for a
# script that always exits 0 is the shell failing to find it at all: a checkout
# caught mid-copy, a tree that predates this file, or a test root with no bins.
# Unguarded, that absence reported itself as rc 127 = "the arm failed", about an
# arm whose engine was up and serving. The guard is in the caller; this file's job
# is to never be the reason an arm's verdict is wrong.
#
# SIZE ESCALATES BY PROMPT LENGTH ONLY. `PREEMPT_LOAD_PROMPT_WORDS` (default
# 20000, the probe's own) is the knob. Never `MAX_NUM_SEQS`, never
# `KV_CACHE_MEMORY_BYTES`: the first is the 8 sequences sharing the pool, the
# second IS the pool (15032385536 bytes = 844,969 tokens), and a run that
# reached a preemption against a shrunk pool would be measuring a pool no
# operator armed. The probe records `peak kv_cache_usage_perc` on both outcomes
# and this step echoes it either way, so a run that stayed under the line is
# still the measurement that says so.
#
# EVERY OUTCOME IS A NOTE, NOT AN ABORT. Reached, not-reached, the rc=2 refusal
# and an instrument failure each print exactly one line beginning `preemption
# load:` and the step exits 0, so the arm still reaches `bench-flash-next.py`
# with its rc unchanged. An arm is the operator's measurement of the engine; a
# load step that could cancel it would make the sweep depend on this instrument
# more than on the thing being measured — the same reasoning the canary above
# gives for its own non-fatal design.
set -uo pipefail

LABEL="${1:?usage: flash-next-preempt-step.sh <arm-label>}"

# ARM_TEST_ROOT is a seam for tests/test_flash_next_arm_preempt.py alone, which
# runs this script against a stub tree so no assertion of it ever launches a
# load at a real engine. Its default is the tree this file lives in, derived the
# way the arm route derives its own ROOT.
ROOT="${ARM_TEST_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROBE="${ENGINE_OUTPUT_PROBE:-$ROOT/eval/engine_output_probe.py}"

# The record goes to the arm's runtime directory, NOT into the repo: the step
# must have no route for dirtying a tracked path, and the committed copy under
# `eval/engine_output/preempt/` is made by the operator promoting a record that
# shows a move. `PREEMPT_DIR` is the second seam for the same test file.
OUT_DIR="${PREEMPT_DIR:-${LLOYD_DATA:-$HOME/lloyd-data}/logs/services/flash-next-arm/preempt}"
LOAD_PROMPT_WORDS="${PREEMPT_LOAD_PROMPT_WORDS:-20000}"   # the probe's own default

if [[ "${PREEMPT_ARM:-0}" != "1" ]]; then
  exit 0
fi

echo "--- forced-preemption load: PREEMPT_ARM=1, ${LOAD_PROMPT_WORDS}-word prompts, out-dir $OUT_DIR ---"

PREEMPT_RC=0
# `|| PREEMPT_RC=$?` on the call, not `$?` on the next line: the latter aborts
# under a `set -e` this header could one day gain. Combined 2>&1 because the
# refusal sentence goes to stderr and the note below has to quote it.
PREEMPT_OUT=$("$ROOT/.venvs/lloyd/bin/python" "$PROBE" preempt \
                --out-dir "$OUT_DIR" --label "arm-$LABEL" \
                --load-prompt-words "$LOAD_PROMPT_WORDS" 2>&1) || PREEMPT_RC=$?

# The probe states its result on one line —
#   wrote <path>: preemptions <before> -> <after>, preemptions_reached=<bool>, peak kv <0.fff>
# — and one awk process pulls each field out of it. One process per field, not
# `grep | head`: under `pipefail` a `head` that closes the pipe early leaves the
# pipeline 141, and this step reports numbers the operator will read as fact.
PREEMPT_RECORD=$(printf '%s\n' "$PREEMPT_OUT" | awk '/^wrote /{p=$2; sub(/:$/,"",p); print p; exit}')
PREEMPT_MOVE=$(printf '%s\n' "$PREEMPT_OUT" | awk '/^wrote /{for(i=1;i<=NF;i++) if($i=="->"){print $(i-1)" -> "$(i+1); exit}}')
PREEMPT_REACHED=$(printf '%s\n' "$PREEMPT_OUT" | awk '/preemptions_reached=true/{print "true"; exit}')
PREEMPT_PEAK_KV=$(printf '%s\n' "$PREEMPT_OUT" | awk '/peak kv/{print $NF; exit}')
PREEMPT_REFUSAL=$(printf '%s\n' "$PREEMPT_OUT" | awk '/cannot decide/{sub(/.*cannot decide — /,""); print; exit}')
PREEMPT_REFUSAL="${PREEMPT_REFUSAL:-the probe printed no reason; engine busy is the likeliest}"

case "$PREEMPT_RC" in
  0)
    if [ "$PREEMPT_REACHED" = "true" ]; then
      # The record #1268 owed 2 has been asking for. It is only evidence once a
      # person has read its load block — `ok: true` with `prompt_tokens` near
      # target and an empty `probe_error` — so the note says where it is and
      # who decides, and commits nothing itself.
      echo "preemption load: arm $LABEL reached a preemption, preemptions ${PREEMPT_MOVE:-?} (peak kv ${PREEMPT_PEAK_KV:-?}) — read its load block, then commit ${PREEMPT_RECORD:-<no path parsed>} into eval/engine_output/preempt/"
    else
      # Not crossing the line is a RESULT, which is how the probe itself exits
      # (0, with `preemptions_reached=false`), and the rail that keeps that
      # reading honest is `test_preempt_not_reached_is_a_stated_result` in
      # tests/test_engine_output_probe.py. What this line owes the operator is
      # which way it went, the numbers, and where to escalate.
      echo "preemption load: arm $LABEL did not cross the line — preemptions ${PREEMPT_MOVE:-?}, peak kv ${PREEMPT_PEAK_KV:-?}, record ${PREEMPT_RECORD:-<no path parsed>}; stated result, not an error. Escalate PREEMPT_LOAD_PROMPT_WORDS on the next arm."
    fi ;;
  2)
    # The driver's own refusal: `ProbeRefused`, raised before it sends a request
    # when the engine is not idle. Nothing here works around it — the arm
    # continues, and a busy engine means this window was not the window.
    echo "preemption load: arm $LABEL did not run (rc=2 — the probe refused it: ${PREEMPT_REFUSAL}); non-fatal, continuing to bench with the engine as it is"
    ;;
  *)
    echo "preemption load: arm $LABEL instrument failed (rc=$PREEMPT_RC); non-fatal, continuing to bench"
    ;;
esac

exit 0
