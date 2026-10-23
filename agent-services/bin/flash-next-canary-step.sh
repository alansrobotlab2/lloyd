#!/usr/bin/env bash
# The engine-output integrity canary, as one executable with one writer of its
# verdict file (#1625 persisted by #2163, lifted out of the arm route by #2404).
#
#   agent-services/bin/flash-next-canary-step.sh <arm-label>
#
# TWO CALLERS, ONE COPY. The sweep arm route runs it on every arm, below its boot
# guard and above its `SKIP_BENCH` exit; `eval/run_kv_dtype_arm.sh` runs it once
# inside a one-shot KV-dtype window, below the gated boot that proves the arm dtype
# is serving and above the restore that ends the window. A second copy of this fold
# in that second runner is how `flash-next-canary.jsonl` would end up with two
# writers and two record shapes, so the fold lives here and both routes call it.
#
# WHY A TOKEN CANARY AT ALL, and why its verdict is persisted rather than echoed, is
# the block below — one file, read by `tests/test_flash_next_launcher.py`, which
# lifts it out between the two markers and runs it under real bash against a stubbed
# probe. Do not reword those markers or the block's shape without retargeting it.
#
# ALWAYS EXITS 0, for the same reason `flash-next-preempt-step.sh` does: both callers
# read their own rc as the arm's verdict, and an arm may not be lost to its own
# instrument. `ENGINE_OUTPUT_PROBE` / `ENGINE_OUTPUT_REF` are its seams and
# `ENGINE_TOOL_PROBE` (#3188) is the tool-call leg's — a second leg the prose probe cannot be,
# because /v1/completions has no `tools` field and the prose corpus, floor and reference files
# are the measuring stick the new leg must leave byte-unchanged. `ARM_TEST_ROOT` points `$ROOT`
# at a fake tree for a test, and it governs both legs' default probe paths.
set -uo pipefail

LABEL="${1:?usage: flash-next-canary-step.sh <arm-label>}"

ROOT="${ARM_TEST_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

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
ENGINE_OUTPUT_REF="${ENGINE_OUTPUT_REF:-$ROOT/eval/engine_output/idle/20261012T041722Z_idle1012-5.json}"

# --- #3188: the tool-call leg. Its own seam, resolving under the SAME $ROOT. --------------
# The prose leg above can only ever measure prose tokens: it posts {prompt, temperature:0,
# logprobs:1} to /v1/completions (`eval/engine_output_probe.py:330-331`), and that endpoint has
# no `tools` field. So `eval/action_equivalence.py` (#2843, landed e7e402d5) has had no caller
# outside its own test, and the quant/KV-dtype decision question — do the two dtypes make the
# SAME tool call? — has never been asked of a live arm. This fold's own traffic says so:
# `~/lloyd-data/logs/services/flash-next-canary.jsonl` ends at 2026-10-12T23:16:34+00:00, arm
# `kv-bfloat16`, four days before the comparator landed, and its verdict is prose.
#
# A separate module and a separate seam, not a flag on the prose probe: that probe's `probe_one`
# (:328) and `parse_completion` (:294-315, reading the legacy logprobs.tokens shape) are
# /v1/completions code and cannot carry a `tools` array — and its corpus, floor and reference
# files are this window's measuring stick, which #3188's clause 1 requires stay byte-unchanged.
# The decision corpus is therefore `eval/tool_call_corpus.yaml`, never a 22nd prompt in
# `eval/engine_output/corpus.yaml`.
#
# Neither leg is fatal, but they degrade differently, and that difference is why this one is a
# separate leg rather than a flag: a missing prose probe takes the WHOLE canary verdict down to
# `could-not-decide`, while a missing tool probe costs one line — the prose verdict and the
# appended row are untouched, and an arm is still never lost to it (#3109's rule, unchanged).
#
# Both legs run through the SAME interpreter as the prose leg, `$ROOT/.venvs/lloyd/bin/python`,
# and the default path resolves under $ROOT — so ARM_TEST_ROOT governs this leg exactly as it
# governs the prose one (the launcher test's fake root pins it). Neither leg is handed a capture
# path: the probe derives its own from `$LLOYD_DATA/logs/services`, the same env this script
# already uses for $CANARY_LOG, so the arm's two files can never disagree about where its bytes
# are, and a test that sets LLOYD_DATA moves both.
ENGINE_TOOL_PROBE="${ENGINE_TOOL_PROBE:-$ROOT/eval/tool_call_equivalence_probe.py}"

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
# The payload is scratch, and scratch is removed on every exit path (#2776). It sat
# unremoved from the first arm run: one file per process id, 821 bytes each, 1,016 of
# them in `/tmp` on 2026-10-15 when that tmpfs was at 81% of its inode budget, and
# nothing in `scripts/groundskeeper/retention-sweep.py` names the pattern — only a
# manual inode sweep ever deleted them, and the count was back inside a day. Most of
# that population came from this repo's own test harness, whose replaced env carried no
# `TMPDIR`, so `${TMPDIR:-/tmp}` resolved to the real `/tmp` on every pytest run.
#
# On EXIT, not inline below the persist block, because the paths that never reach a
# verdict have to clean up too, and an EXIT trap is the only form that covers the
# `set -e` abort as well as the plain `exit 0`. It is `-f` rather than plain `rm` for
# the same reason the statuses above are captured on the call: on the path where
# `run` refused the corpus, `compare` never ran and the file was never written, and a
# plain `rm` there exits 1, which under `set -e` turns a clean could-not-decide arm
# into a failed one — an instrument that costs the arm is exactly what this fold's
# non-fatal rule forbids. Removing it cannot weaken the `absent = could-not-decide`
# signal above: the ONE reader of this path is the heredoc below, which runs before
# the trap does, so a comparator that decided still persists its `diverged` count and
# worst prompt and one that refused still sees no file and persists `diverged: null`.
# And no later arm can inherit a stale payload: the path is per-`$$`, and both callers
# (`flash-next-run-arm.sh`, `eval/kv_dtype_arm.py`) read only the JSONL, never this
# file — `tests/test_flash_next_launcher.py`'s #2776 nodes pin all of it under bash.
trap 'rm -f -- "$CANARY_CMP_JSON"' EXIT
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

# --- #3188: the tool-call leg, after the prose leg, non-fatal on its exact contract ------
# Two subcommands, because one window can only record its own arm and the comparison needs the
# other arm's file to exist too: `capture` appends THIS arm's decisions to
# `$LLOYD_DATA/logs/services/flash-next-canary-toolcalls-<arm>.jsonl`, `report` re-reads every
# arm's capture inside the same window and prints the one per-tier line from
# `eval/action_equivalence.classify`, naming each `args_differ` pair's prompt id and tool name.
#
# Each leg surfaces its probe's canonical line WHATEVER its exit code — exit 1 is a finding
# about the engine and exit 2 the instrument could not decide, the same meanings the prose leg
# above uses — so neither can be dropped silently: a leg that printed no canonical line is a leg
# that crashed or was missing, and the fallback says so with the code. Both codes are then
# DROPPED, here and only here: #3109's rule, no arm is lost to its instrument, and the tool leg
# can no more cancel a gated restart than the prose one can.
TOOL_CAPTURE_RC=0
# No `--base-url` on either leg, which is the point rather than an oversight: both probes
# default to the same `http://127.0.0.1:8096` (`engine_output_probe.py:98` = this module's
# :121), so the two halves of one arm's verdict cannot come from two different processes,
# and `set -u` at :26 makes an `ENGINE_URL` this fold never sets a fatal error, not a
# default — a variable named here would KILL the arm, which is clause 3's exact prohibition.
# A test that wants another engine points the SEAM at a wrapper; that is what
# `test_the_fold_drives_the_real_probe_end_to_end_through_the_seam` does.
TOOL_CAPTURE_OUT=$("$ROOT/.venvs/lloyd/bin/python" "$ENGINE_TOOL_PROBE" capture --label "$LABEL" 2>&1) \
  || TOOL_CAPTURE_RC=$?
# `|| true` inside the braces, not after the pipeline: the launcher harness runs this block
# under `set -euo pipefail`, and a grep that matches nothing would otherwise abort the arm at
# its own report line. This is NOT the masking the warning above warns about — the consumer can
# tell: an empty result takes the fallback branch, which reports the exit code it was given.
TOOL_CAPTURE_LINE=$(printf '%s\n' "$TOOL_CAPTURE_OUT" | { grep '^tool-call capture: ' || true; } | tail -n 1)
[ -n "$TOOL_CAPTURE_LINE" ] || TOOL_CAPTURE_LINE="tool-call capture: probe-failed(exit $TOOL_CAPTURE_RC) $(printf '%s\n' "$TOOL_CAPTURE_OUT" | tail -n 1)"
TOOL_REPORT_RC=0
TOOL_REPORT_OUT=$("$ROOT/.venvs/lloyd/bin/python" "$ENGINE_TOOL_PROBE" report --label "$LABEL" 2>&1) \
  || TOOL_REPORT_RC=$?
TOOL_LINE=$(printf '%s\n' "$TOOL_REPORT_OUT" | { grep '^tool-equivalence: ' || true; } | tail -n 1)
[ -n "$TOOL_LINE" ] || TOOL_LINE="tool-equivalence: probe-failed(exit $TOOL_REPORT_RC) $(printf '%s\n' "$TOOL_REPORT_OUT" | tail -n 1)"
echo "$TOOL_CAPTURE_LINE"
echo "$TOOL_LINE"

# The verdict is appended HERE, above the SKIP_BENCH exit, by the same rule that
# keeps the echoes non-fatal: an arm may not be lost to its own instrument. Here
# that rule has a second reason — this is the LAST point in the arm where a
# verdict can still be written on the canary-only route, so writing it after the
# bench (or after the exit) is what loses a sweep's integrity record today. A
# persistence failure is reported on stdout and swallowed: the engine is already
# booted, the bench has not run, and a lost line is worth less than a lost arm.
export CANARY_LABEL="$LABEL" CANARY_LOG CANARY_CMP_JSON CANARY_REF="$ENGINE_OUTPUT_REF" \
       CANARY_DECISION CANARY_STAGE CANARY_RUN_RC CANARY_CMP_RC CANARY_RECORD \
       CANARY_LAST_OUTPUT \
       CANARY_TOOL_LINE="$TOOL_LINE" CANARY_TOOL_CAPTURE_RC="$TOOL_CAPTURE_RC" \
       CANARY_TOOL_REPORT_RC="$TOOL_REPORT_RC"
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
        # #3188's tool-call leg: the line it printed and the two codes it exited with. A
        # `None` rc means that leg was never reached — a different fact from a leg that ran
        # and said it could not decide, which is rc 2 with a line saying so.
        "tool_line": os.environ.get("CANARY_TOOL_LINE", ""),
        "tool_capture_rc": _rc(os.environ.get("CANARY_TOOL_CAPTURE_RC", "")),
        "tool_report_rc": _rc(os.environ.get("CANARY_TOOL_REPORT_RC", "")),
    }) + "\n")
PY
if [ "$CANARY_PERSIST_RC" -ne 0 ]; then
  echo "engine-output canary: arm $LABEL verdict NOT persisted to $CANARY_LOG (rc=$CANARY_PERSIST_RC)"
fi

# --- end of the engine-output integrity canary fold --------------------------
# Everything above the line is the block `tests/test_flash_next_launcher.py`
# extracts and executes; `exit 0` is below it so the extraction stays exactly the
# fold and nothing else.
exit 0
