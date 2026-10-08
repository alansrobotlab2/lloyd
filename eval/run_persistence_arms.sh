#!/usr/bin/env bash
# eval/run_persistence_arms.sh N — open the persistence-arm window (#2397).
#
#   eval/run_persistence_arms.sh 10
#   LLOYD_CANARY_ROWS=$HOME/tmp/win/rows.jsonl LLOYD_CANARY_REPORT=$HOME/tmp/win/run.md \
#     eval/run_persistence_arms.sh 2
#
# The window is N reps of #2041's three persistence arms with worker dispatch HELD, then
# `grade`, so a payload that survived compaction can be attributed to the summariser
# rather than to whatever job ran beside it. That table is what #2194's owed entries and
# the deploy gate's leak-rate denominator are waiting on: 3 rows at rep:1, unchanged
# since 2026-10-04, is not a denominator.
#
# Why this file is thin, and why there is no `trap ... EXIT INT TERM` in it: the hold it
# needs already exists, hardened — `eval/run_context_rot_eval.py::PoolPause`, reused
# through `eval/persistence_arms.py`. It resumes only a pause it took, and it refuses to
# lift one the automod promoter holds, because resuming as OPERATOR lifts both holders
# (`workers/pool.py:716-728`) and an EXIT trap that un-pauses a landing mid-drain takes
# the restart's protection off with it. A `curl`-and-trap wrapper here would re-derive
# all of that worse, so the wrapper validates N and execs the driver: `exec` keeps the
# signal going to the process that owns the cleanup instead of to a shell that would
# only look less leaky. The leak check itself runs after the event loop is gone — the
# driver reads `(_pool,operator_paused)` out of `workers.db` and exits non-zero if the
# hold it took is still set.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: ${0##*/} N [--backend URL] [--drain-wait S] [--automod-wait S]" >&2
  echo "  arms: persistence-web-digest persistence-relay-email persistence-control-handover" >&2
  exit 2
fi

N="$1"
if ! [[ "$N" =~ ^[0-9]+$ ]] || (( 10#$N < 1 )); then
  echo "${0##*/}: N must be a positive integer of reps, got '$N'" >&2
  exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Which interpreter runs the driver. Default is this checkout's venv; the variable is
# for a test suite that is already running inside one and must not boot a second, and
# for the round worktree, where `.venvs/` is gitignored and so does not exist.
PYTHON="${LLOYD_CANARY_PYTHON:-$ROOT/.venvs/lloyd/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "${0##*/}: no interpreter at $PYTHON (set LLOYD_CANARY_PYTHON)" >&2
  exit 2
fi

exec "$PYTHON" "$ROOT/eval/persistence_arms.py" "$@"
