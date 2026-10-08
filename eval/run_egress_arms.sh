#!/usr/bin/env bash
# eval/run_egress_arms.sh N --on-port PORT — open the egress-arms window (#2435).
#
#   eval/run_egress_arms.sh 1 --on-port 8599
#   LLOYD_CANARY_ROWS=$HOME/tmp/egress-arms/rows.jsonl \
#     eval/run_egress_arms.sh 1 --on-port 8599 --arms enforce-off
#
# N reps of the shipped canary set per arm: the enforce-OFF arm served by the shared
# aggregator, the enforce-ON arm served by a private `LLOYD_EGRESS_ENFORCE=1`
# `python -m agent_mcp.main` this window starts on PORT with `LLOYD_DATA` pointed at a
# run-local root and stops on teardown. That is the A/B #2338 landed the honest arm label
# for and nobody has since taken: every row in
# `eval/measurements/injection-canary/rows.jsonl` still comes from one arm, because the
# shared daemon reads `enforce: false` and says so over `/state`.
#
# Why this file is thin, and why there is no `trap ... EXIT INT TERM` in it: the hold and
# the child both need cleanup that cannot leak, and both already exist, hardened —
# `eval/persistence_arms.py`'s `PersistenceHold` (refuses an already-held pool, resumes
# only a pause it took, refuses to lift one the automod promoter owns) and
# `eval/egress_arms.py`'s teardown of the aggregator it launched. An EXIT trap that
# `curl`s an operator resume lifts BOTH holders (`workers/pool.py:716-728`), so it would
# un-pause a landing's drain mid-flight; and a wrapper that backgrounded the aggregator
# could stop it on the happy path only, leaving a daemon enforcing on a spare port. So
# the wrapper validates N and `exec`s the driver: `exec` sends the signal to the process
# that owns the cleanup rather than to a shell that only looks less leaky. The live-store
# leak check and the persisted-hold re-check both run after the event loop is gone.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: ${0##*/} N --on-port PORT [--arms enforce-off|enforce-on ...] [--mcp-url URL] [--backend URL] [--run-root DIR]" >&2
  echo "  one rep = the whole shipped scenario set, run once per arm" >&2
  exit 2
fi

N="$1"
if ! [[ "$N" =~ ^[0-9]+$ ]] || (( 10#$N < 1 )); then
  echo "${0##*/}: N must be a positive integer of reps, got '$N'" >&2
  exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Which interpreter runs the driver. Default is this checkout's venv; the variables are
# for a test suite already running inside one, which must not boot a second, and for the
# round worktree, where `.venvs/` is gitignored and so does not exist.
PYTHON="${LLOYD_EGRESS_ARMS_PYTHON:-${LLOYD_CANARY_PYTHON:-$ROOT/.venvs/lloyd/bin/python}}"
if [[ ! -x "$PYTHON" ]]; then
  echo "${0##*/}: no interpreter at $PYTHON (set LLOYD_EGRESS_ARMS_PYTHON)" >&2
  exit 2
fi

exec "$PYTHON" "$ROOT/eval/egress_arms.py" "$@"
