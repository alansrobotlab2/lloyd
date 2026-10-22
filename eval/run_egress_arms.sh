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
# the wrapper validates N, proves the aggregator can import what it will be asked to run
# (#3169, below), and then `exec`s the driver: `exec` sends the signal to the process that
# owns the cleanup rather than to a shell that only looks less leaky — and the reason that
# import proof is a line here rather than a `trap` or a driver-side check is that it has to
# be answered before anything holds the pool. The live-store leak check and the
# persisted-hold re-check both run after the event loop is gone.
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

# #3169: the ON arm's aggregator is `python -m agent_mcp.main`, and `agent_mcp` is a plain
# package of this checkout that is installed in no interpreter here, so it resolves ONLY
# through the cwd and PYTHONPATH the CHILD is given. Ask that question of
# `aggregator_binding.aggregator_preflight()` — the function whose sibling dict the driver
# splats into `create_subprocess_exec` for the real spawn, so the check cannot pass and the
# spawn still fail — and note what it is NOT asked of: this script's own cwd.
# `lloyd-egress-window.service` sets no `WorkingDirectory=`, so under systemd that cwd is
# wherever the manager left the wrapper, and a check done THERE would refuse every fire
# forever on a tree that imports perfectly (the trap the item's own clause 2 was worded
# into). Asked HERE, before the driver takes the pool pause, because the 2026-10-19 06:26:56
# fire asked it at the fork instead: by the time the child answered `ModuleNotFoundError: No
# module named 'agent_mcp'` the window had already spent 16.652 s of wall clock on the pause
# and the drain and had nothing to show for it. The check itself is a measured 1.4 s.
#
# `aggregator_binding` and not `egress_arms`, even though it is the driver that spawns: the
# driver imports `persistence_arms` at module level, which imports `workers.pool`, so in a
# partial tree `import egress_arms` fails for a reason that has nothing to do with the
# aggregator and the refusal blames the wrong thing. The pre-flight module is stdlib-only on
# purpose. The node that keeps it that way is
# `tests/test_egress_window_unit.py::test_widening_the_childs_environment_brought_no_arm_flag_and_no_working_directory`,
# which parses this module's imports and refuses anything naming the repo.
# The unquoted `${LLOYD_MCP_CMD:-}` is deliberate: an override has to reach the check, or a
# window launched with a custom aggregator command is pre-flighted against a command it will
# never run. It splits on spaces the way the driver's `shlex.split` does for anything a human
# would type here; a command whose argv needs quoting belongs in the driver's side.
if ! "$PYTHON" -c 'import sys; sys.path.insert(0, sys.argv[1]); import aggregator_binding as ab; raise SystemExit(ab.aggregator_preflight(sys.argv[2:]))' \
    "$ROOT/eval" ${LLOYD_MCP_CMD:-}; then
  echo "${0##*/}: refusing before the pool pause — the enforce-ON arm's aggregator cannot" >&2
  echo "  import the package it is asked to run, from this checkout ($ROOT)." >&2
  exit 2
fi

exec "$PYTHON" "$ROOT/eval/egress_arms.py" "$@"
