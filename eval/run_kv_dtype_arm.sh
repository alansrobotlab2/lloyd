#!/usr/bin/env bash
# One-shot KV cache dtype arm over the gated primary restart (#2404).
#
#   bash eval/run_kv_dtype_arm.sh bf16
#
# The behaviour — which hold it takes, what it refuses, what it stages, how it closes — is
# in `eval/kv_dtype_arm.py` and in `tests/test_run_kv_dtype_arm.py`. This file is a thin
# wrapper in the shape `eval/run_persistence_arms.sh` uses: it resolves the interpreter, then
# `exec`s the driver so that a signal reaches the process that owns the cleanup. It carries
# deliberately no `trap … EXIT`: an EXIT trap here would fire in the wrapper, after the
# driver has already been killed, which is the difference between a released hold and a
# paused pool nobody can explain.
#
# Why the window exists at all: nine `A/B config:` lines in this box's service logs are all
# `kv_dtype=fp8`, so the engine-output integrity canary has never compared one dtype against
# another and #2163's divergence proof has nothing to read. One arm is what closes that.
#
# WHY A SHELL FILE AT ALL, when a round may not run it: firing this at the primary is an
# attended decision, and `app/harness/service_control.py` is not consulted when a restart
# hides inside a shell script — so the refusal this window needs is its own, and it has one:
# it takes a dispatch hold, checks that the queue drained, and does all of that BEFORE it
# stages anything. An engine restart with jobs in flight is the failure it exists to make
# impossible, and the only guard against it here is this file's own check, not the harness.
set -euo pipefail

usage() {
  echo "usage: run_kv_dtype_arm.sh <kv-cache-dtype>   # e.g. bf16; the window restores fp8" >&2
  exit 2
}
[[ $# -eq 1 && -n "${1:-}" && "${1:0:1}" != "-" ]] || usage

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Two roots, deliberately separate. `$REAL_ROOT` is the checkout this wrapper lives in, which
# is where the venv is; `KV_ARM_REPO` is the tree whose service directory gets the boot env
# and whose `agent-services/bin/flash-next-canary-step.sh` gets run, and a test points it at
# a fake checkout. A fake repo has no `.venvs/` in it, so resolving the interpreter through
# `KV_ARM_REPO` would make every test run die on a missing interpreter — and resolving it
# through `$LLOYD_DATA` would make it depend on where the runtime data happens to live, which
# is exactly the moved-root failure #2265 recorded.
REAL_ROOT="$(cd "${HERE}/.." && pwd)"
KV_ARM_REPO="${KV_ARM_REPO:-$REAL_ROOT}"
KV_ARM_VENV_PYTHON="${KV_ARM_VENV_PYTHON:-${REAL_ROOT}/.venvs/lloyd/bin/python}"
if [[ ! -x "$KV_ARM_VENV_PYTHON" ]]; then
  echo "no interpreter at $KV_ARM_VENV_PYTHON. Not running." >&2
  exit 1
fi

# The boot derives the arm env's home from `$LLOYD_DATA` itself
# (`start-qwen38-flash-next.sh:190`), so this file must stage into the same place or the arm
# silently does nothing. No fallback default: guessing a data root is how an arm gets staged
# into a tree nothing reads.
if [[ -z "${LLOYD_DATA:-}" ]]; then
  echo "LLOYD_DATA is not set; refusing to guess which service directory to arm." >&2
  exit 2
fi

# `KV_ARM_DRIVER` is the seam `tests/test_run_kv_dtype_arm.py` uses to pin THIS file's own
# contract — that it replaces itself with the driver, in place, with the dtype as the only
# argument — without that pin booting an engine. It defaults to the driver sitting next to
# this file, and an attended run never sets it, so the window an operator fires is always
# the real one. Every guard above still runs before it: the seam swaps WHAT gets exec'd,
# never whether the interpreter and `$LLOYD_DATA` were checked first.
KV_ARM_DRIVER="${KV_ARM_DRIVER:-${HERE}/kv_dtype_arm.py}"
export KV_ARM_REPO KV_ARM_VENV_PYTHON KV_ARM_DRIVER

exec "$KV_ARM_VENV_PYTHON" "$KV_ARM_DRIVER" "$@"
