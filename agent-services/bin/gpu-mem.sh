#!/usr/bin/env bash
# gpu-mem.sh — the one reading of a GPU's memory that the LLM slot start scripts
# share (#1316).
#
# Source it, then:
#
#     if gpu_mem_read FREE_MIB TOTAL_MIB GPU_NAME "$GPU"; then ... fi
#     gpu_mem_holders "$GPU"        # "pid, used MiB, process" per line, or empty
#
# Or run it for one field:
#
#     gpu-mem.sh free 2             # -> 20
#
# WHY ONE FILE. GPU 2 is single-tenant: it carries the djev ranker or the Qwen3.6
# secondary, never both — config.yaml's `djev:` comment calls that "an either/or,
# not a pair of flags". Every script that wants the card therefore has to ask the
# card first, and until now each one inlined its own
# `nvidia-smi --query-gpu=... | tr -d ' '` pipeline to ask. Two private copies of
# one measurement is how two guards end up disagreeing about one card: change the
# field list, the `--id` form or the whitespace-stripping in one script and only
# that script's notion of "free" moves, while the other keeps printing a confident
# number nobody re-checked. The refusal belongs to each script; the reading belongs
# here, once.
#
# WHAT THIS FILE DELIBERATELY DOES NOT DECIDE: what to do when the card cannot be
# measured. Its two callers answer that differently and both are right for their own
# slot, which is why the decision is not baked in here:
#
#   start-djev.sh        refuses. It is the live recall ranker (app/djev.py), and an
#                        unreadable card is not evidence of a free one. It also
#                        already behaved this way: `set -euo pipefail` took the
#                        script down with the failed command substitution, in
#                        silence. It still does not boot; now it says so.
#   start-secondary.sh   warns and starts anyway. The slot is optional
#                        (`secondary_enabled: false` is the designed state, and
#                        app/secondary_models.py routes post-session jobs to the
#                        primary around it), so a dead driver or an nvidia-smi that
#                        answers `N/A` must not become an engine outage. #1316
#                        clause 5.
#
# So a non-zero return from gpu_mem_read means "there is no reading" and nothing
# more. gpu_mem_query keeps the three ways that happens distinguishable: 2 = no
# nvidia-smi on PATH, 3 = nvidia-smi exited non-zero, 4 = the reply was not a
# number. The last one matters on its own: `N/A` and `ERROR` both come back from a
# working nvidia-smi with status 0, which is exactly why the numeric check lives
# here instead of in each caller.
#
# One field per nvidia-smi call, never a combined `--query-gpu=a,b`: it costs a few
# milliseconds, it keeps each field's failure separable, and the stub nvidia-smi in
# tests/test_start_djev_flags.py — which answers by matching one field name and
# exits 1 on anything else — stays a valid positive control that this file speaks
# nvidia-smi's language.

gpu_mem_query() {
    # gpu_mem_query <numeric-field> <gpu-id>   — memory.free, memory.total. Prints
    # the bare integer, MiB.
    local _field=$1 _gpu=$2 _out
    command -v nvidia-smi >/dev/null 2>&1 || return 2
    if ! _out=$(nvidia-smi --id="$_gpu" --query-gpu="$_field" \
                    --format=csv,noheader,nounits 2>/dev/null); then
        return 3
    fi
    _out=${_out//[$'\t\r\n ']/}
    [[ "$_out" =~ ^[0-9]+$ ]] || return 4
    printf '%s' "$_out"
}

gpu_mem_text() {
    # gpu_mem_text <string-field> <gpu-id>   — `name` and friends, which have no
    # number to validate, so the only check is that something came back.
    local _field=$1 _gpu=$2 _out
    command -v nvidia-smi >/dev/null 2>&1 || return 2
    if ! _out=$(nvidia-smi --id="$_gpu" --query-gpu="$_field" \
                    --format=csv,noheader 2>/dev/null); then
        return 3
    fi
    [[ -n "${_out//[$'\t\r\n ']/}" ]] || return 4
    printf '%s' "$_out"
}

gpu_mem_read() {
    # gpu_mem_read <FREE_VAR> <TOTAL_VAR> <NAME_VAR> <gpu-id>
    # Returns 0 only when free and total both came back as integers, so a caller
    # about to compare `free < need` is comparing two measurements and not a
    # measurement against an empty string — an empty `FREE_MIB` is below every
    # need, which would turn an unread card into a refusal and clause 5 into an
    # outage. The name is decorative: a card nobody can name still has numbers on
    # it, so a failed name query leaves a placeholder rather than failing the read.
    local -n __gm_free=$1
    local -n __gm_total=$2
    local -n __gm_name=$3
    local _gpu=$4 _v
    if ! _v=$(gpu_mem_query memory.free "$_gpu"); then
        return 1
    fi
    __gm_free=$_v
    if ! _v=$(gpu_mem_query memory.total "$_gpu"); then
        return 1
    fi
    __gm_total=$_v
    __gm_name=$(gpu_mem_text name "$_gpu") || __gm_name="GPU $_gpu"
    [[ -n "${__gm_name//[$'\t\r\n']/}" ]] || __gm_name="GPU $_gpu"
    return 0
}

gpu_mem_holders() {
    # gpu_mem_holders <gpu-id> — one "pid, used_memory, process_name" line per
    # compute app on the card; empty output when nothing holds it or the list
    # cannot be had. Always 0: who is on the card is the courtesy that makes a
    # refusal readable, not the basis of it, and a caller must still refuse when the
    # driver will not say. One call, not the two the djev refusal used to make (one
    # to test with grep -q, one to print) — the second could succeed where the first
    # timed out and print nothing under the "what is on the card:" heading.
    local _gpu=$1
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --id="$_gpu" --query-compute-apps=pid,used_memory,process_name \
        --format=csv,noheader 2>/dev/null || true
}

# Act as a command only when executed. When sourced, `${BASH_SOURCE[0]}` is this
# file and `$0` is the caller's, so none of this runs — a sourced file must not set
# shell options, exit, or print at the script that sourced it.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _gm_gpu=${2:-2}
    case "${1:-free}" in
        free)    _gm_out=$(gpu_mem_query memory.free "$_gm_gpu") ||
                     { echo "gpu-mem: cannot read memory.free on GPU $_gm_gpu" >&2; exit 1; } ;;
        total)   _gm_out=$(gpu_mem_query memory.total "$_gm_gpu") ||
                     { echo "gpu-mem: cannot read memory.total on GPU $_gm_gpu" >&2; exit 1; } ;;
        name)    _gm_out=$(gpu_mem_text name "$_gm_gpu") ||
                     { echo "gpu-mem: cannot read name on GPU $_gm_gpu" >&2; exit 1; } ;;
        holders) gpu_mem_holders "$_gm_gpu"; exit 0 ;;
        *) echo "usage: gpu-mem.sh {free|total|name|holders} [gpu-id]" >&2; exit 2 ;;
    esac
    printf '%s\n' "$_gm_out"
fi
