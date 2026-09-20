#!/bin/bash
# vault-commit.sh — branch-safe wrapper for committing to ~/obsidian.
#
# Background: see backlog #341. Multiple vault writers (autonomy-data-pipeline,
# nightly-reflection-*, nightly-skills-management, etc.) used raw
# `cd ~/obsidian && git add -A && git commit -m "..."` patterns. None of them
# checked which branch HEAD was on. When self_improve.py left HEAD on an
# experiment-* branch, those writers committed to the experiment branch
# instead of main, stranding the data.
#
# This wrapper enforces "always commit to main":
#   1. Checks current branch.
#   2. If not on main, force-checks-out main first (with warning).
#   3. Stages all changes and commits with the provided message.
#   4. Skips the commit cleanly if there's nothing to commit.
#   5. Before committing, prints any autonomy task-status transition sitting in
#      the staged tree (see the autonomy-status block below). That step reports
#      and never blocks: it exists so a dispatch-killing `status` flip cannot be
#      certified by a commit message that talks about something else.
#   6. With LLOYD_JOB set, commits as the job rather than as whoever `user.name`
#      says, and adds a `Job:` trailer (see the job-identity block below). Unset
#      means no change of any kind to what this wrapper used to do.
#
# Usage:
#   ~/lloyd/scripts/util/vault-commit.sh "autonomy-data-pipeline: $(date +%Y-%m-%d)"
#   LLOYD_JOB=nightly-knowledge-write ~/lloyd/scripts/util/vault-commit.sh "nightly: knowledge write 2026-09-20"
#
# Environment:
#   VAULT_DIR   vault to commit (default ~/obsidian)
#   LLOYD_JOB   job name; when set, this commit is authored as lloyd-<job>
#               <<job>@jobs.lloyd.local> and carries `Job: <job>`. The one place
#               to change that spelling is the identity block below.
#
# Exit codes:
#   0 — committed successfully OR nothing to commit
#   2 — git operation failed (logged to stderr)
#   3 — invocation error (missing message arg, or LLOYD_JOB is not a usable name)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "vault-commit.sh: usage: vault-commit.sh \"<commit message>\"" >&2
    exit 3
fi

MSG="$1"
VAULT="${VAULT_DIR:-$HOME/obsidian}"

# Job identity (#668). A commit is attributed to the job that wrote it —
# `lloyd-<job> <<job>@jobs.lloyd.local>` plus a `Job: <job>` trailer — so
# `git log` can answer "which job wrote this" without reading the subject.
# Without LLOYD_JOB nothing here happens: the ambient identity and the message
# are exactly what they were, which is what leaves the human path and the
# automod-round path untouched. Refuses (exit 3) on a name that could not form
# a git identity, rather than committing a machine write under Alan by accident.
# The canonical spelling of both strings also lives in
# scripts/util/vault_commit_identity.py, which is the query side; tests/
# test_vault_commit_identity.py pins the two spellings against each other.
JOB="${LLOYD_JOB:-}"
if [ -n "$JOB" ]; then
    if ! [[ "$JOB" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
        echo "vault-commit.sh: LLOYD_JOB='$JOB' is not a usable job name (want [A-Za-z0-9][A-Za-z0-9._+-]*)" >&2
        exit 3
    fi
    GIT_ARGS=(-c "user.name=lloyd-$JOB" -c "user.email=$JOB@jobs.lloyd.local")
    TRAILER_ARGS=(--trailer "Job: $JOB")
else
    GIT_ARGS=()
    TRAILER_ARGS=()
fi

if [ ! -d "$VAULT/.git" ]; then
    echo "vault-commit.sh: $VAULT is not a git repo" >&2
    exit 2
fi

cd "$VAULT"

# Branch guard: never commit to a feature branch.
current=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "DETACHED")
if [ "$current" != "main" ]; then
    echo "vault-commit.sh: WARNING: HEAD on '$current' (expected main). Forcing checkout to main." >&2
    if ! git checkout -f main 2>&1; then
        echo "vault-commit.sh: ERROR: failed to checkout main" >&2
        exit 2
    fi
fi

# Stage and commit. Skip cleanly if nothing to commit.
git add -A
if git diff --cached --quiet; then
    echo "vault-commit.sh: nothing to commit (clean tree on main)" >&2
    exit 0
fi

# Pre-flight review rung (#1127): name every autonomy task-status transition in
# the STAGED tree before this commit certifies it. A flip into `draft`/`paused`
# stops a task dispatching with no other trace — commit 6f657fa9 carried #68
# (every-15min) into `draft` under a message certifying three prose files, and
# the task ran nothing for ~30 h. Read only the index, so the finding does not
# depend on which job is committing. REPORT ONLY: a job claiming its own task
# (`up_next` -> `in_progress`) is a legitimate write, so nothing here blocks.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNG="$SCRIPT_DIR/autonomy_status_findings.py"
PY="${LLOYD_PYTHON:-$SCRIPT_DIR/../../.venvs/lloyd/bin/python}"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3 || true)"
fi
if [ ! -f "$RUNG" ]; then
    echo "vault-commit.sh: autonomy-status CHECK SKIPPED (missing $RUNG)" >&2
elif [ -z "$PY" ]; then
    echo "vault-commit.sh: autonomy-status CHECK SKIPPED (no python interpreter)" >&2
else
    "$PY" "$RUNG" --repo "$VAULT" || echo "vault-commit.sh: autonomy-status rung exited nonzero; committing anyway" >&2
fi

# The `-c` overrides go BEFORE the subcommand: after it, git parses `-c` as
# `git commit -c` (reuse the message and open an editor) and refuses the whole
# command with "options '-m' and '-c' cannot be used together" — the same trap
# `git log -1 -g` is. `--trailer` is a commit option, so it stays after.
git "${GIT_ARGS[@]+"${GIT_ARGS[@]}"}" commit -m "$MSG" "${TRAILER_ARGS[@]+"${TRAILER_ARGS[@]}"}"
