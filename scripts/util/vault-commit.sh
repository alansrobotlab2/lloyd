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
#
# Usage:
#   ~/lloyd/scripts/util/vault-commit.sh "autonomy-data-pipeline: $(date +%Y-%m-%d)"
#
# Exit codes:
#   0 — committed successfully OR nothing to commit
#   2 — git operation failed (logged to stderr)
#   3 — invocation error (missing message arg)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "vault-commit.sh: usage: vault-commit.sh \"<commit message>\"" >&2
    exit 3
fi

MSG="$1"
VAULT="${VAULT_DIR:-$HOME/obsidian}"

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

git commit -m "$MSG"
