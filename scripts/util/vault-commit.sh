#!/bin/bash
# vault-commit.sh — branch-safe, attribution-honest wrapper for committing ~/obsidian.
#
# Background: see backlog #341. Multiple vault writers (autonomy-data-pipeline,
# nightly-reflection-*, nightly-skills-management, etc.) used raw
# `cd ~/obsidian && git add -A && git commit -m "..."` patterns. None of them
# checked which branch HEAD was on. When self_improve.py left HEAD on an
# experiment-* branch, those writers committed to the experiment branch
# instead of main, stranding the data.
#
# Background: see backlog #1070. The wrapper then staged the WHOLE tree and
# committed it under the *calling* job's message, so `git log` answered "who wrote
# this line" with "whoever committed last". Three incidents are still unattributed
# because of it: `0860640e` "autonomy-data-pipeline: pre-flight 2026-09-10"
# (21 files) is where the MEMORY.md one-line truncation first surfaces — its own
# clobber record still says the truncating writer is unidentified, because the
# truncating writer was not the committer; `fe1cada6` "skills-mgmt: pre-run
# snapshot" (34 files) captured the MEMORY.md copy of SOUL.md (#464, closed
# unattributed); and `3274963e` "skills-mgmt: pre-run snapshot" (44 files, +1599
# −99) is where autonomy task #51's silent `model: secondary` revert appears (#937,
# closed unattributed). A `pre-flight`/`pre-run snapshot` step runs before the job
# writes anything, so by construction each of those commits is somebody else's
# state under an innocent job's name.
#
# This wrapper enforces:
#   1. Never commit to a feature branch: if HEAD is not on main, force-checkout
#      main first (with a warning). #341.
#   2. Commit what the caller says it wrote. With a pathspec — `-- <path>…` — only
#      the named paths are staged and committed, and everything else stays dirty
#      for its own writer. `scripts/automod/vault_round.py` has staged exactly the
#      named paths since it was written; this wrapper now takes the same route, so
#      the two vault-landing paths do not diverge. #1070.
#   3. Say so when the caller cannot. Invoked with no pathspec while the tree holds
#      changes, the commit gets an `unattributed dirty state:` block in its body
#      naming every path it carries, and the same list goes to stderr for the job to
#      copy into its own run record. A pre-flight snapshot does commit state it did
#      not write; the message now says that instead of hiding it. #1070.
#   4. Skip the commit cleanly (exit 0) if there is nothing to commit.
#   5. Before committing, print any autonomy task-status transition sitting in the
#      staged tree (see the autonomy-status block below). That step reports and
#      never blocks: it exists so a dispatch-killing `status` flip cannot be
#      certified by a commit message that talks about something else. #1127.
#   6. With LLOYD_JOB set, commit as the job rather than as whoever `user.name`
#      says, and add a `Job:` trailer (see the job-identity block below). Unset
#      means no change of any kind to what this wrapper used to do. #668.
#
# Usage:
#   # Whole-tree snapshot; the commit says it carries state it cannot attribute:
#   ~/lloyd/scripts/util/vault-commit.sh "nightly-reflection: pre-flight $(date +%Y-%m-%d)"
#   # Only what this job wrote:
#   ~/lloyd/scripts/util/vault-commit.sh "autonomy-data-pipeline: $(date +%Y-%m-%d)" \
#       -- memory/ backlog/ autonomy/
#   LLOYD_JOB=nightly-knowledge-write ~/lloyd/scripts/util/vault-commit.sh "nightly: knowledge write 2026-09-20"
#   # …and the same scope, but reporting anything carried that this job did not write:
#   LLOYD_JOB=autonomy-data-pipeline LLOYD_JOB_WRITES="memory/vault-maintenance/2026-09-30.md" \
#       ~/lloyd/scripts/util/vault-commit.sh "autonomy-data-pipeline: 2026-09-30" \
#       -- memory/ knowledge/ backlog/ autonomy/
#
# A named path that holds no change is skipped, not fatal: a job names every
# segment it ever writes and usually dirties two of them.
#
# Environment:
#   VAULT_DIR   vault to commit (default ~/obsidian)
#   LLOYD_JOB   job name; when set, this commit is authored as lloyd-<job>
#               <<job>@jobs.lloyd.local> and carries `Job: <job>`. The one place
#               to change that spelling is the identity block below.
#   LLOYD_JOB_WRITES  the vault-relative paths THIS job wrote, colon-separated
#               (newlines work too, and padding around a separator is ignored),
#               honoured only alongside LLOYD_JOB. A path-scoped commit then
#               reports as unattributed every path it carries that is not on this
#               list — see the ownership block below.
#
# Exit codes:
#   0 — committed successfully OR nothing to commit
#   2 — git operation failed (logged to stderr)
#   3 — invocation error (no message; an argument where a pathspec separator
#       belongs; a pathspec or a LLOYD_JOB_WRITES entry naming the whole tree;
#       LLOYD_JOB not a usable name)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "vault-commit.sh: usage: vault-commit.sh \"<commit message>\" [ -- <path>… ]" >&2
    exit 3
fi

MSG="$1"
shift

# The pathspec: everything after a bare ` -- `. A second argument that is NOT that
# separator is refused rather than ignored — silently dropping a path list is how a
# caller ends up believing it scoped a commit while the wrapper sweeps the whole
# tree under the caller's name, which is the defect this script exists to stop.
PATHSPEC=()
if [ $# -gt 0 ]; then
    if [ "$1" != "--" ]; then
        echo "vault-commit.sh: unexpected argument '$1'; paths must follow ' -- '" >&2
        exit 3
    fi
    shift
    if [ $# -lt 1 ]; then
        echo "vault-commit.sh: '--' given with no paths after it" >&2
        exit 3
    fi
    PATHSPEC=("$@")
fi

# `-- .` or `-- *` claims the whole tree is the caller's own work — the exact claim
# this wrapper was written to stop making — and a glob or an absolute path is not a
# path the ownership check below can compare against a staged path. Name the paths
# this job wrote, or pass no pathspec at all and let the commit report itself as
# unattributed.
for spec in ${PATHSPEC[@]+"${PATHSPEC[@]}"}; do
    case "$spec" in
        ''|'.'|'./'|*'*'*|'/'*|'./'*|*'..'*)
            echo "vault-commit.sh: pathspec '$spec' is not a scoped path; name the paths this job wrote, or omit the pathspec" >&2
            exit 3
            ;;
    esac
done

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

# The declared write list (#1867): the paths the invoking job actually wrote.
#
# #1070 gave a path-scoped commit its `unattributed dirty state:` block, but it
# computed ownership as "lies under one of the paths named on the command line".
# Nightly jobs name whole segments — `-- memory/ knowledge/ backlog/ autonomy/` is
# the literal in autonomy-data-pipeline's Post-Flight step — so every path inside a
# named segment was owned by construction and the block could never fire for one.
# The 2026-09-30 01:17Z run of that job ran that literal and committed
# `backlog/1866-*.md` (95 insertions to another job's in-flight item) plus the audit
# logger's append to `memory/audit/writes.jsonl` under the subject
# `autonomy-data-pipeline: 2026-09-29`, with no block in the body and nothing on
# stderr. That is `05aeb28a` (46 files under one job's subject) returning through the
# path-scoped door, and the safer-looking mode was the silent one: the same run's
# no-`--` pre-flight commit DID print the block, for all 15 paths it carried.
#
# A job that names its writes makes the comparison ask the question it was always
# meant to ask. OPT-IN by design: with the variable unset ownership stays what #1070
# implemented, so every human `git commit`, every call site that has not adopted the
# list, and every test written before it behave byte-for-byte as before. It needs
# LLOYD_JOB too — an invocation with no job identity has no own writes to check
# against, and honouring a declaration from an anonymous caller would let any commit
# silence the guard that exists to interrogate it.
#
# No existing attribution surface knew a job's paths, which is why this is a new
# variable rather than a reader for one: `memory/audit/writes.jsonl` records only the
# `vault_write` route (app/autonomy.py says so outright), and it is itself one of the
# foreign appends this hole swallows, so it cannot be the manifest either.
DECLARED=()
if [ -n "$JOB" ] && [ -n "${LLOYD_JOB_WRITES:-}" ]; then
    # One separator rule: `:` or newline, padding ignored. A vault path containing a
    # colon cannot be declared this way — measured 2026-09-30, `git -C ~/obsidian
    # ls-files | grep -c ':'` is 0, so nothing today needs the escape.
    _DECLARED=$(printf '%s' "${LLOYD_JOB_WRITES}" \
        | tr '\n' ':' \
        | sed -e 's/[[:space:]]*:[[:space:]]*/:/g' -e 's/^[[:space:]:]*//' -e 's/[[:space:]:]*$//')
    if [ -n "$_DECLARED" ]; then
        IFS=':' read -r -a _DECLARED_LIST <<< "$_DECLARED"
        for spec in ${_DECLARED_LIST[@]+"${_DECLARED_LIST[@]}"}; do
            # `a::b` and a trailing `:` survive the sed; an empty entry would match
            # nothing anyway, and dropping it keeps the list honest about its size.
            if [ -n "$spec" ]; then
                DECLARED+=("$spec")
            fi
        done
    fi
    # The same shapes a pathspec may not take, mirrored from the rail above:
    # declaring `.` or `*` claims every path in the vault as this job's own work,
    # which is the claim that disarms the guard, and a glob, an absolute path or a
    # `..` is not something the prefix comparison below can match a staged path
    # against. Refusing here rather than ignoring matters because an entry that
    # matches nothing is indistinguishable, in a commit body, from a job that
    # declared nothing.
    for spec in ${DECLARED[@]+"${DECLARED[@]}"}; do
        case "$spec" in
            '.'|'./'|*'*'*|'/'*|'./'*|*'..'*)
                echo "vault-commit.sh: LLOYD_JOB_WRITES entry '$spec' is not a scoped path; name the paths this job wrote, or leave the variable unset" >&2
                exit 3
                ;;
        esac
    done
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

TMP_CHANGED="$(mktemp)"
TMP_STAGED="$(mktemp)"
trap 'rm -f "$TMP_CHANGED" "$TMP_STAGED"' EXIT

# Which paths hold a change? Ask `git status`, not `git add`: `git add -A -- p`
# dies with "fatal: pathspec 'p' did not match any files" whenever p is clean or
# absent, and a job names every segment it ever writes, so handing the caller's
# list straight to `git add` would turn a quiet night — the common case — into a
# failed commit step that strands the files the job DID write. `git status` answers
# an unmatched pathspec with no output instead. `-z` keeps every path literal (no
# quotePath escaping, no quoting around a name with a space), `--no-renames` keeps
# one path per record so the three-character status prefix is the only thing to
# strip, and `-uall` is what makes a directory of new files visible per-path.
SCAN=()
if [ ${#PATHSPEC[@]} -gt 0 ]; then
    SCAN=(-- "${PATHSPEC[@]}")
fi
if ! git status --porcelain=v1 -z --untracked-files=all --no-renames \
        ${SCAN[@]+"${SCAN[@]}"} > "$TMP_CHANGED"; then
    echo "vault-commit.sh: ERROR: git status failed" >&2
    exit 2
fi

mapfile -d '' -t RECORDS < "$TMP_CHANGED"
CHANGED=()
for record in ${RECORDS[@]+"${RECORDS[@]}"}; do
    if [ -n "$record" ]; then
        CHANGED+=("${record:3}")
    fi
done

if [ ${#CHANGED[@]} -eq 0 ]; then
    echo "vault-commit.sh: nothing to commit (clean tree on main)" >&2
    exit 0
fi

git add -A -- "${CHANGED[@]}"

# Read the answer back off the index rather than trusting the list above: `git
# commit` commits the index, so anything another writer left staged is in this
# commit whether this invocation named it or not, and the message has to account
# for it. A pathspec that scoped the commit therefore still owes an accounting for
# somebody else's staged entry, and it gets one below.
git diff --cached --name-only -z > "$TMP_STAGED"
mapfile -d '' -t STAGED < "$TMP_STAGED"

if [ ${#STAGED[@]} -eq 0 ]; then
    echo "vault-commit.sh: nothing to commit (clean tree on main)" >&2
    exit 0
fi

# Is this staged path one of the caller's own writes? Exact match, or the path
# sitting UNDER the entry — so `memory/` covers `memory/audit/writes.jsonl` while
# `mem` does not cover `memory/x.md`. Comparing raw prefixes without the slash would
# let a declared `mem` claim a sibling directory called `memory/`, which is the same
# bug #1593 is about one layer up.
owns_one_of() {
    local path="$1"
    shift
    local spec
    for spec in "$@"; do
        spec="${spec%/}"
        if [ "$path" = "$spec" ] || [ "${path#"$spec"/}" != "$path" ]; then
            return 0
        fi
    done
    return 1
}

# Whose writes this commit's contents are. A declared list, when the caller passed
# one with a job name and a pathspec attached, IS the answer and the pathspec gets no
# vote: "under a segment I said I might write" was never authorship, and that
# conflation is the hole #1867 names. With no declared list, ownership is exactly the
# #1070 rule — named on the command line — and with no pathspec either nothing is
# owned, which is the honest description of a pre-flight snapshot.
#
# Deliberate: a declared list changes NOTHING in the no-`--` mode. That mode is the
# sweep, its block already names every path it carries, and over-disclosing there is
# the safe direction — a job that sets the list and forgets the pathspec gets today's
# full block, never a quieter commit.
OWNER_SET=()
OWNERSHIP=scope
if [ ${#DECLARED[@]} -gt 0 ] && [ ${#PATHSPEC[@]} -gt 0 ]; then
    OWNER_SET=(${DECLARED[@]+"${DECLARED[@]}"})
    OWNERSHIP=declared
else
    OWNER_SET=(${PATHSPEC[@]+"${PATHSPEC[@]}"})
fi

# Everything staged that the caller cannot show it wrote is state this commit is
# taking responsibility for without having authored it.
UNATTRIBUTED=()
for path in ${STAGED[@]+"${STAGED[@]}"}; do
    if ! owns_one_of "$path" ${OWNER_SET[@]+"${OWNER_SET[@]}"}; then
        UNATTRIBUTED+=("$path")
    fi
done

# `-m` twice is deliberate: git separates the paragraphs with a blank line, so the
# subject stays what `git log --oneline` shows while the block is what
# `git log --format=%B` and `git show` show. The list goes to stderr in the same
# shape because a job cannot read its own commit body from where it is running, and
# #1070 makes the invoking job copy that list into its run record.
MSG_ARGS=(-m "$MSG")
if [ ${#UNATTRIBUTED[@]} -gt 0 ]; then
    COUNT=${#UNATTRIBUTED[@]}
    if [ "$OWNERSHIP" = declared ]; then
        # Say which rule named it: a reader of a commit body cannot see whether the
        # caller declared anything, and the #1070 wording below would read to one as
        # "no pathspec was given". With the list unset both sentences stay exactly as
        # #1070 wrote them, byte for byte — tests/test_vault_commit_attribution.py
        # pins them, because a job's run record quotes this prose and a rewording
        # would be a change to eight call sites' output under an innocent item.
        BODY_SUFFIX="not on this job's declared write list (LLOYD_JOB_WRITES)"
        ERR_SUFFIX="not on this job's declared write list (LLOYD_JOB_WRITES)"
    else
        BODY_SUFFIX="that this invocation did not name as its own writes"
        ERR_SUFFIX="that this invocation did not name"
    fi
    BLOCK="unattributed dirty state: ${COUNT} path(s) in this commit ${BODY_SUFFIX}. The committing job is not necessarily the writer of these lines; a path-scoped invocation ('-- <path>…') is what makes a commit's message mean its authorship."
    REPORT="vault-commit.sh: unattributed dirty state: ${COUNT} path(s) this commit carries ${ERR_SUFFIX} — copy this exact list into the job's run record:"
    for path in "${UNATTRIBUTED[@]}"; do
        BLOCK+=$'\n    '"$path"
        REPORT+=$'\n    '"$path"
    done
    MSG_ARGS+=(-m "$BLOCK")
    echo "$REPORT" >&2
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
# `git log -1 -g` is. The message and `--trailer` are commit options, so they stay
# after, and git appends the trailer to the end of the whole message — so the
# unattributed block sits between the subject and `Job:`, and both survive.
git "${GIT_ARGS[@]+"${GIT_ARGS[@]}"}" commit "${MSG_ARGS[@]}" "${TRAILER_ARGS[@]+"${TRAILER_ARGS[@]}"}"
