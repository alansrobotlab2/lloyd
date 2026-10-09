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
#   5b. Print every repo path a staged skill or task edit newly names that the
#      checkout does not have (the skill-path block below). Reports, never
#      blocks. #1969.
#   5c. Print a warning naming the fixture for every staged `skills/` or
#      `knowledge/` path that has a whole-file witness under `tests/fixtures/`,
#      with both sides' byte and newline counts (the witness block below).
#      Reports, never blocks, and prints nothing when nothing matches. #2381.
#   6. With LLOYD_JOB set, commit as the job rather than as whoever `user.name`
#      says, and add a `Job:` trailer (see the job-identity block below). Unset
#      means no change of any kind to what this wrapper used to do. #668.
#
# Usage:
#   # Whole-tree snapshot; the commit says it carries state it cannot attribute:
#   ~/lloyd/scripts/util/vault-commit.sh "nightly-reflection: pre-flight $(date -u +%Y-%m-%d)"
#   # Only what this job wrote:
#   ~/lloyd/scripts/util/vault-commit.sh "autonomy-data-pipeline: $(date -u +%Y-%m-%d)" \
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
SKIPPED=()
for record in ${RECORDS[@]+"${RECORDS[@]}"}; do
    if [ -n "$record" ]; then
        path="${record:3}"
        # Skip a record whose INDEX column is `D` whose path is absent from the
        # worktree: the deletion is already recorded in the index, which is what
        # `git commit` reads, and `git add -A -- p` on an absent p dies with
        # "fatal: pathspec 'p' did not match any files" (rc 128), aborting the whole
        # commit under `set -euo pipefail`. #2245: another job's staged rename of the
        # promotions ledger to its dated witness name — which `--no-renames` reports as
        # exactly this record for the SOURCE — killed task #24's pre-flight snapshot and
        # its path-scoped commit for ~20 minutes, and both routes hit the same pathspec
        # because the source sits under a directory the scoped call names. The vault
        # commit that finally landed that rename is 7786c598.
        #
        # The `-e` test is what keeps this from being "skip anything absent". A plain
        # worktree deletion (` D p`, nothing staged) is absent too, and `git add -A --
        # p` SUCCEEDS for it because the pathspec matches the entry the index still
        # holds, so filtering on absence alone would quietly stop this wrapper
        # committing deletions. Only the first column separates the two: it is the
        # index's own answer about that path.
        if [ "${record:0:1}" = "D" ] && [ ! -e "$path" ]; then
            SKIPPED+=("$path")
            continue
        fi
        CHANGED+=("$path")
    fi
done

if [ ${#CHANGED[@]} -eq 0 ]; then
    # Two different trees reach here. A genuinely clean one still takes the fast
    # exit; a tree whose only record was a deletion already in the index (a
    # lone staged `git rm`, with no destination to add) does not, because the index
    # is not clean and `git commit` would take it. Say nothing about a clean tree in
    # that case and let the readback below answer from the index — the same source of
    # truth the commit itself reads.
    if [ ${#SKIPPED[@]} -eq 0 ]; then
        echo "vault-commit.sh: nothing to commit (clean tree on main)" >&2
        exit 0
    fi
else
    git add -A -- "${CHANGED[@]}"
fi

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

# Second pre-flight rung (#1969): name every repo path a staged skill or task
# edit ADDS that the checkout does not have. Two suite nodes read the live vault
# and go red on main for exactly that, and the writers are nightly jobs whose
# prose a vault-side rule cannot bind — `~/lloyd/.t` was removed by hand on
# 2026-09-30 and re-added by the next knowledge write. REPORT ONLY, like the rung
# above: this wrapper also commits other writers' state, so a refusal here would
# let one job's sentence block every later job's snapshot.
PATH_RUNG="$SCRIPT_DIR/skill_path_findings.py"
if [ ! -f "$PATH_RUNG" ]; then
    echo "vault-commit.sh: skill-path CHECK SKIPPED (missing $PATH_RUNG)" >&2
elif [ -z "$PY" ]; then
    echo "vault-commit.sh: skill-path CHECK SKIPPED (no python interpreter)" >&2
else
    "$PY" "$PATH_RUNG" --repo "$VAULT" || echo "vault-commit.sh: skill-path rung exited nonzero; committing anyway" >&2
fi

# Third pre-flight rung (#2381): name every whole-file witness in
# `tests/fixtures/*_witness_*.md` for a staged `skills/` or `knowledge/` path, with
# both sides' byte and newline counts. A `live_vault` node byte-compares such a pair
# (`tests/test_memory_ledger_bound.py`), and vault `e45a6b6c` — one appended bullet
# in a nightly knowledge write — left that node red until the NEXT morning's #83
# pre-flight ran it, because nothing on the writer's side said the file it appended
# to was frozen. REPORT ONLY, like the two rungs above, and deliberately so: the
# byte pin exists to make a re-freeze a reviewed act, so the committing job is told
# to file a draft, never to re-freeze it itself. Silence is the normal answer — an
# ordinary commit stages no witnessed file. Prints to stdout, because #1070 makes
# the unattributed path list the last thing on stderr and its nodes pin that tail.
WITNESS_RUNG="$SCRIPT_DIR/witness_fixture_findings.py"
if [ ! -f "$WITNESS_RUNG" ]; then
    echo "vault-commit.sh: witness CHECK SKIPPED (missing $WITNESS_RUNG)" >&2
elif [ -z "$PY" ]; then
    echo "vault-commit.sh: witness CHECK SKIPPED (no python interpreter)" >&2
else
    "$PY" "$WITNESS_RUNG" --repo "$VAULT" || echo "vault-commit.sh: witness rung exited nonzero; committing anyway" >&2
fi

# The `-c` overrides go BEFORE the subcommand: after it, git parses `-c` as
# `git commit -c` (reuse the message and open an editor) and refuses the whole
# command with "options '-m' and '-c' cannot be used together" — the same trap
# `git log -1 -g` is. The message and `--trailer` are commit options, so they stay
# after, and git appends the trailer to the end of the whole message — so the
# unattributed block sits between the subject and `Job:`, and both survive.
git "${GIT_ARGS[@]+"${GIT_ARGS[@]}"}" commit "${MSG_ARGS[@]}" "${TRAILER_ARGS[@]+"${TRAILER_ARGS[@]}"}"

# ── the loaded-memory post-flight check, as a command (#2184) ──────────────
#
# Emitted AFTER the commit, and only if it succeeded (#2488). The command names the
# commit it is about, and a check addressed to `HEAD` names whatever HEAD is when a
# reader gets around to running it — which is how today's run records read:
# `memory/vault-maintenance/2026-10-09.md` quotes the HEAD-addressed line and then
# reports a figure "for `7c58a8bd`" and "for `2eaec21a`", so the quoted command no
# longer reproduces the number beside it, and one record has already hand-minted its
# own `--pretty=format: <sha>` variant of this line to get that figure
# (`2026-10-08.md:367`) — the drift #2184 owed ruling #4 exists to prevent. Emitting
# here is also the only point where a sha exists: the `printf` used to sit ~70 lines
# BEFORE the only `git commit` in this file, and pinning by message subject is unsound
# because the same subject commits twice a day (`6e04fdca` and `e171f4f5`, both
# "autonomy-data-pipeline: 2026-10-07"). Under `set -e` a failed commit aborts the
# script at that line, so nothing here prints a check naming a commit that was never
# created.
#
# The two formatting rules #1070 forces are unchanged by the move, and the move keeps
# them: the command stays at COLUMN 0, because an indented stderr line under the
# unattributed header reads as one of its path entries — to `_paths_in` in
# tests/test_vault_commit_attribution.py and to the run record that copies the block;
# and the `unattributed dirty state:` list stays stderr's TAIL (this block goes before
# it, and `echo "$REPORT"` travelled here with the check for exactly that reason), so
# the tail a job copies verbatim is still the path list and nothing else. The
# witness-fixture rung above prints to stdout for the same reason.
#
# What the command itself is for: #1070 makes a job copy the list above into its run
# record, and #2184 found the records then proving "no MEMORY.md, USER.md or SOUL.md
# path is in this commit" with
#     git show --name-only <sha> | grep -iE 'MEMORY|USER|SOUL'
# That input is the whole commit object — message included — and the message body
# carries the unattributed list just printed plus, almost always, the sentence naming
# those very filenames. The grep was handed the claim it was being used to prove.
# Measured on the vault: a49a256d returns 7 lines, 3 of them message; ca546bac returns
# 3, 1 of them message. Both verdicts were true; the quoted evidence was not
# reproducible. And suppressing the message fixes neither half on its own — with the
# word pattern kept and the message suppressed, memory/audit/writes.jsonl and
# memory/skills-index.md still match on their directory word: ca546bac → 2 lines,
# a49a256d → 4. So emit the reproducible form: message suppressed, the pattern
# anchored to the three filenames the runtime write guard protects
# (app/harness/protected_paths.py:157,224-225), so no path that merely contains one of
# those words can satisfy it, and `|| true` appended because `grep -c` exits 1 on the
# zero this check is looking for and the line exists to be pasted under a caller's
# `set -e`.
if [ ${#UNATTRIBUTED[@]} -gt 0 ]; then
    COMMITTED_SHA="$(git -C "$VAULT" rev-parse HEAD 2>/dev/null || true)"
    # An empty sha means rev-parse could not answer, which is not a reason to say
    # nothing about a commit that did land — and is emphatically not a reason to fall
    # back to `HEAD`, the moving ref this item exists to remove. The list still goes
    # out, because the commit it describes exists either way.
    if [ -n "$COMMITTED_SHA" ]; then
        printf '%s\n' \
            "vault-commit.sh: loaded-memory post-flight check (#2184) — run this and" \
            " record its output. It names THIS commit by sha, so it re-derives the same" \
            " figure days later at any HEAD. It reads only this commit's file list," \
            " never the message body that names this check, and its pattern is anchored" \
            " to the curated filenames, so a path merely containing the word memory" \
            " does not satisfy it. 0 means clear:" >&2
        printf 'git -C "%s" show --name-only --pretty=format: %s | grep -icE %s || true\n' \
            "$VAULT" "$COMMITTED_SHA" "'(^|/)lloyd/(MEMORY|USER|SOUL)\.md'" >&2
    fi

    # the list goes last, so its tail is what a job copies (#1070, and the nodes
    # here that assert stderr's tail is the path list and nothing else)
    echo "$REPORT" >&2
fi
