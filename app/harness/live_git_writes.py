"""Refuse a HEAD-moving `git` write in the live checkout from an unattended session.

Sixth lane of `safety.check_bash_command`, added for backlog **#2604**, which
found `architecture/arch-review.md` §2 rail 4 claiming "the source commits onto
live, the model never runs `git`" while `policy.tool_tier("Bash", "git -C
/home/alansrobotlab/lloyd commit -m x -- architecture/foo.md")` answered **1** —
the same tier as the `git status` the same review prompt orders the turn to run.
The sentence was a prompt rule wearing a rail's clothes. It is a rail now, for the
one property that was never actually delegated to the model's judgement: nothing
but the source may move `HEAD` in the production checkout.

The gap was worse than over-reported. `workers/sources/arch_review.py` baselines
production with `git status --porcelain` twice and reports only the difference, so
a commit — which cleans the tree it dirtied — is invisible to that sweep; and
`revert_live_ledger_writes` walks the change ledger, which records `Edit`/`Write`
calls and, in its own docstring's words, leaves "a Bash write … in the report-only
half". A stray commit was neither reverted nor reported.

Why a refusal rather than an honest re-wording of the doc: #709's standing ruling
that a property which must hold over an open set cannot rest on a prompt sentence,
and the corpus this item measured — `autonomy-task:40` really did run `cd ~/lloyd
&& git add … && git commit -q -F -` (session `20261006_020040_autonomy_74bc`), a
commit onto production main by an unattended turn, so the exposed property is not
the arch-review prompt's alone.

What it refuses, precisely: one of the twelve subcommands in
`_MUTATING_SUBCOMMANDS` (`add`, `amend`, `apply`, `cherry-pick`, `checkout`,
`commit`, `merge`, `rebase`, `reset`, `restore`, `revert`, `switch`), or any `git
stash` form but `list`/`show`/`branch` — with its working directory being
`app.paths.LIVE_CHECKOUT` or a directory inside it (git walks up to find the
repository, so a nested `cd` is the same `HEAD`), arrived at either by `-C` or by a
`cd` earlier in the same command, from a session that is not positively attended. It is a *scoped* rail by construction: the
self-modification loop lands by committing in linked worktrees of this very repo
(`scripts/automod/round.py:111` — `git worktree add … /home/alansrobotlab/lloyd-
work/SM_…`), the arch-review source commits to production **with** a commit, and
the nightly writers commit to the vault. A guard that refused `git commit`
unconditionally in an unattended turn would stop Lloyd landing Lloyd; refusing the
same verb *in one named directory* is the property the doc claims.

What it must not refuse is where the risk of a guard like this lives, and it is
measured rather than asserted: ~2000 `git worktree add --detach` invocations
against `~/lloyd` in the retained session corpus (that is the fleet's read-only
"run pytest in a scratch checkout" pattern, whose own recipe is pasted into
backlog items), plus routine `stash list`, `merge-base --is-ancestor` and
`ls-files`. So the matcher reads subcommands from a real tokeniser, anchored on a
program-position `git`, which is why `grep -rn 'git commit' app/` — a write verb
inside another command's argument, a shape that appears ~70 times in the corpus —
is not a write. Reads against the live tree stay tier 1 because the review prompt
needs `git log --since='30 days ago' -- <paths>`, and `push` is left to its own
lane (`_HARD_DENY_PATTERNS` refuses a force-push to a protected branch; plain
`git push` is tier 2, needing a grant).

Two deliberate asymmetries with the neighbouring lanes:

  * **Attended is proven, never assumed — for an id that exists.** An empty id
    answers attended, matching `check_service_control`'s `if not session_id:
    return None`, because `agent_mcp/desktop/guards.py` calls
    `check_bash_command(text, None, at_dispatch=True)` with no id for text a
    person is typing into a terminal, and refusing the empty id would refuse that
    human's `git commit`. Past that one case the lane is *stricter* than its
    neighbour: an id carrying no positive evidence of a person is refused where
    `check_service_control` would let it pass. `attended_session` says what counts
    as evidence and why `classify_session`'s "attended" cannot be reused as-is.
  * **`cwd=None` means the aggregator's directory**, the same convention as the
    other five lanes, not the turn's real working directory. A bare `git commit`
    is therefore refused only when the caller named the live directory or a `cd`
    got there. A Bash tool call with no `cwd` argument in production runs the
    MCP aggregator, which is not the live checkout — so this is not the hole it
    looks like; `agent_mcp/main.py:536` passes whatever `cwd` the arguments
    carried, and a turn wanting production passes `cwd: ~/lloyd` or `cd`s there.

No state, no ledger of its own: a refusal is a `(label, excerpt)` pair, and the
dispatchers log it like any other lane's. Whether the lane fired only on the
failure mode over a week of traffic is what #2604 leaves owed, read from the
refusal log the aggregator already writes.

Tests: `tests/test_safety_gate_discrimination.py` (the refuse/allow tables, and
the chat exemption), `tests/test_protected_paths.py` (nothing outside the live
checkout is refused), `tests/test_live_git_writes.py` (the classifier and the
walk).
"""

from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# The two decisions this lane has to make, and get right
# ---------------------------------------------------------------------------

#: Subcommands that mutate the checkout they run in: its `HEAD`, its index, or
#: its working tree. `commit`/`reset`/`rebase`/`merge`/`cherry-pick`/`revert`/
#: `switch` move `HEAD`; `checkout`/`restore`/`amend` move the tree or the commit
#: it points at; `apply` rewrites the tree; `add` rewrites the index, which is
#: included because the source's own rail reads that index — `commit_reviewed`
#: requires HEAD, index and working tree to equal the review's `base_blob`, so a
#: turn that stages anything in production is a turn that can manufacture the
#: `commit_conflict` the rail exists to prevent.
#:
#: Deliberately *not* the whole of git's write side: `push` is another lane's
#: (a force-push to a protected branch is a hard deny, a plain push is tier 2 and
#: needs a grant), `fetch`/`pull` move only refs that are not `HEAD`, and
#: `tag`/`branch`/`remote`/`notes`/`gc`/`prune`/`filter-branch` are left out on
#: #2604's own instruction because the corpus shows several of them in routine
#: use — whether the lane should extend to them is that item's second owed
#: ruling, not a gap to close here.
_MUTATING_SUBCOMMANDS = frozenset({
    "add", "amend", "apply", "cherry-pick", "checkout", "commit", "merge",
    "rebase", "reset", "restore", "revert", "switch",
})

#: `git stash` is one verb whose forms split. `list`/`show`/`branch` are reads —
#: `stash list` is routine in the fleet corpus — while the bare `git stash`,
#: `push`, `save`, `create`, `store`, `pop`, `apply`, `drop` and `clear` all take
#: the working tree with them. `pop` sits on the write side on purpose: it
#: re-applies a stashed tree onto whatever checkout it runs in, which is exactly
#: the damage this lane is about, and the corpus that made `stash list` routine
#: contains no `pop`.
_STASH = "stash"
_STASH_READ_FORMS = frozenset({"list", "show", "branch"})

#: git's global options that consume the following token as their value, so the
#: subcommand is the token after that. Read off git's own usage rather than
#: guessed: a matcher that skipped only `-C` would read `git --namespace ns log`
#: as a subcommand called `ns` — harmless here, but `git -c x=y commit` is not,
#: and `-c` is how a scripted commit sets an author.
_GIT_VALUE_FLAGS = frozenset({
    "-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path",
    "--resolve-git-dir", "--config-env", "--super-prefix",
})

#: An interpreter option whose next token is a script to run. Following it is not
#: paranoia: `bash -c "cd ~/lloyd && git commit -m x"` is how a write reaches a
#: shell a turn was not handed directly, and the corpus has `sh -c` wrappers on
#: ordinary git calls, so the walk has to survive the wrapper in both
#: directions — refuse through it, and not fire on it when it is only a read.
_INTERP_SCRIPT_FLAGS = frozenset({"-c", "-lc", "-exc", "-xc", "-gc", "-e"})

#: Only a *shell*'s `-c` payload is a shell script. `python`/`python3`/`perl`/
#: `ruby` are deliberately absent from this list while staying in the flag set
#: above: their `-c` argument is data to a language that cannot run it, so walking
#: one invents a write that cannot happen — `python -c "print('git commit')"`
#: would read as a commit. What a Python program then hands to `subprocess` is
#: invisible to this lane exactly as it is to the pattern lane, which is the same
#: limit `check_service_control` states for a script that starts a service.
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})

#: Words that introduce a command rather than being one, so a git standing after
#: them is in program position: `if git commit -m x; then true; fi` runs the
#: commit. `then`/`do`/`else` never reach here — the tokeniser already splits on
#: them — and a privilege escalator needs no entry because the pattern lane has
#: refused the command before this one is reached.
_CONTROL_WORDS = frozenset({"if", "while", "until", "for", "time", "doas"})

#: A wrapper flag that consumes the next token, so the token after it is not the
#: program. `nice -n 5 git commit`, `timeout -k 5 30 git commit`, `stdbuf -o L
#: git commit`: leaving the value standing where the program belongs produces a
#: segment whose head is `5`, which matches nothing, and a missed write here
#: answers None — the fail-open direction, which is the one this matcher may not
#: have.
_WRAPPER_VALUE_FLAGS = frozenset({
    "-n", "-c", "-k", "--kill-after", "-s", "--signal", "-o", "-e", "-i",
    "-p", "-Q", "--class", "--priority",
})

#: How deep the interpreter walk descends. `bash -c 'bash -c "git commit"'` is
#: refused; a third nesting is not, and no shape in the corpus nests at all. The
#: bound is what stops a self-referential `-c` payload looping the walk forever.
_MAX_WALK = 2

#: Wrapper commands that shift argv without changing who runs it.
_WRAPPERS = frozenset({"env", "timeout", "nice", "nohup", "time", "command",
                       "stdbuf", "ionice"})

# ---------------------------------------------------------------------------
# Is this session a person?
# ---------------------------------------------------------------------------

#: The minted web-chat session id: `%Y%m%d_%H%M%S` then six lowercase hex
#: characters — `app/routers/messages.py:2117`, the only place a human's turn is
#: born. Measured over the 10,310 retained session records: every id is either
#: this shape or has a *source slug* as its third part (`_autotriage_`,
#: `_owedcheck_`, `_review_`, `_bench_`, `_autocode_`, …) or is `task_<8 hex>`,
#: so this pattern is the positive evidence that a person typed the turn, and the
#: narrowness of its character class is the point. A shape that stopped being
#: minted would only ever be refused here, which is the direction this lane errs.
_CHAT_ID = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")

#: A session id may arrive with a surface prefix and a colon (`worker:arch-review
#: :abc`, `autonomy-task:40`). Neither is attended, and no attended prefix exists
#: in the code — `agent_mcp/main.py` decides desktop-vs-chat from the session
#: *record*, never from an id prefix — so an id carrying a colon is refused rather
#: than trusted. A prefix that never fires fails closed here, which is the only
#: direction this lane is allowed to fail in.


def attended_session(session_id: str | None, *,
                     parent_of=None) -> bool:
    """True only for evidence that a person is watching this session.

    Three shapes count: the minted web-chat id (`_CHAT_ID`); a record on disk
    whose `platform` is one of the surfaces `app.sessions_io` does *not* class as
    a machine's, which is what an id a client supplied with no mint gets; and a
    `task:` subagent whose resolved parent is itself attended. Everything else —
    a background run name, an id with a platform prefix, a subagent with no
    resolvable parent, an absent record, and an empty id — is False, and False
    here means the write is refused.

    This is not `service_control.classify_session`, and the difference is not
    cosmetic: that function answers *"is this name recognisable as a background
    run?"*, so anything it does not recognise — `worker:arch-review:abc`,
    `autonomy-task:40`, an empty id — comes back `attended`, on the reasoning in
    its own docstring that an unattributable id must not hand a guard a refusal
    about a *service restart*, where misfiring denies the user their own
    supervisor. On this axis the same accident is a commit on production main, so
    the name is read the other way: recognition is required, and its absence is
    not evidence of a person. `classify_session` is consulted only for the one
    thing it decides soundly, whether a name is a background run — so the two
    guards cannot disagree about a name both recognise.

    `parent_of` is the resolver `check_service_control` takes. A `task:` subagent
    inherits its parent's surface; without a resolver it stays unattended rather
    than inheriting an assumption, which is the same "the guard picks its own
    side" ruling `classify_session` makes for `UNKNOWN`.
    """
    sid = str(session_id or "").strip()
    if not sid:
        # No id at all is no session anyone can attribute a privilege to, and
        # `check_service_control` already answers None for it. Refusing here would
        # block the desktop guard's terminal check, which calls
        # `check_bash_command(text, None, at_dispatch=True)` with an empty
        # `session_id` for text a person is typing into a terminal.
        return True
    if sid.startswith("task:") or sid.startswith("task_"):
        parent = parent_of(sid) if parent_of is not None else None
        if not parent or parent == sid:
            return False
        return attended_session(parent, parent_of=parent_of)

    from app.harness import service_control as SC

    # A recognised background run name (`20261012_051633_autotriage_1a2b`) is
    # never attended, whatever else its record might one day say.
    if SC.is_background_session_id(sid):
        return False
    if ":" in sid:
        return False
    if _CHAT_ID.match(sid):
        return True
    return _record_says_attended(sid)


def _record_says_attended(sid: str) -> bool:
    """The record's own platform field, when there is a record to read."""
    try:
        from app.paths import SESSIONS_DIR
        from app.sessions_io import NON_USER_PLATFORMS, read_session_fields
    except ImportError:      # pragma: no cover - a stripped deployment
        return False
    data = read_session_fields(SESSIONS_DIR / f"{sid}.json")
    if not data:
        return False
    platform = str(data.get("platform") or "").strip().lower()
    return bool(platform) and platform not in NON_USER_PLATFORMS


# ---------------------------------------------------------------------------
# The matcher
# ---------------------------------------------------------------------------

def _norm(target: str, base: str) -> str:
    if not target:
        return os.path.normpath(base)
    p = os.path.expanduser(os.path.expandvars(target))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def _is_live_root(path: str, live: str) -> bool:
    """Is `path` the production checkout, by any name git accepts for it?

    Three ways to be the live tree, and a matcher that takes only the first is
    wrong in both directions:

      * equal to it, after symlinks — `/home/x/lloyd` or `~/lloyd/`;
      * anywhere **inside** it. `cd /home/x/lloyd/architecture && git commit` is a
        write to production: git discovers the enclosing repository by walking up
        from the working directory, so a nested directory is the same checkout as
        far as `HEAD` is concerned, and a matcher keyed on equality alone let that
        through while its own docstring promised containment. Containment is
        decided on *resolved* paths and on directory boundaries, never on a raw
        string prefix: `$LLOYD_DATA` is `/home/x/lloyd-data`, a **sibling** whose
        name starts with the live path as a string, and a `startswith(live)`-only
        test would call that scratch tree production and refuse a triage round's
        own pytest recipe;
      * a symlink that resolves into it, which is how `~/lloyd` reached through a
        link, or a bind mount named in a prompt, becomes production.

    A path that does not exist is judged on its spelling: git would refuse such a
    `-C` itself, and this lane must not be the reason a typo looks like a policy
    refusal — but `/home/x/lloyd/../lloyd` and `./` are real spellings of the live
    tree even where resolution is unavailable, so they normalise before compare.
    """
    live_n = os.path.normpath(str(live))
    p = os.path.normpath(str(path))
    if _within(p, live_n):
        return True
    try:
        rp = os.path.realpath(p)
        rl = os.path.realpath(live_n)
    except OSError:     # pragma: no cover - unreadable cwd on a weird box
        return False
    return _within(rp, rl)


def _within(path: str, root: str) -> bool:
    """`path` is `root`, or a directory below it — on directory boundaries only.

    The separator is the whole function: `/home/x/lloyd-data` starts with
    `/home/x/lloyd` as a string and is not inside it as a path, and that sibling
    is the name this box actually uses for its runtime data.
    """
    if path == root:
        return True
    return path.startswith(root.rstrip("/") + "/")


def _subcommand(tokens: list[str]) -> tuple[str, int]:
    """The first token that could be a subcommand, and the index after it.

    `tokens[0]` is the `git`; a global option is skipped, and skipped *with its
    value* only if it is one of the options that takes one — so a boolean global
    (`-p`/`--paginate`, `--no-pager`, `--no-optional-locks`) is skipped alone.
    That distinction is the whole reason the value set is a list and not a rule:
    assuming a value where there is none swallows the subcommand, and a matcher
    that lost the word `commit` answers None and lets the write through.
    Returns `("", len)` when nothing follows, so a bare `git` is never a write.
    """
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            flag = tok.partition("=")[0]
            if flag in _GIT_VALUE_FLAGS and "=" not in tok:
                i += 2
                continue
            i += 1
            continue
        return tok, i + 1
    return "", len(tokens)


def _walk(text: str, cwd: str, live: str, depth: int) -> str | None:
    """The first refusal among the segments of one shell string, or None.

    Uses `protected_paths._tokens`/`_segments` — the reader the whole protected
    path subsystem shares, whose own docstring warns that a guard with its own
    idea of a token boundary misses the invocations that matter most. It already
    treats `$(…)`, backticks and newlines as command boundaries.
    """
    from app.harness.protected_paths import _segments, _tokens

    running = cwd
    try:
        segs = _segments(_tokens(text))
    except ValueError:      # pragma: no cover - the tokeniser's own fallback
        return None
    for seg in segs:
        if not seg:
            continue
        if seg[0] in ("cd", "pushd"):
            # A `cd` retargets every later segment: this is how `cd <live> && git
            # commit` becomes a write to production, and the same walk
            # `protected_paths._bash_write_targets` makes for the same reason.
            arg = next((t for t in seg[1:] if not t.startswith("-")), "")
            running = _norm(arg, running)
            continue
        hit = _segment_hit(list(seg), running, live, depth)
        if hit:
            return hit
    return None


def _unwrap(argv: list[str]) -> list[str]:
    """Strip wrappers and control words, leaving the program the shell runs.

    `env FOO=bar git commit` and `if git commit -m x` both arrive at `git`, which
    is the position the matcher reads. A wrapper's flag keeps the token that is
    its value — `timeout -k 5 30 git commit` must not leave `5` standing where the
    program belongs, because a leftover number is a segment that matches nothing
    and a missed write answers None.
    """
    argv = list(argv)
    while argv and (os.path.basename(argv[0]) in _WRAPPERS
                    or argv[0] in _CONTROL_WORDS):
        wrapper = os.path.basename(argv[0])
        argv = argv[1:]
        if wrapper not in _WRAPPERS:
            continue
        # The wrapper's own arguments: its flags, the value of a flag that takes
        # one, and `timeout`'s bare duration. All of it is dropped only in this
        # position — immediately after a wrapper was — which is what makes it safe
        # to drop a bare number: a program named `30` is not a thing, and outside
        # this branch no token is ever dropped for looking like a value.
        while argv and (argv[0].isdigit()
                        or (argv[0].startswith("-") and argv[0] != "-")):
            flag = argv[0]
            argv = argv[1:]
            if (flag in _WRAPPER_VALUE_FLAGS and argv
                    and not argv[0].startswith("-")):
                argv = argv[1:]
        # `env FOO=bar git …`: assignments are not commands.
        while argv and "=" in argv[0] and not argv[0].startswith("-"):
            argv = argv[1:]
    return argv


def _interp_script(argv: list[str]) -> str | None:
    """The script a shell's `-c`-family flag carries, or None.

    Read from the already-unwrapped argv, so the flag counts only in program
    position; a bare `bash script.sh` has no script argument and answers None.
    """
    for i, tok in enumerate(argv[1:], 1):
        if tok in _INTERP_SCRIPT_FLAGS and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _segment_hit(argv: list[str], cwd: str, live: str, depth: int = 0) -> str | None:
    """Refusal for the invocation one segment *runs*, or None.

    Program position only, which is the precision the corpus demands. A scan of
    the whole segment would refuse `ps aux | grep 'git commit'`, `awk
    '/git commit/'` and a `docker run` that names `git` twice — and the first two
    are how a turn looks for a commit, so a scan would have to guess which
    mentions of the word are data. `git add f && git commit` is two segments by
    the time this sees them, so each git is judged where it stands and the `add`
    is refused on its own merits rather than carrying the commit through.

    A shell wrapper's payload *is* walked, because `bash -c 'cd <live> && git
    commit'` runs that script; a non-shell interpreter's is not — see `_SHELLS`.
    """
    argv = _unwrap(list(argv))
    if not argv:
        return None
    head = os.path.basename(argv[0])
    if head in _SHELLS:
        script = _interp_script(argv)
        if script is None:
            return None
        return None if depth >= _MAX_WALK else _walk(script, cwd, live, depth + 1)
    if head == "xargs":
        # `xargs git commit`: the git xargs runs is this segment's program.
        return _segment_hit([t for t in argv[1:] if not t.startswith("-")],
                            cwd, live, depth)
    if head != "git":
        return None
    sub, nxt = _subcommand(argv)
    if not sub:
        return None
    target = _dash_c_target(argv[:nxt], cwd)
    if sub == _STASH:
        form = _stash_form(argv[nxt:])
        verb = f"stash {form}".strip()
        write = form not in _STASH_READ_FORMS
    else:
        verb, write = sub, sub in _MUTATING_SUBCOMMANDS
    if write and _is_live_root(target, live):
        return (f"`git {verb}` in the live checkout `{live}` — an unattended "
                f"session may not move HEAD, the index or the working tree there. "
                f"The source commits production (#2604): a turn commits in its "
                f"round's worktree, a scratch clone, or the vault, and the "
                f"reviewed doc reaches live through the source's own blob check. "
                f"Reads (`log`, `status`, `diff`, `show`, `grep`, `merge-base`, "
                f"`stash list`) are unaffected.")
    return None

def _dash_c_target(tokens: list[str], cwd: str) -> str:
    """Where a `git -C <dir>` runs, or `cwd` when it names no directory."""
    j = 1
    while j < len(tokens):
        tok = tokens[j]
        if not tok.startswith("-"):
            break
        flag = tok.partition("=")[0]
        if flag == "-C":
            if "=" in tok:
                return _norm(tok.partition("=")[2], cwd)
            return _norm(tokens[j + 1], cwd) if j + 1 < len(tokens) else cwd
        if flag in _GIT_VALUE_FLAGS and "=" not in tok:
            j += 2
            continue
        j += 1
    return cwd


def _stash_form(rest: list[str]) -> str:
    """The sub-form of a `git stash`, as one bare word — "" for the bare call.

    Bare and compared bare, never as a `stash <form>` label: this value is
    matched against `_STASH_READ_FORMS`, and a two-word label would match none of
    them, which is how `git stash list` — routine in the fleet corpus — gets read
    as a write.
    """
    for tok in rest:
        if not tok.startswith("-"):
            return tok
    return ""


def check_live_git_writes(command: str, session_id: str | None, *,
                          cwd: str | None = None,
                          live: str | None = None,
                          parent_of=None) -> str | None:
    """Reason to refuse `command`, or None.

    Returns a message, never a `(label, excerpt)` pair: `safety.check_bash_command`
    puts its own label on it, the way `check_service_control` and
    `check_install_provenance` do.

    `live` overrides the production tree, which is the only way to ask about a
    checkout other than the one this process boots from; `None` means
    `app.paths.LIVE_CHECKOUT`, imported inside the function for the reason
    `protected_paths` gives for the same choice — this module is imported by
    processes whose tree anchor differs, and a module-level import bakes the
    wrong answer in. `parent_of` resolves a `task:` subagent to its parent, the
    resolver the service-control lane also takes.
    """
    if not command or not isinstance(command, str):
        return None
    if attended_session(session_id, parent_of=parent_of):
        return None
    if live is None:
        from app.paths import LIVE_CHECKOUT
        live = str(LIVE_CHECKOUT)
    return _walk(command, str(cwd or os.getcwd()), os.path.normpath(live), 0)
