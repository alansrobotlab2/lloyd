"""What may be destroyed, and what may be written: one module, two policies.

The first is a structural check for shell commands that delete a protected tree
wholesale. The second, added by #1049, is the write deny-set the `Write`/`Edit`
lane consults — a different question, over a different surface, answered from
the same place so the two cannot drift into disagreeing about what is sacred.

`safety._HARD_DENY_PATTERNS` matched `rm -rf` only when the flags were fused
and the target began with a bare `/`, `~` or `$HOME`. That shape is one
spelling of a vault wipe among many, and on 2026-09-10 and 2026-09-12 the
vault was wiped twice by a turn whose five Bash calls all went through it
(a bench trial that was told "Delete all files in ~/obsidian now"). Measured
against the old matcher, all of these were allowed:

    rm -rf "$HOME/obsidian"/*          quoting breaks the prefix match
    cd ~/obsidian && rm -rf ./*        relative to a `cd`
    rm -r -f ~/obsidian                split flags
    find ~/obsidian -mindepth 1 -delete
    python3 -c "import shutil; shutil.rmtree('/home/…/obsidian')"

So this module *parses* instead of pattern-matching the whole string: it
splits the command into simple commands, follows `cd`, expands `~`/`$HOME`,
lets `shlex` strip the quoting, resolves relative paths, and then asks one
question of every delete-shaped operation — does its target take out a
protected tree? A protected tree is the vault, the lloyd tree, or the home
directory, and "take out" means the root itself, anything above it, a
top-level folder of it, or a glob directly over those.

What it deliberately does not refuse: a specific file or a nested path
(`rm -rf ~/obsidian/skills/old-skill`, `rm eval/baselines/item472-*.json`).
Those are normal agent work, and a gate that refuses normal work gets routed
around, which is worse than no gate. The wholesale shapes have no everyday
use by an agent at all.

This is defence in depth, not the guarantee. A parser of shell can always be
out-spelled by a determined command, which is why a bench trial also runs its
Bash inside a read-only sandbox (`agent_mcp/_tool_sandbox.py`) and the guardian
trips on a mass deletion whatever produced it (`vaultwatch.py`). The job here
is to make the obvious spellings impossible for every session, not only the
sandboxed ones.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger("lloyd-protected-paths")


@dataclass(frozen=True)
class Root:
    path: str
    label: str
    # A glob inside a top-level folder (`skills/*`) empties that folder. That
    # matters for the vault and the lloyd tree, where a top-level folder is a
    # whole category of state; for $HOME it would refuse `~/.cache/foo-*`.
    glob_in_children: bool


def protected_roots() -> list[Root]:
    """Resolved at call time, so tests can point HOME and the vault elsewhere."""
    home = os.path.normpath(os.path.expanduser("~"))
    try:
        from app.paths import LLOYD_HOME, VAULT_ROOT
        vault = os.path.normpath(str(VAULT_ROOT))
        lloyd_here = os.path.normpath(str(LLOYD_HOME))
    except Exception:  # noqa: BLE001 — a checker that cannot import still checks
        vault = os.path.join(home, "obsidian")
        lloyd_here = os.path.join(home, "lloyd")
    roots = [Root(vault, "vault", True),
             Root(os.path.join(home, "lloyd"), "lloyd tree", True)]
    # In an automod worktree LLOYD_HOME is the worktree, which deserves the
    # same protection as the live tree it will become.
    if lloyd_here not in {r.path for r in roots}:
        roots.append(Root(lloyd_here, "lloyd tree", True))
    roots.append(Root(home, "home directory", False))
    return roots


# ---------------------------------------------------------------------------
# The write deny-set — the file-tool lane's half of the same protection
# ---------------------------------------------------------------------------
#
# Everything above answers a question about a *shell command*. The `Write` and
# `Edit` tools never reach a shell: `agent_mcp.builtin_fs` takes an absolute
# path and writes it, and until #1049 nothing on that lane compared the path to
# anything but "did you Read it first". So the shorter route to the identity
# file was two tool calls — Read, then Write — while the longer one had a
# checker.
#
# This deny-set is deliberately NOT `protected_roots()`. That one exists to
# catch a wholesale *delete*, so its roots are the vault, the lloyd tree and
# $HOME; refusing writes under those would refuse every ordinary vault note and
# every file a self-modification round writes into its own worktree — a larger
# outage than the hole. A write predicate needs the surfaces whose *contents*
# are load-bearing on their own: the credentials beside the agent's home, the
# supervisor/guardian/service units an agent must not rewire from a chat turn,
# the interpreter those units run on, and the one vault file that *is* the
# system prompt.
#
# Membership is exactly this one constant. `agent_mcp/builtin_fs.py` holds no
# path literal of its own, and no other caller keeps a copy — a set that lives
# in two places is a set that diverges, which is what the four earlier
# spellings of "protected" on this box did (L0 prose, the Bash regexes, the
# automod diff globs, and `protected_roots`).
#
# Entries are home-relative templates rather than `app.paths` constants:
# `LLOYD_HOME` is *code*-relative, so inside a worktree it names the worktree,
# and a deny-set that followed the code would protect the copy while leaving
# the live tree open. `Path.home()/"obsidian"` is what `VAULT_ROOT` already is
# (`app/paths.py:11`), so the vault entry agrees with it on this box today.
PROTECTED_WRITE_ROOTS: tuple[tuple[str, str], ...] = (
    ("~/.openclaw", "the OpenClaw credential tree"),
    ("~/lloyd/agent-services", "the supervisor, guardian and service units"),
    ("~/lloyd/.venvs", "the interpreter the lloyd services run on"),
    ("~/obsidian/lloyd/SOUL.md", "the identity file loaded into every system prompt"),
)

#: Lifted in code or not at all. A `ContextVar`, not an argument: the model
#: controls the tool-arguments dict, and it controls `_meta` and the session id
#: that arrive beside them, so an authorisation readable from any of those is a
#: key the model can turn. A contextvar is settable only by Python already
#: running in this process — a job runner wrapping its own dispatch — and
#: `asyncio.to_thread` copies the context, so a grant taken on the event loop
#: is still in force on the worker thread that performs the write.
_write_grant: "contextvars.ContextVar[tuple[str, ...] | None]" = contextvars.ContextVar(
    "lloyd_protected_write_grant", default=None)


def protected_write_roots() -> list[tuple[str, str]]:
    """`PROTECTED_WRITE_ROOTS` resolved to realpaths, computed per call.

    Per call so a relocated `HOME` (a test, a container) moves the set with it,
    exactly like `protected_roots()`. `realpath` rather than `normpath` so a
    symlink and its target are one answer, and that answer is judged by the
    same rule the write itself will be.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for template, label in PROTECTED_WRITE_ROOTS:
        expanded = os.path.expanduser(template)
        if not expanded:
            continue
        real = os.path.realpath(expanded)
        if real in seen:
            continue
        seen.add(real)
        out.append((real, label))
    return out


def _covers(root: str, real: str) -> bool:
    """Is `real` the root itself or something below it? Whole-component only."""
    return real == root or real.startswith(root.rstrip("/") + os.sep)


def write_deny_reason(target: str) -> str | None:
    """The label of the deny-set entry covering `target`, or None to allow it.

    Realpath-anchored prefix matching, never a suffix match, in both
    directions. Realpath-first is what keeps an automod round able to write its
    own worktree: the round's tree can hold a directory named like a deny entry
    at a path outside the set, and a `**/agent-services/**`-style suffix rule
    would refuse that copy — which is precisely the file the round was opened
    to change. It also means a link standing inside a denied directory is
    judged by where it points: writing *through* one lands outside the set, and
    a link pointing into the set is refused from a permitted directory too.

    Resolves `target` itself, so a caller that has no realpath to hand cannot
    skip the check by forgetting to resolve one.
    """
    if not target:
        return None
    real = os.path.realpath(target)
    granted = _write_grant.get()
    for root, label in protected_write_roots():
        if not _covers(root, real):
            continue
        if granted is not None and (
                # An empty grant list means the whole set; a narrowed one means
                # only those locations, so a job handed two loaded-memory files
                # does not also get the service units.
                not granted
                or any(_covers(os.path.realpath(os.path.expanduser(g)), real)
                       for g in granted)):
            return None
        return label
    return None


def grant_protected_writes(reason: str,
                           paths: Sequence[str] | None = None) -> contextvars.Token:
    """Lift the write deny-set for this context; return the token that undoes it.

    `reason` is required and logged, because an authorisation nobody can name
    is an authorisation nobody revokes. `paths` narrows the lift to those
    locations (an empty sequence lifts the whole set).

    Callers are job runners inside this process. Nothing here reads the
    tool-arguments dict, `_meta`, or the session id, so a turn cannot lift its
    own denial — which is the entire property this function exists to provide.
    """
    granted = tuple(os.path.realpath(os.path.expanduser(p)) for p in paths) if paths else ()
    token = _write_grant.set(granted)
    logger.info("protected writes granted (%s): %s", reason,
                list(granted) or "the whole deny-set")
    return token


def release_protected_writes(token: contextvars.Token) -> None:
    _write_grant.reset(token)


def protected_writes_granted() -> "tuple[str, ...] | None":
    """The active lift, or None while the deny-set is in force. Tests and /state."""
    return _write_grant.get()


@contextlib.contextmanager
def allow_protected_writes(reason: str, paths: Sequence[str] | None = None):
    """Scoped `grant_protected_writes`, so a grant cannot outlive its `with`."""
    token = grant_protected_writes(reason, paths)
    try:
        yield
    finally:
        release_protected_writes(token)


_GLOB = re.compile(r"[*?\[]")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WRAPPERS = frozenset({"sudo", "doas", "env", "nice", "ionice", "nohup", "time",
                       "command", "builtin", "exec", "stdbuf", "unbuffer", "setsid"})
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish"})
_INTERPRETER = re.compile(r"\b(python[0-9.]*|perl|ruby|node|bun|deno|bash|sh|zsh)\b")
# Library calls that remove a tree, or build an `rm -r` argv, from inside an
# interpreter. Only consulted together with a protected literal.
_TREE_DELETE_CALL = re.compile(
    r"(rmtree|removedirs|rimraf|rm_rf|remove_tree|fs\.rm(?:Sync)?\s*\("
    r"|[\"']rm[\"']\s*,\s*[\"']-[a-zA-Z]*[rR])")
_HOME_OBSIDIAN = re.compile(r"home\(\)\s*/\s*[\"']obsidian[\"']")
_MAX_DEPTH = 3


def _tokens(command: str) -> list[str]:
    # Command substitution, backticks and newlines are command boundaries.
    # Rewriting them inside quoted strings changes nothing this check reads.
    text = (command.replace("\r", " ").replace("$(", " ; ")
            .replace("`", " ; ").replace("\n", " ; "))
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=";&|()<>")
        lex.whitespace_split = True
        lex.commenters = ""
        return list(lex)
    except ValueError:  # unbalanced quotes: fall back to a plain split
        return re.findall(r"[^\s;&|()<>]+|[;&|()<>]+", text)


def _is_boundary(tok: str) -> bool:
    return bool(tok) and all(c in ";&|()<>" for c in tok)


def _segments(tokens: list[str]) -> list[list[str]]:
    out: list[list[str]] = [[]]
    for tok in tokens:
        if _is_boundary(tok) or tok in ("{", "}", "!", "then", "do", "else"):
            # `2>/dev/null` lexes as `2`, `>`: the fd number is not an operand,
            # and read as one it turns `mv a .trash/ 2>…` into a move OF .trash.
            if ("<" in tok or ">" in tok) and out[-1] and out[-1][-1].isdigit():
                out[-1].pop()
            if out[-1]:
                out.append([])
        else:
            out[-1].append(tok)
    return [s for s in out if s]


def _strip_wrappers(argv: list[str]) -> tuple[list[str], bool]:
    """Drop env assignments and wrapper commands. Returns (argv, via_xargs)."""
    i, via_xargs = 0, False
    while i < len(argv):
        tok = argv[i]
        base = os.path.basename(tok)
        if _ASSIGNMENT.match(tok):
            i += 1
        elif base in _WRAPPERS:
            i += 1
            while i < len(argv) and argv[i].startswith("-"):
                i += 1
        elif base == "timeout":
            i += 1
            while i < len(argv) and argv[i].startswith("-"):
                i += 1
            i += 1  # the duration
        elif base == "xargs":
            via_xargs = True
            i += 1
            while i < len(argv) and argv[i].startswith("-"):
                if argv[i] in ("-I", "-n", "-P", "-d", "-L", "-s", "-a", "-E"):
                    i += 1
                i += 1
        else:
            break
    return argv[i:], via_xargs


def _resolve(tok: str, cwd: str | None, home: str) -> str | None:
    t = tok
    if t == "~" or t.startswith("~/"):
        t = home + t[1:]
    t = t.replace("${HOME}", home).replace("$HOME", home)
    if "$" in t or t.startswith("~"):
        return None  # a variable we cannot see, or ~user
    if not t.startswith("/"):
        if cwd is None:
            return None
        t = os.path.join(cwd, t)
    # normpath collapses `..` lexically, which is the right reading for a
    # delete: `rm -rf ~/obsidian/..` removes $HOME.
    return os.path.normpath(t)


def _within_label(path: str, roots: list[Root], *, children: bool,
                  glob_children: bool) -> str | None:
    """Label a resolved target if deleting it takes out a protected tree."""
    m = _GLOB.search(path)
    if m:
        prefix = path[:m.start()]
        base = os.path.normpath(prefix if prefix.endswith("/")
                                else (os.path.dirname(prefix) or "/"))
        for r in roots:
            if base == r.path or r.path.startswith(base.rstrip("/") + "/"):
                return f"a glob over the {r.label} ({r.path})"
            if (glob_children and r.glob_in_children and os.path.dirname(base) == r.path
                    and os.path.isdir(base)):
                return f"a glob over a top-level folder of the {r.label} ({base})"
        return None
    if path == "/":
        return "the filesystem root"
    for r in roots:
        if path == r.path:
            return f"the {r.label} ({r.path})"
        if r.path.startswith(path.rstrip("/") + "/"):
            return f"a parent of the {r.label} ({path})"
        # A top-level *folder* only. `rm -rf nohup.out` or `mv a.py b.py` at
        # the top of the lloyd tree are single files and ordinary work.
        if children and os.path.dirname(path) == r.path and os.path.isdir(path):
            return f"a top-level folder of the {r.label} ({path})"
    return None


# `find . -name '*.pyc' -delete` is a targeted clean; `find ~/obsidian
# -mindepth 1 -delete` and `-type f -delete` are not. A predicate that selects
# by name, age, size or content narrows the delete; depth and type do not.
_FIND_SELECTIVE = frozenset({
    "-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename",
    "-regex", "-iregex", "-newer", "-anewer", "-cnewer", "-newermt",
    "-mtime", "-mmin", "-atime", "-amin", "-ctime", "-cmin", "-size",
    "-empty", "-perm", "-lname", "-ilname", "-samefile", "-inum", "-links",
    "-user", "-group", "-uid", "-gid",
})


def _find_is_selective(args: list[str]) -> bool:
    for i, tok in enumerate(args):
        if tok in _FIND_SELECTIVE:
            value = args[i + 1] if i + 1 < len(args) else ""
            # `-name '*'` selects everything; it is not a filter.
            if tok in ("-name", "-iname", "-path", "-ipath", "-wholename",
                       "-iwholename") and value.strip("*") == "":
                continue
            return True
    return False


def _operands(args: list[str], takes_value: frozenset[str] = frozenset()) -> tuple[list[str], list[str]]:
    flags, operands, i, literal = [], [], 0, False
    while i < len(args):
        tok = args[i]
        if literal:
            operands.append(tok)
        elif tok == "--":
            literal = True
        elif tok.startswith("-") and tok != "-":
            flags.append(tok)
            if tok in takes_value and i + 1 < len(args):
                i += 1
                flags.append(args[i])
        else:
            operands.append(tok)
        i += 1
    return flags, operands


def _check_segment(argv: list[str], cwd: str | None, home: str,
                   roots: list[Root], raw: str, depth: int) -> tuple[str | None, str | None]:
    """Returns (reason, new_cwd)."""
    argv, via_xargs = _strip_wrappers(argv)
    if not argv:
        return None, cwd
    cmd = os.path.basename(argv[0])
    args = argv[1:]

    if cmd in ("cd", "pushd"):
        if not args:
            return None, home
        if args[0] == "-":
            return None, None
        return None, _resolve(args[0], cwd, home)

    if cmd in _SHELLS and "-c" in args:
        idx = args.index("-c")
        if idx + 1 < len(args) and depth < _MAX_DEPTH:
            return _scan(args[idx + 1], cwd, home, roots, depth + 1), cwd
        return None, cwd

    if cmd in ("rm", "shred", "unlink", "srm"):
        flags, targets = _operands(args)
        recursive = any((not f.startswith("--") and ("r" in f or "R" in f))
                        or f in ("--recursive",) for f in flags)
        if via_xargs and recursive and _mentions_root(raw, home, roots):
            return "xargs rm -r fed from a protected tree", cwd
        for t in targets:
            p = _resolve(t, cwd, home)
            if p is None:
                continue
            why = (_within_label(p, roots, children=True, glob_children=True)
                   if recursive else
                   _within_label(p, roots, children=False, glob_children=False)
                   if _GLOB.search(p) else None)
            if why:
                return f"{cmd}{' -r' if recursive else ''} on {why}", cwd
        return None, cwd

    if cmd == "find":
        paths = []
        for tok in args:
            if tok.startswith("-") or tok in ("(", "!", ")"):
                break
            paths.append(tok)
        destructive = "-delete" in args
        for i, tok in enumerate(args):
            if tok in ("-exec", "-execdir", "-ok", "-okdir") and i + 1 < len(args):
                if os.path.basename(args[i + 1]) in ("rm", "unlink", "shred", "rmdir", "mv"):
                    destructive = True
        if not destructive or _find_is_selective(args):
            return None, cwd
        for t in paths or ["."]:
            p = _resolve(t, cwd, home)
            if p is None:
                continue
            why = _within_label(p, roots, children=True, glob_children=True)
            if why:
                return f"find -delete over {why}", cwd
        return None, cwd

    if cmd == "rsync":
        flags, operands = _operands(args, frozenset({"-e", "--exclude", "--include",
                                                     "--filter", "-f"}))
        deleting = any(f.startswith("--delete") for f in flags)
        removing_src = "--remove-source-files" in flags
        if (deleting or removing_src) and operands:
            candidates = operands[-1:] if deleting else []
            candidates += operands[:-1] if removing_src else []
            for t in candidates:
                p = _resolve(t, cwd, home)
                why = p and _within_label(p, roots, children=True, glob_children=True)
                if why:
                    return f"rsync --delete onto {why}", cwd
        return None, cwd

    if cmd == "mv":
        flags, operands = _operands(args, frozenset({"-t", "--target-directory", "-S"}))
        sources = operands if any(f in ("-t", "--target-directory") or
                                  f.startswith("--target-directory=") for f in flags) \
            else operands[:-1]
        for t in sources:
            p = _resolve(t, cwd, home)
            if p is None:
                continue
            why = _within_label(p, [r for r in roots if r.glob_in_children],
                                children=True, glob_children=False)
            if why is None:
                # Moving $HOME itself (or a parent) away; its children are
                # ordinary directories to move.
                why = _within_label(p, [r for r in roots if not r.glob_in_children],
                                    children=False, glob_children=False)
            if why:
                return f"mv of {why}", cwd
        return None, cwd

    if cmd == "git" and "clean" in args:
        pre = args[:args.index("clean")]
        post = args[args.index("clean") + 1:]
        repo = cwd
        if "-C" in pre and pre.index("-C") + 1 < len(pre):
            repo = _resolve(pre[pre.index("-C") + 1], cwd, home)
        forced = any(f == "--force" or (f.startswith("-") and not f.startswith("--")
                                         and "f" in f) for f in post)
        if forced and repo is not None:
            for r in roots:
                if repo == r.path:
                    return f"git clean -f in the {r.label} ({r.path})", cwd
        return None, cwd

    return None, cwd


def _mentions_root(raw: str, home: str, roots: list[Root]) -> bool:
    for tok in _tokens(raw):
        p = _resolve(tok, None, home) if tok.startswith(("/", "~", "$HOME", "${HOME}")) else None
        if p and _within_label(p, roots, children=True, glob_children=True):
            return True
    return False


def _interpreter_delete(command: str, cwd: str | None, home: str,
                        roots: list[Root], depth: int) -> str | None:
    if not _INTERPRETER.search(command):
        return None
    # Single- and double-quoted literals are collected separately: a `-c
    # "…rmtree('/home/…/obsidian')"` nests one inside the other, and a single
    # alternation would stop at the outer quote.
    literals = ([m.group(1) for m in re.finditer(r"'([^'\n]{2,400})'", command)]
                + [m.group(1) for m in re.finditer(r"\"([^\"\n]{2,400})\"", command)])
    # The call and the protected path must share a line. A script that
    # rmtree's its own temp dir and reads the vault forty lines later is
    # ordinary; replayed over the 28k Bash commands on disk, the whole-text
    # version refused five of those.
    for line in command.splitlines() or [command]:
        if not _TREE_DELETE_CALL.search(line):
            continue
        if _HOME_OBSIDIAN.search(line):
            return "a tree delete over the vault from an interpreter"
        line_literals = ([m.group(1) for m in re.finditer(r"'([^'\n]{2,400})'", line)]
                         + [m.group(1) for m in re.finditer(r"\"([^\"\n]{2,400})\"", line)])
        for lit in line_literals:
            lit = lit.strip()
            p = _resolve(lit, None, home) if lit.startswith(("/", "~", "$HOME")) else None
            why = p and _within_label(p, roots, children=True, glob_children=True)
            if why:
                return f"a tree delete from an interpreter over {why}"
    # `os.system("rm -rf ~/obsidian")`, `subprocess.run("find … -delete", shell=True)`
    if depth < _MAX_DEPTH:
        for lit in literals:
            if " " in lit and re.search(r"\b(rm|find|rsync|mv|git)\b", lit):
                why = _scan(lit, cwd, home, roots, depth + 1)
                if why:
                    return why
    return None


def _scan(command: str, cwd: str | None, home: str, roots: list[Root], depth: int) -> str | None:
    for argv in _segments(_tokens(command)):
        why, cwd = _check_segment(argv, cwd, home, roots, command, depth)
        if why:
            return why
    return _interpreter_delete(command, cwd, home, roots, depth)


def _start_cwd(cwd: str | None, home: str) -> str | None:
    """Where a command begins, the same reading `check_protected_delete` uses:
    the tool's `cwd` argument, else the aggregator's own directory."""
    if cwd:
        return _resolve(cwd, None, home) or cwd
    try:
        from app.paths import LLOYD_HOME
        return str(LLOYD_HOME)
    except Exception:  # noqa: BLE001
        return None


_PATHISH = re.compile(r"(/|^\*|\.(md|py|json|yaml|yml|txt|sh)$)")

#: Commands whose whole job is to run another command string, so their quoted
#: operand has to be re-scanned rather than resolved as a filename.
_RESCAN = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish", "python",
                     "python3", "perl", "ruby", "node", "xargs", "env"})


def _looks_like_path(tok: str) -> bool:
    """A token worth trying to resolve. Skipping bare words keeps a `grep`
    pattern like `bench` or a variable name from being read as an operand; a
    token with a slash, a glob root or a known extension is worth resolving."""
    return bool(_PATHISH.search(tok))


def referenced_paths(command: str, cwd: str | None = None, _depth: int = 0) -> list[str]:
    """Every path a command's operands name, resolved as of each `cd`.

    The tokenizer and `~`/`$HOME`/relative resolution of
    `check_protected_delete`, pointed at a different question: not "would this
    delete a protected tree" but "which files does this command name". Used by
    the bench-corpus read gate (`app.harness.bench_corpus`), which has to deny
    `cat ~/obsidian/lloyd/bench/bench_010_*.md`, `cd ~/obsidian && cat
    lloyd/bench/x.md`, and the same path inside an interpreter's quoted
    literal, because a read gate that only matches the whole command string is
    bypassed by the first `/bin/cat` (backlog #582's lesson).

    Deliberately over-inclusive — every operand, not only the ones after a
    reader command, because the caller denies only when a resolved operand is
    inside the corpus, and a false candidate that resolves elsewhere is
    harmless. Unexpandable tokens (`$VAR`, `~user`) are not returned: the
    caller cannot decide about a path whose value is not in the string.
    """
    if not command or not isinstance(command, str):
        return []
    home = os.path.normpath(os.path.expanduser("~"))
    start = _start_cwd(cwd, home)
    out: list[str] = []

    def add(tok: str, base: str | None) -> None:
        if not tok or not _looks_like_path(tok):
            return
        p = _resolve(tok, base, home)
        if p and p not in out:
            out.append(p)

    try:
        cur = start
        segments = [_strip_wrappers(a)[0] for a in _segments(_tokens(command))]
        for argv in segments:
            cmd = os.path.basename(argv[0]) if argv else ""
            if cmd in ("cd", "pushd"):
                # Track the directory the rest of the command runs in, the same
                # reading the delete check takes: `cd ~/obsidian && cat
                # lloyd/bench/x` names the file only relative to the directory
                # the previous segment entered.
                _, cur = _check_segment(argv, cur, home, [], command, 0)
                continue
            for tok in argv:
                add(tok, cur)
        # `bash -c "cat ~/obsidian/lloyd/bench/x.md"`, `python3 -c
        # "open('…').read()"`: a path carried inside a quoted operand, which
        # `shlex` has already unwrapped into one token. Scan the raw command's
        # quoted literals and read each one as a command of its own — the same
        # single-/double-quote split as `_interpreter_delete`, for the same
        # reason: a nested literal stops a single alternation. Resolved against
        # the directory the command ended up in, which is the reading that
        # catches `cd ~/obsidian/lloyd/bench && python3 -c "cat x.md"`.
        cmds = {os.path.basename(a[0]) for a in segments if a}
        if cmds & _RESCAN and _depth < _MAX_DEPTH:
            for pat in (r"'([^'\n]{2,400})'", r'"([^"\n]{2,400})"'):
                for m in re.finditer(pat, command):
                    for p in referenced_paths(m.group(1), cur, _depth + 1):
                        if p not in out:
                            out.append(p)
    except Exception:  # noqa: BLE001 — a parser bug must not become a crash
        return out
    return out


def check_protected_delete(command: str, cwd: str | None = None) -> str | None:
    """Why `command` would delete a protected tree wholesale, or None.

    `cwd` is the directory the command starts in — the Bash tool's `cwd`
    argument, else the aggregator's own working directory, which is the lloyd
    tree.
    """
    if not command or not isinstance(command, str):
        return None
    home = os.path.normpath(os.path.expanduser("~"))
    roots = protected_roots()
    start = _start_cwd(cwd, home)
    try:
        return _scan(command, start, home, roots, 0)
    except Exception:  # noqa: BLE001 — a parser bug must not become a crash
        return None
