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
    data_roots: list[str] = []
    try:
        from app.paths import DATA_ROOT, LLOYD_HOME, PRODUCTION_DATA_ROOT, VAULT_ROOT
        vault = os.path.normpath(str(VAULT_ROOT))
        lloyd_here = os.path.normpath(str(LLOYD_HOME))
        data_roots = [os.path.normpath(str(PRODUCTION_DATA_ROOT)),
                      os.path.normpath(str(DATA_ROOT))]
    except Exception:  # noqa: BLE001 — a checker that cannot import still checks
        vault = os.path.join(home, "obsidian")
        lloyd_here = os.path.join(home, "lloyd")
    roots = [Root(vault, "vault", True),
             Root(os.path.join(home, "lloyd"), "lloyd tree", True)]
    # In an automod worktree LLOYD_HOME is the worktree, which deserves the
    # same protection as the live tree it will become.
    if lloyd_here not in {r.path for r in roots}:
        roots.append(Root(lloyd_here, "lloyd tree", True))
    # The runtime data root (sessions, databases, logs) lives outside the tree
    # since the 2026-09-22 deletion, so it is a root of its own: the account's
    # `~/lloyd-data`, and whatever this process resolved if that differs.
    # Its snapshots too: read-only subvolumes nothing here can delete, but a
    # `mv` of the directory holding them would still hide every one.
    for path in [os.path.join(home, "lloyd-data"), *data_roots,
                 os.path.join(home, ".lloyd-data-snapshots")]:
        if path not in {r.path for r in roots}:
            roots.append(Root(path, "lloyd data", True))
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
#
# The fifth entry exists because the fourth one's label promised more than the
# fourth one covered. `~/.local/share/uv/python` is the store the real
# interpreter lives in: every `.venvs/*/pyvenv.cfg` on this box names
# `home = ~/.local/share/uv/python/cpython-<ver>-linux-x86_64-gnu/bin`, and
# `.venvs/lloyd/bin/python` is a symlink into it, so realpath-first — the rule
# that lets a round write its own worktree — judged a write to that interpreter
# by where it *points*, outside the set, and allowed it. All five venvs
# (lloyd, qwen3-tts, vllm-decider, vllm-djev, vllm-flash-next-main) link the one
# binary, so that single allowed write was every service's interpreter at once.
# The entry stops at `uv/python` on purpose: `~/.local/share/uv/tools/supervisor`
# is a different uv tree, and automod's own promotion path execs
# `~/.local/share/uv/tools/supervisor/bin/supervisorctl`
# (`scripts/automod/promote.py:97`) — the broader entry would have put the
# parachute inside the deny-set. The legitimate writer, `uv python install`, is
# a human command (`SETUP.md:166`); an agent that needs it owes a grant, which is
# the point rather than an inconvenience to route around.
PROTECTED_WRITE_ROOTS: tuple[tuple[str, str], ...] = (
    ("~/.openclaw", "the OpenClaw credential tree"),
    ("~/lloyd/agent-services", "the supervisor, guardian and service units"),
    ("~/lloyd/.venvs", "the interpreter the lloyd services run on"),
    ("~/.local/share/uv/python",
     "the uv-managed CPython installs the lloyd venvs link to"),
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


# ---------------------------------------------------------------------------
# The write deny-set on the Bash lane — #1049's remaining hole, closed by #1620
# ---------------------------------------------------------------------------
#
# `agent_mcp/builtin_fs.py` asks `write_deny_reason` of every `Write` and
# `Edit`, so since #1049 that lane refuses the four entries of
# `PROTECTED_WRITE_ROOTS`. The Bash lane never asked the question at all: at
# HEAD `4bf96749` `safety.check_bash_command` ran four checks (the catastrophic
# regex table, `check_protected_delete`, `check_sync_registration`,
# `check_service_control`) and none of them resolves what a command *writes*,
# so `echo x > ~/.openclaw/credentials.json`, `echo x | tee -a
# ~/lloyd/agent-services/supervisord.conf` and `mv /tmp/evil
# ~/obsidian/lloyd/SOUL.md` were all allowed on both enforcement lanes while
# `Write` on the identical path was refused. Reading SOUL.md was shut and
# writing it was one Bash call.
#
# It parses rather than pattern-matches for the same reason the delete check
# does: the one Bash write refusal that predates this — the desktop lease's
# `(|tee|cp|mv|ln|install|rsync)…desktop/lease.json` — is a regex over a fixed
# verb list, and a regex cannot follow a `cd`, a `$HOME` spelling or an
# interpreter's quoted literal, which are the three spellings
# `check_protected_delete` exists to catch. So this reuses the module's
# tokenizer, `_strip_wrappers`, `_operands` and per-`cd` `_resolve`, exactly as
# the bench-corpus read gate reuses `referenced_paths`.
#
# Write-shaped targets only — never a source, never a reader's operand.
# `referenced_paths` is deliberately over-inclusive (its own docstring), and
# `cat ~/lloyd/agent-services/supervisord.conf` names a deny-set path as a
# matter of fact; denying on that output would refuse reading the supervisor
# conf, which is routine triage work and was this item's own first step. So the
# question here is which operand the command *writes*: a redirect's target, a
# mover's destination, `tee`'s files, a `sed -i` file operand, a `dd of=`, or a
# path sharing an interpreter's argument with a write-mode call. Operand
# *position* is load-bearing, not decoration: `sed -i 's/a/b/' notes.md` read as
# one blob hands back `<cwd>/s/a/b` — a path that exists nowhere — while the
# file it actually rewrites is the second positional.
#
# One deny-set, one predicate, both lanes: `write_deny_reason` is called on the
# resolved target exactly as the file lane calls it. That is what makes an
# `allow_protected_writes` lift cover Bash as well as `Write`, and it also means
# realpath semantics carry over — a symlink out of the set, like
# `.venvs/lloyd/bin/python` → the uv-managed interpreter, is outside the set
# here as it is on the file lane. Widening that is a membership decision (left
# to Alan by #1620), not a per-lane one.

#: Commands that write every file operand they are given.
_WRITE_EVERY_OPERAND = frozenset({"tee"})

#: Commands that write their last operand, or the `-t` target, given sources.
_WRITE_LAST_OPERAND = frozenset({"cp", "mv", "install", "ln", "rsync"})

#: Options that take a separate value on those commands, so the value is not
#: read as an operand: `cp -t DIR a b` writes DIR, `rsync -e ssh src dst`
#: writes dst. `-b`/`--backup` and `ln -s` take no value and are absent.
_MOVE_TAKES_VALUE = frozenset({
    "-t", "--target-directory", "-S", "--suffix",
    "-e", "--rsh", "--exclude", "--include", "--filter", "-f",
})

#: `sed` options that carry the script, which is then not a file operand.
_SED_SCRIPT_FLAGS = frozenset({"-e", "--expression", "-f", "--file"})

#: The quoted literals of an interpreter's argument are read out *whole*, by
#: `_payload_literals`, never by handing the payload to `referenced_paths` as one
#: blob: that merges adjacent literals — shlex strips the quotes and keeps the
#: comma, so `open('/a/x','w')` comes back naming `/a/x,w`, a path that exists
#: nowhere. It still trips a directory entry, which is how the redirect-shaped
#: case is caught either way, but it never equals the identity file, and a check
#: that only ever matched `SOUL.md,w` would let `open("<…>/SOUL.md","w")` through
#: while reporting a clean hit on the other spelling. Until #1740 this was two
#: regexes alternated over the payload, which could not see that a literal sat
#: inside another literal, and so could not tell a write call from one quoted as
#: data.

#: A call inside an interpreter's argument that writes, and which quoted literal
#: of that call is its destination — `_payload_literals` index 0 for
#: `open(p, "w")`, `Path(p).write_text(…)`, `fs.writeFileSync(p, …)`,
#: `File.write(p, …)`; index 1 for the two-argument movers, whose literal 0 is
#: the SOURCE. Position matters because `referenced_paths` does not know it: read
#: as a target, `shutil.copy("~/obsidian/lloyd/SOUL.md", "/tmp/backup")` — which
#: reads the identity file and writes elsewhere — would be refused for writing
#: it, and `cp`'s equivalent is refused nowhere.
#:
#: These patterns name CALLS, never paths: which path a call touches comes from
#: the literal, is resolved by `referenced_paths` exactly as every other operand
#: here is, and is decided by `write_deny_reason`. That is the discipline
#: `_interpreter_delete` applies to the same hole — inside `python3 -c "…"` there
#: is no argv position left to read — and it keeps two known limits: a call whose
#: destination is a variable or an f-string is not resolved (the payload's handle,
#: not its path, is what is visible), and a bare `.write(` is absent, because
#: `sys.stdout.write('/home/…/agent-services/x')` prints a path and writes
#: nothing.
_INTERPRETER_WRITE_CALLS: tuple[tuple[re.Pattern[str], int], ...] = (
    # open()/io.open() in a write mode, positional or `mode=`.
    (re.compile(r"\b(?:io\s*\.\s*)?open\s*\([^)]{0,200}?[\"'](?:mode\s*=\s*)?"
                r"[wax][+btr]*[\"']"), 0),
    # pathlib: the calls that create or replace bytes at a path.
    (re.compile(r"Path\s*\([^)]{0,200}?\)\s*\.\s*"
                r"(?:write_text|write_bytes|touch|mkdir)\s*\("), 0),
    # node, ruby, php: the writers that take a destination first. The node
    # methods are matched on the method alone, because the receiver in real code
    # is `require('fs')`, and a match that started there would put the literal
    # `fs` in the destination's position.
    (re.compile(r"\.(?:writeFileSync|appendFileSync|writeFile|appendFile)\s*\("
                r"|\bFile\s*\.\s*write\s*\(|\bfile_put_contents\s*\("), 0),
    # The two-argument movers: destination is the second literal. Not `rmtree`,
    # which is the delete axis and already answered elsewhere.
    (re.compile(r"\b(?:shutil\s*\.\s*(?:copy\w*|move\w*|copytree)"
                r"|os\s*\.\s*(?:replace|rename)|fs\s*\.\s*copyFileSync"
                r"|FileUtils\s*\.\s*cp)\s*\("), 1),
)


def _sed_inplace_files(args: list[str]) -> list[str]:
    """The files `sed -i` rewrites, or [] when it only prints to stdout."""
    in_place, script_flag, positionals, literal = False, False, [], False
    i = 0
    while i < len(args):
        tok = args[i]
        if literal:
            positionals.append(tok)
        elif tok == "--":
            literal = True
        elif tok.startswith("-") and tok != "-":
            if tok.startswith("-i") or tok.startswith("--in-place"):
                in_place = True
            if tok in _SED_SCRIPT_FLAGS or any(
                    tok.startswith(f + "=") for f in _SED_SCRIPT_FLAGS):
                script_flag = True
                i += 1
        else:
            positionals.append(tok)
        i += 1
    if not in_place:
        return []
    # With `-e`/`-f` every positional is a file; without it the first is the script.
    return positionals if script_flag else positionals[1:]


def _payload_literals(payload: str) -> list[tuple[int, int, str]]:
    """`(start, end, text)` for every string literal at the payload's *own* level.

    One unescape level has already been applied: `_tokens` ran `shlex` over the
    command, which strips the quotes around the payload and leaves `\\\"` inside
    it as `\"`. That one level is exactly what separates code from data. A
    case-table entry spelling out another command arrives as
    `"python3 -c \"open('/home/…/.openclaw/x','w')\""` — its inner quotes are
    still escaped, so this scan sees a single string body spanning the whole
    entry, and the `open(` inside it is text. The same call written at the
    payload's statement level is `open('/home/…/.openclaw/x','w')`, whose
    literals are code and whose path is a destination. `re.finditer` over the
    payload cannot tell those two apart, which is the mechanism behind #1740's
    fourth false positive.

    The limits are under-blocks, not blocks: an apostrophe in prose inside a
    payload (`# don't`) has no closing quote on its line and is skipped, and a
    raw-string body containing a quote is read as though it were escaped.
    """
    out: list[tuple[int, int, str]] = []
    i, n = 0, len(payload)
    while i < n:
        if payload[i] not in "\"'":
            i += 1
            continue
        close = payload[i:i + 3] if payload[i:i + 3] in ('"""', "'''") else payload[i]
        body = i + len(close)
        j = body
        while j < n:
            if payload[j] == "\\":
                j += 2                       # an escaped quote does not close it
                continue
            if payload.startswith(close, j):
                break
            if len(close) == 1 and payload[j] == "\n":
                break
            j += 1
        if j >= n or not payload.startswith(close, j):
            i += 1                      # a stray quote in prose: not a literal
            continue
        out.append((i, j + len(close), payload[body:j]))
        i = j + len(close)
    return out


def _interpreter_destinations(payload: str) -> list[str]:
    """The paths the write calls in one interpreter argument would land on.

    Each call is read in its own window, from the call to the end of its
    statement, so a payload holding two writes never hands the second one's
    literal to the first, and the literal read is the one that position names.

    Only a call at the payload's own statement level counts. A write call quoted
    *inside* a string of the payload is data the payload merely carries — a case
    table of commands to probe, a patch hunk, a log line to match — and reporting
    its path as a write target is what refused three Bash calls in the very
    session that authored this guard (#1740).
    """
    out: list[str] = []
    regions = _payload_literals(payload)
    for pattern, index in _INTERPRETER_WRITE_CALLS:
        for match in pattern.finditer(payload):
            if any(start <= match.start() < end for start, end, _t in regions):
                continue                        # quoted inside a string: data
            window = payload[match.start():]
            cut = min((window.find(sep) for sep in (";", "\n")
                       if window.find(sep) != -1), default=len(window))
            limit = match.start() + (cut or len(window))
            literals = [text for (start, _end, text) in regions
                        if match.start() <= start < limit
                        and "\n" not in text]   # a multi-line body is never a destination
            if len(literals) > index and literals[index] not in out:
                out.append(literals[index])
    return out


#: A heredoc operator, in its four spellings: `<<WORD`, `<<-WORD`, `<<'WORD'`
#: and `<<"WORD"`. `<<<` (bash's here-string) cannot match, because the word has
#: to start with a letter or underscore and `<` is neither — a here-string is one
#: word on the operator's own line, which the shell grammar already reads.
_HEREDOC_OP = re.compile(r"<<(?P<dash>-)?[ \t]*(?P<q>['\"]?)(?P<word>[A-Za-z_]\w*)(?P=q)")


def _heredoc_end(lines: list[str], start: int, word: str, dash: bool) -> int | None:
    """The index of the line carrying the delimiter, or None if there is none.

    A missing terminator means *no heredoc*, so a `<<` that is only text — inside
    a quoted string, or an unbalanced one the shell would have rejected — never
    swallows the rest of the command and quietly under-blocks it.
    """
    pat = re.compile(rf"^[ \t]*{re.escape(word)}\b")
    for k in range(start, len(lines)):
        if pat.match(lines[k].lstrip(" \t") if dash else lines[k]):
            return k
    return None


def _heredoc_recipient(prefix: str) -> str:
    """The bare command name a heredoc body belongs to, '' if there is none.

    Read off the operator's own line, after the last shell boundary and through
    any `FOO=bar` prefix or wrapper, so `env python3 - <<EOF` is recognised as
    python's payload and not `env`'s.
    """
    seg = re.split(r"[;&|()\n]", prefix)[-1].strip()
    if not seg:
        return ""
    argv, _via_xargs = _strip_wrappers(_tokens(seg))
    return os.path.basename(argv[0]) if argv else ""


def _split_heredocs(command: str) -> "tuple[str, list[str]]":
    """`(text for the shell grammar, interpreter bodies)` for one command.

    A heredoc body is program text or data, never a command line, and `_tokens`
    rewriting every `\\n` to ` ; ` meant the shell grammar parsed it as a stream
    of them: `cat > /tmp/probe.py <<'EOF'` with one `echo x > <deny-path>` line
    inside came back naming that path as a redirect target, and a `>` in a JS or
    CSS line read as a redirect too (#1740, three refusals in the guard's own
    authoring session). So a body leaves the shell stream here, and what happens
    to it depends on what was handed it:

    * a **shell** (`bash`, `sh`, `zsh`, `dash`, `ksh`, `fish`) — the body *is*
      shell code, so it stays in the stream and keeps being parsed as before;
    * a **language with write-call patterns** (`python3`, `perl`, `node`, …) —
      the body is a payload, and it is returned to be read by
      `_interpreter_destinations`, for its write calls only. This is also what
      shuts the hole #1740 measured open: `python3 - <<'EOF'` with
      `open('<deny>/x','w')` in it used to return None, because the body reached
      neither the operand list (stdin is not an operand) nor a write-call scan;
    * **anything else** (`cat`, `tee`, `gcc`) — the body is data. It is dropped,
      and the command keeps only what the operator's own line spells, which is
      where a genuine `cat > <deny-path>` still refuses.

    Multiple heredocs on one line are read one at a time, and a body that follows
    an operator this pass did not consume keeps its current (parsed-as-shell)
    behaviour rather than losing a refusal.
    """
    lines = command.split("\n")
    out: list[str] = []
    payloads: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _HEREDOC_OP.search(line)
        if not match:
            out.append(line)
            i += 1
            continue
        word, dash = match.group("word"), bool(match.group("dash"))
        end = _heredoc_end(lines, i + 1, word, dash)
        if end is None:
            out.append(line)
            i += 1
            continue
        recipient = _heredoc_recipient(line[:match.start()])
        body = "\n".join(lines[i + 1:end])
        if recipient in _SHELLS:
            out.append(line)                     # a shell body is shell code
            out.extend(lines[i + 1:end + 1])
        else:
            # Both ends of the splice survive: what followed the operator on its
            # own line (`cat <<EOF | tee target`) is a command, and so is what
            # followed the delimiter (`…\nEOF && echo done`).
            tail = re.match(rf"[ \t]*{re.escape(word)}\b(.*)$", lines[end])
            out.append(line[:match.start()] + line[match.end():]
                       + (tail.group(1) if tail else ""))
            if _INTERPRETER.match(recipient):
                payloads.append(body)            # data for `cat` is simply dropped
        i = end + 1
    return "\n".join(out), payloads


def _move_destination(flags: list[str], operands: list[str]) -> str | None:
    """What a mover writes: its `-t` target, else its last operand.

    The last-operand reading is what keeps a `cp`/`mv` *source* out of the
    answer — `cp ~/lloyd/agent-services/supervisord.conf /tmp/backup` copies
    FROM the deny-set and must stay allowed, and it is: the destination there is
    `/tmp/backup`.
    """
    for j, f in enumerate(flags):
        if f in ("-t", "--target-directory") and j + 1 < len(flags):
            return flags[j + 1]
        if f.startswith("--target-directory="):
            return f.split("=", 1)[1]
    return operands[-1] if len(operands) >= 2 else None


def write_targets(command: str, cwd: str | None = None,
                  _depth: int = 0) -> list[tuple[str, str]]:
    """Every `(path, shape)` the command writes to, resolved as of each `cd`.

    `shape` is a phrase for the refusal — "a redirect into", "a `cp` writing" —
    so the deny reason names the spelling that was caught, which is what makes
    a false block diagnosable from the refusal log alone. Relative operands
    resolve against the directory the command has `cd`-ed to, an unexpandable
    token (`$VAR`, `~user`) is not returned at all, and a shell or interpreter
    argument is re-entered once per level up to `_MAX_DEPTH`, which is how
    `bash -c 'echo x > ~/.openclaw/x'` is caught as well as the plain form.
    """
    if not command or not isinstance(command, str):
        return []
    # Program text comes out of the shell stream first: see `_split_heredocs`.
    command, payloads = _split_heredocs(command)
    home = os.path.normpath(os.path.expanduser("~"))
    state = {"cwd": _start_cwd(cwd, home)}
    out: list[tuple[str, str]] = []

    def keep(path: str, shape: str) -> None:
        if path and (path, shape) not in out:
            out.append((path, shape))

    def add_token(tok: str, shape: str) -> None:
        if not tok:
            return
        path = _resolve(tok, state["cwd"], home)
        if path:
            keep(path, shape)

    def visit(argv: list[str]) -> None:
        argv, _via_xargs = _strip_wrappers(argv)
        if not argv:
            return
        cmd = os.path.basename(argv[0])
        args = argv[1:]
        if cmd in ("cd", "pushd"):
            # Track the directory the rest of the command runs in, the same
            # reading `referenced_paths` and `check_protected_delete` take, so
            # `cd ~/lloyd && sed -i x agent-services/supervisord.conf` names the
            # file it rewrites and not a path relative to where the tool started.
            _, state["cwd"] = _check_segment(argv, state["cwd"], home, [], command, 0)
            return
        # The mover option table applies to the movers only: `-e` takes a value
        # for `rsync` (its remote shell) and carries the program for `node` and
        # `perl`, and read as a value it would swallow the payload the
        # interpreter branch below exists to read. For everything else an option
        # is just an option, and whatever is not one is an operand.
        takes = _MOVE_TAKES_VALUE if cmd in _WRITE_LAST_OPERAND else frozenset()
        flags, operands = _operands(args, takes)
        if cmd in _WRITE_EVERY_OPERAND:
            for tok in operands:
                add_token(tok, "a `tee` writing")
        elif cmd in _WRITE_LAST_OPERAND:
            dest = _move_destination(flags, operands)
            if dest:
                add_token(dest, f"a `{cmd}` writing")
        elif cmd == "sed":
            for tok in _sed_inplace_files(args):
                add_token(tok, "a `sed -i` rewriting")
        elif cmd == "dd":
            for tok in args:
                if tok.startswith("of="):
                    add_token(tok[3:], "a `dd of=` writing")
        if _INTERPRETER.match(cmd) or cmd in _RESCAN:
            if _depth >= _MAX_DEPTH:
                return
            # The payload is whatever the interpreter was handed: `-c` reads as a
            # flag to `_operands`, so its argument lands in `operands` beside the
            # script-file operands of a `sh file.sh`.
            for tok in operands:
                for path, shape in write_targets(tok, state["cwd"], _depth + 1):
                    keep(path, shape)
                for literal in _interpreter_destinations(tok):
                    # Resolved by the same function that resolves every other
                    # operand here, so a `~/…` inside a Python string lands on the
                    # same deny-set entry the shell spelling does.
                    for path in referenced_paths(literal, state["cwd"], _depth + 1):
                        keep(path, "a write from an interpreter to")

    try:
        tokens = _tokens(command)
        argv: list[str] = []
        redirect_out = False
        skip_next = False
        for tok in tokens:
            if not _is_boundary(tok):
                if skip_next:
                    skip_next = False
                    argv.append(tok)      # a heredoc word or an fd: not a target
                    continue
                if redirect_out:
                    redirect_out = False
                    add_token(tok, "a redirect into")
                    argv.append(tok)
                    continue
                argv.append(tok)
                continue
            if tok in (">", ">>", ">|"):
                # `2>/dev/null` lexes as `2`, `>`: the fd number is not an
                # operand, and leaving it in argv would read it as one.
                if argv and argv[-1].isdigit():
                    argv.pop()
                redirect_out = True
            elif tok == "<&" or tok == ">&" or tok == "&>":
                # `>&2` duplicates a descriptor; `&>` is bash's redirect-out,
                # whose target is the next token, so only the fd forms are skipped.
                redirect_out = (tok == "&>")
                skip_next = not redirect_out
            elif tok.startswith("<"):
                skip_next = True         # a redirect IN names a source, not a target
            else:
                visit(argv)
                argv = []
        visit(argv)
        # A payload fed on stdin gets the same write-call reading a `-c` operand
        # gets, and only that reading: `python3 - <<'EOF'` whose body opens a
        # deny-set path for writing used to return None, because stdin is not an
        # operand and nothing scanned the body for write calls (#1740 clause 2).
        for body in payloads:
            for literal in _interpreter_destinations(body):
                for path in referenced_paths(literal, state["cwd"], _depth + 1):
                    keep(path, "a write from an interpreter to")
    except Exception:  # noqa: BLE001 — a parser bug must not become a crash
        return out
    return out


def check_bash_write_denied(command: str, cwd: str | None = None) -> str | None:
    """Why `command` would write into the write deny-set, or None.

    The Bash half of `write_deny_reason`, for the lane that reaches a shell
    instead of a path argument: it names the shape and the resolved target and
    then refuses on exactly the predicate the file lane refuses on, grant
    contextvar included. A reader of a deny-set path is not a target of this
    check at all — see the block comment above.
    """
    if not command or not isinstance(command, str):
        return None
    try:
        targets = write_targets(command, cwd)
    except Exception:  # noqa: BLE001
        return None
    for path, shape in targets:
        label = write_deny_reason(path)
        if label:
            return f"{shape} {path} — inside {label}"
    return None


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
