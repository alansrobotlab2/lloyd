"""Structural check for shell commands that delete a protected tree wholesale.

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

import os
import re
import shlex
from dataclasses import dataclass


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
    start = cwd
    if start is None:
        try:
            from app.paths import LLOYD_HOME
            start = str(LLOYD_HOME)
        except Exception:  # noqa: BLE001
            start = None
    elif start:
        start = _resolve(start, None, home) or start
    try:
        return _scan(command, start, home, roots, 0)
    except Exception:  # noqa: BLE001 — a parser bug must not become a crash
        return None
