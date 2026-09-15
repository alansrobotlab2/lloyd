"""Tool calls may not change the Obsidian Sync registration.

The vault's only off-box copy is Obsidian Sync, run by the headless client
`agent-obsidian-sync`. `ob` keeps a vault's whole registration — `config.json`,
`state.db`, the stored E2E key — in one directory per *vault id*,
`~/.config/obsidian-headless/sync/<vaultId>/`, and `ob sync-unlink` deletes it
by vault id whatever `--path` it is handed. Re-creating it takes Alan's
end-to-end encryption password, which no agent has.

On 2026-09-14 Lloyd deleted it twice from one Mission Control chat (12:03 and
14:15 PDT, session `20260914_190323_iv2eca`) by running
`system_health_check.py --vault-sync-round-trip`: the probe links a scratch
client to the live vault id and unlinks it on the way out. The running client
kept syncing from memory; the next restart of the unit would have ended
replication until a human re-linked. Prose telling agents not to run it was
added to the skill afterwards. This is the part that does not depend on the
agent having read the skill.

Refused, for every session, at the same two enforcement points as the
wholesale-delete check (`safety.check_bash_command`):

* `ob` with any subcommand but the read-only ones (`sync-status`,
  `sync-list-local`, `sync-list-remote`, a bare `login` that only prints the
  login state, help and version). An allow list, not a deny list: `ob sync`
  starts a second client beside the live one, and `sync-create-remote` and the
  `publish-*` commands change the account.
* the health check's end-to-end leg: `--vault-sync-round-trip`, or
  `LLOYD_VAULT_SYNC_ROUND_TRIP` set truthy in the command.
* anything that writes, moves or deletes under `~/.config/obsidian-headless`.

Parsed, not pattern-matched, so `grep -n "sync-unlink" …` and reading the
config directory stay allowed — a text match on `sudo` refused grep and echo
lines and had to be dropped from the dispatch check. Best effort like its
neighbour: a determined spelling can get past a shell parser, which is why the
probe itself also refuses to touch a live registration (#1141).
"""

from __future__ import annotations

import os
import re

from app.harness.protected_paths import (_MAX_DEPTH, _SHELLS, _resolve, _segments,
                                         _strip_wrappers, _tokens)

#: `ob` subcommands that only read. Everything else changes the account, the
#: registration, or runs a second client.
OB_READ_ONLY = frozenset({"sync-status", "sync-list-local", "sync-list-remote",
                          "help", "--help", "-h", "--version", "-V"})
ROUND_TRIP_FLAG = "--vault-sync-round-trip"
ROUND_TRIP_ENV = "LLOYD_VAULT_SYNC_ROUND_TRIP"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

_OB_ENTRY = re.compile(r"(^|/)(ob|obsidian-headless/cli\.js)$")
_INTERPRETERS = re.compile(r"^(python[0-9.]*|node|bun|deno|perl|ruby)$")
# Inside interpreter code: `["ob", "sync-unlink"` or `"ob sync-setup …"`.
_OB_CALL_IN_CODE = re.compile(
    r"""["'](?:[^"'\s]*/)?ob["']\s*,\s*["']([a-z-]+)["']|["'](?:[^"'\s]*/)?ob\s+([a-z-]+)""")
_ROUND_TRIP_IN_CODE = re.compile(
    re.escape(ROUND_TRIP_FLAG) + r"|" + ROUND_TRIP_ENV
    + r"""["']?(?:\s*[\]:=,]\s*)+["']?(1|true|yes|on)\b""",
    re.IGNORECASE)
# A line of interpreter code only counts when it runs or configures something.
# A script that edits SKILL.md prose naming the flag is not running it —
# replayed over 29,687 Bash commands, the whole-text version refused two such.
_CALL_CONTEXT = re.compile(
    r"subprocess|os\.system|os\.popen|Popen|check_output|check_call|\brun\(|"
    r"\bexec[lv]p?e?\(|spawn|child_process|execSync|environ|putenv|\benv\s*=")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
# What `ob` actually names a subcommand. Prose in a string ("ob is short for")
# is not a call.
_OB_SUBCOMMAND = re.compile(r"^(login|logout|sync(-[a-z-]+)?|publish(-[a-z-]+)?)$")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DECLARERS = frozenset({"export", "declare", "typeset", "local", "readonly", "env"})
_WRITERS = frozenset({"rm", "rmdir", "unlink", "shred", "mv", "cp", "rsync", "tee",
                      "truncate", "ln", "install", "chmod", "chown", "touch", "dd",
                      "sqlite3", "sed", "perl"})
_SED_INPLACE = re.compile(r"^-[a-zA-Z]*i")


def registration_dir() -> str:
    return os.path.join(os.path.normpath(os.path.expanduser("~")), ".config", "obsidian-headless")


def _under_registration(tok: str, cwd: str | None, home: str) -> bool:
    if tok.startswith("-") and "=" in tok:
        tok = tok.split("=", 1)[1]
    p = _resolve(tok, cwd, home)
    if p is None:
        return False
    root = registration_dir()
    return p == root or p.startswith(root + os.sep)


def _ob_subcommand(args: list[str]) -> str:
    for a in args:
        if a.startswith("-") and a not in OB_READ_ONLY:
            continue
        return a
    return ""


def _ob_refusal(sub: str, args_after: list[str]) -> str | None:
    if not sub or sub in OB_READ_ONLY or not _OB_SUBCOMMAND.match(sub):
        return None
    if sub == "login" and not args_after:
        return None
    if "--help" in args_after or "-h" in args_after:
        return None
    # `sync-config --path X` with no change option prints the configuration.
    if sub == "sync-config" and not [a for a in args_after if a.startswith("-")
                                     and a != "--path" and not a.startswith("--path=")]:
        return None
    return f"`ob {sub}` changes the Obsidian Sync account or registration"


def _check_code(code: str) -> str | None:
    for line in code.splitlines() or [code]:
        if not _CALL_CONTEXT.search(line):
            continue
        for m in _OB_CALL_IN_CODE.finditer(line):
            sub = m.group(1) or m.group(2) or ""
            # Arguments are not parsed out of code: assume a change option.
            why = _ob_refusal(sub, ["--?"])
            if why:
                return why + " (from interpreter code)"
        if _ROUND_TRIP_IN_CODE.search(line):
            return "the vault-sync round trip (from interpreter code)"
    return None


def _split_heredocs(command: str) -> tuple[str, list[tuple[str, str]]]:
    """`(command without heredoc bodies, [(text before the <<, body)])`.

    A heredoc body is data for whatever reads it. Parsed as shell, a commit
    message line "…a continuous `ob sync` under autorestart…" became a command
    named `ob`."""
    lines = command.split("\n")
    kept: list[str] = []
    docs: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        kept.append(line)
        i += 1
        for m in _HEREDOC.finditer(line):
            if m.start() > 0 and line[m.start() - 1] == "<":   # `<<<` here-string
                continue
            term, body = m.group(2), []
            while i < len(lines) and lines[i].strip() != term:
                body.append(lines[i])
                i += 1
            i += 1
            docs.append((line[:m.start()], "\n".join(body)))
    return "\n".join(kept), docs


def _check_segment(argv: list[str], cwd: str | None, home: str,
                   depth: int) -> tuple[str | None, str | None]:
    # Environment first: `_strip_wrappers` drops assignments, and this one is
    # the switch. Only where it is set — leading assignments, and the operands
    # of `export`/`env`/`declare` — so `grep X=1 file` stays a search.
    assigned: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if _ASSIGN.match(tok):
            assigned.append(tok)
        elif os.path.basename(tok) in _DECLARERS:
            assigned.extend(t for t in argv[i + 1:] if _ASSIGN.match(t))
        else:
            break
        i += 1
    for tok in assigned:
        name, _eq, value = tok.partition("=")
        if name == ROUND_TRIP_ENV and value.strip("'\"").lower() in _TRUTHY:
            return f"{ROUND_TRIP_ENV}={value} turns on the vault-sync round trip", cwd
    stripped, _via_xargs = _strip_wrappers(argv)
    if not stripped:
        return None, cwd
    cmd = os.path.basename(stripped[0])
    args = stripped[1:]

    if cmd in ("cd", "pushd"):
        if not args:
            return None, home
        return None, (None if args[0] == "-" else _resolve(args[0], cwd, home))

    if cmd in _SHELLS and "-c" in args:
        idx = args.index("-c")
        if idx + 1 < len(args) and depth < _MAX_DEPTH:
            return _scan(args[idx + 1], cwd, home, depth + 1), cwd
        return None, cwd

    ob_entry = bool(_OB_ENTRY.search(stripped[0])) or stripped[0].startswith("$")
    if not ob_entry and cmd in ("node", "bun") and args and _OB_ENTRY.search(args[0]):
        ob_entry, args = True, args[1:]
    if ob_entry:
        sub = _ob_subcommand(args)
        # A variable (`"$OB" sync-setup`) counts only when what follows is an
        # `ob` subcommand, so `$EDITOR file` is not read as one.
        is_ob = not stripped[0].startswith("$") or sub.startswith(
            ("sync", "login", "logout", "publish"))
        rest = args[args.index(sub) + 1:] if sub in args else []
        why = _ob_refusal(sub, rest) if is_ob else None
        if why:
            return why, cwd

    if ROUND_TRIP_FLAG in args and cmd not in ("grep", "rg", "ugrep", "egrep", "fgrep",
                                                "echo", "printf", "cat", "less", "head",
                                                "tail", "git", "sed", "awk"):
        return "the vault-sync round trip (it deletes the live sync registration)", cwd

    if _INTERPRETERS.match(cmd):
        for flag in ("-c", "-e", "--eval"):
            if flag in args and args.index(flag) + 1 < len(args):
                why = _check_code(args[args.index(flag) + 1])
                if why:
                    return why, cwd

    writes = cmd in _WRITERS and not (
        (cmd == "sed" and not any(_SED_INPLACE.match(a) for a in args))
        or (cmd == "sqlite3" and "-readonly" in args))
    if cmd == "find":
        writes = any(a in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint")
                     for a in args)
    if writes and any(_under_registration(a, cwd, home) for a in args):
        return f"`{cmd}` on the Obsidian Sync registration ({registration_dir()})", cwd
    return None, cwd


def _scan(command: str, cwd: str | None, home: str, depth: int) -> str | None:
    command, heredocs = _split_heredocs(command)
    for intro, body in heredocs:
        # The body is code for an interpreter reading stdin, a script for a
        # shell reading stdin, and data for anything else.
        segs = _segments(_tokens(intro))
        reader = _strip_wrappers(segs[-1])[0] if segs else []
        name = os.path.basename(reader[0]) if reader else ""
        if _INTERPRETERS.match(name):
            why = _check_code(body)
        elif name in _SHELLS and "-c" not in reader and depth < _MAX_DEPTH:
            why = _scan(body, cwd, home, depth + 1)
        else:
            why = None
        if why:
            return why
    tokens = _tokens(command)
    # A redirect into the registration writes it whatever the command is.
    for i, tok in enumerate(tokens[:-1]):
        if tok in (">", ">>", ">|", "&>") and _under_registration(tokens[i + 1], cwd, home):
            return f"a redirect into the Obsidian Sync registration ({registration_dir()})"
    for argv in _segments(tokens):
        why, cwd = _check_segment(argv, cwd, home, depth)
        if why:
            return why
    return None


def check_sync_registration(command: str, cwd: str | None = None) -> str | None:
    """Why `command` would change the Obsidian Sync registration, or None."""
    if not command or not isinstance(command, str):
        return None
    home = os.path.normpath(os.path.expanduser("~"))
    start = _resolve(cwd, None, home) if cwd else None
    try:
        return _scan(command, start, home, 0)
    except Exception:  # noqa: BLE001 — a parser bug must not become a crash
        return None
