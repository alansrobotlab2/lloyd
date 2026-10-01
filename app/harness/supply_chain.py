"""Where a package came from: the two supply-chain checks #688 ships.

Two questions no other rung asks. The landing gate's ten rungs are all
*correctness* checks (`scripts/automod/gate.py`); none of them asks whether the
thing a change installed is known to be vulnerable, and none asks whether the
name was publishable by an attacker this morning. A test-based gate is
structurally unable to see the second one: the slop-squat attack (a model
hallucinates `graphy`, an attacker publishes that exact name) builds, runs and
passes tests — the detection signature is registry *freshness*, not a failing
build.

Three facts, one policy
-----------------------
A distribution name that is *new to the repo's dependency set* must clear all
three before a background turn may install it (:data:`DEFAULT_THRESHOLDS`):

1. **it exists** on the registry. A name that answers 404 is not "unverified",
   it is the attack's own signature, so it is refused rather than allowed.
2. **its first release is at least 90 days old** (:data:`MIN_FIRST_RELEASE_AGE_DAYS`).
3. **it has at least 2 releases** (:data:`MIN_RELEASE_COUNT`).

Each refusal names which fact failed and by how much, because a block message
that does not say what failed gets overridden reflexively and the check becomes
theatre (the item's own stated risk). The override is
:data:`OVERRIDE_ENV` as an env-assignment prefix carrying a reason, so it is
visible in the command text, in the transcript and in the log — a
denial-by-default with a *cheap* escape, not a wall.

Why the refusal is scoped to background sessions
------------------------------------------------
`service_control` refuses an engine restart to a worker and not to a person's
chat, and the same split applies here: the item's premise is "nobody reads a
`pip install` line at 04:00", which is about *unattended* turns. A chat turn has
its author reading it, and that author is the override this design says must
stay cheap. A call with no session id is refused: #1053 established that on this
box the *absence* of a session id is otherwise itself the bypass.

Fail-open, deliberately, in exactly one place: when the registry cannot be
consulted (no network, DNS, timeout, non-200-non-404). An outage is not a fact
about a package, and a guard that turns a network blip into a denial is a guard
that gets its bypass memorised. Every fail-open is logged and counted.

The scan half is offline by construction
----------------------------------------
`run_offline_scan` invokes a *locally installed* scanner with its no-network
flag and a local database directory, so a scan makes zero network calls
(:data:`SCANNER_SPECS`). It never invents a verdict: when no scanner resolves it
returns a reason naming the binaries it looked for and writes nothing — in
particular it never writes `advisory_count: 0`, which is the "clean bill of
health from an instrument that never ran" failure this file's own history warns
about. A run with a scanner present records the scanner version, the database
freshness, the duration and the advisory count to `eval/supply-chain/`.

Scope notes (measured 2026-09-24, recorded on backlog #688)
-----------------------------------------------------------
* Enforced ecosystem is **PyPI**, because that is the ecosystem the repo's
  dependency set (`requirements.txt`, `requirements.lock`,
  `requirements-dev.txt`) describes and the one the fixture control set can
  measure. `npm install` / `cargo add` / `poetry add` of a non-Python name are
  *recognised* and logged, not blocked: their registry facts
  (`registry.npmjs.org`, `crates.io`) are a different payload, and the measured
  traffic for them on this box is one `npm install` in 4,735 autonomy run
  records. That gap is written up on the item rather than silently closed.
* One live registry request per *new* name. Only the advisory half is fully
  local; the provenance half needs a registry, and the item's human-decision
  list names accepting that GET.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# The repo's dependency set — what counts as "new"
# ---------------------------------------------------------------------------

#: The files that declare what this repo installs, in the order they are read.
#: `requirements-dev.txt` is a declared dependency set but never an install
#: target (see `spec.py`'s note on #1073 and
#: `test_the_dev_requirements_file_is_never_an_install_target`), which is why
#: the gate's venv rung ignores it and this reader must not.
DEPENDENCY_SET_FILES: tuple[str, ...] = (
    "requirements.txt",
    "requirements.lock",
    "requirements-dev.txt",
)

#: A distribution name per PEP 503/426: letters, digits, `.`, `-`, `_`, never
#: leading or trailing punctuation. Anything else an operand can be — a wheel
#: path, a `git+https://` URL, a local directory — is not a registry name.
_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]{0,99}?)(?![A-Za-z0-9._-])")
_EXTRAS = re.compile(r"\[.*\]")


def normalize_dist_name(name: str) -> str:
    """PEP 503 normalisation: lowercase, each run of `-_.` collapsed to `-`.

    `Youtube-Transcript_API`, `youtube.transcript.api` and
    `youtube-transcript-api` are one project to PyPI, and a membership test that
    skips this is a false "new package" on every spelling change.
    """
    return re.sub(r"[-_.]+", "-", str(name or "")).strip().lower()


def _requirements_names(text: str) -> list[str]:
    """Distribution names declared by requirements-file *text*.

    Handles comments, line continuations, `-r`/`-e`/`--hash` option lines,
    environment markers, extras, and `name @ url` direct references. A VCS or
    path reference (`git+https://…`, `./pkg`, `file://…`) contributes nothing:
    it names no registry project, and guessing a name out of a repo URL is how a
    control-set reader starts classifying real dependencies as new.
    """
    names: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split(" #", 1)[0].strip().rstrip("\\").strip()
        if not line or line.startswith("-"):
            continue
        if "+://" in line or line.startswith(("file:", ".", "/", "~", "$", "-")):
            continue
        body = line.split(" @ ", 1)[0] if " @ " in line else line
        body = _EXTRAS.sub("", body).strip()
        m = _NAME_RE.match(body)
        if m:
            names.append(m.group(1))
    return names


def read_dependency_set(root: Path | str | None = None,
                        *, files: Sequence[str] = DEPENDENCY_SET_FILES) -> dict[str, str]:
    """Every distribution name the repo declares, mapped to the file that says so.

    Read from the files on disk, per call, never from a constant: clause 5 of
    #688 is that a name already listed is *never* classed as new, and a
    hard-coded list would be false the moment a human adds a dependency (which
    this box does — five `requirements.txt` commits in the 30 days to 2026-09-18,
    all of them human).
    """
    base = _repo_root(root)
    out: dict[str, str] = {}
    for rel in files:
        path = base / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name in _requirements_names(text):
            out.setdefault(normalize_dist_name(name), rel)
    return out


def _repo_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    try:
        from app.paths import LLOYD_HOME
        return Path(LLOYD_HOME)
    except Exception:  # noqa: BLE001 — a checker still checks without app.paths
        return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Install-command parsing — parsed, never substring-matched
# ---------------------------------------------------------------------------

#: Program + subcommand pairs that pull a distribution from a registry. The
#: item names the verbs; the parser keys on (program, subcommand) so a quoted
#: mention of the verb inside `grep`/`echo` text is an argument to a different
#: program and contributes nothing.
_INSTALL_FORMS: dict[str, tuple[str, ...]] = {
    "pip": ("install",),
    "pip3": ("install",),
    "pipx": ("install",),
    "uv": ("add", "install", "sync"),
    "uvx": (),                      # `uvx <pkg>` resolves and caches <pkg>
    "poetry": ("add",),
    "npm": ("install", "i", "add", "update"),
    "pnpm": ("add", "install"),
    "yarn": ("add",),
    "bun": ("add", "install"),
    "cargo": ("add", "install"),
}

#: Options that consume the token after them. Without this list `-r
#: requirements.txt` would report `requirements.txt` as a distribution name.
_VALUE_OPTIONS: dict[str, frozenset[str]] = {
    "pip": frozenset({"-r", "--requirement", "-c", "--constraint", "-e", "--editable",
                      "-i", "--index-url", "--extra-index-url", "-f", "--find-links",
                      "-t", "--target", "-b", "--build", "-d", "--download", "--log",
                      "--proxy", "--client-cert", "-C", "--config-settings",
                      "--install-option", "--global-option", "--python",
                      "--implementation", "--abi", "--platform", "--report"}),
    "uv": frozenset({"-r", "--requirements", "-c", "--constraint", "-e", "--editable",
                     "--index-url", "--extra-index-url", "--override", "--exclude-newer",
                     "--python", "--directory", "-C", "--index", "--default-index",
                     "--requirements-txt", "--group", "--package", "--project",
                     "--install-dir", "--with", "--from", "--index-name"}),
    "npm": frozenset({"-w", "--workspace", "--prefix", "-C", "--cache", "--registry",
                      "--package-lock-path", "--omit", "--include", "--alias",
                      "--before", "--global", "--location"}),
    "poetry": frozenset({"--group", "-G", "--source", "--directory", "--project",
                         "--lock", "--extras", "-E"}),
    "cargo": frozenset({"--path", "--registry", "--rename", "-F", "--features",
                        "--git", "--tag", "--branch", "--rev", "--index", "--target-dir"}),
    "pnpm": frozenset({"-C", "--dir", "--filter", "--registry", "--save-prefix"}),
    "yarn": frozenset({"-W", "--registry", "--exact"}),
    "bun": frozenset({"-F", "--filter", "--cwd"}),
}

_Ecosystem_PYPI = "pypi"
_Ecosystem_NPM = "npm"
_Ecosystem_CARGO = "cargo"
_ECOSYSTEM_BY_PROGRAM = {
    "pip": _Ecosystem_PYPI, "pip3": _Ecosystem_PYPI, "pipx": _Ecosystem_PYPI,
    "uv": _Ecosystem_PYPI, "uvx": _Ecosystem_PYPI, "poetry": _Ecosystem_PYPI,
    "npm": _Ecosystem_NPM, "pnpm": _Ecosystem_NPM, "yarn": _Ecosystem_NPM,
    "bun": _Ecosystem_NPM, "cargo": _Ecosystem_CARGO,
}

#: Heads worth re-parsing when one turns up inside a quoted string: the install
#: programs themselves, plus the interpreters that take a command line as an
#: argument. Anything else in a literal is prose, and prose stays out of the check —
#: that boundary is what keeps `echo "pip install evilpkg"` unflagged.
_NESTABLE_HEADS = frozenset(_INSTALL_FORMS) | {"python", "python3", "bash", "sh", "env"}

#: Prefix carried on the command itself to override a provenance refusal. Read
#: from the env assignments *before* `_strip_wrappers` discards them, so the
#: override has to appear where a reader and the transcript both see it.
OVERRIDE_ENV = "LLOYD_DEP_OVERRIDE"

_PYTHONS = re.compile(r"^(python|python[0-9.]+|pypy[0-9.]+)$")
_EGG_RE = re.compile(r"[#&]egg=([A-Za-z0-9][A-Za-z0-9._-]*)")
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_MAX_DEPTH = 2


@dataclass(frozen=True)
class InstallRequest:
    """One distribution an install command asks the registry to hand over."""

    name: str                 # as spelled on the command line
    ecosystem: str            # "pypi" | "npm" | "cargo"
    verb: str                 # "pip install", "uv add", "npm install", …
    vetted: bool = True       # False = no registry facts exist for this ecosystem
    vcs: bool = False         # came from a git+/`@ url` reference, not a registry name
    #: The `LLOYD_DEP_OVERRIDE` value carried by *this* command segment: None =
    #: not given, "" = given without a reason, text = a stated reason. It travels
    #: with the request rather than with the whole command because an override is
    #: a decision about one install — `OVERRIDE=why pip install a && pip install b`
    #: licences `a` and not `b`, the same scoping `env KEY=val cmd` already has.
    override: str | None = None
    #: The `-r <file>` this name was read out of, "" for a name on the command
    #: line. The decision is per name either way; the journal uses this to write
    #: a 180-name lockfile as one entry instead of 180 (#1956).
    source: str = ""


#: An npm scoped package: `@scope/name`, optionally with a version suffix.
_SCOPED_RE = re.compile(r"^(@[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)")
_NPM_FAMILY = frozenset({"npm", "pnpm", "yarn", "bun"})


def _strip_version_spec(prog: str, tok: str) -> str:
    """`foo[extra]>=1,<2` → `foo`; `@scope/name@1.2.3` → `@scope/name`.

    npm's separator is `@version` and PyPI's is a comparison operator, so the
    spelling rule is per-ecosystem rather than one regex over both.
    """
    body = tok
    if prog in _NPM_FAMILY:
        if body.startswith("@"):
            m = _SCOPED_RE.match(body)
            body = m.group(1) if m else ""
        elif "@" in body:
            body = body.split("@", 1)[0]
        return body.strip()
    if " @ " in body:                        # PEP 508 direct reference: `name @ url`
        body = body.split(" @ ", 1)[0]
    elif " @" in body:
        body = body.split(" @", 1)[0]
    body = _EXTRAS.sub("", body.split("[", 1)[0])
    body = re.split(r"[=<>!~;]", body, 1)[0]
    return body.strip()


#: Programs that spell the same tool differently: option tables are keyed by the
#: canonical one, so `pip3 install -r x.txt` does not report `x.txt` as a package.
_PROG_ALIAS = {"pip3": "pip", "pipx": "pip", "uvx": "uv", "pnpm": "npm",
               "yarn": "npm", "bun": "npm"}

#: A local artifact is not a registry project. PyPI names may contain dots, so
#: the only reliable tell for "this operand is a file" is its extension.
_ARTIFACT_SUFFIXES = (".whl", ".tar.gz", ".tgz", ".zip", ".egg", ".txt", ".json",
                      ".yaml", ".yml", ".toml", ".deb", ".rpm")


def _candidate_name(prog: str, tok: str) -> tuple[str, bool] | None:
    """(name, is_vcs) if `tok` names a distribution to fetch, else None.

    `prog` decides the spelling rules: PyPI accepts a PEP 508 specifier, npm
    accepts `name@version` and `@scope/name@version`. A path, a wheel, or a VCS
    URL names no registry project of its own — except PEP 440's `#egg=name`
    fragment, which does, and is reported as a VCS reference so the caller does
    not apply registry freshness to a name it got out of a repo URL.
    """
    if not tok or tok.startswith("-"):
        return None
    if "://" in tok or tok.startswith((".", "/", "~", "$", "file:", "git+", "ssh:")):
        m = _EGG_RE.search(tok)
        return (m.group(1), True) if m else None
    name = _strip_version_spec(prog, tok)
    if not name:
        return None
    if name.lower().endswith(_ARTIFACT_SUFFIXES):
        return None
    if prog in _NPM_FAMILY:
        if not (_NAME_RE.match(name) or _SCOPED_RE.match(name)):
            return None
        return name, False
    m = _NAME_RE.match(name)
    if not m:
        return None
    return m.group(1), False


def find_install_commands(command: str, *, _depth: int = 0) -> list[InstallRequest]:
    """Every distribution an executed command asks a registry for.

    Tokenised with `protected_paths._tokens`/`_segments` — the same reader
    `service_control` uses — so `grep 'pip install foo' notes.md` and
    `echo "pip install foo"` are a grep and an echo, and `a && b` is two
    commands. Wrappers and env assignments are peeled first, and a
    `bash -c "…"` / `python -c "…"` one-liner is read inside (two levels).
    """
    text = str(command or "")
    if not text or _depth > _MAX_DEPTH:
        return []
    from app.harness.protected_paths import _segments, _tokens

    out: list[InstallRequest] = []
    for argv in _segments(_tokens(text)):
        out.extend(_install_requests(argv, _depth))
    return out


def _split_override(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """The `LLOYD_DEP_OVERRIDE=…` value on this segment, and argv without it."""
    value: str | None = None
    rest: list[str] = []
    for tok in argv:
        m = _ASSIGNMENT.match(tok)
        if m and m.group(1) == OVERRIDE_ENV:
            value = m.group(2).strip().strip("'\"")
        else:
            rest.append(tok)
    return value, rest


def _install_requests(argv: Sequence[str], depth: int) -> list[InstallRequest]:
    """The install requests in one command segment, stamped with its override.

    The `LLOYD_DEP_OVERRIDE=` assignment is env state, not a program name, so it
    is peeled off before the wrappers — otherwise it would be read as argv[0] and
    the install would vanish from the check entirely. It then travels onto each
    request, because the decision it records belongs to this segment alone.
    """
    value, cleaned = _split_override(argv)
    found = _requests_in_segment(cleaned, depth)
    if value is not None:
        found = [replace(request, override=value) for request in found]
    return found


def _requests_in_segment(argv: Sequence[str], depth: int) -> list[InstallRequest]:
    from app.harness.protected_paths import _strip_wrappers

    peeled, _ = _strip_wrappers(argv)
    if not peeled:
        return []
    prog_full = peeled[0]
    prog = os.path.basename(prog_full)

    # `python -m pip install …` / `python3.11 -m pip install …`
    if _PYTHONS.match(prog) or prog.startswith("python3."):
        rest = peeled[1:]
        if "-c" in rest and depth < _MAX_DEPTH:
            i = rest.index("-c")
            inner = rest[i + 1] if i + 1 < len(rest) else ""
            return (find_install_commands(inner, _depth=depth + 1)
                    + _python_shell_literals(inner, depth))
        if "-m" in rest:
            i = rest.index("-m")
            module = os.path.basename(rest[i + 1]) if i + 1 < len(rest) else ""
            if module in ("pip", "pip3", "pip__main__"):
                after = rest[i + 2:]
                # `python -m pip install …` — the verb is argv of the module, not
                # an operand; dropping it here is what stops `install` itself
                # being reported as a distribution named `install`.
                #
                # And the verb is REQUIRED (#1956). Stripping it when present and
                # parsing whatever followed otherwise made `python -m pip list`
                # an install of a distribution named `list` — refused, four
                # times on 2026-09-30, as "not published on pypi.org" — while
                # the bare `pip list` beside it parsed to nothing, because that
                # path asks for a verb first. The first non-flag token decides.
                verb_at = next((j for j, tok in enumerate(after)
                                if not tok.startswith("-")), None)
                if verb_at is None or after[verb_at] not in _PIP_MODULE_VERBS:
                    return []
                return _requests_for("pip", "pip install", after[verb_at + 1:])
        # A quoted operand that is itself a shell one-liner (`python x.py -c
        # "pip install foo"` is rarer than `python -c`, but wrappers exist).
        for tok in rest:
            if " " in tok and _INSTALL_HINT_RE.search(tok):
                found = find_install_commands(tok, _depth=depth + 1)
                if found:
                    return found
        return []

    if prog in _SHELLS and depth < _MAX_DEPTH:
        rest = peeled[1:]
        if "-c" in rest:
            i = rest.index("-c")
            return find_install_commands(rest[i + 1] if i + 1 < len(rest) else "",
                                         _depth=depth + 1)
        for tok in rest:
            if " " in tok and _INSTALL_HINT_RE.search(tok):
                found = find_install_commands(tok, _depth=depth + 1)
                if found:
                    return found
        return []

    verbs = _INSTALL_FORMS.get(prog)
    if verbs is None:
        return []
    rest = peeled[1:]
    if prog == "uvx":
        # `uvx [--from pkg] cmd` / `uvx pkg==1.2 cmd`: the first operand that is
        # not a flag is the package unless `--from` names one.
        if "--from" in rest:
            i = rest.index("--from")
            pkg = rest[i + 1] if i + 1 < len(rest) else ""
            return _one("uvx", "uvx", _candidate_name("uv", pkg))
        operands = [t for t in rest if not t.startswith("-")]
        return _one("uvx", "uvx", _candidate_name("uv", operands[0]) if operands else None)
    verb = next((t for t in rest if t in verbs), None)
    if verb is None:
        # `uv pip install …`, `uv tool install …`
        operand = [t for t in rest if not t.startswith("-")]
        if prog == "uv" and operand and operand[0] in ("pip", "tool"):
            sub = operand[1] if len(operand) > 1 else ""
            if sub in ("install", "add"):
                return _requests_for("uv", f"uv {operand[0]} {sub}",
                                     rest[rest.index(sub) + 1:])
        return []
    return _requests_for(prog, f"{prog} {verb}", rest[rest.index(verb) + 1:])


_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish"})
#: The `python -m pip <verb>` forms that name distributions. Everything else
#: (`list`, `show`, `check`, `freeze`, `config`, `cache`, …) reads and installs
#: nothing. Unchanged from the set the branch always stripped.
_PIP_MODULE_VERBS = frozenset({"install", "uninstall", "download", "wheel"})
_INSTALL_HINT_RE = re.compile(r"\b(pip|pip3|uv|uvx|npm|pnpm|yarn|poetry|cargo|pipx)\b")


def _one(prog: str, verb: str,
         name: tuple[str, bool] | None) -> list[InstallRequest]:
    if not name:
        return []
    nm, vcs = name
    ecosystem = _ECOSYSTEM_BY_PROGRAM.get(prog, _Ecosystem_PYPI)
    return [InstallRequest(name=nm, ecosystem=ecosystem, verb=verb,
                           vetted=ecosystem == _Ecosystem_PYPI, vcs=vcs)]


#: Re-installing the installer is not adding a dependency: `pip install --upgrade
#: pip` and `npm install -g npm` are bootstrap commands, and refusing them (or
#: paying a lookup for them) would make the check look broken on its own tooling.
_SELF_PACKAGES = frozenset({"pip", "setuptools", "wheel", "pip-tools", "uv", "npm",
                            "poetry", "cargo", "pipx"})


_PYTHON_STRING_LITERAL = re.compile(r"'([^'\n]*)'|\"([^\"\n]*)\"")


def _python_shell_literals(source: str, depth: int) -> list[InstallRequest]:
    """Install commands hiding in a string literal inside a `python -c` one-liner.

    `python -c "import os; os.system('pip install graphy')"` tokenises to
    `os.system('pip`, `install`, `graphy)` — the install sits in a *data* position
    for the shell's grammar but is a command line for the interpreter, which is
    exactly the seam a matcher that reads only argv[0] walks past. So each string
    literal is re-parsed, but only when the literal is itself a command line: a
    literal beginning with anything other than an install program stays prose,
    which is why `python -c "print('run pip install foo first')"` is still not a
    candidate. That distinction is the same one clause 5 draws for grep and echo,
    applied one level in.
    """
    out: list[InstallRequest] = []
    for match in _PYTHON_STRING_LITERAL.finditer(source or ""):
        literal = match.group(1) if match.group(1) is not None else match.group(2)
        if not literal or not _INSTALL_HINT_RE.search(literal):
            continue
        head = literal.strip().split(None, 1)[0] if literal.strip() else ""
        if os.path.basename(head) not in _NESTABLE_HEADS:
            continue
        out.extend(find_install_commands(literal, _depth=depth + 1))
    return out


def _requests_for(prog: str, verb: str, args: Sequence[str]) -> list[InstallRequest]:
    ecosystem = _ECOSYSTEM_BY_PROGRAM.get(prog, _Ecosystem_PYPI)
    value_opts = _VALUE_OPTIONS.get(_PROG_ALIAS.get(prog, prog), frozenset())
    # `--flag=value` is one token; `--flag value` consumes the next.
    pending_value = False
    out: list[InstallRequest] = []
    for tok in args:
        if pending_value:
            pending_value = False
            continue
        if tok.startswith("-"):
            head = tok.split("=", 1)[0]
            if head in value_opts and "=" not in tok:
                pending_value = True
            continue
        if tok == "@" and out:
            # PEP 508 direct reference: `name @ git+https://…`. The name is the
            # project's, but the artifact comes from a VCS, so the registry facts
            # do not apply to it — and #628 owns the destination.
            out[-1] = replace(out[-1], vcs=True)
            continue
        name = _candidate_name(prog, tok)
        if name is None:
            continue
        nm, vcs = name
        if normalize_dist_name(nm) in _SELF_PACKAGES:
            continue  # re-installing the tool that is doing the installing
        out.append(InstallRequest(name=nm, ecosystem=ecosystem, verb=verb,
                                  vetted=ecosystem == _Ecosystem_PYPI, vcs=vcs))
    # `-r other.txt`: the names declared inside that file are being installed too.
    for req in _requirement_files(prog, args):
        try:
            text = Path(req).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for nm in _requirements_names(text):
            out.append(InstallRequest(name=nm, ecosystem=ecosystem, verb=verb,
                                      vetted=ecosystem == _Ecosystem_PYPI,
                                      source=req))
    return out


def _requirement_files(prog: str, args: Sequence[str]) -> list[str]:
    flags = ("-r", "--requirement") if _PROG_ALIAS.get(prog, prog) in ("pip", "uv") else ()
    if not flags:
        return []
    out = []
    for i, tok in enumerate(args):
        head = tok.split("=", 1)[0]
        if head in flags:
            value = tok.split("=", 1)[1] if "=" in tok else (args[i + 1] if i + 1 < len(args) else "")
            if value and not value.startswith("-"):
                out.append(value)
    return out


# ---------------------------------------------------------------------------
# Registry facts and the provenance policy
# ---------------------------------------------------------------------------

MIN_FIRST_RELEASE_AGE_DAYS = 90
MIN_RELEASE_COUNT = 2
REGISTRY_URL = "https://pypi.org/pypi/{name}/json"
#: Hard ceiling on one lookup. The item's budget is "<300 ms" for the check as a
#: whole; a TLS handshake to pypi.org does not reliably fit in 300 ms, so the
#: budget is enforced on the *path that can be slow* — a cached name costs
#: microseconds, and a cold name costs at most this. Measured on this box
#: 2026-09-24: a cold `GET /pypi/fastapi/json` (117 KB gzipped) answers in
#: ~0.6-1.0 s, and it happens for a new name about twice a month (2 executed
#: installs in 3,167 session transcripts).
DEFAULT_REGISTRY_TIMEOUT = 1.5
CACHE_TTL_DAYS = 7


@dataclass(frozen=True)
class Thresholds:
    min_first_release_age_days: int = MIN_FIRST_RELEASE_AGE_DAYS
    min_release_count: int = MIN_RELEASE_COUNT


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class RegistryFacts:
    name: str
    exists: bool
    first_release: dt.datetime | None = None
    release_count: int | None = None
    #: "pypi.org" | "cache" | "fixture" | "unreachable" — recorded so a report
    #: can say whether a verdict rested on a measured fact or on an outage.
    source: str = "pypi.org"
    observed_at: dt.datetime | None = None

    @property
    def unreachable(self) -> bool:
        return self.source == "unreachable"


@dataclass(frozen=True)
class Provenance:
    blocked: bool
    fact: str = ""            # which registry fact failed
    reason: str = ""          # the printed refusal, naming the failed fact
    facts: RegistryFacts | None = None

    @property
    def allowed(self) -> bool:
        return not self.blocked


def _as_utc(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = dt.datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def parse_pypi_payload(name: str, payload: dict, *,
                       observed_at: dt.datetime | None = None) -> RegistryFacts:
    """Read the two facts out of a `pypi.org/pypi/<name>/json` body.

    Shape verified against the live API 2026-09-24: the body carries
    `releases: {version: [file, …]}` where each file has `upload_time` as
    `2018-12-08T08:14:13` (no offset), `info.name`, and — for the payload's own
    reasons — 317 release keys for `fastapi`. First release is the *minimum*
    `upload_time` over every file of every release, and the release count is the
    number of release keys, yanked ones included: an attacker who yanks a
    release does not un-publish it.
    """
    obs = observed_at or dt.datetime.now(dt.timezone.utc)
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        releases = {}
    count = len(releases)
    times: list[dt.datetime] = []
    for files in releases.values():
        if not isinstance(files, list):
            continue
        for item in files:
            if isinstance(item, dict):
                stamp = _as_utc(item.get("upload_time") or item.get("upload_time_iso_8601"))
                if stamp is not None:
                    times.append(stamp)
    if not times and payload.get("info", {}).get("version"):
        # A release list with no timestamps is not "old": record what is unknown
        # and let the caller fail open rather than assert an age it cannot see.
        return RegistryFacts(name=name, exists=True, first_release=None,
                             release_count=count or None, source="pypi.org",
                             observed_at=obs)
    return RegistryFacts(name=name, exists=bool(releases) or bool(payload.get("urls")),
                         first_release=min(times) if times else None,
                         release_count=count or None, source="pypi.org", observed_at=obs)


def evaluate_provenance(facts: RegistryFacts, *, now: dt.datetime | None = None,
                        thresholds: Thresholds = DEFAULT_THRESHOLDS) -> Provenance:
    """The three-fact policy, as a pure function of (facts, clock, thresholds).

    Pure so the fixture eval and the real dispatch path cannot drift: there is
    one place that turns a registry fact into a refusal, and a threshold change
    moves both at once.
    """
    clock = now or dt.datetime.now(dt.timezone.utc)
    if facts.unreachable:
        return Provenance(blocked=False, fact="registry-unreachable",
                          reason="", facts=facts)
    if not facts.exists:
        return Provenance(
            blocked=True, fact="not-published",
            reason=(f"'{facts.name}' is not published on pypi.org (404). A name the "
                    f"registry has never heard of is the slop-squat signature — the "
                    f"hallucinated-name attack — so it is refused, not waved through. "
                    f"Check the spelling, or re-run the same command with "
                    f'{OVERRIDE_ENV}="why this exact package" if you really mean it.'),
            facts=facts)
    if facts.first_release is None:
        return Provenance(blocked=False, fact="first-release-unknown",
                          reason="", facts=facts)
    age_days = (clock - facts.first_release).days
    if age_days < thresholds.min_first_release_age_days:
        return Provenance(
            blocked=True, fact="first-release-too-recent",
            reason=(f"'{facts.name}' was first published {age_days} day(s) ago "
                    f"(threshold: {thresholds.min_first_release_age_days} days, "
                    f"first release {facts.first_release.date().isoformat()}). A "
                    f"distribution that young is exactly the shape of a name someone "
                    f"registered to answer a hallucination. Re-run with "
                    f'{OVERRIDE_ENV}="why this exact package" to override.'),
            facts=facts)
    if facts.release_count is None:
        return Provenance(blocked=False, fact="release-count-unknown", reason="",
                          facts=facts)
    if facts.release_count < thresholds.min_release_count:
        return Provenance(
            blocked=True, fact="too-few-releases",
            reason=(f"'{facts.name}' has {facts.release_count} release(s) on pypi.org "
                    f"(threshold: {thresholds.min_release_count}). One release and no "
                    f"history is a placeholder name, not a maintained project. Re-run "
                    f'with {OVERRIDE_ENV}="why this exact package" to override.'),
            facts=facts)
    return Provenance(blocked=False, fact="cleared", reason="", facts=facts)


# ---------------------------------------------------------------------------
# The registry client — one GET per new name, cached, fail-open
# ---------------------------------------------------------------------------

class MappingRegistry:
    """Facts from a dict: the fixture eval's and every test's registry."""

    def __init__(self, facts: dict[str, RegistryFacts], *, name: str = "fixture"):
        self._facts = {normalize_dist_name(k): v for k, v in facts.items()}
        self.name = name
        self.lookups: list[str] = []

    def lookup(self, name: str) -> RegistryFacts:
        key = normalize_dist_name(name)
        self.lookups.append(key)
        found = self._facts.get(key)
        if found is not None:
            return found
        return RegistryFacts(name=name, exists=False, source=self.name,
                             observed_at=dt.datetime.now(dt.timezone.utc))


class PypiRegistry:
    """`pypi.org/pypi/<name>/json`, with an on-disk cache and a hard timeout.

    Cache-first because the check must not become a latency tax: a name seen
    once is never fetched again within the TTL, and only names *new to the
    dependency set* are ever looked up at all. Any transport or parse failure is
    `source="unreachable"`, which fails open by policy — an outage is not a fact
    about a package.
    """

    def __init__(self, *, cache_path: Path | str | None = None,
                 timeout: float | None = None, ttl_days: int = CACHE_TTL_DAYS,
                 url_template: str = REGISTRY_URL):
        self.cache_path = Path(cache_path) if cache_path else registry_cache_path()
        self.timeout = (timeout if timeout is not None else registry_timeout())
        self.ttl_days = ttl_days
        self.url_template = url_template
        self.name = "pypi.org"
        self._memo: dict[str, RegistryFacts] = {}

    def lookup(self, name: str) -> RegistryFacts:
        import logging
        logger = logging.getLogger("lloyd-supply-chain")
        key = normalize_dist_name(name)
        now = dt.datetime.now(dt.timezone.utc)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        cached = _read_cache(self.cache_path, key, max_age_days=self.ttl_days, now=now)
        if cached is not None:
            self._memo[key] = cached
            return cached
        url = self.url_template.format(name=key)
        try:
            import httpx
            resp = httpx.get(url, timeout=self.timeout,
                             headers={"accept": "application/json",
                                      "user-agent": "lloyd-supply-chain/1 (#688)"})
        except Exception as exc:  # noqa: BLE001 — any transport error fails open
            logger.warning("supply-chain: registry lookup for %r failed (%s: %s); "
                           "allowing on an unknown, not a fact", key,
                           type(exc).__name__, str(exc)[:160])
            return RegistryFacts(name=name, exists=False, source="unreachable",
                                 observed_at=now)
        if resp.status_code == 404:
            facts = RegistryFacts(name=name, exists=False, source="pypi.org",
                                  observed_at=now)
        elif 200 <= resp.status_code < 300:
            try:
                facts = parse_pypi_payload(name, resp.json(), observed_at=now)
            except Exception:  # noqa: BLE001 — an unexpected body is not a fact
                logger.warning("supply-chain: unparseable registry body for %r; "
                               "allowing on an unknown", key)
                return RegistryFacts(name=name, exists=False, source="unreachable",
                                     observed_at=now)
        else:
            logger.warning("supply-chain: registry answered %s for %r; allowing on "
                           "an unknown", resp.status_code, key)
            return RegistryFacts(name=name, exists=False, source="unreachable",
                                 observed_at=now)
        self._memo[key] = facts
        _write_cache(self.cache_path, key, facts)
        return facts


def registry_timeout() -> float:
    raw = os.environ.get("LLOYD_SUPPLY_CHAIN_REGISTRY_TIMEOUT", "").strip()
    try:
        return float(raw) if raw else DEFAULT_REGISTRY_TIMEOUT
    except ValueError:
        return DEFAULT_REGISTRY_TIMEOUT


def registry_cache_path() -> Path:
    override = os.environ.get("LLOYD_SUPPLY_CHAIN_CACHE_DIR", "").strip()
    if override:
        return Path(override) / "registry-cache.json"
    try:
        from app.paths import DATA_ROOT
        return Path(DATA_ROOT) / "supply-chain" / "registry-cache.json"
    except Exception:  # noqa: BLE001
        return Path(os.path.expanduser("~/.cache/lloyd/supply-chain/registry-cache.json"))


def _read_cache(path: Path, key: str, *, max_age_days: float,
                now: dt.datetime) -> RegistryFacts | None:
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    row = blob.get(key) if isinstance(blob, dict) else None
    if not isinstance(row, dict):
        return None
    stamp = _as_utc(row.get("observed_at"))
    if stamp is None or (now - stamp) > dt.timedelta(days=max_age_days):
        return None
    if not row.get("exists"):
        return RegistryFacts(name=key, exists=False, source="cache", observed_at=stamp)
    return RegistryFacts(name=key, exists=True,
                         first_release=_as_utc(row.get("first_release")),
                         release_count=row.get("release_count"),
                         source="cache", observed_at=stamp)


def _write_cache(path: Path, key: str, facts: RegistryFacts) -> None:
    try:
        blob: dict[str, Any] = {}
        if path.exists():
            blob = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(blob, dict):
                blob = {}
    except (OSError, ValueError):
        blob = {}
    blob[key] = {
        "exists": facts.exists,
        "first_release": facts.first_release.isoformat() if facts.first_release else None,
        "release_count": facts.release_count,
        "observed_at": (facts.observed_at or dt.datetime.now(dt.timezone.utc)).isoformat(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


_REGISTRY: PypiRegistry | None = None


def default_registry() -> PypiRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = PypiRegistry()
    return _REGISTRY


def set_default_registry(registry: Any) -> None:
    """Test/job seam: swap the process registry without monkey-patching a client."""
    global _REGISTRY
    _REGISTRY = registry


# ---------------------------------------------------------------------------
# The dispatch check — the fifth check inside safety.check_bash_command
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceResult:
    """What one dispatch-time evaluation decided, for the log and the caller."""

    refusals: list[tuple[InstallRequest, Provenance]] = field(default_factory=list)
    #: Installs this check did NOT vet, and the caller must not read as vetted: a
    #: VCS or local-path operand, a name in an ecosystem whose registry facts
    #: #688 never reads, or a name whose lookup failed or timed out.
    unvetted: list[InstallRequest] = field(default_factory=list)
    #: (distribution name, stated reason) for each install let through by an
    #: override — per name, so the trend #688 step 6 asks for ("any override
    #: used") can be read out of a log line without re-parsing the command.
    overrides: list[tuple[str, str]] = field(default_factory=list)
    #: Why something was allowed without being vetted. The fail-open half of a
    #: guard has to be as legible as its deny half, or an outage reads as a clean
    #: tree — the class this file's own scanner wrapper refuses to join.
    notes: list[str] = field(default_factory=list)
    #: Distributions already declared by the repo's dependency files: allowed by the
    #: declaration itself, never looked up. Kept apart from `cleared` because
    #: `cleared` asserts a registry vetted the name, and here nothing was asked.
    declared: list[InstallRequest] = field(default_factory=list)
    cleared: list[InstallRequest] = field(default_factory=list)

    @property
    def refusal(self) -> str | None:
        return self.refusals[0][1].reason if self.refusals else None


#: The one durable record of what this guard decided (#1839). The log lines around
#: a decision are worthless as a record: `server.err` + `.1..10` is 11 × ~10 MB
#: spanning 2.6 days on 2026-09-29, and during a busy automod round one file closes
#: in minutes. Worse, the deny branch emits no log line at all — a refusal survives
#: only as the caller's boundary line, which truncates the command to an 80-char
#: excerpt and never names the failed registry fact per distribution. So the trend
#: #688 step 6 asks for ("blocks per week, overrides used") has no source. This file
#: is that source, beside `registry-cache.json` because it is the same kind of thing:
#: small, local, and about what the registry said.
PROVENANCE_JOURNAL_NAME = "provenance.jsonl"

#: The only five outcomes a parsed name may carry in the journal. `denied` and
#: `overridden` are the two the trend counts; `declared` and `cleared` say the name
#: needed nothing; `unvetted` is the fail-open half — allowed on an unknown, which
#: must never be folded into `cleared` or an outage reads as a clean tree.
JOURNAL_OUTCOMES = ("declared", "cleared", "denied", "overridden", "unvetted")


#: Outcomes a `-r <file>` name may be folded into its file's one entry under.
_COLLAPSED_OUTCOMES = frozenset({"declared", "cleared", "unvetted"})

#: One decision, one row: a Bash call reaches this guard twice — the PreToolUse
#: hook in the backend, then dispatch in the aggregator, about 11 ms apart and in
#: two processes — and each wrote a row, so 15 of 19 decisions on 2026-10-01 were
#: journalled twice. A row repeating the `(session, command, names)` of one
#: written inside this many seconds is the same decision and is not written
#: again. Read off the file's tail, because no in-process memory spans the two
#: writers. The same command run again later is a new decision and a new row.
JOURNAL_LATCH_SECONDS = 60.0
_LATCH_TAIL_BYTES = 262_144


def _decision_key(row: dict[str, Any]) -> tuple:
    names = row.get("names") or []
    return (row.get("session"), row.get("command"),
            tuple(sorted(str(e.get("name")) for e in names if isinstance(e, dict))))


def _already_journalled(target: Path, row: dict[str, Any], now: dt.datetime) -> bool:
    """Whether the file's tail already holds this decision inside the latch."""
    try:
        size = target.stat().st_size
    except OSError:
        return False
    with target.open("rb") as handle:
        handle.seek(max(0, size - _LATCH_TAIL_BYTES))
        tail = handle.read().decode("utf-8", errors="replace")
    key = _decision_key(row)
    for line in reversed(tail.splitlines()):
        try:
            prior = json.loads(line)
            at = dt.datetime.fromisoformat(str(prior.get("at")))
        except Exception:  # noqa: BLE001 — a cut first line, or a foreign row
            continue
        if (now - at).total_seconds() > JOURNAL_LATCH_SECONDS:
            return False        # rows are appended in time order: nothing older matters
        if _decision_key(prior) == key:
            return True
    return False


def provenance_journal_path() -> Path:
    """Where the decision journal lives: the registry cache's own directory.

    One override moves both files, because they are one store split by format. A
    second helper with its own env var would be a second thing to forget in a test,
    and the failure mode of forgetting it is a test writing into the live data root.
    """
    override = os.environ.get("LLOYD_SUPPLY_CHAIN_CACHE_DIR", "").strip()
    if override:
        return Path(override) / PROVENANCE_JOURNAL_NAME
    try:
        from app.paths import DATA_ROOT
        return Path(DATA_ROOT) / "supply-chain" / PROVENANCE_JOURNAL_NAME
    except Exception:  # noqa: BLE001 — same fail-open as the cache's own resolution
        return Path(os.path.expanduser(
            f"~/.cache/lloyd/supply-chain/{PROVENANCE_JOURNAL_NAME}"))


def journal_entries(result: ProvenanceResult) -> list[dict[str, Any]]:
    """Project a :class:`ProvenanceResult` onto one entry per parsed name.

    A projection, not a second decision: the journal is derived from the same object
    the caller is acting on, so it cannot report an outcome the caller did not see.
    An overridden install therefore reads `overridden`, not the `cleared` it also
    earned — the decision worth counting is the one that needed a reason.

    The `not-needed` entries `check_install_provenance` appends when
    :data:`OVERRIDE_ENV` was present but changed nothing keep their own entries,
    spelled exactly as they are in `result.overrides`. Folding them away would
    under-count the reflexive-override habit those rows exist to measure.

    Only the two facts a reader needs in order to act are copied: `fact` and
    `reason` on a denial (the registry fact that failed is the whole reason to keep
    the row), and `override` on an override, verbatim — a reason cut to a word limit
    is a reason nobody can judge afterwards.
    """
    entries: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    by_file: dict[tuple[str, str], dict[str, Any]] = {}

    def _add(request: InstallRequest, outcome: str) -> None:
        # #1956: names read out of `-r <file>` that needed nothing said about
        # them individually are one entry naming the file and how many. 98.9% of
        # the journal's entries on 2026-10-01 were `declared` names from one
        # lockfile, 180 to a row. A projection only — every name was still
        # parsed and decided on its own — and never applied to a denial or an
        # override, whose whole value is the name and the reason.
        source = getattr(request, "source", "")
        if source and outcome in _COLLAPSED_OUTCOMES \
                and getattr(request, "override", None) is None:
            key = (source, outcome)
            entry = by_file.get(key)
            if entry is None:
                entry = {"name": f"-r {source}", "file": source, "count": 0,
                         "ecosystem": request.ecosystem, "verb": request.verb,
                         "outcome": outcome}
                by_file[key] = entry
                entries.append(entry)
            entry["count"] += 1
            return
        entry = {"name": request.name, "ecosystem": request.ecosystem,
                 "verb": request.verb, "outcome": outcome}
        if source:
            entry["file"] = source
        entries.append(entry)
        by_name[entry["name"]] = entry

    for request in result.declared:
        _add(request, "declared")
    for request in result.cleared:
        _add(request, "cleared")
    for request in result.unvetted:
        _add(request, "unvetted")
    for request, verdict in result.refusals:
        _add(request, "denied")
        by_name[request.name]["fact"] = verdict.fact
        by_name[request.name]["reason"] = verdict.reason
    for name, reason in result.overrides:
        entry = by_name.get(name)
        if entry is None or entry["outcome"] != "cleared":
            entry = {"name": name, "ecosystem": "", "verb": "", "outcome": "overridden"}
            entries.append(entry)
        else:
            entry["outcome"] = "overridden"
        entry["override"] = reason
    return entries


def _journal_decision(command: str, session_id: str | None, session_class: str,
                      result: ProvenanceResult, *, path: Path | None = None,
                      now: dt.datetime | None = None) -> None:
    """Append one JSON line for one decision. Never raises, never delays (#1839).

    Fail-open like the rest of the guard, and for the same reason: a journal that
    can block a dispatch, or turn its own broken into a refusal, is worse than no
    journal. Building the row is inside the `except` too — a projection that trips
    over an unexpected shape must cost one lost row, not the command. Its failure is
    a log line naming the loss, which is itself subject to the rotation this file
    exists to escape; that is accepted, because the caller's refusal is already a
    witness to the decision and only the durable copy is missing.
    """
    import logging
    try:
        target = path if path is not None else provenance_journal_path()
        at = now or dt.datetime.now(dt.timezone.utc)
        row = {
            "at": at.isoformat(),
            "session": str(session_id) if session_id else None,
            "session_class": session_class,
            "command": str(command or "")[:200],
            "names": journal_entries(result),
        }
        unknown = sorted({e["outcome"] for e in row["names"]} - set(JOURNAL_OUTCOMES))
        if unknown:  # a future branch that invented an outcome: say so, still write
            logging.getLogger("lloyd-supply-chain").warning(
                "supply-chain: provenance journal carries outcome(s) outside %s: %s",
                list(JOURNAL_OUTCOMES), unknown)
        target.parent.mkdir(parents=True, exist_ok=True)
        if _already_journalled(target, row, at):
            return      # the hook already wrote this decision; dispatch is its echo
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 — the guard outlives its own journal
        logging.getLogger("lloyd-supply-chain").warning(
            "supply-chain: provenance journal write failed (%s: %s); the decision "
            "stands and only this row is lost", type(exc).__name__, str(exc)[:160])


def check_install_provenance(command: str, session_id: str | None, *,
                             parent_of: Callable[[str], str | None] | None = None,
                             registry: Any = None,
                             dependency_set: dict[str, str] | None = None,
                             root: Path | str | None = None,
                             now: dt.datetime | None = None,
                             thresholds: Thresholds = DEFAULT_THRESHOLDS
                             ) -> ProvenanceResult:
    """Refuse an install whose distribution name cannot clear the three facts.

    Returns a :class:`ProvenanceResult`; `refusal` is the sentence to print, and
    it names the failed registry fact rather than "policy denied". Never raises:
    a guard on the dispatch path that can raise becomes an outage, and every
    failure inside it is an unknown, which is fail-open by design.
    """
    import logging
    logger = logging.getLogger("lloyd-supply-chain")
    result = ProvenanceResult()
    text = str(command or "")
    if not text:
        return result
    try:
        requests = find_install_commands(text)
    except Exception as exc:  # noqa: BLE001 — a parser bug must not break dispatch
        logger.warning("supply-chain: install parser failed (%s: %s); allowing",
                       type(exc).__name__, str(exc)[:160])
        return result
    if not requests:
        return result

    attended = _attended_by_session_id(session_id, parent_of=parent_of)
    if attended:
        # The person's own turn is the cheap override this design keeps, and the
        # registry is never consulted for it: an interactive Bash call must not
        # pay a network round trip for a decision a human is present to make. It
        # also gets no journal row — there is no decision here to record, only a
        # name nobody looked up, and #1839 leaves "should attended installs be
        # journalled at all" an explicit open ruling rather than answering it by
        # accident with a row full of unknowns.
        return result

    known = dependency_set if dependency_set is not None else read_dependency_set(root)
    reg = registry if registry is not None else default_registry()
    seen: set[str] = set()
    for request in requests:
        key = normalize_dist_name(request.name)
        if key in seen:
            continue
        seen.add(key)
        if key in known:
            result.declared.append(request)
            continue
        if request.override:
            # A stated reason on this segment's own command line: let it through,
            # and record name + reason so an override is a logged event rather
            # than silence (step 6's "any override used" is unreadable otherwise).
            result.overrides.append((request.name, request.override.strip()))
            result.cleared.append(request)
            logger.warning("supply-chain: %s of %r overridden: %s", request.verb,
                           request.name, request.override.strip()[:200])
            continue
        if request.vcs:
            # A VCS/direct reference names no registry project: the freshness
            # facts do not apply, and #628 (network destinations) is the guard
            # that owns a `git+https://…` target.
            result.unvetted.append(request)
            result.notes.append(f"{request.name}: a VCS/direct reference names no "
                                f"registry project, so nothing here vetted it (#628 "
                                f"owns network destinations)")
            logger.info("supply-chain: %s of VCS reference %r is not a registry name; "
                        "not vetted here (#628 owns destinations)", request.verb,
                        request.name)
            continue
        if not request.vetted:
            result.unvetted.append(request)
            result.notes.append(f"{request.name}: {request.ecosystem} registry facts "
                                f"are not read by this check")
            logger.warning("supply-chain: %s of new %s distribution %r is not vetted "
                           "(no %s registry facts are read by #688)", request.verb,
                           request.ecosystem, request.name, request.ecosystem)
            continue
        try:
            facts = reg.lookup(request.name)
        except Exception as exc:  # noqa: BLE001
            result.unvetted.append(request)
            result.notes.append(f"{request.name}: the registry lookup raised "
                                f"{type(exc).__name__}, so it was never vetted")
            logger.warning("supply-chain: registry lookup raised for %r (%s: %s)",
                           request.name, type(exc).__name__, str(exc)[:160])
            continue
        verdict = evaluate_provenance(facts, now=now, thresholds=thresholds)
        if verdict.blocked:
            reason = verdict.reason
            if request.override is not None:
                # Blank, so not an override — but say so in the denial, or the
                # next attempt is another blank one.
                reason += (f" ({OVERRIDE_ENV} was set with no reason; an override "
                           f"without one is not an override.)")
            result.refusals.append((request, replace(verdict, reason=reason)))
        elif verdict.fact != "cleared":
            # Allowed on an unknown fact — which is not the same as vetted. A
            # registry that was unreachable, a release list with no timestamps, a
            # name the API answered without an age: each lands here, and putting
            # them in `cleared` would turn an outage into a clean tree.
            result.unvetted.append(request)
            result.notes.append(f"{request.name}: allowed without being vetted "
                                f"({verdict.fact})")
            logger.info("supply-chain: allowing %r on an unknown fact (%s)",
                        request.name, verdict.fact)
        else:
            result.cleared.append(request)
    if not result.overrides:
        unused = [value for value, _ in _overrides_in(text) if value]
        if unused:
            # An override nobody needed is still an override somebody reached for.
            # Counting only the ones that changed an outcome would under-count the
            # reflexive-override habit that turns a provenance check into theatre,
            # which is exactly what step 6's trend is for.
            result.overrides.extend(("not-needed", value) for value in unused)
            logger.info("supply-chain: %s present but nothing new to override: %s",
                        OVERRIDE_ENV, unused[0][:160])
    # One row per decision that actually decided something, after every branch above
    # has had its say: reached only past the two early returns (nothing parsed, or
    # attended), so a command with no install request and a chat turn write nothing.
    # Unattended by construction here — the attended path returned above — and the
    # class is carried as a field rather than hardcoded so a later ruling that
    # journals chat turns too finds the field already correct.
    _journal_decision(text, session_id, "attended" if attended else "unattended", result)
    return result


def _overrides_in(command: str) -> list[tuple[str | None, str]]:
    from app.harness.protected_paths import _segments, _tokens

    out: list[tuple[str | None, str]] = []
    try:
        for argv in _segments(_tokens(command)):
            value, _ = _split_override(argv)
            if value is not None:
                out.append((value, value))
    except Exception:  # noqa: BLE001
        return out
    return out


def _attended_by_session_id(session_id: str | None, *,
                            parent_of: Callable[[str], str | None] | None = None) -> bool:
    """True for a person's chat turn, which is the override this design keeps cheap.

    A background run (worker, autonomy, bench, and a `task:*` subagent of one) is
    the unattended path the item is about. No session id is *not* attended: #1053
    established that treating an absent id as "not sandboxed" is itself the
    bypass.

    Nor is an unresolvable one attended (#1961). This function's own comment used
    to name the miss as the bypass and then close only the `parent_of is None`
    case, so a resolver that was supplied and still returned nothing read as
    attended — and `check_install_provenance` returns at its attended branch
    before the registry is consulted and before any journal row is written, so
    that bypass left no trace at all. Attended is now a claim that needs evidence:
    `classify_session` says so, and an unresolvable `task:*` id pays the lookup
    like any other unattended turn.
    """
    from app.harness.service_control import ATTENDED, classify_session

    sid = str(session_id or "")
    if not sid:
        return False
    return classify_session(sid, parent_of=parent_of,
                            guard="install_provenance") is ATTENDED


# ---------------------------------------------------------------------------
# The offline advisory scan
# ---------------------------------------------------------------------------

BASELINE_DIR = "eval/supply-chain"
BASELINE_FILE = "baseline.yaml"
#: Upstream flags, verified against https://google.github.io/osv-scanner/usage/offline-mode/
#: on 2026-09-24: `--offline` scans against a previously downloaded local
#: database and makes no network call; the database location comes from
#: `OSV_SCANNER_LOCAL_DB_CACHE_DIRECTORY`; `--download-offline-databases` is the
#: *staging* command and is never passed here.
OSV_DB_ENV = "OSV_SCANNER_LOCAL_DB_CACHE_DIRECTORY"
#: pip-audit has no offline flag: it queries a vulnerability service. Its offline
#: knob is `-s osv --osv-url <local mirror>` — point it at a locally served OSV
#: API and the scan makes no external call. Unset means pip-audit is not
#: offline-capable here, so the wrapper declines to run it rather than run it
#: noisily.
OSV_API_URL_ENV = "LLOYD_OSV_API_URL"


@dataclass(frozen=True)
class Advisory:
    id: str
    package: str
    version: str | None
    ecosystem: str
    summary: str = ""


@dataclass
class ScanReport:
    """One scan's record. `advisory_count` is None unless a scanner *ran*.

    The distinction is the whole point: `completed` + 0 means "the instrument
    looked and found nothing"; `scanner_absent` + None means "no instrument, no
    verdict", and must never be printed, read or committed as a clean tree.
    """

    scan_status: str                    # completed | scanner_absent | failed
    scanner_name: str | None
    scanner_version: str | None
    scanner_path: str | None
    argv: list[str] = field(default_factory=list)
    offline: bool = True
    db_directory: str | None = None
    db_freshness: str | None = None
    duration_seconds: float | None = None
    advisory_count: int | None = None
    advisories: list[Advisory] = field(default_factory=list)
    dependency_set: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    generated_at: str = ""
    #: True when this record was written as an explicit coverage-gap disclosure
    #: rather than as the result of a scan.
    coverage_gap: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lloyd-supply-chain-baseline/1",
            "generated_at": self.generated_at or _utc_stamp(),
            "scan_status": self.scan_status,
            "verdict": ("none_recorded" if self.advisory_count is None
                        else ("advisories_found" if self.advisory_count else "no_advisories")),
            "scanner": {"name": self.scanner_name, "version": self.scanner_version,
                        "path": self.scanner_path, "argv": list(self.argv)},
            "offline": bool(self.offline),
            "coverage_gap": bool(self.coverage_gap),
            "database": {"directory": self.db_directory,
                         "freshness": self.db_freshness},
            "duration_seconds": self.duration_seconds,
            "advisory_count": self.advisory_count,
            "advisories": [{"id": a.id, "package": a.package, "version": a.version,
                            "ecosystem": a.ecosystem, "summary": a.summary}
                           for a in self.advisories[:200]],
            "dependency_set": list(self.dependency_set),
            "reason": self.reason,
        }


def _utc_stamp() -> str:
    """Offset-bearing UTC. A naive stamp in a machine-facing payload is read as
    UTC by every later reader and silently shifts the clock (the 2026-09-19
    `ALERT.md` lesson)."""
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


class ScannerSpec:
    """A locally installed scanner and how to invoke it with no network access.

    A plain class rather than a dataclass: the two specs differ in how they
    *build* argv (osv-scanner needs a lockfile flag per file, pip-audit a
    requirements flag plus a local service URL), and each one's offline
    prerequisite is its own method, so there is nothing to parametrise.
    """

    name: str = ""
    binary: str = ""
    #: A requirement beyond "on PATH", reported verbatim when unmet.
    prerequisite_env: str | None = None

    def argv(self, dependency_files: Sequence[Path]) -> list[str] | None:
        raise NotImplementedError

    def env(self) -> dict[str, str]:
        return {}


#: The one verified install route (#1838). PyPI has no `osv-scanner` — the name
#: returns 404, and this module's own install-provenance check refuses it as the
#: slop-squat signature — so the reason string must never send a reader there.
OSV_SCANNER_RELEASES = "https://github.com/google/osv-scanner/releases"
OSV_SCANNER_ASSET = "osv-scanner_linux_amd64"
OSV_SCANNER_SUMS = "osv-scanner_SHA256SUMS"


def default_osv_db_dir() -> Path:
    """Where the offline database is mirrored when `OSV_DB_ENV` is unset."""
    try:
        from app.paths import DATA_ROOT
        return Path(DATA_ROOT) / "supply-chain" / "osv-db"
    except Exception:  # noqa: BLE001
        return Path(os.path.expanduser("~/.cache/lloyd/osv-db"))


def _osv_lockfile_arg(path: Path) -> str:
    """`--lockfile` value for one dependency file. osv-scanner picks a parser by
    file name, and `requirements.lock` matches none — measured 2026-09-30 it exits
    127 ("could not determine extractor suitable to this file") and the whole scan
    reads as failed. That file is `pip freeze` output, so name the format:
    `requirements.txt:<path>`."""
    if path.name.startswith("requirements") and path.suffix == ".lock":
        return f"requirements.txt:{path}"
    return str(path)


class OsvScannerSpec(ScannerSpec):
    name = "osv-scanner"
    binary = "osv-scanner"
    prerequisite_env = OSV_DB_ENV

    def database_dir(self) -> Path:
        override = os.environ.get(OSV_DB_ENV, "").strip()
        if override:
            return Path(override)
        return default_osv_db_dir()

    def argv(self, dependency_files: Sequence[Path]) -> list[str]:
        argv = ["scan", "--offline", "--format", "json"]
        for path in dependency_files:
            argv += ["--lockfile", _osv_lockfile_arg(path)]
        extra = os.environ.get("LLOYD_OSV_SCANNER_ARGS", "").strip()
        if extra:
            import shlex as _shlex
            argv += _shlex.split(extra)
        return argv

    def env(self) -> dict[str, str]:
        return {OSV_DB_ENV: str(self.database_dir())}

    def freshness(self) -> str | None:
        """Newest mtime among the mirrored `*/all.zip` — the database's age."""
        root = self.database_dir()
        newest: float | None = None
        try:
            for path in root.rglob("all.zip"):
                stamp = path.stat().st_mtime
                if newest is None or stamp > newest:
                    newest = stamp
        except OSError:
            return None
        if newest is None:
            return None
        return dt.datetime.fromtimestamp(newest, dt.timezone.utc).isoformat().replace(
            "+00:00", "Z")


class PipAuditSpec(ScannerSpec):
    name = "pip-audit"
    binary = "pip-audit"
    prerequisite_env = OSV_API_URL_ENV

    def osv_url(self) -> str | None:
        raw = os.environ.get(OSV_API_URL_ENV, "").strip()
        if not raw:
            return None
        # Offline means offline: a service URL pointing off-box is not a local
        # mirror, and reporting `offline: true` for it would be a lie.
        if not re.match(r"^https?://(127\.0\.0\.1|localhost|::1|\[::1\])(:|/|$)", raw):
            return None
        return raw

    def argv(self, dependency_files: Sequence[Path]) -> list[str] | None:
        url = self.osv_url()
        if url is None:
            return None
        argv = ["-f", "json", "--progress-spinner", "off", "-s", "osv", "--osv-url", url]
        for path in dependency_files:
            argv += ["-r", str(path)]
        extra = os.environ.get("LLOYD_PIP_AUDIT_ARGS", "").strip()
        if extra:
            import shlex as _shlex
            argv += _shlex.split(extra)
        return argv

    def env(self) -> dict[str, str]:
        url = self.osv_url()
        return {OSV_API_URL_ENV: url} if url else {}


SCANNER_SPECS: tuple[ScannerSpec, ...] = (OsvScannerSpec(), PipAuditSpec())


def resolve_scanner(name: str | None = None) -> tuple[ScannerSpec | None, str]:
    """The first spec that can run offline here, and why the others could not.

    The reason string is clause 2's requirement: a missing scanner is reported as
    a missing scanner, naming the binaries, and never as zero advisories.
    """
    notes: list[str] = []
    for spec in SCANNER_SPECS:
        if name and spec.name != name:
            continue
        path = shutil.which(spec.binary)
        if path is None:
            notes.append(f"{spec.name}: not on PATH")
            continue
        if isinstance(spec, PipAuditSpec) and spec.osv_url() is None:
            notes.append(f"{spec.name}: needs {OSV_API_URL_ENV} pointing at a locally "
                         f"served OSV API (loopback), or it would make network calls")
            continue
        return spec, ""
    return None, "; ".join(notes) or f"no scanner named {name!r} is known"


def dependency_files(root: Path | str | None = None,
                     *, files: Sequence[str] = DEPENDENCY_SET_FILES) -> list[Path]:
    base = _repo_root(root)
    return [base / rel for rel in files if (base / rel).is_file()]


def dependency_set_summary(root: Path | str | None = None,
                           *, files: Sequence[str] = DEPENDENCY_SET_FILES) -> list[dict[str, Any]]:
    """Per-file counts and content hashes, so a baseline says *what* it covered."""
    base = _repo_root(root)
    out: list[dict[str, Any]] = []
    for rel in files:
        path = base / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        names = {normalize_dist_name(n) for n in _requirements_names(text)}
        out.append({"path": rel, "packages": len(names),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]})
    return out


def run_offline_scan(root: Path | str | None = None, *, scanner: str | None = None,
                     timeout: float = 180.0) -> ScanReport:
    """Scan the repo's declared dependency set with a locally installed scanner.

    Zero network calls: `--offline` for osv-scanner, a loopback OSV API for
    pip-audit, both resolved by `resolve_scanner` which refuses a spec whose
    offline prerequisite is unmet. On any failure the report carries a `reason`
    and `advisory_count = None` — an absent or broken instrument produces no
    verdict, never a clean one.
    """
    spec, why = resolve_scanner(scanner)
    if spec is None:
        return ScanReport(scan_status="scanner_absent", scanner_name=scanner,
                          scanner_version=None, scanner_path=None, offline=True,
                          reason=(f"no offline advisory scanner available: {why}. "
                                  f"Install osv-scanner from its GitHub release, "
                                  f"{OSV_SCANNER_RELEASES}: download {OSV_SCANNER_ASSET}, "
                                  f"check it against {OSV_SCANNER_SUMS}, and install it "
                                  f"as ~/.local/bin/osv-scanner (no sudo). The PyPI name "
                                  f"`osv-scanner` is not published there — never install "
                                  f"it from a package registry. Then mirror its database "
                                  f"into {default_osv_db_dir()} (or the directory "
                                  f"{OSV_DB_ENV} names) with `osv-scanner scan --offline "
                                  f"--download-offline-databases`; this record is a "
                                  f"coverage gap, not a clean tree."),
                          dependency_set=dependency_set_summary(root),
                          generated_at=_utc_stamp(), coverage_gap=True)

    if isinstance(spec, OsvScannerSpec) and spec.freshness() is None:
        # Measured 2026-09-30 (#1838): `osv-scanner scan --offline` over a directory
        # holding no mirrored `all.zip` exits 0 with zero findings, so without this
        # check an unmirrored box writes `completed / no_advisories` over a lock
        # that held 91 known advisories. No database is no instrument.
        return ScanReport(scan_status="failed", scanner_name=spec.name,
                          scanner_version=scanner_version(spec.binary),
                          scanner_path=shutil.which(spec.binary), offline=True,
                          db_directory=_db_dir(spec),
                          reason=(f"osv-scanner has no mirrored database under "
                                  f"{spec.database_dir()} (no */all.zip); an offline "
                                  f"scan against it reports zero advisories whatever "
                                  f"the tree holds. Mirror it with `osv-scanner scan "
                                  f"--offline --download-offline-databases` first; this "
                                  f"record is a coverage gap, not a clean tree."),
                          dependency_set=dependency_set_summary(root),
                          generated_at=_utc_stamp(), coverage_gap=True)

    files = dependency_files(root)
    argv = spec.argv(files)
    if argv is None:
        return ScanReport(scan_status="failed", scanner_name=spec.name,
                          scanner_version=None, scanner_path=None, offline=False,
                          reason=f"{spec.name} lost its offline prerequisite",
                          dependency_set=dependency_set_summary(root),
                          generated_at=_utc_stamp())
    path = shutil.which(spec.binary) or spec.binary
    child_env = {**os.environ, **spec.env()}
    started = time.monotonic()
    try:
        proc = subprocess.run([path, *argv], capture_output=True, text=True,
                              timeout=timeout, env=child_env, cwd=str(_repo_root(root)))
    except FileNotFoundError:
        return ScanReport(scan_status="failed", scanner_name=spec.name,
                          scanner_version=None, scanner_path=str(path),
                          argv=list(argv), offline=True, db_directory=_db_dir(spec),
                          duration_seconds=round(time.monotonic() - started, 3),
                          reason=f"{spec.binary} disappeared from PATH mid-run",
                          dependency_set=dependency_set_summary(root),
                          generated_at=_utc_stamp())
    except subprocess.TimeoutExpired:
        return ScanReport(scan_status="failed", scanner_name=spec.name,
                          scanner_version=None, scanner_path=str(path),
                          argv=list(argv), offline=True, db_directory=_db_dir(spec),
                          duration_seconds=round(time.monotonic() - started, 3),
                          reason=f"{spec.binary} exceeded {timeout}s and was killed",
                          dependency_set=dependency_set_summary(root),
                          generated_at=_utc_stamp())
    duration = round(time.monotonic() - started, 3)
    # osv-scanner exits 1 when it *finds* something, so a non-zero status is not
    # automatically a failure: the body decides.
    parsed = _parse_scan_output(spec.name, proc.stdout)
    if parsed is None:
        return ScanReport(scan_status="failed", scanner_name=spec.name,
                          scanner_version=scanner_version(spec.binary),
                          scanner_path=str(path), argv=list(argv), offline=True,
                          db_directory=_db_dir(spec),
                          db_freshness=_db_freshness(spec), duration_seconds=duration,
                          reason=(f"{spec.name} exited {proc.returncode} and its output "
                                  f"is not a report this wrapper can read: "
                                  f"{(proc.stderr or proc.stdout or '').strip()[:300]}"),
                          dependency_set=dependency_set_summary(root),
                          generated_at=_utc_stamp())
    advisories, count = parsed
    return ScanReport(scan_status="completed", scanner_name=spec.name,
                      scanner_version=scanner_version(spec.binary),
                      scanner_path=str(path), argv=list(argv), offline=True,
                      db_directory=_db_dir(spec), db_freshness=_db_freshness(spec),
                      duration_seconds=duration, advisory_count=count,
                      advisories=advisories,
                      dependency_set=dependency_set_summary(root),
                      generated_at=_utc_stamp())


def _db_dir(spec: ScannerSpec) -> str | None:
    return str(spec.database_dir()) if isinstance(spec, OsvScannerSpec) else None


def _db_freshness(spec: ScannerSpec) -> str | None:
    return spec.freshness() if isinstance(spec, OsvScannerSpec) else None


def scanner_version(binary: str, *, timeout: float = 5.0) -> str | None:
    """`<binary> --version` reduced to a version token, or None.

    Best-effort: a version string is recorded when the binary offers one, and an
    unusual version banner must not fail a scan whose body parsed fine.
    """
    path = shutil.which(binary)
    if path is None:
        return None
    try:
        proc = subprocess.run([path, "--version"], capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (proc.stdout or proc.stderr or "").strip()
    m = re.search(r"(\d+\.\d+(?:\.\d+)?[0-9A-Za-z.\-]*)", text)
    return m.group(1) if m else (text[:60] or None)


def _parse_scan_output(scanner: str, text: str) -> tuple[list[Advisory], int] | None:
    """Normalise a scanner's JSON into advisories, or None if it is not a report.

    Returning None on an unreadable body is what keeps "the scanner ran and said
    nothing" distinguishable from "the scanner ran and we could not read it" —
    and the second must not become a zero.
    """
    body = (text or "").strip()
    if not body:
        return None
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    advisories: list[Advisory] = []
    try:
        if scanner == "osv-scanner":
            # osv-scanner v2 `--format json`, as captured from v2.6.0 on 2026-09-30:
            # results[].packages[] = {"package": {name, version, ecosystem},
            # "vulnerabilities": [{id, summary, aliases, …}], "groups": […]}.
            # This parser used to read `packages[].package_vulnerabilities` and a
            # top-level `name` — a shape upstream never printed — so every real
            # finding was dropped and a 91-advisory lock read as zero (#1838). A
            # package entry of any other shape is now an unreadable report, never
            # a quiet zero.
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, list):
                return None
            for group in results:
                if not isinstance(group, dict):
                    return None
                for pkg in group.get("packages") or []:
                    meta = pkg.get("package") if isinstance(pkg, dict) else None
                    if not isinstance(meta, dict):
                        return None
                    for vuln in pkg.get("vulnerabilities") or []:
                        if not isinstance(vuln, dict):
                            continue
                        advisories.append(Advisory(
                            id=str(vuln.get("id") or ""),
                            package=str(meta.get("name") or ""),
                            version=meta.get("version"),
                            ecosystem=str(meta.get("ecosystem") or ""),
                            summary=str(vuln.get("summary") or "")[:200]))
            return advisories, len(advisories)
        # pip-audit: a list of dependency entries, each with a `vulns` list.
        if not isinstance(payload, list):
            return None
        for entry in payload:
            if not isinstance(entry, dict) or "vulns" not in entry:
                return None
            for vuln in entry.get("vulns") or []:
                if not isinstance(vuln, dict):
                    continue
                advisories.append(Advisory(
                    id=str(vuln.get("id") or ""), package=str(entry.get("package") or ""),
                    version=entry.get("version"), ecosystem="PyPI",
                    summary=str(vuln.get("description") or "")[:200]))
        return advisories, len(advisories)
    except (AttributeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# The committed baseline
# ---------------------------------------------------------------------------

def baseline_path(root: Path | str | None = None) -> Path:
    return _repo_root(root) / BASELINE_DIR / BASELINE_FILE


def write_baseline(report: ScanReport, path: Path | str | None = None, *,
                   root: Path | str | None = None) -> Path:
    """Write the YAML baseline. YAML, not JSON, because `.gitignore:38`'s `*.json`
    would silently un-track a `.json` baseline and `.gitignore` is a denied path
    for a round — the same trap that swallowed `eval/supply-chain/baseline.json`
    on the first `git check-ignore` of this item."""
    import yaml

    target = Path(path) if path else baseline_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Supply-chain baseline for the current tree (#688). Committed, "
              "offline, regenerated with\n"
              "#   .venvs/lloyd/bin/python -m app.harness.supply_chain scan "
              "--write-baseline\n"
              "# scan_status: scanner_absent means NO scanner ran and "
              "advisory_count is therefore null —\n"
              "# it is a coverage gap, and must never be read as 'this tree has no "
              "vulnerabilities'.\n")
    target.write_text(header + yaml.safe_dump(report.to_dict(), sort_keys=False,
                                              allow_unicode=True),
                      encoding="utf-8")
    return target


def read_baseline(path: Path | str | None = None, *,
                  root: Path | str | None = None) -> dict[str, Any]:
    import yaml

    target = Path(path) if path else baseline_path(root)
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# The fixture eval — planted names, measured control
# ---------------------------------------------------------------------------

FIXTURES_FILE = "provenance-fixtures.yaml"
#: The fixture set's pinned clock. The fresh-shaped entries are planted "published
#: last week" names, and an unpinned clock would age them past the 90-day
#: threshold inside three months, turning a passing eval into a false failure.
#: Pinned so the eval measures the policy, not the calendar.
FIXTURES_DIR = BASELINE_DIR


def fixtures_path(root: Path | str | None = None) -> Path:
    return _repo_root(root) / FIXTURES_DIR / FIXTURES_FILE


def fixtures_eval_command() -> list[str]:
    """The command that runs the eval: this interpreter, this module, this subcommand.

    A test pins the nightly wiring to it, so the string lives here rather than in a
    second place where it could drift from the documented entry point.
    """
    return [sys.executable, "-m", "app.harness.supply_chain", "fixtures"]


def read_fixtures(path: Path | str | None = None, *,
                  root: Path | str | None = None) -> dict[str, Any]:
    import yaml

    target = Path(path) if path else fixtures_path(root)
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def fixtures_registry(doc: dict[str, Any]) -> MappingRegistry:
    """A registry carrying exactly the fixture document's facts, measured or planted."""
    facts: dict[str, RegistryFacts] = {}
    for section in ("planted", "control"):
        for entry in doc.get(section) or []:
            name = str(entry.get("name") or "")
            if not name:
                continue
            blob = entry.get("facts") or {}
            facts[name] = _facts_from_blob(name, blob)
    return MappingRegistry(facts, name="fixtures")


def _facts_from_blob(name: str, blob: dict[str, Any]) -> RegistryFacts:
    return RegistryFacts(
        name=name, exists=bool(blob.get("exists")),
        first_release=_as_utc(blob.get("first_release")),
        release_count=blob.get("release_count"),
        source="fixture", observed_at=_as_utc(blob.get("observed_at")))


def run_fixtures_eval(doc: dict[str, Any], *, root: Path | str | None = None,
                      registry: Any = None) -> dict[str, Any]:
    """Block every planted name, block none of the repo's own. Returns the rates.

    Runs the *dispatch* entry point — `check_install_provenance` with the real
    command parser, a background session id, and an empty dependency set — so
    what is measured is the path a worker hits, not a helper standing next to
    it. The control set is evaluated against an empty dependency set on purpose:
    the claim "0 of the repo's real dependencies blocked" is only worth anything
    if the *policy* clears them once they look new, which is the exact moment a
    fresh human commit would have to be installed again.

    The summary also carries three self-checks, because a fixture set that only
    reports two rates can be quietly wrong: `expectations_mismatch` (an entry
    whose declared `expect:` disagrees with what the policy does — the prose and
    the behaviour have to be one claim), `contradictions` (a planted entry whose
    *measured* registry reality would not be blocked, so the name is no longer a
    squat-shaped name), and `control_declared_missing` (a control name that is no
    longer in this repo's dependency files, which would let the control drift into
    names the repo never installed).
    """
    clock = _as_utc(doc.get("evaluated_at")) or dt.datetime.now(dt.timezone.utc)
    reg = registry if registry is not None else fixtures_registry(doc)
    session = "20260924_054120_autonomy_fixtures"

    def verdict(name: str) -> bool:
        result = check_install_provenance(f"pip install {name}", session,
                                          registry=reg, dependency_set={},
                                          now=clock)
        return bool(result.refusal)

    planted = list(doc.get("planted") or [])
    control = list(doc.get("control") or [])
    missed = [str(e.get("name")) for e in planted if not verdict(str(e.get("name")))]
    false_blocked = [str(e.get("name")) for e in control
                     if verdict(str(e.get("name")))]
    blocked = len(planted) - len(missed)
    declared = read_dependency_set(root)
    shapes: dict[str, int] = {}
    for entry in planted:
        shapes[str(entry.get("shape") or "unlabelled")] = (
            shapes.get(str(entry.get("shape") or "unlabelled"), 0) + 1)
    return {
        "planted_total": len(planted), "planted_blocked": blocked, "planted_missed": missed,
        "control_total": len(control), "control_blocked": len(false_blocked),
        "control_false_blocks": false_blocked,
        "block_rate": (blocked / len(planted)) if planted else None,
        "false_block_rate": (len(false_blocked) / len(control)) if control else None,
        "planted_shapes": shapes,
        "evaluated_at": clock.isoformat(),
        "clock": ("pinned" if doc.get("evaluated_at") else "wall clock"),
        "control_names": [str(e.get("name")) for e in control],
        "expectations_mismatch": _expectation_mismatches(planted, verdict),
        "contradictions": [str(e.get("name")) for e in planted if _reality_contradicts(e, clock)],
        "unmeasured_control": [str(e.get("name")) for e in control
                               if not (e.get("facts") or {}).get("measured")],
        "control_declared_missing": [
            str(e.get("name")) for e in control
            if normalize_dist_name(str(e.get("name") or "")) not in declared],
    }


def _expectation_mismatches(planted: Sequence[dict[str, Any]],
                            verdict: Any) -> list[str]:
    """Entries whose authored `expect:` disagrees with the policy's verdict."""
    bad = []
    for entry in planted:
        name = str(entry.get("name") or "")
        expect = entry.get("expect")
        if expect is None:
            continue
        if bool(str(expect).strip().lower() == "block") != bool(verdict(name)):
            bad.append(name)
    return bad


def _reality_contradicts(entry: dict[str, Any], clock: dt.datetime) -> bool:
    """True when a planted entry's MEASURED registry reality would sail through.

    A planted name stops being a fixture the moment the registry starts serving it
    with an old first release and several versions: measuring it then says so
    instead of letting the block rate quietly stand on a name nobody would squat.
    """
    reality = entry.get("reality") or {}
    if not reality.get("measured"):
        return False
    facts = _facts_from_blob(str(entry.get("name") or ""), reality)
    return not evaluate_provenance(facts, now=clock).blocked


def format_fixtures_eval(summary: dict[str, Any]) -> str:
    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"
    shapes = summary.get("planted_shapes") or {}
    lines = [
        (f"planted squat-shaped names: {summary['planted_blocked']}/"
         f"{summary['planted_total']} blocked — block rate {pct(summary['block_rate'])}"),
        (f"repo's own dependencies (control): {summary['control_blocked']}/"
         f"{summary['control_total']} blocked — false-block rate "
         f"{pct(summary['false_block_rate'])}"),
        "planted shapes: " + ", ".join(f"{k} {v}" for k, v in sorted(shapes.items())),
        f"clock: {summary['clock']} at {summary['evaluated_at']}",
    ]
    if summary["planted_missed"]:
        lines.append("NOT BLOCKED: " + ", ".join(summary["planted_missed"]))
    if summary["control_false_blocks"]:
        lines.append("FALSELY BLOCKED: " + ", ".join(summary["control_false_blocks"]))
    for key, label in (("expectations_mismatch", "EXPECT DISAGREES WITH POLICY"),
                       ("contradictions", "MEASURED REALITY NOT BLOCKABLE"),
                       ("unmeasured_control", "CONTROL FACT NOT MEASURED"),
                       ("control_declared_missing", "CONTROL NAME NOT IN DEPENDENCY FILES")):
        names = summary.get(key) or []
        if names:
            lines.append(f"{label}: {', '.join(names)}")
    return "\n".join(lines)


def measure_fixtures(doc: dict[str, Any], *, registry: Any = None,
                     control_names: Sequence[str] | None = None,
                     now: dt.datetime | None = None) -> dict[str, Any]:
    """Fill in the fixture document's measured facts, from the live registry.

    `regenerate the committed fixture set` is one command (`fixtures --measure`)
    precisely so the numbers in a security fixture are read off the registry
    rather than asserted by whoever last edited the file. A planted name whose
    measured reality would let it through is a contradiction, not a fixture: it
    is reported in `contradictions` and the caller refuses to commit it.
    """
    clock = _as_utc(doc.get("evaluated_at")) or now or dt.datetime.now(dt.timezone.utc)
    reg = registry if registry is not None else default_registry()
    out = json.loads(json.dumps(doc))  # deep copy without a deepcopy import
    out["measured_at"] = clock.isoformat()

    for entry in out.get("planted") or []:
        facts = _measure_one(reg, str(entry.get("name") or ""))
        reality = {"measured": facts.source != "unreachable",
                   "exists": facts.exists,
                   "first_release": facts.first_release.isoformat() if facts.first_release else None,
                   "release_count": facts.release_count,
                   "observed_at": (facts.observed_at or clock).isoformat()}
        entry["reality"] = reality
        blob = entry.get("facts") or {}
        if not blob.get("measured") and blob.get("exists") is False:
            # An invented/typosquat name asserts "the registry has never heard of
            # it": adopt the measured reality so the fixture's fact IS the world's.
            entry["facts"] = {"exists": facts.exists,
                              "first_release": reality["first_release"],
                              "release_count": reality["release_count"],
                              "measured": reality["measured"],
                              "observed_at": reality["observed_at"]}
        if str(blob.get("first_release_age_days", "")).strip():
            age = int(blob["first_release_age_days"])
            stamp = clock - dt.timedelta(days=age)
            entry["facts"]["first_release"] = stamp.isoformat().replace("+00:00", "Z")
        verdict = evaluate_provenance(_facts_from_blob(str(entry.get("name")),
                                                       entry.get("facts") or {}),
                                      now=clock)
        entry["expected_blocked"] = bool(verdict.blocked)
        if _reality_contradicts(entry, clock):
            out.setdefault("contradictions", []).append(str(entry.get("name") or ""))

    names = list(control_names) if control_names is not None else sorted(
        read_dependency_set())
    control: list[dict[str, Any]] = []
    for name in names:
        facts = _measure_one(reg, name)
        if facts.source == "unreachable":
            out.setdefault("unmeasured_control", []).append(name)
            continue
        control.append({"name": name, "facts": {
            "exists": facts.exists,
            "first_release": (facts.first_release.isoformat() if facts.first_release
                              else None),
            "release_count": facts.release_count, "measured": True,
            "observed_at": (facts.observed_at or clock).isoformat()}})
    out["control"] = control
    return out


def _measure_one(registry: Any, name: str) -> RegistryFacts:
    try:
        return registry.lookup(name)
    except Exception:  # noqa: BLE001
        return RegistryFacts(name=name, exists=False, source="unreachable")


def write_fixtures(doc: dict[str, Any], path: Path | str | None = None, *,
                   root: Path | str | None = None) -> Path:
    import yaml

    target = Path(path) if path else fixtures_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Fixture set for the #688 install-provenance policy: planted names that\n"
              "# must ALL be blocked, and the repo's own dependencies as a control that\n"
              "# must be blocked by NONE of them. `facts.measured: true` means the fact\n"
              "# was read off pypi.org by `fixtures --measure`; a planted fresh-shaped\n"
              "# entry carries synthetic facts on purpose (that is what planting is) and\n"
              "# its `reality:` block records what the registry actually says.\n"
              "# Regenerate: .venvs/lloyd/bin/python -m app.harness.supply_chain "
              "fixtures --measure\n")
    target.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                      encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

USAGE = """usage: python -m app.harness.supply_chain <command>

  scan [--write-baseline] [--scanner NAME]
        Run the offline advisory scan over the repo's declared dependency set.
        Exit 0 on a completed scan, 3 when no scanner can run offline here.
  deps  Print the normalised distribution names the repo declares.
  provenance "<command string>"
        Print what the dispatch check would decide for that command.
  fixtures [--measure] [--write]
        Run the planted-name fixture eval and print both rates. --measure reads
        the live registry (one GET per name) to refresh the facts; --write saves
        the measured document over eval/supply-chain/provenance-fixtures.yaml.
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(USAGE)
        return 1
    command = args[0]
    if command == "deps":
        known = read_dependency_set()
        for name in sorted(known):
            print(f"{name}\t{known[name]}")
        print(f"# {len(known)} distributions declared")
        return 0
    if command == "scan":
        report = run_offline_scan(scanner=_opt_value(args, "--scanner"))
        if "--write-baseline" in args:
            if report.scan_status == "completed" or report.coverage_gap:
                target = write_baseline(report)
                print(f"baseline written: {target}")
            else:
                print("refusing to overwrite the baseline with a failed scan: "
                      + report.reason)
                return 3
        print(_format_report(report))
        return 0 if report.scan_status == "completed" else 3
    if command == "provenance":
        result = check_install_provenance(" ".join(args[1:]), session_id=None)
        print(_format_verdicts(result))
        return 1 if result.refusal else 0
    if command == "fixtures":
        doc = read_fixtures()
        if "--measure" in args:
            doc = measure_fixtures(doc)
            if doc.get("contradictions"):
                print("planted names whose measured reality would let them through — "
                      "replace them, they are not usable as fixtures: "
                      + ", ".join(doc["contradictions"]))
                return 4
        summary = run_fixtures_eval(doc)
        print(format_fixtures_eval(summary))
        if "--write" in args:
            print(f"fixtures written: {write_fixtures(doc)}")
        return 0 if (not summary["planted_missed"] and not summary["control_false_blocks"]
                     and not doc.get("contradictions")) else 4
    print(USAGE)
    return 1


def format_provenance_verdicts(result: ProvenanceResult) -> str:
    """One line per decision, in the order a reader has to act on them.

    Public because the distinct line kinds are the whole output: `BLOCK` carries the
    failed registry fact, `OVERRIDE` records the reason a blocked name was let
    through, `UNVETTED` says the check did not look at all, `ALLOW` is the only line
    that means a registry vetted the name, and `DECLARED` means the repo's own
    dependency file allowed it without any lookup. Folding `UNVETTED` into `ALLOW`
    would report an outage or an uncovered ecosystem as a clean install, and folding
    `DECLARED` into silence would report a recognised install as no install at all —
    both are the failure mode this file exists to avoid.
    """
    lines: list[str] = []
    for request, verdict in result.refusals:
        lines.append(f"BLOCK {request.name} ({request.verb}): {verdict.reason}")
    for name, reason in result.overrides:
        lines.append(f"OVERRIDE {name}: provenance not vetted on stated reason: {reason}")
    for request in result.unvetted:
        lines.append(f"UNVETTED {request.name} ({request.verb})")
    for request in result.cleared:
        if any(name == request.name for name, _ in result.overrides):
            continue  # already carried on its OVERRIDE line
        lines.append(f"ALLOW {request.name} ({request.verb})")
    for request in result.declared:
        # Kept apart from ALLOW for the same reason ALLOW is kept apart from
        # UNVETTED: this install was allowed by a file, and no registry fact was
        # asked about it.
        lines.append(f"DECLARED {request.name} ({request.verb})")
    lines.extend(f"NOTE {note}" for note in result.notes)
    return "\n".join(lines) or "no install command recognised in that string"


def _format_verdicts(result: ProvenanceResult) -> str:
    return format_provenance_verdicts(result)


def _opt_value(args: Sequence[str], flag: str) -> str | None:
    if flag in args and args.index(flag) + 1 < len(args):
        return args[args.index(flag) + 1]
    return None


def _format_report(report: ScanReport) -> str:
    if report.scan_status == "completed":
        return (f"{report.scanner_name} {report.scanner_version or '(version unknown)'}: "
                f"{report.advisory_count} advisory(ies) in {report.duration_seconds}s, "
                f"offline={report.offline}")
    return f"{report.scan_status}: {report.reason}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
