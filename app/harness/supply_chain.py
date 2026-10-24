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

The fourth fact, and why a declared name is looked up too (#3206)
-----------------------------------------------------------------
The three facts above are all facts about a NAME, and #688's contract deliberately
sent a name already in the dependency set to `declared` with no registry read at all.
That is structurally blind to the exhibit this control was written from: a working MCP
utility with **15 clean releases** that added one line at **v16** which BCC-ed the
owner on every email — a package that is *already declared* when the bad release ships.
So one more fact is priced, about the ARTIFACT: **the release being downloaded is at
least :data:`MIN_RELEASE_AGE_DAYS` days old** (:data:`Thresholds.min_release_age_days`),
and for a declared name it is the ONLY fact consulted, so no existing pass gains a new
failure mode. Its fact name is :data:`VERSION_YOUNG_FACT`; the three fail-opens that
mean "age could not be measured" are :data:`VERSION_AGE_UNKNOWN_FACT` (the snapshot has
no date for that version — which absence in `RegistryFacts.version_upload_times` always
and only ever means), :data:`VERSION_TARGET_UNKNOWN_FACT` (nothing says which release
would be taken) and :data:`REGISTRY_UNREACHABLE_NOTE` (no verdict, not a pass either).

The lookup cost that buys it: one cached read when the command pins a version the
cached snapshot already dates, one live read when it pins one the snapshot cannot date
(a release published since the entry was written is the release a hold exists for), and
one live read for any unpinned install, because "latest" is an identity and identities
expire. A pinned install of an old release — the whole content of this repo's own
`requirements.lock` — therefore costs what it always cost, and attended turns still pay
nothing at all, because the turn-level skip is untouched.
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


def _requirements_declared(text: str) -> list[tuple[str, str | None]]:
    """(name, specifier) for every distribution a requirements-file *text* declares.

    The one loop both readers go through: names for #688's declared set, and the pinned
    specifier too since #3206, because a `-r requirements.lock` install asks for a
    SPECIFIC artifact and the version hold has to be priced on that release rather than on
    whatever the registry currently calls latest.

    Handles comments, line continuations, `-r`/`-e`/`--hash` option lines,
    environment markers, extras, and `name @ url` direct references. A VCS or
    path reference (`git+https://…`, `./pkg`, `file://…`) contributes nothing:
    it names no registry project, and guessing a name out of a repo URL is how a
    control-set reader starts classifying real dependencies as new.
    """
    declared: list[tuple[str, str | None]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split(" #", 1)[0].strip().rstrip("\\").strip()
        if not line or line.startswith("-"):
            continue
        if "+://" in line or line.startswith(("file:", ".", "/", "~", "$", "-")):
            continue
        spec: str | None = None
        if " @ " in line:                     # `name @ url` — the URL is the artifact
            body, _, tail = line.partition(" @ ")
            spec = ("@ " + tail.strip()) or None
        else:
            body = line
        # The specifier is read off the line BEFORE the extras are stripped, so
        # `mcp[cli]==2.1.0` keeps `[cli]==2.1.0` here exactly as a command line would, and
        # `mcp>=2.1.0,<3` keeps its range — which is a claim about the newest release, not
        # about a pinned one, and the difference is what the hold turns on.
        spec = spec or _version_spec("pip", body)
        body = _EXTRAS.sub("", body).strip()
        m = _NAME_RE.match(body)
        if m:
            declared.append((m.group(1), spec))
    return declared


def _requirements_names(text: str) -> list[str]:
    """Distribution names declared by requirements-file *text* (specs dropped)."""
    return [name for name, _spec in _requirements_declared(text)]


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
    #: The specifier that followed the name, exactly as written and lowercased — `==2.2.0`,
    #: `>=1.2,<2`, `[cli]==2.0`, `@2.0` — or None for a bare name (#3206).
    #:
    #: #688 throws this away on purpose (`foo[extra]>=1,<2` → `foo`), because its two facts
    #: are facts about the NAME and a specifier is noise to them. The hold this item adds is
    #: a fact about the ARTIFACT, so the specifier is the claim and has to reach the policy:
    #: this field is the raw text, and `target_version` decides which release it names,
    #: because only an exact `==` names one and everything else means "whatever the registry
    #: currently calls latest". A `None` here is therefore a real claim ("nothing was
    #: pinned"), not a missing value.
    version_spec: str | None = None

    @property
    def pins_exact_version(self) -> bool:
        """Whether this request names ONE release rather than a range or "whatever is new".

        The dispatch path asks this before spending a registry read on a name it already
        declared (#3206): an exact pin is a claim about a specific artifact and is worth a
        lookup, and everything else is a claim about the registry's latest release, which on
        a declared name would refuse an old artifact just because a newer one exists.
        """
        return exact_pin(self.version_spec) is not None


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


#: The first character that can begin a PyPI specifier or an extras block.
_SPEC_START_RE = re.compile(r"[\[\]=<>!~;]")


def _version_spec(prog: str, tok: str) -> str | None:
    """What `_strip_version_spec` throws away: the specifier that followed the name (#3206).

    #688 discards it deliberately — `foo[extra]>=1,<2` and `foo` are one name to its two
    facts. The hold this item adds is a fact about the ARTIFACT, so the specifier is the
    claim itself and has to survive the parse. This returns the text after the name, trimmed
    (`==2.2.0`, `>=1.2,<2`, `[cli]==2.0`, `@2.0`, `;python_version<...`), or None for a bare
    name.

    Deliberately NOT decided here: which of those shapes names an exact release. Only an
    exact `==` does, and answering that for the others needs the registry's own latest
    version — which is `target_version`'s job, on the other side of the lookup.
    """
    body = tok.strip()
    if prog in _NPM_FAMILY:
        if body.startswith("@"):
            m = _SCOPED_RE.match(body)
            tail = body[m.end():] if m else ""
        elif "@" in body:
            tail = "@" + body.split("@", 1)[1]
        else:
            tail = ""
        return tail.strip() or None
    if " @ " in body:                        # PEP 508 direct reference: `name @ url`
        return ("@ " + body.split(" @ ", 1)[1].strip()) or None
    if " @" in body:
        return ("@" + body.split(" @", 1)[1]) or None
    m = _SPEC_START_RE.search(body)
    if not m:
        return None
    # The match is the first character of the specifier, so the slice already carries the
    # whole of it: `mcp==2.2.0` → `==2.2.0`, `pkg[cli]==2.0` → `[cli]==2.0`.
    return body[m.start():].strip() or None


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
         name: tuple[str, bool] | None, *, tok: str = "") -> list[InstallRequest]:
    if not name:
        return []
    nm, vcs = name
    ecosystem = _ECOSYSTEM_BY_PROGRAM.get(prog, _Ecosystem_PYPI)
    return [InstallRequest(name=nm, ecosystem=ecosystem, verb=verb,
                           vetted=ecosystem == _Ecosystem_PYPI, vcs=vcs,
                           version_spec=_version_spec(prog, tok) if tok else None)]


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
        # `tok` still holds the specifier `_strip_version_spec` removed for `nm`, and #3206's
        # hold is priced on exactly that: an operand written `mcp==2.2.0` is a claim about a
        # specific release, and one written `mcp` is a claim about whatever is newest.
        out.append(InstallRequest(name=nm, ecosystem=ecosystem, verb=verb,
                                  vetted=ecosystem == _Ecosystem_PYPI, vcs=vcs,
                                  version_spec=_version_spec(prog, tok)))
    # `-r other.txt`: the names declared inside that file are being installed too.
    for req in _requirement_files(prog, args):
        try:
            text = Path(req).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # The pin travels with the name: an `install -r requirements.lock` asks for the
        # EXACT version that file freezes, and a hold priced on the registry's latest release
        # instead would refuse an install of a six-month-old artifact because some other,
        # newer release exists. That is the false-block shape #688's own history warns about.
        for nm, spec in _requirements_declared(text):
            out.append(InstallRequest(name=nm, ecosystem=ecosystem, verb=verb,
                                      vetted=ecosystem == _Ecosystem_PYPI,
                                      source=req, version_spec=spec))
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
#: How long a distribution VERSION must be public before an unattended install may take
#: it (#3206). Note the near-miss with the constant above: `MIN_FIRST_RELEASE_AGE_DAYS`
#: is the age of the *name* (its oldest release), this is the age of the *artifact being
#: downloaded* (the release named on the command line, or the latest one when the command
#: names none). Rybka's exhibit is a working MCP utility that shipped 15 clean releases
#: and added one line at v16 which BCC-ed the owner on every email: no fact about the
#: name saw that release, and #688's two facts are both facts about the name. 14 days is
#: his own recipe ("ingest it maybe in a week or two"), taken as the shorter of the two so
#: the hold bites on a two-week-old release rather than a three-week-old one.
MIN_RELEASE_AGE_DAYS = 14
#: Cap on how many release dates one snapshot carries (#3206). A SIZE bound, not a
#: knowledge bound: the map keeps the NEWEST `MAX_VERSION_DATES` dated releases, and the
#: entries it can drop are exactly the ones a hold could never reach. Dropping one makes
#: precisely one thing true about that version — it is unmeasurable — because absence from
#: this map has exactly one meaning (see `RegistryFacts.version_upload_times`), so the cap
#: can never turn "old" into "young" or the reverse. An entry costs ~40 bytes per release
#: carried, and this file's own `parse_pypi_payload` docstring records a live payload
#: carrying 317 release keys for one ordinary name — so the cap is what makes this map safe
#: to write into the cache file at all, rather than a bound nobody needs.
MAX_VERSION_DATES = 400
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
#: The bound to read beside `min_release_age_days` below, because it is what makes a
#: version fact trustworthy rather than merely recent. An upload time is an absolute
#: instant, so a date that IS in a cached snapshot is exactly as true on day 6 of the
#: cache as on day 0 — the cache never ages a number. What a cache entry does expire into
#: is COVERAGE: a snapshot written 6 days ago cannot contain a release published since
#: then, and that missing release is the one a hold exists for. `_registry_facts` is the
#: rule that pays for it — any version claim the cached map does not date is re-read live
#: rather than answered from the entry — so no verdict in this module is ever priced on a
#: release the snapshot could not have seen.


@dataclass(frozen=True)
class Thresholds:
    min_first_release_age_days: int = MIN_FIRST_RELEASE_AGE_DAYS
    min_release_count: int = MIN_RELEASE_COUNT
    #: The #3206 hold, in days, on the VERSION rather than on the name.
    #:
    #: Its bound is `CACHE_TTL_DAYS = 7`, the age of the per-name registry cache, and the
    #: two interact in one direction only. An upload time is an absolute instant, so a
    #: date the cached map DOES carry is exact for the whole 7 days and can never make a
    #: young release look old. What the cache can do is fail to contain a release published
    #: after it was written — which is why a version claim whose pin the cached map does not
    #: date is read live (`_registry_facts`) instead of being answered from the entry. What
    #: remains, and what this threshold then also governs, is the entry that is OLDER than
    #: the hold itself: such a snapshot cannot date anything inside its own window, and is
    #: answered `version-age-stale-cache` (allowed, named, not vetted) rather than treated as
    #: evidence of age. So raising this threshold above `CACHE_TTL_DAYS` can never turn a
    #: stale cache into a silent hold, and `min_release_age_days=0` means the rule is inert,
    #: since no release can be younger than zero days.
    min_release_age_days: int = MIN_RELEASE_AGE_DAYS


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
    #: Upload time of each release this snapshot could date, keyed by version string
    #: (#3206). The map has ONE rule and it holds for a live read, a warm cache read and a
    #: fixture alike: **a version in it has a measured release date; a version absent from it
    #: has no date, and no version is ever read as old because it is missing.** Absence is
    #: therefore never evidence of age — not "older than some horizon", which is the reading
    #: an earlier shape of this field invited and the review rung disproved on the shipped
    #: code (a release published after the snapshot is absent from it for the very same
    #: reason a two-year-old one is, and it is the release the hold exists for). What
    #: absence does say is that the age could not be measured, which the policy answers by
    #: allowing the install and naming the fact (`version-age-unknown`), the same honesty
    #: rule that already keeps `first_release=None` from reading as "old". Entries are the
    #: newest `MAX_VERSION_DATES` dated releases, so a cache entry is bounded and a dropped
    #: release loses only a date OLDER than every date kept — for the drop to cost anything,
    #: one distribution would have to publish more than `MAX_VERSION_DATES` releases INSIDE
    #: the hold window.
    version_upload_times: dict[str, dt.datetime] = field(default_factory=dict)
    #: What the registry itself calls the latest release (`info.version`, falling back to
    #: the dated release with the newest upload time). An unpinned install takes THIS
    #: artifact, so an unpinned install is judged by its age rather than skipped: `""` means
    #: the payload named no latest release, which is unmeasurable, not "old".
    latest_version: str = ""

    @property
    def unreachable(self) -> bool:
        return self.source == "unreachable"

    def dates_version(self, version: str | None) -> bool:
        """Whether this snapshot carries a release date for `version`.

        The one question that decides whether a cached entry may answer a version claim
        (`_registry_facts`): a date in the map is an absolute instant and stays true for the
        whole cache TTL, while an entry the map does not have may be a release published
        after the snapshot was taken — the one shape a hold must not guess about.
        """
        return version is not None and version in self.version_upload_times


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
    vtimes = release_version_times(releases)
    latest = str((payload.get("info") or {}).get("version") or "").strip()
    if not latest:
        # `info.version` is the authoritative answer and PyPI always sends one, but a
        # trimmed blob or a mirror payload may not: fall back to the newest release we can
        # actually date, and if none is datable leave it empty, so an unpinned install is
        # judged `version-target-unknown` rather than on a guess about which artifact pip
        # would resolve.
        latest = max(vtimes, key=lambda v: vtimes[v]) if vtimes else ""
    if not times and payload.get("info", {}).get("version"):
        # A release list with no timestamps is not "old": record what is unknown
        # and let the caller fail open rather than assert an age it cannot see. The
        # version map travels with it for the same reason — an undated payload yields an
        # EMPTY map — so the #3206 hold reads it as "no version is datable", never as
        # "every version is old enough".
        return RegistryFacts(name=name, exists=True, first_release=None,
                             release_count=count or None, source="pypi.org",
                             observed_at=obs, version_upload_times=vtimes,
                             latest_version=latest)
    return RegistryFacts(name=name, exists=bool(releases) or bool(payload.get("urls")),
                         first_release=min(times) if times else None,
                         release_count=count or None, source="pypi.org", observed_at=obs,
                         version_upload_times=vtimes, latest_version=latest)


def release_version_times(releases: dict[str, Any], *,
                          max_versions: int = MAX_VERSION_DATES,
                          ) -> dict[str, dt.datetime]:
    """Date every release in the payload that HAS a date. Nothing is dated out by age.

    The live payload shape is `releases: {version: [file, …]}`, one `upload_time` per file,
    and #688's reader collapsed the whole map into `min(times)` plus a key count — two facts
    about the NAME. This keeps the per-release dates #3206's hold needs under one rule: a
    version is in the result exactly when one of its files carries a usable `upload_time`,
    and the value is the newest of them (the artifact an installer would fetch).

    No age filter, deliberately, because absence is the one statement this map makes and it
    must have one meaning: **"this snapshot could not date that version"**, which the policy
    answers by allowing the install and naming the fact. Filtering by age makes the same
    absence mean two different things depending on which snapshot is being read — for one
    snapshot it means "measured older than the cut", for the other it means "not published
    yet when we looked", and the second is the release a hold exists for. That two-meaning
    shape is what the review rung refused on the previous round's commit, and it is the
    reason a cut of any width cannot be repaired by a flag saying which snapshot this is: a
    release published after a snapshot is invisible to it however wide the cut is drawn.

    `max_versions` is a SIZE bound. Keeping the newest entries is what makes it safe: every
    dropped release is older than everything kept, so the most a drop can do is turn
    "measured old, allowed" into "unmeasurable, allowed".
    """
    out: dict[str, dt.datetime] = {}
    for raw_version, files in releases.items():
        version = str(raw_version or "").strip()
        if not version or not isinstance(files, list):
            continue
        stamps = [stamp for stamp in (
            _as_utc(item.get("upload_time") or item.get("upload_time_iso_8601"))
            for item in files if isinstance(item, dict)
        ) if stamp is not None]
        if stamps:
            out[version] = max(stamps)
    ceiling = max(int(max_versions), 0)
    if len(out) > ceiling:
        out = dict(sorted(out.items(), key=lambda item: item[1],
                          reverse=True)[:ceiling])
    return out


#: An exact pin and nothing else: `==1.2.3` or the deprecated `=1.2.3`. A wildcard
#: (`==1.*`), a range (`>=`, `<`, `~=`, `!=`), a uv `@2.0` or a bare name all fail it, and
#: every one of those means "whatever the resolver picks", which in practice is the newest
#: artifact on the registry. One regex, because the policy's two questions — which release
#: is this claim about, and is this claim about a release at all — must not answer
#: differently for the same string.
_EXACT_PIN_RE = re.compile(r"^(?:==|=)\s*(?!.*[*?])\S+$")


def exact_pin(version_spec: str | None) -> str | None:
    """The version an exact pin names, or None when the specifier does not name one.

    `==2.2.0` → `"2.2.0"`; `[cli]==2.0` → `"2.0"`; `>=1.2,<2`, `==1.*`, `@2.0`, `""` and
    None → None. Extras are stripped first because the shipped parser lowercases the token
    and does not strip them here, so a pin on an extras-bearing name is still a pin.
    """
    spec = (version_spec or "").strip()
    if spec.startswith("["):                     # `pkg[cli]==2.0` → the spec carries extras
        spec = spec[1:].split("]", 1)[-1].strip()
    if not _EXACT_PIN_RE.match(spec):
        return None
    return spec.split(None, 1)[-1].lstrip("= ") or None


def target_version(facts: RegistryFacts, version_spec: str | None = None,
                   ) -> tuple[str, str]:
    """Which artifact a command actually asks for: ``(version, how)``, ``how`` ∈ pinned/latest/unknown.

    Naming the version the claim is about is the whole of #3206 — a hold on "the version"
    is meaningless if the check cannot say which one it priced — and the three command
    shapes answer differently:

      * ``pkg==1.2.3`` → ``("1.2.3", "pinned")``. The claim is about THAT release.
      * ``pkg`` (bare), ``pkg>=1.2``, ``pkg<3``, ``pkg!=1.0``, ``pkg~=2.0``, ``pkg==1.*``,
        ``pkg@2.0``, ``pkg[extra]`` → the registry's own latest release, ``how="latest"``.
        A range resolves at resolve time to the newest artifact available, and a wildcard
        or a uv ``@`` names no exact release we could date either, so the honest claim is
        "what you are about to fetch is the latest one". An unpinned install is the shape
        that silently takes a day-old release, which is exactly why it is not skipped.
      * neither the command nor the registry names a version → ``("", "unknown")``.
    """
    pinned = exact_pin(version_spec)
    if pinned:
        return pinned, "pinned"
    if facts.latest_version:
        return facts.latest_version, "latest"
    return "", "unknown"


def evaluate_provenance(facts: RegistryFacts, *, now: dt.datetime | None = None,
                        thresholds: Thresholds = DEFAULT_THRESHOLDS,
                        version_spec: str | None = None) -> Provenance:
    """The four-fact policy, as a pure function of (facts, clock, thresholds, requested version).

    Pure so the fixture eval and the real dispatch path cannot drift: there is
    one place that turns a registry fact into a refusal, and a threshold change
    moves both at once.

    Three of the four facts are facts about the NAME — how old its first release is, how
    many releases it has, whether the registry has heard of it at all — and they are
    #688's shipped contract. The fourth is a fact about the ARTIFACT (`version_spec`, the
    specifier the command carried, or none): #3206's hold window on the age of the release
    being fetched. Rybka's poisoned MCP utility had 15 clean releases and a four-year-old
    first release, so all three name facts passed it and only this one can see it.

    The version facts are asked LAST, after the name facts, so a name that is already
    refused for being new or thin is not also charged a version verdict it may not have
    earned, and so an existing refusal's `fact` stays the one a journal reader already
    knows. `version_spec=None` is a real claim, not an absent one: it means the command
    pinned nothing, and it is priced against the registry's own latest release.
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

    # ---- the artifact's own age (#3206) -------------------------------------
    version, how = target_version(facts, version_spec)
    if how == "unknown":
        # The hold is INERT here, which is a different thing from the two fail-opens below
        # and worth the distinction. `unknown` means there is no version CLAIM to price:
        # the command pinned nothing and the payload named no latest release either — and a
        # payload that dates nothing at all has already been answered by
        # `first-release-unknown` above, so what reaches here carries no version
        # information of any kind (a cache blob written before #3206, a hand-built fixture).
        # There is no artifact to hold, the name facts have done their job, and the shipped
        # verdict stands. The two branches below are the fail-opens proper: there the
        # target IS nameable and only its AGE is missing, and those stay distinctly named,
        # because reading that as vetted is the folding #688's finding warns about.
        return Provenance(blocked=False, fact="cleared", reason="", facts=facts)
    observed = facts.observed_at
    if observed is None or (clock - observed).days > thresholds.min_release_age_days:
        # A version fact is only as fresh as the observation it came from, and facts are
        # cached per NAME for CACHE_TTL_DAYS. An entry older than the hold cannot date
        # anything inside it, and refusing on that would let a stale cache file block a
        # real install — the false-block shape #688's first month actually measured (four
        # denials, every one of them a parser false-block). So staleness allows.
        return Provenance(blocked=False, fact="version-age-stale-cache", reason="",
                          facts=facts)
    if not facts.dates_version(version):
        # This snapshot has no release date for the version the command is about to take,
        # and that is the ONE thing absence from `version_upload_times` is ever allowed to
        # mean. It is never evidence of age — not for a payload whose files carry no
        # timestamps, not for a cache entry written before the version map existed, not for a
        # release published after the snapshot was fetched, and not for a tail beyond
        # `MAX_VERSION_DATES` in a 317-release name. All of those answer alike: allowed, on
        # the unknown, and DISTINCTLY NAMED, because folding them into `cleared` is what
        # turns a fail-open into a pass (#688's finding). The same rule already keeps
        # `first_release=None` from reading as "old". `_registry_facts` closes the one gap
        # where this matters operationally: a version claim whose pin a cached entry cannot
        # date re-reads the registry live rather than being answered from the entry.
        return Provenance(
            blocked=False, fact="version-age-unknown",
            reason=(f"'{facts.name}' version {version} has no release date in the registry "
                    f"facts (observed {observed.date().isoformat()} via {facts.source}), so "
                    f"its age could not be measured and the hold cannot price it. Allowed on "
                    f"the unknown — NOT vetted: absence from a set of dated releases is not "
                    f"evidence of age in either direction. If you know the release date and "
                    f"it is inside the "
                    f"{thresholds.min_release_age_days}-day hold, do not override to take "
                    f"it; if it is outside, re-run with {OVERRIDE_ENV}=\"why this version\"."),
            facts=facts)
    upload = facts.version_upload_times[version]
    v_age = (clock - upload).days
    if v_age < thresholds.min_release_age_days:
        which = ("that is the registry's current latest release, so an unpinned "
                 "install would take whatever is newest"
                 if how == "latest" else "you pinned that version")
        return Provenance(
            blocked=True, fact="version-too-recent",
            reason=(
                f"'{facts.name}' version {version} was published {v_age} day(s) ago "
                f"(released {upload.date().isoformat()}; hold: "
                f"{thresholds.min_release_age_days} day(s) on a version, judged from "
                f"facts observed {observed.date().isoformat()} via {facts.source}; "
                f"{which}. The name itself is older and passed its own checks, which "
                f"is the point: a name's history is not evidence about its NEXT "
                f"release. The exhibit this hold was written from is a working MCP "
                f"utility with 15 clean releases that added one line at v16. If this "
                f"release is a security fix, do not override blind — vendor the "
                f"artifact instead: the files and their hashes are at "
                f"https://pypi.org/simple/{facts.name}/, and vendoring the wheel into "
                f"the offline index and pointing `pip` at it with `--index-url` or "
                f"`PIP_INDEX_URL` is how the rest of this box installs. If you mean "
                f"to take the version anyway, re-run with "
                f'{OVERRIDE_ENV}="why this exact version", which journals an override '
                f"with your reason instead of looking like a clean install."),
            facts=facts)
    # dated, and dated older than the hold: this one is a measurement, and it passes on it
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

    def lookup(self, name: str, *, fresh: bool = False) -> RegistryFacts:
        """Facts for `name`; `fresh=True` asks the registry, skipping the cache and the memo.

        #3206's version hold is why that parameter exists, and `_registry_facts` is the only
        caller that sets it, because it is the only caller that knows which of the two cache
        failures a decision can survive:

        * A date the entry DOES carry is exact for the whole `CACHE_TTL_DAYS` — an upload
          time is an absolute instant, and the cache never ages a number. A pinned version
          the entry dates is therefore served from the cache at zero cost.
        * What the entry cannot do is speak about a release it never saw. A snapshot written
          6 days ago has no entry for a release published yesterday, and that release is the
          one the hold exists for. Answering from the entry there is answering
          "unmeasurable" — allowed, on the honesty rule — about precisely the artifact the
          check was written to catch. So `_registry_facts` re-reads live whenever the cached
          map does not date the pin, and always for an unpinned install, where even a
          perfectly-dated `latest_version` has expired as a description of what pip fetches.

        The cost is bounded by the traffic this path actually sees: #688 measured about two
        executed installs per 30 days on this box, each inside `DEFAULT_REGISTRY_TIMEOUT`
        (1.5 s), failing open on any transport error, and attended turns never reach the
        registry at all. The pin that keeps the guard from living only in the policy is
        `tests/test_supply_chain_provenance.py::
        test_a_pinned_release_published_after_the_cached_snapshot_re_reads_the_registry` —
        a test that only ever planted a fresh snapshot by hand would be satisfied by an
        implementation that served the cache always.
        """
        import logging
        logger = logging.getLogger("lloyd-supply-chain")
        key = normalize_dist_name(name)
        now = dt.datetime.now(dt.timezone.utc)
        if not fresh:
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


#: Sentinel for "this lookup is not about a version at all", distinct from `pin=None`, which
#: means "it is about a version and the command pinned nothing". Two meanings, one parameter,
#: and the difference decides whether the 7-day cache may answer (`_registry_facts`).
NO_VERSION_CLAIM = object()


def _registry_facts(registry: Any, name: str, *,
                    pin: Any = NO_VERSION_CLAIM) -> RegistryFacts:
    """The facts this decision is priced on, choosing the read that may answer it.

    Three states of one parameter, because the 7-day per-name cache is admissible for one of
    them and not for the other two:

    * **`pin` is a version string.** A date the cached entry holds for that version is an
      absolute instant and stays true for the whole `CACHE_TTL_DAYS` — the cache never ages a
      number — so the cache is used when it DATES THE PIN (`RegistryFacts.dates_version`) and
      the decision costs no round trip. When it does not date it, the cache may not answer:
      the missing version may be a release published after the entry was written, and
      answering from the entry would report the exact artifact a hold exists for as
      unmeasurable. So it reads live.
    * **`pin` is None** — a bare name, a range, an extra. The artifact is whatever the
      registry calls latest NOW, and that identity is what expires. A 6-day-old entry whose
      `latest_version` it dates perfectly still misses a release published yesterday, and an
      unpinned install is precisely the shape that silently takes one. So the cache is never
      admissible and the read is live.
    * **`NO_VERSION_CLAIM`** — #688's original read, for a caller pricing only the name facts.
      Cache first, because the two facts it uses (first release, release count) go stale in
      the permissive direction only, which is what a TTL is for.

    A registry double that has never heard of `fresh` is served the plain read rather than
    failing the decision: its own answer is what the caller planted, and raising there would
    turn a test seam into a fail-open the shipped path never takes. The round-trip cost is
    bounded by the traffic this path actually sees — #688 measured about two executed installs
    per 30 days on this box — and capped at `registry_timeout()` per lookup.
    """
    if pin is NO_VERSION_CLAIM:
        return registry.lookup(name)
    if pin is not None:
        cached = registry.lookup(name)
        if cached.dates_version(pin):
            return cached
    try:
        return registry.lookup(name, fresh=True)
    except TypeError:
        return registry.lookup(name)


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
                         source="cache", observed_at=stamp,
                         version_upload_times=_versions_from_blob(row.get("version_times")),
                         latest_version=str(row.get("latest_version") or ""))


def _versions_from_blob(raw: Any) -> dict[str, dt.datetime]:
    """Decode the cached version map. Wire form: ``{version: iso-date}``.

    A version is kept only if its stamp parses, because a map entry is the ONLY form in
    which this cache may say "I have a release date for that version" — absence is the form
    for "I do not", and the policy never reads absence as age (#3206: the review rung
    refused a commit that let a cached entry answer "older than the horizon" about a release
    it had never seen, since a release published after the entry was written is absent for
    exactly the same reason). So a malformed, missing or pre-#3206 field decodes to an EMPTY
    map, which the policy reads as "nothing here is dated" and fails open on — never as
    "every version is old". A `null` stamp — what an undated release looked like on the wire
    before this round, and what a half-written entry could still hold — decodes to the same
    absence as an absent key, which is the only pair of states this decoder is allowed to
    equate.
    """
    out: dict[str, dt.datetime] = {}
    if not isinstance(raw, dict):
        return out
    for version, stamp in raw.items():
        version = str(version or "").strip()
        if not version:
            continue
        moment = _as_utc(stamp)
        if moment is not None:
            out[version] = moment
    return out


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
        # #3206: this codec is a hand-written field-by-field projection, NOT an `asdict`, so
        # a field added to RegistryFacts but not to BOTH halves of it silently vanishes on
        # every warm read for CACHE_TTL_DAYS — which would make the version hold fire at most
        # once a week per name, i.e. mostly not at all. Only dated releases are written: a
        # version with no date has no key here, which is the same statement the decoder makes
        # about it, and `MAX_VERSION_DATES` is what bounds the entry's size.
        "version_times": {v: t.isoformat()
                          for v, t in sorted(facts.version_upload_times.items())},
        "latest_version": facts.latest_version,
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
# The dispatch check `check_install_provenance` — the sixth check inside
# safety.check_bash_command
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


def _declared_hold(request: InstallRequest, reg: Any, notes: list[str], *,
                   now: dt.datetime | None = None,
                   thresholds: Thresholds = DEFAULT_THRESHOLDS) -> Provenance | None:
    """Price a DECLARED name's pinned version against #3206's hold, or explain why it cannot.

    Returns the blocking `Provenance` when the artifact this exact command asks for is
    inside the hold window, and `None` when the install may proceed — which includes every
    case where the age could not be measured, each with a note. The name being in
    `requirements.txt` is not evidence about a release: the exhibit this hold exists for was
    a declared, years-old package whose v16 — after 15 clean releases — went out yesterday.

    The three ways this returns None without a verdict are all recorded in `notes` and all
    keep the name's outcome `declared`, exactly as #688 clause 5 requires:

      * the registry read raised — a lookup failure must never block a turn;
      * the registry is unreachable — no verdict, and not a pass either;
      * the policy itself allowed it, including its own `version-*` fail-opens.
    """
    try:
        # The read is chosen by the claim: an exact pin may be answered from the cache if the
        # cached snapshot dates it, and anything else (a bare name, a range) has to be live
        # because "latest" is the artifact and that identity expires.
        facts = _registry_facts(reg, request.name, pin=exact_pin(request.version_spec))
    except Exception as exc:  # noqa: BLE001 — fail OPEN, never block a turn
        notes.append(f"{request.name}: declared name, and its pinned version was never "
                     f"dated because the registry read raised {type(exc).__name__} "
                     f"({exc}); version not checked")
        return None
    if facts.unreachable:
        notes.append(f"{request.name}: declared name, but the registry was unreachable so "
                     f"its pinned version was never dated (not checked, not allowed)")
        return None
    verdict = evaluate_provenance(facts, now=now, thresholds=thresholds,
                                  version_spec=request.version_spec)
    if not verdict.blocked or not verdict.fact.startswith("version-"):
        # A DECLARED name is not judged on the name facts: #688 clause 5 is that a name the
        # repo already lists is never classed as new, and this lookup exists only to date one
        # release. A fixture or a stale cache that answers 404 for a name sitting in
        # `requirements.lock` is therefore not a refusal here — it is the registry failing to
        # answer the one question we asked, and it is said in a note rather than charged to
        # the install.
        if verdict.blocked:
            notes.append(f"{request.name}: declared name, so its {verdict.fact} registry "
                         f"answer was not charged to this install; only the pinned "
                         f"version's age was")
        elif verdict.fact.startswith("version-"):
            # The fail-open half, and the only place it can be seen: the name's outcome
            # stays `declared` (the declaration did license the install), so without this
            # line a held-eligible version whose age could not be measured would leave no
            # trace at all, and "we could not tell" would read as "we checked".
            notes.append(f"{request.name}: declared name, install allowed, but the pinned "
                         f"version's age was never measured ({verdict.fact}) — not vetted, "
                         f"and not held")
        return None
    return verdict


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
            # #688 clause 5: a name already listed is never classed as new, and that
            # outcome stays `declared` here whatever the version rule decides. But the
            # name's own seniority says nothing about its NEXT release — the exhibit this
            # hold was written from was a declared, years-old package that added one line at
            # v16, after 15 clean releases — so a command that names ONE exact release of a
            # declared name is priced against the hold, and only that shape.
            #
            # Not priced: a bare or ranged install of a declared name. Its target is
            # whatever the registry currently calls latest, so holding it would refuse an
            # install of an artifact this repo already ships whenever any newer release
            # exists — #688's own false-block record (4 denials, all parser false-blocks on
            # `list`/`freeze`/`check`/`show`) is the cautionary precedent, and refusing a
            # 6-month-old artifact because a newer one exists is the same class of error.
            if request.pins_exact_version and not request.vcs and request.vetted:
                held = _declared_hold(request, reg, result.notes, now=now,
                                      thresholds=thresholds)
                if held is not None:
                    if request.override:
                        # Deliberately NOT appended to `result.declared`: the journal
                        # projects declared last, so adding it here would overwrite this
                        # row's own `overridden` outcome with `declared` and the override
                        # would vanish from the only durable record of it.
                        result.overrides.append((request.name, request.override.strip()))
                        result.cleared.append(request)
                        logger.warning("supply-chain: %s of %r version %s held by %s and "
                                       "overridden: %s", request.verb, request.name,
                                       exact_pin(request.version_spec), held.fact,
                                       request.override.strip()[:200])
                        continue
                    result.refusals.append((request, held))
                    logger.warning("supply-chain: %s of %r version %s held by %s: %s",
                                   request.verb, request.name,
                                   exact_pin(request.version_spec), held.fact,
                                   held.reason[:200])
                    continue
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
            # Chosen the same way as the declared branch, because the verdict this call feeds
            # is priced on a version too (the pinned one, or the registry's latest) — and a
            # cached snapshot is admissible for the first and not the second.
            facts = _registry_facts(reg, request.name, pin=exact_pin(request.version_spec))
        except Exception as exc:  # noqa: BLE001
            result.unvetted.append(request)
            result.notes.append(f"{request.name}: the registry lookup raised "
                                f"{type(exc).__name__}, so it was never vetted")
            logger.warning("supply-chain: registry lookup raised for %r (%s: %s)",
                           request.name, type(exc).__name__, str(exc)[:160])
            continue
        # The version fact is priced on the SAME call for a new name as for a declared one
        # (#3206): a first release that is four years old and 40 releases deep does not vet
        # yesterday's artifact, and skipping the specifier here would leave the rule
        # guarding only the half of the traffic #688 already looked at.
        verdict = evaluate_provenance(facts, now=now, thresholds=thresholds,
                                       version_spec=request.version_spec)
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

#: The artifact's own schema id, named so a test and a reader can cite it instead
#: of repeating the literal — this file used to type it twice, in `to_dict` and in
#: `write_baseline`'s copy, which is how a schema bump lands on one of them.
BASELINE_SCHEMA = "lloyd-supply-chain-baseline/1"

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
    #: The tree this record describes — path, commit sha and file list, from
    #: `tree_record`. An artifact without it names an address, not a tree, and a
    #: worktree that got deleted is then unanswerable (#2593 clause 1).
    tree: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    #: True when this record was written as an explicit coverage-gap disclosure
    #: rather than as the result of a scan.
    coverage_gap: bool = False

    @property
    def verdict_label(self) -> str:
        """The verdict this record carries *now*, staleness included.

        The single definition: the YAML field and `main`'s printed line both read
        it, so a scan run against a nine-day-old database cannot print
        `no_advisories` while the file it just wrote says `stale` (#2593 clause 5).
        The exit code stays on `scan_status` — a stale scan *did* run, and exit 3
        means "no instrument", which is a different complaint. The status verdict
        is computed first because
        `stale_stamps` speaks only about an artifact that claimed
        `no_advisories` — a coverage gap and a real finding are already honest
        about themselves.
        """
        verdict = status_verdict(self.advisory_count)
        if stale_stamps({"verdict": verdict, "generated_at": self.generated_at,
                         "database": {"directory": self.db_directory,
                                      "freshness": self.db_freshness}}):
            return STALE_VERDICT
        return verdict

    def to_dict(self) -> dict[str, Any]:
        """The artifact as written — `verdict_label` and `stale` included, so the
        file says which of its own clocks ran out."""
        doc: dict[str, Any] = {
            "schema": BASELINE_SCHEMA,
            "generated_at": self.generated_at or _utc_stamp(),
            "scan_status": self.scan_status,
            "verdict": self.verdict_label,
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
        if self.tree:
            doc["tree"] = dict(self.tree)
        # Judged on the status verdict, not on `doc["verdict"]`, which may already
        # read `stale`: see `status_verdict`.
        doc["stale"] = stale_stamps({**doc,
                                     "verdict": status_verdict(self.advisory_count)})
        return doc


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
                      generated_at=_utc_stamp(), tree=tree_record(root))


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
# How long a verdict stays true, and which tree it describes
# ---------------------------------------------------------------------------

#: A `no_advisories` verdict is present tense, and the dataset behind it moves
#: daily: OSV publishes new advisories against versions this tree already runs,
#: so the verdict decays without anything in the tree changing. The committed
#: `baseline.yaml` sat at `generated_at: 2026-10-01` / `database.freshness:
#: 2026-09-30` while carrying `verdict: no_advisories` — 10 and 11 days, with
#: nothing on the box regenerating it or calling either number old (#2593).
#: These two ceilings are the verdict's shelf life, as constants so a cadence, a
#: doc and a test all read the same number.
STALE_GENERATED_MAX_DAYS = 3
#: The mirrored OSV database is a second, independent clock: a scan run today
#: against a nine-day-old mirror is fresh paperwork over stale knowledge, and the
#: failure mode below already says as much about a *missing* mirror.
STALE_DB_FRESHNESS_MAX_DAYS = 7
#: The verdict a stale artifact is allowed to carry. Not `none_recorded`, which
#: means "no scan ran", and not `no_advisories`: advisories may have landed the
#: day after the stamp, and only the reader can tell those two apart.
STALE_VERDICT = "stale"


def _parse_stamp(value: Any) -> dt.datetime | None:
    """An ISO-8601 artifact stamp as an aware UTC datetime, or None.

    A stamp this cannot read counts as aged rather than fresh: an artifact whose
    clock is unreadable cannot claim to have been written a moment ago.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        moment = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def _live_checkout() -> Path:
    """The checkout a committed artifact is allowed to describe.

    Read through `app.paths.LIVE_CHECKOUT` — passwd-derived, so a gate round with
    `$HOME` redirected still names the production tree — and looked up per call,
    never cached at import: a checker that plants its own root has to be able to
    say so.
    """
    try:
        from app.paths import LIVE_CHECKOUT
        return Path(LIVE_CHECKOUT)
    except Exception:  # noqa: BLE001 — a checker still checks without app.paths
        return Path(__file__).resolve().parents[2]


def tree_record(root: Path | str | None = None) -> dict[str, Any]:
    """The tree a scan describes: its path, its commit, and its files by name.

    The committed baseline named only absolute paths, so the round that produced
    it could — and did — record a worktree that no longer exists. A commit sha is
    what keeps the claim checkable after the directory is gone:
    `git show <sha>:requirements.lock` still answers it.
    """
    base = _repo_root(root)
    sha = branch = None
    try:
        proc = subprocess.run(["git", "-C", str(base), "rev-parse", "HEAD",
                               "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True, check=False, timeout=10)
        if proc.returncode == 0:
            lines = proc.stdout.split()
            sha = lines[0] if lines else None
            branch = lines[1] if len(lines) > 1 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"root": str(base), "commit": sha, "branch": branch,
            "worktree": str(base) != str(_live_checkout()),
            "files": [str(f.relative_to(base)) for f in dependency_files(base)]}


class BaselineTargetOutsideCheckout(RuntimeError):
    """The scan's input is not a file a reader can go and look at.

    Raised rather than written past, because the alternative is a committed
    artifact attesting "no advisories" about a tree nobody can re-examine — which
    is what shipped once (`scanner.argv` naming
    `/home/alansrobotlab/lloyd-hand/G/requirements.txt`, a path that no longer
    exists) and stayed green for 40 commits because nothing read the field.
    """


def scan_target_paths(argv: Sequence[str]) -> list[str]:
    """The lockfile paths a scanner argv names, format prefixes stripped.

    `--lockfile requirements.txt:/abs/requirements.lock` carries a format label
    and a path in one token; the path is the thing that has to exist. The split is
    POSIX-only by design — a Windows drive-letter path would be mis-split here,
    and this box has never had one.
    """
    out: list[str] = []
    take = False
    for token in argv:
        if take:
            take = False
            text = str(token)
            head, sep, tail = text.partition(":")
            out.append(tail if sep and re.fullmatch(r"[A-Za-z0-9_.-]+", head) else text)
            continue
        take = str(token) == "--lockfile"
    return out


def check_scan_targets(argv: Sequence[str], *, root: Path | str | None = None,
                       live_root: Path | str | None = None) -> list[str]:
    """Every scan target that is not a real file inside the live checkout.

    Relative names resolve against `root` (the tree the scan ran over); an
    absolute path has to sit under the live checkout. Empty means every target in
    `argv` is a file a reader can open.
    """
    base = _repo_root(root)
    live = Path(live_root) if live_root else _live_checkout()
    offenders: list[str] = []
    for raw in scan_target_paths(argv):
        path = Path(raw)
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            offenders.append(f"{raw}: not a file")
            continue
        try:
            path.resolve().relative_to(live.resolve())
        except ValueError:
            offenders.append(f"{raw}: resolves outside the live checkout {live}")
    return offenders


def status_verdict(advisory_count: int | None) -> str:
    """The verdict a scan's status alone carries, before any clock is consulted.

    Separate from `effective_verdict` because staleness is judged on this one: an
    artifact already downgraded to `stale` would otherwise stop reporting its own
    stale reasons on the next read, and a reader of `stale: []` beside
    `verdict: stale` would be told the clocks are fine.
    """
    if advisory_count is None:
        return "none_recorded"
    return "advisories_found" if advisory_count else "no_advisories"


def stale_stamps(doc: dict[str, Any], *,
                 now: dt.datetime | None = None) -> list[str]:
    """Which of the artifact's own stamps no longer support its verdict.

    Empty for an artifact that never claimed `no_advisories` — a coverage gap or a
    real finding does not decay into `stale`; it was never a clean claim — and each
    entry names *which* stamp failed and by how much, because "stale" without a
    clock is a warning nobody can act on.
    """
    if str(doc.get("verdict") or "") != "no_advisories":
        return []
    moment = now or dt.datetime.now(dt.timezone.utc)
    reasons: list[str] = []
    generated = _parse_stamp(doc.get("generated_at"))
    if generated is None:
        reasons.append(f"`generated_at` {doc.get('generated_at')!r} is not a readable "
                       "stamp, so the verdict cannot claim to be current")
    else:
        age = (moment - generated).total_seconds() / 86400.0
        if age > STALE_GENERATED_MAX_DAYS:
            reasons.append(f"`generated_at` is {age:.1f} days old, past the "
                           f"{STALE_GENERATED_MAX_DAYS}-day life of a "
                           "`no_advisories` verdict")
    database = doc.get("database")
    if isinstance(database, dict):
        fresh = _parse_stamp(database.get("freshness"))
        if fresh is None:
            reasons.append("`database.freshness` is missing or unreadable, so the "
                           "advisory data behind the verdict has no date at all")
        else:
            age = (moment - fresh).total_seconds() / 86400.0
            if age > STALE_DB_FRESHNESS_MAX_DAYS:
                reasons.append(f"`database.freshness` is {age:.1f} days old, past the "
                               f"{STALE_DB_FRESHNESS_MAX_DAYS} days an OSV mirror stays "
                               "worth scanning against")
    return reasons


def effective_verdict(doc: dict[str, Any], *,
                      now: dt.datetime | None = None) -> str:
    """The verdict a reader should take from this artifact right now.

    One function answers for both edges — the writer stamps the file, `show`
    re-reads it — so the artifact and the tool can never disagree about what is
    stale. That is the same one-definition rule the scan's exit code follows.
    """
    if stale_stamps(doc, now=now):
        return STALE_VERDICT
    return str(doc.get("verdict") or "none_recorded")


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
    # No artifact that cannot be re-examined gets written, and the check runs
    # before anything is created: a half-built `eval/supply-chain/` in a tree that
    # is about to be refused is itself a lie about what was scanned (#2593 clause 1).
    #
    # A caller may redirect `path` on purpose — the tests do, and so does any
    # operator writing a copy elsewhere — but that is not permission to describe a
    # tree outside the checkout, so the target check reads the report's own argv
    # against the live checkout rather than against where the file lands.
    offenders = check_scan_targets(report.argv, root=root)
    if offenders:
        raise BaselineTargetOutsideCheckout(
            "refusing to write "
            f"{'(re-runnable with `scan --write-baseline`)' if path is None else str(target)}: "
            f"the scan's input is not a file in the live checkout — {'; '.join(offenders)}. "
            "A baseline is a standing claim about a tree, and a path from a deleted "
            "worktree is not a tree: this is how the 2026-10-01 artifact came to "
            "attest `no_advisories` for `/home/alansrobotlab/lloyd-hand/G/`, which "
            "no longer exists (#2593).")
    target.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Supply-chain baseline for the current tree (#688). Committed, "
              "offline, regenerated with\n"
              "#   .venvs/lloyd/bin/python -m app.harness.supply_chain scan "
              "--write-baseline\n"
              "# scan_status: scanner_absent means NO scanner ran and "
              "advisory_count is therefore null —\n"
              "# it is a coverage gap, and must never be read as 'this tree has no "
              "vulnerabilities'.\n"
              "# verdict: stale means the scan ran but its own stamps aged out "
              "(generated_at > 3 days,\n"
              "# database.freshness > 7): the tree was clean *then*, and this file "
              "is not a present-tense\n"
              "# claim until it is regenerated.\n")
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


#: The fixture shape #3206 exists for: a WELL-ESTABLISHED name at a version published
#: yesterday. The two shipped planted shapes (new-name, unpublished-name) are both attacks
#: on the NAME, which is the half #688 already covers; an attacker who has watched a
#: package work for four years does not need a new name, he needs the next release.
FRESH_VERSION_SHAPE = "fresh-version"


def fixtures_registry(doc: dict[str, Any]) -> MappingRegistry:
    """A registry carrying exactly the fixture document's facts, measured or planted."""
    facts: dict[str, RegistryFacts] = {}
    for section in ("planted", "control"):
        for entry in doc.get(section) or []:
            name = str(entry.get("name") or "")
            if not name:
                continue
            blob = entry.get("facts") or {}
            # Section order is meaningful since #3206: `planted` is read first and a later
            # `control` entry for the SAME name does not overwrite it. Without that, a
            # planted fresh-version `mcp` and a control `mcp` collide and the control's
            # versionless facts win — so the planted attack would be measured against facts
            # its own fixture never claimed, and read as allowed.
            if name in facts:
                continue
            facts[name] = _facts_from_blob(name, blob)
    return MappingRegistry(facts, name="fixtures")


def _facts_from_blob(name: str, blob: dict[str, Any]) -> RegistryFacts:
    # The version map has to be readable here or the `fresh-version` shape cannot be
    # expressed at all. `versions:` means exactly what the live parser's map means: a key
    # with a date is a dated release, and NO KEY (absent, or written `null`) is a release this
    # snapshot could not date — which is never read as "old". A fixture and a live payload
    # therefore cannot mean different things about the same shape.
    versions: dict[str, dt.datetime] = {}
    raw_versions = blob.get("versions")
    if isinstance(raw_versions, dict):
        for version, stamp in raw_versions.items():
            moment = _as_utc(stamp)
            if moment is not None:
                versions[str(version)] = moment
    return RegistryFacts(
        name=name, exists=bool(blob.get("exists")),
        first_release=_as_utc(blob.get("first_release")),
        release_count=blob.get("release_count"),
        source="fixture", observed_at=_as_utc(blob.get("observed_at")),
        version_upload_times=versions,
        latest_version=str(blob.get("latest_version") or ""))


#: The control set for the version hold (#3206): the tree's OWN frozen pins, and how far
#: before the eval clock a pin whose date cannot be had is placed — longer than the hold plus
#: the cache bound, so an undated pin is never held by accident.
LOCK_PINS_FILE = "requirements.lock"
_LOCK_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^;#\s]+)")
#: Cap on the commits :func:`lock_pin_first_seen` walks. This file has been amended a handful
#: of times since it was frozen, so the cap never binds here; it exists so a lock with a long
#: history cannot make the eval's cost proportional to it.
_LOCK_HISTORY_MAX_COMMITS = 64


def lock_pins(root: Path | str | None = None) -> list[tuple[str, str]]:
    """(name, version) for every exact `==` pin in :data:`LOCK_PINS_FILE`, in file order.

    The lock and not `requirements.txt`, and the reason is load-bearing: the txt declares a
    RANGE (`requirements.txt:16` is `mcp>=2.1.0,<3`), and a range is a claim about the
    registry's LATEST release — pricing a range against the hold would refuse the tree's own
    documented intent every time any upstream ships, which is #688's reflexive-override
    failure at the scale of the whole dependency set. A `==` line names exactly one artifact,
    the only claim the hold can be tested against. Option lines, comments and `name @ url`
    references are excluded, so `n` is the count of artifacts a frozen rebuild downloads at a
    pinned version (180 on this tree today), not a chosen subset.
    """
    try:
        text = (_repo_root(root) / LOCK_PINS_FILE).read_text(encoding="utf-8",
                                                             errors="replace")
    except OSError:
        return []
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = _LOCK_PIN_RE.match(line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def lock_pin_first_seen(root: Path | str | None = None, *,
                        clock: dt.datetime) -> dict[tuple[str, str], dt.datetime]:
    """When each `name==version` line in the lock was FIRST committed — a real age floor.

    Offline, no registry may be asked when `aiofiles==25.1.0` was published, so the control
    arm needs a date to price each pin against. Git history supplies one that needs no
    registry at all and needs no assumption: a pin cannot have been published after the
    commit that recorded it, so the commit that first introduced that exact line is an UPPER
    bound on the release date and therefore a LOWER bound on its age. Pricing the pin AT that
    bound is the worst case for the hold — the youngest the artifact could possibly be — so
    if the hold clears there it clears for every date the release could really have.

    Per pin, not per file: the lock has been amended several times since it was frozen, so a
    pin added by the newest commit is genuinely younger than one present in the first, and a
    single file-wide date would price all 180 at the newest one. A pin whose floor is inside
    the hold window is therefore reported as unbounded rather than passed, which is the
    honest answer and is what keeps this arm from claiming a proof it does not have.

    History is walked oldest-first and each line kept at its FIRST sighting, over every commit
    that touched the file, reading each blob once. The command is `git log --follow`, not
    `git rev-list --follow`, and the difference is the reason this walk is written the way it
    is: `--follow` is not a rev-list option, and git refuses it with exit 129 and usage on
    STDERR — measured on this tree, `stdout` was 0 bytes. A walk that reads only stdout and
    never checks the exit code therefore gets a clean EMPTY map, which is how an earlier
    revision of this function reported `n=0`: the arm printed "0 of 0 pins held", the eval
    published it, and nothing was measured. Exit codes are now irrelevant because stdout is
    parsed line by line and a malformed line is dropped, but the empty result is still visible
    as `n=0` rather than as a passed control. `git log` also carries the commit date in the same
    invocation (`%ct`) instead of costing one subprocess per commit. Nine commits touch this
    file on this tree; anything longer is capped, and no git, a failed read or a lock with no
    history yields an EMPTY map, which the arm reports as zero proven pins.
    """
    repo = _repo_root(root)
    try:
        out = subprocess.run(["git", "-C", str(repo), "log", "--reverse", "--follow",
                              "--format=%H %ct", "--", LOCK_PINS_FILE],
                             capture_output=True, text=True, timeout=20.0)
        revs: list[tuple[str, dt.datetime]] = []
        for line in (out.stdout or "").splitlines():
            head, _, epoch = line.strip().partition(" ")
            if len(head) != 40 or not epoch.strip().isdigit():
                continue
            stamp = dt.datetime.fromtimestamp(int(epoch.strip()), dt.timezone.utc)
            if stamp <= clock:
                revs.append((head, stamp))
    except Exception:  # noqa: BLE001 — no git, no timeout, no history: empty map covers it
        return {}
    first: dict[tuple[str, str], dt.datetime] = {}
    for rev, stamp in revs[:_LOCK_HISTORY_MAX_COMMITS]:
        text = _blob_text(repo, rev, LOCK_PINS_FILE)
        if text is None:
            continue
        for line in text.splitlines():
            m = _LOCK_PIN_RE.match(line.strip())
            if m:
                first.setdefault((m.group(1), m.group(2)), stamp)
    return first


_LOCK_HISTORY_MAX_COMMITS = 64


def _blob_text(repo: Path, rev: str, file: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{file}"],
                             capture_output=True, text=True, timeout=10.0)
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def lock_pin_hold_report(facts_for, *, root: Path | str | None = None,
                         thresholds: Thresholds = DEFAULT_THRESHOLDS,
                         now: dt.datetime | None = None,
                         pins: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Run the version hold across the lock's own pins and report how many of them bite.

    Clause 5's second n: *0 of the repo's own `==` pins held*, over every pin in the file —
    the arm that stops the hold from being a control which blocks the box's own rebuild.
    `facts_for` is injected for the reason the registry is injected everywhere in this file:
    the eval may not reach pypi.org, and neither may the question "when was `aiofiles==25.1.0`
    published".

    What makes the 0 mean something instead of being a tautology is the date it prices each
    pin at: the commit that FIRST introduced that exact line (`lock_pin_first_seen`). A release
    cannot postdate the commit that records it, so pricing a pin AT that instant is the worst
    case available without a registry — the youngest the artifact could possibly be — and a
    hold that clears there clears for the release's real date too. A pin whose floor is inside
    the hold window cannot be shown old or young from history, so it goes to `unbounded` and is
    not priced: claiming it cleared would be the same unsupported pass this whole module refuses
    elsewhere, and holding it would red the tree on a fact nobody measured.

    So the arm goes red only on a pin that IS provably older than the hold and was still
    refused — a real bug in the hold, not a property of the tree. The limitation travels with
    the numbers: `n` is the pins priced, `unbounded` the pins history could not place, and
    `youngest_floor` the tightest margin the arm actually relied on.
    """
    clock = now or dt.datetime.now(dt.timezone.utc)
    floors = lock_pin_first_seen(root, clock=clock)
    wanted = pins if pins is not None else lock_pins(root)
    held: list[dict[str, Any]] = []
    checked = 0
    unbounded = 0
    used: list[dt.datetime] = []
    for name, version in wanted:
        floor = floors.get((name, version))
        if floor is None or (clock - floor).days < thresholds.min_release_age_days:
            unbounded += 1
            continue
        checked += 1
        used.append(floor)
        facts = facts_for(name)
        times = dict(facts.version_upload_times)
        # A provider that dates the pin (a future live arm) wins; offline the history floor is
        # the only date available and it is applied as the fact it is — the youngest the
        # artifact can be. Leaving the entry out instead would answer `version-age-unknown` and
        # the arm would prove only that an unknown fails open, not that an old pin clears.
        times.setdefault(version, floor)
        facts = replace(facts, latest_version=version, version_upload_times=times)
        verdict = evaluate_provenance(facts, now=clock, thresholds=thresholds,
                                      version_spec=f"=={version}")
        if verdict.blocked:
            held.append({"name": name, "version": version, "fact": verdict.fact})
    youngest = max(used) if used else None
    return {"shape": "lock-pin", "n": checked, "held": len(held), "ok": not held,
            "pins": len(wanted), "unbounded": unbounded, "file": LOCK_PINS_FILE,
            "youngest_floor": youngest.isoformat() if youngest else None,
            "holding": held,
            "note": (f"{checked} of {len(wanted)} pins priced at the commit that first "
                     f"introduced them — the youngest each could possibly be, every one at "
                     f"least {thresholds.min_release_age_days} days old (tightest: "
                     f"{(clock - youngest).days if youngest else 'n/a'} days); "
                     f"{unbounded} pin(s) history could not place outside the hold window are "
                     "counted separately and not claimed either way")}


def _install_operand(command: str) -> str | None:
    """The first install operand of a command line: `pkg==1.2` from `pip install pkg==1.2`.

    Textual on purpose rather than via `find_install_commands`: an eval fixture is one install
    by construction, and routing through the full matcher would report a fixture the matcher
    refuses to parse as `held: false` — an ALLOW — instead of failing loudly.
    """
    for segment in re.split(r"&&|\|\n|;", str(command or "")):
        toks = segment.split()
        for i, tok in enumerate(toks):
            if tok in ("install", "add") and i + 1 < len(toks) \
                    and not toks[i + 1].startswith("-"):
                return toks[i + 1]
    return None


def evaluate_version_hold(name: str, *, command: str, facts: RegistryFacts,
                          thresholds: Thresholds = DEFAULT_THRESHOLDS,
                          now: dt.datetime | None = None) -> bool:
    """Whether the version hold bites THIS command — the eval's held/allowed oracle.

    The version is read out of the command by the same exact-pin rule the dispatch path uses,
    so a fixture cannot assert a version the checker would not have seen: `version:` in the
    document and the `==` in `command:` must agree or this raises. A command that pins nothing
    is priced against the registry's own latest release, which is the claim an unpinned
    install actually makes (step 3) and the shape that silently takes a day-old one.
    """
    operand = _install_operand(command)
    if operand is None:
        raise ValueError(f"{name}: fixture command {command!r} names no install operand")
    pinned = exact_pin(_version_spec("pip", operand))
    if pinned and facts.latest_version and pinned != facts.latest_version:
        raise ValueError(
            f"{name}: the fixture pins {pinned!r} in its command but names "
            f"{facts.latest_version!r} as the version under test; the hold can only be "
            "measured on the version the command asks for")
    spec = f"=={pinned}" if pinned else (
        f"=={facts.latest_version}" if facts.latest_version else None)
    return evaluate_provenance(facts, now=now, thresholds=thresholds,
                               version_spec=spec).blocked


def entry_is_version_shaped(entry: dict[str, Any]) -> bool:
    """Whether a fixture entry's claim is about a VERSION rather than about a name (#3206).

    One predicate for the two readers, because the eval's held/allowed arm and
    :func:`measure_fixtures`' authored `expected_blocked:` must not disagree about which policy
    decides an entry. Measured on the name facts alone, a `fresh-version` fixture comes back
    `expected_blocked: false` — its first release is years old and it has dozens of releases —
    while the eval holds it, so `fixtures --measure` would write a line into the committed
    document describing a check that is not the one running.
    """
    return (entry.get("version") is not None
            or bool((entry.get("facts") or {}).get("versions")))


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

    def verdict(entry: dict[str, Any]) -> bool:
        """Whether THIS entry is held: the shipped name check, or the version hold when the
        entry carries a version (#3206).

        Two arms deliberately. An entry with no version is the shipped name-shaped case and is
        decided by calling the shipped entry point, exactly as #688 decided it — re-deriving
        that verdict here would let the eval drift from the path it exists to measure. An
        entry that names a version is priced by `evaluate_version_hold`, which reads the
        version out of the entry's OWN command line rather than being told it.
        """
        name = str(entry.get("name") or "")
        command = str(entry.get("command") or f"pip install {name}")
        if entry_is_version_shaped(entry):
            return evaluate_version_hold(name, command=command, facts=reg.lookup(name),
                                         thresholds=DEFAULT_THRESHOLDS, now=clock)
        result = check_install_provenance(command, session, registry=reg,
                                          dependency_set={}, now=clock)
        return bool(result.refusal)

    planted = list(doc.get("planted") or [])
    control = list(doc.get("control") or [])
    missed = [str(e.get("name")) for e in planted if not verdict(e)]
    false_blocked = [str(e.get("name")) for e in control if verdict(e)]
    blocked = len(planted) - len(missed)
    declared = read_dependency_set(root)
    # Clause 5's second arm: the tree's own frozen pins, none of which may be held. Each pin
    # is priced at the commit that first recorded it — the youngest the artifact could
    # possibly be, from the tree's own history rather than a registry — so a hold that clears
    # there clears on the release's real date. That is what makes the printed 0 a measurement
    # rather than an absence, and `unbounded` is what the arm refuses to claim.
    def _pin_facts(name: str) -> RegistryFacts:
        # Name facts only, and dated well outside the hold: an empty version map would answer
        # `version-age-unknown` and the arm would prove only that an unknown fails open, not
        # that an old pin clears. The per-pin release date comes from the arm itself (the
        # first-commit floor), and the clock here is the real one for the same reason — see
        # `lock_pin_hold_report`.
        real = dt.datetime.now(dt.timezone.utc)
        return RegistryFacts(name=name, exists=True,
                             first_release=real - dt.timedelta(days=2_000),
                             release_count=40, source="fixture", observed_at=real)
    # No `now=` on purpose. This arm measures the LIVE tree — its pins, its git history, and
    # the instant a rebuild would actually run — and the floors it prices against are real
    # commit dates, some of them later than the fixture clock this eval's planted arms use.
    # Sharing that fictional 2026-09-24 clock would date every floor in its future and leave
    # the arm nothing to prove, which is the incoherence an earlier revision patched with a
    # special case for "commit date is after the clock".
    pin_arm = lock_pin_hold_report(_pin_facts, root=root)
    shapes: dict[str, int] = {}
    for entry in planted:
        shapes[str(entry.get("shape") or "unlabelled")] = (
            shapes.get(str(entry.get("shape") or "unlabelled"), 0) + 1)
    # The `fresh-version` arm has its own held/missed counts rather than being read back out
    # of the squat total. #3206 clause 5 is a rate about THIS shape: with 63 planted names and
    # only two of them fresh-version, a total block rate can stay at 100 % while this arm goes
    # to zero, and a shape whose rate nobody prints is a shape nobody notices going quiet.
    fresh_planted = [str(e.get("name")) for e in planted
                     if e.get("shape") == FRESH_VERSION_SHAPE]
    fresh_missed = [n for n in fresh_planted if n in set(missed)]
    return {
        "planted_total": len(planted), "planted_blocked": blocked, "planted_missed": missed,
        "control_total": len(control), "control_blocked": len(false_blocked),
        "control_false_blocks": false_blocked,
        "block_rate": (blocked / len(planted)) if planted else None,
        "false_block_rate": (len(false_blocked) / len(control)) if control else None,
        "planted_shapes": shapes,
        "lock_pins": pin_arm,
        "fresh_planted": fresh_planted,
        "fresh_missed": fresh_missed,
        "fresh_held": len(fresh_planted) - len(fresh_missed),
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
        if bool(str(expect).strip().lower() == "block") != bool(verdict(entry)):
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
    fresh_n = len(summary.get("fresh_planted") or [])
    pins = summary.get("lock_pins") or {}
    lines = [
        (f"planted squat-shaped names: {summary['planted_blocked']}/"
         f"{summary['planted_total']} blocked — block rate {pct(summary['block_rate'])}"),
        (f"repo's own dependencies (control): {summary['control_blocked']}/"
         f"{summary['control_total']} blocked — false-block rate "
         f"{pct(summary['false_block_rate'])}"),
        "planted shapes: " + ", ".join(f"{k} {v}" for k, v in sorted(shapes.items())),
        # #3206 clause 5's two n's, each on its own line. The squat total above cannot carry
        # either: two fresh-version names inside a set of 63 planted names is a rate that can
        # read 100 % while this arm is zero, and a control arm over 180 frozen pins is not
        # the same claim as a false-block rate over the 61 names in `control:`.
        (f"fresh-version installs: {summary.get('fresh_held', 0)}/{fresh_n} planted "
         f"well-established names at a held-eligible version HELD"
         + (f" — NOT HELD: {', '.join(summary['fresh_missed'])}"
            if summary.get("fresh_missed") else "")),
        (f"{pins.get('file', 'requirements.lock')} pins (control): "
         f"{pins.get('held', '?')}/{pins.get('n', '?')} of the tree's own frozen `==` pins "
         f"HELD"
         + (f" — HELD: {', '.join(h['name'] + '==' + h['version'] for h in pins['holding'][:8])}"
            if pins.get("holding") else "")
         + (f" (each priced at the commit that first recorded it; youngest "
            f"{str(pins['youngest_floor'])[:10]})" if pins.get("youngest_floor") else "")
         + (f"; {pins['unbounded']} not datable by history"
            if pins.get("unbounded") else "")),
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
        # The same two arms the eval runs, chosen by the same predicate: an entry whose claim
        # is about a version is decided by the version hold, and one whose claim is about a
        # name by the shipped name policy. Asking only the name facts here would stamp
        # `expected_blocked: false` onto a `fresh-version` fixture the eval holds, and the
        # committed document would then carry a number about a check that is not the one
        # running — the exact drift `expectations_mismatch` exists to catch in `expect:`.
        entry_facts = _facts_from_blob(str(entry.get("name")), entry.get("facts") or {})
        entry["expected_blocked"] = (
            evaluate_version_hold(
                str(entry.get("name")),
                command=str(entry.get("command") or f"pip install {entry.get('name')}"),
                facts=entry_facts, now=clock)
            if entry_is_version_shaped(entry)
            else bool(evaluate_provenance(entry_facts, now=clock).blocked))
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
        line = (f"{report.scanner_name} {report.scanner_version or '(version unknown)'}: "
                f"{report.verdict_label} "
                f"({report.advisory_count} advisory(ies) in {report.duration_seconds}s, "
                f"offline={report.offline})")
        for reason in report.to_dict()["stale"]:
            line += f"\n  stale: {reason}"
        return line
    return f"{report.scan_status}: {report.reason}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
