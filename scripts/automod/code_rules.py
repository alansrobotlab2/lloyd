"""Deterministic code-shape rules over a self-mod round's diff (#2574).

The gate's other static rungs are a syntax/import smoke test, a pyflakes delta
and a pyright delta: none of them can say "this *shape* must not appear in a
diff" about anything project-specific, so every recorded code-shape defect was
defended only by somebody remembering the class. This module is the surface
where the class is written down: `code_rules.toml` holds the rules, each one
citing the incident or doc claim that produced it, and the loader refuses an
uncited rule outright.

Seeded from the Qodo workshop's ladder (item #2574), whose interesting member is
Semgrep because it is the only rung whose rules the *repository* writes. Semgrep
is not installed here and `~/lloyd/.venvs/` is protected at the tool layer, so
step 3's decision is a stdlib `ast` matcher run by the interpreter the gate
already has. What that gives up, stated as the item asks: **cross-file flow and
taint**. A matcher here sees one module's syntax tree at a time, so it can name
a shape at a site but cannot know that the value at that site arrived from an
untrusted source three files away, cannot follow a value through a helper, and
cannot learn that a function's return is a dict. Anything needing dataflow is
out of this file by construction, not by taste — which is also why the epoch-vs-
ISO date-filter class did not qualify (see the rule file's header).

Two boundaries the runner is built to honour, both copied from
`scripts/automod/vet.py`:

*   A pass that could not read is **`unevaluated` with a reason**, and its
    result carries no `hits` key at all — an empty hit list is exactly what a
    clean round looks like, and #679's whole finding was that a green nobody can
    interrogate is worse than no green.
*   Suppressions are **counted, never honoured**. `# nosemgrep` is how a rule
    rung gets defanged one commit at a time, so a marker on an offending line
    changes only `totals.suppression_markers`.

The rung is observe-only (the item's own split, and #679's): a hit records and
never fails a round. Flipping a rule to blocking is a human ruling after the
soak, and `test_automod_gate.py` pins that nothing in this path can fail one.
"""
from __future__ import annotations

import ast
import fnmatch
import posixpath
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias, TypeGuard

RULES_PATH = Path(__file__).with_name("code_rules.toml")

#: The item's ceiling on the initial rule set. Enforced by the loader rather
#: than by review, because a hand-maintained rule corpus drifts toward the
#: general anti-pattern catalogue #662 was closed for rejecting.
MAX_RULES = 8

#: A citation has to name something a reader can go and read: a backlog item,
#: a self-mod round, a doc in this repo, or a dated class rule in the corpus.
CITATION_RE = re.compile(r"#\d{2,}|SM_\d{8}_\d+|CLAUDE\.md|architecture/|memory/learnings/")

#: Suppressions this runner *reports* and deliberately does not honour.
SUPPRESSION_RE = re.compile(
    r"#\s*(noqa\b|nosemgrep\b|type:\s*ignore|code-rules:\s*ignore)")

EVALUATED = "evaluated"
UNEVALUATED = "unevaluated"

#: The label list on a gate record. Same reason as `vet`'s: a round that trips a
#: hundred sites is one line of a gate report, not a hundred.
LABEL_CAP = 40

#: The interpreter the guardian unit actually runs on. `CLAUDE.md` says "stdlib-only on
#: /usr/bin/python3", and that path is the rule's whole subject:
#: `sys.stdlib_module_names` is a property of an interpreter, not of Python, so "is this
#: module stdlib" has two answers on this box. The gate runs on the venv (3.12, 301
#: names); the unit runs on the path below (3.14, 297 names). The 3.12-only half — `uu`,
#: `telnetlib`, `crypt`, `cgi`, `aifc`, `imghdr`, `pipes`, `sndhdr`, `sunau`, `xdrlib` and
#: 13 more — is precisely the list of imports a rule consulting the *gate's* interpreter
#: waves through and the watchdog then dies on at boot, on the machine that is already
#: down. The other direction is equally wrong: 18 names are 3.14-only, so the gate's set
#: would flag a stdlib import as a violation. Both counts are read from the interpreters,
#: never typed in — `python -c "import sys;print(len(sys.stdlib_module_names))"` at each.
GUARDIAN_PYTHON = "/usr/bin/python3"

#: One probe per interpreter per process. The whole-tree control scans 1327 files and the
#: suite parametrizes over every rule, so an uncached probe would shell out thousands of
#: times for one answer; keyed by interpreter, which is what lets both universes be held.
_STDLIB_CACHE: dict[str, frozenset[str]] = {}


class StdlibProbeError(RuntimeError):
    """The guardian interpreter's stdlib set could not be read."""


def stdlib_names(interpreter: str = GUARDIAN_PYTHON) -> frozenset[str]:
    """`sys.stdlib_module_names` as `interpreter` sees it.

    Raises rather than falling back, and the fallback that would be *tempting* — trust the
    running interpreter's set — is the specific bug this function exists to refuse: it is
    one of the two wrong answers above, and it is wrong silently, in the direction that
    reads as clean. `run` turns this into `unevaluated`, which is the runner's contract
    for any read it cannot do.
    """
    hit = _STDLIB_CACHE.get(interpreter)
    if hit is not None:
        return hit
    code = "import sys; print('\\n'.join(sorted(sys.stdlib_module_names)))"
    try:
        proc = subprocess.run([interpreter, "-c", code],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StdlibProbeError(
            f"cannot ask {interpreter} for its stdlib set: {exc}") from exc
    if proc.returncode != 0:
        raise StdlibProbeError(
            f"{interpreter} exited {proc.returncode} answering for its stdlib set: "
            f"{proc.stderr.strip().splitlines()[:1] or ['']}")
    names = frozenset(t for t in proc.stdout.split() if t)
    if not names:
        raise StdlibProbeError(f"{interpreter} answered an empty stdlib set")
    _STDLIB_CACHE[interpreter] = names
    return names

_LOCAL_ROOTS = frozenset({"scripts", "agent_mcp", "app", "workers"})


class CodeRulesError(Exception):
    """The rule file is wrong. A rule set that cannot be read must not run as none."""


@dataclass(frozen=True)
class Allow:
    """One named exception, with the reason it is an exception."""
    path: str
    why: str


@dataclass(frozen=True)
class Rule:
    id: str
    matcher: str
    summary: str
    cite: str
    scope: tuple[str, ...]
    exclude: tuple[str, ...]
    allow: tuple[Allow, ...] = field(default_factory=tuple)

    def covers(self, rel: str) -> bool:
        if any(_match(rel, pat) for pat in self.exclude):
            return False
        return any(_match(rel, pat) for pat in self.scope)

    def allows(self, rel: str) -> Allow | None:
        for a in self.allow:
            if rel == a.path or rel.startswith(a.path.rstrip("/") + "/"):
                return a
        return None


def _match(path: str, pattern: str) -> bool:
    """fnmatch over repo-relative posix paths, with `**/` able to mean "or root".

    `fnmatch` has no real `**`: `*` already spans `/`. So `**/*.py` needs one
    slash to match, which would silently exclude a file at the repo root — the
    one file (`server.py`) most worth reading. A `**/` prefix therefore also
    tries the bare remainder.
    """
    if fnmatch.fnmatch(path, pattern):
        return True
    if pattern.startswith("**/"):
        return fnmatch.fnmatch(path, pattern[3:])
    return False


def _one_line(text: str) -> str:
    return " ".join((text or "").split())


def load_rules(path: Path | None = None) -> list[Rule]:
    """Read and validate the rule file. Every failure here is a `CodeRulesError`.

    Refused at load, never warned about: a rule with no citation (an uncited rule
    is deleted, not tuned), an unknown matcher name (a typo'd matcher is a rule
    that silently never fires), an allow entry with no reason (how an allowlist
    becomes a default), a duplicate id, and more than `MAX_RULES` rules.
    """
    p = Path(path) if path is not None else RULES_PATH
    try:
        raw = tomllib.loads(p.read_bytes().decode("utf-8"))
    except OSError as exc:
        raise CodeRulesError(f"cannot read the rule file {p}: {exc.strerror or exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CodeRulesError(f"rule file {p} is not valid TOML: {exc}") from exc

    entries = raw.get("rule")
    if not isinstance(entries, list) or not entries:
        raise CodeRulesError(f"rule file {p} declares no [[rule]] entries")
    if len(entries) > MAX_RULES:
        raise CodeRulesError(
            f"rule file {p} declares {len(entries)} rules; the cap is {MAX_RULES}")

    known = set(_MATCHERS)
    out: list[Rule] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise CodeRulesError(f"rule #{i + 1} is not a table")
        rid = str(e.get("id") or "").strip()
        if not rid:
            raise CodeRulesError(f"rule #{i + 1} has an empty id")
        if rid in seen:
            raise CodeRulesError(f"duplicate rule id {rid!r}")
        seen.add(rid)
        matcher = str(e.get("matcher") or "").strip()
        if matcher not in known:
            raise CodeRulesError(
                f"rule {rid!r} names matcher {matcher!r}; known matchers: "
                f"{', '.join(sorted(known))}")
        cite = _one_line(str(e.get("cite") or ""))
        if not cite:
            raise CodeRulesError(
                f"rule {rid!r} has an empty cite: a rule that names no incident "
                "or doc claim cannot be added (#2574 clause 1)")
        if not CITATION_RE.search(cite):
            raise CodeRulesError(
                f"rule {rid!r} cites {cite!r}, which names no backlog item (#NNNN), "
                "self-mod round (SM_…), in-repo doc (CLAUDE.md / architecture/) or "
                "dated class rule (memory/learnings/)")
        allow: list[Allow] = []
        for a in (e.get("allow") or []):
            if not isinstance(a, dict):
                raise CodeRulesError(
                    f"rule {rid!r}: an allow entry must be a table with path and why")
            apath = str(a.get("path") or "").strip()
            why = _one_line(str(a.get("why") or ""))
            if not apath:
                raise CodeRulesError(f"rule {rid!r}: an allow entry has no path")
            if not why:
                raise CodeRulesError(
                    f"rule {rid!r}: allow entry {apath!r} has no why")
            allow.append(Allow(apath, why))
        scope = tuple(str(s) for s in (e.get("scope") or []))
        if not scope:
            raise CodeRulesError(f"rule {rid!r} has an empty scope")
        out.append(Rule(
            id=rid, matcher=matcher, cite=cite,
            summary=_one_line(str(e.get("summary") or "")),
            scope=scope, exclude=tuple(str(s) for s in (e.get("exclude") or [])),
            allow=tuple(allow)))
    return out


# ---------------------------------------------------------------------------
# The matchers. Each takes one parsed module and returns (line, detail) pairs.
# One module's tree at a time: no cross-file flow, no taint (see the docstring).
# ---------------------------------------------------------------------------

def _iter_imports(tree: ast.AST):
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield a.name, n.lineno
        elif isinstance(n, ast.ImportFrom):
            if n.level:                       # relative import: stays in the dir
                continue
            yield (n.module or ""), n.lineno


def guardian_nonstdlib_import(ctx: "_FileCtx") -> list[tuple[int, str]]:
    """The guardian is staged as a flat copy of its own directory and runs on
    `/usr/bin/python3`, so anything it imports must be stdlib or sit beside it.

    Asks the unit's interpreter which modules are stdlib (`stdlib_names`), and raises
    `StdlibProbeError` — which `run` turns into `unevaluated` — when that cannot be read.
    Consulting the gate's own interpreter was the previous version of this matcher, and it
    is a silently-wrong answer in the direction that reads as clean.
    """
    stdlib = stdlib_names(ctx.interpreter)
    out: list[tuple[int, str]] = []
    for mod, lineno in _iter_imports(ctx.module):
        top = mod.split(".")[0]
        if not top or top in stdlib or top in ctx.siblings:
            continue
        out.append((lineno,
                    f"imports {mod!r}, which is neither stdlib on "
                    f"{ctx.interpreter} nor a module beside it in "
                    f"{posixpath.dirname(ctx.rel)}"))
    return out


#: The two functions that hand back a knowledge-graph store path. `app/data_root.py`
#: owns both spellings, so a call to either is as much "the store" as the literal is.
_KG_PATH_FUNCS = frozenset({"kg_store_for_root", "kg_store_for_tree"})

#: The constants that name it. `KG_DB` is the guardian's own binding
#: (`agent-services/guardian/policy.py`: `KG_DB, KG_DB_SOURCE = kg_db_path()`), and
#: `KG_DB_SOURCE` is deliberately absent: it records *which branch* produced the path,
#: not the path, so exempting it would exempt a string that is never opened.
_KG_PATH_CONSTANTS = frozenset({"KG_DB", "KG_DB_RELATIVE", "VAULT_KG_DB"})


def _kg_path_source(node: ast.AST | None) -> str | None:
    """The spelling by which this expression denotes the knowledge-graph store, or None."""
    if isinstance(node, ast.Call):
        f = node.func
        name = (f.attr if isinstance(f, ast.Attribute)
                else f.id if isinstance(f, ast.Name) else "")
        return f"{name}(...)" if name in _KG_PATH_FUNCS else None
    if isinstance(node, (ast.Attribute, ast.Name)):
        ident = node.attr if isinstance(node, ast.Attribute) else node.id
        return ident if ident in _KG_PATH_CONSTANTS else None
    return None


def _kg_store_names(module: ast.Module) -> dict[str, str]:
    """Local names holding a knowledge-graph store path, each mapped to its spelling.

    Two binders, and each is exactly one hop inside this one module — the module
    boundary is still the limit this file's docstring owns, so nothing here is
    cross-file flow:

    *   `name = <expression naming the store>` — the promoter's shape,
        `db = kg_store_for_root(production_data_root())`;
    *   a parameter whose call site *in this same file* hands over such an
        expression — the guardian's shape, where `count_kg_rows(policy.KG_DB)` at
        `agent-services/guardian/guardian.py:1058` fills the `db_path` the open at
        `:2897` reads. A parameter is a binding like any other; skipping it is how
        the witness stayed invisible.
    """
    names: dict[str, str] = {}
    for n in ast.walk(module):
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            source = _kg_path_source(n.value)
            if source is None:
                continue
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    names.setdefault(t.id, source)

    params: dict[str, list[str]] = {}
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        params.setdefault(node.name, []).extend(
            [p.arg for p in (*a.posonlyargs, *a.args)] + [p.arg for p in a.kwonlyargs])
    for n in ast.walk(module):
        if not isinstance(n, ast.Call):
            continue
        callee = (n.func.attr if isinstance(n.func, ast.Attribute)
                  else n.func.id if isinstance(n.func, ast.Name) else "")
        bound = params.get(callee)
        if not bound:
            continue
        for i, arg in enumerate(n.args):
            if i < len(bound):
                source = _kg_path_source(arg)
                if source is not None:
                    names.setdefault(bound[i], source)
        for kw in n.keywords:
            if kw.arg in bound:
                source = _kg_path_source(kw.value)
                if source is not None:
                    names.setdefault(kw.arg, source)
    return names


def _kg_reference_in_call(call: ast.Call, names: dict[str, str]) -> tuple[str, str] | None:
    """Which store-naming value this `connect` reads: (name-as-written, its spelling)."""
    for arg in (*call.args, *(kw.value for kw in call.keywords)):
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Name) and sub.id in names:
                return sub.id, names[sub.id]
            source = _kg_path_source(sub)
            if source is not None and isinstance(sub, (ast.Name, ast.Attribute)):
                return (sub.id if isinstance(sub, ast.Name) else sub.attr), source
    return None


def opens_kg_store(ctx: "_FileCtx") -> list[tuple[int, str]]:
    """`sqlite3.connect(...)` on a path that names the knowledge-graph store.

    Two arms, and both stop at the module boundary. The literal one is the narrow
    form of the store-ownership claim: the broad one (`kg.sqlite` appearing at all)
    is an open set — 20+ files at HEAD name the path for reasons that are not
    opening it. The resolved one follows the path exactly one hop, through the two
    binders `_kg_store_names` can see, because a store path handed to `connect`
    through a name is the same open as one typed in: since #3357 the two live
    read-only witnesses (`agent-services/guardian/guardian.py:2897`,
    `scripts/automod/promote.py:502`) are both of that shape, so before this arm the
    rule could not report either of them and so could not require either to carry an
    exemption. What stays out is what always stayed out — a path that crossed a file
    or a function boundary to reach the name, which is cross-file flow and out of
    this file by construction, not by taste.
    """
    out: list[tuple[int, str]] = []
    names = _kg_store_names(ctx.module)
    for n in ast.walk(ctx.module):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "connect" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "sqlite3"):
            continue
        if "kg.sqlite" in ast.unparse(n):
            out.append((n.lineno, "opens the knowledge-graph store outside its owner"))
            continue
        reference = _kg_reference_in_call(n, names)
        if reference is not None:
            written, source = reference
            out.append((n.lineno,
                        f"opens the knowledge-graph store outside its owner through "
                        f"`{written}`, one hop from {source}"))
    return out


_BROAD_RE = re.compile(r"\b(BaseException|Exception)\b")


def broad_except_swallow_same_line(ctx: "_FileCtx") -> list[tuple[int, str]]:
    """`except <broad>: pass` (or bare `except: pass`) with the swallow on the
    same line — the form with no legitimate use left in this tree."""
    out: list[tuple[int, str]] = []
    for n in ast.walk(ctx.module):
        if not isinstance(n, ast.ExceptHandler):
            continue
        if len(n.body) != 1 or not isinstance(n.body[0], ast.Pass):
            continue
        if n.body[0].lineno != n.lineno:
            continue                          # the multi-line form is an open set
        text = "bare" if n.type is None else ast.unparse(n.type)
        if n.type is not None and not _BROAD_RE.search(text):
            continue                          # a narrow except that passes is not this
        out.append((n.lineno, f"swallows {text} with a same-line `pass`"))
    return out


_PATHISH = frozenset({"Path", "PurePath", "open"})

#: The node kinds `_is_path_expr` answers `True` for. Named so the guard can say what
#: the matcher actually guarantees: `ast.AST` on its own carries no `lineno`, so a
#: matcher that walks a tree and then reports `n.lineno` is reporting an attribute the
#: declared type does not have — pyright's `reportAttributeAccessIssue` on that line is
#: what a real `AttributeError` on a shape nobody planted looks like.
_PathExpr: TypeAlias = ast.BinOp | ast.Call


def _is_path_expr(n: ast.AST) -> TypeGuard[_PathExpr]:
    """An expression that *builds* a path or filename: `base / f"..."`,
    `Path(...)`, `os.path.join(...)`."""
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
        return True
    if isinstance(n, ast.Call):
        try:
            text = ast.unparse(n.func)
        except Exception:
            return False
        if text.rsplit(".", 1)[-1] in _PATHISH or text.endswith("path.join"):
            return True
    return False


def _is_naive_now(call: ast.Call) -> bool:
    """A no-arg `.now()` — `datetime.now()` / `dt.datetime.now()` / `_dt.now()`.

    No args is the whole predicate: a one-arg `datetime.now(timezone.utc)` is the
    fix, not the defect, and a 101-site grep of the bare call name (measured
    2026-10-11) is why the rule is scoped to where the value goes instead.
    """
    f = call.func
    return (isinstance(f, ast.Attribute) and f.attr == "now"
            and not call.args and not call.keywords)


def naive_now_in_artifact_name(ctx: "_FileCtx") -> list[tuple[int, str]]:
    """A local wall-clock stamp baked into a path or artifact name."""
    out: list[tuple[int, str]] = []
    for n in ast.walk(ctx.module):
        if not _is_path_expr(n):
            continue
        for m in ast.walk(n):
            if isinstance(m, ast.Call) and _is_naive_now(m):
                out.append((n.lineno,
                            "builds a path or artifact name from a no-arg "
                            "datetime.now(), which is local wall clock"))
                break
    return out


def _env_target(node: ast.AST) -> str | None:
    """`os.environ` / `os.environ` read through a name: the base of a subscript."""
    try:
        text = ast.unparse(node)
    except Exception:
        return None
    return text if text.endswith("environ") else None


def process_environ_lloyd_data(ctx: "_FileCtx") -> list[tuple[int, str]]:
    """A mutation of *this process's* environment that sets `LLOYD_DATA`.

    Passing `env={**os.environ, "LLOYD_DATA": round_scratch}` to one child is the
    sanctioned shape and is not what this matches — that dict never outlives the
    spawn, while an assignment here is inherited by every child of the process
    that runs it, which is how a worktree run ends up writing live data.
    """
    out: list[tuple[int, str]] = []
    for n in ast.walk(ctx.module):
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            # `getattr(n, "targets", ...)` worked and read as a clean answer, but it is a
            # union the checker cannot follow: `Assign` has `targets`, `AnnAssign` has a
            # single `target`, so the join has neither name on it and pyright refused the
            # attribute. Asking each kind for its own field is the same fact, stated where
            # the type system can see it — and it is what stops a future third assignment
            # node silently matching neither arm of a `getattr` default.
            targets: list[ast.expr] = (list(n.targets) if isinstance(n, ast.Assign)
                                       else [n.target])
            for t in targets:
                if (isinstance(t, ast.Subscript) and _env_target(t.value)
                        and "LLOYD_DATA" in ast.unparse(t.slice)):
                    out.append((n.lineno,
                                "writes LLOYD_DATA into this process's os.environ"))
        elif isinstance(n, ast.Call):
            try:
                text = ast.unparse(n)
                fn = ast.unparse(n.func)
            except Exception:
                continue
            if (fn.endswith(("environ.setdefault", "environ.update"))
                    and "LLOYD_DATA" in text):
                out.append((n.lineno,
                            "writes LLOYD_DATA into this process's os.environ"))
        elif isinstance(n, ast.Delete):
            for t in n.targets:
                if (isinstance(t, ast.Subscript) and _env_target(t.value)
                        and "LLOYD_DATA" in ast.unparse(t.slice)):
                    out.append((n.lineno,
                                "mutates LLOYD_DATA in this process's os.environ"))
    return out


_MATCHERS = {
    "guardian_nonstdlib_import": guardian_nonstdlib_import,
    "opens_kg_store": opens_kg_store,
    "broad_except_swallow_same_line": broad_except_swallow_same_line,
    "naive_now_in_artifact_name": naive_now_in_artifact_name,
    "process_environ_lloyd_data": process_environ_lloyd_data,
}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

@dataclass
class _FileCtx:
    root: Path
    rel: str
    src: str
    lines: list[str]
    tree_ast: ast.Module
    siblings: frozenset[str]
    #: Which interpreter's stdlib universe a rule may consult. The guardian rule asks the
    #: unit's interpreter and never this process's: see `stdlib_names`.
    interpreter: str = GUARDIAN_PYTHON

    @property
    def module(self) -> ast.Module:
        return self.tree_ast


class _Reader:
    """Counts what a clean record could be hiding: unread files and suppressions."""

    def __init__(self) -> None:
        self.read_failures: list[str] = []
        self.parse_errors: list[str] = []
        self.suppression_markers = 0
        self.scanned = 0
        self.added_lines = 0
        # Files each rule's own `scope` accepted, counted where the scope test is actually
        # applied. `scanned` is the union — a file covered by any rule is read once — so it
        # is not any single rule's figure, and the narrowest rule's is far below it. Without
        # this per-rule count, a rule whose pattern had stopped matching and a rule whose
        # real violations were swept away are the same number: zero.
        self.covered: dict[str, int] = {}
        # Which interpreters' stdlib universes a rule was allowed to consult. A clean
        # guardian record is only worth the interpreter named on it, so the answer travels
        # with the result instead of being assumed from whatever ran the scan.
        self.interpreters: set[str] = set()


def _git(tree: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(tree), *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`git {' '.join(args)}` exited {proc.returncode}: "
            f"{proc.stderr.strip().splitlines()[:1] or ['']}")
    return proc.stdout


def _added_lines(tree: Path, base: str, rel: str) -> set[int]:
    """1-based line numbers the round added or edited at HEAD, per `git diff -U0`.

    Diff-scoped, which is the item's stated choice: fair to the round's author
    and cheap, and the thing it gives up — a shape at a line this round never
    touched — is what a weekly whole-tree pass would be for. Whether that pass is
    needed is one of the rulings this round leaves owed.
    """
    out: set[int] = set()
    diff = _git(tree, "diff", "--no-color", "-U0", f"{base}..HEAD", "--", rel)
    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in diff.splitlines():
        m = hunk.match(line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        out.update(range(start, start + count))
    return out


def _siblings(tree: Path, rel: str) -> frozenset[str]:
    """Module names that sit beside `rel` in the tree — the guardian's staging rule."""
    d = tree / posixpath.dirname(rel)
    try:
        return frozenset(p.stem for p in d.glob("*.py"))
    except OSError:
        return frozenset()


def run(base_ref: str, tree: Path, *, files: list[str] | None = None,
        rules_path: Path | None = None, whole_tree: bool = False) -> dict:
    """Run every rule over the round's changed Python files, or over `files`.

    `base_ref` and `tree` are the two things a round knows about its own change
    set, same as `vet.py`'s entry point: the resolved base and the worktree to
    read. `files` replaces the diff — a whole-tree control passes `git ls-files
    '*.py'` with `whole_tree=True`, which is also what a weekly pass would do.

    Returns `{"status": "unevaluated", "reason": ...}` — and no `hits` key, so it
    cannot be misread as a zero-hit pass — for any failed git or file read.
    """
    out: dict = {"scope": {"kind": "whole_tree" if whole_tree else "diff",
                           "base": base_ref}}
    try:
        rules = load_rules(rules_path)
    except CodeRulesError as exc:
        out.update({"status": UNEVALUATED, "reason": str(exc)})
        return out

    reader = _Reader()
    if files is None:
        try:
            files = [f for f in _git(tree, "diff", "--name-only", "--diff-filter=ACMR",
                                     f"{base_ref}..HEAD", "--", "*.py").splitlines() if f]
        except RuntimeError as exc:
            out.update({"status": UNEVALUATED, "rules_loaded": len(rules),
                        "reason": f"could not enumerate the round's files: {exc}"})
            return out

    hits: list[dict] = []
    allowed: list[dict] = []
    # The census's denominator: every `.py` the caller offered, whether or not a scope
    # accepted it. `reader.scanned` cannot be that number — it is the union of the rules'
    # scopes, so quoting it as "files checked" would make a rule that sees a couple of
    # dozen files indistinguishable from one that saw them all.
    considered = sum(1 for f in files if f.endswith(".py"))
    for rel in files:
        if not rel.endswith(".py"):
            continue
        if not any(r.covers(rel) for r in rules):
            continue
        path = tree / rel
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            reader.read_failures.append(f"{rel}: {exc.strerror or exc}")
            continue
        reader.scanned += 1
        reader.suppression_markers += len(SUPPRESSION_RE.findall(src))
        src_lines = src.splitlines()
        if whole_tree:
            changed = None                      # every line is in scope
        else:
            try:
                changed = _added_lines(tree, base_ref, rel)
            except RuntimeError as exc:
                out.update({"status": UNEVALUATED, "rules_loaded": len(rules),
                            "reason": f"could not diff {rel}: {exc}"})
                return out
            reader.added_lines += len(changed)
        try:
            parsed = ast.parse(src)
        except SyntaxError:
            reader.parse_errors.append(rel)
            continue
        ctx = _FileCtx(root=tree, rel=rel, src=src, lines=src_lines, tree_ast=parsed,
                       siblings=_siblings(tree, rel))
        for rule in rules:
            if not rule.covers(rel):
                continue
            reader.covered[rule.id] = reader.covered.get(rule.id, 0) + 1
            try:
                sites = _MATCHERS[rule.matcher](ctx)
            except StdlibProbeError as exc:
                # A rule that cannot answer is not a rule that found nothing, and the
                # file that provoked it is named: an unevaluated record for the reason
                # that the *unit's* interpreter is missing off this machine is an
                # environment fact a human should see, not a clean round.
                out.update({"status": UNEVALUATED, "rules_loaded": len(rules),
                            "reason": f"rule {rule.id} could not run on {rel}: {exc}"})
                return out
            if rule.matcher == "guardian_nonstdlib_import":
                # What that verdict was measured against, on the record.
                reader.interpreters.add(ctx.interpreter)
            for lineno, detail in sorted(sites):
                if changed is not None and lineno not in changed:
                    continue                    # pre-existing at base: not this round's
                entry = {"rule": rule.id, "path": rel, "line": lineno, "detail": detail}
                entry["message"] = f"{rule.id} at {rel}:{lineno}: {detail}"
                # Asked once, into a name. `if rule.allows(rel) is not None` followed by
                # `rule.allows(rel).path` is the same call twice, and the second one is
                # typed `Allow | None` however sure the guard is — so the checker saw a
                # `.path` on `None` here, which is the failure mode the runner's whole
                # `unevaluated` contract exists to refuse.
                allowance = rule.allows(rel)
                if allowance is not None:
                    entry["allow_path"] = allowance.path
                    allowed.append(entry)
                else:
                    hits.append(entry)

    if reader.read_failures:
        out.update({"status": UNEVALUATED, "rules_loaded": len(rules),
                    "reason": "could not read " + "; ".join(reader.read_failures[:6])})
        return out

    counts: dict[str, int] = {}
    for h in hits:
        counts[h["rule"]] = counts.get(h["rule"], 0) + 1
    labels = [f"{h['rule']}:{h['path']}" for h in hits[:LABEL_CAP]]
    out.update({
        "status": EVALUATED,
        "rules_loaded": len(rules),
        "hits": hits,
        "allowed": allowed,
        "counts": counts,
        "labels": labels,
        "totals": {
            "files": reader.scanned,
            "added_lines": reader.added_lines,
            "suppression_markers": reader.suppression_markers,
            "parse_errors": len(reader.parse_errors),
            # Empty means the guardian rule was never asked — no file in its scope was
            # scanned — which is not the same fact as its answering clean.
            "interpreters": sorted(reader.interpreters),
            # The census clause 4 asks for, per rule: files its own scope accepted over
            # files the caller offered. It is on the record rather than only in the
            # control's arithmetic because the control recomputes it with the same
            # `covers()` the runner uses — an agreement between them proves nothing about a
            # scope that silently narrowed. Carried on the durable event, it is also what
            # makes the soak's denominator auditable months later: a rule that covered
            # nothing for twenty landings has a zero nobody should have believed.
            "coverage": {r.id: {"covered": reader.covered.get(r.id, 0),
                                "of": considered} for r in rules},
        },
    })
    if len(hits) > LABEL_CAP:
        out["labels_truncated"] = len(hits) - LABEL_CAP
    return out


def describe(result: dict) -> str:
    """One line for the gate report. An unevaluated run must not read as clean."""
    if result.get("status") != EVALUATED:
        return (f"UNEVALUATED: {result.get('reason') or 'no reason given'} "
                "(not a clean round)")
    total = sum(result.get("counts", {}).values())
    allowed = len(result.get("allowed", []))
    scope = result["scope"]["kind"]
    head = (f"{total} hit(s) over {result['totals']['files']} changed .py file(s), "
            f"scope={scope}") if scope else ""
    bits = [head or f"{total} hit(s), scope={scope}"]
    if allowed:
        bits.append(f"{allowed} in the rule file's allowlist")
    markers = result["totals"]["suppression_markers"]
    if markers:
        bits.append(f"{markers} suppression marker(s) counted, none honoured")
    return "; ".join(bits)
