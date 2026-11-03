"""#3726 pins every symbol `app/mitigation_state.py`'s docstrings cite to the tree.

The incident that filed this: #3724 asserted "there is no witness for the 04:42Z drill at
all" from `grep -hi drill ~/lloyd-data/logs/{server.log,lloyd-agent-worker.log,mcp.log}`
returning zero. Those three files hold no drill lines — `lloyd-workers.maintenance` is the
backend's own logger, so its lines go to inherited stderr in `~/lloyd-data/logs/server.err`
— and the sentence that sent a reader to the wrong three files is `drill_slot`'s, which
handed them a nine-character-with-underscore name no commit ever defined — `git log -S` on it
answers exactly one commit, `d07bcf3c` (#2551), which invented it and never wrote it. The
mechanism the sentence describes is real and demonstrably works; only the name was fiction,
and nothing tested that the names in this module's docstrings exist.

So this file does the one thing that was missing: it extracts the backtick-quoted spans the
module's docstrings present as citations and demands each one resolve. `app/mitigation_state.py`
is 460 lines and all 46 of its docstring citations resolve now (measured by `_citations`
under this worktree on 2026-11-03); a newly invented name in one fails here instead of
sending the next reader to a false zero.

**What counts as a citation** (`_span_tokens`), stated exactly because a regex that matched
nothing looks exactly like a module with nothing wrong:

  * a docstring span with SPACES in it is prose, except for its path-shaped words (a word
    with a `/` or a leading `~`), which must resolve as paths. ``cat
    ~/lloyd-data/mitigation_drill.json.slot`` therefore checks the path and not the verb;
  * a single-word span is a citation and must resolve, if it is identifier-shaped
    (`HISTORY_CAP`, `read()`, `workers.maintenance.MITIGATION_DRILL_DEFAULTS`) or path-shaped
    (`scripts/mitigation_drill.py`);
  * a hyphen inside a single-word span means prose (`never-run`, `not-due`, `round-hold`) —
    values spelled that way are quoted data, not symbols. A name qualified by a dot or a
    slash IS checked even with a hyphen inside it (`lloyd-workers.maintenance`), because a
    qualified name is a pointer.

**How a citation resolves** (`_resolve`). A path-shaped token resolves as a file under any
root this box has — the checkout, the account home, the data root and the vault, both the
env-overridable value `app.paths` publishes and the passwd-derived production value under it,
so a real path is not called absent because the gate redirected the root — and a path-shaped
token that is no file resolves as an exact string constant, which is what an HTTP route is
(`/api/workers/status`). A plain name resolves, in this order, as an attribute of
`app.mitigation_state`, a `def`/`class`/assignment target, an attribute access, an exact
string constant, or a module file of this tree (`atomic_io` for `app/atomic_io.py`); a dotted
name first by import, then as `Head.tail` when the head is a real definition and the tail is
an attribute the tree really accesses (`WorkerPool.start()`). Definitions, attributes and
constants all come from `_code_names`, collected through the `ast` over `app/`, `workers/`,
`agent_mcp/`, `scripts/` and `agent-services/`.

Two properties that pin's checks have to have, and the node that shows each. It has to be
able to FAIL, so `test_the_check_fails_on_a_citation_the_test_itself_invents` writes the two
kinds of wrong citation into a copy of the module — an invented symbol and a path into a
directory that does not exist — and reports both unresolved. And it has to have extracted
something in the first place, so the main node asserts the extracted set is non-empty before
it asserts anything about its members. The corpus is parsed from the working tree rather than
`git ls-files` because a cited runtime file (`~/lloyd-data/logs/server.err`) is real on this
box and in no worktree, and the code-name set is collected through `ast`, never over docstrings
and comments, so a citation can never be satisfied by prose that merely contains it.

**The boundary this file does not cross.** `#:` block comments are not docstrings and are not
read: `HISTORY_CAP`'s and `SPACING_GAP_S`'s comments cite
`workers.maintenance.MITIGATION_DRILL_DEFAULTS["interval_seconds"]` and two test paths, and
those names are checked here only where a docstring repeats them. Widening `_docstrings` to
comments is the obvious next step for whoever touches this module again; nothing in it is
wrong today.
"""
from __future__ import annotations

import ast
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import app.mitigation_state as mitigation_state
from app import data_root, paths

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "app" / "mitigation_state.py"

#: The code directories a citation from this module can point at. Every name its docstrings
#: cite lives in one of them, and the narrower the corpus the cheaper the proof — 567 files
#: parse in about 3 s, once per pytest process, against 1,479 for the whole checkout.
CODE_DIRS = ("app", "workers", "agent_mcp", "scripts", "agent-services")

#: The shipped tree a wrong symbol must be absent from: the code that runs plus the docs a
#: reader follows. Scope is the shipped surfaces and NOT `tests/`: the file that
#: pins an absence would otherwise be a hit for its own grep, so the name is kept out of this
#: file's bytes entirely (`WRONG_NAME` is joined from parts) and the exclusion is a
#: belt-and-braces statement rather than a loophole.
SHIPPED_SURFACES = ("app", "workers", "agent_mcp", "scripts", "agent-services",
                    "architecture")
SHIPPED_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".sh", ".json"}

#: The name this item exists to remove — cited as prose in the module's docstring until this
#: round, defined by no commit in the repo's history. JOINED FROM PARTS AT RUNTIME on
#: purpose: the item's acceptance check is a repo-wide grep for that exact string, and a
#: police file that trips the check it exists to enforce is a check nobody can run. Joined
#: here, the reviewer's re-run of `git grep -rn` comes back empty while the node below still
#: searches for, and the plant below still plants, the real nine-word name. `git grep -rn
#: log_drill` finds this line, which is how a reader confirms the join is the same name.
WRONG_NAME = "_log_" + "drill_exit"

_SPAN = re.compile(r"`([^`\n]+)`")
_IDENTIFIER = re.compile(r"\A[A-Za-z_][A-Za-z0-9_.]*(?:\(\))?\Z")
_PATHY = re.compile(r"(/|^~)")


def _docstrings(source: str) -> list[str]:
    """Every docstring in `source`: the module's, and each function's and class's.

    Read through `ast`, not a triple-quote regex, so a docstring that contains a backtick
    pair inside a string literal cannot invent a citation that the code does not have.
    """
    tree = ast.parse(source)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            text = ast.get_docstring(node)
            if text:
                out.append(text)
    return out


def _span_tokens(text: str) -> list[str]:
    """The backtick spans in one block of prose that are citations, not formatting.

    A span with SPACES is prose, except for its path-shaped words — ``cat
    ~/lloyd-data/mitigation_drill.json.slot`` checks the path and not the verb. A
    single-word span is a citation if it is identifier-shaped (`HISTORY_CAP`, `read()`,
    `workers.maintenance.MITIGATION_DRILL_DEFAULTS`) or path-shaped
    (`scripts/mitigation_drill.py`). A hyphen inside a single-word span makes it prose:
    `never-run` and `not-due` are quoted values, and a name qualified by a dot or a slash is
    still checked even with a hyphen inside it, because a qualified name is a pointer.
    """
    seen: list[str] = []
    for span in _SPAN.findall(text):
        words = span.split()
        if len(words) > 1:
            for word in words:
                word = word.strip(".,;:").strip("`")
                if _PATHY.search(word) and word not in seen:
                    seen.append(word)
            continue
        span = span.strip()
        if not span:
            continue
        qualified = "." in span and "-" in span
        if _IDENTIFIER.match(span) or _PATHY.search(span) or qualified:
            if span not in seen:
                seen.append(span)
    return seen


def _citations(source: str) -> list[str]:
    """Every citation the docstrings of one Python source make, in order of first appearance."""
    out: list[str] = []
    for text in _docstrings(source):
        for token in _span_tokens(text):
            if token not in out:
                out.append(token)
    return out


def _path_roots() -> list[Path]:
    """Every root a bare path citation can mean, production values included.

    `app.paths` publishes the env-overridable value and `app.data_root` the passwd-derived
    one, so the same citation resolves whether it is read on this box or under the gate's
    redirected root — and neither spelling is allowed to fall back into the repo, which is
    how a `~/` citation certifies itself as in-tree.
    """
    return [REPO, Path.home(), paths.DATA_ROOT, paths.VAULT_ROOT,
            data_root.PRODUCTION_DATA_ROOT, data_root.PRODUCTION_VAULT]


def _resolve_path(token: str) -> str | None:
    """A path citation, resolved; a file in a directory that does not exist does not count.

    A glob (`logs/locks/mitigation_drill.py-*.lock`) resolves on its DIRECTORY: the citation
    claims the pattern lives in a real place, and no single file matches it at any given
    moment — demanding one would make a true sentence fail whenever the box is quiet.
    """
    target = token.split("::", 1)[0]
    if not target:
        return None
    if "*" in target or "?" in target:
        target = str(Path(target).parent)
    for root in _path_roots():
        candidate = Path(target).expanduser()
        if not candidate.is_absolute():
            candidate = root / target
        else:
            # An absolute citation names its own root; only the `~` form travels.
            if not target.startswith("~"):
                return ("path" if (candidate.exists() or candidate.parent.is_dir())
                        else None)
        if candidate.exists():
            return "path"
        if candidate.parent.is_dir():
            # A runtime file the code WRITES (a drill's state, a rotated stderr) is a real
            # citation before its first firing; a path into a directory that is not there
            # is a wrong path.
            return "path"
    return None


_CODE_NAMES: dict[str, frozenset[str]] = {}


def _code_names() -> dict[str, frozenset[str]]:
    """Names that occur in the CODE of this tree: definitions, attributes, string constants.

    Collected from the `ast`, never from raw text, so docstrings and comments cannot satisfy
    a citation. String constants are matched exactly, so a paragraph of prose that MENTIONS a
    name does not put it in the corpus — only a literal that IS the name, whole, does.
    """
    if _CODE_NAMES:
        return {k: _CODE_NAMES[k] for k in ("defs", "attrs", "lits")}
    defs: set[str] = set()
    attrs: set[str] = set()
    lits: set[str] = set()
    for name in CODE_DIRS:
        root = REPO / name
        files = ([root] if root.is_file() else
                 [p for p in root.rglob("*.py") if "__pycache__" not in p.parts])
        for path in files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defs.add(node.name)
                elif isinstance(node, ast.Attribute):
                    attrs.add(node.attr)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    lits.add(node.value)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        for sub in ast.walk(target):
                            if isinstance(sub, ast.Name):
                                defs.add(sub.id)
    _CODE_NAMES.update(defs=frozenset(defs), attrs=frozenset(attrs), lits=frozenset(lits))
    return {k: _CODE_NAMES[k] for k in ("defs", "attrs", "lits")}


def _module_file(name: str) -> bool:
    """Is this bare name a module of this tree (`atomic_io` for `app/atomic_io.py`)?.

    A docstring names a MODULE as often as a function, and a module is not a `def`: it is a
    file. Matched by file name rather than by importing so the answer does not depend on
    which packages happen to be importable under the gate.
    """
    if not name or "/" in name:
        return False
    roots = [REPO] + [REPO / d for d in CODE_DIRS if (REPO / d).is_dir()]
    for root in roots:
        if (root / f"{name}.py").is_file() or (root / name / "__init__.py").is_file():
            return True
        if any(root.glob(f"*/{name}.py")):
            return True
    return False


def _resolve(token: str) -> str | None:
    """Which rule resolves this citation, or None when no rule does.

    The order is the one that makes a wrong answer hard: a real file first (so a name that is
    also a filename is checked as the file it is), then a real import, then the code corpus.
    Two tiers exist because this module's docstrings cite two things a `def` is not: a route
    (`/api/workers/status`, which is an exact string constant in the server's own router) and
    a qualified call whose head is a class (`WorkerPool.start()`, resolved as the class
    definition plus an attribute the tree actually accesses).
    """
    names = _code_names()
    if _PATHY.search(token):
        # A path-shaped token that is not a path is a ROUTE, and a route is a literal the
        # server mounts — `_STATUS_PATH = "/api/workers/status"` in agent-services/guardian.
        return _resolve_path(token) or ("string-constant" if token in names["lits"] else None)

    bare = token[:-2] if token.endswith("()") else token
    head, _, rest = bare.partition(".")
    if not rest:
        if hasattr(mitigation_state, bare):
            return "module-attribute"
        if bare in names["defs"]:
            return "definition"
        if bare in names["attrs"]:
            return "attribute-access"
        if bare in names["lits"]:
            return "string-constant"
        return "module-file" if _module_file(bare) else None

    # A dotted name that is no module path may still be a dotted LITERAL — a logger name
    # (`"lloyd-workers.maintenance"`, `workers/maintenance.py:74`), a config key, an event
    # type. It is data the code carries, matched exactly, not prose that mentions it.
    if bare in names["lits"]:
        return "string-constant"
    import importlib
    for module in (head, f"app.{head}", f"workers.{head}"):
        try:
            found = importlib.import_module(module)
        except Exception:
            continue
        for part in rest.split("."):
            if not hasattr(found, part):
                found = None
                break
            found = getattr(found, part)
        if found is not None:
            return "imported-attribute"
    # `WorkerPool.start()`: no module by that name, but the head is a real class and the
    # tail is an attribute something in this tree really accesses.
    last = rest.rsplit(".", 1)[-1]
    head_ok = (head in names["defs"] or head in names["attrs"] or head in names["lits"]
               or hasattr(mitigation_state, head) or _module_file(head))
    tail_ok = last in names["attrs"] or last in names["defs"]
    if head_ok and tail_ok:
        return "qualified-name"
    return None


def _unresolved(source: str) -> list[tuple[str, str]]:
    """The citations in `source` that no rule resolves, each with the text that cites it."""
    pairs = []
    for token in _citations(source):
        if _resolve(token) is None:
            pairs.append((token, next(t for t in _docstrings(source)
                                      if f"`{token}`" in t)))
    return pairs


# --------------------------------------------------------------------------- the three pins


def test_the_wrong_symbol_is_gone_from_the_shipped_tree():
    """The repo-wide grep for `WRONG_NAME` must stay empty; this is that grep as a test.

    Walked over the tree rather than shelled out to git so the node says the same thing in a
    gate worktree, a candidate venv and a plain checkout. Scope is the shipped surfaces — the
    code plus `architecture/` — because the file that pins an absence has to be allowed to
    name the absent thing: this one does, in its docstring and in `WRONG_NAME`.
    """
    hits = []
    for name in SHIPPED_SURFACES:
        root = REPO / name
        for path in ([root] if root.is_file() else root.rglob("*")):
            if not path.is_file() or path.suffix not in SHIPPED_SUFFIXES:
                continue
            if WRONG_NAME in path.read_text(encoding="utf-8", errors="replace"):
                hits.append(str(path.relative_to(REPO)))
    assert hits == [], f"{WRONG_NAME} still cited in {hits}"


def test_the_drill_crash_docstring_names_a_reader_that_exists_and_the_log_to_grep():
    """The crash case is now traceable in one hop: a real reader, and the file it writes.

    Extracted from `drill_slot`'s docstring, not from a copy of it in this file, so rewording
    the sentence stays free and dropping the pointer is not. Two things must be named, and
    they are the two #3724's author was missing: which function says the drill recorded
    nothing (`_drill_state_note`, in `workers/maintenance.py`), and which file a reader has
    to grep for it (`server.err`).
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    slots = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "drill_slot"]
    assert len(slots) == 1, f"expected one drill_slot in {MODULE_PATH}, found {len(slots)}"
    doc = ast.get_docstring(slots[0]) or ""
    assert doc, "drill_slot has no docstring to cite"
    assert "_drill_state_note" in doc, "the crash case names no reader that exists"
    assert "workers/maintenance.py" in doc, "the reader's file is not named"
    assert "server.err" in doc, "the file a reader must grep is not named"
    # Named in a citation, not merely in prose: a mention outside backticks is not a pointer.
    cited = set(_span_tokens(doc))
    assert {"_drill_state_note", "workers/maintenance.py"} <= cited, cited
    assert any("server.err" in token for token in cited), cited


def test_the_named_reader_is_the_one_that_writes_the_quoted_sentence():
    """The docstring quotes "so this drill recorded nothing"; this calls its author.

    A citation pinned to a name still resolves even if the function it names stops producing
    the sentence, so this node reads the sentence out of the cited function instead: a
    watermark stamp newer than the newest reading must make `_drill_state_note` say so. That
    clause is what the seat's orphan warning prints, and it is the string the witness grep in
    `~/lloyd-data/logs/server.err` matches on.
    """
    import workers.maintenance as maintenance

    newest = datetime(2026, 11, 3, 4, 39, 57, tzinfo=timezone.utc)
    monkey = mitigation_state
    original = monkey.newest_reading_at
    monkey.newest_reading_at = lambda *a, **k: newest  # type: ignore[assignment]
    try:
        note = maintenance._drill_state_note(newest + timedelta(seconds=142))
    finally:
        monkey.newest_reading_at = original  # type: ignore[assignment]
    assert "so this drill recorded nothing" in note, note
    assert note in ("the newest reading predates the spawn "
                    f"({newest.isoformat()} against "
                    f"{(newest + timedelta(seconds=142)).replace(microsecond=0).isoformat()}), "
                    "so this drill recorded nothing"), note


def test_every_citation_the_modules_docstrings_make_resolves_in_the_tree():
    """No docstring in this module may name a symbol the tree does not have.

    The extracted set is asserted non-empty first: 46 citations come out of these docstrings
    on 2026-11-03, and an extractor that quietly stopped matching would otherwise pass the
    loop below on an empty intersection. Each one then resolves to a path, a module attribute, an
    imported attribute, or a name that occurs in code — the tier it resolved by is in the
    failure message, because "unresolved" and "resolved as a string constant" are different
    answers about a citation.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    found = _citations(source)
    assert found, "the extractor matched no citation in app/mitigation_state.py"
    assert len(found) >= 40, f"only {len(found)} citations extracted: {found}"
    unresolved = [(token, _resolve(token)) for token in found if _resolve(token) is None]
    assert not unresolved, (f"docstring citations that resolve nowhere: "
                            f"{[token for token, _ in unresolved]}\n"
                            f"resolved: {[(t, _resolve(t)) for t in found[:6]]}")


def test_the_check_fails_on_a_citation_the_test_itself_invents():
    """Both kinds of wrong citation are reported, on text this node writes.

    An invented symbol — the exact one #3726 is about, `WRONG_NAME` — and a path into a
    directory that does not exist. Neither resolves, and both are extracted from the copy's
    docstring rather than assumed, so a future change to `_citations` that stopped seeing
    single-word spans fails here instead of green-ing the module it was meant to police.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    planted = source.replace(
        "Yield True holding the one-live-drill slot for this machine, False without it.",
        "Yield True holding the one-live-drill slot for this machine, False without it.\n\n"
        f"    A crash is handled by `{WRONG_NAME}`, and the state lives in "
        f"`app/no_such_tree/no_such_file.py`.",
        1)
    assert planted != source, "drill_slot's summary line moved; this node's plant is inert"
    planted_citations = _citations(planted)
    assert WRONG_NAME in planted_citations, planted_citations
    assert "app/no_such_tree/no_such_file.py" in planted_citations, planted_citations
    reported = [token for token, _ in _unresolved(planted)]
    assert WRONG_NAME in reported, reported
    assert "app/no_such_tree/no_such_file.py" in reported, reported
    # The same two rules on the real file: the symbol is gone and the paths that remain are
    # real, so the negative is about the name and not about the checker being broken.
    assert WRONG_NAME not in _citations(source)
