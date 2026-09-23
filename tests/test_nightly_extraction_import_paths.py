"""#755 — the nightly extractor resolves its imports from the tree that owns it.

Why this file exists
--------------------
``scripts/memory/next-gen-memory/nightly_extraction.py`` builds its own import
environment with ``sys.path.insert`` calls. One of them was hardcoded to the live
checkout — ``Path.home() / "lloyd" / "scripts" / "memory"`` — and another inserted
``~/obsidian/agents/memory/scripts/next-gen-memory``, a directory that does not
exist on this box. Run from ``~/lloyd`` every path agrees, so the nightly was
correct and no probe complained. Run from a self-modification worktree or any
other copy of the checkout, the module pulled its own siblings from that copy
while ``content_hasher`` came from the live tree: a hybrid that imports cleanly
and measures nothing. Triage reproduced exactly that on 2026-09-18 with
``importlib.util.find_spec`` over the tree's own inserts — from worktree
``SM_20260918_025936``, ``nightly_extraction``, ``fact_extractor``,
``relations_index`` and ``profile_generator`` resolved under the worktree while
``content_hasher`` resolved to
``/home/alansrobotlab/lloyd/scripts/memory/content_hasher.py``.

That is the run shape a paired before/after extraction comparison needs (the
#537 step-4 A/B, a #608 drift-control run), which is why the defect was latent in
traffic and fatal in measurement: autonomy tasks 24, 48, 67 and 74 all invoke the
script from ``~/lloyd``, where resolution is consistent.

What the fix does NOT do
------------------------
The module's *state* paths stay home-absolute — the run log, the pre-clean
backups, the graph working dir and the single-instance lock. A worktree that
runs the nightly must still write the live ``_pipeline``; only import resolution
became tree-relative. Clause 4 pins that boundary, because the next
"make it portable" pass would otherwise move the live log and lock into a
throwaway tree. One non-claim worth stating: even after this fix, a worktree run
shares ``~/lloyd/_pipeline/content-hashes.json``, so a paired before/after is
still not isolated for incremental-hash state (``content_hasher.py`` reads that
path itself, and its ``LLOYD_CONTENT_HASHES`` override is the isolation lever).

Each test below names the clause it pins. All four were verified red against the
pre-fix file first: run on ``git show fb962366:scripts/memory/next-gen-memory/nightly_extraction.py``
copied into a scratch tree, clause 1 reported ``content_hasher`` resolving to
``/home/alansrobotlab/lloyd/scripts/memory/content_hasher.py`` and clause 2 named
the ``Path.home()``-built entry among the search paths.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_MEMORY = ROOT / "scripts" / "memory"
NGM = SCRIPTS_MEMORY / "next-gen-memory"
NIGHTLY = NGM / "nightly_extraction.py"

# The four names the nightly imports by bare name. `content_hasher` lives one
# directory up in `scripts/memory/`; the other three sit beside the script.
SIBLINGS = ("content_hasher", "fact_extractor", "relations_index", "profile_generator")

LIVE_CHECKOUT = Path.home() / "lloyd"


# ── the instrument ────────────────────────────────────────────────────────────

def _is_path_insert(node: ast.AST) -> bool:
    """True for a statement of the form ``sys.path.insert(...)``."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "insert"
        and isinstance(node.value.func.value, ast.Attribute)
        and node.value.func.value.attr == "path"
        and isinstance(node.value.func.value.value, ast.Name)
        and node.value.func.value.value.id == "sys"
    )


def _top_level(module: Path) -> tuple[dict, list[str]]:
    """Execute the module's top-level assignments and ``sys.path.insert`` calls.

    Returns ``(namespace, search_path)`` where ``search_path`` is what
    ``sys.path`` looks like after the module's own inserts, in effect order —
    the head a later bare import is resolved against.

    No import runs, so no fact store opens and no ``_pipeline`` log line is
    written; that is the same no-execution discipline triage measured under, and
    it is why this file can ask "which file would ``content_hasher`` bind?"
    without running the extractor.

    The assignments are part of the instrument because an insert can be laundered
    through a constant: the line #755 deleted read
    ``sys.path.insert(0, str(VAULT / "agents" / "memory" / "scripts" / "next-gen-memory"))``
    with ``VAULT = Path.home() / "obsidian"`` twelve lines above it. A checker
    that inspected only the call would call that ``__file__``-clean.

    An assignment this cannot evaluate raises rather than being skipped: a skipped
    statement silently shrinks what the instrument can see.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"))
    head: list[str] = []
    ns: dict = {
        "__file__": str(module),
        "Path": Path,
        "os": os,
        "sys": types.SimpleNamespace(path=head),
    }
    for node in tree.body:
        # `from app.paths import …` is executed too: the state paths are built
        # from the data root it names, and it opens nothing.
        is_paths_import = isinstance(node, ast.ImportFrom) and node.module == "app.paths"
        if isinstance(node, (ast.Assign, ast.AnnAssign)) or _is_path_insert(node) \
                or is_paths_import:
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(module), "exec"),
                ns,
            )
    return ns, [str(entry) for entry in head]


def _bind(name: str, search_path: list[str]) -> str | None:
    """Where ``name`` resolves when ``sys.path`` is exactly ``search_path``.

    ``sys.modules`` is consulted *before* ``sys.path``, so a name an earlier test
    already imported would answer for a tree this call was never handed — the
    failure mode ``tests/test_relations_index_read_only.py`` documents for its own
    fixture. Popping the name and restoring it is what makes the answer about
    ``search_path`` and nothing else.
    """
    saved_path = list(sys.path)
    cached = sys.modules.pop(name, None)
    try:
        sys.path[:] = search_path
        spec = importlib.util.find_spec(name)
    finally:
        sys.path[:] = saved_path
        if cached is not None:
            sys.modules[name] = cached
    return None if spec is None else spec.origin


@pytest.fixture(scope="module")
def copied_tree(tmp_path_factory) -> Path:
    """The checkout's ``scripts/memory`` tree, copied under a fresh root.

    The copy keeps the directory depth the module's own path arithmetic assumes —
    ``parents[3]`` of ``<root>/scripts/memory/next-gen-memory/nightly_extraction.py``
    is ``<root>`` — so the copy exercises the same relative logic a self-mod
    worktree exercises. ``__pycache__`` is left behind: a byte-code cache copied
    from the source tree is a second answer to the question being asked.
    """
    root = tmp_path_factory.mktemp("checkout")
    shutil.copytree(
        SCRIPTS_MEMORY,
        root / "scripts" / "memory",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return root


def _copy_search_path(root: Path) -> list[str]:
    """The ``sys.path`` a run inside the copy would resolve against.

    The copy's own ``next-gen-memory`` directory goes first because that is what
    happens to a script's directory when it is run, and what
    ``scripts/memory/kg_rebuild.py::_corpus_size`` injects when it loads the file
    by path; then the entries the module builds for itself.
    """
    copied_ngm = root / "scripts" / "memory" / "next-gen-memory"
    _, head = _top_level(copied_ngm / "nightly_extraction.py")
    return [str(copied_ngm)] + head


# ── clause 1 ──────────────────────────────────────────────────────────────────

def test_a_copy_binds_every_bare_module_inside_itself(copied_tree):
    """Clause 1: importing ``nightly_extraction`` from a copy of the checkout
    binds every bare module — ``content_hasher`` above all, since it lives in
    ``scripts/memory/`` rather than beside the script — to that copy.

    Before #755 this failed on ``content_hasher`` alone: the other three names
    came off the copy's own directory, but the hardcoded
    ``Path.home() / "lloyd" / "scripts" / "memory"`` insert handed the live tree's
    ``content_hasher`` to a run that was measuring something else.
    """
    search_path = _copy_search_path(copied_tree)
    origins = {name: _bind(name, search_path) for name in SIBLINGS}
    for name, origin in origins.items():
        assert origin is not None, (
            f"{name} resolved to nothing against {search_path} — the copy is "
            "incomplete, so this test would be asserting nothing"
        )
        resolved = Path(origin).resolve()
        assert resolved.is_relative_to(copied_tree), (
            f"a nightly run inside the copy bound {name} to {resolved}, outside the "
            f"copy rooted at {copied_tree}"
        )
        assert not resolved.is_relative_to(LIVE_CHECKOUT), (
            f"a nightly run inside the copy bound {name} to {resolved} in the live "
            f"checkout {LIVE_CHECKOUT} — the hybrid #755 is about: the extractor's "
            "logic from one tree, its incremental-hash code from another"
        )


# ── clause 2 ──────────────────────────────────────────────────────────────────

def test_the_module_builds_no_search_path_from_home():
    """Clause 2: the ``scripts/memory`` search path is derived from
    ``Path(__file__)``, and no ``sys.path`` entry is built from ``Path.home()``.

    Both halves are about text that must stay true for a tree that is *not* this
    one, so the check runs the module's own statements with ``__file__`` bound to
    wherever the file actually is and asks where each entry lands.
    """
    _, head = _top_level(NIGHTLY)
    assert head, (
        "no sys.path entry was seen at all — the instrument is not reading "
        f"{NIGHTLY}, so every assert below would pass on an empty list"
    )
    root = NIGHTLY.resolve().parents[3]
    entries = {Path(entry).resolve() for entry in head}
    assert (root / "scripts" / "memory") in entries, (
        f"the module's scripts/memory search path is not this tree's "
        f"{root / 'scripts' / 'memory'}; it built {sorted(entries)}"
    )
    for entry in sorted(entries):
        assert entry.is_relative_to(root), (
            f"nightly_extraction.py puts {entry} on sys.path, outside the tree it "
            f"lives in ({root}) — every bare import in the file is decided by "
            "entries like this one"
        )
    for node in ast.walk(ast.parse(NIGHTLY.read_text(encoding="utf-8"))):
        if _is_path_insert(node):
            source = ast.unparse(node)
            assert "home()" not in source, (
                f"{source} builds a search path off Path.home(); an entry like that "
                "binds one tree's modules no matter which tree runs the script"
            )


# ── clause 3 ──────────────────────────────────────────────────────────────────

#: The item's own verify command, verbatim in shape: insert the nightly's
#: directory relative to the working directory and import the three modules.
PROBE = "\n".join([
    "import sys",
    "sys.path.insert(0, 'scripts/memory/next-gen-memory')",
    "import nightly_extraction, fact_extractor, content_hasher",
    "for m in (nightly_extraction, fact_extractor, content_hasher):",
    "    print(m.__name__ + chr(9) + m.__file__)",
])


def test_running_in_place_resolves_the_three_modules_in_one_tree():
    """Clause 3: run in place, resolution is unchanged — ``nightly_extraction``,
    ``fact_extractor`` and ``content_hasher`` all resolve under the tree that was
    invoked.

    The subprocess is run with ``cwd`` at the repo root of *this* file's tree, so
    the same test reads as "all three under ``~/lloyd``" when the suite runs in
    the live checkout and "all three under the worktree" when the gate runs it
    from one. Both readings are the same property; the live-checkout reading was
    re-measured by hand on 2026-09-22 with this exact probe and is unchanged by
    the fix (all three under ``/home/alansrobotlab/lloyd``).

    This is the only test here that actually imports the modules, and it is
    deliberate that it does so in a throwaway interpreter: importing
    ``nightly_extraction`` reaches ``app.kg_store`` and ``content_hasher``, and the answer
    that matters — which file each name got — must not be contaminated by
    whatever this suite already imported.
    """
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the probe died: {proc.stderr[-2000:]}"
    rows = {
        line.split("\t")[0]: line.split("\t")[1]
        for line in proc.stdout.splitlines()
        if line.count("\t") == 1
    }
    assert set(rows) == {"nightly_extraction", "fact_extractor", "content_hasher"}, (
        f"the probe printed {rows} — three tab-separated module/path rows are "
        "expected, anything else means the imports did not all happen"
    )
    for name, path in sorted(rows.items()):
        resolved = Path(path).resolve()
        assert resolved.is_relative_to(ROOT), (
            f"run from {ROOT}, {name} resolved to {resolved} — outside the invoked "
            "tree, which is how a run starts measuring a hybrid again"
        )


# ── clause 4 ──────────────────────────────────────────────────────────────────

#: Each state path, by the name it is assigned to, and the fragment of the live
#: ``_pipeline`` layout it names. Line numbers in nightly_extraction.py as of
#: this writing: log :105, pre-clean backups :144, graph dir :152, lock :563.
STATE_SITES = {
    "log_file": "nightly-extraction.log",
    "dest": "backups",
    "graph_dir": "memory-graph",
    "_LOCK_PATH": "nightly_extraction.lock",
}


def _assigned_value(tree: ast.Module, target: str) -> str | None:
    """The source assigned to ``self.<target>`` or ``<target>``, unparsed."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            named = (
                (isinstance(t, ast.Attribute) and t.attr == target)
                or (isinstance(t, ast.Name) and t.id == target)
            )
            if named:
                return ast.unparse(node.value)
    return None


def test_state_paths_still_address_the_live_pipeline():
    """Clause 4: only import resolution changed — the log, the pre-clean backup
    dir, the graph working dir and the single-instance lock still land under the
    live data root's ``_pipeline`` (``~/lloyd-data``, or an explicit ``LLOYD_DATA``).

    The scope trap this pins: "make the paths tree-relative" applied to these four
    would relocate the live nightly's log, lock and backups into whatever tree
    happened to invoke it, and two checkouts would then run without the flock that
    keeps them off each other's fact tree. ``self.log_file``/``dest``/``graph_dir``
    are checked as source because they are built inside methods; the lock is
    checked as a *value* because it is a module-level assignment the instrument
    can evaluate.
    """
    tree = ast.parse(NIGHTLY.read_text(encoding="utf-8"))
    for target, fragment in STATE_SITES.items():
        source = _assigned_value(tree, target)
        assert source is not None, (
            f"nightly_extraction.py no longer assigns `{target}` — the site that "
            "kept this state in the live _pipeline is gone, so the asserts below "
            "would have nothing to grade"
        )
        # Checked against `ast.unparse` output, which quotes with single quotes:
        # match the bare word, not a quoted literal, or the assert is testing the
        # unparser's formatting instead of the path.
        assert "_STATE_PIPELINE" in source, (
            f"`{target}` no longer addresses the live data root's _pipeline; it reads: {source}"
        )
        assert fragment in source, f"`{target}` no longer names {fragment}: {source}"
        assert "_REPO_ROOT" not in source and "__file__" not in source, (
            f"`{target}` is now derived from the tree instead of the live pipeline "
            f"({source}) — a worktree run would move live state, not just imports"
        )

    monkeypatched = os.environ.pop("LLOYD_EXTRACTION_LOCK", None)
    try:
        ns, _ = _top_level(NIGHTLY)
    finally:
        if monkeypatched is not None:
            os.environ["LLOYD_EXTRACTION_LOCK"] = monkeypatched
    from app.paths import production_data_root
    live_pipeline = Path(os.environ.get("LLOYD_DATA") or production_data_root()) / "_pipeline"
    assert ns["_STATE_PIPELINE"] == live_pipeline
    assert ns["_LOCK_PATH"] == live_pipeline / "nightly_extraction.lock", (
        f"the default single-instance lock is {ns['_LOCK_PATH']}, not the live one — "
        "two extractors from two trees would then each hold their own lock and race "
        "on _pipeline/content-hashes.json"
    )
