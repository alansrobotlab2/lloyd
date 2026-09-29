"""Every module that DEFINES a contract invariant has to be a prompt-surface trigger.

Why this test exists
--------------------
`app/prompt_surface.py` is the single definition of what the loaded operating
contract must contain and how large it may be: `GATE_HEADS` (the headings that
make up the L0 gate stack), `LOAD_BEARING` (the literals a trim may not drop),
the three shape ratios and `MEMORY_CEILINGS`. #1758 found it sitting in neither
`Gate.PROMPT_SURFACE_PATHS` nor `Gate.PROMPT_SURFACE_VAULT`, so a round that
raised `GATE_STACK_CEILING` or deleted a `LOAD_BEARING` entry — changing what the
identity file is *allowed to look like*, with the identity file itself untouched —
was answered by a `prompt_surface` rung that reported `{"skipped": true, "reason":
"not touched"}`. The `tests` rung agreed with it as well, because
`tests/test_prompt_surface_budget.py` imports these constants instead of restating
them: the check became the round's own assumption.

Adding the one path closes the instance. This file closes the class. The property
that has to hold is over an OPEN set — any module anyone writes tomorrow may
become the second definition site — and a hand-kept tuple cannot hold over an open
set. The tuple stays as the data; this test is what keeps it current, by deriving
the set of modules that assign one of `INVARIANT_NAMES` at module level and asking
the gate's own trigger whether each one is reachable.

Definition sites, not consumers
-------------------------------
The scoping is the whole design. Six more modules name these constants in their
docstrings or import them (`scripts/automod/vault_round.py`,
`scripts/autoresearch/promote.py`, `app/memory_ceiling.py`,
`agent_mcp/session.py`, `agent_mcp/vault.py`, `agent_mcp/builtin_fs.py`), and a
consumer cannot loosen the contract — it reads the number the definition site
set. Triggering the rung on a consumer would spend the live-model tool-choice eval
(`rung_prompt_surface`, `timeout=1800`) on every ordinary round that touches one
of those hot paths, for no added coverage. So the scan keys on an assignment at
module level, and the negative-control test below proves a consumer is NOT
reachable, which is what stops the main assertion being satisfied by a predicate
that matches everything.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automod.gate import Gate  # noqa: E402

#: The names that decide what the loaded contract is allowed to look like. An
#: invariant constant is load-bearing the way a load-bearing wall is: the shape
#: of the prompt is checked against THESE numbers, wherever they end up living.
INVARIANT_NAMES = (
    "LOAD_BEARING",
    "GATE_HEADS",
    "GATE_STACK_CEILING",
    "PROHIBITION_RATIO_CEILING",
    "DUPLICATE_CONTRACT_CEILING",
    "MEMORY_CEILINGS",
    "USER_MD_CEILING_BYTES",
    "MEMORY_MD_CEILING_BYTES",
    # The index cap `MEMORY_MD_CEILING_BYTES` is derived from, and the one
    # `tests/test_prompt_surface_budget.py` branches on. It ships with the other
    # two today, but a module that defined only the index ceiling would otherwise
    # sit outside the scan while sitting inside the contract.
    "MEMORY_MD_INDEX_CEILING_BYTES",
)


def _tracked_python_files(root: Path = ROOT) -> list[str]:
    """Every tracked ``.py`` in the repo, as repo-relative paths.

    Tracked files, not a filesystem walk: the thing that must not land unlisted
    is a file that can be committed, and a walk would sweep build directories.
    """
    out = subprocess.run(["git", "-C", str(root), "ls-files", "*.py"],
                         capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def _module_level_assignments(tree: ast.Module) -> set[str]:
    """Names assigned at module level — the definition sites, nothing else.

    An `import` or `from ... import` is a consumer and is excluded by never
    appearing in `tree.body` as an `Assign`/`AnnAssign`; a name bound inside a
    function or class body is local and is excluded by walking `tree.body` only,
    one level deep. `test_the_scanner_reads_definitions_and_ignores_everything_else`
    holds both exclusions open.
    """
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    bound.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    return bound


def defining_modules(root: Path = ROOT,
                     files=None) -> dict[str, tuple[str, ...]]:
    """Repo-relative module -> the invariant names it assigns at module level.

    `root`/`files` are injectable so the same scan can be pointed at a scratch
    tree by `test_the_scan_and_the_decision_reject_a_second_contract` — a
    synthetic second contract written under a temporary root is the only way to
    show this test bites without committing one to the real tree.
    """
    hits: dict[str, tuple[str, ...]] = {}
    for rel in (_tracked_python_files(root) if files is None else files):
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # A substring prefilter, not the test: every one of these names is a
        # literal identifier, so a file that cannot contain one is skipped
        # without parsing. The parse below is what actually decides.
        if not any(n in text for n in INVARIANT_NAMES):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        found = sorted(_module_level_assignments(tree) & set(INVARIANT_NAMES))
        if found:
            hits[rel] = tuple(found)
    return hits


def _reachable(path: str) -> bool:
    """Would the gate's own trigger fire on a diff naming `path`?

    The real predicate, not a copy of it — a re-typed matcher here would drift
    from `_touches_prompt_surface` and report the gap closed either way. Built
    the way `tests/test_gate_prompt_surface_rung.py` builds it: the method reads
    `report.changed_paths` and nothing else, so constructing a `Gate` (which
    shells out to git) is not needed.
    """
    g = Gate.__new__(Gate)
    g.report = type("_Report", (), {"changed_paths": [path]})()
    return g._touches_prompt_surface()


def unlisted_definition_sites(defs: dict[str, tuple[str, ...]]) -> list[str]:
    """The one decision both this file's tests share: which definition sites the
    gate's trigger would NOT fire on.

    The main assertion and the scratch-tree falsification both call this, so the
    thing that can fail is the thing that is relied on — a copy of the predicate
    in the falsification test would pass while the real assertion rotted.
    """
    return sorted(path for path in defs if not _reachable(path))


def test_every_invariant_definition_site_is_a_prompt_surface_trigger():
    """Clause 3 of #1758. No module that defines a contract invariant may be
    outside the trigger that scores a change to what the model is told.

    Today exactly one module qualifies — `app/prompt_surface.py`, the only file in
    the tree that assigns any of `INVARIANT_NAMES` at module level — so the value
    of this node is entirely prospective: it exists to make the SECOND definition
    site, added by someone who never read this file, fail its own round instead of
    clearing the ladder unobserved. `test_the_scan_and_the_decision_reject_a_second_contract`
    is that future round, run now against a scratch tree.
    """
    defs = defining_modules()
    # Denominator guard: an empty scan reads as a clean bill of health and means
    # nothing. If the scanner matched nothing, the scan itself is broken.
    assert defs, (
        f"no module in the tree assigns any of {INVARIANT_NAMES} — the scan or "
        "the constant names are wrong, and a 0-hit result here is not a pass")
    unlisted = unlisted_definition_sites(defs)
    assert not unlisted, (
        f"{unlisted} assign a contract invariant at module level "
        f"({ {p: defs[p] for p in unlisted} }) but are not reachable from "
        "Gate.PROMPT_SURFACE_PATHS / PROMPT_SURFACE_VAULT, so a round that "
        "loosened them runs no behavioural check. Either list the path in "
        "Gate.PROMPT_SURFACE_PATHS (scripts/automod/gate.py) or import the "
        "constant from app/prompt_surface.py instead of defining a second copy.")
    assert "app/prompt_surface.py" in defs, (
        f"app/prompt_surface.py is the definition site this test was written for "
        f"and the scan did not find it: {sorted(defs)}")


def test_the_scanner_reads_definitions_and_ignores_everything_else():
    """The scan is a parser, so the parser gets tested — otherwise the assertion
    above cannot fail, which is the failure this loop refuses outright.

    One source, three shapes: a module-level assignment (a definition, must be
    found), the same name bound inside a function (local, must not be), and an
    import of another invariant (a consumer, must not be).
    """
    src = (
        "GATE_STACK_CEILING = 0.50\n"
        "import app.prompt_surface as ps\n"
        "from app.prompt_surface import MEMORY_CEILINGS\n"
        "def _inner():\n"
        "    LOAD_BEARING = ('{\"status\": \"blocked\"',)\n"
        "    return LOAD_BEARING\n"
        "class Holder:\n"
        "    def method(self):\n"
        "        USER_MD_CEILING_BYTES = 1\n"
        "        return USER_MD_CEILING_BYTES\n"
    )
    bound = _module_level_assignments(ast.parse(src))
    assert "GATE_STACK_CEILING" in bound, (
        "the scanner missed a module-level assignment, so a new definition site "
        "would land unlisted and the main assertion would never have a chance to fire")
    assert "LOAD_BEARING" not in bound, "a function-local binding is not a definition site"
    assert "USER_MD_CEILING_BYTES" not in bound, "a method-local binding is not a definition site"
    assert "MEMORY_CEILINGS" not in bound, (
        "an import is a consumer, and triggering the eval on every consumer "
        "would spend a live-model run on ordinary rounds for no coverage")


def test_the_scan_and_the_decision_reject_a_second_contract(tmp_path):
    """Falsification of the clause-3 property, run through the same code path.

    The review rung is right that a property over an open set cannot be shown to
    bite by asserting over today's one-member result, and writing a second
    contract into the real tree to prove it would be the very thing the test
    forbids. So a scratch tree gets one: a module named like the listed
    definition site and one named like something nobody listed, both assigning an
    invariant at module level. The scan must find both, and the shared decision
    must report exactly the unlisted one.

    Both directions are load-bearing. If the scan matched nothing the assertion
    passes vacuously; if the decision matched everything the real tree would
    never fail. This node goes red on either failure.
    """
    listed = tmp_path / "app" / "prompt_surface.py"      # in PROMPT_SURFACE_PATHS
    unlisted = tmp_path / "app" / "second_contract.py"    # nobody listed it
    for mod, name in ((listed, "MEMORY_CEILINGS"), (unlisted, "GATE_STACK_CEILING")):
        mod.parent.mkdir(parents=True, exist_ok=True)
        mod.write_text(f'"""A second loaded-contract definition."""\n{name} = {{"a": 1}}\n'
                       if name == "MEMORY_CEILINGS" else
                       f'"""A second loaded-contract definition."""\n{name} = 0.90\n')

    files = ["app/prompt_surface.py", "app/second_contract.py"]
    defs = defining_modules(root=tmp_path, files=files)
    assert set(defs) == set(files), (
        f"the scan found {sorted(defs)} of {files}, so the property below would be "
        "asserted over a set the scan never populated")
    assert unlisted_definition_sites(defs) == ["app/second_contract.py"], (
        "a module that defines a contract invariant outside the gate's tuples was "
        "not reported — the class is not closed, which is the whole point of this file")


def test_a_consumer_path_is_not_reachable():
    """The negative control for the trigger predicate the main assertion leans on.

    `app/memory_ceiling.py` reads `MEMORY_CEILINGS` but does not define it, and it
    is not a prompt-surface path. If this asserted True, the main test would be
    passing because `_reachable` matches everything — a green that means nothing.
    """
    assert _reachable("app/prompt_surface.py") is True, (
        "the fix under test: the invariant module itself must trigger the rung")
    assert _reachable("app/memory_ceiling.py") is False
    assert _reachable("scripts/automod/vault_round.py") is False
    assert _reachable("agent_mcp/session.py") is False
