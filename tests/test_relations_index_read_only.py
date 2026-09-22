"""Backlog #484 — RelationsIndexGenerator must never write a vault source note.

`relations_index.py` used to carry a vault-note rewriter:
``_update_frontmatter()``, reached only by ``add_relation()``, rebuilt a note's
whole frontmatter with ``yaml.dump`` and rejoined the body with
``f"---\\n{dump}---\\n{body}"``. That round-trip reorders keys, drops quoting
and appends one newline byte per call, so it changed bytes without changing
content — and the nightly content-hash gate
(`scripts/memory/content_hasher.py::get_changed_files`, sha256 of bytes) reads
a byte change as new content and re-extracts the note. That is the pattern
``memory/vault-maintenance/vault-maintenance-2026-09-03.md`` named the
"byte-churn family": :167 "churn family unchanged — memory/2026-02-22.md 5×
reprocessed today (byte-churn family confirmed…)", :211 "now at 9 reprocesses
today", :223 "the 02-22 byte-churn loop hit its 10th reprocess today". The log
before it, ``vault-maintenance-2026-08-30.md``, had already recorded the same
note coming back as changed on consecutive runs (:63, :84, :105, :124) without
naming it; there is no 08-31 log. Both functions had no caller left after
f36c522, so this round deleted them (clause 1) rather than leaving an unguarded
writer pointed at the gate.

Each test below names the clause it pins. Nothing here touches ``~/obsidian``
or the live ``~/lloyd/_pipeline/relations-index.json``: the generator's
``vault``, ``index_file`` and ``proposals_file`` are all redirected into
``tmp_path``, and the recorded write targets are asserted, not assumed.

Backlog #1148 added five clauses to this file, on the other half of the same
boundary — writes to the *index* rather than to the vault.
``_pipeline/relations-index.json`` had two writers with two schemas: this
module's ``rebuild()`` wrote ``{edges, stale, built_at}`` and
``scripts/memory/rebuild_index.py`` wrote ``{relationships, total_relationships,
documents_indexed, last_updated}`` (341,373 rows over 3,524 docs on 2026-09-21).
Both run inside scheduled task #24 (Data Pipeline, 6x-daily) in a fixed order,
so the clobber was deterministic, not a race, and this module's own CLI could not
read what the cycle left behind: the no-arg summary printed ``Index loaded: 0
edges`` and ``--query`` raised ``KeyError: 'edges'``. The module also wrote
production state from two read-side paths — its ``--test`` flag built a bare
generator and rebuilt into the live index, and ``get_relations_for_doc`` called
``rebuild()`` whenever ``edges`` was empty, which is exactly what a
``relationships`` file loads as. One file, one owner, one schema now; the five
clauses are pinned below, in ``test_test_flag_leaves_the_live_index_alone``,
``test_get_relations_for_doc_reads_the_relationships_key``,
``test_get_relations_for_doc_performs_no_write``,
``test_no_arg_summary_reports_the_rows_in_the_derived_file`` and
``test_exactly_one_writer_per_index``. Consolidating the two schemas into one
index is left to a person (see the item's needs-human clause); these tests pin
that the collision is gone.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "memory" / "next-gen-memory" / "relations_index.py"
NIGHTLY = ROOT / "scripts" / "memory" / "next-gen-memory" / "nightly_extraction.py"
LIVE_INDEX = Path.home() / "lloyd" / "_pipeline" / "relations-index.json"


def _load(name: str):
    """Import relations_index.py by path — the directory name has a hyphen, so
    it is not an importable package (nightly_extraction.py reaches it the same
    way, through a sys.path insert)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ri(request):
    name = f"ri_{request.node.name.replace('[', '_').replace(']', '_')}"
    mod = _load(name)
    yield mod
    sys.modules.pop(name, None)


@pytest.fixture
def write_targets(monkeypatch):
    """Record every text write the code under test attempts, wherever it aims.

    Both of the repo's text-write helpers are spied: ``pathlib.Path.write_text``
    (and ``write_bytes``, the other way a file gets replaced) and
    ``app.atomic_io.atomic_write_text``. The spy writes through, so the tests
    still exercise the real write path, and the recorded list is what the
    assertions read.
    """
    import app.atomic_io as atomic_io

    recorded: list[Path] = []

    real_write_text = Path.write_text
    real_write_bytes = Path.write_bytes
    real_atomic = atomic_io.atomic_write_text

    def spy_write_text(self, data, *args, **kwargs):
        recorded.append(Path(self))
        return real_write_text(self, data, *args, **kwargs)

    def spy_write_bytes(self, data, *args, **kwargs):
        recorded.append(Path(self))
        return real_write_bytes(self, data, *args, **kwargs)

    def spy_atomic(path, data, *args, **kwargs):
        recorded.append(Path(path) if isinstance(path, str) else Path(str(path)))
        return real_atomic(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy_write_text)
    monkeypatch.setattr(Path, "write_bytes", spy_write_bytes)
    monkeypatch.setattr(atomic_io, "atomic_write_text", spy_atomic)
    return recorded


@pytest.fixture
def vault(tmp_path):
    """A vault root in tmp_path. Keys are deliberately in non-alphabetical
    order and every body is preceded by a blank line, so any code that ever
    re-serialises one of these files shows up in its bytes."""
    root = tmp_path / "vault"
    docs = {
        "knowledge/a.md": "---\ntype: knowledge-note\nsegment: knowledge\ntitle: A\n"
                          "relations:\n  related-to:\n    - knowledge/b.md\n---\n\n# A\n\nbody\n",
        "knowledge/b.md": "---\ntype: knowledge-note\nsegment: knowledge\ntitle: B\n---\n\n# B\n\nbody\n",
        "knowledge/c.md": "---\nsegment: knowledge\ntype: knowledge-note\ntitle: C\n"
                          "relations:\n  depends-on:\n    - knowledge/d.md\n---\n\n# C\n\nbody\n",
        "knowledge/d.md": "---\ntitle: D\ntype: knowledge-note\nsegment: knowledge\n---\n\n# D\n\nbody\n",
        "knowledge/e.md": "---\ntype: knowledge-note\nsegment: knowledge\ntitle: E\n"
                          "relations:\n  superseded-by:\n    - knowledge/f.md\n---\n\n# E\n\nbody\n",
        "knowledge/f.md": "---\ntitle: F\ntype: note\nsegment: memory\n---\n\n# F\n\nbody\n",
        # legacy `related:` spelling, folded into related-to by _parse_relations
        "knowledge/g.md": "---\ntitle: G\ntype: knowledge-note\nsegment: knowledge\n"
                          "related:\n  - knowledge/h.md\n---\n\n# G\n\nbody\n",
        "knowledge/h.md": "---\ntitle: H\ntype: knowledge-note\nsegment: knowledge\n---\n\n# H\n\nbody\n",
        # unrecognised relation type: contributes nothing
        "knowledge/invalid.md": "---\ntitle: X\ntype: knowledge-note\nsegment: knowledge\n"
                                "relations:\n  bogus-type:\n    - knowledge/a.md\n---\n\n# X\n\nbody\n",
    }
    for rel, text in docs.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


REBUILD_SCRIPT = ROOT / "scripts" / "memory" / "rebuild_index.py"
DERIVED_NAME = "relations-index.json"        # the file rebuild_index.py owns
TYPED_NAME = "relations-index-typed.json"    # the file relations_index.py owns
INDEX_NAMES = (DERIVED_NAME, TYPED_NAME)
LIVE_TYPED_INDEX = LIVE_INDEX.parent / TYPED_NAME
# The two modules whose write targets the clause-5 scan resolves fail-closed: a
# blind spot in one of them is exactly how the second writer stayed invisible.
_INDEX_OWNER_PATHS = {
    "scripts/memory/rebuild_index.py",
    "scripts/memory/next-gen-memory/relations_index.py",
}

# What `rebuild_index.py` leaves at the derived path: `relationships`, never
# `edges`. The shape is the live file's (341,373 rows over 3,524 docs on
# 2026-09-21), reduced to three rows so a test can name the ones it expects.
DERIVED_SHAPED = {
    "relationships": [
        {"source": "memory/2026-09-12.md", "target": "knowledge/a.md",
         "type": "wiki-link", "reason": "Both link to [[c]]", "score": 100},
        {"source": "knowledge/b.md", "target": "memory/2026-09-12.md",
         "type": "tag-cluster", "reason": "Share tags: memory, daily-notes",
         "score": 80},
        {"source": "knowledge/c.md", "target": "knowledge/d.md",
         "type": "tag-cluster", "reason": "Share tags: memory", "score": 80},
    ],
    "total_relationships": 3,
    "documents_indexed": 4,
    "last_updated": "2026-09-21T00:00:00",
    "SENTINEL": "derived index — must not move",
}


@pytest.fixture
def generator(ri, vault, tmp_path):
    """A generator pointed entirely at tmp_path — tmp vault, tmp index, no
    proposals file. The live index is ``~/lloyd/_pipeline/relations-index.json``
    and is asserted untouched by every test that rebuilds.

    ``typed_index_file`` (#1148 — the one file this module may write) is
    redirected here as well: a fixture that left it at its default would let any
    test in this file write the real typed index."""
    g = ri.RelationsIndexGenerator()
    g.vault = vault
    g.index_file = tmp_path / "pipeline" / DERIVED_NAME
    g.typed_index_file = tmp_path / "pipeline" / TYPED_NAME
    g.proposals_file = tmp_path / "pipeline" / "no-proposals.json"
    return g


def _digests(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.md"))}


def _redump_like_deleted_writer(ri, raw: bytes) -> bytes:
    """The write the deleted method performed, reconstructed from the shape
    recorded in backlog #484: split the file on `---`, re-serialise the parsed
    frontmatter with `yaml.dump`, rejoin as
    `f"---\\n{dump}---\\n{body}"`.

    The mutation is the add branch run on a triple the note already carries, so
    it changes nothing semantically — which is the whole point: the old code
    reached `write_text` regardless, and the reassembly alone moved the bytes
    (key reordering, and one newline byte per call). Used only to prove the
    digest comparison in the test below has a trigger; nothing here imports the
    module's own writer, which is deleted.
    """
    parts = raw.decode().split("---", 2)
    assert len(parts) == 3, "fixture note is not frontmatter + body"
    frontmatter = ri.yaml.safe_load(parts[1]) or {}
    relations = frontmatter.setdefault("relations", {})
    existing = relations.setdefault("depends-on", [])
    if "knowledge/d.md" not in existing:          # the already-present triple
        existing.append("knowledge/d.md")
    dump = ri.yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    return f"---\n{dump}---\n{parts[2]}".encode()


# --- clause 1: the deletion route -------------------------------------------

def test_deletion_leaves_no_frontmatter_rewriter(ri):
    """#484 clause 1: `_update_frontmatter` and `add_relation` are gone — the
    route taken was deletion, so the module keeps no method that mutates a
    vault note and no attribute that reaches one."""
    g = ri.RelationsIndexGenerator()
    assert not hasattr(g, "_update_frontmatter")
    assert not hasattr(g, "add_relation")

    source = SCRIPT.read_text()
    # the clause's own check, run from inside the suite: that grep has no match
    assert "_update_frontmatter" not in source
    assert "add_relation" not in source

    defined = {n.name for n in ast.walk(ast.parse(source))
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert not defined & {"_update_frontmatter", "add_relation", "remove_relation"}, (
        f"a relation-writing method is back: {sorted(defined & {'_update_frontmatter', 'add_relation', 'remove_relation'})}"
    )


def test_module_imports_and_is_read_only_by_construction(ri):
    """#484 clause 1 (module still imports) plus a static pin of the boundary
    the deletion is supposed to leave behind: the only file the module writes
    is `self.typed_index_file`. A future `doc_file.write_text(...)` in here fails
    this test even if it comes back under a different name.

    The allowed attribute was `index_file` until #1148, which moved this module's
    write to a file it owns: `index_file` is the derived
    `_pipeline/relations-index.json` that `scripts/memory/rebuild_index.py` owns,
    so a write to it is now exactly as forbidden as a write to a vault note —
    it is the second writer that clobbered the first one every cycle."""
    tree = ast.parse(SCRIPT.read_text())

    # Every way this module can put bytes on disk, and what each one's
    # destination argument has to be. `atomic_write_text` is in here as a bare
    # name because the runtime spy in the `write_targets` fixture patches
    # `app.atomic_io.atomic_write_text` — a `from app.atomic_io import
    # atomic_write_text` at module level binds the function before the spy is
    # installed, so that route is invisible to the spy and only catchable
    # statically. Same reasoning for the os/shutil renames below: they move a
    # file into place without any call the spy or the digest test would see as
    # a write to a note.
    # No exemption here: after #1148 the self-test writes nothing at all outside
    # the temp dir `rebuild()` lands in, so every write call in this module —
    # whoever makes it — has to be the typed index. The reachability asserts
    # after the loop keep the self-test's two sandbox functions off every
    # production path, which is the other half of "`--test` cannot touch
    # `_pipeline`".
    write_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"write_text", "write_bytes", "open"}:
            write_calls.append((node, func.value))
        elif isinstance(func, ast.Name) and func.id == "atomic_write_text":
            write_calls.append((node, node.args[0] if node.args else None))

    assert write_calls, "the module stopped writing anything at all — the index write moved?"
    for call, destination in write_calls:
        ok = isinstance(destination, ast.Attribute) and destination.attr == "typed_index_file"
        assert ok, (
            f"relations_index.py line {call.lineno} writes something other than "
            f"self.typed_index_file; a vault-note writer, or a second writer for "
            f"the derived relations-index.json, is back (#484, #1148)"
        )

    callers = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                callers.setdefault(call.func.id, set()).add(fn.name)
    assert callers.get("_run_tests") == {"main"}, (
        f"_run_tests is called from {sorted(callers.get('_run_tests', []))}, not only "
        "from main(): the write exemption above would then cover a production path (#1148)"
    )
    # Equality, not a subset: an empty caller set would mean the sandbox helper
    # is dead code, which is exactly how the self-test goes back to building a
    # bare generator.
    assert callers.get("_sandbox_generator", set()) == {"_run_tests"}, (
        f"_sandbox_generator is called from {sorted(callers.get('_sandbox_generator', []))}, "
        "not from _run_tests: either the self-test no longer sandboxes its writes, "
        "or something else builds a sandbox (#1148)"
    )

    renamed = sorted({
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"os", "shutil"}
        and node.func.attr in {"replace", "rename", "remove", "unlink", "move", "rmtree"}
    })
    assert not renamed, (
        f"relations_index.py now uses os/shutil {renamed}, which relocates a file "
        f"without a write any guard here covers (#484)"
    )


def test_rebuild_edge_count_is_the_document_relations(ri, generator):
    """#484 clause 1: `--rebuild` still produces the same index. The count is
    the one the fixture's frontmatter implies — 1 related-to, 2 for
    depends-on + its inverse, 2 for superseded-by + its inverse, 1 folded in
    from the legacy `related:` field, 0 for the unknown type."""
    result = generator.rebuild()
    assert result["total_relationships"] == 6
    assert result["stale_documents"] == 1
    assert result["status"] == "rebuilt"

    edges = {(e["source"], e["target"], e["type"]) for e in generator.index_data["edges"]}
    assert edges == {
        ("knowledge/a.md", "knowledge/b.md", "related-to"),
        ("knowledge/c.md", "knowledge/d.md", "depends-on"),
        ("knowledge/d.md", "knowledge/c.md", "required-by"),
        ("knowledge/e.md", "knowledge/f.md", "superseded-by"),
        ("knowledge/f.md", "knowledge/e.md", "supersedes"),
        ("knowledge/g.md", "knowledge/h.md", "related-to"),
    }


def test_rebuild_edge_count_is_stable_across_calls(generator):
    """#484 clause 1, the part that matters for churn: rebuilding twice yields
    the same edge count, so the index itself is not a growth source."""
    first = generator.rebuild()["total_relationships"]
    second = generator.rebuild()["total_relationships"]
    assert first == second == 6


# --- clauses 2 and 3, pinned through the deletion ---------------------------
# The add path no longer exists, so the byte-identity these clauses describe is
# asserted over the whole module: nothing in it can take a note from 101 to
# 102 bytes. A repeat rebuild is the closest live equivalent of "call the add
# path twice with a triple already present" — the relation is already in the
# fixture's frontmatter, and the note must not move.

def test_rebuild_leaves_every_note_byte_identical(ri, generator, vault):
    """#484 clauses 2 + 3: the fixture note whose relation is already present,
    with non-alphabetical frontmatter keys and a body preceded by a blank
    line, is unchanged — byte for byte — by any path the module still has.
    Run twice, because the old writer's damage was per-call growth.

    The byte-digest comparison this test rests on is proved able to fail before
    it is trusted to pass: the first block re-applies the deleted writer's
    exact reassembly to one fixture note and requires the digest to move. A
    fixture that could not tell a re-dumped note from an untouched one would
    make the real assertion vacuous, and fails here instead.
    """
    churned = vault / "knowledge" / "c.md"
    original = churned.read_bytes()
    churned.write_bytes(_redump_like_deleted_writer(ri, original))
    assert hashlib.sha256(churned.read_bytes()).hexdigest() != hashlib.sha256(original).hexdigest(), (
        "the fixture survives a yaml frontmatter round-trip unchanged, so the "
        "byte-identity assertion below proves nothing — give the note "
        "non-alphabetical keys and a quoted value"
    )
    churned.write_bytes(original)
    assert _digests(vault)[churned.relative_to(vault).as_posix()] == hashlib.sha256(original).hexdigest()

    before = _digests(vault)
    generator.rebuild()
    generator.rebuild()
    after = _digests(vault)
    assert before == after, (
        "rebuild() rewrote a vault note: "
        + ", ".join(k for k in before if before[k] != after[k])
    )


def test_no_write_target_falls_under_the_vault_or_the_live_index(generator, vault, write_targets):
    """#484 clause 4: no `write_text` / `atomic_write_text` target lands under
    `self.vault`, or at the live `_pipeline/relations-index.json`.

    The live derived index is now forbidden twice over — as a vault note is
    forbidden, and because #1148 gave it a single owner in
    `scripts/memory/rebuild_index.py`. The live *typed* index is checked here too
    so a fixture path that failed to redirect it cannot pass this file."""
    write_targets.clear()
    generator.rebuild()
    assert write_targets, "nothing was written at all — the spy is not on the write path"
    vault_resolved = vault.resolve()
    forbidden = {LIVE_INDEX.resolve(), LIVE_TYPED_INDEX.resolve()}
    for target in write_targets:
        t = target.resolve()
        assert vault_resolved not in t.parents and t != vault_resolved, (
            f"rebuild() wrote a vault note: {t} (#484)"
        )
        assert t not in forbidden, (
            f"rebuild() wrote live pipeline state outside its own tmp fixture: {t} (#484, #1148)"
        )


# --- clause 5: the nightly path ---------------------------------------------

# The four names `nightly_extraction.py` imports by bare name rather than as a
# package, line numbers as of #755: `from content_hasher import …` at :35 — one
# directory up, reached by the `scripts/memory` insert at :29, which #755 made
# `__file__`-derived — and `from fact_extractor import …` at :32/:61,
# `from relations_index import …` at :62, `from profile_generator import …` at :63.
# A `sys.modules` hit under any of them is used *before* `sys.path` is consulted,
# so injecting a path is not enough to decide which file the nightly binds — the
# cache has to be cleared first.
SIBLING_IMPORTS = ("content_hasher", "fact_extractor", "relations_index", "profile_generator")


@pytest.fixture
def nightly_module():
    """`nightly_extraction.py` actually imported, for the clause-5 seam.

    The file is a script, not an importable package module: it binds
    `RelationsIndexGenerator` with a bare `from relations_index import …` at :62,
    which resolves only because running it as a script puts its own directory on
    `sys.path`. `importlib` does not do that for a file loaded by path, so this
    fixture injects the directory first — and then evicts the four sibling names
    from `sys.modules`, because a cache hit under one of those names wins over the
    injected path and would bind a module from some other checkout.

    The eviction is what the first version of this fixture lacked, and it made
    the test below order-dependent: alone it passed (9 passed, 0.7 s), in a full
    `pytest tests/` it went red. The other half is
    `tests/test_extraction_single_instance.py:29`, which hardcodes
    `/home/alansrobotlab/lloyd/scripts/memory/next-gen-memory` and whose `ne`
    fixture (:48-54) imports `nightly_extraction` from that absolute path — so by
    the time this file runs, `sys.modules["relations_index"]` is already bound to
    the *live* checkout. Under `automod_gate`, which runs pytest from a worktree,
    the nightly then imported that module and the origin assert failed naming both
    paths. Reproduce the red without a gate, in one command:
    `pytest tests/test_extraction_single_instance.py tests/test_relations_index_read_only.py`.

    Teardown restores `sys.path` to the list it found and each sibling name to
    whatever was cached before (or to absent, if nothing was), so a later test
    still gets exactly the module it would have got had this fixture never run.
    Anything else the exec pulled off the injected directory goes with it. Until
    #755 the nightly also built two of its own `sys.path` entries from outside this
    tree — a hardcoded `~/lloyd/scripts/memory` for `content_hasher`, and an insert
    of `~/obsidian/agents/memory/scripts/next-gen-memory`, a path that does not
    exist here — either of which lands *ahead* of this fixture's entry and can bind
    a worktree run to live-checkout modules. #755 made every entry
    `__file__`-derived and dropped the dead one;
    `tests/test_nightly_extraction_import_paths.py` pins that. The eviction above is
    still what carries this fixture, because a `sys.modules` hit from
    `tests/test_extraction_single_instance.py` beats any injected path.
    """
    injected = str(NIGHTLY.parent)
    injected_dir = Path(injected).resolve()
    before_path = list(sys.path)
    before_names = set(sys.modules)
    cached = {name: sys.modules[name] for name in SIBLING_IMPORTS if name in sys.modules}
    for name in SIBLING_IMPORTS:
        sys.modules.pop(name, None)
    sys.path.insert(0, injected)
    spec = importlib.util.spec_from_file_location("nightly_extraction_under_test", NIGHTLY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    # Only the modules that came off the injected directory. A plain
    # set-difference would also evict stdlib modules the exec happened to pull in
    # first, which is churn this test has no business causing.
    for name in set(sys.modules) - before_names:
        loaded = getattr(sys.modules[name], "__file__", None)
        if loaded and Path(loaded).resolve().parent == injected_dir:
            sys.modules.pop(name, None)
    for name in SIBLING_IMPORTS:
        if name in cached:
            sys.modules[name] = cached[name]
        else:
            sys.modules.pop(name, None)
    sys.path[:] = before_path


def test_nightly_reaches_the_generator_only_through_rebuild(nightly_module, ri):
    """#484 clause 5, both halves.

    Static half: `nightly_extraction.py` calls `rebuild()` and nothing else on
    the relations generator — the AST walk is the grep that cannot go stale on a
    line number.

    Executed half: the AST alone would also pass on a file that no longer
    imports, so the same test loads the script. An import break, a renamed
    symbol, or the `from relations_index import …` at :42 resolving to a
    different file than the one under test all fail here instead of at 02:00 in
    the nightly job. The last assert is the one that matters for #484: the class
    the nightly binds must be the class this suite just deleted the writer from —
    otherwise the deletion protects a module nothing runs.
    """
    bound = nightly_module.RelationsIndexGenerator
    origin = Path(inspect.getsourcefile(bound)).resolve()
    assert origin == SCRIPT.resolve(), (
        f"nightly_extraction.py bound RelationsIndexGenerator from {origin}, not the "
        f"file this suite deleted the writer out of ({SCRIPT}) — clause 5's read-only "
        "guarantee would be about the wrong module"
    )
    assert hasattr(bound, "rebuild")
    # the deletion is visible from the nightly's own binding, not just from the
    # copy this module loaded: two importlib loads of one path make two distinct
    # class objects, so `is` against the `ri` fixture's class would be false even
    # when both came from this file — origin is the comparison that means it.
    assert not hasattr(bound, "add_relation")
    assert not hasattr(bound, "_update_frontmatter")
    assert hasattr(ri.RelationsIndexGenerator, "rebuild")

    tree = ast.parse(NIGHTLY.read_text())
    method_calls = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "rel_generator"):
            method_calls.add(node.func.attr)
    assert method_calls == {"rebuild"}, (
        f"nightly_extraction.py now calls {sorted(method_calls)} on the relations "
        f"generator; anything past rebuild() is a candidate vault writer (#484)"
    )


def test_write_text_during_rebuild_records_only_the_typed_index_file(generator, write_targets):
    """#484 clause 5, as amended by #1148: with `write_text` monkeypatched, a
    rebuild records only `self.typed_index_file` — no vault note, and no write
    to `self.index_file`, the derived index `rebuild_index.py` owns. Before
    #1148 this recorded `self.index_file`, which is how the two writers stayed
    invisible to each other: each one's own write looked legitimate here."""
    write_targets.clear()
    generator.rebuild()
    recorded = sorted(t.resolve() for t in write_targets)
    assert recorded == [generator.typed_index_file.resolve()], (
        f"rebuild() wrote {recorded}, expected only {generator.typed_index_file}"
    )
    assert generator.index_file.resolve() not in recorded, (
        "rebuild() wrote the derived relations-index.json (#1148: one owner)"
    )


def test_rebuild_merges_approved_proposals_without_touching_the_vault(
    ri, generator, vault, tmp_path, write_targets
):
    """The one input rebuild() is expected to *add* edges from is the proposals
    file — conversation-derived relations, the caller shape this round deleted.
    It is a JSON sidecar, not a note: merging an approved proposal still writes
    only the index, and the notes stay byte-identical."""
    proposals = tmp_path / "pipeline" / "conversation-relation-proposals.json"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text(json.dumps({"proposals": [{
        "source": "knowledge/a.md", "target": "knowledge/c.md",
        "type": "related-to", "status": "approved",
        "reason": "said so in conversation", "confidence": 0.9,
    }]}, indent=2))
    generator.proposals_file = proposals

    before = _digests(vault)
    write_targets.clear()
    result = generator.rebuild()

    assert result["conversation_proposals_merged"] == 1
    assert result["total_relationships"] == 7
    assert _digests(vault) == before
    assert write_targets == [generator.typed_index_file], (
        "merging proposals wrote somewhere other than the typed index (#1148)"
    )


# --- #1148 clauses 1-5: one file, one owner, one schema ----------------------

def _gen_at(tmp_path, ri, derived_payload=None, typed_payload=None):
    """A generator with every path under ``tmp_path``, optionally seeded with the
    two index shapes. Separate from the ``generator`` fixture because these
    clauses start from file contents rather than from a rebuild."""
    g = ri.RelationsIndexGenerator()
    g.vault = tmp_path / "vault"
    g.index_file = tmp_path / "pipeline" / DERIVED_NAME
    g.typed_index_file = tmp_path / "pipeline" / TYPED_NAME
    g.proposals_file = tmp_path / "pipeline" / "no-proposals.json"
    g.vault.mkdir(parents=True, exist_ok=True)
    g.index_file.parent.mkdir(parents=True, exist_ok=True)
    if derived_payload is not None:
        g.index_file.write_text(json.dumps(derived_payload))
    if typed_payload is not None:
        g.typed_index_file.write_text(json.dumps(typed_payload))
    return g


@pytest.fixture
def scratch_home(tmp_path):
    """A fake ``$HOME`` laid out like the live box — empty vault, and a
    ``~/lloyd/_pipeline/relations-index.json`` holding the derived shape with a
    sentinel in it. Every path the CLI resolves off ``Path.home()`` lands here,
    so "production state did not move" becomes "these bytes did not change"."""
    home = tmp_path / "home"
    (home / "obsidian").mkdir(parents=True)
    (home / "lloyd" / "_pipeline").mkdir(parents=True)
    (home / "lloyd" / "_pipeline" / DERIVED_NAME).write_text(json.dumps(DERIVED_SHAPED))
    return home


def _run_cli(args, home, tmp_path):
    """The module's CLI in a subprocess, under the scratch HOME and a TMPDIR
    inside ``tmp_path`` so the self-test's own temp dir is visible to the caller."""
    tmp = tmp_path / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(home), TMPDIR=str(tmp))
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, env=env, timeout=300, cwd=str(ROOT),
    )


def test_test_flag_leaves_the_live_index_alone(scratch_home, tmp_path):
    """Clause 1: ``relations_index.py --test`` leaves ``relations-index.json``
    byte-identical, because Test 4's generator is pointed at a temp dir.

    It used to build a bare ``RelationsIndexGenerator()`` — whose default
    ``index_file`` is the real ``_pipeline/relations-index.json`` — and call
    ``rebuild()``, so the developer's smoke test scanned ``~/obsidian`` and
    replaced the production index with an ``edges`` file. The whole
    ``_pipeline`` directory is compared, not just the derived file: any file
    appearing there means the self-test wrote outside the temp dir it prints."""
    pipeline = scratch_home / "lloyd" / "_pipeline"
    before = {p.name: p.read_bytes() for p in pipeline.iterdir()}
    assert before == {DERIVED_NAME: json.dumps(DERIVED_SHAPED).encode()}, (
        "the fixture did not seed a sentinel-bearing derived index; the byte "
        "comparison below would pass on an empty directory"
    )

    proc = _run_cli(["--test"], scratch_home, tmp_path)
    assert proc.returncode == 0, (
        f"--test exited {proc.returncode}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    )
    assert "All tests passed" in proc.stdout, "--test did not run to completion"

    after = {p.name: p.read_bytes() for p in pipeline.iterdir()}
    assert after == before, (
        f"`--test` changed _pipeline/ from {sorted(before)} to {sorted(after)}; "
        "the self-test writes production state again (#1148)"
    )
    sandbox = [ln for ln in proc.stdout.splitlines() if ln.startswith("Sandbox:")]
    assert sandbox, "--test never reported the temp dir it works in"
    reported = Path(sandbox[0].split("Sandbox:", 1)[1].strip())
    assert str(reported).startswith(str(tmp_path / "tmp")), (
        f"--test reported a sandbox outside TMPDIR: {reported} (#1148 wants a temp dir)"
    )


def test_rebuild_in_the_scratch_home_writes_only_the_typed_index(scratch_home, tmp_path):
    """Positive control for the clause-1 assert above, and the write half of
    #1148 at CLI level. Same scratch HOME, same subprocess, ``--rebuild``: if the
    HOME redirect did not work, "nothing in _pipeline changed" would be vacuous —
    this is the run that must write, and it must write only the file this module
    owns."""
    pipeline = scratch_home / "lloyd" / "_pipeline"
    derived = pipeline / DERIVED_NAME
    before = derived.read_bytes()

    proc = _run_cli(["--rebuild"], scratch_home, tmp_path)
    assert proc.returncode == 0, (
        f"--rebuild exited {proc.returncode}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    )
    assert (pipeline / TYPED_NAME).exists(), (
        "--rebuild wrote no typed index: the subprocess is not running under the "
        "scratch HOME, which would make every 'production did not move' assert in "
        "this file vacuous"
    )
    assert derived.read_bytes() == before, (
        "--rebuild moved the derived index it no longer owns (#1148)"
    )


def test_get_relations_for_doc_reads_the_relationships_key(ri, tmp_path):
    """Clause 2: a generator whose loaded index uses the ``relationships`` key
    returns that document's rows and raises nothing.

    That is the shape ``scripts/memory/rebuild_index.py`` leaves behind — 341,373
    rows over 3,524 docs on 2026-09-21 — and the query path used to read
    ``self.index_data["edges"]``, so against exactly this file ``--query`` raised
    ``KeyError: 'edges'`` (``relations_index.py:553``, via ``main()``) for as long
    as the pipeline had been running. The third fixture row names neither of the
    queried document's paths, so an unfiltered return fails here too."""
    g = _gen_at(tmp_path, ri, derived_payload=DERIVED_SHAPED)
    g.load_index()
    rows = g.get_relations_for_doc("memory/2026-09-12.md")
    assert [(r["source"], r["target"], r["type"]) for r in rows] == [
        ("memory/2026-09-12.md", "knowledge/a.md", "wiki-link"),
        ("knowledge/b.md", "memory/2026-09-12.md", "tag-cluster"),
    ], f"the queried document's derived rows came back wrong: {rows}"


def test_get_relations_for_doc_performs_no_write(ri, tmp_path, monkeypatch, write_targets):
    """Clause 3: ``get_relations_for_doc`` performs no write — called with empty
    ``edges`` it does not rebuild and leaves the index file untouched.

    The body was ``if not self.index_data["edges"]: self.rebuild()``, and a
    ``relationships``-shaped file loads as exactly that: empty ``edges``. So a
    *query* triggered a full vault scan and rewrote the 105 MB live index — and
    with no ``load_index()`` before it, which is what ``--test`` built at ``:610``,
    every call did. ``rebuild`` is trapped, both files are compared by bytes, and
    the write spy is required to stay empty."""
    g = _gen_at(tmp_path, ri, derived_payload=DERIVED_SHAPED,
                typed_payload={"edges": [], "stale": [], "built_at": None})
    g.load_index()
    before = {p.name: p.read_bytes() for p in g.index_file.parent.iterdir()}

    def trap_rebuild():
        raise AssertionError(
            "get_relations_for_doc called rebuild(): a read must not rebuild (#1148)"
        )
    monkeypatch.setattr(g, "rebuild", trap_rebuild)

    write_targets.clear()
    rows = g.get_relations_for_doc("memory/2026-09-12.md")
    assert rows, (
        "no rows came back, so the no-write assertions below would prove nothing"
    )
    assert write_targets == [], (
        f"reading relations wrote {write_targets} (#1148: a read must not write state)"
    )
    assert {p.name: p.read_bytes() for p in g.index_file.parent.iterdir()} == before

    # The other half of the clause: nothing loaded at all. This is the shape the
    # self-test built, and the one that used to trigger the rebuild.
    empty = _gen_at(tmp_path / "empty", ri)
    write_targets.clear()
    assert empty.get_relations_for_doc("memory/2026-09-12.md") == []
    assert write_targets == [], f"an empty read wrote {write_targets} (#1148)"
    assert not empty.typed_index_file.exists() and not empty.index_file.exists(), (
        "the read created an index file it does not own (#1148)"
    )


def test_no_arg_summary_reports_the_rows_in_the_derived_file(scratch_home, tmp_path):
    """Clause 4: the module's no-arg summary reports the row count present in the
    ``relationships``-shaped file rather than printing ``Index loaded: 0 edges``.

    The fixture file carries 3 rows; the live file carried 341,373 while the
    summary printed 0, because it read only the ``edges`` key. Asserted through
    the real CLI, since printing the wrong thing is the failure."""
    proc = _run_cli([], scratch_home, tmp_path)
    assert proc.returncode == 0, (
        f"no-arg run exited {proc.returncode}\n{proc.stdout}\n{proc.stderr[-1500:]}"
    )
    assert "Index loaded: 0 edges" not in proc.stdout, (
        f"the summary is back to reading one key out of a two-schema file:\n{proc.stdout}"
    )
    found = re.search(rf"Derived rows \([^)]*\): {len(DERIVED_SHAPED['relationships'])}\b",
                      proc.stdout)
    assert found, (
        f"the summary never reported the {len(DERIVED_SHAPED['relationships'])} rows "
        f"the derived file holds:\n{proc.stdout}"
    )


def _production_py_files():
    """Every tracked ``.py`` outside ``tests/`` — the corpus a claim about
    "exactly one writer" has to be made over. Tracked files, via
    ``git ls-files``: a walk would count scratch trees and miss the point, and
    the two index writers are both tracked."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "*.py"], cwd=str(ROOT),
        capture_output=True, text=True, timeout=120,
    )
    assert listing.returncode == 0, f"git ls-files failed: {listing.stderr[:400]}"
    files = [ROOT / p for p in listing.stdout.split("\0") if p and not p.startswith("tests/")]
    assert len(files) > 100, f"only {len(files)} files scanned; the corpus is not the checkout"
    return files


def _bind_key(node):
    """The name a write target or assignment binds, for the two shapes the
    checkout's writers actually use: a module-level constant, or ``self.<attr>``."""
    if isinstance(node, ast.Name):
        return node.id
    if (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == "self"):
        return f"self.{node.attr}"
    return None


def _written_index_names(source, strict=False):
    """Which of the two index filenames this module can put bytes on.

    Follows one-level binding chains (``X = Path(...) / "name.json"``,
    ``self.attr = X``, ``X.write_text(...)``). ``strict`` also reports a write
    whose target binds to nothing: a one-file probe can be answered with a plan,
    a checkout-wide scan cannot, because the blind spot *is* the second writer."""
    tree = ast.parse(source)
    binds = {}
    for node in ast.walk(tree):
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None:
            continue
        for tgt in targets:
            key = _bind_key(tgt)
            if key and key not in binds:
                binds[key] = value

    def json_names(node, depth=0):
        if depth > 8:
            return set()
        found = {c.value for c in ast.walk(node)
                 if isinstance(c, ast.Constant)
                 and isinstance(c.value, str) and c.value.endswith(".json")}
        key = _bind_key(node)
        if key and key in binds:
            found |= json_names(binds[key], depth + 1)
        return found

    names, opaque = set(), []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"write_text", "write_bytes", "open"}:
            target = func.value
        elif isinstance(func, ast.Name) and func.id == "atomic_write_text" and node.args:
            target = node.args[0]
        else:
            continue
        reachable = json_names(target)
        if not reachable and strict:
            opaque.append(node.lineno)
        names |= reachable & set(INDEX_NAMES)
    if opaque:
        raise AssertionError(
            f"write targets at line(s) {opaque} resolve to no *.json name; the "
            "one-writer claim cannot be checked past them (#1148)"
        )
    return names


def test_exactly_one_writer_per_index():
    """Clause 5: exactly one script in the checkout writes
    ``_pipeline/relations-index.json``, and the other writer emits to a distinct
    path the module's own readers read — so a Data Pipeline cycle, which runs both
    scripts in one task in a fixed order, cannot clobber either.

    Before #1148 both scripts wrote the same path with different schemas and
    ``nightly_extraction.py`` carried a third pointer to it as a constant; the
    scan is over every tracked ``.py``, because the claim is about the checkout
    and a grep scoped to ``scripts/`` is what let the second writer hide."""
    writers = {name: set() for name in INDEX_NAMES}
    for path in _production_py_files():
        source = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(ROOT).as_posix()
        # The two owners are checked fail-closed: an unresolvable write in either
        # would let a second writer of the derived path through unreported.
        writers_seen = _written_index_names(source, strict=rel in _INDEX_OWNER_PATHS)
        for name in writers_seen:
            writers[name].add(rel)

    assert writers[DERIVED_NAME] == {"scripts/memory/rebuild_index.py"}, (
        f"the derived index has writers {sorted(writers[DERIVED_NAME])}; #1148 wants "
        "exactly scripts/memory/rebuild_index.py"
    )
    assert writers[TYPED_NAME] == {"scripts/memory/next-gen-memory/relations_index.py"}, (
        f"the typed index has writers {sorted(writers[TYPED_NAME])}; #1148 wants "
        "exactly the module that reads it"
    )


def test_index_owner_paths_are_two_distinct_files(ri):
    """The other half of clause 5's wording: the two paths are distinct, and the
    module's readers read both — asserted through the loaded attributes rather
    than a string comparison, so a rename in the module moves this test too."""
    assert ri.DERIVED_INDEX_FILE != ri.TYPED_INDEX_FILE
    assert ri.DERIVED_INDEX_FILE.name == DERIVED_NAME
    assert ri.TYPED_INDEX_FILE.name == TYPED_NAME
    g = ri.RelationsIndexGenerator()
    assert g.index_file == ri.DERIVED_INDEX_FILE, (
        "the module's read target is no longer the derived file: clause 2's read "
        "and this clause are about different files (#1148)"
    )
    assert g.typed_index_file == ri.TYPED_INDEX_FILE
    assert g.typed_index_file != g.index_file


def test_the_writer_scan_detects_a_planted_second_writer():
    """Positive control for clause 5: the scan has to be able to see a write of
    the derived path made through the same ``self.attr`` chain the real second
    writer used. Without it, ``writers[derived] == {one file}`` would pass on a
    checkout with two writers, which is the exact state #1148 was filed about."""
    planted = (
        "from pathlib import Path\n"
        "DERIVED = Path('/tmp') / '_pipeline' / 'relations-index.json'\n"
        "class Writer:\n"
        "    def __init__(self):\n"
        "        self.index_file = DERIVED\n"
        "    def run(self):\n"
        "        self.index_file.write_text('{}')\n"
    )
    assert _written_index_names(planted) == {DERIVED_NAME}
    assert _written_index_names("from pathlib import Path\nPath('/x').read_text()\n") == set()


@pytest.fixture
def derived_writer(request, scratch_home):
    """`rebuild_index.py` imported as a module, with its scan root and its output
    path redirected into the scratch HOME. Only the two module globals the writer
    reads are patched; the function body is the real scheduled-task Step 2."""
    spec = importlib.util.spec_from_file_location(
        f"rebuild_index_{request.node.name.replace('[', '_').replace(']', '_')}",
        REBUILD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.VAULT = scratch_home / "obsidian"
    mod.RELATIONS_INDEX = scratch_home / "lloyd" / "_pipeline" / DERIVED_NAME
    yield mod
    sys.modules.pop(spec.name, None)


def test_the_scheduled_task_cycle_leaves_both_indexes_readable(scratch_home, tmp_path,
                                                               derived_writer, ri):
    """The composition the clobber lived in, run as one sequence: task #24 Data
    Pipeline (`skills/autonomy-data-pipeline/SKILL.md` — `nightly_extraction.py`
    at :126, which calls ``RelationsIndexGenerator.rebuild()``, then
    ``rebuild_index.py`` at :261) executes the two writers as two steps in a fixed
    order. Before #1148 both wrote the one path, so step 2 replaced step 1's file
    and the module's own CLI could not read the result.

    Step 1 runs the real CLI as a subprocess under the scratch HOME; step 2 calls
    the real ``rebuild_relations_index()`` with only its ``VAULT`` and
    ``RELATIONS_INDEX`` globals redirected into the same scratch ``_pipeline``.
    Three things must hold afterwards, none of which was true before: step 2 left
    step 1's file byte-identical, each file carries exactly one schema, and one
    ``get_relations_for_doc`` call answers with a typed edge from step 1 *and* a
    co-occurrence row from step 2.
    """
    notes = scratch_home / "obsidian" / "knowledge"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "alpha.md").write_text(
        "---\ntype: knowledge-note\nsegment: knowledge\ntitle: Alpha\n"
        "relations:\n  depends-on:\n    - knowledge/beta.md\n---\n\n# Alpha\n\n"
        "See [[gamma]] too.\n"
    )
    (notes / "beta.md").write_text(
        "---\ntype: knowledge-note\nsegment: knowledge\ntitle: Beta\n---\n\n# Beta\n\n"
        "Also [[gamma]].\n"
    )

    # Step 1 — nightly_extraction's call, via the CLI, in its own process.
    step1 = _run_cli(["--rebuild"], scratch_home, tmp_path)
    assert step1.returncode == 0, f"step 1 failed: {step1.stderr[-800:]}"
    typed_path = scratch_home / "lloyd" / "_pipeline" / TYPED_NAME
    typed_after_step1 = typed_path.read_bytes()
    typed_rows = json.loads(typed_after_step1)["edges"]
    assert len(typed_rows) == 2, (
        f"step 1 produced {len(typed_rows)} typed edges from one depends-on; "
        "expected 2 (the edge and its required-by inverse), so the reader "
        "assertions below would be testing an index nobody built"
    )

    # Step 2 — the script that used to overwrite it.
    result = derived_writer.rebuild_relations_index()
    assert result["status"] == "rebuilt" and result["total_relationships"] == 1, (
        f"step 2 wrote {result}; expected one wiki-link relationship from two notes "
        "linking the same target — with no rows the clobber check is vacuous"
    )
    derived_path = scratch_home / "lloyd" / "_pipeline" / DERIVED_NAME
    assert typed_path.read_bytes() == typed_after_step1, (
        "step 2 rewrote the typed index: the two writers are back on one path (#1148)"
    )

    derived = json.loads(derived_path.read_text())
    assert set(derived) >= {"relationships", "total_relationships", "documents_indexed"}
    assert "edges" not in derived, "step 2 wrote an `edges` key: two schemas in one file (#1148)"
    assert "relationships" not in json.loads(typed_path.read_text()), (
        "step 1 wrote a `relationships` key: two schemas in one file (#1148)"
    )

    # The end state one cycle leaves behind, read by the module's own reader: a
    # document that both indexes know about must answer with rows from both files.
    # Pre-fix this call raised KeyError: 'edges' against exactly this shape.
    reader = _gen_at(scratch_home / "reader", ri)
    reader.index_file = derived_path
    reader.typed_index_file = typed_path
    reader.load_index()
    kinds = {r.get("type") for r in reader.get_relations_for_doc("knowledge/alpha.md")}
    assert {"depends-on", "required-by", "wiki-link"} <= kinds, (
        f"after a full cycle the reader sees {sorted(kinds)}; it must see the typed "
        "relations from step 1 and the co-occurrence row from step 2 (#1148)"
    )
