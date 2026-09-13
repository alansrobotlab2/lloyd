"""Backlog #484 — RelationsIndexGenerator must never write a vault source note.

`relations_index.py` used to carry a vault-note rewriter:
``_update_frontmatter()``, reached only by ``add_relation()``, rebuilt a note's
whole frontmatter with ``yaml.dump`` and rejoined the body with
``f"---\\n{dump}---\\n{body}"``. That round-trip reorders keys, drops quoting
and appends one newline byte per call, so it changed bytes without changing
content — and the nightly content-hash gate
(`scripts/memory/content_hasher.py::get_changed_files`, sha256 of bytes) reads
a byte change as new content and re-extracts the note. That is the
"byte-churn family" the 2026-08-31 and 2026-09-03 vault-maintenance logs
recorded (``memory/2026-02-22.md`` reprocessed eight times in one day, growing
a blank line each time). Both functions had no caller left after f36c522, so
this round deleted them (clause 1) rather than leaving an unguarded writer
pointed at the gate.

Each test below names the clause it pins. Nothing here touches ``~/obsidian``
or the live ``~/lloyd/_pipeline/relations-index.json``: the generator's
``vault``, ``index_file`` and ``proposals_file`` are all redirected into
``tmp_path``, and the recorded write targets are asserted, not assumed.
"""
import ast
import hashlib
import importlib.util
import json
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


@pytest.fixture
def generator(ri, vault, tmp_path):
    """A generator pointed entirely at tmp_path — tmp vault, tmp index, no
    proposals file. The live index is ``~/lloyd/_pipeline/relations-index.json``
    and is asserted untouched by every test that rebuilds."""
    g = ri.RelationsIndexGenerator()
    g.vault = vault
    g.index_file = tmp_path / "pipeline" / "relations-index.json"
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
    is `self.index_file`. A future `doc_file.write_text(...)` in here fails
    this test even if it comes back under a different name."""
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
        ok = isinstance(destination, ast.Attribute) and destination.attr == "index_file"
        assert ok, (
            f"relations_index.py line {call.lineno} writes something other than "
            f"self.index_file; a vault-note writer is back (#484)"
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
    `self.vault`, or at the live `_pipeline/relations-index.json`."""
    write_targets.clear()
    generator.rebuild()
    assert write_targets, "nothing was written at all — the spy is not on the write path"
    vault_resolved = vault.resolve()
    for target in write_targets:
        t = target.resolve()
        assert vault_resolved not in t.parents and t != vault_resolved, (
            f"rebuild() wrote a vault note: {t} (#484)"
        )
        assert t != LIVE_INDEX.resolve(), f"rebuild() wrote the live index: {t} (#484)"


# --- clause 5: the nightly path ---------------------------------------------

def test_nightly_reaches_the_generator_only_through_rebuild():
    """#484 clause 5: `nightly_extraction.py` touches RelationsIndexGenerator
    only through `rebuild()`. Static, because the import crosses a sys.path
    seam the code graph is blind to; the AST walk is the grep that cannot go
    stale on a line number."""
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


def test_write_text_during_rebuild_records_only_the_index_file(generator, write_targets):
    """#484 clause 5: with `write_text` monkeypatched, a rebuild records only
    `self.index_file` — no vault note. The generator's index is the tmp one the
    fixture set, so this also cannot write the live index from under the run."""
    write_targets.clear()
    generator.rebuild()
    recorded = sorted(t.resolve() for t in write_targets)
    assert recorded == [generator.index_file.resolve()], (
        f"rebuild() wrote {recorded}, expected only {generator.index_file}"
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
    assert write_targets == [generator.index_file]
