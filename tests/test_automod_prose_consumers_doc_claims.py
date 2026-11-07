"""Prose claims in `scripts/automod/prose_consumers.py`, checked against the tree.

The doc-claim surface for #3538's rung. What the module says about OTHER files goes
stale first: it names the migration tool's instruction/record ruling, the resolver's
exemption conventions, and two vault files whose contents the matcher's whole
precision argument rests on. The matcher's BEHAVIOUR is pinned in
`test_automod_prose_consumers.py`; this file is about the sentences around it, and
about the one failure class the item itself flags — #3309's "a module that retypes
another module's rule instead of importing it, so the two drift and each keeps
reading as green".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VAULT = Path.home() / "obsidian"


@pytest.fixture(scope="module")
def module_text() -> str:
    path = ROOT / "scripts/automod/prose_consumers.py"
    assert path.is_file(), "prose_consumers.py is gone; every claim here is stale"
    return path.read_text(encoding="utf-8")


# ── the corpus ruling is the migration tool's, not this module's opinion ─────

def test_the_instruction_corpus_is_the_migration_tools_scopes_narrowed(module_text):
    """`rewrite_vault_data_paths.py` is the file that RULES which vault files are
    instructions, because it is the one that rewrites them when a root moves. This
    rung's corpus is that ruling narrowed, and the narrowing has to be a stated
    decision: `autonomy/*.md` verbatim, `skills/**/SKILL.md` instead of the tool's
    `skills/**/*.md`, because the loader injects the SKILL.md and a sidecar is only
    read by whoever opens it. If the two drift apart silently, the tool and the rung
    disagree about whether a broken command is a problem, and the disagreement is
    invisible — one of them logs and moves on."""
    from scripts.maintenance import rewrite_vault_data_paths as RW
    from scripts.automod import prose_consumers as PC

    assert PC.INSTRUCTION_GLOBS[0] == "autonomy/*.md" == RW.SCOPES[0], (
        f"task files: this rung searches {PC.INSTRUCTION_GLOBS[0]!r} and the "
        f"migration tool rewrites {RW.SCOPES[0]!r}")
    skills = PC.INSTRUCTION_GLOBS[1]
    assert skills == "skills/**/SKILL.md", PC.INSTRUCTION_GLOBS
    assert "skills/**/*.md" in RW.SCOPES, RW.SCOPES
    assert "sidecar" in module_text, (
        "the narrowing to SKILL.md must be said out loud, or the next reader "
        "assumes the tool's scope and thinks this one is a typo")
    assert "rewrite_vault_data_paths.py" in module_text, (
        "the module must name where its corpus split comes from")


def test_the_record_corpus_is_the_record_of_what_was_true(module_text):
    """The other half of the same ruling: `backlog`, `memory`, `knowledge` are the
    record of what was true when written, so a hit there is counted and not listed.
    `projects/` is in that list and NOT in this corpus — an intentional omission
    (#3538 counts the record corpus but only the directories the item named), so the
    module has to say which way it went or it reads as an oversight."""
    from scripts.automod import prose_consumers as PC

    assert {"backlog", "knowledge", "memory"} <= set(PC.RECORD_DIRS), PC.RECORD_DIRS
    assert "projects" not in PC.RECORD_DIRS
    assert "projects" in module_text, (
        "`projects/` is in the migration tool's record list and not in this "
        "corpus; the module has to state that")


def test_the_absence_guard_is_imported_and_never_retyped(module_text):
    """#3309's exact failure. The rung needs a regex for "this line is talking
    ABOUT the path, not calling it"; it could carry its own, and then the nightly
    sweep and the gate disagree about the same sentence in the same note while both
    read as green. So: the resolver owns the rule, this module imports it, and the
    giveaway artefacts of a copy — a second compiled regex, the marker spelled out —
    are what this test refuses."""
    from scripts.maintenance import referential_integrity as RI
    assert hasattr(RI, "ABSENCE_PROSE"), (
        "the resolver no longer exports the guard publicly; import it from "
        "wherever it lives instead of copying it")
    assert "referential_integrity" in module_text, (
        "the exemption conventions have to come from the resolver that invented them")
    assert not re.search(r"_?ABSENCE_PROSE\w* = re\.compile", module_text), (
        "prose_consumers.py compiles its own absence-prose regex: that is the "
        "duplicate copy this test exists to refuse")
    # Prose may NAME the marker — the module's whole point is that a line
    # mentioning the marker is a mention. What it may not do is define a second
    # one, because then two markers exist and only one of them is honoured.
    assert not re.search(r"EXEMPT_MARKER\s*=\s*[\"']", module_text), (
        "prose_consumers.py defines its own exemption marker; the resolver's is "
        "the one the sweep honours")
    assert "EXEMPT_MARKER" in module_text, (
        "the rung has to honour the resolver's marker, not just its regex")


# ── the vault files the precision argument rests on ─────────────────────────

def test_the_named_live_lookalike_still_names_the_live_root(module_text):
    """The module's headline example is a real file: `skills/kg-mention-classifier/
    SKILL.md` cites the store under `~/lloyd-data/`, and the whole root-anchoring
    rule exists so that a retired `~/lloyd/` path never reports it. Cited by path
    and NOT by line number on purpose: SKILL.md files are edited by their own
    skills, and a pinned line number would leave this test able to answer only by
    editing the file it is checking."""
    from scripts.automod import prose_consumers as PC

    rel = "skills/kg-mention-classifier/SKILL.md"
    assert f"`{rel}`" in module_text, (
        "the module names a specific file as its non-consumer; if the example "
        "moved, the docstring and this test move together")
    f = VAULT / rel
    if not f.exists():
        pytest.skip(f"{rel} is gone from the vault; the example needs a new file")
    text = f.read_text(encoding="utf-8", errors="replace")
    assert "~/lloyd-data/_pipeline/vault-derived/kg.sqlite" in text, (
        f"{rel} no longer cites the live store path, so it is no longer the "
        "lookalike the anchoring rule is calibrated against")
    retired = "~/lloyd/_pipeline/vault-derived/kg.sqlite"
    assert not PC._pattern(PC.KIND_PATH, retired).search(text), (
        "the rule this file is the example for has stopped holding")


def test_the_seed_of_the_acceptance_check_is_still_an_instruction_file(module_text):
    """#3538's clause-5 seed is `autonomy/70-skill-lint.md`, which stores an
    absolute interpreter path as a command. If the task file stops naming it, the
    clause has a new seed or it has nothing, and the rung's one worked example would
    become a claim about a file nobody re-checked."""
    from scripts.automod import prose_consumers as PC

    rel = "autonomy/70-skill-lint.md"
    assert rel in module_text, "the module still names this file as its worked example"
    f = VAULT / rel
    if not f.exists():
        pytest.skip(f"{rel} is gone; the acceptance seed needs re-picking")
    assert PC.classify(rel) == "instruction", (
        "the seed file is no longer classified as an instruction, so clause 5's "
        "fixture cannot report it")
    # The clause-5 spelling, quoted here rather than from the module: the seed is
    # the ACCEPTANCE check's, and a constant in the matcher would let a change to
    # the matcher move the goal as well as the test.
    assert ".venvs/lloyd/bin/python" in f.read_text(encoding="utf-8",
                                                    errors="replace"), (
        f"{rel} no longer stores the interpreter spelling this rung was seeded on; "
        "pick a new seed rather than relaxing the assertion")


def test_the_corpus_directories_exist(module_text):
    """A corpus directory that moved is a rung that greps nothing and reports a
    clean vault — the quietest possible way for a check to stop checking."""
    from scripts.automod import prose_consumers as PC

    scopes = PC.corpus_scopes(VAULT)
    assert scopes, (
        f"none of {sorted(PC.INSTRUCTION_DIRS + PC.RECORD_DIRS)} exists under "
        f"{VAULT}: the rung would report zero consumers for every round")
    assert "autonomy" in scopes and "skills" in scopes, scopes


def test_the_short_name_floor_cites_the_measurement_it_comes_from(module_text):
    """Refusing short candidates is the rung's biggest deliberate miss — a retired
    three-character field name is never searched — and its only justification is
    #882's measured ablation. A floor with no citation attached becomes folklore in
    one edit, so the provenance is pinned beside the number: the module names the
    item it came from, and the refusal it emits quotes the floor it applied."""
    from scripts.automod import prose_consumers as PC

    assert PC.MIN_SYMBOL_CHARS >= 5, PC.MIN_SYMBOL_CHARS
    assert "#882" in module_text, (
        "the short-name floor has to cite the measurement it is derived from")
    assert "{MIN_SYMBOL_CHARS} characters" in module_text, (
        "a refused candidate has to record WHICH floor refused it, or a reader "
        "cannot tell a refusal from a name the extractor never saw")
