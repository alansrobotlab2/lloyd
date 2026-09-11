"""The OKF ``type`` vocabulary has exactly one definition, and ``knowledge/`` obeys it.

Backlog #370 consolidated ``knowledge/`` frontmatter onto a 12-value taxonomy
(50 distinct values / 2,357 typed files on 2026-09-08 — ``research`` 249 beside
``research-note`` 21 beside ``research_note`` 16). The vault rename landed as
vault commits ``6eed1445`` + ``c2b0962``. This file pins the two things that make
the rename *mean* something rather than being one more sed pass:

  1. **One vocabulary in code.** ``validate_okf.py``, ``okf_migrate.py`` and
     ``knowledge-frontmatter-backfill.py`` each carried their own copy of the
     fragmented literals. That is why warnings went *up* when the rename landed:
     155 at triage, 194 on 2026-09-09, because the validator still listed
     ``quick-research``/``medium-research``/``deep-research``/``knowledge-note``/
     ``hub`` and did not list what those values had just become. All three now
     import ``scripts/vault/okf_taxonomy``, and the fragmented spellings may not
     reappear as literals in those files.
  2. **The vault stays inside the set.** ``knowledge/`` is scanned live, the same
     histogram the item's acceptance check names, and every ``type`` must be
     canonical.

Test 5 used to carry one deliberate, dated exemption (``TRACKED_RE_FRAGMENTERS``)
because five skill templates still instructed the model to write a
pre-consolidation literal (``deep-research``, ``medium-research``,
``quick-research``, ``entity-overview``, ``source-summary``) while the vault was
already consolidated — so a run of ``quick-research`` could put
``type: quick-research`` back into ``knowledge/`` tomorrow. Backlog #872 retired
the literals in the skills and in ``knowledge/KNOWLEDGE_SCHEMA.md``, so the ledger
is empty and stays empty: ``test_no_active_skill_prescribes_a_non_canonical_
knowledge_type`` fails on any non-canonical value a skill template names, and an
entry added back to the ledger is itself a failure.

#872 also closed the half this file could not reach — nothing on the *write* side
agreed with the vocabulary. Three more things are pinned here: the schema
document an agent reads before writing (``Page Types`` must equal the module's
set, row for row), the skill templates' inline ``type:`` instructions, and
``vault_write`` itself, which now normalises a retired spelling and refuses an
invented one (``test_writing_an_invented_type_fails_and_creates_nothing``).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vault import okf_taxonomy  # noqa: E402
from scripts.vault.okf_migrate import infer_type  # noqa: E402
from scripts.vault.validate_okf import KNOWN_TYPES, STRICT_FM_RE  # noqa: E402

VAULT = Path.home() / "obsidian"
KNOWLEDGE = VAULT / "knowledge"
SKILLS = VAULT / "skills"

# The 12 values #370 chose. Naming them here is the point: a 13th is a decision,
# not a typo that survived because nothing was watching.
EXPECTED_CANONICAL = {
    "research", "research-deep", "research-quick", "reference", "synthesis",
    "video-note", "book-note", "notes", "stack-update", "infrastructure",
    "agent-pattern", "gap-analysis",
}

# The exact literals #370's acceptance check greps for, and the value each must
# resolve to. If one of these is ever re-introduced, this table says what it
# should have been.
ACCEPTANCE_MAP = {
    "research-note": "research",
    "research_note": "research",
    "knowledge-note": "reference",
    "medium-research": "research-quick",
    "quick-research": "research-quick",
    "youtube-note": "video-note",
    "youtube-video": "video-note",
    "youtube-summary": "video-note",
    "book-research": "book-note",
    "paper-note": "notes",
    "redirect": "infrastructure",
    "hub": "infrastructure",
}

# Active skill templates that still instruct a non-canonical ``type`` for a
# knowledge note, and the item that retires them.
#
# Empty since #872 (vault commit ``aa8dda97`` renamed the four templates that
# still said ``deep-research`` / ``medium-research`` / ``quick-research`` /
# ``source-summary``; ``nightly-reflection-knowledge`` had already dropped
# ``entity-overview`` and now lives under ``skills/.archived/``). Being empty is
# asserted, not assumed — see test_no_active_skill_prescribes_a_non_canonical_
# knowledge_type, which fails on any value in here that a live template names,
# and on an entry added back. Re-arming a retired literal on purpose has to be
# argued for in a round's report, not quietly typed into this dict.
TRACKED_RE_FRAGMENTERS: dict[str, str] = {}

_TYPE_LINE = re.compile(r"^\s*type:\s*([A-Za-z0-9_ -]+?)\s*$", re.M)


def _load_script(rel: str, name: str):
    """Load a hyphen-named file under scripts/ as a module."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _non_canonical_type_literals(body: str) -> set[str]:
    """Non-canonical ``type`` literals in a skill's body text.

    Compared RAW, deliberately. Normalizing first (as this did until #872) makes
    the scan blind to exactly the bug it exists to catch: every retired literal
    the skills were carrying — ``deep-research``, ``medium-research``,
    ``quick-research``, ``source-summary`` — maps onto a canonical value, so at
    base the aliases read as clean and this returned the empty set while four
    templates still said them out loud. An instruction must NAME a canonical
    value, not one that resolves to one.
    """
    return {value.strip() for value in _TYPE_LINE.findall(body)
            if value.strip() and value.strip() not in okf_taxonomy.CANONICAL_TYPES}


def test_the_skill_scan_acts_on_a_literal_that_would_normalize_clean():
    """The raw comparison is this round's headline defect; it must be testable.

    A synthetic body shaped like the four templates #872 renamed. Under the old
    normalize-then-compare rule the assertion below could not fail, which is the
    whole reason clause 2 read as already-met at base.
    """
    body = ("writes `knowledge/{domain}/{slug}.md`:\n\n"
            "```markdown\n---\ntype: deep-research\ntags: [x]\n---\n```\n")
    assert _non_canonical_type_literals(body) == {"deep-research"}
    # …and the same value read through normalize_type WOULD have looked clean:
    assert okf_taxonomy.normalize_type("deep-research") in okf_taxonomy.CANONICAL_TYPES


def _skill_prescribed_legacy_types() -> set[str]:
    """Legacy ``type`` literals an active skill still hands a knowledge note.

    Scope is a skill's *body* (its own frontmatter says ``type: skill``, which is
    a different vocabulary on a different surface) of a skill that writes into
    ``knowledge/`` at all. Empty since #872 renamed the four templates — measured
    across every active skill body, the strict scan finds nothing, so tightening
    it cost nothing on the rest of the tree.
    """
    out: set[str] = set()
    if not SKILLS.is_dir():
        return out
    for skill in SKILLS.rglob("SKILL.md"):
        if ".archived" in skill.parts:
            continue
        text = skill.read_text(encoding="utf-8", errors="replace")
        body = text.split("---", 2)[2] if text.startswith("---") else text
        if "knowledge/" not in body:
            continue
        out |= _non_canonical_type_literals(body)
    return out


def _knowledge_type_values() -> dict[str, int]:
    """The #370 histogram: every ``type`` in ``knowledge/``, counted."""
    counts: dict[str, int] = {}
    for path in sorted(KNOWLEDGE.rglob("*.md")):
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        if not head.startswith("---"):
            continue
        block = head.split("---", 2)[1]
        found = _TYPE_LINE.findall(block)
        if not found:
            continue
        value = found[0].strip().strip("\"'").lower()
        counts[value] = counts.get(value, 0) + 1
    return counts


# ── 1. the taxonomy itself ───────────────────────────────────────────────────

def test_the_canonical_set_is_the_twelve_values_370_chose():
    assert set(okf_taxonomy.CANONICAL_TYPES) == EXPECTED_CANONICAL


def test_every_alias_resolves_to_a_canonical_value():
    bad = {k: v for k, v in okf_taxonomy.TYPE_ALIASES.items()
           if v not in okf_taxonomy.CANONICAL_TYPES}
    assert not bad, f"aliases must land on a canonical value: {bad}"
    shadows = set(okf_taxonomy.TYPE_ALIASES) & set(okf_taxonomy.CANONICAL_TYPES)
    assert not shadows, f"a canonical value cannot also be an alias: {shadows}"


def test_each_acceptance_literal_normalises_to_the_promised_value():
    wrong = {lit: okf_taxonomy.normalize_type(lit)
             for lit, want in ACCEPTANCE_MAP.items()
             if okf_taxonomy.normalize_type(lit) != want}
    assert not wrong, f"acceptance literals not mapped as #370 specifies: {wrong}"


def test_normalisation_is_idempotent_and_separator_tolerant():
    for value in sorted(okf_taxonomy.CANONICAL_TYPES):
        once = okf_taxonomy.normalize_type(value)
        assert okf_taxonomy.normalize_type(once) == once == value
    # The drift that created the 50 values: casing and separator variants of the
    # same word have to land in the same place or the map leaks.
    for spelling in ("research-note", "research_note", "Research Note", " research-note "):
        assert okf_taxonomy.normalize_type(spelling) == "research"


# ── 2. one vocabulary, not three copies ──────────────────────────────────────

def test_validator_known_types_is_the_shared_one_not_a_local_copy():
    assert KNOWN_TYPES is okf_taxonomy.KNOWN_TYPES
    assert set(okf_taxonomy.CANONICAL_TYPES) <= set(KNOWN_TYPES)


def test_the_acceptance_literals_are_recognised_by_the_validator():
    # The 194 warnings: renamed files the validator called "unknown type".
    for value in EXPECTED_CANONICAL:
        assert value in KNOWN_TYPES, f"{value} renamed into existence, still unknown"
    # Tolerance for what already exists outside knowledge/ — dropping one of
    # these would ADD warnings, which #370's acceptance forbids.
    for value in ("note", "skill", "autonomy", "person"):
        assert value in KNOWN_TYPES


def test_no_writer_repeats_a_fragmented_literal_it_could_import():
    banned = {"deep-research", "medium-research", "quick-research", "research-note",
              "knowledge-note", "source-summary", "directory-index", "youtube-note"}
    offenders = []
    for rel in ("scripts/vault/validate_okf.py", "scripts/vault/okf_migrate.py",
                "scripts/knowledge-frontmatter-backfill.py"):
        source = (ROOT / rel).read_text(encoding="utf-8")
        offenders += [f"{rel}: {lit}" for lit in banned
                      if f'"{lit}"' in source or f"'{lit}'" in source]
    assert not offenders, (
        "fragmented `type` literal hardcoded in a writer — the vocabulary lives "
        f"in scripts/vault/okf_taxonomy.py: {offenders}")


def test_backfill_classifies_canonical_and_legacy_files_alike():
    backfill = _load_script("scripts/knowledge-frontmatter-backfill.py",
                            "knowledge_frontmatter_backfill")
    assert set(backfill.SYNTHESIZED_TYPES) == set(okf_taxonomy.SYNTHESIZED_TYPES)
    assert set(backfill.CAPTURED_TYPES) == set(okf_taxonomy.CAPTURED_TYPES)
    assert set(backfill.PRIMARY_TYPES) == set(okf_taxonomy.PRIMARY_TYPES)
    body = "\n# nothing\n"
    # A renamed file and its pre-rename twin get the same source_type.
    assert backfill.infer_source_type({"type": "research-deep"}, body) == "synthesized"
    assert backfill.infer_source_type({"type": "deep-research"}, body) == "synthesized"
    assert backfill.infer_source_type({"type": "infrastructure"}, body) == "primary"
    assert backfill.infer_source_type({"type": "hub"}, body) == "primary"


def test_the_migrator_emits_canonical_types_for_knowledge_notes(tmp_path):
    # The writer that manufactured new fragmentation on every --apply run.
    def inferred(rel: str, fm: dict | None = None) -> str:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# x\n", encoding="utf-8")
        return infer_type(path, fm or {}, "# x\n")

    import scripts.vault.okf_migrate as migrator

    real_root = migrator.VAULT_ROOT
    migrator.VAULT_ROOT = tmp_path
    try:
        assert inferred("knowledge/ai/sources/paper-x.md") == "reference"
        assert inferred("knowledge/ai/research-note.md", {"research-depth": "deep"}) in {
            "research", "research-deep"}
        assert inferred("knowledge/ai/plain-thing.md") == "notes"
        assert inferred("knowledge/ai/ai.md") == "infrastructure"
        assert inferred("knowledge/ai/clip.md", {"video_id": "abc"}) == "video-note"
    finally:
        migrator.VAULT_ROOT = real_root


# ── 3. the acceptance check itself, on the live vault ─────────────────────────

@pytest.mark.live_vault
def test_knowledge_frontmatter_uses_only_the_canonical_set():
    # `live_vault`: this reads ~/obsidian/knowledge as it is right now, which
    # no round under test controls. Unmarked, one note with `type: note`
    # (written by a nightly job) failed the `tests` rung for #551's round on
    # 2026-09-11 — and would have failed every round until someone fixed the
    # vault. The gate runs `-m "not live_vault"`; the full suite still runs it.
    counts = _knowledge_type_values()
    # A guard that reads nothing reports "OK" — fail loudly on a missing tree
    # instead of passing a vacuous scan.
    assert sum(counts.values()) > 1000, (
        f"expected >1000 typed files under {KNOWLEDGE}, found {sum(counts.values())} "
        "— the scan found an empty or moved vault, which is not a pass")
    off_set = {v: n for v, n in counts.items() if v not in okf_taxonomy.CANONICAL_TYPES}
    excused = _skill_prescribed_legacy_types()
    unexpected = {v: n for v, n in off_set.items() if v not in excused}
    assert not unexpected, (
        f"{sum(unexpected.values())} knowledge/ files outside the canonical set: "
        f"{unexpected}. Canonical target for each: "
        f"{ {v: okf_taxonomy.normalize_type(v) for v in unexpected} }"
    )


def test_no_active_skill_prescribes_a_non_canonical_knowledge_type():
    """No live skill template may tell a writer to emit a retired literal.

    Renamed from ``test_the_re_fragmenting_skill_templates_are_the_tracked_three``
    when #872 emptied ``TRACKED_RE_FRAGMENTERS``: with the ledger empty, ``found
    <= TRACKED`` is only satisfiable by ``found`` being empty, so the assertion
    went from "as wide as the bug" to "strict" — and the ledger itself is now a
    tested value, so re-arming an exemption is a diff, not a drift.

    The ``is_dir`` guard is first on purpose: a vault that is not there makes
    ``found`` empty, and a check that cannot see its input must not report a
    pass (#780's class — see lloyd/MEMORY.md).
    """
    assert SKILLS.is_dir(), f"{SKILLS} missing — this check cannot see its input"
    found = _skill_prescribed_legacy_types()
    assert found <= set(TRACKED_RE_FRAGMENTERS), (
        f"a skill started prescribing a non-canonical knowledge type: "
        f"{found - set(TRACKED_RE_FRAGMENTERS)}")
    assert TRACKED_RE_FRAGMENTERS == {}, (
        f"the exemption ledger must be empty since #872; it carries "
        f"{TRACKED_RE_FRAGMENTERS}")


# ── 6. the instructions themselves, wherever they appear inline ──────────────
# Test 5 only reads a fenced template block, because that is what a *file*
# writes. But a skill also tells the model what to type in prose and in a routing
# table (`type: medium-research`, `source_type: synthesized`, …), and #872's
# clause 4 is about every line that names the value to write. So this scans any
# inline `type: <value>` — including a value in back-adjacent prose — and requires
# it to normalise onto a canonical value. `source_type`/`subagent_type` are
# excluded by the lookbehind: they are different fields with different vocabularies.
_INLINE_TYPE_ASSIGNMENT = re.compile(r"(?<![\w_])type:\s*([A-Za-z0-9_ -]+)")
_WRITING_SKILLS = ("deep-research", "medium-research", "quick-research", "ingest")


def test_every_inline_type_instruction_names_a_canonical_value():
    offenders: list[str] = []
    for slug in _WRITING_SKILLS:
        skill = SKILLS / slug / "SKILL.md"
        if not skill.is_file():
            pytest.fail(f"{skill} missing — this check cannot see its input")
        text = skill.read_text(encoding="utf-8", errors="replace")
        # A skill's own frontmatter says `type: skill`, which is a different
        # vocabulary on a different surface (OUTSIDE_KNOWLEDGE_TYPES). Same split
        # as _skill_prescribed_legacy_types uses, for the same reason.
        body = text.split("---", 2)[2] if text.startswith("---") else text
        for lineno, line in enumerate(body.splitlines(), 1):
            for match in _INLINE_TYPE_ASSIGNMENT.finditer(line):
                value = match.group(1).strip().strip("`\"'")
                # RAW, for the reason in _skill_prescribed_legacy_types: an alias
                # resolves to a canonical value, so normalizing here would let
                # `type: deep-research` pass the test that exists to fail it.
                if value and value not in okf_taxonomy.CANONICAL_TYPES:
                    offenders.append(f"skills/{slug}/SKILL.md:{lineno}: `type: {value}`")
    assert not offenders, "a skill instructs a non-canonical `type`: " + "; ".join(offenders)


def test_the_inline_type_instruction_check_acts_on_a_retired_literal():
    """The check above must be capable of failing, on live-tree-shaped input."""
    line = "  `type: deep-research`, `source_type: synthesized`, `domain`"
    found = [m.group(1).strip().strip("`\"'")
             for m in _INLINE_TYPE_ASSIGNMENT.finditer(line)]
    assert found == ["deep-research"], found  # the lookbehind drops source_type
    assert "deep-research" not in okf_taxonomy.CANONICAL_TYPES, "the retired literal is retired"
    # …and normalising it WOULD look clean, which is why the line above is the
    # assertion that matters and not a normalize_type() comparison.
    assert okf_taxonomy.normalize_type("deep-research") == "research-deep"
    assert "research-deep" in okf_taxonomy.CANONICAL_TYPES


# ── 7. the schema document follows the module, row for row ───────────────────
# Deliberately NOT marked ``live_vault``: `knowledge/**` notes are rewritten by
# nightly jobs (which is what test 5's marking is for), but KNOWLEDGE_SCHEMA.md is
# the specification writers read, and nothing writes it unattended. A red here
# means the document and the module disagree — the exact drift #477/#872 exist to
# keep closed — so it belongs on the hard gate, next to the skill scan above.
SCHEMA_DOC = KNOWLEDGE / "KNOWLEDGE_SCHEMA.md"
_TABLE_ROW_FIRST_CELL = re.compile(r"^\|\s*`([^`|]+?)`\s*\|", re.M)
_FENCE = re.compile(r"```(?:yaml|markdown)\n(.*?)\n```", re.S)


def _schema_section(text: str, heading: str) -> str:
    start = text.index(heading) + len(heading)
    rest = text[start:]
    nxt = rest.find("\n## ")
    return rest if nxt == -1 else rest[:nxt]


_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _page_type_table_rows(text: str) -> set[str]:
    """First column of the ``## Page Types`` table — the document's vocabulary.

    Only rows below the ``|---|`` separator count: the header row's first cell is
    the *column name* (``type``), and counting it would report a 13th value that
    no file can carry.
    """
    lines = _schema_section(text, "## Page Types").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if _TABLE_SEPARATOR.match(ln))
    except StopIteration:
        return set()
    return {value.strip()
            for value in _TABLE_ROW_FIRST_CELL.findall("\n".join(lines[start + 1:]))}


def _render_page_types_table(values) -> str:
    rows = "\n".join(f"| `{v}` | fixture |" for v in sorted(values))
    return f"## Page Types\n\n| `type` | Meaning |\n|---|---|\n{rows}\n\n## Next\n"


def test_the_page_types_table_parser_discriminates():
    """Proves the two tests below can fail: one row off either way is caught."""
    canonical = set(okf_taxonomy.CANONICAL_TYPES)
    assert _page_type_table_rows(_render_page_types_table(canonical)) == canonical
    assert _page_type_table_rows(_render_page_types_table(canonical - {"gap-analysis"})) != canonical
    assert _page_type_table_rows(_render_page_types_table(canonical | {"deep-research"})) != canonical


def test_knowledge_schema_page_types_table_is_the_canonical_set_exactly():
    """#872 clause 6: no row missing, no extra row, no renamed row."""
    assert SCHEMA_DOC.is_file(), f"{SCHEMA_DOC} missing — this check cannot see its input"
    rows = _page_type_table_rows(SCHEMA_DOC.read_text(encoding="utf-8"))
    missing = set(okf_taxonomy.CANONICAL_TYPES) - rows
    extra = rows - set(okf_taxonomy.CANONICAL_TYPES)
    assert not missing and not extra, (
        f"knowledge/KNOWLEDGE_SCHEMA.md Page Types table disagrees with "
        f"okf_taxonomy.CANONICAL_TYPES: missing={sorted(missing)} extra={sorted(extra)}")
    assert len(rows) == 12, rows


def test_knowledge_schema_type_lines_are_all_canonical_values():
    """#872 clause 5, as the item's grep states it: ``^[^-]*type: ``.

    The pattern is the clause's, dash-lines and all — which is why the document
    carries its ``source_type`` values as a bullet list instead of inside the
    example block: ``source_type: primary|synthesized|captured`` matched the
    grep and read as a non-canonical ``type``. The clause's grep requires a space
    after the colon, so this does too; prose that merely ends on the word
    ``type:`` is not a value assignment.
    """
    assert SCHEMA_DOC.is_file(), f"{SCHEMA_DOC} missing — this check cannot see its input"
    offenders = []
    for lineno, line in enumerate(SCHEMA_DOC.read_text(encoding="utf-8").splitlines(), 1):
        match = re.match(r"^[^-]*type:[ \t]+([^#\r\n]+)", line)
        if match is None:
            continue
        value = match.group(1).strip().strip("\"'")
        if value not in okf_taxonomy.CANONICAL_TYPES:
            offenders.append(f"KNOWLEDGE_SCHEMA.md:{lineno}: `type: {value}`")
    assert not offenders, "the schema document prescribes a non-canonical type: " + "; ".join(offenders)


def test_knowledge_schema_required_frontmatter_example_uses_a_canonical_type():
    """#872 clause 7: the example an agent copies from is not a retired literal."""
    assert SCHEMA_DOC.is_file(), f"{SCHEMA_DOC} missing — this check cannot see its input"
    section = _schema_section(SCHEMA_DOC.read_text(encoding="utf-8"), "## Required Frontmatter")
    fence = _FENCE.search(section)
    assert fence is not None, "no fenced example under Required Frontmatter"
    match = re.search(r"(?<![\w_])type:\s*([^#\r\n]+)", fence.group(1))
    assert match is not None, "the example carries no `type` key"
    value = match.group(1).strip().strip("\"'")
    assert value in okf_taxonomy.CANONICAL_TYPES, (
        f"the required-frontmatter example says `type: {value}`; the vocabulary is "
        f"{sorted(okf_taxonomy.CANONICAL_TYPES)}")


def test_validator_reports_no_unknown_types_in_knowledge():
    """#370's third clause: the strict warnings must FLOOR at zero, not rise.

    155 at triage, 194 on 2026-09-09 — every renamed file was an "unknown type"
    because the validator's vocabulary predated the rename.
    """
    assert _strict_warnings() == 0


def _strict_warnings() -> int:
    """Count the unknown-type warnings the validator emits for knowledge/."""
    import scripts.vault.validate_okf as validator
    from collections import Counter

    n = warnings = 0
    hist: Counter = Counter()
    for path in validator.iter_md(VAULT, "knowledge"):
        n += 1
        text = path.read_text(encoding="utf-8")
        m = STRICT_FM_RE.match(text)
        if not m:
            continue
        try:
            fm = yaml.safe_load(m.group(1))
        except yaml.YAMLError:
            continue
        if not isinstance(fm, dict):
            continue
        value = str(fm.get("type", "") or "").strip()
        if value:
            hist[value] += 1
            if value not in KNOWN_TYPES:
                warnings += 1
    assert n > 1000, f"validator scanned {n} knowledge files — not a real scan"
    return warnings


# ── 8. the write path: an alias is normalised, an invention is refused ────────
# Everything above checks what the tree and the documents *say*. #780 was the
# other half: nothing looked at a `type` on the way in, so #39 wrote
# ``type: note`` into ``knowledge/software/``, the file landed, and every automod
# promotion on this box was blocked until the word was found by hand. These tests
# drive the real handler, so a writer that invents a value is stopped where it
# would otherwise have been caught by a gate seven hours later.
def _scratch(tmp_path, monkeypatch):
    """``vault_write`` aimed at a scratch tree instead of the live vault.

    ``VAULT`` is imported into ``agent_mcp.vault``'s own namespace, so that is
    the name to patch — patching ``agent_mcp._shared`` would leave the handlers
    writing into ~/obsidian.
    """
    import types

    import agent_mcp.vault as vault_mod
    monkeypatch.setattr(vault_mod, "VAULT", tmp_path)
    monkeypatch.setattr(vault_mod, "AUDIT_LOG_DIR", tmp_path / "audit")
    monkeypatch.setattr(vault_mod, "AUDIT_LOG_FILE", tmp_path / "audit" / "writes.jsonl")
    return types.SimpleNamespace(root=tmp_path, mod=vault_mod)


@pytest.fixture
def scratch_vault(tmp_path, monkeypatch):
    return _scratch(tmp_path, monkeypatch)


def _write_note(scratch, rel: str, type_value: str | None = "research") -> dict:
    content = (f"---\ntype: {type_value}\ndomain: ai\n---\n# Note\n"
               if type_value is not None else "---\ndomain: ai\n---\n# Note\n")
    return scratch.mod._vault_write({"path": rel, "content": content})


def test_writing_an_alias_to_knowledge_lands_the_canonical_type(scratch_vault):
    """#872 clause 8: ``type: deep-research`` arrives as ``research-deep``."""
    result = _write_note(scratch_vault, "knowledge/ai/clipper.md", "deep-research")
    assert result.get("success") is True, result
    landed = (scratch_vault.root / "knowledge/ai/clipper.md").read_text(encoding="utf-8")
    assert "type: research-deep" in landed, landed
    assert "deep-research" not in landed, landed
    assert result["type_normalized"] == {"from": "deep-research", "to": "research-deep"}


def test_writing_an_invented_type_fails_and_creates_nothing(scratch_vault):
    """#872 clause 9: refused, by name, with no file and no directory behind it."""
    result = _write_note(scratch_vault, "knowledge/ai/invented.md", "note")
    assert "error" in result, f"an invented type was accepted: {result}"
    assert "note" in result["error"], result["error"]
    assert result.get("invalid_type") == "note", result
    assert not (scratch_vault.root / "knowledge/ai/invented.md").exists()
    assert not (scratch_vault.root / "knowledge/ai").exists(), (
        "the guard ran after the mkdir — a refused write still leaves a directory")


def test_writing_an_unknown_type_fails_and_names_the_value(scratch_vault):
    result = _write_note(scratch_vault, "knowledge/ai/other.md", "definitely-not-a-type")
    assert "error" in result, f"an invented type was accepted: {result}"
    assert "definitely-not-a-type" in result["error"], result["error"]
    assert not (scratch_vault.root / "knowledge/ai/other.md").exists()


def test_a_canonical_type_is_written_verbatim(scratch_vault):
    result = _write_note(scratch_vault, "knowledge/ai/plain.md", "video-note")
    assert result.get("success") is True, result
    assert "type: video-note" in (scratch_vault.root / "knowledge/ai/plain.md").read_text(encoding="utf-8")
    assert "type_normalized" not in result, "nothing was corrected; don't claim it was"


def test_a_note_with_no_type_is_still_writable(scratch_vault):
    """An absent type is #478's orphan-frontmatter sweep, not a vocabulary error.

    Refusing here would make the tool un-writable for the 221 files #478 is about,
    which is a different decision taken by a different item.
    """
    result = _write_note(scratch_vault, "knowledge/ai/headless.md", None)
    assert result.get("success") is True, result


def test_the_write_guard_leaves_the_rest_of_the_vault_alone(scratch_vault):
    """``knowledge/`` is where the 12 values apply. ``memory/`` is #442's question."""
    result = _write_note(scratch_vault, "memory/scratch.md", "note")
    assert result.get("success") is True, result
    landed = (scratch_vault.root / "memory/scratch.md").read_text(encoding="utf-8")
    assert "type: note" in landed, landed
    assert "type_normalized" not in result


def test_the_mcp_dispatch_path_enforces_the_same_rule(scratch_vault):
    """The seam the code graph cannot see: callers name the tool as a string.

    ``agent_mcp/vault.py``'s dispatcher reaches the handler through the
    ``"vault_write"`` key in a dict — no symbol reference anywhere — so calling
    ``_vault_write`` directly proves nothing about what a skill actually gets.
    This goes through ``call_tool`` and ``_wrap``, i.e. the async boundary and the
    ``isError`` flag, which is what decides whether the writer sees a failure.
    """
    import asyncio
    import json

    async def _call(path: str, value: str):
        return await scratch_vault.mod.call_tool(
            "vault_write", {"path": path, "content": f"---\ntype: {value}\n---\n# n\n"})

    ok = json.loads(asyncio.run(_call("knowledge/ai/alias.md", "medium-research")).content[0].text)
    assert ok.get("success") is True, ok
    assert "type: research-quick" in (
        scratch_vault.root / "knowledge/ai/alias.md").read_text(encoding="utf-8")

    bad_result = asyncio.run(_call("knowledge/ai/bad.md", "not-a-type"))
    # `_wrap` is constructed with the `isError` alias; the attribute is snake_case.
    assert bad_result.is_error is True, "a refused write reached the caller as a success"
    bad = json.loads(bad_result.content[0].text)
    assert "not-a-type" in bad["error"], bad
    assert not (scratch_vault.root / "knowledge/ai/bad.md").exists()


def test_the_taxonomy_write_rule_matrix():
    """``canonical_for_write`` in one table: canonical, alias, outside-only, unknown."""
    assert okf_taxonomy.canonical_for_write("research-deep") == "research-deep"
    assert okf_taxonomy.canonical_for_write("Research_Note") == "research"
    for alias, canonical in (("deep-research", "research-deep"),
                             ("medium-research", "research-quick"),
                             ("quick-research", "research-quick"),
                             ("source-summary", "reference"),
                             ("entity-overview", "agent-pattern")):
        assert okf_taxonomy.canonical_for_write(alias) == canonical, alias
    for refused in ("note", "how-to", "comparison", "made-up-type"):
        with pytest.raises(okf_taxonomy.KnowledgeTypeError) as raised:
            okf_taxonomy.canonical_for_write(refused)
        assert refused in str(raised.value), "the refusal must name the value"
        assert raised.value.value == refused


def test_the_document_type_rewriter_acts_on_a_retired_literal():
    """``normalize_document_type`` rewrites the frontmatter and nothing else."""
    text = "---\ntype: quick-research\ndomain: ai\nsources:\n  - type: paper\n---\nbody type: keeps\n"
    rewritten, replaced = okf_taxonomy.normalize_document_type(text)
    assert replaced == "quick-research"
    assert rewritten.splitlines()[1] == "type: research-quick"
    assert "  - type: paper" in rewritten, "an indented type is a nested key, not the page's"
    assert "body type: keeps" in rewritten, "only frontmatter is in scope"
    assert okf_taxonomy.normalize_document_type(rewritten) == (rewritten, None)
    assert okf_taxonomy.normalize_document_type("# no frontmatter\n") == ("# no frontmatter\n", None)
    assert okf_taxonomy.normalize_document_type("---\ndomain: ai\n---\nbody\n") == ("---\ndomain: ai\n---\nbody\n", None)


def test_the_rewriter_keeps_a_trailing_comment():
    """#872 review advisory: the value was wrong, the reason beside it was not."""
    text = "---\ntype: quick-research   # REQUIRED — the page type\n---\nbody\n"
    rewritten, replaced = okf_taxonomy.normalize_document_type(text)
    assert replaced == "quick-research"
    assert rewritten.splitlines()[1] == "type: research-quick  # REQUIRED — the page type"


def test_a_lander_cannot_land_a_type_that_is_not_already_canonical(tmp_path):
    """#872's other writer: a note written with generic file tools lands HERE.

    `automod_vault_land` is a writer of `knowledge/` too, and its validator only
    ever asked whether the front matter *parses*. Stricter than `vault_write` on
    purpose: a lander commits bytes, so it may not accept a value it would have
    to rewrite to be honest about — that value would sit in the tree and fail the
    canonical-set test on the next round.
    """
    from scripts.automod import vault_round

    def note(rel: str, body: str) -> Path:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        return target

    invented = note("knowledge/ai/bad.md", "---\ntype: note\ndomain: ai\n---\n# x\n")
    err = vault_round.knowledge_type_error(invented)
    assert err and "note" in err, err
    assert note("knowledge/ai/good.md", "---\ntype: notes\ndomain: ai\n---\n# x\n") and \
        vault_round.knowledge_type_error(tmp_path / "knowledge/ai/good.md") is None

    alias = note("knowledge/ai/alias.md", "---\ntype: deep-research\n---\n# x\n")
    alias_err = vault_round.knowledge_type_error(alias)
    assert alias_err and "research-deep" in alias_err, (
        f"the refusal must name the value to write instead: {alias_err}")
    assert alias.read_text(encoding="utf-8").startswith("---\ntype: deep-research"), \
        "the lander refuses; it does not rewrite an author's file"

    orphan = note("knowledge/ai/orphan.md", "# no front matter at all\n")
    assert vault_round.knowledge_type_error(orphan) is None, (
        "#478's orphan-frontmatter files must stay landable — this item does not "
        "hold a round hostage to a sweep it does not own")


def _make_taxonomy_unimportable(monkeypatch) -> None:
    """Make `from scripts.vault import okf_taxonomy` raise ImportError.

    Both removals are needed, which is not obvious: `from package import name`
    finds the submodule as an ATTRIBUTE of an already-imported package, so
    clearing only the `sys.modules` entry — the first thing one reaches for —
    leaves the guard reading a perfectly good vocabulary and reporting a clean
    verdict about a check it did not perform. That is the failure mode the branch
    under test exists to prevent, reproduced by the test for it.
    """
    import scripts.vault as vault_pkg
    monkeypatch.delattr(vault_pkg, "okf_taxonomy", raising=False)
    monkeypatch.delitem(sys.modules, "scripts.vault.okf_taxonomy", raising=False)
    monkeypatch.setitem(sys.modules, "scripts.vault.okf_taxonomy", None)


def test_the_lander_reports_an_unusable_vocabulary_instead_of_clean(tmp_path, monkeypatch):
    """A check that cannot read its vocabulary must not report 'clean'."""
    from scripts.automod import vault_round
    target = tmp_path / "knowledge/ai/x.md"
    target.parent.mkdir(parents=True)
    target.write_text("---\ntype: whatever\n---\n# x\n", encoding="utf-8")
    _make_taxonomy_unimportable(monkeypatch)
    err = vault_round.knowledge_type_error(target)
    assert err and "unimportable" in err, err


def test_the_write_guard_fails_closed_when_the_taxonomy_is_unusable(tmp_path, monkeypatch):
    """Same rule on the MCP side: no taxonomy, no write — never a silent pass."""
    scratch = _scratch(tmp_path, monkeypatch)
    _make_taxonomy_unimportable(monkeypatch)
    result = _write_note(scratch, "knowledge/ai/x.md", "research")
    assert "error" in result, f"the guard reported a verdict it could not justify: {result}"
    assert "unimportable" in result["error"], result["error"]
    assert not (scratch.root / "knowledge/ai/x.md").exists()


def test_the_write_guard_reports_an_unexpected_rewrite_failure(tmp_path, monkeypatch):
    scratch = _scratch(tmp_path, monkeypatch)
    def _boom(_text):
        raise RuntimeError("rewriter exploded")
    monkeypatch.setattr(okf_taxonomy, "normalize_document_type", _boom)
    result = _write_note(scratch, "knowledge/ai/x.md", "research")
    assert "error" in result and "rewriter exploded" in result["error"], result
    assert not (scratch.root / "knowledge/ai/x.md").exists()


def test_the_landed_skill_bodies_still_load_through_the_lander_itself():
    """#872 clause 11, across the seam the lander actually uses.

    `automod_vault_land` decides whether a skill or prompt edit may land by
    running `_load_skill` and `build_system_prompt()` in a FRESH interpreter
    (`scripts/automod/vault_round.py`), and every existing test that mentions
    `loader_errors` mocks it — so the suite never proved that seam still ran,
    which is how "it loads" could be claimed without anything pinning it. This
    calls it for real: a ~20 s subprocess against the vault as landed, and the
    only thing standing between a skill body that will not load and a commit.
    """
    from scripts.automod import vault_round
    paths = [f"skills/{slug}/SKILL.md" for slug in _WRITING_SKILLS]
    paths.append("knowledge/KNOWLEDGE_SCHEMA.md")
    assert vault_round.loader_errors(paths) == [], (
        "the lander's own loader check rejects a path #872 renamed — the skill "
        "bodies or the prompt surface no longer load")
