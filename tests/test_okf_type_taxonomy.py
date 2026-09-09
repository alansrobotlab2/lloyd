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

Test 5 has one deliberate, dated exemption — see ``TRACKED_RE_FRAGMENTERS``. Five
skill templates still instruct the model to write a pre-consolidation literal
(``deep-research``, ``medium-research``, ``quick-research``, ``entity-overview``,
``source-summary``), so a run of ``quick-research`` can put
``type: quick-research`` back into ``knowledge/`` tomorrow. The exemption is
computed from the skill files themselves, so it disappears the moment they are
fixed and the check becomes strict on its own — but skill bodies are vault paths
#370's acceptance does not name, so retiring them is its own item, not this
round. It is not a licence for any other synonym: a sixth spelling anywhere makes
test 5 fail immediately.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

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
# knowledge note, and the item that retires them. ``deep-research`` 31 files /
# ``medium-research`` 81 / ``quick-research`` 11 carried these literals, so the
# skills are a live writer of the fragmentation — but skill bodies are vault
# paths #370's acceptance does not name, so they are not edited here.
# Shrink this to empty when the item lands; a NEW entry here is a regression.
TRACKED_RE_FRAGMENTERS = {
    "deep-research": "skills/deep-research/SKILL.md:103",
    "medium-research": "skills/medium-research/SKILL.md:76",
    "quick-research": "skills/quick-research/SKILL.md:48",
    "entity-overview": "skills/nightly-reflection-knowledge/SKILL.md:242",
    "source-summary": "skills/ingest/SKILL.md:72",
}

_TYPE_LINE = re.compile(r"^\s*type:\s*([A-Za-z0-9_ -]+?)\s*$", re.M)


def _load_script(rel: str, name: str):
    """Load a hyphen-named file under scripts/ as a module."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _skill_prescribed_legacy_types() -> set[str]:
    """Legacy ``type`` literals an active skill still hands a knowledge note.

    Scope is a skill's *body* (its own frontmatter says ``type: skill``, which is
    unrelated) of a skill that writes into ``knowledge/`` at all. Measured
    2026-09-09: exactly the five in ``TRACKED_RE_FRAGMENTERS``, one per file.
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
        for raw in _TYPE_LINE.findall(body):
            value = okf_taxonomy.normalize_type(raw)
            if value and value not in okf_taxonomy.CANONICAL_TYPES:
                out.add(value)
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

def test_knowledge_frontmatter_uses_only_the_canonical_set():
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


def test_the_re_fragmenting_skill_templates_are_the_tracked_three():
    found = _skill_prescribed_legacy_types()
    assert found <= set(TRACKED_RE_FRAGMENTERS), (
        f"a skill started prescribing a non-canonical knowledge type: "
        f"{found - set(TRACKED_RE_FRAGMENTERS)}")
    assert SKILLS.is_dir(), f"{SKILLS} missing — the ledger above is unverifiable"


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
