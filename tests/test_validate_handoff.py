"""Nightly handoff validation — the gate between analysis and write.

Why this file exists
--------------------
`validate_handoff.py` is what stands between the Knowledge Analysis artifact and
the stage that rewrites USER.md and MEMORY.md. On 2026-08-24 a handoff claimed
the entity graph was "restored to 12,131 relationships" while the same night's
health report said zero; the claim was consumed and had to be walked back. The
validator itself had no tests, so nothing pinned what it actually catches.

Scope note: it validates *structure*, not truth. It will accept a well-formed
artifact full of false claims — tested explicitly below, because believing the
validator covers more than it does is how the 08-24 overclaim got through.
"""
from __future__ import annotations

import pytest

from scripts import validate_handoff as vh

VALID = """---
generated: 2026-09-05
window: 2026-08-30..2026-09-05
---
# Knowledge Handoff — 2026-09-05

## Person: Alan

### Mental Model — Decision Patterns
- prefers scoped fixes over rewrites

### Mental Model — Communication Preferences
- terse, no sycophancy

### Mental Model — Technical Preferences
- Markdown over JSON

### Mental Model — Project Prioritization
- safety gates before features

### MEMORY.md Additions
- **reflection-pipeline health** — clean for 2 cycles

### MEMORY.md Removals
- (none)

### MEMORY.md Relocations
- (none)

### Profile Updates
- (none)

### Missing Files
- (none)

## Vault Propagations
- knowledge/ai/foo.md

## Tool Patterns — Failures
- `rg` missing post-migration

## Tool Patterns — Successes
- worktree isolation

## Conversation Patterns
- burst then trough

## Priority Order
1. safety
2. hygiene
"""


def run(tmp_path, monkeypatch, capsys, text=None, path=None):
    p = path or (tmp_path / "knowledge-handoff-2026-09-05.md")
    if text is not None:
        p.write_text(text, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_handoff.py", str(p)])
    code = vh.main()
    return code, capsys.readouterr()


def test_valid_handoff_passes(tmp_path, monkeypatch, capsys):
    code, out = run(tmp_path, monkeypatch, capsys, VALID)
    assert code == 0
    assert "[ok]" in out.out


def test_missing_file_fails(tmp_path, monkeypatch, capsys):
    code, out = run(tmp_path, monkeypatch, capsys,
                    path=tmp_path / "absent.md")
    assert code == 1 and "not found" in out.err


def test_no_arguments_is_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["validate_handoff.py"])
    assert vh.main() == 2


def test_too_many_arguments_is_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["validate_handoff.py", "a", "b"])
    assert vh.main() == 2


def test_missing_frontmatter_is_reported(tmp_path, monkeypatch, capsys):
    code, out = run(tmp_path, monkeypatch, capsys, VALID.split("---\n", 2)[2])
    assert code == 1 and "frontmatter" in out.out


def test_missing_h1_is_reported(tmp_path, monkeypatch, capsys):
    code, out = run(tmp_path, monkeypatch, capsys,
                    VALID.replace("# Knowledge Handoff — 2026-09-05", "# Notes"))
    assert code == 1 and "Knowledge Handoff" in out.out


@pytest.mark.parametrize("section", vh.REQUIRED_H2)
def test_each_required_h2_is_required(tmp_path, monkeypatch, capsys, section):
    broken = VALID.replace(section, "## Something Else Entirely")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and section in out.out


@pytest.mark.parametrize("h3", vh.REQUIRED_PERSON_H3)
def test_each_required_person_h3_is_required(tmp_path, monkeypatch, capsys, h3):
    broken = VALID.replace(h3, "### Renamed Subsection")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and h3 in out.out


def test_no_person_block_at_all_is_reported(tmp_path, monkeypatch, capsys):
    broken = VALID.replace("## Person: Alan", "## Nobody")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and "no `## Person:" in out.out


def test_second_person_is_validated_too(tmp_path, monkeypatch, capsys):
    """A second person added without the required subsections must not slip in
    unvalidated."""
    broken = VALID + "\n## Person: Second\n\n### Mental Model — Decision Patterns\n- x\n"
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and "person `Second`" in out.out


def test_frequency_with_trailing_prose_is_rejected(tmp_path, monkeypatch, capsys):
    """The real drift it exists for: `frequency: 4 (tasks ...)` breaks the
    downstream integer parse."""
    broken = VALID.replace("- prefers scoped fixes over rewrites",
                           "frequency: 4 (tasks this week)")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and "non-integer trailing text" in out.out


def test_frequency_as_a_bare_integer_is_accepted(tmp_path, monkeypatch, capsys):
    ok = VALID.replace("- prefers scoped fixes over rewrites", "frequency: 4")
    code, out = run(tmp_path, monkeypatch, capsys, ok)
    assert code == 0, out.out


def test_frequency_that_is_not_an_integer_is_rejected(tmp_path, monkeypatch, capsys):
    broken = VALID.replace("- prefers scoped fixes over rewrites", "frequency: daily")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and "not a bare integer" in out.out


def test_frequency_line_numbers_point_at_the_offending_line(tmp_path, monkeypatch, capsys):
    broken = VALID.replace("- prefers scoped fixes over rewrites", "frequency: 4 (x)")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    expected = VALID.splitlines().index("- prefers scoped fixes over rewrites") + 1
    assert f"line {expected}:" in out.out


def test_all_issues_are_reported_together_not_just_the_first(tmp_path, monkeypatch, capsys):
    broken = VALID.replace("## Priority Order", "## Nope").replace(
        "### Missing Files", "### Nope2")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1
    assert "2 validation issue(s)" in out.out


def test_cosmetic_variation_is_tolerated_deliberately(tmp_path, monkeypatch, capsys):
    """Documented forgiveness: the 35B local model drifts on phrasing, so the
    validator checks presence, not prose."""
    ok = VALID.replace("- terse, no sycophancy",
                       "  - Terse!   No sycophancy whatsoever 🎯")
    code, _ = run(tmp_path, monkeypatch, capsys, ok)
    assert code == 0


def test_structurally_valid_but_factually_false_handoff_passes(tmp_path, monkeypatch, capsys):
    """The limit of this gate, stated as a test.

    The 08-24 incident handoff claimed the entity graph was restored to 12,131
    relationships while the same night's health report said zero. Nothing here —
    or in the validator — catches that. Claim-vs-disk verification has to live
    somewhere else; this only stops malformed artifacts from reaching the writer.
    """
    lie = VALID.replace("- (none)\n\n## Vault Propagations",
                       "- **KG RECOVERED to 12,131 relationships**\n\n## Vault Propagations")
    code, out = run(tmp_path, monkeypatch, capsys, lie)
    assert code == 0, "validator accepted an unfalsifiable claim — by design"
    assert "12,131" not in out.out


# ── split_person_blocks ──────────────────────────────────────────────────────

def test_person_blocks_are_split_by_name():
    text = "## Person: Alan\nbody a\n## Person: Bob\nbody b\n"
    blocks = vh.split_person_blocks(text)
    assert [n for n, _ in blocks] == ["Alan", "Bob"]
    assert "body a" in blocks[0][1] and "body b" in blocks[1][1]


def test_a_person_block_ends_at_the_next_h2():
    """Without this, `### Missing Files` from person one would satisfy person
    two's requirement and every multi-person handoff would pass vacuously."""
    text = ("## Person: Alan\n### Mental Model — Decision Patterns\nx\n"
            "## Vault Propagations\n### Mental Model — Decision Patterns\ny\n")
    blocks = vh.split_person_blocks(text)
    assert len(blocks) == 1
    assert "### Mental Model — Decision Patterns" in blocks[0][1]
    assert "y" not in blocks[0][1]


def test_no_person_blocks_yields_an_empty_list():
    assert vh.split_person_blocks("# just a title\n") == []


def test_person_names_are_trimmed():
    blocks = vh.split_person_blocks("## Person:    Padded Name   \nbody\n")
    assert blocks[0][0] == "Padded Name"


def test_h1_is_not_mistaken_for_an_h2_boundary():
    text = "## Person: Alan\n# H1 inside\nstill alan\n"
    blocks = vh.split_person_blocks(text)
    assert "still alan" in blocks[0][1]


def test_required_section_lists_are_not_empty():
    """A typo that emptied either list would make the validator pass anything."""
    assert vh.REQUIRED_H2 and vh.REQUIRED_PERSON_H3
    assert all(h.startswith("## ") for h in vh.REQUIRED_H2)
    assert all(h.startswith("### ") for h in vh.REQUIRED_PERSON_H3)
    assert len(set(vh.REQUIRED_H2)) == len(vh.REQUIRED_H2)
    assert len(set(vh.REQUIRED_PERSON_H3)) == len(vh.REQUIRED_PERSON_H3)
