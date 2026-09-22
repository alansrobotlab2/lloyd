"""A second frontmatter block stranded in a BODY is visible, counted, and fixable.

Backlog #960. The tree has ~445 files whose real frontmatter sits *below* the
block on disk: a July 2026 batch pass prepended a stamped block over notes that
already had one, sometimes eating the delimiter, so what a strict reader sees is
the stamp's `type: notes` and what a human sees, five lines later, is
`type: book-note`. `knowledge/foundational/thinking-foundations.md` is the
canonical shape — leading block ends at line 9, the real block starts at line 10
with `type: book-note` and never closes.

Before this round neither tool could see it. `validate_okf.py` anchored
``STRICT_FM_RE`` at offset 0 and applied it with ``.match()``, and
`okf_migrate.py`'s ``split_leniently()`` returned ``parts[1], parts[2]`` and parsed
only ``parts[1]`` — so the body, where the stranded block lives, was read by
nothing. The validator printed ``VIOLATIONS: 0`` over 2,574 files on 2026-09-12 and
``--repair-only`` printed ``repaired: 0``, while a plain line walk of the same tree
found 220 files with a body-level ``type:`` outside a code fence, 205 of them
contradicting the type the parser had just read. Both numbers are re-measured by
``test_the_allow_list_covers_every_stranded_knowledge_file``; the defect was not
the count, it was that no machine could see it.

One detector now serves both scripts: `scripts/vault/okf_stranded.py`. Two
properties make it trustworthy where a regex over `type:` lines is not:

  * **fence-aware.** `knowledge/KNOWLEDGE_SCHEMA.md` documents the corruption with
    a fenced ```yaml example that is itself two blocks — a naive check reports the
    schema file as corrupt. A candidate whose opening line sits inside an open
    code fence is prose.
  * **structural, not vocabulary-driven.** A candidate is delimited (`---`-fenced,
    or ≥2 top-level key lines at the body's very start). That is what separates a
    stranded block from `knowledge/tools/openclaw/prs.md`, which quotes five
    `type: user-facing description` lines from a CLI's help output between two
    lines of prose — the family #1003 refuted as a data finding. No list of type
    names could make that distinction, because the word is the same.

Counting is deliberately split. `validate_okf.py` reports allow-listed legacy
files as a separate `known-stranded` total and does NOT fold them into VIOLATIONS:
autonomy task #80 runs this script weekly and reports its counts line, and 445
extra violations every week would bury the only number that matters — a NEW
stranded file, which is a violation and does fail the gate.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vault import okf_migrate  # noqa: E402
from scripts.vault import okf_stranded  # noqa: E402
from scripts.vault import validate_okf  # noqa: E402

VAULT = Path.home() / "obsidian"

# The item's own example, byte for byte: one leading block, then a body block
# whose `type` contradicts it. The leading block wins, so repairing the file is a
# data decision (#478) — but *seeing* it is this round's job.
TWO_BLOCK = (
    "---\n"
    "type: notes\n"
    "tags:\n"
    "- research\n"
    "---\n"
    "type: book-note\n"
    "tags: [synthesis,mental-models]\n"
    "description: a note about four books\n"
    "# Thinking Foundations\n"
    "\n"
    "Prose.\n"
)

# The shape the July pass left when it swallowed only the OPENING delimiter:
# `---`-fenced, and the closing fence is there.
TWO_BLOCK_FENCED = (
    "---\n"
    "type: notes\n"
    "segment: knowledge\n"
    "---\n"
    "# Heading\n"
    "\n"
    "Intro prose.\n"
    "\n"
    "---\n"
    "segment: knowledge\n"
    "type: video-note\n"
    "---\n"
    "\n"
    "More prose.\n"
)


def _one(text: str):
    found = okf_stranded.find_stranded_frontmatter(text)
    assert len(found) == 1, f"expected exactly one stranded block, got {found}"
    return found[0]


# ── clause 1: one block, reported once, with its line ────────────────────────

def test_a_body_block_is_reported_once_with_the_line_it_starts_on():
    hit = _one(TWO_BLOCK)
    assert hit.line == 6, "the leading block is 5 lines; the body block starts at 6"
    assert hit.keys == ("type", "tags", "description")
    assert hit.fenced is False


def test_the_line_number_points_at_the_block_as_a_reader_counts_it():
    # A fenced candidate is reported at its `---` — the line a reader (and an
    # editor's `:9`) would jump to, not the first key under it.
    hit = _one(TWO_BLOCK_FENCED)
    assert hit.line == 9
    assert TWO_BLOCK_FENCED.split("\n")[hit.line - 1] == "---"


def test_the_inner_text_is_the_keys_and_the_text_is_what_gets_excised():
    hit = _one(TWO_BLOCK)
    assert hit.inner == ("type: book-note\ntags: [synthesis,mental-models]"
                 "\ndescription: a note about four books")
    assert TWO_BLOCK[hit.start:hit.end] == hit.text
    assert hit.text.endswith("\n")
    fenced = _one(TWO_BLOCK_FENCED)
    # A fenced block's excision takes its delimiters with it — otherwise the
    # repair leaves a stray `---` sitting in the prose, which is still a
    # delimiter boundary for every lenient parser downstream.
    assert fenced.text == "---\nsegment: knowledge\ntype: video-note\n---\n"


def test_a_repeated_key_is_reported_once_per_block_not_once_per_key():
    # The same key in both blocks is the whole point (#478's 205 contradictions),
    # and it must not make one block look like two.
    text = ("---\ntype: notes\ntimestamp: '2026-07-06'\n---\n"
            "type: book-note\ndate: 2026-04-18\n\nbody\n")
    assert len(okf_stranded.find_stranded_frontmatter(text)) == 1


def test_two_body_blocks_are_reported_as_two():
    text = ("---\ntype: notes\n---\n\n"
            "---\nsegment: knowledge\ntype: video-note\n---\n\nchunk\n\n"
            "---\nsegment: knowledge\ntype: video-note\n---\n\ntail\n")
    assert [h.line for h in okf_stranded.find_stranded_frontmatter(text)] == [5, 12]


def test_a_document_with_no_leading_block_is_not_a_stranded_block():
    # A plain document that merely CONTAINS a frontmatter-looking block (an
    # example, a template) has no leading block to be second to. The validator
    # reports it as "no parseable frontmatter", which is a different violation.
    assert okf_stranded.find_stranded_frontmatter(
        "# Notes\n\n---\ntype: notes\ntags: [x]\n---\n\nbody\n") == []


# ── clause 2: what must NOT be reported ──────────────────────────────────────

def test_a_type_line_inside_a_fenced_code_block_is_not_a_block():
    # knowledge/KNOWLEDGE_SCHEMA.md documents this corruption with a fenced
    # example that is itself two blocks. It must not be reported corrupt.
    text = ("---\ntype: notes\ndescription: schema\n---\n"
            "# Schema\n\n"
            "```yaml\n"
            "---\n"
            "type: research\n"
            "description: an example page\n"
            "---\n"
            "```\n"
            "\nProse.\n")
    assert okf_stranded.find_stranded_frontmatter(text) == []


def test_the_real_schema_file_that_shaped_this_rule_is_clean():
    schema = VAULT / "knowledge" / "KNOWLEDGE_SCHEMA.md"
    # Asserted, not skipped: this is the file the fence rule was derived from, and
    # every other live-vault test here reads the vault unconditionally. A skip
    # would let the check that separates 220 from 221 disappear unnoticed the
    # moment the file moved.
    assert schema.is_file(), f"{schema} must exist for this rule to be pinned"
    # The one file whose body `type:` line is a deliberate example. Counted by the
    # fence-blind walk that found 220 files, so a check keyed on `type:` alone
    # would report the schema itself as corrupt.
    assert okf_stranded.find_stranded_frontmatter(
        schema.read_text(encoding="utf-8")) == []


def test_a_bare_key_line_between_two_prose_lines_is_not_a_block():
    # The shape of the five `type: user-facing description` lines in
    # knowledge/tools/openclaw/prs.md: quoted CLI help output, prose either side.
    # One key line and no delimiters — reporting it would be reporting prose.
    text = ("---\ntype: notes\nsegment: knowledge\n---\n"
            "The CLI prints:\n"
            "\n"
            "```\n"
            "name: pr\n"
            "description: Summarize open pull requests\n"
            "type: user-facing description\n"
            "```\n"
            "\n"
            "That flag is new in 2026.3.8.\n")
    assert okf_stranded.find_stranded_frontmatter(text) == []


def test_a_quoted_help_block_in_prose_without_fences_is_not_a_block():
    # Same rule with the fences removed: prose above, prose below, keys between.
    # The two-key floor and the delimiter are the only thing separating a block
    # from a quotation, so the floor holds here — no delimiters, fewer than two
    # keys, or prose mixed into the run and it is not a block.
    text = ("---\ntype: notes\n---\n"
            "The schema also says:\n"
            "\n"
            "type: user-facing description\n"
            "\n"
            "and that is the only place it appears.\n")
    assert okf_stranded.find_stranded_frontmatter(text) == []


def test_a_few_key_lines_inside_a_prose_section_are_not_a_block():
    # Not at the body's start, no delimiters: a list of fields written in prose.
    text = ("---\ntype: notes\n---\n"
            "# Fields\n\n"
            "The record carries:\n"
            "author: someone\n"
            "title: something\n"
            "type: user-facing description\n"
            "\nand nothing else.\n")
    assert okf_stranded.find_stranded_frontmatter(text) == []


def test_one_key_between_two_fences_is_a_block():
    # The fragment the same batch left behind (`---\nsegment: backlog\n---`) is
    # 196 files of the set. Delimiters make a one-key block unambiguous, which is
    # why the two-key floor applies only to the undelimited case.
    text = ("---\ntype: notes\n---\n\nsome prose\n\n"
            "---\nsegment: backlog\n\n---\n\nmore prose\n")
    assert [h.keys for h in okf_stranded.find_stranded_frontmatter(text)] == [("segment",)]


def test_a_candidate_wrapped_in_an_open_code_fence_is_not_a_block():
    # A wrapper fence whose language line carries the content: everything until its
    # own closing fence is code, including a block that looks perfectly delimited.
    text = ("---\ntype: notes\n---\n"
            "````markdown\n"
            "```python\nprint(1)\n```\n"
            "---\n"
            "type: notes\n"
            "tags: [x]\n"
            "---\n"
            "````\n")
    assert okf_stranded.find_stranded_frontmatter(text) == []


def test_the_detector_is_not_moved_by_key_names_it_has_never_seen():
    # Structural, not vocabulary-driven: a body block carrying keys outside the OKF
    # set is still a block, because nothing in here consults CANONICAL_TYPES. A
    # detector keyed on known frontmatter keys would miss exactly the blocks a
    # foreign tool wrote.
    found = okf_stranded.find_stranded_frontmatter(
        "---\ntype: notes\n---\nsegment: knowledge\nlocale: zh\ntone: dry\n\nbody\n")
    assert [h.keys for h in found] == [("segment", "locale", "tone")]


# ── clause 3: the validator names the file and the line ──────────────────────

def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _run(monkeypatch, capsys, module, *argv) -> tuple[int, str]:
    """Run a script's main() as the CLI the weekly job runs it as."""
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    code = module.main()
    return code, capsys.readouterr().out


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A fixture vault: one clean file, one stranded file, one allow-listed."""
    _write(tmp_path, "knowledge/clean.md",
           "---\ntype: research\ndescription: fine\n---\n\n# Fine\n")
    _write(tmp_path, "knowledge/stranded.md", TWO_BLOCK)
    _write(tmp_path, "knowledge/legacy.md", TWO_BLOCK_FENCED)
    return tmp_path


@pytest.fixture
def listed(monkeypatch):
    """Point the validator's known-stranded allow-list at exactly these paths.

    The allow-list is a file under `scripts/vault/`, and these tests run against a
    fixture tree; leaving it real would make every assertion here depend on the
    445 legacy entries, which is #478's business and not this round's.
    """
    def _apply(*paths: str) -> None:
        monkeypatch.setattr(validate_okf, "load_allowlist",
                            lambda *a, **k: frozenset(paths))
    return _apply


def _violation_lines(out: str) -> list[str]:
    head, _, rest = out.partition("OKF violations")
    return [l for l in rest.splitlines() if l.strip().startswith("knowledge/")]


def test_the_validator_names_the_stranded_file_and_its_line(tree, listed, monkeypatch,
                                                           capsys):
    listed()
    code, out = _run(monkeypatch, capsys, validate_okf, "--root", str(tree))
    assert "VIOLATIONS : 2" in out, out
    assert "known-stranded : 0" in out, out
    assert "knowledge/stranded.md:6" in out, out
    assert "stranded frontmatter block" in out, out
    assert code == 1, "a stranded block the allow-list does not name must fail the gate"


def test_the_validator_reports_the_file_and_line_together_not_either_alone(
        tree, listed, monkeypatch, capsys):
    # Both halves are needed to act on it: a count says nothing about which file,
    # and a bare path says nothing about where the block starts.
    _write(tree, "knowledge/stranded.md",
           "---\ntype: notes\nsegment: knowledge\n---\n\n# H\n\nprose\n\n"
           "---\nsegment: knowledge\ntype: video-note\n---\n\nmore prose\n")
    listed()
    code, out = _run(monkeypatch, capsys, validate_okf, "--root", str(tree))
    assert code == 1
    line = next(l for l in _violation_lines(out) if "stranded.md" in l)
    assert re.search(r"knowledge/stranded\.md:\d+: stranded frontmatter block", line), line


def test_an_allow_listed_file_is_counted_separately_and_not_as_a_violation(
        tree, listed, monkeypatch, capsys):
    listed("knowledge/legacy.md")
    code, out = _run(monkeypatch, capsys, validate_okf, "--root", str(tree))
    assert "known-stranded : 1" in out, out
    assert "VIOLATIONS : 1" in out, out
    assert [l.split(":")[0].strip() for l in _violation_lines(out)] == ["knowledge/stranded.md"], out
    assert "knowledge/stranded.md:6" in out, out
    assert code == 1, "the exit is the un-listed stranded file's, not the legacy one's"


def test_with_every_stranded_file_listed_the_validator_passes(tree, listed, monkeypatch,
                                                             capsys):
    listed("knowledge/legacy.md", "knowledge/stranded.md")
    code, out = _run(monkeypatch, capsys, validate_okf, "--root", str(tree))
    assert "VIOLATIONS : 0" in out, out
    assert "known-stranded : 2" in out, out
    assert code == 0, out


def test_the_stranded_check_survives_strict_on_a_clean_tree(listed, monkeypatch,
                                                           tmp_path, capsys):
    listed()
    _write(tmp_path, "knowledge/a.md",
           "---\ntype: research\ndescription: d\n---\n\n# A\n")
    code, out = _run(monkeypatch, capsys, validate_okf, "--root", str(tmp_path),
                     "--strict")
    assert code == 0, out
    assert "known-stranded : 0" in out, out


def test_the_detector_backs_the_validator_with_a_standing_call(tree, monkeypatch,
                                                              capsys):
    # The wiring is one call per file, not a re-implementation of the rules — a
    # second copy of the rules in the validator is what let the two scripts
    # disagree before this round.
    calls: list[str] = []
    real = okf_stranded.find_stranded_frontmatter

    def spy(text, *a, **k):
        calls.append(text[:24])
        return real(text, *a, **k)

    monkeypatch.setattr(validate_okf, "find_stranded_frontmatter", spy)
    monkeypatch.setattr(sys, "argv", ["prog", "--root", str(tree)])
    validate_okf.main()
    capsys.readouterr()
    assert any("type: notes" in c for c in calls), "the validator never called the detector"


# ── clause 4: the migrator collapses the block, first block wins ─────────────

def test_repair_only_collapses_the_second_block_with_the_first_winning(
        tree, monkeypatch, capsys):
    code, out = _run(monkeypatch, capsys, okf_migrate, "--root", str(tree),
                     "--dir", "knowledge", "--repair-only", "--apply")
    assert "repaired     : 2" in out, out
    assert code == 0, out

    text = (tree / "knowledge/stranded.md").read_text(encoding="utf-8")
    fm = yaml.safe_load(re.match(r"\A---\n(.*?)\n---\n", text, re.S).group(1))
    # The leading block's `type: notes` survives; the buried `type: book-note`
    # does not overwrite it. Same setdefault rule repair_skill_frontmatter.py:111
    # has always applied to skills/, and the reason a July stamp's `notes` must
    # not become the truth over the `book-note` it buried.
    assert fm["type"] == "notes"
    assert fm["tags"] == ["research"], "the leading block's tags win too"
    # Unique keys from the buried block are not lost — that is the whole value of
    # merging instead of deleting.
    assert fm["description"] == "a note about four books"
    assert "Prose." in text, "the body must survive the excision"
    assert "# Thinking Foundations" in text


def test_the_collapsed_file_holds_exactly_one_block_and_no_stray_delimiter(
        tree, monkeypatch, capsys):
    _run(monkeypatch, capsys, okf_migrate, "--root", str(tree),
         "--dir", "knowledge", "--repair-only", "--apply")
    for rel in ("knowledge/stranded.md", "knowledge/legacy.md"):
        text = (tree / rel).read_text(encoding="utf-8")
        assert text.startswith("---\n"), rel
        assert text.count("\n---\n") == 1, f"{rel} left more than one delimiter:\n{text[:400]}"
        assert okf_stranded.find_stranded_frontmatter(text) == [], rel
        assert validate_okf.STRICT_FM_RE.match(text), rel


def test_running_the_same_command_again_reports_every_file_unchanged(
        tree, monkeypatch, capsys):
    _run(monkeypatch, capsys, okf_migrate, "--root", str(tree),
         "--dir", "knowledge", "--repair-only", "--apply")
    before = {p.name: p.read_text(encoding="utf-8")
              for p in (tree / "knowledge").glob("*.md")}
    code, out = _run(monkeypatch, capsys, okf_migrate, "--root", str(tree),
                     "--dir", "knowledge", "--repair-only", "--apply")
    assert "ok/unchanged : 3" in out, out
    assert "repaired     : 0" in out, out
    assert code == 0, out
    after = {p.name: p.read_text(encoding="utf-8")
             for p in (tree / "knowledge").glob("*.md")}
    assert after == before, "the second run must not touch a repaired file"


def test_a_repair_only_run_names_a_stranded_file_as_work_without_apply(tree,
                                                                      monkeypatch,
                                                                      capsys):
    # `--repair-only` without `--apply` is what an operator reads first: it must
    # name the stranded files as work to do, not as ok/unchanged, and must not
    # write anything.
    _, out = _run(monkeypatch, capsys, okf_migrate, "--root", str(tree),
                  "--dir", "knowledge", "--repair-only")
    assert "repaired     : 2" in out, out
    assert "ok/unchanged : 1" in out, out
    assert (tree / "knowledge/stranded.md").read_text(encoding="utf-8") == TWO_BLOCK, \
        "a dry run must not write"


def test_a_body_block_whose_values_will_not_parse_is_refused_and_left_untouched(
        tree, monkeypatch, capsys):
    # #961's flow-glue family: the buried block's own YAML is broken. Merging
    # GUESSED values into the strict-parseable leading block is a worse trade than
    # repairing a block that is already broken, so the file is left byte-identical.
    glued = ("---\ntype: notes\nsegment: knowledge\n---\n\n"
             "---\ntype: video-note\ntags: [youtube,ai  - intel-pipeline\n"
             "  - github\n---\n\nbody prose\n")
    _write(tree, "knowledge/glued.md", glued)
    _, out = _run(monkeypatch, capsys, okf_migrate, "--root", str(tree),
                  "--dir", "knowledge", "--repair-only", "--apply")
    assert "stranded     : 1" in out, out
    assert "knowledge/glued.md" in out, out
    assert (tree / "knowledge/glued.md").read_text(encoding="utf-8") == glued, \
        "the refused file must be byte-identical"
    assert "repaired     : 2" in out, out


def test_the_migrator_and_the_validator_name_the_same_files(tree, listed, monkeypatch,
                                                           capsys):
    # The process boundary this item is about: two scripts, one detector, one set.
    # If these ever part company the gate greenlights a corruption the repairer
    # then silently rewrites — which is precisely the state the item describes:
    # the validator reporting 0 violations while the repairer reported
    # `repaired: 0` over files a hand walk could name. The allow-list is empty
    # here because it is a *counting* policy, covered by its own tests above; this
    # test is about which files each script can see at all.
    listed()
    _, vout = _run(monkeypatch, capsys, validate_okf, "--root", str(tree))
    _, mout = _run(monkeypatch, capsys, okf_migrate, "--root", str(tree),
                   "--dir", "knowledge", "--repair-only")
    flagged = {p.name for p in (tree / "knowledge").glob("*.md")
               if okf_stranded.find_stranded_frontmatter(
                   p.read_text(encoding="utf-8"))}
    assert flagged == {"stranded.md", "legacy.md"}, flagged
    assert all(f"knowledge/{n}:" in vout for n in flagged), vout
    assert all(f"repaired knowledge/{n}" in mout for n in flagged), mout


def test_a_scoped_scan_never_claims_a_file_it_did_not_look_at_is_fixed(tree, listed,
                                                                      monkeypatch,
                                                                      capsys):
    # The stale set is an observation about the LIST, so its denominator has to be
    # the tree the list was generated from. Under `--dir knowledge` a listed path
    # in backlog/ is simply not in the scan, and reporting it as "no longer
    # stranded" would credit #478's data fix with work nobody did — a verdict the
    # scan cannot support, and the zero-denominator shape this change exists to
    # avoid rather than add.
    listed("knowledge/legacy.md", "knowledge/stranded.md",
           "backlog/never-scanned.md")
    code, out = _run(monkeypatch, capsys, validate_okf, "--root", str(tree),
                     "--dir", "knowledge")
    assert "no longer stranded" not in out, out
    assert "known-stranded : 2" in out, out
    assert code == 0, out


def test_a_full_vault_scan_does_report_a_listed_file_it_no_longer_sees(tmp_path,
                                                                      listed,
                                                                      monkeypatch,
                                                                      capsys):
    # The other side of that scope: when the whole vault IS the scan, a listed path
    # the detector no longer flags is #478 progress, and the list going stale is
    # the thing a reader needs to be told.
    listed("knowledge/clean.md")
    monkeypatch.setattr(validate_okf, "VAULT_ROOT", tmp_path)
    code, out = _run(monkeypatch, capsys, validate_okf)
    assert "1 allow-listed file(s) no longer stranded" in out, out
    assert "knowledge/clean.md" in out, out
    assert code == 0, out
