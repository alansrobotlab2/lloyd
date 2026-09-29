"""Step 2.5's `field("…")` labels must be headers the stranded linker prints.

`~/obsidian/skills/autonomy-data-pipeline/SKILL.md` tells a task #24 run to read
the linker's own report back out of its log and into an artefact. The recording
block at `SKILL.md:489-492` (vault commit `46ac2fe6`, 2026-09-28) spells four
labels as string literals:

    "fact_holding_entities": field("fact-holding entities"),
    "degree_zero_candidates": field("degree-zero candidates"),
    "of_which_embed_a_name": field("of-which embed a name"),
    "proposed_edges": field("proposed edges"),

`field` looks for its label in the log text, and a label that is not there yields
the string `"unread"`, which the block then reports as a FAILED step. Only one
side of that pair was pinned: `tests/test_link_stranded_entities.py` asserts
`_shown_number(out, "of-which embed a name")` and friends at `:396`, `:425` and
`:439` against the linker's OWN counts, and that file never opens the vault (0
hits for `obsidian`, `SKILL` or `home()`). So renaming a header in
`scripts/memory/link_stranded_entities.py:247-252`, or a run re-transcribing the
block from memory instead of from its dispatch prompt, minted `UNREAD` on every
run with nothing red anywhere. That is #1809, split from #1805 on 2026-09-29.

Two corrections the triage measured, both of which shape this file:

* **Subset, never equality.** `report_lines` returns SIX labels today — the four
  above plus `skipped: no entities row` and `skipped: edge already live` — and the
  block deliberately records only the four. "The skill's labels equal
  `report_lines`'" is therefore unsatisfiable on the current tree; the assertion
  below is one-directional.
* **Keys are not labels, and pinning keys would be red on arrival.** The linker's
  own keys are `embed_a_name` and `proposed` (`link_stranded_entities.py:250-252`);
  the artefact's are `of_which_embed_a_name` and `proposed_edges`. Only the printed
  labels are shared, so this file compares labels and never keys — which is why
  nothing here reads the block's dict keys.

What the 2026-09-29 symptom was NOT, recorded here so nobody re-reads the
transcripts wrong: task #24 runs printed `STRANDED step 2.5 UNREAD:
['of_which_embed_a_name']` (`20260928_170406_autonomy_4e53` twice,
`20260929_012730_autonomy_e019` four times) while their injected `messages[0]`
each carried `field("of-which embed a name")` once and the underscored form zero
times. The underscored spelling in that output is not evidence of a re-transcribed
label: the diagnostic prints the artefact *key* (`missing = [k for k, v in
out.items() if v == "unread"]`, `SKILL.md:496-497`), which is underscored in the
correct block too. So this file pins the drift half — a rename now breaks a test
instead of silently producing `UNREAD` — and leaves the residue (e019 read a
484-character log at msg54 and the same `run_utc` read `embed=18` at msg57) to the
ordering ruling #1809 reports as owed.

These assertions read the live vault, so they can go red from a nightly skills
pass rather than from the change under review. That is the same trade
`tests/test_yaml_fix_skill_claims.py` makes, and it is deliberate: the gate runner
hardcodes `-m "not live_vault"` (`scripts/automod/gate.py:327`), so a marked check
is deselected from the very run meant to enforce it and pins nothing. Nothing here
skips either — with no vault the reads fail naming the path they looked in, after
`NO_VAULT` in that file's `:96`, and the vault is resolved as
`Path.home() / "obsidian" / "skills"` rather than through `app.paths`, which
re-anchors to a round's worktree where the skills tree does not exist.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "memory" / "link_stranded_entities.py"
THIS_FILE = "tests/test_skill_step25_label_pin.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Never `app.paths.VAULT_ROOT`: that re-anchors to a round's worktree, and the
# skills tree lives in the account's vault, which no round has.
SKILLS_DIR = Path.home() / "obsidian" / "skills"
SKILL_PATH = SKILLS_DIR / "autonomy-data-pipeline" / "SKILL.md"

STEP_HEADING = "## Step 2.5:"
FIELD_RE = re.compile(r'field\("([^"]+)"\)')
HEADING_RE = re.compile(r"^## ", re.M)

# Four denominators the block reads, measured at `SKILL.md:489-492` on
# 2026-09-29. Not a list of labels — the labels themselves must come from the
# linker and the vault file; this is only a floor on the block still recording
# anything, so an extraction that silently returns nothing cannot pass.
RECORDED_FIELDS = 4

# Substring of the one assertion that reports drift. Every node that expects drift
# asserts this phrase too: without it a mutation that makes the drift check vacuous
# still goes green, because a *different* assertion's message happens to name the
# same label. Pinning the phrase is what makes those nodes fail for the right
# reason.
DRIFT_MARK = "NO HEADER FOR"


def _missing_tree(path: Path) -> str:
    """Fail-loud message, after `NO_VAULT` in tests/test_yaml_fix_skill_claims.py."""
    return (f"no skills tree at {path}: these assertions exist to police that "
            "tree, so they report rather than skip")


def _skill_text(path: Path) -> str:
    assert path.is_file(), _missing_tree(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _read_skill() -> str:
    return _skill_text(SKILL_PATH)


def _step_2_5_block(text: str) -> str:
    """The Step 2.5 section: its `## Step 2.5:` heading up to the next `## ` one.

    Scoped to the section so the extraction cannot quietly widen to the rest of a
    958-line skill and start pinning somebody else's `field(` prose, and so a
    deleted or duplicated heading is a loud failure instead of an empty label list
    that would satisfy a subset check by having nothing to say.
    """
    starts = [m.start() for m in HEADING_RE.finditer(text)]
    heading = [s for s in starts if text[s:].startswith(STEP_HEADING)]
    assert len(heading) == 1, (
        f"expected exactly one {STEP_HEADING!r} section in {SKILL_PATH}, found "
        f"{len(heading)} — the block this file pins has moved, gone, or been "
        "duplicated, and an empty extraction would pass the subset check")
    start = heading[0]
    following = [s for s in starts if s > start]
    return text[start:following[0]] if following else text[start:]


def _skill_labels(text: str) -> list[str]:
    """Every `field("<label>")` literal inside the Step 2.5 section."""
    labels = FIELD_RE.findall(_step_2_5_block(text))
    assert len(labels) == RECORDED_FIELDS, (
        f"the Step 2.5 block reads {len(labels)} field(\"…\") labels, expected "
        f"{RECORDED_FIELDS} as measured at {SKILL_PATH}:489-492 — a block that "
        "records a different number of denominators needs this pin re-measured, "
        "not a subset check over a list that may be empty")
    return labels


def _reported(module) -> tuple[list[str], list[str]]:
    """(printed labels, report keys) straight out of the linker's own table."""
    rows = module.report_lines({})
    return [row[0] for row in rows], [row[1] for row in rows]


def _assert_labels_are_printed(labels: list[str], reported: list[str],
                              *, source: str) -> None:
    """Each label the skill reads must be a header the linker prints. One way.

    Deliberately the only assertion here, and the only failure that carries
    `DRIFT_MARK`: an earlier version also asserted strict subset-ness, and a node
    that expected drift passed on that second assertion's message with the drift
    check deleted — a test that cannot fail. Equality is not asserted in either
    direction: `report_lines` carries rows the block does not read today.
    """
    drift = [label for label in labels if label not in reported]
    assert not drift, (
        f"{DRIFT_MARK} {drift!r}: {source} reads label(s) "
        f"{labels!r} but the linker prints {reported!r}. Fix the label in the "
        "skill or the header in scripts/memory/"
        "link_stranded_entities.py::report_lines — a mismatch reaches the run as "
        "the string 'unread', which the block then reports as a FAILED step with "
        "the artefact key, not the label, in it")


def test_extraction_stops_at_the_next_section_and_ignores_later_field_calls():
    """The extraction is scoped, and that scoping can fail.

    Against a fixture doc — not the vault, which happens to contain exactly four
    `field(` literals in TOTAL, so widening the window there is invisible and no
    other node in this file could notice — a `field("…")` in a LATER section must
    not be swept into the Step 2.5 label list. Without this node,
    `_step_2_5_block` could return the whole 958-line skill and every other check
    here would stay green while pinning prose that is not the recording block.
    """
    four = "\n".join(
        f'    "k{i}": field("{lab}"),'
        for i, lab in enumerate(("fact-holding entities", "degree-zero candidates",
                                 "of-which embed a name", "proposed edges")))
    doc = ("# skill\n\n## Step 2: Index Rebuild\n\nwins are fine\n\n"
           f"## Step 2.5: Stranded-entity linking — DRY RUN ONLY\n"
           f"```python\nrec = {{\n{four}\n}}\n```\n\n"
           '## Step 3: Health snapshot\n\nout["x"] = field("decoy from a later '
           'section")\n')
    labels = _skill_labels(doc)
    assert labels == ["fact-holding entities", "degree-zero candidates",
                      "of-which embed a name", "proposed edges"], labels


def _load_linker():
    """The linker as the tree has it now, loaded the way its own test loads it."""
    spec = importlib.util.spec_from_file_location("link_stranded_entities_pinned",
                                                  SCRIPT)
    assert spec and spec.loader, f"the linker script is gone: {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_renamed_linker(tmp_path: Path, old: str, new: str):
    """A copy of the linker with one printed header renamed, as its own module."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert src.count(old) == 1, (
        f"{old!r} appears {src.count(old)} times in {SCRIPT}; the rename witness "
        "needs exactly one header to rename")
    dest = tmp_path / "linker_renamed.py"
    dest.write_text(src.replace(old, new), encoding="utf-8")
    name = "link_stranded_entities_renamed"
    spec = importlib.util.spec_from_file_location(name, dest)
    assert spec and spec.loader, f"could not load the patched copy at {dest}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ── the seam: skill literal ↔ linker header ──────────────────────────

def test_every_step_2_5_field_label_is_a_header_the_linker_prints():
    """Clause 1: the four labels in the live skill are among the linker's six.

    Both sides are measured here, not transcribed: the labels come out of the vault
    file's Step 2.5 section by regex, the headers out of `report_lines({})` on the
    tree's own linker.
    """
    labels = _skill_labels(_read_skill())
    reported, _keys = _reported(_load_linker())
    _assert_labels_are_printed(labels, reported, source=str(SKILL_PATH))
    assert len(labels) == RECORDED_FIELDS and len(reported) > len(labels), (
        f"expected the block to read {RECORDED_FIELDS} of the linker's printed "
        f"headers and leave some unread, got {len(labels)} of {len(reported)}")


def test_renaming_a_header_in_the_linker_alone_reddens_the_check(tmp_path):
    """Clause 2: the expected labels come from `report_lines`, not a literal list.

    The linker copy in `tmp_path` differs from the tree's in exactly one printed
    header — the skill file is untouched — and the same comparison goes red naming
    that header. Had this file carried the expected labels as its own list, the
    rename would have been invisible to it, which is the drift #1809 is about.
    """
    patched = _load_renamed_linker(
        tmp_path, '("of-which embed a name", "embed_a_name"',
        '("of-which embeds a name", "embed_a_name"')
    labels = _skill_labels(_read_skill())
    patched_labels, _keys = _reported(patched)
    assert "of-which embed a name" not in patched_labels
    with pytest.raises(AssertionError) as live:
        _assert_labels_are_printed(labels, patched_labels, source=str(SKILL_PATH))
    assert DRIFT_MARK in str(live.value)
    assert "of-which embed a name" in str(live.value)
    # The tree's own linker still prints the unrenamed header, so the red came from
    # the patched table and not from the skill having drifted underneath it.
    assert "of-which embed a name" in _reported(_load_linker())[0]


def test_a_re_transcribed_label_in_the_block_is_rejected_naming_that_label(tmp_path):
    """Clause 3: the 2026-09-29 symptom, reproduced against a patched COPY.

    `field("of-which embed a name")` rewritten to `field("of_which_embed_a_name")`
    is the underscored form a re-transcribed block would carry — the shape that
    reads as `UNREAD` on every run. The vault file is not written to: the patched
    text lives under `tmp_path`, and the vault bytes are re-read at the end to
    prove this node edited nothing.
    """
    before = SKILL_PATH.read_bytes()
    text = _read_skill()
    assert 'field("of-which embed a name")' in text
    drifted = text.replace('field("of-which embed a name")',
                           'field("of_which_embed_a_name")')
    assert drifted != text, "the anchor this node rewrites is gone from the skill"
    copy = tmp_path / "SKILL-drifted.md"
    copy.write_text(drifted, encoding="utf-8")

    with pytest.raises(AssertionError) as drift:
        _assert_labels_are_printed(
            _skill_labels(copy.read_text(encoding="utf-8")),
            _reported(_load_linker())[0], source=str(copy))
    message = str(drift.value)
    assert DRIFT_MARK in message, (
        "the failure must come from the drift assertion itself; an earlier "
        "revision passed this node off a second, unrelated assertion's message "
        "while the drift check was deleted")
    assert "of_which_embed_a_name" in message, (
        "the failure must name the label that drifted, or the run gets a red test "
        "with no idea which literal to fix")
    assert SKILL_PATH.read_bytes() == before, "this node wrote to the vault"


def test_the_pin_compares_printed_labels_and_never_the_report_keys():
    """The out-of-scope half of #1809: keys differ from labels, and that is fine.

    The linker's keys for the two rows in question are `embed_a_name` and
    `proposed`; the artefact's are `of_which_embed_a_name` and `proposed_edges`. If
    anything here ever compared the block's dict keys against the report's key
    field it would be red on today's tree, so this node pins that it does not: the
    keys are measurably different, and the same assertion is still green.
    """
    labels = _skill_labels(_read_skill())
    reported, keys = _reported(_load_linker())
    assert "embed_a_name" in keys and "proposed" in keys, keys
    assert "of_which_embed_a_name" not in keys, (
        "the linker's key changed shape; re-measure this node, and do not "
        "start pinning keys — only the printed labels are shared")
    assert set(labels) & set(keys) == set(), (
        f"a label now equals a key ({set(labels) & set(keys)}), which would let a "
        "key-only rename slip through unpinned")
    _assert_labels_are_printed(labels, reported, source=str(SKILL_PATH))


# ── enforcement: the gate must actually run what is above ────────────

def test_nothing_in_this_file_is_allowed_to_skip():
    """Clause 4, the half no behaviour test can observe: a skipped pin is a pin off.

    Asking pytest itself would mean running this file inside itself, which recurses,
    so this walks the file's own syntax tree instead and rejects every way a node
    here has to opt out — a `pytest.skip(...)`/`skip(...)` call anywhere, or a
    `@pytest.mark.skip`/`skipif` decorator on any node. Prose is not code: the words
    appear in the docstrings above and must not trip this, which is why it is `ast`
    and not a substring search. A mutation that turns the missing-vault assert into
    a skip leaves the suite reporting `passed, 1 skipped` and stops pinning in
    silence, so this is the node that goes red for it.
    """
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", "")
            if name in {"skip", "xfail"}:
                offenders.append(f"call to {name}() at line {node.lineno}")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                text = ast.unparse(dec)
                if "skip" in text:
                    offenders.append(f"{node.name}: @{text} at line {node.lineno}")
    assert not offenders, (
        f"{THIS_FILE} gained a way to opt out ({offenders}): a check that can skip "
        "cannot pin the skill's labels, because the gate reports a skip alongside "
        "its passes (NO_VAULT, tests/test_yaml_fix_skill_claims.py:96)")
    assert "no skills tree at" in Path(__file__).read_text(encoding="utf-8"), (
        "the fail-loud message must name the path it looked in")


def _pytest(extra: list[str]) -> subprocess.CompletedProcess:
    """Run pytest in a child interpreter, the way the gate's tests rung does."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_the_gate_mark_expression_still_collects_every_node_in_this_file():
    """Clause 4: these checks survive the gate's own `-m` expression.

    Asked of pytest itself rather than read out of this file's source: the same
    nodes are collected with and without the mark expression imported from
    `scripts.automod/gate.py` (imported, not copied, so the pin cannot rot when the
    expression changes). A `live_vault` marker added to any node here — the way
    `live_vault` quietly deselected the checks #1809's triage cites as pinning
    nothing — shows up as a node the gate never ran. Collection only: this node
    executes nothing, so it cannot recurse into itself.
    """
    from scripts.automod.gate import TESTS_MARK_EXPR

    def nodes(out: str) -> list[str]:
        return [ln for ln in out.splitlines() if "::" in ln]

    unfiltered = nodes(_pytest(["--collect-only", "-q", THIS_FILE]).stdout)
    gated = nodes(_pytest(["--collect-only", "-q", "-m", TESTS_MARK_EXPR,
                           THIS_FILE]).stdout)
    assert unfiltered, "collection found no nodes at all — the pin is empty"
    assert gated == unfiltered, (
        f"the gate's selection ({TESTS_MARK_EXPR!r}) deselected "
        f"{sorted(set(unfiltered) - set(gated))} of this file's nodes, so they "
        "would enforce nothing at the gate")
    assert len(unfiltered) >= 5, unfiltered


def test_a_missing_skills_tree_fails_naming_the_path_it_looked_in(tmp_path,
                                                                 monkeypatch):
    """Clause 4, other half: no vault is a loud failure, never a silent pass.

    Pointed at a skills tree that does not exist, the read raises with that path in
    the message. `pytest.skip` would look the same as green in a gate report while
    leaving the labels unpinned, which is how `tests/test_yaml_fix_skill_claims.py`
    reasons about `NO_VAULT` and why nothing in this file is allowed to skip.
    """
    import tests.test_skill_step25_label_pin as me

    absent = tmp_path / "obsidian" / "skills" / "autonomy-data-pipeline" / "SKILL.md"
    assert not absent.exists()
    monkeypatch.setattr(me, "SKILL_PATH", absent)
    with pytest.raises(AssertionError) as missing:
        me._read_skill()
    assert str(absent) in str(missing.value)
    assert "no skills tree" in str(missing.value)

    # And on the real tree the same message names the vault path the file expects,
    # so a machine with a moved vault gets an address, not a bare assert.
    assert str(SKILLS_DIR) in _missing_tree(SKILL_PATH)
