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
hardcodes `-m TESTS_MARK_EXPR` where that is `"not live_vault and not
fault_injection"` (`scripts/automod/gate.py:471`), so a marked check is deselected
from the very run meant to enforce it and pins nothing. Nothing here
skips either — with no vault the reads fail naming the path they looked in, after
`NO_VAULT` in that file's `:96`, and the vault is resolved as
`Path.home() / "obsidian" / "skills"` rather than through `app.paths`, which
re-anchors to a round's worktree where the skills tree does not exist.
"""

from __future__ import annotations

import importlib.util
import os
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


# ── the diagnostic: what a run SEES when a denominator is unread ─────
#
# Everything above pins that a label and a header agree. It cannot say anything about
# the line a run reads when they do not, which is the half #1917 is about: the block
# used to print `UNREAD: ['of_which_embed_a_name']`, the artefact's dict key, and two
# task #24 runs on 2026-09-30 read that underscored spelling as evidence their own
# dispatched prompt had been re-transcribed — a claim that was false, because the key
# is underscored in the CORRECT block too. So the helpers below do not read the skill's
# prose: they extract the Python the Step 2.5 heredoc actually runs and run it over a
# log, because the artefact under test is the printed line.

HEREDOC_OPEN = "<<'EOF'\n"
HEREDOC_CLOSE = "\nEOF"
KEY_FIELD_RE = re.compile(r'^\s*"(\w+)":\s*field\("([^"]+)"\)', re.M)
UNREAD_PREFIX = "STRANDED step 2.5 UNREAD"
# Any report of which header lines matched, printed as part of the unread diagnostic.
# Deliberately NOT a match on the block's present wording (`… UNREAD HEADERS-MATCHED:`):
# clause 2 asks that the report EXIST, and a diagnostic headed `UNREAD matched-headers:`
# reports the same thing, so matching the literal string would have this node reject an
# improvement to the wording it is supposed to be checking. What stays pinned is the
# behaviour: the report is inside the unread diagnostic, where whoever reads `UNREAD` is
# looking, and the matched headers and their count are asserted on whatever line carries it.
MATCHED_REPORT_RE = re.compile(r"\bUNREAD\b.*MATCH", re.I)
LABEL_SHOWN_RE = re.compile(r"label='([^']*)'")
KEY_SHOWN_RE = re.compile(r"key='([^']*)'")


def _matched_report_lines(stdout: str) -> list[str]:
    """The line(s) of the unread diagnostic that report which header lines matched.

    Detected by `MATCHED_REPORT_RE`, not by the block's current wording, for the reason
    recorded there: clause 2 pins that the report is made, not the word it is headed by.
    """
    return [ln for ln in stdout.splitlines() if MATCHED_REPORT_RE.search(ln)]


def _skill_rows(text: str) -> list[tuple[str, str]]:
    """(artefact key, searched label) pairs, in the order the block writes them.

    One regex over one source line, so a key is only ever paired with the label its own
    `field(` call passed. Pairing them any other way — two lists, zipped — would let the
    test assert a key/label pair the block never produced, which is the exact confusion
    the diagnostic exists to remove.
    """
    rows = KEY_FIELD_RE.findall(_step_2_5_block(text))
    assert len(rows) == RECORDED_FIELDS, (
        f"paired {len(rows)} (key, field(\"…\")) rows on one line each, expected "
        f"{RECORDED_FIELDS} — the block's keys and labels can only be read as a pair "
        "while they are written on one line, so this node needs re-measuring if that "
        "shape changed rather than a looser regex")
    assert [lab for _k, lab in rows] == _skill_labels(text), (
        "the paired labels and the label-only extraction disagree, so one of the two "
        "regexes is reading a different part of the block")
    return rows


def _recording_snippet(text: str) -> str:
    """The Python the Step 2.5 heredoc runs, extracted verbatim between its markers."""
    block = _step_2_5_block(text)
    start = block.index(HEREDOC_OPEN) + len(HEREDOC_OPEN)
    end = block.index(HEREDOC_CLOSE, start)
    src = block[start:end]
    assert "def field(" in src, (
        "the text between the heredoc markers is not the recording block; the markers "
        "or the block moved and extracting blind would run arbitrary skill prose")
    return src


def _drive_recording_block(tmp_path: Path, headers: list[tuple[str, str]],
                           *, snippet: str | None = None,
                           ts: str = "2026-09-30") -> str:
    """Run the skill's own recording block over a log; return its stdout.

    `headers` is the (label, value) table the fake log is built from — labels come out
    of `report_lines`, never a hand-typed spelling — and `HOME` is aimed at `tmp_path`,
    because the block resolves BOTH its log and the artefact it writes off
    `Path.home() / "lloyd-data/_pipeline"`. A fixture that let that resolve to the
    account's home would have the suite writing into the pipeline directory the nightly
    job owns.
    """
    src = _recording_snippet(_read_skill()) if snippet is None else snippet
    home = tmp_path / "home"
    base = home / "lloyd-data" / "_pipeline"
    base.mkdir(parents=True, exist_ok=True)
    body = ["Stranded-entity linker — #1019 (dry-run)"] + [
        f"  {label}  {value}" for label, value in headers]
    (base / f"stranded-entities-{ts}.log").write_text("\n".join(body) + "\n",
                                                      encoding="utf-8")
    script = tmp_path / "recording_block.py"
    script.write_text(src, encoding="utf-8")
    run = subprocess.run([sys.executable, str(script), ts], cwd=str(tmp_path),
                         env={**os.environ, "HOME": str(home)},
                         capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, (
        f"the recording block exited {run.returncode} over a well-formed log; its "
        f"stderr is the block's own traceback:\n{run.stderr}")
    return run.stdout


def _unread_lines(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines()
            if ln.startswith(UNREAD_PREFIX) and not MATCHED_REPORT_RE.search(ln)]


def _fields_shown(line: str) -> tuple[str, str]:
    """(label shown, key shown) off one UNREAD line, or a failure naming the line."""
    lab, key = LABEL_SHOWN_RE.search(line), KEY_SHOWN_RE.search(line)
    assert lab and key, (
        f"the UNREAD line must carry the searched label and the artefact key as two "
        f"named fields on the SAME line — split across lines a reader comparing them "
        f"has to hold one in memory, which is how the two get conflated: {line!r}")
    return lab.group(1), key.group(1)


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


# ── the unread diagnostic itself (#1917) ─────────────────────────────

def test_the_unread_line_names_its_label_key_headers_and_the_msg0_rule_is_indexed(
        tmp_path):
    """#1917 clauses 1 and 2, run through the block rather than read off its prose.

    Two failure modes that used to print the identical underscored key — and did, on
    2026-09-30: `run_24_20260930_052951` (session `20260929_222951_autonomy_7173`, msg48)
    and `run_24_20260930_094319` (`20260930_024319_autonomy_e918`, msg39) both fired
    `UNREAD: ['of_which_embed_a_name']` while their `messages[0]` each carried the four
    correct hyphenated labels, so neither could tell a linker header rename from a
    hand-mis-transcribed block. Here both are driven for real, and the two lines must
    now differ in the one field that separates them:

    * **A linker rename** — the log is built from a copy of the linker with
      `of-which embed a name` renamed to `of-which embeds a name`, block untouched: the
      line's label is the hyphenated one and its key is underscored, so label ≠ key.
    * **A mis-transcribed block** — `field("of-which embed a name")` rewritten to the
      underscored form in a COPY under `tmp_path`, log built from the real linker: the
      label is underscored too, so label = key.

    And in both cases the same run must report which of the four header lines DID match,
    which is what makes a rename (three matched, one absent from the log entirely) look
    different from a transcription error (three matched, and the log's own hyphenated
    spelling sitting right there).

    Clause 4 — the class rule this item writes to `lloyd/MEMORY.md` — is asserted by the
    same node, through `_assert_msg0_rule_is_indexed()`, for a reason that is not tidiness:
    the item's own acceptance check (b) is that this file goes from `8 passed` to
    `9 passed`, so a second node would fail the very check that proves the change, and the
    review rung's refusal on the first gate ("no changed test covers MEMORY.md") has to be
    answered inside the node count the contract fixes. Nor does the vault half inherit a
    gate-run check from elsewhere: the node that validates the live index is
    `tests/test_prompt_surface_budget.py::test_the_live_memory_index_validates` (`:202`),
    decorated `@vault_only` and `@live_vault` (`:200-201`), and the gate runs the suite with
    `TESTS_MARK_EXPR = "not live_vault and not fault_injection"`
    (`scripts/automod/gate.py:471`) — measured, that selection reports
    `40 passed, 8 deselected` for that file and this validator is among the eight. A rule
    can therefore reach a green gate with nothing the gate ran enforcing its shape, which
    is the hole `_assert_msg0_rule_is_indexed()` closes.
    """
    rows = _skill_rows(_read_skill())
    wanted = {"of_which_embed_a_name": "of-which embed a name"}
    subject = [(k, lab) for k, lab in rows if k in wanted]
    assert subject == [("of_which_embed_a_name", "of-which embed a name")], (
        f"the row the 2026-09-30 runs tripped over is not the one this node drives: "
        f"{subject}")
    key, label = subject[0]
    renamed = "of-which embeds a name"

    # ── failure mode A: the linker renamed the header, the block did nothing wrong
    patched = _load_renamed_linker(
        tmp_path, f'("{label}", "embed_a_name"', f'("{renamed}", "embed_a_name"')
    log_rows = [(lab, f"{i + 1},{i + 1}0") for i, lab in enumerate(_reported(patched)[0])]
    out = _drive_recording_block(tmp_path / "rename", log_rows)
    unread = _unread_lines(out)
    assert len(unread) == 1, (
        f"a log with one renamed header must produce exactly one UNREAD line, got "
        f"{unread!r} in:\n{out}")
    shown_label, shown_key = _fields_shown(unread[0])
    assert (shown_label, shown_key) == (label, key), (
        f"the line must show the label it searched ({label!r}) and the artefact key "
        f"({key!r}) as those two things; a reader comparing the two spellings is "
        f"deciding whether to blame the skill or the linker: {unread[0]!r}")
    assert shown_label != shown_key, (
        "under a rename the two spellings must be visibly different — that difference "
        "is the whole diagnostic")
    header_lines = _matched_report_lines(out)
    assert len(header_lines) == 1, (
        f"the unread diagnostic must also say which header lines matched, once: "
        f"{header_lines!r} in:\n{out}")
    matched_text = header_lines[0]
    assert label not in matched_text, (
        f"{label!r} matched nothing in that log, so it must not appear in the matched "
        f"list — the list is what DID match: {matched_text!r}")
    assert f"{len(rows) - 1} of {len(rows)}" in matched_text, (
        f"one label of {len(rows)} went unread, so the line has to say 3 of 4 matched; "
        f"a count is what tells a reader the log was opened at all: {matched_text!r}")
    for _k, lab in rows:
        if lab == label:
            continue
        assert lab in matched_text, (
            f"{lab!r} matched a line in that log, so the diagnostic must account for "
            "it: an unread row that hides what its siblings did match leaves the run "
            "with one number and no idea whether the log was read at all")

    # ── failure mode B: the block was re-transcribed, the log is fine
    drifted = _recording_snippet(_read_skill()).replace(
        f'field("{label}")', f'field("{key}")')
    assert drifted.count(f'field("{key}")') == 1, (
        "the mis-transcription this mode reproduces is one label rewritten; if the "
        "replacement did not land exactly once the two modes are not comparable")
    good_rows = [(lab, f"{i + 1},{i + 1}0")
                 for i, lab in enumerate(_reported(_load_linker())[0])]
    out_b = _drive_recording_block(tmp_path / "drift", good_rows, snippet=drifted)
    unread_b = _unread_lines(out_b)
    assert len(unread_b) == 1, (
        f"a block searching one underscored label must produce exactly one UNREAD "
        f"line, got {unread_b!r} in:\n{out_b}")
    shown_label_b, shown_key_b = _fields_shown(unread_b[0])
    assert (shown_label_b, shown_key_b) == (key, key), (
        f"a hand-mis-transcribed block prints the SAME spelling twice — label and key "
        f"both underscored — which is how it is told apart from a rename: {unread_b[0]!r}")
    assert shown_label_b == shown_key_b, (
        "the discriminator is exactly this equality versus A's inequality")
    matched_b = _matched_report_lines(out_b)
    assert len(matched_b) == 1, (
        f"the unread diagnostic must print its matched-header list exactly once: "
        f"{matched_b!r} in:\n{out_b}")
    assert key not in matched_b[0] and all(
        lab in matched_b[0] for _k, lab in rows if lab != label), (
        f"the mis-transcribed label found no header and the other three did, so the "
        f"matched list is the evidence that the log itself was readable — the run "
        f"should conclude its own spelling moved, not the linker's: {matched_b[0]!r}")
    assert unread[0] != unread_b[0], (
        "the two failure modes must not print the same line — that sameness is the bug "
        "this node exists to keep fixed")

    _assert_msg0_rule_is_indexed()


# ── the vault half of #1917: the class rule this item writes ─────────

# ── the vault half: the class rule this item writes (clause 4) ───────

MEMORY_INDEX = Path.home() / "obsidian" / "lloyd" / "MEMORY.md"
MSG0_RULE = re.compile(r"messages\[0\]")
RULE_RUNS = ("run_24_20260930_052951", "run_24_20260930_094319")
VALIDATOR = ROOT / "scripts" / "memory" / "validate_memory_index.py"


def _assert_msg0_rule_is_indexed() -> None:
    """Clause 4, asserted from the one node this diff adds. See that node's docstring
    for why the vault's own rule shape is checked here and not in a node of its own.
    """
    assert MEMORY_INDEX.is_file(), _missing_tree(MEMORY_INDEX)
    lines = MEMORY_INDEX.read_text(encoding="utf-8", errors="replace").splitlines()
    rule = [ln for ln in lines
            if ln.startswith("- ") and MSG0_RULE.search(ln)]
    assert len(rule) == 1, (
        f"the index must carry the msg0 rule exactly once; {len(rule)} top-level lines "
        f"name `messages[0]`, and two copies means one of them is the stale one that a "
        f"later reader trusts: {rule}")
    line = rule[0]

    assert line.startswith("- [feedback] "), (
        f"an untyped index line fails `check()` and is invisible to the reader who scans "
        f"by type — a ruling is feedback, not project state: {line[:24]!r}")
    assert "(2026-09-30)" in line, (
        f"the clause is dated the day the #1809 ruling ordered it; an undated rule "
        f"cannot be aged out or checked against its incident: {line[:36]!r}")
    for run in RULE_RUNS:
        assert run in line, (
            f"the ruling names {run} as one of the two runs that asserted a prompt "
            "contained something its own `messages[0]` disproved; a rule citing neither "
            "is advice, and the incident is what makes it checkable")

    at = lines.index(line)
    heading = next(l for l in reversed(lines[:at]) if l.strip())
    assert heading.startswith("## "), (
        f"the rule is meant to sit under a heading of its own (the 2026-09-22 "
        f"convention MEMORY.md states at its `## How a class rule gets added` line), not "
        f"be appended to whatever section was last: {heading!r}")
    stem = re.sub(r"\s*\((?:added|updated) \d{4}-\d\d-\d\d\)\s*$", "", heading[3:])
    assert not re.search(r"\d{4}-\d\d-\d\d", stem), (
        f"{heading!r} is a DATE-named heading once its parenthetical is stripped, and "
        "clause 4 names the alternative: a heading named for the RULE with the date "
        "inside it, because a date-named heading cannot be found by what it says")

    spec = importlib.util.spec_from_file_location("validate_memory_index_pinned",
                                                  VALIDATOR)
    assert spec and spec.loader, f"the index validator is gone: {VALIDATOR}"
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)
    from app import memory_ceiling as mc
    from app import prompt_surface as ps

    report = validator.check(MEMORY_INDEX.parent, ceiling=ps.MEMORY_MD_CEILING_BYTES)
    assert report["ok"], (
        f"the live index fails its own validator ({report['errors']}), and this is the "
        "node that was supposed to see it before the gate did — the `live_vault` sibling "
        f"reports `1 deselected` on a normal round: {report}")

    assert report["memory_md_bytes"] <= ps.MEMORY_MD_CEILING_BYTES
    # The 300-character bound the clause names, pinned as a NUMBER. `check()` above
    # enforces whatever `INDEX_LINE_MAX_CHARS` currently says, so if that constant moved
    # the rule would be measured against something else and this node — the one standing
    # in for a `live_vault`-marked validator the gate does not run — would stay green
    # while the clause it cites said 300. Not a second measurement of the line, which
    # `check()` already made: a pin on the bound the clause depends on.
    assert mc.INDEX_LINE_MAX_CHARS == 300, (
        f"the clause's limit is 300 characters; the enforced cap is now "
        f"{mc.INDEX_LINE_MAX_CHARS}, so either the bound moved or the clause was "
        "rewritten without this node")
    slugs = validator.links(line)
    assert slugs, (
        "the line has to point at a topic file — that pointer is what keeps one index "
        "line possible, and `check()` cannot fail on an index line that never links")
    detail = MEMORY_INDEX.parent / mc.TOPICS_SUBDIR / f"{slugs[0]}.md"
    # `check()` also reports a dangling link; this one exists so the failure names the
    # FILE the detail should be sitting in, rather than quoting a link string.
    assert detail.is_file(), f"the linked topic file is missing: {detail}"
    body = detail.read_text(encoding="utf-8")
    for token in (*RULE_RUNS, "messages[0]", "grepping", "retract"):
        assert token in body, (
            f"the index line is the hook and this file is the detail; {token!r} is part "
            f"of what makes the ruling re-checkable by a reader who never sees the "
            f"ledger: {detail}")


#


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
    # The floor is this file's own node count, raised when a node is added — not a round
    # number sitting below it. A ">= 5" against a file of ten stays green while five
    # nodes quietly stop being collected, and quiet non-collection is the exact rot this
    # file exists to catch (it is how #1809's cited pins stopped enforcing). The coupling
    # is the mechanism: adding a node here means editing this number, so a node cannot be
    # added without being noticed and cannot be lost without going red.
    assert len(unfiltered) >= 9, (
        f"{len(unfiltered)} nodes collected; 9 were here when #1917 landed — the count "
        "is the item's acceptance check (b), `8 passed` -> `9 passed`, so a node lost "
        f"here is a check that no longer holds: {sorted(unfiltered)}")

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
