"""#1901 — the two nightly skills must route the frontmatter check to the committed
scanner, report the tags half, and no writer may keep emitting the shape they check for.

#1804 shipped `scripts/vault/segment_scan.py`'s tags half (`fc46f7b0`) and stopped there,
so the repo had the check while the instructions that send a run to the check still said
the opposite of the code:

- `skills/nightly-skills-management/SKILL.md` told a run that "`segment_scan.py` has no
  tags test today", and its run-report requirement named only the segment lines — so
  `run_83_20260930_080839.md` printed `total missing: 12` and no tags figure at all, and
  a reader could not tell that the tags half never ran.
- `skills/autonomy-data-pipeline/SKILL.md` routed BOTH halves of its frontmatter step
  through `/tmp/vault_fm_check.py`, a file `/tmp` may delete at any reboot, and told the
  run to "expect that path to already exist, and reuse it" for a count the repo now
  produces.

What is pinned here is the pair at the seam: the sentence in the skill, and the behaviour
the sentence now claims. A wording test alone rots the same way #1745 rotted — the skill
would assert something of `segment_scan.py` that nothing re-checks — so every wording
assertion below sits beside a fixture run of the scanner that proves the claim about the
code is still true.

The skill files are read from the live vault (`~/obsidian/skills/…`), deliberately
unmarked rather than `live_vault`, following
`tests/test_skill_tool_names.py::test_no_active_skill_or_task_names_a_path_absent_from_the_checkout`:
the vault is one shared tree, a round's base cannot make it stale, and the only way this
goes red is a real regression in the instruction a nightly run reads.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

from app import backlog_tags as BT
from scripts.vault import segment_scan as SS
from scripts.vault.validate_okf import STRICT_FM_RE

VAULT = Path.home() / "obsidian"
SKILLS = VAULT / "skills"
SKILLS_MGMT = SKILLS / "nightly-skills-management/SKILL.md"
DATA_PIPELINE = SKILLS / "autonomy-data-pipeline/SKILL.md"

#: The sentence #1745 left behind and #1804 made false. Its presence means a run is
#: being told to skip the tags half of a check that exists.
STALE_CLAIM = "no tags test"


def _section(skill: Path, anchor: str, before: int = 300, after: int = 3000) -> str:
    """The region of a SKILL.md around `anchor`, for an assertion that must not be
    satisfied by a passing mention somewhere else in a 900-line file."""
    text = skill.read_text(encoding="utf-8")
    at = text.index(anchor)
    return text[max(0, at - before):at + after]


# ── what the scanner actually does, so the skills' sentences can be graded ──────

def _fm_of(text: str) -> dict:
    m = STRICT_FM_RE.match(text)
    assert m, "fixture has no strict frontmatter block"
    return yaml.safe_load(m.group(1))


def test_the_scanner_tags_half_names_an_absent_key_an_empty_list_and_a_scalar(tmp_path):
    """The claim both skills now make about the code, measured rather than quoted.

    Four one-file trees, one shape each, and the offender list must be exactly the three
    bad shapes: no `tags` key at all, `tags: []`, and a bare scalar. The control is the
    fourth — a file with one tag, which must NOT be named — because a pattern that has
    never been seen matching cannot be trusted to be failing for a reason (a 0-hit check
    is a green check on a rule that cannot fire).
    """
    def write(name: str, fm: str):
        (tmp_path / "backlog").mkdir(exist_ok=True)
        (tmp_path / "backlog" / f"{name}.md").write_text(
            f"---\ntype: backlog\nsegment: backlog\n{fm}---\n# {name}\n", encoding="utf-8")

    write("no-key", "")
    write("empty-list", "tags: []\n")
    write("bare-scalar", "tags: backlog\n")
    write("one-tag", "tags: [backlog]\n")

    got = SS.scan(tmp_path, ["backlog"], [])["backlog"]["tags"]
    assert sorted(Path(p).stem for p in got) == ["bare-scalar", "empty-list", "no-key"], got


# ── clause 2: the sentence a run reads, and the number it must carry ────────────

def test_no_skill_still_claims_the_scanner_has_no_tags_test():
    """Clause 2 (#1901): the parenthetical is gone from every skill, not just rewritten
    somewhere. Asserted over all active SKILL.md files, because the same sentence had a
    way of being copied: #1745's whole finding was that a `/tmp` check was described as
    the only one in two different skills."""
    assert SKILLS_MGMT.is_file(), SKILLS_MGMT
    hits = sorted(str(p.relative_to(SKILLS)) for p in SKILLS.glob("*/SKILL.md")
                  if STALE_CLAIM in p.read_text(encoding="utf-8"))
    assert hits == [], f"skills still telling a run the tags half does not exist: {hits}"
    # Positive control for the grep: the string the replacement is REQUIRED to introduce.
    # Today this hits only because the rewording landed; an empty corpus would otherwise
    # make the assertion above pass for the wrong reason.
    control = sorted(str(p.relative_to(SKILLS)) for p in SKILLS.glob("*/SKILL.md")
                     if "missing tags" in p.read_text(encoding="utf-8"))
    assert control, "no skill mentions `missing tags` at all — the replacement never landed"


def test_the_skills_management_tags_step_states_the_rule_that_shipped():
    """Clause 2's other half: the replacement says what `fc46f7b0` shipped, at the step
    that decides whether a written block is finished — a parsed list of >= 1, so an absent
    key, `tags: []` and a bare scalar all offend, over the same walk the segment half uses
    including `backlog/`.

    The four facts are asserted as words a run can act on, and the `backlog/` one is
    checked against the scanner's own `EXTRA_DIRS` rather than against the sentence, so the
    skill cannot keep claiming a directory the scan stopped covering.
    """
    from scripts.vault import segment_scan as scanner
    assert "backlog" in scanner.EXTRA_DIRS, scanner.EXTRA_DIRS
    step = _section(SKILLS_MGMT, "`tags:` must be a **non-empty list as written**")
    for phrase in ("segment_scan.py", "at least one item", "absent key", "bare scalar",
                   "backlog/"):
        assert phrase in step, f"the reworded step omits `{phrase}`"
    assert STALE_CLAIM not in step


def test_the_run_report_requirement_names_the_tags_lines_beside_the_segment_lines():
    """Clause 3 (#1901): a report that never printed the tags half has to be visibly
    incomplete. `run_83_20260930_080839.md` is the counterexample — it printed
    `total missing: 12` and an exit status, and carried no tags figure, so nothing in the
    report distinguished "the tags half ran and was clean" from "the tags half never ran".

    The requirement is asserted inside the paragraph that carries it (`and carry its
    output into the run report`), with the per-directory line, the total line, the two
    lines it already named, and the exit status all present. Asserting them near each
    other is the point: a `missing tags` mention in some unrelated section would not make
    an incomplete report recognizable.
    """
    para = _section(SKILLS_MGMT, "and carry its output into the run report", before=600)
    for required in ("missing tags: N", "total missing tags: N", "missing segment: N",
                     "total missing: N", "exit status"):
        assert required in para, f"the run-report requirement omits `{required}`"


# ── clause 4: the /tmp file keeps only what the repo cannot do ─────────────────

def test_the_data_pipeline_routes_the_check_to_the_committed_scanner():
    """Clause 4 (#1901): both halves of the frontmatter step — the pre-count and the
    post-write sweep — run `segment_scan.py`, with `--root` for bytes that have not landed
    yet, and `tags` stays in the required-fields list the step fills.

    `/tmp/vault_fm_check.py` survives on exactly two functions, because the scanner has no
    equivalent of a MUTATION: `insert_key` writes one key and disturbs no other byte, and
    `assert_only_key_changed` proves afterwards that it did. That limit has to be stated on
    the lines that still name the /tmp path, or the next run reads "reuse it" as licence to
    take the count from a file the same run wrote with.
    """
    text = DATA_PIPELINE.read_text(encoding="utf-8")
    assert "segment_scan.py --list" in text, "the scanner is not routed in at all"
    assert "--root" in text, "bytes-before-they-land are still a /tmp-only job"
    step = _section(DATA_PIPELINE, "written once to `/tmp/vault_fm_check.py`")
    assert "segment_scan.py" in step, "the step's own preamble does not name the scanner"
    for mutator in ("insert_key", "assert_only_key_changed"):
        assert mutator in step, f"{mutator} is no longer described where /tmp is"
    assert "mutator" in step.lower(), "the /tmp limit is not stated on those lines"
    # Pinned to the line, not to the section: a passing mention of `tags` anywhere in a
    # 2,600-character window would satisfy the loose form, and this is the clause that
    # says `tags` is still a REQUIRED field rather than merely a discussed one.
    lines = [ln for ln in text.splitlines() if "Required fields:" in ln]
    assert len(lines) == 1, lines
    assert re.search(r"Required fields:\s*`tags` \(array\)", lines[0]), lines[0]


def test_the_data_pipeline_no_longer_tells_a_run_to_take_its_count_from_tmp():
    """The two lines the item names — `:698` ("written once to `/tmp/vault_fm_check.py`")
    and `:706` ("Expect that path to already exist, and reuse it") — must no longer read as
    an instruction to obtain a CHECK from a scratch file.

    Asserted as an absence with a reason attached, not as a diff: the file may still mention
    the path (the mutators live there), it may not send a count there.
    """
    step = _section(DATA_PIPELINE, "written once to `/tmp/vault_fm_check.py`")
    forbidden = ["used two ways", "for the pre-write check, and run as a script"]
    for phrase in forbidden:
        assert phrase not in step, f"the preamble still says `{phrase}`"
    reused = _section(DATA_PIPELINE, "Expect that path to already exist, and reuse it",
                      before=80)
    assert "mutator" in reused[:260].lower(), \
        "the reuse instruction still does not say what may be reused"


# ── clause 5: the files the unfixed writer already produced ────────────────────

SEEDED = ("1811", "1836", "1855", "1882")


def test_the_four_backlog_items_the_writer_left_tagless_now_carry_a_tags_list():
    """Clause 5 (#1901): the four items `scripts/automod/backlog.py::new_item` filed with
    no `tags` key parse to a non-empty list, each fitted to its own subject rather than
    stamped with the writer's default.

    Judged with the scanner's own decision over the live vault, and named by id: a clause
    that said "only #1887's three witness files remain" would be broken by the next item
    the loop files, which is the writer's clause-1 job and not this tree's state.
    """
    offenders = {Path(p).name for p in SS.scan(VAULT, ["backlog"], [])["backlog"]["tags"]}
    for id_ in SEEDED:
        named = [p for p in (VAULT / "backlog").glob(f"{id_}-*.md")]
        assert len(named) == 1, f"expected exactly one backlog item {id_}: {named}"
        assert named[0].name not in offenders, \
            f"{named[0].name} is still missing a usable tags list"
        fm = _fm_of(named[0].read_text(encoding="utf-8"))
        tags = fm.get("tags")
        assert isinstance(tags, list) and tags, f"{named[0].name}: tags is {tags!r}"
        assert tags != list(BT.DEFAULT_NEW_TASK_TAGS), \
            f"{named[0].name} carries the writer's default, not a lineage set"


def test_a_backlog_tree_with_any_offender_exits_non_zero_and_a_clean_one_exits_zero(tmp_path):
    """Clause 5's other half, in a fixture rather than on the live vault.

    The item's acceptance says the scan "still exits 1" because #1887's three witness files
    are left as offenders. Pinned that way it would go red the day #1887 fixes them, which
    is a guard blocking the wrong thing. So what is pinned is the RULE that produced the 1:
    one remaining offender in a tree is enough for exit 1, and none is exit 0 — with the
    control file proving the named offender was named by the tags half and not by luck.
    """
    script = Path(SS.__file__).resolve()

    def tree(with_offender: bool) -> Path:
        root = tmp_path / ("dirty" if with_offender else "clean")
        (root / "backlog").mkdir(parents=True)
        (root / "backlog" / "good.md").write_text(
            "---\ntype: backlog\nsegment: backlog\ntags: [backlog]\n---\n# good\n",
            encoding="utf-8")
        if with_offender:
            (root / "backlog" / "witness.md").write_text(
                "---\ntype: note\nsegment: backlog\n---\n# witness\n", encoding="utf-8")
        return root

    for with_offender, want in ((True, 1), (False, 0)):
        proc = subprocess.run(
            [sys.executable, str(script), "--root", str(tree(with_offender)), "--list"],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == want, (with_offender, proc.stdout, proc.stderr)
        if with_offender:
            assert "missing tags: 1" in proc.stdout, proc.stdout
            assert "witness.md" in proc.stdout, proc.stdout
