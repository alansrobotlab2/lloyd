"""#2317: a task description that quotes a constant's OLD value is caught.

`autonomy/79-retention-sweep.md` states its script's windows as numbers hand-copied
out of `scripts/groundskeeper/retention-sweep.py`. `01dea8bc` moved
`LEDGER_ARCHIVE_AGE_DAYS` from 30 to 14 and the description went on saying the old
number for three weeks, which was the third repair of that one file's prose (#1573,
#1734, #2098) with no guard after any of them. These nodes pin the guard's resolver.

Two rules these nodes hold themselves to, both from how this kind of check has failed
here before:

* **The red→green pair lives in the default selection.** #2036's lesson, written into
  `scripts/automod/vault_guards.py`: `agreement()` is a pytest-delta probe, so it
  refuses a land only when some node already reads BOTH sides — and for this drift no
  node read the description's numbers at all, which is precisely why the stale prose
  landed unchallenged. `test_the_gate_selection_collects_both_halves_of_the_pair`
  below is the node that keeps that true, because #1276's lesson is the other half:
  a red test that lives outside the gate's `-m "not live_vault"` selection enforces
  nothing.
* **No node pins a number the tree owns.** `01dea8bc` is also a constant-MIRROR trap:
  a test that hard-codes `LEDGER_ARCHIVE_AGE_DAYS = 14` goes red the day someone
  legitimately moves it (#1050, and the mirror detector in `scripts/automod/review.py`
  exists to catch exactly that). So every expected number below comes from
  `FIXTURE_TREE` — a tree this file invents, in the shape `tree_constants()` returns —
  and the one node that reads the real tree asserts the pair count is non-zero, never
  a value.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.automod import constant_quotes as CQ

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "constant_quotes"
# The checkout these tests are RUNNING in (#1750), not a main checkout: a round that
# moves `LEDGER_ARCHIVE_AGE_DAYS` must see its own round, and `tree_constants()` of
# the live `~/lloyd` would answer about the tree before the round instead.
REPO = str(HERE.parent)

#: A tree of this file's own making, in the shape `tree_constants()` returns: name to
#: value. It is a module dict rather than a JSON file on purpose — `*.json` is ignored
#: repo-wide (`.gitignore`'s blanket rule, with per-directory negations as the only
#: exception), so a committed `tree.json` here would exist only in the worktree that
#: wrote it, and every node that read it would ERROR in the gate's clean checkout while
#: the suite stayed green. #2317's first gate learned that the hard way.
#:
#: The VALUES are this file's, not the tree's: no node here asserts what
#: `LEDGER_ARCHIVE_AGE_DAYS` equals in `retention-sweep.py`, because that number is
#: somebody else's to move (#1050). They are chosen to keep the three stores the real
#: file contrasts distinguishable from each other (90, 30, 14, 7), so an inverted
#: binding shows up as a wrong value rather than as a coincidence.
FIXTURE_TREE = {
    "LEDGER_ARCHIVE_AGE_DAYS": 14,
    "WORKTREE_DIR_MAX_AGE_DAYS": 7,
    "BRANCH_MAX_AGE_DAYS": 30,
    "VOICE_TURNS_MAX_AGE_DAYS": 90,
    "BACKGROUND_SESSION_ARCHIVE_AGE_DAYS": 30,
    "SESSION_ARCHIVE_AGE_DAYS": 90,
}


@pytest.fixture
def fixture_tree() -> dict[str, int]:
    """A copy, so a node that mutates the tree cannot reach its neighbours."""
    return dict(FIXTURE_TREE)


@pytest.fixture
def task_description() -> str:
    """The front-matter description of the fixture task file, read the way the
    landing route reads it — body and Activity Log excluded.
    """
    return CQ.front_matter_description((FIXTURES / "retention-task.md").read_text())


def test_the_corrected_fixture_yields_no_mismatch(task_description, fixture_tree):
    """Green half of the pair: every number in the fixture matches the fixture tree.

    The count is asserted alongside the emptiness, because `0 mismatches` is what a
    resolver that read nothing also prints — see
    `test_a_description_naming_no_tree_constant_is_instrument_failure`.
    """
    out = CQ.mismatches(task_description, fixture_tree)
    assert list(out) == []
    assert out.resolved == 4


def test_one_wrong_number_yields_exactly_one_tuple(task_description, fixture_tree):
    """Red half of the pair: the ledger window says 90 where the fixture tree says 14.

    Exactly ONE tuple, not two: the same description quotes three other constants
    correctly and names a fourth number (`the 30 the file stores use`) that is prose
    argument, not a quote — a resolver that reported either of those would make the
    refusal it writes worthless.
    """
    stale = task_description.replace("LEDGER_ARCHIVE_AGE_DAYS (14)",
                                     "LEDGER_ARCHIVE_AGE_DAYS (90)")
    assert stale != task_description, "fixture no longer states the pair this edits"
    out = CQ.mismatches(stale, fixture_tree)
    assert list(out) == [("LEDGER_ARCHIVE_AGE_DAYS", 90,
                          fixture_tree["LEDGER_ARCHIVE_AGE_DAYS"])]
    assert out.resolved == 4, "the three correct quotes are still resolved pairs"


def test_all_three_spellings_in_the_real_file_shape_resolve(task_description,
                                                            fixture_tree):
    """The clause's grammar, spelled out. #79's description states its windows three
    ways, and a resolver that read only one of them would wave through a stale move on
    most of the file it exists to protect: `NAME (N)` alone reaches 3 of the 9 pairs
    that file states (`WORKTREE_DIR_MAX_AGE_DAYS`, `BRANCH_MAX_AGE_DAYS`,
    `LEDGER_ARCHIVE_AGE_DAYS`), and across the whole corpus it is 3 of the 8 pairs the
    resolver resolves today — measured by disabling each layer of this grammar in
    turn, not guessed.
    """
    assert CQ.mismatches(task_description, fixture_tree).names == (
        # NAME (N)
        "LEDGER_ARCHIVE_AGE_DAYS", "WORKTREE_DIR_MAX_AGE_DAYS",
        # NAME (N, prose argument that contains a second number)
        "VOICE_TURNS_MAX_AGE_DAYS",
        # Nd ... (NAME — the number comes first and the name is inside brackets
        "BACKGROUND_SESSION_ARCHIVE_AGE_DAYS")


def test_only_the_first_number_in_a_prose_parenthesis_is_the_value(fixture_tree):
    """`VOICE_TURNS_MAX_AGE_DAYS (90, deliberately not the 30 the file stores use…)`.

    The second number is the argument FOR 90, and the tree's value is 90, so a
    resolver that took the last number in the brackets would report this correct
    description as a mismatch — a false refusal on the file that motivated the guard.
    """
    text = ("VOICE_TURNS_MAX_AGE_DAYS (90, deliberately not the 30 the file stores "
            "use, because a voice turn is small)")
    assert list(CQ.mismatches(text, fixture_tree)) == []


def test_a_day_count_before_the_name_binds_to_it(fixture_tree):
    """`deleted >30d on mtime (BACKGROUND_SESSION_ARCHIVE_AGE_DAYS)` — the number is
    written first and the name is named inside the following brackets, the spelling
    #79 uses for four of its stores. The value here is 30, matching the fixture tree,
    so the pair must RESOLVE (count it) and not mismatch.
    """
    text = "sessions deleted >30d on mtime (BACKGROUND_SESSION_ARCHIVE_AGE_DAYS)"
    out = CQ.mismatches(text, fixture_tree)
    assert list(out) == []
    assert out.resolved == 1


def test_a_number_that_is_not_the_trees_in_that_third_spelling_is_caught(fixture_tree):
    """The same spelling, wrong: a moved constant stated as `>90d` is one mismatch.

    This is the case that would be missed entirely by a `NAME (N)`-only grammar, and
    it is the shape `PROVENANCE_ARCHIVE_AGE_DAYS`, `SPILL_MAX_AGE_DAYS` and
    `GROUNDSKEEPER_QUEUE_MAX_AGE_DAYS` take in the real file — all 30 in the tree
    today, so nothing is wrong now, and nothing would ever be caught either.
    """
    text = "sessions deleted >90d on mtime (BACKGROUND_SESSION_ARCHIVE_AGE_DAYS)"
    out = CQ.mismatches(text, fixture_tree)
    assert list(out) == [("BACKGROUND_SESSION_ARCHIVE_AGE_DAYS", 90,
                          fixture_tree["BACKGROUND_SESSION_ARCHIVE_AGE_DAYS"])]


def test_a_git_tree_is_scraped_from_tracked_files_only(tmp_path):
    """The branch the GATE actually takes, pinned: `tree_constants` of a checkout.

    Every other node here points `tree_constants` at a `tmp_path`, which has no `.git`
    and so walks the directory — while both real callers (`vault_round.validate` via
    `LLOYD_HOME`, and the review rung via the round's worktree) hand it a git tree,
    where the scrape is `git grep` and sees tracked files only. A round that moves a
    constant has committed it, so the two branches agree in practice; this node is what
    makes that an observation rather than an assumption, and it pins the one difference
    between them: a constant sitting in an UNtracked file is invisible to the guard,
    which is how a `.py` written but never committed answers as if it did not exist.
    """
    git = subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)],
                         capture_output=True, text=True)
    assert git.returncode == 0, git.stderr
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.com"],
                   capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"],
                   capture_output=True)
    (tmp_path / "tracked.py").write_text("TRACKED_WINDOW = 21\n")
    (tmp_path / "untracked.py").write_text("UNTRACKED_WINDOW = 22\n")
    committed = subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.py"],
                               capture_output=True, text=True)
    assert committed.returncode == 0, committed.stderr
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "base"],
                   capture_output=True)
    out = CQ.tree_constants(tmp_path)
    assert out.get("TRACKED_WINDOW") == 21, out
    assert "UNTRACKED_WINDOW" not in out, out


def test_a_number_too_far_back_or_behind_a_break_binds_nothing(fixture_tree):
    """Both ways a stray `Nd` could be attributed to the wrong constant, refused.

    80 characters is the real requirement (the live file's longest name-before-number
    span is `30d for a background session (platform autonomy or worker, BACKGROUND_…`
    = 56 characters) and a `)` or `;` between number and name means the number
    belongs to a group that already closed — a store list reads `30d (A); 45d (B)`
    and B must not inherit A's window.
    """
    near = ("deleted >30d on mtime (BACKGROUND_SESSION_ARCHIVE_AGE_DAYS) when idle")
    assert CQ.mismatches(near, fixture_tree).resolved == 1, (
        "the same number, right beside the name, must bind — this node's other half")
    far = ("deleted >30d, and every other store the sweep touches is listed here in "
           "prose before we get to the constant itself "
           "(BACKGROUND_SESSION_ARCHIVE_AGE_DAYS)")
    assert CQ.mismatches(far, fixture_tree).resolved == 0, (
        "one word changed from `near` and the number is 106 characters away: the "
        "window is what separates them, so both halves have to be asserted")
    stored = ("ledger rows past 14d (LEDGER_ARCHIVE_AGE_DAYS); sessions past 90d "
              "(BACKGROUND_SESSION_ARCHIVE_AGE_DAYS)")
    out = CQ.mismatches(stored, fixture_tree)
    assert list(out) == [("BACKGROUND_SESSION_ARCHIVE_AGE_DAYS", 90,
                          fixture_tree["BACKGROUND_SESSION_ARCHIVE_AGE_DAYS"])], (
        "the second store bound the FIRST store's 14 instead of its own 90")
    assert out.resolved == 2, "both pairs resolved; only the second one disagrees"


def test_a_name_written_without_a_number_beside_it_resolves_no_pair(fixture_tree):
    """`older than PROVENANCE_ARCHIVE_AGE_DAYS days` states no value, so nothing here
    can disagree — and the denominator says which case the caller is in, which is the
    difference between this module and a check that silently reads nothing.
    """
    text = "provenance jsonl older than PROVENANCE_ARCHIVE_AGE_DAYS days are removed"
    out = CQ.mismatches(text, fixture_tree)
    assert list(out) == []
    assert out.resolved == 0


def test_a_description_naming_no_tree_constant_is_instrument_failure(fixture_tree):
    """Clause 2: zero resolved pairs is distinguishable from agreement.

    A description that quotes nothing and a description that quotes everything
    correctly both produce zero mismatches; #1691's zero-denominator rule is that
    reading the first as the second is the bug, not an edge case.
    """
    empty = CQ.mismatches("sweep the stores daily and report the counts", fixture_tree)
    assert list(empty) == []
    assert empty.resolved == 0
    assert empty.instrument_failed is True
    assert "0 description/constant pairs resolved" in empty.summary()
    correct = CQ.mismatches(task_description_of(fixture_tree), fixture_tree)
    assert correct.instrument_failed is False
    assert "instrument failure" not in correct.summary()


def task_description_of(fixture_tree) -> str:
    """The corrected fixture description, for the assertion that a REAL check
    reports itself as having run."""
    return CQ.front_matter_description((FIXTURES / "retention-task.md").read_text())


def test_an_empty_description_is_instrument_failure(fixture_tree):
    assert CQ.mismatches("", fixture_tree).instrument_failed is True
    assert CQ.mismatches("LEDGER_ARCHIVE_AGE_DAYS (14)", {}).instrument_failed is True


def test_the_scan_reports_one_denominator_for_the_whole_corpus(fixture_tree):
    """Corpus-level denominator: 2 files × 4 pairs, with the silent file listed.

    `pairs` is what a caller must read before believing `mismatched == []`. The
    `silent` list is the same fact one file down, and is NOT an error: most autonomy
    tasks legitimately name no constant, and a land that touches one of them must not
    be refused for it — which is why the rail returns no error for a silent file and
    the witness node is what reads the total.
    """
    good = task_description_of(fixture_tree)
    stale = good.replace("WORKTREE_DIR_MAX_AGE_DAYS (7)",
                         "WORKTREE_DIR_MAX_AGE_DAYS (30)")
    out = CQ.scan({"autonomy/79-retention-sweep.md": stale,
                   "autonomy/quiet-task.md": "do the thing every morning",
                   "autonomy/also-clean.md": good}, fixture_tree)
    assert out["mismatched"] == [("autonomy/79-retention-sweep.md",
                                  "WORKTREE_DIR_MAX_AGE_DAYS", 30,
                                  fixture_tree["WORKTREE_DIR_MAX_AGE_DAYS"])]
    assert out["pairs"] == 8, "4 pairs from each of the two files that name one"
    assert out["silent"] == ["autonomy/quiet-task.md"]
    assert "8 resolved pair(s)" in out["report"]


def test_a_corpus_that_resolves_nothing_reports_instrument_failure(fixture_tree):
    out = CQ.scan({"autonomy/a.md": "nothing here names a constant",
                   "autonomy/b.md": ""}, fixture_tree)
    assert out["pairs"] == 0
    assert out["mismatched"] == []
    assert CQ.INSTRUMENT_FAILURE in out["report"]


def test_the_body_and_activity_log_are_never_inputs(fixture_tree):
    """The Activity Log line in the fixture file quotes 90 for a 14-day window.

    That number is history — what an old run did — and #2317's clause is that the
    check's only input is the front-matter description. Read the whole file as the
    description and this fixture goes red; read the front matter and it stays green,
    which is the same fact the land-side test states from the other side
    (`test_changing_only_the_activity_log_does_not_refuse_the_land`).
    """
    whole = (FIXTURES / "retention-task.md").read_text()
    assert "LEDGER_ARCHIVE_AGE_DAYS (90)" in whole
    assert list(CQ.mismatches(CQ.front_matter_description(whole),
                              fixture_tree)) == []


def test_front_matter_description_takes_only_the_front_matter():
    assert CQ.front_matter_description('---\ndescription: "kept"\n---\nbody: gone\n')\
        == "kept"
    assert CQ.front_matter_description("# no front matter\n") == ""
    assert CQ.front_matter_description('---\nname: x\n---\nbody\n') == ""
    # A file whose YAML cannot be read is not this check's to judge: the loader's
    # graduated recovery hands back whatever its regex fallback extracted, which
    # quotes no pair, and the landing route's own `frontmatter_error` is the rail
    # that refuses such a file. Two rails refusing one unreadable file is how a
    # check becomes the thing authors route around.
    assert CQ.mismatches(CQ.front_matter_description("---\ndescription: [unclosed\n---\n"),
                         {"LEDGER_ARCHIVE_AGE_DAYS": 14}).resolved == 0


def test_tree_constants_scrapes_an_untracked_directory(tmp_path):
    """The non-git branch, which is also the collision rule's test bed.

    `AMBIGUOUS` is bound to two different values, so which one a description means is
    a question about which module — guessing would refuse a correct description, so
    the name is dropped instead and costs a resolved pair the denominator reports.
    """
    (tmp_path / "a.py").write_text("WINDOW = 7\nAMBIGUOUS = 1\nLOWER = 2\n")
    (tmp_path / "b.py").write_text("AMBIGUOUS = 9\n")
    (tmp_path / "noise.py").write_text("Window = 5\nnot_a_const = 6\nX = 7 # tail\n")
    out = CQ.tree_constants(tmp_path)
    assert out == {"WINDOW": 7, "LOWER": 2}


def test_a_description_may_name_a_constant_through_its_code_span(fixture_tree):
    """`` `vault_writer.RELEVANCE_FLOOR` (4) `` — the name's own backtick sits between
    the name and the brackets, because the prose is naming which module's constant it
    means. `autonomy/30-intelligence-pipeline-scan-score.md` writes exactly this, and
    it is the whole of one of the eight pairs the resolver counts over that corpus
    today: anchor the grammar on the name's last letter and `RELEVANCE_FLOOR` drops
    out, taking its pair with it, while the report still reads `0 mismatch(es)` —
    indistinguishable from a corpus that is fully correct.
    """
    tree = dict(fixture_tree, RELEVANCE_FLOOR=4)
    good = "write to vault only what clears `vault_writer.RELEVANCE_FLOOR` (4)."
    assert list(CQ.mismatches(good, tree)) == []
    assert CQ.mismatches(good, tree).resolved == 1
    assert list(CQ.mismatches(good.replace("(4)", "(5)"), tree)) == [
        ("RELEVANCE_FLOOR", 5, 4)]


def test_the_gate_selection_collects_both_halves_of_the_pair():
    """The #1276/#2036 lesson, pinned as a node: the red half must be in the
    selection the gate actually runs.

    `pytest.ini`'s own text says the gate's teeth are the `not live_vault` selection,
    and the gate's `tests` rung runs `-m "not live_vault and not fault_injection"`
    (`scripts/automod/gate.py`) — so a red/green pair marked live, or named so the
    expression drops it, enforces nothing no matter how correct it is. Same subprocess
    for the same reason #1276 gives: an in-process collect leaks this module's own
    marks into the selection being measured.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "not live_vault and not fault_injection",
         "--collect-only", "-q", str(Path(__file__))],
        cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    collected = out.stdout
    for node in ("test_one_wrong_number_yields_exactly_one_tuple",
                 "test_the_corrected_fixture_yields_no_mismatch"):
        assert node in collected, f"{node} is outside the selection that has teeth"
    assert "test_the_live_vault_witness_reports_the_pair_count_it_checked" \
        not in collected, "the live witness must stay out of the hard gate rung"


def _live_vault_nodes() -> list[str]:
    return sorted(name for name, fn in list(globals().items())
                  if name.startswith("test_")
                  and any(m.name == "live_vault" for m in getattr(fn, "pytestmark", [])))


def test_exactly_one_node_in_this_file_reads_the_live_vault():
    """Clause 5's "exactly one", pinned mechanically rather than by counting by eye.

    The mark is what keeps the vault off the gate rung (`pytest.ini`); a second one
    would be the beginning of a corpus check that no hard rung ever runs, which is the
    state that let #2043's stale prose land.
    """
    assert _live_vault_nodes() == [
        "test_the_live_vault_witness_reports_the_pair_count_it_checked"]


@pytest.mark.live_vault
def test_the_live_vault_witness_reports_the_pair_count_it_checked():
    """The reporting copy over the real vault: 0 mismatches, and the pair count that
    makes that number mean something.

    Labeled witness, not enforcement (#979): no round under test controls
    `~/obsidian`, and an hourly autoresearch promotion or a nightly job can rewrite a
    task file between rounds — which is exactly what `pytest.ini` marks this for, and
    the reason the enforced edge is the land (`vault_guards
    .autonomy_description_errors`), not this node.

    The count is PRINTED because it is the number the owed-check job has to read after
    landing (#2317's owed list): a non-zero denominator cannot be recovered from a
    green test with no output. The floor here is 2 and NOT an exact number — the tree
    and the prose both move, and the item's own "2 for #79 today" was already wrong
    when it was written (4 across 2 files on 2026-10-07; 8 across 2 files once the
    code-span spelling is read). Below 2 the check has stopped measuring anything and
    says so as instrument failure rather than as a pass.
    """
    tree = CQ.tree_constants(Path(REPO))
    assert tree, f"no `^NAME = <int>` scraped in {REPO}: the probe cannot run"
    descriptions = CQ.task_descriptions(Path.home() / "obsidian" / "autonomy")
    assert descriptions, "no autonomy task files parsed: the probe cannot run"
    out = CQ.scan(descriptions, tree)
    print(f"\nconstant-quote witness: {out['report']}")
    # The count first, and only the first few names: 40 of the 42 descriptions in
    # this vault state no tree constant, which is expected — most tasks quote no
    # number at all — so the full list is 3KB of noise that buries the one number
    # this node exists to record.
    print(f"silent (resolved no pair): {len(out['silent'])} of {len(descriptions)} "
          f"{out['silent'][:5]}")
    for label, name, quoted, actual in out["mismatched"]:
        print(f"  MISMATCH {label}: {name} quoted {quoted}, tree says {actual}")
    assert out["pairs"] >= 2, (
        f"{out['report']} — a corpus that resolves {out['pairs']} pair(s) has not "
        "checked the prose, so 0 mismatches is not evidence")
    assert out["mismatched"] == []
