"""Whose run is the `tests` rung reading? The parser that answered "a child's" (#2251).

`rung_tests` gets its counts from `_parse_pytest_summary(run's whole output)`, and
until #2251 those were five independent `re.search` calls over that text — FIRST
match wins, anywhere in it. A red test can put ANOTHER pytest run's summary line in
that output: 56 test files spawn a child pytest and 35 assertions across 8 of them
interpolate the child's tail (`{text[-1500:]}`), which pytest then echoes verbatim
under `Captured stdout call` — at LINE START, looking exactly like a summary line.
Round `SM_20261005_175752`'s first gate was refused with
`only 24 tests collected (floor 1000) — did the round delete tests?` for that reason:
24 was `12 passed + 1 failed + 1 xfailed + 10 skipped`, the CHILD's totals, and its
sibling rounds on the same base recorded collected in the 16,000s.

So the property this file pins is not "the numbers are right" but "the numbers are
from ONE run — the one the rung started". The anchor is the last summary-shaped line
(`_PYTEST_SUMMARY_LINE_RE`), and all five counts are read out of that one line,
because a rung holding `passed 16200 / tests_skipped 10` is holding two runs and no
floor check can see it.

Four nodes cross the real seam instead of describing it: `test_the_real_serial_run_and_a_child_summary_inside_it`
and `test_the_real_parallel_run_the_rung_actually_uses` spawn a real pytest over a real
throwaway tree whose red node runs a real child pytest, and parse the parent's actual
bytes; `test_a_nested_child_that_prints_its_own_collection_report` does the same with
the child NOT run under `-q`, which is the one shape the summary-line anchor cannot
defuse alone; and
`test_the_vault_guard_probe_derives_its_denominator_from_the_same_one_line` parses the
same output through the parser `scripts/automod/vault_guards.py` shares by reference
(`_gate_tools`, :190-192) and recomputes the `ran` denominator that file derives at
:585 — the one consumer outside the gate a mis-parse can turn into "this probe ran two
tests" when it ran none. The rest are fixture text, whose shape was copied out of that
measured output rather than invented — see `_collision_text`.

The last node is #2251 clause 6 and is not about parsing: it re-derives the refusal
this item was filed from out of committed vault bytes, because the state file those
numbers were read out of is rewritten by the next gate run.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from app import paths
from scripts.automod import gate as G

#: The witness's own dated name, and the mirror it must not overwrite — the same
#: route tests/test_failure_ledger_witness.py uses for two other witnesses, so the
#: node runs wherever the real vault is readable, including under the gate's round
#: home (whose `obsidian` is a symlink to it) and with no `skipif` to hide behind.
WITNESS_MIRROR_NAME = "gate.json"


def _vault_data() -> "Path":
    return paths.VAULT_ROOT / "backlog" / "data"


# ---------------------------------------------------------------------------
# The collision, as the real run prints it
# ---------------------------------------------------------------------------

#: The child's own summary — the numbers that must never reach the rung. Verbatim
#: from #2251's clause 1, and smaller than the parent's on every count the rung
#: reads but one, so a leak shows up as a floor refusal rather than as a number that
#: happens to look plausible. (`xfailed` is 1 on both sides, which is why the node
#: for clause 1 asserts four counts and not one.)
CHILD_SUMMARY = "12 passed, 1 xfailed, 1 failed, 10 skipped in 31.20s"

#: The parent's own summary, and the line that must win. `5 + 16200 + 1 + 34` is the
#: `collected` this fixture has to produce; note the counts are in pytest's own order
#: (failed, passed, xfailed, skipped), not the order the old code searched them in.
PARENT_SUMMARY = "5 failed, 16200 passed, 1 xfailed, 34 skipped in 745.26s"


def _collision_text(child: str = CHILD_SUMMARY, parent: str = PARENT_SUMMARY,
                    tail: str = "") -> str:
    """A whole `-q` run's output: CHILD printed inside the failure report, PARENT last.

    Not an invented shape. Measured from a real run of a real test that spawns a
    child pytest and prints its tail: pytest renders the assertion's continuation
    lines with an `E   ` prefix, but a failing test's CAPTURED STDOUT is echoed
    unprefixed, so the child's summary stands at column 0 — which is exactly the
    position the old unanchored `re.search` could not tell from the parent's.
    """
    return (
        "....................                                             [100%]\n"
        "=================================== FAILURES ===================\n"
        "_______________ test_the_parallel_rung_reports_the_pins ______________\n"
        "\n"
        "    def test_the_parallel_rung_reports_the_pins():\n"
        "        res = subprocess.run(_link_probe_command(True), ...)\n"
        "        print((res.stdout + res.stderr)[-1500:])\n"
        ">       assert link.is_symlink(), f\"no link was created:\\n{text[-1500:]}\"\n"
        "E       AssertionError: no link was created:\n"
        "E       assert False\n"
        "\n"
        "tests/test_gate_parallel_tests.py:802: AssertionError\n"
        "----------------------------- Captured stdout call -------------------\n"
        f"{child}\n"
        f"{tail}"
        "=========================== short test summary info ====================\n"
        "FAILED tests/test_gate_parallel_tests.py::test_the_parallel_rung_reports_the_"
        "pins - AssertionError\n"
        f"{parent}\n")


def test_the_fixture_is_the_shape_that_actually_bites():
    """The negative control for every fixture-based node below: read the SAME text the
    way the rung used to, and it reports the child.

    Without this, a green parse of a friendly fixture would prove nothing, so the
    first assertion here is that the adversarial part is present — the FIRST
    `N passed` in the text belongs to the child, and the child's totals differ from
    the parent's on all five counts the rung reads.
    """
    text = _collision_text()
    assert G._pytest_summary_block(text) == PARENT_SUMMARY, (
        "the anchor has to land on the LAST summary-shaped line, which is the "
        "parent's; the child's line is above it in the same text")
    for pattern, child_value in ((r"(\d+) passed", "12"), (r"(\d+) failed", "1"),
                                 (r"(\d+) xfailed", "1"), (r"(\d+) skipped", "10")):
        assert re.search(pattern, text).group(1) == child_value, (
            f"{pattern}: the fixture stopped being adversarial — the first match in "
            f"it is no longer the child's, so the nodes below prove nothing")


def test_the_parent_run_wins_over_a_child_summary_in_its_failure_text():
    """Clause 1. The clause-1 fixture, parsed the way `rung_tests` parses it."""
    got = G._parse_pytest_summary(_collision_text())
    assert got["passed"] == 16200, got
    assert got["failed"] == 5, got
    assert got["xfailed"] == 1, got
    assert got["tests_skipped"] == 34, got
    assert got["errors"] == 0, got


def test_every_count_comes_from_that_one_block_not_from_an_earlier_one():
    """Clause 2. A leak does not have to be all five counts: the danger is a dict that
    mixes runs, because `passed 9000 / errors 3` from two runs still clears every
    floor. So the child here disagrees with the parent on ALL FIVE counts the rung
    reads — errors, failed, passed, xfailed, skipped — and each key is asserted
    separately.
    """
    child = "3 errors, 2 failed, 9 passed, 2 xfailed, 7 skipped in 4.10s"
    parent = "1 error, 4 failed, 9000 passed, 9 xfailed, 34 skipped in 61.00s (0:01:01)"
    got = G._parse_pytest_summary(_collision_text(child=child, parent=parent))
    assert (got["failed"], got["errors"], got["passed"], got["xfailed"],
            got["tests_skipped"]) == (4, 1, 9000, 9, 34), got
    for key, child_value in (("errors", 3), ("failed", 2), ("passed", 9),
                             ("xfailed", 2), ("tests_skipped", 7)):
        assert got[key] != child_value, f"{key}: read {child_value} from the child run"
    # The ` (0:01:01)` wall-clock tail is not decoration in this node: pytest's
    # `format_session_duration` only prints it past 60 seconds, and a real suite run
    # here takes ~750 s, so an anchor that cannot read it matches NOTHING and every
    # round is refused for having collected zero tests.
    assert got["collected"] == 1 + 4 + 9000 + 9 + 34, got


def test_the_duration_form_pytest_actually_writes_is_recognised():
    """The ` in <duration>` tail is the anchor, so it is pinned against pytest's own
    formatter rather than against a hand-typed string: under a minute it is
    `745.26s`, past a minute `745.26s (0:12:25)`, past an hour `3725.00s (1:02:05)`.
    A summary line the anchor misses is read as a run that collected nothing.
    """
    from _pytest.terminal import format_session_duration

    for seconds in (0.5, 59.0, 61.0, 745.26, 3725.0):
        duration = format_session_duration(seconds)
        line = f"5 failed, 16200 passed, 1 xfailed, 34 skipped in {duration}"
        assert G._pytest_summary_block(line) == line, (
            f"{duration}: pytest's own summary line at {seconds}s is not recognised "
            f"as a summary line by _PYTEST_SUMMARY_LINE_RE")
        got = G._parse_pytest_summary(line)
        assert (got["passed"], got["failed"], got["tests_skipped"]) == (16200, 5, 34)


def test_collected_is_a_figure_pytest_emitted_never_a_sum_of_failure_prose():
    """Clause 3. Three cases, in the order the parser tries them.

    1. A run whose output has a collection report (`collected N items` — `-q`, the
       shape this rung uses, prints none, and that absence is the reason the prose
       sum was the only live path for as long as the floor existed): pytest's own
       figure wins, and it is the FIRST one, because pytest writes it in its header
       before any test has echoed anything. A child's `collected 7 items`, echoed in
       a captured section afterwards, cannot answer for the parent.
    2. No collection report at all (`-q`): `collected` is derived from the run's own
       summary line, and clearing `PYTEST_MIN_COLLECTED` is that derivation's job.
    3. Numbers that exist ONLY in failure prose, with no summary line and no
       collection report: `collected` stays 0. That is the case the old fallback
       turned into a number, and it is the one the floor must not be able to pass.
    """
    header = ("============================= test session starts =================\n"
              "collected 16300 items\n\n")
    assert G._parse_pytest_summary(header + _collision_text())["collected"] == 16300

    with_child_report = header + _collision_text(tail="collected 7 items\n")
    assert G._parse_pytest_summary(with_child_report)["collected"] == 16300, (
        "the child's collection report is echoed after the parent's header, so the "
        "parent's is the FIRST figure in the text and the only one that is its own")

    derived = G._parse_pytest_summary(_collision_text())
    assert derived["collected"] == 5 + 16200 + 1 + 34, derived
    assert derived["collected"] >= G.PYTEST_MIN_COLLECTED, (
        f"the floor is {G.PYTEST_MIN_COLLECTED}; a real suite run must clear it from "
        f"its own summary line, and 24 (the child's sum) must not be able to")

    prose_only = ("FAILED tests/test_gate_parallel_tests.py::test_pins - "
                  "AssertionError: the child reported\n"
                  "12 passed, 1 xfailed, 1 failed, 10 skipped\n")
    assert G._parse_pytest_summary(prose_only) == {
        "passed": 0, "failed": 0, "errors": 0, "xfailed": 0, "tests_skipped": 0,
        "collected": 0, "pin_findings": []}, (
        "counts quoted in failure prose are a quotation, not a run — summing them is "
        "what let a child's 24 satisfy a 1000-test floor")


def _throwaway_tree(root: Path, *, child_quiet: bool = True) -> Path:
    """A tree with 6 tests: one red node that RUNS a real child pytest over
    `tests/test_child.py` (2 tests) and prints its tail, 3 green neighbours.

    The parent therefore really reports `1 failed, 5 passed` while its captured
    stdout carries a real child `2 passed in ...` — the same two-run collision,
    produced by pytest rather than written by me, and with the child's `passed`
    (2) below the parent's (5) so a leak cannot hide behind a matching number.

    `child_quiet=False` drops `-q` from the CHILD's command, which is what a test
    that wants to read a child's report actually does: that child then prints a
    header with its own `collected 2 items`, and it is the one shape the
    summary-line anchor cannot defuse on its own. Both forms are spawned here, so
    neither seam is described rather than exercised.
    """
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_child.py").write_text(
        "def test_child_one():\n    assert True\n\n"
        "def test_child_two():\n    assert True\n", encoding="utf-8")
    quiet = "'-q'," if child_quiet else ""
    (root / "tests" / "test_parent.py").write_text(
        "import pathlib\nimport subprocess\nimport sys\n\n"
        "def test_a_pin_did_not_run():\n"
        "    d = pathlib.Path(__file__).parent\n"
        "    res = subprocess.run(\n"
        f"        [sys.executable, '-m', 'pytest', {quiet}\n"
        "         '-p', 'no:cacheprovider',\n"
        "         str(d / 'test_child.py')], capture_output=True, text=True,\n"
        "        timeout=300)\n"
        "    print((res.stdout + res.stderr)[-1500:])\n"
        "    assert False, 'the pin did not run'\n\n"
        "def test_green_one():\n    assert True\n\n"
        "def test_green_two():\n    assert True\n\n"
        "def test_green_three():\n    assert True\n", encoding="utf-8")
    return root


def _run_pytest(tree: Path, *extra: str, quiet: bool = True,
                mark: str | None = None) -> str:
    """Run pytest the way the rungs invoke it: `-q`, plus `-m <mark expr>`.

    `quiet=False` drops the parent's own `-q`, which is how a caller that wants
    pytest's header (and its `collected N items` line) asks for one.
    """
    done = subprocess.run(
        [sys.executable, "-m", "pytest", *(["-q"] if quiet else []),
         "-m", mark if mark is not None else G.TESTS_MARK_EXPR,
         "-p", "no:cacheprovider", *extra, "tests/"],
        cwd=str(tree), capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode != 0, (
        "the fixture tree was supposed to have a red node:\n"
        + done.stdout + done.stderr)
    return done.stdout + done.stderr


def test_the_real_serial_run_and_a_child_summary_inside_it(tmp_path):
    """The seam, end to end: a real `-q` run of a real tree whose red node spawns a
    real child pytest, parsed by the rung's own function.

    Three things are asserted, and the third is the one the #2251 triage measured
    and the item asked to be written down: `pytest -q` prints NO
    `collected N items` line, so the only figure available to the parser is the one
    on its own summary line. `collected` has to come to 6, which is the number of
    tests that exist here — a parser that read the child would say 2.
    """
    tree = _throwaway_tree(tmp_path)
    text = _run_pytest(tree)

    assert re.search(r"collected \d+ item", text) is None, (
        "`pytest -q` was expected to print no collection report — if that changed, "
        "the emitted-figure branch is the live one and the fallback's scope here is "
        "no longer what this node pins:\n" + text[-800:])
    assert re.search(r"(\d+) passed", text).group(1) == "2", (
        "the first `N passed` in the parent's output is the CHILD's `2 passed`; if "
        "that is no longer true the tree stopped producing the collision")

    got = G._parse_pytest_summary(text)
    assert got["failed"] == 1, got
    assert got["passed"] == 5, got
    assert got["collected"] == 6, (
        f"6 tests exist in the tree; {got['collected']} is what the parser claims ran")
    assert got["tests_skipped"] == 0 and got["errors"] == 0, got


def test_the_real_parallel_run_the_rung_actually_uses(tmp_path):
    """Same tree, same parse, one layer down: the full-suite rung command is
    `-q -m <TESTS_MARK_EXPR> -n 8 --dist loadfile`, and under `xdist` the summary
    line is written by the controller, not by the terminal reporter this file's
    fixture imitates. If the anchor only matched a serial run's line, every real
    round would be refused.
    """
    tree = _throwaway_tree(tmp_path)
    text = _run_pytest(tree, "-n", "2", "--dist", "loadfile")

    got = G._parse_pytest_summary(text)
    assert got["failed"] == 1, got
    assert got["passed"] == 5, got
    assert got["collected"] == 6, got


def test_a_nested_child_that_prints_its_own_collection_report(tmp_path):
    """The seam the first review named, now exercised rather than described: a
    NON-`-q` child inside the `-q` parent this rung actually runs.

    A child invoked without `-q` prints a header carrying its own
    `collected 2 items`, and the parent echoes that into a Captured-stdout section.
    Because the parent runs with `-q` and prints no collection report of its own,
    the child's is the ONLY such figure anywhere in the text — so searching the
    whole text for it hands `collected` to the child and lets a 2-test child answer
    for a 6-test parent. That is #2251 again in the one shape the summary-line
    anchor cannot defuse on its own, and the guard is positional: a collection
    figure ABOVE the first piece of test output is the run's own, because pytest
    writes its header before any test can echo; one below it is somebody else's.

    `collected` has to come to 6, the tests that exist here, and never to 2.
    """
    tree = _throwaway_tree(tmp_path, child_quiet=False)
    text = _run_pytest(tree)

    assert re.search(r"collected \d+ item", text) is not None, (
        "the nested child was expected to print its own collection report; if it no "
        "longer does, this node describes the seam instead of crossing it:\n"
        + text[-800:])
    assert G._pytest_collection_figure(text) is None, (
        "the only collection figure in this text sits below the first piece of test "
        "output, so the header region must find none — if it finds one, the echo has "
        "moved and the cut is no longer what this node pins")

    got = G._parse_pytest_summary(text)
    assert (got["failed"], got["passed"]) == (1, 5), got
    assert got["collected"] == 6, (
        f"6 tests exist in the parent run; {got['collected']} is the nested child's "
        "own collection report")


def test_the_vault_guard_probe_derives_its_denominator_from_the_same_one_line(
        tmp_path):
    """The second reader of this parser, named by the review as an unexercised seam.

    `scripts/automod/vault_guards.py` is the vault-land agreement probe. It takes the
    parser by reference from `gate` (`_gate_tools`, :190-192 — no second copy), runs
    its selection with `-q --no-header` (:524), and at :585 computes its own `ran`
    denominator as `collected - tests_skipped`. That denominator is the probe's whole
    claim to have executed something, so the class #2251 fixed — a child's number read
    as the run's — would silently turn a probe that ran nothing into a probe that
    appears to have run two tests.

    Two facts are asserted, both measured on a real run: `-q --no-header` emits NO
    collection figure either (`collected` therefore comes from the summary line), and
    the header-region guard means a nested non-`-q` child's `collected 2 items` cannot
    become the probe's denominator.
    """
    from scripts.automod import vault_guards as VG
    mark, _failed_ids, summary = VG._gate_tools()
    assert summary is G._parse_pytest_summary, (
        "the probe was expected to share this parser by reference; a second reader "
        "of pytest's summary line is a second way to count a denominator wrong")

    tree = _throwaway_tree(tmp_path / "vg", child_quiet=False)
    text = _run_pytest(tree, "--no-header", mark=mark)
    assert re.search(r"collected \d+ item", text[:text.find("[100%]")]) is None, (
        "`-q --no-header` was expected to print no collection figure of its own; if "
        "that changed, the header region is the live path here and not the summary "
        "line:\n" + text[-600:])

    counts = summary(text)
    ran = int(counts["collected"]) - int(counts["tests_skipped"])
    assert ran == 6, (
        f"the probe's denominator is {ran}; 6 tests exist in the run, and 2 is what a "
        f"nested child's collection report would have handed it: {counts}")


def test_a_run_that_printed_no_summary_at_all_is_read_as_no_run(tmp_path):
    """Fail closed when the run never got to a summary line — the class
    `PYTEST_MIN_COLLECTED` exists to catch (#1837), and the class the old fallback
    could quietly satisfy with a quotation's arithmetic.

    The first half is measured, not invented: a `-q` run whose `conftest.py` raises
    on import prints `ImportError while loading conftest ...` and NOTHING else — no
    counts, no summary — so it parses to zeros and the floor refuses it. The second
    half is the quotation case: counts in prose with no ` in <duration>` tail are not
    a summary line.

    What the anchor cannot do, stated rather than glossed: if the parent printed no
    summary of its own but a CHILD's survived in the text, the child's line is what
    `_pytest_summary_block` returns — the two are indistinguishable by shape. That
    case is caught one rung-adjacent step away, by the exit code `rung_tests` also
    reads (a session that dies is never rc 0), which is why this node checks the
    zeros and not a verdict.
    """
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text(
        "raise RuntimeError('conftest died')\n", encoding="utf-8")
    (tmp_path / "tests" / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode != 0
    text = done.stdout + done.stderr
    assert "conftest" in text, text
    assert G._parse_pytest_summary(text)["collected"] == 0, text

    quoted = ("INTERNALERROR> RuntimeError: worker crashed\n"
              "INTERNALERROR> after 12 passed, 3 skipped\n")
    assert G._pytest_summary_block(quoted) == ""
    assert G._parse_pytest_summary(quoted)["collected"] == 0


# ---------------------------------------------------------------------------
# clause 6 — the refusal's bytes have a home in the repo
# ---------------------------------------------------------------------------

#: The witness, committed in the vault: the real `gate.json` of the SECOND round
#: #2251 records as refused by this defect (`SM_20261005_115114`, head `5309e38e`).
#: Its path is a dated sibling, not `backlog/data/gate.json`, which already holds
#: #1883's witness (`cfe9a113`) — and its source is a state file the next gate run
#: rewrites in place, which is the whole reason the bytes are copied here.
WITNESS_2251_NAME = "2026-10-05.2251-gate-refusal-witness.json"


def test_the_refusal_this_item_was_filed_from_is_committed_and_re_derivable():
    """#2251 clause 6, and the reason it cannot be satisfied by the clause's own
    named path.

    `wc -l < backlog/data/2026-10-05.2251-gate-refusal-witness.json` is 150, the
    figure this node checks. The refusal is re-derived from those committed bytes
    rather than from `~/.local/state/lloyd-automod/rounds/<id>/gate.json`, which
    lives under a round id and is rewritten by the next gate: the source #2251's
    triage cited had already been overwritten at 12:08Z to `collected 106`, a green
    partial run, reproducing neither the quoted refusal nor the 16,294 the triage
    measured.

    What is checked is what the item quotes, field for field: the rung order that
    stopped at `tests`, `passed 12 / failed 1 / errors 0 / xfailed 1 / skipped 10 /
    collected 24 / workers 8`, the detail's refusal text, and that
    `tests/test_gate_parallel_tests.py` is in the serial retry list — the file whose
    base-red node quotes a child pytest's summary and so is the mechanism. And that
    `backlog/data/gate.json` is still 34 newline characters and still round
    `SM_20260930_063800`'s, because overwriting it is what the named path would have
    done.
    """
    witness = _vault_data() / WITNESS_2251_NAME
    assert witness.is_file(), (
        f"{WITNESS_2251_NAME} is not in the vault: the refusal this item was filed "
        "from has no committed bytes")
    text = witness.read_text(encoding="utf-8")
    # `wc -l` counts newline CHARACTERS, which is the figure the clause quotes.
    assert text.count("\n") == 150, (
        f"wc -l on the witness is {text.count(chr(10))}, not the 150 the clause "
        "quotes — the committed bytes are not the report that was measured")
    report = json.loads(text)

    rung = [r for r in report["rungs"] if r["name"] == "tests"]
    assert len(rung) == 1 and rung[0]["ok"] is False, rung
    assert [r["name"] for r in report["rungs"]] == [
        "preflight", "vet", "static", "frontend", "tests"], (
        "the ladder stopped at `tests`, which is what 'the gate refused the round' "
        "means here: no review ran, so no clause was ever graded on that commit")

    data = rung[0]["data"]
    assert (data["passed"], data["failed"], data["errors"], data["xfailed"],
            data["tests_skipped"], data["collected"], data["workers"]) == (
        12, 1, 0, 1, 10, 24, 8), data
    assert rung[0]["detail"].startswith("only 24 tests collected (floor 1000)"), (
        rung[0]["detail"])
    assert "did the round delete tests?" in rung[0]["detail"], rung[0]["detail"]
    assert "tests/test_gate_parallel_tests.py" in data["serial_retry_files"], (
        "without the file whose red node quotes a child pytest's tail in the retry "
        "list, these bytes are not an instance of the mechanism this item is about")
    assert report["head"] == "5309e38e8394d27a4a4833bde0424c4d20ad7255", report["head"]

    mirror = _vault_data() / WITNESS_MIRROR_NAME
    assert mirror.read_text(encoding="utf-8").count("\n") == 34, (
        "backlog/data/gate.json is #1883's witness at `wc -l` 34; a second witness "
        "written over it is the hazard this clause's named path had")
    assert json.loads(mirror.read_text(encoding="utf-8"))["round_id"] == \
        "SM_20260930_063800", mirror
