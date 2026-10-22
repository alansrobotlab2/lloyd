"""#2258: the theory-fan-out replay instrument's five clauses.

Every node here runs the shipped script as an importable module over a fixture
tree or a fixture ledger, and none of them needs the engine: the study's live
arms take injected `generate`/`investigate`/`rank_fn` callables precisely so
that the RECORDING contract — three theories before any evidence row, claims
with citations, argmax-only ranking, a baseline that can refuse to be evaluated
— is testable without a single token. The nodes are named for the clause each
one pins.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval import diagnosis_fanout_replay as replay  # noqa: E402
from eval.diagnosis_fanout_replay import (  # noqa: E402
    ARM_A, ARM_B, ARM_RETRY, DEFAULT_INVESTIGATION_TURNS, DEFAULT_TOKEN_CEILING,
    MIN_CASES, RecordError, THEORIES_PER_CASE, VAULT_WITNESS, WITNESS_CASES,
    WITNESS_ROWS, WITNESS_SHA, RowWriter, agrees, baseline_problem, build_cases,
    case_complete, citation_rows, excluded_note, format_report, investigate_live,
    learnings_cases, load_cases, rank_theories, run_case_a, run_case_b, score,
    select_cases, slot_labels, verify_citation, verify_citations, witness_census,
    write_baseline, write_cases)

LEDGER = REPO / ".git"  # exists, so a missing-ledger path is never confused with one


# ── fixtures ────────────────────────────────────────────────────────────────

def _ledger_rows(round_id: str, node: str) -> list[dict]:
    """One round's ledger, in time order: blamed, then refuted by the base probe."""
    return [
        {"event": "gate", "rung": "tests", "ok": False, "ts": 1, "round_id": round_id,
         "base": "abcdef1234567890", "detail": "pytest failed"},
        {"event": "red_tree_filed", "ts": 2, "round_id": round_id, "item_id": 900,
         "node_ids": [node], "base": "abcdef1234567890"},
        {"event": "gate", "rung": "tests", "ok": True, "ts": 3, "round_id": round_id,
         "base": "abcdef1234567890", "head": "beefbeef0000",
         "pre_existing_failures": [node], "red_tree_item": 900,
         "detail": "tests pass on this diff — 1 failure(s) PRE-EXISTING at base"},
    ]


def _write_ledger(path: Path, rounds: int, node_fmt: str = "tests/test_x{i}.py::test_{i}"
                  ) -> Path:
    rows: list[dict] = []
    for i in range(rounds):
        rows.extend(_ledger_rows(f"SM_2026{i:02d}_{i:06d}", node_fmt.format(i=i)))
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _learnings_fixture(root: Path, n: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    body = ["# Daily", "", "| # | source | class | detail |", "|---|---|---|---|"]
    for i in range(n):
        body.append(
            f'| {i + 1} | probe-quality | 1 (self-caught) | "The socket probe proves the '
            f'listener is down" — **self-caught**: the probe reads 0 UDP sockets, use '
            f"`probe_socket_{i}.py` instead |")
    (root / "2026-09-09.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    return root


def _cases(n: int, *, marker: str = "pre-existing") -> list[dict]:
    return [{"key": f"case-{i}", "family": "tests-blamed-then-refuted",
             "subsystem": "single-service", "input": f"incident {i} came back red",
             "claimed_cause": f"the diff caused failure {i}",
             "actual_cause": f"pre-existing at base {i}: {marker}",
             "actual_cause_markers": [marker],
             "source_ref": f"promotions.jsonl:{i + 1} (round SM_{i})"}
            for i in range(n)]


# ── clause 1: the corpus builder and its floor ──────────────────────────────

def test_the_corpus_builder_refuses_below_ten_complete_cases(tmp_path):
    """Fewer than `MIN_CASES` complete cases writes no artifact and exits non-zero.

    The floor is the guard against padding a corpus up to a study size it does
    not have, so the test reads the exit code, the absence of the file and the
    number printed, not just "it did not crash".
    """
    ledger = _write_ledger(tmp_path / "promotions.jsonl", MIN_CASES - 1)
    cases, _ = build_cases(ledger, tmp_path / "no-such-learnings")
    assert len(cases) == MIN_CASES - 1
    out = tmp_path / "cases.jsonl"
    code, message = write_cases(cases, out)
    assert code != 0, "a corpus under the floor must not exit 0"
    assert not out.exists() and not out.with_suffix(".jsonl.tmp").exists(), (
        "the refusal wrote an artifact anyway")
    assert "cannot evaluate" in message and str(MIN_CASES - 1) in message


def test_the_corpus_builder_writes_at_the_floor_with_every_field(tmp_path):
    """At the floor the same input writes a file every case of which is complete."""
    ledger = _write_ledger(tmp_path / "promotions.jsonl", MIN_CASES)
    cases, notes = build_cases(ledger, tmp_path / "no-such-learnings")
    assert not notes, f"unexpected drops: {notes}"
    out = tmp_path / "cases.jsonl"
    code, message = write_cases(cases, out)
    assert code == 0, message
    written = load_cases(out)
    assert len(written) == MIN_CASES
    assert all(case_complete(c) for c in written)
    for case in written:
        assert case["input"] and case["claimed_cause"] and case["actual_cause"]
        assert case["source_ref"], f"{case['key']} has no source a reader can re-check"


@pytest.mark.parametrize("missing", ["input", "claimed_cause", "actual_cause",
                                     "source_ref"])
def test_an_incomplete_case_is_dropped_not_written(missing):
    """Each of the four clause-1 fields is required, individually."""
    case = _cases(1)[0]
    case[missing] = ""
    assert not case_complete(case), f"`{missing}` is not required by the builder"


def test_the_ledger_and_the_learnings_tree_both_contribute(tmp_path):
    """The corpus is the union of the two sources the item named, not one of them.

    Three cases from a learnings tree with a retraction table plus the floor
    minus three from the ledger must land at exactly `MIN_CASES` — that is what
    proves the second source is wired, rather than the builder reading one file.
    """
    learnings = _learnings_fixture(tmp_path / "learnings", 3)
    ledger = _write_ledger(tmp_path / "promotions.jsonl", MIN_CASES - 3)
    cases, notes = build_cases(ledger, learnings)
    assert not notes, f"unexpected drops: {notes}"
    from_ledger = [c for c in cases if c["family"] == "tests-blamed-then-refuted"]
    from_learnings = [c for c in cases if c["family"] == "retracted-recorded-cause"]
    assert len(from_ledger) == MIN_CASES - 3 and len(from_learnings) == 3
    assert len(cases) == MIN_CASES
    # The learnings row is only usable because it names the mechanism; a row
    # without one states a reversal nobody can score against.
    assert from_learnings[0]["actual_cause_markers"], from_learnings[0]


def test_the_witness_is_a_dated_extract_and_the_retired_path_stays_absent():
    """#2258's witness is a dated sibling, and the wholesale-copy path stays absent.

    Clause 6 names a path in `backlog/data/` that this repo's own rails retire:
    `tests/test_failure_ledger_witness.py:82` asserts the copy path does not
    exist, and `tests/test_automod_vault_round.py`'s #2054 clause-5 node refuses
    even a new file that names it (the retention sweep retired it as a
    33,113,707-byte copy; #2243 settled the dated-sibling route for the same
    collision). This node pins the resolution so the clause cannot be "satisfied"
    later by re-pointing the instrument at the retired path — which would break a
    landed rail in a diff about something else.

    It also pins that the extract's 2,621 rows are NOT a ledger row count: the
    quoted event classes are complete in it (692 rows across the seven), while
    everything the case builder does not cite is excluded.
    """
    from eval.diagnosis_fanout_replay import LEDGER_PATH

    assert VAULT_WITNESS.name == "2026-10-06.2258-ledger-witness.jsonl"
    assert VAULT_WITNESS.parent.name == "data"
    assert VAULT_WITNESS.parent.parent.name == "backlog"
    assert VAULT_WITNESS.is_file(), "clause 6's witness must be committed"
    assert not (VAULT_WITNESS.parent / "promotions.jsonl").exists(), (
        "the retired wholesale-copy path must stay absent; that is #2054 clause 5")
    assert VAULT_WITNESS != LEDGER_PATH
    census = witness_census(VAULT_WITNESS)
    assert census["rows"] == WITNESS_ROWS == 2621, census
    assert sum(census["events"].values()) == 692, census
    # A citation from the corpus resolves IN the witness, naming it.
    cases, notes = build_cases(VAULT_WITNESS, Path("/nonexistent-2258-learnings"))
    assert len(cases) == WITNESS_CASES and not notes, (len(cases), notes[:2])
    assert all(c["source_ref"].startswith(VAULT_WITNESS.name) for c in cases), (
        "a source_ref must name the file its line number resolves in")
    assert WITNESS_SHA == "5c00fdb5"


def test_the_corpus_command_splits_the_two_sources_by_their_own_field(tmp_path):
    """The printed split comes from each case's `source`, not from a path prefix.

    This is the half of the mislabelling finding the real-source node cannot
    reach: on the live tree `learnings=0` is true either way, so a classification
    that files learnings cases under `ledger` prints a correct-looking line.
    With three usable learnings rows and seven ledger rounds the only honest
    output is `ledger=7 learnings=3`, and the old prefix test (learnings refs
    read `memory/learnings/<file>:<line>`, which never starts with `learnings/`)
    would have printed `learnings=0` for this same input.
    """
    learnings = _learnings_fixture(tmp_path / "learnings", 3)
    ledger = _write_ledger(tmp_path / "promotions.jsonl", 7)
    out = tmp_path / "cases.jsonl"
    proc = subprocess.run(
        [sys.executable, str(REPO / "eval" / "diagnosis_fanout_replay.py"), "corpus",
         "--ledger", str(ledger), "--learnings", str(learnings), "--out", str(out)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert proc.returncode == 0, proc.stdout + proc.stderr[-500:]
    assert "sources: ledger=7 learnings=3" in proc.stdout, proc.stdout
    assert f"wrote {MIN_CASES} case(s)" in proc.stdout, proc.stdout
    written = load_cases(out)
    assert sum(1 for c in written if c["source"] == "learnings") == 3
    assert sum(1 for c in written if c["source"] == "ledger") == 7


def test_a_learnings_row_naming_no_mechanism_is_not_a_case(tmp_path):
    """The scarce source stays scarce: no backticked mechanism, no case."""
    root = tmp_path / "learnings"
    root.mkdir()
    (root / "2026-09-10.md").write_text(
        "| # | source | class | detail |\n|---|---|---|---|\n"
        '| 2 | memory-capture | 1 (retracted) | "Trajectory coverage collapsed" — '
        "**retracted by Job 2**: a same-day measurement artifact |\n", encoding="utf-8")
    assert learnings_cases(root) == []


def test_the_builder_reads_the_committed_witness_and_names_both_sources():
    """The real sources, pinned to numbers, and still writing no artifact.

    Run with the floor raised so high it cannot be met, so the live tree is
    untouched while the actual read happens, and run with `--ledger` pointed at
    the COMMITTED witness rather than the live ledger: the live file has no
    history and grows with every gate, so a number read off it is true for one
    (it measured 23,388, 23,389 and 23,413 inside one working session), so a
    number read off it is true for one second and unpinnable. Over the witness
    (vault commit `5c00fdb5`) the split is 98 ledger cases and 0 from the
    learnings tree — pinned as those numbers rather than matched as a shape,
    `ledger=\\d+ learnings=\\d+` passes whatever the sources actually did, which
    is the finding this node answers. `learnings=0` is the recorded fact about
    this tree (see `learnings_cases`); that the second source is WIRED at all is
    pinned by `test_the_ledger_and_the_learnings_tree_both_contribute`, which
    feeds it a fixture with three usable rows and gets three cases.
    """
    out = Path("/tmp/2258-real-corpus-probe.jsonl")
    if out.exists():
        out.unlink()
    proc = subprocess.run(
        [sys.executable, str(REPO / "eval" / "diagnosis_fanout_replay.py"), "corpus",
         "--ledger", str(VAULT_WITNESS), "--out", str(out), "--min-cases", "100000"],
        capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert proc.returncode != 0, proc.stdout[-500:] + proc.stderr[-500:]
    assert not out.exists(), "the probe wrote a real-tree corpus"
    assert "cannot evaluate" in proc.stdout, proc.stdout[-500:]
    assert "sources: ledger=98 learnings=0" in proc.stdout, proc.stdout[-500:]


# ── clause 6: the witness bytes ─────────────────────────────────────────────

def test_the_witness_census_is_re_derivable_from_the_committed_rows(tmp_path):
    """`witness` counts a ledger file's rows and its quoted event census.

    A fixture with two of each quoted event proves the counting, and the
    refusal proves the identity check: a file whose row count is not the
    committed one is not the corpus the study was built on, so it must exit
    non-zero instead of printing a census of unknown bytes.
    """
    fixture = tmp_path / "promotions.jsonl"
    with fixture.open("w", encoding="utf-8") as fh:
        for event in ("round_aborted", "round_aborted", "alert", "alert",
                      "red_tree_filed", "vault_revert"):
            fh.write(json.dumps({"event": event, "reason": "because", "round_id":
                                 "SM_X"}) + "\n")
        fh.write("{ not json\n")
    census = witness_census(fixture)
    assert census["rows"] == 7 and census["unparsable"] == 1, census
    assert census["reason_rows"] == 6, census
    assert census["events"] == {"round_aborted": 2, "round_abandoned": 0,
                                "red_tree_filed": 1, "red_tree_closed": 0,
                                "alert": 2, "vault_revert": 1,
                                "rollback_succeeded": 0}, census
    assert census["rows"] != WITNESS_ROWS
    proc = subprocess.run(
        [sys.executable, str(REPO / "eval" / "diagnosis_fanout_replay.py"), "witness",
         "--path", str(fixture)],
        capture_output=True, text=True, timeout=120, cwd=str(REPO))
    assert proc.returncode != 0, proc.stdout
    assert "cannot evaluate" in proc.stdout, proc.stdout


def test_the_committed_witness_carries_the_figures_the_item_quotes():
    """The witness is on disk, is the committed size, and its `wc -l` is the figure.

    This is clause 6 made checkable from the tree: the ledger the corpus is
    built from has no history under `~/.local/state`, so the numbers #2258
    quotes have to come from bytes that do. The row count is pinned to what was
    committed (`WITNESS_ROWS`), the event census to what the triage reported,
    and `wc -l` is run over the file rather than trusted from the census, so a
    file whose last line has no newline cannot count short.
    """
    assert VAULT_WITNESS.exists(), (
        f"no witness at {VAULT_WITNESS}: clause 6 is unmet until it is committed")
    census = witness_census(VAULT_WITNESS)
    assert census["rows"] == WITNESS_ROWS, census
    assert census["unparsable"] == 0, census
    assert census["reason_rows"] == 585, census
    assert census["events"]["round_aborted"] == 283, census
    assert census["events"]["round_abandoned"] == 224, census
    assert census["events"]["red_tree_filed"] == 94, census
    assert census["events"]["alert"] == 55, census
    proc = subprocess.run(["wc", "-l", str(VAULT_WITNESS)],
                          capture_output=True, text=True, timeout=120)
    assert int(proc.stdout.split()[0]) == WITNESS_ROWS, proc.stdout
    # The corpus the study will run on is rebuildable from these bytes.
    cases, notes = build_cases(VAULT_WITNESS, Path("/nonexistent-2258-learnings"))
    assert len(cases) == WITNESS_CASES == 98 and not notes, (len(cases), notes[:3])
    assert all(case["source"] == "ledger" for case in cases)


# ── clause 2: the citation verifier ─────────────────────────────────────────

def test_a_cited_line_resolves_and_a_fabricated_path_fails(tmp_path):
    """Both halves in one node: an out-of-range line and a path that is not there.

    A verifier that only ever returns True cannot tell a real citation from an
    invented one, and the whole ranking claim rests on that difference.
    """
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "loop.py").write_text("line one\nline two\nline three\n",
                                              encoding="utf-8")
    ok = verify_citation({"path": "app/loop.py", "line": 2}, root=tmp_path)
    assert ok["ok"] and ok["reason"] == "line-resolves", ok

    out_of_range = verify_citation({"path": "app/loop.py", "line": 99}, root=tmp_path)
    assert not out_of_range["ok"] and out_of_range["reason"] == "line-out-of-range"

    fabricated = verify_citation({"path": "app/never_written.py", "line": 1},
                                 root=tmp_path)
    assert not fabricated["ok"] and fabricated["reason"] == "no-such-path"

    colon_form = verify_citation({"path": "app/loop.py:3"}, root=tmp_path)
    assert colon_form["ok"], "the `path:line` form a transcript actually uses failed"
    assert verify_citation({"path": "app/loop.py:99"}, root=tmp_path)["reason"] == \
        "line-out-of-range"


def test_the_citation_rate_counts_every_citation_of_a_rows_claims(tmp_path):
    """The 90% bar is measured over citations, so the denominator has to be right."""
    (tmp_path / "real.py").write_text("a\nb\n", encoding="utf-8")
    rows = [{"arm": ARM_B, "kind": "tool", "claims": [
        {"claim": "the probe reads 0 sockets",
         "citations": [{"path": "real.py", "line": 1}, {"path": "fake.py", "line": 1}]},
        {"claim": "the retry re-ran it", "citations": [{"path": "real.py"}]}]}]
    assert len(citation_rows(rows)) == 3
    scored = verify_citations(citation_rows(rows), root=tmp_path)
    assert scored["n"] == 3 and scored["passed"] == 2
    assert abs(scored["rate"] - 2 / 3) < 1e-9
    assert [v["ok"] for v in scored["verdicts"]] == [True, False, True]


# ── #2799: an unstatable citation fails, it never aborts the report ─────────

#: One citation `path` verbatim out of the 132 recorded rows of
#: `~/lloyd-data/eval/2258/rows.jsonl` (the 2026-10-13 instrumented run). It is a
#: prose excerpt, not a path: 281 characters in ONE component, past NAME_MAX 255,
#: so `os.stat` answers `OSError: [Errno 36] File name too long` instead of
#: "no such file". 45 of the artifact's citations are prose of this shape, and
#: until #2799 the report could not re-grade a single one of its own rows: the
#: exception escaped `verify_citation` and `cmd_report` prints its collected lines
#: only after `format_report` returns, so stdout held the traceback and nothing else.
PROSE_CITATION_281 = (
    "instrumented pytest run, 2026-10-13 (this session), tree extracted from "
    "dfb18646, single node, under 32 CPU spinners — failing run 12 captured "
    "stdout; plugin wrapped Gate._take_review_prefetch and made "
    "Gate._base_failures, gate._failures_at_base, state.read_red_set fatal-if-called")


def _raise_on_one_path(monkeypatch, name: str, *ops: str) -> None:
    """Make named `Path` operations raise for ONE path name and nothing else.

    Every other path keeps its real behaviour, so the node proves the guard is
    where the call is and not merely that some `except` exists somewhere. The
    three operations the citation path reaches — the existence test, the
    file/dir test and the line count's `open` — each get pinned this way.
    """
    for op in ops:
        original = getattr(Path, op)

        def wrapper(self, *args, _orig=original, _name=name, _op=op, **kwargs):
            if self.name == _name:
                raise OSError(36, "File name too long")
            return _orig(self, *args, **kwargs)

        monkeypatch.setattr(Path, op, wrapper)


def test_a_citation_whose_path_is_prose_fails_as_not_a_path(tmp_path):
    """Clause 1: the 281-char excerpt fails the citation; it does not raise.

    `verify_citation`'s own docstring promises "never raise on a bad one", and
    ENAMETOOLONG is exactly the bad case the docstring did not cover: `exists()`
    swallows ENOENT (that is `no-such-path`) but re-raises errno 36, which is
    what killed the study's ability to re-grade its own recorded rows.
    """
    assert len(PROSE_CITATION_281) == 281, (
        "this fixture must stay the recorded excerpt: a shorter one fits in "
        "NAME_MAX and then merely does not exist, which is the other branch")
    verdict = verify_citation({"path": PROSE_CITATION_281}, root=tmp_path)
    assert verdict["ok"] is False, verdict
    assert verdict["reason"] == "not-a-path", verdict
    # And it is NOT folded into `no-such-path`: a prose citation and a
    # fabricated path are different findings, and the study's citation rate is
    # read against both.
    short = verify_citation({"path": "app/never_written.py", "line": 1}, root=tmp_path)
    assert short["ok"] is False and short["reason"] == "no-such-path", short


def test_the_oserror_guard_covers_every_filesystem_call_on_a_citation(tmp_path,
                                                                     monkeypatch):
    """Clause 2: an OSError at ANY stat of the resolved path yields `not-a-path`.

    Catching only `exists()` would move the abort one line down: `is_file()` and
    `_line_count`'s `open` reach the same errno on a path that is statable in
    parts (a long name under a real directory, a file unreadable to this user).
    Each operation is broken separately, on one file name only.
    """
    (tmp_path / "boom.py").write_text("a\nb\n", encoding="utf-8")
    for op in ("exists", "is_file", "open"):
        _raise_on_one_path(monkeypatch, "boom.py", op)
        verdict = verify_citation({"path": "boom.py", "line": 1}, root=tmp_path)
        assert verdict["ok"] is False, (op, verdict)
        assert verdict["reason"] == "not-a-path", (op, verdict)
        monkeypatch.undo()
    # With nothing patched the same citation verifies, so the guard did not
    # simply make every citation fail.
    assert verify_citation({"path": "boom.py", "line": 1},
                           root=tmp_path)["reason"] == "line-resolves"


def test_a_prose_citation_counts_in_the_denominator_not_as_an_abort(tmp_path):
    """Clause 3: prose citations enter the rate as failures; the report survives.

    The item's bar is a citation RATE, so counting a prose citation is only
    honest if it lands in the denominator — 45 of the 2258 artifact's citations
    are prose, and dropping them would inflate the very number the study is
    graded on. One prose row and one real row is 1/2.
    """
    (tmp_path / "real.py").write_text("a\nb\n", encoding="utf-8")
    rows = [
        {"arm": ARM_B, "kind": "tool", "case_key": "case-0", "seq": 1, "tokens": 10,
         "claims": [{"claim": "the guard is missing",
                     "citations": [{"path": PROSE_CITATION_281}]}]},
        {"arm": ARM_B, "kind": "tool", "case_key": "case-0", "seq": 2, "tokens": 10,
         "claims": [{"claim": "the file is two lines",
                     "citations": [{"path": "real.py", "line": 1}]}]},
    ]
    scored = verify_citations(citation_rows(rows), root=tmp_path)
    assert scored["n"] == 2 and scored["passed"] == 1, scored
    assert abs(scored["rate"] - 0.5) < 1e-9, scored
    assert [v["ok"] for v in scored["verdicts"]] == [False, True], scored
    assert scored["verdicts"][0]["reason"] == "not-a-path", scored


def test_the_reasons_the_verdict_already_wrote_are_all_unchanged(tmp_path):
    """The guard adds ONE reason; the six the artifact already carries stand.

    `no-such-path`, `path-exists`, `line-resolves`, `line-out-of-range`,
    `not-a-file` and `line-not-a-number` are what the recorded citation rate is
    built from, and the order between them and the new `not-a-path` is the whole
    risk of the change: a guard wrapped around the wrong stretch would answer
    `not-a-path` for an ordinary missing file and quietly rewrite the study's
    denominator. (`no-path` is listed in the code but unreachable — `Path("")`
    resolves to `.` and reports `path-exists`, which this node also pins so the
    quirk is on the record rather than in prose.)
    """
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "loop.py").write_text("one\ntwo\n", encoding="utf-8")
    (tmp_path / "app" / "dir.py").mkdir()
    cases = {
        "line-resolves": ({"path": "app/loop.py", "line": 2}, True),
        "path-exists": ({"path": "app/loop.py"}, True),
        "no-such-path": ({"path": "app/never.py", "line": 1}, False),
        "line-out-of-range": ({"path": "app/loop.py", "line": 99}, False),
        "not-a-file": ({"path": "app/dir.py", "line": 1}, False),
        "line-not-a-number": ({"path": "app/loop.py", "line": "two"}, False),
    }
    for reason, (citation, ok) in cases.items():
        verdict = verify_citation(citation, root=tmp_path)
        assert verdict["ok"] is ok, (reason, verdict)
        assert verdict["reason"] == reason, (reason, verdict)
    empty = verify_citation({"path": ""}, root=tmp_path)
    assert empty["reason"] == "path-exists", empty


def test_the_report_prints_a_headline_with_a_prose_citation_in_its_rows(tmp_path):
    """Clause 4: `format_report` reaches its print path with prose in the rows.

    The whole point of #2799 is the PRINT PATH: `cmd_report` collects every line
    from `format_report` and prints them afterwards, so one escaping exception
    loses every number the run recorded. The last line is the headline, and with
    no cached retry baseline it must be the refused one rather than a traceback.
    """
    assert len(PROSE_CITATION_281) == 281
    cases = _cases(1)
    rows = [
        {"arm": ARM_A, "kind": "verdict", "case_key": "case-0", "seq": 1,
         "chosen_cause": "the diff caused it", "tokens": 5000},
        {"arm": ARM_B, "kind": "verdict", "case_key": "case-0", "seq": 2,
         "chosen_cause": "the diff caused it", "tokens": 9000},
        {"arm": ARM_B, "kind": "tool", "case_key": "case-0", "seq": 3, "tokens": 9000,
         "claims": [{"claim": "prose, not a path",
                     "citations": [{"path": PROSE_CITATION_281}]}]},
        {"arm": ARM_A, "kind": "tool", "case_key": "case-0", "seq": 4, "tokens": 5000,
         "claims": [{"claim": "a real line",
                     "citations": [{"path": "real.py", "line": 1}]}]},
    ]
    (tmp_path / "real.py").write_text("a\nb\n", encoding="utf-8")

    scored = score(rows, cases, citation_root=tmp_path)
    assert scored["arms"][ARM_B]["citations"] == {"n": 1, "passed": 0, "rate": 0.0,
                                                 "verdicts": [
        {"path": PROSE_CITATION_281, "line": None, "ok": False,
         "reason": "not-a-path"}]}, scored["arms"][ARM_B]["citations"]
    assert scored["arms"][ARM_A]["citations"]["passed"] == 1, scored

    lines, evaluated = format_report(rows, cases, baseline=None, corpus="fp",
                                     model="primary", token_ceiling=120_000,
                                     citation_root=tmp_path)
    text = "\n".join(lines)
    assert evaluated is False and lines[-1].startswith("HEADLINE:"), text
    assert "cannot evaluate" in lines[-1], text
    assert re.search(rf"citations verified\s+{ARM_A}: 1/1\s+{ARM_B}: 0/1", text), text


def test_the_report_command_exits_3_over_prose_citations_instead_of_dying(tmp_path):
    """The acceptance check's own shape, across the CLI boundary.

    `report` over the 2258 artifact used to exit 1 with stdout holding ONLY a
    traceback, because `cmd_report` prints the lines `format_report` collected
    after it returns — so an escaping `OSError` cost every number the study
    recorded. Fixed, the same command prints its `cases:` line and a `HEADLINE:`
    line and exits 3 (no cached retry baseline for the corpus): a refused
    verdict, which is a result, not a crash.
    """
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(c) + "\n" for c in _cases(1)),
                          encoding="utf-8")
    rows_path = tmp_path / "rows.jsonl"
    rows = [
        {"arm": ARM_A, "kind": "verdict", "case_key": "case-0", "seq": 1,
         "chosen_cause": "the diff caused it", "tokens": 5000},
        {"arm": ARM_B, "kind": "verdict", "case_key": "case-0", "seq": 2,
         "chosen_cause": "pre-existing at base 0: pre-existing", "tokens": 9000},
        {"arm": ARM_B, "kind": "tool", "case_key": "case-0", "seq": 3, "tokens": 9000,
         "claims": [{"claim": "prose, not a path",
                     "citations": [{"path": PROSE_CITATION_281}]}]},
    ]
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(REPO / "eval" / "diagnosis_fanout_replay.py"), "report",
         "--cases", str(cases_path), "--rows", str(rows_path),
         "--baseline", str(tmp_path / "no-baseline.json"),
         "--root", str(tmp_path), "--ceiling", "200000"],
        capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert proc.returncode == 3, proc.stdout + proc.stderr[-800:]
    assert "Traceback" not in proc.stdout and "Traceback" not in proc.stderr, (
        proc.stdout + proc.stderr[-800:])
    assert "cases: 1" in proc.stdout, proc.stdout
    assert "HEADLINE: cannot evaluate" in proc.stdout, proc.stdout
    assert re.search(rf"citations verified\s+{ARM_A}: 0/0\s+{ARM_B}: 0/1", proc.stdout), (
        proc.stdout)


# ── clause 3: three theories before any evidence row ────────────────────────

def _fake_arm_b(theories, claims, *, tokens=1000):
    async def generate(case):
        return list(theories)

    async def investigate(case, theory):
        return {"tool": "Task", "claims": claims[theory], "tokens": tokens}

    return generate, investigate


def _rank_stub(order):
    def rank(query, candidates):
        return [{"index": i, "score": 0.9 - 0.1 * i} for i in order]
    return rank


def test_arm_b_writes_three_theories_before_any_evidence_row(tmp_path):
    """The theories block's sequence precedes the first tool row, and every tool
    row carries claims with citations rather than a conclusion.

    This is the anchoring guard made mechanical: a theory set written after a
    log was read is the failure the technique exists to prevent, so the test
    asserts the recorded ORDER, not a promise in a docstring.
    """
    from eval.diagnosis_fanout_replay import RowWriter, run_case_b

    rows_path = tmp_path / "rows.jsonl"
    theories = ["the cert expired", "the socket binding moved", "the upstream is down"]
    claims = {t: [{"claim": f"evidence for {t}",
                   "citations": [{"path": "app/loop.py", "line": 1}]}]
              for t in theories}
    generate, investigate = _fake_arm_b(theories, claims)
    verdict = asyncio.run(run_case_b(_cases(1)[0], generate=generate,
                                    investigate=investigate,
                                    rank_fn=_rank_stub([1, 0, 2]),
                                    writer=RowWriter(rows_path), run_id="t1",
                                    token_ceiling=99_000))
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    theory_rows = [r for r in rows if r["kind"] == "theories"]
    tool_rows = [r for r in rows if r["kind"] == "tool"]
    assert len(theory_rows) == 1 and len(theory_rows[0]["theories"]) == THEORIES_PER_CASE
    assert len(tool_rows) == THEORIES_PER_CASE
    assert theory_rows[0]["seq"] < min(r["seq"] for r in tool_rows), (
        "an investigation row precedes the theories block: the theories were written "
        "after evidence was gathered")
    assert all(r["claims"] and r["claims"][0]["citations"] for r in tool_rows), (
        "a per-theory row handed back prose instead of a citable evidence case")
    assert verdict["kind"] == "verdict" and verdict["chosen_cause"]


def test_a_second_duplicate_theory_still_raises_and_a_prose_answer_is_recorded(tmp_path):
    """Fan-out with a duplicate theory is pure cost, so it still raises; a theory
    that handed back a summary is #2635, and that one is now recorded.

    The two halves were one node because both came out of the validators as a
    `RecordError`, and the second half is precisely the raise #2635 removed: on
    2026-10-12 it propagated out of `run` at theory 2 of case 1 and killed the rest
    of the pass (`2258/smoke-b.log`: `RecordError: theory 'c2' handed back no
    claims`). The duplicate half keeps raising and writes nothing — a theory set
    that is not disjoint means the generator is broken before any money is spent —
    while three prose answers become three refusal rows and the case still finishes
    with its own rank and verdict rows. What is pinned here is that the raise this
    file has always had did not quietly vanish along with the one that had to go;
    `test_a_theory_that_hands_back_no_claims_is_recorded_and_the_fan_out_continues`
    pins the refusal row's shape.
    """
    from eval.diagnosis_fanout_replay import (RowWriter, THEORIES_PER_CASE,
                                              run_case_b)

    claims = {"a": [{"claim": "x", "citations": [{"path": "p.py"}]}]}
    dup = ["same thing", "same thing", "different"]
    generate, investigate = _fake_arm_b(dup, {**claims, "same thing": claims["a"],
                                              "different": claims["a"]})
    with pytest.raises(RecordError):
        asyncio.run(run_case_b(_cases(1)[0], generate=generate, investigate=investigate,
                               rank_fn=_rank_stub([0, 1, 2]),
                               writer=RowWriter(tmp_path / "rows.jsonl"), run_id="t2"))
    assert not (tmp_path / "rows.jsonl").exists(), "a refused case still wrote rows"

    async def prose(case, theory):
        return {"tool": "Task", "claims": [], "tokens": 10}

    asked = []

    def no_engine_to_choose_from(query, candidates):
        asked.append(list(candidates))
        return [{"index": 0}]

    theories = ["one", "two", "three"]
    rows_path = tmp_path / "rows2.jsonl"
    verdict = asyncio.run(run_case_b(
        _cases(1)[0], generate=_fake_arm_b(theories, {})[0], investigate=prose,
        rank_fn=no_engine_to_choose_from, writer=RowWriter(rows_path), run_id="t3",
        token_ceiling=99_000))
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    assert [r["kind"] for r in rows] == (["theories"] + ["refusal"] * THEORIES_PER_CASE
                                         + ["rank", "verdict"]), (
        "three prose answers must be three refusal rows and then this case's own rank "
        f"and verdict rows, not a raise: {[r['kind'] for r in rows]}")
    assert not asked, (
        f"the engine was handed {asked} candidates when every theory refused — with no "
        "evidence under any of them the candidate list is bare labels, which is what "
        "the refusal exists to keep out of a ranking")
    rank_row = next(r for r in rows if r["kind"] == "rank")
    assert rank_row["ok"] is False and rank_row["reason"].startswith("no-evidence"), (
        f"the all-refused rank row reads {rank_row['reason']!r}; it must name the "
        "refusals rather than blame the engine")
    assert verdict["kind"] == "verdict" and verdict["refused"] == sorted(
        slot_labels(_cases(1)[0]["key"])), (
        "the all-refused case wrote a verdict row that does not say all three refused")


def test_the_fan_out_stops_at_the_matched_token_ceiling(tmp_path):
    """Arm B's fan-out is cut off at arm A's ceiling; it may not buy a third dispatch.

    The bound is exact about what a driver can know: it learns a theory's cost
    only when that theory's investigation returns, so it can refuse the NEXT
    dispatch but not the one in flight. The guarantee this pins is therefore
    "stop at the ceiling, overshoot by at most one theory's own cost, never run
    a theory after the ceiling is reached" — 1200 spent against a 1000 ceiling
    with 600-token theories, and no third tool row. `report` prints B's spend
    beside A's so the owed study sees the real ratio instead of trusting the
    ceiling as if it were enforced mid-request.
    """
    from eval.diagnosis_fanout_replay import RowWriter, run_case_b

    theories = ["one", "two", "three"]
    claims = {t: [{"claim": "x", "citations": [{"path": "p.py"}]}] for t in theories}
    generate, investigate = _fake_arm_b(theories, claims, tokens=600)
    rows_path = tmp_path / "rows.jsonl"
    asyncio.run(run_case_b(_cases(1)[0], generate=generate, investigate=investigate,
                           rank_fn=_rank_stub([0, 1, 2]),
                           writer=RowWriter(rows_path), run_id="t4",
                           token_ceiling=1000))
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    tool_rows = [r for r in rows if r["kind"] == "tool"]
    assert len(tool_rows) == 2, (
        f"{len(tool_rows)} theories dispatched against a 1000 ceiling at 600 each: "
        f"the arm must stop at the ceiling, not run the fan-out out in full")
    spent = sum(int(r["tokens"]) for r in tool_rows)
    assert spent == 1200 and spent - 1000 <= 600, spent
    ceiling = [r for r in rows if r["kind"] == "ceiling"]
    assert ceiling and ceiling[0]["reason"] == "token-ceiling"
    assert ceiling[0]["seq"] > tool_rows[-1]["seq"], "the refusal predates the spend"
    # …and a ceiling that actually fits the fan-out must not cut it short.
    roomy = tmp_path / "roomy.jsonl"
    asyncio.run(run_case_b(_cases(1)[0], generate=generate, investigate=investigate,
                           rank_fn=_rank_stub([0, 1, 2]),
                           writer=RowWriter(roomy), run_id="t4b", token_ceiling=120_000))
    roomy_rows = [json.loads(l) for l in roomy.read_text().splitlines() if l.strip()]
    assert len([r for r in roomy_rows if r["kind"] == "tool"]) == 3
    assert not [r for r in roomy_rows if r["kind"] == "ceiling"]


# ── clause 4: neutral names, argmax only ────────────────────────────────────

def test_the_ranking_persists_the_argmax_order_and_no_score(tmp_path):
    """Only the ORDER and the chosen cause reach the record.

    #1452 measured djev's option-name prior growing with the number of options
    and ruled the read-out argmax-only; a persisted score is an invitation for
    the next reader (or the next `report`) to threshold it, so the absence of a
    score in the row is the clause.
    """
    case = _cases(1)[0]
    slots = slot_labels(case["key"])
    evidence = {s: [{"claim": f"evidence {s}", "citations": [{"path": "x.py", "line": 1}]}]
                for s in slots}
    seen: dict = {}

    def rank_fn(query, candidates):
        seen["query"] = query
        seen["candidates"] = list(candidates)
        return [{"index": 2, "score": 0.02}, {"index": 0, "score": 0.97},
                {"index": 1, "score": 0.50}]

    ranking = rank_theories(case, evidence, slots=slots, rank_fn=rank_fn)
    assert ranking["order"] == [slots[2], slots[0], slots[1]], ranking
    assert ranking["chosen_slot"] == slots[2]
    assert ranking["ok"] is True
    blob = json.dumps(ranking)
    assert "score" not in blob, "a djev score reached the record and can be thresholded"

    def rank_fn_reversed(query, candidates):
        return [{"index": 2, "score": 0.99}, {"index": 0, "score": 0.01},
                {"index": 1, "score": 0.50}]

    flipped = rank_theories(case, evidence, slots=slots, rank_fn=rank_fn_reversed)
    assert flipped["chosen_slot"] == ranking["chosen_slot"], (
        "the choice moved with the scores, not with the order: a threshold is in use")


def test_the_option_names_are_neutral_and_the_slots_rotate():
    """A candidate must never be named for its theory, and must not keep a slot.

    Every claim is pinned to an exact value. The version this answers admitted
    `labels == ["c0","c1","c2"] or set(labels) == {...}` (the `or` made a wrong
    ORDER pass), a negated four-word list (any name that merely avoided four
    words passed, including one naming its theory), and `len(rotations) > 1`
    (two distinct orders out of three passed, so a corpus could still lead with
    the same slot most of the time). The rotation is sha256-derived from the
    case key, so these exact lists are reproducible, not a coincidence of run.
    """
    assert slot_labels("case-0") == ["c0", "c1", "c2"]
    assert slot_labels("case-1") == ["c1", "c2", "c0"]
    assert slot_labels("case-2") == ["c1", "c2", "c0"]
    keys = [f"case-{i}" for i in range(12)]
    for key in keys:
        labels = slot_labels(key)
        assert len(labels) == THEORIES_PER_CASE == 3, labels
        # A label is the letter c and one digit, or it is carrying meaning.
        assert all(re.fullmatch(r"c[0-2]", l) for l in labels), labels
        # And every case holds each slot exactly once, so nothing is unnamed.
        assert sorted(labels) == ["c0", "c1", "c2"], labels
        # A rotation, not a shuffle: the order is a cyclic shift of the base.
        assert any(labels == ["c0", "c1", "c2"][i:] + ["c0", "c1", "c2"][:i]
                   for i in range(3)), labels
        assert slot_labels(key) == labels, "the rotation is not reproducible"
    # All three slots lead somewhere across twelve cases: measured census, so a
    # rotation that favoured one position cannot pass this node.
    census = {}
    for key in keys:
        first = slot_labels(key)[0]
        census[first] = census.get(first, 0) + 1
    assert census == {"c0": 3, "c1": 5, "c2": 4}, census


def test_the_ranker_sees_evidence_under_a_neutral_name_and_no_theory_text(tmp_path):
    """What reaches djev is `[cN]` plus the claims: no label, no theory wording.

    #1452 measured djev's option-name prior, so the neutrality that matters is
    of the TEXT being ranked, not only of the label: a candidate whose body
    reads "the TLS certificate expired" hands the engine the same prior under a
    different hat. Driven through `run_case_b` with a nonce in each theory, so
    if the rank call ever starts receiving the theory text this node fails.
    Slot order here is the rotated `["c1","c2","c0"]` (measured for `case-1`),
    while the candidates must go out as `[c0] [c1] [c2]` — generation order
    would put `[c1]` first, so the assertion also pins that a theory's POSITION
    moves with its label rather than staying first for every case.
    """
    from eval.diagnosis_fanout_replay import RowWriter, run_case_b

    case = _cases(2)[1]                      # key "case-1" → slots c1, c2, c0
    assert slot_labels(case["key"]) == ["c1", "c2", "c0"]
    theories = ["nonce-aa the TLS certificate expired",
                "nonce-bb the media device binding moved",
                "nonce-cc the upstream provider is unreachable"]
    # Each theory's evidence is worded independently of the theory: a claim that
    # paraphrased its own theory would make the absence assertions below vacuous.
    evidence_text = {theories[0]: "the renewal ran and wrote a value of 0 at 03:11",
                     theories[1]: "the capture node reports 0 sockets bound",
                     theories[2]: "the peer answered 200 at the same timestamp"}
    citations = {theories[0]: {"path": "app/loop.py", "line": 2},
                 theories[1]: {"path": "app/loop.py", "line": 1},
                 theories[2]: {"path": "app/loop.py", "line": 3}}
    claims = {t: [{"claim": evidence_text[t], "citations": [citations[t]]}]
              for t in theories}
    generate, investigate = _fake_arm_b(theories, claims)
    seen: dict = {}

    def rank_fn(query, candidates):
        seen["candidates"] = list(candidates)
        seen["query"] = query
        return [{"index": 2}, {"index": 1}, {"index": 0}]

    verdict = asyncio.run(run_case_b(case, generate=generate, investigate=investigate,
                                     rank_fn=rank_fn,
                                     writer=RowWriter(tmp_path / "rows.jsonl"),
                                     run_id="neutral", token_ceiling=99_000))
    candidates = seen["candidates"]
    assert [c.splitlines()[0] for c in candidates] == ["[c0]", "[c1]", "[c2]"], candidates
    for text in candidates:
        for nonce in ("nonce-aa", "nonce-bb", "nonce-cc"):
            assert nonce not in text.lower(), (
                "a candidate was ranked by its theory's wording, not its evidence")
        assert "certificate" not in text and "media device" not in text
    # Sorted by label, so candidate 0 is slot c0 — which for THIS case's
    # rotation (c1, c2, c0) is the third theory generated, holding its evidence.
    assert "app/loop.py" in candidates[0], "the citation is not in the ranked text"
    assert evidence_text[theories[2]] in candidates[0], candidates[0]
    assert evidence_text[theories[0]] in candidates[1], candidates[1]
    assert case["claimed_cause"] not in candidates[0]
    assert case["input"] in seen["query"], "the ranking is not asked about the incident"
    # The engine put candidate index 2 first; sorted-by-label index 2 is slot
    # c2, and under this rotation c2 holds the SECOND generated theory.
    assert verdict["chosen_cause"] == theories[1], (
        "the argmax of the sorted candidate list does not map back to its theory")


def test_a_dead_engine_leaves_no_ranking_and_says_so():
    """`djev.rank` returns None when it did not answer; that is not a tie."""
    case = _cases(1)[0]
    slots = slot_labels(case["key"])
    out = rank_theories(case, {s: [] for s in slots}, slots=slots,
                        rank_fn=lambda q, c: None)
    assert out["ok"] is False and out["chosen_cause"] is None
    assert "engine" in out["reason"]


# ── clause 5: re-grade without the engine, baseline can refuse ──────────────

def _recorded_rows(tmp_path, *, arm_b_citations=(("real.py", 1), ("real.py", 2)),
                   chosen_b="pre-existing at base 1: pre-existing"):
    (tmp_path / "real.py").write_text("a\nb\nc\n", encoding="utf-8")
    cases = _cases(2)
    rows = []
    for i, case in enumerate(cases):
        rows.append({"seq": 1, "run_id": "r", "case_key": case["key"], "arm": ARM_A,
                     "kind": "verdict", "chosen_cause": "the diff caused it",
                     "tokens": 5000})
        rows.append({"seq": 2, "run_id": "r", "case_key": case["key"], "arm": ARM_A,
                     "kind": "tool", "claims": [], "tokens": 5000})
        chosen = chosen_b if i == 0 else "the diff caused it"
        rows.append({"seq": 3, "run_id": "r", "case_key": case["key"], "arm": ARM_B,
                     "kind": "verdict", "chosen_cause": chosen, "tokens": 9000})
        rows.append({"seq": 4, "run_id": "r", "case_key": case["key"], "arm": ARM_B,
                     "kind": "tool", "slot": "c0", "tokens": 9000,
                     "claims": [{"claim": f"c{i}", "citations": [
                         {"path": p, "line": n} for p, n in arm_b_citations]}]})
    return cases, rows


def test_the_report_regrades_rows_with_no_engine_and_prints_every_number(tmp_path):
    """Top-1 per arm, the citation rate and both arms' spend, from rows alone.

    The retry arm is a recorded baseline, and its spend and top-1 must appear
    beside the live arms' — otherwise the compute-matched comparison (#627) has
    one side missing and the margin is meaningless.
    """
    cases, rows = _recorded_rows(tmp_path)
    baseline = {"kind": "retry_baseline", "corpus": "fp", "n": 2, "model": "primary",
                "token_ceiling": 120_000, "token_spend": 11_000, "top1_hits": 1}
    lines, evaluated = format_report(rows, cases, baseline=baseline, corpus="fp",
                                     model="primary", token_ceiling=120_000,
                                     citation_root=tmp_path)
    text = "\n".join(lines)
    assert evaluated is True
    # Every claim here is a LABELLED NUMBER, not an arm's name: `"A" in text` is
    # true of the report header alone and can never fail, so it pinned nothing.
    assert "top-1 agreement" in text and "citations verified" in text
    assert re.search(rf"top-1 agreement\s+{ARM_A}: 0/2\s+{ARM_B}: 1/2", text), text
    assert re.search(rf"citations verified\s+{ARM_A}: 0/0\s+{ARM_B}: 4/4", text), text
    assert re.search(rf"token spend\s+{ARM_A}: 10000\s+{ARM_B}: 18000", text), text
    assert re.search(rf"token spend\s+{ARM_RETRY}: 11000", text), text
    assert re.search(rf"top-1 agreement {ARM_RETRY}: 1/2", text), text
    assert "B citation rate 100.0%" in text, text
    assert "margin over the better single-trace arm: 0 case(s)" in text, (
        "the margin is not measured against the better of the two single-trace arms")
    assert "HEADLINE" in text
    assert "arm B does not win" in text, (
        "a margin of 0 cases passed a bar that asks for 2 — the bar is not applied")


def test_the_report_refuses_to_judge_without_a_matching_retry_baseline(tmp_path):
    """Missing or mismatched baseline: `cannot evaluate`, never a pass (#627's rule).

    All four mismatches go through the same sentence, because a study whose
    baseline was measured on another corpus, another N, another model or another
    ceiling has not matched compute.
    """
    cases, rows = _recorded_rows(tmp_path)
    for baseline, corpus in ((None, "fp"), ({"corpus": "other", "n": 2,
                                             "model": "primary", "token_ceiling": 120_000,
                                             "token_spend": 1, "top1_hits": 0}, "fp"),
                             ({"corpus": "fp", "n": 1, "model": "primary",
                               "token_ceiling": 120_000, "token_spend": 1,
                               "top1_hits": 0}, "fp"),
                             ({"corpus": "fp", "n": 2, "model": "other-model",
                               "token_ceiling": 120_000, "token_spend": 1,
                               "top1_hits": 0}, "fp")):
        lines, evaluated = format_report(rows, cases, baseline=baseline, corpus=corpus,
                                         model="primary", token_ceiling=120_000,
                                     citation_root=tmp_path)
        text = "\n".join(lines)
        assert evaluated is False
        assert "cannot evaluate" in text
        assert "arm B wins" not in text, text
        assert "HEADLINE: cannot evaluate" in text


def test_a_matched_baseline_and_a_winning_margin_prints_a_win(tmp_path):
    """The positive half, so the refusal test is not the only thing that can pass."""
    cases, rows = _recorded_rows(tmp_path, chosen_b="pre-existing at base 0: pre-existing")
    # Both cases now agree: rewrite arm B's second verdict too.
    for row in rows:
        if row["arm"] == ARM_B and row["kind"] == "verdict":
            row["chosen_cause"] = "pre-existing at base 0: pre-existing"
    baseline = {"corpus": "fp", "n": 2, "model": "primary", "token_ceiling": 120_000,
                "token_spend": 11_000, "top1_hits": 0}
    lines, evaluated = format_report(rows, cases, baseline=baseline, corpus="fp",
                                     model="primary", token_ceiling=120_000,
                                     citation_root=tmp_path)
    text = "\n".join(lines)
    assert evaluated is True and "arm B wins" in text, text


def test_the_baseline_file_round_trips_its_fingerprint(tmp_path):
    """The cache the report reads is written with the corpus it was measured on."""
    from eval.diagnosis_fanout_replay import corpus_fingerprint

    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(_cases(1)[0]) + "\n", encoding="utf-8")
    path = tmp_path / "retry_baseline.json"
    record = write_baseline(path, cases_path=cases_path, n=1, model="primary",
                            token_ceiling=120_000, token_spend=4242, top1_hits=0)
    assert record["corpus"] == corpus_fingerprint(cases_path)
    on_disk = json.loads(path.read_text())
    assert on_disk["token_spend"] == 4242
    assert baseline_problem(on_disk, corpus=corpus_fingerprint(cases_path), n=1,
                            model="primary", token_ceiling=120_000) is None
    assert baseline_problem(on_disk, corpus=corpus_fingerprint(cases_path), n=2,
                            model="primary", token_ceiling=120_000)


def test_report_of_the_real_paths_needs_no_engine(tmp_path):
    """`report` runs end to end against the shipped defaults on an empty row file.

    The owed study runs this exact command later; it must not fail because a
    live artifact is missing, and it must not invent a pass — it prints the
    refusal, since no baseline has been measured yet.
    """
    proc = subprocess.run(
        [sys.executable, str(REPO / "eval" / "diagnosis_fanout_replay.py"), "report",
         "--rows", str(tmp_path / "no-rows.jsonl"),
         "--cases", str(tmp_path / "no-cases.jsonl")],
        capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "cannot evaluate" in proc.stdout, proc.stdout


# ── #2466: run and re-grade the 21 non-leaky cases honestly ─────────────────
#
# The owed study runs `run --only <21 keys>`, then `baseline`, then `report`, over the
# 98-row corpus at `~/lloyd-data/eval/owedcheck-2258-cases.jsonl` (census: 16 `flaky-*`
# in family `parallel-load-flake`, 5 `landkill-*` in `landing-killed-externally`, and 77
# `preexisting-*` in `tests-blamed-then-refuted`, which are the leaky ones #2258 excludes
# because their input quotes the refuted claim). Three things stopped that sequence
# before it could spend its ~2.6 engine-hours: arm A recorded an empty cause, `report`
# could not cut the corpus, and neither the exclusions nor the ceiling were in the
# artifact. `_study_corpus` below is that 98/21 shape at fixture size — same key prefixes
# and the same three family names, so the numbers the report prints are the numbers the
# real file will make it print.

FLAKY, LANDKILL, LEAKY = 16, 5, 77
NON_LEAKY = FLAKY + LANDKILL


def _study_case(prefix: str, i: int, family: str, marker: str) -> dict:
    return {"key": f"{prefix}-SM_2026100{i % 7}_{i:06d}", "family": family,
            "subsystem": "single-service",
            "input": f"round {i} came back red on {marker}",
            "claimed_cause": f"the diff caused failure {i}",
            "actual_cause": f"{marker} at base: {marker}",
            "actual_cause_markers": [marker],
            "source_ref": f"promotions.jsonl:{i + 1} (round SM_{i})"}


def _study_corpus() -> list[dict]:
    """98 cases in the real file's shape: 16 `flaky-`, 5 `landkill-`, 77 `preexisting-`."""
    return ([_study_case("flaky", i, "parallel-load-flake", "parallel-load-flake")
             for i in range(FLAKY)]
            + [_study_case("landkill", i, "landing-killed-externally", "killed-by-signal")
               for i in range(LANDKILL)]
            + [_study_case("preexisting", i, "tests-blamed-then-refuted", "pre-existing")
               for i in range(LEAKY)])


def _non_leaky_keys(cases: list[dict]) -> list[str]:
    return [c["key"] for c in cases if c["family"] != "tests-blamed-then-refuted"]


def _study_rows(cases: list[dict], only: list[str]) -> list[dict]:
    """A-retry/A/B rows for the graded subset, every case correct in every arm.

    Agreement is not what these rows are for — the spend columns and the N are — so all
    three arms record the true mechanism and the headline is `arm B does not win` at
    margin 0, which is an evaluation and not a refusal.
    """
    rows, seq = [], 0
    for case in cases:
        if case["key"] not in only:
            continue
        cause = case["actual_cause"]
        for arm, spend in ((ARM_A, 40_000), (ARM_RETRY, 41_000), (ARM_B, 90_000)):
            seq += 1
            rows.append({"seq": seq, "run_id": "r2466", "case_key": case["key"],
                         "arm": arm, "kind": "tool", "slot": None, "claims": [],
                         "tokens": spend})
            seq += 1
            rows.append({"seq": seq, "run_id": "r2466", "case_key": case["key"],
                         "arm": arm, "kind": "verdict", "chosen_cause": cause,
                         "tokens": spend})
    return rows


def test_a_single_trace_arm_records_the_cause_its_investigator_actually_returned(tmp_path):
    """Clause 1: arm A's verdict carries a real cause, not the empty string.

    The fake returns exactly what `investigate_live` returns — `tool`, `claims`, `tokens`,
    `model` — because that missing `conclusion` key IS the bug this node pins: the arm read
    `result["conclusion"]`, which the live investigator has never returned, so every A and
    A-retry row recorded `chosen_cause: ""`, `agrees()` scored them all wrong, and the
    margin line could print `arm B wins` off a wiring fault (#2258's review advisory of
    2026-10-06, never fixed until now).
    """
    case = _study_corpus()[0]
    rows_path = tmp_path / "rows.jsonl"

    async def investigator(case_arg, theory):
        assert case_arg is case
        return {"tool": "Task", "model": "primary", "tokens": 4321,
                "claims": [{"claim": "parallel-load-flake at base: parallel-load-flake",
                            "citations": [{"path": "eval/diagnosis_fanout_replay.py",
                                           "line": 10}]}]}

    verdict = asyncio.run(run_case_a(case, investigate=investigator,
                                     writer=RowWriter(rows_path), run_id="r1"))

    assert verdict["chosen_cause"], "arm A recorded no cause: top-1 can only be 0/N"
    assert "parallel-load-flake" in verdict["chosen_cause"]
    assert agrees(case, verdict["chosen_cause"]), (
        "the recorded cause must be the one the fixture says is correct, or this node "
        "is proving that a string is non-empty and nothing else")
    recorded = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    tool = [r for r in recorded if r["kind"] == "tool"]
    assert len(tool) == 1, recorded
    assert tool[0]["tokens"] == 4321
    assert tool[0]["claims"][0]["citations"] == [
        {"path": "eval/diagnosis_fanout_replay.py", "line": 10}], (
        "the citations are what arm B is scored on; if arm A's are dropped the citation "
        "column is a zero denominator, not a 0% rate")


def test_a_single_trace_arm_reads_the_older_shape_and_refuses_a_silent_answer(tmp_path):
    """The older `conclusion` + `citations` shape is still graded; a silent answer raises.

    Two halves, both needed. An investigator that answers with one conclusion and its
    citations beside it loses neither — that path predates this item, and dropping it would
    turn a reply this arm can grade into a refused run. And an investigator that produces no
    cause at all raises instead of recording a `chosen_cause` filled in from the case's own
    answer key, which would hand the instrument a 100% top-1 it never measured. Arm B's row
    validator, `validate_investigation`, already refuses a record with no claims for the
    same reason; the single-trace arms now treat an empty answer alike as well.
    """
    case = _study_corpus()[1]

    async def old_shape(case_arg, theory):
        return {"tool": "investigate", "tokens": 100, "model": "primary",
                "conclusion": "parallel-load-flake at base: parallel-load-flake",
                "citations": [{"path": "eval/README.md", "line": 1}]}

    verdict = asyncio.run(run_case_a(case, investigate=old_shape,
                                     writer=RowWriter(tmp_path / "a.jsonl"), run_id="r2"))
    assert agrees(case, verdict["chosen_cause"])
    written = [json.loads(ln) for ln in (tmp_path / "a.jsonl").read_text().splitlines()]
    tool = next(r for r in written if r["kind"] == "tool")
    assert [c["citations"] for c in tool["claims"]] == [
        [{"path": "eval/README.md", "line": 1}]], "the older shape lost its citations"

    async def silent(case_arg, theory):
        return {"tool": "investigate", "tokens": 100, "model": "primary", "claims": []}

    with pytest.raises(RuntimeError, match="no non-empty claim"):
        asyncio.run(run_case_a(case, investigate=silent,
                               writer=RowWriter(tmp_path / "b.jsonl"), run_id="r3"))
    refused = tmp_path / "b.jsonl"
    assert not (refused.exists() and refused.read_text().strip()), (
        "the refused cell left rows on disk, and a recorded cell with no cause is still "
        "counted into the denominator as though it had been measured")


# ── #2984: arm A's no-claims raise names its mechanism ───────────────────────
#
# Arm B has said WHICH refusal it got since #2807 (`turn-cap` versus `no-claims`,
# chosen from the `stop_reason`/`num_turns` pair `investigate_live` re-emits). Arm
# A, wired to that same investigator by `cmd_run`, raised one message for both: a
# run the harness cut off at its turn budget and a model that finished with nothing
# to record printed the same line, and the only reader of that line is
# `cmd_run`'s per-arm guard in the pass's `run.log`. The three nodes below pin the
# raise's own TEXT for each of the three shapes an investigator can hand back —
# capped, stopped, and reporting no stop at all.

def test_a_turn_capped_arm_a_raises_the_truncation_message_with_its_turn_count(
        tmp_path):
    """Clause 1: `stop_reason='max_turns'` + `num_turns=40` says truncation, in words.

    The message has to carry the pair AND the cause. `max_turns` alone would be a
    bare value a reader has to interpret, and arm B's `no_claims_message` wording
    — "handed back no claims", which means the model answered and declined to cite
    — is the specific misreading #2807 put a separate sentence in for: putting it
    on a truncated dispatch sends a prompt follow-up at a run that never got to
    answer. So the truncation sentence is asserted present and the prose-declined
    sentence absent, and the raise itself is unchanged: a `RuntimeError`, with the
    base complaint still in it, and no row on disk.
    """
    case = _study_corpus()[1]
    rows_path = tmp_path / "capped.jsonl"

    async def capped(case_arg, theory):
        return {"tool": "Task", "model": "primary", "tokens": 179_186,
                "claims": [], "stop_reason": "max_turns", "num_turns": 40}

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(run_case_a(case, investigate=capped,
                               writer=RowWriter(rows_path), run_id="c2984cap"))
    msg = str(raised.value)
    assert "no non-empty claim" in msg, msg
    assert "max_turns" in msg, msg
    assert "40-turn budget" in msg and "40" in msg, msg
    assert "harness cut the run off" in msg, msg
    assert "handed back no claims" not in msg, (
        f"the message reads {msg!r} — that is arm B's prose-declined wording, and "
        "putting it on a truncated dispatch is the misreading #2807 forbids")
    assert not (rows_path.exists() and rows_path.read_text().strip()), (
        "the truncated cell left rows on disk; the raise still writes no verdict")


def test_a_stopped_arm_a_reports_the_stop_and_no_turn_cap_wording(tmp_path):
    """Clause 2: `stop_reason='stop'` + `num_turns=27` says the model came up empty.

    The negative half is the point. Arm A's message is built from the same pair, so
    a reader who has only `run.log` must be able to tell this dispatch from the
    capped one above — hence no `max_turns`, no "turn budget", no "ran out of
    turns", no "harness cut the run off", no word of truncation anywhere in a
    message about a run that reached its stop. The two numbers the clause names are
    asserted present so the message stays a measurement rather than a guess.
    """
    case = _study_corpus()[1]
    rows_path = tmp_path / "stopped.jsonl"

    async def stopped(case_arg, theory):
        return {"tool": "Task", "model": "primary", "tokens": 100,
                "claims": [], "stop_reason": "stop", "num_turns": 27}

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(run_case_a(case, investigate=stopped,
                               writer=RowWriter(rows_path), run_id="c2984stp"))
    msg = str(raised.value)
    assert "no non-empty claim" in msg, msg
    assert "stop" in msg and "27" in msg, msg
    assert "max_turns" not in msg, msg
    for capped_wording in ("turn budget", "ran out of turns", "cut the run off",
                           "truncat", "turn-cap"):
        assert capped_wording not in msg.lower(), (
            f"{capped_wording!r} appears in a message about a run that reached its "
            f"stop: {msg!r}")
    assert not (rows_path.exists() and rows_path.read_text().strip()), (
        "the refused cell left rows on disk")


def test_an_arm_a_investigator_reporting_no_stop_keeps_the_plain_complaint(tmp_path):
    """Clause 3: no pair reported, no wording invented — and no error of its own.

    Absence is not evidence of a cap (#2807), so an investigator that names neither
    field gets the complaint exactly as it read before this item: the two fakes
    above had to be written, the two fakes below always existed and always raised
    `RuntimeError, match="no non-empty claim"`. What is new here is the shapes that
    could make BUILDING the message fail — a `num_turns` that is not a number, a
    cap with no count beside it — because a `TypeError` escaping the raise would
    replace the complaint rather than improve it. `pytest.raises` is the pin: any
    `KeyError`/`TypeError`/`ValueError` escapes it and fails the node.
    """
    case = _study_corpus()[1]
    rows_path = tmp_path / "nostop.jsonl"

    async def bare(case_arg, theory):
        return {"tool": "Task", "tokens": 100, "claims": []}

    with pytest.raises(RuntimeError, match="no non-empty claim") as raised:
        asyncio.run(run_case_a(case, investigate=bare,
                               writer=RowWriter(rows_path), run_id="c2984bare"))
    msg = str(raised.value)
    assert msg == (
        f"[2258] A: the investigator for case {case['key']} returned no non-empty "
        f"claim, so there is no cause to record and inventing one is the one thing "
        f"this instrument must not do. Keys it did return: "
        f"['claims', 'tokens', 'tool']."), msg

    async def junk_turns(case_arg, theory):
        return {"tool": "Task", "tokens": 100, "claims": [],
                "stop_reason": "stop", "num_turns": "many"}

    with pytest.raises(RuntimeError, match="no non-empty claim") as raised:
        asyncio.run(run_case_a(case, investigate=junk_turns,
                               writer=RowWriter(rows_path), run_id="c2984junk"))
    assert "stop_reason 'stop'" in str(raised.value), str(raised.value)
    assert "many" not in str(raised.value), (
        "a non-numeric turn count was printed as though it were one")

    async def capped_without_a_count(case_arg, theory):
        return {"tool": "Task", "tokens": 100, "claims": [],
                "stop_reason": "max_turns"}

    with pytest.raises(RuntimeError, match="no non-empty claim") as raised:
        asyncio.run(run_case_a(case, investigate=capped_without_a_count,
                               writer=RowWriter(rows_path), run_id="c2984nocount"))
    assert "at its turn budget" in str(raised.value), str(raised.value)
    assert "None-turn" not in str(raised.value), str(raised.value)
    assert not (rows_path.exists() and rows_path.read_text().strip()), (
        "none of the three refusals may leave a row")


# ── #2624: the live wiring's budgets ────────────────────────────────────────
#
# `investigate_live`'s 20-turn default and the 120,000 ceiling were read by
# nothing until this section: `cmd_run` calls the investigator with no `max_turns`
# and the CLI's `--ceiling` defaults to the module constant, so both numbers were
# the study's real configuration while no test could see them. That is how the
# study died on its first case — see each node's docstring for the measurement.

def test_the_live_investigation_budget_reaches_the_engine_turn_it_must_fit_in(
        monkeypatch):
    """Clause 1: the budget the arm asks for is the budget the harness is given.

    Two boundaries in one node because the failure lived between them. The default is
    read off `investigate_live`'s own signature — `cmd_run` never passes `max_turns`,
    so the default IS the study's budget — and is then followed across into
    `_engine_turn`, the single call that hands the number to
    `app.harness.loop.run_query`. A signature saying 60 over a call that dropped the
    argument would satisfy a signature-only test and still stop at the harness's own
    default, and nothing in the file would have looked wrong.

    The value is `DEFAULT_INVESTIGATION_TURNS`, named at module level for #2985 so a
    reader sees it is one arithmetic property with `DEFAULT_TOKEN_CEILING` rather than
    a bare literal in a signature. Its floor is the measured pair, which brackets the
    cost instead of averaging it out: investigating `flaky-SM_20260928_080412` on this
    engine reached `stop_reason: stop` at 27 turns
    (`~/lloyd-data/eval/2258/probe-maxturns-40.log`,
    `META: {"stop_reason": "stop", "num_turns": 27, ...}`), while smoke-b3's theory
    was cut off at `num_turns: 41` with `stop_reason: max_turns`
    (`~/lloyd-data/eval/2258/smoke-b3.jsonl`, line 2). Running out of turns is not a
    shorter answer, it is no answer: the finalizer restates a `final_schema` only for
    `stop`/`end_turn` (`FINALIZABLE_STOP_REASONS`, `app/harness/finalizer.py:78`,
    "no verdict to restate"), so the exhausted arm got `structured: {}`, hence
    `claims: []`, hence a refusal that read as "the model found no evidence".
    """
    case = _study_corpus()[0]
    seen: list[dict] = []

    async def fake_turn(prompt, *, schema, session_id, max_turns):
        seen.append({"schema": schema, "session_id": session_id,
                     "max_turns": max_turns})
        return {"structured": {"claims": [{"claim": "parallel-load-flake at base",
                                           "citations": []}]},
                "tokens": 179_186, "model": "primary"}

    monkeypatch.setattr(replay, "_engine_turn", fake_turn)

    default = inspect.signature(investigate_live).parameters["max_turns"].default
    assert default is DEFAULT_INVESTIGATION_TURNS, (
        f"the budget reached the harness as {default}, which is not the named module "
        "constant — a bare number in the signature is what #2985 removed, because the "
        "turn cap and DEFAULT_TOKEN_CEILING are one arithmetic property and a literal "
        "in a signature hides the pairing")
    assert default >= 60, (
        f"the live investigator's turn budget is {default}; smoke-b3's theory was cut "
        "off at num_turns 41 with stop_reason max_turns, so a cap at or below 41 is "
        "the truncation on record rather than a budget above it")

    out = asyncio.run(investigate_live(case, "parallel-load-flake at base"))

    assert len(seen) == 1, seen
    assert seen[0]["max_turns"] == DEFAULT_INVESTIGATION_TURNS, (
        f"the signature says {DEFAULT_INVESTIGATION_TURNS} but the harness call got "
        f"{seen[0]['max_turns']}")
    assert seen[0]["schema"] is replay.CLAIMS_SCHEMA, (
        "an investigation answers the claims schema; a theories schema here would "
        "silently return no claims for a perfectly good trace")
    assert seen[0]["session_id"].startswith("bench_"), (
        "a non-bench session id is not sandboxed, so an investigation could write")
    # An explicit budget still wins over the default — the probe that measured 27
    # turns was itself run at a different cap, and that has to stay possible.
    asyncio.run(investigate_live(case, "parallel-load-flake at base", max_turns=7))
    assert seen[1]["max_turns"] == 7, seen[1]
    # And the answer arrives in the shape `run_case_a` reads, since that arm's whole
    # bug history is a key the live investigator never returned.
    assert out["claims"][0]["claim"] == "parallel-load-flake at base", out
    assert out["tokens"] == 179_186 and out["model"] == "primary", out


def test_one_completed_investigation_fits_the_default_token_ceiling(tmp_path):
    """Clause 2: one finished arm-A trace is under the ceiling the module ships with.

    179,186 tokens is what the single completed investigation of a #2258 case spent on
    this engine (`~/lloyd-data/eval/2258/probe-maxturns-40.log`,
    `CAP=40 claims_n=1 tokens=179186`) against the 120,000 this module shipped. The
    over-ceiling case is not a warning: `run_case_a` appends a second row of
    `kind: ceiling` for it, so every arm-A result arrived tied to a refusal and
    `format_report`'s spend line printed the same shape for both arms — the one axis
    the arms are matched on went flat, which is what this raise is for.

    No `token_ceiling` argument is passed below on purpose: the thing under test is the
    module default, which is also `run --ceiling`'s default. Arm B's fan-out
    enforcement — the ceiling must actually cut the fan-out off — stays pinned by
    `test_the_fan_out_stops_at_the_matched_token_ceiling`, which passes its own number
    and is untouched by this raise.
    """
    assert DEFAULT_TOKEN_CEILING >= 180_000, (
        f"the default ceiling is {DEFAULT_TOKEN_CEILING}; one completed "
        f"investigation measured 179,186 tokens, so at or under that every arm-A "
        f"row is also an over-ceiling row")
    case = _study_corpus()[0]
    rows_path = tmp_path / "rows.jsonl"

    async def measured_spend(case_arg, theory):
        return {"tool": "Task", "model": "primary", "tokens": 179_186,
                "claims": [{"claim": "parallel-load-flake at base: parallel-load-flake",
                            "citations": [{"path": "eval/README.md", "line": 1}]}]}

    verdict = asyncio.run(run_case_a(case, investigate=measured_spend,
                                     writer=RowWriter(rows_path), run_id="c24a"))

    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["tool", "verdict"], (
        f"a third row means the measured spend tripped the default ceiling: {rows}")
    assert verdict["tokens"] == 179_186
    assert agrees(case, verdict["chosen_cause"])


# ── #2985: the fan-out's budget has to fit three theories ───────────────────
#
# The ceiling is CUMULATIVE and read before each dispatch (`spent >= token_ceiling`
# in `run_case_b`), so it is a budget for the whole three-theory fan-out, not per
# theory: a theory's cost is known only when it returns, so the arm can refuse the
# NEXT dispatch and never the one in flight. 200,000 was therefore below arithmetic —
# one measured theory costs 161,011 tokens
# (`~/lloyd-data/eval/2258/smoke-b3.jsonl`, line 2, `run-20261018T220131Z`), so three
# of THEM cost 483,033 and could never fit. What the old figure did to the record is
# in `~/lloyd-data/eval/2258/rows.jsonl`: all 13 of that file's `kind: ceiling` rows
# are `seq: 4` with `tokens_spent` between 280,279 and 362,808 against 200,000, so the
# third theory of every case was refused, and `rank_theories` deliberately keeps a
# ceiling-cut slot in the candidate list — the 2026-10-13 pass then wrote a verdict
# whose chosen cause had no evidence row under it. The turn cap is the same property
# read from the other side: smoke-b3's truncation happened at `num_turns: 41` under
# the cap of 40 this module shipped, and a longer dispatch costs more tokens.
#
# The two numbers below are therefore ONE decision. The owed live pass (#2985 clause
# 1) is what measures a theory's cost at 60 turns; until it reports, the ceiling is
# sized against the worst dispatch RECORDED, which was taken under a 40-turn cap.

#: What the ceiling must clear: three of the most expensive single dispatch in the
#: study's own ledger (199,283 tokens, `~/lloyd-data/eval/2258/rows.jsonl`, line 126).
WORST_DISPATCH_TOKENS = 199_283

#: The trio the fan-out test below replays: two theories at smoke-b3's measured cost
#: plus the third-worst dispatch on record. 161,011 + 184,976 + 161,011.
MEASURED_TRIO_TOKENS = (161_011, 184_976, 161_011)


def test_the_default_ceiling_clears_three_of_the_worst_dispatch_on_record():
    """Clause 1: `DEFAULT_TOKEN_CEILING` is a named constant >= 3 x the worst theory.

    The floor is arithmetic, not taste: the ceiling is charged `THEORIES_PER_CASE`
    times before the fan-out can finish, so anything under
    `THEORIES_PER_CASE x WORST_DISPATCH_TOKENS` (= 597,849) refuses the third theory
    of a case that spent its worst-ever dispatch on each slot. The item's own stated
    floor, 3 x 184,976 = 554,928, is checked alongside it because that is the figure
    the clause names; the ledger's actual maximum is higher, at 199,283, so the
    stronger bound is the one the constant has to clear — and 600,000 clears it by
    2,151 tokens, which is arithmetic rather than comfort. A test that asserted only
    `>= 600_000` would pass on 6,000,000 and tell nobody the number is a floor with a
    derivation.
    """
    assert DEFAULT_TOKEN_CEILING >= THEORIES_PER_CASE * 184_976, (
        f"{DEFAULT_TOKEN_CEILING} is under the item's own floor of "
        f"{THEORIES_PER_CASE * 184_976}: three dispatched theories do not fit, so the "
        "arm records a ceiling row and the case reaches `report` with a slot that has "
        "no evidence under it")
    assert DEFAULT_TOKEN_CEILING >= THEORIES_PER_CASE * WORST_DISPATCH_TOKENS, (
        f"{DEFAULT_TOKEN_CEILING} is under 3 x the worst dispatch in the study's "
        f"ledger ({THEORIES_PER_CASE * WORST_DISPATCH_TOKENS} = "
        f"{THEORIES_PER_CASE} x {WORST_DISPATCH_TOKENS}, rows.jsonl line 126)")
    assert sum(MEASURED_TRIO_TOKENS) < DEFAULT_TOKEN_CEILING, (
        f"the measured trio {MEASURED_TRIO_TOKENS} sums to "
        f"{sum(MEASURED_TRIO_TOKENS)}, which the shipped ceiling cannot hold")


def test_three_measured_theory_costs_all_dispatch_at_the_default_ceiling(tmp_path):
    """Clause 3: three real theory costs, three evidence rows, one verdict, no ceiling.

    The costs are the measured ones (`MEASURED_TRIO_TOKENS`: smoke-b3's 161,011 twice
    and the 184,976 dispatch at `rows.jsonl` line 50), not round numbers, because the
    failure this guards is arithmetic — 506,998 tokens does not fit under 200,000 and
    does fit under 600,000. No `kind: ceiling` row is the whole clause: that row is
    what made `report` score a case whose third slot was never investigated, and it is
    written by the `spent >= token_ceiling` break, which fires on the CUMULATIVE spend.
    The assertion therefore uses the default ceiling by omission as well as by value —
    a test that passed `token_ceiling=` itself would only prove the arithmetic, not that
    the module's own default is the figure that holds.
    """
    case = _study_corpus()[0]
    root = "parallel-load-flake"          # case 0's own expected-cause marker
    labels = slot_labels(case["key"])
    claims = {t: [{"claim": root, "citations": [{"path": "eval/README.md",
                                                 "line": 1}]}] for t in labels}
    spent = list(MEASURED_TRIO_TOKENS)

    async def generate(c):
        return list(labels)

    async def investigate(c, theory):
        return {"tool": "Task", "claims": claims[theory],
                "tokens": spent[labels.index(theory)]}

    rows_path = tmp_path / "rows.jsonl"
    verdict = asyncio.run(run_case_b(case, run_id="r-trio", generate=generate,
                                     investigate=investigate, writer=RowWriter(rows_path),
                                     rank_fn=_rank_stub([0, 1, 2])))
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    kinds = [r["kind"] for r in rows]
    assert kinds == ["theories", "tool", "tool", "tool", "rank", "verdict"], (
        f"{kinds}: a `ceiling` row here is the bug — the third dispatch would have "
        "been refused for having no budget left, and rank_theories keeps a "
        "ceiling-cut slot in the candidate list anyway")
    assert [r["tokens"] for r in rows if r["kind"] == "tool"] == list(
        MEASURED_TRIO_TOKENS), "the fan-out did not pay the measured costs it was run at"
    assert verdict["tokens"] == sum(MEASURED_TRIO_TOKENS) == 506_998
    assert [r["kind"] for r in rows if r["kind"] == "refusal"] == [], (
        "a theory was refused mid-fan-out, so the case is not three measured theories")
    assert verdict.get("refused", []) == [], (
        f"slots were refused at the default ceiling: {verdict.get('refused')}")
    # All three slots reached the ranker, in slot order, and the top pick is a theory
    # that was actually dispatched. (`agrees()` grades an arm-A claim against the
    # case's markers; arm B's verdict carries the slot ORDER, which is what the
    # ceiling row used to corrupt by leaving an unevidenced slot in the candidates.)
    assert verdict["order"] == ["c0", "c1", "c2"], verdict["order"]
    assert verdict["chosen_cause"] == labels[0], verdict["chosen_cause"]


def test_the_recorded_truncation_still_refuses_with_its_stop_pair_at_the_new_budget(
        tmp_path):
    """Clause 4: smoke-b3's truncated theory still lands as a turn-cap refusal row.

    Row-for-row the input is the witness itself — `claims: []`, `stop_reason:
    max_turns`, `num_turns: 41`, 161,011 tokens
    (`~/lloyd-data/eval/2258/smoke-b3.jsonl`, line 2) — because the thing being
    protected is the row that witness IS: #2807's `REFUSAL_TURN_CAP` path plus the
    `stop_reason`/`num_turns` cells `investigate_live` re-emits. A bigger turn cap and
    a bigger ceiling may change WHEN a run is truncated, never how a truncation is
    recorded: `num_turns` populated is what lets a later reader see the cap was
    crossed rather than met. And the verdict's `refused` list has to name the slot, or
    the case reads as three investigated theories with one that silently declined.
    """
    case = _study_corpus()[0]
    root = "parallel-load-flake"          # case 0's own expected-cause marker
    labels = slot_labels(case["key"])
    truncated = labels[1]
    claims = {t: ([{"claim": root, "citations": [{"path": "eval/README.md",
                                                  "line": 1}]}]
                  if t != truncated else []) for t in labels}

    async def generate(c):
        return list(labels)

    async def investigate(c, theory):
        if theory == truncated:
            return {"tool": "Task", "model": "primary", "tokens": 161_011,
                    "claims": [], "stop_reason": "max_turns", "num_turns": 41}
        return {"tool": "Task", "model": "primary", "tokens": 161_011,
                "claims": claims[theory], "stop_reason": "stop", "num_turns": 27}

    rows_path = tmp_path / "rows.jsonl"
    verdict = asyncio.run(run_case_b(case, run_id="r-cap", generate=generate,
                                     investigate=investigate, writer=RowWriter(rows_path),
                                     rank_fn=_rank_stub([0, 1])))
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    assert [r["kind"] for r in rows if r["kind"] == "ceiling"] == [], (
        "the turn-cap refusal also wrote a token-ceiling row: the two records are "
        "different failures and one row each is the whole point of #2807")
    refusal = next(r for r in rows if r["kind"] == "refusal")
    assert refusal["slot"] == truncated
    assert refusal["reason"] == "turn-cap", refusal
    assert refusal["stop_reason"] == "max_turns", refusal
    assert refusal["num_turns"] == 41, refusal
    assert refusal["tokens"] == 161_011
    assert "41-turn budget" in refusal["detail"], refusal["detail"]
    assert verdict["refused"] == [truncated], (
        f"{verdict['refused']}: the truncated slot is absent from `refused`, so the "
        "verdict claims three investigated theories")
    assert verdict["tokens"] == 3 * 161_011


def _comment_block_above(assignment: str) -> str:
    """The `#:` block sitting directly above a module-level assignment."""
    src = inspect.getsource(replay).splitlines()
    i = next(k for k, ln in enumerate(src) if ln.startswith(assignment))
    j = i
    while j > 0 and (src[j - 1].lstrip().startswith("#") or not src[j - 1].strip()):
        j -= 1
    return "\n".join(src[j:i])


def test_both_budget_settings_cite_both_measured_anchors():
    """Clause 5: a reader can re-derive each figure from the text that sets it.

    Both numbers arrived out of a live truncation nobody had recorded, so the docstring
    IS the artefact: `git grep '161011|161,011'` over eval and tests returned nothing
    before this round, which is why the 200,000/40 pair survived a whole pass of
    evidence. Each setting therefore has to carry BOTH anchors — the new one (smoke-b3:
    161,011 tokens at `num_turns: 41`) and the arm-A one it replaced (27 turns /
    179,186 tokens, `probe-maxturns-40.log`) — plus its own new value. The ceiling's
    text is a `#:` comment block rather than a docstring, so it is read from the
    module's own source; `investigate_live.__doc__` is read as prose. Checking the
    whole module text instead of these two blocks would pass on one lucky mention
    somewhere and leave the setting itself unexplained.
    """
    anchors = {"161,011": "smoke-b3's measured cost of ONE theory",
               "num_turns: 41": "the turn count smoke-b3 was truncated at",
               "27 turns": "the turn count that reached stop_reason: stop",
               "179,186": "the tokens that 27-turn answer cost"}

    ceiling_text = _comment_block_above("DEFAULT_TOKEN_CEILING = ")
    assert "600,000" in ceiling_text, "the ceiling's text never names the new figure"
    turn_text = investigate_live.__doc__ or ""
    assert "60 turns" in turn_text, "the turn budget's text never names the new figure"

    for name, text in (("DEFAULT_TOKEN_CEILING", ceiling_text),
                       ("investigate_live", turn_text)):
        missing = [f"{figure} ({what})" for figure, what in anchors.items()
                   if figure not in text]
        assert not missing, (
            f"{name}'s setting text does not cite: {'; '.join(missing)} — the figures "
            "are the only reason the value is what it is")


def test_the_generation_budget_cites_both_measured_arm_b_runs():
    """Clauses 1-4 of #3170: arm B's generation step is cited to its own measurements.

    `generate_theories_live`'s docstring used to assert the opposite of the record — that
    it was "the one live budget still without a measurement", that "no #2258 run ever
    reached this step", and that "every smoke run died in arm A" (written by `f027def5`,
    2026-10-12, before either probe ran). Both #2258 arm-B smoke runs contradict all
    three: `smoke-b2.jsonl` run-20261013T124552Z and `smoke-b3.jsonl` run-20261018T220131Z
    each carry a `kind: theories` row holding 3 disjoint non-empty theories with
    `generated_before_evidence: true`, written 49 s and 47 s after their own run-id start
    stamps — `run_id` is built from `datetime.now` at run start, so a row's `ts` minus its
    run's stamp IS the generation span, and the span the two rows support is 47–49 s, not
    the ~60 s the item carried as its upper bound. Like `investigate_live`'s, this
    docstring IS the artefact: `git grep` over eval and tests held neither run id before
    this round, so the node reads `generate_theories_live.__doc__` as prose rather than
    the module's whole text, which would pass on one lucky mention somewhere and leave the
    setting itself unexplained. It re-pins the two things the round promised not to
    change: the retired sentence may not come back, and the shipped 6-turn default stands
    because the measurement says 6 suffices — #3170 raises nothing.
    """
    doc = replay.generate_theories_live.__doc__ or ""
    anchors = {
        "run-20261013T124552Z": "the smoke-b2 run whose arm-B theories row is the witness",
        "run-20261018T220131Z": "the smoke-b3 run that repeats it",
        "3 disjoint": "how many theories each of those rows carries",
        "generated_before_evidence: true": "the flag proving the theories preceded evidence",
        "47–49 s": "the generation span measured across the two runs"}
    missing = [f"{figure} ({what})" for figure, what in anchors.items() if figure not in doc]
    assert not missing, (
        "generate_theories_live's docstring does not cite: "
        f"{'; '.join(missing)} — those two runs are the only reason 6 turns is a measured "
        "budget rather than a guess")

    assert "without a measurement" not in doc, (
        "the retired #3170 claim is back in generate_theories_live's docstring: both arm-B "
        "smoke runs reached the generation step, so the budget is not unmeasured")

    default = inspect.signature(replay.generate_theories_live).parameters["max_turns"].default
    assert default == 6, (
        f"generate_theories_live's max_turns moved to {default}: the measurement says the "
        "shipped 6 already suffices, so #3170 raises nothing")


def test_the_refusal_to_record_an_empty_cause_survives_the_raised_budgets(tmp_path):
    """Clause 3: more turns and more tokens bought no licence to record a guess.

    The raise is exactly what makes this worth re-pinning. An exhausted run still
    returns no structured answer, so the arm can still be handed a claim list with
    nothing readable in it — and the row it would then write is the one that fakes the
    headline: `agrees()` asks whether every marker of the expected cause occurs in the
    text, empty text answers no to all of them, and `format_report` subtracts that 0/N
    from arm B's measured rate before printing a HEADLINE. So the investigator below
    answers the way an exhausted one does — the live shape, `claims` present, a blank
    claim inside it — at the raised default ceiling and the measured spend, which
    proves the refusal outlives the numbers this round changed rather than only the
    numbers it was written against.
    """
    case = _study_corpus()[0]
    rows_path = tmp_path / "rows.jsonl"

    async def exhausted(case_arg, theory):
        return {"tool": "Task", "model": "primary", "tokens": 179_186,
                "claims": [{"claim": "   ",
                            "citations": [{"path": "eval/README.md", "line": 1}]}]}

    with pytest.raises(RuntimeError, match="no non-empty claim"):
        asyncio.run(run_case_a(case, investigate=exhausted,
                               writer=RowWriter(rows_path), run_id="c24b"))

    assert not (rows_path.exists() and rows_path.read_text().strip()), (
        "the refused cell still left rows on disk: a tool row without a cause is "
        "counted into the denominator as though the arm had been measured")


def test_report_grades_only_the_named_keys_and_counts_that_subset(tmp_path):
    """Clause 2: the same 98-row corpus can be reported over 21 keys, and N is 21.

    `run --only` already existed; `report` had no way to follow it, so the study's own
    sequence could only print `baseline N 21 != corpus N 98`.
    """
    cases = _study_corpus()
    only = _non_leaky_keys(cases)
    assert len(only) == NON_LEAKY == 21
    picked, bad = select_cases(cases, only)
    assert not bad and len(picked) == 21

    lines, _ = format_report(_study_rows(cases, only), picked, baseline=None,
                             corpus="fp", model="primary", token_ceiling=120_000,
                             citation_root=tmp_path, all_cases=cases)
    text = "\n".join(lines)
    assert "cases: 21" in text, text
    assert "cases: 98" not in text, text


def test_a_matched_baseline_over_the_graded_subset_prints_spend_for_all_three_arms(tmp_path):
    """Clause 3: N matched against the SUBSET evaluates, and every arm's spend is on one line.

    This is the line the study's whole cost comparison rests on: arm B's gain has to be
    read against what A and A-retry spent under the same ceiling, and a reader given only
    two of the three cannot check that A-retry — the compute-matched arm — was matched.
    """
    cases = _study_corpus()
    only = _non_leaky_keys(cases)
    picked, _ = select_cases(cases, only)
    baseline = {"corpus": "fp", "n": NON_LEAKY, "model": "primary",
                "token_ceiling": 120_000, "token_spend": 861_000, "top1_hits": 21}

    lines, evaluated = format_report(_study_rows(cases, only), picked,
                                     baseline=baseline, corpus="fp", model="primary",
                                     token_ceiling=120_000, citation_root=tmp_path,
                                     all_cases=cases)
    text = "\n".join(lines)
    assert evaluated is True, text
    assert "cannot evaluate" not in text, text
    assert re.search(r"token spend\s+A: 840000\s+B: 1890000", text), text
    assert re.search(r"token spend\s+A-retry: 861000", text), text
    assert "ceiling: 120000" in text, (
        "the arms are matched by token budget and by nothing else; an artifact that "
        "prints a spend without the ceiling prints an unverifiable comparison")


def test_the_artifact_names_what_was_left_out_and_which_family_it_was(tmp_path):
    """Clause 4: 77 excluded cases and the family they belong to, in the artifact.

    `cases: 21` on its own reads as though the instrument had 21 cases in the world. The
    owed study's null — if it is one — is only worth what the exclusions say it is worth.
    """
    cases = _study_corpus()
    only = _non_leaky_keys(cases)
    picked, _ = select_cases(cases, only)
    lines, _ = format_report(_study_rows(cases, only), picked, baseline=None,
                             corpus="fp", model="primary", token_ceiling=120_000,
                             citation_root=tmp_path, all_cases=cases)
    text = "\n".join(lines)
    assert "excluded from the graded set: 77 of 98 corpus cases" in text, text
    assert "tests-blamed-then-refuted=77" in text, text

    full, _ = format_report(_study_rows(cases, only), cases, baseline=None,
                            corpus="fp", model="primary", token_ceiling=120_000,
                            citation_root=tmp_path)
    assert "excluded from the graded set: none" in "\n".join(full), (
        "an uncut report must say it graded everything, or the exclusion line reads as "
        "a fixed caption rather than a statement about this run")


def test_a_corpus_sized_baseline_under_a_21_case_report_still_refuses(tmp_path):
    """Clause 5: narrowing the study cannot quietly satisfy #627's matched-baseline rule.

    The failure this pins is the plausible one: `baseline` was run over the whole corpus,
    or was cached from an earlier whole-corpus pass, and 21 cases are then reported. The
    refusal has to name both Ns — the one that was measured and the one being graded.
    """
    cases = _study_corpus()
    only = _non_leaky_keys(cases)
    picked, _ = select_cases(cases, only)
    whole = {"corpus": "fp", "n": len(cases), "model": "primary",
             "token_ceiling": 120_000, "token_spend": 4_000_000, "top1_hits": 90}

    lines, evaluated = format_report(_study_rows(cases, only), picked, baseline=whole,
                                     corpus="fp", model="primary",
                                     token_ceiling=120_000, citation_root=tmp_path,
                                     all_cases=cases)
    text = "\n".join(lines)
    assert evaluated is False, text
    assert "HEADLINE: cannot evaluate" in text, text
    assert "baseline N 98 != graded N 21" in text, text
    assert "arm B wins" not in text, text


def test_select_cases_refuses_a_key_that_names_no_case_in_the_corpus():
    """A subset list that silently matches less than it names is a different study.

    `run --only` used to skip unmatched keys per case, so one mistyped key out of 21
    measured 20 and nothing said so; a wholly mistyped list measured all 98, leaky
    included. Both commands now stop instead.
    """
    cases = _study_corpus()
    picked, bad = select_cases(cases, ["flaky-SM_190701_000000"])
    assert picked == [] and bad == ["flaky-SM_190701_000000"]
    picked, bad = select_cases(cases, [_non_leaky_keys(cases)[0], "typo-key"])
    assert len(picked) == 1 and bad == ["typo-key"]
    assert excluded_note(cases, picked).startswith("excluded from the graded set: 97 of 98")


def test_report_grading_the_real_sequence_end_to_end_needs_no_engine(tmp_path):
    """The item's own sequence across the CLI boundary: `report --only` over 21 keys.

    Everything above calls `format_report` in-process; the study runs the script, and the
    wiring between `main`'s new `--only` and the graded subset is its own place to break.
    Exit 0 with a subset count and an evaluating headline is the shape the owed run
    produces; `cannot evaluate` here would have been clause 1 of this item's own check.
    """
    cases = _study_corpus()
    only = _non_leaky_keys(cases)
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(c) + "\n" for c in cases),
                          encoding="utf-8")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n"
                                 for r in _study_rows(cases, only)), encoding="utf-8")
    baseline_path = tmp_path / "retry_baseline.json"
    # The default ceiling, not a literal: this CLI call passes no `--ceiling`, so the
    # baseline has to be written at whatever `main` will assume. Hard-coding it here
    # held the node to 120,000 and made #2624's raise look like a broken report — the
    # ceiling this node is about is the wiring, not the number.
    write_baseline(baseline_path, cases_path=cases_path, n=NON_LEAKY, model="primary",
                   token_ceiling=DEFAULT_TOKEN_CEILING, token_spend=861_000,
                   top1_hits=NON_LEAKY)

    proc = subprocess.run(
        [sys.executable, str(REPO / "eval" / "diagnosis_fanout_replay.py"), "report",
         "--cases", str(cases_path), "--rows", str(rows_path),
         "--baseline", str(baseline_path), "--only", *only],
        capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "cases: 21" in out, out
    assert "excluded from the graded set: 77 of 98 corpus cases" in out, out
    assert "HEADLINE: cannot evaluate" not in out, out
    assert "HEADLINE: arm B" in out, out

    # And the same file with the whole corpus graded refuses the 21-case baseline: the
    # subset selector is what makes the sequence legal, not a looser check.
    proc2 = subprocess.run(
        [sys.executable, str(REPO / "eval" / "diagnosis_fanout_replay.py"), "report",
         "--cases", str(cases_path), "--rows", str(rows_path),
         "--baseline", str(baseline_path)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert proc2.returncode == 3, proc2.stdout + proc2.stderr
    assert "baseline N 21 != graded N 98" in proc2.stdout, proc2.stdout




# ── #2635: a theory's answer is not a case's, and a case's is not the pass's ───
#
# The route to #2623 clause 4 was blocked on this. `validate_investigation` raises
# `RecordError` on a theory that handed back prose, `run_case_b` called it unguarded
# inside the theory loop, and `cmd_run` guarded nothing — so on 2026-10-12 theory 2
# of case 1 of the arm-B smoke run raised, the process died at 1,078 seconds of
# engine time, and the case wrote no `verdict` row. Absence is worse than a miss for
# this study: `latest_verdicts` keys on `kind: verdict` rows and `score` takes each
# arm's N from those alone, so an aborted case is not scored as wrong — it is not in
# the denominator at all, and `report` cannot even say the arm was run.
#
# The nodes below keep the raise's purpose (nothing unevidenced is ranked) and take
# away its reach. Since #2807 the mechanism IS claimed — the section after this one
# carries `stop_reason`/`num_turns` onto the refusal row and gives a `max_turns` stop
# a reason of its own — so these nodes hold the older shape on purpose: their fake
# investigators name no stop at all, which is exactly the case that still earns a
# `no-claims` row, and the rows they assert record only that a theory was dispatched,
# cost tokens, and produced no citable claim.

def _one_case(key: str) -> dict:
    """One fixture case under a key of the caller's choosing.

    The keys the crashed pass uses are `flaky-SM_*`, and using the real one makes a
    row's slot labels the real rotation (`slot_labels` hashes the key) rather than a
    fixture's.
    """
    return dict(_cases(1)[0], key=key)


def _arm_b(tmp_path, *, key="flaky-SM_20261001_034449", rows_name="rows.jsonl",
           refuse=(), spend=None, rank_fn=None, ceiling=99_000, stop=None):
    """Run arm B once over one case, with the theories in `refuse` answering in prose.

    Returns (case, slots, dispatched, rows, verdict). `dispatched` is the slot order
    the investigator was called in, which is how the tests below distinguish "the
    fan-out continued" from "the fan-out ended and something else wrote the rows".
    A refusal costs its own tokens by default (1,000 against the 10,000 an evidenced
    theory costs) so a row's spend can be attributed to the dispatch that earned it.

    `stop` (#2807) gives a refusing slot its own harness stop as
    `(stop_reason, num_turns)`, which is the pair `_engine_turn` now reads off the
    `result` event; a slot absent from it reports no mechanism at all, which is the
    shape every node written before #2807 drives.
    """
    from eval.diagnosis_fanout_replay import RowWriter, run_case_b

    case = _one_case(key)
    slots = slot_labels(key)
    theories = [f"theory {i}" for i in range(THEORIES_PER_CASE)]
    cost = {s: (spend or {}).get(s, 10_000) for s in slots}
    dispatched: list[str] = []

    async def generate(c):
        return list(theories)

    async def investigate(c, theory):
        slot = slots[theories.index(theory)]
        dispatched.append(slot)
        if slot in refuse:
            row = {"tool": "Task", "claims": [], "tokens": cost[slot]}
            if stop and slot in stop:
                row["stop_reason"], row["num_turns"] = stop[slot]
            return row
        return {"tool": "Task",
                "claims": [{"claim": f"{slot}: the config drifted",
                            "citations": [{"path": "app/config.py", "line": 12}]}],
                "tokens": cost[slot]}

    rows_path = tmp_path / rows_name
    verdict = asyncio.run(run_case_b(case, generate=generate, investigate=investigate,
                                     rank_fn=rank_fn or _rank_stub([0, 1]),
                                     writer=RowWriter(rows_path), run_id="t2635",
                                     token_ceiling=ceiling))
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    return case, slots, dispatched, rows, verdict


def test_a_theory_that_hands_back_no_claims_is_recorded_and_the_fan_out_continues(
        tmp_path):
    """Clause 1: the refused theory becomes a row carrying its `slot`, its `tokens`
    and a refusal reason, and the theories after it are still investigated.

    This is the 2026-10-12 smoke scenario: `2258/smoke-b.jsonl` holds a `theories`
    row and ONE `tool` row (149,520 tokens) and nothing else, because
    `RecordError: theory 'c2' handed back no claims` propagated out at theory 2. The
    third theory never ran, so the record could not distinguish a weak arm B from an
    arm B that had not been measured.
    """
    key = "flaky-SM_20261001_034449"
    slots = slot_labels(key)
    case, slots, dispatched, rows, _ = _arm_b(tmp_path, key=key, refuse=(slots[1],),
                                              spend={slots[1]: 1_000})
    assert dispatched == slots, (
        f"the fan-out dispatched {dispatched} of {slots}: a refused theory must not "
        "cancel the dispatch after it")

    refusals = [r for r in rows if r["kind"] == "refusal"]
    assert len(refusals) == 1, f"expected one refusal row, got {len(refusals)}"
    row = refusals[0]
    assert row["slot"] == slots[1], f"the refusal names slot {row['slot']!r}"
    assert row["arm"] == "B" and row["case_key"] == key, (
        "a refusal row without the arm and the case key cannot be attributed to a case, "
        "which is what makes it reportable at all")
    assert row["tokens"] == 1_000, (
        f"the refusal carries {row['tokens']} tokens; the refused dispatch cost 1,000 "
        "and dropping that spend understates what the arm paid to learn nothing")
    assert row["reason"] == replay.REFUSAL_NO_CLAIMS == "no-claims", (
        f"reason reads {row['reason']!r}")
    assert "handed back no claims" in row["detail"], (
        "the row lost the validator's own complaint, so a reader cannot tell which "
        "refusal this was from any other")
    assert [r["slot"] for r in rows if r["kind"] == "tool"] == [slots[0], slots[2]], (
        "the two evidenced theories did not both get a per-theory row")
    assert [r for r in rows if r["kind"] == "tool" and not r["claims"]] == [], (
        "an empty per-theory row was written beside the refusal — two records of the "
        "same nothing, one of them styled as evidence")
    assert sum(int(r.get("tokens") or 0) for r in rows
               if r["kind"] == "tool") + sum(int(r.get("tokens") or 0) for r in refusals) \
        == 21_000, "the case's recorded spend must include the refused dispatch"


def test_a_case_whose_theory_refused_still_reaches_its_verdict_row(tmp_path):
    """Clause 2: the case ends with a `kind: verdict` row, because being absent from
    the denominator is not the same as being measured and losing.

    `latest_verdicts` keys only on `kind: verdict` rows and `score` derives each
    arm's N from those: the 2026-10-12 smoke case left no verdict row, so `report`
    could not have reported arm B's top-1 at all. The verdict row is the difference
    between a study with N=1 and one with N=0.
    """
    key = "flaky-SM_20261001_034449"
    slots = slot_labels(key)
    case, _, _, rows, verdict = _arm_b(tmp_path, key=key, refuse=(slots[1],),
                                       spend={slots[1]: 1_000})
    assert [r["kind"] for r in rows] == ["theories", "tool", "refusal", "tool",
                                         "rank", "verdict"], (
        f"the run's row sequence was {[r['kind'] for r in rows]} — the refusal belongs "
        "inside the fan-out, before the rank row, or nothing was ranked")
    assert rows[-1]["kind"] == "verdict" and rows[-1]["case_key"] == key
    assert verdict["kind"] == "verdict" and verdict["tokens"] == 21_000, (
        f"the verdict's spend reads {verdict['tokens']}; the refused theory's 1,000 is "
        "part of what the case cost")
    assert verdict["refused"] == [slots[1]], (
        "the verdict row must say which slot it did not choose among, or a two-theory "
        "case reads to a later reader as a three-theory one")
    scored = replay.score(rows, [case])
    assert scored["arms"]["B"]["n"] == 1, (
        f"arm B's N is {scored['arms']['B']['n']} with a verdict row on disk; a refusal "
        "may not move the denominator")
    assert scored["arms"]["A"]["n"] == 0, (
        "arm B's refusal leaked into arm A's N — the two arms are scored per arm")


def test_a_refused_theory_is_never_handed_to_the_ranking_engine(tmp_path):
    """Clause 3: the candidate list `rank_fn` receives has no entry for a refused
    slot, so recording the refusal keeps the raise's entire purpose.

    `rank_theories` already tolerated a partial evidence map (`by_slot` defaulted a
    missing slot to `[]`), so the pre-fix behaviour was worse than a raise at the
    ranking would have been: the refused slot contributed a candidate whose text was
    its own bare label — `rank_candidate_text` with no claims is exactly `"[c2]"` —
    an unevidenced contender competing on nothing, which is the thing
    `validate_investigation` exists to keep out.
    """
    key = "flaky-SM_20261001_034449"
    slots = slot_labels(key)
    seen: list[list[str]] = []

    def spy(query, candidates):
        seen.append(list(candidates))
        return [{"index": i, "score": 0.9 - 0.1 * i} for i in range(len(candidates))]

    case, _, _, rows, verdict = _arm_b(tmp_path, key=key, refuse=(slots[1],),
                                       rank_fn=spy)
    assert seen, "the ranking was never called, so the candidate list proves nothing"
    candidates = seen[0]
    assert len(candidates) == THEORIES_PER_CASE - 1, (
        f"the engine was handed {len(candidates)} candidates for {THEORIES_PER_CASE} "
        "theories, one of which refused")
    assert candidates.count(f"[{slots[1]}]") == 0 and not any(
        c == f"[{slots[1]}]" for c in candidates), (
        f"the refused slot {slots[1]} is still a candidate — that text is a bare "
        "label with no claim under it")
    assert not [c for c in candidates if c.strip() == c.split("]")[0] + "]"], (
        f"a candidate carried no claims: {[c[:20] for c in candidates]}")
    assert [c.split("]")[0] + "]" for c in candidates] == sorted(
        [c.split("]")[0] + "]" for c in candidates]), (
        "the survivors must keep the sorted-by-slot order the per-case rotation "
        "argument depends on; dropping a middle slot must not reshuffle the others")
    assert verdict["order"] and slots[1] not in verdict["order"], (
        f"the persisted order still names the refused slot: {verdict['order']}")
    rank_row = next(r for r in rows if r["kind"] == "rank")
    assert rank_row.get("refused") == [slots[1]], (
        "the rank row does not record that a slot was excluded, so the ordering looks "
        "like a three-way choice")


def test_a_refusal_is_not_reported_as_a_token_ceiling(tmp_path):
    """Clause 4: `no-claims` and `token-ceiling` are different facts about different
    causes, and one row kind each — in both directions.

    A `kind: ceiling` row with `reason: "token-ceiling"` means the fan-out ran out of
    money, which `report` prints as a spend story and #2258 pinned as a fan-out cut
    short by compute. A theory that ran, cost its tokens and answered in prose is
    Pienaar's fourth class. Fold the two together and a study whose model simply
    declined to cite gets read as one whose budgets are still short — the wrong
    follow-up to file, and #2624's budget work would be pointed at the wrong problem.
    """
    from eval.diagnosis_fanout_replay import REFUSAL_NO_CLAIMS

    assert REFUSAL_NO_CLAIMS != "token-ceiling", (
        "the two reasons were collapsed into one word, so the record can no longer "
        "separate a declined theory from an out-of-budget fan-out")
    key = "flaky-SM_20261001_034449"
    slots = slot_labels(key)
    _, _, _, refused_rows, _ = _arm_b(tmp_path, key=key, refuse=(slots[1],),
                                      rows_name="refused.jsonl")
    assert [r["reason"] for r in refused_rows if r["kind"] == "refusal"] \
        == [REFUSAL_NO_CLAIMS]
    assert not [r for r in refused_rows if r["kind"] == "ceiling"], (
        "a fan-out shortened by a refusal wrote a ceiling row: the case was not over "
        "budget, it was refused")
    assert not [r for r in refused_rows if r["kind"] == "refusal"
                and r.get("reason") == "token-ceiling"]

    # The other direction, on the same fixture: a fan-out that runs out of money
    # records the ceiling and no refusal, and nothing is marked refused.
    theories = [f"theory {i}" for i in range(THEORIES_PER_CASE)]
    claims = {"a": [{"claim": "x", "citations": [{"path": "p.py"}]}]}
    generate, investigate = _fake_arm_b(theories, {t: claims["a"] for t in theories},
                                        tokens=10_000)
    from eval.diagnosis_fanout_replay import RowWriter, run_case_b
    tight = tmp_path / "tight.jsonl"
    # 15,000 against 10,000 per dispatch: one theory runs, the check before the
    # second sees the spend already over, and the fan-out stops for money.
    asyncio.run(run_case_b(_one_case(key), generate=generate, investigate=investigate,
                           rank_fn=_rank_stub([0]), writer=RowWriter(tight),
                           run_id="t2635e", token_ceiling=15_000))
    tight_rows = [json.loads(l) for l in tight.read_text().splitlines()]
    assert [r["reason"] for r in tight_rows if r["kind"] == "ceiling"] == ["token-ceiling"]
    assert not [r for r in tight_rows if r["kind"] == "refusal"], (
        "an over-ceiling fan-out recorded a refusal beside its ceiling row")


# ── #2807: a turn-cap stop is never read as an honest empty ───────────────────
#
# The section above records that a theory produced no citable claim and says
# nothing about WHY, because the pair that would say it was dropped twice on its
# way to the row: `_engine_turn` read `structured` and `usage` off the harness
# `result` event and returned `{structured, tokens, model}`, then
# `investigate_live` rebuilt that as `{tool, claims, tokens, model}`. The
# 2026-10-13 arm-B pass is the witness — all 16 `kind: refusal` rows in
# `~/lloyd-data/eval/2258/rows.jsonl` carry key set [arm, case_key, detail, kind,
# reason, run_id, seq, slot, tokens, ts] and none carries the mechanism — which is
# why naming the mechanism for `flaky-SM_20261001_034449` needed a bespoke probe of
# ~1,000 engine-seconds instead of a read of data that already existed.
#
# The nodes below carry the pair from the result event to the row and give a
# `max_turns` stop a reason that is neither `no-claims` nor `token-ceiling`, so a
# budget that is still short and a model that declined to cite stop arriving at
# the study as one row.


def test_the_engine_turn_returns_the_stop_and_turn_count_its_result_event_carried(
        monkeypatch):
    """Clause 1, at the harness seam: `_engine_turn` keeps both fields of the event.

    This is the one boundary no fake investigator can cross, so it is tested
    against the producer's own factory: the stream below yields
    `app.harness.events.result(...)`, the dict the real loop emits, so a renamed or
    dropped field there is a red node here rather than a silently-absent column
    three hops away in a row. Two calls with two different pairs are run because
    the clause is that the values are READ off the event — one fixture and one
    literal cannot tell those apart.

    What the pair is for: a run that ends `max_turns` gets no `final_schema`
    restatement (`FINALIZABLE_STOP_REASONS` is `{stop, end_turn}`,
    `app/harness/finalizer.py:78`), so `structured` comes back empty and the claim
    list under it is empty too. `investigate_live`'s budget was raised to 40 on
    exactly that measurement, and the study still could not tell a truncation from
    an honest empty, because `tokens` and `model` were the only two of the four
    result-event fields read back out.
    """
    from app.harness import events as harness_events
    from app.harness import loop as harness_loop
    import app.prompt_builder as prompt_builder

    scripted: list[dict] = []

    def fake_run_query(messages, options):
        assert str(options.session_id).startswith("bench_"), (
            "the bench prefix is what sandboxes an investigation, and `_engine_turn` "
            "refuses anything else before it ever reaches this stream")

        async def stream():
            yield {"type": "text", "text": "reading the ledger"}
            yield scripted.pop(0)
        return stream()

    monkeypatch.setattr(harness_loop, "run_query", fake_run_query)
    monkeypatch.setattr(prompt_builder, "build_system_prompt",
                        lambda **kwargs: "bench prompt")

    scripted.append(harness_events.result(
        stop_reason="max_turns", num_turns=21, wrapped_up=True,
        usage={"input_tokens": 179_000, "output_tokens": 186},
        structured_error="no verdict to restate"))
    truncated = asyncio.run(replay._engine_turn(
        "investigate this one cause", schema=replay.CLAIMS_SCHEMA,
        session_id="bench_2258_inv_truncated", max_turns=40))
    assert truncated["stop_reason"] == "max_turns", truncated
    assert truncated["num_turns"] == 21, truncated
    assert truncated["tokens"] == 179_186, (
        f"tokens read {truncated['tokens']}; the usage arithmetic is not what this "
        "node is for and must not move with the two new keys")

    scripted.append(harness_events.result(
        stop_reason="stop", num_turns=27, structured={"claims": []},
        usage={"input_tokens": 10, "output_tokens": 5}))
    stopped = asyncio.run(replay._engine_turn(
        "investigate this one cause", schema=replay.CLAIMS_SCHEMA,
        session_id="bench_2258_inv_stopped", max_turns=40))
    assert (stopped["stop_reason"], stopped["num_turns"]) == ("stop", 27), stopped


def test_the_live_investigator_carries_the_stop_and_turn_count_through(monkeypatch):
    """Clause 1's second hop: `investigate_live` does not drop what it was given.

    Patching only `_engine_turn` would fix nothing observable. `investigate_live`
    builds its own dict — `tool`, `claims`, `tokens`, `model` — and that rebuild
    was the second place the mechanism was thrown away, so the pair has to be in
    the return the arm actually reads. The fake below is the shape a capped run
    really has: no structured answer, so an empty claim list, plus the pair naming
    why (the numbers are the ones `investigate_live`'s docstring records for the
    20-turn cap: 21 turns, 179,186 tokens).
    """
    case = _one_case("flaky-SM_20261001_034449")

    async def fake_turn(prompt, *, schema, session_id, max_turns):
        return {"structured": {}, "tokens": 179_186, "model": "primary",
                "stop_reason": "max_turns", "num_turns": 40}

    monkeypatch.setattr(replay, "_engine_turn", fake_turn)
    out = asyncio.run(investigate_live(case, "the config drifted"))
    assert out["claims"] == [], out
    assert (out["stop_reason"], out["num_turns"]) == ("max_turns", 40), (
        f"the investigator returned keys {sorted(out)} — the pair has to reach the "
        "arm, or the row the arm writes cannot name its mechanism")


def test_every_refusal_row_carries_the_pair_its_own_investigation_returned(tmp_path):
    """Clause 2: both fields sit beside `tokens` on every refusal row, and they are
    THAT dispatch's values.

    Two refusing slots with two different stops, because "the values its own
    investigation returned" is the clause and a shared or hard-coded pair would
    pass a one-refusal fixture. A second run with a fake that names no stop at all
    pins the remaining shape: the keys are still present and null, which says "this
    investigator reported no mechanism" instead of inventing a `stop` at zero turns
    — the invention this item exists to end.
    """
    key = "flaky-SM_20261001_034449"
    slots = slot_labels(key)
    stops = {slots[1]: ("max_turns", 40), slots[2]: ("stop", 9)}
    spends = {slots[1]: 1_000, slots[2]: 2_000}
    _, _, _, rows, _ = _arm_b(tmp_path, key=key, refuse=(slots[1], slots[2]),
                              spend=spends, stop=stops, rows_name="paired.jsonl")
    refusals = [r for r in rows if r["kind"] == "refusal"]
    assert len(refusals) == 2, f"expected two refusal rows, got {len(refusals)}"
    for row in refusals:
        assert {"stop_reason", "num_turns"} <= set(row), (
            f"the row carries {sorted(row)}; `tokens` alone cannot say which of the "
            "two failures happened")
        want = stops[row["slot"]]
        assert (row["stop_reason"], row["num_turns"]) == want, (
            f"slot {row['slot']} recorded {row['stop_reason']!r}/"
            f"{row['num_turns']!r}; its own investigation returned "
            f"{want[0]!r}/{want[1]}")
        assert row["tokens"] == spends[row["slot"]], row

    _, _, _, unreported_rows, _ = _arm_b(tmp_path, key=key, refuse=(slots[1],),
                                        rows_name="unreported.jsonl")
    unreported = next(r for r in unreported_rows if r["kind"] == "refusal")
    assert unreported["stop_reason"] is None and unreported["num_turns"] is None, (
        f"an investigator that named no mechanism left "
        f"{unreported['stop_reason']!r}/{unreported['num_turns']!r} on the row")
    assert unreported["reason"] == replay.REFUSAL_NO_CLAIMS, unreported


def test_a_turn_capped_theory_gets_its_own_reason_and_not_the_no_claims_prose(
        tmp_path):
    """Clauses 3 and 4's wording half: three causes, three words, two details.

    `turn-cap` is recorded for a `max_turns` stop and `no-claims` only for the
    empty claim list that arrived on a run which DID stop. The detail is asserted
    against the other wording rather than for a keyword: `no_claims_message` says
    the model handed back prose, and putting that sentence on a truncated dispatch
    is how a reader files a budget follow-up for a model that declined to cite, or
    the other way round — the same misreading #2635 clause 4 was written against.
    """
    key = "flaky-SM_20261001_034449"
    slots = slot_labels(key)
    assert replay.REFUSAL_TURN_CAP == "turn-cap", replay.REFUSAL_TURN_CAP
    assert len({replay.REFUSAL_TURN_CAP, replay.REFUSAL_NO_CLAIMS,
                "token-ceiling"}) == 3, (
        "the three refusal causes are no longer three words, so a row can say "
        "`no-claims` and mean any of them")

    _, _, _, rows, _ = _arm_b(tmp_path, key=key, refuse=(slots[1],),
                              stop={slots[1]: ("max_turns", 40)},
                              rows_name="capped.jsonl")
    capped = next(r for r in rows if r["kind"] == "refusal")
    assert capped["reason"] == replay.REFUSAL_TURN_CAP == "turn-cap", capped["reason"]
    assert capped["stop_reason"] == "max_turns" and capped["num_turns"] == 40, capped
    assert "ran out of turns" in capped["detail"], capped["detail"]
    assert "40-turn budget" in capped["detail"], capped["detail"]
    assert "handed back no claims" not in capped["detail"], (
        f"the detail reads {capped['detail']!r} — that is the wording which means "
        "the model answered and declined to cite")

    _, _, _, honest_rows, _ = _arm_b(tmp_path, key=key, refuse=(slots[1],),
                                     stop={slots[1]: ("stop", 12)},
                                     rows_name="honest.jsonl")
    honest = next(r for r in honest_rows if r["kind"] == "refusal")
    assert honest["reason"] == replay.REFUSAL_NO_CLAIMS, honest
    assert honest["stop_reason"] == "stop" and honest["num_turns"] == 12, honest
    assert "handed back no claims" in honest["detail"], honest["detail"]
    assert not [r for r in honest_rows
                if r.get("reason") == replay.REFUSAL_TURN_CAP], (
        "a run that stopped and still produced no claims was booked as a "
        "truncation, which would send a budget follow-up at a model that declined")


def test_a_turn_capped_refusal_and_a_ceiling_row_stay_two_different_rows(tmp_path):
    """Clause 4's ceiling half: one case, both causes, two rows, two reasons.

    The ceiling is 11,000 against a 10,000 evidenced theory plus a 1,000 refusal,
    so theory 2 truncates on its OWN turn budget and theory 3 is never dispatched
    because the fan-out's money ran out. The `kind: ceiling` row keeps
    `reason: token-ceiling` — it is the fan-out's compute, not theory 2's turn
    cap — and the turn-capped theory writes no ceiling row for itself.
    """
    key = "flaky-SM_20261001_034449"
    slots = slot_labels(key)
    _, _, dispatched, rows, _ = _arm_b(tmp_path, key=key, refuse=(slots[1],),
                                       spend={slots[1]: 1_000},
                                       stop={slots[1]: ("max_turns", 40)},
                                       ceiling=11_000, rows_name="both.jsonl")
    assert dispatched == slots[:2], (
        f"the fan-out dispatched {dispatched}: 11,000 spent before the third theory "
        "means the third dispatch never ran")
    assert [r["reason"] for r in rows if r["kind"] == "refusal"] \
        == [replay.REFUSAL_TURN_CAP], rows
    assert [r["reason"] for r in rows if r["kind"] == "ceiling"] == ["token-ceiling"], (
        "the ceiling row's reason moved: it records the fan-out running out of "
        f"money, not theory {slots[1]}'s turn budget")
    assert not [r for r in rows if r["kind"] == "refusal"
                and r.get("reason") == "token-ceiling"], (
        "a truncated theory was booked as an over-ceiling fan-out, which is the "
        "wrong follow-up to file")


# ── clause 5: the pass outlives one case's arm ────────────────────────────────

THEORY_SET = ["the flake", "the config", "the slot collision"]


def _drive_pass(tmp_path, monkeypatch, capsys, *, arms, keys, fail_theory_set=False,
                fail_case=None, fail_arm=None, fail_stop=None):
    """Run the real `cmd_run` over several cases with only its ENGINE calls replaced.

    The arm functions themselves are NOT stubbed: `run_case_a`, `run_case_b`,
    `validate_theories`, `validate_investigation` and `RowWriter` all run, so what
    the clause asserts — that the other cases' ROWS still get written — is the real
    writer's output and not a fake's return value. Only the three live callables are
    patched (`cmd_run` looks them up as module globals at call time).

    `fail_arm`/`fail_case` choose which arm raises, and it raises through the real
    validator: `fail_arm="B"` makes arm B generate a duplicate theory, so
    `validate_theories` raises `RecordError`; `fail_arm="A"` makes arm A's single
    trace return a claim with no text, which is the empty-cause `RuntimeError` at the
    arm's own guard. `fail_stop=("max_turns", 40)` is that same arm A failure in the
    shape the live investigator hands back (#2807): an EMPTY claim list carrying the
    `stop_reason`/`num_turns` pair, which is what the raise's message is built from.
    """
    import argparse

    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(_one_case(k)) + "\n" for k in keys),
                          encoding="utf-8")
    calls: list[tuple[str, str]] = []
    good = [{"claim": "the config drifted", "citations": [{"path": "p.py", "line": 1}]}]

    async def generate(case):
        calls.append(("generate", case["key"]))
        if fail_arm == ARM_B and fail_case == case["key"] and fail_theory_set:
            return [THEORY_SET[0], THEORY_SET[0], THEORY_SET[2]]   # not disjoint
        return list(THEORY_SET)

    async def investigate(case, theory):
        # Which arm is calling is readable from the argument: `run_case_b` passes the
        # generated theory (:786) and `run_case_a` passes the case's OWN claimed cause
        # (:854), so anything outside the generated set is arm A's single dispatch.
        # Keying the arm-A failure on the case alone would poison arm B's rows for the
        # same case, which is a different bug than the one this node is about.
        arm_a = theory == (case.get("claimed_cause") or "")
        if not arm_a and theory not in THEORY_SET:
            raise AssertionError(f"arm B was handed an unexpected theory {theory!r}")
        slot = "single" if arm_a else slot_labels(case["key"])[THEORY_SET.index(theory)]
        calls.append((slot, case["key"]))
        if fail_arm == ARM_A and fail_case == case["key"] and arm_a:
            if fail_stop is not None:
                # The live truncated shape (#2807): no claims AT ALL, plus the pair
                # `investigate_live` re-emits. The arm's guard still refuses.
                return {"tool": "Task", "tokens": 100, "claims": [],
                        "stop_reason": fail_stop[0], "num_turns": fail_stop[1]}
            # Non-empty list, citations present, but no claim TEXT: the validator
            # passes it and the arm's own guard refuses to record an empty cause.
            return {"tool": "Task", "tokens": 100,
                    "claims": [{"claim": "   ", "citations": [{"path": "p.py",
                                                               "line": 1}]}]}
        return {"tool": "Task", "tokens": 100, "claims": good}

    def rank(query, candidates):
        return [{"index": i, "score": 0.9 - 0.1 * i} for i in range(len(candidates))]

    monkeypatch.setattr(replay, "generate_theories_live", generate)
    monkeypatch.setattr(replay, "investigate_live", investigate)
    monkeypatch.setattr(replay, "rank_live", rank)
    rows_path = tmp_path / "pass.jsonl"
    # `--only` is `nargs="*"` at the parser (:1483): a list of keys, not the comma
    # string `launch-pass.sh` builds with `$(cat graded-keys.txt)`.
    code = asyncio.run(replay.cmd_run(argparse.Namespace(
        cases=str(cases_path), arms=arms, ceiling=99_000, only=list(keys),
        rows=str(rows_path))))
    rows = ([json.loads(l) for l in rows_path.read_text().splitlines()]
            if rows_path.exists() else [])
    return code, calls, rows, capsys.readouterr().out


def test_one_cases_failure_does_not_cancel_the_remaining_cases(tmp_path, monkeypatch,
                                                              capsys):
    """Clause 5: `cmd_run` guards each arm of each case, names the failure on stdout,
    and the remaining cases still run AND still write their rows.

    The 2026-10-12 crash is the shape: `2258/smoke-b.log` ends in `RecordError:
    theory 'c2' handed back no claims` on case 1 with `SMOKE_B_EXIT=1`, the remaining
    graded keys never ran, and `2258/rows.jsonl` still does not exist. The guard is
    per ARM rather than per case so one arm's failure cannot skip that same case's
    other arm, and the exit code stays non-zero because `launch-pass.sh` records
    `PASS_EXIT=$?` and a pass that lost an arm must not read as a clean one to
    whoever opens the log at six in the morning.

    The raise here is arm B's own `RecordError` from `validate_theories` — a theory
    set that is not disjoint — which is what still raises after #2635 moved the
    zero-claim case onto a refusal row. The guard does not care which of the arm's
    refusals it is containing.
    """
    keys = ("flaky-SM_20261001_034449", "flaky-SM_SECOND", "landkill-SM_THIRD")
    code, calls, rows, out = _drive_pass(
        tmp_path, monkeypatch, capsys, arms="A,A-retry,B", keys=keys,
        fail_arm=ARM_B, fail_case=keys[0], fail_theory_set=True)

    for arm in (ARM_A, ARM_RETRY, ARM_B):
        for key in keys[1:]:
            verdicts = [r for r in rows if r["kind"] == "verdict"
                        and r["arm"] == arm and r["case_key"] == key]
            assert len(verdicts) == 1, (
                f"{arm} wrote {len(verdicts)} verdict rows for {key}; after {keys[0]} "
                f"raised, the pass reached {len({k for _, k in calls})} of {len(keys)} "
                "cases and the rest must be recorded, not skipped")
    assert not [r for r in rows if r["case_key"] == keys[0] and r["arm"] == ARM_B
                and r["kind"] == "verdict"], (
        "the failing case's arm B wrote a verdict row it did not earn")
    assert [r for r in rows if r["case_key"] == keys[0] and r["arm"] == ARM_A
            and r["kind"] == "verdict"], (
        "the failing case's own arm A rows were rolled back; a case that lost one arm "
        "still has the others recorded")
    assert f"FAILED {keys[0]} {ARM_B}" in out, (
        f"stdout never named the failed case and its arm: {out!r}")
    assert "RecordError" in out, "the failure's type was swallowed"
    assert code == 1, f"exit {code}; a pass that lost an arm exits non-zero"
    assert f"1 arm-run(s) failed: {keys[0]} {ARM_B}" in out, (
        "the closing line does not count the failures, so a 21-key pass can lose three "
        "arms and still print one summary")


def test_arm_as_no_cause_failure_also_lets_the_remaining_cases_run(tmp_path,
                                                                  monkeypatch,
                                                                  capsys):
    """Clause 5's other half, which the item's own premise would have skipped.

    "Arm A cannot hit this" is true only of the `RecordError`: arm A raises
    `RuntimeError` itself when its single trace returned no non-empty claim, and
    `cmd_run` guarded that either — so arm A's failure cancelled that same case's arm
    B, which is the half of the A-versus-B comparison #2623 clause 4 exists to
    measure.
    """
    keys = ("flaky-SM_A_FAIL", "flaky-SM_B_SURVIVES")
    code, calls, rows, out = _drive_pass(
        tmp_path, monkeypatch, capsys, arms="A,B", keys=keys,
        fail_arm=ARM_A, fail_case=keys[0])
    b_verdicts = [r for r in rows if r["kind"] == "verdict" and r["arm"] == ARM_B]
    assert [r["case_key"] for r in b_verdicts] == list(keys), (
        f"arm B wrote verdicts for {[r['case_key'] for r in b_verdicts]}; arm A's raise "
        "must not cancel that case's own arm B, which is the half of the comparison "
        "the study measures")
    assert [r for r in rows if r["kind"] == "verdict"
            and r["arm"] == ARM_A and r["case_key"] == keys[1]], (
        "the case after the failure did not record its arm A")
    assert not [r for r in rows if r["arm"] == ARM_RETRY], (
        "an arm that was not asked for wrote rows")
    assert f"FAILED {keys[0]} {ARM_A}" in out and "RuntimeError" in out
    assert code == 1


def test_a_truncated_arm_a_reports_its_turn_cap_in_the_pass_log_and_still_exits_one(
        tmp_path, monkeypatch, capsys):
    """#2984 clause 4's last hop: the mechanism survives the guard into `run.log`.

    This is the only place arm A's message is ever read — `cmd_run` prints
    `FAILED <key> <arm>: RuntimeError: <message>` and `launch-pass.sh` redirects that
    stdout to `2258/run.log`, which is the sole artefact of a ~23.6-engine-hour pass.
    So the clause is not satisfied by a better string inside the raise: the truncated
    dispatch has to be recognisable in the guard's own line, while the guard's
    behaviour stays as #2623 left it — the case's arm B still records its verdict,
    the failing arm records nothing, and the pass exits 1.
    """
    keys = ("flaky-SM_A_CAPPED", "flaky-SM_A_CAPPED_SURVIVOR")
    code, calls, rows, out = _drive_pass(
        tmp_path, monkeypatch, capsys, arms="A,B", keys=keys,
        fail_arm=ARM_A, fail_case=keys[0], fail_stop=("max_turns", 40))
    assert f"FAILED {keys[0]} {ARM_A}" in out, out
    assert "RuntimeError" in out, out
    assert "max_turns" in out and "40-turn budget" in out, (
        f"the pass log line carries no mechanism: {out!r}")
    assert "harness cut the run off" in out, out
    assert code == 1, f"exit {code}; a pass that lost an arm exits non-zero"
    assert not [r for r in rows if r["case_key"] == keys[0] and r["arm"] == ARM_A], (
        "the truncated arm A wrote rows; the refusal still records no verdict")
    assert [r for r in rows if r["kind"] == "verdict"
            and r["case_key"] == keys[0] and r["arm"] == ARM_B], (
        "arm A's truncation cancelled that same case's arm B")


def test_a_pass_where_nothing_raises_still_exits_zero(tmp_path, monkeypatch, capsys):
    """The other side of clause 5's exit code: containing a failure may not invent one.

    #2623 clause 4 is a single launch of a ~23.6-engine-hour pass whose only report
    is `PASS_EXIT=$?` in `2258/run.log`. A guard that exited non-zero on a complete
    run would spend that night, report a failure, and read like the bug it replaced.
    """
    keys = ("flaky-SM_ONE", "flaky-SM_TWO")
    code, calls, rows, out = _drive_pass(tmp_path, monkeypatch, capsys,
                                         arms="A,A-retry,B", keys=keys)
    assert code == 0, "a pass in which nothing raised must exit 0"
    verdicts = [(r["arm"], r["case_key"]) for r in rows if r["kind"] == "verdict"]
    assert sorted(verdicts) == sorted((a, k) for a in (ARM_A, ARM_RETRY, ARM_B)
                                      for k in keys), (
        f"the pass recorded {len(verdicts)} verdicts, want {len(keys)} x 3 arms")
    assert "arm-run(s) failed" not in out
    assert out.count("B chose:") == len(keys), (
        "the per-case arm-B line is printed once per case, not once per arm")
