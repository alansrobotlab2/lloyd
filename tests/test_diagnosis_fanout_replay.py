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
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.diagnosis_fanout_replay import (  # noqa: E402
    ARM_A, ARM_B, ARM_RETRY, MIN_CASES, RecordError, THEORIES_PER_CASE,
    VAULT_WITNESS, WITNESS_CASES, WITNESS_ROWS, WITNESS_SHA, baseline_problem,
    build_cases, case_complete, citation_rows, format_report, learnings_cases,
    load_cases, rank_theories, slot_labels, verify_citation, verify_citations,
    witness_census, write_baseline, write_cases)

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


def test_a_second_duplicate_theory_or_a_prose_conclusion_is_refused(tmp_path):
    """Fan-out with a duplicate theory is pure cost, and a summary is not evidence."""
    from eval.diagnosis_fanout_replay import RowWriter, run_case_b

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

    theories = ["one", "two", "three"]
    with pytest.raises(RecordError):
        asyncio.run(run_case_b(_cases(1)[0], generate=_fake_arm_b(theories, {})[0],
                               investigate=prose, rank_fn=_rank_stub([0, 1, 2]),
                               writer=RowWriter(tmp_path / "rows2.jsonl"), run_id="t3"))


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
