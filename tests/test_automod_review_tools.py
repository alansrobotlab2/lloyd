"""Calibrating the grader without a network: the fixture shape, the compare
rule, the stripped-tests variant, and that a backfill writes to its own file
and never to the ledger."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.automod import backlog as B, review as RV, review_tools as RT, state as S


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A repo with a landed commit, a backlog item with clauses, and a ledger
    that says the round settled."""
    repo = tmp_path / "repo"; (repo / "app").mkdir(parents=True); (repo / "tests").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(repo))
    git(repo, "config", "user.email", "t@e"); git(repo, "config", "user.name", "t")
    (repo / "app" / "m.py").write_text("V = 1\n"); git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "base")
    parent = git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "app" / "m.py").write_text("V = 2\n")
    (repo / "tests" / "test_m.py").write_text("def test_v():\n    assert True or True\n")
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "round")
    commit = git(repo, "rev-parse", "HEAD").stdout.strip()

    backlog = tmp_path / "backlog"; backlog.mkdir()
    (backlog / "9-thing.md").write_text(
        "---\nstatus: in_progress\nboard: lloyd\nacceptance_clauses:\n- V is 2\n- a test pins it\n---\n# Thing\n\nbody\n")
    monkeypatch.setattr(B, "BACKLOG_DIR", backlog)
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(S, "LEDGER_PATH", ledger)
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    for e in ({"event": "backlog_implement", "item_id": 9, "phase": "finished", "round_id": "SM_9",
               "outcome": {"acceptance": "met", "clause_outcomes": [
                   {"clause": 1, "outcome": "met"}, {"clause": 2, "outcome": "met"}]}},
              {"event": "promoted", "round_id": "SM_9", "commit": commit, "parent": parent,
               "changed_paths": ["app/m.py", "tests/test_m.py"]},
              {"event": "settled", "commit": commit}):
        S.append_event(e, path=ledger)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return {"repo": repo, "parent": parent, "commit": commit, "ledger": ledger, "tmp": tmp_path}


def _stub(structured):
    def grade(**kw):
        # Observed at call time: the detached worktree is released before
        # grade_commit returns, so existence can only be checked from inside.
        kw["_tree_present"] = (Path(kw["worktree"]) / "app" / "m.py").exists()
        grade.calls.append(kw)
        return {"ok": True, "error": "", "session_id": "sess_cal", "structured": structured}
    grade.calls = []
    return grade


def _met(path="app/m.py", node="tests/test_m.py::test_v"):
    return {"premise": "sound", "summary": "fine", "test_honesty": [], "seams_unverified": [],
            "clauses": [{"clause": i, "verdict": "met", "evidence_path": path, "evidence_line": 1,
                         "test_node_id": node, "how_verified": "ran", "note": ""} for i in (1, 2)]}


def test_landed_rounds_joins_promotion_settle_and_item(world):
    rows = RT.landed_rounds(world["ledger"])
    assert len(rows) == 1 and rows[0]["item_id"] == 9 and rows[0]["commit"] == world["commit"]
    assert rows[0]["author_outcome"]["acceptance"] == "met"


def test_grade_commit_checks_out_the_commit_and_runs_the_prechecks(world):
    grade = _stub(_met())
    out = RT.grade_commit(repo=world["repo"], item_id=9, parent=world["parent"], commit=world["commit"],
                          changed_paths=["app/m.py", "tests/test_m.py"], label="t1", grader=grade)
    kw = grade.calls[0]
    assert Path(kw["worktree"]).name == "lloyd" and kw["_tree_present"]
    assert kw["base"] == world["parent"] and kw["contract"]["clauses"] == ["V is 2", "a test pins it"]
    assert kw["child_env"]["LLOYD_VOICE_ALERTS"] == "0" and "gate-state" in kw["child_env"]["LLOYD_AUTOMOD_STATE"]
    # The stubbed grader said met everywhere, but the deterministic precheck
    # saw `or True` in the added test — and that is a retry regardless.
    assert out["kind"] == "retry" and any("or True" in p["problem"] for p in out["prechecks"])
    assert not Path(kw["worktree"]).exists(), "the detached worktree is released"


def test_strip_tests_turns_a_landing_into_a_known_bad_case(world):
    grade = _stub(_met())
    out = RT.grade_commit(repo=world["repo"], item_id=9, parent=world["parent"], commit=world["commit"],
                          changed_paths=["app/m.py", "tests/test_m.py"], label="t2", grader=grade,
                          strip_tests=True)
    kw = grade.calls[0]
    assert kw["changed_paths"] == ["app/m.py"], "the test file is gone from the diff under review"
    # Every `met` is downgraded: its test_node_id is not in a changed test file.
    assert out["kind"] == "retry" and all(c["verdict"] == "partial" for c in out["clauses"])


def test_fixture_compare_and_calibrate(world, tmp_path):
    fx = tmp_path / "fixtures"
    p = RT.write_fixture(name="nine", round_id="SM_9", expect_kind="retry", expect_unmet=[2],
                         expect_honesty=True, ledger=world["ledger"], fixture_dir=fx, note="n")
    case = json.loads(p.read_text())
    assert case["commit"] == world["commit"] and case["expect"] == {
        "kind": "retry", "unmet_clauses": [2], "honesty_finding": True}
    with pytest.raises(ValueError):
        RT.write_fixture(name="x", round_id="SM_NOPE", expect_kind="pass", ledger=world["ledger"], fixture_dir=fx)

    # A grader that says everything is met disagrees with the case on the clause,
    # even though the precheck supplies the honesty finding and the retry.
    rows = RT.calibrate(repo=world["repo"], fixture_dir=fx, grader=_stub(_met()))
    assert rows[0]["ok"] is False and "clauses [2]" in rows[0]["why"]
    # One that flags clause 2 agrees.
    obj = _met(); obj["clauses"][1]["verdict"] = "unmet"
    rows = RT.calibrate(repo=world["repo"], fixture_dir=fx, grader=_stub(obj))
    assert rows[0]["ok"] is True and rows[0]["why"] == "agrees"
    ok, why = RT.compare(case, {"error": "HTTP 503"})
    assert ok is False and "did not answer" in why


def test_backfill_writes_its_own_file_and_never_the_ledger(world):
    before = world["ledger"].read_text()
    out_path = world["tmp"] / "state" / "review_backfill.jsonl"
    rows = RT.backfill(repo=world["repo"], ledger=world["ledger"], grader=_stub(_met()), out=out_path)
    assert len(rows) == 1 and rows[0]["event"] == "review_backfill"
    assert rows[0]["author_met"] == 2 and rows[0]["grader_met"] == 2 and rows[0]["kind"] == "retry"
    assert out_path.exists() and json.loads(out_path.read_text().splitlines()[-1])["item_id"] == 9
    assert world["ledger"].read_text() == before, "a backfill is a measurement, not a verdict"
    # Idempotent: an already-graded round is skipped on the next run.
    assert RT.backfill(repo=world["repo"], ledger=world["ledger"], grader=_stub(_met()), out=out_path) == []


def test_the_shipped_fixture_for_544_expects_the_human_review():
    """The #544 case still expects the human review's verdict on the two findings that
    decide it: `retry`, and a test-honesty finding.

    It no longer expects clause 4 unmet. #3510 read the charge behind that expectation —
    'classification from `IDEMPOTENT`, skipped' — against the commit this case grades,
    and `agent_mcp/_tool_effects.py:80` imports the annotations module while `:199`
    returns `tool_annotations.side_effecting(name)`: the classification source the clause
    asked for is the one the landing used. The 2026-10-27 calibrate run said the same
    thing live — `[FAIL] 544-first-cut: clauses [4] expected unmet/partial, grader said
    met` — and the case note carries the refutation.
    """
    case = json.loads((RT.FIXTURE_DIR / "544-first-cut.json").read_text())
    assert case["commit"].startswith("0f019f90") and case["expect"]["kind"] == "retry"
    assert case["expect"]["unmet_clauses"] == [] and case["expect"]["honesty_finding"] is True, (
        "the expected clause list is the 2026-09-10 review's charge, which #3510 records "
        "as refuted by the code at 0f019f90; the test-honesty finding is not refuted")
    stripped = json.loads((RT.FIXTURE_DIR / "544-stripped-tests.json").read_text())
    assert stripped["strip_tests"] is True and stripped["expect"]["kind"] == "retry"
    names = {c["name"] for c in RT.load_fixtures()}
    assert {"544-first-cut", "544-stripped-tests"} <= names


#: The four shipped cases and the range each one models, as data. Derived from
#: nothing: these values are what the calibration was built on, and #2822 is what
#: happens when they are re-read wrong — its ruling asked to narrow
#: `447-small-met` to one path after a `git show` of a two-commit range.
_544_PATHS = ["agent_mcp/_tool_effects.py", "agent_mcp/annotations.py", "agent_mcp/main.py",
              "app/harness/loop.py", "app/harness/mcp_pool.py", "app/harness/policy.py",
              "app/routers/dashboard.py", "tests/test_tool_effects.py", "workers/pool.py"]
SHIPPED_RANGES = {
    "392-small-met": ("ad64d312", "ce713bf3", ["tests/test_trajectory_extraction.py"]),
    "447-small-met": ("fe40af38", "01219641", ["scripts/verify_cited_identifier.py",
                                               "tests/test_cited_identifier_verification.py"]),
    "544-first-cut": ("af1e8c1a", "0f019f90", _544_PATHS),
    "544-stripped-tests": ("af1e8c1a", "0f019f90", _544_PATHS),
}


def _range_paths(repo, parent, commit):
    r = git(repo, "diff", "--name-only", parent, commit)
    assert r.returncode == 0, f"`git diff --name-only {parent} {commit}` failed: {r.stderr[:200]}"
    return sorted(r.stdout.split())


def test_every_shipped_case_declares_the_paths_of_the_range_the_grader_grades():
    """#2822: a case's `changed_paths` must be the paths of `parent..commit`, because
    that range is what the grader is handed.

    `grade_commit` takes both ends, hands `parent` to the honesty prechecks and
    `base=parent` to the grader, and `calibrate` feeds it `case["parent"]` — so the
    graded surface is the range and never the tip commit alone. Nothing pinned that
    before this: `git grep -n changed_paths -- tests/` found only stub-fixture uses,
    and `test_the_shipped_fixture_for_544_expects_the_human_review` reads the
    `expect` block. That gap is how a single-commit reading of a two-commit case
    became an item asking to delete a path from the input to `pick_test_files`,
    `paths` and `def_test_delta`.

    This is also the check the ruling's own verdict turned on: it passes on all four
    fixtures as they stand, which is the evidence that #2639's half about `447`'s
    paths was wrong rather than the fixture being stale.
    """
    cases = RT.load_fixtures()
    assert {c["name"] for c in cases} == set(SHIPPED_RANGES), (
        "the shipped calibration set is no longer the four cases this test describes — a "
        "case whose JSON stopped parsing, or whose `expect.kind` is unknown, is silently "
        "skipped by load_fixtures, which would make the loop below grade nothing")
    for case in cases:
        got = _range_paths(RT.LIVE_ROOT, case["parent"], case["commit"])
        assert sorted(case["changed_paths"]) == got, (
            f"{case['name']} declares {sorted(case['changed_paths'])} but the range it "
            f"grades, {case['parent'][:8]}..{case['commit'][:8]}, changes {got}. That list "
            "is what the grader is told the diff touched, so a path outside the range is a "
            "citation the grader cannot find and a test file it never picks.")


def test_no_shipped_case_loses_a_path_to_a_single_commit_reading():
    """#2822 clause 3: the ranges and their path lists are pinned as values, not
    recomputed from the fixtures, so no later edit can quietly narrow one.

    `447-small-met` is the case that catches the wrong read, and the last assertion is
    the positive control for why: `git show --name-only 01219641` really does print one
    path — it is the tip of a two-commit range (`bd033a19` + `01219641`, the round's two
    cherry-picks onto base `fe40af38`) — so the claim was reproducible and still wrong. A
    fixture narrowed to that one path would keep every other check green while being told
    that `scripts/verify_cited_identifier.py` changed in a diff that does not contain it.
    """
    for name, (parent, commit, paths) in SHIPPED_RANGES.items():
        case = json.loads((RT.FIXTURE_DIR / f"{name}.json").read_text())
        assert case["parent"].startswith(parent) and case["commit"].startswith(commit), (
            f"{name}'s range moved to {case['parent'][:8]}..{case['commit'][:8]}; the case "
            "is now a different landing and its `expect` block is a claim about the old one")
        assert sorted(case["changed_paths"]) == sorted(paths), (
            f"{name}'s changed_paths no longer match what this calibration was built on")
    tip = sorted(git(RT.LIVE_ROOT, "show", "--name-only", "--format=",
                     "01219641584c35c2b0c2cfb7f117c76f851672d1").stdout.split())
    assert tip == ["tests/test_cited_identifier_verification.py"], (
        "the tip commit of 447's range no longer changes only the test file, so the "
        "single-commit reading this case was nearly edited into is not even "
        "self-consistent any more — re-derive the range before believing either number")


def test_the_two_diff_read_cases_are_validated_on_a_named_ruling_not_on_a_flag():
    """#2822 clause 1: the two cases that rest on the 2026-10-16 diff read each say
    which ruling confirmed their expectation and when, instead of disclaiming itself.

    (This node was `test_the_two_pass_cases_are_validated_on_a_named_ruling_not_on_a_flag`
    until #3510 re-marked 392 to the `retry` the live gradings answer: one of the two is
    still a `pass` case and the other is not, so the name said `pass` about a case whose
    `expect` no longer does.)

    Dropping the key is behaviour-neutral by design — `calibrate` reads it with
    `case.get("provisional", False)`, and the only other reader appends a
    ` (provisional)` suffix to its line — so what has to be pinned is the
    provenance: a case marked validated has to name the read that validated it, or the
    four-case set is four assertions nobody checked. The ruling named here is #2639's
    owed-check entry #3 of 2026-10-16T10:33:07Z, whose second half ("correcting 447's
    changed_paths to the one path the commit contains") is refuted by the test above
    and was not applied.

    **#3179 part (b) reversed half of that, and this node holds both rulings.** The
    drop applied to 447 stands: it graded PASS in the 2026-10-22 merged-ledger run too
    (`~/lloyd-data/logs/calibrate-2822-merged-213855.log`), so its key stays absent.
    392 is provisional again, because that same run answered `kind retry != pass`
    against it (grader session `20261022_213857_review_2794`) — a live disagreement
    with the diff read this note rests on, which is exactly what the flag records.
    Both cases still name the 2026-10-16 ruling: a re-mark documents the new
    disagreement, it does not delete the provenance of the expectation.

    **#3510 is the third ruling on the same two cases, and it moves 392's `expect`, not
    its provenance.** 447 is still `pass`; 392 is now `retry`, which is what both live
    gradings of `parent..commit` answered, and it stays `provisional` precisely because
    an expectation set to the grader's own answer is agreement by construction rather
    than a confirmed read. What this node holds across all three rulings is the part
    none of them touched: the note has to name the ruling behind its `expect`, so a
    reader can tell a re-mark from a silent edit.
    """
    for name, kind in (("392-small-met", "retry"), ("447-small-met", "pass")):
        case = json.loads((RT.FIXTURE_DIR / f"{name}.json").read_text())
        assert case["expect"]["kind"] == kind, (
            f"{name}'s expected kind moved: 447 has been `pass` since the 2026-10-16 "
            "diff read and graded PASS on 2026-10-22 too, and 392 is `retry` because "
            "#3510 re-marked it to what both live gradings answered")
        note = case["note"]
        assert "a human has not yet confirmed" not in note, f"{name}'s note still disclaims itself"
        assert "20261016_031621_owedcheck_e9d6" in note and "2026-10-16" in note, (
            f"{name}'s note does not name the confirming ruling and its date")
        assert "CONFIRM both expectations as pass and drop provisional:true" in note, (
            f"{name}'s note does not quote the ruling it rests on")
        if name == "447-small-met":
            assert "provisional" not in case, (
                "447's key came back, but it graded PASS on 2026-10-22 as well — #2822's "
                "drop stands for it: the disclaimer goes from the case, not set to false")
            continue
        assert case.get("provisional") is True, (
            "392 is un-flagged while its `expect` is what the 2026-10-22 and 2026-10-27 "
            "runs answered — `compare()` never reads the flag, so that agreement is by "
            "construction and only this key stops it reading as a confirmed read")
        assert "2026-10-22" in note and "20261022_213857_review_2794" in note, (
            "392's note does not name the run and grader session it now disagrees with")


def test_the_calibration_readme_hedges_metric_2_and_still_documents_the_flag():
    """#2822 clause 4: each table row says what validated its expectation, the file
    refuses to lend four agreements to scorecard metric 2, and the field
    `fixture --provisional` writes is still documented.

    Metric 2 is the audit delta, computed in `scripts/automod/scorecard.py` over
    `review` events from real landed rounds — never over these fixtures — so the caveat
    is about what a reader may conclude, not about a bug. The last two assertions are
    why the paragraph stays: the minting route still takes `--provisional`, so a
    freshly-minted case can still arrive unconfirmed, and a reader has to know what the
    field means when one does.

    The `expected` column, by contrast, was unpinned by #2822 and #3510 found it
    diverged: it still promised `pass` for 392 and a clause-4 miss for #544 while the
    case files said `retry` and no unmet clause. Nothing here checked the column, so it
    drifted silently — which is what the two new assertions at the end are for. A row
    whose provenance phrase survives while its expectation rots is the exact failure
    this table can have.
    """
    readme = (RT.FIXTURE_DIR / "README.md").read_text()
    assert "until a human confirms" not in readme, (
        "a shipped row still disclaims its expectation while the case file no longer does")
    for name in ("392-small-met", "447-small-met"):
        rows = [ln for ln in readme.splitlines() if ln.startswith(f"| `{name}`")]
        assert len(rows) == 1 and "confirmed by a diff read on 2026-10-16" in rows[0], (
            f"the `{name}` row is missing or no longer says what validated it")
    table = {n: [ln for ln in readme.splitlines() if ln.startswith(f"| `{n}`")]
             for n in ("392-small-met", "544-first-cut")}
    for name, rows in table.items():
        assert len(rows) == 1, f"the `{name}` row is duplicated or gone from the table"
    assert "`retry`" in table["392-small-met"][0], (
        "392's expected column still says `pass`, the expectation #3510 re-marked — the "
        "table is what a reader who never opens the JSON calibrates against")
    assert "clause 4 not met" not in table["544-first-cut"][0], (
        "544's expected column still promises a clause-4 miss the case no longer "
        "expects, on the charge its note records as refuted by the code at 0f019f90")
    assert "do not license scorecard metric" in readme and "audit delta" in readme, (
        "the file no longer says that four agreements do not make a headline rate")
    assert "`provisional: true` marks an expectation nobody has confirmed" in readme, (
        "the field's documentation was dropped along with the two rows' disclaimers")
    import inspect
    assert '"--provisional"' in inspect.getsource(RT.main), (
        "fixture --provisional no longer exists, so the paragraph this test protects and "
        "the CLI have come apart")


def test_a_harness_only_test_change_is_a_changed_test_to_the_backfill(world):
    """#1322's fourth site: the after-the-fact route built its changed-test set
    from root `tests/` too, so a landing whose test lives in app/harness/tests/
    had every `met` downgraded and `strip_tests` stripped nothing."""
    repo = world["repo"]
    (repo / "pytest.ini").write_text("[pytest]\ntestpaths = tests app/harness/tests scripts\n")
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "ini")
    parent = git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "app" / "m.py").write_text("V = 3\n")
    h = repo / "app" / "harness" / "tests"; h.mkdir(parents=True)
    (h / "test_m.py").write_text("def test_v():\n    assert 3 == 3\n")
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "harness round")
    commit = git(repo, "rev-parse", "HEAD").stdout.strip()
    paths = ["app/m.py", "app/harness/tests/test_m.py"]
    node = "app/harness/tests/test_m.py::test_v"

    grade = _stub(_met(node=node))
    out = RT.grade_commit(repo=repo, item_id=9, parent=parent, commit=commit,
                          changed_paths=paths, label="h1", grader=grade)
    assert out["kind"] != "retry", out
    assert all(c["verdict"] == "met" and "downgraded" not in c for c in out["clauses"])

    grade = _stub(_met(node=node))
    RT.grade_commit(repo=repo, item_id=9, parent=parent, commit=commit,
                    changed_paths=paths, label="h2", grader=grade, strip_tests=True)
    assert grade.calls[0]["changed_paths"] == ["app/m.py"], "the harness test is stripped"


def test_the_backfill_grants_the_same_standing_and_no_more(world, tmp_path):
    """The after-the-fact route must measure the rung that lands code (#1755, clause 3).

    `world`'s round adds `assert True or True`, which the deterministic precheck
    calls `blocking` and `decide` refuses on. Here the same quoted pattern is
    committed a second time, in a landing that also edits the checker itself: the
    LIVE module that scored it is then the version that landing replaced, so the
    finding survives as `advisory` and the scorecard records a pass. Without the
    rule the backfill would grade history against a verdict that round could never
    have got, and the gate and the backfill would disagree on the same diff.
    """
    repo = world["repo"]
    (repo / "scripts" / "automod").mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "automod" / "review.py").write_text("PATTERN = 2\n", encoding="utf-8")
    # The pattern has to be in THIS commit's added lines: the parent is the round
    # above, so an unchanged test file would add nothing for the precheck to see.
    (repo / "tests" / "test_m.py").write_text(
        "def test_v():\n    assert True or True\n\n"
        "def test_v_again():\n    assert True or True\n", encoding="utf-8")
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "round that edits the checker")
    editing = git(repo, "rev-parse", "HEAD").stdout.strip()
    paths = ["scripts/automod/review.py", "app/m.py", "tests/test_m.py"]

    out = RT.grade_commit(repo=repo, item_id=9, parent=world["commit"], commit=editing,
                          changed_paths=paths, label="stale", grader=_stub(_met()))
    hits = [p for p in out["prechecks"] if p.get("demoted_from") == "blocking"]
    assert hits and {p["severity"] for p in hits} == {"advisory"}, out["prechecks"]
    assert out["kind"] == "pass", f"a demoted finding cannot decide it alone: {out}"
    assert "scripts/automod/review.py" in out["honesty_note"], out["honesty_note"]
    assert RV.stale_honesty_modules(paths) == ["scripts/automod/review.py"], \
        "the two callers turn on one shared module list" 

    # The identical pattern, in the landing that left the checker alone: still
    # blocking, still a retry, and nothing to explain.
    keep = RT.grade_commit(repo=world["repo"], item_id=9, parent=world["parent"],
                           commit=world["commit"], changed_paths=["app/m.py", "tests/test_m.py"],
                           label="clean", grader=_stub(_met()))
    assert all(p["severity"] == "blocking" for p in keep["prechecks"] if "or True" in p["problem"]), keep["prechecks"]
    assert all("demoted_from" not in p for p in keep["prechecks"]), keep["prechecks"]
    assert keep["kind"] == "retry" and keep["honesty_note"] == "", keep["kind"]


# Built at runtime so this diff carries no sha-shaped token: the rail that grades
# a note naming one is the unfixed one until #2494 lands.
_PHANTOM_SHA = "0f" * 6


def _cited():
    """The #2492 shape: every clause `met`, cited correctly, and a note quoting a
    token no repo resolves."""
    obj = _met()
    for c in obj["clauses"]:
        c["note"] = f"the pin is the token {_PHANTOM_SHA}, which no repo resolves"
    return obj


def test_the_scorecard_says_when_the_sha_rail_stood_down(world):
    """A backfilled row must not grade history by a stricter engine than
    production runs (#2494, beside #1755's same rule for the prechecks).

    Both landings here cite the identical phantom token. The one whose recorded
    diff edits `scripts/automod/review.py` is graded by the module that landing
    replaced, so its citation finding is advisory, the scorecard records a pass,
    and the row carries the standing sentence naming the file — the field the
    scorecard report reads. The other keeps its non-pass and its silence: with
    nothing demoted there is nothing to explain.
    """
    repo = world["repo"]
    (repo / "scripts" / "automod").mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "automod" / "review.py").write_text("PATTERN = 1\n", encoding="utf-8")
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "round that edits the checker")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()

    out = RT.grade_commit(repo=repo, item_id=9, parent=world["commit"], commit=head,
                          changed_paths=["scripts/automod/review.py", "app/m.py", "tests/test_m.py"],
                          label="cite-stale", grader=_stub(_cited()))
    assert out["kind"] == "pass", f"a citation the round's rail cannot resolve cannot void it: {out}"
    assert "sha citation rail stale-by-construction" in out["honesty_note"], out["honesty_note"]
    assert "scripts/automod/review.py" in out["honesty_note"], out["honesty_note"]
    assert "2 clause(s)" in out["honesty_note"], out["honesty_note"]
    assert all(c.get("citation_advisory") and not c.get("citation_unresolved")
               for c in out["clauses"]), out["clauses"]

    keep = RT.grade_commit(repo=repo, item_id=9, parent=world["commit"], commit=head,
                           changed_paths=["app/m.py", "tests/test_m.py"],
                           label="cite-plain", grader=_stub(_cited()))
    assert _PHANTOM_SHA in " ".join(keep["unreliable"]), \
        "a landing that leaves the rail alone is still voided by it"
    assert keep["citation_note"] == "" and not any(
        c.get("citation_advisory") for c in keep["clauses"]), "nothing was demoted"
    assert "sha citation rail" not in keep["honesty_note"], keep["honesty_note"]


# ── replaying the #1903 confirmation pass over recorded blocks ────────────

def _review_row(round_id, *, kind="retry", blocking=True, findings, head, item_id=9, ts=1.0):
    return {"event": "review", "round_id": round_id, "item_id": item_id, "ok": True,
            "kind": kind, "blocking": blocking, "attempt": 1, "ts": ts,
            "head": head, "findings": findings, "clauses": [], "test_honesty": [],
            "seams_unverified": [], "downgraded": [], "summary": "", "amendments_ok": True}


def _append(world, *rows):
    for r in rows:
        S.append_event(r, path=world["ledger"])


def _round_start(world, round_id, base=None):
    """The row `round.start` writes: the round's own base, which is what a replay
    diffs from (#2017). `world["commit"]` has landed, so `merge-base main <head>` is
    the head itself — the base has to come from here or the diff is empty."""
    return {"event": "round_start", "round_id": round_id,
            "base": world["parent"] if base is None else base}


_TWO_CLAUSE_BLOCKS = ("clause 1 unmet: CONFIRM_ONE_SENTINEL no second reader is wired; "
                      "clause 2 partial (downgraded: not graded): the pin is not in this diff")


def test_replay_confirm_counts_the_blocks_a_second_reader_would_overturn(world):
    """The number the config switch is decided on, out of recorded events only.

    One block made of two synthesized clause entries is askable; a block that is
    only a Python-computed test-honesty finding is not, and says so; a pass and
    a non-review row are not blocks at all. The injected reader retires
    everything it is shown, so exactly one recorded block would have become a
    pass — and nothing on disk changed to say so.
    """
    _append(world,
            _round_start(world, "SM_A"),
            _review_row("SM_A", findings=_TWO_CLAUSE_BLOCKS, head=world["commit"]),
            _review_row("SM_B", findings="test honesty tests/test_m.py:2: `or True`",
                        head=world["commit"]),
            _review_row("SM_C", kind="pass", blocking=False, findings="met 2 of 2",
                        head=world["commit"]),
            {"event": "promoted", "round_id": "SM_A", "commit": world["commit"]})
    before = world["ledger"].read_text()
    # No append of any kind is allowed, so the guard is the loudest thing here.
    def _no_append(*a, **k):
        raise AssertionError("replay-confirm must not write to the ledger")
    monkeypatched = S.append_event
    S.append_event = _no_append
    try:
        res = RT.replay_confirm(ledger=world["ledger"], repo=world["repo"],
                                reader=lambda t: {"retire": True, "reason": "the pin is there"})
    finally:
        S.append_event = monkeypatched

    assert res["blocking"] == 2 and res["askable"] == 1
    assert res["overturned"] == 1 and res["judged"] == 1
    assert res["not_ask_reasons"] == {"all_python_computed": 1}
    row = next(r for r in res["rows"] if r["round_id"] == "SM_A")
    assert [e["text"] for e in row["entries"]] == [
        "clause 1 unmet: CONFIRM_ONE_SENTINEL no second reader is wired",
        "clause 2 partial (downgraded: not graded): the pin is not in this diff"]
    assert row["outcome"] == RV.OVERTURNED and len(row["votes"]) == 2
    assert row["votes"][0]["kind"] == "clause" and row["votes"][0]["verdict"] == "retired"
    assert world["ledger"].read_text() == before, "a replay is a measurement, not a verdict"


def test_replay_confirm_without_a_reader_measures_the_population_not_a_verdict(world):
    """No model runs in CI, so the overturn count is UNMEASURED — not zero.

    Reporting 0 here would read as "the second reader would have overturned
    nothing", which is the exact sentence the owed check on #1903 exists to
    produce, and would settle it with an arm that judged nothing.
    """
    _append(world, _round_start(world, "SM_A"),
            _review_row("SM_A", findings=_TWO_CLAUSE_BLOCKS, head=world["commit"]))
    res = RT.replay_confirm(ledger=world["ledger"], repo=world["repo"])
    assert res["blocking"] == 1 and res["askable"] == 1 and res["judged"] == 0
    assert res["replayable"] == 1 and res["not_ask_reasons"] == {}
    assert res["overturned"] is None and res["upheld"] is None
    row = res["rows"][0]
    assert row["ask"] is True and row["reason"] == "ask"
    assert "outcome" not in row
    assert [e["kind"] for e in row["entries"]] == ["clause", "clause"]


def test_replay_confirm_says_which_recorded_block_it_cannot_recover(world):
    """`findings` is stored truncated at 2000 characters, so an entry set that
    ran past the cut is reported rather than replayed as a shorter block — and a
    head whose objects are gone is reported as `diff_unrecoverable`."""
    _append(world,
            _review_row("SM_LONG", findings=("clause 1 unmet: n" + "; clause 2 unmet: n" * 700),
                        head=world["commit"]),
            _round_start(world, "SM_GONE"),
            _review_row("SM_GONE", findings="clause 1 unmet: gone", head="0f" * 20))
    res = RT.replay_confirm(ledger=world["ledger"], repo=world["repo"],
                            reader=lambda t: {"retire": True, "reason": "r"})
    assert res["blocking"] == 2 and res["askable"] == 1
    # Nothing could be judged, so the count is unmeasured rather than zero.
    assert res["judged"] == 0 and res["overturned"] is None
    by = {r["round_id"]: r for r in res["rows"]}
    assert by["SM_LONG"]["skipped"] == "stored_findings_truncated"
    assert by["SM_GONE"]["skipped"] == "diff_unrecoverable"
    # Both are in the census (#2017): the unrecoverable diff used to be in no count.
    assert res["not_ask_reasons"] == {"stored_findings_truncated": 1,
                                      "diff_unrecoverable": 1}
    assert res["replayable"] == 0


def test_the_replay_confirm_cli_reports_it_and_writes_nothing(world, tmp_path, capsys,
                                                             monkeypatch):
    # The CLI reads diffs from the live root; since #2017 the no-reader arm probes
    # them too, so the fixture repo has to be the one it looks in.
    monkeypatch.setattr(RT, "LIVE_ROOT", world["repo"])
    # ts is epoch seconds on a ledger row, and `--since` is a date: a row dated
    # 1970 would be filtered out by a 2020 floor, so this one is dated 2026.
    _append(world, _round_start(world, "SM_A"),
            _review_row("SM_A", findings=_TWO_CLAUSE_BLOCKS, head=world["commit"],
                        ts=1777000000.0))
    before = world["ledger"].read_text()
    rc = RT.main(["replay-confirm", "--since", "2020-01-01"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 recorded block(s); 1 would be put to the second reader" in out, out
    assert "UNMEASURED" in out, "the default arm judges nothing and must say so"
    assert world["ledger"].read_text() == before


def test_the_known_true_calibrate_block_is_never_offered_to_the_reader(world, tmp_path):
    """The calibration case that exists to catch a released defect still refuses.

    `--strip-tests` turns a landed round into a known-bad case by deleting its
    test files; the decision rule must still block, and every blocking entry it
    writes for that case comes from an evidence rail that checked the tree — the
    grader is stubbed to call the diff fully met, so only Python can disagree.
    `confirm_plan` therefore answers `all_python_computed`: no second reader, in
    the gate or in a replay, can retire the one block the eval directory knows
    is true, because a reader is never asked about it.
    """
    fx = tmp_path / "fixtures"
    RT.write_fixture(name="stripped", round_id="SM_9", expect_kind="retry", expect_unmet=[],
                     expect_honesty=True, strip_tests=True, ledger=world["ledger"],
                     fixture_dir=fx, note="known-bad by construction")
    rows = RT.calibrate(repo=world["repo"], fixture_dir=fx, grader=_stub(_met()))
    assert rows and rows[0]["kind"] == "retry", rows
    entries = RV.blocking_entries_from_text(rows[0]["findings"])
    assert entries, rows[0]["findings"]
    assert all(e["kind"] in RV.PYTHON_COMPUTED_ENTRY_KINDS for e in entries), entries
    assert RV.confirm_plan("retry", entries, confirm_on=True)["reason"] == "all_python_computed"


# ── #2017: the replay shows the round's own diff, and counts what it cannot ──

def test_a_replay_diffs_from_the_recorded_base_and_names_every_unreplayable_row(world):
    """The diff is `round_start.base..head`, never `merge-base main <head>`.

    `world["commit"]` has landed, so its merge-base with main is itself and the
    old recovery returned an empty diff as FOUND — the reader was then told to
    assume the finding mistaken on the evidence of `(no diff)`. Five askable
    blocks: one replayable, and one for each way a diff can be missing or empty.
    Every one that cannot be shown is a census key, so judged plus the census is
    the blocking count.
    """
    shown: list[str] = []

    def factory(row, diff):
        shown.append(diff)
        return lambda entry: {"retire": False, "reason": "stands"}

    _append(world,
            _round_start(world, "SM_OK"),
            _review_row("SM_OK", findings=_TWO_CLAUSE_BLOCKS, head=world["commit"]),
            # Landed head, and a base equal to it: the merge-base shape, empty.
            _round_start(world, "SM_EMPTY", base=world["commit"]),
            _review_row("SM_EMPTY", findings=_TWO_CLAUSE_BLOCKS, head=world["commit"]),
            _review_row("SM_NOBASE", findings=_TWO_CLAUSE_BLOCKS, head=world["commit"]),
            _round_start(world, "SM_NOHEAD"),
            _review_row("SM_NOHEAD", findings=_TWO_CLAUSE_BLOCKS, head=""),
            _round_start(world, "SM_GONE"),
            _review_row("SM_GONE", findings=_TWO_CLAUSE_BLOCKS, head="0f" * 20),
            _review_row("SM_PY", findings="test honesty tests/test_m.py:2: `or True`",
                        head=world["commit"]))
    res = RT.replay_confirm(ledger=world["ledger"], repo=world["repo"], reader_factory=factory)
    by = {r["round_id"]: r for r in res["rows"]}
    assert by["SM_OK"]["outcome"] == RV.UPHELD and "skipped" not in by["SM_OK"]
    assert len(shown) == 1 and "app/m.py" in shown[0], "the reader saw the round's own diff"
    assert {rid: by[rid]["skipped"] for rid in ("SM_EMPTY", "SM_NOBASE", "SM_NOHEAD", "SM_GONE")} == {
        "SM_EMPTY": RT.DIFF_EMPTY, "SM_NOBASE": RT.DIFF_NO_BASE,
        "SM_NOHEAD": RT.DIFF_NO_HEAD, "SM_GONE": RT.DIFF_UNRECOVERABLE}
    assert all("outcome" not in by[rid] for rid in by if rid != "SM_OK")
    assert res["not_ask_reasons"] == {
        RT.DIFF_EMPTY: 1, RT.DIFF_NO_BASE: 1, RT.DIFF_NO_HEAD: 1, RT.DIFF_UNRECOVERABLE: 1,
        "all_python_computed": 1}
    assert res["blocking"] == 6 and res["askable"] == 5 and res["replayable"] == 1
    assert res["judged"] + sum(res["not_ask_reasons"].values()) == res["blocking"]

    # The no-reader arm probes the diffs too, so its headline is the same population.
    dry = RT.replay_confirm(ledger=world["ledger"], repo=world["repo"])
    assert dry["replayable"] == 1 and dry["judged"] == 0
    assert dry["not_ask_reasons"] == res["not_ask_reasons"]


def test_a_row_carrying_its_entry_list_is_replayed_past_the_findings_cap(world):
    """Rows written since #2017 carry `blocking_entries`, so a block whose joined
    text ran past the 2000-character cap is no longer `stored_findings_truncated`:
    the list is replayed as stored, entry for entry, and it round-trips through
    the same splitter the old rows need."""
    texts = [f"clause {i} unmet: " + "x" * 400 for i in range(1, 8)]
    joined = "; ".join(texts)
    assert len(joined) > RT.STORED_FINDINGS_CAP
    entries = [{"text": t, "kind": RV.blocking_entry_kind(t)} for t in texts]
    assert RV.blocking_entries_from_text(joined) == entries
    capped = _review_row("SM_OLD", findings=joined[:RT.STORED_FINDINGS_CAP], head=world["commit"])
    listed = {**_review_row("SM_NEW", findings=joined[:RT.STORED_FINDINGS_CAP],
                            head=world["commit"]), "blocking_entries": entries}
    _append(world, _round_start(world, "SM_OLD"), capped, _round_start(world, "SM_NEW"), listed)
    res = RT.replay_confirm(ledger=world["ledger"], repo=world["repo"],
                            reader=lambda t: {"retire": False, "reason": "stands"})
    by = {r["round_id"]: r for r in res["rows"]}
    assert by["SM_OLD"]["skipped"] == "stored_findings_truncated"
    assert "skipped" not in by["SM_NEW"] and by["SM_NEW"]["entries"] == entries
    assert by["SM_NEW"]["outcome"] == RV.UPHELD
    assert len(by["SM_NEW"]["votes"]) == RV.CONFIRM_MAX_ENTRIES


# The four calibration cases are only as good as the landing they name. #3179
# touched one of them — re-marking 392 `provisional` after the 2026-10-22
# disagreement — and the tool that rewrites a fixture (`write_fixture`) takes
# parent, commit and changed_paths as arguments, so a hand edit that moves one of
# them silently re-points a case at a different landing while its `expect` stays.
# Nothing else in this file would notice, so this node holds all four still.
_FIXTURE_IDENTITIES = {
    "392-small-met": (
        "ad64d31282279f93f91be895bf62c6a31fd85bf4",
        "ce713bf39cf81d2d00b5ab1cbb0404209df4349e",
        ["tests/test_trajectory_extraction.py"]),
    "447-small-met": (
        "fe40af3816782ca09180598509c6c7a752e12d65",
        "01219641584c35c2b0c2cfb7f117c76f851672d1",
        ["scripts/verify_cited_identifier.py",
         "tests/test_cited_identifier_verification.py"]),
    "544-first-cut": (
        "af1e8c1ae802bc57cc681977885af2ff1a9d8e10",
        "0f019f902d045806afde30fffc336cdec32ef775",
        ["agent_mcp/_tool_effects.py", "agent_mcp/annotations.py", "agent_mcp/main.py",
         "app/harness/loop.py", "app/harness/mcp_pool.py", "app/harness/policy.py",
         "app/routers/dashboard.py", "tests/test_tool_effects.py", "workers/pool.py"]),
    "544-stripped-tests": (
        "af1e8c1ae802bc57cc681977885af2ff1a9d8e10",
        "0f019f902d045806afde30fffc336cdec32ef775",
        ["agent_mcp/_tool_effects.py", "agent_mcp/annotations.py", "agent_mcp/main.py",
         "app/harness/loop.py", "app/harness/mcp_pool.py", "app/harness/policy.py",
         "app/routers/dashboard.py", "tests/test_tool_effects.py", "workers/pool.py"]),
}


def test_the_remark_moved_one_provisional_flag_and_no_landing_identity(tmp_path):
    """#3179 clause 5: the 2026-10-22 run disagreed with 392, so the flag goes
    back — with the 2026-10-16 diff-read ruling left in the note as history, and
    nothing else in any fixture moved.

    `provisional` was dropped from 392 and 447 by `69095bb2` (#2822) on the
    2026-10-16 owed-check ruling, 'CONFIRM both expectations as pass and drop
    provisional:true'. On 2026-10-22 the merged-ledger calibrate run graded 392
    `kind retry != pass` ('clause 1 partial: The derivation half is genuinely in
    the tree and shipped before this diff', grader session
    `20261022_213857_review_2794`, line 1 of
    `~/lloyd-data/logs/calibrate-2822-merged-213855.log`, which measured `2/4
    agree`), so 392 is provisional again and its note names the disagreement. 447
    is NOT: it graded PASS in that same run, so its key stays absent. The two 544
    cases carry `"provisional": false` from `write_fixture`'s
    `bool(provisional)` and are counted as not-provisional.

    Pinned together with the identity of all four cases, because the flag edit and
    a re-pointed `parent`/`commit`/`changed_paths` would otherwise be one commit,
    and only this node reads the four files.
    """
    raw = {p.stem: json.loads(p.read_text(encoding="utf-8"))
           for p in sorted(RT.FIXTURE_DIR.glob("*.json"))}
    assert set(raw) == set(_FIXTURE_IDENTITIES), sorted(raw)

    assert raw["392-small-met"]["provisional"] is True, raw["392-small-met"].get("provisional")
    note = raw["392-small-met"]["note"]
    assert "2026-10-22" in note and "20261022_213857_review_2794" in note, note
    assert "2026-10-16" in note, "the diff-read ruling is history, not collateral"
    assert "provisional" not in raw["447-small-met"], "447 graded PASS; it gains no key"

    fx = tmp_path / "fixtures"; fx.mkdir()
    for stem, case in raw.items():
        (fx / f"{stem}.json").write_text(json.dumps(case), encoding="utf-8")
    cases = RT.load_fixtures(fx)
    flagged = sorted(c["name"] for c in cases if c.get("provisional"))
    assert flagged == ["392-small-met"], flagged

    for name, (parent, commit, changed) in _FIXTURE_IDENTITIES.items():
        case = raw[name]
        assert (case["parent"], case["commit"], case["changed_paths"]) == (parent, commit, changed), (
            f"{name}'s graded landing moved: this node pins parent, commit and "
            f"changed_paths so a flag edit cannot silently re-point a case at "
            f"another commit and keep its `expect`"
        )


# ── #3510: the two re-marks, each pinned to the bytes and the runs it cites ─────────
#
# Two calibrate runs disagreed with two fixtures and both disagreements were left in
# place: `~/lloyd-data/logs/calibrate-3179-postlanding-.log` (2026-10-27 15:28) prints
# `[FAIL] 392-small-met (provisional): kind retry != pass` and `[FAIL] 544-first-cut:
# clauses [4] expected unmet/partial, grader said met` over a footer of `2/4 agree`.
# Re-marking them to the graded reality is only honest if the notes carry the evidence
# for that reality, so this node reads the cited lines out of the commits they were
# written against rather than trusting prose, and drives `compare` with the answers the
# two runs recorded so the flip is visible in both directions.


def _blob_lines(commit: str, path: str) -> dict[int, str]:
    """`path` as it stands in `commit`, keyed by 1-based line number.

    Both notes cite line numbers, and for #392's the number belongs to the GRADED tree:
    `_live_buckets` is defined at line 964 of `tests/test_trajectory_extraction.py` at
    `ce713bf3`, and at line 3099 of the same file at `4307f270`, the commit this re-mark
    was cut from. A check that resolved the citation against whichever tree pytest
    happens to run in would therefore pass while proving nothing about the landing the
    case models, so every caller here names its commit explicitly.
    """
    r = git(RT.LIVE_ROOT, "show", f"{commit}:{path}")
    assert r.returncode == 0, f"`git show {commit}:{path}` failed: {r.stderr[:200]}"
    return dict(enumerate(r.stdout.splitlines(), start=1))


def _recorded_392_grading() -> dict:
    """What the grader answered for #392's landing, in the shape `compare` consumes.

    A minimal witness carrying the three facts the two live runs recorded — `kind:
    retry`, clause 1 short of met because the derivation shipped in `75ce62d` before the
    round's parent, and one test-honesty string about `_live_buckets` — not a transcript
    of a grader session. `compare` reads only these three fields, which is what makes a
    hand-built result like this one a fair input to it.
    """
    return {
        "kind": "retry",
        "clauses": [{"clause": 1, "verdict": "partial"}],
        "test_honesty": ["tests/test_trajectory_extraction.py:964: `_live_buckets()` "
                         "reads `_ROOT / _pipeline`, absent both in this worktree and "
                         "in the live checkout"],
    }


def _recorded_544_grading() -> dict:
    """What the 2026-10-27 run answered for #544's landing: `retry`, every clause met,
    and the `or True` assertion still reported as a test-honesty finding."""
    return {
        "kind": "retry",
        "clauses": [{"clause": i, "verdict": "met"} for i in (1, 2, 3, 4, 5)],
        "test_honesty": ["tests/test_tool_effects.py: `assert _rows(...) == [] or True`"],
    }


def test_the_remark_to_the_graded_reality_pins_both_cases_and_their_notes():
    """#3510: the two fixtures say what the grader says, and their notes carry the
    evidence for it — the cited code read out of the commits it names, the older ruling
    kept as history, and `compare` flipping on the re-mark alone.

    Clause by clause. (1) and (2): 392 expects `retry` with a honesty finding and stays
    `provisional: true`, because what rests against it is a live grading and not a
    confirmed read; its note names BOTH disagreements — the 2026-10-22 grader session
    `20261022_213857_review_2794` with its clause-1 reason, and the 2026-10-27 run's
    `:964` honesty finding — while still naming the 2026-10-16 diff read it rests on,
    which the flag records rather than deletes. (3) and (4): 544-first-cut drops
    `unmet_clauses: [4]` because the charge it rested on — that clause (d), classifying
    effects from `IDEMPOTENT`, was skipped — is refuted by the code in its own graded
    commit, and the note says so against the two lines that refute it. (5) is the README
    row, in the node that owns that file.

    The last block is the anti-tautology check. Setting `expect` to what the grader
    answered makes those two cases agree BY CONSTRUCTION — `compare` never reads
    `provisional`, so only the flag and the note stop a self-confirming fixture reading
    as validated — so what has to be pinned is that the FLIP is the whole change: the
    recorded answers still fail the old expectations, and the recorded FAIL strings are
    the ones the log printed.
    """
    raw = {p.stem: json.loads(p.read_text(encoding="utf-8"))
           for p in sorted(RT.FIXTURE_DIR.glob("*.json"))}

    c392 = raw["392-small-met"]
    assert c392["expect"]["kind"] == "retry", (
        "392 still expects `pass`, which every live grading of parent..commit has "
        "answered `retry` since 2026-10-22 — the fixture is the claim, not the record")
    assert c392["expect"]["honesty_finding"] is True, (
        "the grader reported a test-honesty finding on both runs; expecting none "
        "guarantees a FAIL line for a case that is meant to check the grader")
    assert c392.get("provisional") is True, (
        "an expectation set to what the grader said is agreement by construction, so it "
        "must stay flagged provisional or it reads as independently confirmed")

    n392 = c392["note"]
    for phrase, why in (
        ("20261022_213857_review_2794", "names no 2026-10-22 grader session"),
        ("clause 1 partial", "does not give that session's reason"),
        ("75ce62d", "does not name the commit the derivation shipped in"),
        ("before the parent", "does not say the derivation predates this round's parent"),
        ("2026-10-27", "names no 2026-10-27 run"),
        ("tests/test_trajectory_extraction.py:964", "does not cite the honesty finding"),
        ("_live_buckets", "does not name the helper that finding is about"),
        ("_ROOT / _pipeline", "does not say what that helper reads"),
        ("20261016_031621_owedcheck_e9d6", "deleted the older ruling instead of dating it"),
        ("CONFIRM both expectations as pass and drop provisional:true",
         "no longer quotes the ruling the expectation rests on"),
        ("not its resolution", "does not say the re-mark records the disagreement "
                               "rather than settling it"),
    ):
        assert phrase in n392, f"392's note {why}: {phrase!r}"
    assert "line 964 of the graded tree" in n392, (
        "392's note cites :964 without saying which tree the number is in — the same "
        "symbol sits at :3099 at today's HEAD, so an unqualified number sends a reader "
        "to the wrong file position")

    graded_392 = _blob_lines("ce713bf39cf81d2d00b5ab1cbb0404209df4349e",
                             "tests/test_trajectory_extraction.py")
    assert graded_392[964].startswith("def _live_buckets("), (
        f"the note's `:964` is not `_live_buckets` in the graded tree — line 964 there "
        f"reads {graded_392[964][:60]!r}")
    at_base = _blob_lines("4307f2708dd4ffb2301f77622e3e340f5153f4b0",
                          "tests/test_trajectory_extraction.py")
    assert at_base[3099].startswith("def _live_buckets("), (
        "the note also names line 3099 as where that helper sits at `4307f270`, which is "
        "how it says the two numbers belong to two trees — if that moved, the second "
        "number is a claim about a tree nobody is standing in")

    c544 = raw["544-first-cut"]
    assert c544["expect"]["unmet_clauses"] == [], (
        "clause 4 is still expected unmet on the strength of a charge the code at "
        "0f019f90 refutes")
    assert c544["expect"]["kind"] == "retry" and c544["expect"]["honesty_finding"] is True, (
        "#3510 re-marks clause 4 only; `retry` and the honesty finding are what the "
        "human review and every run since agree on")

    n544 = c544["note"]
    for phrase, why in (
        ("0f019f90", "names no commit for the refutation"),
        ("agent_mcp/_tool_effects.py:80", "does not cite the import that refutes it"),
        (":199", "does not cite the call that refutes it"),
        ("IDEMPOTENT", "does not name the classification source the clause asked for"),
        ("refuted", "does not say the 2026-09-10 'skipped' finding was refuted"),
    ):
        assert phrase in n544, f"544-first-cut's note {why}: {phrase!r}"
    effects = _blob_lines("0f019f902d045806afde30fffc336cdec32ef775", "agent_mcp/_tool_effects.py")
    assert "from agent_mcp import annotations" in effects[80], (
        f"line 80 of that commit reads {effects[80][:60]!r}, not the annotations import")
    assert "tool_annotations.side_effecting(name)" in effects[199], (
        f"line 199 of that commit reads {effects[199][:60]!r}, not the delegation")
    annotations = _blob_lines("0f019f902d045806afde30fffc336cdec32ef775", "agent_mcp/annotations.py")
    annot_text = "\n".join(annotations.values())
    assert "IDEMPOTENT" in annot_text and "def side_effecting" in annot_text, (
        "the module no longer holds both the constant the clause named and the "
        "classifier that reads it, which is the substance of the refutation")

    assert raw["447-small-met"]["expect"] == {"kind": "pass", "unmet_clauses": [],
                                             "honesty_finding": False}, (
        "447 graded PASS on 2026-10-22 too, so its expectation is not this item's to move")
    assert raw["544-stripped-tests"]["expect"] == {"kind": "retry", "unmet_clauses": [],
                                                  "honesty_finding": False}, (
        "the stripped-tests variant is a different case: it has never expected an "
        "unmet clause, and #3510 does not touch it")

    ok392, why392 = RT.compare(c392, _recorded_392_grading())
    assert ok392, f"the re-marked case does not agree with the recorded answer: {why392}"
    stale392 = {**c392, "expect": {**c392["expect"], "kind": "pass",
                                   "honesty_finding": False}}
    ok_stale, why_stale = RT.compare(stale392, _recorded_392_grading())
    assert not ok_stale and why_stale == "kind retry != pass", (
        f"the old expectation no longer produces the logged FAIL, so this node is no "
        f"longer measuring the thing that was wrong: {why_stale!r}")

    ok544, why544 = RT.compare(c544, _recorded_544_grading())
    assert ok544, f"the re-marked case does not agree with the recorded answer: {why544}"
    stale544 = {**c544, "expect": {**c544["expect"], "unmet_clauses": [4]}}
    ok_stale544, why_stale544 = RT.compare(stale544, _recorded_544_grading())
    assert not ok_stale544 and why_stale544 == (
        "clauses [4] expected unmet/partial, grader said met"), (
        f"the old expectation no longer produces the logged FAIL: {why_stale544!r}")
